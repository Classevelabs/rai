//! Streaming reader for HuggingFace `.safetensors` checkpoints.
//!
//! The converter never needs a whole checkpoint in memory: it walks one tensor
//! at a time, and large tensors one row block at a time. The `safetensors`
//! crate deserializes from a single `&[u8]` covering the entire file, which
//! would put a 7B checkpoint (or an mmap of it) behind every read, so this
//! module parses the container directly instead:
//!
//! ```text
//!   [u64 LE header_len][header_len bytes of JSON][tensor data]
//! ```
//!
//! The JSON header maps tensor name -> `{dtype, shape, data_offsets}`, where
//! `data_offsets` are relative to the start of the tensor data. Rows of a 2-D
//! tensor are contiguous, so any row range is one `seek` + one `read`.

use anyhow::{bail, Context, Result};
use half::{bf16, f16};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Upper bound on the JSON header of a shard. Real headers are a few hundred
/// kilobytes; this only exists so a corrupt length cannot trigger a huge
/// allocation.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

/// The element types the converter can read. Everything is widened to f32.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    F16,
    BF16,
}

impl Dtype {
    fn parse(name: &str) -> Result<Self> {
        match name {
            "F32" => Ok(Dtype::F32),
            "F16" => Ok(Dtype::F16),
            "BF16" => Ok(Dtype::BF16),
            other => bail!(
                "unsupported safetensors dtype '{other}': the converter reads F32, F16 and BF16 \
                 weights (quantized or integer checkpoints are already compressed and cannot be \
                 re-quantized from here)"
            ),
        }
    }

    fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }
}

/// Where one tensor lives and what it looks like.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Index into [`SafeTensorsSet::files`].
    file: usize,
    /// Absolute byte offset of the first element inside that shard.
    start: u64,
    /// Byte length of the tensor payload.
    nbytes: u64,
}

impl TensorInfo {
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// `(rows, cols)` for a 2-D tensor.
    pub fn dims_2d(&self, name: &str) -> Result<(usize, usize)> {
        match self.shape.as_slice() {
            [rows, cols] => Ok((*rows, *cols)),
            other => bail!("tensor '{name}' has shape {other:?}; a 2-D matrix is required"),
        }
    }
}

/// One or more `.safetensors` shards opened as a single tensor namespace.
pub struct SafeTensorsSet {
    files: Vec<PathBuf>,
    handles: Vec<Option<File>>,
    tensors: HashMap<String, TensorInfo>,
    scratch: Vec<u8>,
}

impl SafeTensorsSet {
    /// Open a HuggingFace checkpoint directory: either a single
    /// `model.safetensors` or a `model.safetensors.index.json` shard set.
    pub fn open(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let single = dir.join("model.safetensors");

        let shards: Vec<PathBuf> = if index.is_file() {
            let raw =
                std::fs::read(&index).with_context(|| format!("reading {}", index.display()))?;
            let parsed: serde_json::Value = serde_json::from_slice(&raw)
                .with_context(|| format!("parsing {}", index.display()))?;
            let map = parsed
                .get("weight_map")
                .and_then(|v| v.as_object())
                .with_context(|| format!("{} has no weight_map object", index.display()))?;
            let mut names: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            names.sort();
            names.dedup();
            if names.is_empty() {
                bail!("{} lists no shards", index.display());
            }
            names.into_iter().map(|n| dir.join(n)).collect()
        } else if single.is_file() {
            vec![single]
        } else {
            bail!(
                "no safetensors checkpoint in {}: expected model.safetensors or \
                 model.safetensors.index.json",
                dir.display()
            );
        };

        let mut tensors: HashMap<String, TensorInfo> = HashMap::new();
        for (file_index, path) in shards.iter().enumerate() {
            read_shard_header(path, file_index, &mut tensors)
                .with_context(|| format!("reading {}", path.display()))?;
        }
        if tensors.is_empty() {
            bail!("checkpoint in {} contains no tensors", dir.display());
        }

        Ok(Self {
            handles: shards.iter().map(|_| None).collect(),
            files: shards,
            tensors,
            scratch: Vec::new(),
        })
    }

    /// Metadata for one tensor, or `None` when it is absent.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Metadata for one tensor, naming it in the error when it is missing.
    pub fn require(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("checkpoint is missing the required tensor '{name}'"))
    }

    /// True when any tensor name satisfies `predicate`.
    pub fn any_name<F: Fn(&str) -> bool>(&self, predicate: F) -> Option<&str> {
        let mut hits: Vec<&str> = self
            .tensors
            .keys()
            .map(String::as_str)
            .filter(|n| predicate(n))
            .collect();
        // The map iterates in an arbitrary order; sort so the example named in
        // a preflight error is the same on every run.
        hits.sort_unstable();
        hits.into_iter().next()
    }

    /// Number of tensor names satisfying `predicate`.
    pub fn count_names<F: Fn(&str) -> bool>(&self, predicate: F) -> usize {
        self.tensors
            .keys()
            .filter(|n| predicate(n.as_str()))
            .count()
    }

    /// Read `row_count` rows of a 2-D tensor starting at `row_start` into
    /// `out`, widened to f32.
    ///
    /// Peak memory is the requested block, never the tensor.
    pub fn read_rows(
        &mut self,
        name: &str,
        row_start: usize,
        row_count: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("checkpoint is missing the required tensor '{name}'"))?
            .clone();
        let (rows, cols) = info.dims_2d(name)?;
        if row_start.saturating_add(row_count) > rows {
            bail!(
                "tensor '{name}': rows {row_start}..{} are outside its {rows} rows",
                row_start + row_count
            );
        }
        let row_bytes = cols
            .checked_mul(info.dtype.size())
            .context("row byte length overflows")?;
        let offset = info.start + (row_start as u64) * (row_bytes as u64);
        let len = (row_count as u64) * (row_bytes as u64);
        self.read_exact_at(info.file, offset, len)?;
        out.clear();
        out.reserve(row_count * cols);
        decode_into(&self.scratch, info.dtype, out);
        Ok(())
    }

    /// Read a whole tensor, widened to f32. Used for the small 1-D RMSNorm
    /// vectors only; `max_elements` bounds the allocation.
    pub fn read_all(&mut self, name: &str, max_elements: usize) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("checkpoint is missing the required tensor '{name}'"))?
            .clone();
        let numel = info.numel();
        if numel > max_elements {
            bail!("tensor '{name}' has {numel} elements; at most {max_elements} were expected");
        }
        self.read_exact_at(info.file, info.start, info.nbytes)?;
        let mut out = Vec::with_capacity(numel);
        decode_into(&self.scratch, info.dtype, &mut out);
        Ok(out)
    }

    fn read_exact_at(&mut self, file_index: usize, offset: u64, len: u64) -> Result<()> {
        let len_usize = usize::try_from(len).context("read length does not fit in memory")?;
        if self.handles[file_index].is_none() {
            let path = &self.files[file_index];
            let handle = File::open(path).with_context(|| format!("opening {}", path.display()))?;
            self.handles[file_index] = Some(handle);
        }
        let path = self.files[file_index].clone();
        let handle = self.handles[file_index].as_mut().expect("handle opened");
        handle
            .seek(SeekFrom::Start(offset))
            .with_context(|| format!("seeking to {offset} in {}", path.display()))?;
        self.scratch.clear();
        self.scratch
            .try_reserve(len_usize)
            .context("allocating tensor read buffer")?;
        self.scratch.resize(len_usize, 0);
        handle.read_exact(&mut self.scratch).with_context(|| {
            format!(
                "reading {len_usize} bytes at {offset} in {}",
                path.display()
            )
        })?;
        Ok(())
    }
}

fn read_shard_header(
    path: &Path,
    file_index: usize,
    tensors: &mut HashMap<String, TensorInfo>,
) -> Result<()> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = file.metadata().context("stat")?.len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .context("reading the 8-byte header length")?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_BYTES || header_len + 8 > file_len {
        bail!("implausible safetensors header length {header_len} for a {file_len}-byte file");
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .context("reading the JSON header")?;
    let parsed: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&header).context("parsing the JSON header")?;

    let data_start = 8 + header_len;
    let data_len = file_len - data_start;

    for (name, value) in parsed {
        if name == "__metadata__" {
            continue;
        }
        let dtype = Dtype::parse(
            value
                .get("dtype")
                .and_then(|v| v.as_str())
                .with_context(|| format!("tensor '{name}' has no dtype"))?,
        )
        .with_context(|| format!("tensor '{name}'"))?;
        let shape: Vec<usize> = value
            .get("shape")
            .and_then(|v| v.as_array())
            .with_context(|| format!("tensor '{name}' has no shape"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .with_context(|| format!("tensor '{name}' has a non-integer shape entry"))
            })
            .collect::<Result<_>>()?;
        let offsets = value
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .with_context(|| format!("tensor '{name}' has no data_offsets"))?;
        if offsets.len() != 2 {
            bail!(
                "tensor '{name}' has {} data_offsets, expected 2",
                offsets.len()
            );
        }
        let begin = offsets[0]
            .as_u64()
            .with_context(|| format!("tensor '{name}' has a non-integer start offset"))?;
        let end = offsets[1]
            .as_u64()
            .with_context(|| format!("tensor '{name}' has a non-integer end offset"))?;
        if end < begin || end > data_len {
            bail!(
                "tensor '{name}' claims bytes {begin}..{end}, outside the {data_len}-byte payload"
            );
        }
        let numel: usize = shape.iter().product();
        let expected = (numel as u64)
            .checked_mul(dtype.size() as u64)
            .context("tensor byte length overflows")?;
        if end - begin != expected {
            bail!(
                "tensor '{name}' spans {} bytes but its shape {shape:?} and dtype need {expected}",
                end - begin
            );
        }
        if tensors
            .insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    file: file_index,
                    start: data_start + begin,
                    nbytes: expected,
                },
            )
            .is_some()
        {
            bail!("tensor '{name}' appears in more than one shard");
        }
    }
    Ok(())
}

/// Widen raw checkpoint bytes to f32.
///
/// Every value passes through f16 on the way, which is what the reference
/// exporter does: `export_rtn.py` loads the checkpoint with
/// `AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16)` and only
/// then calls `.float()`, so the quantizer never sees more precision than f16
/// carries. Reading BF16 straight to f32 would keep two more exponent bits and
/// silently produce a different (though equally valid) model, so this mirrors
/// the reference instead — see `convert::convert` for the byte-identity
/// requirement that pins it.
fn decode_into(bytes: &[u8], dtype: Dtype, out: &mut Vec<f32>) {
    match dtype {
        Dtype::F32 => {
            for chunk in bytes.chunks_exact(4) {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(f16::from_f32(value).to_f32());
            }
        }
        Dtype::F16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        Dtype::BF16 => {
            for chunk in bytes.chunks_exact(2) {
                let value = bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                out.push(f16::from_f32(value).to_f32());
            }
        }
    }
}
