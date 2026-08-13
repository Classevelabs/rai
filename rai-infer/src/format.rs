//! `.raimodel` binary format: validated in-memory reader for GPTQ-4bit models.
//!
//! File layout:
//!   [64 or 128 bytes]     Header (magic, version, model config)
//!   [num_sections * 16]   Section index table (offset, size per section)
//!   [variable]            Section 0: Embedding (8-bit quantized)
//!   [variable]            Sections 1..N: Transformer layers (4-bit quantized linears + f32 norms)
//!   [variable]            Section N+1: Final RMSNorm (f32 weights)
//!   [variable]            Section N+2 (untied only): lm_head (4-bit quantized)
//!
//! # Versions
//!
//! **v1** — 64-byte header. Llama/Mistral-shaped models only: SwiGLU MLP,
//! plain RoPE from `rope_theta`, no bias vectors.
//!
//! **v2** — 128-byte header. Byte-for-byte a superset: bytes `0..56` carry
//! exactly the v1 fields in exactly the v1 positions, so the config parse is
//! shared. Bytes `56..64` stay reserved-zero (they are zero in every v1 file
//! ever written) and `64..128` carry the new capability block:
//!
//! | offset | type | field |
//! |--------|------|-------|
//! | 64     | u8   | `activation` — 0 = SiLU/SwiGLU, 1 = GeLU-tanh/GeGLU |
//! | 65     | u8   | `flags` — bit 0: per-layer projection biases present |
//! | 66     | u8   | `rope_type` — 0 = default, 1 = llama3 |
//! | 67     | u8   | `bias_mask` — bit *i* set: projection *i* carries a bias |
//! | 68..72 | f32  | `rope_factor` |
//! | 72..76 | f32  | `rope_low_freq_factor` |
//! | 76..80 | f32  | `rope_high_freq_factor` |
//! | 80..84 | u32  | `rope_original_max_position` |
//! | 84..88 | f32  | `embed_scale` — input embeddings are multiplied by this |
//! | 88..128| —    | reserved, must be zero |
//!
//! Unknown `activation`, `rope_type`, `flags` bits, `bias_mask` bits and any
//! non-zero reserved byte are hard errors. A reader that cannot implement a
//! capability must refuse the file, never run it with the capability ignored.
//!
//! `bias_mask` bit order is [`PROJECTION_NAMES`] order. When the bias flag is
//! set, a layer section carries, *after* its seven quantized linears and
//! *before* its two norm vectors, one `rows * 4`-byte f32 vector for each set
//! mask bit, in the same order. Biases are stored as f32 rather than quantized:
//! they are `rows` values against a matrix of `rows * cols`, so the space is
//! noise and the quantization error would not be.

use anyhow::{bail, Context, Result};
use half::f16;
use std::io::Read;
use std::path::Path;

// Capacity limits are owned by the modules that enforce them at run time; the
// format validator imports them so that accepting a file here is exactly the
// guarantee the kernels rely on (a loaded model can never exceed the GEMM
// group capacity or the RoPE table budget).
use crate::gemm::MAX_GROUPS;
use crate::layers::{Activation, RopeScaling, MAX_ROPE_TABLE_BYTES};

/// Magic bytes: "RAIM"
const MAGIC: [u8; 4] = *b"RAIM";
/// Highest container version this reader implements.
pub const FORMAT_VERSION: u32 = 2;
const HEADER_SIZE_V1: usize = 64;
const HEADER_SIZE_V2: usize = 128;
const SECTION_ENTRY_SIZE: usize = 16;

/// The seven quantized projections in every layer, in section order. This is
/// also the bit order of the v2 header's `bias_mask`.
pub const PROJECTION_NAMES: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

/// Number of projections per layer (the width of `bias_mask`).
pub const NUM_PROJECTIONS: usize = PROJECTION_NAMES.len();

/// `flags` bit 0: the layer sections carry bias vectors.
pub const FLAG_HAS_BIASES: u8 = 0x01;
/// Every `flags` bit this reader understands.
const KNOWN_FLAGS: u8 = FLAG_HAS_BIASES;
/// Every `bias_mask` bit this reader understands (one per projection).
const KNOWN_BIAS_MASK: u8 = 0x7F;
const MAX_MODEL_FILE_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_HIDDEN_SIZE: u32 = 65_536;
const MAX_INTERMEDIATE_SIZE: u32 = 1_048_576;
const MAX_LAYERS: u32 = 1_024;
const MAX_HEADS: u32 = 1_024;
const MAX_VOCAB_SIZE: u32 = 10_000_000;
const MAX_CONTEXT: u32 = 1_000_000;

/// Model configuration extracted from the header.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub max_context: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub bits: u8,
    pub group_size: u8,
    pub embed_bits: u8,
    pub embed_group_size: u8,
    /// Container version the file was written at (1 or 2).
    pub version: u32,
    /// Gated-MLP non-linearity. v1 files are always [`Activation::Silu`].
    pub activation: Activation,
    /// RoPE frequency transform. v1 files are always [`RopeScaling::None`].
    pub rope_scaling: RopeScaling,
    /// Which projections carry a bias vector; see [`PROJECTION_NAMES`].
    /// Zero for v1 files.
    pub bias_mask: u8,
    /// Multiplier applied to a looked-up input embedding. `1.0` for v1 files.
    ///
    /// Gemma scales embeddings by `sqrt(hidden_size)`. This is *not* folded
    /// into the stored table: Gemma also ties `lm_head` to the same table, and
    /// the output projection uses the *unscaled* weights, so a folded table
    /// would multiply every logit by ~45. See `RaiModel::embed_token`.
    pub embed_scale: f32,
}

impl ModelConfig {
    /// Whether projection `index` (see [`PROJECTION_NAMES`]) carries a bias.
    pub fn has_bias(&self, index: usize) -> bool {
        index < NUM_PROJECTIONS && self.bias_mask & (1 << index) != 0
    }

    /// Total attention width, `num_heads * head_dim`. Equal to `hidden_size`
    /// for Llama-shaped models but not for Gemma-shaped ones.
    pub fn attention_dim(&self) -> usize {
        self.num_heads as usize * self.head_dim as usize
    }

    /// KV width, `num_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads as usize * self.head_dim as usize
    }

    /// `(rows, cols)` of each of the seven layer projections, in section order.
    pub fn projection_dims(&self) -> [(usize, usize); NUM_PROJECTIONS] {
        let hidden = self.hidden_size as usize;
        let inter = self.intermediate_size as usize;
        let q_dim = self.attention_dim();
        let kv_dim = self.kv_dim();
        [
            (q_dim, hidden),  // q_proj
            (kv_dim, hidden), // k_proj
            (kv_dim, hidden), // v_proj
            (hidden, q_dim),  // o_proj
            (inter, hidden),  // gate_proj
            (inter, hidden),  // up_proj
            (hidden, inter),  // down_proj
        ]
    }
}

/// What a `.raimodel` file says about itself, read from the header alone.
///
/// Produced by [`RaiModelFile::read_summary`] without loading the weights.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub config: ModelConfig,
    /// Section count declared in the header.
    pub num_sections: usize,
    /// Size of the file on disk, in bytes.
    pub file_bytes: u64,
}

impl ModelSummary {
    /// True when the output projection is tied to the embedding table, i.e.
    /// the file carries no separate `lm_head` section.
    pub fn tied_embeddings(&self) -> bool {
        self.num_sections == self.config.num_layers as usize + 2
    }
}

/// A section in the file (offset + size).
#[derive(Debug, Clone, Copy)]
pub struct SectionEntry {
    pub offset: u64,
    pub size: u64,
}

/// Borrowed view of a quantized linear layer in the validated model buffer.
#[derive(Debug, Clone)]
pub struct QuantizedLinear<'a> {
    pub rows: usize,
    pub cols: usize,
    pub group_params: &'a [u8], // [rows * num_groups * 4] — f16 scale, f16 zero per row per group
    pub nibble_data: &'a [u8],  // [rows * cols / 2] — packed 4-bit codes
    pub group_size: usize,
}

/// Zero-copy reference to an 8-bit quantized embedding table.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding<'a> {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub group_size: usize,
    pub group_params: &'a [u8], // [vocab_size * num_groups * 4]
    pub data: &'a [u8],         // [vocab_size * hidden_size] — uint8 codes
}

/// Zero-copy reference to RMSNorm weights.
#[derive(Debug, Clone)]
pub struct RMSNormWeights<'a> {
    pub weights: &'a [u8], // [hidden_size * 4] — f32 values as raw bytes
    pub size: usize,
}

/// A transformer layer's borrowed views into the validated model buffer.
#[derive(Debug, Clone)]
pub struct LayerRefs<'a> {
    pub q_proj: QuantizedLinear<'a>,
    pub k_proj: QuantizedLinear<'a>,
    pub v_proj: QuantizedLinear<'a>,
    pub o_proj: QuantizedLinear<'a>,
    pub gate_proj: QuantizedLinear<'a>,
    pub up_proj: QuantizedLinear<'a>,
    pub down_proj: QuantizedLinear<'a>,
    pub input_layernorm: RMSNormWeights<'a>,
    pub post_attn_layernorm: RMSNormWeights<'a>,
    /// Raw little-endian f32 bias vectors, indexed by [`PROJECTION_NAMES`]
    /// order. `None` wherever the header's `bias_mask` bit is clear — which is
    /// every entry for a v1 file.
    pub biases: [Option<&'a [u8]>; NUM_PROJECTIONS],
}

/// The full model file: header + validated heap-allocated data.
pub struct RaiModelFile {
    pub config: ModelConfig,
    pub sections: Vec<SectionEntry>,
    data: Vec<u8>,
}

impl RaiModelFile {
    /// Open and validate a .raimodel file.
    /// Reads the entire file into a heap-allocated buffer for maximum memory bandwidth.
    pub fn open(path: &Path) -> Result<Self> {
        // Open before inspecting metadata so path replacement cannot make us validate one file
        // and read another one.
        let mut file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let file_len_u64 = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        if file_len_u64 > MAX_MODEL_FILE_BYTES {
            bail!("model file is too large: {file_len_u64} bytes exceeds {MAX_MODEL_FILE_BYTES}");
        }
        let file_len =
            usize::try_from(file_len_u64).context("model file does not fit in memory")?;

        let mut data = Vec::new();
        data.try_reserve_exact(file_len)
            .context("allocating model file buffer")?;

        read_expected_plus_one(&mut file, file_len_u64, &mut data).context("reading model file")?;
        if data.len() != file_len {
            bail!(
                "model file changed while reading: expected {file_len} bytes, read {}",
                data.len()
            );
        }

        let HeaderInfo {
            config,
            num_sections,
            header_size,
        } = parse_header(&data)?;

        // Parse section index table (right after header)
        let table_start = header_size;
        let table_size = checked_mul(num_sections, SECTION_ENTRY_SIZE, "section table size")?;
        let table_end = checked_add(table_start, table_size, "section table end")?;
        if data.len() < table_end {
            bail!("file too small for section table");
        }

        let mut sections = Vec::with_capacity(num_sections);
        let mut previous_end = table_end;
        for i in 0..num_sections {
            let off = table_start + i * SECTION_ENTRY_SIZE;
            let offset = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            let size = u64::from_le_bytes(data[off + 8..off + 16].try_into().unwrap());
            let start = usize::try_from(offset)
                .with_context(|| format!("section {i} offset does not fit in memory"))?;
            let size_usize = usize::try_from(size)
                .with_context(|| format!("section {i} size does not fit in memory"))?;
            let end = start
                .checked_add(size_usize)
                .with_context(|| format!("section {i} range overflows"))?;
            if size == 0 {
                bail!("section {i} is empty");
            }
            if start != previous_end {
                bail!("section {i} is not contiguous: starts at {start}, expected {previous_end}");
            }
            if end > data.len() {
                bail!("section {i} extends beyond file");
            }
            previous_end = end;
            sections.push(SectionEntry { offset, size });
        }
        if previous_end != data.len() {
            bail!(
                "unreferenced trailing data: final section ends at {previous_end}, file is {} bytes",
                data.len()
            );
        }

        let model = Self {
            config,
            sections,
            data,
        };
        model.validate_layout()?;
        Ok(model)
    }

    /// Read only the header of a `.raimodel` file.
    ///
    /// [`Self::open`] pulls the whole file into memory, which is the right
    /// trade for inference and the wrong one for `rai models`: listing a
    /// directory of 7B checkpoints would read tens of gigabytes to print a few
    /// integers. This touches the first 128 bytes and runs the same header
    /// parse and validation, so a file that summarises here is a file whose
    /// header `open` would also accept.
    pub fn read_summary(path: &Path) -> Result<ModelSummary> {
        let mut file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let file_bytes = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();

        let mut head = Vec::with_capacity(HEADER_SIZE_V2);
        read_expected_plus_one(&mut file, HEADER_SIZE_V2 as u64, &mut head)
            .with_context(|| format!("reading the header of {}", path.display()))?;

        let HeaderInfo {
            config,
            num_sections,
            ..
        } = parse_header(&head)?;
        Ok(ModelSummary {
            config,
            num_sections,
            file_bytes,
        })
    }

    /// Get the raw bytes for a section.
    fn section_data(&self, idx: usize) -> Result<&[u8]> {
        let s = self
            .sections
            .get(idx)
            .context("section index out of range")?;
        let start = usize::try_from(s.offset).context("section offset does not fit in memory")?;
        let size = usize::try_from(s.size).context("section size does not fit in memory")?;
        let end = start.checked_add(size).context("section range overflows")?;
        if end > self.data.len() {
            bail!("section {idx} extends beyond file");
        }
        Ok(&self.data[start..end])
    }

    fn validate_layout(&self) -> Result<()> {
        let embedding = self.embedding().context("validating embedding section")?;
        validate_group_params(embedding.group_params, "embedding")?;
        for layer in 0..self.config.num_layers as usize {
            let refs = self
                .layer(layer)
                .with_context(|| format!("validating layer {layer}"))?;
            for (name, linear) in [
                ("q_proj", &refs.q_proj),
                ("k_proj", &refs.k_proj),
                ("v_proj", &refs.v_proj),
                ("o_proj", &refs.o_proj),
                ("gate_proj", &refs.gate_proj),
                ("up_proj", &refs.up_proj),
                ("down_proj", &refs.down_proj),
            ] {
                validate_group_params(linear.group_params, &format!("layer {layer} {name}"))?;
            }
            // Every declared bias must be present, exactly `rows` long, and
            // finite — the same standard the norm vectors are held to.
            let dims = self.config.projection_dims();
            for (index, name) in PROJECTION_NAMES.iter().enumerate() {
                match (self.config.has_bias(index), refs.biases[index]) {
                    (false, None) => {}
                    (true, Some(bytes)) => {
                        let expected = checked_mul(dims[index].0, 4, "bias bytes")?;
                        if bytes.len() != expected {
                            bail!(
                                "layer {layer} {name} bias is {} bytes, expected {expected}",
                                bytes.len()
                            );
                        }
                        validate_f32_vector(bytes, &format!("layer {layer} {name} bias"))?;
                    }
                    (declared, _) => bail!(
                        "layer {layer} {name} bias presence disagrees with the header \
                         (bias_mask says {declared})"
                    ),
                }
            }
            validate_norm_weights(&refs.input_layernorm, &format!("layer {layer} input norm"))?;
            validate_norm_weights(
                &refs.post_attn_layernorm,
                &format!("layer {layer} post-attention norm"),
            )?;
        }
        let final_norm = self.final_norm().context("validating final norm")?;
        validate_norm_weights(&final_norm, "final norm")?;
        if let Some(lm_head) = self.lm_head().context("validating lm_head")? {
            validate_group_params(lm_head.group_params, "lm_head")?;
        }
        Ok(())
    }

    /// Total data length in bytes.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Parse the embedding section (section 0).
    pub fn embedding(&self) -> Result<QuantizedEmbedding<'_>> {
        let data = self.section_data(0)?;
        let vocab = self.config.vocab_size as usize;
        let hidden = self.config.hidden_size as usize;
        let gs = self.config.embed_group_size as usize;
        let num_groups = hidden.div_ceil(gs);
        let params_size = checked_mul(
            checked_mul(vocab, num_groups, "embedding parameter groups")?,
            4,
            "embedding parameter bytes",
        )?;
        let data_size = checked_mul(vocab, hidden, "embedding data bytes")?;
        let expected_size = checked_add(params_size, data_size, "embedding section size")?;

        if data.len() != expected_size {
            bail!(
                "embedding section size mismatch: {} != {} + {}",
                data.len(),
                params_size,
                data_size
            );
        }

        Ok(QuantizedEmbedding {
            vocab_size: vocab,
            hidden_size: hidden,
            group_size: gs,
            group_params: &data[..params_size],
            data: &data[params_size..params_size + data_size],
        })
    }

    /// Parse a transformer layer section (sections 1..num_layers).
    pub fn layer(&self, layer_idx: usize) -> Result<LayerRefs<'_>> {
        if layer_idx >= self.config.num_layers as usize {
            bail!("layer index {layer_idx} out of range");
        }
        let section_idx = 1 + layer_idx;
        let data = self.section_data(section_idx)?;
        let gs = self.config.group_size as usize;
        let hidden = self.config.hidden_size as usize;

        // Each linear sub-section: [u32 rows][u32 cols][group_params][nibble_data]
        let linear_dims = self.config.projection_dims();

        let mut offset = 0usize;
        let mut linears: Vec<QuantizedLinear<'_>> = Vec::with_capacity(NUM_PROJECTIONS);

        for &(rows, cols) in &linear_dims {
            // Sub-header: 8 bytes (rows: u32, cols: u32)
            let header_end = checked_add(offset, 8, "linear sub-header")?;
            if header_end > data.len() {
                bail!("layer {layer_idx}: truncated linear sub-header at offset {offset}");
            }
            let r = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            let c = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
            if r != rows || c != cols {
                bail!("layer {layer_idx}: dimension mismatch: expected ({rows},{cols}), got ({r},{c})");
            }
            offset += 8;

            let num_groups = cols.div_ceil(gs);
            let params_size = checked_mul(
                checked_mul(rows, num_groups, "linear parameter groups")?,
                4,
                "linear parameter bytes",
            )?;
            let elements = checked_mul(rows, cols, "linear elements")?;
            let nibble_size = elements.div_ceil(2);
            let linear_size = checked_add(params_size, nibble_size, "linear data size")?;
            let linear_end = checked_add(offset, linear_size, "linear data end")?;

            if linear_end > data.len() {
                bail!("layer {layer_idx}: linear data truncated");
            }

            linears.push(QuantizedLinear {
                rows,
                cols,
                group_params: &data[offset..offset + params_size],
                nibble_data: &data[offset + params_size..offset + params_size + nibble_size],
                group_size: gs,
            });
            offset = linear_end;
        }

        // Bias block: one `rows * 4`-byte f32 vector per set `bias_mask` bit,
        // in projection order. Absent entirely for v1 and for v2 files whose
        // mask is zero, which is why a v1 section's size is unchanged.
        let mut biases: [Option<&[u8]>; NUM_PROJECTIONS] = [None; NUM_PROJECTIONS];
        for (index, &(rows, _)) in linear_dims.iter().enumerate() {
            if !self.config.has_bias(index) {
                continue;
            }
            let bias_bytes = checked_mul(rows, 4, "bias bytes")?;
            let end = checked_add(offset, bias_bytes, "bias block end")?;
            if end > data.len() {
                bail!(
                    "layer {layer_idx}: {} bias truncated",
                    PROJECTION_NAMES[index]
                );
            }
            biases[index] = Some(&data[offset..end]);
            offset = end;
        }

        // Two RMSNorm weight vectors: hidden_size * 4 bytes each
        let norm_bytes = checked_mul(hidden, 4, "norm bytes")?;
        let both_norms = checked_mul(2, norm_bytes, "layer norm bytes")?;
        let expected_end = checked_add(offset, both_norms, "layer section size")?;
        if expected_end != data.len() {
            bail!(
                "layer {layer_idx}: section size mismatch: {} != {expected_end}",
                data.len()
            );
        }
        let input_ln = RMSNormWeights {
            weights: &data[offset..offset + norm_bytes],
            size: hidden,
        };
        offset += norm_bytes;
        let post_attn_ln = RMSNormWeights {
            weights: &data[offset..offset + norm_bytes],
            size: hidden,
        };

        Ok(LayerRefs {
            q_proj: linears[0].clone(),
            k_proj: linears[1].clone(),
            v_proj: linears[2].clone(),
            o_proj: linears[3].clone(),
            gate_proj: linears[4].clone(),
            up_proj: linears[5].clone(),
            down_proj: linears[6].clone(),
            input_layernorm: input_ln,
            post_attn_layernorm: post_attn_ln,
            biases,
        })
    }

    /// Parse the final RMSNorm section.
    /// With untied lm_head: embed + layers + norm + lm_head = num_layers + 3 sections.
    /// With tied lm_head:   embed + layers + norm           = num_layers + 2 sections.
    pub fn final_norm(&self) -> Result<RMSNormWeights<'_>> {
        let norm_idx = 1 + self.config.num_layers as usize; // section after all layers
        let data = self.section_data(norm_idx)?;
        let hidden = self.config.hidden_size as usize;
        let norm_bytes = checked_mul(hidden, 4, "final norm bytes")?;
        if data.len() != norm_bytes {
            bail!(
                "final norm section size mismatch: {} != {norm_bytes}",
                data.len()
            );
        }
        Ok(RMSNormWeights {
            weights: data,
            size: hidden,
        })
    }

    /// Check if the model has a separate (untied) lm_head section.
    pub fn has_lm_head(&self) -> bool {
        self.sections.len() == 1 + self.config.num_layers as usize + 2
    }

    /// Parse the separate lm_head section (last section, if present).
    /// This is a 4-bit quantized linear: [vocab_size x hidden_size].
    pub fn lm_head(&self) -> Result<Option<QuantizedLinear<'_>>> {
        if !self.has_lm_head() {
            return Ok(None);
        }
        let idx = self.sections.len() - 1;
        let data = self.section_data(idx)?;
        let gs = self.config.group_size as usize;
        let vocab = self.config.vocab_size as usize;
        let hidden = self.config.hidden_size as usize;

        if data.len() < 8 {
            bail!("lm_head section too small for header");
        }
        let rows = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let cols = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if rows != vocab || cols != hidden {
            bail!("lm_head dimension mismatch: expected ({vocab},{hidden}), got ({rows},{cols})");
        }

        let num_groups = cols.div_ceil(gs);
        let params_size = checked_mul(
            checked_mul(rows, num_groups, "lm_head parameter groups")?,
            4,
            "lm_head parameter bytes",
        )?;
        let nibble_size = checked_mul(rows, cols, "lm_head elements")?.div_ceil(2);
        let offset = 8;
        let expected_size = checked_add(
            offset,
            checked_add(params_size, nibble_size, "lm_head data size")?,
            "lm_head section size",
        )?;

        if data.len() != expected_size {
            bail!(
                "lm_head section size mismatch: {} != {expected_size}",
                data.len()
            );
        }

        Ok(Some(QuantizedLinear {
            rows,
            cols,
            group_params: &data[offset..offset + params_size],
            nibble_data: &data[offset + params_size..offset + params_size + nibble_size],
            group_size: gs,
        }))
    }
}

/// The fixed-size head of a `.raimodel` file, parsed and validated.
struct HeaderInfo {
    config: ModelConfig,
    num_sections: usize,
    /// Where the section index table starts, i.e. the version's header size.
    header_size: usize,
}

/// Parse and validate the header out of the leading bytes of a model file.
///
/// `data` needs to be at least the header long; anything past it is ignored,
/// which is what lets [`RaiModelFile::read_summary`] run the identical check on
/// 128 bytes that [`RaiModelFile::open`] runs on the whole file.
fn parse_header(data: &[u8]) -> Result<HeaderInfo> {
    if data.len() < HEADER_SIZE_V1 {
        bail!("file too small for header: {} bytes", data.len());
    }

    if data[0..4] != MAGIC {
        bail!("invalid magic bytes");
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let header_size = match version {
        1 => HEADER_SIZE_V1,
        2 => HEADER_SIZE_V2,
        other => bail!(
            "unsupported format version: {other}; this reader implements 1 and \
             {FORMAT_VERSION}"
        ),
    };
    if data.len() < header_size {
        bail!(
            "file too small for a v{version} header: {} bytes, need {header_size}",
            data.len()
        );
    }

    // Bytes 0..56 are identical in both versions, so the core config parse
    // is shared and a v1 file still loads exactly as it always did.
    let mut config = ModelConfig {
        hidden_size: read_u32(data, 8),
        num_layers: read_u32(data, 12),
        num_heads: read_u32(data, 16),
        num_kv_heads: read_u32(data, 20),
        head_dim: read_u32(data, 24),
        intermediate_size: read_u32(data, 28),
        vocab_size: read_u32(data, 32),
        max_context: read_u32(data, 36),
        rope_theta: f32::from_le_bytes(data[40..44].try_into().unwrap()),
        norm_eps: f32::from_le_bytes(data[44..48].try_into().unwrap()),
        bits: data[48],
        group_size: data[49],
        embed_bits: data[50],
        embed_group_size: data[51],
        version,
        activation: Activation::Silu,
        rope_scaling: RopeScaling::None,
        bias_mask: 0,
        embed_scale: 1.0,
    };
    if version >= 2 {
        read_v2_capabilities(data, &mut config)?;
    }
    validate_config(&config)?;

    let num_sections = read_u32(data, 52) as usize;
    let tied_sections = checked_add(config.num_layers as usize, 2, "section count")?;
    let untied_sections = checked_add(config.num_layers as usize, 3, "section count")?;
    if num_sections != tied_sections && num_sections != untied_sections {
        bail!(
            "invalid section count {num_sections}; expected {tied_sections} (tied) or \
             {untied_sections} (untied)"
        );
    }

    Ok(HeaderInfo {
        config,
        num_sections,
        header_size,
    })
}

/// Decode the v2 capability block (`64..128`) into `config`.
///
/// Every unknown code, unknown bit and non-zero reserved byte is an error.
/// Silently ignoring one of these would mean running a file that used a
/// capability this build does not implement — the exact failure mode a version
/// field exists to prevent — and the result would be a model that loads,
/// generates, and is wrong.
fn read_v2_capabilities(data: &[u8], config: &mut ModelConfig) -> Result<()> {
    // Bytes 56..64 are the v1 header's tail. Every v1 writer zeroes them; a v2
    // file that puts something there is using a field this build has no name
    // for.
    if data[56..64].iter().any(|&byte| byte != 0) {
        bail!("header bytes 56..64 are reserved and must be zero");
    }
    if data[88..HEADER_SIZE_V2].iter().any(|&byte| byte != 0) {
        bail!("header bytes 88..128 are reserved and must be zero");
    }

    let activation_code = data[64];
    config.activation = Activation::from_code(activation_code).with_context(|| {
        format!("unsupported activation code {activation_code}; this reader implements 0 (SiLU) and 1 (GeLU-tanh)")
    })?;

    let flags = data[65];
    if flags & !KNOWN_FLAGS != 0 {
        bail!(
            "header flags {flags:#04x} set bits this reader does not implement (known: \
             {KNOWN_FLAGS:#04x})"
        );
    }

    let bias_mask = data[67];
    if bias_mask & !KNOWN_BIAS_MASK != 0 {
        bail!(
            "bias_mask {bias_mask:#04x} sets bits beyond the {NUM_PROJECTIONS} projections \
             (known: {KNOWN_BIAS_MASK:#04x})"
        );
    }
    // The flag and the mask must agree, or a section's size is ambiguous.
    let flag_set = flags & FLAG_HAS_BIASES != 0;
    if flag_set != (bias_mask != 0) {
        bail!(
            "header is inconsistent: bias flag is {} but bias_mask is {bias_mask:#04x}",
            if flag_set { "set" } else { "clear" }
        );
    }
    config.bias_mask = bias_mask;

    let rope_type = data[66];
    let factor = f32::from_le_bytes(data[68..72].try_into().unwrap());
    let low = f32::from_le_bytes(data[72..76].try_into().unwrap());
    let high = f32::from_le_bytes(data[76..80].try_into().unwrap());
    let original_max_position = read_u32(data, 80);
    config.rope_scaling = match rope_type {
        0 => {
            // A default-RoPE file must not smuggle parameters past a reader
            // that would ignore them.
            if factor != 0.0 || low != 0.0 || high != 0.0 || original_max_position != 0 {
                bail!("rope_type 0 (default) must leave the RoPE scaling parameters zero");
            }
            RopeScaling::None
        }
        1 => {
            for (name, value) in [
                ("factor", factor),
                ("low_freq_factor", low),
                ("high_freq_factor", high),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    bail!("llama3 RoPE {name} must be finite and positive, got {value}");
                }
            }
            if (high - low).abs() == 0.0 {
                bail!("llama3 RoPE high_freq_factor must differ from low_freq_factor");
            }
            if original_max_position == 0 || original_max_position > MAX_CONTEXT {
                bail!(
                    "llama3 RoPE original_max_position_embeddings must be in 1..={MAX_CONTEXT}, \
                     got {original_max_position}"
                );
            }
            RopeScaling::Llama3 {
                factor,
                low_freq_factor: low,
                high_freq_factor: high,
                original_max_position,
            }
        }
        other => bail!(
            "unsupported rope_type {other}; this reader implements 0 (default) and 1 (llama3)"
        ),
    };

    let embed_scale = f32::from_le_bytes(data[84..88].try_into().unwrap());
    if !embed_scale.is_finite() || embed_scale <= 0.0 {
        bail!("embed_scale must be finite and positive, got {embed_scale}");
    }
    config.embed_scale = embed_scale;

    Ok(())
}

fn validate_config(config: &ModelConfig) -> Result<()> {
    let bounded_nonzero = [
        ("hidden_size", config.hidden_size, MAX_HIDDEN_SIZE),
        ("num_layers", config.num_layers, MAX_LAYERS),
        ("num_heads", config.num_heads, MAX_HEADS),
        ("num_kv_heads", config.num_kv_heads, MAX_HEADS),
        ("head_dim", config.head_dim, MAX_HIDDEN_SIZE),
        (
            "intermediate_size",
            config.intermediate_size,
            MAX_INTERMEDIATE_SIZE,
        ),
        ("vocab_size", config.vocab_size, MAX_VOCAB_SIZE),
        ("max_context", config.max_context, MAX_CONTEXT),
    ];
    for (name, value, maximum) in bounded_nonzero {
        if value == 0 || value > maximum {
            bail!("invalid {name}: {value}; expected 1..={maximum}");
        }
    }

    if config.bits != 4 {
        bail!("unsupported weight bit width {}; expected 4", config.bits);
    }
    if config.embed_bits != 8 {
        bail!(
            "unsupported embedding bit width {}; expected 8",
            config.embed_bits
        );
    }
    if config.group_size == 0 || config.embed_group_size == 0 {
        bail!("quantization group sizes must be non-zero");
    }
    if !config.group_size.is_multiple_of(2) || !config.embed_group_size.is_multiple_of(2) {
        bail!("quantization group sizes must be even");
    }
    if !config.hidden_size.is_multiple_of(2) || !config.intermediate_size.is_multiple_of(2) {
        bail!("hidden and intermediate dimensions must be even for packed 4-bit kernels");
    }
    if !config.head_dim.is_multiple_of(8) {
        bail!("head_dim must be a multiple of 8 for SIMD attention kernels");
    }
    if !config.num_heads.is_multiple_of(config.num_kv_heads) {
        bail!("num_heads must be divisible by num_kv_heads");
    }

    // `num_heads * head_dim` need not equal `hidden_size`: Gemma decouples them
    // (2b: 8 heads x 256 happens to match, 7b: 16 x 256 = 4096 against a 3072
    // hidden state). q_proj is [q_dim, hidden] and o_proj is [hidden, q_dim],
    // so the attention block reads and writes hidden-sized vectors either way
    // and only its interior is q_dim wide. What the kernels *do* require is
    // that both widths stay packable and inside the GEMM group budget.
    let attention_dim = checked_mul(
        config.num_heads as usize,
        config.head_dim as usize,
        "attention width",
    )?;
    if attention_dim > MAX_INTERMEDIATE_SIZE as usize {
        bail!("num_heads * head_dim is {attention_dim}; the maximum is {MAX_INTERMEDIATE_SIZE}");
    }
    if !attention_dim.is_multiple_of(2) {
        bail!("num_heads * head_dim must be even for packed 4-bit kernels");
    }

    let group_size = config.group_size as usize;
    // `attention_dim` is a *column* count only for o_proj, so it too must fit
    // the group budget.
    let max_linear_groups = (config.hidden_size as usize)
        .div_ceil(group_size)
        .max((config.intermediate_size as usize).div_ceil(group_size))
        .max(attention_dim.div_ceil(group_size));
    if max_linear_groups > MAX_GROUPS {
        bail!(
            "model requires {max_linear_groups} quantization groups; kernel maximum is {MAX_GROUPS}"
        );
    }
    let embedding_groups = (config.hidden_size as usize).div_ceil(config.embed_group_size as usize);
    if embedding_groups > MAX_GROUPS {
        bail!(
            "embedding requires {embedding_groups} quantization groups; kernel maximum is {MAX_GROUPS}"
        );
    }
    if !config.rope_theta.is_finite() || config.rope_theta <= 0.0 {
        bail!("rope_theta must be finite and positive");
    }
    if !config.norm_eps.is_finite() || config.norm_eps <= 0.0 {
        bail!("norm_eps must be finite and positive");
    }

    let rope_values = checked_mul(
        config.max_context as usize,
        (config.head_dim as usize) / 2,
        "RoPE table elements",
    )?;
    let rope_bytes = checked_mul(
        checked_mul(rope_values, 2, "RoPE sine/cosine elements")?,
        std::mem::size_of::<f32>(),
        "RoPE table bytes",
    )?;
    if rope_bytes > MAX_ROPE_TABLE_BYTES {
        bail!("RoPE table requires {rope_bytes} bytes; maximum is {MAX_ROPE_TABLE_BYTES}");
    }

    Ok(())
}

fn read_expected_plus_one(
    reader: &mut impl Read,
    expected_len: u64,
    data: &mut Vec<u8>,
) -> std::io::Result<usize> {
    reader
        .take(expected_len.saturating_add(1))
        .read_to_end(data)
}

fn validate_group_params(params: &[u8], label: &str) -> Result<()> {
    if !params.len().is_multiple_of(4) {
        bail!("{label} quantization parameters are misaligned");
    }
    for (index, values) in params.chunks_exact(4).enumerate() {
        let scale = f16::from_bits(u16::from_le_bytes([values[0], values[1]])).to_f32();
        let zero = f16::from_bits(u16::from_le_bytes([values[2], values[3]])).to_f32();
        if !scale.is_finite() || scale <= 0.0 || !zero.is_finite() {
            bail!("{label} quantization group {index} contains an invalid scale or zero point");
        }
    }
    Ok(())
}

fn validate_norm_weights(norm: &RMSNormWeights<'_>, label: &str) -> Result<()> {
    validate_f32_vector(norm.weights, label)
}

fn validate_f32_vector(bytes: &[u8], label: &str) -> Result<()> {
    if !bytes.len().is_multiple_of(4) {
        bail!("{label} is not a whole number of f32 values");
    }
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        if !value.is_finite() {
            bail!("{label} weight {index} is non-finite");
        }
    }
    Ok(())
}

/// Read little-endian f32 values out of a raw vector in the model buffer.
pub fn read_f32_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("{label} overflows"))
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_add(right)
        .with_context(|| format!("{label} overflows"))
}

/// Read f32 values from RMSNorm raw bytes into a Vec.
pub fn read_norm_weights(norm: &RMSNormWeights<'_>) -> Vec<f32> {
    let mut out = Vec::with_capacity(norm.size);
    for i in 0..norm.size {
        let off = i * 4;
        let val = f32::from_le_bytes(norm.weights[off..off + 4].try_into().unwrap());
        out.push(val);
    }
    out
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn valid_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 256,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            intermediate_size: 512,
            vocab_size: 32_000,
            max_context: 2_048,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            bits: 4,
            group_size: 128,
            embed_bits: 8,
            embed_group_size: 128,
            version: 1,
            activation: Activation::Silu,
            rope_scaling: RopeScaling::None,
            bias_mask: 0,
            embed_scale: 1.0,
        }
    }

    #[test]
    fn test_config_size() {
        // v1 headers stay exactly 64 bytes; v2 extends to 128.
        assert_eq!(HEADER_SIZE_V1, 64);
        assert_eq!(HEADER_SIZE_V2, 128);
    }

    #[test]
    fn test_section_entry_size() {
        assert_eq!(SECTION_ENTRY_SIZE, 16);
    }

    #[test]
    fn validates_safe_model_configuration() {
        assert!(validate_config(&valid_config()).is_ok());
    }

    #[test]
    fn rejects_dimensions_that_do_not_match_attention_layout() {
        // An odd hidden_size cannot be packed into nibbles.
        let mut config = valid_config();
        config.hidden_size = 257;
        assert!(validate_config(&config).is_err());

        // head_dim must stay a multiple of 8 for the SIMD attention kernels.
        let mut config = valid_config();
        config.head_dim = 60;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn accepts_a_decoupled_head_dim() {
        // Gemma-7b shape: 16 heads x 256 = 4096 against a 3072 hidden state.
        // q_proj is [4096, 3072] and o_proj [3072, 4096], so the block still
        // reads and writes hidden-sized vectors.
        let mut config = valid_config();
        config.hidden_size = 3_072;
        config.num_heads = 16;
        config.num_kv_heads = 16;
        config.head_dim = 256;
        config.intermediate_size = 12_288;
        validate_config(&config).expect("a decoupled head_dim must be representable");
        assert_eq!(config.attention_dim(), 4_096);
        assert_eq!(config.projection_dims()[0], (4_096, 3_072));
        assert_eq!(config.projection_dims()[3], (3_072, 4_096));
    }

    #[test]
    fn rejects_zero_groups_and_excessive_kernel_groups() {
        let mut config = valid_config();
        config.group_size = 0;
        assert!(validate_config(&config).is_err());

        let mut config = valid_config();
        config.group_size = 1;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn rejects_non_finite_numeric_configuration() {
        let mut config = valid_config();
        config.rope_theta = f32::NAN;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn rejects_configuration_with_excessive_rope_allocation() {
        let mut config = valid_config();
        config.hidden_size = 65_536;
        config.num_heads = 1;
        config.num_kv_heads = 1;
        config.head_dim = 65_536;
        config.max_context = MAX_CONTEXT;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn opens_a_strictly_valid_minimal_model() {
        let bytes = minimal_model_bytes();
        let path = temporary_model_path("valid");
        std::fs::write(&path, bytes).unwrap();

        let model = RaiModelFile::open(&path).unwrap();
        assert_eq!(model.config.hidden_size, 8);
        assert_eq!(model.sections.len(), 3);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_sections_that_overlap_the_index_table() {
        let mut bytes = minimal_model_bytes();
        bytes[HEADER_SIZE_V1..HEADER_SIZE_V1 + 8].copy_from_slice(&0_u64.to_le_bytes());
        let path = temporary_model_path("overlap");
        std::fs::write(&path, bytes).unwrap();

        let error = RaiModelFile::open(&path).err().unwrap();
        assert!(error.to_string().contains("not contiguous"));

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounded_reader_stops_after_expected_length_plus_sentinel() {
        let mut source = std::io::Cursor::new(vec![0_u8; 64]);
        let mut data = Vec::new();
        read_expected_plus_one(&mut source, 8, &mut data).unwrap();
        assert_eq!(data.len(), 9);
    }

    #[test]
    fn rejects_non_finite_quantization_parameters() {
        let mut bytes = minimal_model_bytes();
        let embedding_start = HEADER_SIZE_V1 + 3 * SECTION_ENTRY_SIZE;
        bytes[embedding_start..embedding_start + 2].copy_from_slice(&f16::NAN.to_le_bytes());
        let path = temporary_model_path("nan-quant-param");
        std::fs::write(&path, bytes).unwrap();

        let error = RaiModelFile::open(&path).err().unwrap();
        assert!(error.to_string().contains("invalid scale"));

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_non_positive_quantization_scale() {
        let mut bytes = minimal_model_bytes();
        let embedding_start = HEADER_SIZE_V1 + 3 * SECTION_ENTRY_SIZE;
        bytes[embedding_start..embedding_start + 2].copy_from_slice(&f16::ZERO.to_le_bytes());
        let path = temporary_model_path("zero-quant-scale");
        std::fs::write(&path, bytes).unwrap();

        let error = RaiModelFile::open(&path).err().unwrap();
        assert!(error.to_string().contains("invalid scale"));

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_non_finite_norm_weights() {
        let mut bytes = minimal_model_bytes();
        let table_start = HEADER_SIZE_V1;
        let layer_entry = table_start + SECTION_ENTRY_SIZE;
        let layer_offset =
            u64::from_le_bytes(bytes[layer_entry..layer_entry + 8].try_into().unwrap()) as usize;
        let layer_size = u64::from_le_bytes(
            bytes[layer_entry + 8..layer_entry + SECTION_ENTRY_SIZE]
                .try_into()
                .unwrap(),
        ) as usize;
        let first_norm = layer_offset + layer_size - 16;
        bytes[first_norm..first_norm + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let path = temporary_model_path("nan-norm");
        std::fs::write(&path, bytes).unwrap();

        let error = RaiModelFile::open(&path).err().unwrap();
        assert!(error.to_string().contains("non-finite"));

        std::fs::remove_file(path).unwrap();
    }

    fn minimal_model_bytes() -> Vec<u8> {
        build_model_bytes(1, 0, |_| {})
    }

    /// Assemble the smallest legal model: 8-wide, 1 layer, 2-token vocab.
    ///
    /// `version` picks the header size, `bias_mask` adds the corresponding
    /// bias vectors to the layer section, and `patch` gets the finished header
    /// so a test can corrupt exactly one field.
    fn build_model_bytes(version: u32, bias_mask: u8, patch: impl Fn(&mut [u8])) -> Vec<u8> {
        let header_size = if version >= 2 {
            HEADER_SIZE_V2
        } else {
            HEADER_SIZE_V1
        };
        let mut header = vec![0_u8; header_size];
        header[0..4].copy_from_slice(&MAGIC);
        header[4..8].copy_from_slice(&version.to_le_bytes());
        for (offset, value) in [
            (8, 8_u32), // hidden
            (12, 1),    // layers
            (16, 1),    // heads
            (20, 1),    // KV heads
            (24, 8),    // head dim
            (28, 8),    // intermediate
            (32, 2),    // vocab
            (36, 4),    // context
        ] {
            header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        header[40..44].copy_from_slice(&10_000_f32.to_le_bytes());
        header[44..48].copy_from_slice(&1e-5_f32.to_le_bytes());
        header[48] = 4;
        header[49] = 8;
        header[50] = 8;
        header[51] = 8;
        header[52..56].copy_from_slice(&3_u32.to_le_bytes());
        if version >= 2 {
            header[65] = if bias_mask != 0 { FLAG_HAS_BIASES } else { 0 };
            header[67] = bias_mask;
            header[84..88].copy_from_slice(&1.0_f32.to_le_bytes());
        }
        patch(&mut header);

        let mut embedding = quant_params(2);
        embedding.extend_from_slice(&[0_u8; 16]);
        let mut layer = Vec::new();
        for _ in 0..NUM_PROJECTIONS {
            layer.extend_from_slice(&8_u32.to_le_bytes());
            layer.extend_from_slice(&8_u32.to_le_bytes());
            layer.extend_from_slice(&quant_params(8));
            layer.extend_from_slice(&[0_u8; 32]);
        }
        // Every projection in this fixture is 8 rows, so each bias is 32 bytes.
        for index in 0..NUM_PROJECTIONS {
            if bias_mask & (1 << index) != 0 {
                for row in 0..8_u32 {
                    layer.extend_from_slice(&(row as f32 * 0.5).to_le_bytes());
                }
            }
        }
        layer.extend_from_slice(&[0_u8; 64]);
        let final_norm = vec![0_u8; 32];
        let sections = [embedding, layer, final_norm];

        let table_size = sections.len() * SECTION_ENTRY_SIZE;
        let mut offset = header_size + table_size;
        let mut table = Vec::with_capacity(table_size);
        for section in &sections {
            table.extend_from_slice(&(offset as u64).to_le_bytes());
            table.extend_from_slice(&(section.len() as u64).to_le_bytes());
            offset += section.len();
        }

        let mut bytes = header;
        bytes.extend_from_slice(&table);
        for section in sections {
            bytes.extend_from_slice(&section);
        }
        bytes
    }

    /// Write `bytes`, open it, and hand back whatever the reader said.
    fn open_bytes(label: &str, bytes: Vec<u8>) -> Result<RaiModelFile> {
        let path = temporary_model_path(label);
        std::fs::write(&path, bytes).unwrap();
        let result = RaiModelFile::open(&path);
        let _ = std::fs::remove_file(&path);
        result
    }

    /// The reader's full error chain (`{:#}`): the interesting detail is often
    /// a `source`, not the outermost context.
    fn open_error(label: &str, bytes: Vec<u8>) -> String {
        format!(
            "{:#}",
            open_bytes(label, bytes)
                .err()
                .expect("the reader should have refused this file")
        )
    }

    #[test]
    fn a_v2_header_is_read_and_defaults_to_v1_behaviour() {
        let model = open_bytes("v2-defaults", build_model_bytes(2, 0, |_| {})).unwrap();
        assert_eq!(model.config.version, 2);
        assert_eq!(model.config.activation, Activation::Silu);
        assert_eq!(model.config.rope_scaling, RopeScaling::None);
        assert_eq!(model.config.bias_mask, 0);
        assert_eq!(model.config.embed_scale, 1.0);
        // The section table starts after the 128-byte header, not the 64-byte
        // one; getting that wrong would misparse every section.
        assert_eq!(model.sections[0].offset as usize, HEADER_SIZE_V2 + 3 * 16);
        for index in 0..NUM_PROJECTIONS {
            assert!(model.layer(0).unwrap().biases[index].is_none());
        }
    }

    #[test]
    fn v2_carries_activation_rope_and_embed_scale() {
        let model = open_bytes(
            "v2-caps",
            build_model_bytes(2, 0, |header| {
                header[64] = 1; // GeLU-tanh
                header[66] = 1; // llama3
                header[68..72].copy_from_slice(&32.0_f32.to_le_bytes());
                header[72..76].copy_from_slice(&1.0_f32.to_le_bytes());
                header[76..80].copy_from_slice(&4.0_f32.to_le_bytes());
                header[80..84].copy_from_slice(&8_192_u32.to_le_bytes());
                header[84..88].copy_from_slice(&45.254_834_f32.to_le_bytes());
            }),
        )
        .unwrap();
        assert_eq!(model.config.activation, Activation::GeluTanh);
        assert_eq!(
            model.config.rope_scaling,
            RopeScaling::Llama3 {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position: 8_192,
            }
        );
        assert_eq!(model.config.embed_scale, 45.254_834);
    }

    #[test]
    fn v2_biases_are_located_and_bounded() {
        // q, k and v — the Qwen2 mask.
        let model = open_bytes("v2-bias", build_model_bytes(2, 0b000_0111, |_| {})).unwrap();
        assert_eq!(model.config.bias_mask, 0b000_0111);
        let layer = model.layer(0).unwrap();
        for (index, name) in PROJECTION_NAMES.iter().enumerate() {
            assert_eq!(
                layer.biases[index].is_some(),
                index < 3,
                "{name} bias presence"
            );
        }
        let q_bias = read_f32_vector(layer.biases[0].unwrap());
        assert_eq!(q_bias, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
        // The two norm vectors must still land at the end of the section.
        assert_eq!(read_norm_weights(&layer.input_layernorm).len(), 8);
        assert_eq!(read_norm_weights(&layer.post_attn_layernorm).len(), 8);
    }

    #[test]
    fn a_v1_file_that_declares_biases_cannot_exist() {
        // A v1 header has no bias_mask, so a v1 layer section is sized without
        // one. This checks the reader does not read v2 fields out of a v1
        // header: byte 65/67 of a v1 file is section-table data.
        let model = open_bytes("v1-no-bias", build_model_bytes(1, 0, |_| {})).unwrap();
        assert_eq!(model.config.version, 1);
        assert_eq!(model.config.bias_mask, 0);
    }

    #[test]
    fn unknown_capabilities_are_refused_not_ignored() {
        for (label, patch, needle) in [
            (
                "activation",
                Box::new(|h: &mut [u8]| h[64] = 2) as Box<dyn Fn(&mut [u8])>,
                "unsupported activation code 2",
            ),
            (
                "rope",
                Box::new(|h: &mut [u8]| h[66] = 7),
                "unsupported rope_type 7",
            ),
            (
                "flags",
                Box::new(|h: &mut [u8]| h[65] = 0x02),
                "set bits this reader does not implement",
            ),
            (
                "bias-mask",
                Box::new(|h: &mut [u8]| {
                    h[65] = FLAG_HAS_BIASES;
                    h[67] = 0x80;
                }),
                "sets bits beyond the 7 projections",
            ),
            (
                "reserved-tail",
                Box::new(|h: &mut [u8]| h[120] = 1),
                "reserved and must be zero",
            ),
            (
                "reserved-v1-tail",
                Box::new(|h: &mut [u8]| h[60] = 1),
                "bytes 56..64 are reserved",
            ),
            (
                "flag-mask-disagreement",
                Box::new(|h: &mut [u8]| h[65] = FLAG_HAS_BIASES),
                "header is inconsistent",
            ),
            (
                "stowaway-rope-params",
                Box::new(|h: &mut [u8]| h[68..72].copy_from_slice(&8.0_f32.to_le_bytes())),
                "must leave the RoPE scaling parameters zero",
            ),
            (
                "bad-embed-scale",
                Box::new(|h: &mut [u8]| h[84..88].copy_from_slice(&0.0_f32.to_le_bytes())),
                "embed_scale must be finite and positive",
            ),
        ] {
            let error = open_error(label, build_model_bytes(2, 0, patch));
            assert!(
                error.contains(needle),
                "{label}: expected {needle:?}, got {error:?}"
            );
        }
    }

    #[test]
    fn a_future_version_is_refused() {
        let error = open_error("v3", build_model_bytes(3, 0, |_| {}));
        assert!(
            error.contains("unsupported format version: 3"),
            "got {error:?}"
        );
    }

    #[test]
    fn a_declared_bias_that_is_missing_is_caught() {
        // Declare a bias on down_proj but assemble the section without one:
        // the section then ends short of the two norm vectors.
        let bytes = build_model_bytes(2, 0, |header| {
            header[65] = FLAG_HAS_BIASES;
            header[67] = 0b100_0000;
        });
        let error = open_error("bias-missing", bytes);
        assert!(error.contains("section size mismatch"), "got {error:?}");
    }

    fn quant_params(groups: usize) -> Vec<u8> {
        let mut params = Vec::with_capacity(groups * 4);
        for _ in 0..groups {
            params.extend_from_slice(&f16::ONE.to_le_bytes());
            params.extend_from_slice(&f16::ZERO.to_le_bytes());
        }
        params
    }

    fn temporary_model_path(label: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rai-format-{label}-{}-{unique}.raimodel",
            std::process::id()
        ))
    }
}
