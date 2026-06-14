//! `.raimodel` binary format: zero-copy mmap reader for GPTQ-4bit models.
//!
//! File layout:
//!   [64 bytes]            Header (magic, version, model config)
//!   [num_sections * 16]   Section index table (offset, size per section)
//!   [variable]            Section 0: Embedding (8-bit quantized)
//!   [variable]            Sections 1..N: Transformer layers (4-bit quantized linears + f32 norms)
//!   [variable]            Section N+1: Final RMSNorm (f32 weights)

use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::Path;

/// Magic bytes: "RAIM"
const MAGIC: [u8; 4] = [b'R', b'A', b'I', b'M'];
const FORMAT_VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
const SECTION_ENTRY_SIZE: usize = 16;

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
}

/// A section in the file (offset + size).
#[derive(Debug, Clone, Copy)]
pub struct SectionEntry {
    pub offset: u64,
    pub size: u64,
}

/// Zero-copy reference to a quantized linear layer in the mmap.
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

/// A transformer layer's references into the mmap.
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
}

/// The full model file: header + heap-allocated data with huge page hints.
pub struct RaiModelFile {
    pub config: ModelConfig,
    pub sections: Vec<SectionEntry>,
    data: Vec<u8>,
}

impl RaiModelFile {
    /// Open and validate a .raimodel file.
    /// Reads the entire file into a heap-allocated buffer for maximum memory bandwidth.
    /// Uses anonymous mmap with MADV_HUGEPAGE for transparent huge page backing,
    /// reducing TLB pressure on the ~85MB weight data.
    pub fn open(path: &Path) -> Result<Self> {
        let file_len = std::fs::metadata(path)
            .with_context(|| format!("stat {}", path.display()))?
            .len() as usize;

        // Allocate buffer and request huge pages BEFORE populating, so the kernel
        // can allocate 2MB pages on first write rather than collapsing 4KB pages later.
        let mut data = Vec::with_capacity(file_len);

        #[cfg(target_os = "linux")]
        {
            extern "C" {
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            unsafe {
                let ptr = data.as_ptr() as *mut std::ffi::c_void;
                madvise(ptr, file_len, 14); // MADV_HUGEPAGE — before faulting pages
            }
        }

        let mut file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        file.read_to_end(&mut data).context("reading model file")?;
        drop(file);

        if data.len() < HEADER_SIZE {
            bail!("file too small for header: {} bytes", data.len());
        }

        // Parse header
        if data[0..4] != MAGIC {
            bail!("invalid magic bytes");
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            bail!("unsupported format version: {version}");
        }

        let config = ModelConfig {
            hidden_size: read_u32(&data, 8),
            num_layers: read_u32(&data, 12),
            num_heads: read_u32(&data, 16),
            num_kv_heads: read_u32(&data, 20),
            head_dim: read_u32(&data, 24),
            intermediate_size: read_u32(&data, 28),
            vocab_size: read_u32(&data, 32),
            max_context: read_u32(&data, 36),
            rope_theta: f32::from_le_bytes(data[40..44].try_into().unwrap()),
            norm_eps: f32::from_le_bytes(data[44..48].try_into().unwrap()),
            bits: data[48],
            group_size: data[49],
            embed_bits: data[50],
            embed_group_size: data[51],
        };

        let num_sections = read_u32(&data, 52) as usize;

        // Parse section index table (right after header)
        let table_start = HEADER_SIZE;
        let table_end = table_start + num_sections * SECTION_ENTRY_SIZE;
        if data.len() < table_end {
            bail!("file too small for section table");
        }

        let mut sections = Vec::with_capacity(num_sections);
        for i in 0..num_sections {
            let off = table_start + i * SECTION_ENTRY_SIZE;
            let offset = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            let size = u64::from_le_bytes(data[off + 8..off + 16].try_into().unwrap());
            sections.push(SectionEntry { offset, size });
        }

        Ok(Self {
            config,
            sections,
            data,
        })
    }

    /// Get the raw bytes for a section.
    fn section_data(&self, idx: usize) -> Result<&[u8]> {
        let s = self
            .sections
            .get(idx)
            .context("section index out of range")?;
        let start = s.offset as usize;
        let end = start + s.size as usize;
        if end > self.data.len() {
            bail!("section {idx} extends beyond file");
        }
        Ok(&self.data[start..end])
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
        let params_size = vocab * num_groups * 4;
        let data_size = vocab * hidden;

        if data.len() < params_size + data_size {
            bail!(
                "embedding section too small: {} < {} + {}",
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
        let section_idx = 1 + layer_idx;
        let data = self.section_data(section_idx)?;
        let gs = self.config.group_size as usize;
        let hidden = self.config.hidden_size as usize;

        // Each linear sub-section: [u32 rows][u32 cols][group_params][nibble_data]
        let linear_dims: [(usize, usize); 7] = [
            (hidden, hidden), // q_proj
            (
                self.config.num_kv_heads as usize * self.config.head_dim as usize,
                hidden,
            ), // k_proj
            (
                self.config.num_kv_heads as usize * self.config.head_dim as usize,
                hidden,
            ), // v_proj
            (hidden, hidden), // o_proj
            (self.config.intermediate_size as usize, hidden), // gate_proj
            (self.config.intermediate_size as usize, hidden), // up_proj
            (hidden, self.config.intermediate_size as usize), // down_proj
        ];

        let mut offset = 0usize;
        let mut linears: Vec<QuantizedLinear<'_>> = Vec::with_capacity(7);

        for &(rows, cols) in &linear_dims {
            // Sub-header: 8 bytes (rows: u32, cols: u32)
            if offset + 8 > data.len() {
                bail!("layer {layer_idx}: truncated linear sub-header at offset {offset}");
            }
            let r = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            let c = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
            if r != rows || c != cols {
                bail!("layer {layer_idx}: dimension mismatch: expected ({rows},{cols}), got ({r},{c})");
            }
            offset += 8;

            let num_groups = cols.div_ceil(gs);
            let params_size = rows * num_groups * 4;
            let nibble_size = rows * cols / 2;

            if offset + params_size + nibble_size > data.len() {
                bail!("layer {layer_idx}: linear data truncated");
            }

            linears.push(QuantizedLinear {
                rows,
                cols,
                group_params: &data[offset..offset + params_size],
                nibble_data: &data[offset + params_size..offset + params_size + nibble_size],
                group_size: gs,
            });
            offset += params_size + nibble_size;
        }

        // Two RMSNorm weight vectors: hidden_size * 4 bytes each
        let norm_bytes = hidden * 4;
        if offset + 2 * norm_bytes > data.len() {
            bail!("layer {layer_idx}: norm weights truncated");
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
        })
    }

    /// Parse the final RMSNorm section.
    /// With untied lm_head: embed + layers + norm + lm_head = num_layers + 3 sections.
    /// With tied lm_head:   embed + layers + norm           = num_layers + 2 sections.
    pub fn final_norm(&self) -> Result<RMSNormWeights<'_>> {
        let norm_idx = 1 + self.config.num_layers as usize; // section after all layers
        let data = self.section_data(norm_idx)?;
        let hidden = self.config.hidden_size as usize;
        if data.len() < hidden * 4 {
            bail!("final norm section too small");
        }
        Ok(RMSNormWeights {
            weights: &data[..hidden * 4],
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
        let params_size = rows * num_groups * 4;
        let nibble_size = rows * cols / 2;
        let offset = 8;

        if data.len() < offset + params_size + nibble_size {
            bail!("lm_head section data truncated");
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

    #[test]
    fn test_config_size() {
        // Header must be exactly 64 bytes
        assert_eq!(HEADER_SIZE, 64);
    }

    #[test]
    fn test_section_entry_size() {
        assert_eq!(SECTION_ENTRY_SIZE, 16);
    }
}
