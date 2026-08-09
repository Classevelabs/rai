//! `.raimodel` binary format: validated in-memory reader for GPTQ-4bit models.
//!
//! File layout:
//!   [64 bytes]            Header (magic, version, model config)
//!   [num_sections * 16]   Section index table (offset, size per section)
//!   [variable]            Section 0: Embedding (8-bit quantized)
//!   [variable]            Sections 1..N: Transformer layers (4-bit quantized linears + f32 norms)
//!   [variable]            Section N+1: Final RMSNorm (f32 weights)

use anyhow::{bail, Context, Result};
use half::f16;
use std::io::Read;
use std::path::Path;

/// Magic bytes: "RAIM"
const MAGIC: [u8; 4] = *b"RAIM";
const FORMAT_VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
const SECTION_ENTRY_SIZE: usize = 16;
const MAX_MODEL_FILE_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_HIDDEN_SIZE: u32 = 65_536;
const MAX_INTERMEDIATE_SIZE: u32 = 1_048_576;
const MAX_LAYERS: u32 = 1_024;
const MAX_HEADS: u32 = 1_024;
const MAX_VOCAB_SIZE: u32 = 10_000_000;
const MAX_CONTEXT: u32 = 1_000_000;
const MAX_GEMM_GROUPS: usize = 128;
const MAX_ROPE_TABLE_BYTES: usize = 512 * 1024 * 1024;

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
        validate_config(&config)?;

        let num_sections = read_u32(&data, 52) as usize;
        let tied_sections = checked_add(config.num_layers as usize, 2, "section count")?;
        let untied_sections = checked_add(config.num_layers as usize, 3, "section count")?;
        if num_sections != tied_sections && num_sections != untied_sections {
            bail!(
                "invalid section count {num_sections}; expected {tied_sections} (tied) or {untied_sections} (untied)"
            );
        }

        // Parse section index table (right after header)
        let table_start = HEADER_SIZE;
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

    let projected_hidden = checked_mul(
        config.num_heads as usize,
        config.head_dim as usize,
        "attention width",
    )?;
    if projected_hidden != config.hidden_size as usize {
        bail!(
            "hidden_size {} does not equal num_heads * head_dim ({projected_hidden})",
            config.hidden_size
        );
    }

    let group_size = config.group_size as usize;
    let max_linear_groups = (config.hidden_size as usize)
        .div_ceil(group_size)
        .max((config.intermediate_size as usize).div_ceil(group_size));
    if max_linear_groups > MAX_GEMM_GROUPS {
        bail!(
            "model requires {max_linear_groups} quantization groups; kernel maximum is {MAX_GEMM_GROUPS}"
        );
    }
    let embedding_groups = (config.hidden_size as usize).div_ceil(config.embed_group_size as usize);
    if embedding_groups > MAX_GEMM_GROUPS {
        bail!(
            "embedding requires {embedding_groups} quantization groups; kernel maximum is {MAX_GEMM_GROUPS}"
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
    for (index, bytes) in norm.weights.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(bytes.try_into().unwrap());
        if !value.is_finite() {
            bail!("{label} weight {index} is non-finite");
        }
    }
    Ok(())
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
        }
    }

    #[test]
    fn test_config_size() {
        // Header must be exactly 64 bytes
        assert_eq!(HEADER_SIZE, 64);
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
        let mut config = valid_config();
        config.hidden_size = 258;
        assert!(validate_config(&config).is_err());
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
        bytes[HEADER_SIZE..HEADER_SIZE + 8].copy_from_slice(&0_u64.to_le_bytes());
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
        let embedding_start = HEADER_SIZE + 3 * SECTION_ENTRY_SIZE;
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
        let embedding_start = HEADER_SIZE + 3 * SECTION_ENTRY_SIZE;
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
        let table_start = HEADER_SIZE;
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
        let mut header = vec![0_u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
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

        let mut embedding = quant_params(2);
        embedding.extend_from_slice(&[0_u8; 16]);
        let mut layer = Vec::new();
        for _ in 0..7 {
            layer.extend_from_slice(&8_u32.to_le_bytes());
            layer.extend_from_slice(&8_u32.to_le_bytes());
            layer.extend_from_slice(&quant_params(8));
            layer.extend_from_slice(&[0_u8; 32]);
        }
        layer.extend_from_slice(&[0_u8; 64]);
        let final_norm = vec![0_u8; 32];
        let sections = [embedding, layer, final_norm];

        let table_size = sections.len() * SECTION_ENTRY_SIZE;
        let mut offset = HEADER_SIZE + table_size;
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
