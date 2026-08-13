//! Rust-native HuggingFace -> `.raimodel` converter (round-to-nearest 4-bit).
//!
//! This is the Rust port of `scripts/export_rtn.py` + `scripts/raimodel.py`.
//! It exists because the Python path contradicts the product's own claim of
//! "no GPU, no Python runtime": converting a model used to require torch and
//! transformers (~2.5 GB of wheels), and the exporter materialized the whole
//! model in RAM, so a 7B checkpoint could not be converted on a 16 GB machine
//! at all.
//!
//! Two properties are load-bearing:
//!
//! * **Byte-identical output.** For the same checkpoint and the same options,
//!   this writes exactly the bytes `export_rtn.py` writes. Everything that
//!   affects a byte is mirrored deliberately: the f64 group statistics, the
//!   f16 round-trip of scale/zero *before* codes are derived, round-half-to-
//!   even code rounding (numpy's `np.round`, not Rust's `f64::round`), the
//!   `MIN_F16_SCALE` floor, low-nibble-first packing, and the cast of every
//!   weight through f16 on load (see `safetensors::decode_into`).
//! * **Bounded memory.** Sections have analytically known sizes, so the file
//!   is written in place: for each matrix the code seeks past the group-param
//!   block, streams row blocks of codes, then seeks back and fills the params
//!   in. Peak memory is one row block plus one matrix's group params, not one
//!   layer and never the model.
//!
//! Binary layout is documented in `format.rs` (the reader) and in the
//! `raimodel.py` module docstring.

use anyhow::{bail, Context, Result};
use half::f16;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::safetensors::SafeTensorsSet;

/// Smallest positive (subnormal) f16; scales are clamped to at least this so a
/// stored scale is always strictly positive. Matches
/// `float(np.nextafter(np.float16(0), np.float16(1)))`.
const MIN_F16_SCALE: f64 = 5.960464477539063e-8;

const HEADER_SIZE: u64 = 64;
const SECTION_ENTRY_SIZE: u64 = 16;
const FORMAT_VERSION: u32 = 1;

// Reader limits, mirrored so a bad export fails before any work is done.
const MAX_HIDDEN_SIZE: u32 = 65_536;
const MAX_INTERMEDIATE_SIZE: u32 = 1_048_576;
const MAX_LAYERS: u32 = 1_024;
const MAX_HEADS: u32 = 1_024;
const MAX_VOCAB_SIZE: u32 = 10_000_000;
const MAX_CONTEXT: u32 = 1_000_000;
const MAX_GEMM_GROUPS: usize = 128;
const MAX_ROPE_TABLE_BYTES: u64 = 512 * 1024 * 1024;

/// Weights read per streaming block, in elements (~8 MB as f32).
const BLOCK_ELEMENTS: usize = 2 << 20;

/// The seven quantized projections in every layer, in section order.
const LAYER_LINEAR_NAMES: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

/// Architectures whose maths this container cannot express even though their
/// tensor names look Llama-shaped.
const UNSUPPORTED_MODEL_TYPES: [(&str, &str); 4] = [
    (
        "gemma",
        "Gemma scales embeddings by sqrt(hidden) and its RMSNorm applies (1 + weight)",
    ),
    (
        "gemma2",
        "Gemma2 adds logit softcapping and (1 + weight) RMSNorm",
    ),
    (
        "gemma3",
        "Gemma3 adds per-head QK norm and (1 + weight) RMSNorm",
    ),
    (
        "gemma3_text",
        "Gemma3 adds per-head QK norm and (1 + weight) RMSNorm",
    ),
];

/// Conversion inputs. Defaults match `export_rtn.py`.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// HuggingFace checkpoint directory.
    pub model_dir: PathBuf,
    /// Output path; defaults to `<dirname lowercased>-q4.raimodel` in the
    /// working directory.
    pub output: Option<PathBuf>,
    pub group_size: u32,
    pub embed_group_size: u32,
    pub max_context: u32,
    /// Where `tokenizer.json` is copied; defaults to next to the output.
    pub tokenizer_out: Option<PathBuf>,
    /// Suppress progress output.
    pub quiet: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            output: None,
            group_size: 128,
            embed_group_size: 64,
            max_context: 2048,
            tokenizer_out: None,
            quiet: false,
        }
    }
}

/// What a conversion produced.
#[derive(Debug, Clone)]
pub struct ConvertSummary {
    pub output_path: PathBuf,
    pub bytes_written: u64,
    pub num_sections: usize,
    pub tokenizer_path: PathBuf,
    /// False when an identical `tokenizer.json` was already in place.
    pub tokenizer_copied: bool,
    pub elapsed: Duration,
}

/// The header fields, i.e. everything the reader validates before touching a
/// section.
#[derive(Debug, Clone)]
struct RaiConfig {
    hidden_size: u32,
    num_layers: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    vocab_size: u32,
    max_context: u32,
    rope_theta: f32,
    norm_eps: f32,
    bits: u8,
    group_size: u8,
    embed_bits: u8,
    embed_group_size: u8,
}

/// Convert a HuggingFace checkpoint directory to `.raimodel`.
pub fn convert(options: &ConvertOptions) -> Result<ConvertSummary> {
    let started = Instant::now();
    validate_options(options)?;

    let config_path = options.model_dir.join("config.json");
    let raw_config = std::fs::read(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let hf: serde_json::Value = serde_json::from_slice(&raw_config)
        .with_context(|| format!("parsing {}", config_path.display()))?;

    let hidden_size = required_u32(&hf, "hidden_size")?;
    let num_layers = required_u32(&hf, "num_hidden_layers")?;
    let num_heads = required_u32(&hf, "num_attention_heads")?;
    let num_kv_heads = optional_u32(&hf, "num_key_value_heads")?.unwrap_or(num_heads);
    let head_dim = resolve_head_dim(optional_u32(&hf, "head_dim")?, hidden_size, num_heads)?;
    let intermediate_size = required_u32(&hf, "intermediate_size")?;
    let vocab_size = required_u32(&hf, "vocab_size")?;
    let rope_theta = optional_f64(&hf, "rope_theta")?.unwrap_or(10_000.0) as f32;
    let norm_eps = optional_f64(&hf, "rms_norm_eps")?.unwrap_or(1e-5) as f32;
    let tied = hf
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let model_type = hf.get("model_type").and_then(|v| v.as_str()).unwrap_or("?");

    let config = RaiConfig {
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_context: options.max_context,
        rope_theta,
        norm_eps,
        bits: 4,
        group_size: options.group_size as u8,
        embed_bits: 8,
        embed_group_size: options.embed_group_size as u8,
    };
    validate_model_config(&config)?;

    let output_path = match &options.output {
        Some(path) => path.clone(),
        None => PathBuf::from(default_output_name(&options.model_dir)?),
    };
    let tokenizer_src = options.model_dir.join("tokenizer.json");
    if !tokenizer_src.is_file() {
        bail!(
            "{} has no tokenizer.json; the runtime needs it next to the model, so there is \
             nothing to convert into a usable model",
            options.model_dir.display()
        );
    }
    let tokenizer_dst = match &options.tokenizer_out {
        Some(path) => path.clone(),
        None => output_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tokenizer.json"),
    };

    let log = |line: &str| {
        if !options.quiet {
            println!("{line}");
            let _ = std::io::stdout().flush();
        }
    };

    log(&format!("Output: {}", output_path.display()));
    log(&format!("Reading {}...", options.model_dir.display()));
    let mut store = SafeTensorsSet::open(&options.model_dir)?;
    assert_exportable_architecture(&hf, &store, options.max_context, num_layers)?;

    let kv_dim = (num_kv_heads * head_dim) as usize;
    let hidden = hidden_size as usize;
    let inter = intermediate_size as usize;
    let vocab = vocab_size as usize;
    let group_size = options.group_size as usize;
    let embed_group_size = options.embed_group_size as usize;
    let linear_dims: [(usize, usize); 7] = [
        (hidden, hidden),
        (kv_dim, hidden),
        (kv_dim, hidden),
        (hidden, hidden),
        (inter, hidden),
        (inter, hidden),
        (hidden, inter),
    ];

    // Confirm every tensor exists with the shape the reader will demand, before
    // a single byte is written.
    check_tensor(&store, "model.embed_tokens.weight", vocab, hidden)?;
    for layer in 0..num_layers {
        for (name, (rows, cols)) in LAYER_LINEAR_NAMES.iter().zip(linear_dims) {
            check_tensor(&store, &layer_linear_name(layer, name), rows, cols)?;
        }
        for suffix in ["input_layernorm", "post_attention_layernorm"] {
            check_vector(
                &store,
                &format!("model.layers.{layer}.{suffix}.weight"),
                hidden,
            )?;
        }
    }
    check_vector(&store, "model.norm.weight", hidden)?;
    if !tied {
        check_tensor(&store, "lm_head.weight", vocab, hidden)?;
    }

    log(&format!(
        "{model_type}: {num_layers}L h={hidden_size} inter={intermediate_size} \
         heads={num_heads}/{num_kv_heads} vocab={vocab_size}"
    ));

    // ---- Plan the container ------------------------------------------------
    // Every section size follows from the config, so offsets can be written
    // before the data exists and each section can be streamed straight out.
    let mut section_sizes: Vec<u64> = Vec::with_capacity(num_layers as usize + 3);
    section_sizes.push(embedding_section_len(vocab, hidden, embed_group_size)?);
    let layer_len = layer_section_len(&linear_dims, hidden, group_size)?;
    for _ in 0..num_layers {
        section_sizes.push(layer_len);
    }
    section_sizes.push(hidden as u64 * 4);
    if !tied {
        section_sizes.push(linear_section_len(vocab, hidden, group_size)?);
    }

    let num_sections = section_sizes.len();
    let data_start = HEADER_SIZE + num_sections as u64 * SECTION_ENTRY_SIZE;
    let mut offsets = Vec::with_capacity(num_sections);
    let mut cursor = data_start;
    for size in &section_sizes {
        offsets.push(cursor);
        cursor += size;
    }
    let total_size = cursor;

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let mut file = File::create(&output_path)
        .with_context(|| format!("creating {}", output_path.display()))?;
    write_header(&mut file, &config, num_sections as u32)?;
    for (offset, size) in offsets.iter().zip(&section_sizes) {
        file.write_all(&offset.to_le_bytes())?;
        file.write_all(&size.to_le_bytes())?;
    }

    let quant_started = Instant::now();

    // ---- Section 0: embedding (8-bit) --------------------------------------
    log("\n=== EMBEDDING 8-BIT ===");
    let t0 = Instant::now();
    let mse = write_matrix(
        &mut file,
        &mut store,
        &MatrixJob {
            tensor: "model.embed_tokens.weight",
            label: "embedding",
            rows: vocab,
            cols: hidden,
            group_size: embed_group_size,
            bits: 8,
            emit_dims: false,
        },
    )?;
    log(&format!(
        "  Embedding [{vocab}x{hidden}] mse={}: {:.1}s",
        fmt_sci(mse),
        t0.elapsed().as_secs_f64()
    ));

    // ---- Sections 1..=L: layers --------------------------------------------
    log("\n=== RTN-4BIT QUANTIZATION ===");
    for layer in 0..num_layers {
        let t0 = Instant::now();
        for (name, (rows, cols)) in LAYER_LINEAR_NAMES.iter().zip(linear_dims) {
            let label = format!("L{layer}.{name}");
            let mse = write_matrix(
                &mut file,
                &mut store,
                &MatrixJob {
                    tensor: &layer_linear_name(layer, name),
                    label: &label,
                    rows,
                    cols,
                    group_size,
                    bits: 4,
                    emit_dims: true,
                },
            )?;
            if *name == "q_proj" || *name == "down_proj" {
                log(&format!(
                    "  L{layer}.{name}: [{rows}x{cols}] mse={}",
                    fmt_sci(mse)
                ));
            }
        }
        for suffix in ["input_layernorm", "post_attention_layernorm"] {
            let tensor = format!("model.layers.{layer}.{suffix}.weight");
            write_norm(&mut file, &mut store, &tensor, hidden)?;
        }
        log(&format!(
            "  Layer {layer}/{num_layers}: {:.1}s",
            t0.elapsed().as_secs_f64()
        ));
    }

    // ---- Section L+1: final norm -------------------------------------------
    write_norm(&mut file, &mut store, "model.norm.weight", hidden)?;

    // ---- Section L+2 (untied only): lm_head --------------------------------
    if tied {
        log("\n  lm_head: tied to embedding");
    } else {
        log("\n=== LM_HEAD 4-BIT (untied) ===");
        let t0 = Instant::now();
        let mse = write_matrix(
            &mut file,
            &mut store,
            &MatrixJob {
                tensor: "lm_head.weight",
                label: "lm_head",
                rows: vocab,
                cols: hidden,
                group_size,
                bits: 4,
                emit_dims: true,
            },
        )?;
        log(&format!(
            "  lm_head [{vocab}x{hidden}] mse={}: {:.1}s",
            fmt_sci(mse),
            t0.elapsed().as_secs_f64()
        ));
    }

    let written = file.stream_position()?;
    if written != total_size {
        bail!("internal error: wrote {written} bytes, planned {total_size}");
    }
    file.flush()?;
    drop(file);

    let quant_elapsed = quant_started.elapsed().as_secs_f64();
    log(&format!(
        "\nQuantization: {quant_elapsed:.1}s ({:.1} min)",
        quant_elapsed / 60.0
    ));
    log(&format!(
        "\nWrote: {} ({:.1} MB, {num_sections} sections)",
        output_path.display(),
        total_size as f64 / 1e6
    ));

    let tokenizer_copied = copy_tokenizer_json(&tokenizer_src, &tokenizer_dst)?;
    if tokenizer_copied {
        log(&format!("Tokenizer: {}", tokenizer_dst.display()));
    } else {
        log(&format!(
            "Tokenizer already present (identical): {}",
            tokenizer_dst.display()
        ));
    }

    let elapsed = started.elapsed();
    log(&format!(
        "\n=== DONE in {:.1}s ({:.1} min) ===",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / 60.0
    ));

    Ok(ConvertSummary {
        output_path,
        bytes_written: total_size,
        num_sections,
        tokenizer_path: tokenizer_dst,
        tokenizer_copied,
        elapsed,
    })
}

// =============================================================================
// Validation
// =============================================================================

fn validate_options(options: &ConvertOptions) -> Result<()> {
    for (flag, value) in [
        ("--group-size", options.group_size),
        ("--embed-group-size", options.embed_group_size),
    ] {
        if !(2..=254).contains(&value) || !value.is_multiple_of(2) {
            bail!("{flag} must be an even integer in 2..=254, got {value}");
        }
    }
    if options.max_context == 0 || options.max_context > MAX_CONTEXT {
        bail!(
            "--max-context must be in 1..={MAX_CONTEXT}, got {}",
            options.max_context
        );
    }
    if !options.model_dir.is_dir() {
        bail!("{} is not a directory", options.model_dir.display());
    }
    Ok(())
}

/// Mirror of `raimodel.validate_model_config` / the reader's `validate_config`.
fn validate_model_config(config: &RaiConfig) -> Result<()> {
    let bounded = [
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
    for (name, value, maximum) in bounded {
        if value == 0 || value > maximum {
            bail!("invalid {name}: {value}; the .raimodel reader requires 1..={maximum}");
        }
    }
    if !config.hidden_size.is_multiple_of(2) || !config.intermediate_size.is_multiple_of(2) {
        bail!("hidden and intermediate dimensions must be even for packed 4-bit kernels");
    }
    if !config.head_dim.is_multiple_of(8) {
        bail!(
            "head_dim must be a multiple of 8 for SIMD attention kernels, got {}",
            config.head_dim
        );
    }
    if !config.num_heads.is_multiple_of(config.num_kv_heads) {
        bail!(
            "num_heads ({}) must be divisible by num_kv_heads ({})",
            config.num_heads,
            config.num_kv_heads
        );
    }
    let projected = config.num_heads as u64 * config.head_dim as u64;
    if projected != config.hidden_size as u64 {
        bail!(
            "hidden_size {} does not equal num_heads * head_dim ({projected})",
            config.hidden_size
        );
    }

    let group_size = config.group_size as usize;
    let max_linear_groups = (config.hidden_size as usize)
        .div_ceil(group_size)
        .max((config.intermediate_size as usize).div_ceil(group_size));
    if max_linear_groups > MAX_GEMM_GROUPS {
        bail!(
            "group_size {group_size} needs {max_linear_groups} quantization groups for \
             hidden={}/intermediate={}; the reader's kernel maximum is {MAX_GEMM_GROUPS}. \
             Use a larger --group-size.",
            config.hidden_size,
            config.intermediate_size
        );
    }
    let embedding_groups = (config.hidden_size as usize).div_ceil(config.embed_group_size as usize);
    if embedding_groups > MAX_GEMM_GROUPS {
        bail!(
            "embed_group_size {} needs {embedding_groups} embedding groups for hidden={}; \
             the reader's kernel maximum is {MAX_GEMM_GROUPS}. Use a larger \
             --embed-group-size.",
            config.embed_group_size,
            config.hidden_size
        );
    }

    for (name, value) in [
        ("rope_theta", config.rope_theta),
        ("norm_eps", config.norm_eps),
    ] {
        if !value.is_finite() || value <= 0.0 {
            bail!("{name} must be finite and positive, got {value}");
        }
    }

    let rope_bytes = config.max_context as u64 * (config.head_dim as u64 / 2) * 2 * 4;
    if rope_bytes > MAX_ROPE_TABLE_BYTES {
        bail!(
            "RoPE table would need {rope_bytes} bytes (max_context={}, head_dim={}); the \
             reader's maximum is {MAX_ROPE_TABLE_BYTES}. Lower --max-context.",
            config.max_context,
            config.head_dim
        );
    }
    Ok(())
}

/// Mirror of `raimodel.resolve_head_dim`.
fn resolve_head_dim(config_head_dim: Option<u32>, hidden_size: u32, num_heads: u32) -> Result<u32> {
    if num_heads == 0 {
        bail!("num_attention_heads must be greater than zero");
    }
    match config_head_dim {
        Some(head_dim) if head_dim != 0 => {
            if num_heads as u64 * head_dim as u64 != hidden_size as u64 {
                bail!(
                    "model config declares head_dim={head_dim} with num_heads={num_heads}, so \
                     num_heads * head_dim = {} != hidden_size {hidden_size}. The .raimodel \
                     format cannot represent a decoupled head_dim yet; this model cannot be \
                     exported.",
                    num_heads as u64 * head_dim as u64
                );
            }
            Ok(head_dim)
        }
        _ => Ok(hidden_size / num_heads),
    }
}

/// Mirror of `raimodel.assert_exportable_architecture`, driven by the config
/// and the checkpoint's tensor names instead of a live torch module tree.
fn assert_exportable_architecture(
    hf: &serde_json::Value,
    store: &SafeTensorsSet,
    max_context: u32,
    num_layers: u32,
) -> Result<()> {
    let mut problems: Vec<String> = Vec::new();

    let model_type = hf.get("model_type").and_then(|v| v.as_str());
    if let Some(model_type) = model_type {
        if let Some((_, reason)) = UNSUPPORTED_MODEL_TYPES
            .iter()
            .find(|(name, _)| *name == model_type)
        {
            problems.push(format!(
                "model_type '{model_type}' is not supported: {reason}, which this format does \
                 not store."
            ));
        }
    }

    if store
        .info("model.layers.0.self_attn.q_proj.weight")
        .is_none()
        && num_layers > 0
    {
        bail!(
            "this checkpoint does not expose model.layers.0.self_attn.q_proj.weight; the \
             converter supports Llama-style causal LMs (LlamaForCausalLM, MistralForCausalLM, \
             and architecturally identical models)."
        );
    }

    let is_projection_bias = |name: &str| {
        name.starts_with("model.layers.")
            && name.ends_with("_proj.bias")
            && LAYER_LINEAR_NAMES
                .iter()
                .any(|proj| name.ends_with(&format!(".{proj}.bias")))
    };
    let biased = store.count_names(is_projection_bias);
    if biased > 0 {
        let example = store.any_name(is_projection_bias).unwrap_or("");
        problems.push(format!(
            "{biased} projection(s) carry bias vectors (e.g. {example}); the format stores \
             weights only, so the biases would be silently dropped. Qwen2/Qwen2.5 are the \
             common case here."
        ));
    }

    let is_qk_norm =
        |name: &str| name.contains(".self_attn.q_norm") || name.contains(".self_attn.k_norm");
    let qk_normed = store.count_names(is_qk_norm);
    if qk_normed > 0 {
        let example = store.any_name(is_qk_norm).unwrap_or("");
        problems.push(format!(
            "{qk_normed} per-head QK norm(s) present (e.g. {example}); the format has no place \
             to store them."
        ));
    }

    // transformers >= 5 normalizes plain RoPE into {"rope_type": "default"}, so
    // the field's presence means nothing on its own.
    if let Some(scaling) = hf.get("rope_scaling") {
        if !scaling.is_null() {
            let rope_type = if let Some(object) = scaling.as_object() {
                object
                    .get("rope_type")
                    .or_else(|| object.get("type"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            } else {
                Some(scaling.to_string())
            };
            match rope_type.as_deref() {
                None | Some("default") => {}
                Some(other) => problems.push(format!(
                    "config declares rope_scaling type '{other}'; the reader builds a plain RoPE \
                     table from rope_theta alone, so positions would be wrong. Llama-3.1/3.2 \
                     (rope_type 'llama3') are the common case here."
                )),
            }
        }
    }

    for attr in ["num_experts", "num_local_experts"] {
        if let Some(value) = hf.get(attr) {
            if truthy_number(value) {
                problems.push(format!(
                    "config declares {attr}={value}; mixture-of-experts routing is not supported."
                ));
                break;
            }
        }
    }

    for attr in ["attn_logit_softcapping", "final_logit_softcapping"] {
        if let Some(value) = hf.get(attr) {
            if truthy_number(value) {
                problems.push(format!(
                    "config declares {attr}; logit softcapping is not supported."
                ));
            }
        }
    }

    let sliding_window = hf
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let uses_sliding = hf
        .get("use_sliding_window")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    if sliding_window > 0 && uses_sliding && max_context as u64 > sliding_window {
        problems.push(format!(
            "config declares sliding_window={sliding_window} but --max-context is {max_context}; \
             the reader always uses full causal attention, so exports beyond the window would \
             diverge. Re-run with --max-context {sliding_window} or lower."
        ));
    }

    if !problems.is_empty() {
        bail!(
            "this checkpoint cannot be represented by the .raimodel format:\n  - {}",
            problems.join("\n  - ")
        );
    }
    Ok(())
}

fn truthy_number(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(flag) => *flag,
        serde_json::Value::Number(number) => number.as_f64().is_some_and(|v| v != 0.0),
        _ => true,
    }
}

fn check_tensor(store: &SafeTensorsSet, name: &str, rows: usize, cols: usize) -> Result<()> {
    let info = store.require(name)?;
    let (actual_rows, actual_cols) = info.dims_2d(name)?;
    if (actual_rows, actual_cols) != (rows, cols) {
        bail!(
            "tensor '{name}' has shape [{actual_rows}, {actual_cols}] but the config implies \
             [{rows}, {cols}]; the .raimodel reader would reject this model"
        );
    }
    Ok(())
}

fn check_vector(store: &SafeTensorsSet, name: &str, len: usize) -> Result<()> {
    let info = store.require(name)?;
    if info.shape.as_slice() != [len] {
        bail!(
            "tensor '{name}' has shape {:?} but the config implies [{len}]",
            info.shape
        );
    }
    Ok(())
}

// =============================================================================
// Quantization
// =============================================================================

struct MatrixJob<'a> {
    tensor: &'a str,
    label: &'a str,
    rows: usize,
    cols: usize,
    group_size: usize,
    bits: u32,
    /// Linears carry an `[u32 rows][u32 cols]` sub-header; the embedding does not.
    emit_dims: bool,
}

/// Quantize one matrix straight into the output file and return its MSE.
///
/// The group-param block precedes the codes in the section but is only known
/// once every row has been seen, so the codes are streamed into their final
/// position first and the params are written afterwards with one seek back.
fn write_matrix(file: &mut File, store: &mut SafeTensorsSet, job: &MatrixJob<'_>) -> Result<f64> {
    let (rows, cols, group_size, bits) = (job.rows, job.cols, job.group_size, job.bits);
    if bits == 4 && !cols.is_multiple_of(2) {
        bail!(
            "{}: cannot pack nibbles, column count must be even, got {cols}",
            job.label
        );
    }
    let n_levels = 1u32 << bits;
    let num_groups = cols.div_ceil(group_size);
    let row_param_bytes = num_groups * 4;
    let row_code_bytes = if bits == 4 { cols / 2 } else { cols };

    if job.emit_dims {
        file.write_all(&(rows as u32).to_le_bytes())?;
        file.write_all(&(cols as u32).to_le_bytes())?;
    }

    let params_pos = file.stream_position()?;
    let params_len = (rows * row_param_bytes) as u64;
    let mut params = vec![0u8; rows * row_param_bytes];
    file.seek(SeekFrom::Start(params_pos + params_len))?;

    let rows_per_block = (BLOCK_ELEMENTS / cols.max(1)).clamp(1, rows.max(1));
    let mut weights: Vec<f32> = Vec::new();
    let mut codes: Vec<u8> = Vec::new();
    let mut squared_error = 0.0f64;

    let mut row_start = 0usize;
    while row_start < rows {
        let block_rows = rows_per_block.min(rows - row_start);
        store.read_rows(job.tensor, row_start, block_rows, &mut weights)?;
        codes.clear();
        codes.resize(block_rows * row_code_bytes, 0);
        let block_params =
            &mut params[row_start * row_param_bytes..(row_start + block_rows) * row_param_bytes];

        let errors: Vec<f64> = weights
            .par_chunks(cols)
            .zip(codes.par_chunks_mut(row_code_bytes))
            .zip(block_params.par_chunks_mut(row_param_bytes))
            .enumerate()
            .map(|(index, ((row, row_codes), row_params))| {
                quantize_row(
                    row,
                    group_size,
                    n_levels,
                    bits == 4,
                    row_params,
                    row_codes,
                    job.label,
                    row_start + index,
                )
            })
            .collect::<Result<Vec<f64>>>()?;
        // Summed sequentially so the reported MSE does not depend on how rayon
        // happened to split the block.
        for error in errors {
            squared_error += error;
        }

        file.write_all(&codes)?;
        row_start += block_rows;
    }

    let end = file.stream_position()?;
    file.seek(SeekFrom::Start(params_pos))?;
    file.write_all(&params)?;
    file.seek(SeekFrom::Start(end))?;

    Ok(squared_error / (rows as f64 * cols as f64))
}

/// Round-to-nearest quantization of a single row.
///
/// Mirrors `raimodel.compute_group_params` + `raimodel.rtn_quantize`: group
/// statistics in f64, scale and zero rounded to f16 *before* any code is
/// derived (so the reader's `code * scale + zero` reproduces exactly what was
/// quantized against), and codes rounded half-to-even like `np.round`.
#[allow(clippy::too_many_arguments)]
fn quantize_row(
    row: &[f32],
    group_size: usize,
    n_levels: u32,
    packed: bool,
    params_out: &mut [u8],
    codes_out: &mut [u8],
    label: &str,
    row_index: usize,
) -> Result<f64> {
    let cols = row.len();
    let max_code = (n_levels - 1) as f64;
    let mut squared_error = 0.0f64;

    for (group_index, group_start) in (0..cols).step_by(group_size).enumerate() {
        let group_end = (group_start + group_size).min(cols);
        let group = &row[group_start..group_end];

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &value in group {
            let value = value as f64;
            if !value.is_finite() {
                bail!(
                    "{label} row {row_index} group {group_index} contains a non-finite weight; \
                     refusing to write a corrupt model"
                );
            }
            // numpy's reduction, not `f64::min`/`f64::max`: `np.minimum` is
            // `(a < b) ? a : b`, so a tie keeps the *later* value. That is
            // invisible except for signed zeros, where it decides whether the
            // stored zero-point is 0.0 or -0.0 — a one-bit difference in the
            // file for every all-zero group (unused vocabulary rows have
            // thousands of them). Rust's `min`/`max` may return either operand
            // for ±0.0, which is exactly how a byte-identical export breaks.
            if value <= min {
                min = value;
            }
            if value >= max {
                max = value;
            }
        }

        let scale = ((max - min) / max_code).max(MIN_F16_SCALE);
        let scale_f16 = f16::from_f64(scale);
        let zero_f16 = f16::from_f64(min);
        if !scale_f16.is_finite() || scale_f16 <= f16::ZERO || !zero_f16.is_finite() {
            bail!(
                "{label} row {row_index} group {group_index} produced non-finite or non-positive \
                 FP16 quantization parameters"
            );
        }
        let param_offset = group_index * 4;
        params_out[param_offset..param_offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
        params_out[param_offset + 2..param_offset + 4].copy_from_slice(&zero_f16.to_le_bytes());

        let scale64 = scale_f16.to_f64();
        let zero64 = zero_f16.to_f64();
        for (offset, &value) in group.iter().enumerate() {
            let column = group_start + offset;
            let value = value as f64;
            let code = (((value - zero64) / scale64).round_ties_even()).clamp(0.0, max_code);
            let error = value - (code * scale64 + zero64);
            squared_error += error * error;

            let code = code as u8;
            if packed {
                // Low nibble = even column, high nibble = odd column.
                let byte = &mut codes_out[column / 2];
                if column.is_multiple_of(2) {
                    *byte = (*byte & 0xF0) | (code & 0x0F);
                } else {
                    *byte = (*byte & 0x0F) | ((code & 0x0F) << 4);
                }
            } else {
                codes_out[column] = code;
            }
        }
    }
    Ok(squared_error)
}

/// Write one RMSNorm vector as little-endian f32.
fn write_norm(
    file: &mut File,
    store: &mut SafeTensorsSet,
    tensor: &str,
    hidden: usize,
) -> Result<()> {
    let weights = store.read_all(tensor, hidden)?;
    if weights.len() != hidden {
        bail!(
            "tensor '{tensor}' has {} values, expected {hidden}",
            weights.len()
        );
    }
    let mut bytes = Vec::with_capacity(hidden * 4);
    for (index, value) in weights.iter().enumerate() {
        if !value.is_finite() {
            bail!("{tensor} weight {index} is non-finite");
        }
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    file.write_all(&bytes)?;
    Ok(())
}

// =============================================================================
// Container layout
// =============================================================================

fn embedding_section_len(vocab: usize, hidden: usize, group_size: usize) -> Result<u64> {
    let groups = hidden.div_ceil(group_size);
    let params = (vocab as u64)
        .checked_mul(groups as u64)
        .and_then(|v| v.checked_mul(4))
        .context("embedding parameter bytes overflow")?;
    let codes = (vocab as u64)
        .checked_mul(hidden as u64)
        .context("embedding code bytes overflow")?;
    Ok(params + codes)
}

fn linear_section_len(rows: usize, cols: usize, group_size: usize) -> Result<u64> {
    let groups = cols.div_ceil(group_size);
    let params = (rows as u64)
        .checked_mul(groups as u64)
        .and_then(|v| v.checked_mul(4))
        .context("linear parameter bytes overflow")?;
    let codes = (rows as u64)
        .checked_mul(cols as u64)
        .context("linear code bytes overflow")?
        / 2;
    Ok(8 + params + codes)
}

fn layer_section_len(
    linear_dims: &[(usize, usize); 7],
    hidden: usize,
    group_size: usize,
) -> Result<u64> {
    let mut total = 0u64;
    for &(rows, cols) in linear_dims {
        total += linear_section_len(rows, cols, group_size)?;
    }
    Ok(total + 2 * hidden as u64 * 4)
}

fn write_header(file: &mut File, config: &RaiConfig, num_sections: u32) -> Result<()> {
    let mut header = [0u8; HEADER_SIZE as usize];
    header[0..4].copy_from_slice(b"RAIM");
    header[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&config.hidden_size.to_le_bytes());
    header[12..16].copy_from_slice(&config.num_layers.to_le_bytes());
    header[16..20].copy_from_slice(&config.num_heads.to_le_bytes());
    header[20..24].copy_from_slice(&config.num_kv_heads.to_le_bytes());
    header[24..28].copy_from_slice(&config.head_dim.to_le_bytes());
    header[28..32].copy_from_slice(&config.intermediate_size.to_le_bytes());
    header[32..36].copy_from_slice(&config.vocab_size.to_le_bytes());
    header[36..40].copy_from_slice(&config.max_context.to_le_bytes());
    header[40..44].copy_from_slice(&config.rope_theta.to_le_bytes());
    header[44..48].copy_from_slice(&config.norm_eps.to_le_bytes());
    header[48] = config.bits;
    header[49] = config.group_size;
    header[50] = config.embed_bits;
    header[51] = config.embed_group_size;
    header[52..56].copy_from_slice(&num_sections.to_le_bytes());
    file.write_all(&header)?;
    Ok(())
}

// =============================================================================
// Helpers
// =============================================================================

fn layer_linear_name(layer: u32, projection: &str) -> String {
    let block = match projection {
        "gate_proj" | "up_proj" | "down_proj" => "mlp",
        _ => "self_attn",
    };
    format!("model.layers.{layer}.{block}.{projection}.weight")
}

fn default_output_name(model_dir: &Path) -> Result<String> {
    let name = model_dir
        .canonicalize()
        .unwrap_or_else(|_| model_dir.to_path_buf())
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .filter(|n| !n.is_empty())
        .with_context(|| {
            format!(
                "cannot derive an output name from {}; pass --output",
                model_dir.display()
            )
        })?;
    Ok(format!("{name}-q4.raimodel"))
}

/// Mirror of `raimodel.copy_tokenizer_json`: never clobber a *different*
/// tokenizer, because a model silently paired with the wrong one produces
/// fluent nonsense.
fn copy_tokenizer_json(src: &Path, dst: &Path) -> Result<bool> {
    let source = std::fs::read(src).with_context(|| format!("reading {}", src.display()))?;
    if dst.exists() {
        let existing = std::fs::read(dst).with_context(|| format!("reading {}", dst.display()))?;
        if existing == source {
            return Ok(false);
        }
        bail!(
            "refusing to overwrite {}: it differs from this model's tokenizer.json. Another \
             model's tokenizer already lives there — export into a separate output directory \
             (--output <dir>/<name>.raimodel).",
            dst.display()
        );
    }
    if let Some(parent) = dst.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    std::fs::write(dst, &source).with_context(|| format!("writing {}", dst.display()))?;
    Ok(true)
}

fn required_u32(config: &serde_json::Value, key: &str) -> Result<u32> {
    optional_u32(config, key)?.with_context(|| format!("config.json is missing '{key}'"))
}

fn optional_u32(config: &serde_json::Value, key: &str) -> Result<Option<u32>> {
    match config.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => {
            let number = value
                .as_u64()
                .with_context(|| format!("config.json field '{key}' is not a positive integer"))?;
            let number = u32::try_from(number)
                .with_context(|| format!("config.json field '{key}' is out of range: {number}"))?;
            Ok(Some(number))
        }
    }
}

fn optional_f64(config: &serde_json::Value, key: &str) -> Result<Option<f64>> {
    match config.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => {
            Ok(Some(value.as_f64().with_context(|| {
                format!("config.json field '{key}' is not a number")
            })?))
        }
    }
}

/// Format like Python's `%.2e` (two-digit, signed exponent) so the progress
/// output matches the reference exporter's.
fn fmt_sci(value: f64) -> String {
    let text = format!("{value:.2e}");
    match text.split_once('e') {
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(digits) => ('-', digits),
                None => ('+', exponent),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scientific_formatting_matches_python() {
        assert_eq!(fmt_sci(1.234e-5), "1.23e-05");
        assert_eq!(fmt_sci(0.0), "0.00e+00");
        assert_eq!(fmt_sci(1.5e12), "1.50e+12");
    }

    #[test]
    fn codes_round_half_to_even_like_numpy() {
        // 0.5 -> 0 and 1.5 -> 2 is numpy's behaviour; Rust's f64::round would
        // give 1 and 2, shifting every tie by one level.
        assert_eq!(0.5f64.round_ties_even(), 0.0);
        assert_eq!(1.5f64.round_ties_even(), 2.0);
        assert_eq!(2.5f64.round_ties_even(), 2.0);
    }

    #[test]
    fn min_f16_scale_is_the_smallest_positive_f16() {
        assert_eq!(f16::from_f64(MIN_F16_SCALE).to_bits(), 1);
        assert_eq!(f16::from_f64(MIN_F16_SCALE).to_f64(), MIN_F16_SCALE);
    }

    #[test]
    fn head_dim_disagreement_is_rejected() {
        assert_eq!(resolve_head_dim(None, 64, 4).unwrap(), 16);
        assert_eq!(resolve_head_dim(Some(16), 64, 4).unwrap(), 16);
        let error = resolve_head_dim(Some(32), 64, 4).unwrap_err().to_string();
        assert!(error.contains("decoupled head_dim"), "{error}");
    }

    #[test]
    fn layer_tensor_names_follow_the_hf_convention() {
        assert_eq!(
            layer_linear_name(3, "q_proj"),
            "model.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            layer_linear_name(3, "down_proj"),
            "model.layers.3.mlp.down_proj.weight"
        );
    }
}
