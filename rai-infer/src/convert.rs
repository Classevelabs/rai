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
use std::cell::Cell;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::format::PROJECTION_NAMES;
use crate::layers::{Activation, RopeScaling};
use crate::safetensors::SafeTensorsSet;

/// Smallest positive (subnormal) f16; scales are clamped to at least this so a
/// stored scale is always strictly positive. Matches
/// `float(np.nextafter(np.float16(0), np.float16(1)))`.
const MIN_F16_SCALE: f64 = 5.960464477539063e-8;

const HEADER_SIZE_V1: u64 = 64;
const HEADER_SIZE_V2: u64 = 128;
const SECTION_ENTRY_SIZE: u64 = 16;

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
///
/// Taken from the reader so the writer cannot drift from the layout the reader
/// parses — this is also the bit order of the v2 header's `bias_mask`.
const LAYER_LINEAR_NAMES: [&str; 7] = PROJECTION_NAMES;

/// Architectures whose maths this container cannot express even though their
/// tensor names look Llama-shaped.
///
/// `gemma` came off this list in container v2: GeGLU is a stored activation
/// code, and Gemma's other differences are folded at conversion time (see
/// `GemmaFolds`). `gemma2` came off it once the header gained the two softcaps
/// and the layer section gained the sandwich norms.
///
/// `gemma3` stays, and the reason is not softcapping or QK norm — the container
/// now stores both. It is the RoPE base: Gemma3 interleaves sliding and global
/// attention layers and gives them *different* rotary bases
/// (`rope_local_base_freq` = 10 000 for the sliding layers, `rope_theta` =
/// 1 000 000 for the global ones; see `transformers`
/// `models/gemma3/configuration_gemma3.py:144-150`). The header carries one
/// `rope_theta` and the runtime builds one RoPE table from it, so five layers in
/// six would be rotated at the wrong frequency. That is a per-layer
/// architectural variation, not a missing parameter, so it is refused rather
/// than approximated.
const UNSUPPORTED_MODEL_TYPES: [(&str, &str); 0] = [];

/// Model families that need the container's Gemma-specific conversion folds:
/// the `1 + w` RMSNorm and the `sqrt(hidden_size)` embedding scale. Gemma2 uses
/// both, on all four of its norms.
const GEMMA_MODEL_TYPES: [&str; 4] = ["gemma", "gemma2", "gemma3", "gemma3_text"];

/// Families that use Gemma2's sandwich normalization, i.e. carry
/// `pre_feedforward_layernorm` and `post_feedforward_layernorm` per layer.
const SANDWICH_NORM_MODEL_TYPES: [&str; 3] = ["gemma2", "gemma3", "gemma3_text"];

/// Conversion inputs. Defaults match `export_rtn.py`.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// HuggingFace checkpoint directory.
    pub model_dir: PathBuf,
    /// Output path; defaults to `<dirname>-q4.raimodel` in the
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

/// A progress event, emitted once per line the converter would print.
///
/// `rai convert` narrates itself on stdout, which is useless to a UI that is
/// not a terminal. Every one of those lines also arrives here, tagged with the
/// phase it belongs to and how much of the model has been written, so a caller
/// can drive a per-layer progress bar without parsing log text.
#[derive(Debug, Clone, Copy)]
pub struct ConvertProgress<'a> {
    /// Coarse phase: `planning`, `embedding`, `layers`, `final-norm`,
    /// `lm_head`, `tokenizer`, `done`.
    pub stage: &'a str,
    /// Sections written so far, as a percentage of the sections planned.
    /// Monotonic, `0.0..=100.0`.
    pub percent: f32,
    /// The human-readable line, byte for byte what the CLI prints.
    pub message: &'a str,
    /// `(layer_index, num_layers)` while quantizing layers, else `None`.
    pub layer: Option<(u32, u32)>,
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
    activation: Activation,
    rope_scaling: RopeScaling,
    /// Bit *i*: projection *i* of [`PROJECTION_NAMES`] carries a bias.
    bias_mask: u8,
    embed_scale: f32,
    /// Per-head `q_norm`/`k_norm` vectors are stored in every layer section.
    has_qk_norm: bool,
    /// Gemma2's two extra per-layer norms are stored in every layer section.
    has_sandwich_norm: bool,
    /// `q_norm`/`k_norm` span the whole projection rather than one head
    /// (OLMo2). Never set together with `has_qk_norm`.
    has_full_qk_norm: bool,
    /// The two tail norms apply to block outputs, not to the residual stream
    /// before each block (OLMo2). Never set together with `has_sandwich_norm`.
    post_norm: bool,
    /// `cap` for `cap * tanh(x/cap)` on attention logits; 0.0 = disabled.
    attn_logit_softcap: f32,
    /// The same on output logits; 0.0 = disabled.
    final_logit_softcap: f32,
    /// Explicit query-key scale; 0.0 = the default `1/sqrt(head_dim)`.
    attn_scale: f32,
    /// RoPE base for non-global layers; 0.0 = one base for every layer.
    rope_local_theta: f32,
    /// Layer *i* is global when `(i + 1) % stride == 0`; 0 = unused.
    global_layer_stride: u8,
}

impl RaiConfig {
    /// The lowest container version that can express this model.
    ///
    /// v1 stays the answer whenever nothing new is used, so every checkpoint
    /// that converted before this change still produces the same bytes it
    /// always did — that is what `convert_matches_python` pins.
    fn version(&self) -> u32 {
        if self.activation == Activation::Silu
            && self.rope_scaling == RopeScaling::None
            && self.bias_mask == 0
            && self.embed_scale == 1.0
            && !self.has_qk_norm
            && !self.has_sandwich_norm
            && !self.has_full_qk_norm
            && !self.post_norm
            && self.attn_logit_softcap == 0.0
            && self.final_logit_softcap == 0.0
            && self.attn_scale == 0.0
            && self.global_layer_stride == 0
        {
            1
        } else {
            2
        }
    }

    /// The `flags` byte the header will carry.
    fn flags(&self) -> u8 {
        let mut flags = 0u8;
        if self.bias_mask != 0 {
            flags |= crate::format::FLAG_HAS_BIASES;
        }
        if self.has_qk_norm {
            flags |= crate::format::FLAG_HAS_QK_NORM;
        }
        if self.has_sandwich_norm {
            flags |= crate::format::FLAG_HAS_SANDWICH_NORM;
        }
        if self.has_full_qk_norm {
            flags |= crate::format::FLAG_HAS_FULL_QK_NORM;
        }
        if self.post_norm {
            flags |= crate::format::FLAG_POST_NORM;
        }
        flags
    }

    fn header_size(&self) -> u64 {
        if self.version() >= 2 {
            HEADER_SIZE_V2
        } else {
            HEADER_SIZE_V1
        }
    }

    fn attention_dim(&self) -> usize {
        self.num_heads as usize * self.head_dim as usize
    }

    fn kv_dim(&self) -> usize {
        self.num_kv_heads as usize * self.head_dim as usize
    }

    fn has_bias(&self, index: usize) -> bool {
        self.bias_mask & (1 << index) != 0
    }

    /// `(rows, cols)` of the seven layer projections, in section order.
    /// Must agree with `format::ModelConfig::projection_dims`.
    fn projection_dims(&self) -> [(usize, usize); 7] {
        let hidden = self.hidden_size as usize;
        let inter = self.intermediate_size as usize;
        let q_dim = self.attention_dim();
        let kv_dim = self.kv_dim();
        [
            (q_dim, hidden),
            (kv_dim, hidden),
            (kv_dim, hidden),
            (hidden, q_dim),
            (inter, hidden),
            (inter, hidden),
            (hidden, inter),
        ]
    }
}

/// The two Gemma differences that are folded into stored weights rather than
/// given a runtime code path.
///
/// * **RMSNorm.** Gemma computes `x/rms * (1 + w)`; the shipped kernel computes
///   `x/rms * w` (`layers.rs::rms_norm`, verified). Storing `w' = 1 + w` makes
///   the existing kernel exactly right, so no reader change and no flag.
/// * **Embedding scale.** Gemma multiplies the *input* embedding by
///   `sqrt(hidden_size)`. This one is deliberately **not** folded, despite
///   looking like the same kind of trick: Gemma ties `lm_head` to the embedding
///   table, and the tied output projection uses the unscaled weights. A folded
///   table would inflate every logit by ~45x. It is carried in the v2 header's
///   `embed_scale` and applied at lookup time instead.
#[derive(Debug, Clone, Copy, Default)]
struct GemmaFolds {
    /// Add 1.0 to every RMSNorm weight before writing it.
    norm_plus_one: bool,
}

/// Convert a HuggingFace checkpoint directory to `.raimodel`.
pub fn convert(options: &ConvertOptions) -> Result<ConvertSummary> {
    convert_with_progress(options, &|_| {})
}

/// [`convert`], with every narration line also delivered to `progress`.
///
/// Split out rather than added to [`ConvertOptions`] so that no existing
/// caller's construction of that struct has to change.
pub fn convert_with_progress(
    options: &ConvertOptions,
    progress: &dyn Fn(ConvertProgress<'_>),
) -> Result<ConvertSummary> {
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

    let is_gemma = GEMMA_MODEL_TYPES.contains(&model_type);
    let has_sandwich_norm = SANDWICH_NORM_MODEL_TYPES.contains(&model_type);
    let activation = resolve_activation(&hf, is_gemma)?;
    let rope_scaling = resolve_rope_scaling(&hf)?;
    let attn_logit_softcap = resolve_softcap(&hf, "attn_logit_softcapping")?;
    let final_logit_softcap = resolve_softcap(&hf, "final_logit_softcapping")?;
    let attn_scale = resolve_attn_scale(&hf, head_dim)?;
    let per_layer_rope = resolve_per_layer_rope(&hf, num_layers)?;
    let folds = GemmaFolds {
        norm_plus_one: is_gemma,
    };
    // sqrt(hidden_size), computed in f64 and rounded once, so the value in the
    // header is the nearest f32 to the true normalizer.
    let embed_scale = if is_gemma {
        (hidden_size as f64).sqrt() as f32
    } else {
        1.0
    };

    let mut config = RaiConfig {
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
        activation,
        rope_scaling,
        bias_mask: 0,
        embed_scale,
        // Settled once the tensor namespace is open, below.
        has_qk_norm: false,
        has_sandwich_norm,
        has_full_qk_norm: false,
        post_norm: false,
        rope_local_theta: per_layer_rope.0,
        global_layer_stride: per_layer_rope.1,
        attn_logit_softcap,
        final_logit_softcap,
        attn_scale,
    };

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

    // Progress accounting: one unit per section (the embedding, each layer,
    // the final norm, and `lm_head` when untied), so `percent` advances once
    // per layer — the only granularity that means anything on a 7B model.
    let total_units = (num_layers + 2 + u32::from(!tied)) as f32;
    let done_units = Cell::new(0u32);
    let stage = Cell::new("planning");
    let layer_of = Cell::new(None::<(u32, u32)>);
    let enter = |name: &'static str, done: u32, layer: Option<(u32, u32)>| {
        stage.set(name);
        done_units.set(done);
        layer_of.set(layer);
    };

    let log = |line: &str| {
        if !options.quiet {
            println!("{line}");
            let _ = std::io::stdout().flush();
        }
        progress(ConvertProgress {
            stage: stage.get(),
            percent: 100.0 * done_units.get() as f32 / total_units,
            message: line,
            layer: layer_of.get(),
        });
    };

    log(&format!("Output: {}", output_path.display()));
    log(&format!("Reading {}...", options.model_dir.display()));
    let mut store = SafeTensorsSet::open(&options.model_dir)?;
    assert_exportable_architecture(&hf, Some(&store), options.max_context, num_layers)?;

    // Which projections carry biases is a property of the checkpoint, not the
    // config, so it can only be settled once the tensor namespace is open.
    config.bias_mask = resolve_bias_mask(&store, num_layers)?;
    let qk_norm = resolve_qk_norm(
        &store,
        num_layers,
        head_dim,
        config.attention_dim(),
        config.kv_dim(),
    )?;
    config.has_qk_norm = qk_norm == QkNormKind::PerHead;
    config.has_full_qk_norm = qk_norm == QkNormKind::FullWidth;
    config.post_norm = resolve_post_norm(&store, num_layers)?;
    if config.post_norm && config.has_sandwich_norm {
        bail!(
            "checkpoint has no input_layernorm but its model_type is sandwich-normed; those \
             describe incompatible layer shapes and the converter cannot tell which is meant."
        );
    }
    validate_model_config(&config)?;

    let hidden = hidden_size as usize;
    let vocab = vocab_size as usize;
    let group_size = options.group_size as usize;
    let embed_group_size = options.embed_group_size as usize;
    let linear_dims = config.projection_dims();

    // Confirm every tensor exists with the shape the reader will demand, before
    // a single byte is written.
    let layout = ProjectionLayout::detect(&store, num_layers)?;
    layout.check_fused_tensors(&store, num_layers, &linear_dims)?;
    check_tensor(&store, "model.embed_tokens.weight", vocab, hidden)?;
    for layer in 0..num_layers {
        for (index, (name, (rows, cols))) in LAYER_LINEAR_NAMES.iter().zip(linear_dims).enumerate()
        {
            // A fused projection was already checked as a whole, above; its
            // parts have no tensor of their own to check.
            let source = layout.source(layer, index, &linear_dims);
            if source.row_offset == 0 && source.tensor == layer_linear_name(layer, name) {
                check_tensor(&store, &source.tensor, rows, cols)?;
            }
            if config.has_bias(index) {
                check_vector(&store, &layer_bias_name(layer, name), rows)?;
            }
        }
        for name in tail_norm_names(layer, config.has_sandwich_norm, config.post_norm) {
            check_vector(&store, &name, hidden)?;
        }
        if config.has_qk_norm {
            for name in qk_norm_names(layer) {
                check_vector(&store, &name, head_dim as usize)?;
            }
        }
    }
    if config.has_sandwich_norm {
        check_sandwich_norms(&store, num_layers, hidden)?;
    }
    check_vector(&store, "model.norm.weight", hidden)?;
    if !tied {
        check_tensor(&store, "lm_head.weight", vocab, hidden)?;
    }

    log(&format!(
        "{model_type}: {num_layers}L h={hidden_size} inter={intermediate_size} \
         heads={num_heads}/{num_kv_heads} head_dim={head_dim} vocab={vocab_size}"
    ));
    log(&format!(
        "container v{}: activation={:?} rope={:?} bias_mask={:#04x} embed_scale={} \
         flags={:#04x} (qk_norm={} sandwich_norm={}) softcap=attn:{} final:{} attn_scale={}",
        config.version(),
        config.activation,
        config.rope_scaling,
        config.bias_mask,
        config.embed_scale,
        config.flags(),
        config.has_qk_norm,
        config.has_sandwich_norm,
        config.attn_logit_softcap,
        config.final_logit_softcap,
        config.attn_scale,
    ));

    // ---- Plan the container ------------------------------------------------
    // Every section size follows from the config, so offsets can be written
    // before the data exists and each section can be streamed straight out.
    let (section_sizes, planned_total) = plan_sections(&config, tied)?;

    let num_sections = section_sizes.len();
    let data_start = config.header_size() + num_sections as u64 * SECTION_ENTRY_SIZE;
    let mut offsets = Vec::with_capacity(num_sections);
    let mut cursor = data_start;
    for size in &section_sizes {
        offsets.push(cursor);
        cursor += size;
    }
    let total_size = cursor;
    debug_assert_eq!(total_size, planned_total);

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
    enter("embedding", 0, None);
    log("\n=== EMBEDDING 8-BIT ===");
    let t0 = Instant::now();
    let mse = write_matrix(
        &mut file,
        &mut store,
        &MatrixJob {
            tensor: "model.embed_tokens.weight",
            source_row_offset: 0,
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
    enter("layers", 1, None);
    log("\n=== RTN-4BIT QUANTIZATION ===");
    for layer in 0..num_layers {
        let t0 = Instant::now();
        enter("layers", 1 + layer, Some((layer, num_layers)));
        for (index, (name, (rows, cols))) in LAYER_LINEAR_NAMES.iter().zip(linear_dims).enumerate()
        {
            let label = format!("L{layer}.{name}");
            let source = layout.source(layer, index, &linear_dims);
            let mse = write_matrix(
                &mut file,
                &mut store,
                &MatrixJob {
                    tensor: &source.tensor,
                    source_row_offset: source.row_offset,
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
        // Bias block, in projection order, immediately after the seven linears.
        for (index, name) in LAYER_LINEAR_NAMES.iter().enumerate() {
            if !config.has_bias(index) {
                continue;
            }
            write_f32_vector(
                &mut file,
                &mut store,
                &layer_bias_name(layer, name),
                linear_dims[index].0,
                false,
            )?;
        }
        // QK-norm block, then the sandwich block, then the two layer norms —
        // exactly the order `format::RaiModelFile::layer` reads them in.
        if config.has_qk_norm || config.has_full_qk_norm {
            let (q_len, k_len) = if config.has_full_qk_norm {
                (config.attention_dim(), config.kv_dim())
            } else {
                (head_dim as usize, head_dim as usize)
            };
            for (name, len) in qk_norm_names(layer).into_iter().zip([q_len, k_len]) {
                write_f32_vector(&mut file, &mut store, &name, len, folds.norm_plus_one)?;
            }
        }
        if config.has_sandwich_norm {
            for name in sandwich_norm_names(layer) {
                write_f32_vector(&mut file, &mut store, &name, hidden, folds.norm_plus_one)?;
            }
        }
        for name in tail_norm_names(layer, config.has_sandwich_norm, config.post_norm) {
            write_f32_vector(&mut file, &mut store, &name, hidden, folds.norm_plus_one)?;
        }
        enter("layers", 2 + layer, Some((layer, num_layers)));
        log(&format!(
            "  Layer {layer}/{num_layers}: {:.1}s",
            t0.elapsed().as_secs_f64()
        ));
    }

    // ---- Section L+1: final norm -------------------------------------------
    enter("final-norm", 1 + num_layers, None);
    write_f32_vector(
        &mut file,
        &mut store,
        "model.norm.weight",
        hidden,
        folds.norm_plus_one,
    )?;

    // ---- Section L+2 (untied only): lm_head --------------------------------
    enter("lm_head", 2 + num_layers, None);
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
                source_row_offset: 0,
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

    enter("tokenizer", total_units as u32, None);
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
    enter("done", total_units as u32, None);
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
    // `num_heads * head_dim` may differ from `hidden_size` (Gemma). Mirror of
    // the reader's `validate_config`.
    let attention_dim = config.num_heads as u64 * config.head_dim as u64;
    if attention_dim > MAX_INTERMEDIATE_SIZE as u64 {
        bail!(
            "num_heads * head_dim is {attention_dim}; the reader's maximum is \
             {MAX_INTERMEDIATE_SIZE}"
        );
    }
    if !attention_dim.is_multiple_of(2) {
        bail!("num_heads * head_dim must be even for packed 4-bit kernels, got {attention_dim}");
    }

    let group_size = config.group_size as usize;
    let max_linear_groups = (config.hidden_size as usize)
        .div_ceil(group_size)
        .max((config.intermediate_size as usize).div_ceil(group_size))
        .max((attention_dim as usize).div_ceil(group_size));
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

/// Resolve `head_dim`, honouring an explicit config value.
///
/// A decoupled `head_dim` (`num_heads * head_dim != hidden_size`, as Gemma
/// uses) is no longer refused: `q_proj` is stored `[num_heads*head_dim, hidden]`
/// and `o_proj` `[hidden, num_heads*head_dim]`, so the attention block still
/// reads and writes hidden-sized vectors and only its interior changes width.
/// When the config omits `head_dim` the derived value must divide exactly —
/// silently truncating would misdescribe the model.
fn resolve_head_dim(config_head_dim: Option<u32>, hidden_size: u32, num_heads: u32) -> Result<u32> {
    if num_heads == 0 {
        bail!("num_attention_heads must be greater than zero");
    }
    match config_head_dim {
        Some(head_dim) if head_dim != 0 => Ok(head_dim),
        _ => {
            if !hidden_size.is_multiple_of(num_heads) {
                bail!(
                    "model config omits head_dim and hidden_size {hidden_size} is not divisible \
                     by num_attention_heads {num_heads}"
                );
            }
            Ok(hidden_size / num_heads)
        }
    }
}

/// Pick the gated-MLP activation from the config.
///
/// Gemma trains against `gelu_pytorch_tanh` and, in older checkpoints, leaves
/// `hidden_activation` null with a misleading `hidden_act: "gelu"` — the
/// reference model resolves that to `gelu_pytorch_tanh`, so this does too.
/// Anything else is refused by name rather than run as SiLU.
fn resolve_activation(hf: &serde_json::Value, is_gemma: bool) -> Result<Activation> {
    let declared = hf
        .get("hidden_activation")
        .and_then(|v| v.as_str())
        .or_else(|| hf.get("hidden_act").and_then(|v| v.as_str()));
    match (is_gemma, declared) {
        // Gemma with hidden_activation unset: transformers warns and uses
        // gelu_pytorch_tanh, ignoring hidden_act.
        (true, None) => Ok(Activation::GeluTanh),
        (true, Some("gelu")) if hf.get("hidden_activation").is_none_or(|v| v.is_null()) => {
            Ok(Activation::GeluTanh)
        }
        (_, Some("gelu_pytorch_tanh")) => Ok(Activation::GeluTanh),
        (false, None | Some("silu") | Some("swish")) => Ok(Activation::Silu),
        (_, Some(other)) => bail!(
            "config declares hidden activation '{other}'; the container stores SiLU/SwiGLU and \
             GeLU-tanh/GeGLU only, and running the wrong one would silently produce a different \
             model."
        ),
    }
}

/// Read `rope_scaling` into the container's representation.
///
/// transformers >= 5 normalizes plain RoPE into `{"rope_type": "default"}`, so
/// the field's presence means nothing on its own. Anything other than absent,
/// null, "default" or "llama3" is refused: the reader would otherwise build a
/// table for the wrong positions.
fn resolve_rope_scaling(hf: &serde_json::Value) -> Result<RopeScaling> {
    let Some(scaling) = hf.get("rope_scaling") else {
        return Ok(RopeScaling::None);
    };
    if scaling.is_null() {
        return Ok(RopeScaling::None);
    }
    let Some(object) = scaling.as_object() else {
        bail!("config field rope_scaling is not an object: {scaling}");
    };
    let rope_type = object
        .get("rope_type")
        .or_else(|| object.get("type"))
        .and_then(|v| v.as_str());
    match rope_type {
        None | Some("default") => Ok(RopeScaling::None),
        Some("llama3") => {
            let number = |key: &str| -> Result<f64> {
                object
                    .get(key)
                    .and_then(serde_json::Value::as_f64)
                    .with_context(|| {
                        format!("rope_scaling declares rope_type 'llama3' but has no numeric {key}")
                    })
            };
            let factor = number("factor")?;
            let low_freq_factor = number("low_freq_factor")?;
            let high_freq_factor = number("high_freq_factor")?;
            let original = object
                .get("original_max_position_embeddings")
                .and_then(serde_json::Value::as_u64)
                .or_else(|| hf.get("max_position_embeddings").and_then(|v| v.as_u64()))
                .context(
                    "rope_scaling declares rope_type 'llama3' but neither it nor the config \
                     provides original_max_position_embeddings",
                )?;
            for (name, value) in [
                ("factor", factor),
                ("low_freq_factor", low_freq_factor),
                ("high_freq_factor", high_freq_factor),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    bail!("rope_scaling {name} must be finite and positive, got {value}");
                }
            }
            if (high_freq_factor - low_freq_factor).abs() == 0.0 {
                bail!("rope_scaling high_freq_factor must differ from low_freq_factor");
            }
            let original = u32::try_from(original)
                .context("rope_scaling original_max_position_embeddings does not fit in a u32")?;
            if original == 0 || original > MAX_CONTEXT {
                bail!(
                    "rope_scaling original_max_position_embeddings must be in 1..={MAX_CONTEXT}, \
                     got {original}"
                );
            }
            Ok(RopeScaling::Llama3 {
                factor: factor as f32,
                low_freq_factor: low_freq_factor as f32,
                high_freq_factor: high_freq_factor as f32,
                original_max_position: original,
            })
        }
        Some(other) => bail!(
            "config declares rope_scaling type '{other}'; the container stores plain RoPE and \
             the llama3 scheme only, so positions would be wrong."
        ),
    }
}

/// Work out which projections carry biases, and insist every layer agrees.
///
/// A checkpoint where only some layers are biased is not a model this (or any)
/// exporter should guess at: the container stores one mask for the whole file,
/// so a partial set would mean silently dropping real parameters.
fn resolve_bias_mask(store: &SafeTensorsSet, num_layers: u32) -> Result<u8> {
    if num_layers == 0 {
        return Ok(0);
    }
    let mask_for = |layer: u32| -> u8 {
        let mut mask = 0u8;
        for (index, name) in LAYER_LINEAR_NAMES.iter().enumerate() {
            if store.info(&layer_bias_name(layer, name)).is_some() {
                mask |= 1 << index;
            }
        }
        mask
    };
    let mask = mask_for(0);
    for layer in 1..num_layers {
        let other = mask_for(layer);
        if other != mask {
            bail!(
                "checkpoint is inconsistent: layer 0 has projection biases {mask:#04x} but layer \
                 {layer} has {other:#04x}. The container stores one bias mask for the whole \
                 model, so exporting this would drop real parameters."
            );
        }
    }
    Ok(mask)
}

/// The tensor names of a layer's per-head QK norms.
fn qk_norm_names(layer: u32) -> [String; 2] {
    [
        format!("model.layers.{layer}.self_attn.q_norm.weight"),
        format!("model.layers.{layer}.self_attn.k_norm.weight"),
    ]
}

/// Read the second RoPE base and the layer stride, for a model whose layers do
/// not all rotate at the same frequency.
///
/// Returns `(0.0, 0)` for the overwhelming majority of checkpoints, which have
/// one base. Gemma3 is the family this exists for: its sliding layers use
/// `rope_local_base_freq` and its global layers `rope_theta`, and
/// `Gemma3Config` derives which is which as
/// `"sliding_attention" if (i + 1) % sliding_window_pattern else "full_attention"`.
///
/// An explicit `layer_types` list is honoured only when it *is* that pattern.
/// The container stores a stride, and silently rounding an irregular list to
/// the nearest stride would rotate some layers at the wrong base — a model
/// that loads, runs, and is quietly worse.
fn resolve_per_layer_rope(hf: &serde_json::Value, num_layers: u32) -> Result<(f32, u8)> {
    let local = resolve_softcap(hf, "rope_local_base_freq").unwrap_or(0.0);
    if local <= 0.0 {
        return Ok((0.0, 0));
    }
    let stride = optional_u32(hf, "sliding_window_pattern")?.unwrap_or(6);
    if stride < 2 {
        bail!(
            "sliding_window_pattern is {stride}; the container stores a stride of 2 or more,              and a stride below that describes no alternation at all."
        );
    }
    let stride = u8::try_from(stride).map_err(|_| {
        anyhow::anyhow!("sliding_window_pattern {stride} does not fit in the header's one byte")
    })?;

    if let Some(types) = hf.get("layer_types").and_then(|value| value.as_array()) {
        if types.len() != num_layers as usize {
            bail!(
                "layer_types lists {} entries but the model has {num_layers} layers",
                types.len()
            );
        }
        for (index, entry) in types.iter().enumerate() {
            let name = entry.as_str().unwrap_or_default();
            let is_global = (index + 1).is_multiple_of(stride as usize);
            let expected = if is_global {
                "full_attention"
            } else {
                "sliding_attention"
            };
            if name != expected {
                bail!(
                    "layer_types[{index}] is '{name}' but a stride of {stride} implies                      '{expected}'. The container stores the stride, not a per-layer list, so an                      irregular pattern cannot be represented and would rotate some layers at the                      wrong RoPE base."
                );
            }
        }
    }
    Ok((local, stride))
}

/// The tensor names of a layer's two sandwich norms, in stored order.
fn sandwich_norm_names(layer: u32) -> [String; 2] {
    [
        format!("model.layers.{layer}.post_attention_layernorm.weight"),
        format!("model.layers.{layer}.post_feedforward_layernorm.weight"),
    ]
}

/// The tensors supplying a layer's two tail norms, in stored order.
///
/// For a pre-norm model these are the norms applied to the residual stream
/// before the attention and MLP blocks. For a post-norm model (OLMo2) the same
/// two slots hold the norms applied to those blocks' *outputs* — there is no
/// pre-block norm to store, so the slots are reused rather than left empty and
/// two more added.
fn tail_norm_names(layer: u32, sandwich: bool, post_norm: bool) -> [String; 2] {
    if post_norm {
        return sandwich_norm_names(layer);
    }
    [
        format!("model.layers.{layer}.input_layernorm.weight"),
        pre_mlp_norm_name(layer, sandwich),
    ]
}

/// Decide whether this checkpoint is post-normed, from the tensors present.
///
/// A post-norm model has no `input_layernorm` anywhere and does have both
/// output norms. Anything in between — some layers pre-normed, an output norm
/// missing — is refused rather than guessed, because the flag covers the whole
/// file and the wrong answer changes every layer's arithmetic.
fn resolve_post_norm(store: &SafeTensorsSet, num_layers: u32) -> Result<bool> {
    if num_layers == 0 {
        return Ok(false);
    }
    let has_input_norm = |layer: u32| {
        store
            .info(&format!("model.layers.{layer}.input_layernorm.weight"))
            .is_some()
    };
    if has_input_norm(0) {
        return Ok(false);
    }
    for layer in 1..num_layers {
        if has_input_norm(layer) {
            bail!(
                "checkpoint is inconsistent: layer 0 has no input_layernorm but layer {layer} \
                 does. The container stores one norm placement for the whole model."
            );
        }
    }
    for layer in 0..num_layers {
        for name in sandwich_norm_names(layer) {
            if store.info(&name).is_none() {
                bail!(
                    "layer {layer} has no input_layernorm and no '{name}' either, so there is \
                     no norm to apply to the block. This converter implements pre-norm models \
                     and OLMo2-style post-norm models, which carry post_attention_layernorm \
                     and post_feedforward_layernorm."
                );
            }
        }
    }
    Ok(true)
}

/// The tensor supplying the norm applied to the residual stream before the MLP.
///
/// For a sandwich-normed model that is `pre_feedforward_layernorm`; for
/// everything else it is `post_attention_layernorm`, which for Gemma2 means
/// something else entirely (see `format.rs`'s layer-section documentation).
fn pre_mlp_norm_name(layer: u32, sandwich: bool) -> String {
    if sandwich {
        format!("model.layers.{layer}.pre_feedforward_layernorm.weight")
    } else {
        format!("model.layers.{layer}.post_attention_layernorm.weight")
    }
}

/// Decide whether this checkpoint carries per-head QK norms, insisting that
/// every layer agrees and that both halves of every pair are present.
///
/// Like `resolve_bias_mask`, a partial set is refused rather than guessed at:
/// the container stores one flag for the whole file, so exporting a checkpoint
/// where only some layers were normed would silently change the maths of the
/// rest.
fn resolve_qk_norm(
    store: &SafeTensorsSet,
    num_layers: u32,
    head_dim: u32,
    attention_dim: usize,
    kv_dim: usize,
) -> Result<QkNormKind> {
    if num_layers == 0 {
        return Ok(QkNormKind::None);
    }
    let present = |layer: u32| -> (bool, bool) {
        let [q, k] = qk_norm_names(layer);
        (store.info(&q).is_some(), store.info(&k).is_some())
    };
    let (q0, k0) = present(0);
    if q0 != k0 {
        bail!(
            "layer 0 carries only one of self_attn.q_norm / self_attn.k_norm; the container \
             stores the pair or neither, and normalizing one side alone would change the \
             attention scores."
        );
    }
    for layer in 1..num_layers {
        let (q, k) = present(layer);
        if (q, k) != (q0, k0) {
            bail!(
                "checkpoint is inconsistent: layer 0 {} per-head QK norms but layer {layer} \
                 {}. The container stores one flag for the whole model.",
                if q0 { "has" } else { "has no" },
                if q { "does" } else { "does not" }
            );
        }
    }
    if !q0 {
        return Ok(QkNormKind::None);
    }

    // Same tensor names, two different operations. The width is what tells
    // them apart, so it is read rather than inferred from `model_type`: a
    // fine-tune that renames itself still has to store one shape or the other.
    let head_dim = head_dim as usize;
    let [q_name, _] = qk_norm_names(0);
    let q_values: usize = store.require(&q_name)?.shape.iter().product();
    let kind = if q_values == head_dim {
        QkNormKind::PerHead
    } else if q_values == attention_dim {
        QkNormKind::FullWidth
    } else {
        bail!(
            "tensor '{q_name}' holds {q_values} values, which is neither head_dim ({head_dim}, \
             a per-head norm as Qwen3 uses) nor num_heads*head_dim ({attention_dim}, a \
             projection-wide norm as OLMo2 uses); the container implements those two shapes."
        );
    };
    // `head_dim == attention_dim` only when there is one head, in which case
    // the two shapes coincide and either reading is correct.
    let (q_expected, k_expected) = match kind {
        QkNormKind::PerHead => (head_dim, head_dim),
        QkNormKind::FullWidth => (attention_dim, kv_dim),
        QkNormKind::None => unreachable!("returned above"),
    };
    for layer in 0..num_layers {
        let [q, k] = qk_norm_names(layer);
        check_vector(store, &q, q_expected)?;
        check_vector(store, &k, k_expected)?;
    }
    Ok(kind)
}

/// Which of the two QK-norm shapes a checkpoint carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QkNormKind {
    None,
    /// One `head_dim` vector applied inside each head, before RoPE (Qwen3).
    PerHead,
    /// Vectors spanning the whole Q and K projections, applied before the
    /// tensor is split into heads (OLMo2). Under grouped-query attention the
    /// two are different lengths.
    FullWidth,
}

/// Confirm a sandwich-normed family really carries all four norms per layer.
fn check_sandwich_norms(store: &SafeTensorsSet, num_layers: u32, hidden: usize) -> Result<()> {
    for layer in 0..num_layers {
        check_vector(store, &pre_mlp_norm_name(layer, true), hidden)?;
        for name in sandwich_norm_names(layer) {
            check_vector(store, &name, hidden)?;
        }
    }
    Ok(())
}

/// Read a logit softcap out of the config. Absent or null means disabled.
fn resolve_softcap(hf: &serde_json::Value, key: &str) -> Result<f32> {
    let Some(value) = hf.get(key) else {
        return Ok(0.0);
    };
    if value.is_null() {
        return Ok(0.0);
    }
    let cap = value
        .as_f64()
        .with_context(|| format!("config field {key} is not a number: {value}"))?;
    if !cap.is_finite() || cap <= 0.0 {
        bail!("config field {key} must be finite and positive, got {cap}");
    }
    Ok(cap as f32)
}

/// The query-key multiplier, stored only when it differs from the default.
///
/// Gemma2 sets `query_pre_attn_scalar` and attends with
/// `query_pre_attn_scalar ** -0.5` (`models/gemma2/modeling_gemma2.py:229`).
/// For gemma-2-2b that is 256 against a `head_dim` of 256, i.e. exactly the
/// default, so nothing is stored and the file stays as small as it can be. For
/// gemma-2-27b it is 144 against a `head_dim` of 128, which is a genuinely
/// different scale — and one whose absence produces plausible-looking but worse
/// output rather than an error, which is why it is checked rather than assumed.
fn resolve_attn_scale(hf: &serde_json::Value, head_dim: u32) -> Result<f32> {
    let Some(value) = hf.get("query_pre_attn_scalar") else {
        return Ok(0.0);
    };
    if value.is_null() {
        return Ok(0.0);
    }
    let scalar = value
        .as_f64()
        .context("config field query_pre_attn_scalar is not a number")?;
    if !scalar.is_finite() || scalar <= 0.0 {
        bail!("config field query_pre_attn_scalar must be finite and positive, got {scalar}");
    }
    let scale = scalar.powf(-0.5) as f32;
    let default = 1.0 / (head_dim as f32).sqrt();
    Ok(if scale == default { 0.0 } else { scale })
}

/// Mirror of `raimodel.assert_exportable_architecture`, driven by the config
/// and the checkpoint's tensor names instead of a live torch module tree.
///
/// `store` is `None` when only a `config.json` is in hand (see [`preflight`]),
/// in which case the three tensor-level rules cannot be evaluated and every
/// config-level rule still is. Conversion itself always passes `Some`.
fn assert_exportable_architecture(
    hf: &serde_json::Value,
    store: Option<&SafeTensorsSet>,
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

    if let Some(store) = store {
        // Either a separate q_proj or a fused qkv_proj (Phi-3) is enough; both
        // absent means the module tree is not one this converter can walk.
        let has_attention_weights = ["q_proj", "qkv_proj"].iter().any(|name| {
            store
                .info(&format!("model.layers.0.self_attn.{name}.weight"))
                .is_some()
        });
        if !has_attention_weights && num_layers > 0 {
            bail!(
                "this checkpoint does not expose model.layers.0.self_attn.q_proj.weight or \
                 model.layers.0.self_attn.qkv_proj.weight; the converter supports Llama-style \
                 causal LMs (LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM, and \
                 architecturally identical models)."
            );
        }

        // Per-layer projection biases (Qwen2/Qwen2.5) are a v2 capability now;
        // see `resolve_bias_mask`, which is where an inconsistent set is
        // caught. The *output* projection is a different matter: `bias_mask`
        // covers the seven layer projections only, so an lm_head bias has
        // nowhere to go and would be dropped in silence.
        if store.info("lm_head.bias").is_some() {
            problems.push(
                "checkpoint carries lm_head.bias; the container stores biases for the seven \
                 layer projections only, so this would be silently dropped."
                    .to_string(),
            );
        }

        // Per-head QK norms (Qwen3, Gemma3) are a v2 capability now; see
        // `resolve_qk_norm`, which is where an inconsistent or wrongly-shaped
        // set is caught by name.
    }

    // `rope_scaling` is handled by `resolve_rope_scaling`, which accepts
    // default and llama3 and refuses every other scheme by name.

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

    // `attn_logit_softcapping` and `final_logit_softcapping` are v2 header
    // fields now; `resolve_softcap` refuses a non-numeric or non-positive one.

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

/// What the container would store for a checkpoint the converter accepts.
#[derive(Debug, Clone)]
pub struct PreflightContainer {
    /// Lowest container version that can express the model (1 or 2).
    pub version: u32,
    pub activation: Activation,
    pub rope_scaling: RopeScaling,
    /// Which of [`PROJECTION_NAMES`] carry biases. Always 0 when the weights
    /// were not available to look at.
    pub bias_mask: u8,
    pub embed_scale: f32,
    pub tied_embeddings: bool,
    /// Exact size of the `.raimodel` this conversion would write.
    pub output_bytes: u64,
    pub num_sections: usize,
    /// Per-head `q_norm`/`k_norm` vectors would be stored (Qwen3-shaped).
    pub has_qk_norm: bool,
    /// Gemma2's two extra per-layer norms would be stored.
    pub has_sandwich_norm: bool,
    /// QK norms span the whole projection rather than one head (OLMo2).
    pub has_full_qk_norm: bool,
    /// The two tail norms apply to block outputs, not the residual stream
    /// before them (OLMo2).
    pub post_norm: bool,
    /// Second RoPE base for non-global layers; 0.0 when the model has one.
    pub rope_local_theta: f32,
    /// Layer *i* is global when `(i + 1) % stride == 0`; 0 when unused.
    pub global_layer_stride: u8,
    /// Attention-logit softcap; 0.0 when the model does not cap.
    pub attn_logit_softcap: f32,
    /// Output-logit softcap; 0.0 when the model does not cap.
    pub final_logit_softcap: f32,
    /// Explicit query-key scale; 0.0 means `1/sqrt(head_dim)`.
    pub attn_scale: f32,
}

/// The result of [`preflight`]: can this checkpoint be converted, and if so
/// into what.
#[derive(Debug, Clone)]
pub struct PreflightReport {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: Option<u32>,
    pub sliding_window: Option<u64>,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub tied_embeddings: bool,
    /// Parameter count implied by the config, in elements.
    pub parameters: u64,
    /// True when `rai convert` would accept this checkpoint.
    pub supported: bool,
    /// Why it would be accepted, or every reason it would not.
    pub reason: String,
    /// `None` when unsupported.
    pub container: Option<PreflightContainer>,
    /// False when only a `config.json` was available, so the three
    /// tensor-level rules (`lm_head.bias`, per-head QK norms, per-layer bias
    /// consistency) went unevaluated and a later conversion could still fail.
    pub weights_checked: bool,
}

/// Answer "would `rai convert` accept this?" without downloading or writing
/// weights.
///
/// This is the converter's own preflight, not a second implementation of it:
/// [`resolve_head_dim`], [`resolve_activation`], [`resolve_rope_scaling`],
/// [`assert_exportable_architecture`], [`resolve_bias_mask`] and
/// [`validate_model_config`] are called in the order `convert` calls them, and
/// the accept/reject answer is theirs. It exists so a user learns a 14 GB
/// checkpoint is unsupported before downloading it, which is why `model_dir`
/// is optional: with no weights on disk, the config-level rules still run.
///
/// `Err` means the config is not a transformer config at all (missing or
/// malformed required fields). An unsupported *architecture* is a successful
/// call with `supported: false`.
pub fn preflight(
    hf: &serde_json::Value,
    model_dir: Option<&Path>,
    group_size: u32,
    embed_group_size: u32,
    max_context: u32,
) -> Result<PreflightReport> {
    let hidden_size = required_u32(hf, "hidden_size")?;
    let num_layers = required_u32(hf, "num_hidden_layers")?;
    let num_heads = required_u32(hf, "num_attention_heads")?;
    let num_kv_heads = optional_u32(hf, "num_key_value_heads")?.unwrap_or(num_heads);
    let head_dim = resolve_head_dim(optional_u32(hf, "head_dim")?, hidden_size, num_heads)?;
    let intermediate_size = required_u32(hf, "intermediate_size")?;
    let vocab_size = required_u32(hf, "vocab_size")?;
    let rope_theta = optional_f64(hf, "rope_theta")?.unwrap_or(10_000.0) as f32;
    let norm_eps = optional_f64(hf, "rms_norm_eps")?.unwrap_or(1e-5) as f32;
    let tied = hf
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let model_type = hf
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    let architectures = hf
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|list| {
            list.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();

    // Elements, not bytes: embedding + per-layer projections + norms + an
    // untied head. The same arithmetic the shape summary is derived from.
    let attention_dim = num_heads as u64 * head_dim as u64;
    let kv_dim = num_kv_heads as u64 * head_dim as u64;
    let hidden = hidden_size as u64;
    let inter = intermediate_size as u64;
    let per_layer =
        hidden * attention_dim * 2 + hidden * kv_dim * 2 + hidden * inter * 3 + hidden * 2;
    let parameters = vocab_size as u64 * hidden
        + num_layers as u64 * per_layer
        + hidden
        + if tied { 0 } else { vocab_size as u64 * hidden };

    let mut report = PreflightReport {
        model_type: model_type.clone(),
        architectures,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings: optional_u32(hf, "max_position_embeddings").ok().flatten(),
        sliding_window: hf.get("sliding_window").and_then(|v| v.as_u64()),
        rope_theta,
        norm_eps,
        tied_embeddings: tied,
        parameters,
        supported: false,
        reason: String::new(),
        container: None,
        weights_checked: false,
    };

    let is_gemma = GEMMA_MODEL_TYPES.contains(&model_type.as_str());
    let store = match model_dir {
        Some(dir) => Some(SafeTensorsSet::open(dir)?),
        None => None,
    };
    report.weights_checked = store.is_some();

    let verdict = (|| -> Result<PreflightContainer> {
        let activation = resolve_activation(hf, is_gemma)?;
        let rope_scaling = resolve_rope_scaling(hf)?;
        assert_exportable_architecture(hf, store.as_ref(), max_context, num_layers)?;
        let bias_mask = match store.as_ref() {
            Some(store) => resolve_bias_mask(store, num_layers)?,
            None => 0,
        };
        let attention_dim = num_heads as usize * head_dim as usize;
        let kv_dim = num_kv_heads as usize * head_dim as usize;
        let qk_norm = match store.as_ref() {
            Some(store) => resolve_qk_norm(store, num_layers, head_dim, attention_dim, kv_dim)?,
            None => QkNormKind::None,
        };
        let has_qk_norm = qk_norm == QkNormKind::PerHead;
        let has_full_qk_norm = qk_norm == QkNormKind::FullWidth;
        let post_norm = match store.as_ref() {
            Some(store) => resolve_post_norm(store, num_layers)?,
            None => false,
        };
        let has_sandwich_norm =
            SANDWICH_NORM_MODEL_TYPES.contains(&model_type.as_str()) && !post_norm;
        let per_layer_rope = resolve_per_layer_rope(hf, num_layers)?;
        if has_sandwich_norm {
            if let Some(store) = store.as_ref() {
                check_sandwich_norms(store, num_layers, hidden_size as usize)?;
            }
        }
        let config = RaiConfig {
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_context,
            rope_theta,
            norm_eps,
            bits: 4,
            group_size: u8::try_from(group_size).map_err(|_| {
                anyhow::anyhow!("--group-size must be in 2..=254, got {group_size}")
            })?,
            embed_bits: 8,
            embed_group_size: u8::try_from(embed_group_size).map_err(|_| {
                anyhow::anyhow!("--embed-group-size must be in 2..=254, got {embed_group_size}")
            })?,
            activation,
            rope_scaling,
            bias_mask,
            embed_scale: if is_gemma {
                (hidden_size as f64).sqrt() as f32
            } else {
                1.0
            },
            has_qk_norm,
            has_sandwich_norm,
            has_full_qk_norm,
            post_norm,
            rope_local_theta: per_layer_rope.0,
            global_layer_stride: per_layer_rope.1,
            attn_logit_softcap: resolve_softcap(hf, "attn_logit_softcapping")?,
            final_logit_softcap: resolve_softcap(hf, "final_logit_softcapping")?,
            attn_scale: resolve_attn_scale(hf, head_dim)?,
        };
        validate_model_config(&config)?;
        // Studio's Check and `rai convert` are documented as running the same
        // preflight, so the fused-projection widths are settled here too: a
        // report saying "supported" that then failed at layer 0 on a shape
        // would be worse than no report.
        if let Some(store) = store.as_ref() {
            let layout = ProjectionLayout::detect(store, num_layers)?;
            layout.check_fused_tensors(store, num_layers, &config.projection_dims())?;
        }
        let (section_sizes, output_bytes) = plan_sections(&config, tied)?;
        Ok(PreflightContainer {
            version: config.version(),
            activation,
            rope_scaling,
            bias_mask,
            embed_scale: config.embed_scale,
            tied_embeddings: tied,
            output_bytes,
            num_sections: section_sizes.len(),
            has_qk_norm,
            has_sandwich_norm,
            has_full_qk_norm,
            post_norm,
            rope_local_theta: per_layer_rope.0,
            global_layer_stride: per_layer_rope.1,
            attn_logit_softcap: config.attn_logit_softcap,
            final_logit_softcap: config.final_logit_softcap,
            attn_scale: config.attn_scale,
        })
    })();

    match verdict {
        Ok(container) => {
            report.reason = format!(
                "`rai convert` accepts this checkpoint: model_type '{model_type}' converts to a \
                 container v{} file of {} bytes ({} sections, activation {:?}, rope {:?}, \
                 bias_mask {:#04x}, qk_norm {}, sandwich_norm {}, softcap attn {} / final {}, \
                 {} lm_head).{}",
                container.version,
                container.output_bytes,
                container.num_sections,
                container.activation,
                container.rope_scaling,
                container.bias_mask,
                container.has_qk_norm,
                container.has_sandwich_norm,
                container.attn_logit_softcap,
                container.final_logit_softcap,
                if tied { "tied" } else { "untied" },
                if report.weights_checked {
                    ""
                } else {
                    " Config-level rules only: no weights were available, so per-tensor rules \
                      (lm_head bias, per-head QK norm shape and consistency, sandwich norms, \
                      per-layer bias consistency) are still unchecked."
                }
            );
            report.supported = true;
            report.container = Some(container);
        }
        Err(error) => {
            report.reason = format!("{error:#}");
            report.supported = false;
        }
    }
    Ok(report)
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
    /// First row of `tensor` to read. Non-zero only when the projection is a
    /// slice of a fused tensor; see [`ProjectionLayout`].
    source_row_offset: usize,
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
        store.read_rows(
            job.tensor,
            job.source_row_offset + row_start,
            block_rows,
            &mut weights,
        )?;
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

/// Write one unquantized vector (an RMSNorm weight or a projection bias) as
/// little-endian f32.
///
/// `plus_one` implements the Gemma RMSNorm fold: Gemma's norm is
/// `x/rms * (1 + w)` while `layers.rs::rms_norm` computes `x/rms * w`, so
/// storing `1 + w` makes the shipped kernel exactly right and costs the format
/// nothing. The addition happens in f32 after the checkpoint's f16 cast, which
/// is the same order `(1.0 + weight.float())` uses in the reference model.
fn write_f32_vector(
    file: &mut File,
    store: &mut SafeTensorsSet,
    tensor: &str,
    len: usize,
    plus_one: bool,
) -> Result<()> {
    let weights = store.read_all(tensor, len)?;
    if weights.len() != len {
        bail!(
            "tensor '{tensor}' has {} values, expected {len}",
            weights.len()
        );
    }
    let mut bytes = Vec::with_capacity(len * 4);
    for (index, value) in weights.iter().enumerate() {
        if !value.is_finite() {
            bail!("{tensor} weight {index} is non-finite");
        }
        let value = if plus_one { 1.0 + *value } else { *value };
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

/// Section sizes and the resulting file size, derived from the config alone.
///
/// Shared with [`preflight`] so a caller can be told how large the output will
/// be before a byte is written, without a second copy of the layout rules.
fn plan_sections(config: &RaiConfig, tied: bool) -> Result<(Vec<u64>, u64)> {
    let hidden = config.hidden_size as usize;
    let vocab = config.vocab_size as usize;
    let group_size = config.group_size as usize;
    let embed_group_size = config.embed_group_size as usize;
    let linear_dims = config.projection_dims();

    let mut section_sizes: Vec<u64> = Vec::with_capacity(config.num_layers as usize + 3);
    section_sizes.push(embedding_section_len(vocab, hidden, embed_group_size)?);
    let layer_len = layer_section_len(&linear_dims, hidden, group_size, config)?;
    for _ in 0..config.num_layers {
        section_sizes.push(layer_len);
    }
    section_sizes.push(hidden as u64 * 4);
    if !tied {
        section_sizes.push(linear_section_len(vocab, hidden, group_size)?);
    }

    let total = config.header_size()
        + section_sizes.len() as u64 * SECTION_ENTRY_SIZE
        + section_sizes.iter().sum::<u64>();
    Ok((section_sizes, total))
}

/// The exact byte length of one layer section, block by block, in the order
/// `format::RaiModelFile::layer` parses them. The reader checks this size
/// exactly, so any disagreement here fails the export rather than corrupting it.
fn layer_section_len(
    linear_dims: &[(usize, usize); 7],
    hidden: usize,
    group_size: usize,
    config: &RaiConfig,
) -> Result<u64> {
    let mut total = 0u64;
    for &(rows, cols) in linear_dims {
        total += linear_section_len(rows, cols, group_size)?;
    }
    // Bias block: one f32 per output row, for each declared projection.
    for (index, &(rows, _)) in linear_dims.iter().enumerate() {
        if config.bias_mask & (1 << index) != 0 {
            total += rows as u64 * 4;
        }
    }
    // QK-norm block: q_norm + k_norm. Per-head is head_dim f32 each;
    // full-width is the two projection widths, which differ under GQA.
    if config.has_qk_norm {
        total += 2 * config.head_dim as u64 * 4;
    } else if config.has_full_qk_norm {
        total += (config.attention_dim() + config.kv_dim()) as u64 * 4;
    }
    // Sandwich block: attention-output + MLP-output norms, hidden f32 each.
    if config.has_sandwich_norm {
        total += 2 * hidden as u64 * 4;
    }
    // The two hidden-sized layer norms that every version carries.
    Ok(total + 2 * hidden as u64 * 4)
}

fn write_header(file: &mut File, config: &RaiConfig, num_sections: u32) -> Result<()> {
    let version = config.version();
    let mut header = vec![0u8; config.header_size() as usize];
    header[0..4].copy_from_slice(b"RAIM");
    header[4..8].copy_from_slice(&version.to_le_bytes());
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
    // 56..64 stays zero in both versions: it is the v1 header's reserved tail
    // and the v2 reader rejects a non-zero value there.
    if version >= 2 {
        header[64] = config.activation.code();
        header[65] = config.flags();
        header[66] = config.rope_scaling.code();
        header[67] = config.bias_mask;
        if let RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        } = config.rope_scaling
        {
            header[68..72].copy_from_slice(&factor.to_le_bytes());
            header[72..76].copy_from_slice(&low_freq_factor.to_le_bytes());
            header[76..80].copy_from_slice(&high_freq_factor.to_le_bytes());
            header[80..84].copy_from_slice(&original_max_position.to_le_bytes());
        }
        header[84..88].copy_from_slice(&config.embed_scale.to_le_bytes());
        header[88..92].copy_from_slice(&config.attn_logit_softcap.to_le_bytes());
        header[92..96].copy_from_slice(&config.final_logit_softcap.to_le_bytes());
        header[96..100].copy_from_slice(&config.attn_scale.to_le_bytes());
        header[100..104].copy_from_slice(&config.rope_local_theta.to_le_bytes());
        header[104] = config.global_layer_stride;
        // 105..128 stays zero: the reader rejects a non-zero value there.
    }
    file.write_all(&header)?;
    Ok(())
}

// =============================================================================
// Helpers
// =============================================================================

/// How a checkpoint lays out the seven projections a layer stores.
///
/// Most families give each projection its own tensor. Phi-3 concatenates Q, K
/// and V into one `qkv_proj` and gate/up into one `gate_up_proj`, so the same
/// seven matrices are row ranges of two larger ones. Nothing about the
/// container changes: the quantizer already reads a tensor by row range, so a
/// fused checkpoint is split during conversion and the resulting file is
/// indistinguishable from one written from separate tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectionLayout {
    /// `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` each stand alone.
    Separate,
    /// `qkv_proj` holds Q then K then V; `gate_up_proj` holds gate then up.
    FusedQkvGateUp,
}

/// Which tensor a projection lives in, and the row it starts at.
struct ProjectionSource {
    tensor: String,
    row_offset: usize,
}

impl ProjectionLayout {
    /// Decide the layout from the tensors that are actually present.
    ///
    /// Detection is by tensor, not by `model_type`: a fine-tune that renames
    /// itself still has to say where its weights are, and a family we have
    /// never seen that happens to fuse the same way converts for free.
    fn detect(store: &SafeTensorsSet, num_layers: u32) -> Result<Self> {
        let fused = |layer: u32| {
            store
                .info(&format!("model.layers.{layer}.self_attn.qkv_proj.weight"))
                .is_some()
        };
        if num_layers == 0 || !fused(0) {
            return Ok(Self::Separate);
        }
        // A checkpoint that fuses only some layers would convert the rest from
        // tensors that are not there; say so here rather than at layer 12.
        for layer in 1..num_layers {
            if !fused(layer) {
                bail!(
                    "checkpoint is inconsistent: layer 0 fuses Q/K/V into qkv_proj but layer \
                     {layer} does not. Every layer must use the same projection layout."
                );
            }
        }
        Ok(Self::FusedQkvGateUp)
    }

    /// Where projection `index` of `layer` is read from.
    ///
    /// `index` is a position in [`LAYER_LINEAR_NAMES`]: q, k, v, o, gate, up,
    /// down. The row offsets follow `Phi3Attention.forward`, which slices the
    /// fused output as Q, then K, then V, and `Phi3MLP.forward`, which splits
    /// the fused MLP output into gate then up.
    fn source(self, layer: u32, index: usize, dims: &[(usize, usize); 7]) -> ProjectionSource {
        let attn = |offset: usize| ProjectionSource {
            tensor: format!("model.layers.{layer}.self_attn.qkv_proj.weight"),
            row_offset: offset,
        };
        let mlp = |offset: usize| ProjectionSource {
            tensor: format!("model.layers.{layer}.mlp.gate_up_proj.weight"),
            row_offset: offset,
        };
        let separate = || ProjectionSource {
            tensor: layer_linear_name(layer, LAYER_LINEAR_NAMES[index]),
            row_offset: 0,
        };
        match (self, index) {
            (Self::Separate, _) => separate(),
            (Self::FusedQkvGateUp, 0) => attn(0),
            (Self::FusedQkvGateUp, 1) => attn(dims[0].0),
            (Self::FusedQkvGateUp, 2) => attn(dims[0].0 + dims[1].0),
            (Self::FusedQkvGateUp, 4) => mlp(0),
            (Self::FusedQkvGateUp, 5) => mlp(dims[4].0),
            // `o_proj` and `down_proj` are never fused: they project back down,
            // so there is nothing to concatenate them with.
            (Self::FusedQkvGateUp, _) => separate(),
        }
    }

    /// Confirm the fused tensors are exactly as wide as the parts they hold.
    ///
    /// Without this a config that disagrees with the checkpoint would silently
    /// read V's rows as K's — a file that loads cleanly and generates nonsense,
    /// which is the failure this converter exists to prevent.
    fn check_fused_tensors(
        self,
        store: &SafeTensorsSet,
        num_layers: u32,
        dims: &[(usize, usize); 7],
    ) -> Result<()> {
        if self != Self::FusedQkvGateUp {
            return Ok(());
        }
        let hidden = dims[0].1;
        let qkv_rows = dims[0].0 + dims[1].0 + dims[2].0;
        let gate_up_rows = dims[4].0 + dims[5].0;
        for layer in 0..num_layers {
            let qkv = format!("model.layers.{layer}.self_attn.qkv_proj.weight");
            let gate_up = format!("model.layers.{layer}.mlp.gate_up_proj.weight");
            check_tensor(store, &qkv, qkv_rows, hidden)?;
            check_tensor(store, &gate_up, gate_up_rows, hidden)?;
            // The bias mask names the seven projections, so a bias on a fused
            // tensor has no slot and would be dropped without a word.
            for name in [&qkv, &gate_up] {
                let bias = name.replace(".weight", ".bias");
                if store.info(&bias).is_some() {
                    bail!(
                        "checkpoint carries '{bias}'; biases are stored per projection and a \
                         fused tensor has no slot for one, so it would be silently dropped."
                    );
                }
            }
        }
        Ok(())
    }
}

fn layer_linear_name(layer: u32, projection: &str) -> String {
    format!(
        "model.layers.{layer}.{}.{projection}.weight",
        projection_block(projection)
    )
}

fn layer_bias_name(layer: u32, projection: &str) -> String {
    format!(
        "model.layers.{layer}.{}.{projection}.bias",
        projection_block(projection)
    )
}

fn projection_block(projection: &str) -> &'static str {
    match projection {
        "gate_proj" | "up_proj" | "down_proj" => "mlp",
        _ => "self_attn",
    }
}

/// `<checkpoint-dir-name>-q4.raimodel`, in the checkpoint's own case.
///
/// The case is preserved deliberately. This used to lowercase the directory
/// name, which is harmless on Windows and macOS — where the filesystem matches
/// case-insensitively — and a trap on Linux: converting `Qwen2.5-0.5B-Instruct`
/// wrote `qwen2.5-0.5b-instruct-q4.raimodel`, so the obvious next command,
/// naming the model after the directory it came from, failed with "no such
/// file" on exactly the platform where the user could not see why.
///
/// `to_string_lossy` is still the fallback for a non-UTF-8 directory name,
/// which Linux permits: replacement characters in a filename the user can see
/// and retype beat refusing to convert at all, and `--output` overrides it.
fn default_output_name(model_dir: &Path) -> Result<String> {
    let name = model_dir
        .canonicalize()
        .unwrap_or_else(|_| model_dir.to_path_buf())
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
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
    fn head_dim_is_taken_from_the_config_when_declared() {
        assert_eq!(resolve_head_dim(None, 64, 4).unwrap(), 16);
        assert_eq!(resolve_head_dim(Some(16), 64, 4).unwrap(), 16);
        // Decoupled: 4 * 32 = 128 against a 64-wide hidden state. Gemma does
        // this, and the container now stores it.
        assert_eq!(resolve_head_dim(Some(32), 64, 4).unwrap(), 32);
        // Without an explicit head_dim the division must be exact.
        let error = resolve_head_dim(None, 66, 4).unwrap_err().to_string();
        assert!(error.contains("not divisible"), "{error}");
    }

    #[test]
    fn activation_is_resolved_per_family() {
        let gemma = serde_json::json!({
            "model_type": "gemma", "hidden_act": "gelu", "hidden_activation": null
        });
        assert_eq!(
            resolve_activation(&gemma, true).unwrap(),
            Activation::GeluTanh
        );
        let gemma_explicit = serde_json::json!({ "hidden_activation": "gelu_pytorch_tanh" });
        assert_eq!(
            resolve_activation(&gemma_explicit, true).unwrap(),
            Activation::GeluTanh
        );
        let llama = serde_json::json!({ "hidden_act": "silu" });
        assert_eq!(resolve_activation(&llama, false).unwrap(), Activation::Silu);
        assert_eq!(
            resolve_activation(&serde_json::json!({}), false).unwrap(),
            Activation::Silu
        );
        // An exact-erf GeLU is a different function and must not be run as the
        // tanh approximation.
        let exact = serde_json::json!({ "hidden_activation": "gelu" });
        let error = resolve_activation(&exact, true).unwrap_err().to_string();
        assert!(error.contains("hidden activation 'gelu'"), "{error}");
    }

    #[test]
    fn rope_scaling_is_resolved_or_refused_by_name() {
        assert_eq!(
            resolve_rope_scaling(&serde_json::json!({})).unwrap(),
            RopeScaling::None
        );
        assert_eq!(
            resolve_rope_scaling(&serde_json::json!({ "rope_scaling": null })).unwrap(),
            RopeScaling::None
        );
        assert_eq!(
            resolve_rope_scaling(
                &serde_json::json!({ "rope_scaling": { "rope_type": "default" } })
            )
            .unwrap(),
            RopeScaling::None
        );
        let llama3 = serde_json::json!({ "rope_scaling": {
            "rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192
        }});
        assert_eq!(
            resolve_rope_scaling(&llama3).unwrap(),
            RopeScaling::Llama3 {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position: 8_192,
            }
        );
        let yarn = serde_json::json!({ "rope_scaling": { "rope_type": "yarn", "factor": 4.0 } });
        let error = resolve_rope_scaling(&yarn).unwrap_err().to_string();
        assert!(error.contains("rope_scaling type 'yarn'"), "{error}");
    }

    #[test]
    fn container_version_is_the_lowest_that_fits() {
        let mut config = RaiConfig {
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 96,
            max_context: 512,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            bits: 4,
            group_size: 64,
            embed_bits: 8,
            embed_group_size: 64,
            activation: Activation::Silu,
            rope_scaling: RopeScaling::None,
            bias_mask: 0,
            embed_scale: 1.0,
            has_qk_norm: false,
            has_sandwich_norm: false,
            has_full_qk_norm: false,
            post_norm: false,
            rope_local_theta: 0.0,
            global_layer_stride: 0,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            attn_scale: 0.0,
        };
        assert_eq!(config.version(), 1, "a plain Llama model must stay v1");
        assert_eq!(config.header_size(), HEADER_SIZE_V1);

        for mutate in [
            (|c: &mut RaiConfig| c.activation = Activation::GeluTanh) as fn(&mut RaiConfig),
            |c: &mut RaiConfig| c.bias_mask = 0b111,
            |c: &mut RaiConfig| c.embed_scale = 45.25,
            |c: &mut RaiConfig| c.has_qk_norm = true,
            |c: &mut RaiConfig| c.has_sandwich_norm = true,
            |c: &mut RaiConfig| c.attn_logit_softcap = 50.0,
            |c: &mut RaiConfig| c.final_logit_softcap = 30.0,
            |c: &mut RaiConfig| c.attn_scale = 0.083_333_336,
            |c: &mut RaiConfig| {
                c.rope_scaling = RopeScaling::Llama3 {
                    factor: 32.0,
                    low_freq_factor: 1.0,
                    high_freq_factor: 4.0,
                    original_max_position: 8_192,
                }
            },
        ] {
            let mut probe = config.clone();
            mutate(&mut probe);
            assert_eq!(probe.version(), 2);
            assert_eq!(probe.header_size(), HEADER_SIZE_V2);
        }
        config.bias_mask = 0b111;
        assert_eq!(config.version(), 2);
    }

    #[test]
    fn optional_block_sizing_matches_the_reader() {
        let dims: [(usize, usize); 7] = [
            (64, 64),
            (32, 64),
            (32, 64),
            (64, 64),
            (128, 64),
            (128, 64),
            (64, 128),
        ];
        let base = RaiConfig {
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 96,
            max_context: 512,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            bits: 4,
            group_size: 64,
            embed_bits: 8,
            embed_group_size: 64,
            activation: Activation::Silu,
            rope_scaling: RopeScaling::None,
            bias_mask: 0,
            embed_scale: 1.0,
            has_qk_norm: false,
            has_sandwich_norm: false,
            has_full_qk_norm: false,
            post_norm: false,
            rope_local_theta: 0.0,
            global_layer_stride: 0,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            attn_scale: 0.0,
        };
        let with = |mutate: fn(&mut RaiConfig)| {
            let mut config = base.clone();
            mutate(&mut config);
            layer_section_len(&dims, 64, 64, &config).unwrap()
        };
        let plain = with(|_| {});
        // q, k, v biases: (64 + 32 + 32) * 4 bytes.
        assert_eq!(with(|c| c.bias_mask = 0b111) - plain, (64 + 32 + 32) * 4);
        // QK norm: two head_dim vectors.
        assert_eq!(with(|c| c.has_qk_norm = true) - plain, 2 * 16 * 4);
        // Sandwich norm: two hidden vectors.
        assert_eq!(with(|c| c.has_sandwich_norm = true) - plain, 2 * 64 * 4);
    }

    #[test]
    fn softcaps_are_read_or_refused_by_name() {
        let none = serde_json::json!({});
        assert_eq!(
            resolve_softcap(&none, "attn_logit_softcapping").unwrap(),
            0.0
        );
        let null = serde_json::json!({ "final_logit_softcapping": serde_json::Value::Null });
        assert_eq!(
            resolve_softcap(&null, "final_logit_softcapping").unwrap(),
            0.0
        );
        let gemma2 = serde_json::json!({
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
        });
        assert_eq!(
            resolve_softcap(&gemma2, "attn_logit_softcapping").unwrap(),
            50.0
        );
        assert_eq!(
            resolve_softcap(&gemma2, "final_logit_softcapping").unwrap(),
            30.0
        );
        let bad = serde_json::json!({ "attn_logit_softcapping": -1.0 });
        let error = resolve_softcap(&bad, "attn_logit_softcapping")
            .unwrap_err()
            .to_string();
        assert!(error.contains("must be finite and positive"), "{error}");
    }

    #[test]
    fn the_attention_scale_is_stored_only_when_it_differs() {
        // gemma-2-2b: query_pre_attn_scalar 256 against head_dim 256 is exactly
        // the default, so nothing is stored and the file stays smaller.
        let same = serde_json::json!({ "query_pre_attn_scalar": 256 });
        assert_eq!(resolve_attn_scale(&same, 256).unwrap(), 0.0);
        // gemma-2-27b: 144 against head_dim 128 is a genuinely different scale.
        let differs = serde_json::json!({ "query_pre_attn_scalar": 144 });
        let scale = resolve_attn_scale(&differs, 128).unwrap();
        assert_eq!(scale, (144.0f64).powf(-0.5) as f32);
        assert_ne!(scale, 1.0 / (128.0f32).sqrt());
        // Absent means the default.
        assert_eq!(
            resolve_attn_scale(&serde_json::json!({}), 128).unwrap(),
            0.0
        );
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

    /// The default output name keeps the checkpoint directory's own case.
    ///
    /// Lowercasing it is invisible on Windows and macOS and breaks on Linux,
    /// where `Qwen2.5-0.5B-Instruct-q4.raimodel` and
    /// `qwen2.5-0.5b-instruct-q4.raimodel` are two different files and only one
    /// of them exists. The directory here is never created, so `canonicalize`
    /// fails and the un-canonicalized path is used — which is also the branch a
    /// relative `--model` argument takes.
    #[test]
    fn the_default_output_name_preserves_the_checkpoint_case() {
        assert_eq!(
            default_output_name(Path::new("models/Qwen2.5-0.5B-Instruct")).unwrap(),
            "Qwen2.5-0.5B-Instruct-q4.raimodel"
        );
        assert_eq!(
            default_output_name(Path::new("TinyLlama-1.1B-Chat-v1.0")).unwrap(),
            "TinyLlama-1.1B-Chat-v1.0-q4.raimodel"
        );
    }

    #[test]
    fn a_nameless_model_dir_asks_for_an_explicit_output() {
        // A bare root has no final component to name the output after.
        let error = default_output_name(Path::new("/")).unwrap_err().to_string();
        assert!(error.contains("--output"), "{error}");
    }
}
