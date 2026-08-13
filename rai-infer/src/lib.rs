//! CPU inference for 4-bit quantized Llama-family language models.
//!
//! RAI loads a `.raimodel` file — a single flat binary holding an 8-bit
//! embedding table, 4-bit transformer weights, and their quantization
//! parameters — and generates text with hand-written AVX2/FMA/F16C kernels. No
//! GPU, no CUDA, no BLAS, and no Python runtime is involved at any point.
//!
//! ```no_run
//! use std::path::Path;
//! use rai_infer::model::RaiModel;
//!
//! let model = RaiModel::load(Path::new("tinyllama-q4.raimodel"))?;
//! let cache = model.create_kv_cache(512)?;
//! println!("{} layers, vocab {}", model.config.num_layers, model.config.vocab_size);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! # Layout
//!
//! The reading path runs [`format`] → [`model`] → [`layers`] + [`gemm`]:
//! [`format`] validates a file and hands out borrowed views into one heap
//! buffer, [`model`] drives the forward pass over those views, and [`layers`]
//! and [`gemm`] are the kernels underneath. [`kv_cache`] holds attention state
//! across steps and [`sampler`] turns logits into a token.
//!
//! Three optional decoding strategies sit on top: [`speculative`] (a small
//! draft model proposes, the target verifies), [`lookup`] (the draft is copied
//! out of the context, so there is no draft model at all), and
//! [`self_speculative`] (the model's own first layers act as the draft —
//! measured as a slowdown without a trained exit head, so it has no CLI
//! surface). [`ponder`] implements the test-time-compute strategies.
//!
//! # Features
//!
//! The default `cli` feature pulls in `clap`, `tokenizers`, `tiny_http`,
//! `serde`, `serde_json`, and `memmap2`, and enables [`cli`], [`convert`],
//! [`safetensors`], and the tokenizer-aware [`chat_template`] helpers.
//! Building with `--no-default-features` leaves the inference library — format
//! reader, kernels, model, sampling, and all three speculative decoders —
//! resting on `half`, `rayon`, `anyhow`, and `rand` alone.
//!
//! # Panics
//!
//! Loading and generation return [`anyhow::Result`]; a malformed or hostile
//! `.raimodel` produces an error, not a panic. The kernels do assert their
//! shape invariants, which `.raimodel` validation establishes before any
//! kernel runs. Every such entry point documents the conditions under
//! `# Panics`.

/// Prompt formatting for instruction-tuned models.
pub mod chat_template;
/// The `rai` command-line surface: `convert`, `run`, `serve`, and `models`.
#[cfg(feature = "cli")]
pub mod cli;
/// HuggingFace checkpoint to `.raimodel` conversion, without Python.
#[cfg(feature = "cli")]
pub mod convert;
/// The `.raimodel` container: header validation, section bounds, borrowed views.
pub mod format;
/// W4A8 GEMM kernels — AVX2/FMA with scalar fallbacks.
pub mod gemm;
/// Key/value attention cache.
pub mod kv_cache;
/// Transformer building blocks: RMSNorm, RoPE, attention, gated MLP.
pub mod layers;
/// Prompt-lookup speculative decoding: the draft is copied from the context.
pub mod lookup;
/// The loaded model and its forward passes.
pub mod model;
/// Test-time compute strategies (CFG, ensembling, adaptive).
pub mod ponder;
/// Streaming `.safetensors` reader used by [`convert`].
#[cfg(feature = "cli")]
pub mod safetensors;
/// Turning logits into a token: temperature, top-k, top-p, repetition penalty.
pub mod sampler;
/// Self-speculative decoding: the model's own first layers draft for it.
pub mod self_speculative;
/// Draft-model speculative decoding with target-model verification.
pub mod speculative;
