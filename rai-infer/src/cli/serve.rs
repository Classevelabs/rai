//! `rai serve` — the local HTTP API, and the chat UI on top of it.
//!
//! This is the body of the old `rai-chat` binary, moved into the library so
//! that one `rai` binary can host it. The server binds loopback only and
//! rejects any request whose `Host` header is not a loopback name at
//! the bound port, which is what stops a web page in the user's browser from
//! driving the local model via DNS rebinding. Every `POST` additionally
//! requires a same-origin `Origin` (when the browser sends one) and a JSON
//! `Content-Type`, so no endpoint here is reachable cross-origin — including
//! the ones that load models and start conversions.
//!
//! The API surface, all JSON in and JSON out:
//!
//! | Route | What it does |
//! |---|---|
//! | `GET /api/info` | the loaded model, or `loaded: false` |
//! | `GET /api/system` | what this machine is actually running on |
//! | `POST /api/chat` | generate a reply |
//! | `POST /api/chat/stream` | the same reply, a token at a time (ndjson) |
//! | `GET /api/models` | `.raimodel` files in a directory, from headers only |
//! | `POST /api/load` | load a model into this server |
//! | `POST /api/inspect` | would `rai convert` accept this checkpoint? |
//! | `POST /api/convert` | start a conversion, return a job id |
//! | `GET /api/convert/<id>` | poll that job |
//! | `GET /api/convert` | every job this server has run |
//!
//! `POST /api/chat/stream` is the only route that does not answer with one
//! JSON body. It answers `application/x-ndjson` over chunked transfer
//! encoding, one JSON object per line, each flushed to the socket as it is
//! produced. It passes exactly the same `Host`, `Origin`, `Content-Type` and
//! body-size checks as every other `POST` — those run before routing, so no
//! route can be added that skips them — and the page's own
//! `connect-src 'self'` allows it, because it is a same-origin `fetch` to the
//! very origin that served the page.
//!
//! A client keeps its connection open until the reply ends. One that closes
//! it, or only shuts its sending side after the request (`nc -N`), stops the
//! generation: from the socket, the two look the same as a client that left.
//!
//! Paths in requests are the user's own filesystem paths and are not confined
//! to a root: this server is the local application's own back end, reachable
//! only from the machine it runs on, and it can do exactly what the `rai`
//! command line can do for the user running it — no more.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::cli::http::{Body, ChunkedWriter, HttpServer, Method, Request, Response};
use anyhow::{Context, Result};
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::chat_format::ChatFormat;
use crate::chat_template::{ChatTemplate, Role};
use crate::cli::catalog;
use crate::cli::jobs::{Jobs, StartError};
use crate::cli::run::incremental_suffix;
use crate::cli::{load_tokenizer_for_model, resolve_tokenizer};
use crate::convert::ConvertOptions;
use crate::kv_cache::KVCache;
use crate::model::{BatchScratch, InferenceWork, RaiModel};
use crate::ponder::{pondered_forward, PonderConfig};
use crate::sampler::{apply_repetition_penalty, sample_token, SamplerConfig};

const MAX_CHAT_REQUEST_BYTES: usize = 64 * 1024;
/// Tokens generated for a request that does not ask for a number.
///
/// This is a *default*, not a ceiling. The ceiling is whatever the prompt
/// leaves free in the loaded model's context window — see
/// [`resolve_max_tokens`] — because a fixed one silently truncated every reply
/// on a model with a 40k window.
const DEFAULT_CHAT_GENERATION_TOKENS: usize = 200;
const MAX_ENSEMBLE_SIZE: usize = 8;
const MAX_ENTROPY_THRESHOLD: f32 = 32.0;
/// Directories one `GET /api/models` will scan. Only ever 1-2 today; the cap
/// exists so the defaulting rule can never turn into an unbounded walk.
const MAX_SCANNED_DIRECTORIES: usize = 8;

#[derive(Debug)]
struct ChatHttpError {
    status: u16,
    message: String,
}

impl ChatHttpError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }

    fn payload_too_large() -> Self {
        Self {
            status: 413,
            message: format!("request body exceeds the {MAX_CHAT_REQUEST_BYTES}-byte limit"),
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: 403,
            message: message.into(),
        }
    }

    fn unsupported_media_type() -> Self {
        Self {
            status: 415,
            message: "Content-Type must be application/json".to_string(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            message: message.into(),
        }
    }

    /// The request is well formed but the server is not in a state to serve
    /// it — no model loaded, or a conversion already running.
    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: 409,
            message: message.into(),
        }
    }

    /// Something the API deliberately does not do, said plainly rather than
    /// failed vaguely.
    fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: 501,
            message: message.into(),
        }
    }

    /// Chunked transfer encoding is an HTTP/1.1 feature, so the streaming
    /// route cannot answer an HTTP/1.0 client at all. Saying which route to
    /// use instead is more useful than a truncated stream.
    fn http_version_not_supported() -> Self {
        Self {
            status: 505,
            message: "streaming responses require HTTP/1.1; use POST /api/chat instead".to_string(),
        }
    }

    fn internal(error: impl fmt::Display) -> Self {
        eprintln!("chat request failed: {error}");
        Self {
            status: 500,
            message: "internal server error".to_string(),
        }
    }
}

impl fmt::Display for ChatHttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// Everything except the model and tokenizer paths, so that `rai serve` (model
/// as a positional, tokenizer optional) and the deprecated `rai-chat`
/// (`--model` / `--tokenizer`, both required) share one definition.
#[derive(clap::Args, Debug, Clone)]
pub struct ServeOptions {
    /// Port to listen on (loopback only)
    #[arg(long, default_value = "8090")]
    pub port: u16,
    /// Context window to allocate the KV cache for, in tokens. Defaults to the
    /// context the model was stored with, reduced to fit free memory when it
    /// would not (a note says so); a smaller value trades history for memory,
    /// a larger one is clamped to what the model supports.
    #[arg(long)]
    pub max_context: Option<usize>,
    /// Chat template: auto (the model's own, recorded at conversion), model,
    /// none, few-shot, mistral, llama3, chatml, zephyr, phi3, gemma
    #[arg(long, default_value = "auto")]
    pub chat_template: String,
}

/// Refuse a `--chat-template` that names no template, before anything reads
/// a model with it.
fn check_template_name(chat_template: &str) -> Result<()> {
    anyhow::ensure!(
        matches!(chat_template, "auto" | "model")
            || ChatTemplate::from_name(chat_template).is_some(),
        "unknown chat template '{chat_template}'; expected auto, model, or one of {}",
        ChatTemplate::NAMES.join(", ")
    );
    Ok(())
}

#[derive(clap::Args, Debug, Clone)]
pub struct ServeArgs {
    /// The .raimodel file to serve; omit to start empty and load one over the
    /// API
    #[arg(value_name = "MODEL")]
    pub model: Option<PathBuf>,
    /// Path to tokenizer.json; defaults to the one beside the model
    #[arg(long, value_name = "FILE")]
    pub tokenizer: Option<PathBuf>,
    #[command(flatten)]
    pub options: ServeOptions,
}

/// A model this server can generate with.
struct Loaded {
    model: RaiModel,
    tokenizer: tokenizers::Tokenizer,
    max_context: usize,
    format: ChatFormat,
    path: PathBuf,
    tokenizer_path: PathBuf,
    load_ms: u64,
    /// Allocated once here rather than per request.
    ///
    /// Two reasons, both of which became load-bearing when the context window
    /// started following the model instead of a constant 512: a 40k-token
    /// window on a mid-sized model is hundreds of megabytes that would
    /// otherwise be allocated and zeroed on every single message, and — more
    /// importantly — allocating it here is what turns "this context does not
    /// fit in memory" into an error at load time, naming the size, instead of
    /// a failure on the user's first message.
    ///
    /// Reuse is safe because every generation truncates the watermark to the
    /// prefix it shares with [`Self::cached_tokens`] and the cache refuses
    /// reads above it: nothing outside that exact prefix can be attended to.
    kv_cache: KVCache,
    /// The tokens whose keys and values fill `kv_cache` positions
    /// `0..cached_tokens.len()`, left behind by the last finished generation.
    ///
    /// A chat client re-sends the whole thread every turn, so consecutive
    /// requests share everything up to the newest message. Reusing that prefix
    /// means each turn prefills only what is new instead of re-reading the
    /// whole conversation. Cleared before any generation touches the cache,
    /// and set again only when one finishes, so an aborted or failed request
    /// can never leave it describing entries it did not write.
    cached_tokens: Vec<usize>,
}

/// Everything the request loop owns.
///
/// The loop is single-threaded, so the loaded model needs no lock: a request
/// is served to completion before the next is read, and `POST /api/load`
/// therefore swaps the model at a point where no other request is in flight.
/// Conversions are the exception — they run on their own threads and share
/// [`Jobs`], which carries its own locks.
struct ServerState {
    loaded: Option<Loaded>,
    jobs: Arc<Jobs>,
    options: ServeOptions,
}

impl Loaded {
    fn open(
        model_path: &Path,
        tokenizer_arg: Option<&Path>,
        requested_context: Option<usize>,
        chat_template: &str,
    ) -> Result<Self> {
        anyhow::ensure!(
            !matches!(requested_context, Some(0)),
            "max_context must be greater than zero"
        );
        anyhow::ensure!(
            model_path.is_file(),
            "{} is not a file",
            model_path.display()
        );
        // Refused before the weights are read, not after.
        check_template_name(chat_template)?;
        let started = Instant::now();
        let tokenizer_path = resolve_tokenizer(model_path, tokenizer_arg)?;
        let model = RaiModel::load(model_path)?;
        let choice = choose_context_window(
            requested_context,
            model.config.max_context as usize,
            |ctx| model.kv_cache_bytes(ctx),
        )?;
        if let Some(note) = context_choice_note(choice, |ctx| model.kv_cache_bytes(ctx)) {
            eprintln!("  {note}");
        }
        let max_context = choice.context;
        let kv_bytes = model.kv_cache_bytes(max_context);
        // Still checked: an explicit request skips the auto-fit entirely, and
        // this is where such a request is refused by name.
        check_kv_cache_fits(kv_bytes, max_context)?;
        let kv_cache = model.create_kv_cache(max_context).with_context(|| {
            format!(
                "a {max_context}-token context needs a {} KV cache",
                crate::cli::format_bytes(kv_bytes as u64)
            )
        })?;
        let tokenizer =
            load_tokenizer_for_model(&tokenizer_path, model.config.vocab_size as usize)?;
        let format = ChatFormat::load(chat_template, model_path, &tokenizer)?;
        Ok(Self {
            model,
            tokenizer,
            max_context,
            format,
            path: model_path.to_path_buf(),
            tokenizer_path,
            load_ms: started.elapsed().as_millis() as u64,
            kv_cache,
            cached_tokens: Vec::new(),
        })
    }

    /// The `GET /api/info` body. The field names predate the rest of this API
    /// and are kept exactly as they were.
    fn info_json(&self) -> Value {
        let config = &self.model.config;
        json!({
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "chat_template": self.format.name(),
            "weights_mb": self.model.file_size() as f64 / (1024.0 * 1024.0),
            // Added for the model-picker UI; everything above is unchanged.
            "loaded": true,
            "model_path": self.path.display().to_string(),
            "model_name": self.path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned()),
            "tokenizer_path": self.tokenizer_path.display().to_string(),
            "max_context": self.max_context,
            "model_max_context": config.max_context,
            "head_dim": config.head_dim,
            "format_version": config.version,
            "activation": catalog::activation_name(config.activation),
            "rope_type": catalog::rope_type_name(config.rope_scaling),
            "has_biases": config.bias_mask != 0,
            "biased_projections": catalog::biased_projections(config.bias_mask),
            "tied_lm_head": !self.model.has_separate_lm_head,
            "size_bytes": self.model.file_size(),
            // The cache that actually exists, not an estimate of one.
            "kv_cache_bytes": self.kv_cache.memory_bytes(),
            "load_ms": self.load_ms,
        })
    }
}

/// One prior turn, as the client stores it.
#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    /// Single-message form — the original wire contract, still accepted so
    /// existing scripts keep working.
    message: Option<String>,
    /// Whole-thread form: ordered oldest first, ending with the user message
    /// being answered. The server re-renders the thread through the model's
    /// chat template every request; it keeps no conversation state of its own.
    messages: Option<Vec<ChatMessage>>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    /// Penalty on tokens the reply has already produced; 1.0 disables it.
    repetition_penalty: Option<f32>,
    ponder_strategy: Option<String>,
    guidance_scale: Option<f32>,
    ensemble_n: Option<usize>,
    noise_sigma: Option<f32>,
    entropy_threshold: Option<f32>,
}

fn parse_chat_request(req_body: &str) -> Result<ChatRequest, ChatHttpError> {
    let request: ChatRequest = serde_json::from_str(req_body)
        .map_err(|error| ChatHttpError::bad_request(format!("invalid JSON request: {error}")))?;

    match (&request.message, &request.messages) {
        (Some(_), Some(_)) => {
            return Err(ChatHttpError::bad_request(
                "send either message or messages, not both",
            ));
        }
        (None, None) => {
            return Err(ChatHttpError::bad_request(
                "request needs a message or a messages array",
            ));
        }
        (Some(message), None) => {
            if message.trim().is_empty() {
                return Err(ChatHttpError::bad_request(
                    "message must contain at least one non-whitespace character",
                ));
            }
        }
        (None, Some(messages)) => {
            for entry in messages {
                if entry.role != "user" && entry.role != "assistant" {
                    return Err(ChatHttpError::bad_request(format!(
                        "message role must be \"user\" or \"assistant\", not {:?}",
                        entry.role
                    )));
                }
                if entry.content.trim().is_empty() {
                    return Err(ChatHttpError::bad_request(
                        "every message must contain at least one non-whitespace character",
                    ));
                }
            }
            // Covers the empty array too: there is no last message to answer.
            if messages.last().is_none_or(|last| last.role != "user") {
                return Err(ChatHttpError::bad_request(
                    "the last message must have role \"user\" — it is the one being answered",
                ));
            }
        }
    }

    validate_chat_options(&request)?;

    Ok(request)
}

/// The request body as text. The HTTP layer has already refused to read any
/// body declared larger than [`MAX_CHAT_REQUEST_BYTES`].
fn read_request_body(body: &Body) -> Result<String, ChatHttpError> {
    match body {
        Body::TooLarge => Err(ChatHttpError::payload_too_large()),
        Body::Complete(bytes) => String::from_utf8(bytes.clone())
            .map_err(|_| ChatHttpError::bad_request("request body must be valid UTF-8")),
    }
}

fn build_ponder(req: &ChatRequest) -> Result<PonderConfig, ChatHttpError> {
    validate_chat_options(req)?;
    let strat = req.ponder_strategy.as_deref().unwrap_or("none");
    let gs = req.guidance_scale.unwrap_or(1.5);
    let en = req.ensemble_n.unwrap_or(3);
    let ns = req.noise_sigma.unwrap_or(0.05);
    let et = req.entropy_threshold.unwrap_or(3.0);

    let config = match strat {
        "none" => PonderConfig::none(),
        "cfg" => PonderConfig::cfg(gs),
        "ensemble" => PonderConfig::ensemble(en, ns),
        "cfg-ensemble" => PonderConfig::cfg_ensemble(gs, en, ns),
        "adaptive" => PonderConfig::adaptive(gs, et),
        _ => unreachable!("strategy was validated"),
    };
    Ok(config)
}

fn validate_chat_options(req: &ChatRequest) -> Result<(), ChatHttpError> {
    // 0 is greedy decoding, which the sampler implements as an argmax.
    validate_optional_f32("temperature", req.temperature, 0.0, 2.0)?;
    // The same range `rai run --repetition-penalty` accepts.
    validate_optional_f32("repetition_penalty", req.repetition_penalty, 0.01, 4.0)?;
    // Only the floor can be checked without the model. The ceiling is the
    // context the prompt leaves free, and that needs the tokenizer — see
    // [`resolve_max_tokens`], which runs once the prompt has been encoded.
    if req.max_tokens.is_some_and(|value| value == 0) {
        return Err(ChatHttpError::bad_request("max_tokens must be at least 1"));
    }
    if req.ponder_strategy.as_deref().is_some_and(|strategy| {
        !matches!(
            strategy,
            "none" | "cfg" | "ensemble" | "cfg-ensemble" | "adaptive"
        )
    }) {
        return Err(ChatHttpError::bad_request(
            "ponder_strategy must be one of: none, cfg, ensemble, cfg-ensemble, adaptive",
        ));
    }
    validate_optional_f32("guidance_scale", req.guidance_scale, 0.0, 4.0)?;
    if req
        .ensemble_n
        .is_some_and(|value| !(1..=MAX_ENSEMBLE_SIZE).contains(&value))
    {
        return Err(ChatHttpError::bad_request(format!(
            "ensemble_n must be between 1 and {MAX_ENSEMBLE_SIZE}"
        )));
    }
    // Ensemble strategies need at least two passes; reject explicitly instead
    // of silently running a different configuration than requested.
    if matches!(
        req.ponder_strategy.as_deref(),
        Some("ensemble") | Some("cfg-ensemble")
    ) && req.ensemble_n.is_some_and(|value| value < 2)
    {
        return Err(ChatHttpError::bad_request(
            "ensemble strategies require ensemble_n >= 2",
        ));
    }
    validate_optional_f32("noise_sigma", req.noise_sigma, 0.0, 1.0)?;
    validate_optional_f32(
        "entropy_threshold",
        req.entropy_threshold,
        0.0,
        MAX_ENTROPY_THRESHOLD,
    )?;
    Ok(())
}

fn validate_optional_f32(
    name: &str,
    value: Option<f32>,
    minimum: f32,
    maximum: f32,
) -> Result<(), ChatHttpError> {
    if value.is_some_and(|value| !value.is_finite() || value < minimum || value > maximum) {
        return Err(ChatHttpError::bad_request(format!(
            "{name} must be a finite number between {minimum} and {maximum}"
        )));
    }
    Ok(())
}

fn validate_prompt_tokens(
    prompt_tokens: &[usize],
    max_context: usize,
    vocab_size: usize,
) -> Result<(), ChatHttpError> {
    if prompt_tokens.is_empty() {
        return Err(ChatHttpError::bad_request(
            "message produced no tokens with the configured tokenizer",
        ));
    }
    if max_context == 0 {
        return Err(ChatHttpError::internal(
            "model has no usable context window",
        ));
    }
    if prompt_tokens.len() > max_context {
        return Err(ChatHttpError::bad_request(format!(
            "message is too long: {} tokens exceeds the {max_context}-token context window",
            prompt_tokens.len()
        )));
    }
    if prompt_tokens.iter().any(|&token| token >= vocab_size) {
        return Err(ChatHttpError::bad_request(
            "configured tokenizer produced a token outside the model vocabulary",
        ));
    }
    Ok(())
}

/// Room left in the context window after the prompt, in tokens.
///
/// A conversation of `prompt_len + generated` tokens is what the window has to
/// hold, so this is simply the difference. `validate_prompt_tokens` has
/// already established `1 <= prompt_len <= max_ctx`, so the subtraction cannot
/// wrap; it saturates rather than relying on that from a distance.
///
/// The decode loop could in fact sample one token beyond this, because the
/// last token generated is never written back into the cache. That token is
/// not offered: it cannot be continued from, and counting it would make
/// `context_used` report one more than `max_context` — a full meter that reads
/// past full is worse than a token nobody asked for.
fn remaining_generation_tokens(prompt_len: usize, max_ctx: usize) -> usize {
    max_ctx.saturating_sub(prompt_len)
}

/// How many tokens to generate, given what the caller asked for.
///
/// A caller who names a number that cannot fit is told so with both numbers,
/// rather than being handed a short reply and left to wonder why. A caller who
/// names nothing gets the default, shortened to fit if the prompt is long —
/// that is a default being chosen, not a request being clipped.
fn resolve_max_tokens(
    requested: Option<usize>,
    prompt_len: usize,
    max_ctx: usize,
) -> Result<usize, ChatHttpError> {
    let remaining = remaining_generation_tokens(prompt_len, max_ctx);
    if remaining == 0 {
        return Err(ChatHttpError::bad_request(format!(
            "a {prompt_len}-token prompt fills the whole {max_ctx}-token context window, \
             leaving no room for a reply"
        )));
    }
    match requested {
        Some(value) if value > remaining => Err(ChatHttpError::bad_request(format!(
            "max_tokens {value} does not fit: a {prompt_len}-token prompt leaves {remaining} \
             of the {max_ctx}-token context window free"
        ))),
        Some(value) => Ok(value),
        None => Ok(DEFAULT_CHAT_GENERATION_TOKENS.min(remaining)),
    }
}

/// Why generation stopped. These are the only three ways out of the decode
/// loop, which is what lets the streaming client treat `done` as final.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopReason {
    Eos,
    MaxTokens,
    StopSequence,
}

impl StopReason {
    fn as_str(self) -> &'static str {
        match self {
            StopReason::Eos => "eos",
            StopReason::MaxTokens => "max_tokens",
            StopReason::StopSequence => "stop_sequence",
        }
    }
}

/// Everything a chat request needs from the model, worked out before
/// generation starts.
///
/// This exists so that both chat routes reject the same bad request with the
/// same status code *before* either has written a byte. A streaming route that
/// has already sent `200 OK` can only report a bad request from inside the
/// stream, which is a far worse thing to consume.
struct Prepared {
    prompt_tokens: Vec<usize>,
    max_tokens: usize,
    sampler: SamplerConfig,
    ponder: PonderConfig,
    max_ctx: usize,
}

/// What one completed generation produced.
struct Outcome {
    text: String,
    tokens_generated: usize,
    /// Tokens in the rendered prompt.
    prompt_tokens: usize,
    /// How many of them came from the previous generation's cache rather
    /// than being prefilled again.
    prompt_tokens_reused: usize,
    prefill_ms: f64,
    decode_ms: f64,
    tok_per_sec: f64,
    avg_passes: f64,
    hard_tokens_pct: f64,
    stop_reason: StopReason,
}

/// Something worth telling a streaming client about, as it happens.
enum GenerationEvent<'a> {
    /// A prefill chunk is about to run. Nothing to show; the route answers
    /// whether its client is still there to read the reply.
    Prefilling,
    /// The prompt is in the KV cache; decoding is about to start. `reused`
    /// leading prompt tokens came from the previous generation's cache.
    Prefilled { prefill_ms: f64, reused: usize },
    /// Newly decoded text, already safe to display (see [`emittable_len`]).
    Text(&'a str),
}

/// Why a generation ended early.
enum GenerateFailure {
    /// The server failed. Reportable to the client, sanitized as usual.
    Http(ChatHttpError),
    /// The client went away mid-stream. There is nobody left to tell.
    Client(std::io::Error),
}

fn prepare(loaded: &Loaded, chat_req: &ChatRequest) -> Result<Prepared, ChatHttpError> {
    validate_chat_options(chat_req)?;
    let ponder = build_ponder(chat_req)?;
    let sampler = SamplerConfig {
        temperature: chat_req.temperature.unwrap_or(0.7),
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: chat_req.repetition_penalty.unwrap_or(1.1),
    };

    // `Loaded::open` already clamped this to the model, and the KV cache was
    // allocated for exactly this many positions.
    let max_ctx = loaded.max_context;

    // The format decides the special-token policy, so serving a model
    // produces exactly the prompt `rai run` does.
    let encode = |turns: &[(Role, &str)]| {
        loaded
            .format
            .encode(&loaded.tokenizer, turns)
            .map_err(|error| ChatHttpError::bad_request(format!("{error:#}")))
    };

    // Render the request through the configured chat template. Multi-turn
    // requests replay the whole thread; the server holds no conversation
    // state between requests, so the prompt is the only memory there is.
    //
    // A replayed thread eventually outgrows the window even though every
    // individual message was fine, and the client cannot know where the token
    // boundary falls — only this side holds the tokenizer. So the oldest
    // turns are dropped until the rendering fits, the turn being answered is
    // never dropped, and a quarter of the window stays reserved for the reply
    // so a long history cannot trim itself into a one-token answer. A lone
    // final turn that still does not fit falls through to the same refusal a
    // single over-long message has always received.
    let prompt_tokens = match &chat_req.messages {
        Some(messages) => {
            let merged = merge_turns(messages);
            let turns: Vec<(Role, &str)> = merged
                .iter()
                .map(|(role, text)| (*role, text.as_str()))
                .collect();
            let reserve = (max_ctx / 4).max(1);
            let start = thread_trim_start(&turns, max_ctx, reserve, |start| {
                Ok(encode(&turns[start..])?.len())
            })?;
            encode(&turns[start..])?
        }
        None => encode(&[(Role::User, chat_req.message.as_deref().unwrap_or_default())])?,
    };
    validate_prompt_tokens(
        &prompt_tokens,
        max_ctx,
        loaded.model.config.vocab_size as usize,
    )?;
    let max_tokens = resolve_max_tokens(chat_req.max_tokens, prompt_tokens.len(), max_ctx)?;

    Ok(Prepared {
        prompt_tokens,
        max_tokens,
        sampler,
        ponder,
        max_ctx,
    })
}

/// The thread as alternating turns: consecutive messages from one role are
/// one turn, joined by a blank line.
///
/// Templates assume alternation and Gemma's refuses anything else; a client
/// that drops a failed reply from its history sends two user messages in a
/// row, and that must not turn into a refusal.
fn merge_turns(messages: &[ChatMessage]) -> Vec<(Role, String)> {
    let mut turns: Vec<(Role, String)> = Vec::with_capacity(messages.len());
    for message in messages {
        let role = if message.role == "assistant" {
            Role::Assistant
        } else {
            Role::User
        };
        match turns.last_mut() {
            Some((last, text)) if *last == role => {
                text.push_str("\n\n");
                text.push_str(&message.content);
            }
            _ => turns.push((role, message.content.clone())),
        }
    }
    turns
}

/// First turn of the longest thread suffix whose rendering leaves `reserve`
/// tokens of the window free.
///
/// Only a user turn can start the suffix, so the thread is trimmed by whole
/// exchanges and never opens on an orphaned reply. `tokens_for_suffix(start)`
/// is the rendered token count of `turns[start..]`, re-rendered per
/// candidate because chat templates are not prefix-additive; dropping turns
/// only shortens a rendering, so the candidates are binary-searched rather
/// than tried one by one. The final turn is kept no matter what: dropping
/// the message being answered is never an option, and whether a lone
/// over-long turn is refused stays the caller's decision.
fn thread_trim_start<S>(
    turns: &[(Role, S)],
    max_ctx: usize,
    reserve: usize,
    mut tokens_for_suffix: impl FnMut(usize) -> Result<usize, ChatHttpError>,
) -> Result<usize, ChatHttpError> {
    let starts: Vec<usize> = turns
        .iter()
        .enumerate()
        .filter(|(_, (role, _))| *role == Role::User)
        .map(|(index, _)| index)
        .collect();
    let Some(&last) = starts.last() else {
        return Ok(turns.len().saturating_sub(1));
    };
    // The earliest start that fits, among `starts[low..high]`.
    let (mut low, mut high) = (0, starts.len() - 1);
    while low < high {
        let middle = low + (high - low) / 2;
        if tokens_for_suffix(starts[middle])? + reserve <= max_ctx {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Ok(if low < starts.len() {
        starts[low]
    } else {
        last
    })
}

/// How much of `pending` can be shown to a reader right now.
///
/// Two things must not escape early. A chat template's stop sequence is cut
/// from the final text, so streaming its opening bytes would leave `<|im_` on
/// screen; and a trailing U+FFFD almost always means a multi-byte character is
/// split across two tokens and will resolve on the next one. Both are held
/// back until the next token settles them, and the flush after the loop
/// releases whatever is genuinely final.
pub(crate) fn emittable_len<S: AsRef<str>>(pending: &str, stops: &[S]) -> usize {
    let mut safe = pending.len();
    while pending[..safe].ends_with('\u{FFFD}') {
        safe -= '\u{FFFD}'.len_utf8();
    }

    let head = &pending[..safe];
    let mut held = 0usize;
    for stop in stops.iter().map(AsRef::as_ref) {
        // The longest *proper* prefix of this stop sequence that `head` ends
        // with. A complete match is not this function's business: the caller
        // has already searched for one and truncated.
        let mut length = stop.len().saturating_sub(1).min(head.len());
        while length > 0 {
            if stop.is_char_boundary(length) && head.ends_with(&stop[..length]) {
                held = held.max(length);
                break;
            }
            length -= 1;
        }
    }
    safe - held
}

/// Run one generation, reporting progress as it happens.
///
/// `on_event` is the only difference between `POST /api/chat` and
/// `POST /api/chat/stream`: the former ignores every event and reads the
/// [`Outcome`], the latter writes each one to the socket. Having one loop
/// means the two routes cannot drift into producing different text.
fn generate(
    loaded: &mut Loaded,
    prepared: &Prepared,
    mut on_event: impl FnMut(GenerationEvent<'_>) -> std::io::Result<()>,
) -> Result<Outcome, GenerateFailure> {
    let Loaded {
        model,
        tokenizer,
        format,
        kv_cache,
        cached_tokens,
        ..
    } = loaded;
    let prompt_tokens = &prepared.prompt_tokens;

    // Leave the final prompt token for the first decode pass. Processing every
    // prompt token here and then the last one again at the next position would
    // store it twice.
    let prefix = &prompt_tokens[..prompt_tokens.len() - 1];

    // Keep the K/V of whatever this prompt shares with the last generation and
    // prefill only the rest. The truncation is what makes that safe: the cache
    // refuses any read at or above its watermark, so nothing past the shared
    // prefix — the rest of an older conversation — is reachable.
    let reused = cached_tokens
        .iter()
        .zip(prefix)
        .take_while(|(cached, new)| cached == new)
        .count();
    cached_tokens.clear();
    kv_cache.truncate(reused);

    let mut work = InferenceWork::new();
    let mut work2 = InferenceWork::new();
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut all_tokens = prompt_tokens.clone();

    let t_prefill = Instant::now();
    let finished = model
        .prefill_while(
            &prefix[reused..],
            reused,
            kv_cache,
            &mut BatchScratch::new(),
            || on_event(GenerationEvent::Prefilling).is_ok(),
        )
        .map_err(|error| GenerateFailure::Http(ChatHttpError::internal(error)))?;
    if !finished {
        return Err(GenerateFailure::Client(std::io::Error::new(
            std::io::ErrorKind::ConnectionAborted,
            "the client left during prefill",
        )));
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    on_event(GenerationEvent::Prefilled { prefill_ms, reused }).map_err(GenerateFailure::Client)?;

    // Decode
    let t_decode = Instant::now();
    let mut generated_text = String::new();
    let mut emitted = String::new();
    let mut tokens_generated = 0;
    let mut total_passes = 0;
    let mut hard_tokens = 0;
    let mut stop_reason = StopReason::MaxTokens;

    // The checkpoint's own end-of-sequence ids, every family's end markers
    // among the tokenizer's control tokens, and the template's turn markers.
    let end_ids = format.end_ids();
    let stops = format.stop_sequences();

    // Each step feeds the newest token at the next position: from the last
    // prompt token to the end of the window, and at most `max_tokens` of them.
    // `max_tokens` is already bounded by the free context; the range keeps the
    // window bound an invariant of this loop rather than of its caller.
    for pos in (prefix.len()..prepared.max_ctx).take(prepared.max_tokens) {
        let last_token = all_tokens.last().copied().ok_or_else(|| {
            GenerateFailure::Http(ChatHttpError::bad_request(
                "message produced no usable prompt tokens",
            ))
        })?;
        let (mut logits, metrics) = pondered_forward(
            model,
            last_token,
            pos,
            kv_cache,
            &prepared.ponder,
            &mut work,
            &mut work2,
            &mut rng,
        )
        .map_err(|error| GenerateFailure::Http(ChatHttpError::internal(error)))?;
        total_passes += metrics.forward_passes;
        if metrics.was_hard_token {
            hard_tokens += 1;
        }

        // The reply only: a chat prompt holds the end-of-turn marker the
        // reply has to end with, and penalizing it kept replies from ending.
        apply_repetition_penalty(
            &mut logits,
            &all_tokens[prompt_tokens.len()..],
            prepared.sampler.repetition_penalty,
        );
        let next_token = sample_token(&mut logits, &prepared.sampler, &mut rng);
        // Hand the vocabulary-sized buffer back for the next step to reuse,
        // as `pondered_forward` asks of every caller.
        work.scratch.logits = logits;

        if end_ids.contains(&next_token) {
            stop_reason = StopReason::Eos;
            break;
        }

        all_tokens.push(next_token);
        tokens_generated += 1;

        // Decode entire generated suffix for correct SentencePiece spacing
        let gen_ids: Vec<u32> = all_tokens[prompt_tokens.len()..]
            .iter()
            .map(|&t| t as u32)
            .collect();
        generated_text = tokenizer.decode(&gen_ids, false).unwrap_or_default();

        // Stop if model generates a template-specific stop sequence
        let mut hit_stop = false;
        if let Some(cut) = stops
            .iter()
            .filter_map(|stop| generated_text.find(stop.as_str()))
            .min()
        {
            generated_text.truncate(cut);
            hit_stop = true;
        }

        let unemitted = incremental_suffix(&emitted, &generated_text);
        let length = if hit_stop {
            // Nothing more is coming, so nothing needs holding back.
            unemitted.len()
        } else {
            emittable_len(unemitted, stops)
        };
        if length > 0 {
            let chunk = &unemitted[..length];
            on_event(GenerationEvent::Text(chunk)).map_err(GenerateFailure::Client)?;
            emitted.push_str(chunk);
        }

        if hit_stop {
            stop_reason = StopReason::StopSequence;
            break;
        }
    }

    // Whatever the hold-back was still sitting on is final now.
    let unemitted = incremental_suffix(&emitted, &generated_text);
    if !unemitted.is_empty() {
        on_event(GenerationEvent::Text(unemitted)).map_err(GenerateFailure::Client)?;
    }

    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let tok_per_sec = if decode_ms > 0.0 {
        tokens_generated as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let avg_passes = if tokens_generated > 0 {
        total_passes as f64 / tokens_generated as f64
    } else {
        0.0
    };
    let hard_tokens_pct = if tokens_generated > 0 {
        100.0 * hard_tokens as f64 / tokens_generated as f64
    } else {
        0.0
    };

    // The generation finished, so the cache holds exactly the tokens that went
    // through the model, in order: the prompt, then every generated token but
    // the last sampled one, which was never fed back.
    let filled = kv_cache.filled_min().min(all_tokens.len());
    cached_tokens.extend_from_slice(&all_tokens[..filled]);

    Ok(Outcome {
        text: generated_text,
        tokens_generated,
        prompt_tokens: prompt_tokens.len(),
        prompt_tokens_reused: reused,
        prefill_ms,
        decode_ms,
        tok_per_sec,
        avg_passes,
        hard_tokens_pct,
        stop_reason,
    })
}

/// `POST /api/chat` — one JSON body. Accepts the original single `message`
/// form and the `messages` thread form; both flow through the same
/// [`prepare`]/[`generate`] pair as streaming.
fn handle_generate(
    loaded: &mut Loaded,
    chat_req: &ChatRequest,
    client: &Request,
) -> Result<String, ChatHttpError> {
    let prepared = prepare(loaded, chat_req)?;
    // Nothing is sent until the end, so the client is asked whether it is
    // still there at every chunk of prefill and every piece of text: a caller
    // that gave up stops the work instead of holding the server for it.
    let outcome = match generate(loaded, &prepared, |_| {
        if client.client_gone() {
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionAborted,
                "the client left",
            ))
        } else {
            Ok(())
        }
    }) {
        Ok(outcome) => outcome,
        Err(GenerateFailure::Http(error)) => return Err(error),
        // Nobody is left to read this.
        Err(GenerateFailure::Client(error)) => return Err(ChatHttpError::internal(error)),
    };

    let response = serde_json::json!({
        "text": outcome.text.trim(),
        "tokens": outcome.tokens_generated,
        "prompt_tokens": outcome.prompt_tokens,
        "prompt_tokens_reused": outcome.prompt_tokens_reused,
        "prefill_ms": outcome.prefill_ms,
        "decode_ms": outcome.decode_ms,
        "tok_per_sec": outcome.tok_per_sec,
        "avg_passes": outcome.avg_passes,
        "hard_tokens_pct": outcome.hard_tokens_pct,
        "strategy": format!("{:?}", prepared.ponder.strategy),
        "stop_reason": outcome.stop_reason.as_str(),
    });

    Ok(response.to_string())
}

// =============================================================================
// POST /api/chat/stream
// =============================================================================

/// One ndjson response, one record per chunk.
///
/// Each record is framed and flushed the moment it exists, which is what puts
/// a token on the wire as it is generated rather than in buffered blocks.
struct NdjsonStream {
    writer: ChunkedWriter,
}

impl NdjsonStream {
    fn start(request: Request) -> std::io::Result<Self> {
        let writer = request.into_chunked(&[
            ("Content-Type", "application/x-ndjson"),
            ("Cache-Control", "no-store"),
            ("X-Content-Type-Options", "nosniff"),
        ])?;
        Ok(Self { writer })
    }

    fn send(&mut self, value: &Value) -> std::io::Result<()> {
        self.writer.send(ndjson_record(value).as_bytes())
    }

    fn finish(self) -> std::io::Result<()> {
        self.writer.finish()
    }
}

/// One JSON object as one ndjson line.
///
/// `serde_json` escapes newlines inside strings, so a serialized value can
/// never contain the byte that separates records: the ndjson framing holds for
/// whatever the model generates, including a reply that is nothing but
/// newlines.
fn ndjson_record(value: &Value) -> String {
    let mut line = value.to_string();
    line.push('\n');
    line
}

/// Generate into an already-opened stream.
///
/// Everything that could have been a 4xx was decided in [`prepare`], before
/// the response line went out. What is left can only be a server failure, and
/// it is reported as a terminal `error` record with the same sanitized text a
/// 500 would carry.
fn stream_chat(request: Request, loaded: &mut Loaded, prepared: &Prepared) {
    let mut stream = match NdjsonStream::start(request) {
        Ok(stream) => stream,
        Err(error) => {
            eprintln!("chat stream could not start: {error}");
            return;
        }
    };

    let prompt_tokens = prepared.prompt_tokens.len();
    let max_ctx = prepared.max_ctx;
    let gone = || std::io::Error::new(std::io::ErrorKind::ConnectionAborted, "the client left");
    let outcome = generate(loaded, prepared, |event| match event {
        // Studio's Stop closes the connection; a write to it can still
        // succeed into the socket buffer, so the connection is asked.
        GenerationEvent::Prefilling if stream.writer.client_gone() => Err(gone()),
        GenerationEvent::Prefilling => Ok(()),
        GenerationEvent::Text(_) if stream.writer.client_gone() => Err(gone()),
        GenerationEvent::Prefilled { prefill_ms, reused } => stream.send(&json!({
            "type": "start",
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_reused": reused,
            "prefill_ms": prefill_ms,
            "max_context": max_ctx,
        })),
        GenerationEvent::Text(text) => stream.send(&json!({
            "type": "token",
            "text": text,
        })),
    });

    match outcome {
        Ok(outcome) => {
            let _ = stream.send(&json!({
                "type": "done",
                "tokens": outcome.tokens_generated,
                "tok_per_sec": outcome.tok_per_sec,
                "decode_ms": outcome.decode_ms,
                "context_used": prompt_tokens + outcome.tokens_generated,
                "max_context": max_ctx,
                "stop_reason": outcome.stop_reason.as_str(),
            }));
            let _ = stream.finish();
        }
        Err(GenerateFailure::Http(error)) => {
            let _ = stream.send(&json!({
                "type": "error",
                "message": error.to_string(),
            }));
            let _ = stream.finish();
        }
        // The socket is already broken; writing a terminator to it would only
        // produce a second error to ignore.
        Err(GenerateFailure::Client(error)) => {
            eprintln!("chat stream ended early: {error}");
        }
    }
}

pub fn run(args: &ServeArgs) -> Result<()> {
    crate::gemm::configure_thread_pool();
    if matches!(args.options.max_context, Some(0)) {
        anyhow::bail!("--max-context must be greater than zero");
    }
    // Checked here as well as at every load: started with no model, a typo
    // would otherwise only surface as every later `/api/load` failing.
    check_template_name(&args.options.chat_template)?;

    // Starting with no model is now legitimate: the UI picks one over
    // `POST /api/load`. A model named on the command line still has to load
    // before the port opens — failing three seconds after printing a URL the
    // user has already clicked is worse than not starting.
    let loaded = match &args.model {
        Some(model_path) => {
            eprintln!("Loading model: {}", model_path.display());
            let loaded = Loaded::open(
                model_path,
                args.tokenizer.as_deref(),
                args.options.max_context,
                &args.options.chat_template,
            )?;
            announce(&loaded);
            Some(loaded)
        }
        None => {
            eprintln!("Starting with no model loaded (POST /api/load to pick one).");
            None
        }
    };

    let state = ServerState {
        loaded,
        jobs: Arc::new(Jobs::new()),
        options: args.options.clone(),
    };

    let server = bind_loopback(args.options.port)?;
    let bound = server.local_addr();
    let bound_port = bound.port();
    // The address actually bound: `localhost` can resolve to the other
    // family first, so on an IPv6-only bind the literal is the URL that works.
    let url = if bound.is_ipv6() {
        format!("http://[::1]:{bound_port}")
    } else {
        format!("http://localhost:{bound_port}")
    };
    eprintln!("\n  Chat UI: {url}\n");
    eprintln!("  Press Ctrl+C to stop.\n");

    serve_on(&server, state, bound_port);
    Ok(())
}

/// The request headers the screening reads.
#[derive(Debug, Clone, Copy, Default)]
struct Screened<'a> {
    host: Option<&'a str>,
    origin: Option<&'a str>,
    /// `Sec-Fetch-Site`, which every current browser attaches to every request
    /// it makes, saying whether the page that caused it is this origin.
    fetch_site: Option<&'a str>,
    content_type: Option<&'a str>,
}

/// The checks every request passes before it can reach a route.
///
/// Kept as one function on the near side of the dispatch rather than as
/// per-route calls: a route added later cannot forget to apply them, because
/// there is nowhere in the dispatch a request arrives without having been
/// through here.
///
/// - `Host` stops DNS rebinding from pointing a page at this port.
/// - For `/api`, `Origin` and `Sec-Fetch-Site` stop a page on another origin
///   from making the request at all — for every method. A cross-site page can
///   send a GET without asking anyone (an `<img src>` needs no preflight), and
///   GET routes read the filesystem: on Windows a GET naming a `\\host\share`
///   directory was enough to make the machine authenticate to that host.
/// - For a POST, requiring a JSON `Content-Type` stops the "simple request"
///   form a cross-origin POST can take without a preflight, for any browser
///   too old to send `Sec-Fetch-Site`.
///
/// The page itself (`/`) only needs the `Host` check: following a link to it
/// from anywhere is harmless, and its CSP refuses to be framed.
fn screen_headers(
    method: Method,
    is_api: bool,
    headers: Screened<'_>,
    port: u16,
) -> Result<(), ChatHttpError> {
    if !headers.host.is_some_and(|host| is_allowed_host(host, port)) {
        return Err(ChatHttpError::forbidden("invalid Host header"));
    }
    if is_api {
        if headers
            .origin
            .is_some_and(|origin| !is_allowed_origin(origin, port))
        {
            return Err(ChatHttpError::forbidden(
                "cross-origin requests are not allowed",
            ));
        }
        // `none` is the user's own action (typing the URL); `same-origin` is
        // Studio itself. `same-site` is another port on localhost, which is
        // another program and gets no more trust than any other origin.
        if headers.fetch_site.is_some_and(|site| {
            !site.eq_ignore_ascii_case("same-origin") && !site.eq_ignore_ascii_case("none")
        }) {
            return Err(ChatHttpError::forbidden(
                "cross-site requests are not allowed",
            ));
        }
    }
    if method == Method::Post {
        let is_json = headers.content_type.is_some_and(|value| {
            value
                .split(';')
                .next()
                .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
        });
        if !is_json {
            return Err(ChatHttpError::unsupported_media_type());
        }
    }
    Ok(())
}

fn screen_request(request: &Request, path: &str, port: u16) -> Result<(), ChatHttpError> {
    screen_headers(
        request.method(),
        path.starts_with("/api/"),
        Screened {
            host: request.header("Host"),
            origin: request.header("Origin"),
            fetch_site: request.header("Sec-Fetch-Site"),
            content_type: request.header("Content-Type"),
        },
        port,
    )
}

/// The request loop.
///
/// Split out of [`run`] so a test can drive a real server on a real socket:
/// the security screening above and the ndjson framing below are properties of
/// what goes over the wire, and asserting them against anything less than a
/// socket would be asserting against a re-implementation.
///
/// Deliberately single-threaded: requests are handled one at a time here, so
/// one generation saturates the CPU without a second request competing for
/// cores, and no synchronization is needed around the model state.
/// Conversions are the one thing that must not be serialized with it — they
/// take minutes — so they run on their own threads and this loop only ever
/// reads their progress.
fn serve_on(server: &HttpServer, mut state: ServerState, bound_port: u16) {
    for request in server.incoming_requests() {
        // One request cannot take the server down with it. A panic inside a
        // generation (an allocation failing deep in the model, say) unwinds
        // to here; the state it leaves is consistent by construction: the KV
        // cache's reuse record is cleared before a generation touches the
        // cache and only restored when one finishes, and the conversion jobs
        // recover their own poisoned locks.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle_request(request, &mut state, bound_port)
        }));
        if outcome.is_err() {
            eprintln!("request failed with an internal error; the server keeps running");
        }
    }
}

/// Route and answer one request.
fn handle_request(request: Request, state: &mut ServerState, bound_port: u16) {
    let url = request.url().to_string();
    let (path, query) = split_query(&url);
    let method = request.method();

    if let Err(error) = screen_request(&request, path, bound_port) {
        respond_json_error(request, &error);
        return;
    }

    // Every POST route takes a JSON body; read it once, here.
    let body = if method == Method::Post {
        match read_request_body(request.body()) {
            Ok(body) => body,
            Err(error) => {
                respond_json_error(request, &error);
                return;
            }
        }
    } else {
        String::new()
    };

    match (method, path) {
        (Method::Get | Method::Head, "/" | "/index.html") => {
            let resp = Response::new(200, CHAT_HTML.as_bytes().to_vec())
                .with_header("Content-Type", "text/html; charset=utf-8")
                .with_header(
                    "Content-Security-Policy",
                    "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
                )
                .with_header("X-Content-Type-Options", "nosniff");
            let _ = request.respond(resp);
        }
        (Method::Post, "/api/chat") => {
            let outcome =
                parse_chat_request(&body).and_then(|chat_request| match state.loaded.as_mut() {
                    Some(loaded) => handle_generate(loaded, &chat_request, &request),
                    None => Err(no_model_loaded()),
                });
            match outcome {
                Ok(json) => {
                    let resp = Response::new(200, json.into_bytes())
                        .with_header("Content-Type", "application/json");
                    let _ = request.respond(resp);
                }
                Err(error) => respond_json_error(request, &error),
            }
        }
        (Method::Post, "/api/chat/stream") => {
            let chat_request = match parse_chat_request(&body) {
                Ok(chat_request) => chat_request,
                Err(error) => {
                    respond_json_error(request, &error);
                    return;
                }
            };
            if !request.is_http11() {
                respond_json_error(request, &ChatHttpError::http_version_not_supported());
                return;
            }
            // Everything that can be a status code is decided here, while
            // an ordinary JSON error response is still possible.
            let prepared = match state.loaded.as_ref() {
                Some(loaded) => prepare(loaded, &chat_request),
                None => Err(no_model_loaded()),
            };
            let prepared = match prepared {
                Ok(prepared) => prepared,
                Err(error) => {
                    respond_json_error(request, &error);
                    return;
                }
            };
            match state.loaded.as_mut() {
                Some(loaded) => stream_chat(request, loaded, &prepared),
                // Unreachable: `prepare` above only succeeds with a model
                // loaded, and this loop is the only thing that can unload
                // one.
                None => respond_json_error(request, &no_model_loaded()),
            }
        }
        (Method::Get, "/api/info") => {
            let info = match state.loaded.as_ref() {
                Some(loaded) => loaded.info_json(),
                None => json!({
                    "loaded": false,
                    "model_path": Value::Null,
                    "chat_template": state.options.chat_template,
                    // Null, not a number: with no model there is no
                    // context window yet, and the flag (if any) is only a
                    // request that the next model may clamp.
                    "max_context": state.options.max_context,
                }),
            };
            respond_json(request, &info);
        }
        (Method::Get, "/api/system") => {
            let system = system_json(state);
            respond_json(request, &system);
        }
        (Method::Get, "/api/models") => match handle_models(state, query) {
            Ok(value) => respond_json(request, &value),
            Err(error) => respond_json_error(request, &error),
        },
        (Method::Post, "/api/load") => match handle_load(state, &body) {
            Ok(value) => respond_json(request, &value),
            Err(error) => respond_json_error(request, &error),
        },
        (Method::Post, "/api/inspect") => match handle_inspect(&body) {
            Ok(value) => respond_json(request, &value),
            Err(error) => respond_json_error(request, &error),
        },
        (Method::Post, "/api/convert") => match handle_convert(state, &body) {
            Ok(value) => {
                let resp = Response::new(202, value.to_string().into_bytes())
                    .with_header("Content-Type", "application/json");
                let _ = request.respond(resp);
            }
            Err(error) => respond_json_error(request, &error),
        },
        (Method::Get, "/api/convert") => respond_json(request, &state.jobs.list()),
        (Method::Get, path) if path.starts_with("/api/convert/") => {
            match handle_job_poll(state, path, query) {
                Ok(value) => respond_json(request, &value),
                Err(error) => respond_json_error(request, &error),
            }
        }
        (Method::Options, _) => respond_json_error(
            request,
            &ChatHttpError::forbidden("cross-origin preflight is not allowed"),
        ),
        _ => {
            let _ = request.respond(Response::new(404, b"404".to_vec()));
        }
    }
}

fn announce(loaded: &Loaded) {
    let config = &loaded.model.config;
    eprintln!(
        "Model loaded (hidden={}, layers={}, heads={}/{}kv, inter={}, vocab={})",
        config.hidden_size,
        config.num_layers,
        config.num_heads,
        config.num_kv_heads,
        config.intermediate_size,
        config.vocab_size
    );
    eprintln!(
        "Weights: {:.1} MB, KV cache: {:.1} MB (max_ctx={})",
        loaded.model.file_size() as f64 / (1024.0 * 1024.0),
        loaded.model.kv_cache_bytes(loaded.max_context) as f64 / (1024.0 * 1024.0),
        loaded.max_context
    );
    eprintln!("Tokenizer: {}", loaded.tokenizer_path.display());
    eprintln!("Chat template: {}", loaded.format.name());
}

fn no_model_loaded() -> ChatHttpError {
    ChatHttpError::conflict(
        "no model is loaded; POST /api/load {\"path\": \"...\"} first, or see GET /api/models",
    )
}

// =============================================================================
// GET /api/models
// =============================================================================

#[derive(Debug, Deserialize)]
struct LoadRequest {
    path: PathBuf,
    tokenizer: Option<PathBuf>,
    max_context: Option<usize>,
    chat_template: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InspectRequest {
    source: String,
    /// Preflight rules that depend on the conversion's own options — the
    /// sliding-window check and the quantization-group limits — need them, so
    /// they can be given here and default to the converter's defaults.
    group_size: Option<u32>,
    embed_group_size: Option<u32>,
    max_context: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ConvertRequest {
    source: PathBuf,
    output: PathBuf,
    group_size: Option<u32>,
    embed_group_size: Option<u32>,
    max_context: Option<u32>,
    tokenizer_out: Option<PathBuf>,
}

/// List `.raimodel` files, from their headers only.
///
/// With no `?dir=`, the answer is the working directory plus the directory the
/// loaded model came from — the two places a user's models actually are when
/// they started the server from one of them.
fn handle_models(state: &ServerState, query: &str) -> Result<Value, ChatHttpError> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut push = |dir: PathBuf| {
        if dirs.len() < MAX_SCANNED_DIRECTORIES
            && !dirs.iter().any(|existing| same_dir(existing, &dir))
        {
            dirs.push(dir);
        }
    };

    match query_param(query, "dir") {
        Some(dir) if !dir.trim().is_empty() => push(local_path(Path::new(&dir), "dir")?),
        _ => {
            push(std::env::current_dir().map_err(ChatHttpError::internal)?);
            if let Some(loaded) = state.loaded.as_ref() {
                if let Some(parent) = loaded.path.parent() {
                    if !parent.as_os_str().is_empty() {
                        push(parent.to_path_buf());
                    }
                }
            }
        }
    }

    let loaded_path = state.loaded.as_ref().map(|loaded| loaded.path.as_path());
    let mut models: Vec<Value> = Vec::new();
    let mut scanned: Vec<Value> = Vec::new();
    for dir in &dirs {
        match catalog::list_directory(dir, loaded_path) {
            Ok(found) => {
                scanned.push(json!({
                    "dir": dir.display().to_string(),
                    "count": found.len(),
                    "error": Value::Null,
                }));
                models.extend(found);
            }
            // One unreadable directory does not fail the request: the user
            // asked what is available, and the answer is "these, and that one
            // could not be read".
            Err(error) => scanned.push(json!({
                "dir": dir.display().to_string(),
                "count": 0,
                "error": format!("{error}"),
            })),
        }
    }

    Ok(json!({
        "dirs": scanned,
        "count": models.len(),
        "models": models,
    }))
}

// =============================================================================
// POST /api/load
// =============================================================================

/// Swap the served model.
///
/// The new model is built completely before the old one is dropped, so a
/// failed load leaves the previous model serving rather than leaving the
/// server empty. The swap itself happens on the request loop, where by
/// construction no other request is in flight.
fn handle_load(state: &mut ServerState, body: &str) -> Result<Value, ChatHttpError> {
    let request: LoadRequest = serde_json::from_str(body)
        .map_err(|error| ChatHttpError::bad_request(format!("invalid JSON request: {error}")))?;

    // Neither the request nor the command line naming a context means "use the
    // model's own", which `Loaded::open` resolves once the header is read. The
    // request's number is checked here; `--max-context` was checked at
    // startup and is clamped to the model, as it is for a model named there.
    if request
        .max_context
        .is_some_and(|value| value == 0 || value > MAX_LOAD_CONTEXT)
    {
        return Err(ChatHttpError::bad_request(format!(
            "max_context must be between 1 and {MAX_LOAD_CONTEXT}"
        )));
    }
    let max_context = request.max_context.or(state.options.max_context);
    let template = request
        .chat_template
        .clone()
        .unwrap_or_else(|| state.options.chat_template.clone());

    let path = local_path(&request.path, "path")?;
    let tokenizer = request
        .tokenizer
        .as_deref()
        .map(|tokenizer| local_path(tokenizer, "tokenizer"))
        .transpose()?;
    if !path.is_file() {
        return Err(ChatHttpError::bad_request(format!(
            "{} is not a file",
            path.display()
        )));
    }

    // The new window is sized against free memory, and a long conversation
    // can have grown the current model's cache to gigabytes. That memory is
    // this server's own and is about to be replaced, so it is given back
    // first; the current model itself stays loaded until the new one has
    // loaded, so a failed load still leaves a working server.
    if let Some(current) = state.loaded.as_mut() {
        current.kv_cache.release();
        current.cached_tokens.clear();
    }

    let loaded = Loaded::open(&path, tokenizer.as_deref(), max_context, &template)
        // A load failure is about the user's own file and is what they need in
        // order to fix it, so it is reported rather than swallowed into a 500.
        .map_err(|error| ChatHttpError::bad_request(format!("{error:#}")))?;

    announce(&loaded);
    let info = loaded.info_json();
    state.loaded = Some(loaded);
    Ok(info)
}

/// Upper bound on a KV cache a single request may ask this server to allocate.
const MAX_LOAD_CONTEXT: usize = 1 << 20;

// =============================================================================
// Context windows and the memory they cost
// =============================================================================

/// The context window to actually use, given what was asked for.
///
/// No request means the window the model was stored with — that is the whole
/// point of a model declaring one. A request for more than the model has is
/// clamped down to it, because the RoPE tables were only built that far and
/// positions past them are undefined rather than merely inadvisable.
///
/// Shared with `rai run`, which resolves the same flag against the same rule.
pub(crate) fn resolve_context_window(requested: Option<usize>, model_max: usize) -> usize {
    requested.unwrap_or(model_max).min(model_max)
}

/// The context window actually chosen, and whether the machine forced it down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextChoice {
    /// The window to run with.
    pub context: usize,
    /// The window the model itself supports.
    pub model_max: usize,
    /// Set when no window was asked for and the model's own would not fit, so
    /// this one was chosen to fit the machine instead.
    pub reduced_to_fit: bool,
}

/// Smallest window worth falling back to. Below this a chat is not useful, so
/// failing with the numbers beats starting something unusable.
const MIN_AUTOFIT_CONTEXT: usize = 512;

/// Round auto-chosen windows down to this, so the reported number is a size a
/// person recognizes rather than an artifact of a division.
const AUTOFIT_GRANULARITY: usize = 256;

/// Leave this share of measured free memory unclaimed. The KV cache is not the
/// only thing that grows during generation - logits, the tokenizer and the
/// activations all want room - and a cache sized to the last free byte turns a
/// working session into a kill halfway through a reply.
const AUTOFIT_MEMORY_SHARE: f64 = 0.80;

/// Choose the window to run with, given what the machine can actually hold.
///
/// The model's own context is the ceiling, never a mandate. Asking for one
/// explicitly and being refused is correct - the user named a number and
/// deserves to be told it does not fit. Being refused a window nobody asked
/// for is not: it turns a model that ran yesterday into one that will not
/// start, which is a worse failure than a shorter conversation.
///
/// So an explicit request is honoured or refused, and an unset one fits itself
/// to the machine and says what it did.
pub(crate) fn choose_context_window(
    requested: Option<usize>,
    model_max: usize,
    kv_bytes_for: impl Fn(usize) -> usize,
) -> Result<ContextChoice> {
    fit_context_window(requested, model_max, kv_bytes_for, available_memory_bytes)
}

/// [`choose_context_window`] with the free memory as an input, so the rule is
/// tested against machines of every size rather than whatever the one running
/// the tests has free at that moment.
fn fit_context_window(
    requested: Option<usize>,
    model_max: usize,
    kv_bytes_for: impl Fn(usize) -> usize,
    available_memory: impl FnOnce() -> Option<u64>,
) -> Result<ContextChoice> {
    let ceiling = resolve_context_window(requested, model_max);
    // An explicit number is the user's call: honour it, and let the caller's
    // fit check refuse it by name if the machine cannot hold it.
    if requested.is_some() {
        return Ok(ContextChoice {
            context: ceiling,
            model_max,
            reduced_to_fit: false,
        });
    }
    let Some(available) = available_memory() else {
        // Nothing measurable to fit to; the fallible allocation still guards it.
        return Ok(ContextChoice {
            context: ceiling,
            model_max,
            reduced_to_fit: false,
        });
    };
    let budget = (available as f64 * AUTOFIT_MEMORY_SHARE) as u64;
    if kv_bytes_for(ceiling) as u64 <= budget {
        return Ok(ContextChoice {
            context: ceiling,
            model_max,
            reduced_to_fit: false,
        });
    }
    // The cache is exactly linear in the window (see `kv_cache_bytes`), so the
    // largest window that fits is one division rather than a search.
    let per_token = kv_bytes_for(1).max(1) as u64;
    let fitting = (budget / per_token) as usize;
    let rounded = (fitting / AUTOFIT_GRANULARITY) * AUTOFIT_GRANULARITY;
    let context = rounded.min(ceiling);
    anyhow::ensure!(
        context >= MIN_AUTOFIT_CONTEXT,
        "this model needs {} for even a {MIN_AUTOFIT_CONTEXT}-token context, and only {} of \
         memory is available. Close something, or use a model with fewer layers or KV heads.",
        crate::cli::format_bytes(kv_bytes_for(MIN_AUTOFIT_CONTEXT) as u64),
        crate::cli::format_bytes(available),
    );
    Ok(ContextChoice {
        context,
        model_max,
        reduced_to_fit: true,
    })
}

/// One line explaining an auto-fitted window, or nothing when the model's own
/// context was used as-is.
pub(crate) fn context_choice_note(
    choice: ContextChoice,
    kv_bytes_for: impl Fn(usize) -> usize,
) -> Option<String> {
    if !choice.reduced_to_fit {
        return None;
    }
    Some(format!(
        "Context: {} of the model's {} — its full window would need a {} KV cache, \
         and this machine has {} free. Pass --max-context to choose your own.",
        choice.context,
        choice.model_max,
        crate::cli::format_bytes(kv_bytes_for(choice.model_max) as u64),
        available_memory_bytes()
            .map(crate::cli::format_bytes)
            .unwrap_or_else(|| "an unknown amount".to_string()),
    ))
}

/// Refuse a KV cache that this machine plainly cannot hold, with the numbers.
///
/// Now that the window follows the model, a large model with a long stored
/// context asks for a cache measured in tens of gigabytes. The allocation
/// itself is fallible (`try_reserve_exact`), but on a system that overcommits
/// the reservation succeeds and it is the *filling* of the cache that fails —
/// as a kill, with no message, several seconds later. Measuring first is what
/// turns that into a sentence the user can act on.
///
/// Where available memory cannot be measured, this says nothing and leaves the
/// fallible allocation to do its job: a guess would refuse work that would
/// have succeeded.
pub(crate) fn check_kv_cache_fits(kv_bytes: usize, max_context: usize) -> Result<()> {
    let Some(available) = available_memory_bytes() else {
        return Ok(());
    };
    anyhow::ensure!(
        kv_bytes as u64 <= available,
        "a {max_context}-token context needs a {} KV cache, and only {} of memory is available. \
         Ask for a smaller window: --max-context on the command line, or \"max_context\" in \
         POST /api/load.",
        crate::cli::format_bytes(kv_bytes as u64),
        crate::cli::format_bytes(available),
    );
    Ok(())
}

// =============================================================================
// GET /api/system
// =============================================================================

/// What this machine is, measured rather than assumed.
///
/// RAI is CPU-only by construction — there is no GPU path to report on — and
/// nothing here is a derived or smoothed figure. Anything this platform will
/// not tell us is `null` rather than a plausible-looking number.
fn system_json(state: &ServerState) -> Value {
    let (avx2, fma, f16c) = cpu_features();
    let (rss_bytes, peak_rss_bytes) = process_memory_bytes();
    let model = match state.loaded.as_ref() {
        Some(loaded) => json!({
            "loaded": true,
            "weights_bytes": loaded.model.file_size(),
            "kv_cache_bytes": loaded.kv_cache.memory_bytes(),
            "max_context": loaded.max_context,
            "model_max_context": loaded.model.config.max_context,
        }),
        None => json!({
            "loaded": false,
            "weights_bytes": 0,
            "kv_cache_bytes": 0,
            "max_context": 0,
            "model_max_context": 0,
        }),
    };

    json!({
        "cpu": {
            "brand": cpu_brand(),
            "logical_cores": std::thread::available_parallelism().ok().map(|count| count.get()),
            // The pool the kernels actually run on, not a core count dressed
            // up as one: `configure_thread_pool` may have sized it from
            // RAYON_NUM_THREADS or from physical cores.
            "threads_used": rayon::current_num_threads(),
            "avx2": avx2,
            "fma": fma,
            "f16c": f16c,
            "kernel": if crate::gemm::has_avx2() { "AVX2 W4A8" } else { "scalar fallback" },
        },
        "memory": {
            "rss_bytes": rss_bytes,
            "peak_rss_bytes": peak_rss_bytes,
        },
        "model": model,
    })
}

/// The processor's own name for itself, from CPUID leaves 0x80000002-4.
///
/// No new dependency: the instruction is baseline on x86_64 and the leaves are
/// the ones every vendor fills in. Architectures without CPUID say nothing.
#[cfg(target_arch = "x86_64")]
fn cpu_brand() -> Option<String> {
    // CPUID is baseline on x86_64, so this cannot fault. The `unsafe` block is
    // written explicitly because `__cpuid` is an unsafe fn on the minimum Rust
    // this crate supports and a safe one on current toolchains; `unused_unsafe`
    // is allowed so the same source compiles warning-free on both. Removing it
    // builds here and fails the MSRV job, which is how it was caught.
    //
    // The extended leaves are read only once leaf 0x80000000 reports they
    // exist: a CPU without them returns its highest supported leaf's contents
    // instead of failing, which would otherwise be read as a brand string.
    #[allow(unused_unsafe)]
    let highest_extended = unsafe { std::arch::x86_64::__cpuid(0x8000_0000) }.eax;
    if highest_extended < 0x8000_0004 {
        return None;
    }
    let mut bytes = Vec::with_capacity(48);
    for leaf in [0x8000_0002u32, 0x8000_0003, 0x8000_0004] {
        #[allow(unused_unsafe)]
        let result = unsafe { std::arch::x86_64::__cpuid(leaf) };
        for word in [result.eax, result.ebx, result.ecx, result.edx] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    let brand = String::from_utf8_lossy(&bytes[..end]).trim().to_string();
    (!brand.is_empty()).then_some(brand)
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_brand() -> Option<String> {
    None
}

/// `(avx2, fma, f16c)` — the three the W4A8 kernels are written against.
#[cfg(target_arch = "x86_64")]
fn cpu_features() -> (bool, bool, bool) {
    (
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("f16c"),
    )
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_features() -> (bool, bool, bool) {
    (false, false, false)
}

/// `(resident, peak resident)` for this process, where the OS will say.
///
/// The peak half duplicates what `jobs.rs` reads for a conversion result. It
/// is written again here rather than shared because `jobs.rs` belongs to
/// another change in flight and its copy is private; the live figure is needed
/// regardless, and it comes from the same call. The two should collapse into
/// one platform module once both changes have landed.
#[cfg(windows)]
fn process_memory_bytes() -> (Option<u64>, Option<u64>) {
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ..Default::default()
    };
    // SAFETY: `counters` is a live, correctly sized PROCESS_MEMORY_COUNTERS,
    // and its size is passed as the API requires. The pseudo-handle from
    // GetCurrentProcess needs no closing.
    let ok = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<ProcessMemoryCounters>() as u32,
        )
    };
    if ok == 0 {
        return (None, None);
    }
    (
        Some(counters.working_set_size as u64),
        Some(counters.peak_working_set_size as u64),
    )
}

#[cfg(not(windows))]
fn process_memory_bytes() -> (Option<u64>, Option<u64>) {
    // /proc/self/status reports both in kB on Linux; elsewhere (macOS, the
    // BSDs) there is no such file and nothing is claimed.
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return (None, None);
    };
    let field = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|kb| kb * 1024)
    };
    (field("VmRSS:"), field("VmHWM:"))
}

/// Physical memory a new allocation could plausibly get, where the OS says.
#[cfg(windows)]
fn available_memory_bytes() -> Option<u64> {
    #[repr(C)]
    #[derive(Default)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_physical: u64,
        available_physical: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        ..Default::default()
    };
    // SAFETY: `status` is a live, correctly sized MEMORYSTATUSEX whose
    // `length` field is set as the API requires.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0).then_some(status.available_physical)
}

#[cfg(not(windows))]
fn available_memory_bytes() -> Option<u64> {
    // MemAvailable is the kernel's own estimate of what a new allocation can
    // have without swapping, which is exactly the question being asked.
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

// =============================================================================
// POST /api/inspect
// =============================================================================

/// Would `rai convert` accept this checkpoint? Answered without weights.
fn handle_inspect(body: &str) -> Result<Value, ChatHttpError> {
    let request: InspectRequest = serde_json::from_str(body)
        .map_err(|error| ChatHttpError::bad_request(format!("invalid JSON request: {error}")))?;
    let source = request.source.trim();
    if source.is_empty() {
        return Err(ChatHttpError::bad_request("source must not be empty"));
    }

    let group_size = request.group_size.unwrap_or(128);
    let embed_group_size = request.embed_group_size.unwrap_or(64);
    // No context in the request means the same thing it means on the command
    // line: follow the model's own window rather than a number this API
    // invented. A constant here would reintroduce the cap the CLI just lost.
    let max_context = request
        .max_context
        .unwrap_or(crate::convert::FOLLOW_MODEL_CONTEXT);

    let kind = catalog::classify(source);
    if kind == catalog::SourceKind::HuggingFaceId {
        // This crate ships no HTTP client and no TLS stack, and pulling one in
        // for a single config.json fetch would put a TLS implementation into
        // an offline inference engine. Say so precisely instead of failing
        // with "not found".
        return Err(ChatHttpError::not_implemented(format!(
            "'{source}' is not a path on this machine. /api/inspect reads local checkpoints \
             only: this build has no HTTP client, so it cannot fetch \
             https://huggingface.co/{source}/raw/main/config.json itself. Download that one \
             file (it is a few kilobytes) into a directory and inspect the directory."
        )));
    }

    local_path(Path::new(source), "source")?;
    let resolved = catalog::resolve_local(source).map_err(ChatHttpError::bad_request)?;
    let raw = std::fs::read(&resolved.config_path).map_err(|error| {
        ChatHttpError::bad_request(format!(
            "cannot read {}: {error}",
            resolved.config_path.display()
        ))
    })?;
    let hf: Value = serde_json::from_slice(&raw).map_err(|error| {
        ChatHttpError::bad_request(format!(
            "{} is not valid JSON: {error}",
            resolved.config_path.display()
        ))
    })?;

    Ok(catalog::inspect(
        source,
        kind,
        &hf,
        resolved.weights_dir.as_deref(),
        group_size,
        embed_group_size,
        max_context,
    ))
}

// =============================================================================
// POST /api/convert  +  GET /api/convert/<job_id>
// =============================================================================

fn handle_convert(state: &ServerState, body: &str) -> Result<Value, ChatHttpError> {
    let request: ConvertRequest = serde_json::from_str(body)
        .map_err(|error| ChatHttpError::bad_request(format!("invalid JSON request: {error}")))?;
    let source = local_path(&request.source, "source")?;
    let output = conversion_output(&request.output)?;
    let tokenizer_out = request
        .tokenizer_out
        .as_deref()
        .map(conversion_tokenizer_output)
        .transpose()?;

    let options = ConvertOptions {
        model_dir: source,
        output: Some(output),
        group_size: request.group_size.unwrap_or(128),
        embed_group_size: request.embed_group_size.unwrap_or(64),
        // The conversion API does not expose calibration: it would mean the
        // server reading a text file the client named, and the endpoint is
        // deliberately narrow.
        calibration_text: None,
        calibration_sequences: 16,
        calibration_seq_len: 512,
        max_context: request
            .max_context
            .unwrap_or(crate::convert::FOLLOW_MODEL_CONTEXT),
        tokenizer_out,
        // The job captures the narration; printing it into the server's
        // terminal too would interleave with request logging.
        quiet: true,
    };

    match state.jobs.start(options) {
        Ok(job_id) => Ok(json!({
            "job_id": job_id,
            "state": "running",
            "poll": format!("/api/convert/{job_id}"),
        })),
        Err(StartError::Busy) => Err(ChatHttpError::conflict(
            "a conversion is already running; conversions use every core, so they are run one \
             at a time",
        )),
        Err(StartError::Invalid(message)) => Err(ChatHttpError::bad_request(message)),
    }
}

/// A path from a request, refused if it reaches beyond this machine.
///
/// On Windows, opening `\\host\share\…` — or only listing it — makes the
/// machine authenticate to that host with the user's credentials. A path in a
/// request is never meant to leave this machine, so network (UNC) and
/// device-namespace forms are refused outright; anything else is the user's
/// own filesystem, which the server reads as the command line would.
fn local_path(raw: &Path, field: &str) -> Result<PathBuf, ChatHttpError> {
    if names_another_machine(raw) {
        return Err(ChatHttpError::bad_request(format!(
            "{field} must be a path on this machine; network (UNC) and device paths are \
             refused"
        )));
    }
    Ok(raw.to_path_buf())
}

#[cfg(windows)]
fn names_another_machine(path: &Path) -> bool {
    use std::path::{Component, Prefix};
    let local_disk = |component: Option<Component<'_>>| {
        matches!(component, Some(Component::Prefix(prefix))
            if matches!(prefix.kind(), Prefix::Disk(_) | Prefix::VerbatimDisk(_)))
    };
    // The raw text decides first, because the parser does not see every form
    // Windows opens: `\\server` with no share and the NT path `\??\UNC\…` both
    // parse as a plain rooted path, and both reach the network when opened.
    let raw = path.as_os_str().to_string_lossy();
    let separator = |c: Option<char>| matches!(c, Some('\\' | '/'));
    let mut chars = raw.chars();
    let (first, second) = (chars.next(), chars.next());
    if separator(first) && separator(second) {
        return !local_disk(path.components().next());
    }
    if separator(first) && raw[1..].starts_with("??") && separator(raw[3..].chars().next()) {
        return true;
    }
    match path.components().next() {
        Some(Component::Prefix(_)) => !local_disk(path.components().next()),
        _ => false,
    }
}

#[cfg(not(windows))]
fn names_another_machine(_path: &Path) -> bool {
    false
}

/// Where a conversion may write its model: a `.raimodel` file, and beside it
/// only that model's `<name>.chat.json`. The converter renames its finished
/// output over this path, so an unconstrained one let any request replace any
/// file the user can write.
fn conversion_output(raw: &Path) -> Result<PathBuf, ChatHttpError> {
    let output = local_path(raw, "output")?;
    let is_model = output
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case(crate::cli::MODEL_EXTENSION));
    if !is_model {
        return Err(ChatHttpError::bad_request(format!(
            "output must name a .{} file",
            crate::cli::MODEL_EXTENSION
        )));
    }
    if output.is_dir() {
        return Err(ChatHttpError::bad_request(
            "output names a directory, not a model file",
        ));
    }
    Ok(output)
}

/// Where a conversion may copy its tokenizer: a file named `tokenizer.json`,
/// the only name the runtime looks for beside a model.
fn conversion_tokenizer_output(raw: &Path) -> Result<PathBuf, ChatHttpError> {
    let path = local_path(raw, "tokenizer_out")?;
    let is_tokenizer = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case(crate::cli::TOKENIZER_FILENAME));
    if !is_tokenizer {
        return Err(ChatHttpError::bad_request(format!(
            "tokenizer_out must name a file called {}",
            crate::cli::TOKENIZER_FILENAME
        )));
    }
    Ok(path)
}

fn handle_job_poll(state: &ServerState, path: &str, query: &str) -> Result<Value, ChatHttpError> {
    let id = path.trim_start_matches("/api/convert/");
    if id.is_empty() || id.contains('/') {
        return Err(ChatHttpError::not_found("no such conversion job"));
    }
    let since = query_param(query, "since")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);

    let job = state
        .jobs
        .get(id)
        .ok_or_else(|| ChatHttpError::not_found("no such conversion job"))?;
    let job = job.lock().unwrap_or_else(|error| error.into_inner());
    Ok(job.snapshot(since))
}

// =============================================================================
// HTTP plumbing
// =============================================================================

fn respond_json(request: Request, value: &Value) {
    let response = Response::new(200, value.to_string().into_bytes())
        .with_header("Content-Type", "application/json")
        .with_header("X-Content-Type-Options", "nosniff");
    let _ = request.respond(response);
}

fn respond_json_error(request: Request, error: &ChatHttpError) {
    let body = serde_json::json!({"error": error.to_string()}).to_string();
    let response = Response::new(error.status, body.into_bytes())
        .with_header("Content-Type", "application/json")
        .with_header("X-Content-Type-Options", "nosniff");
    let _ = request.respond(response);
}

/// Listen on loopback.
///
/// `127.0.0.1` first, because that is what the printed URL and the launchers
/// assume; `[::1]` only on a machine with no IPv4 loopback, where `localhost`
/// resolves to `::1` alone (see [`HttpServer::bind_loopback`] for why a port
/// in use is an error rather than a reason to try the other family). Both are
/// loopback, so neither is reachable from another machine.
fn bind_loopback(port: u16) -> Result<HttpServer> {
    HttpServer::bind_loopback(port, MAX_CHAT_REQUEST_BYTES).map_err(|error| {
        if error.kind() == std::io::ErrorKind::AddrInUse {
            anyhow::anyhow!(
                "port {port} is already in use, probably by another `rai serve`; \
                 stop it or pass --port to pick another"
            )
        } else {
            anyhow::anyhow!("cannot listen on loopback port {port}: {error}")
        }
    })
}

/// Loopback host forms a browser can produce for our own port.
///
/// `[::1]` is here because the bind can land on IPv6 (see [`bind_loopback`]),
/// and because a browser resolving `localhost` to `::1` sends the bracketed
/// literal when the user typed it. It is still loopback: allowing it does not
/// widen who can reach the server, which is what the `Host` check defends.
///
/// The non-product `rai-server` implements this same defence separately, and
/// deliberately so. It is a different HTTP stack (axum against `cli::http`),
/// and sharing this function would mean `rai-server` depending on `rai-infer`
/// — which would put the whole inference engine, and eventually a network
/// stack, back on the wrong side of the product boundary. Two small
/// implementations of one check is the cheaper of the two costs; what makes
/// that safe is that each is tested against a real socket in its own crate.
fn is_allowed_host(host: &str, port: u16) -> bool {
    host.eq_ignore_ascii_case(&format!("localhost:{port}"))
        || host == format!("127.0.0.1:{port}")
        || host == format!("[::1]:{port}")
}

fn is_allowed_origin(origin: &str, port: u16) -> bool {
    origin.eq_ignore_ascii_case(&format!("http://localhost:{port}"))
        || origin == format!("http://127.0.0.1:{port}")
        || origin == format!("http://[::1]:{port}")
}

/// Split `"/api/convert/abc?since=4"` into `("/api/convert/abc", "since=4")`.
fn split_query(url: &str) -> (&str, &str) {
    match url.split_once('?') {
        Some((path, query)) => (path, query),
        None => (url, ""),
    }
}

/// One `key=value` from a query string, percent-decoded.
///
/// `+` is left alone rather than decoded to a space: these values are
/// filesystem paths, where a literal `+` is far likelier than a
/// form-encoded space, and `encodeURIComponent` emits `%20` anyway.
fn query_param(query: &str, key: &str) -> Option<String> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(name, _)| *name == key)
        .map(|(_, value)| percent_decode(value))
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 3 <= bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3])
                .ok()
                .and_then(|hex| u8::from_str_radix(hex, 16).ok());
            if let Some(byte) = hex {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Two paths that name the same directory, as far as the filesystem will say.
fn same_dir(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

/// The Studio interface, served at `/`.
///
/// Kept as a separate file so it can be edited as HTML rather than as a
/// Rust string literal, and compiled into the binary so `rai serve` stays a
/// single self-contained executable with no asset directory to install.
const CHAT_HTML: &str = include_str!("studio.html");

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn empty_prompt_returns_bad_request_instead_of_panicking() {
        let error = parse_chat_request(r#"{"message":"   "}"#).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("non-whitespace"));
    }

    #[test]
    fn malformed_json_returns_bad_request() {
        let error = parse_chat_request("not-json").unwrap_err();
        assert_eq!(error.status, 400);
    }

    /// Network and device paths never leave this machine's control: opening one
    /// on Windows authenticates to whatever host it names.
    #[cfg(windows)]
    #[test]
    fn network_and_device_paths_are_refused_on_windows() {
        for path in [
            r"\\attacker.example\share",
            "//attacker.example/share",
            r"\/attacker.example/share",
            r"\\?\UNC\attacker.example\share\x",
            r"\\.\PhysicalDrive0",
            r"\\?\GLOBALROOT\Device\Mup\attacker.example\share",
            // Not parsed as prefixes at all, and still opened over the network.
            r"\\attacker.example",
            r"\\attacker.example\",
            r"\??\UNC\attacker.example\share\x",
            "/??/UNC/attacker.example/share",
            r"\??\C:\models",
        ] {
            let error = local_path(Path::new(path), "dir").unwrap_err();
            assert_eq!(error.status, 400, "{path}");
            assert!(
                error.message.contains("network"),
                "{path}: {}",
                error.message
            );
        }
        for path in [
            r"C:\models",
            "C:models",
            "models",
            r"\\?\C:\models",
            r".\here",
            r"\models",
            r"\?models",
        ] {
            assert!(local_path(Path::new(path), "dir").is_ok(), "{path}");
        }
    }

    /// A conversion writes a model and its tokenizer and nothing else: the
    /// converter renames its output over the target, so the target must be
    /// one of those, whoever sent the request.
    #[test]
    fn conversion_targets_are_limited_to_models_and_tokenizers() {
        assert!(conversion_output(Path::new("out/model.raimodel")).is_ok());
        assert!(conversion_output(Path::new("out/Model.RAIMODEL")).is_ok());
        for refused in [".bashrc", "out/notes.txt", "out/model.raimodel.bak", "out/"] {
            assert_eq!(
                conversion_output(Path::new(refused)).unwrap_err().status,
                400,
                "{refused}"
            );
        }
        assert!(conversion_tokenizer_output(Path::new("out/tokenizer.json")).is_ok());
        for refused in [
            "out/startup.cmd",
            ".ssh/authorized_keys",
            "out/tokenizer.json.x",
        ] {
            assert_eq!(
                conversion_tokenizer_output(Path::new(refused))
                    .unwrap_err()
                    .status,
                400,
                "{refused}"
            );
        }
    }

    #[test]
    fn oversized_body_returns_payload_too_large() {
        let error = read_request_body(&Body::TooLarge).unwrap_err();
        assert_eq!(error.status, 413);
        let error = read_request_body(&Body::Complete(vec![0xFF, 0xFE])).unwrap_err();
        assert_eq!(
            error.status, 400,
            "a body that is not UTF-8 is a bad request"
        );
    }

    #[test]
    fn host_and_origin_checks_reject_dns_rebinding() {
        assert!(is_allowed_host("localhost:8090", 8090));
        assert!(is_allowed_origin("http://127.0.0.1:8090", 8090));
        assert!(!is_allowed_host("attacker.example:8090", 8090));
        assert!(!is_allowed_origin("https://attacker.example", 8090));
    }

    /// A window nobody asked for fits the machine instead of stopping the
    /// server starting. This is the regression that made a model with a
    /// 40,960-token stored context refuse to launch at all, where before it
    /// had served happily at 512. The free memory is given, not measured: read
    /// live, the test failed whenever the machine running it was busy.
    #[test]
    fn an_unrequested_window_fits_itself_to_the_machine() {
        // 1 MiB per token: the model's own 40,960 would want 40 GiB.
        let per_token = 1024 * 1024;
        let kv = |ctx: usize| ctx * per_token;
        let gib = 1u64 << 30;

        let roomy = fit_context_window(None, 40_960, kv, || Some(64 * gib)).unwrap();
        assert_eq!((roomy.context, roomy.reduced_to_fit), (40_960, false));
        let unmeasured = fit_context_window(None, 40_960, kv, || None).unwrap();
        assert_eq!(
            (unmeasured.context, unmeasured.reduced_to_fit),
            (40_960, false)
        );

        // 4 GiB free: 80% of it holds 3,276 tokens, reported at a round 3,072.
        let tight = fit_context_window(None, 40_960, kv, || Some(4 * gib)).unwrap();
        assert_eq!((tight.context, tight.reduced_to_fit), (3_072, true));
        assert!(tight.context.is_multiple_of(AUTOFIT_GRANULARITY));
        // A window that had to shrink says so rather than shrinking in silence.
        let note = context_choice_note(tight, kv).expect("a reduced window must explain");
        assert!(note.contains("--max-context"), "{note}");
        assert!(
            note.contains("40960"),
            "the note names the model's own window: {note}"
        );

        // Not room for even the smallest window: refused, with the numbers.
        let error = fit_context_window(None, 40_960, kv, || Some(256 << 20))
            .unwrap_err()
            .to_string();
        assert!(error.contains("512-token"), "{error}");
    }

    /// An explicitly requested window is the user's decision. It is honoured,
    /// never quietly shrunk — the fit check refuses it by name instead.
    #[test]
    fn an_explicit_window_is_never_silently_reduced() {
        let kv = |ctx: usize| ctx * 1024 * 1024;
        let choice = choose_context_window(Some(8192), 40_960, kv).expect("explicit is honoured");
        assert_eq!(choice.context, 8192);
        assert!(!choice.reduced_to_fit);
        assert!(context_choice_note(choice, kv).is_none());
        // And still clamped to what the model can actually address.
        let clamped = choose_context_window(Some(99_999), 4096, kv).expect("clamped");
        assert_eq!(clamped.context, 4096);
    }

    /// The server may bind IPv6 loopback, and a browser may resolve
    /// `localhost` to `::1`; neither is a way in from another machine.
    #[test]
    fn ipv6_loopback_is_allowed_and_other_hosts_still_are_not() {
        assert!(is_allowed_host("[::1]:8090", 8090));
        assert!(is_allowed_origin("http://[::1]:8090", 8090));
        // A different port is a different server, loopback or not.
        assert!(!is_allowed_host("[::1]:9999", 8090));
        assert!(!is_allowed_origin("http://[::1]:9999", 8090));
        // Not every IPv6 literal is loopback.
        assert!(!is_allowed_host("[::ffff:1.2.3.4]:8090", 8090));
        assert!(!is_allowed_host("[2001:db8::1]:8090", 8090));
    }

    /// Port 0 asks the OS for a free port, so this binds without racing a
    /// fixed port, and proves the returned server is reachable at loopback.
    #[test]
    fn bind_loopback_binds_a_loopback_address() {
        let server = bind_loopback(0).expect("loopback bind");
        let address = server.local_addr();
        assert!(address.ip().is_loopback(), "bound {address}, not loopback");
        assert_ne!(address.port(), 0, "port 0 should resolve to a real port");
    }

    #[test]
    fn expensive_request_values_are_clamped() {
        let request = ChatRequest {
            message: Some("hello".to_string()),
            messages: None,
            temperature: Some(100.0),
            max_tokens: Some(usize::MAX),
            repetition_penalty: None,
            ponder_strategy: Some("ensemble".to_string()),
            guidance_scale: Some(100.0),
            ensemble_n: Some(usize::MAX),
            noise_sigma: Some(100.0),
            entropy_threshold: None,
        };
        let error = build_ponder(&request).unwrap_err();
        assert_eq!(error.status, 400);
    }

    #[test]
    fn thread_form_is_parsed_and_validated() {
        let parsed = parse_chat_request(
            r#"{"messages":[{"role":"user","content":"hi"},
                {"role":"assistant","content":"hello"},
                {"role":"user","content":"more"}]}"#,
        )
        .expect("valid thread");
        assert_eq!(parsed.messages.as_ref().map(Vec::len), Some(3));

        // Ambiguous, empty, wrong-role, and wrong-tail threads are refused.
        for body in [
            r#"{"message":"hi","messages":[{"role":"user","content":"hi"}]}"#,
            r#"{"messages":[]}"#,
            r#"{}"#,
            r#"{"messages":[{"role":"system","content":"hi"}]}"#,
            r#"{"messages":[{"role":"user","content":"   "}]}"#,
            r#"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}"#,
        ] {
            let error = parse_chat_request(body).unwrap_err();
            assert_eq!(error.status, 400, "{body} should be a 400");
        }
    }

    #[test]
    fn thread_trimming_drops_whole_exchanges_and_never_the_last_turn() {
        use Role::{Assistant, User};
        let thread = [
            (User, ""),
            (Assistant, ""),
            (User, ""),
            (Assistant, ""),
            (User, ""),
        ];
        // Suffix token counts: the whole thread renders to 100 tokens, the
        // newest exchange plus the question to 60, the question alone to 30.
        // Odd starts would open on a reply and must never be tried.
        let counts = [100usize, usize::MAX, 60, usize::MAX, 30];
        let count_for = |start: usize| Ok(counts[start]);

        // 30 + 16 fits a 64-token window; 60 + 16 does not.
        assert_eq!(thread_trim_start(&thread, 64, 16, count_for).unwrap(), 4);
        // 60 + 16 fits 80: one exchange is dropped.
        assert_eq!(thread_trim_start(&thread, 80, 16, count_for).unwrap(), 2);
        // Everything fits: nothing is dropped.
        assert_eq!(thread_trim_start(&thread, 200, 16, count_for).unwrap(), 0);
        // Even the final turn alone is too long: it is still kept, because
        // whether to refuse it is prepare()'s decision, not the trimmer's.
        assert_eq!(thread_trim_start(&thread, 20, 16, count_for).unwrap(), 4);
        assert_eq!(
            thread_trim_start(&[(User, "")], 20, 16, |_| Ok(100)).unwrap(),
            0
        );
    }

    #[test]
    fn consecutive_turns_from_one_role_become_one() {
        let message = |role: &str, content: &str| ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        };
        let merged = merge_turns(&[
            message("user", "first"),
            message("user", "again"),
            message("assistant", "reply"),
            message("user", "next"),
        ]);
        assert_eq!(
            merged,
            vec![
                (
                    Role::User,
                    "first

again"
                        .to_string()
                ),
                (Role::Assistant, "reply".to_string()),
                (Role::User, "next".to_string()),
            ]
        );
    }

    #[test]
    fn unknown_ponder_strategy_is_rejected() {
        let error =
            parse_chat_request(r#"{"message":"hello","ponder_strategy":"typo"}"#).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("ponder_strategy"));
    }

    #[test]
    fn invalid_numeric_options_are_rejected() {
        for body in [
            r#"{"message":"hello","temperature":-0.5}"#,
            r#"{"message":"hello","temperature":2.5}"#,
            r#"{"message":"hello","max_tokens":0}"#,
            r#"{"message":"hello","guidance_scale":5}"#,
            r#"{"message":"hello","ensemble_n":9}"#,
            r#"{"message":"hello","noise_sigma":-1}"#,
            r#"{"message":"hello","entropy_threshold":33}"#,
        ] {
            assert_eq!(parse_chat_request(body).unwrap_err().status, 400, "{body}");
        }
        // Greedy decoding is a legitimate request, not an invalid temperature.
        assert!(parse_chat_request(r#"{"message":"hello","temperature":0}"#).is_ok());
    }

    #[test]
    fn over_context_and_out_of_vocabulary_prompts_are_rejected() {
        let error = validate_prompt_tokens(&[1, 2, 3], 2, 10).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("context window"));

        let error = validate_prompt_tokens(&[1, 10], 2, 10).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("vocabulary"));
    }

    #[test]
    fn internal_failures_are_sanitized() {
        let error = ChatHttpError::internal(r#"failed to open C:\\secret\\model.raimodel"#);
        assert_eq!(error.status, 500);
        assert_eq!(error.message, "internal server error");
    }

    #[test]
    fn a_windows_path_survives_the_query_string() {
        let (path, query) = split_query("/api/models?dir=E%3A%5Crai-work%5Cout");
        assert_eq!(path, "/api/models");
        assert_eq!(query_param(query, "dir").unwrap(), r"E:\rai-work\out");

        // A literal '+' in a path stays a '+', and a stray '%' is not an error.
        assert_eq!(query_param("dir=a+b%zz", "dir").unwrap(), "a+b%zz");
        assert_eq!(query_param("dir=x", "since"), None);
        assert_eq!(split_query("/api/info"), ("/api/info", ""));
    }

    #[test]
    fn the_poll_cursor_is_read_from_the_query_string() {
        let (path, query) = split_query("/api/convert/abc123?since=41");
        assert_eq!(path.trim_start_matches("/api/convert/"), "abc123");
        assert_eq!(
            query_param(query, "since").and_then(|v| v.parse::<usize>().ok()),
            Some(41)
        );
    }

    #[test]
    fn chatting_with_no_model_loaded_is_a_conflict_not_a_crash() {
        let error = no_model_loaded();
        assert_eq!(error.status, 409);
        assert!(error.message.contains("/api/load"));
    }

    #[test]
    fn a_huggingface_id_is_refused_with_the_reason_and_the_url() {
        let error = handle_inspect(r#"{"source":"Qwen/Qwen2.5-0.5B-Instruct"}"#).unwrap_err();
        assert_eq!(error.status, 501);
        assert!(error.message.contains("huggingface.co"), "{error}");
        assert!(error.message.contains("local checkpoints only"), "{error}");
    }

    #[test]
    fn inspect_rejects_a_body_that_is_not_a_request() {
        assert_eq!(handle_inspect("{}").unwrap_err().status, 400);
        assert_eq!(
            handle_inspect(r#"{"source":"  "}"#).unwrap_err().status,
            400
        );
        assert_eq!(handle_inspect("not-json").unwrap_err().status, 400);
    }

    #[test]
    fn inspect_reads_a_local_config_without_weights() {
        let dir = std::env::temp_dir().join(format!("rai-serve-inspect-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 3072,
                "vocab_size": 151936,
            })
            .to_string(),
        )
        .unwrap();

        let body = serde_json::json!({ "source": dir.display().to_string() }).to_string();
        let value = handle_inspect(&body).unwrap();
        assert_eq!(value["kind"], "local-config-only");
        assert_eq!(value["weights_checked"], false);
        assert_eq!(value["shape"]["num_layers"], 28);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn convert_requires_a_source_and_an_output() {
        let state = ServerState {
            loaded: None,
            jobs: Arc::new(Jobs::new()),
            options: ServeOptions {
                port: 8090,
                max_context: Some(512),
                chat_template: "auto".to_string(),
            },
        };
        assert_eq!(
            handle_convert(&state, r#"{"source":"x"}"#)
                .unwrap_err()
                .status,
            400
        );
        let body = r#"{"source":"no-such-dir-anywhere","output":"out.raimodel"}"#;
        let error = handle_convert(&state, body).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("not a directory"), "{error}");
    }

    #[test]
    fn loading_a_path_that_is_not_a_file_is_a_bad_request() {
        let mut state = ServerState {
            loaded: None,
            jobs: Arc::new(Jobs::new()),
            options: ServeOptions {
                port: 8090,
                max_context: Some(512),
                chat_template: "auto".to_string(),
            },
        };
        let error = handle_load(&mut state, r#"{"path":"no-such-model.raimodel"}"#).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(state.loaded.is_none());

        // An out-of-range context is refused before any file is touched.
        let error =
            handle_load(&mut state, r#"{"path":"x.raimodel","max_context":0}"#).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("max_context"), "{error}");
    }

    /// `--max-context` means the same whether the model is named at startup
    /// or loaded later: a number past the model's own is clamped to it. A
    /// server started empty with one used to refuse every load instead.
    #[test]
    fn the_command_line_context_is_clamped_for_a_later_load_too() {
        let dir = std::env::temp_dir().join(format!("rai-serve-clamp-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-tied.raimodel");
        let model_path = dir.join("tiny.raimodel");
        std::fs::copy(&fixture, &model_path).unwrap();
        let stored = RaiModel::load(&model_path).unwrap().config;
        let vocab: serde_json::Map<String, Value> = (0..stored.vocab_size)
            .map(|id| (format!("t{id}"), json!(id)))
            .collect();
        let tokenizer = json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null, "decoder": null,
            "model": { "type": "WordLevel", "unk_token": "t0", "vocab": vocab }
        });
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();

        let mut state = ServerState {
            loaded: None,
            jobs: Arc::new(Jobs::new()),
            options: ServeOptions {
                port: 8090,
                max_context: Some(2 * MAX_LOAD_CONTEXT),
                chat_template: "chatml".to_string(),
            },
        };
        let body = json!({ "path": model_path }).to_string();
        let result = handle_load(&mut state, &body);
        std::fs::remove_dir_all(&dir).ok();
        result.unwrap();
        assert_eq!(
            state.loaded.as_ref().unwrap().max_context,
            stored.max_context as usize
        );
        assert!(check_template_name("chatlm").is_err());
    }

    #[test]
    fn models_lists_an_explicit_directory_without_a_loaded_model() {
        let dir = std::env::temp_dir().join(format!("rai-serve-models-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("broken.raimodel"), vec![0u8; 128]).unwrap();

        let state = ServerState {
            loaded: None,
            jobs: Arc::new(Jobs::new()),
            options: ServeOptions {
                port: 8090,
                max_context: Some(512),
                chat_template: "auto".to_string(),
            },
        };
        let query = format!("dir={}", dir.display());
        let value = handle_models(&state, &query).unwrap();
        assert_eq!(value["count"], 1);
        assert_eq!(value["models"][0]["readable"], false);
        assert_eq!(value["dirs"].as_array().unwrap().len(), 1);

        // A directory that does not exist is reported, not fatal.
        let value = handle_models(&state, "dir=no-such-directory-at-all").unwrap();
        assert_eq!(value["count"], 0);
        assert!(value["dirs"][0]["error"].is_string());
        std::fs::remove_dir_all(&dir).ok();
    }

    // =========================================================================
    // Context windows and generation ceilings
    // =========================================================================

    /// The old default was the constant 512, which is what made a model stored
    /// with a 40k window behave as if it had a 512-token one.
    #[test]
    fn the_context_window_follows_the_model_and_is_clamped_to_it() {
        assert_eq!(resolve_context_window(None, 40_960), 40_960);
        assert_eq!(resolve_context_window(Some(512), 40_960), 512);
        assert_eq!(resolve_context_window(Some(1_000_000), 40_960), 40_960);
        assert_eq!(resolve_context_window(None, 512), 512);
    }

    /// The generation ceiling is the context the prompt leaves free, not a
    /// constant, and prompt plus reply never exceeds the window.
    #[test]
    fn the_generation_ceiling_is_whatever_the_prompt_leaves_free() {
        assert_eq!(remaining_generation_tokens(1, 2048), 2047);
        assert_eq!(remaining_generation_tokens(2000, 2048), 48);
        assert_eq!(remaining_generation_tokens(2048, 2048), 0);

        // The property the `done` record's `context_used` depends on.
        for prompt_len in [1usize, 2, 999, 2047, 2048] {
            let used = prompt_len + remaining_generation_tokens(prompt_len, 2048);
            assert_eq!(used, 2048, "prompt of {prompt_len}");
        }
    }

    /// A number that cannot fit is refused with both numbers, rather than
    /// quietly producing a short reply the caller cannot explain.
    #[test]
    fn max_tokens_beyond_the_remaining_context_is_refused_with_the_numbers() {
        let error = resolve_max_tokens(Some(4000), 100, 2048).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("4000"), "{error}");
        assert!(error.message.contains("1948"), "{error}");
        assert!(error.message.contains("2048"), "{error}");

        // Exactly the remaining count is allowed; one more is not.
        assert_eq!(resolve_max_tokens(Some(1948), 100, 2048).unwrap(), 1948);
        assert!(resolve_max_tokens(Some(1949), 100, 2048).is_err());
    }

    /// The Studio client cannot know the per-reply ceiling before it sends,
    /// because the ceiling is the context minus a prompt only this server can
    /// measure. It therefore learns the number by reading it back out of this
    /// refusal. That makes the wording of the refusal a contract, not prose:
    /// when the server moved from a fixed ceiling to a per-request one, the
    /// client's pattern was left matching the old wording, matched nothing,
    /// never learned a ceiling, and every reply on a context of 4096 or less
    /// dead-ended on the first message. This pins the two together so the next
    /// rewording fails here instead of in front of a reader.
    #[test]
    fn the_studio_client_can_read_the_ceiling_out_of_this_refusal() {
        let error = resolve_max_tokens(Some(2048), 24, 2048).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(
            error
                .message
                .contains("leaves 2024 of the 2048-token context window free"),
            "the refusal no longer states the room it has left: {error}"
        );

        // The pattern studio.html uses to lift the number back out.
        assert!(
            CHAT_HTML.contains("leaves (\\d+) of the (\\d+)-token context window free"),
            "studio.html no longer carries the pattern that reads the ceiling out \
             of this refusal, so it cannot learn the limit and will dead-end"
        );

        // The pattern it used to carry matched a message this server has never
        // sent. If it ever comes back, it is dead code pretending to recover.
        assert!(
            !CHAT_HTML.contains("max_tokens must be between 1 and"),
            "studio.html is matching a refusal this server does not send"
        );
    }

    /// A prompt with no room left for an answer is said plainly, rather than
    /// answered with an empty reply.
    #[test]
    fn a_prompt_that_fills_the_window_is_refused_rather_than_answered_with_nothing() {
        for requested in [None, Some(1), Some(200)] {
            let error = resolve_max_tokens(requested, 2048, 2048).unwrap_err();
            assert_eq!(error.status, 400);
            assert!(error.message.contains("no room for a reply"), "{error}");
        }
    }

    /// Asking for nothing is not the same as asking for too much: the default
    /// shortens to fit rather than failing.
    #[test]
    fn an_unrequested_token_count_takes_the_default_and_shortens_to_fit() {
        assert_eq!(
            resolve_max_tokens(None, 10, 40_960).unwrap(),
            DEFAULT_CHAT_GENERATION_TOKENS
        );
        // 2040-token prompt in a 2048 window leaves 8.
        assert_eq!(resolve_max_tokens(None, 2040, 2048).unwrap(), 8);
    }

    /// There is no longer a fixed ceiling to validate against, so a large
    /// `max_tokens` survives request validation and meets the real limit
    /// later, with the model's numbers in hand.
    #[test]
    fn a_large_token_request_is_no_longer_rejected_by_a_constant() {
        let request = parse_chat_request(r#"{"message":"hi","max_tokens":8192}"#).unwrap();
        assert_eq!(request.max_tokens, Some(8192));
        assert_eq!(
            parse_chat_request(r#"{"message":"hi","max_tokens":0}"#)
                .unwrap_err()
                .status,
            400
        );
    }

    /// The preflight only speaks when the platform gave it a number, and when
    /// it does it names the size and how to ask for less.
    #[test]
    fn an_unaffordable_kv_cache_is_refused_with_its_size() {
        // 16 EiB will not fit on any machine this runs on.
        let error = check_kv_cache_fits(usize::MAX, 131_072);
        match available_memory_bytes() {
            Some(_) => {
                let message = format!("{:#}", error.unwrap_err());
                assert!(message.contains("131072"), "{message}");
                assert!(message.contains("--max-context"), "{message}");
            }
            // Nothing measurable: the fallible allocation is left to answer.
            None => assert!(error.is_ok()),
        }
        // A cache of nothing always fits.
        assert!(check_kv_cache_fits(0, 1).is_ok());
    }

    // =========================================================================
    // Streaming
    // =========================================================================

    #[test]
    fn stop_reasons_are_the_three_the_contract_names() {
        assert_eq!(StopReason::Eos.as_str(), "eos");
        assert_eq!(StopReason::MaxTokens.as_str(), "max_tokens");
        assert_eq!(StopReason::StopSequence.as_str(), "stop_sequence");
    }

    /// Each record is one JSON object plus the one newline that separates
    /// ndjson records (the chunk framing around it is `http`'s, and tested
    /// there). Key order is serde_json's and is not part of the contract.
    #[test]
    fn each_ndjson_record_ends_in_exactly_one_newline() {
        let payload = ndjson_record(&json!({"type":"token","text":"hi"}));
        assert_eq!(payload.matches('\n').count(), 1, "{payload:?}");
        assert!(payload.ends_with('\n'));
        let record: Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(record["type"], "token");
        assert_eq!(record["text"], "hi");
    }

    /// A model that generates newlines must not be able to forge a record
    /// boundary. `serde_json` escapes them, so the payload stays one line.
    #[test]
    fn a_generated_newline_cannot_split_a_record() {
        let payload = ndjson_record(&json!({"type":"token","text":"a\nb\r\nc"}));
        assert_eq!(payload.matches('\n').count(), 1, "{payload:?}");
        assert!(payload.ends_with('\n'));
        assert_eq!(
            serde_json::from_str::<Value>(payload.trim_end()).unwrap()["text"],
            "a\nb\r\nc"
        );
    }

    /// A stop sequence is cut from the final text, so its opening bytes must
    /// never be streamed: a reader would otherwise watch `<|im_` appear and
    /// then have to un-see it.
    #[test]
    fn text_that_could_still_become_a_stop_sequence_is_held_back() {
        let stops = ["<|im_end|>", "<|im_start|>"];
        assert_eq!(emittable_len("hello", &stops), 5);
        assert_eq!(emittable_len("hello<|im_", &stops), 5);
        assert_eq!(emittable_len("hello<", &stops), 5);
        // `<|im` is a prefix of both; the longer hold wins.
        assert_eq!(emittable_len("<|im_st", &stops), 0);
        // With no stop sequences configured, nothing is ever held.
        assert_eq!(emittable_len("hello<|im_", &[] as &[&str]), 10);
    }

    /// A trailing replacement character nearly always means a multi-byte
    /// character is split across two tokens; showing it and then correcting it
    /// is worse than waiting one token.
    #[test]
    fn a_trailing_replacement_character_is_held_back_but_a_settled_one_is_not() {
        assert_eq!(emittable_len("caf\u{FFFD}", &[] as &[&str]), 3);
        assert_eq!(emittable_len("\u{FFFD}\u{FFFD}", &[] as &[&str]), 0);
        assert_eq!(
            emittable_len("caf\u{FFFD}e", &[] as &[&str]),
            "caf\u{FFFD}e".len()
        );
        // The boundary it returns is always a character boundary.
        let held = "caf\u{FFFD}";
        assert!(held.is_char_boundary(emittable_len(held, &[] as &[&str])));
    }

    // =========================================================================
    // The screening every route goes through
    // =========================================================================

    /// The streaming route is a POST like any other, so it is covered by the
    /// same checks — which is the point of screening before routing rather than
    /// inside each handler.
    #[test]
    fn the_streaming_route_is_screened_like_every_other_post() {
        let ok = |origin, fetch_site, content_type| Screened {
            host: Some("localhost:8090"),
            origin,
            fetch_site,
            content_type,
        };
        let json = Some("application/json");
        // The happy cases: Studio itself, and a script with no browser headers.
        assert!(screen_headers(
            Method::Post,
            true,
            ok(Some("http://localhost:8090"), Some("same-origin"), json),
            8090
        )
        .is_ok());
        assert!(screen_headers(Method::Post, true, ok(None, None, json), 8090).is_ok());
        // A rebound name for our port.
        let rebound = Screened {
            host: Some("attacker.example:8090"),
            ..ok(None, None, json)
        };
        assert_eq!(
            screen_headers(Method::Post, true, rebound, 8090)
                .unwrap_err()
                .status,
            403
        );
        // A page on another origin.
        assert_eq!(
            screen_headers(
                Method::Post,
                true,
                ok(Some("https://attacker.example"), None, json),
                8090
            )
            .unwrap_err()
            .status,
            403
        );
        // The form-post shape that needs no preflight.
        assert_eq!(
            screen_headers(
                Method::Post,
                true,
                ok(None, None, Some("text/plain;charset=UTF-8")),
                8090
            )
            .unwrap_err()
            .status,
            415
        );
        // A GET still has to pass the Host check.
        let get = ok(None, None, None);
        assert!(screen_headers(Method::Get, true, get, 8090).is_ok());
        assert_eq!(
            screen_headers(Method::Get, true, Screened::default(), 8090)
                .unwrap_err()
                .status,
            403
        );
    }

    /// A cross-site page can make a browser send a GET with no preflight —
    /// `<img src="http://127.0.0.1:8090/api/models?dir=\\host\share">` —
    /// and a GET route that reads a directory turned that into a Windows
    /// machine authenticating to the attacker's host. Every `/api` request is
    /// held to the origin checks, whatever its method; the page itself is not,
    /// because following a link to it is harmless.
    #[test]
    fn cross_site_gets_to_the_api_are_refused() {
        let from = |origin, fetch_site| Screened {
            host: Some("127.0.0.1:8090"),
            origin,
            fetch_site,
            content_type: None,
        };
        for site in ["cross-site", "same-site", "CROSS-SITE"] {
            assert_eq!(
                screen_headers(Method::Get, true, from(None, Some(site)), 8090)
                    .unwrap_err()
                    .status,
                403,
                "{site}"
            );
        }
        assert_eq!(
            screen_headers(
                Method::Get,
                true,
                from(Some("https://attacker.example"), None),
                8090
            )
            .unwrap_err()
            .status,
            403
        );
        // Studio, the user typing the URL, and a script are all fine.
        for site in [Some("same-origin"), Some("none"), None] {
            assert!(screen_headers(Method::Get, true, from(None, site), 8090).is_ok());
        }
        // The page: a link to it from anywhere is a harmless navigation.
        assert!(screen_headers(Method::Get, false, from(None, Some("cross-site")), 8090).is_ok());
    }

    // =========================================================================
    // Against a real socket
    // =========================================================================

    /// Send one raw request to `port` and return `(status line, body)`.
    ///
    /// Raw rather than through a client library because the things being
    /// asserted — a forged `Host`, a missing `Content-Type` — are exactly the
    /// things a well-behaved client will not send.
    #[cfg(test)]
    fn raw_request(port: u16, request: &str) -> (String, String) {
        use std::io::Write as _;
        let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect");
        socket.write_all(request.as_bytes()).expect("write");
        socket.flush().expect("flush");
        let mut response = String::new();
        socket.read_to_string(&mut response).expect("read");
        let (head, body) = response.split_once("\r\n\r\n").unwrap_or((&response, ""));
        let status = head.lines().next().unwrap_or_default().to_string();
        (status, body.to_string())
    }

    /// Run `body` against a real server on a real loopback port.
    fn with_server(body: impl FnOnce(u16)) {
        let server = bind_loopback(0).expect("loopback bind");
        let port = server.local_addr().port();
        let state = ServerState {
            loaded: None,
            jobs: Arc::new(Jobs::new()),
            options: ServeOptions {
                port,
                max_context: None,
                chat_template: "auto".to_string(),
            },
        };
        std::thread::scope(|scope| {
            scope.spawn(|| serve_on(&server, state, port));
            body(port);
            server.unblock();
        });
    }

    #[test]
    fn the_stream_route_rejects_a_forged_host_and_a_foreign_origin() {
        with_server(|port| {
            let (status, _) = raw_request(
                port,
                "POST /api/chat/stream HTTP/1.1\r\nHost: attacker.example\r\n\
                 Content-Type: application/json\r\nContent-Length: 17\r\n\
                 Connection: close\r\n\r\n{\"message\":\"hi\"}\n",
            );
            assert!(status.contains("403"), "{status}");

            let (status, _) = raw_request(
                port,
                &format!(
                    "POST /api/chat/stream HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
                     Origin: https://attacker.example\r\nContent-Type: application/json\r\n\
                     Content-Length: 17\r\nConnection: close\r\n\r\n{{\"message\":\"hi\"}}\n"
                ),
            );
            assert!(status.contains("403"), "{status}");

            let (status, _) = raw_request(
                port,
                &format!(
                    "POST /api/chat/stream HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
                     Content-Type: text/plain\r\nContent-Length: 17\r\n\
                     Connection: close\r\n\r\n{{\"message\":\"hi\"}}\n"
                ),
            );
            assert!(status.contains("415"), "{status}");
        });
    }

    /// With nothing loaded the streaming route answers with an ordinary status
    /// code, not with a 200 whose first record is an error.
    #[test]
    fn the_stream_route_answers_a_missing_model_before_it_opens_a_stream() {
        with_server(|port| {
            let (status, body) = raw_request(
                port,
                &format!(
                    "POST /api/chat/stream HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
                     Origin: http://127.0.0.1:{port}\r\nContent-Type: application/json\r\n\
                     Content-Length: 16\r\nConnection: close\r\n\r\n{{\"message\":\"hi\"}}"
                ),
            );
            assert!(status.contains("409"), "{status}");
            assert!(body.contains("/api/load"), "{body}");
        });
    }

    /// `/api/system` reports the machine even with no model loaded, and every
    /// field the UI contract names is present.
    #[test]
    fn the_system_route_reports_this_machine() {
        with_server(|port| {
            let (status, body) = raw_request(
                port,
                &format!(
                    "GET /api/system HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
                     Connection: close\r\n\r\n"
                ),
            );
            assert!(status.contains("200"), "{status}");
            let value: Value = serde_json::from_str(&body).expect(&body);

            assert!(value["cpu"]["logical_cores"].as_u64().unwrap() >= 1);
            assert!(value["cpu"]["threads_used"].as_u64().unwrap() >= 1);
            for flag in ["avx2", "fma", "f16c"] {
                assert!(value["cpu"][flag].is_boolean(), "{flag}");
            }
            // Exactly one of the two kernels, never an invented third.
            let kernel = value["cpu"]["kernel"].as_str().unwrap();
            assert!(
                kernel == "AVX2 W4A8" || kernel == "scalar fallback",
                "{kernel}"
            );
            // Nothing fabricated: no GPU, no utilisation percentage.
            assert!(value["gpu"].is_null());
            assert!(value["cpu"]["utilisation"].is_null());
            assert!(value["cpu"]["usage"].is_null());
            // Measured or null, never a placeholder zero.
            for field in ["rss_bytes", "peak_rss_bytes"] {
                let measured = &value["memory"][field];
                assert!(
                    measured.is_null() || measured.as_u64().unwrap() > 0,
                    "{field}"
                );
            }
            assert_eq!(value["model"]["loaded"], false);
            assert_eq!(value["model"]["max_context"], 0);
        });
    }

    /// The brand string, when the architecture can produce one, is a name and
    /// not padding. On x86_64 there is no "cannot produce one": the brand
    /// leaves have been mandatory since long before AVX2, which this engine
    /// needs anyway.
    #[test]
    fn the_cpu_brand_is_a_name_or_nothing() {
        let brand = cpu_brand();
        if cfg!(target_arch = "x86_64") {
            let brand = brand.expect("x86_64 always has the CPUID brand leaves");
            assert_eq!(brand.trim(), brand, "{brand:?}");
            assert!(!brand.is_empty());
            assert!(!brand.contains('\0'), "{brand:?}");
        } else {
            assert!(brand.is_none());
        }
    }
}
