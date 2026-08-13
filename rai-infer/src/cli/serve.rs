//! `rai serve` — the local HTTP API, and the chat UI on top of it.
//!
//! This is the body of the old `rai-chat` binary, moved into the library so
//! that one `rai` binary can host it. The server binds loopback only and
//! rejects any request whose `Host` header is not `localhost`/`127.0.0.1` at
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
//! | `POST /api/chat` | generate a reply |
//! | `GET /api/models` | `.raimodel` files in a directory, from headers only |
//! | `POST /api/load` | load a model into this server |
//! | `POST /api/inspect` | would `rai convert` accept this checkpoint? |
//! | `POST /api/convert` | start a conversion, return a job id |
//! | `GET /api/convert/<id>` | poll that job |
//! | `GET /api/convert` | every job this server has run |
//!
//! Paths in requests are the user's own filesystem paths and are not confined
//! to a root: this server is the local application's own back end, reachable
//! only from the machine it runs on, and it can do exactly what the `rai`
//! command line can do for the user running it — no more.

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::{json, Value};
use tiny_http::{Header, Method, Response, Server, StatusCode};

use crate::chat_template::ChatTemplate;
use crate::cli::catalog;
use crate::cli::jobs::{Jobs, StartError};
use crate::cli::{load_tokenizer, resolve_tokenizer};
use crate::convert::ConvertOptions;
use crate::model::{InferenceWork, RaiModel};
use crate::ponder::{pondered_forward, PonderConfig};
use crate::sampler::{apply_repetition_penalty, sample_token, SamplerConfig};

const MAX_CHAT_REQUEST_BYTES: usize = 64 * 1024;
const MAX_CHAT_GENERATION_TOKENS: usize = 512;
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
    /// Context window to allocate the KV cache for, in tokens
    #[arg(long, default_value = "512")]
    pub max_context: usize,
    /// Chat template: auto, none, few-shot, mistral, llama3, chatml, zephyr
    #[arg(long, default_value = "auto")]
    pub chat_template: String,
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
    template: ChatTemplate,
    path: PathBuf,
    tokenizer_path: PathBuf,
    load_ms: u64,
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
        requested_context: usize,
        chat_template: &str,
    ) -> Result<Self> {
        anyhow::ensure!(
            requested_context > 0,
            "max_context must be greater than zero"
        );
        anyhow::ensure!(
            model_path.is_file(),
            "{} is not a file",
            model_path.display()
        );
        let started = Instant::now();
        let tokenizer_path = resolve_tokenizer(model_path, tokenizer_arg)?;
        let model = RaiModel::load(model_path)?;
        let max_context = requested_context.min(model.config.max_context as usize);
        let tokenizer = load_tokenizer(&tokenizer_path)?;
        let template = ChatTemplate::from_str_arg(chat_template, &tokenizer);
        Ok(Self {
            model,
            tokenizer,
            max_context,
            template,
            path: model_path.to_path_buf(),
            tokenizer_path,
            load_ms: started.elapsed().as_millis() as u64,
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
            "chat_template": self.template.display_name(),
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
            "kv_cache_bytes": self.model.kv_cache_bytes(self.max_context),
            "load_ms": self.load_ms,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    message: String,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    ponder_strategy: Option<String>,
    guidance_scale: Option<f32>,
    ensemble_n: Option<usize>,
    noise_sigma: Option<f32>,
    entropy_threshold: Option<f32>,
}

fn parse_chat_request(req_body: &str) -> Result<ChatRequest, ChatHttpError> {
    let request: ChatRequest = serde_json::from_str(req_body)
        .map_err(|error| ChatHttpError::bad_request(format!("invalid JSON request: {error}")))?;

    if request.message.trim().is_empty() {
        return Err(ChatHttpError::bad_request(
            "message must contain at least one non-whitespace character",
        ));
    }

    validate_chat_options(&request)?;

    Ok(request)
}

fn read_request_body(reader: &mut dyn Read) -> Result<String, ChatHttpError> {
    let mut bytes = Vec::new();
    reader
        .take((MAX_CHAT_REQUEST_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(ChatHttpError::internal)?;

    if bytes.len() > MAX_CHAT_REQUEST_BYTES {
        return Err(ChatHttpError::payload_too_large());
    }

    String::from_utf8(bytes)
        .map_err(|_| ChatHttpError::bad_request("request body must be valid UTF-8"))
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
    validate_optional_f32("temperature", req.temperature, 0.01, 2.0)?;
    if req
        .max_tokens
        .is_some_and(|value| !(1..=MAX_CHAT_GENERATION_TOKENS).contains(&value))
    {
        return Err(ChatHttpError::bad_request(format!(
            "max_tokens must be between 1 and {MAX_CHAT_GENERATION_TOKENS}"
        )));
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

fn handle_generate(state: &Loaded, chat_req: &ChatRequest) -> Result<String, ChatHttpError> {
    validate_chat_options(chat_req)?;
    let temperature = chat_req.temperature.unwrap_or(0.7);
    let max_tokens = chat_req.max_tokens.unwrap_or(200);
    let ponder_config = build_ponder(chat_req)?;

    let sampler_config = SamplerConfig {
        temperature,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.1,
    };

    // Format prompt using the configured chat template
    let prompt = state.template.format_prompt(&chat_req.message);
    let encoding = state
        .tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|_| ChatHttpError::bad_request("message could not be tokenized"))?;
    let prompt_tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();

    let max_ctx = state
        .max_context
        .min(state.model.config.max_context as usize);
    validate_prompt_tokens(
        &prompt_tokens,
        max_ctx,
        state.model.config.vocab_size as usize,
    )?;
    let mut kv_cache = state
        .model
        .create_kv_cache(max_ctx)
        .map_err(ChatHttpError::internal)?;
    let mut work = InferenceWork::new();
    let mut work2 = InferenceWork::new();
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut all_tokens = prompt_tokens.clone();
    let mut pos = 0;

    // Prefill
    let t_prefill = Instant::now();
    // Leave the final prompt token for the first decode pass. Processing every prompt token
    // here and then processing the last one again at the next position duplicates it.
    for &token_id in &prompt_tokens[..prompt_tokens.len() - 1] {
        let _ = pondered_forward(
            &state.model,
            token_id,
            pos,
            &mut kv_cache,
            &PonderConfig::none(),
            &mut work,
            &mut work2,
            &mut rng,
        )
        .map_err(ChatHttpError::internal)?;
        pos += 1;
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    // Decode
    let t_decode = Instant::now();
    let mut generated_text = String::new();
    let mut tokens_generated = 0;
    let mut total_passes = 0;
    let mut hard_tokens = 0;

    for _ in 0..max_tokens {
        if pos >= max_ctx {
            break;
        }
        let last_token = all_tokens.last().copied().ok_or_else(|| {
            ChatHttpError::bad_request("message produced no usable prompt tokens")
        })?;
        let (mut logits, metrics) = pondered_forward(
            &state.model,
            last_token,
            pos,
            &mut kv_cache,
            &ponder_config,
            &mut work,
            &mut work2,
            &mut rng,
        )
        .map_err(ChatHttpError::internal)?;
        total_passes += metrics.forward_passes;
        if metrics.was_hard_token {
            hard_tokens += 1;
        }

        apply_repetition_penalty(&mut logits, &all_tokens, sampler_config.repetition_penalty);
        let next_token = sample_token(&mut logits, &sampler_config, &mut rng);

        // Check all common EOS tokens
        let is_eos = ["</s>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"]
            .iter()
            .any(|tok| {
                state
                    .tokenizer
                    .token_to_id(tok)
                    .is_some_and(|id| next_token == id as usize)
            });
        if is_eos {
            break;
        }

        all_tokens.push(next_token);
        pos += 1;
        tokens_generated += 1;

        // Decode entire generated suffix for correct SentencePiece spacing
        let gen_ids: Vec<u32> = all_tokens[prompt_tokens.len()..]
            .iter()
            .map(|&t| t as u32)
            .collect();
        generated_text = state.tokenizer.decode(&gen_ids, false).unwrap_or_default();

        // Stop if model generates a template-specific stop sequence
        let mut should_stop = false;
        for stop in state.template.stop_sequences() {
            if generated_text.contains(stop) {
                generated_text = generated_text.split(stop).next().unwrap_or("").to_string();
                should_stop = true;
                break;
            }
        }
        if should_stop {
            break;
        }
    }

    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let decode_tps = if decode_ms > 0.0 {
        tokens_generated as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let avg_passes = if tokens_generated > 0 {
        total_passes as f64 / tokens_generated as f64
    } else {
        0.0
    };
    let hard_pct = if tokens_generated > 0 {
        100.0 * hard_tokens as f64 / tokens_generated as f64
    } else {
        0.0
    };

    let response = serde_json::json!({
        "text": generated_text.trim(),
        "tokens": tokens_generated,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "tok_per_sec": decode_tps,
        "avg_passes": avg_passes,
        "hard_tokens_pct": hard_pct,
        "strategy": format!("{:?}", ponder_config.strategy),
    });

    Ok(response.to_string())
}

pub fn run(args: &ServeArgs) -> Result<()> {
    crate::gemm::configure_thread_pool();
    if args.options.max_context == 0 {
        anyhow::bail!("--max-context must be greater than zero");
    }

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

    let mut state = ServerState {
        loaded,
        jobs: Arc::new(Jobs::new()),
        options: args.options.clone(),
    };

    let addr = format!("127.0.0.1:{}", args.options.port);
    let server = Server::http(&addr)
        .map_err(|e| anyhow::anyhow!("cannot listen on {addr}: {e}"))
        .with_context(|| {
            format!(
                "another process may already be using port {}; pass --port to pick another",
                args.options.port
            )
        })?;
    let bound_port = server
        .server_addr()
        .to_ip()
        .ok_or_else(|| anyhow::anyhow!("server did not bind an IP socket"))?
        .port();
    eprintln!("\n  Chat UI: http://localhost:{bound_port}\n");
    eprintln!("  Press Ctrl+C to stop.\n");

    // The server is deliberately single-threaded: requests are handled one at
    // a time on this loop, so one generation saturates the CPU without a
    // second request competing for cores, and no synchronization is needed
    // around the model state. Conversions are the one thing that must not be
    // serialized with it — they take minutes — so they run on their own
    // threads and this loop only ever reads their progress.
    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        let (path, query) = split_query(&url);
        let method = request.method().clone();

        let host = request
            .headers()
            .iter()
            .find(|header| header.field.equiv("Host"))
            .map(|header| header.value.as_str());
        if !host.is_some_and(|host| is_allowed_host(host, bound_port)) {
            respond_json_error(request, &ChatHttpError::forbidden("invalid Host header"));
            continue;
        }

        // Every POST, not just /api/chat: loading a model and starting a
        // conversion are at least as worth protecting as generating text.
        if method == Method::Post {
            let origin = request
                .headers()
                .iter()
                .find(|header| header.field.equiv("Origin"))
                .map(|header| header.value.as_str());
            if origin.is_some_and(|origin| !is_allowed_origin(origin, bound_port)) {
                respond_json_error(
                    request,
                    &ChatHttpError::forbidden("cross-origin requests are not allowed"),
                );
                continue;
            }

            let is_json = request
                .headers()
                .iter()
                .find(|header| header.field.equiv("Content-Type"))
                .map(|header| header.value.as_str())
                .is_some_and(|value| {
                    value
                        .split(';')
                        .next()
                        .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
                });
            if !is_json {
                respond_json_error(request, &ChatHttpError::unsupported_media_type());
                continue;
            }
        }

        match (method, path) {
            (Method::Get, "/") | (Method::Get, "/index.html") => {
                let resp = Response::from_data(CHAT_HTML.as_bytes().to_vec())
                    .with_header(
                        Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap(),
                    )
                    .with_header(
                        Header::from_bytes(
                            "Content-Security-Policy",
                            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
                        )
                        .unwrap(),
                    )
                    .with_header(
                        Header::from_bytes("X-Content-Type-Options", "nosniff").unwrap(),
                    );
                let _ = request.respond(resp);
            }
            (Method::Post, "/api/chat") => {
                let body = match read_request_body(request.as_reader()) {
                    Ok(body) => body,
                    Err(error) => {
                        respond_json_error(request, &error);
                        continue;
                    }
                };
                let chat_request = match parse_chat_request(&body) {
                    Ok(chat_request) => chat_request,
                    Err(error) => {
                        respond_json_error(request, &error);
                        continue;
                    }
                };

                let outcome = match state.loaded.as_ref() {
                    Some(loaded) => handle_generate(loaded, &chat_request),
                    None => Err(no_model_loaded()),
                };
                match outcome {
                    Ok(json) => {
                        let resp = Response::from_data(json.into_bytes()).with_header(
                            Header::from_bytes("Content-Type", "application/json").unwrap(),
                        );
                        let _ = request.respond(resp);
                    }
                    Err(error) => respond_json_error(request, &error),
                }
            }
            (Method::Get, "/api/info") => {
                let info = match state.loaded.as_ref() {
                    Some(loaded) => loaded.info_json(),
                    None => json!({
                        "loaded": false,
                        "model_path": Value::Null,
                        "chat_template": state.options.chat_template,
                        "max_context": state.options.max_context,
                    }),
                };
                respond_json(request, &info);
            }
            (Method::Get, "/api/models") => match handle_models(&state, query) {
                Ok(value) => respond_json(request, &value),
                Err(error) => respond_json_error(request, &error),
            },
            (Method::Post, "/api/load") => {
                let body = match read_request_body(request.as_reader()) {
                    Ok(body) => body,
                    Err(error) => {
                        respond_json_error(request, &error);
                        continue;
                    }
                };
                match handle_load(&mut state, &body) {
                    Ok(value) => respond_json(request, &value),
                    Err(error) => respond_json_error(request, &error),
                }
            }
            (Method::Post, "/api/inspect") => {
                let body = match read_request_body(request.as_reader()) {
                    Ok(body) => body,
                    Err(error) => {
                        respond_json_error(request, &error);
                        continue;
                    }
                };
                match handle_inspect(&body) {
                    Ok(value) => respond_json(request, &value),
                    Err(error) => respond_json_error(request, &error),
                }
            }
            (Method::Post, "/api/convert") => {
                let body = match read_request_body(request.as_reader()) {
                    Ok(body) => body,
                    Err(error) => {
                        respond_json_error(request, &error);
                        continue;
                    }
                };
                match handle_convert(&state, &body) {
                    Ok(value) => {
                        let resp = Response::from_data(value.to_string().into_bytes())
                            .with_status_code(StatusCode(202))
                            .with_header(
                                Header::from_bytes("Content-Type", "application/json").unwrap(),
                            );
                        let _ = request.respond(resp);
                    }
                    Err(error) => respond_json_error(request, &error),
                }
            }
            (Method::Get, "/api/convert") => respond_json(request, &state.jobs.list()),
            (Method::Get, path) if path.starts_with("/api/convert/") => {
                match handle_job_poll(&state, path, query) {
                    Ok(value) => respond_json(request, &value),
                    Err(error) => respond_json_error(request, &error),
                }
            }
            (Method::Options, _) => respond_json_error(
                request,
                &ChatHttpError::forbidden("cross-origin preflight is not allowed"),
            ),
            _ => {
                let _ = request.respond(
                    Response::from_data(b"404".to_vec()).with_status_code(StatusCode(404)),
                );
            }
        }
    }
    Ok(())
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
    eprintln!("Chat template: {}", loaded.template.display_name());
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
        Some(dir) if !dir.trim().is_empty() => push(PathBuf::from(dir)),
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

    let max_context = request.max_context.unwrap_or(state.options.max_context);
    if max_context == 0 || max_context > MAX_LOAD_CONTEXT {
        return Err(ChatHttpError::bad_request(format!(
            "max_context must be between 1 and {MAX_LOAD_CONTEXT}"
        )));
    }
    let template = request
        .chat_template
        .clone()
        .unwrap_or_else(|| state.options.chat_template.clone());

    if !request.path.is_file() {
        return Err(ChatHttpError::bad_request(format!(
            "{} is not a file",
            request.path.display()
        )));
    }

    let loaded = Loaded::open(
        &request.path,
        request.tokenizer.as_deref(),
        max_context,
        &template,
    )
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
    let max_context = request.max_context.unwrap_or(2048);

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

    let options = ConvertOptions {
        model_dir: request.source,
        output: Some(request.output),
        group_size: request.group_size.unwrap_or(128),
        embed_group_size: request.embed_group_size.unwrap_or(64),
        max_context: request.max_context.unwrap_or(2048),
        tokenizer_out: request.tokenizer_out,
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

fn respond_json(request: tiny_http::Request, value: &Value) {
    let response = Response::from_data(value.to_string().into_bytes())
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
        .with_header(Header::from_bytes("X-Content-Type-Options", "nosniff").unwrap());
    let _ = request.respond(response);
}

fn respond_json_error(request: tiny_http::Request, error: &ChatHttpError) {
    let body = serde_json::json!({"error": error.to_string()}).to_string();
    let response = Response::from_data(body.into_bytes())
        .with_status_code(StatusCode(error.status))
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap());
    let _ = request.respond(response);
}

fn is_allowed_host(host: &str, port: u16) -> bool {
    host.eq_ignore_ascii_case(&format!("localhost:{port}")) || host == format!("127.0.0.1:{port}")
}

fn is_allowed_origin(origin: &str, port: u16) -> bool {
    origin.eq_ignore_ascii_case(&format!("http://localhost:{port}"))
        || origin == format!("http://127.0.0.1:{port}")
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
    use std::io::Cursor;

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

    #[test]
    fn oversized_body_returns_payload_too_large() {
        let mut body = Cursor::new(vec![b'x'; MAX_CHAT_REQUEST_BYTES + 1]);
        let error = read_request_body(&mut body).unwrap_err();
        assert_eq!(error.status, 413);
    }

    #[test]
    fn host_and_origin_checks_reject_dns_rebinding() {
        assert!(is_allowed_host("localhost:8090", 8090));
        assert!(is_allowed_origin("http://127.0.0.1:8090", 8090));
        assert!(!is_allowed_host("attacker.example:8090", 8090));
        assert!(!is_allowed_origin("https://attacker.example", 8090));
    }

    #[test]
    fn expensive_request_values_are_clamped() {
        let request = ChatRequest {
            message: "hello".to_string(),
            temperature: Some(100.0),
            max_tokens: Some(usize::MAX),
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
    fn unknown_ponder_strategy_is_rejected() {
        let error =
            parse_chat_request(r#"{"message":"hello","ponder_strategy":"typo"}"#).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("ponder_strategy"));
    }

    #[test]
    fn invalid_numeric_options_are_rejected() {
        for body in [
            r#"{"message":"hello","temperature":0}"#,
            r#"{"message":"hello","max_tokens":0}"#,
            r#"{"message":"hello","guidance_scale":5}"#,
            r#"{"message":"hello","ensemble_n":9}"#,
            r#"{"message":"hello","noise_sigma":-1}"#,
            r#"{"message":"hello","entropy_threshold":33}"#,
        ] {
            assert_eq!(parse_chat_request(body).unwrap_err().status, 400, "{body}");
        }
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
                max_context: 512,
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
                max_context: 512,
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
                max_context: 512,
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
}
