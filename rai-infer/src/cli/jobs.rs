//! Background conversion jobs for `rai serve`.
//!
//! Converting a 7B checkpoint takes minutes. The HTTP server handles requests
//! one at a time on its accept loop, so running a conversion there would make
//! the whole UI — including the progress display — unreachable for the
//! duration. Instead `POST /api/convert` hands the work to a thread and
//! returns a job id, and the UI polls `GET /api/convert/<id>`.
//!
//! Polling rather than SSE or a websocket, deliberately: a poll is one plain
//! request that the existing Host/Origin checks already cover, it survives the
//! client sleeping or reloading, and there is no half-open stream to reason
//! about when a conversion fails.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::Rng;

use crate::convert::{
    convert_with_progress, ConvertOptions, ConvertProgress, ConvertSummary, FOLLOW_MODEL_CONTEXT,
};

/// Log lines kept per job. A 100-layer model produces a few hundred; the cap
/// only exists so a pathological model cannot grow the server without bound.
const MAX_LOG_LINES: usize = 20_000;

/// Finished jobs kept for polling before the oldest is dropped.
const MAX_RETAINED_JOBS: usize = 64;

/// Conversions allowed to run at once. One: they are CPU-saturating (rayon
/// across every core), so a second concurrent job makes both slower and
/// doubles peak memory.
const MAX_RUNNING_JOBS: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobPhase {
    Running,
    Done,
    Error,
}

impl JobPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            JobPhase::Running => "running",
            JobPhase::Done => "done",
            JobPhase::Error => "error",
        }
    }
}

/// What a finished conversion produced.
///
/// The size figures are the point of the whole exercise and used to be
/// half-recorded: the output size was kept and the *input* size was not, so
/// nothing downstream could say what a conversion had bought. Everything a
/// caller needs to answer "was this worth it, and what will it cost to run"
/// now comes out of the conversion itself rather than being re-derived from
/// the filesystem by whoever is rendering it.
#[derive(Debug, Clone)]
pub struct JobResult {
    pub output_path: PathBuf,
    pub size_bytes: u64,
    pub num_sections: usize,
    pub tokenizer_path: PathBuf,
    pub tokenizer_copied: bool,
    pub elapsed_ms: u64,
    pub peak_rss_bytes: Option<u64>,
    /// On-disk bytes of the `.safetensors` shards the conversion read.
    pub source_bytes: u64,
    /// How many shard files that was.
    pub source_files: usize,
    /// `source_bytes / size_bytes`, to two decimals.
    pub compression_ratio: f64,
    /// Parameters implied by the checkpoint's config.
    pub parameters: u64,
    /// `size_bytes * 8 / parameters`, to two decimals.
    pub bits_per_parameter: f64,
    /// Context stored in the header — the ceiling every later run is held to.
    pub max_context: u32,
    /// Where that context came from: `requested`, `model-config` or
    /// `sliding-window`.
    pub context_source: &'static str,
    /// KV cache the runtime allocates if the full stored context is used.
    pub kv_cache_bytes: u64,
}

impl JobResult {
    /// Everything here comes from the [`ConvertSummary`] the conversion
    /// returned; the two derived ratios are computed once, by the summary, so
    /// the CLI and the API cannot report different numbers for the same file.
    fn from_summary(summary: ConvertSummary) -> Self {
        let compression_ratio = round2_f64(summary.compression_ratio());
        let bits_per_parameter = round2_f64(summary.bits_per_parameter());
        Self {
            size_bytes: summary.bytes_written,
            num_sections: summary.num_sections,
            tokenizer_path: summary.tokenizer_path,
            tokenizer_copied: summary.tokenizer_copied,
            elapsed_ms: summary.elapsed.as_millis() as u64,
            peak_rss_bytes: peak_rss_bytes(),
            source_bytes: summary.source_bytes,
            source_files: summary.source_files,
            compression_ratio,
            parameters: summary.parameters,
            bits_per_parameter,
            max_context: summary.context.tokens,
            context_source: summary.context.source.as_str(),
            kv_cache_bytes: summary.kv_cache_bytes,
            output_path: summary.output_path,
        }
    }
}

/// One conversion, running or finished.
#[derive(Debug)]
pub struct Job {
    pub id: String,
    pub phase: JobPhase,
    pub stage: String,
    pub percent: f32,
    /// `(layer_index, num_layers)` while quantizing layers.
    pub layer: Option<(u32, u32)>,
    pub source: PathBuf,
    pub output: PathBuf,
    pub group_size: u32,
    pub embed_group_size: u32,
    pub max_context: u32,
    pub started: Instant,
    pub elapsed_ms: u64,
    /// Every narration line, in order. Poll with a cursor to read the tail.
    pub log: Vec<String>,
    /// Lines dropped once [`MAX_LOG_LINES`] was reached, so a client can tell
    /// its cursor is no longer exact.
    pub log_dropped: usize,
    pub result: Option<JobResult>,
    pub error: Option<String>,
}

impl Job {
    /// A JSON snapshot for `GET /api/convert/<id>`.
    ///
    /// `since` is a cursor into the log: pass the previous response's
    /// `log_next` to get only what has been added since.
    pub fn snapshot(&self, since: usize) -> serde_json::Value {
        let first = self.log_dropped;
        let start = since.saturating_sub(first).min(self.log.len());
        let lines: Vec<&String> = self.log[start..].iter().collect();

        serde_json::json!({
            "job_id": self.id,
            "state": self.phase.as_str(),
            "stage": self.stage,
            "percent": round2(self.percent),
            "layer": self.layer.map(|(index, total)| serde_json::json!({
                "index": index,
                "total": total,
            })),
            "elapsed_ms": self.elapsed_ms,
            "request": {
                "source": self.source.display().to_string(),
                "output": self.output.display().to_string(),
                "group_size": self.group_size,
                "embed_group_size": self.embed_group_size,
                // What was *asked for*. Null means nothing was: the context
                // follows the model's own, and `result.max_context` is the one
                // that was stored.
                "max_context": (self.max_context != FOLLOW_MODEL_CONTEXT)
                    .then_some(self.max_context),
            },
            "log": lines,
            "log_from": first + start,
            "log_next": first + self.log.len(),
            "log_dropped": self.log_dropped,
            "result": self.result.as_ref().map(|result| serde_json::json!({
                "output_path": result.output_path.display().to_string(),
                "size_bytes": result.size_bytes,
                "num_sections": result.num_sections,
                "tokenizer_path": result.tokenizer_path.display().to_string(),
                "tokenizer_copied": result.tokenizer_copied,
                "elapsed_ms": result.elapsed_ms,
                "peak_rss_bytes": result.peak_rss_bytes,
                "source_bytes": result.source_bytes,
                "source_files": result.source_files,
                "compression_ratio": result.compression_ratio,
                "parameters": result.parameters,
                "bits_per_parameter": result.bits_per_parameter,
                "max_context": result.max_context,
                "context_source": result.context_source,
                "kv_cache_bytes": result.kv_cache_bytes,
            })),
            "error": self.error,
        })
    }

    fn push_line(&mut self, line: &str) {
        // Conversion narration arrives with leading blank lines that make sense
        // in a terminal and not in a list.
        for part in line.split('\n') {
            let part = part.trim_end();
            if part.is_empty() {
                continue;
            }
            if self.log.len() >= MAX_LOG_LINES {
                self.log.remove(0);
                self.log_dropped += 1;
            }
            self.log.push(part.to_string());
        }
    }
}

fn round2(value: f32) -> f32 {
    (value * 100.0).round() / 100.0
}

fn round2_f64(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Every conversion this server has started.
#[derive(Debug, Default)]
pub struct Jobs {
    jobs: Mutex<Vec<Arc<Mutex<Job>>>>,
    index: Mutex<HashMap<String, Arc<Mutex<Job>>>>,
}

/// Why a conversion could not even be started.
#[derive(Debug)]
pub enum StartError {
    /// A conversion is already running.
    Busy,
    /// The request itself is wrong (missing source, bad options).
    Invalid(String),
}

impl Jobs {
    pub fn new() -> Self {
        Self::default()
    }

    /// True while any job is still running.
    pub fn running(&self) -> bool {
        let jobs = self.jobs.lock().unwrap_or_else(|error| error.into_inner());
        jobs.iter().any(|job| {
            job.lock()
                .map(|job| job.phase == JobPhase::Running)
                .unwrap_or(false)
        })
    }

    pub fn get(&self, id: &str) -> Option<Arc<Mutex<Job>>> {
        let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        index.get(id).cloned()
    }

    /// Ids newest first, with each job's phase — enough for a UI to show a
    /// history list without polling every id.
    pub fn list(&self) -> serde_json::Value {
        let jobs = self.jobs.lock().unwrap_or_else(|error| error.into_inner());
        let entries: Vec<serde_json::Value> = jobs
            .iter()
            .rev()
            .filter_map(|job| job.lock().ok())
            .map(|job| {
                serde_json::json!({
                    "job_id": job.id,
                    "state": job.phase.as_str(),
                    "stage": job.stage,
                    "percent": round2(job.percent),
                    "output": job.output.display().to_string(),
                })
            })
            .collect();
        serde_json::json!({ "jobs": entries })
    }

    /// Start a conversion on a background thread and return its id.
    ///
    /// Validation that can be done without touching the model — the source
    /// exists, the options are in range — happens here so an obviously bad
    /// request fails at `POST` time rather than a second later inside a job
    /// nobody is polling yet.
    pub fn start(&self, options: ConvertOptions) -> Result<String, StartError> {
        if !options.model_dir.is_dir() {
            return Err(StartError::Invalid(format!(
                "{} is not a directory containing a HuggingFace checkpoint",
                options.model_dir.display()
            )));
        }
        let Some(output) = options.output.clone() else {
            return Err(StartError::Invalid("output path is required".to_string()));
        };
        if self.running_count() >= MAX_RUNNING_JOBS {
            return Err(StartError::Busy);
        }

        let id = new_job_id();
        let job = Arc::new(Mutex::new(Job {
            id: id.clone(),
            phase: JobPhase::Running,
            stage: "queued".to_string(),
            percent: 0.0,
            layer: None,
            source: options.model_dir.clone(),
            output,
            group_size: options.group_size,
            embed_group_size: options.embed_group_size,
            max_context: options.max_context,
            started: Instant::now(),
            elapsed_ms: 0,
            log: Vec::new(),
            log_dropped: 0,
            result: None,
            error: None,
        }));

        {
            let mut jobs = self.jobs.lock().unwrap_or_else(|error| error.into_inner());
            let mut index = self.index.lock().unwrap_or_else(|error| error.into_inner());
            jobs.push(Arc::clone(&job));
            index.insert(id.clone(), Arc::clone(&job));
            // Drop the oldest *finished* jobs once the history is full; a
            // running job is never evicted, or its poll would 404 mid-run.
            while jobs.len() > MAX_RETAINED_JOBS {
                let evictable = jobs.iter().position(|job| {
                    job.lock()
                        .map(|job| job.phase != JobPhase::Running)
                        .unwrap_or(true)
                });
                match evictable {
                    Some(position) => {
                        let old = jobs.remove(position);
                        let old_id = old.lock().ok().map(|old| old.id.clone());
                        if let Some(old_id) = old_id {
                            index.remove(&old_id);
                        }
                    }
                    None => break,
                }
            }
        }

        let worker = Arc::clone(&job);
        std::thread::Builder::new()
            .name(format!("rai-convert-{id}"))
            .spawn(move || run_job(&worker, &options))
            .map_err(|error| {
                StartError::Invalid(format!("cannot start a worker thread: {error}"))
            })?;

        Ok(id)
    }

    fn running_count(&self) -> usize {
        let jobs = self.jobs.lock().unwrap_or_else(|error| error.into_inner());
        jobs.iter()
            .filter(|job| {
                job.lock()
                    .map(|job| job.phase == JobPhase::Running)
                    .unwrap_or(false)
            })
            .count()
    }
}

fn run_job(job: &Arc<Mutex<Job>>, options: &ConvertOptions) {
    let progress = |event: ConvertProgress<'_>| {
        if let Ok(mut job) = job.lock() {
            job.stage = event.stage.to_string();
            job.percent = event.percent;
            job.layer = event.layer;
            job.elapsed_ms = job.started.elapsed().as_millis() as u64;
            job.push_line(event.message);
        }
    };

    let outcome = convert_with_progress(options, &progress);

    if let Ok(mut job) = job.lock() {
        job.elapsed_ms = job.started.elapsed().as_millis() as u64;
        match outcome {
            Ok(summary) => {
                job.phase = JobPhase::Done;
                job.stage = "done".to_string();
                job.percent = 100.0;
                job.layer = None;
                job.result = Some(JobResult::from_summary(summary));
            }
            Err(error) => {
                job.phase = JobPhase::Error;
                job.stage = "error".to_string();
                // A conversion failure is the user's own local path and their
                // own model's shape — the thing they need in order to fix it —
                // so unlike a 500 from the chat path it is reported verbatim.
                let text = format!("{error:#}");
                job.push_line(&text);
                job.error = Some(text);
            }
        }
    }
}

fn new_job_id() -> String {
    // Not a secret — the Host/Origin checks are what keep other origins out —
    // but not a counter either, so one page cannot poll another's job by
    // guessing, and a restarted server never reuses an id.
    let mut rng = rand::thread_rng();
    let bytes: [u8; 12] = rng.gen();
    let mut id = String::with_capacity(24);
    for byte in bytes {
        id.push_str(&format!("{byte:02x}"));
    }
    id
}

/// Peak resident set of this process, if the platform makes it cheap to ask.
///
/// Process-wide and monotonic, so on a server that has already loaded a model
/// this is the peak of everything, not of the conversion alone. It is reported
/// because bounded conversion memory is a property this converter claims, and
/// a number the user can see is how that claim gets checked.
#[cfg(windows)]
fn peak_rss_bytes() -> Option<u64> {
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
    (ok != 0).then_some(counters.peak_working_set_size as u64)
}

#[cfg(not(windows))]
fn peak_rss_bytes() -> Option<u64> {
    // /proc/self/status reports it in kB on Linux; elsewhere, say nothing
    // rather than guess.
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::{ContextSource, ResolvedContext};

    fn job() -> Job {
        Job {
            id: "test".to_string(),
            phase: JobPhase::Running,
            stage: "queued".to_string(),
            percent: 0.0,
            layer: None,
            source: PathBuf::from("src"),
            output: PathBuf::from("out.raimodel"),
            group_size: 128,
            embed_group_size: 64,
            max_context: 2048,
            started: Instant::now(),
            elapsed_ms: 0,
            log: Vec::new(),
            log_dropped: 0,
            result: None,
            error: None,
        }
    }

    #[test]
    fn the_log_cursor_returns_only_new_lines() {
        let mut job = job();
        job.push_line("one");
        job.push_line("two");
        let first = job.snapshot(0);
        assert_eq!(first["log"].as_array().unwrap().len(), 2);
        assert_eq!(first["log_next"], 2);

        job.push_line("three");
        let second = job.snapshot(2);
        assert_eq!(second["log"].as_array().unwrap(), &vec!["three"]);
        assert_eq!(second["log_next"], 3);

        // A cursor past the end is not an error and returns nothing.
        assert!(job.snapshot(99)["log"].as_array().unwrap().is_empty());
    }

    #[test]
    fn multi_line_narration_becomes_separate_lines_without_blanks() {
        let mut job = job();
        job.push_line("\n=== EMBEDDING 8-BIT ===");
        assert_eq!(job.log, vec!["=== EMBEDDING 8-BIT ==="]);
    }

    #[test]
    fn a_missing_source_is_rejected_before_a_thread_is_spawned() {
        let jobs = Jobs::new();
        let error = jobs
            .start(ConvertOptions {
                model_dir: PathBuf::from("no-such-checkpoint-dir"),
                output: Some(PathBuf::from("x.raimodel")),
                ..ConvertOptions::default()
            })
            .unwrap_err();
        assert!(matches!(error, StartError::Invalid(_)));
        assert!(!jobs.running());
    }

    #[test]
    fn job_ids_are_unique_and_opaque() {
        let first = new_job_id();
        assert_eq!(first.len(), 24);
        assert_ne!(first, new_job_id());
    }

    #[test]
    fn peak_rss_is_reported_or_absent_but_never_zero() {
        if let Some(bytes) = peak_rss_bytes() {
            assert!(bytes > 0);
        }
    }

    /// A finished job has to be able to answer "what did this buy me?".
    ///
    /// The output size alone cannot: without the input size there is no
    /// compression figure, and without the parameter count no bits-per-
    /// parameter. Both derive from the conversion's own numbers rather than
    /// from a second measurement taken by whoever renders them.
    #[test]
    fn a_finished_job_reports_what_the_conversion_bought() {
        let mut job = job();
        job.phase = JobPhase::Done;
        job.result = Some(JobResult::from_summary(ConvertSummary {
            output_path: PathBuf::from("out.raimodel"),
            bytes_written: 619_538_088,
            num_sections: 26,
            tokenizer_path: PathBuf::from("tokenizer.json"),
            tokenizer_copied: true,
            elapsed: std::time::Duration::from_millis(47_000),
            source_bytes: 1_503_300_328,
            source_files: 1,
            parameters: 596_049_920,
            context: ResolvedContext {
                tokens: 40_960,
                source: ContextSource::ModelConfig,
            },
            kv_cache_bytes: 9_395_240_960,
        }));

        let result = &job.snapshot(0)["result"];
        assert_eq!(result["source_bytes"], 1_503_300_328u64);
        assert_eq!(result["source_files"], 1);
        assert_eq!(result["size_bytes"], 619_538_088u64);
        // 1_503_300_328 / 619_538_088 = 2.4265..., to two decimals.
        assert_eq!(result["compression_ratio"], 2.43);
        assert_eq!(result["parameters"], 596_049_920u64);
        // 619_538_088 * 8 / 596_049_920 = 8.315..., to two decimals.
        assert_eq!(result["bits_per_parameter"], 8.32);
        assert_eq!(result["output_path"], "out.raimodel");
        // The context that was stored, and what it will cost to run at.
        assert_eq!(result["max_context"], 40_960);
        assert_eq!(result["context_source"], "model-config");
        assert_eq!(result["kv_cache_bytes"], 9_395_240_960u64);
    }

    /// Nothing requested means the model's own context, which is not the same
    /// statement as "0 tokens were requested".
    #[test]
    fn an_unrequested_context_is_reported_as_absent_not_as_zero() {
        let mut job = job();
        job.max_context = FOLLOW_MODEL_CONTEXT;
        assert!(job.snapshot(0)["request"]["max_context"].is_null());

        job.max_context = 4_096;
        assert_eq!(job.snapshot(0)["request"]["max_context"], 4_096);
    }
}
