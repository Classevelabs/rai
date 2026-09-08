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

/// Take a lock, recovering the guard if a previous holder panicked.
///
/// Job state is a progress report, not an invariant that a panic can corrupt
/// into something dangerous: the worst a poisoned guard carries is a
/// half-updated stage string. Refusing to read it — which is what
/// `if let Ok(..)` did in `run_job` — is strictly worse, because the write
/// being skipped is the one that moves a job out of `Running`, and a job that
/// never leaves `Running` holds the single conversion slot for the life of the
/// process.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
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
        let jobs = lock(&self.jobs);
        jobs.iter().any(|job| lock(job).phase == JobPhase::Running)
    }

    pub fn get(&self, id: &str) -> Option<Arc<Mutex<Job>>> {
        lock(&self.index).get(id).cloned()
    }

    /// Ids newest first, with each job's phase — enough for a UI to show a
    /// history list without polling every id.
    pub fn list(&self) -> serde_json::Value {
        let jobs = lock(&self.jobs);
        let entries: Vec<serde_json::Value> = jobs
            .iter()
            .rev()
            .map(|job| {
                let job = lock(job);
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

        // Claiming the slot and registering the job happen under one lock.
        // Counting first and pushing afterwards is a check-then-act: two
        // callers could both see a free slot and both start a conversion, and
        // `MAX_RUNNING_JOBS` exists precisely because two at once make each
        // other slower and double peak memory. The request loop is
        // single-threaded today, so the race was not reachable — but `Jobs` is
        // `Send + Sync` and shared with every worker thread, and an invariant
        // that holds only because of a caller's threading model is not an
        // invariant this type can rely on.
        {
            let mut jobs = lock(&self.jobs);
            let mut index = lock(&self.index);

            if jobs
                .iter()
                .filter(|job| lock(job).phase == JobPhase::Running)
                .count()
                >= MAX_RUNNING_JOBS
            {
                return Err(StartError::Busy);
            }

            jobs.push(Arc::clone(&job));
            index.insert(id.clone(), Arc::clone(&job));
            // Drop the oldest *finished* jobs once the history is full; a
            // running job is never evicted, or its poll would 404 mid-run.
            while jobs.len() > MAX_RETAINED_JOBS {
                let evictable = jobs
                    .iter()
                    .position(|job| lock(job).phase != JobPhase::Running);
                match evictable {
                    Some(position) => {
                        let old = jobs.remove(position);
                        let old_id = lock(&old).id.clone();
                        index.remove(&old_id);
                    }
                    None => break,
                }
            }
        }

        let worker = Arc::clone(&job);
        let spawned = std::thread::Builder::new()
            .name(format!("rai-convert-{id}"))
            .spawn(move || run_job(&worker, &options));

        if let Err(error) = spawned {
            // The job is already registered and already counted as running. If
            // it were left that way the slot would never come back: nothing
            // else transitions a job whose worker does not exist, and a
            // running job is never evicted from the history either.
            let message = format!("cannot start a worker thread: {error}");
            {
                let mut job = lock(&job);
                job.phase = JobPhase::Error;
                job.stage = "error".to_string();
                job.push_line(&message);
                job.error = Some(message.clone());
            }
            return Err(StartError::Invalid(message));
        }

        Ok(id)
    }
}

/// Run one conversion to completion and record how it ended.
///
/// Two guarantees this function owes the rest of the server, because
/// `MAX_RUNNING_JOBS` is 1 and a running job is never evicted from the
/// history: the job it was handed **always** leaves `Running`, and it leaves
/// with a reason attached. A worker that returns without doing that costs the
/// process every future conversion, and the only cure is a restart.
fn run_job(job: &Arc<Mutex<Job>>, options: &ConvertOptions) {
    // The backstop, ahead of every path that could return. `Drop` runs during
    // unwinding as well as on a normal return, so a panic anywhere below —
    // including inside a dependency, where `catch_unwind` has already done its
    // job but any later code has not — still frees the slot.
    let _terminal = TerminalPhaseGuard(job);

    let progress = |event: ConvertProgress<'_>| {
        let mut job = lock(job);
        job.stage = event.stage.to_string();
        job.percent = event.percent;
        job.layer = event.layer;
        job.elapsed_ms = job.started.elapsed().as_millis() as u64;
        job.push_line(event.message);
    };

    // A panic in the conversion is reported as a failed conversion rather than
    // as a thread that vanished. `AssertUnwindSafe` is the honest annotation
    // here: the only state shared across the boundary is the job behind its
    // mutex, whose poisoning this module recovers from by policy (see `lock`),
    // and the caller-owned `options`, which the conversion only reads.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        convert_with_progress(options, &progress)
    }));

    let outcome = match outcome {
        Ok(result) => result,
        Err(payload) => Err(anyhow::anyhow!(
            "the conversion panicked: {}",
            panic_message(&payload)
        )),
    };

    // Scoped, and the scope is load-bearing rather than stylistic:
    // `TerminalPhaseGuard::drop` takes this same mutex, `std::sync::Mutex` is
    // not reentrant, and a guard still held when the drop runs would deadlock
    // the worker thread — wedging the conversion slot in exactly the way this
    // guard exists to prevent. Drop order happens to release this first, but
    // "happens to" is not a property to leave a future edit standing on.
    {
        let mut job = lock(job);
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

/// Forces a job out of `Running` if nothing else already did.
///
/// This exists for the paths no `match` arm covers: a panic that unwinds past
/// the normal terminal write, or a future edit that adds an early return. In
/// the ordinary case the job is already `Done` or `Error` by the time this
/// drops and it does nothing at all.
struct TerminalPhaseGuard<'a>(&'a Arc<Mutex<Job>>);

impl Drop for TerminalPhaseGuard<'_> {
    fn drop(&mut self) {
        let mut job = lock(self.0);
        if job.phase != JobPhase::Running {
            return;
        }
        job.phase = JobPhase::Error;
        job.stage = "error".to_string();
        job.elapsed_ms = job.started.elapsed().as_millis() as u64;
        let text = "the conversion worker ended without recording a result;                     the conversion did not finish";
        job.push_line(text);
        if job.error.is_none() {
            job.error = Some(text.to_string());
        }
    }
}

/// The human-readable half of a panic payload, when there is one.
///
/// `panic!("...")` and `assert!` produce a `&str` or a `String`; anything else
/// carries no text worth printing, and saying so is better than printing a
/// type name.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "no message".to_string()
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

    /// Register an already-running job the way `start` does, without spawning
    /// a worker for it. This is the state a dead worker used to leave behind.
    fn registered_running_job(jobs: &Jobs) -> Arc<Mutex<Job>> {
        let handle = Arc::new(Mutex::new(job()));
        lock(&jobs.jobs).push(Arc::clone(&handle));
        lock(&jobs.index).insert("test".to_string(), Arc::clone(&handle));
        handle
    }

    #[test]
    fn the_terminal_guard_forces_a_running_job_to_error() {
        let handle = Arc::new(Mutex::new(job()));
        drop(TerminalPhaseGuard(&handle));

        let finished = lock(&handle);
        assert_eq!(finished.phase, JobPhase::Error);
        assert_eq!(finished.stage, "error");
        assert!(finished
            .error
            .as_deref()
            .is_some_and(|e| e.contains("did not finish")));
    }

    #[test]
    fn the_terminal_guard_leaves_a_finished_job_alone() {
        let handle = Arc::new(Mutex::new(job()));
        {
            let mut started = lock(&handle);
            started.phase = JobPhase::Done;
            started.stage = "done".to_string();
        }
        drop(TerminalPhaseGuard(&handle));

        let finished = lock(&handle);
        assert_eq!(finished.phase, JobPhase::Done);
        assert_eq!(finished.stage, "done");
        assert!(finished.error.is_none());
    }

    /// The path the guard exists for: a real unwind, not a simulated one.
    #[test]
    fn a_panicking_worker_leaves_the_job_in_error_not_running() {
        let handle = Arc::new(Mutex::new(job()));
        let worker = Arc::clone(&handle);

        // The default hook would print this deliberate panic and make a
        // passing run look like a failing one.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::thread::spawn(move || {
            let _terminal = TerminalPhaseGuard(&worker);
            panic!("quantization exploded");
        })
        .join();
        std::panic::set_hook(previous);

        assert!(result.is_err(), "the worker was supposed to panic");
        assert_eq!(lock(&handle).phase, JobPhase::Error);
    }

    /// The consequence that made this worth fixing: `MAX_RUNNING_JOBS` is 1,
    /// and a job stuck in `Running` holds that slot for the life of the
    /// process, because nothing transitions it and eviction skips it.
    #[test]
    fn a_wedged_job_no_longer_holds_the_conversion_slot() {
        let jobs = Jobs::new();
        let handle = registered_running_job(&jobs);

        // While it is running, the slot is taken — this is correct behaviour.
        assert!(jobs.running());
        // Spread the defaults rather than spelling every field: these tests
        // care about the model directory and nothing else, and an exhaustive
        // literal makes every new option a compile error here for no reason.
        let refused = jobs.start(ConvertOptions {
            model_dir: std::env::temp_dir(),
            output: Some(PathBuf::from("out.raimodel")),
            max_context: 2048,
            quiet: true,
            ..ConvertOptions::default()
        });
        assert!(
            matches!(refused, Err(StartError::Busy)),
            "a running job must hold the single conversion slot"
        );

        // Once the worker is accounted for, the slot comes back. Before the
        // guard existed there was no code path that reached this state.
        drop(TerminalPhaseGuard(&handle));
        assert!(!jobs.running());
    }

    #[test]
    fn a_panic_payload_becomes_a_readable_line() {
        assert_eq!(panic_message(&"static message"), "static message");
        assert_eq!(panic_message(&"owned message".to_string()), "owned message");
        assert_eq!(panic_message(&42u8), "no message");
    }

    #[test]
    fn the_conversion_slot_is_claimed_under_the_same_lock_that_registers_it() {
        // Counting running jobs and pushing the new one happen together, so a
        // second caller cannot observe a free slot that a first caller has
        // already taken. Asserted through the public API: the first start
        // registers, the second is refused, and nothing in between can widen
        // that window.
        let jobs = Jobs::new();
        let _running = registered_running_job(&jobs);
        for _ in 0..8 {
            assert!(matches!(
                jobs.start(ConvertOptions {
                    model_dir: std::env::temp_dir(),
                    output: Some(PathBuf::from("out.raimodel")),
                    max_context: 2048,
                    quiet: true,
                    ..ConvertOptions::default()
                }),
                Err(StartError::Busy)
            ));
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
