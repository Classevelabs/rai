//! `rai perplexity` — what quantization actually cost this model.
//!
//! Every other number this engine reports is a speed. Speed is the easy half:
//! a 4-bit model that generates nonsense is very fast. Perplexity is the half
//! that says whether the weights still mean anything, and until this existed
//! `BENCHMARKS.md` had to say, in as many words, that no perplexity sweep had
//! been run and therefore no perplexity claim was made. A quantizer whose
//! quality is unmeasured is a quantizer nobody should trust, including us.
//!
//! # Method
//!
//! Deliberately the same method `llama-perplexity` uses, because llama.cpp is
//! what a reader will compare against and two numbers computed differently are
//! not a comparison:
//!
//! 1. Tokenize the whole text once, with no special tokens added — the corpus
//!    is continuous prose, not a chat turn, and a BOS inserted every chunk
//!    would score a position that does not occur in the text.
//! 2. Cut it into non-overlapping chunks of `--context` tokens.
//! 3. Run one batched forward pass per chunk, from an empty KV cache.
//! 4. Score only the **second half** of each chunk. Every scored token then
//!    has at least `context / 2` tokens of real context behind it, which is
//!    what stops the first tokens of each chunk — predicted from almost
//!    nothing — from dominating the average.
//! 5. Perplexity is `exp(mean negative log-likelihood)` over every scored
//!    token in the corpus.
//!
//! The mean is accumulated in `f64`. At 512 tokens per chunk over a corpus of
//! any size the sum of per-token NLLs runs to five or six figures, and an `f32`
//! accumulator loses the low bits of every addition long before the end —
//! which shows up as a perplexity that depends on how much text you fed it.
//!
//! # What the number is comparable to
//!
//! Only to another perplexity over the *same text with the same tokenizer*.
//! Perplexity is a function of the tokenizer's vocabulary as much as of the
//! model, so a Qwen number and a Llama number are not comparable to each other
//! even on identical text. What they are good for is the comparison that
//! matters here: the same model, same text, same tokenizer, fp16 against the
//! shipped 4-bit engine. That gap is dominated by 4-bit weight quantization,
//! but it also carries the int8 activation quantization and the approximate
//! fast-exp softmax/SiLU kernels the AVX2 path uses — neither present in the
//! fp16 reference — so read it as the end-to-end cost of this engine, not
//! weight quantization alone.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use clap::Args;

use crate::cli::{load_tokenizer_for_model, resolve_tokenizer};
use crate::model::{BatchScratch, RaiModel};

/// Chunk size, in tokens, when none is given.
///
/// 512 is `llama-perplexity`'s default and the size nearly every published
/// llama.cpp perplexity number was taken at.
const DEFAULT_CONTEXT: usize = 512;

#[derive(Args, Debug)]
pub struct PerplexityArgs {
    /// The .raimodel file to score
    pub model: PathBuf,

    /// UTF-8 text file to score against (e.g. wikitext-2 raw test)
    #[arg(long, value_name = "FILE")]
    pub text: PathBuf,

    /// Tokens per chunk; the second half of each chunk is scored
    #[arg(long, default_value_t = DEFAULT_CONTEXT, value_name = "N")]
    pub context: usize,

    /// Stop after this many chunks (0 = the whole file)
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub max_chunks: usize,

    /// tokenizer.json; defaults to the one beside the model
    #[arg(long, value_name = "PATH")]
    pub tokenizer: Option<PathBuf>,

    /// Also record the top-K next-token distribution at each scored position
    #[arg(long, default_value_t = 0, value_name = "K", requires = "dump_path")]
    pub dump_topk: usize,

    /// Where to write the --dump-topk record
    #[arg(long, value_name = "FILE")]
    pub dump_path: Option<PathBuf>,

    /// Emit one JSON object instead of a progress table
    #[arg(long)]
    pub json: bool,
}

/// Container magic for the top-K record. Eight bytes so the header stays
/// aligned and a truncated file is obvious rather than merely wrong.
const TOPK_MAGIC: &[u8; 8] = b"RAITOPK\0";
const TOPK_VERSION: u32 = 1;

/// The top-K next-token distribution at every scored position.
///
/// Perplexity is one number over a whole corpus, and two models can reach the
/// same one while disagreeing about every individual token. This record is what
/// makes the sharper questions answerable: how often does the 4-bit model pick
/// the token the fp16 model would have, and how far apart are the two
/// distributions where it matters. `scripts/compare_topk.py` reads two of these
/// and answers both.
///
/// Layout, little-endian throughout:
///
/// ```text
///   magic      8   "RAITOPK\0"
///   version    4   u32 = 1
///   k          4   u32
///   positions  8   u64   (filled in on close)
///   vocab      4   u32
///   reserved   4   u32 = 0
///   then, per position:
///     target      4   u32   the token that actually follows
///     target_lp   4   f32   its log-probability under this model
///     k * (id 4 u32, logprob 4 f32), descending by logprob
/// ```
struct TopKWriter {
    file: std::io::BufWriter<std::fs::File>,
    k: usize,
    positions: u64,
    /// Reused between positions so a 150k-entry vocabulary is not re-allocated
    /// ten thousand times.
    order: Vec<u32>,
}

impl TopKWriter {
    fn create(path: &std::path::Path, k: usize, vocab_size: usize) -> Result<Self> {
        ensure!(k >= 1, "--dump-topk must be at least 1");
        ensure!(
            k <= vocab_size,
            "--dump-topk is {k} but the vocabulary is {vocab_size}"
        );
        let file =
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut file = std::io::BufWriter::new(file);
        file.write_all(TOPK_MAGIC)?;
        file.write_all(&TOPK_VERSION.to_le_bytes())?;
        file.write_all(&(k as u32).to_le_bytes())?;
        file.write_all(&0u64.to_le_bytes())?; // positions, rewritten on close
        file.write_all(&(vocab_size as u32).to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        Ok(Self {
            file,
            k,
            positions: 0,
            order: Vec::new(),
        })
    }

    /// Record one scored position. `log_z` is the log-partition function
    /// already computed for the perplexity term, so this costs a selection and
    /// no second pass over the logits.
    fn push(&mut self, logits: &[f32], target: usize, log_z: f64) -> Result<()> {
        // `total_cmp` rather than `partial_cmp().unwrap()`: a NaN logit would
        // panic the second form, and a damaged model file is exactly where one
        // would come from.
        let by_logit_desc = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);

        self.order.clear();
        self.order.extend(0..logits.len() as u32);
        // Partial selection puts the K largest in `[..k]` in O(n); only those K
        // are then sorted, at O(k log k). A full sort of a 150k vocabulary at
        // every one of ten thousand positions would cost more than the forward
        // pass being measured.
        self.order.select_nth_unstable_by(self.k - 1, by_logit_desc);
        let top = &mut self.order[..self.k];
        top.sort_unstable_by(by_logit_desc);

        self.file.write_all(&(target as u32).to_le_bytes())?;
        let target_lp = (f64::from(logits[target]) - log_z) as f32;
        self.file.write_all(&target_lp.to_le_bytes())?;
        for &id in top.iter() {
            self.file.write_all(&id.to_le_bytes())?;
            let lp = (f64::from(logits[id as usize]) - log_z) as f32;
            self.file.write_all(&lp.to_le_bytes())?;
        }
        self.positions += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<u64> {
        use std::io::Seek;
        self.file.flush()?;
        let mut file = self.file.into_inner()?;
        // The count is only known at the end; the header reserved room for it.
        file.seek(std::io::SeekFrom::Start(16))?;
        file.write_all(&self.positions.to_le_bytes())?;
        file.flush()?;
        Ok(self.positions)
    }
}

/// Everything one run measured, in the order a reader needs it.
#[derive(Debug, Clone)]
pub struct PerplexityReport {
    pub perplexity: f64,
    /// Mean negative log-likelihood per scored token, in nats.
    pub nll: f64,
    pub tokens_scored: usize,
    pub tokens_total: usize,
    pub chunks: usize,
    pub context: usize,
    pub elapsed_ms: u64,
    /// Tokens pushed through the model per second, scored or not — the honest
    /// throughput figure for this workload, since every token in a chunk is
    /// evaluated whether or not it is scored.
    pub tokens_per_second: f64,
    /// Standard error of the mean NLL, in nats.
    ///
    /// A perplexity is a sample mean over a finite number of tokens, and a
    /// number quoted to four decimals from ten thousand of them is mostly
    /// decoration. Reporting the error turns "12.5421" into "12.54, and a
    /// re-run on different text of the same size would land within about this
    /// much" — which is what a reader comparing two engines actually needs.
    pub nll_stderr: f64,
    /// The same error expressed on the perplexity scale, as a percentage.
    pub perplexity_stderr_pct: f64,
}

impl PerplexityReport {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "perplexity": round4(self.perplexity),
            "nll": round4(self.nll),
            "tokens_scored": self.tokens_scored,
            "tokens_total": self.tokens_total,
            "chunks": self.chunks,
            "context": self.context,
            "elapsed_ms": self.elapsed_ms,
            "tokens_per_second": round4(self.tokens_per_second),
            "nll_stderr": round4(self.nll_stderr),
            "perplexity_stderr_pct": round4(self.perplexity_stderr_pct),
            "method": "llama.cpp-compatible: non-overlapping chunks, second half scored",
        })
    }
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

/// `log(sum(exp(logits)))`, computed without overflowing.
///
/// Subtracting the maximum before exponentiating is not optional here: a 4-bit
/// model's logits reach the high tens, `exp(80)` is already 5.5e34, and a
/// vocabulary of 150k of them overflows `f32` outright. The result is `f64`
/// because it is immediately differenced against a single logit, and that
/// difference is the quantity being averaged over the whole corpus.
fn log_sum_exp(logits: &[f32]) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f64::INFINITY;
    }
    let sum: f64 = logits
        .iter()
        .map(|&value| f64::from(value - max).exp())
        .sum();
    f64::from(max) + sum.ln()
}

/// Score `tokens` and return the corpus perplexity.
///
/// Separated from argument parsing so it is callable from a test with a
/// fixture model and from the benchmark harness without a process boundary.
pub fn score_tokens(
    model: &RaiModel,
    tokens: &[usize],
    context: usize,
    max_chunks: usize,
    dump: Option<(&std::path::Path, usize)>,
    mut on_chunk: impl FnMut(usize, usize, f64),
) -> Result<PerplexityReport> {
    // Below 4 a chunk scores nothing at all: `first_scored` is `context / 2`
    // and the loop stops at `context - 1`, so a 2-token chunk has an empty
    // range, `scored` stays zero, and every number derived from it is 0/0.
    ensure!(
        context >= 4,
        "--context must be at least 4 tokens; a smaller chunk scores no tokens, \
         because the first half is context for the second and the last position \
         has no token after it to predict"
    );
    ensure!(
        context <= model.config.max_context as usize,
        "--context is {context} but this model stores a {}-token window; \
         convert it with a larger --max-context or score at a smaller one",
        model.config.max_context
    );

    let total_chunks = tokens.len() / context;
    ensure!(
        total_chunks > 0,
        "the text is {} tokens, which is fewer than one {context}-token chunk; \
         pass a longer file or a smaller --context",
        tokens.len()
    );
    let chunks = if max_chunks == 0 {
        total_chunks
    } else {
        total_chunks.min(max_chunks)
    };

    let hidden_size = model.config.hidden_size as usize;
    let vocab_size = model.config.vocab_size as usize;

    // One cache and one set of buffers for the whole run. The cache is reset
    // per chunk rather than reallocated: chunks are independent by
    // construction, and re-allocating a KV cache per chunk would dominate the
    // measured time on a short context.
    let mut kv_cache = model.create_kv_cache(context)?;
    let mut batch_scratch = BatchScratch::new();
    let mut hiddens = vec![0.0f32; context * hidden_size];
    let mut normed = vec![0.0f32; hidden_size];
    let mut logits = vec![0.0f32; vocab_size];
    let positions: Vec<usize> = (0..context).collect();

    // Scoring starts at the halfway point, so every scored token has at least
    // `context / 2` tokens behind it.
    let first_scored = context / 2;

    let mut writer = match dump {
        Some((path, k)) => Some(TopKWriter::create(path, k, vocab_size)?),
        None => None,
    };

    let mut nll_sum = 0.0f64;
    // Sum of squares, for the standard error. Kept alongside the sum rather
    // than derived afterwards because the per-token values are not retained.
    let mut nll_sq_sum = 0.0f64;
    let mut scored = 0usize;
    let start = std::time::Instant::now();

    for chunk in 0..chunks {
        let window = &tokens[chunk * context..(chunk + 1) * context];

        kv_cache.clear();
        for (index, &token) in window.iter().enumerate() {
            model.embed_token(
                token,
                &mut hiddens[index * hidden_size..(index + 1) * hidden_size],
            )?;
        }
        model.forward_batch(&mut hiddens, &positions, &mut kv_cache, &mut batch_scratch)?;

        // Position `i` predicts token `i + 1`, so the last position in a chunk
        // has no target inside it and is not scored.
        let mut chunk_nll = 0.0f64;
        let mut chunk_scored = 0usize;
        for index in first_scored..context - 1 {
            model.hidden_to_logits_into(
                &hiddens[index * hidden_size..(index + 1) * hidden_size],
                &mut normed,
                &mut logits,
            )?;
            let target = window[index + 1];
            ensure!(
                target < vocab_size,
                "token id {target} is outside this model's {vocab_size}-token vocabulary; \
                 the text was tokenized with a different tokenizer than the model was built for"
            );
            let log_z = log_sum_exp(&logits);
            let nll = log_z - f64::from(logits[target]);
            ensure!(
                nll.is_finite(),
                "chunk {chunk} produced a non-finite log-likelihood; the model file is damaged"
            );
            if let Some(writer) = writer.as_mut() {
                writer.push(&logits, target, log_z)?;
            }
            chunk_nll += nll;
            nll_sq_sum += nll * nll;
            chunk_scored += 1;
        }

        nll_sum += chunk_nll;
        scored += chunk_scored;
        on_chunk(chunk + 1, chunks, (nll_sum / scored as f64).exp());
    }

    if let Some(writer) = writer {
        let written = writer.finish()?;
        ensure!(
            written as usize == scored,
            "the top-K record holds {written} positions but {scored} were scored"
        );
    }

    let elapsed = start.elapsed();
    let mean_nll = nll_sum / scored as f64;
    // Population variance of the per-token NLL, clamped at zero: the
    // subtraction is catastrophic cancellation when the variance is tiny, and
    // a negative variance under a square root is a NaN standard error
    // reported as if it meant something.
    let variance = (nll_sq_sum / scored as f64 - mean_nll * mean_nll).max(0.0);
    let nll_stderr = (variance / scored as f64).sqrt();
    let evaluated = chunks * context;

    Ok(PerplexityReport {
        perplexity: mean_nll.exp(),
        nll: mean_nll,
        tokens_scored: scored,
        tokens_total: evaluated,
        chunks,
        context,
        elapsed_ms: elapsed.as_millis() as u64,
        tokens_per_second: evaluated as f64 / elapsed.as_secs_f64(),
        nll_stderr,
        perplexity_stderr_pct: (nll_stderr.exp() - 1.0) * 100.0,
    })
}

pub fn run(args: &PerplexityArgs) -> Result<()> {
    crate::gemm::configure_thread_pool();

    let model =
        RaiModel::load(&args.model).with_context(|| format!("loading {}", args.model.display()))?;
    let tokenizer_path = resolve_tokenizer(&args.model, args.tokenizer.as_deref())?;
    let tokenizer = load_tokenizer_for_model(&tokenizer_path, model.config.vocab_size as usize)?;

    let text = std::fs::read_to_string(&args.text)
        .with_context(|| format!("reading {}", args.text.display()))?;

    // `false`: no BOS/EOS. The corpus is continuous prose cut into chunks, and
    // a special token inserted at every chunk boundary would be scored as if
    // the text contained it.
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|error| anyhow::anyhow!("{error}"))
        .context("tokenizing the text")?;
    let tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();

    if !args.json {
        eprintln!(
            "model      {}\ntext       {} ({} tokens)\ncontext    {} ({} scored per chunk)\n",
            args.model.display(),
            args.text.display(),
            tokens.len(),
            args.context,
            args.context / 2 - 1,
        );
    }

    let quiet = args.json;
    let dump = match (args.dump_topk, args.dump_path.as_deref()) {
        (0, _) => None,
        (k, Some(path)) => Some((path, k)),
        // clap's `requires` makes this unreachable from the CLI; refusing
        // rather than silently dropping the request keeps it that way.
        (_, None) => anyhow::bail!("--dump-topk requires --dump-path"),
    };
    let report = score_tokens(
        &model,
        &tokens,
        args.context,
        args.max_chunks,
        dump,
        |done, total, running| {
            if !quiet {
                eprint!("\r  chunk {done}/{total}   ppl {running:.4}   ");
                let _ = std::io::stderr().flush();
            }
        },
    )?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report.to_json())?);
    } else {
        eprintln!("\n");
        println!(
            "perplexity   {:.2}  (+/- {:.2}% standard error)",
            report.perplexity, report.perplexity_stderr_pct
        );
        println!(
            "nll          {:.4} +/- {:.4} nats/token",
            report.nll, report.nll_stderr
        );
        println!(
            "scored       {} tokens over {} chunks of {}",
            report.tokens_scored, report.chunks, report.context
        );
        println!(
            "throughput   {:.1} tok/s ({} tokens evaluated in {:.1}s)",
            report.tokens_per_second,
            report.tokens_total,
            report.elapsed_ms as f64 / 1000.0
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_exp_survives_logits_that_overflow_f32() {
        // exp(90) is ~1.2e39 — already past f32::MAX. A naive implementation
        // returns inf here and every perplexity built on it is inf.
        let logits = vec![90.0f32; 4];
        let value = log_sum_exp(&logits);
        assert!(value.is_finite(), "log_sum_exp overflowed: {value}");
        // log(4 * e^90) = 90 + ln 4
        assert!((value - (90.0 + 4.0f64.ln())).abs() < 1e-9, "{value}");
    }

    #[test]
    fn a_uniform_distribution_has_perplexity_equal_to_its_vocabulary() {
        // The definition, used as a self-check: if every one of n logits is
        // equal, the model is guessing uniformly and its perplexity is n.
        for n in [2usize, 10, 4096] {
            let logits = vec![0.0f32; n];
            let nll = log_sum_exp(&logits) - 0.0;
            assert!(
                (nll.exp() - n as f64).abs() < 1e-6,
                "uniform over {n} gave perplexity {}",
                nll.exp()
            );
        }
    }

    #[test]
    fn a_confident_correct_prediction_approaches_perplexity_one() {
        let mut logits = vec![0.0f32; 1000];
        logits[7] = 40.0;
        let nll = log_sum_exp(&logits) - f64::from(logits[7]);
        assert!(nll.exp() < 1.000_001, "{}", nll.exp());
    }

    /// The scored range is `[context / 2, context - 1)`, and it has to be
    /// non-empty for every accepted `--context` — otherwise the mean is 0/0
    /// and the whole report is NaN with a zero exit status.
    ///
    /// The guard is at 4 rather than at 3, which is where the arithmetic
    /// actually stops working. That gap is deliberate and this test pins both
    /// halves of it, so the reason in the error message stays true.
    #[test]
    fn every_accepted_context_scores_a_meaningful_number_of_tokens() {
        let scored_at = |context: usize| (context - 1).saturating_sub(context / 2);

        for context in 4..=4096usize {
            assert!(
                scored_at(context) >= 1,
                "context {context} would score {} tokens",
                scored_at(context)
            );
        }

        // Rejected because the range is empty and the mean would be 0/0.
        assert_eq!(scored_at(1), 0);
        assert_eq!(scored_at(2), 0);
        // Rejected for the other reason the message gives: one token, from one
        // token of context.
        assert_eq!(scored_at(3), 1);
        assert_eq!(
            3 / 2,
            1,
            "the single scored token has a single token behind it"
        );
    }

    /// The top-K record is read by `scripts/compare_topk.py`, which parses it
    /// with a hand-written `struct` format. Nothing but this test pins the two
    /// together, so it asserts the exact bytes rather than round-tripping
    /// through the writer's own reader — there is no reader on this side.
    #[test]
    fn the_topk_record_has_the_layout_the_python_reader_expects() {
        let path = std::env::temp_dir().join(format!("rai-topk-{}.bin", std::process::id()));
        let mut writer = TopKWriter::create(&path, 2, 8).expect("create");
        //                   id: 0    1    2    3    4    5    6    7
        let logits = [0.0f32, 5.0, 1.0, 9.0, 2.0, 0.0, 0.0, 0.0];
        let log_z = log_sum_exp(&logits);
        writer.push(&logits, 3, log_z).expect("push");
        assert_eq!(writer.finish().expect("finish"), 1);

        let raw = std::fs::read(&path).expect("read back");
        std::fs::remove_file(&path).ok();

        // header: magic(8) version(4) k(4) positions(8) vocab(4) reserved(4)
        //
        // Asserted against the literal bytes, not against `TOPK_MAGIC`. The
        // first version of this test compared the constant with itself, which
        // is true for any value the constant happens to hold — and it held a
        // different one than `scripts/compare_topk.py` expected, so the record
        // was unreadable by the only thing that reads it while this test was
        // green. The contract is the bytes.
        assert_eq!(&raw[0..8], b"RAITOPK\0");
        assert_eq!(TOPK_MAGIC, b"RAITOPK\0");
        assert_eq!(
            u32::from_le_bytes(raw[8..12].try_into().unwrap()),
            TOPK_VERSION
        );
        assert_eq!(u32::from_le_bytes(raw[12..16].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(raw[16..24].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(raw[24..28].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(raw[28..32].try_into().unwrap()), 0);

        // one position: target(4) target_lp(4) then k * (id(4) lp(4))
        assert_eq!(raw.len(), 32 + 4 + 4 + 2 * 8);
        assert_eq!(u32::from_le_bytes(raw[32..36].try_into().unwrap()), 3);

        let read_f32 = |at: usize| f32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
        let target_lp = read_f32(36);
        assert!(
            (f64::from(target_lp) - (9.0 - log_z)).abs() < 1e-5,
            "{target_lp}"
        );

        // Descending by logit: id 3 (9.0) then id 1 (5.0).
        assert_eq!(u32::from_le_bytes(raw[40..44].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(raw[48..52].try_into().unwrap()), 1);
        assert!(
            read_f32(44) > read_f32(52),
            "top-K must be ordered by descending logprob"
        );

        // Log-probabilities, not logits: both are negative, and the first
        // matches the target's, because the target is the argmax here.
        assert!(read_f32(44) < 0.0 && read_f32(52) < 0.0);
        assert!((read_f32(44) - target_lp).abs() < 1e-6);
    }

    #[test]
    fn a_topk_wider_than_the_vocabulary_is_refused() {
        let path = std::env::temp_dir().join(format!("rai-topk-bad-{}.bin", std::process::id()));
        assert!(TopKWriter::create(&path, 9, 8).is_err());
        assert!(TopKWriter::create(&path, 0, 8).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn non_finite_logits_do_not_silently_become_a_number() {
        assert!(log_sum_exp(&[f32::INFINITY, 1.0]).is_infinite());
        assert!(log_sum_exp(&[f32::NAN]).is_infinite());
    }
}
