//! Throughput benchmark for the W4A8 kernels.
//!
//! # Why this exists
//!
//! Until this release the only way to ask "did that change cost throughput?"
//! was to time `rai run` on a real model and compare. That measurement cannot
//! answer the question. It includes model loading, tokenization, sampling and
//! the page cache; on a shared machine its samples ramp for the first several
//! seconds; and a seven-sample comparison of it has a 95% confidence interval
//! roughly 40 percentage points wide. A kernel change worth 5% is invisible to
//! it, and a run of bad luck looks like a 9% regression. One did.
//!
//! So the kernels are timed directly, in process, on fixed data, with the
//! machine's warm-up discarded by measurement rather than by guesswork, and
//! every figure is reported with the spread that justifies believing it.
//!
//! # Reading the output
//!
//! `median` is the headline. `iqr%` is the interquartile range as a percentage
//! of the median: it says how much the machine moved underneath the
//! measurement. **An iqr% above about 5 means the machine is too noisy for
//! that row to settle a few-percent question** — run it again on a quieter
//! machine rather than believing a small difference.
//!
//! # Comparing two builds
//!
//! ```text
//! cargo bench --bench gemm -- --json > before.json
//! # ... make the change ...
//! cargo bench --bench gemm -- --json > after.json
//! ```
//!
//! Both runs must happen on the same machine, and ideally alternating, because
//! a cloud container's throughput drifts by more than most kernel changes are
//! worth.

use std::time::{Duration, Instant};

use rai_infer::calibrate::accumulate_gram;
use rai_infer::gemm::{w4a8_matmul, w4a8_matvec};

/// Shapes taken from the models this engine is used on, not round numbers.
///
/// A kernel's behaviour changes with the row length: `chunk_rows_for` sizes its
/// L1 working set from it, and the prefetch distance switches at 2048 bytes per
/// row. Benchmarking only square matrices would miss both.
const SHAPES: &[(&str, usize, usize)] = &[
    // (label, rows, cols)
    ("qkv 7B      ", 4096, 4096),
    ("mlp-up 7B   ", 14336, 4096),
    ("mlp-down 7B ", 4096, 14336),
    ("qkv 1.7B    ", 2048, 2048),
    ("mlp-down 1.7B", 2048, 8192),
];

/// Every group size the converter offers, because the point of the benchmark is
/// that the cost of a group size is the group size — not the format's maximum.
const GROUP_SIZES: &[usize] = &[128, 64, 32];

/// Tokens per batched call. 1 is decode; 8 is a small prefill batch, which
/// takes the weight-stationary path instead.
const BATCHES: &[usize] = &[1, 8];

struct Sample {
    median_ns: f64,
    iqr_pct: f64,
    iterations: usize,
}

fn build_weights(rows: usize, cols: usize, group_size: usize) -> (Vec<u8>, Vec<u8>) {
    let num_groups = cols.div_ceil(group_size);
    let mut group_params = vec![0u8; rows * num_groups * 4];
    for (i, chunk) in group_params.chunks_exact_mut(4).enumerate() {
        // half::f16 bit patterns for a small positive scale and a small
        // negative zero point, varied per group so no group's parameters can
        // be reused for another by accident.
        let scale = half::f16::from_f32(0.02 + 0.001 * (i % 31) as f32);
        let zero = half::f16::from_f32(-0.3 + 0.01 * (i % 17) as f32);
        chunk[0..2].copy_from_slice(&scale.to_le_bytes());
        chunk[2..4].copy_from_slice(&zero.to_le_bytes());
    }
    let mut nibble_data = vec![0u8; rows * cols / 2];
    for (i, byte) in nibble_data.iter_mut().enumerate() {
        *byte = (((i * 7 + 3) % 16) | (((i * 11 + 5) % 16) << 4)) as u8;
    }
    (nibble_data, group_params)
}

fn build_input(num_tokens: usize, cols: usize) -> Vec<f32> {
    (0..num_tokens * cols)
        .map(|i| ((i % 97) as f32 / 97.0 - 0.5) * (1.0 + (i % 13) as f32 / 13.0))
        .collect()
}

/// Time `body` until the machine has settled, then until the sample is stable.
///
/// Warm-up runs for a fixed wall-clock budget rather than a fixed iteration
/// count: the first call touches 30 MB of cold weight data, and on a container
/// that has just started, several seconds pass before throughput plateaus. A
/// count-based warm-up tuned on a fast machine is no warm-up at all on a slow
/// one.
fn measure(mut body: impl FnMut(), warmup: Duration, min_samples: usize) -> Sample {
    let start = Instant::now();
    while start.elapsed() < warmup {
        body();
    }

    // One untimed pass sets the iteration count so that a single timed sample
    // spans a target duration rather than a target call count.
    //
    // The target is 200 ms, not the 1 ms a first version used. These kernels
    // take 3-15 ms per call, so a 1 ms target rounds to one call per sample —
    // and a sample of one call on a machine doing anything else measures the
    // machine, not the kernel. The first run of this benchmark reported
    // interquartile ranges between 9% and 224% for exactly that reason.
    let probe = Instant::now();
    body();
    let single = probe.elapsed().as_nanos().max(1) as f64;
    let iterations = ((200_000_000.0 / single).ceil() as usize).max(1);

    let mut samples = Vec::with_capacity(min_samples);
    for _ in 0..min_samples {
        let began = Instant::now();
        for _ in 0..iterations {
            body();
        }
        samples.push(began.elapsed().as_nanos() as f64 / iterations as f64);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let p25 = samples[samples.len() / 4];
    let p75 = samples[samples.len() * 3 / 4];
    Sample {
        median_ns: median,
        iqr_pct: (p75 - p25) / median * 100.0,
        iterations,
    }
}

fn main() {
    let json = std::env::args().any(|a| a == "--json");
    // `cargo bench` passes --bench; a short run is useful when iterating.
    let quick = std::env::args().any(|a| a == "--quick");
    let warmup = if quick {
        Duration::from_millis(200)
    } else {
        Duration::from_millis(1500)
    };
    let samples = if quick { 7 } else { 25 };

    let mut rows_out: Vec<String> = Vec::new();
    if !json {
        println!(
            "\n{:<14} {:>6} {:>7} {:>4} {:>12} {:>8} {:>7}",
            "shape", "group", "tokens", "iter", "median ns", "GB/s", "iqr%"
        );
        println!("{}", "-".repeat(66));
    }

    for &(label, rows, cols) in SHAPES {
        for &group_size in GROUP_SIZES {
            if cols.div_ceil(group_size) > rai_infer::gemm::MAX_GROUPS {
                continue;
            }
            let (nibble_data, group_params) = build_weights(rows, cols, group_size);
            for &tokens in BATCHES {
                let input = build_input(tokens, cols);
                let mut output = vec![0.0f32; tokens * rows];

                let sample = if tokens == 1 {
                    measure(
                        || {
                            w4a8_matvec(
                                &mut output,
                                &nibble_data,
                                &group_params,
                                &input,
                                rows,
                                cols,
                                group_size,
                            )
                        },
                        warmup,
                        samples,
                    )
                } else {
                    measure(
                        || {
                            w4a8_matmul(
                                &mut output,
                                &nibble_data,
                                &group_params,
                                &input,
                                rows,
                                cols,
                                tokens,
                                group_size,
                            )
                        },
                        warmup,
                        samples,
                    )
                };

                // Weight bytes moved per call: the nibble plane plus the group
                // parameters. This is what the kernel is bound by, so it is a
                // fairer figure than FLOPs for comparing shapes.
                let bytes = (rows * cols / 2 + rows * cols.div_ceil(group_size) * 4) as f64;
                let gb_s = bytes / sample.median_ns;

                if json {
                    rows_out.push(format!(
                        "{{\"shape\":\"{}\",\"rows\":{},\"cols\":{},\"group_size\":{},\
                         \"tokens\":{},\"iterations\":{},\"median_ns\":{:.1},\
                         \"gb_per_s\":{:.2},\"iqr_pct\":{:.2}}}",
                        label.trim(),
                        rows,
                        cols,
                        group_size,
                        tokens,
                        sample.iterations,
                        sample.median_ns,
                        gb_s,
                        sample.iqr_pct
                    ));
                } else {
                    println!(
                        "{:<14} {:>6} {:>7} {:>4} {:>12.0} {:>8.2} {:>7.2}",
                        label,
                        group_size,
                        tokens,
                        sample.iterations,
                        sample.median_ns,
                        gb_s,
                        sample.iqr_pct
                    );
                }
            }
        }
    }

    // ---- calibration ------------------------------------------------------
    //
    // One `H = XᵀX` accumulation per projection group per layer. A 7B has 32
    // layers and four such groups, three of them `hidden` wide and one
    // `intermediate` wide, so the projected total below is what a calibrated
    // conversion of a 7B actually costs on this machine.
    if !json {
        println!(
            "\n{:<14} {:>6} {:>7} {:>12} {:>9} {:>7}",
            "gram (H=XtX)", "cols", "rows", "median ms", "GFLOP/s", "iqr%"
        );
        println!("{}", "-".repeat(62));
    }

    let mut seconds_per_token: Vec<(usize, f64)> = Vec::new();
    // A 7B MLP Hessian is 14336x14336 f64 — 1.6 GB — which is not something to
    // allocate on a developer machine because a benchmark ran. The widths a
    // 1.7B uses are measured by default; `--wide` opts into the 7B one.
    let wide = std::env::args().any(|a| a == "--wide");
    let shapes: &[(usize, usize)] = if wide {
        &[(2048, 256), (4096, 256), (8192, 128), (14336, 64)]
    } else {
        &[(2048, 256), (4096, 256), (8192, 128)]
    };
    for &(cols, rows) in shapes {
        let x = build_input(rows, cols);
        let mut h = vec![0.0f64; cols * cols];
        let sample = measure(
            || {
                accumulate_gram(&mut h, &x, rows, cols).unwrap();
            },
            warmup,
            samples.min(9),
        );
        // Upper triangle only: cols*(cols+1)/2 FMAs per row, two flops each.
        let flops = rows as f64 * (cols * (cols + 1) / 2) as f64 * 2.0;
        let gflops = flops / sample.median_ns;
        seconds_per_token.push((cols, sample.median_ns / 1e9 / rows as f64));

        if json {
            rows_out.push(format!(
                "{{\"kernel\":\"gram\",\"cols\":{},\"rows\":{},\"median_ns\":{:.0},\
                 \"gflop_s\":{:.2},\"iqr_pct\":{:.2}}}",
                cols, rows, sample.median_ns, gflops, sample.iqr_pct
            ));
        } else {
            println!(
                "{:<14} {:>6} {:>7} {:>12.1} {:>9.2} {:>7.2}",
                "", cols, rows, sample.median_ns / 1e6, gflops, sample.iqr_pct
            );
        }
    }

    if !json {
        // Mistral-7B: hidden 4096, intermediate 14336, 32 layers, and the
        // default calibration budget of 128 sequences of 512 tokens.
        let tokens = 128.0 * 512.0;
        let per = |width: usize| {
            seconds_per_token
                .iter()
                .find(|(c, _)| *c == width)
                .map(|(_, s)| *s)
                .unwrap_or(0.0)
        };
        let layer = 3.0 * per(4096) + per(14336);
        println!(
            "\nprojected calibration of a 7B (32 layers, {tokens:.0} tokens): \
             {:.0} min on this machine",
            layer * tokens * 32.0 / 60.0
        );
    }

    if json {
        println!("[{}]", rows_out.join(","));
    } else {
        println!(
            "\niqr% above ~5 means the machine moved more than a small kernel \
             change is worth; re-run before believing a few-percent difference.\n"
        );
    }
}
