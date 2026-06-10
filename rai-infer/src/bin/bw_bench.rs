//! Micro-benchmarks: raw memory bandwidth + rayon dispatch overhead.

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    rai_infer::gemm::configure_thread_pool();

    // 1. Raw sequential read bandwidth (single-threaded)
    let size = 85 * 1024 * 1024; // 85 MB
    let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
    let mut sink = 0u64;

    // Warmup
    for chunk in data.chunks(64) {
        sink = sink.wrapping_add(unsafe { *(chunk.as_ptr() as *const u64) });
    }

    let iters = 10;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut s = 0u64;
        for chunk in data.chunks(64) {
            s = s.wrapping_add(unsafe { *(chunk.as_ptr() as *const u64) });
        }
        sink = sink.wrapping_add(s);
    }
    let dt = t0.elapsed().as_secs_f64();
    let bw = (size * iters) as f64 / dt / 1e9;
    eprintln!("1. Sequential read (1 thread): {bw:.1} GB/s  (sink={sink})");

    // 2. Parallel read bandwidth (rayon par_chunks)
    let t0 = Instant::now();
    for _ in 0..iters {
        let s: u64 = data
            .par_chunks(1024 * 1024)
            .map(|chunk| {
                let mut s = 0u64;
                for c in chunk.chunks(64) {
                    s = s.wrapping_add(unsafe { *(c.as_ptr() as *const u64) });
                }
                s
            })
            .sum();
        sink = sink.wrapping_add(s);
    }
    let dt = t0.elapsed().as_secs_f64();
    let bw = (size * iters) as f64 / dt / 1e9;
    eprintln!("2. Parallel read (rayon 1MB chunks): {bw:.1} GB/s");

    // 3. Rayon dispatch overhead (empty work)
    let iters2 = 10000;
    let chunks = 24; // typical chunk count
    let t0 = Instant::now();
    for _ in 0..iters2 {
        (0..chunks).into_par_iter().for_each(|_i| {
            std::hint::black_box(0);
        });
    }
    let dt = t0.elapsed().as_secs_f64();
    let per_dispatch = dt / iters2 as f64 * 1e6;
    eprintln!("3. Rayon dispatch overhead ({chunks} empty items): {per_dispatch:.2} μs");

    // 4. Rayon dispatch with small work (simulates O_proj-sized GEMM)
    let small_data: Vec<u8> = vec![0x55; 177 * 1024]; // 177 KB like O_proj
    let t0 = Instant::now();
    for _ in 0..iters2 {
        let chunk_size = small_data.len() / chunks;
        (0..chunks).into_par_iter().for_each(|ci| {
            let start = ci * chunk_size;
            let end = (start + chunk_size).min(small_data.len());
            let mut s = 0u64;
            for c in small_data[start..end].chunks(64) {
                s = s.wrapping_add(unsafe { *(c.as_ptr() as *const u64) });
            }
            std::hint::black_box(s);
        });
    }
    let dt = t0.elapsed().as_secs_f64();
    let per_call = dt / iters2 as f64 * 1e6;
    let bw = (177.0 * 1024.0) / (per_call * 1e-6) / 1e9;
    eprintln!("4. Rayon 177KB (O_proj-sized) read: {per_call:.2} μs ({bw:.1} GB/s)");

    // 5. Same but single-threaded
    let t0 = Instant::now();
    for _ in 0..iters2 {
        let mut s = 0u64;
        for c in small_data.chunks(64) {
            s = s.wrapping_add(unsafe { *(c.as_ptr() as *const u64) });
        }
        std::hint::black_box(s);
    }
    let dt = t0.elapsed().as_secs_f64();
    let per_call = dt / iters2 as f64 * 1e6;
    let bw = (177.0 * 1024.0) / (per_call * 1e-6) / 1e9;
    eprintln!("5. Single-thread 177KB read: {per_call:.2} μs ({bw:.1} GB/s)");

    // 6. Mmap read bandwidth test
    let model_path = "rai-infer/scripts/smollm-135m-q4.raimodel";
    if let Ok(file) = std::fs::File::open(model_path) {
        let mmap = unsafe { memmap2::MmapOptions::new().populate().map(&file).unwrap() };
        let mmap_size = mmap.len();

        // Warmup
        let mut s = 0u64;
        for c in mmap.chunks(64) {
            s = s.wrapping_add(unsafe { *(c.as_ptr() as *const u64) });
        }
        sink = sink.wrapping_add(s);

        let iters = 5;
        let t0 = Instant::now();
        for _ in 0..iters {
            let s: u64 = mmap
                .par_chunks(1024 * 1024)
                .map(|chunk| {
                    let mut s = 0u64;
                    for c in chunk.chunks(64) {
                        s = s.wrapping_add(unsafe { *(c.as_ptr() as *const u64) });
                    }
                    s
                })
                .sum();
            sink = sink.wrapping_add(s);
        }
        let dt = t0.elapsed().as_secs_f64();
        let bw = (mmap_size * iters) as f64 / dt / 1e9;
        eprintln!(
            "6. Mmap parallel read ({:.1} MB): {bw:.1} GB/s",
            mmap_size as f64 / 1e6
        );
    }

    std::hint::black_box(sink);
}
