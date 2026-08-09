//! Microbenchmark for W4A8 GEMM kernel throughput.

use half::f16;
use std::time::Instant;

fn main() {
    rai_infer::gemm::configure_thread_pool();

    // Simulate SmolLM-135M dimensions
    let scenarios: [(&str, usize, usize); 4] = [
        ("q_proj  (576×576)", 576, 576),
        ("gate_proj(1536×576)", 1536, 576),
        ("down_proj(576×1536)", 576, 1536),
        ("lm_head  (49152×576)", 49152, 576),
    ];

    let group_size: usize = 128;

    for (name, rows, cols) in &scenarios {
        let rows = *rows;
        let cols = *cols;
        let num_groups = cols.div_ceil(group_size);

        // Allocate synthetic data
        let nibble_data: Vec<u8> = (0..rows * cols / 2)
            .map(|i| ((i as u8 * 7 + 3) % 16) | (((i as u8 * 11 + 5) % 16) << 4))
            .collect();

        let mut group_params = vec![0u8; rows * num_groups * 4];
        for r in 0..rows {
            for g in 0..num_groups {
                let off = (r * num_groups + g) * 4;
                let s = f16::from_f32(0.1);
                let z = f16::from_f32(-0.5);
                group_params[off..off + 2].copy_from_slice(&s.to_le_bytes());
                group_params[off + 2..off + 4].copy_from_slice(&z.to_le_bytes());
            }
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let mut output = vec![0.0f32; rows];

        // Warmup
        for _ in 0..3 {
            rai_infer::gemm::w4a8_matvec(
                &mut output,
                &nibble_data,
                &group_params,
                &input,
                rows,
                cols,
                group_size,
            );
        }

        // Benchmark
        let iters = if rows > 10000 { 10 } else { 50 };
        let t0 = Instant::now();
        for _ in 0..iters {
            rai_infer::gemm::w4a8_matvec(
                &mut output,
                &nibble_data,
                &group_params,
                &input,
                rows,
                cols,
                group_size,
            );
        }
        let dt = t0.elapsed().as_secs_f64();
        let per_call = dt / iters as f64 * 1e6; // microseconds

        let data_bytes = (rows * cols / 2 + rows * num_groups * 4) as f64;
        let bw = data_bytes * iters as f64 / dt / 1e9;

        eprintln!("{name}: {per_call:.1} μs ({bw:.1} GB/s effective)");
    }

    // Simulate full forward pass: 30 layers × 7 GEMMs
    eprintln!("\n--- Simulated forward pass (30 layers) ---");
    let layer_shapes: [(usize, usize); 7] = [
        (576, 576),  // q_proj
        (192, 576),  // k_proj
        (192, 576),  // v_proj
        (576, 576),  // o_proj
        (1536, 576), // gate_proj
        (1536, 576), // up_proj
        (576, 1536), // down_proj
    ];

    let mut total_us = 0.0f64;
    for &(rows, cols) in &layer_shapes {
        let num_groups = cols.div_ceil(group_size);
        let nibble_data: Vec<u8> = vec![0x55; rows * cols / 2];
        let mut group_params = vec![0u8; rows * num_groups * 4];
        for r in 0..rows {
            for g in 0..num_groups {
                let off = (r * num_groups + g) * 4;
                let s = f16::from_f32(0.1);
                let z = f16::from_f32(-0.5);
                group_params[off..off + 2].copy_from_slice(&s.to_le_bytes());
                group_params[off + 2..off + 4].copy_from_slice(&z.to_le_bytes());
            }
        }
        let input: Vec<f32> = vec![0.1; cols];
        let mut output = vec![0.0f32; rows];

        // Warmup
        rai_infer::gemm::w4a8_matvec(
            &mut output,
            &nibble_data,
            &group_params,
            &input,
            rows,
            cols,
            group_size,
        );

        let t0 = Instant::now();
        for _ in 0..30 {
            rai_infer::gemm::w4a8_matvec(
                &mut output,
                &nibble_data,
                &group_params,
                &input,
                rows,
                cols,
                group_size,
            );
        }
        let us = t0.elapsed().as_secs_f64() * 1e6;
        total_us += us;
    }

    // LM head (8-bit)
    let lm_rows: usize = 49152;
    let lm_cols: usize = 576;
    let lm_gs: usize = 128;
    let lm_num_groups = lm_cols.div_ceil(lm_gs);
    let embed_data: Vec<u8> = vec![128; lm_rows * lm_cols];
    let mut embed_params = vec![0u8; lm_rows * lm_num_groups * 4];
    for v in 0..lm_rows {
        for g in 0..lm_num_groups {
            let off = (v * lm_num_groups + g) * 4;
            let s = f16::from_f32(0.01);
            let z = f16::from_f32(-1.28);
            embed_params[off..off + 2].copy_from_slice(&s.to_le_bytes());
            embed_params[off + 2..off + 4].copy_from_slice(&z.to_le_bytes());
        }
    }
    let hidden: Vec<f32> = vec![0.1; lm_cols];
    let mut logits = vec![0.0f32; lm_rows];

    rai_infer::gemm::tied_lm_head(
        &mut logits,
        &hidden,
        &embed_data,
        &embed_params,
        lm_rows,
        lm_cols,
        lm_gs,
    );
    let t0 = Instant::now();
    rai_infer::gemm::tied_lm_head(
        &mut logits,
        &hidden,
        &embed_data,
        &embed_params,
        lm_rows,
        lm_cols,
        lm_gs,
    );
    let lm_us = t0.elapsed().as_secs_f64() * 1e6;

    eprintln!("Layer GEMMs (30 layers × 7): {total_us:.0} μs");
    eprintln!("LM head:                      {lm_us:.0} μs");
    eprintln!("Total GEMM compute:           {:.0} μs", total_us + lm_us);
    eprintln!(
        "Theoretical max tok/s:        {:.0}",
        1e6 / (total_us + lm_us)
    );
}
