use half::f16;
use std::time::Instant;

fn main() {
    let rows: usize = 1536;
    let cols: usize = 576;
    let gs: usize = 128;
    let num_groups = cols.div_ceil(gs);

    let scale = f16::from_f32(0.1);
    let zero = f16::from_f32(-0.5);
    let mut gp = vec![0u8; rows * num_groups * 4];
    for r in 0..rows {
        for g in 0..num_groups {
            let off = (r * num_groups + g) * 4;
            gp[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            gp[off + 2..off + 4].copy_from_slice(&zero.to_le_bytes());
        }
    }
    let nd: Vec<u8> = (0..rows * cols / 2)
        .map(|i| (i as u8).wrapping_mul(7))
        .collect();
    let inp: Vec<f32> = (0..cols).map(|i| i as f32 * 0.001).collect();
    let mut out = vec![0.0f32; rows];

    // Warmup
    for _ in 0..5 {
        rai_infer::gemm::w4a32_matvec(&mut out, &nd, &gp, &inp, rows, cols, gs);
    }

    // Benchmark matvec
    let n = 200;
    let t = Instant::now();
    for _ in 0..n {
        rai_infer::gemm::w4a32_matvec(&mut out, &nd, &gp, &inp, rows, cols, gs);
    }
    let us = t.elapsed().as_micros();
    let per_call = us as f64 / n as f64;
    println!(
        "1536x576 matvec: {:.1} μs/call, {:.1} Gelem/s",
        per_call,
        (rows * cols) as f64 / per_call / 1000.0
    );

    // Also benchmark all layer sizes for accurate per-token estimate
    let layer_shapes: [(usize, usize, &str); 4] = [
        (576, 576, "q/o_proj (576x576)"),
        (192, 576, "k/v_proj (192x576)"),
        (1536, 576, "gate/up (1536x576)"),
        (576, 1536, "down (576x1536)"),
    ];

    for &(r, c, label) in &layer_shapes {
        let ng = c.div_ceil(gs);
        let mut gp2 = vec![0u8; r * ng * 4];
        for row in 0..r {
            for g in 0..ng {
                let off = (row * ng + g) * 4;
                gp2[off..off + 2].copy_from_slice(&scale.to_le_bytes());
                gp2[off + 2..off + 4].copy_from_slice(&zero.to_le_bytes());
            }
        }
        let nd2: Vec<u8> = (0..r * c / 2).map(|i| (i as u8).wrapping_mul(7)).collect();
        let inp2: Vec<f32> = (0..c).map(|i| i as f32 * 0.001).collect();
        let mut out2 = vec![0.0f32; r];

        for _ in 0..3 {
            rai_infer::gemm::w4a32_matvec(&mut out2, &nd2, &gp2, &inp2, r, c, gs);
        }
        let t = Instant::now();
        for _ in 0..n {
            rai_infer::gemm::w4a32_matvec(&mut out2, &nd2, &gp2, &inp2, r, c, gs);
        }
        let us = t.elapsed().as_micros();
        let per = us as f64 / n as f64;
        println!("{label}: {:.1} μs/call", per);
    }

    // Estimate per-token time (layer GEMMs)
    // Per layer: q(576x576) + k(192x576) + v(192x576) + o(576x576) + gate(1536x576) + up(1536x576) + down(576x1536)
    // (uses last benchmarks)

    // LM Head benchmark
    let vocab: usize = 49152;
    let hidden: usize = 576;
    let embed_gs: usize = 64;
    let embed_ng = hidden.div_ceil(embed_gs);
    let mut ep = vec![0u8; vocab * embed_ng * 4];
    for v in 0..vocab {
        for g in 0..embed_ng {
            let off = (v * embed_ng + g) * 4;
            ep[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            ep[off + 2..off + 4].copy_from_slice(&zero.to_le_bytes());
        }
    }
    let ed: Vec<u8> = (0..vocab * hidden)
        .map(|i| (i as u8).wrapping_mul(3))
        .collect();
    let h: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.01).collect();
    let mut logits = vec![0.0f32; vocab];

    for _ in 0..3 {
        rai_infer::gemm::tied_lm_head(&mut logits, &h, &ed, &ep, vocab, hidden, embed_gs);
    }
    let n2 = 50;
    let t2 = Instant::now();
    for _ in 0..n2 {
        rai_infer::gemm::tied_lm_head(&mut logits, &h, &ed, &ep, vocab, hidden, embed_gs);
    }
    let us2 = t2.elapsed().as_micros();
    let per_lmhead = us2 as f64 / n2 as f64;
    println!(
        "lm_head (49152x576): {:.1} μs/call, {:.1} Gelem/s",
        per_lmhead,
        (vocab * hidden) as f64 / per_lmhead / 1000.0
    );
}
