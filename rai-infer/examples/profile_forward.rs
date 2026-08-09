//! Profile a complete forward pass to identify where time is spent.
//! Usage: cargo run --release -p rai-infer --example profile_forward

use anyhow::Result;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    rai_infer::gemm::configure_thread_pool();
    let model_path = Path::new("rai-infer/scripts/smollm-135m-q4.raimodel");
    let tok_path = Path::new("rai-infer/scripts/tokenizer.json");

    if !model_path.exists() {
        eprintln!("Model not found at {}", model_path.display());
        return Ok(());
    }

    eprintln!("Loading model...");
    let model = rai_infer::model::RaiModel::load(model_path)?;
    let tokenizer =
        tokenizers::Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;

    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;
    let max_ctx = 512;
    let mut kv_cache = model.create_kv_cache(max_ctx)?;
    let mut work = rai_infer::model::InferenceWork::new();
    let mut hidden = vec![0.0f32; hs];
    let mut normed = vec![0.0f32; hs];
    let mut logits = vec![0.0f32; vs];

    // Encode prompt
    let encoding = tokenizer
        .encode("The future of AI is", false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();

    // Prefill
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        model.embed_token(tok, &mut hidden)?;
        model.forward_from_hidden(&mut hidden, pos, &mut kv_cache, true, &mut work.scratch)?;
    }
    let pos = prompt_tokens.len();

    eprintln!("\n=== Profiling single token at position {pos} ===\n");

    // Profile a single forward pass with detailed timing
    let n = 20;
    let token_id = 262; // common token

    // Phase 1: embed_token
    let t = Instant::now();
    for _ in 0..n {
        model.embed_token(token_id, &mut hidden)?;
    }
    let embed_us = t.elapsed().as_micros() as f64 / n as f64;

    // Phase 2: forward_from_hidden (all 30 layers)
    model.embed_token(token_id, &mut hidden)?;
    let t = Instant::now();
    for _ in 0..n {
        let mut h = hidden.clone();
        model.forward_from_hidden(&mut h, pos, &mut kv_cache, true, &mut work.scratch)?;
    }
    let layers_us = t.elapsed().as_micros() as f64 / n as f64;

    // Phase 3: hidden_to_logits (final norm + lm_head)
    let t = Instant::now();
    for _ in 0..n {
        model.hidden_to_logits_into(&hidden, &mut normed, &mut logits)?;
    }
    let logits_us = t.elapsed().as_micros() as f64 / n as f64;

    // Phase 4: sampling
    let sampler_config = rai_infer::sampler::SamplerConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.1,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let recent = vec![262usize; 10];
    use rand::SeedableRng;

    let t = Instant::now();
    for _ in 0..n {
        let mut l = logits.clone();
        rai_infer::sampler::apply_repetition_penalty(&mut l, &recent, 1.1);
        rai_infer::sampler::sample_token(&mut l, &sampler_config, &mut rng);
    }
    let sample_us = t.elapsed().as_micros() as f64 / n as f64;

    // Phase 5: tokenizer decode
    let t = Instant::now();
    for _ in 0..n {
        let _ = tokenizer.decode(&[262u32], false);
    }
    let decode_us = t.elapsed().as_micros() as f64 / n as f64;

    let total = embed_us + layers_us + logits_us + sample_us + decode_us;

    eprintln!(
        "embed_token:       {:7.1} μs ({:5.1}%)",
        embed_us,
        100.0 * embed_us / total
    );
    eprintln!(
        "forward_layers:    {:7.1} μs ({:5.1}%)",
        layers_us,
        100.0 * layers_us / total
    );
    eprintln!(
        "hidden_to_logits:  {:7.1} μs ({:5.1}%)",
        logits_us,
        100.0 * logits_us / total
    );
    eprintln!(
        "sampling:          {:7.1} μs ({:5.1}%)",
        sample_us,
        100.0 * sample_us / total
    );
    eprintln!(
        "tokenizer_decode:  {:7.1} μs ({:5.1}%)",
        decode_us,
        100.0 * decode_us / total
    );
    eprintln!("─────────────────────────────────────");
    eprintln!(
        "total:             {:7.1} μs → {:.1} tok/s",
        total,
        1_000_000.0 / total
    );

    // Also profile just the GEMM portion of forward_layers
    // by timing layer iteration with different components
    eprintln!("\n=== Layer breakdown (avg of {n} iterations) ===\n");

    let file = &model;
    let layer0 = file.file_ref().layer(0)?;
    let input = vec![0.1f32; hs];
    let mut qout = vec![0.0f32; model.config.num_heads as usize * model.config.head_dim as usize];
    let mut kout =
        vec![0.0f32; model.config.num_kv_heads as usize * model.config.head_dim as usize];

    // Time individual GEMMs
    let t = Instant::now();
    for _ in 0..n {
        rai_infer::gemm::w4a32_matvec(
            &mut qout,
            layer0.q_proj.nibble_data,
            layer0.q_proj.group_params,
            &input,
            layer0.q_proj.rows,
            layer0.q_proj.cols,
            layer0.q_proj.group_size,
        );
    }
    let q_us = t.elapsed().as_micros() as f64 / n as f64;

    let t = Instant::now();
    for _ in 0..n {
        rai_infer::gemm::w4a32_matvec(
            &mut kout,
            layer0.k_proj.nibble_data,
            layer0.k_proj.group_params,
            &input,
            layer0.k_proj.rows,
            layer0.k_proj.cols,
            layer0.k_proj.group_size,
        );
    }
    let k_us = t.elapsed().as_micros() as f64 / n as f64;

    eprintln!("q_proj GEMM (576×576):  {:7.1} μs", q_us);
    eprintln!("k_proj GEMM (192×576):  {:7.1} μs", k_us);
    eprintln!("Estimated 7 GEMMs/layer × 30 = 210 GEMMs");
    eprintln!("forward_layers / 30:    {:7.1} μs/layer", layers_us / 30.0);

    Ok(())
}
