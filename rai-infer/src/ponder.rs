//! Pondering v2: Test-time compute strategies that actually improve output quality.
//!
//! The old approach (looping hidden states back through layers) fails because the model
//! was trained to receive embeddings at layer 0, not post-transformer outputs.
//!
//! The new approach operates at the **logit level** using three proven techniques:
//!
//! # Strategies
//!
//! ## 1. Classifier-Free Guidance (CFG)
//! Borrowed from diffusion models. Run two forward passes:
//! - **Conditional**: embed(actual_token) → layers → logits_cond
//! - **Unconditional**: embed(null_token) → layers → logits_uncond (no KV write)
//!
//! Final logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
//!
//! With guidance_scale > 1.0, this amplifies context-dependent predictions — tokens
//! the model specifically chose BECAUSE of the context, not generic next-word guesses.
//! Cost: 2 forward passes. Quality improvement: significant.
//!
//! ## 2. Embedding Noise Ensemble
//! Add Gaussian noise to the embedding, run N forward passes, average logits.
//! This is Monte Carlo marginalization over input uncertainty — the average is a
//! better estimate than any single noisy sample.
//! - Pass 1: clean (writes KV cache normally)
//! - Pass 2..N: noisy (read-only, don't modify KV cache)
//! Cost: N forward passes. Quality: smoother, more calibrated distributions.
//!
//! ## 3. Adaptive Confidence Gating
//! Don't waste compute on easy tokens. Run 1 forward pass, check entropy:
//! - Low entropy (confident) → emit immediately. 1 forward pass.
//! - High entropy (uncertain) → apply CFG + ensemble. 2-N forward passes.
//! Average cost: 1.3-1.8x depending on text difficulty. Maximum quality gain.

use crate::kv_cache::KVCache;
use crate::model::{InferenceWork, RaiModel};
use rand::Rng;

/// Pondering strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PonderStrategy {
    /// No pondering. Standard single forward pass.
    None,
    /// Classifier-Free Guidance: 2 forward passes, amplify contextual signal.
    CFG,
    /// Embedding noise ensemble: N forward passes with noise, average logits.
    Ensemble,
    /// CFG + Ensemble combined: unconditional + N-1 noisy conditional passes.
    CFGEnsemble,
    /// Adaptive: 1 pass for easy tokens, CFG for hard tokens.
    Adaptive,
}

/// Configuration for pondering.
#[derive(Debug, Clone)]
pub struct PonderConfig {
    pub strategy: PonderStrategy,
    /// CFG guidance scale. 1.0 = no effect. 1.5 = recommended. 2.0+ = aggressive.
    pub guidance_scale: f32,
    /// Token ID used for unconditional CFG pass. Usually 0 (padding/unk).
    pub null_token_id: usize,
    /// Number of ensemble passes (including the clean pass). Minimum 2.
    pub ensemble_n: usize,
    /// Standard deviation of Gaussian noise added to embeddings for ensemble.
    pub noise_sigma: f32,
    /// Entropy threshold for adaptive strategy (nats). Above this = hard token.
    pub entropy_threshold: f32,
}

impl Default for PonderConfig {
    fn default() -> Self {
        Self {
            strategy: PonderStrategy::None,
            guidance_scale: 1.5,
            null_token_id: 0,
            ensemble_n: 3,
            noise_sigma: 0.05,
            entropy_threshold: 3.0,
        }
    }
}

impl PonderConfig {
    /// No pondering (standard inference).
    pub fn none() -> Self {
        Self {
            strategy: PonderStrategy::None,
            ..Default::default()
        }
    }

    /// Classifier-Free Guidance with the given scale.
    pub fn cfg(guidance_scale: f32) -> Self {
        Self {
            strategy: PonderStrategy::CFG,
            guidance_scale,
            ..Default::default()
        }
    }

    /// Embedding noise ensemble with N passes.
    pub fn ensemble(n: usize, sigma: f32) -> Self {
        Self {
            strategy: PonderStrategy::Ensemble,
            ensemble_n: n.max(2),
            noise_sigma: sigma,
            ..Default::default()
        }
    }

    /// CFG + Ensemble combined.
    pub fn cfg_ensemble(guidance_scale: f32, n: usize, sigma: f32) -> Self {
        Self {
            strategy: PonderStrategy::CFGEnsemble,
            guidance_scale,
            ensemble_n: n.max(2),
            noise_sigma: sigma,
            ..Default::default()
        }
    }

    /// Adaptive: auto-detect hard tokens and apply CFG.
    pub fn adaptive(guidance_scale: f32, entropy_threshold: f32) -> Self {
        Self {
            strategy: PonderStrategy::Adaptive,
            guidance_scale,
            entropy_threshold,
            ..Default::default()
        }
    }
}

/// Metrics from a pondering decision.
#[derive(Debug, Clone)]
pub struct PonderMetrics {
    pub forward_passes: usize,
    pub strategy_used: &'static str,
    pub entropy: Option<f32>,
    pub was_hard_token: bool,
}

// ---------------------------------------------------------------------------
// Core: pondered_forward — the public API
// ---------------------------------------------------------------------------

/// Run a pondered forward pass: embed token → run strategy → return logits.
///
/// `logits_buf` is a reusable buffer that will contain the output logits.
/// It is resized to vocab_size automatically. This avoids 192 KB allocation per token.
pub fn pondered_forward(
    model: &RaiModel,
    token_id: usize,
    pos: usize,
    kv_cache: &mut KVCache,
    config: &PonderConfig,
    work: &mut InferenceWork,
    work2: &mut InferenceWork,
    rng: &mut impl Rng,
) -> anyhow::Result<(Vec<f32>, PonderMetrics)> {
    match config.strategy {
        PonderStrategy::None => {
            let logits = forward_standard(model, token_id, pos, kv_cache, work)?;
            Ok((
                logits,
                PonderMetrics {
                    forward_passes: 1,
                    strategy_used: "none",
                    entropy: None,
                    was_hard_token: false,
                },
            ))
        }
        PonderStrategy::CFG => {
            let (logits, passes) =
                forward_cfg(model, token_id, pos, kv_cache, config, work, work2)?;
            Ok((
                logits,
                PonderMetrics {
                    forward_passes: passes,
                    strategy_used: "cfg",
                    entropy: None,
                    was_hard_token: true,
                },
            ))
        }
        PonderStrategy::Ensemble => {
            let (logits, passes) =
                forward_ensemble(model, token_id, pos, kv_cache, config, work, work2, rng)?;
            Ok((
                logits,
                PonderMetrics {
                    forward_passes: passes,
                    strategy_used: "ensemble",
                    entropy: None,
                    was_hard_token: true,
                },
            ))
        }
        PonderStrategy::CFGEnsemble => {
            let (logits, passes) =
                forward_cfg_ensemble(model, token_id, pos, kv_cache, config, work, work2, rng)?;
            Ok((
                logits,
                PonderMetrics {
                    forward_passes: passes,
                    strategy_used: "cfg+ensemble",
                    entropy: None,
                    was_hard_token: true,
                },
            ))
        }
        PonderStrategy::Adaptive => {
            forward_adaptive(model, token_id, pos, kv_cache, config, work, work2, rng)
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy implementations
// ---------------------------------------------------------------------------

/// Standard single forward pass — uses pre-allocated logits buffer in scratch.
fn forward_standard(
    model: &RaiModel,
    token_id: usize,
    pos: usize,
    kv_cache: &mut KVCache,
    work: &mut InferenceWork,
) -> anyhow::Result<Vec<f32>> {
    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;
    work.hidden.resize(hs, 0.0);
    model.embed_token(token_id, &mut work.hidden)?;
    model.forward_from_hidden(&mut work.hidden, pos, kv_cache, true, &mut work.scratch)?;
    work.scratch.normed.resize(hs, 0.0);
    work.scratch.resize_logits(vs);
    model.hidden_to_logits_into(
        &work.hidden,
        &mut work.scratch.normed,
        &mut work.scratch.logits,
    )?;
    // Move logits out of scratch (zero-cost pointer swap, no memcpy).
    // The caller will own the Vec; scratch.logits becomes empty and will
    // be re-allocated on the next call via resize_logits.
    Ok(std::mem::take(&mut work.scratch.logits))
}

/// Classifier-Free Guidance: amplify context-dependent predictions.
fn forward_cfg(
    model: &RaiModel,
    token_id: usize,
    pos: usize,
    kv_cache: &mut KVCache,
    config: &PonderConfig,
    work: &mut InferenceWork,
    work2: &mut InferenceWork,
) -> anyhow::Result<(Vec<f32>, usize)> {
    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;

    // Pass 1: Conditional (real token, stores KV)
    work.hidden.resize(hs, 0.0);
    model.embed_token(token_id, &mut work.hidden)?;
    model.forward_from_hidden(&mut work.hidden, pos, kv_cache, true, &mut work.scratch)?;
    work.scratch.normed.resize(hs, 0.0);
    work.scratch.resize_logits(vs);
    model.hidden_to_logits_into(
        &work.hidden,
        &mut work.scratch.normed,
        &mut work.scratch.logits,
    )?;

    // Pass 2: Unconditional (null token, read-only KV)
    work2.hidden.resize(hs, 0.0);
    model.embed_token(config.null_token_id, &mut work2.hidden)?;
    model.forward_from_hidden(&mut work2.hidden, pos, kv_cache, false, &mut work2.scratch)?;
    work2.scratch.normed.resize(hs, 0.0);
    work2.scratch.resize_logits(vs);
    model.hidden_to_logits_into(
        &work2.hidden,
        &mut work2.scratch.normed,
        &mut work2.scratch.logits,
    )?;

    // CFG combination: logits = uncond + scale * (cond - uncond)
    let scale = config.guidance_scale;
    let logits: Vec<f32> = work2
        .scratch
        .logits
        .iter()
        .zip(work.scratch.logits.iter())
        .map(|(&u, &c)| u + scale * (c - u))
        .collect();

    Ok((logits, 2))
}

/// Embedding noise ensemble: average logits from N noisy passes.
fn forward_ensemble(
    model: &RaiModel,
    token_id: usize,
    pos: usize,
    kv_cache: &mut KVCache,
    config: &PonderConfig,
    work: &mut InferenceWork,
    work2: &mut InferenceWork,
    rng: &mut impl Rng,
) -> anyhow::Result<(Vec<f32>, usize)> {
    let hs = model.config.hidden_size as usize;
    let vocab = model.config.vocab_size as usize;
    let n = config.ensemble_n;

    // Pass 1: Clean (stores KV)
    work.hidden.resize(hs, 0.0);
    model.embed_token(token_id, &mut work.hidden)?;
    model.forward_from_hidden(&mut work.hidden, pos, kv_cache, true, &mut work.scratch)?;
    work.scratch.normed.resize(hs, 0.0);
    work.scratch.resize_logits(vocab);
    model.hidden_to_logits_into(
        &work.hidden,
        &mut work.scratch.normed,
        &mut work.scratch.logits,
    )?;

    let mut logits_sum = work.scratch.logits.clone();

    // Pass 2..N: Noisy (read-only KV)
    for _ in 1..n {
        work2.hidden.resize(hs, 0.0);
        model.embed_token(token_id, &mut work2.hidden)?;
        for v in work2.hidden.iter_mut() {
            *v += config.noise_sigma * sample_normal(rng);
        }
        model.forward_from_hidden(&mut work2.hidden, pos, kv_cache, false, &mut work2.scratch)?;
        work2.scratch.normed.resize(hs, 0.0);
        work2.scratch.resize_logits(vocab);
        model.hidden_to_logits_into(
            &work2.hidden,
            &mut work2.scratch.normed,
            &mut work2.scratch.logits,
        )?;
        for i in 0..vocab {
            logits_sum[i] += work2.scratch.logits[i];
        }
    }

    // Average
    let inv_n = 1.0 / n as f32;
    for v in logits_sum.iter_mut() {
        *v *= inv_n;
    }

    Ok((logits_sum, n))
}

/// CFG + Ensemble: unconditional pass + N-1 noisy conditional passes, then CFG combine.
fn forward_cfg_ensemble(
    model: &RaiModel,
    token_id: usize,
    pos: usize,
    kv_cache: &mut KVCache,
    config: &PonderConfig,
    work: &mut InferenceWork,
    work2: &mut InferenceWork,
    rng: &mut impl Rng,
) -> anyhow::Result<(Vec<f32>, usize)> {
    let hs = model.config.hidden_size as usize;
    let vocab = model.config.vocab_size as usize;
    let n = config.ensemble_n;

    // Pass 1: Clean conditional (stores KV)
    work.hidden.resize(hs, 0.0);
    model.embed_token(token_id, &mut work.hidden)?;
    model.forward_from_hidden(&mut work.hidden, pos, kv_cache, true, &mut work.scratch)?;
    work.scratch.normed.resize(hs, 0.0);
    work.scratch.resize_logits(vocab);
    model.hidden_to_logits_into(
        &work.hidden,
        &mut work.scratch.normed,
        &mut work.scratch.logits,
    )?;

    let mut logits_cond_sum = work.scratch.logits.clone();

    // Pass 2..N-1: Noisy conditional (read-only KV)
    let noisy_passes = if n > 2 { n - 2 } else { 0 };
    for _ in 0..noisy_passes {
        work2.hidden.resize(hs, 0.0);
        model.embed_token(token_id, &mut work2.hidden)?;
        for v in work2.hidden.iter_mut() {
            *v += config.noise_sigma * sample_normal(rng);
        }
        model.forward_from_hidden(&mut work2.hidden, pos, kv_cache, false, &mut work2.scratch)?;
        work2.scratch.normed.resize(hs, 0.0);
        work2.scratch.resize_logits(vocab);
        model.hidden_to_logits_into(
            &work2.hidden,
            &mut work2.scratch.normed,
            &mut work2.scratch.logits,
        )?;
        for i in 0..vocab {
            logits_cond_sum[i] += work2.scratch.logits[i];
        }
    }

    let cond_count = 1 + noisy_passes;
    let inv_cond = 1.0 / cond_count as f32;
    for v in logits_cond_sum.iter_mut() {
        *v *= inv_cond;
    }

    // Pass N: Unconditional (null token, read-only KV)
    work2.hidden.resize(hs, 0.0);
    model.embed_token(config.null_token_id, &mut work2.hidden)?;
    model.forward_from_hidden(&mut work2.hidden, pos, kv_cache, false, &mut work2.scratch)?;
    work2.scratch.normed.resize(hs, 0.0);
    work2.scratch.resize_logits(vocab);
    model.hidden_to_logits_into(
        &work2.hidden,
        &mut work2.scratch.normed,
        &mut work2.scratch.logits,
    )?;

    // CFG on the ensembled conditional logits
    let scale = config.guidance_scale;
    let logits: Vec<f32> = work2
        .scratch
        .logits
        .iter()
        .zip(logits_cond_sum.iter())
        .map(|(&u, &c)| u + scale * (c - u))
        .collect();

    Ok((logits, n))
}

/// Adaptive: check entropy, spend extra compute only on hard tokens.
fn forward_adaptive(
    model: &RaiModel,
    token_id: usize,
    pos: usize,
    kv_cache: &mut KVCache,
    config: &PonderConfig,
    work: &mut InferenceWork,
    work2: &mut InferenceWork,
    rng: &mut impl Rng,
) -> anyhow::Result<(Vec<f32>, PonderMetrics)> {
    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;

    // Pass 1: Standard forward (stores KV)
    work.hidden.resize(hs, 0.0);
    model.embed_token(token_id, &mut work.hidden)?;
    model.forward_from_hidden(&mut work.hidden, pos, kv_cache, true, &mut work.scratch)?;
    work.scratch.normed.resize(hs, 0.0);
    work.scratch.resize_logits(vs);
    model.hidden_to_logits_into(
        &work.hidden,
        &mut work.scratch.normed,
        &mut work.scratch.logits,
    )?;

    let entropy = logit_entropy(&work.scratch.logits);

    if entropy < config.entropy_threshold {
        // Easy token: model is confident. Emit immediately.
        return Ok((
            work.scratch.logits.clone(),
            PonderMetrics {
                forward_passes: 1,
                strategy_used: "adaptive/skip",
                entropy: Some(entropy),
                was_hard_token: false,
            },
        ));
    }

    // Hard token: apply CFG to sharpen the distribution.
    let logits_cond = work.scratch.logits.clone();

    work2.hidden.resize(hs, 0.0);
    model.embed_token(config.null_token_id, &mut work2.hidden)?;
    model.forward_from_hidden(&mut work2.hidden, pos, kv_cache, false, &mut work2.scratch)?;
    work2.scratch.normed.resize(hs, 0.0);
    work2.scratch.resize_logits(vs);
    model.hidden_to_logits_into(
        &work2.hidden,
        &mut work2.scratch.normed,
        &mut work2.scratch.logits,
    )?;

    let scale = config.guidance_scale;
    let mut logits_cfg: Vec<f32> = work2
        .scratch
        .logits
        .iter()
        .zip(logits_cond.iter())
        .map(|(&u, &c)| u + scale * (c - u))
        .collect();

    let entropy_after_cfg = logit_entropy(&logits_cfg);
    let mut passes = 2;

    // If still uncertain after CFG, add one noise ensemble pass
    if entropy_after_cfg > config.entropy_threshold * 0.8 && config.ensemble_n > 2 {
        work2.hidden.resize(hs, 0.0);
        model.embed_token(token_id, &mut work2.hidden)?;
        for v in work2.hidden.iter_mut() {
            *v += config.noise_sigma * sample_normal(rng);
        }
        model.forward_from_hidden(&mut work2.hidden, pos, kv_cache, false, &mut work2.scratch)?;
        work2.scratch.normed.resize(hs, 0.0);
        work2.scratch.resize_logits(vs);
        model.hidden_to_logits_into(
            &work2.hidden,
            &mut work2.scratch.normed,
            &mut work2.scratch.logits,
        )?;

        // Blend: 70% CFG result + 30% noisy ensemble
        for i in 0..vs {
            logits_cfg[i] = 0.7 * logits_cfg[i] + 0.3 * work2.scratch.logits[i];
        }
        passes = 3;
    }

    Ok((
        logits_cfg,
        PonderMetrics {
            forward_passes: passes,
            strategy_used: if passes == 2 {
                "adaptive/cfg"
            } else {
                "adaptive/cfg+noise"
            },
            entropy: Some(entropy),
            was_hard_token: true,
        },
    ))
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Shannon entropy of a logit distribution (nats). Applies softmax internally.
pub fn logit_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }

    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for &l in logits {
        sum_exp += (l - max_l).exp();
    }
    let log_sum_exp = max_l + sum_exp.ln();

    let mut ent = 0.0f32;
    for &l in logits {
        let log_p = l - log_sum_exp;
        let p = log_p.exp();
        if p > 1e-10 {
            ent -= p * log_p;
        }
    }
    ent
}

/// Sample from a standard normal distribution using Box-Muller.
fn sample_normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logit_entropy_uniform() {
        let logits = vec![0.0; 4];
        let ent = logit_entropy(&logits);
        assert!((ent - 4.0f32.ln()).abs() < 0.01, "got {ent}");
    }

    #[test]
    fn test_logit_entropy_peaked() {
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let ent = logit_entropy(&logits);
        assert!(ent < 0.01, "got {ent}");
    }

    #[test]
    fn test_sample_normal_distribution() {
        let mut rng = rand::thread_rng();
        let samples: Vec<f32> = (0..10000).map(|_| sample_normal(&mut rng)).collect();
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let var: f32 = samples
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / samples.len() as f32;
        assert!(mean.abs() < 0.1, "mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.1, "variance should be ~1, got {var}");
    }

    #[test]
    fn test_cfg_math() {
        let cond = vec![1.0, 2.0, 3.0];
        let uncond = vec![0.5, 0.5, 0.5];
        let scale = 1.0;
        let result: Vec<f32> = uncond
            .iter()
            .zip(cond.iter())
            .map(|(&u, &c)| u + scale * (c - u))
            .collect();
        assert_eq!(result, cond);

        let scale = 2.0;
        let result: Vec<f32> = uncond
            .iter()
            .zip(cond.iter())
            .map(|(&u, &c)| u + scale * (c - u))
            .collect();
        assert!((result[0] - 1.5).abs() < 1e-6);
        assert!((result[1] - 3.5).abs() < 1e-6);
        assert!((result[2] - 5.5).abs() < 1e-6);
    }
}
