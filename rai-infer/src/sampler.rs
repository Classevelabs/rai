//! Token sampling: temperature scaling, top-k, top-p (nucleus), random selection.
//!
//! Optimized: in-place operations, O(n) quickselect for top-k, O(k log k) sort
//! only on the ~k surviving candidates for top-p. Eliminates ~1.4 MB of allocations
//! and two O(n log n) sorts that the naive implementation required.

use rand::Rng;

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.0,
        }
    }
}

/// Sample a token from logits using temperature + top-k + top-p.
///
/// **Operates in-place on `logits`** — the slice is modified (temperature-scaled,
/// filtered, converted to probabilities). This eliminates ~1.4 MB of allocations
/// per call compared to the allocation-heavy approach.
pub fn sample_token(logits: &mut [f32], config: &SamplerConfig, rng: &mut impl Rng) -> usize {
    let vocab_size = logits.len();
    if vocab_size == 0 {
        return 0;
    }

    // Greedy: return argmax
    if config.temperature <= 1e-6 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
    }

    // Temperature scaling in-place (no allocation)
    let inv_temp = 1.0 / config.temperature;
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }

    // Top-k: use O(n) quickselect to find threshold, then filter
    // This replaces a full O(n log n) sort + 384 KB indices allocation
    if config.top_k > 0 && config.top_k < vocab_size {
        let k = config.top_k;
        // Only allocation: a copy of the values for quickselect (192 KB)
        let mut vals: Vec<f32> = logits.to_vec();
        // Put the k-th largest value at position k-1 (0-indexed)
        vals.select_nth_unstable_by(k - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        // vals[k-1] is the k-th largest value; keep everything >= threshold
        let threshold = vals[k - 1];
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax in-place (no allocation — replaces 192 KB probs Vec)
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }

    // Top-p (nucleus): only collect non-zero probabilities (~k elements after top-k)
    // This replaces a 576 KB allocation + full sort of 49K elements
    if config.top_p < 1.0 {
        // After top_k + softmax, at most ~k elements have p > 0
        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 1e-10)
            .map(|(i, &p)| (i, p))
            .collect();
        // Sort only the ~k candidates (trivial cost)
        candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0f32;
        let mut cutoff = candidates.len();
        for (rank, &(_, p)) in candidates.iter().enumerate() {
            cumsum += p;
            if cumsum >= config.top_p {
                cutoff = rank + 1;
                break;
            }
        }

        // Zero out everything, then restore nucleus tokens
        for v in logits.iter_mut() {
            *v = 0.0;
        }
        for &(idx, p) in &candidates[..cutoff] {
            logits[idx] = p;
        }

        // Re-normalize
        let sum: f32 = logits.iter().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in logits.iter_mut() {
                *v *= inv;
            }
        }
    }

    // Random sampling from the distribution
    let r: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in logits.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }

    // Fallback: return the highest probability token
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

/// Apply repetition penalty to logits for recently generated tokens.
pub fn apply_repetition_penalty(logits: &mut [f32], recent_tokens: &[usize], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 {
        return;
    }
    for &token in recent_tokens {
        if token < logits.len() {
            if logits[token] > 0.0 {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_greedy_sampling() {
        let mut logits = vec![1.0, 5.0, 2.0, 3.0];
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let token = sample_token(&mut logits, &config, &mut rng);
        assert_eq!(token, 1, "greedy should pick index 1 (logit=5.0)");
    }

    #[test]
    fn test_temperature_affects_distribution() {
        let config_low = SamplerConfig {
            temperature: 0.1,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut counts = [0usize; 3];
        for _ in 0..100 {
            let mut logits = vec![1.0, 2.0, 3.0];
            counts[sample_token(&mut logits, &config_low, &mut rng)] += 1;
        }
        assert!(
            counts[2] > 90,
            "low temp should strongly prefer token 2, got {:?}",
            counts
        );
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        apply_repetition_penalty(&mut logits, &[3], 2.0);
        assert!(
            (logits[3] - 2.0).abs() < 1e-6,
            "token 3 should be penalized: {}",
            logits[3]
        );
        assert!(
            (logits[0] - 1.0).abs() < 1e-6,
            "token 0 should be unchanged"
        );
    }

    #[test]
    fn test_top_k_filtering() {
        let logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut counts = [0usize; 5];
        for _ in 0..100 {
            let mut l = logits.clone();
            counts[sample_token(&mut l, &config, &mut rng)] += 1;
        }
        // Only tokens 1 (logit=5) and 3 (logit=4) should be sampled
        assert_eq!(counts[0], 0, "token 0 should never be sampled with top_k=2");
        assert_eq!(counts[2], 0, "token 2 should never be sampled with top_k=2");
        assert_eq!(counts[4], 0, "token 4 should never be sampled with top_k=2");
        assert!(counts[1] > 0, "token 1 should be sampled");
        assert!(counts[3] > 0, "token 3 should be sampled");
    }
}
