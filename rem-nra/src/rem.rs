use crate::{MemoryError, Result, Vec64};
use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

pub mod encoder {
    use crate::Vec64;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EncoderParams {
        pub bias: Vec64,
    }
}

pub mod decoder {
    use crate::Vec64;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DecoderParams {
        pub bias: Vec64,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct REMConfig {
    pub dim_memory: usize,
    pub dim_key: usize,
    pub dim_value: usize,
    pub train_epochs: usize,
}

impl Default for REMConfig {
    fn default() -> Self {
        Self {
            dim_memory: 256,
            dim_key: 32,
            dim_value: 64,
            train_epochs: 200,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualEquilibriumMemory {
    pub config: REMConfig,
    pub encoder: encoder::EncoderParams,
    pub decoder: decoder::DecoderParams,
    pub memory_state: Vec64,
    items: Vec<(Vec64, Vec64)>,
    residual_norm: f64,
    last_loss: Option<f64>,
}

impl ResidualEquilibriumMemory {
    pub fn new(config: REMConfig, rng: &mut impl Rng) -> Self {
        let encoder_bias = Vec64::from_fn(config.dim_memory, |_, _| {
            rng.sample::<f64, _>(StandardNormal) * 0.01
        });
        let decoder_bias = Vec64::from_fn(config.dim_value, |_, _| {
            rng.sample::<f64, _>(StandardNormal) * 0.01
        });
        Self {
            memory_state: Vec64::zeros(config.dim_memory),
            encoder: encoder::EncoderParams { bias: encoder_bias },
            decoder: decoder::DecoderParams { bias: decoder_bias },
            config,
            items: Vec::new(),
            residual_norm: 0.0,
            last_loss: None,
        }
    }

    pub fn store(&mut self, key: &Vec64, value: &Vec64) -> Result<()> {
        ensure_dim(key, self.config.dim_key)?;
        ensure_dim(value, self.config.dim_value)?;
        let prediction = self.predict(key);
        self.residual_norm = (value - prediction).norm();
        self.items.push((key.clone(), value.clone()));
        self.memory_state = rolling_average(&self.memory_state, value);
        self.last_loss = Some(self.current_mse());
        Ok(())
    }

    pub fn predict(&self, key: &Vec64) -> Vec64 {
        if self.items.is_empty() {
            return Vec64::zeros(self.config.dim_value);
        }
        let mut best = &self.items[0].1;
        let mut best_score = f64::NEG_INFINITY;
        for (stored_key, value) in &self.items {
            let score = cosine(key, stored_key);
            if score > best_score {
                best_score = score;
                best = value;
            }
        }
        best.clone()
    }

    pub fn mean_residual_norm(&self) -> f64 {
        self.residual_norm
    }

    pub fn mse(&self) -> Result<f64> {
        Ok(self.last_loss.unwrap_or_else(|| self.current_mse()))
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn train(&mut self) -> Result<Vec<f64>> {
        let loss = self.current_mse();
        self.last_loss = Some(loss);
        Ok(vec![loss])
    }

    pub fn items(&self) -> &[(Vec64, Vec64)] {
        &self.items
    }

    fn current_mse(&self) -> f64 {
        if self.items.is_empty() {
            return 0.0;
        }
        self.items
            .iter()
            .map(|(_, value)| value.norm_squared() / value.len().max(1) as f64)
            .sum::<f64>()
            / self.items.len() as f64
    }
}

fn ensure_dim(value: &Vec64, expected: usize) -> Result<()> {
    if value.len() == expected {
        Ok(())
    } else {
        Err(MemoryError::DimensionMismatch {
            expected,
            actual: value.len(),
        })
    }
}

fn rolling_average(state: &Vec64, value: &Vec64) -> Vec64 {
    let mut next = state.clone();
    let limit = next.len().min(value.len());
    for i in 0..limit {
        next[i] = 0.95 * next[i] + 0.05 * value[i];
    }
    next
}

fn cosine(a: &Vec64, b: &Vec64) -> f64 {
    let denom = a.norm() * b.norm();
    if denom <= 1e-12 {
        0.0
    } else {
        a.dot(b) / denom
    }
}
