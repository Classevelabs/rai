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
    /// Running sum of the residual norm observed at each `store`. One residual is recorded per
    /// stored item, so the mean is this sum divided by `items.len()`.
    #[serde(default)]
    residual_sum: f64,
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
            residual_sum: 0.0,
            last_loss: None,
        }
    }

    /// Restore an exact persisted state without replaying items through `store`.
    ///
    /// `residual_norm` is the persisted *mean* residual norm, so the running sum is rebuilt from
    /// it and the restored item count.
    pub fn from_snapshot(
        config: REMConfig,
        encoder: encoder::EncoderParams,
        decoder: decoder::DecoderParams,
        memory_state: Vec64,
        items: Vec<(Vec64, Vec64)>,
        residual_norm: f64,
        last_loss: Option<f64>,
    ) -> Result<Self> {
        if config.dim_memory == 0 || config.dim_key == 0 || config.dim_value == 0 {
            return Err(MemoryError::InvalidData(
                "memory, key, and value dimensions must be non-zero".to_string(),
            ));
        }
        ensure_dim(&encoder.bias, config.dim_memory)?;
        ensure_dim(&decoder.bias, config.dim_value)?;
        ensure_dim(&memory_state, config.dim_memory)?;
        ensure_finite(&encoder.bias, "encoder bias")?;
        ensure_finite(&decoder.bias, "decoder bias")?;
        ensure_finite(&memory_state, "memory state")?;
        for (key, value) in &items {
            ensure_dim(key, config.dim_key)?;
            ensure_dim(value, config.dim_value)?;
            ensure_finite(key, "stored key")?;
            ensure_finite(value, "stored value")?;
        }
        if !residual_norm.is_finite() || residual_norm < 0.0 {
            return Err(MemoryError::InvalidData(
                "residual norm must be finite and non-negative".to_string(),
            ));
        }
        if last_loss.is_some_and(|loss| !loss.is_finite() || loss < 0.0) {
            return Err(MemoryError::InvalidData(
                "last loss must be finite and non-negative".to_string(),
            ));
        }

        Ok(Self {
            residual_sum: residual_norm * items.len() as f64,
            config,
            encoder,
            decoder,
            memory_state,
            items,
            last_loss,
        })
    }

    pub fn store(&mut self, key: &Vec64, value: &Vec64) -> Result<()> {
        ensure_dim(key, self.config.dim_key)?;
        ensure_dim(value, self.config.dim_value)?;
        let prediction = self.predict(key);
        self.residual_sum += (value - prediction).norm();
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

    /// Mean residual norm across every stored item, or `0.0` for an empty memory.
    pub fn mean_residual_norm(&self) -> f64 {
        if self.items.is_empty() {
            0.0
        } else {
            self.residual_sum / self.items.len() as f64
        }
    }

    pub fn last_loss(&self) -> Option<f64> {
        self.last_loss
    }

    pub fn mse(&self) -> Result<f64> {
        Ok(self.last_loss.unwrap_or_else(|| self.current_mse()))
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn train(&mut self) -> Result<Vec<f64>> {
        Err(MemoryError::TrainingUnavailable)
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

fn ensure_finite(value: &Vec64, label: &str) -> Result<()> {
    if value.iter().all(|entry| entry.is_finite()) {
        Ok(())
    } else {
        Err(MemoryError::InvalidData(format!("{label} must be finite")))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_reports_unavailable_without_mutating_loss() {
        let mut memory =
            ResidualEquilibriumMemory::new(REMConfig::default(), &mut rand::thread_rng());
        assert!(memory.last_loss.is_none());
        assert!(matches!(
            memory.train(),
            Err(MemoryError::TrainingUnavailable)
        ));
        assert!(memory.last_loss.is_none());
    }

    fn two_dimensional() -> REMConfig {
        REMConfig {
            dim_memory: 2,
            dim_key: 2,
            dim_value: 2,
            ..Default::default()
        }
    }

    #[test]
    fn mean_residual_norm_averages_every_store_not_just_the_last() {
        let mut memory = ResidualEquilibriumMemory::new(two_dimensional(), &mut rand::thread_rng());
        assert_eq!(memory.mean_residual_norm(), 0.0);

        // First store predicts zeros, so the residual is the value norm: 3.
        memory
            .store(
                &Vec64::from_row_slice(&[1.0, 0.0]),
                &Vec64::from_row_slice(&[3.0, 0.0]),
            )
            .unwrap();
        assert!((memory.mean_residual_norm() - 3.0).abs() < 1e-12);

        // The orthogonal key still predicts the only stored value, so the residual is 5.
        memory
            .store(
                &Vec64::from_row_slice(&[0.0, 1.0]),
                &Vec64::from_row_slice(&[0.0, 4.0]),
            )
            .unwrap();
        let mean = memory.mean_residual_norm();
        assert!((mean - 4.0).abs() < 1e-12, "expected mean 4.0, got {mean}");
        assert!(
            (mean - 5.0).abs() > 1e-9,
            "mean must not be the last residual"
        );
    }

    #[test]
    fn snapshot_round_trips_the_persisted_mean_residual_norm() {
        let config = two_dimensional();
        let items = vec![
            (
                Vec64::from_row_slice(&[1.0, 0.0]),
                Vec64::from_row_slice(&[3.0, 0.0]),
            ),
            (
                Vec64::from_row_slice(&[0.0, 1.0]),
                Vec64::from_row_slice(&[0.0, 4.0]),
            ),
        ];
        let restored = ResidualEquilibriumMemory::from_snapshot(
            config.clone(),
            encoder::EncoderParams {
                bias: Vec64::zeros(config.dim_memory),
            },
            decoder::DecoderParams {
                bias: Vec64::zeros(config.dim_value),
            },
            Vec64::zeros(config.dim_memory),
            items,
            4.0,
            None,
        )
        .unwrap();
        assert!((restored.mean_residual_norm() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn snapshot_rejects_non_finite_and_negative_diagnostics() {
        let config = REMConfig::default();
        let encoder = encoder::EncoderParams {
            bias: Vec64::zeros(config.dim_memory),
        };
        let decoder = decoder::DecoderParams {
            bias: Vec64::zeros(config.dim_value),
        };
        let state = Vec64::zeros(config.dim_memory);

        assert!(ResidualEquilibriumMemory::from_snapshot(
            config.clone(),
            encoder.clone(),
            decoder.clone(),
            state.clone(),
            Vec::new(),
            -1.0,
            None,
        )
        .is_err());
        assert!(ResidualEquilibriumMemory::from_snapshot(
            config,
            encoder,
            decoder,
            state,
            Vec::new(),
            0.0,
            Some(f64::NAN),
        )
        .is_err());
    }
}
