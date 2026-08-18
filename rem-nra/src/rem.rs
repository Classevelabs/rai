use crate::{MemoryError, Result, Vec64};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct REMConfig {
    /// Nominal working dimension carried with the store for snapshot identity.
    pub dim_memory: usize,
    /// Dimension of the key vectors accepted by `store` and `predict`.
    pub dim_key: usize,
    /// Dimension of the value vectors accepted by `store` and returned by `predict`.
    pub dim_value: usize,
}

impl Default for REMConfig {
    fn default() -> Self {
        Self {
            dim_memory: 256,
            dim_key: 32,
            dim_value: 64,
        }
    }
}

/// Key/value table whose prediction is the value of the nearest stored key by cosine similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualEquilibriumMemory {
    pub config: REMConfig,
    items: Vec<(Vec64, Vec64)>,
    /// Running sum of the residual norm observed at each `store`. One residual is recorded per
    /// stored item, so the mean is this sum divided by `items.len()`.
    #[serde(default)]
    residual_sum: f64,
}

impl ResidualEquilibriumMemory {
    pub fn new(config: REMConfig) -> Self {
        Self {
            config,
            items: Vec::new(),
            residual_sum: 0.0,
        }
    }

    /// Restore an exact persisted state without replaying items through `store`.
    ///
    /// `residual_norm` is the persisted *mean* residual norm, so the running sum is rebuilt from
    /// it and the restored item count.
    pub fn from_snapshot(
        config: REMConfig,
        items: Vec<(Vec64, Vec64)>,
        residual_norm: f64,
    ) -> Result<Self> {
        if config.dim_memory == 0 || config.dim_key == 0 || config.dim_value == 0 {
            return Err(MemoryError::InvalidData(
                "memory, key, and value dimensions must be non-zero".to_string(),
            ));
        }
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

        Ok(Self {
            residual_sum: residual_norm * items.len() as f64,
            config,
            items,
        })
    }

    pub fn store(&mut self, key: &Vec64, value: &Vec64) -> Result<()> {
        ensure_dim(key, self.config.dim_key)?;
        ensure_dim(value, self.config.dim_value)?;
        let prediction = self.predict(key);
        self.residual_sum += (value - prediction).norm();
        self.items.push((key.clone(), value.clone()));
        Ok(())
    }

    /// Remove the item at `index`.
    ///
    /// Individual residuals are not recorded — the snapshot format keeps only
    /// their mean, and `from_snapshot` rebuilds the sum from it — so the sum
    /// is scaled to keep that mean unchanged, exactly as a save/load round
    /// trip of the shrunken store would.
    pub fn remove(&mut self, index: usize) -> Result<()> {
        if index >= self.items.len() {
            return Err(MemoryError::InvalidData(format!(
                "remove index {index} is out of range for {} stored items",
                self.items.len()
            )));
        }
        let len = self.items.len() as f64;
        self.items.remove(index);
        self.residual_sum *= (len - 1.0) / len;
        Ok(())
    }

    /// Value of the stored key with the highest cosine similarity to `key`.
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

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn items(&self) -> &[(Vec64, Vec64)] {
        &self.items
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

/// Cosine similarity that treats mismatched or degenerate vectors as "no similarity" rather
/// than panicking, so a hand-edited snapshot cannot abort a prediction scan.
fn cosine(a: &Vec64, b: &Vec64) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
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

    fn two_dimensional() -> REMConfig {
        REMConfig {
            dim_memory: 2,
            dim_key: 2,
            dim_value: 2,
        }
    }

    #[test]
    fn mean_residual_norm_averages_every_store_not_just_the_last() {
        let mut memory = ResidualEquilibriumMemory::new(two_dimensional());
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
        let restored = ResidualEquilibriumMemory::from_snapshot(config, items, 4.0).unwrap();
        assert!((restored.mean_residual_norm() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn snapshot_rejects_non_finite_diagnostics_and_ragged_items() {
        let config = two_dimensional();

        assert!(
            ResidualEquilibriumMemory::from_snapshot(config.clone(), Vec::new(), -1.0).is_err()
        );
        assert!(
            ResidualEquilibriumMemory::from_snapshot(config.clone(), Vec::new(), f64::NAN).is_err()
        );

        let ragged = vec![(
            Vec64::from_row_slice(&[1.0, 0.0, 0.0]),
            Vec64::from_row_slice(&[1.0, 0.0]),
        )];
        assert!(ResidualEquilibriumMemory::from_snapshot(config.clone(), ragged, 0.0).is_err());

        let non_finite = vec![(
            Vec64::from_row_slice(&[1.0, 0.0]),
            Vec64::from_row_slice(&[f64::INFINITY, 0.0]),
        )];
        assert!(ResidualEquilibriumMemory::from_snapshot(config, non_finite, 0.0).is_err());
    }

    #[test]
    fn remove_preserves_the_mean_residual_norm() {
        let mut memory = ResidualEquilibriumMemory::new(two_dimensional());
        memory
            .store(
                &Vec64::from_row_slice(&[1.0, 0.0]),
                &Vec64::from_row_slice(&[3.0, 0.0]),
            )
            .unwrap();
        memory
            .store(
                &Vec64::from_row_slice(&[0.0, 1.0]),
                &Vec64::from_row_slice(&[0.0, 4.0]),
            )
            .unwrap();
        let mean_before = memory.mean_residual_norm();

        memory.remove(0).unwrap();
        assert_eq!(memory.items().len(), 1);
        // Individual residuals are not recorded, so removal keeps the mean —
        // the same answer a save/load round trip of the shrunken store gives.
        assert!((memory.mean_residual_norm() - mean_before).abs() < 1e-12);

        assert!(memory.remove(5).is_err());
    }

    #[test]
    fn prediction_returns_the_nearest_key_value() {
        let mut memory = ResidualEquilibriumMemory::new(two_dimensional());
        memory
            .store(
                &Vec64::from_row_slice(&[1.0, 0.0]),
                &Vec64::from_row_slice(&[7.0, 0.0]),
            )
            .unwrap();
        memory
            .store(
                &Vec64::from_row_slice(&[0.0, 1.0]),
                &Vec64::from_row_slice(&[0.0, 9.0]),
            )
            .unwrap();

        let prediction = memory.predict(&Vec64::from_row_slice(&[0.1, 1.0]));
        assert_eq!(prediction.as_slice(), &[0.0, 9.0]);
    }
}
