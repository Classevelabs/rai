use crate::{MemoryError, Result, Vec64};
use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NRAConfig {
    /// Row count of the persisted `omega_basis` matrix.
    pub dim_state: usize,
    /// Dimension of the address vectors accepted by `store` and retrieval.
    pub dim_omega: usize,
    /// Dimension of the value vectors accepted by `store` and returned by retrieval.
    pub dim_value: usize,
    /// Maximum number of items the enclosing store is willing to hold.
    pub num_units: usize,
}

impl Default for NRAConfig {
    fn default() -> Self {
        Self {
            dim_state: 64,
            dim_omega: 32,
            dim_value: 64,
            num_units: 512,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NRAParams {
    /// Fixed random basis carried with the store so a reloaded snapshot keeps its identity.
    /// Retrieval is a similarity scan over stored addresses and does not consult this matrix.
    pub omega_basis: DMatrix<f64>,
}

impl NRAParams {
    pub fn random(config: &NRAConfig, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (config.dim_state.max(1) as f64).sqrt();
        let omega_basis = DMatrix::from_fn(config.dim_state, config.dim_omega, |_, _| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });
        Self { omega_basis }
    }
}

/// Outcome of a retrieval scan.
#[derive(Debug, Clone)]
pub struct RetrievalDiagnostics {
    /// Value stored alongside the best-matching address.
    pub value: Vec64,
    /// `-5 · max(cosine, 0)` for the best-matching address: `0.0` when nothing scored above
    /// zero similarity, `-5.0` for an exact match.
    pub energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonlinearResonanceMemory {
    pub config: NRAConfig,
    pub params: NRAParams,
    items: Vec<(Vec64, Vec64)>,
}

impl NonlinearResonanceMemory {
    pub fn new(config: NRAConfig, rng: &mut impl Rng) -> Self {
        let params = NRAParams::random(&config, rng);
        Self {
            config,
            params,
            items: Vec::new(),
        }
    }

    /// Restore a persisted store, rejecting any shape the retrieval scan could not handle.
    ///
    /// Deserialized state is untrusted: a ragged item list would otherwise reach the similarity
    /// scan with mismatched dimensions.
    pub fn from_snapshot(
        config: NRAConfig,
        params: NRAParams,
        items: Vec<(Vec64, Vec64)>,
    ) -> Result<Self> {
        if config.dim_state == 0 || config.dim_omega == 0 || config.dim_value == 0 {
            return Err(MemoryError::InvalidData(
                "state, address, and value dimensions must be non-zero".to_string(),
            ));
        }
        if config.num_units == 0 || items.len() > config.num_units {
            return Err(MemoryError::InvalidData(
                "stored item count exceeds the configured capacity".to_string(),
            ));
        }
        if params.omega_basis.nrows() != config.dim_state
            || params.omega_basis.ncols() != config.dim_omega
        {
            return Err(MemoryError::InvalidData(
                "omega basis shape does not match the configuration".to_string(),
            ));
        }
        if !params.omega_basis.iter().all(|value| value.is_finite()) {
            return Err(MemoryError::InvalidData(
                "omega basis must be finite".to_string(),
            ));
        }
        for (omega, value) in &items {
            ensure_dim(omega, config.dim_omega)?;
            ensure_dim(value, config.dim_value)?;
            ensure_finite(omega, "stored address")?;
            ensure_finite(value, "stored value")?;
        }

        Ok(Self {
            config,
            params,
            items,
        })
    }

    pub fn store(&mut self, omega: &Vec64, value: &Vec64) -> Result<()> {
        ensure_dim(omega, self.config.dim_omega)?;
        ensure_dim(value, self.config.dim_value)?;
        self.items.push((omega.clone(), value.clone()));
        Ok(())
    }

    /// Return the value of the single best-matching stored item.
    ///
    /// Retrieval is nearest-neighbour by cosine similarity over the stored addresses. Blending
    /// every stored value together would return the corpus centroid rather than the match, so
    /// the best-scoring item wins outright.
    pub fn retrieve_with_diagnostics(&self, omega: &Vec64) -> Result<RetrievalDiagnostics> {
        ensure_dim(omega, self.config.dim_omega)?;
        let Some((_, first_value)) = self.items.first() else {
            return Ok(RetrievalDiagnostics {
                value: Vec64::zeros(self.config.dim_value),
                energy: 0.0,
            });
        };

        let mut best_value = first_value;
        let mut best_similarity = f64::NEG_INFINITY;
        for (stored, stored_value) in &self.items {
            let similarity = cosine(omega, stored);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_value = stored_value;
            }
        }

        Ok(RetrievalDiagnostics {
            value: best_value.clone(),
            energy: -best_similarity.max(0.0) * 5.0,
        })
    }

    /// Score every stored address against its neighbours, leaving the item itself out.
    ///
    /// Including the item would score `cos(w, w) == 1` for every entry and report the same
    /// constant energy forever, which hides all crowding between stored addresses.
    pub fn energy_snapshot(&self) -> Vec<(Vec64, f64)> {
        self.items
            .iter()
            .enumerate()
            .map(|(index, (omega, _))| {
                let best_similarity = self
                    .items
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != index)
                    .map(|(_, (stored, _))| cosine(omega, stored))
                    .fold(0.0f64, f64::max);
                (omega.clone(), -best_similarity * 5.0)
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.items.len()
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
/// than panicking, so a hand-edited snapshot cannot abort a retrieval scan.
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

    fn unit_config() -> NRAConfig {
        NRAConfig {
            dim_state: 2,
            dim_omega: 2,
            dim_value: 2,
            ..Default::default()
        }
    }

    fn memory_with(items: &[([f64; 2], [f64; 2])]) -> NonlinearResonanceMemory {
        let mut memory = NonlinearResonanceMemory::new(unit_config(), &mut rand::thread_rng());
        for (omega, value) in items {
            memory
                .store(&Vec64::from_row_slice(omega), &Vec64::from_row_slice(value))
                .expect("store");
        }
        memory
    }

    #[test]
    fn retrieval_returns_the_best_match_not_a_corpus_blend() {
        // Two stored values that would average to roughly (0.5, 0.5) if blended.
        let memory = memory_with(&[([1.0, 0.0], [1.0, 0.0]), ([0.0, 1.0], [0.0, 1.0])]);

        let matched = memory
            .retrieve_with_diagnostics(&Vec64::from_row_slice(&[1.0, 0.05]))
            .expect("retrieve");
        assert_eq!(matched.value.as_slice(), &[1.0, 0.0]);
        assert!(matched.energy < -4.9, "energy: {}", matched.energy);

        let other = memory
            .retrieve_with_diagnostics(&Vec64::from_row_slice(&[0.05, 1.0]))
            .expect("retrieve");
        assert_eq!(other.value.as_slice(), &[0.0, 1.0]);
    }

    #[test]
    fn retrieval_on_an_empty_memory_reports_no_match() {
        let memory = memory_with(&[]);
        let diagnostics = memory
            .retrieve_with_diagnostics(&Vec64::from_row_slice(&[1.0, 0.0]))
            .expect("retrieve");
        assert_eq!(diagnostics.value.as_slice(), &[0.0, 0.0]);
        assert_eq!(diagnostics.energy, 0.0);
    }

    #[test]
    fn energy_snapshot_leaves_the_scored_item_out() {
        // A singleton has no neighbour to resonate with, so its energy is zero rather than
        // the constant self-similarity floor.
        let singleton = memory_with(&[([1.0, 0.0], [1.0, 0.0])]);
        assert_eq!(singleton.energy_snapshot()[0].1, 0.0);

        // A near-duplicate pair plus a distant item must not all report the same energy.
        let crowded = memory_with(&[
            ([1.0, 0.0], [1.0, 0.0]),
            ([1.0, 0.02], [1.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0]),
        ]);
        let energies: Vec<f64> = crowded.energy_snapshot().iter().map(|(_, e)| *e).collect();
        assert!(energies[0] < -4.9, "crowded neighbour energy: {energies:?}");
        assert!(energies[1] < -4.9, "crowded neighbour energy: {energies:?}");
        assert!(energies[2] > -1.0, "distant item energy: {energies:?}");
        assert!(energies.iter().any(|energy| *energy != energies[0]));
    }

    #[test]
    fn energy_snapshot_ignores_negative_similarity() {
        // Opposed addresses have cosine -1; energy is clamped at zero rather than going positive.
        let opposed = memory_with(&[([1.0, 0.0], [1.0, 0.0]), ([-1.0, 0.0], [0.0, 1.0])]);
        for (_, energy) in opposed.energy_snapshot() {
            assert_eq!(energy, 0.0);
        }
    }

    #[test]
    fn snapshot_restore_rejects_shapes_the_similarity_scan_cannot_handle() {
        let config = unit_config();
        let params = NRAParams::random(&config, &mut rand::thread_rng());
        let good = vec![(
            Vec64::from_row_slice(&[1.0, 0.0]),
            Vec64::from_row_slice(&[1.0, 0.0]),
        )];
        assert!(
            NonlinearResonanceMemory::from_snapshot(config.clone(), params.clone(), good).is_ok()
        );

        // A ragged address would reach the similarity scan with the wrong dimension.
        let ragged = vec![(
            Vec64::from_row_slice(&[1.0, 0.0, 0.0]),
            Vec64::from_row_slice(&[1.0, 0.0]),
        )];
        assert!(
            NonlinearResonanceMemory::from_snapshot(config.clone(), params.clone(), ragged)
                .is_err()
        );

        let non_finite = vec![(
            Vec64::from_row_slice(&[f64::NAN, 0.0]),
            Vec64::from_row_slice(&[1.0, 0.0]),
        )];
        assert!(
            NonlinearResonanceMemory::from_snapshot(config.clone(), params, non_finite).is_err()
        );

        // A basis whose shape disagrees with the configuration is rejected too.
        let wrong_basis = NRAParams {
            omega_basis: DMatrix::zeros(config.dim_state + 1, config.dim_omega),
        };
        assert!(NonlinearResonanceMemory::from_snapshot(config, wrong_basis, Vec::new()).is_err());
    }

    #[test]
    fn mismatched_dimensions_score_as_no_similarity_instead_of_panicking() {
        let short = Vec64::from_row_slice(&[1.0, 0.0]);
        let long = Vec64::from_row_slice(&[1.0, 0.0, 0.0]);
        assert_eq!(cosine(&short, &long), 0.0);
    }
}
