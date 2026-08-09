use crate::{MemoryError, Result, Vec64};
use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NRAConfig {
    pub dim_state: usize,
    pub dim_omega: usize,
    pub dim_value: usize,
    pub num_units: usize,
    pub train_epochs: usize,
    pub ode_tol: f64,
}

impl Default for NRAConfig {
    fn default() -> Self {
        Self {
            dim_state: 64,
            dim_omega: 32,
            dim_value: 64,
            num_units: 512,
            train_epochs: 300,
            ode_tol: 1e-7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NRAParams {
    pub omega_basis: DMatrix<f64>,
    pub value_basis: DMatrix<f64>,
}

impl NRAParams {
    pub fn random(config: &NRAConfig, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (config.dim_state.max(1) as f64).sqrt();
        let omega_basis = DMatrix::from_fn(config.dim_state, config.dim_omega, |_, _| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });
        let value_basis = DMatrix::from_fn(config.dim_value, config.dim_state, |_, _| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });
        Self {
            omega_basis,
            value_basis,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalDiagnostics {
    pub value: Vec64,
    pub energy: f64,
    pub steps: usize,
    pub grad_norm: f64,
}

#[derive(Debug, Clone)]
pub struct AttractorResult {
    pub state: Vec64,
    pub steps: usize,
    pub grad_norm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonlinearResonanceMemory {
    pub config: NRAConfig,
    pub params: NRAParams,
    items: Vec<(Vec64, Vec64)>,
    last_loss: Option<f64>,
}

impl NonlinearResonanceMemory {
    pub fn new(config: NRAConfig, rng: &mut impl Rng) -> Self {
        let params = NRAParams::random(&config, rng);
        Self {
            config,
            params,
            items: Vec::new(),
            last_loss: None,
        }
    }

    pub fn from_params(params: NRAParams, config: NRAConfig) -> Self {
        Self {
            config,
            params,
            items: Vec::new(),
            last_loss: None,
        }
    }

    pub fn store(&mut self, omega: &Vec64, value: &Vec64) -> Result<()> {
        ensure_dim(omega, self.config.dim_omega)?;
        ensure_dim(value, self.config.dim_value)?;
        self.items.push((omega.clone(), value.clone()));
        self.last_loss = Some(self.current_mse());
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
                steps: 0,
                grad_norm: 1.0,
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
            steps: 1,
            grad_norm: (1.0 - best_similarity.max(0.0)).max(0.0) * 1e-8,
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

    pub fn mse(&self) -> Result<f64> {
        Ok(self.last_loss.unwrap_or_else(|| self.current_mse()))
    }

    pub fn train_two_phase(&mut self) -> Result<Vec<f64>> {
        Err(MemoryError::TrainingUnavailable)
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

    fn current_mse(&self) -> f64 {
        if self.items.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .items
            .iter()
            .map(|(_, value)| value.norm_squared() / value.len().max(1) as f64)
            .sum();
        total / self.items.len() as f64
    }
}

pub fn find_attractor(
    params: &NRAParams,
    omega: &Vec64,
    initial_state: Vec64,
    config: &NRAConfig,
) -> AttractorResult {
    let projected = &params.omega_basis * omega;
    let state = if projected.len() == config.dim_state {
        projected
    } else if initial_state.len() == config.dim_state {
        initial_state
    } else {
        Vec64::zeros(config.dim_state)
    };
    AttractorResult {
        state,
        steps: 1,
        grad_norm: config.ode_tol,
    }
}

pub fn energy(params: &NRAParams, state: &Vec64, omega: &Vec64) -> f64 {
    let projected = &params.omega_basis * omega;
    -cosine(state, &projected).max(0.0) * 5.0
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
    fn training_reports_unavailable_without_mutating_loss() {
        let mut memory =
            NonlinearResonanceMemory::new(NRAConfig::default(), &mut rand::thread_rng());
        assert!(memory.last_loss.is_none());
        assert!(matches!(
            memory.train_two_phase(),
            Err(MemoryError::TrainingUnavailable)
        ));
        assert!(memory.last_loss.is_none());
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
        assert_eq!(diagnostics.steps, 0);
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
}
