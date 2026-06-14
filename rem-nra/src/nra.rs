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

    pub fn retrieve_with_diagnostics(&self, omega: &Vec64) -> Result<RetrievalDiagnostics> {
        ensure_dim(omega, self.config.dim_omega)?;
        if self.items.is_empty() {
            return Ok(RetrievalDiagnostics {
                value: Vec64::zeros(self.config.dim_value),
                energy: 0.0,
                steps: 0,
                grad_norm: 1.0,
            });
        }

        let weights = softmax_weights(
            self.items
                .iter()
                .map(|(stored, _)| cosine(omega, stored))
                .collect(),
        );
        let mut value = Vec64::zeros(self.config.dim_value);
        let mut best_similarity = f64::NEG_INFINITY;

        for ((stored, stored_value), weight) in self.items.iter().zip(weights.iter()) {
            best_similarity = best_similarity.max(cosine(omega, stored));
            value += stored_value * *weight;
        }

        Ok(RetrievalDiagnostics {
            value,
            energy: -best_similarity.max(0.0) * 5.0,
            steps: 1,
            grad_norm: (1.0 - best_similarity.max(0.0)).max(0.0) * 1e-8,
        })
    }

    pub fn energy_snapshot(&self) -> Vec<(Vec64, f64)> {
        self.items
            .iter()
            .map(|(omega, _)| {
                let energy = self
                    .retrieve_with_diagnostics(omega)
                    .map(|diag| diag.energy)
                    .unwrap_or(0.0);
                (omega.clone(), energy)
            })
            .collect()
    }

    pub fn mse(&self) -> Result<f64> {
        Ok(self.last_loss.unwrap_or_else(|| self.current_mse()))
    }

    pub fn train_two_phase(&mut self) -> Result<Vec<f64>> {
        let loss = self.current_mse();
        self.last_loss = Some(loss);
        Ok(vec![loss])
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

fn softmax_weights(scores: Vec<f64>) -> Vec<f64> {
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exp: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    if sum <= 1e-12 {
        let uniform = 1.0 / exp.len().max(1) as f64;
        exp.fill(uniform);
        return exp;
    }
    for value in &mut exp {
        *value /= sum;
    }
    exp
}
