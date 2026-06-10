use rand::Rng;
use rand_distr::StandardNormal;
use rem_nra::nra::{energy, find_attractor, NRAConfig, NRAParams};
use rem_nra::Vec64;

/// Basin boundary detection via perturbed ODE starts.
///
/// If perturbed starting points converge to different attractors,
/// the query sits near a basin boundary — this signals ambiguity.
pub struct BasinAnalyzer {
    /// Number of perturbed starting points to test.
    pub num_perturbations: usize,
    /// Scale of perturbation noise.
    pub perturbation_scale: f64,
    /// Minimum distance between attractors to count as distinct.
    pub attractor_distance_threshold: f64,
}

impl Default for BasinAnalyzer {
    fn default() -> Self {
        Self {
            num_perturbations: 8,
            perturbation_scale: 0.5,
            attractor_distance_threshold: 0.5,
        }
    }
}

/// Result of basin analysis.
#[derive(Debug, Clone)]
pub struct BasinResult {
    /// Distinct attractors found from perturbed starts.
    pub attractors: Vec<Vec64>,
    /// Energies at each distinct attractor.
    pub energies: Vec<f64>,
    /// Whether the query is near a basin boundary (ambiguous).
    pub is_ambiguous: bool,
    /// Spread of energies (max - min).
    pub energy_spread: f64,
}

impl BasinAnalyzer {
    /// Analyze the basin structure around an omega query.
    pub fn analyze(
        &self,
        params: &NRAParams,
        omega: &Vec64,
        config: &NRAConfig,
        rng: &mut impl Rng,
    ) -> BasinResult {
        let dim_s = config.dim_state;
        let mut all_attractors = Vec::new();
        let mut all_energies = Vec::new();

        // Start from zero (the canonical start)
        let result_zero = find_attractor(params, omega, Vec64::zeros(dim_s), config);
        let e_zero = energy(params, &result_zero.state, omega);
        all_attractors.push(result_zero.state);
        all_energies.push(e_zero);

        // Perturbed starts
        for _ in 0..self.num_perturbations {
            let s0 = Vec64::from_fn(dim_s, |_, _| {
                rng.sample::<f64, _>(StandardNormal) * self.perturbation_scale
            });
            let result = find_attractor(params, omega, s0, config);
            let e = energy(params, &result.state, omega);

            // Check if this is a new attractor
            let is_new = all_attractors.iter().all(|existing| {
                (&result.state - existing).norm() > self.attractor_distance_threshold
            });

            if is_new {
                all_attractors.push(result.state);
                all_energies.push(e);
            }
        }

        let energy_spread = if all_energies.len() > 1 {
            let min = all_energies.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = all_energies
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            max - min
        } else {
            0.0
        };

        BasinResult {
            is_ambiguous: all_attractors.len() > 1,
            attractors: all_attractors,
            energies: all_energies,
            energy_spread,
        }
    }
}
