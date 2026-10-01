use lnai_models::GNN_OUTPUT_DIM;

use super::gnn::GnnNorm;
use super::gnn_infer::serving_rng_seed;
use super::rng::SimpleRng;
use super::types::StarFeatures;

pub(crate) fn compute_variational_velocities(
    mean_vals: &[f32],
    logvar_vals: &[f32],
    stars: &[StarFeatures],
    norm: &GnnNorm,
    temperature: f32,
) -> Vec<[f32; 3]> {
    let n = stars.len();
    debug_assert_eq!(mean_vals.len(), n * GNN_OUTPUT_DIM);
    debug_assert_eq!(logvar_vals.len(), n * GNN_OUTPUT_DIM);
    let mut velocities = Vec::with_capacity(n);

    if temperature <= 0.0 {
        for i in 0..n {
            velocities.push([
                mean_vals[i * 3] * norm.vx_std + norm.vx_mean,
                mean_vals[i * 3 + 1] * norm.vy_std + norm.vy_mean,
                mean_vals[i * 3 + 2] * norm.vz_std + norm.vz_mean,
            ]);
        }
        return velocities;
    }

    let mut rng = SimpleRng::new(serving_rng_seed(stars, temperature));

    for i in 0..n {
        let vx_mean = mean_vals[i * 3] * norm.vx_std + norm.vx_mean;
        let vy_mean = mean_vals[i * 3 + 1] * norm.vy_std + norm.vy_mean;
        let vz_mean = mean_vals[i * 3 + 2] * norm.vz_std + norm.vz_mean;

        let vx_logvar = logvar_vals[i * 3] * norm.vx_logvar_std + norm.vx_logvar_mean;
        let vy_logvar = logvar_vals[i * 3 + 1] * norm.vy_logvar_std + norm.vy_logvar_mean;
        let vz_logvar = logvar_vals[i * 3 + 2] * norm.vz_logvar_std + norm.vz_logvar_mean;

        let vx_std = (vx_logvar * 0.5).exp();
        let vy_std = (vy_logvar * 0.5).exp();
        let vz_std = (vz_logvar * 0.5).exp();

        velocities.push([
            vx_mean + vx_std * rng.gaussian() * temperature,
            vy_mean + vy_std * rng.gaussian() * temperature,
            vz_mean + vz_std * rng.gaussian() * temperature,
        ]);
    }
    velocities
}

pub(crate) fn compute_deterministic_velocities(
    mean_vals: &[f32],
    stars: &[StarFeatures],
    norm: &GnnNorm,
    temperature: f32,
) -> Vec<[f32; 3]> {
    let n = stars.len();
    debug_assert_eq!(mean_vals.len(), n * GNN_OUTPUT_DIM);
    let mut velocities = Vec::with_capacity(n);

    if temperature <= 0.0 {
        for i in 0..n {
            velocities.push([
                mean_vals[i * 3] * norm.vx_std + norm.vx_mean,
                mean_vals[i * 3 + 1] * norm.vy_std + norm.vy_mean,
                mean_vals[i * 3 + 2] * norm.vz_std + norm.vz_mean,
            ]);
        }
        return velocities;
    }

    let mut rng = SimpleRng::new(serving_rng_seed(stars, temperature));
    let scale = temperature * 0.15;
    for i in 0..n {
        let vx = mean_vals[i * 3] * norm.vx_std + norm.vx_mean;
        let vy = mean_vals[i * 3 + 1] * norm.vy_std + norm.vy_mean;
        let vz = mean_vals[i * 3 + 2] * norm.vz_std + norm.vz_mean;

        velocities.push([
            vx + rng.gaussian() * norm.vx_std * scale,
            vy + rng.gaussian() * norm.vy_std * scale,
            vz + rng.gaussian() * norm.vz_std * scale,
        ]);
    }
    velocities
}
