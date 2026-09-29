#[cfg(test)]
use super::backend_type::B;
#[cfg(test)]
use super::gnn::{GnnModel, GnnNorm};
#[cfg(test)]
use super::gnn_decode::{compute_deterministic_velocities, compute_variational_velocities};
#[cfg(test)]
use super::gnn_infer::{gnn_infer, serving_rng_seed};
#[cfg(test)]
use super::pinn_queue::{PinnRequest, enqueue_pinn_request};
#[cfg(test)]
use super::types::StarFeatures;
#[cfg(test)]
use burn::prelude::*;
#[cfg(test)]
use lnai_models::{GNN_INPUT_DIM, GNN_OUTPUT_DIM, StellarGnnConfig};

#[test]
fn full_pinn_queue_rejects_work_without_running_fallback() {
    let (queue, _worker) = tokio::sync::mpsc::channel(1);
    for attempt in 0..2 {
        let (responder, _response) = tokio::sync::oneshot::channel();
        let result = enqueue_pinn_request(
            &queue,
            PinnRequest {
                inputs: Vec::new(),
                responder,
            },
        );
        if attempt == 0 {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().to_string().contains("queue is full"));
        }
    }
}

#[test]
fn closed_pinn_queue_reports_unavailable() {
    let (queue, worker) = tokio::sync::mpsc::channel(1);
    drop(worker);
    let (responder, _response) = tokio::sync::oneshot::channel();
    let error = enqueue_pinn_request(
        &queue,
        PinnRequest {
            inputs: Vec::new(),
            responder,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("worker is unavailable"));
}

#[cfg(test)]
fn dummy_gnn() -> GnnModel {
    let device: Device<B> = Default::default();
    let model = StellarGnnConfig::new(GNN_INPUT_DIM, 8, GNN_OUTPUT_DIM).init(&device);
    GnnModel {
        model,
        device,
        norm: GnnNorm {
            log_teff_mean: 0.0,
            log_teff_std: 1.0,
            log_rad_mean: 0.0,
            log_rad_std: 1.0,
            log_mass_mean: 0.0,
            log_mass_std: 1.0,
            log_lum_mean: 0.0,
            log_lum_std: 1.0,
            mg_mean: 0.0,
            mg_std: 1.0,
            x_mean: 0.0,
            x_std: 1.0,
            y_mean: 0.0,
            y_std: 1.0,
            z_mean: 0.0,
            z_std: 1.0,
            vx_mean: 0.0,
            vx_std: 1.0,
            vy_mean: 0.0,
            vy_std: 1.0,
            vz_mean: 0.0,
            vz_std: 1.0,
            vx_logvar_mean: 0.0,
            vx_logvar_std: 1.0,
            vy_logvar_mean: 0.0,
            vy_logvar_std: 1.0,
            vz_logvar_mean: 0.0,
            vz_logvar_std: 1.0,
        },
        variational: false,
        model_hash: String::new(),
        norm_hash: String::new(),
    }
}

#[cfg(test)]
fn dummy_star(x: f32) -> StarFeatures {
    StarFeatures {
        coords: [x, 0.0, 0.0],
        log_teff: 3.75,
        log_rad: 0.0,
        log_mass: 0.0,
        log_lum: 0.0,
        mg: 0.0,
    }
}

#[test]
fn gnn_infer_accepts_empty_and_rejects_single_node() {
    let gnn = dummy_gnn();
    assert!(gnn_infer(&gnn, &[], 1, 0.0).unwrap().is_empty());
    let err = gnn_infer(&gnn, &[dummy_star(0.0)], 1, 0.0).expect_err("single-node must fail");
    assert!(err.to_string().contains(">= 2 stars"), "{err:#}");
}

#[test]
fn serving_seed_is_deterministic_and_coordinate_sensitive() {
    let a = vec![dummy_star(0.0), dummy_star(1.0)];
    let b = vec![dummy_star(0.0), dummy_star(1.0)];
    assert_eq!(serving_rng_seed(&a, 0.7), serving_rng_seed(&b, 0.7));
    assert_eq!(serving_rng_seed(&a, 0.0), serving_rng_seed(&b, 0.0));
    let moved = vec![dummy_star(0.0), dummy_star(2.0)];
    assert_ne!(
        serving_rng_seed(&a, 0.7),
        serving_rng_seed(&moved, 0.7),
        "seed must cover full coordinates, not just x of one star"
    );
    assert_ne!(
        serving_rng_seed(&a, 0.0),
        serving_rng_seed(&a, 0.7),
        "seed must cover temperature"
    );
}

#[test]
fn deterministic_decode_at_zero_temperature_returns_exact_mean() {
    let stars = vec![dummy_star(0.0), dummy_star(1.0)];
    let norm = dummy_gnn().norm;
    // Identity norm: denorm(x) == x, so output must equal the split mean.
    let mean = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let first = compute_deterministic_velocities(&mean, &stars, &norm, 0.0);
    let second = compute_deterministic_velocities(&mean, &stars, &norm, 0.0);
    assert_eq!(first, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(first, second);
}

#[test]
fn stochastic_decode_is_seeded_not_wall_clock() {
    let stars = vec![dummy_star(0.0), dummy_star(1.0)];
    let norm = dummy_gnn().norm;
    let mean = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let logvar = vec![0.0; 6];
    let a = compute_deterministic_velocities(&mean, &stars, &norm, 0.7);
    let b = compute_deterministic_velocities(&mean, &stars, &norm, 0.7);
    assert_eq!(a, b, "same inputs must sample identically");
    let c = compute_variational_velocities(&mean, &logvar, &stars, &norm, 0.7);
    let d = compute_variational_velocities(&mean, &logvar, &stars, &norm, 0.7);
    assert_eq!(c, d, "variational path must be seeded too");
    // Zero temperature disables sampling on both heads.
    assert_eq!(
        compute_variational_velocities(&mean, &logvar, &stars, &norm, 0.0),
        vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    );
}
