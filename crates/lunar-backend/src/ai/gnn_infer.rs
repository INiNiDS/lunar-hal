use anyhow::Result;
use burn::prelude::*;
use lnai_models::{GNN_INPUT_DIM, GnnHeadKind, split_mean_logvar};

use super::backend_type::B;
use super::gnn::{GnnModel, GnnNorm};
use super::gnn_decode::{compute_deterministic_velocities, compute_variational_velocities};
use super::types::StarFeatures;

pub(crate) fn prepare_node_data(stars: &[StarFeatures], norm: &GnnNorm) -> Vec<f32> {
    let mut node_data = Vec::with_capacity(stars.len() * GNN_INPUT_DIM);
    for star in stars {
        node_data.push((star.log_teff - norm.log_teff_mean) / norm.log_teff_std);
        node_data.push((star.log_rad - norm.log_rad_mean) / norm.log_rad_std);
        node_data.push((star.log_mass - norm.log_mass_mean) / norm.log_mass_std);
        node_data.push((star.log_lum - norm.log_lum_mean) / norm.log_lum_std);
        node_data.push((star.mg - norm.mg_mean) / norm.mg_std);
        node_data.push((star.coords[0] - norm.x_mean) / norm.x_std);
        node_data.push((star.coords[1] - norm.y_mean) / norm.y_std);
        node_data.push((star.coords[2] - norm.z_mean) / norm.z_std);
    }
    node_data
}

pub(crate) fn serving_rng_seed(stars: &[StarFeatures], temperature: f32) -> u64 {
    let mut h: u64 = 0x9E3779B97F4A7C15;
    for s in stars {
        for c in s.coords {
            h ^= c.to_bits() as u64;
            h = h.wrapping_mul(0xBF58476D1CE4E5B9);
        }
    }
    h ^= (temperature.to_bits() as u64).wrapping_mul(0x94D049BB133111EB);
    h ^= (temperature * 1000.0) as u64;
    if h == 0 { 1 } else { h }
}

pub fn gnn_infer(
    gnn: &GnnModel,
    stars: &[StarFeatures],
    knn_k: usize,
    temperature: f32,
) -> Result<Vec<[f32; 3]>> {
    let n = stars.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if n < 2 {
        anyhow::bail!(
            "gnn_infer requires a group of >= 2 stars, got {n}: single-node GNN is excluded from production (Stage 6.6)"
        );
    }

    let norm = &gnn.norm;
    let knn_k = knn_k.max(1).min(n);

    let coords: Vec<[f32; 3]> = stars.iter().map(|s| s.coords).collect();
    let graph = lnai_training::gnn::GraphCache::get_or_build(&coords, knn_k);
    let node_data = prepare_node_data(stars, norm);
    let nodes =
        Tensor::<B, 2>::from_data(TensorData::new(node_data, [n, GNN_INPUT_DIM]), &gnn.device);

    let output = gnn.model.forward_sparse(nodes, &graph);
    let head = if gnn.variational {
        GnnHeadKind::Variational
    } else {
        GnnHeadKind::Deterministic
    };
    let (mean_t, logvar_t) = split_mean_logvar(output, head);
    let mean_vals: Vec<f32> = mean_t
        .into_data()
        .to_vec()
        .expect("failed to convert GNN mean output");

    if gnn.variational {
        let logvar_vals: Vec<f32> = logvar_t
            .expect("variational head must carry logvar")
            .into_data()
            .to_vec()
            .expect("failed to convert GNN logvar output");
        Ok(compute_variational_velocities(
            &mean_vals,
            &logvar_vals,
            stars,
            norm,
            temperature,
        ))
    } else {
        Ok(compute_deterministic_velocities(
            &mean_vals,
            stars,
            norm,
            temperature,
        ))
    }
}
