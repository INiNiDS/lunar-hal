use anyhow::{Context, Result};
use burn::prelude::*;
use burn_store::{BurnpackStore, ModuleSnapshot};
use lnai_models::{StellarMlp, StellarMlpConfig};
use lnai_training::artifacts::verify_manifest_against_files;
use lnai_training::spec::ModelKind;
use lunar_utils::env::get_lunar_models_dir;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::backend_type::B;
use super::serving::check_serving_manifest;
use super::types::{PinnInputs, StellarNorm};

pub struct PinnModel {
    pub model: StellarMlp<B>,
    pub device: Device<B>,
    pub norm: StellarNorm,
    pub model_hash: String,
    pub norm_hash: String,
}

pub(crate) static PINN: RwLock<Option<Arc<PinnModel>>> = RwLock::const_new(None);

pub async fn get_pinn() -> Result<Arc<PinnModel>> {
    if let Some(pinn) = PINN.read().await.clone() {
        return Ok(pinn);
    }
    let mut current = PINN.write().await;
    if let Some(pinn) = current.as_ref() {
        return Ok(pinn.clone());
    }
    let pinn = load_pinn(&get_lunar_models_dir())?;
    *current = Some(pinn.clone());
    Ok(pinn)
}

pub(crate) fn load_pinn(models_dir: &std::path::Path) -> Result<Arc<PinnModel>> {
    let manifest =
        check_serving_manifest(models_dir, &ModelKind::Pinn).map_err(anyhow::Error::msg)?;
    let norm: StellarNorm =
        serde_json::from_slice(&std::fs::read(models_dir.join("stellar_norm.json"))?)?;
    let device: Device<B> = Default::default();
    let mut model = StellarMlpConfig::new().init(&device);
    let path = models_dir.join("stellar_model.bpk");
    let mut store = BurnpackStore::from_file(&*path.to_string_lossy());
    model
        .load_from(&mut store)
        .context("failed to load PINN model from burnpack")?;
    verify_manifest_against_files(models_dir, &manifest)?;
    Ok(Arc::new(PinnModel {
        model,
        device,
        norm,
        model_hash: manifest.model_hash,
        norm_hash: manifest.norm_hash,
    }))
}

pub async fn loaded_pinn_hashes() -> Option<(String, String)> {
    PINN.read()
        .await
        .as_ref()
        .map(|pinn| (pinn.model_hash.clone(), pinn.norm_hash.clone()))
}

pub(crate) fn pinn_input_row(norm: &StellarNorm, inputs: PinnInputs) -> [f32; 5] {
    let [x_pc, y_pc, z_pc] = inputs.position;
    let d_raw = (x_pc * x_pc + y_pc * y_pc + z_pc * z_pc).sqrt();

    let mg = if d_raw < 0.1 {
        4.67
    } else {
        inputs.g_mag - 5.0 * d_raw.log10() + 5.0
    };

    [
        (x_pc - norm.x_mean) / norm.x_std,
        (y_pc - norm.y_mean) / norm.y_std,
        (z_pc - norm.z_mean) / norm.z_std,
        (inputs.bp_rp - norm.bp_rp_mean) / norm.bp_rp_std,
        (mg - norm.mg_mean) / norm.mg_std,
    ]
}

pub(crate) fn denorm_pinn_row(norm: &StellarNorm, vals: [f32; 4]) -> [f32; 4] {
    let log_teff = vals[0] * norm.log_teff_std + norm.log_teff_mean;
    let log_rad = vals[1] * norm.log_rad_std + norm.log_rad_mean;
    let log_mass = vals[2] * norm.log_mass_std + norm.log_mass_mean;
    let log_lum = vals[3] * norm.log_lum_std + norm.log_lum_mean;

    [
        10f32.powf(log_teff),
        10f32.powf(log_rad),
        10f32.powf(log_mass),
        10f32.powf(log_lum),
    ]
}

pub fn pinn_infer(
    model: &StellarMlp<B>,
    device: &Device<B>,
    norm: &StellarNorm,
    inputs: PinnInputs,
) -> [f32; 4] {
    pinn_infer_batch(model, device, norm, &[inputs])
        .into_iter()
        .next()
        .unwrap_or([0.0, 0.0, 0.0, 0.0])
}

/// Stage 6: per-star batched PINN inference — one forward pass for the
/// whole sector instead of one call per star (or one call for the center
/// with random replication for the rest).
pub fn pinn_infer_batch(
    model: &StellarMlp<B>,
    device: &Device<B>,
    norm: &StellarNorm,
    inputs: &[PinnInputs],
) -> Vec<[f32; 4]> {
    if inputs.is_empty() {
        return Vec::new();
    }
    let flat: Vec<f32> = inputs
        .iter()
        .flat_map(|i| pinn_input_row(norm, *i))
        .collect();
    let input = Tensor::<B, 2>::from_data(TensorData::new(flat, [inputs.len(), 5]), device);
    let output = model.forward(input);
    let vals: Vec<f32> = output
        .into_data()
        .to_vec()
        .expect("failed to convert output");

    vals.chunks_exact(4)
        .map(|c| denorm_pinn_row(norm, [c[0], c[1], c[2], c[3]]))
        .collect()
}
