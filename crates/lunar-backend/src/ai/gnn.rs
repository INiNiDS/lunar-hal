use burn::prelude::*;
use burn_store::{BurnpackStore, ModuleSnapshot};
use lnai_models::{
    GNN_INPUT_DIM, GNN_OUTPUT_DIM, GNN_VARIATIONAL_DIM, GnnHeadKind, StellarGnn, StellarGnnConfig,
};
use lnai_training::spec::ModelKind;
use lunar_utils::env::get_lunar_models_dir;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::backend_type::B;
use super::serving::check_serving_manifest;

fn default_one() -> f32 {
    1.0
}

#[derive(Deserialize, Clone)]
pub struct GnnNorm {
    pub log_teff_mean: f32,
    pub log_teff_std: f32,
    pub log_rad_mean: f32,
    pub log_rad_std: f32,
    pub log_mass_mean: f32,
    pub log_mass_std: f32,
    pub log_lum_mean: f32,
    pub log_lum_std: f32,
    pub mg_mean: f32,
    pub mg_std: f32,
    pub x_mean: f32,
    pub x_std: f32,
    pub y_mean: f32,
    pub y_std: f32,
    pub z_mean: f32,
    pub z_std: f32,
    pub vx_mean: f32,
    pub vx_std: f32,
    pub vy_mean: f32,
    pub vy_std: f32,
    pub vz_mean: f32,
    pub vz_std: f32,
    #[serde(default)]
    pub vx_logvar_mean: f32,
    #[serde(default = "default_one")]
    pub vx_logvar_std: f32,
    #[serde(default)]
    pub vy_logvar_mean: f32,
    #[serde(default = "default_one")]
    pub vy_logvar_std: f32,
    #[serde(default)]
    pub vz_logvar_mean: f32,
    #[serde(default = "default_one")]
    pub vz_logvar_std: f32,
}

pub struct GnnModel {
    pub model: StellarGnn<B>,
    pub device: Device<B>,
    pub norm: GnnNorm,
    pub variational: bool,
    pub model_hash: String,
    pub norm_hash: String,
}

pub(crate) static GNN: RwLock<Option<Arc<GnnModel>>> = RwLock::const_new(None);

pub async fn get_gnn() -> Option<Arc<GnnModel>> {
    if let Some(cached) = GNN.read().await.clone() {
        return Some(cached);
    }
    let loaded = load_gnn().await;
    *GNN.write().await = loaded.clone();
    loaded
}

pub(crate) async fn load_gnn() -> Option<Arc<GnnModel>> {
    let models_dir = get_lunar_models_dir();
    let norm_path = models_dir.join("stellar_gnn_norm.json");
    let bpk_path = models_dir.join("stellar_gnn_model.bpk");

    if !norm_path.exists() || !bpk_path.exists() {
        println!("  GNN model files not found at: {}", models_dir.display());
        return None;
    }

    let manifest = match check_serving_manifest(&models_dir, &ModelKind::GnnKinematics) {
        Ok(manifest) => {
            println!(
                "  GNN serving manifest verified (arch {}, git {})",
                manifest.architecture_version, manifest.git_revision
            );
            manifest
        }
        Err(err) => {
            eprintln!("  GNN model refused: {err}");
            return None;
        }
    };

    let norm: GnnNorm = match std::fs::read_to_string(&norm_path) {
        Ok(json) => match serde_json::from_str(&json) {
            Ok(n) => n,
            Err(_) => return None,
        },
        Err(_) => return None,
    };

    let device: Device<B> = Default::default();
    let path_str = bpk_path.to_string_lossy();

    // Stage 6: explicit readout head detection
    for width in [GNN_OUTPUT_DIM, GNN_VARIATIONAL_DIM] {
        let head = match GnnHeadKind::from_output_dim(width) {
            Some(head) => head,
            None => continue,
        };
        let mut candidate =
            StellarGnnConfig::new(GNN_INPUT_DIM, 256, head.output_width()).init(&device);
        let mut store = BurnpackStore::from_file(&*path_str);
        if candidate.load_from(&mut store).is_ok() {
            let after_load = check_serving_manifest(&models_dir, &ModelKind::GnnKinematics).ok()?;
            if after_load.model_hash != manifest.model_hash
                || after_load.norm_hash != manifest.norm_hash
            {
                return None;
            }
            return Some(Arc::new(GnnModel {
                model: candidate,
                device,
                norm,
                variational: head == GnnHeadKind::Variational,
                model_hash: manifest.model_hash,
                norm_hash: manifest.norm_hash,
            }));
        }
    }

    None
}
