#[cfg(feature = "siren")]
use {
    burn::prelude::*,
    burn_store::{BurnpackStore, ModuleSnapshot},
    lnai_models::{SIREN_INPUT_DIM, StellarSiren, StellarSirenConfig},
    lnai_training::spec::ModelKind,
    lunar_utils::env::get_lunar_models_dir,
    serde::Deserialize,
    std::sync::Arc,
    tokio::sync::RwLock,
};

#[cfg(feature = "siren")]
use super::backend_type::B;
#[cfg(feature = "siren")]
use super::serving::check_serving_manifest;

#[cfg(feature = "siren")]
pub(crate) static SIREN_MODEL: RwLock<Option<Arc<SirenModel>>> = RwLock::const_new(None);

#[cfg(feature = "siren")]
#[derive(Deserialize)]
pub struct SirenNorm {
    pub bp_rp_mean: f32,
    pub bp_rp_std: f32,
    pub mg_mean: f32,
    pub mg_std: f32,
    pub log_teff_mean: f32,
    pub log_teff_std: f32,
}

#[cfg(feature = "siren")]
pub struct SirenModel {
    pub model: StellarSiren<B>,
    pub device: Device<B>,
    pub norm: SirenNorm,
    pub model_hash: String,
    pub norm_hash: String,
}

#[cfg(feature = "siren")]
pub async fn get_siren() -> Option<Arc<SirenModel>> {
    if let Some(cached) = SIREN_MODEL.read().await.clone() {
        return Some(cached);
    }
    let loaded = load_siren().await;
    *SIREN_MODEL.write().await = loaded.clone();
    loaded
}

#[cfg(feature = "siren")]
async fn load_siren() -> Option<Arc<SirenModel>> {
    let models_dir = get_lunar_models_dir();
    let norm_path = models_dir.join("stellar_siren_norm.json");
    let bpk_path = models_dir.join("stellar_siren_model.bpk");

    if !norm_path.exists() || !bpk_path.exists() {
        println!(
            "  SIREN model not available (files not found in {})",
            models_dir.display()
        );
        return None;
    }

    let manifest = match check_serving_manifest(&models_dir, &ModelKind::Siren) {
        Ok(manifest) => {
            println!(
                "  SIREN serving manifest verified (arch {}, git {})",
                manifest.architecture_version, manifest.git_revision
            );
            manifest
        }
        Err(err) => {
            eprintln!("  SIREN model refused: {err}");
            return None;
        }
    };

    let norm: SirenNorm = match std::fs::read_to_string(&norm_path) {
        Ok(json) => match serde_json::from_str(&json) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("  Failed to parse SIREN norm: {e}");
                return None;
            }
        },
        Err(_) => return None,
    };

    let device: Device<B> = Default::default();
    let path_str = bpk_path.to_string_lossy();

    let mut model = StellarSirenConfig::new().init(&device);
    let mut store = BurnpackStore::from_file(&*path_str);
    if model.load_from(&mut store).is_err() {
        return None;
    }
    let after_load = check_serving_manifest(&models_dir, &ModelKind::Siren).ok()?;
    if after_load.model_hash != manifest.model_hash || after_load.norm_hash != manifest.norm_hash {
        return None;
    }

    println!(
        "  SIREN model loaded successfully from {}",
        bpk_path.display()
    );
    Some(Arc::new(SirenModel {
        model,
        device,
        norm,
        model_hash: manifest.model_hash,
        norm_hash: manifest.norm_hash,
    }))
}

#[cfg(feature = "siren")]
#[derive(Clone, Copy, Debug)]
pub struct SirenInputs {
    pub uv: [f32; 2],
    pub bp_rp: f32,
    pub m_g: f32,
    pub log_teff: f32,
}

#[cfg(feature = "siren")]
pub fn siren_infer_point(
    model: &StellarSiren<B>,
    device: &Device<B>,
    norm: &SirenNorm,
    inputs: SirenInputs,
) -> [f32; 3] {
    let n_bp = (inputs.bp_rp - norm.bp_rp_mean) / norm.bp_rp_std;
    let n_mg = (inputs.m_g - norm.mg_mean) / norm.mg_std;
    let n_teff = (inputs.log_teff - norm.log_teff_mean) / norm.log_teff_std;

    let input = Tensor::<B, 2>::from_data(
        TensorData::new(
            vec![inputs.uv[0], inputs.uv[1], n_bp, n_mg, n_teff],
            [1, SIREN_INPUT_DIM],
        ),
        device,
    );
    let output = model.forward(input);
    let data = output.into_data();
    let vals: Vec<f32> = data.to_vec().expect("failed to convert SIREN output");

    [
        vals[0].clamp(0.0, 1.0),
        vals[1].clamp(0.0, 1.0),
        vals[2].clamp(0.0, 1.0),
    ]
}
