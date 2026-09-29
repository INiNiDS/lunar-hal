use burn::prelude::*;
use burn_store::{BurnpackStore, ModuleSnapshot};
use lnai_models::{
    GNN_LOC_INPUT_DIM, StellarGnnLocalization, StellarGnnLocalizationConfig,
    density_poisson_baseline,
};
use lnai_training::localization::LocalizationNorm;
use lnai_training::spec::ModelKind;
use lunar_structures::{LocalizationRequest, LocalizationResponse, StarCandidate};
use lunar_utils::env::get_lunar_models_dir;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::backend_type::B;
use super::serving::check_serving_manifest;

pub struct LocalizationModel {
    pub model: StellarGnnLocalization<B>,
    pub device: Device<B>,
    pub norm: LocalizationNorm,
    pub model_hash: String,
    pub norm_hash: String,
}

pub(crate) static GNN_LOCALIZATION: RwLock<Option<Arc<LocalizationModel>>> =
    RwLock::const_new(None);

pub async fn get_localization() -> Option<Arc<LocalizationModel>> {
    if !localization_enabled() {
        return None;
    }
    if let Some(cached) = GNN_LOCALIZATION.read().await.clone() {
        return Some(cached);
    }
    let loaded = load_localization().await;
    *GNN_LOCALIZATION.write().await = loaded.clone();
    loaded
}

pub(crate) fn localization_enabled() -> bool {
    std::env::var("LUNAR_AI_ENABLE_LOCALIZATION").as_deref() == Ok("1")
}

async fn load_localization() -> Option<Arc<LocalizationModel>> {
    let models_dir = get_lunar_models_dir();
    let norm_path = models_dir.join("stellar_gnn_loc_norm.json");
    let bpk_path = models_dir.join("stellar_gnn_loc_model.bpk");

    if !norm_path.exists() || !bpk_path.exists() {
        return None;
    }

    let manifest = match check_serving_manifest(&models_dir, &ModelKind::GnnLocalization) {
        Ok(manifest) => {
            println!(
                "  GNN Localization serving manifest verified (arch {}, git {})",
                manifest.architecture_version, manifest.git_revision
            );
            manifest
        }
        Err(err) => {
            eprintln!("  GNN Localization model refused: {err}");
            return None;
        }
    };

    let norm: LocalizationNorm = match std::fs::read_to_string(&norm_path) {
        Ok(json) => match serde_json::from_str(&json) {
            Ok(n) => n,
            Err(_) => return None,
        },
        Err(_) => return None,
    };

    let device: Device<B> = Default::default();
    let path_str = bpk_path.to_string_lossy();

    let mut candidate = StellarGnnLocalizationConfig::new()
        .with_input_dim(GNN_LOC_INPUT_DIM)
        .with_hidden_dim(128)
        .with_max_slots(norm.max_slots as usize)
        .init(&device);
    let mut store = BurnpackStore::from_file(&*path_str);
    if candidate.load_from(&mut store).is_ok() {
        let after_load = check_serving_manifest(&models_dir, &ModelKind::GnnLocalization).ok()?;
        if after_load.model_hash != manifest.model_hash
            || after_load.norm_hash != manifest.norm_hash
        {
            return None;
        }
        return Some(Arc::new(LocalizationModel {
            model: candidate,
            device,
            norm,
            model_hash: manifest.model_hash,
            norm_hash: manifest.norm_hash,
        }));
    }

    None
}

pub async fn predict_localization_neighbors(req: &LocalizationRequest) -> LocalizationResponse {
    let seed = req.seed.unwrap_or(42);
    let radius_pc = req.radius_pc.max(1.0);
    let max_slots = req.max_slots.unwrap_or(16).clamp(1, 64) as usize;
    let version = req.version.clone().unwrap_or_else(|| "1.0.0".to_string());
    let anchor_pos = [req.anchor_x, req.anchor_y, req.anchor_z];

    if let Some(loc_model) = get_localization().await {
        let mut visible = Vec::new();
        visible.push(lnai_training::localization::LocalStar {
            source_id: "anchor".to_string(),
            ra_deg: 0.0,
            dec_deg: 0.0,
            x: req.anchor_x,
            y: req.anchor_y,
            z: req.anchor_z,
            bp_rp: 1.0,
            g_mag: 15.0,
            ruwe: 1.0,
            is_visible: true,
        });

        for (i, v) in req.visible_neighbors.iter().enumerate() {
            visible.push(lnai_training::localization::LocalStar {
                source_id: v.source_id.clone().unwrap_or_else(|| format!("vis_{i}")),
                ra_deg: 0.0,
                dec_deg: 0.0,
                x: v.x_pc,
                y: v.y_pc,
                z: v.z_pc,
                bp_rp: v.bp_rp.unwrap_or(1.0),
                g_mag: v.g_mag.unwrap_or(15.0),
                ruwe: v.ruwe.unwrap_or(1.0),
                is_visible: true,
            });
        }

        let graph = lnai_training::localization::build_visible_graph_batch(&visible, 6);
        let node_feats = lnai_training::localization::build_visible_node_features(
            anchor_pos, &visible, radius_pc,
        );
        let node_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(node_feats, [visible.len(), GNN_LOC_INPUT_DIM]),
            &loc_model.device,
        );

        let slots = loc_model.model.forward_slots_sparse(node_tensor, &graph, 0);
        let decoded = loc_model.model.decode_candidates(&slots, radius_pc, 0.35);

        let candidates: Vec<StarCandidate> = decoded
            .candidates
            .into_iter()
            .take(max_slots)
            .map(|c| StarCandidate {
                existence_prob: c.existence_prob,
                relative_position: c.relative_position,
                covariance: c.covariance,
                bp_rp: c.bp_rp,
                g_mag: c.g_mag,
            })
            .collect();

        LocalizationResponse {
            candidates,
            anchor_position: anchor_pos,
            radius_pc,
            version,
            seed,
            model_used: "gnn-localization-v1".to_string(),
        }
    } else {
        let density = 0.0004;
        let poisson = density_poisson_baseline(radius_pc, density, seed);
        let candidates: Vec<StarCandidate> = poisson
            .candidates
            .into_iter()
            .take(max_slots)
            .map(|c| StarCandidate {
                existence_prob: c.existence_prob,
                relative_position: c.relative_position,
                covariance: c.covariance,
                bp_rp: c.bp_rp,
                g_mag: c.g_mag,
            })
            .collect();

        LocalizationResponse {
            candidates,
            anchor_position: anchor_pos,
            radius_pc,
            version,
            seed,
            model_used: "baseline-poisson-generator".to_string(),
        }
    }
}
