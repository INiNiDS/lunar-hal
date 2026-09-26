use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn_store::{BurnpackStore, ModuleSnapshot};
use lnai_models::{
    GNN_INPUT_DIM, GNN_LOC_INPUT_DIM, GNN_OUTPUT_DIM, GNN_VARIATIONAL_DIM, GnnHeadKind, StellarGnn,
    StellarGnnConfig, StellarGnnLocalization, StellarGnnLocalizationConfig, StellarMlp,
    StellarMlpConfig, density_poisson_baseline, split_mean_logvar,
};
#[cfg(feature = "siren")]
use lnai_models::{SIREN_INPUT_DIM, StellarSiren, StellarSirenConfig};
use lnai_training::artifacts::{
    RegisteredArtifact, RegistryStatus, manifest_file_name, scan_model_registry,
    verify_manifest_against_files,
};
use lnai_training::localization::LocalizationNorm;
use lnai_training::spec::ModelKind;
use lunar_structures::{
    LocalizationRequest, LocalizationResponse, LocalizedStarWithPhysics, PinnResponse,
    PipelineLocalizePhysicsRequest, PipelineLocalizePhysicsResponse, StarCandidate,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{OnceCell, RwLock};

use lunar_utils::env::get_lunar_models_dir;

#[cfg(feature = "wgpu")]
type B = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "cuda"))]
type B = burn::backend::Cuda;
#[cfg(all(not(feature = "wgpu"), not(feature = "cuda"), feature = "metal"))]
type B = burn::backend::Metal;
#[cfg(all(
    not(feature = "wgpu"),
    not(feature = "cuda"),
    not(feature = "metal"),
    feature = "rocm"
))]
type B = burn::backend::Rocm;

#[derive(Deserialize)]
pub struct StellarNorm {
    pub x_mean: f32,
    pub x_std: f32,
    pub y_mean: f32,
    pub y_std: f32,
    pub z_mean: f32,
    pub z_std: f32,
    pub bp_rp_mean: f32,
    pub bp_rp_std: f32,
    pub mg_mean: f32,
    pub mg_std: f32,
    pub log_teff_mean: f32,
    pub log_teff_std: f32,
    pub log_rad_mean: f32,
    pub log_rad_std: f32,
    pub log_mass_mean: f32,
    pub log_mass_std: f32,
    pub log_lum_mean: f32,
    pub log_lum_std: f32,
}

pub struct PinnModel {
    pub model: StellarMlp<B>,
    pub device: Device<B>,
    pub norm: StellarNorm,
    pub model_hash: String,
    pub norm_hash: String,
}

static PINN: RwLock<Option<Arc<PinnModel>>> = RwLock::const_new(None);

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

fn load_pinn(models_dir: &std::path::Path) -> Result<Arc<PinnModel>> {
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

#[derive(Clone, Serialize)]
pub struct LoadedModelIdentity {
    pub kind: &'static str,
    pub model_hash: String,
    pub norm_hash: String,
}

pub async fn loaded_model_identities() -> Vec<LoadedModelIdentity> {
    let mut active = Vec::new();
    if let Some((model_hash, norm_hash)) = loaded_pinn_hashes().await {
        active.push(LoadedModelIdentity {
            kind: "pinn",
            model_hash,
            norm_hash,
        });
    }
    if let Some(gnn) = GNN.read().await.as_ref() {
        active.push(LoadedModelIdentity {
            kind: "gnn_kinematics",
            model_hash: gnn.model_hash.clone(),
            norm_hash: gnn.norm_hash.clone(),
        });
    }
    if localization_enabled() {
        if let Some(loc) = GNN_LOCALIZATION.read().await.as_ref() {
            active.push(LoadedModelIdentity {
                kind: "gnn_localization",
                model_hash: loc.model_hash.clone(),
                norm_hash: loc.norm_hash.clone(),
            });
        }
    }
    #[cfg(feature = "siren")]
    if let Some(siren) = SIREN_MODEL.read().await.as_ref() {
        active.push(LoadedModelIdentity {
            kind: "siren",
            model_hash: siren.model_hash.clone(),
            norm_hash: siren.norm_hash.clone(),
        });
    }
    active
}

#[derive(Clone, Copy, Debug)]
pub struct PinnInputs {
    pub position: [f32; 3],
    pub bp_rp: f32,
    pub g_mag: f32,
}

/// Apparent G magnitude of a member at `position` assuming it shares the
/// sector's absolute magnitude `mg_center` (distance modulus, parsecs).
/// Falls back to `mg_center + 5*log10(0.1) - 5` near the origin, mirroring
/// the single-star guard below.
pub fn apparent_g_for_member(position: [f32; 3], mg_center: f32, g_fallback: f32) -> f32 {
    let d =
        (position[0] * position[0] + position[1] * position[1] + position[2] * position[2]).sqrt();
    if d < 0.1 {
        g_fallback
    } else {
        mg_center + 5.0 * d.log10() - 5.0
    }
}

fn pinn_input_row(norm: &StellarNorm, inputs: PinnInputs) -> [f32; 5] {
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

fn denorm_pinn_row(norm: &StellarNorm, vals: [f32; 4]) -> [f32; 4] {
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

/// Stage 7: Bounded worker per device with backpressure and short dynamic batching window.
struct PinnRequest {
    inputs: Vec<PinnInputs>,
    responder: tokio::sync::oneshot::Sender<Result<Vec<[f32; 4]>, String>>,
}

static PINN_QUEUE: OnceCell<tokio::sync::mpsc::Sender<PinnRequest>> = OnceCell::const_new();

const PINN_QUEUE_CAPACITY: usize = 1024;
const DYNAMIC_BATCH_WINDOW: std::time::Duration = std::time::Duration::from_millis(5);
const MAX_DYNAMIC_BATCH_SIZE: usize = 256;

async fn get_pinn_queue() -> tokio::sync::mpsc::Sender<PinnRequest> {
    PINN_QUEUE
        .get_or_init(|| async {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<PinnRequest>(PINN_QUEUE_CAPACITY);
            tokio::spawn(async move {
                let mut pending = None;
                while let Some(first_req) = match pending.take() {
                    Some(req) => Some(req),
                    None => rx.recv().await,
                } {
                    let mut batch_inputs = first_req.inputs;
                    let mut responders = vec![(batch_inputs.len(), first_req.responder)];

                    // Dynamic batching window: collect pending requests up to window or max size
                    let deadline = tokio::time::Instant::now() + DYNAMIC_BATCH_WINDOW;
                    while batch_inputs.len() < MAX_DYNAMIC_BATCH_SIZE {
                        let timeout =
                            deadline.saturating_duration_since(tokio::time::Instant::now());
                        if timeout.is_zero() {
                            break;
                        }
                        tokio::select! {
                            biased;
                            Some(req) = rx.recv() => {
                                let count = req.inputs.len();
                                if batch_inputs.len() + count > MAX_DYNAMIC_BATCH_SIZE {
                                    pending = Some(req);
                                    break;
                                }
                                batch_inputs.extend(req.inputs);
                                responders.push((count, req.responder));
                                if batch_inputs.len() >= MAX_DYNAMIC_BATCH_SIZE {
                                    break;
                                }
                            }
                            _ = tokio::time::sleep(timeout) => {
                                break;
                            }
                        }
                    }

                    // Execute single batched forward pass on device
                    let pinn_ref = match get_pinn().await {
                        Ok(pinn) => pinn,
                        Err(err) => {
                            for (_, responder) in responders {
                                let _ =
                                    responder.send(Err(format!("PINN model unavailable: {err:#}")));
                            }
                            continue;
                        }
                    };
                    let outputs = tokio::task::spawn_blocking(move || {
                        pinn_infer_batch(
                            &pinn_ref.model,
                            &pinn_ref.device,
                            &pinn_ref.norm,
                            &batch_inputs,
                        )
                    })
                    .await
                    .map_err(|err| format!("PINN worker failed: {err}"))
                    .and_then(|rows| {
                        if rows.len() != responders.iter().map(|(count, _)| count).sum::<usize>() {
                            Err("PINN worker returned an incomplete batch".to_string())
                        } else {
                            Ok(rows)
                        }
                    });

                    // Distribute outputs back to each responder
                    let mut offset = 0;
                    for (count, responder) in responders {
                        let slice = outputs
                            .as_ref()
                            .map(|rows| rows[offset..offset + count].to_vec())
                            .map_err(Clone::clone);
                        offset += count;
                        let _ = responder.send(slice);
                    }
                }
            });
            tx
        })
        .await
        .clone()
}

pub async fn infer_pinn_batch_async(inputs: Vec<PinnInputs>) -> Result<Vec<[f32; 4]>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if inputs.len() > MAX_DYNAMIC_BATCH_SIZE {
        bail!("PINN request exceeds the maximum batch size of {MAX_DYNAMIC_BATCH_SIZE}");
    }
    let queue = get_pinn_queue().await;
    let (tx, rx) = tokio::sync::oneshot::channel();
    enqueue_pinn_request(
        &queue,
        PinnRequest {
            inputs,
            responder: tx,
        },
    )?;
    rx.await
        .context("PINN worker stopped without responding")?
        .map_err(anyhow::Error::msg)
}

fn enqueue_pinn_request(
    queue: &tokio::sync::mpsc::Sender<PinnRequest>,
    request: PinnRequest,
) -> Result<()> {
    queue.try_send(request).map_err(|err| match err {
        tokio::sync::mpsc::error::TrySendError::Full(_) => {
            anyhow::anyhow!("PINN queue is full; retry the request later")
        }
        tokio::sync::mpsc::error::TrySendError::Closed(_) => {
            anyhow::anyhow!("PINN worker is unavailable")
        }
    })
}

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

static GNN: RwLock<Option<Arc<GnnModel>>> = RwLock::const_new(None);

pub async fn get_gnn() -> Option<Arc<GnnModel>> {
    if let Some(cached) = GNN.read().await.clone() {
        return Some(cached);
    }
    let loaded = load_gnn().await;
    *GNN.write().await = loaded.clone();
    loaded
}

/// Serving gate: a bundle needs a matching manifest, intact files, complete
/// provenance, and a passing spatial-holdout evaluation.
fn check_serving_manifest(
    models_dir: &std::path::Path,
    kind: &ModelKind,
) -> Result<lnai_training::artifacts::ArtifactManifestV1, String> {
    let path = models_dir.join(manifest_file_name(kind));
    if !path.exists() {
        return Err(format!(
            "serving bundle for {} has no artifact manifest",
            path.display()
        ));
    }
    let raw =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let manifest: lnai_training::artifacts::ArtifactManifestV1 =
        serde_json::from_str(&raw).map_err(|e| format!("parse {}: {e}", path.display()))?;
    if &manifest.model_kind != kind {
        return Err(format!(
            "serving bundle for {} declares model kind {:?}, expected {:?}",
            path.display(),
            manifest.model_kind,
            kind
        ));
    }
    verify_manifest_against_files(models_dir, &manifest)
        .map_err(|e| format!("serving bundle for {} rejected: {e}", path.display()))?;
    let blockers = manifest.release_blockers();
    if !blockers.is_empty() {
        return Err(format!(
            "serving bundle for {} is not release-approved; missing evidence: {}",
            path.display(),
            blockers.join(", ")
        ));
    }
    Ok(manifest)
}

async fn load_gnn() -> Option<Arc<GnnModel>> {
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

    // Stage 6: the readout head is explicit — try each contracted width
    // in order and record which head the artifact carries, instead of
    // sniffing shapes ad hoc.
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

pub struct LocalizationModel {
    pub model: StellarGnnLocalization<B>,
    pub device: Device<B>,
    pub norm: LocalizationNorm,
    pub model_hash: String,
    pub norm_hash: String,
}

static GNN_LOCALIZATION: RwLock<Option<Arc<LocalizationModel>>> = RwLock::const_new(None);

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

fn localization_enabled() -> bool {
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

pub async fn predict_localize_physics_pipeline(
    req: &PipelineLocalizePhysicsRequest,
) -> Result<PipelineLocalizePhysicsResponse> {
    let loc_req = LocalizationRequest {
        anchor_x: req.anchor_x,
        anchor_y: req.anchor_y,
        anchor_z: req.anchor_z,
        radius_pc: req.radius_pc,
        max_slots: req.max_slots,
        seed: req.seed,
        version: req.version.clone(),
        visible_neighbors: req.visible_neighbors.clone(),
    };
    let loc_res = predict_localization_neighbors(&loc_req).await;

    let mut pinn_inputs = Vec::with_capacity(loc_res.candidates.len());
    let mut abs_positions = Vec::with_capacity(loc_res.candidates.len());

    for cand in &loc_res.candidates {
        let abs_pos = [
            req.anchor_x + cand.relative_position[0],
            req.anchor_y + cand.relative_position[1],
            req.anchor_z + cand.relative_position[2],
        ];
        abs_positions.push(abs_pos);
        pinn_inputs.push(PinnInputs {
            position: abs_pos,
            bp_rp: cand.bp_rp.unwrap_or(0.8),
            g_mag: cand.g_mag.unwrap_or(15.0),
        });
    }

    let pinn_outputs = infer_pinn_batch_async(pinn_inputs).await?;

    let mut stars = Vec::with_capacity(loc_res.candidates.len());
    for (i, cand) in loc_res.candidates.into_iter().enumerate() {
        let row = pinn_outputs[i];
        let physics = Some(PinnResponse {
            temperature_k: row[0],
            radius_solar: row[1],
            mass_solar: row[2],
            luminosity_solar: row[3],
        });
        stars.push(LocalizedStarWithPhysics {
            candidate: cand,
            absolute_position: abs_positions[i],
            physics,
        });
    }

    Ok(PipelineLocalizePhysicsResponse {
        stars,
        anchor_position: [req.anchor_x, req.anchor_y, req.anchor_z],
        radius_pc: req.radius_pc,
        version: loc_res.version,
        seed: loc_res.seed,
    })
}

#[derive(Clone)]
pub struct StarFeatures {
    pub coords: [f32; 3],
    pub log_teff: f32,
    pub log_rad: f32,
    pub log_mass: f32,
    pub log_lum: f32,
    pub mg: f32,
}

fn prepare_node_data(stars: &[StarFeatures], norm: &GnnNorm) -> Vec<f32> {
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

/// Stage 6: deterministic serving seed from the full coordinates plus
/// the temperature bits. No wall-clock participates: the same
/// `(stars, temperature)` always yields the same velocities, so the
/// `temperature <= 0` path is exactly reproducible and the `> 0` path is
/// seeded (unlike the pre-fix `SystemTime`-seeded deterministic branch).
fn serving_rng_seed(stars: &[StarFeatures], temperature: f32) -> u64 {
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

/// Variational decoding over the already-split `[N, 3]` mean / `[N, 3]`
/// logvar slices (layout comes from [`split_mean_logvar` shared with
/// train/eval, never re-sliced here).
fn compute_variational_velocities(
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

/// Deterministic decoding over the already-split `[N, 3]` mean slice.
/// `temperature <= 0` returns the exact mean (no RNG consumed).
/// `temperature > 0` adds seeded unit-variance noise scaled by
/// `temperature * 0.15`: the 0.15 keeps UI-level entropy from blowing up
/// physical dispersions (train samples `eps * temperature` in normalized
/// space; serving denormalizes, hence the damped serving scale).
fn compute_deterministic_velocities(
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

/// Stage 6: batched GNN-Kinematics inference over a star group.
/// Single-node (`n < 2`) inference is rejected: the model never trains on
/// singletons (dataset drops them, loss panics on them), so a self-loop
/// forward would be unvalidated physics. Callers fall back to zero
/// velocity explicitly instead.
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
    // Stage 7: sparse GNN inference over cached immutable graph.
    // Avoids allocating and transferring dense N×N adjacency matrices.
    let graph = lnai_training::gnn::GraphCache::get_or_build(&coords, knn_k);

    let node_data = prepare_node_data(stars, norm);

    let nodes =
        Tensor::<B, 2>::from_data(TensorData::new(node_data, [n, GNN_INPUT_DIM]), &gnn.device);

    let output = gnn.model.forward_sparse(nodes, &graph);
    // Stage 6: the readout layout is decoded through the shared
    // `split_mean_logvar` contract (same function as train/eval), so
    // serving can never disagree on mean-vs-logvar column order.
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

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        if s == 0 {
            s = 1;
        }
        Self { state: s }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

pub struct RandomStellarInputs {
    pub x_pc: f32,
    pub y_pc: f32,
    pub z_pc: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
}

pub fn generate_random_inputs(entropy: f32, norm: &StellarNorm) -> RandomStellarInputs {
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let base = (entropy * 1000.0) as u64 ^ epoch;
    let seed = if entropy > 0.5 {
        base.wrapping_mul(6364136223846793005)
    } else {
        base
    };

    let mut rng = SimpleRng::new(seed);

    let x_pc = norm.x_mean + norm.x_std * rng.gaussian();
    let y_pc = norm.y_mean + norm.y_std * rng.gaussian();
    let z_pc = norm.z_mean + norm.z_std * rng.gaussian();

    let bp_rp = (norm.bp_rp_mean + norm.bp_rp_std * rng.gaussian()).clamp(-0.5, 5.0);

    let mg = norm.mg_mean + norm.mg_std * rng.gaussian();
    let d = (x_pc * x_pc + y_pc * y_pc + z_pc * z_pc).sqrt().max(0.1);
    let g_mag = (mg + 5.0 * d.log10() - 5.0).clamp(0.0, 20.0);

    RandomStellarInputs {
        x_pc,
        y_pc,
        z_pc,
        bp_rp,
        g_mag,
    }
}

pub fn classify_star(teff: f32, rad: f32) -> (String, String) {
    let spectral_class = if teff >= 30_000.0 {
        "O".to_string()
    } else if teff >= 10_000.0 {
        "B".to_string()
    } else if teff >= 7_500.0 {
        "A".to_string()
    } else if teff >= 6_000.0 {
        "F".to_string()
    } else if teff >= 5_200.0 {
        "G".to_string()
    } else if teff >= 3_700.0 {
        "K".to_string()
    } else {
        "M".to_string()
    };

    let category = if rad >= 100.0 {
        "Hypergiant".to_string()
    } else if rad >= 10.0 {
        "Supergiant".to_string()
    } else if rad >= 3.0 {
        "Giant".to_string()
    } else if rad >= 1.5 {
        "Subgiant".to_string()
    } else if rad >= 0.8 {
        "Main Sequence".to_string()
    } else {
        "Dwarf".to_string()
    };

    (spectral_class, category)
}

fn compute_stellar_seed(teff: f32, rad: f32, mass: f32, entropy: f32) -> u64 {
    let base = ((teff * 100.0) as u64) ^ ((rad * 1000.0) as u64) ^ ((mass * 100.0) as u64);
    if entropy > 0.5 {
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        base ^ epoch.wrapping_mul((entropy * 1000.0) as u64)
    } else {
        base
    }
}

pub fn generate_stochastic_metadata(
    teff: f32,
    rad: f32,
    mass: f32,
    _lum: f32,
    entropy_temperature: f32,
) -> lunar_structures::StellarMetadata {
    let (spectral_class, category) = classify_star(teff, rad);
    let final_seed = compute_stellar_seed(teff, rad, mass, entropy_temperature);
    let mut rng = SimpleRng::new(final_seed);

    let designated_name = generate_name(&mut rng, entropy_temperature);
    let description = generate_description(
        &mut rng,
        &spectral_class,
        &category,
        entropy_temperature,
        teff,
        rad,
    );

    lunar_structures::StellarMetadata {
        spectral_class,
        category,
        designated_name,
        description,
    }
}

fn generate_name(rng: &mut SimpleRng, entropy: f32) -> String {
    let catalog_prefixes = [
        "UVS",
        "AX",
        "KX",
        "ZQ",
        "HD",
        "TYC",
        "GSC",
        "BD",
        "LP",
        "LHS",
        "Wolf",
        "Ross",
        "Gliese",
        "Kepler",
        "TrES",
        "XO",
        "HAT-P",
        "WASP",
        "K2",
        "TOI",
        "LTT",
        "GJ",
        "HIP",
        "SAO",
        "NGC",
        "IC",
        "Melotte",
        "Collinder",
        "Trumpler",
    ];

    let greek = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ"];

    let name_prefixes = [
        "Aethel", "Belis", "Cygnia", "Draconis", "Eshana", "Ferox", "Glyph", "Helios", "Iridia",
        "Jovant", "Kael", "Lysand", "Mythrix", "Nocturn", "Orvex", "Pyralis", "Quinari",
        "Rhadaman", "Solace", "Thalor", "Umbra", "Vesper", "Wyrmborn", "Xanthic", "Ysolde",
        "Zephyria", "Astrar", "Celestis", "Dawnfire", "Eternis",
    ];

    let name_roots = [
        "gard", "thor", "val", "nox", "ra", "mir", "dun", "fen", "kal", "oth", "ven", "zur", "ash",
        "bel", "cor", "drak", "eld", "fal", "gor", "hak", "ion", "jer", "kre", "lux", "mor", "ner",
        "oph", "pho", "qar", "ryn",
    ];

    let name_suffixes = [
        "is", "us", "ax", "on", "ar", "el", "ix", "um", "or", "an", "ia", "os", "en", "al", "ic",
    ];

    let chaotic_prefixes = [
        "Void-Slayer",
        "Singularity",
        "Rogue-Titan",
        "Chrono-Tear",
        "Aether-Anomaly",
        "Null-Fracture",
        "Entropy-Well",
        "Quantum-Heretic",
        "Oblivion-Seed",
        "Paradox-Engine",
        "Abyss-Walker",
        "Flux-Revenant",
        "Nova-Phage",
        "Dark-Matter-Saint",
        "Gravity-Heretic",
    ];

    if entropy > 1.2 {
        let ci = (rng.next_u64() as usize) % catalog_prefixes.len();
        let num = (rng.next_u64() % 9999) + 1;
        let pi = (rng.next_u64() as usize) % chaotic_prefixes.len();
        format!("{}-{} {}", catalog_prefixes[ci], num, chaotic_prefixes[pi])
    } else if entropy > 0.5 {
        let ci = (rng.next_u64() as usize) % catalog_prefixes.len();
        let gi = (rng.next_u64() as usize) % greek.len();
        let num = (rng.next_u64() % 999) + 1;
        format!(
            "{} {}-{} {}",
            catalog_prefixes[ci],
            greek[gi],
            num,
            name_prefixes[(rng.next_u64() as usize) % name_prefixes.len()]
        )
    } else {
        let pi = (rng.next_u64() as usize) % name_prefixes.len();
        let ri = (rng.next_u64() as usize) % name_roots.len();
        let si = (rng.next_u64() as usize) % name_suffixes.len();
        let num = (rng.next_u64() % 999) + 1;
        format!(
            "{}{}{}-{}",
            name_prefixes[pi], name_roots[ri], name_suffixes[si], num
        )
    }
}

fn generate_description(
    rng: &mut SimpleRng,
    spectral_class: &str,
    category: &str,
    entropy: f32,
    teff: f32,
    rad: f32,
) -> String {
    let classification = format!(
        "Classified as {}-type {}",
        spectral_class,
        category.to_lowercase()
    );

    let stable_traits = [
        "Stable hydrogen fusion cycle with predictable luminosity output.",
        "Convective envelope maintains consistent surface granulation patterns.",
        "Radiative core operates within standard CNO cycle parameters.",
        "Chromospheric activity within nominal bounds for its spectral type.",
        "Metallicity consistent with Population I galactic disk composition.",
        "Rotational velocity and magnetic dynamo in equilibrium.",
        "Photospheric absorption lines indicate normal heavy element abundance.",
        "Hydrostatic balance maintained across all stellar layers.",
        "Lithium depletion consistent with main-sequence age estimates.",
        "Helium ash accumulation in core proceeding at expected rates.",
    ];

    let unusual_traits = [
        "Scattered crystalline structures detected forming in the solar corona.",
        "Stellar scans reveal unnatural heavy element concentrations in the core.",
        "Its stellar flares appear structurally symmetric, suggesting external manipulation.",
        "Periodic radio emissions follow a non-natural prime-number sequence.",
        "Coronal mass ejections exhibit spiral trajectories inconsistent with magnetic models.",
        "Surface granulation patterns display fractal symmetry beyond statistical expectation.",
        "Anomalous spectral lines suggest transuranic elements in the photosphere.",
        "The magnetosphere contains structured plasma formations resembling information encoding.",
        "Doppler shifts indicate subsurface resonance patterns of unknown origin.",
        "X-ray luminosity fluctuates with a precision that suggests artificial regulation.",
        "The stellar wind carries trace isotopes not producible by natural nucleosynthesis.",
        "Helioseismic data reveals a geometric core structure inconsistent with spherical models.",
        "Coronal loops reconnect in synchronized bursts at exact time intervals.",
        "The star's proper motion includes micro-corrections too precise to be gravitational.",
        "Absorption line variations spell out repeating mathematical sequences in base-12.",
        "Photon sphere measurements indicate localized spacetime curvature anomalies.",
    ];

    let anomalous_traits = [
        "Localized gravitational inversion has been detected near the photosphere.",
        "The star appears to be phase-shifting between parallel realities.",
        "Encased in a decaying ancient Dyson Swarm structure of unknown origin.",
        "Emitting anomalous tachyonic pulses that violate local causality.",
        "A micro-wormhole appears to orbit within the corona, connecting to unknown coordinates.",
        "The fusion core has been replaced by an artifact emitting Hawking radiation at unnatural frequencies.",
        "Photons leaving the photosphere carry quantum entanglement signatures from another epoch.",
        "The star's timeline contains embedded temporal loops — events repeat with escalating variance.",
        "Gravitational lensing around this star reveals a shadow biosphere in a higher spatial dimension.",
        "The magnetosphere encodes a complete mathematical proof of a civilization's existence theorem.",
        "Stellar evolution appears to be running in reverse — the core is growing younger.",
        "A crystalline computational substrate surrounds the star, computing an unknown function for eons.",
        "The star emits neutrinos with oscillation patterns that encode compressed data streams.",
        "Space-time in the vicinity exhibits topological defects consistent with engineered metric tensor manipulation.",
        "The photosphere contains stable plasma structures that spell out warnings in a dead language.",
    ];

    let hot_star_notes = [
        "Ultraviolet flux dominates the radiation spectrum.",
        "Stellar wind velocities exceed 2000 km/s.",
        "Intense Lyman-alpha emission ionizes the surrounding interstellar medium.",
        "P-Cygni profiles in the spectrum indicate massive mass loss.",
        "The O-star radiation field creates an HII region spanning several parsecs.",
    ];

    let cool_star_notes = [
        "Molecular absorption bands of TiO and VO dominate the red spectrum.",
        "Chromospheric flare activity can double the star's luminosity within minutes.",
        "Convective cells span a significant fraction of the stellar surface.",
        "Magnetic field loops create persistent starspot regions.",
        "The photosphere is cool enough for dust formation in the outer atmosphere.",
    ];

    let giant_notes = [
        "Helium shell burning produces irregular thermal pulses.",
        "The expanded envelope shows signs of recent mass ejection events.",
        "A-instabilities in the helium shell drive luminosity variations.",
        "The extended atmosphere contains molecular layers at unusual depths.",
        "Stellar pulsations suggest dredge-up of processed material to the surface.",
    ];

    let dwarf_notes = [
        "Fully convective interior allows efficient mixing throughout the star.",
        "Magnetic activity driven by a rotational dynamo despite low luminosity.",
        "The star is expected to maintain fusion for trillions of years.",
        "Flare activity can generate X-ray bursts detectable over interstellar distances.",
        "Tidal locking in close binary systems enhances magnetic field generation.",
    ];

    let spectral_note = if teff >= 7500.0 {
        let idx = (rng.next_u64() as usize) % hot_star_notes.len();
        hot_star_notes[idx]
    } else if teff < 4000.0 {
        let idx = (rng.next_u64() as usize) % cool_star_notes.len();
        cool_star_notes[idx]
    } else if rad >= 3.0 {
        let idx = (rng.next_u64() as usize) % giant_notes.len();
        giant_notes[idx]
    } else {
        let idx = (rng.next_u64() as usize) % dwarf_notes.len();
        dwarf_notes[idx]
    };

    let trait_note = if entropy > 1.5 {
        let idx = (rng.next_u64() as usize) % anomalous_traits.len();
        anomalous_traits[idx]
    } else if entropy > 0.7 {
        let idx = (rng.next_u64() as usize) % unusual_traits.len();
        unusual_traits[idx]
    } else {
        let idx = (rng.next_u64() as usize) % stable_traits.len();
        stable_traits[idx]
    };

    let connector = [
        " Additionally, ",
        " Furthermore, ",
        " Analysis shows ",
        " Deep scans indicate ",
        " Long-range sensors detect ",
        " Survey data reveals ",
        " Spectral analysis confirms ",
        " Gravitometric readings show ",
        " Helioseismic probing reveals ",
        " Interferometric data indicates ",
    ];

    let extra = if entropy > 0.3 {
        let idx = (rng.next_u64() as usize) % connector.len();
        format!("{}{}", connector[idx], spectral_note)
    } else {
        String::new()
    };

    format!("{}. {}{}{}", classification, trait_note, extra, "")
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoreEntry {
    pub id: u32,
    pub star_type: String,
    pub spectral_class: String,
    pub designated_name: String,
    pub visual_profile: String,
    pub description: String,
    pub system_lore: String,
}

pub struct LoreCache {
    entries: Vec<LoreEntry>,
}

static LORE_CACHE: RwLock<Option<Arc<LoreCache>>> = RwLock::const_new(None);

pub async fn get_lore_cache() -> Option<Arc<LoreCache>> {
    if let Some(cached) = LORE_CACHE.read().await.clone() {
        return Some(cached);
    }
    let loaded = load_lore_cache().await;
    *LORE_CACHE.write().await = loaded.clone();
    loaded
}

async fn load_lore_cache() -> Option<Arc<LoreCache>> {
    let models_dir = get_lunar_models_dir();
    let path = models_dir.join("stellar_lore_cache.json");

    if !path.exists() {
        println!("  Lore cache not found ({})", path.display());
        return None;
    }

    let json = match std::fs::read_to_string(&path) {
        Ok(j) => j,
        Err(_) => return None,
    };

    let entries: Vec<LoreEntry> = match serde_json::from_str(&json) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  Failed to parse lore cache: {e}");
            return None;
        }
    };

    println!(
        "  Lore cache loaded: {} entries from {}",
        entries.len(),
        path.display()
    );
    Some(Arc::new(LoreCache { entries }))
}

impl LoreCache {
    pub fn pick(&self, seed: u64) -> Option<&LoreEntry> {
        if self.entries.is_empty() {
            return None;
        }
        let mut rng = SimpleRng::new(seed);
        let idx = (rng.next_u64() as usize) % self.entries.len();
        Some(&self.entries[idx])
    }

    pub fn pick_by_class(&self, seed: u64, spectral_class: &str) -> Option<&LoreEntry> {
        let matching: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.spectral_class == spectral_class)
            .map(|(i, _)| i)
            .collect();

        if matching.is_empty() {
            return self.pick(seed);
        }

        let mut rng = SimpleRng::new(seed);
        let idx = (rng.next_u64() as usize) % matching.len();
        Some(&self.entries[matching[idx]])
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

pub fn is_rare_star(teff: f32, rad: f32, mass: f32, entropy: f32) -> bool {
    if entropy > 1.5 {
        return true;
    }
    if teff >= 30_000.0 || teff <= 2_500.0 {
        return true;
    }
    if rad >= 50.0 || rad <= 0.1 {
        return true;
    }
    if mass >= 15.0 || mass <= 0.08 {
        return true;
    }
    let mut rng = SimpleRng::new(((teff * 100.0) as u64) ^ ((rad * 1000.0) as u64));
    rng.next_f32() < 0.05
}

pub fn generate_hybrid_metadata(
    teff: f32,
    rad: f32,
    mass: f32,
    lum: f32,
    entropy_temperature: f32,
    lore_cache: Option<&LoreCache>,
) -> lunar_structures::StellarMetadata {
    let seed = compute_stellar_seed(teff, rad, mass, entropy_temperature);
    let (spectral_class, category) = classify_star(teff, rad);

    if is_rare_star(teff, rad, mass, entropy_temperature)
        && let Some(cache) = lore_cache
        && let Some(entry) = cache.pick_by_class(seed, &spectral_class)
    {
        return lunar_structures::StellarMetadata {
            spectral_class: entry.spectral_class.clone(),
            category,
            designated_name: entry.designated_name.clone(),
            description: format!("{}. {}", entry.description, entry.system_lore),
        };
    }

    generate_stochastic_metadata(teff, rad, mass, lum, entropy_temperature)
}

#[cfg(feature = "siren")]
static SIREN_MODEL: RwLock<Option<Arc<SirenModel>>> = RwLock::const_new(None);

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

#[cfg(feature = "siren")]
pub fn siren_generate_texture(
    siren: &SirenModel,
    width: u32,
    height: u32,
    bp_rp: f32,
    m_g: f32,
    log_teff: f32,
) -> Vec<u8> {
    let n_bp = (bp_rp - siren.norm.bp_rp_mean) / siren.norm.bp_rp_std;
    let n_mg = (m_g - siren.norm.mg_mean) / siren.norm.mg_std;
    let n_teff = (log_teff - siren.norm.log_teff_mean) / siren.norm.log_teff_std;

    const TEXTURE_CHUNK_PIXELS: usize = 8192;
    let total = width as usize * height as usize;
    let mut pixels = Vec::with_capacity(total * 3);
    for start in (0..total).step_by(TEXTURE_CHUNK_PIXELS) {
        let count = (total - start).min(TEXTURE_CHUNK_PIXELS);
        let mut input_data = Vec::with_capacity(count * SIREN_INPUT_DIM);
        for i in start..start + count {
            let x = (i % width as usize) as f32;
            let y = (i / width as usize) as f32;
            input_data.extend_from_slice(&[
                -1.0 + 2.0 * x / width.saturating_sub(1).max(1) as f32,
                -1.0 + 2.0 * y / height.saturating_sub(1).max(1) as f32,
                n_bp,
                n_mg,
                n_teff,
            ]);
        }
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input_data, [count, SIREN_INPUT_DIM]),
            &siren.device,
        );
        let vals: Vec<f32> = siren
            .model
            .forward(input)
            .into_data()
            .to_vec()
            .expect("failed to convert SIREN texture output");
        for value in vals {
            pixels.push((value.clamp(0.0, 1.0) * 255.0) as u8);
        }
    }

    pixels
}

/// Stage 6.7: fresh registry scan of the serving models directory.
/// Always re-reads from disk so `GET /version` reports ground truth.
pub async fn registry_snapshot() -> Vec<RegisteredArtifact> {
    let models_dir = get_lunar_models_dir();
    // Touch each loader's manifest gate indirectly: the scan itself
    // validates every manifest it finds (see scan_model_registry).
    scan_model_registry(&models_dir)
}

/// Stage 6.7 controlled reload report.
#[derive(Debug, Clone)]
pub struct ReloadReport {
    /// Model kinds successfully reloaded (or confirmed missing).
    pub reloaded: Vec<String>,
    /// Invalid registry entries that blocked the reload; non-empty means
    /// NO state was changed and the old models keep serving.
    pub refused: Vec<String>,
    pub note: String,
}

/// Validates the replacement PINN before swapping it into the live cache.
/// Requests already holding an `Arc` finish with their original weights.
pub async fn reload_models() -> ReloadReport {
    let entries = registry_snapshot().await;
    let serving_dir = get_lunar_models_dir().display().to_string();
    let refused: Vec<String> = entries
        .iter()
        .filter(|entry| entry.dir == serving_dir)
        .filter_map(|entry| match &entry.status {
            RegistryStatus::Verified => None,
            RegistryStatus::ReleaseBlocked(blockers) => Some(format!(
                "{:?}: release blocked ({})",
                entry.kind,
                blockers.join(", ")
            )),
            RegistryStatus::LegacyUnverified => {
                Some(format!("{:?}: artifact manifest required", entry.kind))
            }
            RegistryStatus::Invalid(reason) => Some(format!("{:?}: {reason}", entry.kind)),
        })
        .collect();
    if !refused.is_empty() {
        return ReloadReport {
            reloaded: Vec::new(),
            refused,
            note: "reload refused: fix or remove invalid bundles, old models keep serving"
                .to_string(),
        };
    }
    let pinn = match load_pinn(&get_lunar_models_dir()) {
        Ok(pinn) => pinn,
        Err(err) => {
            return ReloadReport {
                reloaded: Vec::new(),
                refused: vec![format!("pinn: {err:#}")],
                note: "reload refused: existing models keep serving".to_string(),
            };
        }
    };
    *PINN.write().await = Some(pinn);
    *GNN.write().await = None;
    *LORE_CACHE.write().await = None;
    *GNN_LOCALIZATION.write().await = None;
    #[cfg(feature = "siren")]
    {
        *SIREN_MODEL.write().await = None;
    }
    // Eagerly re-verify: a reload that silently serves nothing is a
    // failed deploy, so report per-kind load status now.
    let mut reloaded = vec!["pinn: loaded".to_string()];
    reloaded.push(format!(
        "gnn_kinematics: {}",
        if get_gnn().await.is_some() {
            "loaded"
        } else {
            "missing"
        }
    ));
    reloaded.push(format!(
        "lore_cache: {}",
        if get_lore_cache().await.is_some() {
            "loaded"
        } else {
            "missing"
        }
    ));
    #[cfg(feature = "siren")]
    reloaded.push(format!(
        "siren: {}",
        if get_siren().await.is_some() {
            "loaded"
        } else {
            "missing"
        }
    ));
    ReloadReport {
        reloaded,
        refused: Vec::new(),
        note: "PINN swapped after validation; other models refreshed".to_string(),
    }
}

pub async fn warmup_models() {
    match get_pinn().await {
        Ok(pinn) => {
            println!("  PINN model loaded, warming up GPU shaders...");
            if let Err(err) = tokio::task::spawn_blocking(move || {
                let _ = pinn_infer(
                    &pinn.model,
                    &pinn.device,
                    &pinn.norm,
                    PinnInputs {
                        position: [0.0, 0.0, 0.0],
                        bp_rp: 1.0,
                        g_mag: 10.0,
                    },
                );
            })
            .await
            {
                eprintln!("  PINN warmup failed: {err}");
            }
        }
        Err(err) => eprintln!("  PINN model unavailable: {err:#}"),
    }

    if let Some(gnn) = get_gnn().await {
        let gnn_arc = gnn.clone();
        println!(
            "  GNN model loaded (variational={}), warming up...",
            gnn_arc.variational
        );
        tokio::task::spawn_blocking(move || {
            // Two-node group: single-node inference is excluded (Stage
            // 6.6), so warmup runs the smallest valid group instead.
            let star = StarFeatures {
                coords: [0.0, 0.0, 0.0],
                log_teff: 3.75,
                log_rad: 0.0,
                log_mass: 0.0,
                log_lum: 0.0,
                mg: 0.0,
            };
            let neighbor = StarFeatures {
                coords: [1.0, 0.5, -0.5],
                ..star
            };
            let _ = gnn_infer(&gnn_arc, &[star, neighbor], 1, 0.0);
        })
        .await
        .ok();
    } else {
        println!("  GNN model not available (no .bpk file found)");
    }

    let _ = get_lore_cache().await;

    #[cfg(feature = "siren")]
    if let Some(siren) = get_siren().await {
        let siren_arc = siren.clone();
        println!("  SIREN model loaded, warming up...");
        tokio::task::spawn_blocking(move || {
            let norm = &siren_arc.norm;
            let _ = siren_infer_point(
                &siren_arc.model,
                &siren_arc.device,
                norm,
                SirenInputs {
                    uv: [0.0, 0.0],
                    bp_rp: 1.0,
                    m_g: 5.0,
                    log_teff: 3.75,
                },
            );
        })
        .await
        .ok();
    }

    let _ = get_localization().await;

    println!("  All models warmed up and ready.");
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn serving_manifest_requires_bound_release_evidence() {
        use lnai_training::artifacts::{
            ArtifactManifestV1, architecture_version, expected_feature_schema_hash, norm_file_name,
            sha256_file_hex, weight_file_name,
        };

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let models_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target")
            .join(format!("serving-manifest-test-{nonce}"));
        std::fs::create_dir_all(&models_dir).expect("create test model directory");
        let kind = ModelKind::Pinn;
        let weight_path = models_dir.join(weight_file_name(&kind));
        let norm_path = models_dir.join(norm_file_name(&kind));
        std::fs::write(&weight_path, b"test-weights").unwrap();
        std::fs::write(&norm_path, b"test-normalization").unwrap();

        let mut manifest = ArtifactManifestV1::new(
            kind.clone(),
            architecture_version(&kind).to_string(),
            sha256_file_hex(&weight_path).unwrap(),
            sha256_file_hex(&norm_path).unwrap(),
            expected_feature_schema_hash(&kind).unwrap().to_string(),
            "gaia_dr3".into(),
            "dataset-manifest-v1".into(),
            42,
            serde_json::json!({}),
            "test-revision".into(),
            "cpu".into(),
        );
        let manifest_path = models_dir.join(manifest_file_name(&kind));
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let blocked = check_serving_manifest(&models_dir, &kind).unwrap_err();
        assert!(blocked.contains("spatial_holdout_evaluation"), "{blocked}");

        let report_file = format!("{}_spatial_holdout_report.json", kind.slug());
        let report_path = models_dir.join(&report_file);
        let report_bytes = serde_json::to_vec_pretty(&serde_json::json!({
            "passed": true,
            "gate_version": "test-release-gate-v1",
            "model_kind": kind.clone(),
            "architecture_version": manifest.architecture_version.clone(),
            "model_hash": manifest.model_hash.clone(),
            "norm_hash": manifest.norm_hash.clone(),
            "dataset_id": manifest.dataset_id.clone(),
            "dataset_version": manifest.dataset_version.clone(),
            "feature_schema_hash": manifest.feature_schema_hash.clone(),
            "metrics": { "fixture_metric": 0.0 }
        }))
        .unwrap();
        std::fs::write(&report_path, &report_bytes).unwrap();
        let report_sha256 = sha256_file_hex(&report_path).unwrap();
        manifest.evaluation_metrics = Some(serde_json::json!({
            "spatial_holdout": {
                "passed": true,
                "model_hash": manifest.model_hash.clone(),
                "norm_hash": manifest.norm_hash.clone(),
                "dataset_id": manifest.dataset_id.clone(),
                "dataset_version": manifest.dataset_version.clone(),
                "feature_schema_hash": manifest.feature_schema_hash.clone(),
                "report_file": report_file,
                "report_sha256": report_sha256
            }
        }));
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        check_serving_manifest(&models_dir, &kind)
            .expect("bound passing report should be accepted");
        std::fs::write(&report_path, b"tampered report").unwrap();
        let rejected = check_serving_manifest(&models_dir, &kind).unwrap_err();
        assert!(rejected.contains("checksum mismatch"), "{rejected}");
        std::fs::remove_dir_all(models_dir).expect("remove test bundle");
    }
}
