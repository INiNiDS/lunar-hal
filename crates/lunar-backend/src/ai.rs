use std::sync::Arc;
use burn::prelude::*;
use burn_store::{BurnpackStore, ModuleSnapshot};
use lnai_models::{
    StellarGnn, StellarGnnConfig, StellarMlp, StellarMlpConfig,
    GNN_INPUT_DIM, GNN_OUTPUT_DIM, GNN_VARIATIONAL_DIM,
    compute_knn_adjacency,
};
use serde::Deserialize;
use tokio::sync::OnceCell;

#[cfg(feature = "wgpu")]
type B = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "cuda"))]
type B = burn::backend::Cuda;
#[cfg(all(not(feature = "wgpu"), not(feature = "cuda"), feature = "metal"))]
type B = burn::backend::Metal;
#[cfg(all(not(feature = "wgpu"), not(feature = "cuda"), not(feature = "metal"), feature = "rocm"))]
type B = burn::backend::Rocm;

static PINN_MODEL: &[u8] = include_bytes!("../../../models/stellar_model.bpk");
static STELLAR_NORM: &str = include_str!("../../../models/stellar_norm.json");

#[derive(Deserialize)]
pub struct StellarNorm {
    pub x_mean: f32, pub x_std: f32,
    pub y_mean: f32, pub y_std: f32,
    pub z_mean: f32, pub z_std: f32,
    pub bp_rp_mean: f32, pub bp_rp_std: f32,
    pub mg_mean: f32, pub mg_std: f32,
    pub log_teff_mean: f32, pub log_teff_std: f32,
    pub log_rad_mean: f32, pub log_rad_std: f32,
    pub log_mass_mean: f32, pub log_mass_std: f32,
    pub log_lum_mean: f32, pub log_lum_std: f32,
}

pub struct PinnModel {
    pub model: StellarMlp<B>,
    pub device: Device<B>,
    pub(crate) norm: StellarNorm,
}

static PINN: OnceCell<Arc<PinnModel>> = OnceCell::const_new();

pub async fn get_pinn() -> Arc<PinnModel> {
    PINN.get_or_init(|| async {
        let device: Device<B> = Default::default();
        let model = load_pinn(&device);
        let norm: StellarNorm = serde_json::from_str(STELLAR_NORM)
            .expect("failed to parse stellar_norm.json");
        Arc::new(PinnModel { model, device, norm })
    }).await.clone()
}

fn load_pinn(device: &Device<B>) -> StellarMlp<B> {
    let mut model = StellarMlpConfig::new().init(device);
    let mut store = BurnpackStore::from_static(PINN_MODEL);
    model.load_from(&mut store).expect("failed to load stellar model from burnpack");
    model
}

pub fn pinn_infer(
    model: &StellarMlp<B>, device: &Device<B>, norm: &StellarNorm,
    x_pc: f32, y_pc: f32, z_pc: f32, bp_rp: f32, g_mag: f32,
) -> [f32; 4] {
    let d_raw = (x_pc * x_pc + y_pc * y_pc + z_pc * z_pc).sqrt();

    let mg = if d_raw < 0.1 {
        4.67
    } else {
        g_mag - 5.0 * d_raw.log10() + 5.0
    };

    let nx = (x_pc - norm.x_mean) / norm.x_std;
    let ny = (y_pc - norm.y_mean) / norm.y_std;
    let nz = (z_pc - norm.z_mean) / norm.z_std;
    let nbp = (bp_rp - norm.bp_rp_mean) / norm.bp_rp_std;
    let nmg = (mg - norm.mg_mean) / norm.mg_std;

    let input = Tensor::<B, 2>::from_data(
        TensorData::new(vec![nx, ny, nz, nbp, nmg], [1, 5]),
        device,
    );
    let output = model.forward(input);
    let data = output.into_data();
    let vals: Vec<f32> = data.to_vec().expect("failed to convert output");

    let log_teff = vals[0] * norm.log_teff_std + norm.log_teff_mean;
    let log_rad  = vals[1] * norm.log_rad_std  + norm.log_rad_mean;
    let log_mass = vals[2] * norm.log_mass_std + norm.log_mass_mean;
    let log_lum  = vals[3] * norm.log_lum_std  + norm.log_lum_mean;

    let teff = 10f32.powf(log_teff);
    let rad  = 10f32.powf(log_rad);
    let mass = 10f32.powf(log_mass);
    let lum  = 10f32.powf(log_lum);

    [teff, rad, mass, lum]
}

fn default_one() -> f32 { 1.0 }

#[derive(Deserialize, Clone)]
pub struct GnnNorm {
    pub log_teff_mean: f32, pub log_teff_std: f32,
    pub log_rad_mean: f32, pub log_rad_std: f32,
    pub log_mass_mean: f32, pub log_mass_std: f32,
    pub log_lum_mean: f32, pub log_lum_std: f32,
    pub mg_mean: f32, pub mg_std: f32,
    pub x_mean: f32, pub x_std: f32,
    pub y_mean: f32, pub y_std: f32,
    pub z_mean: f32, pub z_std: f32,
    pub vx_mean: f32, pub vx_std: f32,
    pub vy_mean: f32, pub vy_std: f32,
    pub vz_mean: f32, pub vz_std: f32,
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
}

static GNN: OnceCell<Option<Arc<GnnModel>>> = OnceCell::const_new();

pub async fn get_gnn() -> Option<Arc<GnnModel>> {
    GNN.get_or_init(|| async {
        let norm_path = std::path::Path::new("models/stellar_gnn_norm.json");
        let bpk_path = std::path::Path::new("models/stellar_gnn_model.bpk");

        if !norm_path.exists() || !bpk_path.exists() {
            return None;
        }

        let norm: GnnNorm = match std::fs::read_to_string(norm_path) {
            Ok(json) => match serde_json::from_str(&json) {
                Ok(n) => n,
                Err(_) => return None,
            },
            Err(_) => return None,
        };

        let device: Device<B> = Default::default();
        let path_str = bpk_path.to_str().unwrap_or("");

        let mut deterministic_model = StellarGnnConfig::new(
            GNN_INPUT_DIM, 256, GNN_OUTPUT_DIM,
        ).init(&device);
        let mut store = BurnpackStore::from_file(path_str);
        if deterministic_model.load_from(&mut store).is_ok() {
            return Some(Arc::new(GnnModel {
                model: deterministic_model,
                device,
                norm,
                variational: false,
            }));
        }

        let mut variational_model = StellarGnnConfig::new(
            GNN_INPUT_DIM, 256, GNN_VARIATIONAL_DIM,
        ).init(&device);
        let mut store2 = BurnpackStore::from_file(path_str);
        if variational_model.load_from(&mut store2).is_ok() {
            return Some(Arc::new(GnnModel {
                model: variational_model,
                device,
                norm,
                variational: true,
            }));
        }

        None
    }).await.clone()
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

pub fn gnn_infer(
    gnn: &GnnModel,
    stars: &[StarFeatures],
    knn_k: usize,
    temperature: f32,
) -> Vec<[f32; 3]> {
    let n = stars.len();
    if n == 0 {
        return vec![];
    }

    let norm = &gnn.norm;
    let knn_k = knn_k.max(1).min(n);

    let coords: Vec<[f32; 3]> = stars.iter().map(|s| s.coords).collect();
    let adj = compute_knn_adjacency(&coords, knn_k);

    let mut node_data = Vec::with_capacity(n * GNN_INPUT_DIM);
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

    let mut adj_flat = Vec::with_capacity(n * n);
    for row in &adj {
        adj_flat.extend_from_slice(row);
    }

    let nodes = Tensor::<B, 2>::from_data(
        TensorData::new(node_data, [n, GNN_INPUT_DIM]),
        &gnn.device,
    );
    let adj_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(adj_flat, [n, n]),
        &gnn.device,
    );

    let output = gnn.model.forward(nodes, adj_tensor);
    let data = output.into_data();
    let vals: Vec<f32> = data.to_vec().expect("failed to convert GNN output");

    if gnn.variational {
        let dims_per_star = GNN_VARIATIONAL_DIM;
        let mut velocities = Vec::with_capacity(n);
        let mut rng = SimpleRng::new(
            ((temperature * 1000.0) as u64).wrapping_add(stars.iter().map(|s| s.coords[0].to_bits() as u64).fold(0u64, |a, b| a ^ b))
        );

        for i in 0..n {
            let base = i * dims_per_star;
            let vx_mean = vals[base] * norm.vx_std + norm.vx_mean;
            let vy_mean = vals[base + 1] * norm.vy_std + norm.vy_mean;
            let vz_mean = vals[base + 2] * norm.vz_std + norm.vz_mean;

            let vx_logvar = vals[base + 3] * norm.vx_logvar_std + norm.vx_logvar_mean;
            let vy_logvar = vals[base + 4] * norm.vy_logvar_std + norm.vy_logvar_mean;
            let vz_logvar = vals[base + 5] * norm.vz_logvar_std + norm.vz_logvar_mean;

            if temperature <= 0.0 {
                velocities.push([vx_mean, vy_mean, vz_mean]);
            } else {
                let vx_std = (vx_logvar * 0.5).exp();
                let vy_std = (vy_logvar * 0.5).exp();
                let vz_std = (vz_logvar * 0.5).exp();

                let vx = vx_mean + vx_std * rng.gaussian() * temperature;
                let vy = vy_mean + vy_std * rng.gaussian() * temperature;
                let vz = vz_mean + vz_std * rng.gaussian() * temperature;

                velocities.push([vx, vy, vz]);
            }
        }
        velocities
    } else {
        let dims_per_star = GNN_OUTPUT_DIM;
        let mut velocities = Vec::with_capacity(n);

        for i in 0..n {
            let base = i * dims_per_star;
            let vx = vals[base] * norm.vx_std + norm.vx_mean;
            let vy = vals[base + 1] * norm.vy_std + norm.vy_mean;
            let vz = vals[base + 2] * norm.vz_std + norm.vz_mean;

            if temperature > 0.0 {
                let mut rng = SimpleRng::new(
                    ((temperature * 1000.0) as u64).wrapping_add(
                        ((stars[i].coords[0] * 1000.0) as u64).wrapping_add(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64
                        )
                    )
                );
                let scale = temperature * 0.15;
                velocities.push([
                    vx + rng.gaussian() * norm.vx_std * scale,
                    vy + rng.gaussian() * norm.vy_std * scale,
                    vz + rng.gaussian() * norm.vz_std * scale,
                ]);
            } else {
                velocities.push([vx, vy, vz]);
            }
        }
        velocities
    }
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let mut s = seed;
        if s == 0 { s = 1; }
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    fn gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f32::consts::PI * u2).cos()
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

pub fn generate_stochastic_metadata(
    teff: f32, rad: f32, mass: f32, _lum: f32,
    entropy_temperature: f32,
) -> lunar_structures::StellarMetadata {
    let (spectral_class, category) = classify_star(teff, rad);

    let base_seed = ((teff * 100.0) as u64)
        ^ ((rad * 1000.0) as u64)
        ^ ((mass * 100.0) as u64);

    let final_seed = if entropy_temperature > 0.5 {
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        base_seed ^ epoch.wrapping_mul((entropy_temperature * 1000.0) as u64)
    } else {
        base_seed
    };

    let mut rng = SimpleRng::new(final_seed);

    let designated_name = if entropy_temperature > 1.2 {
        let prefixes = [
            "Void-Slayer", "Singularity", "Rogue-Titan",
            "Chrono-Tear", "Aether-Anomaly",
        ];
        let idx = (rng.next_u64() as usize) % prefixes.len();
        format!("{}-{}", prefixes[idx], (rng.next_u64() % 99) + 1)
    } else {
        let prefixes = ["Aethel", "Belis", "Cygnia", "Draconis", "Eshana"];
        let roots = ["gard", "thor", "val", "nox", "ra"];
        let pi = (rng.next_u64() as usize) % prefixes.len();
        let ri = (rng.next_u64() as usize) % roots.len();
        format!("{}{}-{}", prefixes[pi], roots[ri], (rng.next_u64() % 999) + 1)
    };

    let mut description = format!(
        "This star is designated as {}, classified as a {} ({}). ",
        designated_name, category, spectral_class
    );

    if entropy_temperature > 1.5 {
        let anomalies = [
            "Localized gravitational inversion has been detected near the photosphere. ",
            "The star appears to be phase-shifting between parallel realities. ",
            "Encased in a decaying ancient Dyson Swarm structure of unknown origin. ",
            "Emitting anomalous tachyonic pulses that violate local causality. ",
        ];
        let idx = (rng.next_u64() as usize) % anomalies.len();
        description.push_str(anomalies[idx]);
    } else if entropy_temperature > 0.7 {
        let mild = [
            "Scattered crystalline structures are forming in the solar corona. ",
            "Stellar scans reveal unnatural heavy element concentrations in the core. ",
            "Its stellar flares appear structurally symmetric, suggesting external manipulation. ",
        ];
        let idx = (rng.next_u64() as usize) % mild.len();
        description.push_str(mild[idx]);
    } else {
        description.push_str("Its thermodynamic and convective cycles are fully stable. ");
    }

    lunar_structures::StellarMetadata {
        spectral_class,
        category,
        designated_name,
        description,
    }
}

pub async fn warmup_models() {
    let pinn = get_pinn().await;
    println!("  PINN model loaded, warming up GPU shaders...");
    tokio::task::spawn_blocking(move || {
        let _ = pinn_infer(
            &pinn.model, &pinn.device, &pinn.norm,
            0.0, 0.0, 0.0, 1.0, 10.0,
        );
    }).await.ok();

    if let Some(gnn) = get_gnn().await {
        let gnn_arc = gnn.clone();
        println!("  GNN model loaded (variational={}), warming up...", gnn_arc.variational);
        tokio::task::spawn_blocking(move || {
            let star = StarFeatures {
                coords: [0.0, 0.0, 0.0],
                log_teff: 3.75,
                log_rad: 0.0,
                log_mass: 0.0,
                log_lum: 0.0,
                mg: 0.0,
            };
            let _ = gnn_infer(&gnn_arc, &[star], 1, 0.0);
        }).await.ok();
    } else {
        println!("  GNN model not available (no .bpk file found)");
    }

    println!("  All models warmed up and ready.");
}