use crate::ai::{
    generate_hybrid_metadata, get_gnn, get_lore_cache, get_pinn,
    gnn_infer, pinn_infer, SimpleRng, StarFeatures,
};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use lunar_structures::{
    CreateWorldRequest, ResponseStar, World, WorldListResponse, WorldSummary,
    PinnRequest, PinnResponse, GnnRequest, GnnResponse, SectorRequest,
};
use std::collections::HashMap;
use std::path::{Path as StdPath, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub const STARS_PER_SECTOR: usize = 30;
pub const SEARCH_RADIUS: f32 = 250.0;

pub fn calculate_absolute_magnitude(x: f32, y: f32, z: f32, g_mag: f32) -> f32 {
    let d_raw = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
    if d_raw < 0.1 {
        4.67
    } else {
        g_mag - 5.0 * d_raw.log10() + 5.0
    }
}

pub async fn infer_pinn_async(x: f32, y: f32, z: f32, bp_rp: f32, g_mag: f32) -> [f32; 4] {
    let pinn = get_pinn().await;
    tokio::task::spawn_blocking(move || {
        pinn_infer(&pinn.model, &pinn.device, &pinn.norm, x, y, z, bp_rp, g_mag)
    })
    .await
    .unwrap_or([0.0, 0.0, 0.0, 0.0])
}

pub fn sector_seed(cx: f32, cy: f32, cz: f32) -> u64 {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    s ^= ((cx * 1000.0) as i64) as u64;
    s = s.wrapping_mul(0xBF58476D1CE4E5B9);
    s ^= ((cy * 1000.0) as i64) as u64;
    s = s.wrapping_mul(0x94D049BB133111EB);
    s ^= ((cz * 1000.0) as i64) as u64;
    if s == 0 { 1 } else { s }
}

pub fn generate_sector_stars(
    cx: f32, cy: f32, cz: f32,
    search_radius: f32,
    base_teff: f32, base_rad: f32, base_mass: f32, base_lum: f32,
    base_mg: f32,
    seed: u64,
) -> Vec<StarFeatures> {
    let mut rng = SimpleRng::new(seed);
    let mut stars = Vec::with_capacity(STARS_PER_SECTOR);

    stars.push(StarFeatures {
        coords: [cx, cy, cz],
        log_teff: base_teff.max(0.01).log10(),
        log_rad: base_rad.max(0.01).log10(),
        log_mass: base_mass.max(0.01).log10(),
        log_lum: base_lum.max(0.01).log10(),
        mg: base_mg,
    });

    for _i in 1..STARS_PER_SECTOR {
        let angle1 = rng.next_f32() * std::f32::consts::PI * 2.0;
        let angle2 = rng.next_f32() * std::f32::consts::PI * 2.0;
        let dist = rng.next_f32().sqrt() * search_radius * 0.8;

        let x = cx + dist * angle1.cos() * angle2.cos();
        let y = cy + dist * angle1.sin() * angle2.cos();
        let z = cz + dist * angle2.sin();

        let noise_teff = 1.0 + (rng.gaussian() * 0.08).clamp(-0.2, 0.2);
        let noise_rad  = 1.0 + (rng.gaussian() * 0.10).clamp(-0.25, 0.25);
        let noise_mass = 1.0 + (rng.gaussian() * 0.10).clamp(-0.25, 0.25);
        let noise_lum  = 1.0 + (rng.gaussian() * 0.12).clamp(-0.3, 0.3);

        let mg_val = base_mg + rng.gaussian() * 0.3;

        stars.push(StarFeatures {
            coords: [x, y, z],
            log_teff: (base_teff * noise_teff).max(0.01).log10(),
            log_rad: (base_rad * noise_rad).max(0.01).log10(),
            log_mass: (base_mass * noise_mass).max(0.01).log10(),
            log_lum: (base_lum * noise_lum).max(0.01).log10(),
            mg: mg_val,
        });
    }

    stars
}

pub async fn compile_response_stars(
    stars: &[StarFeatures],
    temperature: f32,
) -> Vec<ResponseStar> {
    if stars.is_empty() {
        return Vec::new();
    }

    let gnn_opt = get_gnn().await;
    let lore = get_lore_cache().await;

    let velocities = if let Some(gnn) = gnn_opt {
        let stars_clone = stars.to_vec();
        tokio::task::spawn_blocking(move || {
            gnn_infer(&gnn, &stars_clone, 8.min(stars_clone.len()), temperature)
        })
        .await
        .unwrap_or_else(|_| vec![[0.0, 0.0, 0.0]; stars.len()])
    } else {
        vec![[0.0, 0.0, 0.0]; stars.len()]
    };

    stars
        .iter()
        .zip(velocities.iter())
        .enumerate()
        .map(|(i, (star, vel))| {
            let t = 10f32.powf(star.log_teff);
            let r = 10f32.powf(star.log_rad);
            let m = 10f32.powf(star.log_mass);
            let l = 10f32.powf(star.log_lum);

            let metadata = generate_hybrid_metadata(
                t, r, m, l,
                temperature, lore.as_deref(),
            );

            ResponseStar {
                id: i as u32,
                x: star.coords[0],
                y: star.coords[1],
                z: star.coords[2],
                temperature_k: t,
                radius: r,
                mass: m,
                luminosity: l,
                description: metadata.description,
                name: metadata.designated_name,
                type_hint: metadata.spectral_class,
                velocity_vector: *vel,
            }
        })
        .collect()
}

pub async fn generate_sector_internal(
    cx: f32, cy: f32, cz: f32,
    search_radius: f32,
    temperature: f32,
    bp_rp: f32,
    g_mag: f32,
    seed: u64,
) -> Vec<ResponseStar> {
    let gnn_opt = get_gnn().await;
    if gnn_opt.is_none() {
        return Vec::new();
    }

    let [teff, rad, mass, lum] = infer_pinn_async(cx, cy, cz, bp_rp, g_mag).await;
    let mg = calculate_absolute_magnitude(cx, cy, cz, g_mag);

    let stars = tokio::task::spawn_blocking(move || {
        generate_sector_stars(
            cx, cy, cz,
            search_radius,
            teff, rad, mass, lum, mg,
            seed,
        )
    }).await.unwrap_or_default();

    compile_response_stars(&stars, temperature).await
}

pub async fn pinn(Json(payload): Json<PinnRequest>) -> Json<PinnResponse> {
    let result = infer_pinn_async(
        payload.x_pc,
        payload.y_pc,
        payload.z_pc,
        payload.bp_rp,
        payload.g_mag,
    )
    .await;

    Json(PinnResponse {
        temperature_k: result[0],
        radius_solar: result[1],
        mass_solar: result[2],
        luminosity_solar: result[3],
    })
}

pub async fn gnn(Json(payload): Json<GnnRequest>) -> Json<GnnResponse> {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
        ^ (payload.temperature * 1000.0) as u64;

    let stars = generate_sector_internal(
        payload.center_x, payload.center_y, payload.center_z,
        payload.search_radius,
        payload.temperature, payload.bp_rp, payload.g_mag,
        seed,
    ).await;

    Json(GnnResponse { stars })
}

pub async fn sector_stars(Json(payload): Json<SectorRequest>) -> Json<GnnResponse> {
    let seed = sector_seed(payload.sector_cx, payload.sector_cy, payload.sector_cz);
    let stars = generate_sector_internal(
        payload.sector_cx, payload.sector_cy, payload.sector_cz,
        payload.search_radius.unwrap_or(200.0),
        payload.temperature, payload.bp_rp, payload.g_mag,
        seed,
    ).await;
    Json(GnnResponse { stars })
}


#[derive(Default)]
pub struct WorldStore {
    inner: Arc<RwLock<HashMap<String, World>>>,
    dir: PathBuf,
}

impl WorldStore {
    pub fn new(dir: PathBuf) -> Self {
        let store = Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            dir,
        };
        store.load_from_disk();
        store
    }

    fn load_from_disk(&self) {
        let Ok(read_dir) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for entry in read_dir.flatten() {
            let _ = self.load_single_entry(&entry.path());
        }
    }

    fn load_single_entry(&self, path: &StdPath) -> Result<(), Box<dyn std::error::Error>> {
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            return Ok(());
        }
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("missing file stem")?;

        let bytes = std::fs::read(path)?;
        let world = serde_json::from_slice::<World>(&bytes)?;

        let mut guard = self.inner.write().map_err(|_| "failed to acquire write lock")?;
        guard.insert(id.to_string(), world);

        Ok(())
    }

    pub async fn list(&self) -> WorldListResponse {
        let Ok(guard) = self.inner.read() else {
            return WorldListResponse { worlds: vec![] };
        };
        let mut worlds: Vec<WorldSummary> = guard
            .values()
            .map(|w| WorldSummary {
                id: w.id.clone(),
                name: w.name.clone(),
                created_at: w.created_at,
                center_x: w.center_x,
                center_y: w.center_y,
                center_z: w.center_z,
                star_count: w.stars.len(),
            })
            .collect();
        worlds.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        WorldListResponse { worlds }
    }

    pub async fn get(&self, id: &str) -> Option<World> {
        let guard = self.inner.read().ok()?;
        guard.get(id).cloned()
    }

    pub async fn insert(&self, world: World) -> Result<(), String> {
        let path = self.dir.join(format!("{}.json", world.id));
        let json = serde_json::to_vec_pretty(&world).map_err(|e| e.to_string())?;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&path, json).map_err(|e| e.to_string())?;
        let mut guard = self.inner.write().map_err(|e| e.to_string())?;
        guard.insert(world.id.clone(), world);
        Ok(())
    }

    pub async fn delete(&self, id: &str) -> bool {
        let path = self.dir.join(format!("{}.json", id));
        let Ok(mut guard) = self.inner.write() else {
            return false;
        };
        let removed = guard.remove(id).is_some();
        let _ = std::fs::remove_file(&path);
        removed
    }
}

fn generate_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut seed = nanos as u64;
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    format!("{:016x}", seed)
}

pub async fn create_world(
    State(store): State<Arc<WorldStore>>,
    Json(req): Json<CreateWorldRequest>,
) -> Result<Json<World>, (StatusCode, String)> {
    let name = req.name.trim();
    if name.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "World name cannot be empty".into()));
    }

    let (stars, bp_rp, g_mag) = generate_world_stars(&req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let temperature = req.temperature;

    let id = generate_id();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let world = World {
        id,
        name: name.to_string(),
        created_at,
        center_x: req.center_x,
        center_y: req.center_y,
        center_z: req.center_z,
        temperature,
        bp_rp,
        g_mag,
        stars,
    };

    store
        .insert(world.clone())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(world))
}

pub async fn list_worlds(State(store): State<Arc<WorldStore>>) -> Json<WorldListResponse> {
    Json(store.list().await)
}

pub async fn get_world(
    State(store): State<Arc<WorldStore>>,
    Path(id): Path<String>,
) -> Result<Json<World>, (StatusCode, String)> {
    store
        .get(&id)
        .await
        .map(Json)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("World {id} not found")))
}

pub async fn delete_world(
    State(store): State<Arc<WorldStore>>,
    Path(id): Path<String>,
) -> StatusCode {
    if store.delete(&id).await {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn generate_world_stars(
    req: &CreateWorldRequest,
) -> Result<(Vec<ResponseStar>, f32, f32), String> {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ ((req.center_x * 100.0) as i64 as u64)
        ^ ((req.center_y * 100.0) as i64 as u64)
        ^ ((req.center_z * 100.0) as i64 as u64)
        ^ (req
            .name
            .bytes()
            .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)));

    let mut rng = SimpleRng::new(seed);
    let world_bp_rp = 0.4 + rng.next_f32() * 2.6;
    let world_g_mag = 4.0 + rng.next_f32() * 12.0;

    let [teff, rad, mass, lum] = infer_pinn_async(
        req.center_x,
        req.center_y,
        req.center_z,
        world_bp_rp,
        world_g_mag,
    ).await;

    let mg = calculate_absolute_magnitude(req.center_x, req.center_y, req.center_z, world_g_mag);

    let features = tokio::task::spawn_blocking({
        let req_cx = req.center_x;
        let req_cy = req.center_y;
        let req_cz = req.center_z;
        move || {
            generate_sector_stars(
                req_cx, req_cy, req_cz,
                SEARCH_RADIUS,
                teff, rad, mass, lum, mg,
                seed,
            )
        }
    })
    .await
    .map_err(|e| e.to_string())?;

    let response_stars = compile_response_stars(&features, req.temperature).await;

    Ok((response_stars, world_bp_rp, world_g_mag))
}