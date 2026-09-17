use crate::ai::{
    PinnInputs, SimpleRng, StarFeatures, apparent_g_for_member, generate_hybrid_metadata, get_gnn,
    get_lore_cache, get_pinn, gnn_infer, infer_pinn_batch_async, pinn_infer,
};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{Sse, sse::{Event, KeepAlive}},
};
use lunar_structures::{
    ClearSceneRequest, CreateGalleryStarRequest, CreateSceneStarRequest, CreateStarSceneRequest,
    GallerySource, GenerateSceneStarsRequest, GnnRequest, GnnResponse, LiveSceneSnapshot,
    PinnRequest, PinnResponse, ResponseStar, SceneEvent, SectorRequest, StarModelInputs,
    StarScene, StarSceneListResponse, StarSceneSummary, StellarMetadata, UpdateSceneStarRequest,
};
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::{Path as StdPath, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::AppState;

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

pub async fn infer_pinn_async(inputs: PinnInputs) -> [f32; 4] {
    let pinn = get_pinn().await;
    tokio::task::spawn_blocking(move || pinn_infer(&pinn.model, &pinn.device, &pinn.norm, inputs))
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

pub struct SectorSeed {
    pub center: [f32; 3],
    pub search_radius: f32,
    pub seed: u64,
}

/// Stage 6: sector geometry only — deterministic member positions, no
/// physics. Every star's physical parameters come from its own batched
/// PINN inference (see [`infer_sector_stars`]), never from random
/// replication of the center value.
pub fn generate_sector_positions(spec: SectorSeed) -> Vec<[f32; 3]> {
    let [cx, cy, cz] = spec.center;
    let search_radius = spec.search_radius;
    let seed = spec.seed;

    let mut rng = SimpleRng::new(seed);
    let mut positions = Vec::with_capacity(STARS_PER_SECTOR);
    positions.push([cx, cy, cz]);

    for _ in 1..STARS_PER_SECTOR {
        let angle1 = rng.next_f32() * std::f32::consts::PI * 2.0;
        let angle2 = rng.next_f32() * std::f32::consts::PI * 2.0;
        let dist = rng.next_f32().sqrt() * search_radius * 0.8;

        positions.push([
            cx + dist * angle1.cos() * angle2.cos(),
            cy + dist * angle1.sin() * angle2.cos(),
            cz + dist * angle2.sin(),
        ]);
    }

    positions
}

/// Stage 6: per-star batched sector inference. Positions are geometric;
/// each member gets its own PINN row with shared color but its own
/// distance-modulus apparent magnitude, so parameter spread is model
/// physics — not gaussian noise around the center.
pub async fn infer_sector_stars(
    center: [f32; 3],
    search_radius: f32,
    seed: u64,
    bp_rp: f32,
    g_mag: f32,
) -> Vec<StarFeatures> {
    let positions = generate_sector_positions(SectorSeed {
        center,
        search_radius,
        seed,
    });
    let mg_center =
        calculate_absolute_magnitude(center[0], center[1], center[2], g_mag);
    let inputs: Vec<PinnInputs> = positions
        .iter()
        .map(|&position| PinnInputs {
            position,
            bp_rp,
            g_mag: apparent_g_for_member(position, mg_center, g_mag),
        })
        .collect();
    let outputs = infer_pinn_batch_async(inputs).await;

    positions
        .iter()
        .zip(outputs.iter().chain(std::iter::repeat(&[0.0, 0.0, 0.0, 0.0])))
        .map(|(&coords, &[teff, rad, mass, lum])| StarFeatures {
            coords,
            log_teff: teff.max(0.01).log10(),
            log_rad: rad.max(0.01).log10(),
            log_mass: mass.max(0.01).log10(),
            log_lum: lum.max(0.01).log10(),
            mg: mg_center,
        })
        .take(STARS_PER_SECTOR)
        .collect()
}

pub async fn compile_response_stars(stars: &[StarFeatures], temperature: f32) -> Vec<ResponseStar> {
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

            let metadata = generate_hybrid_metadata(t, r, m, l, temperature, lore.as_deref());

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

pub struct SectorQuery {
    pub center: [f32; 3],
    pub search_radius: f32,
    pub temperature: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
    pub seed: u64,
}

pub async fn generate_sector_internal(query: SectorQuery) -> Vec<ResponseStar> {
    let gnn_opt = get_gnn().await;
    if gnn_opt.is_none() {
        return Vec::new();
    }

    let stars = infer_sector_stars(
        query.center,
        query.search_radius,
        query.seed,
        query.bp_rp,
        query.g_mag,
    )
    .await;

    compile_response_stars(&stars, query.temperature).await
}

pub async fn pinn(Json(payload): Json<PinnRequest>) -> Json<PinnResponse> {
    let result = infer_pinn_async(PinnInputs {
        position: [payload.x_pc, payload.y_pc, payload.z_pc],
        bp_rp: payload.bp_rp,
        g_mag: payload.g_mag,
    })
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

    let stars = generate_sector_internal(SectorQuery {
        center: [payload.center_x, payload.center_y, payload.center_z],
        search_radius: payload.search_radius,
        temperature: payload.temperature,
        bp_rp: payload.bp_rp,
        g_mag: payload.g_mag,
        seed,
    })
    .await;

    Json(GnnResponse { stars })
}

pub async fn sector_stars(Json(payload): Json<SectorRequest>) -> Json<GnnResponse> {
    let seed = sector_seed(payload.sector_cx, payload.sector_cy, payload.sector_cz);
    let stars = generate_sector_internal(SectorQuery {
        center: [payload.sector_cx, payload.sector_cy, payload.sector_cz],
        search_radius: payload.search_radius.unwrap_or(200.0),
        temperature: payload.temperature,
        bp_rp: payload.bp_rp,
        g_mag: payload.g_mag,
        seed,
    })
    .await;
    Json(GnnResponse { stars })
}

#[derive(Default)]
pub struct SceneStore {
    inner: Arc<RwLock<HashMap<String, StarScene>>>,
    dir: PathBuf,
}

impl SceneStore {
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
        let scene = serde_json::from_slice::<StarScene>(&bytes)?;

        let mut guard = self
            .inner
            .write()
            .map_err(|_| "failed to acquire write lock")?;
        guard.insert(id.to_string(), scene);

        Ok(())
    }

    pub async fn list(&self) -> StarSceneListResponse {
        let Ok(guard) = self.inner.read() else {
            return StarSceneListResponse { scenes: vec![] };
        };
        let mut scenes: Vec<StarSceneSummary> = guard
            .values()
            .map(|w| StarSceneSummary {
                id: w.id.clone(),
                name: w.name.clone(),
                created_at: w.created_at,
                center_x: w.center_x,
                center_y: w.center_y,
                center_z: w.center_z,
                star_count: w.stars.len(),
            })
            .collect();
        scenes.sort_by_key(|w| std::cmp::Reverse(w.created_at));
        StarSceneListResponse { scenes }
    }

    pub async fn get(&self, id: &str) -> Option<StarScene> {
        let guard = self.inner.read().ok()?;
        guard.get(id).cloned()
    }

    pub async fn insert(&self, scene: StarScene) -> Result<(), String> {
        let path = self.dir.join(format!("{}.json", scene.id));
        let json = serde_json::to_vec_pretty(&scene).map_err(|e| e.to_string())?;
        let parent = path
            .parent()
            .ok_or_else(|| "scene path has no parent directory".to_string())?;
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let temporary = parent.join(format!(".{}.{}.tmp", scene.id, stamp));
        std::fs::write(&temporary, json).map_err(|error| error.to_string())?;
        std::fs::rename(&temporary, &path).map_err(|error| error.to_string())?;
        let mut guard = self.inner.write().map_err(|error| error.to_string())?;
        guard.insert(scene.id.clone(), scene);
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

pub async fn create_scene(
    State(state): State<AppState>,
    Json(req): Json<CreateStarSceneRequest>,
) -> Result<Json<LiveSceneSnapshot>, (StatusCode, String)> {
    let name = req.name.trim();
    if name.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "StarScene name cannot be empty".into()));
    }

    let (stars, bp_rp, g_mag) = generate_initial_scene_stars(&req)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    let id = generate_id();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let scene = StarScene {
        id,
        name: name.to_string(),
        created_at,
        center_x: req.center_x,
        center_y: req.center_y,
        center_z: req.center_z,
        temperature: req.temperature,
        bp_rp,
        g_mag,
        stars,
    };
    state
        .scenes
        .insert(scene.clone())
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    Ok(Json(scene.into()))
}

pub async fn list_scenes(State(state): State<AppState>) -> Json<StarSceneListResponse> {
    Json(state.scenes.list().await)
}

pub async fn get_scene(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<LiveSceneSnapshot>, (StatusCode, String)> {
    state
        .scenes
        .get(&id)
        .await
        .map(LiveSceneSnapshot::from)
        .map(Json)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("StarScene {id} not found")))
}

pub async fn delete_scene(State(state): State<AppState>, Path(id): Path<String>) -> StatusCode {
    if state.scenes.delete(&id).await {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

fn event_scene_id(event: &SceneEvent) -> &str {
    match event {
        SceneEvent::StarAdded { scene_id, .. }
        | SceneEvent::StarUpdated { scene_id, .. }
        | SceneEvent::StarRemoved { scene_id, .. }
        | SceneEvent::SceneCleared { scene_id } => scene_id,
    }
}

pub async fn scene_events(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    if state.scenes.get(&id).await.is_none() {
        return Err((StatusCode::NOT_FOUND, format!("StarScene {id} not found")));
    }
    let stream = BroadcastStream::new(state.scene_events.subscribe()).filter_map(move |message| {
        match message {
            Ok(event) if event_scene_id(&event) == id => serde_json::to_string(&event)
                .ok()
                .map(|data| Ok::<Event, Infallible>(Event::default().data(data))),
            _ => None,
        }
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

fn next_star_id(scene: &StarScene) -> u32 {
    scene
        .stars
        .iter()
        .map(|star| star.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn fallback_generated_star(scene: &StarScene, id: u32, entropy: f32) -> ResponseStar {
    let phase = (id as f32 * 0.618_034 + entropy).fract();
    let radius = 0.5 + phase * 1.7;
    let temperature = 3500.0 + phase * 6500.0;
    ResponseStar {
        id,
        x: scene.center_x + (phase - 0.5) * 120.0,
        y: scene.center_y + ((phase * 7.0).fract() - 0.5) * 120.0,
        z: scene.center_z + ((phase * 13.0).fract() - 0.5) * 40.0,
        temperature_k: temperature,
        radius,
        mass: 0.6 + phase * 1.4,
        luminosity: 0.2 + phase * 3.0,
        description: "Backend-generated stellar record".into(),
        name: format!("Generated-{id}"),
        type_hint: if temperature > 7500.0 { "A" } else { "G" }.into(),
        velocity_vector: [0.0; 3],
    }
}

fn gallery_request_for_scene_star(
    request_id: String,
    source: GallerySource,
    star: ResponseStar,
    inputs: StarModelInputs,
) -> CreateGalleryStarRequest {
    CreateGalleryStarRequest {
        request_id,
        source,
        pinn: Some(PinnResponse {
            temperature_k: star.temperature_k,
            radius_solar: star.radius,
            mass_solar: star.mass,
            luminosity_solar: star.luminosity,
        }),
        metadata: Some(StellarMetadata {
            spectral_class: star.type_hint.clone(),
            category: "live scene".into(),
            designated_name: star.name.clone(),
            description: star.description.clone(),
        }),
        name: Some(star.name.clone()),
        tags: vec!["scene".into()],
        notes: None,
        star,
        inputs,
    }
}

async fn archive_scene_star(
    state: &AppState,
    request_id: String,
    source: GallerySource,
    star: ResponseStar,
    inputs: StarModelInputs,
) -> Option<String> {
    let request = gallery_request_for_scene_star(request_id, source, star, inputs);
    let texture = crate::gallery::texture_for(&request).await;
    state.gallery.create(request, texture).ok().map(|record| record.id)
}

pub async fn generate_scene_stars(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<GenerateSceneStarsRequest>,
) -> Result<Json<Vec<ResponseStar>>, (StatusCode, String)> {
    if request.request_id.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "request_id is required".into()));
    }
    let mut scene = state
        .scenes
        .get(&id)
        .await
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("StarScene {id} not found")))?;
    let count = request.count.clamp(1, 32);
    let entropy = request.entropy_temperature.unwrap_or(scene.temperature);
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let candidates = generate_sector_internal(SectorQuery {
        center: [scene.center_x, scene.center_y, scene.center_z],
        search_radius: 120.0,
        temperature: entropy,
        bp_rp: scene.bp_rp,
        g_mag: scene.g_mag,
        seed,
    })
    .await;
    let mut next_id = next_star_id(&scene);
    let mut created = Vec::new();
    for index in 0..count as usize {
        let mut star = candidates
            .get(index)
            .cloned()
            .unwrap_or_else(|| fallback_generated_star(&scene, next_id, entropy));
        star.id = next_id;
        next_id = next_id.saturating_add(1);
        let inputs = StarModelInputs {
            x_pc: star.x,
            y_pc: star.y,
            z_pc: star.z,
            bp_rp: scene.bp_rp,
            g_mag: scene.g_mag,
            entropy_temperature: Some(entropy),
        };
        let gallery_id = archive_scene_star(
            &state,
            format!("{}:{index}", request.request_id),
            GallerySource::AdminGenerated,
            star.clone(),
            inputs,
        )
        .await;
        scene.stars.push(star.clone());
        let _ = state.scene_events.send(SceneEvent::StarAdded {
            scene_id: id.clone(),
            star: star.clone(),
            gallery_id,
        });
        created.push(star);
    }
    state
        .scenes
        .insert(scene)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    Ok(Json(created))
}

pub async fn create_scene_star(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<CreateSceneStarRequest>,
) -> Result<Json<ResponseStar>, (StatusCode, String)> {
    if request.request_id.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "request_id is required".into()));
    }
    let mut scene = state
        .scenes
        .get(&id)
        .await
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("StarScene {id} not found")))?;
    let mut star = if let Some(gallery_id) = &request.gallery_id {
        state
            .gallery
            .get(gallery_id)
            .map(|record| record.star)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Gallery star {gallery_id} not found")))?
    } else {
        request
            .star
            .clone()
            .ok_or_else(|| (StatusCode::BAD_REQUEST, "star or gallery_id is required".into()))?
    };
    star.id = next_star_id(&scene);
    let inputs = request
        .inputs
        .clone()
        .unwrap_or_else(|| StarModelInputs::from_star(&star));
    let gallery_id = if request.gallery_id.is_some() {
        request.gallery_id.clone()
    } else {
        archive_scene_star(
            &state,
            request.request_id.clone(),
            request.gallery_source,
            star.clone(),
            inputs,
        )
        .await
    };
    scene.stars.push(star.clone());
    state
        .scenes
        .insert(scene)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    let _ = state.scene_events.send(SceneEvent::StarAdded {
        scene_id: id,
        star: star.clone(),
        gallery_id,
    });
    Ok(Json(star))
}

pub async fn update_scene_star(
    State(state): State<AppState>,
    Path((id, star_id)): Path<(String, u32)>,
    Json(update): Json<UpdateSceneStarRequest>,
) -> Result<Json<ResponseStar>, (StatusCode, String)> {
    let mut scene = state
        .scenes
        .get(&id)
        .await
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("StarScene {id} not found")))?;
    let star = scene
        .stars
        .iter_mut()
        .find(|star| star.id == star_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Star {star_id} not found")))?;
    if let Some(value) = update.name { star.name = value; }
    if let Some(value) = update.description { star.description = value; }
    if let Some(value) = update.type_hint { star.type_hint = value; }
    if let Some(value) = update.temperature_k { star.temperature_k = value; }
    if let Some(value) = update.radius { star.radius = value; }
    if let Some(value) = update.mass { star.mass = value; }
    if let Some(value) = update.luminosity { star.luminosity = value; }
    let updated = star.clone();
    state
        .scenes
        .insert(scene)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    let _ = state.scene_events.send(SceneEvent::StarUpdated {
        scene_id: id,
        star: updated.clone(),
    });
    Ok(Json(updated))
}

pub async fn delete_scene_star(
    State(state): State<AppState>,
    Path((id, star_id)): Path<(String, u32)>,
) -> StatusCode {
    let Some(mut scene) = state.scenes.get(&id).await else {
        return StatusCode::NOT_FOUND;
    };
    let old_len = scene.stars.len();
    scene.stars.retain(|star| star.id != star_id);
    if scene.stars.len() == old_len {
        return StatusCode::NOT_FOUND;
    }
    if state.scenes.insert(scene).await.is_err() {
        return StatusCode::INTERNAL_SERVER_ERROR;
    }
    let _ = state.scene_events.send(SceneEvent::StarRemoved { scene_id: id, star_id });
    StatusCode::NO_CONTENT
}

pub async fn clear_scene(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<ClearSceneRequest>,
) -> StatusCode {
    if request.request_id.trim().is_empty() {
        return StatusCode::BAD_REQUEST;
    }
    let Some(mut scene) = state.scenes.get(&id).await else {
        return StatusCode::NOT_FOUND;
    };
    scene.stars.clear();
    if state.scenes.insert(scene).await.is_err() {
        return StatusCode::INTERNAL_SERVER_ERROR;
    }
    let _ = state.scene_events.send(SceneEvent::SceneCleared { scene_id: id });
    StatusCode::NO_CONTENT
}

async fn generate_initial_scene_stars(
    req: &CreateStarSceneRequest,
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
    let scene_bp_rp = 0.4 + rng.next_f32() * 2.6;
    let scene_g_mag = 4.0 + rng.next_f32() * 12.0;

    let features = infer_sector_stars(
        [req.center_x, req.center_y, req.center_z],
        SEARCH_RADIUS,
        seed,
        scene_bp_rp,
        scene_g_mag,
    )
    .await;

    let response_stars = compile_response_stars(&features, req.temperature).await;

    Ok((response_stars, scene_bp_rp, scene_g_mag))
}


#[cfg(test)]
mod live_scene_tests {
    use super::*;
    use crate::ai::apparent_g_for_member;

    #[test]
    fn sector_positions_are_deterministic_center_first_and_bounded() {
        let spec = SectorSeed {
            center: [100.0, -50.0, 25.0],
            search_radius: 250.0,
            seed: 12345,
        };
        let a = generate_sector_positions(spec);
        let b = generate_sector_positions(SectorSeed {
            center: [100.0, -50.0, 25.0],
            search_radius: 250.0,
            seed: 12345,
        });
        assert_eq!(a, b);
        assert_eq!(a.len(), STARS_PER_SECTOR);
        assert_eq!(a[0], [100.0, -50.0, 25.0]);
        for p in &a {
            let d = ((p[0] - 100.0).powi(2) + (p[1] + 50.0).powi(2) + (p[2] - 25.0).powi(2)).sqrt();
            assert!(d <= 250.0 * 0.8 + 1e-3, "out of radius: {d}");
        }
    }

    #[test]
    fn apparent_g_round_trips_through_absolute_magnitude() {
        let center = [8000.0, 0.0, 0.0];
        let g_mag = 12.0;
        let mg = calculate_absolute_magnitude(center[0], center[1], center[2], g_mag);
        let back = apparent_g_for_member(center, mg, g_mag);
        assert!((back - g_mag).abs() < 1e-3, "got {back}");
    }


    fn temporary_directory() -> PathBuf {
        std::env::temp_dir().join(format!(
            "lunar-scene-store-{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ))
    }

    fn scene() -> StarScene {
        StarScene {
            id: "scene-persistence".into(),
            name: "Persistence test".into(),
            created_at: 1,
            center_x: 0.0,
            center_y: 0.0,
            center_z: 0.0,
            temperature: 0.7,
            bp_rp: 0.85,
            g_mag: 4.83,
            stars: vec![],
        }
    }

    #[tokio::test]
    async fn scene_store_persists_atomic_snapshots_for_restart() {
        let dir = temporary_directory();
        let store = SceneStore::new(dir.clone());
        let scene = scene();
        store.insert(scene.clone()).await.unwrap();
        assert!(dir.join("scene-persistence.json").exists());
        assert_eq!(store.get("scene-persistence").await, Some(scene.clone()));

        let reloaded = SceneStore::new(dir.clone());
        assert_eq!(reloaded.get("scene-persistence").await, Some(scene));
        let leftovers = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(leftovers, 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn event_ids_are_scoped_to_their_scene() {
        let event = SceneEvent::SceneCleared { scene_id: "scene-a".into() };
        assert_eq!(event_scene_id(&event), "scene-a");
    }
}
