//! The [`StellarScene`] struct: a single source of truth for all stellar-scene state.
//!
//! Every UI (Dioxus, a hypothetical TUI, a future test harness, ...)
//! talks to the same [`StellarScene`] and renders the resulting
//! [`StellarSceneSnapshot`](StellarSceneSnapshot). The frontend never reaches
//! into [`lunar_backend`](https://docs.rs/lunar-backend) directly; the
//! stellar-scene client layer is the only client of the AI HTTP API.

use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::watch;

use lunar_structures::{
    CreateStarSceneRequest, GnnResponse, PipelineRequest, PipelineResponse, RandomStarRequest,
    ResponseStar, SceneEvent, StarScene, StarSceneListResponse, StarSceneSummary,
};

use crate::api_client::ApiClient;
use crate::camera::{Camera, SceneCamera};
use crate::error::StellarSceneError;
use crate::sector::{SectorFetchRequest, SectorKey};
use crate::snapshot::StellarSceneSnapshot;
use crate::validation::{
    ValidationError, ValidationResult, validate_bp_rp, validate_center_x, validate_center_y,
    validate_center_z, validate_entropy, validate_g_mag, validate_pipeline, validate_response_star,
    validate_response_stars, validate_search_radius, validate_sector_key, validate_temperature,
    validate_scene, validate_scene_id, validate_scene_name, validate_scene_summary, validate_zoom,
};

/// How the game reaches the AI backend.
#[derive(Clone, Debug)]
pub struct StellarSceneConfig {
    /// Base URL, e.g. `http://127.0.0.1:25255`. Defaults to
    /// `LUNAR_BACKEND_HOST:LUNAR_BACKEND_PORT` (or 127.0.0.1:25255).
    pub backend_url: String,
}

impl StellarSceneConfig {
    pub fn new(backend_url: impl Into<String>) -> Self {
        Self {
            backend_url: backend_url.into(),
        }
    }
}

impl Default for StellarSceneConfig {
    fn default() -> Self {
        Self {
            backend_url: lunar_utils::env::get_url(),
        }
    }
}

#[derive(Default)]
struct StellarSceneState {
    scenes: Vec<StarSceneSummary>,
    active_scene: Option<StarScene>,
    sector_cache: HashMap<SectorKey, Vec<ResponseStar>>,
    sector_loading: HashSet<SectorKey>,
    camera: Camera,
    selected_star: Option<ResponseStar>,
    pregen: Option<GnnResponse>,
    temperature: f32,
    bp_rp: f32,
    g_mag: f32,
    sector_center: Option<(f32, f32, f32)>,
    last_temp: f32,
    pipeline: Option<PipelineResponse>,
    scene_cameras: HashMap<String, SceneCamera>,
    /// Monotonic counter. Frontends can use this to detect changes
    /// (via a `Signal<u64>` they bump on every `notify()`).
    version: u64,
}

impl StellarSceneState {
    fn new() -> Self {
        Self {
            temperature: 0.7,
            bp_rp: 1.0,
            g_mag: 10.0,
            last_temp: 0.7,
            camera: Camera::new(),
            ..Self::default()
        }
    }
}

type ChangeHandler = Box<dyn Fn(u64) + Send + Sync + 'static>;

/// The shared, framework-agnostic stellar-scene client.
///
/// `StellarScene` is `Clone` (lightweight, internal `Arc`-sharing) and is safe to
/// pass into UI components. It is also `Send + Sync`, so it can be
/// driven from background tasks (e.g., fetch workers).
pub struct StellarScene {
    state: Arc<RwLock<StellarSceneState>>,
    api: Arc<ApiClient>,
    on_change: Arc<RwLock<Option<ChangeHandler>>>,
    change_tx: watch::Sender<u64>,
}

impl Clone for StellarScene {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            api: Arc::clone(&self.api),
            on_change: Arc::clone(&self.on_change),
            change_tx: self.change_tx.clone(),
        }
    }
}

impl Default for StellarScene {
    fn default() -> Self {
        Self::with_config(StellarSceneConfig::default())
    }
}

impl StellarScene {
    /// Build a stellar-scene client pointed at the AI backend URL stored in
    /// `LUNAR_BACKEND_HOST` / `LUNAR_BACKEND_PORT` (or 127.0.0.1:25255).
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: StellarSceneConfig) -> Self {
        let (change_tx, _) = watch::channel(0u64);
        Self {
            state: Arc::new(RwLock::new(StellarSceneState::new())),
            api: Arc::new(ApiClient::new(config.backend_url)),
            on_change: Arc::new(RwLock::new(None)),
            change_tx,
        }
    }

    /// Register a callback fired after every state mutation. The
    /// callback receives the new monotonic version. Frontends use
    /// this to bump a Dioxus `Signal<u64>` (or equivalent) so the UI
    /// re-renders. The callback must be `Send + Sync`; if you need a
    /// non-`Send` subscriber, use [`StellarScene::subscribe`] instead.
    pub fn with_change_handler<F>(self, handler: F) -> Self
    where
        F: Fn(u64) + Send + Sync + 'static,
    {
        *self.on_change.write() = Some(Box::new(handler));
        self
    }

    pub fn set_change_handler<F>(&self, handler: F)
    where
        F: Fn(u64) + Send + Sync + 'static,
    {
        *self.on_change.write() = Some(Box::new(handler));
    }

    /// Get a `watch::Receiver` that fires every time the stellar-scene state
    /// changes. Useful for non-`Send` subscribers (e.g. Dioxus
    /// signals).
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.change_tx.subscribe()
    }

    pub fn backend_url(&self) -> String {
        self.api.base_url()
    }

    pub fn set_backend_url(&self, url: impl Into<String>) {
        self.api.set_base_url(url);
    }

    fn notify(&self) {
        let v = self.state.read().version;
        let _ = self.change_tx.send(v);
        if let Some(cb) = self.on_change.read().as_ref() {
            cb(v);
        }
    }

    /// Monotonic version that increases on every mutation. Frontends
    /// can compare against the previous value to decide whether to
    /// re-render.
    pub fn version(&self) -> u64 {
        self.state.read().version
    }

    /// Read-only snapshot of the entire stellar-scene state. Cheap to clone.
    pub fn snapshot(&self) -> StellarSceneSnapshot {
        let s = self.state.read();
        StellarSceneSnapshot {
            scenes: s.scenes.clone(),
            active_scene: s.active_scene.clone(),
            sector_stars: s.sector_cache.values().flatten().cloned().collect(),
            sector_loading: s.sector_loading.clone(),
            sector_cache: s.sector_cache.clone(),
            camera: s.camera,
            selected_star: s.selected_star.clone(),
            pregen: s.pregen.clone(),
            temperature: s.temperature,
            bp_rp: s.bp_rp,
            g_mag: s.g_mag,
            sector_center: s.sector_center,
            pipeline: s.pipeline.clone(),
            scene_cameras: s.scene_cameras.clone(),
        }
    }

    pub async fn refresh_scenes(&self) -> Result<usize, StellarSceneError> {
        let resp: StarSceneListResponse = self.api.list_scenes().await?;
        for w in &resp.scenes {
            validate_scene_summary(w)?;
        }
        let len = resp.scenes.len();
        let changed = {
            let mut s = self.state.write();
            if s.scenes != resp.scenes {
                s.scenes = resp.scenes;
                s.version += 1;
                true
            } else {
                false
            }
        };
        if changed {
            self.notify();
        }
        Ok(len)
    }

    pub fn scenes(&self) -> Vec<StarSceneSummary> {
        self.state.read().scenes.clone()
    }

    pub async fn load_scene(&self, id: &str) -> Result<StarScene, StellarSceneError> {
        let id = validate_scene_id(id)?;
        let scene = self.api.get_scene(id).await?;
        validate_scene(&scene)?;
        self.adopt_scene(Some(scene.clone()));
        Ok(scene)
    }

    pub async fn create_scene(&self, req: CreateStarSceneRequest) -> Result<StarScene, StellarSceneError> {
        // Validate every field of the user-facing request before any
        // network traffic.
        let name = validate_scene_name(&req.name)?.to_string();
        validate_center_x(req.center_x)?;
        validate_center_y(req.center_y)?;
        validate_center_z(req.center_z)?;
        validate_entropy(req.temperature)?;
        let req = CreateStarSceneRequest { name, ..req };

        let scene = self.api.create_scene(req).await?;
        validate_scene(&scene)?;
        self.adopt_scene(Some(scene.clone()));
        if let Ok(resp) = self.api.list_scenes().await {
            if let Ok(()) = (|| -> Result<(), ValidationError> {
                for w in &resp.scenes {
                    validate_scene_summary(w)?;
                }
                Ok(())
            })() {
                self.state.write().scenes = resp.scenes;
            }
        }
        self.state.write().version += 1;
        self.notify();
        Ok(scene)
    }

    pub async fn delete_scene(&self, id: &str) -> Result<(), StellarSceneError> {
        let id = validate_scene_id(id)?;
        self.api.delete_scene(id).await?;
        {
            let mut s = self.state.write();
            s.scenes.retain(|w| w.id != id);
            if s.active_scene.as_ref().map(|w| w.id == id).unwrap_or(false) {
                s.active_scene = None;
                s.sector_cache.clear();
                s.sector_loading.clear();
                s.sector_center = None;
            }
            s.version += 1;
        }
        self.notify();
        Ok(())
    }

    pub fn active_scene(&self) -> Option<StarScene> {
        self.state.read().active_scene.clone()
    }

    /// Set the active scene directly (e.g., when the user picks a
    /// scene from the archive) and reset the sector cache to match
    /// the new scene. Validates the scene before adopting it.
    pub fn adopt_scene(&self, scene: Option<StarScene>) {
        if let Some(ref w) = scene {
            if let Err(e) = validate_scene(w) {
                tracing::warn!(error = %e, "refusing to adopt invalid scene");
                return;
            }
        }
        {
            let mut s = self.state.write();
            s.active_scene = scene.clone();
            s.sector_cache.clear();
            s.sector_loading.clear();
            s.sector_center = scene.as_ref().map(|w| (w.center_x, w.center_y, w.center_z));
            if let Some(w) = &scene {
                if let Some(wc) = s.scene_cameras.get(&w.id).copied() {
                    s.camera = wc.to_camera();
                } else {
                    s.camera = Camera::new();
                }
                s.bp_rp = w.bp_rp;
                s.g_mag = w.g_mag;
                s.temperature = w.temperature;
                s.last_temp = w.temperature;
            }
            s.version += 1;
        }
        self.notify();
    }

    pub fn clear_active_scene(&self) {
        self.adopt_scene(None);
    }

    /// Apply a backend-owned live-scene event to the local read model. This
    /// deliberately contains no admin command logic: the mutation already
    /// happened in `lunar-backend`; the client only reflects it in the active
    /// scene and lets the renderer re-render.
    pub fn apply_scene_event(&self, event: SceneEvent) {
        let mut changed = false;
        {
            let mut state = self.state.write();
            match event {
                SceneEvent::StarAdded { scene_id, star, .. } => {
                    if state.active_scene.as_ref().is_some_and(|scene| scene.id == scene_id) {
                        let scene = state.active_scene.as_mut().expect("checked active scene");
                        if let Some(index) = scene.stars.iter().position(|item| item.id == star.id) {
                            scene.stars[index] = star;
                        } else {
                            scene.stars.push(star);
                        }
                        changed = true;
                    }
                }
                SceneEvent::StarUpdated { scene_id, star } => {
                    let selected = state
                        .selected_star
                        .as_ref()
                        .is_some_and(|selected| selected.id == star.id);
                    if state.active_scene.as_ref().is_some_and(|scene| scene.id == scene_id) {
                        let scene = state.active_scene.as_mut().expect("checked active scene");
                        if let Some(index) = scene.stars.iter().position(|item| item.id == star.id) {
                            scene.stars[index] = star.clone();
                            if selected {
                                state.selected_star = Some(star);
                            }
                            changed = true;
                        }
                    }
                }
                SceneEvent::StarRemoved { scene_id, star_id } => {
                    let selected = state
                        .selected_star
                        .as_ref()
                        .is_some_and(|selected| selected.id == star_id);
                    if state.active_scene.as_ref().is_some_and(|scene| scene.id == scene_id) {
                        let scene = state.active_scene.as_mut().expect("checked active scene");
                        let old_len = scene.stars.len();
                        scene.stars.retain(|item| item.id != star_id);
                        changed = scene.stars.len() != old_len;
                        if selected {
                            state.selected_star = None;
                            state.pipeline = None;
                        }
                    }
                }
                SceneEvent::SceneCleared { scene_id } => {
                    let has_selection = state.selected_star.is_some();
                    if state.active_scene.as_ref().is_some_and(|scene| scene.id == scene_id) {
                        let scene = state.active_scene.as_mut().expect("checked active scene");
                        if !scene.stars.is_empty() || has_selection {
                            scene.stars.clear();
                            state.selected_star = None;
                            state.pipeline = None;
                            changed = true;
                        }
                    }
                }
            }
            if changed {
                state.version += 1;
            }
        }
        if changed {
            self.notify();
        }
    }

    pub fn camera(&self) -> Camera {
        self.state.read().camera
    }

    pub fn set_camera(&self, camera: Camera) {
        {
            let mut s = self.state.write();
            s.camera = camera;
            s.version += 1;
        }
        self.notify();
    }

    pub fn pan_camera(&self, delta: (f32, f32)) {
        {
            let mut s = self.state.write();
            s.camera = s.camera.pan(delta);
            s.version += 1;
        }
        self.notify();
    }

    pub fn zoom_camera(&self, viewport: (f32, f32), factor: f32) {
        {
            let mut s = self.state.write();
            s.camera = s.camera.zoom_around_center(viewport, factor);
            s.version += 1;
        }
        self.notify();
    }

    pub fn zoom_camera_at(&self, viewport: (f32, f32), anchor: (f32, f32), factor: f32) {
        {
            let mut s = self.state.write();
            s.camera = s.camera.zoom_around(viewport, anchor, factor);
            s.version += 1;
        }
        self.notify();
    }

    pub fn set_dragging(&self, dragging: bool) {
        {
            let mut s = self.state.write();
            s.camera.dragging = dragging;
            s.version += 1;
        }
        self.notify();
    }

    pub fn recenter_camera(&self) {
        {
            let mut s = self.state.write();
            s.camera = s.camera.reset();
            s.version += 1;
        }
        self.notify();
    }

    pub fn scene_camera(&self, scene_id: &str) -> Option<SceneCamera> {
        self.state.read().scene_cameras.get(scene_id).copied()
    }

    pub fn set_scene_camera(&self, scene_id: &str, cam: SceneCamera) -> ValidationResult<()> {
        let scene_id = validate_scene_id(scene_id)?.to_string();
        let zoom = validate_zoom(cam.zoom)?;
        let off_x = validate_center_x(cam.offset.0)?;
        let off_y = validate_center_y(cam.offset.1)?;
        let cam = SceneCamera::new((off_x, off_y), zoom);
        {
            let mut s = self.state.write();
            s.scene_cameras.insert(scene_id, cam);
            s.version += 1;
        }
        self.notify();
        Ok(())
    }

    /// If a per-scene camera has been recorded for `scene_id`, apply
    /// it as the current camera. Returns whether anything was
    /// applied. Frontends call this after hydrating
    /// [`StellarScene::set_scene_camera`] from their storage layer.
    pub fn apply_scene_camera(&self, scene_id: &str) -> ValidationResult<bool> {
        let scene_id = validate_scene_id(scene_id)?.to_string();
        let mut s = self.state.write();
        if let Some(wc) = s.scene_cameras.get(&scene_id).copied() {
            s.camera = wc.to_camera();
            s.version += 1;
            drop(s);
            self.notify();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn remember_current_camera_for(&self, scene_id: &str) -> ValidationResult<()> {
        let scene_id = validate_scene_id(scene_id)?.to_string();
        let zoom = {
            let s = self.state.read();
            validate_zoom(s.camera.zoom)?
        };
        let off_x = {
            let s = self.state.read();
            validate_center_x(s.camera.offset.0)?
        };
        let off_y = {
            let s = self.state.read();
            validate_center_y(s.camera.offset.1)?
        };
        let cam = SceneCamera::new((off_x, off_y), zoom);
        let mut s = self.state.write();
        s.scene_cameras.insert(scene_id, cam);
        Ok(())
    }

    pub fn selected_star(&self) -> Option<ResponseStar> {
        self.state.read().selected_star.clone()
    }

    pub fn select_star(&self, star: Option<ResponseStar>) {
        {
            let mut s = self.state.write();
            s.selected_star = star.clone();
            s.pipeline = None;
            s.version += 1;
        }
        self.notify();
    }

    pub fn temperature(&self) -> f32 {
        self.state.read().temperature
    }

    pub fn set_temperature(&self, t: f32) -> ValidationResult<()> {
        let t = validate_temperature(t)?;
        {
            let mut s = self.state.write();
            s.temperature = t;
            s.version += 1;
        }
        self.notify();
        Ok(())
    }

    pub fn bp_rp(&self) -> f32 {
        self.state.read().bp_rp
    }

    pub fn set_bp_rp(&self, v: f32) -> ValidationResult<()> {
        let v = validate_bp_rp(v)?;
        {
            let mut s = self.state.write();
            s.bp_rp = v;
            s.version += 1;
        }
        self.notify();
        Ok(())
    }

    pub fn g_mag(&self) -> f32 {
        self.state.read().g_mag
    }

    pub fn set_g_mag(&self, v: f32) -> ValidationResult<()> {
        let v = validate_g_mag(v)?;
        {
            let mut s = self.state.write();
            s.g_mag = v;
            s.version += 1;
        }
        self.notify();
        Ok(())
    }

    pub fn sector_center(&self) -> Option<(f32, f32, f32)> {
        self.state.read().sector_center
    }

    pub fn set_sector_center(&self, center: Option<(f32, f32, f32)>) -> ValidationResult<()> {
        if let Some((x, y, z)) = center {
            validate_center_x(x)?;
            validate_center_y(y)?;
            validate_center_z(z)?;
        }
        {
            let mut s = self.state.write();
            s.sector_center = center;
            s.version += 1;
        }
        self.notify();
        Ok(())
    }

    pub fn last_temp(&self) -> f32 {
        self.state.read().last_temp
    }

    /// Internal bookkeeping; intentionally infallible because the
    /// caller has just produced `t` via [`StellarScene::set_temperature`],
    /// which already validated it.
    pub fn set_last_temp(&self, t: f32) {
        self.state.write().last_temp = t;
    }

    pub fn pregen(&self) -> Option<GnnResponse> {
        self.state.read().pregen.clone()
    }

    pub fn set_pregen(&self, pregen: Option<GnnResponse>) {
        {
            let mut s = self.state.write();
            s.pregen = pregen;
            s.version += 1;
        }
        self.notify();
    }

    /// Build a pregen sector request: if a sector center is set,
    /// request a sector there; otherwise ask the AI backend for a
    /// fully random star. All outgoing parameters are validated.
    pub async fn fetch_pregen(&self) -> Result<GnnResponse, StellarSceneError> {
        let center = self.sector_center().unwrap_or((0.0, 0.0, 0.0));
        let t = self.temperature();
        let b = self.bp_rp();
        let g = self.g_mag();
        validate_temperature(t)?;
        validate_bp_rp(b)?;
        validate_g_mag(g)?;

        let resp = if self.sector_center().is_some() {
            validate_center_x(center.0)?;
            validate_center_y(center.1)?;
            validate_center_z(center.2)?;
            self.api
                .sector_stars(lunar_structures::SectorRequest {
                    sector_cx: center.0,
                    sector_cy: center.1,
                    sector_cz: center.2,
                    temperature: t,
                    bp_rp: b,
                    g_mag: g,
                    search_radius: Some(200.0),
                })
                .await?
        } else {
            let r = self
                .api
                .random_star(RandomStarRequest {
                    entropy_temperature: t,
                })
                .await?;
            GnnResponse {
                stars: vec![r.star],
            }
        };

        validate_response_stars(&resp.stars)?;
        self.set_pregen(Some(resp.clone()));
        Ok(resp)
    }

    /// Which sectors should the renderer request from the AI backend
    /// right now, given the current viewport and camera?
    pub fn sectors_to_fetch(&self, viewport: (f32, f32)) -> Vec<(SectorKey, (f32, f32, f32))> {
        let s = self.state.read();
        let center = s
            .active_scene
            .as_ref()
            .map(|w| (w.center_x, w.center_y, w.center_z))
            .or(s.sector_center)
            .unwrap_or((0.0, 0.0, 0.0));
        let req = SectorFetchRequest {
            viewport,
            cam_offset: s.camera.offset,
            cam_zoom: s.camera.zoom,
            scene_center: center,
        };
        crate::sector::sectors_to_fetch(req, &s.sector_cache, &s.sector_loading)
    }

    /// Mark a chunk as in-flight. Returns `false` if the chunk is
    /// already cached or loading, or if the key is out of range.
    pub fn mark_sector_loading(&self, chunk: SectorKey) -> bool {
        if validate_sector_key(chunk).is_err() {
            return false;
        }
        let mut s = self.state.write();
        if s.sector_cache.contains_key(&chunk) || s.sector_loading.contains(&chunk) {
            return false;
        }
        s.sector_loading.insert(chunk);
        s.version += 1;
        drop(s);
        self.notify();
        true
    }

    /// Apply a fetched sector to the cache. The key and every star
    /// are validated; invalid input is dropped on the floor (the
    /// caller is expected to handle the error path).
    pub fn apply_sector(&self, chunk: SectorKey, stars: Vec<ResponseStar>) -> bool {
        if validate_sector_key(chunk).is_err() {
            return false;
        }
        for s in &stars {
            if validate_response_star(s).is_err() {
                return false;
            }
        }
        {
            let mut st = self.state.write();
            st.sector_cache.insert(chunk, stars);
            st.sector_loading.remove(&chunk);
            st.version += 1;
        }
        self.notify();
        true
    }

    pub fn fail_sector(&self, chunk: SectorKey) {
        {
            let mut s = self.state.write();
            if s.sector_loading.remove(&chunk) {
                s.version += 1;
                drop(s);
                self.notify();
            }
        }
    }

    /// Fetch a single sector by chunk coordinate and apply it.
    /// Every parameter is validated up-front, and the response is
    /// validated before it lands in the cache.
    pub async fn fetch_sector(
        &self,
        chunk: SectorKey,
        center: (f32, f32, f32),
        temperature: f32,
        bp_rp: f32,
        g_mag: f32,
    ) -> Result<usize, StellarSceneError> {
        validate_sector_key(chunk)?;
        validate_center_x(center.0)?;
        validate_center_y(center.1)?;
        validate_center_z(center.2)?;
        validate_temperature(temperature)?;
        validate_bp_rp(bp_rp)?;
        validate_g_mag(g_mag)?;

        if !self.mark_sector_loading(chunk) {
            return Ok(0);
        }
        let req = lunar_structures::SectorRequest {
            sector_cx: center.0,
            sector_cy: center.1,
            sector_cz: center.2,
            temperature,
            bp_rp,
            g_mag,
            search_radius: Some(validate_search_radius(200.0)?),
        };
        match self.api.sector_stars(req).await {
            Ok(resp) => {
                validate_response_stars(&resp.stars)?;
                let n = resp.stars.len();
                self.apply_sector(chunk, resp.stars);
                Ok(n)
            }
            Err(e) => {
                self.fail_sector(chunk);
                Err(e.into())
            }
        }
    }

    /// Evict farthest cached sectors. Should be called by the
    /// frontend whenever the camera moves.
    pub fn evict_excess_sectors(&self) {
        let changed = {
            let mut s = self.state.write();
            let center = get_scene_center(&s);
            let cam_pos = crate::sector::eviction_cam_pos(s.camera.offset, s.camera.zoom, center);
            let before = s.sector_cache.len();
            crate::sector::evict_excess_cache(&mut s.sector_cache, cam_pos);
            let evicted = before != s.sector_cache.len();
            if evicted {
                s.version += 1;
            }
            evicted
        };
        if changed {
            self.notify();
        }
    }

    pub fn clear_sector_cache(&self) {
        {
            let mut s = self.state.write();
            s.sector_cache.clear();
            s.sector_loading.clear();
            s.version += 1;
        }
        self.notify();
    }

    pub fn is_sector_cached_or_loading(&self, chunk: SectorKey) -> bool {
        let s = self.state.read();
        s.sector_cache.contains_key(&chunk) || s.sector_loading.contains(&chunk)
    }

    pub fn pipeline(&self) -> Option<PipelineResponse> {
        self.state.read().pipeline.clone()
    }

    pub fn clear_pipeline(&self) {
        {
            let mut s = self.state.write();
            s.pipeline = None;
            s.version += 1;
        }
        self.notify();
    }

    /// Fetch a pipeline response for the given star and cache it.
    /// Both the input star and the response are validated; the
    /// texture is sanity-checked for the expected `width*height*3`
    /// pixel count.
    pub async fn fetch_pipeline(&self, star: ResponseStar) -> Result<PipelineResponse, StellarSceneError> {
        validate_response_star(&star)?;
        let req = PipelineRequest {
            x_pc: star.x,
            y_pc: star.y,
            z_pc: star.z,
            bp_rp: 1.0,
            g_mag: 10.0,
            texture_size: 256,
        };
        let resp = self.api.pipeline(req).await?;
        validate_pipeline(&resp)?;
        {
            let mut s = self.state.write();
            s.pipeline = Some(resp.clone());
            s.version += 1;
        }
        self.notify();
        Ok(resp)
    }

    /// Spawn a star into the sector cache at the chunk that contains
    /// its scene-space position. Unlike [`StellarScene::apply_sector`], this
    /// appends to the chunk rather than overwriting it and never
    /// drops stars that were already there.
    pub fn spawn_star(&self, star: ResponseStar) -> bool {
        if validate_response_star(&star).is_err() {
            return false;
        }
        let chunk = match crate::sector::chunk_at_scene_point((star.x, star.y)) {
            Some(c) => c,
            None => return false,
        };
        if validate_sector_key(chunk).is_err() {
            return false;
        }
        {
            let mut s = self.state.write();
            let entry = s.sector_cache.entry(chunk).or_default();
            entry.retain(|st| st.id != star.id);
            entry.push(star);
            s.sector_loading.remove(&chunk);
            s.version += 1;
        }
        self.notify();
        true
    }
}

fn get_scene_center(s: &StellarSceneState) -> (f32, f32) {
    s.active_scene
        .as_ref()
        .map(|w| (w.center_x, w.center_y))
        .or_else(|| s.sector_center.map(|c| (c.0, c.1)))
        .unwrap_or((0.0, 0.0))
}
