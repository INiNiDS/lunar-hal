use std::collections::HashMap;

use dioxus::prelude::*;
use futures_util::StreamExt;
use gloo_net::eventsource::futures::EventSource;

use crate::api::{
    self, FieldType, HealthResponse, ServiceConfigSchema, ServiceConfigState, ServiceConfigValues,
    ServiceInfo, ServiceLogEvent, ServiceMeta, ServiceStatus, StartServiceRequest,
};
use crate::components::ui::tokio_time_sleep;
use crate::os::manifest::{WindowSizeSpec, app_by_id};

/// Reserve space at the bottom for the Dock / safe-area.
const DOCK_HEIGHT: f64 = 96.0;
/// Distance from a screen edge (in px) that triggers Windows-style snapping.
const SNAP_EDGE_PX: f64 = 16.0;
/// Max log lines retained client-side per service (older lines are dropped;
/// the server keeps its own ring buffer, re-fetchable via `/services/{name}/logs`).
const MAX_CLIENT_LOGS_PER_SERVICE: usize = 500;

/// Coarse boot sequence for the "black room" intro: everything starts dark,
/// then the overhead lamp ignites once `lunar-start-backend` answers
/// `/health`, then the desktop reveals, then the desk/dock become
/// interactive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootPhase {
    Dark,
    LampIgnite,
    DesktopReveal,
    Ready,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowLifecycle {
    Visible,
    Minimized,
    Blocked,
}

/// Derive lifecycle without changing window ownership. The `WindowState` is
/// removed only by `close_window`; a temporary dependency loss is `Blocked`.
pub fn window_lifecycle_for(minimized: bool, missing_dependencies: &[String]) -> WindowLifecycle {
    if minimized {
        WindowLifecycle::Minimized
    } else if missing_dependencies.is_empty() {
        WindowLifecycle::Visible
    } else {
        WindowLifecycle::Blocked
    }
}

/// Resolve the browser-addressable managed frontend once, so desktop gating and
/// the Sandbox iframe cannot disagree about which service is usable.
pub fn managed_web_frontend_url(services: &[ServiceInfo]) -> Option<String> {
    services
        .iter()
        .find(|service| service.name == "frontend" && service.status.is_running())
        .filter(|service| {
            service
                .platform
                .as_deref()
                .is_some_and(|platform| platform.trim().eq_ignore_ascii_case("web"))
        })
        .and_then(|service| service.public_url.as_deref())
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(|url| url.trim_end_matches('/').to_string())
}

fn restore_existing_window(windows: &mut [WindowState], app_id: &str) -> Option<u64> {
    let window = windows.iter_mut().find(|window| window.app_id == app_id)?;
    window.minimized = false;
    Some(window.id)
}

/// Safe to call from a spawned task because it reads a captured signal rather
/// than looking up a Dioxus context from outside the component render.
pub fn is_window_lifecycle_visible(lifecycle: Option<Signal<WindowLifecycle>>) -> bool {
    lifecycle
        .map(|signal| *signal.read() == WindowLifecycle::Visible)
        .unwrap_or(true)
}

#[derive(Clone, Debug)]
pub struct WindowRuntimeContext {
    pub window_id: u64,
    pub app_id: String,
    pub lifecycle: Signal<WindowLifecycle>,
}

pub fn use_window_lifecycle() -> Option<Signal<WindowLifecycle>> {
    try_use_context::<WindowRuntimeContext>().map(|ctx| ctx.lifecycle)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServiceSettingsState {
    pub service: String,
    pub schema: Option<ServiceConfigSchema>,
    pub config: Option<ServiceConfigState>,
    pub form_values: HashMap<String, String>,
    pub client_field_errors: HashMap<String, String>,
    pub server_field_errors: HashMap<String, String>,
    pub error: Option<String>,
    pub loading: bool,
    pub validating: bool,
    pub saving: bool,
    pub starting: bool,
}

impl ServiceSettingsState {
    fn loading(service: &str) -> Self {
        Self {
            service: service.into(),
            schema: None,
            config: None,
            form_values: HashMap::new(),
            client_field_errors: HashMap::new(),
            server_field_errors: HashMap::new(),
            error: None,
            loading: true,
            validating: false,
            saving: false,
            starting: false,
        }
    }
    pub fn busy(&self) -> bool {
        self.loading || self.validating || self.saving || self.starting
    }
}

/// A single open (or minimized) OS window hosting one app/page.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowState {
    pub id: u64,
    pub app_id: String,
    pub title: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub min_width: f64,
    pub min_height: f64,
    pub z: i32,
    pub minimized: bool,
    pub maximized: bool,
    pub restore_rect: Option<(f64, f64, f64, f64)>,
}

/// Which part of a window chrome a drag/resize gesture started from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragKind {
    Move,
    ResizeN,
    ResizeS,
    ResizeE,
    ResizeW,
    ResizeNE,
    ResizeNW,
    ResizeSE,
    ResizeSW,
}

/// In-flight drag/resize gesture, tracked so a single full-viewport overlay
/// (see `window_manager.rs`) can turn mouse moves into window rect updates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DragOp {
    pub window_id: u64,
    pub kind: DragKind,
    pub start_mouse_x: f64,
    pub start_mouse_y: f64,
    pub start_x: f64,
    pub start_y: f64,
    pub start_w: f64,
    pub start_h: f64,
    pub min_width: f64,
    pub min_height: f64,
}

/// Global reactive state for the WebOS shell, installed into context once by
/// [`provide_os_state`] and read anywhere via [`use_os_state`].
#[derive(Clone, Copy)]
pub struct OsState {
    pub backend_online: Signal<bool>,
    pub boot_phase: Signal<BootPhase>,
    pub services: Signal<Vec<ServiceInfo>>,
    pub meta: Signal<Vec<ServiceMeta>>,
    pub logs: Signal<HashMap<String, Vec<ServiceLogEvent>>>,
    pub health: Signal<Option<HealthResponse>>,
    pub windows: Signal<Vec<WindowState>>,
    pub next_z: Signal<i32>,
    pub drag: Signal<Option<DragOp>>,
    pub service_settings: Signal<Option<ServiceSettingsState>>,
    pub service_config_drafts: Signal<HashMap<String, ServiceConfigValues>>,
    pub service_reveal_epochs: Signal<HashMap<String, u64>>,
    pub service_action_errors: Signal<HashMap<String, String>>,
    next_window_id: Signal<u64>,
}

impl OsState {
    fn new() -> Self {
        Self {
            backend_online: Signal::new(false),
            boot_phase: Signal::new(BootPhase::Dark),
            services: Signal::new(Vec::new()),
            meta: Signal::new(Vec::new()),
            logs: Signal::new(HashMap::new()),
            health: Signal::new(None),
            windows: Signal::new(Vec::new()),
            next_z: Signal::new(1),
            drag: Signal::new(None),
            service_settings: Signal::new(None),
            service_config_drafts: Signal::new(HashMap::new()),
            service_reveal_epochs: Signal::new(HashMap::new()),
            service_action_errors: Signal::new(HashMap::new()),
            next_window_id: Signal::new(1),
        }
    }

    pub fn are_app_dependencies_running(&self, app_id: &str) -> bool {
        self.missing_app_dependencies(app_id).is_empty()
    }

    pub fn missing_app_dependencies(&self, app_id: &str) -> Vec<String> {
        let mut missing = Vec::new();
        let services = self.services.read();

        if let Some(service) = app_id.strip_prefix("log:") {
            let is_running = services
                .iter()
                .any(|s| s.name == service && s.status.is_running());
            if !is_running {
                missing.push(service.to_string());
            }
            return missing;
        }

        if app_id == "sandbox" {
            let backend_running = services
                .iter()
                .any(|s| s.name == "backend" && s.status.is_running());
            if !backend_running {
                missing.push("backend".to_string());
            }

            let frontend = services.iter().find(|s| s.name == "frontend");
            let frontend_running = frontend.as_ref().is_some_and(|s| s.status.is_running());
            if !frontend_running {
                missing.push("frontend".to_string());
            } else if managed_web_frontend_url(&services).is_none() {
                missing.push("frontend (web platform with public URL required)".to_string());
            }
            return missing;
        }

        if let Some(app) = app_by_id(app_id) {
            for required in app.required_services {
                let running = services
                    .iter()
                    .any(|s| s.name == *required && s.status.is_running());
                if !running {
                    missing.push((*required).to_string());
                }
            }
        }

        missing
    }

    fn sandbox_has_web_frontend(services: &[ServiceInfo]) -> bool {
        managed_web_frontend_url(services).is_some()
    }

    fn sandbox_is_available(services: &[ServiceInfo]) -> bool {
        services
            .iter()
            .any(|service| service.name == "backend" && service.status.is_running())
            && Self::sandbox_has_web_frontend(services)
    }

    pub fn is_app_visible(&self, app_id: &str) -> bool {
        app_id != "sandbox" || Self::sandbox_is_available(&self.services.read())
    }

    pub fn is_app_available(&self, app_id: &str) -> bool {
        self.is_app_visible(app_id) && self.are_app_dependencies_running(app_id)
    }

    /// Existing windows stay reachable even when launch dependencies go down.
    pub fn has_window(&self, app_id: &str) -> bool {
        self.windows
            .read()
            .iter()
            .any(|window| window.app_id == app_id)
    }

    pub fn service_reveal_epoch(&self, service: &str) -> u64 {
        *self.service_reveal_epochs.read().get(service).unwrap_or(&0)
    }

    pub fn set_services(&mut self, next: Vec<ServiceInfo>) {
        let previous = self.services.read().clone();
        for service in &next {
            let was_running = previous
                .iter()
                .find(|item| item.name == service.name)
                .is_some_and(|item| item.status.is_running());
            if service.status.is_running() && !was_running {
                *self
                    .service_reveal_epochs
                    .write()
                    .entry(service.name.clone())
                    .or_default() += 1;
                self.service_action_errors.write().remove(&service.name);
            }
        }
        // Service/platform changes alter the window lifecycle to `Blocked`;
        // they must never destroy an already-open (possibly minimized) window.
        self.services.set(next);
    }

    pub fn service_status(&self, name: &str) -> Option<ServiceStatus> {
        self.services
            .read()
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.status.clone())
    }

    pub fn service_meta(&self, name: &str) -> Option<ServiceMeta> {
        self.meta.read().iter().find(|m| m.name == name).cloned()
    }

    pub fn open_window(&mut self, app_id: &str, title: &str) {
        // Restore before checking launch prerequisites. A mounted window may be
        // blocked by a temporarily missing service, but it must remain reachable.
        let existing_id = self
            .windows
            .with_mut(|windows| restore_existing_window(windows, app_id));
        if let Some(id) = existing_id {
            self.focus_window(id);
            return;
        }
        if !self.is_app_available(app_id) {
            return;
        }

        let id = *self.next_window_id.read();
        self.next_window_id.set(id + 1);
        let z = *self.next_z.read();
        self.next_z.set(z + 1);

        let spec: WindowSizeSpec = if let Some(app) = app_by_id(app_id) {
            app.size
        } else if app_id.starts_with("log:") {
            WindowSizeSpec {
                preferred_width: 800.0,
                preferred_height: 520.0,
                min_width: 320.0,
                min_height: 240.0,
            }
        } else {
            WindowSizeSpec {
                preferred_width: 860.0,
                preferred_height: 600.0,
                min_width: 280.0,
                min_height: 160.0,
            }
        };

        let (viewport_w, viewport_h) = crate::os::viewport_size();
        let available_w = viewport_w;
        let available_h = (viewport_h - DOCK_HEIGHT).max(0.0);

        let (width, height) = spec.fit_to_available_space(available_w, available_h);

        let count = self.windows.read().len() as f64;
        let offset = (count % 8.0) * 28.0;
        let mut x = 120.0 + offset;
        let mut y = 80.0 + offset;

        if count == 0.0 {
            x = (available_w - width) / 2.0;
            y = (available_h - height) / 2.0;
        }

        if x + width > available_w {
            x = (available_w - width).max(0.0);
        }
        if y + height > available_h {
            y = (available_h - height).max(0.0);
        }

        self.windows.with_mut(|ws| {
            ws.push(WindowState {
                id,
                app_id: app_id.to_string(),
                title: title.to_string(),
                x,
                y,
                width,
                height,
                min_width: spec.min_width,
                min_height: spec.min_height,
                z,
                minimized: false,
                maximized: false,
                restore_rect: None,
            });
        });
    }

    pub fn activate_app(&mut self, app_id: &str) {
        let id = self
            .windows
            .with_mut(|windows| restore_existing_window(windows, app_id));
        if let Some(id) = id {
            self.focus_window(id);
        }
    }

    fn upsert_service(&mut self, service: ServiceInfo) {
        let mut next = self.services.read().clone();
        if let Some(current) = next.iter_mut().find(|item| item.name == service.name) {
            *current = service;
        } else {
            next.push(service);
        }
        self.set_services(next);
    }

    pub fn open_service_settings(&mut self, service: &str) {
        self.service_settings
            .set(Some(ServiceSettingsState::loading(service)));
        let running = self
            .service_status(service)
            .as_ref()
            .is_some_and(ServiceStatus::is_running);
        let service = service.to_string();
        let mut settings = self.service_settings;
        let mut drafts = self.service_config_drafts;
        spawn(async move {
            match futures_util::future::join(
                api::get_service_config_schema(&service),
                api::get_service_config(&service),
            )
            .await
            {
                (Ok(schema), Ok(config)) => {
                    let values = if running {
                        config
                            .effective
                            .clone()
                            .unwrap_or_else(|| config.saved.clone())
                    } else {
                        drafts
                            .read()
                            .get(&service)
                            .cloned()
                            .unwrap_or_else(|| config.saved.clone())
                    };
                    if !running {
                        drafts
                            .write()
                            .entry(service.clone())
                            .or_insert_with(|| values.clone());
                    }
                    settings.set(Some(ServiceSettingsState {
                        service: service.clone(),
                        form_values: form_values_from_config(&service, &schema, &values),
                        schema: Some(schema),
                        config: Some(config),
                        client_field_errors: HashMap::new(),
                        server_field_errors: HashMap::new(),
                        error: None,
                        loading: false,
                        validating: false,
                        saving: false,
                        starting: false,
                    }));
                }
                (Err(error), _) | (_, Err(error)) => {
                    if let Some(state) = settings.write().as_mut() {
                        state.loading = false;
                        state.error = Some(error.to_string());
                    }
                }
            }
        });
    }

    pub fn close_service_settings(&mut self) {
        self.service_settings.set(None);
    }

    pub fn update_service_setting(&mut self, key: &str, value: String) {
        let mut draft = None;
        if let Some(state) = self.service_settings.write().as_mut() {
            state.form_values.insert(key.into(), value);
            state.client_field_errors.remove(key);
            state.server_field_errors.remove(key);
            state.error = None;
            if let (Some(schema), Some(config)) = (&state.schema, &state.config) {
                draft = Some((
                    state.service.clone(),
                    config_from_form(&state.service, schema, &config.saved, &state.form_values),
                ));
            }
        }
        if let Some((service, values)) = draft {
            self.service_config_drafts.write().insert(service, values);
        }
    }

    pub fn reset_service_settings(&mut self) {
        let mut draft = None;
        if let Some(state) = self.service_settings.write().as_mut() {
            if let (Some(schema), Some(config)) = (&state.schema, &state.config) {
                state.form_values =
                    form_values_from_config(&state.service, schema, &config.defaults);
                state.client_field_errors.clear();
                state.server_field_errors.clear();
                state.error = None;
                draft = Some((state.service.clone(), config.defaults.clone()));
            }
        }
        if let Some((service, values)) = draft {
            self.service_config_drafts.write().insert(service, values);
        }
    }

    pub fn validate_and_start_service(&mut self) {
        let Some(snapshot) = self.service_settings.read().clone() else {
            return;
        };
        let (Some(schema), Some(config)) = (snapshot.schema, snapshot.config) else {
            return;
        };
        let values = config_from_form(
            &snapshot.service,
            &schema,
            &config.saved,
            &snapshot.form_values,
        );
        let mut errors = validate_form(&schema, &snapshot.form_values);
        let mut known = self.service_config_drafts.read().clone();
        known.insert(snapshot.service.clone(), values.clone());
        if let Some((field, message)) = known_port_conflict(&snapshot.service, &known) {
            errors.insert(field, message);
        }
        if !errors.is_empty() {
            if let Some(state) = self.service_settings.write().as_mut() {
                state.client_field_errors = errors;
                state.server_field_errors.clear();
                state.error = Some("Check the highlighted settings.".into());
            }
            return;
        }
        if let Some(state) = self.service_settings.write().as_mut() {
            state.validating = true;
            state.error = None;
            state.client_field_errors.clear();
            state.server_field_errors.clear();
        }
        let service = snapshot.service;
        let restart_frontend = service == "frontend"
            && self
                .service_status("frontend")
                .as_ref()
                .is_some_and(ServiceStatus::is_running);
        let mut settings = self.service_settings;
        let mut drafts = self.service_config_drafts;
        let mut os = *self;
        spawn(async move {
            match api::validate_service_config(&service, &values).await {
                Ok(result) if result.ok => {}
                Ok(result) => {
                    if let Some(state) = settings.write().as_mut() {
                        state.validating = false;
                        state.server_field_errors = result.field_errors;
                        state.error = Some("Service configuration is invalid.".into());
                    }
                    return;
                }
                Err(error) => {
                    apply_service_error(settings, error);
                    return;
                }
            }
            if let Some(state) = settings.write().as_mut() {
                state.validating = false;
                state.saving = true;
            }
            if let Err(error) = api::save_service_config(&service, &values).await {
                apply_service_error(settings, error);
                return;
            }
            if restart_frontend {
                match api::list_start_services().await {
                    Ok(services) => {
                        os.set_services(services);
                        drafts.write().remove(&service);
                        settings.set(None);
                    }
                    Err(error) => {
                        if let Some(state) = settings.write().as_mut() {
                            state.saving = false;
                            state.error = Some(error);
                        }
                    }
                }
                return;
            }
            if let Some(state) = settings.write().as_mut() {
                state.saving = false;
                state.starting = true;
            }
            match api::start_service_request(&service, &StartServiceRequest { config: None }).await
            {
                Ok(action) => {
                    os.upsert_service(action.service);
                    drafts.write().remove(&service);
                    settings.set(None);
                }
                Err(error) => apply_service_error(settings, error),
            }
        });
    }

    pub fn restart_managed_service(&mut self, service: &str) {
        self.run_service_action(service, true);
    }

    pub fn stop_managed_service(&mut self, service: &str) {
        self.run_service_action(service, false);
    }

    fn run_service_action(&mut self, service: &str, restart: bool) {
        let service = service.to_string();
        let mut os = *self;
        self.service_action_errors.write().remove(&service);
        spawn(async move {
            let result = if restart {
                api::restart_service(&service).await
            } else {
                api::stop_service(&service).await
            };
            match result {
                Ok(action) => os.upsert_service(action.service),
                Err(error) => {
                    os.service_action_errors
                        .write()
                        .insert(service, error.message);
                }
            }
        });
    }

    fn cancel_drag_for_window(&mut self, id: u64) {
        let dragging_window = self.drag.read().as_ref().map(|drag| drag.window_id);
        if dragging_window == Some(id) {
            self.cancel_drag();
        }
    }

    pub fn cancel_drag(&mut self) {
        self.drag.set(None);
    }

    pub fn close_window(&mut self, id: u64) {
        self.cancel_drag_for_window(id);
        self.windows.with_mut(|ws| ws.retain(|w| w.id != id));
    }

    pub fn minimize_window(&mut self, id: u64) {
        self.cancel_drag_for_window(id);
        self.windows.with_mut(|ws| {
            if let Some(w) = ws.iter_mut().find(|w| w.id == id) {
                w.minimized = true;
            }
        });
    }

    pub fn toggle_maximize_window(&mut self, id: u64, viewport_w: f64, viewport_h: f64) {
        self.windows.with_mut(|ws| {
            if let Some(w) = ws.iter_mut().find(|w| w.id == id) {
                let avail_h = (viewport_h - DOCK_HEIGHT).max(0.0);
                if w.maximized {
                    if let Some((x, y, width, height)) = w.restore_rect.take() {
                        let clamped_w = width.min(viewport_w).max(w.min_width.min(viewport_w));
                        let clamped_h = height.min(avail_h).max(w.min_height.min(avail_h));
                        w.x = x.min(viewport_w - clamped_w).max(0.0);
                        w.y = y.min(avail_h - clamped_h).max(0.0);
                        w.width = clamped_w;
                        w.height = clamped_h;
                    }
                    w.maximized = false;
                } else {
                    w.restore_rect = Some((w.x, w.y, w.width, w.height));
                    w.x = 0.0;
                    w.y = 0.0;
                    w.width = viewport_w;
                    w.height = avail_h;
                    w.maximized = true;
                }
            }
        });
    }

    pub fn focus_window(&mut self, id: u64) {
        let z = *self.next_z.read();
        self.next_z.set(z + 1);
        self.windows.with_mut(|ws| {
            if let Some(w) = ws.iter_mut().find(|w| w.id == id) {
                w.z = z;
            }
        });
    }

    pub fn begin_drag(&mut self, window_id: u64, kind: DragKind, mouse_x: f64, mouse_y: f64) {
        let rect = self
            .windows
            .read()
            .iter()
            .find(|w| w.id == window_id)
            .map(|w| (w.x, w.y, w.width, w.height, w.min_width, w.min_height));
        let Some((x, y, w, h, min_w, min_h)) = rect else {
            return;
        };
        self.focus_window(window_id);
        self.drag.set(Some(DragOp {
            window_id,
            kind,
            start_mouse_x: mouse_x,
            start_mouse_y: mouse_y,
            start_x: x,
            start_y: y,
            start_w: w,
            start_h: h,
            min_width: min_w,
            min_height: min_h,
        }));
    }

    pub fn update_drag(&mut self, mouse_x: f64, mouse_y: f64) {
        let Some(op) = *self.drag.read() else { return };
        let dx = mouse_x - op.start_mouse_x;
        let dy = mouse_y - op.start_mouse_y;

        let (mut x, mut y, mut w, mut h) = (op.start_x, op.start_y, op.start_w, op.start_h);
        let min_w = op.min_width;
        let min_h = op.min_height;

        match op.kind {
            DragKind::Move => {
                x += dx;
                y += dy;
            }
            DragKind::ResizeE => w = (op.start_w + dx).max(min_w),
            DragKind::ResizeW => {
                w = (op.start_w - dx).max(min_w);
                x = op.start_x + (op.start_w - w);
            }
            DragKind::ResizeS => h = (op.start_h + dy).max(min_h),
            DragKind::ResizeN => {
                h = (op.start_h - dy).max(min_h);
                y = op.start_y + (op.start_h - h);
            }
            DragKind::ResizeSE => {
                w = (op.start_w + dx).max(min_w);
                h = (op.start_h + dy).max(min_h);
            }
            DragKind::ResizeSW => {
                w = (op.start_w - dx).max(min_w);
                x = op.start_x + (op.start_w - w);
                h = (op.start_h + dy).max(min_h);
            }
            DragKind::ResizeNE => {
                w = (op.start_w + dx).max(min_w);
                h = (op.start_h - dy).max(min_h);
                y = op.start_y + (op.start_h - h);
            }
            DragKind::ResizeNW => {
                w = (op.start_w - dx).max(min_w);
                x = op.start_x + (op.start_w - w);
                h = (op.start_h - dy).max(min_h);
                y = op.start_y + (op.start_h - h);
            }
        }

        self.windows.with_mut(|ws| {
            if let Some(win) = ws.iter_mut().find(|win| win.id == op.window_id) {
                win.x = x;
                win.y = y;
                win.width = w;
                win.height = h;
                win.maximized = false;
            }
        });
    }

    pub fn end_drag(&mut self, viewport_w: f64, viewport_h: f64, mouse_x: f64, mouse_y: f64) {
        let Some(op) = *self.drag.read() else { return };
        self.drag.set(None);
        if op.kind != DragKind::Move {
            return;
        }

        let win_id = op.window_id;
        let avail_h = (viewport_h - DOCK_HEIGHT).max(0.0);
        let min_w = op.min_width;

        if mouse_y <= SNAP_EDGE_PX {
            self.snap_rect(win_id, 0.0, 0.0, viewport_w, avail_h);
        } else if mouse_x <= SNAP_EDGE_PX {
            let half_w = (viewport_w / 2.0).max(min_w);
            let snap_w = if half_w > viewport_w {
                viewport_w
            } else {
                half_w
            };
            self.snap_rect(win_id, 0.0, 0.0, snap_w, avail_h);
        } else if mouse_x >= viewport_w - SNAP_EDGE_PX {
            let half_w = (viewport_w / 2.0).max(min_w);
            let snap_w = if half_w > viewport_w {
                viewport_w
            } else {
                half_w
            };
            let snap_x = if snap_w >= viewport_w {
                0.0
            } else {
                viewport_w - snap_w
            };
            self.snap_rect(win_id, snap_x, 0.0, snap_w, avail_h);
        }
    }

    fn snap_rect(&mut self, win_id: u64, x: f64, y: f64, width: f64, height: f64) {
        self.windows.with_mut(|ws| {
            if let Some(w) = ws.iter_mut().find(|w| w.id == win_id) {
                if !w.maximized {
                    w.restore_rect = Some((w.x, w.y, w.width, w.height));
                }
                w.x = x;
                w.y = y;
                w.width = width;
                w.height = height;
                w.maximized = true;
            }
        });
    }

    pub fn reflow_windows(&mut self, viewport_w: f64, viewport_h: f64) {
        let avail_h = (viewport_h - DOCK_HEIGHT).max(0.0);
        self.windows.with_mut(|ws| {
            for w in ws.iter_mut() {
                if w.maximized {
                    w.width = viewport_w;
                    w.height = avail_h;
                    w.x = 0.0;
                    w.y = 0.0;
                } else {
                    if w.width > viewport_w {
                        w.width = viewport_w.max(w.min_width.min(viewport_w));
                    }
                    if w.height > avail_h {
                        w.height = avail_h.max(w.min_height.min(avail_h));
                    }
                    if w.x + w.width > viewport_w {
                        w.x = (viewport_w - w.width).max(0.0);
                    }
                    if w.x < 0.0 {
                        w.x = 0.0;
                    }
                    if w.y > avail_h - 40.0 {
                        w.y = (avail_h - 40.0).max(0.0);
                    }
                    if w.y < 0.0 {
                        w.y = 0.0;
                    }
                }
            }
        });
    }
}

fn apply_service_error(
    mut settings: Signal<Option<ServiceSettingsState>>,
    error: api::ServiceApiError,
) {
    if let Some(state) = settings.write().as_mut() {
        state.loading = false;
        state.validating = false;
        state.saving = false;
        state.starting = false;
        state.server_field_errors = error.field_errors;
        state.error = Some(error.message);
    }
}

fn split_lines(value: &str) -> Vec<String> {
    value
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

fn custom_build_args(values: &ServiceConfigValues) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < values.build_args.len() {
        if values.build_args[i] == "--no-default-features" {
            i += 1
        } else if values.build_args[i] == "--features" {
            i += 2
        } else {
            out.push(values.build_args[i].clone());
            i += 1
        }
    }
    out
}

fn compute_features(values: &ServiceConfigValues) -> (String, bool) {
    if let Some(i) = values.build_args.iter().position(|a| a == "--features") {
        if let Some(value) = values.build_args.get(i + 1) {
            let f = value.split(',').collect::<Vec<_>>();
            let backend = ["wgpu", "cuda", "cpu", "metal", "rocm"]
                .into_iter()
                .find(|x| f.contains(x))
                .unwrap_or("wgpu");
            return (backend.into(), f.contains(&"siren"));
        }
    }
    ("wgpu".into(), true)
}

pub fn form_values_from_config(
    _service: &str,
    schema: &ServiceConfigSchema,
    values: &ServiceConfigValues,
) -> HashMap<String, String> {
    let (compute, siren) = compute_features(values);
    schema
        .fields
        .iter()
        .map(|field| {
            let value = match field.key.as_str() {
                "EXTRA_ARGS" => values.extra_args.join("\n"),
                "BUILD_ARGS" => custom_build_args(values).join("\n"),
                "COMPUTE_BACKEND" => compute.clone(),
                "SIREN" => siren.to_string(),
                "LUNAR_FRONTEND_PLATFORM" => values
                    .env
                    .get("LUNAR_FRONTEND_PLATFORM")
                    .cloned()
                    .unwrap_or_else(|| field.default_value.clone()),
                "LUNAR_FRONTEND_PORT" => values
                    .env
                    .get("LUNAR_FRONTEND_PORT")
                    .cloned()
                    .unwrap_or_else(|| field.default_value.clone()),
                "BIND_HOST" | "CRATE" | "CRATE_SUBDIR" => field.default_value.clone(),
                key => values
                    .env
                    .get(key)
                    .cloned()
                    .unwrap_or_else(|| field.default_value.clone()),
            };
            (field.key.clone(), value)
        })
        .collect()
}

pub fn config_from_form(
    service: &str,
    schema: &ServiceConfigSchema,
    base: &ServiceConfigValues,
    form: &HashMap<String, String>,
) -> ServiceConfigValues {
    let mut result = base.clone();
    for field in &schema.fields {
        let value = form
            .get(&field.key)
            .cloned()
            .unwrap_or_else(|| field.default_value.clone());
        if field.key.starts_with("LUNAR_") {
            if value.trim().is_empty() {
                result.env.remove(&field.key);
            } else {
                result.env.insert(field.key.clone(), value.trim().into());
            }
        }
    }
    if service == "testbench-backend" {
        if let Some(port) = result.env.get("LUNAR_TESTBENCH_BACKEND_PORT").cloned() {
            result.env.insert("LUNAR_TESTBENCH_PORT".into(), port);
        }
    }
    result.extra_args = form
        .get("EXTRA_ARGS")
        .map(|v| split_lines(v))
        .unwrap_or_default();
    if service == "frontend" {
        let platform = form
            .get("LUNAR_FRONTEND_PLATFORM")
            .cloned()
            .unwrap_or_else(|| "web".into());
        result
            .env
            .insert("LUNAR_FRONTEND_PLATFORM".into(), platform.clone());
        if platform == "web" {
            let port = form
                .get("LUNAR_FRONTEND_PORT")
                .cloned()
                .unwrap_or_else(|| "8080".into());
            result.env.insert("LUNAR_FRONTEND_PORT".into(), port);
        } else {
            result.env.remove("LUNAR_FRONTEND_PORT");
        }
    }
    result.build_args = form
        .get("BUILD_ARGS")
        .map(|v| split_lines(v))
        .unwrap_or_default();
    if service == "backend" {
        let compute = form
            .get("COMPUTE_BACKEND")
            .map(String::as_str)
            .unwrap_or("wgpu");
        let siren = form.get("SIREN").is_none_or(|v| v == "true");
        if compute != "wgpu" || !siren {
            let features = if siren {
                format!("{compute},siren")
            } else {
                compute.into()
            };
            result.build_args.splice(
                0..0,
                [
                    "--no-default-features".into(),
                    "--features".into(),
                    features,
                ],
            );
        }
    }
    result
}

fn configured_port(service: &str, values: &ServiceConfigValues) -> Option<u16> {
    match service {
        "backend" => values.env.get("LUNAR_BACKEND_PORT")?.parse().ok(),
        "testbench-backend" => values.env.get("LUNAR_TESTBENCH_BACKEND_PORT")?.parse().ok(),
        "frontend"
            if values
                .env
                .get("LUNAR_FRONTEND_PLATFORM")
                .is_some_and(|platform| platform == "web") =>
        {
            values
                .env
                .get("LUNAR_FRONTEND_PORT")
                .and_then(|port| port.parse().ok())
        }
        "frontend" => None,
        _ => None,
    }
}

fn port_field(service: &str) -> &'static str {
    match service {
        "backend" => "LUNAR_BACKEND_PORT",
        "testbench-backend" => "LUNAR_TESTBENCH_BACKEND_PORT",
        "frontend" => "LUNAR_FRONTEND_PORT",
        _ => "port",
    }
}

fn known_port_conflict(
    service: &str,
    configs: &HashMap<String, ServiceConfigValues>,
) -> Option<(String, String)> {
    let port = configured_port(service, configs.get(service)?)?;
    configs.iter().find_map(|(other, values)| {
        if other != service && configured_port(other, values) == Some(port) {
            Some((
                port_field(service).into(),
                format!("Port {port} is already selected for {other}."),
            ))
        } else {
            None
        }
    })
}

pub fn validate_form(
    schema: &ServiceConfigSchema,
    form: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut errors = HashMap::new();
    for field in &schema.fields {
        let value = form.get(&field.key).map(String::as_str).unwrap_or("");
        if field.required && value.trim().is_empty() {
            errors.insert(field.key.clone(), "This field is required.".into());
            continue;
        }
        if matches!(field.field_type, FieldType::Port) && !value.trim().is_empty() {
            if !matches!(value.parse::<u32>(), Ok(1..=65535)) {
                errors.insert(field.key.clone(), "Enter a port from 1 to 65535.".into());
                continue;
            }
        }
        if let Some(allowed) = &field.allowed_values {
            if !allowed.iter().any(|item| item == value) {
                errors.insert(field.key.clone(), "Choose a supported value.".into());
            }
        }
    }
    errors
}

/// Installs [`OsState`] into context. Call exactly once, near the app root (in `Room`).
pub fn provide_os_state() -> OsState {
    use_context_provider(OsState::new)
}

/// Reads the [`OsState`] previously installed by [`provide_os_state`].
pub fn use_os_state() -> OsState {
    use_context::<OsState>()
}

/// Spawns the background tasks that drive the whole shell: polls `/health` to
/// advance the boot sequence, polls `/services` + `/services/meta`, and keeps
/// a live SSE log stream flowing into `logs`. Call exactly once, from `Room`.
pub fn use_os_runtime() {
    let mut os = use_os_state();

    // Health polling -> one-way boot sequence. Once the desktop has reached
    // Ready, a transient backend outage may mark it offline but must not unmount
    // WindowManager/AppHost and destroy the sessions kept by minimized windows.
    use_future(move || async move {
        let mut boot_completed = matches!(*os.boot_phase.read(), BootPhase::Ready);
        loop {
            match api::health().await {
                Ok(h) => {
                    os.health.set(Some(h));
                    os.backend_online.set(true);
                    if !boot_completed {
                        os.boot_phase.set(BootPhase::LampIgnite);
                        tokio_time_sleep(1200).await;
                        os.boot_phase.set(BootPhase::DesktopReveal);
                        tokio_time_sleep(1400).await;
                        os.boot_phase.set(BootPhase::Ready);
                        boot_completed = true;
                    } else if !matches!(*os.boot_phase.read(), BootPhase::Ready) {
                        os.boot_phase.set(BootPhase::Ready);
                    }
                }
                Err(_) => {
                    os.backend_online.set(false);
                    if !boot_completed {
                        os.boot_phase.set(BootPhase::Dark);
                    }
                }
            }
            tokio_time_sleep(2000).await;
        }
    });

    // Service manifest (fetched once, rarely changes) + service list polling.
    use_future(move || async move {
        if let Ok(meta) = api::services_meta().await {
            os.meta.set(meta);
        }
        loop {
            if let Ok(services) = api::list_start_services().await {
                os.set_services(services);
            }
            tokio_time_sleep(1500).await;
        }
    });

    // Live log stream, reconnecting on drop/error.
    use_future(move || async move {
        loop {
            let url = api::start_backend_logs_url();
            if let Ok(mut es) = EventSource::new(&url) {
                if let Ok(mut stream) = es.subscribe("message") {
                    while let Some(Ok((_kind, msg))) = stream.next().await {
                        if let Some(text) = msg.data().as_string() {
                            if let Ok(entry) = serde_json::from_str::<ServiceLogEvent>(&text) {
                                os.logs.with_mut(|logs| {
                                    let buf = logs.entry(entry.service.clone()).or_default();
                                    buf.push(entry);
                                    if buf.len() > MAX_CLIENT_LOGS_PER_SERVICE {
                                        let excess = buf.len() - MAX_CLIENT_LOGS_PER_SERVICE;
                                        buf.drain(0..excess);
                                    }
                                });
                            }
                        }
                    }
                }
            }
            tokio_time_sleep(2000).await;
        }
    });
}

#[cfg(test)]
mod settings_tests {
    use super::*;
    use crate::api::ServiceConfigField;
    fn field(key: &str, kind: FieldType, default: &str) -> ServiceConfigField {
        ServiceConfigField {
            key: key.into(),
            label: key.into(),
            description: None,
            field_type: kind,
            default_value: default.into(),
            is_build_param: false,
            required: true,
            min: None,
            max: None,
            allowed_values: None,
            read_only: false,
        }
    }
    fn running_service(
        name: &str,
        platform: Option<&str>,
        public_url: Option<&str>,
    ) -> ServiceInfo {
        ServiceInfo {
            name: name.into(),
            status: ServiceStatus::Running,
            pid: None,
            platform: platform.map(str::to_owned),
            public_url: public_url.map(str::to_owned),
        }
    }

    #[test]
    fn frontend_url_is_shared_and_normalized() {
        let services = vec![running_service(
            "frontend",
            Some(" WEB "),
            Some("  http://127.0.0.1:8080/  "),
        )];
        assert_eq!(
            managed_web_frontend_url(&services).as_deref(),
            Some("http://127.0.0.1:8080")
        );
    }

    #[test]
    fn an_existing_minimized_window_can_always_be_restored() {
        let mut windows = vec![WindowState {
            id: 7,
            app_id: "sandbox".into(),
            title: "Sandbox".into(),
            x: 0.0,
            y: 0.0,
            width: 800.0,
            height: 600.0,
            min_width: 480.0,
            min_height: 320.0,
            z: 1,
            minimized: true,
            maximized: false,
            restore_rect: None,
        }];
        assert_eq!(restore_existing_window(&mut windows, "sandbox"), Some(7));
        assert!(!windows[0].minimized);
    }

    #[test]
    fn sandbox_requires_backend_and_addressable_web_frontend() {
        let web = vec![
            running_service("backend", None, None),
            running_service("frontend", Some("web"), Some("http://127.0.0.1:8080")),
        ];
        assert!(OsState::sandbox_is_available(&web));

        let missing_backend = vec![running_service(
            "frontend",
            Some("web"),
            Some("http://127.0.0.1:8080"),
        )];
        assert!(!OsState::sandbox_is_available(&missing_backend));

        let desktop = vec![
            running_service("backend", None, None),
            running_service("frontend", Some("desktop"), None),
        ];
        assert!(!OsState::sandbox_is_available(&desktop));
    }

    #[test]
    fn window_lifecycle_preserves_minimized_and_blocked_sessions() {
        assert_eq!(window_lifecycle_for(false, &[]), WindowLifecycle::Visible);
        assert_eq!(
            window_lifecycle_for(false, &["backend".into()]),
            WindowLifecycle::Blocked
        );
        assert_eq!(
            window_lifecycle_for(true, &["backend".into()]),
            WindowLifecycle::Minimized
        );
    }

    #[test]
    fn frontend_round_trip_keeps_platform_and_web_port_structured() {
        let schema = ServiceConfigSchema {
            service: "frontend".into(),
            fields: vec![
                field(
                    "LUNAR_FRONTEND_PLATFORM",
                    FieldType::Select {
                        options: vec!["web".into(), "desktop".into(), "android".into()],
                    },
                    "web",
                ),
                field("LUNAR_FRONTEND_PORT", FieldType::Port, "8080"),
                field("EXTRA_ARGS", FieldType::StringList, ""),
            ],
        };
        let values = ServiceConfigValues {
            env: HashMap::from([
                ("LUNAR_FRONTEND_PLATFORM".into(), "web".into()),
                ("LUNAR_FRONTEND_PORT".into(), "9090".into()),
            ]),
            extra_args: vec!["--hot-reload".into()],
            build_args: vec![],
        };
        let form = form_values_from_config("frontend", &schema, &values);
        assert_eq!(form["LUNAR_FRONTEND_PLATFORM"], "web");
        assert_eq!(form["LUNAR_FRONTEND_PORT"], "9090");
        assert_eq!(form["EXTRA_ARGS"], "--hot-reload");
        let round_trip = config_from_form("frontend", &schema, &values, &form);
        assert_eq!(round_trip.env["LUNAR_FRONTEND_PORT"], "9090");
        assert_eq!(round_trip.extra_args, ["--hot-reload"]);
    }

    #[test]
    fn invalid_and_conflicting_web_ports_are_blocked_client_side() {
        let schema = ServiceConfigSchema {
            service: "frontend".into(),
            fields: vec![field("LUNAR_FRONTEND_PORT", FieldType::Port, "8080")],
        };
        assert!(
            validate_form(
                &schema,
                &HashMap::from([("LUNAR_FRONTEND_PORT".into(), "70000".into())])
            )
            .contains_key("LUNAR_FRONTEND_PORT")
        );
        let configs = HashMap::from([
            (
                "backend".into(),
                ServiceConfigValues {
                    env: HashMap::from([("LUNAR_BACKEND_PORT".into(), "25255".into())]),
                    ..Default::default()
                },
            ),
            (
                "frontend".into(),
                ServiceConfigValues {
                    env: HashMap::from([
                        ("LUNAR_FRONTEND_PLATFORM".into(), "web".into()),
                        ("LUNAR_FRONTEND_PORT".into(), "25255".into()),
                    ]),
                    ..Default::default()
                },
            ),
        ]);
        assert_eq!(
            known_port_conflict("frontend", &configs).unwrap().0,
            "LUNAR_FRONTEND_PORT"
        );
    }
}
