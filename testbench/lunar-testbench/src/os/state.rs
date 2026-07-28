use std::collections::HashMap;

use dioxus::prelude::*;
use futures_util::StreamExt;
use gloo_net::eventsource::futures::EventSource;

use crate::api::{self, HealthResponse, ServiceInfo, ServiceLogEvent, ServiceMeta, ServiceStatus};
use crate::components::ui::tokio_time_sleep;

/// Coarse boot sequence for the "black room" intro: everything starts dark,
/// then the overhead lamp ignites once `lunar-start-backend` answers
/// `/health`, then the service rack reveals, then the desk/dock become
/// interactive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootPhase {
    Dark,
    LampIgnite,
    RackReveal,
    Ready,
}

/// Max log lines retained client-side per service (older lines are dropped;
/// the server keeps its own ring buffer, re-fetchable via `/services/{name}/logs`).
const MAX_CLIENT_LOGS_PER_SERVICE: usize = 500;

/// A single open (or minimized) OS window hosting one app/page.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowState {
    pub id: u64,
    /// Identifies which app/page is hosted in this window (e.g. "dashboard").
    /// Windows are singleton per `app_id`, like reopening a macOS app.
    pub app_id: String,
    pub title: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub z: i32,
    pub minimized: bool,
    pub maximized: bool,
    /// Rect saved from before maximizing/snapping, restored when un-maximized.
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
}

const MIN_WINDOW_W: f64 = 280.0;
const MIN_WINDOW_H: f64 = 160.0;
/// Distance from a screen edge (in px) that triggers Windows-style snapping
/// when a window drag is released there.
const SNAP_EDGE_PX: f64 = 16.0;

/// Global reactive state for the WebOS shell, installed into context once by
/// [`provide_os_state`] and read anywhere via [`use_os_state`].
///
/// All fields are Signals. `OsState` itself is a cheap `Copy` handle; reading
/// only needs `&self`, but Dioxus's `Signal::set`/`with_mut` require `&mut
/// self`, so mutating methods below take `&mut self` and callers keep their
/// local `os` binding as `let mut os = use_os_state();`.
#[derive(Clone, Copy)]
pub struct OsState {
    /// Whether `lunar-start-backend` answered `/health` on the last poll.
    pub backend_online: Signal<bool>,
    pub boot_phase: Signal<BootPhase>,
    pub services: Signal<Vec<ServiceInfo>>,
    pub meta: Signal<Vec<ServiceMeta>>,
    /// Buffered log lines per service name, newest last.
    pub logs: Signal<HashMap<String, Vec<ServiceLogEvent>>>,
    pub health: Signal<Option<HealthResponse>>,
    pub windows: Signal<Vec<WindowState>>,
    pub next_z: Signal<i32>,
    pub drag: Signal<Option<DragOp>>,
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
            next_window_id: Signal::new(1),
        }
    }

    /// Name of the service (if any) that `provides` a given dock app id.
    /// Apps with no owning service (e.g. Sandbox) are always available.
    pub fn service_for_app(&self, app_id: &str) -> Option<String> {
        self.meta
            .read()
            .iter()
            .find(|m| m.provides.iter().any(|p| p.as_str() == app_id))
            .map(|m| m.name.clone())
    }

    /// Whether a dock app should currently be enabled/lit: true if it has no
    /// owning service, or its owning service is running.
    pub fn is_app_available(&self, app_id: &str) -> bool {
        match self.service_for_app(app_id) {
            None => true,
            Some(service_name) => self
                .services
                .read()
                .iter()
                .any(|s| s.name == service_name && s.status.is_running()),
        }
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

    /// Opens a new window for `app_id`, or un-minimizes and focuses the
    /// existing one if it's already open.
    pub fn open_window(&mut self, app_id: &str, title: &str) {
        let existing_id = self
            .windows
            .read()
            .iter()
            .find(|w| w.app_id == app_id)
            .map(|w| w.id);
        if let Some(id) = existing_id {
            self.windows.with_mut(|ws| {
                if let Some(w) = ws.iter_mut().find(|w| w.id == id) {
                    w.minimized = false;
                }
            });
            self.focus_window(id);
            return;
        }

        let id = *self.next_window_id.read();
        self.next_window_id.set(id + 1);
        let z = *self.next_z.read();
        self.next_z.set(z + 1);

        // Cascade new windows slightly so they don't perfectly overlap.
        let count = self.windows.read().len() as f64;
        let offset = (count % 8.0) * 28.0;

        self.windows.with_mut(|ws| {
            ws.push(WindowState {
                id,
                app_id: app_id.to_string(),
                title: title.to_string(),
                x: 120.0 + offset,
                y: 90.0 + offset,
                width: 860.0,
                height: 600.0,
                z,
                minimized: false,
                maximized: false,
                restore_rect: None,
            });
        });
    }

    pub fn close_window(&mut self, id: u64) {
        self.windows.with_mut(|ws| ws.retain(|w| w.id != id));
    }

    pub fn minimize_window(&mut self, id: u64) {
        self.windows.with_mut(|ws| {
            if let Some(w) = ws.iter_mut().find(|w| w.id == id) {
                w.minimized = true;
            }
        });
    }

    /// Toggles between the window's free-floating rect and a full-viewport
    /// rect, remembering the previous rect so it can be restored.
    pub fn toggle_maximize_window(&mut self, id: u64, viewport_w: f64, viewport_h: f64) {
        self.windows.with_mut(|ws| {
            if let Some(w) = ws.iter_mut().find(|w| w.id == id) {
                if w.maximized {
                    if let Some((x, y, width, height)) = w.restore_rect.take() {
                        w.x = x;
                        w.y = y;
                        w.width = width;
                        w.height = height;
                    }
                    w.maximized = false;
                } else {
                    w.restore_rect = Some((w.x, w.y, w.width, w.height));
                    w.x = 0.0;
                    w.y = 0.0;
                    w.width = viewport_w;
                    w.height = viewport_h;
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

    /// Starts a move/resize gesture from a window's current rect; the actual
    /// rect updates happen in [`OsState::update_drag`] as the mouse moves.
    pub fn begin_drag(&mut self, window_id: u64, kind: DragKind, mouse_x: f64, mouse_y: f64) {
        let rect = self
            .windows
            .read()
            .iter()
            .find(|w| w.id == window_id)
            .map(|w| (w.x, w.y, w.width, w.height));
        let Some((x, y, w, h)) = rect else { return };
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
        }));
    }

    /// Applies the in-flight drag/resize gesture (if any) for the current
    /// mouse position.
    pub fn update_drag(&mut self, mouse_x: f64, mouse_y: f64) {
        let Some(op) = *self.drag.read() else { return };
        let dx = mouse_x - op.start_mouse_x;
        let dy = mouse_y - op.start_mouse_y;

        let (mut x, mut y, mut w, mut h) = (op.start_x, op.start_y, op.start_w, op.start_h);
        match op.kind {
            DragKind::Move => {
                x += dx;
                y += dy;
            }
            DragKind::ResizeE => w = (op.start_w + dx).max(MIN_WINDOW_W),
            DragKind::ResizeW => {
                w = (op.start_w - dx).max(MIN_WINDOW_W);
                x = op.start_x + (op.start_w - w);
            }
            DragKind::ResizeS => h = (op.start_h + dy).max(MIN_WINDOW_H),
            DragKind::ResizeN => {
                h = (op.start_h - dy).max(MIN_WINDOW_H);
                y = op.start_y + (op.start_h - h);
            }
            DragKind::ResizeSE => {
                w = (op.start_w + dx).max(MIN_WINDOW_W);
                h = (op.start_h + dy).max(MIN_WINDOW_H);
            }
            DragKind::ResizeSW => {
                w = (op.start_w - dx).max(MIN_WINDOW_W);
                x = op.start_x + (op.start_w - w);
                h = (op.start_h + dy).max(MIN_WINDOW_H);
            }
            DragKind::ResizeNE => {
                w = (op.start_w + dx).max(MIN_WINDOW_W);
                h = (op.start_h - dy).max(MIN_WINDOW_H);
                y = op.start_y + (op.start_h - h);
            }
            DragKind::ResizeNW => {
                w = (op.start_w - dx).max(MIN_WINDOW_W);
                x = op.start_x + (op.start_w - w);
                h = (op.start_h - dy).max(MIN_WINDOW_H);
                y = op.start_y + (op.start_h - h);
            }
        }

        self.windows.with_mut(|ws| {
            if let Some(win) = ws.iter_mut().find(|win| win.id == op.window_id) {
                win.x = x;
                win.y = y;
                win.width = w;
                win.height = h;
                // A move/resize away from the maximized rect effectively
                // "un-maximizes" the window.
                win.maximized = false;
            }
        });
    }

    /// Ends the in-flight drag/resize gesture, applying Windows-style edge
    /// snapping for plain window moves released near a screen edge.
    pub fn end_drag(&mut self, viewport_w: f64, viewport_h: f64, mouse_x: f64, mouse_y: f64) {
        let Some(op) = *self.drag.read() else { return };
        self.drag.set(None);
        if op.kind != DragKind::Move {
            return;
        }

        let win_id = op.window_id;
        if mouse_y <= SNAP_EDGE_PX {
            self.snap_rect(win_id, 0.0, 0.0, viewport_w, viewport_h);
        } else if mouse_x <= SNAP_EDGE_PX {
            self.snap_rect(win_id, 0.0, 0.0, viewport_w / 2.0, viewport_h);
        } else if mouse_x >= viewport_w - SNAP_EDGE_PX {
            self.snap_rect(win_id, viewport_w / 2.0, 0.0, viewport_w / 2.0, viewport_h);
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

    // Health polling -> boot phase state machine.
    use_future(move || async move {
        loop {
            match api::health().await {
                Ok(h) => {
                    os.health.set(Some(h));
                    if !*os.backend_online.read() {
                        os.backend_online.set(true);
                        os.boot_phase.set(BootPhase::LampIgnite);
                        tokio_time_sleep(1200).await;
                        os.boot_phase.set(BootPhase::RackReveal);
                        tokio_time_sleep(1400).await;
                        os.boot_phase.set(BootPhase::Ready);
                    }
                }
                Err(_) => {
                    os.backend_online.set(false);
                    os.boot_phase.set(BootPhase::Dark);
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
                os.services.set(services);
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
