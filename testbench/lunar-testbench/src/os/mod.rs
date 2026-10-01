
pub mod app_host;
pub mod category_lamp;
pub mod desktop;
pub mod dock;
pub mod lamp;
pub mod led;
pub mod log_window;
pub mod manifest;
pub mod ram;
pub mod room;
pub mod service_settings;
pub mod snapshot;
pub mod state;
pub mod state_inventory;
pub mod viewport;
pub mod window;
pub mod window_manager;

pub use ram::{
    LunarOsRamEntry, LunarOsRamEntryV1, LunarOsRamStore, RamEntryState, RamLifecycleState,
};
pub use snapshot::{AppSnapshot, AppSnapshotEnvelopeV1, WindowGeometry, WindowSnapshotV1};

pub use room::Room;
pub use state::{
    BootPhase, DragKind, OsState, WindowLifecycle, WindowRuntimeContext, WindowState, use_os_state,
    use_window_instance_id, use_window_instance_snapshot, use_window_lifecycle, use_window_runtime,
    use_window_snapshot_payload,
};

pub fn viewport_size() -> (f64, f64) {
    match web_sys::window() {
        Some(w) => {
            if let Some(viewport) = w.visual_viewport() {
                let width = viewport.width();
                let height = viewport.height();
                if width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0 {
                    return (width, height);
                }
            }
            let width = w
                .inner_width()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(1280.0);
            let height = w
                .inner_height()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(800.0);
            (width, height)
        }
        None => (1280.0, 800.0),
    }
}
