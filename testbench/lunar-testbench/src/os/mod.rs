//! The WebOS shell: a single-room desktop scene (lamp + service rack + dock +
//! floating windows) that replaces the old sidebar/router navigation. See
//! `state.rs` for the shared reactive state.
//!
//! There is deliberately no separate taskbar: the dock is the single place that
//! lists apps, shows which ones are open, and restores minimized windows.

pub mod dock;
pub mod lamp;
pub mod led;
pub mod log_window;
pub mod manifest;
pub mod rack;
pub mod rack_section;
pub mod room;
pub mod state;
pub mod window;
pub mod window_manager;

pub use room::Room;
pub use state::{BootPhase, DragKind, OsState, WindowState, use_os_state};

/// Current browser viewport size in CSS pixels, used for maximize/snap math.
/// Falls back to a reasonable desktop default if unavailable.
pub fn viewport_size() -> (f64, f64) {
    match web_sys::window() {
        Some(w) => {
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
