use dioxus::prelude::*;

use crate::os::window::Window;
use crate::os::{use_os_state, viewport_size};

/// Renders all open windows and hosts the full-viewport overlay that turns an
/// in-flight drag into window rect updates.
#[component]
pub fn WindowManager() -> Element {
    let mut os = use_os_state();
    let windows = os.windows.read().clone();
    let dragging = os.drag.read().is_some();

    rsx! {
        div { class: "absolute inset-0 z-30 pointer-events-none",
            for win in windows.into_iter() {
                Window { key: "{win.id}", win }
            }
        }
        if dragging {
            div {
                class: "fixed inset-0 z-[999] cursor-move",
                onmousemove: move |e| {
                    let p = e.client_coordinates();
                    os.update_drag(p.x, p.y);
                },
                onmouseup: move |e| {
                    let p = e.client_coordinates();
                    let (vw, vh) = viewport_size();
                    os.end_drag(vw, vh, p.x, p.y);
                },
                onmouseleave: move |e| {
                    let p = e.client_coordinates();
                    let (vw, vh) = viewport_size();
                    os.end_drag(vw, vh, p.x, p.y);
                },
            }
        }
    }
}
