use dioxus::prelude::*;

use crate::api::ServiceStatus;

/// Small glowing status indicator used in the service rack.
///
/// The `.led` component class paints a solid core from `currentColor` plus a
/// blurred halo pseudo-element, so the colour is chosen purely with a Tailwind
/// text colour here.
#[component]
pub fn Led(status: Option<ServiceStatus>) -> Element {
    let (class, label) = match &status {
        None => ("led w-2.5 h-2.5 shrink-0 text-white/15", "unknown"),
        Some(ServiceStatus::Running) => ("led w-2.5 h-2.5 shrink-0 text-ok", "running"),
        Some(ServiceStatus::Starting) => (
            "led w-2.5 h-2.5 shrink-0 text-accent animate-led-pulse",
            "starting",
        ),
        Some(ServiceStatus::Stopped { .. }) => {
            ("led w-2.5 h-2.5 shrink-0 text-white/15", "stopped")
        }
        Some(ServiceStatus::Failed { .. }) => (
            "led w-2.5 h-2.5 shrink-0 text-err animate-led-pulse",
            "failed",
        ),
    };

    rsx! {
        span { class: "{class}", title: "{label}" }
    }
}
