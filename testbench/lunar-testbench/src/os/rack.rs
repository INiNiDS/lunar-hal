use dioxus::prelude::*;

use crate::os::rack_section::RackSection;
use crate::os::use_os_state;

/// The service rack: one horizontal slab under the lamp, split into equal-width
/// sections (one per service).
#[component]
pub fn Rack() -> Element {
    let os = use_os_state();
    let metas = os.meta.read().clone();
    let running = os
        .services
        .read()
        .iter()
        .filter(|s| s.status.is_running())
        .count();
    let total = metas.len();

    rsx! {
        div { class: "glass-strong w-full max-w-5xl rounded-2xl overflow-hidden",
            div { class: "flex items-baseline justify-between px-6 pt-5 pb-4",
                span { class: "font-display text-[13px] tracking-[0.34em] text-white/75",
                    "SERVICE RACK"
                }
                span { class: "font-mono text-[10px] tracking-[0.18em] text-white/30",
                    "{running}/{total} RUNNING"
                }
            }

            if metas.is_empty() {
                div { class: "px-6 pb-7 font-mono text-[11px] tracking-wide text-white/25",
                    "no services reported"
                }
            } else {
                div { class: "flex items-stretch border-t border-white/[0.08] divide-x divide-white/[0.08]",
                    for m in metas.into_iter() {
                        RackSection { key: "{m.name}", meta: m }
                    }
                }
            }
        }
    }
}
