use dioxus::prelude::*;

use crate::os::dock::Dock;
use crate::os::lamp::Lamp;
use crate::os::rack::Rack;
use crate::os::state::{BootPhase, provide_os_state, use_os_runtime};
use crate::os::window_manager::WindowManager;

/// The whole WebOS scene: a dark room lit by one overhead lamp, with the
/// service rack sitting directly under the light cone and the dock along the
/// bottom edge.
#[component]
pub fn Room() -> Element {
    let os = provide_os_state();
    use_os_runtime();

    let phase = *os.boot_phase.read();
    let dark = matches!(phase, BootPhase::Dark);
    let rack_lit = matches!(phase, BootPhase::RackReveal | BootPhase::Ready);

    rsx! {
        div { class: "relative w-screen h-screen overflow-hidden bg-bg-0 font-grotesk text-white/90 select-none antialiased",
            div { class: "room-floor" }

            Lamp {}

            if dark {
                div { class: "absolute inset-0 z-20 flex flex-col items-center justify-center gap-4",
                    span { class: "font-display text-3xl tracking-[0.45em] text-white/[0.12] pl-[0.45em]",
                        "LUNAR"
                    }
                    span { class: "font-mono text-[10px] uppercase tracking-[0.3em] text-white/25 animate-pulse",
                        "waiting for lunar-start-backend"
                    }
                }
            }

            if !dark {
                div {
                    class: if rack_lit {
                        "absolute inset-x-0 top-[34vh] z-20 flex justify-center px-10 animate-rack-reveal"
                    } else {
                        "absolute inset-x-0 top-[34vh] z-20 flex justify-center px-10 opacity-0"
                    },
                    Rack {}
                }
            }

            // Vignette sits above the scene but below the interactive chrome so
            // it darkens the room edges without dimming windows or the dock.
            div { class: "room-vignette z-[25]" }

            if matches!(phase, BootPhase::Ready) {
                WindowManager {}
                Dock {}
            }
        }
    }
}
