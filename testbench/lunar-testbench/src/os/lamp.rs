use dioxus::prelude::*;

use crate::os::{BootPhase, use_os_state};

/// The single overhead lamp that lights the room. Its cone is what actually
/// "reveals" the service rack below, so the fixture is horizontally centred on
/// the same axis as the rack.
#[component]
pub fn Lamp() -> Element {
    let os = use_os_state();
    let phase = *os.boot_phase.read();
    let lit = !matches!(phase, BootPhase::Dark);

    let cone_class = match phase {
        BootPhase::Dark => "lamp-cone opacity-0",
        BootPhase::LampIgnite => "lamp-cone animate-lamp-ignite",
        BootPhase::RackReveal | BootPhase::Ready => "lamp-cone",
    };

    rsx! {
        div { class: "absolute inset-x-0 top-0 z-10 pointer-events-none",
            div { class: "{cone_class}" }

            div { class: "relative flex flex-col items-center",
                // Suspension cord.
                div { class: "w-px h-20 bg-gradient-to-b from-white/[0.04] via-white/10 to-white/25" }
                // Shade.
                div {
                    class: if lit {
                        "relative w-16 h-5 rounded-b-[999px] border-x border-b border-white/20 bg-gradient-to-b from-bg-4 to-bg-1 shadow-[0_8px_26px_rgba(0,0,0,0.85)]"
                    } else {
                        "relative w-16 h-5 rounded-b-[999px] border-x border-b border-white/[0.06] bg-bg-1"
                    },
                    if lit {
                        div { class: "lamp-bulb animate-halo-breathe" }
                    }
                }
            }
        }
    }
}
