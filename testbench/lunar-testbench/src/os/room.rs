use crate::os::desktop::Desktop;
use crate::os::dock::Dock;
use crate::os::lamp::Lamp;
use crate::os::service_settings::ServiceSettings;
use crate::os::state::{BootPhase, provide_os_state, use_os_runtime};
use crate::os::viewport::use_viewport_resize;
use crate::os::window_manager::WindowManager;
use dioxus::prelude::*;

#[component]
pub fn Room() -> Element {
    let os = provide_os_state();
    use_os_runtime();
    use_viewport_resize();

    let phase = *os.boot_phase.read();
    let dark = matches!(phase, BootPhase::Dark);
    let revealed = matches!(phase, BootPhase::DesktopReveal | BootPhase::Ready);
    rsx! {
        div { class: "relative h-screen w-screen select-none overflow-hidden bg-bg-0 font-grotesk text-white/90 antialiased",
            div { class: "room-floor" }
            Lamp {}

            if dark {
                div { class: "absolute inset-0 z-20 flex flex-col items-center justify-center gap-4",
                    span { class: "pl-[0.45em] font-display text-3xl tracking-[0.45em] text-white/[0.12]",
                        "LUNAR"
                    }
                    span { class: "animate-pulse font-mono text-[10px] uppercase tracking-[0.3em] text-white/25",
                        "waiting for lunar-start-backend"
                    }
                }
            }

            if !dark {
                div {
                    class: if revealed {
                        "absolute inset-0 z-20 animate-desktop-reveal"
                    } else {
                        "absolute inset-0 z-20 opacity-0"
                    },
                    Desktop {}
                }
            }

            div { class: "room-vignette pointer-events-none z-[25]" }

            if matches!(phase, BootPhase::Ready) {
                WindowManager {}
                Dock {}
                ServiceSettings {}
            }
        }
    }
}
