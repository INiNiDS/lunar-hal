use dioxus::prelude::*;

use crate::os::manifest::{AppIcon, all_apps};
use crate::os::use_os_state;

/// macOS-style floating dock along the bottom edge. Only apps whose backing
/// service is up are lit; the rest stay dimmed until their service starts.
/// Nothing is pinned -- the dock always shows exactly the known app set.
#[component]
pub fn Dock() -> Element {
    let mut os = use_os_state();

    rsx! {
        div { class: "fixed bottom-4 left-1/2 -translate-x-1/2 z-40",
            div { class: "glass-strong rounded-[22px] px-3 py-2.5 flex items-end gap-1.5",
                for app in all_apps().iter() {
                    {
                        let available = os.is_app_available(app.id);
                        let is_open = os.windows.read().iter().any(|w| w.app_id == app.id);
                        let app_id = app.id;
                        let app_title = app.title;

                        let tile_class = if available {
                            "w-12 h-12 rounded-2xl grid place-items-center border border-white/10 bg-white/[0.05] text-white/85 shadow-[inset_0_1px_0_rgba(255,255,255,0.14)] transition-all duration-200 group-hover:border-white/25 group-hover:bg-white/[0.11] group-hover:text-white group-hover:shadow-[0_0_22px_rgba(110,168,255,0.28),inset_0_1px_0_rgba(255,255,255,0.2)]"
                        } else {
                            "w-12 h-12 rounded-2xl grid place-items-center border border-white/[0.06] bg-white/[0.02] text-white/20"
                        };
                        let button_class = if available {
                            "group relative flex flex-col items-center gap-1.5 transition-transform duration-200 ease-out hover:-translate-y-2 animate-dock-app-reveal"
                        } else {
                            "group relative flex flex-col items-center gap-1.5 cursor-not-allowed"
                        };

                        rsx! {
                            button {
                                key: "{app_id}",
                                class: "{button_class}",
                                disabled: !available,
                                title: "{app_title}",
                                onclick: move |_| {
                                    if available {
                                        os.open_window(app_id, app_title);
                                    }
                                },

                                // Tooltip-style label, revealed on hover like a real dock.
                                span { class: "pointer-events-none absolute -top-9 whitespace-nowrap rounded-lg glass px-2.5 py-1 font-grotesk text-[11px] text-white/85 opacity-0 transition-opacity duration-150 group-hover:opacity-100",
                                    "{app_title}"
                                }

                                div { class: "{tile_class}",
                                    div { class: "w-6 h-6",
                                        AppIcon { app_id: app_id.to_string() }
                                    }
                                }

                                span {
                                    class: if is_open {
                                        "w-1 h-1 rounded-full bg-white/80 shadow-[0_0_6px_rgba(255,255,255,0.7)]"
                                    } else {
                                        "w-1 h-1 rounded-full bg-transparent"
                                    },
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
