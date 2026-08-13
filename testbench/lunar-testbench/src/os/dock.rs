use crate::os::manifest::AppIcon;
use crate::os::use_os_state;
use dioxus::prelude::*;
use std::collections::HashSet;

#[component]
pub fn Dock() -> Element {
    let mut os = use_os_state();
    let mut seen = HashSet::new();
    let windows = os
        .windows
        .read()
        .iter()
        .filter(|window| seen.insert(window.app_id.clone()))
        .cloned()
        .collect::<Vec<_>>();

    if windows.is_empty() {
        return rsx! {};
    }

    rsx! {
        div { class: "fixed bottom-4 left-1/2 z-40 -translate-x-1/2",
            div { class: "glass-strong flex items-end gap-1.5 rounded-[22px] px-3 py-2.5",
                for window in windows {
                    {
                        let app_id = window.app_id.clone();
                        let icon_id = app_id.clone();
                        let title = window.title.clone();
                        let minimized = window.minimized;
                        rsx! {
                            button {
                                key: "{app_id}",
                                class: "group relative flex flex-col items-center gap-1.5 transition-transform duration-200 hover:-translate-y-2 animate-dock-app-reveal",
                                "data-testid": "dock-app-{app_id}",
                                title: "{title}",
                                onclick: move |_| os.activate_app(&app_id),
                                span { class: "pointer-events-none absolute -top-9 whitespace-nowrap rounded-lg glass px-2.5 py-1 text-[11px] text-white/85 opacity-0 group-hover:opacity-100",
                                    "{title}"
                                }
                                div {
                                    class: if minimized {
                                        "grid h-12 w-12 place-items-center rounded-2xl border border-white/[0.06] bg-white/[0.025] text-white/35"
                                    } else {
                                        "grid h-12 w-12 place-items-center rounded-2xl border border-white/10 bg-white/[0.07] text-white/90 shadow-[0_0_22px_rgba(110,168,255,0.18)]"
                                    },
                                    div { class: "h-6 w-6", AppIcon { app_id: icon_id } }
                                }
                                span {
                                    class: if minimized {
                                        "h-1 w-1 rounded-full bg-white/20"
                                    } else {
                                        "h-1 w-1 rounded-full bg-accent shadow-[0_0_6px_rgba(110,168,255,0.8)]"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
