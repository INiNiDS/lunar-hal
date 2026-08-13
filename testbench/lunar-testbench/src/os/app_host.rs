use dioxus::prelude::*;

use crate::os::manifest::{AppIcon, WindowContentMode, app_by_id, app_content};
use crate::os::state::window_lifecycle_for;
use crate::os::{WindowRuntimeContext, use_os_state};

#[component]
pub fn AppHost(window_id: u64, app_id: String, minimized: bool, title: String) -> Element {
    let mut os = use_os_state();
    let missing_deps = os.missing_app_dependencies(&app_id);

    let current_lifecycle = window_lifecycle_for(minimized, &missing_deps);
    let body_class = match app_by_id(&app_id).map(|app| app.content_mode) {
        Some(WindowContentMode::Fill) => "app-window-body--fill",
        _ => "app-window-body--scroll scrollbar-thin",
    };

    let mut lifecycle = use_signal(|| current_lifecycle);
    let lifecycle_app_id = app_id.clone();
    let lifecycle_os = os;
    use_effect(move || {
        // Subscribe to the authoritative window and service signals, then
        // publish the derived lifecycle after render. Mutating `lifecycle`
        // directly during render panics when a minimized window is restored.
        let minimized_now = lifecycle_os
            .windows
            .read()
            .iter()
            .find(|window| window.id == window_id)
            .is_some_and(|window| window.minimized);
        let missing_now = lifecycle_os.missing_app_dependencies(&lifecycle_app_id);
        let next = window_lifecycle_for(minimized_now, &missing_now);
        let previous = *lifecycle.peek();
        if previous != next {
            lifecycle.set(next);
        }
    });

    use_context_provider(|| WindowRuntimeContext {
        window_id,
        app_id: app_id.clone(),
        lifecycle,
    });

    rsx! {
        div {
            class: "app-window-body {body_class}",
            "data-testid": "app-window-body",
            "data-app-id": "{app_id}",
            { app_content(&app_id) }

            if !missing_deps.is_empty() {
                div { class: "absolute inset-0 z-50 flex items-center justify-center p-6 bg-bg-0/85 backdrop-blur-md pointer-events-auto select-none",
                    div { class: "max-w-md w-full glass-strong rounded-2xl p-6 text-center border border-white/10 shadow-2xl flex flex-col items-center gap-4",
                        div { class: "h-12 w-12 grid place-items-center rounded-2xl bg-warn/10 text-warn border border-warn/20",
                            AppIcon { app_id: app_id.clone() }
                        }
                        div {
                            h3 { class: "font-display text-sm tracking-[0.14em] uppercase text-white/90", "{title} PAUSED" }
                            p { class: "mt-2 text-xs leading-relaxed text-white/50",
                                "This application requires the following services to be running:"
                            }
                            ul { class: "mt-3 flex flex-wrap justify-center gap-2 font-mono text-[11px]",
                                for dep in missing_deps.iter() {
                                    li { key: "{dep}", class: "rounded-lg bg-white/[0.06] border border-white/10 px-2.5 py-1 text-warn/90 flex items-center gap-1.5",
                                        span { class: "h-1.5 w-1.5 rounded-full bg-warn" }
                                        "{dep}"
                                    }
                                }
                            }
                        }
                        div { class: "flex flex-wrap justify-center items-center gap-2 mt-2",
                            for dep in missing_deps.iter() {
                                {
                                    let svc_name = dep.split_whitespace().next().unwrap_or(dep).to_string();
                                    rsx! {
                                        button {
                                            key: "cfg-{svc_name}",
                                            class: "rounded-lg border border-accent/35 bg-accent/15 px-3 py-1.5 text-xs text-accent hover:bg-accent/25 transition-colors font-medium",
                                            onclick: move |_| os.open_service_settings(&svc_name),
                                            "Configure {svc_name}"
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
}
