use dioxus::prelude::*;

use crate::api::{self, ServiceMeta, ServiceStatus};
use crate::os::led::Led;
use crate::os::use_os_state;

/// One equal-width slot in the service rack.
///
/// Interaction follows the WebOS spec: the first press on a stopped service
/// starts it, and once it is running the same press opens its log window.
/// Stop/restart live in a hover-revealed corner cluster so the resting state
/// stays clean instead of looking like an admin panel.
#[component]
pub fn RackSection(meta: ServiceMeta) -> Element {
    let mut os = use_os_state();
    let status = os.service_status(&meta.name);

    let running = status.as_ref().is_some_and(ServiceStatus::is_running);
    let starting = matches!(status, Some(ServiceStatus::Starting));
    let failed = matches!(status, Some(ServiceStatus::Failed { .. }));

    let hint = match &status {
        Some(ServiceStatus::Running) => "open logs".to_string(),
        Some(ServiceStatus::Starting) => "starting...".to_string(),
        Some(ServiceStatus::Failed { reason }) => reason.clone(),
        _ => "press to start".to_string(),
    };

    let name_press = meta.name.clone();
    let title_press = meta.title.clone();
    let name_stop = meta.name.clone();
    let name_restart = meta.name.clone();

    let title_class = if running {
        "font-grotesk text-[13px] leading-tight text-white/90 truncate w-full"
    } else {
        "font-grotesk text-[13px] leading-tight text-white/45 truncate w-full"
    };
    let hint_class = if failed {
        "font-mono text-[10px] text-err/70 truncate w-full"
    } else if starting {
        "font-mono text-[10px] text-accent/70 truncate w-full"
    } else {
        "font-mono text-[10px] text-white/25 truncate w-full"
    };

    rsx! {
        div { class: "group relative flex-1 basis-0 min-w-0",
            button {
                class: "w-full h-full px-4 pt-5 pb-4 flex flex-col items-center gap-1.5 text-center transition-colors duration-200 hover:bg-white/[0.05] focus:outline-none focus-visible:bg-white/[0.07]",
                title: "{meta.description}",
                onclick: move |_| {
                    if running {
                        os.open_window(
                            &format!("log:{name_press}"),
                            &format!("{title_press} \u{2014} Logs"),
                        );
                    } else {
                        let n = name_press.clone();
                        spawn(async move {
                            let _ = api::start_service(&n).await;
                        });
                    }
                },

                span { class: "font-mono text-[9px] uppercase tracking-[0.22em] text-white/20",
                    "{meta.icon}"
                }
                span { class: "{title_class}", "{meta.title}" }
                span { class: "{hint_class}", "{hint}" }

                // Status lamp sits at the bottom of the slot, as specified.
                div { class: "mt-auto pt-4",
                    Led { status: status.clone() }
                }
            }

            if running || failed {
                div { class: "absolute top-2 right-2 flex gap-1 opacity-0 transition-opacity duration-200 group-hover:opacity-100",
                    button {
                        class: "w-6 h-6 grid place-items-center rounded-md text-white/40 hover:text-white hover:bg-white/10 transition-colors",
                        title: "Restart",
                        onclick: move |e| {
                            e.stop_propagation();
                            let n = name_restart.clone();
                            spawn(async move {
                                let _ = api::restart_service(&n).await;
                            });
                        },
                        svg {
                            class: "w-3.5 h-3.5",
                            view_box: "0 0 24 24",
                            fill: "none",
                            stroke: "currentColor",
                            stroke_width: "1.6",
                            stroke_linecap: "round",
                            stroke_linejoin: "round",
                            path { d: "M20 11a8 8 0 1 0-2.3 6.1" }
                            path { d: "M20 5v6h-6" }
                        }
                    }
                    button {
                        class: "w-6 h-6 grid place-items-center rounded-md text-white/40 hover:text-err hover:bg-err/15 transition-colors",
                        title: "Stop",
                        onclick: move |e| {
                            e.stop_propagation();
                            let n = name_stop.clone();
                            spawn(async move {
                                let _ = api::stop_service(&n).await;
                            });
                        },
                        svg {
                            class: "w-3 h-3",
                            view_box: "0 0 24 24",
                            fill: "currentColor",
                            rect { x: "6", y: "6", width: "12", height: "12", rx: "2" }
                        }
                    }
                }
            }
        }
    }
}
