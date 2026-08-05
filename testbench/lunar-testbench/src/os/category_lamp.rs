use crate::api::ServiceStatus;
use crate::os::manifest::AppCategory;
use crate::os::use_os_state;
use dioxus::prelude::*;

#[component]
pub fn CategoryLamp(category: AppCategory) -> Element {
    let mut os = use_os_state();
    let service = category.service();
    let status = os.service_status(service);
    let running = status.as_ref().is_some_and(ServiceStatus::is_running);
    let starting = matches!(status, Some(ServiceStatus::Starting));
    let failed = matches!(status, Some(ServiceStatus::Failed { .. }));
    let status_label = status
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "unknown".into());
    let epoch = os.service_reveal_epoch(service);
    let error = os.service_action_errors.read().get(service).cloned();
    let settings_name = service.to_string();
    let log_name = service.to_string();
    let restart_name = service.to_string();
    let stop_name = service.to_string();
    let title = category.title();
    rsx! {
        div { class: "relative flex min-w-0 flex-col items-center",
            div { key: "light-{service}-{epoch}", class: if running { "category-light category-light-on" } else if starting { "category-light category-light-starting" } else if failed { "category-light category-light-failed" } else { "category-light" } }
            div { key: "lamp-{service}-{epoch}", class: if running { "category-lamp-fixture category-lamp-ignite" } else { "category-lamp-fixture" }, div { class: "category-lamp-bulb" } }
            div { class: "relative z-10 mt-2 flex items-center gap-2",
                button { class: "group flex items-center gap-2 rounded-lg px-2 py-1 text-left transition-colors hover:bg-white/[0.06] disabled:cursor-wait", disabled: starting, title: "{status_label}",
                    onclick: move |_| if running { os.open_window(&format!("log:{log_name}"), &format!("{title} — Logs")); } else { os.open_service_settings(&settings_name); },
                    span { class: if running { "led h-2 w-2 text-ok" } else if starting { "led h-2 w-2 animate-led-pulse text-accent" } else if failed { "led h-2 w-2 text-err" } else { "led h-2 w-2 text-white/15" } }
                    span { span { class: "block truncate text-[12px] text-white/80", "{title}" } span { class: "block truncate font-mono text-[8px] uppercase tracking-[0.16em] text-white/30", "{status_label}" } }
                }
                button { class: "h-7 w-7 rounded-md text-white/30 hover:bg-white/10 hover:text-white/80", title: if running { "View effective configuration" } else { "Configure and start" }, onclick: move |_| os.open_service_settings(service), "⚙" }
                if running {
                    button { class: "h-7 w-7 rounded-md text-white/30 hover:bg-white/10 hover:text-white/80", title: "Restart with saved configuration", onclick: move |_| os.restart_managed_service(&restart_name), "↻" }
                    button { class: "h-7 w-7 rounded-md text-white/30 hover:bg-err/15 hover:text-err", title: "Stop", onclick: move |_| os.stop_managed_service(&stop_name), "■" }
                }
            }
            if let Some(error) = error { p { class: "relative z-10 mt-1 max-w-56 truncate font-mono text-[9px] text-err/80", title: "{error}", "{error}" } }
        }
    }
}
