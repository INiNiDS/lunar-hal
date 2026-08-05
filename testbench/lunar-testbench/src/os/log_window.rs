use dioxus::prelude::*;

use crate::api::{self, LogLevel};
use crate::os::use_os_state;

/// Terminal-style log viewer hosted inside a normal OS window.
#[component]
pub fn LogWindow(service: String) -> Element {
    let mut os = use_os_state();
    let svc = service.clone();

    use_effect(move || {
        let svc = svc.clone();
        spawn(async move {
            let already_has_logs = os
                .logs
                .read()
                .get(&svc)
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            if !already_has_logs {
                if let Ok(tail) = api::service_log_tail(&svc, 200).await {
                    os.logs.with_mut(|logs| {
                        logs.entry(svc.clone()).or_default().extend(tail);
                    });
                }
            }
        });
    });

    let lines = os.logs.read().get(&service).cloned().unwrap_or_default();
    let warn_count = lines.iter().filter(|l| l.level == LogLevel::Warn).count();
    let err_count = lines.iter().filter(|l| l.level == LogLevel::Error).count();

    rsx! {
        div { class: "flex flex-col h-full bg-bg-0/70 font-mono text-[11.5px] leading-relaxed",
            div { class: "flex items-center gap-3 px-4 py-2 shrink-0 border-b border-white/[0.07] bg-white/[0.02]",
                span { class: "text-white/50 tracking-wide", "{service}" }
                span { class: "flex-1" }
                if err_count > 0 {
                    span { class: "rounded-full bg-err/15 px-2 py-0.5 text-[10px] text-err",
                        "{err_count} err"
                    }
                }
                if warn_count > 0 {
                    span { class: "rounded-full bg-warn/15 px-2 py-0.5 text-[10px] text-warn",
                        "{warn_count} warn"
                    }
                }
                span { class: "text-[10px] text-white/25", "{lines.len()} lines" }
            }

            div { class: "flex-1 overflow-y-auto scrollbar-thin px-4 py-3",
                if lines.is_empty() {
                    div { class: "text-white/25", "waiting for log output..." }
                } else {
                    for (i , line) in lines.iter().enumerate() {
                        div {
                            key: "{i}",
                            class: match line.level {
                                LogLevel::Error => "flex gap-3 -mx-2 px-2 rounded bg-err/[0.07] text-err/90",
                                LogLevel::Warn => "flex gap-3 -mx-2 px-2 rounded text-warn/90",
                                LogLevel::Info => "flex gap-3 -mx-2 px-2 rounded text-white/70",
                            },
                            span { class: "shrink-0 tabular-nums text-white/20", "{line.timestamp}" }
                            span { class: "whitespace-pre-wrap break-all", "{line.text}" }
                        }
                    }
                }
            }
        }
    }
}
