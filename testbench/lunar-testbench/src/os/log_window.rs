use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::api::{self, LogLevel};
use crate::os::{AppSnapshot, use_os_state, use_window_instance_id, use_window_instance_snapshot};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LogWindowSnapshot {
    pub service: String,
    pub scroll_offset: f64,
}

impl Default for LogWindowSnapshot {
    fn default() -> Self {
        Self {
            service: String::new(),
            scroll_offset: 0.0,
        }
    }
}

impl AppSnapshot for LogWindowSnapshot {
    fn capture_snapshot(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn hydrate_snapshot(&mut self, payload: &serde_json::Value) -> Result<(), String> {
        let snap: LogWindowSnapshot = serde_json::from_value(payload.clone())
            .map_err(|e| format!("LogWindow hydration failed: {e}"))?;
        *self = snap;
        Ok(())
    }
}

/// Terminal-style log viewer hosted inside a normal OS window.
#[component]
pub fn LogWindow(service: String) -> Element {
    let mut os = use_os_state();
    let initial =
        use_window_instance_snapshot::<LogWindowSnapshot>().unwrap_or_else(|| LogWindowSnapshot {
            service: service.clone(),
            scroll_offset: 0.0,
        });
    let scroll_offset = use_signal(|| initial.scroll_offset);
    let app_id = format!("log:{service}");

    let instance_id = use_window_instance_id();
    {
        let svc_for_snapshot = service.clone();
        let inst_id = instance_id.clone();
        use_effect(move || {
            let snap = LogWindowSnapshot {
                service: svc_for_snapshot.clone(),
                scroll_offset: scroll_offset(),
            };
            if let Some(id) = &inst_id {
                os.register_instance_snapshot(id, snap.capture_snapshot());
            }
            os.register_app_snapshot(&app_id, snap.capture_snapshot());
        });
    }

    let svc = service.clone();

    use_future(move || {
        let svc = svc.clone();
        async move {
            let already_has_logs = os
                .logs
                .peek()
                .get(&svc)
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            if !already_has_logs {
                if let Ok(tail) = api::service_log_tail(&svc, 200).await {
                    if !tail.is_empty() {
                        os.logs.with_mut(|logs| {
                            let buf = logs.entry(svc.clone()).or_default();
                            if buf.is_empty() {
                                *buf = tail;
                            }
                        });
                    }
                }
            }
        }
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
