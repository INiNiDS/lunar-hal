use crate::api::{self, ServiceInfo, ServiceLogEvent};
use crate::components::ui::{PageHeader, StatusDot, Tag, tokio_time_sleep};
use dioxus::prelude::*;
use futures_util::StreamExt;
use gloo_net::eventsource::futures::EventSource;

const MAX_LOG_LINES: usize = 2000;

fn status_kind(status: &str) -> String {
    let s = status.to_lowercase();
    if s.contains("running") {
        "ok".to_string()
    } else if s.contains("starting") {
        "busy".to_string()
    } else if s.contains("failed") {
        "err".to_string()
    } else {
        "off".to_string()
    }
}

#[component]
pub fn Services() -> Element {
    let services = use_resource(|| async { api::list_start_services().await });
    let mut logs = use_signal(Vec::<ServiceLogEvent>::new);
    let mut sse_state = use_signal(|| "connecting".to_string());
    let mut log_filter = use_signal(|| None::<String>);
    let mut error = use_signal(|| None::<String>);

    // Poll the service list periodically.
    use_future(move || async move {
        loop {
            tokio_time_sleep(2000).await;
            services.restart();
        }
    });

    // Keep a live log stream open against lunar-start-backend, reconnecting on drop/error.
    use_future(move || async move {
        loop {
            let url = api::start_backend_logs_url();
            match EventSource::new(&url) {
                Ok(mut es) => {
                    sse_state.set("connected".to_string());
                    if let Ok(mut stream) = es.subscribe("message") {
                        while let Some(Ok((_kind, msg))) = stream.next().await {
                            if let Some(text) = msg.data().as_string() {
                                if let Ok(entry) = serde_json::from_str::<ServiceLogEvent>(&text) {
                                    logs.with_mut(|l| {
                                        l.push(entry);
                                        if l.len() > MAX_LOG_LINES {
                                            let excess = l.len() - MAX_LOG_LINES;
                                            l.drain(0..excess);
                                        }
                                    });
                                }
                            }
                        }
                    }
                    sse_state.set("disconnected".to_string());
                }
                Err(_) => {
                    sse_state.set("error".to_string());
                }
            }
            tokio_time_sleep(2000).await;
        }
    });

    let services_snapshot = services.cloned();
    let services_now: Vec<ServiceInfo> = services_snapshot
        .clone()
        .and_then(|r| r.ok())
        .unwrap_or_default();
    let services_err: Option<String> = services_snapshot.and_then(|r| r.err());

    let do_start_all = move |_| {
        error.set(None);
        let mut res = services;
        spawn(async move {
            if let Err(e) = api::start_all_services().await {
                error.set(Some(e));
            }
            res.restart();
        });
    };
    let do_stop_all = move |_| {
        error.set(None);
        let mut res = services;
        spawn(async move {
            if let Err(e) = api::stop_all_services().await {
                error.set(Some(e));
            }
            res.restart();
        });
    };

    let sse_dot = match sse_state().as_str() {
        "connected" => "ok",
        "connecting" => "busy",
        _ => "err",
    }
    .to_string();

    rsx! {
        PageHeader {
            title: "Services".to_string(),
            subtitle: "Launch and monitor lunar-start / lunar-start-backend managed services (testbench, testbench-backend, and more), with live streamed logs.".to_string(),
        }
        div { class: "page",
            div { class: "toolbar",
                button { class: "btn btn-primary", onclick: do_start_all, "Start all" }
                button { class: "btn btn-danger", onclick: do_stop_all, "Stop all" }
                span { class: "spacer" }
                StatusDot { status: sse_dot }
                span { class: "mono", style: "color: var(--text-3);", "logs: {sse_state()}" }
            }
            if let Some(e) = error() {
                div { class: "status-banner status-err", "{e}" }
            }
            if let Some(e) = services_err {
                div { class: "status-banner status-err", "Failed to reach lunar-start-backend: {e}" }
            }
            div { class: "card",
                div { class: "card-title", "Managed services" }
                if services_now.is_empty() {
                    div { class: "empty", "No services reported yet" }
                } else {
                    table { class: "tbl",
                        thead {
                            tr {
                                th { "" }
                                th { "Name" }
                                th { "Status" }
                                th { "PID" }
                                th { "" }
                            }
                        }
                        tbody {
                            for s in services_now.iter() {
                                ServiceRow {
                                    service: s.clone(),
                                    services,
                                    log_filter,
                                }
                            }
                        }
                    }
                }
            }
            div { class: "section-title", "Logs" }
            div { class: "toolbar",
                if let Some(f) = log_filter() {
                    Tag { text: format!("service: {f}"), kind: "mute".to_string() }
                    button { class: "btn btn-sm btn-ghost", onclick: move |_| log_filter.set(None), "Clear filter" }
                }
                span { class: "spacer" }
                button { class: "btn btn-sm", onclick: move |_| logs.set(Vec::new()), "Clear logs" }
            }
            LogConsole { logs, filter: log_filter }
        }
    }
}

#[component]
fn ServiceRow(
    service: ServiceInfo,
    services: Resource<Result<Vec<ServiceInfo>, String>>,
    mut log_filter: Signal<Option<String>>,
) -> Element {
    let kind = status_kind(&service.status);
    let pid_str = service
        .pid
        .map(|p| p.to_string())
        .unwrap_or_else(|| "-".to_string());

    let name_start = service.name.clone();
    let name_stop = service.name.clone();
    let name_restart = service.name.clone();
    let name_filter = service.name.clone();

    rsx! {
        tr {
            td { StatusDot { status: kind } }
            td { "{service.name}" }
            td { "{service.status}" }
            td { class: "mono", "{pid_str}" }
            td {
                div { class: "row",
                    button {
                        class: "btn btn-sm",
                        onclick: move |_| {
                            let n = name_start.clone();
                            let mut res = services;
                            spawn(async move {
                                let _ = api::start_service(&n).await;
                                res.restart();
                            });
                        },
                        "Start"
                    }
                    button {
                        class: "btn btn-sm",
                        onclick: move |_| {
                            let n = name_stop.clone();
                            let mut res = services;
                            spawn(async move {
                                let _ = api::stop_service(&n).await;
                                res.restart();
                            });
                        },
                        "Stop"
                    }
                    button {
                        class: "btn btn-sm",
                        onclick: move |_| {
                            let n = name_restart.clone();
                            let mut res = services;
                            spawn(async move {
                                let _ = api::restart_service(&n).await;
                                res.restart();
                            });
                        },
                        "Restart"
                    }
                    button {
                        class: "btn btn-sm btn-ghost",
                        onclick: move |_| log_filter.set(Some(name_filter.clone())),
                        "Logs"
                    }
                }
            }
        }
    }
}

#[component]
fn LogConsole(logs: Signal<Vec<ServiceLogEvent>>, filter: Signal<Option<String>>) -> Element {
    let filt = filter();
    let entries: Vec<ServiceLogEvent> = logs()
        .into_iter()
        .filter(|l| filt.as_ref().map(|f| f == &l.service).unwrap_or(true))
        .collect();

    rsx! {
        div { class: "log-view",
            if entries.is_empty() {
                div { class: "log-line dim", "Waiting for service logs..." }
            } else {
                for entry in entries.iter() {
                    div {
                        class: if entry.is_stderr { "log-line err" } else { "log-line" },
                        span { class: "mono", style: "color: var(--text-4);", "[{entry.timestamp}] " }
                        span { class: "mono", style: "color: var(--accent);", "[{entry.service}] " }
                        "{entry.text}"
                    }
                }
            }
        }
    }
}
