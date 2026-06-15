use crate::api::{system_snapshot, SystemSnapshot};
use crate::components::ui::{bytes_human, fmt_age, PageHeader, StatusDot, Tag};
use dioxus::prelude::*;

#[component]
pub fn Dashboard() -> Element {
    let snapshot = use_resource(|| async { system_snapshot().await.ok() });
    rsx! {
        PageHeader {
            title: "Dashboard".to_string(),
            subtitle: "Live view of the Lunar-HAL testbench: backend reachability, model artifacts, training jobs, host resources.".to_string(),
        }
        div { class: "page",
            match &*snapshot.read() {
                Some(Some(s)) => rsx! { DashboardBody { snap: s.clone() } },
                Some(None) => rsx! { div { class: "status-banner status-err", "Failed to load snapshot" } },
                None => rsx! {
                    div { class: "status-banner status-info",
                        span { class: "spinner" }
                        span { "Loading system snapshot…" }
                    }
                },
            }
        }
    }
}

#[component]
fn DashboardBody(snap: SystemSnapshot) -> Element {
    let pinn = snap.models.iter().find(|m| m.kind == "pinn" && m.name.ends_with(".bpk"));
    let gnn = snap.models.iter().find(|m| m.kind == "gnn" && m.name.ends_with(".bpk"));
    let siren = snap.models.iter().find(|m| m.kind == "siren" && m.name.ends_with(".bpk"));

    let pinn_norm_exists = snap.norms.iter().find(|n| n.kind == "stellar_norm.json").map(|n| n.exists).unwrap_or(false);
    let gnn_norm_exists = snap.norms.iter().find(|n| n.kind == "stellar_gnn_norm.json").map(|n| n.exists).unwrap_or(false);
    let siren_norm_exists = snap.norms.iter().find(|n| n.kind == "stellar_siren_norm.json").map(|n| n.exists).unwrap_or(false);

    let running_jobs = snap.jobs.iter().filter(|j| j.status == lunar_structures_testbench::JobStatus::Running).count();
    let completed_jobs = snap.jobs.iter().filter(|j| j.status == lunar_structures_testbench::JobStatus::Completed).count();
    let failed_jobs = snap.jobs.iter().filter(|j| j.status == lunar_structures_testbench::JobStatus::Failed).count();

    rsx! {
        div { class: "grid grid-3",
            BackendCard { snap: snap.clone() }
            TestbenchBackendCard { snap: snap.clone() }
            ModelArtifactCard {
                kind: "pinn", label: "PINN", artifact: pinn.cloned(), norm_exists: pinn_norm_exists,
            }
            ModelArtifactCard {
                kind: "gnn", label: "GNN", artifact: gnn.cloned(), norm_exists: gnn_norm_exists,
            }
            ModelArtifactCard {
                kind: "siren", label: "SIREN", artifact: siren.cloned(), norm_exists: siren_norm_exists,
            }
        }
        div { class: "section-title", "Jobs" }
        div { class: "grid grid-3",
            JobCounter { kind: "running".to_string(), value: running_jobs, label: "Running".to_string() }
            JobCounter { kind: "ok".to_string(), value: completed_jobs, label: "Completed".to_string() }
            JobCounter { kind: "err".to_string(), value: failed_jobs, label: "Failed".to_string() }
        }
        if !snap.jobs.is_empty() {
            div { class: "section-title", "Recent Jobs" }
            div { class: "card",
                table { class: "tbl",
                    thead {
                        tr {
                            th { "Title" }
                            th { "Kind" }
                            th { "Status" }
                            th { "Best Val" }
                            th { "Epoch" }
                            th { "Created" }
                        }
                    }
                    tbody {
                        for j in snap.jobs.iter().rev().take(8) {
                            tr {
                                td { "{j.title}" }
                                td { match &j.spec {
                                    lunar_structures_testbench::JobKind::Train(_) => rsx! { Tag { text: "train".to_string(), kind: "pinn".to_string() } },
                                    lunar_structures_testbench::JobKind::Validate(_) => rsx! { Tag { text: "validate".to_string(), kind: "siren".to_string() } },
                                    lunar_structures_testbench::JobKind::Custom(_) => rsx! { Tag { text: "custom".to_string(), kind: "mute".to_string() } },
                                } }
                                td {
                                    StatusDot { status: j.status.tag().to_string() }
                                    span { class: "mono", " {j.status.tag()}" }
                                }
                                td {
                                    {
                                        if let Some(v) = j.best_val_loss {
                                            format!("{:.5}", v)
                                        } else {
                                            "—".to_string()
                                        }
                                    }
                                }
                                td {
                                    "{j.last_metrics.len()} / {j.total_epochs_planned}"
                                }
                                td { "{fmt_age(j.created_ms)}" }
                            }
                        }
                    }
                }
            }
        }
        div { class: "section-title", "Host" }
        div { class: "card",
            div { class: "kv-list",
                div { class: "kv-row",
                    span { class: "kv-key", "Workspace" }
                    span { class: "kv-val", "{snap.host.workspace_root}" }
                }
                div { class: "kv-row",
                    span { class: "kv-key", "CPU cores" }
                    span { class: "kv-val", "{snap.host.cpu_count}" }
                }
                div { class: "kv-row",
                    span { class: "kv-key", "Memory" }
                    span { class: "kv-val", "{bytes_human(snap.host.total_memory_bytes)}" }
                }
                div { class: "kv-row",
                    span { class: "kv-key", "PID" }
                    span { class: "kv-val", "{snap.host.pid}" }
                }
            }
        }
    }
}

#[component]
fn BackendCard(snap: SystemSnapshot) -> Element {
    let latency = snap
        .backend
        .latency_ms
        .map(|l| format!("{l} ms"))
        .unwrap_or_else(|| "—".to_string());
    rsx! {
        div { class: "card",
            div { class: "card-title",
                StatusDot { status: snap.backend.reachable.then(|| "ok".to_string()).unwrap_or_else(|| "err".to_string()) }
                span { "Backend HTTP" }
            }
            div { class: "metric",
                div { class: "metric-value sm",
                    span { class: "mono", "{latency}" }
                }
                div { class: "metric-label", "RTT" }
            }
            div { class: "divider-h" }
            div { class: "kv-list",
                div { class: "kv-row",
                    span { class: "kv-key", "URL" }
                    span { class: "kv-val", "{snap.backend.url}" }
                }
                div { class: "kv-row",
                    span { class: "kv-key", "Status" }
                    span { class: "kv-val",
                        StatusDot { status: snap.backend.reachable.then(|| "ok".to_string()).unwrap_or_else(|| "err".to_string()) }
                        span { class: "mono", " " }
                        span { class: "mono", if snap.backend.reachable { "ONLINE" } else { "OFFLINE" } }
                    }
                }
            }
        }
    }
}

#[component]
fn TestbenchBackendCard(snap: SystemSnapshot) -> Element {
    let tb = snap.testbench_backend.clone();
    let status = tb.as_ref().map(|b| b.reachable).unwrap_or(false);
    let url = tb.as_ref().map(|b| b.url.clone()).unwrap_or_else(|| "—".into());
    let latency = tb
        .as_ref()
        .and_then(|b| b.latency_ms)
        .map(|l| format!("{l} ms"))
        .unwrap_or_else(|| "—".to_string());
    rsx! {
        div { class: "card",
            div { class: "card-title",
                StatusDot { status: if status { "ok".to_string() } else { "err".to_string() } }
                span { "Testbench Backend" }
            }
            div { class: "metric",
                div { class: "metric-value sm",
                    span { class: "mono", "{latency}" }
                }
                div { class: "metric-label", "RTT" }
            }
            div { class: "divider-h" }
            div { class: "kv-list",
                div { class: "kv-row",
                    span { class: "kv-key", "URL" }
                    span { class: "kv-val", "{url}" }
                }
                div { class: "kv-row",
                    span { class: "kv-key", "Status" }
                    span { class: "kv-val",
                        StatusDot { status: if status { "ok".to_string() } else { "err".to_string() } }
                        span { class: "mono", " " }
                        span { class: "mono", if status { "ONLINE" } else { "OFFLINE" } }
                    }
                }
            }
        }
    }
}

#[component]
fn ModelArtifactCard(
    kind: String,
    label: String,
    artifact: Option<crate::api::ModelArtifact>,
    norm_exists: bool,
) -> Element {
    let present = artifact.is_some();
    let status = if present { "ok" } else { "err" };
    let tag_kind = kind.clone();
    rsx! {
        div { class: "card",
            div { class: "card-title",
                StatusDot { status: status.to_string() }
                span { "{label} model" }
                span { class: "tag tag-{tag_kind}", style: "margin-left: auto;", "artifact" }
            }
            if let Some(a) = artifact {
                div { class: "metric",
                    div { class: "metric-value sm mono", "{a.name}" }
                    div { class: "metric-label", "file" }
                }
                div { class: "divider-h" }
                div { class: "kv-list",
                    div { class: "kv-row",
                        span { class: "kv-key", "Size" }
                        span { class: "kv-val", "{bytes_human(a.size_bytes)}" }
                    }
                    div { class: "kv-row",
                        span { class: "kv-key", "Modified" }
                        span { class: "kv-val", "{fmt_age(a.mtime_ms)}" }
                    }
                    div { class: "kv-row",
                        span { class: "kv-key", "Norm file" }
                        span { class: "kv-val",
                            StatusDot { status: if norm_exists { "ok".to_string() } else { "err".to_string() } }
                            span { class: "mono", " " }
                            span { class: "mono", if norm_exists { "present" } else { "missing" } }
                        }
                    }
                }
            } else {
                div { class: "empty", "Model artifact not found" }
            }
        }
    }
}

#[component]
fn JobCounter(kind: String, value: usize, label: String) -> Element {
    rsx! {
        div { class: "card",
            div { class: "card-title",
                StatusDot { status: kind.clone() }
                span { "{label}" }
            }
            div { class: "metric",
                div { class: "metric-value lg mono", "{value}" }
                div { class: "metric-sub", "session total" }
            }
        }
    }
}
