use crate::api::{system_snapshot, SystemSnapshot};
use crate::components::ui::{bytes_human, fmt_age, PageHeader, StatusDot, Tag};
use dioxus::prelude::*;

#[component]
pub fn Datasets() -> Element {
    let snapshot = use_resource(|| async { system_snapshot().await.ok() });
    rsx! {
        PageHeader {
            title: "Datasets".to_string(),
            subtitle: "Inspect the parquet/csv files that the trainers consume. Confirm paths, sizes, and modification times before launching long runs.".to_string(),
        }
        div { class: "page",
            match &*snapshot.read() {
                Some(Some(snap)) => rsx! { DatasetsBody { snap: snap.clone() } },
                _ => rsx! {
                    div { class: "status-banner status-info",
                        span { class: "spinner" }
                        span { "Loading…" }
                    }
                },
            }
        }
    }
}

#[component]
fn DatasetsBody(snap: SystemSnapshot) -> Element {
    let mut datasets = snap.datasets.clone();
    datasets.sort_by(|a, b| b.mtime_ms.cmp(&a.mtime_ms));

    rsx! {
        div { class: "card",
            table { class: "tbl",
                thead {
                    tr {
                        th { "File" }
                        th { "Kind" }
                        th { "Size" }
                        th { "Path" }
                        th { "Modified" }
                    }
                }
                tbody {
                    for d in datasets.iter() {
                        tr {
                            td {
                                span { class: "mono", "{d.name}" }
                            }
                            td { Tag { text: d.kind.clone(), kind: kind_to_tag(&d.kind) } }
                            td { span { class: "mono", "{bytes_human(d.size_bytes)}" } }
                            td { span { class: "mono", style: "color: var(--text-3);", "{d.path}" } }
                            td {
                                div { class: "row",
                                    StatusDot { status: "ok".to_string() }
                                    span { class: "mono", "{fmt_age(d.mtime_ms)}" }
                                }
                            }
                        }
                    }
                }
            }
            if datasets.is_empty() {
                div { class: "empty", "No datasets found in ai_data/, data/chunks/, or data/" }
            }
        }
        div { class: "section-title", "Model artifacts" }
        div { class: "card",
            table { class: "tbl",
                thead {
                    tr {
                        th { "File" }
                        th { "Kind" }
                        th { "Size" }
                        th { "Path" }
                        th { "Modified" }
                    }
                }
                tbody {
                    for m in snap.models.iter() {
                        td { span { class: "mono", "{m.name}" } }
                        td { Tag { text: m.kind.clone(), kind: kind_to_tag(&m.kind) } }
                        td { span { class: "mono", "{bytes_human(m.size_bytes)}" } }
                        td { span { class: "mono", style: "color: var(--text-3);", "{m.path}" } }
                        td { span { class: "mono", "{fmt_age(m.mtime_ms)}" } }
                    }
                }
            }
        }
    }
}

fn kind_to_tag(kind: &str) -> String {
    match kind {
        "pinn" => "pinn".to_string(),
        "gnn" => "gnn".to_string(),
        "siren" => "siren".to_string(),
        "raw" | "clean" | "chunks" | "holdout" | "lore" => "siren".to_string(),
        _ => "mute".to_string(),
    }
}
