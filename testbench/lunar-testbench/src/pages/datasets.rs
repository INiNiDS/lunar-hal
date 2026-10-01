use crate::api::{
    DataCoverage, SystemSnapshot, data_status, start_data_build, start_data_collect,
    start_data_verify, system_snapshot,
};
use crate::components::ui::{PageHeader, StatusDot, Tag, bytes_human, fmt_age};
use crate::os::snapshot::AppSnapshot;
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

const DEFAULT_CANONICAL_DIR: &str = "data/canonical-v1";

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DatasetsSnapshot {
    pub coverage_dir: String,
    pub active_tab: String,
    pub filter: String,
}

impl Default for DatasetsSnapshot {
    fn default() -> Self {
        Self {
            coverage_dir: DEFAULT_CANONICAL_DIR.to_string(),
            active_tab: "overview".to_string(),
            filter: "all".to_string(),
        }
    }
}

impl AppSnapshot for DatasetsSnapshot {
    fn capture_snapshot(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn hydrate_snapshot(&mut self, payload: &serde_json::Value) -> Result<(), String> {
        let restored: DatasetsSnapshot =
            serde_json::from_value(payload.clone()).map_err(|e| e.to_string())?;
        *self = restored;
        Ok(())
    }
}

#[component]
pub fn Datasets() -> Element {
    let mut initial = DatasetsSnapshot::default();
    if let Some(payload) = crate::os::state::use_window_snapshot_payload() {
        let _ = initial.hydrate_snapshot(&payload);
    }

    let snapshot = use_resource(|| async { system_snapshot().await.ok() });
    let coverage_dir = use_signal(|| initial.coverage_dir.clone());
    let mut active_tab = use_signal(|| initial.active_tab.clone());
    let mut filter = use_signal(|| initial.filter.clone());

    let mut os = crate::os::use_os_state();
    use_effect(move || {
        let snap = DatasetsSnapshot {
            coverage_dir: coverage_dir(),
            active_tab: active_tab(),
            filter: filter(),
        };
        if let Some(inst_id) = crate::os::use_window_instance_id() {
            os.register_instance_snapshot(&inst_id, snap.capture_snapshot());
        }
        os.register_app_snapshot("datasets", snap.capture_snapshot());
    });

    rsx! {
        PageHeader {
            title: "Datasets".to_string(),
            subtitle: "Inspect the parquet/csv files that the trainers consume. Confirm paths, sizes, and modification times before launching long runs.".to_string(),
        }
        div { class: "page",
            div { class: "row", style: "gap: 8px; margin-bottom: 12px;",
                button {
                    class: if active_tab() == "overview" { "btn btn-primary btn-sm" } else { "btn btn-sm" },
                    onclick: move |_| active_tab.set("overview".to_string()),
                    "Overview"
                }
                button {
                    class: if active_tab() == "coverage" { "btn btn-primary btn-sm" } else { "btn btn-sm" },
                    onclick: move |_| active_tab.set("coverage".to_string()),
                    "Canonical Coverage"
                }
                button {
                    class: if active_tab() == "artifacts" { "btn btn-primary btn-sm" } else { "btn btn-sm" },
                    onclick: move |_| active_tab.set("artifacts".to_string()),
                    "Model Artifacts"
                }
                div { style: "margin-left: auto; display: flex; align-items: center; gap: 6px;",
                    span { class: "field-label", "Filter:" }
                    select {
                        class: "mono",
                        style: "background: var(--surface-2); color: var(--text-1); border: 1px solid var(--border); padding: 4px 8px; border-radius: 6px;",
                        value: "{filter()}",
                        onchange: move |e: Event<FormData>| filter.set(e.value()),
                        option { value: "all", "All datasets" }
                        option { value: "clean", "Clean only" }
                        option { value: "gnn", "GNN only" }
                        option { value: "chunks", "Chunks only" }
                    }
                }
            }

            if active_tab() == "overview" || active_tab() == "coverage" {
                CanonicalCollection { dir: coverage_dir }
            }

            if active_tab() == "overview" || active_tab() == "artifacts" {
                match &*snapshot.read() {
                    Some(Some(snap)) => rsx! { DatasetsBody { snap: snap.clone(), filter_kind: filter() } },
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
}

#[component]
fn DatasetsBody(snap: SystemSnapshot, filter_kind: String) -> Element {
    let mut datasets = snap.datasets.clone();
    if filter_kind != "all" {
        datasets.retain(|d| d.kind == filter_kind || (filter_kind == "clean" && d.kind == "raw"));
    }
    datasets.sort_by_key(|d| std::cmp::Reverse(d.mtime_ms));

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


#[component]
fn CanonicalCollection(mut dir: Signal<String>) -> Element {
    let coverage = use_resource(move || {
        let d = dir.read().clone();
        async move { data_status(&d).await.ok() }
    });
    let busy = use_signal(|| false);
    let message = use_signal(|| None::<String>);

    let refresh_signal = dir;
    rsx! {
        div { class: "section-title", "Canonical collection (lnai-data)" }
        div { class: "card",
            match &*coverage.read() {
                Some(Some(cov)) => rsx! { CoverageCard {
                    cov: cov.clone(),
                    busy,
                    message,
                    on_retarget: move |d: String| dir.set(d),
                } },
                Some(None) => rsx! {
                    div { class: "status-banner status-err",
                        span { "Backend unreachable or /data/status failed." }
                    }
                },
                None => rsx! {
                    div { class: "status-banner status-info",
                        span { class: "spinner" }
                        span { "Reading manifest…" }
                    }
                },
            }
            div { class: "field-hint", style: "margin-top: 6px;",
                "Collection directory: "
                input {
                    class: "mono",
                    style: "margin-left: 6px; max-width: 320px;",
                    value: "{refresh_signal()}",
                    onchange: move |e: Event<FormData>| dir.set(e.value().trim().to_string()),
                }
            }
        }
    }
}

#[component]
fn CoverageCard(
    cov: DataCoverage,
    mut busy: Signal<bool>,
    mut message: Signal<Option<String>>,
    on_retarget: EventHandler<String>,
) -> Element {
    let cov_dir_for_actions = cov.dir.clone();
    let cov_dir_verify = cov.dir.clone();
    let cov_dir_build = cov.dir.clone();
    let dir_now = cov_dir_for_actions.clone();
    let dot_status = if !cov.manifest_found {
        "warn"
    } else if cov.shards_failed > 0 {
        "err"
    } else if cov.shards_pending + cov.shards_active > 0 {
        "busy"
    } else if cov.shards_total > 0 {
        "ok"
    } else {
        "warn"
    };
    rsx! {
        div { class: "row", style: "gap: 18px;",
            StatusDot { status: dot_status.to_string() }
            Tag {
                text: if cov.manifest_found {
                    format!("schema {}", cov.schema_version.clone().unwrap_or_default())
                } else {
                    "no manifest".to_string()
                },
                kind: if cov.manifest_found {
                    "clean".to_string()
                } else {
                    "mute".to_string()
                },
            }
            span { class: "mono", style: "color: var(--text-3);",
                "{cov.source_release.clone().unwrap_or_default()}"
            }
            span { class: "mono",
                "coverage: {cov.verified_percent}% ({cov.shards_verified}/{cov.shards_total} shards)"
            }
            span { class: "mono",
                "rows: {cov.total_rows}"
            }
        }
        div { class: "row", style: "gap: 10px; margin-top: 8px;",
            span { class: "mono", style: "color: var(--text-3);",
                "failed: {cov.shards_failed} · pending: {cov.shards_pending} · active: {cov.shards_active} · subdivided: {cov.subdivided_shards}"
            }
            if let Some(ms) = cov.last_updated_ms {
                span { class: "mono", style: "color: var(--text-3);", "updated {fmt_age(ms)}" }
            }
        }
        if !cov.gap_explanations.is_empty() {
            details { style: "margin-top: 8px;",
                summary { "{cov.gap_explanations.len()} gap/subdivision note(s)" }
                for note in cov.gap_explanations.iter().take(32) {
                    div { class: "mono", style: "color: var(--text-3);", "{note}" }
                }
            }
        }
        div { class: "toolbar", style: "margin-top: 12px; gap: 8px;",
            button {
                class: "btn btn-primary",
                disabled: busy(),
                onclick: move |_| {
                    let dir_now = dir_now.clone();
                    spawn(async move {
                        busy.set(true);
                        message.set(Some("Starting pilot collect…".into()));
                        match start_data_collect(&dir_now, 0.0, 3.0, 16.0, 4, false).await {
                            Ok(job) => message.set(Some(format!("Collect job started: {}", job.id))),
                            Err(e) => message.set(Some(format!("Failed to start: {e}"))),
                        }
                        busy.set(false);
                        on_retarget.call(dir_now);
                    });
                },
                if busy() { span { class: "spinner" } }
                span { "Pilot collect (RA 0–3°)" }
            }
            button {
                class: "btn",
                disabled: busy(),
                onclick: move |_| {
                    let dir_now = cov_dir_verify.clone();
                    spawn(async move {
                        busy.set(true);
                        match start_data_verify(&dir_now).await {
                            Ok(job) => message.set(Some(format!("Verify job started: {}", job.id))),
                            Err(e) => message.set(Some(format!("Failed to start: {e}"))),
                        }
                        busy.set(false);
                    });
                },
                span { "Verify checksums" }
            }
            button {
                class: "btn",
                disabled: busy(),
                onclick: move |_| {
                    let dir_now = cov_dir_build.clone();
                    spawn(async move {
                        busy.set(true);
                        match start_data_build(&dir_now).await {
                            Ok(job) => message.set(Some(format!("Build job started: {}", job.id))),
                            Err(e) => message.set(Some(format!("Failed to start: {e}"))),
                        }
                        busy.set(false);
                    });
                },
                span { "Assemble dataset" }
            }
            if let Some(m) = message() {
                span { class: "mono", style: "color: var(--text-3);", "{m}" }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datasets_snapshot_capture_and_hydration() {
        let original = DatasetsSnapshot {
            coverage_dir: "data/custom-canonical".to_string(),
            active_tab: "artifacts".to_string(),
            filter: "clean".to_string(),
        };
        let payload = original.capture_snapshot();
        let mut restored = DatasetsSnapshot::default();
        restored
            .hydrate_snapshot(&payload)
            .expect("hydration must succeed");
        assert_eq!(restored, original);
    }
}
