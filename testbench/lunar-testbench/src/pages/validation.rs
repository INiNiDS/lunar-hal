use crate::api;
use crate::components::ui::{
    LossChart, NumberFieldF64, NumberFieldU32, PageHeader, ProgressBar, StatusDot, Tag, TextField,
    fmt_age, fmt_ms, tokio_time_sleep,
};
use crate::os::state::{is_window_lifecycle_visible, use_window_lifecycle};
use crate::os::{AppSnapshot, WindowLifecycle};
use dioxus::prelude::*;
use lunar_structures_testbench::{Job, JobStatus, ModelKind, ValidateSpec};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ValidationSnapshot {
    pub selected: Option<String>,
    pub model_kind: ModelKind,
    pub data_path: String,
    pub output_dir: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub val_frac: f64,
    pub knn_k: u32,
    pub hidden_dim: u32,
    pub texture_size: u32,
    pub max_stars: u32,
    pub error: Option<String>,
}

impl Default for ValidationSnapshot {
    fn default() -> Self {
        Self {
            selected: None,
            model_kind: ModelKind::Pinn,
            data_path: "ai_data/clean_stars2.parquet".to_string(),
            output_dir: "models/validation".to_string(),
            epochs: 50,
            batch_size: 2048,
            val_frac: 0.1,
            knn_k: 8,
            hidden_dim: 256,
            texture_size: 64,
            max_stars: 5000,
            error: None,
        }
    }
}

impl AppSnapshot for ValidationSnapshot {
    fn capture_snapshot(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn hydrate_snapshot(&mut self, payload: &serde_json::Value) -> Result<(), String> {
        let snap: ValidationSnapshot = serde_json::from_value(payload.clone())
            .map_err(|e| format!("Validation hydration failed: {e}"))?;
        *self = snap;
        Ok(())
    }
}

#[component]
pub fn Validation() -> Element {
    let initial =
        crate::os::use_window_instance_snapshot::<ValidationSnapshot>().unwrap_or_default();
    let jobs = use_resource(|| async { api::list_jobs().await.ok() });
    let selected = use_signal(|| initial.selected.clone());

    rsx! {
        PageHeader {
            title: "Validation Checkpoints".to_string(),
            subtitle: "Run read-only evaluation checkpoints without training epochs: compare checkpoint predictions against approved baselines and ground-truth oracle metrics.".to_string(),
        }
        div { class: "page",
            ValidationBody {
                jobs,
                selected,
                initial,
            }
        }
    }
}

#[component]
fn ValidationBody(
    jobs: Resource<Option<Vec<Job>>>,
    mut selected: Signal<Option<String>>,
    initial: ValidationSnapshot,
) -> Element {
    let mut model_kind = use_signal(|| initial.model_kind.clone());
    let mut data_path = use_signal(|| initial.data_path.clone());
    let output_dir = use_signal(|| initial.output_dir.clone());
    let epochs = use_signal(|| initial.epochs);
    let mut batch_size = use_signal(|| initial.batch_size);
    let val_frac = use_signal(|| initial.val_frac);
    let knn_k = use_signal(|| initial.knn_k);
    let hidden_dim = use_signal(|| initial.hidden_dim);
    let texture_size = use_signal(|| initial.texture_size);
    let max_stars = use_signal(|| initial.max_stars);
    let holdout = use_signal(|| "ai_data/holdout.parquet".to_string());
    let mut error = use_signal(|| initial.error.clone());
    let mut starting = use_signal(|| false);

    let mut os = crate::os::use_os_state();
    use_effect(move || {
        let snap = ValidationSnapshot {
            selected: selected(),
            model_kind: model_kind(),
            data_path: data_path(),
            output_dir: output_dir(),
            epochs: epochs(),
            batch_size: batch_size(),
            val_frac: val_frac(),
            knn_k: knn_k(),
            hidden_dim: hidden_dim(),
            texture_size: texture_size(),
            max_stars: max_stars(),
            error: error(),
        };
        if let Some(inst_id) = crate::os::use_window_instance_id() {
            os.register_instance_snapshot(&inst_id, snap.capture_snapshot());
        }
        os.register_app_snapshot("validation", snap.capture_snapshot());
    });

    let on_kind_change = move |k: ModelKind| {
        model_kind.set(k.clone());
        match k {
            ModelKind::Pinn => {
                data_path.set("ai_data/clean_stars2.parquet".to_string());
                batch_size.set(2048);
            }
            ModelKind::Gnn => {
                data_path.set("ai_data/clean_gnn_stars.parquet".to_string());
                batch_size.set(4096);
            }
            ModelKind::Siren => {
                data_path.set("ai_data/clean_stars2.parquet".to_string());
                batch_size.set(512);
            }
            ModelKind::GnnLocalization => {
                data_path.set("ai_data/clean_gnn_stars.parquet".to_string());
                batch_size.set(2048);
            }
        }
    };

    let submit = move |_| {
        error.set(None);
        starting.set(true);
        let model_dto = match model_kind() {
            ModelKind::Pinn => lunar_structures_testbench::typed::ModelKindDto::Pinn,
            ModelKind::Gnn => lunar_structures_testbench::typed::ModelKindDto::Gnn,
            ModelKind::Siren => lunar_structures_testbench::typed::ModelKindDto::Siren,
            ModelKind::GnnLocalization => {
                lunar_structures_testbench::typed::ModelKindDto::GnnLocalization
            }
        };
        let req = lunar_structures_testbench::typed::EvaluationRequest {
            model: model_dto,
            data_path: data_path(),
            output_dir: output_dir(),
            holdout: if holdout().is_empty() {
                None
            } else {
                Some(holdout())
            },
            batch_size: batch_size(),
            seed: None,
            artifact_hash: None,
            dataset_manifest_hash: None,
        };
        let mut res = jobs;
        spawn(async move {
            match api::start_evaluation(&req).await {
                Ok(job) => selected.set(Some(job.id)),
                Err(e) => error.set(Some(e)),
            }
            starting.set(false);
            res.restart();
        });
    };

    let lifecycle = use_window_lifecycle();
    let polling_lifecycle = lifecycle;

    use_future(move || async move {
        loop {
            tokio_time_sleep(2000).await;
            if is_window_lifecycle_visible(polling_lifecycle) {
                jobs.restart();
            }
        }
    });

    use_effect(move || {
        if let Some(lifecycle) = lifecycle {
            if *lifecycle.read() == WindowLifecycle::Visible {
                jobs.restart();
            }
        }
    });

    let jobs_now: Vec<Job> = jobs.cloned().flatten().unwrap_or_default();
    let current = selected
        .cloned()
        .and_then(|id| jobs_now.iter().find(|j| j.id == id).cloned());

    rsx! {
        div { class: "split",
            div { class: "card",
                div { class: "card-title", "Validation config" }
                div { class: "grid",
                    KindSelector { kind: model_kind, on_change: on_kind_change }
                    TextField { label: "Data path".to_string(), value: data_path }
                    TextField { label: "Output dir".to_string(), value: output_dir }
                    TextField { label: "Holdout path".to_string(), value: holdout }
                    NumberFieldU32 { label: "Batch size".to_string(), value: batch_size }
                }
                if let Some(e) = error() {
                    div { class: "status-banner status-err", "{e}" }
                }
                div { class: "toolbar", style: "margin-top: 14px;",
                    button {
                        class: "btn btn-primary",
                        disabled: starting(),
                        onclick: submit,
                        if starting() { span { class: "spinner" } }
                        span { "Run validation checkpoint" }
                    }
                }
                div { class: "field-hint", style: "margin-top: 8px;",
                    "Spawns read-only evaluation checkpoint runner with zero optimizer epochs."
                }
            }
            div { class: "card",
                JobsList {
                    jobs: jobs_now.clone(),
                    selected,
                }
                if let Some(job) = current {
                    JobDetail { job: job.clone(), on_cancel: move |id: String| {
                        let mut res = jobs;
                        spawn(async move {
                            let _ = api::cancel_job(&id).await;
                            res.restart();
                        });
                    } }
                } else {
                    div { class: "empty", "Select a job to see validation metrics" }
                }
            }
        }
    }
}

#[component]
fn JobsList(jobs: Vec<Job>, mut selected: Signal<Option<String>>) -> Element {
    rsx! {
        div { class: "card-title", "Validation runs" }
        if jobs.is_empty() {
            div { class: "empty", "No validation runs yet" }
        } else {
            table { class: "tbl",
                thead {
                    tr {
                        th { "" }
                        th { "Title" }
                        th { "Status" }
                        th { "Best val" }
                        th { "Age" }
                    }
                }
                tbody {
                    for j in jobs.iter() {
                        {
                            let j_id = j.id.clone();
                            let is_active = selected().map(|s| s == j.id).unwrap_or(false);
                            rsx! {
                                tr {
                                    class: if is_active { "is-active" } else { "" },
                                    onclick: move |_| selected.set(Some(j_id.clone())),
                                    td { StatusDot { status: j.status.tag().to_string() } }
                                    td { "{j.title}" }
                                    td { "{j.status.tag()}" }
                                    td {
                                        {
                                            if let Some(v) = j.best_val_loss {
                                                format!("{:.5}", v)
                                            } else {
                                                "—".to_string()
                                            }
                                        }
                                    }
                                    td { "{fmt_age(j.created_ms)}" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn JobDetail(job: Job, on_cancel: EventHandler<String>) -> Element {
    let progress = job.progress();
    let status = job.status.tag().to_string();
    let is_running = job.status == JobStatus::Running;
    let is_queued = job.status == JobStatus::Queued;
    let progress_class = match &job.spec {
        lunar_structures_testbench::JobKind::Validate(_) => "siren",
        _ => "pinn",
    };

    let elapsed = job
        .started_ms
        .or(Some(job.created_ms))
        .map(|s| fmt_ms((crate::components::ui::now_ms()) - (s as i64)))
        .unwrap_or_else(|| "—".to_string());

    rsx! {
        div { class: "section-title", "Validation run" }
        div { class: "card",
            div { class: "row", style: "justify-content: space-between;",
                div { class: "row",
                    Tag { text: status.to_uppercase(), kind: status.clone() }
                    span { class: "mono", style: "color: var(--text-3);", "{job.title}" }
                    span { class: "mono", style: "color: var(--text-3); margin-left: 8px;", "elapsed {elapsed}" }
                }
                if is_running || is_queued {
                    button { class: "btn btn-danger btn-sm",
                        onclick: move |_| on_cancel.call(job.id.clone()),
                        "Cancel" }
                }
            }
            ProgressBar { value: progress, kind: progress_class.to_string() }
        }
        div { class: "section-title", "Evaluation checkpoint (oracle vs. baseline vs. checkpoint)" }
        div { class: "card",
            div { class: "grid", style: "grid-template-columns: repeat(3, 1fr); gap: 12px;",
                div { class: "card", style: "background: var(--surface-2);",
                    div { class: "field-label", "Oracle (Ground Truth)" }
                    div { class: "mono", style: "font-size: 1.25rem; font-weight: bold; color: var(--accent);", "0.00000" }
                    div { class: "field-hint", "Target simulation manifold" }
                }
                div { class: "card", style: "background: var(--surface-2);",
                    div { class: "field-label", "Approved Baseline" }
                    div { class: "mono", style: "font-size: 1.25rem; font-weight: bold; color: var(--text-2);", "0.00500" }
                    div { class: "field-hint", "Stage 7 regression target" }
                }
                div { class: "card", style: "background: var(--surface-2);",
                    div { class: "field-label", "Current Checkpoint" }
                    div { class: "mono", style: "font-size: 1.25rem; font-weight: bold; color: var(--pinn);",
                        {
                            if let Some(v) = job.best_val_loss {
                                format!("{:.5}", v)
                            } else {
                                "evaluating…".to_string()
                            }
                        }
                    }
                    div { class: "field-hint", "Evaluated loss delta" }
                }
            }
        }
        div { class: "section-title", "Loss curve" }
        LossChart { metrics: job.last_metrics.clone(), width: 720.0, height: 280.0 }
        div { class: "section-title", "Per-epoch table" }
        div { class: "card",
            if job.last_metrics.is_empty() {
                div { class: "empty", "Waiting for the first epoch…" }
            } else {
                table { class: "tbl",
                    thead {
                        tr {
                            th { "Epoch" }
                            th { "Train" }
                            th { "Val" }
                            th { "Phys" }
                            th { "LR" }
                        }
                    }
                    tbody {
                        for m in job.last_metrics.iter() {
                            {
                                let train_s = format!("{:.6}", m.train_loss);
                                let val_s = format!("{:.6}", m.val_loss);
                                let phys_s = m.phys_loss.map(|p| format!("{:.6}", p));
                                let lr_s = format!("{:.2e}", m.lr);
                                rsx! {
                                    tr {
                                        td { "{m.epoch}" }
                                        td { "{train_s}" }
                                        td { "{val_s}" }
                                        td {
                                            {
                                                if let Some(p) = phys_s {
                                                    p
                                                } else {
                                                    "—".to_string()
                                                }
                                            }
                                        }
                                        td { "{lr_s}" }
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

#[component]
fn KindSelector(kind: Signal<ModelKind>, on_change: EventHandler<ModelKind>) -> Element {
    let options = [
        (ModelKind::Pinn, "PINN", "pinn"),
        (ModelKind::Gnn, "GNN (Kinematics)", "gnn"),
        (ModelKind::Siren, "SIREN", "siren"),
        (ModelKind::GnnLocalization, "GNN (Localization)", "gnn"),
    ];
    rsx! {
        div { class: "field",
            span { class: "field-label", "Model" }
            div { class: "row",
                for (m, label, tag) in options {
                    button {
                        class: if kind() == m { "btn btn-primary" } else { "btn" },
                        onclick: move |_| on_change.call(m.clone()),
                        span { class: "tag tag-{tag}", style: "margin-right: 6px;", "{label}" }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_snapshot_capture_and_hydration() {
        let original = ValidationSnapshot {
            selected: Some("val-789".into()),
            model_kind: ModelKind::Siren,
            data_path: "ai_data/siren.parquet".into(),
            output_dir: "models/val_siren".into(),
            epochs: 25,
            batch_size: 1024,
            val_frac: 0.2,
            knn_k: 12,
            hidden_dim: 128,
            texture_size: 32,
            max_stars: 2500,
            error: Some("test error".into()),
        };
        let payload = original.capture_snapshot();
        let mut restored = ValidationSnapshot::default();
        restored.hydrate_snapshot(&payload).unwrap();
        assert_eq!(restored, original);
    }
}
