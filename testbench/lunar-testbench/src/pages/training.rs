use crate::api;
use crate::components::ui::{
    LossChart, NumberFieldF64, NumberFieldU32, PageHeader, ProgressBar, StatusDot, Tag, TextField,
    fmt_age, fmt_ms, tokio_time_sleep,
};
use crate::os::state::{is_window_lifecycle_visible, use_window_lifecycle};
use crate::os::{AppSnapshot, WindowLifecycle};
use dioxus::prelude::*;
use lunar_structures_testbench::{Job, JobStatus, ModelKind, TrainSpec};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TrainingSnapshot {
    pub selected: Option<String>,
    pub model_kind: ModelKind,
    pub data_path: String,
    pub output_dir: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub physics_weight: f64,
    pub val_frac: f64,
    pub gpu_index: u32,
    pub patience: u32,
    pub grad_accum: u32,
    pub clip_grad_norm: f64,
    pub knn_k: u32,
    pub hidden_dim: u32,
    pub texture_size: u32,
    pub max_stars: u32,
    pub resume: String,
    pub holdout: String,
    pub error: Option<String>,
    pub starting: bool,
}

impl Default for TrainingSnapshot {
    fn default() -> Self {
        Self {
            selected: None,
            model_kind: ModelKind::Pinn,
            data_path: "ai_data/clean_stars2.parquet".to_string(),
            output_dir: "models".to_string(),
            epochs: 50,
            batch_size: 2048,
            lr: 5e-4,
            physics_weight: 0.1,
            val_frac: 0.1,
            gpu_index: 0,
            patience: 20,
            grad_accum: 2,
            clip_grad_norm: 1.0,
            knn_k: 8,
            hidden_dim: 256,
            texture_size: 64,
            max_stars: 5000,
            resume: String::new(),
            holdout: String::new(),
            error: None,
            starting: false,
        }
    }
}

impl AppSnapshot for TrainingSnapshot {
    fn capture_snapshot(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn hydrate_snapshot(&mut self, payload: &serde_json::Value) -> Result<(), String> {
        let snap: TrainingSnapshot = serde_json::from_value(payload.clone())
            .map_err(|e| format!("Training hydration failed: {e}"))?;
        *self = snap;
        Ok(())
    }
}

#[component]
pub fn Training() -> Element {
    let initial = crate::os::use_window_instance_snapshot::<TrainingSnapshot>().unwrap_or_default();
    let jobs = use_resource(|| async { api::list_jobs().await.ok() });
    let selected = use_signal(|| initial.selected.clone());

    rsx! {
        PageHeader {
            title: "Training".to_string(),
            subtitle: "Launch training jobs against the existing trainers (lnai, lnai-gnn, lnai-siren). Watch live logs, follow epoch metrics, and inspect artifacts.".to_string(),
        }
        div { class: "page",
            TrainingBody {
                jobs_resource: jobs,
                selected,
                initial,
            }
        }
    }
}

#[component]
fn TrainingBody(
    jobs_resource: Resource<Option<Vec<Job>>>,
    mut selected: Signal<Option<String>>,
    initial: TrainingSnapshot,
) -> Element {
    let mut model_kind = use_signal(|| initial.model_kind.clone());
    let mut data_path = use_signal(|| initial.data_path.clone());
    let output_dir = use_signal(|| initial.output_dir.clone());
    let mut epochs = use_signal(|| initial.epochs);
    let mut batch_size = use_signal(|| initial.batch_size);
    let mut lr = use_signal(|| initial.lr);
    let physics_weight = use_signal(|| initial.physics_weight);
    let val_frac = use_signal(|| initial.val_frac);
    let gpu_index = use_signal(|| initial.gpu_index);
    let patience = use_signal(|| initial.patience);
    let grad_accum = use_signal(|| initial.grad_accum);
    let clip_grad_norm = use_signal(|| initial.clip_grad_norm);
    let knn_k = use_signal(|| initial.knn_k);
    let hidden_dim = use_signal(|| initial.hidden_dim);
    let texture_size = use_signal(|| initial.texture_size);
    let max_stars = use_signal(|| initial.max_stars);
    let radius = use_signal(|| 25.0_f64);
    let max_slots = use_signal(|| 16_u32);
    let mask_ratio = use_signal(|| 0.3_f64);
    let latent_dim = use_signal(|| 64_u32);
    let resume = use_signal(|| initial.resume.clone());
    let holdout = use_signal(|| initial.holdout.clone());
    let mut error = use_signal(|| initial.error.clone());
    let mut starting = use_signal(|| initial.starting);

    let mut os = crate::os::use_os_state();
    use_effect(move || {
        let snap = TrainingSnapshot {
            selected: selected(),
            model_kind: model_kind(),
            data_path: data_path(),
            output_dir: output_dir(),
            epochs: epochs(),
            batch_size: batch_size(),
            lr: lr(),
            physics_weight: physics_weight(),
            val_frac: val_frac(),
            gpu_index: gpu_index(),
            patience: patience(),
            grad_accum: grad_accum(),
            clip_grad_norm: clip_grad_norm(),
            knn_k: knn_k(),
            hidden_dim: hidden_dim(),
            texture_size: texture_size(),
            max_stars: max_stars(),
            resume: resume(),
            holdout: holdout(),
            error: error(),
            starting: starting(),
        };
        if let Some(inst_id) = crate::os::use_window_instance_id() {
            os.register_instance_snapshot(&inst_id, snap.capture_snapshot());
        }
        os.register_app_snapshot("training", snap.capture_snapshot());
    });

    let on_kind_change = move |k: ModelKind| {
        model_kind.set(k.clone());
        match k {
            ModelKind::Pinn => {
                data_path.set("ai_data/clean_stars2.parquet".to_string());
                epochs.set(50);
                batch_size.set(2048);
                lr.set(5e-4);
            }
            ModelKind::Gnn => {
                data_path.set("ai_data/clean_gnn_stars.parquet".to_string());
                epochs.set(40);
                batch_size.set(4096);
                lr.set(3e-4);
            }
            ModelKind::Siren => {
                data_path.set("ai_data/clean_stars2.parquet".to_string());
                epochs.set(30);
                batch_size.set(1024);
                lr.set(1e-3);
            }
            ModelKind::GnnLocalization => {
                data_path.set("ai_data/clean_gnn_stars.parquet".to_string());
                epochs.set(30);
                batch_size.set(2048);
                lr.set(4e-4);
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
        let req = lunar_structures_testbench::typed::TrainingRequest {
            model: model_dto,
            data_path: data_path(),
            output_dir: output_dir(),
            epochs: epochs(),
            batch_size: batch_size(),
            lr: lr(),
            val_frac: val_frac() as f32,
            physics_weight: physics_weight(),
            gpu_index: gpu_index(),
            patience: patience(),
            grad_accum: grad_accum(),
            clip_grad_norm: clip_grad_norm(),
            resume_from: if resume().is_empty() {
                None
            } else {
                Some(resume())
            },
            holdout: if holdout().is_empty() {
                None
            } else {
                Some(holdout())
            },
            seed: None,
            dataset_manifest_hash: None,
            hidden_dim: if model_kind() == ModelKind::Gnn {
                Some(hidden_dim())
            } else if model_kind() == ModelKind::GnnLocalization {
                Some(latent_dim())
            } else {
                None
            },
            knn_k: if model_kind() == ModelKind::Gnn {
                Some(knn_k())
            } else {
                None
            },
            max_group_size: None,
            radius_pc: if model_kind() == ModelKind::GnnLocalization {
                Some(radius() as f32)
            } else {
                None
            },
            texture_size: if model_kind() == ModelKind::Siren {
                Some(texture_size())
            } else {
                None
            },
            max_stars: if model_kind() == ModelKind::Siren {
                Some(max_stars())
            } else {
                None
            },
            max_slots: if model_kind() == ModelKind::GnnLocalization {
                Some(max_slots())
            } else {
                None
            },
            mask_ratio: if model_kind() == ModelKind::GnnLocalization {
                Some(mask_ratio() as f32)
            } else {
                None
            },
        };
        let mut res = jobs_resource;
        spawn(async move {
            match api::start_training(&req).await {
                Ok(job) => selected.set(Some(job.id)),
                Err(e) => error.set(Some(e)),
            }
            starting.set(false);
            res.restart();
        });
    };

    let mut tick = use_signal(|| 0);
    let lifecycle = use_window_lifecycle();
    let polling_lifecycle = lifecycle;

    use_future(move || async move {
        loop {
            tokio_time_sleep(2000).await;
            if is_window_lifecycle_visible(polling_lifecycle) {
                tick.set(tick() + 1);
                jobs_resource.restart();
            }
        }
    });

    use_effect(move || {
        if let Some(lifecycle) = lifecycle {
            if *lifecycle.read() == WindowLifecycle::Visible {
                jobs_resource.restart();
            }
        }
    });

    let jobs_now: Vec<Job> = jobs_resource.cloned().flatten().unwrap_or_default();
    let current = selected
        .cloned()
        .and_then(|id| jobs_now.iter().find(|j| j.id == id).cloned());

    rsx! {
        div { class: "split",
            div { class: "card",
                div { class: "card-title", "New training job" }
                div { class: "grid",
                    KindSelector { kind: model_kind, on_change: on_kind_change }
                    TextField { label: "Data path".to_string(), value: data_path }
                    TextField { label: "Output dir".to_string(), value: output_dir }
                    NumberFieldU32 { label: "Epochs".to_string(), value: epochs }
                    NumberFieldU32 { label: "Batch size".to_string(), value: batch_size }
                    NumberFieldF64 { label: "Learning rate".to_string(), value: lr, step: 1e-4 }
                    NumberFieldF64 { label: "Physics weight".to_string(), value: physics_weight, step: 0.01 }
                    NumberFieldF64 { label: "Val fraction".to_string(), value: val_frac, step: 0.01 }
                    NumberFieldU32 { label: "GPU index".to_string(), value: gpu_index }
                    NumberFieldU32 { label: "Patience".to_string(), value: patience }
                    NumberFieldU32 { label: "Grad accum".to_string(), value: grad_accum }
                    NumberFieldF64 { label: "Clip grad norm".to_string(), value: clip_grad_norm, step: 0.1 }
                    if model_kind() == ModelKind::Gnn {
                        NumberFieldU32 { label: "k-NN k".to_string(), value: knn_k }
                        NumberFieldU32 { label: "Hidden dim".to_string(), value: hidden_dim }
                    }
                    if model_kind() == ModelKind::Siren {
                        NumberFieldU32 { label: "Texture size".to_string(), value: texture_size }
                        NumberFieldU32 { label: "Max stars".to_string(), value: max_stars }
                    }
                    if model_kind() == ModelKind::GnnLocalization {
                        NumberFieldF64 { label: "Radius (pc)".to_string(), value: radius, step: 1.0 }
                        NumberFieldU32 { label: "Max slots".to_string(), value: max_slots }
                        NumberFieldF64 { label: "Mask ratio".to_string(), value: mask_ratio, step: 0.05 }
                        NumberFieldU32 { label: "Latent dim".to_string(), value: latent_dim }
                    }
                    TextField { label: "Resume from (optional)".to_string(), value: resume }
                    TextField { label: "Holdout (optional)".to_string(), value: holdout }
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
                        span { "Launch training" }
                    }
                }
                div { class: "field-hint", style: "margin-top: 8px;",
                    "Spawns the corresponding trainer binary (lnai, lnai-gnn, or lnai-siren) as a subprocess."
                }
            }
            div { class: "card",
                JobsList {
                    jobs: jobs_now.clone(),
                    selected,
                }
                if let Some(job) = current {
                    JobDetail { job: job.clone(), on_cancel: move |id: String| {
                        let mut res = jobs_resource;
                        spawn(async move {
                            let _ = api::cancel_job(&id).await;
                            res.restart();
                        });
                    } }
                } else {
                    div { class: "empty", "Select a job to see live metrics" }
                }
            }
        }
    }
}

#[component]
fn JobsList(jobs: Vec<Job>, mut selected: Signal<Option<String>>) -> Element {
    rsx! {
        div { class: "card-title", "Active / recent jobs" }
        if jobs.is_empty() {
            div { class: "empty", "No jobs yet" }
        } else {
            table { class: "tbl",
                thead {
                    tr {
                        th { "" }
                        th { "Title" }
                        th { "Status" }
                        th { "Best val" }
                        th { "Epoch" }
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
                                    td { "{j.last_metrics.len()} / {j.total_epochs_planned}" }
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
    let progress_class = match &job.spec {
        lunar_structures_testbench::JobKind::Train(_) => "pinn",
        lunar_structures_testbench::JobKind::Validate(_) => "siren",
        _ => "pinn",
    };

    let is_running = job.status == JobStatus::Running;
    let is_queued = job.status == JobStatus::Queued;

    let elapsed = job
        .started_ms
        .or(Some(job.created_ms))
        .map(|s| fmt_ms((crate::components::ui::now_ms()) - (s as i64)))
        .unwrap_or_else(|| "—".to_string());

    let last_metric = job.last_metrics.last().cloned();

    rsx! {
        div { class: "section-title", "Job · {job.title}" }
        div { class: "card",
            div { class: "row", style: "justify-content: space-between;",
                div { class: "row",
                    Tag { text: status.to_uppercase(), kind: status.clone() }
                    span { class: "mono", style: "color: var(--text-3);", "elapsed {elapsed}" }
                }
                div { class: "row",
                    if is_running || is_queued {
                        button {
                            class: "btn btn-danger btn-sm",
                            onclick: move |_| on_cancel.call(job.id.clone()),
                            "Cancel"
                        }
                    } else if let Some(code) = job.exit_code {
                        span { class: "mono", style: "color: var(--text-3);", "exit {code}" }
                    }
                }
            }
            ProgressBar { value: progress, kind: progress_class.to_string() }
            div { class: "row", style: "margin-top: 10px;",
                {
                    if let Some(m) = &last_metric {
                        let train_str = format!("{:.5}", m.train_loss);
                        let val_str = format!("{:.5}", m.val_loss);
                        let phys_str = m.phys_loss.map(|p| format!("{:.5}", p));
                        let lr_str = format!("{:.2e}", m.lr);
                        rsx! {
                            div { class: "row",
                                span { class: "metric-label", "train" }
                                span { class: "mono", style: "color: var(--accent);", "{train_str}" }
                            }
                            div { class: "row",
                                span { class: "metric-label", "val" }
                                span { class: "mono", style: "color: var(--gnn);", "{val_str}" }
                            }
                            if let Some(p) = &phys_str {
                                div { class: "row",
                                    span { class: "metric-label", "phys" }
                                    span { class: "mono", style: "color: var(--pinn);", "{p}" }
                                }
                            }
                            div { class: "row",
                                span { class: "metric-label", "lr" }
                                span { class: "mono", style: "color: var(--text-2);", "{lr_str}" }
                            }
                        }
                    } else {
                        rsx! { span { class: "metric-label", "Waiting for first epoch…" } }
                    }
                }
            }
        }
        div { class: "section-title", "Loss curve" }
        LossChart { metrics: job.last_metrics.clone(), width: 720.0, height: 260.0 }
        div { class: "section-title", "Log tail" }
        div { class: "log-view",
            for entry in job.log_tail.iter() {
                div {
                    class: match entry.kind {
                        lunar_structures_testbench::LogLineKind::Error => "log-line err",
                        lunar_structures_testbench::LogLineKind::Warning => "log-line warn",
                        lunar_structures_testbench::LogLineKind::Checkpoint => "log-line ok",
                        _ => "log-line",
                    },
                    "{entry.line}"
                }
            }
        }
    }
}

#[component]
fn KindSelector(kind: Signal<ModelKind>, on_change: EventHandler<ModelKind>) -> Element {
    let options = [
        (ModelKind::Pinn, "PINN (Stellar MLP)", "pinn"),
        (ModelKind::Gnn, "GNN (Stellar GCN)", "gnn"),
        (ModelKind::Siren, "SIREN (Texture)", "siren"),
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
                        span { class: "tag tag-{tag}", style: "margin-right: 6px;", "{m.label()}" }
                        span { "{label}" }
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
    fn test_training_snapshot_capture_and_hydration() {
        let original = TrainingSnapshot {
            selected: Some("job-123".into()),
            model_kind: ModelKind::Gnn,
            data_path: "ai_data/custom.parquet".into(),
            output_dir: "models/custom".into(),
            epochs: 100,
            batch_size: 4096,
            lr: 1e-3,
            physics_weight: 0.2,
            val_frac: 0.15,
            gpu_index: 1,
            patience: 30,
            grad_accum: 4,
            clip_grad_norm: 2.0,
            knn_k: 16,
            hidden_dim: 512,
            texture_size: 128,
            max_stars: 10000,
            resume: "models/prev.pt".into(),
            holdout: "ai_data/test.parquet".into(),
            error: Some("sample err".into()),
            starting: true,
        };
        let payload = original.capture_snapshot();
        let mut restored = TrainingSnapshot::default();
        restored.hydrate_snapshot(&payload).unwrap();
        assert_eq!(restored, original);
    }
}
