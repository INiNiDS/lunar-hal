use crate::api;
use crate::components::ui::{fmt_age, fmt_ms, LossChart, NumberFieldF64, NumberFieldU32, PageHeader, ProgressBar, StatusDot, Tag, TextField, tokio_time_sleep};
use dioxus::prelude::*;
use lunar_structures_testbench::{Job, JobStatus, ModelKind, TrainSpec};

#[component]
pub fn Training() -> Element {
    let jobs = use_resource(|| async { api::list_jobs().await.ok() });
    let selected = use_signal(|| None::<String>);

    rsx! {
        PageHeader {
            title: "Training".to_string(),
            subtitle: "Launch training jobs against the existing trainers (lnai, lnai-gnn, lnai-siren). Watch live logs, follow epoch metrics, and inspect artifacts.".to_string(),
        }
        div { class: "page",
            TrainingBody {
                jobs_resource: jobs.clone(),
                selected: selected.clone(),
            }
        }
    }
}

#[component]
fn TrainingBody(
    jobs_resource: Resource<Option<Vec<Job>>>,
    mut selected: Signal<Option<String>>,
) -> Element {
    let mut model_kind = use_signal(|| ModelKind::Pinn);
    let mut data_path = use_signal(|| "ai_data/clean_stars2.parquet".to_string());
    let output_dir = use_signal(|| "models".to_string());
    let mut epochs = use_signal(|| 50_u32);
    let mut batch_size = use_signal(|| 2048_u32);
    let mut lr = use_signal(|| 5e-4_f64);
    let physics_weight = use_signal(|| 0.1_f64);
    let val_frac = use_signal(|| 0.1_f64);
    let gpu_index = use_signal(|| 0_u32);
    let patience = use_signal(|| 20_u32);
    let grad_accum = use_signal(|| 2_u32);
    let clip_grad_norm = use_signal(|| 1.0_f64);
    let knn_k = use_signal(|| 8_u32);
    let hidden_dim = use_signal(|| 256_u32);
    let texture_size = use_signal(|| 64_u32);
    let max_stars = use_signal(|| 5000_u32);
    let resume = use_signal(|| String::new());
    let holdout = use_signal(|| String::new());
    let mut error = use_signal(|| None::<String>);
    let mut starting = use_signal(|| false);

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
        }
    };

    let submit = move |_| {
        error.set(None);
        starting.set(true);
        let spec = TrainSpec {
            model: model_kind(),
            epochs: epochs(),
            batch_size: batch_size(),
            lr: lr(),
            physics_weight: physics_weight(),
            val_frac: val_frac() as f32,
            data_path: data_path(),
            output_dir: output_dir(),
            resume_from: if resume().is_empty() { None } else { Some(resume()) },
            holdout: if holdout().is_empty() { None } else { Some(holdout()) },
            gpu_index: gpu_index(),
            knn_k: if model_kind() == ModelKind::Gnn { Some(knn_k()) } else { None },
            hidden_dim: if model_kind() == ModelKind::Gnn { Some(hidden_dim()) } else { None },
            texture_size: if model_kind() == ModelKind::Siren { Some(texture_size()) } else { None },
            max_stars: if model_kind() == ModelKind::Siren { Some(max_stars()) } else { None },
            patience: patience(),
            grad_accum: grad_accum(),
            clip_grad_norm: clip_grad_norm(),
        };
        let mut res = jobs_resource.clone();
        spawn(async move {
            match api::start_train(spec).await {
                Ok(job) => selected.set(Some(job.id)),
                Err(e) => error.set(Some(e)),
            }
            starting.set(false);
            res.restart();
        });
    };

    let mut tick = use_signal(|| 0);
    use_future(move || async move {
        loop {
            tokio_time_sleep(2000).await;
            tick.set(tick() + 1);
            jobs_resource.restart();
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
                    selected: selected.clone(),
                }
                if let Some(job) = current {
                    JobDetail { job: job.clone(), on_cancel: move |id: String| {
                        let mut res = jobs_resource.clone();
                        spawn(async move {
                            let _ = api::cancel_job(id).await;
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


