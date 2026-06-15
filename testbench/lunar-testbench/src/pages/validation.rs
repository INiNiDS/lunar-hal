use crate::api;
use crate::components::ui::{fmt_age, fmt_ms, LossChart, NumberFieldF64, NumberFieldU32, PageHeader, ProgressBar, StatusDot, Tag, TextField, tokio_time_sleep};
use dioxus::prelude::*;
use lunar_structures_testbench::{Job, JobStatus, ModelKind, ValidateSpec};

#[component]
pub fn Validation() -> Element {
    let jobs = use_resource(|| async { api::list_jobs().await.ok() });
    let selected = use_signal(|| None::<String>);

    rsx! {
        PageHeader {
            title: "Validation".to_string(),
            subtitle: "Run pure validation jobs: keep patience high, drive 1-3 epochs, capture the loss curve, and use it as a holdout estimate.".to_string(),
        }
        div { class: "page",
            ValidationBody {
                jobs: jobs.clone(),
                selected: selected.clone(),
            }
        }
    }
}

#[component]
fn ValidationBody(
    jobs: Resource<Option<Vec<Job>>>,
    mut selected: Signal<Option<String>>,
) -> Element {
    let mut model_kind = use_signal(|| ModelKind::Pinn);
    let mut data_path = use_signal(|| "ai_data/clean_stars2.parquet".to_string());
    let output_dir = use_signal(|| "models/validation".to_string());
    let mut epochs = use_signal(|| 3_u32);
    let mut batch_size = use_signal(|| 2048_u32);
    let val_frac = use_signal(|| 0.5_f64);
    let knn_k = use_signal(|| 8_u32);
    let hidden_dim = use_signal(|| 256_u32);
    let texture_size = use_signal(|| 32_u32);
    let max_stars = use_signal(|| 1000_u32);
    let mut error = use_signal(|| None::<String>);
    let mut starting = use_signal(|| false);

    let on_kind_change = move |k: ModelKind| {
        model_kind.set(k.clone());
        match k {
            ModelKind::Pinn => { data_path.set("ai_data/clean_stars2.parquet".to_string()); epochs.set(3); batch_size.set(2048); }
            ModelKind::Gnn => { data_path.set("ai_data/clean_gnn_stars.parquet".to_string()); epochs.set(2); batch_size.set(4096); }
            ModelKind::Siren => { data_path.set("ai_data/clean_stars2.parquet".to_string()); epochs.set(1); batch_size.set(512); }
        }
    };

    let submit = move |_| {
        error.set(None);
        starting.set(true);
        let spec = ValidateSpec {
            model: model_kind(),
            data_path: data_path(),
            output_dir: output_dir(),
            epochs: epochs(),
            batch_size: batch_size(),
            val_frac: val_frac() as f32,
            hidden_dim: if model_kind() == ModelKind::Gnn { Some(hidden_dim()) } else { None },
            knn_k: if model_kind() == ModelKind::Gnn { Some(knn_k()) } else { None },
            texture_size: if model_kind() == ModelKind::Siren { Some(texture_size()) } else { None },
            max_stars: if model_kind() == ModelKind::Siren { Some(max_stars()) } else { None },
        };
        let mut res = jobs.clone();
        spawn(async move {
            match api::start_validate(spec).await {
                Ok(job) => selected.set(Some(job.id)),
                Err(e) => error.set(Some(e)),
            }
            starting.set(false);
            res.restart();
        });
    };

    use_future(move || async move {
        loop {
            tokio_time_sleep(2000).await;
            jobs.restart();
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
                    NumberFieldU32 { label: "Epochs".to_string(), value: epochs }
                    NumberFieldU32 { label: "Batch size".to_string(), value: batch_size }
                    NumberFieldF64 { label: "Val fraction".to_string(), value: val_frac, step: 0.05 }
                    if model_kind() == ModelKind::Gnn {
                        NumberFieldU32 { label: "k-NN k".to_string(), value: knn_k }
                        NumberFieldU32 { label: "Hidden dim".to_string(), value: hidden_dim }
                    }
                    if model_kind() == ModelKind::Siren {
                        NumberFieldU32 { label: "Texture size".to_string(), value: texture_size }
                        NumberFieldU32 { label: "Max stars".to_string(), value: max_stars }
                    }
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
                        span { "Run validation" }
                    }
                }
                div { class: "field-hint", style: "margin-top: 8px;",
                    "Same trainer binary, but patience is forced high. Use small epoch counts to keep the run quick."
                }
            }
            div { class: "card",
                JobsList {
                    jobs: jobs_now.clone(),
                    selected: selected.clone(),
                }
                if let Some(job) = current {
                    JobDetail { job: job.clone(), on_cancel: move |id: String| {
                        let mut res = jobs.clone();
                        spawn(async move {
                            let _ = api::cancel_job(id).await;
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
        (ModelKind::Gnn, "GNN", "gnn"),
        (ModelKind::Siren, "SIREN", "siren"),
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


