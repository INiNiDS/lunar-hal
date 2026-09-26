use lunar_structures_testbench::{
    EpochMetric, Job, JobKind, JobStatus, ModelKind, ReportRecord, TrainSpec,
};
use lunar_testbench_backend::db::Database;
use lunar_testbench_backend::jobs::JobRegistry;
use lunar_testbench_backend::reports::compute_baseline_delta;
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;

#[test]
fn sqlite_schema_initialization_and_crud() {
    let db = Database::open_in_memory().expect("open in-memory db");
    let spec = TrainSpec {
        model: ModelKind::Pinn,
        epochs: 5,
        batch_size: 128,
        lr: 0.001,
        physics_weight: 0.2,
        val_frac: 0.1,
        data_path: "data.parquet".into(),
        output_dir: "out".into(),
        resume_from: None,
        holdout: None,
        gpu_index: 0,
        knn_k: None,
        hidden_dim: None,
        texture_size: None,
        max_stars: None,
        patience: 3,
        grad_accum: 1,
        clip_grad_norm: 1.0,
        radius: None,
        max_slots: None,
        mask_ratio: None,
        latent_dim: None,
    };
    let mut job = Job::new(JobKind::Train(spec), "test train".into(), 5);
    job.last_metrics.push(EpochMetric {
        epoch: 1,
        train_loss: 0.05,
        val_loss: 0.04,
        phys_loss: Some(0.01),
        lr: 0.001,
        timestamp_ms: 1000,
    });

    // Upsert and get
    db.upsert_job(&job, Some("{\"version\":\"1.0\"}"))
        .expect("upsert job");
    let retrieved = db.get_job(&job.id).expect("get job").expect("job exists");
    assert_eq!(retrieved.id, job.id);
    assert_eq!(retrieved.title, "test train");
    assert_eq!(retrieved.last_metrics.len(), 1);
    assert_eq!(retrieved.last_metrics[0].train_loss, 0.05);

    // List
    let list = db.list_jobs().expect("list jobs");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].id, job.id);
}

#[test]
fn persistence_across_simulated_restart() {
    let temp_dir =
        std::env::temp_dir().join(format!("testbench-db-{}", uuid::Uuid::new_v4().simple()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let db_path = temp_dir.join("test.db");

    let job_id = {
        let db = Database::open(&db_path).expect("open db");
        let registry = JobRegistry::new(db);
        let spec = TrainSpec {
            model: ModelKind::Pinn,
            epochs: 10,
            batch_size: 64,
            lr: 0.001,
            physics_weight: 0.1,
            val_frac: 0.1,
            data_path: "data.parquet".into(),
            output_dir: "out".into(),
            resume_from: None,
            holdout: None,
            gpu_index: 0,
            knn_k: None,
            hidden_dim: None,
            texture_size: None,
            max_stars: None,
            patience: 3,
            grad_accum: 1,
            clip_grad_norm: 1.0,
            radius: None,
            max_slots: None,
            mask_ratio: None,
            latent_dim: None,
        };
        let mut job = Job::new(JobKind::Train(spec), "interrupted train".into(), 10);
        job.status = JobStatus::Running;
        let id = job.id.clone();
        registry
            .db()
            .upsert_job(&job, None)
            .expect("insert running job");
        id
    };

    // Simulate backend restart by creating a new Database connection & registry
    {
        let db = Database::open(&db_path).expect("reopen db");
        let registry = JobRegistry::new(db);
        let restored_job = registry.get(&job_id).expect("job should be in registry");
        assert_eq!(restored_job.status, JobStatus::Failed);
        assert!(
            restored_job
                .error_summary
                .as_deref()
                .unwrap_or("")
                .contains("Interrupted")
        );
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn report_storage_and_baseline_delta() {
    let db = Database::open_in_memory().expect("open db");
    let ws = PathBuf::from(".");

    let report_payload = serde_json::json!({
        "avg_latency_ms": 0.45,
        "iterations": 100,
        "p95_latency_ms": 0.52
    });

    let delta = compute_baseline_delta("benchmark", "pinn", &report_payload, &ws);
    assert!(delta.is_some());
    let delta_val = delta.unwrap();
    assert_eq!(delta_val["model"], "pinn");

    let report = ReportRecord {
        id: "rep-test-123".into(),
        job_id: "job-xyz".into(),
        kind: "benchmark".into(),
        baseline_delta_json: Some(delta_val),
        report_json: report_payload,
        created_at: 12345678,
    };

    db.insert_report(&report).expect("insert report");

    let fetched = db
        .get_report("rep-test-123")
        .expect("get report")
        .expect("report exists");
    assert_eq!(fetched.id, "rep-test-123");
    assert_eq!(fetched.kind, "benchmark");
    assert!(fetched.baseline_delta_json.is_some());

    let all_reports = db.list_reports().expect("list reports");
    assert_eq!(all_reports.len(), 1);
    assert_eq!(all_reports[0].id, "rep-test-123");
}

#[cfg(unix)]
#[tokio::test]
async fn job_cancellation_terminates_and_updates_sqlite() {
    let db = Database::open_in_memory().expect("open db");
    let registry = JobRegistry::new(db);

    let mut command = Command::new("sleep");
    command.arg("30");
    let spec = TrainSpec {
        model: ModelKind::Pinn,
        epochs: 5,
        batch_size: 64,
        lr: 0.001,
        physics_weight: 0.1,
        val_frac: 0.1,
        data_path: "data.parquet".into(),
        output_dir: "out".into(),
        resume_from: None,
        holdout: None,
        gpu_index: 0,
        knn_k: None,
        hidden_dim: None,
        texture_size: None,
        max_stars: None,
        patience: 3,
        grad_accum: 1,
        clip_grad_norm: 1.0,
        radius: None,
        max_slots: None,
        mask_ratio: None,
        latent_dim: None,
    };
    let job = Job::new(JobKind::Train(spec), "cancellation test".into(), 5);
    let id = registry.spawn(job, command).expect("spawn job");

    // Give it a moment to enter running state
    tokio::time::sleep(Duration::from_millis(50)).await;

    registry.cancel(&id).expect("cancel job");

    let completed = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let job = registry.get(&id).expect("job exists");
            if matches!(
                job.status,
                JobStatus::Cancelled | JobStatus::Completed | JobStatus::Failed
            ) {
                break job;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("job should finish cancellation promptly");

    assert_eq!(completed.status, JobStatus::Cancelled);
    assert!(completed.finished_ms.is_some());

    // Also assert it is recorded as Cancelled directly in the database
    let db_job = registry
        .db()
        .get_job(&id)
        .expect("query db")
        .expect("job in db");
    assert_eq!(db_job.status, JobStatus::Cancelled);
}
