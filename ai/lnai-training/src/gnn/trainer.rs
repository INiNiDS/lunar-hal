//! Stage 5 (task 4): GNN-Kinematics trainer — moved verbatim from `ai/lnai-gnn/src/main.rs`.
//!
//! Deltas vs the pre-Stage-5 binary: seeded split/shuffle/group build,
//! typed NDJSON epoch events, `artifact.json` bundle, cooperative
//! cancellation, plus read-only `run_evaluate` and `run_benchmark` modes.

use super::dataset::{GnnDataset, GnnNormParams, PrefetchBatchedBatcher};
use super::loss::{compute_gnn_loss, compute_gnn_physics_loss};
use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::cuda::CudaDevice;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::ElementConversion;
use burn_store::{BurnpackStore, ModuleSnapshot};
use lnai_models::{GNN_INPUT_DIM, GNN_OUTPUT_DIM, StellarGnn, StellarGnnConfig};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::events::{EpochMetric, JobEvent, format_epoch_line};
use crate::runner::{CancelFlag, RunOutcome, append_event_line, effective_train_seed};
use crate::spec::TrainingSpec;

type TrainBackend = Autodiff<burn::backend::Cuda<f32, i32>>;
type InferBackend = burn::backend::Cuda<f32, i32>;

/// Shared stdout + NDJSON epoch sink.
struct EventSink {
    output_dir: std::path::PathBuf,
    total_epochs: u32,
}

impl EventSink {
    fn emit_epoch(&self, epoch: u32, train_loss: f64, val_loss: f64, phys_loss: f64, lr: f64) {
        let _ = append_event_line(
            &self.output_dir,
            &JobEvent::Metric(EpochMetric {
                epoch,
                train_loss,
                val_loss,
                phys_loss: Some(phys_loss),
                lr,
                timestamp_ms: lunar_utils::time::current_time_ms(),
            }),
        );
        let _ = append_event_line(
            &self.output_dir,
            &JobEvent::Progress {
                epoch,
                total_epochs: self.total_epochs,
            },
        );
    }
}

pub fn run_train(spec: &TrainingSpec) -> Result<RunOutcome> {
    run_train_with_cancel(spec, &CancelFlag::new())
}

#[allow(clippy::too_many_lines)]
pub fn run_train_with_cancel(spec: &TrainingSpec, cancel: &CancelFlag) -> Result<RunOutcome> {
    spec.validate()
        .map_err(|errs| anyhow::anyhow!("invalid GNN spec: {}", errs.join("; ")))?;
    let gnn_cfg = match &spec.config {
        crate::spec::ModelConfig::GnnKinematics(cfg) => cfg.clone(),
        other => anyhow::bail!("GNN trainer requires gnn_kinematics config, got {other:?}"),
    };
    let seed = effective_train_seed(spec);

    let device = CudaDevice::new(spec.gpu_index as usize);
    println!("Compute device: Cuda({})", spec.gpu_index);

    let data_path = match &spec.data_path {
        Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => find_parquet()?,
    };

    let output_dir = Path::new(&spec.output_dir);
    std::fs::create_dir_all(output_dir).map_err(|e| {
        anyhow::anyhow!(
            "failed to create output dir {}: {}",
            output_dir.display(),
            e
        )
    })?;
    let out_model_path = output_dir.join(&spec.model_file);
    let out_norm_path = output_dir.join(&spec.norm_file);
    let sink = EventSink {
        output_dir: output_dir.to_path_buf(),
        total_epochs: spec.epochs,
    };
    let _ = append_event_line(output_dir, &JobEvent::Started);

    let (mut model, norm, dataset): (StellarGnn<TrainBackend>, GnnNormParams, GnnDataset) =
        if let Some(resume_dir) = &spec.resume_from {
            let resume_path = Path::new(resume_dir);
            let model_path = resume_path.join(&spec.model_file);
            let norm_path = resume_path.join(&spec.norm_file);

            if !model_path.exists() {
                anyhow::bail!(
                    "Resume requested but model file not found: {}",
                    model_path.display()
                );
            }
            if !norm_path.exists() {
                anyhow::bail!(
                    "Resume requested but norm file not found: {}",
                    norm_path.display()
                );
            }

            let norm_json = std::fs::read_to_string(&norm_path)?;
            let loaded_norm: GnnNormParams = serde_json::from_str(&norm_json)?;
            println!("Resuming from model: {}", model_path.display());

            let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
            let mut loaded_model =
                StellarGnnConfig::new(GNN_INPUT_DIM, gnn_cfg.hidden_dim as usize, GNN_OUTPUT_DIM)
                    .init::<TrainBackend>(&device);
            loaded_model
                .load_from(&mut store)
                .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

            let dataset = GnnDataset::load_with_norm_and_seed(
                data_path.as_path(),
                loaded_norm.clone(),
                gnn_cfg.knn_k as usize,
                gnn_cfg.max_group_size as usize,
                gnn_cfg.radius_pc,
                seed,
            )?;

            (loaded_model, loaded_norm, dataset)
        } else {
            let dataset = GnnDataset::load_with_seed(
                data_path.as_path(),
                gnn_cfg.knn_k as usize,
                gnn_cfg.max_group_size as usize,
                gnn_cfg.radius_pc,
                seed,
            )?;
            let norm = dataset.norm.clone();
            let fresh =
                StellarGnnConfig::new(GNN_INPUT_DIM, gnn_cfg.hidden_dim as usize, GNN_OUTPUT_DIM)
                    .init::<TrainBackend>(&device);
            (fresh, norm, dataset)
        };

    let (mut train_ds, val_ds) = dataset.split_with_seed(spec.val_frac, seed);

    let n_params: usize = model.num_params();

    let max_nodes = spec.batch_size as usize;
    let max_adj_bytes = max_nodes as u64 * max_nodes as u64 * 4;
    let max_adj_mb = max_adj_bytes / (1024 * 1024);

    println!();
    println!("=== GNN Model Architecture ===");
    println!(
        "Input dim:             {} (node features from Model 1)",
        GNN_INPUT_DIM
    );
    println!("Hidden dim:            {}", gnn_cfg.hidden_dim);
    println!("Output dim:            {} (Vx, Vy, Vz)", GNN_OUTPUT_DIM);
    println!("Total parameters:      {}", n_params);
    println!("k-NN neighbors:        {}", gnn_cfg.knn_k);
    println!("Max group size:        {}", gnn_cfg.max_group_size);
    println!("Group radius:          {} pc", gnn_cfg.radius_pc);
    println!("===============================");
    println!();

    let mut optim = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(
            spec.clip_grad_norm as f32,
        )))
        .init();

    println!();
    println!("=== Training Config ===");
    println!(
        "Max nodes per batch:   {} (~{} MB adj matrix)",
        max_nodes, max_adj_mb
    );
    println!("Grad accumulation:     {}", spec.grad_accum);
    println!("Grad clip norm:        {}", spec.clip_grad_norm);
    println!("Early stop patience:   {}", spec.patience);
    println!("Physics weight:        {}", gnn_cfg.physics_weight);
    println!("=========================");
    println!();

    println!("Seed:                {seed}");
    println!(
        "{:>5} | {:>12} | {:>12} | {:>12} | {:>10}",
        "epoch", "train_loss", "val_loss", "phys_loss", "lr"
    );
    println!("{}", "-".repeat(65));

    let mut best_val_loss = f64::MAX;
    let mut epochs_without_improvement = 0usize;
    let initial_lr = spec.lr;

    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = interrupted.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nCtrl+C received, finishing current epoch and saving model...");
        interrupted_clone.store(true, Ordering::SeqCst);
    })
    .expect("failed to set Ctrl+C handler");

    let epochs = spec.epochs as usize;
    for epoch in 1..=epochs {
        if interrupted.load(Ordering::SeqCst) || cancel.is_cancelled() {
            println!("\nInterrupted at epoch {epoch}. Saving checkpoint...");
            save_checkpoint(&model, &norm, &out_model_path, &out_norm_path)?;
            println!("Checkpoint saved to: {}", out_model_path.display());
            let _ = append_event_line(output_dir, &JobEvent::Cancelled);
            return Ok(RunOutcome::Cancelled);
        }

        let lr = cosine_annealing(epoch, epochs, initial_lr, 1e-6);

        train_ds.shuffle_with_seed(seed.wrapping_add(epoch as u64));
        let mut prefetcher = PrefetchBatchedBatcher::new(&train_ds, max_nodes);
        let mut epoch_train_loss = 0.0f64;
        let mut n_batches = 0usize;

        let mut accumulator: GradientsAccumulator<StellarGnn<TrainBackend>> =
            GradientsAccumulator::new();
        let mut accum_count = 0usize;

        while let Some((nodes, adj, targets)) = prefetcher.next_batch::<TrainBackend>(&device) {
            let predictions = model.forward(nodes, adj);
            let loss = compute_gnn_physics_loss(predictions, targets, gnn_cfg.physics_weight);

            let loss_scalar = loss.clone().into_scalar().elem::<f32>();
            let scaled_loss = loss.div_scalar(spec.grad_accum as f32);
            let grads = scaled_loss.backward();
            drop(scaled_loss);
            let grads = GradientsParams::from_grads(grads, &model);

            accumulator.accumulate(&model, grads);
            epoch_train_loss += loss_scalar as f64;
            n_batches += 1;
            accum_count += 1;

            if accum_count >= spec.grad_accum as usize {
                let grads = accumulator.grads();
                model = optim.step(lr, model, grads);
                accumulator = GradientsAccumulator::new();
                accum_count = 0;
            }
        }

        if accum_count > 0 {
            let grads = accumulator.grads();
            model = optim.step(lr, model, grads);
        }

        let epoch_train_loss = epoch_train_loss / n_batches.max(1) as f64;
        let infer_model = model.valid();
        let val_loss = evaluate_gnn(&infer_model, &val_ds, &device, max_nodes);
        let phys_loss = evaluate_physics(
            &infer_model,
            &val_ds,
            &device,
            gnn_cfg.physics_weight,
            max_nodes,
        );
        drop(infer_model);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            epochs_without_improvement = 0;
            save_checkpoint(&model, &norm, &out_model_path, &out_norm_path)?;
        } else {
            epochs_without_improvement += 1;
        }

        println!(
            "{}",
            format_epoch_line(
                epoch as u32,
                epoch_train_loss,
                val_loss,
                Some(phys_loss),
                lr
            )
        );
        sink.emit_epoch(epoch as u32, epoch_train_loss, val_loss, phys_loss, lr);

        if epochs_without_improvement >= spec.patience as usize {
            println!(
                "\nEarly stopping: no improvement for {} epochs.",
                spec.patience
            );
            let _ = append_event_line(output_dir, &JobEvent::Completed { exit_code: 0 });
            write_artifact_manifest(spec, seed, &norm, best_val_loss)?;
            return Ok(RunOutcome::EarlyStopped);
        }
    }

    println!();
    println!(
        "Training complete. Best validation loss: {:.6}",
        best_val_loss
    );
    println!("Best model saved to: {}", out_model_path.display());
    println!("Normalization params saved to: {}", out_norm_path.display());
    let _ = append_event_line(output_dir, &JobEvent::Completed { exit_code: 0 });
    write_artifact_manifest(spec, seed, &norm, best_val_loss)?;

    Ok(RunOutcome::Completed)
}

fn write_artifact_manifest(
    spec: &TrainingSpec,
    seed: u64,
    norm: &GnnNormParams,
    best_val_loss: f64,
) -> Result<()> {
    use crate::artifacts::{
        ArtifactManifestV1, architecture_version, sha256_file_hex, write_artifact_bundle,
    };
    use crate::spec::ModelKind;

    let output_dir = Path::new(&spec.output_dir);
    let model_hash = sha256_file_hex(&output_dir.join(&spec.model_file)).unwrap_or_default();
    let norm_json = serde_json::to_string(norm).unwrap_or_default();
    let norm_hash = crate::e2e::sha256_hex(norm_json.as_bytes());
    let mut manifest = ArtifactManifestV1::new(
        ModelKind::GnnKinematics,
        architecture_version(&ModelKind::GnnKinematics).to_string(),
        model_hash,
        norm_hash,
        String::new(),
        "gaia_dr3".to_string(),
        spec.dataset_manifest_hash.clone(),
        seed,
        serde_json::to_value(&spec.config).unwrap_or(serde_json::Value::Null),
        option_env!("LUNAR_AI_GIT_REV")
            .unwrap_or("unknown")
            .to_string(),
        format!("cuda:{}", spec.gpu_index),
    );
    manifest.evaluation_metrics = Some(serde_json::json!({ "best_val_loss": best_val_loss }));
    let path = write_artifact_bundle(output_dir, &manifest)
        .map_err(|e| anyhow::anyhow!("failed to write artifact bundle: {e}"))?;
    println!("Artifact manifest: {}", path.display());
    Ok(())
}

/// Read-only evaluation: loads the artifact, reports losses, trains nothing.
pub fn run_evaluate(spec: &TrainingSpec) -> Result<RunOutcome> {
    spec.validate()
        .map_err(|errs| anyhow::anyhow!("invalid GNN spec: {}", errs.join("; ")))?;
    let gnn_cfg = match &spec.config {
        crate::spec::ModelConfig::GnnKinematics(cfg) => cfg.clone(),
        other => anyhow::bail!("GNN trainer requires gnn_kinematics config, got {other:?}"),
    };
    let seed = effective_train_seed(spec);
    let device = CudaDevice::new(spec.gpu_index as usize);
    let data_path = match &spec.data_path {
        Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => find_parquet()?,
    };
    let output_dir = Path::new(&spec.output_dir);
    let model_path = output_dir.join(&spec.model_file);
    let norm_path = output_dir.join(&spec.norm_file);
    if !model_path.exists() {
        anyhow::bail!("evaluate: model file not found: {}", model_path.display());
    }
    if !norm_path.exists() {
        anyhow::bail!("evaluate: norm file not found: {}", norm_path.display());
    }
    let norm_json = std::fs::read_to_string(&norm_path)?;
    let norm: GnnNormParams = serde_json::from_str(&norm_json)?;
    let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
    let mut model =
        StellarGnnConfig::new(GNN_INPUT_DIM, gnn_cfg.hidden_dim as usize, GNN_OUTPUT_DIM)
            .init::<TrainBackend>(&device);
    model
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;
    let dataset = GnnDataset::load_with_norm_and_seed(
        data_path.as_path(),
        norm,
        gnn_cfg.knn_k as usize,
        gnn_cfg.max_group_size as usize,
        gnn_cfg.radius_pc,
        seed,
    )?;
    let (_, val_ds) = dataset.split_with_seed(spec.val_frac, seed);
    let infer_model = model.valid();
    let max_nodes = spec.batch_size as usize;
    let val_loss = evaluate_gnn(&infer_model, &val_ds, &device, max_nodes);
    let phys_loss = evaluate_physics(
        &infer_model,
        &val_ds,
        &device,
        gnn_cfg.physics_weight,
        max_nodes,
    );
    drop(infer_model);
    println!("=== Read-only evaluation (no weight updates) ===");
    println!("Validation data loss:    {val_loss:.6}");
    println!("Validation physics loss: {phys_loss:.6}");
    println!("Evaluation complete; checkpoints untouched.");
    Ok(RunOutcome::Completed)
}

/// Benchmark harness: loads the artifact, times forward passes, trains nothing.
pub fn run_benchmark(spec: &TrainingSpec, iters: u32, warmup: u32) -> Result<RunOutcome> {
    use std::time::Instant;
    let gnn_cfg = match &spec.config {
        crate::spec::ModelConfig::GnnKinematics(cfg) => cfg.clone(),
        other => anyhow::bail!("GNN trainer requires gnn_kinematics config, got {other:?}"),
    };
    let device = CudaDevice::new(spec.gpu_index as usize);
    let output_dir = Path::new(&spec.output_dir);
    let model_path = output_dir.join(&spec.model_file);
    if !model_path.exists() {
        anyhow::bail!("benchmark: model file not found: {}", model_path.display());
    }
    let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
    let mut loaded =
        StellarGnnConfig::new(GNN_INPUT_DIM, gnn_cfg.hidden_dim as usize, GNN_OUTPUT_DIM)
            .init::<TrainBackend>(&device);
    loaded
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;
    let infer = loaded.valid();
    let n = 8usize.min(spec.batch_size.max(1) as usize);
    let nodes = burn::tensor::Tensor::<InferBackend, 2>::zeros([n, GNN_INPUT_DIM], &device);
    let adj = burn::tensor::Tensor::<InferBackend, 2>::zeros([n, n], &device);
    for _ in 0..warmup {
        let _ = infer.forward(nodes.clone(), adj.clone());
    }
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        let _ = infer.forward(nodes.clone(), adj.clone());
    }
    let elapsed = start.elapsed();
    let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / f64::from(iters.max(1));
    println!("=== Benchmark (forward only, no training) ===");
    println!("Iterations: {} (+{} warmup)", iters.max(1), warmup);
    println!(
        "Total: {:.2} ms, per-iter: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0,
        per_iter_ms
    );
    let report = serde_json::json!({
        "model": "gnn_kinematics",
        "batch_size": spec.batch_size,
        "iterations": iters.max(1),
        "warmup": warmup,
        "total_ms": elapsed.as_secs_f64() * 1000.0,
        "per_iter_ms": per_iter_ms,
    });
    crate::runner::write_checkpoint_sidecar(
        output_dir,
        "benchmark.json",
        &serde_json::to_string_pretty(&report).unwrap_or_default(),
    )?;
    Ok(RunOutcome::Completed)
}

fn save_checkpoint(
    model: &StellarGnn<TrainBackend>,
    norm: &GnnNormParams,
    model_path: &Path,
    norm_path: &Path,
) -> Result<()> {
    let mut store = BurnpackStore::from_file(model_path.to_str().unwrap()).overwrite(true);
    model
        .save_into(&mut store)
        .map_err(|e| anyhow::anyhow!("failed to save model: {e}"))?;
    let norm_json = serde_json::to_string_pretty(norm)?;
    std::fs::write(norm_path, norm_json)?;
    Ok(())
}

fn evaluate_gnn(
    model: &StellarGnn<InferBackend>,
    dataset: &GnnDataset,
    device: &CudaDevice,
    max_nodes: usize,
) -> f64 {
    let mut prefetcher = PrefetchBatchedBatcher::new(dataset, max_nodes);
    let mut total_loss = 0.0f64;
    let mut n = 0usize;

    while let Some((nodes, adj, targets)) = prefetcher.next_batch::<InferBackend>(device) {
        let preds = model.forward(nodes, adj);
        let loss = compute_gnn_loss(preds, targets);
        let value: f32 = loss.into_scalar().elem();
        total_loss += value as f64;
        n += 1;
    }

    total_loss / n.max(1) as f64
}

fn evaluate_physics(
    model: &StellarGnn<InferBackend>,
    dataset: &GnnDataset,
    device: &CudaDevice,
    physics_weight: f64,
    max_nodes: usize,
) -> f64 {
    let mut prefetcher = PrefetchBatchedBatcher::new(dataset, max_nodes);
    let mut total_loss = 0.0f64;
    let mut n = 0usize;

    while let Some((nodes, adj, targets)) = prefetcher.next_batch::<InferBackend>(device) {
        let preds = model.forward(nodes, adj);
        let loss = compute_gnn_physics_loss(preds, targets, physics_weight);
        let value: f32 = loss.into_scalar().elem();
        total_loss += value as f64;
        n += 1;
    }

    total_loss / n.max(1) as f64
}

fn cosine_annealing(epoch: usize, total_epochs: usize, initial_lr: f64, min_lr: f64) -> f64 {
    let progress = epoch as f64 / total_epochs as f64;
    min_lr + (initial_lr - min_lr) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

fn find_parquet() -> Result<std::path::PathBuf> {
    let candidates = [
        "ai_data/clean_gnn_stars.parquet",
        "ai_data/clean_stars2.parquet",
        "ai_data/clean_stars.parquet",
    ];
    for c in &candidates {
        let p = Path::new(c);
        if p.exists() {
            println!("Using dataset: {}", p.display());
            return Ok(p.to_path_buf());
        }
    }
    anyhow::bail!(
        "No parquet dataset found in ai_data/. Run 'lnaicli fetch-gnn && lnaicli clean-gnn' first."
    );
}
