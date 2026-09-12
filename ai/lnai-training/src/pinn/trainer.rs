//! Stage 5 (task 2): PINN trainer — moved verbatim from `ai/lnai/src/main.rs`.
//!
//! The only behavioural deltas vs the pre-Stage-5 binary:
//! * train/val split and batch order derive from the spec seed
//!   (`split_with_seed` / `new_with_seed`) so old/new parity is exact;
//! * every epoch additionally appends a typed [`JobEvent`](crate::events::JobEvent)
//!   line to `events.ndjson` via [`format_epoch_line`](crate::events::format_epoch_line)
//!   (same column layout as stdout, machine-readable).
//!
//! Everything else — architecture, loss, optimizer, early stopping,
//! checkpoint policy, holdout block — is untouched.

use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::cuda::CudaDevice;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use burn_store::{BurnpackStore, ModuleSnapshot};
use lnai_models::{MLP_INPUT_DIM, StellarMlp, StellarMlpConfig};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::dataset::{GpuBatcher, INPUT_DIM, NormParams, StellarDataset, TARGET_DIM};
use super::loss::{compute_data_loss, compute_physics_loss, compute_pinn_loss};
use crate::events::{EpochMetric, JobEvent, format_epoch_line};
use crate::runner::{CancelFlag, RunOutcome, append_event_line, effective_train_seed};
use crate::spec::TrainingSpec;

type TrainBackend = Autodiff<burn::backend::Cuda<f32, i32>>;
type InferBackend = burn::backend::Cuda<f32, i32>;

/// Shared stdout + NDJSON epoch sink: human-readable table on stdout,
/// typed metric on `events.ndjson`.
struct EventSink {
    output_dir: std::path::PathBuf,
    total_epochs: u32,
}

impl EventSink {
    fn emit_epoch(&self, epoch: u32, train_loss: f64, val_loss: f64, phys_loss: f64, lr: f64) {
        println!(
            "{}",
            format_epoch_line(epoch, train_loss, val_loss, Some(phys_loss), lr)
        );
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

/// Trains exactly the legacy `lnai` binary would for this spec.
pub fn run_train(spec: &TrainingSpec) -> Result<RunOutcome> {
    run_train_with_cancel(spec, &CancelFlag::new())
}

#[allow(clippy::too_many_lines)]
pub fn run_train_with_cancel(spec: &TrainingSpec, cancel: &CancelFlag) -> Result<RunOutcome> {
    spec.validate()
        .map_err(|errs| anyhow::anyhow!("invalid PINN spec: {}", errs.join("; ")))?;
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

    let (mut model, norm, dataset): (
        StellarMlp<TrainBackend>,
        NormParams,
        StellarDataset<TrainBackend>,
    ) = if let Some(resume_dir) = &spec.resume_from {
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
        let loaded_norm: NormParams = serde_json::from_str(&norm_json)?;
        println!("Resuming from model: {}", model_path.display());
        println!("Using norm from:     {}", norm_path.display());

        let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
        let mut loaded_model = StellarMlpConfig::new().init::<TrainBackend>(&device);
        loaded_model
            .load_from(&mut store)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        let dataset: StellarDataset<TrainBackend> =
            StellarDataset::load_with_norm(data_path.as_path(), loaded_norm.clone(), &device, spec.max_rows, spec.tiles.clone())?;

        println!("=== Fine-tuning mode (using loaded normalization) ===");
        (loaded_model, loaded_norm, dataset)
    } else {
        let dataset: StellarDataset<TrainBackend> =
            StellarDataset::load(data_path.as_path(), &device, spec.max_rows, spec.tiles.clone())?;
        let norm = dataset.norm.clone();
        let fresh = StellarMlpConfig::new().init::<TrainBackend>(&device);
        (fresh, norm, dataset)
    };

    let (train_ds, val_ds) = dataset.split_with_seed(spec.val_frac, seed);

    let n_params: usize = model.num_params();

    println!();
    println!("=== Model Architecture ===");
    println!(
        "Fourier levels:       {}",
        StellarMlpConfig::fourier_levels()
    );
    println!("Fourier dim:          {}", lnai_models::FOURIER_DIM);
    println!("Conditional inputs:   2 (bp_rp, M_G)");
    println!("MLP input dim:        {}", MLP_INPUT_DIM);
    println!("Input features:       5 (x, y, z, bp_rp, M_G)");
    println!("Output features:      4 (log10_teff, log10_rad, log10_mass, log10_lum)");
    println!("Total parameters:     {n_params}");
    println!("=========================");
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

    let effective_batch = spec.batch_size as usize * spec.grad_accum as usize;
    println!();
    println!("=== Training Config ===");
    println!("Micro-batch size:    {}", spec.batch_size);
    println!("Grad accumulation:   {}", spec.grad_accum);
    println!("Effective batch:     {effective_batch}");
    println!("Grad clip norm:      {}", spec.clip_grad_norm);
    println!("Early stop patience: {}", spec.patience);
    println!("Seed:                {seed}");
    println!("=========================");
    println!();

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
            let mut store =
                BurnpackStore::from_file(out_model_path.to_str().unwrap()).overwrite(true);
            model
                .save_into(&mut store)
                .expect("failed to save checkpoint");
            let norm_json = serde_json::to_string_pretty(&norm)?;
            std::fs::write(&out_norm_path, norm_json)?;
            println!("Checkpoint saved to: {}", out_model_path.display());
            let _ = append_event_line(output_dir, &JobEvent::Cancelled);
            return Ok(RunOutcome::Cancelled);
        }

        let lr = cosine_annealing(epoch, epochs, initial_lr, 1e-6);

        // Per-epoch batch order mixes the run seed with the epoch index so
        // every epoch is reproducible yet distinct.
        let mut batcher = GpuBatcher::new_with_seed(
            &train_ds,
            spec.batch_size as usize,
            seed.wrapping_add(epoch as u64),
        );
        let mut n_batches = 0usize;
        let mut loss_sum = Tensor::<TrainBackend, 1>::zeros([1], &device);

        let mut accumulator: GradientsAccumulator<StellarMlp<TrainBackend>> =
            GradientsAccumulator::new();
        let mut accum_count = 0usize;

        while let Some((batch_inputs, batch_targets)) = batcher.next_batch() {
            let physics_weight = match &spec.config {
                crate::spec::ModelConfig::Pinn(cfg) => cfg.physics_weight,
                _ => 0.1,
            };
            let predictions = model.forward(batch_inputs);
            let loss = compute_pinn_loss(predictions, batch_targets, physics_weight, &norm);

            let scaled_loss = if spec.grad_accum > 1 {
                loss.clone().div_scalar(spec.grad_accum as f32)
            } else {
                loss.clone()
            };
            loss_sum = loss_sum + loss.detach();
            let grads = scaled_loss.backward();
            drop(scaled_loss);
            let grads = GradientsParams::from_grads(grads, &model);

            accumulator.accumulate(&model, grads);
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

        let epoch_train_loss = if n_batches > 0 {
            let total: f32 = (loss_sum / n_batches as f32).into_scalar();
            total as f64
        } else {
            0.0
        };
        let infer_model = model.valid();
        let val_loss = evaluate_infer(
            &infer_model,
            &val_ds.inputs.clone().valid(),
            &val_ds.targets.clone().valid(),
            spec.batch_size as usize,
        );
        let phys_loss = evaluate_physics_infer(
            &infer_model,
            &val_ds.inputs.clone().valid(),
            spec.batch_size as usize,
            &norm,
        );
        drop(infer_model);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            epochs_without_improvement = 0;
            let mut store =
                BurnpackStore::from_file(out_model_path.to_str().unwrap()).overwrite(true);
            model
                .save_into(&mut store)
                .expect("failed to save best model");
            let norm_json = serde_json::to_string_pretty(&norm)?;
            std::fs::write(&out_norm_path, norm_json)?;
            let _ = append_event_line(
                output_dir,
                &JobEvent::Checkpoint {
                    epoch: epoch as u32,
                    path: out_model_path.display().to_string(),
                    hash: String::new(),
                },
            );
        } else {
            epochs_without_improvement += 1;
        }

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
    println!("Training complete. Best validation loss: {best_val_loss:.6}");
    println!("Best model saved to: {}", out_model_path.display());
    println!("Normalization params saved to: {}", out_norm_path.display());
    let _ = append_event_line(output_dir, &JobEvent::Completed { exit_code: 0 });
    write_artifact_manifest(spec, seed, &norm, best_val_loss)?;

    if let Some(holdout_path) = &spec.holdout {
        println!();
        println!("=== Holdout Evaluation ===");
        let holdout_path = Path::new(holdout_path);
        if holdout_path.exists() {
            let holdout_ds: StellarDataset<TrainBackend> =
                StellarDataset::load(holdout_path, &device, spec.max_rows, spec.tiles.clone())?;
            let (_, holdout_val) = holdout_ds.split_with_seed(0.0, seed);
            let infer_model = model.valid();
            let holdout_loss = evaluate_infer(
                &infer_model,
                &holdout_val.inputs.clone().valid(),
                &holdout_val.targets.clone().valid(),
                spec.batch_size as usize,
            );
            let holdout_phys = evaluate_physics_infer(
                &infer_model,
                &holdout_val.inputs.clone().valid(),
                spec.batch_size as usize,
                &norm,
            );
            drop(infer_model);
            println!("Holdout data loss:   {holdout_loss:.6}");
            println!("Holdout physics loss: {holdout_phys:.6}");

            if holdout_loss <= best_val_loss * 1.5 {
                println!("Holdout loss is close to validation loss - model generalizes well!");
            } else {
                println!("WARNING: Holdout loss is significantly higher than validation loss.");
                println!(
                    "         The model may be overfitting. Consider regularization or more data."
                );
            }
        } else {
            println!("Holdout file not found: {}", holdout_path.display());
        }
    }

    Ok(RunOutcome::Completed)
}

fn write_artifact_manifest(
    spec: &TrainingSpec,
    seed: u64,
    norm: &NormParams,
    best_val_loss: f64,
) -> Result<()> {
    use crate::artifacts::{
        ArtifactManifestV1, architecture_version, norm_file_name, sha256_file_hex,
        weight_file_name, write_artifact_bundle,
    };
    use crate::spec::ModelKind;

    let output_dir = Path::new(&spec.output_dir);
    let model_hash =
        sha256_file_hex(&output_dir.join(weight_file_name(&ModelKind::Pinn))).unwrap_or_default();
    let norm_json = serde_json::to_string(norm).unwrap_or_default();
    let norm_hash = crate::e2e::sha256_hex(norm_json.as_bytes());
    let mut manifest = ArtifactManifestV1::new(
        ModelKind::Pinn,
        architecture_version(&ModelKind::Pinn).to_string(),
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
    // Keep the legacy norm filename next to the manifest for old tooling.
    let _ = norm_file_name(&ModelKind::Pinn);
    println!("Artifact manifest: {}", path.display());
    Ok(())
}

/// Read-only evaluation: loads the artifact, reports validation (+holdout)
/// losses, runs no optimizer step and rewrites no checkpoint.
pub fn run_evaluate(spec: &TrainingSpec) -> Result<RunOutcome> {
    spec.validate()
        .map_err(|errs| anyhow::anyhow!("invalid PINN spec: {}", errs.join("; ")))?;
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
    let norm: NormParams = serde_json::from_str(&norm_json)?;
    let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
    let mut model = StellarMlpConfig::new().init::<TrainBackend>(&device);
    model
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

    let dataset: StellarDataset<TrainBackend> =
        StellarDataset::load_with_norm(data_path.as_path(), norm.clone(), &device, None, None)?;
    let (_, val_ds) = dataset.split_with_seed(spec.val_frac, seed);
    let infer_model = model.valid();
    let val_loss = evaluate_infer(
        &infer_model,
        &val_ds.inputs.clone().valid(),
        &val_ds.targets.clone().valid(),
        spec.batch_size as usize,
    );
    let phys_loss = evaluate_physics_infer(
        &infer_model,
        &val_ds.inputs.clone().valid(),
        spec.batch_size as usize,
        &norm,
    );
    println!("=== Read-only evaluation (no weight updates) ===");
    println!("Validation data loss:    {val_loss:.6}");
    println!("Validation physics loss: {phys_loss:.6}");
    let (pred_rows, truth_rows) = evaluate_per_target_infer(
        &infer_model,
        &val_ds.inputs.clone().valid(),
        &val_ds.targets.clone().valid(),
        spec.batch_size as usize,
    )?;
    print_per_target_table(
        "Validation",
        &crate::metrics::pinn::per_target_metrics(&pred_rows, &truth_rows),
        &norm,
    );

    if let Some(holdout_path) = &spec.holdout
        && Path::new(holdout_path).exists()
    {
        let holdout_ds: StellarDataset<TrainBackend> =
            StellarDataset::load(Path::new(holdout_path), &device, None, None)?;
        let (_, holdout_val) = holdout_ds.split_with_seed(0.0, seed);
        let holdout_loss = evaluate_infer(
            &infer_model,
            &holdout_val.inputs.clone().valid(),
            &holdout_val.targets.clone().valid(),
            spec.batch_size as usize,
        );
        println!("Holdout data loss:       {holdout_loss:.6}");
        let (pred_rows, truth_rows) = evaluate_per_target_infer(
            &infer_model,
            &holdout_val.inputs.clone().valid(),
            &holdout_val.targets.clone().valid(),
            spec.batch_size as usize,
        )?;
        print_per_target_table(
            "Holdout",
            &crate::metrics::pinn::per_target_metrics(&pred_rows, &truth_rows),
            &norm,
        );
    }
    drop(infer_model);
    println!("Evaluation complete; checkpoints untouched.");
    Ok(RunOutcome::Completed)
}

/// Benchmark harness: loads the artifact, times batched forward passes,
/// writes `benchmark.json`, trains nothing.
pub fn run_benchmark(spec: &TrainingSpec, iters: u32, warmup: u32) -> Result<RunOutcome> {
    use std::time::Instant;
    spec.validate()
        .map_err(|errs| anyhow::anyhow!("invalid PINN spec: {}", errs.join("; ")))?;
    let device = CudaDevice::new(spec.gpu_index as usize);
    let output_dir = Path::new(&spec.output_dir);
    let model_path = output_dir.join(&spec.model_file);
    if !model_path.exists() {
        anyhow::bail!("benchmark: model file not found: {}", model_path.display());
    }
    let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
    let model = StellarMlpConfig::new()
        .init::<TrainBackend>(&device)
        .valid();
    let mut loaded = StellarMlpConfig::new().init::<TrainBackend>(&device);
    loaded
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;
    let infer = loaded.valid();
    drop(model);

    let batch = spec.batch_size.max(1) as usize;
    let zeros = Tensor::<InferBackend, 2>::zeros(
        [batch, INPUT_DIM],
        &CudaDevice::new(spec.gpu_index as usize),
    );
    for _ in 0..warmup {
        let _ = infer.forward(zeros.clone());
    }
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        let _ = infer.forward(zeros.clone());
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
        "model": "pinn",
        "batch_size": batch,
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

/// Batched inference gathering predictions + truths on CPU for per-target
/// metrics. Same batching as [`evaluate_infer`]; read-only.
fn evaluate_per_target_infer(
    model: &StellarMlp<InferBackend>,
    inputs: &Tensor<InferBackend, 2>,
    targets: &Tensor<InferBackend, 2>,
    batch_size: usize,
) -> Result<(Vec<[f32; 4]>, Vec<[f32; 4]>)> {
    let [n, _] = inputs.dims();
    let mut pred_flat = Vec::with_capacity(n * TARGET_DIM);
    let mut truth_flat = Vec::with_capacity(n * TARGET_DIM);
    let mut current = 0usize;
    while current < n {
        let end = (current + batch_size).min(n);
        let preds = model.forward(inputs.clone().slice([current..end, 0..INPUT_DIM]));
        let truth = targets.clone().slice([current..end, 0..TARGET_DIM]);
        pred_flat.extend(
            preds
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("download preds: {e}"))?,
        );
        truth_flat.extend(
            truth
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("download truth: {e}"))?,
        );
        current = end;
    }
    let to_rows = |v: Vec<f32>| {
        v.chunks_exact(TARGET_DIM)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect::<Vec<_>>()
    };
    Ok((to_rows(pred_flat), to_rows(truth_flat)))
}

/// Prints per-target MSE/MAE/max-abs in normalized units plus MAE in dex
/// (mae_norm × target std), which is the physically readable number.
fn print_per_target_table(
    split: &str,
    metrics: &[crate::metrics::pinn::PerTargetMetrics],
    norm: &NormParams,
) {
    let std_of = |target: &str| match target {
        "log10_teff" => norm.log_teff_std,
        "log10_rad" => norm.log_rad_std,
        "log10_mass" => norm.log_mass_std,
        "log10_lum" => norm.log_lum_std,
        _ => 1.0,
    };
    println!("Per-target {split} metrics:");
    println!("target       |        mse |        mae |    max_abs |   mae_dex");
    for m in metrics {
        println!(
            "{:<12} | {:10.6} | {:10.6} | {:10.6} | {:10.6}",
            m.target,
            m.mse,
            m.mae,
            m.max_abs_err,
            m.mae as f32 * std_of(m.target),
        );
    }
}

fn evaluate_infer(
    model: &StellarMlp<InferBackend>,
    inputs: &Tensor<InferBackend, 2>,
    targets: &Tensor<InferBackend, 2>,
    batch_size: usize,
) -> f64 {
    let [n, _] = inputs.dims();
    if n == 0 {
        return 0.0;
    }

    let mut loss_sum = Tensor::<InferBackend, 1>::zeros([1], &inputs.device());
    let mut n_batches = 0usize;
    let mut current = 0usize;

    while current < n {
        let end = (current + batch_size).min(n);

        let preds = model.forward(inputs.clone().slice([current..end, 0..INPUT_DIM]));
        let loss = compute_data_loss(preds, targets.clone().slice([current..end, 0..TARGET_DIM]));
        loss_sum = loss_sum + loss;
        n_batches += 1;

        current = end;
    }

    let total: f32 = (loss_sum / n_batches as f32).into_scalar();
    total as f64
}

fn evaluate_physics_infer(
    model: &StellarMlp<InferBackend>,
    inputs: &Tensor<InferBackend, 2>,
    batch_size: usize,
    norm: &NormParams,
) -> f64 {
    let [n, _] = inputs.dims();
    if n == 0 {
        return 0.0;
    }

    let mut loss_sum = Tensor::<InferBackend, 1>::zeros([1], &inputs.device());
    let mut n_batches = 0usize;
    let mut current = 0usize;

    while current < n {
        let end = (current + batch_size).min(n);

        let preds = model.forward(inputs.clone().slice([current..end, 0..INPUT_DIM]));
        let loss = compute_physics_loss(preds, norm);
        loss_sum = loss_sum + loss;
        n_batches += 1;

        current = end;
    }

    let total: f32 = (loss_sum / n_batches as f32).into_scalar();
    total as f64
}

pub fn cosine_annealing(epoch: usize, total_epochs: usize, initial_lr: f64, min_lr: f64) -> f64 {
    let progress = epoch as f64 / total_epochs as f64;
    min_lr + (initial_lr - min_lr) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

pub fn find_parquet() -> Result<std::path::PathBuf> {
    let candidates = [
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
        "No parquet dataset found in ai_data/. Run 'lnaicli fetch && lnaicli clean' first."
    );
}
