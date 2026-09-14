//! Stage 5 (task 5): SIREN trainer — moved verbatim from `ai/lnai-siren/src/main.rs`.
//!
//! Deltas vs the pre-Stage-5 binary: seeded star selection/split/shuffle,
//! typed NDJSON epoch events, `artifact.json` bundle, cooperative
//! cancellation, plus read-only `run_evaluate` and `run_benchmark` modes.

use super::dataset::{PrefetchBatcher, SirenDataset, SirenNorm, TARGET_DIM};
use super::loss::{compute_data_loss, compute_siren_loss};
use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::cuda::CudaDevice;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};
use lnai_models::{SIREN_INPUT_DIM, StellarSiren, StellarSirenConfig};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::events::{EpochMetric, JobEvent, format_epoch_line};
use crate::runner::{CancelFlag, RunOutcome, append_event_line, effective_train_seed};
use crate::spec::TrainingSpec;

type TrainBackend = Autodiff<burn::backend::Cuda<f32, i32>>;
type InferBackend = burn::backend::Cuda<f32, i32>;

/// Shared stdout + NDJSON epoch sink (SIREN has no phys column).
struct EventSink {
    output_dir: std::path::PathBuf,
    total_epochs: u32,
}

impl EventSink {
    fn emit_epoch(&self, epoch: u32, train_loss: f64, val_loss: f64, lr: f64) {
        let _ = append_event_line(
            &self.output_dir,
            &JobEvent::Metric(EpochMetric {
                epoch,
                train_loss,
                val_loss,
                phys_loss: None,
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
        .map_err(|errs| anyhow::anyhow!("invalid SIREN spec: {}", errs.join("; ")))?;
    let siren_cfg = match &spec.config {
        crate::spec::ModelConfig::Siren(cfg) => cfg.clone(),
        other => anyhow::bail!("SIREN trainer requires siren config, got {other:?}"),
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

    let (mut model, norm, mut train_ds, val_ds): (
        StellarSiren<TrainBackend>,
        SirenNorm,
        SirenDataset,
        SirenDataset,
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
        let loaded_norm: SirenNorm = serde_json::from_str(&norm_json)?;
        println!("Resuming from model: {}", model_path.display());

        let (train_ds, val_ds) = SirenDataset::generate(
            data_path.as_path(),
            siren_cfg.texture_size as usize,
            siren_cfg.max_stars as usize,
            spec.val_frac,
            siren_cfg.seed,
            spec.max_rows,
        )?;

        let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
        let mut loaded_model = StellarSirenConfig::new().init::<TrainBackend>(&device);
        loaded_model
            .load_from(&mut store)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        println!("=== Fine-tuning mode (using loaded normalization) ===");
        (loaded_model, loaded_norm, train_ds, val_ds)
    } else {
        let (train_ds, val_ds) = SirenDataset::generate(
            data_path.as_path(),
            siren_cfg.texture_size as usize,
            siren_cfg.max_stars as usize,
            spec.val_frac,
            siren_cfg.seed,
            spec.max_rows,
        )?;
        let norm = train_ds.norm.clone();
        let fresh = StellarSirenConfig::new().init::<TrainBackend>(&device);
        (fresh, norm, train_ds, val_ds)
    };

    let n_params: usize = model.num_params();

    println!();
    println!("=== SIREN Model Architecture ===");
    println!("Input dim:        {}", SIREN_INPUT_DIM);
    println!("Output dim:       {}", TARGET_DIM);
    println!("Hidden dim:       64");
    println!("Layers:           4 (first + 2 hidden + output)");
    println!("Omega_0:          30.0");
    println!(
        "Texture grid:     {}x{}",
        siren_cfg.texture_size as usize, siren_cfg.texture_size as usize
    );
    println!("Max stars:        {}", siren_cfg.max_stars as usize);
    println!("Total parameters: {}", n_params);
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
    println!("Effective batch:     {}", effective_batch);
    println!("Grad clip norm:      {}", spec.clip_grad_norm);
    println!("Early stop patience: {}", spec.patience);
    println!("=========================");
    println!();

    println!("Seed:                {seed}");
    println!(
        "{:>5} | {:>12} | {:>12} | {:>12}",
        "epoch", "train_loss", "val_loss", "lr"
    );
    println!("{}", "-".repeat(50));

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
            crate::artifacts::atomic_write_through(&out_model_path, "bpk.tmp", |tmp| {
                let mut store = BurnpackStore::from_file(
                    tmp.to_str()
                        .ok_or_else(|| "non-utf8 checkpoint path".to_string())?,
                )
                .overwrite(true);
                model
                    .save_into(&mut store)
                    .map_err(|e| format!("failed to save checkpoint: {e}"))?;
                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let norm_json = serde_json::to_string_pretty(&norm)?;
            crate::artifacts::atomic_write_through(&out_norm_path, "tmp", |tmp| {
                std::fs::write(tmp, &norm_json)
                    .map_err(|e| format!("failed to write norm: {e}"))?;
                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("Checkpoint saved to: {}", out_model_path.display());
            let _ = append_event_line(output_dir, &JobEvent::Cancelled);
            return Ok(RunOutcome::Cancelled);
        }

        let lr = cosine_annealing(epoch, epochs, initial_lr, 1e-6);

        train_ds.shuffle_with_seed(seed.wrapping_add(epoch as u64));
        let mut prefetcher = PrefetchBatcher::new(&train_ds, spec.batch_size as usize);
        let mut epoch_train_loss = 0.0f64;
        let mut n_batches = 0usize;

        let mut accumulator: GradientsAccumulator<StellarSiren<TrainBackend>> =
            GradientsAccumulator::new();
        let mut accum_count = 0usize;

        while let Some((batch_inputs, batch_targets)) =
            prefetcher.next_batch::<TrainBackend>(&device)
        {
            let predictions = model.forward(batch_inputs);
            let loss = compute_siren_loss(predictions, batch_targets);

            let loss_scalar = loss.clone().into_scalar().elem::<f32>();
            let scaled_loss = if spec.grad_accum > 1 {
                loss.div_scalar(spec.grad_accum as f32)
            } else {
                loss
            };
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
        let val_loss = evaluate_infer(
            &infer_model,
            &val_ds.inputs_cpu,
            &val_ds.targets_cpu,
            val_ds.n_samples,
            spec.batch_size as usize,
            &device,
        );
        drop(infer_model);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            epochs_without_improvement = 0;
            crate::artifacts::atomic_write_through(&out_model_path, "bpk.tmp", |tmp| {
                let mut store = BurnpackStore::from_file(
                    tmp.to_str()
                        .ok_or_else(|| "non-utf8 checkpoint path".to_string())?,
                )
                .overwrite(true);
                model
                    .save_into(&mut store)
                    .map_err(|e| format!("failed to save best model: {e}"))?;
                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let norm_json = serde_json::to_string_pretty(&norm)?;
            crate::artifacts::atomic_write_through(&out_norm_path, "tmp", |tmp| {
                std::fs::write(tmp, &norm_json)
                    .map_err(|e| format!("failed to write norm: {e}"))?;
                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        } else {
            epochs_without_improvement += 1;
        }

        println!(
            "{}",
            format_epoch_line(epoch as u32, epoch_train_loss, val_loss, None, lr)
        );
        sink.emit_epoch(epoch as u32, epoch_train_loss, val_loss, lr);

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

    if let Some(holdout_path) = &spec.holdout {
        println!();
        println!("=== Holdout Evaluation ===");
        let holdout_path = Path::new(holdout_path);
        if holdout_path.exists() {
            let (holdout_train, holdout_val) = SirenDataset::generate(
                holdout_path,
                siren_cfg.texture_size as usize,
                siren_cfg.max_stars as usize,
                0.0,
                siren_cfg.seed,
                spec.max_rows,
            )?;
            drop(holdout_train);
            let infer_model = model.valid();
            let holdout_loss = evaluate_infer(
                &infer_model,
                &holdout_val.inputs_cpu,
                &holdout_val.targets_cpu,
                holdout_val.n_samples,
                spec.batch_size as usize,
                &device,
            );
            drop(infer_model);
            println!("Holdout data loss: {:.6}", holdout_loss);

            if holdout_loss <= best_val_loss * 1.5 {
                println!("Holdout loss is close to validation loss - model generalizes well!");
            } else {
                println!("WARNING: Holdout loss significantly higher than validation loss.");
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
    norm: &SirenNorm,
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
        ModelKind::Siren,
        architecture_version(&ModelKind::Siren).to_string(),
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

/// Read-only evaluation: loads the artifact, reports the validation loss,
/// trains nothing and rewrites no checkpoint.
pub fn run_evaluate(spec: &TrainingSpec) -> Result<RunOutcome> {
    spec.validate()
        .map_err(|errs| anyhow::anyhow!("invalid SIREN spec: {}", errs.join("; ")))?;
    let siren_cfg = match &spec.config {
        crate::spec::ModelConfig::Siren(cfg) => cfg.clone(),
        other => anyhow::bail!("SIREN trainer requires siren config, got {other:?}"),
    };
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
    let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
    let mut loaded = StellarSirenConfig::new().init::<TrainBackend>(&device);
    loaded
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;
    let infer = loaded.valid();
    let (train_ds, val_ds) = SirenDataset::generate(
        data_path.as_path(),
        siren_cfg.texture_size as usize,
        siren_cfg.max_stars as usize,
        spec.val_frac,
        siren_cfg.seed,
        None,
    )?;
    drop(train_ds);
    let val_loss = evaluate_infer(
        &infer,
        &val_ds.inputs_cpu,
        &val_ds.targets_cpu,
        val_ds.n_samples,
        spec.batch_size as usize,
        &device,
    );
    drop(infer);
    println!("=== Read-only evaluation (no weight updates) ===");
    println!("Validation data loss: {val_loss:.6}");
    println!("Evaluation complete; checkpoints untouched.");
    let _ = norm_path;
    Ok(RunOutcome::Completed)
}

/// Benchmark harness: loads the artifact, times forward passes, trains nothing.
pub fn run_benchmark(spec: &TrainingSpec, iters: u32, warmup: u32) -> Result<RunOutcome> {
    use std::time::Instant;
    let device = CudaDevice::new(spec.gpu_index as usize);
    let output_dir = Path::new(&spec.output_dir);
    let model_path = output_dir.join(&spec.model_file);
    if !model_path.exists() {
        anyhow::bail!("benchmark: model file not found: {}", model_path.display());
    }
    let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
    let mut loaded = StellarSirenConfig::new().init::<TrainBackend>(&device);
    loaded
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;
    let infer = loaded.valid();
    let batch = spec.batch_size.max(1) as usize;
    let zeros = Tensor::<InferBackend, 2>::zeros([batch, SIREN_INPUT_DIM], &device);
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
        "model": "siren",
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

fn evaluate_infer(
    model: &StellarSiren<InferBackend>,
    inputs: &[f32],
    targets: &[f32],
    n_samples: usize,
    batch_size: usize,
    device: &CudaDevice,
) -> f64 {
    let mut current = 0;
    let mut total_loss = 0.0f64;
    let mut n = 0usize;

    while current < n_samples {
        let end = (current + batch_size).min(n_samples);
        let rows = end - current;

        let inp_slice = &inputs[current * SIREN_INPUT_DIM..end * SIREN_INPUT_DIM];
        let tgt_slice = &targets[current * TARGET_DIM..end * TARGET_DIM];

        let batch_inputs = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(inp_slice.to_vec(), [rows, SIREN_INPUT_DIM]),
            device,
        );
        let batch_targets = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(tgt_slice.to_vec(), [rows, TARGET_DIM]),
            device,
        );

        let preds = model.forward(batch_inputs);
        let loss = compute_data_loss(preds, batch_targets);
        let value: f32 = loss.into_scalar().elem();
        total_loss += value as f64;
        n += 1;

        current = end;
    }

    total_loss / n.max(1) as f64
}

fn cosine_annealing(epoch: usize, total_epochs: usize, initial_lr: f64, min_lr: f64) -> f64 {
    let progress = epoch as f64 / total_epochs as f64;
    min_lr + (initial_lr - min_lr) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

fn find_parquet() -> Result<std::path::PathBuf> {
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
