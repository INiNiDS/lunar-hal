mod dataset;
mod loss;

use anyhow::Result;
use burn::backend::cuda::CudaDevice;
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer, AdamWConfig};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};
use clap::Parser;
use dataset::{PrefetchBatcher, StellarDataset, INPUT_DIM, TARGET_DIM};
use loss::{compute_data_loss, compute_physics_loss, compute_pinn_loss};
use lnai_models::{StellarMlp, StellarMlpConfig, MLP_INPUT_DIM};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

type TrainBackend = Autodiff<burn::backend::Cuda<f32, i32>>;
type InferBackend = burn::backend::Cuda<f32, i32>;

#[derive(Parser)]
#[command(name = "lnai", about = "Stellar MLP trainer with Fourier features and PINN loss")]
struct Args {
    #[arg(long, default_value = "")]
    data: String,
    #[arg(long)]
    holdout: Option<String>,
    #[arg(long, default_value_t = 200)]
    epochs: usize,
    #[arg(long, default_value_t = 2048)]
    batch_size: usize,
    #[arg(long, default_value_t = 2)]
    grad_accum: usize,
    #[arg(long, default_value_t = 5e-4)]
    lr: f64,
    #[arg(long, default_value_t = 0.1)]
    physics_weight: f64,
    #[arg(long, default_value_t = 0.1)]
    val_frac: f32,
    #[arg(long, default_value_t = 0)]
    gpu_index: usize,
    #[arg(long)]
    resume_from: Option<String>,
    #[arg(long, default_value = ".")]
    output_dir: String,
    #[arg(long, default_value = "stellar_model.bpk")]
    model_file: String,
    #[arg(long, default_value = "stellar_norm.json")]
    norm_file: String,
    #[arg(long, default_value_t = 1.0)]
    clip_grad_norm: f64,
    #[arg(long, default_value_t = 20)]
    patience: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = CudaDevice::new(args.gpu_index);
    println!("Compute device: Cuda({})", args.gpu_index);

    let data_path = if args.data.is_empty() {
        find_parquet()?
    } else {
        std::path::PathBuf::from(&args.data)
    };

    let output_dir = Path::new(&args.output_dir);
    std::fs::create_dir_all(output_dir)
        .map_err(|e| anyhow::anyhow!("failed to create output dir {}: {}", output_dir.display(), e))?;
    let out_model_path = output_dir.join(&args.model_file);
    let out_norm_path = output_dir.join(&args.norm_file);

    let (mut model, norm, dataset): (
        StellarMlp<TrainBackend>,
        dataset::NormParams,
        StellarDataset<TrainBackend>,
    ) = if let Some(resume_dir) = &args.resume_from {
        let resume_path = Path::new(resume_dir);
        let model_path = resume_path.join(&args.model_file);
        let norm_path = resume_path.join(&args.norm_file);

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
        let loaded_norm: dataset::NormParams = serde_json::from_str(&norm_json)?;
        println!("Resuming from model: {}", model_path.display());
        println!("Using norm from:     {}", norm_path.display());

        let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
        let mut loaded_model = StellarMlpConfig::new().init::<TrainBackend>(&device);
        loaded_model
            .load_from(&mut store)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        let dataset: StellarDataset<TrainBackend> = StellarDataset::load_with_norm(
            data_path.as_path(),
            loaded_norm.clone(),
            &device,
        )?;

        println!("=== Fine-tuning mode (using loaded normalization) ===");
        (loaded_model, loaded_norm, dataset)
    } else {
        let dataset: StellarDataset<TrainBackend> =
            StellarDataset::load(data_path.as_path(), &device)?;
        let norm = dataset.norm.clone();
        let fresh = StellarMlpConfig::new().init::<TrainBackend>(&device);
        (fresh, norm, dataset)
    };

    let (mut train_ds, val_ds) = dataset.split(args.val_frac);

    let n_params: usize = model.num_params();

    println!();
    println!("=== Model Architecture ===");
    println!("Fourier levels:       {}", StellarMlpConfig::fourier_levels());
    println!("Fourier dim:          {}", lnai_models::FOURIER_DIM);
    println!("Conditional inputs:   2 (bp_rp, M_G)");
    println!("MLP input dim:        {}", MLP_INPUT_DIM);
    println!("Input features:       5 (x, y, z, bp_rp, M_G)");
    println!("Output features:      4 (log10_teff, log10_rad, log10_mass, log10_lum)");
    println!("Total parameters:     {}", n_params);
    println!("=========================");
    println!();

    let mut optim = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(args.clip_grad_norm as f32)))
        .init();

    let effective_batch = args.batch_size * args.grad_accum;
    println!();
    println!("=== Training Config ===");
    println!("Micro-batch size:    {}", args.batch_size);
    println!("Grad accumulation:   {}", args.grad_accum);
    println!("Effective batch:     {}", effective_batch);
    println!("Grad clip norm:      {}", args.clip_grad_norm);
    println!("Early stop patience: {}", args.patience);
    println!("=========================");
    println!();

    println!(
        "{:>5} | {:>12} | {:>12} | {:>12} | {:>10}",
        "epoch", "train_loss", "val_loss", "phys_loss", "lr"
    );
    println!("{}", "-".repeat(65));

    let mut best_val_loss = f64::MAX;
    let mut epochs_without_improvement = 0usize;
    let initial_lr = args.lr;

    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = interrupted.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nCtrl+C received, finishing current epoch and saving model...");
        interrupted_clone.store(true, Ordering::SeqCst);
    }).expect("failed to set Ctrl+C handler");

    for epoch in 1..=args.epochs {
        if interrupted.load(Ordering::SeqCst) {
            println!("\nInterrupted at epoch {epoch}. Saving checkpoint...");
            let mut store =
                BurnpackStore::from_file(out_model_path.to_str().unwrap()).overwrite(true);
            model
                .save_into(&mut store)
                .expect("failed to save checkpoint");
            let norm_json = serde_json::to_string_pretty(&norm)?;
            std::fs::write(&out_norm_path, norm_json)?;
            println!("Checkpoint saved to: {}", out_model_path.display());
            break;
        }

        let lr = cosine_annealing(epoch, args.epochs, initial_lr, 1e-6);

        train_ds.shuffle();
        let mut prefetcher = PrefetchBatcher::new(&train_ds, args.batch_size);
        let mut epoch_train_loss = 0.0f64;
        let mut n_batches = 0usize;

        let mut accumulator: GradientsAccumulator<StellarMlp<TrainBackend>> =
            GradientsAccumulator::new();
        let mut accum_count = 0usize;

        while let Some((batch_inputs, batch_targets)) = prefetcher.next_batch::<TrainBackend>(&device) {
            let predictions = model.forward(batch_inputs);
            let loss = compute_pinn_loss(
                predictions,
                batch_targets,
                args.physics_weight,
                norm.log_teff_mean,
                norm.log_teff_std,
                norm.log_rad_mean,
                norm.log_rad_std,
                norm.log_lum_mean,
                norm.log_lum_std,
            );

            let loss_scalar = loss.clone().into_scalar().elem::<f32>();
            let scaled_loss = if args.grad_accum > 1 {
                loss.div_scalar(args.grad_accum as f32)
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

            if accum_count >= args.grad_accum {
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
            args.batch_size,
            &device,
        );
        let phys_loss = evaluate_physics_infer(
            &infer_model,
            &val_ds.inputs_cpu,
            val_ds.n_samples,
            args.batch_size,
            &device,
            &norm,
        );
        drop(infer_model);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            epochs_without_improvement = 0;
            let mut store = BurnpackStore::from_file(out_model_path.to_str().unwrap())
                .overwrite(true);
            model
                .save_into(&mut store)
                .expect("failed to save best model");
            let norm_json = serde_json::to_string_pretty(&norm)?;
            std::fs::write(&out_norm_path, norm_json)?;
        } else {
            epochs_without_improvement += 1;
        }

        println!(
            "{epoch:5} | {train_loss:12.6} | {val_loss:12.6} | {phys_loss:12.6} | {lr:10.6e}",
            epoch = epoch,
            train_loss = epoch_train_loss,
            val_loss = val_loss,
            phys_loss = phys_loss,
            lr = lr,
        );

        if epochs_without_improvement >= args.patience {
            println!("\nEarly stopping: no improvement for {} epochs.", args.patience);
            break;
        }
    }

    println!();
    println!(
        "Training complete. Best validation loss: {:.6}",
        best_val_loss
    );
    println!("Best model saved to: {}", out_model_path.display());
    println!("Normalization params saved to: {}", out_norm_path.display());

    if let Some(holdout_path) = &args.holdout {
        println!();
        println!("=== Holdout Evaluation ===");
        let holdout_path = Path::new(holdout_path);
        if holdout_path.exists() {
            let holdout_ds: StellarDataset<TrainBackend> =
                StellarDataset::load(holdout_path, &device)?;
            let (_, holdout_val) = holdout_ds.split(0.0);
            let infer_model = model.valid();
            let holdout_loss = evaluate_infer(
                &infer_model,
                &holdout_val.inputs_cpu,
                &holdout_val.targets_cpu,
                holdout_val.n_samples,
                args.batch_size,
                &device,
            );
            let holdout_phys = evaluate_physics_infer(
                &infer_model,
                &holdout_val.inputs_cpu,
                holdout_val.n_samples,
                args.batch_size,
                &device,
                &norm,
            );
            drop(infer_model);
            println!("Holdout data loss:   {:.6}", holdout_loss);
            println!("Holdout physics loss: {:.6}", holdout_phys);

            if holdout_loss <= best_val_loss * 1.5 {
                println!("Holdout loss is close to validation loss - model generalizes well!");
            } else {
                println!("WARNING: Holdout loss is significantly higher than validation loss.");
                println!("         The model may be overfitting. Consider regularization or more data.");
            }
        } else {
            println!("Holdout file not found: {}", holdout_path.display());
        }
    }

    Ok(())
}

fn evaluate_infer(
    model: &StellarMlp<InferBackend>,
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

        let inp_slice = &inputs[current * INPUT_DIM..end * INPUT_DIM];
        let tgt_slice = &targets[current * TARGET_DIM..end * TARGET_DIM];

        let batch_inputs = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(inp_slice.to_vec(), [rows, INPUT_DIM]),
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

fn evaluate_physics_infer(
    model: &StellarMlp<InferBackend>,
    inputs: &[f32],
    n_samples: usize,
    batch_size: usize,
    device: &CudaDevice,
    norm: &dataset::NormParams,
) -> f64 {
    let mut current = 0;
    let mut total_loss = 0.0f64;
    let mut n = 0usize;

    while current < n_samples {
        let end = (current + batch_size).min(n_samples);
        let rows = end - current;

        let inp_slice = &inputs[current * INPUT_DIM..end * INPUT_DIM];

        let batch_inputs = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(inp_slice.to_vec(), [rows, INPUT_DIM]),
            device,
        );

        let preds = model.forward(batch_inputs);
        let loss = compute_physics_loss(
            preds,
            norm.log_teff_mean,
            norm.log_teff_std,
            norm.log_rad_mean,
            norm.log_rad_std,
            norm.log_lum_mean,
            norm.log_lum_std,
        );
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
    anyhow::bail!("No parquet dataset found in ai_data/. Run 'lnaicli fetch && lnaicli clean' first.");
}