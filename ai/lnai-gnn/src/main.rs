mod dataset;
mod loss;

use nah::high_complexity;
use anyhow::Result;
use burn::backend::cuda::CudaDevice;
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer, AdamWConfig};
use burn::tensor::ElementConversion;
use burn_store::{BurnpackStore, ModuleSnapshot};
use clap::Parser;
use dataset::{GnnDataset, GnnNormParams, PrefetchBatchedBatcher, DEFAULT_KNN_K, DEFAULT_MAX_GROUP};
use loss::{compute_gnn_loss, compute_gnn_physics_loss};
use lnai_models::{StellarGnn, StellarGnnConfig, GNN_INPUT_DIM, GNN_OUTPUT_DIM};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

type TrainBackend = Autodiff<burn::backend::Cuda<f32, i32>>;
type InferBackend = burn::backend::Cuda<f32, i32>;

#[derive(Parser)]
#[command(name = "lnai-gnn", about = "Stellar GNN trainer for velocity prediction")]
struct Args {
    #[arg(long, default_value = "")]
    data: String,
    #[arg(long)]
    holdout: Option<String>,
    #[arg(long, default_value_t = 200)]
    epochs: usize,
    #[arg(long, default_value_t = 4096)]
    max_nodes: usize,
    #[arg(long, default_value_t = 8)]
    grad_accum: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 0.05)]
    physics_weight: f64,
    #[arg(long, default_value_t = 0.1)]
    val_frac: f32,
    #[arg(long, default_value_t = 0)]
    gpu_index: usize,
    #[arg(long)]
    resume_from: Option<String>,
    #[arg(long, default_value = ".")]
    output_dir: String,
    #[arg(long, default_value = "stellar_gnn_model.bpk")]
    model_file: String,
    #[arg(long, default_value = "stellar_gnn_norm.json")]
    norm_file: String,
    #[arg(long, default_value_t = 1.0)]
    clip_grad_norm: f64,
    #[arg(long, default_value_t = 20)]
    patience: usize,
    #[arg(long, default_value_t = 256)]
    hidden_dim: usize,
    #[arg(long, default_value_t = DEFAULT_KNN_K)]
    knn_k: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_GROUP)]
    max_group_size: usize,
    #[arg(long, default_value_t = 50.0)]
    radius_pc: f32,
}

#[high_complexity]
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
        StellarGnn<TrainBackend>,
        GnnNormParams,
        GnnDataset,
    ) = if let Some(resume_dir) = &args.resume_from {
        let resume_path = Path::new(resume_dir);
        let model_path = resume_path.join(&args.model_file);
        let norm_path = resume_path.join(&args.norm_file);

        if !model_path.exists() {
            anyhow::bail!("Resume requested but model file not found: {}", model_path.display());
        }
        if !norm_path.exists() {
            anyhow::bail!("Resume requested but norm file not found: {}", norm_path.display());
        }

        let norm_json = std::fs::read_to_string(&norm_path)?;
        let loaded_norm: GnnNormParams = serde_json::from_str(&norm_json)?;
        println!("Resuming from model: {}", model_path.display());

        let mut store = BurnpackStore::from_file(model_path.to_str().unwrap());
        let mut loaded_model = StellarGnnConfig::new(GNN_INPUT_DIM, args.hidden_dim, GNN_OUTPUT_DIM)
            .init::<TrainBackend>(&device);
        loaded_model
            .load_from(&mut store)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        let dataset = GnnDataset::load_with_norm(
            data_path.as_path(),
            loaded_norm.clone(),
            args.knn_k,
            args.max_group_size,
            args.radius_pc,
        )?;

        (loaded_model, loaded_norm, dataset)
    } else {
        let dataset = GnnDataset::load(
            data_path.as_path(),
            args.knn_k,
            args.max_group_size,
            args.radius_pc,
        )?;
        let norm = dataset.norm.clone();
        let fresh = StellarGnnConfig::new(GNN_INPUT_DIM, args.hidden_dim, GNN_OUTPUT_DIM)
            .init::<TrainBackend>(&device);
        (fresh, norm, dataset)
    };

    let (mut train_ds, val_ds) = dataset.split(args.val_frac);

    let n_params: usize = model.num_params();

    let max_nodes = args.max_nodes;
    let max_adj_bytes = max_nodes as u64 * max_nodes as u64 * 4;
    let max_adj_mb = max_adj_bytes / (1024 * 1024);

    println!();
    println!("=== GNN Model Architecture ===");
    println!("Input dim:             {} (node features from Model 1)", GNN_INPUT_DIM);
    println!("Hidden dim:            {}", args.hidden_dim);
    println!("Output dim:            {} (Vx, Vy, Vz)", GNN_OUTPUT_DIM);
    println!("Total parameters:      {}", n_params);
    println!("k-NN neighbors:        {}", args.knn_k);
    println!("Max group size:        {}", args.max_group_size);
    println!("Group radius:          {} pc", args.radius_pc);
    println!("===============================");
    println!();

    let mut optim = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(args.clip_grad_norm as f32)))
        .init();

    println!();
    println!("=== Training Config ===");
    println!("Max nodes per batch:   {} (~{} MB adj matrix)", max_nodes, max_adj_mb);
    println!("Grad accumulation:     {}", args.grad_accum);
    println!("Grad clip norm:        {}", args.clip_grad_norm);
    println!("Early stop patience:   {}", args.patience);
    println!("Physics weight:        {}", args.physics_weight);
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
            save_checkpoint(&model, &norm, &out_model_path, &out_norm_path)?;
            println!("Checkpoint saved to: {}", out_model_path.display());
            break;
        }

        let lr = cosine_annealing(epoch, args.epochs, initial_lr, 1e-6);

        train_ds.shuffle();
        let mut prefetcher = PrefetchBatchedBatcher::new(&train_ds, max_nodes);
        let mut epoch_train_loss = 0.0f64;
        let mut n_batches = 0usize;

        let mut accumulator: GradientsAccumulator<StellarGnn<TrainBackend>> =
            GradientsAccumulator::new();
        let mut accum_count = 0usize;

        while let Some((nodes, adj, targets)) =
            prefetcher.next_batch::<TrainBackend>(&device)
        {
            let predictions = model.forward(nodes, adj);
            let loss = compute_gnn_physics_loss(
                predictions,
                targets,
                args.physics_weight,
            );

            let loss_scalar = loss.clone().into_scalar().elem::<f32>();
            let scaled_loss = loss.div_scalar(args.grad_accum as f32);
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
        let val_loss = evaluate_gnn(&infer_model, &val_ds, &device, max_nodes);
        let phys_loss = evaluate_physics(&infer_model, &val_ds, &device, args.physics_weight, max_nodes);
        drop(infer_model);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            epochs_without_improvement = 0;
            save_checkpoint(&model, &norm, &out_model_path, &out_norm_path)?;
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

    Ok(())
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

    while let Some((nodes, adj, targets)) =
        prefetcher.next_batch::<InferBackend>(device)
    {
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

    while let Some((nodes, adj, targets)) =
        prefetcher.next_batch::<InferBackend>(device)
    {
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
    anyhow::bail!("No parquet dataset found in ai_data/. Run 'lnaicli fetch-gnn && lnaicli clean-gnn' first.");
}