mod dataset;
mod loss;
mod model;

use anyhow::Result;
use candle_core::{DType, Device, Module};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use dataset::{BatchIterator, StellarDataset};
use loss::{compute_data_loss, compute_physics_loss, compute_pinn_loss};
use model::StellarMLP;
use std::path::Path;

const EPOCHS: usize = 150;
const BATCH_SIZE: usize = 4096;
const LEARNING_RATE: f64 = 5e-4;
const PHYSICS_WEIGHT: f64 = 0.1;
const VAL_FRAC: f32 = 0.1;

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Compute device: {:?}", device);

    let data_path = find_parquet()?;
    let dataset = StellarDataset::load(data_path.as_path())?;
    let norm = dataset.norm.clone();
    let (mut train_ds, val_ds) = dataset.split_to_device(VAL_FRAC, &device)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = StellarMLP::new(vs)?;

    let n_params: usize = varmap
        .all_vars()
        .iter()
        .map(|v| v.shape().elem_count())
        .sum();
    println!("Model parameters: {}", n_params);

    let params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    println!();
    println!(
        "{:>5} | {:>12} | {:>12} | {:>12} | {:>10}",
        "epoch", "train_loss", "val_loss", "phys_loss", "lr"
    );
    println!("{}", "-".repeat(65));

    let mut best_val_loss = f64::MAX;
    let initial_lr = LEARNING_RATE;

    for epoch in 1..=EPOCHS {
        let lr = cosine_annealing(epoch, EPOCHS, initial_lr, 1e-6);
        optimizer.set_learning_rate(lr);

        train_ds.shuffle()?;
        let mut train_iter = BatchIterator::new(&train_ds, BATCH_SIZE);
        let mut epoch_train_loss = 0.0f64;
        let mut n_batches = 0usize;

        while let Some((batch_inputs, batch_targets)) = train_iter.next_batch() {
            let predictions = model.forward(&batch_inputs)?;
            let loss = compute_pinn_loss(
                &predictions,
                &batch_targets,
                PHYSICS_WEIGHT,
                norm.teff_mean,
                norm.teff_std,
                norm.rad_mean,
                norm.rad_std,
                norm.lum_mean,
                norm.lum_std,
            )?;

            optimizer.backward_step(&loss)?;

            epoch_train_loss += loss.to_vec0::<f32>()? as f64;
            n_batches += 1;
        }

        let epoch_train_loss = epoch_train_loss / n_batches.max(1) as f64;
        let val_loss = evaluate(&model, &val_ds)?;
        let phys_loss = evaluate_physics(&model, &val_ds, &norm)?;

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            varmap.save("stellar_model.safetensors")?;
        }

        println!(
            "{epoch:5} | {train_loss:12.6} | {val_loss:12.6} | {phys_loss:12.6} | {lr:10.6e}",
            epoch = epoch,
            train_loss = epoch_train_loss,
            val_loss = val_loss,
            phys_loss = phys_loss,
            lr = lr,
        );
    }

    println!();
    println!(
        "Training complete. Best validation loss: {:.6}",
        best_val_loss
    );
    println!("Best model saved to: stellar_model.safetensors");
    Ok(())
}

fn evaluate(model: &StellarMLP, ds: &dataset::StellarDataset) -> Result<f64> {
    let mut iter = BatchIterator::new(ds, BATCH_SIZE);
    let mut total_loss = 0.0f64;
    let mut n = 0usize;

    while let Some((inputs, targets)) = iter.next_batch() {
        let preds = model.forward(&inputs)?;
        let loss = compute_data_loss(&preds, &targets)?;
        total_loss += loss.to_vec0::<f32>()? as f64;
        n += 1;
    }

    Ok(total_loss / n.max(1) as f64)
}

fn evaluate_physics(
    model: &StellarMLP,
    ds: &dataset::StellarDataset,
    norm: &dataset::NormParams,
) -> Result<f64> {
    let mut iter = BatchIterator::new(ds, BATCH_SIZE);
    let mut total_loss = 0.0f64;
    let mut n = 0usize;

    while let Some((inputs, _targets)) = iter.next_batch() {
        let preds = model.forward(&inputs)?;
        let loss = compute_physics_loss(
            &preds,
            norm.teff_mean,
            norm.teff_std,
            norm.rad_mean,
            norm.rad_std,
            norm.lum_mean,
            norm.lum_std,
        )?;
        total_loss += loss.to_vec0::<f32>()? as f64;
        n += 1;
    }

    Ok(total_loss / n.max(1) as f64)
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
    anyhow::bail!("No parquet dataset found in ai_data/")
}
