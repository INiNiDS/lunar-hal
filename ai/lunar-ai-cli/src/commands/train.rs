use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::commands::train_spec::{TrainOptions, training_spec_from_opts};
use crate::util::{copy_file_if_present, find_lnai_binary, resolve_data_path};

pub fn run_train(opts: &TrainOptions<'_>) -> Result<()> {
    let data_path = resolve_data_path(opts.data)?;
    let worker = find_lnai_binary(opts.lnai_bin, opts.model)?;

    let output_path = Path::new(opts.output_dir);
    std::fs::create_dir_all(output_path)
        .with_context(|| format!("failed to create output dir {}", output_path.display()))?;
    let out_model = output_path.join(opts.model_file);
    let out_norm = output_path.join(opts.norm_file);

    let mut resume_dir: Option<PathBuf> = None;
    if let Some(resume_path) = opts.resume {
        let resume_pb = PathBuf::from(resume_path);
        let resume_dir_path = if resume_pb.is_dir() {
            resume_pb.clone()
        } else {
            resume_pb
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        };

        let src_model = resume_pb
            .join(opts.model_file)
            .components()
            .collect::<PathBuf>();
        let src_norm = resume_pb.join(opts.norm_file);

        let resolved_model = if src_model.exists() {
            src_model
        } else {
            resume_dir_path.join(opts.model_file)
        };
        let resolved_norm = if src_norm.exists() {
            src_norm
        } else {
            resume_dir_path.join(opts.norm_file)
        };

        if !resolved_model.exists() {
            anyhow::bail!(
                "Resume requested but model file not found: {}",
                resolved_model.display()
            );
        }
        if !resolved_norm.exists() {
            anyhow::bail!(
                "Resume requested but norm file not found: {}",
                resolved_norm.display()
            );
        }

        println!("Staging resume files into output dir:");
        println!("  {} -> {}", resolved_model.display(), out_model.display());
        copy_file_if_present(&resolved_model, &out_model)?;
        println!("  {} -> {}", resolved_norm.display(), out_norm.display());
        copy_file_if_present(&resolved_norm, &out_norm)?;

        if let Some(extra_norm) = opts.norm {
            let extra_norm_path = Path::new(extra_norm);
            if !extra_norm_path.exists() {
                anyhow::bail!("--norm file not found: {}", extra_norm_path.display());
            }
            println!(
                "  {} -> {} (overrides any copied norm)",
                extra_norm_path.display(),
                out_norm.display()
            );
            copy_file_if_present(extra_norm_path, &out_norm)?;
        }

        resume_dir = Some(output_path.to_path_buf());
    } else if let Some(extra_norm) = opts.norm {
        let extra_norm_path = Path::new(extra_norm);
        if !extra_norm_path.exists() {
            anyhow::bail!("--norm file not found: {}", extra_norm_path.display());
        }
        println!(
            "Staging norm into output dir: {} -> {}",
            extra_norm_path.display(),
            out_norm.display()
        );
        copy_file_if_present(extra_norm_path, &out_norm)?;
    }

    let spec = training_spec_from_opts(opts, &data_path)?;

    println!();
    println!("=== lnaicli train ===");
    println!("Model:       {}", spec.model.slug());
    println!("Data:        {}", data_path.display());
    if let Some(rd) = &resume_dir {
        println!("Resume from: {}", rd.display());
    } else {
        println!("Resume from: <none, training from scratch>");
    }
    println!("Output dir:  {}", output_path.display());
    println!("Worker:      {}", worker.display());
    println!(
        "Hyperparams: epochs={}, batch={}, lr={:.2e}, phys_w={}, val_frac={}, gpu={}",
        opts.epochs, opts.batch_size, opts.lr, opts.physics_weight, opts.val_frac, opts.gpu_index
    );
    println!();

    let mut worker_spec = spec.clone();
    if resume_dir.is_some() {
        worker_spec.resume_from = Some(output_path.display().to_string());
    }
    let mut cmd = Command::new(&worker);
    cmd.env(
        "LUNAR_AI_DATASET_MANIFEST_HASH",
        &worker_spec.dataset_manifest_hash,
    );
    for arg in worker_spec.worker_argv() {
        if arg.is_empty() {
            continue;
        }
        cmd.arg(arg);
    }

    println!("Running: {:?}", cmd);
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn worker at {}", worker.display()))?;
    if !status.success() {
        anyhow::bail!("training worker failed with exit code {:?}", status.code());
    }

    println!();
    println!("Done. Model artifacts:");
    println!("  {}", out_model.display());
    println!("  {}", out_norm.display());
    Ok(())
}
