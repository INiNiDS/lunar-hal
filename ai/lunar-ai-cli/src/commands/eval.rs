use anyhow::{Context, Result, anyhow};
use std::path::Path;
use std::process::Command;

use crate::cli::args_train::CliModel;
use crate::util::{find_lnai_binary, resolve_data_path, sha256_file};

#[allow(clippy::too_many_arguments)]
pub fn run_evaluate_cmd(
    model: CliModel,
    output_dir: &str,
    data: Option<&str>,
    holdout: Option<&str>,
    batch_size: usize,
    seed: Option<u64>,
    model_file: &str,
    norm_file: &str,
) -> Result<()> {
    use lnai_training::spec::{EvaluationSpec, ModelKind};
    let kind = match model {
        CliModel::Pinn => ModelKind::Pinn,
        CliModel::Gnn => ModelKind::GnnKinematics,
        CliModel::Siren => ModelKind::Siren,
    };
    let output_path = Path::new(output_dir);
    if !output_path.join(model_file).exists() {
        anyhow::bail!(
            "evaluate: model file not found: {}",
            output_path.join(model_file).display()
        );
    }
    let data_path = resolve_data_path(data)
        .ok()
        .map(|p| p.display().to_string());
    let dataset_manifest_hash = data_path
        .as_deref()
        .map(|path| lnai_training::artifacts::dataset_fingerprint(Path::new(path)))
        .transpose()?
        .unwrap_or_default();
    let spec = EvaluationSpec {
        model: kind,
        artifact_hash: sha256_file(&output_path.join(model_file).display().to_string())
            .unwrap_or_default(),
        dataset_manifest_hash,
        data_path,
        batch_size: batch_size as u32,
        output_dir: output_dir.to_string(),
        seed,
    };
    spec.validate()
        .map_err(|errs| anyhow!("invalid evaluation spec: {}", errs.join("; ")))?;
    let worker = find_lnai_binary(None, model)?;
    let mut cmd = Command::new(&worker);
    cmd.env(
        "LUNAR_AI_DATASET_MANIFEST_HASH",
        &spec.dataset_manifest_hash,
    );
    for arg in spec.worker_argv(holdout) {
        if arg.is_empty() {
            continue;
        }
        cmd.arg(arg);
    }
    cmd.arg("--model-file").arg(model_file);
    cmd.arg("--norm-file").arg(norm_file);
    println!("Running: {:?}", cmd);
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn worker at {}", worker.display()))?;
    if !status.success() {
        anyhow::bail!(
            "evaluation worker failed with exit code {:?}",
            status.code()
        );
    }
    Ok(())
}

pub fn run_benchmark_cmd(
    model: CliModel,
    output_dir: &str,
    iters: u32,
    warmup: u32,
    batch_size: usize,
    seed: Option<u64>,
) -> Result<()> {
    use lnai_training::spec::{BenchmarkSpec, ModelKind};
    let kind = match model {
        CliModel::Pinn => ModelKind::Pinn,
        CliModel::Gnn => ModelKind::GnnKinematics,
        CliModel::Siren => ModelKind::Siren,
    };
    let spec = BenchmarkSpec {
        model: kind,
        artifact_hash: String::new(),
        iterations: iters,
        warmup_iterations: warmup,
        output_dir: output_dir.to_string(),
        batch_size: batch_size as u32,
        seed,
    };
    spec.validate()
        .map_err(|errs| anyhow!("invalid benchmark spec: {}", errs.join("; ")))?;
    let worker = find_lnai_binary(None, model)?;
    let mut cmd = Command::new(&worker);
    for arg in spec.worker_argv() {
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
        anyhow::bail!("benchmark worker failed with exit code {:?}", status.code());
    }
    Ok(())
}
