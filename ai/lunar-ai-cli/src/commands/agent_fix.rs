use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;

pub fn run_agent_fix(
    dir: &str,
    model: &str,
    message: Option<String>,
    auto_approve: bool,
) -> Result<()> {
    let output_dir = PathBuf::from(dir);
    if !output_dir.join("events.ndjson").exists() {
        anyhow::bail!(
            "no events.ndjson in {}: point --dir at a training output dir",
            output_dir.display()
        );
    }
    let extra = message
        .map(|m| format!("\nOperator note: {m}\n"))
        .unwrap_or_default();
    let prompt = format!(
        "You are the fixer for a STOPPED Stellar GNN-kinematics training run.\n\
         Output dir: {out} (read {out}/events.ndjson tail, {out}/artifact.json).\n\
         Repo root is your working directory.\n\
         Diagnose the failure from the logs, then REPAIR the code so a fresh \
         `lnaicli train --model gnn ...` run (from scratch, no --resume) would \
         get past it. Verify with `cargo check` on the touched crates only; \
         do NOT start training yourself.{extra}\n\
         End with a short summary: root cause, files changed, and the exact \
         train command to restart from scratch.",
        out = output_dir.display(),
    );
    println!(
        "Spawning fixer agent (model {model}) over {} ...",
        output_dir.display()
    );
    if auto_approve {
        println!("NOTE: --auto-approve is on: the agent may edit files and run commands.");
    }
    let mut cmd = Command::new("opencode");
    cmd.args(["run", "--model", model, "--dir", "."])
        .arg(&prompt);
    if auto_approve {
        cmd.arg("--auto");
    }
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("spawn opencode: {e} (is it on PATH?)"))?;
    if !status.success() {
        anyhow::bail!("fixer agent exited with {:?}", status.code());
    }
    Ok(())
}
