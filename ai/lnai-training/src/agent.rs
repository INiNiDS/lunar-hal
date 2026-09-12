//! Epoch-watch AI supervisor hook (GNN-kinematics training).
//!
//! After each epoch the trainer can consult an OpenCode agent (`opencode run`
//! subprocess, see <https://opencode.ai/docs/cli/#run-1>) with the epoch
//! table and log paths. The agent is read-only (see
//! `.opencode/agents/gnn-watch.md`) and answers with a machine-readable
//! verdict block:
//!
//! ```text
//! VERDICT: CONTINUE
//! ```
//!
//! ```text
//! VERDICT: STOP
//! REASON: <what is wrong and which file/line proves it>
//! ```
//!
//! Tool mapping back to the requested interface:
//! * `read_file` — the agent's native read/glob/grep tools (allowed).
//! * `continue` — the agent finishing its turn with `VERDICT: CONTINUE`;
//!   the trainer blocks on the subprocess, so the next epoch cannot start
//!   before the agent is done.
//! * `stop` — `VERDICT: STOP` halts the whole run after a checkpoint save.
//!   Repairs go through a separate fixer invocation
//!   (`lnaicli agent-fix`, full tools), then `lnaicli train` restarts
//!   from scratch via `start` semantics (no `--resume`).
//! * `opencode_agent` — nested agent calls happen supervisor-side
//!   (`lnaicli agent-fix`); the watcher itself has `task` denied so it
//!   can never recurse.
//!
//! Failure policy is fail-open: an agent timeout, a missing binary, or a
//! missing verdict logs loudly and training CONTINUES. Only an explicit
//! `VERDICT: STOP` halts. A crashed supervisor must never kill a healthy
//! 10-hour run.

use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Model used first for epoch-watch calls (OpenRouter, user key).
pub const DEFAULT_AGENT_MODEL: &str = "openrouter/meta/muse-spark-1.3-contributor";
/// Fallback when the primary model call fails (Zen, free).
pub const DEFAULT_AGENT_FALLBACK_MODEL: &str = "opencode/muse-spark-1.3-contributor-free";
/// Agent defined in `.opencode/agents/gnn-watch.md` (repo root).
pub const DEFAULT_AGENT_NAME: &str = "gnn-watch";
/// Seconds to wait for one agent call before giving up (fail-open).
pub const DEFAULT_AGENT_TIMEOUT_SECS: u64 = 300;
/// Epoch event-log lines attached to each call.
pub const DEFAULT_AGENT_LOG_LINES: usize = 30;

/// Supervisor hook configuration. `None` in [`crate::spec::TrainingSpec`]
/// disables the hook entirely (legacy behaviour, zero overhead).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentHookConfig {
    /// Consult the agent every N epochs (1 = after every epoch).
    pub every: u64,
    /// Primary model id (`provider/model`).
    pub model: String,
    /// Fallback model id on primary execution failure.
    pub fallback_model: String,
    /// Per-call timeout in seconds (fail-open on expiry).
    pub timeout_secs: u64,
    /// Tail lines of `events.ndjson` attached to the prompt.
    pub log_lines: usize,
    /// Agent name from `.opencode/agents` (or a built-in).
    pub agent: String,
    /// When true, print the prompt instead of spawning (zero-cost plumbing
    /// check); always continues.
    pub dry_run: bool,
}

impl Default for AgentHookConfig {
    fn default() -> Self {
        Self {
            every: 1,
            model: DEFAULT_AGENT_MODEL.to_string(),
            fallback_model: DEFAULT_AGENT_FALLBACK_MODEL.to_string(),
            timeout_secs: DEFAULT_AGENT_TIMEOUT_SECS,
            log_lines: DEFAULT_AGENT_LOG_LINES,
            agent: DEFAULT_AGENT_NAME.to_string(),
            dry_run: false,
        }
    }
}

/// Machine verdict parsed from the agent's output.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentVerdict {
    Continue,
    Stop { reason: String },
}

/// Parses the LAST `VERDICT:` block. Returns `None` when the agent never
/// issued one (fail-open → caller continues with a warning).
pub fn parse_verdict(output: &str) -> Option<AgentVerdict> {
    let mut verdict: Option<AgentVerdict> = None;
    let mut lines = output.lines().peekable();
    while let Some(line) = lines.next() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("VERDICT:") {
            match rest.trim() {
                "CONTINUE" => verdict = Some(AgentVerdict::Continue),
                "STOP" => {
                    let mut reason = String::new();
                    while let Some(next) = lines.peek() {
                        let nt = next.trim();
                        if let Some(r) = nt.strip_prefix("REASON:") {
                            if !reason.is_empty() {
                                reason.push(' ');
                            }
                            reason.push_str(r.trim());
                            lines.next();
                        } else if nt.is_empty() {
                            lines.next();
                        } else {
                            break;
                        }
                    }
                    if reason.is_empty() {
                        reason = "no reason given".to_string();
                    }
                    verdict = Some(AgentVerdict::Stop { reason });
                }
                _ => {}
            }
        }
    }
    verdict
}

/// One row of the epoch table attached to the prompt.
#[derive(Debug, Clone)]
pub struct EpochRow {
    pub epoch: u32,
    pub train_loss: f64,
    pub val_loss: f64,
    pub phys_loss: f64,
    pub lr: f64,
}

/// Builds the per-epoch prompt: epoch table + file pointers. The agent reads
/// the logs itself; the prompt stays small and stateless (no session reuse).
pub fn epoch_prompt(
    model_slug: &str,
    output_dir: &Path,
    data_path: &str,
    history: &[EpochRow],
    best_val_loss: f64,
    log_tail: &str,
) -> String {
    let mut table = String::from("epoch | train_loss | val_loss | phys_loss\n");
    for r in history {
        table.push_str(&format!(
            "{} | {:.6} | {:.6} | {:.6}\n",
            r.epoch, r.train_loss, r.val_loss, r.phys_loss
        ));
    }
    format!(
        "Training run under watch: model={model_slug} data={data_path} \
         output_dir={output} best_val_loss={best:.6} finished_epochs={n}.\n\
         Epoch table so far (train/val/physics losses):\n{table}\n\
         Recent event-log tail (events.ndjson):\n{log_tail}\n\
         Useful files (read what you need, you are read-only): \
         {output}/events.ndjson, {output}/artifact.json.\n\
         Analyze this epoch in the context of the table, then end with the \
         verdict block exactly as specified in your instructions.",
        output = output_dir.display(),
        best = best_val_loss,
        n = history.len(),
    )
}

/// Last `n` lines of a text file; missing/unreadable file → placeholder note.
pub fn tail_file(path: &Path, n: usize) -> String {
    let mut text = String::new();
    match std::fs::File::open(path).and_then(|mut f| f.read_to_string(&mut text)) {
        Ok(_) => {
            let lines: Vec<&str> = text.lines().collect();
            let skip = lines.len().saturating_sub(n);
            lines[skip..].join("\n")
        }
        Err(e) => format!("<could not read {}: {e}>", path.display()),
    }
}

/// Runs one `opencode run` call with a hard timeout. Returns stdout on
/// success; stderr is folded into the error for diagnostics.
fn run_agent_call(
    opencode_bin: &str,
    agent: &str,
    model: &str,
    workdir: &Path,
    prompt: &str,
    timeout: Duration,
) -> Result<String, String> {
    let mut child = Command::new(opencode_bin)
        .args([
            "run",
            "--agent",
            agent,
            "--model",
            model,
            "--dir",
            &workdir.to_string_lossy(),
        ])
        .arg(prompt)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn {opencode_bin}: {e}"))?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_string(&mut out);
                }
                if !status.success() {
                    let mut err = String::new();
                    if let Some(mut stderr) = child.stderr.take() {
                        let _ = stderr.read_to_string(&mut err);
                    }
                    return Err(format!(
                        "opencode run --model {model} exited {}: {}",
                        status.code().unwrap_or(-1),
                        tail_str(&err, 500)
                    ));
                }
                return Ok(out);
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("opencode run timed out after {}s", timeout.as_secs()));
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(e) => return Err(format!("wait on opencode run: {e}")),
        }
    }
}

fn tail_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...[truncated]{}", &s[s.len() - max..])
    }
}

/// Consults the agent when `epoch` is due (`epoch % every == 0`).
/// Returns the parsed verdict, or `AgentVerdict::Continue` fail-open with a
/// loud warning when anything goes wrong (timeout, spawn failure, fallback
/// failure, missing verdict). Only an explicit STOP halts training.
pub fn maybe_consult_agent(
    cfg: &AgentHookConfig,
    epoch: u64,
    workdir: &Path,
    prompt: &str,
) -> AgentVerdict {
    if cfg.every == 0 || epoch % cfg.every != 0 {
        return AgentVerdict::Continue;
    }
    if cfg.dry_run {
        println!("--- agent watch (epoch {epoch}): DRY RUN, prompt follows ---");
        println!("{prompt}");
        println!("--- end dry-run prompt ---");
        return AgentVerdict::Continue;
    }
    println!("--- agent watch (epoch {epoch}): consulting {} ---", cfg.model);
    let timeout = Duration::from_secs(cfg.timeout_secs.max(10));
    let attempt = |model: &str| run_agent_call("opencode", &cfg.agent, model, workdir, prompt, timeout);
    match attempt(&cfg.model) {
        Ok(out) => match parse_verdict(&out) {
            Some(v) => {
                println!("--- agent verdict: {v:?} ---");
                v
            }
            None => {
                println!(
                    "--- agent watch: no VERDICT block, continuing (fail-open) ---"
                );
                AgentVerdict::Continue
            }
        },
        Err(e) => {
            println!("--- agent watch: primary model failed ({e}); trying fallback {} ---", cfg.fallback_model);
            match attempt(&cfg.fallback_model) {
                Ok(out) => match parse_verdict(&out) {
                    Some(v) => {
                        println!("--- agent verdict (fallback): {v:?} ---");
                        v
                    }
                    None => {
                        println!("--- agent watch: fallback gave no verdict, continuing ---");
                        AgentVerdict::Continue
                    }
                },
                Err(e2) => {
                    println!("--- agent watch: fallback also failed ({e2}); continuing ---");
                    AgentVerdict::Continue
                }
            }
        }
    }
}

/// Absolute workspace root for `--dir` (agent file discovery) — the caller's
/// current directory at hook time.
pub fn workspace_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_continue_verdict() {
        let out = "some analysis\nVERDICT: CONTINUE\n";
        assert_eq!(parse_verdict(out), Some(AgentVerdict::Continue));
    }

    #[test]
    fn parses_stop_verdict_with_reason() {
        let out = "analysis\nVERDICT: STOP\nREASON: val exploded, see events.ndjson:12\n";
        assert_eq!(
            parse_verdict(out),
            Some(AgentVerdict::Stop {
                reason: "val exploded, see events.ndjson:12".to_string()
            })
        );
    }

    #[test]
    fn last_verdict_wins_and_garbage_is_ignored() {
        let out = "VERDICT: CONTINUE\nblah VERDICT: MAYBE\nVERDICT: STOP\nREASON: bad\n";
        assert_eq!(
            parse_verdict(out),
            Some(AgentVerdict::Stop {
                reason: "bad".to_string()
            })
        );
    }

    #[test]
    fn missing_verdict_is_none() {
        assert_eq!(parse_verdict("just some text\nno markers\n"), None);
    }

    #[test]
    fn stop_without_reason_gets_placeholder() {
        assert_eq!(
            parse_verdict("VERDICT: STOP\n"),
            Some(AgentVerdict::Stop {
                reason: "no reason given".to_string()
            })
        );
    }

    #[test]
    fn prompt_contains_epoch_table_and_paths() {
        let history = vec![
            EpochRow {
                epoch: 1,
                train_loss: 1.0,
                val_loss: 0.9,
                phys_loss: 0.95,
                lr: 5e-4,
            },
            EpochRow {
                epoch: 2,
                train_loss: 0.8,
                val_loss: 0.85,
                phys_loss: 0.9,
                lr: 4e-4,
            },
        ];
        let p = epoch_prompt(
            "gnn_kinematics",
            Path::new("/tmp/gnn-smoke"),
            "canonical.parquet",
            &history,
            0.85,
            "tail...",
        );
        assert!(p.contains("1 | 1.000000 | 0.900000 | 0.950000"), "{p}");
        assert!(p.contains("/tmp/gnn-smoke/events.ndjson"), "{p}");
        assert!(p.contains("best_val_loss=0.850000"), "{p}");
    }

    #[test]
    fn interval_gating_skips_undue_epochs() {
        let cfg = AgentHookConfig {
            every: 5,
            ..AgentHookConfig::default()
        };
        // Epoch 3 is not due: returns Continue without spawning anything.
        // (Due epochs spawn a real `opencode run` — never in unit tests.)
        assert_eq!(
            maybe_consult_agent(&cfg, 3, Path::new("/nonexistent"), "prompt"),
            AgentVerdict::Continue
        );
    }

    #[test]
    fn dry_run_prints_prompt_and_continues() {
        let cfg = AgentHookConfig {
            every: 1,
            dry_run: true,
            ..AgentHookConfig::default()
        };
        assert_eq!(
            maybe_consult_agent(&cfg, 7, Path::new("/nonexistent"), "hello-agent"),
            AgentVerdict::Continue
        );
    }
}
