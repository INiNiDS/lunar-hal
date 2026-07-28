//! Launcher — connects LogBackend to stdout output.

use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;

use crate::backend::{cargo_build, LogBackend, LogEvent};
use crate::config::LauncherConfig;

/// Launcher: starts services and streams logs to stdout.
pub struct Launcher {
    config: LauncherConfig,
}

impl Launcher {
    pub fn new(config: LauncherConfig) -> Self {
        Self { config }
    }

    /// Start all services and stream logs to stdout.
    pub async fn run(self) -> Result<()> {
        self.config.apply_env();

        if self.config.build_release && !self.config.watch {
            cargo_build(&self.config.workspace, &self.config.all_build_args())?;
        }

        let (log_tx, mut log_rx) = mpsc::channel::<LogEvent>(200);
        let mut backend = LogBackend::new(&self.config, log_tx);

        backend.start_all().await;

        let (ctrl_tx, mut ctrl_rx) = mpsc::channel::<()>(1);
        ctrlc::set_handler(move || {
            let _ = ctrl_tx.try_send(());
        })?;

        let mut tick_interval = tokio::time::interval(Duration::from_millis(200));

        loop {
            tokio::select! {
                log = log_rx.recv() => {
                    if let Some(log) = log {
                        let prefix = if log.is_stderr { "[ERR]" } else { "[OUT]" };
                        println!("[{}] [{}] {} {}", log.timestamp, log.service, prefix, log.text);
                    }
                }
                _ = tick_interval.tick() => {
                    backend.poll();
                    if !backend.any_running() {
                        break;
                    }
                }
                Some(()) = ctrl_rx.recv() => {
                    eprintln!("\nStopping all services...");
                    break;
                }
            }
        }

        backend.stop_all().await;
        Ok(())
    }
}
