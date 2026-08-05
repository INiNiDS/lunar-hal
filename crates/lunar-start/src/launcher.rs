//! Launcher — connects LogBackend to stdout output.

use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;

use crate::backend::{LogBackend, LogEvent, cargo_build};
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
        let service_names = self
            .config
            .services
            .iter()
            .map(|service| service.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "[lns] workspace={} build_release={} watch={} services=[{}]",
            self.config.workspace.display(),
            self.config.build_release,
            self.config.watch,
            service_names
        );

        if self.config.build_release && !self.config.watch {
            eprintln!("[lns] Building release binaries...");
            cargo_build(&self.config.workspace, &self.config.all_build_args()).await?;
            eprintln!("[lns] Release build completed");
        }

        let (log_tx, mut log_rx) = mpsc::channel::<LogEvent>(200);
        let mut backend = LogBackend::new(&self.config, log_tx);

        backend.start_all().await;
        for service in backend.services() {
            eprintln!(
                "[lns] [{}] status={:?} pid={:?}",
                service.config.name, service.status, service.pid
            );
        }
        if !backend.any_running() {
            anyhow::bail!("[lns] No services started; see errors above");
        }

        let (ctrl_tx, mut ctrl_rx) = mpsc::channel::<()>(1);
        ctrlc::set_handler(move || {
            let _ = ctrl_tx.try_send(());
        })?;

        let mut tick_interval = tokio::time::interval(Duration::from_millis(200));

        loop {
            tokio::select! {
                log = log_rx.recv() => {
                    if let Some(log) = log {
                        let level = match log.level {
                            crate::backend::LogLevel::Info => "INFO",
                            crate::backend::LogLevel::Warn => "WARN",
                            crate::backend::LogLevel::Error => "ERROR",
                        };
                        let stream = if log.is_stderr { "stderr" } else { "stdout" };
                        println!("[{}] [{}] [{level}] [{stream}] {}", log.timestamp, log.service, log.text);
                    }
                }
                _ = tick_interval.tick() => {
                    backend.poll();
                    if !backend.any_running() {
                        eprintln!("[lns] All services have stopped");
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
        eprintln!("[lns] Shutdown complete");
        Ok(())
    }
}
