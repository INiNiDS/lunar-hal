use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use lunar_start::{LogBackend, LogEvent};
use tokio::sync::{broadcast, mpsc, Mutex};

/// Max number of buffered log lines retained per service for the `/services/{name}/logs`
/// and `/services/{name}/stats` endpoints.
const MAX_LOGS_PER_SERVICE: usize = 2000;

pub type LogRingBuffers = Arc<Mutex<HashMap<String, VecDeque<LogEvent>>>>;

pub struct ServiceManager {
    log_tx: broadcast::Sender<LogEvent>,
    backend: Arc<Mutex<LogBackend>>,
    logs: LogRingBuffers,
}

impl ServiceManager {
    pub fn new(config: &lunar_start::LauncherConfig) -> Result<Self> {
        let (mpsc_tx, mut mpsc_rx) = mpsc::channel::<LogEvent>(500);
        let (bcast_tx, _) = broadcast::channel::<LogEvent>(500);
        let logs: LogRingBuffers = Arc::new(Mutex::new(HashMap::new()));

        let bcast_tx_clone = bcast_tx.clone();
        let logs_clone = Arc::clone(&logs);
        tokio::spawn(async move {
            while let Some(log) = mpsc_rx.recv().await {
                {
                    let mut guard = logs_clone.lock().await;
                    let buf = guard.entry(log.service.clone()).or_insert_with(VecDeque::new);
                    buf.push_back(log.clone());
                    if buf.len() > MAX_LOGS_PER_SERVICE {
                        buf.pop_front();
                    }
                }
                let _ = bcast_tx_clone.send(log);
            }
        });

        let backend = Arc::new(Mutex::new(LogBackend::new(config, mpsc_tx)));

        let backend_clone = Arc::clone(&backend);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            loop {
                interval.tick().await;
                backend_clone.lock().await.poll();
            }
        });

        Ok(Self {
            log_tx: bcast_tx,
            backend,
            logs,
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        broadcast::Sender<LogEvent>,
        Arc<Mutex<LogBackend>>,
        LogRingBuffers,
    ) {
        (self.log_tx, self.backend, self.logs)
    }
}
