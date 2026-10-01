use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum JobEvent {
    Queued,
    Started,
    Progress { epoch: u32, total_epochs: u32 },
    Metric(EpochMetric),
    Checkpoint {
        epoch: u32,
        path: String,
        hash: String,
    },
    Completed { exit_code: i32 },
    Failed {
        error_summary: String,
        exit_code: i32,
    },
    Cancelled,
}

impl JobEvent {
    pub fn to_status(&self) -> JobStatus {
        match self {
            JobEvent::Queued => JobStatus::Queued,
            JobEvent::Started => JobStatus::Running,
            JobEvent::Progress { .. } => JobStatus::Running,
            JobEvent::Metric(_) => JobStatus::Running,
            JobEvent::Checkpoint { .. } => JobStatus::Running,
            JobEvent::Completed { .. } => JobStatus::Completed,
            JobEvent::Failed { .. } => JobStatus::Failed,
            JobEvent::Cancelled => JobStatus::Cancelled,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn tag(&self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpochMetric {
    pub epoch: u32,
    pub train_loss: f64,
    pub val_loss: f64,
    pub phys_loss: Option<f64>,
    pub lr: f64,
    pub timestamp_ms: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogLineKind {
    Header,
    Info,
    Warning,
    Error,
    Metric,
    EpochStart,
    EpochEnd,
    Checkpoint,
    Raw,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LogEntry {
    pub timestamp_ms: u64,
    pub line: String,
    pub kind: LogLineKind,
}

pub fn write_event_line(writer: &mut dyn std::io::Write, event: &JobEvent) -> std::io::Result<()> {
    let line = serde_json::to_string(event)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writeln!(writer, "{line}")
}

pub fn read_event_line(line: &str) -> Option<Result<JobEvent, String>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(serde_json::from_str(trimmed).map_err(|e| e.to_string()))
}

pub fn format_epoch_line(
    epoch: u32,
    train_loss: f64,
    val_loss: f64,
    phys_loss: Option<f64>,
    lr: f64,
) -> String {
    match phys_loss {
        Some(phys) => {
            format!("{epoch:5} | {train_loss:12.6} | {val_loss:12.6} | {phys:12.6} | {lr:10.6e}",)
        }
        None => format!("{epoch:5} | {train_loss:12.6} | {val_loss:12.6} | {lr:10.6e}",),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_events_round_trip_with_typed_tags() {
        let events = vec![
            JobEvent::Queued,
            JobEvent::Started,
            JobEvent::Progress {
                epoch: 3,
                total_epochs: 50,
            },
            JobEvent::Metric(EpochMetric {
                epoch: 3,
                train_loss: 0.42,
                val_loss: 0.51,
                phys_loss: None,
                lr: 5e-4,
                timestamp_ms: 1_720_000_000_000,
            }),
            JobEvent::Checkpoint {
                epoch: 10,
                path: "runs/pinn/model.safetensors".into(),
                hash: "deadbeef".into(),
            },
            JobEvent::Completed { exit_code: 0 },
            JobEvent::Failed {
                error_summary: "OOM".into(),
                exit_code: 1,
            },
            JobEvent::Cancelled,
        ];
        for event in events {
            let value = serde_json::to_value(&event).expect("serialize JobEvent");
            assert!(value.get("event").is_some(), "tag missing: {value}");
            let back: JobEvent = serde_json::from_value(value).expect("deserialize JobEvent");
            assert_eq!(back, event);
        }
    }

    #[test]
    fn to_status_maps_every_event() {
        assert_eq!(JobEvent::Queued.to_status(), JobStatus::Queued);
        assert_eq!(
            JobEvent::Progress {
                epoch: 1,
                total_epochs: 2
            }
            .to_status(),
            JobStatus::Running
        );
        assert_eq!(
            JobEvent::Checkpoint {
                epoch: 1,
                path: "p".into(),
                hash: "h".into()
            }
            .to_status(),
            JobStatus::Running
        );
        assert_eq!(
            JobEvent::Completed { exit_code: 0 }.to_status(),
            JobStatus::Completed
        );
        assert_eq!(
            JobEvent::Failed {
                error_summary: "e".into(),
                exit_code: 1
            }
            .to_status(),
            JobStatus::Failed
        );
        assert_eq!(JobEvent::Cancelled.to_status(), JobStatus::Cancelled);
    }

    #[test]
    fn log_entries_round_trip_through_json() {
        let entry = LogEntry {
            timestamp_ms: 1,
            line: "epoch 1 done".into(),
            kind: LogLineKind::EpochEnd,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: LogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, entry);
    }
}
