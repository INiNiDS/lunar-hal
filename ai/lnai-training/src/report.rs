
use serde::{Deserialize, Serialize};

pub const REPORT_SCHEMA_VERSION: &str = "1.0.0";

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReportKind {
    Correctness,
    Performance,
    E2E,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RunIdentity {
    pub dataset_manifest_hash: String,
    pub artifact_hash: Option<String>,
    pub device: String,
    pub seed: u64,
    pub git_revision: Option<String>,
    pub created_ms: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Metric {
    pub name: String,
    pub value: f64,
}

impl Metric {
    pub fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BaselineStats {
    pub runs_ms: Vec<f64>,
    pub mean_ms: f64,
    pub stddev_ms: f64,
    pub rel_spread_pct: f64,
}

impl BaselineStats {
    pub fn from_runs(runs_ms: Vec<f64>) -> Option<Self> {
        if runs_ms.is_empty() {
            return None;
        }
        let n = runs_ms.len() as f64;
        let mean = runs_ms.iter().sum::<f64>() / n;
        let variance = runs_ms.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let stddev = variance.sqrt();
        let rel_spread_pct = if mean > 0.0 {
            stddev / mean * 100.0
        } else {
            0.0
        };
        Some(Self {
            runs_ms,
            mean_ms: mean,
            stddev_ms: stddev,
            rel_spread_pct,
        })
    }

    pub fn is_stable_enough(&self) -> bool {
        self.rel_spread_pct < 25.0
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ReportV1 {
    pub version: String,
    pub kind: ReportKind,
    pub suite: String,
    pub identity: RunIdentity,
    pub metrics: Vec<Metric>,
    pub passed: bool,
    pub notes: Vec<String>,
}

impl ReportV1 {
    pub fn new(kind: ReportKind, suite: &str, identity: RunIdentity) -> Self {
        Self {
            version: REPORT_SCHEMA_VERSION.to_string(),
            kind,
            suite: suite.to_string(),
            identity,
            metrics: Vec::new(),
            passed: false,
            notes: Vec::new(),
        }
    }

    pub fn add_metric(&mut self, name: &str, value: f64) -> &mut Self {
        self.metrics.push(Metric::new(name, value));
        self
    }

    pub fn add_note(&mut self, note: impl Into<String>) -> &mut Self {
        self.notes.push(note.into());
        self
    }

    pub fn metric(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.value)
    }

    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn write_to_dir(&self, dir: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{}.json", self.suite));
        std::fs::write(&path, self.to_json_string().unwrap_or_default())?;
        Ok(path)
    }

    pub fn default_dir() -> std::path::PathBuf {
        std::path::PathBuf::from("target/ai-reports")
    }

    pub fn configured_dir() -> std::path::PathBuf {
        match std::env::var("LUNAR_AI_REPORT_DIR") {
            Ok(dir) if !dir.is_empty() => std::path::PathBuf::from(dir),
            _ => Self::default_dir(),
        }
    }
}

pub fn identity_from_env(default_suite_seed: u64) -> RunIdentity {
    let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    RunIdentity {
        dataset_manifest_hash: var("LUNAR_AI_DATASET_MANIFEST_HASH").unwrap_or_default(),
        artifact_hash: var("LUNAR_AI_ARTIFACT_HASH"),
        device: var("LUNAR_AI_DEVICE").unwrap_or_else(|| "cpu".to_string()),
        seed: var("LUNAR_AI_SEED")
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_suite_seed),
        git_revision: var("LUNAR_AI_GIT_REV"),
        created_ms: lunar_utils::time::current_time_ms(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_identity() -> RunIdentity {
        RunIdentity {
            dataset_manifest_hash: "ds-hash".into(),
            artifact_hash: Some("artifact-hash".into()),
            device: "cpu".into(),
            seed: 42,
            git_revision: Some("abc1234".into()),
            created_ms: 1_720_000_000_000,
        }
    }

    #[test]
    fn report_round_trips_through_json() {
        let mut report = ReportV1::new(ReportKind::Correctness, "pinn_accuracy", sample_identity());
        report.add_metric("mse_log10_teff", 0.01);
        report.add_note("report-only baseline");
        let json = report.to_json_string().expect("serialize report");
        let back: ReportV1 = serde_json::from_str(&json).expect("deserialize report");
        assert_eq!(back, report);
        assert_eq!(back.metric("mse_log10_teff"), Some(0.01));
    }

    #[test]
    fn report_kind_is_tagged_snake_case() {
        let report = ReportV1::new(ReportKind::Performance, "bench", sample_identity());
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["kind"], "performance");
        assert_eq!(value["version"], REPORT_SCHEMA_VERSION);
    }

    #[test]
    fn baseline_stats_compute_mean_stddev_and_spread() {
        let stats = BaselineStats::from_runs(vec![100.0, 110.0, 90.0]).unwrap();
        assert!((stats.mean_ms - 100.0).abs() < 1e-9);
        assert!(stats.rel_spread_pct < 25.0 && stats.is_stable_enough());

        let noisy = BaselineStats::from_runs(vec![50.0, 150.0]).unwrap();
        assert!(noisy.rel_spread_pct > 25.0 && !noisy.is_stable_enough());
        assert!(BaselineStats::from_runs(vec![]).is_none());
    }

    #[test]
    fn report_version_is_frozen() {
        assert_eq!(REPORT_SCHEMA_VERSION, "1.0.0");
    }
}
