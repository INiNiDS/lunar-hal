
use crate::sources::SourceProvenance;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DatasetProvenanceV1 {
    pub version: String,
    pub dataset_dir: String,
    pub sources: Vec<SourceProvenance>,
    pub impact_report_reference: Option<String>,
}

impl DatasetProvenanceV1 {
    pub const VERSION: &'static str = "1.0.0";

    pub fn new(dataset_dir: &str) -> Self {
        Self {
            version: Self::VERSION.to_string(),
            dataset_dir: dataset_dir.to_string(),
            sources: Vec::new(),
            impact_report_reference: None,
        }
    }

    pub fn register(&mut self, provenance: SourceProvenance) {
        if !self
            .sources
            .iter()
            .any(|s| s.adapter_id == provenance.adapter_id)
        {
            self.sources.push(provenance);
        }
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let path = dir.join("provenance.json");
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    pub fn load(dir: &Path) -> Option<Self> {
        let raw = std::fs::read_to_string(dir.join("provenance.json")).ok()?;
        serde_json::from_str(&raw).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::SourceAdapter;

    #[test]
    fn provenance_round_trip_and_registration_is_idempotent() {
        let mut p = DatasetProvenanceV1::new("/tmp/demo");
        p.register(crate::sources::nasa_exoplanet::NasaExoplanetAdapter.provenance());
        let again = crate::sources::nasa_exoplanet::NasaExoplanetAdapter.provenance();
        p.register(again);
        assert_eq!(p.sources.len(), 1);

        let dir = tempfile::tempdir().unwrap();
        p.save(dir.path()).unwrap();
        let loaded = DatasetProvenanceV1::load(dir.path()).unwrap();
        assert_eq!(loaded, p);
    }

    #[test]
    fn provenance_never_contains_secret_values() {
        let mut p = DatasetProvenanceV1::new("d");
        p.register(crate::sources::SourceProvenance {
            adapter_id: "mast_tic_v1".into(),
            catalog_name: "MAST TIC".into(),
            release_or_version: "tic-8".into(),
            endpoint_url: "https://mast.stsci.edu/api/v0.13/Invoke".into(),
            auth: crate::sources::SourceAuth::EnvKeys {
                requires: vec!["MAST_API_TOKEN".into()],
            },
            enters_stellar_backbone: false,
        });
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.to_lowercase().contains("token_value"));
        assert!(json.contains("MAST_API_TOKEN"), "only the NAME is recorded");
        assert!(json.contains("\"requires\""));
    }
}
