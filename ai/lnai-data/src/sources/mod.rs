
pub mod irsa;
pub mod jpl_horizons;
pub mod mast;
pub mod nasa_exoplanet;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceAuth {
    Anonymous,
    EnvKeys {
        requires: Vec<String>,
    },
}

impl SourceAuth {
    pub fn env_keys(&self) -> &[String] {
        static EMPTY: [String; 0] = [];
        match self {
            SourceAuth::Anonymous => &EMPTY,
            SourceAuth::EnvKeys { requires } => requires,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SourceProvenance {
    pub adapter_id: String,
    pub catalog_name: String,
    pub release_or_version: String,
    pub endpoint_url: String,
    pub auth: SourceAuth,
    pub enters_stellar_backbone: bool,
}

pub trait SourceAdapter {
    fn id(&self) -> &'static str;
    fn provenance(&self) -> SourceProvenance;
}
