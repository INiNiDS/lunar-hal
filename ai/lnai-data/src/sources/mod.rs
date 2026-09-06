//! Stage 4A: source adapters and auth contracts.
//!
//! Design rules from the plan:
//! * Gaia remains the canonical backbone; NASA sources are enrichment only.
//! * Anonymous access must work for every public endpoint (`PSCompPars` TAP,
//!   MAST public products); authenticated sources are opt-in via env only.
//! * No credential value may reach logs, manifests, artifacts or the frontend.

pub mod irsa;
pub mod jpl_horizons;
pub mod mast;
pub mod nasa_exoplanet;

use serde::{Deserialize, Serialize};

/// How an adapter authenticates. `Anonymous` MUST work for all public
/// endpoints used by the default pipeline.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceAuth {
    /// No credentials involved at all.
    Anonymous,
    /// Credentials resolved out-of-band from environment variables whose
    /// *names* are recorded here, never their values.
    EnvKeys {
        /// Environment variable names required when this source is active.
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

/// Machine-readable provenance appended to every derived artifact so the
/// manifest can answer "where did this column come from".
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SourceProvenance {
    /// Stable adapter id, e.g. "nasa_exoplanet_tap_pscomppars_v1".
    pub adapter_id: String,
    /// Human-readable catalog name, e.g. "NASA Exoplanet Archive PSCompPars".
    pub catalog_name: String,
    pub release_or_version: String,
    /// Public sync/sync-TAP URL used by the adapter.
    pub endpoint_url: String,
    pub auth: SourceAuth,
    /// Whether rows contributed by this source must never enter the stellar
    /// backbone (e.g. JPL Horizons Solar System bodies).
    pub enters_stellar_backbone: bool,
}

/// Contract shared by every external source adapter. Implementations must be
/// deterministic given identical fixture/response bytes, so CI can replay
/// recorded fixtures without any network.
pub trait SourceAdapter {
    fn id(&self) -> &'static str;
    fn provenance(&self) -> SourceProvenance;
}
