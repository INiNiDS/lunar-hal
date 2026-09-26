use serde::{Deserialize, Serialize};

/// Known visible neighbor in the neighborhood of the anchor star.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct VisibleStarDto {
    pub source_id: Option<String>,
    pub x_pc: f32,
    pub y_pc: f32,
    pub z_pc: f32,
    #[serde(default)]
    pub bp_rp: Option<f32>,
    #[serde(default)]
    pub g_mag: Option<f32>,
    #[serde(default)]
    pub ruwe: Option<f32>,
}

/// Request to reconstruct hidden/missing neighbor stars around an anchor.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizationRequest {
    pub anchor_x: f32,
    pub anchor_y: f32,
    pub anchor_z: f32,
    pub radius_pc: f32,
    #[serde(default)]
    pub max_slots: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub visible_neighbors: Vec<VisibleStarDto>,
}

/// A predicted star candidate in the localized neighborhood.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StarCandidate {
    /// Probability [0.0, 1.0] that this candidate actually exists.
    pub existence_prob: f32,
    /// Relative 3D position [dx, dy, dz] from the anchor star (in parsecs).
    pub relative_position: [f32; 3],
    /// Lower triangle of the 3x3 covariance matrix representing positional uncertainty.
    /// Order: [xx, yy, zz, xy, xz, yz].
    pub covariance: [f32; 6],
    /// Predicted or estimated color index bp - rp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bp_rp: Option<f32>,
    /// Predicted or estimated apparent magnitude in Gaia G band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub g_mag: Option<f32>,
}

impl StarCandidate {
    /// Diagonal variances [var_x, var_y, var_z].
    pub fn positional_variances(&self) -> [f32; 3] {
        [self.covariance[0], self.covariance[1], self.covariance[2]]
    }
}

/// Response returned by the GNN-Localization service.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizationResponse {
    pub candidates: Vec<StarCandidate>,
    pub anchor_position: [f32; 3],
    pub radius_pc: f32,
    pub version: String,
    pub seed: u64,
    pub model_used: String,
}

/// Summary metrics for bipartite-matched localization evaluation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MatchedEvaluationSummary {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub count_mae: f32,
    pub median_error_pc: f32,
    pub p95_error_pc: f32,
    pub coverage_50: f32,
    pub coverage_90: f32,
    pub coverage_95: f32,
    pub chamfer_distance: f32,
}

/// Candidate paired with its physical properties derived via PINN.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizedStarWithPhysics {
    pub candidate: StarCandidate,
    pub absolute_position: [f32; 3],
    pub physics: Option<crate::PinnResponse>,
}

/// Request for chained GNN-Localization -> PINN pipeline.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PipelineLocalizePhysicsRequest {
    pub anchor_x: f32,
    pub anchor_y: f32,
    pub anchor_z: f32,
    pub radius_pc: f32,
    #[serde(default)]
    pub max_slots: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub visible_neighbors: Vec<VisibleStarDto>,
}

/// Response containing candidates enriched with PINN physical properties.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PipelineLocalizePhysicsResponse {
    pub stars: Vec<LocalizedStarWithPhysics>,
    pub anchor_position: [f32; 3],
    pub radius_pc: f32,
    pub version: String,
    pub seed: u64,
}
