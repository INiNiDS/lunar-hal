use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StarCandidate {
    pub existence_prob: f32,
    pub relative_position: [f32; 3],
    pub covariance: [f32; 6],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bp_rp: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub g_mag: Option<f32>,
}

impl StarCandidate {
    pub fn positional_variances(&self) -> [f32; 3] {
        [self.covariance[0], self.covariance[1], self.covariance[2]]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizationResponse {
    pub candidates: Vec<StarCandidate>,
    pub anchor_position: [f32; 3],
    pub radius_pc: f32,
    pub version: String,
    pub seed: u64,
    pub model_used: String,
}

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizedStarWithPhysics {
    pub candidate: StarCandidate,
    pub absolute_position: [f32; 3],
    pub physics: Option<crate::PinnResponse>,
}

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PipelineLocalizePhysicsResponse {
    pub stars: Vec<LocalizedStarWithPhysics>,
    pub anchor_position: [f32; 3],
    pub radius_pc: f32,
    pub version: String,
    pub seed: u64,
}
