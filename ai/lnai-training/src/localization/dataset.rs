use lnai_data::split::{is_spatial_holdout, spatial_tile_id};
use lnai_models::GraphBatch;
use serde::{Deserialize, Serialize};

/// Record for a single star in a neighborhood.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalStar {
    pub source_id: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
    pub ruwe: f32,
    pub is_visible: bool,
}

/// A localized neighborhood extracted around an anchor star.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Neighborhood {
    pub anchor_id: String,
    pub anchor_pos: [f32; 3],
    pub radius_pc: f32,
    pub visible_stars: Vec<LocalStar>,
    pub hidden_stars: Vec<LocalStar>,
}

impl Neighborhood {
    /// Total star count in the neighborhood (visible + hidden).
    pub fn total_stars(&self) -> usize {
        self.visible_stars.len() + self.hidden_stars.len()
    }

    /// Whether this is a negative neighborhood (has no hidden stars).
    pub fn is_negative_example(&self) -> bool {
        self.hidden_stars.is_empty()
    }
}

/// Error returned when leakage audit fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeakageError {
    HiddenCoordinateLeaked(String),
    HiddenEdgePresent(String),
    SpatialHoldoutViolation(String),
}

impl std::fmt::Display for LeakageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeakageError::HiddenCoordinateLeaked(m) => {
                write!(f, "leakage: hidden coordinate leaked into inputs: {m}")
            }
            LeakageError::HiddenEdgePresent(m) => {
                write!(
                    f,
                    "leakage: graph contains edge to or from hidden node: {m}"
                )
            }
            LeakageError::SpatialHoldoutViolation(m) => {
                write!(f, "leakage: spatial holdout tile violation: {m}")
            }
        }
    }
}

impl std::error::Error for LeakageError {}

/// Strict audit to guarantee zero target leakage in localization inputs.
/// Exit gate requirement for Stage 8 & 9:
/// 1. Encoder/inference graph is built strictly over visible nodes.
/// 2. Hidden coordinates (x/y/z) and target-derived edges are NEVER present in visible features or adjacency.
/// 3. Stars located in spatial holdout tiles are strictly forbidden from training sets.
pub fn audit_leakage(
    visible_stars: &[LocalStar],
    hidden_stars: &[LocalStar],
    graph: &GraphBatch,
    is_training: bool,
) -> Result<(), LeakageError> {
    // 1. Check graph node count matches visible stars only
    if graph.num_nodes != visible_stars.len() {
        return Err(LeakageError::HiddenEdgePresent(format!(
            "graph num_nodes ({}) does not match visible_stars ({})",
            graph.num_nodes,
            visible_stars.len()
        )));
    }

    // 2. Ensure visible stars do not contain hidden star IDs or coordinates
    let hidden_ids: std::collections::HashSet<&str> =
        hidden_stars.iter().map(|s| s.source_id.as_str()).collect();

    for star in visible_stars {
        if hidden_ids.contains(star.source_id.as_str()) {
            return Err(LeakageError::HiddenCoordinateLeaked(format!(
                "star {} is present in both visible and hidden sets",
                star.source_id
            )));
        }
        if !star.is_visible {
            return Err(LeakageError::HiddenCoordinateLeaked(format!(
                "star {} is marked is_visible=false in visible set",
                star.source_id
            )));
        }
        if is_training && is_spatial_holdout(star.ra_deg, star.dec_deg) {
            return Err(LeakageError::SpatialHoldoutViolation(format!(
                "star {} falls in spatial holdout tile {}",
                star.source_id,
                spatial_tile_id(star.ra_deg, star.dec_deg)
            )));
        }
    }

    // 3. Ensure edge indices are strictly within [0, visible_stars.len())
    for (i, &col) in graph.col_indices.iter().enumerate() {
        if col >= visible_stars.len() {
            return Err(LeakageError::HiddenEdgePresent(format!(
                "edge {i} references node index {col} outside visible nodes (0..{})",
                visible_stars.len()
            )));
        }
    }

    Ok(())
}

/// Builds visible node feature matrix of shape [N, 8]:
/// [rel_x, rel_y, rel_z, bp_rp, g_mag, ruwe, is_visible, is_anchor]
/// scaled by `radius_pc`.
pub fn build_visible_node_features(
    anchor_pos: [f32; 3],
    visible_stars: &[LocalStar],
    radius_pc: f32,
) -> Vec<f32> {
    let n = visible_stars.len();
    let mut features = Vec::with_capacity(n * 8);
    let inv_r = if radius_pc > 1e-4 {
        1.0 / radius_pc
    } else {
        1.0
    };

    for star in visible_stars {
        let dx = (star.x - anchor_pos[0]) * inv_r;
        let dy = (star.y - anchor_pos[1]) * inv_r;
        let dz = (star.z - anchor_pos[2]) * inv_r;
        let is_anchor = if (star.x - anchor_pos[0]).abs() < 1e-4
            && (star.y - anchor_pos[1]).abs() < 1e-4
            && (star.z - anchor_pos[2]).abs() < 1e-4
        {
            1.0f32
        } else {
            0.0f32
        };

        features.push(dx);
        features.push(dy);
        features.push(dz);
        features.push((star.bp_rp - 1.0).clamp(-5.0, 5.0)); // normalized bp_rp
        features.push((star.g_mag - 15.0) * 0.1); // normalized g_mag
        features.push((star.ruwe - 1.0).clamp(-2.0, 5.0)); // normalized ruwe
        features.push(1.0); // is_visible = 1.0
        features.push(is_anchor);
    }

    features
}

/// Builds a sparse CSR k-NN graph strictly over visible nodes.
pub fn build_visible_graph_batch(visible_stars: &[LocalStar], knn_k: usize) -> GraphBatch {
    let n = visible_stars.len();
    if n == 0 {
        return GraphBatch::empty();
    }

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut edge_weights = Vec::new();

    row_ptr.push(0);
    let k = knn_k.min(n.saturating_sub(1)).max(1);

    for i in 0..n {
        let si = &visible_stars[i];
        let mut dists = Vec::with_capacity(n);
        for (j, sj) in visible_stars.iter().enumerate() {
            if i == j {
                continue;
            }
            let d2 = (si.x - sj.x).powi(2) + (si.y - sj.y).powi(2) + (si.z - sj.z).powi(2);
            dists.push((d2, j));
        }
        dists.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Self-loop first
        col_indices.push(i);
        edge_weights.push(1.0f32);

        // K nearest neighbors
        for (d2, j) in dists.into_iter().take(k) {
            let dist = d2.sqrt();
            let weight = 1.0 / (1.0 + dist);
            col_indices.push(j);
            edge_weights.push(weight);
        }

        row_ptr.push(col_indices.len());
    }

    GraphBatch::new(n, row_ptr, col_indices, edge_weights)
}

/// Deterministic synthetic catalog generator for unit/integration tests and baseline verification.
pub fn generate_synthetic_stars(count: usize, seed: u64, max_radius_pc: f32) -> Vec<LocalStar> {
    let mut lcg = seed;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 11) % 10_000) as f32 / 10_000.0
    };

    (0..count)
        .map(|i| {
            let ra_deg = (next() * 360.0) as f64;
            let dec_deg = (next() * 180.0 - 90.0) as f64;

            let r = max_radius_pc * next().cbrt();
            let theta = next() * std::f32::consts::PI;
            let phi = next() * 2.0 * std::f32::consts::PI;

            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();

            let bp_rp = next() * 2.5 + 0.3;
            let g_mag = next() * 10.0 + 8.0;
            let ruwe = 1.0 + (next() - 0.5) * 0.4;

            LocalStar {
                source_id: format!("syn_{i}_{seed}"),
                ra_deg,
                dec_deg,
                x,
                y,
                z,
                bp_rp,
                g_mag,
                ruwe,
                is_visible: true,
            }
        })
        .collect()
}

/// Partitions a localized group into visible and hidden stars according to mask_ratio.
pub fn mask_neighborhood(
    mut stars: Vec<LocalStar>,
    mask_ratio: f32,
    seed: u64,
) -> (Vec<LocalStar>, Vec<LocalStar>) {
    if stars.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Star at index 0 is always anchor and visible
    stars[0].is_visible = true;

    let mut lcg = seed;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 11) % 10_000) as f32 / 10_000.0
    };

    let mut visible = vec![stars[0].clone()];
    let mut hidden = Vec::new();

    for star in stars.into_iter().skip(1) {
        if next() < mask_ratio {
            let mut s = star;
            s.is_visible = false;
            hidden.push(s);
        } else {
            visible.push(star);
        }
    }

    (visible, hidden)
}
