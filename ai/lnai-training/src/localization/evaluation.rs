use lnai_models::{LocalizationOutput, StarCandidate};
use serde::{Deserialize, Serialize};

use crate::localization::loss::{chamfer_distance_3d, hungarian_match};

/// Evaluation report for masked-coordinate localization (Stage 8).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaskedEvaluationReport {
    pub sample_count: usize,
    pub median_error_pc: f32,
    pub p95_error_pc: f32,
    pub angular_error_deg_median: f32,
    pub distance_error_pc_median: f32,
    pub coverage_50: f32,
    pub coverage_90: f32,
    pub coverage_95: f32,
    pub baseline_mlp_median_pc: f32,
    pub baseline_knn_median_pc: f32,
}

/// Evaluation report for missing-neighbor set reconstruction (Stage 9).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeighborsEvaluationReport {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub recall_at_k: f32,
    pub false_discoveries_per_neighborhood: f32,
    pub count_mae: f32,
    pub median_matched_error_pc: f32,
    pub p95_matched_error_pc: f32,
    pub chamfer_distance: f32,
    pub coverage_50: f32,
    pub coverage_90: f32,
    pub coverage_95: f32,
    pub baseline_poisson_chamfer: f32,
}

/// Single masked-coordinate sample evaluation data.
#[derive(Debug, Clone)]
pub struct MaskedSampleEval {
    pub pred_pos: [f32; 3],
    pub true_pos: [f32; 3],
    pub variances: [f32; 3],
    pub mlp_pred_pos: [f32; 3],
    pub knn_pred_pos: [f32; 3],
}

/// Computes evaluation report across multiple masked-coordinate evaluation samples.
pub fn evaluate_masked_set(samples: &[MaskedSampleEval]) -> MaskedEvaluationReport {
    if samples.is_empty() {
        return MaskedEvaluationReport {
            sample_count: 0,
            median_error_pc: 0.0,
            p95_error_pc: 0.0,
            angular_error_deg_median: 0.0,
            distance_error_pc_median: 0.0,
            coverage_50: 0.0,
            coverage_90: 0.0,
            coverage_95: 0.0,
            baseline_mlp_median_pc: 0.0,
            baseline_knn_median_pc: 0.0,
        };
    }

    let n = samples.len();
    let mut errors = Vec::with_capacity(n);
    let mut angular_errors = Vec::with_capacity(n);
    let mut distance_errors = Vec::with_capacity(n);
    let mut mlp_errors = Vec::with_capacity(n);
    let mut knn_errors = Vec::with_capacity(n);

    let mut in_cov_50 = 0usize;
    let mut in_cov_90 = 0usize;
    let mut in_cov_95 = 0usize;

    for s in samples {
        // Euclidean error
        let dx = s.pred_pos[0] - s.true_pos[0];
        let dy = s.pred_pos[1] - s.true_pos[1];
        let dz = s.pred_pos[2] - s.true_pos[2];
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        errors.push(err);

        // Baselines
        let mlp_err = ((s.mlp_pred_pos[0] - s.true_pos[0]).powi(2)
            + (s.mlp_pred_pos[1] - s.true_pos[1]).powi(2)
            + (s.mlp_pred_pos[2] - s.true_pos[2]).powi(2))
        .sqrt();
        mlp_errors.push(mlp_err);

        let knn_err = ((s.knn_pred_pos[0] - s.true_pos[0]).powi(2)
            + (s.knn_pred_pos[1] - s.true_pos[1]).powi(2)
            + (s.knn_pred_pos[2] - s.true_pos[2]).powi(2))
        .sqrt();
        knn_errors.push(knn_err);

        // Distance / radial error
        let pred_r = (s.pred_pos[0].powi(2) + s.pred_pos[1].powi(2) + s.pred_pos[2].powi(2)).sqrt();
        let true_r = (s.true_pos[0].powi(2) + s.true_pos[1].powi(2) + s.true_pos[2].powi(2)).sqrt();
        distance_errors.push((pred_r - true_r).abs());

        // Angular error in degrees
        let dot = s.pred_pos[0] * s.true_pos[0]
            + s.pred_pos[1] * s.true_pos[1]
            + s.pred_pos[2] * s.true_pos[2];
        let denom = (pred_r * true_r).max(1e-4);
        let cos_angle = (dot / denom).clamp(-1.0, 1.0);
        let angle_deg = cos_angle.acos().to_degrees();
        angular_errors.push(angle_deg);

        // Normalized Mahalanobis-like statistic under diagonal covariance:
        // stat = sum_c (diff_c^2 / var_c) ~ ChiSquared(df=3)
        // Chi-squared critical values for df=3:
        // 50%: ~2.366, 90%: ~6.251, 95%: ~7.815
        let stat = (dx * dx / s.variances[0].max(1e-4))
            + (dy * dy / s.variances[1].max(1e-4))
            + (dz * dz / s.variances[2].max(1e-4));

        if stat <= 2.366 {
            in_cov_50 += 1;
        }
        if stat <= 6.251 {
            in_cov_90 += 1;
        }
        if stat <= 7.815 {
            in_cov_95 += 1;
        }
    }

    errors.sort_by(|a, b| a.total_cmp(b));
    angular_errors.sort_by(|a, b| a.total_cmp(b));
    distance_errors.sort_by(|a, b| a.total_cmp(b));
    mlp_errors.sort_by(|a, b| a.total_cmp(b));
    knn_errors.sort_by(|a, b| a.total_cmp(b));

    let median_idx = n / 2;
    let p95_idx = ((n as f32 * 0.95).round() as usize).min(n - 1);

    MaskedEvaluationReport {
        sample_count: n,
        median_error_pc: errors[median_idx],
        p95_error_pc: errors[p95_idx],
        angular_error_deg_median: angular_errors[median_idx],
        distance_error_pc_median: distance_errors[median_idx],
        coverage_50: in_cov_50 as f32 / n as f32,
        coverage_90: in_cov_90 as f32 / n as f32,
        coverage_95: in_cov_95 as f32 / n as f32,
        baseline_mlp_median_pc: mlp_errors[median_idx],
        baseline_knn_median_pc: knn_errors[median_idx],
    }
}

/// Evaluates predicted candidates against true hidden star positions for one neighborhood.
pub fn evaluate_neighborhood_set(
    candidates: &[StarCandidate],
    true_positions: &[[f32; 3]],
    tolerance_radius_pc: f32,
) -> (usize, usize, usize, Vec<f32>, f32, [usize; 3]) {
    // Returns: (tp, fp, fn, matched_distances, chamfer, [cov50, cov90, cov95])
    let k = candidates.len();
    let m = true_positions.len();

    let cand_positions: Vec<[f32; 3]> = candidates.iter().map(|c| c.relative_position).collect();
    let chamfer = chamfer_distance_3d(&cand_positions, true_positions);

    if k == 0 {
        return (0, 0, m, Vec::new(), chamfer, [0, 0, 0]);
    }
    if m == 0 {
        return (0, k, 0, Vec::new(), chamfer, [0, 0, 0]);
    }

    // Cost matrix based on Euclidean distance
    let mut cost_matrix = vec![vec![0.0f32; m]; k];
    for i in 0..k {
        for j in 0..m {
            let d = ((cand_positions[i][0] - true_positions[j][0]).powi(2)
                + (cand_positions[i][1] - true_positions[j][1]).powi(2)
                + (cand_positions[i][2] - true_positions[j][2]).powi(2))
            .sqrt();
            cost_matrix[i][j] = d;
        }
    }

    let matches = hungarian_match(&cost_matrix);
    let mut tp = 0usize;
    let mut matched_dists = Vec::new();
    let mut cov_hits = [0usize; 3];

    for &(cand_idx, tgt_idx) in &matches {
        let dist = cost_matrix[cand_idx][tgt_idx];
        if dist <= tolerance_radius_pc {
            tp += 1;
            matched_dists.push(dist);

            let c = &candidates[cand_idx];
            let t = true_positions[tgt_idx];
            let var = c.positional_variances();
            let stat = (c.relative_position[0] - t[0]).powi(2) / var[0].max(1e-4)
                + (c.relative_position[1] - t[1]).powi(2) / var[1].max(1e-4)
                + (c.relative_position[2] - t[2]).powi(2) / var[2].max(1e-4);

            if stat <= 2.366 {
                cov_hits[0] += 1;
            }
            if stat <= 6.251 {
                cov_hits[1] += 1;
            }
            if stat <= 7.815 {
                cov_hits[2] += 1;
            }
        }
    }

    let fp = k.saturating_sub(tp);
    let false_neg = m.saturating_sub(tp);

    (tp, fp, false_neg, matched_dists, chamfer, cov_hits)
}

/// Evaluates missing-neighbor reconstruction across a batch of neighborhoods.
pub fn evaluate_neighbors_dataset(
    neighborhood_predictions: &[(LocalizationOutput, Vec<[f32; 3]>)],
    tolerance_radius_pc: f32,
    poisson_baseline_predictions: &[(LocalizationOutput, Vec<[f32; 3]>)],
) -> NeighborsEvaluationReport {
    let num_hoods = neighborhood_predictions.len();
    if num_hoods == 0 {
        return NeighborsEvaluationReport {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            recall_at_k: 0.0,
            false_discoveries_per_neighborhood: 0.0,
            count_mae: 0.0,
            median_matched_error_pc: 0.0,
            p95_matched_error_pc: 0.0,
            chamfer_distance: 0.0,
            coverage_50: 0.0,
            coverage_90: 0.0,
            coverage_95: 0.0,
            baseline_poisson_chamfer: 0.0,
        };
    }

    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;
    let mut all_matched_dists = Vec::new();
    let mut total_chamfer = 0.0f32;
    let mut count_diff_sum = 0.0f32;
    let mut total_cov_hits = [0usize; 3];

    for (output, truth) in neighborhood_predictions {
        let (tp, fp, fn_count, dists, chamfer, cov) =
            evaluate_neighborhood_set(&output.candidates, truth, tolerance_radius_pc);

        total_tp += tp;
        total_fp += fp;
        total_fn += fn_count;
        all_matched_dists.extend(dists);
        total_chamfer += chamfer;
        count_diff_sum += (output.candidates.len() as f32 - truth.len() as f32).abs();

        total_cov_hits[0] += cov[0];
        total_cov_hits[1] += cov[1];
        total_cov_hits[2] += cov[2];
    }

    // Baseline Poisson chamfer
    let mut baseline_chamfer_sum = 0.0f32;
    for (output, truth) in poisson_baseline_predictions {
        let cand_positions: Vec<[f32; 3]> = output
            .candidates
            .iter()
            .map(|c| c.relative_position)
            .collect();
        baseline_chamfer_sum += chamfer_distance_3d(&cand_positions, truth);
    }
    let baseline_poisson_chamfer =
        baseline_chamfer_sum / poisson_baseline_predictions.len().max(1) as f32;

    let precision = if total_tp + total_fp > 0 {
        total_tp as f32 / (total_tp + total_fp) as f32
    } else {
        0.0
    };

    let recall = if total_tp + total_fn > 0 {
        total_tp as f32 / (total_tp + total_fn) as f32
    } else {
        0.0
    };

    let f1 = if precision + recall > 1e-6 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    all_matched_dists.sort_by(|a, b| a.total_cmp(b));
    let (median_matched_error, p95_matched_error) = if !all_matched_dists.is_empty() {
        let mid = all_matched_dists.len() / 2;
        let p95 = ((all_matched_dists.len() as f32 * 0.95).round() as usize)
            .min(all_matched_dists.len() - 1);
        (all_matched_dists[mid], all_matched_dists[p95])
    } else {
        (tolerance_radius_pc, tolerance_radius_pc)
    };

    let matched_count = total_tp.max(1) as f32;

    NeighborsEvaluationReport {
        precision,
        recall,
        f1,
        recall_at_k: recall,
        false_discoveries_per_neighborhood: total_fp as f32 / num_hoods as f32,
        count_mae: count_diff_sum / num_hoods as f32,
        median_matched_error_pc: median_matched_error,
        p95_matched_error_pc: p95_matched_error,
        chamfer_distance: total_chamfer / num_hoods as f32,
        coverage_50: total_cov_hits[0] as f32 / matched_count,
        coverage_90: total_cov_hits[1] as f32 / matched_count,
        coverage_95: total_cov_hits[2] as f32 / matched_count,
        baseline_poisson_chamfer,
    }
}
