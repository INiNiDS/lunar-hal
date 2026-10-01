pub fn hungarian_match(cost_matrix: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let k = cost_matrix.len();
    if k == 0 {
        return Vec::new();
    }
    let m = cost_matrix[0].len();
    if m == 0 {
        return Vec::new();
    }

    let n = k.max(m);
    let mut cost = vec![vec![0.0f64; n + 1]; n + 1];
    for i in 0..m {
        for j in 0..k {
            cost[i + 1][j + 1] = cost_matrix[j][i] as f64;
        }
    }

    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;

            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0][j] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignments = Vec::with_capacity(m);
    for j in 1..=n {
        let i = p[j];
        if i >= 1 && i <= m && j >= 1 && j <= k {
            assignments.push((j - 1, i - 1));
        }
    }
    assignments.sort_by_key(|a| a.0);
    assignments
}

pub fn huber_loss_1d(diff: f32, delta: f32) -> f32 {
    let abs_diff = diff.abs();
    if abs_diff <= delta {
        0.5 * abs_diff * abs_diff
    } else {
        delta * (abs_diff - 0.5 * delta)
    }
}

pub fn huber_loss_3d(pred: [f32; 3], target: [f32; 3], delta: f32) -> f32 {
    huber_loss_1d(pred[0] - target[0], delta)
        + huber_loss_1d(pred[1] - target[1], delta)
        + huber_loss_1d(pred[2] - target[2], delta)
}

pub fn gaussian_nll_3d(pred_pos: [f32; 3], target_pos: [f32; 3], variances: [f32; 3]) -> f32 {
    let mut nll = 0.0f32;
    for c in 0..3 {
        let var = variances[c].max(1e-4);
        let diff = pred_pos[c] - target_pos[c];
        nll += 0.5 * (diff * diff / var + var.ln());
    }
    nll
}

pub fn chamfer_distance_3d(set_a: &[[f32; 3]], set_b: &[[f32; 3]]) -> f32 {
    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }

    let mut sum_a_to_b = 0.0f32;
    for a in set_a {
        let mut min_d2 = f32::INFINITY;
        for b in set_b {
            let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
            if d2 < min_d2 {
                min_d2 = d2;
            }
        }
        sum_a_to_b += min_d2.sqrt();
    }

    let mut sum_b_to_a = 0.0f32;
    for b in set_b {
        let mut min_d2 = f32::INFINITY;
        for a in set_a {
            let d2 = (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2);
            if d2 < min_d2 {
                min_d2 = d2;
            }
        }
        sum_b_to_a += min_d2.sqrt();
    }

    (sum_a_to_b / set_a.len() as f32 + sum_b_to_a / set_b.len() as f32) * 0.5
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SetLossBreakdown {
    pub total_loss: f32,
    pub existence_loss: f32,
    pub position_loss: f32,
    pub position_nll: f32,
    pub feature_loss: f32,
    pub count_loss: f32,
    pub chamfer_loss: f32,
    pub calibration_loss: f32,
    pub matched_slots: usize,
    pub true_count: usize,
}

pub fn compute_set_loss(
    slot_raw: &[[f32; 16]],

    hidden_targets: &[[f32; 5]],

    radius_pc: f32,
    weights: &crate::spec::LocalizationLossWeights,
) -> SetLossBreakdown {
    let k = slot_raw.len();
    let m = hidden_targets.len();

    let mut slot_probs = Vec::with_capacity(k);
    let mut slot_positions = Vec::with_capacity(k);
    let mut slot_variances = Vec::with_capacity(k);
    let mut slot_features = Vec::with_capacity(k);

    for s in slot_raw {
        let logit = s[0];
        let prob = 1.0 / (1.0 + (-logit).exp());
        slot_probs.push(prob);
        slot_positions.push([s[1], s[2], s[3]]);
        slot_variances.push([
            (s[4].exp() * 0.1).clamp(1e-4, 100.0),
            (s[5].exp() * 0.1).clamp(1e-4, 100.0),
            (s[6].exp() * 0.1).clamp(1e-4, 100.0),
        ]);
        slot_features.push([s[10], s[11]]);
    }

    let mut cost_matrix = vec![vec![0.0f32; m]; k];
    for i in 0..k {
        for j in 0..m {
            let pos_cost = (slot_positions[i][0] - hidden_targets[j][0]).abs()
                + (slot_positions[i][1] - hidden_targets[j][1]).abs()
                + (slot_positions[i][2] - hidden_targets[j][2]).abs();
            let feat_cost = (slot_features[i][0] - hidden_targets[j][3]).abs()
                + (slot_features[i][1] - hidden_targets[j][4]).abs();
            let exist_cost = -slot_raw[i][0];


            cost_matrix[i][j] = pos_cost * 2.0 + feat_cost * 0.5 + exist_cost;
        }
    }

    let matches = hungarian_match(&cost_matrix);
    let matched_slot_indices: std::collections::HashSet<usize> =
        matches.iter().map(|(s, _)| *s).collect();

    let mut existence_loss = 0.0f32;
    let mut position_loss = 0.0f32;
    let mut position_nll = 0.0f32;
    let mut feature_loss = 0.0f32;
    let mut calibration_loss = 0.0f32;

    for &(slot_idx, tgt_idx) in &matches {
        let logit = slot_raw[slot_idx][0];
        existence_loss += (1.0 + (-logit).exp()).ln();

        let pred_pos = slot_positions[slot_idx];
        let tgt_pos = [
            hidden_targets[tgt_idx][0],
            hidden_targets[tgt_idx][1],
            hidden_targets[tgt_idx][2],
        ];

        let huber = huber_loss_3d(pred_pos, tgt_pos, 0.1);
        position_loss += huber;

        let nll = gaussian_nll_3d(pred_pos, tgt_pos, slot_variances[slot_idx]);
        position_nll += nll;

        let f_diff = (slot_features[slot_idx][0] - hidden_targets[tgt_idx][3]).abs()
            + (slot_features[slot_idx][1] - hidden_targets[tgt_idx][4]).abs();
        feature_loss += f_diff;

        let res2 = (pred_pos[0] - tgt_pos[0]).powi(2)
            + (pred_pos[1] - tgt_pos[1]).powi(2)
            + (pred_pos[2] - tgt_pos[2]).powi(2);
        let mean_var = (slot_variances[slot_idx][0]
            + slot_variances[slot_idx][1]
            + slot_variances[slot_idx][2])
            / 3.0;
        calibration_loss += (mean_var - res2).abs();
    }

    for i in 0..k {
        if !matched_slot_indices.contains(&i) {
            let logit = slot_raw[i][0];
            existence_loss += (1.0 + logit.exp()).ln();
        }
    }

    let total_pred_count: f32 = slot_probs.iter().sum();
    let count_loss = (total_pred_count - m as f32).abs();

    let active_pred_pos: Vec<[f32; 3]> = slot_positions
        .iter()
        .enumerate()
        .filter(|(i, _)| slot_probs[*i] >= 0.5)
        .map(|(_, p)| [p[0] * radius_pc, p[1] * radius_pc, p[2] * radius_pc])
        .collect();

    let tgt_pos_scaled: Vec<[f32; 3]> = hidden_targets
        .iter()
        .map(|t| [t[0] * radius_pc, t[1] * radius_pc, t[2] * radius_pc])
        .collect();

    let chamfer_loss = if active_pred_pos.is_empty() || tgt_pos_scaled.is_empty() {
        count_loss * radius_pc * 0.1
    } else {
        chamfer_distance_3d(&active_pred_pos, &tgt_pos_scaled)
    };

    let denom = m.max(1) as f32;
    let k_denom = k.max(1) as f32;

    let norm_exist = existence_loss / k_denom;
    let norm_pos = position_loss / denom;
    let norm_nll = position_nll / denom;
    let norm_feat = feature_loss / denom;
    let norm_cal = calibration_loss / denom;

    let total = norm_exist * weights.existence
        + norm_nll * weights.position_nll
        + chamfer_loss * weights.chamfer
        + norm_feat * weights.feature
        + norm_cal * weights.calibration
        + count_loss * 0.2;

    SetLossBreakdown {
        total_loss: total,
        existence_loss: norm_exist,
        position_loss: norm_pos,
        position_nll: norm_nll,
        feature_loss: norm_feat,
        count_loss,
        chamfer_loss,
        calibration_loss: norm_cal,
        matched_slots: matches.len(),
        true_count: m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_matching_finds_minimum_cost_permutation() {
        let cost = vec![
            vec![1.0, 5.0, 9.0],
            vec![8.0, 2.0, 4.0],
            vec![7.0, 6.0, 3.0],
        ];
        let matches = hungarian_match(&cost);
        assert_eq!(matches, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn hungarian_matching_rectangular_more_slots_than_targets() {
        let cost = vec![
            vec![10.0, 2.0],
            vec![1.0, 10.0],
            vec![5.0, 5.0],
            vec![8.0, 8.0],
        ];
        let matches = hungarian_match(&cost);
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(1, 0)));
        assert!(matches.contains(&(0, 1)));
    }

    #[test]
    fn chamfer_distance_zero_for_identical_sets() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, 0.5]];
        let cd = chamfer_distance_3d(&pts, &pts);
        assert!(
            cd < 1e-6,
            "chamfer distance between identical sets must be 0"
        );
    }

    #[test]
    fn huber_and_gaussian_nll_finite() {
        let p = [1.0, -2.0, 0.5];
        let t = [1.2, -1.8, 0.4];
        let v = [0.1, 0.2, 0.05];

        let h = huber_loss_3d(p, t, 0.1);
        let nll = gaussian_nll_3d(p, t, v);

        assert!(h.is_finite() && h > 0.0);
        assert!(nll.is_finite());
    }
}
