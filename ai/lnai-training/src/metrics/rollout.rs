//! Position rollout kinematic-consistency test (Stage 3).
//!
//! Integrates positions from predicted velocities and compares the rollout
//! against the ground-truth trajectory. For a constant velocity error the
//! gap must grow **linearly** with step count — sub-linear growth indicates
//! error cancellation (suspicious smoothing), super-linear growth indicates
//! instability.

/// Result of a rollout comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct RolloutResult {
    /// Mean position error (in position units) after each step, index 0 = first step.
    pub errors: Vec<f64>,
}

impl RolloutResult {
    /// Final-step mean error.
    pub fn final_error(&self) -> f64 {
        self.errors.last().copied().unwrap_or(0.0)
    }
}

/// Rolls both prediction and truth forward with their respective velocities.
///
/// * `p0`      — initial positions `[N, 3]`
/// * `v_pred`  — velocities under test `[N, 3]`
/// * `v_truth` — reference velocities `[N, 3]`
/// * `dt`      — integration step
pub fn position_rollout(
    p0: &[[f32; 3]],
    v_pred: &[[f32; 3]],
    v_truth: &[[f32; 3]],
    dt: f32,
    steps: usize,
) -> RolloutResult {
    assert_eq!(p0.len(), v_pred.len(), "p0/v_pred length mismatch");
    assert_eq!(p0.len(), v_truth.len(), "p0/v_truth length mismatch");

    let mut errors = Vec::with_capacity(steps);
    for step in 1..=steps {
        let t = dt * step as f32;
        let mut sq_sum = 0.0_f64;
        for i in 0..p0.len() {
            let pred = [
                p0[i][0] + v_pred[i][0] * t,
                p0[i][1] + v_pred[i][1] * t,
                p0[i][2] + v_pred[i][2] * t,
            ];
            let truth = [
                p0[i][0] + v_truth[i][0] * t,
                p0[i][1] + v_truth[i][1] * t,
                p0[i][2] + v_truth[i][2] * t,
            ];
            let d = [pred[0] - truth[0], pred[1] - truth[1], pred[2] - truth[2]];
            sq_sum += (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f64;
        }
        errors.push((sq_sum / p0.len() as f64).sqrt());
    }
    RolloutResult { errors }
}

/// Mean squared error over `[N, 3]` batches (shared with GNN metrics).
pub fn mean_squared_error3(a: &[[f32; 3]], b: &[[f32; 3]]) -> f64 {
    assert_eq!(a.len(), b.len(), "batch length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f64
        })
        .sum::<f64>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_velocities_produce_zero_rollout_error() {
        let p0 = [[0.0_f32; 3]; 4];
        let v = [[1.0_f32; 3]; 4];
        let result = position_rollout(&p0, &v, &v, 0.5, 10);
        assert!(result.errors.iter().all(|e| e.abs() < 1e-6));
        assert_eq!(result.final_error(), 0.0);
    }

    #[test]
    fn constant_velocity_bias_grows_linearly() {
        let p0 = [[0.0_f32; 3]; 8];
        let v_pred = [[1.1_f32; 3]; 8];
        let v_truth = [[1.0_f32; 3]; 8];
        let result = position_rollout(&p0, &v_pred, &v_truth, 1.0, 5);

        // Error after step k must be k*dt*|bias| (linear growth).
        for (idx, err) in result.errors.iter().enumerate() {
            let k = idx + 1;
            let expected = (k as f32 * 0.1_f32 * 3.0_f32.sqrt()) as f64;
            assert!(
                (*err - expected).abs() < 1e-6,
                "step {k}: {err} != {expected}"
            );
        }
    }

    #[test]
    fn mse_helper_matches_manual_computation() {
        let a = [[1.0_f32, 2.0, 3.0]];
        let b = [[0.0_f32, 2.0, 5.0]];
        assert!((mean_squared_error3(&a, &b) - 5.0).abs() < 1e-6);
        assert_eq!(mean_squared_error3(&[], &[]), 0.0);
    }
}
