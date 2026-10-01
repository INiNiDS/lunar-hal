
use super::rollout::mean_squared_error3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicsMetrics {
    pub oracle_mse: f64,
    pub chained_mse: f64,
    pub baseline_mse: f64,
    pub chained_skill: f64,
}

impl KinematicsMetrics {
    pub fn evaluate(chained: &[[f32; 3]], truth: &[[f32; 3]], baseline: &[[f32; 3]]) -> Self {
        let oracle_mse = mean_squared_error3(truth, truth);
        let chained_mse = mean_squared_error3(chained, truth);
        let baseline_mse = mean_squared_error3(baseline, truth);
        let chained_skill = if baseline_mse > 0.0 {
            1.0 - chained_mse / baseline_mse
        } else {
            0.0
        };
        Self {
            oracle_mse,
            chained_mse,
            baseline_mse,
            chained_skill,
        }
    }

    pub fn ordering_is_sane(&self) -> bool {
        self.oracle_mse <= self.chained_mse && self.chained_mse < f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_chain_scores_full_skill() {
        let truth = [[1.0_f32; 3]; 4];
        let metrics = KinematicsMetrics::evaluate(&truth, &truth, &[[0.0_f32; 3]; 4]);
        assert_eq!(metrics.oracle_mse, 0.0);
        assert_eq!(metrics.chained_mse, 0.0);
        assert_eq!(metrics.chained_skill, 1.0);
        assert!(metrics.ordering_is_sane());
    }

    #[test]
    fn mean_predictor_baseline_has_zero_skill_against_itself() {
        let truth = [
            [2.0_f32, -1.0, 0.5],
            [3.0, 0.0, 1.5],
            [1.0, -2.0, -0.5],
            [4.0, 1.0, 2.5],
        ];
        let mean = [
            truth.iter().map(|v| v[0]).sum::<f32>() / 4.0,
            truth.iter().map(|v| v[1]).sum::<f32>() / 4.0,
            truth.iter().map(|v| v[2]).sum::<f32>() / 4.0,
        ];
        let baseline = vec![mean; 4];

        let at_baseline = KinematicsMetrics::evaluate(&baseline, &truth, &baseline);
        assert!(
            (at_baseline.chained_skill - 0.0).abs() < 1e-9,
            "predicting exactly the baseline scores zero skill"
        );

        let worse = KinematicsMetrics::evaluate(&[[9.0_f32; 3]; 4], &truth, &baseline);
        assert!(
            worse.chained_skill < 0.0,
            "worse-than-baseline must have negative skill"
        );
        assert!(worse.ordering_is_sane());
    }

    #[test]
    fn zero_variance_batch_disables_skill() {
        let same = [[1.0_f32; 3]; 3];
        let metrics = KinematicsMetrics::evaluate(&same, &same, &same);
        assert_eq!(metrics.baseline_mse, 0.0);
        assert_eq!(metrics.chained_skill, 0.0);
    }
}
