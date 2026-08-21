use serde::{Deserialize, Serialize};

/// Output of the GNN-Localization model.
/// Contains a set of predicted candidates for missing/hidden neighbors.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizationOutput {
    /// Variable-cardinality set of predicted star candidates
    pub candidates: Vec<StarCandidate>,
}

/// A single predicted star candidate in the localized neighborhood.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StarCandidate {
    /// Probability [0.0, 1.0] that this candidate actually exists
    pub existence_prob: f32,
    /// Relative 3D position [dx, dy, dz] from the anchor star (in parsecs)
    pub relative_position: [f32; 3],
    /// Lower triangle of the 3x3 covariance matrix representing positional uncertainty.
    /// Order: [xx, yy, zz, xy, xz, yz]
    pub covariance: [f32; 6],
}

impl StarCandidate {
    /// Extracts the diagonal variances [var_x, var_y, var_z] for quick UI rendering
    pub fn positional_variances(&self) -> [f32; 3] {
        [self.covariance[0], self.covariance[1], self.covariance[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_output() -> LocalizationOutput {
        LocalizationOutput {
            candidates: vec![
                StarCandidate {
                    existence_prob: 0.92,
                    relative_position: [1.5, -2.0, 0.25],
                    covariance: [0.01, 0.02, 0.03, 0.001, 0.002, 0.003],
                },
                StarCandidate {
                    existence_prob: 0.35,
                    relative_position: [-3.0, 4.0, 1.0],
                    covariance: [0.2, 0.3, 0.4, 0.0, 0.0, 0.0],
                },
            ],
        }
    }

    #[test]
    fn localization_output_round_trips_through_json() {
        let output = sample_output();
        let json = serde_json::to_string(&output).expect("serialize LocalizationOutput");
        let back: LocalizationOutput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, output);
    }

    #[test]
    fn candidate_contract_fields_are_frozen() {
        let value = serde_json::to_value(&sample_output().candidates[0]).unwrap();
        assert_eq!(
            value.as_object().map(|o| o.len()),
            Some(3),
            "StarCandidate must expose exactly existence/position/uncertainty"
        );
        assert!(value.get("existence_prob").is_some());
        assert!(value.get("relative_position").is_some());
        assert!(value.get("covariance").is_some());
    }

    #[test]
    fn positional_variances_read_covariance_diagonal() {
        let candidate = &sample_output().candidates[0];
        assert_eq!(candidate.positional_variances(), [0.01, 0.02, 0.03]);
    }
}
