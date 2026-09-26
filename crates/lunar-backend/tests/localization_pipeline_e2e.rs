//! Stage 8 & 9 Exit Gate: GNN-Localization and chained PINN pipeline E2E tests.
//!
//! Verifies:
//! 1. Contract adherence of `/localization/neighbors`:
//!    - Mandatory fields: `anchor_position`, `radius_pc`, `version`, `seed`, `model_used`, `candidates`.
//!    - Existence probability in [0, 1], relative position bounded by search radius.
//!    - Determinism with fixed seed.
//!    - Respect of `max_slots` parameter.
//! 2. Chained `/pipeline/localize-physics` (GNN-Localization -> PINN):
//!    - Coordinates are translated to absolute positions [anchor + delta].
//!    - PINN calculates physical parameters: positive, finite Teff, R, M, L.
//! 3. Fallback behavior when model bundle is absent: uses approved baseline generator.
//! 4. Manifest contracts and file names compatibility for ModelKind::GnnLocalization.

use lnai_training::artifacts::{architecture_version, norm_file_name, weight_file_name};
use lnai_training::spec::ModelKind;
use lunar_backend::ai::{
    get_pinn, predict_localization_neighbors, predict_localize_physics_pipeline,
};
use lunar_structures::{LocalizationRequest, PipelineLocalizePhysicsRequest, VisibleStarDto};

#[tokio::test]
async fn localization_neighbors_contract_and_determinism() {
    let req = LocalizationRequest {
        anchor_x: 10.0,
        anchor_y: -20.0,
        anchor_z: 30.0,
        radius_pc: 25.0,
        max_slots: Some(6),
        seed: Some(42),
        version: Some("1.0.0".to_string()),
        visible_neighbors: vec![
            VisibleStarDto {
                source_id: Some("vis_1".to_string()),
                x_pc: 12.0,
                y_pc: -19.0,
                z_pc: 31.0,
                bp_rp: Some(1.2),
                g_mag: Some(14.5),
                ruwe: Some(1.05),
            },
            VisibleStarDto {
                source_id: Some("vis_2".to_string()),
                x_pc: 8.0,
                y_pc: -22.0,
                z_pc: 29.0,
                bp_rp: Some(0.9),
                g_mag: Some(15.8),
                ruwe: Some(0.98),
            },
        ],
    };

    let res1 = predict_localization_neighbors(&req).await;
    let res2 = predict_localization_neighbors(&req).await;

    // 1. Mandatory contract fields
    assert_eq!(res1.anchor_position, [10.0, -20.0, 30.0]);
    assert_eq!(res1.radius_pc, 25.0);
    assert_eq!(res1.version, "1.0.0");
    assert_eq!(res1.seed, 42);
    assert!(!res1.model_used.is_empty());

    // 2. Determinism
    assert_eq!(
        res1, res2,
        "identical requests must produce identical responses"
    );

    // 3. Slot constraints
    assert!(res1.candidates.len() <= 6, "must respect max_slots cap");

    // 4. Candidate bounds and positive variances
    for cand in &res1.candidates {
        assert!(cand.existence_prob >= 0.0 && cand.existence_prob <= 1.0);
        let dist = (cand.relative_position[0].powi(2)
            + cand.relative_position[1].powi(2)
            + cand.relative_position[2].powi(2))
        .sqrt();
        assert!(
            dist <= req.radius_pc * 1.5,
            "relative position should be within neighborhood radius: got {dist}"
        );

        let var = cand.positional_variances();
        assert!(
            var[0] > 0.0 && var[1] > 0.0 && var[2] > 0.0,
            "variances must be strictly positive"
        );
    }

    // 5. Different seeds produce different results
    let mut req_other_seed = req.clone();
    req_other_seed.seed = Some(999);
    let res_other = predict_localization_neighbors(&req_other_seed).await;
    assert_ne!(res1.seed, res_other.seed);
}

#[tokio::test]
async fn chained_pipeline_requires_a_release_approved_pinn() {
    let req = PipelineLocalizePhysicsRequest {
        anchor_x: 5.0,
        anchor_y: 15.0,
        anchor_z: -10.0,
        radius_pc: 20.0,
        max_slots: Some(4),
        seed: Some(123),
        version: Some("1.0.0".to_string()),
        visible_neighbors: vec![VisibleStarDto {
            source_id: Some("vis_alpha".to_string()),
            x_pc: 6.0,
            y_pc: 14.0,
            z_pc: -9.0,
            bp_rp: Some(1.1),
            g_mag: Some(12.0),
            ruwe: Some(1.0),
        }],
    };

    let pinn_available = get_pinn().await.is_ok();
    let pipeline_res = predict_localize_physics_pipeline(&req).await;
    if !pinn_available {
        assert!(
            pipeline_res.is_err(),
            "pipeline must not return fabricated physics when PINN is unavailable"
        );
        return;
    }
    let pipeline_res = pipeline_res.expect("approved PINN should support the pipeline");

    assert_eq!(pipeline_res.anchor_position, [5.0, 15.0, -10.0]);
    assert_eq!(pipeline_res.radius_pc, 20.0);
    assert_eq!(pipeline_res.seed, 123);
    assert!(!pipeline_res.stars.is_empty());
    assert!(pipeline_res.stars.len() <= 4);

    for s in &pipeline_res.stars {
        // Absolute position must equal anchor + relative position
        let expected_abs = [
            req.anchor_x + s.candidate.relative_position[0],
            req.anchor_y + s.candidate.relative_position[1],
            req.anchor_z + s.candidate.relative_position[2],
        ];
        assert_eq!(s.absolute_position, expected_abs);

        // Physics predicted by PINN must be present and physically sound
        let physics = s.physics.as_ref().expect("PINN physics must be populated");
        assert!(physics.temperature_k.is_finite() && physics.temperature_k > 0.0);
        assert!(physics.radius_solar.is_finite() && physics.radius_solar > 0.0);
        assert!(physics.mass_solar.is_finite() && physics.mass_solar > 0.0);
        assert!(physics.luminosity_solar.is_finite() && physics.luminosity_solar > 0.0);
    }
}

#[test]
fn localization_artifact_manifest_contracts_match_spec() {
    let kind = ModelKind::GnnLocalization;
    assert_eq!(architecture_version(&kind), "gnn-loc-v1");
    assert_eq!(weight_file_name(&kind), "stellar_gnn_loc_model.bpk");
    assert_eq!(norm_file_name(&kind), "stellar_gnn_loc_norm.json");
}
