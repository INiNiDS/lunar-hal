use anyhow::Result;
use lunar_structures::{
    LocalizationRequest, LocalizedStarWithPhysics, PinnResponse, PipelineLocalizePhysicsRequest,
    PipelineLocalizePhysicsResponse,
};

use super::localization::predict_localization_neighbors;
use super::pinn_queue::infer_pinn_batch_async;
use super::types::PinnInputs;

pub async fn predict_localize_physics_pipeline(
    req: &PipelineLocalizePhysicsRequest,
) -> Result<PipelineLocalizePhysicsResponse> {
    let loc_req = LocalizationRequest {
        anchor_x: req.anchor_x,
        anchor_y: req.anchor_y,
        anchor_z: req.anchor_z,
        radius_pc: req.radius_pc,
        max_slots: req.max_slots,
        seed: req.seed,
        version: req.version.clone(),
        visible_neighbors: req.visible_neighbors.clone(),
    };
    let loc_res = predict_localization_neighbors(&loc_req).await;

    let mut pinn_inputs = Vec::with_capacity(loc_res.candidates.len());
    let mut abs_positions = Vec::with_capacity(loc_res.candidates.len());

    for cand in &loc_res.candidates {
        let abs_pos = [
            req.anchor_x + cand.relative_position[0],
            req.anchor_y + cand.relative_position[1],
            req.anchor_z + cand.relative_position[2],
        ];
        abs_positions.push(abs_pos);
        pinn_inputs.push(PinnInputs {
            position: abs_pos,
            bp_rp: cand.bp_rp.unwrap_or(0.8),
            g_mag: cand.g_mag.unwrap_or(15.0),
        });
    }

    let pinn_outputs = infer_pinn_batch_async(pinn_inputs).await?;

    let mut stars = Vec::with_capacity(loc_res.candidates.len());
    for (i, cand) in loc_res.candidates.into_iter().enumerate() {
        let row = pinn_outputs[i];
        let physics = Some(PinnResponse {
            temperature_k: row[0],
            radius_solar: row[1],
            mass_solar: row[2],
            luminosity_solar: row[3],
        });
        stars.push(LocalizedStarWithPhysics {
            candidate: cand,
            absolute_position: abs_positions[i],
            physics,
        });
    }

    Ok(PipelineLocalizePhysicsResponse {
        stars,
        anchor_position: [req.anchor_x, req.anchor_y, req.anchor_z],
        radius_pc: req.radius_pc,
        version: loc_res.version,
        seed: loc_res.seed,
    })
}
