use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Debug)]
pub struct StellarNorm {
    pub x_mean: f32,
    pub x_std: f32,
    pub y_mean: f32,
    pub y_std: f32,
    pub z_mean: f32,
    pub z_std: f32,
    pub bp_rp_mean: f32,
    pub bp_rp_std: f32,
    pub mg_mean: f32,
    pub mg_std: f32,
    pub log_teff_mean: f32,
    pub log_teff_std: f32,
    pub log_rad_mean: f32,
    pub log_rad_std: f32,
    pub log_mass_mean: f32,
    pub log_mass_std: f32,
    pub log_lum_mean: f32,
    pub log_lum_std: f32,
}

#[derive(Clone, Serialize, Debug)]
pub struct LoadedModelIdentity {
    pub kind: &'static str,
    pub model_hash: String,
    pub norm_hash: String,
}

#[derive(Clone, Copy, Debug)]
pub struct PinnInputs {
    pub position: [f32; 3],
    pub bp_rp: f32,
    pub g_mag: f32,
}

/// Apparent G magnitude of a member at `position` assuming it shares the
/// sector's absolute magnitude `mg_center` (distance modulus, parsecs).
/// Falls back to `mg_center + 5*log10(0.1) - 5` near the origin, mirroring
/// the single-star guard below.
pub fn apparent_g_for_member(position: [f32; 3], mg_center: f32, g_fallback: f32) -> f32 {
    let d =
        (position[0] * position[0] + position[1] * position[1] + position[2] * position[2]).sqrt();
    if d < 0.1 {
        g_fallback
    } else {
        mg_center + 5.0 * d.log10() - 5.0
    }
}

#[derive(Clone, Debug)]
pub struct StarFeatures {
    pub coords: [f32; 3],
    pub log_teff: f32,
    pub log_rad: f32,
    pub log_mass: f32,
    pub log_lum: f32,
    pub mg: f32,
}

pub struct RandomStellarInputs {
    pub x_pc: f32,
    pub y_pc: f32,
    pub z_pc: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
}
