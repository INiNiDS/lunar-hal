#[cfg(feature = "siren")]
use {burn::prelude::*, lnai_models::SIREN_INPUT_DIM};

#[cfg(feature = "siren")]
use super::backend_type::B;
#[cfg(feature = "siren")]
use super::siren::SirenModel;

#[cfg(feature = "siren")]
pub fn siren_generate_texture(
    siren: &SirenModel,
    width: u32,
    height: u32,
    bp_rp: f32,
    m_g: f32,
    log_teff: f32,
) -> Vec<u8> {
    let n_bp = (bp_rp - siren.norm.bp_rp_mean) / siren.norm.bp_rp_std;
    let n_mg = (m_g - siren.norm.mg_mean) / siren.norm.mg_std;
    let n_teff = (log_teff - siren.norm.log_teff_mean) / siren.norm.log_teff_std;

    const TEXTURE_CHUNK_PIXELS: usize = 8192;
    let total = width as usize * height as usize;
    let mut pixels = Vec::with_capacity(total * 3);
    for start in (0..total).step_by(TEXTURE_CHUNK_PIXELS) {
        let count = (total - start).min(TEXTURE_CHUNK_PIXELS);
        let mut input_data = Vec::with_capacity(count * SIREN_INPUT_DIM);
        for i in start..start + count {
            let x = (i % width as usize) as f32;
            let y = (i / width as usize) as f32;
            input_data.extend_from_slice(&[
                -1.0 + 2.0 * x / width.saturating_sub(1).max(1) as f32,
                -1.0 + 2.0 * y / height.saturating_sub(1).max(1) as f32,
                n_bp,
                n_mg,
                n_teff,
            ]);
        }
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input_data, [count, SIREN_INPUT_DIM]),
            &siren.device,
        );
        let vals: Vec<f32> = siren
            .model
            .forward(input)
            .into_data()
            .to_vec()
            .expect("failed to convert SIREN texture output");
        for value in vals {
            pixels.push((value.clamp(0.0, 1.0) * 255.0) as u8);
        }
    }

    pixels
}
