#[cfg(feature = "siren")]
use crate::ai::{get_siren, siren_generate_texture};
use crate::gallery::GalleryStore;
use crate::scenes::SceneStore;
use lunar_structures::SceneEvent;
use std::sync::Arc;
use tokio::sync::broadcast;

pub mod ai;
pub mod gallery;
pub mod scenes;
pub mod version;

pub const MAX_TEXTURE_PIXELS: u64 = 1024 * 1024;

pub fn texture_dimensions_valid(width: u32, height: u32) -> bool {
    width > 0 && height > 0 && (width as u64 * height as u64) <= MAX_TEXTURE_PIXELS
}

#[derive(Clone)]
pub struct AppState {
    pub scenes: Arc<SceneStore>,
    pub gallery: Arc<GalleryStore>,
    pub scene_events: broadcast::Sender<SceneEvent>,
}

pub async fn generate_siren_pixels(
    width: u32,
    height: u32,
    bp_rp: f32,
    m_g: f32,
    log_teff: f32,
) -> Option<Vec<u8>> {
    if !texture_dimensions_valid(width, height) {
        return None;
    }
    #[cfg(feature = "siren")]
    {
        let siren = get_siren().await?;
        tokio::task::spawn_blocking(move || {
            siren_generate_texture(&siren, width, height, bp_rp, m_g, log_teff)
        })
        .await
        .ok()
    }
    #[cfg(not(feature = "siren"))]
    {
        let _ = (width, height, bp_rp, m_g, log_teff);
        None
    }
}

#[cfg(test)]
mod texture_tests {
    use super::*;

    #[test]
    fn texture_dimensions_are_bounded_before_allocation() {
        assert!(!texture_dimensions_valid(0, 256));
        assert!(!texture_dimensions_valid(u32::MAX, u32::MAX));
        assert!(texture_dimensions_valid(1024, 1024));
        assert!(!texture_dimensions_valid(1025, 1024));
    }
}
