#![forbid(unsafe_code)]

pub mod api_client;
pub mod camera;
pub mod error;
pub mod scene;
pub mod sector;
pub mod snapshot;
pub mod validation;

pub use camera::{Camera, MAX_ZOOM, MIN_ZOOM, SceneCamera, SceneCameraStore};
pub use error::StellarSceneError;
pub use scene::{StellarScene, StellarSceneConfig};
pub use sector::{
    CHUNK_SIZE_PC, CHUNK_SIZE_PC as CHUNK_SIZE, INNER_EXCLUSION_PC, MAX_CACHED_CHUNKS,
    MAX_CONCURRENT_FETCHES, MIN_FETCH_OBJECTS, MIN_FETCH_RECORDS, PX_PER_PC, SectorKey,
    chunk_center,
};
pub use snapshot::StellarSceneSnapshot;
pub use validation::{
    ValidationError, ValidationResult, limits, validate_bp_rp, validate_center_x,
    validate_center_y, validate_center_z, validate_entropy, validate_g_mag, validate_pipeline,
    validate_response_star, validate_response_stars, validate_search_radius, validate_sector_key,
    validate_temperature, validate_scene, validate_scene_id, validate_scene_name,
    validate_scene_summary, validate_zoom,
};
