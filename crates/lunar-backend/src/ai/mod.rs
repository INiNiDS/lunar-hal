pub mod backend_type;
pub mod gnn;
pub mod gnn_decode;
pub mod gnn_infer;
pub mod localization;
pub mod lore;
pub mod lore_desc;
pub mod metadata;
pub mod metadata_names;
pub mod pipeline;
pub mod pinn;
pub mod pinn_queue;
pub mod rng;
pub mod serving;
#[cfg(feature = "siren")]
pub mod siren;
#[cfg(feature = "siren")]
pub mod siren_texture;
pub mod types;
pub mod warmup;

#[cfg(test)]
mod tests_manifest;
#[cfg(test)]
mod tests_queue_gnn;

pub use gnn::{GnnModel, GnnNorm, get_gnn};
pub use gnn_infer::gnn_infer;
pub use localization::{LocalizationModel, get_localization, predict_localization_neighbors};
pub use lore::{LoreCache, LoreEntry, get_lore_cache};
pub use metadata::{
    classify_star, generate_hybrid_metadata, generate_random_inputs, generate_stochastic_metadata,
    is_rare_star,
};
pub use pipeline::predict_localize_physics_pipeline;
pub use pinn::{PinnModel, get_pinn, loaded_pinn_hashes, pinn_infer, pinn_infer_batch};
pub use pinn_queue::infer_pinn_batch_async;
pub use rng::SimpleRng;
pub use serving::{
    ReloadReport, loaded_model_identities, registry_snapshot, reload_models,
};
#[cfg(feature = "siren")]
pub use siren::{SirenInputs, SirenModel, SirenNorm, get_siren, siren_infer_point};
#[cfg(feature = "siren")]
pub use siren_texture::siren_generate_texture;
pub use types::{
    LoadedModelIdentity, PinnInputs, RandomStellarInputs, StarFeatures, StellarNorm,
    apparent_g_for_member,
};
pub use warmup::warmup_models;
