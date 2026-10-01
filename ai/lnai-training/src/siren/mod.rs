
pub mod dataset;
pub mod loss;
pub mod trainer;

pub use dataset::{
    PrefetchBatcher, SirenDataset, SirenNorm, StarParams, StarTexturePlan,
    StratifiedStreamingBatcher, StreamingBatcher, TARGET_DIM, split_star_indices,
};
pub use loss::{compute_data_loss, compute_siren_loss, compute_siren_loss_conditioned};
