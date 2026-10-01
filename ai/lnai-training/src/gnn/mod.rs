
pub mod dataset;
pub mod graph;
pub mod loss;
pub mod trainer;

pub use dataset::{
    DEFAULT_KNN_K, DEFAULT_MAX_GROUP, GnnDataset, GnnNormParams, NODE_FEATURE_DIM,
    PrefetchBatchedBatcher, StarGroup, VELOCITY_DIM,
};
pub use graph::{GraphCache, SpatialIndex3D, hash_coords};
pub use loss::{
    compute_gnn_loss, compute_gnn_physics_loss, compute_gnn_total_loss, gnn_loss_scalars,
    target_centered_physics,
};
