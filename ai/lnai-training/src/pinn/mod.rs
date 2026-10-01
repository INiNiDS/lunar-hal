
pub mod dataset;
pub mod loss;
pub mod trainer;

pub use dataset::{GpuBatcher, INPUT_DIM, NormParams, StellarDataset, TARGET_DIM};
pub use loss::{compute_data_loss, compute_physics_loss, compute_pinn_loss};
