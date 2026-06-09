#[cfg(feature = "pinn")]
pub mod pinn;
#[cfg(feature = "pinn")]
pub use pinn::*;

#[cfg(feature = "gnn")]
pub mod gnn;
#[cfg(feature = "gnn")]
pub use gnn::{StellarGnn, StellarGnnConfig, GcnLayer, GNN_INPUT_DIM, GNN_OUTPUT_DIM, GNN_VARIATIONAL_DIM, compute_adjacency_matrix, compute_knn_adjacency, sample_stellar_dynamics};