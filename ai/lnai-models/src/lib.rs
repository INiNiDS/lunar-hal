#[cfg(feature = "pinn")]
pub mod pinn;
#[cfg(feature = "pinn")]
pub use pinn::*;

#[cfg(feature = "gnn")]
pub mod gnn;
#[cfg(feature = "gnn")]
pub use gnn::{
    DeterministicGnnHead, GNN_INPUT_DIM, GNN_OUTPUT_DIM, GNN_VARIATIONAL_DIM, GcnLayer,
    GnnHeadKind, GraphBatch, KinematicsOutput, StellarGnn, StellarGnnConfig, VariationalGnnHead,
    compute_adjacency_matrix, compute_knn_adjacency, compute_sparse_knn_graph,
    sample_stellar_dynamics, split_mean_logvar, variational_kl,
};

#[cfg(feature = "siren")]
pub mod siren;
#[cfg(feature = "siren")]
pub use siren::{
    SIREN_HIDDEN_DIM, SIREN_INPUT_DIM, SIREN_OUTPUT_DIM, SIREN_W0, StellarSiren, StellarSirenConfig,
};

#[cfg(feature = "localization")]
pub mod localization;
#[cfg(feature = "localization")]
pub use localization::{
    GNN_LOC_DEFAULT_MAX_SLOTS, GNN_LOC_INPUT_DIM, GNN_LOC_MASKED_OUTPUT_DIM, GNN_LOC_SLOT_DIM,
    LocalizationOutput, NoGraphMlpBaseline, NoGraphMlpConfig, StarCandidate,
    StellarGnnLocalization, StellarGnnLocalizationConfig, density_poisson_baseline,
    knn_interpolation_baseline,
};
