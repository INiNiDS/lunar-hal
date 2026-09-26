pub mod dataset;
pub mod evaluation;
pub mod loss;
pub mod trainer;

pub use dataset::{
    LeakageError, LocalStar, Neighborhood, audit_leakage, build_visible_graph_batch,
    build_visible_node_features, generate_synthetic_stars, mask_neighborhood,
};
pub use evaluation::{
    MaskedEvaluationReport, MaskedSampleEval, NeighborsEvaluationReport, evaluate_masked_set,
    evaluate_neighborhood_set, evaluate_neighbors_dataset,
};
pub use loss::{
    SetLossBreakdown, chamfer_distance_3d, compute_set_loss, gaussian_nll_3d, huber_loss_1d,
    huber_loss_3d, hungarian_match,
};
pub use trainer::{LocalizationNorm, run_train_masked, run_train_neighbors};
