use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use serde::{Deserialize, Serialize};

use crate::gnn::{GcnLayer, GraphBatch};

pub const GNN_LOC_INPUT_DIM: usize = 8;

pub const GNN_LOC_MASKED_OUTPUT_DIM: usize = 9;

pub const GNN_LOC_SLOT_DIM: usize = 16;

pub const GNN_LOC_DEFAULT_MAX_SLOTS: usize = 16;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizationOutput {
    pub candidates: Vec<StarCandidate>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StarCandidate {
    pub existence_prob: f32,
    pub relative_position: [f32; 3],
    pub covariance: [f32; 6],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bp_rp: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub g_mag: Option<f32>,
}

impl StarCandidate {
    pub fn new(existence_prob: f32, relative_position: [f32; 3], covariance: [f32; 6]) -> Self {
        Self {
            existence_prob,
            relative_position,
            covariance,
            bp_rp: None,
            g_mag: None,
        }
    }

    pub fn with_photometry(mut self, bp_rp: f32, g_mag: f32) -> Self {
        self.bp_rp = Some(bp_rp);
        self.g_mag = Some(g_mag);
        self
    }

    pub fn positional_variances(&self) -> [f32; 3] {
        [self.covariance[0], self.covariance[1], self.covariance[2]]
    }
}

#[derive(Config, Debug)]
pub struct StellarGnnLocalizationConfig {
    #[config(default = 8)]
    pub input_dim: usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = 16)]
    pub max_slots: usize,
    #[config(default = 16)]
    pub slot_dim: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
    #[config(default = false)]
    pub equivariant: bool,
}

#[derive(Module, Debug)]
pub struct StellarGnnLocalization<B: Backend> {
    gcn1: GcnLayer<B>,
    gcn2: GcnLayer<B>,
    gcn3: GcnLayer<B>,
    readout_masked: Linear<B>,
    slot_expansion: Linear<B>,
    slot_dense1: Linear<B>,
    slot_dense2: Linear<B>,
    slot_out: Linear<B>,
    max_slots: usize,
    slot_dim: usize,
    hidden_dim: usize,
}

impl StellarGnnLocalizationConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> StellarGnnLocalization<B> {
        let eps = self.layer_norm_eps;
        let gcn1 = GcnLayer::new(device, self.input_dim, self.hidden_dim, eps);
        let gcn2 = GcnLayer::new(device, self.hidden_dim, self.hidden_dim, eps);
        let gcn3 = GcnLayer::new(device, self.hidden_dim, self.hidden_dim, eps);

        let readout_masked =
            LinearConfig::new(self.hidden_dim, GNN_LOC_MASKED_OUTPUT_DIM).init(device);
        let slot_expansion =
            LinearConfig::new(self.hidden_dim, self.max_slots * self.hidden_dim).init(device);
        let slot_dense1 = LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device);
        let slot_dense2 = LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device);
        let slot_out = LinearConfig::new(self.hidden_dim, self.slot_dim).init(device);

        StellarGnnLocalization {
            gcn1,
            gcn2,
            gcn3,
            readout_masked,
            slot_expansion,
            slot_dense1,
            slot_dense2,
            slot_out,
            max_slots: self.max_slots,
            slot_dim: self.slot_dim,
            hidden_dim: self.hidden_dim,
        }
    }
}

impl<B: Backend> StellarGnnLocalization<B> {
    pub fn forward_encoder(&self, nodes: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let h1 = self.gcn1.forward(nodes, adj.clone());
        let h2 = self.gcn2.forward(h1.clone(), adj.clone());
        self.gcn3.forward(h2 + h1, adj)
    }

    pub fn forward_encoder_sparse(&self, nodes: Tensor<B, 2>, graph: &GraphBatch) -> Tensor<B, 2> {
        let h1 = self.gcn1.forward_sparse(nodes, graph);
        let h2 = self.gcn2.forward_sparse(h1.clone(), graph);
        self.gcn3.forward_sparse(h2 + h1, graph)
    }

    pub fn forward_masked(&self, nodes: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.forward_encoder(nodes, adj);
        self.readout_masked.forward(h)
    }

    pub fn forward_masked_sparse(&self, nodes: Tensor<B, 2>, graph: &GraphBatch) -> Tensor<B, 2> {
        let h = self.forward_encoder_sparse(nodes, graph);
        self.readout_masked.forward(h)
    }

    pub fn forward_slots(
        &self,
        nodes: Tensor<B, 2>,
        adj: Tensor<B, 2>,
        anchor_idx: usize,
    ) -> Tensor<B, 2> {
        let h = self.forward_encoder(nodes, adj);
        self.decode_from_node_embeddings(h, anchor_idx)
    }

    pub fn forward_slots_sparse(
        &self,
        nodes: Tensor<B, 2>,
        graph: &GraphBatch,
        anchor_idx: usize,
    ) -> Tensor<B, 2> {
        let h = self.forward_encoder_sparse(nodes, graph);
        self.decode_from_node_embeddings(h, anchor_idx)
    }

    fn decode_from_node_embeddings(&self, h: Tensor<B, 2>, anchor_idx: usize) -> Tensor<B, 2> {
        let [n, _d] = h.dims();
        let _device = h.device();
        let safe_anchor = anchor_idx.min(n.saturating_sub(1));

        let anchor_slice = h
            .clone()
            .slice([safe_anchor..(safe_anchor + 1), 0..self.hidden_dim]);
        let mean_pool = h.mean_dim(0);
        let context = anchor_slice + mean_pool;


        let expanded = self.slot_expansion.forward(context);

        let slot_tokens = expanded.reshape([self.max_slots, self.hidden_dim]);

        let s1 = burn::tensor::activation::silu(self.slot_dense1.forward(slot_tokens));
        let s2 = burn::tensor::activation::silu(self.slot_dense2.forward(s1));
        self.slot_out.forward(s2)

    }

    pub fn decode_candidates(
        &self,
        slot_preds: &Tensor<B, 2>,
        radius_pc: f32,
        threshold: f32,
    ) -> LocalizationOutput {
        let [k, d] = slot_preds.dims();
        assert!(d >= 10, "slot output dimension must be at least 10");
        let data = slot_preds.clone().into_data();
        let slice: &[f32] = data.as_slice().expect("f32 data");

        let mut candidates = Vec::new();
        for i in 0..k {
            let offset = i * d;
            let raw_logit = slice[offset];
            let existence_prob = 1.0 / (1.0 + (-raw_logit).exp());

            if existence_prob >= threshold {
                let dx = slice[offset + 1] * radius_pc;
                let dy = slice[offset + 2] * radius_pc;
                let dz = slice[offset + 3] * radius_pc;

                let var_x = (slice[offset + 4].exp() * 0.1).clamp(1e-4, 100.0);
                let var_y = (slice[offset + 5].exp() * 0.1).clamp(1e-4, 100.0);
                let var_z = (slice[offset + 6].exp() * 0.1).clamp(1e-4, 100.0);
                let cov_xy = slice[offset + 7] * 0.05;
                let cov_xz = slice[offset + 8] * 0.05;
                let cov_yz = slice[offset + 9] * 0.05;

                let bp_rp = if d > 10 {
                    Some(slice[offset + 10])
                } else {
                    None
                };
                let g_mag = if d > 11 {
                    Some(slice[offset + 11])
                } else {
                    None
                };

                candidates.push(StarCandidate {
                    existence_prob,
                    relative_position: [dx, dy, dz],
                    covariance: [var_x, var_y, var_z, cov_xy, cov_xz, cov_yz],
                    bp_rp,
                    g_mag,
                });
            }
        }

        candidates.sort_by(|a, b| b.existence_prob.total_cmp(&a.existence_prob));
        LocalizationOutput { candidates }
    }
}

#[derive(Config, Debug)]
pub struct NoGraphMlpConfig {
    #[config(default = 8)]
    pub input_dim: usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = 9)]
    pub output_dim: usize,
}

#[derive(Module, Debug)]
pub struct NoGraphMlpBaseline<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    out: Linear<B>,
}

impl NoGraphMlpConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> NoGraphMlpBaseline<B> {
        let fc1 = LinearConfig::new(self.input_dim, self.hidden_dim).init(device);
        let fc2 = LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device);
        let out = LinearConfig::new(self.hidden_dim, self.output_dim).init(device);
        NoGraphMlpBaseline { fc1, fc2, out }
    }
}

impl<B: Backend> NoGraphMlpBaseline<B> {
    pub fn forward(&self, nodes: Tensor<B, 2>) -> Tensor<B, 2> {
        let h1 = burn::tensor::activation::silu(self.fc1.forward(nodes));
        let h2 = burn::tensor::activation::silu(self.fc2.forward(h1));
        self.out.forward(h2)
    }
}

pub fn density_poisson_baseline(
    radius_pc: f32,
    density_stars_per_pc3: f32,
    seed: u64,
) -> LocalizationOutput {
    let volume = (4.0 / 3.0) * std::f32::consts::PI * radius_pc.powi(3);
    let expected_count = (volume * density_stars_per_pc3).round().clamp(0.0, 32.0) as usize;

    let mut lcg = seed;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 11) % 10_000) as f32 / 10_000.0
    };

    let mut candidates = Vec::with_capacity(expected_count);
    for _ in 0..expected_count {
        let u = next();
        let r = radius_pc * u.cbrt();
        let costheta = next() * 2.0 - 1.0;
        let sintheta = (1.0 - costheta * costheta).max(0.0).sqrt();
        let phi = next() * 2.0 * std::f32::consts::PI;

        let dx = r * sintheta * phi.cos();
        let dy = r * sintheta * phi.sin();
        let dz = r * costheta;

        candidates.push(StarCandidate {
            existence_prob: 0.5,
            relative_position: [dx, dy, dz],
            covariance: [radius_pc * 0.1; 6],
            bp_rp: Some(0.8),
            g_mag: Some(15.0),
        });
    }

    LocalizationOutput { candidates }
}

pub fn knn_interpolation_baseline(visible_positions: &[[f32; 3]], k: usize) -> [f32; 3] {
    if visible_positions.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let take_k = k.min(visible_positions.len()).max(1);
    let mut sum = [0.0f32; 3];
    for pos in &visible_positions[..take_k] {
        sum[0] += pos[0];
        sum[1] += pos[1];
        sum[2] += pos[2];
    }
    let n = take_k as f32;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_output() -> LocalizationOutput {
        LocalizationOutput {
            candidates: vec![
                StarCandidate {
                    existence_prob: 0.92,
                    relative_position: [1.5, -2.0, 0.25],
                    covariance: [0.01, 0.02, 0.03, 0.001, 0.002, 0.003],
                    bp_rp: None,
                    g_mag: None,
                },
                StarCandidate {
                    existence_prob: 0.35,
                    relative_position: [-3.0, 4.0, 1.0],
                    covariance: [0.2, 0.3, 0.4, 0.0, 0.0, 0.0],
                    bp_rp: None,
                    g_mag: None,
                },
            ],
        }
    }

    #[test]
    fn localization_output_round_trips_through_json() {
        let output = sample_output();
        let json = serde_json::to_string(&output).expect("serialize LocalizationOutput");
        let back: LocalizationOutput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, output);
    }

    #[test]
    fn candidate_contract_fields_are_frozen() {
        let value = serde_json::to_value(&sample_output().candidates[0]).unwrap();
        assert_eq!(
            value.as_object().map(|o| o.len()),
            Some(3),
            "StarCandidate must expose exactly existence/position/uncertainty"
        );
        assert!(value.get("existence_prob").is_some());
        assert!(value.get("relative_position").is_some());
        assert!(value.get("covariance").is_some());
    }

    #[test]
    fn positional_variances_read_covariance_diagonal() {
        let candidate = &sample_output().candidates[0];
        assert_eq!(candidate.positional_variances(), [0.01, 0.02, 0.03]);
    }

    #[test]
    fn model_initialization_and_forward_shapes() {
        type B = burn::backend::NdArray;
        let device = Default::default();
        let cfg = StellarGnnLocalizationConfig::new();
        let model = cfg.init::<B>(&device);

        let n = 5;
        let nodes = Tensor::<B, 2>::zeros([n, GNN_LOC_INPUT_DIM], &device);
        let adj = Tensor::<B, 2>::eye(n, &device);

        let masked = model.forward_masked(nodes.clone(), adj.clone());
        assert_eq!(masked.dims(), [n, GNN_LOC_MASKED_OUTPUT_DIM]);

        let slots = model.forward_slots(nodes, adj, 0);
        assert_eq!(slots.dims(), [cfg.max_slots, cfg.slot_dim]);

        let decoded = model.decode_candidates(&slots, 25.0, 0.0);
        assert_eq!(decoded.candidates.len(), cfg.max_slots);
    }

    #[test]
    fn no_graph_mlp_baseline_forward_shape() {
        type B = burn::backend::NdArray;
        let device = Default::default();
        let cfg = NoGraphMlpConfig::new();
        let baseline = cfg.init::<B>(&device);

        let nodes = Tensor::<B, 2>::zeros([4, GNN_LOC_INPUT_DIM], &device);
        let out = baseline.forward(nodes);
        assert_eq!(out.dims(), [4, 9]);
    }

    #[test]
    fn density_poisson_baseline_deterministic() {
        let b1 = density_poisson_baseline(20.0, 0.001, 42);
        let b2 = density_poisson_baseline(20.0, 0.001, 42);
        assert_eq!(b1, b2);
    }
}
