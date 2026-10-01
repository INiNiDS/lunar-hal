use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct GraphBatch {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub row_ptr: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub edge_weights: Vec<f32>,
}

impl GraphBatch {
    pub fn empty() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            row_ptr: vec![0],
            col_indices: Vec::new(),
            edge_weights: Vec::new(),
        }
    }

    pub fn new(
        num_nodes: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        edge_weights: Vec<f32>,
    ) -> Self {
        assert_eq!(
            row_ptr.len(),
            num_nodes + 1,
            "row_ptr length must be num_nodes + 1"
        );
        let num_edges = col_indices.len();
        assert_eq!(
            edge_weights.len(),
            num_edges,
            "edge_weights length must match col_indices"
        );
        assert_eq!(
            row_ptr[num_nodes], num_edges,
            "row_ptr[num_nodes] must equal num_edges"
        );
        Self {
            num_nodes,
            num_edges,
            row_ptr,
            col_indices,
            edge_weights,
        }
    }

    pub fn to_dense_adjacency<B: Backend>(&self, device: &Device<B>) -> Tensor<B, 2> {
        let n = self.num_nodes;
        let mut dense = vec![0.0f32; n * n];
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for edge_idx in start..end {
                let j = self.col_indices[edge_idx];
                let w = self.edge_weights[edge_idx];
                dense[i * n + j] = w;
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(dense, [n, n]), device)
    }

    pub fn propagate<B: Backend>(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n, d] = features.dims();
        assert_eq!(
            n, self.num_nodes,
            "feature row count {n} must match graph node count {}",
            self.num_nodes
        );
        if n == 0 {
            return features;
        }
        let device = features.device();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Tensor::<B, 2>::zeros([1, d], &device);
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for edge_idx in start..end {
                let j = self.col_indices[edge_idx];
                let w = self.edge_weights[edge_idx];
                let source = features.clone().slice([j..j + 1, 0..d]);
                row = row + source.mul_scalar(w);
            }
            rows.push(row);
        }
        Tensor::cat(rows, 0)
    }
}

#[derive(Module, Debug)]
pub struct GcnLayer<B: Backend> {
    linear: Linear<B>,
    norm: LayerNorm<B>,
}

impl<B: Backend> GcnLayer<B> {
    pub fn new(device: &Device<B>, input_dim: usize, output_dim: usize, eps: f64) -> Self {
        let linear = LinearConfig::new(input_dim, output_dim).init(device);
        let norm = LayerNormConfig::new(output_dim)
            .with_epsilon(eps)
            .init(device);
        Self { linear, norm }
    }

    pub fn forward(&self, nodes: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let projected = self.linear.forward(nodes);
        let propagated = adj.matmul(projected);
        burn::tensor::activation::silu(self.norm.forward(propagated))
    }

    pub fn forward_sparse(&self, nodes: Tensor<B, 2>, graph: &GraphBatch) -> Tensor<B, 2> {
        let projected = self.linear.forward(nodes);
        let propagated = graph.propagate(projected);
        burn::tensor::activation::silu(self.norm.forward(propagated))
    }
}

#[derive(Config, Debug)]
pub struct StellarGnnConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

#[derive(Module, Debug)]
pub struct StellarGnn<B: Backend> {
    gcn1: GcnLayer<B>,
    gcn2: GcnLayer<B>,
    gcn3: GcnLayer<B>,
    readout: Linear<B>,
}

impl StellarGnnConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> StellarGnn<B> {
        let eps = self.layer_norm_eps;
        let gcn1 = GcnLayer::new(device, self.input_dim, self.hidden_dim, eps);
        let gcn2 = GcnLayer::new(device, self.hidden_dim, self.hidden_dim, eps);
        let gcn3 = GcnLayer::new(device, self.hidden_dim, self.hidden_dim, eps);
        let readout = LinearConfig::new(self.hidden_dim, self.output_dim).init(device);

        StellarGnn {
            gcn1,
            gcn2,
            gcn3,
            readout,
        }
    }
}

impl<B: Backend> StellarGnn<B> {
    pub fn forward(&self, nodes: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let h1 = self.gcn1.forward(nodes, adj.clone());
        let h2 = self.gcn2.forward(h1.clone(), adj.clone());
        let h3 = self.gcn3.forward(h2.clone() + h1, adj);
        self.readout.forward(h3)
    }

    pub fn forward_sparse(&self, nodes: Tensor<B, 2>, graph: &GraphBatch) -> Tensor<B, 2> {
        let h1 = self.gcn1.forward_sparse(nodes, graph);
        let h2 = self.gcn2.forward_sparse(h1.clone(), graph);
        let h3 = self.gcn3.forward_sparse(h2.clone() + h1, graph);
        self.readout.forward(h3)
    }
}

pub const GNN_INPUT_DIM: usize = 8;
pub const GNN_OUTPUT_DIM: usize = 3;
pub const GNN_VARIATIONAL_DIM: usize = 6;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GnnHeadKind {
    Deterministic,
    Variational,
}

impl GnnHeadKind {
    pub fn from_output_dim(width: usize) -> Option<Self> {
        match width {
            GNN_OUTPUT_DIM => Some(GnnHeadKind::Deterministic),
            GNN_VARIATIONAL_DIM => Some(GnnHeadKind::Variational),
            _ => None,
        }
    }

    pub fn output_width(&self) -> usize {
        match self {
            GnnHeadKind::Deterministic => GNN_OUTPUT_DIM,
            GnnHeadKind::Variational => GNN_VARIATIONAL_DIM,
        }
    }
}

#[derive(Module, Debug)]
pub struct DeterministicGnnHead<B: Backend> {
    pub readout: Linear<B>,
}

impl<B: Backend> DeterministicGnnHead<B> {
    pub fn new(device: &Device<B>, hidden_dim: usize) -> Self {
        Self {
            readout: LinearConfig::new(hidden_dim, GNN_OUTPUT_DIM).init(device),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.readout.forward(hidden)
    }
}

#[derive(Module, Debug)]
pub struct VariationalGnnHead<B: Backend> {
    pub mean_head: Linear<B>,
    pub logvar_head: Linear<B>,
}

impl<B: Backend> VariationalGnnHead<B> {
    pub fn new(device: &Device<B>, hidden_dim: usize) -> Self {
        Self {
            mean_head: LinearConfig::new(hidden_dim, GNN_OUTPUT_DIM).init(device),
            logvar_head: LinearConfig::new(hidden_dim, GNN_OUTPUT_DIM).init(device),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        (
            self.mean_head.forward(hidden.clone()),
            self.logvar_head.forward(hidden),
        )
    }

    pub fn forward_cat(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mean, logvar) = self.forward(hidden);
        Tensor::cat(vec![mean, logvar], 1)
    }
}

pub fn split_mean_logvar<B: Backend>(
    readout: Tensor<B, 2>,
    head: GnnHeadKind,
) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
    let [n, _] = readout.dims();
    let mean = readout.clone().slice([0..n, 0..GNN_OUTPUT_DIM]);
    let logvar = match head {
        GnnHeadKind::Deterministic => None,
        GnnHeadKind::Variational => {
            Some(readout.slice([0..n, GNN_OUTPUT_DIM..GNN_VARIATIONAL_DIM]))
        }
    };
    (mean, logvar)
}

pub fn variational_kl<B: Backend>(mean: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 1> {
    let one = Tensor::<B, 2>::ones_like(&logvar);
    let kl = one + logvar.clone() - mean.square() - logvar.exp();
    let n = kl.dims()[0] as f32 * 3.0;
    kl.sum().mul_scalar(-0.5).div_scalar(n)
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct KinematicsOutput {
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
}

impl KinematicsOutput {
    pub fn from_row(row: &[f32]) -> Option<Self> {
        match row {
            [vx, vy, vz] => Some(Self {
                vx: *vx,
                vy: *vy,
                vz: *vz,
            }),
            _ => None,
        }
    }

    pub fn as_components(&self) -> [f32; 3] {
        [self.vx, self.vy, self.vz]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinematics_output_round_trips_through_json() {
        let output = KinematicsOutput {
            vx: 12.5,
            vy: -3.25,
            vz: 40.0,
        };
        let json = serde_json::to_string(&output).expect("serialize KinematicsOutput");
        let back: KinematicsOutput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, output);
    }

    #[test]
    fn kinematics_contract_fields_are_vx_vy_vz() {
        let value = serde_json::to_value(KinematicsOutput {
            vx: 1.0,
            vy: 2.0,
            vz: 3.0,
        })
        .unwrap();
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, vec!["vx", "vy", "vz"]);
    }

    #[test]
    fn from_row_rejects_wrong_widths_and_maps_components() {
        assert!(KinematicsOutput::from_row(&[1.0]).is_none());
        assert!(KinematicsOutput::from_row(&[1.0, 2.0, 3.0, 4.0]).is_none());
        let output = KinematicsOutput::from_row(&[9.0, 8.0, 7.0]).unwrap();
        assert_eq!(output.as_components(), [9.0, 8.0, 7.0]);
    }

    #[test]
    fn head_contract_maps_widths() {
        assert_eq!(
            GnnHeadKind::from_output_dim(3),
            Some(GnnHeadKind::Deterministic)
        );
        assert_eq!(
            GnnHeadKind::from_output_dim(6),
            Some(GnnHeadKind::Variational)
        );
        assert_eq!(GnnHeadKind::from_output_dim(0), None);
        assert_eq!(GnnHeadKind::from_output_dim(4), None);
        assert_eq!(GnnHeadKind::Deterministic.output_width(), GNN_OUTPUT_DIM);
        assert_eq!(GnnHeadKind::Variational.output_width(), GNN_VARIATIONAL_DIM);
    }

    #[test]
    fn sparse_graph_propagation_preserves_autodiff_gradients() {
        use burn::backend::Autodiff;
        use burn::backend::NdArray;

        type Train = Autodiff<NdArray<f32>>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let graph = GraphBatch::new(
            3,
            vec![0, 2, 4, 5],
            vec![0, 1, 1, 2, 2],
            vec![0.5, 0.5, 0.5, 0.5, 1.0],
        );
        let nodes = Tensor::<Train, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]),
            &device,
        )
        .require_grad();
        let propagated = graph.propagate(nodes.clone());
        let loss = propagated.powf_scalar(2.0).mean();
        let gradients = loss.backward();
        let node_gradients: Vec<f32> = nodes
            .grad(&gradients)
            .expect("sparse aggregation must preserve input gradients")
            .into_data()
            .to_vec()
            .unwrap();
        assert!(node_gradients.iter().all(|gradient| gradient.is_finite()));
        assert!(node_gradients.iter().any(|gradient| gradient.abs() > 1e-6));
    }
}

pub fn compute_adjacency_matrix(coords: &[[f32; 3]]) -> Vec<Vec<f32>> {
    let n = coords.len();
    let mut adj = vec![vec![0.0; n]; n];

    for i in 0..n {
        let mut degree_sum = 0.0;
        for j in 0..n {
            if i == j {
                adj[i][j] = 1.0;
            } else {
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let dist_sq = dx * dx + dy * dy + dz * dz + 1e-5;
                adj[i][j] = 1.0 / dist_sq;
            }
            degree_sum += adj[i][j];
        }

        for j in adj[i].iter_mut().take(n) {
            *j /= degree_sum;
        }
    }
    adj
}

pub fn sample_stellar_dynamics<B: Backend>(
    gnn_output: Tensor<B, 2>,
    temperature: f32,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let [num_stars, dims] = gnn_output.dims();
    let head = if dims >= GNN_VARIATIONAL_DIM {
        GnnHeadKind::Variational
    } else {
        GnnHeadKind::Deterministic
    };
    let (mean, logvar) = split_mean_logvar(gnn_output, head);

    if temperature <= 0.0 || head == GnnHeadKind::Deterministic {
        return mean;
    }

    let std = match logvar {
        Some(logvar) => log_var_to_std(logvar),
        None => return mean,
    };

    let epsilon = Tensor::<B, 2>::random(
        [num_stars, 3],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    let scaled_noise = epsilon.mul(std).mul_scalar(temperature as f64);

    mean.add(scaled_noise)
}

fn log_var_to_std<B: Backend>(logvar: Tensor<B, 2>) -> Tensor<B, 2> {
    logvar.mul_scalar(0.5_f64).exp()
}

pub fn compute_knn_adjacency(coords: &[[f32; 3]], k: usize) -> Vec<Vec<f32>> {
    let n = coords.len();
    let k = k.min(n - 1).max(1);
    let mut adj = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        let mut dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                (j, dx * dx + dy * dy + dz * dz)
            })
            .collect();

        let effective_k = k.min(dists.len());
        if effective_k > 0 && effective_k < dists.len() {
            dists.select_nth_unstable_by(effective_k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists[..effective_k]
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        adj[i][i] = 1.0;
        let mut degree_sum = 1.0f32;
        for &(j, dist_sq) in &dists[..effective_k] {
            let w = 1.0 / (dist_sq + 1e-5);
            adj[i][j] = w;
            degree_sum += w;
        }

        for j in adj[i].iter_mut().take(n) {
            *j /= degree_sum;
        }
    }
    adj
}

pub fn compute_sparse_knn_graph(coords: &[[f32; 3]], k: usize) -> GraphBatch {
    let n = coords.len();
    if n == 0 {
        return GraphBatch::empty();
    }
    let k = k.min(n - 1).max(1);
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(n * (k + 1));
    let mut edge_weights = Vec::with_capacity(n * (k + 1));

    row_ptr.push(0);

    for i in 0..n {
        let mut dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                (j, dx * dx + dy * dy + dz * dz)
            })
            .collect();

        let effective_k = k.min(dists.len());
        if effective_k > 0 && effective_k < dists.len() {
            dists.select_nth_unstable_by(effective_k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists[..effective_k]
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        let mut degree_sum = 1.0f32;
        let mut neighbors = Vec::with_capacity(effective_k + 1);
        neighbors.push((i, 1.0f32));

        for &(j, dist_sq) in &dists[..effective_k] {
            let w = 1.0 / (dist_sq + 1e-5);
            neighbors.push((j, w));
            degree_sum += w;
        }

        neighbors.sort_by_key(|&(col, _)| col);

        for (col, w) in neighbors {
            col_indices.push(col);
            edge_weights.push(w / degree_sum);
        }
        row_ptr.push(col_indices.len());
    }

    GraphBatch {
        num_nodes: n,
        num_edges: col_indices.len(),
        row_ptr,
        col_indices,
        edge_weights,
    }
}
