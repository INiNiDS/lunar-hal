use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

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
}

pub const GNN_INPUT_DIM: usize = 8;
pub const GNN_OUTPUT_DIM: usize = 3;
pub const GNN_VARIATIONAL_DIM: usize = 6;

/// Frozen output contract of the GNN-Kinematics model.
/// Per contract v1 the readout head must produce Cartesian velocity
/// components `vx/vy/vz` in km/s (Galactic frame) for every star node.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct KinematicsOutput {
    /// Cartesian velocity X in km/s (Galactic)
    pub vx: f32,
    /// Cartesian velocity Y in km/s (Galactic)
    pub vy: f32,
    /// Cartesian velocity Z in km/s (Galactic)
    pub vz: f32,
}

impl KinematicsOutput {
    /// Builds the contract from a raw `[N, 3]` readout slice for a single star.
    /// Returns `None` when the row does not contain exactly three components.
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

    /// Component accessor matching schema column order (`vx_kms`, `vy_kms`, `vz_kms`).
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
        let keys: Vec<&str> = value.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["vx", "vy", "vz"]);
    }

    #[test]
    fn from_row_rejects_wrong_widths_and_maps_components() {
        assert!(KinematicsOutput::from_row(&[1.0]).is_none());
        assert!(KinematicsOutput::from_row(&[1.0, 2.0, 3.0, 4.0]).is_none());
        let output = KinematicsOutput::from_row(&[9.0, 8.0, 7.0]).unwrap();
        assert_eq!(output.as_components(), [9.0, 8.0, 7.0]);
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

    let mean = gnn_output.clone().narrow(1, 0, 3);
    let log_var = gnn_output.narrow(1, 3, dims - 3);

    if temperature <= 0.0 {
        return mean;
    }

    let std = log_var.mul_scalar(0.5_f64).exp();

    let epsilon = Tensor::<B, 2>::random(
        [num_stars, 3],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    let scaled_noise = epsilon.mul(std).mul_scalar(temperature as f64);

    mean.add(scaled_noise)
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
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        adj[i][i] = 1.0;
        let mut degree_sum = 1.0f32;
        for &(j, dist_sq) in &dists[..k.min(dists.len())] {
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
