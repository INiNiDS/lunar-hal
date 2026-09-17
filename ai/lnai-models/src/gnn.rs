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

/// Stage 6: explicit readout-head contract for GNN-Kinematics.
///
/// * `Deterministic` — `[N, 3]` mean velocities `(vx, vy, vz)`.
/// * `Variational` — `[N, 6]` mean velocities followed by per-component
///   `logvar`; sampling goes through [`split_mean_logvar`] so train,
///   eval and serving can never disagree on the layout.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GnnHeadKind {
    Deterministic,
    Variational,
}

impl GnnHeadKind {
    /// Maps a readout width to its head; `None` for unsupported widths.
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

/// Splits a `[N, W]` readout into mean `[N, 3]` and — for the variational
/// head — `logvar [N, 3]`, following [`GnnHeadKind`].
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

/// Standard VAE KL of `N(mean, exp(logvar))` against `N(0, 1)`, averaged
/// over nodes and components (normalized velocity space, so the unit
/// prior matches the dataset scale).
pub fn variational_kl<B: Backend>(mean: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 1> {
    let one = Tensor::<B, 2>::ones_like(&logvar);
    let kl = one + logvar.clone() - mean.square() - logvar.exp();
    let n = kl.dims()[0] as f32 * 3.0;
    kl.sum().mul_scalar(-0.5).div_scalar(n)
}

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

    if temperature <= 0.0 {
        return mean;
    }

    let std = match logvar {
        Some(logvar) => log_var_to_std(logvar),
        // Deterministic head: unit sampling noise (no learned variance).
        None => Tensor::<B, 2>::ones([num_stars, GNN_OUTPUT_DIM], device),
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
