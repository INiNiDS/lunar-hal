use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use std::f32::consts::PI;

pub const FOURIER_LEVELS: usize = 8;
pub const FOURIER_DIM: usize = 3 * 2 * FOURIER_LEVELS;
pub const COND_DIM: usize = 2;
pub const MLP_INPUT_DIM: usize = FOURIER_DIM + COND_DIM;

pub fn fourier_encode<B: Backend>(xyz: Tensor<B, 2>, num_levels: usize) -> Tensor<B, 2> {
    let device = xyz.device();
    let [batch, _] = xyz.dims();

    let freqs: Vec<f32> = (0..num_levels).map(|l| 2f32.powi(l as i32) * PI).collect();
    let freq_tensor = Tensor::<B, 1>::from_data(TensorData::new(freqs, [num_levels]), &device)
        .reshape([1, num_levels]);

    let mut all_features: Vec<Tensor<B, 2>> = Vec::with_capacity(3);

    for dim in 0..3 {
        let col = xyz.clone().slice([0..batch, dim..dim + 1]);
        let scaled = col * freq_tensor.clone();
        let sin_f = scaled.clone().sin();
        let cos_f = scaled.cos();
        let stacked = Tensor::cat(vec![sin_f, cos_f], 1);
        let interleaved = stacked.reshape([batch, num_levels * 2]);
        all_features.push(interleaved);
    }

    Tensor::cat(all_features, 1)
}

#[derive(Module, Debug)]
pub struct StellarMlp<B: Backend> {
    fc1: Linear<B>,
    ln1: LayerNorm<B>,
    fc2: Linear<B>,
    ln2: LayerNorm<B>,
    fc3: Linear<B>,
    ln3: LayerNorm<B>,
    fc4: Linear<B>,
    ln4: LayerNorm<B>,
    fc5: Linear<B>,
    ln5: LayerNorm<B>,
    fc6: Linear<B>,
    out: Linear<B>,
}

#[derive(Config, Debug)]
pub struct StellarMlpConfig {
    #[config(default = 512)]
    hidden: usize,
    #[config(default = 256)]
    hidden2: usize,
    #[config(default = 128)]
    hidden3: usize,
    #[config(default = 1e-5)]
    layer_norm_eps: f64,
}

impl StellarMlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StellarMlp<B> {
        let ln = |d| LayerNormConfig::new(d).with_epsilon(self.layer_norm_eps).init(device);

        StellarMlp {
            fc1: LinearConfig::new(MLP_INPUT_DIM, self.hidden).init(device),
            ln1: ln(self.hidden),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            ln2: ln(self.hidden),
            fc3: LinearConfig::new(self.hidden, self.hidden).init(device),
            ln3: ln(self.hidden),
            fc4: LinearConfig::new(self.hidden, self.hidden).init(device),
            ln4: ln(self.hidden),
            fc5: LinearConfig::new(self.hidden, self.hidden2).init(device),
            ln5: ln(self.hidden2),
            fc6: LinearConfig::new(self.hidden2, self.hidden3).init(device),
            out: LinearConfig::new(self.hidden3, 4).init(device),
        }
    }
}

impl StellarMlpConfig {
    pub const fn fourier_levels() -> usize {
        FOURIER_LEVELS
    }
}

impl<B: Backend> StellarMlp<B> {
    /// # Shapes
    ///   - Input [batch_size, 5]: (x, y, z, bp_rp, M_G) where M_G = g_mag - 5*log10(d) + 5
    ///   - Output [batch_size, 4]: (log10_teff, log10_rad, log10_mass, log10_lum) in normalized space
    pub fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, _] = xs.dims();
        let xyz = xs.clone().slice([0..batch, 0..3]);
        let cond = xs.slice([0..batch, 3..5]);

        let fourier = fourier_encode(xyz, FOURIER_LEVELS);
        let mlp_input = Tensor::cat(vec![fourier, cond], 1);

        let h1 = burn::tensor::activation::silu(self.ln1.forward(self.fc1.forward(mlp_input)));
        let h2 = burn::tensor::activation::silu(
            self.ln2.forward(self.fc2.forward(h1.clone()) + h1),
        );
        let h3 = burn::tensor::activation::silu(
            self.ln3.forward(self.fc3.forward(h2.clone()) + h2),
        );
        let h4 = burn::tensor::activation::silu(
            self.ln4.forward(self.fc4.forward(h3.clone()) + h3),
        );
        let h5 = burn::tensor::activation::silu(self.ln5.forward(self.fc5.forward(h4)));
        let h6 = burn::tensor::activation::silu(self.fc6.forward(h5));
        self.out.forward(h6)
    }
}
