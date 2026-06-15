use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

pub const SIREN_W0: f64 = 30.0;
pub const SIREN_INPUT_DIM: usize = 5;
pub const SIREN_HIDDEN_DIM: usize = 64;
pub const SIREN_OUTPUT_DIM: usize = 3;

#[derive(Module, Debug)]
pub struct StellarSiren<B: Backend> {
    first: Linear<B>,
    hidden1: Linear<B>,
    hidden2: Linear<B>,
    output: Linear<B>,
}

#[derive(Config, Debug)]
pub struct StellarSirenConfig {
    #[config(default = 64)]
    hidden: usize,
}

impl StellarSirenConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StellarSiren<B> {
        let hidden = self.hidden;
        let first_bound = 1.0 / SIREN_INPUT_DIM as f64;
        let hidden_bound = (6.0 / hidden as f64).sqrt() / SIREN_W0;

        StellarSiren {
            first: LinearConfig::new(SIREN_INPUT_DIM, hidden)
                .with_initializer(burn::nn::Initializer::Uniform {
                    min: -first_bound,
                    max: first_bound,
                })
                .init(device),
            hidden1: LinearConfig::new(hidden, hidden)
                .with_initializer(burn::nn::Initializer::Uniform {
                    min: -hidden_bound,
                    max: hidden_bound,
                })
                .init(device),
            hidden2: LinearConfig::new(hidden, hidden)
                .with_initializer(burn::nn::Initializer::Uniform {
                    min: -hidden_bound,
                    max: hidden_bound,
                })
                .init(device),
            output: LinearConfig::new(hidden, SIREN_OUTPUT_DIM).init(device),
        }
    }
}

impl<B: Backend> StellarSiren<B> {
    pub fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.first.forward(xs).mul_scalar(SIREN_W0).sin();
        let h = self.hidden1.forward(h).mul_scalar(SIREN_W0).sin();
        let h = self.hidden2.forward(h).mul_scalar(SIREN_W0).sin();
        burn::tensor::activation::sigmoid(self.output.forward(h))
    }
}