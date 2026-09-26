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
    #[module(skip)]
    pub w0: f64,
}

#[derive(Config, Debug)]
pub struct StellarSirenConfig {
    #[config(default = 64)]
    pub hidden: usize,
    #[config(default = 30.0)]
    pub w0: f64,
}

impl StellarSirenConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StellarSiren<B> {
        let hidden = self.hidden;
        let w0 = self.w0;
        let first_bound = 1.0 / SIREN_INPUT_DIM as f64;
        let hidden_bound = (6.0 / hidden as f64).sqrt() / w0;

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
            w0,
        }
    }
}

impl<B: Backend> StellarSiren<B> {
    pub fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.first.forward(xs).mul_scalar(self.w0).sin();
        let h = self.hidden1.forward(h).mul_scalar(self.w0).sin();
        let h = self.hidden2.forward(h).mul_scalar(self.w0).sin();
        burn::tensor::activation::sigmoid(self.output.forward(h))
    }

    /// Stage 7: chunked texture inference.
    /// Evaluates `xs` in bounded row chunks of size `chunk_size` to limit peak VRAM/RAM
    /// consumption during high-resolution texture generation (e.g. 256x256 or 512x512).
    pub fn forward_chunked(&self, xs: Tensor<B, 2>, chunk_size: usize) -> Tensor<B, 2> {
        let [total_rows, _] = xs.dims();
        if total_rows <= chunk_size || chunk_size == 0 {
            return self.forward(xs);
        }

        let mut chunks = Vec::with_capacity((total_rows + chunk_size - 1) / chunk_size);
        let mut start = 0;
        while start < total_rows {
            let end = (start + chunk_size).min(total_rows);
            let chunk_input = xs.clone().slice([start..end]);
            chunks.push(self.forward(chunk_input));
            start = end;
        }
        Tensor::cat(chunks, 0)
    }
}
