use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, linear};

pub struct StellarMLP {
    fc1: Linear,
    ln1: LayerNorm,
    fc2: Linear,
    ln2: LayerNorm,
    fc3: Linear,
    ln3: LayerNorm,
    fc4: Linear,
    ln4: LayerNorm,
    fc5: Linear,
    out: Linear,
}

impl StellarMLP {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let fc1 = linear(3, 256, vs.pp("fc1"))?;
        let ln1 = candle_nn::layer_norm(256, 1e-5, vs.pp("ln1"))?;
        let fc2 = linear(256, 256, vs.pp("fc2"))?;
        let ln2 = candle_nn::layer_norm(256, 1e-5, vs.pp("ln2"))?;
        let fc3 = linear(256, 256, vs.pp("fc3"))?;
        let ln3 = candle_nn::layer_norm(256, 1e-5, vs.pp("ln3"))?;
        let fc4 = linear(256, 128, vs.pp("fc4"))?;
        let ln4 = candle_nn::layer_norm(128, 1e-5, vs.pp("ln4"))?;
        let fc5 = linear(128, 64, vs.pp("fc5"))?;
        let out = linear(64, 4, vs.pp("out"))?;

        Ok(Self {
            fc1,
            ln1,
            fc2,
            ln2,
            fc3,
            ln3,
            fc4,
            ln4,
            fc5,
            out,
        })
    }
}

impl Module for StellarMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h1 = candle_nn::ops::leaky_relu(&self.ln1.forward(&self.fc1.forward(xs)?)?, 0.01)?;
        let h2 =
            candle_nn::ops::leaky_relu(&self.ln2.forward(&(&h1 + self.fc2.forward(&h1)?)?)?, 0.01)?;
        let h3 =
            candle_nn::ops::leaky_relu(&self.ln3.forward(&(&h2 + self.fc3.forward(&h2)?)?)?, 0.01)?;
        let h4 = candle_nn::ops::leaky_relu(&self.ln4.forward(&self.fc4.forward(&h3)?)?, 0.01)?;
        let h5 = candle_nn::ops::leaky_relu(&self.fc5.forward(&h4)?, 0.01)?;
        self.out.forward(&h5)
    }
}
