use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct StellarMLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    out: Linear,
}

impl StellarMLP {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        // Simple 3-layer MLP architecture
        // Input: X, Y, Z coordinates (3 features)
        let fc1 = linear(3, 128, vs.pp("fc1"))?;
        let fc2 = linear(128, 128, vs.pp("fc2"))?;
        let fc3 = linear(128, 64, vs.pp("fc3"))?;
        // Output: Temp (T), Radius (R), Mass (M), Luminosity (L) (4 features)
        let out = linear(64, 4, vs.pp("out"))?;

        Ok(Self { fc1, fc2, fc3, out })
    }
}

impl Module for StellarMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h1 = xs.apply(&self.fc1)?.relu()?;
        let h2 = h1.apply(&self.fc2)?.relu()?;
        let h3 = h2.apply(&self.fc3)?.relu()?;

        // Output layer uses relu or softplus to prevent predicting negative physical values
        let raw_out = h3.apply(&self.out)?;
        candle_nn::ops::softplus(&raw_out)
    }
}

/// Computes Physics-Informed loss using Stefan-Boltzmann law in solar units:
/// L/L_sun = (R/R_sun)^2 * (T/T_sun)^4
pub fn compute_pinn_loss(
    predictions: &Tensor,
    targets: &Tensor,
    temp_mean: f32, temp_std: f32,
    rad_mean: f32, rad_std: f32,
    lum_mean: f32, lum_std: f32,
) -> Result<Tensor> {
    // Standard Mean Squared Error (MSE) for data alignment
    let data_loss = predictions.sub(targets)?.sqr()?.mean_all()?;

    // Unpack normalized predicted values (batch size x 4)
    // Predictions are: [Temp, Rad, Mass, Lum]
    let t_norm = predictions.narrow(1, 0, 1)?;
    let r_norm = predictions.narrow(1, 1, 1)?;
    let l_norm = predictions.narrow(1, 3, 1)?;

    // Denormalize Temp, Radius, and Luminosity to physical units
    let t_unnorm = t_norm.mul(temp_std as f64)?.add(temp_mean as f64)?;
    let r_unnorm = r_norm.mul(rad_std as f64)?.add(rad_mean as f64)?;
    let l_unnorm = l_norm.mul(lum_std as f64)?.add(lum_mean as f64)?;

    // Compute theoretical Luminosity based on stellar physics
    let t_sun = 5778.0f64; // Solar effective temperature in Kelvin

    // t_ratio = T_predicted / T_sun
    let t_ratio = t_unnorm.div(t_sun)?;
    // t_ratio_pow4 = (T_predicted / T_sun)^4
    let t_ratio_pow4 = t_ratio.sqr()?.sqr()?;

    // r_sq = R_predicted^2
    let r_sq = r_unnorm.sqr()?;

    // L_theory = R^2 * (T / 5778)^4
    let l_theory = r_sq.mul(&t_ratio_pow4)?;

    // Calculate physical inconsistency loss (difference between predicted and theoretical L)
    let physics_loss = l_unnorm.sub(&l_theory)?.sqr()?.mean_all()?;

    // Combine data loss and physical constraints
    // Weights are adjusted to balance loss contributions
    let total_loss = data_loss.add(&physics_loss.mul(0.01)?)?;

    Ok(total_loss)
}