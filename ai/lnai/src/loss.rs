use candle_core::{Result, Tensor};

const T_SUN: f64 = 5778.0;

fn denorm(norm: &Tensor, mean: f32, std: f32) -> Result<Tensor> {
    let scaled = (norm * f64::from(std))?;
    scaled + f64::from(mean)
}

pub fn compute_pinn_loss(
    predictions: &Tensor,
    targets: &Tensor,
    physics_weight: f64,
    teff_mean: f32,
    teff_std: f32,
    rad_mean: f32,
    rad_std: f32,
    lum_mean: f32,
    lum_std: f32,
) -> Result<Tensor> {
    let data_loss = (&predictions.sub(targets)?).sqr()?.mean_all()?;

    let t_norm = predictions.narrow(1, 0, 1)?;
    let r_norm = predictions.narrow(1, 1, 1)?;
    let l_norm = predictions.narrow(1, 3, 1)?;

    let t_phys = denorm(&t_norm, teff_mean, teff_std)?;
    let r_phys = denorm(&r_norm, rad_mean, rad_std)?;
    let l_phys = denorm(&l_norm, lum_mean, lum_std)?;

    let t_ratio = (&t_phys / T_SUN)?;
    let t_ratio_pow4 = t_ratio.sqr()?.sqr()?;
    let r_sq = r_phys.sqr()?;
    let l_theory = (&r_sq * &t_ratio_pow4)?;

    let physics_loss = (&l_phys - &l_theory)?.sqr()?.mean_all()?;

    let scaled_physics = (&physics_loss * physics_weight)?;
    data_loss.add(&scaled_physics)
}

pub fn compute_data_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    (&predictions.sub(targets)?).sqr()?.mean_all()
}

pub fn compute_physics_loss(
    predictions: &Tensor,
    teff_mean: f32,
    teff_std: f32,
    rad_mean: f32,
    rad_std: f32,
    lum_mean: f32,
    lum_std: f32,
) -> Result<Tensor> {
    let t_norm = predictions.narrow(1, 0, 1)?;
    let r_norm = predictions.narrow(1, 1, 1)?;
    let l_norm = predictions.narrow(1, 3, 1)?;

    let t_phys = denorm(&t_norm, teff_mean, teff_std)?;
    let r_phys = denorm(&r_norm, rad_mean, rad_std)?;
    let l_phys = denorm(&l_norm, lum_mean, lum_std)?;

    let t_ratio = (&t_phys / T_SUN)?;
    let t_ratio_pow4 = t_ratio.sqr()?.sqr()?;
    let r_sq = r_phys.sqr()?;
    let l_theory = (&r_sq * &t_ratio_pow4)?;

    (&l_phys - &l_theory)?.sqr()?.mean_all()
}
