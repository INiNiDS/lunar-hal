use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use polars::prelude::*;
use rand::seq::SliceRandom;
use rand::rng;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct NormParams {
    pub x_mean: f32, pub x_std: f32,
    pub y_mean: f32, pub y_std: f32,
    pub z_mean: f32, pub z_std: f32,
    pub teff_mean: f32, pub teff_std: f32,
    pub rad_mean: f32, pub rad_std: f32,
    pub mass_mean: f32, pub mass_std: f32,
    pub lum_mean: f32, pub lum_std: f32,
}

pub struct StellarDataset {
    pub inputs: Tensor,
    pub targets: Tensor,
    pub norm: NormParams,
    pub n_samples: usize,
}

impl StellarDataset {
    pub fn load(parquet_path: &Path) -> Result<Self> {
        println!("Loading parquet: {}", parquet_path.display());
        let file = File::open(parquet_path).context("failed to open parquet")?;
        let df = ParquetReader::new(file).finish().context("failed to read parquet")?;

        let target_cols: &[&str] = &["x_pc", "y_pc", "z_pc", "st_teff", "st_rad", "st_mass", "st_lum"];
        let df = df.drop_nulls(Some(target_cols))?;

        let df = df.lazy()
            .filter(
                col("x_pc").abs().lt(lit(10000.0))
                    .and(col("y_pc").abs().lt(lit(10000.0)))
                    .and(col("z_pc").abs().lt(lit(10000.0)))
            )
            .collect()?;

        let n = df.height();
        println!("Loaded {} complete rows (all 7 features non-null, spatial outliers filtered)", n);

        let x_arr = extract_f32(&df, "x_pc")?;
        let y_arr = extract_f32(&df, "y_pc")?;
        let z_arr = extract_f32(&df, "z_pc")?;
        let t_arr = extract_f32(&df, "st_teff")?;
        let r_arr = extract_f32(&df, "st_rad")?;
        let m_arr = extract_f32(&df, "st_mass")?;
        let l_arr = extract_f32(&df, "st_lum")?;

        let (x_m, x_s) = mean_std(&x_arr);
        let (y_m, y_s) = mean_std(&y_arr);
        let (z_m, z_s) = mean_std(&z_arr);
        let (t_m, t_s) = mean_std(&t_arr);
        let (r_m, r_s) = mean_std(&r_arr);
        let (m_m, m_s) = mean_std(&m_arr);
        let (l_m, l_s) = mean_std(&l_arr);

        let norm = NormParams {
            x_mean: x_m, x_std: x_s,
            y_mean: y_m, y_std: y_s,
            z_mean: z_m, z_std: z_s,
            teff_mean: t_m, teff_std: t_s,
            rad_mean: r_m, rad_std: r_s,
            mass_mean: m_m, mass_std: m_s,
            lum_mean: l_m, lum_std: l_s,
        };

        println!("Normalization parameters:");
        println!("  x_pc:    mean={:.4}, std={:.4}", norm.x_mean, norm.x_std);
        println!("  y_pc:    mean={:.4}, std={:.4}", norm.y_mean, norm.y_std);
        println!("  z_pc:    mean={:.4}, std={:.4}", norm.z_mean, norm.z_std);
        println!("  st_teff: mean={:.4}, std={:.4}", norm.teff_mean, norm.teff_std);
        println!("  st_rad:  mean={:.4}, std={:.4}", norm.rad_mean, norm.rad_std);
        println!("  st_mass: mean={:.4}, std={:.4}", norm.mass_mean, norm.mass_std);
        println!("  st_lum:  mean={:.4}, std={:.4}", norm.lum_mean, norm.lum_std);

        let x_n = normalize_vec(&x_arr, x_m, x_s);
        let y_n = normalize_vec(&y_arr, y_m, y_s);
        let z_n = normalize_vec(&z_arr, z_m, z_s);
        let t_n = normalize_vec(&t_arr, t_m, t_s);
        let r_n = normalize_vec(&r_arr, r_m, r_s);
        let m_n = normalize_vec(&m_arr, m_m, m_s);
        let l_n = normalize_vec(&l_arr, l_m, l_s);

        let inputs = Tensor::stack(&[x_n, y_n, z_n], 1)?;
        let targets = Tensor::stack(&[t_n, r_n, m_n, l_n], 1)?;

        Ok(Self { inputs, targets, norm, n_samples: n })
    }

    pub fn split_to_device(self, val_frac: f32, device: &Device) -> Result<(Self, Self)> {
        let n = self.n_samples;
        let n_val = ((n as f32) * val_frac) as usize;
        let n_train = n - n_val;

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng());

        let train_idx: Vec<u32> = indices[..n_train].iter().map(|&i| i as u32).collect();
        let val_idx: Vec<u32> = indices[n_train..].iter().map(|&i| i as u32).collect();

        let train_idx_t = Tensor::from_vec(train_idx, n_train, &Device::Cpu)?;
        let val_idx_t = Tensor::from_vec(val_idx, n_val, &Device::Cpu)?;

        let train_inputs = self.inputs.index_select(&train_idx_t, 0)?.to_device(device)?;
        let train_targets = self.targets.index_select(&train_idx_t, 0)?.to_device(device)?;
        let val_inputs = self.inputs.index_select(&val_idx_t, 0)?.to_device(device)?;
        let val_targets = self.targets.index_select(&val_idx_t, 0)?.to_device(device)?;

        let train = StellarDataset {
            inputs: train_inputs,
            targets: train_targets,
            norm: self.norm.clone(),
            n_samples: n_train,
        };
        let val = StellarDataset {
            inputs: val_inputs,
            targets: val_targets,
            norm: self.norm,
            n_samples: n_val,
        };

        println!("Train samples: {}, Validation samples: {}", n_train, n_val);
        Ok((train, val))
    }

    pub fn shuffle(&mut self) -> Result<()> {
        let mut indices: Vec<u32> = (0..self.n_samples as u32).collect();
        indices.shuffle(&mut rng());
        let idx_tensor = Tensor::from_vec(indices, self.n_samples, self.inputs.device())?;
        self.inputs = self.inputs.index_select(&idx_tensor, 0)?;
        self.targets = self.targets.index_select(&idx_tensor, 0)?;
        Ok(())
    }
}

fn extract_f32(df: &DataFrame, name: &str) -> Result<Vec<f32>> {
    let s = df.column(name).context(format!("column {name} not found"))?;
    let ca = s.f64().context(format!("column {name} is not f64"))?;
    let values: Vec<f32> = ca
        .into_iter()
        .map(|opt| opt.map(|v| v as f32).unwrap_or(0.0f32))
        .collect();
    Ok(values)
}

fn mean_std(data: &[f32]) -> (f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let variance = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std = variance.sqrt();
    (mean, std)
}

fn normalize_vec(data: &[f32], mean: f32, std: f32) -> Tensor {
    let normalized: Vec<f32> = data.iter().map(|v| (v - mean) / std).collect();
    let len = normalized.len();
    Tensor::from_vec(normalized, len, &Device::Cpu).unwrap()
}

pub struct BatchIterator {
    inputs: Tensor,
    targets: Tensor,
    batch_size: usize,
    current: usize,
    n_samples: usize,
}

impl BatchIterator {
    pub fn new(dataset: &StellarDataset, batch_size: usize) -> Self {
        Self {
            inputs: dataset.inputs.clone(),
            targets: dataset.targets.clone(),
            batch_size,
            current: 0,
            n_samples: dataset.n_samples,
        }
    }

    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.current >= self.n_samples {
            return None;
        }
        let end = (self.current + self.batch_size).min(self.n_samples);
        let batch_inputs = self.inputs.narrow(0, self.current, end - self.current).ok()?;
        let batch_targets = self.targets.narrow(0, self.current, end - self.current).ok()?;
        self.current = end;
        Some((batch_inputs.contiguous().ok()?, batch_targets.contiguous().ok()?))
    }
}