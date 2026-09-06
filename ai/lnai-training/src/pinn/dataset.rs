use anyhow::{Context, Result};
use burn::prelude::*;
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

pub const INPUT_DIM: usize = 5;
pub const TARGET_DIM: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormParams {
    pub x_mean: f32,
    pub x_std: f32,
    pub y_mean: f32,
    pub y_std: f32,
    pub z_mean: f32,
    pub z_std: f32,
    pub bp_rp_mean: f32,
    pub bp_rp_std: f32,
    pub mg_mean: f32,
    pub mg_std: f32,
    pub log_teff_mean: f32,
    pub log_teff_std: f32,
    pub log_rad_mean: f32,
    pub log_rad_std: f32,
    pub log_mass_mean: f32,
    pub log_mass_std: f32,
    pub log_lum_mean: f32,
    pub log_lum_std: f32,
}
pub struct StellarDataset<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
    pub norm: NormParams,
    pub n_samples: usize,
    pub device: B::Device,
}

impl<B: Backend> StellarDataset<B> {
    pub fn load(parquet_path: &Path, device: &B::Device) -> Result<Self> {
        let (df, n) = read_filtered_parquet(parquet_path)?;
        println!(
            "Loaded {} complete rows (all required features non-null, outliers filtered)",
            n
        );

        let raw = RawColumns::extract(&df)?;
        let norm = raw.compute_norm();
        let (inputs_cpu, targets_cpu) = raw.build_cpu(&norm);
        raw.print_norm(&norm);

        let inputs = Tensor::<B, 2>::from_data(TensorData::new(inputs_cpu, [n, INPUT_DIM]), device);
        let targets =
            Tensor::<B, 2>::from_data(TensorData::new(targets_cpu, [n, TARGET_DIM]), device);

        Ok(Self {
            inputs,
            targets,
            norm,
            n_samples: n,
            device: device.clone(),
        })
    }

    pub fn load_with_norm(
        parquet_path: &Path,
        norm: NormParams,
        device: &B::Device,
    ) -> Result<Self> {
        let (df, n) = read_filtered_parquet(parquet_path)?;
        println!(
            "Loaded {} complete rows (all required features non-null, outliers filtered)",
            n
        );
        println!("Using external normalization (resuming from saved model).");

        let raw = RawColumns::extract(&df)?;
        let (inputs_cpu, targets_cpu) = raw.build_cpu(&norm);

        let inputs = Tensor::<B, 2>::from_data(TensorData::new(inputs_cpu, [n, INPUT_DIM]), device);
        let targets =
            Tensor::<B, 2>::from_data(TensorData::new(targets_cpu, [n, TARGET_DIM]), device);

        Ok(Self {
            inputs,
            targets,
            norm,
            n_samples: n,
            device: device.clone(),
        })
    }

    /// Deterministic split on `seed`: same seed + same data always yields
    /// the same train/val partition (old/new trainer parity).
    pub fn split(self, val_frac: f32) -> (Self, Self) {
        self.split_with_seed(val_frac, crate::runner::DEFAULT_TRAIN_SEED)
    }

    pub fn split_with_seed(self, val_frac: f32, seed: u64) -> (Self, Self) {
        let n = self.n_samples;
        let n_val = ((n as f32) * val_frac) as usize;
        let n_train = n - n_val;

        let mut split_indices: Vec<usize> = (0..n).collect();
        split_indices.shuffle(&mut StdRng::seed_from_u64(seed));

        let (train_idx, val_idx) = split_indices.split_at(n_train);

        let to_idx_tensor = |ids: &[usize]| -> Tensor<B, 1, Int> {
            let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
            let len = ids.len();
            Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [len]), &self.device)
        };

        let train_inputs = self.inputs.clone().select(0, to_idx_tensor(train_idx));
        let train_targets = self.targets.clone().select(0, to_idx_tensor(train_idx));
        let val_inputs = self.inputs.select(0, to_idx_tensor(val_idx));
        let val_targets = self.targets.select(0, to_idx_tensor(val_idx));

        let train = StellarDataset {
            inputs: train_inputs,
            targets: train_targets,
            norm: self.norm.clone(),
            n_samples: n_train,
            device: self.device.clone(),
        };
        let val = StellarDataset {
            inputs: val_inputs,
            targets: val_targets,
            norm: self.norm,
            n_samples: n_val,
            device: self.device,
        };

        println!("Train samples: {}, Validation samples: {}", n_train, n_val);
        (train, val)
    }
}

/// GPU-resident batch iterator: shuffles rows once per epoch via a single
/// gather on device, then serves batches as zero-copy row slices.
pub struct GpuBatcher<B: Backend> {
    inputs: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    batch_size: usize,
    n_samples: usize,
    current: usize,
}

impl<B: Backend> GpuBatcher<B> {
    pub fn new(dataset: &StellarDataset<B>, batch_size: usize) -> Self {
        Self::new_with_seed(dataset, batch_size, crate::runner::DEFAULT_TRAIN_SEED)
    }

    pub fn new_with_seed(dataset: &StellarDataset<B>, batch_size: usize, seed: u64) -> Self {
        let n_samples = dataset.n_samples;
        let mut idx: Vec<u32> = (0..n_samples as u32).collect();
        idx.shuffle(&mut StdRng::seed_from_u64(seed));
        let idx_dev =
            Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [n_samples]), &dataset.device);
        Self {
            inputs: dataset.inputs.clone().select(0, idx_dev.clone()),
            targets: dataset.targets.clone().select(0, idx_dev),
            batch_size,
            n_samples,
            current: 0,
        }
    }

    pub fn next_batch(&mut self) -> Option<(Tensor<B, 2>, Tensor<B, 2>)> {
        if self.current >= self.n_samples {
            return None;
        }
        let start = self.current;
        let end = (start + self.batch_size).min(self.n_samples);
        self.current = end;
        let inputs = self.inputs.clone().slice([start..end, 0..INPUT_DIM]);
        let targets = self.targets.clone().slice([start..end, 0..TARGET_DIM]);
        Some((inputs, targets))
    }
}

fn read_filtered_parquet(parquet_path: &Path) -> Result<(DataFrame, usize)> {
    println!("Loading parquet: {}", parquet_path.display());
    let file = File::open(parquet_path).context("failed to open parquet")?;
    let df = ParquetReader::new(file)
        .finish()
        .context("failed to read parquet")?;

    let required_cols: &[&str] = &[
        "x_pc", "y_pc", "z_pc", "bp_rp", "g_mag", "st_teff", "st_rad", "st_mass", "st_lum",
    ];

    for &col_name in required_cols {
        if df.column(col_name).is_err() {
            anyhow::bail!(
                "Column '{}' not found in dataset. \
                 Please re-fetch data from Gaia using: lnaicli fetch (release mode) \
                 and then: lnaicli clean",
                col_name
            );
        }
    }

    let df = df
        .drop_nulls(Some(required_cols))
        .context("drop_nulls failed")?;

    let df = df
        .lazy()
        .filter(
            col("x_pc")
                .abs()
                .lt(lit(10000.0))
                .and(col("y_pc").abs().lt(lit(10000.0)))
                .and(col("z_pc").abs().lt(lit(10000.0)))
                .and(col("bp_rp").gt(lit(-1.0)))
                .and(col("bp_rp").lt(lit(10.0)))
                .and(col("g_mag").gt(lit(0.0)))
                .and(col("g_mag").lt(lit(25.0))),
        )
        .collect()?;

    let n = df.height();
    Ok((df, n))
}

struct RawColumns {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    bp_rp: Vec<f32>,
    mg: Vec<f32>,
    log_teff: Vec<f32>,
    log_rad: Vec<f32>,
    log_mass: Vec<f32>,
    log_lum: Vec<f32>,
}

impl RawColumns {
    fn extract(df: &DataFrame) -> Result<Self> {
        let x = extract_f32(df, "x_pc")?;
        let y = extract_f32(df, "y_pc")?;
        let z = extract_f32(df, "z_pc")?;
        let bp_rp = extract_f32(df, "bp_rp")?;
        let g_mag = extract_f32(df, "g_mag")?;
        let teff = extract_f32(df, "st_teff")?;
        let rad = extract_f32(df, "st_rad")?;
        let mass = extract_f32(df, "st_mass")?;
        let lum = extract_f32(df, "st_lum")?;

        let mg: Vec<f32> = x
            .par_iter()
            .zip(&y)
            .zip(&z)
            .zip(&g_mag)
            .map(|(((xi, yi), zi), &g)| {
                let d = (xi * xi + yi * yi + zi * zi).sqrt().max(1e-6);
                g - 5.0 * d.log10() + 5.0
            })
            .collect();

        let log_teff: Vec<f32> = teff.par_iter().map(|&v| v.max(1e-10).log10()).collect();
        let log_rad: Vec<f32> = rad.par_iter().map(|&v| v.max(1e-10).log10()).collect();
        let log_mass: Vec<f32> = mass.par_iter().map(|&v| v.max(1e-10).log10()).collect();
        let log_lum: Vec<f32> = lum.par_iter().map(|&v| v.max(1e-10).log10()).collect();

        Ok(Self {
            x,
            y,
            z,
            bp_rp,
            mg,
            log_teff,
            log_rad,
            log_mass,
            log_lum,
        })
    }

    fn compute_norm(&self) -> NormParams {
        let (x_m, x_s) = mean_std(&self.x);
        let (y_m, y_s) = mean_std(&self.y);
        let (z_m, z_s) = mean_std(&self.z);
        let (bp_rp_m, bp_rp_s) = mean_std(&self.bp_rp);
        let (mg_m, mg_s) = mean_std(&self.mg);
        let (lt_m, lt_s) = mean_std(&self.log_teff);
        let (lr_m, lr_s) = mean_std(&self.log_rad);
        let (lm_m, lm_s) = mean_std(&self.log_mass);
        let (ll_m, ll_s) = mean_std(&self.log_lum);

        NormParams {
            x_mean: x_m,
            x_std: x_s,
            y_mean: y_m,
            y_std: y_s,
            z_mean: z_m,
            z_std: z_s,
            bp_rp_mean: bp_rp_m,
            bp_rp_std: bp_rp_s,
            mg_mean: mg_m,
            mg_std: mg_s,
            log_teff_mean: lt_m,
            log_teff_std: lt_s,
            log_rad_mean: lr_m,
            log_rad_std: lr_s,
            log_mass_mean: lm_m,
            log_mass_std: lm_s,
            log_lum_mean: ll_m,
            log_lum_std: ll_s,
        }
    }

    fn print_norm(&self, norm: &NormParams) {
        println!("Normalization parameters:");
        println!(
            "  x_pc:      mean={:.4}, std={:.4}",
            norm.x_mean, norm.x_std
        );
        println!(
            "  y_pc:      mean={:.4}, std={:.4}",
            norm.y_mean, norm.y_std
        );
        println!(
            "  z_pc:      mean={:.4}, std={:.4}",
            norm.z_mean, norm.z_std
        );
        println!(
            "  bp_rp:     mean={:.4}, std={:.4}",
            norm.bp_rp_mean, norm.bp_rp_std
        );
        println!(
            "  M_G:       mean={:.4}, std={:.4}",
            norm.mg_mean, norm.mg_std
        );
        println!(
            "  log_teff:  mean={:.4}, std={:.4}",
            norm.log_teff_mean, norm.log_teff_std
        );
        println!(
            "  log_rad:   mean={:.4}, std={:.4}",
            norm.log_rad_mean, norm.log_rad_std
        );
        println!(
            "  log_mass:  mean={:.4}, std={:.4}",
            norm.log_mass_mean, norm.log_mass_std
        );
        println!(
            "  log_lum:   mean={:.4}, std={:.4}",
            norm.log_lum_mean, norm.log_lum_std
        );
    }

    fn build_cpu(&self, norm: &NormParams) -> (Vec<f32>, Vec<f32>) {
        let x_n = normalize_vec(&self.x, norm.x_mean, norm.x_std);
        let y_n = normalize_vec(&self.y, norm.y_mean, norm.y_std);
        let z_n = normalize_vec(&self.z, norm.z_mean, norm.z_std);
        let bp_rp_n = normalize_vec(&self.bp_rp, norm.bp_rp_mean, norm.bp_rp_std);
        let mg_n = normalize_vec(&self.mg, norm.mg_mean, norm.mg_std);
        let lt_n = normalize_vec(&self.log_teff, norm.log_teff_mean, norm.log_teff_std);
        let lr_n = normalize_vec(&self.log_rad, norm.log_rad_mean, norm.log_rad_std);
        let lm_n = normalize_vec(&self.log_mass, norm.log_mass_mean, norm.log_mass_std);
        let ll_n = normalize_vec(&self.log_lum, norm.log_lum_mean, norm.log_lum_std);

        let inputs = interleave(&[&x_n, &y_n, &z_n, &bp_rp_n, &mg_n]);
        let targets = interleave(&[&lt_n, &lr_n, &lm_n, &ll_n]);
        (inputs, targets)
    }
}

fn extract_f32(df: &DataFrame, name: &str) -> Result<Vec<f32>> {
    let s = df
        .column(name)
        .context(format!("column {name} not found"))?;
    let s = s
        .cast(&DataType::Float64)
        .context(format!("column {name} cast to f64 failed"))?;
    let ca = s.f64().context(format!("column {name} is not f64"))?;
    let values: Vec<f32> = ca
        .into_iter()
        .map(|opt| opt.map(|v| v as f32).unwrap_or(0.0f32))
        .collect();
    Ok(values)
}

fn mean_std(data: &[f32]) -> (f32, f32) {
    let n = data.len() as f64;
    let mean = data.par_iter().map(|&v| v as f64).sum::<f64>() / n;
    let variance = data
        .par_iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean as f32, variance.sqrt() as f32)
}

fn normalize_vec(data: &[f32], mean: f32, std: f32) -> Vec<f32> {
    data.par_iter().map(|v| (v - mean) / std).collect()
}

fn interleave(column_vecs: &[&Vec<f32>]) -> Vec<f32> {
    let n_cols = column_vecs.len();
    let n_rows = column_vecs[0].len();
    let mut out = vec![0.0f32; n_cols * n_rows];
    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row, chunk)| {
            for (col, col_vec) in column_vecs.iter().enumerate() {
                chunk[col] = col_vec[row];
            }
        });
    out
}
