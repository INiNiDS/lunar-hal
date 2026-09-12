use anyhow::Result;
use burn::prelude::*;
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

use lnai_models::SIREN_INPUT_DIM;

pub const TARGET_DIM: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirenNorm {
    pub bp_rp_mean: f32,
    pub bp_rp_std: f32,
    pub mg_mean: f32,
    pub mg_std: f32,
    pub ruwe_mean: f32,
    pub ruwe_std: f32,
}

pub struct SirenDataset {
    pub inputs_cpu: Vec<f32>,
    pub targets_cpu: Vec<f32>,
    pub norm: SirenNorm,
    pub n_samples: usize,
}

struct StarParams {
    bp_rp: f32,
    mg: f32,
    ruwe: f32,
}

impl SirenDataset {
    pub fn generate(
        parquet_path: &Path,
        texture_size: usize,
        max_stars: usize,
        val_frac: f32,
        seed: u64,
        max_rows: Option<u64>,
    ) -> Result<(Self, Self)> {
        let (df, n_total) = read_filtered_parquet(parquet_path, max_rows)?;
        println!("Loaded {} filtered stars from parquet", n_total);

        let mut stars = extract_star_params(&df)?;
        println!("Extracted {} valid star parameter sets", stars.len());

        let n_stars = max_stars.min(stars.len());
        stars.shuffle(&mut StdRng::seed_from_u64(seed));
        stars.truncate(n_stars);
        println!("Using {} stars (max_stars={})", n_stars, max_stars);

        let norm = compute_norm(&stars);
        print_norm(&norm);

        let u_coords = generate_uv_grid(texture_size);
        let n_pixels = texture_size * texture_size;
        let total_samples = n_stars * n_pixels;
        let ram_gb =
            total_samples as f64 * (SIREN_INPUT_DIM + TARGET_DIM) as f64 * 4.0 / 1073741824.0;
        println!(
            "Texture grid: {}x{} = {} pixels/star",
            texture_size, texture_size, n_pixels
        );
        println!(
            "Total samples: {} stars x {} pixels = {} samples (~{:.2} GB)",
            n_stars, n_pixels, total_samples, ram_gb
        );

        let mut all_inputs: Vec<f32> = Vec::with_capacity(total_samples * SIREN_INPUT_DIM);
        let mut all_targets: Vec<f32> = Vec::with_capacity(total_samples * TARGET_DIM);

        for (star_idx, star) in stars.iter().enumerate() {
            let n_bp = (star.bp_rp - norm.bp_rp_mean) / norm.bp_rp_std;
            let n_mg = (star.mg - norm.mg_mean) / norm.mg_std;
            let n_ruwe = (star.ruwe - norm.ruwe_mean) / norm.ruwe_std;

            let base_color = star_base_color(star.bp_rp, star.mg);
            let spot_params = compute_spot_params(star.bp_rp, star.mg);

            for i in 0..n_pixels {
                let u = u_coords[i * 2];
                let v = u_coords[i * 2 + 1];

                all_inputs.push(u);
                all_inputs.push(v);
                all_inputs.push(n_bp);
                all_inputs.push(n_mg);
                all_inputs.push(n_ruwe);

                let (r, g, b) = generate_pixel(
                    u,
                    v,
                    &base_color,
                    &spot_params,
                    seed.wrapping_add((star_idx as u64) * 1000),
                );
                all_targets.push(r);
                all_targets.push(g);
                all_targets.push(b);
            }

            if (star_idx + 1) % 500 == 0 || star_idx + 1 == n_stars {
                print!(
                    "\r  Generated textures for {}/{} stars...",
                    star_idx + 1,
                    n_stars
                );
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
        }
        println!();

        let n_val = ((total_samples as f32) * val_frac) as usize;
        let n_train = total_samples - n_val;

        let mut split_rng = StdRng::seed_from_u64(seed);
        let mut split_indices: Vec<usize> = (0..total_samples).collect();
        split_indices.shuffle(&mut split_rng);

        let train_idx: Vec<usize> = split_indices[..n_train].to_vec();
        let val_idx: Vec<usize> = split_indices[n_train..].to_vec();

        let (train_inputs, train_targets) = gather_rows(&all_inputs, &all_targets, &train_idx);
        let (val_inputs, val_targets) = gather_rows(&all_inputs, &all_targets, &val_idx);
        drop(all_inputs);
        drop(all_targets);

        println!("Train: {} pixels, Val: {} pixels", n_train, n_val);

        let train = SirenDataset {
            inputs_cpu: train_inputs,
            targets_cpu: train_targets,
            norm: norm.clone(),
            n_samples: n_train,
        };
        let val = SirenDataset {
            inputs_cpu: val_inputs,
            targets_cpu: val_targets,
            norm,
            n_samples: n_val,
        };

        Ok((train, val))
    }

    pub fn shuffle(&mut self) {
        self.shuffle_with_seed(crate::runner::DEFAULT_TRAIN_SEED);
    }

    pub fn shuffle_with_seed(&mut self, seed: u64) {
        let n = self.n_samples;
        if n == 0 {
            return;
        }
        let mut shuffle_rng = StdRng::seed_from_u64(seed);
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut shuffle_rng);
        let mut new_inputs = vec![0.0f32; n * SIREN_INPUT_DIM];
        let mut new_targets = vec![0.0f32; n * TARGET_DIM];
        for (new_i, &old_i) in perm.iter().enumerate() {
            let src_inp = old_i * SIREN_INPUT_DIM;
            let dst_inp = new_i * SIREN_INPUT_DIM;
            new_inputs[dst_inp..dst_inp + SIREN_INPUT_DIM]
                .copy_from_slice(&self.inputs_cpu[src_inp..src_inp + SIREN_INPUT_DIM]);
            let src_tgt = old_i * TARGET_DIM;
            let dst_tgt = new_i * TARGET_DIM;
            new_targets[dst_tgt..dst_tgt + TARGET_DIM]
                .copy_from_slice(&self.targets_cpu[src_tgt..src_tgt + TARGET_DIM]);
        }
        self.inputs_cpu = new_inputs;
        self.targets_cpu = new_targets;
    }
}

pub struct PrefetchBatcher {
    inputs: Vec<f32>,
    targets: Vec<f32>,
    n_samples: usize,
    batch_size: usize,
    current: usize,
}

impl PrefetchBatcher {
    pub fn new(dataset: &SirenDataset, batch_size: usize) -> Self {
        Self {
            inputs: dataset.inputs_cpu.clone(),
            targets: dataset.targets_cpu.clone(),
            n_samples: dataset.n_samples,
            batch_size,
            current: 0,
        }
    }

    pub fn next_batch<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Option<(Tensor<B, 2>, Tensor<B, 2>)> {
        if self.current >= self.n_samples {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.n_samples);
        let rows = end - self.current;

        let inp_start = self.current * SIREN_INPUT_DIM;
        let inp_end = end * SIREN_INPUT_DIM;
        let tgt_start = self.current * TARGET_DIM;
        let tgt_end = end * TARGET_DIM;

        let inp_batch: Vec<f32> = self.inputs[inp_start..inp_end].to_vec();
        let tgt_batch: Vec<f32> = self.targets[tgt_start..tgt_end].to_vec();

        self.current = end;

        let inputs =
            Tensor::<B, 2>::from_data(TensorData::new(inp_batch, [rows, SIREN_INPUT_DIM]), device);
        let targets =
            Tensor::<B, 2>::from_data(TensorData::new(tgt_batch, [rows, TARGET_DIM]), device);
        Some((inputs, targets))
    }
}

fn gather_rows(inputs: &[f32], targets: &[f32], indices: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let n = indices.len();
    let mut new_inputs = vec![0.0f32; n * SIREN_INPUT_DIM];
    let mut new_targets = vec![0.0f32; n * TARGET_DIM];

    new_inputs
        .par_chunks_mut(SIREN_INPUT_DIM)
        .zip(indices.par_iter())
        .for_each(|(chunk, &src)| {
            let s = src * SIREN_INPUT_DIM;
            chunk.copy_from_slice(&inputs[s..s + SIREN_INPUT_DIM]);
        });
    new_targets
        .par_chunks_mut(TARGET_DIM)
        .zip(indices.par_iter())
        .for_each(|(chunk, &src)| {
            let s = src * TARGET_DIM;
            chunk.copy_from_slice(&targets[s..s + TARGET_DIM]);
        });

    (new_inputs, new_targets)
}

fn extract_star_params(df: &DataFrame) -> Result<Vec<StarParams>> {
    let bp_rp = extract_f32(df, "bp_rp")?;
    let mag_g = extract_f32(df, "mag_g")?;
    let x = extract_f32(df, "x_pc")?;
    let y = extract_f32(df, "y_pc")?;
    let z = extract_f32(df, "z_pc")?;
    let ruwe = extract_f32(df, "ruwe")?;

    let mg: Vec<f32> = x
        .par_iter()
        .zip(&y)
        .zip(&z)
        .zip(&mag_g)
        .map(|(((xi, yi), zi), &g)| {
            let d = (xi * xi + yi * yi + zi * zi).sqrt().max(1e-6);
            g - 5.0 * d.log10() + 5.0
        })
        .collect();

    let result: Vec<StarParams> = (0..bp_rp.len())
        .filter_map(|i| {
            let c = bp_rp[i];
            let m = mg[i];
            let r = ruwe[i];
            if c.is_finite() && m.is_finite() && r.is_finite() && r >= 0.0 {
                Some(StarParams {
                    bp_rp: c,
                    mg: m,
                    ruwe: r,
                })
            } else {
                None
            }
        })
        .collect();

    Ok(result)
}

fn compute_norm(stars: &[StarParams]) -> SirenNorm {
    let bp_rps: Vec<f32> = stars.iter().map(|s| s.bp_rp).collect();
    let mgs: Vec<f32> = stars.iter().map(|s| s.mg).collect();
    let ruwes: Vec<f32> = stars.iter().map(|s| s.ruwe).collect();

    let (bp_rp_mean, bp_rp_std) = mean_std(&bp_rps);
    let (mg_mean, mg_std) = mean_std(&mgs);
    let (ruwe_mean, ruwe_std) = mean_std(&ruwes);

    SirenNorm {
        bp_rp_mean,
        bp_rp_std,
        mg_mean,
        mg_std,
        ruwe_mean,
        ruwe_std,
    }
}

fn print_norm(norm: &SirenNorm) {
    println!("SIREN normalization parameters:");
    println!(
        "  bp_rp:      mean={:.4}, std={:.4}",
        norm.bp_rp_mean, norm.bp_rp_std
    );
    println!(
        "  M_G:        mean={:.4}, std={:.4}",
        norm.mg_mean, norm.mg_std
    );
    println!(
        "  ruwe:       mean={:.4}, std={:.4}",
        norm.ruwe_mean, norm.ruwe_std
    );
}

fn generate_uv_grid(size: usize) -> Vec<f32> {
    let mut coords = Vec::with_capacity(size * size * 2);
    for y in 0..size {
        let v = -1.0 + 2.0 * (y as f32) / (size.saturating_sub(1)).max(1) as f32;
        for x in 0..size {
            let u = -1.0 + 2.0 * (x as f32) / (size.saturating_sub(1)).max(1) as f32;
            coords.push(u);
            coords.push(v);
        }
    }
    coords
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

fn extract_f32(df: &DataFrame, name: &str) -> Result<Vec<f32>> {
    let s = anyhow::Context::context(df.column(name), format!("column {name} not found"))?;
    let s = anyhow::Context::context(
        s.cast(&DataType::Float64),
        format!("column {name} cast to f64 failed"),
    )?;
    let ca = anyhow::Context::context(s.f64(), format!("column {name} is not f64"))?;
    Ok(ca
        .iter()
        .map(|opt| opt.map(|v| v as f32).unwrap_or(0.0f32))
        .collect())
}

fn read_filtered_parquet(
    parquet_path: &Path,
    max_rows: Option<u64>,
) -> Result<(DataFrame, usize)> {
    println!("Loading parquet: {}", parquet_path.display());
    let path_str = parquet_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 data path"))?;
    // Lazy scan with column projection: SIREN needs only 6 columns.
    let mut lf = anyhow::Context::context(
        LazyFrame::scan_parquet(PlRefPath::from(path_str), Default::default()),
        "failed to scan parquet",
    )?;
    let schema = anyhow::Context::context(lf.collect_schema(), "read parquet schema")?;

    // Canonical-v1 (Stage 4) schema: textures are conditioned on measured
    // photometry/astrometry (bp_rp color, absolute M_G, ruwe) instead of
    // legacy stellar-pipeline columns (st_teff/st_rad/st_mass/st_lum).
    let required_cols: &[&str] = &["x_pc", "y_pc", "z_pc", "bp_rp", "mag_g", "ruwe"];

    for &col_name in required_cols {
        if !schema.contains(col_name) {
            anyhow::bail!(
                "Column '{col_name}' not found in dataset. SIREN training expects the \
                 canonical-v1 schema (x_pc/y_pc/z_pc, bp_rp, mag_g, ruwe)."
            );
        }
    }

    let mut lf = lf.select(required_cols.iter().map(|c| col(*c)).collect::<Vec<_>>());
    for &col_name in required_cols {
        lf = lf.filter(col(col_name).is_not_null());
    }
    let df = anyhow::Context::context(
        lf.filter(
            col("x_pc")
                .abs()
                .lt(lit(10000.0))
                .and(col("y_pc").abs().lt(lit(10000.0)))
                .and(col("z_pc").abs().lt(lit(10000.0)))
                .and(col("bp_rp").gt(lit(-1.0)))
                .and(col("bp_rp").lt(lit(10.0)))
                .and(col("mag_g").gt(lit(0.0)))
                .and(col("mag_g").lt(lit(25.0))),
        )
        .collect(),
        "failed to load filtered rows",
    )?;

    let kept = df.height();
    let df = apply_max_rows(df, max_rows)?;
    let n = df.height();
    println!("Loaded {n} rows ({kept} after filters)");
    Ok((df, n))
}

/// Deterministic systematic sample: every k-th row in file order, at most
/// `cap` rows (file order follows RA-shard assembly, so a stride stays
/// spatially uniform).
fn apply_max_rows(df: DataFrame, max_rows: Option<u64>) -> Result<DataFrame> {
    let cap = match max_rows {
        Some(n) if n > 0 && (n as usize) < df.height() => n as usize,
        _ => return Ok(df),
    };
    let height = df.height();
    let step = height.div_ceil(cap).max(1) as u32;
    let idx: Vec<u32> = (0..height as u32).step_by(step as usize).collect();
    let idx = Series::new("row_idx".into(), idx);
    let idx = idx
        .u32()
        .map_err(|e| anyhow::anyhow!("sample index: {e}"))?;
    anyhow::Context::context(df.take(idx), "max-rows sample")
}

struct BaseColor {
    r: f32,
    g: f32,
    b: f32,
}

/// Base photosphere color from Gaia BP-RP color and absolute G magnitude.
/// Spectral classes are color classes, so bp_rp bins mirror the old Teff bins
/// (O/B < 0.0, A 0.0-0.35, F 0.35-0.6, G 0.6-0.9, K 0.9-1.3/1.9, M > 1.9).
/// Red giants (bright M_G at red colors) get a mild luminous lift.
fn star_base_color(bp_rp: f32, mg: f32) -> BaseColor {
    let (r, g, b) = if bp_rp < 0.0 {
        (0.62, 0.69, 1.0)
    } else if bp_rp < 0.35 {
        let f = (bp_rp - 0.0) / 0.35;
        (0.62 + f * 0.08, 0.69 + f * 0.08, 1.0 + f * -0.05)
    } else if bp_rp < 0.6 {
        let f = (bp_rp - 0.35) / 0.25;
        (0.70 + f * 0.12, 0.77 + f * 0.08, 0.95 + f * 0.0)
    } else if bp_rp < 0.9 {
        let f = (bp_rp - 0.6) / 0.3;
        (0.82 + f * 0.13, 0.85 + f * 0.08, 0.95 + f * -0.05)
    } else if bp_rp < 1.3 {
        let f = (bp_rp - 0.9) / 0.4;
        (1.0, 0.93 + f * 0.07, 0.90 + f * -0.08)
    } else if bp_rp < 1.9 {
        let f = (bp_rp - 1.3) / 0.6;
        (1.0, 0.93 + f * -0.08, 0.90 + f * -0.25)
    } else {
        (1.0, 0.55, 0.35)
    };

    // Red-giant branch: luminous and slightly desaturated vs M dwarfs.
    let giant_lift = if mg < 3.5 && bp_rp > 0.8 { 0.03 } else { 0.0 };

    let bp_tint: f32 = (bp_rp - 0.5) / 4.0;
    BaseColor {
        r: (r.clamp(0.0, 1.0) - bp_tint * 0.05 + giant_lift).clamp(0.0, 1.0),
        g: (g.clamp(0.0, 1.0) - bp_tint * 0.03 + giant_lift).clamp(0.0, 1.0),
        b: (b.clamp(0.0, 1.0) + bp_tint * 0.05 + giant_lift).clamp(0.0, 1.0),
    }
}

struct SpotParams {
    spot_contrast: f32,
    spot_frequency: f32,
    spot_size: f32,
    granulation_amplitude: f32,
    granulation_frequency: f32,
    limb_darkening_coeff: f32,
    corona_intensity: f32,
}

/// Activity/granulation regimes from color class; red giants (bright M_G at
/// red colors) get larger, lower-contrast granulation than dwarfs.
fn compute_spot_params(bp_rp: f32, mg: f32) -> SpotParams {
    let (spot_contrast, spot_freq, spot_size, gran_amp, gran_freq) = if bp_rp < 0.6 {
        (0.02, 0.5, 0.03, 0.01, 30.0)
    } else if bp_rp < 1.0 {
        (0.15, 1.5, 0.08, 0.04, 15.0)
    } else if bp_rp < 1.6 {
        (0.25, 2.5, 0.12, 0.08, 10.0)
    } else {
        (0.35, 3.5, 0.18, 0.12, 6.0)
    };

    let limb = if bp_rp < 0.35 {
        0.2
    } else if bp_rp < 1.0 {
        0.5
    } else {
        0.7
    };
    let corona = if bp_rp < 0.5 {
        0.15
    } else if bp_rp < 1.1 {
        0.05
    } else {
        0.01
    };

    // Giants: big granulation cells, muted spots.
    let is_giant = mg < 3.5 && bp_rp > 0.8;
    let (gran_amp, gran_freq, spot_contrast) = if is_giant {
        (gran_amp * 2.0, gran_freq * 0.5, spot_contrast * 0.7)
    } else {
        (gran_amp, gran_freq, spot_contrast)
    };

    let rad_factor = (bp_rp / 2.0 - 0.5).max(0.0);
    let spot_contrast = (spot_contrast + rad_factor * 0.1).min(0.5);

    SpotParams {
        spot_contrast,
        spot_frequency: spot_freq,
        spot_size,
        granulation_amplitude: gran_amp,
        granulation_frequency: gran_freq,
        limb_darkening_coeff: limb,
        corona_intensity: corona,
    }
}

fn generate_pixel(
    u: f32,
    v: f32,
    base: &BaseColor,
    params: &SpotParams,
    seed: u64,
) -> (f32, f32, f32) {
    let r_sq = u * u + v * v;
    let disk_mask = if r_sq <= 1.0 { 1.0 } else { 0.0 };
    let limb_factor = if r_sq < 1.0 {
        let mu = (1.0 - r_sq).sqrt();
        1.0 - params.limb_darkening_coeff * (1.0 - mu)
    } else {
        0.0
    };

    let gran = granulation_noise(u, v, seed, params.granulation_frequency);
    let spot = sunspot_pattern(
        u,
        v,
        seed,
        params.spot_contrast,
        params.spot_frequency,
        params.spot_size,
    );

    let inside = disk_mask * limb_factor;

    let r = base.r * (1.0 + params.granulation_amplitude * gran) * (1.0 - spot);
    let g = base.g * (1.0 + params.granulation_amplitude * gran * 0.8) * (1.0 - spot * 1.1);
    let b = base.b * (1.0 + params.granulation_amplitude * gran * 0.5) * (1.0 - spot * 0.7);

    let edge_glow = if (0.8..1.2).contains(&r_sq) {
        let t = 1.0 - (r_sq - 0.8) / 0.4;
        t * t * params.corona_intensity
    } else {
        0.0
    };

    let r_out = inside * r + (1.0 - inside) * edge_glow * base.b * 0.3 + edge_glow * base.r * 0.1;
    let g_out = inside * g + (1.0 - inside) * edge_glow * base.b * 0.15 + edge_glow * base.g * 0.05;
    let b_out = inside * b + (1.0 - inside) * edge_glow * 0.5 + edge_glow * base.b * 0.3;

    (
        r_out.clamp(0.0, 1.0),
        g_out.clamp(0.0, 1.0),
        b_out.clamp(0.0, 1.0),
    )
}

fn granulation_noise(u: f32, v: f32, seed: u64, freq: f32) -> f32 {
    let n1 = (u * freq + v * freq * 0.7 + hash_float(seed, 0) * std::f32::consts::TAU).sin();
    let n2 = (u * freq * 1.3 - v * freq * 0.9 + hash_float(seed, 1) * std::f32::consts::TAU).sin();
    let n3 = (u * freq * 0.7 + v * freq * 1.1 + hash_float(seed, 2) * std::f32::consts::TAU).sin();
    (n1 + n2 + n3) / 3.0
}

fn sunspot_pattern(u: f32, v: f32, seed: u64, contrast: f32, freq: f32, size: f32) -> f32 {
    let mut total = 0.0f32;
    for i in 0..3 {
        let cx = hash_float(seed, i * 3) * 0.8;
        let cy = hash_float(seed, i * 3 + 1) * 0.8;
        let spot_r = size * (0.5 + hash_float(seed, i * 3 + 2) * 0.5);

        let du = u - cx;
        let dv = v - cy;
        let dist_sq = du * du + dv * dv;
        let spot_val = if dist_sq < spot_r * spot_r {
            1.0 - (dist_sq / (spot_r * spot_r)).sqrt()
        } else {
            0.0
        };
        total += contrast * spot_val / freq.max(1.0);
    }
    total.min(1.0)
}

fn hash_float(seed: u64, idx: u32) -> f32 {
    let mut s = seed.wrapping_add(idx as u64);
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((s >> 33) as f32) / (1u64 << 31) as f32 * 2.0 - 1.0
}
