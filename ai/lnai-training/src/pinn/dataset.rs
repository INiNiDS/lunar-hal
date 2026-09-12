use anyhow::Result;
use burn::prelude::*;
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
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
    pub fn load(parquet_path: &Path, device: &B::Device, max_rows: Option<u64>, tiles: Option<String>) -> Result<Self> {
        let (df, n) = read_filtered_parquet(parquet_path, max_rows, tiles)?;
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
        max_rows: Option<u64>,
        tiles: Option<String>,
    ) -> Result<Self> {
        let (df, n) = read_filtered_parquet(parquet_path, max_rows, tiles)?;
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

/// Input schema flavor: legacy `clean` output vs canonical-v1 enriched with
/// Gaia astrophysical parameters (`lnaicli enrich-stellar`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchemaKind {
    Legacy,
    ApEnriched,
}

fn detect_schema(df: &DataFrame) -> Result<SchemaKind> {
    if df.column("st_teff").is_ok() {
        Ok(SchemaKind::Legacy)
    } else if df.column("teff_gspphot").is_ok() {
        Ok(SchemaKind::ApEnriched)
    } else {
        anyhow::bail!(
            "PINN training needs stellar targets: legacy clean schema \
             (x_pc/y_pc/z_pc, bp_rp, g_mag, st_teff/st_rad/st_mass/st_lum) or \
             canonical-v1 + AP enrichment (x_pc/y_pc/z_pc, bp_rp, mag_g, \
             teff_gspphot/radius_gspphot/mass_flame/lum_flame via \
             `lnaicli enrich-stellar`)."
        );
    }
}

fn read_filtered_parquet(
    parquet_path: &Path,
    max_rows: Option<u64>,
    tiles: Option<String>,
) -> Result<(DataFrame, usize)> {
    println!("Loading parquet: {}", parquet_path.display());
    let path_str = parquet_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 data path"))?;
    // Lazy scan with column projection: the enriched canonical file is ~8GB
    // across 27 columns, but PINN needs only 9 — never materialize the rest.
    let mut lf = anyhow::Context::context(
        LazyFrame::scan_parquet(PlRefPath::from(path_str), Default::default()),
        "failed to scan parquet",
    )?;

    let schema = anyhow::Context::context(lf.collect_schema(), "read parquet schema")?;
    let kind = if schema.contains("st_teff") {
        SchemaKind::Legacy
    } else if schema.contains("teff_gspphot") {
        SchemaKind::ApEnriched
    } else {
        anyhow::bail!(
            "PINN training needs stellar targets: legacy clean schema \
             (x_pc/y_pc/z_pc, bp_rp, g_mag, st_teff/st_rad/st_mass/st_lum) or \
             canonical-v1 + AP enrichment (x_pc/y_pc/z_pc, bp_rp, mag_g, \
             teff_gspphot/radius_gspphot/mass_flame/lum_flame via \
             `lnaicli enrich-stellar`)."
        );
    };
    let (mag_col, required_cols): (&str, &[&str]) = match kind {
        SchemaKind::Legacy => (
            "g_mag",
            &[
                "x_pc", "y_pc", "z_pc", "bp_rp", "g_mag", "st_teff", "st_rad", "st_mass",
                "st_lum",
            ],
        ),
        SchemaKind::ApEnriched => (
            "mag_g",
            &[
                "x_pc",
                "y_pc",
                "z_pc",
                "bp_rp",
                "mag_g",
                "teff_gspphot",
                "radius_gspphot",
                "mass_flame",
                "lum_flame",
            ],
        ),
    };

    for &col_name in required_cols {
        if !schema.contains(col_name) {
            anyhow::bail!(
                "Column '{col_name}' not found in dataset ({kind:?} schema)."
            );
        }
    }

    let mut proj: Vec<Expr> = required_cols.iter().map(|c| col(*c)).collect();
    // Tile filtering needs the tile column, which legacy clean files lack.
    if tiles.as_deref().is_some_and(|t| !t.is_empty()) {
        if !schema.contains("spatial_tile") {
            anyhow::bail!("--tiles needs the spatial_tile column (canonical schema).");
        }
        proj.push(col("spatial_tile"));
    }
    let mut lf = lf.select(proj);
    for &col_name in required_cols {
        lf = lf.filter(col(col_name).is_not_null());
    }
    // Optional spatial-tile subset: shard the sky without loading the rest.
    if let Some(wanted) = tiles.as_deref() {
        let wanted: Vec<&str> = wanted
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if !wanted.is_empty() {
            println!("PINN tile subset: {} tiles", wanted.len());
            let mut pred: Option<Expr> = None;
            for t in wanted {
                let e = col("spatial_tile").eq(lit(t.to_string()));
                pred = Some(match pred {
                    None => e,
                    Some(p) => p.or(e),
                });
            }
            if let Some(p) = pred {
                lf = lf.filter(p);
            }
        }
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
                .and(col(mag_col).gt(lit(0.0)))
                .and(col(mag_col).lt(lit(25.0))),
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
/// `cap` rows. File order follows RA-shard assembly, so a stride stays
/// spatially uniform (unlike a head slice).
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
        let kind = detect_schema(df)?;
        let (mag_name, teff_name, rad_name, mass_name, lum_name) = match kind {
            SchemaKind::Legacy => ("g_mag", "st_teff", "st_rad", "st_mass", "st_lum"),
            SchemaKind::ApEnriched => (
                "mag_g",
                "teff_gspphot",
                "radius_gspphot",
                "mass_flame",
                "lum_flame",
            ),
        };
        let x = extract_f32(df, "x_pc")?;
        let y = extract_f32(df, "y_pc")?;
        let z = extract_f32(df, "z_pc")?;
        let bp_rp = extract_f32(df, "bp_rp")?;
        let g_mag = extract_f32(df, mag_name)?;
        let teff = extract_f32(df, teff_name)?;
        let rad = extract_f32(df, rad_name)?;
        let mass = extract_f32(df, mass_name)?;
        let lum = extract_f32(df, lum_name)?;

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
    let s = anyhow::Context::context(df.column(name), format!("column {name} not found"))?;
    let s = anyhow::Context::context(
        s.cast(&DataType::Float64),
        format!("column {name} cast to f64 failed"),
    )?;
    let ca = anyhow::Context::context(s.f64(), format!("column {name} is not f64"))?;
    let values: Vec<f32> = ca
        .iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ap_frame() -> DataFrame {
        df![
            "x_pc" => [100.0f32, 200.0],
            "y_pc" => [0.0f32, 0.0],
            "z_pc" => [0.0f32, 0.0],
            "bp_rp" => [1.0f32, 0.8],
            "mag_g" => [10.0f32, 9.0],
            "teff_gspphot" => [5778.0, 6000.0],
            "radius_gspphot" => [1.0, 1.1],
            "mass_flame" => [1.0, 1.05],
            "lum_flame" => [1.0, 1.5],
        ]
        .expect("frame")
    }

    #[test]
    fn detect_schema_prefers_legacy_but_finds_ap() {
        assert_eq!(
            detect_schema(&ap_frame()).expect("detect"),
            SchemaKind::ApEnriched
        );
        let legacy = df![
            "st_teff" => [5778.0],
            "teff_gspphot" => [5778.0],
        ]
        .expect("frame");
        assert_eq!(
            detect_schema(&legacy).expect("detect"),
            SchemaKind::Legacy
        );
        let bare = df!["x_pc" => [1.0f32]].expect("frame");
        assert!(detect_schema(&bare).is_err());
    }

    #[test]
    fn extract_ap_columns_shapes_and_mg() {
        let df = ap_frame();
        let raw = RawColumns::extract(&df).expect("extract");
        assert_eq!(raw.x.len(), 2);
        // M_G = g - 5*log10(d) + 5; d=100 -> M_G = g - 5.
        assert!((raw.mg[0] - 5.0).abs() < 1e-4);
        assert!((raw.mg[1] - (9.0 - 5.0 * 200f32.log10() + 5.0)).abs() < 1e-4);
        assert!((raw.log_teff[0] - 5778f32.log10()).abs() < 1e-5);
        assert_eq!(raw.log_mass.len(), 2);
    }

    fn write_parquet(
        df: &mut DataFrame,
        name: &str,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join(name);
        let file = std::fs::File::create(&path).expect("create");
        ParquetWriter::new(file).finish(df).expect("write");
        (dir, path)
    }

    fn ap_file_frame() -> DataFrame {
        df![
            "x_pc" => [100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0],
            "y_pc" => [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0],
            "z_pc" => [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0],
            "bp_rp" => [1.0f32, 0.8, 1.2, 0.9, 1.1, 1.0],
            "mag_g" => [10.0f32, 9.0, 11.0, 10.5, 9.5, 10.0],
            "teff_gspphot" => [Some(5778.0), Some(6000.0), None, Some(5000.0), Some(4500.0), Some(6200.0)],
            "radius_gspphot" => [Some(1.0), Some(1.1), Some(1.0), Some(0.9), Some(2.0), Some(1.2)],
            "mass_flame" => [Some(1.0), Some(1.05), Some(1.0), Some(0.8), Some(1.3), Some(1.1)],
            "lum_flame" => [Some(1.0), Some(1.5), Some(1.0), Some(0.5), Some(5.0), Some(2.0)],
            "spatial_tile" => ["tileA", "tileA", "tileA", "tileB", "tileB", "tileB"],
        ]
        .expect("frame")
    }

    #[test]
    fn read_ap_file_filters_tiles_and_nulls() {
        let (_dir, path) = write_parquet(&mut ap_file_frame(), "ap.parquet");
        // tileA has 3 rows, one with null teff -> 2 survive.
        let (df, n) =
            read_filtered_parquet(&path, None, Some("tileA".to_string())).expect("read");
        assert_eq!(n, 2);
        assert_eq!(df.height(), 2);
        // No tile filter: 5 valid rows of 6.
        let (_, n_all) = read_filtered_parquet(&path, None, None).expect("read");
        assert_eq!(n_all, 5);
    }

    #[test]
    fn read_ap_file_max_rows_strides() {
        let (_dir, path) = write_parquet(&mut ap_file_frame(), "ap.parquet");
        // 5 valid rows capped at 2 -> stride 3 -> filtered rows 0 and 3.
        let (df, n) = read_filtered_parquet(&path, Some(2), None).expect("read");
        assert_eq!(n, 2);
        let bp: Vec<f32> = df
            .column("bp_rp")
            .unwrap()
            .f32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(bp, vec![1.0, 1.1]);
    }

    #[test]
    fn tiles_on_legacy_without_tile_column_bails() {
        let mut legacy = df![
            "x_pc" => [100.0f32],
            "y_pc" => [0.0f32],
            "z_pc" => [0.0f32],
            "bp_rp" => [1.0f32],
            "g_mag" => [10.0f32],
            "st_teff" => [5778.0],
            "st_rad" => [1.0],
            "st_mass" => [1.0],
            "st_lum" => [1.0],
        ]
        .expect("frame");
        let (_dir, path) = write_parquet(&mut legacy, "legacy.parquet");
        assert!(read_filtered_parquet(&path, None, Some("tileA".to_string())).is_err());
        let (_, n) = read_filtered_parquet(&path, None, None).expect("legacy read");
        assert_eq!(n, 1);
    }
}
