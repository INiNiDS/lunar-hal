use anyhow::Result;
use burn::prelude::*;
use lnai_models::compute_knn_adjacency;
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;

pub const NODE_FEATURE_DIM: usize = 8;
pub const VELOCITY_DIM: usize = 3;
pub const DEFAULT_KNN_K: usize = 8;
pub const DEFAULT_MAX_GROUP: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnNormParams {
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
    pub mag_bp_mean: f32,
    pub mag_bp_std: f32,
    pub mag_rp_mean: f32,
    pub mag_rp_std: f32,
    pub ruwe_mean: f32,
    pub ruwe_std: f32,
    pub vx_mean: f32,
    pub vx_std: f32,
    pub vy_mean: f32,
    pub vy_std: f32,
    pub vz_mean: f32,
    pub vz_std: f32,
}

#[derive(Debug, Clone)]
pub struct StarGroup {
    pub coords: Vec<[f32; 3]>,
    pub node_features: Vec<[f32; NODE_FEATURE_DIM]>,
    pub velocities: Vec<[f32; VELOCITY_DIM]>,
    pub adjacency: Vec<Vec<f32>>,
}

impl StarGroup {
    pub fn n_nodes(&self) -> usize {
        self.coords.len()
    }
}

pub struct GnnDataset {
    pub groups: Arc<Vec<StarGroup>>, // Wrapped in Arc for fast reference cloning
    pub norm: GnnNormParams,
    indices: Vec<usize>,
}

impl GnnDataset {
    pub fn load(
        parquet_path: &Path,
        knn_k: usize,
        max_group_size: usize,
        radius_pc: f32,
        max_rows: Option<u64>,
        tiles: Option<String>,
    ) -> Result<Self> {
        Self::load_with_seed(
            parquet_path,
            knn_k,
            max_group_size,
            radius_pc,
            crate::runner::DEFAULT_TRAIN_SEED,
            max_rows,
            tiles,
        )
    }

    pub fn load_with_seed(
        parquet_path: &Path,
        knn_k: usize,
        max_group_size: usize,
        radius_pc: f32,
        seed: u64,
        max_rows: Option<u64>,
        tiles: Option<String>,
    ) -> Result<Self> {
        let (groups, norm) = build_groups_from_parquet(
            parquet_path,
            knn_k,
            max_group_size,
            radius_pc,
            seed,
            max_rows,
            tiles,
        )?;
        let n = groups.len();
        println!("Built {} star groups from parquet", n);

        Ok(Self {
            groups: Arc::new(groups),
            norm,
            indices: (0..n).collect(),
        })
    }

    pub fn load_with_norm(
        parquet_path: &Path,
        norm: GnnNormParams,
        knn_k: usize,
        max_group_size: usize,
        radius_pc: f32,
        max_rows: Option<u64>,
        tiles: Option<String>,
    ) -> Result<Self> {
        Self::load_with_norm_and_seed(
            parquet_path,
            norm,
            knn_k,
            max_group_size,
            radius_pc,
            crate::runner::DEFAULT_TRAIN_SEED,
            max_rows,
            tiles,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_with_norm_and_seed(
        parquet_path: &Path,
        norm: GnnNormParams,
        knn_k: usize,
        max_group_size: usize,
        radius_pc: f32,
        seed: u64,
        max_rows: Option<u64>,
        tiles: Option<String>,
    ) -> Result<Self> {
        let (groups, _) = build_groups_from_parquet_with_norm(
            parquet_path,
            &norm,
            knn_k,
            max_group_size,
            radius_pc,
            seed,
            max_rows,
            tiles,
        )?;
        let n = groups.len();
        println!("Built {} star groups (external norm)", n);

        Ok(Self {
            groups: Arc::new(groups),
            norm,
            indices: (0..n).collect(),
        })
    }

    pub fn split(self, val_frac: f32) -> (Self, Self) {
        self.split_with_seed(val_frac, crate::runner::DEFAULT_TRAIN_SEED)
    }

    pub fn split_with_seed(self, val_frac: f32, seed: u64) -> (Self, Self) {
        let n = self.groups.len();
        let n_val = ((n as f32) * val_frac) as usize;
        let n_train = n - n_val;

        let mut split_indices: Vec<usize> = (0..n).collect();
        split_indices.shuffle(&mut StdRng::seed_from_u64(seed));

        let train_idx = &split_indices[..n_train];
        let val_idx = &split_indices[n_train..];

        let train_groups: Vec<StarGroup> =
            train_idx.iter().map(|&i| self.groups[i].clone()).collect();
        let val_groups: Vec<StarGroup> = val_idx.iter().map(|&i| self.groups[i].clone()).collect();

        println!("Train groups: {}, Validation groups: {}", n_train, n_val);

        (
            GnnDataset {
                groups: Arc::new(train_groups),
                norm: self.norm.clone(),
                indices: (0..n_train).collect(),
            },
            GnnDataset {
                groups: Arc::new(val_groups),
                norm: self.norm,
                indices: (0..n_val).collect(),
            },
        )
    }

    pub fn shuffle(&mut self) {
        self.shuffle_with_seed(crate::runner::DEFAULT_TRAIN_SEED);
    }

    pub fn shuffle_with_seed(&mut self, seed: u64) {
        self.indices.shuffle(&mut StdRng::seed_from_u64(seed));
    }
}

struct PrefetchBatchedItem {
    nodes_data: Vec<f32>,
    adj_data: Vec<f32>,
    target_data: Vec<f32>,
    total_nodes: usize,
}

pub struct PrefetchBatchedBatcher {
    receiver: mpsc::Receiver<PrefetchBatchedItem>,
}

impl PrefetchBatchedBatcher {
    pub fn new(dataset: &GnnDataset, max_nodes_per_batch: usize) -> Self {
        let groups = Arc::clone(&dataset.groups);
        let indices = dataset.indices.clone();
        let n_groups = indices.len();

        let (tx, rx) = mpsc::sync_channel(32);

        std::thread::spawn(move || {
            let mut idx = 0;
            while idx < n_groups {
                let mut nodes_data = Vec::new();
                let mut targets_data = Vec::new();
                let mut total_nodes = 0usize;
                let mut batch_groups: Vec<usize> = Vec::new();

                while idx < n_groups {
                    let gi = indices[idx];
                    let group_size = groups[gi].n_nodes();
                    if total_nodes + group_size > max_nodes_per_batch && total_nodes > 0 {
                        break;
                    }
                    batch_groups.push(gi);
                    total_nodes += group_size;
                    idx += 1;
                }

                if total_nodes == 0 {
                    break;
                }

                nodes_data.reserve(total_nodes * NODE_FEATURE_DIM);
                targets_data.reserve(total_nodes * VELOCITY_DIM);
                let mut block_adj = vec![0.0f32; total_nodes * total_nodes];

                let mut row_off = 0usize;
                let mut col_off = 0usize;

                for &gi in &batch_groups {
                    let group = &groups[gi];
                    let n = group.n_nodes();

                    for feat in &group.node_features {
                        nodes_data.extend_from_slice(feat);
                    }
                    for vel in &group.velocities {
                        targets_data.extend_from_slice(vel);
                    }

                    for r in 0..n {
                        for c in 0..n {
                            block_adj[(row_off + r) * total_nodes + col_off + c] =
                                group.adjacency[r][c];
                        }
                    }

                    row_off += n;
                    col_off += n;
                }

                if tx
                    .send(PrefetchBatchedItem {
                        nodes_data,
                        adj_data: block_adj,
                        target_data: targets_data,
                        total_nodes,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });

        PrefetchBatchedBatcher { receiver: rx }
    }

    pub fn next_batch<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Option<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)> {
        self.receiver.recv().ok().map(|batch| {
            let n = batch.total_nodes;
            let nodes = Tensor::<B, 2>::from_data(
                TensorData::new(batch.nodes_data, [n, NODE_FEATURE_DIM]),
                device,
            );
            let adj = Tensor::<B, 2>::from_data(TensorData::new(batch.adj_data, [n, n]), device);
            let targets = Tensor::<B, 2>::from_data(
                TensorData::new(batch.target_data, [n, VELOCITY_DIM]),
                device,
            );
            (nodes, adj, targets)
        })
    }
}

fn build_groups_from_parquet(
    path: &Path,
    knn_k: usize,
    max_group_size: usize,
    radius_pc: f32,
    seed: u64,
    max_rows: Option<u64>,
    tiles: Option<String>,
) -> Result<(Vec<StarGroup>, GnnNormParams)> {
    let df = read_gnn_parquet(path, max_rows, tiles)?;

    let x = extract_f32(&df, "x_pc")?;
    let y = extract_f32(&df, "y_pc")?;
    let z = extract_f32(&df, "z_pc")?;
    let bp_rp = extract_f32(&df, "bp_rp")?;
    let mag_g = extract_f32(&df, "mag_g")?;
    let mag_bp = extract_f32(&df, "mag_bp")?;
    let mag_rp = extract_f32(&df, "mag_rp")?;
    let ruwe = extract_f32(&df, "ruwe")?;
    let vx = extract_f32(&df, "vx_kms")?;
    let vy = extract_f32(&df, "vy_kms")?;
    let vz = extract_f32(&df, "vz_kms")?;

    let n = df.height();

    // Absolute G magnitude from apparent mag + Cartesian distance (pc).
    let mg: Vec<f32> = (0..n)
        .map(|i| {
            let d = (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt().max(1e-6);
            mag_g[i] - 5.0 * d.log10() + 5.0
        })
        .collect();

    let (x_m, x_s) = mean_std(&x);
    let (y_m, y_s) = mean_std(&y);
    let (z_m, z_s) = mean_std(&z);
    let (bp_rp_m, bp_rp_s) = mean_std(&bp_rp);
    let (mg_m, mg_s) = mean_std(&mg);
    let (mag_bp_m, mag_bp_s) = mean_std(&mag_bp);
    let (mag_rp_m, mag_rp_s) = mean_std(&mag_rp);
    let (ruwe_m, ruwe_s) = mean_std(&ruwe);
    let (vx_m, vx_s) = mean_std(&vx);
    let (vy_m, vy_s) = mean_std(&vy);
    let (vz_m, vz_s) = mean_std(&vz);

    let norm = GnnNormParams {
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
        mag_bp_mean: mag_bp_m,
        mag_bp_std: mag_bp_s,
        mag_rp_mean: mag_rp_m,
        mag_rp_std: mag_rp_s,
        ruwe_mean: ruwe_m,
        ruwe_std: ruwe_s,
        vx_mean: vx_m,
        vx_std: vx_s,
        vy_mean: vy_m,
        vy_std: vy_s,
        vz_mean: vz_m,
        vz_std: vz_s,
    };

    let groups = build_star_groups(&GroupBuildConfig {
        features: StarFeatures {
            x: &x,
            y: &y,
            z: &z,
            bp_rp: &bp_rp,
            mg: &mg,
            mag_bp: &mag_bp,
            mag_rp: &mag_rp,
            ruwe: &ruwe,
            vx: &vx,
            vy: &vy,
            vz: &vz,
        },
        norm: &norm,
        knn_k,
        max_group_size,
        radius_pc,
        seed,
    });

    Ok((groups, norm))
}

fn build_groups_from_parquet_with_norm(
    path: &Path,
    norm: &GnnNormParams,
    knn_k: usize,
    max_group_size: usize,
    radius_pc: f32,
    seed: u64,
    max_rows: Option<u64>,
    tiles: Option<String>,
) -> Result<(Vec<StarGroup>, GnnNormParams)> {
    let df = read_gnn_parquet(path, max_rows, tiles)?;

    let x = extract_f32(&df, "x_pc")?;
    let y = extract_f32(&df, "y_pc")?;
    let z = extract_f32(&df, "z_pc")?;
    let bp_rp = extract_f32(&df, "bp_rp")?;
    let mag_g = extract_f32(&df, "mag_g")?;
    let mag_bp = extract_f32(&df, "mag_bp")?;
    let mag_rp = extract_f32(&df, "mag_rp")?;
    let ruwe = extract_f32(&df, "ruwe")?;
    let vx = extract_f32(&df, "vx_kms")?;
    let vy = extract_f32(&df, "vy_kms")?;
    let vz = extract_f32(&df, "vz_kms")?;

    let n = df.height();

    // Absolute G magnitude from apparent mag + Cartesian distance (pc).
    let mg: Vec<f32> = (0..n)
        .map(|i| {
            let d = (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt().max(1e-6);
            mag_g[i] - 5.0 * d.log10() + 5.0
        })
        .collect();

    let groups = build_star_groups(&GroupBuildConfig {
        features: StarFeatures {
            x: &x,
            y: &y,
            z: &z,
            bp_rp: &bp_rp,
            mg: &mg,
            mag_bp: &mag_bp,
            mag_rp: &mag_rp,
            ruwe: &ruwe,
            vx: &vx,
            vy: &vy,
            vz: &vz,
        },
        norm,
        knn_k,
        max_group_size,
        radius_pc,
        seed,
    });

    Ok((groups, norm.clone()))
}

struct StarFeatures<'a> {
    x: &'a [f32],
    y: &'a [f32],
    z: &'a [f32],
    bp_rp: &'a [f32],
    mg: &'a [f32],
    mag_bp: &'a [f32],
    mag_rp: &'a [f32],
    ruwe: &'a [f32],
    vx: &'a [f32],
    vy: &'a [f32],
    vz: &'a [f32],
}

struct GroupBuildConfig<'a> {
    features: StarFeatures<'a>,
    norm: &'a GnnNormParams,
    knn_k: usize,
    max_group_size: usize,
    radius_pc: f32,
    seed: u64,
}

fn build_star_groups(config: &GroupBuildConfig<'_>) -> Vec<StarGroup> {
    let StarFeatures {
        x,
        y,
        z,
        bp_rp,
        mg,
        mag_bp,
        mag_rp,
        ruwe,
        vx,
        vy,
        vz,
    } = config.features;
    let norm = config.norm;
    let knn_k = config.knn_k;
    let max_group_size = config.max_group_size;
    let radius_pc = config.radius_pc;
    let n = x.len();
    let mut assigned = vec![false; n];
    let mut groups = Vec::new();

    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut StdRng::seed_from_u64(config.seed));

    // Spatial hash grid (cell = search radius): neighbor lookup is O(1)
    // amortized instead of O(n) per seed — required at canonical scale.
    // With cell == radius, every point within `radius_pc` of the seed is
    // guaranteed to sit in the 27 cells around it, and ascending candidate
    // order reproduces the exact output of the old full scan.
    let cell = radius_pc;
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        let key = (
            (xi / cell).floor() as i32,
            (yi / cell).floor() as i32,
            (z[i] / cell).floor() as i32,
        );
        grid.entry(key).or_default().push(i);
    }

    for &seed in &order {
        if assigned[seed] {
            continue;
        }

        let sx = x[seed];
        let sy = y[seed];
        let sz = z[seed];
        let r2 = radius_pc * radius_pc;

        let mut candidates: Vec<usize> = Vec::new();
        let cx = (sx / cell).floor() as i32;
        let cy = (sy / cell).floor() as i32;
        let cz = (sz / cell).floor() as i32;
        for ox in -1..=1 {
            for oy in -1..=1 {
                for oz in -1..=1 {
                    if let Some(bucket) = grid.get(&(cx + ox, cy + oy, cz + oz)) {
                        candidates.extend_from_slice(bucket);
                    }
                }
            }
        }
        candidates.sort_unstable();

        let mut members: Vec<usize> = Vec::new();
        for j in candidates {
            if assigned[j] {
                continue;
            }
            let dx = x[j] - sx;
            let dy = y[j] - sy;
            let dz = z[j] - sz;
            if dx * dx + dy * dy + dz * dz <= r2 {
                members.push(j);
                if members.len() >= max_group_size {
                    break;
                }
            }
        }

        if members.len() < 2 {
            continue;
        }

        for &m in &members {
            assigned[m] = true;
        }

        let coords: Vec<[f32; 3]> = members.iter().map(|&i| [x[i], y[i], z[i]]).collect();

        let node_features: Vec<[f32; NODE_FEATURE_DIM]> = members
            .iter()
            .map(|&i| {
                [
                    (x[i] - norm.x_mean) / norm.x_std,
                    (y[i] - norm.y_mean) / norm.y_std,
                    (z[i] - norm.z_mean) / norm.z_std,
                    (bp_rp[i] - norm.bp_rp_mean) / norm.bp_rp_std,
                    (mg[i] - norm.mg_mean) / norm.mg_std,
                    (mag_bp[i] - norm.mag_bp_mean) / norm.mag_bp_std,
                    (mag_rp[i] - norm.mag_rp_mean) / norm.mag_rp_std,
                    (ruwe[i] - norm.ruwe_mean) / norm.ruwe_std,
                ]
            })
            .collect();

        let velocities: Vec<[f32; VELOCITY_DIM]> = members
            .iter()
            .map(|&i| {
                [
                    (vx[i] - norm.vx_mean) / norm.vx_std,
                    (vy[i] - norm.vy_mean) / norm.vy_std,
                    (vz[i] - norm.vz_mean) / norm.vz_std,
                ]
            })
            .collect();

        let adjacency = compute_knn_adjacency(&coords, knn_k);

        groups.push(StarGroup {
            coords,
            node_features,
            velocities,
            adjacency,
        });
    }

    println!(
        "Grouped {} / {} stars into {} groups",
        assigned.iter().filter(|&&a| a).count(),
        n,
        groups.len()
    );

    groups
}

fn read_gnn_parquet(path: &Path, max_rows: Option<u64>, tiles: Option<String>) -> Result<DataFrame> {
    println!("Loading parquet: {}", path.display());
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 data path"))?;
    // Lazy scan with column projection: the canonical file is gigabytes wide,
    // but GNN needs only 12 columns — never materialize the rest.
    let mut lf = anyhow::Context::context(
        LazyFrame::scan_parquet(PlRefPath::from(path_str), Default::default()),
        "failed to scan parquet",
    )?;
    let schema = anyhow::Context::context(lf.collect_schema(), "read parquet schema")?;

    // Canonical-v1 (Stage 4) schema: positions + photometry + full 3D velocities.
    // radial_velocity_kms must be present (non-null): rows without a measured RV
    // would otherwise carry an RV=0 assumption baked into vx/vy/vz
    // (see lnai-data clean.rs), which must not become a training target.
    let required_cols: &[&str] = &[
        "x_pc",
        "y_pc",
        "z_pc",
        "bp_rp",
        "mag_g",
        "mag_bp",
        "mag_rp",
        "ruwe",
        "radial_velocity_kms",
        "vx_kms",
        "vy_kms",
        "vz_kms",
    ];

    for &col_name in required_cols {
        if !schema.contains(col_name) {
            anyhow::bail!(
                "Column '{col_name}' not found in dataset. GNN training expects the \
                 canonical-v1 schema (x_pc/y_pc/z_pc, bp_rp, mag_g/mag_bp/mag_rp, ruwe, \
                 radial_velocity_kms, vx_kms/vy_kms/vz_kms)."
            );
        }
    }

    let mut proj: Vec<Expr> = required_cols.iter().map(|c| col(*c)).collect();
    proj.push(col("spatial_tile"));
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
            println!("GNN tile subset: {} tiles", wanted.len());
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
                .and(col("mag_g").gt(lit(0.0)))
                .and(col("mag_g").lt(lit(25.0))),
        )
        .collect(),
        "failed to load filtered rows",
    )?;

    let kept = df.height();
    let df = apply_max_rows(df, max_rows)?;
    println!(
        "Loaded {} rows with velocity data ({} after filters)",
        df.height(),
        kept
    );
    Ok(df)
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

pub fn extract_f32(df: &DataFrame, name: &str) -> Result<Vec<f32>> {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    fn canonical_like_frame() -> DataFrame {
        df![
            "x_pc" => [0.0f32, 10.0, 20.0, 1000.0],
            "y_pc" => [0.0f32, 0.0, 0.0, 0.0],
            "z_pc" => [0.0f32, 0.0, 0.0, 0.0],
            "bp_rp" => [1.0f32, 0.8, 1.1, 0.9],
            "mag_g" => [10.0f32, 9.0, 11.0, 8.0],
            "mag_bp" => [10.5f32, 9.4, 11.6, 8.4],
            "mag_rp" => [9.5f32, 8.6, 10.5, 7.5],
            "ruwe" => [1.0f32, 1.0, 1.1, 1.0],
            "radial_velocity_kms" => [Some(5.0), None, Some(-3.0), Some(10.0)],
            "vx_kms" => [1.0f32, 2.0, 3.0, 4.0],
            "vy_kms" => [0.0f32, 0.0, 0.0, 0.0],
            "vz_kms" => [0.0f32, 0.0, 0.0, 0.0],
            "spatial_tile" => ["tileA", "tileA", "tileB", "tileB"],
        ]
        .expect("frame")
    }

    #[test]
    fn read_tiles_filter_and_rv_requirement() {
        let (_dir, path) = write_parquet(&mut canonical_like_frame(), "gnn.parquet");
        // tileA: 2 rows, one without RV -> 1 survives.
        let df = read_gnn_parquet(&path, None, Some("tileA".to_string())).expect("read");
        assert_eq!(df.height(), 1);
        // No filter: 3 rows (null-RV row dropped).
        let df_all = read_gnn_parquet(&path, None, None).expect("read");
        assert_eq!(df_all.height(), 3);
    }

    #[test]
    fn read_max_rows_strides() {
        let (_dir, path) = write_parquet(&mut canonical_like_frame(), "gnn.parquet");
        // 3 valid rows capped at 2 -> stride 2 -> 2 rows.
        let df = read_gnn_parquet(&path, Some(2), None).expect("read");
        assert_eq!(df.height(), 2);
    }
}
