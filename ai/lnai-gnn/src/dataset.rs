use anyhow::{Context, Result};
use burn::prelude::*;
use lnai_models::compute_knn_adjacency;
use polars::prelude::*;
use rand::rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;

pub const NODE_FEATURE_DIM: usize = 8;
pub const VELOCITY_DIM: usize = 3;
pub const DEFAULT_KNN_K: usize = 8;
pub const DEFAULT_MAX_GROUP: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnNormParams {
    pub log_teff_mean: f32,
    pub log_teff_std: f32,
    pub log_rad_mean: f32,
    pub log_rad_std: f32,
    pub log_mass_mean: f32,
    pub log_mass_std: f32,
    pub log_lum_mean: f32,
    pub log_lum_std: f32,
    pub mg_mean: f32,
    pub mg_std: f32,
    pub x_mean: f32,
    pub x_std: f32,
    pub y_mean: f32,
    pub y_std: f32,
    pub z_mean: f32,
    pub z_std: f32,
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
    pub groups: Arc<Vec<StarGroup>>, // Обернуто в Arc для быстрого клонирования ссылок
    pub norm: GnnNormParams,
    indices: Vec<usize>,
}

impl GnnDataset {
    pub fn load(
        parquet_path: &Path,
        knn_k: usize,
        max_group_size: usize,
        radius_pc: f32,
    ) -> Result<Self> {
        let (groups, norm) = build_groups_from_parquet(parquet_path, knn_k, max_group_size, radius_pc)?;
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
    ) -> Result<Self> {
        let (groups, _) = build_groups_from_parquet_with_norm(parquet_path, &norm, knn_k, max_group_size, radius_pc)?;
        let n = groups.len();
        println!("Built {} star groups (external norm)", n);

        Ok(Self {
            groups: Arc::new(groups),
            norm,
            indices: (0..n).collect(),
        })
    }

    pub fn split(self, val_frac: f32) -> (Self, Self) {
        let n = self.groups.len();
        let n_val = ((n as f32) * val_frac) as usize;
        let n_train = n - n_val;

        let mut split_indices: Vec<usize> = (0..n).collect();
        split_indices.shuffle(&mut rng());

        let train_idx = &split_indices[..n_train];
        let val_idx = &split_indices[n_train..];

        let train_groups: Vec<StarGroup> = train_idx.iter().map(|&i| self.groups[i].clone()).collect();
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
        self.indices.shuffle(&mut rng());
    }
}

struct PrefetchBatchedItem {
    nodes_data: Vec<f32>,
    adj_data: Vec<f32>,
    target_data: Vec<f32>,
    total_nodes: usize,
    #[allow(dead_code)]
    n_graphs: usize,
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
                let mut n_graphs = 0usize;
                let mut batch_groups: Vec<usize> = Vec::new();

                while idx < n_groups {
                    let gi = indices[idx];
                    let group_size = groups[gi].n_nodes();
                    if total_nodes + group_size > max_nodes_per_batch && total_nodes > 0 {
                        break;
                    }
                    batch_groups.push(gi);
                    total_nodes += group_size;
                    n_graphs += 1;
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

                if tx.send(PrefetchBatchedItem {
                    nodes_data,
                    adj_data: block_adj,
                    target_data: targets_data,
                    total_nodes,
                    n_graphs,
                }).is_err() {
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
            let adj = Tensor::<B, 2>::from_data(
                TensorData::new(batch.adj_data, [n, n]),
                device,
            );
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
) -> Result<(Vec<StarGroup>, GnnNormParams)> {
    let df = read_gnn_parquet(path)?;

    let x = extract_f32(&df, "x_pc")?;
    let y = extract_f32(&df, "y_pc")?;
    let z = extract_f32(&df, "z_pc")?;
    let _bp_rp = extract_f32(&df, "bp_rp")?;
    let g_mag = extract_f32(&df, "g_mag")?;
    let teff = extract_f32(&df, "st_teff")?;
    let rad = extract_f32(&df, "st_rad")?;
    let mass = extract_f32(&df, "st_mass")?;
    let lum = extract_f32(&df, "st_lum")?;
    let vx = extract_f32(&df, "vx")?;
    let vy = extract_f32(&df, "vy")?;
    let vz = extract_f32(&df, "vz")?;

    let n = df.height();

    let mg: Vec<f32> = (0..n)
        .map(|i| {
            let d = (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt().max(1e-6);
            g_mag[i] - 5.0 * d.log10() + 5.0
        })
        .collect();

    let log_teff: Vec<f32> = teff.par_iter().map(|&v| v.max(1e-10).log10()).collect();
    let log_rad: Vec<f32> = rad.par_iter().map(|&v| v.max(1e-10).log10()).collect();
    let log_mass: Vec<f32> = mass.par_iter().map(|&v| v.max(1e-10).log10()).collect();
    let log_lum: Vec<f32> = lum.par_iter().map(|&v| v.max(1e-10).log10()).collect();

    let (log_teff_m, log_teff_s) = mean_std(&log_teff);
    let (log_rad_m, log_rad_s) = mean_std(&log_rad);
    let (log_mass_m, log_mass_s) = mean_std(&log_mass);
    let (log_lum_m, log_lum_s) = mean_std(&log_lum);
    let (mg_m, mg_s) = mean_std(&mg);
    let (x_m, x_s) = mean_std(&x);
    let (y_m, y_s) = mean_std(&y);
    let (z_m, z_s) = mean_std(&z);
    let (vx_m, vx_s) = mean_std(&vx);
    let (vy_m, vy_s) = mean_std(&vy);
    let (vz_m, vz_s) = mean_std(&vz);

    let norm = GnnNormParams {
        log_teff_mean: log_teff_m, log_teff_std: log_teff_s,
        log_rad_mean: log_rad_m, log_rad_std: log_rad_s,
        log_mass_mean: log_mass_m, log_mass_std: log_mass_s,
        log_lum_mean: log_lum_m, log_lum_std: log_lum_s,
        mg_mean: mg_m, mg_std: mg_s,
        x_mean: x_m, x_std: x_s,
        y_mean: y_m, y_std: y_s,
        z_mean: z_m, z_std: z_s,
        vx_mean: vx_m, vx_std: vx_s,
        vy_mean: vy_m, vy_std: vy_s,
        vz_mean: vz_m, vz_std: vz_s,
    };

    let groups = build_star_groups(
        &x, &y, &z, &log_teff, &log_rad, &log_mass, &log_lum, &mg,
        &vx, &vy, &vz,
        &norm, knn_k, max_group_size, radius_pc,
    );

    Ok((groups, norm))
}

fn build_groups_from_parquet_with_norm(
    path: &Path,
    norm: &GnnNormParams,
    knn_k: usize,
    max_group_size: usize,
    radius_pc: f32,
) -> Result<(Vec<StarGroup>, GnnNormParams)> {
    let df = read_gnn_parquet(path)?;

    let x = extract_f32(&df, "x_pc")?;
    let y = extract_f32(&df, "y_pc")?;
    let z = extract_f32(&df, "z_pc")?;
    let _bp_rp = extract_f32(&df, "bp_rp")?;
    let g_mag = extract_f32(&df, "g_mag")?;
    let teff = extract_f32(&df, "st_teff")?;
    let rad = extract_f32(&df, "st_rad")?;
    let mass = extract_f32(&df, "st_mass")?;
    let lum = extract_f32(&df, "st_lum")?;
    let vx = extract_f32(&df, "vx")?;
    let vy = extract_f32(&df, "vy")?;
    let vz = extract_f32(&df, "vz")?;

    let n = df.height();

    let mg: Vec<f32> = (0..n)
        .map(|i| {
            let d = (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt().max(1e-6);
            g_mag[i] - 5.0 * d.log10() + 5.0
        })
        .collect();

    let log_teff: Vec<f32> = teff.par_iter().map(|&v| v.max(1e-10).log10()).collect();
    let log_rad: Vec<f32> = rad.par_iter().map(|&v| v.max(1e-10).log10()).collect();
    let log_mass: Vec<f32> = mass.par_iter().map(|&v| v.max(1e-10).log10()).collect();
    let log_lum: Vec<f32> = lum.par_iter().map(|&v| v.max(1e-10).log10()).collect();

    let groups = build_star_groups(
        &x, &y, &z, &log_teff, &log_rad, &log_mass, &log_lum, &mg,
        &vx, &vy, &vz,
        norm, knn_k, max_group_size, radius_pc,
    );

    Ok((groups, norm.clone()))
}

fn build_star_groups(
    x: &[f32], y: &[f32], z: &[f32],
    log_teff: &[f32], log_rad: &[f32], log_mass: &[f32], log_lum: &[f32], mg: &[f32],
    vx: &[f32], vy: &[f32], vz: &[f32],
    norm: &GnnNormParams,
    knn_k: usize,
    max_group_size: usize,
    radius_pc: f32,
) -> Vec<StarGroup> {
    let n = x.len();
    let mut assigned = vec![false; n];
    let mut groups = Vec::new();

    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut rng());

    for &seed in &order {
        if assigned[seed] {
            continue;
        }

        let sx = x[seed];
        let sy = y[seed];
        let sz = z[seed];
        let r2 = radius_pc * radius_pc;

        let mut members: Vec<usize> = Vec::new();
        for j in 0..n {
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

        let coords: Vec<[f32; 3]> = members
            .iter()
            .map(|&i| [x[i], y[i], z[i]])
            .collect();

        let node_features: Vec<[f32; NODE_FEATURE_DIM]> = members
            .iter()
            .map(|&i| [
                (log_teff[i] - norm.log_teff_mean) / norm.log_teff_std,
                (log_rad[i] - norm.log_rad_mean) / norm.log_rad_std,
                (log_mass[i] - norm.log_mass_mean) / norm.log_mass_std,
                (log_lum[i] - norm.log_lum_mean) / norm.log_lum_std,
                (mg[i] - norm.mg_mean) / norm.mg_std,
                (x[i] - norm.x_mean) / norm.x_std,
                (y[i] - norm.y_mean) / norm.y_std,
                (z[i] - norm.z_mean) / norm.z_std,
            ])
            .collect();

        let velocities: Vec<[f32; VELOCITY_DIM]> = members
            .iter()
            .map(|&i| [
                (vx[i] - norm.vx_mean) / norm.vx_std,
                (vy[i] - norm.vy_mean) / norm.vy_std,
                (vz[i] - norm.vz_mean) / norm.vz_std,
            ])
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

fn read_gnn_parquet(path: &Path) -> Result<DataFrame> {
    println!("Loading parquet: {}", path.display());
    let file = File::open(path).context("failed to open parquet")?;
    let df = ParquetReader::new(file).finish().context("failed to read parquet")?;

    let required_cols: &[&str] = &[
        "x_pc", "y_pc", "z_pc", "bp_rp", "g_mag",
        "st_teff", "st_rad", "st_mass", "st_lum",
        "vx", "vy", "vz",
    ];

    for &col_name in required_cols {
        if df.column(col_name).is_err() {
            anyhow::bail!(
                "Column '{}' not found in dataset. GNN training requires velocity columns (vx, vy, vz).",
                col_name
            );
        }
    }

    let df = df.drop_nulls(Some(required_cols)).context("drop_nulls failed")?;

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

    println!("Loaded {} rows with velocity data", df.height());
    Ok(df)
}

pub fn extract_f32(df: &DataFrame, name: &str) -> Result<Vec<f32>> {
    let s = df.column(name).context(format!("column {name} not found"))?;
    let s = s.cast(&DataType::Float64).context(format!("column {name} cast to f64 failed"))?;
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