use anyhow::{Result, anyhow};
use polars::prelude::*;
use rand::seq::SliceRandom;
use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::util::sha256_file;

pub fn clean_and_transform(input_path: &str, output_path: &str, print_sha256: bool) -> Result<()> {
    println!("Reading and preprocessing: {}", input_path);

    if !Path::new(input_path).exists() {
        return Err(anyhow!("Input file does not exist: {}", input_path));
    }

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(input_path.into()))?
        .finish()?;

    let has_gaia_cols = df.column("bp_rp").is_ok() && df.column("g_mag").is_ok();

    if has_gaia_cols {
        println!("Detected Gaia columns (bp_rp, g_mag) - photometric data included");
    } else {
        println!("WARNING: bp_rp and g_mag columns not found.");
        println!("         The model requires these for conditional inputs.");
        println!("         Re-fetch data with: lnaicli fetch --username USER --password PASS");
    }

    let lazy_df = df.lazy();

    let mut target_cols = vec![
        PlSmallStr::from_str("ra"),
        PlSmallStr::from_str("dec"),
        PlSmallStr::from_str("sy_dist"),
        PlSmallStr::from_str("st_teff"),
        PlSmallStr::from_str("st_rad"),
        PlSmallStr::from_str("st_mass"),
    ];

    if has_gaia_cols {
        target_cols.push(PlSmallStr::from_str("bp_rp"));
        target_cols.push(PlSmallStr::from_str("g_mag"));
    }

    let selector = Selector::ByName {
        names: Arc::from(target_cols),
        strict: true,
    };

    let mut agg_exprs = vec![
        col("ra").first(),
        col("dec").first(),
        col("sy_dist").first(),
        col("st_teff").first(),
        col("st_rad").first(),
        col("st_mass").first(),
        col("st_lum").first(),
    ];

    if has_gaia_cols {
        agg_exprs.push(col("bp_rp").first());
        agg_exprs.push(col("g_mag").first());
    }

    let mut select_exprs = vec![
        col("hostname"),
        col("x_pc"),
        col("y_pc"),
        col("z_pc"),
        col("st_teff"),
        col("st_rad"),
        col("st_mass"),
        col("st_lum"),
    ];

    if has_gaia_cols {
        select_exprs.push(col("bp_rp"));
        select_exprs.push(col("g_mag"));
    }

    let cleaned_lazy = lazy_df
        .drop_nulls(Some(selector.clone()))
        .group_by([col("hostname")])
        .agg(agg_exprs)
        .with_columns([
            (col("ra") * lit(PI / 180.0)).alias("ra_rad"),
            (col("dec") * lit(PI / 180.0)).alias("dec_rad"),
        ])
        .with_columns([
            (col("sy_dist") * col("dec_rad").cos() * col("ra_rad").cos()).alias("x_pc"),
            (col("sy_dist") * col("dec_rad").cos() * col("ra_rad").sin()).alias("y_pc"),
            (col("sy_dist") * col("dec_rad").sin()).alias("z_pc"),
        ])
        .select(select_exprs);

    let mut final_df = cleaned_lazy.collect()?;

    let tmp_path = format!("{}.tmp", output_path);
    {
        let file = File::create(&tmp_path)?;
        ParquetWriter::new(file).finish(&mut final_df)?;
    }
    std::fs::rename(&tmp_path, output_path)?;

    println!(
        "Done. {} unique stars. Output: {}",
        final_df.height(),
        output_path
    );

    if print_sha256 {
        let h = sha256_file(output_path)?;
        println!("SHA256  {}", h);
    }

    Ok(())
}

pub fn combine_datasets(input_paths: &[String], output_path: &str) -> Result<()> {
    if input_paths.len() < 2 {
        return Err(anyhow!("At least 2 input files required for combining"));
    }

    println!("Combining {} datasets...", input_paths.len());

    let mut dfs = Vec::new();
    let mut total_rows: usize = 0;
    for path in input_paths {
        println!("  Reading: {}", path);
        if !Path::new(path).exists() {
            return Err(anyhow!("Input parquet missing: {}", path));
        }
        let file = File::open(path)?;
        let df = ParquetReader::new(file).finish()?;
        println!("    {} rows", df.height());
        total_rows += df.height();
        dfs.push(df);
    }
    println!("  Total rows to merge: {}", total_rows);

    let mut combined = dfs.remove(0);
    for df in &dfs {
        combined = combined.vstack(df)?;
    }

    let mut indices: Vec<usize> = (0..combined.height()).collect();
    indices.shuffle(&mut rand::rng());
    let idx_ca = UInt32Chunked::from_vec(
        PlSmallStr::from_str("idx"),
        indices.iter().map(|&i| i as u32).collect(),
    );
    combined = combined.take(&idx_ca)?;

    let tmp_path = format!("{}.tmp", output_path);
    {
        let file = File::create(&tmp_path)?;
        ParquetWriter::new(file).finish(&mut combined)?;
    }
    std::fs::rename(&tmp_path, output_path)?;

    println!(
        "Combined: {} rows. Output: {}",
        combined.height(),
        output_path
    );

    Ok(())
}
