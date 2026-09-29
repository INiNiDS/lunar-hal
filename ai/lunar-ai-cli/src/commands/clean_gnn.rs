use anyhow::{Result, anyhow};
use polars::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::util::sha256_file;

pub fn clean_and_transform_gnn(
    input_path: &str,
    output_path: &str,
    print_sha256: bool,
) -> Result<()> {
    println!(
        "Reading and preprocessing (GNN velocity mode): {}",
        input_path
    );

    if !Path::new(input_path).exists() {
        return Err(anyhow!("Input file does not exist: {}", input_path));
    }

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(input_path.into()))?
        .finish()?;

    let has_vel_cols = df.column("pmra").is_ok()
        && df.column("pmdec").is_ok()
        && df.column("radial_velocity").is_ok()
        && df.column("parallax").is_ok();

    if !has_vel_cols {
        anyhow::bail!(
            "GNN velocity columns (pmra, pmdec, radial_velocity, parallax) not found. \
             Use 'lnaicli fetch-gnn' to download data with velocities."
        );
    }

    println!("Detected velocity columns (pmra, pmdec, radial_velocity, parallax)");

    let target_cols = vec![
        PlSmallStr::from_str("ra"),
        PlSmallStr::from_str("dec"),
        PlSmallStr::from_str("parallax"),
        PlSmallStr::from_str("pmra"),
        PlSmallStr::from_str("pmdec"),
        PlSmallStr::from_str("radial_velocity"),
        PlSmallStr::from_str("st_teff"),
        PlSmallStr::from_str("st_rad"),
        PlSmallStr::from_str("st_mass"),
        PlSmallStr::from_str("bp_rp"),
        PlSmallStr::from_str("g_mag"),
    ];

    let selector = Selector::ByName {
        names: Arc::from(target_cols),
        strict: true,
    };

    let agg_exprs = vec![
        col("ra").first(),
        col("dec").first(),
        col("parallax").first(),
        col("pmra").first(),
        col("pmdec").first(),
        col("radial_velocity").first(),
        col("st_teff").first(),
        col("st_rad").first(),
        col("st_mass").first(),
        col("st_lum").first(),
        col("bp_rp").first(),
        col("g_mag").first(),
    ];

    let k_ast: f64 = 4.74047;

    let cleaned_lazy = df
        .lazy()
        .drop_nulls(Some(selector.clone()))
        .group_by([col("hostname")])
        .agg(agg_exprs)
        .with_columns([
            (col("ra") * lit(PI / 180.0)).alias("ra_rad"),
            (col("dec") * lit(PI / 180.0)).alias("dec_rad"),
        ])
        .with_columns([(lit(1000.0) / col("parallax")).alias("dist_pc")])
        .with_columns([
            (lit(k_ast) * col("pmra") / col("parallax")).alias("v_alpha"),
            (lit(k_ast) * col("pmdec") / col("parallax")).alias("v_delta"),
        ])
        .with_columns([
            (col("dist_pc") * col("dec_rad").cos() * col("ra_rad").cos()).alias("x_pc"),
            (col("dist_pc") * col("dec_rad").cos() * col("ra_rad").sin()).alias("y_pc"),
            (col("dist_pc") * col("dec_rad").sin()).alias("z_pc"),
        ])
        .with_columns([
            (col("radial_velocity") * col("dec_rad").cos() * col("ra_rad").cos()
                - col("v_alpha") * col("ra_rad").sin()
                - col("v_delta") * col("dec_rad").sin() * col("ra_rad").cos())
            .alias("vx"),
            (col("radial_velocity") * col("dec_rad").cos() * col("ra_rad").sin()
                + col("v_alpha") * col("ra_rad").cos()
                - col("v_delta") * col("dec_rad").sin() * col("ra_rad").sin())
            .alias("vy"),
            (col("radial_velocity") * col("dec_rad").sin() + col("v_delta") * col("dec_rad").cos())
                .alias("vz"),
        ])
        .select([
            col("hostname"),
            col("x_pc"),
            col("y_pc"),
            col("z_pc"),
            col("bp_rp"),
            col("g_mag"),
            col("st_teff"),
            col("st_rad"),
            col("st_mass"),
            col("st_lum"),
            col("vx"),
            col("vy"),
            col("vz"),
        ]);

    let mut final_df = cleaned_lazy.collect()?;

    let n = final_df.height();
    println!("Computed Cartesian velocities (vx, vy, vz) for {} stars", n);

    let tmp_path = format!("{}.tmp", output_path);
    {
        let file = File::create(&tmp_path)?;
        ParquetWriter::new(file).finish(&mut final_df)?;
    }
    std::fs::rename(&tmp_path, output_path)?;

    println!(
        "Done. {} unique stars with velocities. Output: {}",
        n, output_path
    );

    if print_sha256 {
        let h = sha256_file(output_path)?;
        println!("SHA256  {}", h);
    }

    Ok(())
}
