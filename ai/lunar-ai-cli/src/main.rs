use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand};
use polars::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "nasa_stellar_parser")]
#[command(about = "CLI tool to download and process stellar data", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Fetch {
        #[arg(short, long, default_value = "raw_stars.csv")]
        output: String,
    },
    Clean {
        #[arg(short, long, default_value = "raw_stars.csv")]
        input: String,
        #[arg(short, long, default_value = "clean_stars.parquet")]
        output: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Fetch { output } => {
            fetch_stellar_data(output)?;
        }
        Commands::Clean { input, output } => {
            clean_and_transform(input, output)?;
        }
    }

    Ok(())
}

fn fetch_stellar_data(output_path: &str) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Some(std::time::Duration::from_secs(3600)))
        .pool_max_idle_per_host(0)
        .build()?;

    #[cfg(debug_assertions)]
    {
        let url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync";
        let query = "select hostname, ra, dec, sy_dist, st_teff, st_rad, st_mass, st_lum from ps";

        println!("Sending request to NASA Exoplanet Archive (Debug Mode)...");

        let response = client
            .get(url)
            .query(&[("query", query), ("format", "csv")])
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "NASA server returned an error status: {}",
                response.status()
            ));
        }

        save_filtered_response(response, output_path)?;
    }

    #[cfg(not(debug_assertions))]
    {
        let url = "https://gea.esac.esa.int/tap-server/tap/async";
        let query = "SELECT \
            CAST(gs.source_id AS varchar) AS hostname, \
            gs.ra, \
            gs.dec, \
            1000.0/gs.parallax AS sy_dist, \
            ap.teff_gspphot AS st_teff, \
            ap.radius_gspphot AS st_rad, \
            ap.mass_flame AS st_mass, \
            ap.lum_flame AS st_lum \
         FROM gaiadr3.gaia_source gs \
         JOIN gaiadr3.astrophysical_parameters ap USING (source_id) \
         WHERE gs.parallax IS NOT NULL \
           AND ap.teff_gspphot IS NOT NULL \
           AND ap.radius_gspphot IS NOT NULL \
           AND ap.mass_flame IS NOT NULL";

        println!("Submitting asynchronous job to ESA Gaia Archive (Release Mode)...");

        let response = client
            .post(url)
            .form(&[
                ("REQUEST", "doQuery"),
                ("LANG", "ADQL"),
                ("FORMAT", "csv"),
                ("QUERY", query),
                ("PHASE", "RUN"),
            ])
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Gaia server rejected job submission. Status: {}",
                response.status()
            ));
        }

        let job_url = response.url().clone();
        println!("Job successfully created. Monitoring status at: {}", job_url);

        let phase_url = format!("{}/phase", job_url);
        let result_url = format!("{}/results/result", job_url);

        loop {
            let phase_resp_res = client
                .get(&phase_url)
                .header(reqwest::header::CONNECTION, "close")
                .send();

            let phase = match phase_resp_res {
                Ok(resp) => {
                    match resp.text() {
                        Ok(text) => text.trim().to_uppercase(),
                        Err(_) => {
                            eprintln!("Warning: failed to read response text, retrying in 5 seconds...");
                            std::thread::sleep(std::time::Duration::from_secs(5));
                            continue;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: network error ({}), retrying in 5 seconds...", e);
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    continue;
                }
            };

            println!("Current job phase: {}", phase);

            match phase.as_str() {
                "COMPLETED" => {
                    println!("Job completed successfully!");
                    break;
                }
                "ERROR" | "ABORTED" => {
                    return Err(anyhow!(
                        "Background job failed or was aborted on the ESA server. Phase: {}",
                        phase
                    ));
                }
                _ => {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
            }
        }

        println!("Downloading raw results from: {}", result_url);
        let response = client
            .get(&result_url)
            .header(reqwest::header::CONNECTION, "close")
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to download Gaia results. Status: {}",
                response.status()
            ));
        }

        save_filtered_response(response, output_path)?;
    }

    Ok(())
}

fn save_filtered_response(response: reqwest::blocking::Response, output_path: &str) -> Result<()> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let reader = BufReader::new(response);

    for line_result in reader.lines() {
        let line = line_result?;
        if !line.trim_start().starts_with('#') {
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
        }
    }
    writer.flush()?;
    println!("Raw data successfully saved to: {}", output_path);
    Ok(())
}

fn clean_and_transform(input_path: &str, output_path: &str) -> Result<()> {
    println!("Reading and preprocessing file: {}", input_path);

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(input_path.into()))?
        .finish()?;

    let lazy_df = df.lazy();

    let target_cols = vec![
        PlSmallStr::from_str("ra"),
        PlSmallStr::from_str("dec"),
        PlSmallStr::from_str("sy_dist"),
        PlSmallStr::from_str("st_teff"),
        PlSmallStr::from_str("st_rad"),
        PlSmallStr::from_str("st_mass"),
    ];

    let selector = Selector::ByName {
        names: Arc::from(target_cols),
        strict: true,
    };

    let cleaned_lazy = lazy_df
        .drop_nulls(Some(selector.clone()))
        .group_by([col("hostname")])
        .agg([
            col("ra").first(),
            col("dec").first(),
            col("sy_dist").first(),
            col("st_teff").first(),
            col("st_rad").first(),
            col("st_mass").first(),
            col("st_lum").first(),
        ])
        .with_columns([
            (col("ra") * lit(PI / 180.0)).alias("ra_rad"),
            (col("dec") * lit(PI / 180.0)).alias("dec_rad"),
        ])
        .with_columns([
            (col("sy_dist") * col("dec_rad").cos() * col("ra_rad").cos()).alias("x_pc"),
            (col("sy_dist") * col("dec_rad").cos() * col("ra_rad").sin()).alias("y_pc"),
            (col("sy_dist") * col("dec_rad").sin()).alias("z_pc"),
        ])
        .select([
            col("hostname"),
            col("x_pc"),
            col("y_pc"),
            col("z_pc"),
            col("st_teff"),
            col("st_rad"),
            col("st_mass"),
            col("st_lum"),
        ]);

    let mut final_df = cleaned_lazy.collect()?;

    let file = File::create(output_path)?;
    ParquetWriter::new(file).finish(&mut final_df)?;

    println!(
        "Processing complete. Unique stars processed: {}. Output saved to: {}",
        final_df.height(),
        output_path
    );

    Ok(())
}