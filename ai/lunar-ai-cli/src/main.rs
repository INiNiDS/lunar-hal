use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use polars::prelude::*;
use rand::seq::SliceRandom;
use sha2::{Digest, Sha256};
use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "lnaicli")]
#[command(about = "CLI tool to download, process, and combine stellar data from Gaia", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Fetch {
        #[arg(short, long, default_value = "raw_stars.csv")]
        output: String,
        #[arg(short, long)]
        username: Option<String>,
        #[arg(short, long)]
        password: Option<String>,
        #[arg(long, default_value = "1000000")]
        max_rows: usize,
        #[arg(long, default_value_t = 0.0)]
        ra_min: f64,
        #[arg(long, default_value_t = 180.0)]
        ra_max: f64,
        #[arg(long, default_value_t = 1.4)]
        max_ruwe: f64,
        #[arg(long, default_value_t = 10)]
        poll_initial_secs: u64,
        #[arg(long, default_value_t = 120)]
        poll_max_secs: u64,
    },
    FetchGnn {
        #[arg(short, long, default_value = "raw_gnn_stars.csv")]
        output: String,
        #[arg(short, long)]
        username: Option<String>,
        #[arg(short, long)]
        password: Option<String>,
        #[arg(long, default_value = "1000000")]
        max_rows: usize,
        #[arg(long, default_value_t = 0.0)]
        ra_min: f64,
        #[arg(long, default_value_t = 180.0)]
        ra_max: f64,
        #[arg(long, default_value_t = 1.4)]
        max_ruwe: f64,
        #[arg(long, default_value_t = 10)]
        poll_initial_secs: u64,
        #[arg(long, default_value_t = 120)]
        poll_max_secs: u64,
    },
    Clean {
        #[arg(short, long, default_value = "raw_stars.csv")]
        input: String,
        #[arg(short, long, default_value = "clean_stars.parquet")]
        output: String,
        #[arg(long, default_value_t = false)]
        print_sha256: bool,
    },
    CleanGnn {
        #[arg(short, long, default_value = "raw_gnn_stars.csv")]
        input: String,
        #[arg(short, long, default_value = "clean_gnn_stars.parquet")]
        output: String,
        #[arg(long, default_value_t = false)]
        print_sha256: bool,
    },
    Sha256 {
        #[arg(short, long)]
        input: String,
    },
    Combine {
        #[arg(short, long, num_args = 2..)]
        inputs: Vec<String>,
        #[arg(short, long, default_value = "combined_stars.parquet")]
        output: String,
    },
    Train {
        #[arg(short, long)]
        data: Option<String>,

        #[arg(short, long)]
        resume: Option<String>,

        #[arg(long)]
        norm: Option<String>,

        #[arg(short = 'O', long, default_value = "stellar_model")]
        output_dir: String,

        #[arg(long, default_value_t = 200)]
        epochs: usize,

        #[arg(long, default_value_t = 4096)]
        batch_size: usize,

        #[arg(long, default_value_t = 5e-4)]
        lr: f64,

        #[arg(long, default_value_t = 0.1)]
        physics_weight: f64,

        #[arg(long, default_value_t = 0.1)]
        val_frac: f32,

        #[arg(long, default_value_t = 0)]
        gpu_index: usize,

        #[arg(long)]
        holdout: Option<String>,

        #[arg(long)]
        lnai_bin: Option<String>,

        #[arg(long, default_value = "stellar_model.bpk")]
        model_file: String,

        #[arg(long, default_value = "stellar_norm.json")]
        norm_file: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Fetch {
            output,
            username,
            password,
            max_rows,
            ra_min,
            ra_max,
            max_ruwe,
            poll_initial_secs,
            poll_max_secs,
        } => {
            fetch_stellar_data(
                output,
                username.as_deref(),
                password.as_deref(),
                max_rows,
                *ra_min,
                *ra_max,
                *max_ruwe,
                *poll_initial_secs,
                *poll_max_secs,
                false,
            )?;
        }
        Commands::FetchGnn {
            output,
            username,
            password,
            max_rows,
            ra_min,
            ra_max,
            max_ruwe,
            poll_initial_secs,
            poll_max_secs,
        } => {
            fetch_stellar_data(
                output,
                username.as_deref(),
                password.as_deref(),
                max_rows,
                *ra_min,
                *ra_max,
                *max_ruwe,
                *poll_initial_secs,
                *poll_max_secs,
                true,
            )?;
        }
        Commands::Clean {
            input,
            output,
            print_sha256,
        } => {
            clean_and_transform(input, output, *print_sha256)?;
        }
        Commands::CleanGnn {
            input,
            output,
            print_sha256,
        } => {
            clean_and_transform_gnn(input, output, *print_sha256)?;
        }
        Commands::Sha256 { input } => {
            let h = sha256_file(input)?;
            println!("{}  {}", h, input);
        }
        Commands::Combine { inputs, output } => {
            combine_datasets(inputs, output)?;
        }
        Commands::Train {
            data,
            resume,
            norm,
            output_dir,
            epochs,
            batch_size,
            lr,
            physics_weight,
            val_frac,
            gpu_index,
            holdout,
            lnai_bin,
            model_file,
            norm_file,
        } => {
            run_train(
                data.as_deref(),
                resume.as_deref(),
                norm.as_deref(),
                output_dir,
                *epochs,
                *batch_size,
                *lr,
                *physics_weight,
                *val_frac,
                *gpu_index,
                holdout.as_deref(),
                lnai_bin.as_deref(),
                model_file,
                norm_file,
            )?;
        }
    }

    Ok(())
}

fn resolve_data_path(data: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = data {
        return Ok(PathBuf::from(p));
    }
    for candidate in [
        "ai_data/clean_stars2.parquet",
        "ai_data/clean_stars.parquet",
    ] {
        let path = Path::new(candidate);
        if path.exists() {
            println!("Using dataset: {}", path.display());
            return Ok(path.to_path_buf());
        }
    }
    anyhow::bail!(
        "No --data given and no cleaned parquet found in ai_data/. \
         Run 'lnaicli fetch' and 'lnaicli clean' first, or pass --data <path>."
    )
}

fn find_lnai_binary(override_path: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("--lnai-bin does not exist: {}", path.display());
        }
        return Ok(path);
    }

    let exe_suffix = env::consts::EXE_SUFFIX;
    let bin_name = format!("lnai{}", exe_suffix);

    if let Ok(path_env) = env::var("LNAI_BIN") {
        let p = PathBuf::from(path_env);
        if p.exists() {
            return Ok(p);
        }
    }

    if let Ok(paths) = env::var("PATH") {
        for dir in paths.split(std::path::MAIN_SEPARATOR_STR) {
            if dir.is_empty() {
                continue;
            }
            let candidate = Path::new(dir).join(&bin_name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    let workspace_root = env::current_dir().ok();
    if let Some(cwd) = workspace_root {
        for profile in ["release", "debug"] {
            for sub in ["", "ai/lnai"] {
                let candidate = cwd.join("target").join(profile);
                let candidate = if sub.is_empty() {
                    candidate.join(&bin_name)
                } else {
                    candidate.join(sub).join(&bin_name)
                };
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
        }
    }

    anyhow::bail!(
        "Could not find the 'lnai' binary. Build it with `cargo build -p lnai --release` \
         or pass --lnai-bin /path/to/lnai."
    )
}

fn copy_file_if_present(src: &Path, dst: &Path) -> Result<()> {
    if !src.exists() {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::copy(src, dst)
        .with_context(|| format!("failed to copy {} -> {}", src.display(), dst.display()))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_train(
    data: Option<&str>,
    resume: Option<&str>,
    norm: Option<&str>,
    output_dir: &str,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    physics_weight: f64,
    val_frac: f32,
    gpu_index: usize,
    holdout: Option<&str>,
    lnai_bin: Option<&str>,
    model_file: &str,
    norm_file: &str,
) -> Result<()> {
    let data_path = resolve_data_path(data)?;
    let lnai = find_lnai_binary(lnai_bin)?;

    let output_path = Path::new(output_dir);
    std::fs::create_dir_all(output_path)
        .with_context(|| format!("failed to create output dir {}", output_path.display()))?;
    let out_model = output_path.join(model_file);
    let out_norm = output_path.join(norm_file);

    let mut resume_dir: Option<PathBuf> = None;
    if let Some(resume_path) = resume {
        let resume_pb = PathBuf::from(resume_path);
        let resume_dir_path = if resume_pb.is_dir() {
            resume_pb.clone()
        } else {
            resume_pb
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        };

        let src_model = resume_pb
            .join(model_file)
            .components()
            .collect::<PathBuf>();
        let src_norm = resume_pb.join(norm_file);

        let resolved_model = if src_model.exists() {
            src_model
        } else {
            resume_dir_path.join(model_file)
        };
        let resolved_norm = if src_norm.exists() {
            src_norm
        } else {
            resume_dir_path.join(norm_file)
        };

        if !resolved_model.exists() {
            anyhow::bail!(
                "Resume requested but model file not found: {}",
                resolved_model.display()
            );
        }
        if !resolved_norm.exists() {
            anyhow::bail!(
                "Resume requested but norm file not found: {}",
                resolved_norm.display()
            );
        }

        println!("Staging resume files into output dir:");
        println!("  {} -> {}", resolved_model.display(), out_model.display());
        copy_file_if_present(&resolved_model, &out_model)?;
        println!("  {} -> {}", resolved_norm.display(), out_norm.display());
        copy_file_if_present(&resolved_norm, &out_norm)?;

        if let Some(extra_norm) = norm {
            let extra_norm_path = Path::new(extra_norm);
            if !extra_norm_path.exists() {
                anyhow::bail!("--norm file not found: {}", extra_norm_path.display());
            }
            println!(
                "  {} -> {} (overrides any copied norm)",
                extra_norm_path.display(),
                out_norm.display()
            );
            copy_file_if_present(extra_norm_path, &out_norm)?;
        }

        resume_dir = Some(output_path.to_path_buf());
    } else if let Some(extra_norm) = norm {
        let extra_norm_path = Path::new(extra_norm);
        if !extra_norm_path.exists() {
            anyhow::bail!("--norm file not found: {}", extra_norm_path.display());
        }
        println!(
            "Staging norm into output dir: {} -> {}",
            extra_norm_path.display(),
            out_norm.display()
        );
        copy_file_if_present(extra_norm_path, &out_norm)?;
    }

    println!();
    println!("=== lnaicli train ===");
    println!("Data:        {}", data_path.display());
    if let Some(rd) = &resume_dir {
        println!("Resume from: {}", rd.display());
    } else {
        println!("Resume from: <none, training from scratch>");
    }
    println!("Output dir:  {}", output_path.display());
    println!("lnai binary: {}", lnai.display());
    println!(
        "Hyperparams: epochs={}, batch={}, lr={:.2e}, phys_w={}, val_frac={}, gpu={}",
        epochs, batch_size, lr, physics_weight, val_frac, gpu_index
    );
    println!();

    let mut cmd = Command::new(&lnai);
    cmd.arg("--data").arg(&data_path);
    cmd.arg("--output-dir").arg(output_path);
    cmd.arg("--model-file").arg(model_file);
    cmd.arg("--norm-file").arg(norm_file);
    cmd.arg("--epochs").arg(epochs.to_string());
    cmd.arg("--batch-size").arg(batch_size.to_string());
    cmd.arg("--lr").arg(lr.to_string());
    cmd.arg("--physics-weight").arg(physics_weight.to_string());
    cmd.arg("--val-frac").arg(val_frac.to_string());
    cmd.arg("--gpu-index").arg(gpu_index.to_string());
    if let Some(rd) = &resume_dir {
        cmd.arg("--resume-from").arg(rd);
    }
    if let Some(h) = holdout {
        cmd.arg("--holdout").arg(h);
    }

    println!("Running: {:?}", cmd);
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn lnai at {}", lnai.display()))?;
    if !status.success() {
        anyhow::bail!("lnai training failed with exit code {:?}", status.code());
    }

    println!();
    println!("Done. Model artifacts:");
    println!("  {}", out_model.display());
    println!("  {}", out_norm.display());
    Ok(())
}

fn sha256_file(path: &str) -> Result<String> {
    let mut file = File::open(path)
        .map_err(|e| anyhow!("Cannot open {} for hashing: {}", path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(hex_encode(&digest))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

fn fetch_stellar_data(
    output_path: &str,
    username: Option<&str>,
    password: Option<&str>,
    max_rows: &usize,
    ra_min: f64,
    ra_max: f64,
    max_ruwe: f64,
    poll_initial_secs: u64,
    poll_max_secs: u64,
    include_velocities: bool,
) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Some(std::time::Duration::from_secs(7200)))
        .pool_max_idle_per_host(0)
        .cookie_store(true)
        .build()?;

    #[cfg(debug_assertions)]
    {
        let _ = (username, password, max_rows, ra_min, ra_max, max_ruwe, poll_initial_secs, poll_max_secs, include_velocities);
        let url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync";
        let query = "select hostname, ra, dec, sy_dist, st_teff, st_rad, st_mass, st_lum from ps";

        println!("Sending request to NASA Exoplanet Archive (Debug Mode)...");
        println!("Note: NASA Exoplanet Archive does not include bp_rp and g_mag.");

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
        if username.is_none() || password.is_none() {
            eprintln!("WARNING: No Gaia credentials provided.");
            eprintln!("         Anonymous access may limit result set size.");
            eprintln!("         Use --username and --password to authenticate.");
            eprintln!();
        }

        if let (Some(user), Some(pass)) = (username, password) {
            println!("Authenticating with ESA Gaia Archive as {}...", user);
            let login_url = "https://gea.esac.esa.int/tap-server/login";
            let login_resp = client
                .post(login_url)
                .form(&[("username", user), ("password", pass)])
                .send()?;

            if !login_resp.status().is_success() {
                return Err(anyhow!(
                    "Gaia login failed. Status: {}. Check your credentials.",
                    login_resp.status()
                ));
            }
            println!("Authentication successful!");
        }

        let query = if include_velocities {
            format!(
        "SELECT TOP {} \
        CAST(gs.source_id AS varchar) AS hostname, \
        gs.ra, \
        gs.dec, \
        gs.parallax, \
        gs.pmra, \
        gs.pmdec, \
        gs.radial_velocity, \
        gs.phot_g_mean_mag AS g_mag, \
        gs.bp_rp, \
        ap.teff_gspphot AS st_teff, \
        ap.radius_gspphot AS st_rad, \
        ap.mass_flame AS st_mass, \
        ap.lum_flame AS st_lum \
     FROM gaiadr3.gaia_source gs \
     JOIN gaiadr3.astrophysical_parameters ap USING (source_id) \
     WHERE gs.parallax IS NOT NULL \
       AND gs.parallax > 0 \
       AND gs.ra BETWEEN {} AND {} \
       AND gs.pmra IS NOT NULL \
       AND gs.pmdec IS NOT NULL \
       AND gs.radial_velocity IS NOT NULL \
       AND ap.teff_gspphot IS NOT NULL \
       AND ap.radius_gspphot IS NOT NULL \
       AND ap.mass_flame IS NOT NULL \
       AND gs.bp_rp IS NOT NULL \
       AND gs.phot_g_mean_mag IS NOT NULL \
       AND gs.ruwe < {}",
        max_rows, ra_min, ra_max, max_ruwe
    )
        } else {
            format!(
        "SELECT TOP {} \
        CAST(gs.source_id AS varchar) AS hostname, \
        gs.ra, \
        gs.dec, \
        1000.0/gs.parallax AS sy_dist, \
        gs.phot_g_mean_mag AS g_mag, \
        gs.bp_rp, \
        ap.teff_gspphot AS st_teff, \
        ap.radius_gspphot AS st_rad, \
        ap.mass_flame AS st_mass, \
        ap.lum_flame AS st_lum \
     FROM gaiadr3.gaia_source gs \
     JOIN gaiadr3.astrophysical_parameters ap USING (source_id) \
     WHERE gs.parallax IS NOT NULL \
       AND gs.parallax > 0 \
       AND gs.ra BETWEEN {} AND {} \
       AND ap.teff_gspphot IS NOT NULL \
       AND ap.radius_gspphot IS NOT NULL \
       AND ap.mass_flame IS NOT NULL \
       AND gs.bp_rp IS NOT NULL \
       AND gs.phot_g_mean_mag IS NOT NULL \
       AND gs.ruwe < {}",
        max_rows, ra_min, ra_max, max_ruwe
    )
        };

        let url = "https://gea.esac.esa.int/tap-server/tap/async";
        println!("Submitting asynchronous job to ESA Gaia Archive...");
        println!(
            "Query TOP {} rows, RA=[{}, {}], ruwe<{} ...",
            max_rows, ra_min, ra_max, max_ruwe
        );

        let response = client
            .post(url)
            .form(&[
                ("REQUEST", "doQuery"),
                ("LANG", "ADQL"),
                ("FORMAT", "csv"),
                ("QUERY", &query),
                ("PHASE", "RUN"),
            ])
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(anyhow!(
                "Gaia server rejected job submission. Status: {}\nBody: {}",
                status,
                body
            ));
        }

        let job_url = response.url().clone();
        println!("Job created. Monitoring: {}", job_url);

        let phase_url = format!("{}/phase", job_url);
        let result_url = format!("{}/results/result", job_url);

        let mut backoff = poll_initial_secs.max(1);
        let backoff_max = poll_max_secs.max(backoff);
        let mut attempts: u32 = 0;
        loop {
            let phase_resp = client
                .get(&phase_url)
                .header(reqwest::header::CONNECTION, "close")
                .send();

            let phase = match phase_resp {
                Ok(resp) => match resp.text() {
                    Ok(text) => text.trim().to_uppercase(),
                    Err(_) => {
                        eprintln!(
                            "Warning: failed to read phase (attempt {}), retrying in {}s...",
                            attempts + 1,
                            backoff
                        );
                        std::thread::sleep(std::time::Duration::from_secs(backoff));
                        backoff = (backoff.saturating_mul(2)).min(backoff_max);
                        attempts += 1;
                        if attempts > 200 {
                            return Err(anyhow!(
                                "Gaia job aborted: too many failed phase reads"
                            ));
                        }
                        continue;
                    }
                },
                Err(e) => {
                    eprintln!(
                        "Warning: network error '{}' (attempt {}), retrying in {}s...",
                        e,
                        attempts + 1,
                        backoff
                    );
                    std::thread::sleep(std::time::Duration::from_secs(backoff));
                    backoff = (backoff.saturating_mul(2)).min(backoff_max);
                    attempts += 1;
                    if attempts > 200 {
                        return Err(anyhow!(
                            "Gaia job aborted: too many network errors"
                        ));
                    }
                    continue;
                }
            };

            attempts = 0;
            backoff = poll_initial_secs.max(1);
            println!("  Job phase: {} (next poll in {}s)", phase, backoff);

            match phase.as_str() {
                "COMPLETED" => {
                    println!("Job completed!");
                    break;
                }
                "ERROR" | "ABORTED" => {
                    return Err(anyhow!("Job failed on server. Phase: {}", phase));
                }
                _ => {
                    std::thread::sleep(std::time::Duration::from_secs(backoff));
                    backoff = (backoff.saturating_mul(2)).min(backoff_max);
                }
            }
        }

        println!("Downloading results from: {}", result_url);
        let response = client
            .get(&result_url)
            .header(reqwest::header::CONNECTION, "close")
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to download results. Status: {}",
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

    let mut lines = 0usize;
    for line_result in reader.lines() {
        let line = line_result?;
        if !line.trim_start().starts_with('#') {
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
            lines += 1;
        }
    }
    writer.flush()?;
    println!("Raw data saved to: {} ({} lines)", output_path, lines);
    Ok(())
}

fn clean_and_transform(input_path: &str, output_path: &str, print_sha256: bool) -> Result<()> {
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

fn clean_and_transform_gnn(input_path: &str, output_path: &str, print_sha256: bool) -> Result<()> {
    println!("Reading and preprocessing (GNN velocity mode): {}", input_path);

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

    // k = 4.74047 km/s per (mas/yr at distance 1/parallax mas)
    // distance_pc = 1000 / parallax_mas
    // v_tangential = k * pm / parallax (km/s)
    //
    // ICRS Cartesian velocity:
    //   vx = vr*cos(d)*cos(a) - v_a*sin(a) - v_d*sin(d)*cos(a)
    //   vy = vr*cos(d)*sin(a) + v_a*cos(a) - v_d*sin(d)*sin(a)
    //   vz = vr*sin(d) + v_d*cos(d)
    // where a=ra_rad, d=dec_rad, vr=radial_velocity,
    //       v_a = k*pmra/parallax, v_d = k*pmdec/parallax

    let k_ast: f64 = 4.74047;

    let cleaned_lazy = df.lazy()
        .drop_nulls(Some(selector.clone()))
        .group_by([col("hostname")])
        .agg(agg_exprs)
        .with_columns([
            (col("ra") * lit(PI / 180.0)).alias("ra_rad"),
            (col("dec") * lit(PI / 180.0)).alias("dec_rad"),
        ])
        .with_columns([
            (lit(1000.0) / col("parallax")).alias("dist_pc"),
        ])
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
                - col("v_delta") * col("dec_rad").sin() * col("ra_rad").cos()
            ).alias("vx"),
            (col("radial_velocity") * col("dec_rad").cos() * col("ra_rad").sin()
                + col("v_alpha") * col("ra_rad").cos()
                - col("v_delta") * col("dec_rad").sin() * col("ra_rad").sin()
            ).alias("vy"),
            (col("radial_velocity") * col("dec_rad").sin()
                + col("v_delta") * col("dec_rad").cos()
            ).alias("vz"),
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

fn combine_datasets(input_paths: &[String], output_path: &str) -> Result<()> {
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
