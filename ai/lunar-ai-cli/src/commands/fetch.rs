use anyhow::{Result, anyhow};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

pub struct FetchOptions<'a> {
    pub output_path: &'a str,
    pub username: Option<&'a str>,
    pub password: Option<&'a str>,
    pub max_rows: usize,
    pub ra_min: f64,
    pub ra_max: f64,
    pub max_ruwe: f64,
    pub poll_initial_secs: u64,
    pub poll_max_secs: u64,
    pub include_velocities: bool,
}

pub fn fetch_stellar_data(opts: &FetchOptions<'_>) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Some(std::time::Duration::from_secs(7200)))
        .pool_max_idle_per_host(0)
        .cookie_store(true)
        .build()?;

    #[cfg(debug_assertions)]
    {
        let _ = (
            opts.username,
            opts.password,
            opts.max_rows,
            opts.ra_min,
            opts.ra_max,
            opts.max_ruwe,
            opts.poll_initial_secs,
            opts.poll_max_secs,
            opts.include_velocities,
        );
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

        save_filtered_response(response, opts.output_path)?;
    }

    #[cfg(not(debug_assertions))]
    {
        if opts.username.is_none() || opts.password.is_none() {
            eprintln!("WARNING: No Gaia credentials provided.");
            eprintln!("         Anonymous access may limit result set size.");
            eprintln!("         Use --username and --password to authenticate.");
            eprintln!();
        }

        if let (Some(user), Some(pass)) = (opts.username, opts.password) {
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

        let query = if opts.include_velocities {
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
                opts.max_rows, opts.ra_min, opts.ra_max, opts.max_ruwe
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
                opts.max_rows, opts.ra_min, opts.ra_max, opts.max_ruwe
            )
        };

        let url = "https://gea.esac.esa.int/tap-server/tap/async";
        println!("Submitting asynchronous job to ESA Gaia Archive...");
        println!(
            "Query TOP {} rows, RA=[{}, {}], ruwe<{} ...",
            opts.max_rows, opts.ra_min, opts.ra_max, opts.max_ruwe
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

        let mut backoff = opts.poll_initial_secs.max(1);
        let backoff_max = opts.poll_max_secs.max(backoff);
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
                            return Err(anyhow!("Gaia job aborted: too many failed phase reads"));
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
                        return Err(anyhow!("Gaia job aborted: too many network errors"));
                    }
                    continue;
                }
            };

            attempts = 0;
            backoff = opts.poll_initial_secs.max(1);
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

        save_filtered_response(response, opts.output_path)?;
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
