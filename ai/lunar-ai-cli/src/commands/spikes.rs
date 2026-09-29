use anyhow::Result;

pub fn run_spike_mast(ra: f64, dec: f64, radius: f64) -> Result<()> {
    let params = lnai_data::sources::mast::ConeParams {
        ra_deg: ra,
        dec_deg: dec,
        radius_deg: radius,
        page_size: 50,
    };
    let (raw, records) =
        lnai_data::sources::mast::fetch_cone(&params).map_err(anyhow::Error::msg)?;
    let with_pm = records.iter().filter(|r| r.pm_ra_mas_yr.is_some()).count();
    let with_gaia = records
        .iter()
        .filter(|r| r.gaia_source_id.is_some())
        .count();
    println!(
        "MAST TIC cone ({ra},{dec},r={radius}): {} metadata rows; pm coverage {:.0}%, cross-id GAIA {:.0}% (query hash {})",
        records.len(),
        100.0 * with_pm as f64 / records.len().max(1) as f64,
        100.0 * with_gaia as f64 / records.len().max(1) as f64,
        &params.query_hash()[..16]
    );
    println!(
        "response sha256: {}",
        &lnai_data::integrity::sha256_hex(raw.as_bytes())[..16]
    );
    Ok(())
}

pub fn run_spike_irsa(ra_min: f64, ra_max: f64, dec_min: f64, dec_max: f64, top: usize) -> Result<()> {
    let query = lnai_data::sources::irsa::adql_query(ra_min, ra_max, dec_min, dec_max, top);
    let csv = lnai_data::sources::irsa::fetch_box_csv(&query).map_err(anyhow::Error::msg)?;
    let rows = lnai_data::sources::irsa::parse_two_mass_csv(&csv).map_err(anyhow::Error::msg)?;
    let stats = lnai_data::sources::irsa::coverage_stats(&rows);
    println!(
        "IRSA 2MASS fp_psc box RA[{ra_min},{ra_max}) Dec[{dec_min},{dec_max}): {} rows; full JHK {:.0}%; null-rates j={:.2} h={:.2} k={:.2} pm={:.2}",
        stats.row_count,
        100.0 * stats.full_photometry_fraction,
        stats.null_rate.j_m,
        stats.null_rate.h_m,
        stats.null_rate.k_m,
        stats.null_rate.pm
    );
    Ok(())
}

pub fn run_jpl_scenes(
    body: &str,
    start_time: &str,
    stop_time: &str,
    step_size: &str,
    center: &str,
) -> Result<()> {
    let req = lnai_data::sources::jpl_horizons::SceneRequest {
        body_code: body.to_string(),
        center: center.to_string(),
        start_time: start_time.to_string(),
        stop_time: stop_time.to_string(),
        step_size: step_size.to_string(),
    };
    let (_raw, rows) =
        lnai_data::sources::jpl_horizons::fetch_scene(&req).map_err(anyhow::Error::msg)?;
    println!(
        "JPL Horizons scene (body={body}, center={center}): {} ephemeris rows, query hash {}",
        rows.len(),
        &req.query_hash()[..16]
    );
    for r in rows.iter().take(3) {
        println!(
            "  {} RA={:.5}deg DEC={:.5} Delta={:.4}AU",
            r.time_utc, r.ra_deg, r.dec_deg, r.delta_au
        );
    }
    println!("scene rows are scene-provider output only: enters_stellar_backbone=false");
    Ok(())
}
