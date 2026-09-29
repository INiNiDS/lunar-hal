use anyhow::{Context, Result, anyhow};
use polars::prelude::*;
use std::fs::File;
use std::path::Path;

pub fn collect_gaia_sample_rows_from_parquet(
    path: &Path,
    max_rows: usize,
) -> Result<Vec<lnai_data::enrich::GaiaSampleRow>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let df = ParquetReader::new(file).finish()?;

    let col_req_f64 = |name: &str| -> Result<Vec<f64>> {
        let series = df
            .column(name)
            .ok()
            .with_context(|| format!("column {name} required in {}", path.display()))?;
        Ok(series.f64()?.into_no_null_iter().collect())
    };
    let col_opt_f64 = |name: &str| -> Vec<Option<f64>> {
        df.column(name)
            .ok()
            .and_then(|series| series.f64().ok().map(|ca| ca.iter().collect()))
            .unwrap_or_default()
    };
    let col_ids = || -> Result<Vec<String>> {
        let series = df
            .column("source_id")
            .map_err(|_| anyhow!("column source_id required"))?;
        Ok(series.str()?.iter().flatten().map(String::from).collect())
    };

    let ra = col_req_f64("ra_deg")?;
    let dec = col_req_f64("dec_deg")?;
    let ids = col_ids()?;
    let mag_g: Vec<Option<f64>> = col_opt_f64("mag_g");
    let bp: Vec<Option<f64>> = col_opt_f64("mag_bp");
    let rp: Vec<Option<f64>> = col_opt_f64("mag_rp");
    let plx: Vec<Option<f64>> = col_opt_f64("parallax_mas");

    let n = ra.len().min(dec.len()).min(ids.len()).min(max_rows);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (Some(g), Some(bp_i), Some(rp_i), Some(parallax)) = (mag_g[i], bp[i], rp[i], plx[i])
        else {
            continue;
        };
        if !g.is_finite() || parallax <= 0.0 || !(bp_i - rp_i).is_finite() {
            continue;
        }
        out.push(lnai_data::enrich::GaiaSampleRow {
            source_id: ids[i].clone(),
            ra_deg: ra[i],
            dec_deg: dec[i],
            epoch_year: 2016.0,
            pm_ra_mas_yr: None,
            pm_dec_mas_yr: None,
            mag_g: g,
            mag_bp: bp_i,
            mag_rp: rp_i,
            parallax_mas: parallax,
        });
    }
    Ok(out)
}

pub fn run_enrich_report(out_dir: &Path, gaia_parquet: Option<&str>) -> Result<()> {
    let rows_raw = std::fs::read(out_dir.join("nasa_enriched_rows.json")).with_context(|| {
        format!(
            "run `lnaicli enrich-fixtures --out-dir {}` first",
            out_dir.display()
        )
    })?;
    let raw: Vec<serde_json::Value> =
        serde_json::from_slice(&rows_raw).context("nasa_enriched_rows.json is valid JSON")?;
    let grab = |v: &serde_json::Value, k: &str| -> Result<f64> {
        v.get(k)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow!("row missing numeric field {k}"))
    };
    let nasa_rows = raw
        .into_iter()
        .map(|v| {
            Ok(lnai_data::sources::nasa_exoplanet::NasaExoplanetRecord {
                planet_name: v
                    .get("planet_name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                hostname: v
                    .get("hostname")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                ra_deg: grab(&v, "ra_deg")?,
                dec_deg: grab(&v, "dec_deg")?,
                parallax_mas: None,
                distance_pc: None,
                teff_k: v.get("teff_k").and_then(serde_json::Value::as_f64),
                radius_rsun: v.get("radius_rsun").and_then(serde_json::Value::as_f64),
                mass_msun: v.get("mass_msun").and_then(serde_json::Value::as_f64),
                luminosity_lsun: None,
                discovery_year: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let Some(parquet_path) = gaia_parquet else {
        let note = serde_json::json!({
            "version": "1.0.0",
            "status": "insufficient_data",
            "reason": "no Gaia backbone provided; pass --gaia-parquet pointing at canonical stars.parquet",
            "nasa_rows": nasa_rows.len(),
            "verdict": "not_evaluated"
        });
        let path = out_dir.join("enrichment_report.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&note).unwrap())?;
        println!("Honest no-data report written: {}", path.display());
        return Ok(());
    };

    const TOLERANCE_ARCSEC: f64 = 1.0;
    const MAX_GAIA_ROWS: usize = 400_000;
    let gaia_rows = collect_gaia_sample_rows_from_parquet(Path::new(parquet_path), MAX_GAIA_ROWS)?;
    let (samples, unmatched) = lnai_data::enrich::enrichment_samples_from_records(
        &nasa_rows,
        &gaia_rows,
        TOLERANCE_ARCSEC,
    );

    let report_json = match lnai_data::enrich::evaluate(&samples) {
        Some(report) => {
            println!(
                "Enrichment evaluation on {}/{} matched samples:\n  before {}\n  after  {}\n  delta {:.1}% verdict={:?}",
                samples.len(),
                nasa_rows.len(),
                report.before,
                report.after,
                report.mae_delta_fraction * 100.0,
                report.verdict
            );
            serde_json::to_string_pretty(&report)?
        }
        None => {
            let note = serde_json::json!({
                "version": "1.0.0",
                "status": "insufficient_matched_samples",
                "matched_complete_samples": samples.len(),
                "unmatched_or_incomplete": unmatched,
                "verdict": "not_evaluated"
            });
            println!(
                "Only {}/{} matched complete samples (<8); wrote not_evaluated report honestly.",
                samples.len(),
                nasa_rows.len()
            );
            serde_json::to_string_pretty(&note)?
        }
    };
    let path = out_dir.join("enrichment_report.json");
    std::fs::write(&path, report_json)?;
    println!("Report written: {}", path.display());
    Ok(())
}
