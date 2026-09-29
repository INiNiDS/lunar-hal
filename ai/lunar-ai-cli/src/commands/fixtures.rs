use anyhow::Result;
use std::env;
use std::path::Path;

pub fn load_dotenv_if_present() {
    let Ok(content) = std::fs::read_to_string(".env") else {
        return;
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if key.is_empty() || env::var_os(key).is_some() {
            continue;
        }
        unsafe { env::set_var(key, value) };
    }
}

const NASA_KEY_ENV: &str = "NASA_API_KEY";
const MAST_TOKEN_ENV: &str = "MAST_API_TOKEN";
const GAIA_USER_ENV: &str = "GAIA_USERNAME";
const GAIA_PASS_ENV: &str = "GAIA_PASSWORD";

pub fn run_auth_status() {
    use lnai_data::sources::{SourceAdapter as _, SourceAuth};

    println!("Source auth matrix (secret values are NEVER printed):");
    for (id, p) in [
        (
            "exoplanet",
            lnai_data::sources::nasa_exoplanet::NasaExoplanetAdapter.provenance(),
        ),
        (
            "mast",
            lnai_data::sources::mast::MastTicAdapter.provenance(),
        ),
        ("irsa", lnai_data::sources::irsa::IrsaAdapter.provenance()),
        (
            "jpl",
            lnai_data::sources::jpl_horizons::JplHorizonsAdapter.provenance(),
        ),
    ] {
        let mode = match &p.auth {
            SourceAuth::Anonymous => "anonymous".to_string(),
            SourceAuth::EnvKeys { requires } => format!("env:{}", requires.join(",")),
        };
        println!(
            "  [{id}] {catalog:<40} mode={mode} backbone={}",
            p.enters_stellar_backbone,
            catalog = p.catalog_name
        );
    }

    let optional = [
        (
            NASA_KEY_ENV,
            "api.nasa.gov key-based endpoints (NOT required for Exoplanet TAP)",
        ),
        (MAST_TOKEN_ENV, "protected/EAP MAST products only"),
        (GAIA_USER_ENV, "Gaia user space / long async jobs"),
        (GAIA_PASS_ENV, "Gaia password half"),
    ];
    for (name, purpose) in optional {
        let present = env::var(name)
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);
        println!(
            "  env {name}={present} — {purpose}",
            present = if present { "<set>" } else { "<unset>" }
        );
    }

    match lnai_data::storage::sink_status() {
        lnai_data::storage::SinkStatus::Unconfigured => {
            println!(
                "  storage sink: unconfigured (set S3_ENDPOINT/S3_BUCKET; MinIO via install/data-minio.compose.yml)"
            );
        }
        lnai_data::storage::SinkStatus::AnonymousRead { endpoint, bucket } => {
            println!("  storage sink: anonymous-read endpoint={endpoint} bucket={bucket}");
        }
        lnai_data::storage::SinkStatus::Authenticated {
            endpoint,
            bucket,
            key_head,
        } => {
            println!(
                "  storage sink: authenticated endpoint={endpoint} bucket={bucket} access_key={key_head}"
            );
        }
    }
}

pub fn run_enrich_fixtures(out_dir: &str) -> Result<()> {
    use lnai_data::sources::SourceAdapter as _;
    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../lnai-data/tests/fixtures/nasa_exoplanet_pscomppars_sample.csv"
    ));
    std::fs::create_dir_all(out_dir)?;
    let recs = lnai_data::sources::nasa_exoplanet::parse_pscomppars_csv(FIXTURE)
        .map_err(anyhow::Error::msg)?;

    anyhow::ensure!(
        (100..=1000).contains(&recs.len()),
        "fixture row count {} violates the plan band",
        recs.len()
    );

    let adapter_id = lnai_data::sources::nasa_exoplanet::ADAPTER_ID;
    let query_hash = lnai_data::integrity::sha256_hex(
        lnai_data::sources::nasa_exoplanet::adql_query(500).as_bytes(),
    );
    let mut manifest = lnai_data::source_manifest::SourceManifestV1::new();
    manifest.register(
        lnai_data::source_manifest::SourceManifestEntry::from_payload(
            adapter_id,
            lnai_data::sources::nasa_exoplanet::PS_COMPPARS_TAP_SYNC_URL,
            &query_hash,
            1_787_637_000_000,
            FIXTURE,
            recs.len(),
        ),
    );
    manifest
        .save(&Path::new(out_dir).join("source_manifest.json"))
        .map_err(anyhow::Error::msg)?;

    let enriched: Vec<serde_json::Value> = recs
        .iter()
        .map(|r| {
            serde_json::json!({
                "planet_name": r.planet_name,
                "hostname": r.hostname,
                "ra_deg": r.ra_deg,
                "dec_deg": r.dec_deg,
                "teff_k": r.teff_k,
                "radius_rsun": r.radius_rsun,
                "mass_msun": r.mass_msun,
                "enters_stellar_backbone": false,
            })
        })
        .collect();
    let rows_path = Path::new(out_dir).join("nasa_enriched_rows.json");
    std::fs::write(&rows_path, serde_json::to_vec_pretty(&enriched).unwrap())?;

    let mut provenance = lnai_data::provenance::DatasetProvenanceV1::new(out_dir);
    provenance.register(lnai_data::sources::nasa_exoplanet::NasaExoplanetAdapter.provenance());
    provenance.impact_report_reference =
        Some("enrichment_report.json (produced by `lnaicli enrich-report`)".into());
    provenance
        .save(Path::new(out_dir))
        .map_err(anyhow::Error::msg)?;

    std::fs::write(Path::new(out_dir).join("raw_pscomppars.csv"), FIXTURE)?;

    println!(
        "Enriched fixtures written to {out_dir}: {} objects; manifest entries={}, sha256={}",
        recs.len(),
        manifest.entries.len(),
        &manifest.entries[0].payload_sha256[..16]
    );
    Ok(())
}
