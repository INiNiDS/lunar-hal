//! Stage 4A / пункт 8: MAST TIC metadata spike (anonymous).
//!
//! The public TIC cone search is served by the STScI Mashup endpoint
//! (`POST /portal/Mashup/Mashup.asmx/invoke`, service
//! `Mast.Catalogs.Tic.Cone`) and requires no token for public catalog data.
//! `MAST_API_TOKEN` stays optional for protected/EAP products only; this
//! adapter never sends credentials unless explicitly configured by the caller.
//!
//! Scope guard: metadata (coordinates/PM/photometry/cross-IDs) only. Bulk
//! light-curve download is intentionally NOT part of this source.

use crate::sources::{SourceAdapter, SourceAuth, SourceProvenance};
use serde_json::Value;

pub const MAST_INVOKE_URL: &str = "https://mast.stsci.edu/portal/Mashup/Mashup.asmx/invoke";
pub const ADAPTER_ID: &str = "mast_tic_cone_v1";

/// Frozen cone parameters used by the spike; part of provenance hashing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConeParams {
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub radius_deg: f64,
    pub page_size: usize,
}

impl Default for ConeParams {
    fn default() -> Self {
        Self {
            ra_deg: 59.0,
            dec_deg: 6.0,
            radius_deg: 0.2,
            page_size: 10,
        }
    }
}

impl ConeParams {
    /// Encodes the documented Mashup request envelope deterministically.
    pub fn request_body(&self) -> String {
        format!(
            r#"{{"service":"Mast.Catalogs.Tic.Cone","format":"json","pageSize":{},"page":1,"params":{{"ra":{:.4},"dec":{:.4},"radius":{:.4}}}}}"#,
            self.page_size, self.ra_deg, self.dec_deg, self.radius_deg
        )
    }

    pub fn query_hash(&self) -> String {
        crate::integrity::sha256_hex(self.request_body().as_bytes())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TicRecord {
    pub tic_id: u64,
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// μα* and μδ in mas/yr; TIC reports them null for many faint sources.
    pub pm_ra_mas_yr: Option<f64>,
    pub pm_dec_mas_yr: Option<f64>,
    pub teff_k: Option<f64>,
    pub magnitude_t: Option<f64>,
    /// Stable cross-identifiers straight from TIC (typed as strings server-side).
    pub twomass_id: Option<String>,
    pub gaia_source_id: Option<String>,
}

fn opt_f64(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64().filter(|f| f.is_finite()),
        Value::String(s) => s.trim().parse::<f64>().ok().filter(|f| f.is_finite()),
        _ => None,
    }
}

fn opt_str(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) if !s.trim().is_empty() && !s.eq("null") => Some(s.trim().to_string()),
        _ => None,
    }
}

/// Parses the `data` array of a COMPLETE Mashup response. Rows missing ID or
/// coordinates are rejected loudly instead of being dropped silently.
pub fn parse_tic_cone_response(body: &str) -> Result<Vec<TicRecord>, String> {
    let root: Value =
        serde_json::from_str(body).map_err(|e| format!("tic response is not JSON: {e}"))?;
    let status = root
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !status.eq_ignore_ascii_case("COMPLETE") {
        return Err(format!("tic response status is not COMPLETE: {status:?}"));
    }
    let rows = root
        .get("data")
        .and_then(Value::as_array)
        .ok_or("tic response lacks data array")?;

    let mut out = Vec::new();
    for row in rows {
        let id = row
            .get("ID")
            .and_then(Value::as_u64)
            .ok_or("tic row without numeric ID")?;
        // ra/dec mandatory per contract (same policy as PSCompPars).
        let (Some(ra), Some(dec)) = (opt_f64(row.get("ra")), opt_f64(row.get("dec"))) else {
            return Err(format!("tic row {id} has unparseable coordinates"));
        };
        out.push(TicRecord {
            tic_id: id,
            ra_deg: ra,
            dec_deg: dec,
            pm_ra_mas_yr: opt_f64(row.get("pmRA")),
            pm_dec_mas_yr: opt_f64(row.get("pmDEC")),
            teff_k: opt_f64(row.get("Teff")),
            magnitude_t: opt_f64(row.get("Tmag")),
            twomass_id: opt_str(row.get("TWOMASS")),
            gaia_source_id: opt_str(row.get("GAIA")),
        });
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct MastTicAdapter;

impl SourceAdapter for MastTicAdapter {
    fn id(&self) -> &'static str {
        ADAPTER_ID
    }

    fn provenance(&self) -> SourceProvenance {
        SourceProvenance {
            adapter_id: ADAPTER_ID.to_string(),
            catalog_name: "MAST TIC-8".to_string(),
            release_or_version: "tic-8 (public)".to_string(),
            endpoint_url: MAST_INVOKE_URL.to_string(),
            auth: SourceAuth::Anonymous,
            enters_stellar_backbone: false,
        }
    }
}

/// Live anonymous download helper (CI replays the recorded fixture).
pub fn fetch_cone(params: &ConeParams) -> Result<(String, Vec<TicRecord>), String> {
    use std::io::Read;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .post(MAST_INVOKE_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!("request={}", urlencode(&params.request_body())))
        .send()
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("MAST returned HTTP {}", resp.status()));
    }
    let mut body = String::new();
    let mut resp = resp;
    resp.read_to_string(&mut body).map_err(|e| e.to_string())?;
    let records = parse_tic_cone_response(&body)?;
    Ok((body, records))
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::SourceAdapter as _;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mast_tic_cone.json"
    ));

    #[test]
    fn recorded_fixture_parses_offline_with_cross_ids() {
        let recs = parse_tic_cone_response(FIXTURE).expect("recorded fixture must parse");
        assert_eq!(recs.len(), 10);
        let first = &recs[0];
        assert_eq!(first.tic_id, 459_913_166);
        assert!((first.ra_deg - 59.0103044661196).abs() < 1e-9);
        assert_eq!(first.pm_ra_mas_yr, Some(11.4091));
        assert_eq!(first.magnitude_t, Some(16.1809));
        assert_eq!(first.gaia_source_id.as_deref(), Some("3273882668699031936"));
    }

    #[test]
    fn rows_without_coordinates_are_rejected() {
        let broken = r#"{"status":"COMPLETE","msg":"","data":[{"ID":7}]}"#;
        assert!(parse_tic_cone_response(broken).is_err());
    }

    #[test]
    fn incomplete_status_is_rejected_for_replay_safety() {
        let partial = r#"{"status":"EXECUTING","data":[]}"#;
        assert!(parse_tic_cone_response(partial).is_err());
    }

    #[test]
    fn provenance_is_anonymous_and_enrichment_only() {
        let p = MastTicAdapter.provenance();
        assert_eq!(p.auth, SourceAuth::Anonymous);
        assert!(!p.enters_stellar_backbone);
        assert!(p.endpoint_url.contains("mast.stsci.edu"));
    }

    #[test]
    fn query_hash_is_stable_for_frozen_params() {
        assert_eq!(
            ConeParams::default().query_hash(),
            ConeParams::default().query_hash()
        );
        assert_ne!(
            ConeParams {
                ra_deg: 1.0,
                ..Default::default()
            }
            .query_hash(),
            ConeParams::default().query_hash()
        );
    }
}
