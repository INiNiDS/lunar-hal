//! Anonymous TAP adapter for the NASA Exoplanet Archive `PSCompPars`
//! (Planetary Systems Composite Parameters) table.
//!
//! Contract notes (stage 4A):
//! * Endpoint is public and works without any API key (`SourceAuth::Anonymous`).
//! * This source contributes *enrichment features* (host-star astrophysical
//!   parameters) only — it must never replace or define the stellar backbone
//!   rows themselves, hence `enters_stellar_backbone = false`.
//! * Deterministic parsing: identical CSV bytes always yield identical rows,
//!   which lets CI replay the recorded fixture instead of hitting NASA.

use crate::sources::{SourceAdapter, SourceAuth, SourceProvenance};

pub const PS_COMPPARS_TAP_SYNC_URL: &str = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync";

pub const ADAPTER_ID: &str = "nasa_exoplanet_tap_pscomppars_v1";

/// Frozen default query used by the collector; part of provenance hashing.
pub fn adql_query(top_rows: usize) -> String {
    format!(
        "select top {top_rows} pl_name,hostname,ra,dec,sy_plx,sy_dist,st_teff,st_rad,st_mass,st_lum,disc_year from pscomppars"
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct NasaExoplanetRecord {
    pub planet_name: String,
    pub hostname: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// Stellar parallax, mas (may be absent).
    pub parallax_mas: Option<f64>,
    /// Distance in pc when archive already computed it.
    pub distance_pc: Option<f64>,
    pub teff_k: Option<f64>,
    pub radius_rsun: Option<f64>,
    pub mass_msun: Option<f64>,
    pub luminosity_lsun: Option<f64>,
    pub discovery_year: Option<i32>,
}

fn parse_opt_f64(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("null") || t.eq_ignore_ascii_case("nan") {
        return None;
    }
    let v: f64 = t.parse().ok()?;
    v.is_finite().then_some(v)
}

/// Splits one CSV line honoring double-quoted fields ("a,b" stays one field).
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' => {
                    if chars.peek() == Some(&'"') {
                        cur.push('"');
                        chars.next();
                    } else {
                        in_quotes = false;
                    }
                }
                _ => cur.push(c),
            }
        } else {
            match c {
                ',' => fields.push(std::mem::take(&mut cur)),
                '"' => in_quotes = true,
                _ => cur.push(c),
            }
        }
    }
    fields.push(cur);
    fields
}

/// Public alias so sibling adapters (IRSA/MAST-CSV paths) reuse the exact same
/// quoted-CSV splitting semantics instead of duplicating them.
pub fn split_csv_line_public(line: &str) -> Vec<String> {
    split_csv_line(line)
}

fn header_index(cols: &[String], name: &str) -> usize {
    cols.iter()
        .position(|c| c.eq_ignore_ascii_case(name))
        .unwrap_or(usize::MAX)
}

/// Parses the PSCompPars TAP/CSV payload. Returns an error when required
/// identifier/coordinate columns are missing or a row is unparseable.
pub fn parse_pscomppars_csv(csv: &str) -> Result<Vec<NasaExoplanetRecord>, String> {
    let mut lines = csv.lines();
    let header_line = lines.next().ok_or("empty pscomppars payload")?;
    let cols: Vec<String> = split_csv_line(header_line)
        .into_iter()
        .map(|c| c.trim().to_string())
        .collect();

    let i_pl = header_index(&cols, "pl_name");
    let i_host = header_index(&cols, "hostname");
    let i_ra = header_index(&cols, "ra");
    let i_dec = header_index(&cols, "dec");
    let i_plx = header_index(&cols, "sy_plx");
    let i_dist = header_index(&cols, "sy_dist");
    let i_teff = header_index(&cols, "st_teff");
    let i_rad = header_index(&cols, "st_rad");
    let i_mass = header_index(&cols, "st_mass");
    let i_lum = header_index(&cols, "st_lum");
    let i_year = header_index(&cols, "disc_year");

    if i_pl == usize::MAX || i_ra == usize::MAX || i_dec == usize::MAX || i_host == usize::MAX {
        return Err(format!(
            "pscomppars payload missing required columns (found: {cols:?})"
        ));
    }

    let mut out = Vec::new();
    for (line_no, raw_line) in lines.enumerate() {
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let f = split_csv_line(line);
        let get = |i: usize| f.get(i).map(String::as_str).unwrap_or("");
        let name = get(i_pl).trim().to_string();
        let host = get(i_host).trim().to_string();
        // RA/Dec are mandatory per contract; ID-bearing rows without them are
        // reported instead of silently dropped (ambiguity diagnostics later).
        let (Some(ra), Some(dec)) = (parse_opt_f64(get(i_ra)), parse_opt_f64(get(i_dec))) else {
            return Err(format!(
                "pscomppars row {} ({name}/{host}) has unparseable coordinates",
                line_no + 2
            ));
        };
        out.push(NasaExoplanetRecord {
            planet_name: name,
            hostname: host,
            ra_deg: ra,
            dec_deg: dec,
            parallax_mas: parse_opt_f64(get(i_plx)),
            distance_pc: parse_opt_f64(get(i_dist)),
            teff_k: parse_opt_f64(get(i_teff)),
            radius_rsun: parse_opt_f64(get(i_rad)),
            mass_msun: parse_opt_f64(get(i_mass)),
            luminosity_lsun: parse_opt_f64(get(i_lum)),
            discovery_year: get(i_year).trim().parse::<i32>().ok(),
        });
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct NasaExoplanetAdapter;

impl Default for NasaExoplanetAdapter {
    fn default() -> Self {
        Self
    }
}

impl SourceAdapter for NasaExoplanetAdapter {
    fn id(&self) -> &'static str {
        ADAPTER_ID
    }

    fn provenance(&self) -> SourceProvenance {
        SourceProvenance {
            adapter_id: ADAPTER_ID.to_string(),
            catalog_name: "NASA Exoplanet Archive PSCompPars".to_string(),
            release_or_version: "pscomppars (rolling service)".to_string(),
            endpoint_url: PS_COMPPARS_TAP_SYNC_URL.to_string(),
            auth: SourceAuth::Anonymous,
            enters_stellar_backbone: false,
        }
    }
}

/// Live anonymous download helper (used by CLI/collector paths; CI replays the
/// recorded fixture instead).
pub fn fetch_sync_csv(query: &str) -> Result<String, String> {
    use std::io::Read;
    let url = format!(
        "{}?query={}&format=csv",
        PS_COMPPARS_TAP_SYNC_URL,
        urlencode(query)
    );
    let resp = reqwest::blocking::get(url).map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("NASA TAP returned HTTP {}", resp.status()));
    }
    let mut body = String::new();
    let mut resp = resp;
    resp.read_to_string(&mut body).map_err(|e| e.to_string())?;
    Ok(body)
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::SourceAdapter as _;

    #[test]
    fn parses_recorded_fixture_with_quoted_and_null_fields() {
        const FIXTURE: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/nasa_exoplanet_pscomppars_sample.csv"
        ));
        let recs = parse_pscomppars_csv(FIXTURE).expect("fixture must parse");
        assert_eq!(recs.len(), 500);
        let hd2039 = recs.iter().find(|r| r.planet_name == "HD 2039 b").unwrap();
        assert_eq!(hd2039.hostname, "HD 2039");
        assert!((hd2039.ra_deg - 6.0851069).abs() < 1e-7);
        assert!((hd2039.dec_deg - -56.6499880).abs() < 1e-7);
        assert_eq!(hd2039.parallax_mas, Some(11.6408));
        assert_eq!(hd2039.teff_k, Some(5945.0));
        assert_eq!(hd2039.discovery_year, Some(2002));
    }

    #[test]
    fn rows_missing_coordinates_are_rejected_not_silently_dropped() {
        let csv = "pl_name,hostname,ra,dec\nBad,Host,,\nGood,Host2,10.0,-3.0\n";
        assert!(parse_pscomppars_csv(csv).is_err());
        let csv_ok = "pl_name,hostname,ra,dec\nGood,Host2,10.0,-3.0\n";
        let recs = parse_pscomppars_csv(csv_ok).unwrap();
        assert_eq!(recs.len(), 1);
    }

    #[test]
    fn provenance_marks_enrichment_only_and_anonymous_auth() {
        let p = NasaExoplanetAdapter.provenance();
        assert_eq!(p.adapter_id, ADAPTER_ID);
        assert!(!p.enters_stellar_backbone);
        assert_eq!(p.auth, SourceAuth::Anonymous);
        assert!(p.auth.env_keys().is_empty(), "no keys needed publicly");
        assert!(
            p.endpoint_url
                .starts_with("https://exoplanetarchive.ipac.caltech.edu")
        );
    }

    #[test]
    fn query_contract_is_frozen_and_contains_core_columns() {
        let q = adql_query(1000);
        assert!(q.contains("top 1000"));
        for col in [
            "pl_name", "hostname", "ra", "dec", "sy_plx", "st_teff", "st_rad", "st_mass",
        ] {
            assert!(q.contains(col), "{col} missing");
        }
    }
}
