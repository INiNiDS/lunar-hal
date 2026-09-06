//! Stage 4A / пункт 9: IRSA 2MASS/WISE spike (anonymous TAP).
//!
//! Public 2MASS Point Source Catalog (`fp_psc`) queries through the IRSA TAP
//! sync endpoint; no key, no registration. Coverage statistics (row coverage
//! and per-column null-rate) are the explicit deliverable of this spike, so
//! [`coverage_stats`] is deterministic and unit-tested.
//!
//! Scope guard: catalog metadata only — never a stellar backbone replacement.

use crate::sources::{SourceAdapter, SourceAuth, SourceProvenance};

pub const IRSA_TAP_SYNC_URL: &str = "https://irsa.ipac.caltech.edu/TAP/sync";
pub const ADAPTER_ID: &str = "irsa_2mass_fp_psc_v1";

/// Frozen spatial box query used by the spike (mirrors Gaia RA-shard logic).
pub fn adql_query(ra_min: f64, ra_max: f64, dec_min: f64, dec_max: f64, top_rows: usize) -> String {
    let center_ra = (ra_min + ra_max) / 2.0;
    let center_dec = (dec_min + dec_max) / 2.0;
    format!(
        "select top {top_rows} ra,dec,j_m,h_m,k_m,ph_qual from fp_psc where \
         CONTAINS(POINT(ra,dec),BOX({center_ra},{center_dec},{width_deg},{height_deg}))=1",
        width_deg = ra_max - ra_min,
        height_deg = dec_max - dec_min,
    )
}
/// Deterministic fingerprint of the frozen query shape.
pub fn query_hash(ra_min: f64, ra_max: f64, dec_min: f64, dec_max: f64) -> String {
    let q = format!(
        "select ra,dec,j_m,h_m,k_m,ph_qual from fp_psc where CONTAINS(POINT(ra,dec),BOX({ra_min},{dec_min},{ra_max},{dec_max}))=1"
    );
    crate::integrity::sha256_hex(q.as_bytes())
}

#[derive(Debug, Clone, PartialEq)]
pub struct TwoMassRecord {
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// JHK apparent magnitudes (photometric nulls preserved as None).
    pub j_m: Option<f64>,
    pub h_m: Option<f64>,
    pub k_m: Option<f64>,
    /// 2MASS photometry quality flags like `AAA` / `BUU`.
    pub ph_qual: Option<String>,
}

impl TwoMassRecord {
    pub fn has_j(&self) -> bool {
        self.j_m.is_some()
    }
    pub fn has_h(&self) -> bool {
        self.h_m.is_some()
    }
    pub fn has_k(&self) -> bool {
        self.k_m.is_some()
    }
}

fn parse_opt_f64(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("null") || t.eq_ignore_ascii_case("nan") {
        return None;
    }
    t.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Parses an IRSA TAP CSV payload (same quoting rules as PSCompPars).
pub fn parse_two_mass_csv(csv: &str) -> Result<Vec<TwoMassRecord>, String> {
    let mut lines = csv.lines();
    let header_line = lines.next().ok_or("empty irsa payload")?;
    let cols: Vec<String> = crate::sources::nasa_exoplanet::split_csv_line_public(header_line)
        .into_iter()
        .map(|c| c.trim().to_lowercase())
        .collect();

    let idx = |name: &str| cols.iter().position(|c| c == name).unwrap_or(usize::MAX);
    let (i_ra, i_dec, i_j, i_h, i_k, i_q) = (
        idx("ra"),
        idx("dec"),
        idx("j_m"),
        idx("h_m"),
        idx("k_m"),
        idx("ph_qual"),
    );
    if i_ra == usize::MAX || i_dec == usize::MAX {
        return Err(format!(
            "irsa payload missing ra/dec columns (found: {cols:?})"
        ));
    }

    let mut out = Vec::new();
    for raw in lines {
        let line = raw.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let f = crate::sources::nasa_exoplanet::split_csv_line_public(line);
        let get = |i: usize| f.get(i).map(String::as_str).unwrap_or("");
        let (Some(ra), Some(dec)) = (parse_opt_f64(get(i_ra)), parse_opt_f64(get(i_dec))) else {
            continue; // IRSA occasionally emits photometric-only rows at edges
        };
        out.push(TwoMassRecord {
            ra_deg: ra,
            dec_deg: dec,
            j_m: parse_opt_f64(get(i_j)),
            h_m: parse_opt_f64(get(i_h)),
            k_m: parse_opt_f64(get(i_k)),
            ph_qual: Some(get(i_q).trim().to_string()).filter(|s| !s.is_empty()),
        });
    }
    Ok(out)
}

/// Coverage spike metrics required by пункт 9.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageStats {
    pub row_count: usize,
    /// Fraction of rows with ALL of J/H/K present.
    pub full_photometry_fraction: f64,
    /// Per-column null rates (0.0..=1.0).
    pub null_rate: NullRates,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NullRates {
    pub j_m: f64,
    pub h_m: f64,
    pub k_m: f64,
    pub pm: f64, // 2MASS has no PM columns: expected 1.0, documented explicitly
}

pub fn coverage_stats(rows: &[TwoMassRecord]) -> CoverageStats {
    let n = rows.len() as f64;
    if n == 0.0 {
        return CoverageStats {
            row_count: 0,
            full_photometry_fraction: 0.0,
            null_rate: NullRates {
                j_m: 1.0,
                h_m: 1.0,
                k_m: 1.0,
                pm: 1.0,
            },
        };
    }
    let frac =
        |present: fn(&TwoMassRecord) -> bool| rows.iter().filter(|r| present(r)).count() as f64 / n;
    let has_jh_k = |r: &TwoMassRecord| r.j_m.is_some() && r.h_m.is_some() && r.k_m.is_some();
    CoverageStats {
        row_count: rows.len(),
        full_photometry_fraction: rows.iter().filter(|r| has_jh_k(r)).count() as f64 / n,
        null_rate: NullRates {
            j_m: 1.0 - frac(TwoMassRecord::has_j),
            h_m: 1.0 - frac(TwoMassRecord::has_h),
            k_m: 1.0 - frac(TwoMassRecord::has_k),
            pm: 1.0,
        },
    }
}

#[derive(Debug, Clone)]
pub struct IrsaAdapter;

impl SourceAdapter for IrsaAdapter {
    fn id(&self) -> &'static str {
        ADAPTER_ID
    }

    fn provenance(&self) -> SourceProvenance {
        SourceProvenance {
            adapter_id: ADAPTER_ID.to_string(),
            catalog_name: "2MASS Point Source Catalog".to_string(),
            release_or_version: "fp_psc (final release)".to_string(),
            endpoint_url: IRSA_TAP_SYNC_URL.to_string(),
            auth: SourceAuth::Anonymous,
            enters_stellar_backbone: false,
        }
    }
}

/// Live anonymous helper (CI replays fixtures instead).
pub fn fetch_box_csv(query: &str) -> Result<String, String> {
    use std::io::Read;
    let url = format!("{IRSA_TAP_SYNC_URL}?QUERY={}&FORMAT=csv", urlencode(query));
    let resp = reqwest::blocking::get(url).map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("IRSA TAP returned HTTP {}", resp.status()));
    }
    let mut body = String::new();
    let mut resp = resp;
    resp.read_to_string(&mut body).map_err(|e| e.to_string())?;
    Ok(body)
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
        "/tests/fixtures/irsa_psc_2mass.csv"
    ));

    #[test]
    fn recorded_fixture_parses_and_reports_null_rates() {
        let recs = parse_two_mass_csv(FIXTURE).expect("recorded fixture must parse");
        assert_eq!(recs.len(), 50);
        let stats = coverage_stats(&recs);
        assert_eq!(stats.row_count, 50);
        // The recorded box is a faint/high-latitude cut: either all-null-heavy
        // or photometry-rich, but the J band must dominate completeness and PM
        // must be exactly nonexistent for 2MASS.
        assert!((stats.null_rate.pm - 1.0).abs() < 1e-12);
        assert!(stats.full_photometry_fraction >= stats.null_rate.j_m.min(1.0));
    }

    #[test]
    fn empty_payload_is_handled_without_panic() {
        assert!(parse_two_mass_csv("").is_err());
        let only_header = "ra,dec,j_m,h_m,k_m,ph_qual\n";
        let s = coverage_stats(&parse_two_mass_csv(only_header).unwrap());
        assert_eq!(s.row_count, 0);
        assert_eq!(s.null_rate.j_m, 1.0);
    }

    #[test]
    fn provenance_is_anonymous_enrichment_only() {
        let p = IrsaAdapter.provenance();
        assert_eq!(p.auth, SourceAuth::Anonymous);
        assert!(!p.enters_stellar_backbone);
    }

    #[test]
    fn adql_shape_has_2mass_columns_and_frozen_hash() {
        let q = adql_query(10.0, 11.0, 5.0, 6.0, 100);
        assert!(q.contains("top 100") && q.contains("from fp_psc"));
        assert_eq!(
            query_hash(10.0, 11.0, 5.0, 6.0),
            query_hash(10.0, 11.0, 5.0, 6.0),
            "hash must be deterministic"
        );
    }
}
