//! Stage 4A / пункт 10: JPL Horizons Solar System scene provider (anonymous).
//!
//! Ephemerides for Solar System bodies (planets/moons/observatories targets)
//! rendered as *scene* rows: they must NEVER enter the stellar training
//! backbone (`enters_stellar_backbone = false`), per the plan's P2 contract.
//!
//! Live endpoint: `GET {HORIZONS_URL}?format=json&COMMAND='<body>'&...` — all
//! request parameters quoted, `QUANTITIES='1,2'` (apparent RA/Dec + range).
//! The recorded fixture replays Uranus (799) observed from Earth center over
//! one day at 1d step; parsing is stateless and deterministic.

use crate::sources::{SourceAdapter, SourceAuth, SourceProvenance};
use serde_json::Value;

pub const JPL_HORIZONS_URL: &str = "https://ssd.jpl.nasa.gov/api/horizons.api";
pub const ADAPTER_ID: &str = "jpl_horizons_scene_v1";

#[derive(Debug, Clone, PartialEq)]
pub struct SceneRequest {
    /// Horizons body code (e.g. "799" = Uranus, "301" = Moon).
    pub body_code: String,
    pub center: String,
    pub start_time: String,
    pub stop_time: String,
    pub step_size: String,
}

impl Default for SceneRequest {
    fn default() -> Self {
        Self {
            body_code: "799".into(),
            center: "500@399".into(),
            start_time: "2026-08-27".into(),
            stop_time: "2026-08-28".into(),
            step_size: "1d".into(),
        }
    }
}

impl SceneRequest {
    /// Builds the documented query string. Every value is single-quoted — the
    /// unquoted variant triggers Horizons "Too many constants" input errors.
    /// QUANTITIES freezes to astrometric+apparent RA/Dec plus observer range,
    /// which fixes the positional column contract below.
    pub fn to_query(&self) -> String {
        let q = |s: &str| format!("'{s}'");
        format!(
            "format=json&COMMAND={}&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='OBSERVER'&CENTER={}&START_TIME={}&STOP_TIME={}&STEP_SIZE={}&QUANTITIES='1%2C2%2C20'",
            q(&self.body_code),
            q(&self.center),
            q(&self.start_time),
            q(&self.stop_time),
            q(&self.step_size),
        )
    }

    pub fn query_hash(&self) -> String {
        crate::integrity::sha256_hex(self.to_query().as_bytes())
    }
}

/// One ephemeris row extracted from the text payload embedded in the JSON
/// envelope (`result` field) between `$$(...)$$` markers.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SceneEphemerisRow {
    /// UT timestamp exactly as reported ("2026-Aug-27 00:00").
    pub time_utc: String,
    /// Apparent right ascension in degrees (0..=360).
    pub ra_deg: f64,
    /// Apparent declination in degrees (-90..=90).
    pub dec_deg: f64,
    /// Observer range in AU.
    pub delta_au: f64,
}

fn extract_ephemeris_text(result_field: &str) -> &str {
    let Some(start_rel) = result_field.find("$$SOE") else {
        return "";
    };
    let start = start_rel + "$$SOE".len();
    match result_field[start..].find("$$EOE") {
        Some(end_rel) => &result_field[start..start + end_rel],
        None => "",
    }
}

/// Parses fixed-width Horizons OBSERVER lines for the frozen QUANTITIES
/// contract `1,2,20`, whose positional layout after `date time` is exactly:
///   [RA_ast h m s, Dec_ast d m s (signed degrees), RA_app h m s,
///    Dec_app d m s (signed), Delta_AU, range-rate km/s]
pub fn parse_horizons_result(result_field: &str) -> Result<Vec<SceneEphemerisRow>, String> {
    let block = extract_ephemeris_text(result_field);
    if block.is_empty() {
        return Err("horizons result contains no $$SOE..$$EOE data block".to_string());
    }

    let parse_trio = |tokens: &[&str], base: usize| -> Result<f64, String> {
        let big = tokens
            .get(base)
            .and_then(|t| t.parse::<f64>().ok())
            .ok_or_else(|| {
                format!(
                    "unparseable angle at offset {base} in `{}`",
                    tokens.join(" ")
                )
            })?;
        let min = tokens
            .get(base + 1)
            .and_then(|t| t.parse::<f64>().ok())
            .ok_or_else(|| format!("unparseable arcmin near offset {}", base + 1))?;
        let sec = tokens
            .get(base + 2)
            .and_then(|t| t.parse::<f64>().ok())
            .ok_or_else(|| format!("unparseable arcsec near offset {}", base + 2))?;
        Ok(big + min / 60.0 + sec / 3600.0)
    };

    let mut out = Vec::new();
    for line in block.lines() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || !trimmed.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        let mut it = trimmed.split_ascii_whitespace();
        let date = it.next().ok_or("empty ephem row")?;
        let Some(time) = it.next() else { continue };
        let rest: Vec<&str> = it.collect();
        if rest.len() < 13 {
            return Err(format!(
                "ephem row `{trimmed}` does not match the frozen QUANTITIES=1,2,20 column contract"
            ));
        }

        // Astrometric ICRF pair.
        let ra_hours = parse_trio(&rest, 0)?;
        let dec_deg_tok = rest[3];
        let dec_sign = dec_deg_tok.starts_with('-');
        let dec_abs = parse_trio(&rest, 3)?;

        // Apparent pair occupies offsets 6..=11 — validated for shape but the
        // astrometric coordinates are canonical for scene placement.
        let _apparent_ra = parse_trio(&rest, 6)?;
        let _apparent_dec = parse_trio(&rest, 9)?;

        let delta_au = rest[12]
            .parse::<f64>()
            .map_err(|_| format!("range column unparseable in `{trimmed}`"))?;

        out.push(SceneEphemerisRow {
            time_utc: format!("{date} {time}"),
            ra_deg: ra_hours.mul_add(15.0, 0.0).rem_euclid(360.0),
            dec_deg: if dec_sign { -dec_abs } else { dec_abs },
            delta_au,
        });
    }

    if out.is_empty() {
        return Err("no parsable ephemeris rows found inside SOE block".to_string());
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct JplHorizonsAdapter;

impl SourceAdapter for JplHorizonsAdapter {
    fn id(&self) -> &'static str {
        ADAPTER_ID
    }

    fn provenance(&self) -> SourceProvenance {
        SourceProvenance {
            adapter_id: ADAPTER_ID.to_string(),
            catalog_name: "JPL Horizons ephemerides".to_string(),
            release_or_version: "api v1.2 (rolling)".to_string(),
            endpoint_url: JPL_HORIZONS_URL.to_string(),
            auth: SourceAuth::Anonymous,
            enters_stellar_backbone: false,
        }
    }
}

/// Live anonymous download helper (CI replays the recorded fixture).
pub fn fetch_scene(req: &SceneRequest) -> Result<(String, Vec<SceneEphemerisRow>), String> {
    use std::io::Read;
    let url = format!("{JPL_HORIZONS_URL}?{q}", q = req.to_query());
    let resp = reqwest::blocking::get(url).map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("Horizons returned HTTP {}", resp.status()));
    }
    let mut body = String::new();
    let mut resp = resp;
    resp.read_to_string(&mut body).map_err(|e| e.to_string())?;
    let records = parse_horizons_json_envelope(&body)?;
    Ok((body, records))
}

fn parse_horizons_json_envelope(body: &str) -> Result<Vec<SceneEphemerisRow>, String> {
    let root: Value =
        serde_json::from_str(body).map_err(|e| format!("horizons response is not JSON: {e}"))?;
    if let Some(err) = root.get("error").and_then(Value::as_str) {
        return Err(format!("horizons error: {err}"));
    }
    let result = root
        .get("result")
        .and_then(Value::as_str)
        .unwrap_or_default();
    parse_horizons_result(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::SourceAdapter as _;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/jpl_horizons_uranus.json"
    ));

    #[test]
    fn recorded_fixture_parses_three_rows_for_uranus_scene() {
        let recs = parse_horizons_json_envelope(FIXTURE).expect("recorded fixture must parse");
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].time_utc, "2026-Aug-27 00:00");
        // Real values from the recorded dump: RA 04h13m19.15s -> 63.32965 deg,
        // Dec +21d01m02.9s -> 21.017472, Delta 19.4537279825014 AU.
        let expected_ra = (4.0 + 13.0 / 60.0 + 19.15 / 3600.0) * 15.0;
        let expected_dec = 21.0 + 1.0 / 60.0 + 2.9 / 3600.0;
        assert!(
            (recs[0].ra_deg - expected_ra).abs() < 1e-9,
            "{}",
            recs[0].ra_deg
        );
        assert!((recs[0].dec_deg - expected_dec).abs() < 1e-9);
        assert!((recs[0].delta_au - 19.453_727_982_501_4).abs() < 1e-9);
        // Range shrinks monotonically over the three recorded days.
        assert!(recs[2].delta_au < recs[1].delta_au && recs[1].delta_au < recs[0].delta_au);
    }

    #[test]
    fn error_envelope_is_surfaced_not_parsed() {
        let err_body = r#"{"signature":{"version":"1.2"},"result":"","error":" INPUT ERROR ..."}"#;
        assert!(parse_horizons_json_envelope(err_body).is_err());
    }

    #[test]
    fn missing_soe_block_is_rejected() {
        assert!(parse_horizons_result("no markers here").is_err());
    }

    #[test]
    fn provenance_is_anonymous_never_stellar_backbone() {
        let p = JplHorizonsAdapter.provenance();
        assert_eq!(p.auth, SourceAuth::Anonymous);
        assert!(!p.enters_stellar_backbone);
    }

    #[test]
    fn default_request_is_quoted_and_hashed_deterministically() {
        let q = SceneRequest::default().to_query();
        assert!(q.contains("COMMAND='799'"), "{q}");
        assert!(q.contains("QUANTITIES='1%2C2%2C20'"), "{q}");
        assert_eq!(
            SceneRequest::default().query_hash(),
            SceneRequest::default().query_hash()
        );
    }
}
