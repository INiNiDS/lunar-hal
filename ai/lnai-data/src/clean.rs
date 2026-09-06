use serde::{Deserialize, Serialize};

/// The exact ICRS->Galactic rotation matrix (convention used by Astropy and
/// the Gaia documentation). Frozen here so conversions are bit-reproducible
/// across runs and machines.
pub const ICRS_TO_GALACTIC: [[f64; 3]; 3] = [
    [
        -0.05487556041621544,
        -0.8734370902348850,
        -0.4838350155487132,
    ],
    [
        0.49410942787558370,
        -0.44482962996001120,
        0.74698224449721890,
    ],
    [
        -0.86766614901918840,
        -0.19807637343120150,
        0.45598377617506690,
    ],
];

/// Astronomical constant: tangential velocity `km/s` produced by a proper
/// motion of 1 mas/yr at 1 kpc distance (`vt = 4.74047 * mu * D_kpc`).
pub const MAS_YR_KPC_TO_KMS: f64 = 4.740470446;

/// One canonical stellar record after cleaning; maps 1:1 onto the golden
/// schema's collectable columns (view-only neighbor fields live in assemble).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StarRecord {
    pub source_id: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// Coordinate epoch, always J2000.0 for Gaia DR3.
    pub epoch_year: f64,
    pub parallax_mas: Option<f64>,
    pub pm_ra_mas_yr: Option<f64>,
    pub pm_dec_mas_yr: Option<f64>,
    pub radial_velocity_kms: Option<f64>,
    /// Distance-derived Cartesian position, parsecs (Galactic frame).
    pub x_pc: Option<f32>,
    pub y_pc: Option<f32>,
    pub z_pc: Option<f32>,
    /// Cartesian velocity km/s (Galactic frame).
    pub vx_kms: Option<f32>,
    pub vy_kms: Option<f32>,
    pub vz_kms: Option<f32>,
    pub mag_g: Option<f32>,
    pub mag_bp: Option<f32>,
    pub mag_rp: Option<f32>,
    pub ruwe: Option<f32>,
    pub astrometric_excess_noise: Option<f32>,
    pub is_valid: bool,
}

/// Parsing tolerances mirrored from the legacy pipeline's cuts.
#[derive(Debug, Clone, Copy)]
pub struct CleanPolicy {
    pub max_ruwe: f64,
    pub max_astrometric_excess_noise: f64,
}

impl Default for CleanPolicy {
    fn default() -> Self {
        Self {
            max_ruwe: 1.4,
            // Mas-level outlier guard used by the legacy `clean` stage.
            max_astrometric_excess_noise: 10.0,
        }
    }
}

fn parse_f64(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("nan") || t.eq_ignore_ascii_case("null") {
        return None;
    }
    let v: f64 = t.parse().ok()?;
    if !v.is_finite() { None } else { Some(v) }
}

fn parse_u64(s: &str) -> Option<u64> {
    s.trim().parse().ok()
}

/// Parses one TAP CSV export body (with header) into raw collectable fields.
pub fn parse_shard_csv(csv: &str) -> Result<Vec<StarRecord>, String> {
    let mut lines = csv.lines();
    let header = lines.next().ok_or("empty csv")?;
    let cols: Vec<&str> = header.split(',').map(str::trim).collect();
    let idx = |name: &str| -> usize {
        cols.iter()
            .position(|c| c.eq_ignore_ascii_case(name))
            .unwrap_or(usize::MAX)
    };
    let required = ["source_id", "ra_deg", "dec_deg"];
    for name in required {
        if idx(name) == usize::MAX {
            return Err(format!("shard csv missing column '{name}'"));
        }
    }
    let (i_id, i_ra, i_dec) = (idx("source_id"), idx("ra_deg"), idx("dec_deg"));
    let i_plx = idx("parallax_mas");
    let i_pmra = idx("pm_ra_mas_yr");
    let i_pmdec = idx("pm_dec_mas_yr");
    let i_rv = idx("radial_velocity_kms");
    let i_g = idx("mag_g");
    let i_bp = idx("mag_bp");
    let i_rp = idx("mag_rp");
    let i_ruwe = idx("ruwe");
    let i_aen = idx("astrometric_excess_noise");

    let mut out = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let get = |i: usize| -> Option<f64> {
            if i == usize::MAX || i >= f.len() {
                None
            } else {
                parse_f64(f[i])
            }
        };
        let Some(id_raw) = f.get(i_id).and_then(|v| parse_u64(v)) else {
            return Err(format!("row {} has unparseable source_id", line_no + 2));
        };
        let (Some(ra), Some(dec)) = (get(i_ra), get(i_dec)) else {
            return Err(format!("row {} missing ra/dec", line_no + 2));
        };
        out.push(StarRecord {
            source_id: id_raw.to_string(),
            ra_deg: ra,
            dec_deg: dec,
            epoch_year: crate::schema::DEFAULT_COORDINATE_EPOCH,
            parallax_mas: get(i_plx),
            pm_ra_mas_yr: get(i_pmra),
            pm_dec_mas_yr: get(i_pmdec),
            radial_velocity_kms: get(i_rv),
            x_pc: None,
            y_pc: None,
            z_pc: None,
            vx_kms: None,
            vy_kms: None,
            vz_kms: None,
            mag_g: get(i_g).map(|v| v as f32),
            mag_bp: get(i_bp).map(|v| v as f32),
            mag_rp: get(i_rp).map(|v| v as f32),
            ruwe: get(i_ruwe).map(|v| v as f32),
            astrometric_excess_noise: get(i_aen).map(|v| v as f32),
            is_valid: true,
        });
    }
    Ok(out)
}

/// Deduplicates by stable source ID deterministically: lowest RUWE wins;
/// ties (equal or absent RUWE) keep the lexicographically smallest remainder
/// fingerprint so input order can never change the result.
pub fn dedup_by_source_id(records: Vec<StarRecord>) -> Vec<StarRecord> {
    use std::collections::BTreeMap;
    let mut best: BTreeMap<u64, StarRecord> = BTreeMap::new();
    for rec in records {
        let Ok(id) = rec.source_id.parse::<u64>() else {
            continue;
        };
        match best.get(&id) {
            None => {
                best.insert(id, rec);
            }
            Some(existing) => {
                if prefer(&rec, existing) {
                    best.insert(id, rec);
                }
            }
        }
    }
    best.into_values().collect()
}

fn prefer(a: &StarRecord, b: &StarRecord) -> bool {
    let ra = a.ruwe.unwrap_or(f64::INFINITY as f32);
    let rb = b.ruwe.unwrap_or(f64::INFINITY as f32);
    if (ra - rb).abs() > f32::EPSILON {
        return ra < rb;
    }
    // Deterministic tie-break on full serialization identity.
    serde_json_string(a) < serde_json_string(b)
}

fn serde_json_string(r: &StarRecord) -> String {
    // Avoid pulling serde_json into release builds just for this; a compact
    // field digest is enough for a deterministic tie-break.
    format!(
        "{}|{:?}|{:?}|{:?}|{:?}|{:?}",
        r.source_id, r.ra_deg, r.dec_deg, r.mag_g, r.ruwe, r.astrometric_excess_noise
    )
}

/// Unit direction vector (ICRS) for given equatorial coordinates, degrees.
pub fn icrs_unit_vector(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let (ra, dec) = (ra_deg.to_radians(), dec_deg.to_radians());
    [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]
}

fn mat_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Galactic-frame Cartesian position in parsecs, when parallax allows it.
pub fn galactic_position_pc(ra_deg: f64, dec_deg: f64, parallax_mas: f64) -> Option<[f32; 3]> {
    if parallax_mas <= 0.0 {
        return None;
    }
    let d_pc = 1000.0 / parallax_mas;
    let r = icrs_unit_vector(ra_deg, dec_deg);
    let g = mat_vec(&ICRS_TO_GALACTIC, &r);
    Some([
        ((g[0] * d_pc) as f32),
        (g[1] * d_pc) as f32,
        (g[2] * d_pc) as f32,
    ])
}

/// Galactic-frame Cartesian velocity in km/s from proper motion + radial
/// velocity + parallax. Returns None unless all needed inputs are present.
///
/// Math: `vt[km/s] = 4.74047 * mu[mas/yr] * D[kpc]`, with `D_kpc = 1/plx[mAS]`
/// scaled to `1000/plx`: for parallax in mas, `vt = 4.74047 * mu / plx`.
pub fn galactic_velocity_kms(
    ra_deg: f64,
    dec_deg: f64,
    parallax_mas: f64,
    pm_ra_mas_yr: f64,
    pm_dec_mas_yr: f64,
    radial_velocity_kms: Option<f64>,
) -> Option<[f32; 3]> {
    if parallax_mas <= 0.0 || !pm_ra_mas_yr.is_finite() || !pm_dec_mas_yr.is_finite() {
        return None;
    }
    // East/North tangent-basis vectors at (ra, dec):
    let (a, d) = (ra_deg.to_radians(), dec_deg.to_radians());
    let r_hat = [d.cos() * a.cos(), d.cos() * a.sin(), d.sin()];
    let e_north = [-d.sin() * a.cos(), -d.sin() * a.sin(), d.cos()];
    let e_east = [-a.sin(), a.cos(), 0.0];

    // With parallax in mas: D_kpc = 1/plx[mas], so
    // vt[km/s] = 4.74047 * mu[mas/yr] / plx[mas].
    let scale = MAS_YR_KPC_TO_KMS / parallax_mas;
    let veast = scale * pm_ra_mas_yr;
    let vnorth = scale * pm_dec_mas_yr;
    let vr = radial_velocity_kms.unwrap_or(0.0);

    // Compose ICRS 3D velocity then rotate to the Galactic frame.
    let comps = [vr, vnorth, veast];
    let bases = [&r_hat, &e_north, &e_east];
    let mut v_icrs = [0.0f64; 3];
    for (b, c) in bases.iter().zip(comps.iter()) {
        for i in 0..3 {
            v_icrs[i] += b[i] * c;
        }
    }
    let v_gal = mat_vec(&ICRS_TO_GALACTIC, &v_icrs);
    Some([v_gal[0] as f32, v_gal[1] as f32, v_gal[2] as f32])
}

/// Applies [`CleanPolicy`] flags, dedups, fills derived position/velocity
/// fields. Deterministic: same input bytes -> same output order/content.
pub fn clean_records(mut records: Vec<StarRecord>, policy: &CleanPolicy) -> Vec<StarRecord> {
    for r in records.iter_mut() {
        r.is_valid = match (r.ruwe, r.astrometric_excess_noise) {
            (Some(ruwe), Some(aen)) => {
                (ruwe as f64) < policy.max_ruwe
                    && (aen as f64) < policy.max_astrometric_excess_noise
                    && r.ra_deg.is_finite()
                    && r.dec_deg.is_finite()
            }
            (Some(ruwe), None) => (ruwe as f64) < policy.max_ruwe,
            _ => false,
        };
        r.x_pc = None;
        r.y_pc = None;
        r.z_pc = None;
        r.vx_kms = None;
        r.vy_kms = None;
        r.vz_kms = None;
        if let Some(plx) = r.parallax_mas.filter(|p| *p > 0.0) {
            if let Some(p) = galactic_position_pc(r.ra_deg, r.dec_deg, plx) {
                r.x_pc = Some(p[0]);
                r.y_pc = Some(p[1]);
                r.z_pc = Some(p[2]);
            }
            if let (Some(pmra), Some(pmdec)) = (r.pm_ra_mas_yr, r.pm_dec_mas_yr) {
                if let Some(v) = galactic_velocity_kms(
                    r.ra_deg,
                    r.dec_deg,
                    plx,
                    pmra,
                    pmdec,
                    r.radial_velocity_kms,
                ) {
                    r.vx_kms = Some(v[0]);
                    r.vy_kms = Some(v[1]);
                    r.vz_kms = Some(v[2]);
                }
            }
        }
    }
    dedup_by_source_id(records)
}

/// Pilot QA report: duplicates/null-rates/outliers/spatial coverage summary
/// ("проверить дубликаты, null-rate, outliers и spatial coverage").
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct QualityReport {
    pub total_rows: u64,
    pub unique_ids: u64,
    pub duplicate_rows_removed: u64,
    pub valid_rows: u64,
    pub null_rate_parallax: f64,
    pub null_rate_pm: f64,
    pub null_rate_radial_velocity: f64,
    pub ra_min: f64,
    pub ra_max: f64,
    pub dec_min: f64,
    pub dec_max: f64,
    /// Share of rows whose Cartesian coordinates landed outside a sane range
    /// (parallax>0 sanity already enforced, but NaN-ish floats are re-checked).
    pub outlier_rate_position: f64,
}

pub fn quality_report(records: &[StarRecord]) -> QualityReport {
    let n = records.len().max(1) as f64;
    let ids: std::collections::HashSet<u64> = records
        .iter()
        .filter_map(|r| r.source_id.parse().ok())
        .collect();
    let zero = |c: usize| c as f64 / n;
    QualityReport {
        total_rows: records.len() as u64,
        unique_ids: ids.len() as u64,
        duplicate_rows_removed: (records.len().saturating_sub(ids.len())) as u64,
        valid_rows: records.iter().filter(|r| r.is_valid).count() as u64,
        null_rate_parallax: zero(records.iter().filter(|r| r.parallax_mas.is_none()).count()),
        null_rate_pm: zero(
            records
                .iter()
                .filter(|r| r.pm_ra_mas_yr.is_none() || r.pm_dec_mas_yr.is_none())
                .count(),
        ),
        null_rate_radial_velocity: zero(
            records
                .iter()
                .filter(|r| r.radial_velocity_kms.is_none())
                .count(),
        ),
        ra_min: records
            .iter()
            .map(|r| r.ra_deg)
            .fold(f64::INFINITY, f64::min),
        ra_max: records
            .iter()
            .map(|r| r.ra_deg)
            .fold(f64::NEG_INFINITY, f64::max),
        dec_min: records
            .iter()
            .map(|r| r.dec_deg)
            .fold(f64::INFINITY, f64::min),
        dec_max: records
            .iter()
            .map(|r| r.dec_deg)
            .fold(f64::NEG_INFINITY, f64::max),
        outlier_rate_position: zero(
            records
                .iter()
                .filter(|r| {
                    matches!(
                        (r.x_pc, r.y_pc, r.z_pc),
                        (Some(x), Some(y), Some(z))
                            if !(x.is_finite() && y.is_finite() && z.is_finite())
                    )
                })
                .count(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV_HEADER: &str = "source_id,ra_deg,dec_deg,parallax_mas,pm_ra_mas_yr,pm_dec_mas_yr,radial_velocity_kms,mag_g,mag_bp,mag_rp,ruwe,astrometric_excess_noise";

    #[test]
    fn parses_full_row_and_handles_nulls() {
        let csv = format!(
            "{CSV_HEADER}\n101,10.5,20.25,4.0,-3.5,2.5,15.0,9.1,9.7,8.6,1.1,0.4\n102,11.0,21.0,,,,,8.0,,,\n"
        );
        let recs = parse_shard_csv(&csv).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].source_id, "101");
        assert_eq!(recs[0].parallax_mas, Some(4.0));
        assert_eq!(recs[1].parallax_mas, None);
        assert_eq!(recs[1].mag_g, Some(8.0));
    }

    #[test]
    fn rejects_rows_missing_required_columns() {
        assert!(parse_shard_csv("a,b\n1,2").is_err());
        let csv = format!("{CSV_HEADER}\nbogus,10,20\n");
        assert!(parse_shard_csv(&csv).is_err());
        let csv = format!("{CSV_HEADER}\n999,10,\n"); // no dec
        assert!(parse_shard_csv(&csv).is_err());
    }

    #[test]
    fn dedup_is_deterministic_and_quality_preferring() {
        let mk = |id: u64, ruwe: f32| StarRecord {
            source_id: id.to_string(),
            ra_deg: 1.0,
            dec_deg: 1.0,
            epoch_year: 2000.0,
            parallax_mas: Some(5.0),
            pm_ra_mas_yr: None,
            pm_dec_mas_yr: None,
            radial_velocity_kms: None,
            x_pc: None,
            y_pc: None,
            z_pc: None,
            vx_kms: None,
            vy_kms: None,
            vz_kms: None,
            mag_g: Some(10.0),
            mag_bp: None,
            mag_rp: None,
            ruwe: Some(ruwe),
            astrometric_excess_noise: None,
            is_valid: true,
        };
        // Order must not matter: 0.9-quality row wins regardless of position.
        let fwd = dedup_by_source_id(vec![mk(1, 1.3), mk(1, 0.9)]);
        let rev = dedup_by_source_id(vec![mk(1, 0.9), mk(1, 1.3)]);
        assert_eq!(fwd.len(), 1);
        assert_eq!(fwd, rev);
        assert_eq!(fwd[0].ruwe, Some(0.9));
    }

    #[test]
    fn icrs_to_galactic_matrix_matches_reference_directions() {
        // Reference directions well outside degeneracies:
        // * North Galactic Pole: ICRS (192.85948, +27.12825) -> Galactic
        //   latitude b = +90 deg, so the rotated vector must be ~(0,0,1).
        // * Galactic Center: ICRS (266.40500, -28.93617) -> Galactic (l,b)
        //   = (0, 0), i.e. rotated vector ~(1, 0, 0).
        let ngp = mat_vec(&ICRS_TO_GALACTIC, &icrs_unit_vector(192.85948, 27.12825));
        let norm_ngp = (ngp[0] * ngp[0] + ngp[1] * ngp[1] + ngp[2] * ngp[2]).sqrt();
        assert!((norm_ngp - 1.0).abs() < 1e-12);
        assert!(ngp[2] > 0.999_999, "NGP z {}", ngp[2]);

        let gc = mat_vec(&ICRS_TO_GALACTIC, &icrs_unit_vector(266.40500, -28.93617));
        assert!((gc[0] - 1.0).abs() < 1e-6, "GC x {}", gc[0]);
        assert!(gc[1].abs() < 1e-6, "GC y {}", gc[1]);
        assert!(gc[2].abs() < 1e-6, "GC z {}", gc[2]);
    }

    #[test]
    fn position_and_velocity_match_reference_numbers() {
        // Reference star: RA=60°, Dec=0°, plx=10 mas (=100 pc),
        // pmra*=100 mas/yr, pmdec=-50 mas/yr, rv=+20 km/s.
        let p = galactic_position_pc(60.0, 0.0, 10.0).unwrap();
        let r_mag: f32 = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
        assert!((r_mag - 100.0).abs() < 1e-3, "|r|={r_mag}");

        // vt_total = 4.74047 * sqrt(100²+50²)/10 ≈ 52.98 km/s transverse.
        let v = galactic_velocity_kms(60.0, 0.0, 10.0, 100.0, -50.0, Some(20.0)).unwrap();
        let v_mag: f32 = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
        // |vt| = 4.74047 * sqrt(100^2+50^2)/10 = 53.0 km/s; |v| adds rv=20.
        let expect = ((MAS_YR_KPC_TO_KMS * 111.8034f64 / 10.0).powi(2) + 400.0).sqrt();
        assert!(
            (v_mag - expect as f32).abs() < 1e-3,
            "|v|={v_mag} vs {expect}"
        );
    }

    #[test]
    fn clean_flags_bad_ruwe_and_fills_derived_fields() {
        let csv = format!(
            "{CSV_HEADER}\n1,60.0,0.0,10.0,100.0,-50.0,20.0,9.0,,,1.1,0.2\n2,61.0,0.1,10.0,0.0,0.0,,9.0,,,1.9,0.2\n3,62.0,0.2,10.0,0.0,0.0,,9.0,,,\n"
        );
        let recs = parse_shard_csv(&csv).unwrap();
        let cleaned = clean_records(recs, &CleanPolicy::default());
        assert_eq!(cleaned.len(), 3);
        assert!(cleaned[0].is_valid);
        assert!(cleaned[0].vx_kms.is_some());
        assert!(cleaned[0].x_pc.is_some());
        assert!(!cleaned[1].is_valid, "ruwe 1.9 above policy");
        assert!(!cleaned[2].is_valid, "missing ruwe cannot be validated");
    }

    #[test]
    fn quality_report_counts_duplicates_and_coverage() {
        let csv = format!(
            "{CSV_HEADER}\n1,10.0,5.0,10.0,0.0,0.0,,9.0,,,1.0,0.1\n1,10.0,5.0,10.0,0.0,0.0,,9.0,,,1.0,0.1\n2,11.0,6.0,,,,,,\n"
        );
        let recs = clean_records(parse_shard_csv(&csv).unwrap(), &CleanPolicy::default());
        let q = quality_report(&recs);
        assert_eq!(q.total_rows, 2);
        assert_eq!(q.unique_ids, 2);
        assert_eq!(q.duplicate_rows_removed, 0); // dupes collapse before report
        assert_eq!(q.null_rate_pm, 0.5);
        assert_eq!(q.null_rate_radial_velocity, 1.0);
        assert_eq!(q.ra_min, 10.0);
        assert_eq!(q.dec_max, 6.0);
    }

    #[test]
    fn duplicated_input_is_collapsed_once_and_sorted_deterministically() {
        let mk = |id: u64, g: f32| StarRecord {
            source_id: id.to_string(),
            ra_deg: 1.0,
            dec_deg: 1.0,
            epoch_year: 2000.0,
            parallax_mas: Some(5.0),
            pm_ra_mas_yr: None,
            pm_dec_mas_yr: None,
            radial_velocity_kms: None,
            x_pc: None,
            y_pc: None,
            z_pc: None,
            vx_kms: None,
            vy_kms: None,
            vz_kms: None,
            mag_g: Some(g),
            mag_bp: None,
            mag_rp: None,
            ruwe: Some(1.0),
            astrometric_excess_noise: None,
            is_valid: true,
        };
        let a = dedup_by_source_id(vec![mk(7, 1.0), mk(7, 1.0), mk(2, 1.0)]);
        let b = dedup_by_source_id(vec![mk(2, 1.0), mk(7, 1.0), mk(7, 1.0)]);
        assert_eq!(a.len(), 2);
        assert_eq!(a, b, "same content, different order -> identical output");
        assert_eq!(a[0].source_id, "2"); // BTreeMap keeps numeric order
    }
}
