//! Integration: NASA Exoplanet Archive offline fixture replay (4A / пункт 4).
//! No network involved — CI must run without credentials.

use lnai_data::sources::nasa_exoplanet::{adql_query, parse_pscomppars_csv};
use lnai_data::sources::{SourceAdapter, SourceAuth};

const FIXTURE: &str = include_str!("fixtures/nasa_exoplanet_pscomppars_sample.csv");

#[test]
fn recorded_fixture_replays_deterministically_without_network() {
    let a = parse_pscomppars_csv(FIXTURE).unwrap();
    let b = parse_pscomppars_csv(FIXTURE).unwrap();
    assert_eq!(a, b, "parsing is deterministic");
    assert!(a.len() >= 50);
}

#[test]
fn fixture_rows_carry_enrichment_features_and_valid_coordinates() {
    use lnai_data::sources::nasa_exoplanet::NasaExoplanetRecord;
    let recs = parse_pscomppars_csv(FIXTURE).unwrap();
    // Real composite dumps contain honest nulls; require healthy *coverage*
    // (a majority of rows carrying core stellar parameters), not perfection.
    let has_mass = |r: &NasaExoplanetRecord| r.mass_msun.is_some();
    let has_teff = |r: &NasaExoplanetRecord| r.teff_k.is_some();
    assert!(
        recs.iter().filter(|r| has_mass(r)).count() * 2 >= recs.len(),
        "mass coverage unexpectedly low"
    );
    assert!(
        recs.iter().filter(|r| has_teff(r)).count() * 2 >= recs.len(),
        "teff coverage unexpectedly low"
    );
    for r in &recs {
        assert!((-90.0..=90.0).contains(&r.dec_deg));
        assert!((0.0..360.0).contains(&r.ra_deg));
    }
}

#[test]
fn parser_preserves_honest_nulls_for_partial_rows() {
    // Synthetic counterpart: composite tables often lack individual stellar
    // parameters; missing values must stay None, never zero-filled.
    let csv = "pl_name,hostname,ra,dec,sy_plx,sy_dist,st_teff,st_rad,st_mass,st_lum,disc_year\n\
               Test b,Star,10.5,-3.25,12.1,,,,,\n";
    let recs = parse_pscomppars_csv(csv).unwrap();
    assert_eq!(recs.len(), 1);
    let r = &recs[0];
    assert_eq!(r.parallax_mas, Some(12.1));
    assert_eq!(r.teff_k, None);
    assert_eq!(r.radius_rsun, None);
    assert_eq!(r.mass_msun, None);
    assert_eq!(r.luminosity_lsun, None);
    assert_eq!(r.discovery_year, None);
}

#[test]
fn adapter_contract_requires_no_secrets_for_public_path() {
    use lnai_data::sources::nasa_exoplanet::NasaExoplanetAdapter;
    let p = NasaExoplanetAdapter.provenance();
    match &p.auth {
        SourceAuth::Anonymous => {}
        SourceAuth::EnvKeys { requires } => panic!("public TAP needs no keys: {requires:?}"),
    }
    assert!(
        adql_query(1).starts_with("select top 1"),
        "query contract drifted: {}",
        adql_query(1)
    );
}
