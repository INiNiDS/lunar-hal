use lnai_data::assemble::{assemble_dataset, assemble_model_views_from_parquet};
use lnai_data::clean::{CleanPolicy, clean_records, parse_shard_csv};
use lnai_data::manifest::DatasetManifestV1;
use lnai_data::schema::SchemaView;
use polars::prelude::*;
use std::fs::File;

const HEADER: &str = "source_id,ra_deg,dec_deg,parallax_mas,pm_ra_mas_yr,pm_dec_mas_yr,radial_velocity_kms,mag_g,mag_bp,mag_rp,ruwe,astrometric_excess_noise";

fn records() -> Vec<lnai_data::clean::StarRecord> {
    let csv = format!(
        "{HEADER}\n1,10,20,5,2,3,10,9,10,8,1,0.1\n2,11,20,5,2,3,,9,10,8,1,0.1\n3,12,20,5,,3,10,9,10,8,1,0.1\n4,13,20,,2,3,10,9,10,8,1,0.1\n"
    );
    clean_records(parse_shard_csv(&csv).unwrap(), &CleanPolicy::default())
}

#[test]
fn no_cartesian_position_cannot_be_valid_but_missing_rv_does_not_invalidate_position() {
    let recs = records();
    assert!(recs[0].is_valid);
    assert!(recs[1].is_valid, "RV is not needed to calculate a position");
    assert!(
        !recs[3].is_valid,
        "missing parallax means no Cartesian position"
    );
    assert!(recs[3].x_pc.is_none());
    let extreme =
        format!("{HEADER}\n5,15,20,0.000000000000000000000000000000000001,2,3,10,9,10,8,1,0.1\n");
    let rec = clean_records(parse_shard_csv(&extreme).unwrap(), &CleanPolicy::default());
    assert!(
        !rec[0].is_valid,
        "an overflowing f32 position cannot be valid"
    );
}

#[test]
fn gnn_view_has_trainer_features_and_only_measured_full_kinematics() {
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = DatasetManifestV1::new("gaia_dr3", "fixture", "schema");
    let report = assemble_dataset(dir.path(), &mut manifest, &records()).unwrap();
    let path = &report
        .view_paths
        .iter()
        .find(|(view, _)| *view == SchemaView::GnnKinematics)
        .unwrap()
        .1;
    let df = ParquetReader::new(File::open(path).unwrap())
        .finish()
        .unwrap();
    assert_eq!(
        df.height(),
        1,
        "no null RV or incomplete position/velocity targets"
    );
    for name in [
        "source_id",
        "spatial_tile",
        "x_pc",
        "y_pc",
        "z_pc",
        "bp_rp",
        "mag_g",
        "mag_bp",
        "mag_rp",
        "ruwe",
        "radial_velocity_kms",
        "vx_kms",
        "vy_kms",
        "vz_kms",
    ] {
        assert!(df.column(name).is_ok(), "GNN loader needs {name}");
    }
    assert_eq!(df.column("bp_rp").unwrap().f32().unwrap().get(0), Some(2.0));
    assert_eq!(
        df.column("source_id").unwrap().str().unwrap().get(0),
        Some("1")
    );

    let canonical = ParquetReader::new(File::open(&report.canonical_path).unwrap())
        .finish()
        .unwrap();
    assert_eq!(
        canonical.height(),
        4,
        "filtering views must not discard base rows"
    );
}

#[test]
fn enriched_views_are_usable_without_changing_base_canonical() {
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = DatasetManifestV1::new("gaia_dr3", "fixture", "schema");
    let report = assemble_dataset(dir.path(), &mut manifest, &records()).unwrap();
    let base_hash = lnai_data::integrity::sha256_file(&report.canonical_path).unwrap();

    let mut enriched = ParquetReader::new(File::open(&report.canonical_path).unwrap())
        .finish()
        .unwrap();
    enriched
        .with_column(
            Series::new(
                "teff_gspphot".into(),
                &[Some(5700.0_f32), None, Some(4200.0), Some(4800.0)][..],
            )
            .into(),
        )
        .unwrap();
    enriched
        .with_column(
            Series::new(
                "radius_gspphot".into(),
                &[Some(1.0_f32), Some(2.0), Some(1.5), Some(1.0)][..],
            )
            .into(),
        )
        .unwrap();
    enriched
        .with_column(
            Series::new(
                "mass_flame".into(),
                &[Some(1.0_f32), Some(2.0), Some(0.8), Some(1.0)][..],
            )
            .into(),
        )
        .unwrap();
    enriched
        .with_column(
            Series::new(
                "lum_flame".into(),
                &[Some(1.0_f32), Some(2.0), Some(0.4), Some(1.0)][..],
            )
            .into(),
        )
        .unwrap();
    let enriched_path = dir.path().join("enriched.parquet");
    ParquetWriter::new(File::create(&enriched_path).unwrap())
        .finish(&mut enriched)
        .unwrap();
    let enriched_hash = lnai_data::integrity::sha256_file(&enriched_path).unwrap();

    let views = assemble_model_views_from_parquet(&enriched_path).unwrap();
    let load = |view| {
        let path = &views.iter().find(|(v, _)| *v == view).unwrap().1;
        ParquetReader::new(File::open(path).unwrap())
            .finish()
            .unwrap()
    };
    let pinn = load(SchemaView::Pinn);
    assert_eq!(pinn.height(), 2, "only positioned rows with all AP targets");
    for name in [
        "bp_rp",
        "mag_g",
        "x_pc",
        "y_pc",
        "z_pc",
        "teff_gspphot",
        "radius_gspphot",
        "mass_flame",
        "lum_flame",
    ] {
        assert!(pinn.column(name).is_ok(), "PINN loader needs {name}");
    }
    assert_eq!(
        pinn.column("source_id").unwrap().str().unwrap().get(0),
        Some("1")
    );
    let siren = load(SchemaView::Siren);
    assert_eq!(
        siren.height(),
        3,
        "SIREN does not require AP or radial velocity"
    );
    for name in ["bp_rp", "mag_g", "ruwe", "x_pc", "y_pc", "z_pc"] {
        assert!(siren.column(name).is_ok(), "SIREN loader needs {name}");
    }
    assert_eq!(load(SchemaView::GnnKinematics).height(), 1);
    assert_eq!(
        base_hash,
        lnai_data::integrity::sha256_file(&report.canonical_path).unwrap()
    );
    assert_eq!(
        enriched_hash,
        lnai_data::integrity::sha256_file(&enriched_path).unwrap()
    );
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("views_manifest.json")).unwrap())
            .unwrap();
    assert_eq!(meta["source_sha256"], enriched_hash);
    assert_eq!(meta["schema_version"], "1.0.0");
    assert_eq!(meta["generator_version"], "1.0.0");
    assert_eq!(
        meta["canonical_schema_hash"],
        lnai_data::integrity::schema_hash()
    );
    for entry in meta["views"].as_array().unwrap() {
        let path = dir.path().join(entry["file"].as_str().unwrap());
        assert_eq!(
            entry["sha256"],
            lnai_data::integrity::sha256_file(&path).unwrap()
        );
        assert_eq!(
            entry["rows"],
            ParquetReader::new(File::open(path).unwrap())
                .finish()
                .unwrap()
                .height()
        );
    }
}

#[test]
fn partial_ap_enrichment_is_reported_instead_of_producing_broken_pinn_view() {
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = DatasetManifestV1::new("gaia_dr3", "fixture", "schema");
    let report = assemble_dataset(dir.path(), &mut manifest, &records()).unwrap();
    let mut enriched = ParquetReader::new(File::open(&report.canonical_path).unwrap())
        .finish()
        .unwrap();
    enriched
        .with_column(
            Series::new(
                "teff_gspphot".into(),
                vec![Some(5000.0_f32); enriched.height()],
            )
            .into(),
        )
        .unwrap();
    let path = dir.path().join("partial.parquet");
    ParquetWriter::new(File::create(&path).unwrap())
        .finish(&mut enriched)
        .unwrap();
    let error = assemble_model_views_from_parquet(&path).unwrap_err();
    assert!(
        error.contains("incomplete AP") && error.contains("radius_gspphot"),
        "{error}"
    );
}
