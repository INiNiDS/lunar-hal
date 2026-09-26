use lnai_data::assemble::{StreamingAssembleOptions, assemble_dataset_streaming};
use lnai_data::clean::CleanPolicy;
use lnai_data::manifest::{DatasetManifestV1, ShardState, ShardStatus};
use std::path::Path;

fn shard(
    dir: &Path,
    manifest: &mut DatasetManifestV1,
    id: &str,
    ra: (f32, f32),
    dec: (f32, f32),
    status: ShardStatus,
) {
    let mut state = ShardState::new(id.into(), ra, dec);
    state.status = status;
    state.row_count = 1;
    manifest.shards.push(state);
    if status == ShardStatus::Verified {
        std::fs::write(
            dir.join(format!("{id}.csv")),
            format!(
                "source_id,ra_deg,dec_deg,parallax_mas,ruwe\n{},0.5,0,5,1\n",
                manifest.shards.len()
            ),
        )
        .unwrap();
    }
}

fn rejected_gap(shards: &[(&str, (f32, f32), (f32, f32), ShardStatus)], location: &str) {
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = DatasetManifestV1::new("gaia_dr3", "fixture", "schema");
    for (id, ra, dec, status) in shards {
        shard(dir.path(), &mut manifest, id, *ra, *dec, *status);
    }
    let result = assemble_dataset_streaming(
        dir.path(),
        &mut manifest,
        &CleanPolicy::default(),
        StreamingAssembleOptions::default(),
    );
    let error = result.unwrap_err();
    assert!(
        error.contains("coverage") && error.contains(location),
        "{error}"
    );
    assert!(
        !dir.path().join("assembled").exists(),
        "fail before writing parts or changing the manifest"
    );
    assert!(manifest.checksum.is_empty());
}

#[test]
fn missing_ra_strip_cannot_be_silently_assembled() {
    rejected_gap(
        &[
            ("left", (0.0, 1.0), (-90.0, 90.0), ShardStatus::Verified),
            ("gap", (1.0, 2.0), (-90.0, 90.0), ShardStatus::Failed),
            ("right", (2.0, 3.0), (-90.0, 90.0), ShardStatus::Verified),
        ],
        "RA [1",
    );
}

#[test]
fn missing_dec_child_cannot_be_silently_assembled() {
    rejected_gap(
        &[
            ("parent", (0.0, 1.0), (-90.0, 90.0), ShardStatus::Failed),
            ("south", (0.0, 1.0), (-90.0, 0.0), ShardStatus::Verified),
            (
                "missing-child",
                (0.0, 1.0),
                (0.0, 10.0),
                ShardStatus::Failed,
            ),
            ("north", (0.0, 1.0), (10.0, 90.0), ShardStatus::Verified),
        ],
        "Dec [0",
    );
}
