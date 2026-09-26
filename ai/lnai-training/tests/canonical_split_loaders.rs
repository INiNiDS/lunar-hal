use burn::backend::NdArray;
use lnai_models::SIREN_INPUT_DIM;
use lnai_training::gnn::dataset::GnnDataset;
use lnai_training::pinn::dataset::StellarDataset;
use lnai_training::siren::dataset::{PrefetchBatcher, SirenDataset};
use polars::prelude::*;

fn frame() -> DataFrame {
    df![
        "x_pc" => [10.0f32, 11.0, 12.0, 20.0, 21.0, 22.0, 1000.0, 1001.0, 2000.0, 2001.0],
        "y_pc" => [1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0],
        "z_pc" => [1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0],
        "bp_rp" => [1.0f32, 1.2, 1.4, 2.0, 2.2, 2.4, 6.0, 6.2, 8.0, 8.2],
        "mag_g" => [10.0f32, 10.1, 10.2, 11.0, 11.1, 11.2, 12.0, 12.1, 13.0, 13.1],
        "mag_bp" => [11.0f32, 11.1, 11.2, 12.0, 12.1, 12.2, 13.0, 13.1, 14.0, 14.1],
        "mag_rp" => [9.0f32, 9.1, 9.2, 10.0, 10.1, 10.2, 11.0, 11.1, 12.0, 12.1],
        "ruwe" => [1.0f32, 1.2, 1.4, 2.0, 2.2, 2.4, 3.0, 3.2, 4.0, 4.2],
        "teff_gspphot" => [5000.0f32, 5100.0, 5200.0, 6000.0, 6100.0, 6200.0, 7000.0, 7100.0, 8000.0, 8100.0],
        "radius_gspphot" => [1.0f32, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 4.0, 4.1],
        "mass_flame" => [1.0f32, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 4.0, 4.1],
        "lum_flame" => [1.0f32, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 4.0, 4.1],
        "radial_velocity_kms" => [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "vx_kms" => [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "vy_kms" => [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "vz_kms" => [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "spatial_tile" => ["a", "a", "a", "b", "b", "b", "c", "c", "d", "d"],
    ].unwrap()
}

fn write_frame(mut frame: DataFrame) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stars.parquet");
    ParquetWriter::new(std::fs::File::create(&path).unwrap())
        .finish(&mut frame)
        .unwrap();
    (dir, path)
}

fn with_split(splits: &[Option<&str>]) -> DataFrame {
    let mut df = frame();
    df.with_column(Series::new("split".into(), splits).into())
        .unwrap();
    df
}

fn canonical_splits() -> [Option<&'static str>; 10] {
    [
        Some("train"),
        Some("train"),
        Some("train"),
        Some("validation"),
        Some("validation"),
        Some("validation"),
        Some("test"),
        Some("test"),
        None,
        None,
    ]
}

fn gnn_groups(path: &std::path::Path) -> (GnnDataset, GnnDataset) {
    GnnDataset::load_with_seed(path, 2, 8, 3.0, 42, None, None)
        .unwrap()
        .split_with_seed(0.5, 7)
}

fn siren_bp_rp_values(
    dataset: &SirenDataset,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> Vec<f32> {
    let mut batcher = PrefetchBatcher::new(dataset, 7);
    let mut values = Vec::with_capacity(dataset.n_samples);
    while let Some((inputs, _targets)) = batcher.next_batch::<NdArray<f32>>(device) {
        let inputs: Vec<f32> = inputs.into_data().to_vec().unwrap();
        values.extend(
            inputs
                .chunks_exact(SIREN_INPUT_DIM)
                .map(|row| row[2] * dataset.norm.bp_rp_std + dataset.norm.bp_rp_mean),
        );
    }
    values
}

#[test]
fn canonical_splits_are_authoritative_and_norm_uses_train_only() {
    let (_dir, path) = write_frame(with_split(&canonical_splits()));
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let ds = StellarDataset::<NdArray<f32>>::load(&path, &device, None, None).unwrap();
    assert!((ds.norm.x_mean - 11.0).abs() < 1e-4);
    let (train, val) = ds.split_with_seed(0.5, 42);
    assert_eq!((train.n_samples, val.n_samples), (3, 3));
    for (part, expected) in [(&train, [10.0, 11.0, 12.0]), (&val, [20.0, 21.0, 22.0])] {
        let mut actual: Vec<f32> = part
            .inputs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .chunks_exact(5)
            .map(|row| row[0] * part.norm.x_std + part.norm.x_mean)
            .collect();
        actual.sort_by(f32::total_cmp);
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-3);
        }
    }

    let (train, val) = gnn_groups(&path);
    assert!((train.norm.x_mean - 11.0).abs() < 1e-4);
    assert_eq!((train.groups.len(), val.groups.len()), (1, 1));
    let coords = |ds: &GnnDataset| -> Vec<f32> {
        ds.groups
            .iter()
            .flat_map(|g| g.coords.iter().map(|c| c[0]))
            .collect()
    };
    assert_eq!(coords(&train), vec![10.0, 11.0, 12.0]);
    assert_eq!(coords(&val), vec![20.0, 21.0, 22.0]);

    let (train, val) = SirenDataset::generate(&path, 2, 10, 0.5, 42, None).unwrap();
    assert_eq!((train.n_stars, val.n_stars), (3, 3));
    assert_eq!((train.n_samples, val.n_samples), (12, 12));
    assert!((train.norm.bp_rp_mean - 1.2).abs() < 1e-4);
    for (part, expected) in [(&train, [1.0, 1.2, 1.4]), (&val, [2.0, 2.2, 2.4])] {
        let mut actual = siren_bp_rp_values(part, &device);
        actual.sort_by(f32::total_cmp);
        actual.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-3);
        }
    }
}

#[test]
fn no_split_keeps_legacy_seeded_fraction_and_norm() {
    let (_dir, path) = write_frame(frame());
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let ds = StellarDataset::<NdArray<f32>>::load(&path, &device, None, None).unwrap();
    assert!((ds.norm.x_mean - 609.8).abs() < 1e-3);
    let (train, val) = ds.split_with_seed(0.3, 42);
    assert_eq!((train.n_samples, val.n_samples), (7, 3));
    let (train, val) = gnn_groups(&path);
    assert_eq!((train.groups.len(), val.groups.len()), (2, 2));
    let (train, val) = SirenDataset::generate(&path, 2, 10, 0.4, 42, None).unwrap();
    assert_eq!((train.n_stars, val.n_stars), (6, 4));
    let train_values = siren_bp_rp_values(&train, &device);
    let train_mean = train_values.iter().sum::<f32>() / train_values.len() as f32;
    assert!((train.norm.bp_rp_mean - train_mean).abs() < 1e-4);
    let all_bp_rp = [1.0f32, 1.2, 1.4, 2.0, 2.2, 2.4, 6.0, 6.2, 8.0, 8.2];
    let full_mean = all_bp_rp.iter().sum::<f32>() / all_bp_rp.len() as f32;
    assert!((train.norm.bp_rp_mean - full_mean).abs() > 1e-3);
}

#[test]
fn unexpected_split_is_rejected_independently_by_each_loader() {
    let mut splits = canonical_splits();
    splits[0] = Some("holdout");
    let (_dir, path) = write_frame(with_split(&splits));
    let device = burn::backend::ndarray::NdArrayDevice::default();
    assert!(StellarDataset::<NdArray<f32>>::load(&path, &device, None, None).is_err());
    assert!(GnnDataset::load_with_seed(&path, 2, 8, 3.0, 42, None, None).is_err());
    assert!(SirenDataset::generate(&path, 2, 10, 0.5, 42, None).is_err());
}
