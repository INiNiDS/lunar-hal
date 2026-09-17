//! Local-weights liveness probe (ignored by default).
//!
//! These tests load the real `.bpk` weights from `models/` and run one
//! CPU forward pass per model, asserting finite outputs. Weight files are
//! gitignored, so the suite only runs where weights exist:
//! `cargo test -p lnai-training --test model_liveness -- --ignored`.

use burn::backend::NdArray;
use burn::prelude::*;
use burn_store::{BurnpackStore, ModuleSnapshot};

type B = NdArray<f32>;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn weight_file(name: &str) -> std::path::PathBuf {
    repo_root().join("models").join(name)
}

fn require_weights() -> bool {
    for f in [
        "stellar_model.bpk",
        "stellar_gnn_model.bpk",
        "stellar_siren_model.bpk",
    ] {
        if !weight_file(f).exists() {
            eprintln!("skipped: local weights not present ({f} missing)");
            return false;
        }
    }
    true
}

fn finite_check(name: &str, vals: &[f32]) {
    let bad = vals.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(bad, 0, "{name} has non-finite outputs: {vals:?}");
}

#[test]
#[ignore = "needs local gitignored weights; run with -- --ignored"]
fn serving_bundles_verify_against_manifests() {
    use lnai_training::artifacts::{RegistryStatus, discover_bundle};
    if !weight_file("stellar_model.bpk").exists() {
        eprintln!("skipped: local weights not present");
        return;
    }
    for dir in ["models/gnn-v1", "models/pinn-trend", "models/pinn-v1"] {
        let entry = discover_bundle(&repo_root().join(dir)).expect("bundle must verify");
        assert!(
            matches!(entry.status, RegistryStatus::Verified),
            "{dir} must verify"
        );
        println!("{dir}: VERIFIED");
    }
}

#[test]
#[ignore = "needs local gitignored weights; run with -- --ignored"]
fn serving_weights_load_and_forward_finite() {
    if !require_weights() {
        return;
    }
    let device = burn::backend::ndarray::NdArrayDevice::default();

    // PINN.
    let mut pinn = lnai_models::StellarMlpConfig::new().init::<B>(&device);
    let mut store = BurnpackStore::from_file(weight_file("stellar_model.bpk").to_str().unwrap());
    pinn.load_from(&mut store).expect("pinn weights must load");
    let inp: Tensor<B, 2> = Tensor::from_floats(
        [[0.1, -0.2, 0.3, 0.5, 4.0], [0.0, 0.0, 0.0, -0.5, 6.0]],
        &device,
    );
    let out: Vec<f32> = pinn.forward(inp).into_data().to_vec().unwrap();
    assert_eq!(out.len(), 8);
    finite_check("pinn", &out);

    // GNN (either head, resolved by load).
    let adj: Tensor<B, 2> = Tensor::from_floats([[1.0, 0.5], [0.5, 1.0]], &device);
    let nodes: Tensor<B, 2> = Tensor::from_floats(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3],
            [0.0, 0.1, -0.1, 0.2, 0.4, -0.2, 0.1, 0.0],
        ],
        &device,
    );
    let mut loaded = false;
    for width in [3usize, 6usize] {
        let mut gnn = lnai_models::StellarGnnConfig::new(8, 256, width).init::<B>(&device);
        let mut store =
            BurnpackStore::from_file(weight_file("stellar_gnn_model.bpk").to_str().unwrap());
        if gnn.load_from(&mut store).is_ok() {
            let out: Vec<f32> = gnn
                .forward(nodes.clone(), adj.clone())
                .into_data()
                .to_vec()
                .unwrap();
            assert_eq!(out.len(), 2 * width);
            finite_check("gnn", &out);
            println!("gnn head width: {width}");
            loaded = true;
            break;
        }
    }
    assert!(loaded, "gnn weights must load as det or var");

    // SIREN.
    let mut siren = lnai_models::StellarSirenConfig::new().init::<B>(&device);
    let mut store =
        BurnpackStore::from_file(weight_file("stellar_siren_model.bpk").to_str().unwrap());
    siren
        .load_from(&mut store)
        .expect("siren weights must load");
    let inp: Tensor<B, 2> = Tensor::from_floats([[0.0, 0.0, 0.5, 0.2, 3.75]], &device);
    let out: Vec<f32> = siren.forward(inp).into_data().to_vec().unwrap();
    assert_eq!(out.len(), 3);
    finite_check("siren", &out);
}
