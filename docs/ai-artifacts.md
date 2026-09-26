# AI Model Artifact Contracts & Registry Specification

## Overview
This document describes artifact integrity, release-evidence requirements, registry discovery, and current reload semantics in the Lunar-HAL workspace.

Serving is fail-closed: a model requires a compatible sidecar, matching weight/normalization checksums, an accepted feature-schema identifier, complete provenance, and a spatial-holdout evaluation bound to the same artifact. A passing unit or synthetic test is not model-quality evidence.

---

## 1. Top-Level Models Manifest (`models/manifest.json`)
`models/manifest.json` is a human-readable inventory. The backend currently discovers per-model `*.artifact.json` sidecars and does not use this top-level file as a serving gate.

```json
{
  "version": "1.0.0",
  "created_ms": 1789729507630,
  "git_revision": "754091aa7b0bf52fd06a63385bbbd2de2244714a",
  "models": [
    {
      "kind": "pinn",
      "architecture_version": "pinn-v1",
      "weights": "stellar_model.bpk",
      "norm": "stellar_norm.json",
      "manifest": "stellar_model.artifact.json",
      "model_hash": "7c19f8dfde41f8f85df4894a79b073f4266776ff6f3751ca9f5bab5d075acdca",
      "norm_hash": "7f2a8b841a4df9b99af8af2e78a8e0c7f2e5f659161eb349b25b94c070eea30e",
      "feature_schema_hash": "pinn-input-x,y,z,bp_rp,g_mag-v1"
    },
    {
      "kind": "gnn_kinematics",
      "architecture_version": "gnn-kinematics-v1",
      "weights": "stellar_gnn_model.bpk",
      "norm": "stellar_gnn_norm.json",
      "manifest": "stellar_gnn_model.artifact.json",
      "model_hash": "e46c8caecfe95ff78c50bb643810e7ee3102ce2ac9dc65e91081a44a5816d922",
      "norm_hash": "4b5effc3035e59c7d1ae72b40588e75a1ba0da58b35f7df9597decd38d123c59",
      "feature_schema_hash": "gnn-serving-pinn-derived-log_teff,log_rad,log_mass,log_lum,mg,x,y,z-v1"
    },
    {
      "kind": "siren",
      "architecture_version": "siren-v1",
      "weights": "stellar_siren_model.bpk",
      "norm": "stellar_siren_norm.json",
      "manifest": "stellar_siren_model.artifact.json",
      "model_hash": "df7d3d503851238f3e8a99cf388581a29f042f8f20191bcda7ae2793b93c7fce",
      "norm_hash": "c700765c1f7d17718b262bf578022d522b91064b892b22287804567877399093",
      "feature_schema_hash": "siren-serving-uv,bp_rp,m_g,log_teff-v1"
    }
  ]
}
```

---

## 2. Artifact Schema (`ArtifactManifestV1`)
Each individual model artifact carries an accompanying `*.artifact.json` manifest conforming to the Rust struct `ArtifactManifestV1` in `ai/lnai-training/src/artifacts.rs`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `version` | `String` | Manifest specification version (currently `"1.0.0"`). |
| `model_kind` | `String` | Type of model: `"pinn"`, `"gnn_kinematics"`, `"gnn_localization"`, `"siren"`. |
| `architecture_version` | `String` | Semantic model architecture version (e.g. `"pinn-v1"`, `"gnn-kinematics-v1"`). |
| `model_hash` | `String` | Hex-encoded SHA-256 hash of the binary weights file (`.bpk` or `.safetensors`). |
| `norm_hash` | `String` | Hex-encoded SHA-256 hash of the normalization JSON file. |
| `feature_schema_hash` | `String` | Canonical feature schema identifier representing input feature column ordering. |
| `dataset_id` | `String` | Name of training dataset (e.g. `"gaia_dr3"`). |
| `dataset_version` | `String` | SHA-256 fingerprint of the exact input file plus sibling canonical/view manifests; manually edited parquet bytes are still fingerprinted independently. |
| `seed` | `u64` | Random seed used during training run. |
| `hyperparameters` | `Object` | Arbitrary JSON record of model hyperparameters (dimensions, learning rates, physics weights). |
| `git_revision` | `String` | Compile-time/runtime Git revision; tracked local modifications are marked `-dirty`, and `unknown` blocks release. |
| `backend_device` | `String` | Burn backend target (e.g. `"cuda:0"`, `"cpu"`, `"serving"`). |
| `evaluation_metrics` | `Object?` | Optional validation metrics evaluated on holdout splits. |
| `created_ms` | `u64` | Millisecond Unix timestamp of creation. |
| `localization_meta` | `Object?` | Optional specialized configuration for GNN localization (e.g., k-NN radius, coordinate frame). |

The registry marks a hash-valid bundle `release_blocked` until `feature_schema_hash` matches the serving feature order, `dataset_id`, `dataset_version`, and a known `git_revision` are present, and `evaluation_metrics.spatial_holdout` records `passed: true`, matching artifact identities, an in-bundle `report_file`, and a 64-character `report_sha256`. Serving re-hashes the report, rejects paths outside the bundle, and checks the report's `gate_version`, non-empty `metrics`, passing status, and model/norm/dataset/schema bindings. The report still requires independent review; this file-integrity check cannot prove the evaluator or thresholds were honest.

---

## 3. Architecture Versioning & Schema Hashes
To protect against silent feature permutation or dimensional drift, serving requires the following exact feature identifiers (the `feature_schema_hash` field is currently a stable identifier, not necessarily a cryptographic hash):

1. **PINN (Physics-Informed Neural Network)**
   - Architecture: `pinn-v1`
   - Inputs: $[x, y, z, bp\_rp, g\_mag]$
   - Schema Hash: `pinn-input-x,y,z,bp_rp,g_mag-v1`
   - Targets: $[\log T_{eff}, \log R, \log M, \log L]$ constrained by Stefan-Boltzmann relation:
     $$L = 4\pi R^2 \sigma T^4 \implies \log L - (2\log R + 4\log T) = \text{const}$$

2. **GNN-Kinematics (Group Stellar Kinematics)**
   - Architecture: `gnn-kinematics-v1`
   - Inputs: Node features derived from PINN $[\log T_{eff}, \log R, \log M, \log L, M_G, x, y, z]$
   - Schema Hash: `gnn-serving-pinn-derived-log_teff,log_rad,log_mass,log_lum,mg,x,y,z-v1`
   - Targets: 3D Peculiar velocities $[v_x, v_y, v_z]$ and calibrated uncertainty heads $[\sigma_{vx}, \sigma_{vy}, \sigma_{vz}]$.

3. **SIREN (Sinusoidal Representation Network)**
   - Architecture: `siren-v1`
   - Inputs: $[u, v, bp\_rp, M_G, \log T_{eff}]$ with sinusoidal activation $\sin(\omega_0(W x + b))$.
   - Schema Hash: `siren-serving-uv,bp_rp,m_g,log_teff-v1`
   - Outputs: Normalized RGB star surface irradiance $[R, G, B]$.

4. **GNN-Localization (Missing-Neighbor Reconstruction - Experimental)**
   - Architecture: `gnn-localization-v1`
   - No production serving schema or trained/release-approved artifact is currently defined; it remains blocked and disabled by default.

---

## 4. Normalization Snapshot Requirements
Normalization snapshots (`*.norm.json`) must define scalar affine transformation parameters for all input and target features:
- Mean and standard deviation ($\mu, \sigma$) for z-score standardization, or min/max bounds ($min, max$) for min-max scaling.
- Epsilon guarding against zero-division ($\epsilon = 10^{-7}$).
- All normalization files must match their recorded `norm_hash` SHA-256 checksum upon server boot or dynamic reload. If checksum verification fails, the runtime aborts loading and retains previously healthy weights.

---

## 5. Registry Reload & Version API Semantics
`lunar-backend` exposes HTTP endpoints for runtime model inspection and controlled reload:

### `GET /version`
Returns registry entries (sidecar hashes, release status, and blockers) separately from actually loaded models. `active_models` contains only loaded kinds and their in-memory weight/normalization hashes; it may be empty:
```json
{
  "service": "lunar-backend",
  "git_revision": "<server-revision>",
  "models_dir": "<models-directory>",
  "models": [{
    "kind": "pinn",
    "status": "release_blocked",
    "model_hash": "<manifest-sha256>",
    "norm_hash": "<manifest-sha256>",
    "release_blockers": ["dataset_version", "spatial_holdout_evaluation"]
  }],
  "active_models": [{
    "kind": "pinn",
    "model_hash": "<loaded-weight-sha256>",
    "norm_hash": "<loaded-norm-sha256>"
  }]
}
```

### `POST /models/reload`
Scans the flat serving sidecars, refuses the whole reload if a flat bundle is invalid or release-blocked, and validates `stellar_model.bpk` before replacing the active PINN. Optional GNN/SIREN/localization caches are invalidated and loaded lazily on the next request; this is not an atomic multi-model generation swap, and no dummy forward pass is run during reload. In-flight requests holding an existing `Arc` can finish with that model.

---

## 6. Atomic Publish Protocol
`atomic_write_through` (`ai/lnai-training/src/artifacts.rs`) writes to a sibling temporary path and renames each file individually. The weight, normalization, and manifest updates are not one multi-file transaction and the helper does not currently promise an `fsync` durability barrier. Readers must verify all declared checksums and fail closed during an incomplete publish; versioned-directory promotion would provide a stronger all-files-at-once deployment contract.
