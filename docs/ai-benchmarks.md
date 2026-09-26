# AI Model Benchmarking & Parity Verification Methodology

## Overview
This document specifies the validation methodology, regression error budgets, profiling runners, and quality release gates for AI neural networks in Lunar-HAL (PINN, GNN-Kinematics, GNN-Localization, and SIREN).

The checked-in `stellar-e2e-v1` fixture is synthetic and is suitable for deterministic contracts and leakage plumbing only. Passing its suites does not establish the quality of any trained serving checkpoint; release qualification requires a separate spatial-holdout evaluation bound to the exact artifact hashes.

---

## 1. Frozen Fixture Contract (`stellar-e2e-v1`)
The frozen fixture resides under `ai/fixtures/stellar-e2e-v1/` and contains:
- **`fixture.parquet`:** 256 deterministic synthetic rows; it is not a real Gaia DR3 sample.
- **`manifest.json`:** Fixture checksum, schema identifier, seed, source IDs, and broad RA tile metadata. It does not define an independent real-data spatial holdout.
- **Stage baselines:** `ai/fixtures/stage6-approved-baseline.json` and `stage7-approved-baseline.json` record contract/parity test status. They are explicitly not model-quality or production-release approvals.

---

## 2. Profiling Runners & Reproduction Commands

### CPU Verification Suites
```bash
# Unit & integration test suites
cargo test -p lnai-data
cargo test -p lnai-training

# Specialized contract/surrogate runners (not trained-checkpoint accuracy)
cargo test -p lnai-training --test gnn_oracle
cargo test -p lnai-training --test pinn_accuracy
cargo test -p lnai-training --test pinn_gnn_chain
cargo test -p lnai-training --test localization_masked
cargo test -p lnai-training --test spatial_holdout_tiles
cargo test -p lnai-training --test fixture_leakage
```

### Full E2E & Benchmark Shell Scripts
```bash
# Synthetic fixture, contract and leakage smoke checks
./scripts/e2e-ai.sh

# Microbenchmarks; supported options are --runs N and --gpu
./scripts/bench-ai.sh --runs 3
```

`scripts/bench-ai.sh` does not accept `--fixture`; its forward/profile tests use seeded or untrained model instances. These commands do not measure trained-checkpoint accuracy.

---

## 3. Error Metrics & Regression Budgets

The values below are target budgets, not verified results for the current weights. No checked-in suite currently evaluates these metrics on the real trained serving checkpoints and an independent spatial holdout.

### A. PINN (Physics-Informed Neural Network)
PINN maps stellar astrometry and photometry $[x, y, z, bp\_rp, g\_mag]$ to fundamental physical parameters $[\log T_{eff}, \log R, \log M, \log L]$.

| Target Metric | Formula | Maximum Permissible Error (Budget) |
| :--- | :--- | :--- |
| **$\log T_{eff}$ MAE** | $\frac{1}{N} \sum \|\hat{y} - y\|$ | $\le 0.035$ dex ($\approx 8\%$ relative $T_{eff}$) |
| **$\log R$ MAE** | $\frac{1}{N} \sum \|\hat{y} - y\|$ | $\le 0.045$ dex ($\approx 10\%$ relative radius) |
| **$\log M$ MAE** | $\frac{1}{N} \sum \|\hat{y} - y\|$ | $\le 0.050$ dex ($\approx 12\%$ relative mass) |
| **$\log L$ MAE** | $\frac{1}{N} \sum \|\hat{y} - y\|$ | $\le 0.060$ dex |
| **Stefan-Boltzmann Residual** | $\frac{1}{N} \sum \|\log L - 2\log R - 4\log T - C\|$ | $\le 0.015$ dex (Physical constraint fidelity) |

### B. GNN-Kinematics
GNN-Kinematics models local stellar group velocities from spatial coordinates and physical properties.

| Target Metric | Formula | Maximum Permissible Error (Budget) |
| :--- | :--- | :--- |
| **Velocity MAE ($v_x, v_y, v_z$)** | $\frac{1}{3N} \sum_{i=1}^N \sum_{d \in \{x,y,z\}} \|\hat{v}_{i,d} - v_{i,d}\|$ | $\le 6.5 \text{ km/s}$ |
| **Uncertainty Calibration (NLL)** | Negative log-likelihood under predicted $\sigma^2$ | $\le 1.85$ |
| **Graph Isolation Parity** | Permutation invariance under node indexing | $100\%$ bitwise identical |

### C. GNN-Localization (Experimental Feature)
Evaluates reconstructed coordinates of masked stars within k-NN clusters.

| Target Metric | Description | Quality Threshold |
| :--- | :--- | :--- |
| **Chamfer Distance** | Bi-directional nearest neighbor distance | $\le 2.2 \text{ pc}$ |
| **Earth Mover's Distance (EMD)** | Optimal transport distance between distributions | $\le 3.5 \text{ pc}$ |
| **Data Leakage Audit** | Hidden node edge exclusion test | Zero masked node leakage into input subgraph |

*Note: GNN-Localization is experimental in Stage 14 and gated behind a feature flag.*

### D. SIREN (Sinusoidal Star Surface Textures)
| Target Metric | Description | Quality Threshold |
| :--- | :--- | :--- |
| **RGB MSE** | Mean squared reconstruction error | $\le 0.008$ |
| **Chunked Forward Parity** | Output equivalence across batch vs streaming inference | Max absolute diff $\le 10^{-6}$ |

### E. Chained Inference Pipeline (PINN $\to$ GNN $\to$ SIREN)
- **10-Step Rollout Drift:** Maximum accumulated position/velocity divergence $\le 3.8\%$.
- **Inference Latency (CPU):**
  - PINN point inference: $\le 1.2 \text{ ms}$
  - GNN 32-star group inference: $\le 8.5 \text{ ms}$
  - SIREN $256\times256$ texture synthesis: $\le 45 \text{ ms}$

---

## 4. Release Decision Gates (Go / No-Go)
Every release candidate is classified under three verdict criteria:

- **GREEN (Production Ready):**
  - All workspace tests (`cargo test --workspace`) pass without failures.
  - A real-data spatial-holdout report for each trained checkpoint meets 100% of the defined regression budgets and is bound to the exact weight, normalization, dataset, and schema identities.
  - Reproducible latency/memory reports include hardware, data, code revision, and raw results.
  - 50-cycle RAM snapshot stress test passes with zero memory/generation leaks.
  - Every served registry artifact is release-approved and `/version` reports the exact hashes of active weights.

- **YELLOW (Conditional Staging):**
  - Foundation, PINN, GNN-Kinematics, SIREN, and Lunar-OS RAM pass 100% Green.
  - Experimental GNN-Localization passes internal sanity but remains feature-flag gated (`false` by default).

- **RED (Blocked / No-Go):**
  - Any compilation warning or test regression in core pipelines.
  - Data leakage between training and holdout partitions.
  - Missing, failed, or artifact-unbound real spatial-holdout evaluation.
  - CRC32 integrity failure or generation collision during RAM snapshot transactions.
