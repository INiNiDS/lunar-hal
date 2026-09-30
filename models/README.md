<div align="center">

# ✦ LUNAR-HAL NEURAL MODEL REGISTRY

### Deep Learning Architectures for Stellar Astrophysics, Galactic Kinematics & Procedural Radiance

[![Burn](https://img.shields.io/badge/Framework-Burn_0.21-f34b7d?style=for-the-badge&logo=rust&logoColor=white)](https://burn.dev)
[![Backend](https://img.shields.io/badge/Compute-WGPU%20%7C%20CUDA%20%7C%20CPU-blue?style=for-the-badge)](https://github.com/tracel-ai/burn)
[![Contract](https://img.shields.io/badge/Serving-Fail--Closed-green?style=for-the-badge)](#serving--runtime-contracts)

</div>

---

## 🌌 Overview

The **LUNAR-HAL** AI subsystem is an ensemble of specialized neural network architectures designed to process, model, and visualize real astronomical data from **ESA Gaia DR3** and the **NASA Exoplanet Archive**. 

All models run **locally** with strict offline evaluation, deterministic seeding, hash-verified weights, and fail-closed serving guarantees.

```
                          ┌───────────────────────┐
                          │     ESA Gaia DR3      │
                          │   Astrometry/Color    │
                          └──────────┬────────────┘
                                     │ [x, y, z, bp-rp, g_mag]
                                     ▼
                   ┌───────────────────────────────────┐
                   │  1. PINN Stellar Model (pinn-v1)  │
                   │   Stefan-Boltzmann Constrained    │
                   └─────────────────┬─────────────────┘
                                     │ [log_Teff, log_R, log_M, log_L]
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
┌─────────────────────────────────────┐ ┌──────────────────────────────────────┐
│ 2. GNN Kinematics (gnn-kinematics-v1)│ │       3. SIREN (siren-v1)            │
│    Stellar Peculiar Velocities      │ │   Continuous Neural Photosphere      │
│  [vx, vy, vz] + Uncertainty Heads   │ │    Procedural Irradiance [R, G, B]   │
└─────────────────────────────────────┘ └──────────────────────────────────────┘
```

---

## 1. Physics-Informed Neural Network (PINN)

### Architecture & Specification

| Property | Value |
| :--- | :--- |
| **Model ID** | `stellar_model.bpk` |
| **Architecture Version** | `pinn-v1` |
| **Weights Checksum (SHA-256)** | `7c19f8dfde41f8f85df4894a79b073f4266776ff6f3751ca9f5bab5d075acdca` |
| **Normalization Checksum** | `7f2a8b841a4df9b99af8af2e78a8e0c7f2e5f659161eb349b25b94c070eea30e` |
| **Input Features** | 5: `[x, y, z, bp_rp, g_mag]` |
| **Output Targets** | 4: `[log_teff, log_rad, log_mass, log_lum]` |
| **Physics Loss Weight** | $\lambda_{\text{phys}} = 0.15$ |

### Physical Principles

PINN directly enforces the **Stefan-Boltzmann Radiation Law** across all forward inferences:
$$L = 4\pi R^2 \sigma T_{\text{eff}}^4 \implies \log L - (2\log R + 4\log T_{\text{eff}}) = \text{const}$$

The network optimizes a dual objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}}(\hat{y}, y) + \lambda_{\text{phys}} \|\log \hat{L} - 2\log \hat{R} - 4\log \hat{T} - C\|^2$$

### Error Budgets & Quality Thresholds
- **$\log T_{\text{eff}}$ MAE:** $\le 0.035\text{ dex}$ ($\approx 8\%$ relative temperature error)
- **$\log R$ MAE:** $\le 0.045\text{ dex}$ ($\approx 10\%$ relative radius error)
- **$\log M$ MAE:** $\le 0.050\text{ dex}$ ($\approx 12\%$ relative stellar mass)
- **Stefan-Boltzmann Residual:** $\le 0.015\text{ dex}$ across spatial holdouts

---

## 2. Graph Neural Network for Kinematics (GNN)

### Architecture & Specification

| Property | Value |
| :--- | :--- |
| **Model ID** | `stellar_gnn_model.bpk` |
| **Architecture Version** | `gnn-kinematics-v1` |
| **Weights Checksum (SHA-256)** | `e46c8caecfe95ff78c50bb643810e7ee3102ce2ac9dc65e91081a44a5816d922` |
| **Normalization Checksum** | `4b5effc3035e59c7d1ae72b40588e75a1ba0da58b35f7df9597decd38d123c59` |
| **Input Node Features** | 8: `[log_teff, log_rad, log_mass, log_lum, mg, x, y, z]` (derived from PINN) |
| **Output Targets** | 3 (Deterministic): `[vx, vy, vz]` or 6 (Variational): `[vx, vy, vz, log_var_x, log_var_y, log_var_z]` |
| **Graph Topology** | Sparse $k$-NN Adjacency Matrix ($k=16$, distance-weighted $w_{ij} = 1 / d_{ij}^2$) |

### Dynamics & Message Passing

GNN models collective gravitational drift and galactic rotation curves. Message passing aggregates localized velocity perturbations over stellar neighbors:
$$h_i^{(l+1)} = \sigma \left( W^{(l)} h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} W_{\text{edge}}^{(l)} h_j^{(l)} \right)$$

### Quality Thresholds
- **Velocity MAE ($v_x, v_y, v_z$):** $\le 6.5\text{ km/s}$
- **Uncertainty Calibration (NLL):** $\le 1.85$
- **Permutation Invariance:** $100\%$ bitwise identical across node permutations

---

## 3. Sinusoidal Representation Networks (SIREN)

### Architecture & Specification

| Property | Value |
| :--- | :--- |
| **Model ID** | `stellar_siren_model.bpk` |
| **Architecture Version** | `siren-v1` |
| **Weights Checksum (SHA-256)** | `df7d3d503851238f3e8a99cf388581a29f042f8f20191bcda7ae2793b93c7fce` |
| **Normalization Checksum** | `c700765c1f7d17718b262bf578022d522b91064b892b22287804567877399093` |
| **Input Coordinates & Physical Traits** | 5: `[u, v, bp_rp, m_g, log_teff]` |
| **Output Irradiance** | 3: `[R, G, B]` (Normalized radiant surface intensity) |
| **Periodic Activation** | $\sin(\omega_0(Wx + b))$ with $\omega_0 = 30.0$ |

### Infinite-Resolution Procedural Photospheres

SIREN represents stellar photospheres implicitly as continuous coordinate functions. Rather than storing discrete pixel textures, SIREN models convective plasma cells, limb darkening, and solar flare filaments at arbitrary zoom levels without interpolation artifacts.

### Quality Thresholds
- **RGB Mean Squared Error (MSE):** $\le 0.008$
- **Streaming Parity:** Chunked streaming vs. batch forward parity $\le 10^{-6}$ max absolute error
- **Synthesis Latency:** $\le 45\text{ ms}$ for $256\times256$ texture on CPU

---

## 4. GNN Localization & Astrometric Infilling (Experimental)

### Architecture & Specification

| Property | Value |
| :--- | :--- |
| **Architecture Version** | `gnn-localization-v1` |
| **Stage** | Experimental / Stage 14 (Gated behind feature flag) |
| **Task** | Reconstruct 3D positions of masked/occluded stars from cluster geometry |
| **Target Chamfer Distance** | $\le 2.2\text{ pc}$ |
| **Target EMD** | $\le 3.5\text{ pc}$ |

---

## 🔒 Serving & Runtime Contracts

1. **Fail-Closed Bootstrapping:**
   Every model loaded into memory must match its SHA-256 weight checksum, SHA-256 normalization checksum, and declared `feature_schema_hash`. If any check fails, the backend refuses to serve the model and retains healthy weights.

2. **Hot Reloading:**
   The backend exposes `POST /models/reload` and `GET /version` for runtime inspection and zero-downtime hot reload with atomic lock transitions.

3. **Multi-Backend Acceleration:**
   Powered by the [Burn](https://burn.dev) framework, models seamlessly run across:
   - **WGPU:** Cross-platform hardware acceleration (Vulkan, Metal, DirectX 12)
   - **LibTorch / CUDA:** High-throughput GPU training and server inference
   - **NdArray:** Pure Rust CPU inference with SIMD auto-vectorization
