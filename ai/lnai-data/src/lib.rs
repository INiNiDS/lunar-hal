//! `lnai-data` — canonical stellar dataset collection, cleaning, assembly,
//! splitting and integrity verification.
//!
//! Contract overview:
//! * [`schema`] — frozen golden schema v1 (`DATASET_SCHEMA_VERSION`).
//! * [`manifest`] — dataset/shard state machine with additive 1.1 fields.
//! * [`collector`] — bounded-concurrency shard downloads with retry/backoff,
//!   atomic writes, checksums, resume, adaptive subdivision.
//! * [`clean`] — deterministic dedup/coordinate-velocity conversion.
//! * [`assemble`] — canonical parquet + model views (PINN/GNN/SIREN).
//! * [`split`] — deterministic hash split + spatial-tile holdout.
//! * [`integrity`] — hashing/row-count helpers.
//! * stage 4A additions: [`sources`], [`auth`], [`provenance`], [`crossmatch`],
//!   [`source_manifest`], [`enrich`]; new-object-store sink: [`s3`]/[`storage`].
//! * stage 4B: [`stellar_params`] — Gaia astrophysical_parameters enrichment.

pub mod assemble;
pub mod auth;
pub mod clean;
pub mod collector;
pub mod crossmatch;
pub mod enrich;
pub mod integrity;
pub mod manifest;
pub mod provenance;
pub mod s3;
pub mod schema;
pub mod source_manifest;
pub mod sources;
pub mod split;
pub mod stellar_params;
pub mod storage;
pub mod tap;
