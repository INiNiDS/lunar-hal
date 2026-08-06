//! Durable star records used by the SIREN Gallery.
//!
//! The Gallery deliberately stores metadata separately from binary texture
//! assets. `GalleryStore` owns the on-disk layout; these types are the stable
//! API contract shared by the backend, frontend, and WebOS.

use crate::{PinnResponse, ResponseStar, StellarMetadata};
use serde::{Deserialize, Serialize};

pub const GALLERY_SCHEMA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GallerySource {
    #[default]
    AdminGenerated,
    SceneSaved,
    Imported,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StarModelInputs {
    pub x_pc: f32,
    pub y_pc: f32,
    pub z_pc: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entropy_temperature: Option<f32>,
}

impl StarModelInputs {
    pub fn from_star(star: &ResponseStar) -> Self {
        Self {
            x_pc: star.x,
            y_pc: star.y,
            z_pc: star.z,
            bp_rp: 1.0,
            g_mag: 10.0,
            entropy_temperature: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GalleryStar {
    pub id: String,
    #[serde(default = "gallery_schema_version")]
    pub schema_version: u32,
    pub created_at: u64,
    pub updated_at: u64,
    pub source: GallerySource,
    /// Full technical star record. Pixels are intentionally not embedded here.
    pub star: ResponseStar,
    pub inputs: StarModelInputs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinn: Option<PinnResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<StellarMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thumbnail_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// Persisted only to make repeated POST/drop requests idempotent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

fn gallery_schema_version() -> u32 {
    GALLERY_SCHEMA_VERSION
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CreateGalleryStarRequest {
    pub request_id: String,
    #[serde(default)]
    pub source: GallerySource,
    pub star: ResponseStar,
    pub inputs: StarModelInputs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinn: Option<PinnResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<StellarMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct UpdateGalleryStarRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GalleryListResponse {
    pub stars: Vec<GalleryStar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Transport payload shared by Gallery cards and a live scene. It contains an
/// identifier and a serializable star snapshot, never a mutation command.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StarDragPayload {
    pub source: StarDragSource,
    pub scene_id: Option<String>,
    pub star_id: u32,
    pub gallery_id: Option<String>,
    pub star: ResponseStar,
    pub texture_url: Option<String>,
    pub request_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StarDragSource {
    Scene,
    Gallery,
}
