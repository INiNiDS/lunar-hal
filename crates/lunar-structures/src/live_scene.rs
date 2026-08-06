//! Backend-owned live scene DTOs.
//!
//! Mutations are expressed as backend events so an embedded frontend can
//! update from its own snapshot/event stream without accepting parent commands.

use crate::{GallerySource, ResponseStar, StarModelInputs, StarScene};
use serde::{Deserialize, Serialize};

pub const LIVE_SCENE_SCHEMA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LiveSceneSnapshot {
    #[serde(default = "live_scene_schema_version")]
    pub schema_version: u32,
    pub scene: StarScene,
}

impl From<StarScene> for LiveSceneSnapshot {
    fn from(scene: StarScene) -> Self {
        Self {
            schema_version: LIVE_SCENE_SCHEMA_VERSION,
            scene,
        }
    }
}

fn live_scene_schema_version() -> u32 {
    LIVE_SCENE_SCHEMA_VERSION
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SceneEvent {
    StarAdded {
        scene_id: String,
        star: ResponseStar,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gallery_id: Option<String>,
    },
    StarUpdated {
        scene_id: String,
        star: ResponseStar,
    },
    StarRemoved {
        scene_id: String,
        star_id: u32,
    },
    SceneCleared {
        scene_id: String,
    },
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GenerateSceneStarsRequest {
    pub request_id: String,
    #[serde(default = "one")]
    pub count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entropy_temperature: Option<f32>,
}

fn one() -> u32 {
    1
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CreateSceneStarRequest {
    pub request_id: String,
    /// A direct custom star, used only when `gallery_id` is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub star: Option<ResponseStar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<StarModelInputs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gallery_id: Option<String>,
    #[serde(default)]
    pub gallery_source: GallerySource,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct UpdateSceneStarRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub type_hint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_k: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mass: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub luminosity: Option<f32>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ClearSceneRequest {
    pub request_id: String,
}

/// Optional, strictly UI-only event an embedded frontend may report to WebOS.
/// No mutation variant is intentionally present here.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FrontendUiEvent {
    Selection { scene_id: String, star_id: Option<u32> },
    DragStarted { scene_id: String, star_id: u32 },
    FocusRequested { scene_id: String },
}
