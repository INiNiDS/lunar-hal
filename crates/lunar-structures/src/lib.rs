use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct PinnRequest {
    pub x_pc: f32,
    pub y_pc: f32,
    pub z_pc: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct PinnResponse {
    pub temperature_k: f32,
    pub radius_solar: f32,
    pub mass_solar: f32,
    pub luminosity_solar: f32,
}


#[derive(Deserialize, Serialize, Debug)]
pub struct GnnRequest {
    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
    pub search_radius: f32,
    #[serde(default)]
    pub temperature: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ResponseStar {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub temperature_k: f32,
    pub radius: f32,
    pub mass: f32,
    pub luminosity: f32,
    pub description: String,
    pub name: String,
    pub type_hint: String,
    pub velocity_vector: [f32; 3],
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GnnResponse {
    pub stars: Vec<ResponseStar>,
}

#[derive(Deserialize, Serialize)]
pub struct ImageRequest {
    pub temperature_k: f32,
    pub radius_solar: f32,
    pub luminosity_solar: f32,
}

#[derive(Serialize, Deserialize)]
pub struct ImageResponse {
    pub shader_seed: u32,
    pub latent_surface_parameters: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StellarMetadata {
    pub spectral_class: String,
    pub category: String,
    pub designated_name: String,
    pub description: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LoreMetadata {
    pub simulation_engine: String,
    pub data_source: String,
    pub complexity_level: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StarLore {
    pub designated_name: String,
    pub category: String,
    pub visual_profile: String,
    pub system_lore: String,
    pub metadata: LoreMetadata,
}


#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SirenTextureRequest {
    pub width: u32,
    pub height: u32,
    pub bp_rp: f32,
    pub m_g: f32,
    pub log_teff: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SirenTextureResponse {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PipelineRequest {
    pub x_pc: f32,
    pub y_pc: f32,
    pub z_pc: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
    pub texture_size: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PipelineResponse {
    pub pinn: PinnResponse,
    pub siren: SirenTextureResponse,
    pub metadata: StellarMetadata,
}

#[derive(Deserialize, Serialize)]
pub struct RandomStarRequest {
    pub entropy_temperature: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RandomStarResponse {
    pub bp_rp: f32,
    pub g_mag: f32,
    pub x_pc: f32,
    pub y_pc: f32,
    pub z_pc: f32,
    pub star: ResponseStar,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StarDescriptionPayload {
    pub pinn_payload: PinnResponse,
    pub gnn_payload: GnnResponse,
}


