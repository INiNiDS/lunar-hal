use lunar_structures::{
    CreateWorldRequest, GnnResponse, PinnRequest, PinnResponse, PipelineRequest,
    PipelineResponse, RandomStarRequest, RandomStarResponse, SectorRequest, World, WorldListResponse,
};
use std::sync::LazyLock;

use lunar_utils::env::get_url;

static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(reqwest::Client::new);
const BASE_URL: LazyLock<String> = LazyLock::new(get_url);

pub async fn fetch_pinn(x_pc: f32, y_pc: f32, z_pc: f32, bp_rp: f32, g_mag: f32) -> Result<PinnResponse, reqwest::Error> {
    let req = PinnRequest { x_pc, y_pc, z_pc, bp_rp, g_mag };

    HTTP_CLIENT
        .post(&format!("{}/pinn", *BASE_URL))
        .json(&req)
        .send()
        .await?
        .json::<PinnResponse>()
        .await
}

pub async fn fetch_pipeline(req: PipelineRequest) -> Result<PipelineResponse, reqwest::Error> {
    HTTP_CLIENT
        .post(&format!("{}/pipeline", *BASE_URL))
        .json(&req)
        .send()
        .await?
        .json::<PipelineResponse>()
        .await
}

pub async fn list_worlds() -> Result<WorldListResponse, reqwest::Error> {
    HTTP_CLIENT
        .get(&format!("{}/worlds", *BASE_URL))
        .send()
        .await?
        .json::<WorldListResponse>()
        .await
}

pub async fn get_world(id: &str) -> Result<World, reqwest::Error> {
    HTTP_CLIENT
        .get(&format!("{}/worlds/{}", *BASE_URL, id))
        .send()
        .await?
        .json::<World>()
        .await
}

pub async fn create_world(req: CreateWorldRequest) -> Result<World, reqwest::Error> {
    HTTP_CLIENT
        .post(&format!("{}/worlds/create", *BASE_URL))
        .json(&req)
        .send()
        .await?
        .json::<World>()
        .await
}

pub async fn delete_world(id: &str) -> Result<(), reqwest::Error> {
    HTTP_CLIENT
        .delete(&format!("{}/worlds/{}", *BASE_URL, id))
        .send()
        .await?;
    Ok(())
}

pub async fn fetch_sector_stars(
    sector_cx: f32,
    sector_cy: f32,
    sector_cz: f32,
    temperature: f32,
    bp_rp: f32,
    g_mag: f32,
) -> Result<GnnResponse, reqwest::Error> {
    let req = SectorRequest {
        sector_cx,
        sector_cy,
        sector_cz,
        temperature,
        bp_rp,
        g_mag,
        search_radius: Some(200.0),
    };
    HTTP_CLIENT
        .post(&format!("{}/sector/stars", *BASE_URL))
        .json(&req)
        .send()
        .await?
        .json::<GnnResponse>()
        .await
}

pub async fn fetch_random_star(entropy_temperature: f32) -> Result<RandomStarResponse, reqwest::Error> {
    let req = RandomStarRequest { entropy_temperature };
    HTTP_CLIENT
        .post(&format!("{}/random_star", *BASE_URL))
        .json(&req)
        .send()
        .await?
        .json::<RandomStarResponse>()
        .await
}
