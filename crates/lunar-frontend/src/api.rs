use lunar_structures::{GnnRequest, GnnResponse, PinnRequest, PinnResponse, PipelineRequest, PipelineResponse, RandomStarRequest, RandomStarResponse, StarDescriptionPayload, StarLore};
use std::sync::LazyLock;

static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(reqwest::Client::new);
const BASE_URL: &str = "http://localhost:25255";

pub async fn fetch_pinn(x_pc: f32, y_pc: f32, z_pc: f32, bp_rp: f32, g_mag: f32) -> Result<PinnResponse, reqwest::Error> {
    let req = PinnRequest { x_pc, y_pc, z_pc, bp_rp, g_mag };

    HTTP_CLIENT
        .post(&format!("{BASE_URL}/pinn"))
        .json(&req)
        .send()
        .await?
        .json::<PinnResponse>()
        .await
}

pub async fn fetch_gnn_by_temp(
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
    temperature: f32,
    bp_rp: f32,
    g_mag: f32,
) -> Result<GnnResponse, reqwest::Error> {
    let req = GnnRequest {
        center_x: cx,
        center_y: cy,
        center_z: cz,
        bp_rp,
        g_mag,
        search_radius: radius,
        temperature,
    };

    HTTP_CLIENT
        .post(&format!("{BASE_URL}/gnn"))
        .json(&req)
        .send()
        .await?
        .json::<GnnResponse>()
        .await
}

pub async fn fetch_star_description(gnn_response: GnnResponse, pinn_response: PinnResponse) -> Result<StarLore, reqwest::Error> {
    let payload = StarDescriptionPayload {
        pinn_payload: pinn_response,
        gnn_payload: gnn_response,
    };
    HTTP_CLIENT
        .post(&format!("{BASE_URL}/description"))
        .json(&payload)
        .send()
        .await?
        .json::<StarLore>()
        .await
}

pub async fn fetch_random_star(entropy_temperature: f32) -> Result<RandomStarResponse, reqwest::Error> {
    let req = RandomStarRequest { entropy_temperature };

    HTTP_CLIENT
        .post(&format!("{BASE_URL}/random_star"))
        .json(&req)
        .send()
        .await?
        .json::<RandomStarResponse>()
        .await
}
pub async fn fetch_pipeline(req: PipelineRequest) -> Result<PipelineResponse, reqwest::Error> {
    HTTP_CLIENT
        .post(&format!("{BASE_URL}/pipeline"))
        .json(&req)
        .send()
        .await?
        .json::<PipelineResponse>()
        .await
}
