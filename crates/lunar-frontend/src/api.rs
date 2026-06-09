use lunar_structures::{GnnRequest, GnnResponse, ImageRequest, ImageResponse, PinnRequest, PinnResponse, StarLore};
use once_cell::sync::Lazy;

static HTTP_CLIENT: Lazy<reqwest::Client> = Lazy::new(reqwest::Client::new);
const BASE_URL: &str = "https://api.star-system.space";

pub async fn fetch_pinn(target_id: u64) -> Result<PinnResponse, reqwest::Error> {
    let req = PinnRequest { target_id, ..Default::default() };

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
    temperature: f32
) -> Result<GnnResponse, reqwest::Error> {
    let req = GnnRequest {
        center_x: cx,
        center_y: cy,
        center_z: cz,
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

pub async fn fetch_star_image(star_id: u64, resolution: (u32, u32)) -> Result<ImageResponse, reqwest::Error> {
    let req = ImageRequest {
        star_id,
        width: resolution.0,
        height: resolution.1
    };

    HTTP_CLIENT
        .post(&format!("{BASE_URL}/image"))
        .json(&req)
        .send()
        .await?
        .json::<ImageResponse>()
        .await
}

pub async fn fetch_star_description(gnn_response: GnnResponse, pinn_response: PinnResponse) -> Result<StarLore, reqwest::Error> {
    dbg!(HTTP_CLIENT
        .post(&format!("{BASE_URL}/description"))
        .json(&serde_json::json!({
                "gnn_response": gnn_response,
                "pinn_response": pinn_response,
            }))
        .send()
        .await?
        .json::<StarLore>()
        .await)
}