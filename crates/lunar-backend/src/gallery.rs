//! Filesystem-backed SIREN Gallery repository and REST handlers.
//!
//! Metadata lives in `gallery/<id>/metadata.json`; generated binary assets are
//! persisted alongside it as `texture.png` and `thumb.png`. Every write goes
//! through a temporary file plus rename so a partial process crash cannot
//! produce a valid-looking JSON record with truncated contents.

use std::collections::HashMap;
use std::path::{Path as StdPath, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use lunar_structures::{
    CreateGalleryStarRequest, GalleryListResponse, GalleryStar, UpdateGalleryStarRequest,
};
use lunar_utils::encode_rgb_png;
use serde::Deserialize;

use crate::{AppState, generate_siren_pixels};
use crate::scenes::calculate_absolute_magnitude;

#[derive(Default)]
pub struct GalleryStore {
    inner: Arc<RwLock<HashMap<String, GalleryStar>>>,
    request_ids: Arc<RwLock<HashMap<String, String>>>,
    dir: PathBuf,
}

impl GalleryStore {
    pub fn new(dir: PathBuf) -> Self {
        let store = Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            request_ids: Arc::new(RwLock::new(HashMap::new())),
            dir,
        };
        store.load_from_disk();
        store
    }

    fn load_from_disk(&self) {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path().join("metadata.json");
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(star) = serde_json::from_slice::<GalleryStar>(&bytes) else {
                // One corrupt record must not prevent the rest of the gallery
                // from loading. The file remains available for diagnostics.
                continue;
            };
            if let Some(request_id) = &star.request_id {
                if let Ok(mut requests) = self.request_ids.write() {
                    requests.insert(request_id.clone(), star.id.clone());
                }
            }
            if let Ok(mut records) = self.inner.write() {
                records.insert(star.id.clone(), star);
            }
        }
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0)
    }

    fn generate_id(&self, request_id: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let mut seed = nanos as u64;
        for byte in request_id.bytes() {
            seed ^= byte as u64;
            seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        format!("g{:016x}", seed)
    }

    fn record_dir(&self, id: &str) -> PathBuf {
        self.dir.join(id)
    }

    fn atomic_write(path: &StdPath, bytes: &[u8]) -> Result<(), String> {
        let parent = path
            .parent()
            .ok_or_else(|| "asset path has no parent directory".to_string())?;
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "asset path has no file name".to_string())?;
        let temporary = parent.join(format!(".{file_name}.{stamp}.tmp"));
        std::fs::write(&temporary, bytes).map_err(|error| error.to_string())?;
        std::fs::rename(&temporary, path).map_err(|error| error.to_string())
    }

    fn persist(&self, record: &GalleryStar, texture: Option<&[u8]>) -> Result<(), String> {
        let root = self.record_dir(&record.id);
        let metadata = serde_json::to_vec_pretty(record).map_err(|error| error.to_string())?;
        Self::atomic_write(&root.join("metadata.json"), &metadata)?;
        if let Some(texture) = texture {
            Self::atomic_write(&root.join("texture.png"), texture)?;
            // The 128px rendering is already suitable as a responsive
            // thumbnail and avoids embedding pixels inside metadata JSON.
            Self::atomic_write(&root.join("thumb.png"), texture)?;
        }
        Ok(())
    }

    pub fn create(
        &self,
        request: CreateGalleryStarRequest,
        texture: Option<Vec<u8>>,
    ) -> Result<GalleryStar, String> {
        let request_id = request.request_id.trim();
        if request_id.is_empty() {
            return Err("request_id is required for idempotent Gallery creation".into());
        }
        if let Some(existing_id) = self
            .request_ids
            .read()
            .map_err(|error| error.to_string())?
            .get(request_id)
            .cloned()
        {
            return self
                .get(&existing_id)
                .ok_or_else(|| "idempotency index points to a missing Gallery record".into());
        }

        let id = self.generate_id(request_id);
        let now = Self::now();
        let has_texture = texture.as_ref().is_some_and(|bytes| !bytes.is_empty());
        let record = GalleryStar {
            id: id.clone(),
            schema_version: lunar_structures::GALLERY_SCHEMA_VERSION,
            created_at: now,
            updated_at: now,
            source: request.source,
            star: request.star,
            inputs: request.inputs,
            pinn: request.pinn,
            metadata: request.metadata,
            texture_url: has_texture.then(|| format!("/gallery/stars/{id}/texture.png")),
            thumbnail_url: has_texture.then(|| format!("/gallery/stars/{id}/thumbnail")),
            name: request.name,
            tags: normalize_tags(request.tags),
            notes: normalize_optional(request.notes),
            request_id: Some(request_id.to_string()),
        };
        self.persist(&record, texture.as_deref())?;
        self.inner
            .write()
            .map_err(|error| error.to_string())?
            .insert(id.clone(), record.clone());
        self.request_ids
            .write()
            .map_err(|error| error.to_string())?
            .insert(request_id.to_string(), id);
        Ok(record)
    }

    pub fn get(&self, id: &str) -> Option<GalleryStar> {
        self.inner.read().ok()?.get(id).cloned()
    }

    pub fn list(
        &self,
        cursor: Option<&str>,
        limit: usize,
        sort: Option<&str>,
        query: Option<&str>,
    ) -> GalleryListResponse {
        let query = query.map(|value| value.trim().to_lowercase()).filter(|value| !value.is_empty());
        let mut records = self
            .inner
            .read()
            .map(|records| records.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        records.retain(|record| {
            let Some(query) = &query else { return true };
            record.id.to_lowercase().contains(query)
                || record.name.as_deref().is_some_and(|name| name.to_lowercase().contains(query))
                || record.star.name.to_lowercase().contains(query)
                || record.tags.iter().any(|tag| tag.to_lowercase().contains(query))
        });
        match sort.unwrap_or("updated_desc") {
            "name" => records.sort_by(|left, right| gallery_name(left).cmp(&gallery_name(right))),
            "created_asc" => records.sort_by_key(|record| record.created_at),
            _ => records.sort_by_key(|record| std::cmp::Reverse((record.updated_at, record.id.clone()))),
        }
        let start = cursor
            .and_then(|cursor| records.iter().position(|record| record.id == cursor))
            .map(|index| index + 1)
            .unwrap_or(0);
        let limit = limit.clamp(1, 100);
        let stars = records.into_iter().skip(start).take(limit).collect::<Vec<_>>();
        let next_cursor = (stars.len() == limit)
            .then(|| stars.last().map(|record| record.id.clone()))
            .flatten();
        GalleryListResponse { stars, next_cursor }
    }

    pub fn update(&self, id: &str, update: UpdateGalleryStarRequest) -> Result<GalleryStar, String> {
        let mut record = self
            .get(id)
            .ok_or_else(|| format!("Gallery star {id} not found"))?;
        if let Some(name) = update.name {
            record.name = normalize_optional(Some(name));
        }
        if let Some(tags) = update.tags {
            record.tags = normalize_tags(tags);
        }
        if let Some(notes) = update.notes {
            record.notes = normalize_optional(Some(notes));
        }
        record.updated_at = Self::now();
        self.persist(&record, None)?;
        self.inner
            .write()
            .map_err(|error| error.to_string())?
            .insert(record.id.clone(), record.clone());
        Ok(record)
    }

    pub fn delete(&self, id: &str) -> bool {
        let Some(record) = self.inner.write().ok().and_then(|mut records| records.remove(id)) else {
            return false;
        };
        if let Some(request_id) = record.request_id {
            if let Ok(mut request_ids) = self.request_ids.write() {
                request_ids.remove(&request_id);
            }
        }
        let _ = std::fs::remove_dir_all(self.record_dir(id));
        true
    }

    pub fn asset(&self, id: &str, name: &str) -> Option<Vec<u8>> {
        self.get(id)?;
        std::fs::read(self.record_dir(id).join(name)).ok()
    }
}

fn normalize_optional(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn normalize_tags(tags: Vec<String>) -> Vec<String> {
    let mut result = tags
        .into_iter()
        .filter_map(|tag| normalize_optional(Some(tag)))
        .collect::<Vec<_>>();
    result.sort();
    result.dedup();
    result
}

fn gallery_name(record: &GalleryStar) -> String {
    record
        .name
        .clone()
        .unwrap_or_else(|| record.star.name.clone())
        .to_lowercase()
}

#[derive(Deserialize, Default)]
pub struct GalleryListQuery {
    pub cursor: Option<String>,
    pub limit: Option<usize>,
    pub sort: Option<String>,
    pub query: Option<String>,
}

pub(crate) async fn texture_for(request: &CreateGalleryStarRequest) -> Option<Vec<u8>> {
    let size = 128;
    let star = &request.star;
    let magnitude = calculate_absolute_magnitude(
        request.inputs.x_pc,
        request.inputs.y_pc,
        request.inputs.z_pc,
        request.inputs.g_mag,
    );
    let log_teff = if star.temperature_k > 0.0 {
        star.temperature_k.log10()
    } else {
        3.75
    };
    let pixels = generate_siren_pixels(size, size, request.inputs.bp_rp, magnitude, log_teff).await?;
    Some(encode_rgb_png(&pixels, size, size))
}

pub async fn list_gallery_stars(
    State(state): State<AppState>,
    Query(query): Query<GalleryListQuery>,
) -> Json<GalleryListResponse> {
    Json(state.gallery.list(
        query.cursor.as_deref(),
        query.limit.unwrap_or(36),
        query.sort.as_deref(),
        query.query.as_deref(),
    ))
}

pub async fn create_gallery_star(
    State(state): State<AppState>,
    Json(request): Json<CreateGalleryStarRequest>,
) -> Result<Json<GalleryStar>, (StatusCode, String)> {
    let texture = texture_for(&request).await;
    state
        .gallery
        .create(request, texture)
        .map(Json)
        .map_err(|error| (StatusCode::BAD_REQUEST, error))
}

pub async fn get_gallery_star(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<GalleryStar>, (StatusCode, String)> {
    state
        .gallery
        .get(&id)
        .map(Json)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Gallery star {id} not found")))
}

pub async fn update_gallery_star(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(update): Json<UpdateGalleryStarRequest>,
) -> Result<Json<GalleryStar>, (StatusCode, String)> {
    state
        .gallery
        .update(&id, update)
        .map(Json)
        .map_err(|error| {
            let status = if state.gallery.get(&id).is_some() {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::NOT_FOUND
            };
            (status, error)
        })
}

pub async fn delete_gallery_star(State(state): State<AppState>, Path(id): Path<String>) -> StatusCode {
    if state.gallery.delete(&id) {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

fn image_response(bytes: Option<Vec<u8>>) -> Result<Response, (StatusCode, String)> {
    let bytes = bytes.ok_or_else(|| (StatusCode::NOT_FOUND, "Gallery asset not found".to_string()))?;
    Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response())
}

pub async fn gallery_texture(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, (StatusCode, String)> {
    image_response(state.gallery.asset(&id, "texture.png"))
}

pub async fn gallery_thumbnail(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, (StatusCode, String)> {
    image_response(state.gallery.asset(&id, "thumb.png"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lunar_structures::{GallerySource, ResponseStar, StarModelInputs};

    fn temporary_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lunar-gallery-{label}-{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ))
    }

    fn request(id: &str) -> CreateGalleryStarRequest {
        CreateGalleryStarRequest {
            request_id: id.to_string(),
            source: GallerySource::SceneSaved,
            star: ResponseStar {
                id: 7,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                temperature_k: 5778.0,
                radius: 1.0,
                mass: 1.0,
                luminosity: 1.0,
                description: "solar reference".into(),
                name: "Sol".into(),
                type_hint: "G".into(),
                velocity_vector: [0.0; 3],
            },
            inputs: StarModelInputs {
                x_pc: 1.0,
                y_pc: 2.0,
                z_pc: 3.0,
                bp_rp: 0.65,
                g_mag: 4.83,
                entropy_temperature: None,
            },
            pinn: None,
            metadata: None,
            name: Some("Solar reference".into()),
            tags: vec!["reference".into()],
            notes: None,
        }
    }

    #[test]
    fn gallery_records_persist_and_post_is_idempotent() {
        let dir = temporary_directory("idempotent");
        let store = GalleryStore::new(dir.clone());
        let first = store.create(request("drop-1"), Some(vec![137, 80, 78, 71])).unwrap();
        let duplicate = store.create(request("drop-1"), None).unwrap();
        assert_eq!(first.id, duplicate.id);
        assert!(dir.join(&first.id).join("metadata.json").exists());
        assert!(dir.join(&first.id).join("texture.png").exists());

        let reloaded = GalleryStore::new(dir.clone());
        assert_eq!(reloaded.get(&first.id).unwrap().star.name, "Sol");
        assert_eq!(reloaded.list(None, 10, None, None).stars.len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn corrupt_records_do_not_block_valid_gallery_records() {
        let dir = temporary_directory("corrupt-record");
        let store = GalleryStore::new(dir.clone());
        let record = store.create(request("valid-record"), None).unwrap();
        let broken = dir.join("broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("metadata.json"), b"this is not JSON").unwrap();

        let reloaded = GalleryStore::new(dir.clone());
        assert!(reloaded.get(&record.id).is_some());
        assert!(reloaded.get("broken").is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
