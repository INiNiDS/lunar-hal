use std::sync::Arc;

pub mod ai_jobs;
pub mod data_jobs;
pub mod db;
pub mod jobs;
pub mod reports;
pub mod system;

use crate::jobs::JobRegistry;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<JobRegistry>,
}
