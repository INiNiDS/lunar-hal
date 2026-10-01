
use thiserror::Error;

use crate::api_client::ApiError;
use crate::validation::ValidationError;

#[derive(Debug, Error)]
pub enum StellarSceneError {
    #[error(transparent)]
    Validation(#[from] ValidationError),

    #[error(transparent)]
    Api(#[from] ApiError),
}

impl From<StellarSceneError> for ValidationError {
    fn from(e: StellarSceneError) -> Self {
        match e {
            StellarSceneError::Validation(v) => v,
            StellarSceneError::Api(_) => ValidationError::Empty {
                field: "api.backend",
            },
        }
    }
}

impl StellarSceneError {
    pub fn is_validation(&self) -> bool {
        matches!(self, Self::Validation(_))
    }

    pub fn field(&self) -> Option<&'static str> {
        match self {
            Self::Validation(v) => Some(v.field()),
            Self::Api(_) => None,
        }
    }
}
