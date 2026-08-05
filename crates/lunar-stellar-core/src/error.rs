//! Unified error type for the game layer.
//!
//! Every fallible operation on [`crate::Game`] returns
//! [`Result<T, GameError>`]. The two variants cover the two trust
//! boundaries the game crosses:
//!
//! * [`GameError::Validation`] — the caller's input was invalid.
//! * [`GameError::Api`] — the AI backend returned an error.
//!
//! Both variants are lightweight to construct and easy to match.

use thiserror::Error;

use crate::api_client::ApiError;
use crate::validation::ValidationError;

/// Top-level error for game operations.
#[derive(Debug, Error)]
pub enum GameError {
    /// The caller provided an invalid value (empty name, NaN, out of
    /// range, etc.). The contained [`ValidationError`] is local to
    /// the game and does not depend on the AI backend.
    #[error(transparent)]
    Validation(#[from] ValidationError),

    /// The AI backend rejected the request or returned a malformed
    /// response. Surfaced verbatim so the UI can display it.
    #[error(transparent)]
    Api(#[from] ApiError),
}

impl From<GameError> for ValidationError {
    fn from(e: GameError) -> Self {
        match e {
            GameError::Validation(v) => v,
            GameError::Api(_) => ValidationError::Empty {
                field: "api.backend",
            },
        }
    }
}

impl GameError {
    /// Returns `true` if this error is the caller's fault (the input
    /// was bad). The UI can use this to decide whether to show a
    /// validation hint or a network error toast.
    pub fn is_validation(&self) -> bool {
        matches!(self, Self::Validation(_))
    }

    /// Field name that caused the error if known.
    pub fn field(&self) -> Option<&'static str> {
        match self {
            Self::Validation(v) => Some(v.field()),
            Self::Api(_) => None,
        }
    }
}
