//! Defensive input validation for the game layer.
//!
//! Every value that crosses the trust boundary (network responses,
//! user-driven UI state, internal helpers) flows through one of the
//! `validate_*` functions below. The rules are deliberately
//! conservative: an obviously broken input is rejected with a
//! [`ValidationError`], never silently clamped. Callers can decide
//! how to react (log, ignore, surface to the user).
//!
//! The game layer also defends against the AI backend returning
//! garbage — see [`validate_response_star`].

use thiserror::Error;

use lunar_structures::{PipelineResponse, ResponseStar, World, WorldSummary};

/// Bounds used across the validation module. Centralized so the UI
/// and the game layer can't drift apart.
pub mod limits {
    /// Inclusive max length for a world name (characters, not bytes).
    pub const WORLD_NAME_MAX: usize = 100;
    /// Inclusive max length for a world id (characters, not bytes).
    pub const WORLD_ID_MAX: usize = 64;

    /// Allowed stellar parameter range.
    pub const TEMPERATURE_MIN: f32 = 0.0;
    pub const TEMPERATURE_MAX: f32 = 2.0;

    /// Allowed `bp_rp` range.
    pub const BP_RP_MIN: f32 = 0.0;
    pub const BP_RP_MAX: f32 = 5.0;

    /// Allowed `g_mag` range.
    pub const G_MAG_MIN: f32 = -10.0;
    pub const G_MAG_MAX: f32 = 30.0;

    /// Allowed entropy slider range.
    pub const ENTROPY_MIN: f32 = 0.0;
    pub const ENTROPY_MAX: f32 = 2.0;

    /// Allowed per-sector search radius (parsecs).
    pub const SEARCH_RADIUS_MIN: f32 = 1.0;
    pub const SEARCH_RADIUS_MAX: f32 = 100_000.0;

    /// Inclusive bounds for a world-center coordinate in parsecs.
    pub const COORD_MIN: f32 = -1_000_000.0;
    pub const COORD_MAX: f32 = 1_000_000.0;

    /// Sector key bounds. Anything outside this is off the playable
    /// map.
    pub const SECTOR_KEY_MIN: i32 = -100_000;
    pub const SECTOR_KEY_MAX: i32 = 100_000;

    /// Allowed camera zoom range.
    pub const ZOOM_MIN: f32 = 0.01;
    pub const ZOOM_MAX: f32 = 100.0;
}

/// A type returned by every `validate_*` function.
pub type ValidationResult<T> = Result<T, ValidationError>;

/// All the ways a value can be invalid. Includes the field name so
/// the caller (typically a UI) can show a useful error.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ValidationError {
    #[error("{field} must not be empty")]
    Empty { field: &'static str },

    #[error("{field} is too short: need at least {min} characters, got {actual}")]
    TooShort {
        field: &'static str,
        min: usize,
        actual: usize,
    },

    #[error("{field} is too long: at most {max} characters, got {actual}")]
    TooLong {
        field: &'static str,
        max: usize,
        actual: usize,
    },

    #[error("{field} contains invalid characters: {reason}")]
    InvalidCharacters {
        field: &'static str,
        reason: &'static str,
    },

    #[error("{field} is not a finite number (NaN or infinity)")]
    NotFinite { field: &'static str },

    #[error("{field} = {actual} is out of range [{min}, {max}]")]
    OutOfRange {
        field: &'static str,
        min: f32,
        max: f32,
        actual: f32,
    },

    #[error("{field} = {actual} is out of range [{min}, {max}]")]
    OutOfRangeInt {
        field: &'static str,
        min: i64,
        max: i64,
        actual: i64,
    },

    #[error("response contained {count} invalid entries in field {field}")]
    ResponseContainedInvalid { field: &'static str, count: usize },
}

impl ValidationError {
    /// Convenience: get the field name that produced this error.
    pub fn field(&self) -> &'static str {
        match self {
            Self::Empty { field }
            | Self::TooShort { field, .. }
            | Self::TooLong { field, .. }
            | Self::InvalidCharacters { field, .. }
            | Self::NotFinite { field }
            | Self::OutOfRange { field, .. }
            | Self::OutOfRangeInt { field, .. }
            | Self::ResponseContainedInvalid { field, .. } => field,
        }
    }
}

macro_rules! validate_length {
    ($val:expr, $max:expr, $field:expr) => {
        let len = $val.chars().count();
        if len > $max {
            return Err(ValidationError::TooLong {
                field: $field,
                max: $max,
                actual: len,
            });
        }
    };
}
/// Returns true for finite, non-NaN numbers. Almost every numeric
/// validator below delegates to this.
#[inline]
pub fn is_finite(v: f32) -> bool {
    v.is_finite()
}

fn check_finite(v: f32, field: &'static str) -> ValidationResult<f32> {
    if is_finite(v) {
        Ok(v)
    } else {
        Err(ValidationError::NotFinite { field })
    }
}

fn check_range(v: f32, min: f32, max: f32, field: &'static str) -> ValidationResult<f32> {
    let v = check_finite(v, field)?;
    if v >= min && v <= max {
        Ok(v)
    } else {
        Err(ValidationError::OutOfRange {
            field,
            min,
            max,
            actual: v,
        })
    }
}

fn check_range_int(v: i32, min: i32, max: i32, field: &'static str) -> ValidationResult<i32> {
    if v >= min && v <= max {
        Ok(v)
    } else {
        Err(ValidationError::OutOfRangeInt {
            field,
            min: i64::from(min),
            max: i64::from(max),
            actual: i64::from(v),
        })
    }
}

fn check_non_empty<'a>(s: &'a str, field: &'static str) -> ValidationResult<&'a str> {
    if s.is_empty() {
        Err(ValidationError::Empty { field })
    } else {
        Ok(s)
    }
}

/// Validate a user-supplied world name. Strips leading/trailing
/// whitespace, rejects control characters, and enforces length.
pub fn validate_world_name(name: &str) -> ValidationResult<&str> {
    let field = "world.name";
    let trimmed = name.trim();
    check_non_empty(trimmed, field)?;
    validate_length!(trimmed, limits::WORLD_NAME_MAX, field);
    for c in trimmed.chars() {
        if c.is_control() {
            return Err(ValidationError::InvalidCharacters {
                field,
                reason: "control characters are not allowed",
            });
        }
    }
    Ok(trimmed)
}

/// Validate a world id used in URL paths. Restricts to a safe ASCII
/// subset to avoid path traversal and surprises on the wire.
pub fn validate_world_id(id: &str) -> ValidationResult<&str> {
    let field = "world.id";
    check_non_empty(id, field)?;
    validate_length!(id, limits::WORLD_ID_MAX, field);
    for c in id.chars() {
        let ok = c.is_ascii_alphanumeric() || c == '-' || c == '_';
        if !ok {
            return Err(ValidationError::InvalidCharacters {
                field,
                reason: "only ASCII letters, digits, '-' and '_' are allowed",
            });
        }
    }
    Ok(id)
}

pub fn validate_search_radius(r: f32) -> ValidationResult<f32> {
    check_range(
        r,
        limits::SEARCH_RADIUS_MIN,
        limits::SEARCH_RADIUS_MAX,
        "search_radius",
    )
}

pub fn validate_zoom(z: f32) -> ValidationResult<f32> {
    check_range(z, limits::ZOOM_MIN, limits::ZOOM_MAX, "zoom")
}

pub fn validate_center_x(v: f32) -> ValidationResult<f32> {
    check_range(v, limits::COORD_MIN, limits::COORD_MAX, "center_x")
}

pub fn validate_center_y(v: f32) -> ValidationResult<f32> {
    check_range(v, limits::COORD_MIN, limits::COORD_MAX, "center_y")
}

pub fn validate_center_z(v: f32) -> ValidationResult<f32> {
    check_range(v, limits::COORD_MIN, limits::COORD_MAX, "center_z")
}

pub fn validate_temperature(v: f32) -> ValidationResult<f32> {
    check_range(
        v,
        limits::TEMPERATURE_MIN,
        limits::TEMPERATURE_MAX,
        "temperature",
    )
}

pub fn validate_entropy(v: f32) -> ValidationResult<f32> {
    check_range(v, limits::ENTROPY_MIN, limits::ENTROPY_MAX, "entropy")
}

pub fn validate_bp_rp(v: f32) -> ValidationResult<f32> {
    check_range(v, limits::BP_RP_MIN, limits::BP_RP_MAX, "bp_rp")
}

pub fn validate_g_mag(v: f32) -> ValidationResult<f32> {
    check_range(v, limits::G_MAG_MIN, limits::G_MAG_MAX, "g_mag")
}

pub fn validate_sector_key(key: (i32, i32)) -> ValidationResult<(i32, i32)> {
    let (x, y) = key;
    check_range_int(
        x,
        limits::SECTOR_KEY_MIN,
        limits::SECTOR_KEY_MAX,
        "sector_key.x",
    )?;
    check_range_int(
        y,
        limits::SECTOR_KEY_MIN,
        limits::SECTOR_KEY_MAX,
        "sector_key.y",
    )?;
    Ok(key)
}

/// Validate a [`ResponseStar`] coming back from the
/// AI backend. Catches NaN/Inf coordinates and obviously broken
/// physical parameters.
pub fn validate_response_star(star: &ResponseStar) -> ValidationResult<&ResponseStar> {
    if !is_finite(star.x) || !is_finite(star.y) || !is_finite(star.z) {
        return Err(ValidationError::NotFinite {
            field: "star.coords",
        });
    }
    for (field, v) in [
        ("star.temperature_k", star.temperature_k),
        ("star.radius", star.radius),
        ("star.mass", star.mass),
        ("star.luminosity", star.luminosity),
    ] {
        if !is_finite(v) {
            return Err(ValidationError::NotFinite { field });
        }
        if v < 0.0 {
            return Err(ValidationError::OutOfRange {
                field,
                min: 0.0,
                max: f32::MAX,
                actual: v,
            });
        }
    }
    for v in star.velocity_vector {
        if !is_finite(v) {
            return Err(ValidationError::NotFinite {
                field: "star.velocity_vector",
            });
        }
    }
    if !star.x.is_finite() {
        return Err(ValidationError::NotFinite { field: "star.x" });
    }
    Ok(star)
}

/// Validate a vector of stars. Returns the first invalid entry.
pub fn validate_response_stars(stars: &[ResponseStar]) -> ValidationResult<()> {
    for star in stars {
        validate_response_star(star)?;
    }
    Ok(())
}

/// Validate a [`World`] coming from the AI backend.
pub fn validate_world(world: &World) -> ValidationResult<&World> {
    let _ = validate_world_id(&world.id)?;
    let _ = validate_world_name(&world.name)?;
    validate_center_x(world.center_x)?;
    validate_center_y(world.center_y)?;
    validate_center_z(world.center_z)?;
    validate_entropy(world.temperature)?;
    validate_bp_rp(world.bp_rp)?;
    validate_g_mag(world.g_mag)?;
    validate_response_stars(&world.stars)?;
    Ok(world)
}

/// Validate a [`WorldSummary`].
pub fn validate_world_summary(world: &WorldSummary) -> ValidationResult<&WorldSummary> {
    let _ = validate_world_id(&world.id)?;
    let _ = validate_world_name(&world.name)?;
    validate_center_x(world.center_x)?;
    validate_center_y(world.center_y)?;
    validate_center_z(world.center_z)?;
    Ok(world)
}

/// Validate the texture inside a [`PipelineResponse`].
pub fn validate_pipeline(pipeline: &PipelineResponse) -> ValidationResult<&PipelineResponse> {
    if pipeline.siren.width == 0 || pipeline.siren.height == 0 {
        return Err(ValidationError::OutOfRangeInt {
            field: "siren.width_or_height",
            min: 1,
            max: i64::from(i32::MAX),
            actual: 0,
        });
    }
    let expected = (pipeline.siren.width as usize)
        .checked_mul(pipeline.siren.height as usize)
        .and_then(|n| n.checked_mul(3));
    let Some(expected) = expected else {
        return Err(ValidationError::OutOfRangeInt {
            field: "siren.size",
            min: 0,
            max: i64::from(i32::MAX),
            actual: i64::from(i32::MAX),
        });
    };
    if pipeline.siren.pixels.len() != expected {
        return Err(ValidationError::ResponseContainedInvalid {
            field: "siren.pixels",
            count: 1,
        });
    }
    if !is_finite(pipeline.pinn.temperature_k)
        || !is_finite(pipeline.pinn.radius_solar)
        || !is_finite(pipeline.pinn.mass_solar)
        || !is_finite(pipeline.pinn.luminosity_solar)
    {
        return Err(ValidationError::NotFinite { field: "pinn.*" });
    }
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn star_finite() -> ResponseStar {
        ResponseStar {
            id: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            temperature_k: 5778.0,
            radius: 1.0,
            mass: 1.0,
            luminosity: 1.0,
            description: String::new(),
            name: String::new(),
            type_hint: String::new(),
            velocity_vector: [0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn world_name_strips_whitespace_and_accepts_normal() {
        let trimmed = validate_world_name("  Vela Rim  ").unwrap();
        assert_eq!(trimmed, "Vela Rim");
    }

    #[test]
    fn world_name_rejects_empty() {
        assert!(matches!(
            validate_world_name("   "),
            Err(ValidationError::Empty { .. })
        ));
    }

    #[test]
    fn world_name_rejects_too_long() {
        let long = "a".repeat(limits::WORLD_NAME_MAX + 1);
        assert!(matches!(
            validate_world_name(&long),
            Err(ValidationError::TooLong { .. })
        ));
    }

    #[test]
    fn world_name_rejects_control_characters() {
        assert!(matches!(
            validate_world_name("Vela\u{0000}Rim"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }

    #[test]
    fn world_id_accepts_safe_ascii() {
        assert_eq!(validate_world_id("abc-123_XYZ").unwrap(), "abc-123_XYZ");
    }

    #[test]
    fn world_id_rejects_path_traversal() {
        assert!(matches!(
            validate_world_id("../etc/passwd"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }

    #[test]
    fn world_id_rejects_unicode() {
        assert!(matches!(
            validate_world_id("вела"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }

    #[test]
    fn floats_reject_nan_and_infinity() {
        assert!(matches!(
            validate_center_x(f32::NAN),
            Err(ValidationError::NotFinite { .. })
        ));
        assert!(matches!(
            validate_temperature(f32::INFINITY),
            Err(ValidationError::NotFinite { .. })
        ));
        assert!(matches!(
            validate_temperature(f32::NEG_INFINITY),
            Err(ValidationError::NotFinite { .. })
        ));
    }

    #[test]
    fn temperature_range_enforced() {
        assert!(validate_temperature(0.7).is_ok());
        assert!(validate_temperature(0.0).is_ok());
        assert!(validate_temperature(2.0).is_ok());
        assert!(matches!(
            validate_temperature(-0.1),
            Err(ValidationError::OutOfRange { .. })
        ));
        assert!(matches!(
            validate_temperature(2.5),
            Err(ValidationError::OutOfRange { .. })
        ));
    }

    #[test]
    fn search_radius_rejects_zero_and_negatives() {
        assert!(validate_search_radius(0.0).is_err());
        assert!(validate_search_radius(-5.0).is_err());
        assert!(validate_search_radius(50.0).is_ok());
    }

    #[test]
    fn sector_key_rejects_insane_values() {
        assert!(validate_sector_key((0, 0)).is_ok());
        assert!(validate_sector_key((100_000, 100_000)).is_ok());
        assert!(validate_sector_key((100_001, 0)).is_err());
        assert!(validate_sector_key((-100_001, 0)).is_err());
    }

    #[test]
    fn response_star_rejects_nan_coords() {
        let mut s = star_finite();
        s.x = f32::NAN;
        assert!(matches!(
            validate_response_star(&s),
            Err(ValidationError::NotFinite { .. })
        ));
    }

    #[test]
    fn response_star_rejects_negative_physical() {
        let mut s = star_finite();
        s.mass = -1.0;
        assert!(matches!(
            validate_response_star(&s),
            Err(ValidationError::OutOfRange { .. })
        ));
    }

    #[test]
    fn response_star_rejects_nan_velocity() {
        let mut s = star_finite();
        s.velocity_vector = [f32::NAN, 0.0, 0.0];
        assert!(matches!(
            validate_response_star(&s),
            Err(ValidationError::NotFinite { .. })
        ));
    }

    #[test]
    fn world_validates_id_name_and_coords() {
        let mut w = World {
            id: "good-id".into(),
            name: "Good".into(),
            created_at: 0,
            center_x: 0.0,
            center_y: 0.0,
            center_z: 0.0,
            temperature: 0.5,
            bp_rp: 1.0,
            g_mag: 10.0,
            stars: vec![],
        };
        assert!(validate_world(&w).is_ok());
        w.id = "../bad".into();
        assert!(validate_world(&w).is_err());
    }
}
