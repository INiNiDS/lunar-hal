//! Secret handling primitives (stage 4A / пункт 11):
//!
//! * [`resolve_env_secret`] — read named credentials from the process
//!   environment with a strict allow-listing flow;
//! * [`redact`] — one canonical way to render any potentially sensitive value
//!   in diagnostics. Redaction tests assert shapes, not secrets.

/// Reads a secret from the environment. Returns `(value_present, name)` so
/// diagnostics can report availability WITHOUT exposing the value itself.
pub fn resolve_env_secret(name: &str) -> Option<String> {
    let value = std::env::var(name).ok()?;
    let trimmed = value.trim().to_string();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed)
}

/// Renders a diagnostic-safe string for arbitrary input:
/// * empty -> `<unset>`
/// * otherwise first 4 chars + fixed length marker (never the tail).
///
/// This intentionally keeps enough head characters to distinguish between two
/// different keys in bug reports while hiding ~all entropy.
pub fn redact(secret: &str) -> String {
    if secret.trim().is_empty() {
        return "<unset>".to_string();
    }
    let head: String = secret.chars().take(4).collect();
    format!("{head}…({})", secret.len())
}

/// A guard-style container that prevents accidental secret leakage through
/// Debug/Display: the inner value is only reachable through [`Self::expose`],
/// used exclusively in code paths that talk to the remote service.
#[derive(Clone)]
pub struct SecretBox(String);

impl std::fmt::Debug for SecretBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecretBox({})", redact(&self.0))
    }
}

impl std::fmt::Display for SecretBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", redact(&self.0))
    }
}

impl SecretBox {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Only call sites performing actual authentication may use this.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_never_shows_full_value() {
        let rendered = redact("sk-1234567890abcdef");
        assert_eq!(rendered, "sk-1…(19)");
        assert!(rendered.ends_with(')'));
        assert!(!rendered.contains("7890"));
    }

    #[test]
    fn redact_of_empty_is_unset_marker() {
        assert_eq!(redact(""), "<unset>");
        assert_eq!(redact("   "), "<unset>");
    }

    #[test]
    fn secret_box_hides_value_in_every_formatter() {
        let s = SecretBox::new("GAIA-very-secret-token-value");
        let dbg = format!("{s:?}");
        let disp = format!("{s}");
        // Neither formatter may contain more than the 4-char head.
        assert!(!dbg.contains("very-secret"), "{dbg}");
        assert!(!disp.contains("very-secret"), "{disp}");
        assert!(s.expose() == "GAIA-very-secret-token-value");
    }

    #[test]
    fn env_resolution_ignores_empty_and_whitespace_values() {
        let probe = "LNAI_DATA_TEST_SECRET_PROBE";
        unsafe { std::env::set_var(probe, "   ") };
        assert!(resolve_env_secret(probe).is_none());
        unsafe { std::env::set_var(probe, " real ") };
        assert_eq!(resolve_env_secret(probe).as_deref(), Some("real"));
        unsafe { std::env::remove_var(probe) };
    }
}
