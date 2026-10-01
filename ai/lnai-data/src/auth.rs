
pub fn resolve_env_secret(name: &str) -> Option<String> {
    let value = std::env::var(name).ok()?;
    let trimmed = value.trim().to_string();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed)
}

pub fn redact(secret: &str) -> String {
    if secret.trim().is_empty() {
        return "<unset>".to_string();
    }
    let head: String = secret.chars().take(4).collect();
    format!("{head}…({})", secret.len())
}

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
