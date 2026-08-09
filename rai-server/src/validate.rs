//! Input validation shared by every transport.
//!
//! There is exactly one text limit for the whole stack, [`rai_core::MAX_TEXT_BYTES`], and it is
//! measured in **bytes** — not characters, so a multi-byte body cannot amplify past the limit.
//! The layering is:
//!
//! 1. **Transport** (these functions, used by the REST handlers and the MCP server): rejects
//!    oversized or empty input before an embedding request is made, so the client sees a clear
//!    error instead of a provider failure.
//! 2. **Library** (`rai_core`'s memory manager): enforces the same bound again, because the
//!    crate is usable without either transport.
//! 3. **Persistence**: tolerates a larger historical bound so snapshots written by earlier
//!    releases still load. Nothing new can reach that size.

pub use rai_core::{MAX_INTERSECTION_CONCEPTS, MAX_TEXT_BYTES};

/// Validate one free-text field. The error is a client-facing message.
pub fn validate_text(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(format!("{field} exceeds the {MAX_TEXT_BYTES}-byte limit"));
    }
    Ok(())
}

/// Validate an intersection concept list, including its combined size.
pub fn validate_concepts(concepts: &[String]) -> Result<(), String> {
    if concepts.len() < 2 {
        return Err("at least two concepts are required".to_string());
    }
    if concepts.len() > MAX_INTERSECTION_CONCEPTS {
        return Err(format!(
            "concepts exceeds the {MAX_INTERSECTION_CONCEPTS}-item limit"
        ));
    }

    let mut total_bytes = 0usize;
    for concept in concepts {
        validate_text("concept", concept)?;
        total_bytes = total_bytes
            .checked_add(concept.len())
            .ok_or_else(|| "combined concept text is too large".to_string())?;
    }
    if total_bytes > MAX_TEXT_BYTES {
        return Err(format!(
            "combined concept text exceeds the {MAX_TEXT_BYTES}-byte limit"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_rejects_empty_and_amplifying_inputs() {
        assert!(validate_text("query", "   ").is_err());
        assert!(validate_text("query", &"x".repeat(MAX_TEXT_BYTES + 1)).is_err());
        assert!(validate_concepts(&[]).is_err());
        assert!(validate_concepts(&vec!["x".to_string(); MAX_INTERSECTION_CONCEPTS + 1]).is_err());
        assert!(validate_concepts(&["valid".to_string(), " ".to_string()]).is_err());
    }

    /// The limit is bytes, so a multi-byte string that fits a character count must still be
    /// rejected once it exceeds the byte budget.
    #[test]
    fn the_limit_is_measured_in_bytes_not_characters() {
        let multibyte = "é".repeat(MAX_TEXT_BYTES / 2);
        assert_eq!(multibyte.chars().count(), MAX_TEXT_BYTES / 2);
        assert_eq!(multibyte.len(), MAX_TEXT_BYTES);
        assert!(validate_text("content", &multibyte).is_ok());

        let over = format!("{multibyte}é");
        assert!(over.chars().count() < MAX_TEXT_BYTES);
        assert!(validate_text("content", &over).is_err());
    }

    /// The library enforces the same constant, so a transport-accepted value can never be
    /// rejected by the layer beneath it.
    #[test]
    fn the_transport_limit_matches_the_library_limit() {
        assert_eq!(MAX_TEXT_BYTES, rai_core::MAX_TEXT_BYTES);
        assert_eq!(
            MAX_INTERSECTION_CONCEPTS,
            rai_core::MAX_INTERSECTION_CONCEPTS
        );
    }
}
