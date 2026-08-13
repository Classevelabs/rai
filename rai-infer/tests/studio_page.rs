//! Guards on the interface that ships inside the binary.
//!
//! `studio.html` is compiled in with `include_str!`, so a malformed page is a
//! shipped defect that no Rust test would otherwise catch: the server serves
//! it happily and the browser renders a blank screen. A stray newline inside a
//! JavaScript string literal did exactly that once.

const PAGE: &str = include_str!("../src/cli/studio.html");

fn script_body() -> &'static str {
    let start = PAGE.find("<script>").expect("page has a script block") + "<script>".len();
    let end = PAGE.rfind("</script>").expect("script block is closed");
    &PAGE[start..end]
}

/// A string literal that runs off the end of its line is a syntax error, and a
/// syntax error anywhere in the block stops every line of it from running.
#[test]
fn no_javascript_string_literal_spans_a_line_break() {
    for (number, line) in script_body().lines().enumerate() {
        for quote in ['"', '\''] {
            let mut open = false;
            let mut escaped = false;
            for character in line.chars() {
                if escaped {
                    escaped = false;
                } else if character == '\\' {
                    escaped = true;
                } else if character == quote {
                    open = !open;
                }
            }
            assert!(
                !open,
                "unterminated {quote} literal on script line {}: {}",
                number + 1,
                line.trim()
            );
        }
    }
}

#[test]
fn the_page_is_structurally_complete() {
    assert!(PAGE.starts_with("<!DOCTYPE html>"), "missing doctype");
    assert!(PAGE.trim_end().ends_with("</html>"), "page is truncated");
    assert_eq!(PAGE.matches("<script>").count(), 1, "expected one script block");
    assert_eq!(PAGE.matches("</script>").count(), 1);
    assert_eq!(PAGE.matches("<style>").count(), 1, "expected one style block");
    assert_eq!(PAGE.matches("</style>").count(), 1);
    for id in [
        "s-library", "s-convert", "s-run", "modelList", "convertBtn", "inspectBtn", "sendBtn",
    ] {
        assert!(PAGE.contains(&format!("id=\"{id}\"")), "missing element #{id}");
    }
}

/// The Content-Security-Policy the server sends allows only same-origin
/// connections and inline style/script. An external URL would be blocked at
/// runtime and the feature would silently not work.
#[test]
fn the_page_loads_nothing_from_the_network() {
    for pattern in ["src=\"http", "href=\"http", "@import", "fonts.googleapis", "cdn."] {
        assert!(
            !PAGE.contains(pattern),
            "page references an external resource ({pattern}); the CSP forbids it"
        );
    }
}
