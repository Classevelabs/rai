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
///
/// Trailing `//` comments are excluded before the check. An apostrophe in
/// ordinary English prose ("the block's contents") is not an unterminated
/// literal, and a guard that says it is gets worked around rather than obeyed
/// — which is how a guard stops protecting anything.
#[test]
fn no_javascript_string_literal_spans_a_line_break() {
    for (number, line) in script_body().lines().enumerate() {
        let code = strip_line_comment(line);
        for quote in ['"', '\''] {
            let mut open = false;
            let mut escaped = false;
            for character in code.chars() {
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

/// The code part of one line: everything before a `//` that is not itself
/// inside a string or escaped (so `"http://x"` and a `\/` in a regex are left
/// alone).
fn strip_line_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let (mut single, mut double, mut escaped) = (false, false, false);
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'\'' && !double {
            single = !single;
        } else if byte == b'"' && !single {
            double = !double;
        } else if byte == b'/' && !single && !double && bytes.get(index + 1) == Some(&b'/') {
            return &line[..index];
        }
        index += 1;
    }
    line
}

/// The guard must still catch the defect it exists for: a literal that runs
/// off the end of its line. This is the shape that once blanked the page.
#[test]
fn the_guard_still_catches_a_real_unterminated_literal() {
    assert_eq!(
        strip_line_comment("const a = 1; // note's here"),
        "const a = 1; "
    );
    // A `//` inside a string is not a comment.
    assert_eq!(strip_line_comment("f(\"http://x\");"), "f(\"http://x\");");
    // And an apostrophe in prose after `//` is invisible to the check.
    let commented = strip_line_comment("// the block's contents");
    assert!(
        !commented.contains('\''),
        "prose apostrophes must be stripped, got {commented:?}"
    );
}

#[test]
fn the_page_is_structurally_complete() {
    assert!(PAGE.starts_with("<!DOCTYPE html>"), "missing doctype");
    assert!(PAGE.trim_end().ends_with("</html>"), "page is truncated");
    assert_eq!(
        PAGE.matches("<script>").count(),
        1,
        "expected one script block"
    );
    assert_eq!(PAGE.matches("</script>").count(), 1);
    assert_eq!(
        PAGE.matches("<style>").count(),
        1,
        "expected one style block"
    );
    assert_eq!(PAGE.matches("</style>").count(), 1);
    for id in [
        "s-library",
        "s-convert",
        "s-run",
        "modelList",
        "convertBtn",
        "inspectBtn",
        "sendBtn",
    ] {
        assert!(
            PAGE.contains(&format!("id=\"{id}\"")),
            "missing element #{id}"
        );
    }
}

/// The Content-Security-Policy the server sends allows only same-origin
/// connections and inline style/script. An external URL would be blocked at
/// runtime and the feature would silently not work.
#[test]
fn the_page_loads_nothing_from_the_network() {
    for pattern in [
        "src=\"http",
        "href=\"http",
        "@import",
        "fonts.googleapis",
        "cdn.",
    ] {
        assert!(
            !PAGE.contains(pattern),
            "page references an external resource ({pattern}); the CSP forbids it"
        );
    }
}
