//! Guards on the double-click launchers.
//!
//! These scripts are how someone who never opens a terminal starts RAI, and
//! nothing else tests them: they are not compiled, not linted, and not run by
//! CI. That gap shipped a real defect — every launcher looked for the binary
//! only in its own directory, while the release archive puts the binary at the
//! root and the launchers in `launchers/`, so a downloaded archive failed on
//! the first double-click with "Could not find rai".
//!
//! The archive layout is fixed by `.github/workflows/release.yml`:
//!
//! ```text
//!   rai-0.2.0-<target>/
//!     rai(.exe)          <- the binary
//!     launchers/         <- these scripts, one level down
//! ```

use std::path::{Path, PathBuf};

fn launcher_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is rai-infer/; the launchers live at the repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("launchers")
}

fn read(name: &str) -> String {
    let path = launcher_dir().join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// Every launcher must look for the binary in the directory *above* itself as
/// well as beside itself, because that is where the release archive puts it.
#[test]
fn every_launcher_searches_the_archive_root() {
    for (name, parent_reference) in [
        ("rai-studio.cmd", ".."),
        ("rai-studio.sh", ".."),
        ("rai-studio.command", ".."),
    ] {
        let script = read(name);
        assert!(
            script.contains(parent_reference),
            "{name} never looks at its parent directory, so it cannot find the binary in a \
             release archive"
        );
        assert!(
            script.contains("ROOT"),
            "{name} should resolve an archive root rather than assuming its own directory"
        );
    }
}

/// The Windows launcher is a batch file, where a lone `\r` is a carriage
/// return rather than a path separator. `"%HERE%..\rai.exe"` written with a
/// mangled escape becomes `"%HERE%..<CR>ai.exe"`, which silently never matches
/// — exactly how this file was broken once already.
#[test]
fn the_windows_launcher_has_no_stray_carriage_returns_inside_a_line() {
    let bytes = std::fs::read(launcher_dir().join("rai-studio.cmd")).expect("reading launcher");
    for (index, window) in bytes.windows(2).enumerate() {
        if window[0] == b'\r' {
            assert_eq!(
                window[1], b'\n',
                "stray carriage return at byte {index}: a batch file uses CRLF line endings, so a \
                 lone CR is a corrupted escape, not a line break"
            );
        }
    }
    assert!(
        bytes.ends_with(b"\n"),
        "the launcher should end with a newline"
    );
}

/// The launchers name the binary they start. If the CLI is ever renamed, these
/// scripts must move with it rather than silently failing to find anything.
#[test]
fn the_launchers_start_the_binary_this_workspace_builds() {
    assert!(read("rai-studio.cmd").contains("rai.exe"));
    for name in ["rai-studio.sh", "rai-studio.command"] {
        let script = read(name);
        assert!(
            script.contains("/rai\""),
            "{name} should invoke the `rai` binary"
        );
        assert!(
            script.contains("serve"),
            "{name} should start the local server"
        );
    }
}
