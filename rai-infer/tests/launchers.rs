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
//!   rai-<version>-<target>/
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
    for (name, parent_reference) in [("rai-studio.cmd", ".."), ("rai-studio.sh", "..")] {
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
    let script = read("rai-studio.sh");
    assert!(
        script.contains("/rai\""),
        "rai-studio.sh should invoke the `rai` binary"
    );
    assert!(
        script.contains("serve"),
        "rai-studio.sh should start the local server"
    );
}

/// The macOS launcher hands over to the Linux one, so a fix to one is a fix
/// to both: the two were near-copies that had already drifted apart.
#[test]
fn the_macos_launcher_runs_the_shared_one() {
    let command = read("rai-studio.command");
    assert!(
        command.contains("exec ") && command.contains("/rai-studio.sh"),
        "rai-studio.command should exec rai-studio.sh beside it:\n{command}"
    );
}

/// The Windows launcher, run from a folder whose name holds the characters
/// batch files trip on: a bare `&` ran a command, a bare `)` ended a block
/// (an archive extracted twice lands in "rai (1)"), and delayed expansion ate
/// `!`. It must find the binary and the model there and hand rai the exact
/// model path. `RAI_NO_BROWSER` keeps the browser probe out of the test.
#[cfg(windows)]
#[test]
fn the_windows_launcher_survives_awkward_folder_names() {
    use std::os::windows::process::CommandExt;
    use std::time::{Duration, Instant};
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let root = std::env::temp_dir().join(format!("rai launcher & (1) !x {}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("launchers")).expect("creating the folder");
    std::fs::copy(env!("CARGO_BIN_EXE_rai"), root.join("rai.exe")).expect("placing rai.exe");
    let model = root.join("model.raimodel");
    std::fs::write(&model, b"not a model").expect("placing a model file");
    let launcher = root.join("launchers").join("rai-studio.cmd");
    std::fs::copy(launcher_dir().join("rai-studio.cmd"), &launcher).expect("placing the launcher");

    let mut child = std::process::Command::new(&launcher)
        .env("RAI_NO_BROWSER", "1")
        .env_remove("RAI_MODEL")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .expect("starting the launcher");
    // The model is not a model, so rai refuses it and the launcher ends; a
    // launcher that hangs instead is a failure, not a slow test.
    let started = Instant::now();
    while child.try_wait().expect("polling the launcher").is_none() {
        if started.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            panic!("the launcher did not finish within a minute");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let output = child
        .wait_with_output()
        .expect("reading the launcher's output");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _ = std::fs::remove_dir_all(&root);

    assert!(
        stdout.contains(&format!("Model:  \"{}\"", model.display())),
        "the launcher did not pick the model by its exact path:\n{stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("rai serve exited with code"),
        "the launcher did not reach `rai serve`:\n{stdout}\n{stderr}"
    );
    assert!(
        !output.status.success(),
        "a refused model must fail the launcher"
    );
}

/// `Exec=` holds no path: an unquoted one broke on the first space, and the
/// directory belongs in `Path=`, a plain string the script runs relative to.
#[test]
fn the_desktop_entry_keeps_the_path_out_of_exec() {
    let entry = read("rai-studio.desktop");
    let exec = entry
        .lines()
        .find(|line| line.starts_with("Exec="))
        .expect("the entry has an Exec line");
    assert!(!exec.contains("@LAUNCHER_DIR@"), "{exec}");
    assert!(entry.lines().any(|line| line == "Path=@LAUNCHER_DIR@"));
}

/// rai serve binds IPv6 loopback on a machine without an IPv4 one, so a
/// launcher that probes only 127.0.0.1 never sees the server come up.
#[test]
fn the_unix_launchers_probe_both_loopback_families() {
    assert!(
        read("rai-studio.sh").contains("for host in 127.0.0.1 ::1"),
        "rai-studio.sh should probe IPv4 and IPv6 loopback"
    );
}

/// A port already taken, so `rai serve` exits at once and the launcher with it.
fn taken_port() -> (std::net::TcpListener, u16) {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("binding a port");
    let port = listener.local_addr().expect("its address").port();
    (listener, port)
}

/// With no model anywhere and no one to answer the download question, the
/// launcher still starts the page, with no model loaded, instead of stopping
/// at "No .raimodel file found". The port is taken, so the server exits at
/// once and the test sees how far the launcher got.
#[cfg(windows)]
#[test]
fn the_windows_launcher_starts_the_page_without_a_model() {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let root = std::env::temp_dir().join(format!("rai launcher empty {}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("launchers")).expect("creating the folder");
    std::fs::copy(env!("CARGO_BIN_EXE_rai"), root.join("rai.exe")).expect("placing rai.exe");
    for name in [
        "rai-studio.cmd",
        "rai-get-model.cmd",
        "rai-get-model.ps1",
        "models.tsv",
    ] {
        std::fs::copy(launcher_dir().join(name), root.join("launchers").join(name))
            .expect("placing the launchers");
    }
    let (_listener, port) = taken_port();
    let output = std::process::Command::new(root.join("launchers").join("rai-studio.cmd"))
        .env("RAI_NO_BROWSER", "1")
        .env("RAI_PORT", port.to_string())
        .env("RAI_HOME", root.join("home"))
        .env_remove("RAI_MODEL")
        .stdin(std::process::Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .expect("running the launcher");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _ = std::fs::remove_dir_all(&root);
    assert!(
        stdout.contains("Model:  none yet"),
        "the launcher stopped before starting the page:\n{stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("rai serve exited with code"),
        "the launcher did not reach `rai serve`:\n{stdout}\n{stderr}"
    );
}

#[cfg(unix)]
#[test]
fn the_unix_launcher_starts_the_page_without_a_model() {
    let root = std::env::temp_dir().join(format!("rai launcher empty {}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("launchers")).expect("creating the folder");
    std::fs::copy(env!("CARGO_BIN_EXE_rai"), root.join("rai")).expect("placing rai");
    for name in ["rai-studio.sh", "rai-get-model", "models.tsv"] {
        std::fs::copy(launcher_dir().join(name), root.join("launchers").join(name))
            .expect("placing the launchers");
    }
    let (_listener, port) = taken_port();
    let output = std::process::Command::new("bash")
        .arg(root.join("launchers").join("rai-studio.sh"))
        .env("RAI_NO_BROWSER", "1")
        .env("RAI_PORT", port.to_string())
        .env("RAI_HOME", root.join("home"))
        .env_remove("RAI_MODEL")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("running the launcher");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _ = std::fs::remove_dir_all(&root);
    assert!(
        stdout.contains("Model:  none yet"),
        "the launcher stopped before starting the page:\n{stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("rai serve exited with code"),
        "the launcher did not reach `rai serve`:\n{stdout}\n{stderr}"
    );
}

/// The starter list rai-get-model reads: every row pinned to one commit and
/// one SHA-256, and every model complete enough to convert. A row typed by
/// hand, or a model missing its tokenizer, would only fail on someone's
/// computer after a multi-gigabyte download.
#[test]
fn every_starter_model_is_pinned_and_complete() {
    use std::collections::BTreeMap;
    let list = read("models.tsv");
    let hex =
        |text: &str, len: usize| text.len() == len && text.bytes().all(|b| b.is_ascii_hexdigit());
    let mut files: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for line in list
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
    {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 8, "a row needs eight fields: {line}");
        let [name, repo, revision, file, bytes, sha256, _args, description] = fields[..] else {
            unreachable!()
        };
        assert!(
            name.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.' || b == b'-'),
            "{name}"
        );
        assert!(repo.contains('/'), "{repo}");
        assert!(
            hex(revision, 40),
            "{name}: revision {revision} is not a commit"
        );
        assert!(hex(sha256, 64), "{name}/{file}: {sha256} is not a SHA-256");
        assert!(
            bytes.parse::<u64>().is_ok_and(|n| n > 0),
            "{name}/{file}: {bytes}"
        );
        assert!(!description.is_empty(), "{name} has no description");
        files
            .entry(name.to_string())
            .or_default()
            .push(file.to_string());
    }
    assert!(
        files.contains_key("qwen2.5-1.5b"),
        "the recommended model is missing"
    );
    for (name, files) in &files {
        for needed in ["config.json", "tokenizer.json"] {
            assert!(files.iter().any(|f| f == needed), "{name} has no {needed}");
        }
        assert!(
            files.iter().any(|f| f.ends_with(".safetensors")),
            "{name} has no weights"
        );
    }
}
