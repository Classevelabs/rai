//! The container's capacity limits exist once per language, and this test is
//! what keeps the second copy honest.
//!
//! Eight limits decide which `.raimodel` files exist: seven container bounds
//! and the GEMM group budget the kernels are compiled around. In Rust they are
//! defined at their point of enforcement — `format.rs` for the container,
//! `gemm.rs` for the group budget, `layers.rs` for the RoPE table — and
//! `convert.rs` imports them, so the writer cannot believe a different limit
//! than the reader enforces.
//!
//! `scripts/raimodel.py` is the reference exporter and cannot import Rust, so
//! it declares the same eight values as Python literals. That is a second copy
//! by necessity, not by choice. Before this test existed there was no gate on
//! it at all: the three sets were kept equal by hand, and nothing failed if
//! they stopped being equal.
//!
//! Drift is not hypothetical damage. If the exporter's limit is looser than
//! the reader's, it writes a file `rai run` refuses — after however many
//! minutes the quantization took. If it is tighter, it refuses checkpoints RAI
//! can actually run. Either way the failure surfaces far from the edit that
//! caused it, which is exactly the failure this test converts into a red
//! build.

use std::path::PathBuf;

/// Read one `NAME = <integer expression>` assignment out of the exporter.
///
/// Deliberately tiny: the only forms the constant block uses are a decimal
/// literal with optional `_` separators and a product of them
/// (`512 * 1024 * 1024`). Anything else is reported rather than guessed at, so
/// a future edit that writes the limit in a form this cannot read fails the
/// test instead of silently skipping the comparison.
fn python_limit(source: &str, name: &str) -> u64 {
    let line = source
        .lines()
        .map(str::trim)
        .find(|line| {
            line.starts_with(name)
                && line[name.len()..].trim_start().starts_with('=')
                // `MAX_HEADS` must not match `MAX_HEADS_SOMETHING`.
                && !line[name.len()..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
        })
        .unwrap_or_else(|| panic!("scripts/raimodel.py does not define {name}"));

    let value = line
        .split_once('=')
        .expect("assignment has an =")
        .1
        .split('#')
        .next()
        .expect("split always yields one part")
        .trim();

    value
        .split('*')
        .map(|factor| {
            let factor = factor.trim().replace('_', "");
            factor.parse::<u64>().unwrap_or_else(|_| {
                panic!("{name} in scripts/raimodel.py is `{value}`, which this test cannot read; \
                        write it as decimal literals joined by `*`, or teach this parser the new form")
            })
        })
        .product()
}

fn exporter_source() -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "scripts", "raimodel.py"]
        .iter()
        .collect();
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

#[test]
fn the_python_exporter_agrees_with_the_rust_limits() {
    let source = exporter_source();

    // (name in both languages, the Rust value)
    let expected: [(&str, u64); 8] = [
        (
            "MAX_HIDDEN_SIZE",
            u64::from(rai_infer::format::MAX_HIDDEN_SIZE),
        ),
        (
            "MAX_INTERMEDIATE_SIZE",
            u64::from(rai_infer::format::MAX_INTERMEDIATE_SIZE),
        ),
        ("MAX_LAYERS", u64::from(rai_infer::format::MAX_LAYERS)),
        ("MAX_HEADS", u64::from(rai_infer::format::MAX_HEADS)),
        (
            "MAX_VOCAB_SIZE",
            u64::from(rai_infer::format::MAX_VOCAB_SIZE),
        ),
        ("MAX_CONTEXT", u64::from(rai_infer::format::MAX_CONTEXT)),
        ("MAX_GEMM_GROUPS", rai_infer::gemm::MAX_GROUPS as u64),
        (
            "MAX_ROPE_TABLE_BYTES",
            rai_infer::layers::MAX_ROPE_TABLE_BYTES as u64,
        ),
    ];

    let mut disagreements = Vec::new();
    for (name, rust_value) in expected {
        let python_value = python_limit(&source, name);
        if python_value != rust_value {
            disagreements.push(format!(
                "  {name}: Rust says {rust_value}, scripts/raimodel.py says {python_value}"
            ));
        }
    }

    assert!(
        disagreements.is_empty(),
        "the Python exporter and the Rust reader disagree about what a .raimodel may contain:\n{}\n\
         \nThe Rust value is authoritative — it is enforced at load time by the code that runs \
         the model. Update scripts/raimodel.py to match.",
        disagreements.join("\n")
    );
}

/// The parser itself has to be trustworthy: a silent mis-read would make the
/// test above pass while the values disagreed.
#[test]
fn the_limit_parser_reads_both_forms_and_refuses_the_rest() {
    let source = "\
MAX_A = 65_536
MAX_B = 512 * 1024 * 1024
MAX_A_SUFFIXED = 1
MAX_C = 128  # trailing comment
";
    assert_eq!(python_limit(source, "MAX_A"), 65_536);
    assert_eq!(python_limit(source, "MAX_B"), 536_870_912);
    assert_eq!(python_limit(source, "MAX_C"), 128);
    // A prefix of a longer name must not be matched by it.
    assert_eq!(python_limit(source, "MAX_A_SUFFIXED"), 1);
}

/// `convert.rs` no longer declares its own copies; this proves the writer and
/// the reader are looking at the same numbers rather than at two that happen
/// to agree today.
#[test]
fn the_writer_and_the_reader_share_one_definition() {
    let convert = include_str!("../src/convert.rs");
    for name in [
        "MAX_HIDDEN_SIZE",
        "MAX_INTERMEDIATE_SIZE",
        "MAX_LAYERS",
        "MAX_HEADS",
        "MAX_VOCAB_SIZE",
        "MAX_CONTEXT",
        "MAX_GEMM_GROUPS",
        "MAX_ROPE_TABLE_BYTES",
    ] {
        assert!(
            !convert.contains(&format!("const {name}:")),
            "convert.rs declares its own {name}; import it from the module that enforces it \
             instead, or this whole gate is decorative"
        );
    }
}
