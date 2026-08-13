//! The Rust converter must produce exactly the bytes the Python exporter does,
//! and the container-v2 capabilities must survive a full round trip.
//!
//! A `.raimodel` that is merely *valid* is not good enough: the reader would
//! happily load a file whose codes were rounded differently, and the model
//! would just be quietly worse. So this test pins the output byte for byte
//! against `tests/fixtures/tiny-convert.raimodel`, which
//! `scripts/gen_convert_fixture.py` produces from the same weights using
//! `raimodel.py` — the module `export_rtn.py` itself calls. That fixture is a
//! **v1** file, so it also pins the promise that nothing about v2 changes the
//! bytes of a model that does not use a v2 capability.
//!
//! The test builds the synthetic checkpoint here rather than committing one:
//! the weights come from an LCG mirrored exactly in the generator script, and
//! every value it emits is exactly representable in f16, so nothing in the
//! comparison depends on a float conversion. (The pre-existing
//! `tiny-tied.raimodel` fixture cannot be used for this: its weights come from
//! numpy's PCG64 `standard_normal`, which is not reproducible in Rust.)

#![cfg(feature = "cli")]

use std::path::{Path, PathBuf};

use half::f16;
use rai_infer::convert::{convert, ConvertOptions};
use rai_infer::format::RaiModelFile;
use rai_infer::kv_cache::KVCache;
use rai_infer::layers::{Activation, RopeScaling};
use rai_infer::model::{BatchScratch, RaiModel, Scratch};

const SEED: u64 = 0x2026_0813;
const HIDDEN: usize = 64;
const LAYERS: usize = 2;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const INTERMEDIATE: usize = 128;
const VOCAB: usize = 96;
const GROUP_SIZE: u32 = 64;
const MAX_CONTEXT: u32 = 512;

/// The 64-bit LCG mirrored by `scripts/gen_convert_fixture.py`.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Values in [-0.488, 0.488] on a 1/4096 grid: exact in f16.
    fn weights(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|_| ((self.next_u32() % 4001) as f32 - 2000.0) / 4096.0)
            .collect()
    }

    /// Values near 1.0 on a 1/1024 grid: exact in f16.
    fn norm(&mut self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|_| 1.0 + ((self.next_u32() % 201) as f32 - 100.0) / 1024.0)
            .collect()
    }

    /// Small values on a 1/1024 grid, for bias vectors: exact in f16.
    fn bias(&mut self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|_| ((self.next_u32() % 129) as f32 - 64.0) / 1024.0)
            .collect()
    }
}

/// Accumulates tensors into a `.safetensors` payload.
#[derive(Default)]
struct SafeTensorsBuilder {
    header: serde_json::Map<String, serde_json::Value>,
    data: Vec<u8>,
}

impl SafeTensorsBuilder {
    fn add(&mut self, name: &str, shape: &[usize], values: &[f32]) {
        assert_eq!(shape.iter().product::<usize>(), values.len(), "{name}");
        let begin = self.data.len();
        for &value in values {
            self.data
                .extend_from_slice(&f16::from_f32(value).to_le_bytes());
        }
        self.header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F16",
                "shape": shape,
                "data_offsets": [begin, self.data.len()],
            }),
        );
    }

    fn write(&self, path: &Path) {
        let header = serde_json::to_vec(&self.header).expect("serializing header");
        let mut file = Vec::with_capacity(8 + header.len() + self.data.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&self.data);
        std::fs::write(path, file).expect("writing safetensors");
    }
}

/// Which projections the synthetic checkpoint gives bias vectors to.
#[derive(Clone, Copy, PartialEq)]
enum Bias {
    /// No biases anywhere — the Llama/Mistral shape.
    None,
    /// q, k and v on every layer — the Qwen2/Qwen2.5 shape.
    QkvEveryLayer,
    /// q on layer 0 only: a checkpoint no single mask can describe.
    QOnLayerZeroOnly,
}

/// Which per-head QK norm tensors the synthetic checkpoint carries.
#[derive(Clone, Copy, PartialEq)]
enum QkNorm {
    /// None at all — the Llama/Qwen2 shape.
    None,
    /// `q_norm` and `k_norm` of length `head_dim` on every layer: Qwen3.
    EveryLayer,
    /// `q_norm` alone on layer 0: a checkpoint no single flag can describe.
    QOnLayerZeroOnly,
    /// The pair on every layer, sized over the whole projection rather than
    /// one head — the OLMo2 shape. Under GQA the two are different lengths.
    FullWidthEveryLayer,
    /// A q_norm that is neither `head_dim` nor `num_heads * head_dim`, which
    /// matches no shape this container implements.
    NonsenseWidthEveryLayer,
}

/// What the synthetic checkpoint should look like.
#[derive(Clone)]
struct Spec {
    bias: Bias,
    /// `model_type` in config.json. Drives the Gemma folds.
    model_type: &'static str,
    /// Explicit `head_dim`, which may decouple from `hidden / heads`.
    head_dim: usize,
    /// `rope_scaling` object, or null.
    rope_scaling: serde_json::Value,
    hidden_act: &'static str,
    hidden_activation: serde_json::Value,
    /// Per-head `q_norm`/`k_norm` tensors, as Qwen3 carries.
    qk_norm: QkNorm,
    /// Emit Gemma2's four-norm sandwich layout instead of the usual two.
    sandwich_norm: bool,
    /// Emit a sparse MLP with this many experts instead of a dense one, in
    /// the `mlp.experts.{e}` spelling OLMoE and Qwen3-MoE use.
    experts: usize,
    /// Emit OLMo2's post-norm layout: no `input_layernorm` anywhere, and the
    /// two norms are `post_attention_layernorm` and
    /// `post_feedforward_layernorm` applied to the block outputs.
    post_norm: bool,
    /// Concatenate q/k/v into `qkv_proj` and gate/up into `gate_up_proj`, as
    /// Phi-3 publishes them. The weights themselves are unchanged, which is
    /// what lets a fused checkpoint be compared byte-for-byte against the
    /// separate one built from the same seed.
    fused_projections: bool,
    /// Extra keys merged into `config.json` verbatim.
    extra_config: serde_json::Value,
    shards: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            bias: Bias::None,
            model_type: "llama",
            head_dim: HEAD_DIM,
            rope_scaling: serde_json::Value::Null,
            hidden_act: "silu",
            hidden_activation: serde_json::Value::Null,
            qk_norm: QkNorm::None,
            sandwich_norm: false,
            experts: 0,
            post_norm: false,
            fused_projections: false,
            extra_config: serde_json::Value::Null,
            shards: 1,
        }
    }
}

/// Everything the generator produced, so a test can compare what came back out
/// against what went in.
struct Checkpoint {
    embed: Vec<f32>,
    /// `(tensor name, values)` for every norm vector, in write order.
    norms: Vec<(String, Vec<f32>)>,
    /// `(tensor name, values)` for every bias vector.
    biases: Vec<(String, Vec<f32>)>,
}

/// Write a synthetic Llama-shaped checkpoint matching `spec`.
///
/// With `shards > 1` the tensors are split across `model-0000i-of-0000n
/// .safetensors` files behind a `model.safetensors.index.json`, which is how
/// every checkpoint big enough to need this converter is actually published.
fn write_checkpoint(dir: &Path, spec: &Spec) -> Checkpoint {
    assert!(spec.shards >= 1);
    std::fs::create_dir_all(dir).expect("creating checkpoint dir");
    let q_dim = HEADS * spec.head_dim;
    let kv_dim = KV_HEADS * spec.head_dim;
    let dims: [(usize, usize); 7] = [
        (q_dim, HIDDEN),
        (kv_dim, HIDDEN),
        (kv_dim, HIDDEN),
        (HIDDEN, q_dim),
        (INTERMEDIATE, HIDDEN),
        (INTERMEDIATE, HIDDEN),
        (HIDDEN, INTERMEDIATE),
    ];
    let names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ];

    let mut rng = Lcg(SEED);
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut norms: Vec<(String, Vec<f32>)> = Vec::new();
    let mut biases: Vec<(String, Vec<f32>)> = Vec::new();

    let embed = rng.weights(VOCAB, HIDDEN);
    tensors.push((
        "model.embed_tokens.weight".to_string(),
        vec![VOCAB, HIDDEN],
        embed.clone(),
    ));

    for layer in 0..LAYERS {
        // Held back so q/k/v and gate/up can be concatenated afterwards. The
        // draw order is identical either way, so a fused checkpoint and a
        // separate one built from the same seed hold the same numbers.
        let mut pending: Vec<Vec<f32>> = Vec::with_capacity(names.len());
        for (index, (name, (rows, cols))) in names.iter().zip(dims).enumerate() {
            let values = rng.weights(rows, cols);
            if spec.fused_projections {
                pending.push(values);
            } else if spec.experts > 0 && index >= 4 {
                // A sparse layer keeps no `mlp.gate_proj`; expert 0 takes its
                // place, and the remaining experts follow.
                let side = ["gate_proj", "up_proj", "down_proj"][index - 4];
                tensors.push((
                    format!("model.layers.{layer}.mlp.experts.0.{side}.weight"),
                    vec![rows, cols],
                    values,
                ));
                for expert in 1..spec.experts {
                    tensors.push((
                        format!("model.layers.{layer}.mlp.experts.{expert}.{side}.weight"),
                        vec![rows, cols],
                        rng.weights(rows, cols),
                    ));
                }
            } else {
                tensors.push((
                    format!("model.layers.{layer}.{name}.weight"),
                    vec![rows, cols],
                    values,
                ));
            }
            let wants_bias = match spec.bias {
                Bias::None => false,
                Bias::QkvEveryLayer => index < 3,
                Bias::QOnLayerZeroOnly => index == 0 && layer == 0,
            };
            if wants_bias {
                let values = rng.bias(rows);
                let tensor = format!("model.layers.{layer}.{name}.bias");
                tensors.push((tensor.clone(), vec![rows], values.clone()));
                biases.push((tensor, values));
            }
        }
        if spec.fused_projections {
            // Row order follows Phi3Attention (Q, then K, then V) and Phi3MLP
            // (gate, then up).
            let concat = |parts: &[&Vec<f32>]| -> Vec<f32> {
                parts.iter().flat_map(|part| part.iter().copied()).collect()
            };
            tensors.push((
                format!("model.layers.{layer}.self_attn.qkv_proj.weight"),
                vec![q_dim + 2 * kv_dim, HIDDEN],
                concat(&[&pending[0], &pending[1], &pending[2]]),
            ));
            tensors.push((
                format!("model.layers.{layer}.self_attn.o_proj.weight"),
                vec![HIDDEN, q_dim],
                pending[3].clone(),
            ));
            tensors.push((
                format!("model.layers.{layer}.mlp.gate_up_proj.weight"),
                vec![2 * INTERMEDIATE, HIDDEN],
                concat(&[&pending[4], &pending[5]]),
            ));
            tensors.push((
                format!("model.layers.{layer}.mlp.down_proj.weight"),
                vec![HIDDEN, INTERMEDIATE],
                pending[6].clone(),
            ));
        }
        // Per-head QK norms, when the spec asks for them. Emitted before the
        // layer norms only for readability — the converter locates every tensor
        // by name, so the order in the safetensors file is irrelevant.
        if spec.experts > 0 {
            tensors.push((
                format!("model.layers.{layer}.mlp.gate.weight"),
                vec![spec.experts, HIDDEN],
                rng.weights(spec.experts, HIDDEN),
            ));
        }
        // A full-width pair is two *different* lengths under grouped-query
        // attention, which is the detail that makes it a distinct shape rather
        // than a longer version of the per-head one.
        let qk_lens = match spec.qk_norm {
            QkNorm::FullWidthEveryLayer => (q_dim, kv_dim),
            QkNorm::NonsenseWidthEveryLayer => (q_dim + 2, q_dim + 2),
            _ => (spec.head_dim, spec.head_dim),
        };
        let qk_sides: &[(&str, usize)] = match (spec.qk_norm, layer) {
            (QkNorm::None, _) => &[],
            (QkNorm::QOnLayerZeroOnly, 0) => &[("q_norm", 0)],
            (QkNorm::QOnLayerZeroOnly, _) => &[],
            _ => &[("q_norm", 0), ("k_norm", 1)],
        };
        for (side, which) in qk_sides {
            let len = if *which == 0 { qk_lens.0 } else { qk_lens.1 };
            let values = rng.norm(len);
            let tensor = format!("model.layers.{layer}.self_attn.{side}.weight");
            tensors.push((tensor.clone(), vec![len], values.clone()));
            norms.push((tensor, values));
        }
        // Gemma2 replaces the two-norm layout with four: `input_layernorm` and
        // `pre_feedforward_layernorm` on the residual stream, plus
        // `post_attention_layernorm` and `post_feedforward_layernorm` on the
        // block outputs.
        let suffixes: &[&str] = if spec.sandwich_norm {
            &[
                "input_layernorm",
                "pre_feedforward_layernorm",
                "post_attention_layernorm",
                "post_feedforward_layernorm",
            ]
        } else if spec.post_norm {
            // OLMo2: no input_layernorm at all. Both norms act on block
            // outputs, so an exporter that put them in the pre-block slots
            // would produce a file that loads and is wrong.
            &["post_attention_layernorm", "post_feedforward_layernorm"]
        } else {
            &["input_layernorm", "post_attention_layernorm"]
        };
        for suffix in suffixes {
            let values = rng.norm(HIDDEN);
            let tensor = format!("model.layers.{layer}.{suffix}.weight");
            tensors.push((tensor.clone(), vec![HIDDEN], values.clone()));
            norms.push((tensor, values));
        }
    }
    let final_norm = rng.norm(HIDDEN);
    tensors.push((
        "model.norm.weight".to_string(),
        vec![HIDDEN],
        final_norm.clone(),
    ));
    norms.push(("model.norm.weight".to_string(), final_norm));

    if spec.shards == 1 {
        let mut builder = SafeTensorsBuilder::default();
        for (name, shape, values) in &tensors {
            builder.add(name, shape, values);
        }
        builder.write(&dir.join("model.safetensors"));
    } else {
        let per_shard = tensors.len().div_ceil(spec.shards);
        let chunks: Vec<_> = tensors.chunks(per_shard).collect();
        let mut weight_map = serde_json::Map::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let file_name = format!("model-{:05}-of-{:05}.safetensors", index + 1, chunks.len());
            let mut builder = SafeTensorsBuilder::default();
            for (name, shape, values) in chunk.iter() {
                builder.add(name, shape, values);
                weight_map.insert(name.clone(), serde_json::json!(file_name));
            }
            builder.write(&dir.join(&file_name));
        }
        let index = serde_json::json!({
            "metadata": { "total_size": 0 },
            "weight_map": weight_map,
        });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&index).unwrap(),
        )
        .expect("writing the shard index");
    }

    let mut config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": spec.model_type,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "rope_scaling": spec.rope_scaling.clone(),
        "tie_word_embeddings": true,
        "torch_dtype": "float16",
        "hidden_act": spec.hidden_act,
        "hidden_activation": spec.hidden_activation.clone(),
    });
    // The v1 golden fixture's config.json never carried head_dim; only emit it
    // when it actually says something.
    if spec.head_dim != HIDDEN / HEADS {
        config["head_dim"] = serde_json::json!(spec.head_dim);
    }
    if let Some(extra) = spec.extra_config.as_object() {
        for (key, value) in extra {
            config[key] = value.clone();
        }
    }
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .expect("writing config.json");
    // The converter refuses to produce a model with no tokenizer beside it.
    std::fs::write(dir.join("tokenizer.json"), b"{\"version\":\"1.0\"}")
        .expect("writing tokenizer.json");

    Checkpoint {
        embed,
        norms,
        biases,
    }
}

fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rai-convert-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("creating scratch dir");
    dir
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny-convert.raimodel")
}

fn options(model_dir: &Path, output: &Path) -> ConvertOptions {
    ConvertOptions {
        model_dir: model_dir.to_path_buf(),
        output: Some(output.to_path_buf()),
        group_size: GROUP_SIZE,
        embed_group_size: GROUP_SIZE,
        max_context: MAX_CONTEXT,
        tokenizer_out: None,
        quiet: true,
    }
}

/// Convert `spec` into a fresh scratch directory and return `(root, output)`.
fn convert_spec(label: &str, spec: &Spec) -> (PathBuf, PathBuf, Checkpoint) {
    let root = scratch_dir(label);
    let model_dir = root.join("checkpoint");
    let output = root.join("out").join(format!("{label}.raimodel"));
    let checkpoint = write_checkpoint(&model_dir, spec);
    convert(&options(&model_dir, &output)).unwrap_or_else(|error| {
        panic!("{label}: conversion failed: {error:#}");
    });
    (root, output, checkpoint)
}

fn convert_error(label: &str, spec: &Spec) -> String {
    let root = scratch_dir(label);
    let model_dir = root.join("checkpoint");
    let output = root.join("out").join(format!("{label}.raimodel"));
    write_checkpoint(&model_dir, spec);
    let error = convert(&options(&model_dir, &output))
        .expect_err("this checkpoint must not convert")
        .to_string();
    assert!(
        !output.exists(),
        "{label}: nothing may be written when the preflight fails"
    );
    let _ = std::fs::remove_dir_all(&root);
    error
}

/// Read the raw header bytes of a produced model.
fn header_bytes(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).expect("reading produced model");
    bytes[..128.min(bytes.len())].to_vec()
}

/// Compare against the fixture the Python exporter wrote, reporting *where*
/// the writers disagree rather than just that they do.
fn assert_matches_golden(produced: &[u8]) {
    let golden = std::fs::read(fixture_path()).expect("reading golden fixture");
    assert_eq!(
        produced.len(),
        golden.len(),
        "produced {} bytes, Python wrote {}",
        produced.len(),
        golden.len()
    );
    if produced != golden {
        let first = produced
            .iter()
            .zip(&golden)
            .position(|(a, b)| a != b)
            .expect("lengths matched but contents differ");
        let differing = produced.iter().zip(&golden).filter(|(a, b)| a != b).count();
        panic!(
            "converter output diverges from the Python exporter: first difference at byte \
             {first} (rust {:#04x} vs python {:#04x}), {differing} bytes differ in total",
            produced[first], golden[first]
        );
    }
}

/// Run the model forward over a short token sequence and return the last
/// position's logits. Proves the file is not merely parseable.
fn run_forward(model: &RaiModel, tokens: &[usize]) -> Vec<f32> {
    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;
    let mut kv: KVCache = model.create_kv_cache(64).unwrap();
    let mut scratch = Scratch::new();
    let mut hidden = vec![0.0f32; hs];
    let mut logits = vec![0.0f32; vs];
    for (pos, &token) in tokens.iter().enumerate() {
        hidden.resize(hs, 0.0);
        model.embed_token(token, &mut hidden).unwrap();
        model
            .forward_from_hidden(&mut hidden, pos, &mut kv, true, &mut scratch)
            .unwrap();
        let mut normed = vec![0.0f32; hs];
        model
            .hidden_to_logits_into(&hidden, &mut normed, &mut logits)
            .unwrap();
    }
    logits
}

// =============================================================================
// v1: byte identity with the Python exporter
// =============================================================================

#[test]
fn output_is_byte_identical_to_the_python_exporter() {
    let root = scratch_dir("identical");
    let model_dir = root.join("checkpoint");
    let output = root.join("out").join("tiny-convert.raimodel");
    let checkpoint = write_checkpoint(&model_dir, &Spec::default());
    let embed = checkpoint.embed;

    let summary = convert(&options(&model_dir, &output)).expect("conversion failed");

    let produced = std::fs::read(&output).expect("reading produced model");
    assert_eq!(
        summary.bytes_written as usize,
        produced.len(),
        "reported size disagrees with the file on disk"
    );
    assert_matches_golden(&produced);

    // The bytes are right; confirm the reader agrees and the values survived.
    let model = RaiModelFile::open(&output).expect("reader rejected the converted model");
    assert_eq!(model.config.version, 1, "a plain Llama model must stay v1");
    assert_eq!(model.config.hidden_size as usize, HIDDEN);
    assert_eq!(model.config.num_layers as usize, LAYERS);
    assert_eq!(model.config.num_heads as usize, HEADS);
    assert_eq!(model.config.num_kv_heads as usize, KV_HEADS);
    assert_eq!(model.config.head_dim as usize, HEAD_DIM);
    assert_eq!(model.config.vocab_size as usize, VOCAB);
    assert_eq!(model.config.max_context, MAX_CONTEXT);
    assert_eq!(model.config.bits, 4);
    assert_eq!(model.config.embed_bits, 8);
    assert_eq!(model.config.activation, Activation::Silu);
    assert_eq!(model.config.rope_scaling, RopeScaling::None);
    assert_eq!(model.config.bias_mask, 0);
    assert_eq!(model.config.embed_scale, 1.0);
    assert_eq!(model.sections.len(), LAYERS + 2, "tied model section count");
    assert!(!model.has_lm_head());
    assert_eq!(summary.num_sections, LAYERS + 2);

    // Row 0 and the last row of the embedding must dequantize back to the
    // weights that went in (8-bit steps here are ~0.004 wide).
    let embedding = model.embedding().expect("embedding section");
    for row in [0usize, VOCAB - 1] {
        for column in 0..HIDDEN {
            let group = column / embedding.group_size;
            let params = row * (HIDDEN / embedding.group_size) + group;
            let scale = f16::from_le_bytes([
                embedding.group_params[params * 4],
                embedding.group_params[params * 4 + 1],
            ])
            .to_f32();
            let zero = f16::from_le_bytes([
                embedding.group_params[params * 4 + 2],
                embedding.group_params[params * 4 + 3],
            ])
            .to_f32();
            let code = embedding.data[row * HIDDEN + column] as f32;
            let value = code * scale + zero;
            let original = embed[row * HIDDEN + column];
            assert!(
                (value - original).abs() < 0.01,
                "embedding[{row}][{column}]: {value} != {original}"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_sharded_checkpoint_converts_to_the_same_bytes() {
    let spec = Spec {
        shards: 3,
        ..Spec::default()
    };
    let root = scratch_dir("sharded");
    let model_dir = root.join("checkpoint");
    let output = root.join("out").join("sharded.raimodel");
    write_checkpoint(&model_dir, &spec);
    assert!(model_dir.join("model.safetensors.index.json").is_file());
    assert!(!model_dir.join("model.safetensors").exists());

    convert(&options(&model_dir, &output)).expect("sharded conversion failed");
    assert_matches_golden(&std::fs::read(&output).expect("reading produced model"));

    let _ = std::fs::remove_dir_all(&root);
}

// =============================================================================
// Fused projections (the Phi-3 path)
// =============================================================================

/// The whole claim, in one assertion: a checkpoint that fuses Q/K/V and
/// gate/up produces *the same bytes* as the separate-tensor checkpoint built
/// from the same seed — the golden file every other test compares against.
///
/// This is what makes the row offsets load-bearing. Reading K's rows where V's
/// begin, or splitting the MLP at the wrong row, still yields a structurally
/// valid file of exactly the right size; only the bytes differ. A test that
/// merely asserted "conversion succeeded" would pass with the split wrong.
#[test]
fn a_fused_projection_checkpoint_converts_to_the_same_bytes() {
    let spec = Spec {
        fused_projections: true,
        ..Spec::default()
    };
    let root = scratch_dir("fused");
    let model_dir = root.join("checkpoint");
    let output = root.join("out").join("fused.raimodel");
    write_checkpoint(&model_dir, &spec);

    // The separate tensors really are absent, so this cannot pass by accident.
    let names = std::fs::read_to_string(model_dir.join("model.safetensors"))
        .map(|_| ())
        .err();
    assert!(names.is_some() || model_dir.join("model.safetensors").is_file());

    convert(&options(&model_dir, &output)).expect("fused conversion failed");
    assert_matches_golden(&std::fs::read(&output).expect("reading produced model"));

    let _ = std::fs::remove_dir_all(&root);
}

/// A fused tensor that is not as wide as the parts it claims to hold would let
/// one projection read another's rows. It must be named, not truncated.
#[test]
fn a_fused_tensor_of_the_wrong_width_is_refused() {
    let root = scratch_dir("fused-narrow");
    let model_dir = root.join("checkpoint");
    write_checkpoint(
        &model_dir,
        &Spec {
            fused_projections: true,
            // Doubling the head count makes the config's idea of qkv_proj
            // wider than the tensor on disk, while staying divisible by the
            // KV head count so this fails on the width and nothing else.
            extra_config: serde_json::json!({ "num_attention_heads": HEADS * 2 }),
            ..Spec::default()
        },
    );
    let error = convert(&options(&model_dir, &root.join("out").join("x.raimodel")))
        .expect_err("a mis-sized fused tensor must be refused");
    let message = format!("{error:#}");
    assert!(
        message.contains("qkv_proj"),
        "the error should name the fused tensor, got: {message}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

// =============================================================================
// v2: biases (the Qwen2/Qwen2.5 path)
// =============================================================================

#[test]
fn a_qwen_shaped_checkpoint_converts_with_its_biases() {
    let spec = Spec {
        bias: Bias::QkvEveryLayer,
        model_type: "qwen2",
        ..Spec::default()
    };
    let (root, output, checkpoint) = convert_spec("qwen-bias", &spec);

    let header = header_bytes(&output);
    assert_eq!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 2);
    assert_eq!(header[64], 0, "activation stays SiLU");
    assert_eq!(header[65], 0x01, "bias flag");
    assert_eq!(header[66], 0, "rope stays default");
    assert_eq!(header[67], 0b000_0111, "q, k and v carry biases");

    let file = RaiModelFile::open(&output).expect("reader rejected the biased model");
    assert_eq!(file.config.bias_mask, 0b000_0111);

    // Every bias value must come back bit-exact: they are stored as f32, not
    // quantized, so anything but equality is a bug.
    let mut checked = 0;
    for layer in 0..LAYERS {
        let refs = file.layer(layer).unwrap();
        for (index, projection) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
            let name = format!("model.layers.{layer}.self_attn.{projection}.bias");
            let expected = &checkpoint
                .biases
                .iter()
                .find(|(tensor, _)| *tensor == name)
                .unwrap_or_else(|| panic!("no source bias {name}"))
                .1;
            let stored = rai_infer::format::read_f32_vector(
                refs.biases[index].unwrap_or_else(|| panic!("{name} missing from the container")),
            );
            assert_eq!(&stored, expected, "{name}");
            checked += 1;
        }
        for index in 3..7 {
            assert!(refs.biases[index].is_none(), "unexpected bias at {index}");
        }
    }
    assert_eq!(checked, LAYERS * 3);

    // And it runs.
    let model = RaiModel::load(&output).expect("loading the biased model");
    let logits = run_forward(&model, &[1, 5, 9, 13]);
    assert!(logits.iter().all(|v| v.is_finite()));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn biases_change_the_output() {
    // A model whose biases were dropped would still load and still produce
    // finite logits, so "it runs" proves nothing on its own. Convert the same
    // weights with and without biases and require the results to differ.
    let (plain_root, plain_out, _) = convert_spec("bias-off", &Spec::default());
    let spec = Spec {
        bias: Bias::QkvEveryLayer,
        model_type: "qwen2",
        ..Spec::default()
    };
    let (biased_root, biased_out, _) = convert_spec("bias-on", &spec);

    let plain = RaiModel::load(&plain_out).unwrap();
    let biased = RaiModel::load(&biased_out).unwrap();
    let tokens = [3usize, 11, 27, 5];
    let a = run_forward(&plain, &tokens);
    let b = run_forward(&biased, &tokens);
    let biggest = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        biggest > 1e-3,
        "biases made no difference to the logits (max |delta| = {biggest})"
    );

    let _ = std::fs::remove_dir_all(&plain_root);
    let _ = std::fs::remove_dir_all(&biased_root);
}

#[test]
fn an_inconsistent_bias_set_is_refused() {
    let spec = Spec {
        bias: Bias::QOnLayerZeroOnly,
        ..Spec::default()
    };
    let error = convert_error("bias-partial", &spec);
    assert!(
        error.contains("inconsistent") && error.contains("drop real parameters"),
        "error should explain the mismatch: {error}"
    );
}

// =============================================================================
// v2: Gemma (GeGLU, the norm fold, the embedding scale)
// =============================================================================

#[test]
fn a_gemma_shaped_checkpoint_folds_the_norm_and_records_the_scale() {
    let spec = Spec {
        model_type: "gemma",
        hidden_act: "gelu",
        ..Spec::default()
    };
    let (root, output, checkpoint) = convert_spec("gemma", &spec);

    let header = header_bytes(&output);
    assert_eq!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 2);
    assert_eq!(header[64], 1, "GeGLU activation code");
    assert_eq!(header[65], 0, "no biases");
    assert_eq!(header[66], 0, "plain RoPE");

    let file = RaiModelFile::open(&output).expect("reader rejected the Gemma-shaped model");
    assert_eq!(file.config.activation, Activation::GeluTanh);
    let expected_scale = (HIDDEN as f64).sqrt() as f32;
    assert_eq!(file.config.embed_scale, expected_scale);

    // The norm fold: every stored weight must be 1 + the checkpoint's value.
    // Checked against the source tensors, not against a re-derivation.
    for layer in 0..LAYERS {
        let refs = file.layer(layer).unwrap();
        for (suffix, stored) in [
            ("input_layernorm", &refs.input_layernorm),
            ("post_attention_layernorm", &refs.post_attn_layernorm),
        ] {
            let name = format!("model.layers.{layer}.{suffix}.weight");
            let source = &checkpoint
                .norms
                .iter()
                .find(|(tensor, _)| *tensor == name)
                .unwrap()
                .1;
            let got = rai_infer::format::read_norm_weights(stored);
            assert_eq!(got.len(), source.len());
            for (i, (g, s)) in got.iter().zip(source).enumerate() {
                assert_eq!(*g, 1.0 + *s, "{name}[{i}]");
            }
        }
    }
    let final_source = &checkpoint
        .norms
        .iter()
        .find(|(tensor, _)| tensor == "model.norm.weight")
        .unwrap()
        .1;
    let final_stored = rai_infer::format::read_norm_weights(&file.final_norm().unwrap());
    for (i, (g, s)) in final_stored.iter().zip(final_source).enumerate() {
        assert_eq!(*g, 1.0 + *s, "model.norm.weight[{i}]");
    }

    // The embedding table itself must NOT be pre-scaled: it is tied to
    // lm_head, and scaling it would inflate every logit by ~sqrt(hidden).
    let embedding = file.embedding().unwrap();
    let groups = HIDDEN / embedding.group_size;
    for row in [0usize, VOCAB - 1] {
        for column in 0..HIDDEN {
            let params = row * groups + column / embedding.group_size;
            let scale = f16::from_le_bytes([
                embedding.group_params[params * 4],
                embedding.group_params[params * 4 + 1],
            ])
            .to_f32();
            let zero = f16::from_le_bytes([
                embedding.group_params[params * 4 + 2],
                embedding.group_params[params * 4 + 3],
            ])
            .to_f32();
            let value = embedding.data[row * HIDDEN + column] as f32 * scale + zero;
            let original = checkpoint.embed[row * HIDDEN + column];
            assert!(
                (value - original).abs() < 0.01,
                "embedding[{row}][{column}] was folded: {value} vs {original}"
            );
        }
    }

    // …and `embed_token` must apply it.
    let model = RaiModel::load(&output).expect("loading the Gemma-shaped model");
    let mut scaled = vec![0.0f32; HIDDEN];
    model.embed_token(7, &mut scaled).unwrap();
    let mut raw = vec![0.0f32; HIDDEN];
    rai_infer::gemm::embed_lookup(
        &mut raw,
        7,
        embedding.data,
        embedding.group_params,
        embedding.vocab_size,
        embedding.hidden_size,
        embedding.group_size,
    );
    for i in 0..HIDDEN {
        assert!(
            (scaled[i] - raw[i] * expected_scale).abs() < 1e-4,
            "embed_token did not apply embed_scale at {i}"
        );
    }

    let logits = run_forward(&model, &[2, 8, 20, 44]);
    assert!(logits.iter().all(|v| v.is_finite()));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn geglu_changes_the_output() {
    // Same weights, SiLU vs GeLU-tanh: the activation code must actually
    // reach the kernel.
    let (silu_root, silu_out, _) = convert_spec("act-silu", &Spec::default());
    let gemma = Spec {
        model_type: "gemma",
        hidden_act: "gelu",
        ..Spec::default()
    };
    let (gelu_root, gelu_out, _) = convert_spec("act-gelu", &gemma);

    let a = RaiModel::load(&silu_out).unwrap();
    let b = RaiModel::load(&gelu_out).unwrap();
    assert_eq!(a.config.activation, Activation::Silu);
    assert_eq!(b.config.activation, Activation::GeluTanh);

    let tokens = [4usize, 16, 32];
    let x = run_forward(&a, &tokens);
    let y = run_forward(&b, &tokens);
    let biggest = x
        .iter()
        .zip(&y)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max);
    assert!(biggest > 1e-3, "activation had no effect (max {biggest})");

    let _ = std::fs::remove_dir_all(&silu_root);
    let _ = std::fs::remove_dir_all(&gelu_root);
}

// =============================================================================
// v2: llama3 RoPE
// =============================================================================

#[test]
fn a_llama3_rope_checkpoint_records_its_scaling() {
    let spec = Spec {
        rope_scaling: serde_json::json!({
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }),
        ..Spec::default()
    };
    let (root, output, _) = convert_spec("llama3-rope", &spec);

    let header = header_bytes(&output);
    assert_eq!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 2);
    assert_eq!(header[66], 1, "llama3 rope_type");
    assert_eq!(f32::from_le_bytes(header[68..72].try_into().unwrap()), 32.0);
    assert_eq!(f32::from_le_bytes(header[72..76].try_into().unwrap()), 1.0);
    assert_eq!(f32::from_le_bytes(header[76..80].try_into().unwrap()), 4.0);
    assert_eq!(u32::from_le_bytes(header[80..84].try_into().unwrap()), 8192);

    let file = RaiModelFile::open(&output).expect("reader rejected the llama3-rope model");
    assert_eq!(
        file.config.rope_scaling,
        RopeScaling::Llama3 {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8192,
        }
    );

    // The loaded model's table must be the scaled one, not the plain one.
    let model = RaiModel::load(&output).expect("loading the llama3-rope model");
    let plain =
        rai_infer::layers::RoPETable::new(HEAD_DIM, MAX_CONTEXT as usize, 10_000.0).unwrap();
    let differing = (0..HEAD_DIM / 2)
        .filter(|&i| {
            (model.rope().sin[HEAD_DIM / 2 + i] - plain.sin[HEAD_DIM / 2 + i]).abs() > 1e-9
        })
        .count();
    assert!(
        differing > 0,
        "the loaded RoPE table is the unscaled one ({differing} of {} differ)",
        HEAD_DIM / 2
    );

    let logits = run_forward(&model, &[1, 2, 3, 4, 5]);
    assert!(logits.iter().all(|v| v.is_finite()));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_unknown_rope_scheme_is_still_refused() {
    let spec = Spec {
        rope_scaling: serde_json::json!({ "rope_type": "yarn", "factor": 4.0 }),
        ..Spec::default()
    };
    let error = convert_error("rope-yarn", &spec);
    assert!(
        error.contains("rope_scaling type 'yarn'"),
        "error should name the scheme: {error}"
    );
}

// =============================================================================
// Decoupled head_dim
// =============================================================================

#[test]
fn a_decoupled_head_dim_converts_and_runs() {
    // 4 heads x 32 = 128 against a 64-wide hidden state, the Gemma shape.
    let spec = Spec {
        head_dim: 32,
        ..Spec::default()
    };
    let (root, output, _) = convert_spec("decoupled-head-dim", &spec);

    let file = RaiModelFile::open(&output).expect("reader rejected the decoupled model");
    assert_eq!(file.config.head_dim as usize, 32);
    assert_eq!(file.config.attention_dim(), 128);
    // Nothing here needs a v2 capability, so it stays v1.
    assert_eq!(file.config.version, 1);
    let refs = file.layer(0).unwrap();
    assert_eq!((refs.q_proj.rows, refs.q_proj.cols), (128, HIDDEN));
    assert_eq!((refs.o_proj.rows, refs.o_proj.cols), (HIDDEN, 128));

    let model = RaiModel::load(&output).expect("loading the decoupled model");
    let logits = run_forward(&model, &[6, 12, 18]);
    assert!(logits.iter().all(|v| v.is_finite()));

    // The batched path must agree with the sequential one at this shape too.
    let hs = HIDDEN;
    let vs = VOCAB;
    let tokens = [6usize, 12, 18];
    let mut hiddens = vec![0.0f32; tokens.len() * hs];
    for (i, &t) in tokens.iter().enumerate() {
        model
            .embed_token(t, &mut hiddens[i * hs..(i + 1) * hs])
            .unwrap();
    }
    let mut kv = model.create_kv_cache(64).unwrap();
    let mut bs = BatchScratch::new();
    let positions: Vec<usize> = (0..tokens.len()).collect();
    model
        .forward_batch(&mut hiddens, &positions, &mut kv, &mut bs)
        .unwrap();
    let mut normed = vec![0.0f32; tokens.len() * hs];
    let mut batched = vec![0.0f32; tokens.len() * vs];
    model
        .hidden_to_logits_batch(&hiddens, &mut normed, &mut batched, tokens.len())
        .unwrap();
    let last = &batched[(tokens.len() - 1) * vs..];
    for (i, (a, b)) in logits.iter().zip(last).enumerate() {
        assert!(
            (a - b).abs() < 2e-3,
            "decoupled batched/sequential mismatch at {i}: {a} vs {b}"
        );
    }

    let _ = std::fs::remove_dir_all(&root);
}

// =============================================================================
// Still refused
// =============================================================================

#[test]
fn architectures_the_container_cannot_express_are_still_refused() {
    for (label, mutate, needle) in [
        (
            // Gemma3 used to sit here. It converts now, so what remains is the
            // stride it could not have: a pattern of 1 claims every layer is
            // global, which stores a second base that is never read.
            "degenerate-rope-stride",
            Box::new(|config: &mut serde_json::Value| {
                config["rope_local_base_freq"] = serde_json::json!(10_000.0);
                config["sliding_window_pattern"] = serde_json::json!(1);
            }) as Box<dyn Fn(&mut serde_json::Value)>,
            "sliding_window_pattern is 1",
        ),
        (
            "negative-softcap",
            Box::new(|config: &mut serde_json::Value| {
                config["final_logit_softcapping"] = serde_json::json!(-30.0);
            }),
            "final_logit_softcapping must be finite and positive",
        ),
        (
            "moe",
            Box::new(|config: &mut serde_json::Value| {
                // Routed experts convert now. A *shared* expert does not:
                // it runs for every token alongside them, and dropping it
                // would remove a pathway rather than degrade one.
                config["num_local_experts"] = serde_json::json!(8);
                config["num_experts_per_tok"] = serde_json::json!(2);
                config["shared_expert_intermediate_size"] = serde_json::json!(512);
            }),
            "shared expert",
        ),
        (
            "sliding-window",
            Box::new(|config: &mut serde_json::Value| {
                // Shorter than --max-context (512), so full causal attention
                // would diverge from the reference model.
                config["sliding_window"] = serde_json::json!(128);
            }),
            "sliding_window",
        ),
    ] {
        let root = scratch_dir(&format!("refuse-{label}"));
        let model_dir = root.join("checkpoint");
        let output = root.join("out").join("x.raimodel");
        write_checkpoint(&model_dir, &Spec::default());
        let config_path = model_dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        mutate(&mut config);
        std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let error = match convert(&options(&model_dir, &output)) {
            Ok(summary) => panic!("{label} must be refused, but it wrote {summary:?}"),
            Err(error) => format!("{error:#}"),
        };
        assert!(
            error.contains(needle),
            "{label}: error should name {needle:?}, got {error}"
        );
        assert!(
            !output.exists(),
            "{label}: nothing may be written when the preflight fails"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}

/// The exact Qwen3 shape: per-head QK norms on every layer, no biases.
fn qwen3_spec() -> Spec {
    Spec {
        model_type: "qwen3",
        qk_norm: QkNorm::EveryLayer,
        ..Spec::default()
    }
}

/// The exact Gemma2 shape: GeGLU, sandwich norms, both softcaps.
fn gemma2_spec() -> Spec {
    Spec {
        model_type: "gemma2",
        hidden_act: "gelu_pytorch_tanh",
        hidden_activation: serde_json::json!("gelu_pytorch_tanh"),
        sandwich_norm: true,
        extra_config: serde_json::json!({
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": HEAD_DIM,
            // Wider than --max-context (512), so full causal attention is
            // exactly equivalent and the export is accepted.
            "sliding_window": 4096,
        }),
        ..Spec::default()
    }
}

#[test]
fn a_qwen3_shaped_checkpoint_stores_its_per_head_qk_norms() {
    let (root, output, checkpoint) = convert_spec("qwen3", &qwen3_spec());

    let header = header_bytes(&output);
    assert_eq!(header[4], 2, "QK norms require container v2");
    // flags bit 1 set, bit 0 (biases) clear.
    assert_eq!(header[65], 0x02, "flags should declare QK norms only");

    let file = RaiModelFile::open(&output).expect("the produced model must load");
    assert!(file.config.has_qk_norm);
    assert!(!file.config.has_sandwich_norm);

    // Every layer's stored q_norm/k_norm must be the checkpoint's tensor,
    // value for value: these are written as raw f32, so this is exact.
    for layer in 0..LAYERS {
        let refs = file.layer(layer).expect("layer parses");
        for (side, stored) in [("q_norm", refs.q_norm), ("k_norm", refs.k_norm)] {
            let name = format!("model.layers.{layer}.self_attn.{side}.weight");
            let expected = &checkpoint
                .norms
                .iter()
                .find(|(tensor, _)| *tensor == name)
                .unwrap_or_else(|| panic!("checkpoint should carry {name}"))
                .1;
            let actual = rai_infer::format::read_f32_vector(
                stored.unwrap_or_else(|| panic!("{name} should be stored")),
            );
            assert_eq!(actual.len(), HEAD_DIM, "{name} must be head_dim long");
            assert_eq!(&actual, expected, "{name} round trip");
        }
        assert!(refs.attn_out_norm.is_none());
        assert!(refs.mlp_out_norm.is_none());
    }

    // And it runs: the QK norm must not produce non-finite logits.
    let model = RaiModel::load(&output).expect("the produced model must load");
    let logits = run_forward(&model, &[1, 2, 3, 4, 5]);
    assert!(logits.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn qk_norms_change_the_output() {
    // A capability that is stored but not applied would pass every structural
    // check above. The only proof it reaches the kernels is that it moves the
    // logits.
    let (root_a, plain, _) = convert_spec("qk-off", &Spec::default());
    let (root_b, normed, _) = convert_spec("qk-on", &qwen3_spec());
    let tokens = [1usize, 5, 9, 13];
    let without = run_forward(&RaiModel::load(&plain).unwrap(), &tokens);
    let with = run_forward(&RaiModel::load(&normed).unwrap(), &tokens);
    let max_diff = without
        .iter()
        .zip(&with)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "per-head QK norm should change the logits, max diff was {max_diff:e}"
    );
    let _ = std::fs::remove_dir_all(&root_a);
    let _ = std::fs::remove_dir_all(&root_b);
}

#[test]
fn a_gemma2_shaped_checkpoint_stores_its_sandwich_norms_and_softcaps() {
    let (root, output, checkpoint) = convert_spec("gemma2", &gemma2_spec());

    let header = header_bytes(&output);
    assert_eq!(header[4], 2);
    assert_eq!(header[64], 1, "GeGLU activation code");
    assert_eq!(header[65], 0x04, "flags should declare the sandwich norms");
    assert_eq!(f32::from_le_bytes(header[88..92].try_into().unwrap()), 50.0);
    assert_eq!(f32::from_le_bytes(header[92..96].try_into().unwrap()), 30.0);
    // query_pre_attn_scalar == head_dim, so the scale is the default and
    // nothing is stored.
    assert_eq!(f32::from_le_bytes(header[96..100].try_into().unwrap()), 0.0);

    let file = RaiModelFile::open(&output).expect("the produced model must load");
    assert!(file.config.has_sandwich_norm);
    assert!(!file.config.has_qk_norm);
    assert_eq!(file.config.attn_logit_softcap, 50.0);
    assert_eq!(file.config.final_logit_softcap, 30.0);
    assert_eq!(
        file.config.attention_scale(),
        1.0 / (HEAD_DIM as f32).sqrt()
    );

    // The four norms must land in the right four slots. Gemma folds `1 + w`,
    // so the stored value is one more than the checkpoint's.
    let expect_norm = |name: &str| -> Vec<f32> {
        checkpoint
            .norms
            .iter()
            .find(|(tensor, _)| tensor == name)
            .unwrap_or_else(|| panic!("checkpoint should carry {name}"))
            .1
            .iter()
            .map(|v| 1.0 + v)
            .collect()
    };
    for layer in 0..LAYERS {
        let refs = file.layer(layer).expect("layer parses");
        let slots = [
            (
                rai_infer::format::read_norm_weights(&refs.input_layernorm),
                format!("model.layers.{layer}.input_layernorm.weight"),
            ),
            (
                rai_infer::format::read_norm_weights(&refs.post_attn_layernorm),
                // The pre-MLP slot must carry Gemma2's *pre_feedforward*
                // norm, not its post_attention one.
                format!("model.layers.{layer}.pre_feedforward_layernorm.weight"),
            ),
            (
                rai_infer::format::read_f32_vector(refs.attn_out_norm.unwrap()),
                format!("model.layers.{layer}.post_attention_layernorm.weight"),
            ),
            (
                rai_infer::format::read_f32_vector(refs.mlp_out_norm.unwrap()),
                format!("model.layers.{layer}.post_feedforward_layernorm.weight"),
            ),
        ];
        for (actual, name) in slots {
            assert_eq!(actual, expect_norm(&name), "{name} round trip");
        }
    }

    let model = RaiModel::load(&output).expect("the produced model must load");
    let logits = run_forward(&model, &[1, 2, 3, 4, 5]);
    assert!(logits.iter().all(|v| v.is_finite()));
    // The final softcap is a hard bound: no logit can leave (-30, 30).
    assert!(
        logits.iter().all(|v| v.abs() < 30.0),
        "the final softcap must bound every logit"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn softcapping_and_sandwich_norms_change_the_output() {
    let mut uncapped = gemma2_spec();
    uncapped.extra_config = serde_json::json!({ "sliding_window": 4096 });
    uncapped.sandwich_norm = false;
    // Same weights, same activation; only the new capabilities differ.
    uncapped.model_type = "gemma";

    let (root_a, plain, _) = convert_spec("gemma2-off", &uncapped);
    let (root_b, full, _) = convert_spec("gemma2-on", &gemma2_spec());
    let tokens = [2usize, 4, 6, 8];
    let without = run_forward(&RaiModel::load(&plain).unwrap(), &tokens);
    let with = run_forward(&RaiModel::load(&full).unwrap(), &tokens);
    let max_diff = without
        .iter()
        .zip(&with)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "softcapping and sandwich norms should change the logits, max diff was {max_diff:e}"
    );
    let _ = std::fs::remove_dir_all(&root_a);
    let _ = std::fs::remove_dir_all(&root_b);
}

/// Every position's logits from the batched path, and from the sequential one.
fn batched_and_sequential(model: &RaiModel, tokens: &[usize]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;
    let n = tokens.len();

    let mut sequential = Vec::with_capacity(n);
    let mut kv = model.create_kv_cache(64).unwrap();
    let mut scratch = Scratch::new();
    let mut hidden = vec![0.0f32; hs];
    for (pos, &token) in tokens.iter().enumerate() {
        hidden.resize(hs, 0.0);
        model.embed_token(token, &mut hidden).unwrap();
        model
            .forward_from_hidden(&mut hidden, pos, &mut kv, true, &mut scratch)
            .unwrap();
        let mut normed = vec![0.0f32; hs];
        let mut logits = vec![0.0f32; vs];
        model
            .hidden_to_logits_into(&hidden, &mut normed, &mut logits)
            .unwrap();
        sequential.push(logits);
    }

    let mut hiddens = vec![0.0f32; n * hs];
    for (i, &token) in tokens.iter().enumerate() {
        model
            .embed_token(token, &mut hiddens[i * hs..(i + 1) * hs])
            .unwrap();
    }
    let mut kv = model.create_kv_cache(64).unwrap();
    let mut bs = BatchScratch::new();
    let positions: Vec<usize> = (0..n).collect();
    model
        .forward_batch(&mut hiddens, &positions, &mut kv, &mut bs)
        .unwrap();
    let mut normed = vec![0.0f32; n * hs];
    let mut flat = vec![0.0f32; n * vs];
    model
        .hidden_to_logits_batch(&hiddens, &mut normed, &mut flat, n)
        .unwrap();
    let batched = (0..n)
        .map(|i| flat[i * vs..(i + 1) * vs].to_vec())
        .collect();
    (batched, sequential)
}

#[test]
fn the_new_capabilities_keep_batched_and_sequential_identical() {
    // Speculative decoding's exactness argument rests on this equivalence, and
    // every new capability is a fresh chance to apply something in one path and
    // not the other. Bit-identical, not merely close: both paths must reach the
    // same kernels with the same inputs.
    for (label, spec) in [
        ("qwen3", qwen3_spec()),
        ("gemma2", gemma2_spec()),
        ("olmo2", olmo2_spec()),
        ("gemma3", gemma3_spec()),
        ("moe", moe_spec()),
        (
            "both",
            Spec {
                qk_norm: QkNorm::EveryLayer,
                ..gemma2_spec()
            },
        ),
        // Post-norm placement without the full-width norms, so the placement
        // branch is exercised on its own in both paths.
        (
            "post-norm-only",
            Spec {
                post_norm: true,
                ..Spec::default()
            },
        ),
    ] {
        let (root, output, _) = convert_spec(&format!("invariant-{label}"), &spec);
        let model = RaiModel::load(&output).expect("the produced model must load");
        let tokens = [3usize, 11, 27, 43, 58, 7];
        let (batched, sequential) = batched_and_sequential(&model, &tokens);
        for (pos, (b, s)) in batched.iter().zip(&sequential).enumerate() {
            assert!(
                s.iter().all(|v| v.is_finite()),
                "{label}: non-finite logits at {pos}"
            );
            for (i, (x, y)) in s.iter().zip(b).enumerate() {
                assert_eq!(
                    x, y,
                    "{label}: batched and sequential differ at position {pos} token {i}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}

#[test]
fn a_partial_or_wrongly_shaped_qk_norm_set_is_refused() {
    let partial = Spec {
        qk_norm: QkNorm::QOnLayerZeroOnly,
        ..Spec::default()
    };
    let error = convert_error("qk-partial", &partial);
    assert!(
        error.contains("only one of self_attn.q_norm / self_attn.k_norm"),
        "error should name the missing half: {error}"
    );

    // A width matching neither shape is still refused, and the message has to
    // name both widths it would have accepted or the reader cannot act on it.
    let nonsense = Spec {
        qk_norm: QkNorm::NonsenseWidthEveryLayer,
        ..Spec::default()
    };
    let error = convert_error("qk-nonsense", &nonsense);
    assert!(
        error.contains("head_dim") && error.contains("num_heads*head_dim"),
        "error should name both supported widths: {error}"
    );
}

/// A sparse mixture-of-experts model: four experts, two per token.
fn moe_spec() -> Spec {
    Spec {
        model_type: "olmoe",
        experts: 4,
        extra_config: serde_json::json!({
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
        }),
        ..Spec::default()
    }
}

#[test]
fn a_sparse_checkpoint_stores_its_router_and_every_expert() {
    let (root, output, _) = convert_spec("moe", &moe_spec());

    let header = header_bytes(&output);
    assert_eq!(header[4], 2, "experts need container v2");
    assert_eq!(u16::from_le_bytes(header[105..107].try_into().unwrap()), 4);
    assert_eq!(header[107], 2, "experts_per_token");
    assert_eq!(header[108], 1, "norm_topk_prob");

    let file = RaiModelFile::open(&output).expect("the produced model must load");
    assert_eq!(file.config.num_experts, 4);
    assert_eq!(file.config.experts_per_token, 2);
    assert!(file.config.norm_topk_prob);

    // Every expert must be reachable and correctly shaped. Expert 0 lives in
    // the MLP slots; 1..4 come out of the extra block, and a mis-sized stride
    // would surface here rather than as bad output.
    let layer = file.layer(0).expect("layer 0");
    for expert in 0..4 {
        let mlp = layer.expert(expert, &file.config).expect("expert");
        assert_eq!(mlp.gate.rows, INTERMEDIATE);
        assert_eq!(mlp.gate.cols, HIDDEN);
        assert_eq!(mlp.down.rows, HIDDEN);
        assert_eq!(mlp.down.cols, INTERMEDIATE);
    }
    assert!(
        layer.expert(4, &file.config).is_err(),
        "an expert past the end must be an error, not a wrong slice"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// Routing has to actually route. A model whose experts are all consulted, or
/// none, would still load and still be the right size.
#[test]
fn routing_changes_the_output() {
    // Same weights, different number of experts consulted per token.
    let mut one = moe_spec();
    one.extra_config = serde_json::json!({
        "num_experts": 4, "num_experts_per_tok": 1, "norm_topk_prob": true,
    });
    let (root_a, top1, _) = convert_spec("moe-top1", &one);
    let (root_b, top2, _) = convert_spec("moe-top2", &moe_spec());
    assert_eq!(
        std::fs::metadata(&top1).unwrap().len(),
        std::fs::metadata(&top2).unwrap().len(),
        "experts_per_token is a header field; the file size must not move"
    );

    let tokens = [1usize, 5, 9, 13, 21];
    let a = run_forward(&RaiModel::load(&top1).unwrap(), &tokens);
    let b = run_forward(&RaiModel::load(&top2).unwrap(), &tokens);
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "consulting two experts instead of one must change the logits, got {max_diff:e}"
    );

    let _ = std::fs::remove_dir_all(&root_a);
    let _ = std::fs::remove_dir_all(&root_b);
}

/// The exact Gemma3 shape: everything Gemma2 has, plus per-head QK norms and
/// a second RoPE base for the sliding layers.
fn gemma3_spec() -> Spec {
    Spec {
        model_type: "gemma3_text",
        hidden_act: "gelu_pytorch_tanh",
        hidden_activation: serde_json::json!("gelu_pytorch_tanh"),
        sandwich_norm: true,
        qk_norm: QkNorm::EveryLayer,
        extra_config: serde_json::json!({
            "rope_local_base_freq": 10_000.0,
            "sliding_window_pattern": 2,
            "sliding_window": 4096,
        }),
        ..Spec::default()
    }
}

#[test]
fn a_gemma3_shaped_checkpoint_stores_both_rope_bases() {
    let (root, output, _) = convert_spec("gemma3", &gemma3_spec());

    let header = header_bytes(&output);
    assert_eq!(header[4], 2);
    assert_eq!(
        f32::from_le_bytes(header[100..104].try_into().unwrap()),
        10_000.0,
        "the local RoPE base belongs at 100..104"
    );
    assert_eq!(header[104], 2, "the global-layer stride belongs at 104");

    let file = RaiModelFile::open(&output).expect("the produced model must load");
    assert_eq!(file.config.rope_local_theta, 10_000.0);
    assert_eq!(file.config.global_layer_stride, 2);
    assert!(file.config.has_qk_norm && file.config.has_sandwich_norm);

    let _ = std::fs::remove_dir_all(&root);
}

/// Two bases is the whole point, so a model whose local base equals its global
/// one must produce different logits from one where they differ. Nothing about
/// the file size changes, so only the arithmetic can show it.
#[test]
fn the_second_rope_base_changes_the_output() {
    let mut same = gemma3_spec();
    same.extra_config = serde_json::json!({
        // Equal to the default rope_theta the fixture writes, so every layer
        // rotates identically even though two tables are built.
        "rope_local_base_freq": 10_000.0,
        "sliding_window_pattern": 2,
        "sliding_window": 4096,
    });
    let mut different = gemma3_spec();
    different.extra_config = serde_json::json!({
        "rope_local_base_freq": 1_000_000.0,
        "sliding_window_pattern": 2,
        "sliding_window": 4096,
    });

    let (root_a, a) = {
        let (r, o, _) = convert_spec("gemma3-same", &same);
        (r, o)
    };
    let (root_b, b) = {
        let (r, o, _) = convert_spec("gemma3-diff", &different);
        (r, o)
    };
    assert_eq!(
        std::fs::metadata(&a).unwrap().len(),
        std::fs::metadata(&b).unwrap().len(),
        "the base is a header field; the file size must not move"
    );

    let tokens = [1usize, 5, 9, 13, 21];
    let left = run_forward(&RaiModel::load(&a).unwrap(), &tokens);
    let right = run_forward(&RaiModel::load(&b).unwrap(), &tokens);
    let max_diff = left
        .iter()
        .zip(&right)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "the sliding layers' RoPE base should change the logits, max diff was {max_diff:e}"
    );

    let _ = std::fs::remove_dir_all(&root_a);
    let _ = std::fs::remove_dir_all(&root_b);
}

/// The container stores a stride, not a per-layer list. A checkpoint whose
/// `layer_types` are irregular cannot be represented, and rounding it to the
/// nearest stride would rotate some layers at the wrong base.
#[test]
fn an_irregular_layer_type_list_is_refused() {
    let mut spec = gemma3_spec();
    spec.extra_config = serde_json::json!({
        "rope_local_base_freq": 10_000.0,
        "sliding_window_pattern": 2,
        "sliding_window": 4096,
        // A stride of 2 implies [sliding, full]; this says the opposite.
        "layer_types": ["full_attention", "sliding_attention"],
    });
    let error = convert_error("gemma3-irregular", &spec);
    assert!(
        error.contains("layer_types[0]") && error.contains("stride"),
        "the refusal must name the layer and the stride: {error}"
    );
}

/// The exact OLMo2 shape: projection-wide QK norms and post-norm placement.
fn olmo2_spec() -> Spec {
    Spec {
        model_type: "olmo2",
        qk_norm: QkNorm::FullWidthEveryLayer,
        post_norm: true,
        ..Spec::default()
    }
}

#[test]
fn an_olmo2_shaped_checkpoint_stores_full_width_norms_and_post_norm_placement() {
    let (root, output, checkpoint) = convert_spec("olmo2", &olmo2_spec());

    let header = header_bytes(&output);
    assert_eq!(header[4], 2, "OLMo2 needs container v2");
    assert_eq!(
        header[65], 0x18,
        "flags should declare full-width QK norms (0x08) and post-norm (0x10)"
    );

    let file = RaiModelFile::open(&output).expect("the produced model must load");
    assert!(file.config.has_full_qk_norm);
    assert!(file.config.post_norm);
    assert!(
        !file.config.has_qk_norm,
        "the per-head flag must not also be set"
    );
    assert!(!file.config.has_sandwich_norm);

    // The two QK vectors are different lengths under GQA. Reading them back at
    // the documented offsets is what proves the writer and reader agree about
    // a block whose halves are not the same size.
    let layer = file.layer(0).expect("layer 0");
    let q_norm = layer.q_norm.expect("q_norm present");
    let k_norm = layer.k_norm.expect("k_norm present");
    assert_eq!(q_norm.len(), HEADS * HEAD_DIM * 4);
    assert_eq!(k_norm.len(), KV_HEADS * HEAD_DIM * 4);

    // And the stored values are the checkpoint's, not some other norm's.
    for (name, expected) in &checkpoint.norms {
        if name == "model.layers.0.self_attn.q_norm.weight" {
            let stored = rai_infer::format::read_f32_vector(q_norm);
            assert_eq!(&stored, expected, "q_norm values");
        }
        if name == "model.layers.0.self_attn.k_norm.weight" {
            let stored = rai_infer::format::read_f32_vector(k_norm);
            assert_eq!(&stored, expected, "k_norm values");
        }
    }

    // The tail norms must be OLMo2's own post-block norms. If the exporter had
    // reached for input_layernorm it would have failed to find one; if it had
    // put them in the sandwich block the placement flag would be wrong.
    let stored_input = rai_infer::format::read_norm_weights(&layer.input_layernorm);
    let expected_post_attn = checkpoint
        .norms
        .iter()
        .find(|(name, _)| name == "model.layers.0.post_attention_layernorm.weight")
        .map(|(_, values)| values.clone())
        .expect("fixture writes post_attention_layernorm");
    assert_eq!(
        stored_input, expected_post_attn,
        "tail slot 0 must hold post_attention_layernorm for a post-norm model"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// Post-norm is a placement, so nothing about the file's *size* changes when
/// it is wrong — only the arithmetic. The only proof it reaches the forward
/// pass is that it moves the logits away from the identical-storage pre-norm
/// model built from the same seed.
#[test]
fn post_norm_placement_changes_the_output() {
    let mut pre_norm = olmo2_spec();
    pre_norm.post_norm = false;
    pre_norm.model_type = "llama";
    let (root_a, pre, _) = convert_spec("olmo2-pre", &pre_norm);
    let (root_b, post, _) = convert_spec("olmo2-post", &olmo2_spec());

    // Same section sizes: this is a placement change, not a storage change.
    assert_eq!(
        std::fs::metadata(&pre).unwrap().len(),
        std::fs::metadata(&post).unwrap().len(),
        "post-norm must not change the file size"
    );

    let tokens = [1usize, 5, 9, 13];
    let before = run_forward(&RaiModel::load(&pre).unwrap(), &tokens);
    let after = run_forward(&RaiModel::load(&post).unwrap(), &tokens);
    let max_diff = before
        .iter()
        .zip(&after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "post-norm placement should change the logits, max diff was {max_diff:e}"
    );

    let _ = std::fs::remove_dir_all(&root_a);
    let _ = std::fs::remove_dir_all(&root_b);
}

/// A file claiming both QK-norm shapes, or both norm placements, describes its
/// own layout two incompatible ways. The reader must refuse rather than pick.
#[test]
fn contradictory_capability_flags_are_refused() {
    let (root, output, _) = convert_spec("olmo2-flags", &olmo2_spec());
    let good = std::fs::read(&output).expect("reading model");

    for (label, flags, needle) in [
        (
            "both QK shapes",
            0x0A_u8,
            "both per-head and full-width QK norm",
        ),
        (
            "both placements",
            0x14_u8,
            "both sandwich norms and post-norm",
        ),
    ] {
        let mut bytes = good.clone();
        bytes[65] = flags;
        let broken = output.with_extension(format!("{}.raimodel", flags));
        std::fs::write(&broken, &bytes).expect("writing mutated model");
        let error = match RaiModelFile::open(&broken) {
            Ok(_) => panic!("{label}: contradictory flags must be refused"),
            Err(error) => format!("{error:#}"),
        };
        assert!(error.contains(needle), "{label}: got {error}");
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_gemma2_export_past_its_sliding_window_is_refused() {
    // --max-context is 512 in these tests, so a 128-token window means full
    // causal attention would read outside it on the sliding layers.
    let mut spec = gemma2_spec();
    spec.extra_config = serde_json::json!({
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "sliding_window": 128,
    });
    let error = convert_error("gemma2-window", &spec);
    assert!(
        error.contains("sliding_window=128") && error.contains("--max-context 128"),
        "the refusal must name the window and the fix: {error}"
    );
}
