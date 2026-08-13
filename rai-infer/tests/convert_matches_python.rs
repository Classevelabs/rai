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
    /// Add a per-head `q_norm` tensor, as Qwen3/Gemma3 carry.
    qk_norm: bool,
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
            qk_norm: false,
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
        for (index, (name, (rows, cols))) in names.iter().zip(dims).enumerate() {
            tensors.push((
                format!("model.layers.{layer}.{name}.weight"),
                vec![rows, cols],
                rng.weights(rows, cols),
            ));
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
        for suffix in ["input_layernorm", "post_attention_layernorm"] {
            let values = rng.norm(HIDDEN);
            let tensor = format!("model.layers.{layer}.{suffix}.weight");
            tensors.push((tensor.clone(), vec![HIDDEN], values.clone()));
            norms.push((tensor, values));
        }
    }
    if spec.qk_norm {
        tensors.push((
            "model.layers.0.self_attn.q_norm.weight".to_string(),
            vec![spec.head_dim],
            vec![1.0f32; spec.head_dim],
        ));
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
            "gemma2",
            Box::new(|config: &mut serde_json::Value| {
                config["model_type"] = serde_json::json!("gemma2");
            }) as Box<dyn Fn(&mut serde_json::Value)>,
            "gemma2",
        ),
        (
            "gemma3",
            Box::new(|config: &mut serde_json::Value| {
                config["model_type"] = serde_json::json!("gemma3_text");
            }),
            "gemma3",
        ),
        (
            "softcap",
            Box::new(|config: &mut serde_json::Value| {
                config["final_logit_softcapping"] = serde_json::json!(30.0);
            }),
            "logit softcapping",
        ),
        (
            "moe",
            Box::new(|config: &mut serde_json::Value| {
                config["num_local_experts"] = serde_json::json!(8);
            }),
            "mixture-of-experts",
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

#[test]
fn per_head_qk_norms_are_still_refused() {
    // Qwen3/Gemma3 add q_norm/k_norm tensors the container has nowhere to put.
    let spec = Spec {
        qk_norm: true,
        ..Spec::default()
    };
    let error = convert_error("qk-norm", &spec);
    assert!(
        error.contains("QK norm"),
        "error should name the QK norms: {error}"
    );
}
