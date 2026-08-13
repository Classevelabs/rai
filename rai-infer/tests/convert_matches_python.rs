//! The Rust converter must produce exactly the bytes the Python exporter does.
//!
//! A `.raimodel` that is merely *valid* is not good enough: the reader would
//! happily load a file whose codes were rounded differently, and the model
//! would just be quietly worse. So this test pins the output byte for byte
//! against `tests/fixtures/tiny-convert.raimodel`, which
//! `scripts/gen_convert_fixture.py` produces from the same weights using
//! `raimodel.py` — the module `export_rtn.py` itself calls.
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

/// Write a synthetic Llama-shaped checkpoint and return the weights the
/// embedding was built from (for the round-trip check).
///
/// With `shards > 1` the tensors are split across `model-0000i-of-0000n
/// .safetensors` files behind a `model.safetensors.index.json`, which is how
/// every checkpoint big enough to need this converter is actually published.
fn write_checkpoint(dir: &Path, with_bias: bool, shards: usize) -> Vec<f32> {
    assert!(shards >= 1);
    std::fs::create_dir_all(dir).expect("creating checkpoint dir");
    let kv_dim = KV_HEADS * HEAD_DIM;
    let dims: [(usize, usize); 7] = [
        (HIDDEN, HIDDEN),
        (kv_dim, HIDDEN),
        (kv_dim, HIDDEN),
        (HIDDEN, HIDDEN),
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

    let embed = rng.weights(VOCAB, HIDDEN);
    tensors.push((
        "model.embed_tokens.weight".to_string(),
        vec![VOCAB, HIDDEN],
        embed.clone(),
    ));

    for layer in 0..LAYERS {
        for (name, (rows, cols)) in names.iter().zip(dims) {
            tensors.push((
                format!("model.layers.{layer}.{name}.weight"),
                vec![rows, cols],
                rng.weights(rows, cols),
            ));
        }
        for suffix in ["input_layernorm", "post_attention_layernorm"] {
            tensors.push((
                format!("model.layers.{layer}.{suffix}.weight"),
                vec![HIDDEN],
                rng.norm(HIDDEN),
            ));
        }
    }
    tensors.push((
        "model.norm.weight".to_string(),
        vec![HIDDEN],
        rng.norm(HIDDEN),
    ));

    if with_bias {
        tensors.push((
            "model.layers.0.self_attn.q_proj.bias".to_string(),
            vec![HIDDEN],
            vec![0.25f32; HIDDEN],
        ));
    }

    if shards == 1 {
        let mut builder = SafeTensorsBuilder::default();
        for (name, shape, values) in &tensors {
            builder.add(name, shape, values);
        }
        builder.write(&dir.join("model.safetensors"));
    } else {
        let per_shard = tensors.len().div_ceil(shards);
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

    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "rope_scaling": serde_json::Value::Null,
        "tie_word_embeddings": true,
        "torch_dtype": "float16",
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .expect("writing config.json");
    // The converter refuses to produce a model with no tokenizer beside it.
    std::fs::write(dir.join("tokenizer.json"), b"{\"version\":\"1.0\"}")
        .expect("writing tokenizer.json");

    embed
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

#[test]
fn output_is_byte_identical_to_the_python_exporter() {
    let root = scratch_dir("identical");
    let model_dir = root.join("checkpoint");
    let output = root.join("out").join("tiny-convert.raimodel");
    let embed = write_checkpoint(&model_dir, false, 1);

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
    assert_eq!(model.config.hidden_size as usize, HIDDEN);
    assert_eq!(model.config.num_layers as usize, LAYERS);
    assert_eq!(model.config.num_heads as usize, HEADS);
    assert_eq!(model.config.num_kv_heads as usize, KV_HEADS);
    assert_eq!(model.config.head_dim as usize, HEAD_DIM);
    assert_eq!(model.config.vocab_size as usize, VOCAB);
    assert_eq!(model.config.max_context, MAX_CONTEXT);
    assert_eq!(model.config.bits, 4);
    assert_eq!(model.config.embed_bits, 8);
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
    let root = scratch_dir("sharded");
    let model_dir = root.join("checkpoint");
    let output = root.join("out").join("sharded.raimodel");
    write_checkpoint(&model_dir, false, 3);
    assert!(model_dir.join("model.safetensors.index.json").is_file());
    assert!(!model_dir.join("model.safetensors").exists());

    convert(&options(&model_dir, &output)).expect("sharded conversion failed");
    assert_matches_golden(&std::fs::read(&output).expect("reading produced model"));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn projection_biases_are_refused() {
    let root = scratch_dir("bias");
    let model_dir = root.join("checkpoint");
    let output = root.join("out").join("biased.raimodel");
    write_checkpoint(&model_dir, true, 1);

    let error = convert(&options(&model_dir, &output))
        .expect_err("a checkpoint with projection biases must not convert")
        .to_string();
    assert!(
        error.contains("bias vectors"),
        "error should name the biases: {error}"
    );
    assert!(
        !output.exists(),
        "nothing may be written when the preflight fails"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_decoupled_head_dim_is_refused() {
    let root = scratch_dir("headdim");
    let model_dir = root.join("checkpoint");
    let output = root.join("out").join("headdim.raimodel");
    write_checkpoint(&model_dir, false, 1);

    // head_dim * num_heads != hidden_size: the format cannot express it.
    let config_path = model_dir.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["head_dim"] = serde_json::json!(32);
    std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

    let error = convert(&options(&model_dir, &output))
        .expect_err("a decoupled head_dim must not convert")
        .to_string();
    assert!(
        error.contains("decoupled head_dim"),
        "error should explain the head_dim: {error}"
    );

    let _ = std::fs::remove_dir_all(&root);
}
