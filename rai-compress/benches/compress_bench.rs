use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use rai_compress::{compress, decompress_matrix, RCConfig};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_structured_weights(rows: usize, cols: usize, rank: usize) -> DMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(0x5EED_BE7C);
    let u = DMatrix::from_fn(rows, rank, |i, j| (i as f64 * 0.01 + j as f64 * 0.1).sin());
    let s = DMatrix::from_diagonal(&nalgebra::DVector::from_fn(rank, |i, _| {
        10.0 / (i as f64 + 1.0)
    }));
    let v = DMatrix::from_fn(cols, rank, |i, j| (i as f64 * 0.02 + j as f64 * 0.07).cos());
    let clean = &u * s * v.transpose();
    let noise = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-0.01..0.01));
    clean + noise
}

fn bench_compress(c: &mut Criterion) {
    let weights = make_structured_weights(256, 256, 8);
    let config = RCConfig::default();

    c.bench_function("rc_compress_256x256", |b| {
        b.iter(|| compress(&weights, &config).unwrap())
    });
}

fn bench_decompress(c: &mut Criterion) {
    let weights = make_structured_weights(256, 256, 8);
    let config = RCConfig::default();
    let compressed = compress(&weights, &config).unwrap();

    c.bench_function("rc_decompress_256x256", |b| {
        b.iter(|| decompress_matrix(&compressed).unwrap())
    });
}

criterion_group!(benches, bench_compress, bench_decompress);
criterion_main!(benches);
