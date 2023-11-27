use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use bench::{load};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenize");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));

    let (tokenizer, articles) = load();

    group.bench_function("single", |b| {
        b.iter_batched(
            || (tokenizer.clone(), articles.iter().map(AsRef::as_ref).collect::<Vec<&str>>()),
            |(t, a)| {
                for text in a {
                    let _tokens = t.encode(text, true).unwrap();
                }
            },
            BatchSize::SmallInput)
    });
    group.bench_function("batch", |b| {
        b.iter_batched(
            || (tokenizer.clone(), articles.iter().map(AsRef::as_ref).collect::<Vec<&str>>()),
            |(t, a)| t.encode_batch(a, true),
            BatchSize::SmallInput)
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
