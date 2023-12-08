// copy of https://github.com/huggingface/tokenizers/blob/8f9b945c75435d9120b71dfd14364d2571d83c0b/tokenizers/benches/common/mod.rs

use std::time::{Duration, Instant};

use criterion::black_box;

use tokenizers::{
    Decoder, EncodeInput, Model, Normalizer, PostProcessor, PreTokenizer, TokenizerImpl,
};

pub fn iter_bench_encode<M, N, PT, PP, D>(
    iters: u64,
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    lines: &[EncodeInput],
) -> Duration
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        for l in lines.iter() {
            let input = l.clone();
            let start = Instant::now();
            let _ = black_box(tokenizer.encode(input, false));
            duration = duration.checked_add(start.elapsed()).unwrap();
        }
    }
    duration
}

pub fn iter_bench_encode_batch<M, N, PT, PP, D>(
    iters: u64,
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    batches: &[Vec<EncodeInput>],
) -> Duration
where
    M: Model + Send + Sync,
    N: Normalizer + Send + Sync,
    PT: PreTokenizer + Send + Sync,
    PP: PostProcessor + Send + Sync,
    D: Decoder + Send + Sync,
{
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        for b in batches.iter() {
            let batch = b.clone();
            let start = Instant::now();
            let _ = black_box(tokenizer.encode_batch(batch, false));
            duration = duration.checked_add(start.elapsed()).unwrap();
        }
    }
    duration
}
