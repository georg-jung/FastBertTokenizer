#[macro_use]
extern crate criterion;

mod common;

// adapted from https://github.com/huggingface/tokenizers/blob/8f9b945c75435d9120b71dfd14364d2571d83c0b/tokenizers/benches/bert_benchmark.rs

use bench::{load};

use criterion::Criterion;
use tokenizers::models::wordpiece::{WordPiece};
use tokenizers::normalizers::{BertNormalizer};
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{decoders, EncodeInput, Model, TokenizerImpl};

use common::{iter_bench_encode, iter_bench_encode_batch};

static BATCH_SIZE: usize = 1_000;

type BertTokenizer = TokenizerImpl<
    WordPiece,
    BertNormalizer,
    BertPreTokenizer,
    BertProcessing,
    decoders::wordpiece::WordPiece,
>;

/// Resembling the BertTokenizer implementation from the Python bindings.
fn create_bert_tokenizer(wp: WordPiece) -> BertTokenizer {
    let sep_id = *wp.get_vocab().get("[SEP]").unwrap();
    let cls_id = *wp.get_vocab().get("[CLS]").unwrap();
    let mut tokenizer = TokenizerImpl::new(wp);
    tokenizer.with_pre_tokenizer(BertPreTokenizer);
    tokenizer.with_normalizer(BertNormalizer::default());
    tokenizer.with_decoder(decoders::wordpiece::WordPiece::default());
    tokenizer.with_post_processor(BertProcessing::new(
        ("[SEP]".to_string(), sep_id),
        ("[CLS]".to_string(), cls_id),
    ));
    tokenizer
}

pub fn bench_bert(c: &mut Criterion) {
    let vocab_path = "../../../data/baai-bge-small-en/vocab.txt";
    let (_we_dont_use_this_tokenizer, articles) = load();

    let wp = WordPiece::from_file(vocab_path)
        .build()
        .unwrap();
    let tokenizer = create_bert_tokenizer(wp);
    let mut lines: Vec<EncodeInput> = vec![];
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    for line in articles {
        let line: EncodeInput = line.into();
        lines.push(line.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }

    c.bench_function("WordPiece BERT encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });

    c.bench_function("WordPiece BERT encode batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });
}

criterion_group! {
    name = bert_benches;
    config = Criterion::default().sample_size(20);
    targets = bench_bert
}

criterion_main!(bert_benches);
