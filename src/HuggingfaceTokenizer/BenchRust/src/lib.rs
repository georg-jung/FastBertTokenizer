use std::fs::File;
use std::io::Read;
use std::path::Path;
use tokenizers::Tokenizer;
use brotli::Decompressor;
use serde_json::Value;

#[inline]
pub fn load_wikidata_from_brotli_json(file_path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let file = File::open(Path::new(file_path))?;
    let mut decompressed = Vec::new();
    let mut brotli_decompressor = Decompressor::new(file, 4096);

    brotli_decompressor.read_to_end(&mut decompressed)?;

    let articles: Value = serde_json::from_slice(&decompressed)?;
    let mut contents: Vec<String> = Vec::new();

    if let Value::Object(map) = articles {
        for (_, article) in map {
            if let Value::String(text) = article {
                contents.push(text);
            }
        }
    }

    Ok(contents)
}

#[inline]
pub fn load_tokenizer(tokenizer_json_path: &str, pad_to: usize) -> tokenizers::tokenizer::Result<Tokenizer> {
    // Load the tokenizer
    let mut tokenizer = Tokenizer::from_file(tokenizer_json_path)?;
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(pad_to),
        ..Default::default()
    }));

    Ok(tokenizer)
}

#[inline]
pub fn load() -> (Tokenizer, Vec<String>) {
    let tokenizer_path = "../../../data/baai-bge-small-en-tokenizer.json";
    let wikidata_path = "../../../data/wiki-simple.json.br";

    let articles = load_wikidata_from_brotli_json(wikidata_path).unwrap();
    println!("Loaded {} articles.", articles.len());

    let tokenizer = load_tokenizer(tokenizer_path, 512).unwrap();
    println!("Loaded tokenizer with vocab size {}.", tokenizer.get_vocab_size(true));

    (tokenizer, articles)
}
