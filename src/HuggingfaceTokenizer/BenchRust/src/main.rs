use std::fs;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() {
    // Specify the paths
    let tokenizer_path = "C:/Users/georg/git/bge-small-en/tokenizer.json";
    let directory_path = "C:/Users/georg/simplewikicorpus_more";

    // Load the tokenizer
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(0),
        ..Default::default()
    }));

    // Read all text files from the specified directory
    let entries = fs::read_dir(directory_path).expect("Failed to read directory");

    let mut all_texts = Vec::new();
    for entry in entries {
        let entry = entry.expect("Failed to read entry");
        if entry.path().extension().unwrap_or_default() == "txt" {
            let content = fs::read_to_string(entry.path()).expect("Failed to read text file");
            all_texts.push(content);
        }
    }

    // warmup?
    /*
    for text in &all_texts {
        let _tokens = tokenizer.encode(text.as_str(), true).expect("Failed to tokenize text");
    }
    for text in &all_texts {
        let _tokens = tokenizer.encode(text.as_str(), true).expect("Failed to tokenize text");
    }

    // Benchmark the tokenization process
    let start = Instant::now();
    for text in &all_texts {
        let _tokens = tokenizer.encode(text.as_str(), true).expect("Failed to tokenize text");
    }    
    let duration = start.elapsed();

    println!("Tokenized {} text files in {:?}", all_texts.len(), duration);

    // batched

    let batch_size = 100; // Set an appropriate batch size for your needs

    let start = Instant::now();

    for batch in all_texts.chunks(batch_size) {
        let batch_str: Vec<&str> = batch.iter().map(AsRef::as_ref).collect();
        let _tokens = tokenizer.encode_batch(batch_str, true).expect("Failed to tokenize batch");
    }

    let duration = start.elapsed();
    println!("Tokenized {} text files in batches of {} in {:?}", all_texts.len(), batch_size, duration);
    */
    
    // batched2
    /*let batch_str: Vec<&str> = all_texts.iter().map(AsRef::as_ref).collect();
    let encodings = tokenizer.encode_batch(batch_str, true).expect("Failed to tokenize batch");*/
    let batch_str: Vec<&str> = all_texts.iter().map(AsRef::as_ref).collect();
    let encodings = tokenizer.encode_batch(batch_str, true).expect("Failed to tokenize batch");
    let batch_str: Vec<&str> = all_texts.iter().map(AsRef::as_ref).collect();
    let start = Instant::now();    
    let encodings = tokenizer.encode_batch(batch_str, true).expect("Failed to tokenize batch");
    let duration = start.elapsed();

    // Calculate the total number of token IDs produced
    let total_token_ids = encodings.iter().map(|e| e.get_ids().len()).sum::<usize>();

    println!("Tokenization took: {:?}", duration);
    println!("Total number of text segments in the batch: {}", encodings.len());
    println!("Total number of token IDs produced: {}", total_token_ids);
}

