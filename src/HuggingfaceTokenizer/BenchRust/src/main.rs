use std::time::Instant;
use bench::{load};

fn main() {
    let (tokenizer, articles) = load();

    let max = 5;
    for i in 0..max {
        let batch_str: Vec<&str> = articles.iter().map(AsRef::as_ref).collect();
        let start = Instant::now();
        let encodings = tokenizer.encode_batch(batch_str, true).expect("Failed to tokenize batch");
        let duration = start.elapsed();

        println!("Iteration {:?} took: {:?}", i, duration);
        if i == max - 1 {
            // Calculate the total number of token IDs produced
            let total_token_ids = encodings.iter().map(|e| e.get_ids().len()).sum::<usize>();

            println!("Tokenization took: {:?}", duration);
            println!("Total number of text segments in the batch: {}", encodings.len());
            println!("Total number of token IDs produced: {}", total_token_ids);
        }
    }
}

