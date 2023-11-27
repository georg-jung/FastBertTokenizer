# Benchmark Huggingface's `tokenizers` crate

The code in this directory is used to benchmark Huggingface's [`tokenizers`](https://github.com/huggingface/tokenizers/tree/main/tokenizers). Because I don't have much experience with Rust, there are two approaches:

* A naive benchmarking approach in `main.rs` that can be run using `cargo run --release`.
* A proper benchmark using `criterion` that can be run using `cargo bench`.

Criterion benchmarks single threaded as well as batched. The naive approach just uses the batched variant.

## Results

On my machine, criterion outputs values like:

```txt
tokenize/single         time:   [11.920 s 11.964 s 12.008 s]
tokenize/batch          time:   [2.3812 s 2.5472 s 2.7433 s]
```
