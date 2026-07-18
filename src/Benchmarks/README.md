# Benchmarks

BenchmarkDotNet-based benchmarks of FastBertTokenizer. Two suites:

* **`TokenizeSpeed`** measures the different usage patterns of FastBertTokenizer itself
  (single threaded, memory-reusing, batched/multi threaded, batch enumerators). It runs
  every benchmark for the local build as well as for the released NuGet baseline version,
  each on all supported (non-EOL) runtimes. This answers "did my change make it faster?"
  and "is the next release faster than the current one?".
* **`OtherLibs`** compares FastBertTokenizer against other tokenizer libraries available
  for .NET: [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers),
  [Tokenizers.DotNet](https://github.com/sappho192/Tokenizers.DotNet) (bindings for
  Hugging Face's Rust tokenizers) and this repo's own Rust FFI wrapper around
  [tokenizers](https://github.com/huggingface/tokenizers)
  (see [`../HuggingfaceTokenizer`](../HuggingfaceTokenizer)).

All benchmarks tokenize the same corpus - 15,000 articles from simple english wikipedia -
with the same vocabulary (baai-bge-small-en, which uses bert-base-uncased's vocab) and
truncate to 512 tokens per input.

## Running

Prerequisites:

* The .NET SDK version pinned in [`global.json`](../../global.json) plus the .NET 8 runtime.
* A Rust toolchain: the `OtherLibs` suite P/Invokes a native library that needs to be
  built once via `cargo build --release` in [`../HuggingfaceTokenizer/RustLib`](../HuggingfaceTokenizer/RustLib).
* The repo's Git LFS files (the corpus) need to be pulled.

```bash
cd src/HuggingfaceTokenizer/RustLib && cargo build --release && cd ../../Benchmarks

# FastBertTokenizer usage patterns (local build + NuGet baseline, net8.0 + net10.0):
dotnet run -c Release -f net10.0 -- --filter '*TokenizeSpeed*'

# comparison against other libraries:
dotnet run -c Release -f net10.0 -- --filter '*OtherLibs*'

# everything, but with quick short-running jobs (what CI does on PRs):
dotnet run -c Release -f net10.0 -- --smoke --filter '*'

# list all benchmarks:
dotnet run -c Release -f net10.0 -- --list flat
```

`--smoke` is this project's own flag and selects BenchmarkDotNet's short-run jobs (and skips
the NuGet baseline); everything else is passed through to BenchmarkDotNet, see
[its docs](https://benchmarkdotnet.org/articles/guides/console-args.html) for all arguments.
Results are written to `BenchmarkDotNet.Artifacts/`.

The released baseline version measured by `TokenizeSpeed` is defined in
[`BenchmarkConfigs.cs`](BenchmarkConfigs.cs) (`BenchmarkDefaults.BaselineNuGetVersion`) and
must be kept in sync with the `FastBertTokenizer` `PackageReference` in
[`Benchmarks.csproj`](Benchmarks.csproj) and the centrally pinned version in
[`Directory.Packages.props`](../../Directory.Packages.props).

## CI

[`benchmark.yml`](../../.github/workflows/benchmark.yml) runs a quick smoke pass on every
push/PR (to verify the benchmarks and the measured API surface still work) and the full
suite on demand (workflow_dispatch) and monthly. Every run uploads the complete
`BenchmarkDotNet.Artifacts` results as workflow artifacts.

## Related benchmarks

* [`../HuggingfaceTokenizer/BenchPython`](../HuggingfaceTokenizer/BenchPython): cross-language
  comparison of Hugging Face tokenizers (Rust) vs.
  [flash-tokenizer](https://github.com/NLPOptimize/flash-tokenizer) (C++) on the same
  corpus/vocabulary, including an id-level parity check between the two.
* [`../HuggingfaceTokenizer/BenchRust`](../HuggingfaceTokenizer/BenchRust): criterion.rs
  benchmarks of Hugging Face tokenizers without any FFI overhead.
