# Benchmarks

BenchmarkDotNet-based benchmarks of FastBertTokenizer. Two suites:

* **`TokenizeSpeed`** measures the different usage patterns of FastBertTokenizer itself
  (single threaded, memory-reusing, batched/multi threaded, batch enumerators). It runs
  every benchmark for the local build as well as for the released NuGet baseline version,
  each on all supported (non-EOL) runtimes. This answers "did my change make it faster?"
  and "is the next release faster than the current one?".
* **`OtherLibs`** compares FastBertTokenizer against other tokenizer libraries usable
  from .NET: [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers)
  and [Tokenizers.DotNet](https://github.com/sappho192/Tokenizers.DotNet) (bindings for
  Hugging Face's Rust tokenizers).

Tokenizers that are not natively usable from .NET are benchmarked from their own
ecosystems instead, so interop overhead doesn't skew their numbers:

* [`../HuggingfaceTokenizer/BenchPython`](../HuggingfaceTokenizer/BenchPython) measures
  [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) (Rust core) and
  [flash-tokenizer](https://github.com/NLPOptimize/flash-tokenizer) (C++ core) through
  their Python APIs — the way virtually all of their users consume them. It also contains
  an id-level parity check between the two (`verify.py`).
* [`../HuggingfaceTokenizer/BenchRust`](../HuggingfaceTokenizer/BenchRust) measures
  Hugging Face tokenizers natively via criterion.rs, without any FFI or Python overhead.

All benchmarks tokenize the same corpus - 15,000 articles from simple english wikipedia -
with the same vocabulary (baai-bge-small-en, which uses bert-base-uncased's vocab) and
truncate to 512 tokens per input.

## Running

Prerequisites: the .NET SDK version pinned in [`global.json`](../../global.json) plus the
.NET 8 runtime. The benchmark corpus (`data/wiki-simple.json.br`) ships in the repo and is
available after a normal clone.

```bash
cd src/Benchmarks

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

For the Python harness see [`../HuggingfaceTokenizer/BenchPython`](../HuggingfaceTokenizer/BenchPython):
`pip install -r requirements.txt && python verify.py && python bench.py`.

## CI

[`benchmark.yml`](../../.github/workflows/benchmark.yml) runs a quick smoke pass on every
push/PR (except markdown-only changes; it verifies the benchmarks and the measured API
surface still work) and the full suite on demand (workflow_dispatch) and monthly. Every run
uploads the complete `BenchmarkDotNet.Artifacts` results as workflow artifacts.

## Results

Numbers below are from [this full CI run](https://github.com/georg-jung/FastBertTokenizer/actions/runs/29640464691)
on a shared GitHub Actions runner. They are reproducible by anyone, but noisier than numbers
from dedicated hardware — treat small differences as noise, and note that multi-threaded
results depend on the runner's (few) cores.

```txt
BenchmarkDotNet v0.15.8, Linux Ubuntu 24.04.4 LTS (Noble Numbat)
AMD EPYC 9V74 2.86GHz, 1 CPU, 4 logical and 2 physical cores (GitHub Actions shared runner)
.NET SDK 10.0.302
```

### FastBertTokenizer usage patterns: .NET 8 vs. .NET 10

* Workload: Encode up to 512 tokens from each of the 15,000 articles (3,657,145 tokens produced).
* ~11.8m tokens/s single threaded, ~31.6m tokens/s multi threaded on the runner's 4 vCPUs.
* `local` jobs measure the working tree built from source, `nuget` jobs the released baseline package.

| Method                       | Job                  | Runtime   | Mean     | Error   | StdDev   | Ratio | Allocated    | Alloc Ratio |
|----------------------------- |--------------------- |---------- |---------:|--------:|---------:|------:|-------------:|------------:|
| SinglethreadedAllocating     | local-net10.0        | .NET 10.0 | 310.0 ms | 5.92 ms |  7.28 ms |  1.00 |   2039.61 KB |        1.00 |
| SingleThreadedMemReuse       | local-net10.0        | .NET 10.0 | 286.9 ms | 2.87 ms |  2.54 ms |  0.93 |    996.98 KB |        0.49 |
| MultithreadedMemReuseBatched | local-net10.0        | .NET 10.0 | 115.9 ms | 2.23 ms |  2.98 ms |  0.37 |  13009.46 KB |        6.38 |
| MultithreadedMemReuseAtOnce  | local-net10.0        | .NET 10.0 | 117.0 ms | 1.25 ms |  1.17 ms |  0.38 |    180987 KB |       88.74 |
| ParallelBatchEnumerator      | local-net10.0        | .NET 10.0 | 237.4 ms | 3.42 ms |  3.20 ms |  0.77 |   4797.79 KB |        2.35 |
| BatchEnumerator              | local-net10.0        | .NET 10.0 | 488.1 ms | 8.27 ms |  7.74 ms |  1.58 |   2308.48 KB |        1.13 |
|                              |                      |           |          |         |          |       |              |             |
| SinglethreadedAllocating     | local-net8.0         | .NET 8.0  | 308.5 ms | 3.08 ms |  2.58 ms |  1.00 |   2039.61 KB |        1.00 |
| SingleThreadedMemReuse       | local-net8.0         | .NET 8.0  | 313.4 ms | 6.19 ms | 10.17 ms |  1.02 |    996.98 KB |        0.49 |
| MultithreadedMemReuseBatched | local-net8.0         | .NET 8.0  | 133.0 ms | 2.64 ms |  3.78 ms |  0.43 |  13006.34 KB |        6.38 |
| MultithreadedMemReuseAtOnce  | local-net8.0         | .NET 8.0  | 135.5 ms | 1.20 ms |  1.06 ms |  0.44 | 180986.88 KB |       88.74 |
| ParallelBatchEnumerator      | local-net8.0         | .NET 8.0  | 260.7 ms | 4.43 ms |  4.15 ms |  0.85 |   4802.16 KB |        2.35 |
| BatchEnumerator              | local-net8.0         | .NET 8.0  | 488.2 ms | 3.59 ms |  3.36 ms |  1.58 |   2308.48 KB |        1.13 |
|                              |                      |           |          |         |          |       |              |             |
| SinglethreadedAllocating     | nuget-1.0.28-net10.0 | .NET 10.0 | 292.0 ms | 1.72 ms |  1.61 ms |  1.00 |   2039.61 KB |        1.00 |
| SingleThreadedMemReuse       | nuget-1.0.28-net10.0 | .NET 10.0 | 290.0 ms | 1.26 ms |  1.18 ms |  0.99 |    996.98 KB |        0.49 |
| MultithreadedMemReuseBatched | nuget-1.0.28-net10.0 | .NET 10.0 | 112.7 ms | 2.17 ms |  2.50 ms |  0.39 |     13009 KB |        6.38 |
| MultithreadedMemReuseAtOnce  | nuget-1.0.28-net10.0 | .NET 10.0 | 115.9 ms | 0.98 ms |  0.87 ms |  0.40 | 180987.01 KB |       88.74 |
| ParallelBatchEnumerator      | nuget-1.0.28-net10.0 | .NET 10.0 | 228.3 ms | 1.20 ms |  1.07 ms |  0.78 |   4791.18 KB |        2.35 |
| BatchEnumerator              | nuget-1.0.28-net10.0 | .NET 10.0 | 447.9 ms | 1.45 ms |  1.29 ms |  1.53 |   2308.48 KB |        1.13 |
|                              |                      |           |          |         |          |       |              |             |
| SinglethreadedAllocating     | nuget-1.0.28-net8.0  | .NET 8.0  | 296.2 ms | 1.66 ms |  1.29 ms |  1.00 |   2039.61 KB |        1.00 |
| SingleThreadedMemReuse       | nuget-1.0.28-net8.0  | .NET 8.0  | 296.4 ms | 1.58 ms |  1.32 ms |  1.00 |    996.98 KB |        0.49 |
| MultithreadedMemReuseBatched | nuget-1.0.28-net8.0  | .NET 8.0  | 158.8 ms | 3.15 ms |  3.87 ms |  0.54 |  13006.02 KB |        6.38 |
| MultithreadedMemReuseAtOnce  | nuget-1.0.28-net8.0  | .NET 8.0  | 154.0 ms | 1.08 ms |  0.96 ms |  0.52 | 180986.88 KB |       88.74 |
| ParallelBatchEnumerator      | nuget-1.0.28-net8.0  | .NET 8.0  | 255.9 ms | 4.78 ms |  5.12 ms |  0.86 |   4814.29 KB |        2.36 |
| BatchEnumerator              | nuget-1.0.28-net8.0  | .NET 8.0  | 496.8 ms | 8.29 ms |  7.75 ms |  1.68 |   2308.48 KB |        1.13 |

### vs. other tokenizer libraries for .NET

* [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers)' `BertTokenizer` (v2.0.0)
* [Tokenizers.DotNet](https://github.com/sappho192/Tokenizers.DotNet) (v1.4.1), .NET bindings for Hugging Face's Rust [tokenizers](https://github.com/huggingface/tokenizers)

All single threaded on .NET 10, same environment and run as above:

| Method                | Mean       | Error    | StdDev   | Ratio | Allocated    | Alloc Ratio |
|---------------------- |-----------:|---------:|---------:|------:|-------------:|------------:|
| FastBertTokenizer     |   291.0 ms |  1.60 ms |  1.50 ms |  1.00 |   2039.61 KB |       1.000 |
| MicrosoftMLTokenizers |   829.6 ms |  3.39 ms |  3.17 ms |  2.85 | 106349.56 KB |      52.142 |
| TokenizersDotNet      | 5,246.8 ms | 15.00 ms | 14.03 ms | 18.03 |  14778.24 KB |       7.246 |

Fairness notes: the libraries don't do exactly the same work — FastBertTokenizer emits
input_ids and attention_mask, Microsoft.ML.Tokenizers and Tokenizers.DotNet emit just
input_ids, and Hugging Face tokenizers (behind Tokenizers.DotNet) computes offsets and
more. Tokenizers.DotNet's number includes its per-call .NET↔Rust interop cost, which is
inherent to using it from .NET; for interop-free Hugging Face tokenizers numbers see the
cross-language results below. Correctness also differs: FastBertTokenizer's output is
[continuously tested](../FastBertTokenizer.Tests) to match Hugging Face transformers'
`AutoTokenizer`.

### Cross-language: Hugging Face tokenizers (Rust) and flash-tokenizer (C++)

From the [same CI run](https://github.com/georg-jung/FastBertTokenizer/actions/runs/29640464691)
(Python 3.12, tokenizers 0.23.1, flash-tokenizer 1.2.0), measured from Python — the way
virtually all users of these libraries consume them — tokenizing the full corpus once:

| Benchmark                        | Mean          |
|--------------------------------- |--------------:|
| hf_tokenizers_singlethreaded     | 9.59 s ± 0.08 |
| flash_tokenizer_singlethreaded   | 1.12 s ± 0.01 |
| hf_tokenizers_batch (parallel)   | 4.05 s ± 0.03 |
| flash_tokenizer_batch (parallel) | 786 ms ± 24   |

The single-threaded Hugging Face number includes considerable per-call Python overhead (its
Rust core is much faster, as the batch mode shows — and `BenchRust` measures it without any
Python involved). flash-tokenizer's batch number includes its pure-Python
attention_mask/token_type_ids construction; `flash_tokenizer_batch_ids_only` measures it
without that. An id-level parity check (`verify.py`) shows flash-tokenizer produces ids
identical to Hugging Face tokenizers for 99.6% of the corpus documents. For scale:
FastBertTokenizer tokenizes the same corpus single threaded in ~0.3 s on the same runner.
