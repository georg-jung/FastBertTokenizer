# Benchmarks

BenchmarkDotNet-based benchmarks of FastBertTokenizer. Two suites:

* **`TokenizeSpeed`** measures the different usage patterns of FastBertTokenizer itself
  (single threaded, memory-reusing, batched/multi threaded, batch enumerators). It runs
  every benchmark for the local build as well as for the released NuGet baseline version,
  each on all supported (non-EOL) runtimes. This answers "did my change make it faster?"
  and "is the next release faster than the current one?".
* **`OtherLibs`** compares FastBertTokenizer against other tokenizer libraries usable
  from .NET: [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers),
  [Tokenizers.DotNet](https://github.com/sappho192/Tokenizers.DotNet) (bindings for
  Hugging Face's Rust tokenizers) and [BlingFire](https://github.com/microsoft/BlingFire)
  (Microsoft's C++ tokenizer, unmaintained since ~2021 but long the go-to fast BERT
  tokenizer for .NET; it doesn't read vocab.txt — its precompiled
  bert-base-uncased model ships in `data/blingfire/`).

Tokenizers that are not natively usable from .NET are benchmarked from their own
ecosystems instead, so interop overhead doesn't skew their numbers:

* [`../HuggingfaceTokenizer/BenchPython`](../HuggingfaceTokenizer/BenchPython) measures
  [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) (Rust core),
  [flash-tokenizer](https://github.com/NLPOptimize/flash-tokenizer) (C++ core) and
  [tokie](https://github.com/chonkie-inc/tokie) (Rust core) through their Python APIs —
  the way virtually all of their users consume them. It also contains an id-level parity
  check against Hugging Face tokenizers (`verify.py`).
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

Numbers below are from [this full CI run](https://github.com/georg-jung/FastBertTokenizer/actions/runs/34832739276)
on a shared GitHub Actions runner. They are reproducible by anyone, but noisier than numbers
from dedicated hardware — treat small differences as noise, and note that multi-threaded
results depend on the runner's (few) cores.

```txt
BenchmarkDotNet v0.15.8, Linux Ubuntu 24.04.5 LTS (Noble Numbat)
AMD EPYC 7763 2.45GHz, 1 CPU, 4 logical and 2 physical cores (GitHub Actions shared runner)
.NET SDK 10.0.401
```

### FastBertTokenizer usage patterns: .NET 8 vs. .NET 10

* Workload: Encode up to 512 tokens from each of the 15,000 articles (3,657,145 tokens produced).
* ~14.5m tokens/s single threaded, ~36m tokens/s multi threaded on the runner's 4 vCPUs.
* `local` jobs measure the working tree built from source, `nuget` jobs the released baseline package.

| Method                       | Job                  | Runtime   | Mean     | Error   | StdDev  | Ratio | Allocated    | Alloc Ratio |
|----------------------------- |--------------------- |---------- |---------:|--------:|--------:|------:|-------------:|------------:|
| SinglethreadedAllocating     | local-net10.0        | .NET 10.0 | 252.5 ms | 2.04 ms | 1.91 ms |  1.00 |   2039.61 KB |        1.00 |
| SingleThreadedMemReuse       | local-net10.0        | .NET 10.0 | 230.3 ms | 1.53 ms | 1.43 ms |  0.91 |    996.98 KB |        0.49 |
| MultithreadedMemReuseBatched | local-net10.0        | .NET 10.0 | 103.6 ms | 2.06 ms | 3.39 ms |  0.41 |  13009.16 KB |        6.38 |
| MultithreadedMemReuseAtOnce  | local-net10.0        | .NET 10.0 | 101.7 ms | 1.27 ms | 1.18 ms |  0.40 | 180987.01 KB |       88.74 |
| ParallelBatchEnumerator      | local-net10.0        | .NET 10.0 | 202.3 ms | 2.82 ms | 2.50 ms |  0.80 |   4793.70 KB |        2.35 |
| BatchEnumerator              | local-net10.0        | .NET 10.0 | 408.1 ms | 5.17 ms | 4.58 ms |  1.62 |   2308.48 KB |        1.13 |
|                              |                      |           |          |         |         |       |              |             |
| SinglethreadedAllocating     | local-net8.0         | .NET 8.0  | 308.8 ms | 2.81 ms | 2.63 ms |  1.00 |   2039.61 KB |        1.00 |
| SingleThreadedMemReuse       | local-net8.0         | .NET 8.0  | 312.9 ms | 3.92 ms | 3.67 ms |  1.01 |    996.98 KB |        0.49 |
| MultithreadedMemReuseBatched | local-net8.0         | .NET 8.0  | 122.7 ms | 2.16 ms | 1.81 ms |  0.40 |  13006.28 KB |        6.38 |
| MultithreadedMemReuseAtOnce  | local-net8.0         | .NET 8.0  | 125.1 ms | 1.16 ms | 1.09 ms |  0.41 | 180986.88 KB |       88.74 |
| ParallelBatchEnumerator      | local-net8.0         | .NET 8.0  | 240.7 ms | 2.46 ms | 2.30 ms |  0.78 |   4797.59 KB |        2.35 |
| BatchEnumerator              | local-net8.0         | .NET 8.0  | 481.2 ms | 9.34 ms | 9.60 ms |  1.56 |   2308.48 KB |        1.13 |
|                              |                      |           |          |         |         |       |              |             |
| SinglethreadedAllocating     | nuget-1.0.28-net10.0 | .NET 10.0 | 291.1 ms | 3.78 ms | 3.35 ms |  1.00 |   2039.61 KB |        1.00 |
| SingleThreadedMemReuse       | nuget-1.0.28-net10.0 | .NET 10.0 | 279.3 ms | 1.89 ms | 1.77 ms |  0.96 |    996.98 KB |        0.49 |
| MultithreadedMemReuseBatched | nuget-1.0.28-net10.0 | .NET 10.0 | 120.6 ms | 2.30 ms | 1.92 ms |  0.41 |  13006.03 KB |        6.38 |
| MultithreadedMemReuseAtOnce  | nuget-1.0.28-net10.0 | .NET 10.0 | 122.4 ms | 2.37 ms | 2.22 ms |  0.42 | 180987.01 KB |       88.74 |
| ParallelBatchEnumerator      | nuget-1.0.28-net10.0 | .NET 10.0 | 237.8 ms | 2.67 ms | 2.37 ms |  0.82 |   4802.96 KB |        2.35 |
| BatchEnumerator              | nuget-1.0.28-net10.0 | .NET 10.0 | 467.4 ms | 1.93 ms | 1.80 ms |  1.61 |   2308.48 KB |        1.13 |
|                              |                      |           |          |         |         |       |              |             |
| SinglethreadedAllocating     | nuget-1.0.28-net8.0  | .NET 8.0  | 377.2 ms | 3.19 ms | 2.98 ms |  1.00 |   2039.61 KB |        1.00 |
| SingleThreadedMemReuse       | nuget-1.0.28-net8.0  | .NET 8.0  | 283.3 ms | 1.52 ms | 1.35 ms |  0.75 |    996.98 KB |        0.49 |
| MultithreadedMemReuseBatched | nuget-1.0.28-net8.0  | .NET 8.0  | 130.2 ms | 2.58 ms | 3.17 ms |  0.35 |  13006.35 KB |        6.38 |
| MultithreadedMemReuseAtOnce  | nuget-1.0.28-net8.0  | .NET 8.0  | 161.1 ms | 2.64 ms | 2.47 ms |  0.43 | 180987.09 KB |       88.74 |
| ParallelBatchEnumerator      | nuget-1.0.28-net8.0  | .NET 8.0  | 306.9 ms | 1.80 ms | 1.60 ms |  0.81 |   4798.42 KB |        2.35 |
| BatchEnumerator              | nuget-1.0.28-net8.0  | .NET 8.0  | 604.7 ms | 4.16 ms | 4.09 ms |  1.60 |   2308.48 KB |        1.13 |

### vs. other tokenizer libraries for .NET

* [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers)' `BertTokenizer` (v2.0.0)
* [Tokenizers.DotNet](https://github.com/sappho192/Tokenizers.DotNet) (v1.4.1), .NET bindings for Hugging Face's Rust [tokenizers](https://github.com/huggingface/tokenizers)
* [BlingFire](https://github.com/microsoft/BlingFire) (v0.1.8), Microsoft's C++ tokenizer with an official .NET package

All single threaded on .NET 10, same environment and run as above:

| Method                | Mean       | Error    | StdDev   | Ratio | Allocated    | Alloc Ratio |
|---------------------- |-----------:|---------:|---------:|------:|-------------:|------------:|
| FastBertTokenizer     |   265.2 ms |  2.38 ms |  2.22 ms |  1.00 |   2039.61 KB |       1.000 |
| MicrosoftMLTokenizers |   785.3 ms |  1.25 ms |  1.17 ms |  2.96 | 106349.56 KB |      52.142 |
| BlingFire             | 1,218.7 ms |  2.32 ms |  2.05 ms |  4.59 |      0.02 KB |       0.000 |
| TokenizersDotNet      | 4,870.8 ms | 12.25 ms | 10.86 ms | 18.37 |  14778.24 KB |       7.246 |

Fairness notes: the libraries don't do exactly the same work — FastBertTokenizer emits
input_ids and attention_mask, Microsoft.ML.Tokenizers and Tokenizers.DotNet emit just
input_ids, and Hugging Face tokenizers (behind Tokenizers.DotNet) computes offsets and
more. Tokenizers.DotNet's number includes its per-call .NET↔Rust interop cost, which is
inherent to using it from .NET; for Hugging Face tokenizers numbers without .NET interop
see the cross-language results below (Python-driven — a fully native measurement is
possible via `BenchRust`, whose results are not included in these tables). BlingFire does
the least work of all: ids only, without [CLS]/[SEP], and its precompiled model agrees
with Hugging Face on ~99.9% of tokens rather than exactly.
Also note that the Allocated column tracks managed GC allocations only: whatever BlingFire
(C++) and Tokenizers.DotNet's Rust side allocate natively is invisible to BenchmarkDotNet's
MemoryDiagnoser, so the column is only meaningful for the pure-managed libraries.
Correctness also differs: FastBertTokenizer's output is
[continuously tested](../FastBertTokenizer.Tests) to match Hugging Face transformers'
`AutoTokenizer`.

### Cross-language: Hugging Face tokenizers (Rust), flash-tokenizer (C++) and tokie (Rust)

From the [same CI run](https://github.com/georg-jung/FastBertTokenizer/actions/runs/34832739276)
(Python 3.12, tokenizers 0.23.2, flash-tokenizer 1.2.0, tokie 0.1.4), measured from Python — the way
virtually all users of these libraries consume them — tokenizing the full corpus once:

| Benchmark                        | Mean          |
|--------------------------------- |--------------:|
| hf_tokenizers_singlethreaded     | 9.30 s ± 0.11 |
| flash_tokenizer_singlethreaded   | 1.13 s ± 0.04 |
| tokie_sequential_calls           |   513 ms ± 12 |
| hf_tokenizers_batch (parallel)   | 3.93 s ± 0.03 |
| flash_tokenizer_batch (parallel) |   767 ms ± 24 |
| flash_tokenizer_batch_ids_only   |    502 ms ± 5 |
| tokie_batch (parallel)           |    230 ms ± 6 |

The single-threaded Hugging Face number includes per-call Python overhead; the batch mode
amortizes that and additionally parallelizes across documents, so these numbers don't
isolate the Rust core's raw speed (`BenchRust` can measure that natively, but its results
are not included in these tables). flash-tokenizer's batch number includes its pure-Python
attention_mask/token_type_ids construction; `flash_tokenizer_batch_ids_only` measures it
without that. tokie may parallelize internally even for single calls, so read its
`tokie_sequential_calls` number as "sequential API calls", not necessarily "one core".
An id-level parity check (`verify.py`) shows flash-tokenizer produces ids identical to
Hugging Face tokenizers for 99.6% of the corpus documents, while tokie matches exactly.
For scale: FastBertTokenizer tokenizes the same corpus in ~0.25 s single threaded and
~0.1 s multi threaded in the same CI run (tables above).
