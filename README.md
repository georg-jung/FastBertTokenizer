<p align="center" id="toplogo">
  <a href="https://www.nuget.org/packages/FastBertTokenizer/">
    <!-- https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#specifying-the-theme-an-image-is-shown-to -->
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="logo-darkmode.svg">
      <source media="(prefers-color-scheme: light)" srcset="logo.svg">
      <img alt="FastBertTokenizer Logo" src="logo.svg" width="100">
    </picture>
  </a>
</p>

# FastBertTokenizer

[![NuGet version (FastBertTokenizer)](https://img.shields.io/nuget/v/FastBertTokenizer.svg?style=flat)](https://www.nuget.org/packages/FastBertTokenizer/)
[![Docs](https://img.shields.io/badge/Docs-fastberttokenizer.gjung.com-blue)](https://fastberttokenizer.gjung.com/)
![.NET Build](https://github.com/georg-jung/FastBertTokenizer/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/github/georg-jung/FastBertTokenizer/graph/badge.svg?token=PEINHYEBGH)](https://codecov.io/github/georg-jung/FastBertTokenizer)

A fast and memory-efficient library for WordPiece tokenization as it is used by BERT. Tokenization correctness and speed are automatically evaluated in extensive unit tests and benchmarks. Native AOT compatible and support for `netstandard2.0`.

## Goals

* Enabling you to run your AI workloads on .NET in production.
* **Correctness** - Results that are equivalent to [HuggingFace Transformers' `AutoTokenizer`'s](https://huggingface.co/docs/transformers/v4.33.0/en/model_doc/auto#transformers.AutoTokenizer) in all practical cases.
* **Speed** - Tokenization should be as fast as reasonably possible.
* **Ease of use** - The API should be easy to understand and use.

## Getting Started

```bash
dotnet new console
dotnet add package FastBertTokenizer
```

```csharp
using FastBertTokenizer;

var tok = new BertTokenizer();
await tok.LoadFromHuggingFaceAsync("bert-base-uncased");
var (inputIds, attentionMask, tokenTypeIds) = tok.Encode("Lorem ipsum dolor sit amet.");
Console.WriteLine(string.Join(", ", inputIds.ToArray()));
var decoded = tok.Decode(inputIds.Span);
Console.WriteLine(decoded);

// Output:
// 101, 19544, 2213, 12997, 17421, 2079, 10626, 4133, 2572, 3388, 1012, 102
// [CLS] lorem ipsum dolor sit amet. [SEP]
```

[*example project*](src/examples/QuickStart/)

## Comparison to [BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers)

* about 1 order of magnitude faster
* allocates more than 1 order of magnitude less memory
* [better whitespace handling](https://github.com/NMZivkovic/BertTokenizers/issues/24)
* [handles unknown characters correctly](https://github.com/NMZivkovic/BertTokenizers/issues/26)
* [does not throw if text is longer than maximum sequence length](https://github.com/NMZivkovic/BertTokenizers/issues/18)
* handles unicode control chars
* handles other alphabets such as greek and right-to-left languages

Note that while [BERTTokenizers handles token type incorrectly](https://github.com/NMZivkovic/BertTokenizers/issues/18), it does support input of two pieces of text that are tokenized with a separator in between. *FastBertTokenizer* currently does not support this.

## Speed / Benchmarks

> tl;dr: FastBertTokenizer can encode 1 GB of text in around 2 s on a typical notebook CPU from 2020 (measured multi-threaded on a ThinkPad T14s Gen 1, AMD Ryzen 7 PRO 4750U, with v1.x: ~51 ms for the ~26 MB benchmark corpus on .NET 8).

The benchmark suite lives in [`src/Benchmarks`](src/Benchmarks/) — see [its README](src/Benchmarks/README.md) for how to run it yourself. All benchmarks tokenize the same corpus (15,000 articles from simple english wikipedia) with the same vocabulary (baai-bge-small-en, which uses bert-base-uncased's vocab), truncating to 512 tokens per input. They cover

* the different usage patterns of FastBertTokenizer itself — for the local build as well as the latest released NuGet version, each on all supported (non-EOL) runtimes, and
* comparisons against other tokenizer libraries usable from .NET.

[CI](https://github.com/georg-jung/FastBertTokenizer/actions/workflows/benchmark.yml) runs a quick smoke pass on every push/PR (except markdown-only changes) and the full suite on demand and monthly; every run uploads the complete BenchmarkDotNet results as workflow artifacts. The numbers below come from a full run on a shared GitHub Actions runner (`ubuntu-24.04`): anyone can reproduce them, but they are noisier than numbers from dedicated hardware — treat small differences as noise, and note that multi-threaded results depend on the runner's (few) cores.

### FastBertTokenizer usage patterns: .NET 8 vs. .NET 10

* Workload: Encode up to 512 tokens from each of the 15,000 articles (3,657,145 tokens produced).
* Numbers from [this benchmark run](https://github.com/georg-jung/FastBertTokenizer/actions/runs/29640464691): ~11.8m tokens/s single threaded, ~31.6m tokens/s multi threaded on the runner's 4 vCPUs.
* `local` jobs measure this branch built from source, `nuget` jobs the released baseline package.

```txt
BenchmarkDotNet v0.15.8, Linux Ubuntu 24.04.4 LTS (Noble Numbat)
AMD EPYC 9V74 2.86GHz, 1 CPU, 4 logical and 2 physical cores (GitHub Actions shared runner)
.NET SDK 10.0.302
```

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
* this repo's own minimal Rust FFI wrapper around Hugging Face tokenizers (see [`src/HuggingfaceTokenizer`](src/HuggingfaceTokenizer))

Same environment and [benchmark run](https://github.com/georg-jung/FastBertTokenizer/actions/runs/29640464691) as above, all single threaded on .NET 10:

| Method                                       | Mean        | Error    | StdDev   | Ratio | Allocated    | Alloc Ratio |
|--------------------------------------------- |------------:|---------:|---------:|------:|-------------:|------------:|
| FastBertTokenizer                            |    291.0 ms |  1.60 ms |  1.50 ms |  1.00 |   2039.61 KB |       1.000 |
| MicrosoftMLTokenizers                        |    829.6 ms |  3.39 ms |  3.17 ms |  2.85 | 106349.56 KB |      52.142 |
| TokenizersDotNet                             |  5,246.8 ms | 15.00 ms | 14.03 ms | 18.03 |  14778.24 KB |       7.246 |
| RustHuggingfaceWrapperSinglethreadedMemReuse | 10,042.1 ms | 50.69 ms | 47.42 ms | 34.51 |      4.08 KB |       0.002 |

Mind that the compared libraries don't do exactly the same work: FastBertTokenizer emits input_ids and attention_mask, Microsoft.ML.Tokenizers and Tokenizers.DotNet emit just input_ids, and the Hugging Face tokenizers library computes offsets and more. The Rust FFI wrapper additionally pads every input to 512 tokens and crosses the interop boundary once per document, so it is not representative of Hugging Face tokenizers' raw speed — see the cross-language numbers below for a fairer look. Correctness also differs: FastBertTokenizer's output is [continuously tested](src/FastBertTokenizer.Tests) to match Hugging Face transformers' `AutoTokenizer`.

### vs. Hugging Face tokenizers (Rust) and flash-tokenizer (C++)

For a cross-language perspective, [`src/HuggingfaceTokenizer/BenchPython`](src/HuggingfaceTokenizer/BenchPython) benchmarks [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) against [flash-tokenizer](https://github.com/NLPOptimize/flash-tokenizer) from Python on the same corpus and vocabulary, single-threaded and batched. Cross-language numbers are only roughly comparable to the .NET ones (different drivers, process startup, etc.). An id-level parity check (`verify.py`) shows flash-tokenizer produces ids identical to Hugging Face tokenizers for 99.6% of the corpus documents (Hugging Face's fast `AutoTokenizer` is itself backed by the tokenizers library); [`src/HuggingfaceTokenizer/BenchRust`](src/HuggingfaceTokenizer/BenchRust) additionally measures Hugging Face tokenizers natively via criterion.rs, without any FFI or Python overhead.

From the [same benchmark run](https://github.com/georg-jung/FastBertTokenizer/actions/runs/29640464691) (Python 3.12, tokenizers 0.23.1, flash-tokenizer 1.2.0), tokenizing the full corpus once:

| Benchmark                        | Mean          |
|--------------------------------- |--------------:|
| hf_tokenizers_singlethreaded     | 9.59 s ± 0.08 |
| flash_tokenizer_singlethreaded   | 1.12 s ± 0.01 |
| hf_tokenizers_batch (parallel)   | 4.05 s ± 0.03 |
| flash_tokenizer_batch (parallel) | 786 ms ± 24   |

The single-threaded Hugging Face number includes considerable per-call Python overhead (its Rust core is much faster, as the batch mode shows); flash-tokenizer's batch number includes its pure-Python attention_mask/token_type_ids construction. For scale: FastBertTokenizer tokenizes the same corpus single threaded in ~0.3 s on the same runner.

## Logo

Created by combining <https://icons.getbootstrap.com/icons/cursor-text/> in .NET brand color with <https://icons.getbootstrap.com/icons/braces/>.
