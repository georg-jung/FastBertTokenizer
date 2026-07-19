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

The benchmark suite lives in [`src/Benchmarks`](src/Benchmarks/) and covers the different usage patterns of FastBertTokenizer (local build and released NuGet baseline, on all supported runtimes) as well as comparisons against other tokenizer libraries usable from .NET; [`src/HuggingfaceTokenizer`](src/HuggingfaceTokenizer) additionally benchmarks Hugging Face tokenizers (Rust) and [flash-tokenizer](https://github.com/NLPOptimize/flash-tokenizer) (C++) from their native ecosystems, so no interop overhead skews their numbers. [CI](https://github.com/georg-jung/FastBertTokenizer/actions/workflows/benchmark.yml) runs a quick smoke pass on every push/PR (except markdown-only changes) and the full suite on demand and monthly, uploading complete results as workflow artifacts.

Headline results from [a full CI run](https://github.com/georg-jung/FastBertTokenizer/actions/runs/29640464691) on a shared GitHub Actions runner (`ubuntu-24.04`, 4 vCPUs), tokenizing 15,000 simple english wikipedia articles (3,657,145 tokens) with bert-base-uncased's vocabulary, truncated to 512 tokens per input:

* **FastBertTokenizer: ~11.8m tokens/s single threaded, ~31.6m tokens/s multi threaded** on the runner's 4 vCPUs.
* Single threaded on .NET 10, same corpus: FastBertTokenizer **291 ms** — Microsoft.ML.Tokenizers **830 ms** (2.9x, 52x allocations) — Tokenizers.DotNet (Hugging Face tokenizers .NET bindings) **5.2 s** (18x).
* Cross-language, measured from Python without .NET involved: Hugging Face tokenizers **9.6 s** single threaded / **4.1 s** batch-parallel; flash-tokenizer **1.1 s** / **0.8 s**. FastBertTokenizer needs ~0.3 s single threaded for the same corpus.

All tables, the exact environment, and fairness notes (the libraries don't all do exactly the same work) are in [src/Benchmarks/README.md](src/Benchmarks/README.md#results). Shared-runner numbers are reproducible by anyone but noisier than dedicated hardware — treat small differences as noise.

## Logo

Created by combining <https://icons.getbootstrap.com/icons/cursor-text/> in .NET brand color with <https://icons.getbootstrap.com/icons/braces/>.
