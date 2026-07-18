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

> tl;dr: FastBertTokenizer can encode 1 GB of text in around 2 s on a typical notebook CPU from 2020.

The benchmark suite lives in [`src/Benchmarks`](src/Benchmarks/) — see [its README](src/Benchmarks/README.md) for how to run it yourself. All benchmarks tokenize the same corpus (15,000 articles from simple english wikipedia) with the same vocabulary (baai-bge-small-en, which uses bert-base-uncased's vocab), truncating to 512 tokens per input. They cover

* the different usage patterns of FastBertTokenizer itself — for the local build as well as the latest released NuGet version, each on all supported (non-EOL) runtimes, and
* comparisons against other tokenizer libraries usable from .NET.

[CI](https://github.com/georg-jung/FastBertTokenizer/actions/workflows/benchmark.yml) runs a quick smoke pass on every push/PR and the full suite on demand and monthly; every run uploads the complete BenchmarkDotNet results as workflow artifacts. The numbers below come from a full run on a shared GitHub Actions runner (`ubuntu-24.04`): anyone can reproduce them, but they are noisier than numbers from dedicated hardware — treat small differences as noise, and note that multi-threaded results depend on the runner's (few) cores.

### FastBertTokenizer usage patterns: .NET 8 vs. .NET 10

* Workload: Encode up to 512 tokens from each of the 15,000 articles (≈3.66m tokens produced).

*(Table pending — will be filled in from the first full CI run of this benchmark setup.)*

### vs. other tokenizer libraries for .NET

* [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers)' `BertTokenizer` (v2.0.0)
* [Tokenizers.DotNet](https://github.com/sappho192/Tokenizers.DotNet) (v1.4.1), .NET bindings for Hugging Face's Rust [tokenizers](https://github.com/huggingface/tokenizers)
* this repo's own minimal Rust FFI wrapper around Hugging Face tokenizers (see [`src/HuggingfaceTokenizer`](src/HuggingfaceTokenizer))

*(Table pending — will be filled in from the first full CI run of this benchmark setup.)*

Mind that the compared libraries don't do exactly the same work: FastBertTokenizer emits input_ids and attention_mask, Microsoft.ML.Tokenizers and Tokenizers.DotNet emit just input_ids, and the Hugging Face tokenizers library computes offsets and more. Correctness also differs: FastBertTokenizer's output is [continuously tested](src/FastBertTokenizer.Tests) to match Hugging Face transformers' `AutoTokenizer`.

### vs. Hugging Face tokenizers (Rust) and flash-tokenizer (C++)

For a cross-language perspective, [`src/HuggingfaceTokenizer/BenchPython`](src/HuggingfaceTokenizer/BenchPython) benchmarks [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) against [flash-tokenizer](https://github.com/NLPOptimize/flash-tokenizer) from Python on the same corpus and vocabulary, single-threaded and batched. Cross-language numbers are only roughly comparable to the .NET ones (different drivers, process startup, etc.). An id-level parity check (`verify.py`) shows flash-tokenizer produces identical ids for 99.6% of the corpus documents while Hugging Face tokenizers matches `AutoTokenizer` exactly; [`src/HuggingfaceTokenizer/BenchRust`](src/HuggingfaceTokenizer/BenchRust) additionally measures Hugging Face tokenizers natively via criterion.rs, without any FFI or Python overhead.

*(Numbers pending — will be filled in from the first full CI run of this benchmark setup.)*

## Logo

Created by combining <https://icons.getbootstrap.com/icons/cursor-text/> in .NET brand color with <https://icons.getbootstrap.com/icons/braces/>.
