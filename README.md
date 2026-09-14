# ![FastBertTokenizer logo](https://raw.githubusercontent.com/georg-jung/FastBertTokenizer/master/logo.svg) FastBertTokenizer

[![NuGet version (FastBertTokenizer)](https://img.shields.io/nuget/v/FastBertTokenizer.svg?style=flat)](https://www.nuget.org/packages/FastBertTokenizer/)
[![Docs](https://img.shields.io/badge/Docs-fastberttokenizer.gjung.com-blue)](https://fastberttokenizer.gjung.com/)
![.NET Build](https://github.com/georg-jung/FastBertTokenizer/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/github/georg-jung/FastBertTokenizer/graph/badge.svg?token=PEINHYEBGH)](https://codecov.io/github/georg-jung/FastBertTokenizer)

A fast and memory-efficient library for WordPiece tokenization as it is used by BERT. Tokenization correctness and speed are automatically evaluated in extensive unit tests and benchmarks. Native AOT compatible and targets `net10.0`, `net8.0` and `netstandard2.0`.

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

Note: FastBertTokenizer currently does not support encoding two pieces of text into a single input with a separator in between and corresponding token_type_ids, as some models (e.g. cross-encoders) expect.

## Speed / Benchmarks

> tl;dr: FastBertTokenizer encodes ~14.5 million tokens per second on a single core, enough to tokenize a full-length novel in under 10 ms. Batched across the 4 vCPUs of a GitHub Actions runner, that grows to ~35 million tokens per second.

Market overview from [a full CI run](https://github.com/georg-jung/FastBertTokenizer/actions/runs/34832739276) (GitHub Actions shared runner, `ubuntu-24.04`, 4 vCPUs): tokenizing 15,000 simple english wikipedia articles (3,657,145 tokens) with bert-base-uncased's vocabulary, truncated to 512 tokens per input. For FastBertTokenizer that is ~14.5m tokens/s single threaded and ~35.3m tokens/s multi threaded.

| Library                                                                            | Measured from | Single threaded | Parallel |
|----------------------------------------------------------------------------------- |-------------- |----------------:|---------:|
| **FastBertTokenizer**                                                              | .NET          |      **265 ms** | **104 ms** |
| [tokie](https://github.com/chonkie-inc/tokie) (Rust)                               | Python        |          513 ms |   230 ms |
| [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers)  | .NET          |          785 ms |        — |
| [flash-tokenizer](https://github.com/NLPOptimize/flash-tokenizer) (C++)            | Python        |          1.13 s |   767 ms |
| [BlingFire](https://github.com/microsoft/BlingFire) (C++)                          | .NET          |          1.22 s |        — |
| [Tokenizers.DotNet](https://github.com/sappho192/Tokenizers.DotNet) (HF bindings)  | .NET          |          4.87 s |        — |
| [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) (Rust)        | Python        |          9.30 s |   3.93 s |

The libraries don't all do exactly the same work and cross-language numbers are only roughly comparable: e.g. Hugging Face tokenizers' single-threaded number includes per-call Python overhead, and tokie may use multiple cores even for sequential calls. See [`src/Benchmarks/README.md`](src/Benchmarks/README.md) for all detailed results (incl. FastBertTokenizer's different usage patterns and runtimes), the exact environment, fairness notes, and how to run the benchmarks yourself.

## Logo

Created by combining <https://icons.getbootstrap.com/icons/cursor-text/> in .NET brand color with <https://icons.getbootstrap.com/icons/braces/>.
