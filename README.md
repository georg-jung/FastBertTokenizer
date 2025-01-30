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

All benchmarks were performed on a typical end user notebook, a ThinkPad T14s Gen 1:

```txt
BenchmarkDotNet v0.13.12, Windows 11 (10.0.22631.3527/23H2/2023Update/SunValley3)
AMD Ryzen 7 PRO 4750U with Radeon Graphics, 1 CPU, 16 logical and 8 physical cores
.NET SDK 8.0.204
```

Similar results can also be observed using [GitHub Actions](https://github.com/georg-jung/FastBertTokenizer/actions/workflows/benchmark.yml). Note that using shared CI runners for benchmarking has drawbacks and can lead to varying results though.

### on NET 6.0 vs. on NET 8.0

* `.NET 6.0.29 (6.0.2924.17105), X64 RyuJIT AVX2` vs `.NET 8.0.4 (8.0.424.16909), X64 RyuJIT AVX2`
* Workload: Encode up to 512 tokens from each of 15,000 articles from simple english wikipedia.
* Results: Total tokens produced: 3,657,145; on .NET 8: ~11m tokens/s single threaded, 73m tokens/s multi threaded.

| Method                       | Runtime  | Mean      | Error    | StdDev   | Ratio | Gen0       | Gen1       | Gen2     | Allocated | Alloc Ratio |
|----------------------------- |--------- |----------:|---------:|---------:|------:|-----------:|-----------:|---------:|----------:|------------:|
| Singlethreaded               | .NET 6.0 | 450.39 ms | 7.340 ms | 6.866 ms |  1.00 |          - |          - |        - |      2 MB |        1.00 |
| MultithreadedMemReuseBatched | .NET 6.0 |  72.46 ms | 1.337 ms | 1.251 ms |  0.16 |   750.0000 |   250.0000 | 250.0000 |  12.75 MB |        6.39 |
|                              |          |           |          |          |       |            |            |          |           |             |
| Singlethreaded               | .NET 8.0 | 332.51 ms | 6.574 ms | 7.826 ms |  1.00 |          - |          - |        - |   1.99 MB |        1.00 |
| MultithreadedMemReuseBatched | .NET 8.0 |  50.83 ms | 0.999 ms | 1.995 ms |  0.15 |   500.0000 |          - |        - |  12.75 MB |        6.40 |

### vs. [SharpToken](https://github.com/dmitry-brazhenko/SharpToken)

* `SharpToken v2.0.2`
* `.NET 8.0.4 (8.0.424.16909), X64 RyuJIT AVX2`
* Workload: Fully encode 15,000 articles from simple english wikipedia. Total tokens produced by FastBertTokenizer: 5,807,949 (~9.4m tokens/s single threaded).

This isn't an apples to apples comparison as BPE (what SharpToken does) and WordPiece encoding (what FastBertTokenizer does) are different tasks/algorithms. Both were applied to exactly the same texts/corpus though.

| Method                        | Mean       | Error    | StdDev   | Gen0      | Gen1      | Allocated |
|------------------------------ |-----------:|---------:|---------:|----------:|----------:|----------:|
| SharpTokenFullArticles        | 1,551.9 ms | 25.82 ms | 24.15 ms | 5000.0000 | 2000.0000 |  32.56 MB |
| FastBertTokenizerFullArticles |   620.3 ms |  7.00 ms |  6.21 ms |         - |         - |   2.26 MB |

### vs. HuggingFace [tokenizers](https://github.com/huggingface/tokenizers) (Rust)

`tokenizers v0.19.1`

I'm not really experienced in benchmarking rust code, but my attempts using criterion.rs (see `src/HuggingfaceTokenizer/BenchRust`) suggest that it takes tokenizers around

* batched/multi threaded: ~2 s (~2.9m tokens/s)
* single threaded: ~10 s (~0.6m tokens/s)

to produce 5,807,947 tokens from the same 15k simple english wikipedia articles. Contrary to what one might expect, this does mean that FastBertTokenizer, beeing a managed implementation, outperforms tokenizers. It should be noted though that tokenizers has a much more complete feature set while FastBertTokenizer is specifically optimized for WordPiece/Bert encoding.

The tokenizers repo states `Takes less than 20 seconds to tokenize a GB of text on a server's CPU.` As 26 MB of text take ~2s on my notebook CPU, 1 GB would take roughly 80 s. I think it makes sense that "a server's CPU" might be 4x as fast as my notebook's CPU and thus think my results seem plausible. It is however also possible that I unintentionally handicapped tokenizers somehow. Please let me know if you think so!

### vs. [BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers)

* `BERTTokenizers v1.2.0`
* `.NET 8.0.4 (8.0.424.16909), X64 RyuJIT AVX2`
* Workload: Prefixes of the contents of 15k simple english wikipedia articles, preprocessed to make them encodable by BERTTokenizers.

| Method                                     | Mean       | Error    | StdDev   | Gen0        | Gen1       | Gen2      | Allocated  |
|------------------------------------------- |-----------:|---------:|---------:|------------:|-----------:|----------:|-----------:|
| NMZivkovic_BertTokenizers                  | 2,576.0 ms | 15.49 ms | 13.73 ms | 968000.0000 | 40000.0000 | 1000.0000 | 3430.51 MB |
| FastBertTokenizer_SameDataAsBertTokenizers |   229.8 ms |  4.55 ms |  6.23 ms |           - |          - |         - |    1.03 MB |

## Logo

Created by combining <https://icons.getbootstrap.com/icons/cursor-text/> in .NET brand color with <https://icons.getbootstrap.com/icons/braces/>.
