<p align="center">
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
![.NET Build](https://github.com/georg-jung/FastBertTokenizer/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/github/georg-jung/FastBertTokenizer/graph/badge.svg?token=PEINHYEBGH)](https://codecov.io/github/georg-jung/FastBertTokenizer)

A fast and memory-efficient library for WordPiece tokenization as it is used by BERT. Tokenization correctness and speed are automatically evaluated in extensive unit tests and benchmarks.

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

## Logo

Created by combining <https://icons.getbootstrap.com/icons/cursor-text/> in .NET brand color with <https://icons.getbootstrap.com/icons/braces/>.
