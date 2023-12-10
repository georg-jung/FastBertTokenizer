<p align="center">
  <a href="https://www.nuget.org/packages/FastBertTokenizer/">
    <img
      alt="FastBertTokenizer logo"
      src="logo.svg"
      width="100"
    />
  </a>
</p>

# FastBertTokenizer

[![NuGet version (FastBertTokenizer)](https://img.shields.io/nuget/v/FastBertTokenizer.svg?style=flat)](https://www.nuget.org/packages/FastBertTokenizer/)
![.NET Build](https://github.com/georg-jung/FastBertTokenizer/actions/workflows/ci.yml/badge.svg)

A fast and memory-efficient library for WordPiece tokenization as it is used by BERT. Tokenization correctness and speed are automatically evaluated in extensive unit tests and benchmarks.

## Goals

* Enabling you to run your AI workloads on .NET in production.
* **Correctness** - Results that are equivalent to [HuggingFace Transformers' `AutoTokenizer`'s](https://huggingface.co/docs/transformers/v4.33.0/en/model_doc/auto#transformers.AutoTokenizer) in all practical cases.
* **Speed** - Tokenization should be as fast as reasonably possible.
* **Ease of use** - The API should be easy to understand and get started with.

## Features

* same results as [HuggingFace Transformers' `AutoTokenizer`](https://huggingface.co/docs/transformers/v4.33.0/en/model_doc/auto#transformers.AutoTokenizer) in all relevant cases.
* purely managed and dependency-free
* optimized for high performance and low memory usage

## Getting started

```csharp
using FastBertTokenizer;

var tok = new BertTokenizer();
var maxTokensForModel = 512;
await tok.LoadVocabularyAsync("vocab.txt", true); // https://huggingface.co/BAAI/bge-small-en/blob/main/vocab.txt
var text = File.ReadAllText("TextFile.txt");
var (inputIds, attentionMask, tokenTypeIds) = tok.Tokenize(text, maxTokensForModel);
Console.WriteLine(string.Join(", ", inputIds.ToArray().Select(x => x.ToString())));
```

## Comparison of Tokenization Results to [HuggingFace Transformers' `AutoTokenizer`](https://huggingface.co/docs/transformers/v4.33.0/en/model_doc/auto#transformers.AutoTokenizer)

For correctness verification about 10.000 articles of [simple english Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) were tokenized using FastBertTokenizer and Huggingface using the [baai bge vocab.txt](https://huggingface.co/BAAI/bge-small-en/blob/main/vocab.txt) file. The tokenization results were exactly the same apart from these two cases:

* [*Letter*](https://simple.wikipedia.org/wiki/Letter) (id 6309) contains assamese characters. Many of them are not represented in the vocabulary used. Huggingface's tokenizer skips exactly one [UNK] token for one of the chars were *FastBertTokenizer* emits one.
* [*Avignon*](https://simple.wikipedia.org/wiki/Avignon) (id 30153) has Rhône as the last word before hitting the 512 token id limit. If a word can not directly be found in the vocabulary, *FastBertTokenizer* we tries to tokenize prefixes of the word first, while Huggingface directly starts with a diacritic-free version of the word. Thus, *FastBertTokenizer*'s result ends with token id for `r` while huggingface (correctly) emits `rhone`. This edge case is just relevant
    1. for the last word, after which the tokenized output is cut off and
    2. if this last word contains diacritics.

These minor differences might be irrelevant in most real-world use cases. All other tested >10.000 articles including chinese and korean characters as well as much less common scripts and right-to-left letters were tokenized exactly the same as by Huggingface's Tokenizer.

## Comparison to [BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers)

* about 1 order of magnitude faster
* allocates more than 1 order of magnitude less memory
* [better whitespace handling](https://github.com/NMZivkovic/BertTokenizers/issues/24)
* [handles unknown characters correctly](https://github.com/NMZivkovic/BertTokenizers/issues/26)
* [does not throw if text is longer than maximum sequence length](https://github.com/NMZivkovic/BertTokenizers/issues/18)
* handles unicode control chars
* handles other alphabets such as greek and right-to-left languages

Note that while [BERTTokenizers handles token type incorrectly](https://github.com/NMZivkovic/BertTokenizers/issues/18), it does support input of two pieces of text that are tokenized with a separator in between. *FastBertTokenizer* currently does not support this.

## Benchmark

> Tokenizing the first 5000 characters of 10254 articles of simple english Wikipedia.
> ThinkPad T14s Gen 1, AMD Ryzen 7 PRO 4750U, 32GB memory

| Method                      | Mean       | Error    | StdDev   | Gen0         | Gen1       | Gen2      | Allocated  |
|---------------------------- |-----------:|---------:|---------:|-------------:|-----------:|----------:|-----------:|
| [BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers)                    | 4,942.0 ms | 54.79 ms | 48.57 ms | 1001000.0000 | 95000.0000 | 4000.0000 | 5952.43 MB |
| FastBertTokenizerAllocating |   529.5 ms |  8.90 ms | 10.59 ms |   61000.0000 | 31000.0000 | 2000.0000 |  350.75 MB |
| FastBertTokenizerMemReuse   |   404.5 ms |  7.72 ms |  7.22 ms |   68000.0000 |          - |         - |  136.83 MB |

The `FastBertTokenizerMemReuse` benchmark writes the results of the tokenization to the same memory area while `FastBertTokenizerAllocating` allocates new memory for it's return values. See [`src/Benchmarks`](/src/Benchmarks/) for details how these benchmarks were perfomed.

## Logo

Created by combining <https://icons.getbootstrap.com/icons/cursor-text/> in .NET brand color with <https://icons.getbootstrap.com/icons/braces/>.
