# FastBertTokenizer

A fast and memory-efficient library for WordPiece tokenization as it is used by BERT. Tokenization results are tested against the outputs of [HuggingFace Transformers' `AutoTokenizer`](https://huggingface.co/docs/transformers/v4.33.0/en/model_doc/auto#transformers.AutoTokenizer).

Serves similar needs and initially inspired by [BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers) - thanks for the great work.

## Comparison to [BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers)

* about 1 order of magnitude faster
* allocates more then 1 order of magnitude less memory
* [better whitespace handling](https://github.com/NMZivkovic/BertTokenizers/issues/24)
* [handles unknown characters correctly](https://github.com/NMZivkovic/BertTokenizers/issues/26)
* [does not throw if text is longer than maximum sequence length](https://github.com/NMZivkovic/BertTokenizers/issues/18)
* While [BERTTokenizers handles token type incorrectly](https://github.com/NMZivkovic/BertTokenizers/issues/18), it does support input of to pieces of text that are tokenized with a separator in between. FastBertTokenizer currently does not support this.
* handles unicode control chars
* handles other alphabets such as greek and right-to-left languages

## Benchmark

> Tokenizing the first 5000 characters of 10254 articles of simple english Wikipedia.
> ThinkPad T14s Gen 1, AMD Ryzen 7 PRO 4750U, 32GB memory

| Method                      | Mean       | Error    | StdDev   | Gen0         | Gen1       | Gen2      | Allocated  |
|---------------------------- |-----------:|---------:|---------:|-------------:|-----------:|----------:|-----------:|
| OtherLib                    | 4,942.0 ms | 54.79 ms | 48.57 ms | 1001000.0000 | 95000.0000 | 4000.0000 | 5952.43 MB |
| FastBertTokenizerAllocating |   529.5 ms |  8.90 ms | 10.59 ms |   61000.0000 | 31000.0000 | 2000.0000 |  350.75 MB |
| FastBertTokenizerMemReuse   |   404.5 ms |  7.72 ms |  7.22 ms |   68000.0000 |          - |         - |  136.83 MB |

The `FastBertTokenizerMemReuse` benchmark writes the results of the tokenization to the same memory area while `FastBertTokenizerAllocating` allocates new memory for it's return values.
