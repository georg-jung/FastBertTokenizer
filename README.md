## Performance

## Comparison to [BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers)

* almost 1 order of magnitude faster
* allocates more then 1 order of magnitude less memory
* [better whitespace handling](https://github.com/NMZivkovic/BertTokenizers/issues/24)
* [handles unknown characters correctly](https://github.com/NMZivkovic/BertTokenizers/issues/26)
* [does not throw if text is longer than maximum sequence length](https://github.com/NMZivkovic/BertTokenizers/issues/18)
* [correct handling of token type ids](https://github.com/NMZivkovic/BertTokenizers/issues/18)
* handles unicode control chars
* handles other alphabets such as greek
