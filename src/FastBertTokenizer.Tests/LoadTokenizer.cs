// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer.Tests;

public class LoadTokenizer
{
    [Theory]
    [InlineData("data/baai-bge-small-en/tokenizer.json")]
    [InlineData("data/bert-base-chinese/tokenizer.json")]
    [InlineData("data/bert-base-multilingual-cased/tokenizer.json")]
    [InlineData("data/bert-base-uncased/tokenizer.json")]
    public async Task LoadTokenizerFromJson(string path)
    {
        var tokenizer = new BertTokenizer();
        await tokenizer.LoadTokenizerJsonAsync(path);
    }

    [Theory]
    [InlineData("data/baai-bge-small-en/vocab.txt", true)]
    [InlineData("data/bert-base-chinese/vocab.txt", true)]
    [InlineData("data/bert-base-multilingual-cased/vocab.txt", true)]
    [InlineData("data/bert-base-uncased/vocab.txt", true)]
    public async Task LoadTokenizerFromVocab(string path, bool lowercase)
    {
        var tokenizer = new BertTokenizer();
        await tokenizer.LoadVocabularyAsync(path, lowercase);
    }
}
