// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.IO;

namespace FastBertTokenizer.Tests;

public class AssertContracts
{
    [Fact]
    public async Task Parallel_Encode_tokenTypeIds_length()
    {
        var tokenizer = new BertTokenizer();
        await tokenizer.LoadVocabularyAsync("data/bert-base-uncased/vocab.txt", convertInputToLowercase: true);
        var resultLen = 512 * 2;
        string[] inputs = ["Hello, my dog is cute", "Hello, my cat is cute"];
        var inputIds = new long[resultLen];
        var attM = new long[resultLen];
        var tokenTypeIds = new long[resultLen];
        tokenizer.Encode(inputs, inputIds, attM, tokenTypeIds);
        tokenizer.Encode(inputs, inputIds, attM, tokenTypeIds); // coverage: range cache is used?
        Assert.Throws<ArgumentException>(() => tokenizer.Encode(inputs, new long[resultLen - 1], attM, tokenTypeIds));
        Assert.Throws<ArgumentException>(() => tokenizer.Encode(inputs, inputIds, new long[resultLen - 1], tokenTypeIds));
        Assert.Throws<ArgumentException>(() => tokenizer.Encode(inputs, inputIds, attM, new long[resultLen - 1]));
    }
}
