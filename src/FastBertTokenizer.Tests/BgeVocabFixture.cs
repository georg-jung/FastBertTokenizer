// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using RustLibWrapper;

namespace FastBertTokenizer.Tests;

public class BgeVocabFixture
{
    public BgeVocabFixture()
    {
        RustTokenizer.LoadTokenizer("data/baai-bge-small-en-tokenizer.json", 512);
        using var sr = new StreamReader("data/baai-bge-small-en-vocab.txt");
        UnderTest.LoadVocabulary(sr, true);
    }

    public BertTokenizer UnderTest { get; } = new();
}
