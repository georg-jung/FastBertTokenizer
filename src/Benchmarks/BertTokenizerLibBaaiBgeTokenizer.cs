// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BERTTokenizers;
using BERTTokenizers.Base;

namespace SemanticKernel.BaaiBgeEmbeddingGenerator;

public class BertTokenizerLibBaaiBgeTokenizer : UncasedTokenizer
{
    public BertTokenizerLibBaaiBgeTokenizer()
        : base("Vocabularies\\baai-bge-small-en-vocab.txt")
    {
    }

    public (Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds) Tokenize(string input, CancellationToken cancellationToken = default)
    {
        var enc = Encode(512, input);
        return (enc.Select(x => x.InputIds).ToArray(), enc.Select(x => x.AttentionMask).ToArray(), enc.Select(x => x.TokenTypeIds).ToArray());
    }
}
