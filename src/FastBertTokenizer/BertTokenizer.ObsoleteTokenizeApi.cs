// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace FastBertTokenizer;

public partial class BertTokenizer
{
    [ExcludeFromCodeCoverage]
    [Obsolete("Use BertTokenizer.Encode instead.")]
    public int Tokenize(string input, Span<long> inputIds, Span<long> attentionMask, Span<long> tokenTypeIds, int? padTo = null)
        => Encode(input, inputIds, attentionMask, tokenTypeIds, padTo);

    [ExcludeFromCodeCoverage]
    [Obsolete("Use BertTokenizer.Encode instead.")]
    public int Tokenize(string input, Span<long> inputIds, Span<long> attentionMask, int? padTo = null)
        => Encode(input, inputIds, attentionMask, padTo);

    [ExcludeFromCodeCoverage]
    [Obsolete("Use BertTokenizer.Encode instead.")]
    public (ReadOnlyMemory<long> InputIds, ReadOnlyMemory<long> AttentionMask, ReadOnlyMemory<long> TokenTypeIds) Tokenize(string input, int maximumTokens = 512, int? padTo = null)
        => Encode(input, maximumTokens, padTo);

    [ExcludeFromCodeCoverage]
    [Obsolete("Use BertTokenizer.Encode instead.")]
    public void Tokenize(ReadOnlyMemory<string> inputs, Memory<long> inputIds, Memory<long> attentionMask, Memory<long> tokenTypeIds, int maximumTokens = 512)
        => Encode(inputs, inputIds, attentionMask, tokenTypeIds, maximumTokens);

    [ExcludeFromCodeCoverage]
    [Obsolete("Use BertTokenizer.Encode instead.")]
    public void Tokenize(ReadOnlyMemory<string> inputs, Memory<long> inputIds, Memory<long> attentionMask, int maximumTokens = 512)
        => Encode(inputs, inputIds, attentionMask, maximumTokens);
}
