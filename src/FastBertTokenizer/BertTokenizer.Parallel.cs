// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer;

public partial class BertTokenizer
{
    public void Tokenize(IReadOnlyList<string> inputs, Memory<long> inputIds, Memory<long> attentionMask, Memory<long> tokenTypeIds, int maximumTokens = 512)
    {
        var resultLen = maximumTokens * inputs.Count;
        if (tokenTypeIds.Length != resultLen)
        {
            throw new ArgumentException($"{nameof(tokenTypeIds)} must have {resultLen} elements but had {tokenTypeIds.Length}.", nameof(tokenTypeIds));
        }

        Tokenize(inputs, inputIds, attentionMask, maximumTokens);
        tokenTypeIds.Span.Fill(0);
    }

    public void Tokenize(IReadOnlyList<string> inputs, Memory<long> inputIds, Memory<long> attentionMask, int maximumTokens = 512)
    {
        var resultLen = maximumTokens * inputs.Count;
        if (inputIds.Length != resultLen || attentionMask.Length != resultLen)
        {
            throw new ArgumentException($"{nameof(inputIds)} and {nameof(attentionMask)} must have {resultLen} elements, but had {inputIds.Length} and {attentionMask.Length}.");
        }

        inputs.Select((x, i) => (InputIds: x, i)).AsParallel().ForAll(x =>
        {
            var startIdx = maximumTokens * x.i;
            var (_, nonPad) = Tokenize(x.InputIds, inputIds.Slice(startIdx, maximumTokens), maximumTokens);
            var span = attentionMask.Slice(startIdx, maximumTokens).Span;
            span.Slice(0, nonPad).Fill(1);
            span.Slice(nonPad, maximumTokens - nonPad).Fill(0);
        });
    }
}
