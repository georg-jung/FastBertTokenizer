// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer;

public partial class BertTokenizer
{
    /// <summary>
    /// Encode the given batch of input strings to token ids per the loaded vocabulary using
    /// multiple threads in parallel. Write the results to the given memory areas. When encoding
    /// multiple batches successivly it is more efficient to reuse the memory.
    /// When inferencing a model that supports batch processing you might be able to directly
    /// pass the written memory areas to the model.
    /// </summary>
    /// <param name="inputs">The batch of input strings to encode.</param>
    /// <param name="inputIds">
    /// The resulting token ids/input_ids will be written here. For each input in the batch a
    /// maximum of <paramref name="maximumTokens"/> token_ids will be written here. Unused
    /// positions will be filled with padding tokens so that each input in the batch uses
    /// exactly <paramref name="maximumTokens"/> positions.
    /// E.g. if the batch contains 50 inputs and <paramref name="maximumTokens"/> is 512, this
    /// memory area must have a length of 50 * 512 = 25600.
    /// </param>
    /// <param name="attentionMask">
    /// The attention masks for the given inputs will be written here. At the positions
    /// of padding tokens in <paramref name="inputIds"/> attention mask will be 0.
    /// All other (relevant/interesting) positions will have a value of 1.
    /// Must have the same length as <paramref name="inputIds"/>.
    /// </param>
    /// <param name="tokenTypeIds">
    /// Will be filled with 0s. Use the overload without this parameter for optimized speed.
    /// Some models which can take multiple sequences as input might need this but this is
    /// currently not supported by FastBertTokenizer.
    /// /// Must have the same length as <paramref name="inputIds"/>.
    /// </param>
    /// <param name="maximumTokens">
    /// The maximum number of token ids to produce for every single input in the batch.
    /// Most BERT models support a maximum of 512 tokens per input.
    /// </param>
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

    /// <inheritdoc cref="Tokenize(IReadOnlyList{string}, Memory{long}, Memory{long}, Memory{long}, int)"/>
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
