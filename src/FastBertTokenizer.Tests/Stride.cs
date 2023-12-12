// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Shouldly;

namespace FastBertTokenizer.Tests;

public class Stride : IAsyncLifetime
{
    private readonly BertTokenizer _bertUncasedTok = new();

    public async Task InitializeAsync()
    {
        await _bertUncasedTok.LoadVocabularyAsync("data/bert-base-uncased/vocab.txt", true);
    }

    public Task DisposeAsync() => Task.CompletedTask;

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task Stride25ContinuesBetweenBatchElements(Dictionary<int, string> articles)
    {
        await StrideContinuesFromElementToElementImpl(_bertUncasedTok, articles, 512, 100, 25);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task Stride77ContinuesBetweenBatchElements(Dictionary<int, string> articles)
    {
        await StrideContinuesFromElementToElementImpl(_bertUncasedTok, articles, 512, 100, 77);
    }

    private async Task StrideContinuesFromElementToElementImpl(BertTokenizer uut, Dictionary<int, string> articles, int maxInputTokens, int batchSize, int stride)
    {
        async IAsyncEnumerable<(int, string)> EnumerateContent()
        {
            await Task.Delay(1);
            foreach (var (key, value) in articles)
            {
                yield return (key, value);
            }
        }

        var allNulls = new long[maxInputTokens];
        Memory<long>? beforeInputIds = null;
        await foreach (var batch in uut.CreateAsyncBatchEnumerator(EnumerateContent(), maxInputTokens, batchSize, stride))
        {
            for (var i = 0; i < batch.InputIds.Length / maxInputTokens; i++)
            {
                var tokRgNullable = batch.OutputCorrelation.Span[i];
                if (!tokRgNullable.HasValue)
                {
                    batch.InputIds.Slice(i * maxInputTokens, maxInputTokens).ShouldBe(allNulls);
                    batch.AttentionMask.Slice(i * maxInputTokens, maxInputTokens).ShouldBe(allNulls);
                    beforeInputIds = null;
                    continue;
                }

                var tokRg = tokRgNullable.Value;

                try
                {
                    var currentInputIds = batch.InputIds.Slice(i * maxInputTokens, maxInputTokens);
                    currentInputIds.Span[^1].ShouldBeOneOf(0, 102); // [PAD] = 0, [SEP] = 102
                    if (tokRg.Offset == 0 || !beforeInputIds.HasValue)
                    {
                        beforeInputIds = currentInputIds;
                        continue;
                    }

                    var beforeSepIdx = beforeInputIds.Value.Span.LastIndexOf(102);
                    var beforeStrideStart = beforeSepIdx - stride;
                    var beforeStride = beforeInputIds.Value.Slice(beforeStrideStart, stride);

                    var currentStride = currentInputIds.Slice(1, stride);
                    currentInputIds.Span[0].ShouldBe(101); // [CLS] = 101
                    currentStride.ShouldBe(beforeStride);
                    beforeInputIds = currentInputIds;
                }
                catch (Exception ex)
                {
                    var content = articles[tokRg.Key];
                    throw new Exception($"Error comparing tokenization stride overlap for {content.Substring(0, Math.Min(content.Length, 100))}", ex);
                }
            }

            beforeInputIds = beforeInputIds.HasValue ? beforeInputIds!.Value.Span.ToArray() : null;
        }
    }
}
