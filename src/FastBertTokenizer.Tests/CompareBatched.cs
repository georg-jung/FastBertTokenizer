// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using RustLibWrapper;
using Shouldly;

namespace FastBertTokenizer.Tests;

public class CompareBatched : IClassFixture<BgeVocabFixture>
{
    private readonly BgeVocabFixture _bgeVocab;

    public CompareBatched(BgeVocabFixture bgeVocab)
    {
        _bgeVocab = bgeVocab;
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIs512(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/baai-bge-small-en/tokenizer.json", 512);
        await CompareSimpleWikipediaCorpusAsIsImpl(articles, 512, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIs333(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/baai-bge-small-en/tokenizer.json", 333);
        await CompareSimpleWikipediaCorpusAsIsImpl(articles, 333, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIs27(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/baai-bge-small-en/tokenizer.json", 27);
        await CompareSimpleWikipediaCorpusAsIsImpl(articles, 27, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIs2048(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/baai-bge-small-en/tokenizer.json", 2048);
        await CompareSimpleWikipediaCorpusAsIsImpl(articles, 2048, 100);
    }

    private async Task CompareSimpleWikipediaCorpusAsIsImpl(Dictionary<int, string> articles, int maxInputTokens, int batchSize)
    {
        async IAsyncEnumerable<(int, string)> EnumerateContent()
        {
            await Task.Delay(1);
            foreach (var (key, value) in articles)
            {
                yield return (key, value);
            }
        }

        var tok = _bgeVocab.UnderTest;
        var allNulls = new long[maxInputTokens];
        await foreach (var batch in tok.CreateAsyncBatchEnumerator(EnumerateContent(), maxInputTokens, batchSize, 0))
        {
            for (var i = 0; i < batch.InputIds.Length / maxInputTokens; i++)
            {
                var tokRgNullable = batch.OutputCorrelation.Span[i];
                if (!tokRgNullable.HasValue)
                {
                    batch.InputIds.Slice(i * maxInputTokens, maxInputTokens).ShouldBe(allNulls);
                    batch.AttentionMask.Slice(i * maxInputTokens, maxInputTokens).ShouldBe(allNulls);
                    continue;
                }

                var tokRg = tokRgNullable.Value;
                if (tokRg.Offset > 0)
                {
                    continue;
                }

                if (tokRg.Key == 6309 || tokRg.Key == 30153 || tokRg.Key == 60246)
                {
                    continue;
                }

                var content = articles[tokRg.Key];
                var huggF = RustTokenizer.TokenizeAndGetIds(content, maxInputTokens);

                var currentInputIds = batch.InputIds.Slice(i * maxInputTokens, maxInputTokens);
                try
                {
                    currentInputIds.ShouldBe(huggF.InputIds);
                    batch.AttentionMask.Slice(i * maxInputTokens, maxInputTokens).ShouldBe(huggF.AttentionMask);
                }
                catch (Exception exFirst)
                {
                    try
                    {
                        // FastBertTokenizer searches for partial words first, Huggingface removes diacritics first.
                        // The last token is always [SEP].
                        // Thus, we accept if the last token is different.
                        var firstSuffixNullIdx = currentInputIds.Length - 1;
                        while (currentInputIds.Span[firstSuffixNullIdx--] == 0) // assume [SEP] == 0 here
                        {
                            // count how many [SEP] our result has at it's end
                        }

                        // there might have been no [SEP] at the end but some real failure
                        if (currentInputIds.Span[firstSuffixNullIdx] != 0)
                        {
                            throw new Exception($"Error comparing tokenization results for {tokRg.Key}", exFirst);
                        }

                        currentInputIds.Slice(0, firstSuffixNullIdx).ShouldBe(huggF.InputIds.Slice(0, firstSuffixNullIdx));
                        batch.AttentionMask.Slice(i * maxInputTokens, firstSuffixNullIdx).ShouldBe(huggF.AttentionMask.Slice(0, firstSuffixNullIdx));
                    }
                    catch (Exception ex)
                    {
                        throw new Exception($"Error comparing tokenization results for {tokRg.Key}", ex);
                    }
                }
            }
        }
    }
}
