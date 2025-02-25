// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using RustLibWrapper;
using Shouldly;

namespace FastBertTokenizer.Tests;

[Collection("UsesRustLib")]
public class AsyncBatchEnumeratorVsHuggingface : IAsyncLifetime
{
    private readonly BertTokenizer _baaiBgeTok = new();
    private readonly BertTokenizer _bertUncasedTok = new();
    private readonly BertTokenizer _bertMultilingualTok = new();
    private readonly BertTokenizer _bertChineseTok = new();
    private readonly BertTokenizer _issue100Tok = new();

    public async Task InitializeAsync()
    {
        await _baaiBgeTok.LoadVocabularyAsync("data/baai-bge-small-en/vocab.txt", true);
        await _bertUncasedTok.LoadVocabularyAsync("data/bert-base-uncased/vocab.txt", true);
        await _bertMultilingualTok.LoadVocabularyAsync("data/bert-base-multilingual-cased/vocab.txt", false);
        await _bertChineseTok.LoadVocabularyAsync("data/bert-base-chinese/vocab.txt", false);
        await _issue100Tok.LoadTokenizerJsonAsync("data/issue-100/tokenizer.json");
    }

    public Task DisposeAsync() => Task.CompletedTask;

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIsBertUncased512(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/bert-base-uncased/tokenizer.json", 512);
        await CompareSimpleWikipediaCorpusAsIsImpl(_bertUncasedTok, articles, 512, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIsBertMultilingualBge512(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/bert-base-multilingual-cased/tokenizer.json", 512);
        await CompareSimpleWikipediaCorpusAsIsImpl(_bertMultilingualTok, articles, 512, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIsBertChineseBge512(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/bert-base-chinese/tokenizer.json", 512);
        await CompareSimpleWikipediaCorpusAsIsImpl(_bertChineseTok, articles, 512, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIsBaaiBge512(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/baai-bge-small-en/tokenizer.json", 512);
        await CompareSimpleWikipediaCorpusAsIsImpl(_baaiBgeTok, articles, 512, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIsBaaiBge333(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/baai-bge-small-en/tokenizer.json", 333);
        await CompareSimpleWikipediaCorpusAsIsImpl(_baaiBgeTok, articles, 333, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIsBaaiBge27(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/baai-bge-small-en/tokenizer.json", 27);
        await CompareSimpleWikipediaCorpusAsIsImpl(_baaiBgeTok, articles, 27, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIsBaaiBge2048(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/baai-bge-small-en/tokenizer.json", 2048);
        await CompareSimpleWikipediaCorpusAsIsImpl(_baaiBgeTok, articles, 2048, 100);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task CompareSimpleWikipediaCorpusAsIsIssue100WithAddedTokens512(Dictionary<int, string> articles)
    {
        RustTokenizer.LoadTokenizer("data/issue-100/tokenizer.json", 512);
        await CompareSimpleWikipediaCorpusAsIsImpl(_issue100Tok, articles, 512, 100, padId: 0, unkId: 1, sepId: 3);
    }

    private async Task CompareSimpleWikipediaCorpusAsIsImpl(BertTokenizer uut, Dictionary<int, string> articles, int maxInputTokens, int batchSize, int padId = 0, int unkId = 100, int sepId = 102)
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
        await foreach (var batch in uut.CreateAsyncBatchEnumerator(EnumerateContent(), maxInputTokens, batchSize, 0))
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
                var currentInputIds = batch.InputIds.Slice(i * maxInputTokens, maxInputTokens);
                currentInputIds.Span[^1].ShouldBeOneOf(padId, sepId);
                if (tokRg.Offset > 0)
                {
                    continue;
                }

                var content = articles[tokRg.Key];
                var huggF = RustTokenizer.TokenizeAndGetIds(content, maxInputTokens);

                try
                {
                    currentInputIds.ShouldBe(huggF.InputIds);
                    batch.AttentionMask.Slice(i * maxInputTokens, maxInputTokens).ShouldBe(huggF.AttentionMask);
                }
                catch (Exception exFirst)
                {
                    try
                    {
                        if (CouldWeTokenizeInsteadOfUnk(huggF.InputIds.Span, currentInputIds.Span, unkId, sepId))
                        {
                            continue;
                        }
                        else if (DidWeEmitTwoUnkWhereHuggingFaceEmittedOne(huggF.InputIds.Span, currentInputIds.Span, unkId, sepId))
                        {
                            continue;
                        }
                        else if (DidWeEmitOneUnkWhereHuggingFaceJustSkippedSomething(huggF.InputIds.Span, currentInputIds.Span, unkId, sepId))
                        {
                            continue;
                        }
#if NETFRAMEWORK
                        else if (DidWeSkipSomethingWhereHuggingFaceEmittedUnk(huggF.InputIds.Span, currentInputIds.Span, unkId, sepId))
                        {
                            continue;
                        }
#endif

                        var needsToMatchUpToIdx = currentInputIds.Length - 1;

                        if (currentInputIds.Span[^1] == padId)
                        {
                            // Our result ends with padding. That might be because we don't tokenize partial words, while
                            // Hugging Face does.
                            while (currentInputIds.Span[--needsToMatchUpToIdx] == padId)
                            {
                                // Skip padding at the end if there is any. There might be because we didn't tokenize
                                // partial words.
                            }

                            // But then there needs to be a [SEP], otherwise there is some fault.
                            if (currentInputIds.Span[needsToMatchUpToIdx] != sepId)
                            {
                                throw new Exception($"Error comparing tokenization results for {tokRg.Key}", exFirst);
                            }

                            // It was a [SEP], so we skip it.
                            needsToMatchUpToIdx--;
                        }
                        else if (currentInputIds.Span[^1] == sepId)
                        {
                            // Our result ends with [SEP]. As FastBertTokenizer searches for partial words first,
                            // while Huggingface removes diacritics first, the last token before [SEP] might be
                            // different. We accept that.
                            needsToMatchUpToIdx--; // skip [SEP]
                            needsToMatchUpToIdx--; // skip the token before [SEP]
                        }
                        else
                        {
                            throw new Exception($"Error comparing tokenization results for {tokRg.Key}", exFirst);
                        }

                        var needsToMatchUpToLen = needsToMatchUpToIdx + 1;
                        currentInputIds.Slice(0, needsToMatchUpToLen).ShouldBe(huggF.InputIds.Slice(0, needsToMatchUpToLen));
                        batch.AttentionMask.Slice(i * maxInputTokens, needsToMatchUpToLen).ShouldBe(huggF.AttentionMask.Slice(0, needsToMatchUpToLen));
                    }
                    catch (Exception ex)
                    {
                        throw new Exception($"Error comparing tokenization results for {tokRg.Key}", ex);
                    }
                }
            }
        }
    }

    // Were we able to tokenize something, that Hugging Face tokenized as [UNK]?
    private bool CouldWeTokenizeInsteadOfUnk(ReadOnlySpan<long> huggF, ReadOnlySpan<long> ours, int unkId, int sepId)
    {
        var skippedHfUnk = false;
        var (iHF, iOurs) = (0, 0);
        while (iHF < huggF.Length && iOurs < ours.Length)
        {
            if (huggF[iHF] == ours[iOurs])
            {
                iHF++;
                iOurs++;
                continue;
            }

            // We might be at the end earlier because what we tokenized instead of [UNK] might be longer than 1
            if (skippedHfUnk && ours[iOurs] == sepId)
            {
                iOurs++;
                break;
            }

            // Skip [UNK]s in Hugging Face's result
            while (iHF < huggF.Length && huggF[iHF] == unkId)
            {
                skippedHfUnk = true;
                iHF++;
            }

            // Skip tokens in our result that are not in Hugging Face's result
            while (iOurs < ours.Length && ours[iOurs] != huggF[iHF])
            {
                iOurs++;
            }
        }

        return
            (iHF == huggF.Length && iOurs == ours.Length)
            || (skippedHfUnk && iOurs == ours.Length);
    }

    // Did we emit two [UNK] where Hugging Face just emitted one?
    private bool DidWeEmitTwoUnkWhereHuggingFaceEmittedOne(ReadOnlySpan<long> huggF, ReadOnlySpan<long> ours, int unkId, int sepId)
    {
        var lastHuggFWasUnk = false;
        var (iHF, iOurs) = (0, 0);
        var cases = 0;
        while (iHF < huggF.Length && iOurs < ours.Length)
        {
            if (huggF[iHF] == ours[iOurs])
            {
                lastHuggFWasUnk = huggF[iHF] == unkId;
                iHF++;
                iOurs++;
                continue;
            }

            if (lastHuggFWasUnk && ours[iOurs] == unkId)
            {
                iOurs++;
                lastHuggFWasUnk = false;
                cases++;
                continue;
            }

            // We might be at the end earlier because we tokenized two [UNK] instead of one (possibly in more than one case too).
            if (cases > 0 && iHF + cases == iOurs && ours[iOurs] == sepId)
            {
                iOurs++;
                break;
            }

            // The tokens were just different and that we emitted two [UNK] where Hugging Face emitted one
            // was not the reason for that.
            break;
        }

        return
            (iHF == huggF.Length && iOurs == ours.Length)
            || (iHF + cases == iOurs && iOurs == ours.Length);
    }

    private bool DidWeEmitOneUnkWhereHuggingFaceJustSkippedSomething(ReadOnlySpan<long> huggF, ReadOnlySpan<long> ours, int unkId, int sepId)
    {
        var (iHF, iOurs) = (0, 0);
        var cases = 0;
        while (iHF < huggF.Length && iOurs < ours.Length)
        {
            if (huggF[iHF] == ours[iOurs])
            {
                iHF++;
                iOurs++;
                continue;
            }

            if (ours[iOurs] == unkId && iOurs + 1 < ours.Length && ours[iOurs + 1] == huggF[iHF])
            {
                iOurs++;
                cases++;
                continue;
            }

            // We might be at the end earlier because we emited [UNK] were Hugging Face just skipped something.
            if (cases > 0 && iHF + cases == iOurs && ours[iOurs] == sepId)
            {
                iOurs++;
                break;
            }

            // The tokens were just different and that we emitted [UNK] where Hugging Face didn't was not the reason for that.
            break;
        }

        return
            (iHF == huggF.Length && iOurs == ours.Length)
            || (iHF + cases == iOurs && iOurs == ours.Length);
    }

    // Did Hugging Face emit one or more [UNK] where we just skipped something?
    // This is relevant for .netframework, probably due to it's outdated unicode data.
    private bool DidWeSkipSomethingWhereHuggingFaceEmittedUnk(ReadOnlySpan<long> huggF, ReadOnlySpan<long> ours, int unkId, int sepId)
    {
        var skippedHfUnk = 0;
        var (iHF, iOurs) = (0, 0);
        while (iHF < huggF.Length && iOurs < ours.Length)
        {
            if (huggF[iHF] == ours[iOurs])
            {
                iHF++;
                iOurs++;
                continue;
            }

            // Hugging face will be at the end earlier if it emitted [UNK] where we skipped something.
            if (skippedHfUnk > 0 && huggF[iHF] == sepId)
            {
                break;
            }

            // Skip [UNK]s in Hugging Face's result
            if (iHF < huggF.Length && huggF[iHF] == unkId)
            {
                skippedHfUnk++;
                iHF++;
                continue;
            }

            // The tokens were just different and that Hugging Face emitted [UNK] where we didn't was not the reason for that.
            break;
        }

        return
            (iHF == huggF.Length && iOurs == ours.Length)
            || (iHF == huggF.Length && iOurs == ours.Length - skippedHfUnk);
    }
}
