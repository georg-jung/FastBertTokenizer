// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Shouldly;

namespace FastBertTokenizer.Tests;

public class Decode : IAsyncLifetime
{
    private readonly BertTokenizer _uut = new();

    public async Task InitializeAsync()
    {
        await _uut.LoadTokenizerJsonAsync("data/bert-base-uncased/tokenizer.json");
    }

    public Task DisposeAsync() => Task.CompletedTask;

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public void TokenizeWithBatchEnumerator(Dictionary<int, string> articles)
    {
        IEnumerable<(int, string)> Source()
        {
            foreach (var (key, value) in articles)
            {
                yield return (key, value);
            }
        }

        foreach (var batch in _uut.CreateBatchEnumerator(Source(), 512, 100, 0))
        {
            for (var i = 0; i < 100; i++)
            {
                var decoded = _uut.Decode(batch.InputIds.Span.Slice(i * 512, 512));
                decoded.Length.ShouldBeGreaterThan(1);
            }
        }
    }

    [Fact]
    public void DecodeStartingFromSuffix()
    {
        // 19544 is lore
        // 2213 is ##m
        long[] loremIpsum = [101, 19544, 2213, 12997, 17421, 2079, 10626, 4133, 2572, 3388, 1012, 102];
        var decoded = _uut.Decode(loremIpsum);
        decoded.ShouldStartWith("[CLS] lorem ipsum");

        long[] startsWithSuffix = loremIpsum[2..];
        decoded = _uut.Decode(startsWithSuffix);
        decoded.ShouldStartWith("m ipsum");
    }

    [Fact]
    public void DecodeEmpty()
    {
        long[] empty = [];
        var decoded = _uut.Decode(empty);
        decoded.ShouldBe(string.Empty);
    }
}
