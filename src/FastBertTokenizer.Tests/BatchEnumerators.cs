// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Shouldly;

namespace FastBertTokenizer.Tests;

public class BatchEnumerators : IAsyncLifetime
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
            batch.InputIds.Span[0].ShouldBe(101); // [CLS] = 101
        }
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public void TokenizeWithBatchEnumeratorNonGenericEnumerable(Dictionary<int, string> articles)
    {
        IEnumerable<(int, string)> Source()
        {
            foreach (var (key, value) in articles.Take(100)) // some data is sufficient here, as TokenizeWithBatchEnumerator already checks the same.
            {
                yield return (key, value);
            }
        }

        var enumerable = (System.Collections.IEnumerable)_uut.CreateBatchEnumerator(Source(), 512, 10, 0);
        foreach (var batch in enumerable)
        {
            var casted = (TokenizedBatch<int>)batch;
            casted.InputIds.Span[0].ShouldBe(101); // [CLS] = 101
        }

        var anotherEnumerable = (System.Collections.IEnumerable)_uut.CreateBatchEnumerator(Source(), 512, 10, 0);
        var enumerator = anotherEnumerable.GetEnumerator();
        Should.Throw<NotSupportedException>(enumerator.Reset);
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public async Task TokenizeWithAsyncBatchEnumerator(Dictionary<int, string> articles)
    {
        async IAsyncEnumerable<(int, string)> Source()
        {
            foreach (var (key, value) in articles)
            {
                await Task.Yield();
                yield return (key, value);
            }
        }

        await foreach (var batch in _uut.CreateAsyncBatchEnumerator(Source(), 512, 100, 0))
        {
            batch.InputIds.Span[0].ShouldBe(101); // [CLS] = 101
        }
    }

    [Fact]
    public void PreventAsyncBatchEnumeratorSyncMisuse()
    {
        async IAsyncEnumerable<(int, string)> Source()
        {
            await Task.Yield();
            yield return (0, "Lorem ipsum");
        }

        Should.Throw<InvalidCastException>(() =>
        {
            var x = (IEnumerable<TokenizedBatch<int>>)_uut.CreateAsyncBatchEnumerator(Source(), 512, 100, 0);
            foreach (var batch in x)
            {
                batch.InputIds.Span[0].ShouldBe(101); // [CLS] = 101
            }
        });
    }

    [Fact]
    public async Task PreventSyncBatchEnumeratorAsyncMisuse()
    {
        IEnumerable<(int, string)> Source()
        {
            yield return (0, "Lorem ipsum");
        }

        await Should.ThrowAsync<InvalidCastException>(async () =>
        {
            var x = (IAsyncEnumerable<TokenizedBatch<int>>)_uut.CreateBatchEnumerator(Source(), 512, 100, 0);
            await foreach (var batch in x)
            {
                batch.InputIds.Span[0].ShouldBe(101); // [CLS] = 101
            }
        });
    }
}
