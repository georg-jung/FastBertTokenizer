// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Shouldly;

namespace FastBertTokenizer.Tests;

public class CompareDifferentEncodeFlavors : IAsyncLifetime
{
    private readonly BertTokenizer _uut = new();

    public Task DisposeAsync() => Task.CompletedTask;

    public async Task InitializeAsync()
    {
        await _uut.LoadTokenizerJsonAsync("data/bert-base-uncased/tokenizer.json");
    }

    [Theory]
    [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
    public void CompareFlavors(Dictionary<int, string> articles)
    {
        var inputs = articles.Values.ToArray();
        var iids = new long[inputs.Length * 512];
        var attm = new long[inputs.Length * 512];
        var toktypIds = new long[inputs.Length * 512];
        var i = 0;
        foreach (var input in inputs)
        {
            var res = _uut.Encode(input, 512, 512);
            res.InputIds.CopyTo(iids.AsMemory(i * 512, 512));
            res.AttentionMask.CopyTo(attm.AsMemory(i * 512, 512));
            res.TokenTypeIds.CopyTo(toktypIds.AsMemory(i * 512, 512));
            i++;
        }

        var iids2 = new long[inputs.Length * 512];
        var attm2 = new long[inputs.Length * 512];
        var toktypIds2 = new long[inputs.Length * 512];
        _uut.Encode(inputs, iids2, attm2, toktypIds2, 512);

        iids2.ShouldBe(iids);
        attm2.ShouldBe(attm);
        toktypIds2.ShouldBe(toktypIds);

        var iids3 = new long[512];
        var attm3 = new long[512];
        var toktypIds3 = new long[512];
        i = 0;
        foreach (var input in inputs)
        {
            var res = _uut.Encode(input, iids3, attm3, toktypIds3, 512);
            iids3.AsMemory().ShouldBe(iids.AsMemory(i * 512, 512));
            attm3.AsMemory().ShouldBe(attm.AsMemory(i * 512, 512));
            toktypIds3.AsMemory().ShouldBe(toktypIds.AsMemory(i * 512, 512));
            i++;
        }
    }
}
