// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Threading.Channels;
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
    public async Task CompareFlavors(Dictionary<int, string> articles)
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

        // batches
        // CreateBatchEnumerator includes partial last words if stride == 0
        var iids4Dic = new Dictionary<(int InputIdx, int Offset), (long[] Iids, long[] Attm)>(inputs.Length);
        foreach (var batch in _uut.CreateBatchEnumerator(inputs.Select((x, idx) => (idx, x)), 512, 100, stride: 0))
        {
            for (var idx = 0; idx < batch.OutputCorrelation.Length; idx++)
            {
                var corrNullable = batch.OutputCorrelation.Span[idx];
                if (corrNullable is TokenizedRange<int> corr)
                {
                    var arrIids = batch.InputIds.Slice(idx * 512, 512).ToArray();
                    var arrAttm = batch.AttentionMask.Slice(idx * 512, 512).ToArray();
                    iids4Dic.Add((corr.Key, corr.Offset), (arrIids, arrAttm));
                    if (corr.Offset > 0)
                    {
                        continue;
                    }

                    var inputIdx = corr.Key;
                    arrIids.AsMemory().ShouldBe(iids.AsMemory(inputIdx * 512, 512));
                    arrAttm.AsMemory().ShouldBe(attm.AsMemory(inputIdx * 512, 512));
                }
            }
        }

        var channel = Channel.CreateUnbounded<(int, string)>();
        foreach (var (idx, input) in inputs.Select((x, idx) => (idx, x)))
        {
            channel.Writer.TryWrite((idx, input)).ShouldBeTrue();
        }

        channel.Writer.TryComplete().ShouldBeTrue();

#if NETFRAMEWORK
        var asyncEnum = channel.Reader.AsAsyncEnumerable();
#else
        var asyncEnum = channel.Reader;
#endif
        await foreach (var batch in _uut.CreateAsyncBatchEnumerator(asyncEnum, 512, 100, stride: 0))
        {
            for (var idx = 0; idx < batch.OutputCorrelation.Length; idx++)
            {
                var corrNullable = batch.OutputCorrelation.Span[idx];
                if (corrNullable is TokenizedRange<int> corr)
                {
                    iids4Dic.TryGetValue((corr.Key, corr.Offset), out var arrs).ShouldBeTrue();
                    batch.InputIds.Slice(idx * 512, 512).ShouldBe(arrs.Iids);
                    batch.AttentionMask.Slice(idx * 512, 512).ShouldBe(arrs.Attm);
                }
            }
        }

        // batches with stride
        var iids5Dic = new Dictionary<(int InputIdx, int Offset), (long[] Iids, long[] Attm)>(inputs.Length);
        foreach (var batch in _uut.CreateBatchEnumerator(inputs.Select((x, idx) => (idx, x)), 512, 100, stride: 27))
        {
            for (var idx = 0; idx < batch.OutputCorrelation.Length; idx++)
            {
                var corrNullable = batch.OutputCorrelation.Span[idx];
                if (corrNullable is TokenizedRange<int> corr)
                {
                    var arrIids = batch.InputIds.Slice(idx * 512, 512).ToArray();
                    var arrAttm = batch.AttentionMask.Slice(idx * 512, 512).ToArray();
                    iids5Dic.Add((corr.Key, corr.Offset), (arrIids, arrAttm));
                    if (corr.Offset > 0)
                    {
                        continue;
                    }

                    var inputIdx = corr.Key;

                    // 10 is rather arbitrarily chosen here but works for our current test data.
                    // Because partial last words are not included, as stride is > 0, the last
                    // tokens of our batched results might differ from the non batched ones.
                    arrIids.AsMemory(0, 512 - 10).ShouldBe(iids.AsMemory(inputIdx * 512, 512 - 10));
                    arrAttm.AsMemory(0, 512 - 10).ShouldBe(attm.AsMemory(inputIdx * 512, 512 - 10));
                }
            }
        }

        var channel2 = Channel.CreateUnbounded<(int, string)>();
        foreach (var (idx, input) in inputs.Select((x, idx) => (idx, x)))
        {
            channel2.Writer.TryWrite((idx, input)).ShouldBeTrue();
        }

        channel2.Writer.TryComplete().ShouldBeTrue();

#if NETFRAMEWORK
        var asyncEnum2 = channel.Reader.AsAsyncEnumerable();
#else
        var asyncEnum2 = channel.Reader;
#endif
        await foreach (var batch in _uut.CreateAsyncBatchEnumerator(asyncEnum2, 512, 100, stride: 27))
        {
            for (var idx = 0; idx < batch.OutputCorrelation.Length; idx++)
            {
                var corrNullable = batch.OutputCorrelation.Span[idx];
                if (corrNullable is TokenizedRange<int> corr)
                {
                    iids5Dic.TryGetValue((corr.Key, corr.Offset), out var arrs).ShouldBeTrue();
                    batch.InputIds.Slice(idx * 512, 512).ShouldBe(arrs.Iids);
                    batch.AttentionMask.Slice(idx * 512, 512).ShouldBe(arrs.Attm);
                }
            }
        }
    }
}
