// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Collections;

namespace FastBertTokenizer;

internal class AsyncBatchEnumerator<TKey>
{
    private readonly BertTokenizer _tokenizer;
    private readonly int _tokensPerInput;
    private readonly int _batchSize;
    private readonly int _stride;
    private readonly long[] _inputIds;
    private readonly long[] _attentionMask;
    private readonly TokenizedRange<TKey>?[] _outputCorrelation;
    private readonly IAsyncEnumerator<(TKey Key, string Content)>? _asyncSourceEnumerator;
    private readonly IEnumerator<(TKey Key, string Content)>? _sourceEnumerator;
    private (TKey Key, string Conent, int Offset)? _pivot;
    private ReadOnlyMemory<long> _strideInputIds = default;

    private AsyncBatchEnumerator(BertTokenizer tokenizer, IAsyncEnumerable<(TKey Key, string Content)> asyncSourceEnumerable, int tokensPerInput, int batchSize, int stride)
        : this(tokenizer, tokensPerInput, batchSize, stride)
    {
        _asyncSourceEnumerator = asyncSourceEnumerable.GetAsyncEnumerator();
    }

    private AsyncBatchEnumerator(BertTokenizer tokenizer, IEnumerable<(TKey Key, string Content)> sourceEnumerable, int tokensPerInput, int batchSize, int stride)
        : this(tokenizer, tokensPerInput, batchSize, stride)
    {
        _sourceEnumerator = sourceEnumerable.GetEnumerator();
    }

    private AsyncBatchEnumerator(BertTokenizer tokenizer, int tokensPerInput, int batchSize, int stride)
    {
        _tokenizer = tokenizer;
        _tokensPerInput = tokensPerInput;
        _batchSize = batchSize;
        _stride = stride;
        _inputIds = new long[batchSize * tokensPerInput];
        _attentionMask = new long[batchSize * tokensPerInput];
        _outputCorrelation = new TokenizedRange<TKey>?[batchSize];
        Current = new() { AttentionMask = _attentionMask, InputIds = _inputIds, OutputCorrelation = _outputCorrelation };
    }

    private TokenizedBatch<TKey> Current { get; }

    public static IAsyncEnumerable<TokenizedBatch<TKey>> CreateAsync(BertTokenizer tokenizer, IAsyncEnumerable<(TKey Key, string Content)> asyncSourceEnumerable, int tokensPerInput, int batchSize, int stride)
    {
        var impl = new AsyncBatchEnumerator<TKey>(tokenizer, asyncSourceEnumerable, tokensPerInput, batchSize, stride);
        return new AsyncEnumerable(impl);
    }

    public static IEnumerable<TokenizedBatch<TKey>> CreateSync(BertTokenizer tokenizer, IEnumerable<(TKey Key, string Content)> asyncSourceEnumerable, int tokensPerInput, int batchSize, int stride)
    {
        var impl = new AsyncBatchEnumerator<TKey>(tokenizer, asyncSourceEnumerable, tokensPerInput, batchSize, stride);
        return new SyncEnumerable(impl);
    }

    public async ValueTask<bool> MoveNextAsync()
    {
        var i = 0;
        var withStride = _stride > 0;
        for (i = 0; i < _batchSize; i++)
        {
            if (!_pivot.HasValue)
            {
                if (!(_sourceEnumerator?.MoveNext() ?? await _asyncSourceEnumerator!.MoveNextAsync()))
                {
                    break;
                }

                var current = _sourceEnumerator?.Current ?? _asyncSourceEnumerator!.Current;
                _pivot = (current.Key, current.Content, 0);
            }

            var p = _pivot.Value;
            var (corr, nonPadding) = _tokenizer.TokenizeBatchElement(
                p.Key,
                p.Conent,
                p.Offset,
                _strideInputIds.Span,
                _inputIds.AsSpan(i * _tokensPerInput, _tokensPerInput),
                _attentionMask.AsSpan(i * _tokensPerInput, _tokensPerInput),
                withStride);

            _outputCorrelation[i] = corr;
            _pivot = corr switch
            {
                { LastTokenizedWordStartIndex: int idx } => (p.Key, p.Conent, idx),
                _ => null,
            };

            _strideInputIds = _pivot switch
            {
                null => Array.Empty<long>(),
                _ => _inputIds.AsMemory((i * _tokensPerInput) + nonPadding - 1 - _stride, _stride),
            };
        }

        if (i > 0 && i < _batchSize)
        {
            _inputIds.AsSpan(i * _tokensPerInput).Clear();
            _attentionMask.AsSpan(i * _tokensPerInput).Clear();
            _outputCorrelation.AsSpan(i).Clear();
        }

        return i != 0;
    }

    public ValueTask DisposeAsync() => _asyncSourceEnumerator?.DisposeAsync() ?? ValueTask.CompletedTask;

    private sealed class AsyncEnumerable(AsyncBatchEnumerator<TKey> parent) :
        IAsyncEnumerator<TokenizedBatch<TKey>>,
        IAsyncEnumerable<TokenizedBatch<TKey>>
    {
        public TokenizedBatch<TKey> Current => parent.Current;

        public ValueTask DisposeAsync() => parent.DisposeAsync();

        public IAsyncEnumerator<TokenizedBatch<TKey>> GetAsyncEnumerator(CancellationToken cancellationToken = default)
            => this;

        public ValueTask<bool> MoveNextAsync() => parent.MoveNextAsync();
    }

    private sealed class SyncEnumerable(AsyncBatchEnumerator<TKey> parent) :
        IEnumerator<TokenizedBatch<TKey>>,
        IEnumerable<TokenizedBatch<TKey>>
    {
        public TokenizedBatch<TKey> Current => parent.Current;

        object IEnumerator.Current => parent.Current;

        public void Dispose()
        {
        }

        public IEnumerator<TokenizedBatch<TKey>> GetEnumerator() => this;

        IEnumerator IEnumerable.GetEnumerator() => this;

        public bool MoveNext()
        {
            var vt = parent.MoveNextAsync();
            if (!vt.IsCompletedSuccessfully)
            {
                throw new NotImplementedException();
            }
            else
            {
                return vt.Result;
            }
        }

        public void Reset() => throw new InvalidOperationException("Multiple enumeration is not supported.");
    }
}

[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1202:Elements should be ordered by access", Justification = "Reviewed.")]
[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.MaintainabilityRules", "SA1402:File may only contain a single type", Justification = "Reviewed.")]
public class TokenizedBatch<TKey>
{
    public ReadOnlyMemory<long> InputIds { get; internal set; }

    public ReadOnlyMemory<long> AttentionMask { get; internal set; }

    public ReadOnlyMemory<TokenizedRange<TKey>?> OutputCorrelation { get; set; }
}
