// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer;

internal class AsyncBatchEnumerator<TKey> : IAsyncEnumerable<TokenizedBatch<TKey>>, IAsyncEnumerator<TokenizedBatch<TKey>>
{
    private readonly BertTokenizer _tokenizer;
    private readonly int _tokensPerInput;
    private readonly int _batchSize;
    private readonly int _stride;
    private readonly long[] _inputIds;
    private readonly long[] _attentionMask;
    private readonly TokenizedRange<TKey>?[] _outputCorrelation;
    private readonly IAsyncEnumerator<(TKey Key, string Content)> _sourceEnumerator;
    private (TKey Key, string Conent, int Offset)? _pivot;
    private bool _gotEnumerator = false;

    internal AsyncBatchEnumerator(BertTokenizer tokenizer, IAsyncEnumerable<(TKey Key, string Content)> sourceEnumerable, int tokensPerInput, int batchSize, int stride)
    {
        _tokenizer = tokenizer;
        _sourceEnumerator = sourceEnumerable.GetAsyncEnumerator();
        _tokensPerInput = tokensPerInput;
        _batchSize = batchSize;
        _stride = stride;
        _inputIds = new long[batchSize * tokensPerInput];
        _attentionMask = new long[batchSize * tokensPerInput];
        _outputCorrelation = new TokenizedRange<TKey>?[batchSize];
        Current = new() { AttentionMask = _attentionMask, InputIds = _inputIds, OutputCorrelation = _outputCorrelation };
    }

    public TokenizedBatch<TKey> Current { get; }

    public IAsyncEnumerator<TokenizedBatch<TKey>> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        if (_gotEnumerator)
        {
            throw new InvalidOperationException("Multiple enumeration of this IAsyncEnumerable is not supported.");
        }

        _gotEnumerator = true;
        return this;
    }

    public async ValueTask<bool> MoveNextAsync()
    {
        var i = 0;
        ReadOnlyMemory<long> strideInputIds = Array.Empty<long>();
        for (i = 0; i < _batchSize; i++)
        {
            if (!_pivot.HasValue)
            {
                if (!await _sourceEnumerator.MoveNextAsync())
                {
                    break;
                }

                _pivot = (_sourceEnumerator.Current.Key, _sourceEnumerator.Current.Content, 0);
            }

            var p = _pivot.Value;
            var corr = _tokenizer.TokenizeBatchElement(
                p.Key,
                p.Conent,
                p.Offset,
                strideInputIds.Span,
                _inputIds.AsSpan(i * _tokensPerInput, _tokensPerInput),
                _attentionMask.AsSpan(i * _tokensPerInput, _tokensPerInput));

            _outputCorrelation[i] = corr;
            _pivot = corr switch
            {
                { LastTokenizedWordStartIndex: int idx } => (p.Key, p.Conent, idx),
                _ => null,
            };

            strideInputIds = _pivot switch
            {
                null => Array.Empty<long>(),
                _ => _inputIds.AsMemory(((i + 1) * _tokensPerInput) - _stride, _stride),
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

    public ValueTask DisposeAsync() => _sourceEnumerator.DisposeAsync();
}

[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1202:Elements should be ordered by access", Justification = "Reviewed.")]
[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.MaintainabilityRules", "SA1402:File may only contain a single type", Justification = "Reviewed.")]
public class TokenizedBatch<TKey>
{
    public ReadOnlyMemory<long> InputIds { get; internal set; }

    public ReadOnlyMemory<long> AttentionMask { get; internal set; }

    public ReadOnlyMemory<TokenizedRange<TKey>?> OutputCorrelation { get; set; }
}
