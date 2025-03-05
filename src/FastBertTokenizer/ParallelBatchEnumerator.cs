// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace FastBertTokenizer;

[Experimental("FBERTTOK001")]
internal class ParallelBatchEnumerator<TKey>
    : IAsyncEnumerator<TokenizedBatch<TKey>>, IAsyncEnumerable<TokenizedBatch<TKey>>
{
    private readonly IAsyncEnumerator<TokenizedBatch<TKey>>?[] _enumerators;
    private Task<bool>[] _tasks;
    private int _currentIndex = -1;

    public ParallelBatchEnumerator(int maxDegreeOfParallelism, Func<IAsyncEnumerator<TokenizedBatch<TKey>>> enumeratorFactory)
    {
        _enumerators = Enumerable.Range(0, maxDegreeOfParallelism)
            .Select(_ => enumeratorFactory())
            .ToArray();
    }

    public TokenizedBatch<TKey> Current => _currentIndex switch
    {
        -1 => throw new InvalidOperationException("Call MoveNextAsync() first."),
        _ => _enumerators[_currentIndex]?.Current
            ?? throw new InvalidOperationException("The last call to MoveNextAsync returned false, thus, Current is not accessible."),
    };

    public async ValueTask DisposeAsync()
    {
        foreach (var asyncEnumerator in _enumerators.OfType<IAsyncEnumerator<TokenizedBatch<TKey>>>())
        {
            await asyncEnumerator.DisposeAsync();
        }
    }

    public IAsyncEnumerator<TokenizedBatch<TKey>> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        if (_tasks is not null)
        {
            throw new InvalidOperationException("GetAsyncEnumerator() can only be called once.");
        }

        _tasks = _enumerators
            .Select(e => Task.Run(async () => await e.MoveNextAsync(), cancellationToken))
            .ToArray();

        return this;
    }

    public async ValueTask<bool> MoveNextAsync()
    {
        if (_currentIndex != -1 && _enumerators[_currentIndex] is IAsyncEnumerator<TokenizedBatch<TKey>> e)
        {
            _tasks[_currentIndex] = Task.Run(async () => await e.MoveNextAsync());
        }

        var tries = 0;
        do
        {
            _currentIndex = (_currentIndex + 1) % (_enumerators.Length - 1);
            tries++;
        }
        while (_enumerators[_currentIndex] is null && tries < _enumerators.Length);

        if (tries == _enumerators.Length)
        {
            return false;
        }

        var res = await _tasks[_currentIndex];
        if (res)
        {
            return true;
        }

        _enumerators[_currentIndex] = null;
        return await MoveNextAsync();
    }
}
