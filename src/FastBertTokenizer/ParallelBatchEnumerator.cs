// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer;

internal class ParallelBatchEnumerator<TKey>
    : IAsyncEnumerator<TokenizedBatch<TKey>>, IAsyncEnumerable<TokenizedBatch<TKey>>
{
    private readonly IAsyncEnumerator<TokenizedBatch<TKey>>?[] _enumerators;
    private Task<bool>[] _tasks;
    private int _currentIndex = -1;

    public ParallelBatchEnumerator(int maxDegreeOfParallelism, Func<IAsyncEnumerator<TokenizedBatch<TKey>>> enumeratorFactory)
    {
        _enumerators = new IAsyncEnumerator<TokenizedBatch<TKey>>[maxDegreeOfParallelism];
        for (var i = 0; i < maxDegreeOfParallelism; i++)
        {
            _enumerators[i] = enumeratorFactory();
        }
    }

    public TokenizedBatch<TKey> Current => _currentIndex switch
    {
        -1 => throw new InvalidOperationException("Call MoveNextAsync() first."),
        _ => _enumerators[_currentIndex]?.Current
            ?? throw new InvalidOperationException("The last call to MoveNextAsync returned false, thus, Current is not accessible."),
    };

    public async ValueTask DisposeAsync()
    {
        for (var i = 0; i < _enumerators.Length; i++)
        {
            if (_enumerators[i] is IAsyncEnumerator<TokenizedBatch<TKey>> e)
            {
                await e.DisposeAsync();
            }
        }
    }

    public IAsyncEnumerator<TokenizedBatch<TKey>> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        if (_tasks is not null)
        {
            throw new InvalidOperationException("GetAsyncEnumerator() can only be called once.");
        }

        _tasks = new Task<bool>[_enumerators.Length];
        for (var i = 0; i < _tasks.Length; i++)
        {
            var e = _enumerators[i];
            _tasks[i] = Task.Run(async () => await e.MoveNextAsync());
        }

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
