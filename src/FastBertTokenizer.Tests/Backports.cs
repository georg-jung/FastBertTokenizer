// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Threading.Channels;

namespace FastBertTokenizer;

#if NETFRAMEWORK
internal static class Backports
{
    public static void Deconstruct<TKey, TValue>(this KeyValuePair<TKey, TValue> kvp, out TKey key, out TValue value)
    {
        key = kvp.Key;
        value = kvp.Value;
    }
}

[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.MaintainabilityRules", "SA1402:File may only contain a single type", Justification = "Keep the backports together")]
internal static class ChannelReaderExtensions
{
    public static IAsyncEnumerable<T> AsAsyncEnumerable<T>(this ChannelReader<T> reader)
    {
        if (reader == null)
        {
            throw new ArgumentNullException(nameof(reader));
        }

        return new ChannelReaderAsyncEnumerable<T>(reader);
    }

    // This is probably rather naive and may have bugs but it is for testing only and seems sufficient at this point.
    private class ChannelReaderAsyncEnumerable<T> : IAsyncEnumerable<T>
    {
        private readonly ChannelReader<T> _reader;

        public ChannelReaderAsyncEnumerable(ChannelReader<T> reader)
        {
            _reader = reader;
        }

        public IAsyncEnumerator<T> GetAsyncEnumerator(CancellationToken cancellationToken = default)
        {
            return new ChannelReaderAsyncEnumerator(_reader, cancellationToken);
        }

        private class ChannelReaderAsyncEnumerator : IAsyncEnumerator<T>
        {
            private readonly ChannelReader<T> _reader;
            private readonly CancellationToken _cancellationToken;

            public ChannelReaderAsyncEnumerator(ChannelReader<T> reader, CancellationToken cancellationToken)
            {
                _reader = reader;
                _cancellationToken = cancellationToken;
                Current = default!;
            }

            public T Current { get; private set; }

            public async ValueTask<bool> MoveNextAsync()
            {
                if (_reader.Completion.IsCompleted)
                {
                    return false;
                }

                Current = await _reader.ReadAsync();
                return true;
            }

            public ValueTask DisposeAsync()
            {
                return default;
            }
        }
    }
}
#endif
