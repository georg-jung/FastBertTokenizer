using System.Threading.Channels;
using FastBertTokenizer;

namespace Benchmarks;

// These will be used if they are not implemented in the referenced FastBertTokenizer library.
// The real ones will be used if they are available as they take precedence over extension methods.
public static class NotImplementedExtensions {
    public static IAsyncEnumerable<TokenizedBatch<TKey>> CreateAsyncBatchEnumerator<TKey>(this BertTokenizer tok, ChannelReader<(TKey Key, string Content)> sourceChannel, int tokensPerInput, int batchSize, int stride)
        => throw new NotImplementedException();

    public static IAsyncEnumerable<TokenizedBatch<TKey>> CreateAsyncBatchEnumerator<TKey>(this BertTokenizer tok, IAsyncEnumerable<(TKey Key, string Content)> sourceChannel, int tokensPerInput, int batchSize, int stride)
        => throw new NotImplementedException();
}


public class TokenizedBatch<TKey>
{
    public ReadOnlyMemory<long> InputIds { get; internal set; }

    public ReadOnlyMemory<long> AttentionMask { get; internal set; }

    public ReadOnlyMemory<TokenizedRange<TKey>?> OutputCorrelation { get; set; }
}

public record struct TokenizedRange<TKey>(TKey Key, int Offset, int? LastTokenizedWordStartIndex)
{
}
