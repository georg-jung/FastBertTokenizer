// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

/*
mostly copied from https://github.com/dotnet/machinelearning/blob/72cfdf611a510ba0570170a708ddcc1a1928f329/src/Microsoft.ML.Tokenizers/Utils/StringSpanOrdinalKey.cs
*/

#if NET8_0_OR_GREATER
using System.Collections.Frozen;
#endif

namespace FastBertTokenizer;

/// <summary>Used as a key in a dictionary to enable querying with either a string or a span.</summary>
/// <remarks>
/// This should only be used with a Ptr/Length for querying. For storing in a dictionary, this should
/// always be used with a string.
/// </remarks>
internal unsafe readonly struct StringSpanOrdinalKey : IEquatable<StringSpanOrdinalKey>
{
    public readonly char* Ptr;
    public readonly int Length;
    public readonly string? Data;

    public StringSpanOrdinalKey(char* ptr, int length)
    {
        Ptr = ptr;
        Length = length;
    }

    public StringSpanOrdinalKey(string data) =>
        Data = data;

    private ReadOnlySpan<char> Span => Ptr is not null ?
        new ReadOnlySpan<char>(Ptr, Length) :
        Data.AsSpan();

    public override string ToString() => Data ?? Span.ToString();

    public override bool Equals(object? obj) =>
        obj is StringSpanOrdinalKey wrapper && Equals(wrapper);

    public bool Equals(StringSpanOrdinalKey other) =>
        Span.SequenceEqual(other.Span);

    public override int GetHashCode() => Helpers.GetHashCode(Span);
}

internal static class StringSpanOrdinalKeyDictExtensions
{
    internal static unsafe bool TryGetValue(this Dictionary<StringSpanOrdinalKey, long> dict, ReadOnlySpan<char> key, out long value)
    {
        fixed (char* ptr = key)
        {
            return dict.TryGetValue(new StringSpanOrdinalKey(ptr, key.Length), out value!);
        }
    }

#if NET8_0_OR_GREATER
    internal static unsafe bool TryGetValue(this FrozenDictionary<StringSpanOrdinalKey, long> dict, ReadOnlySpan<char> key, out long value)
    {
        fixed (char* ptr = key)
        {
            return dict.TryGetValue(new StringSpanOrdinalKey(ptr, key.Length), out value!);
        }
    }
#endif
}
