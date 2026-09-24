// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace FastBertTokenizer;

/// <summary>
/// A read-only hash table that maps the tokens of a vocabulary to their ids and is queried with spans. It works the
/// same on all target frameworks and is tailored to what <see cref="BertTokenizer"/> does with it: very many lookups
/// of short keys. All keys are stored back to back in one char array and the table is one array of entries that is
/// probed linearly, thus a lookup typically touches just two cache lines and calls neither delegates nor virtual methods.
/// </summary>
/// <remarks>
/// The hash function is not randomized. This is not a hash flooding risk as the table is never modified after it
/// was created: how many entries a lookup probes depends only on the vocabulary, which is trusted, and not on the
/// input that is tokenized.
/// </remarks>
internal sealed class VocabDictionary : IEnumerable<KeyValuePair<string, long>>
{
    private readonly Entry[] _entries;
    private readonly char[] _keys;
    private readonly int _shift;
    private readonly int? _emptyKeyValue;

    /// <summary>
    /// Initializes a new instance of the <see cref="VocabDictionary"/> class.
    /// </summary>
    /// <param name="source">The tokens and their ids.</param>
    public VocabDictionary(Dictionary<string, int> source)
    {
        var keyChars = 0;
        foreach (var key in source.Keys)
        {
            keyChars += key.Length;
            MaxKeyLength = Math.Max(MaxKeyLength, key.Length);
        }

        // Using at most half of the entries keeps the probe sequences short.
        var log2Size = 2;
        while ((1 << log2Size) < source.Count * 2)
        {
            log2Size++;
        }

        _entries = new Entry[1 << log2Size];
        _shift = 64 - log2Size;
        _keys = new char[keyChars];

        var mask = _entries.Length - 1;
        var keyStart = 0;
        foreach (var kvp in source)
        {
            var key = kvp.Key;
            if (key.Length == 0)
            {
                // An entry with length 0 marks an empty slot. The tokenizer never looks up an empty key,
                // it is just kept to be enumerated.
                _emptyKeyValue = kvp.Value;
                continue;
            }

            key.CopyTo(0, _keys, keyStart, key.Length);
            var hash = GetHashCode(key.AsSpan());
            var i = (int)(hash >> _shift);
            while (_entries[i].Length != 0)
            {
                i = (i + 1) & mask;
            }

            _entries[i] = new Entry((uint)hash, key.Length, keyStart, kvp.Value);
            keyStart += key.Length;
        }
    }

    /// <summary>
    /// Gets the length of the longest key, or 0 if there are no keys.
    /// </summary>
    public int MaxKeyLength { get; }

    /// <summary>
    /// Gets the id of the given token.
    /// </summary>
    /// <param name="key">The token to look up. Must not be empty.</param>
    /// <param name="value">The id of the token if it was found.</param>
    /// <returns>True if the token was found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(ReadOnlySpan<char> key, out int value)
    {
        var hash = GetHashCode(key);
        var entries = _entries;
        var mask = entries.Length - 1;
        var i = (int)(hash >> _shift);
        while (true)
        {
            ref var entry = ref entries[i];
            if (entry.Length == 0)
            {
                value = 0;
                return false;
            }

            if (entry.HashCode == (uint)hash
                && entry.Length == key.Length
                && key.SequenceEqual(_keys.AsSpan(entry.KeyStart, entry.Length)))
            {
                value = entry.Value;
                return true;
            }

            i = (i + 1) & mask;
        }
    }

    /// <inheritdoc/>
    public IEnumerator<KeyValuePair<string, long>> GetEnumerator()
    {
        if (_emptyKeyValue is int emptyKeyValue)
        {
            yield return new(string.Empty, emptyKeyValue);
        }

        foreach (var entry in _entries)
        {
            if (entry.Length != 0)
            {
                yield return new(new string(_keys, entry.KeyStart, entry.Length), entry.Value);
            }
        }
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>
    /// A fast hash for short strings: Instead of processing the key char by char it reads it in chunks of
    /// four chars, whereby the first and the last chunk overlap if the length is no multiple of four.
    /// </summary>
    /// <param name="key">The key to hash.</param>
    /// <returns>The hash code. The table index is taken from the high bits, the stored hash code from the low bits.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong GetHashCode(ReadOnlySpan<char> key)
    {
        const ulong K1 = 0x9E3779B97F4A7C15;
        const ulong K2 = 0xC2B2AE3D27D4EB4F;

        ref var start = ref Unsafe.As<char, byte>(ref MemoryMarshal.GetReference(key));
        var byteLength = key.Length * sizeof(char);
        ulong first, last;
        if (key.Length >= 4)
        {
            first = Unsafe.ReadUnaligned<ulong>(ref start);
            last = Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref start, byteLength - sizeof(ulong)));
            for (var i = sizeof(ulong); i < byteLength - sizeof(ulong); i += sizeof(ulong))
            {
                first *= K1;
                first = ((first << 29) | (first >> 35)) ^ Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref start, i));
            }
        }
        else if (key.Length >= 2)
        {
            first = Unsafe.ReadUnaligned<uint>(ref start);
            last = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref start, byteLength - sizeof(uint)));
        }
        else
        {
            first = key.Length == 0 ? 0UL : key[0];
            last = 0;
        }

        var hash = ((first ^ K2) * K1) ^ ((last + (ulong)key.Length) * K2);
        return hash ^ (hash >> 32);
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct Entry(uint hashCode, int length, int keyStart, int value)
    {
        public readonly uint HashCode = hashCode;
        public readonly int Length = length;
        public readonly int KeyStart = keyStart;
        public readonly int Value = value;
    }
}
