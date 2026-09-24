// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace FastBertTokenizer;

/// <summary>
/// The per-tokenizer part of pre-tokenization: the added tokens and the classification of chars into word,
/// whitespace and punctuation that <see cref="PreTokenizingEnumerator"/> splits the input on. Built once
/// per tokenizer and shared by all inputs.
/// </summary>
internal class PreTokenizer
{
    /// <summary>
    /// A cache of the classes of non-ASCII chars that don't depend on the added tokens, indexed by char and shared by all
    /// instances. It is filled lazily: <see cref="CharClass.NonAscii"/> is set in all entries that were already computed.
    /// Concurrent writes are benign, as all of them write the same value.
    /// </summary>
    private static readonly CharClass[] NonAsciiCharClasses = new CharClass[char.MaxValue + 1];

    /// <summary>
    /// A cache of <see cref="char.ToLowerInvariant(char)"/> for non-ASCII chars, indexed by char and shared by all instances.
    /// It is filled lazily: entries that were not computed yet are 0. Concurrent writes are benign, as all of them write the same value.
    /// </summary>
    private static readonly char[] NonAsciiLowercase = new char[char.MaxValue + 1];

    private readonly (string Content, bool Normalize)[] _addedTokens;
#if NET8_0_OR_GREATER
    private readonly SearchValues<char> _addedTokenFirstLetters;
#else
    private readonly char[] _addedTokenFirstLetters;
#endif
    private readonly CharClass[] _asciiCharClasses;
    private readonly bool _hasNonAsciiAddedTokenFirstLetters;

    public PreTokenizer(IEnumerable<(string Content, bool Normalize)> addedTokens)
    {
        _addedTokens = [.. addedTokens];

        // This logic might not be perfect. Are there chars that are equal to others in an invariant case insesitive comparison
        // but are neither the upper nor the lower variant of the original?
        char[] firstLettersToSearch = [.. addedTokens
            .SelectMany(x => x.Normalize
                ? (IEnumerable<char>)[x.Content[0], char.ToLowerInvariant(x.Content[0]), char.ToUpperInvariant(x.Content[0])]
                : [x.Content[0]])
            .Distinct()];
#if NET8_0_OR_GREATER
        _addedTokenFirstLetters = SearchValues.Create(firstLettersToSearch);
#else
        _addedTokenFirstLetters = firstLettersToSearch;
#endif
        _hasNonAsciiAddedTokenFirstLetters = firstLettersToSearch.Any(c => c >= 128);

        // ClassifySlow reads _addedTokenFirstLetters, so the table must be filled after it is assigned.
        _asciiCharClasses = new CharClass[128];
        for (var i = 0; i < _asciiCharClasses.Length; i++)
        {
            _asciiCharClasses[i] = ClassifySlow((char)i);
        }

        for (var c = 'A'; c <= 'Z'; c++)
        {
            _asciiCharClasses[c] |= CharClass.AsciiUppercase;
        }

        AsciiAlphanumericsAreWordChars = true;
        for (var c = '0'; c <= 'z'; c++)
        {
            if (char.IsLetterOrDigit(c) && (_asciiCharClasses[c] & ~CharClass.AsciiUppercase) != CharClass.Word)
            {
                AsciiAlphanumericsAreWordChars = false;
            }
        }
    }

    /// <summary>
    /// How <see cref="PreTokenizingEnumerator"/> handles a single char. Whitespace and Punctuation are mutually exclusive,
    /// MayStartAddedToken is independent of both. AsciiUppercase and NonAscii just tell how to lowercase a word.
    /// </summary>
    [Flags]
    internal enum CharClass : byte
    {
        /// <summary>Part of a word, i.e. it belongs to the segment that is currently being read.</summary>
        Word = 0,

        /// <summary>Whitespace, i.e. it ends the current segment and is dropped itself.</summary>
        Whitespace = 1,

        /// <summary>Punctuation or a chinese char, i.e. it is a segment of its own.</summary>
        Punctuation = 2,

        /// <summary>The char is a first letter of one of the added tokens, thus an added token might start here.</summary>
        MayStartAddedToken = 4,

        /// <summary>The char is an ASCII uppercase letter, i.e. lowercasing changes it.</summary>
        AsciiUppercase = 8,

        /// <summary>The char is not ASCII, i.e. it is not classified by the ASCII lookup table.</summary>
        NonAscii = 16,

        /// <summary>The classes that end a word.</summary>
        EndsWord = Whitespace | Punctuation | MayStartAddedToken,
    }

    /// <summary>
    /// Gets the classes of the ASCII chars, indexed by char. Chars that are not contained in this table need to be
    /// classified by <see cref="ClassifyNonAscii(char)"/>.
    /// </summary>
    public CharClass[] AsciiCharClasses => _asciiCharClasses;

    /// <summary>
    /// Gets a value indicating whether all ASCII letters and digits are plain word chars, i.e. none of them might start an added token.
    /// </summary>
    public bool AsciiAlphanumericsAreWordChars { get; }

    /// <summary>
    /// Lowercases the given chars like <see cref="MemoryExtensions.ToLowerInvariant(ReadOnlySpan{char}, Span{char})"/> does,
    /// but char by char using a cache. This yields the same result as long as there are no surrogate pairs, which are
    /// lowercased as a whole.
    /// </summary>
    /// <param name="source">The chars to lowercase.</param>
    /// <param name="destination">Where to write the lowercased chars to. Must be at least as long as <paramref name="source"/>.</param>
    /// <returns>False if <paramref name="source"/> contains surrogates. <paramref name="destination"/> is incomplete then.</returns>
    public static bool TryToLowerInvariantWithoutSurrogates(ReadOnlySpan<char> source, Span<char> destination)
    {
        var lowercase = NonAsciiLowercase;
        for (var i = 0; i < source.Length; i++)
        {
            var c = source[i];
            if (c < 128)
            {
                destination[i] = (char)(c | ((uint)(c - 'A') <= 'Z' - 'A' ? 0x20 : 0));
                continue;
            }

            if (char.IsSurrogate(c))
            {
                return false;
            }

            var lower = lowercase[c];
            if (lower == 0)
            {
                lower = char.ToLowerInvariant(c);
                lowercase[c] = lower;
            }

            destination[i] = lower;
        }

        return true;
    }

    /// <summary>
    /// Splits the given input into the segments that are subsequently tokenized into word pieces.
    /// </summary>
    /// <param name="input">The input to split.</param>
    /// <param name="convertToLowercase">Whether the segments should be lowercased.</param>
    /// <param name="vocabNf">The normalization form of the vocabulary.</param>
    /// <param name="inputOffset">The index in <paramref name="input"/> to start at.</param>
    /// <returns>An enumerator over the segments of the input.</returns>
    public PreTokenizingEnumerator PreTokenize(string input, bool convertToLowercase, NormalizationForm vocabNf, int inputOffset = 0)
        => new(input, convertToLowercase, vocabNf, this, inputOffset);

    /// <summary>
    /// Determines how <see cref="PreTokenizingEnumerator"/> handles the given char.
    /// </summary>
    /// <param name="c">Char to classify.</param>
    /// <returns>The class of the given char.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public CharClass Classify(char c)
    {
        // ASCII chars typically make up the vast majority of the input. For them, this single lookup replaces the
        // whitespace, punctuation, chinese char and added token first letter checks that all other chars go through.
        var asciiCharClasses = _asciiCharClasses;
        return c < asciiCharClasses.Length ? asciiCharClasses[c] : ClassifyNonAscii(c);
    }

    /// <summary>
    /// Determines how <see cref="PreTokenizingEnumerator"/> handles the given char, which is not contained in <see cref="AsciiCharClasses"/>.
    /// </summary>
    /// <param name="c">Char to classify.</param>
    /// <returns>The class of the given char.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public CharClass ClassifyNonAscii(char c)
    {
        var res = NonAsciiCharClasses[c];
        if (res == 0)
        {
            res = ClassifyNonAsciiUncached(c);
        }

        if (_hasNonAsciiAddedTokenFirstLetters && _addedTokenFirstLetters.Contains(c))
        {
            res |= CharClass.MayStartAddedToken;
        }

        return res;
    }

    /// <summary>
    /// Checks whether the given value starts with one of the added tokens. Only worth calling if the first char
    /// of the value is classified as <see cref="CharClass.MayStartAddedToken"/>.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>The length of the matching added token and whether it is normalized, or null if there is no match.</returns>
    public (int Length, bool Normalize)? StartsWithAddedToken(ReadOnlySpan<char> value)
    {
        foreach (var (content, normalize) in _addedTokens)
        {
            if (value.StartsWith(content.AsSpan(), normalize ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
            {
                return (content.Length, normalize);
            }
        }

        return null;
    }

    // AggressiveInlining the methods below does seem to provide a performance boost in the 2-6% ballpark of the overall tokenizer.

    /// <summary>
    /// Translated from https://github.com/huggingface/transformers/blob/05de038f3d249ce96740885f85fd8d0aa00c29bc/src/transformers/tokenization_utils.py#L292-L304.
    /// </summary>
    /// <param name="cp">Character to check.</param>
    /// <returns>True we consider the character a punctuation character, otherwise false.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsPunctuation(char cp)
    {
        // We treat all non-letter/number ASCII as punctuation.
        // Characters such as "^", "$", and "`" are not in the Unicode
        // Punctuation class but we treat them as punctuation anyways, for
        // consistency.
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
        {
            return true;
        }

#if NETSTANDARD
        // inpired by / taken from source of modern .net
        static bool IsBetween(UnicodeCategory c, UnicodeCategory min, UnicodeCategory max) =>
            (uint)(c - min) <= (uint)(max - min);

        // char.GetUnicodeCategory(c); returns wrong values for some chars on netframework
        return IsBetween(CharUnicodeInfo.GetUnicodeCategory(cp), UnicodeCategory.ConnectorPunctuation, UnicodeCategory.OtherPunctuation);
#else
        return char.IsPunctuation(cp);
#endif
    }

    /// <summary>
    /// Tranlated from https://github.com/huggingface/transformers/blob/32ec7345f2d752c294ddf5aff495b657c9cd9d3b/src/transformers/models/bert/tokenization_bert.py#L495-L517.
    /// </summary>
    /// <param name="cp">Char to check.</param>
    /// <returns>True if passed char is a chinese char.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsChineseCharacter(char cp)
    {
        // This defines a "chinese character" as anything in the CJK Unicode block:
        //   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        //
        // Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        // despite its name. The modern Korean Hangul alphabet is a different block,
        // as is Japanese Hiragana and Katakana. Those alphabets are used to write
        // space-separated words, so they are not treated specially and handled
        // like all of the other languages.
#pragma warning disable SA1025 // Code should not contain multiple whitespace in a row
#pragma warning disable S2198 //  Comparison to this constant is useless; the constant is outside the range of type 'char'. They are not useless, the rule is just wrong.
#pragma warning disable SA1108 // BlockStatementsMustNotContainEmbeddedComments. Better readable here.
        if (
            (cp >= 0x4E00 && cp <= 0x9FFF)       // CJK Unified Ideographs
            || (cp >= 0x3400 && cp <= 0x4DBF)    // CJK Unified Ideographs Extension A
            || (cp >= 0x20000 && cp <= 0x2A6DF)  // CJK Unified Ideographs Extension B
            || (cp >= 0x2A700 && cp <= 0x2B73F)  // CJK Unified Ideographs Extension C
            || (cp >= 0x2B740 && cp <= 0x2B81F)  // CJK Unified Ideographs Extension D
            || (cp >= 0x2B820 && cp <= 0x2CEAF)  // CJK Unified Ideographs Extension E
            || (cp >= 0xF900 && cp <= 0xFAFF)    // CJK Compatibility Ideographs
            || (cp >= 0x2F800 && cp <= 0x2FA1F)) // CJK Compatibility Ideographs Supplement
        {
            return true;
        }
#pragma warning restore SA1108 // BlockStatementsMustNotContainEmbeddedComments
#pragma warning restore S2198 //  Comparison to this constant is useless; the constant is outside the range of type 'char'.
#pragma warning restore SA1025 // Code should not contain multiple whitespace in a row

        return false;
    }

    /// <summary>
    /// Classifies the given non-ASCII char apart from whether it might start an added token, and caches the result.
    /// </summary>
    /// <param name="c">Char to classify.</param>
    /// <returns>The class of the given char.</returns>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static CharClass ClassifyNonAsciiUncached(char c)
    {
        var res = ClassifyWithoutAddedTokens(c) | CharClass.NonAscii;
        NonAsciiCharClasses[c] = res;
        return res;
    }

    /// <summary>
    /// The predicate based classification of a char, apart from whether it might start an added token.
    /// </summary>
    /// <param name="c">Char to classify.</param>
    /// <returns>The class of the given char.</returns>
    private static CharClass ClassifyWithoutAddedTokens(char c)
        => char.IsWhiteSpace(c)
            ? CharClass.Whitespace
            : IsPunctuation(c) || IsChineseCharacter(c)
                ? CharClass.Punctuation
                : CharClass.Word;

    /// <summary>
    /// The predicate based classification that the ASCII table is filled from.
    /// </summary>
    /// <param name="c">Char to classify.</param>
    /// <returns>The class of the given char.</returns>
    private CharClass ClassifySlow(char c)
    {
        var res = ClassifyWithoutAddedTokens(c);
        if (_addedTokenFirstLetters.Contains(c))
        {
            res |= CharClass.MayStartAddedToken;
        }

        return res;
    }
}
