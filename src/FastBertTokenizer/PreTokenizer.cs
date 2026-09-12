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
    private readonly (string Content, bool Normalize)[] _addedTokens;
#if NET8_0_OR_GREATER
    private readonly SearchValues<char> _addedTokenFirstLetters;
#else
    private readonly char[] _addedTokenFirstLetters;
#endif
    private readonly CharClass[] _asciiCharClasses;

    public PreTokenizer(IEnumerable<(string Content, bool Normalize)> addedTokens)
    {
        _addedTokens = [.. addedTokens];

        // This logic might not be perfect. Are there chars that are equal to others in an invariant case insesitive comparison
        // but are neither the upper nor the lower variant of the original?
        var firstLettersToSearch = addedTokens
            .SelectMany(x => x.Normalize
                ? (IEnumerable<char>)[x.Content[0], char.ToLowerInvariant(x.Content[0]), char.ToUpperInvariant(x.Content[0])]
                : [x.Content[0]])
            .Distinct();
#if NET8_0_OR_GREATER
        _addedTokenFirstLetters = SearchValues.Create([.. firstLettersToSearch]);
#else
        _addedTokenFirstLetters = [.. firstLettersToSearch];
#endif

        // ClassifySlow reads _addedTokenFirstLetters, so the table must be filled after it is assigned.
        _asciiCharClasses = new CharClass[128];
        for (var i = 0; i < _asciiCharClasses.Length; i++)
        {
            _asciiCharClasses[i] = ClassifySlow((char)i);
        }
    }

    /// <summary>
    /// How <see cref="PreTokenizingEnumerator"/> handles a single char. Whitespace and Punctuation are mutually exclusive,
    /// MayStartAddedToken is independent of both.
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
        return c < asciiCharClasses.Length ? asciiCharClasses[c] : ClassifySlow(c);
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
    /// The predicate based classification that <see cref="Classify"/> uses for non-ASCII chars
    /// and that the ASCII table is filled from.
    /// </summary>
    /// <param name="c">Char to classify.</param>
    /// <returns>The class of the given char.</returns>
    private CharClass ClassifySlow(char c)
    {
        var res = char.IsWhiteSpace(c)
            ? CharClass.Whitespace
            : IsPunctuation(c) || IsChineseCharacter(c)
                ? CharClass.Punctuation
                : CharClass.Word;

        if (_addedTokenFirstLetters.Contains(c))
        {
            res |= CharClass.MayStartAddedToken;
        }

        return res;
    }
}
