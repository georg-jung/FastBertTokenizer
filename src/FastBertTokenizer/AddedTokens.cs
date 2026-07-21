// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;

namespace FastBertTokenizer;

internal class AddedTokens
{
    public const byte CharClassWhitespace = 1;
    public const byte CharClassPunctuationOrChinese = 2;
    public const byte CharClassAddedTokenFirstLetter = 4;

    public AddedTokens(IEnumerable<(string Content, bool Normalize)> addedTokens)
    {
        Tokens = [.. addedTokens];

        // This logic might not be perfect. Are there chars that are equal to others in an invariant case insesitive comparison
        // but are neither the upper nor the lower variant of the original?
        var firstLettersToSearch = addedTokens
            .SelectMany(x => x.Normalize
                ? (IEnumerable<char>)[x.Content[0], char.ToLowerInvariant(x.Content[0]), char.ToUpperInvariant(x.Content[0])]
                : [x.Content[0]])
            .Distinct();
#if NET8_0_OR_GREATER
        FirstLetters = SearchValues.Create([.. firstLettersToSearch]);
#else
        FirstLetters = [.. firstLettersToSearch];
#endif

        // Classify every BMP char once so that the pre-tokenizer's per-char hot loop
        // is a single table lookup instead of multiple class checks per char.
        var charClasses = new byte[char.MaxValue + 1];
        for (var i = 0; i <= char.MaxValue; i++)
        {
            var c = (char)i;
            byte cls = 0;
            if (char.IsWhiteSpace(c))
            {
                cls |= CharClassWhitespace;
            }

            if (PreTokenizingEnumerator.IsPunctuation(c) || PreTokenizingEnumerator.IsChineseCharacter(c))
            {
                cls |= CharClassPunctuationOrChinese;
            }

            charClasses[i] = cls;
        }

        foreach (var (content, normalize) in Tokens)
        {
            var c = content[0];
            charClasses[c] |= CharClassAddedTokenFirstLetter;
            if (normalize)
            {
                charClasses[char.ToLowerInvariant(c)] |= CharClassAddedTokenFirstLetter;
                charClasses[char.ToUpperInvariant(c)] |= CharClassAddedTokenFirstLetter;
            }
        }

        CharClasses = charClasses;
    }

    public (string Content, bool Normalize)[] Tokens { get; }

    /// <summary>
    /// Gets a per-char classification table for all BMP chars, built from the
    /// <see cref="CharClassWhitespace"/>, <see cref="CharClassPunctuationOrChinese"/> and
    /// <see cref="CharClassAddedTokenFirstLetter"/> flags. A value of 0 means the char is an
    /// ordinary word char.
    /// </summary>
    public byte[] CharClasses { get; }

#if NET8_0_OR_GREATER
    public SearchValues<char> FirstLetters { get; }
#else
    public char[] FirstLetters { get; }
#endif
}
