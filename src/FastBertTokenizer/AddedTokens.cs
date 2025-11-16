// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;

namespace FastBertTokenizer;

internal class AddedTokens
{
    public AddedTokens(IEnumerable<(string Content, bool Normalize)> addedTokens)
    {
        Tokens = [.. addedTokens];

        // This logic might not be perfect. Are there chars that are equal to others in an invariant case insesitive comparison
        // but are neither the upper nor the lower variant of the original?
        var firstLettersSet = new HashSet<char>();
        foreach (var (content, normalize) in addedTokens)
        {
            var firstChar = content[0];
            firstLettersSet.Add(firstChar);
            if (normalize)
            {
                firstLettersSet.Add(char.ToLowerInvariant(firstChar));
                firstLettersSet.Add(char.ToUpperInvariant(firstChar));
            }
        }

#if NET8_0_OR_GREATER
        FirstLetters = SearchValues.Create([.. firstLettersSet]);
#else
        FirstLetters = [.. firstLettersSet];
#endif
    }

    public (string Content, bool Normalize)[] Tokens { get; }

#if NET8_0_OR_GREATER
    public SearchValues<char> FirstLetters { get; }
#else
    public char[] FirstLetters { get; }
#endif
}
