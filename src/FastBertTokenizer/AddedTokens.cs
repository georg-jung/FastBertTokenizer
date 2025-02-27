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
    }

    public (string Content, bool Normalize)[] Tokens { get; }

#if NET8_0_OR_GREATER
    public SearchValues<char> FirstLetters { get; }
#else
    public char[] FirstLetters { get; }
#endif
}
