// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;

namespace FastBertTokenizer;

internal class AddedTokens
{
    public AddedTokens(IEnumerable<(string Content, bool Normalize)> addedTokens)
    {
        Tokens = [.. addedTokens];
#if NET8_0_OR_GREATER
        FirstLetters = SearchValues.Create([.. addedTokens.Select(x => x.Content[0])]);
#else
        FirstLetters = [.. addedTokens.Select(x => x.Content[0])];
#endif
    }

    public (string Content, bool Normalize)[] Tokens { get; }

#if NET8_0_OR_GREATER
    public SearchValues<char> FirstLetters { get; }
#else
    public char[] FirstLetters { get; }
#endif
}
