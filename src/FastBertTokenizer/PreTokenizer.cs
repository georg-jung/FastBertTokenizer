// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;

namespace FastBertTokenizer;

public class PreTokenizer
{
    /* https://github.com/NMZivkovic/BertTokenizers defines the following punctuation:
     * ".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray()
     */

    private static char[] morePunctuation = "$=+`<>^|~".ToArray();

    public delegate bool ReadOnlySpanFunc<T>(ReadOnlySpan<T> span);

    /// <summary>
    /// Pre-tokenize text input. Turns sentences into words and punctuation, removes whitespace. Allocation free.
    /// </summary>
    /// <param name="input">Input to pre-tokenize.</param>
    /// <param name="processToken">A function to process the next token. Pre-tokenization is stopped as soon as the function returns false.</param>
    public static void PreTokenize(ReadOnlySpan<char> input, ReadOnlySpanFunc<char> processToken)
    {
        var start = -1;
        for (var i = 0; i < input.Length; i++)
        {
            bool Flush(ReadOnlySpan<char> input)
            {
                if (start != -1)
                {
                    var ret = processToken(input.Slice(start, i - start));
                    start = -1;
                    return ret;
                }

                return true;
            }

            var c = input[i];
            if (char.IsWhiteSpace(c))
            {
                if (!Flush(input))
                {
                    return;
                }
            }
            else if (char.IsPunctuation(c) || Array.IndexOf(morePunctuation, c) != -1)
            {
                if (!Flush(input) || !processToken(input.Slice(i, 1)))
                {
                    return;
                }
            }
            else
            {
                if (start == -1)
                {
                    start = i;
                }
            }
        }
    }
}
