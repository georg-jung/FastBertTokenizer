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
        int i;

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

        for (i = 0; i < input.Length; i++)
        {
            var c = input[i];

            // HuggingFace tokenizer would remove 0xFFFD (REPLACEMENT CHARACTER) and control chars here.
            // That would require us to allocate memory, which we don't want to do.
            // https://github.com/huggingface/transformers/blob/7db1ad63d9a9a8f705e13d68f90269df78a16df5/src/transformers/models/bert/tokenization_bert.py#L519

            if (char.IsWhiteSpace(c))
            {
                if (!Flush(input))
                {
                    return;
                }
            }
            else if (char.IsPunctuation(c) || Array.IndexOf(morePunctuation, c) != -1 || IsChineseCharacter(c))
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

        Flush(input);
    }

    /// <summary>
    /// Tranlated from https://github.com/huggingface/transformers/blob/32ec7345f2d752c294ddf5aff495b657c9cd9d3b/src/transformers/models/bert/tokenization_bert.py#L495-L517.
    /// </summary>
    /// <param name="cp">Char to check.</param>
    /// <returns>True if passed char is a chinese char.</returns>
    public static bool IsChineseCharacter(char cp)
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
        if (
            (cp >= 0x4E00 && cp <= 0x9FFF)       // CJK Unified Ideographs
            || (cp >= 0x3400 && cp <= 0x4DBF)    // CJK Unified Ideographs Extension A
            || (cp >= 0x20000 && cp <= 0x2A6DF)  // CJK Unified Ideographs Extension B
            || (cp >= 0x2A700 && cp <= 0x2B73F)  // CJK Unified Ideographs Extension C
            || (cp >= 0x2B740 && cp <= 0x2B81F)  // CJK Unified Ideographs Extension D
            || (cp >= 0x2B820 && cp <= 0x2CEAF)  // CJK Unified Ideographs Extension E
            || (cp >= 0xF900 && cp <= 0xFAFF)    // CJK Compatibility Ideographs
            || (cp >= 0x2F800 && cp <= 0x2FA1F))  // CJK Compatibility Ideographs Supplement
        {
            return true;
        }
#pragma warning restore SA1025 // Code should not contain multiple whitespace in a row

        return false;
    }
}
