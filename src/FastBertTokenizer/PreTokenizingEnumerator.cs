// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace FastBertTokenizer;

internal ref struct PreTokenizingEnumerator
{
    private readonly bool _convertToLowercase;
    private readonly int _inputOffset;
    private readonly ReadOnlySpan<char> _input;
    private readonly AddedTokens _addedTokens;
    private int start;
    private int currentIndex;
    private char[]? buffer = null;

    public PreTokenizingEnumerator(string input, bool convertToLowercase, NormalizationForm vocabNf, AddedTokens addedTokens, int inputOffset = 0)
    {
        // The BertTokenizer itself will try normalizing the string if it can not find a matching token id in the vocabulary.
        // If the vocabulary uses FormD and our input is in FormC, we will not find a matching token id for the input as the composed
        // chars are not contained in the dictionary then. Thus, we don't need to normalize and copy memory here, as we can handle
        // this for the individual tokens later on.
        // If the vocabulary uses FormC and our input is in FormD, we need to normalize here, as we might be able to encode the
        // FormD variant while there might be a more specific FormC vocabulary match which we couldn't find due to wrong normalization.
        // The KC and KD variants need to be normalized here as well.
        if (vocabNf == NormalizationForm.FormD || input.IsNormalized(vocabNf))
        {
            _input = input.AsSpan(inputOffset);
        }
        else
        {
            _input = input.Normalize(vocabNf).AsSpan(inputOffset);
        }

        _convertToLowercase = convertToLowercase;
        _addedTokens = addedTokens;
        _inputOffset = inputOffset;
        start = -1;
        currentIndex = 0;
        if (_convertToLowercase)
        {
            buffer = ArrayPool<char>.Shared.Rent(64);
        }
    }

    public PreTokenizerResult Current { get; private set; }

    public readonly PreTokenizingEnumerator GetEnumerator() => this;

    public bool MoveNext()
    {
        while (currentIndex < _input.Length)
        {
            var c = _input[currentIndex];

            if (_addedTokens.FirstLetters.Contains(c) && StartsWithAddedToken(_input.Slice(currentIndex)) is (int len, bool normalize))
            {
                if (Flush())
                {
                    return true;
                }

                SetCurrent(_input.Slice(currentIndex, len), currentIndex, normalize);
                currentIndex += len;
                return true;
            }
            else if (char.IsWhiteSpace(c))
            {
                if (Flush())
                {
                    currentIndex++;
                    return true;
                }

                currentIndex++;
            }
            else if (IsPunctuation(c) || IsChineseCharacter(c))
            {
                if (Flush())
                {
                    return true;
                }

                Current = new() { Segment = _input.Slice(currentIndex, 1), SegmentStartIndex = currentIndex + _inputOffset };
                currentIndex++;
                return true;
            }
            else
            {
                if (start == -1)
                {
                    start = currentIndex;
                }

                currentIndex++;
            }
        }

        return Flush();
    }

    public void Dispose()
    {
        if (buffer is not null)
        {
            ArrayPool<char>.Shared.Return(buffer);
            buffer = null;
        }
    }

    private bool Flush()
    {
        if (start != -1)
        {
            var toProcess = _input.Slice(start, currentIndex - start);
            SetCurrent(toProcess, start, true);

            start = -1;
            return true;
        }

        return false;
    }

    private void ExpandBuffer()
    {
        if (buffer is null)
        {
            throw new ObjectDisposedException(nameof(PreTokenizingEnumerator));
        }

        var newLen = buffer!.Length * 2;
        ArrayPool<char>.Shared.Return(buffer);
        buffer = ArrayPool<char>.Shared.Rent(newLen);
    }

    private void SetCurrent(ReadOnlySpan<char> toProcess, int currentStartIdx, bool canLowercase)
    {
        if (_convertToLowercase && canLowercase)
        {
            int lowerLen;
            while ((lowerLen = toProcess.ToLowerInvariant(buffer)) == -1)
            {
                ExpandBuffer();
            }

            Current = new() { Segment = buffer.AsSpan(0, lowerLen), SegmentStartIndex = currentStartIdx + _inputOffset };
        }
        else
        {
            Current = new() { Segment = toProcess, SegmentStartIndex = currentStartIdx + _inputOffset };
        }
    }

    private readonly (int Length, bool Normalize)? StartsWithAddedToken(ReadOnlySpan<char> value)
    {
        (int Length, bool Normalize)? maxMatch = null;

        foreach (var (content, normalize) in _addedTokens.Tokens)
        {
            if (value.StartsWith(content.AsSpan()))
            {
                if (maxMatch == null || content.Length > maxMatch.Value.Length)
                {
                    maxMatch = (content.Length, normalize);
                }
            }
        }

        return maxMatch;
    }

    // AggressiveInlining the methods below does seem to provide a performance boost in the 2-6% ballpark of the overall tokenizer.

    /// <summary>
    /// Translated from https://github.com/huggingface/transformers/blob/05de038f3d249ce96740885f85fd8d0aa00c29bc/src/transformers/tokenization_utils.py#L292-L304.
    /// </summary>
    /// <param name="cp">Character to check.</param>
    /// <returns>True we consider the character a punctuation character, otherwise false.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1204:Static elements should appear before instance elements", Justification = "Makes more sense here.")]
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
}

internal ref struct PreTokenizerResult
{
    public ReadOnlySpan<char> Segment;
    public int SegmentStartIndex;
}
