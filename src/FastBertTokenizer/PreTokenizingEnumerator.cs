// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;
#if NET8_0_OR_GREATER
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
#endif
using System.Runtime.InteropServices;
using System.Text;
using CharClass = FastBertTokenizer.PreTokenizer.CharClass;

namespace FastBertTokenizer;

/// <summary>
/// Splits one input into the segments that are subsequently tokenized into word pieces: added tokens,
/// single punctuation chars and whitespace separated words. Which chars count as what is decided by
/// the <see cref="PreTokenizer"/> this enumerator is created from.
/// </summary>
internal ref struct PreTokenizingEnumerator
{
    private readonly bool _convertToLowercase;
    private readonly int _inputOffset;
    private readonly ReadOnlySpan<char> _input;
    private readonly PreTokenizer _preTokenizer;
    private int currentIndex;
    private char[]? buffer = null;

    public PreTokenizingEnumerator(string input, bool convertToLowercase, NormalizationForm vocabNf, PreTokenizer preTokenizer, int inputOffset = 0)
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
        _preTokenizer = preTokenizer;
        _inputOffset = inputOffset;
        currentIndex = 0;
    }

    public PreTokenizerResult Current { get; private set; }

    public readonly PreTokenizingEnumerator GetEnumerator() => this;

    public bool MoveNext()
    {
        // The fast path: ASCII chars that the lookup table fully classifies. Everything else, i.e. non-ASCII chars and
        // chars that might start an added token, is handled by MoveNextSlow. This loop works on locals and doesn't
        // call anything while they are needed, so that they can stay in registers.
        var input = _input;
        var asciiCharClasses = _preTokenizer.AsciiCharClasses;
#if NET8_0_OR_GREATER
        var asciiAlphanumericsAreWordChars = _preTokenizer.AsciiAlphanumericsAreWordChars;
#endif
        var i = currentIndex;

        while (true)
        {
            if ((uint)i >= (uint)input.Length)
            {
                currentIndex = i;
                return false;
            }

            var c = input[i];
            if (c >= asciiCharClasses.Length)
            {
                break;
            }

            var charClass = asciiCharClasses[c];
            if ((charClass & CharClass.MayStartAddedToken) != 0)
            {
                break;
            }

            if ((charClass & CharClass.Whitespace) != 0)
            {
                i++;
                continue;
            }

            if ((charClass & CharClass.Punctuation) != 0)
            {
                currentIndex = i + 1;
                Current = new() { Segment = input.Slice(i, 1), SegmentStartIndex = i + _inputOffset };
                return true;
            }

            // A word starts at i. It ends before the next char that is whitespace, punctuation or the start of an added token.
            var start = i;
            var wordCharClasses = charClass;
            i++;
#if NET8_0_OR_GREATER
            if (Vector128.IsHardwareAccelerated && asciiAlphanumericsAreWordChars)
            {
                // Skip the ASCII letters and digits of the word, eight chars at a time. The loop below then takes care
                // of the char that ends this run and determines whether it ends the word too.
                ref var inputStart = ref Unsafe.As<char, ushort>(ref MemoryMarshal.GetReference(input));
                while (i <= input.Length - Vector128<ushort>.Count)
                {
                    var chars = Vector128.LoadUnsafe(ref inputStart, (nuint)i);
                    var isLetterOrDigit = Vector128.LessThan(chars - Vector128.Create((ushort)'0'), Vector128.Create((ushort)10))
                        | Vector128.LessThan((chars | Vector128.Create((ushort)0x20)) - Vector128.Create((ushort)'a'), Vector128.Create((ushort)26));
                    var isUpper = Vector128.LessThan(chars - Vector128.Create((ushort)'A'), Vector128.Create((ushort)26));
                    var letterOrDigitCount = BitOperations.TrailingZeroCount(~isLetterOrDigit.ExtractMostSignificantBits());
                    if ((isUpper.ExtractMostSignificantBits() & ((1u << letterOrDigitCount) - 1)) != 0)
                    {
                        wordCharClasses |= CharClass.AsciiUppercase;
                    }

                    i += letterOrDigitCount;
                    if (letterOrDigitCount < Vector128<ushort>.Count)
                    {
                        break;
                    }
                }
            }
#endif

            var next = i;
            while ((uint)i < (uint)input.Length)
            {
                c = input[i];
                if (c >= asciiCharClasses.Length)
                {
                    return ContinueWordSlow(start, i, wordCharClasses);
                }

                charClass = asciiCharClasses[c];
                if ((charClass & CharClass.EndsWord) != 0)
                {
                    if ((charClass & (CharClass.Whitespace | CharClass.Punctuation)) == 0)
                    {
                        return ContinueWordSlow(start, i, wordCharClasses);
                    }

                    // Whitespace that ends a word can be skipped right away, unless an added token might start with it.
                    next = charClass == CharClass.Whitespace ? i + 1 : i;
                    break;
                }

                wordCharClasses |= charClass;
                i++;
                next = i;
            }

            currentIndex = next;
            SetWord(input.Slice(start, i - start), start, wordCharClasses);
            return true;
        }

        currentIndex = i;
        return MoveNextSlow();
    }

    public void Dispose()
    {
        if (buffer is not null)
        {
            ArrayPool<char>.Shared.Return(buffer);
            buffer = null;
        }
    }

    /// <summary>
    /// Maps A-Z to a-z and copies all other chars as they are.
    /// </summary>
    /// <param name="source">The chars to lowercase, all of which must be ASCII.</param>
    /// <param name="destination">Where to write the lowercased chars to. Must be as long as <paramref name="source"/>.</param>
    private static void AsciiToLower(ReadOnlySpan<char> source, Span<char> destination)
    {
        // Four chars at a time: 0x80 is set in the lanes of chars that are >= 'A' but not > 'Z'. As all chars are ASCII,
        // no lane overflows into the next one.
        var source4 = MemoryMarshal.Cast<char, ulong>(source);
        var destination4 = MemoryMarshal.Cast<char, ulong>(destination);
        for (var i = 0; i < source4.Length && i < destination4.Length; i++)
        {
            var chars = source4[i];
            var isUpper = ((chars + 0x003F_003F_003F_003FUL) ^ (chars + 0x0025_0025_0025_0025UL)) & 0x0080_0080_0080_0080UL;
            destination4[i] = chars | (isUpper >> 2);
        }

        for (var i = source4.Length * 4; i < source.Length; i++)
        {
            var c = source[i];
            destination[i] = (char)(c | ((uint)(c - 'A') <= 'Z' - 'A' ? 0x20 : 0));
        }
    }

    /// <summary>
    /// The general variant of <see cref="MoveNext"/> that handles all chars, including non-ASCII ones and added tokens.
    /// </summary>
    /// <returns>True if a segment was found.</returns>
    private bool MoveNextSlow()
    {
        var input = _input;
        var preTokenizer = _preTokenizer;
        var i = currentIndex;
        while ((uint)i < (uint)input.Length)
        {
            var charClass = preTokenizer.Classify(input[i]);
            if ((charClass & CharClass.MayStartAddedToken) != 0 && preTokenizer.StartsWithAddedToken(input.Slice(i)) is (int len, bool normalize))
            {
                currentIndex = i + len;
                SetCurrent(input.Slice(i, len), i, normalize);
                return true;
            }

            if ((charClass & CharClass.Whitespace) != 0)
            {
                i++;
                continue;
            }

            if ((charClass & CharClass.Punctuation) != 0)
            {
                currentIndex = i + 1;
                Current = new() { Segment = input.Slice(i, 1), SegmentStartIndex = i + _inputOffset };
                return true;
            }

            return ContinueWordSlow(i, i + 1, charClass);
        }

        currentIndex = i;
        return false;
    }

    /// <summary>
    /// Finds the end of the word that starts at <paramref name="start"/>, of which all chars before <paramref name="i"/>
    /// are known to be word chars, and makes it <see cref="Current"/>.
    /// </summary>
    /// <param name="start">The index the word starts at.</param>
    /// <param name="i">The index of the first char that is not yet known to belong to the word.</param>
    /// <param name="wordCharClasses">The combined classes of the chars before <paramref name="i"/>.</param>
    /// <returns>Always true.</returns>
    private bool ContinueWordSlow(int start, int i, CharClass wordCharClasses)
    {
        var input = _input;
        var preTokenizer = _preTokenizer;
        while ((uint)i < (uint)input.Length)
        {
            var charClass = preTokenizer.Classify(input[i]);
            if ((charClass & CharClass.EndsWord) != 0
                && ((charClass & (CharClass.Whitespace | CharClass.Punctuation)) != 0 || preTokenizer.StartsWithAddedToken(input.Slice(i)) is not null))
            {
                break;
            }

            wordCharClasses |= charClass;
            i++;
        }

        currentIndex = i;
        SetWord(input.Slice(start, i - start), start, wordCharClasses);
        return true;
    }

    private Span<char> GetBuffer(int minLength)
    {
        if (buffer is null || buffer.Length < minLength)
        {
            if (buffer is not null)
            {
                ArrayPool<char>.Shared.Return(buffer);
            }

            buffer = ArrayPool<char>.Shared.Rent(Math.Max(64, minLength));
        }

        return buffer;
    }

    private void SetWord(ReadOnlySpan<char> word, int startIdx, CharClass wordCharClasses)
    {
        if (_convertToLowercase && (wordCharClasses & (CharClass.AsciiUppercase | CharClass.NonAscii)) != 0)
        {
            if ((wordCharClasses & CharClass.NonAscii) != 0)
            {
                var lowerNonAscii = GetBuffer(word.Length).Slice(0, word.Length);
                if (PreTokenizer.TryToLowerInvariantWithoutSurrogates(word, lowerNonAscii))
                {
                    Current = new() { Segment = lowerNonAscii, SegmentStartIndex = startIdx + _inputOffset };
                }
                else
                {
                    SetCurrent(word, startIdx, true);
                }

                return;
            }

            // All chars are ASCII, thus ToLowerInvariant would just map A-Z to a-z.
            var lower = GetBuffer(word.Length).Slice(0, word.Length);
            AsciiToLower(word, lower);
            Current = new() { Segment = lower, SegmentStartIndex = startIdx + _inputOffset };
        }
        else
        {
            // Words without uppercase ASCII and non-ASCII chars are lowercase already, no need to copy them.
            Current = new() { Segment = word, SegmentStartIndex = startIdx + _inputOffset };
        }
    }

    private void SetCurrent(ReadOnlySpan<char> toProcess, int currentStartIdx, bool canLowercase)
    {
        if (_convertToLowercase && canLowercase)
        {
            // ToLowerInvariant maps every char to exactly one char, so it can't run out of space here.
            var lower = GetBuffer(toProcess.Length);
            var lowerLen = toProcess.ToLowerInvariant(lower);
            Current = new() { Segment = lower.Slice(0, lowerLen), SegmentStartIndex = currentStartIdx + _inputOffset };
        }
        else
        {
            Current = new() { Segment = toProcess, SegmentStartIndex = currentStartIdx + _inputOffset };
        }
    }
}

internal ref struct PreTokenizerResult
{
    public ReadOnlySpan<char> Segment;
    public int SegmentStartIndex;
}
