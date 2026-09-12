// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;

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
    private int start;
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
            var charClass = _preTokenizer.Classify(c);

            if ((charClass & PreTokenizer.CharClass.MayStartAddedToken) != 0 && _preTokenizer.StartsWithAddedToken(_input.Slice(currentIndex)) is (int len, bool normalize))
            {
                if (Flush())
                {
                    return true;
                }

                SetCurrent(_input.Slice(currentIndex, len), currentIndex, normalize);
                currentIndex += len;
                return true;
            }
            else if ((charClass & PreTokenizer.CharClass.Whitespace) != 0)
            {
                if (Flush())
                {
                    currentIndex++;
                    return true;
                }

                currentIndex++;
            }
            else if ((charClass & PreTokenizer.CharClass.Punctuation) != 0)
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
}

internal ref struct PreTokenizerResult
{
    public ReadOnlySpan<char> Segment;
    public int SegmentStartIndex;
}
