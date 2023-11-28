// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#if NET8_0_OR_GREATER
using System.Collections.Frozen;
#endif

using System.Globalization;
using System.Text;

namespace FastBertTokenizer;

/// <summary>
/// How attention_mask, input_ids and token_type_ids are created: https://huggingface.co/transformers/v3.2.0/glossary.html.
/// </summary>
[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1204:Static elements should appear before instance elements", Justification = "Have private overload close to public one.")]
public partial class BertTokenizer
{
#if NET8_0_OR_GREATER
    private FrozenDictionary<string, long>? _prefixes;
    private FrozenDictionary<string, long>? _suffixes;
#else
    private Dictionary<string, long>? _prefixes;
    private Dictionary<string, long>? _suffixes;
#endif
    private (int Id, string Token) _unk = default!;
    private (int Id, string Token) _cls = default!;
    private (int Id, string Token) _sep = default!;
    private (int Id, string Token) _pad = default!;
    private bool _lowercaseInput;
    private NormalizationForm _normalization;

    // These will just be used if the consumer calls an API that _returns_ ReadOnlyMemory.
    // They will be reused for subsequent calls to avoid allocations.
    private long[]? _inputIdReturnBuffer = null;
    private long[]? _attentionMaskReturnBuffer = null;
    private long[]? _tokenTypeIdsReturnBuffer = null;

    /// <summary>
    /// Encode the given input string to token ids per the loaded vocabulary. Write the results to the
    /// given memory areas. When encoding multiple inputs successivly it is more efficient to reuse the
    /// memory for the results than allocating new memory and returing new arrays.
    /// </summary>
    /// <param name="input">The input to encode.</param>
    /// <param name="inputIds">
    /// The resulting token ids/input_ids will be written here. The token_ids
    /// list will be truncated (and correctly ended with a [SEP] token) if the
    /// input translates to more tokens than the list can hold.
    /// </param>
    /// <param name="attentionMask">
    /// The attention mask for the given input will be written here. At the positions
    /// of padding tokens in <paramref name="inputIds"/> attention mask will be 0.
    /// All other (relevant/interesting) positions will have a value of 1.
    /// </param>
    /// <param name="tokenTypeIds">
    /// Will be filled with 0s. Use the overload without this parameter for optimized speed.
    /// Some models which can take multiple sequences as input might need this but this is
    /// currently not supported by FastBertTokenizer.
    /// </param>
    /// <param name="padTo">
    /// Fill the given destination memory areas with padding tokens up to this length.
    /// </param>
    /// <returns>The number of token ids produced.</returns>
    public int Tokenize(string input, Span<long> inputIds, Span<long> attentionMask, Span<long> tokenTypeIds, int? padTo = null)
    {
        var inputIdCnt = Tokenize(input, inputIds, attentionMask, padTo);
        tokenTypeIds.Slice(0, inputIdCnt).Fill(0);

        return inputIdCnt;
    }

    /// <inheritdoc cref="Tokenize(string, Span{long}, Span{long}, Span{long}, int?)"/>
    public int Tokenize(string input, Span<long> inputIds, Span<long> attentionMask, int? padTo = null)
    {
        var (inputIdCnt, nonPaddedCnt) = Tokenize(input, 0, inputIds, out var _, padTo);
        attentionMask.Slice(0, nonPaddedCnt).Fill(1);
        attentionMask.Slice(nonPaddedCnt, inputIdCnt - nonPaddedCnt).Fill(0);
        return inputIdCnt;
    }

    /// <summary>
    /// Create an <see cref="IAsyncEnumerable{T}"/> that can be used to enumerate batches of tokenized inputs. Provide a source enumerable that yields
    /// inputs that should be tokenized, e.g. from a database. This source enumerable is just enumerated as needed. If an input is longer
    /// than your model allows for a single input - specify this in <paramref name="tokensPerInput"/> - it will be split into multiple
    /// model inputs. The <paramref name="batchSize"/> parameter specifies how many inputs your model can/should process batched. The
    /// returned IAsyncEnumerable will yield batches of tokenized inputs according to this batch size. If one content item is larger
    /// than one model input allows, it is split accross multiple inputs in a batch. If the content item is larger than one batch
    /// allows (or if the batch is full but one content item didn't end at the end of the batch) it continues in the next batch.
    /// The <paramref name="stride"/> parameter specifies how many tokens of overlap should be added when splitting a content item.
    /// </summary>
    /// <typeparam name="TKey">The type of the key you identify one content item in your backend storage with.</typeparam>
    /// <param name="sourceEnumerable">A producer of contents to tokenize with their identifying keys.</param>
    /// <param name="tokensPerInput">How many tokens the target models can process consequtively. E.g. 512 for many BERT models.</param>
    /// <param name="batchSize">
    /// How many consequitve inputs you want to process at once/batched in one call to your model. Choosing a small number might lead to
    /// mediocre performance. Choosing a large number might recuire a lot of memory. You might want to experiment with this parameter
    /// to find a value that leads to good performance for your use case.
    /// </param>
    /// <param name="stride">
    /// How many tokens of overlap should be added if one content item is split accross multiple inputs.
    /// E.g. if your content is 1234, your max input size is 2 and stride is 1, you get 12, 23, 34 as inputs.
    /// </param>
    /// <returns>An <see cref="IAsyncEnumerable{T}"/> which yields tokenized batches for processing by e.g. an AI model.</returns>
    public IAsyncEnumerable<TokenizedBatch<TKey>> CreateAsyncBatchEnumerator<TKey>(IAsyncEnumerable<(TKey Key, string Content)> sourceEnumerable, int tokensPerInput, int batchSize, int stride)
    {
        return new AsyncBatchEnumerator<TKey>(this, sourceEnumerable, tokensPerInput, batchSize, stride);
    }

    /// <summary>
    /// Encode the given input string to token ids per the loaded vocabulary. This overload allocated new memory to write its results to.
    /// Thus, it is less efficient than the overloads that take memory areas to write to. Consider using those if you need to encode multiple
    /// inputs successively.
    /// </summary>
    /// <param name="input">The input to encode.</param>
    /// <param name="maximumTokens">The maximum number of token ids to encode. Most bert models support inputs of up to 512 tokens.</param>
    /// <param name="padTo">Create an input_ids array of at least this length and fill possible unused positions at the end with the padding token id.</param>
    /// <returns>input_ids, attention_mask and token_type_ids that might be passed to typical BERT models.</returns>
    public (ReadOnlyMemory<long> InputIds, ReadOnlyMemory<long> AttentionMask, ReadOnlyMemory<long> TokenTypeIds) Tokenize(string input, int maximumTokens = 512, int? padTo = null)
    {
        if (_inputIdReturnBuffer is null || _inputIdReturnBuffer.Length < maximumTokens)
        {
            _inputIdReturnBuffer = new long[maximumTokens];
            _attentionMaskReturnBuffer = new long[maximumTokens];
            _tokenTypeIdsReturnBuffer = new long[maximumTokens];
            Array.Fill(_tokenTypeIdsReturnBuffer, 0);
        }

        var (inputIdCnt, nonPaddedCnt) = Tokenize(input, 0, _inputIdReturnBuffer, out var _, padTo);
        Array.Fill(_attentionMaskReturnBuffer!, 1, 0, nonPaddedCnt);
        Array.Fill(_attentionMaskReturnBuffer!, 0, nonPaddedCnt, inputIdCnt - nonPaddedCnt);
        return (
            _inputIdReturnBuffer.AsMemory(0, inputIdCnt),
            _attentionMaskReturnBuffer.AsMemory(0, inputIdCnt),
            _tokenTypeIdsReturnBuffer.AsMemory(0, inputIdCnt));
    }

    internal TokenizedRange<TKey> TokenizeBatchElement<TKey>(
        TKey inputKey,
        string input,
        int inputOffset,
        ReadOnlySpan<long> strideInputIds,
        Span<long> inputIds,
        Span<long> attentionMask)
    {
        if (attentionMask.Length != inputIds.Length)
        {
            throw new ArgumentException($"{nameof(attentionMask)}.Length must be equal to {nameof(inputIds)}.Length.", nameof(attentionMask));
        }

        var used = 0;
        if (!strideInputIds.IsEmpty)
        {
            inputIds[0] = _cls.Id;
            strideInputIds.CopyTo(inputIds.Slice(1));
            used = strideInputIds.Length + 1;
        }

        var (_, nonPaddedCnt) = Tokenize(input, inputOffset, inputIds.Slice(used), out var lastTokenizedWordStartIndex, inputIds.Length - used, emitClsToken: used == 0);
        attentionMask.Slice(0, used + nonPaddedCnt).Fill(1);
        attentionMask.Slice(used + nonPaddedCnt).Fill(0);

        return new TokenizedRange<TKey>(inputKey, inputOffset, lastTokenizedWordStartIndex);
    }

    private (int Length, int NonPadding) Tokenize(string input, int inputOffset, Span<long> inputIds, out int? nextIndexToTokenize, int? padTo = null, bool emitClsToken = true)
    {
        _ = _prefixes ?? throw new InvalidOperationException("Vocabulary not loaded.");
        _ = _suffixes ?? throw new InvalidOperationException("Vocabulary not loaded.");

        var maximumTokens = inputIds.Length;
        var inputIdCnt = 0;
        if (emitClsToken)
        {
            inputIdCnt++;
            inputIds[0] = _cls.Id;
        }

        bool moreRemainingInput = false;
        int offset = 0;
        foreach (var pivot in new PreTokenizingEnumerator(input, _lowercaseInput, _normalization, inputOffset))
        {
            offset = pivot.SegmentStartIndex;
            var added = TokenizeSubword(pivot.Segment, inputIds.Slice(inputIdCnt, inputIds.Length - inputIdCnt - 1), ref offset);

            inputIdCnt += added;

            if (inputIdCnt + 1 == maximumTokens)
            {
                moreRemainingInput = input.Length > offset;
                break;
            }
        }

        nextIndexToTokenize = moreRemainingInput ? offset : null;
        inputIds[inputIdCnt] = _sep.Id;
        inputIdCnt++;
        var nonPaddedCnt = inputIdCnt;

        if (padTo is int paddedLen && paddedLen > inputIdCnt)
        {
            inputIds.Slice(inputIdCnt, paddedLen - inputIdCnt).Fill(_pad.Id);
            inputIdCnt = paddedLen;
        }

        return (inputIdCnt, nonPaddedCnt);
    }

    /// <summary>
    /// Inspired by https://github.com/huggingface/transformers/blob/7db1ad63d9a9a8f705e13d68f90269df78a16df5/src/transformers/tokenization_utils.py#L280.
    /// We don't filter \t, \r, \n because splitting by whitespace was already done.
    /// As per https://en.wikipedia.org/wiki/Unicode_character_property#General_Category, Control, Format, Surrogate, PrivateUse and OtherNotAssigned
    /// are all categories starting with "C".
    /// </summary>
    /// <param name="text">Text to remove special unicode chars from.</param>
    /// <param name="cleaned">Contains the cleaned text.</param>
    /// <returns>True if characters were removed.</returns>
    private static bool RemoveControlAndReplacement(ReadOnlySpan<char> text, out ReadOnlySpan<char> cleaned)
    {
        bool NeedsRemoval(ReadOnlySpan<char> text)
        {
            foreach (Rune r in text.EnumerateRunes())
            {
                if (r.Value == 0xFFFD)
                {
                    return true;
                }

                var cat = Rune.GetUnicodeCategory(r);
                switch (cat)
                {
                    case UnicodeCategory.Control:
                    case UnicodeCategory.Format:
                    case UnicodeCategory.Surrogate:
                    case UnicodeCategory.PrivateUse:
                    case UnicodeCategory.OtherNotAssigned:
                        return true;
                    default:
                        break;
                }
            }

            return false;
        }

        if (!NeedsRemoval(text))
        {
            cleaned = text;
            return false;
        }

        int i = 0;
        Span<char> span = new char[text.Length];

        foreach (Rune r in text.EnumerateRunes())
        {
            if (r.Value == 0xFFFD)
            {
                continue;
            }

            switch (Rune.GetUnicodeCategory(r))
            {
                case UnicodeCategory.Control:
                case UnicodeCategory.Format:
                case UnicodeCategory.Surrogate:
                case UnicodeCategory.PrivateUse:
                case UnicodeCategory.OtherNotAssigned:
                    break;
                default:
                    r.EncodeToUtf16(span.Slice(i++));
                    if (!r.IsBmp)
                    {
                        i++;
                    }

                    break;
            }
        }

        cleaned = span.Slice(0, i);
        return true;
    }

    /// <summary>
    /// Source: https://stackoverflow.com/a/67190157/1200847.
    /// Similar to what HuggingFace tokenizer does in _run_strip_accents:
    /// https://github.com/huggingface/transformers/blob/7db1ad63d9a9a8f705e13d68f90269df78a16df5/src/transformers/models/bert/tokenization_bert.py#L449.
    /// </summary>
    /// <param name="text">String to remove diacritics from.</param>
    /// <param name="targetNf">The returned value will be unicode normalized in the form.</param>
    /// <returns>String without diacritics.</returns>
    private static string RemoveDiacritics(string text, NormalizationForm targetNf)
    {
        bool NeedsRemoval(ReadOnlySpan<char> formD)
        {
            foreach (char c in formD)
            {
                if (CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.NonSpacingMark)
                {
                    return true;
                }
            }

            return false;
        }

        ReadOnlySpan<char> normalizedString = text.Normalize(NormalizationForm.FormD);

        if (!NeedsRemoval(normalizedString))
        {
            return text;
        }

        int i = 0;
        Span<char> span = normalizedString.Length < Constants.StackallocCharThreshold
            ? stackalloc char[normalizedString.Length]
            : new char[normalizedString.Length];

        foreach (char c in normalizedString)
        {
            var cat = CharUnicodeInfo.GetUnicodeCategory(c);

            // ToLowerInvariant performed by pre-tokenizer does not lower all chars with diacrits.
            if (cat == UnicodeCategory.UppercaseLetter || cat == UnicodeCategory.TitlecaseLetter)
            {
                span[i++] = char.ToLowerInvariant(c);
            }
            else if (cat != UnicodeCategory.NonSpacingMark)
            {
                span[i++] = c;
            }
        }

        return new string(span.Slice(0, i)).Normalize(targetNf);
    }

    private int TokenizeSubword(ReadOnlySpan<char> word, Span<long> tokenIdSink, ref int offset)
    {
        int OnUnknown(ReadOnlySpan<char> word, Span<long> tokenIdSink, ref int offset)
        {
            if (RemoveControlAndReplacement(word, out var withoutControl))
            {
                if (withoutControl.Length == 0)
                {
                    return 0;
                }

                return TokenizeSubword(withoutControl, tokenIdSink, ref offset);
            }

            // Normalize and IsNormalized for ReadOnlySpan<char> is not yet implemented:
            // https://github.com/dotnet/runtime/issues/87757
            // RemoveDiacritics ends up in form _normalization too.
            // If we have a vocab that includes diacritics we might sill want to normalize first
            // and try again before removing diacritics.
            var wordStr = word.ToString();
            if (!wordStr.IsNormalized(_normalization))
            {
                return TokenizeSubword(wordStr.Normalize(_normalization), tokenIdSink, ref offset);
            }

            var withoutDiacrit = RemoveDiacritics(wordStr, _normalization);
            if (!MemoryExtensions.Equals(withoutDiacrit, word, StringComparison.Ordinal))
            {
                return TokenizeSubword(withoutDiacrit.AsSpan(), tokenIdSink, ref offset);
            }

            tokenIdSink[0] = _unk.Id;
            offset += word.Length;
            return 1;
        }

        // No null checks for _prefixes and _suffixes because this is a private method.
        var prefix = word;
        var cnt = 0;
        long id = -1;

        // ToDo: Remove string allocation; related: https://github.com/dotnet/runtime/issues/27229
        while (prefix.Length > 0)
        {
            if (_prefixes!.TryGetValue(new string(prefix), out var outId))
            {
                id = outId;
                break;
            }

            prefix = prefix.Slice(0, prefix.Length - 1);
        }

        if (id == -1)
        {
            return OnUnknown(word, tokenIdSink, ref offset);
        }

        tokenIdSink[0] = id;
        cnt++;

        var remaining = word.Slice(prefix.Length);
        offset += prefix.Length;
        while (remaining.Length > 0 && cnt < tokenIdSink.Length)
        {
            var suffix = remaining;
            id = -1;

            // ToDo: Remove string allocation; related: https://github.com/dotnet/runtime/issues/27229
            while (suffix.Length > 0)
            {
                if (_suffixes!.TryGetValue(new string(suffix), out var outId))
                {
                    id = outId;
                    break;
                }

                suffix = suffix.Slice(0, suffix.Length - 1);
            }

            if (id == -1)
            {
                return OnUnknown(word, tokenIdSink, ref offset);
            }

            tokenIdSink[cnt] = id;
            cnt++;
            remaining = remaining.Slice(suffix.Length);
            offset += suffix.Length;
        }

        return cnt;
    }
}
