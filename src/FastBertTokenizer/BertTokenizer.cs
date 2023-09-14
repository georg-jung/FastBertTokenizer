// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;

namespace FastBertTokenizer;

/// <summary>
/// How attention_mask, input_ids and token_type_ids are created: https://huggingface.co/transformers/v3.2.0/glossary.html.
/// </summary>
[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1204:Static elements should appear before instance elements", Justification = "Have private overload close to public one.")]
public partial class BertTokenizer
{
    private Dictionary<string, long>? _prefixes;
    private Dictionary<string, long>? _suffixes;
    private (int Id, string Token) _unk = default!;
    private (int Id, string Token) _cls = default!;
    private (int Id, string Token) _sep = default!;
    private (int Id, string Token) _pad = default!;
    private bool _lowercaseInput;

    public async Task LoadVocabularyAsync(string vocabFilePath, bool convertInputToLowercase, string unknownToken = "[UNK]", string clsToken = "[CLS]", string sepToken = "[SEP]", string padToken = "[PAD]")
    {
        using var sr = new StreamReader(vocabFilePath);
        await LoadVocabularyAsync(sr, convertInputToLowercase, unknownToken, clsToken, sepToken, padToken);
    }

    /// <summary>
    /// Load a vocab.txt file that assigns an id to each token based on the line number.
    /// </summary>
    /// <param name="vocabFile">Path to the vocab.txt file.</param>
    /// <param name="convertInputToLowercase">
    /// Convert tokenization inputs to lowercase before encoding using this vocabulary?
    /// Set to true when using an uncased model, false when using a cased model.
    /// Can be set to false when using an uncased model if the input is already lowercased, which might
    /// lead to a performance gain in the 5% ballpark (lowercasing is non-allocating nonetheless).
    /// </param>
    /// <param name="unknownToken">Special token for unkown, e.g. [UNK].</param>
    /// <param name="clsToken">Special token for cls/sequence start, e.g. [CLS].</param>
    /// <param name="sepToken">Special token for sperator, e.g. [SEP].</param>
    /// <param name="padToken">Special token for padding, e.g. [PAD].</param>
    /// <returns>A task that represents the loading operation.</returns>
    /// <exception cref="ArgumentNullException">If one of the requred arguments is null.</exception>
    /// <exception cref="InvalidOperationException">If a vocabulary is already loaded.</exception>
    public async Task LoadVocabularyAsync(TextReader vocabFile, bool convertInputToLowercase, string unknownToken = "[UNK]", string clsToken = "[CLS]", string sepToken = "[SEP]", string padToken = "[PAD]")
    {
        _ = vocabFile ?? throw new ArgumentNullException(nameof(vocabFile));
        _ = unknownToken ?? throw new ArgumentNullException(nameof(unknownToken));
        _ = clsToken ?? throw new ArgumentNullException(nameof(clsToken));
        _ = sepToken ?? throw new ArgumentNullException(nameof(sepToken));
        _ = padToken ?? throw new ArgumentNullException(nameof(padToken));

        if (_prefixes is not null)
        {
            throw new InvalidOperationException("Vocabulary already loaded.");
        }

        _prefixes = new Dictionary<string, long>(StringComparer.Ordinal);
        _suffixes = new Dictionary<string, long>(StringComparer.Ordinal);
        (int? unkId, int? clsId, int? sepId, int? padId) = (null, null, null, null);
        var i = 0;
        while (await vocabFile.ReadLineAsync() is string line)
        {
            if (!string.IsNullOrEmpty(line))
            {
                if (line.StartsWith("##", StringComparison.Ordinal))
                {
                    _suffixes[line[2..].Normalize()] = i;
                }
                else if (line.Equals(unknownToken, StringComparison.Ordinal))
                {
                    unkId = i;
                }
                else if (line.Equals(clsToken, StringComparison.Ordinal))
                {
                    clsId = i;
                }
                else if (line.Equals(sepToken, StringComparison.Ordinal))
                {
                    sepId = i;
                }
                else if (line.Equals(padToken, StringComparison.Ordinal))
                {
                    padId = i;
                }
                else
                {
                    _prefixes[line.Normalize()] = i;
                }
            }

            i++;
        }

        _lowercaseInput = convertInputToLowercase;
        _unk = (unkId ?? throw new InvalidOperationException($"Vocabulary does not contain unknown token {unknownToken}."), unknownToken);
        _cls = (clsId ?? throw new InvalidOperationException($"Vocabulary does not contain cls token {clsToken}."), clsToken);
        _sep = (sepId ?? throw new InvalidOperationException($"Vocabulary does not contain sep token {sepToken}."), sepToken);
        _pad = (padId ?? throw new InvalidOperationException($"Vocabulary does not contain pad token {padToken}."), padToken);
    }

    public int Tokenize(ReadOnlySpan<char> input, Memory<long> inputIds, Span<long> attentionMask, Span<long> tokenTypeIds, int? padTo = null)
    {
        var inputIdCnt = Tokenize(input, inputIds, attentionMask, padTo);
        tokenTypeIds.Slice(0, inputIdCnt).Fill(0);

        return inputIdCnt;
    }

    public int Tokenize(ReadOnlySpan<char> input, Memory<long> inputIds, Span<long> attentionMask, int? padTo = null)
    {
        var (inputIdCnt, nonPaddedCnt) = Tokenize(input, inputIds, padTo);
        attentionMask.Slice(0, nonPaddedCnt).Fill(1);
        attentionMask.Slice(nonPaddedCnt, inputIdCnt - nonPaddedCnt).Fill(0);
        return inputIdCnt;
    }

    public (Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds) Tokenize(ReadOnlySpan<char> input, int maximumTokens = 512, int? padTo = null)
    {
        var inputIds = new long[maximumTokens];
        var (inputIdCnt, nonPaddedCnt) = Tokenize(input, inputIds, padTo);
        var attM = new long[inputIdCnt];
        var tokTypI = new long[inputIdCnt];
        Array.Fill(attM, 1, 0, nonPaddedCnt);
        Array.Fill(attM, 1, nonPaddedCnt, inputIdCnt - nonPaddedCnt);
        Array.Fill(tokTypI, 0);
        return (inputIds.AsMemory(0, inputIdCnt), attM, tokTypI);
    }

    private (int Length, int NonPadding) Tokenize(ReadOnlySpan<char> input, Memory<long> inputIds, int? padTo = null)
    {
        _ = _prefixes ?? throw new InvalidOperationException("Vocabulary not loaded.");
        _ = _suffixes ?? throw new InvalidOperationException("Vocabulary not loaded.");

        var inputIdsSpan = inputIds.Span;
        var maximumTokens = inputIds.Length;
        var inputIdCnt = 1;
        inputIdsSpan[0] = _cls.Id;
        PreTokenizer.PreTokenize(input, OnWordToken, _lowercaseInput);

        bool OnWordToken(ReadOnlySpan<char> word)
        {
            var span = inputIds.Span;
            var added = TokenizeSubword(word, span.Slice(inputIdCnt, span.Length - inputIdCnt));
            if (inputIdCnt + added + 1 > maximumTokens)
            {
                // HuggingFace tokenizer does add partial words.
                inputIdCnt = maximumTokens - 1; // leave one out for the final [SEP] token
                return false;
            }

            inputIdCnt += added;
            return inputIdCnt + 1 < maximumTokens;
        }

        inputIds.Span[inputIdCnt] = _sep.Id;
        inputIdCnt++;
        var nonPaddedCnt = inputIdCnt;

        if (padTo is int padLen && padLen > inputIdCnt)
        {
            inputIdsSpan.Slice(inputIdCnt, padLen - inputIdCnt).Fill(_pad.Id);
            inputIdCnt = padLen;
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
    /// <returns>String without diacritics.</returns>
    private static string RemoveDiacritics(string text)
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
        Span<char> span = normalizedString.Length < 1000
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

        return new string(span.Slice(0, i)).Normalize(NormalizationForm.FormC);
    }

    private int TokenizeSubword(ReadOnlySpan<char> word, Span<long> tokenIdSink)
    {
        int OnUnknown(ReadOnlySpan<char> word, Span<long> tokenIdSink)
        {
            if (RemoveControlAndReplacement(word, out var withoutControl))
            {
                if (withoutControl.Length == 0)
                {
                    return 0;
                }

                return TokenizeSubword(withoutControl, tokenIdSink);
            }

            var withoutDiacrit = RemoveDiacritics(word.ToString());
            if (!MemoryExtensions.Equals(withoutDiacrit, word, StringComparison.Ordinal))
            {
                return TokenizeSubword(withoutDiacrit.AsSpan(), tokenIdSink);
            }

            var formD = withoutDiacrit.Normalize(NormalizationForm.FormD);
            if (!MemoryExtensions.Equals(formD, word, StringComparison.Ordinal))
            {
                return TokenizeSubword(formD.AsSpan(), tokenIdSink);
            }

            tokenIdSink[0] = _unk.Id;
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
            return OnUnknown(word, tokenIdSink);
        }

        tokenIdSink[0] = id;
        cnt++;

        var remaining = word.Slice(prefix.Length);
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
                return OnUnknown(word, tokenIdSink);
            }

            tokenIdSink[cnt] = id;
            cnt++;
            remaining = remaining.Slice(suffix.Length);
        }

        return cnt;
    }
}
