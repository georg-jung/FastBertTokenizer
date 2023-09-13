// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer;

/// <summary>
/// How attention_mask, input_ids and token_type_ids are created: https://huggingface.co/transformers/v3.2.0/glossary.html.
/// </summary>
public class BertTokenizer
{
    private Dictionary<string, int>? _prefixes;
    private Dictionary<string, int>? _suffixes;
    private int _unkId;
    private int _clsId;
    private int _sepId;
    private int _padId;

    public async Task LoadVocabularyAsync(string vocabFilePath, bool casedVocabulary, string unknownToken = "[UNK]", string clsToken = "[CLS]", string sepToken = "[SEP]", string padToken = "[PAD]")
    {
        using var sr = new StreamReader(vocabFilePath);
        await LoadVocabularyAsync(sr, casedVocabulary, unknownToken, clsToken, sepToken, padToken);
    }

    public async Task LoadVocabularyAsync(TextReader vocabFile, bool casedVocabulary, string unknownToken = "[UNK]", string clsToken = "[CLS]", string sepToken = "[SEP]", string padToken = "[PAD]")
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

        _prefixes = new Dictionary<string, int>(casedVocabulary ? StringComparer.Ordinal : StringComparer.OrdinalIgnoreCase);
        _suffixes = new Dictionary<string, int>(casedVocabulary ? StringComparer.Ordinal : StringComparer.OrdinalIgnoreCase);
        (int? unkId, int? clsId, int? sepId, int? padId) = (null, null, null, null);
        var i = 0;
        while (await vocabFile.ReadLineAsync() is string line)
        {
            if (!string.IsNullOrEmpty(line))
            {
                if (line.StartsWith("##", StringComparison.Ordinal))
                {
                    _suffixes[line[2..]] = i;
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
                    _prefixes[line] = i;
                }
            }

            i++;
        }

        _unkId = unkId ?? throw new InvalidOperationException($"Vocabulary does not contain unknown token {unknownToken}.");
        _clsId = clsId ?? throw new InvalidOperationException($"Vocabulary does not contain cls token {clsToken}.");
        _sepId = sepId ?? throw new InvalidOperationException($"Vocabulary does not contain sep token {sepToken}.");
        _padId = padId ?? throw new InvalidOperationException($"Vocabulary does not contain pad token {padToken}.");
    }

    public (Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds) Tokenize(ReadOnlySpan<char> input, int maximumTokens = 512, int? padTo = null)
    {
        _ = _prefixes ?? throw new InvalidOperationException("Vocabulary not loaded.");
        _ = _suffixes ?? throw new InvalidOperationException("Vocabulary not loaded.");

        var inputIdCnt = 1;
        var inputIds = new long[maximumTokens];
        inputIds[0] = _clsId;
        PreTokenizer.PreTokenize(input, OnWordToken);

        bool OnWordToken(ReadOnlySpan<char> word)
        {
            var added = TokenizeSubword(word, inputIds.AsSpan(inputIdCnt, inputIds.Length - inputIdCnt));
            if (inputIdCnt + added + 1 > maximumTokens)
            {
                // don't add partial words for now.
                // ToDo: Optimize performance
                // inputIds.AddRange(subwordIds.Take(maximumTokens - inputIds.Count - 1));
                return false;
            }

            inputIdCnt += added;
            return inputIdCnt + 1 < maximumTokens;
        }

        inputIds[inputIdCnt] = _sepId;
        inputIdCnt++;
        var nonPaddedCnt = inputIdCnt;

        if (padTo is int padLen && padLen > inputIdCnt)
        {
            Array.Fill(inputIds, _padId, inputIdCnt, padLen - inputIdCnt);
            inputIdCnt = padLen;
        }

        var attM = new long[inputIdCnt];
        var tokTypI = new long[inputIdCnt];
        Array.Fill(attM, 1, 0, nonPaddedCnt - 1);
        Array.Fill(attM, 1, nonPaddedCnt, inputIdCnt - nonPaddedCnt);
        Array.Fill(tokTypI, 0);
        return (inputIds.AsMemory(0, inputIdCnt), attM, tokTypI);
    }

    private int TokenizeSubword(ReadOnlySpan<char> word, Span<long> tokenIdSink)
    {
        // No null checks for _prefixes and _suffixes because this is a private method.
        var prefix = word;
        var cnt = 0;
        int id = -1;

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
            tokenIdSink[0] = _unkId;
            return 1;
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
                tokenIdSink[0] = _unkId;
                return 1;
            }

            tokenIdSink[cnt] = id;
            cnt++;
            remaining = remaining.Slice(suffix.Length);
        }

        return cnt;
    }
}
