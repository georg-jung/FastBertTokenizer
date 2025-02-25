// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#if NET8_0_OR_GREATER
using System.Collections.Frozen;
#endif

using System.Text;

namespace FastBertTokenizer;

public partial class BertTokenizer
{
    /// <summary>
    /// Load a vocab.txt file that assigns an id to each token based on the line number.
    /// </summary>
    /// <param name="vocabTxtFilePath">Path to the vocab.txt file.</param>
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
    /// <param name="normalization">The unicode normalization form used by this vocabulary.</param>
    /// <returns>A task that represents the loading operation.</returns>
    /// <exception cref="ArgumentNullException">If one of the requred arguments is null.</exception>
    /// <exception cref="InvalidOperationException">If a vocabulary is already loaded OR if the vocabulary does not contain one of the specified special tokens.</exception>
    public async Task LoadVocabularyAsync(string vocabTxtFilePath, bool convertInputToLowercase, string unknownToken = "[UNK]", string clsToken = "[CLS]", string sepToken = "[SEP]", string padToken = "[PAD]", NormalizationForm normalization = NormalizationForm.FormD)
    {
        using var sr = new StreamReader(vocabTxtFilePath);
        await LoadVocabularyAsync(sr, convertInputToLowercase, unknownToken, clsToken, sepToken, padToken, normalization)!;
    }

    // these are actually inherited
#pragma warning disable CS1573 // Parameter besitzt kein übereinstimmendes param-Tag im XML-Kommentar (andere Parameter jedoch schon)

    /// <param name="vocabTxtFile">A text reader that provides the vocab.txt file.</param>
    /// <inheritdoc cref="LoadVocabularyAsync(string, bool, string, string, string, string, NormalizationForm)"/>
    public async Task LoadVocabularyAsync(TextReader vocabTxtFile, bool convertInputToLowercase, string unknownToken = "[UNK]", string clsToken = "[CLS]", string sepToken = "[SEP]", string padToken = "[PAD]", NormalizationForm normalization = NormalizationForm.FormD)
    {
        await LoadVocabularyImpl(true, vocabTxtFile, convertInputToLowercase, unknownToken, clsToken, sepToken, padToken, normalization)!;
    }

    /// <inheritdoc cref="LoadVocabulary(TextReader, bool, string, string, string, string, NormalizationForm)"/>
    public void LoadVocabulary(TextReader vocabTxtFile, bool convertInputToLowercase, string unknownToken = "[UNK]", string clsToken = "[CLS]", string sepToken = "[SEP]", string padToken = "[PAD]", NormalizationForm normalization = NormalizationForm.FormD)
    {
        LoadVocabularyImpl(false, vocabTxtFile, convertInputToLowercase, unknownToken, clsToken, sepToken, padToken, normalization);
    }

#pragma warning restore CS1573 // Parameter besitzt kein übereinstimmendes param-Tag im XML-Kommentar (andere Parameter jedoch schon)

    private Task? LoadVocabularyImpl(bool execAsync, TextReader vocabTxtFile, bool convertInputToLowercase, string unknownToken = "[UNK]", string clsToken = "[CLS]", string sepToken = "[SEP]", string padToken = "[PAD]", NormalizationForm normalization = NormalizationForm.FormD)
    {
        _ = vocabTxtFile ?? throw new ArgumentNullException(nameof(vocabTxtFile));
        _ = unknownToken ?? throw new ArgumentNullException(nameof(unknownToken));
        _ = clsToken ?? throw new ArgumentNullException(nameof(clsToken));
        _ = sepToken ?? throw new ArgumentNullException(nameof(sepToken));
        _ = padToken ?? throw new ArgumentNullException(nameof(padToken));

        if (_prefixes is not null)
        {
            throw new InvalidOperationException("Vocabulary already loaded.");
        }

        var prefixes = new Dictionary<StringSpanOrdinalKey, long>();
        var suffixes = new Dictionary<StringSpanOrdinalKey, long>();
        (int? unkId, int? clsId, int? sepId, int? padId) = (null, null, null, null);
        var i = 0;

        void HandleLine(string line)
        {
            if (!string.IsNullOrEmpty(line))
            {
                if (line.StartsWith("##", StringComparison.Ordinal))
                {
                    suffixes[new StringSpanOrdinalKey(line[2..])] = i;
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
                    prefixes[new StringSpanOrdinalKey(line)] = i;
                }
            }

            i++;
        }

        async Task Async()
        {
            while (await vocabTxtFile.ReadLineAsync() is string line)
            {
                HandleLine(line);
            }

            Finish();
        }

        void Sync()
        {
            while (vocabTxtFile.ReadLine() is string line)
            {
                HandleLine(line);
            }

            Finish();
        }

        if (execAsync)
        {
            return Async();
        }
        else
        {
            Sync();
            return null;
        }

        void Finish()
        {
            _unk = (unkId ?? throw new InvalidOperationException($"Vocabulary does not contain unknown token {unknownToken}."), unknownToken);
            _cls = (clsId ?? throw new InvalidOperationException($"Vocabulary does not contain cls token {clsToken}."), clsToken);
            _sep = (sepId ?? throw new InvalidOperationException($"Vocabulary does not contain sep token {sepToken}."), sepToken);
            _pad = (padId ?? throw new InvalidOperationException($"Vocabulary does not contain pad token {padToken}."), padToken);

#if NET8_0_OR_GREATER
            _prefixes = prefixes.ToFrozenDictionary();
            _suffixes = suffixes.ToFrozenDictionary();
#else
            _prefixes = prefixes;
            _suffixes = suffixes;
#endif
            _lowercaseInput = convertInputToLowercase;
            _normalization = normalization;
            _addedTokens = new([(unknownToken, false), (clsToken, false), (sepToken, false), (padToken, false)]);
        }
    }
}
