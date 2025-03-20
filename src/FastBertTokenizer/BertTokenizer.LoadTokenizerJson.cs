// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#if NET8_0_OR_GREATER
using System.Collections.Frozen;
#endif
#if !NETSTANDARD2_0
using System.Net.Http.Json;
#endif
using System.Text;
using System.Text.Json;

namespace FastBertTokenizer;

public partial class BertTokenizer
{
    /// <summary>
    /// Load a tokenizer.json file that contains a tokenizer configuration in the format used by Hugging Face libraries.
    /// Supports version 1.0 of the tokenizer.json format.
    /// </summary>
    /// <param name="tokenizerJsonFilePath">Path to the tokenizer.json file.</param>
    /// <returns>A task that represents the loading operation.</returns>
    /// <exception cref="ArgumentNullException">If one of the requred arguments is null.</exception>
    /// <exception cref="InvalidOperationException">If a vocabulary is already loaded OR if the vocabulary does not contain one of the specified special tokens.</exception>
    public async Task LoadTokenizerJsonAsync(string tokenizerJsonFilePath)
    {
        using var file = File.OpenRead(tokenizerJsonFilePath);
        await LoadTokenizerJsonAsync(file);
    }

    // these are actually inherited
#pragma warning disable CS1573 // Parameter besitzt kein übereinstimmendes param-Tag im XML-Kommentar (andere Parameter jedoch schon)

    /// <param name="tokenizerJsonStream">A stream that can be used to read the contents of a tokenizer.json file.</param>
    /// <inheritdoc cref="LoadTokenizerJsonAsync(string)"/>
    public async Task LoadTokenizerJsonAsync(Stream tokenizerJsonStream)
    {
        var tok = await JsonSerializer.DeserializeAsync(tokenizerJsonStream, TokenizerJsonContext.Default.TokenizerJson)
            ?? throw new ArgumentException("Tokenizer configuration could not be deserialised.");
        LoadTokenizerJsonImpl(tok);
    }

    /// <inheritdoc cref="LoadTokenizerJson(Stream)"/>
    public void LoadTokenizerJson(Stream tokenizerJsonStream)
    {
        var tok = JsonSerializer.Deserialize(tokenizerJsonStream, TokenizerJsonContext.Default.TokenizerJson)
            ?? throw new ArgumentException("Tokenizer configuration could not be deserialised.");
        LoadTokenizerJsonImpl(tok);
    }

#if !NETSTANDARD2_0
    internal async Task LoadTokenizerJsonAsync(HttpClient httpClient, string url)
    {
        var tok = await httpClient.GetFromJsonAsync(url, TokenizerJsonContext.Default.TokenizerJson)
            ?? throw new ArgumentException("Tokenizer configuration could not be deserialised.");
        LoadTokenizerJsonImpl(tok);
    }
#endif

#pragma warning restore CS1573 // Parameter besitzt kein übereinstimmendes param-Tag im XML-Kommentar (andere Parameter jedoch schon)

    private void LoadTokenizerJsonImpl(TokenizerJson tok)
    {
        if (_prefixes is not null)
        {
            throw new InvalidOperationException("Tokenizer configuration already loaded.");
        }

        if (tok.Version != "1.0")
        {
            throw new ArgumentException($"The confiuguration specifies version {tok.Version}, but currently only version 1.0 is supported.");
        }

        if (tok.Model.Type != "WordPiece" && tok.Model.Type is not null)
        {
            throw new ArgumentException($"The confiuguration specifies model type {tok.Model.Type}, but currently only WordPiece is supported.");
        }

        if (tok.Normalizer.Type != "BertNormalizer")
        {
            throw new ArgumentException($"The confiuguration specifies normalizer type {tok.Normalizer.Type}, but currently only BertNormalizer is supported.");
        }

        if (tok.PreTokenizer.Type != "BertPreTokenizer")
        {
            throw new ArgumentException($"The confiuguration specifies pre-tokenizer type {tok.PreTokenizer.Type}, but currently only BertPreTokenizer is supported.");
        }

        if (tok.Normalizer.HandleChineseChars is false)
        {
            throw new ArgumentException("The confiuguration specifies Normalizer.HandleChineseChars = false, but currently only HandleChineseChars = true is supported.");
        }

        if (tok.Normalizer.CleanText is false)
        {
            throw new ArgumentException("The confiuguration specifies Normalizer.CleanText = false, but currently only CleanText = true is supported.");
        }

        var suffixPrefix = tok.Model.ContinuingSubwordPrefix; // e.g. "##"
        var normalization = NormalizationForm.FormD; // bert uses FormD per default
        var clsSpecialToken = tok.PostProcessor.SpecialTokens["[CLS]"];
        var sepSpecialToken = tok.PostProcessor.SpecialTokens["[SEP]"];
        var unkToken = tok.Model.UnkToken;
        var clsToken = clsSpecialToken.Id;
        var sepToken = sepSpecialToken.Id;
        var padToken = "[PAD]"; // In e.g. https://huggingface.co/bert-base-uncased/raw/main/tokenizer.json there is no nice way to detect this.

        var prefixes = new Dictionary<StringSpanOrdinalKey, long>();
        var suffixes = new Dictionary<StringSpanOrdinalKey, long>();
        (int? unkId, int? clsId, int? sepId, int? padId) = (null, null, null, null);

        void HandleLine(string line, int tokenId)
        {
            if (string.IsNullOrEmpty(line))
            {
                return;
            }

            if (line.StartsWith(suffixPrefix, StringComparison.Ordinal))
            {
                suffixes[new(line[suffixPrefix.Length..])] = tokenId;
                return;
            }

            if (line.Equals(unkToken, StringComparison.Ordinal))
            {
                unkId = tokenId;
            }
            else if (line.Equals(clsToken, StringComparison.Ordinal))
            {
                clsId = tokenId;
            }
            else if (line.Equals(sepToken, StringComparison.Ordinal))
            {
                sepId = tokenId;
            }
            else if (line.Equals(padToken, StringComparison.Ordinal))
            {
                padId = tokenId;
            }

            prefixes[new(line)] = tokenId;
        }

        foreach (var (token, id) in tok.Model.Vocab)
        {
            HandleLine(token, id);
        }

        foreach (var addedToken in tok.AddedTokens)
        {
            // We ignore LStrip and RStrip as FastBertTokenizer always pre-tokenizes on whitespace,
            // true and false would result in the same behavior.
            if (addedToken.SingleWord)
            {
                throw new ArgumentException($"FastBertTokenizer only supports added tokens with single_word=false. The given tokenizer.json does however contain an added token {addedToken.Content} = {addedToken.Id} with single_word = true.");
            }

            // This does not seem to be required in all cases, e.g. [CLS] with bert-base-uncased, but required in others, e.g. 21.22 with issue #100 tokenizer.
            prefixes[new(addedToken.Content)] = addedToken.Id;
        }

        _addedTokens = new(tok.AddedTokens.Select(x => (x.Content, x.Normalized)).OrderByDescending(x => x.Content.Length));

#if NET8_0_OR_GREATER
        _prefixes = prefixes.ToFrozenDictionary();
        _suffixes = suffixes.ToFrozenDictionary();
#else
        _prefixes = prefixes;
        _suffixes = suffixes;
#endif
        _lowercaseInput = tok.Normalizer.Lowercase;

        // https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.normalizers
        // strip_accents (bool, optional) – Whether to strip all accents. If this option is not specified (ie == None), then it will be determined by the value for lowercase (as in the original Bert).
        _stripAccents = tok.Normalizer.StripAccents ?? _lowercaseInput;
        _decoderPrefix = tok.Decoder?.Prefix ?? "##";
        _cleanupTokenizationSpaces = tok.Decoder?.Cleanup ?? true;
        _normalization = normalization;
        _unk = (unkId ?? throw new InvalidOperationException($"Vocabulary does not contain unknown token {unkToken}."), unkToken);
        _cls = (clsId ?? throw new InvalidOperationException($"Vocabulary does not contain cls token {clsToken}."), clsToken);
        _sep = (sepId ?? throw new InvalidOperationException($"Vocabulary does not contain sep token {sepToken}."), sepToken);
        _pad = (padId ?? throw new InvalidOperationException($"Vocabulary does not contain pad token {padToken}."), padToken);
    }
}
