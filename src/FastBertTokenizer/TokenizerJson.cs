// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text.Json.Serialization;

namespace FastBertTokenizer;

// It might be better to use the snake_case naming convention here,
// but it is not available in .NET 6. Note that the tokenizer.json
// format does not use snake_case consequently though.
internal record TokenizerJson
{
    public required string Version { get; init; }

    [JsonPropertyName("added_tokens")]
    public required AddedToken[] AddedTokens { get; init; }

    public required NormalizerSection Normalizer { get; init; }

    [JsonPropertyName("pre_tokenizer")]
    public required PreTokenizerSection PreTokenizer { get; init; }

    [JsonPropertyName("post_processor")]
    public required PostProcessorSection PostProcessor { get; init; }

    public required ModelSection Model { get; init; }

    public DecoderSection? Decoder { get; init; }

    // https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.AddedToken
    internal record AddedToken
    {
        public required int Id { get; init; }

        public required string Content { get; init; }

        [JsonPropertyName("single_word")]
        public bool SingleWord { get; init; }

        public bool LStrip { get; init; }

        public bool RStrip { get; init; }

        public bool Normalized { get; init; }
    }

    internal record NormalizerSection
    {
        public required string Type { get; init; }

        [JsonPropertyName("clean_text")]
        public bool CleanText { get; init; } = true;

        [JsonPropertyName("handle_chinese_chars")]
        public bool HandleChineseChars { get; init; } = true;

        [JsonPropertyName("strip_accents")]
        public bool? StripAccents { get; init; }

        public bool Lowercase { get; init; } = true;
    }

    internal record PreTokenizerSection
    {
        public required string Type { get; init; }
    }

    internal record PostProcessorSection
    {
        public required string Type { get; init; }

        [JsonPropertyName("special_tokens")]
        public required Dictionary<string, SpecialTokenDetails> SpecialTokens { get; init; }

        internal record SpecialTokenDetails
        {
            /// <summary>
            /// E.g. [CLS] or [SEP].
            /// </summary>
            public required string Id { get; init; }
        }
    }

    internal record ModelSection
    {
        public string? Type { get; init; }

        [JsonPropertyName("unk_token")]
        public required string UnkToken { get; init; }

        [JsonPropertyName("continuing_subword_prefix")]
        public required string ContinuingSubwordPrefix { get; init; }

        public required Dictionary<string, int> Vocab { get; set; }
    }

    internal record DecoderSection
    {
        public string? Type { get; init; }

        public string? Prefix { get; init; }

        public bool? Cleanup { get; init; }
    }
}
