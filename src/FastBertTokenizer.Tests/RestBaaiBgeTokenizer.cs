// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace FastBertTokenizer.Tests;

public class RestBaaiBgeTokenizer
{
    private readonly string _requestUri;

    public RestBaaiBgeTokenizer(string requestUri)
    {
        _requestUri = requestUri;
    }

    public async Task<(Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds)> TokenizeAsync(string input, CancellationToken cancellationToken = default)
    {
        using var hc = new HttpClient();

        var content = new StringContent(JsonSerializer.Serialize(new { text = input }), Encoding.UTF8, "application/json");

        var response = await hc.PostAsync(_requestUri, content, cancellationToken);

        var model = await JsonSerializer.DeserializeAsync<RestModel>(await response.Content.ReadAsStreamAsync(cancellationToken), cancellationToken: cancellationToken)
            ?? throw new TokenizerServiceException("The tokenizer service did not provide a proper response.");

        return (model.InputIds, model.AttentionMask, model.TokenTypeIds);
    }

    [System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.LayoutRules", "SA1502:Element should not be on a single line", Justification = "Easy to understand here anyways.")]
    public class TokenizerServiceException : Exception
    {
        public TokenizerServiceException()
            : base() { }

        public TokenizerServiceException(string message)
            : base(message) { }

        public TokenizerServiceException(string message, Exception innerException)
            : base(message, innerException) { }
    }

    private record class RestModel(
        [property: JsonPropertyName("attention_mask")] long[] AttentionMask,
        [property: JsonPropertyName("input_ids")] long[] InputIds,
        [property: JsonPropertyName("token_type_ids")] long[] TokenTypeIds)
    {
    }
}
