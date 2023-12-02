// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FastBertTokenizer
{
    public static class BertTokenizerExtensions
    {
        /// <summary>
        /// Download and load a tokenizer configuration from huggingface.co.
        /// </summary>
        /// <param name="tokenizer">The tokenizer to load the configuration into.</param>
        /// <param name="huggingFaceRepo">The Hugging Face repository to download the configuration from. E.g. bert-base-uncased.</param>
        /// <returns>A <see cref="Task"/> representing the operation.</returns>
        public static async Task LoadFromHuggingFaceAsync(this BertTokenizer tokenizer, string huggingFaceRepo)
        {
            var url = $"https://huggingface.co/{huggingFaceRepo}/resolve/main/tokenizer.json?download=true";
            using var hc = new HttpClient();
            hc.DefaultRequestHeaders.Add("User-Agent", $"FastBertTokenizer {ThisAssembly.AssemblyInformationalVersion}");
            await tokenizer.LoadTokenizerJsonAsync(hc, url);
        }
    }
}
