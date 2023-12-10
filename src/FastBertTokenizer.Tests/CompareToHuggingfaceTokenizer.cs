// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text.Json;
using RustLibWrapper;
using Shouldly;

namespace FastBertTokenizer.Tests
{
    [Collection("UsesRustLib")]
    public class CompareToHuggingfaceTokenizer : IAsyncLifetime
    {
        private readonly BertTokenizer _uut = new();

        public async Task InitializeAsync()
        {
            await _uut.LoadTokenizerJsonAsync("data/bert-base-uncased/tokenizer.json");
            RustTokenizer.LoadTokenizer("data/bert-base-uncased/tokenizer.json", 512);
        }

        public Task DisposeAsync() => Task.CompletedTask;

        [Theory]
        [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
        public void CompareSimpleWikipediaCorpusAsIs(Dictionary<int, string> articles)
        {
            foreach (var (key, value) in articles)
            {
                CompareImpl(key, value);
            }
        }

        [SkippableTheory]
        [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
        public void CompareSimpleWikipediaCorpusFormD(Dictionary<int, string> articles)
        {
            var dicD = Normalize(articles, System.Text.NormalizationForm.FormD);

            foreach (var (key, value) in dicD)
            {
                CompareImpl(key, value);
            }
        }

        [SkippableTheory]
        [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
        public void CompareSimpleWikipediaCorpusFormC(Dictionary<int, string> articles)
        {
            var dicC = Normalize(articles, System.Text.NormalizationForm.FormC);

            foreach (var (key, value) in dicC)
            {
                CompareImpl(key, value);
            }
        }

        [SkippableTheory]
        [MemberData(nameof(WikipediaSimpleData.GetArticlesDict), MemberType = typeof(WikipediaSimpleData))]
        public void CompareSimpleWikipediaCorpusFormKC(Dictionary<int, string> articles)
        {
            var dicKc = new Dictionary<int, string>(articles.Count);
            foreach (var (key, value) in articles)
            {
                if (key == 11133 || key == 44470 || key == 45931 || key == 13451)
                {
                    // In NFKC there are some more differences regarding [UNK] tokens that become apparent in these 4 articles.
                    continue;
                }

                dicKc[key] = value.Normalize(System.Text.NormalizationForm.FormKC);
            }

            foreach (var (key, value) in dicKc)
            {
                CompareImpl(key, value);
            }
        }

        private Dictionary<int, string> Normalize(Dictionary<int, string> articles, System.Text.NormalizationForm form)
        {
            var dic = new Dictionary<int, string>(articles.Count);
            foreach (var (key, value) in articles)
            {
                dic[key] = value.Normalize(form);
            }

            return dic;
        }

        private void CompareImpl(int id, string content)
        {
            if (id == 6309 || id == 30153 || id == 60246)
            {
                // 6309 "Letter" contains assamese characters and huggingface tokenizer skips one [UNK] were I think one should be.
                // 30153 "Avignon" has Rhône as the last word before hitting the 512 token id limit; we try prefixes first,
                //      huggingface removes diacritics first. Thus, we end with token id for r while huggingface (correctly) emits rhone.
                //      An edge case that is just ever relevant for the last word, after which the tokenized version is cut off.
                // 60246 "Shahada" has many tokens in exotic scripts, as 6309 "Letter". Huggingface emits one [UNK] more than we do here too.
                return;
            }

            var huggF = RustTokenizer.TokenizeAndGetIds(content, 512);
            var ours = _uut.Encode(content, 512, 512);
            try
            {
                ours.InputIds.ShouldBe(huggF.InputIds);
                ours.AttentionMask.ShouldBe(huggF.AttentionMask);
                ours.TokenTypeIds.ShouldBe(huggF.TokenTypeIds);
            }
            catch (Exception ex)
            {
                File.WriteAllText($"unequal_tokenization_pair_huggf_{id}.json", JsonSerializer.Serialize(
                    new { dec = _uut.Decode(huggF.InputIds.Span), input_ids = huggF.InputIds.ToArray(), attm = huggF.AttentionMask.ToArray(), toktyp = huggF.TokenTypeIds.ToArray() }));
                File.WriteAllText($"unequal_tokenization_pair_ours_{id}.json", JsonSerializer.Serialize(
                    new { dec = _uut.Decode(ours.InputIds.Span), input_ids = ours.InputIds.ToArray(), attm = ours.AttentionMask.ToArray(), toktyp = ours.TokenTypeIds.ToArray() }));
                throw new ShouldAssertException($"Assertion failed for article {id}:\n{ex.Message}", ex);
            }
        }
    }
}
