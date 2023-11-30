// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using RustLibWrapper;
using Shouldly;

namespace FastBertTokenizer.Tests
{
    [Collection("UsesRustLib")]
    public class CompareToHuggingfaceTokenizer : IClassFixture<BgeVocabFixture>
    {
        private readonly BgeVocabFixture _bgeVocab;

        public CompareToHuggingfaceTokenizer(BgeVocabFixture bgeVocab)
        {
            _bgeVocab = bgeVocab;
            RustTokenizer.LoadTokenizer("data/baai-bge-small-en/tokenizer.json", 512);
        }

        [SkippableTheory]
        [MemberData(nameof(WikipediaSimpleData.GetArticles), MemberType = typeof(WikipediaSimpleData))]
        public void CompareSimpleWikipediaCorpusAsIs(int id, string content)
        {
            CompareImpl(id, content);
        }

        [SkippableTheory]
        [MemberData(nameof(WikipediaSimpleData.GetArticles), MemberType = typeof(WikipediaSimpleData))]
        public void CompareSimpleWikipediaCorpusFormD(int id, string content)
        {
            CompareImpl(id, content.Normalize(System.Text.NormalizationForm.FormD));
        }

        [SkippableTheory]
        [MemberData(nameof(WikipediaSimpleData.GetArticles), MemberType = typeof(WikipediaSimpleData))]
        public void CompareSimpleWikipediaCorpusFormC(int id, string content)
        {
            CompareImpl(id, content.Normalize(System.Text.NormalizationForm.FormC));
        }

        [SkippableTheory]
        [MemberData(nameof(WikipediaSimpleData.GetArticles), MemberType = typeof(WikipediaSimpleData))]
        public void CompareSimpleWikipediaCorpusFormKC(int id, string content)
        {
            Skip.If(id == 11133 || id == 44470 || id == 45931 || id == 13451, "In NFKC there are some more differences regarding [UNK] tokens that become apparent in these 4 articles.");
            CompareImpl(id, content.Normalize(System.Text.NormalizationForm.FormKC));
        }

        [SuppressMessage("StyleCop.CSharp.ReadabilityRules", "SA1118:ParameterMustNotSpanMultipleLines", Justification = "Reviewed.")]
        private void CompareImpl(int id, string content)
        {
            Skip.If(id == 6309, "6309 \"Letter\" contains assamese characters and huggingface tokenizer skips one [UNK] were I think one should be.");
            Skip.If(id == 30153, "30153 \"Avignon\" has Rhône as the last word before hitting the 512 token id limit; we try prefixes first, "
                + "huggingface removes diacritics first. Thus, we end with token id for r while huggingface (correctly) emits rhone. Quite in "
                + "edge case that is just ever relevant for the last word, after which the tokenized version is cut off.");
            Skip.If(id == 60246, "60246 \"Shahada\" has many tokens in exotic scripts as 6309 letter. Huggingface emits one [UNK] more than we do here too.");

            var tok = _bgeVocab.UnderTest;
            var huggF = RustTokenizer.TokenizeAndGetIds(content, 512);
            var ours = tok.Tokenize(content, 512, 512);
            try
            {
                ours.InputIds.ShouldBe(huggF.InputIds);
                ours.AttentionMask.ShouldBe(huggF.AttentionMask);
                ours.TokenTypeIds.ShouldBe(huggF.TokenTypeIds);
            }
            catch (Exception ex)
            {
                File.WriteAllText($"unequal_tokenization_pair_huggf_{id}.json", JsonSerializer.Serialize(
                    new { dec = tok.Decode(huggF.InputIds.Span), input_ids = huggF.InputIds.ToArray(), attm = huggF.AttentionMask.ToArray(), toktyp = huggF.TokenTypeIds.ToArray() }));
                File.WriteAllText($"unequal_tokenization_pair_ours_{id}.json", JsonSerializer.Serialize(
                    new { dec = tok.Decode(ours.InputIds.Span), input_ids = ours.InputIds.ToArray(), attm = ours.AttentionMask.ToArray(), toktyp = ours.TokenTypeIds.ToArray() }));
                throw new ShouldAssertException($"Assertion failed for article {id}:\n{ex.Message}", ex);
            }
        }
    }
}
