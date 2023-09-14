using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using Shouldly;

namespace FastBertTokenizer.Tests
{
    public class CompareToHuggingfaceTokenizer
    {
        private readonly RestBaaiBgeTokenizer _rest = new("http://localhost:8080/tokenize");
        private readonly BertTokenizer _underTest = new();

        public CompareToHuggingfaceTokenizer()
        {
            _underTest.LoadVocabularyAsync("Vocabularies/baai-bge-small-en-vocab.txt", true).GetAwaiter().GetResult();
        }

        [Fact]
        public async Task Test1()
        {
            string corpusFolder = "C:\\Users\\georg\\simplewikicorpus";
            var files = Directory.GetFiles(corpusFolder);
            foreach (var file in files.Where(x => x.Contains("30153", StringComparison.Ordinal)))
            {
                var tx = File.ReadAllText(file).Normalize();
                var huggF = await _rest.TokenizeAsync(tx);
                var ours = _underTest.Tokenize(tx, 512);
                try
                {
                    ours.InputIds.ShouldBe(huggF.InputIds);
                    ours.AttentionMask.ShouldBe(huggF.AttentionMask);
                    ours.TokenTypeIds.ShouldBe(huggF.TokenTypeIds);
                }
                catch (Exception ex)
                {
                    var hash = file.GetHashCode(StringComparison.OrdinalIgnoreCase);
                    File.WriteAllText($"huggf_{hash}.json", JsonSerializer.Serialize(
                        new { dec = _underTest.Decode(huggF.InputIds.Span), input_ids = huggF.InputIds.ToArray(), attm = huggF.AttentionMask.ToArray(), toktyp = huggF.TokenTypeIds.ToArray() }));
                    File.WriteAllText($"ours_{hash}.json", JsonSerializer.Serialize(
                        new { dec = _underTest.Decode(ours.InputIds.Span), input_ids = ours.InputIds.ToArray(), attm = ours.AttentionMask.ToArray(), toktyp = ours.TokenTypeIds.ToArray() }));
                    throw new ShouldAssertException($"Assertion failed for file {file}:\n{ex.Message}", ex);
                }
            }
        }
    }
}
