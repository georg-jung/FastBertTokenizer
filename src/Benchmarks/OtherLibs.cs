// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using BenchmarkDotNet.Attributes;
using FastBertTokenizer;
using Microsoft.ML.Tokenizers;
using RustLibWrapper;
using BertTokenizer = FastBertTokenizer.BertTokenizer;

namespace Benchmarks;

/// <summary>
/// Compares FastBertTokenizer to other tokenizer libraries. All benchmarks tokenize the same
/// corpus with the same vocabulary (baai-bge-small-en, which uses bert-base-uncased's vocab)
/// and truncate to the same maximum sequence length. Note that the compared libraries don't
/// do exactly the same work: FastBertTokenizer emits input_ids and attention_mask,
/// Microsoft.ML.Tokenizers and Tokenizers.DotNet emit just input_ids, while the Hugging Face
/// tokenizers library (behind RustLibWrapper and Tokenizers.DotNet) computes offsets and more.
/// </summary>
[Config(typeof(CompareConfig))]
[MemoryDiagnoser]
public class OtherLibs
{
    private readonly string _corpusPath;
    private readonly string _vocabTxtFile;
    private readonly string _tokenizerJsonPath;
    private readonly int _maxSequenceLength;
    private readonly BertTokenizer _tokenizer = new();
    private string[] _corpus = null!;
    private Microsoft.ML.Tokenizers.BertTokenizer _mlTokenizer = null!;
    private Tokenizers.DotNet.Tokenizer _tokenizersDotNetTokenizer = null!;

    public OtherLibs()
        : this("data/wiki-simple.json.br", "data/baai-bge-small-en/vocab.txt", "data/baai-bge-small-en/tokenizer.json", 512)
    {
    }

    public OtherLibs(string corpusPath, string vocabTxtFile, string tokenizerJsonPath, int maxSequenceLength)
    {
        _corpusPath = corpusPath;
        _vocabTxtFile = vocabTxtFile;
        _tokenizerJsonPath = tokenizerJsonPath;
        _maxSequenceLength = maxSequenceLength;
    }

    [GlobalSetup]
    public async Task SetupAsync()
    {
        _corpus = await CorpusReader.ReadBrotliJsonCorpusAsync(_corpusPath);

        await _tokenizer.LoadTokenizerJsonAsync(_tokenizerJsonPath);

        RustTokenizer.LoadTokenizer(_tokenizerJsonPath, _maxSequenceLength);

        // RemoveNonSpacingMarks defaults to false, but Hugging Face's BERT tokenizers strip
        // accents when lowercasing, as do FastBertTokenizer and the tokenizer.json used here.
        _mlTokenizer = Microsoft.ML.Tokenizers.BertTokenizer.Create(
            _vocabTxtFile,
            new BertOptions { RemoveNonSpacingMarks = true });

        // Tokenizers.DotNet has no truncation API; the underlying Hugging Face tokenizers
        // library only truncates if the tokenizer.json says so. Ours doesn't, so write a
        // temporary copy with a truncation section to make the comparison fair.
        var tokenizerJson = JsonNode.Parse(await File.ReadAllTextAsync(_tokenizerJsonPath))!;
        tokenizerJson["truncation"] = new JsonObject
        {
            ["direction"] = "Right",
            ["max_length"] = _maxSequenceLength,
            ["strategy"] = "LongestFirst",
            ["stride"] = 0,
        };
        var truncatingTokenizerJsonPath = Path.Combine(Path.GetTempPath(), $"fastberttokenizer-bench-truncating-tokenizer.json");
        await File.WriteAllTextAsync(truncatingTokenizerJsonPath, tokenizerJson.ToJsonString());
        _tokenizersDotNetTokenizer = new(vocabPath: truncatingTokenizerJsonPath);
    }

    [Benchmark(Baseline = true)]
    public IReadOnlyCollection<object> FastBertTokenizer()
    {
        List<object> res = new(_corpus.Length);
        foreach (var text in _corpus)
        {
            res.Add(_tokenizer.Encode(text, _maxSequenceLength));
        }

        return res;
    }

    [Benchmark]
    public IReadOnlyCollection<object> MicrosoftMLTokenizers()
    {
        List<object> res = new(_corpus.Length);
        foreach (var text in _corpus)
        {
            res.Add(_mlTokenizer.EncodeToIds(text, _maxSequenceLength, out _, out _));
        }

        return res;
    }

    [Benchmark]
    public IReadOnlyCollection<object> TokenizersDotNet()
    {
        List<object> res = new(_corpus.Length);
        foreach (var text in _corpus)
        {
            res.Add(_tokenizersDotNetTokenizer.Encode(text));
        }

        return res;
    }

    [Benchmark]
    public object RustHuggingfaceWrapperSinglethreadedMemReuse()
    {
        var inputIds = new uint[_maxSequenceLength];
        var attMask = new uint[_maxSequenceLength];
        foreach (var text in _corpus)
        {
            RustTokenizer.TokenizeAndGetIds(text, inputIds.AsSpan(), attMask.AsSpan());
        }

        return (inputIds, attMask);
    }
}
