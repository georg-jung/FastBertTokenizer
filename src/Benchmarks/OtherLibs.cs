// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json.Nodes;
using BenchmarkDotNet.Attributes;
using BlingFire;
using FastBertTokenizer;
using Microsoft.ML.Tokenizers;
using BertTokenizer = FastBertTokenizer.BertTokenizer;

namespace Benchmarks;

/// <summary>
/// Compares FastBertTokenizer to other tokenizer libraries usable from .NET. All benchmarks
/// tokenize the same corpus with the same vocabulary (baai-bge-small-en, which uses
/// bert-base-uncased's vocab) and truncate to the same maximum sequence length. Note that the
/// compared libraries don't do exactly the same work: FastBertTokenizer emits input_ids and
/// attention_mask, Microsoft.ML.Tokenizers and Tokenizers.DotNet emit just input_ids, while
/// the Hugging Face tokenizers library (behind Tokenizers.DotNet) computes offsets and more.
/// For interop-free numbers of non-.NET tokenizers see the cross-language benchmarks in
/// ../HuggingfaceTokenizer/BenchPython and ../HuggingfaceTokenizer/BenchRust.
/// </summary>
[Config(typeof(CompareConfig))]
[MemoryDiagnoser]
public class OtherLibs
{
    private readonly string _corpusPath;
    private readonly string _vocabTxtFile;
    private readonly string _tokenizerJsonPath;
    private readonly int _maxSequenceLength;
    private const int BertBaseUncasedUnkId = 100;
    private readonly BertTokenizer _tokenizer = new();
    private string[] _corpus = null!;
    private Microsoft.ML.Tokenizers.BertTokenizer _mlTokenizer = null!;
    private Tokenizers.DotNet.Tokenizer _tokenizersDotNetTokenizer = null!;
    private string? _truncatingTokenizerJsonPath;
    private ulong _blingFireModel;
    private byte[] _blingFireUtf8Buffer = null!;

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
        _truncatingTokenizerJsonPath = Path.Combine(Path.GetTempPath(), $"fastberttokenizer-bench-truncating-tokenizer-{Guid.NewGuid():N}.json");
        await File.WriteAllTextAsync(_truncatingTokenizerJsonPath, tokenizerJson.ToJsonString());
        _tokenizersDotNetTokenizer = new(vocabPath: _truncatingTokenizerJsonPath);

        // BlingFire doesn't read vocab.txt; it needs its precompiled FSM for the same
        // bert-base-uncased vocabulary. It takes utf-8 input, so reuse one buffer for that.
        _blingFireModel = BlingFireUtils.LoadModel("data/blingfire/bert_base_tok.bin");
        _blingFireUtf8Buffer = new byte[_corpus.Max(x => Encoding.UTF8.GetByteCount(x))];
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_truncatingTokenizerJsonPath is not null)
        {
            File.Delete(_truncatingTokenizerJsonPath);
        }

        if (_blingFireModel != 0)
        {
            BlingFireUtils.FreeModel(_blingFireModel);
        }
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

    // Mind that BlingFire does less than the others here: it emits input_ids only, without
    // [CLS]/[SEP], and its precompiled model is not built from our vocab.txt (it agrees with
    // Hugging Face on ~99.9% of tokens per flash-tokenizer's measurements).
    [Benchmark]
    public object BlingFire()
    {
        Span<int> ids = stackalloc int[_maxSequenceLength];
        var cnt = 0;
        foreach (var text in _corpus)
        {
            var utf8Len = Encoding.UTF8.GetBytes(text.AsSpan(), _blingFireUtf8Buffer);
            cnt += BlingFireUtils2.TextToIds(_blingFireModel, _blingFireUtf8Buffer.AsSpan(0, utf8Len), utf8Len, ids, _maxSequenceLength, BertBaseUncasedUnkId);
        }

        return cnt;
    }
}
