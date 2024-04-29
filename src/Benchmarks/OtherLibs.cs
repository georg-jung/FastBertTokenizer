// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using BERTTokenizers.Base;
using FastBertTokenizer;
using RustLibWrapper;

namespace Benchmarks;

[MemoryDiagnoser]
public class OtherLibs
{
    private readonly string _corpusPath;
    private readonly string _vocabTxtFile;
    private readonly string _tokenizerJsonPath;
    private readonly int _maxSequenceLength;
    private ConcreteUncasedTokenizer _nmZivkovicTokenizer;
    private string[] _corpus = null!;
    private List<string> _nmZivkovicCorpus = null!;
    private readonly BertTokenizer _tokenizer = new();

    public OtherLibs()
        : this("data/wiki-simple.json.br", "data/baai-bge-small-en/vocab.txt", "data/baai-bge-small-en/tokenizer.json", 512)
    {
    }

    public OtherLibs(string corpusPath, string vocabTxtFile, string tokenizerJsonPath, int maxSequenceLength)
    {
        _nmZivkovicTokenizer = new(vocabTxtFile);
        _corpusPath = corpusPath;
        _vocabTxtFile = vocabTxtFile;
        _tokenizerJsonPath = tokenizerJsonPath;
        _maxSequenceLength = maxSequenceLength;
    }

    [GlobalSetup]
    public async Task SetupAsync()
    {
        RustTokenizer.LoadTokenizer(_tokenizerJsonPath, _maxSequenceLength);
        await _tokenizer.LoadTokenizerJsonAsync(_tokenizerJsonPath);
        _corpus = await CorpusReader.ReadBrotliJsonCorpusAsync(_corpusPath);

        _nmZivkovicCorpus = new(_corpus.Length);
        var cnt = 0;
        foreach (var tx in _corpus)
        {
            _corpus[cnt] = tx;

            // this preprocessing gives NMZivkovic/BertTokenizers kind of an unfair advantage, but it throws otherwise
            var nmZivkovicText = tx.Substring(0, Math.Min(tx.Length, 1250)); // NMZivkovic/BertTokenizers throws if text is too long; 1250 works with 512 tokens, 1500 doesn't; 5000 works with 2048 tokens
            nmZivkovicText = Regex.Replace(nmZivkovicText, @"\s+", " "); // required due to bad whitespace processing of NMZivkovic/BertTokenizers
            nmZivkovicText = Regex.Replace(nmZivkovicText, @"[^A-Za-z0-9\s\.\,;:\\/?!#$%()=+\-*\""'–_`<>&^@{}[\]\|~']+", string.Empty); // NMZivkovic/BertTokenizers doesn't handle unknown characters
            _nmZivkovicCorpus.Add(nmZivkovicText);

            cnt++;
        }

        _nmZivkovicTokenizer = new(_vocabTxtFile);
    }

    [Benchmark]
    public IReadOnlyCollection<object> NMZivkovic_BertTokenizers()
    {
        List<object> res = new(_nmZivkovicCorpus.Count);
        foreach (var text in _nmZivkovicCorpus)
        {
            res.Add(_nmZivkovicTokenizer.Encode(_maxSequenceLength, text));
        }

        return res;
    }

    [Benchmark]
    public IReadOnlyCollection<object> FastBertTokenizer_SameDataAsBertTokenizers()
    {
        List<object> res = new(_nmZivkovicCorpus.Count);
        foreach (var text in _nmZivkovicCorpus)
        {
            res.Add(_tokenizer.Encode(text, _maxSequenceLength));
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

    private sealed class ConcreteUncasedTokenizer : UncasedTokenizer
    {
        public ConcreteUncasedTokenizer(string vocabularyFilePath)
            : base(vocabularyFilePath)
        {
        }
    }
}
