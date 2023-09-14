// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using BERTTokenizers.Base;
using FastBertTokenizer;

namespace Benchmarks;

[MemoryDiagnoser]
/*
[PerfCollectProfiler(performExtraBenchmarksRun: false)]
[EtwProfiler(performExtraBenchmarksRun: false)]
[EventPipeProfiler(EventPipeProfile.CpuSampling)] // for speedscope files
*/
public class TokenizeSpeed
{
    private static (List<string> Corpus, int MaxLen)? _cache;
    private readonly List<string> _corpus;
    private readonly ConcreteUncasedTokenizer _otherLibTokenizer;
    private readonly BertTokenizer _tokenizer;
    private readonly int _maxSequenceLength;

    public TokenizeSpeed()
        : this("C:\\Users\\georg\\simplewikicorpus", "Vocabularies/baai-bge-small-en-vocab.txt", 2048)
    {
    }

    public TokenizeSpeed(string corpusFolder, string vocabTxtFile, int maxSequenceLength)
    {
        if (_cache is { } cache)
        {
            _corpus = cache.Corpus;
            _maxSequenceLength = cache.MaxLen;
        }
        else
        {
            var files = Directory.GetFiles(corpusFolder);
            _corpus = new(files.Length);
            foreach (var file in files)
            {
                var tx = File.ReadAllText(file);
                tx = tx.Substring(0, Math.Min(tx.Length, 5000)); // other lib throw if text is too long
                tx = Regex.Replace(tx, @"\s+", " "); // required due to bad whitespace processing of other lib
                tx = Regex.Replace(tx, @"[^A-Za-z0-9\s\.\,;:\\/?!#$%()=+\-*\""'–_`<>&^@{}[\]\|~']+", string.Empty); // other lib doesn't handle unknown characters */
                _corpus.Add(tx);
            }
        }

        _otherLibTokenizer = new(vocabTxtFile);
        _tokenizer = new();

        _tokenizer.LoadVocabularyAsync(vocabTxtFile, true).GetAwaiter().GetResult();
        _maxSequenceLength = maxSequenceLength;
    }

    [Benchmark]
    public IReadOnlyCollection<object> OtherLib()
    {
        List<object> res = new(_corpus.Count);
        foreach (var text in _corpus)
        {
            res.Add(_otherLibTokenizer.Encode(_maxSequenceLength, text));
        }

        return res;
    }

    [Benchmark]
    public IReadOnlyCollection<object> FastBertTokenizerAllocating()
    {
        List<object> res = new(_corpus.Count);
        foreach (var text in _corpus)
        {
            res.Add(_tokenizer.Tokenize(text, _maxSequenceLength));
        }

        return res;
    }

    [Benchmark]
    public object FastBertTokenizerMemReuse()
    {
        var iids = new long[_maxSequenceLength];
        var attm = new long[_maxSequenceLength];
        foreach (var text in _corpus)
        {
            _tokenizer.Tokenize(text, iids, attm);
        }

        return (iids, attm);
    }

    private sealed class ConcreteUncasedTokenizer : UncasedTokenizer
    {
        public ConcreteUncasedTokenizer(string vocabularyFilePath)
            : base(vocabularyFilePath)
        {
        }
    }
}
