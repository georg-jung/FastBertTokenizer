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
//[EventPipeProfiler(EventPipeProfile.CpuSampling)]
public class TokenizeSpeed
{
    private readonly List<string> _corpus;
    private readonly List<string> _otherLibCorpus;
    private readonly ConcreteUncasedTokenizer _otherLibTokenizer;
    private readonly BertTokenizer _tokenizer;
    private readonly int _maxSequenceLength;

    public TokenizeSpeed()
        : this("C:\\Users\\georg\\simplewikicorpus_more", "Vocabularies/baai-bge-small-en-vocab.txt", 512)
    {
    }

    public TokenizeSpeed(string corpusFolder, string vocabTxtFile, int maxSequenceLength)
    {
        var files = Directory.GetFiles(corpusFolder);
        _corpus = new(files.Length);
        _otherLibCorpus = new(files.Length);
        foreach (var file in files)
        {
            var tx = File.ReadAllText(file);
            _corpus.Add(tx);

            // this preprocessing gives the other lib kind of an unfair advantage, but it throws otherwise
            var otherLib = tx.Substring(0, Math.Min(tx.Length, 1250)); // other lib throw if text is too long; 1250 works with 512 tokens, 1500 doesn't; 5000 works with 2048 tokens
            otherLib = Regex.Replace(otherLib, @"\s+", " "); // required due to bad whitespace processing of other lib
            otherLib = Regex.Replace(otherLib, @"[^A-Za-z0-9\s\.\,;:\\/?!#$%()=+\-*\""'–_`<>&^@{}[\]\|~']+", string.Empty); // other lib doesn't handle unknown characters
            _otherLibCorpus.Add(otherLib);
        }

        _otherLibTokenizer = new(vocabTxtFile);
        _tokenizer = new();

        _tokenizer.LoadVocabularyAsync(vocabTxtFile, true).GetAwaiter().GetResult();
        _maxSequenceLength = maxSequenceLength;
    }

    [Benchmark]
    public IReadOnlyCollection<object> OtherLib()
    {
        List<object> res = new(_otherLibCorpus.Count);
        foreach (var text in _otherLibCorpus)
        {
            res.Add(_otherLibTokenizer.Encode(_maxSequenceLength, text));
        }

        return res;
    }

    [Benchmark(Baseline = true)]
    public IReadOnlyCollection<object> FastBertTokenizerSinglethreadedAllocating()
    {
        List<object> res = new(_corpus.Count);
        foreach (var text in _corpus)
        {
            res.Add(_tokenizer.Tokenize(text, _maxSequenceLength));
        }

        return res;
    }

    [Benchmark]
    public object FastBertTokenizerSingleThreadedMemReuse()
    {
        var iids = new long[_maxSequenceLength];
        var attm = new long[_maxSequenceLength];
        var toktyp = new long[_maxSequenceLength];
        Array.Fill(toktyp, 0);
        foreach (var text in _corpus)
        {
            _tokenizer.Tokenize(text, iids, attm);
        }

        return (iids, attm, toktyp);
    }

    [Benchmark]
    public IReadOnlyCollection<(Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds)> FastBertTokenizerMultithreadedAllocating()
    {
        // this might be interesting to benchmark but doesn't make much sense as a real world use case
        List<(Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds)> res = new(_corpus.Count);
        var x = _corpus.AsParallel().AsOrdered().Select(x => _tokenizer.Tokenize(x, _maxSequenceLength));
        res.AddRange(x);
        return res;
    }

    [Benchmark]
    public (Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds) FastBertTokenizerMultithreadedMemReuse()
    {
        var batchSize = 100;
        var iids = new long[_maxSequenceLength * batchSize];
        var attm = new long[_maxSequenceLength * batchSize];
        var toktyp = new long[_maxSequenceLength * batchSize];
        Array.Fill(toktyp, 0);

        foreach (var batch in _corpus.Buffer(batchSize).Cast<IReadOnlyList<string>>())
        {
            var batchSeqLen = _maxSequenceLength * batch.Count;
            var iidsM = iids.AsMemory(0, batchSeqLen);
            var attmM = attm.AsMemory(0, batchSeqLen);
            _tokenizer.Tokenize(batch, iidsM, attmM, _maxSequenceLength);
        }

        return (iids.AsMemory(), attm.AsMemory(), toktyp.AsMemory());
    }

    private sealed class ConcreteUncasedTokenizer : UncasedTokenizer
    {
        public ConcreteUncasedTokenizer(string vocabularyFilePath)
            : base(vocabularyFilePath)
        {
        }
    }
}
