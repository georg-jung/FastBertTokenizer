// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
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
    private readonly BertTokenizer _tokenizer;
    private readonly string _corpusPath;
    private readonly string _vocabTxtFile;
    private readonly int _maxSequenceLength;
    private string[] _corpus = null!;

    public TokenizeSpeed()
        : this("data/wiki-simple.json.br", "data/baai-bge-small-en-vocab.txt", 512)
    {
    }

    public TokenizeSpeed(string corpusPath, string vocabTxtFile, int maxSequenceLength)
    {
        _corpusPath = corpusPath;
        _vocabTxtFile = vocabTxtFile;
        _maxSequenceLength = maxSequenceLength;
        _tokenizer = new();
    }

    [GlobalSetup]
    public async Task SetupAsync()
    {
        using var sr = File.OpenText(_vocabTxtFile);
        _tokenizer.LoadVocabulary(sr, true);
        _corpus = await CorpusReader.ReadBrotliJsonCorpusAsync(_corpusPath);
    }

    [Benchmark(Baseline = true)]
    public IReadOnlyCollection<object> SinglethreadedAllocating()
    {
        List<object> res = new(_corpus.Length);
        foreach (var text in _corpus)
        {
            res.Add(_tokenizer.Tokenize(text, _maxSequenceLength));
        }

        return res;
    }

    [Benchmark]
    public object SingleThreadedMemReuse()
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
    public IReadOnlyCollection<(Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds)> MultithreadedAllocating()
    {
        // this might be interesting to benchmark but doesn't make much sense as a real world use case
        List<(Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds)> res = new(_corpus.Length);
        var x = _corpus.AsParallel().AsOrdered().Select(x => _tokenizer.Tokenize(x, _maxSequenceLength));
        res.AddRange(x);
        return res;
    }

    [Benchmark]
    public (Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds) MultithreadedMemReuse()
    {
        var batchSize = 1000;
        var iids = new long[_maxSequenceLength * batchSize];
        var attm = new long[_maxSequenceLength * batchSize];
        var toktyp = new long[_maxSequenceLength * batchSize];
        Array.Fill(toktyp, 0);

        var corpMem = _corpus.AsMemory();
        for (var i = 0; i < corpMem.Length; i += batchSize)
        {
            var len = Math.Min(batchSize, corpMem.Length - i);
            var batchSeqLen = _maxSequenceLength * len;
            var iidsM = iids.AsMemory(0, batchSeqLen);
            var attmM = attm.AsMemory(0, batchSeqLen);
            _tokenizer.Tokenize(corpMem.Slice(i, len), iidsM, attmM, _maxSequenceLength);
        }

        return (iids.AsMemory(), attm.AsMemory(), toktyp.AsMemory());
    }
}
