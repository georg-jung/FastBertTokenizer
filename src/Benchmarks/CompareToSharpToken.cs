// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;
using FastBertTokenizer;
using SharpToken;

namespace Benchmarks;

[Config(typeof(Config))]
[MemoryDiagnoser]
public class CompareToSharpToken
{
    private readonly string _corpusPath;
    private readonly string _vocabTxtFile;
    private readonly int _maxSequenceLength;
    private readonly GptEncoding _encoding;
    private readonly BertTokenizer _tokenizer = new();
    private string[] _corpus = null!;

    // FastBertTokenizer requires us to specify the maximum sequence length supported by the target model,
    // while SharpToken just tokenizes the whole text. When tokenizing our corpus with FastBertTokenizer,
    // the longest single document produces 11,196 tokens. We need to use a maximum sequence length of at
    // least that, otherwise, our comparison will be unfair because SharpToken tokenizes more text then
    // FastBertTokenizer. I chose 16384 ( = 8192 * 2) as a maximum sequence length here, beeing the smallest
    // power of two that is larger than the required number. Choosing exactly 11,196 wouldn't be a very
    // realistic comparison, as that exact number wouldn't be known beforehand in reality. Thus, choosing
    // the next power of two seems like a reasonable compromise to me. The number chosen shouldn't have a
    // to big impact on the results of FastBertTokenizer though.
    public CompareToSharpToken()
        : this("data/wiki-simple.json.br", "data/baai-bge-small-en/vocab.txt", 16384)
    {
    }

    public CompareToSharpToken(string corpusPath, string vocabTxtFile, int maxSequenceLength)
    {
        _corpusPath = corpusPath;
        _vocabTxtFile = vocabTxtFile;
        _maxSequenceLength = maxSequenceLength;
        _encoding = GptEncoding.GetEncoding("cl100k_base");
    }

    [GlobalSetup]
    public async Task SetupAsync()
    {
        await _tokenizer.LoadVocabularyAsync(_vocabTxtFile, true);
        _corpus = await CorpusReader.ReadBrotliJsonCorpusAsync(_corpusPath);
    }

    [Benchmark]
    public IReadOnlyCollection<object> SharpTokenFullArticles()
    {
        List<object> res = new(_corpus.Length);
        foreach (var text in _corpus)
        {
            _encoding.Encode(text);
        }

        return res;
    }

    [Benchmark]
    public IReadOnlyCollection<object> FastBertTokenizerFullArticles()
    {

        List<object> res = new(_corpus.Length);
        foreach (var text in _corpus)
        {
            var (iid, attM, _) = _tokenizer.Encode(text, _maxSequenceLength);
        }
        
        return res;
    }

    private sealed class Config : ManualConfig
    {
        public Config()
        {
            var baseJob = Job.Default;
            var localJob = baseJob.WithCustomBuildConfiguration("LocalBuild");
            AddJob(localJob.WithRuntime(CoreRuntime.Core80));
        }
    }
}
