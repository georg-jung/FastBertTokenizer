// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.IO.Compression;
using System.Text.Json;
using System.Threading.Channels;
using FastBertTokenizer;
using Microsoft.ML.OnnxRuntime;
using SimpleSimd;

const string Path = "../../../../data/wiki-simple.json.br";

static async Task<Dictionary<int, string>> GetArticlesDictAsync()
{
    using var fs = File.OpenRead(Path);
    using var uncompress = new BrotliStream(fs, CompressionMode.Decompress);
    return (await JsonSerializer.DeserializeAsync<Dictionary<int, string>>(uncompress))!;
}

Console.WriteLine("Welcome to local semantic search!");
Console.WriteLine("Loading wikipedia articles...");

var articles = await GetArticlesDictAsync();

using var liteDb = new LiteDB.LiteDatabase("semantic.litedb");
var semanticMemory = liteDb.GetCollection<SemanticMemory>();
semanticMemory.EnsureIndex(x => x.TokenizedRange, unique: true);
semanticMemory.EnsureIndex(x => x.TokenizedRange.Key);

Console.WriteLine("Loading model...");

var sessOpt = new SessionOptions();
var session = new InferenceSession(@"D:\UAE-Large-V1\model.onnx", sessOpt);

Console.WriteLine("Loading tokenizer...");

var tok = new BertTokenizer();
await tok.LoadFromHuggingFaceAsync("WhereIsAI/UAE-Large-V1");

Console.WriteLine("Finished loading.");

Console.WriteLine("Please enter 1 to generate remaining embeddings or 2 to perform semantic search queries against the existing embeddings.");
Console.Write("Enter choice (1 or 2): ");
var choice = Console.ReadLine();
if (choice == "1")
{
    var cts = new CancellationTokenSource();
    var genEmbeddingsTask = Task.Run(() => GenerateEmbeddings(cts.Token));
    Console.WriteLine("Press any key to stop generating embeddings...");
    Console.WriteLine("After requesting to stop, embedding generation will continue for some time but won't feed any new inputs.");
    Console.ReadKey();
    Console.WriteLine("Stopping embedding generation...");
    cts.Cancel();
    await genEmbeddingsTask;
    Console.WriteLine("Generation of embeddings ended.");
}
else if (choice == "2")
{
    while (true)
    {
        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.Write("Enter query: ");
        Console.ForegroundColor = ConsoleColor.Gray;
        var q = Console.ReadLine();
        await SemanticSearch(q ?? string.Empty);
    }
}
else
{
    Console.WriteLine("Invalid choice. Exiting.");
}

static void ToUnitLen(Span<float> vector)
{
    Span<float> intermediate = stackalloc float[1024];
    SimdOps.Multiply(vector, vector, intermediate);
    var len = SimdOps.Sum<float>(intermediate);
    len = MathF.Sqrt(len);
    if (len > 1.001 || len < 0.999)
    {
        SimdOps.Divide(vector, len, vector);
    }
}

async Task GenerateEmbeddings(CancellationToken cancellationToken)
{
    var articleChannel = Channel.CreateBounded<(int, string)>(1);
    async Task FillChannel()
    {
        foreach (var (key, text) in articles)
        {
            if (semanticMemory.Exists(x => x.TokenizedRange.Key == key && !x.TokenizedRange.LastTokenizedWordStartIndex.HasValue))
            {
                continue;
            }

            await articleChannel.Writer.WriteAsync((key, text));
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }
        }

        articleChannel.Writer.Complete();
    }

    var channelTask = Task.Run(FillChannel);

    const int BatchSize = 5;
    using var tokenTypeIdsTensor = OrtValue.CreateTensorValueFromMemory(new long[BatchSize * 512], [BatchSize, 512]);
    using var output = OrtValue.CreateTensorValueFromMemory(new float[BatchSize * 512 * 1024], [BatchSize, 512, 1024]);
    await foreach (var batch in tok.CreateAsyncBatchEnumerator(articleChannel.Reader, 512, BatchSize, 25, maxDegreeOfParallelism: 2))
    {
        var inputIds = batch.InputIds;
        var attentionMask = batch.AttentionMask;

        using var iidsTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputIds, [BatchSize, 512]);
        using var attmTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, attentionMask, [BatchSize, 512]);
        await session.RunAsync(
            new RunOptions() { },
            ["input_ids", "attention_mask", "token_type_ids"],
            [iidsTensor, attmTensor, tokenTypeIdsTensor],
            ["last_hidden_state"],
            [output]);

        for (var i = 0; i < BatchSize; i++)
        {
            var corr = batch.OutputCorrelation.Span[i];
            if (!corr.HasValue)
            {
                continue;
            }

            var embedding = output.GetTensorDataAsSpan<float>().Slice(i * 512 * 1024, 1024).ToArray();
            ToUnitLen(embedding);
            semanticMemory.Upsert(new SemanticMemory
            {
                TokenizedRange = corr.Value,
                Embedding = embedding,
            });
            Console.WriteLine($"{corr.Value.Key}: {corr.Value.Offset}-{corr.Value.LastTokenizedWordStartIndex}");
        }
    }

    await channelTask;
}

async Task SemanticSearch(string query)
{
    using var tokenTypeIdsTensor = OrtValue.CreateTensorValueFromMemory(new long[512], [1, 512]);
    using var output = OrtValue.CreateTensorValueFromMemory(new float[512 * 1024], [1, 512, 1024]);

    var fullQueryString = $"Represent this sentence for searching relevant passages: {query}";
    var queryModelInputs = tok.Encode(fullQueryString, padTo: 512);
    var queryInputIds = queryModelInputs.InputIds;
    var queryAttentionMask = queryModelInputs.AttentionMask;

    using var iidsTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, queryInputIds, [1, 512]);
    using var attmTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, queryAttentionMask, [1, 512]);
    await session.RunAsync(
        new RunOptions() { },
        ["input_ids", "attention_mask", "token_type_ids"],
        [iidsTensor, attmTensor, tokenTypeIdsTensor],
        ["last_hidden_state"],
        [output]);

    var queryEmbedding = output.GetTensorDataAsSpan<float>().Slice(0, 1024).ToArray();
    ToUnitLen(queryEmbedding);
    var results = semanticMemory.FindAll()
        .Select(x => (Entity: x, Distance: SimdOps.Dot(x.Embedding.AsSpan(), (ReadOnlySpan<float>)queryEmbedding)))
        .OrderByDescending(x => x.Distance)
        .Take(5);
    foreach (var r in results)
    {
        Console.WriteLine("===========================");
        var article = articles[r.Entity.TokenizedRange.Key];
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"{r.Distance} {r.Entity.TokenizedRange} {article.Substring(0, Math.Min(50, article.Length)).ReplaceLineEndings(" ")}");
        Console.ForegroundColor = ConsoleColor.Gray;
        Console.WriteLine("-- Relevant section: --");
        Console.WriteLine($"{article.Substring(r.Entity.TokenizedRange.Offset, (r.Entity.TokenizedRange.LastTokenizedWordStartIndex ?? article.Length) - r.Entity.TokenizedRange.Offset)}");
    }
}

record class SemanticMemory
{
    [LiteDB.BsonId]
    public required TokenizedRange<int> TokenizedRange { get; set; }

    public required float[] Embedding { get; set; }
}
