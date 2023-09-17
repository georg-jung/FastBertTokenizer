// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.IO.Compression;
using System.Text.Json;

namespace FastBertTokenizer.Tests;

public static class WikipediaSimpleData
{
    private const string Path = "data/wiki-simple.json.br";
    private static readonly Lazy<List<object[]>> _articles = new(GetArticlesImpl);

    public static IEnumerable<object[]> GetArticles() => _articles.Value;

    private static List<object[]> GetArticlesImpl()
    {
        using var fs = File.OpenRead(Path);
        using var uncompress = new BrotliStream(fs, CompressionMode.Decompress);
        return JsonSerializer.Deserialize<Dictionary<int, string>>(uncompress)!.Select(x => new object[] { x.Key, x.Value }).ToList();
    }
}
