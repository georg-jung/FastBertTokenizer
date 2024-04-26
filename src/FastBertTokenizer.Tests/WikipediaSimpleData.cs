// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace FastBertTokenizer.Tests;

public static class WikipediaSimpleData
{
    private const string Path = "data/wiki-simple.json";
    private static readonly Lazy<List<object[]>> _articles = new(GetArticlesImpl);
    private static readonly Lazy<Dictionary<int, string>> _articlesDict = new(GetArticlesDictImpl);

    public static IEnumerable<object[]> GetArticles() => _articles.Value;

    public static IEnumerable<object[]> GetArticlesDict() => [new object[] { _articlesDict.Value }];

    private static List<object[]> GetArticlesImpl()
    {
        return _articlesDict.Value!.Select(x => new object[] { x.Key, x.Value }).ToList();
    }

    private static Dictionary<int, string> GetArticlesDictImpl()
    {
        using var fs = File.OpenRead(Path);
        return JsonSerializer.Deserialize<Dictionary<int, string>>(fs)!;
    }
}
