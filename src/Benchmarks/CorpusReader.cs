// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.IO.Compression;
using System.Text.Json;

namespace Benchmarks;

internal static class CorpusReader
{
    public static async Task<string[]> ReadBrotliJsonCorpusAsync(string filePath)
    {
        using var fs = File.OpenRead(filePath);
        using var uncompress = new BrotliStream(fs, CompressionMode.Decompress);
        var dict = await JsonSerializer.DeserializeAsync<Dictionary<int, string>>(uncompress);

        return dict!.Values.ToArray();
    }
}
