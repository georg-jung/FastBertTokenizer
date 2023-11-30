// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

Console.WriteLine("Path of vocab.txt:");
var vocabTxtPath = Console.ReadLine();
var vocabTxt = await File.ReadAllLinesAsync(vocabTxtPath!);
Console.WriteLine($"Loaded {vocabTxt.Length} tokens from {vocabTxtPath}.");

while (true)
{
    Console.Write("TokenId: ");
    var tokenId = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(tokenId))
    {
        continue;
    }

    if (!int.TryParse(tokenId, out var tokenIdInt))
    {
        Console.WriteLine("Invalid token id.");
        continue;
    }

    if (tokenIdInt < 0 || tokenIdInt >= vocabTxt.Length)
    {
        Console.WriteLine("Token id out of range.");
        continue;
    }

    Console.WriteLine($"Token: {vocabTxt[tokenIdInt]}");
}
