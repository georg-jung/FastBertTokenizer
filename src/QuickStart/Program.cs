// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using FastBertTokenizer;

var tok = new BertTokenizer();
var maxTokensForModel = 512;
await tok.LoadVocabularyAsync("vocab.txt", true); // https://huggingface.co/BAAI/bge-small-en/blob/main/vocab.txt
var text = File.ReadAllText("TextFile.txt");
var (inputIds, attentionMask, tokenTypeIds) = tok.Tokenize(text, maxTokensForModel);
Console.WriteLine(string.Join(", ", inputIds.ToArray().Select(x => x.ToString())));
