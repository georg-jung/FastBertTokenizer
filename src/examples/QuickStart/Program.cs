// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using FastBertTokenizer;

var tok = new BertTokenizer();
await tok.LoadFromHuggingFaceAsync("bert-base-uncased");
var (inputIds, attentionMask, tokenTypeIds) = tok.Encode("Lorem ipsum dolor sit amet.");
Console.WriteLine(string.Join(", ", inputIds.ToArray()));
var decoded = tok.Decode(inputIds.Span);
Console.WriteLine(decoded);

// Output:
// [101,19544,2213,12997,17421,2079,10626,4133,2572,3388,1012,102]
// [CLS] lorem ipsum dolor sit amet. [SEP]
