// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Shouldly;

namespace FastBertTokenizer.Tests;

/// <summary>
/// A word piece can never be longer than the longest entry of the vocabulary, so the greedy longest-prefix
/// match starts there instead of at the full word. These tests pin down that this does not change what is
/// emitted - especially for candidates that are exactly as long as the longest entry - and that the work
/// spent on a single word stays bounded by its length.
/// </summary>
public class LongWords
{
    // Token ids are line numbers. The longest prefix is "abcd" (4 chars), the longest suffix "xyz" (3 chars).
    private const string VocabTxt = "[UNK]\n[CLS]\n[SEP]\n[PAD]\na\nab\nabcd\n##x\n##xyz\n";

    private const long Unk = 0;
    private const long Cls = 1;
    private const long Sep = 2;
    private const long A = 4;
    private const long Ab = 5;
    private const long Abcd = 6;
    private const long SuffixX = 7;
    private const long SuffixXyz = 8;

    [Theory]

    // Candidates that are exactly as long as the longest prefix/suffix must still be looked up.
    [InlineData("abcd", new[] { Cls, Abcd, Sep })]
    [InlineData("axyz", new[] { Cls, A, SuffixXyz, Sep })]

    // Words longer than the longest prefix are matched piece by piece, as before.
    [InlineData("abcdx", new[] { Cls, Abcd, SuffixX, Sep })]
    [InlineData("abcdxyz", new[] { Cls, Abcd, SuffixXyz, Sep })]
    [InlineData("abcdxyzx", new[] { Cls, Abcd, SuffixXyz, SuffixX, Sep })]

    // Shorter than the longest prefix, thus unaffected by the cap.
    [InlineData("ab", new[] { Cls, Ab, Sep })]
    [InlineData("abc", new[] { Cls, Unk, Sep })]

    // A word the vocabulary can't represent is unknown as a whole, no matter where the matching fails.
    [InlineData("zzzz", new[] { Cls, Unk, Sep })]
    [InlineData("abzz", new[] { Cls, Unk, Sep })]
    [InlineData("abcdxyzzz", new[] { Cls, Unk, Sep })]
    public void MatchesAtTheLengthOfTheLongestVocabularyEntry(string input, long[] expected)
    {
        var tokenizer = CreateTokenizer(VocabTxt);
        tokenizer.Encode(input, 512).InputIds.ToArray().ShouldBe(expected);
    }

    [Fact]
    public void VocabularyWithoutSuffixesYieldsUnknownForMultiPieceWords()
    {
        // The longest suffix is 0 chars long here. Starting at that length needs to behave just like
        // shrinking a candidate that never matches: the word as a whole is unknown.
        var tokenizer = CreateTokenizer("[UNK]\n[CLS]\n[SEP]\n[PAD]\na\nab\n");
        tokenizer.Encode("ab", 512).InputIds.ToArray().ShouldBe(new[] { Cls, Ab, Sep });
        tokenizer.Encode("aab", 512).InputIds.ToArray().ShouldBe(new[] { Cls, Unk, Sep });
    }

    [Fact]
    public void PathologicallyLongWordIsTokenizedInBoundedTime()
    {
        // A real vocabulary on purpose: what a failed lookup costs depends on the dictionary
        // implementation, and for just a handful of entries some of them compare candidates by
        // length instead of hashing them.
        var tokenizer = new BertTokenizer();
        using var vocabTxt = File.OpenText("data/bert-base-uncased/vocab.txt");
        tokenizer.LoadVocabulary(vocabTxt, convertInputToLowercase: true);
        var clsSep = tokenizer.Encode(string.Empty, 512).InputIds.ToArray();

        // Every failed lookup hashes the whole candidate, so without starting at the length of the
        // longest vocabulary entry this single word takes hours on net8.0/netstandard2.0 and ~160 ms
        // on net9.0+ (where FrozenDictionary rejects overlong keys by length before hashing). Starting
        // at the cap it is well below a millisecond. The timeout is generous on purpose - it is not a
        // measurement, it just turns a regression into a failing test instead of a hanging test run.
        var word = new string('a', 100_000);
        var inputIds = Should.CompleteIn(() => tokenizer.Encode(word, 512).InputIds.ToArray(), TimeSpan.FromSeconds(30));

        // The word is cut off at maximumTokens; only [CLS] and [SEP] are known here without hardcoding ids.
        inputIds.Length.ShouldBe(512);
        inputIds[0].ShouldBe(clsSep[0]);
        inputIds[511].ShouldBe(clsSep[1]);
    }

    private static BertTokenizer CreateTokenizer(string vocabTxt)
    {
        var tokenizer = new BertTokenizer();
        tokenizer.LoadVocabulary(new StringReader(vocabTxt), convertInputToLowercase: true);
        return tokenizer;
    }
}
