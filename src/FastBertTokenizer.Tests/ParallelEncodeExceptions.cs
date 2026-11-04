// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Shouldly;

namespace FastBertTokenizer.Tests;

public class ParallelEncodeExceptions
{
    /// <summary>
    /// Exceptions thrown by the thread pool workers of the parallel batch Encode must propagate
    /// to the caller. Before this was the case, they were unhandled exceptions on thread pool
    /// threads and thus terminated the process (especially relevant when FastBertTokenizer is
    /// embedded in a non-.NET host via FastBertTokenizer.Native).
    /// </summary>
    [Fact]
    public void EncodingParallelWithoutLoadedVocabularyThrows()
    {
        var uut = new BertTokenizer();
        var inputs = Enumerable.Repeat("Lorem ipsum dolor sit amet.", 100).ToArray();
        var iids = new long[inputs.Length * 128];
        var attm = new long[inputs.Length * 128];
        Should.Throw<InvalidOperationException>(() => uut.Encode(inputs, iids, attm, 128));
    }

    [Fact]
    public async Task ExceptionInOneWorkerDoesNotHangTheBatch()
    {
        var uut = new BertTokenizer();
        await uut.LoadTokenizerJsonAsync("data/bert-base-uncased/tokenizer.json");

        // maximumTokens: 1 leaves no room to write the trailing [SEP] token, which makes the
        // encoding workers throw. The call must fail fast instead of deadlocking on the
        // never-signaled countdown or tearing down the process.
        var inputs = Enumerable.Repeat("Lorem ipsum dolor sit amet.", 100).ToArray();
        var iids = new long[inputs.Length];
        var attm = new long[inputs.Length];
        var act = () => uut.Encode(inputs, iids, attm, 1);
        act.ShouldThrow<Exception>();
    }
}
