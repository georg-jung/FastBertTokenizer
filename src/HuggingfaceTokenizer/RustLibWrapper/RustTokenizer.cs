// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.Marshalling;

namespace RustLibWrapper;

public static partial class RustTokenizer
{
    public static void LoadTokenizer(string path, int sequenceLength)
    {
        if (!NativeMethods.load_tokenizer(path, sequenceLength))
        {
            throw new InvalidOperationException("Failed to load the tokenizer.");
        }
    }

    public static void TokenizeAndGetIds(string input, Span<uint> ids, Span<uint> attentionMask)
    {
        unsafe
        {
            fixed (uint* idsPtr = ids, attentionMaskPtr = attentionMask)
            {
                if (!NativeMethods.tokenize_and_get_ids(input, idsPtr, ids.Length, attentionMaskPtr, attentionMask.Length))
                {
                    throw new InvalidOperationException("Failed to tokenize the input string.");
                }
            }
        }
    }

    public static (Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds) TokenizeAndGetIds(string input, int maxTokenIds = 512)
    {
        var ids = ArrayPool<uint>.Shared.Rent(maxTokenIds);
        var attentionMask = ArrayPool<uint>.Shared.Rent(maxTokenIds);

        try
        {
            TokenizeAndGetIds(input, ids, attentionMask);

            return (new Memory<long>(ids.Select(x => (long)x).ToArray()), new Memory<long>(attentionMask.Select(x => (long)x).ToArray()), new Memory<long>(new long[ids.Length]));
        }
        finally
        {
            ArrayPool<uint>.Shared.Return(ids);
            ArrayPool<uint>.Shared.Return(attentionMask);
        }
    }

    internal static partial class NativeMethods
    {
        [LibraryImport("tokenize")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static partial bool load_tokenizer(
            [MarshalUsing(typeof(Utf8StringMarshaller))] string tokenizerPath,
            int sequenceLength);

        [LibraryImport("tokenize")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(System.Runtime.CompilerServices.CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static unsafe partial bool tokenize_and_get_ids(
            [MarshalUsing(typeof(Utf8StringMarshaller))] string inputUtf8,
            uint* ids,
            int idsLen,
            uint* attentionMask,
            int attentionMaskLen);
    }
}
