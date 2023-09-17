// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Runtime.InteropServices;

namespace RustLibWrapper;

public static class RustTokenizer
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
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(input + '\0');
        unsafe
        {
            fixed (uint* idsPtr = ids, attentionMaskPtr = attentionMask)
            {
                if (!NativeMethods.tokenize_and_get_ids(bytes, idsPtr, ids.Length, attentionMaskPtr, attentionMask.Length))
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

    internal static class NativeMethods
    {
        [DllImport("tokenize.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern bool load_tokenizer(string tokenizerPath, int sequenceLength);

        [DllImport("tokenize.dll", CallingConvention = CallingConvention.Cdecl)]
        public static unsafe extern bool tokenize_and_get_ids(byte[] inputUtf8, uint* ids, int idsLen, uint* attentionMask, int attentionMaskLen);
    }
}
