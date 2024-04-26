// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
#if NET8_0_OR_GREATER
using System.Runtime.InteropServices.Marshalling;

[assembly:ExcludeFromCodeCoverage]
#endif

namespace RustLibWrapper;

[ExcludeFromCodeCoverage]
public static partial class RustTokenizer
{
    private static long[]? _inputIds = null;
    private static long[]? _attentionMask = null;
    private static long[]? _tokenTypeIds = null;

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
        if (_inputIds == null || _inputIds.Length < maxTokenIds || _attentionMask is null)
        {
            _inputIds = new long[maxTokenIds];
            _attentionMask = new long[maxTokenIds];
            _tokenTypeIds = new long[maxTokenIds];
        }

        var ids = ArrayPool<uint>.Shared.Rent(maxTokenIds);
        var attentionMask = ArrayPool<uint>.Shared.Rent(maxTokenIds);

        try
        {
            TokenizeAndGetIds(input, ids.AsSpan().Slice(0, maxTokenIds), attentionMask.AsSpan().Slice(0, maxTokenIds));

            for (var i = 0; i < maxTokenIds; i++)
            {
                _inputIds[i] = ids[i];
                _attentionMask[i] = attentionMask[i];
            }

            return (
                _inputIds.AsMemory().Slice(0, maxTokenIds),
                _attentionMask.AsMemory().Slice(0, maxTokenIds),
                _tokenTypeIds.AsMemory().Slice(0, maxTokenIds));
        }
        finally
        {
            ArrayPool<uint>.Shared.Return(ids);
            ArrayPool<uint>.Shared.Return(attentionMask);
        }
    }

#if NET8_0_OR_GREATER
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
#else
    internal static partial class NativeMethods
    {
        // DllImport attribute to import functions from the tokenizer library
        [DllImport("tokenize", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool load_tokenizer(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string tokenizerPath,
            int sequenceLength);

        // DllImport attribute for tokenization and getting IDs
        [DllImport("tokenize", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern unsafe bool tokenize_and_get_ids(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string inputUtf8,
            uint* ids,
            int idsLen,
            uint* attentionMask,
            int attentionMaskLen);
    }
#endif
}
