// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace FastBertTokenizer.Native;

/// <summary>
/// Flat C ABI over <see cref="BertTokenizer"/> for consumption from non-.NET languages
/// (Python via ctypes/cffi, Node via ffi, etc.) when published as a NativeAOT shared library.
///
/// Conventions:
/// <list type="bullet">
/// <item>All text crosses the boundary as UTF-8 (pointer + byte length, no NUL termination required).</item>
/// <item>Functions returning <c>int</c> return 0 on success and a negative error code on failure;
/// <c>fbt_last_error</c> returns a UTF-8 error message for the last failure on the calling thread.</item>
/// <item>Output buffers are allocated by the caller. For encode, <c>input_ids</c>/<c>attention_mask</c>
/// must hold <c>count * max_tokens</c> int64 values; results are written row-major, padded per row —
/// i.e. exactly the memory layout of a C-contiguous numpy array of shape (count, max_tokens).</item>
/// </list>
/// </summary>
internal static unsafe class NativeExports
{
    private const int Ok = 0;
    private const int ErrException = -1;
    private const int ErrInvalidHandle = -2;
    private const int ErrInvalidArgument = -3;
    private const int ErrBufferTooSmall = -4;

    // Handles are opaque ids into this table rather than raw GCHandles: resolving an arbitrary
    // caller-supplied value through GCHandle.FromIntPtr is undefined behavior (a garbage value
    // can hard-abort the process with an access violation), while a dictionary lookup makes
    // invalid, stale and double-destroyed handles reliably fail with ErrInvalidHandle.
    private static readonly ConcurrentDictionary<nint, BertTokenizer> Instances = new();
    private static long _nextHandle;

    [ThreadStatic]
    private static string? _lastError;

    [ThreadStatic]
    private static string? _lastErrorMaterialized;

    [ThreadStatic]
    private static nint _lastErrorUtf8;

    /// <summary>Create a new tokenizer instance. Returns an opaque handle, or 0 on failure.</summary>
    [UnmanagedCallersOnly(EntryPoint = "fbt_create", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static nint Create()
    {
        try
        {
            var handle = (nint)Interlocked.Increment(ref _nextHandle);
            Instances[handle] = new BertTokenizer();
            return handle;
        }
        catch (Exception ex)
        {
            SetLastError(ex);
            return 0;
        }
    }

    /// <summary>
    /// Destroy a tokenizer instance created by <c>fbt_create</c>. Passing 0 is a no-op.
    /// Best-effort: an invalid or already-destroyed handle is reported via <c>fbt_last_error</c>
    /// instead of failing.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "fbt_destroy", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static void Destroy(nint handle)
    {
        if (handle != 0 && !Instances.TryRemove(handle, out _))
        {
            _lastError = "handle does not refer to a live tokenizer instance.";
        }
    }

    /// <summary>
    /// Load a Hugging Face tokenizer.json (UTF-8 bytes). The caller keeps ownership of the buffer;
    /// it is not referenced after this call returns.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "fbt_load_tokenizer_json", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int LoadTokenizerJson(nint handle, byte* json, nint jsonByteLen)
    {
        try
        {
            if (Resolve(handle) is not { } tok)
            {
                return ErrInvalidHandle;
            }

            if (json is null || jsonByteLen < 0)
            {
                _lastError = "json must not be null and jsonByteLen must be >= 0.";
                return ErrInvalidArgument;
            }

            using var stream = new UnmanagedMemoryStream(json, jsonByteLen);
            tok.LoadTokenizerJson(stream);
            return Ok;
        }
        catch (Exception ex)
        {
            SetLastError(ex);
            return ErrException;
        }
    }

    /// <summary>
    /// Load a vocab.txt (UTF-8 bytes). <paramref name="convertInputToLowercase"/> != 0 enables lowercasing,
    /// as for uncased models. The caller keeps ownership of the buffer.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "fbt_load_vocab_txt", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int LoadVocabTxt(nint handle, byte* vocab, nint vocabByteLen, int convertInputToLowercase)
    {
        try
        {
            if (Resolve(handle) is not { } tok)
            {
                return ErrInvalidHandle;
            }

            if (vocab is null || vocabByteLen < 0)
            {
                _lastError = "vocab must not be null and vocabByteLen must be >= 0.";
                return ErrInvalidArgument;
            }

            using var stream = new UnmanagedMemoryStream(vocab, vocabByteLen);
            using var reader = new StreamReader(stream, Encoding.UTF8);
            tok.LoadVocabulary(reader, convertInputToLowercase != 0);
            return Ok;
        }
        catch (Exception ex)
        {
            SetLastError(ex);
            return ErrException;
        }
    }

    /// <summary>
    /// Encode a batch of UTF-8 texts. Writes <c>count * max_tokens</c> int64 values to
    /// <paramref name="inputIds"/> and <paramref name="attentionMask"/> (row-major, one padded row
    /// per input). <paramref name="tokenTypeIds"/> may be null; if given it is zero-filled.
    /// <paramref name="parallel"/> != 0 tokenizes on the .NET thread pool (recommended for large batches).
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "fbt_encode_batch", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int EncodeBatch(
        nint handle,
        byte** texts,
        int* textByteLens,
        int count,
        int maxTokens,
        long* inputIds,
        long* attentionMask,
        long* tokenTypeIds,
        int parallel)
    {
        try
        {
            if (Resolve(handle) is not { } tok)
            {
                return ErrInvalidHandle;
            }

            if (texts is null || textByteLens is null || inputIds is null || attentionMask is null
                || count < 0 || maxTokens <= 0)
            {
                _lastError = "texts, textByteLens, inputIds and attentionMask must not be null; count must be >= 0 and maxTokens > 0.";
                return ErrInvalidArgument;
            }

            long totalLen = (long)count * maxTokens;
            if (totalLen > int.MaxValue)
            {
                _lastError = $"count * maxTokens must be <= {int.MaxValue} per call; split the batch.";
                return ErrInvalidArgument;
            }

            for (var i = 0; i < count; i++)
            {
                if (textByteLens[i] < 0 || (texts[i] is null && textByteLens[i] > 0))
                {
                    _lastError = $"texts[{i}] is null with a positive length or textByteLens[{i}] is negative.";
                    return ErrInvalidArgument;
                }
            }

            var inputs = MaterializeStrings(texts, textByteLens, count, parallel != 0);

            if (parallel != 0)
            {
                using var idsOwner = new PointerMemoryManager<long>(inputIds, (int)totalLen);
                using var maskOwner = new PointerMemoryManager<long>(attentionMask, (int)totalLen);
                tok.Encode(inputs, idsOwner.Memory, maskOwner.Memory, maxTokens);
            }
            else
            {
                for (var i = 0; i < count; i++)
                {
                    var offset = (long)i * maxTokens;
                    tok.Encode(
                        inputs[i],
                        new Span<long>(inputIds + offset, maxTokens),
                        new Span<long>(attentionMask + offset, maxTokens),
                        padTo: maxTokens);
                }
            }

            if (tokenTypeIds is not null)
            {
                new Span<long>(tokenTypeIds, (int)totalLen).Clear();
            }

            return Ok;
        }
        catch (Exception ex)
        {
            SetLastError(ex);
            return ErrException;
        }
    }

    /// <summary>
    /// Encode a single UTF-8 text. Writes up to <paramref name="maxTokens"/> int64 values to
    /// <paramref name="inputIds"/> and <paramref name="attentionMask"/> (padded to <paramref name="maxTokens"/>).
    /// Returns the number of non-padding tokens produced, or a negative error code.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "fbt_encode", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int EncodeSingle(nint handle, byte* text, int textByteLen, int maxTokens, long* inputIds, long* attentionMask)
    {
        try
        {
            if (Resolve(handle) is not { } tok)
            {
                return ErrInvalidHandle;
            }

            if (text is null || inputIds is null || attentionMask is null || textByteLen < 0 || maxTokens <= 0)
            {
                _lastError = "text, inputIds and attentionMask must not be null; textByteLen must be >= 0 and maxTokens > 0.";
                return ErrInvalidArgument;
            }

            var input = Encoding.UTF8.GetString(text, textByteLen);
            var maskSpan = new Span<long>(attentionMask, maxTokens);
            tok.Encode(input, new Span<long>(inputIds, maxTokens), maskSpan, padTo: maxTokens);

            // With padTo set, Encode returns the padded length; the documented return value is
            // the non-padding count, which the attention mask (1s followed by 0s) tells us.
            var nonPadded = maskSpan.IndexOf(0L);
            return nonPadded < 0 ? maxTokens : nonPadded;
        }
        catch (Exception ex)
        {
            SetLastError(ex);
            return ErrException;
        }
    }

    /// <summary>
    /// Decode token ids back to text. Always writes the required UTF-8 byte count to
    /// <paramref name="outRequiredByteLen"/>. If <paramref name="outUtf8"/> is non-null and
    /// <paramref name="outByteLen"/> is large enough, writes the UTF-8 result (no NUL terminator)
    /// and returns 0; otherwise returns <c>ErrBufferTooSmall</c> (-4) without writing text.
    /// Call with <paramref name="outUtf8"/> = null to query the required size.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "fbt_decode", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int Decode(nint handle, long* tokenIds, int count, byte* outUtf8, long outByteLen, long* outRequiredByteLen)
    {
        try
        {
            if (Resolve(handle) is not { } tok)
            {
                return ErrInvalidHandle;
            }

            if (tokenIds is null || count < 0 || outRequiredByteLen is null || outByteLen < 0)
            {
                _lastError = "tokenIds and outRequiredByteLen must not be null; count and outByteLen must be >= 0.";
                return ErrInvalidArgument;
            }

            var decoded = tok.Decode(new ReadOnlySpan<long>(tokenIds, count));
            var byteCount = Encoding.UTF8.GetByteCount(decoded);
            *outRequiredByteLen = byteCount;
            if (outUtf8 is null || outByteLen < byteCount)
            {
                return ErrBufferTooSmall;
            }

            Encoding.UTF8.GetBytes(decoded, new Span<byte>(outUtf8, byteCount));
            return Ok;
        }
        catch (Exception ex)
        {
            SetLastError(ex);
            return ErrException;
        }
    }

    /// <summary>
    /// Returns a NUL-terminated UTF-8 message describing the last error on the calling thread,
    /// or 0 if none occurred. The pointer stays valid until the next failing call on the same thread.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "fbt_last_error", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static nint LastError()
    {
        try
        {
            var msg = _lastError;
            if (msg is null)
            {
                return 0;
            }

            // Repeated queries must return the same buffer while no new failure occurred - the
            // documented lifetime is "valid until the next failing call on this thread". Only
            // re-encode (and free the previous buffer) once the message actually changed.
            if (_lastErrorUtf8 != 0 && ReferenceEquals(_lastErrorMaterialized, msg))
            {
                return _lastErrorUtf8;
            }

            if (_lastErrorUtf8 != 0)
            {
                NativeMemory.Free((void*)_lastErrorUtf8);
                _lastErrorUtf8 = 0;
            }

            var bytes = Encoding.UTF8.GetBytes(msg);
            var buf = (byte*)NativeMemory.Alloc((nuint)bytes.Length + 1);
            bytes.CopyTo(new Span<byte>(buf, bytes.Length));
            buf[bytes.Length] = 0;
            _lastErrorUtf8 = (nint)buf;
            _lastErrorMaterialized = msg;
            return _lastErrorUtf8;
        }
        catch (Exception)
        {
            // Even allocation failure must not escape an [UnmanagedCallersOnly] export;
            // "no message available" is the only safe answer here.
            return 0;
        }
    }

    private static void SetLastError(Exception ex) => _lastError = ex.ToString();

    private static BertTokenizer? Resolve(nint handle)
    {
        if (Instances.TryGetValue(handle, out var tok))
        {
            return tok;
        }

        _lastError = handle == 0
            ? "handle must not be 0."
            : "handle does not refer to a live tokenizer instance.";
        return null;
    }

    private static string[] MaterializeStrings(byte** texts, int* textByteLens, int count, bool parallel)
    {
        var result = new string[count];

        // Converting UTF-8 to .NET strings is a real share of small-batch runtime, so parallelize
        // it alongside parallel tokenization. Pointers can't be captured by lambdas; smuggle them
        // through nints.
        if (parallel && count >= 256)
        {
            var textsAddr = (nint)texts;
            var lensAddr = (nint)textByteLens;
            Parallel.For(0, count, i =>
            {
                var t = ((byte**)textsAddr)[i];
                var l = ((int*)lensAddr)[i];
                result[i] = l == 0 || t is null ? string.Empty : Encoding.UTF8.GetString(t, l);
            });
        }
        else
        {
            for (var i = 0; i < count; i++)
            {
                var t = texts[i];
                var l = textByteLens[i];
                result[i] = l == 0 || t is null ? string.Empty : Encoding.UTF8.GetString(t, l);
            }
        }

        return result;
    }

    /// <summary>Exposes caller-owned native memory as <see cref="Memory{T}"/> without copying.</summary>
    private sealed class PointerMemoryManager<T> : MemoryManager<T>
        where T : unmanaged
    {
        private readonly nint _ptr;
        private readonly int _length;

        public PointerMemoryManager(T* ptr, int length)
        {
            _ptr = (nint)ptr;
            _length = length;
        }

        public override Span<T> GetSpan() => new((void*)_ptr, _length);

        public override MemoryHandle Pin(int elementIndex = 0) => new((T*)_ptr + elementIndex);

        public override void Unpin()
        {
        }

        protected override void Dispose(bool disposing)
        {
        }
    }
}
