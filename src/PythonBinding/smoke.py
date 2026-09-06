# Copyright (c) Georg Jung. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# Smoke tests for the FastBertTokenizer.Native C ABI contract, exercised through ctypes:
# handle lifecycle, error codes, last-error semantics, encode/decode buffer conventions.
# Complements verify.py (corpus-level correctness) - this checks the ABI itself.
#
#   dotnet publish ../FastBertTokenizer.Native -c Release -r linux-x64
#   pip install numpy
#   python smoke.py

import ctypes
import pathlib
import sys

import numpy as np

from fastberttokenizer import BertTokenizer, NativeError

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOKENIZER_JSON = REPO_ROOT / "data" / "baai-bge-small-en" / "tokenizer.json"

OK = 0
ERR_EXCEPTION = -1
ERR_INVALID_HANDLE = -2
ERR_INVALID_ARGUMENT = -3
ERR_BUFFER_TOO_SMALL = -4

failures = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  {status}: {name}" + (f" ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def int64_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def encode1(handle, text, max_tokens, ids, mask):
    """Encode a single text through the batch entry point (count = 1, sequential)."""
    texts = (ctypes.c_char_p * 1)(text)
    lens = (ctypes.c_int32 * 1)(len(text))
    return lib.fbt_encode_batch(handle, texts, lens, 1, max_tokens, int64_ptr(ids), int64_ptr(mask), 0)


tok = BertTokenizer.from_tokenizer_json(TOKENIZER_JSON)
lib = tok._lib
ids = np.empty(8, dtype=np.int64)
mask = np.empty(8, dtype=np.int64)

print("handle lifecycle:")
check("create returns non-zero handle", tok._handle != 0)
check("zero handle -> invalid handle", encode1(0, b"x", 8, ids, mask) == ERR_INVALID_HANDLE)
check("garbage handle -> invalid handle", encode1(987654321, b"x", 8, ids, mask) == ERR_INVALID_HANDLE)
stale = lib.fbt_create()
lib.fbt_destroy(stale)
lib.fbt_destroy(stale)  # double destroy must be survivable
check("stale handle after destroy -> invalid handle", encode1(stale, b"x", 8, ids, mask) == ERR_INVALID_HANDLE)

print("loading:")
unloaded = lib.fbt_create()
rc = lib.fbt_load_tokenizer_json(unloaded, b"{ not json", 10)
check("bad tokenizer.json -> exception code", rc == ERR_EXCEPTION)
check("error message available", bool(lib.fbt_last_error()))
big_ids = np.empty((4, 128), dtype=np.int64)
big_mask = np.empty((4, 128), dtype=np.int64)
texts = (ctypes.c_char_p * 4)(b"a", b"b", b"c", b"d")
lens = (ctypes.c_int32 * 4)(1, 1, 1, 1)
rc = lib.fbt_encode_batch(unloaded, texts, lens, 4, 128, int64_ptr(big_ids), int64_ptr(big_mask), 1)
check("parallel encode without vocabulary -> error, process survives", rc == ERR_EXCEPTION)
lib.fbt_destroy(unloaded)

print("encode:")
encode1(tok._handle, b"hi", 8, ids, mask)
check("attention mask matches count", int(mask.sum()) == 3)
check("wrapper mask sum", int(tok.encode("hi", max_tokens=8)[1].sum()) == 3)
check("wrapper truncates to max_tokens", int(tok.encode("lorem ipsum dolor " * 20, max_tokens=8)[1].sum()) == 8)
check("empty batch (parallel) -> empty arrays", tok.encode_batch([], max_tokens=8)[0].shape == (0, 8))

print("encode_batch argument validation:")
bad_lens = (ctypes.c_int32 * 4)(1, -1, 1, 1)
rc = lib.fbt_encode_batch(tok._handle, texts, bad_lens, 4, 128, int64_ptr(big_ids), int64_ptr(big_mask), 0)
check("negative per-item length -> invalid argument", rc == ERR_INVALID_ARGUMENT)
null_texts = (ctypes.c_char_p * 1)(None)
one_len = (ctypes.c_int32 * 1)(3)
rc = lib.fbt_encode_batch(tok._handle, null_texts, one_len, 1, 128, int64_ptr(big_ids), int64_ptr(big_mask), 0)
check("null text with positive length -> invalid argument", rc == ERR_INVALID_ARGUMENT)
rc = lib.fbt_encode_batch(tok._handle, None, lens, 4, 128, int64_ptr(big_ids), int64_ptr(big_mask), 0)
check("null texts array -> invalid argument", rc == ERR_INVALID_ARGUMENT)

print("decode:")
encode1(tok._handle, b"hi", 8, ids, mask)
required = ctypes.c_int64()
rc = lib.fbt_decode(tok._handle, int64_ptr(ids), 3, None, 0, ctypes.byref(required))
check("size query -> buffer too small + required size", rc == ERR_BUFFER_TOO_SMALL and required.value > 0)
buf = ctypes.create_string_buffer(required.value)
rc = lib.fbt_decode(tok._handle, int64_ptr(ids), 3, buf, required.value, ctypes.byref(required))
check("decode roundtrip", rc == OK and buf.raw[: required.value].decode() == "[CLS] hi [SEP]")
rc = lib.fbt_decode(tok._handle, int64_ptr(ids), 3, buf, -1, ctypes.byref(required))
check("negative outByteLen -> invalid argument", rc == ERR_INVALID_ARGUMENT)
check("wrapper empty decode", tok.decode([]) == "")

print("last_error semantics:")
encode1(0, b"x", 8, ids, mask)  # provoke a failure
lib.fbt_last_error.restype = ctypes.c_void_p  # compare pointers, not bytes
p1, p2 = lib.fbt_last_error(), lib.fbt_last_error()
lib.fbt_last_error.restype = ctypes.c_char_p
check("pointer stable while no new failure", p1 == p2)

print("wrapper-level errors:")
try:
    BertTokenizer().encode_batch(["x"], max_tokens=512)
    check("unloaded wrapper raises NativeError", False)
except NativeError:
    check("unloaded wrapper raises NativeError", True)
try:
    tok.encode_batch(["x"], max_tokens=0)
    check("max_tokens=0 raises ValueError", False)
except ValueError:
    check("max_tokens=0 raises ValueError", True)

if failures:
    print(f"\nFAIL: {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("\nOK: all ABI smoke tests passed")
