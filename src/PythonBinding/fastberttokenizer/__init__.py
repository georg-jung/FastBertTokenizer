# Copyright (c) Georg Jung. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""Python binding for FastBertTokenizer via its NativeAOT-compiled C ABI.

Prototype status: loads the shared library produced by
``dotnet publish src/FastBertTokenizer.Native -c Release -r <rid>`` and exposes
batch encoding into numpy arrays, single-text encoding, and decoding.

The heavy lifting happens outside the GIL (ctypes releases it for the duration
of the native call), so ``encode_batch(..., parallel=True)`` uses all cores.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import platform
from typing import Sequence

import numpy as np

__all__ = ["BertTokenizer", "NativeError"]

_ERROR_CODES = {
    -1: "exception in native code",
    -2: "invalid handle",
    -3: "invalid argument",
    -4: "buffer too small",
}


class NativeError(RuntimeError):
    """An error reported by the FastBertTokenizer native library."""


def _default_lib_names() -> list[str]:
    system = platform.system()
    if system == "Windows":
        return ["FastBertTokenizer.Native.dll"]
    if system == "Darwin":
        return ["FastBertTokenizer.Native.dylib"]
    return ["FastBertTokenizer.Native.so"]


def _default_rid() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
    osname = {"Windows": "win", "Darwin": "osx"}.get(system, "linux")
    return f"{osname}-{arch}"


def _find_library() -> str:
    override = os.environ.get("FBT_NATIVE_LIB")
    if override:
        if os.path.isfile(override):
            return override
        raise FileNotFoundError(f"FBT_NATIVE_LIB points to {override!r}, which does not exist.")

    names = _default_lib_names()
    pkg_dir = pathlib.Path(__file__).resolve().parent
    # In a built wheel the native lib would ship inside the package; during repo-local
    # development we fall back to the dotnet publish output.
    repo_root = pkg_dir.parents[2]
    publish_dir = (
        repo_root / "bin" / "FastBertTokenizer.Native" / "Release" / "net10.0" / _default_rid() / "publish"
    )
    candidates = [pkg_dir / n for n in names] + [publish_dir / n for n in names]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "FastBertTokenizer native library not found. Looked for "
        + ", ".join(str(c) for c in candidates)
        + ". Build it with: dotnet publish src/FastBertTokenizer.Native -c Release -r "
        + _default_rid()
    )


def _load(lib_path: str | None) -> ctypes.CDLL:
    lib = ctypes.CDLL(lib_path or _find_library())

    lib.fbt_create.restype = ctypes.c_void_p
    lib.fbt_create.argtypes = []
    lib.fbt_destroy.restype = None
    lib.fbt_destroy.argtypes = [ctypes.c_void_p]
    lib.fbt_last_error.restype = ctypes.c_char_p
    lib.fbt_last_error.argtypes = []
    lib.fbt_load_tokenizer_json.restype = ctypes.c_int32
    lib.fbt_load_tokenizer_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ssize_t]
    lib.fbt_load_vocab_txt.restype = ctypes.c_int32
    lib.fbt_load_vocab_txt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_int32]
    lib.fbt_encode_batch.restype = ctypes.c_int32
    lib.fbt_encode_batch.argtypes = [
        ctypes.c_void_p,                    # handle
        ctypes.POINTER(ctypes.c_char_p),    # texts
        ctypes.POINTER(ctypes.c_int32),     # text byte lens
        ctypes.c_int32,                     # count
        ctypes.c_int32,                     # max_tokens
        ctypes.POINTER(ctypes.c_int64),     # input_ids out
        ctypes.POINTER(ctypes.c_int64),     # attention_mask out
        ctypes.POINTER(ctypes.c_int64),     # token_type_ids out (nullable)
        ctypes.c_int32,                     # parallel
    ]
    lib.fbt_encode.restype = ctypes.c_int32
    lib.fbt_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.fbt_decode.restype = ctypes.c_int32
    lib.fbt_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    ]
    return lib


class BertTokenizer:
    """WordPiece/BERT tokenizer backed by FastBertTokenizer's native library."""

    def __init__(self, lib_path: str | None = None) -> None:
        self._lib = _load(lib_path)
        handle = self._lib.fbt_create()
        if not handle:
            self._raise("fbt_create")
        self._handle = handle

    @classmethod
    def from_tokenizer_json(cls, path: str | os.PathLike, lib_path: str | None = None) -> "BertTokenizer":
        """Create a tokenizer from a Hugging Face tokenizer.json file."""
        self = cls(lib_path)
        data = pathlib.Path(path).read_bytes()
        rc = self._lib.fbt_load_tokenizer_json(self._handle, data, len(data))
        if rc != 0:
            self._raise("fbt_load_tokenizer_json", rc)
        return self

    @classmethod
    def from_vocab_txt(
        cls, path: str | os.PathLike, lowercase: bool = True, lib_path: str | None = None
    ) -> "BertTokenizer":
        """Create a tokenizer from a vocab.txt file, as for bert-base-uncased with lowercase=True."""
        self = cls(lib_path)
        data = pathlib.Path(path).read_bytes()
        rc = self._lib.fbt_load_vocab_txt(self._handle, data, len(data), 1 if lowercase else 0)
        if rc != 0:
            self._raise("fbt_load_vocab_txt", rc)
        return self

    def encode_batch(
        self,
        texts: Sequence[str],
        max_tokens: int = 512,
        parallel: bool = True,
        return_token_type_ids: bool = False,
    ):
        """Encode a batch of texts.

        Returns (input_ids, attention_mask) — or (input_ids, attention_mask, token_type_ids)
        if return_token_type_ids is set — as numpy int64 arrays of shape (len(texts), max_tokens),
        padded per row. The arrays can be fed directly to e.g. onnxruntime or torch.
        """
        count = len(texts)
        input_ids = np.empty((count, max_tokens), dtype=np.int64)
        attention_mask = np.empty((count, max_tokens), dtype=np.int64)
        token_type_ids = np.empty((count, max_tokens), dtype=np.int64) if return_token_type_ids else None

        encoded = [t.encode("utf-8") for t in texts]
        text_array = (ctypes.c_char_p * count)(*encoded)
        len_array = (ctypes.c_int32 * count)(*(len(b) for b in encoded))

        int64_ptr = ctypes.POINTER(ctypes.c_int64)
        rc = self._lib.fbt_encode_batch(
            self._handle,
            text_array,
            len_array,
            count,
            max_tokens,
            input_ids.ctypes.data_as(int64_ptr),
            attention_mask.ctypes.data_as(int64_ptr),
            token_type_ids.ctypes.data_as(int64_ptr) if token_type_ids is not None else None,
            1 if parallel else 0,
        )
        if rc != 0:
            self._raise("fbt_encode_batch", rc)
        if token_type_ids is not None:
            return input_ids, attention_mask, token_type_ids
        return input_ids, attention_mask

    def encode(self, text: str, max_tokens: int = 512):
        """Encode a single text. Returns (input_ids, attention_mask) as 1-D numpy int64 arrays."""
        input_ids = np.empty(max_tokens, dtype=np.int64)
        attention_mask = np.empty(max_tokens, dtype=np.int64)
        data = text.encode("utf-8")
        int64_ptr = ctypes.POINTER(ctypes.c_int64)
        rc = self._lib.fbt_encode(
            self._handle,
            data,
            len(data),
            max_tokens,
            input_ids.ctypes.data_as(int64_ptr),
            attention_mask.ctypes.data_as(int64_ptr),
        )
        if rc < 0:
            self._raise("fbt_encode", rc)
        return input_ids, attention_mask

    def decode(self, token_ids) -> str:
        """Decode token ids (any int sequence or numpy array) back to text."""
        ids = np.ascontiguousarray(token_ids, dtype=np.int64)
        int64_ptr = ctypes.POINTER(ctypes.c_int64)
        required = ctypes.c_int64()
        rc = self._lib.fbt_decode(
            self._handle, ids.ctypes.data_as(int64_ptr), ids.size, None, 0, ctypes.byref(required)
        )
        if rc == 0:
            return ""
        if rc != -4:  # anything but "buffer too small" is a real error
            self._raise("fbt_decode", rc)
        buf = ctypes.create_string_buffer(required.value)
        rc = self._lib.fbt_decode(
            self._handle, ids.ctypes.data_as(int64_ptr), ids.size, buf, required.value, ctypes.byref(required)
        )
        if rc != 0:
            self._raise("fbt_decode", rc)
        return buf.raw[: required.value].decode("utf-8")

    def close(self) -> None:
        """Free the native tokenizer. The object must not be used afterwards."""
        if getattr(self, "_handle", None):
            self._lib.fbt_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "BertTokenizer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _raise(self, func: str, rc: int | None = None):
        msg = self._lib.fbt_last_error()
        detail = msg.decode("utf-8", errors="replace") if msg else "unknown error"
        code = f" ({_ERROR_CODES.get(rc, rc)})" if rc is not None else ""
        raise NativeError(f"{func} failed{code}: {detail}")
