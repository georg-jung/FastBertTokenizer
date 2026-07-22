# FastBertTokenizer from Python (prototype)

A prototype Python binding for FastBertTokenizer. Instead of re-implementing the tokenizer,
the existing .NET library is compiled to a self-contained native shared library with
[NativeAOT](https://learn.microsoft.com/dotnet/core/deploying/native-aot/) — no .NET runtime
required at run time — and consumed from Python via `ctypes` + numpy. The exported C ABI lives
in [`../FastBertTokenizer.Native`](../FastBertTokenizer.Native).

```bash
# 1. Build the native library (once per platform; needs the .NET SDK + clang on Linux/macOS)
dotnet publish ../FastBertTokenizer.Native -c Release -r linux-x64   # or osx-arm64, win-x64, ...

# 2. Use it
pip install numpy
python -c "
from fastberttokenizer import BertTokenizer
tok = BertTokenizer.from_tokenizer_json('../../data/baai-bge-small-en/tokenizer.json')
input_ids, attention_mask = tok.encode_batch(['Lorem ipsum dolor sit amet.'], max_tokens=512)
print(input_ids[0][:12])
print(tok.decode(input_ids[0][attention_mask[0] == 1]))
"
```

`encode_batch` writes directly into pre-allocated numpy `int64` arrays of shape
`(len(texts), max_tokens)` — the exact layout models expect — so there is no per-document
Python object overhead. ctypes releases the GIL during the native call; with `parallel=True`
tokenization uses all cores via the .NET thread pool.

## Correctness

[`verify.py`](verify.py) checks id-level parity against Hugging Face `tokenizers` on the
benchmark corpus (15,000 simple english wikipedia articles), analogous to
[`../HuggingfaceTokenizer/BenchPython/verify.py`](../HuggingfaceTokenizer/BenchPython/verify.py).
Result: 3 of 15,000 documents (0.02 %) differ — the same known corpus-level differences the
.NET library itself has, i.e. the binding adds no drift.

## Speed

[`bench.py`](bench.py) mirrors the methodology of
[`../HuggingfaceTokenizer/BenchPython/bench.py`](../HuggingfaceTokenizer/BenchPython/bench.py)
(same corpus, same vocabulary, truncation at 512 tokens, pyperf). Measured in a 4-vCPU
Linux x64 container (`pyperf --fast`, so treat as indicative rather than rigorous):

| Called from Python                  | Single-text loop | Batch (parallel) |
|-------------------------------------|-----------------:|-----------------:|
| **FastBertTokenizer (this binding)**|       **542 ms** |       **211 ms** |
| tokie (Rust)                        |          1.11 s  |           830 ms |
| Hugging Face tokenizers (Rust)      |          11.0 s  |           3.01 s |

FastBertTokenizer's batch call additionally ran at 531 ms with `parallel=False`. At ~3.66 M
tokens for the corpus, 211 ms is ≈17 M tokens/s from Python — including the Python → native
UTF-8 marshalling and numpy allocation, i.e. the interop cost does not eat the library's
advantage.

## What a real release would still need

* **Wheel packaging**: bundle the published native library per platform
  (`manylinux`/`musllinux` x64+arm64, macOS universal, Windows x64) into platform wheels.
  No compiler needed at `pip install` time — wheels just repackage the AOT binaries.
* **ICU**: NativeAOT loads ICU dynamically for Unicode normalization/casing. Manylinux does
  not guarantee libicu, so wheels should either link ICU statically
  (`StaticICULinking=true`) or ship app-local ICU.
* **API polish**: streaming/`overflowing` batch APIs (`CreateBatchEnumerator`), loading from
  the Hugging Face Hub (best done in Python via `huggingface_hub`), and a `cffi`/HPy layer if
  ctypes overhead ever becomes measurable (it is negligible for batch calls).
