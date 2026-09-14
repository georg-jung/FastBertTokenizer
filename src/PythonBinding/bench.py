# Copyright (c) Georg Jung. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# Benchmarks the FastBertTokenizer Python binding against Hugging Face tokenizers and tokie,
# measured from Python the way a Python user would call them. Uses the same corpus (simple
# english wikipedia), vocabulary (baai-bge-small-en) and methodology as
# ../HuggingfaceTokenizer/BenchPython/bench.py, so numbers are comparable.
#
#   dotnet publish ../FastBertTokenizer.Native -c Release -r linux-x64
#   pip install numpy tokenizers tokie pyperf
#   python bench.py
#
# pyperf runs each benchmark in fresh worker processes; pass e.g. --fast to reduce runtime.

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = str(REPO_ROOT / "data" / "wiki-simple.json")
TOKENIZER_JSON = str(REPO_ROOT / "data" / "baai-bge-small-en" / "tokenizer.json")
MAX_LENGTH = 512
PKG_DIR = str(pathlib.Path(__file__).resolve().parent)

SETUP_CORPUS = f"""
import json
with open({CORPUS!r}, encoding="utf-8") as f:
    corpus = list(json.load(f).values())
"""

SETUP_FBT = f"""
import sys
sys.path.insert(0, {PKG_DIR!r})
""" + SETUP_CORPUS + f"""
from fastberttokenizer import BertTokenizer
tok = BertTokenizer.from_tokenizer_json({TOKENIZER_JSON!r})
"""

SETUP_HF = SETUP_CORPUS + f"""
from tokenizers import Tokenizer
tok = Tokenizer.from_file({TOKENIZER_JSON!r})
tok.enable_truncation(max_length={MAX_LENGTH})
"""

SETUP_TOKIE = SETUP_CORPUS + f"""
from tokie import Tokenizer
tok = Tokenizer.from_json({TOKENIZER_JSON!r})
tok.enable_truncation({MAX_LENGTH})
"""

if __name__ == "__main__":
    import pyperf

    runner = pyperf.Runner()

    # Single-text loops: one Python -> native call per document, including each library's
    # per-call overhead. All effectively single-threaded except possibly tokie (see
    # BenchPython/bench.py for the caveat).
    runner.timeit(
        name="fbt_singlethreaded",
        stmt=f"for text in corpus:\n    tok.encode(text, max_tokens={MAX_LENGTH})",
        setup=SETUP_FBT,
    )
    runner.timeit(
        name="hf_tokenizers_singlethreaded",
        stmt="for text in corpus:\n    tok.encode(text)",
        setup=SETUP_HF,
    )
    runner.timeit(
        name="tokie_sequential_calls",
        stmt="for text in corpus:\n    tok.encode(text)",
        setup=SETUP_TOKIE,
    )

    # Batch mode: one call for the whole corpus, each library parallelizing natively.
    # fbt returns padded (n, 512) int64 numpy arrays for input_ids and attention_mask —
    # ready for model consumption — while hf/tokie return per-document Encoding objects.
    runner.timeit(
        name="fbt_batch",
        stmt=f"tok.encode_batch(corpus, max_tokens={MAX_LENGTH}, parallel=True)",
        setup=SETUP_FBT,
    )
    runner.timeit(
        name="fbt_batch_singlethreaded",
        stmt=f"tok.encode_batch(corpus, max_tokens={MAX_LENGTH}, parallel=False)",
        setup=SETUP_FBT,
    )
    runner.timeit(
        name="hf_tokenizers_batch",
        stmt="tok.encode_batch(corpus)",
        setup=SETUP_HF,
    )
    runner.timeit(
        name="tokie_batch",
        stmt="tok.encode_batch(corpus)",
        setup=SETUP_TOKIE,
    )
