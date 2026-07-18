# Cross-language tokenizer benchmark: Hugging Face tokenizers (Rust) vs. flash-tokenizer (C++).
#
# Uses the same corpus (simple english wikipedia) and vocabulary (baai-bge-small-en, which uses
# bert-base-uncased's vocab) as the .NET benchmarks in src/Benchmarks, so the tokens/s numbers
# are roughly comparable across languages. Mind the differences though: e.g. process startup,
# corpus loading and the benchmark drivers differ. See verify.py for an id-level parity check.
#
#   pip install -r requirements.txt
#   python bench.py
#
# pyperf runs each benchmark in fresh worker processes; pass e.g. --fast
# (see python bench.py --help) to reduce runtime.

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CORPUS = str(REPO_ROOT / "data" / "wiki-simple.json")
TOKENIZER_JSON = str(REPO_ROOT / "data" / "baai-bge-small-en" / "tokenizer.json")
VOCAB_TXT = str(REPO_ROOT / "data" / "baai-bge-small-en" / "vocab.txt")
MAX_LENGTH = 512

SETUP_CORPUS = f"""
import json
with open({CORPUS!r}, encoding="utf-8") as f:
    corpus = list(json.load(f).values())
"""

SETUP_HF = SETUP_CORPUS + f"""
from tokenizers import Tokenizer
tok = Tokenizer.from_file({TOKENIZER_JSON!r})
tok.enable_truncation(max_length={MAX_LENGTH})
"""

SETUP_FLASH = SETUP_CORPUS + f"""
from flash_tokenizer import BertTokenizerFlash
tok = BertTokenizerFlash({VOCAB_TXT!r}, do_lower_case=True, model_max_length={MAX_LENGTH})
# First batch call is much slower while flash-tokenizer's native thread pool warms up.
tok(corpus[:100], padding="longest", max_length={MAX_LENGTH}, do_multiprocess=True)
"""

if __name__ == "__main__":
    import pyperf

    runner = pyperf.Runner()

    # Single-text loops: effectively single-threaded for both libraries.
    runner.timeit(
        name="hf_tokenizers_singlethreaded",
        stmt="for text in corpus:\n    tok.encode(text)",
        setup=SETUP_HF,
    )
    runner.timeit(
        name="flash_tokenizer_singlethreaded",
        stmt=f"for text in corpus:\n    tok(text, padding=\"longest\", max_length={MAX_LENGTH})",
        setup=SETUP_FLASH,
    )

    # Batch mode: both libraries run their native tokenizers in parallel (Rust/rayon vs. C++
    # thread pool). Note that flash-tokenizer's returned BatchEncoding additionally builds
    # attention_mask/token_type_ids as single-threaded pure-Python lists (roughly a third of
    # its batch time here), while Hugging Face keeps masks in Rust. The numbers thus reflect
    # each library's default user-facing API, not pure native tokenization time.
    runner.timeit(
        name="hf_tokenizers_batch",
        stmt="tok.encode_batch(corpus)",
        setup=SETUP_HF,
    )
    runner.timeit(
        name="flash_tokenizer_batch",
        stmt=f"tok(corpus, padding=\"longest\", max_length={MAX_LENGTH}, do_multiprocess=True)",
        setup=SETUP_FLASH,
    )
