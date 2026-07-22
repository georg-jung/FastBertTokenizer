# Copyright (c) Georg Jung. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# Checks id-level parity of the FastBertTokenizer Python binding against Hugging Face tokenizers
# on the benchmark corpus — the same check BenchPython/verify.py does for flash-tokenizer/tokie.
#
#   dotnet publish ../FastBertTokenizer.Native -c Release -r linux-x64
#   pip install numpy tokenizers
#   python verify.py

import json
import pathlib
import sys

from tokenizers import Tokenizer

from fastberttokenizer import BertTokenizer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "data" / "wiki-simple.json"
TOKENIZER_JSON = REPO_ROOT / "data" / "baai-bge-small-en" / "tokenizer.json"
MAX_LENGTH = 512
MAX_MISMATCH_RATE = 0.001

with open(CORPUS, encoding="utf-8") as f:
    corpus = list(json.load(f).values())

hf = Tokenizer.from_file(str(TOKENIZER_JSON))
hf.enable_truncation(max_length=MAX_LENGTH)
hf_ids = [enc.ids for enc in hf.encode_batch(corpus)]

fbt = BertTokenizer.from_tokenizer_json(TOKENIZER_JSON)
ids, mask = fbt.encode_batch(corpus, max_tokens=MAX_LENGTH, parallel=True)

mismatches = 0
for i, expected in enumerate(hf_ids):
    actual = ids[i, : int(mask[i].sum())].tolist()
    if actual != expected:
        mismatches += 1
        if mismatches <= 3:
            print(f"mismatch in doc {i}:")
            print(f"  hf : {expected[:30]}")
            print(f"  fbt: {actual[:30]}")

rate = mismatches / len(corpus)
print(f"documents:                  {len(corpus)}")
print(f"total tokens (hf):          {sum(len(x) for x in hf_ids)}")
print(f"total tokens (fbt):         {int(mask.sum())}")
print(f"docs with id mismatch:      {mismatches} ({rate:.2%})")

if rate > MAX_MISMATCH_RATE:
    print(f"FAIL: mismatch rate exceeds the expected maximum of {MAX_MISMATCH_RATE:.2%}")
    sys.exit(1)
print("OK")
