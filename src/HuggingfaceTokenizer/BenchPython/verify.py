# Checks id-level parity between Hugging Face tokenizers and flash-tokenizer on the benchmark
# corpus, so the speed comparison in bench.py is honest about correctness differences.
#
#   pip install -r requirements.txt
#   python verify.py

import json
import sys

from bench import CORPUS, MAX_LENGTH, TOKENIZER_JSON, VOCAB_TXT
from flash_tokenizer import BertTokenizerFlash
from tokenizers import Tokenizer

# Fail if parity regresses clearly beyond the known ~0.4% (flash-tokenizer 1.2.0 on this corpus),
# so CI notices instead of burying it in logs. Adjust deliberately if flash's behavior changes.
MAX_MISMATCH_RATE = 0.01

with open(CORPUS, encoding="utf-8") as f:
    corpus = list(json.load(f).values())

hf = Tokenizer.from_file(TOKENIZER_JSON)
hf.enable_truncation(max_length=MAX_LENGTH)
flash = BertTokenizerFlash(VOCAB_TXT, do_lower_case=True, model_max_length=MAX_LENGTH)

PAD_ID = 0


def strip_padding(ids):
    """Remove trailing [PAD] tokens so the comparison measures tokenizer parity, not padding.

    flash-tokenizer 1.2.0 returns ragged, unpadded lists for padding="longest", but this keeps
    the check honest even if that behavior changes. hf.encode_batch never pads (no padding
    configured), so stripping both sides is symmetric.
    """
    end = len(ids)
    while end > 0 and ids[end - 1] == PAD_ID:
        end -= 1
    return ids[:end]


hf_ids = [strip_padding(enc.ids) for enc in hf.encode_batch(corpus)]
flash_ids = flash(corpus, padding="longest", max_length=MAX_LENGTH, do_multiprocess=True).input_ids
flash_ids = [strip_padding(ids) for ids in flash_ids]

assert len(hf_ids) == len(flash_ids)
mismatches = sum(1 for a, b in zip(hf_ids, flash_ids) if a != b)
total_tokens_hf = sum(len(x) for x in hf_ids)
total_tokens_flash = sum(len(x) for x in flash_ids)

mismatch_rate = mismatches / len(corpus)
print(f"documents:                    {len(corpus)}")
print(f"total tokens (hf tokenizers): {total_tokens_hf}")
print(f"total tokens (flash):         {total_tokens_flash}")
print(f"documents with id mismatch:   {mismatches} ({mismatch_rate:.2%})")

if mismatch_rate > MAX_MISMATCH_RATE:
    print(f"FAIL: mismatch rate exceeds the expected maximum of {MAX_MISMATCH_RATE:.2%}")
    sys.exit(1)
