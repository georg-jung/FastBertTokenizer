# Checks id-level parity between Hugging Face tokenizers and flash-tokenizer on the benchmark
# corpus, so the speed comparison in bench.py is honest about correctness differences.
#
#   pip install -r requirements.txt
#   python verify.py

import json

from bench import CORPUS, MAX_LENGTH, TOKENIZER_JSON, VOCAB_TXT
from flash_tokenizer import BertTokenizerFlash
from tokenizers import Tokenizer

with open(CORPUS, encoding="utf-8") as f:
    corpus = list(json.load(f).values())

hf = Tokenizer.from_file(TOKENIZER_JSON)
hf.enable_truncation(max_length=MAX_LENGTH)
flash = BertTokenizerFlash(VOCAB_TXT, do_lower_case=True, model_max_length=MAX_LENGTH)

hf_ids = [enc.ids for enc in hf.encode_batch(corpus)]
flash_ids = flash(corpus, padding="longest", max_length=MAX_LENGTH, do_multiprocess=True).input_ids

assert len(hf_ids) == len(flash_ids)
mismatches = sum(1 for a, b in zip(hf_ids, flash_ids) if a != b)
total_tokens_hf = sum(len(x) for x in hf_ids)
total_tokens_flash = sum(len(x) for x in flash_ids)

print(f"documents:                    {len(corpus)}")
print(f"total tokens (hf tokenizers): {total_tokens_hf}")
print(f"total tokens (flash):         {total_tokens_flash}")
print(f"documents with id mismatch:   {mismatches} ({mismatches / len(corpus):.2%})")
