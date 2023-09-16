import pyperf

setup = """
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('C:/Users/georg/git/bge-small-en/')

corpus_folder = 'C:/Users/georg/simplewikicorpus_more'
files = os.listdir(corpus_folder)

corpus = []

for file in files:
    with open(os.path.join(corpus_folder, file), 'r', encoding='utf-8') as f:
        tx = f.read()
        corpus.append(tx)
"""

stmt = """
batch_enc = tokenizer(corpus, padding=False, truncation=True, max_length=512)

# uncomment to print total number of tokens produced
# should not be executed during benchmarking if the results are to be compared
# total_tokens = sum([len(entry) for entry in batch_enc["input_ids"]])
# print(f"Total number of tokens: {total_tokens}")
"""

runner = pyperf.Runner()
runner.timeit(name="tokenize",
              stmt=stmt,
              setup=setup)
