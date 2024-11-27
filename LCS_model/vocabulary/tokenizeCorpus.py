# Tokenizes a text corpus using a pretrained Huggingface tokenizer.

# Note: only tested with GPT2 models, so recommended use with that family.
# Should work with other models, but look out for potential issues.

import torch
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_read_path", type=str, default="scratch/corpora/english_wikipedia.txt",
                   help="Path to read corpus")
parser.add_argument("--corpus_write_path", type=str, default="scratch/corpora/english_tok_gpt2.txt",
                   help="Path to write subword-tokenized corpus. Expected filename: {language}_tok_{tokenizer}.txt")
parser.add_argument("--tokenizer", type=str, default="gpt2", 
                    help="Huggingface identifier to pretrained tokenizer for your model") 

args=parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

# for consistency - this function should exactly match compute_surprisal/compute_lcs.py
def processCorpusLine(line, tknzr=tokenizer):
    return [tknzr.decode(x).replace(" ","_") for x in tknzr.encode(line, return_tensors='pt').view(-1).cpu().numpy().tolist()]

i = 0
with open(args.corpus_read_path, "r") as inFile:
 with open(args.corpus_write_path, "w") as outFile:
  for line in inFile:
    i += 1
    if i % 1000 == 0:
      print(i)
    initial_text = processCorpusLine(line)
    print((" ".join(initial_text)), file=outFile)

