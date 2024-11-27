import glob
from collections import defaultdict

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--corpus_path", type=str, default='scratch/corpora/english_tok_*.txt',
                   help="Path of the subword tokenized corpus/corpora to read")
parser.add_argument("--line_threshold", type=int, default=1000000,
                   help="Write vocabulary count only after you've seen (some multiple of) this many corpus lines")
parser.add_argument("--vocab_path", type=str, default="english_gpt2.txt",
                   help="Vocab file to write. Expected name: {language}_{tokenization}.txt")

args=parser.parse_args()

fs = glob.glob(args.corpus_path)
vocab = defaultdict(int)
streams = []
j = 0
for f in fs:
 with open(f, "r") as inFile:
  for l in inFile:
   if len(l) < 2:
     continue
   for c in l.strip().split(" "):
    j+=1
    vocab[c] += 1
    if j %args.line_threshold == 0:
      print("Printing vocab")
      with open(args.vocab_path, "w") as outFile:
         d = sorted(list(vocab.items()), key=lambda x:x[1], reverse=True)
         for w, n in d:
           print(f"{w}\t{n}", file=outFile)
print(j)  
