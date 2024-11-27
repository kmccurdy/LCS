# Process partioned corpus data 
# expected file format is {unique_prefix}_{train/dev/test}.txt

import time
import numpy as np
import os
import glob
import random
import gzip 

np.random.seed(42)

# this function is also used in compute_lcs.py
def readCorpusLine(line, EOS=True):
    line = [x.strip() for x in line.strip().split(" _")] 
    if EOS:
        line += ["<EOS>"] 
    return line
    
def load(data_path, partition="test"):
  with open(f"{data_path}_{partition}.txt", "r") as inFile:
    collected = 0
    chunk = [None for _ in range(1000000)]
    startTime = time.time()
    for line in inFile:
        if len(line) < 3:
            continue
        line = readCorpusLine(line)
        chunk[collected:collected+len(line)] = line
        collected += len(line)
        if collected > 1000000:
           yield chunk
           chunk = [None for _ in range(1000000)]
           collected = 0
           startTime = time.time()
    yield chunk[:collected]
  

def training(data_path):
  return load(data_path, "train")

def dev(data_path):
  return load(data_path, "dev")

def test(data_path):
  return load(data_path, "test")


