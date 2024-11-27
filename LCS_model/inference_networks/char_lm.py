# Script to train an inference network modeling next word prediction conditional on a lossy context representation 
# q(w|c'), approximating p(w|c') 

# Requirements:
#  - Tokenized and partioned training corpus, specified in --corpus_path
#    - Pretrained tokenizer used on corpus, specified in --tokenizer
#    - Tokenized corpus vocabulary, parent directory specified in --vocab_dir


import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__file__ = __file__.split("/")[-1]

import random
import torch
import numpy
import math
from torch.autograd import Variable
import time
import corpusIterator  
from transformers import AutoTokenizer

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - ln %(lineno)d: %(message)s')
logger = logging.getLogger()

import argparse
parser = argparse.ArgumentParser()

# language-specific parameters
parser.add_argument("--language", dest="language", type=str, default="English")
parser.add_argument("--maxSeqLength", type=int, default=80)
parser.add_argument("--tokenizer", type=str, default="gpt2") # identifier for subword tokenizer used
parser.add_argument("--corpus_path", type=str, default=None) # path prefix to corpus file partitioned into train/dev/test. 
# defaults to SCRATCH/corpora/{langauge}_tok_{tokenizer}, completed by _{partition}.txt, see paths section below
parser.add_argument("--vocab_dir", type=str, default="./vocabulary") # vocab file should be {language}_{tokenizer}.txt, e.g. english_gpt2.txt
parser.add_argument("--vocabulary_size", type=int, default=50000) # sets an upper bound on BPE tokens used, keep consistent across languages

# hyperparameters
parser.add_argument("--batchSize", type=int, default=random.choice([512]))
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([512]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
parser.add_argument("--learning_rate", type = float, default= random.choice([1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([25]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))

# parameter of scientific interest
parser.add_argument("--deletion_rate", type=float, default=random.choice([0.2, 0.5]))

# local project info
parser.add_argument("--SCRATCH", default="/scratch") # default scratch directory. models and outputs stored here
parser.add_argument("--log_level", default="INFO")
# you can set the logger to your desired level
# e.g. logger.DEBUG to include debugging messages, or logger.WARNING for only warnings
parser.add_argument("--load-from", dest="load_from", type=str, default=None) # optional: use ID to load model from checkpoint

args=parser.parse_args()

##### set paths ############

def check_dir(path): 
    if not os.path.exists(path): 
        os.makedirs(path) 
        print(f"Directory created: {path}") 

check_dir(f"{args.SCRATCH}/CODEBOOKS/")
check_dir(f"{args.SCRATCH}/logs/")
check_dir(f"{args.SCRATCH}/loss_estimates/")

vocab_path = f"{args.vocab_dir}/{args.language}_{args.tokenizer}.txt"
corpus_path = f"{args.corpus_path if args.corpus_path is not None else args.SCRATCH+'/corpora/'+args.language+'_tok_'+args.tokenizer}"
model_load_path = f"{args.SCRATCH}/CODEBOOKS/{args.language}_{__file__}_code_{args.load_from}.txt"
model_save_path = f"{args.SCRATCH}/CODEBOOKS/{args.language}_{__file__}_code_{args.myID}.txt"
log_path = f"{args.SCRATCH}/logs/{args.language}_{__file__}_model_{args.myID}.txt"
estimated_dev_loss_path = f"{args.SCRATCH}/loss_estimates/{args.language}_{__file__}_model_{args.myID}.txt"

logger.setLevel(logging.getLevelName(args.log_level)) 
handler = logging.FileHandler(log_path)
logger.addHandler(handler)

logger.info(args)

#############################################################
# Vocabulary

with open(vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:args.vocabulary_size]]
stoi = dict([(itos[i],i) for i in range(len(itos))])

special = ["<MASK>", "<PAD>", "<SOS>", "<EOS>", "<EOSeq>", "OOV"]
MASK = 0
PAD = 1
SOS = 2
EOS = 3
EOSeq = 4
OOV = 5
itos_total = special + itos + ["_"+x for x in itos]
stoi_total = dict([(itos_total[i],i) for i in range(len(itos_total))])



############################################################
# Model

rnn = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
logger.debug(rnn_parameter_names)


rnn_drop = rnn 

output = torch.nn.Linear(args.hidden_dim, len(itos_total)).cuda()

word_embeddings = torch.nn.Embedding(num_embeddings=len(itos_total), embedding_dim=2*args.word_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=MASK)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=MASK)
char_dropout = torch.nn.Dropout1d(p=args.char_dropout_prob)


train_loss_chars = torch.nn.NLLLoss(ignore_index=MASK, reduction='sum')

modules = [rnn, output, word_embeddings]


def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]

learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) 

if args.load_from is not None:
  checkpoint = torch.load(model_load_path)
  for i in range(len(checkpoint["components"])):
      modules[i].load_state_dict(checkpoint["components"][i])

def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      logger.info("Prepare chunks")
      for chunk in data:
       assert args.sequence_length == 25
       wordCountForEachBatch = numpy.random.normal(loc=25, scale=10.0, size=(len(chunk)//args.sequence_length,)).clip(min=5)
       wordCountForEachBatch = (wordCountForEachBatch * len(chunk)/wordCountForEachBatch.sum()).round().astype('int')
       difference = len(chunk)- wordCountForEachBatch.sum()
       wordCountForEachBatch[:int(abs(difference))] = wordCountForEachBatch[:int(abs(difference))] + (1 if difference > 0 else -1)
       assert (len(chunk)- wordCountForEachBatch.sum()) == 0

       cumulativeLengths = wordCountForEachBatch.cumsum() - wordCountForEachBatch
       sequenceLengths = [sum([len(chunk[cumulativeLengths[batch] + j].split(" ")) for j in range(wordCountForEachBatch[batch])]) for batch in range(len(chunk)//args.sequence_length)]

       indices = list(range(len(chunk)//args.sequence_length))
       indices = sorted(indices, key=lambda x:sequenceLengths[x])
       batches = [[indices[args.batchSize*j + x] for x in range(args.batchSize) if args.batchSize*j + x < len(indices)] for j in range(1+len(indices)//args.batchSize)]
       random.shuffle(batches)
       for batch_list in batches:
         batchSizeHere = len(batch_list)
         assert len(batch_list) <= args.batchSize

         cutoff = batchSizeHere*args.sequence_length

         numerified = []
         asCharacters = []
         mask = []
         maxSeqLengthHere = 2+max([sum([len(chunk[cumulativeLengths[batch] + j].split(" ")) for j in range(wordCountForEachBatch[batch])]) for batch in batch_list])
         if maxSeqLengthHere > args.maxSeqLength:
             logger.warning("This should be rare. Truncating the long portions to prevent OOM errors")
             maxSeqLengthHere = args.maxSeqLength
         for batch in batch_list:
             numerified.append([])
             relevantLength = 2+sum([len(chunk[cumulativeLengths[batch] + j].split(" ")) if chunk[cumulativeLengths[batch] + j] not in special else 1 for j in range(wordCountForEachBatch[batch])])
             assert maxSeqLengthHere - relevantLength >= 0 or maxSeqLengthHere == args.maxSeqLength, relevantLength
             relevantSubstring = [itos_total[SOS]] + chunk[cumulativeLengths[batch]:cumulativeLengths[batch]+wordCountForEachBatch[batch]] + [itos_total[MASK]]
             for j in range(wordCountForEachBatch[batch]+2):
                 word = relevantSubstring[j].strip()
                 if word in special:
                     numerified[-1].append(stoi_total[word])
                     mask.append(j)
                     if len(numerified[-1]) >= args.maxSeqLength:
                         break
                 else:
                  word = word.split(" ")
                  for n, char in enumerate(word):
                     numerified[-1].append((stoi_total.get(("" if n>0 else "_")+char, OOV)))
                     mask.append(j)
                     if len(numerified[-1]) >= args.maxSeqLength:
                         break
                 if len(numerified[-1]) >= args.maxSeqLength:
                     break


             for _ in range(maxSeqLengthHere - relevantLength):
                 numerified[-1].append(MASK)
                 mask.append(-1)
                
             assert len(numerified[-1]) == maxSeqLengthHere, (len(numerified[-1]), maxSeqLengthHere) 

         assert len(numerified) == batchSizeHere
         numerified = torch.LongTensor(numerified).view(batchSizeHere, maxSeqLengthHere).t().cuda()
         mask = torch.LongTensor(mask).view(batchSizeHere, -1, maxSeqLengthHere).transpose(0,1).transpose(1,2).cuda()
         yield numerified, mask, asCharacters

def decodeItos(x):
    if x in [0,1,2,3]:
        return itos_total[x]
    x = x-len(special)
    return itos_total[x]

hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim)]).cuda())
noiseBernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.deletion_rate for _ in range(args.batchSize * args.sequence_length)]).cuda().view(args.sequence_length, args.batchSize))

def forward(numeric, indices_mask, asCharacters, train=True, printHere=False):
      global hidden
      global beginning
      global beginning_chars
      if True:
          hidden = None
          beginning = zeroBeginning

      batchSizeHere = numeric.size()[1]
      assert batchSizeHere <= args.batchSize

      #######################################
      # APPLY LOSS MODEL
      indices_mask = indices_mask.squeeze(0)
      numeric = numeric.squeeze(0)
      EOSeqIndicesPerString = indices_mask.max(dim=0).indices.cpu().numpy().tolist()
      assert min(EOSeqIndicesPerString) > 0, EOSeqIndicesPerString
      masked_noised = [[1 if (x == 0 or x == EOSeqIndicesPerString[y] or random.random() > args.deletion_rate) else 0 for x in range(indices_mask.size()[0])] for y in range(batchSizeHere)]
      numeric_noised = [[] for _ in range(batchSizeHere)]
      numeric_cpu = numeric.cpu().numpy().tolist()
      indices_mask_cpu = indices_mask.cpu().numpy().tolist()
      for i in range(batchSizeHere):
          j = 0
          k = 0
          while indices_mask_cpu[j][i] == -1:
              j += 1
          assert indices_mask_cpu[j][i] == k
          while j < len(indices_mask_cpu) and indices_mask_cpu[j][i] > -1:
              assert indices_mask_cpu[j][i] == k, (i,j,k,indices_mask_cpu[j][i], indices_mask[:,i], EOSeqIndicesPerString[i])
              currentToken = k
              assert len(masked_noised) <= batchSizeHere
              assert len(masked_noised[i]) == indices_mask.size()[0]
              if masked_noised[i][k] == 0: # masked
                  numeric_noised[i].append(MASK)
                  while j < len(indices_mask_cpu) and indices_mask_cpu[j][i] == currentToken:
                      j += 1
              else:
                  while j < len(indices_mask_cpu) and indices_mask_cpu[j][i] == currentToken:
                      numeric_noised[i].append(numeric_cpu[j][i])
                      j += 1
              k += 1

          numeric_noised[i] = numeric_noised[i] + [MASK for _ in range(len(indices_mask_cpu)-len(numeric_noised[i]))]
      numeric_noised = torch.LongTensor(numeric_noised).cuda().t()

      # Input to the decoding RNN
      input_tensor = Variable(numeric_noised[:-1], requires_grad=False)
      # Target for the decoding RNN
      target_tensor = Variable(numeric_noised[1:], requires_grad=False)
      # Input for the encoding RNN

      embedded = word_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)
         mask = bernoulli_input.sample()
         mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
         embedded = embedded * mask[:,:batchSizeHere]

      out, hidden = rnn_drop(embedded, hidden)

      if train:
        mask = bernoulli_output.sample()
        mask = mask.view(1, args.batchSize, args.hidden_dim)
        out = out * mask[:,:batchSizeHere]

      logits = output(out) 
      log_probs = logsoftmax(logits)

      # Calculate loss.
      loss = train_loss(log_probs.reshape(-1, len(itos_total)), target_tensor.reshape(-1))

      if printHere:
         lossTensor = print_loss(log_probs.reshape(-1, len(itos_total)), target_tensor.reshape(-1)).reshape(-1, batchSizeHere)
         losses = lossTensor.data.cpu().numpy().tolist()
         numericCPU = numeric.cpu().data.numpy().tolist()
         numeric_noisedCPU = numeric_noised.cpu().data.numpy().tolist()

         logger.debug(f"NONE {itos_total[numericCPU[0][0]]}")
         for i in range(len(losses)):
            logger.debug(f"{losses[i][0]} {itos_total[numeric_noisedCPU[i+1][0]]} {itos_total[numeric_noisedCPU[i+1][0]]}")
      return loss, batchSizeHere*args.sequence_length #target_tensor.reshape(-1).size()[0]

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         logger.debug(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_cached, 5.0) #, norm_type="inf")
      optim.step()

lossHasBeenBad = 0

totalStartTime = time.time()

lastSaved = (None, None)
devLosses = []
for epoch in range(10000):
   logger.info(f"Epoch {epoch}")
   training_data = corpusIterator.training(corpus_path)
   logger.info("Got training data")
   training_chars = prepareDatasetChunks(training_data, train=True)

   rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden, beginning = None, None
   runningAverage = 10
   while True:
      counter += 1
      try:
         numeric, indices_mask, asCharacters = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      loss, charCounts = forward(numeric, indices_mask, asCharacters, printHere=printHere, train=True)
      backward(loss, printHere)
      if loss.data.cpu().numpy() > 15.0:
          lossHasBeenBad += 1
      else:
          lossHasBeenBad = 0
      if lossHasBeenBad > 100:
          logger.warning(f"Loss exploding, has been bad for a while. Loss: {loss}")
          quit()
      runningAverage = 0.99 * runningAverage + (1-0.99) * float(loss)
      trainChars += charCounts 
      if printHere:
          logger.debug(f"Loss here {loss} average {runningAverage}")
          logger.debug(f"Epoch {epoch} Counter {counter}")
          logger.debug(f"Dev losses {devLosses}")
          logger.debug(f"Words per sec {trainChars/(time.time()-startTime)}")
          logger.debug(learning_rate)
          logger.debug(lastSaved)
          logger.debug(__file__)
          logger.debug(args)
      if counter % 20000 == 0: 
          state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules]}
          torch.save(state, model_save_path)
          lastSaved = (epoch, counter)
      if counter % 1000 == 0: 
          logger.info(str(args))
          logger.info(" ".join([str(x) for x in devLosses]))
          logger.info(counter)
          logger.info(f"Running average {runningAverage}")
          logger.info(f"Words per sec {trainChars/(time.time()-startTime)}")
          logger.info(os.environ.get('HOSTNAME')) # .get("ClusterId"), os.environ.get("ProcId")

      if epoch == 0 and (time.time() - totalStartTime)/60 > 1440:
          logger.warning("Breaking early to get some result within 24 hours")
          totalStartTime = time.time()
          break

   rnn_drop.train(False)

   dev_data = corpusIterator.dev(corpus_path)
   logger.info("Got dev data")
   dev_chars = prepareDatasetChunks(dev_data, train=False)
     
   dev_loss = 0
   dev_char_count = 0
   counter = 0
   hidden, beginning = None, None
   while True:
       counter += 1
       try:
         numeric, indices_mask, asCharacters = next(dev_chars)
       except StopIteration:
          break
       printHere = (counter % 50 == 0)
       loss, numberOfCharacters = forward(numeric, indices_mask, asCharacters, printHere=printHere, train=False)
       dev_loss += numberOfCharacters * loss.cpu().data.numpy()
       dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   logger.info(f"{devLosses}, DevCharacters {numberOfCharacters}")

   with open(estimated_dev_loss_path, "w") as outFile:
       print(str(args), file=outFile)
       print(" ".join([str(x) for x in devLosses]), file=outFile)

   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
      break

   state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules]}
   torch.save(state, model_save_path)
   lastSaved = (epoch, counter)
   
   learning_rate = args.learning_rate * math.pow(args.lr_decay, len(devLosses))
   optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9
