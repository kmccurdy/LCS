# Script to train noisy memory model, i.e. the 'lossy context' of lossy context surprisal.

# Requirements:
#  - Pretrained amortized prediction model, specified in --load-from-lm
#  - Pretrained amortized reconstruction model, specified in --load-from-autoencoder
#  - Tokenized and partioned training corpus, specified in --corpus_path
#    - Pretrained tokenizer used on corpus, specified in --tokenizer
#    - Tokenized corpus vocabulary, parent directory specified in --vocab_dir

# This script trains a memory model, which uses the amortized inference models 
# to calculate retention probabilities of individual words in a given context.

# It also continues to train the reconstruction model in conjunction with the memory model,
# as this will be used as a proposal distribution during surprisal computation.

# To compute lossy context surprisal, a pretrained LM is additionally needed for the prior distribution, 
# see elsewhere in this directory.

import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__file__ = __file__.split("/")[-1]

import random
import time
from collections import defaultdict
import numpy
import math
import torch
from torch.autograd import Variable
import corpusIterator

import logging

import argparse

parser = argparse.ArgumentParser()

# Language specific parameters
parser.add_argument("--language", dest="language", type=str, default="English")
parser.add_argument("--tokenizer", type=str, default="gpt2") # identifier for subword tokenizer used
parser.add_argument("--vocab_dir", type=str, default="./vocabulary") # vocab file should be {language}_{tokenizer}.txt, e.g. english_gpt2.txt
parser.add_argument("--vocabulary_size", type=int, default=50000)  # sets an upper bound on BPE tokens used, keep consistent across languages
parser.add_argument("--maxSeqLength", type=int, default=100)
parser.add_argument("--corpus_path", type=str, default=None) # path prefix to corpus file partitioned into train/dev/test. 
# defaults to SCRATCH/corpora/{langauge}_tok_{tokenizer}, completed by _{partition}.txt, see paths section below

# Pretrained inference models
parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=random.choice([851053323])) # Amortized Prediction
parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=random.choice([147863665,529325683]))  # Amortized Reconstruction Posterior

# Unique ID for this model run
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))

# Sequence length
parser.add_argument("--sequence_length", type=int, default=random.choice([20]))

# Parameters of the neural network models
parser.add_argument("--batchSize", type=int, default=random.choice([1]))
parser.add_argument("--NUMBER_OF_REPLICATES", type=int, default=random.choice([12,20]))

## Layer size
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))

parser.add_argument("--hidden_dim_lm", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim_autoencoder", type=int, default=random.choice([512]))

## Layer number
parser.add_argument("--layer_num_lm", type=int, default=random.choice([2]))
parser.add_argument("--layer_num_autoencoder", type=int, default=random.choice([2]))

## Regularization
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))

## Learning Rates
parser.add_argument("--learning_rate_memory", type = float, default= random.choice([0.00001, 0.00001, 0.00001, 0.00001, 0.00002, 0.00002, 0.00002, 0.00002, 0.00002, 0.00002, 0.00002, 0.00002, 0.00005, 0.00005, 0.00005, 0.00005, 0.0001, 0.0001, 0.0001, 0.0002]))  # Can also use 0.0001, which leads to total convergence to deterministic solution withtin maximum iterations (March 25, 2021)   #, 0.0001, 0.0002 # 1e-7, 0.000001, 0.000002, 0.000005, 0.000007, 
parser.add_argument("--learning_rate_autoencoder", type = float, default= random.choice([0.001, 0.01])) # 0.0001, 
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--reward_multiplier_baseline", type=float, default=0.1)
parser.add_argument("--dual_learning_rate", type=float, default=random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--momentum", type=float, default=random.choice([0.0, 0.2, 0.5, 0.7, 0.9])) # Momentum is helpful in facilitating convergence to a low-loss solution (March 25, 2021). It might be even more important for getting fast convergence than a high learning rate
parser.add_argument("--entropy_weight", type=float, default=random.choice([0.0])) # 0.0,  0.005, 0.01, 0.1, 0.4]))

parser.add_argument("--maxUpdates", type=int, default=random.choice([4000000])) # how many steps to train

# Control
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--tuning", type=int, default=1) 

# Lambda and Delta Parameters
parser.add_argument("--deletion_rate", type=float, default=random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])) 
parser.add_argument("--predictability_weight", type=float, default=random.choice([1.0])) # NB don't change this!

# Local project info
parser.add_argument("--SCRATCH", default="/proj/crosslinguistic-rrlcs.shadow") # default scratch directory on LST cluster. TODO: add for SIC cluster
parser.add_argument("--SCRATCH_write", type=str, default="") # optional, in case you want to write to a different scratch directory
# NB default save directory for models is in SCRATCH_write/CODEBOOKS, for loss evaluation SCRATCH_write/loss_estimates, logs SCRATCH_write/logs
parser.add_argument("--log_level", default="INFO")
# you can set the logger to your desired level
# e.g. logger.DEBUG to include debugging messages, or logger.WARNING for only warnings

TRAIN_LM = False
assert not TRAIN_LM

args=parser.parse_args()

if not args.SCRATCH_write:
    args.SCRATCH_write = args.SCRATCH

############################

assert args.predictability_weight >= 0
assert args.predictability_weight <= 1
assert args.deletion_rate > 0.0
assert args.deletion_rate < 1.0
assert args.tuning in [0,1]
assert args.batchSize == 1

STDOUT = sys.stdout

##### set paths ############

vocab_path = f"{args.vocab_dir}/{args.language}_{args.tokenizer}.txt"
corpus_path = f"{args.corpus_path if args.corpus_path else args.SCRATCH+'/corpora/'+args.language+'_tok_'+args.tokenizer}"
autoencoder_load_path = f"{args.SCRATCH}/CODEBOOKS/{args.language}_autoencoder.py_code_{args.load_from_autoencoder}.txt"
char_lm_load_path = f"{args.SCRATCH}/CODEBOOKS/{args.language}_char_lm.py_code_{args.load_from_lm}.txt"
model_save_path = f"{args.SCRATCH_write}/CODEBOOKS/{args.language}_{__file__}_delrate_{args.deletion_rate}_code_{args.myID}.txt"
log_path = f"{args.SCRATCH_write}/logs/{args.language}_{__file__}_model_{args.myID}.txt"
estimated_loss_path = f"{args.SCRATCH_write}/loss_estimates/{args.language}_{__file__}_model_{args.myID}" # to be completed by _partition.txt

# set logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - ln %(lineno)d - %(message)s')
logger = logging.getLogger()

logger.setLevel(logging.getLevelName(args.log_level)) 
handler = logging.FileHandler(log_path)
logger.addHandler(handler)

logger.info(args.myID)
logger.info(args)
logger.info(sys.argv)

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

#############################################################
# Models

logger.debug(torch.__version__)

# Pretrained inference models

class Autoencoder:
  """ Amortized Reconstruction Posterior """
  def __init__(self):
    # This model describes a standard sequence-to-sequence LSTM model with attention
    self.rnn_encoder = torch.nn.LSTM(2*args.word_embedding_size, int(args.hidden_dim_autoencoder/2.0), args.layer_num_autoencoder, bidirectional=True).cuda() # encoder reads a noised input
    self.rnn_decoder = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim_autoencoder, args.layer_num_autoencoder).cuda() # outputs a denoised reconstruction
    self.output = torch.nn.Linear(args.hidden_dim_autoencoder, len(itos_total)).cuda()
    self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos_total), embedding_dim=2*args.word_embedding_size).cuda()
    self.logsoftmax = torch.nn.LogSoftmax(dim=2)
    self.softmax = torch.nn.Softmax(dim=2)
    self.attention_softmax = torch.nn.Softmax(dim=1)
    self.train_loss = torch.nn.NLLLoss(ignore_index=PAD)
    self.print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=PAD)
    self.char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)
    self.attention_proj = torch.nn.Linear(args.hidden_dim_autoencoder, args.hidden_dim_autoencoder, bias=False).cuda()
    self.attention_proj.weight.data.fill_(0)
    self.output_mlp = torch.nn.Linear(2*args.hidden_dim_autoencoder, args.hidden_dim_autoencoder).cuda()
    self.relu = torch.nn.ReLU()
    self.modules_autoencoder = [self.rnn_decoder, self.rnn_encoder, self.output, self.word_embeddings, self.attention_proj, self.output_mlp]


  def forward(self, numeric, numeric_noised):
      input_tensor_pure = numeric[:-1]
      target_tensor = numeric[1:]
      input_tensor_noised = numeric_noised
      autoencoder_embedded = self.word_embeddings(input_tensor_pure)
      autoencoder_embedded_noised = self.word_embeddings(input_tensor_noised)
      autoencoder_out_encoder, _ = self.rnn_encoder(autoencoder_embedded_noised, None)
      autoencoder_out_decoder, _ = self.rnn_decoder(autoencoder_embedded, None)

      autoencoder_attention = torch.bmm(self.attention_proj(autoencoder_out_encoder).transpose(0,1), autoencoder_out_decoder.transpose(0,1).transpose(1,2))
      autoencoder_attention = self.attention_softmax(autoencoder_attention).transpose(0,1)
      autoencoder_from_encoder = (autoencoder_out_encoder.unsqueeze(2) * autoencoder_attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
      autoencoder_out_full = torch.cat([autoencoder_out_decoder, autoencoder_from_encoder], dim=2)

      autoencoder_logits = self.output(self.relu(self.output_mlp(autoencoder_out_full) ))
      autoencoder_log_probs = self.logsoftmax(autoencoder_logits)

      # Prediction Loss 
      autoencoder_lossTensor = self.print_loss(autoencoder_log_probs.reshape(-1, len(itos_total)), target_tensor.reshape(-1))
      
      return autoencoder_lossTensor.view(numeric.size()[0]-1, numeric.size()[1])


class LanguageModel:
   """ Amortized Prediction Posterior """
   def __init__(self):
      self.rnn = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim_lm, args.layer_num_lm).cuda()
      self.rnn_drop = self.rnn
      self.output = torch.nn.Linear(args.hidden_dim_lm, len(itos_total)).cuda()
      self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos_total), embedding_dim=2*args.word_embedding_size).cuda()
      self.logsoftmax = torch.nn.LogSoftmax(dim=2)
      self.train_loss = torch.nn.NLLLoss(ignore_index=0)
      self.print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
      self.char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)
      self.train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
      self.modules_lm = [self.rnn, self.output, self.word_embeddings]
   def forward(self, numeric_noised_WithNextWord, NUMBER_OF_REPLICATES):
       lm_embedded = self.word_embeddings(numeric_noised_WithNextWord[:-1])
       lm_out, lm_hidden = self.rnn_drop(lm_embedded, None)
       lm_out = lm_out
       lm_logits = self.output(lm_out) 
       lm_log_probs = self.logsoftmax(lm_logits)

       # Prediction Loss 
       lm_lossTensor = self.print_loss(lm_log_probs.reshape(-1, len(itos_total)), numeric_noised_WithNextWord[1:].reshape(-1)).reshape(-1, NUMBER_OF_REPLICATES) # , args.batchSize is 1
       return lm_lossTensor 

# Memory model to train

memory_logits_mask = torch.zeros(args.sequence_length+2).cuda()
memory_logits_mask[0] = 1e10
memory_logits_mask[-2] = 1e10
memory_logits_mask[-1] = 1e10

baseline_memory_logits_mask = torch.zeros(args.sequence_length+2).cuda()
baseline_memory_logits_mask[0] = 1e10
baseline_memory_logits_mask[-2] = 1e10
baseline_memory_logits_mask[-1] = 1e10
numberOfRelevantEntries = round((1-args.deletion_rate) * (args.sequence_length-1))
baseline_memory_logits_mask[1:args.sequence_length-numberOfRelevantEntries] = -1e10
baseline_memory_logits_mask[args.sequence_length-numberOfRelevantEntries:-2] = 1e10
assert ((baseline_memory_logits_mask[1:-2]<0).float().mean().item() - (args.deletion_rate)) < 0.1, (baseline_memory_logits_mask, args.deletion_rate)

class MemoryModel():
  """ Noise Model """
  def __init__(self):
     self.memory_mlp_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.memory_mlp_inner_bilinear = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.memory_mlp_inner_from_pos = torch.nn.Linear(256, 500).cuda()
     self.memory_mlp_outer = torch.nn.Linear(500, 1).cuda()
     self.sigmoid = torch.nn.Sigmoid()
     self.relu = torch.nn.ReLU()
     self.positional_embeddings = torch.nn.Embedding(num_embeddings=args.sequence_length+2, embedding_dim=256).cuda()
     self.memory_word_pos_inter = torch.nn.Linear(256, 1, bias=False).cuda()
     self.memory_word_pos_inter.weight.data.fill_(0)
     self.perword_baseline_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.perword_baseline_outer = torch.nn.Linear(500, 1).cuda()
     self.memory_bilinear = torch.nn.Linear(256, 500, bias=False).cuda()
     self.memory_bilinear.weight.data.fill_(0)
     self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos_total), embedding_dim=2*args.word_embedding_size).cuda()

     self.apply_to_embeddings_by_position = [torch.nn.Linear(2*args.word_embedding_size, 2*args.word_embedding_size, bias=False).cuda() for _ in range(5)]

     self.modules_memory = [self.memory_mlp_inner, self.memory_mlp_outer, self.memory_mlp_inner_from_pos, self.positional_embeddings, self.perword_baseline_inner, self.perword_baseline_outer, self.memory_word_pos_inter, self.memory_bilinear, self.memory_mlp_inner_bilinear, self.word_embeddings] + self.apply_to_embeddings_by_position
  def forward(self, numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask, baseline=False):
      embedded_tokens = self.word_embeddings(numerified_arrangedByWords) #(length, batch, wordLength, 1024)
      embedded_tokens[numerified_arrangedByWords==0] = 0
      wordLengths = (numerified_arrangedByWords > 0).float().sum(dim=2).unsqueeze(2)
      aggregated = 0
      for i in range(min(5, embedded_tokens.size()[2])):
          aggregated += self.apply_to_embeddings_by_position[i](embedded_tokens[:,:,i,:]) + embedded_tokens[:,:,i,:] # (length, batch, 1024)
      embedded_everything_mem = torch.relu(aggregated)

      sequenceLengthHere = numeric.size()[0]
      batchSizeHere = numeric.size()[1]
      numeric_cpu = numeric.cpu().numpy().tolist()
      indices_mask_cpu = indices_mask.cpu().numpy().tolist()

      # Positional embeddings
      numeric_positions = torch.LongTensor(range(args.sequence_length+2)).cuda().unsqueeze(1)
      embedded_positions = self.positional_embeddings(numeric_positions)
      numeric_embedded = self.memory_word_pos_inter(embedded_positions)

      # Retention probabilities
      memory_byword_inner = self.memory_mlp_inner(embedded_everything_mem)
      memory_hidden_logit_per_wordtype = self.memory_mlp_outer(self.relu(memory_byword_inner))

      attention_bilinear_term = torch.bmm(self.memory_bilinear(embedded_positions), self.relu(self.memory_mlp_inner_bilinear(embedded_everything_mem)).transpose(1,2)).transpose(1,2)

      memory_hidden_logit = numeric_embedded + memory_hidden_logit_per_wordtype + attention_bilinear_term

      if not baseline:
         memory_hidden = self.sigmoid(memory_hidden_logit + memory_logits_mask.view(-1, 1, 1) + 1e10 * punctuationMask.unsqueeze(2))
      else:
         memory_hidden = self.sigmoid(memory_hidden_logit + baseline_memory_logits_mask.view(-1, 1, 1) + 1e10 * punctuationMask.unsqueeze(2))

      memory_filter = torch.bernoulli(input=memory_hidden).detach()


      bernoulli_logprob = torch.where(memory_filter == 1, torch.log(memory_hidden+1e-10), torch.log(1-memory_hidden+1e-10))
      bernoulli_logprob_perBatch = bernoulli_logprob.mean(dim=0)
      if args.entropy_weight > 0:
         entropy = -(memory_hidden * torch.log(memory_hidden+1e-10) + (1-memory_hidden) * torch.log(1-memory_hidden+1e-10)).mean()
      else:
         entropy=-1.0

      memory_filter = memory_filter.squeeze(2)

      batchSizeHere = numeric.size()[1]
      assert batchSizeHere <= args.batchSize * args.NUMBER_OF_REPLICATES

      #######################################
      # APPLY LOSS MODEL
      indices_mask = indices_mask.squeeze(0)
      numeric = numeric.squeeze(0)
      masked_noised = memory_filter.transpose(0,1).cpu().numpy().tolist()
      numeric_WithNextWord = [[] for _ in range(batchSizeHere)]
      numeric_WithoutNextWord = [[] for _ in range(batchSizeHere)]
      numeric_noised_WithNextWord = [[] for _ in range(batchSizeHere)]
      numeric_noised_WithoutNextWord = [[] for _ in range(batchSizeHere)]
      maskRelevantForNextWord = [[] for _ in range(batchSizeHere)]
      numeric_cpu = numeric.cpu().numpy().tolist()
      indices_mask_cpu = indices_mask.cpu().numpy().tolist()
      for i in range(batchSizeHere):
          j = 0
          k = 0
          while indices_mask_cpu[j][i] == -1:
              j += 1
          assert indices_mask_cpu[j][i] == k
          while j < len(indices_mask_cpu) and k <= args.sequence_length+1:
              assert indices_mask_cpu[j][i] == k, (i,j,k,indices_mask_cpu[j][i])
              currentToken = k
              assert len(masked_noised) <= batchSizeHere
              assert len(masked_noised[i]) == args.sequence_length+2
              #print(currentToken)
              if masked_noised[i][k] == 0: # masked
                  numeric_noised_WithNextWord[i].append(MASK)
                  maskRelevantForNextWord[i].append(0)
                  if currentToken != args.sequence_length:
                    numeric_noised_WithoutNextWord[i].append(MASK)
                  while j < len(indices_mask_cpu) and indices_mask_cpu[j][i] == currentToken:
                      numeric_WithNextWord[i].append(numeric_cpu[j][i])
                      if currentToken != args.sequence_length:
                         numeric_WithoutNextWord[i].append(numeric_cpu[j][i])
                      j += 1
              else:
                  while j < len(indices_mask_cpu) and indices_mask_cpu[j][i] == currentToken:
                      numeric_WithNextWord[i].append(numeric_cpu[j][i])
                      if currentToken != args.sequence_length:
                         numeric_WithoutNextWord[i].append(numeric_cpu[j][i])
                      numeric_noised_WithNextWord[i].append(numeric_cpu[j][i])
                      if currentToken == args.sequence_length:
                         maskRelevantForNextWord[i].append(1)
                      else:
                         maskRelevantForNextWord[i].append(0)

                      if currentToken != args.sequence_length:
                         numeric_noised_WithoutNextWord[i].append(numeric_cpu[j][i])
                      j += 1
              k += 1
          maskRelevantForNextWord[i] = maskRelevantForNextWord[i] + [0 for _ in range(len(indices_mask_cpu)-len(maskRelevantForNextWord[i]))]
          numeric_WithNextWord[i] = numeric_WithNextWord[i] + [PAD for _ in range(len(indices_mask_cpu)-len(numeric_WithNextWord[i]))]
          numeric_WithoutNextWord[i] = numeric_WithoutNextWord[i] + [PAD for _ in range(len(indices_mask_cpu)-len(numeric_WithoutNextWord[i]))]
          numeric_noised_WithNextWord[i] = numeric_noised_WithNextWord[i] + [PAD for _ in range(len(indices_mask_cpu)-len(numeric_noised_WithNextWord[i]))]
          numeric_noised_WithoutNextWord[i] = numeric_noised_WithoutNextWord[i] + [PAD for _ in range(len(indices_mask_cpu)-len(numeric_noised_WithoutNextWord[i]))]
      maskRelevantForNextWord = torch.ByteTensor(maskRelevantForNextWord).cuda().t()
      numeric_WithNextWord = torch.LongTensor(numeric_WithNextWord).cuda().t()
      numeric_WithoutNextWord = torch.LongTensor(numeric_WithoutNextWord).cuda().t()
      numeric_noised_WithNextWord = torch.LongTensor(numeric_noised_WithNextWord).cuda().t()
      numeric_noised_WithoutNextWord = torch.LongTensor(numeric_noised_WithoutNextWord).cuda().t()

      return memory_hidden, memory_filter, bernoulli_logprob_perBatch, embedded_everything_mem, numeric_WithNextWord, numeric_noised_WithNextWord, numeric_WithoutNextWord, numeric_noised_WithoutNextWord, maskRelevantForNextWord





# Build all three parts of the model
autoencoder = Autoencoder()
lm = LanguageModel()
memory = MemoryModel()

# Set up optimization

# Parameters for the retention probabilities
def parameters_memory():
   for module in memory.modules_memory:
       for param in module.parameters():
            yield param

parameters_memory_cached = [x for x in parameters_memory()]


# Dual parameter (for Lagrangian dual)
dual_weight = torch.cuda.FloatTensor([0.0])
dual_weight.requires_grad=True

# Parameters for inference networks
def parameters_autoencoder():
   for module in autoencoder.modules_autoencoder:
       for param in module.parameters():
            yield param

def parameters_lm():
   for module in lm.modules_lm:
       for param in module.parameters():
            yield param

parameters_lm_cached = [x for x in parameters_lm()]


assert not TRAIN_LM
optim_autoencoder = torch.optim.SGD(parameters_autoencoder(), lr=args.learning_rate_autoencoder, momentum=0.0) # 0.02, 0.9
optim_memory = torch.optim.SGD(parameters_memory(), lr=args.learning_rate_memory, momentum=args.momentum) # 0.02, 0.9

###############################################3


# Load pretrained prior and amortized posteriors

# Amortized Reconstruction Posterior
if True: 
  logger.info(args.load_from_autoencoder)
  checkpoint = torch.load(autoencoder_load_path)
  for i in range(len(checkpoint["components"])):
      autoencoder.modules_autoencoder[i].load_state_dict(checkpoint["components"][i])
  del checkpoint
 
# Amortized Prediction Posterior
if True: 
  checkpoint = torch.load(char_lm_load_path)
  for i in range(len(checkpoint["components"])):
      lm.modules_lm[i].load_state_dict(checkpoint["components"][i])
  del checkpoint


# Transferring word embeddings from LM to memory model
memory.word_embeddings.weight.data.copy_(lm.word_embeddings.weight.data)

##########################################################################
# Encode dataset chunks into tensors

RNG = random.Random(2023)


def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      logger.info("Prepare chunks")
      for chunk in data:
       wordCountForEachBatch = args.sequence_length + 0*numpy.random.normal(loc=25, scale=10.0, size=(len(chunk)//args.sequence_length,)).clip(min=5)
       wordCountForEachBatch = (wordCountForEachBatch * len(chunk)/wordCountForEachBatch.sum()).round().astype('int')
       difference = len(chunk)- wordCountForEachBatch.sum()
       wordCountForEachBatch[:int(abs(difference))] = wordCountForEachBatch[:int(abs(difference))] + (1 if difference > 0 else -1)
       assert (len(chunk)- wordCountForEachBatch.sum()) == 0

       cumulativeLengths = wordCountForEachBatch.cumsum() - wordCountForEachBatch
       sequenceLengths = [sum([len(chunk[cumulativeLengths[batch] + j].split(" ")) for j in range(wordCountForEachBatch[batch])]) for batch in range(len(chunk)//args.sequence_length)]

       indices = list(range(len(chunk)//args.sequence_length))
       indices = sorted(indices, key=lambda x:sequenceLengths[x])
       batches = [[indices[args.batchSize*j + x] for x in range(args.batchSize) if args.batchSize*j + x < len(indices)] for j in range(1+len(indices)//args.batchSize)]
       RNG.shuffle(batches)
       for batch_identifier, batch_list in enumerate(batches):
         batchSizeHere = len(batch_list) 
         assert len(batch_list) <= args.batchSize
         cutoff = batchSizeHere*args.sequence_length

         numerified = []
         numerified_arrangedByWords = []
         asCharacters = []
         mask = []
         punctuationMask = []
         if len(batch_list) == 0:
            continue
         maxSeqLengthHere = 2+max([sum([len(chunk[cumulativeLengths[batch] + j].split(" ")) for j in range(wordCountForEachBatch[batch])]) for batch in batch_list])
         if maxSeqLengthHere > args.maxSeqLength:
             logger.warning("This should be rare. Truncating the long portions to prevent OOM errors")
             maxSeqLengthHere = args.maxSeqLength
         for batch in batch_list:
             numerified.append([])
             numerified_arrangedByWords.append([])
             punctuationMask.append([])
             relevantLength = 2+sum([len(chunk[cumulativeLengths[batch] + j].split(" ")) if chunk[cumulativeLengths[batch] + j] not in special else 1 for j in range(wordCountForEachBatch[batch])])
             assert maxSeqLengthHere - relevantLength >= 0 or maxSeqLengthHere == args.maxSeqLength, relevantLength
             relevantSubstring = [itos_total[SOS]] + chunk[cumulativeLengths[batch]:cumulativeLengths[batch]+wordCountForEachBatch[batch]] + [itos_total[EOSeq]]
             for j in range(wordCountForEachBatch[batch]+2):
                 word = relevantSubstring[j].strip()
                 if word in special:
                     punctuationMask[-1].append(0)
                     numerified[-1].append(stoi_total[word])
                     numerified_arrangedByWords[-1].append([numerified[-1][-1]])
                     if False:
                        asCharacters[-1].append(word)
                     mask.append(j)
                     if len(numerified[-1]) >= args.maxSeqLength:
                         break

                 else:
                  if word in punctuation_list:
                      punctuationMask[-1].append(1)
                  else:
                      punctuationMask[-1].append(0)

                  numerified_arrangedByWords[-1].append([])
                  word = word.split(" ")
                  for n, char in enumerate(word):
                     numerified[-1].append((stoi_total.get(("" if n>0 else "_")+char, OOV)))
                     numerified_arrangedByWords[-1][-1].append(numerified[-1][-1])
                     mask.append(j)
                     if len(numerified[-1]) >= args.maxSeqLength:
                         break
                 if len(numerified[-1]) >= args.maxSeqLength:
                     break


             for _ in range(maxSeqLengthHere - relevantLength):
                 numerified[-1].append(PAD)
                 mask.append(-1)
                
             assert len(numerified[-1]) == maxSeqLengthHere, (len(numerified[-1]), maxSeqLengthHere) 

         assert len(numerified) == batchSizeHere

         longestSentence = max([len(q) for q in numerified_arrangedByWords])
         longestWord = max([max([len(q) for q in r]) for r in numerified_arrangedByWords])
         for i in range(len(numerified_arrangedByWords)):
           numerified_arrangedByWords[i] = numerified_arrangedByWords[i] + [[] for _ in range(longestSentence-len(numerified_arrangedByWords[i]))]
           for j in range(len(numerified_arrangedByWords[i])):
              numerified_arrangedByWords[i][j] = numerified_arrangedByWords[i][j] + [0 for _ in range(longestWord-len(numerified_arrangedByWords[i][j]))]
         numerified_arrangedByWords = torch.LongTensor(numerified_arrangedByWords).view(batchSizeHere, longestSentence, longestWord).transpose(0,1).cuda()
         numerified = torch.LongTensor(numerified).view(batchSizeHere, maxSeqLengthHere).t().cuda()
         mask = torch.LongTensor(mask).view(batchSizeHere, -1, maxSeqLengthHere).transpose(0,1).transpose(1,2).cuda()
         punctuationMask = torch.LongTensor(punctuationMask).view(batchSizeHere, -1).transpose(0,1).cuda()


         logger.debug(f'{numerified.size()}, {numerified_arrangedByWords.size()}, {mask.size()}, {punctuationMask.size()}, {batch_identifier}, {len(batches)}')

         if numerified_arrangedByWords.size()[0] != args.sequence_length+2:
             logger.warning(f"numerified_arrangedByWords doesn't have the expected length {numerified_arrangedByWords.size()}")
             logger.warning("Thus, skipping this batch")
             continue

         assert numerified_arrangedByWords.size()[0] == args.sequence_length+2
         assert punctuationMask.size()[0] == args.sequence_length+2
         assert mask.size()[1] == numerified.size()[0]

         yield numerified, mask, asCharacters, numerified_arrangedByWords, punctuationMask

runningAverageReward = 5.0
runningAverageBaselineDeviation = 2.0
runningAveragePredictionLoss = 10.0
runningAverageReconstructionLoss = 90.0
expectedRetentionRate = 0.5
forDividingTheAverages = 1

def getPunctuationMask(masks):
   assert len(masks) > 0
   if len(masks) == 1:
      return masks[0]
   else:
      punc1 = punctuation[:int(len(punctuation)/2)]
      punc2 = punctuation[int(len(punctuation)/2):]
      return torch.logical_or(getPunctuationMask(punc1), getPunctuationMask(punc2))

########################## Punctuation ##################

# Previously:
# The list of tokens that the model is constrained to never erase, in order to
#  preserve information about sentence boundaries
# Now:
#  Unused, for language-general subtokenized schemes - punctuation mask set to zero
punctuation_list = [".", "OOV", '"', "(", ")", "'", '"', ":", ",", "'s", "[", "]"] 
punctuation_list = [] 
PUNCTUATION = torch.zeros(len(punctuation_list)).cuda()

memory_hidden_runningAverage = .5 + torch.zeros(args.sequence_length+2)

def forward(numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False, NUMBER_OF_REPLICATES=args.NUMBER_OF_REPLICATES, expandReplicates=True, baseline=False):
      """ Forward pass through the entire model
        @param numeric
      """
      assert numeric.size()[1] <= args.batchSize
      sequenceLengthHere = numeric.size()[0]
      batchSizeHere = numeric.size()[1]
      ######################################################
      ######################################################
      # Step 1: replicate input to a batch
      if expandReplicates:
         assert batchSizeHere == 1
         numeric = numeric.expand(-1, NUMBER_OF_REPLICATES)
         indices_mask = indices_mask.expand(-1, -1, NUMBER_OF_REPLICATES)
         numerified_arrangedByWords = numerified_arrangedByWords.expand(-1, NUMBER_OF_REPLICATES, -1)
      # Input: numeric
      # Output: memory_hidden

      # Step 2: Compute retention probabilities
      memory_hidden, memory_filter, bernoulli_logprob_perBatch, embedded_everything_mem, numeric_WithNextWord, numeric_noised_WithNextWord, numeric_WithoutNextWord, numeric_noised_WithoutNextWord, maskRelevantForNextWord = memory.forward(numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask, baseline=baseline)

      global memory_hidden_runningAverage
      memory_hidden_runningAverage = 0.98 * memory_hidden_runningAverage + (1-0.98) * memory_hidden.detach().mean(dim=1).mean(dim=1).cpu()

      if provideAttention:
         return memory_hidden

      # Step 3: Compute control variate
      baselineValues = 20*memory.sigmoid(memory.perword_baseline_outer(memory.relu(memory.perword_baseline_inner(embedded_everything_mem[-1].detach())))).squeeze(1)
      assert tuple(baselineValues.size()) == (NUMBER_OF_REPLICATES,)

      if onlyProvideMemoryResult:
        return numeric, numeric_noised

      ##########################################
      ##########################################
      # Step 5: Run reconstruction inference network
      autoencoder_lossTensor = autoencoder.forward(numeric_WithoutNextWord, numeric_noised_WithoutNextWord)
      autoencoder_lossTensor_summed = autoencoder_lossTensor.sum(dim=0)
      assert autoencoder_lossTensor_summed.size() == torch.Size((numeric.size()[1],))
      ##########################################
      ##########################################
      # Step 6: Run prediction inference network
      if args.predictability_weight > 0:
       lm_lossTensor = lm.forward(numeric_noised_WithNextWord, NUMBER_OF_REPLICATES)
       lm_lossTensor_summed = (lm_lossTensor * maskRelevantForNextWord[1:].float()).sum(dim=0)
      ##########################################
      ##########################################

      # Step 7: Collect loss function for training signal
      # Reward, term 1
      if args.predictability_weight > 0:
        negativeRewardsTerm1 = (args.predictability_weight * lm_lossTensor_summed + (1-args.predictability_weight) * autoencoder_lossTensor_summed) / args.sequence_length
      else:
        negativeRewardsTerm1 = autoencoder_lossTensor_summed / args.sequence_length


      # Reward, term 2
      # Regularization towards lower retention rates
      negativeRewardsTerm2 = memory_filter[1:-2].mean(dim=0)
      retentionTarget = 1-args.deletion_rate
      loss = 0

      # Reconstruction Loss (referred to as L2 in SI Appendix, Section 1)
      loss += autoencoder_lossTensor.mean()

      # Overall Reward (referred to as L1 in SI Appendix, Section 1)
      negativeRewardsTerm = negativeRewardsTerm1 + dual_weight * (negativeRewardsTerm2-retentionTarget)
      # for the dual weight
      loss += (dual_weight * (negativeRewardsTerm2-retentionTarget).detach()).mean()
      if printHere:
          logger.info(f"REWARDS - Pred {lm_lossTensor_summed.mean()} Rec {autoencoder_lossTensor_summed.mean()}, {negativeRewardsTerm1.mean()}, {dual_weight}, {negativeRewardsTerm2.mean()}, {retentionTarget}")

      # baselineValues: the baselines for the prediction loss (term 1)
      # memory_hidden: baseline for term 2
      # Important to detach all but the baseline values

      # Reward Minus Baseline
      # Detached surprisal and mean retention
      # Subtract control variate for unbiased variance reduction
      rewardMinusBaseline = (negativeRewardsTerm.detach() - baselineValues - (dual_weight * (memory_hidden[1:-2].mean(dim=0).squeeze(dim=1) - retentionTarget)).detach())

      # Apply REINFORCE estimator
      # Important to detach from the baseline!!! 
      loss += (rewardMinusBaseline.detach() * bernoulli_logprob_perBatch.squeeze(1)).mean()
      if args.entropy_weight > 0:
         assert False
         loss -= args.entropy_weight  * entropy

      # Training signal for control variate
      loss += args.reward_multiplier_baseline * rewardMinusBaseline.pow(2).mean()


      ############################
      # Construct running averages
      factor = 0.996 ** args.batchSize

      # Update running averages
      global runningAverageBaselineDeviation
      global runningAveragePredictionLoss
      global runningAverageReconstructionLoss
      global runningAverageReward
      global expectedRetentionRate

      expectedRetentionRate = factor * expectedRetentionRate + (1-factor) * float(memory_hidden[1:-2].mean())
      runningAverageBaselineDeviation += float((rewardMinusBaseline).abs().mean())

      if args.predictability_weight > 0:
       runningAveragePredictionLoss += round(float(lm_lossTensor_summed.mean()),3)
      runningAverageReconstructionLoss += round(float(autoencoder_lossTensor_summed.mean()),3)
      runningAverageReward += float(negativeRewardsTerm.mean())
      global forDividingTheAverages
      forDividingTheAverages += 1
      ############################

      if printHere:
         if args.predictability_weight > 0:
          lm_losses = lm_lossTensor.data.cpu().numpy().tolist()
         autoencoder_losses = autoencoder_lossTensor.data.cpu().numpy().tolist()

         entropy = float('nan') 
         logger.info(f"PREDICTION_LOSS {runningAveragePredictionLoss/forDividingTheAverages} RECONSTRUCTION_LOSS {runningAverageReconstructionLoss/forDividingTheAverages}\tTERM2 {round(float(negativeRewardsTerm2.mean()),3)}\tAVERAGE_RETENTION {expectedRetentionRate}\tDEVIATION FROM BASELINE {runningAverageBaselineDeviation/forDividingTheAverages}\tREWARD {runningAverageReward/forDividingTheAverages}\tENTROPY {float(entropy)}")
       
         numeric_WithNextWord_cpu = numeric_WithNextWord.cpu().numpy().tolist()
         numeric_WithoutNextWord_cpu = numeric_WithoutNextWord.cpu().numpy().tolist()
         numeric_noised_WithNextWord_cpu = numeric_noised_WithNextWord.cpu().numpy().tolist()
         numeric_noised_WithoutNextWord_cpu = numeric_noised_WithoutNextWord.cpu().numpy().tolist()
         indices_mask_cpu = indices_mask.cpu().numpy().tolist()
         autoencoder_lossTensor_cpu = autoencoder_lossTensor.detach().cpu().numpy().tolist()
         memory_hidden_cpu = memory_hidden.detach().cpu().numpy().tolist()
         punctuationMask_cpu = punctuationMask.cpu().numpy().tolist()
         maskRelevantForNextWord_cpu = maskRelevantForNextWord.cpu().numpy().tolist()
         memory_filter_cpu = memory_filter.cpu().numpy().tolist()
         for i in range(numeric_WithNextWord.size()[0]):
            logger.debug(f'{i}, {itos_total[numeric_WithNextWord_cpu[i][0]]}, {indices_mask_cpu[0][i][0]}, {memory_hidden_cpu[indices_mask_cpu[0][i][0]][0][0]}, {memory_filter_cpu[indices_mask_cpu[0][i][0]][0]}, {punctuationMask_cpu[indices_mask_cpu[0][i][0]][0]}, {itos_total[numeric_WithoutNextWord_cpu[i][0]]}, {autoencoder_lossTensor_cpu[i-1][0] if i > 0 else "--"}')
         for i in range(numeric_noised_WithNextWord.size()[0]):
            logger.debug(f'{i}, {itos_total[numeric_noised_WithNextWord_cpu[i][0]]}, {itos_total[numeric_noised_WithoutNextWord_cpu[i][0]]}, {lm_losses[i-1][0] if i>0 else "-"},{ maskRelevantForNextWord_cpu[i][0]}')
         logger.info(f"REWARD LM {lm_lossTensor_summed} REWARD Rec {autoencoder_lossTensor_summed}")

      if printHere:
        logger.debug(memory_hidden_runningAverage)
        logger.info(f"PRED_LOSS {round(runningAveragePredictionLoss/forDividingTheAverages,3)} REC_LOSS  {round(runningAverageReconstructionLoss/forDividingTheAverages,3)}\tTERM2 {round(float(negativeRewardsTerm2.mean()),3)} \tAVG_RETENTION {round(expectedRetentionRate,3)}\tDEV FROM BASELINE {round(runningAverageBaselineDeviation/forDividingTheAverages,3)}\tREWARD {runningAverageReward/forDividingTheAverages} {float(dual_weight)}")

      if updatesCount % 5000 == 0:
         logger.info("updatesCount", updatesCount, updatesCount/args.maxUpdates)
         print("\t".join([str(x) for x in ("PREDICTION_LOSS", runningAveragePredictionLoss/forDividingTheAverages, "RECONSTRUCTION_LOSS", runningAverageReconstructionLoss/forDividingTheAverages, "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", expectedRetentionRate, "\tDEVIATION FROM BASELINE", runningAverageBaselineDeviation/forDividingTheAverages, "\tREWARD", runningAverageReward/forDividingTheAverages #, "\tENTROPY", float(entropy)
                                          )]), file=sys.stderr)

      return loss, numeric.size()[1] * args.sequence_length


def backward(loss, printHere):
      """ An optimization step for the resource-rational objective function """
      # Set stored gradients to zero
      optim_autoencoder.zero_grad()
      optim_memory.zero_grad()

      if dual_weight.grad is not None:
         dual_weight.grad.data.fill_(0.0)
      if printHere:
         logger.debug(loss)
      # Calculate new gradients
      loss.backward()
      # Gradient clipping
      torch.nn.utils.clip_grad_value_(parameters_memory_cached, 5.0) #, norm_type="inf")
      if TRAIN_LM:
         assert False
         torch.nn.utils.clip_grad_value_(parameters_lm_cached, 5.0) #, norm_type="inf")

      # Adapt parameters
      optim_autoencoder.step()
      optim_memory.step()

      dual_weight.data.add_(args.dual_learning_rate*dual_weight.grad.data)
      dual_weight.data.clamp_(min=0)

lossHasBeenBad = 0

totalStartTime = time.time()

lastSaved = (None, None)
devLosses = []
updatesCount = 0

startTimePredictions = time.time()
startTimeTotal = time.time()

trajectory = []

runningAverage = 10

for epoch in range(1000):
   logger.info(f'\nEpoch {epoch}\n')

   # Get training data
   training_data = corpusIterator.training(corpus_path)
   logger.info("Got training data")
   training_chars = prepareDatasetChunks(training_data, train=True)


   # Set the model up for training
   lm.rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   # End optimization when args.maxUpdates is reached
   if updatesCount > args.maxUpdates:
     break
   while updatesCount <= args.maxUpdates:
      counter += 1
      updatesCount += 1

     # Get a batch from the training set
      try:
         numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      # Run this through the model: forward pass of the resource-rational objective function
      loss, charCounts = forward(numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask, printHere=printHere, train=True)
      # Calculate gradients and update parameters
      backward(loss, printHere)


      # Bad learning rate parameters might make the loss explode. In this case, stop.
      if lossHasBeenBad > 100:
          logger.warning("Loss exploding, has been bad for a while")
          logger.warning(loss)
          assert False
      runningAverage = 0.99 * runningAverage + (1-0.99) * float(loss)
      trainChars += charCounts 
      if printHere:
          logger.info(f"Loss here {loss} average {runningAverage}")
          logger.info(f"{epoch} Updates {updatesCount} {str((100.0*updatesCount)/args.maxUpdates)+' %'} {args.maxUpdates} {counter} {trainChars} \nETA {((time.time()-startTimeTotal)/updatesCount * (args.maxUpdates-updatesCount))/3600.0} hours") 
          logger.info(f"Dev losses {devLosses}")
          logger.info(f"Words per sec {str(trainChars/(time.time()-startTime))}")
          logger.debug(f'{args.learning_rate_memory} {args.learning_rate_autoencoder}')
          logger.debug(lastSaved)
          logger.debug(__file__)
          logger.debug(args)
      if updatesCount % 200000 == 0:
            trajectory.append(float(runningAverageReward/forDividingTheAverages))

with open(f"{estimated_loss_path}_train.txt", "w") as outFile:
   print(args, file=outFile)
   print(runningAverageReward/forDividingTheAverages, file=outFile)
   print(expectedRetentionRate, file=outFile)
   print(runningAverageBaselineDeviation/forDividingTheAverages, file=outFile)
   print(runningAveragePredictionLoss/forDividingTheAverages, file=outFile)
   print(runningAverageReconstructionLoss/forDividingTheAverages, file=outFile)
   print(" ".join([str(w) for w in trajectory]), file=outFile)

state = {"arguments" : args, "words" : itos, "memory" : [c.state_dict() for c in memory.modules_memory], "autoencoder" : [c.state_dict() for c in autoencoder.modules_autoencoder]}
torch.save(state, model_save_path)


runningAverageReward = 5.0
runningAverageBaselineDeviation = 2.0
runningAveragePredictionLoss = 10.0
runningAverageReconstructionLoss = 90.0
expectedRetentionRate = 0.5
forDividingTheAverages = 1


updatesCount = 0
for epoch in range(1):
   print(epoch)

   # Get dev data
   dev_data = corpusIterator.dev(corpus_path)
   logger.info("Got dev data for heldout")
   dev_chars = prepareDatasetChunks(dev_data, train=True)


   # Set the model up for dev
   lm.rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   # End optimization when args.maxUpdates is reached
   if updatesCount > 500:
     break
   while updatesCount <= 500:
      counter += 1
      updatesCount += 1

     # Get a batch from the dev set
      try:
         numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask = next(dev_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      # Run this through the model: forward pass of the resource-rational objective function
      loss_Model, charCounts = forward(numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask, printHere=printHere, train=True)

with open(f"{estimated_loss_path}_dev_heldout.txt", "w") as outFile:
   print(args, file=outFile)
   print(runningAverageReward/forDividingTheAverages, file=outFile)
   print(expectedRetentionRate, file=outFile)
   print(runningAverageBaselineDeviation/forDividingTheAverages, file=outFile)
   print(runningAveragePredictionLoss/forDividingTheAverages, file=outFile)
   print(runningAverageReconstructionLoss/forDividingTheAverages, file=outFile)
   print(" ".join([str(w) for w in trajectory]), file=outFile)

runningAverageReward = 5.0
runningAverageBaselineDeviation = 2.0
runningAveragePredictionLoss = 10.0
runningAverageReconstructionLoss = 90.0
expectedRetentionRate = 0.5
forDividingTheAverages = 1


updatesCount = 0
for epoch in range(1):
   print(epoch)

   # Get dev data
   dev_data = corpusIterator.dev(corpus_path)
   logger.info("Got dev data for baseline")
   dev_chars = prepareDatasetChunks(dev_data, train=True)


   # Set the model up for dev
   lm.rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   # End optimization when args.maxUpdates is reached
   if updatesCount > 500:
     break
   while updatesCount <= 500:
      counter += 1
      updatesCount += 1

     # Get a batch from the dev set
      try:
         numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask = next(dev_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      # Run this through the model: forward pass of the resource-rational objective function
      loss_Model, charCounts = forward(numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask, printHere=printHere, train=True, baseline=True)

with open(f"{estimated_loss_path}_dev_baseline.txt", "w") as outFile:
   print(args, file=outFile)
   print(runningAverageReward/forDividingTheAverages, file=outFile)
   print(expectedRetentionRate, file=outFile)
   print(runningAverageBaselineDeviation/forDividingTheAverages, file=outFile)
   print(runningAveragePredictionLoss/forDividingTheAverages, file=outFile)
   print(runningAverageReconstructionLoss/forDividingTheAverages, file=outFile)
   print(" ".join([str(w) for w in trajectory]), file=outFile)

