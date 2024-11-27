# Script to compute lossy context surprisal for a given dataset.

# Requirements:
#  - Pretrained amortized prediction model, specified in --load-from-lm
#  - Jointly pretrained amortized reconstruction model and memory model, specified in --load-from-joint
#     - parameter --deletion_rate should match deletion rate used to train joint model
#  - Pretrained Huggingface language model, specified in --plm
#     - if cached locally, cache specified in --plm_cache
#  - Stimulus file, specified in --stimulus_file
#     - expected format per line: ID;sentence;CRindex - CRindex will be converted to negative if not already
#     - specify a different field separator with --stimulus_sep
#     - skip stimulus file header (first line) by adding --skip-header


import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__file__ = __file__.split("/")[-1]

import glob
import time
import random
from collections import defaultdict
import json
import math
import numpy
import random
import torch
from torch.autograd import Variable
import scoreWithPLM as scoreSentences
import corpusIterator

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - ln %(lineno)d: %(message)s')
logger = logging.getLogger()

import argparse
parser = argparse.ArgumentParser()

# Language-specific parameters
parser.add_argument("--language", dest="language", type=str, default="English")
parser.add_argument("--tokenizer", type=str, default="gpt2") # identifier for subword tokenizer used
parser.add_argument("--vocab_dir", type=str, default="./vocabulary") # vocab file should be {language}_{tokenizer}.txt, e.g. english_gpt2.txt
parser.add_argument("--vocabulary_size", type=int, default=50000)  # sets an upper bound on BPE tokens used, keep consistent across languages
parser.add_argument("--maxSeqLength", type=int, default=100)

# Pretrained Language Model for this language
# should use same tokenizer as inference models and corpus
# should be trained only on language modeling objective
parser.add_argument("--plm", type=str, default="gpt2")

# Unique ID for this surprisal computation run
# the load_from_joint ID is used to identify the model, which can be used for different runs
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))

# Sequence length for surprisal computation
parser.add_argument("--sequence_length", type=int, default=random.choice([20]))

# Pretrained inference models
parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=random.choice([851053323])) # Amortized Prediction
parser.add_argument("--load-from-joint", dest="load_from_joint", type=str)  # Jointly trained Amortized Reconstruction Posterior and Memory Model

# Inference model hyperparameters
# Inference parameters 
parser.add_argument("--batchSize", type=int, default=random.choice([1])) # keep this value
parser.add_argument("--NUMBER_OF_REPLICATES", type=int, default=random.choice([12,20])) # can vary if needed
# Architecture parameters - should match values used in training
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim_lm", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim_autoencoder", type=int, default=random.choice([512]))
parser.add_argument("--layer_num_lm", type=int, default=random.choice([2]))
parser.add_argument("--layer_num_autoencoder", type=int, default=random.choice([2]))
## Learning parameters - shouldn't matter here since we're not training the model
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
parser.add_argument("--learning_rate_memory", type = float, default= random.choice([0.00002, 0.00005, 0.00005, 0.0001, 0.0001, 0.0001]))  # Can also use 0.0001, which leads to total convergence to deterministic solution withtin maximum iterations (March 25, 2021)   #, 0.0001, 0.0002 # 1e-7, 0.000001, 0.000002, 0.000005, 0.000007, 
parser.add_argument("--learning_rate_autoencoder", type = float, default= random.choice([0.001, 0.01, 0.1, 0.1, 0.1, 0.1])) # 0.0001, 
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--reward_multiplier_baseline", type=float, default=0.1)
parser.add_argument("--dual_learning_rate", type=float, default=random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--momentum", type=float, default=random.choice([0.5, 0.7, 0.7, 0.7, 0.7, 0.9])) # Momentum is helpful in facilitating convergence to a low-loss solution (March 25, 2021). It might be even more important for getting fast convergence than a high learning rate
parser.add_argument("--entropy_weight", type=float, default=random.choice([0.0])) # 0.0,  0.005, 0.01, 0.1, 0.4]))

# Control
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--tuning", type=int, default=1) #random.choice([0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0008, 0.001])) # 0.0,  0.005, 0.01, 0.1, 0.4]))

# Lambda and Delta Parameters
# Deletion rate should match the deletion rate used to train the lossy context memory model
parser.add_argument("--deletion_rate", type=float, default=0.5) 
parser.add_argument("--predictability_weight", type=float, default=random.choice([1.0])) # NB don't change this!

# Local project info
parser.add_argument("--SCRATCH", default="/proj/crosslinguistic-rrlcs.shadow") # default scratch directory on LST cluster. TODO: add for SIC cluster
parser.add_argument("--SCRATCH_write", type=str, default="") # optional, in case you want to write to a different scratch directory
# NB default save directory for computed surprisal values is in SCRATCH_write/output, invalid stats SCRATCH_write/invalid, logs SCRATCH_write/logs
parser.add_argument("--log_level", default="INFO")
# you can set the logger to your desired level
# e.g. logger.DEBUG to include debugging messages, or logger.WARNING for only warnings
# Optional: path to local pretrained language model cache
parser.add_argument("--plm_cache", type=str, default=None)

# Path to file with stimuli for surprisal computation
# Assume two fields per line: ID, sentence
parser.add_argument("--stimulus_file", type=str, default="stimuli.txt")
# Character separating the stimulus file fields, will split on this
parser.add_argument("--stimulus_sep", type=str, default=";")
# Skip header line
parser.add_argument("--stimulus_skip_header", action="store_true")
# Optionally write lossy context reconstructions to a separate file for inspection
parser.add_argument("--save_reconstructions", action="store_true")


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

def check_dir(path): 
    if not os.path.exists(path): 
        os.makedirs(path) 
        print(f"Directory created: {path}") 

check_dir(f"{args.SCRATCH}/invalid/")
check_dir(f"{args.SCRATCH}/output/")

vocab_path = f"{args.vocab_dir}/{args.language}_{args.tokenizer}.txt"
joint_load_path = f"{args.SCRATCH}/CODEBOOKS/{args.language}_lossy_context_model.py_delrate_{args.deletion_rate}_code_{args.load_from_joint}.txt"
char_lm_load_path = f"{args.SCRATCH}/CODEBOOKS/{args.language}_char_lm.py_code_{args.load_from_lm}.txt"
log_path = f"{args.SCRATCH_write}/logs/{args.language}_{__file__}_model_{args.load_from_joint}_run_{args.myID}.txt"
invalid_stats_path = f"{args.SCRATCH_write}/invalid/{args.language}_{__file__}_model_{args.load_from_joint}_run_{args.myID}_stats.tsv" 
output_path = f"{args.SCRATCH_write}/output/{args.language}_{__file__}_delrate_{args.deletion_rate}_model_{args.load_from_joint}_plm_{args.plm.replace('/','.')}_run_{args.myID}.txt" 

if args.save_reconstructions:
    check_dir(f"{args.SCRATCH}/reconstructions/")
    reconstructions_path = f"{args.SCRATCH_write}/reconstructions/{args.language}_{__file__}_model_{args.load_from_joint}_plm_{args.plm.replace('/','.')}_run_{args.myID}.txt" 


if args.plm_cache is not None:
    scoreSentences.setCache(args.plm_cache)
    
# set logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - ln %(lineno)d - %(message)s')
logger = logging.getLogger()

logger.setLevel(logging.getLevelName(args.log_level)) 
handler = logging.FileHandler(log_path)
logger.addHandler(handler)

logger.info(args.myID)
logger.info(args)
logger.info(sys.argv)

# monitor how often the inference models reconstruct sequences with invalid lengths
invalidPrevalence = None
averageInvalidPrevalence = 0
hasReportedInvalidPrevalence = False

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

#####################################################################
# Models

logger.debug(torch.__version__)

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

  def sampleReconstructions(self, numeric_noised, offset, numberOfBatches=None, fillInBefore=-1, computeProbabilityStartingFrom=0):
      """ Draws samples from the amortized reconstruction posterior """

      input_tensor_noised = Variable(numeric_noised, requires_grad=False)

      logger.debug(numeric_noised.size())
      logger.debug(f'{[itos_total[x] for x in numeric_noised[:,0]]}')

      embedded_noised = self.word_embeddings(input_tensor_noised)
      out_encoder, _ = self.rnn_encoder(embedded_noised, None)

      numberOfBatches = embedded_noised.size()[1]
      hidden = None
      result  = ["<SOS>" for _ in range(numberOfBatches)]
      result_numeric = [[2] for _ in range(numberOfBatches)]
      embeddedLast = embedded_noised[0].unsqueeze(0)
      assert (input_tensor_noised[0] == 2).all()
      amortizedPosterior = torch.zeros(numberOfBatches, device='cuda')
      zeroLogProb = torch.zeros(numberOfBatches, device='cuda')
      hasFinished = [False for _ in range(numberOfBatches)]
      hasFinished_gpu = torch.ByteTensor(hasFinished).cuda()

      for i in range(numeric_noised.size()[0]):
          out_decoder, hidden = self.rnn_decoder(embeddedLast, hidden)
          attention = torch.bmm(self.attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
          attention = self.attention_softmax(attention).transpose(0,1)
          from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
          out_full = torch.cat([out_decoder, from_encoder], dim=2)
          logits = self.output(self.relu(self.output_mlp(out_full) )) 
          probs = self.softmax(logits)

          dist = torch.distributions.Categorical(probs=probs)
       
          sampledFromDist = dist.sample()
          logProbForSampledFromDist = dist.log_prob(sampledFromDist).squeeze(0)
          amortizedPosterior += torch.where(hasFinished_gpu, zeroLogProb, logProbForSampledFromDist)

          nextWord = sampledFromDist

          hasFinished_gpu = torch.logical_or(hasFinished_gpu, (sampledFromDist==EOSeq).squeeze(0)) # this is very important for avoiding meaningless variance in the posterior
          nextWordDistCPU = nextWord.cpu().numpy()[0]
          nextWordStrings = [itos_total[x] for x in nextWordDistCPU]
          for i in range(numberOfBatches):
            if not hasFinished[i]:
              result[i] += " "+nextWordStrings[i]
              result_numeric[i].append( nextWordDistCPU[i] )
              if nextWordStrings[i] == "<EOSeq>":
                  hasFinished[i] = True
            else:
               result[i] += " "+"<PAD>"
               result_numeric[i].append(stoi_total["<PAD>"])
          embeddedLast = self.word_embeddings(nextWord)

      logger.debug(hasFinished_gpu)
      result_numeric = torch.LongTensor(result_numeric).cuda()
      if not all(hasFinished):
          logger.warning("some reconstruction isn't done yet")
      assert result_numeric.size()[0] == numberOfBatches
      return result, result_numeric, amortizedPosterior

    


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

memory_logits_mask = torch.zeros(args.sequence_length+2).cuda()
memory_logits_mask[0] = 1e10
memory_logits_mask[-2] = 1e10
memory_logits_mask[-1] = 1e10


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

  def forward(self, numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask, compute_likelihood=None):

      embedded_tokens = self.word_embeddings(numerified_arrangedByWords) #(length, batch, wordLength, 1024)
      embedded_tokens[numerified_arrangedByWords==0] = 0
      wordLengths = (numerified_arrangedByWords > 0).float().sum(dim=2).unsqueeze(2)
      aggregated = 0

      for i in range(min(5, embedded_tokens.size()[2])):
          aggregated += self.apply_to_embeddings_by_position[i](embedded_tokens[:,:,i,:]) + embedded_tokens[:,:,i,:] # (length, batch, 1024)

      embedded_everything_mem = torch.relu(aggregated)

      if indices_mask is not None:
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

      logger.debug(punctuationMask)
      memory_hidden = self.sigmoid(memory_hidden_logit + memory_logits_mask.view(-1, 1, 1) + 1e10 * punctuationMask.unsqueeze(2))

      if compute_likelihood is not None:
          memory_filter = compute_likelihood.unsqueeze(2)
      else:
          memory_filter = torch.bernoulli(input=memory_hidden).detach()

      bernoulli_logprob = torch.where(memory_filter == 1, torch.log(memory_hidden+1e-10), torch.log(1-memory_hidden+1e-10))
      if compute_likelihood is not None:
          return torch.where(punctuationMask.unsqueeze(2)==0, bernoulli_logprob, -100 + 0*bernoulli_logprob).squeeze(2)

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
      masked_noised = memory_filter.transpose(0,1).cpu().numpy().tolist() # [[1 if (x == 0 or x == args.sequence_length+1 or random.random() > args.deletion_rate) else 0 for x in range(args.sequence_length+2)] for y in range(batchSizeHere)]
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


  # now define the Memory model's comptue_likelihood function
  def compute_likelihood(self, reconstructed_numeric, numeric_noised_WithoutNextWord, numeric_arrangedByWords, memory_filter):
      logger.debug(f'{reconstructed_numeric.size()}, {numeric_noised_WithoutNextWord.size()}')
      logger.debug(f'{[itos_total[x] for x in reconstructed_numeric[0]]}')
      logger.debug(f'{[itos_total[x] for x in numeric_noised_WithoutNextWord[:,0]]}')

      counterfactual_asCharacters = None
      counterfactual_indices_mask = None
      counterfactual_numerified_arrangedByWords = None


      counterfactual_numerified_arrangedByWords = []
      reconstructed_numeric_cpu = reconstructed_numeric.cpu().numpy().tolist()
      numeric_noised_WithoutNextWord_cpu = numeric_noised_WithoutNextWord.cpu().numpy() #.tolist()
      for b in range(reconstructed_numeric.size()[0]):
          counterfactual_numerified_arrangedByWords.append([])
          counterfactual_numerified_arrangedByWords[-1].append([])
          for i in range(reconstructed_numeric.size()[1]):
              if reconstructed_numeric_cpu[b][i] == PAD:
                 if not all([x == PAD for x in reconstructed_numeric_cpu[b][i:]]):
                    logger.warning(f"the reconstruction inference network has output a non-PAD token after a PAD token: {[itos_total[z] for z in reconstructed_numeric_cpu[b]]}")
                 break
              if len(counterfactual_numerified_arrangedByWords[-1][-1]) > 0:
                if (itos_total[reconstructed_numeric_cpu[b][i]].startswith("_") or reconstructed_numeric_cpu[b][i] < len(special)):
                   counterfactual_numerified_arrangedByWords[-1].append([])
              
              counterfactual_numerified_arrangedByWords[-1][-1].append(reconstructed_numeric_cpu[b][i])
          
          logger.debug(counterfactual_numerified_arrangedByWords[-1])
          logger.debug(reconstructed_numeric_cpu[b])
          logger.debug(f'{[itos_total[x] for x in reconstructed_numeric_cpu[b]]}')
          logger.debug(f'{[itos_total[x] for x in numeric_noised_WithoutNextWord_cpu[:,b]]}')
          logger.debug(f"SENTENCE LENGTH: {len(counterfactual_numerified_arrangedByWords[-1])}")
      
      logger.debug(f"Sentence Lengths Without Last Word: {[len(q) for q in counterfactual_numerified_arrangedByWords]}")
      
      samplesWithInvalidLength = torch.ByteTensor([len(q) != args.sequence_length+1 for q in counterfactual_numerified_arrangedByWords])

      for b in range(reconstructed_numeric.size()[0]):
        if samplesWithInvalidLength[b]:
          counterfactual_numerified_arrangedByWords[b] = counterfactual_numerified_arrangedByWords[b][len(counterfactual_numerified_arrangedByWords[b])-args.sequence_length-1:]
          counterfactual_numerified_arrangedByWords[b] = [[SOS] for _ in range(args.sequence_length+1-len(counterfactual_numerified_arrangedByWords[b]))] + counterfactual_numerified_arrangedByWords[b]
          assert len(counterfactual_numerified_arrangedByWords[b]) == args.sequence_length+1, (len(counterfactual_numerified_arrangedByWords[b]), args.sequence_length+1)
      
      samplesWithInvalidLength = samplesWithInvalidLength.cuda()
      global invalidPrevalence
      invalidPrevalence = (samplesWithInvalidLength.float().cpu().mean().item())
      global averageInvalidPrevalence
      averageInvalidPrevalence = 0.9 * averageInvalidPrevalence + 0.1 * invalidPrevalence
      assert invalidPrevalence is not None
      
      for b in range(reconstructed_numeric.size()[0]):
          counterfactual_numerified_arrangedByWords[b] = counterfactual_numerified_arrangedByWords[b][:-1] + [[x for x in z if x != MASK] for z in numeric_arrangedByWords[-2:,0].cpu().numpy().tolist()]

      counterfactual_punctuationMask = torch.zeros(len(counterfactual_numerified_arrangedByWords), len(counterfactual_numerified_arrangedByWords[0])).byte()
     
      for b in range(reconstructed_numeric.size()[0]):
          for i in range(len(counterfactual_numerified_arrangedByWords[b])):
              if itos_total[counterfactual_numerified_arrangedByWords[b][i][0]] in punctuation_list and len(counterfactual_numerified_arrangedByWords[b][i]) == 1:
                 counterfactual_punctuationMask[b,i] = 1

      counterfactual_punctuationMask = counterfactual_punctuationMask.cuda().t()

      longestSentence = max([len(q) for q in counterfactual_numerified_arrangedByWords])
      logger.debug(f"Sentence Lengths: {[len(q) for q in counterfactual_numerified_arrangedByWords]}")
      longestWord = max([max([len(q) for q in r]) for r in counterfactual_numerified_arrangedByWords])
      for i in range(len(counterfactual_numerified_arrangedByWords)):
        counterfactual_numerified_arrangedByWords[i] = counterfactual_numerified_arrangedByWords[i] + [[] for _ in range(longestSentence-len(counterfactual_numerified_arrangedByWords[i]))]
        for j in range(len(counterfactual_numerified_arrangedByWords[i])):
           counterfactual_numerified_arrangedByWords[i][j] = counterfactual_numerified_arrangedByWords[i][j] + [0 for _ in range(longestWord-len(counterfactual_numerified_arrangedByWords[i][j]))]
      
      batchSizeHere = reconstructed_numeric.size()[0]
      counterfactual_numerified_arrangedByWords = torch.LongTensor(counterfactual_numerified_arrangedByWords).view(batchSizeHere, longestSentence, longestWord).transpose(0,1).cuda()
      logger.debug(counterfactual_numerified_arrangedByWords.size()) # torch.Size([21, 6, 3])
      logger.debug(numeric_noised_WithoutNextWord.size())
      assert numeric_arrangedByWords.size()[0] == args.sequence_length+2 # torch.Size([22, 1, 3])

      # Now copy over the last (i.e. next) word
      # Now we need to add the last word to the counterfactual_numerified_arrangedByWords
      logger.debug(f'{[[itos_total[numeric_arrangedByWords[w,0,i]] for i in range(numeric_arrangedByWords.size()[2])] for w in range(numeric_arrangedByWords.size()[0])]}')
      logger.debug(f'{[[itos_total[counterfactual_numerified_arrangedByWords[w,0,i]] for i in range(counterfactual_numerified_arrangedByWords.size()[2])] for w in range(counterfactual_numerified_arrangedByWords.size()[0])]}')

      assert counterfactual_numerified_arrangedByWords.size()[0] == numeric_arrangedByWords.size()[0], counterfactual_numerified_arrangedByWords.size()
      assert counterfactual_numerified_arrangedByWords.size()[0] == args.sequence_length+2, counterfactual_numerified_arrangedByWords.size()
      
      reconstructed_numeric = reconstructed_numeric.t()
      reconstructed_numeric = None
      return self.forward(reconstructed_numeric, counterfactual_indices_mask, counterfactual_asCharacters, counterfactual_numerified_arrangedByWords, counterfactual_punctuationMask, compute_likelihood=memory_filter) + torch.where(samplesWithInvalidLength, -100, 0)

    


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



###############################################3

checkpoint = torch.load(joint_load_path)
# Load pretrained prior and amortized posteriors

assert checkpoint["arguments"].sequence_length == args.sequence_length
assert set(list(checkpoint)) == set(["arguments", "words", "memory", "autoencoder"]), list(checkpoint)
assert itos == checkpoint["words"], (itos[:10], len(itos), checkpoint["words"][:10], len(checkpoint["words"]))

for i in range(len(checkpoint["memory"])):
   memory.modules_memory[i].load_state_dict(checkpoint["memory"][i])
for i in range(len(checkpoint["autoencoder"])):
   autoencoder.modules_autoencoder[i].load_state_dict(checkpoint["autoencoder"][i])


RNG = random.Random(2023)


def prepareDatasetFromChunk(chunk):
      numeric = [0]
      count = 0
      logger.info("Prepare chunks")
      if True:
       logger.debug(f'{len(chunk)}, {args.sequence_length}')
       wordCountForEachBatch = args.sequence_length + 0*numpy.random.normal(loc=25, scale=10.0, size=(len(chunk)//args.sequence_length,)).clip(min=5)
       wordCountForEachBatch = (wordCountForEachBatch * len(chunk)/wordCountForEachBatch.sum()).round().astype('int')
       logger.debug(chunk)
       logger.debug(f"7171556 word count for each batch {wordCountForEachBatch}, {len(chunk)}")
       assert sum(wordCountForEachBatch) == args.sequence_length
       difference = len(chunk)- wordCountForEachBatch.sum()
       wordCountForEachBatch[:int(abs(difference))] = wordCountForEachBatch[:int(abs(difference))] + (1 if difference > 0 else -1)
       assert (len(chunk)- wordCountForEachBatch.sum()) == 0, (len(chunk), wordCountForEachBatch.sum(), wordCountForEachBatch)

       cumulativeLengths = wordCountForEachBatch.cumsum() - wordCountForEachBatch
       sequenceLengths = [sum([len(chunk[cumulativeLengths[batch] + j].split(" ")) for j in range(wordCountForEachBatch[batch])]) for batch in range(len(chunk)//args.sequence_length)]

       indices = list(range(len(chunk)//args.sequence_length))
       indices = sorted(indices, key=lambda x:sequenceLengths[x])
       logger.debug(f"7171556 word count for each batch @@@@ {indices}, {args.batchSize}")
       batches = [[indices[args.batchSize*j + x] for x in range(args.batchSize) if args.batchSize*j + x < len(indices)] for j in range(1+len(indices)//args.batchSize)]
       RNG.shuffle(batches)
       logger.debug(indices)
       logger.debug(batches)
       for batch_identifier, batch_list in enumerate(batches):
         batchSizeHere = len(batch_list) 
         assert len(batch_list) <= args.batchSize
         cutoff = batchSizeHere*args.sequence_length

         numerified = []
         numerified_arrangedByWords = []
         asCharacters = []
         mask = []
         punctuationMask = []
         if len(batch_list) == 0: # this may happen, even in the batch size=1 situation, and should be harmless - such batches will just be skipped
            continue
         maxSeqLengthHere = 2+max([sum([len(chunk[cumulativeLengths[batch] + j].split(" ")) for j in range(wordCountForEachBatch[batch])]) for batch in batch_list])
         if maxSeqLengthHere > args.maxSeqLength:
             logger.warning("This should be rare. Truncating the long portions to prevent OOM errors")
             maxSeqLengthHere = args.maxSeqLength
         for batch in range(batchSizeHere):
             numerified.append([])
             numerified_arrangedByWords.append([])
             punctuationMask.append([])
             if False:
               asCharacters.append([])
             relevantLength = 2+sum([len(chunk[cumulativeLengths[batch] + j].split(" ")) if chunk[cumulativeLengths[batch] + j] not in special else 1 for j in range(wordCountForEachBatch[batch])])
             assert maxSeqLengthHere - relevantLength >= 0 or maxSeqLengthHere == args.maxSeqLength, relevantLength
             relevantSubstring = [itos_total[SOS]] + chunk[cumulativeLengths[batch]:cumulativeLengths[batch]+wordCountForEachBatch[batch]] + [itos_total[EOSeq]]
             for j in range(wordCountForEachBatch[batch]+2):
                 word = relevantSubstring[j].strip()
                 if word in special:
                     punctuationMask[-1].append(1)
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
                     if False:
                       asCharacters[-1].append(("_" if n == 0 else "")+char)
                     mask.append(j)
                     if len(numerified[-1]) >= args.maxSeqLength:
                         break
                 if len(numerified[-1]) >= args.maxSeqLength:
                     break


             for _ in range(maxSeqLengthHere - relevantLength):
                 numerified[-1].append(PAD)
                 if False:
                   asCharacters[-1].append(None)
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
             logger.warning("numerified_arrangedByWords doesn't have the expected length", numerified_arrangedByWords.size())
             logger.warning("Thus, skipping this batch")
             assert False

         assert numerified_arrangedByWords.size()[0] == args.sequence_length+2
         assert punctuationMask.size()[0] == args.sequence_length+2
         assert mask.size()[1] == numerified.size()[0]

         sequenceLengthHere = numerified_arrangedByWords.size()[0]
         return numerified, mask, asCharacters, numerified_arrangedByWords, punctuationMask, sequenceLengthHere
         chunk = chunk[cutoff:]
         assert len(chunk) == 0

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

########################## Forward pass to compute lossy context suprisal

def forward(numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False, NUMBER_OF_REPLICATES=args.NUMBER_OF_REPLICATES, expandReplicates=True):
      """ Forward pass through the entire model
        @param numeric
      """
      #print(numeric.size(), indices_mask.size())
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
      memory_hidden, memory_filter, bernoulli_logprob_perBatch, embedded_everything_mem, numeric_WithNextWord, numeric_noised_WithNextWord, numeric_WithoutNextWord, numeric_noised_WithoutNextWord, maskRelevantForNextWord = memory.forward(numeric, indices_mask, asCharacters, numerified_arrangedByWords, punctuationMask)

      return memory_hidden, memory_filter, bernoulli_logprob_perBatch, embedded_everything_mem, numeric_WithNextWord, numeric_noised_WithNextWord, numeric_WithoutNextWord, numeric_noised_WithoutNextWord, maskRelevantForNextWord


########################## Prepare and run on stimuli

lossHasBeenBad = 0
totalStartTime = time.time()
lastSaved = (None, None)
devLosses = []
updatesCount = 0
maxUpdates = 200000 if args.tuning == 1 else 10000000000
plm_tokenizer, plm_model = scoreSentences.loadPLM(args.plm)

# TODO should probably adjust this per language
dummyContext = "Here is a pretty long context, just in case we need more prior context when using a model with a very, very large context window. Indeed, the models might use a context size of up to 40 tokens. This is a meaningless dummy context without any relevant meaning just in case the thing is too short, well this is it."
 
def encodeContextCrop(inp, context):
     sentence = context.strip() + " " + inp.strip()
     logger.info(f"ENCODING {sentence}")
     numerified = [stoi_total[char] if char in stoi_total else 2 for char in sentence.split(" ")]
     logger.debug(len(numerified))
     numerified = numerified[-args.sequence_length-1:]
     numerified = torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
     return numerified

# for consistency, this function should exactly match vocabulary/tokenizeCorpus.py
def processCorpusLine(line, tknzr=plm_tokenizer):
    return [tknzr.decode(x).replace(" ","_") for x in tknzr.encode(line, return_tensors='pt').view(-1).cpu().numpy().tolist()]

def tokenize(line, tokenizer=plm_tokenizer, EOS=False):
    # the first step mirrors the preparation of the tokenized corpus
    tokenized = " ".join(processCorpusLine(line, tokenizer))
    # second step mirrors the way the corpus is read by corpusIterator.py
    #tokenized = [x.strip() for x in tokenized.strip().split(" _")] #+ ["<EOS>"]
    tokenized = corpusIterator.readCorpusLine(tokenized, EOS=EOS)
    logger.debug(tokenized)
#    quit()
    return tokenized

def getSurprisalsStimuli(SANITY="Model", numberOfSamples=6, dummyContext=tokenize(dummyContext)):

    assert SANITY in ["Model"]
    
    with torch.no_grad():
     with open(output_path, "w") as outFile:
      print("\t".join(["Sentence", "SentenceID", "Region", "Word", "Surprisal", "SurprisalReweighted", "Repetition"]), file=outFile)
      TRIALS_COUNT = 0
      
      for sentenceID in range(len(calibrationSentences)):
        
        logger.info(f"{sentenceID}: {calibrationSentenceIDs[sentenceID]}, {calibrationCRindices[sentenceID]}")
        sentence = calibrationSentences[sentenceID].split(" ")
        CRindex = int(calibrationCRindices[sentenceID])
        if CRindex > 0: 
            CRindex = -CRindex # CRindex counts from end of sentence, should be negative
        
        for region in range(len(sentence)+CRindex,len(sentence)):
         for repetition in range(2):       
          context = tokenize(" ".join(sentence[:region+1]))
          
          logger.info(f'{context}, {len(context)}')
          logger.debug(f'{dummyContext[-args.sequence_length-len(context):] + context}')
          logger.info("now running prepareDatasetFromChunk")
          
          forProcessing = (dummyContext + ["<EOS>"] + context)[-args.sequence_length:]
          numeric, indices_mask, asCharacters, numeric_arrangedByWords, punctuationMask, sequenceLengthHere = prepareDatasetFromChunk(forProcessing) # there will need to be a better way of constraining the context size
          
          logger.debug(f'{[itos_total[x] for x in numeric.view(-1).cpu().numpy().tolist()]}')
          logger.info(f"The preprocessed stimulus is {[itos_total[x] for x in numeric.view(-1).cpu().numpy().tolist()]}")
          logger.debug(sequenceLengthHere) # this is also the index of the critical word
          
          # now need to tokenize

          for repetition in range(2):

              memory_hidden, memory_filter, bernoulli_logprob_perBatch, embedded_everything_mem, numeric_WithNextWord, numeric_noised_WithNextWord, numeric_WithoutNextWord, numeric_noised_WithoutNextWord, maskRelevantForNextWord = forward(numeric, indices_mask, asCharacters, numeric_arrangedByWords, punctuationMask, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=numberOfSamples)

              logger.debug(memory_filter)
              logger.debug(numeric_noised_WithNextWord)
              logger.debug(f'{[itos_total[x] for x in numeric_noised_WithNextWord[:,0]]}')

              NUMBER_OF_REPLICATES=args.NUMBER_OF_REPLICATES
              logger.debug(numeric_noised_WithoutNextWord.size()) # [25,numberOfSamples]
              # composite batch size = (numberOfSamples * NUMBER_OF_REPLICATES)
              numeric_noised_WithoutNextWord_Broadcast = numeric_noised_WithoutNextWord.unsqueeze(2).expand(-1, -1, NUMBER_OF_REPLICATES).contiguous().view(numeric_noised_WithoutNextWord.size()[0], NUMBER_OF_REPLICATES*numberOfSamples).contiguous()
              logger.debug(memory_filter.size())
              memory_filter_Broadcast = memory_filter.unsqueeze(2).expand(-1, -1, NUMBER_OF_REPLICATES).contiguous().view(memory_filter.size()[0], NUMBER_OF_REPLICATES*numberOfSamples).contiguous()
              reconstructed, reconstructed_numeric, amortizedPosterior = autoencoder.sampleReconstructions(numeric_noised_WithoutNextWord_Broadcast, 0, numberOfBatches=None, fillInBefore=-1, computeProbabilityStartingFrom=0)
              
              # now for each reconstruction, get the LLM probability for reconstruction + next word
              # compute likelihood under the noise model
              reconstructedNatural = [x.replace("<SOS>", "").replace(" ", "").replace("_", " ").replace("<EOSeq>", "").replace("<PAD>", "").replace("<EOS>", "") for x in reconstructed]
              nextWord = ["" for _ in reconstructedNatural]
              numeric_noised_WithNextWord_Broadcast = numeric_noised_WithNextWord.unsqueeze(2).expand(-1, -1, NUMBER_OF_REPLICATES).contiguous().view(numeric_noised_WithNextWord.size()[0], NUMBER_OF_REPLICATES*numberOfSamples).contiguous()
              maskRelevantForNextWord_Broadcast = maskRelevantForNextWord.unsqueeze(2).expand(-1, -1, NUMBER_OF_REPLICATES).contiguous().view(maskRelevantForNextWord.size()[0], NUMBER_OF_REPLICATES*numberOfSamples).contiguous()
              logger.debug(f'{numeric_noised_WithNextWord.size()}, {maskRelevantForNextWord.size()}')
              numeric_noised_WithNextWord_Broadcast_cpu = numeric_noised_WithNextWord_Broadcast.cpu().numpy().tolist()
              maskRelevantForNextWord_Broadcast_cpu = maskRelevantForNextWord_Broadcast.cpu().numpy().tolist()
              for i1 in range(len(numeric_noised_WithNextWord_Broadcast_cpu)):
                  for j1 in range(len(nextWord)):
                      if maskRelevantForNextWord_Broadcast_cpu[i1][j1]:
                          nextWord[j1] = nextWord[j1] + itos_total[numeric_noised_WithNextWord_Broadcast_cpu[i1][j1]]


                          # Three things remain
                          # 1. call to LLM
                          # 2. interface with full dataset
                          # 3. for different model variants, including simple truncation

              reconstructed_stripped = [x.split("<EOS>")[-1].replace("<SOS>", "").replace("<EOSeq>", "").replace("<PAD>", "").replace("<EOS>", "").strip() for x in reconstructed]
              logger.debug(reconstructed_stripped)
              logger.debug(nextWord)

              if args.save_reconstructions:
                  context_stripped =  " ".join(['_' + x for x in context if x not in special])
                  with open(reconstructions_path, "a") as outfile:
                      for x in reconstructed_stripped:
                          outfile.write(f'{calibrationSentenceIDs[sentenceID]}\t{context_stripped}\t{x}\n')
                      logger.info(f"Wrote reconstructions to {reconstructions_path}")
              
              logger.debug(f'ORIGINAL {checkpoint["arguments"]}')
              logger.debug(f"HERE {args}")
              logger.debug(__file__)
              logger.debug(f"what fraction of reconstructions has the wrong number of words? {invalidPrevalence}")
              global hasReportedInvalidPrevalence
              if invalidPrevalence is not None:
                if not hasReportedInvalidPrevalence:
                  with open(invalid_stats_path, "a") as statsFile:
                    print(__file__, invalidPrevalence, checkpoint["arguments"], args, file=statsFile)
                  hasReportedInvalidPrevalence = True
                if sentenceID > 3 and averageInvalidPrevalence > 0.5:
                  assert False, "this model leads to many invalid reconstructions."
              scored = scoreSentences.scoreSentences([(x,y) for x,y in zip(reconstructed_stripped, nextWord)], plm_tokenizer, plm_model)
              assert numeric_arrangedByWords.size()[1] == 1 # batch size of 1
              log_likelihood = memory.compute_likelihood(reconstructed_numeric, numeric_noised_WithoutNextWord_Broadcast, numeric_arrangedByWords, memory_filter_Broadcast)
              log_likelihood = log_likelihood.sum(dim=0)
              assert invalidPrevalence is not None

              amortizedPosterior = amortizedPosterior.view(numberOfSamples, NUMBER_OF_REPLICATES)
              log_likelihood = log_likelihood.view(numberOfSamples, NUMBER_OF_REPLICATES)

              surprisals_past = torch.FloatTensor([x["past"] for x in scored]).cuda().view(numberOfSamples, NUMBER_OF_REPLICATES)
              surprisals_next = torch.FloatTensor([x["next"] for x in scored]).cuda().view(numberOfSamples, NUMBER_OF_REPLICATES)

              assert float(surprisals_past.min()) >= -1e-5, float(surprisals_past.min())
              assert float(surprisals_next.min()) >= -1e-5, float(surprisals_next.min())
              assert float(log_likelihood.max()) <= 1e-5, float(log_likelihood.max())
              assert float(amortizedPosterior.max()) <= 1e-5, float(amortizedPosterior.max())
              log_importance_weights = log_likelihood - surprisals_past - amortizedPosterior
              logger.debug(reconstructed_stripped)
              logger.debug(log_importance_weights)
              log_importance_weights_maxima, _ = log_importance_weights.max(dim=1, keepdim=True)

              log_importance_weighted_probs_unnormalized = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima - surprisals_next).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              log_importance_weights_sum = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              reweightedSurprisals = -(log_importance_weighted_probs_unnormalized - log_importance_weights_sum)
              reweightedSurprisalsMean = reweightedSurprisals.mean()

              surprisalOfNextWord = surprisals_next.exp().mean(dim=1).log().mean()
              nextWordSurprisal_cpu = surprisals_next.view(-1).detach().cpu()
              print("\t".join([str(w) for w in [sentenceID, calibrationSentenceIDs[sentenceID], region, sentence[region], round(float( surprisalOfNextWord),3), round(float( reweightedSurprisalsMean),3), repetition]]), file=outFile)
              logger.info(f"Wrote surprisal predictions to {output_path}.")

# Replace these sentences with the sentences of interest
calibrationSentenceIDs, calibrationSentences, calibrationCRindices = [], [], []

with open(args.stimulus_file, "r") as infile:
    for i, line in enumerate(infile):
        if args.stimulus_skip_header and not i:
            continue
        ID, sentence, CRindex = line.split(args.stimulus_sep)
        calibrationSentenceIDs.append(ID)
        calibrationSentences.append(sentence)
        calibrationCRindices.append(CRindex)

getSurprisalsStimuli()
