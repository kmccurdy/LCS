# Functions to score sentences using a Huggingface pretrained language model (PLM)
# Called from compute_lcs.py

# Note: only tested with GPT2 models, so recommended use with that family.
# Should work with other models, but look out for potential issues.

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - ln %(lineno)d: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# point to directory with any locally cached PLMs
def setCache(cache_path=None):
    if cache_path:
        os.environ['TRANSFORMERS_CACHE'] = cache_path

# can either load from Huggingface Hub, or local cache
def loadPLM(plm="gpt2-medium"):
    tokenizer = AutoTokenizer.from_pretrained(plm)
    # add the EOS token as PAD token to avoid warnings
    model = AutoModelForCausalLM.from_pretrained(plm, pad_token_id=tokenizer.eos_token_id).cuda()
    logger.info(f"Finished loading PLM {plm}")
    return tokenizer, model

def scoreSentences(batch, tokenizer, model, word_init_char="Ġ"): # N.B. defaults to word separator from GPT2 tokenizer, you may need to change this
       contexts = [x[0] for x in batch]
       nextWords = [x[1] for x in batch]
       assert nextWords == [x.strip() for x in nextWords]
       nextWordCount = [len(x.split(" ")) for x in nextWords]
       batch = [x[0] + " " + x[1] for x in batch]
       logger.debug(nextWordCount)
       logger.debug(batch[0])
       logger.debug("Tokenized:") 
       logger.debug([[x for x in y.split(" ")] for y in batch][0])
       encoded_input = [[tokenizer.convert_tokens_to_ids(x.replace("_",word_init_char)) for x in y.split(" ")] for y in batch] 
       logger.debug(encoded_input[0]) # this sometimes includes an emty token, suggesting some duplicate whitespace "  " in the input
       tensors = [tokenizer.encode(text.replace(" ", "").replace("_", " "), return_tensors='pt') for text in batch] # below using bos, so should be no need for adding "<|endoftext|> "+
       logger.debug([tokenizer.convert_ids_to_tokens(x) for x in tensors[0].view(-1).numpy().tolist()])
       if encoded_input[0] != tensors[0].view(-1).numpy().tolist():
           logger.info("Note tokenization mismatch") 
       maxLength = max([x.size()[1] for x in tensors])
       for i in range(len(tensors)):
          tensors[i] = torch.cat([torch.LongTensor([tokenizer.bos_token_id]).view(1,1), tensors[i], torch.LongTensor([tokenizer.eos_token_id for _ in range(maxLength - tensors[i].size()[1])]).view(1, -1)], dim=1)
       tensors = torch.cat(tensors, dim=0)
       # Accounting for a change in transformers library since this was originally written
       if False: #int(VERSION[0]) < 3:
          predictions, _ = model(tensors.cuda())
       else:
        # Transformers v 3:
          predictions = model(tensors.cuda())
          predictions = predictions["logits"]     
       surprisals = torch.nn.CrossEntropyLoss(reduction='none')(predictions[:,:-1].contiguous().view(-1, 50257), tensors[:,1:].contiguous().view(-1).cuda()).view(len(batch), -1)
       surprisals = surprisals.detach().cpu()
       surprisalsCollected = []
       for batchElem in range(len(batch)):
         words = [[]]
         if batchElem == 0:
           logger.debug(tensors[batchElem])
         for q in range(1, maxLength+1):
            word = tokenizer.decode(int(tensors[batchElem][q]))
            if batchElem == 0:
               logger.debug(f'{q}, {int(tensors[batchElem][q])}, {word}, {maxLength}')
            if word == '<|endoftext|>':
                break
            if word.startswith(" ") or q == 0:
                words.append([])
            words[-1].append((word, float(surprisals[batchElem][q-1])))
         # find where last word starts and separately get the surprisals
         surprisalsPast = sum([sum(x[1] for x in y) for y in words[:-1]])
         surprisalsFirstFutureWord = sum(x[1] for x in words[-1])
         if batchElem == 0:
            logger.debug(words)
         surprisalsCollected.append({"past" : surprisalsPast, "next" : surprisalsFirstFutureWord})
       return surprisalsCollected

