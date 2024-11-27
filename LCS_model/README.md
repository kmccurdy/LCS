# Resource-Rational Lossy Context Surprisal

This implementation extends [the original model](https://gitlab.com/m-hahn/resource-rational-surprisal) to work with subword-tokenized language models.
Note that this code has passed through many hands, and is not as organized as it could be - use at your own risk!

To reproduce the modeling results reported in McCurdy and Hahn, CoNLL 2024, train inference networks on an English-language corpus and use [GPT-2 Medium](https://huggingface.co/openai-community/gpt2-medium) as the pre-trained language model (PLM) --- or alternatively, simply run the analysis on the model predictions provided in `LCS/analysis/LCS_output`.

# Training the model

## First steps

- Identify a pretrained language model from the Huggingface hub. 
	- This will serve as our language model prior, and also define the tokenization scheme.
    - Ideally, you should select a model which has been trained **only on the language modeling objective**, with no instruction-tuning, RLHF or similar interventions; we want a prior model which only predicts a distribution over upcoming text.
- Identify a suitable corpus to train the inference models. 
- Run the following script to tokenize the corpus.
    - `vocabulary/tokenizeCorpus.py "--corpus_read_path path/to/corpus.file --corpus_write_path path/to/subtokenized/corpus.file --tokenizer huggingface/path` 
    - Output: subword-tokenized corpus, used to train inference models
- Identify the most frequent subwords in the corpus.
    - Script: `vocabulary/collectCharVoc.py ----corpus_path path/to/your/corpus.file --vocab_path vocabulary/your_vocab.file`
    - Output: text file in the `vocabulary` subdirectory
- Partition into train/dev/test depending on the size of the corpus.

## Training the inference networks

Train the amortized prediction model for next-word prediction conditional on a lossy context representation:

```
python inference_networks/char_lm.py --language your_language --corpus_path path/to/your/corpus --deletion_rate [0.2, 0.5]
```

Train the amortized reconstruction model to reconstruct context conditional on a lossy context representation:

```
python inference_networks/autoencoder.py --language your_language --corpus_path path/to/your/corpus --deletion_rate [0.2, 0.5]
```

## Training the lossy context model

Once you have the inference models, you use them to train the noisy memory model, which computes optimal retention probabilities for each word in a given context.
The memory model does **not** need to be trained using the same deletion rate as the inference networks. 

The training code randomly shuffles some hyperparameters to avoid overfitting. We recommend training at least 3 separate memory model instances at each deletion rate of interest.

```
python surprisal/lossy_context_model.py --language yourlanguage --corpus_path path/to/your/corpus --tokenizer [name of subword tokenizer used on corpus, e.g. gpt2; matched to vocabulary file] --deletion_rate [check] --load_from_autoencoder ID --load_from_lm ID
```

# Computing LCS

This requires the trained inference networks and memory model, as well as the pretrained language model identified at the beginning. If you've already saved the language model somewhere locally, you can avoid downloading it again by specifying the `--plm_cache` argument.

If using a language other than English, you'll likely want to replace the dummyContext variable in `compute_lcs.py` with something from your language.

The script assumes that your stimulus file has the format "ID;sentence;CRIndex" on each line, where CRIndex (Critical Region Index) is a negative value counting from the end of the script. It will compute LCS for each word from CRIndex to the end of the sentence, e.g. if CRIndex is -2, it computes LCS for the last two words of the sentence. 
N.B. this assumes that the critical region of interest occurs at the end of your stimuli. You can either alter your stimuli to cut off all words following the critical region, or pass in the length of the sentence as CRindex to get LCS calculations for every word.

```
python surprisal/compute_lcs.py --language your_language --stimulus_file path/to/stimuli --load-from-lm ID --load-from-joint ID --deletion_rate [same as joint model] --plm your/plm 
```


# Citation

If you use this code, please cite the following references:

```
@inproceedings{mccurdy_lossy_2024,
	address = {Miami, FL, USA},
	title = {Lossy {Context} {Surprisal} {Predicts} {Task}-{Dependent} {Patterns} in {Relative} {Clause} {Processing}},
	url = {https://aclanthology.org/2024.conll-1.4},
	booktitle = {Proceedings of the 28th {Conference} on {Computational} {Natural} {Language} {Learning}},
	publisher = {Association for Computational Linguistics},
	author = {McCurdy, Kate and Hahn, Michael},
	editor = {Barak, Libby and Alikhani, Malihe},
	month = nov,
	year = {2024},
	pages = {36--45}
}

@article{hahn_resource-rational_2022,
	title = {A resource-rational model of human processing of recursive linguistic structure},
	volume = {119},
	url = {https://www.pnas.org/doi/10.1073/pnas.2122602119},
	doi = {10.1073/pnas.2122602119},
	number = {43},
	urldate = {2023-04-23},
	journal = {Proceedings of the National Academy of Sciences},
	author = {Hahn, Michael and Futrell, Richard and Levy, Roger and Gibson, Edward},
	month = oct,
	year = {2022},
}
```

