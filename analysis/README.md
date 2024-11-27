# Relative Clause analysis

This directory contains R code to reproduce the English relative clause analysis by  [McCurdy and Hahn (2024)](https://aclanthology.org/2024.conll-1.4). 

## Filler validation for Vani et al. (2021)

To reproduce the filler validation in section 4.1, run the following:
```
Rscript  filler_analysis.R
```

This will compute the LCS fit to filler RTs and generate the equivalent to the paper's Figure 3.

If you get an error message about needing the "curl" library, then run this shell command: `wget https://raw.githubusercontent.com/wilcoxeg/maze_src_orc/refs/heads/main/data/src-orc-results.csv`, uncomment line 41 in the script, comment out the URL in line 40, and run the script again

There are some slight differences to the analysis reported in the paper. To recover the exact plot:

- incorporate word frequency as a predictor from CELEX; this is excluded in the current script.
- add filler RT data from control experiment (Experiment 2) and calculate AIC over both; current script calculates only for Experiment 1.

## RC test predictions

As discussed in the paper, we generate test predictions at two retention rates: 20% (i.e. deletion rate 0.8, better fit to eye-tracking) and 60% (i.e. deletion rate 0.4, better fit to Maze).

Run `RC_test_analysis.R` to analyze LCS model predictions for test items and generate plots comparing LCS on critical regions in object and subject relative clauses.

Note that the generated plots use pre-computed averages for the RT data. This is for expedience. To reproduce these computations from the raw RT data, see `v21_test_means.R` for the Maze data from Vani et al., and [the original analysis code](https://osf.io/4rq3m/) for eye-tracking data from Roland et al.

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
