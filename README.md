# Language model chunker

This repository contains the code for utlizing the language models for chunking. The experiements were done on conll-2000 dataset. 

## Pre-requisite Python Libraries

Please install the following libraries specified in the **requirements.txt** first before running our code.

    transformers==2.2.0
    numpy==1.15.4
    tqdm==4.26.0
    torch==1.3.1
    nltk==3.4
    matplotlib==2.2.3
    
## Data preparation (CONLL)

Please refer to the code inside data_conll for all the processed files and the original conll ata files. 
Details will be updated soon. 

## Files

There are 3 important file beyond the preprocessing of conll:

- **run.py** to extract the features from different layers and different attemtion heads
- **evaluate_distances.py** to evaluate the distances obtained from run.py and create a file ready for evaluation script.  
- **eval_conll2000_updated.pl** updated eval script which only prints overall phrase level F1 and tag level accuracy

## How to Run Code

> python run.py --help

	usage: run.py [--data-path DATA_PATH] [--result-path RESULT_PATH]

**Important** Inside run.py the input pickle file for the conll tokens must be given. I.e. Update the following line:

data_tokens = pickle.load(open("data_conll/conll/data_train_tokens.pkl", "rb"))


    following arguments are not needed:
      -h, --help    show this help message and exit
      --data-path DATA_PATH
      --result-path RESULT_PATH
      --from-scratch
      --gpu GPU
      --bias BIAS   the right-branching bias hyperparameter lambda
      --seed SEED
      --token-heuristic TOKEN_HEURISTIC     Available options: mean, first, last
      --use-coo-not-parser  Turning on this option will allow you to exploit the
                            COO-NOT parser (named by Dyer et al. 2019), which has
                            been broadly adopted by recent methods for
                            unsupervised parsing. As this parser utilizes the
                            right-branching bias in its inner workings, it may
                            give rise to some unexpected gains or latent issues
                            for the resulting trees. For more details, see
                            https://arxiv.org/abs/1909.09428.


![alt text](https://github.com/[Anup-Deshmukh]/[LM-Unsupervised-Chunker]/blob/[master]/res1.png?raw=true)

## Acknowledgments

- Some utility functions and datasets used in this repo are originally from the source code for 
**Are Pre-trained Language Models Aware of Phrases? Simple but Strong Baselines for Grammar Induction** (Y. Kim et al., ACL 2019).
For more details, visit [the original repo](https://github.com/galsang/trees_from_transformers). 