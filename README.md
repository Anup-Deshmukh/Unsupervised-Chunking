# Unsupervised Chunking

This repository contains the code of our knowledge transfer approach for unsupervised chunking. 

## Pre-requisite Python Libraries
Install the following libraries specified in the **requirements.txt** before running our code.

    transformers==2.2.0
    numpy==1.15.4
    tqdm==4.26.0
    torch==1.3.1
    nltk==3.4
    matplotlib==2.2.3
    
## Data preparation (CONLL)

All the experiements and results in this repository are done on the CoNLL-2000 dataset. Please refer to the code inside data_conll for all the processed files and the original conll data files. 

## Files

All the output files ready for conll eval script are given in the output_chunks directory. The reported results on the test set for CPCFG, LM and HRNN can be replicated by running ***eval_conll2000_updated.pl*** (updated official perl eval script which only prints overall phrase level F1 and tag level accuracy)

Langugae model for unsupervised chunking
- **run.py** to extract the features from different layers and different attemtion heads
- **evaluate_distances.py** to evaluate the distances obtained from run.py and create a file ready for evaluation script. 
- **evaluate_heuristics.py** to evaluate maximal left branching, maximal right branching heuristics on top of LM parser.

Hierarchical RNN for unsupervised chunking
- **model4_hrnn.py** 
- **test_model4_hrnn.py**  

## How to run the code?

Langugae model for unsupervised chunking
- python3 run.py [--data-path DATA_PATH] [--result-path RESULT_PATH]
- python3 evaluate_distances.py [--traintestval train] [--distmeasure avg_hellinger] [--model bert-base-cased] [--distpath DIST_PATH] [--gttagpath GT_TAG_PATH]

Compound PCFG 
- 

Hierarchical RNN for unsupervised chunking
- python3 model4_hrnn.py [--is-training 1] [--dire SAVE_PATH] 
- python3 test_model4_hrnn.py 

## Acknowledgments

- Utility functions from the folder "utils" and datasets used in this repo are originally from the source code for: 
- Are Pre-trained Language Models Aware of Phrases? Simple but Strong Baselines for Grammar Induction** (Y. Kim et al., ACL 2019).
- For more details, visit [the original repo](https://github.com/galsang/trees_from_transformers). 