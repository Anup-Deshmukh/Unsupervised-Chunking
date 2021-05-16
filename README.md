# Language model chunker

This repository contains the code for utlizing the language models for chunking. All the experiements are done on the conll-2000 dataset. 

## Pre-requisite Python Libraries

Install the following libraries specified in the **requirements.txt** before running our code.

    transformers==2.2.0
    numpy==1.15.4
    tqdm==4.26.0
    torch==1.3.1
    nltk==3.4
    matplotlib==2.2.3
    
## Data preparation (CONLL)

Please refer to the code inside data_conll for all the processed files and the original conll data files. 
Details will be updated soon. 

## Files

There are 3 important file apart from the preprocessing of conll:

- **run.py** to extract the features from different layers and different attemtion heads
- **evaluate_distances.py** to evaluate the distances obtained from run.py and create a file ready for evaluation script.  
- **eval_conll2000_updated.pl** updated eval script which only prints overall phrase level F1 and tag level accuracy

## How to run the code?

> python3 run.py [--data-path DATA_PATH] [--result-path RESULT_PATH]

  **important** Inside run.py the input pickle file for the conll tokens must be given. I.e. Update the following line:

  data_tokens = pickle.load(open("data_conll/conll/data_train_tokens.pkl", "rb"))

> python3 evaluate_distances.py

  **important** set the following four variables: data_type, dist_types, model_types, distances pickle file

## Acknowledgments

- Utility functions from the folder "utils" and datasets used in this repo are originally from the source code for 
**Are Pre-trained Language Models Aware of Phrases? Simple but Strong Baselines for Grammar Induction** (Y. Kim et al., ACL 2019).
For more details, visit [the original repo](https://github.com/galsang/trees_from_transformers). 