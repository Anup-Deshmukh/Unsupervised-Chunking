import argparse
import datetime
import logging
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
import numpy 
from transformers import *

from data.dataset import Dataset
from utils.measure import Measure
from utils.parser import not_coo_parser, parser
from utils.tools import set_seed, select_indices, group_indices
from utils.yk import get_actions, get_nonbinary_spans


#MODELS = [(AutoModelForMaskedLM, AutoTokenizer, BertConfig, 'bert-base-german-cased')]
MODELS = [(BertModel, BertTokenizer, BertConfig, 'bert-base-cased')]

# MODELS = [(BertModel, BertTokenizer, BertConfig, 'bert-base-cased'),
#           (BertModel, BertTokenizer, BertConfig, 'bert-large-cased'),
#           (GPT2Model, GPT2Tokenizer, GPT2Config, 'gpt2'),
#           (GPT2Model, GPT2Tokenizer, GPT2Config, 'gpt2-medium')]


def compute_dist(args):
    scores = dict()
    syn_dists_all = dict()
    dist_type = args.dist_type

    for model_class, tokenizer_class, model_config, pretrained_weights in MODELS:
        tokenizer = tokenizer_class.from_pretrained(
            pretrained_weights, cache_dir=args.lm_cache_path)

        if args.from_scratch:
            config = model_config.from_pretrained(pretrained_weights)
            config.output_hidden_states = True
            config.output_attentions = True
            model = model_class(config).to(args.device)
        else:
            model = model_class.from_pretrained(
                pretrained_weights,
                cache_dir=args.lm_cache_path,
                output_hidden_states=True,
                output_attentions=True).to(args.device)

        with torch.no_grad():
            test_sent = tokenizer.encode('test', add_special_tokens=False)
            token_ids = torch.tensor([test_sent]).to(args.device)
            all_hidden, all_att = model(token_ids)[-2:]
            n_layers = len(all_att)
            n_att = all_att[0].size(1)
            n_hidden = all_hidden[0].size(-1)
       
        measure = Measure(n_layers, n_att)
        data_tokens = pickle.load(open(args.data_path, "rb")) ### change to train token file to get distances for trianing distances ###
        #data_tokens = data_tokens[0:5657]
        syn_dists_sents = []
        buggy = []
        for idx, s_tokens in tqdm(enumerate(data_tokens)):
            # if len(s_tokens) <= 2:
            #     print("buggy sent ids", idx)
            #     buggy.append(idx)
            #     continue
            raw_tokens = s_tokens
            s = ' '.join(s_tokens)
            tokens = tokenizer.tokenize(s)
            
            # print("\n")
            # print("index is:", idx)
            # print("sentence is:", s)
            # print("raw tokens is:", raw_tokens)
            # print("tokens is:", tokens)
            
            token_ids = tokenizer.encode(s, add_special_tokens=False)
            token_ids_tensor = torch.tensor([token_ids]).to(args.device)
            with torch.no_grad():
                all_hidden, all_att = model(token_ids_tensor)[-2:]
            all_hidden, all_att = list(all_hidden[1:]), list(all_att)

            # (n_layers, seq_len, hidden_dim)
            all_hidden = torch.cat([all_hidden[n] for n in range(n_layers)], dim=0)
            # (n_layers, n_att, seq_len, seq_len)
            all_att = torch.cat([all_att[n] for n in range(n_layers)], dim=0)

            ################ ################ ################ ################ ##########
            ################ ################ ################ ################ ##########
            ################ ################ ################ ################ ##########

            #################### further pre processing ##############
            if len(tokens) > len(raw_tokens):
                #print("further")
                th = args.token_heuristic
                if th == 'first' or th == 'last':
                    mask = select_indices(tokens, raw_tokens, pretrained_weights, th)
                    assert len(mask) == len(raw_tokens)
                    all_hidden = all_hidden[:, mask]
                    all_att = all_att[:, :, mask, :]
                    all_att = all_att[:, :, :, mask]
                else:

                    mask = group_indices(tokens, raw_tokens, pretrained_weights)
                    # if mask is None:
                    #     print("buggy sent ids", idx)
                    #     buggy.append(idx)
                    #     continue
                    raw_seq_len = len(raw_tokens)
                    all_hidden = torch.stack(
                        [all_hidden[:, mask == i].mean(dim=1)
                         for i in range(raw_seq_len)], dim=1)
                    all_att = torch.stack(
                        [all_att[:, :, :, mask == i].sum(dim=3)
                         for i in range(raw_seq_len)], dim=3)
                    all_att = torch.stack(
                        [all_att[:, :, mask == i].mean(dim=2)
                         for i in range(raw_seq_len)], dim=2)

            l_hidden, r_hidden = all_hidden[:, :-1], all_hidden[:, 1:]
            l_att, r_att = all_att[:, :, :-1], all_att[:, :, 1:]
      
            syn_dists = measure.derive_dists(l_hidden, r_hidden, l_att, r_att)
            

            ############ induce bias ##################
            for m, d in syn_dists.items():
                for i in range(measure.scores[m].n):
                    dist = syn_dists[m][i].tolist()
                    if len(dist) > 1:
                        bias_base = (sum(dist) / len(dist)) * args.bias
                        bias = [bias_base * (1 - (1 / (len(dist) - 1)) * x)
                                for x in range(len(dist))]
                        dist = [dist[i] + bias[i] for i in range(len(dist))]
                    
                    syn_dists[m][i] = torch.FloatTensor(dist)
        

            new_dist = syn_dists[dist_type].numpy()
            syn_dists_sents.append(new_dist)
           
            ################ ################ ################ ################ ##########
            ################ generate parse trees from syn dists and evaluate ############      
            ################ ################ ################ ################ ##########
        
        syn_dists_all[pretrained_weights] = syn_dists_sents
        print("### model done ###")
    return syn_dists_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        default='data_conll/conll/data_test_tokens.pkl', type=str) ## must give
    parser.add_argument('--dist-type', type=str) ## must give
    parser.add_argument('--result-path', default='outputs', type=str) ## must give
    parser.add_argument('--lm-cache-path',
                        default='data/transformers', type=str)
    parser.add_argument('--from-scratch', default=False, action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--bias', default=0.0, type=float, ## must give
                        help='the right-branching bias hyperparameter lambda')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--token-heuristic', default='mean', type=str,
                        help='Available options: mean, first, last')

    args = parser.parse_args()

    setattr(args, 'device', f'cuda:{args.gpu}'
    if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    dataset_name = args.data_path.split('/')[-1].split('.')[0]
    pretrained = 'scratch' if args.from_scratch else 'pretrained'
    result_path = args.result_path
    setattr(args, 'result_path', result_path)
    set_seed(args.seed)
    
    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')
    print("###")
    
    distances = compute_dist(args)
    
    with open(result_path, 'wb') as f:
        pickle.dump(distances, f)


if __name__ == '__main__':
    main()
