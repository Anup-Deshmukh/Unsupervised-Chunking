import argparse
import torch
import pickle
import numpy as np
import datetime
import logging
import os
from tqdm import tqdm
from transformers import *

from torchtext.data import Field, BucketIterator
from utils.measure import Measure
from utils.parser import not_coo_parser, parser
from utils.tools import set_seed, select_indices, group_indices
from utils.yk import get_actions, get_nonbinary_spans


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def compute_feat(model_disc, data_tokens, word_to_ix, ix_to_word, device, token_heuristic="mean"):
    scores = dict()
    syn_dists_all = dict()
    max_seq_len = 0
   
    model_class = model_disc[0]
    tokenizer_class  = model_disc[1]
    model_config = model_disc[2]
    pretrained_weights = model_disc[3]

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, cache_dir='LM/cache')
    model = model_class.from_pretrained(pretrained_weights,cache_dir='LM/cache', 
        output_hidden_states=True, output_attentions=True).to(device)

    with torch.no_grad():
        test_sent = tokenizer.encode('test', add_special_tokens=False)
        token_ids = torch.tensor([test_sent]).to(device)
        all_hidden, all_att = model(token_ids)[-2:]
        
        n_layers = len(all_att)
        n_att = all_att[0].size(1)
        n_hidden = all_hidden[0].size(-1)
    
    measure = Measure(n_layers, n_att)
    feat_sents = torch.zeros([len(data_tokens), len(data_tokens[0]), 1024]) 

    for idx, s_tokens in tqdm(enumerate(data_tokens)):
        #################### read words and extract ##############
        
        s_tokens = [ix_to_word[ix] for ix in s_tokens.cpu().numpy()]

       
        raw_tokens = s_tokens
        s = ' '.join(s_tokens)
        tokens = tokenizer.tokenize(s)

        token_ids = tokenizer.encode(s, add_special_tokens=False)
        token_ids_tensor = torch.tensor([token_ids]).to(device)
        with torch.no_grad():
            all_hidden, all_att = model(token_ids_tensor)[-2:]
        all_hidden, all_att = list(all_hidden[1:]), list(all_att)
        
        # (n_layers, seq_len, hidden_dim)
        all_hidden = torch.cat([all_hidden[n] for n in range(n_layers)], dim=0)
        # (n_layers, n_att, seq_len, seq_len)
        all_att = torch.cat([all_att[n] for n in range(n_layers)], dim=0)
        
        #################### further pre processing ##############
        if len(tokens) > len(raw_tokens):
            th = token_heuristic
            if th == 'first' or th == 'last':
                mask = select_indices(tokens, raw_tokens, pretrained_weights, th)
                assert len(mask) == len(raw_tokens)
                all_hidden = all_hidden[:, mask]
                all_att = all_att[:, :, mask, :]
                all_att = all_att[:, :, :, mask]
            else:

                mask = group_indices(tokens, raw_tokens, pretrained_weights)
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

    
        all_hidden = all_hidden[n_layers - 1]
        feat_sents[idx] = all_hidden	
    
    return feat_sents








