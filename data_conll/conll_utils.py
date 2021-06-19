import time
import pickle
import gensim
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import transformers

from nltk.tokenize import word_tokenize
from transformers import *
from extract_features import compute_feat
from torchtext.data import Field, Iterator, BucketIterator
from test_model4_hrnn import conll_eval
from subprocess import run, PIPE

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def build_vocab(t_data, v_data, test_data):
	word_to_ix = {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3}
	net = []
	for a,b in t_data:
		net.append(a)
	for c,d in v_data:
		net.append(c)
	for e,f in test_data:
		net.append(e)
	
	for sent in net:
		for word in sent:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
	
	ix_to_word = {v: k for k, v in word_to_ix.items()}
	tag_to_ix = {"<pad>": 0, "1": 1, "2": 2}

	return word_to_ix, ix_to_word, tag_to_ix

def build_w2v(file_path, word_to_ix, dec_emb_dim):
	print("------------->")
	print('[INFO] Create W2V matrix for AE.....')
	
	with open(file_path, 'r') as f:
		snli_data = f.readlines()
	
	print("conll data reading done.....")
	sent_mat = [s.strip() for s in snli_data]
	np.random.shuffle(sent_mat)
	sent_mat = [word_tokenize(s) for s in sent_mat]
	
	print("conll data tokenizing done.....")
	w2v_model = gensim.models.Word2Vec(sent_mat,
                                   size=dec_emb_dim,
                                   min_count=1,
                                   iter=50)
	
	print("w2v model initialized.....")
	embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(word_to_ix), dec_emb_dim))
	for word, i in word_to_ix.items():
		try:
			embeddings_matrix[i] = w2v_model[word]
		except KeyError:
			pass

	return embeddings_matrix

def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)

def tile_rl(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx)).to(device)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index).to(device)

def data_padding(data, word_to_ix, tag_to_ix, max_seq_len=20):
	data_lengths = [len(sentence)+2 for sentence,tags in data]
	
	max_seq_len = max(data_lengths)
	padded_data = torch.empty(len(data), max_seq_len, dtype=torch.long).to(device)
	padded_data.fill_(0.)
	# copy over the actual sequences
	for i, x_len in enumerate(data_lengths): 
		sequence,tags = data[i]
		sequence.insert(0,'SOS')
		sequence.append('EOS')
		
		sequence = prepare_sequence(sequence, word_to_ix)
		#print(sequence)
		padded_data[i, 0:x_len] = sequence[0:x_len]

	#print(padded_data)
	tag_lengths = [len(tags)+2 for sentence, tags in data]
	padded_tags = torch.empty(len(data), max_seq_len, dtype=torch.long).to(device)
	padded_tags.fill_(0.)

	for i, y_len in enumerate(tag_lengths):
		sequence,tags = data[i]
		tags.insert(0,'<pad>')  # for SOS
		tags.append('<pad>')  ## for EOS

		tags = prepare_sequence(tags, tag_to_ix)
		padded_tags[i, 0:y_len] = tags[:y_len]

	return padded_data, padded_tags, max_seq_len

def valid_conll_eval(fname):

	with open(fname, 'r') as file:
		data = file.read()

	pipe = run(["perl", "eval_conll2000_updated.pl"], stdout=PIPE, input=data, encoding='ascii')
	output = pipe.stdout

	tag_acc = float(output.split()[0])
	phrase_f1 = float(output.split()[1])

	print("tag_acc, phrase_f1", tag_acc, phrase_f1)
	return phrase_f1


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def convert_to_word(sents, preds, dicti):
	
	ori_new = []
	pred_new = []
	for sent in sents:
		sent = sent[(sent!=0) & (sent!=3)] # remove pad and eos tokens
		l1 = []
		for i in range(len(sent)):
			if sent[i] == 1: # THIS IS A JUGAAD (replace UNK with a random word since it is anyways predicted wrong)
				l1.append('Serendipity')
			else:
				l1.append(dicti[sent[i]])

		ori_new.append([l1]) # list of l1 because of corplu bleu requires that
	
	for pred in preds:
		if 3 in pred:
			eos_ind = np.where(pred==3)[0][0] # index of eos token
			pred = pred[:eos_ind]
		else:
			eos_ind = len(pred)-1
			pred = pred[:eos_ind+1]

		pred = pred[(pred!=0) & (pred!=3)] # remove pad and eos tokens
		l2 = []
		for i in range(len(pred)):
			l2.append(dicti[pred[i]])
		
		pred_new.append(l2)
	
	return ori_new, pred_new