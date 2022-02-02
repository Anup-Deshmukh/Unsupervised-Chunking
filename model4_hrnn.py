import argparse
import pickle
import gensim
import math
import time
import random
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import transformers
from transformers import *
from extract_features import compute_feat
from torchtext.data import Field, Iterator, BucketIterator
from test_model4_hrnn import conll_eval
from subprocess import run, PIPE
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

from data_conll.conll_utils import epoch_time, valid_conll_eval, data_padding
from data_conll.conll_utils import build_w2v, build_vocab, convert_to_word
#torch.autograd.set_detect_anomaly(True)


EMBEDDING_DIM = 1024 # 768 base, 1024 large
HIDDEN_DIM = 100
NUM_LAYERS = 1
is_training = 0

BATCH_SIZE = 1
NUM_ITER = 50
L_RATE = 0.001
warmup = 50

##### load psuedo labels from compound pcfg model ####
training_data_ori = pickle.load(open("data_conll/from_comppcfg/train_psuedo_data_hrnn_model.pkl", "rb"))
val_data_ori = pickle.load(open("data_conll/from_comppcfg/val_psuedo_data_hrnn_model.pkl", "rb"))
test_data_ori = pickle.load(open("data_conll/from_comppcfg/test_psuedo_data_hrnn_model.pkl", "rb"))

##### load test and val set ####
BI_test_gt = pickle.load(open("data_conll/conll/data_test_tags.pkl", "rb"))
BI_val_gt = pickle.load(open("data_conll/conll/data_val_tags.pkl", "rb"))


# ##### initialize BERT model ####
# bert_model = [BertModel, BertTokenizer, BertConfig, 'bert-large-cased']
# training_data = training_data_ori
# val_data = val_data_ori
# test_data = test_data_ori

training_data = training_data2
val_data = training_data3
test_data = training_data4
bert_model = [BertModel, BertTokenizer, BertConfig, 'bert-large-cased']
BI_val_gt = [['B','B','I','B', 'B', 'I' ], ['B','B','B']]
BI_test_gt = [['B','B','B','I', 'I'], ['B','B','B']]

class HRNNtagger(nn.ModuleList):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size, device):
		super(HRNNtagger, self).__init__()
		self.device = device
		self.hidden_dim = hidden_dim
		#self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
		self.rnn11 = nn.RNNCell(embedding_dim, hidden_dim).to(device)
		self.rnn12 = nn.RNNCell(embedding_dim, hidden_dim).to(device)

		self.rnn21 = nn.RNNCell(hidden_dim, hidden_dim).to(device)

		#self.dropout = nn.Dropout(p=0.5)

		self.hidden2tag = nn.Linear(hidden_dim+hidden_dim+embedding_dim, tagset_size - 1).to(device)
		self.soft = nn.Softmax(dim=1)

		#init some variables
		self.batch_size = batch_size
		self.tagset_size = tagset_size
		

	def forward(self, sentence, h_init, bert_sent, seqlens, max_seq_len):
		output_seq = torch.zeros((seqlens, self.tagset_size - 1)).to(self.device)
		#mask_vec = torch.zeros(seqlens).type(torch.LongTensor).to(self.device)
		
		# print("max seq len", max_seq_len)
		# print("seqlens", seqlens)
		# print("sentence", sentence)


		h11 = h_init.to(self.device)
		h12 = h_init.to(self.device)
		h1_actual = h_init.to(self.device)

		h21 = h_init.to(self.device)
		h22 = h_init.to(self.device)
		h2_actual = h_init.to(self.device)

		for t in range(seqlens):
			#print("t", t)

			if t == 0:
				mask_ind = 1
				entry = torch.unsqueeze(bert_sent[t], 0).to(self.device)
				next_entry = torch.unsqueeze(bert_sent[t+1], 0).to(self.device)

				h11 = self.rnn11(entry, h1_actual)
				h12 = self.rnn12(entry, h_init)
				
				h22 = h2_actual
				h21 = self.rnn21(h1_actual, h2_actual)
				
				h1_actual = mask_ind*h12 + (1-mask_ind)*h11
				h2_actual = mask_ind*h21 + (1-mask_ind)*h22
				h_init = h1_actual

				tag_rep = self.hidden2tag(torch.cat((h1_actual, h2_actual, next_entry), dim=1)).to(self.device)	
				output = torch.squeeze(self.soft(tag_rep))
				output_seq[t] = output

			else:  
				# var_chop = output_seq[t-1].cpu().detach().numpy()
				# mask_ind = np.where(var_chop == np.amax(var_chop))[0][0]
				# mask_vec[t-1] = mask_ind

				entry = torch.unsqueeze(bert_sent[t], 0).to(self.device)
				if t == seqlens-1:
					next_entry = torch.unsqueeze(bert_sent[t], 0).to(self.device)
				else:
					next_entry = torch.unsqueeze(bert_sent[t+1], 0).to(self.device)

				h11 = self.rnn11(entry, h1_actual)
				h12 = self.rnn12(entry, h_init)
				
				h22 = h2_actual
				h21 = self.rnn21(h1_actual, h2_actual)

				h1_actual = torch.mul(h11, output[0]) + torch.mul(h12, output[1])
				h2_actual = torch.mul(h22, output[0]) + torch.mul(h21, output[1])

				tag_rep = self.hidden2tag(torch.cat((h1_actual, h2_actual, next_entry), dim=1)).to(self.device)				
				output = torch.squeeze(self.soft(tag_rep))
				output_seq[t] = output

		return output_seq, h2_actual

	def init_hidden(self):

		# initialize the hidden state and the cell state to zeros
		return (torch.zeros(self.batch_size, self.hidden_dim))


def train(model, optimizer, criterion, iterator, bert_embed, max_seq_len):
	model.train()
	loss_avg = 0.
	
	iterator.create_batches()

	for sample_id, batch in tqdm(enumerate(iterator.batches)):
		tokens = batch[0][0].to(device)
		tags = batch[0][1].to(device)
		bert_sent = bert_embed[sample_id].to(device)
		seqlens = torch.as_tensor(torch.count_nonzero(tokens, dim=-1), dtype=torch.int64, device='cpu')
		hc = model.init_hidden().to(device)

		model.zero_grad()	
		tag_scores, nimp = model(tokens, hc, bert_sent, seqlens, max_seq_len)

		tag_scores = torch.log(tag_scores[1:seqlens-1])
		mask_out = torch.argmax(tag_scores, dim=1)
		
		tags = tags - 1 
		tags = tags[1:seqlens-1]

		loss = criterion(tag_scores, tags)
		loss.backward()	
		optimizer.step()

		loss_avg = loss_avg + loss.item()
		
	return loss_avg / len(iterator)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--is-training', default=1, type=int) 
	parser.add_argument('--dire', default="save_files/" , type=str) 	
	args = parser.parse_args()
	best_model_path = args.dire + 'best-model-hrnn.pt'
	pred_path_test_out = 'best_test.txt'
	opt_path = args.dire + 'sgd.opt'
	
	print("------------->")
	print('[INFO] Build vocabulary.....')
	word_to_ix, ix_to_word, tag_to_ix = build_vocab(training_data, test_data, val_data)
	
	print("device is:", device)
	hrnn_model = HRNNtagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), BATCH_SIZE, device).to(device)

	loss_function = nn.NLLLoss().to(device)
	#optimizer = optim.SGD(hrnn_model.parameters(), lr=L_RATE)
	optimizer = optim.Adam(hrnn_model.parameters(), lr=L_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	
	scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
		num_warmup_steps = warmup, num_training_steps = NUM_ITER, 
		num_cycles = 0.5, last_epoch = - 1)

	# scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
	# 	num_warmup_steps = warmup, num_training_steps = NUM_ITER, 
	# 	num_cycles = 2, last_epoch = - 1)

	# scheduler = transformers.get_constant_schedule_with_warmup(optimizer, 
	# 	num_warmup_steps = warmup, last_epoch = - 1)

	if args.is_training:

		############ validation data ############
		v_tokens, v_tags, val_msl = data_padding(val_data, word_to_ix, tag_to_ix)
		get_sent = {v: k for k, v in word_to_ix.items()}

		print("------------->")
		print('[INFO] Create validation embeddings matrix.....')
		v_matrix = compute_feat(bert_model, v_tokens.to(device), word_to_ix, ix_to_word, device).to(device)
	
		val_final_data = [(v_tokens[i], v_tags[i]) for i in range(0, len(v_tokens))] 
		val_iterator = BucketIterator(val_final_data, 
			batch_size=BATCH_SIZE, sort_key=lambda x: np.count_nonzero(x[0]), sort=False, 
			shuffle=False, sort_within_batch=False, device = device)

		############ training data ############
		tokens, tags, train_msl = data_padding(training_data, word_to_ix, tag_to_ix)
		print("------------->")
		print('[INFO] Create training embeddings matrix.....')
		matrix = compute_feat(bert_model, tokens.to(device), word_to_ix, ix_to_word, device).to(device)
		
		train_final_data = [(tokens[i], tags[i]) for i in range(0, len(tokens))] 	
		train_iterator = BucketIterator(train_final_data, 
			batch_size=BATCH_SIZE, sort_key=lambda x: np.count_nonzero(x[0]), sort=False, 
			shuffle=False, sort_within_batch=False, device = device)

		train_loss_vec = []
		val_fscore_vec = []
		best_fscore = 0.

		for epoch in range(NUM_ITER):  
			start_time = time.time()
			print("------------->")
			print("[INFO] Training procedure.....")		

			train_loss = train(hrnn_model, optimizer, loss_function, train_iterator, matrix, train_msl)
			
			print("[INFO] Valid procedure.....")	
			pred_path_val_out = args.dire + 'validation/test_outputs' + str(epoch) + '.out'
			loss_nimp = conll_eval(hrnn_model,  pred_path_val_out, BI_val_gt, val_iterator, loss_function, v_matrix, val_msl, model_path=None)
			fscore = valid_conll_eval(pred_path_val_out)

			end_time = time.time()
			epoch_mins, epoch_secs = epoch_time(start_time, end_time)

			train_loss_vec.append(train_loss)
			val_fscore_vec.append(fscore)

			scheduler.step()

			if fscore > best_fscore:
				best_fscore = fscore
				torch.save(hrnn_model.state_dict(), best_model_path)	

			if epoch%10 == 0:
				torch.save(hrnn_model.state_dict(), args.dire+ str(epoch)+'hrnn.pt')

			torch.save(optimizer.state_dict(), opt_path)

			print("------------->")
			print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
			print('LR:', scheduler.get_last_lr())
			print(f'\tTrain Loss: {train_loss:.3f}')
			print(f'\t Validation F score: {fscore:.3f}')
	
			plot_var = {'p1': train_loss_vec, 'p2': val_fscore_vec}
			with open(args.dire+'plot.pkl', 'wb') as f:
				pickle.dump(plot_var, f) 

	else:
		############ test data ############
		test_tokens, test_tags, test_msl = data_padding(test_data, word_to_ix, tag_to_ix)
		get_sent = {v: k for k, v in word_to_ix.items()}
		
		print("------------->")
		print('[INFO] Create test embeddings matrix.....')
		
		test_matrix = compute_feat(bert_model, test_tokens, word_to_ix, ix_to_word, device)
		with open(args.dire+'test_matrix.pkl', 'wb') as f:
			pickle.dump(test_matrix, f) 
		
		#test_matrix = pickle.load(open("hrnn_pcfg/hrnn5/" + "test_matrix.pkl", "rb"))

		test_final_data = [(test_tokens[i], test_tags[i]) for i in range(0, len(test_tokens))] 	
		test_iterator = BucketIterator(test_final_data, 
			batch_size=BATCH_SIZE, sort_key=lambda x: np.count_nonzero(x[0]), sort=False, 
			shuffle=False, sort_within_batch=False, device = device)

		loss_nimp = conll_eval(hrnn_model, pred_path_test_out, BI_test_gt, test_iterator, loss_function, test_matrix, test_msl, best_model_path)
	
		print(f'| Test Loss: {loss_nimp:.3f}')
		

if __name__ == "__main__":
	main()
