import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
import glob
import os, string, os.path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def conll_eval(model, pred_path, BI_gt, iterator, criterion, bert_embed, test_msl, model_path):
	
	if model_path:		
		model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
		print("------------->")
		print("[INFO] Model path added...")
	model.eval()
	#####################################
	print('[INFO] Generating a CONLL file.....')

	hc = model.init_hidden().to(device)
	loss_avg = 0.
	total_time = 0.
	with torch.no_grad():
		iterator.create_batches()
		with open(pred_path, 'a') as fp:
			p = 0
			for sample_id, batch in tqdm(enumerate(iterator.batches)):	
				tokens = torch.unsqueeze(batch[0][0], 0).long().to(device)
				seqlens = torch.as_tensor(torch.count_nonzero(tokens, dim=-1), dtype=torch.int64, device='cpu')
				tags = batch[0][1].to(device)
				tags = tags[1:seqlens-1]
				tags = tags - 1

				bert_sent = bert_embed[sample_id].to(device)
				
				start_time = time.time()
				tag_scores, nimp = model(tokens[0], hc, bert_sent, seqlens, test_msl)

				end_time = time.time()
				elapsed_time = end_time - start_time
				total_time = total_time + elapsed_time
					
				tag_scores = torch.log(tag_scores[1:seqlens-1])
				ind = torch.argmax(tag_scores, dim=1)

				loss = criterion(tag_scores, tags)
				loss_avg = loss_avg + loss.item()
				
				prev_pred = ind[0] 
				for j in range(len(ind)):
					if j == 0:		
						fp.write("x "+"y "+ "B"+ " "+ "B")
						fp.write("\n")
					else:	
						if prev_pred == 0:		
							curr_pred = "I"
								
						elif prev_pred == 1:
							curr_pred = "B"

						fp.write("x "+"y "+ str(BI_gt[p][j]) + " "+ curr_pred)
						fp.write("\n")
						prev_pred = ind[j]	
				p = p+1
	print("total_time", total_time)
	return loss_avg / len(iterator)