import numpy as np
import pickle
from sklearn.metrics import classification_report
import collections
from chunk_heu import *
from utils.measure import Measure
from utils.parser import not_coo_parser, parser
from utils.tools import set_seed, select_indices, group_indices
from utils.yk import get_actions, get_nonbinary_spans

bias = str(0) 
data_type = "test"  # options: ["train", "val", "test"]
dist_types = ['avg_hellinger'] # options = ['avg_hellinger', 'avg_jsd', 'l2', 'cos']
model_types = ['bert-base-cased'] # options = ['bert-base-cased', 'bert-base-german-cased']

ah_distances = pickle.load(open("random/avg_hellinger-"+bias+"-test-distances.pickle", "rb"))

################################## CoNLL2003 dataset #############################################
# data_tokens_gt = pickle.load(open("../german_small_updated/german_test_token.pkl", "rb"))  
# data_tags_gt = pickle.load(open("../german_small_updated/german_test_tag.pkl", "rb"))  
# BI_gt = np.concatenate([np.array(g) for g in data_tags_gt])


################################## CoNLL2000 dataset #############################################
data_tokens_gt = pickle.load(open("data_conll/conll/data_test_tokens.pkl", "rb"))
data_tags_gt = pickle.load(open("data_conll/conll/data_test_tags.pkl", "rb"))
BI_gt = np.concatenate([np.array(g) for g in data_tags_gt])

#############################################################################################################
for dist_type in dist_types:
	
	if dist_type == 'avg_hellinger':
		dist = ah_distances
	elif dist_type == 'avg_jsd':
		dist = aj_distances
	elif dist_type == "l2":
		dist = l2_distances
	elif dist_type == "cos":
		dist = cos_distances

	threshold = []
	for model in model_types:
		print("#################################")
		print("model name: ", model)

		num_sents = len(dist[model])
		num_layers = dist[model][0].shape[0] # 0th sentences's distances
		
		for i in range(num_layers): #ith layer
			print("#################################")
			print("layer no.: ", i)
			pred_chunks = []
			for j in range(num_sents): # jth sent
				raw_tokens = data_tokens_gt[j]

				vec = dist[model][j][i]
				
				pred_tree = parser(vec, raw_tokens)                    
				ps = get_nonbinary_spans(get_actions(pred_tree))[0]
				sent_chunks = max_right(ps)  ## can be changed to nax_left, two_word.
				
				#print(sent_chunks)
				pred_chunks.append(sent_chunks)
				
			BI_pred = np.concatenate([np.array(g) for g in pred_chunks])
			print("len BI pred", len(BI_pred))
			print("len BI gt", len(BI_gt))
			
			pred_path = f'outputs/max_right/test/{dist_type}-{model}-{dist_type}-{i}.out'
			fc = 0
			with open(pred_path, 'a') as fp:
				for p in range(len(BI_pred)):
					fc+=1
					fp.write("x "+"y "+str(BI_gt[p])+" "+str(BI_pred[p]))
					fp.write("\n")

			
			print("representation taken from layer no: ", i)
			print("dist type: ", dist_type)
			print("number of layers in a model: ", num_layers)
			print("test sents processed: ", num_sents)