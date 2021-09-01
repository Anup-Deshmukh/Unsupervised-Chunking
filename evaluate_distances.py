import argparse
import numpy as np
import pickle
from sklearn.metrics import classification_report
import collections


parser.add_argument('--train-test-val', default="train", type=int) # options: ["train", "val", "test"]
parser.add_argument('--dist-measure', default="avg_hellinger" , type=str)  # options = ['avg_hellinger', 'avg_jsd', 'l2', 'cos']
parser.add_argument('--model', default="bert-base-cased" , type=str) # options = ['bert-base-cased', 'bert-base-german-cased']
parser.add_argument('--dist-path', default="output_distances/train.pkl" , type=str) 	
parser.add_argument('--gt-tag-path', default="save_files/train_tag.pkl" , type=str) 	


args = parser.parse_args()

bias = str(0) 
data_type = args.train_test_val 
dist_types = [str(args.dist_measure)] 
model_types = [str(args.model)]
ah_distances = pickle.load(open(str(args.dist_path), "rb"))
data_tags_gt = pickle.load(open(str(args.gt_tag_path), "rb"))

####################################### CONLL2000 ENGLISH DATASET ##################################################
if data_type == "train":
	num_cuts = 87780 
	maxi = 71 #max length of dist vector for a sent (i.e max sent length - 1)
elif data_type == "val":
	num_cuts = 10280 
	maxi = 57 
elif data_type == "test":
	num_cuts = 21839 # 
	maxi = 61
###################################### CONLL2012 REVIEW ENGLISH DATASET ##################################################
# if data_type == "train":
# 	num_cuts = 41433 
# 	maxi = 89 #max length of dist vector for a sent (i.e max sent length - 1)
# elif data_type == "val":
# 	num_cuts = 10394 
# 	maxi = 80 
# elif data_type == "test":
# 	num_cuts = 5209 # 
# 	maxi = 72
####################################### CONLL2003 GERMAN DATASET ##################################################
# if data_type == "train":
# 	num_cuts = 448828 
# 	maxi = 156 #max length of dist vector for a sent (i.e max sent length - 1)
# elif data_type == "val":
# 	num_cuts = 45743 
# 	maxi = 239 
# elif data_type == "test":
# 	num_cuts = 63178 # 
# 	maxi = 222

#############################################################################################################

BI_gt = np.concatenate([np.array(g) for g in data_tags_gt])
num_b = np.count_nonzero(BI_gt == 'B')
print("Num of phrases: ", num_b)
target_names = ['B', 'I'] 
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
			mod_vec = np.zeros((num_sents, maxi))
			#max_dists = [] for threshold
			pred = np.ones((num_sents, maxi+1)).astype(object)
			BI_pred = []
			maxl_sent = 0
			for j in range(num_sents): # jth sent
				
				vec = dist[model][j][i]

				add_zeros = maxi - vec.shape[0]
				if vec.shape[0] < maxi:
					vec = np.pad(vec, (0, add_zeros), 'constant')

				#maxl_sent = max(maxl_sent, vec.shape[0]) # to find maxi
				mod_vec[j] = vec
		
			###### for measures which are directly proportional to distances
			indices = (-mod_vec).argpartition(num_cuts, axis=None)[:num_cuts]

			###### for measures which are inversly proportional to distances (cos)
			#indices = (mod_vec).argpartition(num_cuts, axis=None)[:num_cuts]

			x, y = np.unravel_index(indices, mod_vec.shape)
			
			for row, col in zip(x, y):
					pred[row][col+1] = 'B'   # getting a cut when distances are higher (eg. for avg_helliner)

			for k in range(pred.shape[0]):
				pred[k][0] = 'B'   # first tag for every sent would always be "B"
				non_zero = np.count_nonzero(mod_vec[k]) + 1
				pred[k][non_zero:] = 0

			pred = np.where(pred == 1, 'I', pred) # apart from max dist positions all other tags should be "I"
			
			for l in range(pred.shape[0]):
				for m in range(pred.shape[1]):
					if pred[l][m] != 0:
						BI_pred.append(pred[l][m])
			

			BI_pred = np.array(BI_pred)
			print("len BI pred", len(BI_pred))
			print("len BI gt", len(BI_gt))
			pred_path = f'output_distances/{data_type}-{dist_type}-{i}.out'
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