import numpy as np
import pickle
from sklearn.metrics import classification_report
import collections

data_type = "val"  # options: ["train", "val", "test"]

#dist_types = ['avg_hellinger', 'avg_jsd', 'l2', 'cos']
dist_types = ['avg_hellinger']

model_types = ['bert-large-cased']
#rem_model_types = ['xlnet-base-cased', 'xlnet-large-cased']

ah_distances = pickle.load(open("outputs/avg_hellinger-val-distances.pickle", "rb"))
#aj_distances = pickle.load(open("outputs/run4_test_dists/test_filtered-mean-avg_jsd-distances.pickle", "rb"))
#l2_distances = pickle.load(open("outputs/run4_test_dists/test_filtered-mean-l2-distances.pickle", "rb"))

#############################################################################################################
if data_type == "train":
	num_cuts = 87780 
	maxi = 71 #max length of dist vector for a sent (i.e max sent length - 1)
elif data_type == "val":
	num_cuts = 10280 
	maxi = 57 
elif data_type == "test":
	num_cuts = 21839 # 
	maxi = 61

#############################################################################################################
data_tags_gt = pickle.load(open("data_conll/conll/data_"+data_type+"_tags.pkl", "rb"))
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

				maxl_sent = max(maxl_sent, vec.shape[0]) # to find maxi
				mod_vec[j] = vec
			print("max_length of a sent - 1: ", maxl_sent)
			# print("dataset vec shape", mod_vec.shape)
			# print("vec dist for 10th sent:", mod_vec[10])

			###### for measures which are directly proportional to distances
			indices = (-mod_vec).argpartition(num_cuts, axis=None)[:num_cuts]

			###### for measures which are inversly proportional to distances (cos)
			#indices = (-mod_vec).argpartition(num_cuts, axis=None)[:num_cuts]

			x, y = np.unravel_index(indices, mod_vec.shape)
			
			for row, col in zip(x, y):
					pred[row][col+1] = 'B'
					#max_dists.append(mod_vec[row][col]) # for threshold

			#threshold.append(np.amin(max_dists)) # for threshold

			for k in range(pred.shape[0]):
				pred[k][0] = 'B'
				non_zero = np.count_nonzero(mod_vec[k]) + 1
				pred[k][non_zero:] = 0

			pred = np.where(pred == 1, 'I', pred)
			
			#print("changed pred for 10th", pred[142])
			#print("gt tag:", data_tags_gt[142])
			
			for l in range(pred.shape[0]):
				for m in range(pred.shape[1]):
					if pred[l][m] != 0:
						BI_pred.append(pred[l][m])
			

			BI_pred = np.array(BI_pred)
			
			pred_path = f'outputs/val/{dist_type}-{model}-{dist_type}-{i}.out'
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

			#print("results: \n")
			#print(classification_report(BI_gt, BI_pred, target_names=target_names))
		
	#print(len(threshold))
	#with open(f'outputs/run6/{dist_type}.pickle', 'wb') as f:
	#	pickle.dump(threshold, f)