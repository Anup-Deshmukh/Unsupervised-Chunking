import numpy as np
import pickle
from sklearn.metrics import classification_report
import collections

maxi = 61 #max length of dist vector for a sent (i.e max sent length - 1)

#dist_types = ['avg_hellinger', 'avg_jsd', 'l2', 'cos']
dist_types = ['avg_hellinger']


model_types = ['bert-large-cased']
rem_model_types = ['roberta-base', 'roberta-large','xlnet-base-cased', 'xlnet-large-cased']

data_tags_gt = pickle.load(open("data_conll/conll/data_test_tags.pkl", "rb"))
BI_gt = np.concatenate([np.array(g) for g in data_tags_gt])
num_b = np.count_nonzero(BI_gt == 'B')
print("Num of phrases: ", num_b)
target_names = ['B', 'I']  

threshold_avg_h = pickle.load(open("outputs/run6_val/avg_hellinger.pickle", "rb"))
#threshold_avg_j = pickle.load(open("outputs/run6_val/avg_jsd.pickle", "rb"))
#threshold_l2 = pickle.load(open("outputs/run6_val/l2.pickle", "rb"))

ah_distances = pickle.load(open("random/avg_hellinger-test-distances.pickle", "rb"))
#aj_distances = pickle.load(open("outputs/run4_test_dists/test_filtered-mean-avg_jsd-distances.pickle", "rb"))
#l2_distances = pickle.load(open("outputs/run4_test_dists/test_filtered-mean-l2-distances.pickle", "rb"))

for dist_type in dist_types:
	
	if dist_type == 'avg_hellinger':
		dist = ah_distances
		threshold = threshold_avg_h
	elif dist_type == 'avg_jsd':
		dist = aj_distances
		threshold = threshold_avg_j
	elif dist_type == "l2":
		dist = l2_distances
		threshold = threshold_l2

	model_id = 0
	for model in model_types:
		print("#################################")
		print("model name: ", model)

		num_sents = len(dist[model])
		num_layers = dist[model][0].shape[0] # 0th sentences's distances
		
		for i in range(num_layers): #ith layer

			mod_vec = np.zeros((num_sents, maxi))
			max_dists = []
			pred = np.ones((num_sents, maxi+1)).astype(object)
			BI_pred = []
			maxl_sent = 0
			for j in range(num_sents): # jth sent
				
				vec = dist[model][j][i]
				maxl_sent = max(maxl_sent, vec.shape[0])
				add_zeros = maxi - vec.shape[0]
				if vec.shape[0] < maxi:
					vec = np.pad(vec, (0, add_zeros), 'constant')
				mod_vec[j] = vec
			print("max_length of a sent - 1: ", maxl_sent)

			print("Threshold for this model is: ", threshold[model_id])
			for g in mod_vec:
				g[g >= threshold[model_id]] = 100
			indi_tuple = np.where(mod_vec == 100)
			x = indi_tuple[0]
			y = indi_tuple[1]
			model_id = model_id + 1

			for row, col in zip(x, y):
					pred[row][col+1] = 'B'

			for k in range(pred.shape[0]):
				pred[k][0] = 'B'
				non_zero = np.count_nonzero(mod_vec[k]) + 1
				pred[k][non_zero:] = 0

			pred = np.where(pred == 1, 'I', pred)
			
			for l in range(pred.shape[0]):
				for m in range(pred.shape[1]):
					if pred[l][m] != 0:
						BI_pred.append(pred[l][m])
			
			BI_pred = np.array(BI_pred)
		
			pred_path = f'random/bias0/{dist_type}/{model}-{dist_type}-{i}.out'
			fc = 0
			with open(pred_path, 'a') as fp:
				for p in range(len(BI_pred)):
					fc+=1
					fp.write("x "+"y "+str(BI_gt[p])+" "+ str(BI_pred[p]))
					fp.write("\n")
			print("representation taken from layer no: ", i)
			print("dist type: ", dist_type)
			print("number of layers in a model: ", num_layers)
			print("test sents processed: ", num_sents)

			#print("results: \n")


			#print(classification_report(BI_gt, BI_pred, target_names=target_names))
			print("#################################")