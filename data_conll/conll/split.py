import pickle
import numpy as np
filepath1 = 'valid_ori.txt'
filepath2 = 'train_val_ori.txt'
train_path = 'train_filtered.txt'

cnt_train_sent = 0
cnt_val_sent = 0
cnt = 0
cnt_sent = 0

sent = []
tag1_vec = []
tag2_vec = []
all_sent = []
all_tag1 = []
all_tag2 = []

big_sent = []
big_tag1_vec = []
big_tag2_vec = []
big_all_sent = []
big_all_tag1 = []
big_all_tag2 = []

with open(filepath1) as fp:
	line = fp.readline()
	
	word = line.split(" ")[0]
	tag1 = line.split(" ")[1]
	tag2 = line.split(" ")[2]
	
	sent.append(word)
	tag1_vec.append(tag1)
	tag2_vec.append(tag2)

	while line:
		line = fp.readline()
		
		if line == "\n" or line == "":		
			all_sent.append(sent)
			all_tag1.append(tag1_vec)
			all_tag2.append(tag2_vec)
			cnt += 1
			
			sent = []
			all_tag1 = []
			all_tag2 = []
			continue

		word = line.split(" ")[0]
		tag1 = line.split(" ")[1]
		tag2 = line.split(" ")[2]

		sent.append(word)
		tag1_vec.append(tag1)
		tag2_vec.append(tag2)

print(cnt)
# print(all_sent[-1])

prev = "hello"
with open(filepath2) as fp, open(train_path, 'a') as wfp:
	line = fp.readline()
	cnt_sent += 1
	word = line.split(" ")[0]
	tag1 = line.split(" ")[1]
	tag2 = line.split(" ")[2]		

	big_sent.append(word)
	big_tag1_vec.append(tag1)
	big_tag2_vec.append(tag2)

	while line:
		line = fp.readline()
		if prev == "Francisco NNP I-NP" and line == "instead RB B-ADVP":
			break
		prev = line
		
		if line == "\n" or line == "":		
			#big_all_sent.append(big_sent)
			#big_all_tag1.append(big_tag1_vec)
			#big_all_tag2.append(big_tag2_vec)
			cnt_sent += 1
			if big_sent not in all_sent:
				cnt_train_sent += 1
				for i in range(len(big_sent)):
					wfp.write(big_sent[i]+" "+big_tag1_vec[i]+" "+big_tag2_vec[i])
				wfp.write("\n")
			else: 
				print(big_sent)
				cnt_val_sent += 1

			big_sent = []
			big_tag1_vec = []
			big_tag2_vec = []

			continue

		word = line.split(" ")[0]
		tag1 = line.split(" ")[1]
		tag2 = line.split(" ")[2]

		big_sent.append(word)
		big_tag1_vec.append(tag1)
		big_tag2_vec.append(tag2)


print(cnt_sent)
print(cnt_val_sent)
print(cnt_train_sent)