import pickle
import numpy as np


sent = []
sent_tag = []
cnt_punct = 0
cnt_sents = 0
cnt_tokens = 0
cnt_phrases = 0
#filepath = 'test_filtered.txt'
#filepath = 'valid_ori.txt'
filepath = 'train_filtered.txt'

data_tokens = []
data_tags = []
O_words = []
cnt_O = 0
cut_tokens = 0
max_sent_len = 0
list_punct = [",", ".", ":", ";", "-","--","``","''","!","?","$","&", "%", "#","...","'","`"]
with open(filepath) as fp:
	line = fp.readline()
	
	word = line.split(" ")[0]
	tag2 = line.split(" ")[2]
	tag = tag2.split("-")[0]
	
	cnt_tokens += 1
	sent.append(word)
	sent_tag.append(tag)
	
	while line:
		line = fp.readline()
		
		if line == "\n" or line == "":
			if len(sent) <= 1:
				continue				
			data_tokens.append(sent)
			data_tags.append(sent_tag)
			
			if len(sent) >  max_sent_len:
				max_sent_len = len(sent)
			
			sent = []
			sent_tag = []
			cnt_sents += 1
			continue
		
		word = line.split(" ")[0]
		tag2 = line.split(" ")[2]
	
		if tag2 == "O\n" or tag2 == "O":
			cnt_O += 1
			if word in list_punct:
				cnt_punct += 1

			if word not in O_words:
				O_words.append(word)
			continue	
		if word == "#":
			cut_tokens += 1
			continue

		tag = tag2.split("-")[0]	
		cnt_tokens += 1	
		sent.append(word)
		sent_tag.append(tag)
	
data_tokens = np.array(data_tokens)
data_tags = np.array(data_tags)	
O_words = np.array(O_words)

print(cnt_O)
print(cnt_punct)
print(cnt_sents)
print(cnt_tokens)

print(data_tokens.shape)
print(data_tags.shape)
print(O_words.shape)
print(cut_tokens)
print("max sent len:", max_sent_len)

pickle.dump(data_tokens, open( "data_train_tokens.pkl", "wb" ) )
pickle.dump(data_tags, open( "data_train_tags.pkl", "wb" ) )

