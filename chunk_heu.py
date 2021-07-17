import argparse
import datetime
import logging
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
import numpy 

def max_left(sp):
	bi = np.ones(sp[-1][1]+1).astype(object) 
	#print(bi)
	for pair in sp:
		if pair[1] == pair[0]+1:
			bi[pair[0]] = 'B'
			bi[pair[1]] = 'I'
			for nextpair in sp:
				if nextpair[0] == pair[0] and nextpair[1] == pair[1]+1:
					bi[nextpair[1]] = 'I'
					pair = nextpair
	#print(bi)
	bi[0] = 'B'
	#print(bi)
	bi = np.where(bi == 1, 'B', bi)		
	#print(bi)
	return bi

def max_right(sp):
	bi = np.ones(sp[-1][1]+1).astype(object) 
	for pair in sp:
		if pair[1] == pair[0]+1:
			bi[pair[0]] = 'B'
			bi[pair[1]] = 'I'
			for nextpair in sp:
				if nextpair[1] == pair[1] and nextpair[0] == pair[0]-1:
					bi[nextpair[0]] = 'B'
					bi[pair[0]] = 'I'
					pair = nextpair
	bi[0] = 'B'
	bi = np.where(bi == 1, 'B', bi)		
	return bi	

def two_word(sp):
	bi = np.ones(sp[-1][1]+1).astype(object) 
	for pair in sp:
		if pair[1] == pair[0]+1:
			bi[pair[0]] = 'B'
			bi[pair[1]] = 'I'
			
	bi[0] = 'B'
	bi = np.where(bi == 1, 'B', bi)		
	return bi	
