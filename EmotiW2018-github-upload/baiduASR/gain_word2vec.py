'''
Target: gain word2vec features
------------------------------------------------
Input:      
    text
            
Output: 
    word2vec features
------------------------------------------------
Author: Robert Lian

Date: 2018/06/01  Init version
'''

import io
import os
import sys
import glob
import json
import time
import shutil
import torch as t
import numpy as np

# read pretrained fasttext word vector
def load_vectors(fname='wiki-news-300d-1M.vec'):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

word2vec = load_vectors(fname='wiki-news-300d-1M.vec')

# change word to vec
def change_txt_vec(features_path='EmotiW_txt.npz', save_path='EmotiW_txt_vec.npz'):
	data_path=np.load(features_path)['data_path']
	texts_all=np.load(features_path)['text']
	features_all = []
	for texts in texts_all:
		features = []
		texts = texts.split(' ')
		texts = [_ for _ in texts if _!='']
		for text in texts:
			if text in word2vec: features.append(word2vec[text])

		features_all.append(features)

	np.savez_compressed(save_path,
						data_path=data_path,
						word2vec=features_all,
						)

# del empty features
def del_empty(features_path='EmotiW_txt_vec.npz'):
	data_path = np.load(features_path)['data_path']
	features = np.load(features_path)['word2vec']

	data_path_new = []
	features_new = []
	for ii, feature in enumerate(features):
		if len(np.shape(feature))!=1:
			data_path_new.append(data_path[ii])
			features_new.append(feature)

	np.savez_compressed(features_path,
						data_path=data_path_new,
						word2vec=features_new,
						)

if __name__ == '__main__':
	import fire
	fire.Fire()
	#change_txt_vec(features_path='EmotiW_txt.npz')
	#del_empty(features_path='EmotiW_txt_vec.npz')