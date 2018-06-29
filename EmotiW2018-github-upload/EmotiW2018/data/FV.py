#coding:utf8
import os
import cv2
import json
import time
import tqdm
import glob
import shutil
import argparse
import numpy as np

from cyvlfeat.fisher import fisher
from numpy.testing import assert_allclose

# datas: [N, D] -> [N, len_norm]
def features_norm(datas, len_norm):
	if not isinstance(datas,list): datas = datas.tolist()
	N, D = np.shape(datas)
	if N>len_norm:
		return datas[:len_norm]
	else:
		while np.shape(datas)[0]!= len_norm: datas.append(datas[-1])
		return datas

def gain_mu_sigma_prior(K, D):
	mu = np.zeros(D * K, dtype=np.float32)
	for i in range(D * K): mu[i] = i
	mu = mu.reshape(D, K)

	sigma2 = np.ones((D, K), dtype=np.float32)
	prior = (1.0 / K) * np.ones(K, dtype=np.float32)

	return mu, sigma2, prior

def fv_encoder(datas, N, K, D):
	datas = features_norm(datas, N) # [N, D]
	datas = np.array(datas).transpose() # [D, N]
	mu, sigma2, prior = gain_mu_sigma_prior(K, D) # [D, K]
	observed_enc = fisher(datas.astype('float32'), mu, sigma2, prior, verbose=False)
	return observed_enc
