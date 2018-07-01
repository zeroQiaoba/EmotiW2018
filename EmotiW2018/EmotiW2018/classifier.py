#coding:utf8
import os
import cv2
import copy
import json
import time
import tqdm
import glob
import shutil
import argparse
import numpy as np

import torch 
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from config import opt
from data import *
import models

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

def read_datas(type_, temporal_, name_2_features):
	len_=0
	datas = []
	labels = []
	train_datas = np.load(opt.label_root)[type_]
	for item in train_datas:
		pic_features = []
		pic_path = item['pic_path'] # video name
		pic_label = item['label']
		
		# extract features
		if temporal_.find('Video')!=-1: # 'Video_None'
			if pic_path not in name_2_features: continue
			pic_features = name_2_features[pic_path]
		else:
			for frame_path in glob.glob('%s/%s/*' %(opt.pic_root, pic_path)):
				frame_path = "/".join(frame_path.split('/')[-2:])
				pic_features.append(name_2_features[frame_path])

		# fusion methods
		if temporal_.find('max')!=-1:
			features = np.max(pic_features, axis=0)
			datas.append(features)
			labels.append(pic_label)
		if temporal_.find('mean')!=-1:
			features = np.mean(pic_features, axis=0)
			datas.append(features)
			labels.append(pic_label)
		if temporal_.find('FV')!=-1: # 'FV_K'
			K=int(temporal_.split('_')[-1]) # 1,2,4,8,16
			N=int(temporal_.split('_')[-2]) # 100
			D=np.shape(pic_features)[-1]
			features = fv_encoder(pic_features, N, K, D)
			datas.append(features) # 2*K*D
			labels.append(pic_label)
		if temporal_.find('None')!=-1:
			features = pic_features
			datas.append(features)
			labels.append(pic_label)

	return datas, labels

def train_NN(model, scaler, X_train, y_train, X_val, y_val):
	# feature normlization
	lr,lr2=opt.lr,opt.lr2
	loss_function = torch.nn.CrossEntropyLoss()
	optimizer = model.get_optimizer(lr, lr2, opt.weight_decay)
	best_score = -float('inf')
	best_model = ''
	for epoch in range(opt.max_epoch):
		images = torch.from_numpy(X_train).float()
		labels = torch.from_numpy(np.array(y_train)).long()
		images, labels = Variable(images), Variable(labels)
		optimizer.zero_grad()
		outputs = model(images)
		loss = loss_function(outputs, opt.weight*labels)
		loss.backward()
		optimizer.step()

		# test for each epoch
		model.eval()
		images = torch.from_numpy(X_val).float()
		labels = torch.from_numpy(np.array(y_val)).long()
		images = Variable(images)
		outputs = model(images) # [batch, 2]
		_, predicted = torch.max(outputs.data.cpu(), 1)
		total = labels.size(0)
		correct = (predicted == labels).sum()
		top1 = 100 * correct / float(total)
		model.train()

		# update best_model
		now_score = top1
		print 'best_score:%s, now_score=%s' %(str(best_score), str(now_score))
		if best_score > now_score:
			model.load(str(best_score))
			lr = lr * opt.lr_decay
			lr2= 2e-4 if lr2==0 else lr2*0.8
			optimizer = model.get_optimizer(lr,lr2,0) # update optimizer 
		
		if best_score < now_score:
			best_score = now_score
			model.save(model, scaler, str(best_score))


def main(**kwargs):

	opt.parse(kwargs,print_=True)
	
	# read name_2_features map
	features = np.load(opt.features_path)[opt.features_name]
	name = np.load(opt.features_path)['data_path']
	name_2_features = {}
	for ii in range(len(name)): name_2_features[name[ii]['pic_path']] = features[ii]

	# read datas
	X_train, y_train = read_datas(type_='train_datas', temporal_=opt.temporal_, name_2_features=name_2_features)
	X_val, y_val = read_datas(type_='eval_datas', temporal_=opt.temporal_, name_2_features=name_2_features)

	# train
	opt.in_features = np.shape(X_train[0])[-1]
	opt.num_classes = 7 # classifier for EmotiW
	model = getattr(models, opt.model)(opt)
	if opt.model in ['linearSVC','RBFSVC','RF','LR']:
		best_score = model.train(X_train, y_train, X_val, y_val)
		model.save(str(best_score))

	if opt.model in ['NN']:
		scaler = preprocessing.StandardScaler().fit(X_train)
		scaler.transform(X_train)
		scaler.transform(X_val)
		X_train = preprocessing.normalize(X_train, norm='l2')
		X_val = preprocessing.normalize(X_val, norm='l2')
		best_score, best_model = train_NN(model, X_train, y_train, X_val, y_val)
		model.save(best_model, scaler, str(best_score))

	if opt.model in ['LSTM']:
		X_train_norm = np.array([features_norm(_, 100) for _ in X_train])
		X_val_norm = np.array([features_norm(_, 100) for _ in X_val])

		batch_1, seq, feat_dim = np.shape(X_train_norm)
		X_train_norm = X_train_norm.reshape(-1, feat_dim)
		X_val_norm = X_val_norm.reshape(-1, feat_dim)
		scaler = preprocessing.StandardScaler().fit(X_train_norm)
		scaler.transform(X_train_norm)
		scaler.transform(X_val_norm)
		X_train_norm = preprocessing.normalize(X_train_norm, norm='l2')
		X_val_norm = preprocessing.normalize(X_val_norm, norm='l2')
		X_train_norm = X_train_norm.reshape(-1, seq, feat_dim)
		X_val_norm = X_val_norm.reshape(-1, seq, feat_dim)

		train_NN(model, scaler, X_train_norm, y_train, X_val_norm, y_val)


if __name__ == '__main__':
	import fire
	fire.Fire()