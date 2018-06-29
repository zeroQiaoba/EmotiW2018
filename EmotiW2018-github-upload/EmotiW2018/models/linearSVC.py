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

import torch 
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from .BasicModule import BasicModule 

class linearSVC(BasicModule):
	def __init__(self, opt):
		super(linearSVC, self).__init__()
		self.model_name = 'linearSVC'
		self.opt=opt
		self.features_path = opt.features_path.split('/')[-1].split('.')[0] # EmotiW_DenseNet_SFEW_Fintuning
		self.features_name = self.features_path + '_' + opt.features_name # EmotiW_DenseNet_SFEW_Fintuning_features_1
		self.temporal_ = opt.temporal_
		self.classifier_save_root = opt.classifier_save_root
		self.scaler=""
		self.best_model=""

	# Train the Model: linearSVC
	def train(self, X_train, y_train, X_val, y_val):
		# feature normlization
		self.scaler = preprocessing.StandardScaler().fit(X_train)
		self.scaler.transform(X_train)
		self.scaler.transform(X_val)
		X_train = preprocessing.normalize(X_train, norm='l2')
		X_val = preprocessing.normalize(X_val, norm='l2')

		best_score = 0
		for epoch in range(self.opt.max_epoch):
			for C_value in range(-5,16,1):
				linear_svm = SVC(C = pow(2, C_value), probability = True, kernel = 'linear')
				linear_svm.fit(X_train, y_train)
				y_val_pred=np.array(linear_svm.predict_proba(X_val))
				now_score = np.sum(np.argmax(y_val_pred,axis=1)==y_val)/float(len(y_val))
				print 'best_score:%s,   now_score=%s' %(str(best_score), str(now_score))
				if best_score < now_score:
					best_score =  now_score
					self.best_model=linear_svm
		return best_score
		
	def test(self, datas):
		self.scaler.transform(datas)
		datas = preprocessing.normalize(datas, norm='l2')
		probs = self.model.predict_proba(datas)
		return probs

	def save(self, name=None):
		prefix = self.classifier_save_root + '/' + self.features_name + '_' + self.temporal_ + '_' + self.model_name + '_'
		if name is None:
			name = time.strftime('%m%d_%H:%M:%S.pth')
		path = prefix+name
		data = {'best_model':self.best_model,'scaler':self.scaler}
		torch.save(data, path)


	def load(self, path):
		self.model = torch.load(path)['best_model']
		self.scaler = torch.load(path)['scaler']
		return self
		