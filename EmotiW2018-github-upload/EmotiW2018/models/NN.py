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
from sklearn.linear_model import LogisticRegression

from .BasicModule import BasicModule 

class NN(BasicModule):
	def __init__(self, opt):
		super(NN, self).__init__()
		self.model_name = 'NN'
		self.opt=opt
		self.features_path = opt.features_path.split('/')[-1].split('.')[0] # EmotiW_DenseNet_SFEW_Fintuning
		self.features_name = self.features_path + '_' + opt.features_name # EmotiW_DenseNet_SFEW_Fintuning_features_1
		self.temporal_ = opt.temporal_
		self.classifier_save_root = opt.classifier_save_root
		self.scaler=""
		self.best_model=""

		## model structure
		self.fc = nn.Linear(opt.in_features, 128)
		self.relu = nn.ReLU(inplace=True)
		self.classifier = nn.Linear(128, opt.num_classes)

	def forward(self, x):
		x = self.fc(x)
		x = self.relu(x)
		x = self.classifier(x)
		return x # [batch, class_num]

	def test(self, datas):
		self.scaler.transform(datas)
		datas = preprocessing.normalize(datas, norm='l2')
		datas = torch.from_numpy(datas).float()
		datas = Variable(datas)
		probs = self.model(datas).data.cpu().numpy()
		return probs

	def save(self, best_model, scaler, name=None):
		prefix = self.classifier_save_root + '/' + self.features_name + '_' + self.temporal_ + '_' + self.model_name + '_'
		if name is None:
			name = time.strftime('%m%d_%H:%M:%S.pth')
		path = prefix+name
		data = {'best_model':best_model,'scaler':scaler}
		torch.save(data, path)

	def load(self, path):
		self.model = torch.load(path)['best_model']
		self.scaler = torch.load(path)['scaler']
		return self
		