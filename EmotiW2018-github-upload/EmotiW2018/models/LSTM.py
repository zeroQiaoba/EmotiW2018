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

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class LSTM(BasicModule):
	def __init__(self, opt):
		super(LSTM, self).__init__()
		self.model_name = 'LSTM'
		self.opt=opt
		self.features_path = opt.features_path.split('/')[-1].split('.')[0] # EmotiW_DenseNet_SFEW_Fintuning
		self.features_name = self.features_path + '_' + opt.features_name # EmotiW_DenseNet_SFEW_Fintuning_features_1
		self.temporal_ = opt.temporal_
		self.classifier_save_root = opt.classifier_save_root
		self.prefix = self.classifier_save_root + '/' + self.features_name + '_' + self.temporal_ + '_' + self.model_name + '_'
		self.scaler=""
		self.best_model=""

		self.encoder = nn.LSTM(
			input_size=opt.in_features, #4096
			hidden_size=opt.hidden_size,
			num_layers=1, #2
			bidirectional=False,
			batch_first=False,
			dropout=0.5
		)

		 # self.dropout = nn.Dropout()
		self.fc = nn.Sequential(
			nn.Linear(opt.kmax_pooling*(opt.hidden_size),opt.linear_hidden_size),
			nn.BatchNorm1d(opt.linear_hidden_size),
			nn.ReLU(inplace=True),
			nn.Linear(opt.linear_hidden_size,opt.num_classes)
		)

	
	def forward(self, x): # x: [batch, seq, features]
		title_out = self.encoder(x.permute(1,0,2))[0].permute(1,2,0) 
		title_conv_out = kmax_pooling((title_out),2,self.opt.kmax_pooling)
		conv_out = title_conv_out
		reshaped = conv_out.view(conv_out.size(0), -1)
		logits = self.fc((reshaped))
		return logits # [batch, class_num]

	def features_norm(self, datas, len_norm):
		if not isinstance(datas,list): datas = datas.tolist()
		N, D = np.shape(datas)
		if N>len_norm:
			return datas[:len_norm]
		else:
			while np.shape(datas)[0]!= len_norm: datas.append(datas[-1])
		return datas

	def test(self, datas):
		self.model.eval()
		datas = np.array([self.features_norm(_, 100) for _ in datas])
		batch_1, seq, feat_dim = np.shape(datas)
		datas = datas.reshape(-1, feat_dim)
		self.scaler.transform(datas)
		datas = preprocessing.normalize(datas, norm='l2')
		datas = datas.reshape(-1, seq, feat_dim)
		datas = torch.from_numpy(datas).float()
		datas = Variable(datas)
		probs = self.model(datas).data.cpu().numpy()
		return probs

	def save(self, model, scaler, name=None):
		prefix = self.prefix
		if name is None:
			name = time.strftime('%m%d_%H:%M:%S.pth')
		path = prefix+name
		data = {'best_model':model,'scaler':scaler}
		torch.save(data, path)

	def load(self, path):
		if path.find(self.prefix) == -1: path = self.prefix+path
		data = torch.load(path)
		self.model = data['best_model']
		self.scaler = data['scaler']
		return self
		