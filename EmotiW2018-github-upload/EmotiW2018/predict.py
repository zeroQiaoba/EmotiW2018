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

import torch as t
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
		if temporal_.find('Video')!=-1:
			if pic_path not in name_2_features: continue # for LBPTOP
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

	return datas, labels, train_datas

# name_2_features: must be video_name->features
def change_to_normal_format(type_, name_2_features, preds):
	i = 0
	results = []
	train_datas = np.load(opt.label_root)[type_]
	for item in train_datas:
		pic_path = item['pic_path'] # video name
		pic_label = item['label']

		if pic_path not in name_2_features:
			results.append(np.ones((7,))/7.0)
		else: 
			results.append(preds[i])
			i += 1
	assert i == len(preds)
	return np.array(results)


def predict_and_save(model, X_val, y_val, name_label_val, save_path):
	results = []
	y_val_pred=model.test(X_val)
	y_val_pred=np.argmax(y_val_pred,axis=1)
	print(np.sum(y_val_pred==y_val)/float(len(y_val)))
	for ii, item in enumerate(name_label_val):
		pic_path = item['pic_path']
		pic_label = item['label']
		pir_pred = y_val_pred[ii]
		assert pic_label == y_val[ii]
		new_item = {'pic_path':pic_path, 'pic_label':pic_label, 'pic_pred':pir_pred}
		results.append(new_item)
	#with open(save_path, 'w') as jsonFile: jsonFile.write(json.dumps(results))


def main(**kwargs):

	opt.parse(kwargs,print_=True)

	# read name_2_features map [exist: name->[-1]]
	features = np.load(opt.features_path)[opt.features_name]
	name = np.load(opt.features_path)['data_path']
	name_2_features = {}
	for ii in range(len(name)): name_2_features[name[ii]['pic_path']] = features[ii]

	# read datas
	X_train, y_train, name_label_train = read_datas(type_='train_datas', temporal_=opt.temporal_, name_2_features=name_2_features)
	X_val, y_val, name_label_val = read_datas(type_='eval_datas', temporal_=opt.temporal_, name_2_features=name_2_features)
	X_test, y_test, name_label_test = read_datas(type_='test_datas', temporal_=opt.temporal_, name_2_features=name_2_features)
	
	# test
	opt.in_features = np.shape(X_train[0])[-1]
	opt.num_classes = 7 # classifier for EmotiW
	model = getattr(models, opt.model)(opt)
	model.load(opt.model_path)

	# predict
	y_train_pred=np.array(model.test(X_train))
	y_val_pred=np.array(model.test(X_val))
	y_test_pred=np.array(model.test(X_test))
	print(np.sum(np.argmax(y_val_pred,axis=1)==y_val)/float(len(y_val)))

	# add absent result
	if opt.temporal_.find('Video')!=-1:
		y_train_pred = change_to_normal_format(type_='train_datas', name_2_features=name_2_features, preds=y_train_pred)
		y_val_pred = change_to_normal_format(type_='eval_datas', name_2_features=name_2_features, preds=y_val_pred)
		y_test_pred = change_to_normal_format(type_='test_datas', name_2_features=name_2_features, preds=y_test_pred)

	# save results
	t.save(t.from_numpy(y_train_pred).float(), opt.classifier_save_root + '/train_result/%s.pth' %(os.path.basename(opt.model_path)))
	t.save(t.from_numpy(y_val_pred).float(), opt.classifier_save_root + '/eval_result/%s.pth' %(os.path.basename(opt.model_path)))
	t.save(t.from_numpy(y_test_pred).float(), opt.classifier_save_root + '/test_result/%s.pth' %(os.path.basename(opt.model_path)))



def main_all_classifiers(**kwargs):

	opt.parse(kwargs,print_=True)

	for model_path in glob.glob(opt.classifier_save_root+'/*'):
		if not os.path.isfile(model_path): continue
		if os.path.isfile(opt.classifier_save_root + '/train_result/%s.pth' %(os.path.basename(model_path))): continue
		opt.model_path = model_path
		opt.parse(kwargs,print_=True)

		# read name_2_features map
		features = np.load(opt.features_path)[opt.features_name]
		name = np.load(opt.features_path)['data_path']
		name_2_features = {}
		for ii in range(len(name)): name_2_features[name[ii]['pic_path']] = features[ii]

		# read datas
		X_train, y_train, name_label_train = read_datas(type_='train_datas', temporal_=opt.temporal_, name_2_features=name_2_features)
		X_val, y_val, name_label_val = read_datas(type_='eval_datas', temporal_=opt.temporal_, name_2_features=name_2_features)
		X_test, y_test, name_label_test = read_datas(type_='test_datas', temporal_=opt.temporal_, name_2_features=name_2_features)
		
		# test
		opt.in_features = np.shape(X_train[0])[-1]
		opt.num_classes = 7 # classifier for EmotiW
		model = getattr(models, opt.model)(opt)
		model.load(opt.model_path)

		# predict
		y_train_pred=np.array(model.test(X_train))
		y_val_pred=np.array(model.test(X_val))
		y_test_pred=np.array(model.test(X_test))
		print(np.sum(np.argmax(y_val_pred,axis=1)==y_val)/float(len(y_val)))

		# add absent result
		if opt.temporal_.find('Video')!=-1:
			y_train_pred = change_to_normal_format(type_='train_datas', name_2_features=name_2_features, preds=y_train_pred)
			y_val_pred = change_to_normal_format(type_='eval_datas', name_2_features=name_2_features, preds=y_val_pred)
			y_test_pred = change_to_normal_format(type_='test_datas', name_2_features=name_2_features, preds=y_test_pred)

		# save results
		t.save(t.from_numpy(y_train_pred).float(), opt.classifier_save_root + '/train_result/%s.pth' %(os.path.basename(opt.model_path)))
		t.save(t.from_numpy(y_val_pred).float(), opt.classifier_save_root + '/eval_result/%s.pth' %(os.path.basename(opt.model_path)))
		t.save(t.from_numpy(y_test_pred).float(), opt.classifier_save_root + '/test_result/%s.pth' %(os.path.basename(opt.model_path)))


if __name__ == '__main__':
	import fire
	fire.Fire()