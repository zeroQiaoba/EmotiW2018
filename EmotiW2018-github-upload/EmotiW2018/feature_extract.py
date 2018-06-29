#coding:utf8
import os
import cv2
import json
import time
import tqdm
import glob
import shutil
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from config import opt
from data import *
import models

# when forward, save values
features_blobs = []
def hook_feature(module, input, output): # input and output are all variables
	features_blobs.append(np.squeeze(output).data.cpu()) # Varibale->Tensor


# nohup python -u feature_extract.py main --model='DenseNet' --type_='eval' --save_='eval_features_dense.npz'  --model_path='checkpoints/DenseNet_80.9635722679' --cuda=-1
def main_DenseNet(**kwargs):
	# load model
	opt.parse(kwargs)
	model = getattr(models,opt.model)(opt)
	model.classifier = nn.Linear(model.classifier.in_features, opt.num_classes)
	if opt.model_path is not None: model.load(opt.model_path)
	print(model)
	opt.parse(kwargs)
	if opt.cuda != -1: model = model.cuda(opt.cuda)
	model = model.eval()

	# hook the feature extractor
	features_names = ['features', 'classifier'] # this is the last conv layer of the densenet
	for name in features_names:
		model._modules.get(name).register_forward_hook(hook_feature)

	# load datas
	train_datas = np.load(opt.train_eval_test_path)['train_datas']
	eval_datas = np.load(opt.train_eval_test_path)['eval_datas']
	test_datas = np.load(opt.train_eval_test_path)['test_datas']
	dataset = SogouImageData(opt.pic_root, train_datas, eval_datas, test_datas, type_='train')
	dataset.change_type(type_=opt.type_)
	loader = torch.utils.data.DataLoader(dataset=dataset,
										 batch_size=opt.batch_size,
										 shuffle=False, # 不打乱数据顺序
										 num_workers=0, 
										 pin_memory=opt.pin_memory) 
	features_1 = []
	features_2 = []
	features_3 = []
	features_4 = []
	for ii, (datas, labels, _) in tqdm.tqdm(enumerate(loader)):
		datas = Variable(datas) # [batch,3,64,64]
		if opt.cuda != -1: datas = datas.cuda(opt.cuda)
		outputs = model(datas) # [batch,2]
		features = features_blobs[-2] # the last append features
		features = Variable(features)
		features_1.append(features.view(features.size(0), -1).data.cpu().numpy()) # [batch, 4096], array
		features = F.relu(features) # [batch, 1024,2,2]
		features_2.append(features.view(features.size(0), -1).data.cpu().numpy())
		features = F.avg_pool2d(features, kernel_size=2).view(features.size(0), -1) # [batch, 1024]
		features_3.append(features.data.cpu().numpy())
		features_4.append(features_blobs[-1].cpu().numpy())

	features_1 = np.concatenate(features_1, axis=0)
	features_2 = np.concatenate(features_2, axis=0)
	features_3 = np.concatenate(features_3, axis=0)
	features_4 = np.concatenate(features_4, axis=0)	

	np.savez_compressed(opt.save_,
						data_path=test_datas,
						features_1=features_1,
						features_2=features_2,
						features_3=features_3,
						features_4=features_4,
						)


def main_VGG(**kwargs):
	# load model
	opt.parse(kwargs)
	model = getattr(models,opt.model)(opt)
	model.classifier = nn.Linear(model.classifier.in_features, opt.num_classes)
	if opt.model_path is not None: model.load(opt.model_path)
	print(model)
	opt.parse(kwargs)
	if opt.cuda != -1: model = model.cuda(opt.cuda)
	model = model.eval()

	# hook the feature extractor
	features_names = ['features.37', 'features.38', 'features.40', 'features.41', 'features', 'fc6', 'relu6', 'fc7', 'relu7']
	for name in features_names:
		if name.find('.')!=-1:
			name1, name2 = name.split('.')
			name2 = int(name2)
			model._modules.get(name1)[name2].register_forward_hook(hook_feature)
		else:
			model._modules.get(name).register_forward_hook(hook_feature)


	# load datas
	train_datas = np.load(opt.train_eval_test_path)['train_datas']
	eval_datas = np.load(opt.train_eval_test_path)['eval_datas']
	test_datas = np.load(opt.train_eval_test_path)['test_datas']
	dataset = SogouImageData(opt.pic_root, train_datas, eval_datas, test_datas, type_='train')
	dataset.change_type(type_=opt.type_)
	loader = torch.utils.data.DataLoader(dataset=dataset,
										 batch_size=opt.batch_size,
										 shuffle=False, # 不打乱数据顺序
										 num_workers=0, 
										 pin_memory=opt.pin_memory) 
	features_1 = []
	features_2 = []
	features_3 = []
	features_4 = []
	features_5 = []
	features_6 = []
	features_7 = []
	features_8 = []
	features_9 = []
	for ii, (datas, labels, _) in tqdm.tqdm(enumerate(loader)):
		datas = Variable(datas) # [batch,3,64,64]
		if opt.cuda != -1: datas = datas.cuda(opt.cuda)
		outputs = model(datas) # [batch,2]
		features = features_blobs[-9] # tensor
		features = Variable(features)
		features_1.append(features.view(features.size(0), -1).data.cpu().numpy()) # [batch, 4096], array

		features = features_blobs[-8] # tensor
		features = Variable(features)
		features_2.append(features.view(features.size(0), -1).data.cpu().numpy()) # [batch, 4096], array

		features = features_blobs[-7] # tensor
		features = Variable(features)
		features_3.append(features.view(features.size(0), -1).data.cpu().numpy()) # [batch, 4096], array

		features = features_blobs[-6] # tensor
		features = Variable(features)
		features_4.append(features.view(features.size(0), -1).data.cpu().numpy()) # [batch, 4096], array

		features = features_blobs[-5] # tensor
		features = Variable(features)
		features_5.append(features.view(features.size(0), -1).data.cpu().numpy()) # [batch, 4096], array

		features_6.append(features_blobs[-4].cpu().numpy())
		features_7.append(features_blobs[-3].cpu().numpy())
		features_8.append(features_blobs[-2].cpu().numpy())
		features_9.append(features_blobs[-1].cpu().numpy())


	features_1 = np.concatenate(features_1, axis=0)
	features_2 = np.concatenate(features_2, axis=0)
	features_3 = np.concatenate(features_3, axis=0)
	features_4 = np.concatenate(features_4, axis=0)	
	features_5 = np.concatenate(features_5, axis=0)
	features_6 = np.concatenate(features_6, axis=0)
	features_7 = np.concatenate(features_7, axis=0)
	features_8 = np.concatenate(features_8, axis=0)	
	features_9 = np.concatenate(features_9, axis=0)	

	np.savez_compressed(opt.save_,
						data_path=test_datas,
						features_1=features_1,
						features_2=features_2,
						features_3=features_3,
						features_4=features_4,
						features_5=features_5,
						features_6=features_6,
						features_7=features_7,
						features_8=features_8,
						features_9=features_9,
						)

if __name__ == '__main__':
	import fire
	fire.Fire()
