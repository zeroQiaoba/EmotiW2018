#coding:utf8
import os
import cv2
import json
import time
import tqdm
import glob
import shutil
import numpy as np

edict = {}
edict['Angry'] = 0
edict['Disgust'] = 1
edict['Fear'] = 2
edict['Happy'] = 3
edict['Sad'] = 4
edict['Surprise'] = 5
edict['Neutral'] = 6


# gather label files
def data_format_normalize(save_path='train_eval_test_EmotiW2017.npz'):
	train_datas = []
	eval_datas = []
	test_datas = []
	for path in ['Train', 'Val']:
		for sub_path in glob.glob(path+'/*'):
			for sub_sub_path in glob.glob(sub_path+'/*'): # 'Train/Happy/xxx.avi'
				type_, label_, name_ = sub_sub_path.split('/')
				name_ = name_.split('.')[0]
				item = {'pic_path': name_, 'label': edict[label_]}
				if type_ == 'Train': train_datas.append(item)
				elif type_ == 'Val': eval_datas.append(item)

	for path in ['Test']:
		for sub_path in glob.glob(path+'/*'): # 'Test/xxx.avi'
			_, name_ = sub_path.split('/')
			name_ = name_.split('.')[0]
			item = {'pic_path': name_, 'label': -1}
			test_datas.append(item)

	np.savez_compressed(save_path,
						train_datas=train_datas,
						eval_datas=eval_datas,
						test_datas=test_datas,
						)

# write a.txt files: (video_name, label) in each line
# a.txt is for C3D features extraction
def change_name_2_txt_for_face_extract(path = 'train_eval_test_EmotiW2017.npz'):
	train_datas = np.load(path)['train_datas']
	eval_datas = np.load(path)['eval_datas']
	test_datas = np.load(path)['test_datas']

	datas = [train_datas, eval_datas, test_datas]
	datas = np.concatenate(datas, axis=0)

	output = open('a.txt', 'w')
	for item in datas:
		output.write('%s %s\n' %(item['pic_path'], item['label']))
	output.close()


# gain xxx_bottleneck_Faces.npz
# this file is for frame-level feature extraction (like DenseNet, VGG, Inception)
def change_name_2_bottleneck_for_feature_extract_Faces(path = 'train_eval_test_EmotiW2017.npz'):
	name = path.split('.')[0]
	save_path = '%s_bottleneck_Faces.npz' %(name)

	train_datas = np.load(path)['train_datas']
	eval_datas = np.load(path)['eval_datas']
	test_datas = np.load(path)['test_datas']

	datas = [train_datas, eval_datas, test_datas]
	datas = np.concatenate(datas, axis=0)

	result_datas = []
	for item in datas:
		pic_path = item['pic_path']
		for frame_path in glob.glob('Faces/'+pic_path+'/*'):
			frame_path = "/".join(frame_path.split('/')[-2:])
			result_datas.append({'pic_path': frame_path, 'label':-1})

	np.savez_compressed(save_path,
						train_datas=result_datas,
						eval_datas=result_datas,
						test_datas=result_datas,
						)

# gain xxx_bottleneck_Origin_Faces.npz
# this file is for frame-level feature extraction for Origin Frames (especially for Inception)
def change_name_2_bottleneck_for_feature_extract_Origin_Faces(path = 'train_eval_test_EmotiW2017.npz'):
	name = path.split('.')[0]
	save_path = '%s_bottleneck_Origin_Faces.npz' %(name)

	train_datas = np.load(path)['train_datas']
	eval_datas = np.load(path)['eval_datas']
	test_datas = np.load(path)['test_datas']

	datas = [train_datas, eval_datas, test_datas]
	datas = np.concatenate(datas, axis=0)

	result_datas = []
	for item in datas:
		pic_path = item['pic_path']
		for frame_path in glob.glob('Origin_Faces/'+pic_path+'/*'):
			frame_path = "/".join(frame_path.split('/')[-2:])
			result_datas.append({'pic_path': frame_path, 'label':-1})

	np.savez_compressed(save_path,
						train_datas=result_datas,
						eval_datas=result_datas,
						test_datas=result_datas,
						)

if __name__ == '__main__':
	import fire
	fire.Fire()