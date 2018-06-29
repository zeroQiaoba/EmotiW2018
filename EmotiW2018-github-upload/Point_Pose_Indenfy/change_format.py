#coding:utf8
import os
import sys
import glob
import json
import time
import tqdm
import shutil
import torch as t
import numpy as np

edict = {}
edict[0] = 'Angry'
edict[1] = 'Disgust'
edict[2] = 'Fear'
edict[3] = 'Happy'
edict[4] = 'Sad'
edict[5] = 'Surprise'
edict[6] = 'Neutral'

edict_r = {}
edict_r['Angry'] = 0
edict_r['Disgust'] = 1
edict_r['Fear'] = 2
edict_r['Happy'] = 3
edict_r['Sad'] = 4
edict_r['Surprise'] = 5
edict_r['Neutral'] = 6

def change(features_root):
    name_2_features = {}
    for features_path in tqdm.tqdm(glob.glob(features_root + '/*')):
        video_name = os.path.basename(features_path).split('.')[0]

        # read features
        feature = []
        f = open(features_path)
        for line in f.readlines():
            feature_list = line.strip().split(' ')
            feature_list = np.array([float(_) for _ in feature_list])
            feature.append(feature_list)

        if len(feature)!=0: name_2_features[video_name] = feature
    return name_2_features

def add_dev(name_2_features):
    name_2_features_new = {}
    for name in name_2_features:
        new_features = []
        features = name_2_features[name]
        for i in range(len(features)):
            dev_features = features[i+1]-features[i] if i < len(features)-2 else features[i]
            new_features.append(np.concatenate([features[i], dev_features], axis=0))

        name_2_features_new[name] = new_features
    return name_2_features_new



name_2_pose = change('Point_Features')
name_2_pose_new = add_dev(name_2_pose)
name_2_indenty = change('Face_embedding')

poses = []
poses_new = []
data_path = []
for name in name_2_pose:
    poses.append(name_2_pose[name])
    poses_new.append(name_2_pose_new[name])
    data_path.append({'pic_path': name})
np.savez_compressed('EmotiW_Pose.npz',
                    data_path=data_path,
                    poses=poses,
                    posesDev=poses_new)


identifies = []
data_path = []
for name in name_2_indenty:
    identifies.append(name_2_indenty[name])
    data_path.append({'pic_path': name})
np.savez_compressed('EmotiW_Identify.npz',
                    data_path=data_path,
                    identifies=identifies)


