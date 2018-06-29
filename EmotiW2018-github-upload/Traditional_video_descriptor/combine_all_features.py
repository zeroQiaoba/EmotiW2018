# -*- coding: utf-8 -*-
import os
import cv2
import sys
import math
import glob
import json
import time
import tqdm
import shutil
import numpy as np

name_2_LBP = {}
LBP = np.load('LBP_features.npz')['LBP']
data_path = np.load('LBP_features.npz')['data_path']
for ii, item in enumerate(data_path):
    pic_path = item['pic_path']
    features = LBP[ii]
    name_2_LBP[pic_path] = features


name_2_HOGLBP = {}
LBP = np.load('HogLBP_features.npz')['HOGLBP']
data_path = np.load('HogLBP_features.npz')['data_path']
for ii, item in enumerate(data_path):
    pic_path = item['pic_path']
    features = LBP[ii]
    name_2_HOGLBP[pic_path] = features

name_2_HOG = {}
LBP = np.load('Hog_features.npz')['HOG']
data_path = np.load('Hog_features.npz')['data_path']
for ii, item in enumerate(data_path):
    pic_path = item['pic_path']
    features = LBP[ii]
    name_2_HOG[pic_path] = features

name_2_DSIFT = {}
LBP = np.load('Dsift_features.npz')['DSIFT']
data_path = np.load('Dsift_features.npz')['data_path']
for ii, item in enumerate(data_path):
    pic_path = item['pic_path']
    features = LBP[ii]
    name_2_DSIFT[pic_path] = features

data_path = []
features_1 = []
features_2 = []
features_3 = []
features_4 = []
for name in name_2_DSIFT:
    data_path.append({'pic_path': name})
    features_1.append(name_2_LBP[name])
    features_2.append(name_2_HOGLBP[name])
    features_3.append(name_2_HOG[name])
    features_4.append(name_2_DSIFT[name])

np.savez_compressed('./EmotiW_traditional_descriptor.npz',
                    data_path=data_path,
                    LBP=features_1,
                    HOGLBP=features_2,
                    HOG=features_3,
                    DSIFT=features_4,
                    )

