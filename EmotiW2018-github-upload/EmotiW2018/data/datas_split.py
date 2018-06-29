## target: split datas into train, eval and test
#coding:utf8
import os
import cv2
import glob
import json
import time
import shutil
import argparse
import numpy as np

# sell: 1; no_sell: 0
# ('pic_path', 'label')
# train: all label_1/sell and equal number label_1/no_sell
# test: other label_1/no_sell
def split_datas_to_train_test(pic_root_path):
    train_datas = [] # all sell data and partial no_sell datas
    test_datas = [] # other no_sell datas
    for pic_path in glob.glob(pic_root_path + '/label_1/sell/*'):
        item = {'pic_path': pic_path, 'label': 1}
        train_datas.append(item)

    sell_num = len(train_datas)
    non_sell_num = sell_num
    for ii, pic_path in enumerate(glob.glob(pic_root_path + '/label_1/no_sell/*')):
        item = {'pic_path': pic_path, 'label': 0}
        train_datas.append(item) if ii < non_sell_num else test_datas.append(item)
    return train_datas, test_datas


# train_datas: 0.9 train
# eval_datas: 0.1 train
def split_train_to_train_eval(datas, split_rate = 0.1):
    # save input data and output data
    train_datas = []
    eval_datas = []

    # split 0 and 1 data for training
    label_1_item = [item for item in datas if item['label'] == 1]
    label_0_item = [item for item in datas if item['label'] == 0]

    # gain output data and input data
    index = np.arange(len(label_1_item))
    np.random.shuffle(index)
    for i in range(len(index)):
        item = label_1_item[index[i]]
        eval_datas.append(item) if i < int(split_rate*len(index)) else train_datas.append(item)

    index = np.arange(len(label_0_item))
    np.random.shuffle(index)
    for i in range(len(index)):
        item = label_0_item[index[i]]
        eval_datas.append(item) if i < int(split_rate*len(index)) else train_datas.append(item)

    return train_datas, eval_datas