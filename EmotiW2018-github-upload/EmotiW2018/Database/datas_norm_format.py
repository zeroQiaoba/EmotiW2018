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

def data_format_normalize():
    train_label_path = 'trainLabel.txt'
    test_label_path = 'testLabel.txt'
    train_datas = []
    f = open(train_label_path)
    for line in f.readlines():
        pic_name, pic_label = line.strip().split(' ')
        pic_path = os.path.join('FER2013Train', pic_name)
        train_datas.append({'pic_path': pic_path, 'label': int(pic_label)})

    test_datas = []
    f = open(test_label_path)
    for line in f.readlines():
        pic_name, pic_label = line.strip().split(' ')
        pic_path = os.path.join('FER2013Test', pic_name)
        test_datas.append({'pic_path': pic_path, 'label': int(pic_label)})

    np.savez_compressed('train_eval_test_2.npz',
                        train_datas=train_datas,
                        eval_datas=test_datas,
                        test_datas=test_datas
                        )

if __name__ == '__main__':          
    import fire
    fire.Fire()

