#coding:utf8
'''
gain fusion parameters according to eval
gain prediction for test dataset
'''
import os
import sys
import glob
import json
import time
import copy
import torch as t
import numpy as np
from gain_cnn_para import *
from config import opt

############################# step 1 #############################
def gain_best_fusion_paras_all_train_eval(best_para_path='best_para.npz', number=100):
    train_datas = np.load(opt.label_root)['train_datas']
    train_targets = [item['label'] for item in train_datas]
    files = sorted(glob.glob(opt.classifier_save_root+'/train_result/*.pth'))
    train_predictions_all = [t.load(file).numpy() for file in files]
    
    eval_datas = np.load(opt.label_root)['eval_datas']
    eval_targets = [item['label'] for item in eval_datas]
    files = sorted(glob.glob(opt.classifier_save_root+'/eval_result/*.pth'))
    eval_predictions_all = [t.load(file).numpy() for file in files]
    
    targets = np.concatenate((eval_targets, train_targets), axis=0)
    predictions_all = np.concatenate((eval_predictions_all, train_predictions_all), axis=1) # [6, 773, 7]

    modelNum = len(files)

    best_best_top1, best_best_para = 0, 0
    for time in range(number):
        best_top1, best_para = gain_cnn_para(modelNum, predictions_all, targets)

        print ('best_top1: %.4f    best_para:\n' %(best_top1))
        assert len(best_para['weight']) == len(files)
        if best_top1>best_best_top1:
            best_best_top1 = copy.copy(best_top1)
            best_best_para = copy.copy(best_para)
            np.savez_compressed('%s_%.4f.npz' %(best_para_path, best_best_top1), best_para=best_best_para)

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

############################# step 2 #############################
def gain_unlabel_submission(best_para_path='best_para.npz', save_root='submit'):
    test_datas = np.load(opt.label_root)['test_datas']
    test_targets = [item['label'] for item in test_datas]

    files = sorted(glob.glob(opt.classifier_save_root+'/test_result/*.pth'))
    val_predictions_all= [t.load(file).numpy() for file in files]
    best_para = np.load(best_para_path)['best_para'].item()
    predTest = gain_cnn_fusion(best_para, val_predictions_all)
    predTest = np.argmax(predTest, axis=1)
 
    for ii, item in enumerate(test_datas):
        save_path=os.path.join(save_root, item['pic_path']+'.txt')
        save_content = edict[predTest[ii]]
        output = open(save_path, 'w')
        output.write(save_content)
        output.close()


def main(**kwargs):

    opt.parse(kwargs,print_=True)
    if opt.fusion_type == 'gain_para':
        gain_best_fusion_paras_all_train_eval(best_para_path=opt.best_para_path, number=opt.max_epoch)
    if opt.fusion_type == 'gain_submit':
        gain_unlabel_submission(best_para_path=opt.best_para_path, save_root='submit')


if __name__=='__main__':
    import fire
    fire.Fire()