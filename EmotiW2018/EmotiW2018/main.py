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

from config import opt
from data import *
import models

# target: test model on eval
def val_top1(model, dataset):
    assert opt.loss != 'mseloss'
    dataset.change_type(type_='eval')
    model.eval()

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=opt.batch_size, 
                                         shuffle=False,
                                         num_workers=opt.num_workers, 
                                         pin_memory=opt.pin_memory) 
    # gain top1 on the eval
    correct, total = 0, 0
    for ii, (datas, labels, _) in tqdm.tqdm(enumerate(loader)):
        labels=labels.long()
        datas = Variable(datas)
        if opt.cuda != -1: datas = datas.cuda(opt.cuda)
        outputs = model(datas) # [batch, 2]
        _, predicted = torch.max(outputs.data.cpu(), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    top1 = 100 * correct / float(total)
    
    model.train()
    dataset.change_type(type_='train')
    return top1


# target: test model on eval
def val_mse(model, dataset):
    assert opt.loss == 'mseloss'
    dataset.change_type(type_='eval')
    model.eval()

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=opt.batch_size, 
                                         shuffle=False,
                                         num_workers=opt.num_workers, 
                                         pin_memory=opt.pin_memory) 
    # gain total loss on the eval
    loss_ = 0
    loss_function = torch.nn.MSELoss(size_average=False) # 设置False,使得不取平均的计算mse
    for ii, (images, labels, _) in tqdm.tqdm(enumerate(loader)):
        labels=labels.float()
        images, labels = Variable(images), Variable(labels)
        if opt.cuda != -1: images,labels = images.cuda(opt.cuda),labels.cuda(opt.cuda)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss_ += loss.cpu().data[0]

    model.train()
    dataset.change_type(type_='train')
    return loss_

### train image-level emotion classifier ####
# python main.py main --model='DenseNet' --plot_every=100 --batch-size=128  --lr=0.001 --lr2=0 --lr_decay=0.5  --decay-every=234 --max_epoch=5 --cuda=-1

### fintuning ###
# python main.py main --model='DenseNet' --model_path='xxx.pkl' --train_eval_test_path='' --pic_root='' --plot_every=100 --batch-size=128  --lr=0.001 --lr2=0 --lr_decay=0.5  --decay-every=330 --max_epoch=5 --cuda=-1
def main(**kwargs):
    
    # load model
    opt.parse(kwargs,print_=False)
    model = getattr(models, opt.model)(opt)
    if opt.model_path: model.load(opt.model_path)
    opt.parse(kwargs,print_=True)
    model.classifier = nn.Linear(model.classifier.in_features, opt.num_classes) # change last layer and loss in fintuning
    print(model)
    if opt.cuda != -1: model = model.cuda(opt.cuda)

    # load params
    lr,lr2=opt.lr,opt.lr2
    loss_function = getattr(models, opt.loss)() 
    optimizer = model.get_optimizer(lr, lr2, opt.weight_decay)
    best_score = -float('inf')
    best_path = ""
    
    # load data
    train_datas = np.load(opt.train_eval_test_path)['train_datas']
    eval_datas = np.load(opt.train_eval_test_path)['eval_datas']
    test_datas = np.load(opt.train_eval_test_path)['test_datas']
    dataset = SogouImageData(opt.pic_root, train_datas, eval_datas, test_datas, type_='train')
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=opt.shuffle,
                                         num_workers=opt.num_workers, 
                                         pin_memory=opt.pin_memory) 

    # Train the Model
    for epoch in range(opt.max_epoch):
        for ii, (images, labels, _) in tqdm.tqdm(enumerate(loader)):
            labels = labels.long() if opt.loss!='mseloss' else labels.float()
            images, labels = Variable(images), Variable(labels)
            if opt.cuda != -1: images,labels = images.cuda(opt.cuda),labels.cuda(opt.cuda)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, opt.weight*labels)
            loss.backward()
            optimizer.step()

            # Test the Model
            if ii%opt.plot_every==opt.plot_every-1:
                ### visualize one batch result every opt.plot_every times
                if opt.loss=='mseloss':
                    print ('loss:%.2f' %(loss.cpu().data[0]))
                else:
                    _, predicted = torch.max(outputs.data.cpu(), 1)
                    correct = (predicted == labels.data.cpu()).sum()
                    top1 = 100 * correct / float(labels.size(0))
                    print ('top1:%.2f' %(top1))

            if ii%opt.decay_every == opt.decay_every-1:
                # save model and adjust learning rate each opt.decay_every times
                scores=0
                if opt.loss=='mseloss':
                    scores = -val_mse(model, dataset)# 加个负号，使得scores越大越好
                else:
                    scores = val_top1(model, dataset)
                
                print ('scores:%.2f   best_score:%.2f' %(scores, best_score))
                if scores>best_score:
                    best_score = scores
                    if opt.model_path!=None and opt.loss=='mseloss':
                        best_path = model.save(name=str(scores)+'_fintuning_AVEC',new=True)
                    elif opt.model_path!=None and opt.loss!='mseloss':
                        best_path = model.save(name=str(scores)+'_fintuning_SFEW',new=True)
                    elif opt.model_path==None:
                        best_path = model.save(name=str(scores),new=True)

                if scores < best_score:
                    model.load(best_path,change_opt=False) # return back to the best model
                    if opt.cuda != None: model = model.cuda(opt.cuda)
                    lr = lr * opt.lr_decay
                    lr2= 2e-4 if lr2==0 else  lr2*0.8
                    optimizer = model.get_optimizer(lr,lr2,0) # update optimizer   


if __name__ == '__main__':
	import fire
	fire.Fire()