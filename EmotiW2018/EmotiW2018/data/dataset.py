## Gain loader
## Date: 2017_9_27
## Author: Lian Zheng

import os
import json
import glob
import numpy as np
from PIL import Image
from img_process import multiType_image_loader
import torch 
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

train_transform = transforms.Compose([transforms.Scale(70),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
                                                            

eval_transform = transforms.Compose([transforms.Scale(70),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Scale(70),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])   

## dataloader
class SogouImageData(torch.utils.data.Dataset):
    def __init__(self, root, train_datas, eval_datas, test_datas, type_='train'):
        self.root=root
        self.train_datas = train_datas
        self.eval_datas = eval_datas
        self.test_datas = test_datas

        self.type_=type_ # 'train', 'eval', 'test'
        self.datas_ = self.train_datas
        self.len_ = len(self.datas_)
        self.transform = train_transform

    # update type_
    def change_type(self, type_='test'):
        if type_ == 'train':
            self.datas_ = self.train_datas
            self.transform = train_transform
        elif type_ == 'eval':
            self.datas_ = self.eval_datas
            self.transform = eval_transform
        elif type_ == 'test':
            self.datas_ = self.test_datas
            self.transform = test_transform
        self.type_=type_
        self.len_ = len(self.datas_)
        return self

    def __getitem__(self, index):

        item = self.datas_[index]
        pic_path = os.path.join(self.root, item['pic_path'])
        label = item['label']

        res, img = multiType_image_loader(pic_path)
        if res == 0:
            print('%s is not a good image path, please filter your images' %(pic_path))

        if self.transform is not None: img = self.transform(img)

        return img, float(label), pic_path

    def __len__(self):
        return self.len_

