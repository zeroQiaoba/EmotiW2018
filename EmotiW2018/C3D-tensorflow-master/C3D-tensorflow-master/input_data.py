from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import glob

def get_frames_data(pic_root, filename, num_frames_per_clip=16):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  result_all = []
  s_index = 0
  filenames = glob.glob('%s/%s/*' %(pic_root, filename)) # 'xx/xx/Faces/001742240/*'
  filenames = [_ for _ in filenames if _[-3:]!='txt']
  filenames = sorted(filenames)
  if(len(filenames)>num_frames_per_clip):
    for s_index in range(0, len(filenames) - num_frames_per_clip, int(num_frames_per_clip/2)):
      ret_arr = []
      for i in range(s_index, s_index + num_frames_per_clip):
        image_name = filenames[i]
        img = Image.open(image_name)
        img_data = np.array(img)
        ret_arr.append(img_data)
      result_all.append(ret_arr)
  else:
    print (filename)
    s_index = 0
    ret_arr = []
    for i in range(len(filenames)):
      image_name = filenames[i]
      img = Image.open(image_name)
      img_data = np.array(img)
      ret_arr.append(img_data)
    while len(ret_arr)!= num_frames_per_clip: ret_arr.append(ret_arr[-1])
    result_all.append(ret_arr)

  return result_all # [6, 16, imgs]


def read_all_frames_data(pic_root, filename, num_frames_per_clip=16):
  # tmp_data_all: [N, len, 16, imgs] tmp_data_len: [N, ] tmp_data_label: [N, ]
  tmp_data_all = []
  tmp_data_len = []
  tmp_data_label = []
  tmp_data_name = []
  lines = open(filename,'r')
  lines = list(lines)
  for index in range(0, len(lines)):
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    tmp_data = get_frames_data(pic_root, dirname, num_frames_per_clip)
    tmp_data_all.append(tmp_data) # [len, 16, imgs]
    tmp_data_len.append(len(tmp_data))
    tmp_data_label.append(tmp_label)
    tmp_data_name.append(dirname)
  return tmp_data_all, tmp_data_len, tmp_data_label, tmp_data_name

def read_clip_and_label(pic_root, filename, batch_size, start_pos=0, num_frames_per_clip=16, crop_size=112, shuffle=False):  
  # tmp_data_all: [N, len, 16, imgs]
  tmp_data_all, tmp_data_len, tmp_data_label, tmp_data_name = read_all_frames_data(pic_root, filename, num_frames_per_clip=16)
  datas = []
  for _ in tmp_data_all: datas.extend(_) # [len_all, 16, imgs]

  # crop and normalize images
  np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  img_datas_all = [] # [len_all, 16, imgs]
  for datas_one_batch in datas: # [len_all, 16, imgs]
    datas_one_batch_crop = []
    for j, img_data in enumerate(datas_one_batch): # [16, imgs]
      img = Image.fromarray(img_data.astype(np.uint8))
      if(img.width>img.height):
        scale = float(crop_size)/float(img.height)
        img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
      else:
        scale = float(crop_size)/float(img.width)
        img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
      crop_x = int((img.shape[0] - crop_size)/2)
      crop_y = int((img.shape[1] - crop_size)/2)
      img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
      datas_one_batch_crop.append(img)
    img_datas_all.append(datas_one_batch_crop)

  # change img_datas_all:[len_all, 16, imgs] to [N, batch, 16, imgs]
  while len(img_datas_all)%batch_size!=0: img_datas_all.append(img_datas_all[-1])
  img_datas_all_batch = []
  img_datas_batch = [] # [N, batch, 16, imgs]
  for _ in img_datas_all: # [len_all, 16, imgs]
    img_datas_batch.append(_)
    if len(img_datas_batch)==batch_size:
      img_datas_all_batch.append(img_datas_batch)
      img_datas_batch = []

  img_datas_all_batch = np.array(img_datas_all_batch).astype(np.float32)
  tmp_data_label = np.array(tmp_data_label).astype(np.int64)
  tmp_data_len = np.array(tmp_data_len)

  # img_datas_all_batch: [N, batch, 16, imgs]
  return img_datas_all_batch, tmp_data_label, tmp_data_len, tmp_data_name

if __name__ == '__main__':
  import fire
  fire.Fire()