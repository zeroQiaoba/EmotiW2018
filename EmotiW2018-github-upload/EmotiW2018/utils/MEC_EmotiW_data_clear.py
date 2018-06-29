#coding:utf8
import os
import cv2
import json
import time
import tqdm
import glob
import shutil
import numpy as np
import dlib

edict = {}
edict['angry'] = 0
edict['disgust'] = 1
edict['Fear'] = 2  ## not exist
edict['happy'] = 3
edict['sad'] = 4
edict['surprise'] = 5
edict['neutral'] = 6


# del empty files
def data_item_clear(pic_root = 'Faces'):
	for pic_path in glob.glob(pic_root+'/*'):
		if os.path.isdir(pic_path):
			if len(glob.glob(pic_path+'/*')) == 0:
				print(pic_path)
				os.system('rm -rf %s' %(pic_path))
			

if __name__ == '__main__':
	import fire
	fire.Fire()