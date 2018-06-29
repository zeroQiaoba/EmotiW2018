# read image
import os
import json
import glob
import numpy as np
from PIL import Image

# gain gif frame length
def gif_len(path):
    frame_num = 1
    im = Image.open(path)  
    try:  
        while True:
            if im.tile:  
                tile = im.tile[0]  
                update_region = tile[1]  
                update_region_dimensions = update_region[2:]  
                if update_region_dimensions != im.size:  
                    break  
            im.seek(im.tell() + 1)
            frame_num += 1
    except EOFError:  
        pass  
    return frame_num  


# process on .gif -> RGB frame list
def read_gif(gif_path):
    im_list = [] # convert GIF to im_list
    frame_num = gif_len(gif_path)
    try:
        im = Image.open(gif_path)
        while (frame_num >= 2 and len(im_list) <= frame_num-2) or (frame_num <= 1 and len(im_list) == 0):
            new_im = Image.new("RGB", im.size)
            new_im.paste(im)
            im_list.append(new_im)
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return im_list


# judge whether path is a valid image file
def IsValidImage(path):
    bValid = True
    try:
        Image.open(path).verify()
    except:
        bValid = False
    return bValid


# read gif and other image type
# return: 0: not a good image, -1 good image path
def multiType_image_loader(path):
    img = ""
    if not IsValidImage(path): return 0, img
    
    basename = os.path.basename(path)
    pic_name, pic_type = basename.split('.')

     # only consider the last img in the gif
    if pic_type == 'GIF' or pic_type == 'gif':
        img = read_gif(path)[-1]
    else:
        img = Image.open(path).convert('RGB')
    return 1, img