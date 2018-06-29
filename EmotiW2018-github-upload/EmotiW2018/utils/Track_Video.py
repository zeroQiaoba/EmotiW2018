'''
Target:	Aligned faces in the video by openface
------------------------------------------------
Input:
	use the bb from the first frame in video to track the video
	
Output:  
	Aligned_faces of the video
------------------------------------------------
Main process: 
	## Process 1: generate bb.txt in TXT_SAVE_PATH
	#all_video_to_bb
		
	## Process 2: modify the bb.txt in TXT_SAVE_PATH and see the result
	#all_video_show_bb
		
	## Process 3: Track the video use the first bb
	#track_frames_from_video
------------------------------------------------
Author: Robert Lian

Date:	2017/5/30	Init version
'''

#!/usr/bin/env python2
import os
import glob
import cv2
import dlib
import openface
import numpy as np
import shutil

IMGDIM = 128
fileDir = '/root/openface/demos'
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(dlibFacePredictor)

#########################################
##Process 0: change video into frame list##
#########################################
def all_video_to_frames(VIDEO_PATH, ORIGIN_FACES_PATH):
	for video_path in glob.glob(VIDEO_PATH + '/*'):
		video_name = os.path.basename(video_path).split('.')[0]
		save_path = os.path.join(ORIGIN_FACES_PATH, video_name)
		if not os.path.exists(save_path): os.makedirs(save_path)
		cmd = 'ffmpeg -i %s -q:v 2 %s/%s.jpg' %(video_path, save_path, 'I_1%3d')
		os.system(cmd)

#########################################
##process 1: gain bb in the first frame##
#########################################
# return bb of the first frames
def gain_pic_bb(first_frame_path, first_frame_save_path):
	bgrImg = cv2.imread(first_frame_path)
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
	bb = align.getLargestFaceBoundingBox(rgbImg)
	cv2.imwrite(first_frame_save_path, bgrImg)

	# store bb
	if bb is None:
		return dlib.rectangle(-1,-1,-1,-1)
	else:
		# save aligned face, judge the bb is good or not
		alignedFace = align.align(IMGDIM, rgbImg, bb, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
		alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		alignedFace = clahe.apply(alignedFace)
		cv2.imwrite(first_frame_save_path + '_bb.jpg', alignedFace)
		return bb

# save process
def save_dict_to_txt(bb_dict,save_path):
	file = open(save_path, 'w')
	for key in sorted(bb_dict):
		bb = bb_dict[key]
		file.write(	key+' '+
					str(bb.left())+' '+
					str(bb.top())+' '+
					str(bb.right())+' '+
					str(bb.bottom())+'\n')
	file.close()

# main process
def all_video_to_bb(frames_root, save_root, bb_txt_save_path):
	print ('----------- Start ----------')

	if not os.path.exists(save_root): os.makedirs(save_root)
	bb_dict = {}
	for video_path in glob.glob(frames_root+'/*'):
		video_name = os.path.basename(video_path) # 'xxx'
		first_frame_path = sorted(glob.glob(video_path+'/*'))[0]
		first_frame_save_path = os.path.join(save_root, video_name + '.jpg')
		bb = gain_pic_bb(first_frame_path, first_frame_save_path)
		bb_dict[video_name] = bb
	save_dict_to_txt(bb_dict, bb_txt_save_path)
	
	print ('----------- Done ----------')
	
#########################################
##process 2: check bb we modified##
#########################################
# change process	
def change_txt_to_dict(txt_path):
	file = open(txt_path)
	bb_dict = {}
	for line in file.readlines():
		line_split = line.split(' ')
		bb = dlib.rectangle(long(line_split[1]),
							long(line_split[2]),
							long(line_split[3]),
							long(line_split[4]))
		bb_dict[line_split[0]] = bb
	file.close()
	return bb_dict

# main process
def all_video_show_bb(image_root, bb_txt_save_path):
	print ('----------- Start ----------')

	bb_dict = change_txt_to_dict(bb_txt_save_path)
	for image_path in glob.glob(image_root+'/*'):
		if image_path.find('bb')!=-1: continue
		video_name = os.path.basename(image_path).split('.')[0]
		bb = bb_dict[video_name]
		if bb != dlib.rectangle(-1,-1,-1,-1): # change first_frame_save_path to new first_frame_save_path_bb
			bgrImg = cv2.imread(image_path)
			rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
			alignedFace = align.align(IMGDIM, rgbImg, bb,
										landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
			alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2GRAY)
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			alignedFace = clahe.apply(alignedFace)
			cv2.imwrite(image_path + '_bb.jpg', alignedFace)
	
	print ('----------- Done ----------')	


#########################################
##process 3: track video with first bb##
#########################################	
# dlib.drectangle to dlib.rectangle
def drect2rect(x):
	left = long(x.left())
	top = long(x.top())
	right = long(x.right())
	bottom = long(x.bottom())
	return dlib.rectangle(left, top, right, bottom)

def video2frames(frames_path, save_path, txt_save_path):
	
	print ('-----------start change ' + frames_path +'------------')
	video_name = os.path.basename(frames_path)
	bb_dict = change_txt_to_dict(txt_save_path)
	first_bb = bb_dict[video_name]

	# track frames
	track_flag = 0
	alignedFaces = []
	originPaths = []
	tracker = dlib.correlation_tracker()
	for frame_path in sorted(glob.glob(frames_path+'/*')):
		bgrImg = cv2.imread(frame_path)
		rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
		if track_flag == 0:
			tracker.start_track(rgbImg, first_bb)
			track_flag = 1
		else:
			tracker.update(rgbImg)
		position = tracker.get_position()
		alignedFace = align.align(IMGDIM, rgbImg, drect2rect(position),
							landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
		alignedFaces.append(alignedFace)
		originPaths.append(frame_path)

	# save track frames
	for ii, alignedFace in enumerate(alignedFaces):
		originPath = originPaths[ii]
		originName = os.path.basename(originPath)
		savePath = os.path.join(save_path, originName)
		alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		alignedFace = clahe.apply(alignedFace)
		cv2.imwrite(savePath, alignedFace)
	
	print ('-----------end change ' + frames_path +'------------')
	
def track_frames_from_video (origin_faces_root, aligned_faces_root, txt_save_path):
	print ('----------- Start ----------')

	for frames_path in glob.glob(origin_faces_root + '/*'):
		video_name = os.path.basename(frames_path)
		save_path = os.path.join(aligned_faces_root, video_name)
		if not os.path.exists(save_path): os.makedirs(save_path)
		video2frames(frames_path, save_path, txt_save_path)
			
	print ('----------- Done ----------')


def usage():
	## Process 0: change video into frame list (select and del other frames first)[in windows]
	#all_video_to_frames(VIDEO_PATH, ORIGIN_FACES_PATH)

	## Process 1: generate bb.txt in TXT_SAVE_PATH
	#all_video_to_bb(ORIGIN_FACES_PATH, SAVE_PATH, TXT_SAVE_PATH)
	
	## Process 2: modify the bb.txt in TXT_SAVE_PATH and see the result
	#all_video_show_bb(SAVE_PATH, TXT_SAVE_PATH)
	
	## Process 3: Track the video use the first bb
	#track_frames_from_video(ORIGIN_FACES_PATH, ALIGNED_FACES_PATH, TXT_SAVE_PATH)


if __name__ == '__main__':

	# VIDEO_PATH = 'Video'
	# ORIGIN_FACES_PATH = 'Origin_Faces'
	# SAVE_PATH = 'IMAGE'
	# TXT_SAVE_PATH = os.path.join(SAVE_PATH,'bb.txt')
	# ALIGNED_FACES_PATH = 'Aligned_openface'
	usage()
	
	import fire
	fire.Fire()

	
	