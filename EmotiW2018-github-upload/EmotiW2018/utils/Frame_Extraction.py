import os
import glob

###########################################
##Process 0: change video into frame list##
###########################################
def all_video_to_frames(VIDEO_PATH='Video', ORIGIN_FACES_PATH='Origin_Faces'):
	for video_path in glob.glob(VIDEO_PATH + '/*'):
		video_name = os.path.basename(video_path).split('.')[0]
		save_path = os.path.join(ORIGIN_FACES_PATH, video_name)
		if not os.path.exists(save_path): os.makedirs(save_path)
		cmd = 'ffmpeg -i %s -q:v 2 %s/%s.jpg' %(video_path, save_path, 'I_1%3d')
		os.system(cmd)


if __name__ == '__main__':
	import fire
	fire.Fire()
	