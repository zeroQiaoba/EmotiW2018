'''
Target:	Extract audio features by opensmile (Realized on Window 10)
------------------------------------------------
Input:      
		audio
            
Output: 
		audio_features .npy file
------------------------------------------------
Author: Robert Lian

Date:	2017/08/28	Init version
'''

import os
import sys 
import glob
import numpy as np

# Change the features into .npy file
def txt_to_npy(txt_path, npy_path):   
	# Gain features
	ifData=0
	fr=open(txt_path)
	for line in fr.readlines():
		if (line.find('@data') != -1 and ifData==0):
			ifData=1
		if (line.find(',') != -1 and ifData==1):
			lineStr=line.strip().split(',')
			lineData=lineStr[1:-1] # Delete name and label
			np.save(npy_path, lineData)
	return

# case 1: windows
opensmile_path = "opensmile-2.3.0\\bin\\Win32\\SMILExtract_Release.exe"
config_root = 'opensmile-2.3.0\\config'
config_names = ["IS09_emotion.conf", "IS11_speaker_state.conf", "IS13_ComParE.conf"]
def main(audio_root='Audio'):
	for conf_name in config_names:
		conf_path = os.path.join(config_root, conf_name)
		for audio_path in glob.glob(audio_root+'/*'):
			save_root = os.path.basename(conf_path)
			if not os.path.exists(save_root): os.makedirs(save_root)

			audio_name = os.path.basename(audio_path).split('.')[0]
			txt_path = os.path.join(save_root, audio_name+'.txt')
			npy_path = os.path.join(save_root, audio_name+'.npy')

			# Gain audio features using opensmile
			cmd = opensmile_path + ' -C ' + conf_path + ' -I ' + audio_path + ' -O ' + txt_path
			os.system(cmd)
			# Change txt to npy
			txt_to_npy(txt_path, npy_path)
			# remove txt file 
			cmd = 'rm -rf ' + txt_path
			os.system(cmd)

def change_features_into_npz():
	name_2_IS09 = {}
	for features_path in glob.glob('IS09_emotion.conf/*'):
		if features_path.find('.npy')!=-1:
			features = np.load(features_path).astype('float')
			name = os.path.basename(features_path).split('.')[0]
			name_2_IS09[name] = features

	name_2_IS11 = {}
	for features_path in glob.glob('IS11_speaker_state.conf/*'):
		if features_path.find('.npy')!=-1:
			features = np.load(features_path).astype('float')
			name = os.path.basename(features_path).split('.')[0]
			name_2_IS11[name] = features

	name_2_IS13 = {}
	for features_path in glob.glob('IS13_ComParE.conf/*'):
		if features_path.find('.npy')!=-1:
			features = np.load(features_path).astype('float')
			name = os.path.basename(features_path).split('.')[0]
			name_2_IS13[name] = features

	data_path = []
	features_IS09 = []
	features_IS11 = []
	features_IS13 = []
	for name in name_2_IS13:
		data_path.append({'pic_path': name})
		features_IS09.append(name_2_IS09[name])
		features_IS11.append(name_2_IS11[name])
		features_IS13.append(name_2_IS13[name])

	np.savez_compressed('./EmotiW_IS091113.npz',
						data_path=data_path,
						IS09=features_IS09, # [1799, 384]
						IS11=features_IS11, # [1799, 4368]
						IS13=features_IS13, # [1799, 6373]
						)


if __name__ == '__main__':
	import file
	fire.Fire()
	#main()
	#change_features_into_npz()
	
	