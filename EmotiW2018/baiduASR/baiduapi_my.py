'''
Target: BaiduASR
------------------------------------------------
Input:      
    audio
            
Output: 
    content in the audio
------------------------------------------------
Author: Robert Lian

Date: 2018/06/01  Init version
'''

#encoding=utf-8
import wave
import json
import os
import glob
import numpy as np
from aip import AipSpeech

""" input your own APP_ID AK SK """
APP_ID = 'xxxx'
API_KEY = 'xxxxxxx'
SECRET_KEY = 'xxxxxxxxxxxxxxxxxxxxx'

aipSpeech = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# read files
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def asr_for_one_file(file_path):
	file_name = os.path.basename(file_path)

	new_file_path = os.path.join('pcm', file_name + '.pcm')
	cmd = 'ffmpeg -y -i %s -acodec pcm_s16le -f s16le -ac 1 -ar 16000 %s' %(file_path, new_file_path)
	os.system(cmd)

	result = aipSpeech.asr(get_file_content(new_file_path), 'pcm', 16000, {'dev_pid': 1737})

	if result['err_no'] == 0: 
		return result['result'][0]
	else:
		return ''

def asr_for_one_root(file_root='wav', save_path='EmotiW_txt.npz'):
	test_datas = []
	results = []
	for file_path in glob.glob(file_root+'/*'):
		result = asr_for_one_file(file_path)
		file_path = os.path.basename(file_path).split('.')[0]
		results.append(result)
		test_datas.append({'pic_path': file_path})

	np.savez_compressed(save_path,
						data_path=test_datas,
						text=results,
						)

if __name__ == '__main__':
	import fire
	fire.Fire()
	#asr_for_one_root()