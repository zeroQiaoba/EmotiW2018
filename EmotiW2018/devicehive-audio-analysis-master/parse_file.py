import os
import glob
import json
import tqdm
import argparse
import numpy as np
from scipy.io import wavfile
from audio.processor import WavProcessor, format_predictions

#os.environ["CUDA_VISIBLE_DEVICES"] = "2" # set cuda number

def gain_features(wav_root='Audio_Dataset', save_path='./EmotiW_youtubeAudio.npz'):

	# save path
	data_path = []
	log_spectrograms_save = []
	embeddings_save = []
	quantized_embeddings_save = []

	proc = WavProcessor()
	for wav_path in tqdm.tqdm(glob.glob(wav_root+'/*')):
		wav_name = os.path.basename(wav_path).split('.')[0]
		sr, data = wavfile.read(wav_path) # sr: sample rate; data: original sample data; data.type uint16
		if data.dtype != np.int16: raise TypeError('Bad sample type: %r' % data.dtype) 
		if len(data) < sr: continue # less than 1s

		examples_batch, features, quantized_embeddings, predictions = proc.get_predictions(sr, data)

		# examples_batch: [3, 96, 60]
		# features: [3, 128]
		# quantized_embeddings: [3, 128]

		frame_num = np.shape(examples_batch)[0]
		data_path.append({'pic_path': wav_name})
		log_spectrograms_save.append(examples_batch.reshape(frame_num,-1))
		embeddings_save.append(features)
		quantized_embeddings_save.append(quantized_embeddings)

	np.savez_compressed(save_path,
						data_path=data_path,
						logspec=log_spectrograms_save,
						emb=embeddings_save,
						qemb=quantized_embeddings_save,
						)

if __name__ == '__main__':
	import fire
	fire.Fire()
	#gain_features(wav_root='Audio_Dataset')