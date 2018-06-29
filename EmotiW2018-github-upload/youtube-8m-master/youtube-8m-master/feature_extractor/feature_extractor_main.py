from feature_extractor import *
from PIL import Image
import numpy as np
import os

# finish
def extract_from_origin_faces(	train_eval_test_path = '../../../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_bottleneck_Origin_Faces.npz',
								pic_root='../../../EmotiW2017/For_EmotiW2018/Origin_Faces',
								save_path='./EmotiW_Inception_features_Origin.npz'):
	model_path = './yt8m'
	extractor = YouTube8MFeatureExtractor(model_dir=model_path)
	features = []
	datas = np.load(train_eval_test_path)['train_datas'] # all pics
	for item in datas:
	  pic_name = item['pic_path']

	  pic_path = os.path.join(pic_root, pic_name)
	  im = numpy.array(Image.open(pic_path))
	  feature = extractor.extract_rgb_frame_features(im)
	  features.append(feature)

	np.savez_compressed(save_path, data_path=datas, features_1=features)


def extract_from_faces(	train_eval_test_path = './../../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_bottleneck_Faces.npz',
						pic_root='../../../EmotiW2017/For_EmotiW2018/Faces',
						save_path='./EmotiW_Inception_features_Faces.npz'):

	model_path = './yt8m'
	extractor = YouTube8MFeatureExtractor(model_dir=model_path)
	features = []
	datas = np.load(train_eval_test_path)['train_datas'] # all pics
	for item in datas:
	  pic_name = item['pic_path']

	  pic_path = os.path.join(pic_root, pic_name)
	  im = Image.open(pic_path)
	  im = im.convert('RGB') 
	  im = numpy.array(im)
	  feature = extractor.extract_rgb_frame_features(im)
	  features.append(feature)

	np.savez_compressed(save_path, data_path=datas, features_1=features)


if __name__ == '__main__':
	import fire
	fire.Fire()
