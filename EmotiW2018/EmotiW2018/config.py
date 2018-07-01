#coding:utf8
import time
import warnings

tfmt = '%m%d_%H%M%S'
class Config(object):

	model='DenseNet'
	loss = 'crossentropyloss'
	pic_root = 'Database'
	train_eval_test_path = 'Database/train_eval_test.npz'
	model_path = None
	embedding_path=None

	### overall para
	num_classes = 8 # label numbers
	shuffle = True # weather shuffle datas
	num_workers = 4 # 多线程加载所需要的线程数目
	pin_memory = True # 数据从CPU->pin_memory—>GPU加速
	batch_size = 32
	plot_every = 100
	max_epoch=5
	lr = 1e-3 # learning rate
	lr2 = 1e-3
	lr_decay = 0.5 # 当一个epoch的损失开始上升lr = lr*lr_decay 
	weight_decay = 0
	weight = 1 # 正负样本的weight
	decay_every = 100 #每多少个batch 查看一下score,并随之修改学习率
	cuda=-1

	### densenet paras
	num_init_features = 64
	growth_rate = 32
	block_config = (6, 12, 24, 16)
	bn_size=4
	drop_rate=0
	num_classes=8

	# save features
	type_=['eval']
	save_=['eval_features_dense.npz']

	# classifer
	label_root = '../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_filter.npz'
	pic_root = '../EmotiW2017/For_EmotiW2018/Faces'
	features_path = '../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz'
	features_name = 'features_1'
	temporal_ = 'max'
	classifier_save_root = 'classifier_1'
	in_features = 4096
	seq_len = 100
	cluster_size = 64 # NetVLAD
	output_dim = 1024 # NetVLAD
	kmax_pooling = 3 # LSTM
	hidden_size = 128
	linear_hidden_size = 128

	# define three different label files
	label_type = 1 # 1,2,3
	fusion_type = 'gain_para' # 'gain_submit'
	best_para_path = 'best_para.npz'
	
	features_extract_flag = True # False

def parse(self,kwargs,print_=True):
		'''
		根据字典kwargs 更新 config参数 [update accordinng to kwargs]
		'''
		for k,v in kwargs.iteritems():
			if not hasattr(self,k):
				raise Exception("opt has not attribute <%s>" %k)
			setattr(self,k,v) 

		if self.cuda==-1: self.pin_memory=False

		if self.label_type == 1:
			self.label_root = '../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_filter.npz'
			self.classifier_save_root = 'classifier_1'

		# gain necessary info from model_path
		if self.model_path!=None and self.model_path.find('checkpoints')==-1:
			assert self.model_path.find(self.classifier_save_root)!=-1
			info = self.model_path.split('/')[-1]
			info_split = info.split('_')
			info_split_pos = []
			if info.find('EmotiW_VGG_FER_Fintuning')!=-1: info_split_pos=[4,6]
			elif info.find('EmotiW_DenseNet_FER_Fintuning')!=-1: info_split_pos=[4,6]
			elif info.find('EmotiW_C3D_features_all')!=-1: info_split_pos=[4,6]
			elif info.find('EmotiW_Inception_features_Origin')!=-1: info_split_pos=[4,6]
			elif info.find('EmotiW_Inception_features_Faces')!=-1: info_split_pos=[4,6]
			elif info.find('EmotiW_Pose')!=-1: info_split_pos=[2,3]
			elif info.find('EmotiW_Identify')!=-1: info_split_pos=[2,3]
			elif info.find('EmotiW_EmotiW2017_LBPTOP')!=-1: info_split_pos=[3,4]
			elif info.find('EmotiW_traditional_descriptor')!=-1: info_split_pos=[3,4]

			elif info.find('EmotiW_TFIDF')!=-1: info_split_pos=[2,3]
			elif info.find('EmotiW_TFIDFNoEmpty')!=-1: info_split_pos=[2,3]
			elif info.find('EmotiW_txt_vec2')!=-1: info_split_pos=[3,4]

			elif info.find('EmotiW_audio')!=-1: info_split_pos=[2,3]
			elif info.find('EmotiW_IS091113')!=-1: info_split_pos=[2,3]
			elif info.find('EmotiW_youtubeAudio')!=-1: info_split_pos=[2,3]
			elif info.find('EmotiW_EmotiW2017_features')!=-1: info_split_pos=[3,4]

			else: print ('no such features.npz')

			self.features_path = '../EmotiW2017/For_EmotiW2018/%s.npz' %("_".join(info_split[:info_split_pos[0]]))
			self.features_name = "_".join(info_split[info_split_pos[0]:info_split_pos[1]])
			self.temporal_ = "_".join(info_split[info_split_pos[1]:-2])
			self.model=info_split[-2]

		# only 'EmotiW_Inception_features_Origin' considers on origin frames
		if self.features_path.find('EmotiW_Inception_features_Origin')!=-1:
			self.pic_root = '../EmotiW2017/For_EmotiW2018/Origin_Faces'
		else:
			self.pic_root = '../EmotiW2017/For_EmotiW2018/Faces'


		if print_:
			print('user config:')
			print('#################################')
			for k in dir(self):
				if not k.startswith('_') and k!='parse' and k!='state_dict':
					print k,getattr(self,k)
			print('#################################')
		return self

def state_dict(self):
	return {k:getattr(self,k) for k in dir(self) if not k.startswith('_') and k!='parse' and k!='state_dict' }


Config.parse = parse
Config.state_dict = state_dict
opt = Config()
