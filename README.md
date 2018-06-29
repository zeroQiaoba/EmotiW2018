# EmotiW2018—NLPR Teams Code

[EmotiW2018](https://sites.google.com/view/emotiw2018) focuses on audio-video emotion classification tasks, which contains seven basic emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral. In the experiment, we adopt additional databases: [FER+ database](https://github.com/Microsoft/FERPlus), MEC2017 database and SFEW2015 database. 

(Note: The [MEC2017](http://www.chineseldc.org/emotion.html) database can only be avaliable if you participate in the MEC2017 challenges. The SFEW2015 database can be avalible if you ask organizers for the download link)

Since lack of permission, we only show our solution in EmotiW2018. Any samples in the challenge database or related information can no be found.

## 1. Setup

- [PyTorch](https://github.com/pytorch/pytorch)
- [Tensorflow](https://github.com/tensorflow/tensorflow): extract C3D, Inception features, sound features
- [sklearn](http://scikit-learn.org/stable/): training linearSVC, RBFSVC, RF classifiers
- [cyvlfeat](https://github.com/menpo/cyvlfeat): realize Fisher Vectors (Change Frame-level features into Video-level features)
- [openface](https://github.com/cmusatyalab/openface): face extration (which is based on [dlib](https://github.com/davisking/dlib))
- intraface: face extraction (which is appplied under vs2010 + opencv)
- ffmpeg: split audio from video, extract frames from video
- fire: available command line input
- tqdm: show progress


## 2. Files

- `Audio_OpenSmile/`: Extract opensmile featrues for wav files
  - `Audio/`: save origin wav files
  - `opensmile-2.3.0/`: opensmile source path
  - `featureExtract.py`: Extract audio features
- `baiduASR/`: Recognize the content for each audio and extract textual features
- `C3D-tensorflow-master/`: Extract C3D features
- `devicehive-audio-analysis-master/`: Extract youtube-8M audio features
- `EmotiW2017/`: save samples and features for EmotiW2018
- `EmotiW2018/`: the main fold
  - `checkpoints/`: save fintuned models (like DenseNet mdoels, VGG models)
  - `data/`: for data loader, Fisher Vector and other functions
  - `Database/`: FER+ databases
  - `models/`: classification models and fintuning models
  - `utils/`: extract faces
  - `config.py`: configuration files
  - `main.py`: fintuning pretrained models on FER+ database and SFEW database
  - `feature_extract.py`: extract DenseNet and VGG bottleneck features
  - `classifier.py` and `run_train.py`: train classifier for different features
  - `classifier_lp.py` and `loupe.py`: train 'NetVLAD', 'NetRVLAD', 'SoftDBoW', 'NetFV' classifiers
  - `predict.py`: gain predictions for samples and save them.
  - `fusion_prediction.py` and `gain_cnn_para.py`: gain fusion parameters and submission
- `MEC2017/`: save samples and features for MEC2017 (same as `EmotiW2017/`)
- `Point_Pose_Indenfy/`: Extract head pose, landmark points and Identification features.
- `Traditional_video_descriptor/`: Extract DSIFT, HOG, HOG_LBP, LBP features. Since these codes are not written by me. I have no ability to open them. You can find extraction methods in the web.
- `youtube-8m-master/`: Extract Inception features of youtube-8M
- Multi-model Emotion Recognition.pptx: conclusion of pervious EmotiW challenges.

## 3. Data Preporcess

In this section, we show our face recognition methods.

Source codes are in `EmotiW2018/utils`

### 3.1 label Discription

Since MEC2017 and EmotiW2018 both are discrete emotion classification problems. Their labels have overlapping parts.

|          | MEC  |          | EmotiW/SFEW |
| -------- | :--- | -------- | ----------- |
| angry    | 0    | Angry    | 0           |
| disgust  | 1    | Disgust  | 1           |
|          | 2    | Fear     | 2           |
| happy    | 3    | Happy    | 3           |
| sad      | 4    | Sad      | 4           |
| surprise | 5    | Surprise | 5           |
| neutral  | 6    | Neutral  | 6           |
| anxious  | 7    |          | 7           |
| worried  | 8    |          | 8           |

### 3.2 Face Extraction

In this section, we change `Video` into `Origin_Faces` and `Faces`.

- `Origin_Faces`: the origin frames in the video. 
- `Faces`: the aligned faces of each videos.

```sh
## step 0: change video into frames
python utils/Frame_Extraction.py all_video_to_frames --VIDEO_PATH='Video' --ORIGIN_FACES_PATH='Origin_Faces'

## step 1: recognize faces through intraface (realized under vs2010, C++, Window 10)
## source code is in 'utils/IntraFacev-2018-0628-EmotiW.rar' (change Origin_Faces into Faces)

## step 2: del empty files(which means faces in the video is false detected)
python utils/MEC_EmotiW_data_clear.py data_item_clear --pic_root='Faces'

## step 3: find empty files and track faces through openface
### step 3.1: init bb in the first frame
python utils/Track_Video.py all_video_to_bb --frames_root='Origin_Faces' --save_root='IMAGE' --bb_txt_save_path='IMAGE/bb.txt'
### step 3.2: mannual correct bb in the first frame (mannual change the bb.txt)
### step 3.3: show whether bb is correct
python utils/Track_Video.py all_video_show_bb --image_root='IMAGE' --bb_txt_save_path='IMAGE/bb.txt'
### step 3.4: track bb
python utils/Track_Video.py track_frames_from_video --origin_faces_root='Origin_Faces' --aligned_faces_root='Aligned_openface' --txt_save_path='IMAGE/bb.txt'
```

###3.3 Data Format Normalization

MEC2017 path: `../MEC2017`

data format：

- `Faces`: Extract faces based on openface
- `train_eval_test_MEC2017_filter.npz`: save emotion labels [pic_path='2_004_03_022_F003_e', label=1]
- `train_eval_test_MEC2017_filter_bottleneck.npz`: save frame path for features extraction
- `data_format.py`: generate two .npz files




EmoitW2018 path: `../EmotiW2017/For_EmotiW2018`

data format：

- `Faces`:  Extracted based on openface
- `train_eval_test_EmotiW2017.npz`: save emotion labels  [pic_path='020205600', label=1]
  - train_datas(773), eval_datas(383), test_datas(653)
- `train_eval_test_EmotiW2017.npz_bottleneck.npz`: save frame path for features extraction
- `data_format.py`: generate two .npz files

## 4. Image Feature Extractor

To extract frame features, we need to train frame-level emotion classifiers. And then we can extract frame features through bottleneck layers.

- Training frame-level emotion classifiers
  - Fintuning on FER+ database
  - Fintuning on  FER+ database at first, and then fintuning on SFEW database
- Bottleneck features extraction: visual features are listed behind.
  - EmotiW_DenseNet_FER_Fintuning.npz (frame-level features): `features_1` ~` features_4`
  - EmotiW_VGG_FER_Fintuning.npz (frame-level features): `features_1` ~` features_9`
  - EmotiW_C3D_features_all.npz (segment-level features): `features_1`
  - EmotiW_Inception_features_Faces.npz (frame-level features): `features_1`
  - EmotiW_Inception_features_Origin.npz (frame-level features): `features_1`
  - EmotiW_traditional_descriptor.npz (video-level features): `HOG`, `LBP`, `HOGLBP`, `DSIFT`
  - EmotiW_EmotiW2017_LBPTOP.npz (video-level features): `LBPTOP` (不是每个视频都有LBPTOP)
  - EmotiW_Pose.npz (frame-level features): `poses`, `posesDev`
  - EmotiW_Identify.npz (frame-level features): `identifies`

### 4.1 Training frame-level emotion classifiers

#### 4.1.1 Fintuning on FER+ Database

Source code is in `EmotiW2018/main.py`, parameters are following behind:

- `--decay_every`: change according to `batch-size` and training_sample_numbers. Must follow the following equation: decay_every*batch_size<training_sample_numbers. Because  we save model and adjust learning rate each opt.decay_every time
- `--plot-every`:  visualize one batch result every opt.plot_every time
- `--lr` and `--lr2` : learning rate
- `--lr_decay`: if scores don't imporve in the val dataset, we will reduce`lr` into `lr*lr_decay`
- `--cuda`: -1(CPU), >=0(GPU)
- `--num_classes`: output number of the model
- `--max_epoch`: max training epoch
- `--train_eval_test_path`: generated through Database/datas_norm_format.py

```sh
# DenseNet
python main.py main --model='DenseNet' --model_path=None --loss='crossentropyloss' --num_classes=8 --train_eval_test_path='Database/train_eval_test.npz' --pic_root='Database' --plot_every=100 --batch-size=128  --lr=0.001 --lr2=0 --lr_decay=0.5  --decay-every=234 --max_epoch=30 --cuda=-1

# VGG (add bn on VGG is better)
nohup python -u main.py main --model='VGG' --model_path=None --loss='crossentropyloss' --num_classes=8 --train_eval_test_path='Database/train_eval_test.npz' --pic_root='Database' --plot_every=100 --batch-size=128 --lr=0.01 --lr2=0 --lr_decay=0.5 --decay-every=234 --max_epoch=20 --cuda=-1
```

#### 4.1.2 Fintuning on SFEW2015

In this section, we need to change `--num_classes ` to 7. Because SFEW2015 has seven emotion labels. Therefore, the output number of the model is 7. And we fintune on FER+ pretrained models.

```sh
python main.py main --model='DenseNet' --model_path='checkpoints/DenseNet_80.9635722679' --loss='crossentropyloss' --num_classes=7 --train_eval_test_path='../SFEW2015/train_eval_test_SFEW2015_filter.npz' --pic_root='../SFEW2015/Faces' --plot_every=1 --batch-size=128  --lr=0.0001 --lr2=0 --lr_decay=0.5  --decay-every=1 --max_epoch=20 --cuda=-1

python main.py main --model='VGG' --model_path='checkpoints/VGG_80.0528789659' --loss='crossentropyloss' --num_classes=7 --train_eval_test_path='../SFEW2015/train_eval_test_SFEW2015_filter.npz' --pic_root='../SFEW2015/Faces' --plot_every=1 --batch-size=128  --lr=0.0001 --lr2=0 --lr_decay=0.5  --decay-every=1 --max_epoch=20 --cuda=-1
```

#### 4.1.3 scores

| model              | score on val |
| ------------------ | ------------ |
| DenseNet, FER+     | 0.8096       |
| VGG, FER+          | 0.8005       |
| DenseNet, SFEW2015 | 0.8081       |
| VGG, SFEW2015      | 0.7980       |

### 4.2 Bottleneck features extraction

In this section, we extract bottleneck features of DenseNet and VGG.

Source code is in `feature_extract.py`.

- `--model`: 'DenseNet' or 'VGG'
- `--model_path`: model path for feature extraction
- `--train_eval_test_path`: generated from '../EmotiW2017/For_EmotiW2018/data_format.py' and 'MEC2017/data_format.py'
- `--pic_root`: aligned Face path
- `--save_`: save path for features
- `--cuda`: -1(CPU), 0~(GPU)

#### 4.2.1 EmotiW: Bottleneck features extraction

```sh
######################## DenseNet ##############################
nohup python -u feature_extract.py main_DenseNet --model='DenseNet' --num_classes=8 --model_path='checkpoints/DenseNet_80.9635722679' --train_eval_test_path='../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_bottleneck_Faces.npz' --pic_root='../EmotiW2017/For_EmotiW2018/Faces' --type_='test' --save_='EmotiW_DenseNet_FER_Fintuning.npz' --cuda=-1

nohup python -u feature_extract.py main_DenseNet --model='DenseNet' --num_classes=7 --model_path='checkpoints/DenseNet_80.8080808081_fintuning_SFEW' --train_eval_test_path='../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_bottleneck_Faces.npz' --pic_root='../EmotiW2017/For_EmotiW2018/Faces' --type_='test' --save_='EmotiW_DenseNet_SFEW_Fintuning.npz' --cuda=-1

######################## VGG ##############################
nohup python -u feature_extract.py main_VGG --model='VGG' --num_classes=8 --model_path='checkpoints/VGG_80.0528789659' --train_eval_test_path='../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_bottleneck_Faces.npz' --pic_root='../EmotiW2017/For_EmotiW2018/Faces' --type_='test' --save_='EmotiW_VGG_FER_Fintuning.npz' --cuda=-1

nohup python -u feature_extract.py main_VGG --model='VGG' --num_classes=7 --model_path='checkpoints/VGG_79.797979798_fintuning_SFEW' --train_eval_test_path='../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_bottleneck_Faces.npz' --pic_root='../EmotiW2017/For_EmotiW2018/Faces' --type_='test' --save_='EmotiW_VGG_SFEW_Fintuning.npz' --cuda=-1
```

#### 4.2.2 MEC: Bottleneck features extraction

```sh
######################## DenseNet ##############################
nohup python -u feature_extract.py main_DenseNet --model='DenseNet' --num_classes=8 --model_path='checkpoints/DenseNet_80.9635722679' --train_eval_test_path='../MEC2017/train_eval_test_MEC2017_filter_bottleneck.npz' --pic_root='../MEC2017/Faces' --type_='test' --save_='MEC_DenseNet_FER_Fintuning.npz' --cuda=-1

nohup python -u feature_extract.py main_DenseNet --model='DenseNet' --num_classes=7 --model_path='checkpoints/DenseNet_80.8080808081_fintuning_SFEW' --train_eval_test_path='../MEC2017/train_eval_test_MEC2017_filter_bottleneck.npz' --pic_root='../MEC2017/Faces' --type_='test' --save_='MEC_DenseNet_SFEW_Fintuning.npz' --cuda=-1

######################## VGG ##############################
nohup python -u feature_extract.py main_VGG --model='VGG' --num_classes=8 --model_path='checkpoints/VGG_80.0528789659' --train_eval_test_path='../MEC2017/train_eval_test_MEC2017_filter_bottleneck.npz' --pic_root='../MEC2017/Faces' --type_='test' --save_='MEC_VGG_FER_Fintuning.npz' --cuda=-1

nohup python -u feature_extract.py main_VGG --model='VGG' --num_classes=7 --model_path='checkpoints/VGG_79.797979798_fintuning_SFEW' --train_eval_test_path='../MEC2017/train_eval_test_MEC2017_filter_bottleneck.npz' --pic_root='../MEC2017/Faces' --type_='test' --save_='MEC_VGG_SFEW_Fintuning.npz' --cuda=-1
```

### 4.3. C3D feature extraction

This section extract C3D features, which follows the [link](https://github.com/hx173149/C3D-tensorflow).

To run the code, you must come into `../C3D-tensorflow-master/C3D-tensorflow-master`

There are several important parameters in the `predict_c3d_ucf101.py`:

- `--pic_root`: aligned face path
- `--test_list_file`: 'a.txt' save 'video_name label' in each line. For example: '000534800 3'
- `--save_path`: save features

```sh
# Extract C3D for EmotiW2018
python predict_c3d_ucf101.py --pic_root='../../EmotiW2017/For_EmotiW2018/Faces' --test_list_file='../../EmotiW2017/For_EmotiW2018/a.txt' --save_path='EmotiW_C3D_features_all.npz'

# Extract C3D for MEC2017
python predict_c3d_ucf101.py --pic_root='../../MEC2017/Faces' --test_list_file='../../MEC2017/a.txt' --save_path='MEC_C3D_features_all.npz'
```

### 4.4. Inception feature extraction(from Youtube8M)

This section extract Inception features for both aligned faces and original frames, which follows the [link](https://github.com/google/youtube-8m).

To run the code, you must come into `youtube-8m-master/youtube-8m-master/feature_extractor`

There are several important parameters in the `feature_extractor_main.py`:

- `--train_eval_test_path`: show the frame path of Origin_Faces or Faces
- `--pic_root`: show the frame root of Origin_Faces or Faces
- `--save_path`: save path

```sh
# Extract 1024D inception features from aligned faces
python feature_extractor_main.py extract_from_faces --train_eval_test_path='../../../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_bottleneck_Origin_Faces.npz' --pic_root='../../../EmotiW2017/For_EmotiW2018/Origin_Faces' --save_path='./EmotiW_Inception_features_Origin.npz'

# Extract 1024D inception features from origin frames
python feature_extractor_main.py extract_from_origin_faces --train_eval_test_path='./../../EmotiW2017/For_EmotiW2018/train_eval_test_EmotiW2017_bottleneck_Faces.npz' --pic_root='../../../EmotiW2017/For_EmotiW2018/Faces' --save_path='./EmotiW_Inception_features_Faces.npz'
```

### 4.5 Pose features extraction and Indentify features Extraction 

This section extract Pose features and Indentify features.

Source codes are in ` Point_Pose_Indenfy/`, They are all writen in C++.

- 'Pose features' include head pose and facial landmark points. These Features are extracted using Dlib toolkit under vs2015.
- 'Identify features' : These Features are extracted using Seetaface toolkit under vs2013.

### 4.6 Traditional Video Features Extraction 

This section extract `HOG`, `LBP`, `HOGLBP`, `DSIFT`.

Source codes are in ` Traditional_video_descriptor/`

```sh
## Dsift features
python dsift.py dsift_extractor --video_root='../EmotiW2017/For_EmotiW2018/Video' --face_root='../EmotiW2017/For_EmotiW2018/Faces' --save_path='./Dsift_features.npz'

## HOG features
python hog.py hog_extractor --video_root='../EmotiW2017/For_EmotiW2018/Video' --face_root='../EmotiW2017/For_EmotiW2018/Faces' --save_path='./Hog_features.npz'

## HOG_LBP features
python hog_lbp.py hoglbp_extractor --video_root='../EmotiW2017/For_EmotiW2018/Video' --face_root='../EmotiW2017/For_EmotiW2018/Faces' --save_path='./HogLBP_features.npz'

## LBP features
python LBP.py lbp_extractor --video_root='../EmotiW2017/For_EmotiW2018/Video' --face_root='../EmotiW2017/For_EmotiW2018/Faces' --save_path='./LBP_features.npz'
```

### 4.7 EmotiW_EmotiW2017_LBPTOP

LBPTOP features are provided by organizers.

## 5. Audio Feature Extraction

Original audio files are extracted from video by ffmpeg.

```sh
ffmpeg -i video_path audio_path
```

After we separate audio files from video, we extract different audio features sets.

- EmotiW_audio.npz (frame-level features): `mfcc`, `soundnet`,`IS10`, `Egemaps `
- EmotiW_EmotiW2017_features.npz (frame-level features): `English`, `Chinese`
- EmotiW_EmotiW2017_features.npz (utterance-level features): `Egmaps`
- EmotiW_IS091113.npz (utterance-level features): `IS09` ,`IS11`, `IS13`
- EmotiW_youtubeAudio.npz (segment-level features): `logspec`, `emb`, `qemb`

### 5.1 EmotiW_audio   

 This section includes `mfcc`, `soundnet`,`IS10`, `Egemaps `. They are all frame-level features.

- soundnet: extract based on torch7. [link]((https://github.com/cvondrick/soundnet))
- mfcc, IS10, Egemaps: extract based on opensmile

### 5.2 EmotiW_EmotiW2017_features

 This section includes`English`, `Chinese`, `Egmaps`. 

- `English`, `Chinese`: They are all frame-level features. These features are extracted from Chinese-ASR bottleneck layers and English-ASR bottleneck layers, respectively.
- Egmaps: utterance-level features, extracted through opensmile

### 5.3 EmotiW_IS091113

This section includes `IS09` ,`IS11`, `IS13`. They are all utterance-level features. They are extract in windows though OpenSmile.

Source codes are in ` Audio_OpenSmile/featureExtract.py`.

```sh
# extract audio features
python featureExtract.py main

# save all features into .npz files
python featureExtract.py change_features_into_npz
```

### 5.4 EmotiW_youtubeAudio

This section includes `logspec`, `emb`, `qemb`. They are all segment-level features. Each segment is one second.

Source codes are in ` devicehive-audio-analysis-master/`, which refers to [link](https://github.com/devicehive/devicehive-audio-analysis)

- `--wav_root`: wav dir contains all wav files.
- `--save_path`: save path

```sh
# extract audio features
python parse_file.py  gain_features --wav_root='Audio_Dataset' --save_path='./EmotiW_youtubeAudio.npz'
```

## 6. Textual Feature Extraction

We utilize open-source toolkit, baidu ASR API, to recognize text in the audio.

After we recognize text in audio, we extract different audio features sets:

- EmotiW_TFIDF.npz (utterance-level features): `TFIDF`
- EmotiW_TFIDFNoEmpty.npz (utterance-level features): TFIDFNoEmpty（不是每句话都有）
- EmotiW_txt_vec.npz (utterance-level features): word2vec（不是每句话都有）

### 6.1 ASR in Baidu API

Source codes are in `baiduASR/baiduapi_my.py`

To utilize baiduAPI, we need to register. Please follow [link](http://ai.baidu.com/docs#/ASR-Online-Python-SDK/top).

Then set your own APP_ID, API_KEY and SECRET_KEY in `baiduASR/baiduapi_my.py`.

- `--file_root`: wav root path
- `--save_path`: save path

```sh
python baiduapi_my.py asr_for_one_root --file_root='wav' --save_path='EmotiW_txt_pcm.npz'
```

Through this file, we change 'wav' to 'EmotiW_txt.npz'

### 6.2 Extract TFIDF

Source codes are in `baiduASR/Text_Features.py`

```sh
## preporcess on text. Add 'Word_Tabel' in 'EmotiW_txt.npz'
python Text_Features.py emotiW_extraction --asr_root='EmotiW_txt.npz'

## Extract TFIDF features (which will utilize word_Table)
python Text_Features.py extract_TFIDF
```

### 6.3 Extract Word2Vec Features

Source codes are in `baiduASR/gain_word2vec.py`. We utilize pre-trained fasttext word vectors. [link](https://fasttext.cc/docs/en/english-vectors.html)

```sh
# change text to vector
python gain_word2vec.py change_txt_vec --features_path='EmotiW_txt.npz' --save_path='EmotiW_txt_vec.npz'

# del empty features (whose text is false recognized through ASR)
python gain_word2vec.py del_empty --features_path='EmotiW_txt_vec.npz'
```

## 7. Classifier

Source code is in `EmotiW2018/classifier.py`

`--label_type`: change `label_root` and `classifier_save_root` in `config.py`

`--features_path`：

- As for frame level features, data_path['pic_path'] refers to frame path
- As for video level features, data_path['pic_path'] refers to video name

`--features_name`:
- visual features
  - EmotiW_DenseNet_FER_Fintuning.npz (frame-level features): `features_1` ~` features_4`
  - EmotiW_VGG_FER_Fintuning.npz (frame-level features): `features_1` ~` features_9`
  - EmotiW_C3D_features_all.npz (segment-level features): `features_1`
  - EmotiW_Inception_features_Faces.npz (frame-level features): `features_1`
  - EmotiW_Inception_features_Origin.npz (frame-level features): `features_1`
  - EmotiW_traditional_descriptor.npz (video-level features): `HOG`, `LBP`, `HOGLBP`, `DSIFT`
  - EmotiW_EmotiW2017_LBPTOP.npz (video-level features): `LBPTOP` (不是每个视频都有LBPTOP)
  - EmotiW_Pose.npz (frame-level features): `poses`, `posesDev`
  - EmotiW_Identify.npz (frame-level features): `identifies`
- auditory features:
  - EmotiW_audio.npz (frame-level features): `mfcc`, `soundnet`,`IS10`, `Egemaps `
  - EmotiW_EmotiW2017_features.npz (frame-level features): `English`, `Chinese`
  - EmotiW_EmotiW2017_features.npz (utterance-level features): `Egmaps`
  - EmotiW_IS091113.npz (utterance-level features): `IS09` ,`IS11`, `IS13`
  - EmotiW_youtubeAudio.npz (segment-level features): `logspec`, `emb`, `qemb`
- textual features:
  - EmotiW_TFIDF.npz (utterance-level features): `TFIDF`
  - EmotiW_TFIDFNoEmpty.npz (utterance-level features): TFIDFNoEmpty（不是每句话都有）
  - EmotiW_txt_vec.npz (utterance-level features): word2vec（不是每句话都有）

`--temporal_`：

- Video: whether data_path['pic_path'] refers to frame path or video path
- max, mean, FV_N_K, _None: methods for encoding frame-level features into vidoe-level features

`--model`:

- linearSVC, RBFSVC, RF, LR, NN: ignore temporal information
- LSTM: consider temporal information
- 'NetVLAD', 'NetRVLAD', 'SoftDBoW', 'NetFV': realized through classifier_lp.py. [reference](https://github.com/antoine77340/LOUPE)

```sh
# train all models through one file
sh run_train.sh 1 # 基于train_eval_test_EmotiW2017_filter.npz classifier_1
```

## 8. Prediction

In this section, we gain predictions for train_datas, eval_datas and test_datas, respectively. And results are saved as .pth file into `classifier_1/train_result`, `classifier_1/eval_result` and  `classifier_1/test_result` , respectively.

Source code is in `EmotiW2018/predict.py`, which has two different calling methods.

```sh
# method 1: gain prediction for single model
python predict.py main --label_type=1 --model_path='classifier_1/EmotiW_VGG_FER_Fintuning_features_1_mean_RF_0.425267993874'

# method 2: gain all predictions of all models in 'classifier_1/'
python predict.py main_all_classifiers --label_type=1
```

## 9. Fusion

In this section, we gain fusion parameters and final submission.

Source code is in `EmotiW2018/fusion_predict.py`, which has following parameters:

- `--label_type`: change `label_root` and `classifier_save_root` in `config.py`
- `--fusion_type`: 'gain_para' or 'gain_submit'
- `--best_para_path`: save path for 'gain_para'  and call path for 'gain_submit'
- `--max_epoch`: number of interations in 'gain_para'

```sh
# step1 1: Gain fusion parameters
python fusion_predict.py main --label_type=1 --fusion_type='gain_para' --max_epoch=100 --best_para_path='1_best_para.npz'

# step 2: Gain submission files
python fusion_predict.py  main --label_type=1 --fusion_type='gain_submit' --best_para_path='1_best_para_58.81.npz'
```
