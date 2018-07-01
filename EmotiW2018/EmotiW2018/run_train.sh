echo 'training all models: '$1

## VGG_FER
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_1' --temporal_='mean' --model='RF' --max_epoch=1
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_2' --temporal_='mean' --model='RF' --max_epoch=1
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_4' --temporal_='max' --model='RF' --max_epoch=1
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_5' --temporal_='mean' --model='RF' --max_epoch=1
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_6' --temporal_='max' --model='RF' --max_epoch=1

python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50

python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_7' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_7' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_7' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_7' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_7' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_7' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_7' --temporal_='None_' --model='LSTM' --max_epoch=50

python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_8' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_8' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_8' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_8' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_8' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_8' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_VGG_FER_Fintuning.npz' --features_name='features_8' --temporal_='None_' --model='LSTM' --max_epoch=50

## DenseNet_FER
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_3' --temporal_='None_' --model='LSTM' --max_epoch=50

python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_2' --temporal_='max' --model='linearSVC' --max_epoch=1

python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_1' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_1' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_1' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_1' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_1' --temporal_='None_' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_DenseNet_FER_Fintuning.npz' --features_name='features_1' --temporal_='None_' --model='LSTM' --max_epoch=50

## C3D
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_C3D_features_all.npz' --features_name='features_1' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_C3D_features_all.npz' --features_name='features_1' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_C3D_features_all.npz' --features_name='features_1' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_C3D_features_all.npz' --features_name='features_1' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_C3D_features_all.npz' --features_name='features_1' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_C3D_features_all.npz' --features_name='features_1' --temporal_='Video_None' --model='LSTM' --max_epoch=50

## Inception+PCA (Faces)
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Origin.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Origin.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Origin.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Origin.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Origin.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Origin.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Origin.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50

## Inception+PCA (Origin_Faces)
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Faces.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Faces.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Faces.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Faces.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Faces.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Faces.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Inception_features_Faces.npz' --features_name='features_1' --temporal_='_None' --model='LSTM' --max_epoch=50


## Egemaps
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='Egemaps' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='Egemaps' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='Egemaps' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='Egemaps' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='Egemaps' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='Egemaps' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='Egemaps' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='Egemaps' --temporal_='Video_None' --model='LSTM' --max_epoch=50

## IS10
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='IS10' --temporal_='Video_max' --model='RF' --max_epoch=1

## mfcc
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='mfcc' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='mfcc' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='mfcc' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='mfcc' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='mfcc' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='mfcc' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='mfcc' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='mfcc' --temporal_='Video_None' --model='LSTM' --max_epoch=50

## Soundnet
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='soundnet' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='soundnet' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='soundnet' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='soundnet' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='soundnet' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='soundnet' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='soundnet' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_audio.npz' --features_name='soundnet' --temporal_='Video_None' --model='LSTM' --max_epoch=50

## LBPTOP
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_EmotiW2017_LBPTOP.npz' --features_name='LBPTOP' --temporal_='Video_None' --model='linearSVC' --max_epoch=1

## Egmaps(Video)
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_EmotiW2017_features.npz' --features_name='Egmaps' --temporal_='Video_None' --model='RF' --max_epoch=1

## English
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_EmotiW2017_features.npz' --features_name='English' --temporal_='Video_mean' --model='RF' --max_epoch=1

## Chinese
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_EmotiW2017_features.npz' --features_name='Chinese' --temporal_='Video_max' --model='RF' --max_epoch=1

## TFIDF
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_TFIDF.npz' --features_name='TFIDF' --temporal_='Video_None' --model='linearSVC' --max_epoch=1
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_TFIDF.npz' --features_name='TFIDF' --temporal_='Video_None' --model='RBFSVC' --max_epoch=1
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_TFIDF.npz' --features_name='TFIDF' --temporal_='Video_None' --model='RF' --max_epoch=1

## word2vec
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_txt_vec2.npz' --features_name='word2vec' --temporal_='Video_max' --model='linearSVC' --max_epoch=1
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_txt_vec2.npz' --features_name='word2vec' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_txt_vec2.npz' --features_name='word2vec' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_txt_vec2.npz' --features_name='word2vec' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_txt_vec2.npz' --features_name='word2vec' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_txt_vec2.npz' --features_name='word2vec' --temporal_='Video_None' --model='LSTM' --max_epoch=50
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_txt_vec2.npz' --features_name='word2vec' --temporal_='Video_None' --model='LSTM' --max_epoch=50

# LBP
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_traditional_descriptor.npz' --features_name='LBP' --temporal_='Video_None' --model='linearSVC' --max_epoch=1
	
# HOG
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_traditional_descriptor.npz' --features_name='HOG' --temporal_='Video_None' --model='RF' --max_epoch=1

# DSIFT
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_traditional_descriptor.npz' --features_name='DSIFT' --temporal_='Video_None' --model='RF' --max_epoch=1

# HOGLBP
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_traditional_descriptor.npz' --features_name='HOGLBP' --temporal_='Video_None' --model='RF' --max_epoch=1

# Identify
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Identify.npz' --features_name='identifies' --temporal_='Video_mean' --model='RF' --max_epoch=1

# Pose, PoseDev
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Pose.npz' --features_name='posesDev' --temporal_='Video_max' --model='linearSVC' --max_epoch=1
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_Pose.npz' --features_name='poses' --temporal_='Video_max' --model='linearSVC' --max_epoch=1

# IS09(video-level)
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_IS091113.npz' --features_name='IS09' --temporal_='Video_None' --model='RF' --max_epoch=1

# IS11(video-level)
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_IS091113.npz' --features_name='IS11' --temporal_='Video_None' --model='RF' --max_epoch=1

# IS13(video-level)
python classifier.py main --label_type=$1 --features_path='../EmotiW2017/For_EmotiW2018/EmotiW_IS091113.npz' --features_name='IS13' --temporal_='Video_None' --model='RF' --max_epoch=1
