#coding:utf8
import os
import cv2
import json
import time
import tqdm
import glob
import shutil
import argparse
import numpy as np

import loupe as lp
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from config import opt

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# datas: [N, D] -> [N, len_norm]
def features_norm(datas, len_norm):
    if not isinstance(datas,list): datas = datas.tolist()
    N, D = np.shape(datas)
    if N>len_norm:
        return datas[:len_norm]
    else:
        while np.shape(datas)[0]!= len_norm: datas.append(datas[-1])
        return datas

def read_datas(type_, temporal_, name_2_features):
    len_=0
    datas = []
    labels = []
    train_datas = np.load(opt.label_root)[type_]
    for item in train_datas:
        pic_features = []
        pic_path = item['pic_path'] # video name
        pic_label = item['label']
        
        # extract features
        if temporal_.find('Video')!=-1: # 'Video_None'
            pic_features = name_2_features[pic_path]
        else:
            for frame_path in glob.glob('%s/%s/*' %(opt.pic_root, pic_path)):
                frame_path = "/".join(frame_path.split('/')[-2:])
                pic_features.append(name_2_features[frame_path])

        # fusion methods
        if temporal_.find('max')!=-1:
            features = np.max(pic_features, axis=0)
            datas.append(features)
            labels.append(pic_label)
        if temporal_.find('mean')!=-1:
            features = np.mean(pic_features, axis=0)
            datas.append(features)
            labels.append(pic_label)
        if temporal_.find('FV')!=-1: # 'FV_K'
            K=int(temporal_.split('_')[-1]) # 1,2,4,8,16
            N=int(temporal_.split('_')[-2]) # 100
            D=np.shape(pic_features)[-1]
            features = fv_encoder(pic_features, N, K, D)
            datas.append(features) # 2*K*D
            labels.append(pic_label)
        if temporal_.find('None')!=-1:
            features = pic_features
            datas.append(features)
            labels.append(pic_label)

    return datas, labels

def RNN(tensor_input, weights, biases):
    if opt.model == 'NetVLAD':
        NetVLAD = lp.NetVLAD(feature_size=opt.in_features, max_samples=opt.seq_len, cluster_size=opt.cluster_size, 
                         output_dim=opt.output_dim, gating=True, add_batch_norm=True,
                         is_training=True)
    if opt.model == 'NetRVLAD':
        NetVLAD = lp.NetRVLAD(feature_size=opt.in_features, max_samples=opt.seq_len, cluster_size=opt.cluster_size, 
                         output_dim=opt.output_dim, gating=True, add_batch_norm=True,
                         is_training=True)
    if opt.model == 'SoftDBoW':
        NetVLAD = lp.SoftDBoW(feature_size=opt.in_features, max_samples=opt.seq_len, cluster_size=opt.cluster_size, 
                         output_dim=opt.output_dim, gating=True, add_batch_norm=True,
                         is_training=True)
    if opt.model == 'NetFV':
        NetVLAD = lp.NetFV(feature_size=opt.in_features, max_samples=opt.seq_len, cluster_size=opt.cluster_size, 
                         output_dim=opt.output_dim, gating=True, add_batch_norm=True,
                         is_training=True)
    reshaped_input = tf.reshape(tensor_input, [-1, opt.in_features])
    tensor_output = NetVLAD.forward(reshaped_input)
    results = tf.matmul(tensor_output, weights['out']) + biases['out'] # [batch_size, n_classes]
    return results

def train_NN(X_train, y_train, X_val, y_val):
    lr = opt.lr
    opt.batch_size, opt.seq_len, opt.in_features = np.shape(X_train)
    opt.num_classes = 7 # classifier for EmotiW
    features_path = opt.features_path.split('/')[-1].split('.')[0] # EmotiW_audio
    features_name = features_path + '_' + opt.features_name # EmotiW_audio_mfcc
    prefix = 'classifier/' + features_name + '_' + opt.temporal_ + '_' + opt.model + '_'

    # optimize
    x = tf.placeholder(tf.float32, shape=(None, opt.seq_len, opt.in_features))
    y = tf.placeholder(tf.float32, shape=(None, opt.num_classes))
    weights = {'out': tf.Variable(tf.random_normal([opt.output_dim, opt.num_classes]))}
    biases = {'out': tf.Variable(tf.constant(0.1, shape=[opt.num_classes, ]))}
    pred = RNN(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    # gain acc
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # train and save the best model
    saver = tf.train.Saver(max_to_keep=1)
    best_score = 0
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # batch_xs: [batch_size, seq_len, in_features]
        for epoch in range(opt.max_epoch):
            sess.run([train_op], feed_dict={x: X_train, y: y_train}) # train on X_train
            acc = sess.run(accuracy, feed_dict={x: X_val, y: y_val}) # test on X_val
            now_score = acc
            print 'best_score:%s,   now_score=%s' %(str(best_score), str(now_score))
            if best_score < now_score:
                best_score = now_score
                saver.save(sess, prefix+str(best_score))
    return best_score


def test_NN(X_val, y_val, model_path):
    tf.reset_default_graph()
    opt.batch_size, opt.seq_len, opt.in_features = np.shape(X_val)
    opt.num_classes = 7 # classifier for EmotiW

   # optimize
    x = tf.placeholder(tf.float32, shape=(None, opt.seq_len, opt.in_features))
    y = tf.placeholder(tf.float32, shape=(None, opt.num_classes))
    weights = {'out': tf.Variable(tf.random_normal([opt.output_dim, opt.num_classes]))}
    biases = {'out': tf.Variable(tf.constant(0.1, shape=[opt.num_classes, ]))}
    pred = RNN(x, weights, biases)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        data_pred = sess.run(pred, feed_dict={x: X_val})
        acc = sess.run(accuracy, feed_dict={x: X_val, y: y_val})
        print acc
    return data_pred


def main(**kwargs):

    opt.parse(kwargs,print_=True)

    # read name_2_features map
    features = np.load(opt.features_path)[opt.features_name]
    name = np.load(opt.features_path)['data_path']
    name_2_features = {}
    for ii in range(len(name)): name_2_features[name[ii]['pic_path']] = features[ii]

    # read datas
    X_train, y_train = read_datas(type_='train_datas', temporal_=opt.temporal_, name_2_features=name_2_features)
    X_val, y_val = read_datas(type_='eval_datas', temporal_=opt.temporal_, name_2_features=name_2_features)

    # train
    if opt.model in ['NetVLAD', 'NetRVLAD', 'SoftDBoW', 'NetFV']:
        # change X_train to X_train_norm: [?, seq, feats]
        X_train_norm = np.array([features_norm(_, 100) for _ in X_train])
        X_val_norm = np.array([features_norm(_, 100) for _ in X_val])

        batch_1, seq, feat_dim = np.shape(X_train_norm)
        X_train_norm = X_train_norm.reshape(-1, feat_dim)
        X_val_norm = X_val_norm.reshape(-1, feat_dim)
        scaler = preprocessing.StandardScaler().fit(X_train_norm)
        scaler.transform(X_train_norm)
        scaler.transform(X_val_norm)
        X_train_norm = preprocessing.normalize(X_train_norm, norm='l2')
        X_val_norm = preprocessing.normalize(X_val_norm, norm='l2')
        X_train_norm = X_train_norm.reshape(-1, seq, feat_dim)
        X_val_norm = X_val_norm.reshape(-1, seq, feat_dim)

        # change y_train to y_train_norm:[?, 7]
        y_train_norm = np.zeros((len(y_train), 7))
        for ii, _ in enumerate(y_train): y_train_norm[ii][_]=1
        y_val_norm = np.zeros((len(y_val), 7))
        for ii, _ in enumerate(y_val): y_val_norm[ii][_]=1

        best_score = train_NN(X_train_norm, y_train_norm, X_val_norm, y_val_norm)
        #y_val_pred = test_NN(X_val_norm, y_val_norm, str(best_score))
        

if __name__ == '__main__':
    import fire
    fire.Fire()