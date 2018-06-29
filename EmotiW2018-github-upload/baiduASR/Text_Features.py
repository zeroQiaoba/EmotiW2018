'''
Target: Extract TFIDF features
------------------------------------------------
Input:      
    text
            
Output: 
    TFIDF_features .npy file
------------------------------------------------
Author: Robert Lian

Date: 2018/06/01  Init version
'''

import sys
import os
import glob
import logging
import numpy as np
from time import time
from optparse import OptionParser
from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

edict = {}
edict[0] = 'Angry'
edict[1] = 'Disgust'
edict[2] = 'Fear'
edict[3] = 'Happy'
edict[4] = 'Sad'
edict[5] = 'Surprise'
edict[6] = 'Neutral'

# reduce the size of `Word_Tabel`
def noralize_name(name_2_asr):
  name_2_asr_new = {}
  for name in name_2_asr:
    asr = name_2_asr[name]
    asr = asr.lower()
    asr_split = asr.split(' ')
    asr_split = [text for text in asr_split if text != '']
    for ii, txt in enumerate(asr_split):
      if txt.find('fuck')!=-1: asr_split[ii] = 'fuck'
      if txt.find('shit')!=-1: asr_split[ii] = 'shit'
      if txt=='holly': asr_split[ii] = 'holy'
    name_2_asr_new[name] = " ".join(asr_split)
  return name_2_asr_new

# if freq less than 3, then delete corresponded words
def noralize_word_table(word_table):
  word_tabel_new = {}
  word_table = sorted(word_table.items(),key=lambda x:x[1],reverse=True)
  for item in word_table:
    if item[1]>3: word_tabel_new[item[0]] = item[1]
  return word_tabel_new

## step 1: extract EmotiW.txt
def emotiW_extraction(asr_root='EmotiW_txt.npz'):
  
  data_path=np.load(asr_root)['data_path']
  texts=np.load(asr_root)['text']

  name_2_asr = {}
  for ii, item in enumerate(data_path):
    pic_path=item['pic_path']
    content=texts[ii]
    name_2_asr[pic_path] = content

  name_2_asr = noralize_name(name_2_asr)

  # gain word table
  word_table = {}
  for name in name_2_asr:
    asr = name_2_asr[name]
    asr_split = asr.split(' ')
    asr_split = [text for text in asr_split if text != '']
    for text in asr_split:
      if text not in word_table:
        word_table[text]=1
      else:
        word_table[text] += 1

  word_table = noralize_word_table(word_table)

  # chaneg tabel to index
  word_index = {}
  for ii, key in enumerate(word_table):
    word_index[key] = ii

  # change asr to index
  features = []
  names = []
  for name in name_2_asr:
    asr = name_2_asr[name]
    names.append({'pic_path': name})
    features.append(asr)

  np.savez_compressed('EmotiW_txt.npz',
                      data_path=names,
                      text=features,
                      word_index=word_index
                      )

###########################################################################
def extract_TFIDF():
  
  data_path = np.load('EmotiW_txt.npz')['data_path']
  texts = np.load('EmotiW_txt.npz')['text']
  word_index = np.load('EmotiW_txt.npz')['word_index']
  word_index = word_index.item() # 275
  vectorizer = TfidfVectorizer(sublinear_tf=True, vocabulary=word_index, stop_words='english')
  TFIDF = vectorizer.fit_transform(texts).toarray() # [1799, 275]
  np.savez_compressed('EmotiW_TFIDF.npz',
                      data_path=data_path,
                      TFIDF=TFIDF,
                      )

  ## del empty featrues
  texts_no_empty = []
  data_path_no_empty = []
  for ii, text in enumerate(texts):
    if text != '':
      texts_no_empty.append(text)
      data_path_no_empty.append(data_path[ii])

  TFIDFNoEmpty = vectorizer.fit_transform(texts_no_empty).toarray() # [1799, 275]
  np.savez_compressed('EmotiW_TFIDFNoEmpty.npz',
                      data_path=data_path_no_empty,
                      TFIDFNoEmpty=TFIDFNoEmpty,
                      )

if __name__ == '__main__':
  import fire
  fire.Fire()