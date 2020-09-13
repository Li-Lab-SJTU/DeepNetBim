#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:52:05 2020

@author: yangxiaoyun
"""

import numpy as np
import pandas as pd
import os
from numpy import *
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from random import randint,sample
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score  
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
import seaborn as sns
import scipy.sparse.linalg
from sklearn.metrics import mean_absolute_error
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
import gc
import pickle
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from math import log
from keras.utils.np_utils import to_categorical
from keras import optimizers
import scipy.stats as ss
import keras
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add, Dense, Dot, Flatten, Input, Lambda, RepeatVector
from collections import Counter
from keras.models import load_model

#os.chdir('/Users/yangxiaoyun/Downloads/研究课题/script/deephlapan')

from MyEncode import blo_encode_920
from Closest_pep_net import closest_pep_net,mhc_net
from Model_ import ModelTrain
#from Net_Dict import MHC_Net_Dict,Pep_Net_Dict
from Net_Dict_all import MHC_Net_Dict,Pep_Net_Dict

os.chdir('/lustre/home/acct-clslj/clslj-6/replicate/')

pep_length = 9
f = lambda x: len(x)

random.seed(1234)

mhci = pd.read_csv('immuno_all.csv',index_col = 0)
mhci = mhci.sample(frac=1).reset_index(drop=True)
mhci['peptide_length'] = mhci.sequence.apply(f)
mhci = mhci[mhci.peptide_length == pep_length]

immunity_bin=[1 if value ==1 else 0 for value in mhci['Label']]

categorical_labels = to_categorical(immunity_bin, num_classes=None)


pep_net_dict = Pep_Net_Dict(mhci)
mhc_net_dict = MHC_Net_Dict(mhci)

    
network_feature  = []
peptides = []
for mhc,pep in zip(mhci.mhc,mhci.sequence):
    network_feature.append(mhc_net_dict[mhc].tolist()+pep_net_dict[pep].tolist())
    peptides.append(blo_encode_920(pep))
network_feature = np.array(network_feature).reshape(-1,8,1)

split =int(floor(len(peptides)*0.7))
train_pep,test_pep = peptides[:split],peptides[split:]
train_network,test_network = network_feature[:split],network_feature[split:]
train_cate,test_cate = categorical_labels[:split],categorical_labels[split:]
#train_affini,test_affini = mhci.affinity[:split],mhci.affinity[split:]
'''
model,train_log = ModelTrain(pattern = 'immunogenicity',epoch = 200,train_pep = train_pep,train_network = train_network,train_affini = None,train_cate = train_cate,\
      test_pep = test_pep,test_network = test_network,test_affini = None,test_cate=test_cate)
'''
model,train_log = ModelTrain(pattern = 'immunogenicity',epoch = 200,train_pep = peptides,train_network = network_feature,train_affini = None,train_cate = categorical_labels,\
      test_pep = test_pep,test_network = test_network,test_affini = None,test_cate=test_cate)



#model.save('my_model_immuno_split_0513_200.h5')


#output = open('log_9_8_immuno_allin_0517_200.pkl','wb')
#pickle.dump(train_log,output)
#output.close()

model.save('model_9_8_immuno_allin_0523_200_5.h5')

'''

plt.figure(figsize=(12,9))
plt.plot(train_log.history['categorical_accuracy'],color='skyblue',label = 'train',lw = 3)
plt.plot(train_log.history['val_categorical_accuracy'],color = 'orange',label ='test',lw = 3)


plt.legend(fontsize = 12)
plt.xlabel('Epoch',fontsize = 12)
plt.ylabel('Accuracy',fontsize = 12)

plt.savefig('immuno_allin_0517_200.png')





pkl_file = open('model_9_8_immuno_shuffle.pkl','rb')
model = pickle.load(pkl_file)
pkl_file.close()

model = load_model('model_9_8_immuno_allin_0517_200.h5')
'''
##test on independent data
mhci_response = pd.read_csv('kosalogu.csv')
#mhci_response = pd.read_csv('CD8responseA.csv')

pep_length = 9
f = lambda x: len(x)
mhci_response['peptide_length'] = mhci_response.sequence.apply(f)
mhci_response = mhci_response[mhci_response.peptide_length == pep_length].reset_index(drop = True)


peptide = {}
network_pep = {}
network_mhc = {}
for pep in list(set(mhci_response.sequence)):
    peptide[pep] = blo_encode_920(pep)
    network_pep[pep] = closest_pep_net(pep,pep_net_dict)
    
for mhc in list(set(mhci_response.mhc)):
    network_mhc[mhc] = mhc_net(mhc,mhci,mhc_net_dict)

network = []
peptides = []
pep_dist = []
mhc_class = []
for pep,mhc in zip(mhci_response.sequence,mhci_response.mhc):
    network.append(np.array(network_pep[pep][:4]+network_mhc[mhc][:4]).reshape(8,1))
    pep_dist.append(network_pep[pep][4])
    peptides.append(peptide[pep])
    mhc_class.append(network_mhc[mhc][4])

prediction =np.argmax(model.predict([np.array(peptides),np.array(network)]),axis=1)
probability = [value[1] for value in model.predict([np.array(peptides),np.array(network)])]

result = pd.DataFrame({'mhc':mhci_response.mhc,'sequence':mhci_response.sequence,\
                   'pred_immuno':prediction.flatten(),'probability':probability,'response':mhci_response.response,\
                   'pep_dist':pep_dist,'mhc_class':mhc_class})


result.to_csv('kosalogu_result_immuno_allin_5.csv')
#result.to_csv('CDA_result_immuno_allin_0523_200.csv')

threshold = 1-log(float(500))/log(50000)
y_true_bind = [1 if value =='positive' else 0 for value in result.response]
#y_true_bind = [1 if value ==True else 0 for value in result.response]
y_pred_bind = result.pred_immuno
con = confusion_matrix(y_true_bind,y_pred_bind)
print(con)
print(accuracy_score(y_true_bind,y_pred_bind))
print(f1_score(y_true_bind,y_pred_bind))
print(recall_score(y_true_bind,y_pred_bind))
print(con[1,1]/(con[1,1]+con[0,1]))
