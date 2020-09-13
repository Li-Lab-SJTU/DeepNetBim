#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:38:02 2020

@author: yangxiaoyun
"""

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

from MyEncode import blo_encode_920
from Closest_pep_net import closest_pep_net,mhc_net
from Model_ import ModelTrain
from Net_Dict_all import MHC_Net_Dict,Pep_Net_Dict

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
train_mhc,test_mhc = mhci.mhc[:split],mhci.mhc[split:]

model,train_log = ModelTrain(pattern = 'immunogenicity',epoch = 2,train_pep = peptides,train_network = network_feature,train_affini = None,train_cate = categorical_labels,\
      test_pep = test_pep,test_network = test_network,test_affini = None,test_cate=test_cate)


prediction =np.argmax(model.predict([np.array(test_pep),np.array(test_network)]),axis=1)
probability = [value[1] for value in model.predict([np.array(test_pep),np.array(test_network)])]

result = pd.DataFrame({'mhc':test_mhc,'pred_immuno':prediction.flatten(),'probability':probability,'true_immuno':test_cate})
result.to_csv('immuno_test.csv')

