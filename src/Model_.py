#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:57:41 2020

@author: yangxiaoyun
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from random import randint,sample
#from sklearn.preprocessing import Imputer 
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import scipy.sparse.linalg
from sklearn.metrics import mean_absolute_error
import scipy.sparse as sps
# to render the graphs
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
#import theano
import keras
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam,Adadelta
from keras.layers import Add, Dense, Dot, Flatten, Input, Lambda, RepeatVector

def ModelTrain(train_pep,train_network,train_affini,train_cate,\
               test_pep,test_network,test_affini,test_cate,pattern = 'bind',epoch = 200,pep_length = 9):
    
    filters, fc1_size, fc2_size, fc3_size= 256, 256, 64, 4
    kernel_size = 2
    models = []
    inputs_1 = Input(shape = (pep_length,21))
    inputs_3 = Input(shape = (8,1))
    pep_conv = Conv1D(filters,kernel_size,padding = 'same',activation = 'relu',strides = 1)(inputs_1)
    pep_conv = Dropout(0.7)(pep_conv)
    pep_maxpool = MaxPooling1D(pool_size=1)(pep_conv)


    flk_conv = Conv1D(filters,kernel_size,padding = 'same',activation = 'relu',strides = 1)(inputs_3)
    flk_conv = Dropout(0.7)(flk_conv)
    flk_maxpool = MaxPooling1D(pool_size=1)(flk_conv)


    flat_pep_0 = Flatten()(pep_conv)
    flat_pep_1 = Flatten()(pep_conv)
    flat_pep_2 = Flatten()(pep_conv)
    flat_pep_3 = Flatten()(pep_conv)
    flat_flk_0 = Flatten()(flk_conv)
    flat_flk_1 = Flatten()(flk_conv)
    flat_flk_2 = Flatten()(flk_conv)
    flat_flk_3 = Flatten()(flk_conv)
    
    cat_0 = Concatenate()([flat_pep_0, flat_flk_0])
    cat_1 = Concatenate()([flat_pep_1, flat_flk_1])
    cat_2 = Concatenate()([flat_pep_2, flat_flk_2]) 
    cat_3 = Concatenate()([flat_pep_3, flat_flk_3]) 
    fc1_0 = Dense(fc1_size,activation = "relu")(cat_0)
    fc1_1 = Dense(fc1_size,activation = "relu")(cat_1)
    fc1_2 = Dense(fc1_size,activation = "relu")(cat_2)
    fc1_3 = Dense(fc1_size,activation = "relu")(cat_3)
    
    merge_1 = Concatenate()([fc1_0, fc1_1, fc1_2,fc1_3])
#    merge_1 = Dropout(0.2)(merge_1)
    fc2 = Dense(fc2_size,activation = "relu")(merge_1)
    fc3 = Dense(fc3_size,activation = "relu")(fc2)
    pep_attention_weights = Flatten()(TimeDistributed(Dense(1))(pep_conv))
    flk_attention_weights = Flatten()(TimeDistributed(Dense(1))(flk_conv))
    pep_attention_weights = Activation('softmax')(pep_attention_weights)  
    flk_attention_weights = Activation('softmax')(flk_attention_weights)  
    pep_conv_permute = Permute((2,1))(pep_conv)
    flk_conv_permute = Permute((2,1))(flk_conv)
    pep_attention = Dot(-1)([pep_conv_permute, pep_attention_weights])
    flk_attention = Dot(-1)([flk_conv_permute, flk_attention_weights])
    merge_2 = Concatenate()([pep_attention,flk_attention,fc3])
#    merge_2 = Dropout(0.2)(merge_2)
    
    if pattern == 'bind':
        out = Dense(1,activation = "sigmoid")(merge_2)
        model = Model(inputs=[inputs_1, inputs_3],outputs=out)
#        adadelta = keras.optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.005), metrics=['mse'])
        
        train_log= model.fit([np.array(train_pep),np.array(train_network)], np.array(train_affini), batch_size=256,\
               epochs = epoch,validation_data=([np.array(test_pep),np.array(test_network)],np.array(test_affini)),verbose =0 )
    if pattern == 'immunogenicity':
        out = Dense(2,activation = "sigmoid")(merge_2)
        model= Model(inputs=[inputs_1, inputs_3],outputs=out)  
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
        
        train_log = model.fit([np.array(train_pep),np.array(train_network)], np.array(train_cate), batch_size=256,\
                   epochs = epoch,validation_data=([np.array(test_pep),np.array(test_network)],np.array(test_cate)),verbose =0 )
            
    
    return(model,train_log)