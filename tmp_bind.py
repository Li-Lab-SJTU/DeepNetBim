#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:25:58 2020

@author: yangxiaoyun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:36:24 2020

@author: yangxiaoyun
"""


import numpy as np
import pandas as pd
import random 
from math import log
from keras.utils.np_utils import to_categorical
from MyEncode import blo_encode_920
from Closest_pep_net import closest_pep_net,mhc_net
from Model_ import ModelTrain
from Net_Dict_all import MHC_Net_Dict,Pep_Net_Dict

pep_length = 9
f = lambda x: len(x)

random.seed(1234)

mhci = pd.read_csv('data/bind_train.csv')
mhci = mhci.sample(frac=1).reset_index(drop=True)
mhci['peptide_length'] = mhci.sequence.apply(f)
mhci = mhci[mhci.peptide_length == pep_length]

mhci['affinity'] = [1-log(value)/log(50000) for value in mhci.affinity_raw]
mhci.affinity = [0 if value<=0 else value for value in mhci.affinity]
immunity_bin=[1 if value !='Negative' else 0 for value in mhci['Binding.category']]

categorical_labels = to_categorical(immunity_bin, num_classes=None)


pep_net_dict = Pep_Net_Dict(mhci)
mhc_net_dict = MHC_Net_Dict(mhci)

   
network_feature  = []
peptides = []
for mhc,pep in zip(mhci.mhc,mhci.sequence):
    network_feature.append(mhc_net_dict[mhc].tolist()+pep_net_dict[pep].tolist())
    peptides.append(blo_encode_920(pep))
    
    
network_feature = np.array(network_feature).reshape(-1,8,1)

split =int(len(peptides)*0.7)
train_pep,test_pep = peptides[:split],peptides[split:]
train_network,test_network = network_feature[:split],network_feature[split:]
train_cate,test_cate = categorical_labels[:split],categorical_labels[split:]
train_affini,test_affini = mhci.affinity[:split],mhci.affinity[split:]
train_mhc,test_mhc = mhci.mhc[:split],mhci.mhc[split:]

model,train_log = ModelTrain(pattern = 'bind',epoch = 2,train_pep = peptides,train_network = network_feature,train_affini = mhci.affinity,train_cate = categorical_labels,\
      test_pep = test_pep,test_network = test_network,test_affini = test_affini,test_cate=test_cate)

model.save('my_model_allin_0507_60.h5')


prediction = model.predict([np.array(test_pep),np.array(test_network)])

result = pd.DataFrame({'mhc':test_mhc,'pred_affinity':prediction.flatten(),'true_affinity':test_affini})
result.to_csv('bind_predict_test.csv')