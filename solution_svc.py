# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:30:53 2019

@author: x270
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import time

start = time.clock()

data_processed = pd.read_csv('data_processed.csv')
#data_processed.loc[0:10000,'故障_传输故障'] = 1
X = data_processed[['告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
       '告警7', '告警8', '告警9']].values
y = data_processed[['故障_传输故障','故障_动环故障', '故障_电力故障', '故障_硬件故障',
       '故障_误告警', '故障_软件故障']].values
#y = data_processed[['故障_传输故障','故障_动环故障', '故障_硬件故障',
#       '故障_误告警', '故障_软件故障']].values
                    
X = X/87
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
clf = OneVsRestClassifier(SVC(kernel='linear',verbose=True))
#clf = OneVsRestClassifier(SVC(kernel='linear',class_weight={0:1,1:0.3},verbose=True))
clf.fit(X_train,y_train)
y_test_pred = clf.predict(X_test)

count = 0
for i in range((y_test.shape)[0]):
    if (y_test[i] == y_test_pred[i]).all():
        count+=1
print('accuracy: %f'%(count/(y_test.shape)[0]))
print(time.clock() - start)