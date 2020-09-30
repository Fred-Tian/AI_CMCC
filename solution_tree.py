# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:28:49 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import time

start = time.clock()

data_processed = pd.read_csv('data_processed.csv')
X = data_processed[['告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
       '告警7', '告警8', '告警9']].values
y = data_processed[['故障_传输故障','故障_动环故障', '故障_电力故障', '故障_硬件故障',
       '故障_误告警', '故障_软件故障']].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

y_train_len = (y_train.shape)[0]
X_train_add = []
y_train_add = []
for i in range(y_train_len):
    if(y_train[i,2]!=1):
        X_train_add.append(X_train[i].tolist())
        y_train_add.append(y_train[i].tolist())
X_train = np.concatenate((X_train,np.array(X_train_add)),axis = 0)
y_train = np.concatenate((y_train,np.array(y_train_add)),axis = 0)
X_train,y_train = shuffle(X_train,y_train,random_state=33)

test = []
train = []
for i in range(10):
    clf = OneVsRestClassifier(DecisionTreeClassifier(splitter= 'best', max_depth= 5,min_samples_leaf = 301,min_samples_split=2))
    clf.fit(X_train,y_train)
    score_train = clf.score(X_train,y_train)
    score_test = clf.score(X_test,y_test)
    test.append(score_test)
    train.append(score_train)

#y_test_pred = clf.predict(X_test)
#y_train_pred = clf.predict(X_train)

plt.plot(range(1,11),test,color='red',label = 'test')
plt.plot(range(1,11),train,color = 'blue', label = 'train')
plt.plot(range(1,11),[0.767225] * 10,color = 'yellow', label = 'baseline')
plt.legend()
plt.show()

#print('train accuracy: %f'%accuracy_score(y_train_pred,y_train))
#print('base accuracy: %f'%accuracy_score(np.array([0,0,1,0,0,0]*(y_test.shape)[0]).reshape(y_test.shape),y_test))
#print('test accuracy: %f'%accuracy_score(y_test_pred,y_test))
print(time.clock() - start)