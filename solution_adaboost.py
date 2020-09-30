# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:04:55 2019

@author: Administrator
"""



import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import time

start = time.clock()

data_processed = pd.read_csv('data_processed.csv')
X = data_processed[['告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
       '告警7', '告警8', '告警9']].values
y = data_processed[['故障_传输故障','故障_动环故障', '故障_电力故障', '故障_硬件故障',
       '故障_误告警', '故障_软件故障']].values

test = []
train = []
for i in range(10):
#    my_cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state=33)
    X,y = shuffle(X,y,random_state=33)
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(splitter= 'random', max_depth= 2,min_samples_leaf = 15), 
                                n_estimators= 10 + 10*i, algorithm='SAMME', learning_rate=0.2)
    clf = OneVsRestClassifier(ada_clf)
    
    scores = cross_validate(clf,X,y,cv=4,return_train_score=True)
  
    test.append(scores['test_score'].mean())
    train.append(scores['train_score'].mean())



plt.plot(range(1,11),test,color='red',label = 'test')
plt.plot(range(1,11),train,color = 'blue', label = 'train')
plt.plot(range(1,11),[0.767225] * 10,color = 'yellow', label = 'baseline')
plt.legend()
plt.show()


print(time.clock() - start)