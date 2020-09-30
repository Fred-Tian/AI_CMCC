# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:35:01 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import time

start = time.clock()

data_processed = pd.read_csv('data_processed.csv')
X = data_processed[['告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
       '告警7', '告警8', '告警9']].values
y = data_processed[['故障_传输故障','故障_动环故障', '故障_电力故障', '故障_硬件故障',
       '故障_误告警', '故障_软件故障']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
gbdt_clf = GradientBoostingClassifier(max_depth=5, n_estimators=150, learning_rate=0.05,min_samples_leaf= 310, min_samples_split=2 ,
                                      subsample=0.85,max_features= 8,random_state=33)
clf = OneVsRestClassifier(gbdt_clf)
#param_test = {'estimator__min_samples_leaf':range(200,360,10)}
param_test = {'estimator__max_depth':range(2,3,1)}
#param_test = {'estimator__n_estimators':range(110,180,10)}
#param_test = {'estimator__subsample':[0.6,0.65,0.7,0.75,0.8,0.85,0.9]}
#param_test = {'estimator__max_features':range(5,11,1)}
gsearch = GridSearchCV(estimator=clf, param_grid=param_test, scoring='roc_auc', iid=False, cv=4)                    
gsearch.fit(X_train,y_train)
(gsearch.cv_results_)['mean_test_score']
print(gsearch.best_score_)
print(gsearch.best_params_)

#clf = OneVsRestClassifier(gbdt_clf)
#clf.fit(X_train,y_train)
#y_test_pred = clf.predict_proba(X_test)
#print(roc_auc_score(y_test,y_test_pred))
#print(accuracy_score(y_test,clf.predict(X_test)))

#scores = cross_validate(clf,X,y,cv=4,return_train_score=True,scoring='accuracy')
#print(scores['test_score'].mean())
#print(scores['train_score'].mean())

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
#y_train_len = (y_train.shape)[0]
#X_train_add = []
#y_train_add = []
#for i in range(y_train_len):
#    if(y_train[i,2]!=1):
#        X_train_add.append(X_train[i].tolist())
#        y_train_add.append(y_train[i].tolist())
#X_train = np.concatenate((X_train,np.array(X_train_add)),axis = 0)
#y_train = np.concatenate((y_train,np.array(y_train_add)),axis = 0)
#X_train,y_train = shuffle(X_train,y_train,random_state=33)
#
#clf.fit(X_train,y_train)
#score_test = clf.score(X_test,y_test)
#print(score_test)

#test = []
#train = []
#for i in range(10):
##    my_cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state=33)
##    X,y = shuffle(X,y,random_state=33)
#    gbdt_clf = GradientBoostingClassifier(max_depth=1+i, n_estimators=50, learning_rate=0.1,min_samples_leaf= 11, min_samples_split=12 ,random_state=33)
#    clf = OneVsRestClassifier(gbdt_clf)
#    
#    scores = cross_validate(clf,X,y,cv=4,return_train_score=True)
#  
#    test.append(scores['test_score'].mean())
#    train.append(scores['train_score'].mean())
#
#
#
#plt.plot(range(1,11),test,color='red',label = 'test')
#plt.plot(range(1,11),train,color = 'blue', label = 'train')
#plt.plot(range(1,11),[0.767225] * 10,color = 'yellow', label = 'baseline')
#plt.legend()
#plt.show()


print(time.clock() - start)