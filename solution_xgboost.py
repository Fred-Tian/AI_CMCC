# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:34:09 2019

@author: x270
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import time

start = time.clock()

data_processed = pd.read_csv('data_processed.csv')
X = data_processed[['告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
       '告警7', '告警8', '告警9']].values
y = data_processed['故障_传输故障'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

def tun_parameters(train_X, train_y):
    xgbc = XGBClassifier(learning_rate = 0.1, n_estimators = 300, max_depth = 5,
                         min_child_weight = 1, gamma = 0, subsample = 0.8,
                         colsample_bytree = 0.8, objective = 'binary:logistic',
                         scale_pos_weight = 1, seed = 27)
    modelfit(xgbc,train_X,train_y)

def modelfit(alg, X, y, useTrainCV = True, cv_folds = 4, early_stopping_rounds = 50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label = y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'],
                          nfold = cv_folds, metrics = 'auc', early_stopping_rounds = early_stopping_rounds)
        alg.set_params(n_estimators = cvresult.shape[0])
        print(cvresult)
        
    alg.fit(X, y)
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]
    
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending = False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    print('n_estimators = ', cvresult.shape[0])
tun_parameters(X_train,y_train)

#param_test1={'max_depth':range(3,10,1),
#             'min_child_weight':range(1,6,1)}
#gsearch1 = GridSearchCV(estimator=,
#                        param_grid=param_test1,scoring='roc_auc',n_jobs=-1,iid=False,cv=5)