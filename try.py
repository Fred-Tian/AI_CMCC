# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:08:12 2019

@author: x270
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate,StratifiedKFold
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import time

start = time.clock()

#data_processed = pd.read_csv('data_processed.csv')
#X = data_processed[['告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
#       '告警7', '告警8', '告警9']].values
#data_processed = pd.read_csv('data_processed_future3_last15.csv')
#X = data_processed[['告警-1','告警-2','告警-3','告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
#       '告警7', '告警8', '告警9','告警10','告警11','告警12','告警13','告警14']].values
#yy = data_processed[['故障_传输故障','故障_动环故障', '故障_电力故障', '故障_硬件故障',
#       '故障_误告警', '故障_软件故障']]
#
#yy.loc[:,'告警'] = 2**5*yy.loc[:,'故障_传输故障']+2**4*yy.loc[:,'故障_动环故障'] +2**3*yy.loc[:,'故障_电力故障']+2**2*yy.loc[:,'故障_硬件故障']+2**1*yy.loc[:,'故障_误告警']+yy.loc[:,'故障_软件故障'] 
#labelencoder_warning = LabelEncoder()
#yy.loc[:,'告警'] = labelencoder_warning.fit_transform(yy.loc[:,'告警'])   
#y = yy['告警'].values
                    
data_processed = pd.read_csv('data_processed_future5_last15_L_E.csv')
X = data_processed[['基站','告警-1','告警-2','告警-3','告警-4','告警-5','告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
       '告警7', '告警8', '告警9','上一次故障']].values
y = data_processed['故障'].values
#添加error feature
error_total = np.zeros((max(X[:,0])+1,6),np.int64)
X_train_o,X_test_o,y_train_o,y_test_o = train_test_split(X,y,test_size=0.20,random_state=33)
X_train = np.zeros((X_train_o.shape[0],X_train_o.shape[1]+6),np.int64)
X_test = np.zeros((X_test_o.shape[0],X_test_o.shape[1]+6),np.int64)
y_train = y_train_o
y_test = y_test_o
for i in range(X_train_o.shape[0]):
    error_total[X_train_o[i,0],y_train_o[i]] += 1

l = X_train_o.shape[1]
for i in range(X_train.shape[0]):
    X_train[i] = np.append(X_train_o[i],error_total[X_train_o[i,0]])
    X_train[i,l + y_train_o[i]] -= 1

for i in range(X_test.shape[0]):
    X_test[i] = np.append(X_test_o[i],error_total[X_test_o[i,0]])


def tun_parameters(train_X, train_y):
    xgbc = XGBClassifier(learning_rate = 0.1, n_estimators = 1000, max_depth = 4,
                         min_child_weight = 1, gamma = 0, subsample = 0.8,
                         colsample_bytree = 0.8, objective = 'multi:softprob',num_class = 6,
                         scale_pos_weight = 1, seed = 27,n_jobs = -1,reg_alpha = 0, reg_lambda = 1)
    modelfit(xgbc,train_X,train_y)

def modelfit(alg, X, y, useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label = y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'],
                          nfold = cv_folds, metrics = 'mlogloss', early_stopping_rounds = early_stopping_rounds)
        alg.set_params(n_estimators = cvresult.shape[0])
        print(cvresult)
        
    alg.fit(X, y)
    dtrain_predictions = alg.predict(X)
    
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, alg.predict(X_test)))
#    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))#multi-class not support auc
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending = False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    print('n_estimators = ', cvresult.shape[0])
#tun_parameters(X_train,y_train)

xgbc = XGBClassifier(learning_rate = 0.05, n_estimators = 2000, max_depth = 4,
                     min_child_weight = 1, gamma = 0, subsample = 0.7,
                     colsample_bytree = 0.8, objective = 'multi:softprob',num_class = 6,
                     scale_pos_weight = 1, seed = 27,reg_alpha = 0, reg_lambda = 1)

##xgbc.fit(X_train,y_train)
##y_test_pred = xgbc.predict(X_test)
##print(metrics.accuracy_score(y_test,y_test_pred))
##
#param_test = {'max_depth':range(2,9,1),
#             'min_child_weight':range(1,10,1)}
##param_test = {'n_estimators':range(80,180,10)}
##param_test = {'subsample':[0.6,0.7,0.8,0.9],
##              'colsample_bytree':[0.6,0.7,0.8,0.9]}
##param_test = {'gamma':[i / 10.0 for i in range(0, 10)]}
#gsearch = GridSearchCV(estimator=xgbc, param_grid=param_test, scoring='neg_log_loss', n_jobs = -1, iid=False, cv=5)                    
#gsearch.fit(X_train,y_train)
#(gsearch.cv_results_)['mean_test_score']
#print(gsearch.best_score_)
#print(gsearch.best_params_)



#X_train = X
#y_train = y
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
#y_test_pred_prod = np.zeros((y_test.shape[0],11),dtype = np.float)
y_train_pred_prod = np.zeros((X.shape[0],6), dtype = np.float)
for i,(tr,va) in enumerate(skf.split(X,y)):
    print("training %dth fold"%i)
    #添加error feature
    error_total = np.zeros((max(X[:,0])+1,6),np.int64)
    X_train_o,X_test_o,y_train_o,y_test_o = X[tr],X[va],y[tr],y[va]
    X_train = np.zeros((X_train_o.shape[0],X_train_o.shape[1]+6),np.int64)
    X_test = np.zeros((X_test_o.shape[0],X_test_o.shape[1]+6),np.int64)
    y_train = y_train_o
    y_test = y_test_o
    for i in range(X_train_o.shape[0]):
        error_total[X_train_o[i,0],y_train_o[i]] += 1
    l = X_train_o.shape[1]
    for i in range(X_train.shape[0]):
        X_train[i] = np.append(X_train_o[i],error_total[X_train_o[i,0]])
        X_train[i,l + y_train_o[i]] -= 1    
    for i in range(X_test.shape[0]):
        X_test[i] = np.append(X_test_o[i],error_total[X_test_o[i,0]])

    dtrain = xgb.DMatrix(X_train,y_train)
    dvalid = xgb.DMatrix(X_test,y_test)    
#    dtrain = xgb.DMatrix(X_train[tr],y_train[tr])
#    dvalid = xgb.DMatrix(X_train[va],y_train[va])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
    bst = xgb.train(dtrain=dtrain, num_boost_round=2000, evals=watchlist, 
                    verbose_eval=50, params=xgbc.get_xgb_params(),early_stopping_rounds = 100)
    y_train_pred_prod[va] += bst.predict(xgb.DMatrix(X_test), ntree_limit=bst.best_ntree_limit)
    print(bst.best_ntree_limit)
#    y_test_pred_prod += bst.predict(xgb.DMatrix(X_test), ntree_limit=bst.best_ntree_limit)
#print('the roc_auc_score for train:',metrics.accuracy_score(y_test,np.argmax(y_test_pred_prod,axis=1)))
print('the roc_auc_score for train:',metrics.accuracy_score(y,np.argmax(y_train_pred_prod,axis=1)))
print(time.clock() - start)