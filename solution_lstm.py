# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:23:31 2019

@author: x270
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Mydataset import Mydataset
import time

start = time.clock()
tf.reset_default_graph()
#基础参数设置
BATCH_SIZE = 64
TIME_STEP = 23
INPUT_SIZE = 1
OUTPUT_SIZE = 6
CELL_SIZE = 64
LR = 0.01
IS_TRAINING = 1

##数据读入
#data_processed = pd.read_csv('data_processed_future5_last15_L.csv')
#X = (data_processed[['基站','告警-1','告警-2','告警-3','告警-4','告警-5','告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
#       '告警7', '告警8', '告警9']].values).astype(np.float32)
#X = X/87
#X[:,0] = X[:,0]/max(X[:,0])
#y = pd.get_dummies(data_processed['故障']).values
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=33)

#数据读入2
data_processed = pd.read_csv('data_processed_future5_last15_L_E.csv')
X = data_processed[['基站','告警-1','告警-2','告警-3','告警-4','告警-5','告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
       '告警7', '告警8', '告警9','上一次故障']].values.astype(np.float32)
y = data_processed['故障'].values
#添加error feature
error_total = np.zeros((int(max(X[:,0]))+1,6),np.float32)
X_train_o,X_test_o,y_train_o,y_test_o = train_test_split(X,y,test_size=0.20,random_state=33)
X_train = np.zeros((X_train_o.shape[0],X_train_o.shape[1]+6),np.float32)
X_test = np.zeros((X_test_o.shape[0],X_test_o.shape[1]+6),np.float32)
y_train = y_train_o
y_test = y_test_o
for i in range(X_train_o.shape[0]):
    error_total[int(X_train_o[i,0]),y_train_o[i]] += 1

l = X_train_o.shape[1]
for i in range(X_train.shape[0]):
    X_train[i] = np.append(X_train_o[i],error_total[int(X_train_o[i,0])])
    X_train[i,l + y_train_o[i]] -= 1

for i in range(X_test.shape[0]):
    X_test[i] = np.append(X_test_o[i],error_total[int(X_test_o[i,0])])

y = pd.get_dummies(data_processed['故障']).values
_,_,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=33)

X_train[:,1:16] = X_train[:,1:16]/87
X_train[:,0] = X_train[:,0]/9036
X_test[:,1:16] = X_test[:,1:16]/87
X_test[:,0] = X_test[:,0]/9036

train_set = Mydataset(X_train,y_train)

#tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP*INPUT_SIZE],name = 'tf_x')
tf_x_3D = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE],name = 'tf_x_3D')
tf_y = tf.placeholder(tf.int32, [None,OUTPUT_SIZE],name = 'tf_y')

#RNN
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=CELL_SIZE)
outputs, (h_c,h_n) = tf.nn.dynamic_rnn(rnn_cell,tf_x_3D,initial_state = None,dtype = tf.float32,time_major = False)
output = tf.layers.dense(outputs[:,-1,:],OUTPUT_SIZE)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

labels=tf.argmax(tf_y,axis=1)
predictions=tf.argmax(output,axis=1)

#使用前需初始化局部变量
accuracy = tf.metrics.accuracy(labels=labels,predictions=predictions)[1]

correct_prediction = tf.equal(labels, predictions)
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
saver = tf.train.Saver()

if IS_TRAINING:
#    for var in tf.trainable_variables():
#        print(var.name)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    for step in range(20000):
        b_x,b_y = train_set.next_batch(BATCH_SIZE)
        _,loss_ = sess.run([train_op,loss],{tf_x:b_x,tf_y:b_y})
        if step % 200 == 0:
#            #连续使用两个accuracy也有问题
#            accuracy_test = sess.run(accuracy,{tf_x:X_test,tf_y:y_test})
#            accuracy_train = sess.run(accuracy,{tf_x:X_train,tf_y:y_train})
#            print('train loss: %.4f' %loss_,'| train accuracy: %.4f'%accuracy_train,'| test accuracy: %.4f'%accuracy_test)
            accuracy_train2 = sess.run(accuracy2,{tf_x:X_train,tf_y:y_train})
            accuracy_test2 = sess.run(accuracy2,{tf_x:X_test,tf_y:y_test})
            print('train loss: %.4f' %loss_,'| train accuracy: %.4f'%accuracy_train2,'| test accuracy: %.4f'%accuracy_test2)

    y_test_label = sess.run(labels,{tf_x:X_test,tf_y:y_test})
    y_test_pred = sess.run(predictions,{tf_x:X_test,tf_y:y_test})
#    accuracy_ = sess.run(accuracy,{tf_x:X_test,tf_y:y_test})
#    print('test accuracy: %.4f'%accuracy_) 
    save_path = saver.save(sess,'my_net\save_net.ckpt')
else:
    saver.restore(sess,'my_net\save_net.ckpt')
#    sess.run(tf.local_variables_initializer())
    accuracy_ = sess.run(accuracy2,{tf_x:X_test,tf_y:y_test})
    print('test accuracy: %.4f'%accuracy_)

print(time.clock() - start)