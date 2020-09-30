# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:16:47 2019

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
BATCH_SIZE = 32
TIME_STEP = 15
INPUT_SIZE = 1
OUTPUT_SIZE = 2
CELL_SIZE = 32
LR = 0.01
IS_RESTORED = 1
alpha = 0.000001

#数据读入
data_processed = pd.read_csv('data_processed_future5_last15_L.csv')
#X = (data_processed[['基站','告警-1','告警-2','告警-3','告警-4','告警-5','告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
#       '告警7', '告警8', '告警9']].values).astype(np.float32)
X = (data_processed[['告警-5','告警-4','告警-3','告警-2','告警-1','告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
       '告警7', '告警8', '告警9']].values).astype(np.float32)
X = X/87
#X[:,0] = X[:,0]/max(X[:,0])
#y = data_processed['故障'].values
y = pd.get_dummies(data_processed['故障'],prefix = '故障')
#data_processed = pd.read_csv('data_processed.csv')
#X = data_processed[['告警0', '告警1', '告警2', '告警3', '告警4', '告警5', '告警6',
#       '告警7', '告警8', '告警9']].values/87
#y = data_processed[['故障_传输故障','故障_动环故障', '故障_电力故障', '故障_硬件故障',
#       '故障_误告警', '故障_软件故障']].values
yy = pd.get_dummies(y['故障_2']).values
X_train,X_test,y_train,y_test = train_test_split(X,yy,test_size=0.2,random_state=33)

#id2 = y_train[:,1]!=1
#div2 = True
#for i in range(id2.shape[0]):
#    if id2[i] == False:
#        if div2 == True:
#            id2[i] = True
#        div2 = not div2
#X_train = X_train[id2,:]
#y_train = y_train[id2,:]

train_set = Mydataset(X_train,y_train)

#tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP*INPUT_SIZE],name = 'tf_x')
tf_x_3D = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE],name = 'tf_x_3D')
tf_y = tf.placeholder(tf.int32, [None,OUTPUT_SIZE],name = 'tf_y')
tf_is_training = tf.placeholder(tf.bool,None,name = 'tf_is_training')

#RNN
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=CELL_SIZE)
outputs, (h_c,h_n) = tf.nn.dynamic_rnn(rnn_cell,tf_x_3D,initial_state = None,dtype = tf.float32,time_major = False)
outputs_dropout = tf.layers.dropout(outputs,rate=0,training=tf_is_training)
output = tf.layers.dense(outputs_dropout[:,-1,:],OUTPUT_SIZE)

#output_2 = tf.layers.dense(outputs_dropout[:,-1,:],12, tf.nn.relu)
#output = tf.layers.dense(output_2,OUTPUT_SIZE)

tv = tf.trainable_variables()
L2_reg = alpha * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)# + L2_reg
#loss = tf.losses.sigmoid_cross_entropy(tf_y, logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

labels=tf.argmax(tf_y,axis=1)
predictions=tf.argmax(output,axis=1)

#labels = tf_y
#predictions = tf.cast(output<0.5,tf.int32)
#使用前需初始化局部变量
accuracy = tf.metrics.accuracy(labels=labels,predictions=predictions)[1]

correct_prediction = tf.equal(labels, predictions)
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
saver = tf.train.Saver()

if IS_RESTORED:
#    for var in tf.trainable_variables():
#        print(var.name)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    for step in range(50000):
        b_x,b_y = train_set.next_batch(BATCH_SIZE)
        _,loss_ = sess.run([train_op,loss],{tf_x:b_x,tf_y:b_y,tf_is_training:True})
        if step % 200 == 0:
#            #连续使用两个accuracy也有问题
#            accuracy_test = sess.run(accuracy,{tf_x:X_test,tf_y:y_test})
#            accuracy_train = sess.run(accuracy,{tf_x:X_train,tf_y:y_train})
#            print('train loss: %.4f' %loss_,'| train accuracy: %.4f'%accuracy_train,'| test accuracy: %.4f'%accuracy_test)
            accuracy_train2 = sess.run(accuracy2,{tf_x:X_train,tf_y:y_train,tf_is_training:False})
            accuracy_test2 = sess.run(accuracy2,{tf_x:X_test,tf_y:y_test,tf_is_training:False})
            print('train loss: %.4f' %loss_,'| train accuracy: %.4f'%accuracy_train2,'| test accuracy: %.4f'%accuracy_test2)

#    y_test_label = sess.run(labels,{tf_x:X_test,tf_y:y_test})
#    y_test_pred = sess.run(predictions,{tf_x:X_test,tf_y:y_test})
    y_test_label = sess.run(labels,{tf_x:X_train,tf_y:y_train,tf_is_training:False})
    y_test_pred = sess.run(predictions,{tf_x:X_train,tf_y:y_train,tf_is_training:False})
    save_path = saver.save(sess,'my_net\save_net.ckpt')
else:
    saver.restore(sess,'my_net\save_net.ckpt')
#    sess.run(tf.local_variables_initializer())
    accuracy_ = sess.run(accuracy2,{tf_x:X_test,tf_y:y_test})
    print('test accuracy: %.4f'%accuracy_)

print(time.clock() - start)