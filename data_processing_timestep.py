# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:43:14 2019

@author: x270
"""

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
start = time.clock()
###读取数据
data_error = pd.read_csv('train_error.csv')
data_warning = pd.read_csv('train_warning.csv')

###查看是否有空值
#print(data_error.isna().sum())
#print(data_warning.isna().sum())
data_warning = data_warning.dropna()

###去除重复项
data_error.drop_duplicates(subset = ['故障发生时间','涉及基站eNBID或小区ECGI','故障原因定位（大类）'], inplace = True)
data_warning.drop_duplicates(inplace = True)

###格式化数据，故障时间转化为时间戳，单位s；标签格式化
data_error['故障发生时间'] = data_error['故障发生时间'].apply(lambda x: time.mktime(time.strptime(x,'%Y/%m/%d %H:%M')))
data_warning['告警开始时间'] = data_warning['告警开始时间'].apply(lambda x: time.mktime(time.strptime(x,'%Y/%m/%d %H:%M')))

labelencoder_bsid = LabelEncoder()
data_error['涉及基站eNBID或小区ECGI'] = labelencoder_bsid.fit_transform(data_error['涉及基站eNBID或小区ECGI'])
data_warning['基站eNBID或小区ECGI'] = labelencoder_bsid.transform(data_warning['基站eNBID或小区ECGI'])

labelencoder_error = LabelEncoder()
data_error['故障原因定位（大类）'] = labelencoder_error.fit_transform(data_error['故障原因定位（大类）'])
data_error = data_error.reset_index(drop = True)

labelencoder_warningid = LabelEncoder()
data_warning['告警标题'] = labelencoder_warningid.fit_transform(data_warning['告警标题'])

data_error = data_error.sort_values(by = '故障发生时间')
data_warning = data_warning.sort_values(by = '告警开始时间')

#data_error = data_error.reset_index(drop = True)
#data_warning = data_warning.reset_index(drop = True)

###合并数据
data = pd.DataFrame(columns = ['基站','故障时间','故障原因定位（大类）','过去告警标题及时间','未来告警标题及时间'])
index_id = 0
for i in range(max(data_error['涉及基站eNBID或小区ECGI']) + 1):
#for i in range(0,10):
    data_error_tmp = data_error[data_error['涉及基站eNBID或小区ECGI'] == i]
    data_warning_tmp = data_warning[data_warning['基站eNBID或小区ECGI'] == i]
    timestamp_before = 0
    for j in data_error_tmp.index:
        data_tmp = [data_error_tmp.loc[j,'涉及基站eNBID或小区ECGI'], data_error_tmp.loc[j,'故障发生时间'],
                    data_error_tmp.loc[j,'故障原因定位（大类）']]
#        warning_tmp = [data_warning_tmp.loc[k,['告警标题','告警开始时间']].values for k in data_warning_tmp.index 
#                       if data_warning_tmp.loc[k,'告警开始时间'] > timestamp_before and data_warning_tmp.loc[k,'告警开始时间'] <= data_error_tmp.loc[j,'故障发生时间']]
        warning_tmp = [[data_warning_tmp.loc[k,'告警标题'],(data_error_tmp.loc[j,'故障发生时间'] - data_warning_tmp.loc[k,'告警开始时间'])/(24*3600)] for k in data_warning_tmp.index 
                       if data_warning_tmp.loc[k,'告警开始时间'] > timestamp_before and data_warning_tmp.loc[k,'告警开始时间'] <= data_error_tmp.loc[j,'故障发生时间']]
        timestamp_before = data_error_tmp.loc[j,'故障发生时间']
        
        warning_future_tmp = [[data_warning_tmp.loc[k,'告警标题'],(data_error_tmp.loc[j,'故障发生时间'] - data_warning_tmp.loc[k,'告警开始时间'])/(24*3600)] for k in data_warning_tmp.index
                               if data_warning_tmp.loc[k,'告警开始时间'] > data_error_tmp.loc[j,'故障发生时间']]
        
        data_tmp.append(warning_tmp)
        data_tmp.append(warning_future_tmp)
        data.loc[index_id] = data_tmp
        index_id += 1
timestep_matrix = [0,0.3,0.6,1.2,1.8] 
num_warning = len(timestep_matrix) - 1
num_warning_future = len(timestep_matrix) - 1
num_day = 15
num_day_future = 5
columns = ['基站','故障']
for i in range(num_warning_future):
    for j in range(87):
        columns.append('告警'+str(-timestep_matrix[len(timestep_matrix)-1-i])+'_'+str(j))
for i in range(num_warning):
    for j in range(87):
        columns.append('告警'+str(timestep_matrix[i+1])+'_'+str(j))
data_processed = pd.DataFrame(columns = columns)
index_id = 0
for i in range(max(data.index) + 1):
    data_tmp = [data.loc[i,'基站'],data.loc[i,'故障原因定位（大类）']] + [0]*87*(len(timestep_matrix) - 1)*2
    for j in range(len(data.loc[i,'未来告警标题及时间'])):
        list_tmp = (data.loc[i,'未来告警标题及时间'])[j]   
        if -list_tmp[1] > timestep_matrix[-1]:
            break
        else:
            for k in range(len(timestep_matrix) - 1):
                if -list_tmp[1] > timestep_matrix[k] and -list_tmp[1] <= timestep_matrix[k+1]:
                    data_tmp[2 + (len(timestep_matrix) - 2 - k) * 87 + list_tmp[0]] = 1        
        
    for j in range(len(data.loc[i,'过去告警标题及时间'])):
        list_tmp = (data.loc[i,'过去告警标题及时间'])[-j-1]        
        if list_tmp[1] >= timestep_matrix[-1]:
            break
        else:
            for k in range(len(timestep_matrix) - 1):
                if list_tmp[1] >= timestep_matrix[k] and list_tmp[1] < timestep_matrix[k+1]:
                    data_tmp[2 + (len(timestep_matrix)-1)*87+ k*87 + list_tmp[0]] = 1
    data_processed.loc[index_id] = data_tmp
    index_id += 1
#data_processed = data_processed[(data_processed['告警0'] != 0) | (data_processed['告警-1'] != 0)]
data_processed = data_processed.reset_index(drop = True)
data_processed.to_csv('data_processed_timestep_03_06_12_18.csv')       
print(time.clock() - start)