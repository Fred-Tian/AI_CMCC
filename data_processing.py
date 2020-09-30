# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:38:19 2019

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

#onehotencoder_errorid = OneHotEncoder()
#data_error = data_error.join(onehotencoder_errorid.fit_transform(data_error['故障原因定位（大类）'].values.reshape(-1,1)).toarray())
data_error = data_error.join(pd.get_dummies(data_error['故障原因定位（大类）'],prefix = '故障'))

dict_mapping = {'故障发生时间':'mean','涉及基站eNBID或小区ECGI':'mean','故障_传输故障':'sum','故障_动环故障':'sum',
                '故障_电力故障':'sum','故障_硬件故障':'sum','故障_误告警':'sum','故障_软件故障':'sum'}
data_error = data_error.groupby(by = ['故障发生时间','涉及基站eNBID或小区ECGI']).agg(dict_mapping)
data_error = data_error.reset_index(drop = True)

labelencoder_warningid = LabelEncoder()
data_warning['告警标题'] = labelencoder_warningid.fit_transform(data_warning['告警标题'])

data_error = data_error.sort_values(by = '故障发生时间')
data_warning = data_warning.sort_values(by = '告警开始时间')

#data_error = data_error.reset_index(drop = True)
#data_warning = data_warning.reset_index(drop = True)

###合并数据
data = pd.DataFrame(columns = ['基站','故障时间','故障_传输故障','故障_动环故障','故障_电力故障',
                               '故障_硬件故障','故障_误告警','故障_软件故障','告警标题及时间'])
index_id = 0
for i in range(max(data_error['涉及基站eNBID或小区ECGI']) + 1):
#for i in range(0,4):
    data_error_tmp = data_error[data_error['涉及基站eNBID或小区ECGI'] == i]
    data_warning_tmp = data_warning[data_warning['基站eNBID或小区ECGI'] == i]
    timestamp_before = 0
    for j in data_error_tmp.index:
        data_tmp = [data_error_tmp.loc[j,'涉及基站eNBID或小区ECGI'], data_error_tmp.loc[j,'故障发生时间'],data_error_tmp.loc[j,'故障_传输故障'],
                    data_error_tmp.loc[j,'故障_动环故障'],data_error_tmp.loc[j,'故障_电力故障'],data_error_tmp.loc[j,'故障_硬件故障'],
                    data_error_tmp.loc[j,'故障_误告警'],data_error_tmp.loc[j,'故障_软件故障']]
#        warning_tmp = [data_warning_tmp.loc[k,['告警标题','告警开始时间']].values for k in data_warning_tmp.index 
#                       if data_warning_tmp.loc[k,'告警开始时间'] > timestamp_before and data_warning_tmp.loc[k,'告警开始时间'] <= data_error_tmp.loc[j,'故障发生时间']]
        warning_tmp = [[data_warning_tmp.loc[k,'告警标题'],(data_error_tmp.loc[j,'故障发生时间'] - data_warning_tmp.loc[k,'告警开始时间'])/(24*3600)] for k in data_warning_tmp.index 
                       if data_warning_tmp.loc[k,'告警开始时间'] > timestamp_before and data_warning_tmp.loc[k,'告警开始时间'] <= data_error_tmp.loc[j,'故障发生时间']]
        timestamp_before = data_error_tmp.loc[j,'故障发生时间']
        data_tmp.append(warning_tmp)
        data.loc[index_id] = data_tmp
        index_id += 1
num_warning = 10
num_day = 15
columns = ['基站','故障_传输故障','故障_动环故障','故障_电力故障','故障_硬件故障','故障_误告警','故障_软件故障']
for i in range(num_warning):
    columns.append('告警'+str(i))
data_processed = pd.DataFrame(columns = columns)
index_id = 0
for i in range(max(data.index) + 1):
    data_tmp = [data.loc[i,'基站'],data.loc[i,'故障_传输故障'],data.loc[i,'故障_动环故障'],data.loc[i,'故障_电力故障'],
                data.loc[i,'故障_硬件故障'],data.loc[i,'故障_误告警'],data.loc[i,'故障_软件故障']]
    for j in range(len(data.loc[i,'告警标题及时间'])):
        list_tmp = (data.loc[i,'告警标题及时间'])[-j-1]
        if (len(data_tmp) >= 7 + num_warning) or list_tmp[1] >= num_day:
            break
        else:   
            data_tmp.append(list_tmp[0] + 1)    #告警值+1
    for k in range(7 + num_warning - len(data_tmp)):
        data_tmp.append(0)  #告警值0表示空值
    data_processed.loc[index_id] = data_tmp
    index_id += 1
data_processed = data_processed[data_processed['告警0'] != 0]
data_processed.to_csv('data_processed.csv')       
print(time.clock() - start)

