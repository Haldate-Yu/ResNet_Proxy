import numpy as np
import os
import scipy as sp
import pandas as pd
import shutil
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt

data_dir = 'D:/5. 程序运行文件/dataWash/'

natural_flow = pd.read_csv(data_dir + 'natural_flow_1998_24h.csv', header=None)
stairstep_flow = pd.read_csv(data_dir + 'stairstep_flow_102_24h.csv', header=None)
# print(natural_flow.shape, stairstep_flow.shape)
stairstep_flow = stairstep_flow.iloc[:, 1:]
# print(stairstep_flow.shape)

appended_data = pd.concat([natural_flow, stairstep_flow], axis=1)
#
fixed_column = appended_data.iloc[:, 0]
appended_data = appended_data.iloc[:, 1:]
# print(appended_data)
# #

columns = list(range(1, appended_data.shape[1]+1)) #重新对列索引经行按顺序编号，否则会出错（合并两个df之后需要对索引经行重置）
appended_data.columns = columns
np.random.shuffle(columns)
#
# # 使用这个随机索引列表来重新排序DataFrame的列
appended_data_shuffled = appended_data[columns]
appended_data_shuffled.insert(0,0, fixed_column)
# print(appended_data_shuffled)

appended_data_shuffled.to_csv(data_dir+'appended_24h.csv', index=False, header=False)




