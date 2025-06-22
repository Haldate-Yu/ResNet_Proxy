import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import warnings
import pickle
import utils
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
num_hours = 24

num_cases = 10

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sim_base_dir = os.path.join(parent_dir, 'data_prediction', f'sim_data_series')
cnn_base_dir = os.path.join(parent_dir, 'data_prediction', f'cnn_data_series')
error_figs_dir = os.path.join(parent_dir, 'data_prediction', 'error_figs')
if not os.path.exists(error_figs_dir):
    os.makedirs(error_figs_dir)

scaler_y = pickle.load(open('minmax_scaler_y.pkl', 'rb'))

for icase in range(1, num_cases+1):

    cnn_data = pd.read_csv(os.path.join(cnn_base_dir, f'water_depth_CNN_case{icase:d}.csv'))
    bias_error = []
    rmse_error = []
    cc_coeff = []

    for hour in range(1, num_hours+1):
        print(hour)
        sim_path = os.path.join(sim_base_dir, f'case{icase:d}', f'data_{hour:d}h.csv')
        sim_res = pd.read_csv(sim_path)['WaterDepth'].to_numpy()
        cnn_res = cnn_data[f'hour_{hour:d}'].to_numpy()

        bias_error.append(utils.bias(cnn_res, sim_res))
        rmse_error.append(utils.rmse(sim_res, cnn_res))
        cc_coeff.append(utils.cc(sim_res, cnn_res))

    print(bias_error, rmse_error, cc_coeff)
    print('Average Relative error : ', sum(bias_error)/len(bias_error))
    print('Average RMSE : ', sum(rmse_error)/len(rmse_error))
    print('Average Pearson correlation coeff : ',sum(cc_coeff)/len(cc_coeff))
    # ==================== 绘制测试集的真实值和预测值
    plt.figure(figsize=(8, 6))
    # 创建一个大小为(8, 6)的图形
    plt.plot(bias_error, label='bias loss')
    # 绘制测试集的真实标签,并添加标签
    plt.plot(rmse_error, label='rmse loss')
    plt.plot(cc_coeff, label='pc coeff')

    plt.xlabel('hours')
    plt.ylabel('error')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(error_figs_dir, f'error_case{icase:d}.png'))


