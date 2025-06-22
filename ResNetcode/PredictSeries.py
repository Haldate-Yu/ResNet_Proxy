import torch
import torch.nn as nn
import numpy as np
import os
import NeuralNets
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

num_hours = 24

inflow_file_name = 'test_flow_series.csv'


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_in_path = os.path.join(parent_dir, 'data_prediction')
in_flow = os.path.join(base_in_path, inflow_file_name)
out_save_path = os.path.join(base_in_path, 'cnn_data_series')
if not os.path.exists(out_save_path):
    os.makedirs(out_save_path)


df_in = pd.read_csv(in_flow, header=None, index_col=False)
df_in = df_in.iloc[:, 1:].T # 删除第一例并转置，每一行代表一个样本
# print(df_in)
df_in.reset_index(drop=True, inplace=True) # 重置索引号使其从0开始

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('============ start predict =============')
scaler_X = pickle.load(open('minmax_scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('minmax_scaler_y.pkl', 'rb'))

# 导入网络模型
model = torch.load('best_network_24h.pth')
model.eval()

for i, row in df_in.iterrows():
    print(f'Start for case {i+1}' )
    df_pred = pd.DataFrame()

    # 预测24个小时的结果
    for hour in range(1, num_hours+1):

        new_row = [hour] + [row.iloc[hour]] + row.tolist()

        new_row = pd.DataFrame(new_row).T
        new_row.reset_index(drop=True, inplace=True)  # 重置索引号使其从0开始
        X = scaler_X.transform(new_row)

        X = torch.tensor(X, dtype=torch.float32).to(device)
        X = X.unsqueeze(1)

        y_pred = model(X)

        df_pred[f'hour_{hour:d}'] = scaler_y.inverse_transform(y_pred.cpu().detach().numpy())[0]

        print(f'Successfully for {hour:d}h')

    df_pred.to_csv(os.path.join(out_save_path, f'water_depth_CNN_case{i+1}.csv'))



























