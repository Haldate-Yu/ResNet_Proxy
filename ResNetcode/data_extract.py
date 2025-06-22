import utils
import pandas as pd
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
num_cases = 500
# %% ======= 提取生产标签数据（out) ==============================================

num_hours = 24
base_dir_out = os.path.join(parent_dir, f'data_{num_cases:d}')
save_dir_out = os.path.join(parent_dir, 'data_combine')
if not os.path.exists(save_dir_out):  # 判断如果文件不存在,则创建
    os.makedirs(save_dir_out)

dataframes = []
for icase in range(1, num_cases + 1):
    dataframes = utils.MergeDataLess(base_dir_out, icase, num_hours,
                                     dataframes, 'WaterDepth').merge_csvs()
    print('Finished for case' + str(icase))
df_out = pd.concat(dataframes, ignore_index=True, axis=1)
print(df_out.shape)
df_out = df_out.T
# df_out.to_csv(os.path.join(save_dir_out, 'data_out.csv'), index=False, header=False)
df_out.to_hdf(os.path.join(save_dir_out, 'data_out.h5'), 'df', mode='w')

# %%  ======= 提取生产特征数据（in) ==============================================
base_dir_in = os.path.join(parent_dir, 'dataWash', f'natural_flow_{num_cases:d}_24h.csv')
save_dir_in = os.path.join(parent_dir, 'data_combine')

df_in = pd.read_csv(base_dir_in, header=None, index_col=False)
dataframe2 = df_in.iloc[:, 0]  # 第一列数据代表时间
# print(dataframe2.shape)
dataframe1 = df_in.iloc[:, 1:].T  # 删除第一例并转置，每一行代表一个样本
# print(dataframe1.shape)
result_list = []

# 遍历 dataframe1 的每一行
for i, row in dataframe1.iterrows():
    print('Start for case' + str(i))
    # 对 dataframe2 中的每一列进行循环，并将 dataframe2 中的值依次加到 dataframe1 的当前行前面
    for j in range(1, dataframe2.shape[0]):
        new_row = [dataframe2.iloc[j] / 3600.0] + [row.iloc[j]] + row.tolist()
        # print(new_row)
        result_list.append(new_row)

# 将结果列表转换为 DataFrame
df_in = pd.DataFrame(result_list)
# print(df_in.head(10))
df_in.reset_index(drop=True, inplace=True)  # 重置索引号使其从0开始
df_in.to_csv(os.path.join(save_dir_in, 'data_in.csv'), index=False, header=False)
df_in.to_hdf(os.path.join(save_dir_in, 'data_in.h5'), 'df', mode='w')
