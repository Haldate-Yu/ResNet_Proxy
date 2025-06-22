import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# # 原始数据
# data = {
#     'time': list(range(0, 86401, 3600)),
#     'flow': [488.8572704, 545.2193674, 601.5814643, 657.9435613, 714.3056582, 770.6677552,
#              827.0298522, 883.3406771, 1285.812027, 2310.620807, 2108.640989, 1782.092502,
#              1330.437386, 1129.854322, 990.5182354, 883.1069202, 803.0467248, 750.4214955,
#              702.7567941, 668.2285172, 643.309844, 649.0598106, 644.9108699, 623.3130579, 787.6029985]
# }
# df = pd.DataFrame(data)
#
# # 设置流量列的最大值和最小值
# max_flow = 4000
# min_flow = 10

random.seed(20)


def gradient_scale(flow_data, peak_index, scale_factor, gradient_half_width):
    """
    对流量数据进行渐变式缩放。

    :param flow_data: 原始流量数据（numpy数组）
    :param peak_index: 峰值所在的索引
    :param scale_factor: 峰值附近的缩放因子
    :param gradient_half_width: 渐变区域的半宽
    :return: 缩放后的流量数据（numpy数组）
    """
    length = len(flow_data)
    scaled_flow = flow_data.copy()

    # 计算渐变区域的起始和结束索引
    start_index = max(0, peak_index - gradient_half_width)
    end_index = min(length - 1, peak_index + gradient_half_width)

    # 应用线性渐变缩放
    for i in range(start_index, end_index + 1):
        # 计算当前索引相对于峰值索引的距离
        distance_to_peak = abs(i - peak_index)
        # 计算渐变因子（从0到1）
        gradient_factor = 1 - (distance_to_peak / gradient_half_width)
        # 确保渐变因子不小于0（虽然由于我们的计算方式，它永远不会小于0）
        gradient_factor = max(0, gradient_factor)
        # 应用缩放因子
        scaled_flow[i] *= scale_factor * gradient_factor + (1 - gradient_factor)

    # 注意：这里的实现方式确保了渐变区域外部的点保持原样（即缩放因子为1）
    # 因为我们在for循环之外没有修改scaled_flow数组的其他部分

    return scaled_flow


# 数据增强函数
def complex_augment_data(original_flow, Qr, num_augmentations=20, noise_scale=0.01,
                         scale_factor_range=(0.5, 4.0), flip_prob=0.8,peak_shift_prob=0.5,
                         peak_value_range=(500, 3000),
                         trend_slope_range=(-0.3, 0.3)):
    augmented_data = []

    Q = random.sample(Qr, num_augmentations)
    # print(Q)
    for j in range(num_augmentations):
        # 复制原始流量数据作为增强数据的起点
        augmented_flow = original_flow.copy()

        # 应用缩放
        peak_value = max(augmented_flow)
        reset_flow = Q[j]
        # print(reset_flow)
        augmented_flow = augmented_flow * reset_flow / peak_value

        # 找到峰值位置
        # peak_index = np.argmax(augmented_flow)
        # scale_factor = random.uniform(scale_factor_range[0], scale_factor_range[1])
        # if scale_factor >= 0.5:
        #     augmented_flow = gradient_scale(augmented_flow, peak_index, scale_factor, gradient_half_width=10)
        #     augmented_flow = np.clip(augmented_flow, min_flow, max_flow)
        # else:
        #     augmented_flow *= scale_factor
        #     augmented_flow = np.clip(augmented_flow, min_flow, max_flow)

        # # 根据概率翻转数据
        # if random.random() < flip_prob:
        #     augmented_flow = augmented_flow[::-1]

        # # 根据概率改变峰值位置并设置新的峰值
        # if random.random() < peak_shift_prob:
        #     new_peak_index = random.randint(0, len(original_flow) - 1)
        #     new_peak_value = random.uniform(peak_value_range[0], peak_value_range[1])
        #     augmented_flow[new_peak_index] = new_peak_value

        # 添加趋势（线性趋势，斜率随机）
        # slope = random.uniform(trend_slope_range[0], trend_slope_range[1])
        # trend = np.arange(len(original_flow)) * slope
        # augmented_flow += trend
        # augmented_flow = np.clip(augmented_flow, min_flow, max_flow)

        # # 添加噪声
        # noise = np.random.normal(0, max(augmented_flow)*noise_scale, size=len(original_flow))
        #
        # # print(noise)
        # augmented_flow += noise
        # augmented_flow = np.clip(augmented_flow, np.random.uniform(30,80), max_flow)

        # 将增强后的数据添加到列表中
        # augmented_data.append(pd.DataFrame({'time': df['time'], 'flow': augmented_flow}))
        augmented_data.append(augmented_flow)

    return augmented_data

# 生成增强数据
# augmented_data_list = complex_augment_data(df['flow'].values, num_augmentations=20)  # 减少增强数量以便快速展示

# # 遍历增强后的数据并打印每条曲线的图像
# for i, augmented_df in enumerate(augmented_data_list):
#     plt.figure(figsize=(10, 4))
#     plt.plot(augmented_df['time'], augmented_df['flow'], label=f'Augmented Data {i + 1}')
#     plt.plot(df['time'], df['flow'], label='Original Data', linestyle='--')
#     plt.xlabel('Time')
#     plt.ylabel('Flow')
#     plt.title(f'Augmented Flow Data {i + 1}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# # 生成增强数据
# augmented_data_list = complex_augment_data(df['flow'].values, num_augmentations=20)  # 减少增强数量以便快速展示
#
# # 遍历增强后的数据并打印每条曲线的图像
# for i, augmented_df in enumerate(augmented_data_list):
#     plt.figure(figsize=(10, 4))
#     plt.plot(augmented_df['time'], augmented_df['flow'], label=f'Augmented Data {i + 1}')
#     plt.plot(df['time'], df['flow'], label='Original Data', linestyle='--')
#     plt.xlabel('Time')
#     plt.ylabel('Flow')
#     plt.title(f'Augmented Flow Data {i + 1}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()