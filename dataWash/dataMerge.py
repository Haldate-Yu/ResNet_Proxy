import numpy as np
import torch
import os
import torch
import pandas as pd
from scipy import stats

class MergeData:
    def __init__(self, base_dir, num_cases, hour, variable_name):
        """
                初始化CSVMerger类
                :param base_dir: 包含子目录的基目录
                """
        self.base_dir = base_dir
        self.num_cases = num_cases
        self.hour = hour
        self.variable_name = variable_name
        self.dataframes = []

    def read_csv_from_dir(self, dir_path, hour):
        """
        从指定目录读取CSV文件
        :param dir_path: 目录路径
        :return: DataFrame对象
        """
        csv_file = dir_path + '\data_' + str(hour) + 'h.csv'
        if csv_file:
            return pd.read_csv(csv_file)
        else:
            return None

    def merge_csvs(self):
        """
        遍历所有子目录，读取CSV文件，并将它们拼接成一个DataFrame
        :return: 拼接后的DataFrame
        """
        # 遍历case1到case100
        for i in range(1, self.num_cases+1):
            case_dir = os.path.join(self.base_dir, f'case{i:d}')  # 使用格式化字符串确保数字总是两位数
            if os.path.isdir(case_dir):
                df = self.read_csv_from_dir(case_dir, self.hour)
                if df is not None:
                    self.dataframes.append(df[self.variable_name])

                    # 使用pandas.concat拼接DataFrame
        if self.dataframes:
            df = pd.concat(self.dataframes, ignore_index=True, axis=1)
            return df
        else:
            return None
