import numpy as np
import torch
import os
import torch
import pandas as pd
from scipy import stats


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, hour=24, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.hour = hour

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'     >>> EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Save model when validation loss decrease.'''
        if self.verbose:
            print(f'     >>> Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, f'best_network_{self.hour:d}h.pth')
        # torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        torch.save(model, path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(
            self, optimizer, patience=3, min_lr=1e-7, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class MergeData():
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
        # 遍历case
        for i in range(1, self.num_cases + 1):
            case_dir = os.path.join(self.base_dir, f'case{i:d}')  # 使用格式化字符串确保数字总是两位数
            if os.path.isdir(case_dir):
                df = self.read_csv_from_dir(case_dir, self.hour)
                if df is not None:
                    if df[self.variable_name].isnull().any():
                        print(f'Warning: NaN values found in {case_dir}/data_{self.hour:d}h.csv. Replacing NaN with 0.')
                        df[self.variable_name].fillna(0, inplace=True)
                    self.dataframes.append(df[self.variable_name])

                    # 使用pandas.concat拼接DataFrame
        if self.dataframes:
            df = pd.concat(self.dataframes, ignore_index=True, axis=1)
            return df
        else:
            return None


class MergeDataLess():
    def __init__(self, base_dir, icase, num_hours, dataframes, variable_name):
        """
                初始化CSVMerger类
                :param base_dir: 包含子目录的基目录
                """
        self.base_dir = base_dir
        self.icase = icase
        self.num_hours = num_hours
        self.variable_name = variable_name
        self.dataframes = dataframes

    def merge_csvs(self):
        """
        遍历所有子目录，读取CSV文件，并将它们拼接成一个DataFrame
        :return: 拼接后的DataFrame
        """
        # 遍历case
        for hour in range(1, self.num_hours + 1):
            case_dir = os.path.join(self.base_dir, f'case{self.icase:d}')  # 使用格式化字符串确保数字总是两位数
            if os.path.isdir(case_dir):
                df = pd.read_csv(os.path.join(case_dir, f'data_{hour:d}h.csv'))
                if df is not None:
                    if df[self.variable_name].isnull().any():
                        print(f'Warning: NaN values found in {case_dir}/data_{hour:d}h.csv. Replacing NaN with 0.')
                        df[self.variable_name].fillna(0, inplace=True)

                    # 将小于0.001的值设置为0
                    # df.loc[df[self.variable_name] < 0.001, self.variable_name] = 0

                    self.dataframes.append(df[self.variable_name])

                    # 使用pandas.concat拼接DataFrame
        if self.dataframes:
            return self.dataframes
        else:
            print('~~~~~~ No dataframes found! ~~~~~~~~~~~~')
            return None


def bias(aa, bb):  # 相对误差/偏差
    # sum_cz = 0
    # sum_o = 0
    # for i in range(len(aa)):
    #     sum_cz = sum_cz + abs(aa[i] - bb[i])
    #     sum_o = sum_o + abs(bb[i])
    # bias = abs(sum_cz / sum_o)

    sum_cz = sum(abs(aa - bb))
    bias = abs(sum_cz / len(aa))
    return bias


def rmse(aa, bb):  # 均方根误差
    sum = 0
    for i in range(len(aa)):
        sum = sum + (aa[i] - bb[i]) ** 2
    mse = sum / len(aa)
    rmse = mse ** 0.5
    return rmse


def cc(bb, aa):
    # aa,bb=pd.Series(aa),pd.Series(bb)
    # r=bb.corr(aa,method='pearson')
    r1, _ = stats.pearsonr(aa, bb)
    # # print('皮尔逊相关系数为{:.3f},p值为{:.5f}'.format(r1, p_value1))
    # if np.isnan(r):
    #     r=0
    r = r1
    return r
