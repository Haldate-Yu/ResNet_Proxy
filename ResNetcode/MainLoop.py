import warnings
import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from utils import LRScheduler, EarlyStopping
import utils
import NeuralNets
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import time

warnings.filterwarnings('ignore')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

t0 = time.time()
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")


# 读入数据，初始读入时速度比较慢，将数据保存为hdf文件，后期再运行时直接读取hdf文件，以提升读入速度。
def read_data(base_dir, file_name):
    if os.path.exists(os.path.join(base_dir, f'{file_name}.h5')):
        print('*** Load hdf data ***')
        df = pd.read_hdf(os.path.join(base_dir, f'{file_name}.h5'), 'df')
    else:
        print('*** No hdf data exists, please have it in this file first! ***')

    return df


# # 确保随机种子一致，以便结果可复现
# torch.manual_seed(0)
# np.random.seed(0)
def data_loader_2S(df_in, df_out, scaler_X, scaler_y, test_size, batch_size):
    X = scaler_X.transform(df_in)
    y = scaler_y.transform(df_out)

    # 使用torch.tensor函数将数组X转换为PyTorch张量
    X = torch.tensor(X, dtype=torch.float32)  # 将数据进行转置，每一行即为一个样本
    y = torch.tensor(y, dtype=torch.float32)

    # 划分训练集和测试集数据
    train_data, val_data, train_labels, val_labels = \
        train_test_split(X, y, test_size=test_size, random_state=4)

    # 创建数据加载器
    # 使用TensorDataset将训练集、验证集和测试集的数据和标签打包成数据集对象
    # # TensorDataset接受多个张量作为参数,将它们组合成一个数据集
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    # 使用DataLoader创建数据加载器,用于批次读取数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def data_loader_3S(df_in, df_out, scaler_X, scaler_y, split_ratio, batch_size):
    X = scaler_X.transform(df_in)
    y = scaler_y.transform(df_out)

    # 使用torch.tensor函数将数组X转换为PyTorch张量
    X = torch.tensor(X, dtype=torch.float32)  # 将数据进行转置，每一行即为一个样本
    y = torch.tensor(y, dtype=torch.float32)

    # 创建一个TensorDataset
    dataset = TensorDataset(X, y)

    # 定义划分比例
    train_size = int(split_ratio[0] * len(dataset))
    val_size = int(split_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(train_size / 24.0, val_size / 24.0, test_size / 24.0)

    # 划分数据集
    train_dataset, val_dataset, test_dataset = \
        random_split(dataset,
                     [train_size, val_size, test_size],
                     torch.Generator().manual_seed(487))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def training_loop(num_epochs, optimizer, model,
                  criterion, train_loader, val_loader,
                  lr_scheduler, early_stopping):
    # 定义用于存储训练损失和验证损失的列表
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ======== 将模型设置为训练模式 =========
        model.train()
        # 初始化训练损失为0.0
        train_loss = 0.0

        for data, labels in train_loader:
            # 遍历验证数据加载器,获取每个批次的数据和标签
            data = data.to(device)
            labels = labels.to(device)
            # 将优化器的梯度置零
            optimizer.zero_grad()
            # 将数据输入模型,得到预测输出
            outputs = model(data.unsqueeze(1)).to(device)
            # print(outputs, labels.unsqueeze(1))
            # 计算预测输出和真实标签之间的损失
            loss = criterion(outputs, labels).to(device)
            # 反向传播计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 累加训练损失,乘以批次大小以得到总损失
            train_loss += loss.item() * data.size(0)
        # 计算平均训练损失,除以训练集的样本数
        train_loss /= len(train_loader.dataset)
        # 将平均训练损失添加到训练损失列表中
        train_losses.append(train_loss)

        # ========= 将模型设置为评估模式 =========
        model.eval()
        # 初始化验证损失为0.0
        val_loss = 0.0

        with torch.no_grad():
            # 禁用梯度计算,以减少内存占用和加速计算
            for data, labels in val_loader:
                # 遍历验证数据加载器,获取每个批次的数据和标签
                data = data.to(device)
                labels = labels.to(device)
                # 将数据输入模型,得到预测输出
                outputs = model(data.unsqueeze(1)).to(device)
                # 计算预测输出和真实标签之间的损失,需要将标签增加一个维度以匹配输出的形状
                loss = criterion(outputs, labels)
                # 累加验证损失,乘以批次大小以得到总损失
                val_loss += loss.item() * data.size(0)
        # 计算平均验证损失,除以验证集的样本数
        val_loss /= len(val_loader.dataset)

        # 将平均验证损失添加到验证损失列表中
        val_losses.append(val_loss)

        # 打印当前轮数、训练损失和验证损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}")

        lr_scheduler(val_loss)
        print('     >>> learning rate:', optimizer.param_groups[0]['lr'])
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练
    # path = os.path.join(save_path, 'best_network_24h.pth')
    # torch.save(model.state_dict(), path)

    return train_losses, val_losses


# %%
# *******************************************************************************************
# ==================== 导入数据集 =============================================================
# *******************************************************************************************
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 读取输入数据 ---------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.join(parent_dir, 'data_combine')
print('Reading csvfiles (processing...)')
df_in = read_data(base_dir, 'data_in')
df_out = read_data(base_dir, 'data_out')

# batch_size = 128
batch_size = 146
lr = 0.001
num_epochs = 300

# ================== 训练模型 ===================================================
scaler_X = MinMaxScaler().fit(df_in)
scaler_y = MinMaxScaler().fit(df_out)

# train_loader, val_loader = data_loader_2S(df_in, df_out, scaler_X, scaler_y, test_size=0.3, batch_size=batch_size)
train_loader, val_loader, test_loader = data_loader_3S(df_in, df_out, scaler_X, scaler_y,
                                                       split_ratio=[0.8, 0.18, 0.02], batch_size=batch_size)
# 保存模型
with open('minmax_scaler_X.pkl', 'wb') as file:
    pickle.dump(scaler_X, file)
with open('minmax_scaler_y.pkl', 'wb') as file:
    pickle.dump(scaler_y, file)

# # 神经网络模型
model = NeuralNets.ResNet18(num_classes=df_out.shape[1], use_se=False).to(device)
# model = NeuralNets.CNN1D(df_out.shape[1]).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.NAdam(model.parameters(), lr=lr, weight_decay=1e-5)

# Early Stopping for quick calculation
save_path = ".\\"  # 当前目录下
early_stopping = EarlyStopping(save_path, verbose=True, patience=30)
lr_scheduler = LRScheduler(optimizer, patience=10)

train_losses, val_losses = \
    training_loop(
        num_epochs=num_epochs,
        optimizer=optimizer,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_scheduler=lr_scheduler,
        early_stopping=early_stopping)

# 创建DataFrame
df_loss = pd.DataFrame({
    'train loss': train_losses,
    'val loss': val_losses
})

# 保存DataFrame到CSV文件
csv_filename = 'train_val losses.csv'
df_loss.to_csv(csv_filename, index=False)

# ==================== 绘制测试集的真实值和预测值 =================================================
plt.figure(figsize=(8, 6))
# 创建一个大小为(8, 6)的图形
plt.plot(train_losses, label='Train loss')
# 绘制测试集的真实标签,并添加标签
plt.plot(val_losses, label='Validation loss')

# 绘制测试集的预测值,并添加标签
plt.xlabel('step')
# 设置x轴标签为"Sample"
plt.ylabel('loss')
# 设置y轴标签为"House Price"
# 设置图形标题为"True vs. Predicted Values (Testing Set)"
plt.legend()
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
# 添加图例
plt.tight_layout()
# 调整子图参数,使之填充整个图像区域
plt.show()
# 显示图形


# %% ================ 测试集 验证模型训练效果 ========================================================
model = torch.load('best_network_24h.pth')
scaler_y = pickle.load(open('minmax_scaler_y.pkl', 'rb'))
model.eval()

bias_error = []
rmse_error = []
cc_coeff = []

# 禁用梯度计算,以减少内存占用和加速计算
with torch.no_grad():
    # 遍历验证数据加载器,获取每个批次的数据和标签
    for data, labels in test_loader:
        print(data.shape)
        data = data.to(device)
        labels = labels.to(device)
        # 将数据输入模型,得到预测输出
        outputs = model(data.unsqueeze(1)).to(device)

        for i in range(data.shape[0]):
            print(i)
            # 反归一化数据从而得到真实的误差
            ioutputs = scaler_y.inverse_transform(outputs[i].detach().cpu().numpy().reshape(1, -1))[0]
            ilabels = scaler_y.inverse_transform(labels[i].detach().cpu().numpy().reshape(1, -1))[0]

            # print(sim_res,cnn_res)
            bias_error.append(utils.bias(ioutputs, ilabels))
            rmse_error.append(utils.rmse(ioutputs, ilabels))
            cc_coeff.append(utils.cc(ioutputs, ilabels))

# 创建DataFrame
df1 = pd.DataFrame({
    'MAE Error': bias_error,
    'RMSE Error': rmse_error,
    'PCC Coefficient': cc_coeff
})

# 保存DataFrame到CSV文件
csv_filename = 'sample errors.csv'
df1.to_csv(csv_filename, index=False)

print(bias_error, rmse_error, cc_coeff)
print('Average Mean Error : ', sum(bias_error) / len(bias_error))
print('Average RMSE : ', sum(rmse_error) / len(rmse_error))
print('Average Pearson correlation coeff : ', sum(cc_coeff) / len(cc_coeff))
# ==================== 绘制测试集的真实值和预测值
plt.figure(figsize=(8, 6))
# 创建一个大小为(8, 6)的图形
plt.plot(bias_error, label='mean error (m)')
# 绘制测试集的真实标签,并添加标签
plt.plot(rmse_error, label='rmse error (m)')
plt.plot(cc_coeff, label='cc coeff')
plt.title('Mean/RMSE/Correlation')
plt.xlabel('sample')
plt.ylabel('error')
plt.legend()
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.tight_layout()
plt.show()

t1 = time.time()
print(f'---- Time cost: {(t1 - t0) / 3600.0:.2f} hours ----')
