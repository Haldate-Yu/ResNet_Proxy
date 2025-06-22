import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from utils import LRScheduler, EarlyStopping
import NeuralNets
import warnings
warnings.filterwarnings('ignore')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

# 确保随机种子一致，以便结果可复现
torch.manual_seed(0)
np.random.seed(0)

num_col = 90
num_samp = 5000
data = np.zeros((num_samp, num_col))

# fre = np.linspace(0.1, 8, num_samp)
# amp = np.linspace(0.1, 8, num_samp)
fre = np.random.rand(num_samp)*2
amp = np.random.rand(num_samp)*8

t1 = np.linspace(0, 30, 30)
t2 = np.linspace(0, 30, 60)

for i in range(num_samp):
    theta = np.random.rand()
    # print(theta)
    data[i, 0:30] = amp[i] * np.sin(fre[i] * t1 + theta)
    data[i, 30:90] = amp[i] * np.cos(fre[i] * t2 + theta)

X = data[:, 0:30]
y = data[:, 30:90]

# # 读取数据
# data = np.loadtxt('boston.txt')
# # 使用NumPy的loadtxt函数从文件'housing.txt'中读取数据
# # 假设数据文件的格式为每行代表一个样本,不同的特征值和目标值之间用空格或制表符分隔
# # 读取的数据将被存储在NumPy数组data中
#
# X = data[:, :13]
# # 数据集中前13列是输入特征,每一行代表一个样本,每一列代表一个特征
# # data[:, :13]表示选取数组的所有行和前13列
#
# y = data[:, 13]

# 数据归一化处理
scaler_x = MinMaxScaler()
# 创建一个MinMaxScaler对象scaler_x,用于对输入特征X进行归一化处理
# MinMaxScaler会将数据缩放到[0, 1]的范围内

X = scaler_x.fit_transform(X)
# 使用scaler_x对输入特征X进行拟合和转换
# fit_transform方法会计算数据的最小值和最大值,并将数据缩放到[0, 1]的范围内
# 转换后的数据将覆盖原始的X

scaler_y = MinMaxScaler()
# 创建另一个MinMaxScaler对象scaler_y,用于对目标值y进行归一化处理

y = scaler_y.fit_transform(y)
# 使用scaler_y对目标值y进行拟合和转换
# 由于MinMaxScaler期望输入是二维数组,因此需要使用reshape(-1, 1)将y转换为二维数组
# reshape(-1, 1)表示将y转换为一个列向量,行数自动推断
# fit_transform方法会计算数据的最小值和最大值,并将数据缩放到[0, 1]的范围内
# 最后使用flatten()将转换后的二维数组重新转换为一维数组,覆盖原始的y
# 将数据转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
# 使用torch.tensor函数将NumPy数组X转换为PyTorch张量
# dtype=torch.float32指定张量的数据类型为32位浮点数
# 转换后的张量X将用于模型的输入

y = torch.tensor(y, dtype=torch.float32)
# 转换后的张量y将用于模型的训练和评估
# 数据集划分
train_ratio = 0.7
val_ratio = 0.2
# 定义训练集和验证集的比例
# train_ratio表示训练集占总数据的比例,这里设置为0.7,即70%的数据用于训练
# val_ratio表示验证集占总数据的比例,这里设置为0.1,即10%的数据用于验证
# 剩下的20%用于测试

num_samples = len(X)
# 获取数据集的样本数,即X的长度

num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)
# 计算训练集和验证集的样本数
# num_train表示训练集的样本数,通过总样本数乘以训练集比例并取整得到
# num_val表示验证集的样本数,通过总样本数乘以验证集比例并取整得到

train_data = X[:num_train]
train_labels = y[:num_train]
# 使用切片操作提取训练集数据和标签
# train_data表示训练集的输入特征,取X的前num_train个样本
# train_labels表示训练集的目标值,取y的前num_train个样本

val_data = X[num_train:num_train + num_val]
val_labels = y[num_train:num_train + num_val]
# 使用切片操作提取验证集数据和标签
# val_data表示验证集的输入特征,取X从num_train到num_train+num_val的样本
# val_labels表示验证集的目标值,取y从num_train到num_train+num_val的样本

test_data = X[num_train + num_val:]
test_labels = y[num_train + num_val:]
# 使用切片操作提取测试集数据和标签
# test_data表示测试集的输入特征,取X从num_train+num_val到最后的样本
# test_labels表示测试集的目标值,取y从num_train+num_val到最后的样本
# 创建数据加载器
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)
# 使用TensorDataset将训练集、验证集和测试集的数据和标签打包成数据集对象
# TensorDataset接受多个张量作为参数,将它们组合成一个数据集
# train_dataset表示训练集的数据集对象,包含训练数据和标签
# val_dataset表示验证集的数据集对象,包含验证数据和标签
# test_dataset表示测试集的数据集对象,包含测试数据和标签

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# 使用DataLoader创建数据加载器,用于批次读取数据
# train_loader表示训练集的数据加载器,batch_size=64表示每个批次包含64个样本,shuffle=True表示在每个epoch开始时打乱数据顺序
# val_loader表示验证集的数据加载器,batch_size=64表示每个批次包含64个样本
# test_loader表示测试集的数据加载器,batch_size=64表示每个批次包含64个样本



def training_loop(num_epochs, optimizer, model, criterion, train_loader, val_loader):
    # 设置训练的轮数为20
    train_losses = []
    val_losses = []
    # 定义用于存储训练损失和验证损失的列表
    for epoch in range(num_epochs):
        model.train()
        # 将模型设置为训练模式
        train_loss = 0.0
        # 初始化训练损失为0.0
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            # 遍历训练数据加载器,获取每个批次的数据和标签
            optimizer.zero_grad()
            # 将优化器的梯度置零
            outputs = model(data.unsqueeze(1)).to(device)
            # print(outputs, labels.unsqueeze(1))
            # 将数据输入模型,得到预测输出
            loss = criterion(outputs, labels).to(device)

            # 计算预测输出和真实标签之间的损失,需要将标签增加一个维度以匹配输出的形状
            loss.backward()
            # 反向传播计算梯度
            optimizer.step()
            # 更新模型参数
            train_loss += loss.item() * data.size(0)
            # 累加训练损失,乘以批次大小以得到总损失
        train_loss /= len(train_loader.dataset)
        # 计算平均训练损失,除以训练集的样本数
        train_losses.append(train_loss)
        # 将平均训练损失添加到训练损失列表中

        model.eval()
        # 将模型设置为评估模式
        val_loss = 0.0
        # 初始化验证损失为0.0
        with torch.no_grad():
            # 禁用梯度计算,以减少内存占用和加速计算
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                # 遍历验证数据加载器,获取每个批次的数据和标签
                outputs = model(data.unsqueeze(1)).to(device)
                # 将数据输入模型,得到预测输出
                loss = criterion(outputs, labels)
                # 计算预测输出和真实标签之间的损失,需要将标签增加一个维度以匹配输出的形状
                val_loss += loss.item() * data.size(0)
                # 累加验证损失,乘以批次大小以得到总损失
        val_loss /= len(val_loader.dataset)

        # 计算平均验证损失,除以验证集的样本数
        val_losses.append(val_loss)
        # 将平均验证损失添加到验证损失列表中

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # 打印当前轮数、训练损失和验证损失


        lr_scheduler(val_loss)
        print('learning rate:', optimizer.param_groups[0]['lr'])
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

    return train_losses, val_losses


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNets.CNN1D(60).to(device)

lr = 0.001

criterion = nn.MSELoss().to(device)
optimizer = optim.NAdam(model.parameters(), lr=lr)

save_path = ".\\" #当前目录下
early_stopping = EarlyStopping(save_path, verbose=True, patience=7)
lr_scheduler = LRScheduler(optimizer, patience=3)

num_epochs = 5000

train_losses, val_losses = \
    training_loop(
        num_epochs=num_epochs,
        optimizer=optimizer,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader)

# ============================ 在测试集上评估模型
model.eval()
# 将模型设置为评估模式
test_preds = []
test_loss = 0.0
# 定义用于存储测试集预测值的列表
with torch.no_grad():
    # 禁用梯度计算,以减少内存占用和加速计算
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)
        # 遍历测试数据加载器,获取每个批次的数据
        outputs = model(data.unsqueeze(1)).to(device)
        # print(outputs, labels.unsqueeze(1))
        print(outputs.shape)
        loss = criterion(outputs, labels)
        # 将数据输入模型,得到预测输出
        test_preds.extend(outputs.cpu().numpy())
        # 将预测输出转换为NumPy数组并添加到测试集预测值列表中
        test_loss += loss.item() * data.size(0)
        # 累加验证损失,乘以批次大小以得到总损失
    test_loss /= len(test_loader.dataset)
    # 将平均验证损失添加到验证损失列表中

    print(f"Test Loss: {test_loss:.4f}")
    # 打印当前轮数、训练损失和验证损失

test_preds = scaler_y.inverse_transform(np.array(test_preds))
# print(test_preds)
# 对测试集预测值进行反归一化,将其转换为原始尺度
test_labels = scaler_y.inverse_transform(test_labels.numpy())
# 对测试集真实标签进行反归一化,将其转换为原始尺度


# ==================== 绘制测试集的真实值和预测值

plt.figure(figsize=(8, 6))
# 创建一个大小为(8, 6)的图形
plt.plot(test_labels[30], label='True Values (Testing Set)')
# 绘制测试集的真实标签,并添加标签
plt.plot(test_preds[30], label='Predicted Values (Testing Set)')
plt.plot(test_labels[40], label='True Values (Testing Set)')
# 绘制测试集的真实标签,并添加标签
plt.plot(test_preds[40], label='Predicted Values (Testing Set)')
plt.plot(test_labels[25], label='True Values (Testing Set)')
# 绘制测试集的真实标签,并添加标签
plt.plot(test_preds[25], label='Predicted Values (Testing Set)')
# 绘制测试集的预测值,并添加标签
plt.xlabel('Sample')
# 设置x轴标签为"Sample"
plt.ylabel('House Price')
# 设置y轴标签为"House Price"
plt.title('True vs. Predicted Values (Testing Set)')
# 设置图形标题为"True vs. Predicted Values (Testing Set)"
plt.legend()
# 添加图例
plt.tight_layout()
# 调整子图参数,使之填充整个图像区域
plt.show()
# 显示图形

# ==================== 绘制测试集的真实值和预测值
plt.figure(figsize=(8, 6))
# 创建一个大小为(8, 6)的图形
plt.plot(train_losses, label='Train loss')
# 绘制测试集的真实标签,并添加标签
plt.plot(val_losses, label='Validation loss')

# 绘制测试集的预测值,并添加标签
plt.xlabel('Sample')
# 设置x轴标签为"Sample"
plt.ylabel('loss')
# 设置y轴标签为"House Price"
# 设置图形标题为"True vs. Predicted Values (Testing Set)"
plt.legend()
# 添加图例
plt.tight_layout()
# 调整子图参数,使之填充整个图像区域
plt.show()
# 显示图形


