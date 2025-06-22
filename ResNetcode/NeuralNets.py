import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, num_labels):
        super(CNN1D, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=4),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(), # nn.ReLU(inplace=True)
                                  nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=4),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU(),  # nn.ReLU(inplace=True)
                                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.layer3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=4),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),  # nn.ReLU(inplace=True)
                                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Dropout层
        self.dropout = nn.Dropout(0.3)

        # 第一个全连接层
        self.fc1 = nn.Linear(128, 256)
        # 第二个全连接层（输出层）
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512,num_labels)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class SEblock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEblock, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool1d(1)

        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):

    expansion = 1  # 残差模块的输出通道数是输入通道数的多少倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=False):
        super(BasicBlock, self).__init__()

        # 主路径上的第一个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 主路径上的第二个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        #添加注意力机制在shortcut之前
        self.cbam = CBAM(out_channels)
        self.use_se = use_se
        self.se = SEblock(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample    # 如果输入和输出的尺寸不匹配，则使用downsample来匹配尺寸

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.use_se:
            out = self.se(out)
            # out = self.cbam(out)

        if self.downsample is not None: # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4  # 输出通道数是输入通道数的多少倍，ResNet中Bottleneck的扩展因子通常为4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=False, groups=1,
                 base_width=64):
        super(Bottleneck, self).__init__()

        width = int(out_channels * (base_width / 64.)) * groups

        # 第一个1x1卷积，用于减少输入通道数
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)

        # 第二个3x3卷积，是主要的特征提取层，可以使用分组卷积来进一步减少计算量
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(width)

        # 第三个1x1卷积，用于扩展输出通道数到最终的out_channels
        self.conv3 = nn.Conv1d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.use_se = use_se
        self.se = SEblock(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 如果输入和输出的尺寸不匹配，则使用downsample来匹配尺寸

        # 添加注意力机制在shortcut之前
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_se:
            out = self.se(out)
            # out = self.cbam(out)

        if self.downsample is not None: # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(x)

        out += identity  # 快捷连接（残差连接）
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_block, num_classes, use_se=False, drop_rate=0.1):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=2, bias=False),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(), # nn.ReLU(inplace=True)
                                  nn.MaxPool1d(kernel_size=5, stride=2, padding=2))

        # 多个残差模块，构成多层CNN结构
        self.block1 = self.build_resblock(block, 64,  num_block[0], stride=1, use_se=use_se)
        self.block2 = self.build_resblock(block, 128, num_block[1], stride=2, use_se=use_se)
        self.block3 = self.build_resblock(block, 256, num_block[2], stride=2, use_se=use_se)
        self.block4 = self.build_resblock(block, 512, num_block[3], stride=2, use_se=use_se)

        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def build_resblock(self, block, out_channels, num_blocks, stride=1, use_se=False):

        downsample = None
        # 用在每个block残差组块的第一层的shortcut分支上，此时上个残差组块输出out_channel与本残差组块所要求的输入in_channel通道数不同，
        # 所以用downsample调整进行升维，使输出out_channel调整到本残差组块后续处理所要求的维度。
        # 同时stride=2进行下采样减小尺寸size (注：block1时没有进行下采样，block2,3,4进行了下采样)
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion))

        res_blocks = []
        # 添加每一个残差组块里的第一层，第一层决定此组块是否需要下采样(后续层不需要)
        res_blocks.append(block(self.in_channels, out_channels, stride, downsample, use_se=use_se))
        self.in_channels = out_channels * block.expansion # 输出通道out_channel扩张

        for _ in range(1, num_blocks):
            res_blocks.append(block(self.in_channels, out_channels, use_se=use_se))

        # 非关键字参数的特征是一个星号*加上参数名，比如*number，定义后，number可以接收任意数量的参数，并将它们储存在一个tuple中
        return nn.Sequential(*res_blocks)


    def forward(self, x):
        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

# ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
def ResNet18(num_classes, use_se=False):
    return ResNet1D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, use_se=use_se)


def ResNet34(num_classes, use_se=False):
    return ResNet1D(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, use_se=use_se)


def ResNet50(num_classes, use_se=False):
    return ResNet1D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, use_se=use_se)


def ResNet101(num_classes, use_se=False):
    return ResNet1D(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, use_se=use_se)


def ResNet152(num_classes, use_se=False):
    return ResNet1D(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, use_se=use_se)
