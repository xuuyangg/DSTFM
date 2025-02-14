# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from crf import CRF
from er import EntropyRegularization
from base_model import *


class FreqBaseModel(nn.Module):
    """
        单频率模型 
    """
    def __init__(self):
        super().__init__()
        base = 16
        # 合并 CNN 部分
        # 使用Patch的处理方式修改 
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, base, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1),
            nn.ReLU()
        )
        self.positional_encoding = nn.Parameter(torch.randn(100, base * 4))
        # 将所有 TSLANet_layer 组合在一起
        self.tsla_layers = nn.Sequential(
            TSLANet_layer(base * 4),
            TSLANet_layer(base * 4),
        )
        self.fc1 = nn.Linear(base * 4 * 100, 1024)
        self.fc2 = nn.Linear(1024, 1)
    def forward(self, x):
        # 分支1流程
        x = x.unsqueeze(1)
        # 使用合并后的 CNN 层
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)
        x = x + self.positional_encoding
        # 使用组合后的 TSLANet_layer
        x = self.tsla_layers(x)
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FreqBaseStateModel(nn.Module):
    """
        单频率状态模型 
    """
    def __init__(self, num_state):
        super().__init__()
        base = 16
        # 合并 CNN 部分
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, base, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1),
            nn.ReLU()
        )
        self.positional_encoding = nn.Parameter(torch.randn(100, base * 4))
        self.tsla_1 = TSLANet_layer(base * 4)
        self.tsla_2 = TSLANet_layer(base * 4)
        self.tsla_3 = TSLANet_layer(base * 4)
        self.tsla_4 = TSLANet_layer(base * 4)
        self.fc1 = nn.Linear(base * 4 * 100, 1024)
        self.fc2 = nn.Linear(1024, num_state)

        self.fc1_s = nn.Linear(base * 4 * 100, 1024)
        self.fc2_s = nn.Linear(1024, num_state)

    def forward(self, x):
        # 分支1流程
        x = x.unsqueeze(1)
        # 使用合并后的 CNN 层
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)
        x = x + self.positional_encoding
        ### 在这里添加位置编码
        x = self.tsla_1(x)
        x = self.tsla_2(x)
        x = x.flatten(-2, -1)
        y = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y = F.relu(self.fc1_s(y))
        y = self.fc2_s(y)
        return x, y


class TimeBaseModel(nn.Module):
    """
        时间模型
    """
    def __init__(self):
        super().__init__()
        base = 16
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, base, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1),
            nn.ReLU()
        )
        self.mscr = MSCR(base * 4, base * 4)
        self.attn = SimplifiedLinearAttention(base * 4, 100, 2)
        # 全连接层用于预测
        self.fc1 = nn.Linear(base * 4 * 100, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        # 分支1流程
        x = x.unsqueeze(1)
        # 合并的 CNN 操作
        x = self.cnn_layers(x)
        x = self.mscr(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.attn(x)
        # 融合分支
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TFFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = 16
        # 频域分支
        self.frequency_cnn_layers = nn.Sequential(
            nn.Conv1d(1, base, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1),
            nn.ReLU()
        )
        self.positional_encoding = nn.Parameter(torch.randn(100, base * 4))
        self.tsla_1 = TSLANet_layer(base * 4)
        self.tsla_2 = TSLANet_layer(base * 4)
        # 时域分支
        self.time_cnn_layers = nn.Sequential(
            nn.Conv1d(1, base, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1),
            nn.ReLU()
        )
        self.mscr = MSCR(base * 4, base * 4)
        self.attn = SimplifiedLinearAttention(base * 4, 100, 2)

        self.fc1 = nn.Linear(base * 8 * 100, 1024)
        self.fc2 = nn.Linear(1024, 1)


    def forward(self, x):
        # 频域分支处理
        x = x.unsqueeze(1)
        f_out = self.frequency_cnn_layers(x)
        # 添加位置编码
        f_out = f_out.transpose(1, 2) + self.positional_encoding
        f_out = self.tsla_1(f_out)
        f_out = self.tsla_2(f_out)
        # 时域分支处理
        t_out = self.time_cnn_layers(x)
        t_out = self.mscr(t_out)
        t_out = self.attn(t_out.transpose(1, 2))

        out = torch.cat((f_out.flatten(-2), t_out.flatten(-2)), dim = 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class TFFusionStateModel(nn.Module):
    def __init__(self,num_state):
        super().__init__()
        base = 16
        # 频域分支
        self.frequency_cnn_layers = nn.Sequential(
            nn.Conv1d(1, base, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1),
            nn.ReLU()
        )
        self.positional_encoding = nn.Parameter(torch.randn(100, base * 4))
        self.tsla_1 = TSLANet_layer(base * 4)
        self.tsla_2 = TSLANet_layer(base * 4)
        # 时域分支
        self.time_cnn_layers = nn.Sequential(
            nn.Conv1d(1, base, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1),
            nn.ReLU()
        )
        self.mscr = MSCR(base * 4, base * 4)
        self.attn = SimplifiedLinearAttention(base * 4, 100, 2)

        self.fc1 = nn.Linear(base * 8 * 100, 1024)
        self.fc2 = nn.Linear(1024, num_state)

        self.fc1_s = nn.Linear(base * 8 * 100, 1024)
        self.fc2_s = nn.Linear(1024, num_state)

    def forward(self, x):
        # 频域分支处理
        x = x.unsqueeze(1)
        f_out = self.frequency_cnn_layers(x)
        # 添加位置编码
        f_out = f_out.transpose(1, 2) + self.positional_encoding
        f_out = self.tsla_1(f_out)
        f_out = self.tsla_2(f_out)
        # 时域分支处理
        t_out = self.time_cnn_layers(x)
        t_out = self.mscr(t_out)
        t_out = self.attn(t_out.transpose(1, 2))

        out = torch.cat((f_out.flatten(-2), t_out.flatten(-2)), dim = 1)
        out_s = out

        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        out_s = F.relu(self.fc1_s(out_s))
        out_s = self.fc2_s(out_s)

        return out, out_s




class SSFusionStateS2P(nn.Module):
    """
    双分支融合模型（状态预测版本）
    分为两个并行分支：
      - 分支1：卷积 + TSLANet_layer
      - 分支2：卷积 + My_ICB + SimplifiedLinearAttention
    最后将两个分支融合，经过全连接层得到两个预测输出，
    同时内置 CRF 与熵正则化模块，用于序列标注任务。
    """
    def __init__(self, window_len, num_state):
        super().__init__()
        base = 32
        # 分支1
        self.conv1_p = nn.Conv1d(1, base, 3, stride=1, padding=1)
        self.conv2_p = nn.Conv1d(base, base * 2, 3, stride=2, padding=1)
        self.conv3_p = nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1)
        self.tsla_1 = TSLANet_layer(base * 4)
        self.tsla_2 = TSLANet_layer(base * 4)
        # 分支2
        self.conv1_p_bran2 = nn.Conv1d(1, base, 3, stride=1, padding=1)
        self.conv2_p_bran2 = nn.Conv1d(base, base * 2, 3, stride=2, padding=1)
        self.conv3_p_bran2 = nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1)
        self.ICB = MSCR(base * 4, base * 4)
        self.attn = SimplifiedLinearAttention(base * 4, [10, 10], 2)
        # 全连接层用于预测
        self.fc1_p = nn.Linear(base * 8 * 100, 1024)
        self.fc2_p = nn.Linear(1024, num_state)
        self.fc1_s = nn.Linear(base * 8 * 100, 1024)
        self.fc2_s = nn.Linear(1024, num_state)
        # CRF 与熵正则化模块
        self.crf = CRF(num_tags=num_state, batch_first=True)
        self.er = EntropyRegularization()

    def calc_crf_loss(self, out, tgt):
        return -self.crf(out, tgt, reduction='mean')

    def calc_er_loss(self, out):
        return self.er(out)

    def decode(self, out):
        return self.crf.decode(out)

    def forward(self, x):
        # 分支1流程
        x = x.unsqueeze(1)
        y = x  # 备用分支
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = x.transpose(1, 2)
        x = self.tsla_1(x)
        x = self.tsla_2(x)
        # 分支2流程
        y = F.relu(self.conv1_p_bran2(y))
        y = F.relu(self.conv2_p_bran2(y))
        y = F.relu(self.conv3_p_bran2(y))
        y = self.ICB(y)
        y = torch.permute(y, (0, 2, 1))
        y = self.attn(y)
        # 融合分支
        x = x.flatten(-2, -1)
        y = y.flatten(-2, -1)
        x = torch.cat((x, y), axis=1)
        # 分别经过两个全连接分支
        x_p = F.relu(self.fc1_p(x))
        x_p = self.fc2_p(x_p)
        x_s = F.relu(self.fc1_s(x))
        x_s = self.fc2_s(x_s)
        return x_p, x_s


class SSFusionS2P(nn.Module):
    """
    双分支融合模型（单输出版本）
    结构与 SSFusionStateS2P 类似，但最终仅输出一个预测值。
    """
    def __init__(self, window_len):
        super().__init__()
        base = 32
        # 分支1
        self.conv1_p = nn.Conv1d(1, base, 3, stride=1, padding=1)
        self.conv2_p = nn.Conv1d(base, base * 2, 3, stride=2, padding=1)
        self.conv3_p = nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1)
        self.tsla_1 = TSLANet_layer(base * 4)
        self.tsla_2 = TSLANet_layer(base * 4)
        # 分支2
        self.conv1_p_bran2 = nn.Conv1d(1, base, 3, stride=1, padding=1)
        self.conv2_p_bran2 = nn.Conv1d(base, base * 2, 3, stride=2, padding=1)
        self.conv3_p_bran2 = nn.Conv1d(base * 2, base * 4, 3, stride=3, padding=1)
        self.ICB = My_ICB(base * 4, base * 4)
        self.attn = SimplifiedLinearAttention(base * 4, [10, 10], 2)
        # 全连接层
        self.fc1_p = nn.Linear(base * 8 * 100, 1024)
        self.fc2_p = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        # 分支1
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = x.transpose(1, 2)
        x = self.tsla_1(x)
        x = self.tsla_2(x)
        # 分支2
        y = F.relu(self.conv1_p_bran2(y))
        y = F.relu(self.conv2_p_bran2(y))
        y = F.relu(self.conv3_p_bran2(y))
        y = self.ICB(y)
        y = torch.permute(y, (0, 2, 1))
        y = self.attn(y)
        # 融合与预测
        x = x.flatten(-2, -1)
        y = y.flatten(-2, -1)
        x = torch.cat((x, y), axis=1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)
        return x


class S2P(nn.Module):
    """
    基础 S2P 模型
    采用多层卷积（核大小递减）后接全连接层进行预测。
    """
    def __init__(self, window_len):
        super().__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_p = nn.Conv1d(30, 40, 9, padding=4)
        self.conv4_p = nn.Conv1d(40, 50, 7, padding=3)
        self.conv5_p = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_p = nn.Linear(50 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)
        return x
