# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from einops import rearrange
import lightning as L
from timm.models.layers import trunc_normal_

#############################
# 1. 基础模块
#############################

class ICB(L.LightningModule):
    """
    基础 Inception Convolution Block (ICB)
    包含 1x1 和 3x3 两种卷积，经过 GELU 激活和 Dropout 后进行特征交互融合。
    """
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # 将特征维度转换到通道维度上（适用于 Conv1d）
        x = x.transpose(1, 2)
        # 分支1：1x1 卷积
        x1 = self.conv1(x)
        x1_act = self.act(x1)
        x1_drop = self.drop(x1_act)
        # 分支2：3x3 卷积
        x2 = self.conv2(x)
        x2_act = self.act(x2)
        x2_drop = self.drop(x2_act)
        # 融合两个分支（交叉乘积后求和）
        out = self.conv3(x1 * x2_drop + x2 * x1_drop)
        out = out.transpose(1, 2)
        return out


class PatchEmbed(L.LightningModule):
    """
    Patch Embedding 模块
    利用一维卷积将输入序列划分为 patch，并映射到嵌入空间。
    """
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # 先卷积再展平、转置
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out

#############################
# 2. 频域处理模块
#############################

class Adaptive_Spectral_Block(nn.Module):
    """
    自适应频谱块
    利用 FFT 将输入转换到频域，并对频域信息进行加权（权重参数为复数参数）。
    可选择性地生成高频自适应掩码（代码中注释部分）。
    """
    def __init__(self, dim):
        super().__init__()
        # 定义用于低频和高频的复数权重参数
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        # self.complex_weight = nn.Parameter(torch.randn(51, dim, 2, dtype=torch.float32) * 0.02) # 适合复杂状态的用电器
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        # 计算频域能量
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        # 展平后计算中值
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        # 能量归一化
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)
        # 生成自适应掩码
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        # 转换为 float32
        x = x_in.to(torch.float32)
        # FFT 变换，沿时间维度
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        # 应用复数权重
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        # 若需要自适应高频滤波，可启用以下代码
        # freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        # x_masked = x_fft * freq_mask.to(x.device)
        # weight_high = torch.view_as_complex(self.complex_weight_high)
        # x_weighted2 = x_masked * weight_high
        # x_weighted += x_weighted2

        # 逆 FFT 回到时域
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        return x.view(B, N, C)

#############################
# 3. 网络层模块
#############################

class TSLANet_layer(L.LightningModule):
    """
    TSLANet 层
    结合 LayerNorm、Adaptive_Spectral_Block 与 ICB 模块，支持多种连接方式（ICB、ASB 或同时使用）。
    """
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        # 当 drop_path > 0 时，采用 DropPath，否则直接 Identity
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, ICB=True, ASB=True):
        if ICB and ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x

#############################
# 4. 注意力模块
#############################

class SimplifiedLinearAttention(nn.Module):
    """
    简化线性注意力模块
    支持窗口划分，并结合位置编码与深度卷积以增强局部特征。
    """
    def __init__(self, dim, len, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.len = len# 例如 [height, width]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        # 深度卷积模块
        self.dwc = nn.Conv1d(in_channels=head_dim, out_channels=head_dim,
                             kernel_size=kernel_size, groups=head_dim, padding=kernel_size // 2)
        # 位置编码参数
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.len, dim))

    def forward(self, x, mask=None):
        """
        输入： x 形状为 (num_windows * B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        # 添加位置编码到 key
        k = k + self.positional_encoding
        # 使用 ReLU 激活作为 kernel 函数
        kernel_function = nn.ReLU()
        q = kernel_function(q)
        k = kernel_function(k)
        # 将头数信息分解到 batch 维度
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)
            # 归一化系数
            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            # 根据张量大小选择不同的计算策略
            if i * j * (c + d) > c * d * (i + j):
                kv = torch.einsum("b j c, b j d -> b c d", k, v)
                x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)
                x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)
        # 深度卷积特征增强
        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b l c -> b c l")
        feature_map = rearrange(self.dwc(feature_map), "b c l -> b l c")
        x = x + feature_map
        # 恢复原始形状
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def eval(self):
        super().eval()
        print('eval')

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

#############################
# 5. 辅助卷积模块
#############################

class DepthwiseSeparableConv(nn.Module):
    """
    一维深度可分离卷积：先进行深度卷积再进行逐点卷积，后接 BN 和 ReLU。
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 模块：通过全局平均池化与全连接层进行通道注意力重标定。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MSCR(nn.Module):
    """
    进阶 Inception Convolution Block
    采用多分支结构，包括：
      - 1x1 卷积分支
      - 3x3 深度可分离卷积分支
      - 5x5 深度可分离卷积分支
      - 最大池化后 1x1 卷积分支
    同时融合了 SE 通道注意力和空间注意力（通过深度卷积实现）。
    """
    def __init__(self, in_features, hidden_features, drop=0.0, reduction=16):
        super().__init__()
        # 分支1: 1x1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
        )
        # 分支2: 3x3 深度可分离卷积
        self.branch2 = nn.Sequential(
            DepthwiseSeparableConv(in_features, hidden_features, kernel_size=3, padding=1),
        )
        # 分支3: 5x5 深度可分离卷积
        self.branch3 = nn.Sequential(
            DepthwiseSeparableConv(in_features, hidden_features, kernel_size=5, padding=2),
        )
        # 分支4: 最大池化 + 1x1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
        )
        # 通道注意力模块（SEBlock）
        self.channel_attention = SEBlock(hidden_features * 4, reduction=reduction)
        # 合并各分支后降维
        self.combine_conv = nn.Sequential(
            nn.Conv1d(hidden_features * 4, in_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True)
        )
        # 空间注意力模块（利用分组卷积实现）
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_features, in_features, kernel_size=7, padding=3, groups=in_features, bias=False),
            nn.BatchNorm1d(in_features),
            nn.Sigmoid()
        )
        self.final_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        # 拼接各分支特征（通道维度上）
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.channel_attention(out)
        out = self.combine_conv(out)
        attention = self.spatial_attention(out)
        out = out * attention
        out = out + identity  # 残差连接
        out = self.final_activation(out)
        return out
