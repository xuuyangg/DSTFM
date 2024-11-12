# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from crf import CRF
from er import EntropyRegularization
from einops import rearrange
# import lightning as L
# from timm.models.layers import trunc_normal_

# class Adaptive_Spectral_Block(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
#         self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

#         trunc_normal_(self.complex_weight_high, std=.02)
#         trunc_normal_(self.complex_weight, std=.02)
#         self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)

#     def create_adaptive_high_freq_mask(self, x_fft):
#         B, _, _ = x_fft.shape

#         # Calculate energy in the frequency domain
#         energy = torch.abs(x_fft).pow(2).sum(dim=-1)

#         # Flatten energy across H and W dimensions and then compute median
#         flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
#         median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
#         median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

#         # Normalize energy
#         normalized_energy = energy / (median_energy + 1e-6)

#         adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
#         adaptive_mask = adaptive_mask.unsqueeze(-1)

#         return adaptive_mask

#     def forward(self, x_in, adaptive_filter = False):
#         B, N, C = x_in.shape

#         dtype = x_in.dtype
#         x = x_in.to(torch.float32)

#         # Apply FFT along the time dimension
#         x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
#         weight = torch.view_as_complex(self.complex_weight)
#         x_weighted = x_fft * weight

#         if adaptive_filter:
#             # Adaptive High Frequency Mask (no need for dimensional adjustments)
#             freq_mask = self.create_adaptive_high_freq_mask(x_fft)
#             x_masked = x_fft * freq_mask.to(x.device)

#             weight_high = torch.view_as_complex(self.complex_weight_high)
#             x_weighted2 = x_masked * weight_high

#             x_weighted += x_weighted2

#         # Apply Inverse FFT
#         x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

#         x = x.to(dtype)
#         x = x.view(B, N, C)  # Reshape back to original shape

#         return x



class SimplifiedLinearAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        # self.dwc = nn.Sequential(nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=3,
        #                                    groups=head_dim, padding=1),
        #                          nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=3,
        #                                    groups=head_dim, padding=1)
        #                          )
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

        print('Linear Attention window{} f{} kernel{}'.
              format(window_size, focusing_factor, kernel_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        k = k + self.positional_encoding

        kernel_function = nn.ReLU()
        q = kernel_function(q)
        k = kernel_function(k)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            if i * j * (c + d) > c * d * (i + j):
                kv = torch.einsum("b j c, b j d -> b c d", k, v)
                x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)
                x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)


        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def eval(self):
        super().eval()
        print('eval')

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'




class SpAttnS2P(nn.Module):
    def __init__(self, window_len):
        super(SpAttnS2P, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 3, stride=1, padding = 1)
        self.conv2_p = nn.Conv1d(30, 60, 3, stride=2, padding= 1)
        self.conv3_p = nn.Conv1d(60, 120, 3, stride=3, padding= 1)
        self.asb = Adaptive_Spectral_Block(120)
        self.sum_attn = BahdanauAttention(120)
        self.fc1_p = nn.Linear(120      , 1)
        # self.fc2_p = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x)) # output batch, 128, 300
        x = F.relu(self.conv3_p(x)) # output batch, 128, 300
        x = x.transpose(1, 2)
        x = self.asb(x)
        x = torch.permute(x, (0, 1 ,2)) 

        x,_ = self.sum_attn(x)
        x = F.relu(x)
        # x = x.flatten(-2, -1)
        x = self.fc1_p(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze operation: Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # Excitation operation: Two fully connected layers
        self.fc1 = nn.Conv1d(channel, channel // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channel // reduction, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Squeeze: Global Average Pooling
        y = self.global_avg_pool(x)
        # Excitation: Bottleneck with two FC layers
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ICB(nn.Module):
    """
    Advanced Inception Convolutional Block with multi-scale convolutions,
    depthwise separable convolutions, SE blocks, and attention mechanisms.
    """
    def __init__(self, in_features, hidden_features, drop=0.0, reduction=16):
        super(ICB, self).__init__()
        
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )
        
        # Branch 2: 3x3 depthwise separable convolution
        self.branch2 = nn.Sequential(
            DepthwiseSeparableConv(in_features, hidden_features, kernel_size=3, padding=1),
            nn.Dropout(drop)
        )
        
        # Branch 3: 5x5 depthwise separable convolution
        self.branch3 = nn.Sequential(
            DepthwiseSeparableConv(in_features, hidden_features, kernel_size=5, padding=2),
            nn.Dropout(drop)
        )
        
        # Branch 4: Max Pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )
        
        # Channel Attention Module
        self.channel_attention = SEBlock(hidden_features * 4, reduction=reduction)
        
        # Combine branches
        self.combine_conv = nn.Sequential(
            nn.Conv1d(hidden_features * 4, in_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True)
        )
        
        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_features, in_features, kernel_size=7, padding=3, groups=in_features, bias=False),
            nn.BatchNorm1d(in_features),
            nn.Sigmoid()
        )
        
        # Final activation function
        self.final_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x  # Residual connection

        # Apply branches
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        # Concatenate along the channel dimension
        out = torch.cat([x1, x2, x3, x4], dim=1)

        # Apply channel attention
        out = self.channel_attention(out)

        # Combine and reduce channels
        out = self.combine_conv(out)

        # Apply spatial attention
        attention = self.spatial_attention(out)
        out = out * attention

        # Add residual
        out += identity

        # Final activation
        out = self.final_activation(out)

        return out


# Define the RightModel
class RightModel(nn.Module):
    def __init__(self):
        super(RightModel, self).__init__()
        # Separable Convolutions on layer3
        # GRU Layer for layer3
        self.bi_gru1 = nn.GRU(input_size=64, hidden_size=32, bidirectional=True, batch_first=True)
        # Additional Conv1d Layer after GRU
        self.conv1d_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # Separable Convolutions on layer2
        self.layer2_SepConv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Additional Conv1d Layer after Add operation on layer2
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=3)
        # GRU Layer for layer2
        self.bi_gru2 = nn.GRU(input_size=64, hidden_size=32, bidirectional=True, batch_first=True)
        self.out_cnn = nn.Conv1d(in_channels=64, out_channels=64, kernel_size= 3, padding= 1)

            
    def forward(self, layer3, layer2):
        # First Separable Conv Block on layer3

        # x2 = F.relu(self.layer3_total_SepConv2(layer3))
        sep_add = layer3 
        # Remove the unnecessary transpose
        # Prepare for GRU input (batch_size, sequence_length, features)
        dil_add = sep_add.permute(0, 2, 1)
        bi_gru1_out, _ = self.bi_gru1(dil_add)
        # Reshape back and apply Conv1d
        gru1_output = bi_gru1_out.permute(0, 2, 1)
        gru1_output = F.relu(self.conv1d_1(gru1_output))
        
        # Second Separable Conv Block on layer2
        x3 = F.relu(self.layer2_SepConv1(layer2))
        # x4 = F.relu(self.layer2_SepConv2(layer2))
        sep_add_1 = x3 
        # Remove the unnecessary transpose
        sep_add_1 = F.relu(self.conv1d_2(sep_add_1))
        # Prepare for GRU input
        dil_add_1 = sep_add_1.permute(0, 2, 1)
        bi_gru2_out, _ = self.bi_gru2(dil_add_1)
        gru2_output = bi_gru2_out.permute(0, 2, 1)
        
        # Final Add operation and mean pooling
        right_output = self.out_cnn((gru1_output + gru2_output))
        # right_output = self.se(right_output)
        return right_output


class ParallelS2P(nn.Module):
    def __init__(self, window_len):
        super(ParallelS2P, self).__init__()
        self.conv1_p = nn.Conv1d(1, 16, 3, stride=1, padding = 1)
        self.conv2_p = nn.Conv1d(16, 32, 3, stride=2, padding= 1)
        self.conv3_p = nn.Conv1d(32, 64, 3, stride=3, padding= 1)
        self.ICB = ICB(64, 64)
        self.attn = SimplifiedLinearAttention(64, [10, 10], 2)
        self.right = RightModel()
        self.fc1_p = nn.Linear(64  * 100 * 2, 1024)
        # self.fc2_p = nn.Linear(32, 1)
        # self.layernorm = nn.LayerNorm(256)
        self.fc2_p = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x)) # output batch, 128, 300
        layer2 = x.clone()
        x = F.relu(self.conv3_p(x)) # output batch, 256, 100
        layer3 = x.clone()
        x = self.ICB(x)
        x = torch.permute(x, (0, 2 ,1))
        x = self.attn(x)
        right = self.right(layer3, layer2)
        # x = F.relu(x)
        x = x.flatten(-2, -1)
        right  = right.flatten(-2, -1)
        x = torch.cat((x, right), axis = 1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)
        return x


class LinearAttnS2P(nn.Module):
    def __init__(self, window_len):
        super(LinearAttnS2P, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 3, stride=1, padding = 1)
        self.conv2_p = nn.Conv1d(30, 60, 3, stride=2, padding= 1)
        self.conv3_p = nn.Conv1d(60, 120, 3, stride=3, padding= 1)
        self.attn = SimplifiedLinearAttention(120, [10, 10], 2)
        # self.sum_attn = BahdanauAttention(120)
        self.fc1_p = nn.Linear(120      , 1)
        # self.fc2_p = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x)) # output batch, 128, 300
        x = F.relu(self.conv3_p(x)) # output batch, 256, 100
        x = torch.permute(x, (0, 2 ,1))
        x = self.attn(x)
        # x = torch.permute(x, (0, 1 ,2)) 
        # x,_ = self.sum_attn(x)
        x = F.relu(x)
        # x = x.flatten(-2, -1)
        x = self.fc1_p(x)
        return x


class S2P(nn.Module):
    def __init__(self, window_len):
        super(S2P, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_p = nn.Conv1d(30, 40, 9, padding=4)  # 6
        self.conv4_p = nn.Conv1d(40, 50, 7, padding=3)
        self.conv5_p = nn.Conv1d(50, 50, 5, padding=2)
        # self.conv6_p = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_p = nn.Linear(50 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, 1)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        # x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)

        return x

class S2P_on(nn.Module):
    def __init__(self, window_len):
        super(S2P_on, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_p = nn.Conv1d(30, 40, 9, padding=4)  # 6
        self.conv4_p = nn.Conv1d(40, 50, 7, padding=3)
        self.conv5_p = nn.Conv1d(50, 50, 5, padding=2)
        # self.conv6_p = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_p = nn.Linear(50 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, 1)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_s = nn.Conv1d(30, 40, 9, padding=4)  # 6
        self.conv4_s = nn.Conv1d(40, 50, 7, padding=3)
        self.conv5_s = nn.Conv1d(50, 50, 5, padding=2)
        # self.conv6_s = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_s = nn.Linear(50 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        # x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        # y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = torch.sigmoid(self.fc2_s(y))

        return x, y


class S2P_State(nn.Module):
    def __init__(self, window_len, state_num):
        super(S2P_State, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        # self.fc2_p = nn.Linear(512,32)
        self.fc3_p = nn.Linear(1024, 1)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        # self.fc2_s = nn.Linear(512,32)
        self.fc3_s = nn.Linear(1024, state_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        # x = F.relu(self.fc2_p(x))
        x = self.fc3_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        # y = F.relu(self.fc2_s(y))
        y = self.fc3_s(y)

        return x, y


class S2P_State2(nn.Module):
    def __init__(self, window_len, state_num):
        super(S2P_State2, self).__init__()
        self.conv11_p = nn.Conv1d(1, 50, 11, padding=5)  # 10
        self.conv12_p = nn.Conv1d(1, 50, 9, padding=4)  # 8
        self.conv13_p = nn.Conv1d(1, 50, 7, padding=3)  # 6
        self.conv2_p = nn.Conv1d(150, 50, 5, padding=2)
        self.fc1_p = nn.Linear(50 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, state_num)

        self.conv11_s = nn.Conv1d(1, 50, 11, padding=5)  # 10
        self.conv12_s = nn.Conv1d(1, 50, 9, padding=4)  # 8
        self.conv13_s = nn.Conv1d(1, 50, 7, padding=3)  # 6
        self.conv2_s = nn.Conv1d(150, 50, 5, padding=2)
        self.fc1_s = nn.Linear(50 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, state_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x1 = F.relu(self.conv11_p(x))
        x2 = F.relu(self.conv12_p(x))
        x3 = F.relu(self.conv13_p(x))
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.conv2_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)

        y1 = F.relu(self.conv11_s(y))
        y2 = F.relu(self.conv12_s(y))
        y3 = F.relu(self.conv13_s(y))
        y = torch.cat([y1, y2, y3], dim=1)
        y = F.relu(self.conv2_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = self.fc2_s(y)

        return x, y


class S2P_State_a(nn.Module):
    def __init__(self, window_len, state_num):
        super(S2P_State_a, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        # self.fc2_p = nn.Linear(512,32)
        self.fc3_p = nn.Linear(1024, state_num)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        # self.fc2_s = nn.Linear(512,32)
        self.fc3_s = nn.Linear(1024, state_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        # x = F.relu(self.fc2_p(x))
        x = self.fc3_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        # y = F.relu(self.fc2_s(y))
        y = self.fc3_s(y)

        return x, y


# on-off state
class S2S_on(nn.Module):
    def __init__(self, window_len, out_len):
        super(S2S_on, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 11, padding=5)  # 10
        self.conv2_p = nn.Conv1d(30, 30, 9, padding=4)  # 8
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 50, 5, padding=2)
        self.conv6_p = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_p = nn.Linear(50 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, out_len)

        self.conv1_s = nn.Conv1d(1, 30, 11, padding=5)  # 10
        self.conv2_s = nn.Conv1d(30, 30, 9, padding=4)  # 8
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 50, 5, padding=2)
        self.conv6_s = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_s = nn.Linear(50 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, out_len)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = torch.sigmoid(self.fc2_s(y))

        return x, y


class S2S_state(nn.Module):
    def __init__(self, window_len, out_len, state_num):
        super(S2S_state, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, out_len * state_num)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, out_len * state_num)

        self.crf = CRF(num_tags=state_num, batch_first=True)

    def calc_crf_loss(self, out, tgt):
        return -self.crf(out, tgt, reduction='mean')

    def decode(self, out):
        return self.crf.decode(out)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        # x.shape=[batchsize,]
        x = x.flatten(-2, -1)
        #         print('x.shape', x.shape)
        x = F.relu(self.fc1_p(x))
        #         print('x.shape', x.shape)
        x = self.fc2_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = self.fc2_s(y)

        return x, y


class S2S_state_sin_crf(nn.Module):
    def __init__(self, window_len, out_len, state_num):
        super(S2S_state_sin_crf, self).__init__()
        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, out_len * state_num)

    def decode(self, out):
        return self.crf.decode(out)

    def forward(self, y):
        y = y.unsqueeze(1)
        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = self.fc2_s(y)

        return y
        '''
class S2P_State_a(nn.Module):
    def __init__(self, window_len, state_num):
        super(S2P_State_a), self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)#10
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)#8
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)#6
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        #self.fc2_p = nn.Linear(512,32)
        self.fc3_p = nn.Linear(1024, state_num)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)#10
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)#8
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)#6
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        #self.fc2_s = nn.Linear(512,32)
        self.fc3_s = nn.Linear(1024, state_num)

        self.crf = CRF(num_tags=state_num, batch_first=True)

    def calc_crf_loss(self, out, tgt):
        return -self.crf(out, tgt, reduction='mean')

    def decode(self, out):
        return self.crf.decode(out)    

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        #x = F.relu(self.fc2_p(x))
        x = self.fc3_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        #y = F.relu(self.fc2_s(y))
        y = self.fc3_s(y)

        return x, y
        '''


class S2S_state_(nn.Module):
    def __init__(self, window_len, out_len, state_num):
        super(S2S_state_, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, out_len * state_num)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, out_len * state_num)

        self.crf = CRF(num_tags=state_num, batch_first=True)
        self.er = EntropyRegularization()

    def calc_crf_loss(self, out, tgt):
        return -self.crf(out, tgt, reduction='mean')

    def calc_er_loss(self, out):
        return self.er(out)

    def decode(self, out):
        return self.crf.decode(out)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = self.fc2_s(y)

        return x, y




