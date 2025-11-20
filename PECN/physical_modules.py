import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import *
from .layers import *
from typing import List

# ----------- 改进CBAM模块 -------------
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        # 通道注意力
        self.channel_attention = ChannelAttention(in_planes, ratio)
        # 空间注意力
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 增加中间层通道数
        mid_channels = max(in_planes // ratio, 8)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid_channels, 1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_planes, 1, bias=False),
            nn.GroupNorm(8, in_planes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.GroupNorm(1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.conv(concat)
        return self.sigmoid(spatial_attn)


# ----------- 自适应融合模块 -------------
class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.GroupNorm(8, channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        # 初始化为较小的值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        channel_weights = self.channel_attention(concat)
        spatial_weights = self.spatial_attention(concat)
        out = x2 * channel_weights * spatial_weights + x1
        return out

# ----------- 改进非局部注意力模块 -------------
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inter_channels = in_channels // 2 if in_channels > 1 else 1
        
        # 添加输入平滑
        self.input_smooth = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        
        # 改进输出变换
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1),
            nn.GroupNorm(8, in_channels)
        )
        # 更保守的初始化
        nn.init.normal_(self.W[0].weight, 0, 0.01)
        nn.init.constant_(self.W[0].bias, 0)

    def forward(self, x):
        x = self.input_smooth(x)
        batch, c, h, w = x.size()
        
        g_x = self.g(x).view(batch, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x).view(batch, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch, self.inter_channels, h, w)
        
        W_y = self.W(y)
        # 添加残差连接的权重
        z = W_y * 0.1 + x
        return z



# ----------- 改进Context Block实现 -------------
class ContextBlock(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        mid_channels = max(in_channels // ratio, 8)
        
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.GroupNorm(8, in_channels)
        )
        
        # 添加局部上下文
        self.local_context = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_feat = self.global_context(x)
        local_feat = self.local_context(x)
        context = self.sigmoid(global_feat + local_feat)
        return x * context

# ----------- 改进多尺度特征变换网络 -------------
class MultiScaleTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=3, dilation=3),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合和平滑
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels*4, 3, padding=1),
            nn.GroupNorm(8, out_channels*4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        out = torch.cat([b1, b3, b5, b7], dim=1)
        out = self.fusion(out)
        return out

# ----------- 改进物理增强模块 -------------
class PhysicalAwareEnhancement(nn.Module):
    def __init__(self, in_channels, mid_channels=64, feature_channels=None):
        super().__init__()
        # 增加中间通道数
        self.mid_channels = min(mid_channels*2, in_channels)
        # 特征提取
        self.ms_transform = MultiScaleTransform(in_channels, self.mid_channels)
        # 注意力模块串联
        self.nonlocal_block = NonLocalBlock(self.mid_channels*4)
        self.cbam = CBAM(self.mid_channels*4)
        self.context = ContextBlock(self.mid_channels*4)
        # 每个注意力模块后都加GroupNorm
        self.bn1 = nn.GroupNorm(8, self.mid_channels*4)
        self.bn2 = nn.GroupNorm(8, self.mid_channels*4)
        self.bn3 = nn.GroupNorm(8, self.mid_channels*4)
        # 渐进式特征压缩
        self.compress = nn.Sequential(
            nn.Conv2d(self.mid_channels*4, self.mid_channels*2, 1),
            nn.GroupNorm(8, self.mid_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels*2, in_channels, 1),
            nn.GroupNorm(8, in_channels)
        )
        # 改进自适应融合
        self.fusion = AdaptiveFusion(in_channels)

    def forward(self, x):
        identity = x
        # 特征提取
        out = self.ms_transform(x)
        # 注意力模块串联，每个后面都加GroupNorm
        out = self.nonlocal_block(out)
        out = self.bn1(out)
        out = self.cbam(out)
        out = self.bn2(out)
        out = self.context(out)
        out = self.bn3(out)
        # 特征压缩
        out = self.compress(out)
        # 自适应融合
        out = self.fusion(identity, out)
        # 添加全局残差连接
        return out + identity * 0.1

