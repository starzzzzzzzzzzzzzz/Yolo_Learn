"""
YOLOv8 基础模块
核心改进：
1. 使用 SiLU 激活函数（而不是 Leaky ReLU）
2. C2f 模块：比 YOLOv3 的残差块更强，梯度流更丰富
"""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    标准卷积块：Conv + BN + SiLU
    
    YOLOv8 的改进：
    - 激活函数从 Leaky ReLU 改为 SiLU (Swish)
    - SiLU(x) = x * sigmoid(x)，更平滑，梯度更好
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super(Conv, self).__init__()
        # 自动计算 padding（保持特征图尺寸）
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)  # SiLU = x * sigmoid(x)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    瓶颈块：1×1 降维 -> 3×3 卷积 -> 残差连接
    
    与 YOLOv3 的区别：
    - YOLOv3: 1×1 降维 -> 3×3 升维
    - YOLOv8: 类似结构，但用 SiLU 激活
    """
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)  # 中间通道数（压缩）
        
        self.conv1 = Conv(in_channels, hidden_channels, 3, 1)  # 3×3 卷积
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1)  # 3×3 卷积
        self.add = shortcut and in_channels == out_channels  # 是否残差连接
    
    def forward(self, x):
        if self.add:
            return x + self.conv2(self.conv1(x))  # 残差连接
        else:
            return self.conv2(self.conv1(x))


class C2f(nn.Module):
    """
    C2f 模块：YOLOv8 的核心创新
    
    设计思想：
    - 名字：C2f = CSP (Cross Stage Partial) + 2 convolutions + faster
    - 比 YOLOv5 的 C3 模块更快、梯度流更丰富
    
    结构：
    1. 输入通过 1×1 卷积分成两部分
    2. 一部分直接传递，另一部分经过多个 Bottleneck
    3. 所有 Bottleneck 的输出都会被保留（Split 思想）
    4. 最后 concat 所有分支 + 1×1 卷积融合
    
    与 YOLOv3 ResidualBlock 的对比：
    - YOLOv3: 纯残差连接，梯度只有一条主路径
    - C2f: 多分支梯度流，信息更丰富
    """
    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcut=False, expansion=0.5):
        super(C2f, self).__init__()
        self.hidden_channels = int(out_channels * expansion)  # 隐藏层通道数
        
        # 第一次卷积：分流
        self.conv1 = Conv(in_channels, 2 * self.hidden_channels, 1, 1)
        
        # 多个 Bottleneck
        self.bottlenecks = nn.ModuleList(
            Bottleneck(self.hidden_channels, self.hidden_channels, shortcut, expansion=1.0)
            for _ in range(num_bottlenecks)
        )
        
        # 最后的融合卷积
        # 输入：hidden_channels（直接路径）+ num_bottlenecks * hidden_channels（所有 Bottleneck 输出）
        self.conv2 = Conv((2 + num_bottlenecks) * self.hidden_channels, out_channels, 1, 1)
    
    def forward(self, x):
        # 第一步：1×1 卷积，然后分成两部分
        y = self.conv1(x)
        y1, y2 = y.chunk(2, dim=1)  # 在通道维度分成两半
        
        # y1: 直接传递
        # y2: 经过 Bottleneck 链
        y_list = [y1, y2]  # 保存所有分支
        
        for bottleneck in self.bottlenecks:
            y2 = bottleneck(y2)
            y_list.append(y2)  # 保存每个 Bottleneck 的输出
        
        # 第二步：concat 所有分支 + 融合
        y = torch.cat(y_list, dim=1)  # 拼接所有分支
        return self.conv2(y)  # 1×1 卷积融合






class SPPF(nn.Module):
    """
    SPPF：Spatial Pyramid Pooling - Fast
    
    作用：增大感受野，捕获多尺度信息
    
    结构：
    - 连续 3 次 5×5 MaxPool（等效于 5×5, 9×9, 13×13 的并行池化）
    - 比原始 SPP 更快，效果相当
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        
        # 连续 3 次 MaxPool
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        
        # Concat 原始 x + 三次池化结果
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 基础模块测试")
    print("=" * 60)
    
    # 1. 测试 Conv
    print("\n1. Conv 模块：")
    conv = Conv(3, 32, 3, 1)
    x = torch.randn(1, 3, 640, 640)
    out = conv(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    
    # 2. 测试 C2f
    print("\n2. C2f 模块：")
    c2f = C2f(32, 64, num_bottlenecks=3)
    out = c2f(out)
    print(f"   输入: (1, 32, 640, 640)")
    print(f"   输出: {out.shape}")
    print(f"   说明: 通道数从 32 -> 64，空间尺寸不变")
    
    # 3. 测试 SPPF
    print("\n3. SPPF 模块：")
    sppf = SPPF(64, 64)
    out = sppf(out)
    print(f"   输入: (1, 64, 640, 640)")
    print(f"   输出: {out.shape}")
    print(f"   说明: 增大感受野，捕获多尺度信息")
    
    print("\n" + "=" * 60)
    print("✅ 所有基础模块测试通过！")
    print("=" * 60)
    
    # 4. 对比分析
    print("\n【核心对比】")
    print("┌────────────────┬─────────────────┬─────────────────┐")
    print("│ 模块           │ YOLOv3          │ YOLOv8          │")
    print("├────────────────┼─────────────────┼─────────────────┤")
    print("│ 激活函数       │ Leaky ReLU      │ SiLU            │")
    print("│ 特征提取       │ ResidualBlock   │ C2f             │")
    print("│ 梯度流         │ 单路径          │ 多分支          │")
    print("│ 空间池化       │ 无              │ SPPF            │")
    print("└────────────────┴─────────────────┴─────────────────┘")

