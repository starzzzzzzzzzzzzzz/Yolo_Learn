"""
Darknet-53 实现（YOLOv3 骨干网络）
核心特点：
1. 引入残差连接（ResNet 思想）
2. 53 层卷积网络
3. 没有池化层，用步长为 2 的卷积下采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================
# 1. 基础卷积块
# ================================

class ConvBlock(nn.Module):
    """基础卷积块：Conv + BN + Leaky ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        
        padding = (kernel_size - 1) // 2  # 保持尺寸
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding,
            bias=False  # BN 层会有 bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# ================================
# 2. 残差块（核心创新！）
# ================================

class ResidualBlock(nn.Module):
    """
    残差块（YOLOv3 的关键）
    
    结构：
    x → Conv 1×1 (降维) → Conv 3×3 (升维) → (+) x
                                              ↑
                                         残差连接
    """
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        # 1×1 卷积降维（减少计算量）
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1)
        
        # 3×3 卷积升维（提取特征）
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3)
    
    def forward(self, x):
        residual = x  # 保存输入（残差连接）
        
        x = self.conv1(x)  # 1×1 降维
        x = self.conv2(x)  # 3×3 升维
        
        x = x + residual  # 残差连接（关键！）
        
        return x


# ================================
# 3. 残差块组（多个残差块堆叠）
# ================================

class ResidualBlockGroup(nn.Module):
    """
    残差块组：先下采样，再堆叠 n 个残差块
    """
    
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ResidualBlockGroup, self).__init__()
        
        # 下采样（用步长为 2 的卷积代替池化）
        self.downsample = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)
        
        # 堆叠 num_blocks 个残差块
        self.blocks = nn.ModuleList([
            ResidualBlock(out_channels) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x = self.downsample(x)  # 下采样
        
        for block in self.blocks:
            x = block(x)  # 残差块
        
        return x


# ================================
# 4. 完整的 Darknet-53
# ================================

class Darknet53(nn.Module):
    """
    Darknet-53 骨干网络
    
    架构：
    输入 416×416
    ├─ Conv (stride=1)      → 416×416×32
    ├─ ResGroup×1 (stride=2) → 208×208×64
    ├─ ResGroup×2 (stride=2) → 104×104×128
    ├─ ResGroup×8 (stride=2) → 52×52×256   ← 输出1（用于 52×52 预测）
    ├─ ResGroup×8 (stride=2) → 26×26×512   ← 输出2（用于 26×26 预测）
    └─ ResGroup×4 (stride=2) → 13×13×1024  ← 输出3（用于 13×13 预测）
    
    总层数：1 + 2*(1+2+8+8+4) = 1 + 2*23 = 1 + 46 = 47？
    实际：1 + (2+4+16+16+8) = 1 + 46 = 47 卷积层
          + 5 个下采样卷积 + 1 = 53 层！
    """
    
    def __init__(self):
        super(Darknet53, self).__init__()
        
        # 初始卷积（不下采样）
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1)
        
        # 残差块组（逐步下采样）
        self.group1 = ResidualBlockGroup(32, 64, num_blocks=1)    # 1 个残差块
        self.group2 = ResidualBlockGroup(64, 128, num_blocks=2)   # 2 个残差块
        self.group3 = ResidualBlockGroup(128, 256, num_blocks=8)  # 8 个残差块 ← 输出用于 52×52
        self.group4 = ResidualBlockGroup(256, 512, num_blocks=8)  # 8 个残差块 ← 输出用于 26×26
        self.group5 = ResidualBlockGroup(512, 1024, num_blocks=4) # 4 个残差块 ← 输出用于 13×13
    
    def forward(self, x):
        """
        x: (batch, 3, 416, 416)
        返回 3 个不同尺度的特征图
        """
        x = self.conv1(x)    # (B, 32, 416, 416)
        
        x = self.group1(x)   # (B, 64, 208, 208)
        x = self.group2(x)   # (B, 128, 104, 104)
        
        out_52 = self.group3(x)  # (B, 256, 52, 52)   ← 输出1（细节丰富）
        out_26 = self.group4(out_52)  # (B, 512, 26, 26)  ← 输出2（中等语义）
        out_13 = self.group5(out_26)  # (B, 1024, 13, 13) ← 输出3（高层语义）
        
        return out_52, out_26, out_13


# ================================
# 5. 对比 Darknet-19 和 Darknet-53
# ================================

def compare_architectures():
    """对比两种架构"""
    print("=" * 80)
    print("Darknet-19 vs Darknet-53 对比")
    print("=" * 80)
    
    comparison = """
    ┌─────────────────┬──────────────────────┬─────────────────────────┐
    │     特性        │    Darknet-19        │      Darknet-53         │
    ├─────────────────┼──────────────────────┼─────────────────────────┤
    │ 层数            │ 19 层                │ 53 层                   │
    │ 残差连接        │ 无 ❌                │ 有 ✅（关键！）         │
    │ 下采样方式      │ MaxPool              │ 步长为2的卷积           │
    │ 输出            │ 1 个 (13×13)         │ 3 个 (52×52,26×26,13×13)│
    │ 梯度流          │ 较弱                 │ 强（残差）              │
    │ 深度瓶颈        │ 有（难训练）         │ 无（残差解决）          │
    │ 参数量          │ ~50M                 │ ~42M（更高效）          │
    │ 性能            │ 较好                 │ 更好 ⭐                 │
    └─────────────────┴──────────────────────┴─────────────────────────┘
    
    关键区别：
    1. 残差连接（ResNet 思想）
       • 解决梯度消失问题
       • 允许网络更深
       • 性能提升明显
    
    2. 多尺度输出
       • Darknet-19: 只输出 13×13
       • Darknet-53: 输出 52×52, 26×26, 13×13
       • 为 YOLOv3 的多尺度预测提供基础
    
    3. 去掉 MaxPool
       • 用步长为 2 的卷积下采样
       • 避免信息丢失
       • 更平滑的特征传递
    """
    
    print(comparison)


# ================================
# 6. 残差块详解
# ================================

def explain_residual_block():
    """解释残差块的工作原理"""
    print("\n" + "=" * 80)
    print("残差块（Residual Block）详解")
    print("=" * 80)
    
    explanation = """
    普通网络的问题：
    ┌─────┐    ┌─────┐    ┌─────┐
    │  x  │ → │Conv1│ → │Conv2│ → │ out │
    └─────┘    └─────┘    └─────┘
    
    问题：
    • 网络越深，梯度越难传回
    • 性能下降（退化问题）
    • 难以训练
    
    
    残差块的解决方案：
    ┌─────┐    ┌─────┐    ┌─────┐
    │  x  │ → │Conv1│ → │Conv2│ → │ (+) │ → │ out │
    └──┬──┘    └─────┘    └─────┘    └──▲──┘
       │                                  │
       └──────────────────────────────────┘
              残差连接（跳跃连接）
    
    公式：
    out = F(x) + x
         ↑      ↑
       卷积   原始输入
    
    优势：
    1. 梯度可以直接通过"捷径"回传 ✅
    2. 学习残差（更容易） ✅
    3. 允许网络更深 ✅
    4. 性能更好 ✅
    
    
    YOLOv3 的残差块设计：
    
    输入 (256 通道)
      ↓
    Conv 1×1, 128  (降维，减少计算)
      ↓
    Conv 3×3, 256  (升维，提取特征)
      ↓
    (+) 输入        (残差连接)
      ↓
    输出 (256 通道)
    
    为什么 1×1 → 3×3？
    • 1×1 降维：256 → 128（减少 3×3 的计算量）
    • 3×3 特征提取：在低维空间提取特征
    • 再升回 256：保持通道数一致
    • 这叫 "bottleneck" 设计（瓶颈结构）
    """
    
    print(explanation)


# ================================
# 7. 测试和验证
# ================================

def test_darknet53():
    """测试 Darknet-53"""
    print("\n" + "=" * 80)
    print("Darknet-53 测试")
    print("=" * 80)
    
    # 创建模型
    model = Darknet53()
    model.eval()
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ 模型创建成功")
    print(f"   总参数量: {total_params:,}")
    
    # 测试前向传播
    print("\n🔄 测试前向传播:")
    x = torch.randn(1, 3, 416, 416)
    
    with torch.no_grad():
        out_52, out_26, out_13 = model(x)
    
    print(f"   输入: {x.shape}")
    print(f"   输出1 (52×52): {out_52.shape}  ← 用于检测小目标")
    print(f"   输出2 (26×26): {out_26.shape}  ← 用于检测中目标")
    print(f"   输出3 (13×13): {out_13.shape}  ← 用于检测大目标")
    
    # 验证残差块
    print("\n🔍 验证残差块:")
    res_block = ResidualBlock(256)
    x_test = torch.randn(1, 256, 52, 52)
    
    with torch.no_grad():
        out = res_block(x_test)
    
    print(f"   输入: {x_test.shape}")
    print(f"   输出: {out.shape}")
    print(f"   ✅ 形状保持不变（残差连接的特点）")


# ================================
# 8. 架构可视化
# ================================

def visualize_architecture():
    """可视化架构"""
    print("\n" + "=" * 80)
    print("Darknet-53 完整架构")
    print("=" * 80)
    
    architecture = """
    输入: 416×416×3
    ├─ Conv 3×3, 32, stride=1      → 416×416×32
    │
    ├─ ┌─ Conv 3×3, 64, stride=2   → 208×208×64
    │  └─ ResBlock×1
    │
    ├─ ┌─ Conv 3×3, 128, stride=2  → 104×104×128
    │  └─ ResBlock×2
    │
    ├─ ┌─ Conv 3×3, 256, stride=2  → 52×52×256  ──┐
    │  └─ ResBlock×8                              │ 输出1（细节）
    │                                              │ 用于检测小目标
    │                                              │
    ├─ ┌─ Conv 3×3, 512, stride=2  → 26×26×512 ──┤
    │  └─ ResBlock×8                              │ 输出2（中等）
    │                                              │ 用于检测中目标
    │                                              │
    └─ ┌─ Conv 3×3, 1024, stride=2 → 13×13×1024──┘
       └─ ResBlock×4                              │ 输出3（语义）
                                                  │ 用于检测大目标
    
    残差块（ResBlock）结构：
    x → Conv1×1(降维) → Conv3×3(升维) → (+)x → out
    
    总层数计算：
    • 1 个初始 Conv
    • 5 个下采样 Conv
    • (1+2+8+8+4) × 2 = 46 个残差块内的 Conv
    • 总计：1 + 5 + 46 = 52 ≈ 53 层（算法略有不同）
    """
    
    print(architecture)


# ================================
# 主函数
# ================================

if __name__ == "__main__":
    # 对比架构
    compare_architectures()
    
    # 解释残差块
    explain_residual_block()
    
    # 测试模型
    test_darknet53()
    
    # 可视化架构
    visualize_architecture()
    
    print("\n" + "=" * 80)
    print("🎉 Darknet-53 学习完成！")
    print("=" * 80)
    print("\n关键要点:")
    print("1. 残差连接（ResNet 思想）← 核心创新")
    print("2. 更深的网络（53 层 vs 19 层）")
    print("3. 多尺度输出（52×52, 26×26, 13×13）")
    print("4. 步长为 2 的卷积下采样（代替 MaxPool）")
    print("5. Bottleneck 设计（1×1 降维 → 3×3 特征 → 升维）")

