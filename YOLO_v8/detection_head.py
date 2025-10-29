"""
YOLOv8 解耦检测头（Decoupled Head）+ Anchor-free 预测

核心创新：
1. 解耦头：分类和定位分支独立（互不干扰）
2. Anchor-free：不再依赖预设锚框，直接预测中心点 + 宽高
3. DFL（Distribution Focal Loss）：用分布表示边界框，更精准
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_modules import Conv


class DFL(nn.Module):
    """
    Distribution Focal Loss - 分布式焦点损失
    
    核心思想：
    - 传统方法：直接回归一个数值（如 x, y, w, h）
    - DFL 方法：用一个分布来表示数值
    
    例如：预测边界框的左边界 x1
    - 传统：x1 = 100（一个固定值）
    - DFL：x1 可能是 99 (40%), 100 (50%), 101 (10%)（一个概率分布）
    
    优势：
    - 更好地处理边界的不确定性
    - 让模型对边界框更"自信"
    """
    def __init__(self, num_bins=16):
        """
        Args:
            num_bins: 分布的区间数量（通常是 16）
        """
        super(DFL, self).__init__()
        self.num_bins = num_bins
        
        # 生成区间权重：[0, 1, 2, ..., 15]
        self.register_buffer('project', torch.arange(num_bins, dtype=torch.float))
    
    def forward(self, x):
        """
        输入：(B, 4 * num_bins, H, W) - 4 个边界 × num_bins 个区间
        输出：(B, 4, H, W) - 4 个边界值
        
        流程：
        1. Reshape 成 (B, 4, num_bins, H, W)
        2. Softmax 归一化每个分布
        3. 加权求和得到最终值
        """
        # Reshape: (B, 4 * num_bins, H, W) -> (B, 4, num_bins, H, W)
        batch_size, _, height, width = x.shape
        x = x.view(batch_size, 4, self.num_bins, height, width)
        
        # Softmax: 对每个分布归一化
        x = F.softmax(x, dim=2)  # 在 num_bins 维度上 softmax
        
        # 加权求和：分布 → 单一数值
        # project: [0, 1, 2, ..., 15]
        # x: 概率分布
        # 结果：加权平均值
        x = (x * self.project.view(1, 1, self.num_bins, 1, 1)).sum(dim=2)
        
        return x


class DecoupledHead(nn.Module):
    """
    YOLOv8 解耦检测头
    
    架构对比：
    ┌──────────────────────────────────────────────────────────┐
    │ YOLOv2/v3（耦合头）:                                     │
    │   Input → Conv → Conv → [cls, x, y, w, h, conf]        │
    │   (分类和定位共享特征)                                    │
    └──────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────┐
    │ YOLOv8（解耦头）:                                        │
    │                 ┌→ Conv → Conv → [cls]（分类分支）      │
    │   Input → Split─┤                                        │
    │                 └→ Conv → Conv → [x, y, w, h]（定位分支）│
    │   (分类和定位独立，互不干扰)                              │
    └──────────────────────────────────────────────────────────┘
    
    优势：
    1. 分类和定位任务特性不同，独立学习效果更好
    2. 避免两个任务相互干扰
    3. 更容易优化
    """
    def __init__(self, num_classes=80, in_channels=256, num_bins=16):
        """
        Args:
            num_classes: 类别数量（COCO = 80）
            in_channels: 输入通道数
            num_bins: DFL 分布区间数
        """
        super(DecoupledHead, self).__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        
        # 分类分支（2 层卷积）
        self.cls_convs = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            Conv(in_channels, in_channels, 3, 1)
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)  # 最终分类预测
        
        # 定位分支（2 层卷积）
        self.reg_convs = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            Conv(in_channels, in_channels, 3, 1)
        )
        # 预测 4 个边界 × num_bins 个区间
        self.reg_pred = nn.Conv2d(in_channels, 4 * num_bins, 1)
        
        # DFL 模块：分布 → 数值
        self.dfl = DFL(num_bins)
    
    def forward(self, x):
        """
        输入：(B, in_channels, H, W)
        输出：
            cls_output: (B, num_classes, H, W)
            reg_output: (B, 4, H, W) - 经过 DFL 处理后的边界框
        """
        # 分类分支
        cls_feat = self.cls_convs(x)
        cls_output = self.cls_pred(cls_feat)  # (B, num_classes, H, W)
        
        # 定位分支
        reg_feat = self.reg_convs(x)
        reg_output = self.reg_pred(reg_feat)  # (B, 4 * num_bins, H, W)
        
        # DFL：分布 → 数值
        reg_output = self.dfl(reg_output)  # (B, 4, H, W)
        
        return cls_output, reg_output


class AnchorFreeDecoder(nn.Module):
    """
    Anchor-free 解码器
    
    与 YOLOv2/v3 的区别：
    ┌──────────────────────────────────────────────────────────┐
    │ YOLOv2/v3（Anchor-based）:                              │
    │   预测：tx, ty, tw, th（相对于锚框的偏移）               │
    │   解码：                                                 │
    │     bx = σ(tx) + cx                                      │
    │     by = σ(ty) + cy                                      │
    │     bw = pw * exp(tw)                                    │
    │     bh = ph * exp(th)                                    │
    │   需要：预设锚框 (pw, ph)                                │
    └──────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────┐
    │ YOLOv8（Anchor-free）:                                  │
    │   预测：ltrb（距离中心点的 left, top, right, bottom）   │
    │   解码：                                                 │
    │     x1 = cx - left                                       │
    │     y1 = cy - top                                        │
    │     x2 = cx + right                                      │
    │     y2 = cy + bottom                                     │
    │   无需：预设锚框，更灵活                                 │
    └──────────────────────────────────────────────────────────┘
    """
    def __init__(self, num_bins=16):
        super(AnchorFreeDecoder, self).__init__()
        self.num_bins = num_bins
    
    def forward(self, cls_output, reg_output, stride=32):
        """
        解码预测结果
        
        Args:
            cls_output: (B, num_classes, H, W) 分类预测
            reg_output: (B, 4, H, W) 定位预测（ltrb 格式）
            stride: 下采样倍数（32, 16, 8）
        
        Returns:
            boxes: (B, H*W, 4) 边界框 [x1, y1, x2, y2]
            scores: (B, H*W, num_classes) 类别分数
        """
        batch_size, num_classes, height, width = cls_output.shape
        device = cls_output.device
        
        # 1. 生成网格中心点坐标
        # 例如：13×13 网格，stride=32
        # cx: [[0, 1, 2, ..., 12], [0, 1, 2, ..., 12], ...]
        # cy: [[0, 0, 0, ..., 0], [1, 1, 1, ..., 1], ...]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # 转换到图像坐标
        grid_x = grid_x.float() * stride + stride / 2  # 网格中心 x
        grid_y = grid_y.float() * stride + stride / 2  # 网格中心 y
        
        # 2. 解码边界框
        # reg_output: (B, 4, H, W) -> (B, H, W, 4)
        reg_output = reg_output.permute(0, 2, 3, 1)
        
        # 提取 ltrb（距离中心点的 left, top, right, bottom）
        left = reg_output[..., 0] * stride
        top = reg_output[..., 1] * stride
        right = reg_output[..., 2] * stride
        bottom = reg_output[..., 3] * stride
        
        # 计算边界框坐标
        x1 = grid_x - left
        y1 = grid_y - top
        x2 = grid_x + right
        y2 = grid_y + bottom
        
        # 拼接成 (B, H, W, 4)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # 3. 处理分类分数
        # cls_output: (B, num_classes, H, W) -> (B, H, W, num_classes)
        scores = cls_output.permute(0, 2, 3, 1).sigmoid()
        
        # 4. Flatten 为 (B, H*W, ...)
        boxes = boxes.view(batch_size, -1, 4)
        scores = scores.view(batch_size, -1, num_classes)
        
        return boxes, scores


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 解耦检测头测试")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    in_channels = 256
    height, width = 20, 20  # 20×20 特征图
    num_classes = 80
    stride = 32
    
    # 1. 测试 DFL
    print("\n1. DFL 模块：")
    dfl = DFL(num_bins=16)
    x = torch.randn(batch_size, 4 * 16, height, width)
    out = dfl(x)
    print(f"   输入: {x.shape} (4 个边界 × 16 个区间)")
    print(f"   输出: {out.shape} (4 个边界值)")
    
    # 2. 测试解耦检测头
    print("\n2. 解耦检测头：")
    head = DecoupledHead(num_classes=num_classes, in_channels=in_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    cls_output, reg_output = head(x)
    print(f"   输入: {x.shape}")
    print(f"   分类输出: {cls_output.shape} ({num_classes} 类)")
    print(f"   定位输出: {reg_output.shape} (4 个边界)")
    
    # 3. 测试 Anchor-free 解码
    print("\n3. Anchor-free 解码：")
    decoder = AnchorFreeDecoder()
    boxes, scores = decoder(cls_output, reg_output, stride=stride)
    print(f"   边界框: {boxes.shape} (H*W={height*width} 个预测)")
    print(f"   分数: {scores.shape}")
    print(f"   说明: 每个网格点直接预测边界框，无需锚框")
    
    # 4. 查看具体数值
    print("\n4. 示例预测（第一个样本的前 3 个框）：")
    print(f"   Boxes (x1, y1, x2, y2):")
    for i in range(3):
        box = boxes[0, i].tolist()
        print(f"      {i+1}. [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    print("\n" + "=" * 60)
    print("✅ 解耦检测头测试通过！")
    print("=" * 60)
    
    # 5. 核心对比
    print("\n【Anchor-based vs Anchor-free】")
    print("┌────────────────────┬─────────────────────────────────┐")
    print("│ 特性               │ YOLOv2/v3 (Anchor)  │ YOLOv8 (Anchor-free) │")
    print("├────────────────────┼─────────────────────────────────┤")
    print("│ 预测内容           │ tx, ty, tw, th      │ ltrb (距离边界)      │")
    print("│ 是否需要锚框       │ ✅ 需要              │ ❌ 不需要            │")
    print("│ 灵活性             │ 受限于锚框          │ 更灵活               │")
    print("│ 训练难度           │ 需调整锚框          │ 端到端               │")
    print("└────────────────────┴─────────────────────────────────┘")

