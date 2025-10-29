"""
YOLOv3 FPN 风格特征融合
展示如何从 Darknet-53 的三个输出构建完整的 YOLOv3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================
# 1. 基础卷积块（复用）
# ================================

class ConvBlock(nn.Module):
    """Conv + BN + Leaky ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


# ================================
# 2. YOLOv3 检测块
# ================================

class YOLOv3DetectionBlock(nn.Module):
    """
    YOLOv3 的检测块
    5 个卷积层 + 检测层
    """
    def __init__(self, in_channels, num_anchors=3, num_classes=80):
        super(YOLOv3DetectionBlock, self).__init__()
        
        # 5 个卷积层（特征提取）
        self.conv1 = ConvBlock(in_channels, in_channels // 2, 1)
        self.conv2 = ConvBlock(in_channels // 2, in_channels, 3)
        self.conv3 = ConvBlock(in_channels, in_channels // 2, 1)
        self.conv4 = ConvBlock(in_channels // 2, in_channels, 3)
        self.conv5 = ConvBlock(in_channels, in_channels // 2, 1)
        
        # 检测层前的卷积
        self.conv6 = ConvBlock(in_channels // 2, in_channels, 3)
        
        # 检测层（预测）
        self.conv7 = nn.Conv2d(
            in_channels, 
            num_anchors * (5 + num_classes),  # 每个 anchor: tx,ty,tw,th,conf + classes
            kernel_size=1
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # ← 这个特征会用于上采样
        
        route = x  # 保存用于后续融合
        
        x = self.conv6(x)
        detection = self.conv7(x)  # 检测输出
        
        return detection, route


# ================================
# 3. YOLOv3 完整网络（FPN 融合）
# ================================

class YOLOv3(nn.Module):
    """
    YOLOv3 完整网络
    核心：FPN 风格的自顶向下特征融合
    """
    
    def __init__(self, backbone, num_anchors=3, num_classes=80):
        super(YOLOv3, self).__init__()
        
        self.backbone = backbone
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # ===== 第一个检测头（13×13，大目标）=====
        self.detection_block_1 = YOLOv3DetectionBlock(1024, num_anchors, num_classes)
        
        # 用于上采样的卷积
        self.upsample_conv_1 = ConvBlock(512, 256, 1)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # ===== 第二个检测头（26×26，中目标）=====
        # 输入：256 (上采样) + 512 (骨干) = 768
        self.detection_block_2 = YOLOv3DetectionBlock(768, num_anchors, num_classes)
        
        # 用于上采样的卷积
        self.upsample_conv_2 = ConvBlock(384, 128, 1)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # ===== 第三个检测头（52×52，小目标）=====
        # 输入：128 (上采样) + 256 (骨干) = 384
        self.detection_block_3 = YOLOv3DetectionBlock(384, num_anchors, num_classes)
    
    def forward(self, x):
        """
        x: (B, 3, 416, 416)
        返回 3 个不同尺度的检测结果
        """
        # 骨干网络提取特征
        out_52, out_26, out_13 = self.backbone(x)
        # out_52: (B, 256, 52, 52)  ← 浅层，细节丰富
        # out_26: (B, 512, 26, 26)  ← 中层
        # out_13: (B, 1024, 13, 13) ← 深层，语义丰富
        
        # ========================================
        # 第 1 步：13×13 检测（大目标）
        # ========================================
        detection_13, route_13 = self.detection_block_1(out_13)
        # detection_13: (B, 3*(5+80), 13, 13) - 检测输出
        # route_13: (B, 512, 13, 13) - 用于上采样
        
        # ========================================
        # 第 2 步：13×13 → 26×26 融合
        # ========================================
        # 上采样
        x = self.upsample_conv_1(route_13)  # (B, 512, 13, 13) → (B, 256, 13, 13)
        x = self.upsample_1(x)              # (B, 256, 13, 13) → (B, 256, 26, 26)
        
        # 融合：深层上采样 + 浅层特征
        x = torch.cat([x, out_26], dim=1)   # (B, 256, 26, 26) + (B, 512, 26, 26) = (B, 768, 26, 26)
        
        # 26×26 检测（中目标）
        detection_26, route_26 = self.detection_block_2(x)
        # detection_26: (B, 3*(5+80), 26, 26) - 检测输出
        # route_26: (B, 384, 26, 26) - 用于上采样
        
        # ========================================
        # 第 3 步：26×26 → 52×52 融合
        # ========================================
        # 上采样
        x = self.upsample_conv_2(route_26)  # (B, 384, 26, 26) → (B, 128, 26, 26)
        x = self.upsample_2(x)              # (B, 128, 26, 26) → (B, 128, 52, 52)
        
        # 融合：中层上采样 + 浅层特征
        x = torch.cat([x, out_52], dim=1)   # (B, 128, 52, 52) + (B, 256, 52, 52) = (B, 384, 52, 52)
        
        # 52×52 检测（小目标）
        detection_52, _ = self.detection_block_3(x)
        # detection_52: (B, 3*(5+80), 52, 52) - 检测输出
        
        return detection_13, detection_26, detection_52


# ================================
# 4. 完整流程可视化
# ================================

def visualize_fpn_flow():
    """可视化 FPN 融合流程"""
    print("=" * 80)
    print("YOLOv3 FPN 融合流程（自顶向下）")
    print("=" * 80)
    
    flow = """
    骨干网络 Darknet-53 输出:
    ┌─────────────────────────────────────┐
    │ out_52: (B, 256, 52, 52)   浅层    │ ← 细节丰富，定位准
    │ out_26: (B, 512, 26, 26)   中层    │
    │ out_13: (B, 1024, 13, 13)  深层    │ ← 语义丰富，识别准
    └─────────────────────────────────────┘
    
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    第 1 层：13×13 检测（大目标）
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    out_13 (1024, 13, 13)
        ↓
    [DetectionBlock] ← 5个卷积提取特征
        ↓
    ├─→ detection_13 (255, 13, 13)  ✅ 第1个检测输出（大目标）
    └─→ route_13 (512, 13, 13)      → 用于上采样
    
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    第 2 层：26×26 融合 + 检测（中目标）
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    route_13 (512, 13, 13)
        ↓ Conv 1×1 降维
    (256, 13, 13)
        ↓ Upsample 2×  ← 上采样！
    (256, 26, 26)
        ↓
        ├─────────┐ Concat  ← 融合！
        ↓         ↓
    (256, 26, 26) + out_26 (512, 26, 26)
        ↓
    (768, 26, 26)  ← 融合后的特征
        ↓
    [DetectionBlock]
        ↓
    ├─→ detection_26 (255, 26, 26)  ✅ 第2个检测输出（中目标）
    └─→ route_26 (384, 26, 26)      → 用于上采样
    
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    第 3 层：52×52 融合 + 检测（小目标）
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    route_26 (384, 26, 26)
        ↓ Conv 1×1 降维
    (128, 26, 26)
        ↓ Upsample 2×  ← 上采样！
    (128, 52, 52)
        ↓
        ├─────────┐ Concat  ← 融合！
        ↓         ↓
    (128, 52, 52) + out_52 (256, 52, 52)
        ↓
    (384, 52, 52)  ← 融合后的特征
        ↓
    [DetectionBlock]
        ↓
    detection_52 (255, 52, 52)  ✅ 第3个检测输出（小目标）
    
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    最终输出
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    detection_13: (B, 255, 13, 13)  → 13×13×3 = 507 个大目标框
    detection_26: (B, 255, 26, 26)  → 26×26×3 = 2028 个中目标框
    detection_52: (B, 255, 52, 52)  → 52×52×3 = 8112 个小目标框
    
    总计：10647 个预测框！
    """
    
    print(flow)


# ================================
# 5. 关键代码片段对比
# ================================

def compare_fusion_methods():
    """对比 YOLOv2 和 YOLOv3 的融合方式"""
    print("\n" + "=" * 80)
    print("YOLOv2 vs YOLOv3 融合方式对比")
    print("=" * 80)
    
    comparison = """
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                    YOLOv2 Passthrough                                  ║
    ╚════════════════════════════════════════════════════════════════════════╝
    
    26×26×512 (浅层)
        ↓ Passthrough (space-to-depth)
    13×13×2048
        ↓ Concat
    13×13×(1024+2048) = 13×13×3072
        ↓ 卷积
    13×13 预测 ← 只有一个输出
    
    特点：
    • 单向融合（浅 → 深）
    • 只有一个预测尺度
    • Passthrough 是空间到通道的重组
    
    
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                    YOLOv3 FPN 风格                                     ║
    ╚════════════════════════════════════════════════════════════════════════╝
    
    深层 13×13×1024
        ↓ 检测
    13×13 预测 ← 输出1（大目标）✅
        ↓ Upsample + Conv
    13×13×512 → 26×26×256
        ↓ Concat ← 融合
    26×26×(256+512) = 26×26×768
        ↓ 检测
    26×26 预测 ← 输出2（中目标）✅
        ↓ Upsample + Conv
    26×26×384 → 52×52×128
        ↓ Concat ← 融合
    52×52×(128+256) = 52×52×384
        ↓ 检测
    52×52 预测 ← 输出3（小目标）✅
    
    特点：
    • 自顶向下融合（深 → 浅）
    • 三个预测尺度
    • 每一层都融合 + 预测
    • 类似 FPN (Feature Pyramid Network)
    
    
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                      核心区别                                          ║
    ╚════════════════════════════════════════════════════════════════════════╝
    
    ┌──────────────┬─────────────────────┬──────────────────────────┐
    │   特性       │      YOLOv2         │         YOLOv3           │
    ├──────────────┼─────────────────────┼──────────────────────────┤
    │ 融合方向     │ 浅 → 深（单向）     │ 深 → 浅（自顶向下）⭐   │
    │ 融合方式     │ Passthrough 重组    │ Upsample + Concat ⭐     │
    │ 融合次数     │ 1 次                │ 2 次（更充分）           │
    │ 预测尺度     │ 1 个 (13×13)        │ 3 个 (13,26,52) ⭐       │
    │ 预测框数     │ 845                 │ 10647 (更多)             │
    │ 小目标检测   │ 较弱                │ 强 ⭐⭐⭐                │
    └──────────────┴─────────────────────┴──────────────────────────┘
    """
    
    print(comparison)


# ================================
# 6. 测试
# ================================

def test_yolov3_fpn():
    """测试 YOLOv3 FPN 融合"""
    print("\n" + "=" * 80)
    print("YOLOv3 FPN 测试")
    print("=" * 80)
    
    # 简化的骨干网络（模拟 Darknet-53）
    class SimplifiedBackbone(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            B = x.size(0)
            # 模拟三个输出
            out_52 = torch.randn(B, 256, 52, 52)
            out_26 = torch.randn(B, 512, 26, 26)
            out_13 = torch.randn(B, 1024, 13, 13)
            return out_52, out_26, out_13
    
    # 创建模型
    backbone = SimplifiedBackbone()
    model = YOLOv3(backbone, num_anchors=3, num_classes=80)
    model.eval()
    
    # 测试
    x = torch.randn(2, 3, 416, 416)
    
    with torch.no_grad():
        det_13, det_26, det_52 = model(x)
    
    print(f"\n输入: {x.shape}")
    print(f"\n三个检测输出:")
    print(f"  13×13: {det_13.shape}  → {13*13*3} 个框（大目标）")
    print(f"  26×26: {det_26.shape}  → {26*26*3} 个框（中目标）")
    print(f"  52×52: {det_52.shape}  → {52*52*3} 个框（小目标）")
    print(f"\n总预测框数: {13*13*3 + 26*26*3 + 52*52*3} 个")
    
    print("\n✅ FPN 融合成功！")
    print("\n关键点:")
    print("  1. 自顶向下（13 → 26 → 52）")
    print("  2. 每层都融合：Upsample + Concat")
    print("  3. 每层都预测：3 个独立检测头")
    print("  4. 小目标检测大幅提升（52×52 高分辨率）")


# ================================
# 主函数
# ================================

if __name__ == "__main__":
    # 可视化融合流程
    visualize_fpn_flow()
    
    # 对比融合方法
    compare_fusion_methods()
    
    # 测试
    test_yolov3_fpn()
    
    print("\n" + "=" * 80)
    print("🎉 YOLOv3 FPN 融合学习完成！")
    print("=" * 80)
    print("\n核心要点:")
    print("  1. 自顶向下特征融合（深层语义 → 浅层细节）")
    print("  2. Upsample + Concat（而不是 add）")
    print("  3. 三个独立的检测头（多尺度预测）")
    print("  4. 52×52 高分辨率 → 小目标检测提升 ⭐⭐⭐")

