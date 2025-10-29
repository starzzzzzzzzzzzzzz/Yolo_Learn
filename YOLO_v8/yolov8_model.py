"""
YOLOv8 完整模型

架构：
Input (640×640)
    ↓
Backbone (CSPDarknet + C2f)
    ├─→ P3: 80×80 (浅层特征 - 小物体)
    ├─→ P4: 40×40 (中层特征 - 中物体)
    └─→ P5: 20×20 (深层特征 - 大物体)
    ↓
Neck (PAN-FPN 融合)
    ├─→ N3: 80×80 融合特征
    ├─→ N4: 40×40 融合特征
    └─→ N5: 20×20 融合特征
    ↓
Head (Decoupled Head - Anchor-free)
    ├─→ 分类分支 (cls_conv)
    └─→ 定位分支 (reg_conv + DFL)
    ↓
Output: [x1, y1, x2, y2, scores...]
"""

import torch
import torch.nn as nn
from basic_modules import Conv, C2f, SPPF
from detection_head import DecoupledHead, AnchorFreeDecoder


class YOLOv8Backbone(nn.Module):
    """
    YOLOv8 骨干网络（CSPDarknet）
    
    结构：
    - Stage 1: 640 → 320 (stride=2)
    - Stage 2: 320 → 160 (stride=4)
    - Stage 3: 160 → 80 (stride=8) → P3 输出
    - Stage 4: 80 → 40 (stride=16) → P4 输出
    - Stage 5: 40 → 20 (stride=32) → P5 输出 + SPPF
    """
    def __init__(self):
        super(YOLOv8Backbone, self).__init__()
        
        # Stage 0: 初始卷积
        # 640×640×3 → 320×320×64
        self.stem = Conv(3, 64, 3, 2)
        
        # Stage 1: 320×320×64 → 160×160×128
        self.stage1 = nn.Sequential(
            Conv(64, 128, 3, 2),
            C2f(128, 128, num_bottlenecks=3)
        )
        
        # Stage 2: 160×160×128 → 80×80×256
        self.stage2 = nn.Sequential(
            Conv(128, 256, 3, 2),
            C2f(256, 256, num_bottlenecks=6)
        )
        
        # Stage 3: 80×80×256 → 40×40×512
        self.stage3 = nn.Sequential(
            Conv(256, 512, 3, 2),
            C2f(512, 512, num_bottlenecks=6)
        )
        
        # Stage 4: 40×40×512 → 20×20×1024
        self.stage4 = nn.Sequential(
            Conv(512, 1024, 3, 2),
            C2f(1024, 1024, num_bottlenecks=3),
            SPPF(1024, 1024)  # 增大感受野
        )
    
    def forward(self, x):
        """
        输入：(B, 3, 640, 640)
        输出：
            p3: (B, 256, 80, 80)  - 浅层特征
            p4: (B, 512, 40, 40)  - 中层特征
            p5: (B, 1024, 20, 20) - 深层特征
        """
        x = self.stem(x)      # 640 → 320
        x = self.stage1(x)    # 320 → 160
        p3 = self.stage2(x)   # 160 → 80 (浅层输出)
        p4 = self.stage3(p3)  # 80 → 40 (中层输出)
        p5 = self.stage4(p4)  # 40 → 20 (深层输出)
        
        return p3, p4, p5


class YOLOv8Neck(nn.Module):
    """
    YOLOv8 颈部网络（PAN-FPN）
    
    结构：
    1. Top-down 路径（FPN）：深层 → 浅层，传递语义信息
       P5 (20×20) → Upsample → 与 P4 Concat → N4 (40×40)
       N4 (40×40) → Upsample → 与 P3 Concat → N3 (80×80)
    
    2. Bottom-up 路径（PAN）：浅层 → 深层，传递定位信息
       N3 (80×80) → Downsample → 与 N4 Concat → N4 (40×40)
       N4 (40×40) → Downsample → 与 P5 Concat → N5 (20×20)
    
    对比 YOLOv3：
    - YOLOv3: 只有 Top-down（FPN）
    - YOLOv8: Top-down + Bottom-up（PAN-FPN），信息流动更充分
    """
    def __init__(self):
        super(YOLOv8Neck, self).__init__()
        
        # ========== Top-down 路径（FPN）==========
        # P5 → N4
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_up1 = C2f(1024 + 512, 512, num_bottlenecks=3)  # Concat 后融合
        
        # N4 → N3
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_up2 = C2f(512 + 256, 256, num_bottlenecks=3)
        
        # ========== Bottom-up 路径（PAN）==========
        # N3 → N4
        self.downsample1 = Conv(256, 256, 3, 2)
        self.c2f_down1 = C2f(256 + 512, 512, num_bottlenecks=3)
        
        # N4 → N5
        self.downsample2 = Conv(512, 512, 3, 2)
        self.c2f_down2 = C2f(512 + 1024, 1024, num_bottlenecks=3)
    
    def forward(self, p3, p4, p5):
        """
        输入：
            p3: (B, 256, 80, 80)   Backbone 浅层输出
            p4: (B, 512, 40, 40)   Backbone 中层输出
            p5: (B, 1024, 20, 20)  Backbone 深层输出
        
        输出：
            n3: (B, 256, 80, 80)   融合后的浅层特征
            n4: (B, 512, 40, 40)   融合后的中层特征
            n5: (B, 1024, 20, 20)  融合后的深层特征
        """
        # ========== Top-down：深层 → 浅层 ==========
        # P5 (20×20) → 40×40
        fpn_p5 = self.upsample1(p5)
        # Concat P5 + P4
        fpn_n4 = torch.cat([fpn_p5, p4], dim=1)
        fpn_n4 = self.c2f_up1(fpn_n4)
        
        # N4 (40×40) → 80×80
        fpn_p4 = self.upsample2(fpn_n4)
        # Concat N4 + P3
        n3 = torch.cat([fpn_p4, p3], dim=1)
        n3 = self.c2f_up2(n3)  # 最终 N3 输出
        
        # ========== Bottom-up：浅层 → 深层 ==========
        # N3 (80×80) → 40×40
        pan_n3 = self.downsample1(n3)
        # Concat N3 + N4
        pan_n4 = torch.cat([pan_n3, fpn_n4], dim=1)
        n4 = self.c2f_down1(pan_n4)  # 最终 N4 输出
        
        # N4 (40×40) → 20×20
        pan_n4 = self.downsample2(n4)
        # Concat N4 + P5
        pan_n5 = torch.cat([pan_n4, p5], dim=1)
        n5 = self.c2f_down2(pan_n5)  # 最终 N5 输出
        
        return n3, n4, n5


class YOLOv8(nn.Module):
    """
    YOLOv8 完整模型
    """
    def __init__(self, num_classes=80, num_bins=16):
        super(YOLOv8, self).__init__()
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = YOLOv8Backbone()
        
        # Neck
        self.neck = YOLOv8Neck()
        
        # Head（3 个尺度的检测头）
        self.head_large = DecoupledHead(num_classes, in_channels=256, num_bins=num_bins)  # 小物体 (80×80)
        self.head_medium = DecoupledHead(num_classes, in_channels=512, num_bins=num_bins) # 中物体 (40×40)
        self.head_small = DecoupledHead(num_classes, in_channels=1024, num_bins=num_bins) # 大物体 (20×20)
        
        # Decoder
        self.decoder = AnchorFreeDecoder(num_bins=num_bins)
    
    def forward(self, x):
        """
        输入：(B, 3, 640, 640)
        输出：
            boxes: List[(B, H*W, 4)] - 3 个尺度的边界框
            scores: List[(B, H*W, num_classes)] - 3 个尺度的分数
        """
        # Backbone: 提取多尺度特征
        p3, p4, p5 = self.backbone(x)
        
        # Neck: 融合特征
        n3, n4, n5 = self.neck(p3, p4, p5)
        
        # Head: 预测
        cls_large, reg_large = self.head_large(n3)    # 80×80 (stride=8)
        cls_medium, reg_medium = self.head_medium(n4)  # 40×40 (stride=16)
        cls_small, reg_small = self.head_small(n5)     # 20×20 (stride=32)
        
        # Decode: 解码成边界框
        boxes_large, scores_large = self.decoder(cls_large, reg_large, stride=8)
        boxes_medium, scores_medium = self.decoder(cls_medium, reg_medium, stride=16)
        boxes_small, scores_small = self.decoder(cls_small, reg_small, stride=32)
        
        return {
            'boxes': [boxes_large, boxes_medium, boxes_small],
            'scores': [scores_large, scores_medium, scores_small]
        }
    
    def predict(self, x, conf_threshold=0.25, iou_threshold=0.45):
        """
        推理接口（包含 NMS）
        
        Args:
            x: (B, 3, 640, 640) 输入图像
            conf_threshold: 置信度阈值
            iou_threshold: NMS 的 IoU 阈值
        
        Returns:
            predictions: List[Dict]，每个样本的检测结果
                {
                    'boxes': (N, 4) [x1, y1, x2, y2]
                    'scores': (N,)
                    'classes': (N,)
                }
        """
        outputs = self.forward(x)
        
        # 合并三个尺度的预测
        boxes = torch.cat(outputs['boxes'], dim=1)    # (B, total_anchors, 4)
        scores = torch.cat(outputs['scores'], dim=1)  # (B, total_anchors, num_classes)
        
        batch_size = boxes.shape[0]
        predictions = []
        
        for i in range(batch_size):
            # 获取最大类别分数和类别 ID
            max_scores, class_ids = scores[i].max(dim=-1)
            
            # 置信度过滤
            keep = max_scores > conf_threshold
            filtered_boxes = boxes[i][keep]
            filtered_scores = max_scores[keep]
            filtered_classes = class_ids[keep]
            
            # NMS（简化版）
            if len(filtered_boxes) > 0:
                keep_nms = self._nms(filtered_boxes, filtered_scores, iou_threshold)
                predictions.append({
                    'boxes': filtered_boxes[keep_nms],
                    'scores': filtered_scores[keep_nms],
                    'classes': filtered_classes[keep_nms]
                })
            else:
                predictions.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'classes': torch.empty(0, dtype=torch.long)
                })
        
        return predictions
    
    def _nms(self, boxes, scores, iou_threshold):
        """简化的 NMS 实现"""
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 完整模型测试")
    print("=" * 60)
    
    # 创建模型
    model = YOLOv8(num_classes=80)
    model.eval()
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640)
    
    print(f"\n输入: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
    
    print("\n【多尺度输出】")
    for i, (boxes, scores) in enumerate(zip(outputs['boxes'], outputs['scores'])):
        scale_name = ['Large (小物体)', 'Medium (中物体)', 'Small (大物体)'][i]
        stride = [8, 16, 32][i]
        print(f"{i+1}. {scale_name} (stride={stride})")
        print(f"   Boxes:  {boxes.shape}")
        print(f"   Scores: {scores.shape}")
    
    # 推理测试（含 NMS）
    print("\n【推理测试（含 NMS）】")
    predictions = model.predict(x, conf_threshold=0.25, iou_threshold=0.45)
    for i, pred in enumerate(predictions):
        print(f"样本 {i+1}:")
        print(f"   检测到 {len(pred['boxes'])} 个物体")
        if len(pred['boxes']) > 0:
            print(f"   Boxes:   {pred['boxes'].shape}")
            print(f"   Scores:  {pred['scores'].shape}")
            print(f"   Classes: {pred['classes'].shape}")
    
    print("\n" + "=" * 60)
    print("✅ YOLOv8 完整模型测试通过！")
    print("=" * 60)
    
    # 模型统计
    print("\n【模型统计】")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

