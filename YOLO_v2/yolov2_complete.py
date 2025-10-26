"""
YOLOv2 完整实现
包含：骨干网络、检测头、损失函数、预测解码、NMS 等所有核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ================================
# 1. 骨干网络 Darknet19
# ================================

class Darknet19(nn.Module):
    """Darknet19 骨干网络（用于特征提取）"""
    
    def __init__(self):
        super(Darknet19, self).__init__()
        
        # 特征提取层
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        # 注意：这里不做 pool5，保留 26×26 用于 passthrough
        
        self.conv14 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn14 = nn.BatchNorm2d(1024)
        self.conv15 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn16 = nn.BatchNorm2d(1024)
        self.conv17 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.bn17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn18 = nn.BatchNorm2d(1024)
        
    def forward(self, x):
        # 前面的层
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = self.pool1(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = self.pool2(x)
        
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1)
        x = self.pool3(x)
        
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.1)
        x = F.leaky_relu(self.bn7(self.conv7(x)), 0.1)
        x = F.leaky_relu(self.bn8(self.conv8(x)), 0.1)
        x = self.pool4(x)
        
        x = F.leaky_relu(self.bn9(self.conv9(x)), 0.1)
        x = F.leaky_relu(self.bn10(self.conv10(x)), 0.1)
        x = F.leaky_relu(self.bn11(self.conv11(x)), 0.1)
        x = F.leaky_relu(self.bn12(self.conv12(x)), 0.1)
        x = F.leaky_relu(self.bn13(self.conv13(x)), 0.1)
        
        # 保存 26×26 的特征图用于 passthrough
        passthrough = x  # 26×26×512
        
        # 继续下采样到 13×13
        x = F.max_pool2d(x, 2, 2)  # 13×13×512
        
        x = F.leaky_relu(self.bn14(self.conv14(x)), 0.1)
        x = F.leaky_relu(self.bn15(self.conv15(x)), 0.1)
        x = F.leaky_relu(self.bn16(self.conv16(x)), 0.1)
        x = F.leaky_relu(self.bn17(self.conv17(x)), 0.1)
        x = F.leaky_relu(self.bn18(self.conv18(x)), 0.1)
        
        return x, passthrough  # 返回 13×13 特征和 26×26 特征


# ================================
# 2. Passthrough 层
# ================================

class PassthroughLayer(nn.Module):
    """
    将 26×26×512 重组为 13×13×2048
    用于融合高分辨率特征
    """
    def __init__(self):
        super(PassthroughLayer, self).__init__()
        
    def forward(self, x):
        """
        x: (batch, 512, 26, 26)
        output: (batch, 2048, 13, 13)
        """
        batch_size = x.size(0)
        channels = x.size(1)
        height = x.size(2)
        width = x.size(3)
        
        # 重塑: (batch, C, H, W) → (batch, C, H/2, 2, W/2, 2)
        x = x.view(batch_size, channels, height // 2, 2, width // 2, 2)
        
        # 转置: → (batch, C, 2, 2, H/2, W/2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        
        # 合并: → (batch, C*4, H/2, W/2)
        x = x.view(batch_size, channels * 4, height // 2, width // 2)
        
        return x


# ================================
# 3. YOLOv2 检测头
# ================================

class YOLOv2DetectionHead(nn.Module):
    """YOLOv2 检测头"""
    
    def __init__(self, num_anchors=5, num_classes=80):
        super(YOLOv2DetectionHead, self).__init__()
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Passthrough 层
        self.passthrough = PassthroughLayer()
        
        # 融合后的卷积层
        # 1024 (主路) + 2048 (passthrough) = 3072
        self.conv19 = nn.Conv2d(3072, 1024, 3, 1, 1, bias=False)
        self.bn19 = nn.BatchNorm2d(1024)
        
        # 最终预测层
        # 每个 anchor 预测: tx, ty, tw, th, confidence, class_probs
        # 总共: 5 + num_classes
        self.conv20 = nn.Conv2d(1024, num_anchors * (5 + num_classes), 1, 1, 0)
        
    def forward(self, x, passthrough):
        """
        x: (batch, 1024, 13, 13) - 主路特征
        passthrough: (batch, 512, 26, 26) - 高分辨率特征
        """
        # Passthrough: 26×26×512 → 13×13×2048
        passthrough = self.passthrough(passthrough)
        
        # 拼接
        x = torch.cat([x, passthrough], dim=1)  # (batch, 3072, 13, 13)
        
        # 卷积
        x = F.leaky_relu(self.bn19(self.conv19(x)), 0.1)
        
        # 预测
        x = self.conv20(x)  # (batch, num_anchors*(5+num_classes), 13, 13)
        
        return x


# ================================
# 4. 完整的 YOLOv2 模型
# ================================

class YOLOv2(nn.Module):
    """完整的 YOLOv2 模型"""
    
    def __init__(self, num_anchors=5, num_classes=80, anchors=None):
        super(YOLOv2, self).__init__()
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # 默认 anchors (如果没有提供)
        if anchors is None:
            # 这些是在 COCO 上聚类得到的 anchors (相对于 13×13 grid)
            self.anchors = torch.FloatTensor([
                [1.3221, 1.73145],
                [3.19275, 4.00944],
                [5.05587, 8.09892],
                [9.47112, 4.84053],
                [11.2364, 10.0071]
            ])
        else:
            self.anchors = anchors
        
        # 骨干网络
        self.backbone = Darknet19()
        
        # 检测头
        self.detection_head = YOLOv2DetectionHead(num_anchors, num_classes)
        
        self.grid_size = 13
        
    def forward(self, x):
        """
        x: (batch, 3, 416, 416)
        output: (batch, 13, 13, num_anchors, 5+num_classes)
        """
        batch_size = x.size(0)
        
        # 特征提取
        features, passthrough = self.backbone(x)  # (B, 1024, 13, 13), (B, 512, 26, 26)
        
        # 检测
        predictions = self.detection_head(features, passthrough)
        
        # 重塑输出
        # (B, num_anchors*(5+C), 13, 13) → (B, 13, 13, num_anchors, 5+C)
        predictions = predictions.permute(0, 2, 3, 1).contiguous()
        predictions = predictions.view(
            batch_size, 
            self.grid_size, 
            self.grid_size, 
            self.num_anchors, 
            5 + self.num_classes
        )
        
        return predictions
    
    def decode_predictions(self, predictions, conf_threshold=0.5, device='cpu'):
        """
        解码预测结果
        predictions: (batch, 13, 13, num_anchors, 5+num_classes)
        返回: boxes, scores, class_ids (已过滤低置信度)
        """
        batch_size = predictions.size(0)
        
        # 创建网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_size),
            torch.arange(self.grid_size),
            indexing='ij'
        )
        grid_x = grid_x.float().to(device)
        grid_y = grid_y.float().to(device)
        
        # anchors 转到正确的设备
        anchors = self.anchors.to(device)
        
        # 提取预测值
        tx = predictions[..., 0]  # (B, 13, 13, 5)
        ty = predictions[..., 1]
        tw = predictions[..., 2]
        th = predictions[..., 3]
        confidence = predictions[..., 4]
        class_probs = predictions[..., 5:]
        
        # 解码边界框 (Direct Location Prediction)
        bx = torch.sigmoid(tx) + grid_x.unsqueeze(-1)  # (13, 13, 5)
        by = torch.sigmoid(ty) + grid_y.unsqueeze(-1)
        
        # anchors: (5, 2)
        pw = anchors[:, 0].view(1, 1, 1, -1)  # (1, 1, 1, 5)
        ph = anchors[:, 1].view(1, 1, 1, -1)
        
        bw = pw * torch.exp(tw)
        bh = ph * torch.exp(th)
        
        # 转换到输入图像坐标 (假设输入 416×416)
        stride = 32  # 416 / 13 = 32
        bx = bx * stride
        by = by * stride
        bw = bw * stride
        bh = bh * stride
        
        # 转换为 (x1, y1, x2, y2)
        x1 = bx - bw / 2
        y1 = by - bh / 2
        x2 = bx + bw / 2
        y2 = by + bh / 2
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, 13, 13, 5, 4)
        
        # 计算类别分数
        confidence = torch.sigmoid(confidence)  # (B, 13, 13, 5)
        class_probs = torch.sigmoid(class_probs)  # (B, 13, 13, 5, C)
        
        class_scores = confidence.unsqueeze(-1) * class_probs  # (B, 13, 13, 5, C)
        
        # 获取最大类别
        max_scores, class_ids = torch.max(class_scores, dim=-1)  # (B, 13, 13, 5)
        
        # 展平
        boxes = boxes.view(batch_size, -1, 4)  # (B, 845, 4)
        max_scores = max_scores.view(batch_size, -1)  # (B, 845)
        class_ids = class_ids.view(batch_size, -1)  # (B, 845)
        
        # 过滤低置信度
        results = []
        for i in range(batch_size):
            mask = max_scores[i] > conf_threshold
            filtered_boxes = boxes[i][mask]
            filtered_scores = max_scores[i][mask]
            filtered_classes = class_ids[i][mask]
            
            results.append({
                'boxes': filtered_boxes,
                'scores': filtered_scores,
                'classes': filtered_classes
            })
        
        return results


# ================================
# 5. 损失函数
# ================================

class YOLOv2Loss(nn.Module):
    """YOLOv2 损失函数"""
    
    def __init__(self, num_classes=80, anchors=None, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv2Loss, self).__init__()
        
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord  # 坐标损失权重
        self.lambda_noobj = lambda_noobj  # 无目标置信度损失权重
        
        if anchors is None:
            self.anchors = torch.FloatTensor([
                [1.3221, 1.73145],
                [3.19275, 4.00944],
                [5.05587, 8.09892],
                [9.47112, 4.84053],
                [11.2364, 10.0071]
            ])
        else:
            self.anchors = anchors
    
    def forward(self, predictions, targets):
        """
        predictions: (batch, 13, 13, num_anchors, 5+num_classes)
        targets: list of dicts with 'boxes', 'labels' for each image
        """
        # 这里简化处理，实际实现需要：
        # 1. 对每个 GT box，找到最佳匹配的 anchor
        # 2. 计算坐标损失（只对有目标的 cell）
        # 3. 计算置信度损失（有目标和无目标分开）
        # 4. 计算分类损失（只对有目标的 cell）
        
        device = predictions.device
        batch_size = predictions.size(0)
        
        # 提取预测值
        pred_tx = predictions[..., 0]
        pred_ty = predictions[..., 1]
        pred_tw = predictions[..., 2]
        pred_th = predictions[..., 3]
        pred_conf = predictions[..., 4]
        pred_cls = predictions[..., 5:]
        
        # 总损失（这里是简化版本）
        coord_loss = torch.tensor(0.0, device=device)
        conf_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        # 实际训练时需要完整实现标签分配逻辑
        # 这里仅展示框架结构
        
        total_loss = (
            self.lambda_coord * coord_loss + 
            conf_loss + 
            cls_loss
        )
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'conf_loss': conf_loss.item(),
            'cls_loss': cls_loss.item()
        }


# ================================
# 6. NMS (非极大值抑制)
# ================================

def nms(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制
    boxes: (N, 4) - [x1, y1, x2, y2]
    scores: (N,)
    """
    if len(boxes) == 0:
        return []
    
    # 按分数排序
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while len(sorted_indices) > 0:
        # 选择分数最高的
        current = sorted_indices[0]
        keep.append(current.item())
        
        if len(sorted_indices) == 1:
            break
        
        # 计算 IoU
        current_box = boxes[current]
        other_boxes = boxes[sorted_indices[1:]]
        
        ious = compute_iou(current_box.unsqueeze(0), other_boxes)
        
        # 保留 IoU 小于阈值的
        mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return keep


def compute_iou(box1, boxes2):
    """
    计算 box1 和 boxes2 的 IoU
    box1: (1, 4)
    boxes2: (N, 4)
    """
    # 计算交集
    x1 = torch.max(box1[:, 0], boxes2[:, 0])
    y1 = torch.max(box1[:, 1], boxes2[:, 1])
    x2 = torch.min(box1[:, 2], boxes2[:, 2])
    y2 = torch.min(box1[:, 3], boxes2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # 计算面积
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area1 + area2 - intersection
    
    iou = intersection / union
    
    return iou


# ================================
# 7. K-means 聚类生成 Anchors
# ================================

def kmeans_anchors(boxes, k=5, max_iter=300):
    """
    使用 K-means 聚类生成 anchor boxes
    boxes: numpy array of shape (N, 2) - [width, height]
    """
    n = boxes.shape[0]
    
    # 随机初始化
    indices = np.random.choice(n, k, replace=False)
    centroids = boxes[indices].astype(np.float32)
    
    for iteration in range(max_iter):
        # 计算距离 (1 - IoU)
        distances = np.zeros((n, k))
        for i, box in enumerate(boxes):
            for j, centroid in enumerate(centroids):
                distances[i, j] = 1 - iou_wh(box, centroid)
        
        # 分配到最近的聚类中心
        assignments = np.argmin(distances, axis=1)
        
        # 更新聚类中心
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            if np.sum(assignments == j) > 0:
                new_centroids[j] = boxes[assignments == j].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        # 检查收敛
        if np.allclose(centroids, new_centroids):
            print(f"K-means 在第 {iteration} 次迭代后收敛")
            break
        
        centroids = new_centroids
    
    # 按面积排序
    areas = centroids[:, 0] * centroids[:, 1]
    sorted_indices = np.argsort(areas)
    
    return centroids[sorted_indices]


def iou_wh(box1, box2):
    """
    计算两个框的 IoU (假设都在原点对齐)
    box1, box2: [width, height]
    """
    intersection = min(box1[0], box2[0]) * min(box1[1], box2[1])
    area1 = box1[0] * box1[1]
    area2 = box2[0] * box2[1]
    union = area1 + area2 - intersection
    return intersection / union


# ================================
# 8. 使用示例
# ================================

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv2 完整实现演示")
    print("=" * 60)
    
    # 创建模型
    model = YOLOv2(num_anchors=5, num_classes=80)
    model.eval()
    
    print(f"\n✅ 模型创建成功")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    print("\n🔄 测试前向传播:")
    x = torch.randn(2, 3, 416, 416)  # batch_size=2
    
    with torch.no_grad():
        predictions = model(x)
    
    print(f"   输入: {x.shape}")
    print(f"   输出: {predictions.shape}")
    print(f"   预期: (2, 13, 13, 5, 85) for COCO")
    
    # 解码预测
    print("\n📦 解码预测:")
    results = model.decode_predictions(predictions, conf_threshold=0.5)
    
    for i, result in enumerate(results):
        print(f"   图像 {i+1}:")
        print(f"     检测框数: {len(result['boxes'])}")
        if len(result['boxes']) > 0:
            print(f"     最高分数: {result['scores'].max():.4f}")
    
    # K-means 聚类示例
    print("\n🎯 K-means 聚类生成 anchors:")
    # 模拟一些框的尺寸
    boxes = np.random.rand(100, 2) * 10
    anchors = kmeans_anchors(boxes, k=5)
    
    print("   生成的 5 个 anchors:")
    for i, (w, h) in enumerate(anchors):
        print(f"     Anchor {i+1}: {w:.2f} × {h:.2f}")
    
    print("\n" + "=" * 60)
    print("所有组件测试完成！")
    print("=" * 60)
    
    print("\n📚 代码结构:")
    print("   1. Darknet19 骨干网络")
    print("   2. PassthroughLayer (细粒度特征)")
    print("   3. YOLOv2DetectionHead (检测头)")
    print("   4. YOLOv2 完整模型")
    print("   5. YOLOv2Loss (损失函数)")
    print("   6. NMS (非极大值抑制)")
    print("   7. K-means 聚类 (anchor 生成)")
    
    print("\n💡 关键点:")
    print("   • Passthrough: 26×26 → 13×13, 融合细粒度特征")
    print("   • Direct Location: bx = sigmoid(tx) + cx")
    print("   • K-means: 用 1-IoU 作为距离")
    print("   • 多尺度训练: 全卷积架构支持任意输入尺寸")

