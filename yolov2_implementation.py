"""
YOLOv2 核心概念实现
这个文件展示了 YOLOv2 的关键组件和概念
"""

import numpy as np
from typing import List, Tuple


# ================================
# 1. Anchor Boxes - K-means 聚类
# ================================

def iou_kmeans(box: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    计算用于 K-means 的 IoU 距离
    YOLOv2 使用 1 - IoU 作为距离度量
    
    Args:
        box: (width, height) - 单个框的尺寸
        centroids: (k, 2) - k 个聚类中心的尺寸
    
    Returns:
        distances: (k,) - 到每个聚类中心的距离
    """
    # 计算交集面积
    intersection = np.minimum(box[0], centroids[:, 0]) * np.minimum(box[1], centroids[:, 1])
    
    # 计算框的面积
    box_area = box[0] * box[1]
    centroid_area = centroids[:, 0] * centroids[:, 1]
    
    # 计算 IoU
    iou = intersection / (box_area + centroid_area - intersection)
    
    # 返回距离 (1 - IoU)
    return 1 - iou


def kmeans_anchors(boxes: np.ndarray, k: int = 5, max_iter: int = 300) -> np.ndarray:
    """
    使用 K-means 聚类生成 anchor boxes
    
    Args:
        boxes: (n, 2) - 所有训练框的宽高
        k: anchor boxes 的数量
        max_iter: 最大迭代次数
    
    Returns:
        anchors: (k, 2) - k 个 anchor boxes 的尺寸
    
    示例:
        >>> boxes = np.array([[100, 200], [150, 300], [50, 100]])
        >>> anchors = kmeans_anchors(boxes, k=3)
        >>> print(f"生成的 anchors: {anchors}")
    """
    n = boxes.shape[0]
    
    # 随机初始化聚类中心
    indices = np.random.choice(n, k, replace=False)
    centroids = boxes[indices]
    
    for iteration in range(max_iter):
        # 计算每个框到所有聚类中心的距离
        distances = np.array([iou_kmeans(box, centroids) for box in boxes])
        
        # 分配到最近的聚类中心
        assignments = np.argmin(distances, axis=1)
        
        # 更新聚类中心
        new_centroids = np.array([boxes[assignments == i].mean(axis=0) 
                                   for i in range(k)])
        
        # 检查收敛
        if np.allclose(centroids, new_centroids):
            print(f"K-means 在第 {iteration} 次迭代后收敛")
            break
            
        centroids = new_centroids
    
    # 按面积排序
    areas = centroids[:, 0] * centroids[:, 1]
    sorted_indices = np.argsort(areas)
    
    return centroids[sorted_indices]


# ================================
# 2. 边界框预测和解码
# ================================

def decode_yolov2_predictions(
    predictions: np.ndarray,
    anchors: np.ndarray,
    grid_size: int = 13,
    input_size: int = 416,
    num_classes: int = 80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解码 YOLOv2 的预测输出
    
    YOLOv2 预测格式 (每个 grid cell):
    - tx, ty: 中心点偏移 (相对于 grid cell)
    - tw, th: 宽高缩放因子 (相对于 anchor)
    - confidence: 目标置信度
    - class_probs: 类别概率
    
    Args:
        predictions: (grid_size, grid_size, num_anchors, 5+num_classes)
        anchors: (num_anchors, 2) - anchor boxes 尺寸
        grid_size: 网格大小 (默认 13)
        input_size: 输入图像尺寸 (默认 416)
        num_classes: 类别数量
    
    Returns:
        boxes: (n, 4) - [x, y, w, h] 在原图上的坐标
        scores: (n,) - 置信度分数
        class_ids: (n,) - 类别 ID
    """
    num_anchors = anchors.shape[0]
    
    # 创建网格坐标
    cx = np.arange(grid_size).reshape(1, grid_size, 1)
    cy = np.arange(grid_size).reshape(grid_size, 1, 1)
    
    # 提取预测值
    tx = predictions[..., 0]
    ty = predictions[..., 1]
    tw = predictions[..., 2]
    th = predictions[..., 3]
    confidence = predictions[..., 4]
    class_probs = predictions[..., 5:]
    
    # 解码边界框
    # bx = sigmoid(tx) + cx
    # by = sigmoid(ty) + cy
    bx = sigmoid(tx) + cx
    by = sigmoid(ty) + cy
    
    # bw = pw * exp(tw)
    # bh = ph * exp(th)
    pw = anchors[:, 0].reshape(1, 1, num_anchors)
    ph = anchors[:, 1].reshape(1, 1, num_anchors)
    bw = pw * np.exp(tw)
    bh = ph * np.exp(th)
    
    # 转换到输入图像尺寸
    scale = input_size / grid_size
    bx = bx * scale
    by = by * scale
    bw = bw * scale
    bh = bh * scale
    
    # 计算类别分数
    confidence = sigmoid(confidence)
    class_probs = sigmoid(class_probs)
    class_scores = confidence[..., np.newaxis] * class_probs
    
    # 获取最大类别
    class_ids = np.argmax(class_scores, axis=-1)
    scores = np.max(class_scores, axis=-1)
    
    # 展平并过滤
    boxes = np.stack([bx, by, bw, bh], axis=-1).reshape(-1, 4)
    scores = scores.reshape(-1)
    class_ids = class_ids.reshape(-1)
    
    return boxes, scores, class_ids


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))


# ================================
# 3. 非极大值抑制 (NMS)
# ================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的 IoU
    
    Args:
        box1, box2: [x, y, w, h] 格式
    
    Returns:
        iou: IoU 值
    """
    # 转换为 [x1, y1, x2, y2]
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # 计算并集
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5
) -> List[int]:
    """
    非极大值抑制
    
    Args:
        boxes: (n, 4) - 边界框坐标
        scores: (n,) - 置信度分数
        iou_threshold: IoU 阈值
        score_threshold: 分数阈值
    
    Returns:
        keep_indices: 保留的框的索引
    """
    # 过滤低分框
    mask = scores > score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    if len(boxes) == 0:
        return []
    
    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        # 选择分数最高的框
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # 计算与其他框的 IoU
        current_box = boxes[current]
        other_boxes = boxes[sorted_indices[1:]]
        
        ious = np.array([compute_iou(current_box, box) for box in other_boxes])
        
        # 保留 IoU 小于阈值的框
        sorted_indices = sorted_indices[1:][ious < iou_threshold]
    
    return keep


# ================================
# 4. Passthrough 层（细粒度特征）
# ================================

def passthrough_layer(x: np.ndarray) -> np.ndarray:
    """
    YOLOv2 的 Passthrough 层
    将 26×26×512 的特征图重组为 13×13×2048
    
    类似于像素重排（space-to-depth）
    
    Args:
        x: (batch, 26, 26, 512)
    
    Returns:
        output: (batch, 13, 13, 2048)
    
    示例:
        输入: 26×26 特征图
        [a b c d]    重组后    [a c]
        [e f g h]    ------>   [e g]
        变为 2×2 且通道数×4
    """
    batch, height, width, channels = x.shape
    
    # 确保尺寸可以被 2 整除
    assert height % 2 == 0 and width % 2 == 0
    
    # 重塑: (batch, 13, 2, 13, 2, 512)
    x = x.reshape(batch, height // 2, 2, width // 2, 2, channels)
    
    # 转置: (batch, 13, 13, 2, 2, 512)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    
    # 合并: (batch, 13, 13, 2048)
    x = x.reshape(batch, height // 2, width // 2, channels * 4)
    
    return x


# ================================
# 5. 使用示例
# ================================

def example_anchor_generation():
    """示例：生成 anchor boxes"""
    print("=" * 60)
    print("YOLOv2 Anchor 生成示例")
    print("=" * 60)
    
    # 模拟训练集中的边界框尺寸 (归一化到 grid cell)
    np.random.seed(42)
    boxes = np.random.rand(100, 2) * 10  # 100 个框，尺寸在 0-10 之间
    
    # 使用 K-means 生成 5 个 anchors
    anchors = kmeans_anchors(boxes, k=5)
    
    print(f"\n生成的 5 个 anchor boxes (宽, 高):")
    for i, (w, h) in enumerate(anchors):
        print(f"  Anchor {i+1}: {w:.2f} × {h:.2f} (面积: {w*h:.2f})")
    
    # 计算平均 IoU
    avg_iou = 0
    for box in boxes:
        ious = 1 - iou_kmeans(box, anchors)
        avg_iou += np.max(ious)
    avg_iou /= len(boxes)
    
    print(f"\n平均最大 IoU: {avg_iou:.4f}")
    print("提示: 更高的 IoU 意味着 anchors 更好地匹配训练数据\n")


def example_prediction_decoding():
    """示例：解码预测结果"""
    print("=" * 60)
    print("YOLOv2 预测解码示例")
    print("=" * 60)
    
    # 模拟预测输出
    grid_size = 13
    num_anchors = 5
    num_classes = 80
    
    # 随机生成预测 (实际应该来自网络输出)
    predictions = np.random.randn(grid_size, grid_size, num_anchors, 5 + num_classes)
    
    # 使用之前生成的 anchors
    anchors = np.array([
        [1.3221, 1.73145],
        [3.19275, 4.00944],
        [5.05587, 8.09892],
        [9.47112, 4.84053],
        [11.2364, 10.0071]
    ])
    
    # 解码
    boxes, scores, class_ids = decode_yolov2_predictions(
        predictions, anchors, grid_size=13, input_size=416, num_classes=80
    )
    
    # 应用 NMS
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.5)
    
    print(f"\n总预测框数: {len(boxes)}")
    print(f"NMS 后保留: {len(keep_indices)} 个框")
    
    if len(keep_indices) > 0:
        print(f"\n前 3 个检测结果:")
        for i, idx in enumerate(keep_indices[:3]):
            x, y, w, h = boxes[idx]
            score = scores[idx]
            class_id = class_ids[idx]
            print(f"  检测 {i+1}: 类别={class_id}, 置信度={score:.4f}, "
                  f"位置=({x:.1f}, {y:.1f}), 尺寸=({w:.1f}×{h:.1f})")


def example_passthrough():
    """示例：Passthrough 层"""
    print("\n" + "=" * 60)
    print("YOLOv2 Passthrough 层示例")
    print("=" * 60)
    
    # 创建 26×26×512 的特征图
    x = np.random.randn(1, 26, 26, 512)
    print(f"\n输入特征图形状: {x.shape}")
    
    # 应用 Passthrough
    output = passthrough_layer(x)
    print(f"输出特征图形状: {output.shape}")
    
    print("\n解释:")
    print("- 空间分辨率减半: 26×26 → 13×13")
    print("- 通道数翻4倍: 512 → 2048")
    print("- 保留了高分辨率特征的细节信息")
    print("- 与主干网络的 13×13 特征图拼接")


if __name__ == "__main__":
    # 运行所有示例
    example_anchor_generation()
    example_prediction_decoding()
    example_passthrough()
    
    print("\n" + "=" * 60)
    print("YOLOv2 核心概念演示完成！")
    print("=" * 60)
    print("\n关键要点:")
    print("1. K-means 聚类生成适合数据集的 anchors")
    print("2. 使用 sigmoid 约束中心点偏移在 [0,1]")
    print("3. 使用 exp 预测宽高的缩放因子")
    print("4. Passthrough 层融合细粒度特征")
    print("5. NMS 去除重复检测框")

