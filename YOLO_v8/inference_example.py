"""
YOLOv8 推理示例

展示如何使用 YOLOv8 进行目标检测，包括：
1. 加载模型
2. 图像预处理
3. 模型推理
4. 结果可视化
5. 特征提取（用于 VLM 幻觉检测研究）
"""

import torch
import torch.nn as nn
from yolov8_model import YOLOv8
import numpy as np


class YOLOv8Detector:
    """
    YOLOv8 检测器封装
    提供简洁的推理接口
    """
    def __init__(self, num_classes=80, conf_threshold=0.25, iou_threshold=0.45, device='cpu'):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 创建模型
        self.model = YOLOv8(num_classes=num_classes).to(device)
        self.model.eval()
        
        # COCO 类别名称（示例）
        self.class_names = self._get_coco_names()
    
    def preprocess(self, image):
        """
        图像预处理
        
        Args:
            image: numpy.ndarray (H, W, 3) BGR 格式，值域 [0, 255]
        
        Returns:
            tensor: (1, 3, 640, 640) 归一化后的 Tensor
        """
        # 1. Resize 到 640×640
        # 实际使用中应保持宽高比，这里简化处理
        from PIL import Image
        img = Image.fromarray(image[..., ::-1])  # BGR → RGB
        img = img.resize((640, 640))
        
        # 2. 转换为 Tensor 并归一化
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (H, W, 3) → (1, 3, H, W)
        
        return img.to(self.device)
    
    def detect(self, image):
        """
        目标检测
        
        Args:
            image: numpy.ndarray (H, W, 3) BGR 格式
        
        Returns:
            detections: Dict
                {
                    'boxes': (N, 4) [x1, y1, x2, y2]
                    'scores': (N,)
                    'classes': (N,)
                    'class_names': List[str]
                }
        """
        # 预处理
        img_tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            predictions = self.model.predict(
                img_tensor,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )
        
        # 获取第一个样本的结果
        pred = predictions[0]
        
        # 添加类别名称
        class_names = [self.class_names[int(cls)] for cls in pred['classes']]
        
        return {
            'boxes': pred['boxes'].cpu().numpy(),
            'scores': pred['scores'].cpu().numpy(),
            'classes': pred['classes'].cpu().numpy(),
            'class_names': class_names
        }
    
    def extract_features(self, image, layer='neck'):
        """
        提取中间层特征（用于 VLM 研究）
        
        Args:
            image: numpy.ndarray (H, W, 3)
            layer: str - 'backbone', 'neck', 'all'
        
        Returns:
            features: Dict
                {
                    'p3': (1, 256, 80, 80),   # 浅层特征
                    'p4': (1, 512, 40, 40),   # 中层特征
                    'p5': (1, 1024, 20, 20)   # 深层特征
                }
        """
        img_tensor = self.preprocess(image)
        
        with torch.no_grad():
            # Backbone
            p3, p4, p5 = self.model.backbone(img_tensor)
            
            if layer == 'backbone':
                return {'p3': p3, 'p4': p4, 'p5': p5}
            
            # Neck
            n3, n4, n5 = self.model.neck(p3, p4, p5)
            
            if layer == 'neck':
                return {'n3': n3, 'n4': n4, 'n5': n5}
            
            # All
            return {
                'backbone': {'p3': p3, 'p4': p4, 'p5': p5},
                'neck': {'n3': n3, 'n4': n4, 'n5': n5}
            }
    
    def _get_coco_names(self):
        """COCO 80 类别名称"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]


def visualize_detections(image, detections):
    """
    可视化检测结果（简化版）
    
    Args:
        image: numpy.ndarray (H, W, 3)
        detections: Dict - detect() 的输出
    
    Returns:
        None (打印检测结果)
    """
    print("\n" + "=" * 60)
    print(f"检测到 {len(detections['boxes'])} 个物体")
    print("=" * 60)
    
    for i, (box, score, cls_name) in enumerate(
        zip(detections['boxes'], detections['scores'], detections['class_names'])
    ):
        x1, y1, x2, y2 = box
        print(f"{i+1}. {cls_name}")
        print(f"   置信度: {score:.3f}")
        print(f"   位置: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"   尺寸: {x2-x1:.1f} × {y2-y1:.1f}")


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 推理示例")
    print("=" * 60)
    
    # 1. 创建检测器
    print("\n1. 初始化检测器...")
    detector = YOLOv8Detector(
        num_classes=80,
        conf_threshold=0.5,  # 提高阈值，减少误检
        iou_threshold=0.45,
        device='cpu'
    )
    print("   ✅ 检测器初始化完成")
    
    # 2. 创建测试图像（随机图像）
    print("\n2. 准备测试图像...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"   图像尺寸: {test_image.shape}")
    
    # 3. 目标检测
    print("\n3. 执行目标检测...")
    detections = detector.detect(test_image)
    
    # 可视化结果
    visualize_detections(test_image, detections)
    
    # 4. 特征提取（用于 VLM 研究）
    print("\n" + "=" * 60)
    print("4. 特征提取（用于 VLM 研究）")
    print("=" * 60)
    
    features = detector.extract_features(test_image, layer='neck')
    
    print("\n提取的特征：")
    for name, feat in features.items():
        print(f"   {name}: {feat.shape}")
    
    print("\n【应用场景】")
    print("这些特征可以用于：")
    print("  1. 与 CLIP 特征对齐（多模态融合）")
    print("  2. 作为 VLM 的视觉输入")
    print("  3. 用于幻觉检测的注意力分析")
    
    # 5. 与 VLM 结合的示例流程
    print("\n" + "=" * 60)
    print("5. 与 VLM 结合的研究流程示例")
    print("=" * 60)
    print("""
步骤 1: YOLO 检测物体
    输入：图像
    输出：边界框 + 类别 + 置信度
    
步骤 2: CLIP 特征对齐
    输入：图像 + YOLO 边界框 + VLM 生成的文本
    输出：图像-文本相似度分数
    
步骤 3: VLM 生成描述
    输入：图像 + （可选）检测到的物体列表
    输出：图像描述 + 注意力权重
    
步骤 4: 幻觉检测
    方法 A：对比检测
        - YOLO 检测到的物体 vs VLM 描述的物体
        - 如果 VLM 描述了不存在的物体 → 幻觉
    
    方法 B：注意力分析
        - 检查 VLM 的注意力是否集中在正确的区域
        - 如果注意力分散或指向错误位置 → 幻觉
    
    方法 C：语义一致性
        - 使用 CLIP 评估描述与图像的语义一致性
        - 低一致性分数 → 可能幻觉
    """)
    
    print("\n" + "=" * 60)
    print("✅ 推理示例完成！")
    print("=" * 60)
    
    print("\n【下一步】")
    print("1. 手敲核心代码，理解每个模块")
    print("2. 画出完整的 YOLOv8 架构图")
    print("3. 理解与 YOLOv2/v3 的区别")
    print("4. 思考如何与 CLIP、VLM 结合")

