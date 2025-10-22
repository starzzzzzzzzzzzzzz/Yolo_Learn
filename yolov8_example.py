"""
YOLOv8 使用示例和核心概念
使用 Ultralytics YOLOv8 库
"""

import numpy as np
from typing import List, Dict, Tuple


# ================================
# 1. YOLOv8 安装和基础使用
# ================================

def installation_guide():
    """
    YOLOv8 安装指南
    """
    print("=" * 60)
    print("YOLOv8 安装指南")
    print("=" * 60)
    print("""
1. 安装 Ultralytics 包:
   pip install ultralytics

2. 验证安装:
   python -c "from ultralytics import YOLO; print('YOLOv8 安装成功！')"

3. 依赖包:
   - torch >= 1.8.0
   - torchvision
   - opencv-python
   - pillow
   - pyyaml
   - tqdm

4. 可选（加速推理）:
   pip install onnx onnxruntime
   pip install openvino-dev  # Intel 设备
    """)


# ================================
# 2. YOLOv8 基础使用示例
# ================================

def basic_usage_example():
    """
    YOLOv8 基础使用示例（需要安装 ultralytics）
    
    注意：这个函数展示了使用方法，但需要实际安装库才能运行
    """
    print("\n" + "=" * 60)
    print("YOLOv8 基础使用示例")
    print("=" * 60)
    
    code_example = '''
# ===== 示例 1: 图像检测 =====
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # n, s, m, l, x 版本

# 对图像进行推理
results = model('path/to/image.jpg')

# 处理结果
for result in results:
    boxes = result.boxes  # 边界框
    for box in boxes:
        # 获取坐标 (xyxy 格式)
        x1, y1, x2, y2 = box.xyxy[0]
        
        # 获取置信度和类别
        conf = box.conf[0]
        cls = box.cls[0]
        
        print(f"检测到: 类别={int(cls)}, 置信度={conf:.2f}")

# ===== 示例 2: 视频检测 =====
results = model('path/to/video.mp4', stream=True)

for result in results:
    # 实时处理每一帧
    frame_with_boxes = result.plot()  # 绘制边界框
    # cv2.imshow('YOLOv8', frame_with_boxes)

# ===== 示例 3: 批量图像处理 =====
results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# ===== 示例 4: 使用摄像头 =====
results = model(source=0, show=True)  # 0 表示默认摄像头

# ===== 示例 5: 保存结果 =====
results = model('image.jpg', save=True, save_txt=True)
# 结果保存在 runs/detect/predict/ 目录
    '''
    
    print(code_example)


# ================================
# 3. YOLOv8 训练自定义数据集
# ================================

def training_example():
    """
    YOLOv8 自定义训练示例
    """
    print("\n" + "=" * 60)
    print("YOLOv8 训练自定义数据集")
    print("=" * 60)
    
    code_example = '''
from ultralytics import YOLO

# ===== 1. 准备数据集 =====
# 数据集结构:
# dataset/
# ├── images/
# │   ├── train/
# │   │   ├── img1.jpg
# │   │   └── img2.jpg
# │   └── val/
# │       ├── img3.jpg
# │       └── img4.jpg
# └── labels/
#     ├── train/
#     │   ├── img1.txt
#     │   └── img2.txt
#     └── val/
#         ├── img3.txt
#         └── img4.txt

# ===== 2. 创建数据配置文件 (data.yaml) =====
"""
path: /path/to/dataset
train: images/train
val: images/val

nc: 80  # 类别数量
names: ['person', 'bicycle', 'car', ...]  # 类别名称
"""

# ===== 3. 标注格式 (YOLO 格式) =====
# 每个 txt 文件对应一张图像，每行一个目标:
# <class_id> <x_center> <y_center> <width> <height>
# 所有值都归一化到 [0, 1]
# 
# 示例: 0 0.5 0.5 0.3 0.4
#      类别0，中心在(0.5, 0.5)，宽0.3，高0.4

# ===== 4. 开始训练 =====
model = YOLO('yolov8n.pt')  # 使用预训练模型

results = model.train(
    data='data.yaml',           # 数据配置文件
    epochs=100,                 # 训练轮数
    imgsz=640,                  # 输入图像尺寸
    batch=16,                   # 批次大小
    name='my_model',            # 实验名称
    device=0,                   # GPU 设备 (0, 1, 2...) 或 'cpu'
    workers=8,                  # 数据加载线程数
    
    # 超参数
    lr0=0.01,                   # 初始学习率
    lrf=0.01,                   # 最终学习率 (lr0 * lrf)
    momentum=0.937,             # SGD momentum
    weight_decay=0.0005,        # 权重衰减
    
    # 数据增强
    hsv_h=0.015,                # HSV-Hue 增强
    hsv_s=0.7,                  # HSV-Saturation 增强
    hsv_v=0.4,                  # HSV-Value 增强
    degrees=0.0,                # 旋转角度
    translate=0.1,              # 平移
    scale=0.5,                  # 缩放
    shear=0.0,                  # 剪切
    perspective=0.0,            # 透视变换
    flipud=0.0,                 # 上下翻转概率
    fliplr=0.5,                 # 左右翻转概率
    mosaic=1.0,                 # Mosaic 增强概率
    mixup=0.0,                  # MixUp 增强概率
    
    # 其他设置
    patience=50,                # 早停耐心值
    save=True,                  # 保存检查点
    save_period=-1,             # 保存间隔 (-1 表示只保存最后)
    cache=False,                # 缓存图像到内存
    project='runs/train',       # 项目目录
    exist_ok=False,             # 覆盖已存在的实验
    pretrained=True,            # 使用预训练权重
    optimizer='SGD',            # 优化器 (SGD, Adam, AdamW)
    verbose=True,               # 详细输出
    seed=0,                     # 随机种子
    deterministic=True,         # 确定性模式
    single_cls=False,           # 单类别训练
    rect=False,                 # 矩形训练
    cos_lr=False,               # 余弦学习率
    close_mosaic=10,            # 最后 N 轮关闭 mosaic
    resume=False,               # 恢复训练
    amp=True,                   # 自动混合精度
)

# ===== 5. 验证模型 =====
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP75: {metrics.box.map75}")

# ===== 6. 导出模型 =====
# ONNX 格式
model.export(format='onnx')

# TensorRT (需要 NVIDIA GPU)
model.export(format='engine')

# CoreML (iOS/macOS)
model.export(format='coreml')

# TFLite (移动端)
model.export(format='tflite')
    '''
    
    print(code_example)


# ================================
# 4. YOLOv8 核心概念：Anchor-Free 检测
# ================================

def anchor_free_concept():
    """
    解释 YOLOv8 的 Anchor-Free 检测机制
    """
    print("\n" + "=" * 60)
    print("YOLOv8 Anchor-Free 检测原理")
    print("=" * 60)
    
    explanation = """
## Anchor-Free vs Anchor-Based

### Anchor-Based (YOLOv2-v5):
1. 预定义多个 anchor boxes
2. 对每个 anchor 预测偏移量
3. 需要 K-means 聚类选择 anchors
4. 对数据集敏感

预测: (tx, ty, tw, th) → 相对于 anchor 的偏移
解码: bx = anchor_x + sigmoid(tx)
     by = anchor_y + sigmoid(ty)
     bw = anchor_w * exp(tw)
     bh = anchor_h * exp(th)

### Anchor-Free (YOLOv8):
1. 直接预测目标位置
2. 不需要预定义 anchors
3. 更少的超参数
4. 更好的泛化能力

预测: (cx, cy, w, h) → 直接预测中心点和尺寸
     + (class_probs) → 类别概率

## Task-Aligned Assigner (TAL)

YOLOv8 使用 TAL 进行标签分配:

1. 对每个 GT box，计算所有预测框的:
   - 分类得分 (s_cls)
   - 定位得分 (IoU)
   
2. 对齐度量:
   t = s_cls^α * IoU^β
   其中 α=0.5, β=6.0
   
3. 选择 top-k 个预测框作为正样本

4. 归一化对齐度量作为软标签

优势:
- 动态标签分配
- 分类和定位任务对齐
- 提升训练效果

## Distribution Focal Loss (DFL)

YOLOv8 使用 DFL 进行边界框回归:

传统方法: 直接回归 (x, y, w, h)
DFL: 将回归问题转换为分类问题

示例（预测边界框的一条边）:
- 不直接预测距离 d
- 预测 d 附近的概率分布
- 使用 softmax 归一化
- 期望值作为最终预测

优势:
- 更准确的定位
- 提供不确定性估计
- 更好的梯度
    """
    
    print(explanation)


# ================================
# 5. YOLOv8 模型结构代码示例
# ================================

def model_architecture_pseudo_code():
    """
    YOLOv8 模型结构伪代码
    """
    print("\n" + "=" * 60)
    print("YOLOv8 模型结构（简化版）")
    print("=" * 60)
    
    pseudo_code = '''
# ===== C2f 模块 =====
class C2f(nn.Module):
    """
    CSP Bottleneck with 2 Convolutions
    核心改进：更丰富的梯度流
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) 
                               for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ===== SPPF 模块 =====
class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# ===== Backbone =====
class YOLOv8Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # P1/2
        self.conv1 = Conv(3, 32, 3, 2)  # 320x320
        
        # P2/4
        self.conv2 = Conv(32, 64, 3, 2)  # 160x160
        self.c2f1 = C2f(64, 64, n=1)
        
        # P3/8
        self.conv3 = Conv(64, 128, 3, 2)  # 80x80
        self.c2f2 = C2f(128, 128, n=2)
        
        # P4/16
        self.conv4 = Conv(128, 256, 3, 2)  # 40x40
        self.c2f3 = C2f(256, 256, n=2)
        
        # P5/32
        self.conv5 = Conv(256, 512, 3, 2)  # 20x20
        self.c2f4 = C2f(512, 512, n=1)
        self.sppf = SPPF(512, 512, k=5)

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.c2f1(self.conv2(p1))
        p3 = self.c2f2(self.conv3(p2))  # → Neck
        p4 = self.c2f3(self.conv4(p3))  # → Neck
        p5 = self.sppf(self.c2f4(self.conv5(p4)))  # → Neck
        return p3, p4, p5


# ===== Neck (PAN) =====
class YOLOv8Neck(nn.Module):
    def __init__(self):
        super().__init__()
        # Top-down
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f5 = C2f(512 + 256, 256, n=1)  # P5 + P4
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f6 = C2f(256 + 128, 128, n=1)  # P4 + P3
        
        # Bottom-up
        self.conv6 = Conv(128, 128, 3, 2)
        self.c2f7 = C2f(128 + 256, 256, n=1)  # P3 + P4
        
        self.conv7 = Conv(256, 256, 3, 2)
        self.c2f8 = C2f(256 + 512, 512, n=1)  # P4 + P5

    def forward(self, p3, p4, p5):
        # Top-down
        p4_up = self.c2f5(torch.cat([self.upsample1(p5), p4], 1))
        p3_up = self.c2f6(torch.cat([self.upsample2(p4_up), p3], 1))
        
        # Bottom-up
        p4_out = self.c2f7(torch.cat([self.conv6(p3_up), p4_up], 1))
        p5_out = self.c2f8(torch.cat([self.conv7(p4_out), p5], 1))
        
        return p3_up, p4_out, p5_out  # 80x80, 40x40, 20x20


# ===== 解耦检测头 =====
class DetectHead(nn.Module):
    def __init__(self, nc=80, ch=(128, 256, 512)):
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = 16  # DFL 通道数
        
        # 检测头
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, 64, 3), Conv(64, 64, 3),
                nn.Conv2d(64, 4 * self.reg_max, 1)
            ) for x in ch
        )  # 边界框回归
        
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, 80, 3), Conv(80, 80, 3),
                nn.Conv2d(80, self.nc, 1)
            ) for x in ch
        )  # 类别预测

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x


# ===== 完整模型 =====
class YOLOv8(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        self.backbone = YOLOv8Backbone()
        self.neck = YOLOv8Neck()
        self.head = DetectHead(nc=nc)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        outputs = self.head([n3, n4, n5])
        return outputs
    '''
    
    print(pseudo_code)


# ================================
# 6. 实用工具函数
# ================================

def practical_tips():
    """
    YOLOv8 实用技巧
    """
    print("\n" + "=" * 60)
    print("YOLOv8 实用技巧")
    print("=" * 60)
    
    tips = """
## 1. 模型选择建议

| 模型 | 参数量 | 速度 | 精度 | 使用场景 |
|------|--------|------|------|----------|
| YOLOv8n | 3.2M | 最快 | 较低 | 移动端、实时应用 |
| YOLOv8s | 11.2M | 快 | 中等 | 边缘设备、嵌入式 |
| YOLOv8m | 25.9M | 中等 | 良好 | 通用检测任务 |
| YOLOv8l | 43.7M | 慢 | 高 | 高精度要求 |
| YOLOv8x | 68.2M | 最慢 | 最高 | 离线处理、竞赛 |

## 2. 训练技巧

### 数据准备:
- 至少每类 1500+ 张图像
- 数据质量 > 数据数量
- 标注准确性很重要
- 数据增强可以提升泛化

### 超参数调优:
- batch_size: 根据 GPU 内存调整（16-32 常用）
- epochs: 100-300（小数据集可以更多）
- imgsz: 640 是标准，可以尝试 512 或 1024
- lr0: 0.01 是好的起点

### 训练监控:
- 关注 mAP50-95（主要指标）
- 查看损失曲线是否收敛
- 验证集不过拟合
- 使用 TensorBoard 可视化

## 3. 推理优化

### 速度优化:
- 使用 TensorRT (NVIDIA GPU)
- ONNX Runtime (CPU/GPU)
- 降低输入分辨率
- 使用更小的模型

### 批量推理:
results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'], batch=8)

### 半精度推理:
model = YOLO('yolov8n.pt')
model.model.half()  # FP16

## 4. 后处理技巧

### 调整置信度阈值:
results = model('image.jpg', conf=0.5)  # 默认 0.25

### 调整 NMS 阈值:
results = model('image.jpg', iou=0.7)  # 默认 0.45

### 过滤特定类别:
results = model('image.jpg', classes=[0, 2, 3])  # 只检测某些类

### 限制检测数量:
results = model('image.jpg', max_det=100)  # 最多 100 个检测

## 5. 常见问题

### Q: 训练时显存不足?
A: 减小 batch_size 或使用更小的模型

### Q: mAP 很低?
A: 检查数据标注质量、增加训练轮数、调整学习率

### Q: 检测漏检或误检?
A: 调整 conf 和 iou 阈值、增加训练数据

### Q: 推理速度慢?
A: 使用 TensorRT、降低分辨率、使用更小模型

## 6. 部署示例

### ONNX 导出并使用:
# 导出
model.export(format='onnx', simplify=True)

# 使用 ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession('yolov8n.onnx')

### TensorRT 加速:
# 导出
model.export(format='engine', half=True)

# 加载使用
model = YOLO('yolov8n.engine')
results = model('image.jpg')
    """
    
    print(tips)


# ================================
# 主函数
# ================================

if __name__ == "__main__":
    installation_guide()
    basic_usage_example()
    training_example()
    anchor_free_concept()
    model_architecture_pseudo_code()
    practical_tips()
    
    print("\n" + "=" * 60)
    print("YOLOv8 学习资料展示完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 安装: pip install ultralytics")
    print("2. 尝试预训练模型推理")
    print("3. 准备自己的数据集")
    print("4. 训练自定义模型")
    print("5. 优化和部署")
    print("\n官方资源:")
    print("- 文档: https://docs.ultralytics.com")
    print("- GitHub: https://github.com/ultralytics/ultralytics")
    print("- 教程: https://docs.ultralytics.com/tutorials/")

