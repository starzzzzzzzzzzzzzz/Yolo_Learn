# YOLO 训练实战指南

这份指南将手把手教你如何训练自己的 YOLO 模型（重点是 YOLOv8）。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据集准备](#2-数据集准备)
3. [数据标注](#3-数据标注)
4. [YOLOv8 训练](#4-yolov8-训练)
5. [模型评估](#5-模型评估)
6. [模型优化](#6-模型优化)
7. [模型部署](#7-模型部署)
8. [常见问题](#8-常见问题)

---

## 1. 环境准备

### 1.1 硬件要求

**最低配置:**
- CPU: 4 核心
- RAM: 8GB
- GPU: 4GB 显存（可选，但强烈推荐）
- 硬盘: 50GB 可用空间

**推荐配置:**
- CPU: 8 核心或更多
- RAM: 16GB 或更多
- GPU: NVIDIA GPU (8GB+ 显存)
  - 入门: RTX 3060 (12GB)
  - 推荐: RTX 3080 (10GB)
  - 理想: RTX 3090/4090 或 A100
- 硬盘: SSD 100GB+

### 1.2 软件安装

#### 步骤 1: 安装 Python

```bash
# 推荐使用 Python 3.8 - 3.11
python --version
```

#### 步骤 2: 创建虚拟环境

```bash
# 使用 conda
conda create -n yolo python=3.10
conda activate yolo

# 或使用 venv
python -m venv yolo_env
source yolo_env/bin/activate  # Linux/Mac
# yolo_env\Scripts\activate  # Windows
```

#### 步骤 3: 安装 PyTorch

访问 https://pytorch.org/get-started/locally/ 获取适合你系统的命令

```bash
# CUDA 11.8 示例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 版本（不推荐，训练会很慢）
pip install torch torchvision torchaudio
```

#### 步骤 4: 安装 YOLOv8

```bash
pip install ultralytics
```

#### 步骤 5: 验证安装

```python
# test_install.py
import torch
from ultralytics import YOLO

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 测试 YOLOv8
model = YOLO('yolov8n.pt')
print("✅ YOLOv8 安装成功！")
```

### 1.3 可选工具

```bash
# 数据标注工具
pip install labelimg  # 或使用 LabelMe, CVAT

# 可视化工具
pip install tensorboard
pip install wandb  # Weights & Biases

# 图像处理
pip install opencv-python pillow
pip install albumentations
```

---

## 2. 数据集准备

### 2.1 数据集结构

YOLO 需要特定的目录结构：

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img100.jpg
│   │   └── ...
│   └── test/  (可选)
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   ├── val/
│   │   ├── img100.txt
│   │   └── ...
│   └── test/  (可选)
│       └── ...
└── data.yaml
```

### 2.2 创建 data.yaml

```yaml
# data.yaml
path: /path/to/my_dataset  # 数据集根目录
train: images/train        # 训练集路径（相对于 path）
val: images/val            # 验证集路径
test: images/test          # 测试集路径（可选）

# 类别
nc: 3  # 类别数量
names: ['cat', 'dog', 'bird']  # 类别名称（索引对应 0, 1, 2...）
```

### 2.3 数据收集建议

#### 数量建议:
- **最少**: 每类 100-200 张
- **推荐**: 每类 500-1000 张
- **理想**: 每类 1000-5000 张

#### 质量要求:
1. **多样性**
   - 不同光照条件（白天、夜晚、室内、室外）
   - 不同角度（正面、侧面、俯视、仰视）
   - 不同距离（近景、中景、远景）
   - 不同背景
   - 不同姿态/状态

2. **图像质量**
   - 清晰度：避免模糊
   - 分辨率：至少 640×640
   - 格式：JPG, PNG
   - 避免过度压缩

3. **标注质量**
   - 准确的边界框
   - 正确的类别标签
   - 完整标注（不遗漏目标）
   - 一致的标注标准

#### 数据来源:
- 自己拍摄（最好）
- 公开数据集
  - COCO: https://cocodataset.org/
  - Open Images: https://storage.googleapis.com/openimages/web/index.html
  - Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/
- 爬虫采集（注意版权）
- 数据增强（扩充数据）

### 2.4 数据划分

```python
# split_dataset.py
import os
import shutil
from pathlib import Path
import random

def split_dataset(image_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    划分数据集为训练集、验证集和测试集
    """
    assert train_ratio + val_ratio + test_ratio == 1.0
    
    # 获取所有图像
    images = list(Path(image_dir).glob('*.jpg')) + \
             list(Path(image_dir).glob('*.png'))
    
    # 打乱
    random.shuffle(images)
    
    # 计算划分点
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # 划分
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    # 创建目录
    base_dir = Path(image_dir).parent
    for split in ['train', 'val', 'test']:
        (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    def copy_files(image_list, split):
        for img_path in image_list:
            # 复制图像
            shutil.copy(img_path, base_dir / 'images' / split / img_path.name)
            
            # 复制标签
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy(label_path, base_dir / 'labels' / split / label_path.name)
    
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    copy_files(test_images, 'test')
    
    print(f"✅ 数据集划分完成:")
    print(f"   训练集: {len(train_images)} 张")
    print(f"   验证集: {len(val_images)} 张")
    print(f"   测试集: {len(test_images)} 张")

# 使用示例
split_dataset('/path/to/all_images', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
```

---

## 3. 数据标注

### 3.1 YOLO 标注格式

每个图像对应一个同名的 `.txt` 文件，每行表示一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 类别索引（从 0 开始）
- `x_center`: 边界框中心 x 坐标（归一化到 [0, 1]）
- `y_center`: 边界框中心 y 坐标（归一化到 [0, 1]）
- `width`: 边界框宽度（归一化到 [0, 1]）
- `height`: 边界框高度（归一化到 [0, 1]）

**示例:**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

### 3.2 标注工具

#### LabelImg (推荐新手)

```bash
pip install labelimg
labelimg
```

**使用步骤:**
1. 打开目录
2. 选择 "PascalVOC" 或 "YOLO" 格式
3. 按 `W` 键开始标注
4. 拖动框选目标
5. 选择类别
6. 按 `Ctrl+S` 保存
7. 按 `D` 键下一张图

#### Roboflow (在线工具)

https://roboflow.com/

**优势:**
- 在线协作
- 自动数据增强
- 导出多种格式
- 数据集版本管理

#### CVAT (专业工具)

https://www.cvat.ai/

**优势:**
- 支持团队协作
- 视频标注
- 半自动标注
- 质量控制

### 3.3 标注质量检查

```python
# check_annotations.py
import cv2
from pathlib import Path

def visualize_annotations(image_dir, label_dir, class_names):
    """
    可视化标注结果，检查质量
    """
    image_files = list(Path(image_dir).glob('*.jpg'))
    
    for img_path in image_files:
        # 读取图像
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # 读取标签
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"⚠️ 缺少标签: {img_path.name}")
            continue
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"⚠️ 格式错误: {label_path.name} - {line}")
                    continue
                
                cls_id, x_c, y_c, box_w, box_h = map(float, parts)
                
                # 转换为像素坐标
                x1 = int((x_c - box_w/2) * w)
                y1 = int((y_c - box_h/2) * h)
                x2 = int((x_c + box_w/2) * w)
                y2 = int((y_c + box_h/2) * h)
                
                # 检查坐标范围
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    print(f"⚠️ 坐标越界: {img_path.name}")
                
                # 绘制
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = class_names[int(cls_id)]
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示
        cv2.imshow('Annotation Check', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

# 使用
class_names = ['cat', 'dog', 'bird']
visualize_annotations('images/train', 'labels/train', class_names)
```

---

## 4. YOLOv8 训练

### 4.1 基础训练

```python
# train.py
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # n, s, m, l, x

# 开始训练
results = model.train(
    data='data.yaml',       # 数据配置文件
    epochs=100,             # 训练轮数
    imgsz=640,              # 输入图像尺寸
    batch=16,               # 批次大小
    name='my_experiment',   # 实验名称
    device=0,               # GPU 设备 ID (或 'cpu')
)
```

### 4.2 完整训练配置

```python
# train_advanced.py
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model.train(
    # ===== 数据配置 =====
    data='data.yaml',
    
    # ===== 训练配置 =====
    epochs=300,              # 训练轮数
    patience=50,             # 早停耐心值
    batch=32,                # 批次大小（根据 GPU 内存调整）
    imgsz=640,               # 输入尺寸 (640, 1280...)
    save=True,               # 保存检查点
    save_period=10,          # 每 N 轮保存一次 (-1=只保存最后)
    cache='ram',             # 缓存图像 (False, 'ram', 'disk')
    device=0,                # GPU ID 或 'cpu'
    workers=8,               # 数据加载器线程数
    project='runs/train',    # 项目目录
    name='custom_model',     # 实验名称
    exist_ok=False,          # 覆盖现有实验
    pretrained=True,         # 使用预训练权重
    optimizer='SGD',         # 优化器 (SGD, Adam, AdamW, NAdam, RAdam, RMSProp)
    verbose=True,            # 详细输出
    seed=0,                  # 随机种子
    deterministic=True,      # 确定性模式
    single_cls=False,        # 单类别训练
    rect=False,              # 矩形训练
    cos_lr=False,            # 余弦学习率
    close_mosaic=10,         # 最后 N 轮关闭 mosaic
    resume=False,            # 恢复训练
    amp=True,                # 自动混合精度
    fraction=1.0,            # 使用数据集的比例 (0.0-1.0)
    profile=False,           # 性能分析
    
    # ===== 超参数 =====
    lr0=0.01,                # 初始学习率
    lrf=0.01,                # 最终学习率 (lr0 * lrf)
    momentum=0.937,          # SGD momentum / Adam beta1
    weight_decay=0.0005,     # 权重衰减
    warmup_epochs=3.0,       # warmup 轮数
    warmup_momentum=0.8,     # warmup 初始 momentum
    warmup_bias_lr=0.1,      # warmup 初始 bias 学习率
    box=7.5,                 # 边界框损失权重
    cls=0.5,                 # 分类损失权重
    dfl=1.5,                 # DFL 损失权重
    pose=12.0,               # 姿态损失权重（仅姿态）
    kobj=2.0,                # 关键点目标损失权重
    label_smoothing=0.0,     # 标签平滑 (0.0-1.0)
    nbs=64,                  # 名义批次大小
    overlap_mask=True,       # 训练时 mask 是否重叠
    mask_ratio=4,            # mask 下采样比率
    dropout=0.0,             # 使用 dropout 正则化
    val=True,                # 训练时验证
    
    # ===== 数据增强 =====
    hsv_h=0.015,             # HSV-Hue 增强
    hsv_s=0.7,               # HSV-Saturation 增强
    hsv_v=0.4,               # HSV-Value 增强
    degrees=0.0,             # 旋转角度 (+/- deg)
    translate=0.1,           # 平移 (+/- 比例)
    scale=0.5,               # 缩放增益 (+/- 比例)
    shear=0.0,               # 剪切角度 (+/- deg)
    perspective=0.0,         # 透视变换 (+/- 比例)
    flipud=0.0,              # 上下翻转概率
    fliplr=0.5,              # 左右翻转概率
    mosaic=1.0,              # Mosaic 增强概率
    mixup=0.0,               # MixUp 增强概率
    copy_paste=0.0,          # Copy-paste 增强概率
)

print("训练完成！")
print(f"最佳模型: {results.save_dir}/weights/best.pt")
```

### 4.3 从检查点恢复训练

```python
# 恢复训练
model = YOLO('runs/train/custom_model/weights/last.pt')
results = model.train(resume=True)
```

### 4.4 监控训练进程

#### 方法 1: TensorBoard

```bash
# 训练时自动生成 TensorBoard 日志
tensorboard --logdir runs/train
```

#### 方法 2: Weights & Biases

```python
# 集成 W&B
import wandb

wandb.login()

model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    project='my_yolo_project',  # W&B 会自动记录
)
```

### 4.5 训练技巧

#### 技巧 1: 学习率调优

```python
# 使用学习率查找器
model = YOLO('yolov8n.pt')

# 先用小学习率训练几轮，观察损失
# 然后逐步增加学习率
```

#### 技巧 2: 渐进式训练

```python
# 阶段 1: 冻结骨干，只训练头部
model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=50,
    freeze=10,  # 冻结前 10 层
)

# 阶段 2: 全模型训练
results = model.train(
    data='data.yaml',
    epochs=150,
    freeze=0,
)
```

#### 技巧 3: 多GPU 训练

```python
# 使用多个 GPU
model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    device=[0, 1, 2, 3],  # 使用 GPU 0-3
    batch=64,  # 总批次大小会分配到所有 GPU
)
```

---

## 5. 模型评估

### 5.1 验证模型

```python
# validate.py
from ultralytics import YOLO

model = YOLO('runs/train/custom_model/weights/best.pt')

# 在验证集上评估
metrics = model.val()

# 打印指标
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP75: {metrics.box.map75}")
print(f"各类别 mAP50-95: {metrics.box.maps}")
```

### 5.2 测试集推理

```python
# test.py
from ultralytics import YOLO
import cv2

model = YOLO('runs/train/custom_model/weights/best.pt')

# 单张图像
results = model('test_image.jpg')

# 批量图像
results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# 保存结果
for i, result in enumerate(results):
    result.save(f'output_{i}.jpg')
    
    # 或手动处理
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        print(f"检测: 类别={int(cls)}, 置信度={conf:.2f}, "
              f"位置=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
```

### 5.3 评估指标详解

```python
# analyze_metrics.py
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('runs/train/custom_model/weights/best.pt')
metrics = model.val()

# 主要指标
print("=" * 50)
print("模型性能指标")
print("=" * 50)
print(f"mAP50-95 (主要指标): {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP75: {metrics.box.map75:.4f}")
print(f"精确率 (Precision): {metrics.box.mp:.4f}")
print(f"召回率 (Recall): {metrics.box.mr:.4f}")

# 各类别指标
print("\n每个类别的 mAP50:")
for i, (name, ap) in enumerate(zip(model.names.values(), metrics.box.maps)):
    print(f"  {name}: {ap:.4f}")

# 不同尺寸目标的性能
print(f"\n小目标 mAP: {metrics.box.map_small:.4f}")
print(f"中目标 mAP: {metrics.box.map_medium:.4f}")
print(f"大目标 mAP: {metrics.box.map_large:.4f}")
```

### 5.4 混淆矩阵分析

训练后会自动生成混淆矩阵：

```
runs/train/custom_model/confusion_matrix.png
```

分析混淆矩阵可以发现：
- 哪些类别容易混淆
- 是否有系统性错误
- 需要增加哪些训练数据

---

## 6. 模型优化

### 6.1 超参数调优

```python
# tune.py
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# 自动超参数调优（实验性功能）
results = model.tune(
    data='data.yaml',
    epochs=100,
    iterations=50,  # 尝试次数
    optimizer='AdamW',
    plots=True,
    save=True,
    val=True,
)
```

### 6.2 模型剪枝

```python
# 使用更小的模型
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']

for model_name in models:
    model = YOLO(model_name)
    results = model.train(data='data.yaml', epochs=100)
    metrics = model.val()
    print(f"{model_name}: mAP={metrics.box.map:.4f}")
```

### 6.3 知识蒸馏（高级）

```python
# distillation.py
from ultralytics import YOLO

# 教师模型（大模型）
teacher = YOLO('yolov8x.pt')
teacher.train(data='data.yaml', epochs=300)

# 学生模型（小模型）
student = YOLO('yolov8n.pt')

# 使用教师模型的预测作为软标签
# （需要自定义训练循环，这里仅为概念示例）
```

### 6.4 数据增强调整

```python
# 如果过拟合，增加数据增强
results = model.train(
    data='data.yaml',
    epochs=100,
    mosaic=1.0,    # 始终使用 Mosaic
    mixup=0.2,     # 20% 概率使用 MixUp
    hsv_h=0.02,    # 增加颜色增强
    hsv_s=0.8,
    hsv_v=0.5,
    degrees=10.0,  # 增加旋转
    scale=0.7,     # 增加缩放范围
)

# 如果欠拟合，减少数据增强
results = model.train(
    data='data.yaml',
    epochs=100,
    mosaic=0.0,    # 关闭 Mosaic
    mixup=0.0,     # 关闭 MixUp
    degrees=0.0,   # 关闭旋转
)
```

---

## 7. 模型部署

### 7.1 导出模型

```python
# export.py
from ultralytics import YOLO

model = YOLO('runs/train/custom_model/weights/best.pt')

# ===== ONNX (通用) =====
model.export(format='onnx', simplify=True)

# ===== TensorRT (NVIDIA GPU) =====
model.export(format='engine', half=True, device=0)

# ===== CoreML (iOS/macOS) =====
model.export(format='coreml')

# ===== TFLite (移动端) =====
model.export(format='tflite')

# ===== OpenVINO (Intel) =====
model.export(format='openvino')

# ===== TorchScript =====
model.export(format='torchscript')
```

### 7.2 ONNX 推理

```python
# onnx_inference.py
import onnxruntime as ort
import numpy as np
import cv2

# 加载模型
session = ort.InferenceSession('best.onnx')

# 准备输入
img = cv2.imread('test.jpg')
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1)  # HWC -> CHW
img = np.expand_dims(img, 0)  # 添加 batch 维度
img = img.astype(np.float32) / 255.0

# 推理
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img})

# 处理输出
print(f"输出形状: {[out.shape for out in outputs]}")
```

### 7.3 TensorRT 加速

```python
# tensorrt_inference.py
from ultralytics import YOLO

# 加载 TensorRT 模型
model = YOLO('best.engine')

# 推理（自动使用 TensorRT）
results = model('test.jpg')

# TensorRT 通常比 PyTorch 快 2-5 倍
```

### 7.4 实时视频检测

```python
# realtime_detection.py
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 推理
    results = model(frame)
    
    # 绘制结果
    annotated_frame = results[0].plot()
    
    # 显示
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 7.5 Flask API 部署

```python
# app.py
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO('best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # 接收图像
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    
    # 推理
    results = model(img)
    
    # 提取结果
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': int(box.cls[0]),
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist()
        })
    
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 8. 常见问题

### Q1: GPU 显存不足怎么办？

**解决方案:**
1. 减小 `batch` 大小
2. 使用更小的模型 (n < s < m < l < x)
3. 降低 `imgsz` (640 → 512)
4. 关闭数据缓存 (`cache=False`)
5. 启用混合精度 (`amp=True`)

### Q2: mAP 很低怎么办？

**检查清单:**
1. 数据质量
   - 标注是否准确
   - 类别是否平衡
   - 数据量是否足够
2. 训练配置
   - epochs 是否足够 (100+)
   - 学习率是否合适
   - 是否使用预训练权重
3. 数据增强
   - 是否过度或不足
4. 模型选择
   - 模型是否太小

### Q3: 训练很慢怎么办？

**优化方法:**
1. 使用 GPU (`device=0`)
2. 增加 `workers` 数量
3. 缓存数据到内存 (`cache='ram'`)
4. 使用更小的图像尺寸
5. 减少数据增强复杂度

### Q4: 过拟合怎么办？

**解决方案:**
1. 增加训练数据
2. 增强数据增强
   - `mosaic=1.0`
   - `mixup=0.2`
3. 使用正则化
   - `weight_decay=0.001`
   - `dropout=0.1`
4. 早停 (`patience=50`)
5. 使用更小的模型

### Q5: 欠拟合怎么办？

**解决方案:**
1. 使用更大的模型
2. 增加训练轮数
3. 提高学习率
4. 减少数据增强
5. 减少正则化

### Q6: 推理速度慢怎么办？

**优化方法:**
1. 使用 TensorRT (`format='engine'`)
2. 使用 ONNX Runtime
3. 降低输入分辨率
4. 使用更小的模型
5. 批量推理
6. 半精度推理 (`half=True`)

### Q7: 小目标检测不好怎么办？

**改进方法:**
1. 增加输入尺寸 (`imgsz=1280`)
2. 使用更大的模型
3. 增加小目标训练样本
4. 使用 Mosaic 增强
5. 调整 anchor 尺寸

### Q8: 如何处理类别不平衡？

**方法:**
1. 数据层面
   - 过采样少数类
   - 欠采样多数类
   - 数据增强
2. 损失函数
   - 类别权重
   - Focal Loss
3. 后处理
   - 调整置信度阈值

---

## 总结

这份指南涵盖了从环境搭建到模型部署的完整流程。记住：

1. **数据质量 > 模型复杂度**
   - 高质量的标注
   - 充足的数据量
   - 合理的数据增强

2. **迭代优化**
   - 从小模型开始
   - 快速实验
   - 逐步改进

3. **监控指标**
   - 关注 mAP50-95
   - 查看混淆矩阵
   - 分析错误案例

4. **实践为王**
   - 多尝试不同配置
   - 记录实验结果
   - 总结经验教训

祝你训练成功！🎉

---

## 参考资源

- **YOLOv8 官方文档**: https://docs.ultralytics.com
- **GitHub**: https://github.com/ultralytics/ultralytics
- **论文**: https://arxiv.org/abs/...
- **社区**: https://community.ultralytics.com
- **教程**: https://docs.ultralytics.com/tutorials/

