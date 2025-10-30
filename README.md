# YOLO 学习笔记 & 多模态研究

**研究方向：** VLM 幻觉检测 (Hallucination Detection in Vision-Language Models)

---

## 📚 学习进度

### ✅ 已完成

#### **YOLOv2**
- `YOLO_v2/Darknet19.py` - Darknet-19 骨干网络实现
- `YOLO_v2/yolov2.ipynb` - YOLOv2 学习笔记
- **核心理解：**
  - Anchor-based 检测策略
  - K-means 聚类生成锚框
  - Direct Location Prediction
  - Passthrough Layer

#### **YOLOv3**
- `YOLO_v3/Darknet53.py` - Darknet-53 骨干网络（ResNet风格）
- `YOLO_v3/Darknet-53.ipynb` - YOLOv3 学习笔记
- **核心理解：**
  - 多尺度预测（3个尺度）
  - FPN 特征金字塔
  - 独立逻辑分类器

#### **YOLOv8** ⭐
- `YOLO_v8/yolo_v8.ipynb` - 完整手写实现笔记
- `YOLO_v8/basic_modules.py` - Conv, C2f, SPPF 模块
- `YOLO_v8/detection_head.py` - 解耦头 + DFL
- `YOLO_v8/yolov8_model.py` - 完整模型架构
- **核心理解：**
  - **Anchor-free 检测策略** - 无需预定义锚框
  - **FPN-PAN 双向特征融合** - 语义+细节增强
  - **解耦头 (Decoupled Head)** - 分类和回归分支独立
  - **DFL (Distribution Focal Loss)** - 回归转分类，分布预测
  - **TAL (Task-Aligned Assigner)** - 任务对齐的正负样本分配
  - **CIoU Loss** - 完整的定位损失（IoU + 中心距离 + 宽高比）

---

## 🎯 核心概念总结

### **从 Anchor-based 到 Anchor-free**
```
YOLOv2/v3: 预定义锚框 → K-means聚类 → 预测偏移量
YOLOv8:    无锚框 → 直接预测 ltrb → 更灵活、泛化性强
```

### **FPN vs PAN**
```
FPN (Top-Down):  深层语义 → 浅层  ✓ 增强小物体检测
PAN (Bottom-Up): 浅层细节 → 深层  ✓ 增强大物体定位
协同效果: 所有尺度都兼具语义和细节
```

### **检测点与感受野**
```
特征图 (80×80) = 6400 个检测点
每个检测点 = 一个潜在的检测位置（Anchor Point）
感受野 = 该检测点在原图中"看到"的区域
语义信息 = 包含物体存在性、类别、位置三重含义
```

---

## 🔬 研究应用

### **目标：VLM 幻觉检测**

```
Pipeline:
输入图像
  ↓
YOLO 检测 → 获取物体列表 [dog, grass, tree]
  ↓
VLM/BLIP → 生成描述 "一只狗在草地上玩飞盘"
  ↓
对齐模块 → 检测哪些词有视觉证据
  ↓
幻觉检测 → "飞盘"是幻觉 (YOLO未检测到)
```

---

## 📂 项目结构

```
YOLO/
├── YOLO_v2/
│   ├── Darknet19.py          # 手写骨干网络
│   └── yolov2.ipynb           # 学习笔记
│
├── YOLO_v3/
│   ├── Darknet53.py           # ResNet风格骨干
│   └── Darknet-53.ipynb       # 学习笔记
│
├── YOLO_v8/
│   ├── yolo_v8.ipynb          # 完整手写实现 ⭐
│   ├── basic_modules.py       # Conv, C2f, SPPF
│   ├── detection_head.py      # 解耦头 + DFL
│   └── yolov8_model.py        # 完整模型
│
└── BLIP/                      # 下一步学习
    └── (待创建)
```

---

## 🚀 下一步学习计划

### **BLIP (Bootstrapping Language-Image Pre-training)**

**为什么学 BLIP：**
- ✅ 直接相关 VLM 幻觉检测研究
- ✅ 有 ViT + CLIP 基础，学习曲线平滑
- ✅ 理解多模态生成机制
- ✅ 可与 YOLO 结合做实验

**学习重点：**
1. Encoder-Decoder 统一架构
2. 三个训练目标：ITC + ITM + LM
3. CapFilt 数据自举策略
4. 图像描述生成 (Image Captioning)
5. 与 YOLO 结合进行幻觉检测

---

## 🛠️ 技术栈

- **Deep Learning Framework:** PyTorch
- **Vision:** YOLOv8, ViT
- **Language-Vision:** CLIP, BLIP (learning)
- **Research Focus:** Attention Analysis, Hallucination Detection

---

## 📖 学习资源

### **论文**
- YOLOv2: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
- YOLOv3: [An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- YOLOv8: [Ultralytics Documentation](https://docs.ultralytics.com/)
- BLIP: [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)

### **代码**
- YOLOv8 Official: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- BLIP Official: [salesforce/BLIP](https://github.com/salesforce/BLIP)

---

**Last Updated:** 2025-10-30

