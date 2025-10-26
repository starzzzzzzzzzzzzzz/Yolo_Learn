import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Darknet19(nn.Module):
    """
    Darknet19 骨干网络
    YOLOv2 的核心特性：
    1. 所有卷积层后都加入 BN 层（改善收敛性）
    2. 使用 Leaky ReLU 激活函数
    3. 使用 1×1 和 3×3 卷积的组合
    """
    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        
        # Conv Block 1: 输入 3 → 32
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Block 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Block 3: 64 → 128 → 64 → 128
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Conv Block 4: 128 → 256 → 128 → 256
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Conv Block 5: 256 → 512 → 256 → 512 → 256 → 512
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
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Conv Block 6: 512 → 1024 → 512 → 1024 → 512 → 1024
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
        
        # 分类层
        self.conv19 = nn.Conv2d(1024, num_classes, 1, 1, 0)
        self.pool6 = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        """
        前向传播
        注意：每个卷积层后都有 BN + Leaky ReLU
        """
        # Block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = self.pool1(x)
        
        # Block 2
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = self.pool2(x)
        
        # Block 3
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1)
        x = self.pool3(x)
        
        # Block 4
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.1)
        x = F.leaky_relu(self.bn7(self.conv7(x)), 0.1)
        x = F.leaky_relu(self.bn8(self.conv8(x)), 0.1)
        x = self.pool4(x)
        
        # Block 5
        x = F.leaky_relu(self.bn9(self.conv9(x)), 0.1)
        x = F.leaky_relu(self.bn10(self.conv10(x)), 0.1)
        x = F.leaky_relu(self.bn11(self.conv11(x)), 0.1)
        x = F.leaky_relu(self.bn12(self.conv12(x)), 0.1)
        x = F.leaky_relu(self.bn13(self.conv13(x)), 0.1)
        x = self.pool5(x)
        
        # Block 6
        x = F.leaky_relu(self.bn14(self.conv14(x)), 0.1)
        x = F.leaky_relu(self.bn15(self.conv15(x)), 0.1)
        x = F.leaky_relu(self.bn16(self.conv16(x)), 0.1)
        x = F.leaky_relu(self.bn17(self.conv17(x)), 0.1)
        x = F.leaky_relu(self.bn18(self.conv18(x)), 0.1)
        
        # 分类
        x = self.conv19(x)
        x = self.pool6(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        return x