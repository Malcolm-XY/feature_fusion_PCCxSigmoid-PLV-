# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:06:57 2024

@author: 18307
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_3layers(nn.Module):
    def __init__(self, num_classes=3, input_size=5673):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class LeNet5Modern(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNet5Like(nn.Module):
    def __init__(self, channels=3, num_classes=3, input_size=(32, 32)):
        super().__init__()

        # 第1层卷积
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第2层卷积
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = nn.ReLU()
        
        # --- 动态计算 fc1 的输入特征数 ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, input_size[0], input_size[1])
            dummy_output = self.pool1(self.activation(self.conv1(dummy_input)))
            dummy_output = self.pool2(self.activation(self.conv2(dummy_output)))
            fc1_input_features = torch.flatten(dummy_output, 1).size(1)
        # --------------------------------

        self.fc1 = nn.Linear(fc1_input_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        return x

class LeNetAdaptive(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1) # kernel_size=3, 5
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # kernel_size=3, 5
        self.bn2 = nn.BatchNorm2d(128)
        
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1)) 
        
        self.activation = nn.ReLU()
        
        self.fc1 = nn.Linear(in_features=128, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.global_pool(x)  
        
        x = torch.flatten(x, 1)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
