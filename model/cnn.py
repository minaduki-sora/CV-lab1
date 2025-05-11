import torch.nn as nn
import torch.nn.functional as F

class EMNIST_CNN(nn.Module):
    def __init__(self, activation_fn='relu', use_bn=True, dropout_rate=0.5):
        super().__init__()
        
        # 定义激活函数
        if activation_fn == 'relu':
            self.activation = F.relu
        elif activation_fn == 'sigmoid':
            self.activation = F.sigmoid
        elif activation_fn == 'tanh':
            self.activation = F.tanh
        else:
            raise ValueError("Unsupported activation function")
            
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        
        # 定义网络层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 定义批归一化层
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            
        # 定义dropout层
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
            
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入图像是28x28，经过两次池化后为7x7
        self.fc2 = nn.Linear(128, 47)  # EMNIST有47个类别
        
    def forward(self, x):
        # 第一卷积层
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # 第二卷积层
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = self.fc1(x)
        x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
            
        x = self.fc2(x)
        return x