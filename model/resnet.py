import torch.nn as nn
import torch.nn.functional as F

class EMNIST_ResNet(nn.Module):
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
        
        # 定义批归一化层
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            
        # 定义dropout层
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
            
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 残差连接用的1x1卷积
        self.shortcut1 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 7 * 7, 256)  # 减少全连接层大小
        self.fc2 = nn.Linear(256, 47)  # EMNIST有47个类别
        
    def forward(self, x):
        # 第一残差块
        identity = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        
        # 残差连接
        identity = self.shortcut1(identity)
        out += identity
        out = self.activation(out)
        out = self.pool(out)
        
        # 第二残差块
        identity = out
        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)
        
        # 残差连接
        identity = self.shortcut2(identity)
        out += identity
        out = self.activation(out)
        out = self.pool(out)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.fc1(out)
        out = self.activation(out)
        
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.fc2(out)
        
        return out