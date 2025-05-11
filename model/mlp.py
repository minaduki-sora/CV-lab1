import torch.nn as nn
import torch.nn.functional as F

class EMNIST_MLP(nn.Module):
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
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 47)  # EMNIST有47个类别
        
        # 定义批归一化层
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
            
        # 定义dropout层
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # 展平输入图像
        x = x.view(-1, 28 * 28)
        
        # 第一层
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.activation(x)
        if hasattr(self,"dropout") and self.dropout is not None:
            x = self.dropout(x)
        
        # 第二层
        x = self.fc2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.activation(x)
        if hasattr(self,"dropout") and self.dropout is not None:
            x = self.dropout(x)
        
        # 第三层
        x = self.fc3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.activation(x)
        if hasattr(self,"dropout") and self.dropout is not None:
            x = self.dropout(x)
        
        # 输出层
        x = self.fc4(x)
        
        return x