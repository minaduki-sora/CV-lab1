# 计算机视觉实验一报告

组号：27

组员1：SA23011062 孙豪

组员2：SA24011060 马俊杰

组员3：SA24011174 李京

组员4：SA24231094 钱佳伟

## 引言

### 数据集

本项目使用EMNIST（Extended MNIST）数据集，这是由NIST特殊数据库19扩展而来的手写字符数据集。我们选择了"Balanced"子集，该子集解决了原始数据集中类别不平衡的问题，并统一了大小写字母的表示方式。具体数据统计如下：

- 训练集：112,800个样本
- 测试集：18,800个样本
- 总样本数：131,600
- 类别数：47（平衡分布）

每个样本为28×28像素的灰度图像，包含数字、大小写字母等多种字符。该数据集在字符识别领域具有重要研究价值，能够有效评估模型处理复杂手写字符的能力。

### 模型结构

#### 1. 多层感知机(MLP)

MLP模型采用全连接网络结构：

- **输入层**：将28×28图像展平为784维向量

- **隐藏层**：3个全连接层（512→256→128神经元）

- **输出层**：47个神经元对应分类类别

- **激活函数**：支持ReLU、LeakyReLU和ELU三种选择

- **正则化技术**：

  - 批归一化(BatchNorm)：每个隐藏层后可选
  - Dropout：默认概率0.5

- 示例：

  ```
  Flatten output shape: 	 torch.Size([1, 784])
  Linear output shape: 	 torch.Size([1, 512])
  ReLU output shape: 	 torch.Size([1, 512])
  Dropout output shape: 	 torch.Size([1, 512])
  Linear output shape: 	 torch.Size([1, 256])
  ReLU output shape: 	 torch.Size([1, 256])
  Dropout output shape: 	 torch.Size([1, 256])
  Linear output shape: 	 torch.Size([1, 128])
  ReLU output shape: 	 torch.Size([1, 128])
  Dropout output shape: 	 torch.Size([1, 128])
  Linear output shape: 	 torch.Size([1, 47])

#### 2. 卷积神经网络(CNN)

CNN模型结构如下：

- **卷积部分**：

  - 2个卷积块，每块包含：
    - 3×3卷积层（通道数32→64）
    - 批归一化层
    - 激活函数
    - 2×2最大池化

- **全连接部分**：

  - 将特征图展平后通过256神经元全连接层
  - 最终输出47维分类结果

- **技术特点**：

  - 支持多种激活函数
  - 可配置Dropout和批归一化
  - 自动适应输入尺寸变化

- 示例:

  ```
  卷积层:
  Conv2d output shape: 	 torch.Size([1, 32, 28, 28])
  Identity output shape: 	 torch.Size([1, 32, 28, 28])
  ReLU output shape: 	 torch.Size([1, 32, 28, 28])
  MaxPool2d output shape: 	 torch.Size([1, 32, 14, 14])
  Conv2d output shape: 	 torch.Size([1, 64, 14, 14])
  Identity output shape: 	 torch.Size([1, 64, 14, 14])
  ReLU output shape: 	 torch.Size([1, 64, 14, 14])
  MaxPool2d output shape: 	 torch.Size([1, 64, 7, 7])
  
  全连接层:
  Flatten output shape: 	 torch.Size([1, 3136])
  Linear output shape: 	 torch.Size([1, 256])
  Identity output shape: 	 torch.Size([1, 256])
  ReLU output shape: 	 torch.Size([1, 256])
  Dropout output shape: 	 torch.Size([1, 256])
  Linear output shape: 	 torch.Size([1, 47])
  ```

#### 3.残差网络(ResNet)：

ResNet结构如下：

- **基础结构**：

  - 初始卷积层（64通道）
  - 3个残差块组（64→128→256通道）
  - 全局平均池化替代全连接层

- **残差块设计**：

  - 包含两个3×3卷积层
  - 快捷连接处理维度变化
  - 支持多种激活函数

- **技术优化**：

  - 跳跃连接实现恒等映射
  - 批归一化加速收敛
  - Dropout防止过拟合

- 示例：

  ```
  初始卷积层:
  Conv2d output shape: 	 torch.Size([1, 64, 28, 28])
  Identity output shape: 	 torch.Size([1, 64, 28, 28])
  Activation output shape: 	 torch.Size([1, 64, 28, 28])
  
  残差块层1:
  Layer1 output shape: 	 torch.Size([1, 64, 28, 28])
  
  残差块层2:
  Layer2 output shape: 	 torch.Size([1, 128, 14, 14])
  
  残差块层3:
  Layer3 output shape: 	 torch.Size([1, 256, 7, 7])
  
  池化和全连接层:
  AvgPool2d output shape: 	 torch.Size([1, 256, 1, 1])
  Flatten output shape: 	 torch.Size([1, 256])
  Dropout output shape: 	 torch.Size([1, 256])
  Linear output shape: 	 torch.Size([1, 47])
  
  完整模型前向传播:
  最终输出形状: torch.Size([1, 47])
  ```



## 实验方法

本项目采用系统化的实验流程：

1. **基准模型建立**：为每种网络结构设置合理的初始参数
2. **超参数调优**：依次优化以下关键因素：
   - 学习率调度策略（StepLR和ReduceLROnPlateau）
   - 激活函数类型（ReLU/LeakyReLU/ELU）
   - 优化器选择（Adam/SGD/RMSprop）
   - 批归一化效果验证
   - L1/L2正则化比较
   - Dropout比例调整
3. **模型评估**：
   - 训练/测试损失曲线
   - 分类准确率变化
   - 混淆矩阵分析
   - 计算精确率、召回率和F1分数
4. **对比分析**：
   - 三种模型性能差异比较
   - 残差结构对CNN的影响
   - 计算效率评估

通过这一系统化方法，我们能够全面评估不同网络结构在EMNIST数据集上的表现，并深入理解各种深度学习技术的实际效果。