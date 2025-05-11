# utils/data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

def get_emnist_loaders(batch_size=64, num_workers=2, k_folds=5, fold_idx=0):
    """获取EMNIST数据集，支持交叉验证"""
    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 加载完整数据集
    train_dataset = datasets.EMNIST(
        root='./data',
        split='balanced',
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.EMNIST(
        root='./data',
        split='balanced',
        train=False,
        download=True,
        transform=transform_test
    )
    
    # 创建交叉验证分割
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    train_indices, val_indices = list(kfold.split(train_dataset))[fold_idx]
    
    # 创建子集
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader