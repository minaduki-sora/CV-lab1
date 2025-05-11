# train/train_resnet.py
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from accelerate import Accelerator
from accelerate.utils import set_seed
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.resnet import EMNIST_ResNet
from utils.data_loader import get_emnist_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet on EMNIST')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results')
    return parser.parse_args()

def train_one_epoch(model, train_loader, optimizer, criterion, accelerator, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        optimizer.zero_grad()
        
        with accelerator.accumulate(model):
            output = model(data)
            loss = criterion(output, target)
            
            accelerator.backward(loss)
            optimizer.step()
            
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    train_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate(model, val_loader, criterion, accelerator):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss = total_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def train_model(args):
    # 初始化accelerate
    accelerator = Accelerator()
    set_seed(42)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录训练数据
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 加载数据
    train_loader, val_loader, test_loader = get_emnist_loaders(
        batch_size=args.batch_size,
        k_folds=args.k_folds,
        fold_idx=args.fold
    )
    
    # 初始化模型
    model = EMNIST_ResNet(
        activation_fn=args.activation,
        use_bn=args.use_bn,
        dropout_rate=args.dropout
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    
    # 准备模型和数据
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    
    # 记录内存和时间
    start_time = time.time()
    if accelerator.is_local_main_process:
        print(f"Starting training on {accelerator.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_acc = 0.0
    
    # 训练循环
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, accelerator, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, accelerator)
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 记录结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if accelerator.is_local_main_process:
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
        
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # 训练结束
    end_time = time.time()
    training_time = end_time - start_time
    
    if accelerator.is_local_main_process:
        # 打印训练时间和内存使用
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # 测试最佳模型
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
        test_loss, test_acc = validate(model, test_loader, criterion, accelerator)
        print(f"Test accuracy: {test_acc:.2f}%")
        
        # 绘制损失和准确率曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
        plt.close()
        
        # 保存训练结果
        np.savez(os.path.join(args.output_dir, 'training_stats.npz'),
                 train_losses=train_losses,
                 train_accs=train_accs,
                 val_losses=val_losses,
                 val_accs=val_accs,
                 test_acc=test_acc,
                 training_time=training_time)

def cross_validate(args):
    """执行交叉验证"""
    best_acc = 0.0
    best_params = None
    
    # 测试不同的超参数组合
    for activation in ['relu', 'tanh']:
        for use_bn in [True, False]:
            for dropout in [0.3, 0.5]:
                args.activation = activation
                args.use_bn = use_bn
                args.dropout = dropout
                
                fold_accs = []
                
                # 执行k折交叉验证
                for fold in range(args.k_folds):
                    args.fold = fold
                    print(f"\nCross-validation: activation={activation}, use_bn={use_bn}, dropout={dropout}, fold={fold+1}/{args.k_folds}")
                    
                    # 训练模型
                    train_model(args)
                    
                    # 加载训练结果
                    stats = np.load(os.path.join(args.output_dir, 'training_stats.npz'))
                    fold_accs.append(stats['test_acc'])
                
                # 计算平均准确率
                mean_acc = np.mean(fold_accs)
                print(f"Mean accuracy for activation={activation}, use_bn={use_bn}, dropout={dropout}: {mean_acc:.2f}%")
                
                # 更新最佳参数
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_params = {
                        'activation': activation,
                        'use_bn': use_bn,
                        'dropout': dropout
                    }
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best mean accuracy: {best_acc:.2f}%")
    
    # 用最佳参数训练最终模型
    args.activation = best_params['activation']
    args.use_bn = best_params['use_bn']
    args.dropout = best_params['dropout']
    args.fold = 0  # 使用全部训练数据
    
    print("\nTraining final model with best parameters...")
    train_model(args)

if __name__ == '__main__':
    args = parse_args()
    
    # 执行交叉验证或直接训练
    if args.k_folds > 1:
        cross_validate(args)
    else:
        train_model(args)