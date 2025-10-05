---
title: "PyTorch-09-使用示例与最佳实践"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 最佳实践
  - 实战经验
  - 源码分析
categories:
  - PyTorch
description: "源码剖析 - PyTorch-09-使用示例与最佳实践"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# PyTorch-09-使用示例与最佳实践

## 完整训练流程示例

### 图像分类任务（ResNet-18）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ============ 1. 数据准备 ============
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('data/train', transform=transform_train)
val_dataset = datasets.ImageFolder('data/val', transform=transform_val)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True  # CUDA优化：固定内存加速传输
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ============ 2. 模型构建 ============
model = models.resnet18(pretrained=False, num_classes=10)
model = model.cuda()

# 多GPU训练
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    # 或使用DDP（推荐）
    # model = nn.parallel.DistributedDataParallel(model)

# ============ 3. 损失函数和优化器 ============
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# 学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# ============ 4. 混合精度训练 ============
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# ============ 5. 训练循环 ============
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        # 混合精度前向传播
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(loader)}, '
                  f'Loss: {running_loss/(batch_idx+1):.3f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# ============ 6. 训练主循环 ============
num_epochs = 200
best_acc = 0

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 60)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # 学习率调度
    scheduler.step()
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': val_acc,
        }, 'best_model.pth')
        print(f'Saved best model with acc: {best_acc:.2f}%')
```

## 性能优化最佳实践

### 1. 数据加载优化

```python
# ✅ 推荐：多进程加载 + pin_memory
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # 使用多进程
    pin_memory=True,    # 固定内存（CUDA优化）
    prefetch_factor=2,  # 预取批次数
    persistent_workers=True  # 保持worker进程（减少重启开销）
)

# ❌ 避免：单进程加载
train_loader = DataLoader(dataset, batch_size=32, num_workers=0)
```

### 2. 内存优化

```python
# ✅ 梯度累积（模拟大batch size）
accumulation_steps = 4
model.zero_grad()

for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps  # 归一化
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        model.zero_grad()

# ✅ 梯度检查点（减少显存）
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)  # 不保存中间激活
        x = checkpoint(self.layer2, x)
        return x

# ✅ 释放不需要的张量
tensor = tensor.detach()  # 从计算图分离
del tensor  # 显式删除
torch.cuda.empty_cache()  # 清空显存缓存
```

### 3. 计算优化

```python
# ✅ 使用inplace操作（节省内存）
x.relu_()  # inplace
x.add_(y)

# ✅ 避免不必要的CPU-GPU传输
# ❌ 避免：
for i in range(len(tensor)):
    print(tensor[i].item())  # 每次都CPU-GPU同步

# ✅ 推荐：
print(tensor.tolist())  # 一次性传输

# ✅ 使用torch.no_grad()禁用autograd
with torch.no_grad():
    for inputs, targets in val_loader:
        outputs = model(inputs)
        # 推理不需要构建计算图

# ✅ 设置benchmark模式（cuDNN自动调优）
torch.backends.cudnn.benchmark = True  # 训练时开启
torch.backends.cudnn.deterministic = False  # 允许非确定性算法（更快）
```

### 4. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for inputs, targets in train_loader:
    optimizer.zero_grad()
    
    # 自动混合精度
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # 缩放梯度防止下溢
    scaler.scale(loss).backward()
    
    # 梯度裁剪（防止爆炸）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()

# 性能提升：1.5-2x（Tensor Core加速）
# 显存占用：减少约50%
```

### 5. 分布式训练（DDP）

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # 设置当前进程的设备
    torch.cuda.set_device(rank)
    
    # 创建模型并移到对应GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 分布式采样器（每个进程处理不同数据）
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 打乱数据
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 保存模型（仅rank 0）
        if rank == 0:
            torch.save(model.module.state_dict(), 'model.pth')
    
    cleanup()

# 启动多进程训练
import torch.multiprocessing as mp

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

## 常见陷阱与解决

### 1. 显存泄漏

```python
# ❌ 陷阱：保留了计算图引用
losses = []
for inputs, targets in train_loader:
    loss = criterion(model(inputs), targets)
    losses.append(loss)  # loss持有整个计算图！
    loss.backward()

# ✅ 解决：只保存标量值
losses = []
for inputs, targets in train_loader:
    loss = criterion(model(inputs), targets)
    losses.append(loss.item())  # 只保存数值
    loss.backward()

# ❌ 陷阱：不必要的tensor累积
total = 0
for x in tensors:
    total = total + x  # 创建新tensor

# ✅ 解决：使用inplace或.item()
total = 0.0
for x in tensors:
    total += x.item()
```

### 2. 梯度爆炸/消失

```python
# ✅ 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# ✅ 使用BatchNorm/LayerNorm稳定训练
model = nn.Sequential(
    nn.Linear(100, 100),
    nn.BatchNorm1d(100),  # 归一化
    nn.ReLU(),
    nn.Linear(100, 10)
)

# ✅ 检测梯度异常
torch.autograd.set_detect_anomaly(True)  # 开发时开启
```

### 3. 数据类型错误

```python
# ❌ 陷阱：混合精度类型
x = torch.randn(10)  # float32
w = torch.randn(10, dtype=torch.float64)  # float64
y = x @ w  # 错误：类型不匹配

# ✅ 解决：统一类型
y = x.double() @ w  # 转为float64
# 或
y = x @ w.float()  # 转为float32

# ✅ 设置默认类型
torch.set_default_dtype(torch.float32)
```

### 4. 推理模式错误

```python
# ❌ 陷阱：忘记切换到eval模式
model.train()  # 训练模式
with torch.no_grad():
    outputs = model(inputs)  # BatchNorm/Dropout仍在训练模式！

# ✅ 解决：明确设置eval
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

## 模型调试技巧

### 1. 检查梯度流

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f'{name}: grad_norm={grad_norm:.4f}')
            if grad_norm == 0:
                print(f'Warning: {name} has zero gradient!')

# 使用
loss.backward()
check_gradients(model)
```

### 2. 可视化特征图

```python
import matplotlib.pyplot as plt

def visualize_feature_maps(model, input_image):
    activation = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册hook
    model.layer1.register_forward_hook(hook_fn('layer1'))
    model.layer2.register_forward_hook(hook_fn('layer2'))
    
    # 前向传播
    model(input_image)
    
    # 可视化
    for name, feat in activation.items():
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < feat.shape[1]:
                ax.imshow(feat[0, i].cpu(), cmap='viridis')
                ax.axis('off')
        plt.suptitle(f'{name} Feature Maps')
        plt.show()
```

### 3. Profiling性能

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_forward"):
        output = model(input)
    with record_function("loss_backward"):
        loss = criterion(output, target)
        loss.backward()

# 打印结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出Chrome trace
prof.export_chrome_trace("trace.json")
```

## 生产部署实践

### 1. 模型量化

```python
import torch.quantization

# 动态量化（仅权重量化）
model_fp32 = MyModel()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear, nn.Conv2d},  # 量化的层类型
    dtype=torch.qint8
)

# 模型大小减少约4x，推理速度提升2-3x

# 静态量化（激活也量化）
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model_fp32)

# 校准（使用代表性数据）
for inputs, _ in calibration_loader:
    model_prepared(inputs)

model_int8 = torch.quantization.convert(model_prepared)
```

### 2. 模型剪枝

```python
import torch.nn.utils.prune as prune

# 非结构化剪枝
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)  # 剪枝30%

# 结构化剪枝
prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)  # 剪枝50%输出通道

# 永久移除剪枝掩码
for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, 'weight')
```

### 3. ONNX导出与优化

```python
import onnx
import onnxruntime

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=13,
    do_constant_folding=True,  # 常量折叠优化
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}}  # 动态batch size
)

# 优化ONNX模型
model_onnx = onnx.load("model.onnx")
model_onnx = onnx.optimizer.optimize(model_onnx)

# 使用ONNXRuntime推理
session = onnxruntime.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
outputs = session.run(None, {'input': input_array})
```

### 4. TorchServe部署

```bash
# 打包模型
torch-model-archiver \
    --model-name resnet18 \
    --version 1.0 \
    --serialized-file model.pt \
    --handler image_classifier \
    --export-path model_store

# 启动服务
torchserve --start \
    --model-store model_store \
    --models resnet18=resnet18.mar \
    --ncs

# 推理请求
curl -X POST http://localhost:8080/predictions/resnet18 \
    -T image.jpg
```

## 总结建议

### 训练阶段

1. **数据**：使用DataLoader多进程 + pin_memory
2. **模型**：使用DDP而非DataParallel
3. **精度**：启用混合精度训练（AMP）
4. **显存**：梯度累积或梯度检查点
5. **调试**：开启异常检测（detect_anomaly）

### 推理阶段

1. **模式**：`model.eval()` + `torch.no_grad()`
2. **优化**：TorchScript编译 + freeze
3. **量化**：动态或静态量化（INT8）
4. **批处理**：使用更大batch size
5. **硬件**：CUDA Graph + TensorRT

### 部署阶段

1. **格式**：TorchScript（跨语言）或ONNX（跨框架）
2. **服务**：TorchServe或Triton Inference Server
3. **监控**：集成Prometheus + Grafana
4. **弹性**：Kubernetes + HPA
5. **安全**：模型加密 + API认证

---

**文档版本**: v1.0  
**最后更新**: 2025-01-01

## 完整文档列表

1. **PyTorch-00-总览.md** - 整体架构与设计理念
2. **PyTorch-01-c10核心库-概览.md** - c10基础抽象
3. **PyTorch-01-c10核心库-TensorImpl详解.md** - 张量底层实现
4. **PyTorch-01-c10核心库-时序图与API.md** - 核心流程与API
5. **PyTorch-02-ATen张量库.md** - Dispatcher与算子实现
6. **PyTorch-03-Autograd自动微分.md** - 计算图与梯度引擎
7. **PyTorch-04-torch.nn模块与JIT编译.md** - 神经网络与编译优化
8. **PyTorch-09-使用示例与最佳实践.md** - 实战案例与性能优化

**全部文档已完成！**

