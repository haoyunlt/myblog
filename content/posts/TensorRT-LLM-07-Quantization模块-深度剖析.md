---
title: "TensorRT-LLM-07-Quantization模块-深度剖析"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 源码分析
categories:
  - TensorRT
description: "源码剖析 - TensorRT-LLM-07-Quantization模块-深度剖析"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# TensorRT-LLM-07-Quantization模块-深度剖析

## 一、模块概览

### 1.1 模块定位

Quantization 模块是TensorRT-LLM的量化加速核心，提供多种精度量化技术，在保持模型精度的同时显著降低内存占用和提升推理速度。

**核心职责：**
- 量化算法：FP8、INT8、INT4、INT3等
- 校准技术：SmoothQuant、AWQ、GPTQ等
- KV Cache量化：减少内存占用
- 权重量化：压缩模型大小
- 激活量化：降低计算精度

### 1.2 支持的量化技术

| 量化类型 | 精度 | 加速比 | 精度保持 | 硬件需求 | 适用场景 |
|---------|------|--------|---------|---------|---------|
| **FP8** | 8位浮点 | 1.5-2x | ~100% | H100/H200 | 高精度推理 |
| **INT8 SmoothQuant** | 8位整数 | 1.5-2x | 99%+ | V100+ | 通用加速 |
| **INT4 AWQ** | 4位整数 | 2-3x | 95%+ | A100+ | 内存受限 |
| **INT4 GPTQ** | 4位整数 | 2-3x | 95%+ | A100+ | 权重压缩 |
| **W8A16** | 权重8位,激活16位 | 1.2-1.5x | 98%+ | 通用 | 权重压缩 |
| **KV Cache INT8** | KV 8位 | 内存减半 | 99%+ | 通用 | 长序列 |

### 1.3 模块架构

```
tensorrt_llm/quantization/
├── __init__.py                     # 量化接口导出
├── quantize.py                     # 统一量化入口
├── mode.py                         # QuantMode定义
├── layers.py                       # 量化层实现
├── functional.py                   # 量化函数
│
├── algorithms/                     # 量化算法
│   ├── smooth_quant.py            # SmoothQuant实现
│   ├── awq.py                     # AWQ实现
│   ├── gptq.py                    # GPTQ实现
│   └── fp8_quant.py               # FP8量化
│
├── calib/                         # 校准相关
│   ├── int8/                      # INT8校准
│   └── calibrator.py              # 校准器实现
│
└── kernels/                       # 量化kernel
    ├── fp8_gemm_kernel.py         # FP8 GEMM
    ├── int8_gemm_kernel.py        # INT8 GEMM
    └── int4_gemm_kernel.py        # INT4 GEMM
```

### 1.4 量化流程

```
原始模型(FP16) → 校准数据集 → 量化校准 → 量化参数 → 量化模型 → TRT引擎
      ↓              ↓           ↓          ↓          ↓         ↓
   权重+激活      代表性样本   统计分析    scale/zp   压缩表示   优化推理
```

## 二、核心API详细剖析

### 2.1 quantize()统一接口

#### 2.1.1 函数签名

```python
def quantize(
    model: PretrainedModel,              # 待量化模型
    quant_config: Union[str, QuantConfig], # 量化配置
    calib_dataset: Optional[Dataset] = None, # 校准数据集
    calib_size: int = 512,               # 校准样本数
    random_seed: int = 42,               # 随机种子
    tokenizer = None,                    # 分词器
    **kwargs
) -> PretrainedModel:
    """
    统一量化接口
    
    Args:
        model: 预训练模型
        quant_config: 量化配置（"fp8", "int8_sq", "int4_awq"等）
        calib_dataset: 校准数据集（INT8/INT4需要）
        
    Returns:
        量化后的模型
    """
```

#### 2.1.2 量化配置

**QuantConfig结构体**

| 字段 | 类型 | 说明 | 示例值 |
|-----|------|------|--------|
| quant_algo | QuantAlgo | 量化算法 | FP8, INT8_SQ, INT4_AWQ |
| kv_cache_quant_algo | QuantAlgo | KV Cache量化 | INT8, FP8 |
| exclude_modules | List[str] | 排除模块 | ["lm_head"] |
| per_channel | bool | 是否按通道量化 | True |
| per_token | bool | 是否按token量化 | True |
| use_plugin | bool | 是否使用插件 | True |
| calib_method | str | 校准方法 | "max", "percentile" |

#### 2.1.3 核心实现

```python
def quantize(model, quant_config, calib_dataset=None, **kwargs):
    # 1. 解析量化配置
    if isinstance(quant_config, str):
        quant_config = _parse_quant_config(quant_config)
    
    # 2. 选择量化算法
    if quant_config.quant_algo == QuantAlgo.FP8:
        return _quantize_fp8(model, quant_config)
    
    elif quant_config.quant_algo == QuantAlgo.INT8_SMOOTHQUANT:
        return _quantize_int8_smoothquant(model, quant_config, calib_dataset)
    
    elif quant_config.quant_algo == QuantAlgo.INT4_AWQ:
        return _quantize_int4_awq(model, quant_config, calib_dataset)
    
    elif quant_config.quant_algo == QuantAlgo.INT4_GPTQ:
        return _quantize_int4_gptq(model, quant_config, calib_dataset)
    
    else:
        raise ValueError(f"Unsupported quantization algorithm: {quant_config.quant_algo}")

def _parse_quant_config(config_str: str) -> QuantConfig:
    """解析字符串配置"""
    config_map = {
        "fp8": QuantConfig(
            quant_algo=QuantAlgo.FP8,
            kv_cache_quant_algo=QuantAlgo.FP8,
        ),
        "int8_sq": QuantConfig(
            quant_algo=QuantAlgo.INT8_SMOOTHQUANT,
            kv_cache_quant_algo=QuantAlgo.INT8,
            per_channel=True,
            per_token=True,
        ),
        "int4_awq": QuantConfig(
            quant_algo=QuantAlgo.INT4_AWQ,
            per_channel=True,
            group_size=128,
        ),
        "w8a16": QuantConfig(
            quant_algo=QuantAlgo.W8A16,
            per_channel=True,
        ),
    }
    return config_map.get(config_str, QuantConfig())
```

### 2.2 FP8量化

#### 2.2.1 原理

```
FP8格式（IEEE标准）：
- E4M3：4位指数，3位尾数（动态范围大）
- E5M2：5位指数，2位尾数（精度高）

优势：
- 接近FP16精度（精度损失<1%）
- 硬件原生支持（H100/H200）
- 不需要复杂校准
- 支持权重和激活量化

应用：
- 权重：E4M3格式（-448 to 448）
- 激活：E5M2格式（更大动态范围）
```

#### 2.2.2 实现

```python
def _quantize_fp8(model: PretrainedModel, quant_config: QuantConfig):
    """
    FP8量化实现
    """
    # 1. 设置量化模式
    model.config.quantization = quant_config
    quant_mode = QuantMode.from_quant_algo(QuantAlgo.FP8)
    
    # 2. 遍历所有线性层
    for name, module in model.named_modules():
        if isinstance(module, (Linear, ColumnLinear, RowLinear)):
            # 2.1 跳过排除的模块
            if any(exclude in name for exclude in quant_config.exclude_modules):
                continue
            
            # 2.2 转换为FP8量化层
            fp8_module = _convert_to_fp8_layer(module, quant_config)
            
            # 2.3 替换原模块
            parent = model
            attrs = name.split('.')
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, attrs[-1], fp8_module)
    
    # 3. 设置KV Cache量化
    if quant_config.kv_cache_quant_algo == QuantAlgo.FP8:
        _enable_fp8_kv_cache(model)
    
    return model

def _convert_to_fp8_layer(module, quant_config):
    """
    转换为FP8量化层
    """
    # 1. 创建FP8线性层
    fp8_layer = FP8Linear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        dtype=module.dtype,
        tp_group=getattr(module, 'tp_group', None),
        tp_size=getattr(module, 'tp_size', 1),
    )
    
    # 2. 转换权重为FP8
    with torch.no_grad():
        # 2.1 获取原始权重
        weight_fp16 = module.weight.data  # [out_features, in_features]
        
        # 2.2 计算缩放因子
        # FP8 E4M3 范围：[-448, 448]
        max_val = weight_fp16.abs().max()
        scale = 448.0 / max_val
        
        # 2.3 量化权重
        weight_fp8 = (weight_fp16 * scale).clamp(-448, 448)
        weight_fp8 = weight_fp8.to(torch.float8_e4m3fn)  # PyTorch 2.1+
        
        # 2.4 设置权重和缩放因子
        fp8_layer.weight.data = weight_fp8
        fp8_layer.weight_scale = Parameter(torch.tensor(scale))
        
        # 2.5 处理偏置
        if module.bias is not None:
            fp8_layer.bias.data = module.bias.data
    
    return fp8_layer

class FP8Linear(Module):
    """
    FP8量化线性层
    """
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()
        
        # FP8权重
        self.weight = Parameter(
            shape=(out_features, in_features),
            dtype=torch.float8_e4m3fn,  # FP8格式
        )
        
        # 权重缩放因子
        self.weight_scale = Parameter(
            shape=(1,),
            dtype=torch.float32,
        )
        
        # 激活缩放因子（动态计算）
        self.activation_scale = Parameter(
            shape=(1,),
            dtype=torch.float32,
        )
        
        if bias:
            self.bias = Parameter(
                shape=(out_features,),
                dtype=torch.float16,
            )
    
    def forward(self, input: Tensor) -> Tensor:
        # 1. 计算激活缩放因子（动态）
        input_max = input.abs().max()
        act_scale = 240.0 / input_max  # E5M2范围较大
        
        # 2. 量化激活
        input_fp8 = (input * act_scale).to(torch.float8_e5m2)
        
        # 3. FP8矩阵乘法
        # 现代GPU直接支持FP8 GEMM
        output_fp8 = torch.matmul(input_fp8, self.weight.T)
        
        # 4. 反量化到FP16
        output = output_fp8.to(torch.float16)
        output = output / (self.weight_scale * act_scale)
        
        # 5. 添加偏置
        if hasattr(self, 'bias'):
            output = output + self.bias
        
        return output
```

### 2.3 INT8 SmoothQuant

#### 2.3.1 原理

```
SmoothQuant核心思想：
1. 观察：激活值分布不均匀，有outlier
2. 解决：通过数学变换平滑激活分布
3. 公式：Y = (X diag(s)^(-1)) · (diag(s) W)
   其中s是平滑因子

步骤：
1. 统计激活值分布
2. 计算平滑因子s
3. 调整权重：W' = diag(s) * W  
4. 调整激活：X' = X * diag(s)^(-1)
5. 量化W'和X'

优势：
- 激活分布更均匀
- 量化误差更小
- 不需要复杂校准
```

#### 2.3.2 实现

```python
def _quantize_int8_smoothquant(model, quant_config, calib_dataset):
    """
    INT8 SmoothQuant量化
    """
    # 1. 收集激活统计信息
    act_scales = _collect_activation_scales(model, calib_dataset)
    
    # 2. 计算平滑因子
    smooth_scales = _compute_smooth_scales(model, act_scales, alpha=0.5)
    
    # 3. 应用平滑变换
    _apply_smooth_transform(model, smooth_scales)
    
    # 4. 量化权重和激活
    _quantize_weights_int8(model, quant_config)
    _setup_activation_quantization(model, quant_config)
    
    return model

def _collect_activation_scales(model, calib_dataset):
    """
    收集激活值范围统计
    """
    scales = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            # 记录激活值的最大绝对值
            if name not in scales:
                scales[name] = []
            scales[name].append(input[0].abs().max().item())
        return hook
    
    # 1. 注册hook
    for name, module in model.named_modules():
        if isinstance(module, (Linear, ColumnLinear, RowLinear)):
            handle = module.register_forward_hook(hook_fn(name))
            hooks.append(handle)
    
    # 2. 前向传播收集统计
    model.eval()
    with torch.no_grad():
        for batch in calib_dataset:
            input_ids = batch['input_ids']
            model(input_ids)
    
    # 3. 清理hook
    for handle in hooks:
        handle.remove()
    
    # 4. 计算平均scale
    final_scales = {}
    for name, scale_list in scales.items():
        final_scales[name] = np.mean(scale_list)
    
    return final_scales

def _compute_smooth_scales(model, act_scales, alpha=0.5):
    """
    计算平滑因子
    
    Args:
        alpha: 平滑强度（0.0-1.0）
               0.0：不平滑，1.0：完全平滑
    """
    smooth_scales = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (Linear, ColumnLinear, RowLinear)):
            # 1. 获取权重统计
            weight = module.weight.data  # [out_features, in_features]
            weight_scales = weight.abs().max(dim=0).values  # 按输入通道
            
            # 2. 获取激活统计
            act_scale = act_scales.get(name, 1.0)
            
            # 3. 计算平滑因子
            # s_j = (max_weight_j)^alpha / (max_activation_j)^alpha
            smooth_scale = (weight_scales ** alpha) / (act_scale ** alpha)
            
            # 4. 裁剪到合理范围
            smooth_scale = torch.clamp(smooth_scale, 0.1, 10.0)
            
            smooth_scales[name] = smooth_scale
    
    return smooth_scales

def _apply_smooth_transform(model, smooth_scales):
    """
    应用平滑变换
    """
    for name, module in model.named_modules():
        if name in smooth_scales:
            scale = smooth_scales[name]
            
            # 1. 调整权重：W' = diag(s) * W
            module.weight.data = module.weight.data * scale.unsqueeze(0)
            
            # 2. 调整对应的LayerNorm/RmsNorm（如果存在）
            # 需要找到前一个normalization层并调整其权重
            prev_norm = _find_previous_norm_layer(model, name)
            if prev_norm is not None:
                # norm_weight' = norm_weight / s
                prev_norm.weight.data = prev_norm.weight.data / scale

class INT8Linear(Module):
    """
    INT8量化线性层
    """
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        
        # 量化权重（INT8）
        self.qweight = Parameter(
            shape=(out_features, in_features),
            dtype=torch.int8,
        )
        
        # 权重量化参数
        self.weight_scale = Parameter(
            shape=(out_features, 1) if kwargs.get('per_channel') else (1,),
            dtype=torch.float32,
        )
        
        # 激活量化参数
        self.input_scale = Parameter(
            shape=(1,),
            dtype=torch.float32,
        )
    
    def forward(self, input: Tensor) -> Tensor:
        # 1. 动态量化激活
        input_scale = input.abs().max() / 127.0
        input_int8 = torch.round(input / input_scale).clamp(-128, 127).to(torch.int8)
        
        # 2. INT8矩阵乘法
        output_int32 = torch.matmul(input_int8.to(torch.int32), self.qweight.T.to(torch.int32))
        
        # 3. 反量化
        output = output_int32.to(torch.float32) * (input_scale * self.weight_scale)
        
        return output.to(torch.float16)
```

### 2.4 INT4 AWQ量化

#### 2.4.1 原理

```
AWQ (Activation-aware Weight Quantization)：
1. 观察：不同权重通道的重要性不同
2. 策略：保护重要通道，积极量化不重要通道
3. 方法：基于激活值幅度确定通道重要性

核心算法：
1. 收集激活统计：A = {a1, a2, ..., an}
2. 计算重要性：importance_j = mean(|a_j|)
3. 计算缩放因子：s_j = importance_j^(-α)
4. 应用缩放：W' = diag(s) * W, X' = X * diag(s)^(-1)
5. 量化W'为INT4

分组量化：
- 将权重按组量化（如128个权重一组）
- 每组独立计算量化参数
- 减少量化误差
```

#### 2.4.2 实现

```python
def _quantize_int4_awq(model, quant_config, calib_dataset):
    """
    INT4 AWQ量化
    """
    # 1. 收集激活统计
    act_stats = _collect_activation_statistics(model, calib_dataset)
    
    # 2. 计算AWQ缩放因子
    awq_scales = _compute_awq_scales(model, act_stats, alpha=0.5)
    
    # 3. 应用AWQ变换
    _apply_awq_transform(model, awq_scales)
    
    # 4. 分组量化权重
    _quantize_weights_int4_grouped(model, quant_config)
    
    return model

def _compute_awq_scales(model, act_stats, alpha=0.5):
    """
    计算AWQ缩放因子
    """
    awq_scales = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (Linear, ColumnLinear, RowLinear)):
            # 1. 获取激活统计
            if name in act_stats:
                act_mean = act_stats[name]['mean']  # 每个通道的平均激活值
                
                # 2. 计算重要性（激活值越大越重要）
                importance = act_mean
                
                # 3. 计算缩放因子
                # 重要通道缩放因子小（保护），不重要通道缩放因子大
                scale = importance ** (-alpha)
                scale = scale / scale.mean()  # 归一化
                
                awq_scales[name] = scale
    
    return awq_scales

def _quantize_weights_int4_grouped(model, quant_config):
    """
    分组INT4权重量化
    """
    group_size = quant_config.group_size  # 默认128
    
    for name, module in model.named_modules():
        if isinstance(module, (Linear, ColumnLinear, RowLinear)):
            # 1. 获取权重
            weight = module.weight.data  # [out_features, in_features]
            out_features, in_features = weight.shape
            
            # 2. 按组量化
            # 将in_features维度按group_size分组
            num_groups = (in_features + group_size - 1) // group_size
            
            qweight = torch.zeros(out_features, in_features // 2, dtype=torch.uint8)  # pack 2个4位数字
            scales = torch.zeros(out_features, num_groups, dtype=torch.float16)
            zeros = torch.zeros(out_features, num_groups, dtype=torch.float16)
            
            for g in range(num_groups):
                start_idx = g * group_size
                end_idx = min((g + 1) * group_size, in_features)
                
                # 2.1 获取当前组权重
                group_weight = weight[:, start_idx:end_idx]  # [out_features, group_size]
                
                # 2.2 计算量化参数（per channel）
                group_min = group_weight.min(dim=1, keepdim=True).values
                group_max = group_weight.max(dim=1, keepdim=True).values
                
                # INT4范围：0-15
                scale = (group_max - group_min) / 15.0
                zero_point = -group_min / scale
                
                # 2.3 量化
                qweight_group = torch.round(group_weight / scale + zero_point).clamp(0, 15)
                
                # 2.4 打包（2个4位数字打包成1个8位）
                if end_idx - start_idx == group_size:  # 完整组
                    packed = qweight_group[:, ::2] + (qweight_group[:, 1::2] << 4)
                    qweight[:, start_idx//2:end_idx//2] = packed.to(torch.uint8)
                
                # 2.5 保存量化参数
                scales[:, g] = scale.squeeze()
                zeros[:, g] = zero_point.squeeze()
            
            # 3. 替换为INT4量化层
            int4_layer = INT4Linear(
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
            )
            int4_layer.qweight.data = qweight
            int4_layer.scales.data = scales
            int4_layer.zeros.data = zeros
            
            # 4. 替换原层
            _replace_module(model, name, int4_layer)

class INT4Linear(Module):
    """
    INT4分组量化线性层
    """
    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        
        num_groups = (in_features + group_size - 1) // group_size
        
        # 打包的INT4权重（2个4位打包成1个8位）
        self.qweight = Parameter(
            shape=(out_features, in_features // 2),
            dtype=torch.uint8,
        )
        
        # 每组的量化参数
        self.scales = Parameter(
            shape=(out_features, num_groups),
            dtype=torch.float16,
        )
        
        self.zeros = Parameter(
            shape=(out_features, num_groups),
            dtype=torch.float16,
        )
        
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, input: Tensor) -> Tensor:
        # 使用自定义CUDA kernel进行INT4矩阵乘法
        return int4_linear_forward(
            input, 
            self.qweight, 
            self.scales, 
            self.zeros, 
            self.group_size
        )

def int4_linear_forward(input, qweight, scales, zeros, group_size):
    """
    INT4线性层前向传播（调用CUDA kernel）
    """
    # 1. 解包INT4权重
    batch_size, seq_len, in_features = input.shape
    out_features = qweight.shape[0]
    
    # 2. 调用优化的CUDA kernel
    # 实际实现中会使用C++/CUDA编写的高效kernel
    output = torch.empty(
        batch_size, seq_len, out_features,
        dtype=input.dtype, device=input.device
    )
    
    # 伪代码：实际使用CUDA kernel
    # cutlass_int4_gemm(input, qweight, scales, zeros, output, group_size)
    
    return output
```

## 三、关键功能深度剖析

### 3.1 量化精度对比

#### 3.1.1 数值范围对比

```python
# 不同精度的数值范围
precisions = {
    "FP16": {
        "range": "±65504",
        "precision": "~4位有效数字",
        "memory": "2 bytes",
    },
    "FP8 E4M3": {
        "range": "±448", 
        "precision": "~2位有效数字",
        "memory": "1 byte",
    },
    "FP8 E5M2": {
        "range": "±57344",
        "precision": "~1.5位有效数字", 
        "memory": "1 byte",
    },
    "INT8": {
        "range": "-128 to 127",
        "precision": "整数",
        "memory": "1 byte",
    },
    "INT4": {
        "range": "0 to 15 (unsigned)",
        "precision": "整数",
        "memory": "0.5 bytes",
    }
}
```

#### 3.1.2 精度损失分析

```python
def analyze_quantization_error(original_weights, quantized_weights):
    """
    量化误差分析
    """
    # 1. 均方误差
    mse = torch.mean((original_weights - quantized_weights) ** 2)
    
    # 2. 信噪比
    signal_power = torch.mean(original_weights ** 2)
    noise_power = mse
    snr_db = 10 * torch.log10(signal_power / noise_power)
    
    # 3. 相对误差
    relative_error = torch.abs(original_weights - quantized_weights) / torch.abs(original_weights)
    mean_relative_error = torch.mean(relative_error)
    
    return {
        "mse": mse.item(),
        "snr_db": snr_db.item(),
        "mean_relative_error": mean_relative_error.item(),
    }

# 典型结果：
# FP8: SNR ~40dB, 相对误差 ~1%
# INT8 SmoothQuant: SNR ~35dB, 相对误差 ~2%
# INT4 AWQ: SNR ~25dB, 相对误差 ~5%
```

### 3.2 KV Cache量化

#### 3.2.1 原理

```
KV Cache特点：
1. 占用内存大（长序列时占主导）
2. 数值范围相对稳定
3. 对量化不敏感

量化策略：
1. 按token动态量化
2. 按head独立量化
3. 使用INT8或FP8格式

内存节省：
- FP16 KV Cache: 2 bytes/element
- INT8 KV Cache: 1 byte/element  
- 节省50%内存
```

#### 3.2.2 实现

```python
class QuantizedKVCache(Module):
    """
    量化KV Cache
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int, 
        head_size: int,
        max_seq_len: int,
        dtype: str = "int8",
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        
        # 量化的KV Cache存储
        if dtype == "int8":
            cache_dtype = torch.int8
            self.qkv_cache = torch.zeros(
                num_layers, 2, max_seq_len, num_heads, head_size,
                dtype=cache_dtype
            )
            
            # 量化参数（每个head独立）
            self.kv_scales = torch.ones(
                num_layers, 2, num_heads,
                dtype=torch.float32
            )
    
    def store_kv(self, layer_idx: int, key: Tensor, value: Tensor, seq_pos: int):
        """
        存储量化的KV
        """
        # 1. 计算量化参数
        k_scale = key.abs().max() / 127.0
        v_scale = value.abs().max() / 127.0
        
        # 2. 量化
        k_quantized = torch.round(key / k_scale).clamp(-128, 127).to(torch.int8)
        v_quantized = torch.round(value / v_scale).clamp(-128, 127).to(torch.int8)
        
        # 3. 存储
        self.qkv_cache[layer_idx, 0, seq_pos] = k_quantized
        self.qkv_cache[layer_idx, 1, seq_pos] = v_quantized
        
        # 4. 保存量化参数
        self.kv_scales[layer_idx, 0] = k_scale
        self.kv_scales[layer_idx, 1] = v_scale
    
    def get_kv(self, layer_idx: int, seq_len: int) -> Tuple[Tensor, Tensor]:
        """
        获取反量化的KV
        """
        # 1. 获取量化数据
        k_quantized = self.qkv_cache[layer_idx, 0, :seq_len]  # [seq_len, num_heads, head_size]
        v_quantized = self.qkv_cache[layer_idx, 1, :seq_len]
        
        # 2. 反量化
        k_scale = self.kv_scales[layer_idx, 0]
        v_scale = self.kv_scales[layer_idx, 1]
        
        key = k_quantized.to(torch.float16) * k_scale
        value = v_quantized.to(torch.float16) * v_scale
        
        return key, value
```

### 3.3 量化感知训练 vs 训练后量化

#### 3.3.1 对比

| 方法 | 优势 | 劣势 | 适用场景 |
|-----|------|------|---------|
| **训练后量化(PTQ)** | 简单、快速、无需重训练 | 精度损失较大 | 对精度要求不高 |
| **量化感知训练(QAT)** | 精度损失小、效果好 | 需要重训练、时间长 | 对精度要求高 |

#### 3.3.2 QAT实现示例

```python
class QATLinear(Module):
    """
    量化感知训练线性层
    """
    def __init__(self, in_features, out_features, quant_bits=8):
        super().__init__()
        
        self.weight = Parameter(torch.randn(out_features, in_features))
        self.quant_bits = quant_bits
        
        # 可学习的量化参数
        self.weight_scale = Parameter(torch.ones(1))
        self.weight_zero_point = Parameter(torch.zeros(1))
    
    def quantize_weight(self, weight):
        """
        伪量化（训练时模拟量化过程）
        """
        # 1. 计算量化范围
        qmin = 0
        qmax = 2 ** self.quant_bits - 1
        
        # 2. 量化
        scale = self.weight_scale
        zero_point = self.weight_zero_point
        
        # 3. 伪量化（前向量化，反向保持FP32梯度）
        qweight = torch.round(weight / scale + zero_point).clamp(qmin, qmax)
        dequant_weight = (qweight - zero_point) * scale
        
        # 4. 直通估计器（Straight Through Estimator）
        # 前向使用量化值，反向传播原始梯度
        return weight + (dequant_weight - weight).detach()
    
    def forward(self, input):
        # 训练时使用伪量化
        if self.training:
            qweight = self.quantize_weight(self.weight)
        else:
            # 推理时使用真实量化
            qweight = self.real_quantize(self.weight)
        
        return torch.matmul(input, qweight.T)
```

## 四、使用示例

### 4.1 FP8量化部署

```python
from tensorrt_llm import LLM
from tensorrt_llm.quantization import quantize

# 1. 加载模型
llm = LLM("meta-llama/Llama-3-8B")

# 2. FP8量化（无需校准数据）
quantized_model = quantize(llm.model, "fp8")

# 3. 构建引擎
llm.model = quantized_model
engine = llm._build_engine()

# 4. 推理
output = llm.generate("Hello world", max_tokens=100)

# 性能提升：
# - 内存使用：~50%减少
# - 推理速度：1.5-2x加速（H100）
# - 精度损失：<1%
```

### 4.2 INT4 AWQ量化

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. 准备校准数据集
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

calib_dataset = dataset.map(preprocess, batched=True).select(range(128))

# 2. INT4 AWQ量化
llm = LLM("meta-llama/Llama-3-8B")
quantized_model = quantize(
    llm.model, 
    quant_config="int4_awq",
    calib_dataset=calib_dataset,
    calib_size=128,
)

# 3. 构建引擎
llm.model = quantized_model
engine = llm._build_engine()

# 性能提升：
# - 内存使用：~75%减少
# - 推理速度：2-3x加速
# - 精度损失：~5%
```

### 4.3 混合精度量化

```python
from tensorrt_llm.quantization import QuantConfig, QuantAlgo

# 自定义量化配置
quant_config = QuantConfig(
    # 权重INT4，激活FP16
    quant_algo=QuantAlgo.INT4_AWQ,
    
    # KV Cache INT8
    kv_cache_quant_algo=QuantAlgo.INT8,
    
    # 排除敏感层
    exclude_modules=["lm_head"],
    
    # 分组量化
    group_size=128,
    per_channel=True,
)

# 应用量化
quantized_model = quantize(llm.model, quant_config, calib_dataset)
```

### 4.4 量化效果评估

```python
def evaluate_quantization_quality(original_model, quantized_model, test_dataset):
    """
    评估量化效果
    """
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    original_outputs = []
    quantized_outputs = []
    
    # 1. 收集输出
    for batch in test_dataset:
        with torch.no_grad():
            orig_out = original_model(batch["input_ids"])
            quant_out = quantized_model(batch["input_ids"])
            
            original_outputs.append(orig_out.logits.cpu().numpy())
            quantized_outputs.append(quant_out.logits.cpu().numpy())
    
    # 2. 计算指标
    orig_concat = np.concatenate(original_outputs, axis=0)
    quant_concat = np.concatenate(quantized_outputs, axis=0)
    
    # 均方误差
    mse = mean_squared_error(orig_concat.flatten(), quant_concat.flatten())
    
    # 余弦相似度
    cosine_sim = np.dot(orig_concat.flatten(), quant_concat.flatten()) / \
                (np.linalg.norm(orig_concat.flatten()) * np.linalg.norm(quant_concat.flatten()))
    
    # Top-1准确率差异
    orig_pred = np.argmax(orig_concat, axis=-1)
    quant_pred = np.argmax(quant_concat, axis=-1)
    top1_accuracy = np.mean(orig_pred == quant_pred)
    
    return {
        "mse": mse,
        "cosine_similarity": cosine_sim,
        "top1_accuracy": top1_accuracy,
    }

# 使用示例
results = evaluate_quantization_quality(original_model, quantized_model, test_dataset)
print(f"MSE: {results['mse']:.6f}")
print(f"Cosine Similarity: {results['cosine_similarity']:.4f}")
print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
```

## 五、性能优化建议

### 5.1 量化算法选择

```python
# 根据硬件和需求选择量化算法

def select_quantization_strategy(model_size, hardware, precision_requirement):
    """
    量化策略选择指南
    """
    strategies = {
        # 高精度场景
        "high_precision": {
            "H100/H200": "fp8",           # 原生支持，速度快
            "A100": "int8_smoothquant",   # 通用，精度好
            "V100": "w8a16",              # 权重量化，激活保持FP16
        },
        
        # 内存受限场景
        "memory_constrained": {
            "A100+": "int4_awq",          # 激活感知，精度相对好
            "V100+": "int4_gptq",         # 通用，压缩比高
        },
        
        # 超大模型场景
        "large_model": {
            "multi_gpu": "int4_awq",      # 显存节省明显
            "single_gpu": "int8_smoothquant", # 平衡精度和速度
        }
    }
    
    return strategies.get(precision_requirement, {}).get(hardware, "int8_smoothquant")

# 使用示例
strategy = select_quantization_strategy(
    model_size="70B",
    hardware="A100", 
    precision_requirement="memory_constrained"
)
print(f"Recommended strategy: {strategy}")
```

### 5.2 校准数据集优化

```python
def create_optimal_calib_dataset(model_name, domain="general"):
    """
    创建最优校准数据集
    """
    datasets = {
        "general": [
            "wikitext-2-raw-v1",
            "c4",
            "openwebtext",
        ],
        "code": [
            "codeparrot/github-code",
            "bigcode/the-stack",
        ],
        "math": [
            "hendrycks/math",
            "gsm8k",
        ],
        "chat": [
            "alpaca",
            "sharegpt",
        ]
    }
    
    # 组合多个数据集
    combined_dataset = []
    for dataset_name in datasets[domain]:
        dataset = load_dataset(dataset_name, split="train")
        # 采样均匀分布的样本
        sampled = dataset.shuffle(seed=42).select(range(128))
        combined_dataset.extend(sampled)
    
    return combined_dataset

# 关键原则：
# 1. 多样性：覆盖不同类型文本
# 2. 代表性：反映实际使用场景
# 3. 长度分布：包含不同长度的样本
# 4. 数量适中：128-512样本通常足够
```

### 5.3 量化后优化

```python
def optimize_quantized_model(quantized_model):
    """
    量化后模型优化
    """
    # 1. 层融合
    fused_model = fuse_quantized_layers(quantized_model)
    
    # 2. 常量折叠
    folded_model = fold_quantization_constants(fused_model)
    
    # 3. 死代码消除
    optimized_model = eliminate_unused_quantization_ops(folded_model)
    
    return optimized_model

def fuse_quantized_layers(model):
    """
    融合量化层
    
    例如：QuantLinear + QuantReLU → QuantLinearReLU
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # 查找后续的激活函数
            next_module = get_next_module(model, name)
            if isinstance(next_module, QuantReLU):
                # 融合为单个kernel
                fused = QuantLinearReLU(module, next_module)
                replace_module(model, name, fused)
    
    return model
```

## 六、常见问题

**Q1：FP8量化需要什么硬件？**
- 需要H100或H200 GPU
- 支持FP8 Tensor Core
- 其他GPU可以模拟FP8但无加速效果

**Q2：INT4量化为什么需要校准数据集？**
- INT4动态范围小，需要精确的量化参数
- 校准数据帮助统计权重和激活分布
- AWQ/GPTQ算法需要激活统计信息

**Q3：量化后精度损失如何评估？**
```python
# 标准评估流程
metrics = [
    "perplexity",      # 困惑度（语言模型）
    "bleu_score",      # BLEU分数（生成任务）
    "accuracy",        # 准确率（分类任务）
    "cosine_similarity" # 输出相似度
]

# 可接受的精度损失：
# FP8: <1%
# INT8: <3%  
# INT4: <8%
```

**Q4：如何选择group_size？**
- 较小group_size（64-128）：精度好，开销大
- 较大group_size（256-512）：精度差，开销小
- 推荐：128（平衡精度和性能）

**Q5：量化模型如何微调？**
```python
# QLoRA：量化模型+LoRA微调
quantized_model = quantize(base_model, "int4_awq")
lora_model = add_lora_adapters(quantized_model, rank=16)

# 只训练LoRA参数，量化权重冻结
for param in quantized_model.parameters():
    param.requires_grad = False
for param in lora_model.parameters():
    param.requires_grad = True
```

---

**文档版本：** 2.0（深度剖析版）  
**生成时间：** 2025-10-05  
**对应代码版本：** TensorRT-LLM v1.2.0rc1
