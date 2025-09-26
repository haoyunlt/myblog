---
title: "TensorRT-LLM 框架使用手册"
date: 2024-12-19T10:00:00+08:00
draft: false
tags: ['TensorRT-LLM', 'NVIDIA', '推理优化', '深度学习']
categories: ["tensorrtllm", "技术分析"]
description: "深入分析 TensorRT-LLM 框架使用手册 的技术实现和架构设计"
weight: 580
slug: "TensorRTLLM-源码剖析_框架使用手册"
---

# TensorRT-LLM 框架使用手册

## 概述

TensorRT-LLM 是 NVIDIA 开源的大语言模型推理优化库，基于 PyTorch 架构，提供高性能的 LLM 推理能力。该框架支持从单 GPU 到多 GPU、多节点的部署，内置多种并行策略和高级优化特性。

## 核心特性

### 🔥 基于 PyTorch 架构
- 高级 Python LLM API，支持广泛的推理设置
- 内置多种并行策略支持
- 与 NVIDIA Dynamo 和 Triton Inference Server 无缝集成

### ⚡ 顶级性能
- 在最新 NVIDIA GPU 上提供突破性性能
- DeepSeek R1：在 Blackwell GPU 上创世界纪录推理性能
- Llama 4 Maverick：在 B200 GPU 上突破 1,000 TPS/用户屏障

### 🎯 全面模型支持
- 支持最新和最流行的 LLM 架构
- FP4 格式支持（NVIDIA B200 GPU）
- 自动利用优化的 FP4 内核

## 安装指南

### 系统要求
- Python 3.10-3.12
- CUDA 13.0.0+
- TensorRT 10.13.2+
- PyTorch 2.0+

### 安装方式

#### 1. 通过 pip 安装（推荐）
```bash
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
```

#### 2. 从源码构建
```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
pip install -e .
```

## 快速开始

### 基础使用示例

```python
from tensorrt_llm import LLM

# 初始化 LLM
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 生成文本
output = llm.generate("Hello, my name is")
print(output)
```

### 高级配置示例

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig, SamplingParams

# 构建配置
build_config = BuildConfig(
    max_batch_size=8,
    max_input_len=256,
    max_seq_len=512,
    max_beam_width=4
)

# KV 缓存配置
kv_cache_config = KvCacheConfig(
    free_gpu_memory_fraction=0.9,
    enable_block_reuse=True
)

# 初始化 LLM
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,
    build_config=build_config,
    kv_cache_config=kv_cache_config
)

# 采样参数
sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.8,
    top_p=0.9
)

# 生成
outputs = llm.generate(
    ["Tell me about artificial intelligence"],
    sampling_params=sampling_params
)
```

### 完整生产环境示例

```python
import torch
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (
    BuildConfig, KvCacheConfig, SamplingParams,
    QuantConfig, QuantAlgo, LoRARequest
)

class ProductionLLMService:
    """生产环境 LLM 服务封装"""

    def __init__(self, model_path: str, config: dict):
        """
        初始化生产级 LLM 服务

        Args:
            model_path: 模型路径
            config: 配置字典
        """
        self.model_path = model_path
        self.config = config
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM 实例"""

        # 构建配置
        build_config = BuildConfig(
            max_batch_size=self.config.get('max_batch_size', 16),
            max_input_len=self.config.get('max_input_len', 2048),
            max_seq_len=self.config.get('max_seq_len', 4096),
            max_beam_width=self.config.get('max_beam_width', 1),
            strongly_typed=True,
            use_refit=True,  # 支持权重更新
            weight_streaming=self.config.get('weight_streaming', False)
        )

        # KV 缓存配置
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=self.config.get('kv_cache_fraction', 0.85),
            enable_block_reuse=True,
            max_tokens_in_paged_kv_cache=None  # 自动计算
        )

        # 量化配置
        quant_config = None
        if self.config.get('enable_quantization', False):
            quant_config = QuantConfig(
                quant_algo=QuantAlgo.FP8,
                kv_cache_quant_algo=QuantAlgo.INT8,
                group_size=128
            )

        # 初始化 LLM
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.config.get('tensor_parallel_size', 1),
            pipeline_parallel_size=self.config.get('pipeline_parallel_size', 1),
            build_config=build_config,
            kv_cache_config=kv_cache_config,
            quant_config=quant_config,
            trust_remote_code=True
        )

    def generate_text(self,
                     prompts: list,
                     max_tokens: int = 100,
                     temperature: float = 0.8,
                     top_p: float = 0.9,
                     lora_request: LoRARequest = None) -> list:
        """
        生成文本

        Args:
            prompts: 输入提示列表
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p 采样参数
            lora_request: LoRA 请求（可选）

        Returns:
            生成结果列表
        """

        # 采样参数
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            length_penalty=1.0
        )

        try:
            # 执行生成
            outputs = self.llm.generate(
                prompts,
                sampling_params=sampling_params,
                lora_request=lora_request
            )

            # 提取生成文本
            results = []
            for output in outputs:
                generated_text = output.outputs[0].text
                results.append({
                    'prompt': output.prompt,
                    'generated_text': generated_text,
                    'finish_reason': output.outputs[0].finish_reason,
                    'token_ids': output.outputs[0].token_ids
                })

            return results

        except Exception as e:
            print(f"生成失败: {e}")
            return []

    def generate_streaming(self, prompt: str, **kwargs):
        """流式生成"""
        sampling_params = SamplingParams(**kwargs)

        for output in self.llm.generate(
            prompt,
            sampling_params=sampling_params,
            streaming=True
        ):
            yield output.outputs[0].text

    def shutdown(self):
        """优雅关闭"""
        if self.llm:
            self.llm.shutdown()

# 使用示例
config = {
    'max_batch_size': 8,
    'max_input_len': 1024,
    'max_seq_len': 2048,
    'tensor_parallel_size': 2,
    'enable_quantization': True,
    'kv_cache_fraction': 0.8
}

service = ProductionLLMService("meta-llama/Llama-2-7b-hf", config)

# 批量生成
prompts = [
    "解释人工智能的基本概念",
    "描述深度学习的工作原理",
    "什么是大语言模型？"
]

results = service.generate_text(
    prompts,
    max_tokens=150,
    temperature=0.7
)

for result in results:
    print(f"输入: {result['prompt']}")
    print(f"输出: {result['generated_text']}")
    print("-" * 50)
```

### 多模态使用示例

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import MultimodalEncoder

# 初始化多模态编码器
mm_encoder = MultimodalEncoder(
    vision_encoder_path="/path/to/vision/encoder"
)

# 多模态 LLM
llm = LLM(
    model="llava-v1.5-7b",
    multimodal_encoder=mm_encoder
)

# 图像+文本输入
from PIL import Image
image = Image.open("example.jpg")
text_prompt = "描述这张图片中的内容"

output = llm.generate(
    inputs={"text": text_prompt, "image": image},
    sampling_params=SamplingParams(max_tokens=200)
)
```

## 命令行工具

TensorRT-LLM 提供了多个命令行工具：

### 1. trtllm-build - 构建引擎
```bash
trtllm-build --model_dir /path/to/model \
             --output_dir /path/to/engine \
             --max_batch_size 8 \
             --max_input_len 1024 \
             --max_seq_len 2048
```

### 2. trtllm-serve - 启动服务
```bash
trtllm-serve --model /path/to/model \
             --host 0.0.0.0 \
             --port 8000 \
             --tp_size 2
```

### 3. trtllm-bench - 性能测试
```bash
trtllm-bench --model /path/to/model \
             --batch_size 8 \
             --input_len 128 \
             --output_len 128
```

### 4. trtllm-eval - 模型评估
```bash
trtllm-eval --model /path/to/model \
            --backend tensorrt \
            --tp_size 2
```

## 配置参数详解

### BuildConfig 参数
- `max_input_len`: 最大输入长度（默认 1024）
- `max_seq_len`: 最大序列长度（默认 None，自动推导）
- `max_batch_size`: 最大批次大小（默认 2048）
- `max_beam_width`: 最大束搜索宽度（默认 1）
- `max_num_tokens`: 最大批次 token 数（默认 8192）

### KvCacheConfig 参数
- `free_gpu_memory_fraction`: KV 缓存使用的 GPU 内存比例（默认 0.9）
- `enable_block_reuse`: 启用块重用优化（默认 True）

### SamplingParams 参数
- `max_tokens`: 最大生成 token 数
- `temperature`: 温度参数（默认 1.0）
- `top_p`: Top-p 采样（默认 1.0）
- `top_k`: Top-k 采样（默认 0）
- `beam_width`: 束搜索宽度（默认 1）

## 并行策略

### 张量并行（Tensor Parallelism）
```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8  # 8 GPU 张量并行
)
```

### 流水线并行（Pipeline Parallelism）
```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    pipeline_parallel_size=2  # 2 阶段流水线
)
```

### 专家并行（Expert Parallelism）
```python
llm = LLM(
    model="mixtral-8x7b",
    tensor_parallel_size=2,
    moe_expert_parallel_size=4  # MoE 专家并行
)
```

## 量化支持

### FP8 量化
```python
from tensorrt_llm.quantization import QuantConfig, QuantAlgo

quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
llm = LLM(model="meta-llama/Llama-2-7b-hf", quant_config=quant_config)
```

### INT4 AWQ 量化
```python
quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
llm = LLM(model="meta-llama/Llama-2-7b-hf", quant_config=quant_config)
```

### FP4 量化（B200 GPU）
```python
quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
llm = LLM(model="meta-llama/Llama-2-7b-hf", quant_config=quant_config)
```

## 高级特性

### 投机解码（Speculative Decoding）
```python
from tensorrt_llm.llmapi import SpeculativeDecodingConfig

spec_config = SpeculativeDecodingConfig(
    draft_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    draft_tokens=4
)

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    speculative_decoding_config=spec_config
)
```

### LoRA 适配器
```python
from tensorrt_llm.executor import LoRARequest

lora_request = LoRARequest(
    lora_name="math_lora",
    lora_path="/path/to/lora/adapter"
)

outputs = llm.generate(
    ["Solve this equation: 2x + 3 = 7"],
    lora_request=lora_request
)
```

### 多模态支持
```python
from tensorrt_llm.llmapi import MultimodalEncoder

# 初始化多模态编码器
mm_encoder = MultimodalEncoder(
    vision_encoder_path="/path/to/vision/encoder"
)

# 多模态 LLM
llm = LLM(
    model="llava-v1.5-7b",
    multimodal_encoder=mm_encoder
)
```

## 性能优化建议

### 1. 内存优化
- 使用 KV 缓存块重用：`enable_block_reuse=True`
- 调整 GPU 内存分配：`free_gpu_memory_fraction=0.9`
- 启用权重流式传输：`weight_streaming=True`

### 2. 计算优化
- 使用 CUDA 图：`enable_cuda_graph=True`
- 启用融合注意力：`use_fused_attention=True`
- 优化批次大小：根据 GPU 内存调整 `max_batch_size`

### 3. 并行优化
- 合理选择张量并行度：通常为 GPU 数量的因子
- 流水线并行适用于大模型：减少单卡内存压力
- MoE 模型使用专家并行：提高专家利用率

## 故障排除

### 常见问题

#### 1. 内存不足
```bash
# 解决方案：减少批次大小或序列长度
trtllm-build --max_batch_size 4 --max_seq_len 1024
```

#### 2. 编译失败
```bash
# 检查 CUDA 和 TensorRT 版本
nvidia-smi
python -c "import tensorrt; print(tensorrt.__version__)"
```

#### 3. 性能不佳
```bash
# 使用性能分析工具
trtllm-bench --model /path/to/model --profile
```

### 调试模式
```python
import os
os.environ["TRTLLM_LOG_LEVEL"] = "DEBUG"

from tensorrt_llm import LLM
llm = LLM(model="meta-llama/Llama-2-7b-hf")
```

## 最佳实践

### 1. 模型选择
- 根据硬件选择合适的模型大小
- 考虑量化以减少内存使用
- 使用预优化的模型检查点

### 2. 配置优化
- 根据实际需求设置序列长度
- 合理配置并行策略
- 启用适当的优化特性

### 3. 部署建议
- 使用容器化部署
- 配置适当的监控和日志
- 实施负载均衡和故障转移

## 社区和支持

- **GitHub**: https://github.com/NVIDIA/TensorRT-LLM
- **文档**: https://nvidia.github.io/TensorRT-LLM/
- **讨论群**: 微信讨论群（见 GitHub Issues #5359）
- **技术博客**: NVIDIA 开发者博客

## 版本信息

当前版本：1.1.0rc6
- 支持 Python 3.10-3.12
- 兼容 CUDA 13.0.0+
- 需要 TensorRT 10.13.2+

---

*本手册基于 TensorRT-LLM 最新版本编写，如有更新请参考官方文档。*
