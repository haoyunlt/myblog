---
title: "vLLM-99-最佳实践与案例"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 最佳实践
  - 实战经验
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - vLLM-99-最佳实践与案例"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# vLLM-99-最佳实践与案例

## 框架使用示例

### 1. 基础离线推理

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.9,
)

# 准备 prompts
prompts = [
    "Hello, my name is",
    "The future of AI is",
]

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

# 批量推理
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")
```

**要点说明**：
- `tensor_parallel_size`：根据 GPU 数量设置，单卡为 1
- `dtype`：FP16 或 BF16 提供最佳性能
- `gpu_memory_utilization`：默认 0.9，可根据 OOM 情况调整
- `generate()` 会自动进行 continuous batching

### 2. 在线服务部署

#### 2.1 启动 API Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --max-model-len 4096 \
    --port 8000
```

#### 2.2 客户端调用

```python
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

completion = openai.ChatCompletion.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ],
    temperature=0.7,
    max_tokens=50,
    stream=True,  # 流式输出
)

for chunk in completion:
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**要点说明**：
- `stream=True` 启用流式输出，降低首 token 延迟
- API Server 自动处理并发请求
- 兼容 OpenAI API，方便迁移

### 3. 多模态推理

```python
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# 图像 + 文本输入
prompt = {
    "prompt": "USER: <image>\nWhat's in this image? ASSISTANT:",
    "multi_modal_data": {
        "image": "https://example.com/image.jpg"
    },
}

outputs = llm.generate([prompt], SamplingParams(temperature=0.2, max_tokens=50))
print(outputs[0].outputs[0].text)
```

**要点说明**：
- 支持图像、视频、音频等多模态输入
- 自动下载和处理多模态数据
- 支持本地文件和 URL

### 4. 量化推理

#### 4.1 AWQ 量化

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="float16",
)
```

#### 4.2 GPTQ 量化

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16",
)
```

#### 4.3 FP8 量化

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8",
    dtype="float16",
)
```

**要点说明**：
- AWQ/GPTQ：INT4 权重量化，降低内存占用 ~75%
- FP8：权重和 KV 缓存量化，平衡性能和精度
- 量化模型推理速度提升 20-50%

### 5. LoRA 推理

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_lora_rank=64,
    max_loras=4,
)

# 使用不同的 LoRA 适配器
lora_request_1 = LoRARequest("lora1", 1, "/path/to/lora1")
lora_request_2 = LoRARequest("lora2", 2, "/path/to/lora2")

outputs1 = llm.generate("Hello", lora_request=lora_request_1)
outputs2 = llm.generate("Hello", lora_request=lora_request_2)
```

**要点说明**：
- 支持动态切换 LoRA 适配器
- `max_loras`：同时加载的 LoRA 数量
- LoRA 推理开销 <5%

### 6. 分布式推理

#### 6.1 张量并行

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4 卡张量并行
)
```

#### 6.2 流水线并行

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    pipeline_parallel_size=2,  # 2 卡流水线并行
    tensor_parallel_size=2,    # 每个 pipeline stage 2 卡张量并行
)
```

**要点说明**：
- 张量并行：适合单机多卡，通信开销小
- 流水线并行：适合跨机场景，容忍高延迟
- 总 GPU 数 = `tensor_parallel_size × pipeline_parallel_size`

## 实战经验

### 1. 性能调优

#### 1.1 内存优化

**问题**：频繁 OOM
**方案**：
```python
llm = LLM(
    model="...",
    gpu_memory_utilization=0.85,  # 降低内存利用率
    swap_space=4,                  # 增大 swap 空间
    max_num_batched_tokens=4096,  # 减小批次大小
)
```

**效果**：内存使用下降 10-15%，吞吐量下降 <5%

#### 1.2 吞吐量优化

**问题**：吞吐量不理想
**方案**：
```python
llm = LLM(
    model="...",
    max_num_seqs=128,              # 增大并发度
    max_num_batched_tokens=8192,   # 增大批次
    enable_prefix_caching=True,    # 启用前缀缓存
)
```

**效果**：吞吐量提升 30-50%（取决于请求相似度）

#### 1.3 延迟优化

**问题**：首 token 延迟（TTFT）过高
**方案**：
```python
llm = LLM(
    model="...",
    enable_prefix_caching=True,    # 缓存常见 prompt
    enable_chunked_prefill=True,   # 启用 chunked prefill
)
```

**效果**：TTFT 降低 40-60%（缓存命中时）

### 2. 部署策略

#### 2.1 单模型服务

**场景**：单一模型，高并发
**配置**：
- 单实例，多卡张量并行
- `max_num_seqs=256`，充分利用 GPU
- 启用 Prefix Caching

**适用**：通用对话、代码生成

#### 2.2 多模型服务

**场景**：多个模型，中等并发
**配置**：
- 每个模型独立实例
- 负载均衡（Nginx/Envoy）
- 根据模型大小分配 GPU

**适用**：多租户场景

#### 2.3 LoRA 多租户

**场景**：基座模型 + 多个 LoRA
**配置**：
- 单实例，启用 LoRA 支持
- `max_loras=8-16`
- 动态加载 LoRA

**适用**：个性化服务

### 3. 监控与故障排查

#### 3.1 关键指标

**吞吐量**：
- `tokens_per_second`：整体吞吐
- `requests_per_second`：请求吞吐

**延迟**：
- `TTFT`：首 token 时间
- `TPOT`：每 token 时间
- `E2E_latency`：端到端延迟

**资源**：
- `gpu_util`：GPU 利用率（目标 >70%）
- `kv_cache_util`：KV 缓存利用率
- `memory_usage`：显存使用

#### 3.2 常见问题

**问题 1**：GPU 利用率低（<50%）
**原因**：并发度不足或批次过小
**解决**：增大 `max_num_seqs` 和 `max_num_batched_tokens`

**问题 2**：TTFT 高
**原因**：prompt 长或缓存未命中
**解决**：启用 Prefix Caching，优化 prompt 设计

**问题 3**：频繁抢占
**原因**：KV 缓存容量不足
**解决**：降低 `max_num_seqs`，增大 GPU 内存或使用量化

## 具体案例

### 案例 1：高并发聊天服务

**需求**：
- 模型：Llama-2-70B
- QPS：100
- 延迟：P99 < 2s

**方案**：
- 硬件：4×A100 80GB
- 配置：TP=4，FP16
- 优化：Prefix Caching，CUDA Graph

**结果**：
- 吞吐：120 QPS
- TTFT：P50=200ms，P99=600ms
- GPU 利用率：75%

### 案例 2：代码生成

**需求**：
- 模型：Code Llama-34B
- 场景：IDE 插件，低延迟
- 目标：TTFT < 500ms

**方案**：
- 硬件：2×A100 40GB
- 配置：TP=2，AWQ 量化
- 优化：Chunked Prefill

**结果**：
- TTFT：P99=450ms
- 吞吐：50 QPS
- 内存节省：60%（量化）

### 案例 3：多模态内容理解

**需求**：
- 模型：LLaVA-1.5-13B
- 场景：图像理解 + 对话
- 输入：图像 + 文本

**方案**：
- 硬件：1×A100 80GB
- 配置：单卡，FP16
- 优化：Vision encoder 缓存

**结果**：
- 吞吐：30 QPS
- TTFT：P99=800ms
- 准确率：与原始模型一致

### 案例 4：LoRA 多租户服务

**需求**：
- 基座模型：Llama-2-7B
- LoRA 数量：50+
- 租户数：100+

**方案**：
- 硬件：2×A100 40GB
- 配置：TP=2，max_loras=16
- 策略：LRU 驱逐，动态加载

**结果**：
- 吞吐：80 QPS（混合场景）
- LoRA 切换延迟：<50ms
- 内存开销：+15%

## 最佳实践总结

### 1. 模型选择
- 优先选择经过优化的模型（HF Transformers 格式）
- 考虑使用量化模型（AWQ/GPTQ）节省内存
- 大模型使用张量并行，小模型单卡即可

### 2. 配置调优
- 从默认配置开始，逐步调优
- 根据延迟/吞吐需求调整 `max_num_seqs`
- 启用 Prefix Caching 提升缓存命中率

### 3. 部署建议
- 生产环境使用 API Server 模式
- 配置负载均衡和健康检查
- 监控关键指标，设置告警

### 4. 故障处理
- OOM：降低 `gpu_memory_utilization` 或使用量化
- 低吞吐：增大并发度和批次大小
- 高延迟：启用 Prefix Caching，优化 prompt

### 5. 性能目标
- GPU 利用率：>70%
- KV 缓存利用率：60-80%
- TTFT：<500ms（交互场景）
- 吞吐量：>100 tokens/s/GPU
