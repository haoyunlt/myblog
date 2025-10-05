---
title: "vLLM-00-总览"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 源码剖析
  - 架构分析
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - vLLM-00-总览"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# vLLM-00-总览

## 摘要

### 项目目标
vLLM 是一个高性能的大语言模型（LLM）推理和服务框架。核心设计目标包括：

- **高吞吐量**：通过创新的 PagedAttention 机制和持续批处理技术实现业界领先的服务吞吐量
- **高效内存管理**：利用操作系统虚拟内存的分页思想管理注意力机制的 KV 缓存  
- **易用性**：提供与 HuggingFace Transformers 无缝集成的 API 接口
- **灵活性**：支持多种硬件平台（NVIDIA GPU、AMD GPU/CPU、Intel CPU/GPU、TPU）和并行策略（张量并行、流水线并行、数据并行、专家并行）

###运行环境与部署形态

**支持的硬件平台**：
- NVIDIA CUDA GPU（主要支持）
- AMD ROCm GPU 和 CPU
- Intel CPU、GPU、Gaudi
- Google Cloud TPU
- Huawei Ascend、IBM Spyre（通过插件）

**部署模式**：
- 离线推理：通过 LLM 类进行批量推理
- 在线服务：通过 AsyncLLMEngine 和 OpenAI 兼容的 API Server 提供服务
- 分布式推理：支持张量并行、流水线并行、数据并行和专家并行

**量化支持**：
- GPTQ、AWQ、AutoRound
- INT4、INT8、FP8
- 集成 FlashAttention 和 FlashInfer

### 非目标与边界

vLLM 专注于推理服务，不包括：
- 模型训练和微调（仅支持 LoRA 推理）
- 模型转换（依赖 HuggingFace 格式）
- 端侧部署（面向服务器端）

## 项目模块清单

基于仓库源码目录结构，vLLM 的核心模块划分如下：

| 模块编号 | 模块名称 | 源码路径 | 职责说明 |
|---------|---------|---------|---------|
| 01 | Engine | vllm/engine/, vllm/v1/engine/ | 推理引擎，请求管理和调度协调 |
| 02 | ModelExecutor | vllm/model_executor/, vllm/v1/worker/ | 模型执行器，模型加载和前向传播 |
| 03 | Attention | vllm/attention/, vllm/v1/attention/ | 注意力机制，PagedAttention 实现 |
| 04 | Distributed | vllm/distributed/ | 分布式通信，并行策略实现 |
| 05 | Entrypoints | vllm/entrypoints/ | API 入口，LLM 类和 API Server |
| 06 | Executor | vllm/executor/ | 执行器抽象，支持多种后端 |
| 07 | Config | vllm/config/ | 配置管理，各类配置定义 |
| 08 | LoRA | vllm/lora/ | LoRA 适配器支持 |
| 09 | Multimodal | vllm/multimodal/ | 多模态输入处理 |
| 10 | InputsOutputs | vllm/inputs/, vllm/outputs.py | 输入输出数据结构 |
| 11 | Compilation | vllm/compilation/ | 编译优化，torch.compile 集成 |
| 12 | Scheduler | vllm/v1/core/sched/ | V1 调度器，请求调度和资源分配 |
| 13 | KVCache | vllm/v1/core/kv_cache_utils.py | KV 缓存管理 |
| 14 | CSRC | csrc/ | C++/CUDA 内核实现 |

## 总结

vLLM 通过 PagedAttention、Continuous Batching、CUDA Graph 等技术创新，实现了业界领先的 LLM 推理性能。整体设计遵循高性能系统的最佳实践：无锁并发、零拷贝、预分配内存、批处理优化。

vLLM 的设计理念是"让 LLM 推理又快又便宜"，通过极致的工程优化，将推理成本降低到可接受的水平，使大语言模型能够广泛应用于实际场景。

