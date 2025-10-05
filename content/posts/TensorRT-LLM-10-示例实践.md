---
title: "TensorRT-LLM-10-示例实践"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 源码分析
categories:
  - TensorRT
description: "源码剖析 - TensorRT-LLM-10-示例实践"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# TensorRT-LLM-10-示例实践

## 一、快速开始

### 1.1 基础推理示例

```python
from tensorrt_llm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="meta-llama/Llama-3-8B")

# 单个prompt推理
output = llm.generate("What is artificial intelligence?")
print(output.outputs[0].text)

# 批量推理
prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "Tell me about neural networks"
]
outputs = llm.generate(prompts)
for output in outputs:
    print(f"输入：{output.prompt}")
    print(f"输出：{output.outputs[0].text}\n")
```

### 1.2 自定义采样参数

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="Llama-3-8B")

# 创意写作（高temperature，多样性）
creative_params = SamplingParams(
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    max_new_tokens=256
)

# 精确问答（低temperature，确定性）
precise_params = SamplingParams(
    temperature=0.1,
    top_k=10,
    max_new_tokens=100
)

# 使用不同参数
creative_output = llm.generate("Write a creative story", creative_params)
precise_output = llm.generate("What is 2+2?", precise_params)
```

## 二、实战案例

### 2.1 案例1：构建聊天机器人

```python
from tensorrt_llm import LLM, SamplingParams

class ChatBot:
    def __init__(self, model_name="Llama-3-8B"):
        self.llm = LLM(model=model_name)
        self.conversation_history = []
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512
        )
    
    def chat(self, user_input):
        # 构建对话历史
        self.conversation_history.append(f"User: {user_input}")
        
        # 构建完整prompt
        prompt = "\n".join(self.conversation_history) + "\nAssistant:"
        
        # 生成响应
        output = self.llm.generate(prompt, self.sampling_params)
        response = output.outputs[0].text
        
        # 更新历史
        self.conversation_history.append(f"Assistant: {response}")
        
        return response

# 使用示例
bot = ChatBot()
print(bot.chat("Hello! How are you?"))
print(bot.chat("Can you explain what you can do?"))
```

### 2.2 案例2：文本摘要服务

```python
from tensorrt_llm import LLM, SamplingParams
import asyncio

class SummarizationService:
    def __init__(self):
        self.llm = LLM(
            model="Llama-3-8B",
            max_batch_size=128  # 支持大批次
        )
        self.params = SamplingParams(
            temperature=0.3,
            max_new_tokens=200
        )
    
    async def summarize_batch(self, texts):
        """批量摘要"""
        prompts = [
            f"Summarize the following text:\n{text}\n\nSummary:"
            for text in texts
        ]
        
        # 批量处理
        outputs = self.llm.generate(prompts, self.params)
        
        return [output.outputs[0].text for output in outputs]
    
    def summarize_single(self, text):
        """单个摘要"""
        prompt = f"Summarize: {text}\n\nSummary:"
        output = self.llm.generate(prompt, self.params)
        return output.outputs[0].text

# 使用示例
service = SummarizationService()

# 单个摘要
summary = service.summarize_single("Long article text here...")
print(summary)

# 批量摘要
texts = ["Article 1...", "Article 2...", "Article 3..."]
summaries = asyncio.run(service.summarize_batch(texts))
```

### 2.3 案例3：代码生成助手

```python
from tensorrt_llm import LLM, SamplingParams

class CodeAssistant:
    def __init__(self):
        self.llm = LLM(
            model="DeepSeek-Coder-33B",  # 使用代码模型
            tensor_parallel_size=2        # 2 GPU并行
        )
        self.params = SamplingParams(
            temperature=0.2,  # 代码生成需要低temperature
            max_new_tokens=1024
        )
    
    def generate_code(self, instruction, language="python"):
        prompt = f"""Generate {language} code for the following task:
{instruction}

Code:
```{language}
"""
        output = self.llm.generate(prompt, self.params)
        return output.outputs[0].text
    
    def explain_code(self, code):
        prompt = f"""Explain the following code:
{code}

Explanation:
"""
        output = self.llm.generate(prompt, self.params)
        return output.outputs[0].text

# 使用示例
assistant = CodeAssistant()
code = assistant.generate_code("Create a binary search function")
explanation = assistant.explain_code(code)
```

## 三、最佳实践

### 3.1 性能优化最佳实践

#### 3.1.1 批处理优化

```python
# ❌ 不推荐：逐个处理
for prompt in prompts:
    output = llm.generate(prompt)
    process(output)

# ✅ 推荐：批量处理
outputs = llm.generate(prompts)
for output in outputs:
    process(output)
```

**性能提升：** 批量处理可提升吞吐量 5-10 倍

#### 3.1.2 使用异步API

```python
import asyncio

async def process_requests(llm, prompts):
    # 并发提交所有请求
    futures = [llm.generate_async(p) for p in prompts]
    
    # 并发等待
    results = await asyncio.gather(*[f.aresult() for f in futures])
    
    return results

# 使用
results = asyncio.run(process_requests(llm, prompts))
```

#### 3.1.3 流式生成优化

```python
# 对于长文本生成，使用流式输出
future = llm.generate_async(prompt, streaming=True)

# 实时显示
for chunk in future:
    print(chunk.outputs[0].text, end="", flush=True)
    if chunk.finished:
        break
```

### 3.2 显存优化最佳实践

#### 3.2.1 大模型部署

```python
# 70B 模型需要多GPU
llm = LLM(
    model="Llama-3-70B",
    tensor_parallel_size=8,      # 8 GPU TP
    dtype="float16",              # 使用FP16
    quantization="fp8",           # FP8量化
    max_batch_size=64,            # 控制批次大小
    kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.85,  # 留15%显存
        enable_block_reuse=True         # 启用KV Cache复用
    )
)
```

#### 3.2.2 量化部署

```python
# INT8量化（节省50%显存）
llm = LLM(
    model="Llama-3-8B",
    quantization="int8",
    max_batch_size=256
)

# FP8量化（Hopper架构，性能最优）
llm = LLM(
    model="Llama-3-70B",
    quantization="fp8",
    tensor_parallel_size=4
)
```

### 3.3 长上下文处理最佳实践

```python
# 启用分块上下文处理
llm = LLM(
    model="Llama-3-8B",
    max_input_len=16384,
    max_seq_len=32768,
    enable_chunked_context=True,  # 分块处理
    max_num_tokens=8192           # 每个chunk大小
)

# 处理长文档
long_document = "..." # 20k tokens
summary = llm.generate(f"Summarize: {long_document}")
```

### 3.4 错误处理最佳实践

```python
from tensorrt_llm import LLM
import logging

def create_llm_with_fallback(model, **kwargs):
    """创建LLM实例，带自动降级"""
    try:
        # 尝试最优配置
        return LLM(model=model, **kwargs)
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.warning("显存不足，降低批次大小")
            kwargs['max_batch_size'] = kwargs.get('max_batch_size', 256) // 2
            return LLM(model=model, **kwargs)
        raise
    
    except Exception as e:
        logging.error(f"LLM创建失败：{e}")
        raise

# 使用
llm = create_llm_with_fallback(
    "Llama-3-70B",
    tensor_parallel_size=8,
    max_batch_size=128
)
```

## 四、生产环境部署

### 4.1 高可用部署

```python
from tensorrt_llm import LLM
import threading
from queue import Queue

class LLMService:
    """生产级LLM服务"""
    
    def __init__(self, model, num_instances=2):
        self.instances = []
        self.request_queue = Queue()
        
        # 创建多个实例（不同GPU）
        for i in range(num_instances):
            llm = LLM(
                model=model,
                tensor_parallel_size=4,
                device_ids=[i*4, i*4+1, i*4+2, i*4+3]
            )
            self.instances.append(llm)
        
        # 启动worker线程
        for llm in self.instances:
            t = threading.Thread(target=self._worker, args=(llm,))
            t.daemon = True
            t.start()
    
    def _worker(self, llm):
        while True:
            request = self.request_queue.get()
            if request is None:
                break
            
            try:
                result = llm.generate(request['prompt'])
                request['callback'](result)
            except Exception as e:
                request['callback'](None, error=e)
    
    def generate_async(self, prompt, callback):
        """异步生成接口"""
        self.request_queue.put({
            'prompt': prompt,
            'callback': callback
        })

# 使用
service = LLMService("Llama-3-70B", num_instances=2)
```

### 4.2 监控和日志

```python
from tensorrt_llm import LLM
import time
import logging

class MonitoredLLM:
    """带监控的LLM封装"""
    
    def __init__(self, model, **kwargs):
        self.llm = LLM(model=model, **kwargs)
        self.metrics = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }
    
    def generate(self, prompts, **kwargs):
        start_time = time.time()
        
        try:
            outputs = self.llm.generate(prompts, **kwargs)
            
            # 记录指标
            self.metrics['total_requests'] += len(prompts) if isinstance(prompts, list) else 1
            self.metrics['total_tokens'] += sum(len(o.outputs[0].token_ids) for o in outputs)
            self.metrics['total_time'] += time.time() - start_time
            
            # 定期输出统计
            if self.metrics['total_requests'] % 100 == 0:
                self._log_metrics()
            
            return outputs
        
        except Exception as e:
            logging.error(f"生成失败：{e}")
            raise
    
    def _log_metrics(self):
        avg_time = self.metrics['total_time'] / self.metrics['total_requests']
        tokens_per_sec = self.metrics['total_tokens'] / self.metrics['total_time']
        
        logging.info(f"""
        总请求数：{self.metrics['total_requests']}
        总Token数：{self.metrics['total_tokens']}
        平均延迟：{avg_time:.3f}s
        吞吐量：{tokens_per_sec:.1f} tokens/s
        """)

# 使用
llm = MonitoredLLM("Llama-3-8B")
```

## 五、常见问题与解决方案

### 5.1 性能问题

**问题1：首Token延迟(TTFT)过高**

**原因：** Context Phase 需要处理整个输入序列

**解决方案：**
```python
# 启用分块上下文
llm = LLM(
    model="Llama-3-8B",
    enable_chunked_context=True,
    max_num_tokens=2048  # 减小chunk大小
)
```

**问题2：吞吐量低**

**解决方案：**
```python
# 增加批次大小
llm = LLM(
    model="Llama-3-8B",
    max_batch_size=512,  # 增大批次
    enable_chunked_context=True
)

# 使用批量处理
outputs = llm.generate(large_batch_of_prompts)
```

### 5.2 显存问题

**问题：OOM (Out of Memory)**

**解决方案：**
1. 增加GPU数量（TP）
2. 启用量化
3. 减小批次大小
4. 减小KV Cache

```python
llm = LLM(
    model="Llama-3-70B",
    tensor_parallel_size=8,      # 增加GPU
    quantization="fp8",           # 量化
    max_batch_size=64,            # 减小批次
    kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.8
    )
)
```

### 5.3 精度问题

**问题：量化后精度下降**

**解决方案：**
```python
# 使用更高精度的量化
llm = LLM(
    model="Llama-3-70B",
    quantization="fp8",  # 而非int4
    dtype="float16"
)

# 或使用AWQ量化
llm = LLM(
    model="Llama-3-70B-AWQ",  # 使用预量化模型
)
```

## 六、总结

本文档涵盖了 TensorRT-LLM 的：

- ✅ 基础使用示例
- ✅ 实战案例（聊天机器人、摘要、代码生成）
- ✅ 性能优化最佳实践
- ✅ 生产环境部署方案
- ✅ 常见问题解决方案

**关键要点：**

1. **批处理优化：** 始终使用批量处理提升吞吐量
2. **异步API：** 长时间任务使用异步接口
3. **显存管理：** 根据模型大小选择合适的并行和量化策略
4. **错误处理：** 实现自动降级和重试机制
5. **监控指标：** 记录TTFT、TPS等关键指标

---

**文档版本：** 1.0  
**生成时间：** 2025-10-05  
**对应代码版本：** TensorRT-LLM v1.2.0rc1
