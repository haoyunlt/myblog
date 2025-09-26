---
title: "TensorRT-LLM 实战经验与最佳实践"
date: 2024-12-19T10:00:00+08:00
draft: false
tags: ['TensorRT-LLM', 'NVIDIA', '推理优化', '深度学习']
categories: ["tensorrtllm", "技术分析"]
description: "深入分析 TensorRT-LLM 实战经验与最佳实践 的技术实现和架构设计"
weight: 580
slug: "TensorRTLLM-源码剖析_实战经验与最佳实践"
---

# TensorRT-LLM 实战经验与最佳实践

## 1. 性能优化实战

### 1.1 内存优化策略

#### KV 缓存优化
```python
# 最佳实践：启用 KV 缓存块重用
from tensorrt_llm.llmapi import KvCacheConfig

kv_cache_config = KvCacheConfig(
    free_gpu_memory_fraction=0.85,  # 为 KV 缓存预留 85% GPU 内存
    enable_block_reuse=True,        # 启用块重用，提高缓存效率
    max_tokens_in_paged_kv_cache=None,  # 自动计算最大 token 数
    kv_cache_free_gpu_memory_fraction=0.9  # KV 缓存内存分配比例
)

# 实战经验：根据模型大小调整内存分配
def get_optimal_kv_cache_config(model_size_gb: float, gpu_memory_gb: float):
    """根据模型大小和 GPU 内存优化 KV 缓存配置"""

    if model_size_gb <= 7:  # 7B 模型
        return KvCacheConfig(
            free_gpu_memory_fraction=0.9,
            enable_block_reuse=True
        )
    elif model_size_gb <= 13:  # 13B 模型
        return KvCacheConfig(
            free_gpu_memory_fraction=0.8,
            enable_block_reuse=True
        )
    else:  # 更大模型
        return KvCacheConfig(
            free_gpu_memory_fraction=0.7,
            enable_block_reuse=True
        )
```

#### 批次大小优化
```python
def find_optimal_batch_size(model_path: str, max_seq_len: int, gpu_memory_gb: float):
    """动态寻找最优批次大小"""

    # 基于 GPU 内存的初始估算
    base_batch_size = max(1, int(gpu_memory_gb / 4))  # 保守估算

    # 二分搜索最优批次大小
    low, high = 1, base_batch_size * 2
    optimal_batch_size = 1

    while low <= high:
        mid = (low + high) // 2

        try:
            # 测试批次大小
            build_config = BuildConfig(
                max_batch_size=mid,
                max_seq_len=max_seq_len,
                max_input_len=max_seq_len // 2
            )

            # 尝试构建引擎（dry run）
            build_config.dry_run = True
            test_build(model_path, build_config)

            optimal_batch_size = mid
            low = mid + 1

        except torch.cuda.OutOfMemoryError:
            high = mid - 1

    return optimal_batch_size

# 实战配置示例
def create_production_build_config(model_size: str, use_case: str):
    """生产环境构建配置"""

    configs = {
        "7b_chat": BuildConfig(
            max_batch_size=32,
            max_seq_len=4096,
            max_input_len=2048,
            max_beam_width=4,
            strongly_typed=True,
            use_refit=True,  # 支持权重更新
            weight_streaming=False
        ),
        "13b_chat": BuildConfig(
            max_batch_size=16,
            max_seq_len=4096,
            max_input_len=2048,
            max_beam_width=2,
            strongly_typed=True,
            use_refit=True,
            weight_streaming=True  # 大模型启用权重流式传输
        ),
        "70b_inference": BuildConfig(
            max_batch_size=4,
            max_seq_len=2048,
            max_input_len=1024,
            max_beam_width=1,
            strongly_typed=True,
            weight_streaming=True,
            use_strip_plan=True  # 减少引擎大小
        )
    }

    return configs.get(f"{model_size}_{use_case}", configs["7b_chat"])
```

### 1.2 并行策略优化

#### 张量并行最佳实践
```python
def calculate_optimal_tp_size(model_params: int, num_gpus: int, gpu_memory_gb: float):
    """计算最优张量并行大小"""

    # 模型参数量到内存需求的映射（GB）
    model_memory_gb = model_params * 2 / 1e9  # FP16 权重

    # 单卡能否容纳模型
    if model_memory_gb <= gpu_memory_gb * 0.7:
        return 1  # 单卡足够

    # 计算最小需要的 GPU 数量
    min_gpus = math.ceil(model_memory_gb / (gpu_memory_gb * 0.7))

    # 选择合适的 TP 大小（必须是 num_gpus 的因子）
    possible_tp_sizes = [i for i in [1, 2, 4, 8] if i <= num_gpus and num_gpus % i == 0]

    for tp_size in possible_tp_sizes:
        if tp_size >= min_gpus:
            return tp_size

    return min(possible_tp_sizes[-1], num_gpus)

# 实战配置示例
class ParallelismConfig:
    """并行策略配置类"""

    @staticmethod
    def get_config(model_name: str, num_gpus: int):
        """获取并行配置"""

        configs = {
            "llama-7b": {
                1: {"tp": 1, "pp": 1},
                2: {"tp": 2, "pp": 1},
                4: {"tp": 4, "pp": 1},
                8: {"tp": 8, "pp": 1}
            },
            "llama-13b": {
                2: {"tp": 2, "pp": 1},
                4: {"tp": 4, "pp": 1},
                8: {"tp": 8, "pp": 1}
            },
            "llama-70b": {
                4: {"tp": 4, "pp": 1},
                8: {"tp": 8, "pp": 1},
                16: {"tp": 8, "pp": 2}  # 混合并行
            },
            "mixtral-8x7b": {
                4: {"tp": 2, "pp": 1, "ep": 2},  # 专家并行
                8: {"tp": 4, "pp": 1, "ep": 2}
            }
        }

        return configs.get(model_name, {}).get(num_gpus, {"tp": 1, "pp": 1})

# 使用示例
def create_parallel_llm(model_path: str, num_gpus: int):
    """创建并行 LLM 实例"""

    model_name = extract_model_name(model_path)
    config = ParallelismConfig.get_config(model_name, num_gpus)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=config["tp"],
        pipeline_parallel_size=config.get("pp", 1),
        moe_expert_parallel_size=config.get("ep", None)
    )

    return llm
```

### 1.3 量化策略优化

#### 量化算法选择
```python
def select_quantization_strategy(model_size_gb: float, target_latency_ms: float, accuracy_threshold: float):
    """选择最优量化策略"""

    strategies = []

    # FP8 量化 - 最佳性能，轻微精度损失
    if target_latency_ms < 50:
        strategies.append({
            "algo": QuantAlgo.FP8,
            "expected_speedup": 1.8,
            "accuracy_loss": 0.02,
            "memory_reduction": 0.5
        })

    # INT4 AWQ - 平衡性能和精度
    if model_size_gb > 10:
        strategies.append({
            "algo": QuantAlgo.W4A16_AWQ,
            "expected_speedup": 1.4,
            "accuracy_loss": 0.05,
            "memory_reduction": 0.75
        })

    # FP4 量化 - 最大压缩（B200 GPU）
    if model_size_gb > 20:
        strategies.append({
            "algo": QuantAlgo.NVFP4,
            "expected_speedup": 2.0,
            "accuracy_loss": 0.08,
            "memory_reduction": 0.875
        })

    # 选择最佳策略
    best_strategy = None
    for strategy in strategies:
        if strategy["accuracy_loss"] <= (1 - accuracy_threshold):
            if best_strategy is None or strategy["expected_speedup"] > best_strategy["expected_speedup"]:
                best_strategy = strategy

    return best_strategy["algo"] if best_strategy else QuantAlgo.NO_QUANT

# 实战量化配置
def create_production_quant_config(model_path: str, target_use_case: str):
    """生产环境量化配置"""

    use_case_configs = {
        "high_throughput": {
            "quant_algo": QuantAlgo.FP8,
            "kv_cache_quant_algo": QuantAlgo.FP8,
            "group_size": 128
        },
        "memory_constrained": {
            "quant_algo": QuantAlgo.W4A16_AWQ,
            "kv_cache_quant_algo": QuantAlgo.INT8,
            "group_size": 64
        },
        "balanced": {
            "quant_algo": QuantAlgo.W8A8_SQ_PER_CHANNEL,
            "kv_cache_quant_algo": QuantAlgo.INT8,
            "group_size": 128
        }
    }

    config_dict = use_case_configs.get(target_use_case, use_case_configs["balanced"])

    return QuantConfig(
        quant_algo=config_dict["quant_algo"],
        kv_cache_quant_algo=config_dict["kv_cache_quant_algo"],
        group_size=config_dict["group_size"],
        calib_size=512,
        calib_dataset="cnn_dailymail"
    )
```

## 2. 部署实战经验

### 2.1 生产环境部署

#### Docker 容器化部署
```dockerfile
# 生产级 Dockerfile
FROM nvcr.io/nvidia/tensorrt:24.05-py3

# 安装 TensorRT-LLM
RUN pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

# 设置环境变量
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1

# 创建工作目录
WORKDIR /app

# 复制模型和配置
COPY models/ /app/models/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# 设置启动脚本
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
```

```bash
#!/bin/bash
# entrypoint.sh - 生产启动脚本

set -e

# 环境检查
echo "Checking GPU availability..."
nvidia-smi

# 模型路径检查
if [ ! -d "/app/models" ]; then
    echo "Error: Models directory not found"
    exit 1
fi

# 启动参数配置
MODEL_PATH=${MODEL_PATH:-"/app/models/llama-7b"}
TP_SIZE=${TP_SIZE:-1}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}

# 健康检查端点
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

# 启动服务
exec trtllm-serve \
    --model "$MODEL_PATH" \
    --tp_size "$TP_SIZE" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --host 0.0.0.0 \
    --port 8000
```

#### Kubernetes 部署配置
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trtllm-service
  labels:
    app: trtllm-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: trtllm-service
  template:
    metadata:
      labels:
        app: trtllm-service
    spec:
      containers:
      - name: trtllm-container
        image: your-registry/trtllm:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/llama-7b"
        - name: TP_SIZE
          value: "2"
        - name: MAX_BATCH_SIZE
          value: "16"
        resources:
          requests:
            nvidia.com/gpu: 2
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 2
            memory: "32Gi"
            cpu: "8"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100
---
apiVersion: v1
kind: Service
metadata:
  name: trtllm-service
spec:
  selector:
    app: trtllm-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 2.2 监控和日志

#### 性能监控
```python
import time
import psutil
import torch
from typing import Dict, Any
import logging

class TRTLLMMonitor:
    """TensorRT-LLM 性能监控器"""

    def __init__(self, llm):
        self.llm = llm
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0.0,
            "gpu_memory_usage": [],
            "cpu_usage": [],
            "throughput_history": []
        }

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trtllm_performance.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def monitor_request(self, inputs: str, sampling_params: Any) -> Dict[str, Any]:
        """监控单个请求的性能"""

        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            # 执行推理
            outputs = self.llm.generate(inputs, sampling_params)

            # 计算指标
            end_time = time.time()
            inference_time = end_time - start_time

            # 统计 token 数量
            if isinstance(outputs, list):
                total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            else:
                total_tokens = len(outputs.outputs[0].token_ids)

            # 更新指标
            self.metrics["requests_total"] += 1
            self.metrics["requests_success"] += 1
            self.metrics["total_tokens_generated"] += total_tokens
            self.metrics["total_inference_time"] += inference_time

            # 计算吞吐量
            throughput = total_tokens / inference_time
            self.metrics["throughput_history"].append(throughput)

            # 记录系统资源
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                self.metrics["gpu_memory_usage"].append(gpu_memory)

            cpu_percent = psutil.cpu_percent()
            self.metrics["cpu_usage"].append(cpu_percent)

            # 日志记录
            self.logger.info(
                f"Request completed - "
                f"Tokens: {total_tokens}, "
                f"Time: {inference_time:.3f}s, "
                f"Throughput: {throughput:.2f} tokens/s, "
                f"GPU Memory: {gpu_memory / 1e9:.2f}GB"
            )

            return {
                "success": True,
                "inference_time": inference_time,
                "total_tokens": total_tokens,
                "throughput": throughput,
                "outputs": outputs
            }

        except Exception as e:
            self.metrics["requests_failed"] += 1
            self.logger.error(f"Request failed: {str(e)}")

            return {
                "success": False,
                "error": str(e)
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""

        if self.metrics["requests_total"] == 0:
            return {"message": "No requests processed yet"}

        avg_inference_time = self.metrics["total_inference_time"] / self.metrics["requests_success"]
        avg_throughput = sum(self.metrics["throughput_history"]) / len(self.metrics["throughput_history"])
        success_rate = self.metrics["requests_success"] / self.metrics["requests_total"]

        return {
            "total_requests": self.metrics["requests_total"],
            "success_rate": success_rate,
            "avg_inference_time": avg_inference_time,
            "avg_throughput": avg_throughput,
            "total_tokens_generated": self.metrics["total_tokens_generated"],
            "avg_gpu_memory_gb": sum(self.metrics["gpu_memory_usage"]) / len(self.metrics["gpu_memory_usage"]) / 1e9 if self.metrics["gpu_memory_usage"] else 0,
            "avg_cpu_usage": sum(self.metrics["cpu_usage"]) / len(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0
        }

# 使用示例
def setup_monitoring(llm):
    """设置监控"""
    monitor = TRTLLMMonitor(llm)

    # 定期报告性能
    import threading
    import time

    def periodic_report():
        while True:
            time.sleep(60)  # 每分钟报告一次
            summary = monitor.get_performance_summary()
            monitor.logger.info(f"Performance Summary: {summary}")

    report_thread = threading.Thread(target=periodic_report, daemon=True)
    report_thread.start()

    return monitor
```

## 3. 故障排除实战

### 3.1 常见问题诊断

#### 内存不足问题
```python
def diagnose_memory_issues():
    """诊断内存问题"""

    print("=== GPU 内存诊断 ===")

    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return

    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"

        # 获取内存信息
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        cached_memory = torch.cuda.memory_reserved(i)
        free_memory = total_memory - cached_memory

        print(f"GPU {i}:")
        print(f"  总内存: {total_memory / 1e9:.2f} GB")
        print(f"  已分配: {allocated_memory / 1e9:.2f} GB")
        print(f"  已缓存: {cached_memory / 1e9:.2f} GB")
        print(f"  可用内存: {free_memory / 1e9:.2f} GB")
        print(f"  利用率: {(allocated_memory / total_memory) * 100:.1f}%")

        # 内存碎片检查
        try:
            # 尝试分配大块内存
            test_tensor = torch.zeros((1000, 1000, 1000), device=device)
            del test_tensor
            print(f"  内存碎片: 正常")
        except torch.cuda.OutOfMemoryError:
            print(f"  内存碎片: 严重，建议重启进程")

def memory_optimization_suggestions(model_size_gb: float, available_memory_gb: float):
    """内存优化建议"""

    suggestions = []

    if model_size_gb > available_memory_gb * 0.8:
        suggestions.extend([
            "启用权重流式传输 (weight_streaming=True)",
            "使用量化减少模型大小",
            "增加张量并行度分散内存负载"
        ])

    if available_memory_gb < 16:
        suggestions.extend([
            "减少 max_batch_size",
            "减少 max_seq_len",
            "启用 KV 缓存压缩"
        ])

    return suggestions
```

#### 性能问题诊断
```python
def diagnose_performance_issues(llm, test_inputs: List[str]):
    """性能问题诊断"""

    print("=== 性能诊断 ===")

    # 1. 基准测试
    latencies = []
    throughputs = []

    for i, input_text in enumerate(test_inputs):
        start_time = time.time()

        outputs = llm.generate(
            input_text,
            sampling_params=SamplingParams(max_tokens=100)
        )

        end_time = time.time()
        latency = end_time - start_time

        if isinstance(outputs, list):
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        else:
            total_tokens = len(outputs.outputs[0].token_ids)

        throughput = total_tokens / latency

        latencies.append(latency)
        throughputs.append(throughput)

        print(f"测试 {i+1}: 延迟={latency:.3f}s, 吞吐量={throughput:.2f} tokens/s")

    # 2. 统计分析
    avg_latency = sum(latencies) / len(latencies)
    avg_throughput = sum(throughputs) / len(throughputs)

    print(f"\n平均延迟: {avg_latency:.3f}s")
    print(f"平均吞吐量: {avg_throughput:.2f} tokens/s")

    # 3. 性能建议
    suggestions = []

    if avg_latency > 2.0:
        suggestions.append("延迟过高，考虑启用 CUDA 图优化")

    if avg_throughput < 50:
        suggestions.append("吞吐量偏低，检查批次大小和并行配置")

    if len(suggestions) > 0:
        print("\n优化建议:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

def profile_model_performance(llm, num_warmup: int = 5, num_iterations: int = 20):
    """模型性能分析"""

    # 预热
    warmup_input = "Hello, how are you today?"
    for _ in range(num_warmup):
        llm.generate(warmup_input, sampling_params=SamplingParams(max_tokens=10))

    # 性能测试
    test_inputs = [
        "Explain the concept of artificial intelligence.",
        "Write a short story about a robot.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis."
    ]

    results = {
        "first_token_latency": [],
        "total_latency": [],
        "throughput": [],
        "gpu_utilization": []
    }

    for input_text in test_inputs:
        for _ in range(num_iterations):
            # 记录 GPU 利用率
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated()

            start_time = time.time()

            # 流式生成以测量首 token 延迟
            first_token_time = None
            total_tokens = 0

            for output in llm.generate(
                input_text,
                sampling_params=SamplingParams(max_tokens=50),
                streaming=True
            ):
                if first_token_time is None:
                    first_token_time = time.time()
                total_tokens += 1

            end_time = time.time()

            # 计算指标
            first_token_latency = first_token_time - start_time
            total_latency = end_time - start_time
            throughput = total_tokens / total_latency

            results["first_token_latency"].append(first_token_latency)
            results["total_latency"].append(total_latency)
            results["throughput"].append(throughput)

            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                memory_usage = (end_memory - start_memory) / 1e9
                results["gpu_utilization"].append(memory_usage)

    # 统计结果
    for metric, values in results.items():
        if values:
            avg_val = sum(values) / len(values)
            p95_val = sorted(values)[int(len(values) * 0.95)]
            print(f"{metric}: 平均={avg_val:.3f}, P95={p95_val:.3f}")
```

### 3.2 调试技巧

#### 启用详细日志
```python
import os
import logging

def enable_debug_logging():
    """启用调试日志"""

    # TensorRT-LLM 日志级别
    os.environ["TRTLLM_LOG_LEVEL"] = "DEBUG"

    # CUDA 调试
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # NCCL 调试（多 GPU）
    os.environ["NCCL_DEBUG"] = "INFO"

    # Python 日志配置
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )

    print("调试日志已启用")

def trace_model_execution(llm, input_text: str):
    """跟踪模型执行"""

    # 启用 PyTorch 分析器
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        outputs = llm.generate(
            input_text,
            sampling_params=SamplingParams(max_tokens=20)
        )

    # 保存分析结果
    prof.export_chrome_trace("trace.json")

    # 打印关键统计
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return outputs
```

## 4. 高级优化技巧

### 4.1 自定义算子优化

```python
def register_custom_attention_kernel():
    """注册自定义注意力内核"""

    # 这是一个示例，实际需要 C++/CUDA 实现
    custom_kernel_config = {
        "kernel_name": "optimized_attention",
        "block_size": 256,
        "num_warps": 8,
        "enable_flash_attention": True,
        "use_tensor_cores": True
    }

    # 注册到 TensorRT-LLM
    # 实际实现需要通过插件系统
    pass

def optimize_for_specific_hardware(gpu_arch: str):
    """针对特定硬件优化"""

    optimizations = {
        "A100": {
            "use_tf32": True,
            "enable_flash_attention_2": True,
            "optimal_block_size": 128
        },
        "H100": {
            "use_fp8": True,
            "enable_transformer_engine": True,
            "optimal_block_size": 256
        },
        "V100": {
            "use_mixed_precision": True,
            "enable_gradient_checkpointing": True,
            "optimal_block_size": 64
        }
    }

    return optimizations.get(gpu_arch, optimizations["V100"])
```

### 4.2 动态批处理优化

```python
class DynamicBatcher:
    """动态批处理器"""

    def __init__(self, llm, max_batch_size: int = 32, max_wait_time: float = 0.1):
        self.llm = llm
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.request_lock = threading.Lock()

    async def add_request(self, input_text: str, sampling_params: SamplingParams):
        """添加请求到批处理队列"""

        future = asyncio.Future()

        with self.request_lock:
            self.pending_requests.append({
                "input": input_text,
                "sampling_params": sampling_params,
                "future": future,
                "timestamp": time.time()
            })

        return await future

    async def process_batches(self):
        """处理批次的后台任务"""

        while True:
            batch_requests = []

            # 收集批次请求
            with self.request_lock:
                current_time = time.time()

                # 按时间或大小触发批处理
                if self.pending_requests:
                    oldest_request_time = self.pending_requests[0]["timestamp"]
                    time_elapsed = current_time - oldest_request_time

                    if (len(self.pending_requests) >= self.max_batch_size or
                        time_elapsed >= self.max_wait_time):

                        batch_requests = self.pending_requests[:self.max_batch_size]
                        self.pending_requests = self.pending_requests[self.max_batch_size:]

            if batch_requests:
                await self._process_batch(batch_requests)
            else:
                await asyncio.sleep(0.01)  # 短暂休眠

    async def _process_batch(self, batch_requests):
        """处理单个批次"""

        try:
            # 准备批次输入
            inputs = [req["input"] for req in batch_requests]
            sampling_params = [req["sampling_params"] for req in batch_requests]

            # 执行批次推理
            outputs = self.llm.generate(inputs, sampling_params)

            # 分发结果
            for i, request in enumerate(batch_requests):
                request["future"].set_result(outputs[i])

        except Exception as e:
            # 处理错误
            for request in batch_requests:
                request["future"].set_exception(e)

# 使用示例
async def main():
    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    batcher = DynamicBatcher(llm)

    # 启动批处理任务
    asyncio.create_task(batcher.process_batches())

    # 并发请求
    tasks = []
    for i in range(10):
        task = batcher.add_request(
            f"Question {i}: What is AI?",
            SamplingParams(max_tokens=50)
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"Response {i}: {result}")
```

## 5. 生产环境最佳实践总结

### 5.1 配置检查清单

```python
def production_readiness_check(llm_config: dict) -> dict:
    """生产就绪性检查"""

    checks = {
        "memory_optimization": False,
        "error_handling": False,
        "monitoring": False,
        "security": False,
        "scalability": False,
        "performance": False,
        "reliability": False
    }

    recommendations = []

    # 内存优化检查
    if llm_config.get("kv_cache_config", {}).get("enable_block_reuse", False):
        checks["memory_optimization"] = True
    else:
        recommendations.append("启用 KV 缓存块重用")

    # 错误处理检查
    if llm_config.get("error_handling", {}).get("retry_config"):
        checks["error_handling"] = True
    else:
        recommendations.append("配置重试机制")

    # 监控检查
    if llm_config.get("monitoring", {}).get("enabled", False):
        checks["monitoring"] = True
    else:
        recommendations.append("启用性能监控")

    # 安全检查
    if llm_config.get("security", {}).get("input_validation", False):
        checks["security"] = True
    else:
        recommendations.append("添加输入验证")

    # 可扩展性检查
    if llm_config.get("parallel_config", {}).get("tensor_parallel_size", 1) > 1:
        checks["scalability"] = True
    else:
        recommendations.append("考虑并行配置")

    # 性能检查
    if llm_config.get("optimization", {}).get("enable_cuda_graph", False):
        checks["performance"] = True
    else:
        recommendations.append("启用 CUDA 图优化")

    # 可靠性检查
    if llm_config.get("reliability", {}).get("health_check_enabled", False):
        checks["reliability"] = True
    else:
        recommendations.append("配置健康检查")

    return {
        "checks": checks,
        "recommendations": recommendations,
        "ready": all(checks.values()),
        "score": sum(checks.values()) / len(checks) * 100
    }

# 生产配置模板
PRODUCTION_CONFIG_TEMPLATE = {
    "model_config": {
        "max_batch_size": 16,
        "max_seq_len": 4096,
        "max_input_len": 2048,
        "strongly_typed": True,
        "use_refit": True,
        "weight_streaming": False
    },
    "kv_cache_config": {
        "free_gpu_memory_fraction": 0.85,
        "enable_block_reuse": True,
        "max_tokens_in_paged_kv_cache": None
    },
    "quantization_config": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "INT8",
        "group_size": 128,
        "calib_size": 512
    },
    "parallel_config": {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "moe_expert_parallel_size": None
    },
    "optimization": {
        "enable_cuda_graph": True,
        "use_fused_attention": True,
        "enable_kv_cache_reuse": True,
        "max_beam_width": 4
    },
    "monitoring": {
        "enabled": True,
        "metrics_interval": 60,
        "log_level": "INFO",
        "export_prometheus": True,
        "trace_requests": False
    },
    "error_handling": {
        "retry_config": {
            "max_retries": 3,
            "backoff_factor": 2.0,
            "retry_exceptions": ["TimeoutError", "CudaError"]
        },
        "timeout_seconds": 30,
        "circuit_breaker": {
            "failure_threshold": 5,
            "recovery_timeout": 60
        }
    },
    "security": {
        "input_validation": True,
        "max_input_length": 8192,
        "content_filtering": True,
        "rate_limiting": {
            "requests_per_minute": 100,
            "burst_size": 10
        },
        "authentication": {
            "enabled": True,
            "token_validation": True
        }
    },
    "reliability": {
        "health_check_enabled": True,
        "health_check_interval": 30,
        "graceful_shutdown_timeout": 60,
        "auto_recovery": True
    }
}
```

### 5.2 高级性能调优技巧

#### 5.2.1 内存池优化

```python
class AdvancedMemoryManager:
    """高级内存管理器"""

    def __init__(self, gpu_memory_gb: float):
        self.total_memory = gpu_memory_gb * 1024**3
        self.memory_pools = {}
        self.allocation_strategy = "best_fit"

    def optimize_memory_layout(self, model_config: dict):
        """优化内存布局"""

        # 计算各组件内存需求
        model_memory = self._estimate_model_memory(model_config)
        kv_cache_memory = self._estimate_kv_cache_memory(model_config)
        activation_memory = self._estimate_activation_memory(model_config)

        # 内存分配策略
        memory_allocation = {
            "model_weights": model_memory,
            "kv_cache": kv_cache_memory,
            "activations": activation_memory,
            "workspace": self.total_memory * 0.1,  # 10% 工作空间
            "reserved": self.total_memory * 0.05   # 5% 预留
        }

        total_required = sum(memory_allocation.values())

        if total_required > self.total_memory:
            # 内存不足，启用优化策略
            return self._apply_memory_optimization(memory_allocation)

        return memory_allocation

    def _apply_memory_optimization(self, allocation: dict):
        """应用内存优化策略"""

        optimizations = []

        # 1. 启用权重流式传输
        if allocation["model_weights"] > self.total_memory * 0.6:
            optimizations.append("weight_streaming")
            allocation["model_weights"] *= 0.3  # 减少70%内存占用

        # 2. 压缩KV缓存
        if allocation["kv_cache"] > self.total_memory * 0.4:
            optimizations.append("kv_cache_compression")
            allocation["kv_cache"] *= 0.5  # 减少50%内存占用

        # 3. 激活检查点
        if allocation["activations"] > self.total_memory * 0.2:
            optimizations.append("activation_checkpointing")
            allocation["activations"] *= 0.3  # 减少70%内存占用

        return {
            "allocation": allocation,
            "optimizations": optimizations,
            "memory_saved": self._calculate_memory_saved(optimizations)
        }

#### 5.2.2 动态批处理优化

```python
class DynamicBatchOptimizer:
    """动态批处理优化器"""

    def __init__(self, max_batch_size: int = 32, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = asyncio.Queue()
        self.batch_stats = {
            "total_batches": 0,
            "avg_batch_size": 0,
            "avg_wait_time": 0,
            "throughput_history": []
        }

    async def optimize_batching(self, llm):
        """优化批处理策略"""

        while True:
            batch = await self._collect_batch()

            if not batch:
                await asyncio.sleep(0.01)
                continue

            # 动态调整批次大小
            optimal_batch_size = self._calculate_optimal_batch_size(batch)

            if len(batch) > optimal_batch_size:
                # 分割批次
                batches = self._split_batch(batch, optimal_batch_size)
                for sub_batch in batches:
                    await self._process_batch(sub_batch, llm)
            else:
                await self._process_batch(batch, llm)

    def _calculate_optimal_batch_size(self, batch: list) -> int:
        """计算最优批次大小"""

        # 基于序列长度分布计算
        seq_lengths = [len(req.prompt_token_ids) for req in batch]
        avg_seq_len = sum(seq_lengths) / len(seq_lengths)
        max_seq_len = max(seq_lengths)

        # 内存约束下的最大批次大小
        memory_constrained_size = self._estimate_max_batch_size(max_seq_len)

        # 延迟约束下的最优批次大小
        latency_optimal_size = self._estimate_latency_optimal_size(avg_seq_len)

        return min(memory_constrained_size, latency_optimal_size, self.max_batch_size)

    async def _collect_batch(self) -> list:
        """收集批次请求"""

        batch = []
        start_time = time.time()

        # 收集第一个请求
        try:
            first_request = await asyncio.wait_for(
                self.request_queue.get(),
                timeout=self.max_wait_time
            )
            batch.append(first_request)
        except asyncio.TimeoutError:
            return batch

        # 收集更多请求直到达到限制
        while (len(batch) < self.max_batch_size and
               time.time() - start_time < self.max_wait_time):
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=0.01
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        return batch

#### 5.2.3 智能缓存策略

```python
class IntelligentCacheManager:
    """智能缓存管理器"""

    def __init__(self, cache_size_gb: float = 8.0):
        self.cache_size = cache_size_gb * 1024**3
        self.semantic_cache = {}
        self.lru_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    def get_cached_result(self, prompt: str, sampling_params: dict) -> Optional[str]:
        """获取缓存结果"""

        # 1. 精确匹配缓存
        exact_key = self._generate_exact_key(prompt, sampling_params)
        if exact_key in self.lru_cache:
            self.cache_stats["hits"] += 1
            return self.lru_cache[exact_key]

        # 2. 语义相似性缓存
        semantic_result = self._semantic_cache_lookup(prompt, sampling_params)
        if semantic_result:
            self.cache_stats["hits"] += 1
            return semantic_result

        self.cache_stats["misses"] += 1
        return None

    def cache_result(self, prompt: str, sampling_params: dict, result: str):
        """缓存结果"""

        exact_key = self._generate_exact_key(prompt, sampling_params)

        # 检查缓存空间
        if self._get_cache_size() > self.cache_size * 0.9:
            self._evict_lru_entries()

        # 存储到精确匹配缓存
        self.lru_cache[exact_key] = result

        # 存储到语义缓存
        self._store_semantic_cache(prompt, sampling_params, result)

    def _semantic_cache_lookup(self, prompt: str, sampling_params: dict) -> Optional[str]:
        """语义缓存查找"""

        # 计算提示的语义嵌入
        prompt_embedding = self._compute_embedding(prompt)

        # 查找相似的缓存条目
        for cached_prompt, cached_data in self.semantic_cache.items():
            similarity = self._compute_similarity(
                prompt_embedding,
                cached_data["embedding"]
            )

            # 相似度阈值和参数匹配
            if (similarity > 0.95 and
                self._params_compatible(sampling_params, cached_data["params"])):
                return cached_data["result"]

        return None

    def _params_compatible(self, params1: dict, params2: dict) -> bool:
        """检查参数兼容性"""

        # 关键参数必须完全匹配
        critical_params = ["max_tokens", "temperature", "top_p", "top_k"]

        for param in critical_params:
            if abs(params1.get(param, 0) - params2.get(param, 0)) > 1e-6:
                return False

        return True
```

### 5.2 关键性能指标 (KPI)

```python
class ProductionKPIs:
    """生产环境关键性能指标"""

    TARGET_METRICS = {
        "latency_p95_ms": 2000,      # P95 延迟 < 2秒
        "throughput_tokens_per_sec": 100,  # 吞吐量 > 100 tokens/s
        "availability_percent": 99.9,      # 可用性 > 99.9%
        "error_rate_percent": 0.1,         # 错误率 < 0.1%
        "gpu_utilization_percent": 80,     # GPU 利用率 > 80%
        "memory_utilization_percent": 85   # 内存利用率 < 85%
    }

    @staticmethod
    def evaluate_performance(metrics: dict) -> dict:
        """评估性能是否达标"""

        results = {}

        for metric, target in ProductionKPIs.TARGET_METRICS.items():
            actual = metrics.get(metric, 0)

            if metric in ["latency_p95_ms", "error_rate_percent", "memory_utilization_percent"]:
                # 越小越好的指标
                passed = actual <= target
            else:
                # 越大越好的指标
                passed = actual >= target

            results[metric] = {
                "target": target,
                "actual": actual,
                "passed": passed,
                "deviation": ((actual - target) / target) * 100
            }

        overall_passed = all(result["passed"] for result in results.values())

        return {
            "overall_passed": overall_passed,
            "metrics": results
        }
```

这份实战经验文档涵盖了 TensorRT-LLM 在生产环境中的性能优化、部署实践、故障排除和高级优化技巧，为实际项目提供了详细的技术指导和最佳实践建议。
