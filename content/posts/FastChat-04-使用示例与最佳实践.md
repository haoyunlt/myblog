---
title: "FastChat-04-使用示例与最佳实践"
date: 2025-10-05T10:45:52+08:00
draft: false
tags:
  - 最佳实践
  - 实战经验
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - FastChat-04-使用示例与最佳实践"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# FastChat-04-使用示例与最佳实践

## 1. 快速开始

### 1.1 环境准备

**系统要求**：
- Python 3.8+
- CUDA 11.7+（GPU 推理）
- 磁盘空间：模型权重 + 缓存至少 50GB

**安装 FastChat**：

```bash
# 方法1：从PyPI安装（推荐）
pip3 install "fschat[model_worker,webui]"

# 方法2：从源码安装（开发者）
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install -e ".[model_worker,webui]"
```

**验证安装**：

```bash
python3 -c "import fastchat; print(fastchat.__version__)"
# 输出：0.2.36
```

---

### 1.2 最小可运行示例

#### 场景 1：CLI 命令行对话（单机模式）

**启动方式**（无需 Controller）：

```bash
python3 -m fastchat.serve.cli \
    --model-path lmsys/vicuna-7b-v1.5 \
    --device cuda
```

**交互示例**：

```
USER: Hello! Can you introduce yourself?
ASSISTANT: Hello! I'm Vicuna, a language model trained by researchers...

USER: What can you help me with?
ASSISTANT: I can help you with various tasks such as answering questions...
```

**适用场景**：
- 本地测试模型效果
- 快速验证模型权重是否正确加载
- 开发调试对话模板

**限制**：
- 不支持并发请求
- 无 Web UI
- 无 API 接口

---

#### 场景 2：单模型 Web 服务（分布式模式）

**第 1 步：启动 Controller**

```bash
# 终端 1
python3 -m fastchat.serve.controller \
    --host 0.0.0.0 \
    --port 21001
```

**输出日志**：
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:21001
```

**第 2 步：启动 Model Worker**

```bash
# 终端 2
python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001 \
    --worker http://localhost:31000 \
    --host 0.0.0.0 \
    --port 31000
```

**输出日志**：
```
Loading the model vicuna-7b-v1.5 on worker abc12345 ...
Register to controller
Register done: http://localhost:31000, {'model_names': ['vicuna-7b-v1.5'], 'speed': 1, 'queue_length': 0}
Uvicorn running on http://0.0.0.0:31000
```

**第 3 步：启动 Gradio Web Server**

```bash
# 终端 3
python3 -m fastchat.serve.gradio_web_server \
    --controller http://localhost:21001 \
    --host 0.0.0.0 \
    --port 7860
```

**访问 Web UI**：
- 打开浏览器：http://localhost:7860
- 选择模型：vicuna-7b-v1.5
- 开始对话

**验证服务**：

```bash
# 测试消息发送
python3 -m fastchat.serve.test_message --model-name vicuna-7b-v1.5
```

---

#### 场景 3：OpenAI 兼容 API 服务

**启动 OpenAI API Server**：

```bash
# 前提：Controller 和 Worker 已启动
python3 -m fastchat.serve.openai_api_server \
    --controller-address http://localhost:21001 \
    --host 0.0.0.0 \
    --port 8000
```

**调用示例（curl）**：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.5",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**响应示例**：

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1696464000,
  "model": "vicuna-7b-v1.5",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 25,
    "total_tokens": 37
  }
}
```

**调用示例（Python SDK）**：

```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"  # FastChat默认不需要API Key

response = openai.ChatCompletion.create(
    model="vicuna-7b-v1.5",
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

**流式调用**：

```python
response = openai.ChatCompletion.create(
    model="vicuna-7b-v1.5",
    messages=[
        {"role": "user", "content": "Write a short story"}
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.get("content"):
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 2. 生产部署实践

### 2.1 多模型并行服务

**场景**：同时服务 Vicuna-7B 和 Vicuna-13B

**部署架构**：

```
Controller (21001)
  ├── Worker 1: Vicuna-7B (GPU 0, 31000)
  ├── Worker 2: Vicuna-13B (GPU 1, 31001)
  ├── OpenAI API Server (8000)
  └── Gradio Web Server (7860)
```

**启动脚本**：

```bash
#!/bin/bash
# start_all.sh

# 启动 Controller
python3 -m fastchat.serve.controller --port 21001 &
sleep 5

# 启动 Worker 1 (Vicuna-7B on GPU 0)
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001 \
    --worker http://localhost:31000 \
    --port 31000 &
sleep 10

# 启动 Worker 2 (Vicuna-13B on GPU 1)
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-13b-v1.5 \
    --controller http://localhost:21001 \
    --worker http://localhost:31001 \
    --port 31001 &
sleep 10

# 启动 API Server
python3 -m fastchat.serve.openai_api_server \
    --controller-address http://localhost:21001 \
    --port 8000 &

# 启动 Gradio Web Server
python3 -m fastchat.serve.gradio_web_server \
    --controller-address http://localhost:21001 \
    --port 7860 &

echo "All services started!"
```

**验证多模型**：

```bash
curl http://localhost:8000/v1/models
```

**响应**：
```json
{
  "object": "list",
  "data": [
    {"id": "vicuna-7b-v1.5", "object": "model", ...},
    {"id": "vicuna-13b-v1.5", "object": "model", ...}
  ]
}
```

---

### 2.2 混合云部署（本地模型 + 第三方 API）

**场景**：本地部署 Vicuna-7B，同时接入 OpenAI GPT-4 和 Anthropic Claude

**步骤 1：创建 API 配置文件** `api_endpoints.json`

```json
{
  "gpt-4": {
    "model_name": "gpt-4",
    "api_base": "https://api.openai.com/v1",
    "api_type": "openai",
    "api_key": "sk-xxxxxxxxxxxxxxxx",
    "anony_only": false
  },
  "claude-3-opus": {
    "model_name": "claude-3-opus-20240229",
    "api_base": "https://api.anthropic.com",
    "api_type": "anthropic",
    "api_key": "sk-ant-xxxxxxxxxxxxxxxx",
    "anony_only": false
  }
}
```

**步骤 2：启动服务**

```bash
# 启动 Controller
python3 -m fastchat.serve.controller &

# 启动本地 Worker (Vicuna-7B)
python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001 &

# 启动 Gradio Multi Server（自动加载 API）
python3 -m fastchat.serve.gradio_web_server_multi \
    --controller http://localhost:21001 \
    --register-api-endpoint-file api_endpoints.json \
    --port 7860
```

**结果**：Web UI 中可选择 3 个模型：
- vicuna-7b-v1.5（本地）
- gpt-4（OpenAI API）
- claude-3-opus（Anthropic API）

---

### 2.3 高可用与负载均衡

#### 水平扩展：多 Worker 服务同一模型

**场景**：3 个 Worker 同时服务 Vicuna-7B（提升吞吐量）

```bash
# Worker 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001 \
    --worker http://localhost:31000 \
    --port 31000 &

# Worker 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001 \
    --worker http://localhost:31001 \
    --port 31001 &

# Worker 3 on GPU 2
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001 \
    --worker http://localhost:31002 \
    --port 31002 &
```

**负载均衡验证**：

```bash
# 查看 Controller 日志，观察请求分配
tail -f controller.log | grep "get_worker_address"
```

**预期输出**：
```
names: [...31000, ...31001, ...31002], queue_lens: [0.5, 0.5, 1.0], ret: http://localhost:31000
names: [...31000, ...31001, ...31002], queue_lens: [1.5, 0.5, 1.0], ret: http://localhost:31001
```

#### 切换负载均衡策略

```bash
# Controller 启动时指定策略
python3 -m fastchat.serve.controller \
    --dispatch-method shortest_queue  # 或 lottery（默认）
```

**Lottery vs Shortest Queue**：

| 指标 | Lottery | Shortest Queue |
|---|---|---|
| 负载均衡 | 概率均衡 | 严格最短队列 |
| 适用场景 | 异构环境（不同GPU） | 同构环境（相同GPU） |
| 延迟 | 中等 | 最优 |

---

### 2.4 进程管理与自动重启

#### 使用 systemd 管理服务

**Controller 服务** `/etc/systemd/system/fastchat-controller.service`

```ini
[Unit]
Description=FastChat Controller
After=network.target

[Service]
Type=simple
User=fastchat
WorkingDirectory=/opt/fastchat
ExecStart=/usr/bin/python3 -m fastchat.serve.controller --port 21001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Worker 服务** `/etc/systemd/system/fastchat-worker@.service`

```ini
[Unit]
Description=FastChat Worker %i
After=fastchat-controller.service
Requires=fastchat-controller.service

[Service]
Type=simple
User=fastchat
WorkingDirectory=/opt/fastchat
Environment="CUDA_VISIBLE_DEVICES=%i"
ExecStart=/usr/bin/python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001 \
    --worker http://localhost:3100%i \
    --port 3100%i
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

**启动服务**：

```bash
# 启动 Controller
sudo systemctl start fastchat-controller
sudo systemctl enable fastchat-controller

# 启动 Worker 0（GPU 0）
sudo systemctl start fastchat-worker@0
sudo systemctl enable fastchat-worker@0

# 启动 Worker 1（GPU 1）
sudo systemctl start fastchat-worker@1
sudo systemctl enable fastchat-worker@1

# 查看状态
sudo systemctl status fastchat-controller
sudo systemctl status fastchat-worker@0
```

---

## 3. 性能调优实战

### 3.1 显存优化

#### 8bit 量化（显存减半）

```bash
python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-13b-v1.5 \
    --load-8bit \
    --controller http://localhost:21001 \
    --worker http://localhost:31000 \
    --port 31000
```

**效果**：
- Vicuna-13B：28GB → 14GB
- 精度损失：< 1%
- 推理速度：略微下降（5-10%）

#### 4bit GPTQ 量化（显存降至 1/4）

**前提**：需要预量化的 GPTQ 模型

```bash
python3 -m fastchat.serve.model_worker \
    --model-path TheBloke/vicuna-13B-v1.5-GPTQ \
    --gptq wbits=4 groupsize=128 \
    --controller http://localhost:21001 \
    --worker http://localhost:31000 \
    --port 31000
```

**效果**：
- Vicuna-13B：28GB → 7GB
- 精度损失：2-3%
- 推理速度：略微提升（GPU 利用率更高）

#### CPU 卸载（超大模型）

```bash
python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-33b-v1.3 \
    --load-8bit \
    --cpu-offloading \
    --controller http://localhost:21001 \
    --worker http://localhost:31000 \
    --port 31000
```

**效果**：
- 部分权重卸载到 CPU 内存
- 显存需求降低，但推理速度显著下降（2-5 倍）

---

### 3.2 吞吐量优化

#### 使用 vLLM 高吞吐引擎

```bash
pip install vllm

python3 -m fastchat.serve.vllm_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001 \
    --worker http://localhost:31000 \
    --port 31000 \
    --trust-remote-code
```

**性能提升**：
- 吞吐量：10-24 倍（通过 PagedAttention 和连续批处理）
- 适用场景：高并发 API 服务
- 限制：仅支持部分模型架构

#### 调整并发数

```bash
python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --limit-worker-concurrency 10 \  # 默认5，根据显存调整
    --controller http://localhost:21001 \
    --worker http://localhost:31000 \
    --port 31000
```

**容量规划**：

| 模型 | 单请求显存 | A100-40GB 并发数 | A100-80GB 并发数 |
|---|---|---|---|
| Vicuna-7B (FP16) | ~6GB | 5-6 | 12-13 |
| Vicuna-13B (FP16) | ~12GB | 2-3 | 6-7 |
| Vicuna-7B (8bit) | ~3GB | 12-13 | 26-27 |

---

### 3.3 延迟优化

#### 流式间隔调整

```bash
python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --stream-interval 1 \  # 每1个token返回（默认2）
    --controller http://localhost:21001 \
    --worker http://localhost:31000 \
    --port 31000
```

**权衡**：
- `stream-interval 1`：最低延迟，但网络开销大
- `stream-interval 5`：降低网络开销，略增延迟

#### Flash Attention 加速

```bash
pip install flash-attn --no-build-isolation

# 模型自动检测并使用 Flash Attention
python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001
```

**效果**：
- 推理速度提升：20-30%
- 显存占用降低：10-20%
- 要求：Ampere+ 架构 GPU（A100/A6000/RTX 3090+）

---

## 4. 监控与告警

### 4.1 关键指标收集

#### Prometheus 指标暴露

**Worker 指标** (需自定义添加)

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 指标定义
REQUEST_COUNT = Counter('worker_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('worker_request_latency_seconds', 'Request latency')
QUEUE_LENGTH = Gauge('worker_queue_length', 'Current queue length')
GPU_MEMORY = Gauge('worker_gpu_memory_mb', 'GPU memory usage')

# 在 Worker 中埋点
@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    # ... 推理逻辑 ...
    
    REQUEST_LATENCY.observe(time.time() - start_time)
    QUEUE_LENGTH.set(worker.get_queue_length())
    
# 启动 Prometheus 服务器
start_http_server(9090)
```

#### Grafana 监控面板

**关键面板**：
1. **吞吐量**：QPS 曲线
2. **延迟**：P50/P95/P99 分布
3. **队列长度**：实时队列深度
4. **GPU 利用率**：`nvidia-smi` 集成
5. **错误率**：按 error_code 分类

---

### 4.2 日志分析

#### 结构化日志配置

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": record.created,
            "level": record.levelname,
            "message": record.getMessage(),
            "worker_id": worker.worker_id,
            "queue_length": worker.get_queue_length(),
        }
        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

#### ELK Stack 集成

**Filebeat 配置** `filebeat.yml`

```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/fastchat/controller.log
      - /var/log/fastchat/model_worker_*.log
    json.keys_under_root: true
    json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "fastchat-%{+yyyy.MM.dd}"
```

---

### 4.3 告警规则

#### Alertmanager 规则

```yaml
groups:
  - name: fastchat
    rules:
      # Worker 失联告警
      - alert: WorkerDown
        expr: up{job="fastchat-worker"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Worker {{ $labels.instance }} is down"
      
      # 队列积压告警
      - alert: HighQueueLength
        expr: worker_queue_length > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Worker {{ $labels.instance }} queue length is {{ $value }}"
      
      # 高延迟告警
      - alert: HighLatency
        expr: histogram_quantile(0.95, worker_request_latency_seconds) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency is {{ $value }}s on {{ $labels.instance }}"
      
      # GPU OOM 告警
      - alert: GPUOutOfMemory
        expr: rate(worker_errors_total{error_code="50002"}[5m]) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GPU OOM errors on {{ $labels.instance }}"
```

---

## 5. 故障排查手册

### 5.1 Worker 无法注册

**症状**：Worker 启动后 Controller 日志无 "Register done"

**排查步骤**：

```bash
# 1. 检查 Controller 是否可达
curl http://localhost:21001/test_connection

# 2. 检查 Worker 日志
tail -f model_worker_*.log | grep -i error

# 3. 检查网络连通性
telnet localhost 21001

# 4. 检查防火墙
sudo iptables -L -n | grep 21001
```

**常见原因**：
- Controller 未启动：先启动 Controller
- `--worker-address` 配置错误：必须是 Controller 可访问的地址（不能用 localhost）
- 防火墙阻止：开放 21001、31000 等端口

**解决方案**：

```bash
# 正确的 Worker 启动命令（多机部署）
python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://controller-host:21001 \
    --worker http://$(hostname -I | awk '{print $1}'):31000 \  # 使用实际IP
    --host 0.0.0.0 \
    --port 31000
```

---

### 5.2 推理请求超时

**症状**：API 返回 `{"error_code": 50006, "text": "CONTROLLER_WORKER_TIMEOUT"}`

**排查步骤**：

```bash
# 1. 检查 Worker 状态
curl -X POST http://localhost:31000/worker_get_status

# 2. 检查队列长度
# 响应中 queue_length 是否持续 >= limit_worker_concurrency

# 3. 检查 GPU 利用率
nvidia-smi -l 1

# 4. 检查生成参数
# max_new_tokens 是否过大（如 >2048）
```

**常见原因**：
- Worker 过载：并发请求超过 `limit_worker_concurrency`
- 长序列生成：生成时间超过 100 秒
- GPU 利用率 100%：模型过大或批量过大

**解决方案**：

```bash
# 方案1：增加并发数（需更多显存）
--limit-worker-concurrency 10

# 方案2：增加超时时间
export FASTCHAT_WORKER_API_TIMEOUT=300

# 方案3：启动多个 Worker 水平扩展
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker ... --port 31000 &
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker ... --port 31001 &

# 方案4：使用 vLLM 加速
python3 -m fastchat.serve.vllm_worker ...
```

---

### 5.3 显存不足 (OOM)

**症状**：Worker 日志中出现 `CUDA out of memory`

**排查步骤**：

```bash
# 1. 检查显存使用
nvidia-smi

# 2. 估算显存需求
# FP16: 模型参数量 * 2 字节 + 上下文长度 * batch_size * hidden_size * 2

# 3. 检查是否有显存泄漏
# 多次推理后显存是否持续增长
```

**解决方案优先级**：

1. **启用 8bit 量化**（推荐，性价比最高）：
   ```bash
   --load-8bit
   ```

2. **减少并发数**：
   ```bash
   --limit-worker-concurrency 2
   ```

3. **减少 max_new_tokens**：
   ```python
   max_new_tokens=512  # 从 2048 降低到 512
   ```

4. **模型并行**（跨多张 GPU）：
   ```bash
   --num-gpus 2
   ```

5. **使用 GPTQ/AWQ 量化模型**：
   ```bash
   --model-path TheBloke/vicuna-13B-GPTQ
   ```

6. **CPU 卸载**（最后手段，性能损失大）：
   ```bash
   --load-8bit --cpu-offloading
   ```

---

### 5.4 输出质量问题

**症状**：生成内容重复、乱码、不相关

**排查步骤**：

```bash
# 1. 检查 Conversation 模板
curl -X POST http://localhost:31000/worker_get_conv_template

# 2. 检查模型是否正确加载
# Worker 日志中查找 "Loading the model"

# 3. 尝试不同生成参数
temperature=0.7  # 默认
temperature=1.0  # 更随机
temperature=0.0  # 贪心（可能重复）
```

**常见原因**：
- Conversation 模板与模型不匹配
- temperature=0 导致贪心解码重复
- prompt 格式错误

**解决方案**：

```python
# 明确指定 Conversation 模板
python3 -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --conv-template vicuna_v1.1 \  # 明确指定
    --controller http://localhost:21001

# 调整生成参数
{
  "temperature": 0.7,  # 0.5-1.0 之间
  "top_p": 0.9,        # 核采样
  "repetition_penalty": 1.1  # 惩罚重复
}
```

---

## 6. 安全加固

### 6.1 API Key 鉴权

```bash
python3 -m fastchat.serve.openai_api_server \
    --controller-address http://localhost:21001 \
    --api-keys key1 key2 key3 \  # 允许的 API Key 列表
    --port 8000
```

**客户端调用**：

```python
import openai

openai.api_key = "key1"  # 必须是允许的 Key
openai.api_base = "http://localhost:8000/v1"

response = openai.ChatCompletion.create(...)
```

---

### 6.2 HTTPS 加密

**使用 Nginx 反向代理**：

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

### 6.3 速率限制

**Nginx 配置**：

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

server {
    location /v1/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://localhost:8000;
    }
}
```

---

## 7. 案例研究

### 7.1 Chatbot Arena 生产部署

**规模**：
- 70+ 模型
- 日均 10 万+ 请求
- 峰值 QPS：100+

**架构**：
- 分布式 Controller 集群（高可用）
- 混合部署：本地 Worker + 第三方 API
- Nginx 负载均衡 + Cloudflare CDN

**经验教训**：
1. 使用 vLLM 加速本地模型（吞吐量提升 20 倍）
2. 第三方 API 配额管理（避免超限）
3. 监控告警完善（Prometheus + Grafana）

---

### 7.2 企业内部知识库问答

**需求**：
- 私有化部署
- 低延迟（P95 < 2s）
- 支持长文档（32K tokens）

**方案**：
- 模型：LongChat-7B-32K
- 加速：LightLLM 引擎
- 硬件：4 x A100-80GB

**效果**：
- P95 延迟：1.8 秒
- 吞吐量：50 QPS
- 成本：$0.02/1K tokens（自建）

---

## 8. 常见问题 FAQ

**Q1：FastChat 支持哪些模型？**

A：支持所有 HuggingFace Transformers 兼容的模型，包括：
- Llama 系列（Vicuna、Alpaca、LongChat）
- ChatGLM
- Falcon
- Baichuan
- 更多见：`fastchat/model/model_adapter.py`

**Q2：如何添加自定义模型？**

A：实现 `BaseModelAdapter` 并注册：

```python
from fastchat.model.model_adapter import BaseModelAdapter, register_model_adapter

class CustomAdapter(BaseModelAdapter):
    def match(self, model_path):
        return "custom-model" in model_path
    
    def load_model(self, model_path, kwargs):
        # 自定义加载逻辑
        pass

register_model_adapter(CustomAdapter)
```

**Q3：FastChat 与 vLLM/TGI 的区别？**

| 维度 | FastChat | vLLM | TGI |
|---|---|---|---|
| 定位 | 完整服务平台 | 高性能推理引擎 | HF 官方推理服务 |
| 吞吐量 | 中等（标准模式） | 极高 | 高 |
| 易用性 | 高（开箱即用） | 中等 | 高 |
| 扩展性 | 高（支持多种引擎） | 低 | 中 |

**Q4：如何优化首 Token 延迟？**

A：
1. 使用 Flash Attention
2. 减少 prompt 长度
3. 使用 SGLang/vLLM（支持 prefix caching）
4. 启用 KV Cache

---

## 9. 附录：完整启动脚本

```bash
#!/bin/bash
# FastChat 生产部署脚本

set -e

# 配置
CONTROLLER_PORT=21001
API_SERVER_PORT=8000
WEB_SERVER_PORT=7860
MODEL_PATH="lmsys/vicuna-7b-v1.5"
WORKER_PORTS=(31000 31001 31002)
GPUS=(0 1 2)

# 启动 Controller
echo "Starting Controller..."
python3 -m fastchat.serve.controller --port $CONTROLLER_PORT > controller.log 2>&1 &
CONTROLLER_PID=$!
sleep 5

# 启动 Workers
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    PORT=${WORKER_PORTS[$i]}
    echo "Starting Worker $i on GPU $GPU, port $PORT..."
    
    CUDA_VISIBLE_DEVICES=$GPU python3 -m fastchat.serve.model_worker \
        --model-path $MODEL_PATH \
        --controller http://localhost:$CONTROLLER_PORT \
        --worker http://localhost:$PORT \
        --port $PORT \
        --limit-worker-concurrency 5 \
        > worker_$i.log 2>&1 &
    
    sleep 15
done

# 启动 API Server
echo "Starting OpenAI API Server..."
python3 -m fastchat.serve.openai_api_server \
    --controller-address http://localhost:$CONTROLLER_PORT \
    --port $API_SERVER_PORT \
    > api_server.log 2>&1 &
API_SERVER_PID=$!

# 启动 Web Server
echo "Starting Gradio Web Server..."
python3 -m fastchat.serve.gradio_web_server \
    --controller-address http://localhost:$CONTROLLER_PORT \
    --port $WEB_SERVER_PORT \
    > web_server.log 2>&1 &
WEB_SERVER_PID=$!

echo "All services started!"
echo "Controller PID: $CONTROLLER_PID"
echo "API Server PID: $API_SERVER_PID"
echo "Web Server PID: $WEB_SERVER_PID"
echo ""
echo "Access Web UI: http://localhost:$WEB_SERVER_PORT"
echo "API Endpoint: http://localhost:$API_SERVER_PORT/v1"
echo ""
echo "To stop all services: kill $CONTROLLER_PID $API_SERVER_PID $WEB_SERVER_PID"
```

**使用方法**：

```bash
chmod +x start_fastchat.sh
./start_fastchat.sh
```

---

**文档完成**。本文档涵盖了 FastChat 的快速开始、生产部署、性能调优、监控告警、故障排查和安全加固的完整实践指南。
