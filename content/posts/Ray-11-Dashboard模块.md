---
title: "Ray-11-Dashboard模块"
date: 2024-12-28T11:11:00+08:00
series: ["Ray源码剖析"]
categories: ['Ray']
tags: ['Ray', '源码剖析', '分布式计算', '机器学习', '监控面板', '可视化', '系统监控']
description: "Ray Dashboard模块模块源码剖析 - 详细分析Dashboard模块模块的架构设计、核心功能和实现机制"
---


# Ray-11-Dashboard模块（监控面板）

## 模块概览

Ray Dashboard是Ray的Web UI，提供集群状态可视化、任务监控和日志查看。

### 核心功能

- **集群概览**：节点状态、资源使用
- **任务监控**：任务执行、Actor状态
- **日志查看**：实时日志、错误追踪
- **性能分析**：调用栈、时间线
- **Job管理**：Job提交、状态查询

## 访问Dashboard

```bash
# 启动集群
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# 访问
# http://localhost:8265
```

## 主要页面

### 1. Cluster页面

显示集群整体状态：
- 节点列表（IP、资源、状态）
- 资源使用趋势图（CPU、GPU、内存、Object Store）
- 集群拓扑图

### 2. Jobs页面

管理Ray Jobs：
- Job列表（ID、状态、提交时间）
- Job详情（日志、指标）
- 提交新Job

```python
# 提交Job
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://localhost:8265")

job_id = client.submit_job(
    entrypoint="python train.py",
    runtime_env={"pip": ["torch"]},
)

# 查询状态
status = client.get_job_status(job_id)
print(status)

# 获取日志
logs = client.get_job_logs(job_id)
print(logs)
```

### 3. Actors页面

监控Actor状态：
- Actor列表（ID、类名、状态、资源）
- Actor详情（调用统计、日志）
- Actor重启历史

### 4. Tasks页面

查看任务执行：
- 任务列表（ID、函数名、状态）
- 任务依赖图
- 任务执行时间分布

### 5. Logs页面

查看集群日志：
- 按节点筛选
- 按组件筛选（Raylet、GCS、Worker）
- 日志搜索和过滤

### 6. Metrics页面

查看Prometheus指标：
- 自定义指标查询
- Grafana集成

## 编程接口

### State API

```python
from ray.util.state import list_actors, list_tasks, list_nodes

# 列出所有Actors
actors = list_actors()
for actor in actors:
    print(f"Actor {actor['actor_id']}: {actor['state']}")

# 列出所有Tasks
tasks = list_tasks()
for task in tasks:
    print(f"Task {task['task_id']}: {task['state']}")

# 列出所有节点
nodes = list_nodes()
for node in nodes:
    print(f"Node {node['node_id']}: CPU={node['cpu']}, GPU={node['gpu']}")
```

### Job API

```python
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://localhost:8265")

# 提交Job
job_id = client.submit_job(
    entrypoint="python script.py",
    submission_id="my-job-001",  # 可选，幂等性
    runtime_env={
        "working_dir": "s3://bucket/code.zip",
        "pip": ["pandas", "numpy"]
    },
    metadata={"owner": "alice"}
)

# 停止Job
client.stop_job(job_id)

# 删除Job
client.delete_job(job_id)

# 列出所有Jobs
jobs = client.list_jobs()
for job in jobs:
    print(f"Job {job.submission_id}: {job.status}")
```

## 性能分析

### Timeline Profiling

```python
import ray

@ray.remote
def task():
    # 任务逻辑
    pass

# 执行任务
ray.get([task.remote() for _ in range(100)])

# 导出Timeline
ray.timeline(filename="timeline.json")
```

在Chrome浏览器打开`chrome://tracing`，加载`timeline.json`查看任务执行时间线。

### Stack Trace

```python
# 获取Worker的调用栈
from ray.util.state import get_actor

actor_info = get_actor("actor_id")
stack_trace = actor_info["state"]["stack_trace"]
print(stack_trace)
```

## 自定义指标

```python
from ray.util.metrics import Counter, Histogram, Gauge

# Counter：累加
requests_total = Counter(
    "requests_total",
    description="Total number of requests",
    tag_keys=("method", "status")
)
requests_total.inc(tags={"method": "GET", "status": "200"})

# Histogram：分布
request_duration = Histogram(
    "request_duration_seconds",
    description="Request duration in seconds",
    boundaries=[0.1, 0.5, 1.0, 5.0]
)
request_duration.observe(0.3)

# Gauge：瞬时值
queue_size = Gauge(
    "queue_size",
    description="Number of items in queue"
)
queue_size.set(42)
```

访问`http://localhost:8265/metrics`查看Prometheus格式指标。

## 与Prometheus/Grafana集成

### Prometheus配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ray'
    static_configs:
      - targets: ['localhost:8265']
```

### Grafana Dashboard

Ray提供官方Grafana Dashboard：
1. 导入Dashboard ID: `13997`
2. 配置Prometheus数据源
3. 查看集群指标

## 安全配置

```bash
# 启用认证
ray start --head \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --dashboard-ssl \
  --dashboard-ssl-certfile=/path/to/cert.pem \
  --dashboard-ssl-keyfile=/path/to/key.pem
```

## 故障排查

### 常见问题

1. **Dashboard无法访问**
   ```bash
   # 检查Dashboard进程
   ps aux | grep dashboard
   
   # 查看Dashboard日志
   tail -f /tmp/ray/session_latest/logs/dashboard.log
   ```

2. **指标不更新**
   ```bash
   # 确认metrics端口开放
   curl http://localhost:8265/metrics
   ```

3. **性能问题**
   ```python
   # 禁用Dashboard（生产环境可选）
   ray.init(include_dashboard=False)
   ```

## 最佳实践

### 1. 生产环境监控

```bash
# 持久化日志
ray start --head \
  --dashboard-host=0.0.0.0 \
  --log-to-driver=False \
  --logging-rotate-bytes=100000000 \
  --logging-rotate-backup-count=10
```

### 2. 定期导出指标

```python
import requests
import time

while True:
    metrics = requests.get("http://localhost:8265/metrics").text
    with open(f"metrics_{time.time()}.txt", "w") as f:
        f.write(metrics)
    time.sleep(60)
```

### 3. 告警配置

```yaml
# prometheus alerts
groups:
  - name: ray
    rules:
      - alert: RayNodeDown
        expr: ray_node_count < 3
        for: 5m
        annotations:
          summary: "Ray cluster has fewer than 3 nodes"
```

## 总结

Ray Dashboard提供全面的集群监控能力，关键特性：

1. **可视化**：直观展示集群状态
2. **实时监控**：任务、Actor、资源
3. **日志聚合**：统一查看分布式日志
4. **性能分析**：Timeline、Stack Trace
5. **可扩展**：自定义指标、告警集成

