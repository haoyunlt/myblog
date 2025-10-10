---
title: "VoiceHelper源码剖析 - 12Infrastructure基础设施"
date: 2025-10-10T12:00:00+08:00
draft: false
tags: ["源码剖析", "VoiceHelper", "基础设施", "Kubernetes", "监控", "CI/CD"]
categories: ["VoiceHelper", "源码剖析"]
description: "Infrastructure基础设施详解：Kubernetes容器编排、Prometheus+Grafana监控、ELK日志聚合、Consul服务发现、CI/CD"
weight: 13
---

# VoiceHelper-12-Infrastructure基础设施

## 1. 概览

VoiceHelper基础设施层提供容器化部署、监控告警、日志聚合、服务发现等功能,确保系统高可用和可观测。

**核心组件**:
- **Kubernetes**:容器编排和服务部署
- **Monitoring**:Prometheus+Grafana监控体系
- **Logging**:ELK/Loki日志聚合
- **Service Discovery**:Consul服务发现
- **Message Queue**:RabbitMQ消息队列
- **Object Storage**:MinIO对象存储
- **CI/CD**:GitHub Actions持续集成

---

## 2. Kubernetes部署

### 2.1 架构设计

```yaml
# infrastructure/kubernetes/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: voicehelper
spec:
  replicas: 3                    # 3个副本实现高可用
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: gateway
        image: voicehelper/api-gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretRef:           # 敏感信息使用Secret
              name: db-secret
              key: url
        resources:
          requests:              # 资源请求
            memory: "256Mi"
            cpu: "250m"
          limits:                # 资源限制
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:           # 存活探针
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:          # 就绪探针
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 2.2 Service配置

```yaml
# infrastructure/kubernetes/service.yaml

apiVersion: v1
kind: Service
metadata:
  name: api-gateway
  namespace: voicehelper
spec:
  type: LoadBalancer         # 负载均衡器类型
  selector:
    app: api-gateway
  ports:
  - protocol: TCP
    port: 80                 # 外部端口
    targetPort: 8080         # 容器端口
  sessionAffinity: ClientIP  # 会话亲和性
```

### 2.3 ConfigMap与Secret

```yaml
# ConfigMap - 非敏感配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  
---
# Secret - 敏感信息
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  url: cG9zdGdyZXNxbDovL...    # Base64编码
  password: cGFzc3dvcmQxMjM=
```

---

## 3. 监控体系

### 3.1 Prometheus配置

```yaml
# infrastructure/monitoring/prometheus.yml

global:
  scrape_interval: 15s          # 采集间隔
  evaluation_interval: 15s      # 规则评估间隔

scrape_configs:
  - job_name: 'api-gateway'
    static_configs:
      - targets: ['gateway:8080']
    metrics_path: /metrics
    
  - job_name: 'graphrag-service'
    static_configs:
      - targets: ['graphrag:8001']
    
  - job_name: 'llm-router'
    static_configs:
      - targets: ['llm-router:8005']

# 告警规则
rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### 3.2 告警规则

```yaml
# infrastructure/monitoring/alerts.yml

groups:
  - name: service_alerts
    rules:
    # 服务不可用告警
    - alert: ServiceDown
      expr: up == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "服务{{ $labels.job }}不可用"
        description: "服务已停止响应超过1分钟"
    
    # 高错误率告警
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "高错误率检测"
        description: "5xx错误率超过5%"
    
    # 高延迟告警
    - alert: HighLatency
      expr: histogram_quantile(0.95, http_request_duration_seconds) > 1
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "高延迟检测"
        description: "P95延迟超过1秒"
```

### 3.3 Grafana仪表盘

**关键指标**:
1. **QPS**(每秒查询数)
2. **延迟分布**(P50/P95/P99)
3. **错误率**(4xx/5xx)
4. **资源使用**(CPU/内存/磁盘)
5. **依赖健康**(数据库/Redis/外部API)

---

## 4. 日志聚合

### 4.1 ELK Stack

**架构**:
```
应用日志 → Filebeat → Logstash → Elasticsearch → Kibana
```

**Filebeat配置**:
```yaml
# infrastructure/logging/filebeat.yml

filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/voicehelper/*.log
    fields:
      service: voicehelper
      environment: production

output.logstash:
  hosts: ["logstash:5044"]
  
processors:
  - add_cloud_metadata: ~
  - add_docker_metadata: ~
```

**Logstash Pipeline**:
```ruby
# infrastructure/logging/logstash.conf

input {
  beats {
    port => 5044
  }
}

filter {
  # 解析JSON日志
  json {
    source => "message"
  }
  
  # 提取时间戳
  date {
    match => ["timestamp", "ISO8601"]
  }
  
  # 添加标签
  mutate {
    add_field => { "parsed" => "true" }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "voicehelper-%{+YYYY.MM.dd}"
  }
}
```

---

## 5. 服务发现

### 5.1 Consul配置

```hcl
# infrastructure/consul/config.hcl

datacenter = "dc1"
data_dir = "/opt/consul"
log_level = "INFO"

server = true
bootstrap_expect = 3    # 3节点集群

ui_config {
  enabled = true
}

# 健康检查
checks = [
  {
    name = "Memory Usage"
    args = ["/usr/local/bin/check_mem.sh"]
    interval = "30s"
  }
]
```

**服务注册**:
```go
// Go服务注册到Consul
func RegisterService(consul *api.Client, serviceName string, port int) error {
    registration := &api.AgentServiceRegistration{
        ID:      fmt.Sprintf("%s-%s", serviceName, hostname),
        Name:    serviceName,
        Port:    port,
        Address: localIP,
        Check: &api.AgentServiceCheck{
            HTTP:     fmt.Sprintf("http://%s:%d/health", localIP, port),
            Interval: "10s",
            Timeout:  "5s",
        },
    }
    
    return consul.Agent().ServiceRegister(registration)
}
```

---

## 6. 消息队列

### 6.1 RabbitMQ配置

```python
# Python生产者
import pika

connection = pika.BlockingConnection(
    pika.ConnectionParameters('rabbitmq')
)
channel = connection.channel()

# 声明队列
channel.queue_declare(
    queue='task_queue',
    durable=True         # 持久化队列
)

# 发送消息
channel.basic_publish(
    exchange='',
    routing_key='task_queue',
    body=message,
    properties=pika.BasicProperties(
        delivery_mode=2  # 持久化消息
    )
)
```

```python
# Python消费者
def callback(ch, method, properties, body):
    """消息处理回调"""
    print(f"收到消息: {body}")
    # 处理消息...
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)  # 公平调度
channel.basic_consume(
    queue='task_queue',
    on_message_callback=callback
)

channel.start_consuming()
```

---

## 7. 对象存储

### 7.1 MinIO配置

```yaml
# infrastructure/minio/docker-compose.yml

version: '3.8'
services:
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: password123
    ports:
      - "9000:9000"     # API端口
      - "9001:9001"     # Console端口
    volumes:
      - minio_data:/data
```

**Python客户端**:
```python
from minio import Minio

# 初始化客户端
client = Minio(
    "minio:9000",
    access_key="admin",
    secret_key="password123",
    secure=False
)

# 创建bucket
if not client.bucket_exists("documents"):
    client.make_bucket("documents")

# 上传文件
client.fput_object(
    "documents",
    "file.pdf",
    "/path/to/file.pdf",
    content_type="application/pdf"
)

# 下载文件
client.fget_object(
    "documents",
    "file.pdf",
    "/path/to/download.pdf"
)
```

---

## 8. CI/CD

### 8.1 GitHub Actions

```yaml
# .github/workflows/ci.yml

name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/
      
      - name: Lint code
        run: |
          flake8 .
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: |
          docker build -t voicehelper/api-gateway:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          docker push voicehelper/api-gateway:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/api-gateway \
            gateway=voicehelper/api-gateway:${{ github.sha }}
```

---

## 9. 高可用架构

### 9.1 多副本部署

```yaml
# 每个服务至少3个副本
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphrag-service
spec:
  replicas: 3
  strategy:
    type: RollingUpdate     # 滚动更新
    rollingUpdate:
      maxSurge: 1           # 最多额外1个pod
      maxUnavailable: 1     # 最多不可用1个pod
```

### 9.2 Pod反亲和性

```yaml
# 确保pod分布在不同节点
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - api-gateway
        topologyKey: "kubernetes.io/hostname"
```

### 9.3 HPA自动扩缩容

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 10. 灾备方案

### 10.1 数据库备份

```bash
#!/bin/bash
# 定时备份PostgreSQL

BACKUP_DIR="/backup/postgres"
DATE=$(date +%Y%m%d_%H%M%S)

# 备份数据库
pg_dump -U postgres -h localhost voicehelper > \
  $BACKUP_DIR/voicehelper_$DATE.sql

# 压缩备份
gzip $BACKUP_DIR/voicehelper_$DATE.sql

# 删除30天前的备份
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

# 上传到OSS
aws s3 cp $BACKUP_DIR/voicehelper_$DATE.sql.gz \
  s3://backup-bucket/postgres/
```

### 10.2 Redis持久化

```conf
# redis.conf

# RDB持久化
save 900 1      # 900秒内至少1个key变化
save 300 10     # 300秒内至少10个key变化
save 60 10000   # 60秒内至少10000个key变化

# AOF持久化
appendonly yes
appendfsync everysec    # 每秒fsync一次
```

---

## 11. 总结

VoiceHelper基础设施提供了完善的DevOps支持:

1. **Kubernetes**:容器编排,3副本高可用,HPA自动扩缩容
2. **Prometheus+Grafana**:全面监控,实时告警
3. **ELK Stack**:集中式日志,便于问题排查
4. **Consul**:服务发现和健康检查
5. **MinIO**:对象存储,支持海量文件
6. **RabbitMQ**:异步任务处理
7. **CI/CD**:GitHub Actions自动化部署

通过完善的基础设施,VoiceHelper实现了高可用、可观测、易扩展的生产级架构。

---

**文档状态**:✅ 已完成  
**覆盖度**:100%(K8s、监控、日志、服务发现、消息队列、CI/CD)  
**下一步**:生成框架使用示例与最佳实践文档(13-框架使用示例与最佳实践)

