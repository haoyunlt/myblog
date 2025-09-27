---
title: "MinIO 实战部署运维指南"
date: 2024-12-19T10:00:00+08:00
draft: false
tags: ['源码分析', '技术文档']
categories: ["mimio", "技术分析"]
description: "深入分析 MinIO 实战部署运维指南 的技术实现和架构设计"
weight: 700
slug: "mimio-实战部署运维指南"
---

# MinIO 实战部署运维指南

---

## 1. 生产环境部署

### 1.1 硬件规划

#### 推荐配置

**小型部署 (< 100TB)**
- CPU: 8 核心以上
- 内存: 32GB 以上
- 网络: 10Gbps
- 磁盘: 4-8 块 SATA SSD 或 NVMe
- 节点数: 4 节点

**中型部署 (100TB - 1PB)**
- CPU: 16 核心以上
- 内存: 64GB 以上
- 网络: 25Gbps
- 磁盘: 8-12 块 NVMe SSD
- 节点数: 8-16 节点

**大型部署 (> 1PB)**
- CPU: 32 核心以上
- 内存: 128GB 以上
- 网络: 100Gbps
- 磁盘: 12-24 块 NVMe SSD
- 节点数: 16+ 节点

#### 磁盘配置建议

```bash
# 磁盘格式化（XFS 推荐）
mkfs.xfs -f -i size=512 /dev/nvme0n1

# 挂载选项优化
mount -o noatime,nodiratime,nobarrier,inode64 /dev/nvme0n1 /mnt/disk1

# /etc/fstab 配置
/dev/nvme0n1 /mnt/disk1 xfs defaults,noatime,nodiratime,nobarrier,inode64 0 2
```

### 1.2 分布式部署

#### Docker Compose 部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  minio1:
    image: minio/minio:latest
    hostname: minio1
    volumes:
      - /mnt/disk1:/data1
      - /mnt/disk2:/data2
      - /mnt/disk3:/data3
      - /mnt/disk4:/data4
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
      MINIO_PROMETHEUS_AUTH_TYPE: public
    command: server --console-address ":9001" http://minio{1...4}/data{1...4}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio2:
    image: minio/minio:latest
    hostname: minio2
    volumes:
      - /mnt/disk1:/data1
      - /mnt/disk2:/data2
      - /mnt/disk3:/data3
      - /mnt/disk4:/data4
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
      MINIO_PROMETHEUS_AUTH_TYPE: public
    command: server --console-address ":9001" http://minio{1...4}/data{1...4}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  # 类似配置 minio3, minio4...

networks:
  default:
    driver: bridge
```

#### Kubernetes 部署

```yaml
# minio-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: minio
  namespace: minio-system
spec:
  serviceName: minio-headless
  replicas: 4
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        image: minio/minio:latest
        args:
        - server
        - --console-address
        - ":9001"
        - http://minio-{0...3}.minio-headless.minio-system.svc.cluster.local/data{1...4}
        env:
        - name: MINIO_ROOT_USER
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: root-user
        - name: MINIO_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: root-password
        - name: MINIO_PROMETHEUS_AUTH_TYPE
          value: "public"
        ports:
        - containerPort: 9000
          name: api
        - containerPort: 9001
          name: console
        volumeMounts:
        - name: data1
          mountPath: /data1
        - name: data2
          mountPath: /data2
        - name: data3
          mountPath: /data3
        - name: data4
          mountPath: /data4
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /minio/health/live
            port: 9000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /minio/health/ready
            port: 9000
          initialDelaySeconds: 10
          periodSeconds: 10
  volumeClaimTemplates:
  - metadata:
      name: data1
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 1Ti
  - metadata:
      name: data2
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 1Ti
  - metadata:
      name: data3
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 1Ti
  - metadata:
      name: data4
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 1Ti

---
apiVersion: v1
kind: Service
metadata:
  name: minio-headless
  namespace: minio-system
spec:
  clusterIP: None
  selector:
    app: minio
  ports:
  - port: 9000
    name: api
  - port: 9001
    name: console

---
apiVersion: v1
kind: Service
metadata:
  name: minio-service
  namespace: minio-system
spec:
  type: LoadBalancer
  selector:
    app: minio
  ports:
  - port: 9000
    targetPort: 9000
    name: api
  - port: 9001
    targetPort: 9001
    name: console
```

### 1.3 负载均衡配置

#### Nginx 配置

```nginx
# /etc/nginx/sites-available/minio
upstream minio_backend {
    least_conn;
    server minio1:9000 max_fails=3 fail_timeout=30s;
    server minio2:9000 max_fails=3 fail_timeout=30s;
    server minio3:9000 max_fails=3 fail_timeout=30s;
    server minio4:9000 max_fails=3 fail_timeout=30s;
}

upstream minio_console {
    least_conn;
    server minio1:9001 max_fails=3 fail_timeout=30s;
    server minio2:9001 max_fails=3 fail_timeout=30s;
    server minio3:9001 max_fails=3 fail_timeout=30s;
    server minio4:9001 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name minio.example.com;

    # SSL 配置
    ssl_certificate /etc/ssl/certs/minio.crt;
    ssl_certificate_key /etc/ssl/private/minio.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    # 客户端最大上传大小
    client_max_body_size 1000m;

    # API 端点
    location / {
        proxy_pass http://minio_backend;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 长连接配置
        proxy_connect_timeout 300;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # 缓冲区配置
        proxy_buffering off;
        proxy_request_buffering off;
    }
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name console.minio.example.com;

    # SSL 配置
    ssl_certificate /etc/ssl/certs/minio.crt;
    ssl_certificate_key /etc/ssl/private/minio.key;

    # 控制台
    location / {
        proxy_pass http://minio_console;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### HAProxy 配置

```haproxy
# /etc/haproxy/haproxy.cfg
global
    daemon
    maxconn 4096
    log stdout local0

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull

frontend minio_frontend
    bind *:9000
    bind *:9443 ssl crt /etc/ssl/certs/minio.pem
    redirect scheme https if !{ ssl_fc }
    default_backend minio_backend

backend minio_backend
    balance roundrobin
    option httpchk GET /minio/health/live
    http-check expect status 200
    server minio1 minio1:9000 check inter 30s
    server minio2 minio2:9000 check inter 30s
    server minio3 minio3:9000 check inter 30s
    server minio4 minio4:9000 check inter 30s

frontend minio_console_frontend
    bind *:9001
    bind *:9444 ssl crt /etc/ssl/certs/minio.pem
    default_backend minio_console_backend

backend minio_console_backend
    balance roundrobin
    server minio1 minio1:9001 check inter 30s
    server minio2 minio2:9001 check inter 30s
    server minio3 minio3:9001 check inter 30s
    server minio4 minio4:9001 check inter 30s
```

---

## 2. 性能调优

### 2.1 系统级优化

#### 内核参数调优

```bash
# /etc/sysctl.conf
# 网络优化
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 30000
net.ipv4.tcp_max_syn_backlog = 30000

# 文件系统优化
fs.file-max = 1000000
fs.nr_open = 1000000

# 虚拟内存优化
vm.swappiness = 1
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# 应用生效
sysctl -p
```

#### 文件描述符限制

```bash
# /etc/security/limits.conf
* soft nofile 1000000
* hard nofile 1000000
* soft nproc 1000000
* hard nproc 1000000

# /etc/systemd/system.conf
DefaultLimitNOFILE=1000000
DefaultLimitNPROC=1000000
```

#### 磁盘 I/O 调度器优化

```bash
# 对于 SSD，使用 noop 或 deadline
echo noop > /sys/block/nvme0n1/queue/scheduler

# 对于机械硬盘，使用 cfq
echo cfq > /sys/block/sda/queue/scheduler

# 永久设置
echo 'ACTION=="add|change", KERNEL=="nvme[0-9]*", ATTR{queue/scheduler}="noop"' > /etc/udev/rules.d/60-ssd-scheduler.rules
```

### 2.2 MinIO 配置优化

#### 环境变量配置

```bash
# 性能相关环境变量
export MINIO_API_REQUESTS_MAX=10000
export MINIO_API_REQUESTS_DEADLINE=10s
export MINIO_API_CORS_ALLOW_ORIGIN="*"

# 缓存配置
export MINIO_CACHE_DRIVES="/mnt/cache1,/mnt/cache2"
export MINIO_CACHE_EXCLUDE="*.tmp,*.log"
export MINIO_CACHE_QUOTA=80
export MINIO_CACHE_AFTER=3
export MINIO_CACHE_WATERMARK_LOW=70
export MINIO_CACHE_WATERMARK_HIGH=90

# 压缩配置
export MINIO_COMPRESS_ENABLE=on
export MINIO_COMPRESS_EXTENSIONS=".txt,.log,.csv,.json,.tar,.xml,.bin"
export MINIO_COMPRESS_MIME_TYPES="text/*,application/json,application/xml"

# 批处理配置
export MINIO_BATCH_EXPIRATION_WORKERS=100
export MINIO_BATCH_REPLICATION_WORKERS=100

# 扫描器配置
export MINIO_SCANNER_SPEED=default
export MINIO_SCANNER_IDLE_SPEED=default
```

#### 存储类配置

```bash
# 设置默认存储类
mc admin config set myminio storage_class standard=EC:4

# 设置减少冗余存储类
mc admin config set myminio storage_class rrs=EC:2

# 应用配置
mc admin service restart myminio
```

### 2.3 客户端优化

#### Go 客户端优化

```go
package main

import (
    "context"
    "log"
    "net/http"
    "time"
    
    "github.com/minio/minio-go/v7"
    "github.com/minio/minio-go/v7/pkg/credentials"
)

func main() {
    // 优化的 HTTP 传输配置
    transport := &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 100,
        IdleConnTimeout:     90 * time.Second,
        DisableCompression:  true, // MinIO 已处理压缩
        WriteBufferSize:     32 * 1024,
        ReadBufferSize:      32 * 1024,
    }
    
    // 创建 MinIO 客户端
    minioClient, err := minio.New("minio.example.com:9000", &minio.Options{
        Creds:     credentials.NewStaticV4("ACCESS_KEY", "SECRET_KEY", ""),
        Secure:    true,
        Transport: transport,
    })
    if err != nil {
        log.Fatalln(err)
    }
    
    // 并行上传示例
    uploadFile := func(bucketName, objectName, filePath string) error {
        ctx := context.Background()
        
        // 使用分片上传优化大文件
        _, err := minioClient.FPutObject(ctx, bucketName, objectName, filePath, minio.PutObjectOptions{
            ContentType:  "application/octet-stream",
            PartSize:     64 * 1024 * 1024, // 64MB 分片
            NumThreads:   4,                 // 并行线程数
        })
        return err
    }
    
    // 批量上传
    for i := 0; i < 100; i++ {
        go uploadFile("mybucket", fmt.Sprintf("object-%d", i), fmt.Sprintf("/path/to/file-%d", i))
    }
}
```

#### Python 客户端优化

```python
from minio import Minio
from minio.error import S3Error
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

# 禁用 SSL 警告（如果使用自签名证书）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 创建客户端
client = Minio(
    "minio.example.com:9000",
    access_key="ACCESS_KEY",
    secret_key="SECRET_KEY",
    secure=True,
    http_client=urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=60, read=300),
        maxsize=100,
        retries=urllib3.Retry(
            total=5,
            backoff_factor=0.2,
            status_forcelist=[500, 502, 503, 504]
        )
    )
)

def upload_file(bucket_name, object_name, file_path):
    """优化的文件上传函数"""
    try:
        # 使用分片上传
        result = client.fput_object(
            bucket_name,
            object_name,
            file_path,
            part_size=64*1024*1024,  # 64MB 分片
            num_parallel_uploads=4,   # 并行上传数
        )
        return f"Upload successful: {result.etag}"
    except S3Error as err:
        return f"Upload failed: {err}"

# 并行上传示例
def batch_upload(files):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for bucket, object_name, file_path in files:
            future = executor.submit(upload_file, bucket, object_name, file_path)
            futures.append(future)
        
        for future in as_completed(futures):
            print(future.result())
```

---

## 3. 监控告警

### 3.1 Prometheus 监控

#### MinIO 指标配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'minio'
    metrics_path: /minio/v2/metrics/cluster
    scheme: http
    static_configs:
      - targets: ['minio1:9000', 'minio2:9000', 'minio3:9000', 'minio4:9000']
    scrape_interval: 30s
    scrape_timeout: 10s

  - job_name: 'minio-node'
    metrics_path: /minio/v2/metrics/node
    scheme: http
    static_configs:
      - targets: ['minio1:9000', 'minio2:9000', 'minio3:9000', 'minio4:9000']
    scrape_interval: 30s
```

#### 关键指标监控

```yaml
# alerts.yml
groups:
- name: minio
  rules:
  # 磁盘使用率告警
  - alert: MinioDiskUsageHigh
    expr: (minio_disk_storage_used_bytes / minio_disk_storage_total_bytes) * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "MinIO disk usage is high"
      description: "MinIO disk usage is {{ $value }}% on {{ $labels.instance }}"

  # 磁盘离线告警
  - alert: MinioDiskOffline
    expr: minio_disk_storage_available == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "MinIO disk is offline"
      description: "MinIO disk {{ $labels.disk }} is offline on {{ $labels.instance }}"

  # API 错误率告警
  - alert: MinioAPIErrorRateHigh
    expr: rate(minio_s3_requests_errors_total[5m]) / rate(minio_s3_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "MinIO API error rate is high"
      description: "MinIO API error rate is {{ $value | humanizePercentage }} on {{ $labels.instance }}"

  # 节点离线告警
  - alert: MinioNodeDown
    expr: up{job="minio"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "MinIO node is down"
      description: "MinIO node {{ $labels.instance }} is down"

  # 修复操作告警
  - alert: MinioHealingInProgress
    expr: minio_heal_objects_heal_total > 0
    for: 10m
    labels:
      severity: info
    annotations:
      summary: "MinIO healing in progress"
      description: "MinIO is healing {{ $value }} objects on {{ $labels.instance }}"
```

### 3.2 Grafana 仪表板

#### 核心指标面板

```json
{
  "dashboard": {
    "title": "MinIO Cluster Overview",
    "panels": [
      {
        "title": "Cluster Storage Usage",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(minio_disk_storage_used_bytes) / sum(minio_disk_storage_total_bytes) * 100",
            "legendFormat": "Usage %"
          }
        ]
      },
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(minio_s3_requests_total[5m])) by (api)",
            "legendFormat": "{{ api }}"
          }
        ]
      },
      {
        "title": "Request Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(minio_s3_requests_duration_seconds_bucket[5m])) by (le, api))",
            "legendFormat": "95th percentile - {{ api }}"
          }
        ]
      },
      {
        "title": "Disk Status",
        "type": "table",
        "targets": [
          {
            "expr": "minio_disk_storage_available",
            "format": "table"
          }
        ]
      }
    ]
  }
}
```

### 3.3 日志监控

#### ELK Stack 配置

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/minio/*.log
  fields:
    service: minio
    environment: production
  multiline.pattern: '^\d{4}-\d{2}-\d{2}'
  multiline.negate: true
  multiline.match: after

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "minio-logs-%{+yyyy.MM.dd}"

processors:
- add_host_metadata:
    when.not.contains.tags: forwarded
```

```json
# Logstash 配置
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "minio" {
    grok {
      match => { 
        "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" 
      }
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if "API" in [message] {
      grok {
        match => { 
          "message" => "API: %{WORD:method} %{URIPATH:path} %{NUMBER:status_code:int} %{NUMBER:response_time:float}ms" 
        }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "minio-logs-%{+YYYY.MM.dd}"
  }
}
```

---

## 4. 故障排查

### 4.1 常见问题诊断

#### 磁盘故障诊断

```bash
#!/bin/bash
# minio-disk-check.sh

echo "=== MinIO 磁盘健康检查 ==="

# 检查磁盘挂载状态
echo "1. 检查磁盘挂载:"
df -h | grep -E "(mnt|data)"

# 检查磁盘 I/O 状态
echo -e "\n2. 检查磁盘 I/O:"
iostat -x 1 3

# 检查磁盘错误
echo -e "\n3. 检查磁盘错误:"
dmesg | grep -i error | tail -10

# 检查 SMART 状态
echo -e "\n4. 检查 SMART 状态:"
for disk in /dev/sd* /dev/nvme*; do
    if [ -e "$disk" ]; then
        echo "=== $disk ==="
        smartctl -H "$disk" 2>/dev/null || echo "SMART not available"
    fi
done

# 检查文件系统错误
echo -e "\n5. 检查文件系统:"
for mount in $(df | grep -E "(mnt|data)" | awk '{print $6}'); do
    echo "=== $mount ==="
    fsck -n "$mount" 2>/dev/null || echo "Cannot check $mount"
done
```

#### 网络连接诊断

```bash
#!/bin/bash
# minio-network-check.sh

MINIO_NODES=("minio1" "minio2" "minio3" "minio4")
MINIO_PORT=9000

echo "=== MinIO 网络连接检查 ==="

for node in "${MINIO_NODES[@]}"; do
    echo "检查节点: $node"
    
    # 检查 DNS 解析
    if nslookup "$node" >/dev/null 2>&1; then
        echo "  ✓ DNS 解析正常"
    else
        echo "  ✗ DNS 解析失败"
        continue
    fi
    
    # 检查端口连通性
    if nc -z "$node" "$MINIO_PORT" 2>/dev/null; then
        echo "  ✓ 端口 $MINIO_PORT 连通"
    else
        echo "  ✗ 端口 $MINIO_PORT 不通"
    fi
    
    # 检查延迟
    ping_result=$(ping -c 3 "$node" 2>/dev/null | grep "avg" | awk -F'/' '{print $5}')
    if [ -n "$ping_result" ]; then
        echo "  ✓ 平均延迟: ${ping_result}ms"
    else
        echo "  ✗ 无法 ping 通"
    fi
    
    echo ""
done
```

#### 性能问题诊断

```bash
#!/bin/bash
# minio-perf-check.sh

echo "=== MinIO 性能诊断 ==="

# 检查 CPU 使用率
echo "1. CPU 使用率:"
top -bn1 | grep "minio" | head -5

# 检查内存使用
echo -e "\n2. 内存使用:"
ps aux | grep minio | awk '{print $2, $3, $4, $11}' | column -t

# 检查网络流量
echo -e "\n3. 网络流量:"
sar -n DEV 1 3 | grep -E "(eth|ens|enp)"

# 检查磁盘 I/O
echo -e "\n4. 磁盘 I/O:"
iotop -ao -d 1 -n 3 | grep minio

# 检查文件描述符使用
echo -e "\n5. 文件描述符:"
for pid in $(pgrep minio); do
    echo "PID $pid: $(ls /proc/$pid/fd 2>/dev/null | wc -l) 个文件描述符"
done

# 检查 MinIO 内部指标
echo -e "\n6. MinIO 指标:"
curl -s http://localhost:9000/minio/v2/metrics/cluster | grep -E "(minio_disk|minio_s3_requests)"
```

### 4.2 日志分析

#### 错误日志分析脚本

```bash
#!/bin/bash
# minio-log-analyzer.sh

LOG_FILE="/var/log/minio/minio.log"
HOURS=${1:-1}

echo "=== 分析最近 $HOURS 小时的 MinIO 日志 ==="

# 统计错误类型
echo "1. 错误统计:"
grep -i error "$LOG_FILE" | \
    grep "$(date -d "$HOURS hours ago" "+%Y-%m-%d")" | \
    awk '{print $4}' | sort | uniq -c | sort -nr

# 统计 API 错误
echo -e "\n2. API 错误统计:"
grep "API:" "$LOG_FILE" | \
    grep -E "(4[0-9][0-9]|5[0-9][0-9])" | \
    awk '{print $6}' | sort | uniq -c | sort -nr

# 慢请求分析
echo -e "\n3. 慢请求 (>1s):"
grep "API:" "$LOG_FILE" | \
    awk '$NF > 1000 {print $0}' | \
    tail -10

# 磁盘错误
echo -e "\n4. 磁盘错误:"
grep -i "disk\|storage" "$LOG_FILE" | \
    grep -i error | tail -10

# 网络错误
echo -e "\n5. 网络错误:"
grep -i "network\|connection\|timeout" "$LOG_FILE" | \
    grep -i error | tail -10
```

### 4.3 自动修复脚本

#### 磁盘自动修复

```bash
#!/bin/bash
# minio-auto-heal.sh

MINIO_ALIAS="myminio"
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# 发送告警
send_alert() {
    local message="$1"
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"MinIO Alert: $message\"}" \
        "$SLACK_WEBHOOK"
}

# 检查并修复磁盘
check_and_heal() {
    echo "开始健康检查..."
    
    # 获取集群状态
    status=$(mc admin info "$MINIO_ALIAS" --json)
    
    # 检查离线磁盘
    offline_disks=$(echo "$status" | jq -r '.info.servers[].disks[] | select(.state == "offline") | .endpoint')
    
    if [ -n "$offline_disks" ]; then
        echo "发现离线磁盘: $offline_disks"
        send_alert "发现离线磁盘: $offline_disks"
        
        # 尝试修复
        echo "开始自动修复..."
        mc admin heal "$MINIO_ALIAS" --recursive --force-start
        
        # 等待修复完成
        while true; do
            heal_status=$(mc admin heal "$MINIO_ALIAS" --json)
            if echo "$heal_status" | jq -e '.healSequence.status == "finished"' >/dev/null; then
                echo "修复完成"
                send_alert "磁盘修复完成"
                break
            fi
            echo "修复进行中..."
            sleep 30
        done
    else
        echo "所有磁盘正常"
    fi
}

# 主循环
while true; do
    check_and_heal
    sleep 300  # 5分钟检查一次
done
```

---

## 5. 备份恢复

### 5.1 数据备份策略

#### 跨区域复制配置

```bash
#!/bin/bash
# setup-replication.sh

SOURCE_ALIAS="source-minio"
TARGET_ALIAS="target-minio"
BUCKET="important-data"

# 配置源和目标
mc alias set "$SOURCE_ALIAS" https://source.minio.com ACCESS_KEY SECRET_KEY
mc alias set "$TARGET_ALIAS" https://target.minio.com ACCESS_KEY SECRET_KEY

# 启用版本控制
mc version enable "$SOURCE_ALIAS/$BUCKET"
mc version enable "$TARGET_ALIAS/$BUCKET"

# 配置复制规则
cat > replication-config.json << EOF
{
  "Role": "arn:minio:replication:::role",
  "Rules": [
    {
      "ID": "ReplicateEverything",
      "Status": "Enabled",
      "Priority": 1,
      "DeleteMarkerReplication": {"Status": "Enabled"},
      "Filter": {"Prefix": ""},
      "Destination": {
        "Bucket": "arn:minio:replication:::$BUCKET",
        "StorageClass": "STANDARD"
      }
    }
  ]
}
EOF

# 应用复制配置
mc replicate add "$SOURCE_ALIAS/$BUCKET" --remote-bucket "$TARGET_ALIAS/$BUCKET" --replicate-config replication-config.json

# 验证复制状态
mc replicate status "$SOURCE_ALIAS/$BUCKET"
```

#### 定期备份脚本

```bash
#!/bin/bash
# minio-backup.sh

SOURCE_ALIAS="prod-minio"
BACKUP_ALIAS="backup-minio"
BACKUP_BUCKET="backups"
DATE=$(date +%Y%m%d)
LOG_FILE="/var/log/minio-backup.log"

# 日志函数
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

# 备份单个桶
backup_bucket() {
    local bucket="$1"
    local backup_path="$BACKUP_BUCKET/daily/$DATE/$bucket"
    
    log "开始备份桶: $bucket"
    
    # 创建备份目录
    mc mb "$BACKUP_ALIAS/$backup_path" 2>/dev/null || true
    
    # 同步数据
    if mc mirror "$SOURCE_ALIAS/$bucket" "$BACKUP_ALIAS/$backup_path" --overwrite; then
        log "桶 $bucket 备份成功"
        
        # 记录备份元数据
        mc stat "$SOURCE_ALIAS/$bucket" --json > "/tmp/${bucket}_metadata.json"
        mc cp "/tmp/${bucket}_metadata.json" "$BACKUP_ALIAS/$backup_path/"
        
        return 0
    else
        log "桶 $bucket 备份失败"
        return 1
    fi
}

# 清理旧备份
cleanup_old_backups() {
    local retention_days=30
    local cutoff_date=$(date -d "$retention_days days ago" +%Y%m%d)
    
    log "清理 $retention_days 天前的备份"
    
    mc ls "$BACKUP_ALIAS/$BACKUP_BUCKET/daily/" | while read line; do
        backup_date=$(echo "$line" | awk '{print $6}' | sed 's/\///g')
        if [[ "$backup_date" < "$cutoff_date" ]]; then
            log "删除旧备份: $backup_date"
            mc rm "$BACKUP_ALIAS/$BACKUP_BUCKET/daily/$backup_date" --recursive --force
        fi
    done
}

# 主备份流程
main() {
    log "开始每日备份任务"
    
    # 获取所有桶列表
    buckets=$(mc ls "$SOURCE_ALIAS" | awk '{print $5}' | sed 's/\///g')
    
    success_count=0
    total_count=0
    
    for bucket in $buckets; do
        total_count=$((total_count + 1))
        if backup_bucket "$bucket"; then
            success_count=$((success_count + 1))
        fi
    done
    
    log "备份完成: $success_count/$total_count 个桶备份成功"
    
    # 清理旧备份
    cleanup_old_backups
    
    # 发送备份报告
    if [ "$success_count" -eq "$total_count" ]; then
        log "所有桶备份成功"
    else
        log "部分桶备份失败，请检查日志"
        # 发送告警邮件
        echo "MinIO 备份任务部分失败，详情请查看 $LOG_FILE" | \
            mail -s "MinIO Backup Alert" admin@example.com
    fi
}

# 执行备份
main
```

### 5.2 数据恢复

#### 恢复脚本

```bash
#!/bin/bash
# minio-restore.sh

BACKUP_ALIAS="backup-minio"
TARGET_ALIAS="prod-minio"
BACKUP_BUCKET="backups"

usage() {
    echo "用法: $0 <backup_date> <bucket_name> [target_bucket]"
    echo "示例: $0 20231201 mybucket"
    echo "      $0 20231201 mybucket restored-mybucket"
    exit 1
}

# 参数检查
if [ $# -lt 2 ]; then
    usage
fi

BACKUP_DATE="$1"
SOURCE_BUCKET="$2"
TARGET_BUCKET="${3:-$SOURCE_BUCKET}"

LOG_FILE="/var/log/minio-restore.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

# 验证备份是否存在
verify_backup() {
    local backup_path="$BACKUP_BUCKET/daily/$BACKUP_DATE/$SOURCE_BUCKET"
    
    if ! mc ls "$BACKUP_ALIAS/$backup_path" >/dev/null 2>&1; then
        log "错误: 备份不存在 $backup_path"
        exit 1
    fi
    
    log "找到备份: $backup_path"
}

# 执行恢复
restore_data() {
    local backup_path="$BACKUP_BUCKET/daily/$BACKUP_DATE/$SOURCE_BUCKET"
    
    log "开始恢复数据从 $backup_path 到 $TARGET_BUCKET"
    
    # 创建目标桶
    mc mb "$TARGET_ALIAS/$TARGET_BUCKET" 2>/dev/null || true
    
    # 恢复数据
    if mc mirror "$BACKUP_ALIAS/$backup_path" "$TARGET_ALIAS/$TARGET_BUCKET" --overwrite; then
        log "数据恢复成功"
        
        # 验证恢复的数据
        source_count=$(mc ls "$BACKUP_ALIAS/$backup_path" --recursive | wc -l)
        target_count=$(mc ls "$TARGET_ALIAS/$TARGET_BUCKET" --recursive | wc -l)
        
        log "源文件数: $source_count, 目标文件数: $target_count"
        
        if [ "$source_count" -eq "$target_count" ]; then
            log "数据验证成功"
        else
            log "警告: 文件数量不匹配"
        fi
        
    else
        log "数据恢复失败"
        exit 1
    fi
}

# 主恢复流程
main() {
    log "开始恢复任务: $BACKUP_DATE -> $SOURCE_BUCKET -> $TARGET_BUCKET"
    
    verify_backup
    restore_data
    
    log "恢复任务完成"
}

# 执行恢复
main
```

### 5.3 增量备份

#### 增量备份实现

```bash
#!/bin/bash
# minio-incremental-backup.sh

SOURCE_ALIAS="prod-minio"
BACKUP_ALIAS="backup-minio"
BACKUP_BUCKET="incremental-backups"
STATE_FILE="/var/lib/minio-backup/last-backup-time"

# 获取上次备份时间
get_last_backup_time() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        # 如果没有记录，使用24小时前
        date -d "24 hours ago" -u +"%Y-%m-%dT%H:%M:%SZ"
    fi
}

# 保存备份时间
save_backup_time() {
    local timestamp="$1"
    mkdir -p "$(dirname "$STATE_FILE")"
    echo "$timestamp" > "$STATE_FILE"
}

# 增量备份单个桶
incremental_backup_bucket() {
    local bucket="$1"
    local since="$2"
    local backup_path="$BACKUP_BUCKET/incremental/$(date +%Y%m%d_%H%M%S)/$bucket"
    
    log "开始增量备份桶: $bucket (since: $since)"
    
    # 创建备份目录
    mc mb "$BACKUP_ALIAS/$backup_path" 2>/dev/null || true
    
    # 获取变更的对象列表
    changed_objects=$(mc ls "$SOURCE_ALIAS/$bucket" --recursive --json | \
        jq -r --arg since "$since" 'select(.lastModified > $since) | .key')
    
    if [ -z "$changed_objects" ]; then
        log "桶 $bucket 没有变更"
        return 0
    fi
    
    # 备份变更的对象
    echo "$changed_objects" | while read -r object; do
        if [ -n "$object" ]; then
            log "备份对象: $object"
            mc cp "$SOURCE_ALIAS/$bucket/$object" "$BACKUP_ALIAS/$backup_path/$object"
        fi
    done
    
    # 保存变更清单
    echo "$changed_objects" > "/tmp/${bucket}_changes.txt"
    mc cp "/tmp/${bucket}_changes.txt" "$BACKUP_ALIAS/$backup_path/"
    
    log "桶 $bucket 增量备份完成"
}

# 主函数
main() {
    local current_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local last_backup_time=$(get_last_backup_time)
    
    log "开始增量备份 (从 $last_backup_time 到 $current_time)"
    
    # 获取所有桶
    buckets=$(mc ls "$SOURCE_ALIAS" | awk '{print $5}' | sed 's/\///g')
    
    for bucket in $buckets; do
        incremental_backup_bucket "$bucket" "$last_backup_time"
    done
    
    # 保存当前备份时间
    save_backup_time "$current_time"
    
    log "增量备份完成"
}

main
```

---

## 6. 安全配置

### 6.1 访问控制

#### IAM 策略配置

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:user/readonly-user"
      },
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::public-bucket",
        "arn:aws:s3:::public-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:user/admin-user"
      },
      "Action": "s3:*",
      "Resource": "*"
    },
    {
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:DeleteObject",
      "Resource": "arn:aws:s3:::critical-bucket/*",
      "Condition": {
        "StringNotEquals": {
          "aws:username": "admin-user"
        }
      }
    }
  ]
}
```

#### 用户管理脚本

```bash
#!/bin/bash
# minio-user-management.sh

MINIO_ALIAS="myminio"

# 创建只读用户
create_readonly_user() {
    local username="$1"
    local password="$2"
    
    # 创建用户
    mc admin user add "$MINIO_ALIAS" "$username" "$password"
    
    # 创建只读策略
    cat > readonly-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::*"
      ]
    }
  ]
}
EOF
    
    # 添加策略
    mc admin policy create "$MINIO_ALIAS" readonly readonly-policy.json
    
    # 分配策略给用户
    mc admin policy attach "$MINIO_ALIAS" readonly --user "$username"
    
    echo "只读用户 $username 创建成功"
}

# 创建管理员用户
create_admin_user() {
    local username="$1"
    local password="$2"
    
    mc admin user add "$MINIO_ALIAS" "$username" "$password"
    mc admin policy attach "$MINIO_ALIAS" consoleAdmin --user "$username"
    
    echo "管理员用户 $username 创建成功"
}

# 创建应用用户（特定桶权限）
create_app_user() {
    local username="$1"
    local password="$2"
    local bucket="$3"
    
    mc admin user add "$MINIO_ALIAS" "$username" "$password"
    
    # 创建桶特定策略
    cat > "${bucket}-policy.json" << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::$bucket",
        "arn:aws:s3:::$bucket/*"
      ]
    }
  ]
}
EOF
    
    mc admin policy create "$MINIO_ALIAS" "${bucket}-access" "${bucket}-policy.json"
    mc admin policy attach "$MINIO_ALIAS" "${bucket}-access" --user "$username"
    
    echo "应用用户 $username 创建成功，可访问桶 $bucket"
}

# 示例用法
create_readonly_user "reader" "ReadOnlyPass123"
create_admin_user "admin" "AdminPass123"
create_app_user "app1" "AppPass123" "app1-data"
```

### 6.2 加密配置

#### KMS 集成

```bash
#!/bin/bash
# setup-kms-encryption.sh

MINIO_ALIAS="myminio"
KMS_ENDPOINT="https://vault.example.com:8200"
KMS_KEY_ID="minio-master-key"

# 配置 KMS
mc admin config set "$MINIO_ALIAS" kms_vault \
    endpoint="$KMS_ENDPOINT" \
    auth_type="approle" \
    auth_approle_id="your-role-id" \
    auth_approle_secret="your-secret-id" \
    key_name="$KMS_KEY_ID"

# 重启服务应用配置
mc admin service restart "$MINIO_ALIAS"

# 创建加密桶
mc mb "$MINIO_ALIAS/encrypted-bucket"

# 设置默认加密
mc encrypt set sse-kms "$KMS_KEY_ID" "$MINIO_ALIAS/encrypted-bucket"

# 验证加密配置
mc encrypt info "$MINIO_ALIAS/encrypted-bucket"

echo "KMS 加密配置完成"
```

#### SSL/TLS 配置

```bash
#!/bin/bash
# setup-ssl.sh

CERT_DIR="/etc/minio/certs"
DOMAIN="minio.example.com"

# 创建证书目录
mkdir -p "$CERT_DIR"

# 生成自签名证书（生产环境请使用 CA 签名证书）
openssl req -new -x509 -days 365 -nodes \
    -out "$CERT_DIR/public.crt" \
    -keyout "$CERT_DIR/private.key" \
    -subj "/C=US/ST=CA/L=San Francisco/O=Example/CN=$DOMAIN"

# 设置权限
chmod 600 "$CERT_DIR/private.key"
chmod 644 "$CERT_DIR/public.crt"

# 生成客户端证书（可选）
openssl req -new -x509 -days 365 -nodes \
    -out "$CERT_DIR/client.crt" \
    -keyout "$CERT_DIR/client.key" \
    -subj "/C=US/ST=CA/L=San Francisco/O=Example/CN=client"

echo "SSL 证书生成完成"
echo "证书文件: $CERT_DIR/public.crt"
echo "私钥文件: $CERT_DIR/private.key"
```

### 6.3 网络安全

#### 防火墙配置

```bash
#!/bin/bash
# setup-firewall.sh

# 清空现有规则
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X

# 设置默认策略
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# 允许本地回环
iptables -A INPUT -i lo -j ACCEPT

# 允许已建立的连接
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# 允许 SSH (修改为你的 SSH 端口)
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# 允许 MinIO API 端口 (仅来自可信网络)
iptables -A INPUT -p tcp --dport 9000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 9000 -s 172.16.0.0/12 -j ACCEPT
iptables -A INPUT -p tcp --dport 9000 -s 192.168.0.0/16 -j ACCEPT

# 允许 MinIO Console 端口 (仅来自管理网络)
iptables -A INPUT -p tcp --dport 9001 -s 192.168.1.0/24 -j ACCEPT

# 允许集群内部通信
iptables -A INPUT -p tcp --dport 9000 -s 10.0.1.0/24 -j ACCEPT

# 记录被拒绝的连接
iptables -A INPUT -j LOG --log-prefix "DROPPED: "

# 保存规则
iptables-save > /etc/iptables/rules.v4

echo "防火墙规则配置完成"
```

---

## 7. 运维自动化

### 7.1 健康检查自动化

#### 综合健康检查脚本

```bash
#!/bin/bash
# minio-health-monitor.sh

MINIO_ALIAS="myminio"
ALERT_EMAIL="admin@example.com"
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
LOG_FILE="/var/log/minio-health.log"

# 配置阈值
DISK_USAGE_THRESHOLD=85
API_ERROR_RATE_THRESHOLD=5
RESPONSE_TIME_THRESHOLD=1000

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

send_alert() {
    local severity="$1"
    local message="$2"
    
    # 发送邮件
    echo "$message" | mail -s "MinIO Alert [$severity]" "$ALERT_EMAIL"
    
    # 发送 Slack 通知
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"MinIO [$severity]: $message\"}" \
        "$SLACK_WEBHOOK"
}

# 检查集群状态
check_cluster_status() {
    log "检查集群状态..."
    
    local status=$(mc admin info "$MINIO_ALIAS" --json 2>/dev/null)
    if [ $? -ne 0 ]; then
        send_alert "CRITICAL" "无法连接到 MinIO 集群"
        return 1
    fi
    
    # 检查节点状态
    local offline_servers=$(echo "$status" | jq -r '.info.servers[] | select(.state == "offline") | .endpoint')
    if [ -n "$offline_servers" ]; then
        send_alert "CRITICAL" "检测到离线节点: $offline_servers"
    fi
    
    # 检查磁盘状态
    local offline_disks=$(echo "$status" | jq -r '.info.servers[].disks[] | select(.state == "offline") | .endpoint')
    if [ -n "$offline_disks" ]; then
        send_alert "CRITICAL" "检测到离线磁盘: $offline_disks"
    fi
    
    return 0
}

# 检查磁盘使用率
check_disk_usage() {
    log "检查磁盘使用率..."
    
    local usage=$(mc admin info "$MINIO_ALIAS" --json | \
        jq -r '.info.servers[].disks[] | select(.state == "online") | (.usedSpace / .totalSpace * 100)')
    
    echo "$usage" | while read -r disk_usage; do
        if (( $(echo "$disk_usage > $DISK_USAGE_THRESHOLD" | bc -l) )); then
            send_alert "WARNING" "磁盘使用率过高: ${disk_usage}%"
        fi
    done
}

# 检查 API 性能
check_api_performance() {
    log "检查 API 性能..."
    
    local start_time=$(date +%s%N)
    mc ls "$MINIO_ALIAS" >/dev/null 2>&1
    local end_time=$(date +%s%N)
    
    local response_time=$(( (end_time - start_time) / 1000000 ))  # 转换为毫秒
    
    if [ "$response_time" -gt "$RESPONSE_TIME_THRESHOLD" ]; then
        send_alert "WARNING" "API 响应时间过长: ${response_time}ms"
    fi
    
    log "API 响应时间: ${response_time}ms"
}

# 检查数据完整性
check_data_integrity() {
    log "检查数据完整性..."
    
    # 运行数据扫描器
    local scan_result=$(mc admin heal "$MINIO_ALIAS" --dry-run --json)
    local corrupted_objects=$(echo "$scan_result" | jq -r '.healSequence.numHealedObjects // 0')
    
    if [ "$corrupted_objects" -gt 0 ]; then
        send_alert "WARNING" "检测到 $corrupted_objects 个损坏对象，已自动修复"
    fi
}

# 检查备份状态
check_backup_status() {
    log "检查备份状态..."
    
    local last_backup=$(find /var/log -name "minio-backup.log" -exec grep "备份完成" {} \; | tail -1)
    if [ -n "$last_backup" ]; then
        local backup_time=$(echo "$last_backup" | awk '{print $1, $2}')
        local backup_timestamp=$(date -d "$backup_time" +%s)
        local current_timestamp=$(date +%s)
        local hours_since_backup=$(( (current_timestamp - backup_timestamp) / 3600 ))
        
        if [ "$hours_since_backup" -gt 25 ]; then  # 超过25小时没有备份
            send_alert "WARNING" "备份任务可能失败，上次成功备份时间: $backup_time"
        fi
    else
        send_alert "WARNING" "未找到备份日志记录"
    fi
}

# 生成健康报告
generate_health_report() {
    local report_file="/tmp/minio-health-report-$(date +%Y%m%d).txt"
    
    {
        echo "MinIO 集群健康报告 - $(date)"
        echo "=================================="
        echo ""
        
        echo "集群信息:"
        mc admin info "$MINIO_ALIAS"
        echo ""
        
        echo "存储使用情况:"
        mc admin info "$MINIO_ALIAS" --json | jq -r '.info.usage'
        echo ""
        
        echo "性能指标:"
        curl -s http://localhost:9000/minio/v2/metrics/cluster | grep -E "(minio_disk|minio_s3_requests)" | head -10
        echo ""
        
        echo "最近错误日志:"
        tail -20 /var/log/minio/minio.log | grep -i error
        
    } > "$report_file"
    
    # 发送报告
    mail -s "MinIO 每日健康报告" -a "$report_file" "$ALERT_EMAIL" < /dev/null
    
    log "健康报告已生成: $report_file"
}

# 主检查流程
main() {
    log "开始健康检查..."
    
    check_cluster_status || exit 1
    check_disk_usage
    check_api_performance
    check_data_integrity
    check_backup_status
    
    # 每天生成一次详细报告
    if [ "$(date +%H:%M)" = "08:00" ]; then
        generate_health_report
    fi
    
    log "健康检查完成"
}

# 执行检查
main
```

### 7.2 自动扩容

#### 动态扩容脚本

```bash
#!/bin/bash
# minio-auto-scale.sh

MINIO_ALIAS="myminio"
SCALE_UP_THRESHOLD=80    # 磁盘使用率超过80%时扩容
SCALE_DOWN_THRESHOLD=30  # 磁盘使用率低于30%时缩容
MIN_NODES=4              # 最小节点数
MAX_NODES=16             # 最大节点数

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

# 获取当前集群状态
get_cluster_metrics() {
    local metrics=$(mc admin info "$MINIO_ALIAS" --json)
    
    # 计算平均磁盘使用率
    local avg_usage=$(echo "$metrics" | jq -r '
        [.info.servers[].disks[] | select(.state == "online") | (.usedSpace / .totalSpace * 100)] |
        add / length
    ')
    
    # 获取当前节点数
    local current_nodes=$(echo "$metrics" | jq -r '.info.servers | length')
    
    echo "$avg_usage $current_nodes"
}

# 添加新节点
add_node() {
    local new_node_ip="$1"
    
    log "添加新节点: $new_node_ip"
    
    # 在新节点上部署 MinIO
    ssh "$new_node_ip" "
        docker run -d --name minio \
            --restart unless-stopped \
            -p 9000:9000 -p 9001:9001 \
            -v /mnt/disk1:/data1 \
            -v /mnt/disk2:/data2 \
            -v /mnt/disk3:/data3 \
            -v /mnt/disk4:/data4 \
            -e MINIO_ROOT_USER=minioadmin \
            -e MINIO_ROOT_PASSWORD=minioadmin123 \
            minio/minio server \
            --console-address ':9001' \
            http://minio{1..$(($current_nodes + 1))}/data{1..4}
    "
    
    # 更新负载均衡器配置
    update_load_balancer_config
    
    log "新节点 $new_node_ip 添加完成"
}

# 移除节点
remove_node() {
    local node_ip="$1"
    
    log "移除节点: $node_ip"
    
    # 迁移数据到其他节点
    migrate_data_from_node "$node_ip"
    
    # 停止节点上的 MinIO 服务
    ssh "$node_ip" "docker stop minio && docker rm minio"
    
    # 更新负载均衡器配置
    update_load_balancer_config
    
    log "节点 $node_ip 移除完成"
}

# 数据迁移
migrate_data_from_node() {
    local source_node="$1"
    
    log "开始从节点 $source_node 迁移数据"
    
    # 获取节点上的数据
    local buckets=$(mc ls "http://$source_node:9000" --json | jq -r '.key')
    
    for bucket in $buckets; do
        log "迁移桶: $bucket"
        
        # 使用 mc mirror 迁移数据
        mc mirror "http://$source_node:9000/$bucket" "$MINIO_ALIAS/$bucket" --remove
    done
    
    log "数据迁移完成"
}

# 更新负载均衡器配置
update_load_balancer_config() {
    log "更新负载均衡器配置"
    
    # 获取当前活跃节点列表
    local active_nodes=$(mc admin info "$MINIO_ALIAS" --json | \
        jq -r '.info.servers[] | select(.state == "online") | .endpoint')
    
    # 生成新的 Nginx 配置
    cat > /tmp/nginx-upstream.conf << EOF
upstream minio_backend {
    least_conn;
EOF
    
    echo "$active_nodes" | while read -r node; do
        echo "    server $node max_fails=3 fail_timeout=30s;" >> /tmp/nginx-upstream.conf
    done
    
    echo "}" >> /tmp/nginx-upstream.conf
    
    # 应用新配置
    sudo cp /tmp/nginx-upstream.conf /etc/nginx/conf.d/
    sudo nginx -t && sudo systemctl reload nginx
    
    log "负载均衡器配置更新完成"
}

# 获取可用的新节点
get_available_node() {
    # 从节点池中获取可用节点
    local node_pool=("10.0.1.10" "10.0.1.11" "10.0.1.12" "10.0.1.13")
    local current_nodes=$(mc admin info "$MINIO_ALIAS" --json | \
        jq -r '.info.servers[].endpoint' | cut -d: -f1)
    
    for node in "${node_pool[@]}"; do
        if ! echo "$current_nodes" | grep -q "$node"; then
            echo "$node"
            return
        fi
    done
}

# 主扩缩容逻辑
main() {
    local metrics=$(get_cluster_metrics)
    local avg_usage=$(echo "$metrics" | awk '{print $1}')
    local current_nodes=$(echo "$metrics" | awk '{print $2}')
    
    log "当前状态: 节点数=$current_nodes, 平均磁盘使用率=${avg_usage}%"
    
    # 检查是否需要扩容
    if (( $(echo "$avg_usage > $SCALE_UP_THRESHOLD" | bc -l) )) && [ "$current_nodes" -lt "$MAX_NODES" ]; then
        log "磁盘使用率过高，开始扩容"
        
        local new_node=$(get_available_node)
        if [ -n "$new_node" ]; then
            add_node "$new_node"
        else
            log "警告: 没有可用的新节点进行扩容"
        fi
        
    # 检查是否需要缩容
    elif (( $(echo "$avg_usage < $SCALE_DOWN_THRESHOLD" | bc -l) )) && [ "$current_nodes" -gt "$MIN_NODES" ]; then
        log "磁盘使用率较低，开始缩容"
        
        # 选择使用率最低的节点进行移除
        local node_to_remove=$(mc admin info "$MINIO_ALIAS" --json | \
            jq -r '.info.servers[] | select(.state == "online") | 
                   [.endpoint, (.disks[] | .usedSpace / .totalSpace)] | 
                   @csv' | \
            sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d '"')
        
        if [ -n "$node_to_remove" ]; then
            remove_node "$node_to_remove"
        fi
    else
        log "集群状态正常，无需扩缩容"
    fi
}

# 执行扩缩容检查
main
```

### 7.3 配置管理自动化

#### Ansible Playbook

```yaml
# minio-cluster.yml
---
- name: Deploy MinIO Cluster
  hosts: minio_nodes
  become: yes
  vars:
    minio_version: "latest"
    minio_user: "minio"
    minio_group: "minio"
    minio_data_dirs:
      - "/mnt/disk1"
      - "/mnt/disk2"
      - "/mnt/disk3"
      - "/mnt/disk4"
    minio_root_user: "minioadmin"
    minio_root_password: "{{ vault_minio_root_password }}"
    
  tasks:
    - name: Create minio user
      user:
        name: "{{ minio_user }}"
        group: "{{ minio_group }}"
        system: yes
        shell: /bin/false
        home: /var/lib/minio
        create_home: yes

    - name: Create data directories
      file:
        path: "{{ item }}"
        state: directory
        owner: "{{ minio_user }}"
        group: "{{ minio_group }}"
        mode: '0755'
      loop: "{{ minio_data_dirs }}"

    - name: Download MinIO binary
      get_url:
        url: "https://dl.min.io/server/minio/release/linux-amd64/minio"
        dest: "/usr/local/bin/minio"
        mode: '0755'
        owner: root
        group: root

    - name: Create MinIO configuration directory
      file:
        path: /etc/minio
        state: directory
        owner: "{{ minio_user }}"
        group: "{{ minio_group }}"
        mode: '0755'

    - name: Create MinIO environment file
      template:
        src: minio.env.j2
        dest: /etc/default/minio
        owner: root
        group: root
        mode: '0644'
      notify: restart minio

    - name: Create MinIO systemd service
      template:
        src: minio.service.j2
        dest: /etc/systemd/system/minio.service
        owner: root
        group: root
        mode: '0644'
      notify:
        - reload systemd
        - restart minio

    - name: Start and enable MinIO service
      systemd:
        name: minio
        state: started
        enabled: yes
        daemon_reload: yes

    - name: Configure firewall
      firewalld:
        port: "{{ item }}"
        permanent: yes
        state: enabled
        immediate: yes
      loop:
        - "9000/tcp"
        - "9001/tcp"

  handlers:
    - name: reload systemd
      systemd:
        daemon_reload: yes

    - name: restart minio
      systemd:
        name: minio
        state: restarted
```

```jinja2
# templates/minio.env.j2
# MinIO configuration
MINIO_ROOT_USER={{ minio_root_user }}
MINIO_ROOT_PASSWORD={{ minio_root_password }}
MINIO_VOLUMES="{% for host in groups['minio_nodes'] %}{% for dir in minio_data_dirs %}http://{{ host }}:9000{{ dir }}{% if not loop.last %} {% endif %}{% endfor %}{% if not loop.last %} {% endif %}{% endfor %}"
MINIO_OPTS="--console-address :9001"
MINIO_PROMETHEUS_AUTH_TYPE=public
```

```ini
# templates/minio.service.j2
[Unit]
Description=MinIO
Documentation=https://docs.min.io
Wants=network-online.target
After=network-online.target
AssertFileIsExecutable=/usr/local/bin/minio

[Service]
WorkingDirectory=/usr/local/

User={{ minio_user }}
Group={{ minio_group }}

EnvironmentFile=/etc/default/minio
ExecStartPre=/bin/bash -c "if [ -z \"${MINIO_VOLUMES}\" ]; then echo \"Variable MINIO_VOLUMES not set in /etc/default/minio\"; exit 1; fi"

ExecStart=/usr/local/bin/minio server $MINIO_OPTS $MINIO_VOLUMES

# Let systemd restart this service always
Restart=always

# Specifies the maximum file descriptor number that can be opened by this process
LimitNOFILE=65536

# Specifies the maximum number of threads this process can create
TasksMax=infinity

# Disable timeout logic and wait until process is stopped
TimeoutStopSec=infinity
SendSIGKILL=no

[Install]
WantedBy=multi-user.target
```

---

## 总结

本实战指南涵盖了 MinIO 在生产环境中的完整部署和运维流程，包括：

1. **生产环境部署**: 从硬件规划到分布式部署的完整方案
2. **性能调优**: 系统级和应用级的全方位优化
3. **监控告警**: 基于 Prometheus 和 Grafana 的完整监控体系
4. **故障排查**: 常见问题的诊断和解决方案
5. **备份恢复**: 完整的数据保护策略
6. **安全配置**: 全面的安全防护措施
7. **运维自动化**: 智能化的运维管理工具

这些实战经验和工具脚本可以帮助运维团队快速构建稳定、高效的 MinIO 存储集群，并确保其在生产环境中的可靠运行。
