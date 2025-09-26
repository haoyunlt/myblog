---
title: "VoiceHelper智能语音助手 - 部署与运维"
date: "2025-09-22T14:00:00+08:00"
draft: false
description: "VoiceHelper系统部署与运维指南，涵盖容器化部署、CI/CD流水线、运维自动化等关键技术"
slug: "voicehelper-deployment-operations"
author: "tommie blog"
categories: ["voicehelper", "AI", "运维部署"]
tags: ["VoiceHelper", "Docker", "Kubernetes", "CI/CD", "运维自动化", "部署"]
showComments: false
toc: true
tocOpen: false
showReadingTime: true
showWordCount: true
pinned: true
weight: 10
# 性能优化配置
paginated: true
lazyLoad: true
performanceOptimized: true
---

# VoiceHelper部署与运维

本文档详细介绍VoiceHelper智能语音助手系统的部署与运维方案，涵盖容器化部署、CI/CD流水线、运维自动化等关键技术。

## 9. 部署与运维

### 9.1 容器化部署

#### 9.1.0 部署架构总览

```mermaid
graph TB
    subgraph "开发环境"
        Dev[开发者]
        Git[Git仓库]
        IDE[开发工具]
    end
    
    subgraph "CI/CD流水线"
        CI[持续集成]
        Build[镜像构建]
        Test[自动化测试]
        Deploy[自动部署]
    end
    
    subgraph "容器化平台"
        Docker[Docker容器]
        K8s[Kubernetes集群]
        Helm[Helm包管理]
        Registry[镜像仓库]
    end
    
    subgraph "生产环境"
        LB[负载均衡器]
        Frontend[前端服务]
        Backend[后端服务]
        AI[AI服务]
        DB[(数据库集群)]
    end
    
    subgraph "监控运维"
        Monitor[监控系统]
        Log[日志系统]
        Alert[告警系统]
        Backup[备份系统]
    end
    
    Dev --> Git
    Git --> CI
    CI --> Build
    Build --> Test
    Test --> Deploy
    
    Build --> Registry
    Deploy --> K8s
    K8s --> Helm
    
    K8s --> LB
    LB --> Frontend
    LB --> Backend
    LB --> AI
    Backend --> DB
    
    K8s --> Monitor
    K8s --> Log
    Monitor --> Alert
    DB --> Backup
    
    style CI fill:#e1f5fe
    style K8s fill:#f3e5f5
    style LB fill:#fff3e0
    style Monitor fill:#e8f5e8
```
  </div>
</div>

#### 9.1.1 Docker容器化配置

```dockerfile
# 后端服务Dockerfile
# 文件路径: backend/Dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app

# 安装依赖
RUN apk add --no-cache git ca-certificates tzdata

# 复制go mod文件
COPY go.mod go.sum ./
RUN go mod download

# 复制源代码
COPY . .

# 构建应用
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/server

# 运行阶段
FROM alpine:latest

RUN apk --no-cache add ca-certificates tzdata
WORKDIR /root/

# 复制构建的二进制文件
COPY --from=builder /app/main .
COPY --from=builder /app/configs ./configs

# 设置时区
ENV TZ=Asia/Shanghai

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# 启动应用
CMD ["./main"]
```

```dockerfile
# 前端应用Dockerfile
# 文件路径: frontend/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# 复制package文件
COPY package*.json ./
RUN npm ci --only=production

# 复制源代码
COPY . .

# 构建应用
RUN npm run build

# 运行阶段
FROM nginx:alpine

# 复制构建文件
COPY --from=builder /app/out /usr/share/nginx/html

# 复制nginx配置
COPY nginx.conf /etc/nginx/nginx.conf

# 暴露端口
EXPOSE 80

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
```

```dockerfile
# AI服务Dockerfile
# 文件路径: algo/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 9.1.2 Docker Compose配置

```yaml
# Docker Compose配置
# 文件路径: docker-compose.yml
version: '3.8'

services:
  # 前端服务
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    environment:
      - NODE_ENV=production
    depends_on:
      - backend
    networks:
      - voicehelper-network
    restart: unless-stopped

  # 后端服务
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - GO_ENV=production
      - DATABASE_URL=postgres://user:password@postgres:5432/voicehelper
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - voicehelper-network
    restart: unless-stopped
    volumes:
      - ./configs:/root/configs:ro

  # AI服务
  ai-service:
    build:
      context: ./algo
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PYTHON_ENV=production
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
    depends_on:
      - milvus
    networks:
      - voicehelper-network
    restart: unless-stopped
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=voicehelper
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - voicehelper-network
    restart: unless-stopped
    ports:
      - "5432:5432"

  # Redis缓存
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass redispassword
    volumes:
      - redis_data:/data
    networks:
      - voicehelper-network
    restart: unless-stopped
    ports:
      - "6379:6379"

  # Milvus向量数据库
  milvus:
    image: milvusdb/milvus:v2.3.4
    command: ["milvus", "run", "standalone"]
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    depends_on:
      - etcd
      - minio
    networks:
      - voicehelper-network
    restart: unless-stopped
    ports:
      - "19530:19530"

  # Etcd (Milvus依赖)
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    volumes:
      - etcd_data:/etcd
    networks:
      - voicehelper-network
    restart: unless-stopped

  # MinIO对象存储
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin123
    volumes:
      - minio_data:/data
    networks:
      - voicehelper-network
    restart: unless-stopped
    ports:
      - "9000:9000"
      - "9001:9001"

  # Neo4j图数据库
  neo4j:
    image: neo4j:5.0
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
    networks:
      - voicehelper-network
    restart: unless-stopped
    ports:
      - "7474:7474"
      - "7687:7687"

  # Nginx负载均衡
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - frontend
      - backend
    networks:
      - voicehelper-network
    restart: unless-stopped

  # Prometheus监控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml:ro
      - prometheus_data:/prometheus
    networks:
      - voicehelper-network
    restart: unless-stopped

  # Grafana可视化
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - voicehelper-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  milvus_data:
  etcd_data:
  minio_data:
  neo4j_data:
  prometheus_data:
  grafana_data:

networks:
  voicehelper-network:
    driver: bridge
```

### 9.2 Kubernetes部署

#### 9.2.1 Kubernetes部署配置

```yaml
# Namespace配置
# 文件路径: k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: voicehelper
  labels:
    name: voicehelper
---
# ConfigMap配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: voicehelper-config
  namespace: voicehelper
data:
  database.url: "postgres://user:password@postgres:5432/voicehelper"
  redis.url: "redis://redis:6379"
  milvus.host: "milvus"
  milvus.port: "19530"
---
# Secret配置
apiVersion: v1
kind: Secret
metadata:
  name: voicehelper-secrets
  namespace: voicehelper
type: Opaque
data:
  database-password: cGFzc3dvcmQ=  # base64 encoded
  redis-password: cmVkaXNwYXNzd29yZA==
  jwt-secret: and0c2VjcmV0a2V5
```

```yaml
# 后端服务部署
# 文件路径: k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voicehelper-backend
  namespace: voicehelper
  labels:
    app: voicehelper-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voicehelper-backend
  template:
    metadata:
      labels:
        app: voicehelper-backend
    spec:
      containers:
      - name: backend
        image: voicehelper/backend:latest
        ports:
        - containerPort: 8080
        env:
        - name: GO_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: voicehelper-config
              key: database.url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: voicehelper-config
              key: redis.url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: voicehelper-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /root/configs
          readOnly: true
      volumes:
      - name: config-volume
        configMap:
          name: voicehelper-config
---
apiVersion: v1
kind: Service
metadata:
  name: voicehelper-backend-service
  namespace: voicehelper
spec:
  selector:
    app: voicehelper-backend
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
  type: ClusterIP
```

```yaml
# AI服务部署
# 文件路径: k8s/ai-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voicehelper-ai-service
  namespace: voicehelper
  labels:
    app: voicehelper-ai-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: voicehelper-ai-service
  template:
    metadata:
      labels:
        app: voicehelper-ai-service
    spec:
      containers:
      - name: ai-service
        image: voicehelper/ai-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYTHON_ENV
          value: "production"
        - name: MILVUS_HOST
          valueFrom:
            configMapKeyRef:
              name: voicehelper-config
              key: milvus.host
        - name: MILVUS_PORT
          valueFrom:
            configMapKeyRef:
              name: voicehelper-config
              key: milvus.port
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
            nvidia.com/gpu: 1
          limits:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: model-volume
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-storage-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100
---
apiVersion: v1
kind: Service
metadata:
  name: voicehelper-ai-service
  namespace: voicehelper
spec:
  selector:
    app: voicehelper-ai-service
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
```

```yaml
# 数据库部署
# 文件路径: k8s/database-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: voicehelper
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: voicehelper
        - name: POSTGRES_USER
          value: user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: voicehelper-secrets
              key: database-password
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: init-script
          mountPath: /docker-entrypoint-initdb.d
      volumes:
      - name: init-script
        configMap:
          name: postgres-init-script
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: voicehelper
spec:
  selector:
    app: postgres
  ports:
  - protocol: TCP
    port: 5432
    targetPort: 5432
  type: ClusterIP
```

#### 9.2.2 Helm Chart配置

```yaml
# Helm Chart配置
# 文件路径: helm/voicehelper/Chart.yaml
apiVersion: v2
name: voicehelper
description: VoiceHelper智能语音助手系统
type: application
version: 1.0.0
appVersion: "1.0.0"
dependencies:
  - name: postgresql
    version: 12.1.9
    repository: https://charts.bitnami.com/bitnami
  - name: redis
    version: 17.4.3
    repository: https://charts.bitnami.com/bitnami
  - name: prometheus
    version: 15.18.0
    repository: https://prometheus-community.github.io/helm-charts
  - name: grafana
    version: 6.44.11
    repository: https://grafana.github.io/helm-charts
```

```yaml
# Helm Values配置
# 文件路径: helm/voicehelper/values.yaml
global:
  imageRegistry: ""
  imagePullSecrets: []

replicaCount:
  backend: 3
  frontend: 2
  aiService: 2

image:
  backend:
    repository: voicehelper/backend
    tag: "latest"
    pullPolicy: IfNotPresent
  frontend:
    repository: voicehelper/frontend
    tag: "latest"
    pullPolicy: IfNotPresent
  aiService:
    repository: voicehelper/ai-service
    tag: "latest"
    pullPolicy: IfNotPresent

service:
  type: ClusterIP
  backend:
    port: 8080
  frontend:
    port: 80
  aiService:
    port: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: api.voicehelper.com
      paths:
        - path: /
          pathType: Prefix
          service: backend
    - host: voicehelper.com
      paths:
        - path: /
          pathType: Prefix
          service: frontend
  tls:
    - secretName: voicehelper-tls
      hosts:
        - voicehelper.com
        - api.voicehelper.com

resources:
  backend:
    limits:
      cpu: 500m
      memory: 512Mi
    requests:
      cpu: 250m
      memory: 256Mi
  frontend:
    limits:
      cpu: 200m
      memory: 256Mi
    requests:
      cpu: 100m
      memory: 128Mi
  aiService:
    limits:
      cpu: 1000m
      memory: 2Gi
      nvidia.com/gpu: 1
    requests:
      cpu: 500m
      memory: 1Gi
      nvidia.com/gpu: 1

autoscaling:
  enabled: true
  backend:
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
  aiService:
    minReplicas: 2
    maxReplicas: 5
    targetCPUUtilizationPercentage: 80

nodeSelector:
  aiService:
    accelerator: nvidia-tesla-v100

tolerations: []

affinity:
  backend:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
            - key: app
              operator: In
              values:
              - voicehelper-backend
          topologyKey: kubernetes.io/hostname

# 外部依赖配置
postgresql:
  enabled: true
  auth:
    postgresPassword: "password"
    database: "voicehelper"
  primary:
    persistence:
      enabled: true
      size: 20Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: "redispassword"
  master:
    persistence:
      enabled: true
      size: 8Gi

prometheus:
  enabled: true
  server:
    persistentVolume:
      size: 20Gi

grafana:
  enabled: true
  persistence:
    enabled: true
    size: 10Gi
  adminPassword: "admin123"
```

### 9.3 CI/CD流水线

#### 9.3.0 CI/CD流水线架构图

```mermaid
sequenceDiagram
    participant Dev as 开发者
    participant Git as Git仓库
    participant CI as CI系统
    participant Build as 构建系统
    participant Test as 测试系统
    participant Security as 安全扫描
    participant Registry as 镜像仓库
    participant Staging as 测试环境
    participant Prod as 生产环境
    participant Monitor as 监控系统

    Dev->>Git: 推送代码
    Git->>CI: 触发流水线
    
    par 并行执行
        CI->>Build: 构建后端服务
        CI->>Build: 构建前端应用
        CI->>Build: 构建AI服务
    end
    
    par 并行测试
        Build->>Test: 单元测试
        Build->>Test: 集成测试
        Build->>Security: 安全扫描
    end
    
    alt 测试通过
        Test->>Build: 测试成功
        Security->>Build: 安全检查通过
        
        Build->>Registry: 推送镜像
        Registry->>Staging: 部署到测试环境
        
        Staging->>Test: 冒烟测试
        Test->>Staging: 测试通过
        
        alt 生产部署
            Staging->>Prod: 部署到生产环境
            Prod->>Monitor: 启动监控
            Monitor->>Dev: 部署成功通知
        end
        
    else 测试失败
        Test->>Dev: 发送失败通知
        Security->>Dev: 发送安全问题通知
    end
```
  </div>
</div>

#### 9.3.1 GitHub Actions配置

```yaml
# GitHub Actions工作流
# 文件路径: .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4

    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'

    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    # 后端测试
    - name: Test Backend
      run: |
        cd backend
        go mod download
        go test -v -race -coverprofile=coverage.out ./...
        go tool cover -html=coverage.out -o coverage.html

    # 前端测试
    - name: Test Frontend
      run: |
        cd frontend
        npm ci
        npm run test:ci
        npm run build

    # AI服务测试
    - name: Test AI Service
      run: |
        cd algo
        pip install -r requirements.txt
        python -m pytest tests/ -v --cov=./ --cov-report=xml

    # 上传测试覆盖率
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        files: ./backend/coverage.out,./algo/coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    needs: test
    steps:
    - uses: actions/checkout@v4

    # Go安全扫描
    - name: Run Gosec Security Scanner
      uses: securecodewarrior/github-action-gosec@master
      with:
        args: './backend/...'

    # Node.js安全扫描
    - name: Run npm audit
      run: |
        cd frontend
        npm audit --audit-level moderate

    # Python安全扫描
    - name: Run Safety
      run: |
        cd algo
        pip install safety
        safety check -r requirements.txt

    # Docker镜像安全扫描
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'voicehelper/backend:latest'
        format: 'sarif'
        output: 'trivy-results.sarif'

  build-and-push:
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    strategy:
      matrix:
        service: [backend, frontend, ai-service]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.service }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: ./${{ matrix.service }}
        file: ./${{ matrix.service }}/Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.ref == 'refs/heads/develop'
    environment: staging

    steps:
    - uses: actions/checkout@v4

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig

    - name: Deploy to staging
      run: |
        helm upgrade --install voicehelper-staging ./helm/voicehelper \
          --namespace voicehelper-staging \
          --create-namespace \
          --set image.backend.tag=${{ github.sha }} \
          --set image.frontend.tag=${{ github.sha }} \
          --set image.aiService.tag=${{ github.sha }} \
          --values ./helm/voicehelper/values-staging.yaml

  deploy-production:
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - uses: actions/checkout@v4

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig

    - name: Deploy to production
      run: |
        helm upgrade --install voicehelper ./helm/voicehelper \
          --namespace voicehelper \
          --create-namespace \
          --set image.backend.tag=${{ github.sha }} \
          --set image.frontend.tag=${{ github.sha }} \
          --set image.aiService.tag=${{ github.sha }} \
          --values ./helm/voicehelper/values-production.yaml

    - name: Run smoke tests
      run: |
        kubectl wait --for=condition=ready pod -l app=voicehelper-backend -n voicehelper --timeout=300s
        kubectl run smoke-test --rm -i --restart=Never --image=curlimages/curl -- \
          curl -f http://voicehelper-backend-service:8080/health

  notify:
    runs-on: ubuntu-latest
    needs: [deploy-staging, deploy-production]
    if: always()
    
    steps:
    - name: Notify deployment status
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        fields: repo,message,commit,author,action,eventName,ref,workflow
```

#### 9.3.2 GitLab CI/CD配置

```yaml
# GitLab CI/CD配置
# 文件路径: .gitlab-ci.yml
stages:
  - test
  - security
  - build
  - deploy-staging
  - deploy-production

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  REGISTRY: $CI_REGISTRY
  IMAGE_TAG: $CI_COMMIT_SHA

# 测试阶段
test-backend:
  stage: test
  image: golang:1.21
  services:
    - postgres:15
    - redis:7
  variables:
    POSTGRES_DB: test
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
    REDIS_URL: redis://redis:6379
  script:
    - cd backend
    - go mod download
    - go test -v -race -coverprofile=coverage.out ./...
    - go tool cover -func=coverage.out
  coverage: '/total:\s+\(statements\)\s+(\d+\.\d+\%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: backend/coverage.xml

test-frontend:
  stage: test
  image: node:18
  script:
    - cd frontend
    - npm ci
    - npm run test:ci
    - npm run build
  artifacts:
    reports:
      junit: frontend/junit.xml
      coverage_report:
        coverage_format: cobertura
        path: frontend/coverage/cobertura-coverage.xml

test-ai-service:
  stage: test
  image: python:3.11
  script:
    - cd algo
    - pip install -r requirements.txt
    - python -m pytest tests/ -v --cov=./ --cov-report=xml --junitxml=junit.xml
  artifacts:
    reports:
      junit: algo/junit.xml
      coverage_report:
        coverage_format: cobertura
        path: algo/coverage.xml

# 安全扫描
security-scan:
  stage: security
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker run --rm -v "$PWD:/src" securecodewarrior/gosec /src/backend/...
    - docker run --rm -v "$PWD:/src" aquasec/trivy fs --security-checks vuln /src
  allow_failure: true

# 构建镜像
.build-template: &build-template
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $REGISTRY/$CI_PROJECT_PATH/$SERVICE:$IMAGE_TAG ./$SERVICE
    - docker push $REGISTRY/$CI_PROJECT_PATH/$SERVICE:$IMAGE_TAG
    - docker tag $REGISTRY/$CI_PROJECT_PATH/$SERVICE:$IMAGE_TAG $REGISTRY/$CI_PROJECT_PATH/$SERVICE:latest
    - docker push $REGISTRY/$CI_PROJECT_PATH/$SERVICE:latest
  only:
    - main
    - develop

build-backend:
  <<: *build-template
  variables:
    SERVICE: backend

build-frontend:
  <<: *build-template
  variables:
    SERVICE: frontend

build-ai-service:
  <<: *build-template
  variables:
    SERVICE: ai-service

# 部署到测试环境
deploy-staging:
  stage: deploy-staging
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging.voicehelper.com
  script:
    - kubectl config use-context $KUBE_CONTEXT_STAGING
    - helm upgrade --install voicehelper-staging ./helm/voicehelper
        --namespace voicehelper-staging
        --create-namespace
        --set image.backend.tag=$IMAGE_TAG
        --set image.frontend.tag=$IMAGE_TAG
        --set image.aiService.tag=$IMAGE_TAG
        --values ./helm/voicehelper/values-staging.yaml
  only:
    - develop

# 部署到生产环境
deploy-production:
  stage: deploy-production
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://voicehelper.com
  script:
    - kubectl config use-context $KUBE_CONTEXT_PRODUCTION
    - helm upgrade --install voicehelper ./helm/voicehelper
        --namespace voicehelper
        --create-namespace
        --set image.backend.tag=$IMAGE_TAG
        --set image.frontend.tag=$IMAGE_TAG
        --set image.aiService.tag=$IMAGE_TAG
        --values ./helm/voicehelper/values-production.yaml
  when: manual
  only:
    - main
```

### 9.4 运维自动化

#### 9.4.1 自动化运维脚本

```bash
#!/bin/bash
# 自动化部署脚本
# 文件路径: scripts/deploy.sh

set -e

# 配置变量
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
NAMESPACE="voicehelper-${ENVIRONMENT}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl未安装"
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        log_error "helm未安装"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到Kubernetes集群"
        exit 1
    fi
    
    log_info "依赖检查完成"
}

# 预部署检查
pre_deploy_check() {
    log_info "执行预部署检查..."
    
    # 检查镜像是否存在
    local images=("backend" "frontend" "ai-service")
    for image in "${images[@]}"; do
        if ! docker manifest inspect "ghcr.io/voicehelper/${image}:${VERSION}" &> /dev/null; then
            log_error "镜像 ghcr.io/voicehelper/${image}:${VERSION} 不存在"
            exit 1
        fi
    done
    
    # 检查命名空间
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_info "创建命名空间 ${NAMESPACE}"
        kubectl create namespace "${NAMESPACE}"
    fi
    
    log_info "预部署检查完成"
}

# 备份当前部署
backup_current_deployment() {
    log_info "备份当前部署..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${backup_dir}"
    
    # 备份Helm release
    helm get values voicehelper -n "${NAMESPACE}" > "${backup_dir}/values.yaml" 2>/dev/null || true
    helm get manifest voicehelper -n "${NAMESPACE}" > "${backup_dir}/manifest.yaml" 2>/dev/null || true
    
    # 备份数据库
    if kubectl get pod -n "${NAMESPACE}" -l app=postgres &> /dev/null; then
        log_info "备份数据库..."
        kubectl exec -n "${NAMESPACE}" -it deployment/postgres -- pg_dump -U user voicehelper > "${backup_dir}/database.sql"
    fi
    
    log_info "备份完成: ${backup_dir}"
}

# 执行部署
deploy() {
    log_info "开始部署 VoiceHelper ${VERSION} 到 ${ENVIRONMENT} 环境..."
    
    # 更新Helm依赖
    helm dependency update ./helm/voicehelper
    
    # 执行部署
    helm upgrade --install voicehelper ./helm/voicehelper \
        --namespace "${NAMESPACE}" \
        --create-namespace \
        --set image.backend.tag="${VERSION}" \
        --set image.frontend.tag="${VERSION}" \
        --set image.aiService.tag="${VERSION}" \
        --values "./helm/voicehelper/values-${ENVIRONMENT}.yaml" \
        --wait \
        --timeout=600s
    
    log_info "部署完成"
}

# 部署后验证
post_deploy_verification() {
    log_info "执行部署后验证..."
    
    # 等待Pod就绪
    log_info "等待Pod就绪..."
    kubectl wait --for=condition=ready pod -l app=voicehelper-backend -n "${NAMESPACE}" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=voicehelper-frontend -n "${NAMESPACE}" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=voicehelper-ai-service -n "${NAMESPACE}" --timeout=300s
    
    # 健康检查
    log_info "执行健康检查..."
    local backend_pod=$(kubectl get pod -n "${NAMESPACE}" -l app=voicehelper-backend -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n "${NAMESPACE}" "${backend_pod}" -- curl -f http://localhost:8080/health; then
        log_info "后端服务健康检查通过"
    else
        log_error "后端服务健康检查失败"
        exit 1
    fi
    
    # 运行烟雾测试
    log_info "运行烟雾测试..."
    kubectl run smoke-test --rm -i --restart=Never --image=curlimages/curl -n "${NAMESPACE}" -- \
        curl -f "http://voicehelper-backend-service:8080/api/health"
    
    log_info "部署验证完成"
}

# 回滚函数
rollback() {
    log_warn "执行回滚..."
    helm rollback voicehelper -n "${NAMESPACE}"
    log_info "回滚完成"
}

# 清理函数
cleanup() {
    log_info "清理临时资源..."
    kubectl delete pod smoke-test -n "${NAMESPACE}" --ignore-not-found=true
}

# 主函数
main() {
    log_info "开始部署流程..."
    
    # 设置错误处理
    trap 'log_error "部署失败，执行清理..."; cleanup; exit 1' ERR
    trap 'cleanup' EXIT
    
    check_dependencies
    pre_deploy_check
    backup_current_deployment
    deploy
    post_deploy_verification
    
    log_info "部署成功完成！"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [环境] [版本]"
    echo ""
    echo "参数:"
    echo "  环境    部署环境 (staging|production)，默认: staging"
    echo "  版本    镜像版本标签，默认: latest"
    echo ""
    echo "示例:"
    echo "  $0 staging v1.2.3"
    echo "  $0 production latest"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -r, --rollback 回滚到上一个版本"
}

# 处理命令行参数
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -r|--rollback)
        rollback
        exit 0
        ;;
    *)
        main
        ;;
esac
```

#### 9.4.2 监控和告警自动化

```python
# 自动化监控脚本
# 文件路径: scripts/monitoring_automation.py
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yaml
import json

class MonitoringAutomation:
    """监控自动化系统"""
    
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_service_health(self) -> Dict[str, bool]:
        """检查服务健康状态"""
        health_status = {}
        
        for service in self.config['services']:
            try:
                url = f"{service['url']}/health"
                async with self.session.get(url, timeout=10) as response:
                    health_status[service['name']] = response.status == 200
            except Exception as e:
                self.logger.error(f"Health check failed for {service['name']}: {e}")
                health_status[service['name']] = False
        
        return health_status
    
    async def check_metrics(self) -> Dict[str, Any]:
        """检查关键指标"""
        metrics = {}
        
        # 查询Prometheus指标
        prometheus_url = self.config['prometheus']['url']
        queries = self.config['prometheus']['queries']
        
        for metric_name, query in queries.items():
            try:
                url = f"{prometheus_url}/api/v1/query"
                params = {'query': query}
                
                async with self.session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data['status'] == 'success' and data['data']['result']:
                        value = float(data['data']['result'][0]['value'][1])
                        metrics[metric_name] = value
                    else:
                        metrics[metric_name] = None
                        
            except Exception as e:
                self.logger.error(f"Failed to query metric {metric_name}: {e}")
                metrics[metric_name] = None
        
        return metrics
    
    async def auto_scale_services(self, metrics: Dict[str, Any]):
        """自动扩缩容服务"""
        scaling_rules = self.config['auto_scaling']
        
        for rule in scaling_rules:
            service_name = rule['service']
            metric_name = rule['metric']
            threshold = rule['threshold']
            action = rule['action']
            
            if metric_name in metrics and metrics[metric_name] is not None:
                current_value = metrics[metric_name]
                
                if current_value > threshold:
                    await self._scale_service(service_name, action)
    
    async def _scale_service(self, service_name: str, action: str):
        """执行服务扩缩容"""
        try:
            if action == 'scale_up':
                # 扩容逻辑
                cmd = f"kubectl scale deployment {service_name} --replicas=5 -n voicehelper"
            elif action == 'scale_down':
                # 缩容逻辑
                cmd = f"kubectl scale deployment {service_name} --replicas=2 -n voicehelper"
            else:
                return
            
            import subprocess
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully scaled {service_name}: {action}")
            else:
                self.logger.error(f"Failed to scale {service_name}: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error scaling service {service_name}: {e}")
    
    async def cleanup_old_data(self):
        """清理旧数据"""
        cleanup_tasks = self.config['cleanup']
        
        for task in cleanup_tasks:
            if task['type'] == 'database':
                await self._cleanup_database(task)
            elif task['type'] == 'logs':
                await self._cleanup_logs(task)
            elif task['type'] == 'metrics':
                await self._cleanup_metrics(task)
    
    async def _cleanup_database(self, task: Dict[str, Any]):
        """清理数据库数据"""
        try:
            # 这里应该连接到数据库执行清理
            table = task['table']
            retention_days = task['retention_days']
            
            # 示例：删除超过保留期的数据
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            self.logger.info(f"Cleaning up {table} data older than {cutoff_date}")
            # 实际的数据库清理逻辑
            
        except Exception as e:
            self.logger.error(f"Database cleanup failed: {e}")
    
    async def _cleanup_logs(self, task: Dict[str, Any]):
        """清理日志文件"""
        try:
            import os
            import glob
            
            log_pattern = task['pattern']
            retention_days = task['retention_days']
            
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            for log_file in glob.glob(log_pattern):
                file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                if file_time < cutoff_time:
                    os.remove(log_file)
                    self.logger.info(f"Removed old log file: {log_file}")
                    
        except Exception as e:
            self.logger.error(f"Log cleanup failed: {e}")
    
    async def generate_report(self, health_status: Dict[str, bool], metrics: Dict[str, Any]):
        """生成监控报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'metrics': metrics,
            'summary': {
                'healthy_services': sum(health_status.values()),
                'total_services': len(health_status),
                'critical_metrics': []
            }
        }
        
        # 检查关键指标
        critical_thresholds = self.config['critical_thresholds']
        for metric_name, threshold in critical_thresholds.items():
            if metric_name in metrics and metrics[metric_name] is not None:
                if metrics[metric_name] > threshold:
                    report['summary']['critical_metrics'].append({
                        'metric': metric_name,
                        'value': metrics[metric_name],
                        'threshold': threshold
                    })
        
        # 保存报告
        report_file = f"reports/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Monitoring report saved: {report_file}")
        
        return report
    
    async def run_monitoring_cycle(self):
        """运行一次完整的监控周期"""
        self.logger.info("Starting monitoring cycle...")
        
        try:
            # 检查服务健康状态
            health_status = await self.check_service_health()
            
            # 检查指标
            metrics = await self.check_metrics()
            
            # 自动扩缩容
            await self.auto_scale_services(metrics)
            
            # 清理旧数据
            await self.cleanup_old_data()
            
            # 生成报告
            report = await self.generate_report(health_status, metrics)
            
            # 发送告警（如果需要）
            await self._send_alerts_if_needed(report)
            
            self.logger.info("Monitoring cycle completed successfully")
            
        except Exception as e:
            self.logger.error(f"Monitoring cycle failed: {e}")
    
    async def _send_alerts_if_needed(self, report: Dict[str, Any]):
        """根据需要发送告警"""
        # 检查是否有服务不健康
        unhealthy_services = [
            service for service, healthy in report['health_status'].items()
            if not healthy
        ]
        
        if unhealthy_services:
            await self._send_alert(
                "Service Health Alert",
                f"Unhealthy services detected: {', '.join(unhealthy_services)}"
            )
        
        # 检查关键指标
        if report['summary']['critical_metrics']:
            critical_metrics_text = '\n'.join([
                f"- {metric['metric']}: {metric['value']} (threshold: {metric['threshold']})"
                for metric in report['summary']['critical_metrics']
            ])
            
            await self._send_alert(
                "Critical Metrics Alert",
                f"Critical metrics detected:\n{critical_metrics_text}"
            )
    
    async def _send_alert(self, title: str, message: str):
        """发送告警"""
        alert_config = self.config['alerts']
        
        # 发送到钉钉
        if 'dingtalk' in alert_config:
            await self._send_dingtalk_alert(title, message, alert_config['dingtalk'])
        
        # 发送邮件
        if 'email' in alert_config:
            await self._send_email_alert(title, message, alert_config['email'])
    
    async def _send_dingtalk_alert(self, title: str, message: str, config: Dict[str, str]):
        """发送钉钉告警"""
        webhook_url = config['webhook_url']
        
        payload = {
            "msgtype": "text",
            "text": {
                "content": f"🚨 {title}\n\n{message}\n\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        }
        
        try:
            async with self.session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    self.logger.info("DingTalk alert sent successfully")
                else:
                    self.logger.error(f"Failed to send DingTalk alert: {response.status}")
        except Exception as e:
            self.logger.error(f"Error sending DingTalk alert: {e}")

async def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async with MonitoringAutomation('config/monitoring.yaml') as monitor:
        # 运行监控周期
        await monitor.run_monitoring_cycle()

if __name__ == "__main__":
    asyncio.run(main())
```

#### 9.4.3 灾难恢复自动化

```bash
#!/bin/bash
# 灾难恢复脚本
# 文件路径: scripts/disaster_recovery.sh

set -e

# 配置变量
BACKUP_LOCATION=${BACKUP_LOCATION:-"s3://voicehelper-backups"}
RECOVERY_NAMESPACE=${RECOVERY_NAMESPACE:-"voicehelper-recovery"}
RECOVERY_TYPE=${1:-"full"}  # full, database, files

# 日志函数
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# 检查备份完整性
check_backup_integrity() {
    log "检查备份完整性..."
    
    local backup_date=${2:-$(date -d "yesterday" +%Y%m%d)}
    local backup_path="${BACKUP_LOCATION}/${backup_date}"
    
    # 检查数据库备份
    if aws s3 ls "${backup_path}/database.sql.gz" > /dev/null 2>&1; then
        log "数据库备份文件存在"
    else
        log "错误: 数据库备份文件不存在"
        exit 1
    fi
    
    # 检查文件备份
    if aws s3 ls "${backup_path}/files.tar.gz" > /dev/null 2>&1; then
        log "文件备份存在"
    else
        log "错误: 文件备份不存在"
        exit 1
    fi
    
    # 检查配置备份
    if aws s3 ls "${backup_path}/configs.tar.gz" > /dev/null 2>&1; then
        log "配置备份存在"
    else
        log "错误: 配置备份不存在"
        exit 1
    fi
    
    log "备份完整性检查通过"
}

# 恢复数据库
recover_database() {
    log "开始恢复数据库..."
    
    local backup_date=${1:-$(date -d "yesterday" +%Y%m%d)}
    local backup_path="${BACKUP_LOCATION}/${backup_date}"
    
    # 下载数据库备份
    aws s3 cp "${backup_path}/database.sql.gz" /tmp/database.sql.gz
    gunzip /tmp/database.sql.gz
    
    # 创建临时PostgreSQL实例
    kubectl run postgres-recovery --image=postgres:15 \
        --env="POSTGRES_PASSWORD=recovery123" \
        --env="POSTGRES_DB=voicehelper" \
        -n "${RECOVERY_NAMESPACE}" \
        --restart=Never
    
    # 等待PostgreSQL启动
    kubectl wait --for=condition=ready pod/postgres-recovery -n "${RECOVERY_NAMESPACE}" --timeout=300s
    
    # 恢复数据库
    kubectl cp /tmp/database.sql "${RECOVERY_NAMESPACE}/postgres-recovery:/tmp/database.sql"
    kubectl exec -n "${RECOVERY_NAMESPACE}" postgres-recovery -- \
        psql -U postgres -d voicehelper -f /tmp/database.sql
    
    log "数据库恢复完成"
}

# 恢复文件
recover_files() {
    log "开始恢复文件..."
    
    local backup_date=${1:-$(date -d "yesterday" +%Y%m%d)}
    local backup_path="${BACKUP_LOCATION}/${backup_date}"
    
    # 下载文件备份
    aws s3 cp "${backup_path}/files.tar.gz" /tmp/files.tar.gz
    
    # 创建恢复PVC
    kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: recovery-files-pvc
  namespace: ${RECOVERY_NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
EOF
    
    # 创建恢复Pod
    kubectl run file-recovery --image=alpine \
        --command -- sleep 3600 \
        -n "${RECOVERY_NAMESPACE}" \
        --restart=Never
    
    kubectl patch pod file-recovery -n "${RECOVERY_NAMESPACE}" -p '
    {
        "spec": {
            "volumes": [
                {
                    "name": "recovery-storage",
                    "persistentVolumeClaim": {
                        "claimName": "recovery-files-pvc"
                    }
                }
            ],
            "containers": [
                {
                    "name": "file-recovery",
                    "volumeMounts": [
                        {
                            "name": "recovery-storage",
                            "mountPath": "/recovery"
                        }
                    ]
                }
            ]
        }
    }'
    
    # 等待Pod启动
    kubectl wait --for=condition=ready pod/file-recovery -n "${RECOVERY_NAMESPACE}" --timeout=300s
    
    # 复制并解压文件
    kubectl cp /tmp/files.tar.gz "${RECOVERY_NAMESPACE}/file-recovery:/recovery/files.tar.gz"
    kubectl exec -n "${RECOVERY_NAMESPACE}" file-recovery -- \
        tar -xzf /recovery/files.tar.gz -C /recovery/
    
    log "文件恢复完成"
}

# 恢复配置
recover_configs() {
    log "开始恢复配置..."
    
    local backup_date=${1:-$(date -d "yesterday" +%Y%m%d)}
    local backup_path="${BACKUP_LOCATION}/${backup_date}"
    
    # 下载配置备份
    aws s3 cp "${backup_path}/configs.tar.gz" /tmp/configs.tar.gz
    tar -xzf /tmp/configs.tar.gz -C /tmp/
    
    # 恢复ConfigMaps和Secrets
    kubectl apply -f /tmp/configs/ -n "${RECOVERY_NAMESPACE}"
    
    log "配置恢复完成"
}

# 验证恢复
verify_recovery() {
    log "验证恢复结果..."
    
    # 检查数据库连接
    kubectl exec -n "${RECOVERY_NAMESPACE}" postgres-recovery -- \
        psql -U postgres -d voicehelper -c "SELECT COUNT(*) FROM users;"
    
    # 检查文件
    kubectl exec -n "${RECOVERY_NAMESPACE}" file-recovery -- \
        ls -la /recovery/
    
    # 检查配置
    kubectl get configmaps,secrets -n "${RECOVERY_NAMESPACE}"
    
    log "恢复验证完成"
}

# 清理恢复环境
cleanup_recovery() {
    log "清理恢复环境..."
    
    kubectl delete namespace "${RECOVERY_NAMESPACE}" --ignore-not-found=true
    rm -f /tmp/database.sql /tmp/files.tar.gz /tmp/configs.tar.gz
    rm -rf /tmp/configs/
    
    log "清理完成"
}

# 完整恢复流程
full_recovery() {
    local backup_date=$1
    
    log "开始完整恢复流程..."
    
    # 创建恢复命名空间
    kubectl create namespace "${RECOVERY_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    check_backup_integrity "${backup_date}"
    recover_database "${backup_date}"
    recover_files "${backup_date}"
    recover_configs "${backup_date}"
    verify_recovery
    
    log "完整恢复流程完成"
    log "恢复的资源位于命名空间: ${RECOVERY_NAMESPACE}"
    log "请验证恢复结果后手动清理恢复环境"
}

# 主函数
main() {
    case "${RECOVERY_TYPE}" in
        "full")
            full_recovery "$2"
            ;;
        "database")
            recover_database "$2"
            ;;
        "files")
            recover_files "$2"
            ;;
        "configs")
            recover_configs "$2"
            ;;
        "cleanup")
            cleanup_recovery
            ;;
        *)
            echo "用法: $0 [full|database|files|configs|cleanup] [backup_date]"
            echo "示例: $0 full 20231201"
            exit 1
            ;;
    esac
}

# 错误处理
trap 'log "恢复过程中发生错误，执行清理..."; cleanup_recovery; exit 1' ERR

main "$@"
```

## 相关文档

- [系统架构概览](/posts/voicehelper-architecture-overview/)
- [前端模块深度解析](/posts/voicehelper-frontend-modules/)
- [后端服务核心实现](/posts/voicehelper-backend-services/)
- [AI算法引擎深度分析](/posts/voicehelper-ai-algorithms/)
- [数据存储架构](/posts/voicehelper-data-storage/)
- [系统交互时序图](/posts/voicehelper-system-interactions/)
- [第三方集成与扩展](/posts/voicehelper-third-party-integration/)
- [性能优化与监控](/posts/voicehelper-performance-optimization/)
- [总结与最佳实践](/posts/voicehelper-best-practices/)
- [项目功能清单](/posts/voicehelper-feature-inventory/)
- [版本迭代历程](/posts/voicehelper-version-history/)
- [竞争力分析](/posts/voicehelper-competitive-analysis/)
- [API接口清单](/posts/voicehelper-api-reference/)
- [错误码系统](/posts/voicehelper-error-codes/)
- [版本迭代计划](/posts/voicehelper-version-roadmap/)
