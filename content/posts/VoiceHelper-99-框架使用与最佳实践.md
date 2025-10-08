---
title: "VoiceHelper-99-框架使用与最佳实践"
date: 2025-01-08T11:30:00+08:00
draft: false
tags:
  - 源码剖析
  - 架构分析
  - 源码分析
  - 最佳实践
  - 开发指南
  - 实战经验
categories:
  - AI应用
  - VoiceHelper
  - 最佳实践
description: "源码剖析 - VoiceHelper 框架使用与最佳实践指南"
author: "源码分析"
weight: 599
ShowToc: true
TocOpen: true
---

# VoiceHelper-99-框架使用与最佳实践

## 概述

本文档提供 VoiceHelper 项目的框架使用示例、实战经验和最佳实践，帮助开发者快速上手并避免常见问题。

---

## 1. 快速开始：5分钟搭建本地开发环境

### 1.1 前置条件

```bash
# 检查依赖
docker --version       # Docker 20.10+
docker-compose --version  # Docker Compose 1.29+
go version            # Go 1.21+
python --version      # Python 3.11+
node --version        # Node.js 18+
```

### 1.2 克隆仓库

```bash
git clone https://github.com/yourcompany/voicehelper.git
cd voicehelper
```

### 1.3 配置环境变量

```bash
cp .env.example .env

# 编辑 .env 文件，填写必要的API密钥
vim .env
```

最小配置：

```bash
# OpenAI API (必需)
OPENAI_API_KEY=sk-xxx

# JWT 密钥 (必需)
JWT_SECRET=your-secret-key-here

# 数据库配置 (可选，Docker Compose 会自动创建)
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USERNAME=voicehelper
DATABASE_PASSWORD=voicehelper123
DATABASE_NAME=voicehelper

# Redis 配置 (可选)
REDIS_URL=redis://localhost:6379/0
```

### 1.4 启动服务

```bash
# 使用 Docker Compose 一键启动
make docker-up

# 或手动启动
docker-compose -f deployment/docker-compose/docker-compose-v04-full.yml up -d
```

### 1.5 验证服务

```bash
# 检查 Backend 网关
curl http://localhost:8080/health

# 检查算法服务
curl http://localhost:8000/health

# 预期输出
{
  "status": "healthy",
  "version": "v0.6",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "algo_service": "healthy"
  }
}
```

### 1.6 运行示例

```bash
# 1. 注册用户
curl -X POST http://localhost:8080/api/v01/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "password": "Demo@123456",
    "email": "demo@example.com"
  }'

# 2. 登录获取 Token
TOKEN=$(curl -X POST http://localhost:8080/api/v01/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "password": "Demo@123456"
  }' | jq -r '.access_token')

# 3. 创建会话
SESSION_ID=$(curl -X POST http://localhost:8080/api/v01/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "测试会话"}' | jq -r '.session_id')

# 4. 发起聊天
curl -X POST http://localhost:8080/api/v01/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"query\": \"你好，请介绍一下自己\"
  }"
```

---

## 2. 核心功能使用示例

### 2.1 文档摄取与检索

**场景**：将公司内部文档导入系统，实现智能问答

**步骤 1：准备文档**

```bash
# 支持的文件格式
mkdir -p /tmp/docs
echo "VoiceHelper是一个多平台AI语音助手" > /tmp/docs/intro.txt
```

**步骤 2：文档摄取**

```python
import requests
import time

# 1. 发起摄取请求
response = requests.post(
    "http://localhost:8000/ingest",
    json={
        "files": ["/tmp/docs/intro.txt"],
        "collection_name": "company_docs",
        "metadata": {
            "department": "产品部",
            "version": "1.0"
        }
    }
)

task_id = response.json()["task_id"]
print(f"任务ID: {task_id}")

# 2. 轮询任务状态
while True:
    status = requests.get(f"http://localhost:8000/tasks/{task_id}").json()
    print(f"进度: {status['progress']}% - {status['status']}")
    
    if status["status"] in ["completed", "failed"]:
        break
    
    time.sleep(2)

print("摄取完成！")
```

**步骤 3：智能检索**

```python
import requests
import json

# 流式查询
response = requests.post(
    "http://localhost:8000/query",
    json={
        "messages": [
            {"role": "user", "content": "VoiceHelper是什么？"}
        ],
        "top_k": 5,
        "temperature": 0.7
    },
    stream=True
)

print("检索结果：")
for line in response.iter_lines():
    if line:
        data = json.loads(line)
        
        if data["type"] == "refs":
            # 打印检索到的文档
            print("\n相关文档：")
            for ref in data.get("data", []):
                print(f"- {ref['title']} (相似度: {ref['score']:.2f})")
        
        elif data["type"] == "token":
            # 打印生成的文本
            print(data["content"], end="", flush=True)
        
        elif data["type"] == "end":
            print("\n\n生成完成！")
```

**最佳实践**：

1. **分批摄取**：大量文档分批处理，每批 10-20 个文件
2. **元数据标记**：添加丰富的元数据（部门、版本、作者）
3. **定期更新**：文档更新后重新摄取，保持索引新鲜
4. **监控进度**：使用任务状态 API 监控摄取进度

---

### 2.2 实时语音对话

**场景**：实现端到端的语音问答系统

**前端代码（Web）**：

```typescript
// components/VoiceChat.tsx
import { useState, useRef } from 'react';

export default function VoiceChat() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [response, setResponse] = useState('');
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // 开始录音
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        processAudio();
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('无法访问麦克风', error);
    }
  };

  // 停止录音
  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // 处理音频
  const processAudio = async () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
    
    // 转换为 Base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = async () => {
      const base64Audio = reader.result?.toString().split(',')[1];
      
      // 发送到后端
      await sendVoiceQuery(base64Audio!);
    };
  };

  // 发送语音查询
  const sendVoiceQuery = async (audioData: string) => {
    const token = localStorage.getItem('access_token');
    const sessionId = localStorage.getItem('session_id');

    const response = await fetch('http://localhost:8000/voice/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        audio_data: audioData,
        session_id: sessionId,
        format: 'wav',
      }),
    });

    // 处理流式响应
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (!line) continue;
        const data = JSON.parse(line);

        if (data.type === 'transcription') {
          setTranscription(data.text);
        } else if (data.type === 'audio_chunk') {
          // 播放音频
          playAudio(data.data);
        }
      }
    }
  };

  // 播放音频
  const playAudio = (base64Audio: string) => {
    const audioBlob = base64ToBlob(base64Audio, 'audio/mp3');
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
  };

  return (
    <div className="voice-chat">
      <button 
        onMouseDown={startRecording}
        onMouseUp={stopRecording}
        className={isRecording ? 'recording' : ''}
      >
        {isRecording ? '松开发送' : '按住说话'}
      </button>
      
      {transcription && (
        <div className="transcription">
          <strong>你说：</strong> {transcription}
        </div>
      )}
      
      {response && (
        <div className="response">
          <strong>回答：</strong> {response}
        </div>
      )}
    </div>
  );
}
```

**WebSocket 实时语音**：

```typescript
// hooks/useVoiceWebSocket.ts
import { useState, useEffect, useRef } from 'react';

export function useVoiceWebSocket(sessionId: string) {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // 建立 WebSocket 连接
    const token = localStorage.getItem('access_token');
    const ws = new WebSocket(
      `ws://localhost:8080/api/v01/voice/realtime?token=${token}&session_id=${sessionId}`
    );

    ws.onopen = () => {
      console.log('WebSocket 连接已建立');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleMessage(data);
    };

    ws.onclose = () => {
      console.log('WebSocket 连接已关闭');
      setIsConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [sessionId]);

  // 发送音频数据
  const sendAudio = (audioData: ArrayBuffer) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'audio',
        data: arrayBufferToBase64(audioData),
      }));
    }
  };

  // 处理消息
  const handleMessage = (data: any) => {
    switch (data.type) {
      case 'transcription':
        console.log('识别结果:', data.text);
        break;
      case 'audio_chunk':
        playAudio(data.data);
        break;
      case 'error':
        console.error('错误:', data.message);
        break;
    }
  };

  return { isConnected, sendAudio };
}
```

**最佳实践**：

1. **音频格式**：使用 PCM 16bit 16kHz 单声道，延迟最低
2. **Chunk 大小**：320 字节（20ms），平衡延迟和网络开销
3. **打断处理**：检测到新语音立即发送 interrupt 信号
4. **错误重试**：WebSocket 断开后指数退避重连
5. **音频缓冲**：客户端维护音频缓冲队列，平滑播放

---

### 2.3 多模态分析（图像理解）

**场景**：上传图片并进行智能分析

**Python 示例**：

```python
import requests
import base64

# 1. 读取图片
with open("product_image.jpg", "rb") as f:
    image_data = f.read()

# 2. 调用图像分析 API
response = requests.post(
    "http://localhost:8000/api/v3/image/analyze",
    files={"file": ("product_image.jpg", image_data, "image/jpeg")},
    data={
        "query": "这个产品有什么特点？",
        "provider": "openai"
    }
)

result = response.json()
print("分析结果:", result["data"]["description"])
```

**OCR 文字提取**：

```python
# 提取图片中的文字
response = requests.post(
    "http://localhost:8000/api/v3/image/ocr",
    files={"file": ("document.png", image_data, "image/png")},
    data={"engine": "paddle"}
)

text = response.json()["data"]["text"]
print("提取的文字:", text)
```

**最佳实践**：

1. **图片预处理**：压缩大图片（最大 4MB）
2. **批量处理**：使用异步队列处理大量图片
3. **成本控制**：优先使用免费 OCR（PaddleOCR），复杂场景使用 GPT-4V
4. **结果缓存**：相同图片缓存分析结果

---

## 3. 架构最佳实践

### 3.1 API 设计

**RESTful API 规范**：

```typescript
// ✅ 好的 API 设计
POST /api/v01/sessions          // 创建会话
GET  /api/v01/sessions          // 列出会话
GET  /api/v01/sessions/:id      // 获取会话详情
DELETE /api/v01/sessions/:id    // 删除会话

// ❌ 不好的设计
POST /api/v01/createSession     // 动词不应出现在 URL 中
GET  /api/v01/getSessionById    // 冗余
```

**统一响应格式**：

```typescript
// 成功响应
{
  "code": 0,
  "message": "success",
  "data": { ... }
}

// 错误响应
{
  "code": 40001,
  "message": "用户名已存在",
  "request_id": "abc-123-def"
}
```

### 3.2 错误处理

**Go 后端**：

```go
// 定义错误类型
type AppError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
    Cause   error  `json:"-"`
}

func (e *AppError) Error() string {
    return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// 使用自定义错误
func (h *Handler) CreateSession(c *gin.Context) {
    session, err := h.service.Create(userID)
    if err != nil {
        if errors.Is(err, ErrSessionLimitExceeded) {
            c.JSON(429, &AppError{
                Code:    "SESSION_LIMIT_EXCEEDED",
                Message: "会话数量已达上限",
            })
            return
        }
        
        // 未知错误
        logrus.WithError(err).Error("Failed to create session")
        c.JSON(500, &AppError{
            Code:    "INTERNAL_ERROR",
            Message: "服务器内部错误",
        })
        return
    }
    
    c.JSON(201, session)
}
```

**Python 算法服务**：

```python
from enum import Enum
from fastapi import HTTPException

class ErrorCode(Enum):
    RAG_INVALID_QUERY = "RAG_INVALID_QUERY"
    RAG_INDEXING_FAILED = "RAG_INDEXING_FAILED"
    VOICE_PROCESSING_FAILED = "VOICE_PROCESSING_FAILED"

class VoiceHelperError(Exception):
    def __init__(self, code: ErrorCode, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}
        self.http_status = self._get_http_status(code)
    
    def _get_http_status(self, code: ErrorCode) -> int:
        mapping = {
            ErrorCode.RAG_INVALID_QUERY: 400,
            ErrorCode.RAG_INDEXING_FAILED: 500,
        }
        return mapping.get(code, 500)
    
    def to_dict(self):
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details
        }

# 使用
@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    if not request.files:
        raise VoiceHelperError(
            ErrorCode.RAG_INVALID_QUERY,
            "没有提供文件"
        )
    
    try:
        result = await service.ingest(request)
        return result
    except Exception as e:
        raise VoiceHelperError(
            ErrorCode.RAG_INDEXING_FAILED,
            f"文档入库失败: {str(e)}"
        )
```

### 3.3 日志规范

**结构化日志**：

```go
// Go - 使用 logrus
logrus.WithFields(logrus.Fields{
    "user_id":    userID,
    "session_id": sessionID,
    "query":      query,
    "duration":   duration,
}).Info("Query completed")
```

```python
# Python - 使用自定义 logger
logger.business("文档入库请求", context={
    "files_count": len(request.files),
    "collection_name": request.collection_name,
    "client_ip": request.client.host,
})
```

**日志级别**：

- `DEBUG`：详细的调试信息（仅开发环境）
- `INFO`：正常业务日志（请求、响应）
- `WARN`：警告信息（降级、重试）
- `ERROR`：错误信息（异常、失败）
- `FATAL`：致命错误（服务无法启动）

### 3.4 性能优化

**数据库查询优化**：

```go
// ❌ N+1 查询问题
sessions, _ := db.Query("SELECT * FROM sessions WHERE user_id = ?", userID)
for _, session := range sessions {
    messages, _ := db.Query("SELECT * FROM messages WHERE session_id = ?", session.ID)
    // 处理消息
}

// ✅ 使用 JOIN 或预加载
sessions, _ := db.Query(`
    SELECT s.*, m.* 
    FROM sessions s
    LEFT JOIN messages m ON m.session_id = s.id
    WHERE s.user_id = ?
`, userID)
```

**缓存策略**：

```python
# 语义缓存
import hashlib

def get_cache_key(query: str) -> str:
    # 使用查询向量的哈希作为缓存键
    query_embedding = embeddings.embed_query(query)
    return hashlib.md5(str(query_embedding).encode()).hexdigest()

async def retrieve_with_cache(query: str):
    cache_key = get_cache_key(query)
    
    # 检查缓存
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 执行检索
    result = await retrieve(query)
    
    # 缓存结果（24小时）
    await redis.setex(cache_key, 86400, json.dumps(result))
    
    return result
```

**批处理优化**：

```python
# ❌ 逐个处理
for text in texts:
    vector = embeddings.embed_query(text)
    vectors.append(vector)

# ✅ 批量处理
vectors = embeddings.embed_documents(texts)  # 一次 API 调用
```

---

## 4. 生产环境部署

### 4.1 Docker 部署

**最小化 Docker 镜像**：

```dockerfile
# Backend - 多阶段构建
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /gateway ./cmd/gateway

FROM alpine:latest
RUN apk --no-cache add ca-certificates
COPY --from=builder /gateway /gateway
EXPOSE 8080
CMD ["/gateway"]
```

```dockerfile
# Algo Service
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 Kubernetes 部署

**Deployment 配置**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-gateway
spec:
  replicas: 3  # 3个副本
  selector:
    matchLabels:
      app: backend-gateway
  template:
    metadata:
      labels:
        app: backend-gateway
    spec:
      containers:
      - name: gateway
        image: voicehelper/backend:v0.6
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_HOST
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database_host
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: jwt_secret
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
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

**HPA 自动扩缩容**：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend-gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 4.3 监控告警

**Prometheus 告警规则**：

```yaml
groups:
- name: voicehelper_alerts
  rules:
  - alert: HighErrorRate
    expr: |
      rate(http_requests_total{status=~"5.."}[5m]) /
      rate(http_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "高错误率检测"
      description: "{{ $labels.instance }} 错误率超过 5%"

  - alert: HighLatency
    expr: |
      histogram_quantile(0.95, 
        rate(http_request_duration_seconds_bucket[5m])
      ) > 1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "高延迟检测"
      description: "{{ $labels.instance }} P95 延迟超过 1 秒"
```

---

## 5. 常见问题与解决方案

### 5.1 数据库连接池耗尽

**问题现象**：

```
Error: pq: sorry, too many clients already
```

**解决方案**：

```go
// 优化连接池配置
db.SetMaxOpenConns(100)  // 最大连接数
db.SetMaxIdleConns(10)   // 空闲连接数
db.SetConnMaxLifetime(time.Hour)  // 连接最大生命周期

// 确保每次查询后释放连接
rows, err := db.Query(...)
defer rows.Close()  // 必须 defer Close
```

### 5.2 Redis 内存不足

**问题现象**：

```
OOM command not allowed when used memory > 'maxmemory'
```

**解决方案**：

```redis
# 配置内存淘汰策略
maxmemory 2gb
maxmemory-policy allkeys-lru  # LRU 淘汰策略

# 为缓存设置过期时间
SET cache_key value EX 3600  # 1小时过期
```

### 5.3 LLM API 超时

**问题现象**：

```
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='api.openai.com', port=443)
```

**解决方案**：

```python
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def call_llm_with_retry(prompt: str):
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            timeout=30,  # 30秒超时
        )
        return response
    except openai.error.Timeout:
        logger.warning("LLM API timeout, retrying...")
        raise  # 触发重试
    except openai.error.APIError as e:
        logger.error(f"LLM API error: {e}")
        # 降级到备用模型
        return await fallback_model.generate(prompt)
```

---

## 6. 代码审查清单

### 6.1 通用清单

- [ ] 所有函数都有清晰的注释
- [ ] 错误处理完整（不能忽略错误）
- [ ] 日志记录完整（关键操作都有日志）
- [ ] 参数验证完整（防止注入攻击）
- [ ] 单元测试覆盖率 > 80%
- [ ] 没有硬编码的配置（使用环境变量）
- [ ] 敏感信息不能打印到日志
- [ ] 数据库连接、文件句柄正确关闭

### 6.2 Go 特定清单

- [ ] goroutine 泄漏检查（使用 context 控制生命周期）
- [ ] channel 是否正确关闭
- [ ] defer 语句顺序正确（LIFO）
- [ ] 使用 errgroup 管理并发错误
- [ ] 避免在循环中创建大量 goroutine

### 6.3 Python 特定清单

- [ ] 异步函数使用 async/await
- [ ] 避免阻塞事件循环（使用 asyncio.to_thread）
- [ ] 正确使用上下文管理器（with 语句）
- [ ] 类型注解完整（typing）
- [ ] 避免全局可变状态

---

## 7. 性能测试

### 7.1 压力测试脚本

```python
# locust_test.py
from locust import HttpUser, task, between
import random

class VoiceHelperUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # 登录获取 Token
        response = self.client.post("/api/v01/auth/login", json={
            "username": "test_user",
            "password": "Test@123456"
        })
        self.token = response.json()["access_token"]
        
        # 创建会话
        response = self.client.post(
            "/api/v01/sessions",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        self.session_id = response.json()["session_id"]
    
    @task(3)
    def chat(self):
        # 发起聊天
        self.client.post(
            "/api/v01/chat/stream",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "session_id": self.session_id,
                "query": random.choice([
                    "你好",
                    "今天天气怎么样",
                    "帮我总结一下文档"
                ])
            }
        )
    
    @task(1)
    def list_sessions(self):
        # 列出会话
        self.client.get(
            "/api/v01/sessions",
            headers={"Authorization": f"Bearer {self.token}"}
        )
```

运行压测：

```bash
locust -f locust_test.py --host http://localhost:8080 --users 100 --spawn-rate 10
```

### 7.2 性能基准

**目标指标**：

| 指标 | 目标值 | 测量方法 |
|---|---|---|
| 用户登录 | P95 < 200ms | Locust |
| 文档检索 | P95 < 500ms | Locust |
| 语音对话（端到端） | P95 < 3s | 手动测试 |
| API 错误率 | < 1% | Prometheus |
| 并发用户 | 1000+ | Locust |

---

**文档版本**：v1.0  
**最后更新**：2025-01-08  
**维护者**：VoiceHelper 团队

