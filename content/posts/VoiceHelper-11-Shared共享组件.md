---
title: "VoiceHelper-11-Shared共享组件"
date: 2025-10-10T10:11:00+08:00
draft: false
tags: ["VoiceHelper", "共享组件", "中间件", "工具函数", "类型定义", "SDK"]
categories: ["VoiceHelper", "源码剖析"]
description: "VoiceHelper 共享组件详细设计，包含Python/Go共享模块、中间件系统、工具库、类型定义、SDK完整实现"
series: ["VoiceHelper源码剖析"]
weight: 11
---

# VoiceHelper-11-Shared共享组件

## 1. 模块概览

### 1.1 职责边界

Shared共享组件是VoiceHelper架构的横切关注点,为所有微服务提供统一的基础设施能力。通过标准化的接口和实现,确保系统的一致性、可维护性和可观测性。

**Python共享组件(algo/shared)** 职责:
- **日志系统**:统一日志格式(时间戳+模块名+级别+消息+结构化字段)
- **配置管理**:基于Pydantic的强类型配置验证,支持环境变量和.env文件
- **错误处理**:统一错误码体系(RAG/Voice/Agent/Multimodal)和异常封装
- **健康检查**:标准化健康检查接口,支持组件级健康状态聚合
- **HTTP客户端**:封装httpx异步客户端,支持重试、超时、熔断
- **指标收集**:Prometheus格式指标导出(Counter/Gauge/Histogram)
- **响应格式**:统一API响应格式`{code, message, data/details}`
- **分布式追踪**:基于trace_id的链路追踪支持

**Go共享组件(backend/pkg)** 职责:
- **中间件系统**:JWT认证、API Key认证、RBAC权限、租户隔离、限流、日志、追踪
- **数据库连接池**:PostgreSQL连接池(最大100连接,空闲10,生命周期1h)
- **多级缓存**:L1(LRU内存缓存,毫秒级) + L2(Redis缓存,秒级)
- **消息队列**:RabbitMQ生产者/消费者封装,支持延迟队列和死信队列
- **安全组件**:AES-256-GCM加密、JWT签发/验证、密钥轮换、RSA非对称加密
- **监控系统**:健康检查、Prometheus指标、性能监控、内存监控
- **工具库**:参数验证、错误处理、通用工具函数

**边界与非目标**:
- 边界:仅提供基础设施能力,不包含业务逻辑
- 非目标:不处理业务数据转换、不实现业务流程编排
- 演进:支持按需扩展,保持向后兼容

---

### 1.2 整体架构图

```mermaid
flowchart TB
    subgraph "客户端层"
        CLIENT[Web/Mobile/小程序]
    end
    
    subgraph "网关层"
        GATEWAY[API Gateway<br/>Go]
    end
    
    subgraph "业务服务层"
        subgraph "Go Services"
            AUTH[Auth Service]
            DOC[Document Service]
            SESSION[Session Service]
            NOTIFY[Notification Service]
        end
        
        subgraph "Python Services"
            GRAPHRAG[GraphRAG Service]
            VOICE[Voice Service]
            AGENT[Agent Service]
            MULTIMODAL[Multimodal Service]
            LLMROUTER[LLM Router]
        end
    end
    
    subgraph "共享组件层 - Python (algo/shared)"
        direction LR
        PY_LOGGER[Logger<br/>日志系统<br/>结构化日志+级别控制]
        PY_CONFIG[Config<br/>配置管理<br/>Pydantic验证]
        PY_ERROR[Errors<br/>错误处理<br/>统一错误码]
        PY_HEALTH[Health<br/>健康检查<br/>组件状态聚合]
        PY_METRICS[Metrics<br/>指标收集<br/>Prometheus导出]
        PY_HTTP[HTTPClient<br/>HTTP客户端<br/>重试+熔断]
    end
    
    subgraph "共享组件层 - Go (backend/pkg)"
        direction LR
        GO_MIDDLEWARE[Middleware<br/>JWT认证<br/>RBAC权限<br/>限流]
        GO_DATABASE[Database<br/>连接池管理<br/>慢查询检测]
        GO_CACHE[Cache<br/>多级缓存<br/>L1内存+L2Redis]
        GO_QUEUE[Queue<br/>RabbitMQ<br/>生产者/消费者]
        GO_SECURITY[Security<br/>AES-256-GCM<br/>JWT+RSA]
        GO_MONITORING[Monitoring<br/>健康检查<br/>性能指标]
    end
    
    subgraph "基础设施层"
        POSTGRES[(PostgreSQL)]
        REDIS[(Redis)]
        NEO4J[(Neo4j)]
        RABBITMQ[RabbitMQ]
        PROMETHEUS[Prometheus]
        JAEGER[Jaeger]
    end
    
    CLIENT --> GATEWAY
    GATEWAY --> AUTH & DOC & SESSION & NOTIFY
    GATEWAY -.调用.-> GRAPHRAG & VOICE & AGENT & MULTIMODAL
    
    AUTH & DOC & SESSION & NOTIFY --> GO_MIDDLEWARE
    AUTH & DOC & SESSION & NOTIFY --> GO_DATABASE
    AUTH & DOC & SESSION & NOTIFY --> GO_CACHE
    AUTH & DOC & SESSION & NOTIFY --> GO_SECURITY
    
    GRAPHRAG & VOICE & AGENT & MULTIMODAL & LLMROUTER --> PY_LOGGER
    GRAPHRAG & VOICE & AGENT & MULTIMODAL & LLMROUTER --> PY_CONFIG
    GRAPHRAG & VOICE & AGENT & MULTIMODAL & LLMROUTER --> PY_ERROR
    GRAPHRAG & VOICE & AGENT & MULTIMODAL & LLMROUTER --> PY_HEALTH
    GRAPHRAG & VOICE & AGENT & MULTIMODAL & LLMROUTER --> PY_METRICS
    
    GO_DATABASE --> POSTGRES
    GO_CACHE --> REDIS
    GO_QUEUE --> RABBITMQ
    GRAPHRAG --> NEO4J
    
    PY_METRICS --> PROMETHEUS
    GO_MONITORING --> PROMETHEUS
    GO_MIDDLEWARE --> JAEGER
    
    style PY_LOGGER fill:#87CEEB
    style PY_METRICS fill:#87CEEB
    style GO_MIDDLEWARE fill:#FFB6C1
    style GO_CACHE fill:#FFB6C1
```

**架构图说明**:

1. **分层设计**:
   - 客户端层:多端接入(Web/Mobile/小程序)
   - 网关层:统一入口,路由转发,协议转换
   - 业务服务层:Go服务(基础服务) + Python服务(算法服务)
   - 共享组件层:Python组件(算法服务用) + Go组件(基础服务用)
   - 基础设施层:数据库、缓存、消息队列、监控

2. **调用关系**:
   - 客户端 → 网关 → 业务服务 → 共享组件 → 基础设施
   - Python服务通过PY_HTTP调用其他服务
   - Go服务通过GO_MIDDLEWARE处理所有请求
   - 所有服务通过PY_METRICS/GO_MONITORING上报指标到Prometheus

3. **关键特性**:
   - **解耦**:业务服务不直接依赖基础设施,通过共享组件解耦
   - **复用**:配置、日志、错误处理等通用能力统一封装
   - **可观测**:日志、指标、追踪三位一体
   - **可扩展**:新增服务可直接复用共享组件

4. **边界条件**:
   - Python/Go共享组件相互独立,通过HTTP/gRPC通信
   - 共享组件版本升级需保持向后兼容
   - 配置通过环境变量注入,支持多环境部署

5. **异常处理**:
   - 共享组件故障不影响业务服务(降级机制)
   - 基础设施故障通过健康检查快速发现
   - 指标采集失败不阻塞业务请求

6. **性能考量**:
   - L1缓存命中率>80%,延迟<1ms
   - L2缓存命中率>95%,延迟<10ms
   - 数据库连接池利用率<80%
   - 日志异步写入,不阻塞主流程

---

### 1.3 典型请求全链路时序图

以用户发起语音对话请求为例,展示从上游接口到各层组件的完整调用链路。

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant Gateway as API Gateway
    participant AuthMW as Auth中间件
    participant RateLimitMW as 限流中间件
    participant Cache as 多级缓存
    participant VoiceService as Voice服务
    participant Logger as 日志组件
    participant Metrics as 指标组件
    participant Health as 健康检查
    participant Redis as Redis
    participant Prometheus as Prometheus
    
    Note over Client,Prometheus: 1. 认证与限流阶段
    Client->>Gateway: POST /api/v1/voice/transcribe<br/>{audio: base64, language: zh}
    Gateway->>AuthMW: 验证JWT token
    AuthMW->>Cache: 查询token黑名单<br/>GET token:blacklist:{token}
    Cache->>Redis: L1未命中,查L2
    Redis-->>Cache: 未在黑名单
    Cache-->>AuthMW: token有效
    AuthMW->>Gateway: 注入user_id/tenant_id到Context
    Gateway->>RateLimitMW: 检查限流
    RateLimitMW->>Redis: INCR rate:user:{user_id}:minute
    Redis-->>RateLimitMW: 当前计数=15<100(限制)
    RateLimitMW->>Gateway: 放行请求
    
    Note over Gateway,Prometheus: 2. 业务处理阶段
    Gateway->>VoiceService: HTTP POST /transcribe<br/>转发请求+trace_id
    VoiceService->>Logger: 记录请求日志<br/>log_info("ASR请求", audio_len=1024)
    Logger->>Logger: 异步写入日志<br/>格式化为JSON
    VoiceService->>Metrics: 增加ASR请求计数<br/>asr_requests_total{model=whisper}
    Metrics->>Prometheus: 更新指标(异步)
    
    VoiceService->>VoiceService: 执行ASR处理<br/>(Whisper模型推理)
    VoiceService->>Metrics: 记录ASR延迟<br/>asr_latency_seconds=0.85s
    VoiceService->>Logger: 记录处理结果<br/>log_info("ASR成功", text="你好")
    
    Note over VoiceService,Cache: 3. 缓存与响应阶段
    VoiceService->>Cache: 缓存识别结果<br/>SET asr:{audio_hash} -> text
    Cache->>Cache: 写入L1(LRU)
    Cache->>Redis: 写入L2(TTL=5min)
    
    VoiceService-->>Gateway: 返回响应<br/>{code:0, data:{text:"你好"}}
    Gateway->>Metrics: 记录请求延迟<br/>request_duration_seconds=0.95s
    Gateway-->>Client: 200 OK + 响应体
    
    Note over Health,Prometheus: 4. 健康检查(定时任务)
    Health->>Redis: PING (检查连接)
    Redis-->>Health: PONG
    Health->>VoiceService: 检查服务状态
    VoiceService-->>Health: {status: healthy}
    Health->>Prometheus: 上报健康状态<br/>service_health{service=voice}=1
```

**全链路时序图说明**:

1. **认证与限流阶段(步骤1-7)**:
   - **目的**:确保请求合法且不超过限流阈值
   - **关键组件**:Auth中间件(JWT验证) + 多级缓存(token黑名单查询) + 限流中间件(Redis计数器)
   - **决策逻辑**:
     - 先查L1缓存(内存),命中直接返回(耗时<1ms)
     - L1未命中查L2(Redis),回写L1(耗时<10ms)
     - 限流使用滑动窗口算法,Redis原子操作保证并发安全
   - **异常场景**:
     - Token在黑名单 → 401 Unauthorized
     - 限流超限 → 429 Too Many Requests + Retry-After header
     - Redis不可用 → 降级到仅L1缓存或放行(根据配置)
   - **性能指标**:认证+限流总耗时<15ms(P95)

2. **业务处理阶段(步骤8-13)**:
   - **目的**:执行实际的ASR语音识别任务
   - **关键组件**:Voice服务(业务逻辑) + Logger(日志记录) + Metrics(指标上报)
   - **日志记录**:
     - 请求开始:记录trace_id、user_id、audio_len
     - 处理中:记录关键步骤(模型加载、推理)
     - 处理结束:记录结果、耗时、错误(如有)
     - 日志异步写入,不阻塞主流程
   - **指标上报**:
     - Counter:asr_requests_total(按model/language/status标签)
     - Histogram:asr_latency_seconds(分位数统计)
     - Gauge:active_connections(当前活跃连接)
   - **异常场景**:
     - 模型推理失败 → 记录错误日志 + 指标(status=failed) + 返回500
     - 超时 → context.WithTimeout取消 + 返回408
   - **性能指标**:ASR处理耗时<1s(P95)

3. **缓存与响应阶段(步骤14-18)**:
   - **目的**:缓存识别结果以加速后续相同请求,返回响应给客户端
   - **缓存策略**:
     - Key生成:audio_hash(SHA256) + language
     - TTL:L1=30s(热数据), L2=5min(温数据)
     - 写策略:同时写L1和L2(异步)
     - 失效:L1 LRU淘汰, L2 TTL过期
   - **响应格式**:
     - 成功:`{code:0, message:"success", data:{text:"..."}}`
     - 失败:`{code:"ERROR_CODE", message:"...", details:{...}}`
   - **性能指标**:缓存写入耗时<5ms

4. **健康检查(定时任务,步骤19-23)**:
   - **目的**:定期检查各组件健康状态,及时发现故障
   - **检查频率**:每10秒一次(配置可调)
   - **检查项**:
     - Redis连接:PING命令(超时5s)
     - 服务状态:调用服务health_check接口
     - 模型加载:检查模型是否可用
   - **健康状态聚合**:
     - 所有必需组件健康 → overall_status=healthy
     - 部分非必需组件不健康 → overall_status=degraded
     - 必需组件不健康 → overall_status=unhealthy
   - **告警**:状态变为unhealthy时触发Prometheus告警

**并发与一致性**:
- 缓存:最终一致性,短时间不一致可接受
- 限流:Redis原子操作保证严格一致
- 日志:异步写入,可能有秒级延迟
- 指标:批量上报,可能有10s延迟

**可观测性**:
- 日志:每个请求生成trace_id贯穿全链路
- 指标:多维度标签(service/endpoint/status)
- 追踪:通过Jaeger可视化调用链

**容量规划**:
- 网关QPS:10000(单实例)
- Redis QPS:50000(主从)
- 日志写入:1000条/s(异步批量)
- 指标上报:100次/s(批量)

---

## 2. Python共享组件

### 2.1 日志系统(Logger)

#### 2.1.1 模块职责与设计目标

**职责**:
- 提供统一的日志记录接口
- 支持结构化日志(JSON格式)
- 自动注入trace_id、timestamp等元信息
- 支持多级别日志(DEBUG/INFO/WARNING/ERROR/CRITICAL)
- 支持懒加载Logger实例

**设计目标**:
- **性能**:异步写入,不阻塞主流程(写入耗时<1ms)
- **可读性**:人类可读的格式+机器可解析的JSON
- **可扩展**:支持自定义formatter和handler
- **兼容性**:基于Python标准库logging,无额外依赖

#### 2.1.2 核心代码实现

```python
# algo/shared/logger.py

import logging
import sys
from typing import Any, Dict

# 配置基本日志(全局初始化)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # 输出到标准输出
    ]
)

def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    参数:
        name: 日志名称,通常为__name__(模块名)
        
    返回:
        logging.Logger: 配置好的日志记录器
        
    设计要点:
    - 每个模块/类使用独立的logger,便于日志过滤
    - logger名称反映代码结构,便于定位问题源头
    - 复用已存在的logger实例(logging.getLogger缓存)
    """
    return logging.getLogger(name)

class LoggerMixin:
    """
    日志混合类(Mixin模式)
    
    作用:为类提供logger属性和便捷日志方法
    使用:继承此类即可获得日志能力
    
    优点:
    - 避免在每个类中重复初始化logger
    - 自动使用类名作为logger名称
    - 提供简化的日志方法(log_info/log_error等)
    """
    
    @property
    def logger(self) -> logging.Logger:
        """
        获取logger实例(懒加载)
        
        实现细节:
        - 首次访问时创建logger并缓存到_logger
        - 后续访问直接返回缓存的logger
        - 使用类名(__class__.__name__)作为logger名称
        """
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
    
    def log_info(self, message: str, **kwargs):
        """
        记录信息日志
        
        参数:
            message: 日志消息
            **kwargs: 结构化字段(通过extra传递)
            
        示例:
            self.log_info("用户登录", user_id="123", ip="1.2.3.4")
        """
        self.logger.info(message, extra=kwargs)
    
    def log_error(self, message: str, **kwargs):
        """记录错误日志"""
        self.logger.error(message, extra=kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(message, extra=kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.logger.debug(message, extra=kwargs)
```

#### 2.1.3 日志系统内部时序图

```mermaid
sequenceDiagram
    autonumber
    participant Service as 业务服务<br/>(继承LoggerMixin)
    participant Mixin as LoggerMixin
    participant Logger as logging.Logger
    participant Handler as StreamHandler
    participant Formatter as Formatter
    participant Stdout as sys.stdout
    
    Note over Service,Stdout: 首次日志记录(懒加载)
    Service->>Mixin: self.logger (访问属性)
    Mixin->>Mixin: 检查_logger是否存在
    Mixin->>Mixin: _logger = get_logger(类名)
    Mixin->>Logger: logging.getLogger("GraphRAGService")
    Logger->>Logger: 查找或创建logger实例
    Logger-->>Mixin: 返回logger
    Mixin-->>Service: 返回logger
    
    Note over Service,Stdout: 记录INFO日志
    Service->>Mixin: self.log_info("文档索引", doc_id="123")
    Mixin->>Logger: logger.info(msg, extra={doc_id:"123"})
    Logger->>Logger: 判断日志级别(INFO>=INFO)
    Logger->>Logger: 创建LogRecord对象
    Logger->>Handler: 传递LogRecord到Handler
    Handler->>Handler: 判断Handler级别
    Handler->>Formatter: 格式化LogRecord
    Formatter->>Formatter: 应用format模板<br/>生成:"2025-01-15 10:30:45 - GraphRAGService - INFO - 文档索引"
    Formatter-->>Handler: 返回格式化字符串
    Handler->>Stdout: 写入标准输出
    
    Note over Service,Stdout: 记录ERROR日志(带异常)
    Service->>Service: 捕获异常
    Service->>Mixin: self.log_error("索引失败", error="...")
    Mixin->>Logger: logger.error(msg, extra={...})
    Logger->>Logger: 创建LogRecord(包含exc_info)
    Logger->>Handler: 传递LogRecord
    Handler->>Formatter: 格式化(包含堆栈跟踪)
    Formatter-->>Handler: 格式化字符串+堆栈
    Handler->>Stdout: 写入(多行输出)
```

**时序图功能介绍**:

1. **懒加载机制(步骤1-7)**:
   - **触发时机**:首次访问`self.logger`属性
   - **目的**:延迟logger初始化,减少启动开销
   - **实现**:检查`_logger`属性是否存在,不存在则创建
   - **缓存**:创建后存储在实例属性`_logger`中,后续访问直接返回
   - **线程安全**:logging.getLogger内部有锁保护,多线程安全

2. **日志记录流程(步骤8-16)**:
   - **级别判断**:Logger先判断消息级别是否满足当前设置(例如INFO>=INFO)
   - **LogRecord创建**:封装消息、时间戳、模块名、级别等信息
   - **Handler处理**:遍历所有Handler(此处为StreamHandler)
   - **格式化**:Formatter应用format模板生成最终字符串
   - **输出**:Handler将格式化后的日志写入目标(此处为stdout)
   - **异步写入**:Python logging默认同步,生产环境可配置QueueHandler实现异步

3. **错误日志处理(步骤17-24)**:
   - **异常捕获**:业务代码捕获异常后调用log_error
   - **额外信息**:通过extra参数传递结构化字段
   - **堆栈跟踪**:ERROR级别自动包含exc_info(异常信息)
   - **格式化**:Formatter将堆栈跟踪附加到消息后
   - **输出**:完整的错误信息+堆栈,便于问题排查

**性能优化点**:
- 级别过滤:不满足级别的日志直接丢弃,不进行格式化
- 懒加载:避免不必要的logger初始化
- 字符串懒拼接:使用%格式化而非f-string,仅在需要时才拼接

**并发安全**:
- logging模块内部使用锁保护,多线程/多协程安全
- Handler.emit有锁保护,避免多个线程同时写入

#### 2.1.4 上游调用示例与集成

**方式1:函数级使用**:
```python
# algo/graphrag-service/core/indexer.py

from shared import get_logger

logger = get_logger(__name__)  # logger名称: "core.indexer"

def index_document(doc_id: str, content: str):
    """索引文档"""
    # 记录开始
    logger.info("开始索引文档", extra={
        "doc_id": doc_id,
        "content_len": len(content),
        "timestamp": time.time()
    })
    
    try:
        # 业务逻辑...
        chunks = split_text(content)
        logger.info("文本分块完成", extra={"chunk_count": len(chunks)})
        
        # 索引到Neo4j
        graph_db.index(chunks)
        
        # 记录成功
        logger.info("文档索引成功", extra={"doc_id": doc_id})
        
    except Exception as e:
        # 记录错误(自动包含堆栈跟踪)
        logger.exception("文档索引失败", extra={
            "doc_id": doc_id,
            "error": str(e)
        })
        raise
```

**输出示例**:
```
2025-01-15 10:30:45 - core.indexer - INFO - 开始索引文档
2025-01-15 10:30:46 - core.indexer - INFO - 文本分块完成
2025-01-15 10:30:47 - core.indexer - INFO - 文档索引成功
```

**方式2:类级使用(Mixin)**:
```python
# algo/voice-service/core/asr_engine.py

from shared import LoggerMixin

class ASREngine(LoggerMixin):
    """ASR引擎(继承LoggerMixin)"""
    
    def transcribe(self, audio_data: bytes) -> str:
        """语音识别"""
        # 使用self.logger或self.log_info
        self.log_info("开始ASR识别", audio_len=len(audio_data))
        
        try:
            # 调用Whisper模型
            text = self.model.transcribe(audio_data)
            self.log_info("ASR识别成功", text=text, confidence=0.95)
            return text
            
        except Exception as e:
            # 记录错误
            self.log_error("ASR识别失败", error=str(e))
            raise
```

**集成FastAPI(自动注入trace_id)**:
```python
# algo/voice-service/app/main.py

from fastapi import FastAPI, Request
from shared import get_logger
import uuid

logger = get_logger(__name__)
app = FastAPI()

@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    """中间件:为每个请求生成trace_id"""
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    
    # 注入到日志context(需要配置ContextFilter)
    logger.info("收到请求", extra={
        "trace_id": trace_id,
        "method": request.method,
        "path": request.url.path
    })
    
    response = await call_next(request)
    
    # 在响应头中返回trace_id
    response.headers["X-Trace-ID"] = trace_id
    return response
```

---

### 2.2 配置管理

**核心实现**:
```python
# algo/shared/config.py

from pydantic_settings import BaseSettings
from pydantic import Field

class BaseServiceConfig(BaseSettings):
    """
    服务配置基类
    
    特点:
    - 基于Pydantic,自动验证类型
    - 支持环境变量(.env文件)
    - 提供默认值
    """
    
    # 服务基本信息
    service_name: str = Field(default="service", env="SERVICE_NAME")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # 环境
    environment: str = Field(default="development", env="ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Redis
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # LLM Router
    llm_router_url: str = Field(default="http://localhost:8005", env="LLM_ROUTER_URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
```

**使用示例**:
```python
from shared.config import BaseServiceConfig

class GraphRAGConfig(BaseServiceConfig):
    """GraphRAG服务配置"""
    
    neo4j_uri: str = Field(..., env="NEO4J_URI")  # 必填
    neo4j_user: str = Field(..., env="NEO4J_USER")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")

# 加载配置
config = GraphRAGConfig()
print(f"Neo4j URI: {config.neo4j_uri}")
print(f"Chunk Size: {config.chunk_size}")
```

---

### 2.3 错误处理

**核心实现**:
```python
# algo/shared/errors.py

from enum import Enum

class ErrorCode(str, Enum):
    """统一错误代码"""
    
    # 通用错误
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    
    # RAG相关
    RAG_INDEXING_FAILED = "RAG_INDEXING_FAILED"
    RAG_QUERY_FAILED = "RAG_QUERY_FAILED"
    
    # 语音相关
    VOICE_ASR_FAILED = "VOICE_ASR_FAILED"
    VOICE_TTS_FAILED = "VOICE_TTS_FAILED"

class VoiceHelperError(Exception):
    """
    VoiceHelper自定义异常
    
    统一异常格式,便于错误处理和日志记录
    """
    
    def __init__(self, code: ErrorCode, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self):
        """转换为字典格式(用于API响应)"""
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details
        }
```

**使用示例**:
```python
from shared import VoiceHelperError, ErrorCode

# 抛出异常
def process_document(doc_id):
    if not doc_id:
        raise VoiceHelperError(
            code=ErrorCode.INVALID_REQUEST,
            message="文档ID不能为空",
            details={"field": "doc_id"}
        )
    
    # 处理逻辑...

# 捕获异常
try:
    process_document(None)
except VoiceHelperError as e:
    logger.error(f"处理失败: {e.code} - {e.message}")
    return error_response(e.code, e.message)
```

---

### 2.4 统一响应格式

**核心实现**:
```python
# algo/shared/responses.py

def success_response(data: Any, message: str = "success") -> Dict:
    """
    成功响应格式
    
    {
        "code": 0,
        "message": "success",
        "data": {...}
    }
    """
    return {
        "code": 0,
        "message": message,
        "data": data
    }

def error_response(code: ErrorCode, message: str, details: dict = None) -> Dict:
    """
    错误响应格式
    
    {
        "code": "ERROR_CODE",
        "message": "错误描述",
        "details": {...}
    }
    """
    return {
        "code": code.value if isinstance(code, ErrorCode) else code,
        "message": message,
        "details": details or {}
    }
```

---

### 2.5 健康检查

**核心实现**:
```python
# algo/shared/health.py

class HealthChecker:
    """
    健康检查器
    
    检查项:
    - 数据库连接
    - Redis连接
    - 依赖服务可用性
    """
    
    async def check_all(self) -> HealthResult:
        """执行所有健康检查"""
        checks = {
            "database": await self._check_database(),
            "redis": await self._check_redis(),
            "dependencies": await self._check_dependencies()
        }
        
        # 判断整体状态
        status = "healthy" if all(checks.values()) else "unhealthy"
        
        return HealthResult(
            status=status,
            checks=checks,
            timestamp=datetime.now()
        )
    
    async def _check_database(self) -> bool:
        """检查数据库连接"""
        # 此处省略数据库连接检查逻辑
        return True
    
    async def _check_redis(self) -> bool:
        """检查Redis连接"""
        # 此处省略Redis连接检查逻辑
        return True
```

---

### 2.6 指标收集

**核心实现**:
```python
# algo/shared/metrics.py

from prometheus_client import Counter, Histogram, Gauge

class MetricsCollector:
    """
    Prometheus指标收集器
    
    指标类型:
    - Counter: 计数器(请求总数)
    - Histogram: 直方图(延迟分布)
    - Gauge: 仪表盘(当前值)
    """
    
    def __init__(self):
        # 请求计数
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        # 请求延迟
        self.request_latency = Histogram(
            'http_request_duration_seconds',
            'HTTP request latency',
            ['method', 'endpoint']
        )
        
        # 当前活跃连接
        self.active_connections = Gauge(
            'active_connections',
            'Current active connections'
        )
    
    def record_request(self, method: str, endpoint: str, status: int, latency: float):
        """记录HTTP请求"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(latency)
    
    def export_prometheus(self) -> str:
        """导出Prometheus格式指标"""
        from prometheus_client import generate_latest
        return generate_latest().decode('utf-8')
```

---

## 3. Go共享组件

### 3.1 中间件系统

#### 3.1.1 中间件架构与职责

**中间件链模式**:
```
Client请求 
  → Auth中间件(JWT验证) 
  → RBAC中间件(权限检查) 
  → 租户中间件(租户隔离) 
  → 限流中间件(流量控制)
  → 日志中间件(请求日志)
  → 追踪中间件(分布式追踪)
  → 业务Handler(实际处理)
```

**职责划分**:
- **Auth中间件**:验证JWT token,提取用户身份
- **RBAC中间件**:基于角色的权限验证
- **租户中间件**:多租户数据隔离
- **限流中间件**:防止滥用,保护服务
- **日志中间件**:记录请求/响应,用于审计
- **追踪中间件**:注入trace_id,支持链路追踪

#### 3.1.2 Auth中间件完整实现与时序图

**核心代码**:
```go
// backend/pkg/middleware/auth.go

package middleware

import (
    "strings"
    "time"
    "github.com/gin-gonic/gin"
    "github.com/golang-jwt/jwt/v5"
)

// AuthMiddleware JWT认证中间件
type AuthMiddleware struct {
    secretKey      []byte
    skipPaths      []string                 // 跳过认证的路径
    tokenBlacklist map[string]time.Time     // Token黑名单(简化实现)
}

// Claims JWT声明结构体
type Claims struct {
    UserID   string   `json:"user_id"`
    TenantID string   `json:"tenant_id"`
    Role     string   `json:"role"`
    Scopes   []string `json:"scopes"`
    jwt.RegisteredClaims
}

// Handle JWT验证中间件处理函数
func (a *AuthMiddleware) Handle() gin.HandlerFunc {
    return func(c *gin.Context) {
        // 1. 检查是否需要跳过验证(/health, /metrics等公开端点)
        if a.shouldSkip(c.Request.URL.Path) {
            c.Next()
            return
        }

        // 2. 提取Token(优先级:Header > Query > Cookie)
        tokenString := a.extractToken(c)
        if tokenString == "" {
            c.JSON(401, gin.H{"error": "未提供认证令牌"})
            c.Abort()
            return
        }

        // 3. 检查黑名单
        if a.isBlacklisted(tokenString) {
            c.JSON(401, gin.H{"error": "令牌已被撤销"})
            c.Abort()
            return
        }

        // 4. 验证Token
        claims, err := a.validateToken(tokenString)
        if err != nil {
            c.JSON(401, gin.H{"error": "无效的令牌: " + err.Error()})
            c.Abort()
            return
        }

        // 5. 检查Token是否过期
        if claims.ExpiresAt != nil && claims.ExpiresAt.Before(time.Now()) {
            c.JSON(401, gin.H{"error": "令牌已过期"})
            c.Abort()
            return
        }

        // 6. 将用户信息注入Context
        c.Set("user_id", claims.UserID)
        c.Set("tenant_id", claims.TenantID)
        c.Set("role", claims.Role)
        c.Set("scopes", claims.Scopes)
        c.Set("token", tokenString)

        // 7. Token自动续期(如果快过期)
        if a.shouldRenew(claims) {
            newToken, err := a.renewToken(claims)
            if err == nil {
                c.Header("X-New-Token", newToken)
            }
        }

        // 8. 继续处理下一个中间件/Handler
        c.Next()
    }
}

// extractToken 从请求中提取Token(支持多种方式)
func (a *AuthMiddleware) extractToken(c *gin.Context) string {
    // 方式1: Authorization Header (Bearer Token)
    if authHeader := c.GetHeader("Authorization"); authHeader != "" {
        parts := strings.SplitN(authHeader, " ", 2)
        if len(parts) == 2 && strings.ToLower(parts[0]) == "bearer" {
            return parts[1]
        }
    }

    // 方式2: Query参数(用于WebSocket连接)
    if token := c.Query("token"); token != "" {
        return token
    }

    // 方式3: Cookie(用于Web前端)
    if cookie, err := c.Cookie("access_token"); err == nil && cookie != "" {
        return cookie
    }

    return ""
}

// validateToken 验证JWT token并解析Claims
func (a *AuthMiddleware) validateToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        return a.secretKey, nil
    })

    if err != nil {
        return nil, err
    }

    if claims, ok := token.Claims.(*Claims); ok && token.Valid {
        return claims, nil
    }

    return nil, jwt.ErrSignatureInvalid
}

// shouldRenew 检查Token是否需要续期(10分钟内过期)
func (a *AuthMiddleware) shouldRenew(claims *Claims) bool {
    if claims.ExpiresAt == nil {
        return false
    }
    return time.Until(claims.ExpiresAt.Time) < 10*time.Minute
}

// renewToken 续期Token(延长24小时)
func (a *AuthMiddleware) renewToken(oldClaims *Claims) (string, error) {
    newClaims := &Claims{
        UserID:   oldClaims.UserID,
        TenantID: oldClaims.TenantID,
        Role:     oldClaims.Role,
        Scopes:   oldClaims.Scopes,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
        },
    }

    token := jwt.NewWithClaims(jwt.SigningMethodHS256, newClaims)
    return token.SignedString(a.secretKey)
}
```

**Auth中间件时序图**:
```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant AuthMW as Auth中间件
    participant Cache as 多级缓存
    participant Context as gin.Context
    participant NextHandler as 下一个中间件/Handler
    
    Note over Client,NextHandler: 认证流程
    Client->>AuthMW: 请求(Header: Authorization: Bearer xxx)
    AuthMW->>AuthMW: shouldSkip(path)?<br/>检查是否跳过认证
    
    alt 路径需要跳过(/health, /metrics)
        AuthMW->>NextHandler: c.Next() 直接放行
    else 路径需要认证
        AuthMW->>AuthMW: extractToken(c)<br/>从Header/Query/Cookie提取
        
        alt Token不存在
            AuthMW-->>Client: 401 Unauthorized<br/>"未提供认证令牌"
        else Token存在
            AuthMW->>Cache: 检查黑名单<br/>GET token:blacklist:{token_hash}
            Cache-->>AuthMW: 未在黑名单
            
            AuthMW->>AuthMW: validateToken(tokenString)<br/>解析JWT并验证签名
            
            alt Token无效或签名错误
                AuthMW-->>Client: 401 Unauthorized<br/>"无效的令牌"
            else Token有效
                AuthMW->>AuthMW: 检查过期时间<br/>claims.ExpiresAt vs time.Now()
                
                alt Token已过期
                    AuthMW-->>Client: 401 Unauthorized<br/>"令牌已过期"
                else Token未过期
                    AuthMW->>Context: c.Set("user_id", claims.UserID)<br/>注入用户信息
                    AuthMW->>Context: c.Set("tenant_id", claims.TenantID)
                    AuthMW->>Context: c.Set("role", claims.Role)
                    
                    AuthMW->>AuthMW: shouldRenew(claims)?<br/>检查是否需要续期(10min内过期)
                    
                    alt 需要续期
                        AuthMW->>AuthMW: renewToken(claims)<br/>生成新Token(延长24h)
                        AuthMW->>Client: 设置响应头<br/>X-New-Token: new_token
                    end
                    
                    AuthMW->>NextHandler: c.Next()<br/>继续处理请求
                    NextHandler-->>AuthMW: 处理完成
                    AuthMW-->>Client: 返回响应
                end
            end
        end
    end
```

**时序图功能介绍**:

1. **跳过认证检查(步骤2-3)**:
   - **目的**:公开端点(/health, /metrics, /docs)无需认证
   - **实现**:维护skipPaths列表,前缀匹配
   - **示例**:skipPaths = ["/health", "/metrics", "/api/v1/public"]

2. **Token提取(步骤4-5)**:
   - **优先级**:Header > Query > Cookie
   - **Header**:标准方式,适用于API调用
   - **Query**:用于WebSocket(无法设置Header)
   - **Cookie**:用于Web前端(自动携带)

3. **黑名单检查(步骤6-7)**:
   - **目的**:撤销已签发的token(用户登出、强制下线)
   - **实现**:Redis存储`token:blacklist:{token_hash}`,TTL=token剩余时间
   - **性能**:L1缓存命中率>90%,延迟<1ms

4. **Token验证(步骤8-9)**:
   - **签名验证**:HMAC-SHA256算法,密钥存储在环境变量
   - **Claims解析**:提取user_id、tenant_id、role、scopes
   - **异常**:签名错误、格式错误、Claims缺失

5. **过期检查(步骤10-11)**:
   - **ExpiresAt**:Unix时间戳,与当前时间比较
   - **容忍度**:无时钟偏移容忍,严格过期
   - **异常**:返回401,客户端需刷新token

6. **Context注入(步骤12-14)**:
   - **目的**:后续中间件/Handler可直接获取用户信息
   - **字段**:user_id、tenant_id、role、scopes、token
   - **使用**:`userID := c.GetString("user_id")`

7. **Token续期(步骤15-17)**:
   - **触发条件**:token在10分钟内过期
   - **续期方式**:生成新token(延长24小时),通过响应头返回
   - **客户端**:收到X-New-Token后更新本地token

8. **继续处理(步骤18-20)**:
   - **c.Next()**:执行中间件链中的下一个
   - **顺序**:Auth → RBAC → 租户 → 限流 → 业务Handler
   - **Abort()**:任何中间件可调用c.Abort()中断链

**性能优化**:
- Token验证:使用缓存避免重复解析(同一token短时间内多次请求)
- 黑名单查询:L1(内存)+L2(Redis)双级缓存
- Claims注入:直接设置Context,无序列化开销

**安全考量**:
- Token存储:客户端存储在httpOnly Cookie或localStorage
- 密钥管理:从环境变量读取,定期轮换
- 续期策略:仅在即将过期时续期,避免token永久有效

#### 3.1.3 限流中间件实现与时序图

**核心代码**:
```go
// backend/pkg/middleware/rate_limit.go

func RateLimitMiddleware(limiter *ratelimit.RateLimiter) gin.HandlerFunc {
    /**
     * 限流中间件
     * 
     * 算法:滑动窗口(基于Redis)
     * 维度:IP/User/APIKey
     * 策略:用户级100req/min, IP级1000req/min
     */
    return func(c *gin.Context) {
        // 1. 确定限流key(优先用户ID,其次IP)
        userID := c.GetString("user_id")
        var key string
        if userID != "" {
            key = "rate:user:" + userID + ":minute"
        } else {
            key = "rate:ip:" + c.ClientIP() + ":minute"
        }
        
        // 2. 检查限流(Redis INCR+EXPIRE)
        allowed, remaining := limiter.Allow(key)
        
        // 3. 设置响应头(告知客户端限流状态)
        c.Header("X-RateLimit-Limit", fmt.Sprintf("%d", limiter.Limit))
        c.Header("X-RateLimit-Remaining", fmt.Sprintf("%d", remaining))
        c.Header("X-RateLimit-Reset", fmt.Sprintf("%d", time.Now().Add(time.Minute).Unix()))
        
        if !allowed {
            // 4. 超限,返回429
            c.Header("Retry-After", "60")  // 60秒后重试
            c.JSON(429, gin.H{
                "error": "请求过于频繁",
                "retry_after": 60
            })
            c.Abort()
            return
        }
        
        c.Next()
    }
}
```

**限流中间件时序图**:
```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant RateLimitMW as 限流中间件
    participant Redis as Redis
    participant NextHandler as 下一个Handler
    
    Note over Client,NextHandler: 限流检查流程
    Client->>RateLimitMW: 请求
    RateLimitMW->>RateLimitMW: 提取user_id或IP<br/>生成key="rate:user:123:minute"
    
    RateLimitMW->>Redis: INCR rate:user:123:minute
    Redis-->>RateLimitMW: 返回当前计数=15
    
    RateLimitMW->>Redis: TTL rate:user:123:minute
    alt TTL=-1 (key无过期时间)
        RateLimitMW->>Redis: EXPIRE rate:user:123:minute 60
    end
    
    RateLimitMW->>RateLimitMW: 判断:15 < 100(限制)?
    
    alt 未超限
        RateLimitMW->>Client: 设置响应头<br/>X-RateLimit-Remaining: 85
        RateLimitMW->>NextHandler: c.Next()
        NextHandler-->>Client: 处理请求并返回
    else 超限
        RateLimitMW->>Client: 429 Too Many Requests<br/>Retry-After: 60
    end
```

**时序图说明**:
- **INCR操作**:原子递增,返回递增后的值
- **TTL检查**:首次INCR后设置60秒过期,实现滑动窗口
- **响应头**:X-RateLimit-*标准头,客户端可据此调整请求速率

---

### 3.2 数据库连接池

**PostgreSQL**:
```go
// backend/pkg/database/postgres.go

func NewPostgresDB(config *DatabaseConfig) (*gorm.DB, error) {
    /**
     * 创建PostgreSQL连接池
     * 
     * 配置:
     * - MaxOpenConns: 最大打开连接数
     * - MaxIdleConns: 最大空闲连接数
     * - ConnMaxLifetime: 连接最大生命周期
     */
    dsn := fmt.Sprintf(
        "host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
        config.Host, config.Port, config.User, config.Password, config.DBName,
    )
    
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Info),
    })
    
    if err != nil {
        return nil, err
    }
    
    // 配置连接池
    sqlDB, err := db.DB()
    if err != nil {
        return nil, err
    }
    
    sqlDB.SetMaxOpenConns(100)              // 最大打开连接
    sqlDB.SetMaxIdleConns(10)               // 最大空闲连接
    sqlDB.SetConnMaxLifetime(time.Hour)     // 连接最大生命周期
    
    return db, nil
}
```

**Redis**:
```go
// backend/pkg/database/redis.go

func NewRedisClient(config *RedisConfig) (*redis.Client, error) {
    /**
     * 创建Redis客户端
     * 
     * 配置:
     * - PoolSize: 连接池大小
     * - MinIdleConns: 最小空闲连接
     * - MaxRetries: 最大重试次数
     */
    client := redis.NewClient(&redis.Options{
        Addr:         config.Addr,
        Password:     config.Password,
        DB:           config.DB,
        PoolSize:     50,
        MinIdleConns: 10,
        MaxRetries:   3,
    })
    
    // 测试连接
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    if err := client.Ping(ctx).Err(); err != nil {
        return nil, err
    }
    
    return client, nil
}
```

---

### 3.3 多级缓存系统

#### 3.3.1 多级缓存架构设计

**缓存层级**:
```
┌──────────────────────────────────────────┐
│          业务服务(Handler)                │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│     L1缓存 (LRU内存缓存)                  │
│   - 容量: 1000个key                       │
│   - TTL: 30秒                            │
│   - 延迟: <1ms                           │
│   - 命中率: >80%                         │
└──────────────────────────────────────────┘
         ↓ (L1未命中)
┌──────────────────────────────────────────┐
│     L2缓存 (Redis)                        │
│   - 容量: 无限制(受内存限制)              │
│   - TTL: 5分钟                           │
│   - 延迟: <10ms                          │
│   - 命中率: >95%                         │
└──────────────────────────────────────────┘
         ↓ (L2未命中)
┌──────────────────────────────────────────┐
│     数据源 (数据库/外部API)               │
│   - 延迟: >100ms                         │
└──────────────────────────────────────────┘
```

**设计优势**:
- **分层策略**:热数据放L1,温数据放L2,冷数据回源
- **性能优化**:L1命中无网络开销,L2命中减轻数据源压力
- **容错性**:L2不可用时降级到L1+数据源
- **成本优化**:减少数据源查询,降低延迟和成本

#### 3.3.2 核心代码实现

```go
// backend/pkg/cache/multilevel_cache.go

package cache

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
    
    lru "github.com/hashicorp/golang-lru"
    "github.com/redis/go-redis/v9"
)

// MultiLevelCache 多级缓存
type MultiLevelCache struct {
    l1Cache     *lru.Cache         // L1: 内存LRU缓存
    l2Cache     *redis.Client      // L2: Redis缓存
    l1TTL       time.Duration      // L1缓存TTL
    l2TTL       time.Duration      // L2缓存TTL
    metrics     *CacheMetrics      // 指标收集
    enableStats bool               // 是否启用统计
}

// Config 缓存配置
type Config struct {
    L1Size      int           // L1缓存大小(个数)
    L1TTL       time.Duration // L1缓存TTL
    L2TTL       time.Duration // L2缓存TTL
    RedisClient *redis.Client // Redis客户端
    EnableStats bool          // 启用统计
}

// NewMultiLevelCache 创建多级缓存
func NewMultiLevelCache(config *Config) (*MultiLevelCache, error) {
    // 默认值
    if config.L1Size == 0 {
        config.L1Size = 1000
    }
    if config.L1TTL == 0 {
        config.L1TTL = 30 * time.Second
    }
    if config.L2TTL == 0 {
        config.L2TTL = 5 * time.Minute
    }

    // 创建L1缓存(LRU)
    l1Cache, err := lru.New(config.L1Size)
    if err != nil {
        return nil, fmt.Errorf("创建L1缓存失败: %w", err)
    }

    return &MultiLevelCache{
        l1Cache:     l1Cache,
        l2Cache:     config.RedisClient,
        l1TTL:       config.L1TTL,
        l2TTL:       config.L2TTL,
        metrics:     &CacheMetrics{},
        enableStats: config.EnableStats,
    }, nil
}

// Get 获取缓存值(核心方法)
func (m *MultiLevelCache) Get(ctx context.Context, key string) (interface{}, error) {
    // 1. 查询L1缓存
    if value, ok := m.l1Cache.Get(key); ok {
        // L1命中
        if m.enableStats {
            m.metrics.L1Hits++
        }
        logger.Debug("缓存L1命中", map[string]interface{}{"key": key})
        return value, nil
    }

    // 2. L1未命中,查询L2(Redis)
    value, err := m.l2Cache.Get(ctx, key).Result()
    if err == redis.Nil {
        // L2也未命中,缓存穿透
        if m.enableStats {
            m.metrics.Misses++
        }
        logger.Debug("缓存未命中", map[string]interface{}{"key": key})
        return nil, fmt.Errorf("缓存未命中")
    } else if err != nil {
        // Redis错误,降级
        logger.Error("Redis查询失败", map[string]interface{}{"error": err})
        return nil, err
    }

    // L2命中
    if m.enableStats {
        m.metrics.L2Hits++
    }
    logger.Debug("缓存L2命中", map[string]interface{}{"key": key})

    // 3. 回写L1缓存(Cache Aside模式)
    m.l1Cache.Add(key, value)

    return value, nil
}

// GetWithLoader 获取缓存值,未命中时使用loader加载
func (m *MultiLevelCache) GetWithLoader(
    ctx context.Context,
    key string,
    loader func() (interface{}, error),
) (interface{}, error) {
    // 1. 尝试从缓存获取
    value, err := m.Get(ctx, key)
    if err == nil {
        return value, nil
    }

    // 2. 缓存未命中,使用loader从数据源加载
    logger.Info("缓存未命中,从数据源加载", map[string]interface{}{"key": key})
    value, err = loader()
    if err != nil {
        return nil, fmt.Errorf("loader加载失败: %w", err)
    }

    // 3. 写入缓存
    if err := m.Set(ctx, key, value); err != nil {
        logger.Error("写入缓存失败", map[string]interface{}{"error": err, "key": key})
    }

    return value, nil
}

// Set 设置缓存值(核心方法)
func (m *MultiLevelCache) Set(ctx context.Context, key string, value interface{}) error {
    // 1. 写入L1缓存(同步)
    m.l1Cache.Add(key, value)

    // 2. 写入L2缓存(同步)
    var data []byte
    var err error

    // 序列化值
    switch v := value.(type) {
    case string:
        data = []byte(v)
    case []byte:
        data = v
    default:
        data, err = json.Marshal(value)
        if err != nil {
            return fmt.Errorf("序列化失败: %w", err)
        }
    }

    // 写入Redis
    if err := m.l2Cache.Set(ctx, key, data, m.l2TTL).Err(); err != nil {
        logger.Error("Redis写入失败", map[string]interface{}{"error": err})
        return fmt.Errorf("L2缓存写入失败: %w", err)
    }

    logger.Debug("缓存写入成功", map[string]interface{}{"key": key})
    return nil
}

// Delete 删除缓存
func (m *MultiLevelCache) Delete(ctx context.Context, key string) error {
    // 从L1删除
    m.l1Cache.Remove(key)

    // 从L2删除
    if err := m.l2Cache.Del(ctx, key).Err(); err != nil {
        return fmt.Errorf("L2删除失败: %w", err)
    }

    return nil
}

// GetMetrics 获取缓存统计指标
func (m *MultiLevelCache) GetMetrics() *CacheMetrics {
    if !m.enableStats {
        return nil
    }

    return &CacheMetrics{
        L1Hits: m.metrics.L1Hits,
        L2Hits: m.metrics.L2Hits,
        Misses: m.metrics.Misses,
        L1Size: m.l1Cache.Len(),
    }
}
```

#### 3.3.3 多级缓存时序图

```mermaid
sequenceDiagram
    autonumber
    participant Handler as 业务Handler
    participant Cache as MultiLevelCache
    participant L1 as L1缓存<br/>(LRU内存)
    participant L2 as L2缓存<br/>(Redis)
    participant Loader as 数据源<br/>(DB/API)
    participant Metrics as 指标收集
    
    Note over Handler,Metrics: 场景1: L1命中(热数据)
    Handler->>Cache: GetWithLoader(key="user:123")
    Cache->>L1: Get(key)
    L1-->>Cache: 返回value="用户数据"
    Cache->>Metrics: L1Hits++
    Cache-->>Handler: 返回value (耗时<1ms)
    
    Note over Handler,Metrics: 场景2: L1未命中,L2命中(温数据)
    Handler->>Cache: GetWithLoader(key="user:456")
    Cache->>L1: Get(key)
    L1-->>Cache: 未命中
    Cache->>L2: GET user:456
    L2-->>Cache: 返回value="用户数据"
    Cache->>Metrics: L2Hits++
    Cache->>L1: Add(key, value)<br/>回写L1
    Cache-->>Handler: 返回value (耗时<10ms)
    
    Note over Handler,Metrics: 场景3: L1、L2都未命中(冷数据)
    Handler->>Cache: GetWithLoader(key="user:789", loader)
    Cache->>L1: Get(key)
    L1-->>Cache: 未命中
    Cache->>L2: GET user:789
    L2-->>Cache: redis.Nil (未命中)
    Cache->>Metrics: Misses++
    
    Cache->>Loader: 调用loader()<br/>从数据源加载
    Loader->>Loader: 查询数据库<br/>SELECT * FROM users WHERE id=789
    Loader-->>Cache: 返回value="用户数据"
    
    Cache->>L1: Add(key, value)<br/>写入L1
    Cache->>L2: SET user:789 value EX 300<br/>写入L2(TTL=5min)
    
    Cache-->>Handler: 返回value (耗时>100ms)
    
    Note over Handler,Metrics: 场景4: 缓存失效与更新
    Handler->>Cache: Set(key="user:123", newValue)
    Cache->>L1: Add(key, newValue)<br/>更新L1
    Cache->>L2: SET user:123 newValue EX 300<br/>更新L2
    L2-->>Cache: OK
    Cache-->>Handler: 写入成功
    
    Note over Handler,Metrics: 场景5: Redis不可用,降级处理
    Handler->>Cache: Get(key="user:999")
    Cache->>L1: Get(key)
    L1-->>Cache: 未命中
    Cache->>L2: GET user:999
    L2--xCache: 连接超时/错误
    Cache->>Cache: 降级:返回错误<br/>或调用loader
    Cache-->>Handler: 返回错误或数据
```

**时序图功能介绍**:

1. **场景1 - L1命中(步骤1-5)**:
   - **触发条件**:key在L1缓存中存在(热数据)
   - **流程**:Handler → Cache → L1 → 直接返回
   - **性能**:延迟<1ms,无网络开销
   - **命中率**:正常情况下>80%
   - **适用**:高频访问数据(如当前用户session)

2. **场景2 - L2命中(步骤6-12)**:
   - **触发条件**:L1未命中但L2命中(温数据)
   - **流程**:L1未命中 → 查L2 → L2命中 → 回写L1 → 返回
   - **回写L1**:Cache Aside模式,将L2数据写入L1,加速下次访问
   - **性能**:延迟<10ms,一次Redis查询
   - **命中率**:L1+L2总命中率>95%
   - **适用**:中频访问数据(如用户配置)

3. **场景3 - 缓存穿透(步骤13-23)**:
   - **触发条件**:L1、L2都未命中(冷数据或新数据)
   - **流程**:L1未命中 → L2未命中 → 调用loader加载 → 写入L1和L2 → 返回
   - **loader**:业务提供的数据加载函数,通常查询数据库或调用API
   - **双写**:同时写入L1和L2,保证后续访问可从缓存读取
   - **性能**:延迟>100ms,取决于数据源响应时间
   - **优化**:使用空值缓存防止缓存穿透攻击

4. **场景4 - 缓存更新(步骤24-28)**:
   - **触发条件**:业务数据变更,需更新缓存
   - **流程**:同时更新L1和L2,保证一致性
   - **策略**:写入L1(同步) → 写入L2(同步)
   - **一致性**:L1/L2最终一致(可能有短暂不一致)
   - **失败处理**:L2写入失败时,L1仍保留旧值,等待TTL过期

5. **场景5 - 降级处理(步骤29-34)**:
   - **触发条件**:Redis不可用(网络故障、超时、宕机)
   - **降级策略**:
     - 策略1:返回错误,由业务决定如何处理
     - 策略2:直接调用loader,绕过缓存
     - 策略3:仅使用L1缓存,L2降级
   - **熔断**:连续失败后开启熔断,一段时间内不访问L2
   - **恢复**:熔断器半开状态尝试恢复,成功后关闭熔断

**性能优化点**:
- **LRU淘汰**:L1使用LRU算法,自动淘汰冷数据
- **批量预热**:启动时预热热数据到L1
- **TTL差异化**:L1短TTL(30s),L2长TTL(5min)
- **异步写入**:考虑异步写入L2,不阻塞主流程

**一致性保证**:
- **写入顺序**:先写L1再写L2,保证L1最新
- **删除顺序**:先删L2再删L1,避免脏数据
- **过期时间**:L1 TTL < L2 TTL,L1过期后从L2恢复

**监控指标**:
- **命中率**:L1命中率、L2命中率、总命中率
- **延迟**:P50/P95/P99延迟
- **错误率**:L2连接错误率、序列化错误率
- **容量**:L1当前大小、L2总键数

---

### 3.4 安全组件

**JWT工具**:
```go
// backend/pkg/security/jwt.go

func GenerateToken(userID string, expiresIn time.Duration) (string, error) {
    /**
     * 生成JWT token
     * 
     * Claims:
     * - user_id: 用户ID
     * - exp: 过期时间
     * - iat: 签发时间
     */
    claims := jwt.MapClaims{
        "user_id": userID,
        "exp":     time.Now().Add(expiresIn).Unix(),
        "iat":     time.Now().Unix(),
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    
    return token.SignedString([]byte(jwtSecret))
}

func ValidateToken(tokenString string) (jwt.MapClaims, error) {
    /**
     * 验证JWT token
     * 
     * 检查:
     * 1. 签名有效性
     * 2. 是否过期
     * 3. 是否在黑名单
     */
    token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
        return []byte(jwtSecret), nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if !token.Valid {
        return nil, errors.New("invalid token")
    }
    
    claims := token.Claims.(jwt.MapClaims)
    
    // 检查黑名单(此处省略)
    
    return claims, nil
}
```

**加密工具**:
```go
// backend/pkg/security/encryption.go

func HashPassword(password string) (string, error) {
    /**
     * 密码哈希
     * 
     * 使用bcrypt算法
     * Cost: 12 (2^12次迭代)
     */
    bytes, err := bcrypt.GenerateFromPassword([]byte(password), 12)
    return string(bytes), err
}

func CheckPassword(password, hash string) bool {
    /**
     * 密码验证
     * 
     * 比较明文密码与哈希值
     */
    err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
    return err == nil
}
```

---

## 4. 最佳实践

### 4.1 配置管理最佳实践

**分层配置**:
```python
# 1. 基础配置(BaseServiceConfig)
# 2. 服务特定配置(继承)
# 3. 环境变量覆盖

class MyServiceConfig(BaseServiceConfig):
    """我的服务配置"""
    
    # 必填配置
    database_url: str = Field(..., env="DATABASE_URL")
    
    # 可选配置(带默认值)
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # 验证逻辑
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith('postgresql://'):
            raise ValueError('数据库URL必须以postgresql://开头')
        return v
```

### 4.2 错误处理最佳实践

**统一异常捕获**:
```python
# FastAPI异常处理器
@app.exception_handler(VoiceHelperError)
async def voicehelper_exception_handler(request: Request, exc: VoiceHelperError):
    """处理VoiceHelper自定义异常"""
    logger.error(f"错误: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理通用异常"""
    logger.exception("未处理的异常", exc)
    return error_response(ErrorCode.INTERNAL_ERROR, "系统内部错误")
```

### 4.3 日志最佳实践

**结构化日志**:
```python
# 使用extra参数记录结构化信息
logger.info("用户登录", extra={
    "user_id": user_id,
    "ip": client_ip,
    "user_agent": user_agent,
    "timestamp": datetime.now().isoformat()
})

# 错误日志包含traceback
try:
    risky_operation()
except Exception as e:
    logger.exception("操作失败", extra={
        "operation": "risky_operation",
        "error": str(e)
    })
```

### 4.4 缓存最佳实践

**缓存失效策略**:
```go
// 1. TTL过期
cache.Set(ctx, key, value, 5*time.Minute)

// 2. 主动失效
cache.Delete(ctx, key)

// 3. 模式匹配失效
cache.DeleteByPattern(ctx, "user:*:profile")

// 4. Cache Aside模式
func GetUser(userID string) (*User, error) {
    // 1. 查缓存
    user, err := cache.Get(ctx, "user:"+userID)
    if err == nil {
        return user, nil
    }
    
    // 2. 查数据库
    user, err = db.FindUserByID(userID)
    if err != nil {
        return nil, err
    }
    
    // 3. 写缓存
    cache.Set(ctx, "user:"+userID, user, 10*time.Minute)
    
    return user, nil
}
```

---

## 5. 共享组件交互矩阵

### 5.1 跨服务调用关系

| 上游服务 | 共享组件 | 调用频率 | 关键场景 | 性能要求 | 降级策略 |
|---------|---------|---------|---------|---------|---------|
| GraphRAG Service | Logger | 高(每请求) | 记录索引/查询日志 | <1ms | 异步写入,失败不阻塞 |
| GraphRAG Service | Config | 启动时 | 加载Neo4j配置 | <10ms | 环境变量fallback |
| GraphRAG Service | Metrics | 高(每请求) | 上报查询延迟 | <5ms | 批量上报,失败丢弃 |
| Voice Service | Logger | 高(每请求) | 记录ASR/TTS日志 | <1ms | 同上 |
| Voice Service | HTTPClient | 中(调用LLM) | 调用LLM Router | <3s | 重试3次+熔断 |
| API Gateway | AuthMW | 高(每请求) | JWT验证 | <15ms | token缓存,Redis降级 |
| API Gateway | RateLimitMW | 高(每请求) | 限流检查 | <10ms | Redis降级到内存 |
| API Gateway | Cache | 高(每请求) | 缓存查询结果 | <10ms | L2降级到L1 |
| Auth Service | Database | 高(每请求) | 查询用户信息 | <50ms | 连接池+慢查询告警 |
| Auth Service | Security | 中(登录/注册) | 密码加密/JWT生成 | <20ms | 无降级(核心功能) |
| Session Service | Cache | 高(每请求) | session存储 | <5ms | 直接查数据库 |
| Notification Service | Queue | 中(消息发送) | 异步消息队列 | <100ms | 重试+死信队列 |

### 5.2 依赖关系图

```mermaid
graph TB
    subgraph "业务服务"
        GRAPHRAG[GraphRAG]
        VOICE[Voice]
        GATEWAY[Gateway]
        AUTH[Auth]
    end
    
    subgraph "Python共享组件"
        PYLOGGER[Logger]
        PYCONFIG[Config]
        PYMETRICS[Metrics]
        PYHTTP[HTTPClient]
    end
    
    subgraph "Go共享组件"
        AUTHMW[AuthMW]
        CACHE[Cache]
        DB[Database]
        SECURITY[Security]
    end
    
    subgraph "基础设施"
        REDIS[(Redis)]
        PG[(PostgreSQL)]
        PROM[Prometheus]
    end
    
    GRAPHRAG --> PYLOGGER
    GRAPHRAG --> PYCONFIG
    GRAPHRAG --> PYMETRICS
    
    VOICE --> PYLOGGER
    VOICE --> PYHTTP
    
    GATEWAY --> AUTHMW
    GATEWAY --> CACHE
    
    AUTH --> DB
    AUTH --> SECURITY
    
    PYMETRICS --> PROM
    AUTHMW --> CACHE
    CACHE --> REDIS
    DB --> PG
    
    style CACHE fill:#FFD700
    style AUTHMW fill:#FFD700
    style PYLOGGER fill:#87CEEB
```

### 5.3 关键路径分析

#### 路径1: 用户认证请求
```
Client → Gateway → AuthMW → Cache(L1) → Cache(L2/Redis) → NextHandler
```
- **关键组件**:AuthMW(JWT验证) + Cache(token黑名单查询)
- **性能目标**:P95 < 15ms
- **瓶颈**:Redis连接延迟
- **优化**:L1缓存命中率>90%

#### 路径2: 语音识别请求
```
Client → Gateway → Voice Service → Logger → Metrics → ASR Model → Response
```
- **关键组件**:Logger(请求日志) + Metrics(性能指标)
- **性能目标**:ASR延迟 < 1s
- **瓶颈**:模型推理时间
- **优化**:异步日志+批量指标上报

#### 路径3: 文档查询请求
```
Client → Gateway → GraphRAG Service → Cache(查询结果) → Neo4j/Vector DB → Response
```
- **关键组件**:Cache(查询结果缓存)
- **性能目标**:缓存命中 < 10ms,未命中 < 500ms
- **瓶颈**:图数据库查询
- **优化**:查询结果缓存TTL=5min

---

## 6. 最佳实践与演进建议

### 6.1 开发最佳实践

**日志规范**:
```python
# ✅ 推荐:结构化日志
logger.info("用户登录成功", extra={
    "user_id": "123",
    "ip": "1.2.3.4",
    "login_method": "password"
})

# ❌ 不推荐:字符串拼接
logger.info(f"用户{user_id}从{ip}登录")
```

**配置管理**:
```python
# ✅ 推荐:环境变量+默认值
class ServiceConfig(BaseServiceConfig):
    redis_url: str = Field(env="REDIS_URL")
    neo4j_uri: str = Field(..., env="NEO4J_URI")  # 必填

# ❌ 不推荐:硬编码
REDIS_URL = "redis://localhost:6379"
```

**缓存使用**:
```go
// ✅ 推荐:使用GetWithLoader
value, err := cache.GetWithLoader(ctx, key, func() (interface{}, error) {
    return db.Query(query)
})

// ❌ 不推荐:手动处理
value, err := cache.Get(ctx, key)
if err != nil {
    value, err = db.Query(query)
    cache.Set(ctx, key, value)
}
```

### 6.2 性能优化建议

1. **日志优化**:
   - 生产环境使用INFO级别,开发环境使用DEBUG
   - 配置QueueHandler实现异步日志写入
   - 敏感信息(密码、token)脱敏处理

2. **缓存优化**:
   - 根据访问频率动态调整L1大小
   - 热数据预热到L1缓存
   - 使用空值缓存防止缓存穿透

3. **数据库优化**:
   - 使用PreparedStatement缓存
   - 开启慢查询日志(>100ms)
   - 连接池大小根据并发量调整

4. **中间件优化**:
   - JWT验证结果缓存(同一token短时间内多次请求)
   - 限流使用滑动窗口而非固定窗口
   - RBAC权限缓存到内存

### 6.3 监控告警配置

**关键指标**:
```yaml
# Prometheus告警规则
groups:
  - name: shared_components
    rules:
      # L1缓存命中率
      - alert: L1CacheHitRateLow
        expr: rate(cache_l1_hits_total[5m]) / rate(cache_requests_total[5m]) < 0.8
        for: 5m
        annotations:
          summary: "L1缓存命中率低于80%"
      
      # Redis连接失败率
      - alert: RedisConnectionError
        expr: rate(redis_connection_errors_total[5m]) > 0.01
        for: 1m
        annotations:
          summary: "Redis连接错误率>1%"
      
      # JWT验证失败率
      - alert: AuthFailureRateHigh
        expr: rate(auth_failures_total[5m]) / rate(auth_requests_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "认证失败率>10%,可能遭受攻击"
      
      # 数据库连接池满
      - alert: DatabasePoolExhausted
        expr: db_pool_active_connections >= db_pool_max_connections
        for: 1m
        annotations:
          summary: "数据库连接池已满"
```

### 6.4 安全加固建议

1. **JWT安全**:
   - 密钥定期轮换(每90天)
   - token过期时间不超过24小时
   - 敏感操作需短期token(15分钟)
   - 实现token撤销机制(黑名单)

2. **加密安全**:
   - 使用AES-256-GCM而非ECB模式
   - 密钥存储在Vault/KMS,不存代码
   - 定期密钥轮换,保留旧密钥解密历史数据

3. **限流安全**:
   - 实现多维度限流(IP+User+APIKey)
   - 异常流量自动封禁
   - 添加验证码防止暴力破解

### 6.5 演进路线图

**短期(1-3个月)**:
- [x] 完成Python/Go共享组件基础功能
- [ ] 添加分布式追踪(Jaeger)集成
- [ ] 完善健康检查机制
- [ ] 补充单元测试(覆盖率>80%)

**中期(3-6个月)**:
- [ ] 引入服务网格(Istio)替代部分中间件
- [ ] 实现配置中心(Consul/Etcd)
- [ ] 优化缓存策略(智能预热)
- [ ] 添加灰度发布支持

**长期(6-12个月)**:
- [ ] 多语言SDK支持(Java/Rust)
- [ ] 自适应限流算法
- [ ] AI驱动的性能优化
- [ ] 混沌工程测试

---

## 7. 总结

### 7.1 核心价值

Shared共享组件是VoiceHelper架构的基石,提供了:

**统一性**:
- 统一的日志格式、错误码、响应格式
- 统一的配置管理、健康检查、指标收集
- 统一的认证、限流、缓存策略

**高性能**:
- 多级缓存(L1+L2)降低延迟,提升命中率
- 连接池复用,减少连接建立开销
- 异步日志/指标,不阻塞业务流程

**高可用**:
- 降级机制(Redis不可用时降级到L1)
- 熔断器保护(连续失败后开启熔断)
- 健康检查及时发现故障

**可观测**:
- 日志+指标+追踪三位一体
- trace_id贯穿全链路
- 多维度监控告警

### 7.2 技术亮点

1. **多级缓存设计**:
   - L1(LRU内存)+L2(Redis)双级缓存
   - Cache Aside模式实现
   - 总命中率>95%,P95延迟<10ms

2. **中间件链模式**:
   - Auth → RBAC → 租户 → 限流 → 日志 → 追踪
   - 灵活组合,按需启用
   - 支持跳过认证的公开端点

3. **懒加载与缓存**:
   - Logger懒加载,减少启动开销
   - JWT验证结果缓存
   - 预编译SQL语句缓存

4. **降级与容错**:
   - Redis不可用时降级到L1或数据源
   - 限流降级到内存计数器
   - 日志写入失败不阻塞业务

### 7.3 关键指标

| 指标类别 | 指标名称 | 目标值 | 当前值 | 备注 |
|---------|---------|-------|-------|------|
| 性能 | Auth中间件延迟(P95) | <15ms | 12ms | ✅ |
| 性能 | L1缓存命中率 | >80% | 85% | ✅ |
| 性能 | L1+L2缓存命中率 | >95% | 96% | ✅ |
| 性能 | 数据库连接池利用率 | <80% | 65% | ✅ |
| 可用性 | Redis可用性 | >99.9% | 99.95% | ✅ |
| 可用性 | PostgreSQL可用性 | >99.9% | 99.98% | ✅ |
| 安全 | JWT验证失败率 | <5% | 2% | ✅ |
| 安全 | 限流拦截率 | <1% | 0.5% | ✅ |

### 7.4 后续工作

1. **完善测试**:
   - 补充单元测试(目前覆盖率60%,目标80%)
   - 添加集成测试(中间件链、缓存穿透)
   - 压力测试(验证性能指标)

2. **文档补充**:
   - 补充API文档(OpenAPI规范)
   - 补充运维手册(部署/监控/故障处理)
   - 补充开发指南(如何接入共享组件)

3. **功能增强**:
   - 分布式追踪完整集成(Jaeger)
   - 配置中心集成(Consul)
   - 服务网格集成(Istio)

