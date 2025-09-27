---
title: "Dify平台完整架构指南：从概览到深度设计"
date: 2025-01-27T20:00:00+08:00
draft: false
featured: true
series: "dify-architecture"
tags: ["Dify", "架构设计", "系统设计", "技术架构", "平台架构"]
categories: ["dify", "架构设计"]
description: "Dify平台的完整架构指南，涵盖系统概览、深度设计、架构图谱和技术洞察"
toc: true
tocOpen: true
showReadingTime: true
showWordCount: true
weight: 10
slug: "dify-architecture-complete"
---

## 概述

本文档提供Dify平台的完整架构指南，从系统概览到深度设计，帮助读者全面理解Dify的架构理念、设计原则和技术实现。

<!--more-->

## 1. 平台架构概览

### 1.1 系统总体架构

Dify是一个开源的大模型应用开发平台，提供AI工作流、RAG管道、智能体功能与模型管理能力。平台采用现代化的分层架构设计，通过清晰的职责分离和模块化组织，实现了高度的可扩展性和可维护性。

```mermaid
graph TB
    subgraph "用户界面层"
        Web[Web界面]
        Mobile[移动端H5]
        API_Client[API客户端]
    end
    
    subgraph "网关与负载均衡"
        LB[负载均衡器]
        Gateway[API网关]
    end
    
    subgraph "应用服务层"
        subgraph "Console服务"
            Console_API[Console API]
            Console_Auth[认证授权]
            Console_Workspace[工作空间管理]
        end
        
        subgraph "Service API服务"
            Service_API[Service API]
            App_Runtime[应用运行时]
            Workflow_Engine[工作流引擎]
        end
        
        subgraph "Web API服务"
            Web_API[Web API]
            Chat_Service[对话服务]
            File_Service[文件服务]
        end
    end
    
    subgraph "核心业务层"
        subgraph "应用核心"
            App_Manager[应用管理器]
            Agent_Runner[智能体运行器]
            Chat_App[对话应用]
            Workflow_App[工作流应用]
            Agent_App[智能体应用]
        end
        
        subgraph "RAG引擎"
            Knowledge_Base[知识库管理]
            Vector_Store[向量存储]
            Retrieval_Engine[检索引擎]
            Document_Processor[文档处理器]
        end
        
        subgraph "工作流引擎"
            Flow_Executor[流程执行器]
            Node_Manager[节点管理器]
            Variable_Manager[变量管理器]
        end
        
        subgraph "模型运行时"
            Model_Manager[模型管理器]
            Provider_Manager[提供商管理]
            Token_Manager[令牌管理器]
        end
    end
    
    subgraph "服务支撑层"
        subgraph "数据服务"
            Account_Service[账户服务]
            Dataset_Service[数据集服务]
            Message_Service[消息服务]
            Workflow_Service[工作流服务]
        end
        
        subgraph "基础设施"
            Database[数据库]
            Cache[缓存]
            Queue[消息队列]
            Storage[对象存储]
        end
    end
    
    %% 连接关系
    Web --> LB
    Mobile --> LB
    API_Client --> LB
    
    LB --> Gateway
    Gateway --> Console_API
    Gateway --> Service_API
    Gateway --> Web_API
    
    Console_API --> App_Manager
    Service_API --> Agent_Runner
    Web_API --> Chat_Service
    
    App_Manager --> Knowledge_Base
    Agent_Runner --> Flow_Executor
    Chat_Service --> Model_Manager
    
    Knowledge_Base --> Account_Service
    Flow_Executor --> Dataset_Service
    Model_Manager --> Message_Service
    
    Account_Service --> Database
    Dataset_Service --> Cache
    Message_Service --> Queue
    Workflow_Service --> Storage
    
    style Web fill:#e3f2fd
    style Console_API fill:#e8f5e8
    style Service_API fill:#fff3e0
    style Web_API fill:#fce4ec
```

### 1.2 技术栈选择

#### 前端技术栈
- **Next.js 15**: React全栈框架，支持SSR/SSG
- **React 19**: 用户界面库，提供组件化开发
- **TypeScript**: 类型安全的JavaScript超集
- **Tailwind CSS**: 实用优先的CSS框架
- **Zustand**: 轻量级状态管理库

#### 后端技术栈
- **Python 3.11+**: 主要编程语言
- **Flask**: Web应用框架
- **SQLAlchemy**: ORM框架
- **Celery**: 分布式任务队列
- **Redis**: 缓存和消息代理
- **PostgreSQL**: 主数据库
- **Qdrant/Weaviate**: 向量数据库

#### 基础设施
- **Docker**: 容器化部署
- **Nginx**: 反向代理和负载均衡
- **S3兼容存储**: 文件存储
- **Prometheus**: 监控和告警
- **OpenTelemetry**: 分布式追踪

### 1.3 架构设计原则

#### 1.3.1 分层架构原则
```mermaid
graph TB
    subgraph "分层架构设计"
        UI[表示层<br/>UI Layer]
        API[接口层<br/>API Layer]
        Service[服务层<br/>Service Layer]
        Core[核心层<br/>Core Layer]
        Data[数据层<br/>Data Layer]
    end
    
    UI --> API
    API --> Service
    Service --> Core
    Core --> Data
    
    style UI fill:#e3f2fd
    style API fill:#e8f5e8
    style Service fill:#fff3e0
    style Core fill:#fce4ec
    style Data fill:#f3e5f5
```

**设计原则**：
- **单向依赖**：上层依赖下层，下层不依赖上层
- **接口隔离**：层间通过明确的接口通信
- **职责分离**：每层专注于特定职责
- **可替换性**：同层组件可独立替换

#### 1.3.2 模块化设计
```python
# 模块化设计示例
class ModuleInterface:
    """模块标准接口"""
    
    def initialize(self) -> bool:
        """模块初始化"""
        pass
    
    def health_check(self) -> bool:
        """健康检查"""
        pass
    
    def shutdown(self) -> bool:
        """优雅关闭"""
        pass
    
    def get_metrics(self) -> dict:
        """获取性能指标"""
        pass

class WorkflowModule(ModuleInterface):
    """工作流模块实现"""
    
    def __init__(self):
        self.engine = WorkflowEngine()
        self.node_manager = NodeManager()
        self.variable_pool = VariablePool()
    
    def initialize(self) -> bool:
        """初始化工作流模块"""
        try:
            self.engine.start()
            self.node_manager.load_nodes()
            return True
        except Exception as e:
            logger.error(f"Workflow module initialization failed: {e}")
            return False
```

## 2. 蜂巢架构设计理念

### 2.1 架构设计哲学

Dify采用了独特的**蜂巢架构（Beehive Architecture）**设计理念：

```mermaid
graph TB
    subgraph "蜂巢架构核心理念"
        subgraph "独立蜂房单元"
            AppModule[应用模块]
            RAGModule[RAG模块]
            WorkflowModule[工作流模块]
            AgentModule[Agent模块]
            ModelModule[模型运行时模块]
        end
        
        subgraph "统一接口层"
            API[API接口层]
            MessageBus[消息总线]
            EventBus[事件总线]
        end
        
        subgraph "共享基础设施"
            Database[数据存储层]
            Cache[缓存层]
            Queue[队列系统]
            Monitor[监控系统]
        end
    end
    
    AppModule -.-> API
    RAGModule -.-> MessageBus
    WorkflowModule -.-> EventBus
    AgentModule -.-> API
    ModelModule -.-> MessageBus
    
    API --> Database
    MessageBus --> Cache
    EventBus --> Queue
    
    style AppModule fill:#e3f2fd
    style RAGModule fill:#e8f5e8
    style WorkflowModule fill:#fff3e0
    style AgentModule fill:#fce4ec
    style ModelModule fill:#f3e5f5
```

**蜂巢架构的核心优势**：

1. **模块独立性**：每个功能模块如蜂巢中的独立单元
2. **热插拔能力**：模块可单独升级或替换而不影响整体系统
3. **水平扩展**：新功能模块可无缝集成
4. **故障隔离**：单个模块故障不会导致系统崩溃

### 2.2 微服务架构演进

```mermaid
graph LR
    subgraph "架构演进路径"
        Monolith[单体架构<br/>Monolithic]
        Modular[模块化架构<br/>Modular]
        Microservices[微服务架构<br/>Microservices]
        Beehive[蜂巢架构<br/>Beehive]
    end
    
    Monolith --> Modular
    Modular --> Microservices
    Microservices --> Beehive
    
    style Beehive fill:#e8f5e8
```

**演进特点**：
- **渐进式拆分**：从单体到模块化再到微服务
- **业务驱动**：按业务边界划分服务
- **技术独立**：每个服务可选择最适合的技术栈
- **数据隔离**：服务间数据独立，通过API通信

## 3. 核心架构组件

### 3.1 应用运行时架构

```mermaid
graph TB
    subgraph "应用运行时架构"
        subgraph "应用类型层"
            ChatApp[聊天应用<br/>Chat App]
            AgentApp[智能体应用<br/>Agent App]
            WorkflowApp[工作流应用<br/>Workflow App]
            CompletionApp[补全应用<br/>Completion App]
        end
        
        subgraph "运行时引擎"
            AppRunner[应用运行器<br/>App Runner]
            TaskPipeline[任务管道<br/>Task Pipeline]
            QueueManager[队列管理器<br/>Queue Manager]
        end
        
        subgraph "执行环境"
            ModelRuntime[模型运行时<br/>Model Runtime]
            ToolRuntime[工具运行时<br/>Tool Runtime]
            CodeRuntime[代码运行时<br/>Code Runtime]
        end
        
        subgraph "资源管理"
            MemoryManager[内存管理<br/>Memory Manager]
            TokenManager[令牌管理<br/>Token Manager]
            CacheManager[缓存管理<br/>Cache Manager]
        end
    end
    
    ChatApp --> AppRunner
    AgentApp --> AppRunner
    WorkflowApp --> AppRunner
    CompletionApp --> AppRunner
    
    AppRunner --> TaskPipeline
    TaskPipeline --> QueueManager
    
    QueueManager --> ModelRuntime
    QueueManager --> ToolRuntime
    QueueManager --> CodeRuntime
    
    ModelRuntime --> MemoryManager
    ToolRuntime --> TokenManager
    CodeRuntime --> CacheManager
    
    style AppRunner fill:#e3f2fd
    style TaskPipeline fill:#e8f5e8
    style QueueManager fill:#fff3e0
```

### 3.2 数据流架构

```mermaid
sequenceDiagram
    participant User as 用户
    participant API as API网关
    participant App as 应用层
    participant Core as 核心引擎
    participant Model as 模型运行时
    participant DB as 数据库
    
    Note over User,DB: 完整数据流时序
    
    User->>API: 发送请求
    API->>API: 认证授权
    API->>App: 转发请求
    
    App->>App: 参数验证
    App->>Core: 创建任务
    
    Core->>Core: 任务调度
    Core->>Model: 模型调用
    
    Model->>Model: 推理计算
    Model-->>Core: 返回结果
    
    Core->>DB: 保存结果
    Core-->>App: 返回响应
    
    App-->>API: 格式化响应
    API-->>User: 返回结果
```

### 3.3 事件驱动架构

```mermaid
graph TB
    subgraph "事件驱动架构"
        subgraph "事件生产者"
            UserAction[用户操作]
            SystemEvent[系统事件]
            ModelCallback[模型回调]
            WorkflowTrigger[工作流触发]
        end
        
        subgraph "事件总线"
            EventBus[事件总线<br/>Event Bus]
            MessageQueue[消息队列<br/>Message Queue]
            EventRouter[事件路由<br/>Event Router]
        end
        
        subgraph "事件消费者"
            NotificationService[通知服务]
            AuditService[审计服务]
            MetricsCollector[指标收集器]
            WorkflowEngine[工作流引擎]
        end
    end
    
    UserAction --> EventBus
    SystemEvent --> EventBus
    ModelCallback --> EventBus
    WorkflowTrigger --> EventBus
    
    EventBus --> MessageQueue
    MessageQueue --> EventRouter
    
    EventRouter --> NotificationService
    EventRouter --> AuditService
    EventRouter --> MetricsCollector
    EventRouter --> WorkflowEngine
    
    style EventBus fill:#e3f2fd
    style MessageQueue fill:#e8f5e8
    style EventRouter fill:#fff3e0
```

## 4. 安全架构设计

### 4.1 多层安全防护

```mermaid
graph TB
    subgraph "多层安全架构"
        subgraph "网络安全层"
            WAF[Web应用防火墙]
            DDoS[DDoS防护]
            RateLimit[限流控制]
        end
        
        subgraph "应用安全层"
            Authentication[身份认证]
            Authorization[权限控制]
            InputValidation[输入验证]
            OutputSanitization[输出净化]
        end
        
        subgraph "数据安全层"
            Encryption[数据加密]
            AccessControl[访问控制]
            DataMasking[数据脱敏]
            AuditLog[审计日志]
        end
        
        subgraph "基础设施安全"
            NetworkSecurity[网络安全]
            ContainerSecurity[容器安全]
            SecretManagement[密钥管理]
        end
    end
    
    WAF --> Authentication
    DDoS --> Authorization
    RateLimit --> InputValidation
    
    Authentication --> Encryption
    Authorization --> AccessControl
    InputValidation --> DataMasking
    OutputSanitization --> AuditLog
    
    Encryption --> NetworkSecurity
    AccessControl --> ContainerSecurity
    DataMasking --> SecretManagement
    
    style WAF fill:#ffcdd2
    style Authentication fill:#f8bbd9
    style Encryption fill:#e1bee7
    style NetworkSecurity fill:#d1c4e9
```

### 4.2 API安全机制

```python
# API安全实现示例
class APISecurityManager:
    """API安全管理器"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.auth_manager = AuthenticationManager()
        self.validator = InputValidator()
    
    def secure_endpoint(self, func):
        """安全端点装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 限流检查
            if not self.rate_limiter.allow_request():
                raise RateLimitExceeded()
            
            # 2. 身份认证
            user = self.auth_manager.authenticate_request()
            if not user:
                raise Unauthorized()
            
            # 3. 权限验证
            if not self.auth_manager.authorize_request(user, func.__name__):
                raise Forbidden()
            
            # 4. 输入验证
            validated_args = self.validator.validate_inputs(*args, **kwargs)
            
            # 5. 执行业务逻辑
            result = func(*validated_args, **kwargs)
            
            # 6. 输出净化
            sanitized_result = self.validator.sanitize_output(result)
            
            return sanitized_result
        
        return wrapper
```

## 5. 性能架构设计

### 5.1 缓存架构

```mermaid
graph TB
    subgraph "多级缓存架构"
        subgraph "客户端缓存"
            BrowserCache[浏览器缓存]
            AppCache[应用缓存]
        end
        
        subgraph "CDN缓存"
            EdgeCache[边缘缓存]
            StaticCache[静态资源缓存]
        end
        
        subgraph "应用缓存"
            L1Cache[L1缓存<br/>本地内存]
            L2Cache[L2缓存<br/>Redis]
            L3Cache[L3缓存<br/>数据库缓存]
        end
        
        subgraph "数据缓存"
            QueryCache[查询缓存]
            ModelCache[模型缓存]
            VectorCache[向量缓存]
        end
    end
    
    BrowserCache --> EdgeCache
    AppCache --> EdgeCache
    
    EdgeCache --> L1Cache
    StaticCache --> L1Cache
    
    L1Cache --> L2Cache
    L2Cache --> L3Cache
    
    L3Cache --> QueryCache
    L3Cache --> ModelCache
    L3Cache --> VectorCache
    
    style L1Cache fill:#e3f2fd
    style L2Cache fill:#e8f5e8
    style L3Cache fill:#fff3e0
```

### 5.2 负载均衡架构

```mermaid
graph TB
    subgraph "负载均衡架构"
        subgraph "入口层"
            DNS[DNS负载均衡]
            CDN[CDN分发]
        end
        
        subgraph "网关层"
            LB[负载均衡器]
            APIGateway[API网关]
        end
        
        subgraph "应用层"
            WebServer1[Web服务器1]
            WebServer2[Web服务器2]
            WebServer3[Web服务器3]
        end
        
        subgraph "服务层"
            Service1[服务实例1]
            Service2[服务实例2]
            Service3[服务实例3]
        end
        
        subgraph "数据层"
            MasterDB[主数据库]
            SlaveDB1[从数据库1]
            SlaveDB2[从数据库2]
        end
    end
    
    DNS --> CDN
    CDN --> LB
    LB --> APIGateway
    
    APIGateway --> WebServer1
    APIGateway --> WebServer2
    APIGateway --> WebServer3
    
    WebServer1 --> Service1
    WebServer2 --> Service2
    WebServer3 --> Service3
    
    Service1 --> MasterDB
    Service2 --> SlaveDB1
    Service3 --> SlaveDB2
    
    style LB fill:#e3f2fd
    style APIGateway fill:#e8f5e8
```

## 6. 部署架构

### 6.1 容器化部署架构

```mermaid
graph TB
    subgraph "Kubernetes集群"
        subgraph "Ingress层"
            Ingress[Ingress Controller]
            TLS[TLS终止]
        end
        
        subgraph "应用层"
            WebPod[Web Pod]
            APIPod[API Pod]
            WorkerPod[Worker Pod]
        end
        
        subgraph "服务层"
            WebService[Web Service]
            APIService[API Service]
            WorkerService[Worker Service]
        end
        
        subgraph "存储层"
            PVC[持久化存储]
            ConfigMap[配置映射]
            Secret[密钥管理]
        end
        
        subgraph "外部服务"
            Database[数据库]
            Redis[缓存]
            S3[对象存储]
        end
    end
    
    Ingress --> TLS
    TLS --> WebService
    TLS --> APIService
    
    WebService --> WebPod
    APIService --> APIPod
    WorkerService --> WorkerPod
    
    WebPod --> PVC
    APIPod --> ConfigMap
    WorkerPod --> Secret
    
    APIPod --> Database
    WorkerPod --> Redis
    WebPod --> S3
    
    style Ingress fill:#e3f2fd
    style WebPod fill:#e8f5e8
    style APIPod fill:#fff3e0
    style WorkerPod fill:#fce4ec
```

### 6.2 云原生架构

```yaml
# Kubernetes部署配置示例
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dify-api
  labels:
    app: dify-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dify-api
  template:
    metadata:
      labels:
        app: dify-api
    spec:
      containers:
      - name: api
        image: dify/api:latest
        ports:
        - containerPort: 5001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: dify-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: dify-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5001
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 7. 监控与可观测性架构

### 7.1 监控架构

```mermaid
graph TB
    subgraph "监控与可观测性架构"
        subgraph "数据收集层"
            Metrics[指标收集<br/>Prometheus]
            Logs[日志收集<br/>Fluentd]
            Traces[链路追踪<br/>Jaeger]
        end
        
        subgraph "数据存储层"
            MetricsDB[指标存储<br/>Prometheus]
            LogsDB[日志存储<br/>Elasticsearch]
            TracesDB[追踪存储<br/>Jaeger]
        end
        
        subgraph "数据分析层"
            AlertManager[告警管理<br/>AlertManager]
            Dashboard[仪表盘<br/>Grafana]
            LogAnalysis[日志分析<br/>Kibana]
        end
        
        subgraph "通知层"
            Email[邮件通知]
            Slack[Slack通知]
            Webhook[Webhook通知]
        end
    end
    
    Metrics --> MetricsDB
    Logs --> LogsDB
    Traces --> TracesDB
    
    MetricsDB --> AlertManager
    MetricsDB --> Dashboard
    LogsDB --> LogAnalysis
    
    AlertManager --> Email
    AlertManager --> Slack
    AlertManager --> Webhook
    
    style Metrics fill:#e3f2fd
    style Logs fill:#e8f5e8
    style Traces fill:#fff3e0
    style Dashboard fill:#fce4ec
```

### 7.2 健康检查机制

```python
# 健康检查实现示例
class HealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'model_service': self.check_model_service,
            'workflow_engine': self.check_workflow_engine
        }
    
    async def health_check(self) -> dict:
        """执行全面健康检查"""
        results = {}
        overall_status = 'healthy'
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                status = await check_func()
                response_time = time.time() - start_time
                
                results[name] = {
                    'status': 'healthy' if status else 'unhealthy',
                    'response_time': response_time,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                if not status:
                    overall_status = 'unhealthy'
                    
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def check_database(self) -> bool:
        """检查数据库连接"""
        try:
            # 执行简单查询
            result = await db.execute("SELECT 1")
            return result is not None
        except Exception:
            return False
    
    async def check_redis(self) -> bool:
        """检查Redis连接"""
        try:
            await redis_client.ping()
            return True
        except Exception:
            return False
```

## 8. 架构演进与扩展

### 8.1 架构演进路线图

```mermaid
timeline
    title Dify架构演进路线图
    
    section 当前版本 v1.0
        单体架构 : 基础功能
                : 核心模块
                : 基本API
    
    section 近期目标 v2.0
        模块化架构 : 模块拆分
                  : 接口标准化
                  : 插件系统
    
    section 中期目标 v3.0
        微服务架构 : 服务拆分
                  : 服务网格
                  : 分布式部署
    
    section 长期目标 v4.0
        云原生架构 : 容器化
                  : 自动扩缩容
                  : 多云部署
```

### 8.2 扩展性设计

```python
# 扩展性设计示例
class ExtensionManager:
    """扩展管理器"""
    
    def __init__(self):
        self.extensions = {}
        self.hooks = defaultdict(list)
    
    def register_extension(self, name: str, extension: Extension):
        """注册扩展"""
        self.extensions[name] = extension
        
        # 注册钩子
        for hook_name in extension.get_hooks():
            self.hooks[hook_name].append(extension)
    
    def execute_hook(self, hook_name: str, *args, **kwargs):
        """执行钩子"""
        results = []
        for extension in self.hooks.get(hook_name, []):
            try:
                result = extension.execute_hook(hook_name, *args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Extension {extension.name} hook {hook_name} failed: {e}")
        
        return results

class Extension:
    """扩展基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_hooks(self) -> List[str]:
        """获取支持的钩子列表"""
        return []
    
    def execute_hook(self, hook_name: str, *args, **kwargs):
        """执行钩子"""
        method_name = f"on_{hook_name}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(*args, **kwargs)
```

## 9. 总结

Dify平台的架构设计体现了现代化软件架构的最佳实践：

### 9.1 架构优势

1. **高可扩展性**：蜂巢架构支持模块独立扩展
2. **高可用性**：多层冗余和故障隔离机制
3. **高性能**：多级缓存和负载均衡优化
4. **高安全性**：多层安全防护体系
5. **易维护性**：清晰的分层和模块化设计

### 9.2 技术创新

1. **蜂巢架构**：独特的模块化设计理念
2. **事件驱动**：异步处理和解耦设计
3. **云原生**：容器化和微服务架构
4. **AI优化**：针对AI应用的专门优化

### 9.3 未来发展

1. **智能化运维**：AIOps和自动化运维
2. **边缘计算**：边缘部署和分布式推理
3. **多云架构**：跨云部署和灾备
4. **生态扩展**：更丰富的插件和扩展机制

通过这套完整的架构设计，Dify平台为AI应用开发提供了坚实的技术基础，支持从小规模原型到大规模生产环境的各种需求。

---

*最后更新时间：2025-01-27*  
*文档版本：v1.0*  
*维护者：Dify架构团队*
