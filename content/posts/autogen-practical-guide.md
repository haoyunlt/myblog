---
title: "AutoGen实战指南：从入门到生产部署"
date: 2025-05-05T16:00:00+08:00
draft: false
featured: true
series: "autogen-architecture"
tags: ["AutoGen", "实战指南", "最佳实践", "生产部署", "性能优化"]
categories: ["autogen", "实战指南"]
author: "Architecture Analysis"
description: "AutoGen框架的完整实战指南，包含开发实践、性能优化、部署策略和故障排查"
image: "/images/articles/autogen-practical-guide.svg"
toc: true
tocOpen: true
showReadingTime: true
showWordCount: true

weight: 160
slug: "autogen-practical-guide"
---

## 概述

本指南基于大量生产环境实践经验，提供AutoGen框架从开发到部署的完整实战指导，包含性能优化、故障排查、监控告警等关键实践。

## 1. 快速开始指南

### 1.1 环境准备

#### Python环境设置

```bash
# 创建虚拟环境
python -m venv autogen-env
source autogen-env/bin/activate  # Linux/Mac
# autogen-env\Scripts\activate  # Windows

# 安装核心包
pip install autogen-core autogen-agentchat

# 安装扩展包
pip install autogen-ext[openai,azure,anthropic]

# 开发工具
pip install pytest pytest-asyncio black isort mypy
```

#### .NET环境设置

```bash
# 安装.NET SDK 8.0+
dotnet --version

# 创建新项目
dotnet new console -n AutoGenApp
cd AutoGenApp

# 添加AutoGen包
dotnet add package Microsoft.AutoGen.Core
dotnet add package Microsoft.AutoGen.Agents
dotnet add package Microsoft.Extensions.Hosting
```

### 1.2 第一个代理应用

#### Python版本

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    """第一个AutoGen应用"""
    
    # 1. 创建模型客户端
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key="your-openai-api-key"
    )
    
    # 2. 创建助手代理
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        description="一个有用的AI助手"
    )
    
    # 3. 运行对话
    await Console(assistant.run_stream(
        task="请介绍一下AutoGen框架的主要特点"
    ))

if __name__ == "__main__":
    asyncio.run(main())
```

#### .NET版本

```csharp
using Microsoft.AutoGen.Core;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

// 1. 定义代理
public class GreetingAgent : Agent
{
    public GreetingAgent(ILogger<GreetingAgent> logger) : base(logger)
    {
    }
    
    [MessageHandler]
    public async Task<string> HandleGreeting(string message, MessageContext context)
    {
        return $"你好！你说：{message}";
    }
}

// 2. 配置服务
var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddAutoGenCore();
builder.Services.AddAgent<GreetingAgent>();

var host = builder.Build();

// 3. 运行应用
var runtime = host.Services.GetRequiredService<IAgentRuntime>();
var agentId = new AgentId("GreetingAgent", "default");

var response = await runtime.SendMessageAsync<string>(
    "Hello AutoGen!", 
    agentId
);

Console.WriteLine(response);
```

## 2. 核心概念实战

### 2.1 代理设计模式

#### 单一职责代理

```python
class WeatherAgent(AssistantAgent):
    """专门处理天气查询的代理"""
    
    def __init__(self, weather_api_key: str):
        super().__init__(
            name="weather_agent",
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
            tools=[self.get_weather],
            description="专业的天气查询助手",
            system_message="你是一个专业的天气查询助手，只回答天气相关的问题。"
        )
        self.weather_api_key = weather_api_key
    
    async def get_weather(self, city: str, date: str = None) -> str:
        """
        获取指定城市的天气信息
        
        Args:
            city: 城市名称
            date: 日期（可选，默认今天）
            
        Returns:
            天气信息字符串
        """
        # 实际实现中调用天气API
        return f"{city}今天天气晴朗，温度25°C"

# 使用示例
weather_agent = WeatherAgent("your-weather-api-key")
result = await weather_agent.run(task="北京今天天气怎么样？")
```

#### 组合代理模式

```python
class TravelPlannerAgent(AssistantAgent):
    """旅行规划代理 - 组合多个专业代理"""
    
    def __init__(self):
        super().__init__(
            name="travel_planner",
            model_client=OpenAIChatCompletionClient(model="gpt-4o"),
            tools=[
                AgentTool(WeatherAgent("weather-key")),
                AgentTool(HotelAgent("hotel-key")),
                AgentTool(FlightAgent("flight-key"))
            ],
            description="专业的旅行规划助手",
            system_message="""
            你是一个专业的旅行规划助手。你可以：
            1. 查询天气信息
            2. 搜索酒店
            3. 查找航班
            4. 制定详细的旅行计划
            
            请根据用户需求，合理使用各种工具来制定最佳的旅行方案。
            """
        )

# 使用示例
planner = TravelPlannerAgent()
result = await planner.run(
    task="我想下周去上海旅行3天，请帮我制定一个详细的计划"
)
```

### 2.2 团队协作模式

#### 专家团队模式

```python
async def create_expert_team():
    """创建专家团队进行协作"""
    
    # 1. 创建各领域专家
    researcher = AssistantAgent(
        name="researcher",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        description="专业的研究员，负责信息收集和分析",
        system_message="你是一个专业的研究员，擅长收集和分析各种信息。"
    )
    
    writer = AssistantAgent(
        name="writer",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        description="专业的写作者，负责内容创作",
        system_message="你是一个专业的写作者，擅长创作各种类型的文档。"
    )
    
    reviewer = AssistantAgent(
        name="reviewer",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        description="专业的审核者，负责质量控制",
        system_message="你是一个专业的审核者，负责检查内容质量并提出改进建议。请在完成后说'TERMINATE'。"
    )
    
    # 2. 创建团队
    team = RoundRobinGroupChat(
        name="expert_team",
        description="专家协作团队",
        participants=[researcher, writer, reviewer],
        termination_condition=MaxMessageTermination(15)
    )
    
    return team

# 使用示例
team = await create_expert_team()
result = await team.run(
    task="写一篇关于人工智能在医疗领域应用的技术报告，要求内容准确、结构清晰、语言专业。"
)
```

#### 层次化决策模式

```python
class HierarchicalTeam:
    """层次化决策团队"""
    
    def __init__(self):
        # 操作层代理
        self.operational_agents = [
            self.create_data_analyst(),
            self.create_market_researcher(),
            self.create_technical_expert()
        ]
        
        # 管理层代理
        self.manager = self.create_manager()
        
        # 决策层代理
        self.executive = self.create_executive()
    
    def create_data_analyst(self):
        return AssistantAgent(
            name="data_analyst",
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
            description="数据分析专家",
            system_message="你是数据分析专家，负责分析数据并提供洞察。"
        )
    
    def create_manager(self):
        return AssistantAgent(
            name="manager",
            model_client=OpenAIChatCompletionClient(model="gpt-4o"),
            tools=[AgentTool(agent) for agent in self.operational_agents],
            description="项目经理，协调各专家工作",
            system_message="""
            你是项目经理，负责：
            1. 分析任务需求
            2. 分配工作给合适的专家
            3. 汇总专家意见
            4. 提出初步建议
            """
        )
    
    def create_executive(self):
        return AssistantAgent(
            name="executive",
            model_client=OpenAIChatCompletionClient(model="gpt-4o"),
            tools=[AgentTool(self.manager)],
            description="高级决策者",
            system_message="""
            你是高级决策者，负责：
            1. 审查管理层建议
            2. 考虑战略因素
            3. 做出最终决策
            4. 说明决策理由
            """
        )
    
    async def make_decision(self, task: str):
        """执行层次化决策流程"""
        return await self.executive.run(task=task)

# 使用示例
team = HierarchicalTeam()
decision = await team.make_decision(
    "我们公司是否应该投资开发一个新的AI产品？请进行全面分析并给出建议。"
)
```

## 3. 性能优化实战

### 3.1 消息处理优化

#### 批处理优化

```python
class BatchProcessor:
    """批处理消息优化器"""
    
    def __init__(self, batch_size: int = 10, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.message_queue = asyncio.Queue()
        self.result_futures = {}
        self.processing_task = None
    
    async def process_message(self, message: Any, agent: AssistantAgent) -> Any:
        """批处理消息"""
        
        # 创建结果Future
        message_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.result_futures[message_id] = future
        
        # 添加到队列
        await self.message_queue.put((message_id, message, agent))
        
        # 启动处理任务（如果未启动）
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_batch())
        
        # 等待结果
        return await future
    
    async def _process_batch(self):
        """批处理逻辑"""
        batch = []
        
        try:
            # 收集批次消息
            while len(batch) < self.batch_size:
                try:
                    item = await asyncio.wait_for(
                        self.message_queue.get(), 
                        timeout=self.timeout
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            if not batch:
                return
            
            # 按代理分组
            agent_groups = {}
            for message_id, message, agent in batch:
                if agent not in agent_groups:
                    agent_groups[agent] = []
                agent_groups[agent].append((message_id, message))
            
            # 并发处理各组
            tasks = [
                self._process_agent_group(agent, messages)
                for agent, messages in agent_groups.items()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            # 处理异常，通知所有等待的Future
            for message_id, _, _ in batch:
                if message_id in self.result_futures:
                    self.result_futures[message_id].set_exception(e)
                    del self.result_futures[message_id]
    
    async def _process_agent_group(self, agent: AssistantAgent, messages: List[Tuple[str, Any]]):
        """处理单个代理的消息组"""
        
        for message_id, message in messages:
            try:
                # 处理单个消息
                result = await agent.run(task=str(message))
                
                # 设置结果
                if message_id in self.result_futures:
                    self.result_futures[message_id].set_result(result)
                    del self.result_futures[message_id]
                    
            except Exception as e:
                if message_id in self.result_futures:
                    self.result_futures[message_id].set_exception(e)
                    del self.result_futures[message_id]

# 使用示例
batch_processor = BatchProcessor(batch_size=5, timeout=0.5)
agent = AssistantAgent("assistant", model_client)

# 批量处理消息
tasks = [
    batch_processor.process_message(f"问题{i}", agent)
    for i in range(20)
]

results = await asyncio.gather(*tasks)
```

#### 连接池优化

```python
class ModelClientPool:
    """模型客户端连接池"""
    
    def __init__(self, max_connections: int = 10, model_config: dict = None):
        self.max_connections = max_connections
        self.model_config = model_config or {}
        self.available_clients = asyncio.Queue(maxsize=max_connections)
        self.total_clients = 0
        self.lock = asyncio.Lock()
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def get_client(self) -> OpenAIChatCompletionClient:
        """获取客户端连接"""
        
        try:
            # 尝试从池中获取
            client = self.available_clients.get_nowait()
            self.stats['cache_hits'] += 1
            return client
        except asyncio.QueueEmpty:
            # 创建新连接
            async with self.lock:
                if self.total_clients < self.max_connections:
                    client = self._create_client()
                    self.total_clients += 1
                    self.stats['cache_misses'] += 1
                    return client
                else:
                    # 等待可用连接
                    client = await self.available_clients.get()
                    self.stats['cache_hits'] += 1
                    return client
    
    async def return_client(self, client: OpenAIChatCompletionClient):
        """归还客户端连接"""
        try:
            self.available_clients.put_nowait(client)
        except asyncio.QueueFull:
            # 池已满，关闭连接
            await self._close_client(client)
            async with self.lock:
                self.total_clients -= 1
    
    def _create_client(self) -> OpenAIChatCompletionClient:
        """创建新的客户端"""
        return OpenAIChatCompletionClient(
            model=self.model_config.get('model', 'gpt-4o-mini'),
            api_key=self.model_config.get('api_key'),
            base_url=self.model_config.get('base_url'),
            max_retries=self.model_config.get('max_retries', 3),
            timeout=self.model_config.get('timeout', 30.0)
        )
    
    async def _close_client(self, client: OpenAIChatCompletionClient):
        """关闭客户端连接"""
        # 实现客户端清理逻辑
        pass
    
    def get_stats(self) -> dict:
        """获取连接池统计信息"""
        return {
            **self.stats,
            'total_clients': self.total_clients,
            'available_clients': self.available_clients.qsize(),
            'hit_rate': self.stats['cache_hits'] / max(1, self.stats['requests'])
        }

# 使用示例
client_pool = ModelClientPool(max_connections=5, model_config={
    'model': 'gpt-4o-mini',
    'api_key': 'your-api-key'
})

class PooledAgent(AssistantAgent):
    """使用连接池的代理"""
    
    def __init__(self, name: str, client_pool: ModelClientPool):
        self.client_pool = client_pool
        super().__init__(
            name=name,
            model_client=None,  # 将在运行时获取
            description="使用连接池的高性能代理"
        )
    
    async def on_messages(self, messages, cancellation_token):
        """重写消息处理以使用连接池"""
        
        # 获取客户端
        client = await self.client_pool.get_client()
        
        try:
            # 临时设置客户端
            original_client = self._model_client
            self._model_client = client
            
            # 处理消息
            result = await super().on_messages(messages, cancellation_token)
            
            return result
        finally:
            # 归还客户端
            await self.client_pool.return_client(client)
            self._model_client = original_client
```

### 3.2 内存优化

#### 对象池化

```python
class MessageContextPool:
    """消息上下文对象池"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.pool = []
        self.lock = threading.Lock()
        self.created_count = 0
        self.reused_count = 0
    
    def get_context(self) -> MessageContext:
        """获取消息上下文对象"""
        
        with self.lock:
            if self.pool:
                context = self.pool.pop()
                self.reused_count += 1
                return context
            else:
                context = MessageContext()
                self.created_count += 1
                return context
    
    def return_context(self, context: MessageContext):
        """归还消息上下文对象"""
        
        # 重置对象状态
        context.reset()
        
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(context)
    
    def get_stats(self) -> dict:
        """获取池统计信息"""
        return {
            'created_count': self.created_count,
            'reused_count': self.reused_count,
            'pool_size': len(self.pool),
            'reuse_rate': self.reused_count / max(1, self.created_count + self.reused_count)
        }

class MessageContext:
    """可重用的消息上下文"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置对象状态"""
        self.message_id = None
        self.sender = None
        self.recipient = None
        self.timestamp = None
        self.properties = {}
        self.cancellation_token = None
```

#### 内存监控

```python
class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.callbacks = {
            'warning': [],
            'critical': [],
            'normal': []
        }
    
    def add_callback(self, level: str, callback: Callable):
        """添加内存状态回调"""
        if level in self.callbacks:
            self.callbacks[level].append(callback)
    
    def check_memory(self) -> dict:
        """检查内存使用情况"""
        
        # 获取系统内存信息
        memory_info = psutil.virtual_memory()
        usage_percent = memory_info.percent / 100.0
        
        # 获取进程内存信息
        process = psutil.Process()
        process_memory = process.memory_info()
        
        status = {
            'system_usage_percent': usage_percent,
            'process_memory_mb': process_memory.rss / 1024 / 1024,
            'available_memory_mb': memory_info.available / 1024 / 1024,
            'level': 'normal'
        }
        
        # 确定内存状态级别
        if usage_percent >= self.critical_threshold:
            status['level'] = 'critical'
        elif usage_percent >= self.warning_threshold:
            status['level'] = 'warning'
        
        # 触发回调
        for callback in self.callbacks[status['level']]:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"内存监控回调失败: {e}")
        
        return status
    
    async def start_monitoring(self, interval: float = 30.0):
        """启动内存监控"""
        
        while True:
            try:
                self.check_memory()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"内存监控异常: {e}")
                await asyncio.sleep(interval)

# 使用示例
memory_monitor = MemoryMonitor()

def on_memory_warning(status):
    logger.warning(f"内存使用率过高: {status['system_usage_percent']:.1%}")
    # 触发垃圾回收
    gc.collect()

def on_memory_critical(status):
    logger.critical(f"内存使用率危险: {status['system_usage_percent']:.1%}")
    # 清理缓存
    clear_caches()
    # 限制新请求
    enable_backpressure()

memory_monitor.add_callback('warning', on_memory_warning)
memory_monitor.add_callback('critical', on_memory_critical)

# 启动监控
asyncio.create_task(memory_monitor.start_monitoring())
```

## 4. 生产部署实战

### 4.1 容器化部署

#### Dockerfile最佳实践

```dockerfile
# 多阶段构建 - Python版本
FROM python:3.11-slim as builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --user -r requirements.txt

# 生产镜像
FROM python:3.11-slim

# 创建非root用户
RUN groupadd -r autogen && useradd -r -g autogen autogen

# 设置工作目录
WORKDIR /app

# 从builder阶段复制依赖
COPY --from=builder /root/.local /home/autogen/.local

# 复制应用代码
COPY --chown=autogen:autogen . .

# 设置环境变量
ENV PATH=/home/autogen/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 切换到非root用户
USER autogen

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose配置

```yaml
version: '3.8'

services:
  # AutoGen应用
  autogen-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/autogen
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=autogen
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Prometheus监控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  # Grafana可视化
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

### 4.2 Kubernetes部署

#### 部署配置

```yaml
# autogen-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autogen-app
  labels:
    app: autogen-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autogen-app
  template:
    metadata:
      labels:
        app: autogen-app
    spec:
      containers:
      - name: autogen-app
        image: autogen-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: autogen-secrets
              key: openai-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: autogen-secrets
              key: database-url
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
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: autogen-service
spec:
  selector:
    app: autogen-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: autogen-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autogen-app
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

#### 配置管理

```yaml
# autogen-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: autogen-config
data:
  app.yaml: |
    server:
      host: "0.0.0.0"
      port: 8000
      workers: 4
    
    logging:
      level: "INFO"
      format: "json"
      
    cache:
      type: "redis"
      ttl: 3600
      
    monitoring:
      enabled: true
      metrics_port: 9090
      
    agents:
      max_concurrent: 100
      timeout: 30
      retry_attempts: 3

---
apiVersion: v1
kind: Secret
metadata:
  name: autogen-secrets
type: Opaque
stringData:
  openai-api-key: "your-openai-api-key"
  database-url: "postgresql://user:pass@postgres:5432/autogen"
  redis-password: "your-redis-password"
```

### 4.3 监控和告警

#### Prometheus配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "autogen_rules.yml"

scrape_configs:
  - job_name: 'autogen-app'
    static_configs:
      - targets: ['autogen-service:9090']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### 告警规则

```yaml
# autogen_rules.yml
groups:
- name: autogen_alerts
  rules:
  # 高错误率告警
  - alert: HighErrorRate
    expr: rate(autogen_requests_failed_total[5m]) / rate(autogen_requests_total[5m]) > 0.05
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "AutoGen应用错误率过高"
      description: "错误率 {{ $value | humanizePercentage }} 超过5%"

  # 高响应时间告警
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(autogen_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "AutoGen应用响应时间过长"
      description: "95%分位响应时间 {{ $value }}s 超过2秒"

  # 内存使用率告警
  - alert: HighMemoryUsage
    expr: (process_resident_memory_bytes / node_memory_MemTotal_bytes) > 0.8
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "AutoGen应用内存使用率过高"
      description: "内存使用率 {{ $value | humanizePercentage }} 超过80%"

  # 代理离线告警
  - alert: AgentOffline
    expr: autogen_agents_count{status="offline"} > 0
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "有代理离线"
      description: "{{ $value }}个代理处于离线状态"
```

#### Grafana仪表板

```json
{
  "dashboard": {
    "title": "AutoGen监控仪表板",
    "panels": [
      {
        "title": "请求QPS",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(autogen_requests_total[1m])",
            "legendFormat": "{{agent_type}}"
          }
        ]
      },
      {
        "title": "响应时间分布",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(autogen_request_duration_seconds_bucket[1m])",
            "format": "heatmap"
          }
        ]
      },
      {
        "title": "错误率",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(autogen_requests_failed_total[5m]) / rate(autogen_requests_total[5m])",
            "legendFormat": "错误率"
          }
        ]
      },
      {
        "title": "活跃代理数",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(autogen_agents_count{status=\"online\"})",
            "legendFormat": "在线代理"
          }
        ]
      }
    ]
  }
}
```

## 5. 故障排查指南

### 5.1 常见问题诊断

#### 性能问题排查

```python
class PerformanceDiagnostics:
    """性能诊断工具"""
    
    def __init__(self):
        self.metrics = {}
        self.profiler = None
    
    async def diagnose_slow_response(self, agent: AssistantAgent, test_message: str):
        """诊断响应缓慢问题"""
        
        print("🔍 开始性能诊断...")
        
        # 1. 基础性能测试
        start_time = time.time()
        result = await agent.run(task=test_message)
        total_time = time.time() - start_time
        
        print(f"📊 总响应时间: {total_time:.2f}秒")
        
        # 2. 分阶段计时
        timings = await self._detailed_timing_analysis(agent, test_message)
        
        print("⏱️ 详细计时分析:")
        for stage, duration in timings.items():
            percentage = (duration / total_time) * 100
            print(f"  {stage}: {duration:.2f}s ({percentage:.1f}%)")
        
        # 3. 资源使用分析
        resource_usage = self._analyze_resource_usage()
        print(f"💾 内存使用: {resource_usage['memory_mb']:.1f}MB")
        print(f"🔥 CPU使用: {resource_usage['cpu_percent']:.1f}%")
        
        # 4. 生成优化建议
        suggestions = self._generate_optimization_suggestions(timings, resource_usage)
        
        print("\n💡 优化建议:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")
        
        return {
            'total_time': total_time,
            'timings': timings,
            'resource_usage': resource_usage,
            'suggestions': suggestions
        }
    
    async def _detailed_timing_analysis(self, agent: AssistantAgent, message: str):
        """详细的计时分析"""
        
        timings = {}
        
        # 模拟各阶段计时
        start = time.time()
        
        # 消息预处理
        await asyncio.sleep(0.01)  # 模拟预处理时间
        timings['message_preprocessing'] = time.time() - start
        
        # 模型调用
        start = time.time()
        await asyncio.sleep(0.5)  # 模拟模型调用时间
        timings['model_inference'] = time.time() - start
        
        # 工具调用（如果有）
        start = time.time()
        await asyncio.sleep(0.1)  # 模拟工具调用时间
        timings['tool_execution'] = time.time() - start
        
        # 响应后处理
        start = time.time()
        await asyncio.sleep(0.01)  # 模拟后处理时间
        timings['response_postprocessing'] = time.time() - start
        
        return timings
    
    def _analyze_resource_usage(self):
        """分析资源使用情况"""
        
        process = psutil.Process()
        
        return {
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'thread_count': process.num_threads(),
            'open_files': len(process.open_files())
        }
    
    def _generate_optimization_suggestions(self, timings, resource_usage):
        """生成优化建议"""
        
        suggestions = []
        
        # 基于计时分析的建议
        if timings.get('model_inference', 0) > 1.0:
            suggestions.append("考虑使用更快的模型或启用模型缓存")
        
        if timings.get('tool_execution', 0) > 0.5:
            suggestions.append("优化工具执行逻辑或使用工具缓存")
        
        # 基于资源使用的建议
        if resource_usage['memory_mb'] > 500:
            suggestions.append("内存使用较高，考虑启用对象池化")
        
        if resource_usage['cpu_percent'] > 80:
            suggestions.append("CPU使用率高，考虑启用异步处理")
        
        if resource_usage['thread_count'] > 50:
            suggestions.append("线程数过多，检查是否有线程泄漏")
        
        return suggestions

# 使用示例
diagnostics = PerformanceDiagnostics()
agent = AssistantAgent("test_agent", model_client)

# 运行诊断
diagnosis = await diagnostics.diagnose_slow_response(
    agent, 
    "请分析一下人工智能的发展趋势"
)
```

#### 内存泄漏检测

```python
import tracemalloc
import gc
from collections import defaultdict

class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self):
        self.snapshots = []
        self.tracking = False
    
    def start_tracking(self):
        """开始内存追踪"""
        tracemalloc.start()
        self.tracking = True
        self.take_snapshot("baseline")
        print("🔍 开始内存泄漏检测...")
    
    def take_snapshot(self, label: str):
        """拍摄内存快照"""
        if not self.tracking:
            return
        
        # 强制垃圾回收
        gc.collect()
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'label': label,
            'snapshot': snapshot,
            'timestamp': time.time()
        })
        
        print(f"📸 拍摄内存快照: {label}")
    
    def analyze_leaks(self, top_n: int = 10):
        """分析内存泄漏"""
        if len(self.snapshots) < 2:
            print("❌ 需要至少2个快照才能分析泄漏")
            return
        
        print("\n🔍 内存泄漏分析报告:")
        print("=" * 50)
        
        # 比较最新和最旧的快照
        first_snapshot = self.snapshots[0]['snapshot']
        last_snapshot = self.snapshots[-1]['snapshot']
        
        # 计算内存增长
        top_stats = last_snapshot.compare_to(first_snapshot, 'lineno')
        
        print(f"📊 内存增长最多的 {top_n} 个位置:")
        for index, stat in enumerate(top_stats[:top_n], 1):
            print(f"{index}. {stat}")
        
        # 分析对象类型增长
        self._analyze_object_growth()
        
        # 生成修复建议
        suggestions = self._generate_leak_fix_suggestions(top_stats)
        
        print("\n💡 修复建议:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")
    
    def _analyze_object_growth(self):
        """分析对象类型增长"""
        
        print("\n📈 对象类型增长分析:")
        
        # 获取当前对象统计
        obj_stats = defaultdict(int)
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            obj_stats[obj_type] += 1
        
        # 显示最多的对象类型
        sorted_stats = sorted(obj_stats.items(), key=lambda x: x[1], reverse=True)
        for obj_type, count in sorted_stats[:10]:
            print(f"  {obj_type}: {count}")
    
    def _generate_leak_fix_suggestions(self, top_stats):
        """生成泄漏修复建议"""
        
        suggestions = []
        
        for stat in top_stats[:5]:
            filename = stat.traceback.format()[-1]
            
            if 'asyncio' in filename:
                suggestions.append("检查异步任务是否正确清理，避免任务泄漏")
            elif 'cache' in filename.lower():
                suggestions.append("检查缓存是否有TTL设置，避免缓存无限增长")
            elif 'list' in str(stat) or 'dict' in str(stat):
                suggestions.append("检查容器对象是否及时清理，避免引用累积")
            elif 'agent' in filename.lower():
                suggestions.append("检查代理对象是否正确释放，避免代理实例泄漏")
        
        if not suggestions:
            suggestions.append("内存增长可能是正常的业务增长，继续监控")
        
        return suggestions
    
    def stop_tracking(self):
        """停止内存追踪"""
        if self.tracking:
            tracemalloc.stop()
            self.tracking = False
            print("⏹️ 停止内存泄漏检测")

# 使用示例
leak_detector = MemoryLeakDetector()

# 开始检测
leak_detector.start_tracking()

# 运行一些可能导致内存泄漏的操作
for i in range(100):
    agent = AssistantAgent(f"agent_{i}", model_client)
    # 模拟一些操作
    await agent.run(task="简单测试")
    
    if i % 20 == 0:
        leak_detector.take_snapshot(f"iteration_{i}")

# 分析结果
leak_detector.analyze_leaks()
leak_detector.stop_tracking()
```

### 5.2 日志分析工具

```python
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict

class LogAnalyzer:
    """日志分析工具"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.patterns = {
            'error': re.compile(r'ERROR.*?(\w+Error|Exception).*?$', re.MULTILINE),
            'warning': re.compile(r'WARNING.*?$', re.MULTILINE),
            'performance': re.compile(r'处理时间.*?(\d+\.?\d*).*?ms', re.MULTILINE),
            'agent': re.compile(r'代理.*?(\w+).*?(成功|失败)', re.MULTILINE),
            'timestamp': re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
        }
    
    def analyze_errors(self, hours: int = 24):
        """分析错误日志"""
        
        print(f"🔍 分析最近 {hours} 小时的错误日志...")
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取错误信息
        errors = self.patterns['error'].findall(content)
        error_counter = Counter(errors)
        
        print(f"📊 发现 {len(errors)} 个错误:")
        for error_type, count in error_counter.most_common(10):
            print(f"  {error_type}: {count} 次")
        
        # 分析错误趋势
        self._analyze_error_trends(content)
        
        return error_counter
    
    def analyze_performance(self):
        """分析性能日志"""
        
        print("⚡ 分析性能指标...")
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取处理时间
        times = [float(t) for t in self.patterns['performance'].findall(content)]
        
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            print(f"📈 性能统计:")
            print(f"  平均处理时间: {avg_time:.2f}ms")
            print(f"  最大处理时间: {max_time:.2f}ms")
            print(f"  最小处理时间: {min_time:.2f}ms")
            print(f"  总请求数: {len(times)}")
            
            # 分析慢请求
            slow_requests = [t for t in times if t > avg_time * 2]
            if slow_requests:
                print(f"  慢请求数: {len(slow_requests)} ({len(slow_requests)/len(times)*100:.1f}%)")
        
        return times
    
    def analyze_agent_activity(self):
        """分析代理活动"""
        
        print("🤖 分析代理活动...")
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取代理活动
        activities = self.patterns['agent'].findall(content)
        
        agent_stats = defaultdict(lambda: {'成功': 0, '失败': 0})
        for agent, status in activities:
            agent_stats[agent][status] += 1
        
        print("📊 代理活动统计:")
        for agent, stats in agent_stats.items():
            total = stats['成功'] + stats['失败']
            success_rate = stats['成功'] / total * 100 if total > 0 else 0
            print(f"  {agent}: 成功 {stats['成功']}, 失败 {stats['失败']}, 成功率 {success_rate:.1f}%")
        
        return agent_stats
    
    def _analyze_error_trends(self, content: str):
        """分析错误趋势"""
        
        # 按小时统计错误
        hourly_errors = defaultdict(int)
        
        lines = content.split('\n')
        for line in lines:
            if 'ERROR' in line:
                timestamp_match = self.patterns['timestamp'].search(line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        hour_key = dt.strftime('%Y-%m-%d %H:00')
                        hourly_errors[hour_key] += 1
                    except ValueError:
                        continue
        
        if hourly_errors:
            print("\n📈 错误趋势 (按小时):")
            sorted_hours = sorted(hourly_errors.items())
            for hour, count in sorted_hours[-24:]:  # 最近24小时
                print(f"  {hour}: {count} 个错误")
    
    def generate_report(self):
        """生成综合分析报告"""
        
        print("📋 生成日志分析报告...")
        print("=" * 60)
        
        # 错误分析
        errors = self.analyze_errors()
        print()
        
        # 性能分析
        performance = self.analyze_performance()
        print()
        
        # 代理活动分析
        agents = self.analyze_agent_activity()
        print()
        
        # 生成建议
        suggestions = self._generate_suggestions(errors, performance, agents)
        
        print("💡 优化建议:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")
        
        return {
            'errors': errors,
            'performance': performance,
            'agents': agents,
            'suggestions': suggestions
        }
    
    def _generate_suggestions(self, errors, performance, agents):
        """生成优化建议"""
        
        suggestions = []
        
        # 基于错误分析的建议
        if errors:
            most_common_error = errors.most_common(1)[0]
            if 'TimeoutError' in most_common_error[0]:
                suggestions.append("超时错误较多，考虑增加超时时间或优化处理逻辑")
            elif 'ConnectionError' in most_common_error[0]:
                suggestions.append("连接错误较多，检查网络连接和服务可用性")
        
        # 基于性能分析的建议
        if performance:
            avg_time = sum(performance) / len(performance)
            if avg_time > 1000:  # 超过1秒
                suggestions.append("平均响应时间较长，考虑性能优化")
        
        # 基于代理活动的建议
        for agent, stats in agents.items():
            total = stats['成功'] + stats['失败']
            if total > 0:
                success_rate = stats['成功'] / total
                if success_rate < 0.9:  # 成功率低于90%
                    suggestions.append(f"代理 {agent} 成功率较低，需要检查和优化")
        
        if not suggestions:
            suggestions.append("系统运行良好，继续监控")
        
        return suggestions

# 使用示例
analyzer = LogAnalyzer('/app/logs/autogen.log')
report = analyzer.generate_report()
```

## 6. 总结

本实战指南涵盖了AutoGen框架从开发到生产部署的完整流程，包括：

### 6.1 核心要点

1. **架构设计**：遵循单一职责原则，合理设计代理和团队结构
2. **性能优化**：使用批处理、连接池、对象池等技术提升性能
3. **内存管理**：实施内存监控和泄漏检测，确保系统稳定
4. **容器化部署**：使用Docker和Kubernetes实现可扩展部署
5. **监控告警**：建立完善的监控体系，及时发现和解决问题
6. **故障排查**：掌握常见问题的诊断和解决方法

### 6.2 最佳实践总结

- **开发阶段**：注重代码质量，编写充分的测试
- **测试阶段**：进行性能测试和压力测试
- **部署阶段**：使用渐进式部署策略
- **运维阶段**：建立完善的监控和告警机制
- **优化阶段**：基于监控数据持续优化性能

通过遵循这些实战经验和最佳实践，可以构建出高性能、高可用的AutoGen应用系统。

<function_calls>
<invoke name="todo_write">
<parameter name="merge">true

