---
title: "AutoGen-01-PythonCore-API"
date: 2025-10-05T01:01:58+08:00
draft: false
tags:
  - API设计
  - 接口文档
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - AutoGen-01-PythonCore-API"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# AutoGen-01-PythonCore-API

## 核心API列表

本模块提供的核心API按功能分类如下：

### 代理相关API
- `Agent`：代理协议定义
- `BaseAgent`：代理基础实现类
- `RoutedAgent`：基于路由的代理实现类

### 运行时API
- `AgentRuntime`：运行时协议定义
- `SingleThreadedAgentRuntime`：单线程运行时实现

### 消息处理API
- `message_handler`：通用消息处理装饰器
- `event`：事件消息处理装饰器
- `rpc`：RPC消息处理装饰器

### 基础设施API
- `AgentId`：代理唯一标识
- `MessageContext`：消息上下文
- `CancellationToken`：取消令牌

---

## API详细规格

### Agent协议

#### 基本信息
- **名称**：`Agent`
- **类型**：Protocol协议定义
- **作用**：定义代理的核心接口规范

#### 接口定义

```python
@runtime_checkable
class Agent(Protocol):
    @property
    def metadata(self) -> AgentMetadata:
        """获取代理元数据，包含类型和描述信息"""
        
    @property  
    def id(self) -> AgentId:
        """获取代理唯一标识"""
        
    async def bind_id_and_runtime(self, id: AgentId, runtime: AgentRuntime) -> None:
        """绑定代理ID和运行时实例"""
        
    async def on_message(self, message: Any, ctx: MessageContext) -> Any:
        """处理接收到的消息，返回响应结果"""
        
    async def save_state(self) -> Mapping[str, Any]:
        """保存代理状态为JSON可序列化对象"""
        
    async def load_state(self, state: Mapping[str, Any]) -> None:
        """从保存的状态中恢复代理状态"""
        
    async def close(self) -> None:
        """清理代理资源，在运行时关闭时调用"""
```

#### 方法说明

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| metadata | 无 | AgentMetadata | 代理元数据，包含key、type、description |
| id | 无 | AgentId | 代理唯一标识，格式为type:key |
| bind_id_and_runtime | id, runtime | None | 将代理实例绑定到特定运行时 |
| on_message | message, ctx | Any | 消息处理入口，业务逻辑核心 |
| save_state | 无 | Mapping[str, Any] | 状态持久化，必须JSON兼容 |
| load_state | state | None | 状态恢复，与save_state配对 |
| close | 无 | None | 资源清理，可选实现 |

#### 实现示例

```python
class SimpleAgent:
    def __init__(self, description: str):
        self._description = description
        self._state = {}
    
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            key=self._id.key,
            type=self._id.type, 
            description=self._description
        )
    
    async def on_message(self, message: Any, ctx: MessageContext) -> Any:
        # 1. 消息类型检查
        if isinstance(message, TextMessage):
            # 2. 业务逻辑处理
            response = f"收到消息: {message.content}"
            # 3. 状态更新
            self._state['last_message'] = message.content
            return TextResponse(content=response)
        
        # 4. 未知消息类型处理
        raise CantHandleException(f"不支持的消息类型: {type(message)}")
```

---

### BaseAgent基础实现

#### 基本信息
- **名称**：`BaseAgent`
- **类型**：抽象基类
- **继承**：实现Agent协议
- **作用**：提供代理的基础实现和注册机制

#### 核心方法

```python
class BaseAgent(ABC, Agent):
    def __init__(self, description: str) -> None:
        """初始化代理，设置描述信息"""
        
    async def bind_id_and_runtime(self, id: AgentId, runtime: AgentRuntime) -> None:
        """绑定代理ID和运行时，确保唯一性"""
        
    @abstractmethod
    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        """抽象消息处理方法，子类必须实现"""
        
    async def send_message(self, message: Any, recipient: AgentId, 
                          cancellation_token: CancellationToken = None) -> Any:
        """发送消息到指定代理"""
        
    async def publish_message(self, message: Any, topic_id: TopicId,
                             cancellation_token: CancellationToken = None) -> None:
        """发布消息到主题"""
    
    @classmethod
    async def register(cls, runtime: AgentRuntime, type: str, 
                      factory: Callable[[], Self]) -> AgentType:
        """注册代理类型到运行时"""
```

#### 注册机制详解

```python
# 代理工厂注册示例
async def register_agent_example():
    runtime = SingleThreadedAgentRuntime()
    
    # 1. 定义代理工厂函数
    def create_echo_agent() -> EchoAgent:
        return EchoAgent("回声代理")
    
    # 2. 注册代理类型
    agent_type = await EchoAgent.register(
        runtime=runtime,
        type="echo_agent",  # 代理类型名称
        factory=create_echo_agent  # 工厂函数
    )
    
    # 3. 启动运行时
    await runtime.start()
    
    # 4. 创建代理实例
    agent_id = await runtime.get("echo_agent", key="default")
    
    # 5. 发送消息
    response = await runtime.send_message(
        message=TextMessage("你好"),
        recipient=agent_id
    )
```

#### 订阅机制

```python
# 使用装饰器定义订阅
@subscription_factory(TypeSubscription("user_message", "echo_agent"))
class EchoAgent(BaseAgent):
    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        if isinstance(message, UserMessage):
            return EchoResponse(content=f"回声: {message.content}")
        return None
```

---

### RoutedAgent路由代理

#### 基本信息
- **名称**：`RoutedAgent`
- **类型**：具体实现类
- **继承**：BaseAgent
- **作用**：基于装饰器的消息路由和处理

#### 路由装饰器

```python
class ChatAgent(RoutedAgent):
    def __init__(self):
        super().__init__("聊天代理")
    
    @event
    async def handle_user_message(self, message: UserMessage, ctx: MessageContext) -> None:
        """处理用户消息事件（无返回值）"""
        print(f"用户说: {message.content}")
        # 发布响应事件
        await self.publish_message(
            BotResponse(content=f"我收到了: {message.content}"),
            ctx.topic_id
        )
    
    @rpc
    async def get_status(self, message: StatusRequest, ctx: MessageContext) -> StatusResponse:
        """处理状态查询RPC（有返回值）"""
        return StatusResponse(
            status="在线",
            message_count=self._message_count
        )
    
    @message_handler(match=lambda msg, ctx: msg.priority == "high")
    async def handle_urgent_message(self, message: UrgentMessage, ctx: MessageContext) -> None:
        """使用条件匹配的消息处理器"""
        print(f"紧急消息: {message.content}")
```

#### 路由机制说明

| 装饰器 | 适用场景 | 返回值 | is_rpc标志 |
|--------|----------|---------|-----------|
| @event | 事件通知、异步处理 | None | False |
| @rpc | 同步调用、需要响应 | Any | True |
| @message_handler | 通用处理、条件路由 | Any | 根据上下文 |

#### 消息路由算法

```python
async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any | None:
    # 1. 按消息类型查找处理器
    message_type = type(message)
    handlers = self._handlers.get(message_type, [])
    
    # 2. 遍历处理器，应用路由条件
    for handler in handlers:
        if handler.router(message, ctx):  # 路由条件匹配
            return await handler(self, message, ctx)
    
    # 3. 没有匹配的处理器
    await self.on_unhandled_message(message, ctx)
    return None
```

---

### SingleThreadedAgentRuntime运行时

#### 基本信息
- **名称**：`SingleThreadedAgentRuntime`
- **类型**：具体实现类
- **实现**：AgentRuntime协议
- **作用**：单线程异步运行时实现

#### 核心API

```python
class SingleThreadedAgentRuntime(AgentRuntime):
    async def start(self) -> None:
        """启动运行时，开始处理消息队列"""
        
    async def stop(self) -> None:
        """停止运行时，等待处理完成后退出"""
        
    async def send_message(self, message: Any, recipient: AgentId,
                          sender: AgentId = None, 
                          cancellation_token: CancellationToken = None) -> Any:
        """发送消息并等待响应"""
        
    async def publish_message(self, message: Any, topic_id: TopicId,
                             sender: AgentId = None,
                             cancellation_token: CancellationToken = None) -> None:
        """发布消息到订阅者"""
        
    async def register_factory(self, type: str, agent_factory: Callable[[], Agent]) -> AgentType:
        """注册代理工厂"""
        
    async def add_subscription(self, subscription: Subscription) -> None:
        """添加消息订阅"""
```

#### 消息处理流程

```python
# 发送消息的完整流程
async def send_message_flow():
    # 1. 创建消息信封
    envelope = SendMessageEnvelope(
        message=user_message,
        recipient=target_agent_id,
        sender=sender_agent_id,
        future=asyncio.Future(),
        cancellation_token=token
    )
    
    # 2. 投递到消息队列
    await self._message_queue.put(envelope)
    
    # 3. 异步处理消息
    await self._process_send(envelope)
    
    # 4. 获取目标代理实例
    recipient_agent = await self._get_agent(envelope.recipient)
    
    # 5. 构造消息上下文
    context = MessageContext(
        sender=envelope.sender,
        topic_id=None,  # RPC消息无主题
        is_rpc=True,
        cancellation_token=envelope.cancellation_token
    )
    
    # 6. 调用代理消息处理器
    response = await recipient_agent.on_message(envelope.message, context)
    
    # 7. 设置返回结果
    envelope.future.set_result(response)
```

#### 发布-订阅流程

```python
async def publish_message_flow():
    # 1. 查找订阅者
    recipients = await self._subscription_manager.get_subscribed_recipients(topic_id)
    
    # 2. 并发发送给所有订阅者
    tasks = []
    for agent_id in recipients:
        # 跳过发送者自己
        if sender and agent_id == sender:
            continue
            
        # 创建消息处理任务
        task = self._send_to_subscriber(agent_id, message, context)
        tasks.append(task)
    
    # 3. 等待所有处理完成
    await asyncio.gather(*tasks, return_exceptions=True)
```

---

### 消息装饰器API

#### @event装饰器

```python
@event
async def handle_notification(self, message: NotificationMessage, ctx: MessageContext) -> None:
    """事件处理器特征：
    - 无返回值（返回类型必须是None）
    - 用于异步事件通知
    - is_rpc上下文标志为False
    """
    # 业务逻辑处理
    pass
```

#### @rpc装饰器

```python
@rpc  
async def process_request(self, message: RequestMessage, ctx: MessageContext) -> ResponseMessage:
    """RPC处理器特征：
    - 必须有返回值
    - 用于同步请求-响应
    - is_rpc上下文标志为True
    """
    return ResponseMessage(result="处理完成")
```

#### @message_handler装饰器

```python
@message_handler(match=lambda msg, ctx: msg.category == "important")
async def handle_important(self, message: CategoryMessage, ctx: MessageContext) -> Any:
    """通用处理器特征：
    - 支持条件路由匹配
    - 返回值可选
    - 根据上下文确定处理模式
    """
    if ctx.is_rpc:
        return ProcessResult(status="已处理")
    else:
        # 事件模式，无需返回
        await self.log_important_event(message)
        return None
```

---

## 异常处理与最佳实践

### 异常类型

- `CantHandleException`：代理无法处理特定消息类型
- `UndeliverableException`：消息无法投递到目标代理
- `LookupError`：代理或订阅不存在
- `MessageDroppedException`：消息被干预机制丢弃

### 最佳实践

#### 代理实现

```python
class RobustAgent(RoutedAgent):
    async def on_unhandled_message(self, message: Any, ctx: MessageContext) -> None:
        """重写未处理消息的默认行为"""
        logger.warning(f"未处理的消息类型: {type(message).__name__}")
        
        # 可选：发送错误响应
        if ctx.is_rpc:
            return ErrorResponse(message="不支持的消息类型")
    
    @event
    async def handle_with_error_recovery(self, message: DataMessage, ctx: MessageContext) -> None:
        try:
            await self.process_data(message.data)
        except Exception as e:
            logger.error(f"处理失败: {e}")
            # 发布错误事件
            await self.publish_message(
                ErrorEvent(agent_id=self.id, error=str(e)),
                ctx.topic_id
            )
```

#### 运行时使用

```python
async def runtime_best_practices():
    runtime = SingleThreadedAgentRuntime()
    
    try:
        # 1. 设置取消令牌
        cancel_token = CancellationToken()
        
        # 2. 带超时的消息发送
        response = await asyncio.wait_for(
            runtime.send_message(message, target_id, cancellation_token=cancel_token),
            timeout=30.0
        )
        
    except asyncio.TimeoutError:
        # 3. 主动取消操作
        cancel_token.cancel()
        logger.warning("消息发送超时")
        
    except CantHandleException:
        logger.error("目标代理无法处理该消息")
        
    finally:
        # 4. 确保运行时正常关闭
        await runtime.stop()
```
