# OpenAI Agents Python SDK - Streaming 模块 API 详解

## 1. API 总览

Streaming 模块提供了流式事件系统，支持实时获取Agent执行过程中的各种事件。

### API 分类

| API类别 | 核心API | 功能描述 |
|---------|---------|---------|
| **流式执行** | `run_streamed(agent, input)` | 流式运行Agent |
| **事件迭代** | `async for event in stream` | 异步迭代事件 |
| **事件类型** | `RunStartEvent` | 运行开始事件 |
| | `RunStepDoneEvent` | 步骤完成事件 |
| | `RunDoneEvent` | 运行完成事件 |

## 2. run_streamed() API

**API签名：**
```python
async def run_streamed(
    agent: Agent,
    input: str | list,
    *,
    config: RunConfig | None = None,
    **kwargs
) -> AsyncIterator[StreamEvent]
```

**使用示例：**

```python
from agents import Agent, run_streamed

agent = Agent(name="assistant", instructions="你是助手")

# 流式执行
async for event in run_streamed(agent, "你好"):
    if event["type"] == "response.text.delta":
        print(event["delta"], end="", flush=True)
    elif event["type"] == "response.done":
        print("\n完成!")
```

## 3. 流式事件类型

### 3.1 RunStartEvent
```python
{
    "type": "run.start",
    "run_id": "run_abc123"
}
```

### 3.2 TextDeltaEvent
```python
{
    "type": "response.text.delta",
    "delta": "你好",
    "content_index": 0
}
```

### 3.3 FunctionCallEvent
```python
{
    "type": "response.function_call_arguments.delta",
    "delta": '{"query": "天气"}',
    "call_id": "call_123"
}
```

### 3.4 RunDoneEvent
```python
{
    "type": "run.done",
    "run_result": RunResult(...)
}
```

## 4. 高级用法

### 4.1 事件过滤
```python
async for event in run_streamed(agent, "查询"):
    # 只处理文本增量
    if event["type"] == "response.text.delta":
        handle_text(event["delta"])
```

### 4.2 进度追踪
```python
async for event in run_streamed(agent, input):
    if event["type"] == "run.step.done":
        step_num = event.get("step_number")
        print(f"完成步骤 {step_num}")
```

### 4.3 实时UI更新
```python
async for event in run_streamed(agent, query):
    if event["type"] == "response.text.delta":
        await websocket.send_text(event["delta"])
    elif event["type"] == "response.function_call":
        await websocket.send_json({
            "tool": event["name"],
            "status": "calling"
        })
```

Streaming模块通过丰富的事件类型，支持构建实时响应的AI应用。

