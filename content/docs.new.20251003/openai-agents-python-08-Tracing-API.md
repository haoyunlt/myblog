# OpenAI Agents Python SDK - Tracing 模块 API 详解

## 1. API 总览

Tracing 模块提供了完整的可观测性支持，通过Trace和Span追踪Agent执行流程。

### API 分类

| API类别 | 核心API | 功能描述 |
|---------|---------|---------|
| **Trace创建** | `trace(name)` | 创建工作流追踪 |
| **Span创建** | `agent_span()` | 创建Agent执行span |
| | `generation_span()` | 创建模型生成span |
| | `function_span()` | 创建函数执行span |
| | `custom_span()` | 创建自定义span |
| **处理器** | `TracingProcessor.on_trace_start()` | 追踪开始回调 |
| | `TracingProcessor.on_span_end()` | Span结束回调 |

## 2. trace() API

**API签名：**
```python
def trace(
    name: str,
    *,
    group_id: str | None = None,
    metadata: dict[str, Any] | None = None
) -> Trace
```

**使用示例：**
```python
from agents import trace, run

# 基础用法
with trace("Customer Service"):
    result = await run(agent, "用户查询")

# 带分组和元数据
with trace(
    "Order Processing",
    group_id="order_123",
    metadata={"customer_id": "user_456", "priority": "high"}
):
    await run(validator, order)
    await run(processor, order)
```

## 3. TracingProcessor API

**核心方法：**

```python
class CustomProcessor(TracingProcessor):
    def on_trace_start(self, trace: Trace):
        """追踪开始时调用"""
        print(f"开始追踪: {trace.name}")
    
    def on_trace_end(self, trace: Trace):
        """追踪结束时调用"""
        print(f"完成追踪: {trace.trace_id}")
    
    def on_span_start(self, span: Span):
        """Span开始时调用"""
        data = span.span_data
        print(f"开始Span: {data.type}")
    
    def on_span_end(self, span: Span):
        """Span结束时调用"""
        duration = calculate_duration(span)
        print(f"Span耗时: {duration}ms")
    
    def shutdown(self):
        """清理资源"""
        pass
    
    def force_flush(self):
        """强制刷新"""
        pass
```

Tracing模块通过Trace和Span的层次结构，提供了完整的执行追踪和监控能力。

