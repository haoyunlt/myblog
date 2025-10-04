# LangChain-03-LanguageModels-时序图

## 文档说明

本文档通过详细的时序图展示 **Language Models 模块**在各种场景下的执行流程，包括简单调用、流式生成、工具调用、结构化输出等。

---

## 1. 基础调用场景

### 1.1 同步 invoke 完整流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Model as BaseChatModel
    participant Conv as Input Converter
    participant Sub as ChatOpenAI
    participant API as OpenAI API
    participant Cache
    participant CB as Callbacks

    User->>Model: invoke(messages, config)
    Model->>Conv: _convert_input(messages)
    Conv-->>Model: List[BaseMessage]

    Model->>Cache: get(cache_key)
    alt 缓存命中
        Cache-->>Model: cached_response
        Model-->>User: AIMessage (from cache)
    else 缓存未命中
        Model->>CB: on_chat_model_start(messages)
        Model->>Sub: _generate(messages, **kwargs)

        Sub->>Sub: 转换消息格式<br/>LangChain → OpenAI
        Sub->>API: POST /v1/chat/completions<br/>{model, messages, temperature, ...}

        Note over API: 模型推理<br/>（200-5000ms）

        API-->>Sub: HTTP 200<br/>{choices, usage, ...}
        Sub->>Sub: 解析响应<br/>OpenAI → LangChain
        Sub-->>Model: ChatResult(generations, llm_output)

        Model->>Cache: set(cache_key, response)
        Model->>CB: on_llm_end(llm_output)
        Model-->>User: AIMessage(content, usage_metadata)
    end
```

**关键步骤说明**：

1. **输入转换**（步骤 2-3）：
   - 字符串 → `[HumanMessage]`
   - 保持消息列表不变
   - `PromptValue` → `to_messages()`

2. **缓存检查**（步骤 4-5）：
   - 基于输入和模型参数生成缓存键
   - 命中率约 20-40%（重复查询）
   - 节省 API 调用成本

3. **API 调用**（步骤 8-13）：
   - 格式转换：约 1-5ms
   - 网络延迟：50-200ms
   - 模型推理：200-5000ms
   - 响应解析：1-10ms

4. **回调通知**（步骤 7、15）：
   - 异步执行，不阻塞主流程
   - 用于追踪、日志、监控

---

### 1.2 异步 ainvoke 流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Model
    participant Loop as asyncio Event Loop
    participant API as OpenAI API (async)
    participant CB as AsyncCallbacks

    User->>Model: await ainvoke(messages)
    Model->>Loop: create_task(callbacks.on_start())
    Loop-->>CB: on_chat_model_start()

    Model->>API: await client.acreate(...)
    Note over Model,API: 异步等待<br/>事件循环可处理其他任务

    API-->>Model: response
    Model->>Loop: create_task(callbacks.on_end())
    Loop-->>CB: on_llm_end()

    Model-->>User: AIMessage
```

**异步优势**：
- 单线程处理 1000+ 并发请求
- I/O 等待时 CPU 不空闲
- 内存开销低（无线程栈）

---

## 2. 流式输出场景

### 2.1 stream 流式生成

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Model
    participant API as OpenAI API
    participant CB as Callbacks

    User->>Model: for chunk in stream(input)
    Model->>CB: on_llm_start()
    Model->>API: POST /chat/completions<br/>stream=True

    loop 流式接收 SSE
        API-->>Model: data: {"choices": [{"delta": {"content": "Hello"}}]}
        Model->>Model: 解析 delta
        Model->>CB: on_llm_new_token("Hello")
        Model-->>User: yield AIMessageChunk("Hello")
        User->>User: 实时显示: "Hello"

        API-->>Model: data: {"choices": [{"delta": {"content": " World"}}]}
        Model->>CB: on_llm_new_token(" World")
        Model-->>User: yield AIMessageChunk(" World")
        User->>User: 实时显示: " World"
    end

    API-->>Model: data: [DONE]
    Model->>CB: on_llm_end()
```

**SSE 格式示例**：

```text
data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" World"}}]}

data: [DONE]
```

**性能对比**：

| 指标 | invoke | stream | 改善 |
|-----|--------|--------|------|
| 首字节延迟 | 2000ms | 200ms | 10x |
| 总延迟 | 2000ms | 2100ms | -5% |
| 用户体验 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 显著提升 |

---

### 2.2 astream_events 细粒度事件流

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Chain as prompt | model | parser
    participant Prompt
    participant Model
    participant Parser

    User->>Chain: async for event in astream_events(input)

    Chain->>Prompt: astream(input)
    Prompt-->>Chain: on_chain_start(name="ChatPromptTemplate")
    Chain-->>User: {"event": "on_chain_start", "name": "ChatPromptTemplate"}

    Prompt-->>Chain: on_chain_end(output=prompt_value)
    Chain-->>User: {"event": "on_chain_end", "name": "ChatPromptTemplate"}

    Chain->>Model: astream(prompt_value)
    Model-->>Chain: on_llm_start()
    Chain-->>User: {"event": "on_llm_start", "name": "ChatOpenAI"}

    loop 流式生成
        Model-->>Chain: on_llm_stream(chunk="Hi")
        Chain-->>User: {"event": "on_llm_stream", "data": {"chunk": "Hi"}}
    end

    Model-->>Chain: on_llm_end()
    Chain-->>User: {"event": "on_llm_end", "name": "ChatOpenAI"}

    Chain->>Parser: astream(ai_message)
    Parser-->>Chain: on_chain_start(name="StrOutputParser")
    Chain-->>User: {"event": "on_chain_start", "name": "StrOutputParser"}

    Parser-->>Chain: on_chain_stream(chunk="Hi")
    Chain-->>User: {"event": "on_chain_stream", "data": {"chunk": "Hi"}}

    Parser-->>Chain: on_chain_end()
    Chain-->>User: {"event": "on_chain_end", "name": "StrOutputParser"}
```

**事件过滤示例**：

```python
async for event in chain.astream_events(
    input_data,
    version="v2",
    include_types=["llm"]  # 仅 LLM 事件
):
    if event["event"] == "on_llm_stream":
        print(event["data"]["chunk"], end="", flush=True)
```

---

## 3. 批处理场景

### 3.1 batch 并发调用

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Model
    participant Pool as ThreadPoolExecutor
    participant W1 as Worker 1
    participant W2 as Worker 2
    participant W3 as Worker 3
    participant API

    User->>Model: batch([input1, input2, input3])
    Model->>Model: 检查 max_concurrency=10
    Model->>Pool: submit 3 tasks

    par 并发执行
        Pool->>W1: invoke(input1)
        W1->>API: request1
        API-->>W1: response1 (AIMessage)
        W1-->>Pool: result1
    and
        Pool->>W2: invoke(input2)
        W2->>API: request2
        API-->>W2: response2 (AIMessage)
        W2-->>Pool: result2
    and
        Pool->>W3: invoke(input3)
        W3->>API: request3
        API-->>W3: response3 (AIMessage)
        W3-->>Pool: result3
    end

    Pool-->>Model: [result1, result2, result3]
    Model-->>User: [AIMessage, AIMessage, AIMessage]
```

**并发控制**：

```python
# 默认并发数：min(32, len(inputs))
results = model.batch(inputs)

# 自定义并发数
results = model.batch(
    inputs,
    config={"max_concurrency": 5}
)
```

---

## 4. 工具调用场景

### 4.1 bind_tools 完整流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Model as model.bind_tools([search])
    participant API
    participant Tool as search_tool

    User->>Model: invoke("What's the weather in SF?")
    Model->>API: POST /chat/completions<br/>tools=[{name: "search", ...}]

    API-->>Model: AIMessage(<br/>  content="",<br/>  tool_calls=[{<br/>    name: "search",<br/>    args: {"query": "weather SF"}<br/>  }]<br/>)

    Model-->>User: AIMessage with tool_calls

    User->>Tool: search("weather SF")
    Tool-->>User: "Sunny, 72°F"

    User->>Model: invoke([<br/>  ...,<br/>  AIMessage(tool_calls=[...]),<br/>  ToolMessage(content="Sunny, 72°F", tool_call_id="call_123")<br/>])

    Model->>API: POST /chat/completions<br/>messages=[..., tool result]
    API-->>Model: AIMessage("The weather in SF is sunny, 72°F")
    Model-->>User: AIMessage
```

**关键点**：

1. **工具绑定**：
   - 模型接收工具 schema
   - 自动生成 `tools` 参数

2. **工具调用**：
   - 模型返回 `tool_calls` 列表
   - 包含工具名和参数

3. **工具结果**：
   - 用户执行工具
   - 构建 `ToolMessage`
   - 继续对话

---

### 4.2 并行工具调用

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Model
    participant API
    participant Search as search_tool
    participant Weather as weather_tool

    User->>Model: invoke("Weather in SF and latest news?")
    Model->>API: POST /chat/completions

    API-->>Model: AIMessage(tool_calls=[<br/>  {name: "weather", args: {"city": "SF"}},<br/>  {name: "search", args: {"query": "news"}}<br/>])

    par 并行执行工具
        User->>Weather: weather("SF")
        Weather-->>User: "Sunny, 72°F"
    and
        User->>Search: search("news")
        Search-->>User: "Breaking: ..."
    end

    User->>Model: invoke([<br/>  ...,<br/>  ToolMessage("Sunny, 72°F"),<br/>  ToolMessage("Breaking: ...")<br/>])

    Model->>API: POST /chat/completions
    API-->>Model: AIMessage("In SF it's sunny..., and the news is...")
    Model-->>User: AIMessage
```

---

## 5. 结构化输出场景

### 5.1 with_structured_output (function_calling)

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Structured as model.with_structured_output(Person)
    participant Model
    participant API
    participant Parser

    User->>Structured: invoke("Alice is 30 years old")
    Structured->>Model: bind_tools([extract_person], tool_choice=...)

    Model->>API: POST /chat/completions<br/>tools=[{<br/>  name: "extract_person",<br/>  parameters: {...}<br/>}]<br/>tool_choice: required

    API-->>Model: AIMessage(tool_calls=[{<br/>  name: "extract_person",<br/>  args: {"name": "Alice", "age": 30}<br/>}])

    Model-->>Structured: AIMessage with tool_calls
    Structured->>Parser: parse_tool_calls()
    Parser->>Parser: 验证 Pydantic schema

    alt 验证成功
        Parser-->>Structured: Person(name="Alice", age=30)
        Structured-->>User: Person(name="Alice", age=30)
    else 验证失败
        Parser-->>Structured: raise ValidationError
        Structured-->>User: raise ValidationError
    end
```

**Schema 定义**：

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    email: Optional[str] = Field(description="Email address")

structured_model = model.with_structured_output(Person)
person = structured_model.invoke("Extract: Alice (30)")
```

---

### 5.2 with_structured_output (json_mode)

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Structured
    participant Model
    participant API
    participant Parser as JsonOutputParser

    User->>Structured: invoke("Alice is 30")
    Structured->>Model: bind(response_format={"type": "json_object"})

    Model->>API: POST /chat/completions<br/>response_format: json_object

    API-->>Model: AIMessage(content='{"name": "Alice", "age": 30}')
    Model-->>Structured: AIMessage

    Structured->>Parser: parse(content)
    Parser->>Parser: json.loads(content)
    Parser->>Parser: 验证 schema

    Parser-->>Structured: {"name": "Alice", "age": 30}
    Structured-->>User: {"name": "Alice", "age": 30}
```

**两种方法对比**：

| 特性 | function_calling | json_mode |
|-----|-----------------|-----------|
| 准确性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 速度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 复杂 schema | ✅ | ⚠️ |
| 模型支持 | OpenAI, Anthropic | OpenAI GPT-4+ |

---

## 6. 错误处理场景

### 6.1 重试机制

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Retry as model.with_retry()
    participant Model
    participant API

    User->>Retry: invoke(input)

    Retry->>Model: 尝试 1
    Model->>API: POST /chat/completions
    API-->>Model: 503 Service Unavailable
    Model-->>Retry: raise APIError
    Note over Retry: 等待 1s

    Retry->>Model: 尝试 2
    Model->>API: POST /chat/completions
    API-->>Model: 429 Rate Limit
    Model-->>Retry: raise RateLimitError
    Note over Retry: 等待 2s

    Retry->>Model: 尝试 3
    Model->>API: POST /chat/completions
    API-->>Model: 200 OK + response
    Model-->>Retry: AIMessage

    Retry-->>User: AIMessage
```

**配置示例**：

```python
model_with_retry = model.with_retry(
    retry_if_exception_type=(RateLimitError, APIError),
    max_attempt_number=3,
    wait_exponential_jitter=True
)
```

---

### 6.2 回退机制

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Fallback as gpt4.with_fallbacks([gpt3.5, claude])
    participant GPT4
    participant GPT35
    participant Claude

    User->>Fallback: invoke(input)

    Fallback->>GPT4: invoke(input)
    GPT4-->>Fallback: raise QuotaError
    Note over Fallback: GPT-4 失败，尝试备用

    Fallback->>GPT35: invoke(input)
    GPT35-->>Fallback: raise RateLimitError
    Note over Fallback: GPT-3.5 失败，尝试下一个

    Fallback->>Claude: invoke(input)
    Claude-->>Fallback: AIMessage
    Note over Fallback: Claude 成功

    Fallback-->>User: AIMessage
```

---

## 7. 性能优化场景

### 7.1 缓存命中流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Model
    participant Cache

    User->>Model: invoke("What is 2+2?")
    Model->>Cache: get(hash(input + params))
    Cache-->>Model: None (未命中)

    Model->>Model: 调用 API (慢)
    Note over Model: 2000ms
    Model->>Cache: set(key, response)
    Model-->>User: AIMessage ("4")

    User->>Model: invoke("What is 2+2?")
    Model->>Cache: get(hash(input + params))
    Cache-->>Model: AIMessage ("4")
    Note over Model: 缓存命中 (5ms)
    Model-->>User: AIMessage ("4")
```

**缓存配置**：

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# 内存缓存（进程内）
set_llm_cache(InMemoryCache())

# 持久化缓存（跨进程）
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

---

## 8. 总结

本文档详细展示了 **Language Models 模块**的关键执行时序：

1. **基础调用**：invoke、ainvoke 的完整流程
2. **流式输出**：stream、astream_events 的实时传输
3. **批处理**：batch 的并发执行
4. **工具调用**：bind_tools 的多轮对话
5. **结构化输出**：with_structured_output 的两种实现
6. **错误处理**：重试和回退机制
7. **性能优化**：缓存策略

每张时序图包含：
- 详细的步骤编号
- 关键数据流转
- 性能瓶颈标注
- 配置示例和最佳实践

