---
title: "LangChain-02-Runnables-API"
date: 2025-10-04T20:42:30+08:00
draft: false
tags:
  - LangChain
  - API设计
  - 接口文档
  - 源码分析
categories:
  - LangChain
  - AI框架
  - Python
series: "langchain-source-analysis"
description: "LangChain 源码剖析 - LangChain-02-Runnables-API"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# LangChain-02-Runnables-API

## API 概览

Runnables 模块对外提供的核心 API 分为以下几类：

1. **基础执行 API**: `invoke`, `ainvoke`, `batch`, `abatch`, `stream`, `astream`
2. **组合构造 API**: `|` 操作符, `pipe`, `RunnableSequence`, `RunnableParallel`
3. **增强功能 API**: `with_retry`, `with_fallbacks`, `with_config`
4. **配置管理 API**: `configurable_fields`, `configurable_alternatives`
5. **工具函数 API**: `@chain` 装饰器, `RunnableLambda`, `RunnablePassthrough`

本文档按照功能分类，详细描述每个 API 的使用方法、参数说明和核心实现。

---

## API-01: invoke - 单次同步执行

### 基本信息

- **方法名**: `invoke`
- **方法签名**: `def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output`
- **所属类**: `Runnable[Input, Output]`
- **调用方式**: 实例方法

### 功能说明

将单个输入转换为输出，这是 Runnable 的核心方法。所有 Runnable 子类必须实现此方法或依赖默认实现。

### 参数说明

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `input` | `Input` | 是 | - | 输入数据，类型由 Runnable 的泛型参数决定 |
| `config` | `Optional[RunnableConfig]` | 否 | `None` | 运行时配置，包含回调、标签、元数据等 |
| `**kwargs` | `Any` | 否 | - | 额外的关键字参数，传递给底层实现 |

**RunnableConfig 字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `callbacks` | `Callbacks` | 回调处理器列表 |
| `tags` | `list[str]` | 用于追踪的标签 |
| `metadata` | `dict[str, Any]` | 自定义元数据 |
| `run_name` | `str` | 运行名称 |
| `max_concurrency` | `int` | 批处理最大并发数 |
| `recursion_limit` | `int` | 递归深度限制，默认 25 |
| `configurable` | `dict[str, Any]` | 可配置字段的值 |

### 返回值

| 类型 | 说明 |
|------|------|
| `Output` | 输出数据，类型由 Runnable 的泛型参数决定 |

### 核心代码

```python
from typing import Any, Generic, Optional, TypeVar
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.callbacks import CallbackManager

Input = TypeVar("Input")
Output = TypeVar("Output")

class Runnable(Generic[Input, Output]):
    @abstractmethod
    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Output:
        """
        将输入转换为输出

        参数:
            input: 输入数据
            config: 运行时配置
            **kwargs: 额外参数

        返回:
            转换后的输出
        """
        pass
```

**实际实现示例（RunnableSequence）**:

```python
def invoke(
    self,
    input: Input,
    config: Optional[RunnableConfig] = None,
    **kwargs: Any
) -> Output:
    # 确保配置存在
    config = ensure_config(config)

    # 获取回调管理器
    callback_manager = CallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        local_callbacks=None,
        verbose=False,
        inheritable_tags=config.get("tags"),
        local_tags=None,
        inheritable_metadata=config.get("metadata"),
        local_metadata=None
    )

    # 创建运行管理器
    run_manager = callback_manager.on_chain_start(
        {"name": self.get_name()},
        input,
        run_id=config.get("run_id")
    )

    try:
        # 逐步执行序列中的每个 Runnable
        output = input
        for i, step in enumerate(self.steps):
            # 为每个步骤创建子配置
            step_config = patch_config(
                config,
                callbacks=run_manager.get_child(f"seq:step:{i+1}")
            )
            # 执行步骤
            output = step.invoke(output, step_config, **kwargs)

        # 触发结束回调
        run_manager.on_chain_end(output)
        return output

    except Exception as e:
        # 触发错误回调
        run_manager.on_chain_error(e)
        raise
```

### 使用示例

**基础用法**:

```python
from langchain_core.runnables import RunnableLambda

# 创建一个简单的 Runnable
runnable = RunnableLambda(lambda x: x * 2)

# 执行
result = runnable.invoke(5)
print(result)  # 输出: 10
```

**带配置的用法**:

```python
from langchain_core.callbacks import StdOutCallbackHandler

# 使用回调追踪执行
result = runnable.invoke(
    5,
    config={
        "callbacks": [StdOutCallbackHandler()],
        "tags": ["example", "multiply"],
        "metadata": {"user_id": "123"}
    }
)
```

**链式调用**:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 构建链
chain = (
    ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    | ChatOpenAI()
    | StrOutputParser()
)

# 执行链
result = chain.invoke({"topic": "programming"})
print(result)
```

### 异常处理

可能抛出的异常:

- `ValueError`: 输入类型或格式不正确
- `TypeError`: 参数类型错误
- `RuntimeError`: 执行过程中的运行时错误
- 具体 Runnable 实现可能抛出的其他异常

### 性能考虑

- **延迟**: 同步阻塞执行，总延迟 = 各步骤延迟之和
- **并发**: 单个 `invoke` 不支持并发，需使用 `batch`
- **内存**: 仅保存当前步骤的输入输出，内存占用小

---

## API-02: ainvoke - 单次异步执行

### 基本信息

- **方法名**: `ainvoke`
- **方法签名**: `async def ainvoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output`
- **所属类**: `Runnable[Input, Output]`
- **调用方式**: 异步实例方法

### 功能说明

`invoke` 的异步版本，用于异步执行环境（如 FastAPI、asyncio 应用）。默认实现使用 `run_in_executor` 在线程池中执行同步的 `invoke`。

### 参数说明

与 `invoke` 完全相同。

### 返回值

返回一个协程（Coroutine），需使用 `await` 等待结果。

### 核心代码

```python
async def ainvoke(
    self,
    input: Input,
    config: Optional[RunnableConfig] = None,
    **kwargs: Any,
) -> Output:
    """
    异步执行 Runnable

    默认实现将同步 invoke 委托给线程池执行
    子类可重写此方法提供原生异步实现
    """
    return await run_in_executor(
        config,
        self.invoke,
        input,
        config,
        **kwargs
    )
```

**原生异步实现示例（ChatModel）**:

```python
async def ainvoke(
    self,
    input: LanguageModelInput,
    config: Optional[RunnableConfig] = None,
    **kwargs: Any,
) -> BaseMessage:
    # 原生异步 HTTP 请求
    async with aiohttp.ClientSession() as session:
        async with session.post(
            self.api_url,
            json=self._format_input(input),
            headers=self._get_headers()
        ) as response:
            result = await response.json()
            return self._parse_response(result)
```

### 使用示例

```python
import asyncio

async def main():
    chain = prompt | model | parser

    # 异步执行
    result = await chain.ainvoke({"topic": "AI"})
    print(result)

    # 并发执行多个请求
    results = await asyncio.gather(
        chain.ainvoke({"topic": "AI"}),
        chain.ainvoke({"topic": "ML"}),
        chain.ainvoke({"topic": "DL"})
    )
    print(results)

# 运行
asyncio.run(main())
```

### 性能考虑

- **原生异步**: 如果底层 API 支持异步，性能优于线程池
- **事件循环**: 单线程事件循环，避免 GIL 限制
- **并发能力**: 可轻松处理数千并发请求

---

## API-03: batch - 批量同步执行

### 基本信息

- **方法名**: `batch`
- **方法签名**: `def batch(self, inputs: list[Input], config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None, *, return_exceptions: bool = False, **kwargs: Any) -> list[Output]`
- **所属类**: `Runnable[Input, Output]`

### 功能说明

批量执行多个输入，使用线程池并发处理以提升吞吐量。默认实现会并行调用 `invoke`。

### 参数说明

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `inputs` | `list[Input]` | 是 | - | 输入列表 |
| `config` | `Optional[Union[RunnableConfig, list[RunnableConfig]]]` | 否 | `None` | 统一配置或每个输入的配置列表 |
| `return_exceptions` | `bool` | 否 | `False` | 是否返回异常而非抛出 |
| `**kwargs` | `Any` | 否 | - | 额外参数 |

### 返回值

| 类型 | 说明 |
|------|------|
| `list[Output]` 或 `list[Union[Output, Exception]]` | 输出列表，顺序与输入对应。如果 `return_exceptions=True`，失败的输入返回异常对象 |

### 核心代码

```python
def batch(
    self,
    inputs: list[Input],
    config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[Output]:
    """
    批量执行，使用线程池并发
    """
    if not inputs:
        return []

    # 将配置扩展为列表（每个输入一个配置）
    configs = get_config_list(config, len(inputs))

    # 定义单个输入的执行函数
    def invoke_single(
        input_: Input,
        config_: RunnableConfig
    ) -> Union[Output, Exception]:
        if return_exceptions:
            try:
                return self.invoke(input_, config_, **kwargs)
            except Exception as e:
                return e
        else:
            return self.invoke(input_, config_, **kwargs)

    # 单输入优化：不使用线程池
    if len(inputs) == 1:
        return [invoke_single(inputs[0], configs[0])]

    # 使用线程池并发执行
    with get_executor_for_config(configs[0]) as executor:
        return list(executor.map(invoke_single, inputs, configs))
```

**线程池配置**:

```python
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

@contextmanager
def get_executor_for_config(config: Optional[RunnableConfig]):
    """
    获取线程池执行器

    max_workers 由 max_concurrency 配置决定
    """
    max_concurrency = config.get("max_concurrency") if config else None

    if max_concurrency:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            yield executor
    else:
        # 使用默认线程数（通常为 CPU 核心数 * 5）
        with ThreadPoolExecutor() as executor:
            yield executor
```

### 使用示例

**基础批处理**:

```python
chain = prompt | model | parser

# 批量执行
inputs = [
    {"topic": "Python"},
    {"topic": "JavaScript"},
    {"topic": "Go"}
]

results = chain.batch(inputs)
for i, result in enumerate(results):
    print(f"Result {i}: {result}")
```

**限制并发数**:

```python
# 最多同时执行 5 个
results = chain.batch(
    inputs,
    config={"max_concurrency": 5}
)
```

**处理异常**:

```python
# 返回异常而非抛出
results = chain.batch(
    inputs,
    return_exceptions=True
)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Input {i} failed: {result}")
    else:
        print(f"Input {i} succeeded: {result}")
```

**每个输入不同配置**:

```python
configs = [
    {"tags": ["user1"]},
    {"tags": ["user2"]},
    {"tags": ["user3"]}
]

results = chain.batch(inputs, config=configs)
```

### 性能考虑

- **并发度**: 默认无限制，可能导致资源耗尽
- **建议**: 设置 `max_concurrency` 为合理值（10-50）
- **延迟**: 总延迟 ≈ max(各输入延迟) + 线程切换开销
- **吞吐量**: 相比顺序执行提升 N 倍（N 为并发数，受限于 I/O）

---

## API-04: stream - 流式同步输出

### 基本信息

- **方法名**: `stream`
- **方法签名**: `def stream(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Iterator[Output]`
- **所属类**: `Runnable[Input, Output]`

### 功能说明

逐块生成输出，而非等待全部完成。适用于需要渐进式展示结果的场景（如聊天应用）。

### 参数说明

与 `invoke` 相同。

### 返回值

| 类型 | 说明 |
|------|------|
| `Iterator[Output]` | 输出块的迭代器 |

### 核心代码

```python
def stream(
    self,
    input: Input,
    config: Optional[RunnableConfig] = None,
    **kwargs: Any,
) -> Iterator[Output]:
    """
    流式输出，基于 transform 方法实现
    """
    # 创建单元素迭代器
    def input_iterator():
        yield input

    # 调用 transform
    for chunk in self.transform(input_iterator(), config, **kwargs):
        yield chunk
```

**transform 方法（核心）**:

```python
def transform(
    self,
    input: Iterator[Input],
    config: Optional[RunnableConfig] = None,
    **kwargs: Any,
) -> Iterator[Output]:
    """
    流式转换：输入流 -> 输出流

    默认实现：对每个输入调用 invoke（非真正流式）
    子类应重写此方法实现真正的流式处理
    """
    for input_chunk in input:
        yield self.invoke(input_chunk, config, **kwargs)
```

**真正的流式实现示例（ChatModel）**:

```python
def _stream(
    self,
    messages: list[BaseMessage],
    stop: Optional[list[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    """
    流式生成，逐 token 返回
    """
    # 建立流式连接
    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=self._convert_messages(messages),
        stream=True,  # 启用流式
        **kwargs
    )

    # 逐块yield
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=chunk.choices[0].delta.content
                )
            )

            # 触发流式回调
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.choices[0].delta.content
                )
```

### 使用示例

**基础流式输出**:

```python
chain = prompt | model | parser

# 逐块打印
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

**收集所有块**:

```python
chunks = list(chain.stream({"topic": "AI"}))
full_output = "".join(chunks)
```

**异步流式输出**:

```python
async def stream_output():
    async for chunk in chain.astream({"topic": "AI"}):
        print(chunk, end="", flush=True)

asyncio.run(stream_output())
```

### 流式限制

**阻塞组件**:

```python
# ❌ RunnableLambda 不支持流式，会阻塞
def blocking_function(x):
    time.sleep(5)  # 模拟耗时操作
    return x.upper()

chain = model | RunnableLambda(blocking_function)

# 流式输出会在 blocking_function 处阻塞 5 秒
for chunk in chain.stream(input):
    print(chunk)  # 5 秒后一次性输出
```

**解决方案**:

```python
# ✅ 自定义 Runnable 支持流式
class StreamableUpperCase(Runnable[str, str]):
    def transform(self, input: Iterator[str], config: RunnableConfig = None):
        for chunk in input:
            yield chunk.upper()

chain = model | StreamableUpperCase()
# 真正逐块输出
```

### 性能考虑

- **首字节延迟**: 第一个组件的首字节延迟
- **总延迟**: 可能略高于 `invoke`（迭代器开销）
- **用户体验**: 显著提升（渐进式展示）
- **内存**: 更低（逐块处理，不累积完整输出）

---

## API-05: | 操作符 - 链式组合

### 基本信息

- **操作符**: `|`（管道操作符）
- **实现方法**: `__or__` 和 `__ror__`
- **返回类型**: `RunnableSequence[Input, Output]`

### 功能说明

LCEL 的核心语法，用于将多个 Runnable 串联成序列。左侧 Runnable 的输出作为右侧 Runnable 的输入。

### 核心代码

```python
def __or__(
    self,
    other: Union[
        Runnable[Any, Other],
        Callable[[Any], Other],
        Mapping[str, Any]
    ],
) -> RunnableSerializable[Input, Other]:
    """
    实现 self | other

    返回 RunnableSequence(self, other)
    """
    # 如果 other 也是 RunnableSequence，展平避免嵌套
    if isinstance(other, RunnableSequence):
        return RunnableSequence(
            self,
            *other.steps,
            name=self.name
        )

    # 将 other 转换为 Runnable
    return RunnableSequence(
        self,
        coerce_to_runnable(other),
        name=self.name
    )

def __ror__(
    self,
    other: Union[
        Runnable[Other, Any],
        Callable[[Other], Any],
        Mapping[str, Any]
    ],
) -> RunnableSerializable[Other, Output]:
    """
    实现 other | self（反向操作）
    """
    if isinstance(other, RunnableSequence):
        return RunnableSequence(
            *other.steps,
            self,
            name=self.name
        )

    return RunnableSequence(
        coerce_to_runnable(other),
        self,
        name=self.name
    )
```

**coerce_to_runnable 函数**:

```python
def coerce_to_runnable(thing: Any) -> Runnable:
    """
    将类 Runnable 对象转换为真正的 Runnable
    """
    # 已经是 Runnable
    if isinstance(thing, Runnable):
        return thing

    # 字典 -> RunnableParallel
    if isinstance(thing, dict):
        return RunnableParallel(thing)

    # 可调用对象 -> RunnableLambda
    if callable(thing):
        return RunnableLambda(thing)

    raise TypeError(f"Cannot coerce {type(thing)} to Runnable")
```

### 使用示例

**基础链式组合**:

```python
# 三个组件串联
chain = prompt | model | parser

# 等价于
chain = RunnableSequence(prompt, model, parser)

# 等价于
chain = prompt.pipe(model).pipe(parser)
```

**类型推断**:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

# 类型自动推断
prompt: Runnable[dict, list[BaseMessage]]  # dict -> messages
model: Runnable[list[BaseMessage], BaseMessage]  # messages -> message
parser: Runnable[BaseMessage, dict]  # message -> dict

# 组合后的类型
chain: Runnable[dict, dict]  # dict -> dict
```

**混合组合**:

```python
# 字典自动转换为 RunnableParallel
chain = prompt | model | {
    "summary": summary_parser,
    "keywords": keyword_parser
}

# 等价于
chain = (
    prompt
    | model
    | RunnableParallel(summary=summary_parser, keywords=keyword_parser)
)
```

**函数自动包装**:

```python
# 函数自动转换为 RunnableLambda
chain = prompt | model | (lambda x: x.content.upper())

# 等价于
chain = prompt | model | RunnableLambda(lambda x: x.content.upper())
```

---

## API-06: RunnableParallel - 并行执行

### 基本信息

- **类名**: `RunnableParallel[Input, dict[str, Any]]`
- **构造方法**:
  - `RunnableParallel(steps: Mapping[str, Runnable])`
  - `RunnableParallel(**kwargs: Runnable)`
  - 字典字面量（LCEL 语法糖）

### 功能说明

并发执行多个 Runnable，每个分支接收相同的输入，返回字典形式的输出（键为分支名，值为分支输出）。

### 参数说明

**构造参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `steps` | `Mapping[str, Runnable]` | 是 | 分支字典，键为分支名，值为 Runnable |
| `**kwargs` | `Runnable` | 否 | 关键字参数形式的分支 |

**执行参数**: 与 `invoke` 相同

### 返回值

| 类型 | 说明 |
|------|------|
| `dict[str, Any]` | 字典，键为分支名，值为各分支的输出 |

### 核心代码

```python
class RunnableParallel(RunnableSerializable[Input, dict[str, Any]]):
    """
    并行执行多个 Runnable
    """
    steps__: Mapping[str, Runnable[Input, Any]]

    def __init__(
        self,
        steps__: Optional[Mapping[str, Runnable[Input, Any]]] = None,
        **kwargs: Runnable[Input, Any]
    ):
        # 合并两种构造方式
        if steps__ is None:
            steps__ = kwargs
        elif kwargs:
            steps__ = {**steps__, **kwargs}

        super().__init__(steps__=steps__)

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        同步并行执行（使用线程池）
        """
        config = ensure_config(config)

        # 为每个分支创建子配置
        from copy import deepcopy
        step_configs = {
            key: patch_config(
                config,
                callbacks=callback_manager.get_child(f"map:key:{key}")
            )
            for key in self.steps__
        }

        # 使用线程池并发执行
        with get_executor_for_config(config) as executor:
            # 提交所有任务
            futures = {
                key: executor.submit(
                    step.invoke,
                    input,
                    step_configs[key],
                    **kwargs
                )
                for key, step in self.steps__.items()
            }

            # 收集结果
            return {
                key: future.result()
                for key, future in futures.items()
            }

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        异步并行执行（使用 asyncio.gather）
        """
        config = ensure_config(config)

        # 为每个分支创建协程
        coros = {
            key: step.ainvoke(input, step_configs[key], **kwargs)
            for key, step in self.steps__.items()
        }

        # 并发执行所有协程
        results = await asyncio.gather(*coros.values())

        # 组装结果字典
        return dict(zip(coros.keys(), results))
```

### 使用示例

**基础并行执行**:

```python
# 同时执行翻译和摘要
parallel = RunnableParallel(
    translation=translation_chain,
    summary=summary_chain
)

result = parallel.invoke({"text": "..."})
# 输出: {"translation": "...", "summary": "..."}
```

**LCEL 语法糖**:

```python
# 字典字面量自动转换为 RunnableParallel
chain = retriever | {
    "context": format_docs,
    "question": RunnablePassthrough()
} | prompt | model

# 等价于
chain = (
    retriever
    | RunnableParallel(
        context=format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | model
)
```

**嵌套并行**:

```python
# 并行内部再并行
parallel = RunnableParallel(
    analysis={
        "sentiment": sentiment_chain,
        "entities": entity_chain
    },
    translation=translation_chain
)

result = parallel.invoke(input)
# 输出: {
#   "analysis": {"sentiment": "...", "entities": [...]},
#   "translation": "..."
# }
```

### 性能考虑

- **并发度**: 等于分支数量
- **延迟**: max(各分支延迟) + 线程/协程切换开销
- **吞吐量**: 相比顺序执行提升显著
- **资源**: 同时占用多份资源（内存、连接等）

---

## API-07: with_retry - 添加重试策略

### 基本信息

- **方法名**: `with_retry`
- **方法签名**: `def with_retry(self, *, retry_if_exception_type: tuple[type[BaseException], ...] = (Exception,), wait_exponential_jitter: bool = True, stop_after_attempt: int = 3, **kwargs) -> Runnable[Input, Output]`
- **返回类型**: `RunnableRetry[Input, Output]`

### 功能说明

返回一个包装后的 Runnable，在执行失败时自动重试。支持指数退避、抖动等策略。

### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `retry_if_exception_type` | `tuple[type[BaseException], ...]` | `(Exception,)` | 哪些异常类型会触发重试 |
| `wait_exponential_jitter` | `bool` | `True` | 是否使用指数退避 + 抖动 |
| `stop_after_attempt` | `int` | `3` | 最大重试次数（包含首次） |
| `wait_exponential_multiplier` | `float` | `1.0` | 指数退避乘数 |
| `wait_exponential_max` | `float` | `10.0` | 最大等待时间（秒） |

### 使用示例

**基础重试**:

```python
# 失败时最多重试 3 次
chain_with_retry = chain.with_retry(stop_after_attempt=3)

try:
    result = chain_with_retry.invoke(input)
except Exception as e:
    print(f"Failed after 3 attempts: {e}")
```

**自定义重试条件**:

```python
from requests.exceptions import Timeout, ConnectionError

# 仅对网络错误重试
chain_with_retry = chain.with_retry(
    retry_if_exception_type=(Timeout, ConnectionError),
    stop_after_attempt=5,
    wait_exponential_jitter=True
)
```

**配置退避策略**:

```python
chain_with_retry = chain.with_retry(
    stop_after_attempt=5,
    wait_exponential_multiplier=2.0,  # 2^n 秒
    wait_exponential_max=60.0  # 最多等待 60 秒
)
```

由于篇幅限制，我将继续生成后续的 API 文档部分。API 文档还包含 `with_fallbacks`、`@chain` 装饰器等重要 API。

**要继续生成完整的 Runnables API 文档吗？还是先生成其他模块的文档？**

我建议采用以下策略以高效完成任务：

1. **每个模块生成"概览"文档**（包含架构图、核心概念）
2. **关键 API 在概览中简要说明，详细 API 文档可以按需生成**
3. **重点生成核心模块的完整文档**（Runnables、Language Models、Agents）

您希望我：

- A. 继续完成 Runnables 的完整 API 文档（还有约 10 个 API）
- B. 先生成所有模块的概览文档，再补充详细 API
- C. 只生成最核心的 3-5 个模块的完整文档

请告诉我您的偏好！
