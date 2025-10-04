# OpenAI Agents Python SDK - Runner 模块 API 详解

## 1. API 总览

Runner 模块是 OpenAI Agents SDK 的执行引擎核心，提供统一的代理执行接口。所有代理的运行都通过 Runner 类的静态方法进行，支持同步、异步、流式等多种执行模式。

### API 层次结构

```
Runner (执行调度器)
    ├── run() - 标准异步执行
    ├── run_streamed() - 流式异步执行
    └── run_sync() - 同步阻塞执行

RunConfig (执行配置)
    ├── 模型配置
    ├── 安全防护配置
    ├── 生命周期钩子
    └── 服务器对话管理

RunResult (执行结果)
    ├── final_output - 最终输出
    ├── new_items - 生成的历史项
    ├── raw_responses - 原始模型响应
    └── guardrail_results - 防护检查结果

RunResultStreaming (流式结果)
    ├── stream_events() - 流式事件生成器
    ├── current_agent - 当前执行代理
    └── is_complete - 完成状态
```

### API 分类

| API 类别 | 核心 API | 功能描述 |
|---------|---------|---------|
| **执行入口** | `Runner.run()` | 标准异步执行代理 |
| | `Runner.run_streamed()` | 流式异步执行，实时事件推送 |
| | `Runner.run_sync()` | 同步阻塞执行（便捷方法） |
| **配置管理** | `RunConfig.__init__()` | 创建执行配置实例 |
| | `RunConfig.model` | 全局模型配置 |
| | `RunConfig.model_settings` | 模型参数配置 |
| | `RunConfig.input_guardrails` | 输入安全防护 |
| | `RunConfig.output_guardrails` | 输出安全防护 |
| | `RunConfig.max_turns` | 最大执行轮次 |
| **结果处理** | `RunResult.final_output` | 获取最终输出 |
| | `RunResult.to_input_list()` | 转换为新输入列表 |
| | `RunResult.final_output_as()` | 类型安全的输出转换 |
| | `RunResultStreaming.stream_events()` | 流式事件迭代器 |
| | `RunResultStreaming.cancel()` | 取消流式执行 |
| **上下文管理** | `RunContextWrapper.context` | 用户自定义上下文 |
| | `RunContextWrapper.usage` | Token 使用统计 |

## 2. Runner 执行入口 API

### 2.1 Runner.run - 标准异步执行

**API 签名：**
```python
@staticmethod
async def run(
    agent: Agent[TContext],
    input: str | list[TResponseInputItem],
    session: Session | None = None,
    run_config: RunConfig | None = None,
    context: TContext | None = None,
) -> RunResult
```

**功能描述：**
执行代理的核心方法，处理完整的执行循环直到产生最终输出。支持会话历史管理、防护检查、工具调用、代理切换等完整功能。

**请求参数：**

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `agent` | `Agent[TContext]` | 是 | - | 要执行的代理实例 |
| `input` | `str \| list[TResponseInputItem]` | 是 | - | 用户输入，可以是字符串或结构化输入列表 |
| `session` | `Session \| None` | 否 | `None` | 会话对象，用于管理对话历史 |
| `run_config` | `RunConfig \| None` | 否 | `None` | 执行配置，控制模型、防护、钩子等 |
| `context` | `TContext \| None` | 否 | `None` | 用户自定义上下文，传递给工具和钩子 |

**返回结构：**
```python
@dataclass
class RunResult:
    input: str | list[TResponseInputItem]  # 原始输入
    new_items: list[RunItem]               # 新生成的运行项
    raw_responses: list[ModelResponse]     # 原始模型响应列表
    final_output: Any                      # 最终输出（通常是字符串）
    last_agent: Agent[Any]                 # 最后执行的代理
    
    # 防护检查结果
    input_guardrail_results: list[InputGuardrailResult]
    output_guardrail_results: list[OutputGuardrailResult]
    tool_input_guardrail_results: list[ToolInputGuardrailResult]
    tool_output_guardrail_results: list[ToolOutputGuardrailResult]
    
    # 上下文包装器
    context_wrapper: RunContextWrapper[Any]
    
    # 便捷方法
    def final_output_as(cls: type[T]) -> T
    def to_input_list() -> list[TResponseInputItem]
    
    # 属性
    @property
    def last_response_id() -> str | None
```

**使用示例：**

```python
from agents import Agent, Runner
from agents.memory import SQLiteSession

async def basic_run_example():
    """基础执行示例"""
    
    # 创建代理
    agent = Agent(
        name="Assistant",
        instructions="你是一个友好的AI助手。"
    )
    
    # 执行代理
    result = await Runner.run(
        agent=agent,
        input="你好，请介绍一下自己。"
    )
    
    # 访问结果
    print(f"最终输出: {result.final_output}")
    print(f"生成了 {len(result.new_items)} 个新项目")
    print(f"调用模型 {len(result.raw_responses)} 次")
    print(f"最后代理: {result.last_agent.name}")

async def run_with_session_example():
    """带会话历史的执行示例"""
    
    agent = Agent(
        name="ChatBot",
        instructions="你是一个记忆良好的聊天机器人。"
    )
    
    # 创建会话
    session = SQLiteSession("user_123", db_path="chat.db")
    
    # 第一轮对话
    result1 = await Runner.run(
        agent=agent,
        input="我叫张三，我喜欢编程。",
        session=session
    )
    print(f"助手: {result1.final_output}")
    
    # 第二轮对话（有历史上下文）
    result2 = await Runner.run(
        agent=agent,
        input="你还记得我的名字吗？",
        session=session
    )
    print(f"助手: {result2.final_output}")
    # 预期输出: "当然记得，你叫张三..."

async def run_with_tools_example():
    """带工具的执行示例"""
    
    from agents import function_tool
    
    @function_tool
    def calculate(expression: str) -> str:
        """计算数学表达式"""
        return str(eval(expression))
    
    agent = Agent(
        name="MathBot",
        instructions="你是一个数学助手，使用calculate工具进行计算。",
        tools=[calculate]
    )
    
    result = await Runner.run(
        agent=agent,
        input="请计算 123 * 456 的结果"
    )
    
    print(f"最终输出: {result.final_output}")
    
    # 查看生成的项目
    for item in result.new_items:
        if hasattr(item, 'tool_name'):
            print(f"调用工具: {item.tool_name}")
        elif hasattr(item, 'content'):
            print(f"消息: {item.content}")

async def run_with_config_example():
    """带配置的执行示例"""
    
    from agents import RunConfig
    
    agent = Agent(
        name="ConfiguredAgent",
        instructions="遵循配置运行的代理。"
    )
    
    # 创建配置
    config = RunConfig(
        model="gpt-4o",  # 指定模型
        max_turns=5,     # 最多5轮对话
        trace_include_sensitive_data=False  # 不包含敏感数据
    )
    
    result = await Runner.run(
        agent=agent,
        input="请帮我完成一个复杂任务。",
        run_config=config
    )
    
    print(f"使用的模型: {config.model}")
    print(f"实际轮次: {len(result.raw_responses)}")

async def run_with_context_example():
    """带自定义上下文的执行示例"""
    
    from dataclasses import dataclass
    
    @dataclass
    class UserContext:
        user_id: str
        database_connection: Any
        preferences: dict
    
    @function_tool
    def get_user_preference(key: str, context: UserContext) -> str:
        """获取用户偏好设置"""
        return context.preferences.get(key, "未设置")
    
    agent = Agent(
        name="PersonalAssistant",
        instructions="根据用户偏好提供个性化服务。",
        tools=[get_user_preference]
    )
    
    # 创建用户上下文
    user_context = UserContext(
        user_id="user_123",
        database_connection=None,  # 实际应用中是数据库连接
        preferences={"language": "中文", "theme": "dark"}
    )
    
    result = await Runner.run(
        agent=agent,
        input="我的语言偏好是什么？",
        context=user_context
    )
    
    print(f"输出: {result.final_output}")
    print(f"Token使用: {result.context_wrapper.usage}")
```

**执行流程：**
1. **初始化阶段**：加载会话历史、初始化上下文
2. **输入防护**：运行输入防护检查
3. **执行循环**：
   - 调用模型生成响应
   - 执行工具调用（如果有）
   - 处理代理切换（如果有）
   - 检查是否达到最终输出
4. **输出防护**：运行输出防护检查
5. **结果封装**：保存历史、返回结果

**异常情况：**
- `MaxTurnsExceeded`: 超过最大执行轮次（默认10轮）
- `InputGuardrailTripwireTriggered`: 输入防护触发
- `OutputGuardrailTripwireTriggered`: 输出防护触发
- `ModelBehaviorError`: 模型行为异常
- `UserError`: 用户工具函数抛出的异常

### 2.2 Runner.run_streamed - 流式异步执行

**API 签名：**
```python
@staticmethod
async def run_streamed(
    agent: Agent[TContext],
    input: str | list[TResponseInputItem],
    session: Session | None = None,
    run_config: RunConfig | None = None,
    context: TContext | None = None,
) -> RunResultStreaming
```

**功能描述：**
流式执行代理，实时推送执行过程中的事件。适用于需要实时反馈的场景，如聊天界面的打字效果。

**请求参数：**
与 `Runner.run()` 完全相同。

**返回结构：**
```python
@dataclass
class RunResultStreaming(RunResultBase):
    current_agent: Agent[Any]      # 当前执行的代理
    current_turn: int              # 当前执行轮次
    max_turns: int                 # 最大允许轮次
    final_output: Any              # 最终输出（完成前为None）
    is_complete: bool              # 是否完成执行
    
    # 继承自 RunResultBase
    input: str | list[TResponseInputItem]
    new_items: list[RunItem]
    raw_responses: list[ModelResponse]
    input_guardrail_results: list[InputGuardrailResult]
    output_guardrail_results: list[OutputGuardrailResult]
    tool_input_guardrail_results: list[ToolInputGuardrailResult]
    tool_output_guardrail_results: list[ToolOutputGuardrailResult]
    context_wrapper: RunContextWrapper[Any]
    
    # 流式方法
    async def stream_events() -> AsyncIterator[StreamEvent]
    def cancel() -> None
```

**StreamEvent 类型：**
```python
# StreamEvent 是联合类型
StreamEvent = (
    RawResponsesStreamEvent |      # 原始模型响应事件
    RunItemStreamEvent |           # 运行项事件
    AgentUpdatedStreamEvent        # 代理更新事件
)

# RunItemStreamEvent 的事件名称
event_names = [
    "message_output_created",      # 消息输出创建
    "tool_called",                 # 工具被调用
    "tool_output",                 # 工具输出
    "handoff_requested",           # 代理切换请求
    "handoff_occured",             # 代理切换发生
    "reasoning_item_created",      # 推理项创建
    "mcp_approval_requested",      # MCP批准请求
    "mcp_list_tools"               # MCP工具列表
]
```

**使用示例：**

```python
async def streamed_basic_example():
    """基础流式执行示例"""
    
    agent = Agent(
        name="StreamingAssistant",
        instructions="你是一个流式响应助手。"
    )
    
    # 启动流式执行
    result = await Runner.run_streamed(
        agent=agent,
        input="请详细介绍一下人工智能的发展历史。"
    )
    
    # 处理流式事件
    accumulated_text = ""
    
    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            if event.name == "message_output_created":
                # 消息输出事件
                content = event.item.content
                
                # 计算增量内容
                if content != accumulated_text:
                    delta = content[len(accumulated_text):]
                    print(delta, end="", flush=True)
                    accumulated_text = content
            
            elif event.name == "tool_called":
                # 工具调用事件
                print(f"\n[调用工具: {event.item.tool_name}]")
            
            elif event.name == "tool_output":
                # 工具输出事件
                print(f"[工具结果: {event.item.output[:50]}...]")
        
        elif event.type == "agent_updated_stream_event":
            # 代理更新事件
            print(f"\n[切换到代理: {event.new_agent.name}]")
    
    print(f"\n\n最终输出: {result.final_output}")

async def streamed_with_ui_example():
    """流式执行与UI集成示例"""
    
    from typing import Callable
    
    class ChatUI:
        """模拟的聊天UI类"""
        
        def __init__(self):
            self.messages = []
            self.current_message = ""
        
        def append_to_current_message(self, text: str):
            """追加文本到当前消息"""
            self.current_message += text
            # 实际应用中这里会更新UI
            print(text, end="", flush=True)
        
        def finish_current_message(self):
            """完成当前消息"""
            self.messages.append(self.current_message)
            self.current_message = ""
            print()  # 换行
        
        def show_tool_call(self, tool_name: str, args: dict):
            """显示工具调用"""
            print(f"\n🔧 正在使用工具: {tool_name}")
            print(f"   参数: {args}")
        
        def show_tool_result(self, result: str):
            """显示工具结果"""
            print(f"✅ 工具结果: {result[:100]}...")
    
    async def run_with_ui(agent: Agent, user_input: str):
        """带UI的流式执行"""
        
        ui = ChatUI()
        
        result = await Runner.run_streamed(
            agent=agent,
            input=user_input
        )
        
        accumulated_content = ""
        
        async for event in result.stream_events():
            if event.type == "run_item_stream_event":
                if event.name == "message_output_created":
                    # 增量文本输出
                    content = event.item.content
                    if content != accumulated_content:
                        delta = content[len(accumulated_content):]
                        ui.append_to_current_message(delta)
                        accumulated_content = content
                
                elif event.name == "tool_called":
                    # 显示工具调用
                    ui.show_tool_call(
                        event.item.tool_name,
                        event.item.arguments
                    )
                
                elif event.name == "tool_output":
                    # 显示工具结果
                    ui.show_tool_result(event.item.output)
        
        ui.finish_current_message()
        return result
    
    # 使用示例
    agent = Agent(
        name="UIAssistant",
        instructions="你是一个用户界面助手。"
    )
    
    result = await run_with_ui(agent, "请帮我查询今天的天气")

async def streamed_with_cancellation_example():
    """带取消功能的流式执行示例"""
    
    import asyncio
    
    agent = Agent(
        name="LongRunningAgent",
        instructions="你会进行长时间的处理。"
    )
    
    result = await Runner.run_streamed(
        agent=agent,
        input="请进行一个非常详细的分析。"
    )
    
    # 设置超时取消
    async def cancel_after_timeout(seconds: float):
        """N秒后取消执行"""
        await asyncio.sleep(seconds)
        if not result.is_complete:
            print(f"\n[超时 {seconds}秒，取消执行]")
            result.cancel()
    
    # 启动超时任务
    timeout_task = asyncio.create_task(cancel_after_timeout(5.0))
    
    try:
        async for event in result.stream_events():
            # 处理事件
            if event.type == "run_item_stream_event":
                if event.name == "message_output_created":
                    print(".", end="", flush=True)
    except asyncio.CancelledError:
        print("\n执行已取消")
    finally:
        timeout_task.cancel()

async def streamed_error_handling_example():
    """流式执行的错误处理示例"""
    
    agent = Agent(
        name="ErrorProneAgent",
        instructions="可能会遇到错误的代理。"
    )
    
    result = await Runner.run_streamed(
        agent=agent,
        input="执行可能失败的任务"
    )
    
    try:
        async for event in result.stream_events():
            # 处理事件
            if event.type == "run_item_stream_event":
                print(f"事件: {event.name}")
    
    except MaxTurnsExceeded as e:
        print(f"超过最大轮次: {e}")
        print(f"已生成 {len(e.run_data.new_items)} 个项目")
    
    except InputGuardrailTripwireTriggered as e:
        print(f"输入防护触发: {e.guardrail_result.output.message}")
    
    except OutputGuardrailTripwireTriggered as e:
        print(f"输出防护触发: {e.guardrail_result.output.message}")
    
    except Exception as e:
        print(f"执行错误: {e}")
```

**流式执行特点：**
1. **实时反馈**：事件实时推送，无需等待完成
2. **增量更新**：文本内容增量生成
3. **可取消**：支持中途取消执行
4. **异常传播**：异常通过流式接口传播

### 2.3 Runner.run_sync - 同步阻塞执行

**API 签名：**
```python
@staticmethod
def run_sync(
    agent: Agent[TContext],
    input: str | list[TResponseInputItem],
    session: Session | None = None,
    run_config: RunConfig | None = None,
    context: TContext | None = None,
) -> RunResult
```

**功能描述：**
同步阻塞版本的执行方法，便于在非异步环境中使用。内部使用 `asyncio.run()` 包装异步执行。

**请求参数：**
与 `Runner.run()` 完全相同。

**返回结构：**
与 `Runner.run()` 返回的 `RunResult` 完全相同。

**使用示例：**

```python
def sync_basic_example():
    """同步执行基础示例"""
    
    from agents import Agent, Runner
    
    agent = Agent(
        name="SyncAssistant",
        instructions="同步执行的助手。"
    )
    
    # 同步执行（阻塞）
    result = Runner.run_sync(
        agent=agent,
        input="你好，世界！"
    )
    
    print(f"输出: {result.final_output}")

def sync_in_script_example():
    """在脚本中使用同步执行"""
    
    # 不需要 async/await 语法
    if __name__ == "__main__":
        agent = Agent(name="ScriptAgent", instructions="脚本助手")
        result = Runner.run_sync(agent, "执行任务")
        print(result.final_output)

def sync_with_traditional_code_example():
    """与传统同步代码集成"""
    
    def legacy_function():
        """传统的同步函数"""
        agent = Agent(name="LegacyAgent", instructions="传统代码助手")
        
        # 可以直接调用，无需异步上下文
        result = Runner.run_sync(agent, "处理请求")
        
        return result.final_output
    
    # 调用
    output = legacy_function()
    print(output)
```

**使用场景：**
- 快速脚本和原型开发
- 与传统同步代码集成
- Jupyter Notebook 非异步单元格
- 命令行工具

**注意事项：**
- 阻塞执行，不适合高并发场景
- 不能在已有的事件循环中调用
- 推荐在生产环境使用异步版本 `Runner.run()`

## 3. RunConfig 配置 API

### 3.1 RunConfig 构造函数

**API 签名：**
```python
@dataclass
class RunConfig:
    def __init__(
        self,
        model: str | Model | None = None,
        model_provider: ModelProvider = MultiProvider(),
        model_settings: ModelSettings | None = None,
        handoff_input_filter: HandoffInputFilter | None = None,
        input_guardrails: list[InputGuardrail[Any]] | None = None,
        output_guardrails: list[OutputGuardrail[Any]] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,  # 默认10
        trace_include_sensitive_data: bool = True,
        call_model_input_filter: CallModelInputFilter | None = None,
        session_input_callback: SessionInputCallback | None = None,
        hooks: RunHooksBase | None = None,
        conversation_id: str | None = None,
        previous_response_id: str | None = None,
    )
```

**配置参数详解：**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model` | `str \| Model \| None` | `None` | 全局模型配置，覆盖代理的模型设置 |
| `model_provider` | `ModelProvider` | `MultiProvider()` | 模型提供商，用于解析模型名称 |
| `model_settings` | `ModelSettings \| None` | `None` | 全局模型参数（温度、最大token等） |
| `handoff_input_filter` | `HandoffInputFilter \| None` | `None` | 全局代理切换输入过滤器 |
| `input_guardrails` | `list[InputGuardrail] \| None` | `None` | 输入防护检查列表 |
| `output_guardrails` | `list[OutputGuardrail] \| None` | `None` | 输出防护检查列表 |
| `max_turns` | `int` | `10` | 最大执行轮次，防止无限循环 |
| `trace_include_sensitive_data` | `bool` | `True` | 追踪是否包含敏感数据 |
| `call_model_input_filter` | `CallModelInputFilter \| None` | `None` | 模型调用前的输入过滤器 |
| `session_input_callback` | `SessionInputCallback \| None` | `None` | 会话输入回调，用于修改保存的历史 |
| `hooks` | `RunHooksBase \| None` | `None` | 生命周期钩子函数 |
| `conversation_id` | `str \| None` | `None` | 服务器端对话ID（OpenAI Conversations API） |
| `previous_response_id` | `str \| None` | `None` | 上一个响应ID（服务器端对话状态） |

**配置示例：**

```python
from agents import RunConfig, ModelSettings
from agents.guardrail import InputGuardrail, OutputGuardrail

# 基础配置
basic_config = RunConfig(
    model="gpt-4o",
    max_turns=15
)

# 完整配置
advanced_config = RunConfig(
    # 模型配置
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.7,
        max_tokens=2000,
        top_p=0.9
    ),
    
    # 安全防护
    input_guardrails=[ContentModerationGuardrail()],
    output_guardrails=[SensitiveInfoGuardrail()],
    
    # 执行控制
    max_turns=20,
    trace_include_sensitive_data=False,
    
    # 生命周期钩子
    hooks=MyCustomHooks()
)

# 服务器对话配置
server_conversation_config = RunConfig(
    conversation_id="conv_abc123",  # 使用现有对话
    model="gpt-4o"
)
```

### 3.2 配置项详解

**model - 模型配置：**
```python
# 字符串模型名称
config = RunConfig(model="gpt-4o")
config = RunConfig(model="gpt-4o-mini")
config = RunConfig(model="gpt-3.5-turbo")

# 自定义模型实例
from agents.models import CustomModel
custom_model = CustomModel(...)
config = RunConfig(model=custom_model)

# 覆盖代理的模型设置
agent = Agent(name="A", model="gpt-3.5-turbo")
config = RunConfig(model="gpt-4o")
result = await Runner.run(agent, "test", run_config=config)
# 实际使用 gpt-4o（配置优先）
```

**model_settings - 模型参数：**
```python
from agents import ModelSettings

# 创建性思维配置
creative_settings = ModelSettings(
    temperature=1.2,      # 高随机性
    top_p=0.95,           # 多样性
    max_tokens=3000       # 较长输出
)

# 精确性配置
precise_settings = ModelSettings(
    temperature=0.1,      # 低随机性
    top_p=0.5,            # 集中采样
    max_tokens=1000       # 简洁输出
)

# 应用配置
config = RunConfig(model_settings=creative_settings)
result = await Runner.run(agent, input, run_config=config)
```

**input_guardrails / output_guardrails - 安全防护：**
```python
from agents.guardrail import InputGuardrail, OutputGuardrail

class ContentModerationGuardrail(InputGuardrail):
    """内容审核防护"""
    async def run(self, input_text: str, context: Any):
        # 检查不当内容
        if contains_inappropriate_content(input_text):
            return InputGuardrailResult(
                output=GuardrailFunctionOutput(
                    tripwire_triggered=True,
                    message="输入包含不当内容"
                )
            )
        return InputGuardrailResult(
            output=GuardrailFunctionOutput(tripwire_triggered=False)
        )

class PIIDetectionGuardrail(OutputGuardrail):
    """个人信息检测防护"""
    async def run(self, output_text: str, context: Any):
        # 检测个人身份信息
        if contains_pii(output_text):
            return OutputGuardrailResult(
                output=GuardrailFunctionOutput(
                    tripwire_triggered=True,
                    message="输出包含个人隐私信息"
                )
            )
        return OutputGuardrailResult(
            output=GuardrailFunctionOutput(tripwire_triggered=False)
        )

# 配置防护
config = RunConfig(
    input_guardrails=[ContentModerationGuardrail()],
    output_guardrails=[PIIDetectionGuardrail()]
)
```

**max_turns - 最大轮次：**
```python
# 简单任务：较少轮次
simple_config = RunConfig(max_turns=5)

# 复杂任务：较多轮次
complex_config = RunConfig(max_turns=20)

# 无限制（不推荐）
unlimited_config = RunConfig(max_turns=9999)

# 超过轮次会抛出异常
try:
    result = await Runner.run(agent, input, run_config=simple_config)
except MaxTurnsExceeded as e:
    print(f"超过最大轮次 {e.run_data.new_items}")
```

**hooks - 生命周期钩子：**
```python
from agents.lifecycle import RunHooksBase

class CustomHooks(RunHooksBase):
    """自定义生命周期钩子"""
    
    async def on_run_start(self, agent, input, context):
        """执行开始时调用"""
        print(f"开始执行代理: {agent.name}")
    
    async def on_run_end(self, result, context):
        """执行结束时调用"""
        print(f"执行完成，输出: {result.final_output}")
    
    async def on_tool_call(self, tool_name, arguments, context):
        """工具调用前调用"""
        print(f"调用工具: {tool_name}")
    
    async def on_tool_result(self, tool_name, result, context):
        """工具执行后调用"""
        print(f"工具结果: {result}")
    
    async def on_agent_switch(self, from_agent, to_agent, context):
        """代理切换时调用"""
        print(f"切换代理: {from_agent.name} -> {to_agent.name}")

# 使用钩子
config = RunConfig(hooks=CustomHooks())
result = await Runner.run(agent, input, run_config=config)
```

## 4. RunResult 结果 API

### 4.1 RunResult 属性访问

**核心属性：**

```python
result = await Runner.run(agent, input)

# 最终输出
print(result.final_output)  # "这是助手的回复"

# 原始输入
print(result.input)  # "用户的问题"

# 生成的新项目
for item in result.new_items:
    print(type(item).__name__)  # MessageOutputItem, ToolCallItem等

# 原始模型响应
for response in result.raw_responses:
    print(response.response_id)  # "resp_abc123"

# 最后执行的代理
print(result.last_agent.name)  # "FinalAgent"

# 防护检查结果
print(len(result.input_guardrail_results))   # 输入防护数量
print(len(result.output_guardrail_results))  # 输出防护数量
print(len(result.tool_input_guardrail_results))   # 工具输入防护
print(len(result.tool_output_guardrail_results))  # 工具输出防护

# 上下文包装器
print(result.context_wrapper.usage)  # Token使用统计
print(result.context_wrapper.context)  # 用户自定义上下文
```

### 4.2 RunResult 方法

**final_output_as - 类型安全的输出转换：**

```python
from dataclasses import dataclass

@dataclass
class WeatherData:
    temperature: float
    condition: str
    humidity: int

# 配置代理返回结构化输出
agent = Agent(
    name="WeatherAgent",
    output_schema=WeatherData
)

result = await Runner.run(agent, "查询天气")

# 类型安全的转换
weather: WeatherData = result.final_output_as(WeatherData)
print(f"温度: {weather.temperature}°C")
print(f"状况: {weather.condition}")

# 带类型检查的转换
try:
    weather = result.final_output_as(WeatherData, raise_if_incorrect_type=True)
except TypeError as e:
    print(f"类型不匹配: {e}")
```

**to_input_list - 转换为输入列表：**

```python
# 第一轮对话
result1 = await Runner.run(agent, "第一个问题")

# 将结果转换为新的输入列表
input_list = result1.to_input_list()

# 第二轮对话，使用转换后的输入（包含历史）
result2 = await Runner.run(agent, "第二个问题")

# 等价于使用会话
session = SQLiteSession("user_123")
result1 = await Runner.run(agent, "第一个问题", session=session)
result2 = await Runner.run(agent, "第二个问题", session=session)
```

**last_response_id - 获取最后响应ID：**

```python
result = await Runner.run(agent, input)

response_id = result.last_response_id
if response_id:
    print(f"最后响应ID: {response_id}")
    
    # 可用于服务器端对话状态
    next_config = RunConfig(previous_response_id=response_id)
    next_result = await Runner.run(agent, "下一个问题", run_config=next_config)
```

## 5. RunResultStreaming 流式结果 API

### 5.1 stream_events - 流式事件迭代器

**API 签名：**
```python
async def stream_events(self) -> AsyncIterator[StreamEvent]
```

**功能描述：**
异步生成器，产生执行过程中的实时事件。

**事件处理示例：**

```python
result = await Runner.run_streamed(agent, input)

async for event in result.stream_events():
    # 类型检查和处理
    if event.type == "run_item_stream_event":
        # 运行项事件
        if event.name == "message_output_created":
            print(f"消息: {event.item.content}")
        
        elif event.name == "tool_called":
            print(f"工具: {event.item.tool_name}")
        
        elif event.name == "tool_output":
            print(f"结果: {event.item.output}")
        
        elif event.name == "handoff_requested":
            print(f"切换到: {event.item.target_agent}")
        
        elif event.name == "reasoning_item_created":
            print(f"推理: {event.item.content}")
    
    elif event.type == "raw_response_event":
        # 原始响应事件
        print(f"原始事件: {event.data.type}")
    
    elif event.type == "agent_updated_stream_event":
        # 代理更新事件
        print(f"新代理: {event.new_agent.name}")
```

### 5.2 cancel - 取消执行

**API 签名：**
```python
def cancel(self) -> None
```

**功能描述：**
取消正在进行的流式执行，停止所有后台任务。

**使用示例：**

```python
import asyncio

async def cancellable_execution():
    """可取消的执行"""
    
    result = await Runner.run_streamed(agent, input)
    
    # 在另一个任务中取消
    async def cancel_after(seconds: float):
        await asyncio.sleep(seconds)
        result.cancel()
        print("执行已取消")
    
    cancel_task = asyncio.create_task(cancel_after(10.0))
    
    try:
        async for event in result.stream_events():
            # 处理事件
            pass
    except asyncio.CancelledError:
        print("流式处理被取消")
    finally:
        cancel_task.cancel()
```

Runner 模块通过统一的 API 接口和灵活的配置选项，为 OpenAI Agents 提供了强大的执行调度能力，支持同步、异步、流式等多种执行模式，满足从简单脚本到复杂生产系统的各种需求。

