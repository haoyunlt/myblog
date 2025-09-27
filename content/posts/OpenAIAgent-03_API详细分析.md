---
title: "OpenAI Agents Python SDK API详细分析"
date: 2025-09-28T00:47:17+08:00
draft: false
tags: ['Python', '源码分析', 'API']
categories: ['Python']
description: "OpenAI Agents Python SDK API详细分析的深入技术分析文档"
keywords: ['Python', '源码分析', 'API']
author: "技术分析师"
weight: 1
---

## 3.1 核心API概览

OpenAI Agents SDK 提供了一套简洁而强大的API，主要包含以下几个核心组件：

- **`Runner`**: 代理执行器，负责运行代理工作流
- **`Agent`**: 代理定义，配置代理的行为和能力  
- **`function_tool`**: 工具装饰器，将Python函数转换为代理工具
- **`handoff`**: 代理切换机制
- **各种配置类**: 用于详细配置代理行为

## 3.2 Runner类 API详细分析

### 3.2.1 Runner.run() 方法

**函数签名**：
```python
@classmethod
async def run(
    cls,
    starting_agent: Agent[TContext],                      # 起始代理
    input: str | list[TResponseInputItem],                # 输入内容
    *,
    context: TContext | None = None,                      # 运行上下文
    max_turns: int = DEFAULT_MAX_TURNS,                   # 最大轮次
    hooks: RunHooks[TContext] | None = None,              # 生命周期钩子
    run_config: RunConfig | None = None,                  # 运行配置
    previous_response_id: str | None = None,              # 前一个响应ID
    conversation_id: str | None = None,                   # 对话ID
    session: Session | None = None,                       # 会话管理
) -> RunResult:
```

**核心实现分析**：

```python
# 位于 src/agents/run.py:233
@classmethod
async def run(cls, starting_agent, input, **kwargs) -> RunResult:
    """
    执行代理工作流的主要入口方法
    
    执行流程:
    1. 准备输入和会话历史
    2. 初始化追踪和上下文
    3. 运行代理执行循环
    4. 处理工具调用和代理切换
    5. 返回最终结果
    """
    runner = DEFAULT_AGENT_RUNNER  # 获取默认运行器实例
    return await runner.run(starting_agent, input, **kwargs)
```

**实际执行逻辑在AgentRunner.run()中**：

```python
# 位于 src/agents/run.py:456
async def run(self, starting_agent: Agent[TContext], input, **kwargs) -> RunResult:
    """
    代理运行的核心实现逻辑
    """
    # 1. 参数处理和初始化
    context = kwargs.get("context")
    max_turns = kwargs.get("max_turns", DEFAULT_MAX_TURNS)
    hooks = self._validate_run_hooks(kwargs.get("hooks"))
    run_config = kwargs.get("run_config") or RunConfig()
    session = kwargs.get("session")
    
    # 2. 会话历史准备
    prepared_input = await self._prepare_input_with_session(
        input, session, run_config.session_input_callback
    )
    
    # 3. 初始化追踪和工具使用追踪器
    tool_use_tracker = AgentToolUseTracker()
    
    # 4. 开始追踪上下文
    with TraceCtxManager(
        workflow_name=run_config.workflow_name,
        trace_id=run_config.trace_id,
        group_id=run_config.group_id,
        metadata=run_config.trace_metadata,
        disabled=run_config.tracing_disabled,
    ):
        # 5. 主执行循环
        current_turn = 0
        current_agent = starting_agent
        generated_items: list[RunItem] = []
        model_responses: list[ModelResponse] = []
        
        while True:  # 代理执行循环
            current_turn += 1
            
            # 检查最大轮次限制
            if current_turn > max_turns:
                raise MaxTurnsExceeded(f"Max turns ({max_turns}) exceeded")
            
            # 获取当前代理的所有工具
            all_tools = await self._get_all_tools(current_agent, context_wrapper)
            
            # 6. 执行单轮对话
            turn_result = await self._run_single_turn(
                agent=current_agent,
                all_tools=all_tools,
                original_input=original_input,
                generated_items=generated_items,
                hooks=hooks,
                context_wrapper=context_wrapper,
                run_config=run_config,
                # ... 其他参数
            )
            
            # 7. 处理执行结果
            if isinstance(turn_result.next_step, NextStepFinalOutput):
                # 达到最终输出，执行输出防护并返回结果
                output_guardrail_results = await self._run_output_guardrails(...)
                return RunResult(
                    input=original_input,
                    new_items=generated_items,
                    final_output=turn_result.next_step.output,
                    # ... 其他字段
                )
            elif isinstance(turn_result.next_step, NextStepHandoff):
                # 代理切换，更新当前代理继续循环
                current_agent = turn_result.next_step.new_agent
            elif isinstance(turn_result.next_step, NextStepRunAgain):
                # 继续执行当前代理（通常在工具调用后）
                pass
```

**关键函数调用链路**：

1. **`Runner.run()`** → 
2. **`DEFAULT_AGENT_RUNNER.run()`** → 
3. **`AgentRunner._run_single_turn()`** → 
4. **`AgentRunner._get_new_response()`** → 
5. **`Model.get_response()`** → 
6. **`RunImpl.execute_tools_and_side_effects()`**

### 3.2.2 AgentRunner._run_single_turn() 方法详解

这是单轮代理执行的核心方法：

```python
# 位于 src/agents/run.py:1199
@classmethod
async def _run_single_turn(
    cls,
    *,
    agent: Agent[TContext],                    # 当前代理
    all_tools: list[Tool],                    # 可用工具列表
    original_input: str | list[TResponseInputItem],  # 原始输入
    generated_items: list[RunItem],           # 已生成的项目
    hooks: RunHooks[TContext],               # 生命周期钩子
    context_wrapper: RunContextWrapper[TContext],  # 上下文包装器
    run_config: RunConfig,                   # 运行配置
    # ... 其他参数
) -> SingleStepResult:
    """
    执行代理的单轮交互
    
    流程:
    1. 触发代理开始钩子
    2. 获取系统提示和提示配置
    3. 调用模型获取响应
    4. 处理模型响应（工具调用、代理切换、最终输出）
    5. 返回单步执行结果
    """
    
    # 1. 执行代理开始钩子
    if should_run_agent_start_hooks:
        await asyncio.gather(
            hooks.on_agent_start(context_wrapper, agent),
            (agent.hooks.on_start(context_wrapper, agent) 
             if agent.hooks else _coro.noop_coroutine()),
        )

    # 2. 获取系统提示和提示配置
    system_prompt, prompt_config = await asyncio.gather(
        agent.get_system_prompt(context_wrapper),  # 获取系统提示
        agent.get_prompt(context_wrapper),         # 获取提示配置
    )

    # 3. 准备输入数据
    output_schema = cls._get_output_schema(agent)
    handoffs = await cls._get_handoffs(agent, context_wrapper)
    input = ItemHelpers.input_to_new_input_list(original_input)
    input.extend([item.to_input_item() for item in generated_items])

    # 4. 调用模型获取新响应
    new_response = await cls._get_new_response(
        agent=agent,
        system_prompt=system_prompt,
        input=input,
        output_schema=output_schema,
        all_tools=all_tools,
        handoffs=handoffs,
        hooks=hooks,
        context_wrapper=context_wrapper,
        run_config=run_config,
        tool_use_tracker=tool_use_tracker,
        previous_response_id=previous_response_id,
        conversation_id=conversation_id,
        prompt_config=prompt_config,
    )

    # 5. 处理模型响应并返回结果
    return await cls._get_single_step_result_from_response(
        agent=agent,
        original_input=original_input,
        pre_step_items=generated_items,
        new_response=new_response,
        output_schema=output_schema,
        all_tools=all_tools,
        handoffs=handoffs,
        hooks=hooks,
        context_wrapper=context_wrapper,
        run_config=run_config,
        tool_use_tracker=tool_use_tracker,
    )
```

### 3.2.3 模型调用核心方法

**`AgentRunner._get_new_response()`** 负责实际的模型调用：

```python
# 位于 src/agents/run.py:1439
@classmethod
async def _get_new_response(
    cls,
    agent: Agent[TContext],
    system_prompt: str | None,
    input: list[TResponseInputItem],
    output_schema: AgentOutputSchemaBase | None,
    all_tools: list[Tool],
    handoffs: list[Handoff],
    hooks: RunHooks[TContext],
    context_wrapper: RunContextWrapper[TContext],
    run_config: RunConfig,
    tool_use_tracker: AgentToolUseTracker,
    previous_response_id: str | None,
    conversation_id: str | None,
    prompt_config: ResponsePromptParam | None,
) -> ModelResponse:
    """
    调用模型获取响应的核心方法
    """
    
    # 1. 应用输入过滤器（如果配置了）
    filtered = await cls._maybe_filter_model_input(
        agent=agent,
        run_config=run_config,
        context_wrapper=context_wrapper,
        input_items=input,
        system_instructions=system_prompt,
    )

    # 2. 获取模型实例和设置
    model = cls._get_model(agent, run_config)
    model_settings = agent.model_settings.resolve(run_config.model_settings)
    model_settings = RunImpl.maybe_reset_tool_choice(agent, tool_use_tracker, model_settings)

    # 3. 执行LLM开始钩子
    await asyncio.gather(
        hooks.on_llm_start(context_wrapper, agent, filtered.instructions, filtered.input),
        (agent.hooks.on_llm_start(context_wrapper, agent, filtered.instructions, filtered.input)
         if agent.hooks else _coro.noop_coroutine()),
    )

    # 4. 调用模型获取响应
    new_response = await model.get_response(
        system_instructions=filtered.instructions,  # 系统指令
        input=filtered.input,                       # 输入消息列表
        model_settings=model_settings,              # 模型设置
        tools=all_tools,                           # 可用工具
        output_schema=output_schema,               # 输出模式
        handoffs=handoffs,                         # 代理切换选项
        tracing=get_model_tracing_impl(...),       # 追踪设置
        previous_response_id=previous_response_id, # 前一个响应ID
        conversation_id=conversation_id,           # 对话ID
        prompt=prompt_config,                      # 提示配置
    )

    # 5. 更新使用统计
    context_wrapper.usage.add(new_response.usage)

    # 6. 执行LLM结束钩子
    await asyncio.gather(
        (agent.hooks.on_llm_end(context_wrapper, agent, new_response)
         if agent.hooks else _coro.noop_coroutine()),
        hooks.on_llm_end(context_wrapper, agent, new_response),
    )

    return new_response
```

## 3.3 Agent类 API详细分析

### 3.3.1 Agent类定义和字段

```python
# 位于 src/agents/agent.py:133
@dataclass
class Agent(AgentBase, Generic[TContext]):
    """
    AI代理的核心定义类
    
    包含代理的所有配置信息：指令、工具、模型设置、
    安全防护、代理切换等
    """
    
    # 核心配置
    name: str                                           # 代理名称，必填
    instructions: (                                     # 代理指令（系统提示）
        str | 
        Callable[[RunContextWrapper[TContext], Agent[TContext]], MaybeAwaitable[str]] | 
        None
    ) = None
    
    # 提示和工具配置  
    prompt: Prompt | DynamicPromptFunction | None = None              # 动态提示配置
    tools: list[Tool] = field(default_factory=list)                  # 工具列表
    handoffs: list[Agent[Any] | Handoff[TContext, Any]] = field(default_factory=list)  # 代理切换
    
    # 模型配置
    model: str | Model | None = None                                  # 模型名称或实例
    model_settings: ModelSettings = field(default_factory=get_default_model_settings)  # 模型设置
    
    # 安全防护
    input_guardrails: list[InputGuardrail[TContext]] = field(default_factory=list)   # 输入防护
    output_guardrails: list[OutputGuardrail[TContext]] = field(default_factory=list) # 输出防护
    
    # 输出和行为配置
    output_type: type[Any] | AgentOutputSchemaBase | None = None      # 输出类型定义
    tool_use_behavior: (                                             # 工具使用行为
        Literal["run_llm_again", "stop_on_first_tool"] | 
        StopAtTools | 
        ToolsToFinalOutputFunction
    ) = "run_llm_again"
    reset_tool_choice: bool = True                                   # 是否重置工具选择
    
    # 生命周期钩子
    hooks: AgentHooks[TContext] | None = None                        # 代理级别的钩子
```

### 3.3.2 Agent关键方法分析

**`get_system_prompt()`** - 获取系统提示：

```python
# 位于 src/agents/agent.py:443
async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
    """
    获取代理的系统提示
    
    支持三种类型的指令:
    1. 字符串类型 - 直接返回
    2. 可调用类型 - 动态生成指令
    3. None - 返回None
    """
    if isinstance(self.instructions, str):
        # 静态字符串指令
        return self.instructions
        
    elif callable(self.instructions):
        # 动态指令函数
        sig = inspect.signature(self.instructions)
        params = list(sig.parameters.values())
        
        # 强制要求函数接受两个参数：(context, agent)
        if len(params) != 2:
            raise TypeError(
                f"'instructions' callable must accept exactly 2 arguments "
                f"(context, agent), but got {len(params)}: {[p.name for p in params]}"
            )
        
        # 调用指令函数
        if inspect.iscoroutinefunction(self.instructions):
            return await cast(Awaitable[str], self.instructions(run_context, self))
        else:
            return cast(str, self.instructions(run_context, self))
            
    elif self.instructions is not None:
        # 无效的指令类型
        logger.error(
            f"Instructions must be a string or a callable function, "
            f"got {type(self.instructions).__name__}"
        )
        
    return None
```

**`get_all_tools()`** - 获取所有可用工具：

```python
# 位于 src/agents/agent.py:111
async def get_all_tools(self, run_context: RunContextWrapper[TContext]) -> list[Tool]:
    """
    获取代理的所有工具，包括MCP工具和函数工具
    """
    # 1. 获取MCP工具
    mcp_tools = await self.get_mcp_tools(run_context)
    
    # 2. 检查函数工具的启用状态
    async def _check_tool_enabled(tool: Tool) -> bool:
        if not isinstance(tool, FunctionTool):
            return True
            
        # 获取工具的启用属性
        attr = tool.is_enabled
        if isinstance(attr, bool):
            return attr
            
        # 调用启用检查函数
        res = attr(run_context, self)
        if inspect.isawaitable(res):
            return bool(await res)
        return bool(res)
    
    # 3. 并行检查所有工具的启用状态
    results = await asyncio.gather(*(_check_tool_enabled(t) for t in self.tools))
    enabled: list[Tool] = [t for t, ok in zip(self.tools, results) if ok]
    
    # 4. 返回MCP工具 + 启用的函数工具
    return [*mcp_tools, *enabled]
```

**`clone()`** - 克隆代理：

```python
# 位于 src/agents/agent.py:367
def clone(self, **kwargs: Any) -> Agent[TContext]:
    """
    创建代理的副本，可以修改指定的参数
    
    注意：
    - 使用浅拷贝，共享引用对象
    - 可变属性如tools和handoffs只有在覆盖时才创建新列表
    - 要独立修改这些属性，需要传入新的列表
    
    示例:
        new_agent = agent.clone(
            instructions="新的指令",
            tools=[new_tool_1, new_tool_2]
        )
    """
    return dataclasses.replace(self, **kwargs)
```

**`as_tool()`** - 将代理转换为工具：

```python  
# 位于 src/agents/agent.py:382
def as_tool(
    self,
    tool_name: str | None,                    # 工具名称
    tool_description: str | None,             # 工具描述
    custom_output_extractor: Callable[[RunResult], Awaitable[str]] | None = None,  # 自定义输出提取器
    is_enabled: bool | Callable[[RunContextWrapper[Any], AgentBase[Any]], MaybeAwaitable[bool]] = True,  # 启用状态
    run_config: RunConfig | None = None,      # 运行配置
    max_turns: int | None = None,             # 最大轮次
    hooks: RunHooks[TContext] | None = None,  # 钩子
    previous_response_id: str | None = None,  # 前一个响应ID
    conversation_id: str | None = None,       # 对话ID
    session: Session | None = None,           # 会话
) -> Tool:
    """
    将代理转换为可被其他代理调用的工具
    
    与代理切换(handoffs)的区别:
    1. 在handoffs中，新代理接收完整的对话历史
    2. 在工具中，新代理只接收生成的输入，原代理继续对话
    """
    
    @function_tool(
        name_override=tool_name or _transforms.transform_string_function_style(self.name),
        description_override=tool_description or "",
        is_enabled=is_enabled,
    )
    async def run_agent(context: RunContextWrapper, input: str) -> str:
        """内部工具函数，执行代理并返回结果"""
        from .run import DEFAULT_MAX_TURNS, Runner
        
        resolved_max_turns = max_turns if max_turns is not None else DEFAULT_MAX_TURNS
        
        # 运行代理
        output = await Runner.run(
            starting_agent=self,
            input=input,
            context=context.context,
            run_config=run_config,
            max_turns=resolved_max_turns,
            hooks=hooks,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            session=session,
        )
        
        # 提取输出
        if custom_output_extractor:
            return await custom_output_extractor(output)
        
        return ItemHelpers.text_message_outputs(output.new_items)
    
    return run_agent
```

## 3.4 function_tool装饰器 API详细分析

### 3.4.1 function_tool装饰器定义

```python
# 位于 src/agents/tool.py - function_tool装饰器的多个重载定义

@overload
def function_tool(
    func: ToolFunction[ToolParams],  # 要装饰的函数
) -> FunctionTool: ...

@overload
def function_tool(
    *,
    name_override: str | None = None,              # 覆盖工具名称
    description_override: str | None = None,       # 覆盖工具描述
    strict_json_schema: bool = True,              # 严格JSON模式
    is_enabled: bool | Callable[[RunContextWrapper[Any], AgentBase], MaybeAwaitable[bool]] = True,  # 启用状态
    docstring_style: DocstringStyle = "google",   # 文档字符串风格
) -> Callable[[ToolFunction[ToolParams]], FunctionTool]: ...

def function_tool(
    func: ToolFunction[ToolParams] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    strict_json_schema: bool = True,
    is_enabled: bool | Callable[[RunContextWrapper[Any], AgentBase], MaybeAwaitable[bool]] = True,
    docstring_style: DocstringStyle = "google",
) -> FunctionTool | Callable[[ToolFunction[ToolParams]], FunctionTool]:
    """
    将Python函数转换为代理工具的装饰器
    
    支持三种函数签名模式:
    1. 无上下文函数: func(param1, param2, ...)
    2. 带运行上下文函数: func(context: RunContextWrapper, param1, param2, ...)  
    3. 带工具上下文函数: func(tool_context: ToolContext, param1, param2, ...)
    """
```

### 3.4.2 function_tool实现逻辑

```python
# 位于 src/agents/tool.py 的实现细节
def function_tool(func=None, **kwargs):
    """function_tool装饰器的具体实现"""
    
    def decorator(f: ToolFunction[ToolParams]) -> FunctionTool:
        # 1. 获取函数信息
        func_name = name_override or f.__name__
        func_signature = inspect.signature(f)
        
        # 2. 分析函数参数，确定上下文类型
        has_context, context_type = _analyze_function_parameters(func_signature)
        
        # 3. 生成工具描述和参数JSON模式  
        description = description_override or _extract_function_description(f, docstring_style)
        params_schema = function_schema(f, strict=strict_json_schema, docstring_style=docstring_style)
        
        # 4. 创建工具调用处理函数
        async def on_invoke_tool(tool_context: ToolContext[Any], args_json: str) -> Any:
            """处理工具调用的内部函数"""
            try:
                # 解析参数JSON
                if args_json.strip():
                    args = json.loads(args_json)
                else:
                    args = {}
                
                # 根据函数签名调用函数
                if has_context:
                    if context_type == "run_context":
                        result = f(tool_context.run_context, **args)
                    elif context_type == "tool_context": 
                        result = f(tool_context, **args)
                else:
                    result = f(**args)
                
                # 处理异步结果
                if inspect.isawaitable(result):
                    result = await result
                    
                return result
                
            except Exception as e:
                # 工具执行错误处理
                logger.error(f"Tool {func_name} execution failed: {e}")
                return f"Error executing tool {func_name}: {str(e)}"
        
        # 5. 创建FunctionTool实例
        return FunctionTool(
            name=func_name,
            description=description,
            params_json_schema=params_schema,
            on_invoke_tool=on_invoke_tool,
            strict_json_schema=strict_json_schema,
            is_enabled=is_enabled,
        )
    
    # 支持直接装饰和参数化装饰两种用法
    if func is not None:
        return decorator(func)
    else:
        return decorator
```

### 3.4.3 工具参数模式分析

framework支持三种不同的工具函数签名模式：

**模式1：无上下文函数**
```python
@function_tool
def simple_calculator(x: int, y: int, operation: str) -> int:
    """执行简单的数学计算
    
    Args:
        x: 第一个数字
        y: 第二个数字  
        operation: 操作类型 ('add', 'subtract', 'multiply', 'divide')
    """
    if operation == 'add':
        return x + y
    elif operation == 'subtract':
        return x - y
    # ...
```

**模式2：带运行上下文函数**
```python
@function_tool  
def context_aware_tool(
    context: RunContextWrapper, 
    message: str
) -> str:
    """需要访问运行上下文的工具
    
    Args:
        message: 要处理的消息
    """
    # 可以访问context.usage, context.context等
    current_usage = context.usage
    print(f"当前使用情况: {current_usage}")
    return f"处理消息: {message}"
```

**模式3：带工具上下文函数**  
```python
@function_tool
def advanced_tool(
    tool_context: ToolContext,
    data: str
) -> str:
    """需要访问完整工具上下文的高级工具
    
    Args:
        data: 要处理的数据
    """
    # 可以访问工具调用的详细信息
    tool_name = tool_context.tool_name
    call_id = tool_context.tool_call_id
    agent = tool_context.agent
    
    return f"工具 {tool_name} (ID: {call_id}) 处理数据: {data}"
```

## 3.5 核心数据结构API

### 3.5.1 RunResult类

```python
# 位于 src/agents/result.py
@dataclass  
class RunResult:
    """代理运行的结果"""
    
    input: str | list[TResponseInputItem]              # 原始输入
    new_items: list[RunItem]                          # 新生成的项目
    raw_responses: list[ModelResponse]                # 原始模型响应
    final_output: Any                                 # 最终输出
    _last_agent: Agent[Any]                          # 最后执行的代理
    input_guardrail_results: list[InputGuardrailResult]   # 输入防护结果
    output_guardrail_results: list[OutputGuardrailResult] # 输出防护结果
    context_wrapper: RunContextWrapper[Any]          # 上下文包装器
    
    def to_input_list(self) -> list[TResponseInputItem]:
        """将结果转换为可用于下次输入的列表"""
        input_list = ItemHelpers.input_to_new_input_list(self.input)
        input_list.extend([item.to_input_item() for item in self.new_items])
        return input_list
```

### 3.5.2 RunConfig类

```python
# 位于 src/agents/run.py:129
@dataclass
class RunConfig:
    """代理运行的全局配置"""
    
    model: str | Model | None = None                   # 全局模型覆盖
    model_provider: ModelProvider = field(default_factory=MultiProvider)  # 模型提供商
    model_settings: ModelSettings | None = None       # 全局模型设置
    handoff_input_filter: HandoffInputFilter | None = None  # 代理切换输入过滤器
    input_guardrails: list[InputGuardrail[Any]] | None = None    # 全局输入防护
    output_guardrails: list[OutputGuardrail[Any]] | None = None  # 全局输出防护
    tracing_disabled: bool = False                     # 禁用追踪
    trace_include_sensitive_data: bool = field(default_factory=_default_trace_include_sensitive_data)  # 追踪敏感数据
    workflow_name: str = "Agent workflow"              # 工作流名称
    trace_id: str | None = None                       # 自定义追踪ID
    group_id: str | None = None                       # 分组ID
    trace_metadata: dict[str, Any] | None = None      # 追踪元数据
    session_input_callback: SessionInputCallback | None = None  # 会话输入回调
    call_model_input_filter: CallModelInputFilter | None = None  # 模型调用输入过滤器
```

这些API构成了OpenAI Agents SDK的核心接口，为开发者提供了灵活而强大的多代理应用开发能力。通过这些API，可以轻松构建从简单的单代理助手到复杂的多代理协作工作流。
