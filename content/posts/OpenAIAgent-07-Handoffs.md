---
title: "OpenAIAgent-07-Handoffs"
date: 2025-10-04T21:26:31+08:00
draft: false
tags:
  - OpenAI Agent
  - 架构设计
  - 概览
  - 源码分析
categories:
  - OpenAIAgent
  - Python
series: "openai agent-source-analysis"
description: "OpenAIAgent 源码剖析 - 07-Handoffs"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true

---

# OpenAIAgent-07-Handoffs

## 模块概览

## 1. 模块职责与边界

Handoffs 模块是 OpenAI Agents Python SDK 的代理协作核心，负责实现智能代理之间的任务委派和流程转接。该模块通过工具化的方式将代理间交接封装为标准化的操作，支持复杂的多代理协作场景和分层任务处理。

### 核心职责

- **任务委派**：支持代理向其他专门代理委派特定任务
- **上下文传递**：管理代理间交接时的历史和状态信息传递
- **流程控制**：控制多代理协作的执行流程和生命周期
- **输入过滤**：提供灵活的输入数据过滤和转换机制
- **状态追踪**：追踪代理交接的执行状态和结果
- **钩子集成**：支持交接生命周期事件的钩子函数

### 交接机制体系

| 交接类型 | 实现方式 | 主要特点 | 适用场景 |
|----------|----------|----------|----------|
| 标准交接 | `Handoff` | 通用代理间任务委派 | 客服分流、专家咨询、任务分解 |
| 实时交接 | `RealtimeHandoff` | 实时代理间的即时切换 | 语音对话、实时协作 |
| 输入交接 | `OnHandoffWithInput` | 带参数的代理调用 | 参数化任务、数据传递 |
| 无参交接 | `OnHandoffWithoutInput` | 简单的代理切换 | 流程转换、状态切换 |

### 交接数据流转

| 数据类型 | 数据内容 | 处理方式 | 生命周期 |
|----------|----------|----------|----------|
| `input_history` | 历史对话记录 | 可过滤、可修改 | 传递到新代理 |
| `pre_handoff_items` | 交接前生成的项目 | 上下文保留 | 用于状态恢复 |
| `new_items` | 当前回合生成项目 | 包含交接触发信息 | 完整传递 |
| `run_context` | 执行上下文 | 运行时状态 | 连续传递 |

### 输入输出接口

**输入：**

- 源代理实例（`Agent`）
- 交接参数（JSON格式）
- 执行上下文（`RunContextWrapper`）
- 历史数据（`HandoffInputData`）

**输出：**

- 目标代理实例（`Agent`）  
- 交接结果信息
- 过滤后的输入数据
- 执行状态和追踪信息

### 上下游依赖关系

**上游调用者：**

- `RunImpl`：执行引擎中的交接处理逻辑
- `Agent`：代理配置中的交接规则定义
- `RealtimeSession`：实时会话中的代理切换

**下游依赖：**

- `exceptions`：交接异常和错误处理
- `tracing`：交接执行的追踪和监控
- `items`：交接过程中的数据项定义
- `run_context`：执行上下文的管理

## 2. 模块架构图

```mermaid
flowchart TB
    subgraph "Handoffs 代理交接模块"
        subgraph "核心交接类"
            HANDOFF[Handoff]
            HANDOFFINPUTDATA[HandoffInputData]
            HANDOFFINPUTFILTER[HandoffInputFilter]
        end
        
        subgraph "交接类型"
            STANDARDHANDOFF[标准交接]
            REALTIMEHANDOFF[实时交接]
            INPUTHANDOFF[带参数交接]
            NOINPUTHANDOFF[无参数交接]
        end
        
        subgraph "回调函数"
            ONHANDOFFWITHINPUT[OnHandoffWithInput]
            ONHANDOFFWITHOUTINPUT[OnHandoffWithoutInput]
            INVOKEHANDOFF[on_invoke_handoff]
        end
        
        subgraph "工厂函数"
            HANDOFFFACTORY[handoff() 函数]
            REALTIMEHANDOFFFACTORY[realtime_handoff() 函数]
        end
        
        subgraph "数据处理"
            JSONSCHEMA[input_json_schema]
            TYPEADAPTER[TypeAdapter 验证]
            INPUTVALIDATION[输入验证]
        end
        
        subgraph "执行控制"
            ENABLEDCHECK[is_enabled 检查]
            MULTIPLEHANDOFFHANDLING[多交接处理]
            TRANSFERMESSAGE[交接消息生成]
        end
        
        subgraph "生命周期管理"
            HANDOFFHOOKS[交接钩子]
            SPANHANDOFF[交接追踪]
            ERRORHANDLING[异常处理]
        end
    end
    
    subgraph "执行集成"
        RUNIMPL[RunImpl 执行引擎]
        AGENT[Agent 代理]
        REALTIMESESSION[RealtimeSession 实时会话]
        TOOLCALL[工具调用机制]
    end
    
    subgraph "支撑系统"
        TRACING[tracing 追踪系统]
        EXCEPTIONS[异常处理系统]
        ITEMS[items 数据项系统]
        RUNCONTEXT[run_context 上下文系统]
        PYDANTIC[Pydantic 验证系统]
    end
    
    HANDOFF --> HANDOFFINPUTDATA
    HANDOFF --> HANDOFFINPUTFILTER
    HANDOFF --> INVOKEHANDOFF
    
    STANDARDHANDOFF --> HANDOFF
    REALTIMEHANDOFF --> HANDOFF
    INPUTHANDOFF --> ONHANDOFFWITHINPUT
    NOINPUTHANDOFF --> ONHANDOFFWITHOUTINPUT
    
    HANDOFFFACTORY --> STANDARDHANDOFF
    HANDOFFFACTORY --> INPUTHANDOFF
    HANDOFFFACTORY --> NOINPUTHANDOFF
    REALTIMEHANDOFFFACTORY --> REALTIMEHANDOFF
    
    HANDOFF --> JSONSCHEMA
    HANDOFF --> TYPEADAPTER
    HANDOFF --> INPUTVALIDATION
    
    HANDOFF --> ENABLEDCHECK
    HANDOFF --> TRANSFERMESSAGE
    
    RUNIMPL --> HANDOFF
    RUNIMPL --> MULTIPLEHANDOFFHANDLING
    AGENT --> HANDOFF
    REALTIMESESSION --> REALTIMEHANDOFF
    
    HANDOFF --> TOOLCALL
    
    HANDOFF --> HANDOFFHOOKS
    HANDOFF --> SPANHANDOFF
    HANDOFF --> ERRORHANDLING
    
    HANDOFF --> TRACING
    HANDOFF --> EXCEPTIONS
    HANDOFF --> ITEMS
    HANDOFF --> RUNCONTEXT
    HANDOFF --> PYDANTIC
    
    style HANDOFF fill:#e1f5fe
    style STANDARDHANDOFF fill:#f3e5f5  
    style REALTIMEHANDOFF fill:#e8f5e8
    style HANDOFFFACTORY fill:#fff3e0
    style RUNIMPL fill:#ffebee
```

**架构说明：**

### 分层设计原理

1. **核心交接层**：`Handoff` 类定义交接的基本结构和行为
2. **类型分化层**：不同类型的交接实现满足不同场景需求
3. **工厂方法层**：便利函数简化交接对象的创建过程
4. **执行集成层**：与代理执行引擎的深度集成

### 工具化设计

- **工具抽象**：交接通过工具调用的形式向LLM暴露
- **JSON Schema**：标准化的参数验证和类型安全
- **工具描述**：自动生成工具描述帮助LLM理解交接用途
- **工具命名**：基于代理名称的默认工具命名策略

### 数据流控制

- **输入过滤**：`HandoffInputFilter` 允许自定义输入数据处理
- **上下文传递**：完整的执行上下文在代理间传递
- **状态保持**：交接前后的状态信息得到妥善管理
- **历史管理**：灵活的对话历史传递和过滤机制

### 扩展能力设计

- **回调扩展**：支持自定义交接逻辑和副作用处理
- **类型安全**：通过泛型和类型适配器确保类型安全
- **条件交接**：`is_enabled` 机制支持动态交接启用/禁用
- **钩子集成**：完整的生命周期钩子支持

## 3. 关键算法与流程剖析

### 3.1 交接创建与配置算法

```python
def handoff(
    agent: Agent[TContext],
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    on_handoff: OnHandoffWithInput[THandoffInput] | OnHandoffWithoutInput | None = None,
    input_type: type[THandoffInput] | None = None,
    input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None,
    is_enabled: bool | Callable[[RunContextWrapper[Any], Agent[TContext]], MaybeAwaitable[bool]] = True,
) -> Handoff[TContext, Agent[TContext]]:
    """交接创建的核心算法"""
    
    # 1) 参数验证和类型适配器创建
    type_adapter: TypeAdapter[Any] | None
    if input_type is not None:
        # 验证回调函数签名
        assert callable(on_handoff)
        sig = inspect.signature(on_handoff)
        if len(sig.parameters) != 2:
            raise UserError("on_handoff must take two arguments: context and input")
        
        # 创建类型适配器和JSON Schema
        type_adapter = TypeAdapter(input_type)
        input_json_schema = type_adapter.json_schema()
    else:
        type_adapter = None
        input_json_schema = {}
        
        # 验证无参数回调函数
        if on_handoff is not None:
            sig = inspect.signature(on_handoff)
            if len(sig.parameters) != 1:
                raise UserError("on_handoff must take one argument: context")
    
    # 2) 创建交接调用函数
    async def _invoke_handoff(
        ctx: RunContextWrapper[Any], input_json: str | None = None
    ) -> Agent[TContext]:
        
        # 处理带参数的交接
        if input_type is not None and type_adapter is not None:
            if input_json is None:
                raise ModelBehaviorError("Handoff function expected non-null input, but got None")
            
            # JSON验证和类型转换
            validated_input = _json.validate_json(
                json_str=input_json,
                type_adapter=type_adapter,
                partial=False,
            )
            
            # 调用带参数的回调函数
            input_func = cast(OnHandoffWithInput[THandoffInput], on_handoff)
            if inspect.iscoroutinefunction(input_func):
                await input_func(ctx, validated_input)
            else:
                input_func(ctx, validated_input)
                
        # 处理无参数的交接
        elif on_handoff is not None:
            no_input_func = cast(OnHandoffWithoutInput, on_handoff)
            if inspect.iscoroutinefunction(no_input_func):
                await no_input_func(ctx)
            else:
                no_input_func(ctx)
        
        # 返回目标代理
        return agent
    
    # 3) 创建交接对象
    tool_name = tool_name_override or Handoff.default_tool_name(agent)
    tool_description = tool_description_override or Handoff.default_tool_description(agent)
    
    return Handoff(
        tool_name=tool_name,
        tool_description=tool_description,
        input_json_schema=input_json_schema,
        on_invoke_handoff=_invoke_handoff,
        agent_name=agent.name,
        input_filter=input_filter,
        is_enabled=is_enabled,
    )
```

**算法目的：** 创建类型安全、功能完整的代理交接对象，支持多种参数和配置选项。

**关键设计特点：**

1. **类型安全**：通过 `TypeAdapter` 确保输入参数的类型正确性
2. **签名验证**：检查回调函数的参数签名确保正确性
3. **异步支持**：同时支持同步和异步回调函数
4. **默认生成**：自动生成工具名称和描述

### 3.2 交接执行与多交接处理算法

```python
async def execute_handoffs(
    cls,
    *,
    agent: Agent[TContext],
    original_input: str | list[TResponseInputItem],
    pre_step_items: list[RunItem],
    new_step_items: list[RunItem],
    new_response: ModelResponse,
    run_handoffs: list[ToolRunHandoff],
    hooks: RunHooks[TContext],
    context_wrapper: RunContextWrapper[TContext],
    run_config: RunConfig,
) -> SingleStepResult:
    """交接执行的核心算法"""
    
    # 1) 多交接冲突处理
    multiple_handoffs = len(run_handoffs) > 1
    if multiple_handoffs:
        # 拒绝多余的交接请求
        rejected_handoffs = run_handoffs[1:]
        new_step_items.extend([
            ToolCallOutputItem(
                tool_call=rejected_handoff.tool_call,
                output="错误：不能同时执行多个代理交接",
            )
            for rejected_handoff in rejected_handoffs
        ])
    
    # 2) 执行有效交接
    actual_handoff = run_handoffs[0]
    with handoff_span(from_agent=agent.name) as span_handoff:
        handoff = actual_handoff.handoff
        
        # 调用交接函数获取新代理
        new_agent: Agent[Any] = await handoff.on_invoke_handoff(
            context_wrapper, actual_handoff.tool_call.arguments
        )
        
        # 更新追踪信息
        span_handoff.span_data.to_agent = new_agent.name
        if multiple_handoffs:
            requested_agents = [h.handoff.agent_name for h in run_handoffs]
            span_handoff.set_error(SpanError(
                message="Multiple handoffs requested",
                data={"requested_agents": requested_agents}
            ))
        
        # 3) 生成交接输出项
        new_step_items.append(
            HandoffOutputItem(
                agent=agent,
                raw_item=ItemHelpers.tool_call_output_item(
                    actual_handoff.tool_call,
                    handoff.get_transfer_message(new_agent),
                ),
                source_agent=agent,
                target_agent=new_agent,
            )
        )
        
        # 4) 执行交接钩子
        await asyncio.gather(
            hooks.on_handoff(
                context=context_wrapper,
                from_agent=agent,
                to_agent=new_agent,
            ),
            (
                agent.hooks.on_handoff(
                    context_wrapper,
                    agent=new_agent,
                    source=agent,
                )
                if agent.hooks
                else _coro.noop_coroutine()
            ),
        )
        
        # 5) 应用输入过滤器
        input_filter = handoff.input_filter or (
            run_config.handoff_input_filter if run_config else None
        )
        
        if input_filter:
            handoff_input_data = HandoffInputData(
                input_history=original_input,
                pre_handoff_items=tuple(pre_step_items),
                new_items=tuple(new_step_items),
                run_context=context_wrapper,
            )
            
            # 应用过滤器
            filtered_data = input_filter(handoff_input_data)
            if inspect.isawaitable(filtered_data):
                filtered_data = await filtered_data
                
            # 构建新的输入历史
            all_items = []
            if isinstance(filtered_data.input_history, str):
                all_items.append(ItemHelpers.user_input_item(filtered_data.input_history))
            else:
                all_items.extend(filtered_data.input_history)
            
            all_items.extend(filtered_data.pre_handoff_items)
            all_items.extend(filtered_data.new_items)
        else:
            # 默认传递所有历史
            all_items = ItemHelpers.collect_all_items(
                original_input, pre_step_items, new_step_items
            )
        
        # 6) 返回交接结果
        return SingleStepResult(
            new_agent=new_agent,
            new_step_items=new_step_items,
            all_items=all_items,
            handoff_executed=True,
        )
```

**算法目的：** 安全高效地执行代理交接，处理多交接冲突，管理数据传递和生命周期。

**执行策略分析：**

1. **冲突处理**：优先执行第一个交接，拒绝其他冲突交接
2. **追踪集成**：完整记录交接过程和相关错误信息
3. **钩子执行**：并行执行全局和代理级别的交接钩子
4. **数据过滤**：支持自定义输入数据过滤逻辑

### 3.3 输入过滤与数据传递算法

```python
@dataclass(frozen=True)
class HandoffInputData:
    """交接输入数据的结构化管理"""
    
    input_history: str | tuple[TResponseInputItem, ...]
    """交接前的输入历史"""
    
    pre_handoff_items: tuple[RunItem, ...]
    """交接前生成的项目"""
    
    new_items: tuple[RunItem, ...]
    """当前回合生成的新项目"""
    
    run_context: RunContextWrapper[Any] | None = None
    """运行时上下文"""
    
    def clone(self, **kwargs: Any) -> HandoffInputData:
        """创建修改后的副本"""
        return dataclasses_replace(self, **kwargs)

# 输入过滤器的实现示例
def context_aware_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """上下文感知的输入过滤器"""
    
    # 1) 历史长度控制
    MAX_HISTORY_ITEMS = 20
    if isinstance(handoff_data.input_history, tuple):
        filtered_history = handoff_data.input_history[-MAX_HISTORY_ITEMS:]
    else:
        # 字符串历史保持不变
        filtered_history = handoff_data.input_history
    
    # 2) 敏感信息过滤
    filtered_pre_items = []
    for item in handoff_data.pre_handoff_items:
        if not contains_sensitive_content(item):
            filtered_pre_items.append(item)
        else:
            # 替换为安全的占位符
            filtered_pre_items.append(create_placeholder_item(item))
    
    # 3) 工具调用结果过滤
    filtered_new_items = []
    for item in handoff_data.new_items:
        if item.type == "tool_call_output":
            # 过滤工具输出中的敏感信息
            filtered_item = filter_tool_output(item)
            filtered_new_items.append(filtered_item)
        else:
            filtered_new_items.append(item)
    
    # 4) 返回过滤后的数据
    return handoff_data.clone(
        input_history=filtered_history,
        pre_handoff_items=tuple(filtered_pre_items),
        new_items=tuple(filtered_new_items),
    )
```

**算法目的：** 提供灵活的输入数据过滤机制，支持上下文长度控制、敏感信息过滤等。

**过滤策略特点：**

1. **不可变设计**：通过 `frozen=True` 确保数据不可变性
2. **灵活修改**：`clone` 方法支持选择性字段修改
3. **层次过滤**：分别处理历史、预交接项目、新项目
4. **类型安全**：保持原有数据类型和结构

### 3.4 实时交接特殊处理算法

```python
async def _handle_tool_call(self, event: RealtimeModelToolCallEvent) -> None:
    """实时会话中的交接处理"""
    
    # 1) 检查是否为交接工具
    handoff_map = {h.tool_name: h for h in self._current_agent.handoffs}
    
    if event.name in handoff_map:
        handoff = handoff_map[event.name]
        
        # 2) 创建工具上下文
        tool_context = ToolContext(
            context=self._context_wrapper.context,
            usage=self._context_wrapper.usage,
            tool_name=event.name,
            tool_call_id=event.call_id,
            tool_arguments=event.arguments,
        )
        
        # 3) 执行交接获取新代理
        result = await handoff.on_invoke_handoff(self._context_wrapper, event.arguments)
        if not isinstance(result, RealtimeAgent):
            raise UserError(
                f"Handoff {handoff.tool_name} returned invalid result: {type(result)}"
            )
        
        # 4) 更新当前代理
        previous_agent = self._current_agent
        self._current_agent = result
        
        # 5) 获取新代理的模型设置
        updated_settings = await self._get_updated_model_settings_from_agent(
            starting_settings=None,
            agent=self._current_agent,
        )
        
        # 6) 发送交接事件
        await self._put_event(
            RealtimeHandoffEvent(
                from_agent=previous_agent,
                to_agent=self._current_agent,
                info=self._event_info,
            )
        )
        
        # 7) 更新会话设置
        await self._model.send_event(
            RealtimeModelSendSessionUpdate(session_settings=updated_settings)
        )
        
        # 8) 发送工具输出完成交接
        transfer_message = handoff.get_transfer_message(result)
        await self._model.send_event(
            RealtimeModelSendToolCallOutput(
                call_id=event.call_id,
                output=transfer_message,
            )
        )
```

**算法目的：** 处理实时会话中的代理交接，确保会话状态的连续性和一致性。

**实时交接特点：**

1. **即时切换**：立即更新当前代理，无需等待响应完成
2. **状态同步**：同步更新模型设置和会话配置
3. **事件驱动**：通过事件机制通知交接状态变化
4. **连续性保障**：确保实时对话的连续性

## 4. 数据结构与UML图

```mermaid
classDiagram
    class Handoff~TContext, TAgent~ {
        +str tool_name
        +str tool_description
        +dict input_json_schema
        +Callable on_invoke_handoff
        +str agent_name
        +HandoffInputFilter? input_filter
        +bool|Callable is_enabled
        
        +default_tool_name(agent) str
        +default_tool_description(agent) str
        +get_transfer_message(agent) str
        +is_enabled_for_context(context, agent) bool
    }
    
    class HandoffInputData {
        <<frozen>>
        +input_history: str | tuple[TResponseInputItem, ...]
        +pre_handoff_items: tuple[RunItem, ...]
        +new_items: tuple[RunItem, ...]
        +run_context: RunContextWrapper[Any]?
        
        +clone(**kwargs) HandoffInputData
    }
    
    class OnHandoffWithInput~THandoffInput~ {
        <<Callable Type>>
        +__call__(context: RunContextWrapper[Any], input: THandoffInput) Any
    }
    
    class OnHandoffWithoutInput {
        <<Callable Type>>
        +__call__(context: RunContextWrapper[Any]) Any
    }
    
    class HandoffInputFilter {
        <<Type Alias>>
        +__call__(data: HandoffInputData) MaybeAwaitable[HandoffInputData]
    }
    
    class ToolRunHandoff {
        +Handoff handoff
        +ToolCall tool_call
        +Agent source_agent
    }
    
    class HandoffOutputItem {
        +Agent agent
        +TResponseInputItem raw_item
        +Agent source_agent
        +Agent target_agent
    }
    
    class RealtimeHandoffEvent {
        +RealtimeAgent from_agent
        +RealtimeAgent to_agent
        +EventInfo info
    }
    
    class handoff_function {
        <<Factory Function>>
        +agent: Agent[TContext]
        +tool_name_override?: str
        +tool_description_override?: str
        +on_handoff?: OnHandoffWithInput | OnHandoffWithoutInput
        +input_type?: type[THandoffInput]
        +input_filter?: HandoffInputFilter
        +is_enabled?: bool | Callable
        
        +create_handoff() Handoff[TContext, Agent[TContext]]
    }
    
    class realtime_handoff_function {
        <<Factory Function>>
        +agent: RealtimeAgent[TContext]
        +tool_name_override?: str
        +tool_description_override?: str
        +on_handoff?: OnHandoffWithInput | OnHandoffWithoutInput
        +input_type?: type[THandoffInput]
        +is_enabled?: bool | Callable
        
        +create_realtime_handoff() Handoff[TContext, RealtimeAgent[TContext]]
    }
    
    class SingleStepResult {
        +Agent? new_agent
        +list[RunItem] new_step_items
        +list[RunItem] all_items
        +bool handoff_executed
    }
    
    Handoff --> HandoffInputData : uses
    Handoff --> HandoffInputFilter : uses
    Handoff --> OnHandoffWithInput : uses
    Handoff --> OnHandoffWithoutInput : uses
    
    handoff_function --> Handoff : creates
    realtime_handoff_function --> Handoff : creates
    
    ToolRunHandoff --> Handoff : contains
    HandoffOutputItem --> Handoff : result of
    
    RealtimeHandoffEvent --> Handoff : triggered by
    
    SingleStepResult --> Handoff : execution result
    
    Handoff --> Agent : delegates to
    
    note for Handoff "核心交接类\n定义代理间委派机制"
    note for HandoffInputData "交接数据容器\n管理上下文传递"
    note for handoff_function "交接工厂函数\n简化创建过程"
    note for ToolRunHandoff "交接执行容器\n包含执行信息"
```

**类图说明：**

### 核心类型层次

1. **Handoff类**：交接的核心实现，包含所有交接逻辑和配置
2. **HandoffInputData**：交接时传递的数据容器，支持不可变操作
3. **回调函数类型**：定义不同类型的交接回调接口
4. **工厂函数**：简化交接对象的创建过程

### 数据流转关系

- **输入传递**：`HandoffInputData` 封装所有需要传递的历史和状态信息
- **过滤处理**：`HandoffInputFilter` 对传递数据进行自定义处理
- **执行结果**：`SingleStepResult` 包含交接执行后的完整状态
- **事件通知**：`RealtimeHandoffEvent` 用于实时交接的事件通知

### 扩展性设计

- **泛型支持**：`Handoff` 类支持上下文和代理类型的泛型参数
- **回调灵活性**：支持带参数和无参数两种回调模式
- **条件启用**：`is_enabled` 支持静态和动态启用条件
- **类型安全**：通过 `TypeAdapter` 确保输入参数的类型安全

## 5. 典型使用场景时序图

### 场景一：客服分流交接流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant TriageAgent as 分流代理
    participant Runner as Runner
    participant RunImpl as RunImpl
    participant BillingAgent as 账单代理
    participant Tracing as 追踪系统
    
    User->>TriageAgent: "我的账单有问题"
    TriageAgent->>TriageAgent: 分析用户问题类型
    TriageAgent->>TriageAgent: 决定交接给账单专家
    
    TriageAgent->>Runner: 调用 billing_handoff 工具
    Runner->>RunImpl: 处理交接工具调用
    
    RunImpl->>RunImpl: 检查是否有多个交接冲突
    RunImpl->>Tracing: 开始交接追踪 (from: triage, to: billing)
    
    RunImpl->>BillingAgent: 执行 on_invoke_handoff()
    BillingAgent-->>RunImpl: 返回账单代理实例
    
    RunImpl->>RunImpl: 生成 HandoffOutputItem
    RunImpl->>RunImpl: "已将您转接到账单专家"
    
    RunImpl->>RunImpl: 执行交接钩子函数
    RunImpl->>RunImpl: 应用输入过滤器（如果有）
    
    RunImpl->>RunImpl: 构建新代理的输入历史
    RunImpl-->>Runner: SingleStepResult(new_agent=BillingAgent, ...)
    
    Runner->>BillingAgent: 使用过滤后的历史继续对话
    BillingAgent->>BillingAgent: "我是账单专家，我来帮您处理账单问题"
    BillingAgent-->>User: 专业的账单问题解答
    
    Tracing->>Tracing: 记录交接完成事件
    
    note over TriageAgent, BillingAgent: 从通用分流代理交接到<br/>专业账单处理代理
```

### 场景二：多交接冲突处理

```mermaid
sequenceDiagram
    autonumber
    participant Agent as 主代理
    participant Model as 模型
    participant RunImpl as RunImpl
    participant TechAgent as 技术代理
    participant SalesAgent as 销售代理
    participant SupportAgent as 支持代理
    participant Tracing as 追踪系统
    
    Agent->>Model: "用户需要技术支持和销售咨询"
    Model-->>Agent: 同时调用多个交接工具
    
    Agent->>RunImpl: [技术交接, 销售交接, 支持交接] (3个工具调用)
    
    RunImpl->>RunImpl: 检测到多交接冲突 (len(run_handoffs) > 1)
    RunImpl->>Tracing: 记录多交接冲突事件
    
    RunImpl->>RunImpl: 选择第一个交接：技术交接
    RunImpl->>TechAgent: 执行 tech_handoff.on_invoke_handoff()
    TechAgent-->>RunImpl: 返回技术代理实例
    
    RunImpl->>RunImpl: 拒绝其他交接请求
    
    loop 处理被拒绝的交接
        RunImpl->>RunImpl: 创建 ToolCallOutputItem
        RunImpl->>RunImpl: output = "错误：不能同时执行多个代理交接"
    end
    
    RunImpl->>RunImpl: 生成成功交接的输出
    RunImpl->>RunImpl: "已将您转接到技术专家"
    
    RunImpl->>Tracing: 更新追踪信息
    Tracing->>Tracing: 记录冲突的代理列表 [tech, sales, support]
    Tracing->>Tracing: 记录实际执行的代理：tech
    
    RunImpl-->>Agent: SingleStepResult(new_agent=TechAgent, 包含拒绝消息)
    
    note over Model, RunImpl: 模型同时调用了多个交接工具<br/>系统自动处理冲突，只执行第一个
```

### 场景三：带参数交接与输入过滤

```mermaid
sequenceDiagram
    autonumber
    participant Coordinator as 协调代理
    participant RunImpl as RunImpl
    participant Filter as 输入过滤器
    participant DataAgent as 数据分析代理
    participant TypeAdapter as 类型验证器
    participant Hooks as 生命周期钩子
    
    Coordinator->>RunImpl: 调用 data_analysis_handoff({query: "销售数据", period: "Q3"})
    
    RunImpl->>TypeAdapter: 验证输入参数类型
    TypeAdapter->>TypeAdapter: 检查 query: str, period: str
    TypeAdapter-->>RunImpl: 验证通过
    
    RunImpl->>DataAgent: on_invoke_handoff(context, {query, period})
    DataAgent->>DataAgent: 处理参数化交接逻辑
    DataAgent-->>RunImpl: 返回配置好的数据分析代理
    
    RunImpl->>Filter: 应用输入过滤器
    Filter->>Filter: 构建 HandoffInputData
    
    Filter->>Filter: 历史长度控制 (保留最近20条)
    Filter->>Filter: 敏感信息过滤
    Filter->>Filter: 上下文优化
    
    Filter->>Filter: 添加参数上下文信息
    Filter-->>RunImpl: 返回过滤后的 HandoffInputData
    
    RunImpl->>RunImpl: 构建新代理的完整输入
    RunImpl->>RunImpl: all_items = 过滤历史 + 预交接项 + 新项
    
    RunImpl->>Hooks: 并行执行交接钩子
    
    par 全局钩子
        Hooks->>Hooks: hooks.on_handoff(coordinator -> data_agent)
    and 代理钩子  
        Hooks->>Hooks: coordinator.hooks.on_handoff(data_agent)
    end
    
    Hooks-->>RunImpl: 钩子执行完成
    
    RunImpl-->>Coordinator: SingleStepResult(包含数据分析代理和过滤后输入)
    
    Coordinator->>DataAgent: 开始专业数据分析对话
    DataAgent->>DataAgent: "我将为您分析Q3销售数据，已收到您的查询参数"
    DataAgent-->>User: 基于参数的专业分析结果
    
    note over Filter, DataAgent: 参数验证确保类型安全<br/>输入过滤优化上下文传递
```

## 6. 最佳实践与使用模式

### 6.1 客服分流场景实现

```python
from agents import Agent, handoff
from dataclasses import dataclass
from typing import Literal

@dataclass
class HandoffRequest:
    """交接请求的标准化数据结构"""
    department: Literal["billing", "technical", "sales", "general"]
    priority: Literal["low", "medium", "high", "urgent"]
    user_info: str
    issue_summary: str

# 创建专业代理
billing_agent = Agent(
    name="BillingExpert",
    instructions="""
    你是专业的账单专家，负责处理：

    - 账单查询和解释
    - 付款问题和退费申请
    - 订阅管理和升级建议
    请提供准确、专业的账单相关服务。
    """

)

technical_agent = Agent(
    name="TechnicalSupport",
    instructions="""
    你是技术支持专家，负责处理：

    - 产品功能问题和故障排除
    - API集成和开发支持
    - 系统配置和性能优化
    请提供详细的技术解决方案。
    """

)

sales_agent = Agent(
    name="SalesConsultant",
    instructions="""
    你是销售顾问，负责处理：

    - 产品介绍和方案推荐
    - 价格咨询和商务谈判
    - 合同条款和服务协议
    请提供专业的销售咨询服务。
    """

)

def create_intelligent_handoff(target_agent: Agent, department: str):
    """创建智能交接函数"""
    
    async def on_handoff_with_context(context, request: HandoffRequest):
        """带上下文的交接处理"""
        # 记录交接原因和用户信息
        print(f"交接到{department}部门: {request.issue_summary}")
        print(f"优先级: {request.priority}, 用户: {request.user_info}")
        
        # 可以根据优先级和问题类型进行额外处理
        if request.priority == "urgent":
            # 紧急问题的特殊处理
            context.context["priority_flag"] = True
            context.context["escalated_at"] = datetime.now()
    
    return handoff(
        target_agent,
        on_handoff=on_handoff_with_context,
        input_type=HandoffRequest,
        tool_description=f"将用户转接到{department}专业团队处理相关问题",
    )

# 创建分流代理
triage_agent = Agent(
    name="CustomerTriage",
    instructions="""
    你是客服分流专家，负责理解用户问题并将其转接到合适的专业团队。
    
    分流规则：

    - 账单、付款、退费问题 → 使用 billing_handoff
    - 技术故障、API、集成问题 → 使用 technical_handoff
    - 产品咨询、购买、升级问题 → 使用 sales_handoff
    
    在转接前，请收集用户的基本信息和问题描述，确保转接准确。
    """,
    handoffs=[
        create_intelligent_handoff(billing_agent, "账单"),
        create_intelligent_handoff(technical_agent, "技术"),
        create_intelligent_handoff(sales_agent, "销售"),
    ]

)
```

### 6.2 输入过滤与上下文管理

```python
from agents import HandoffInputData
import re
from typing import List

def create_context_aware_filter(max_history: int = 15, filter_sensitive: bool = True):
    """创建上下文感知的输入过滤器"""
    
    def smart_handoff_filter(data: HandoffInputData) -> HandoffInputData:
        """智能交接输入过滤"""
        
        # 1. 历史长度控制
        if isinstance(data.input_history, tuple) and len(data.input_history) > max_history:
            # 保留重要的初始上下文和最近的对话
            important_start = data.input_history[:2]  # 保留开始的问候和问题描述
            recent_history = data.input_history[-(max_history-2):]  # 保留最近对话
            filtered_history = important_start + recent_history
        else:
            filtered_history = data.input_history
        
        # 2. 敏感信息过滤
        if filter_sensitive:
            filtered_pre_items = []
            for item in data.pre_handoff_items:
                filtered_item = filter_sensitive_info(item)
                filtered_pre_items.append(filtered_item)
        else:
            filtered_pre_items = list(data.pre_handoff_items)
        
        # 3. 添加交接上下文摘要
        context_summary = create_handoff_summary(data)
        summary_item = {
            "type": "message",
            "role": "system",
            "content": f"[交接上下文摘要] {context_summary}"
        }
        
        # 4. 工具调用历史清理
        filtered_new_items = []
        for item in data.new_items:
            if item.get("type") == "tool_call_output":
                # 简化工具调用输出，只保留关键信息
                simplified_item = simplify_tool_output(item)
                filtered_new_items.append(simplified_item)
            else:
                filtered_new_items.append(item)
        
        return data.clone(
            input_history=filtered_history,
            pre_handoff_items=tuple(filtered_pre_items),
            new_items=tuple([summary_item] + filtered_new_items),
        )
    
    return smart_handoff_filter

def filter_sensitive_info(item):
    """过滤敏感信息"""
    if isinstance(item, dict) and "content" in item:
        content = item["content"]
        
        # 过滤常见敏感信息模式
        patterns = {
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b': '[CARD_NUMBER]',  # 信用卡
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',  # 社会安全号
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',  # 邮箱
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': '[PHONE]',  # 电话号码
        }
        
        filtered_content = content
        for pattern, replacement in patterns.items():
            filtered_content = re.sub(pattern, replacement, filtered_content)
        
        # 创建过滤后的副本
        filtered_item = item.copy()
        filtered_item["content"] = filtered_content
        return filtered_item
    
    return item

def create_handoff_summary(data: HandoffInputData) -> str:
    """创建交接摘要"""
    # 分析对话历史，提取关键信息
    key_points = []
    
    if isinstance(data.input_history, tuple):
        for item in data.input_history[-5:]:  # 分析最近5条
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content", "")
                if len(content) > 50:  # 较长的用户输入可能是重要信息
                    key_points.append(content[:100] + "...")
    
    # 提取预交接项目中的重要信息
    for item in data.pre_handoff_items[-3:]:  # 最近3个项目
        if hasattr(item, 'type') and item.type == "message_output":
            # 代理的重要输出
            key_points.append(f"代理输出: {str(item)[:80]}...")
    
    if key_points:
        return f"用户关注: {'; '.join(key_points[:3])}"  # 最多3个要点
    else:
        return "新用户会话，无特殊背景"

# 使用过滤器的代理配置
filtered_triage_agent = Agent(
    name="FilteredTriage",
    instructions="智能客服分流，带上下文过滤",
    handoffs=[
        handoff(
            billing_agent,
            input_filter=create_context_aware_filter(max_history=10),
            tool_description="转接到账单专家（已过滤敏感信息）"
        ),
        handoff(
            technical_agent,
            input_filter=create_context_aware_filter(max_history=20),  # 技术问题需要更多上下文
            tool_description="转接到技术支持（保留详细技术上下文）"
        ),
    ]
)
```

### 6.3 条件交接与动态启用

```python
from agents import Agent, handoff, RunContextWrapper
from datetime import datetime, time

def create_business_hours_check():
    """创建营业时间检查函数"""
    
    async def is_business_hours(context: RunContextWrapper, agent: Agent) -> bool:
        """检查是否在营业时间内"""
        now = datetime.now()
        business_start = time(9, 0)  # 9:00 AM
        business_end = time(17, 0)   # 5:00 PM
        
        # 工作日且在营业时间内
        is_weekday = now.weekday() < 5
        is_business_hour = business_start <= now.time() <= business_end
        
        return is_weekday and is_business_hour
    
    return is_business_hours

def create_user_tier_check(required_tier: str):
    """创建用户等级检查函数"""
    
    async def check_user_tier(context: RunContextWrapper, agent: Agent) -> bool:
        """检查用户等级是否满足要求"""
        user_tier = context.context.get("user_tier", "basic")
        
        tier_hierarchy = {"basic": 1, "premium": 2, "enterprise": 3}
        required_level = tier_hierarchy.get(required_tier, 1)
        user_level = tier_hierarchy.get(user_tier, 1)
        
        return user_level >= required_level
    
    return check_user_tier

# 创建不同级别的代理
basic_support_agent = Agent(
    name="BasicSupport",
    instructions="提供基础技术支持服务"
)

premium_support_agent = Agent(
    name="PremiumSupport",
    instructions="提供高级技术支持，包括优先处理和深度分析"
)

enterprise_support_agent = Agent(
    name="EnterpriseSupport",
    instructions="提供企业级技术支持，包括架构建议和定制方案"
)

# 条件交接配置
conditional_triage_agent = Agent(
    name="ConditionalTriage",
    instructions="""
    根据用户等级和时间条件提供相应的服务转接：

    - 基础用户：工作时间内提供基础支持
    - 高级用户：扩展时间提供高级支持  
    - 企业用户：24/7企业级支持
    """,
    handoffs=[
        # 基础支持：仅工作时间
        handoff(
            basic_support_agent,
            is_enabled=create_business_hours_check(),
            tool_description="转接基础技术支持（工作时间内）"
        ),
        
        # 高级支持：高级用户可用
        handoff(
            premium_support_agent,
            is_enabled=create_user_tier_check("premium"),
            tool_description="转接高级技术支持（高级用户专享）"
        ),
        
        # 企业支持：企业用户24/7可用
        handoff(
            enterprise_support_agent,
            is_enabled=create_user_tier_check("enterprise"),
            tool_description="转接企业技术支持（企业用户24/7）"
        ),
    ]

)
```

### 6.4 复杂交接流程与状态管理

```python
from agents import Agent, handoff
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class EscalationLevel(Enum):
    L1 = "level1"  # 一线支持
    L2 = "level2"  # 二线支持  
    L3 = "level3"  # 专家支持
    MANAGER = "manager"  # 经理介入

@dataclass
class SupportCase:
    """支持案例数据结构"""
    case_id: str
    user_id: str
    issue_type: str
    severity: str
    escalation_level: EscalationLevel = EscalationLevel.L1
    previous_agents: List[str] = field(default_factory=list)
    resolution_attempts: int = 0
    specialized_knowledge_needed: List[str] = field(default_factory=list)

class SupportFlowManager:
    """支持流程管理器"""
    
    def __init__(self):
        self.active_cases: Dict[str, SupportCase] = {}
    
    def create_case(self, user_id: str, issue_type: str, severity: str) -> SupportCase:
        """创建新的支持案例"""
        case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}"
        case = SupportCase(
            case_id=case_id,
            user_id=user_id,
            issue_type=issue_type,
            severity=severity
        )
        self.active_cases[case_id] = case
        return case
    
    def escalate_case(self, case_id: str, to_level: EscalationLevel, reason: str):
        """升级案例到更高级别"""
        if case_id in self.active_cases:
            case = self.active_cases[case_id]
            case.escalation_level = to_level
            case.resolution_attempts += 1
            print(f"案例 {case_id} 升级到 {to_level.value}: {reason}")

# 全局流程管理器
flow_manager = SupportFlowManager()

# 创建各级别支持代理
l1_support = Agent(
    name="L1Support",
    instructions="""
    你是一线技术支持，处理常见问题：

    - 基础故障排除
    - 账户和密码问题
    - 简单配置指导
    如果问题复杂或无法解决，及时升级到L2支持。
    """

)

l2_support = Agent(
    name="L2Support",
    instructions="""
    你是二线技术支持，处理复杂问题：

    - 深度技术分析
    - 系统集成问题
    - 性能优化建议
    如果需要产品专家介入，升级到L3支持。
    """

)

l3_support = Agent(
    name="L3Support",
    instructions="""
    你是专家级技术支持，处理最复杂问题：

    - 产品架构设计
    - 定制化解决方案
    - 紧急故障处理
    """

)

def create_escalation_handoff(target_agent: Agent, target_level: EscalationLevel):
    """创建升级交接"""
    
    async def on_escalation(context, escalation_data: SupportCase):
        """处理升级逻辑"""
        case_id = escalation_data.case_id
        
        # 更新案例状态
        flow_manager.escalate_case(
            case_id,
            target_level,
            f"升级到{target_agent.name}"
        )
        
        # 记录代理历史
        escalation_data.previous_agents.append(context.context.get("current_agent", "unknown"))
        
        # 添加专业知识标记
        if target_level == EscalationLevel.L3:
            escalation_data.specialized_knowledge_needed.append("expert_analysis")
        
        # 更新上下文
        context.context["case_data"] = escalation_data
        context.context["escalation_reason"] = f"需要{target_level.value}级别处理"
        
        print(f"案例升级: {case_id} -> {target_agent.name}")
    
    def escalation_filter(handoff_data: HandoffInputData) -> HandoffInputData:
        """升级时的输入过滤"""
        # 添加升级上下文摘要
        case_data = handoff_data.run_context.context.get("case_data")
        if case_data:
            escalation_summary = {
                "type": "message",
                "role": "system",
                "content": f"""
                [案例升级摘要]
                案例ID: {case_data.case_id}
                当前级别: {case_data.escalation_level.value}
                解决尝试次数: {case_data.resolution_attempts}
                之前经手代理: {', '.join(case_data.previous_agents)}
                问题类型: {case_data.issue_type}
                严重程度: {case_data.severity}
                """
            }
            
            # 将摘要添加到新项目开头
            new_items = [escalation_summary] + list(handoff_data.new_items)
            return handoff_data.clone(new_items=tuple(new_items))
        
        return handoff_data
    
    return handoff(
        target_agent,
        on_handoff=on_escalation,
        input_type=SupportCase,
        input_filter=escalation_filter,
        tool_description=f"升级到{target_level.value}级别支持"
    )

# 创建支持流程代理
support_coordinator = Agent(
    name="SupportCoordinator",
    instructions="""
    你是技术支持协调员，负责：

    1. 评估问题复杂度和紧急程度
    2. 选择合适的支持级别
    3. 必要时进行案例升级
    
    升级标准：
    - L1无法解决的技术问题 → L2
    - 需要深度产品知识的问题 → L3
    - 系统性故障或紧急问题 → 直接L3
    """,
    handoffs=[
        handoff(
            l1_support,
            tool_description="分配给一线技术支持处理"
        ),
        create_escalation_handoff(l2_support, EscalationLevel.L2),
        create_escalation_handoff(l3_support, EscalationLevel.L3),
    ]

)

# 使用示例
async def support_flow_example():
    """支持流程使用示例"""
    
    # 创建支持案例
    case = flow_manager.create_case(
        user_id="user_12345",
        issue_type="api_integration",
        severity="high"
    )
    
    # 启动支持流程
    result = await Runner.run(
        support_coordinator,
        f"用户报告API集成问题，案例ID: {case.case_id}",
        context={"case_data": case}
    )
    
    print(f"支持流程结果: {result.final_output}")
```

Handoffs模块通过灵活的交接机制和完善的数据管理，为OpenAI Agents提供了强大的多代理协作能力，支持从简单的任务委派到复杂的工作流程管理。

---

## API接口

## 1. API 总览

Handoffs 模块提供了Agent之间任务委派的机制。通过Handoff，一个Agent可以将任务转交给专门的Agent处理，实现多Agent协作。

### API 分类

| API 类别 | 核心 API | 功能描述 |
|---------|---------|---------|
| **Handoff创建** | `handoff(agent)` | 创建简单的代理切换 |
| | `handoff(agent, on_handoff, input_type)` | 创建带输入验证的切换 |
| | `handoff(agent, input_filter)` | 创建带历史过滤的切换 |
| **执行控制** | `is_enabled` | 动态启用/禁用切换 |
| **数据处理** | `HandoffInputFilter` | 过滤传递的数据 |
| | `HandoffInputData.clone()` | 克隆并修改数据 |

## 2. handoff() 函数 API

### 2.1 基础用法 - 简单切换

**API 签名：**

```python
def handoff(
    agent: Agent,
    *,
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    input_filter: HandoffInputFilter | None = None,
    is_enabled: bool | Callable = True,
) -> Handoff
```

**功能描述：**
创建一个简单的Agent切换，不需要额外参数，直接将任务交给目标Agent。

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `agent` | `Agent` | 必填 | 目标Agent |
| `tool_name_override` | `str \| None` | `None` | 自定义工具名称 |
| `tool_description_override` | `str \| None` | `None` | 自定义工具描述 |
| `input_filter` | `HandoffInputFilter \| None` | `None` | 历史数据过滤器 |
| `is_enabled` | `bool \| Callable` | `True` | 是否启用 |

**使用示例：**

```python
from agents import Agent, handoff

# 创建专门的Agent
billing_agent = Agent(
    name="billing_agent",
    instructions="处理账单相关问题",
    handoff_description="处理账单、付款和发票问题"
)

support_agent = Agent(
    name="support_agent",
    instructions="处理技术支持问题",
    handoff_description="解决技术问题和故障排除"
)

# 创建主Agent，配置handoffs
triage_agent = Agent(
    name="triage_agent",
    instructions="""你是客服分流agent。

    - 账单问题 -> billing_agent
    - 技术问题 -> support_agent
    """,
    handoffs=[
        handoff(billing_agent),
        handoff(support_agent)
    ]

)

# 使用
result = await run(triage_agent, "我的账单有问题")
# triage_agent会自动切换到billing_agent
```

### 2.2 带输入参数的切换

**API 签名：**

```python
def handoff(
    agent: Agent,
    *,
    on_handoff: Callable[[RunContextWrapper, TInput], None],
    input_type: type[TInput],
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    input_filter: HandoffInputFilter | None = None,
    is_enabled: bool | Callable = True,
) -> Handoff
```

**功能描述：**
创建需要额外输入参数的切换，可以在切换时传递结构化数据。

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `on_handoff` | `Callable` | 切换时的回调函数，接收context和input |
| `input_type` | `type` | 输入数据的类型（用于验证） |

**使用示例：**

```python
from pydantic import BaseModel
from agents import Agent, handoff

# 定义切换输入类型
class TransferInput(BaseModel):
    reason: str
    priority: str
    customer_id: str | None = None

# 创建带输入验证的切换
def on_transfer_to_billing(context: RunContextWrapper, data: TransferInput):
    """切换时记录信息"""
    context.set("transfer_reason", data.reason)
    context.set("priority", data.priority)
    print(f"切换原因: {data.reason}, 优先级: {data.priority}")

billing_agent = Agent(name="billing_agent", instructions="...")

# 创建handoff
billing_handoff = handoff(
    billing_agent,
    on_handoff=on_transfer_to_billing,
    input_type=TransferInput,
    tool_description_override="切换到账单部门，需要提供原因和优先级"
)

triage_agent = Agent(
    name="triage",
    instructions="""
    切换到billing_agent时，必须提供：

    - reason: 切换原因
    - priority: 优先级(high/medium/low)
    - customer_id: 客户ID（可选）
    """,
    handoffs=[billing_handoff]

)
```

### 2.3 不带输入的回调切换

**API 签名：**

```python
def handoff(
    agent: Agent,
    *,
    on_handoff: Callable[[RunContextWrapper], None],
    ...
) -> Handoff
```

**使用示例：**

```python
# 切换时执行清理操作
def on_handoff_cleanup(context: RunContextWrapper):
    """切换前清理敏感数据"""
    context.delete("credit_card")
    context.delete("password")
    print("已清理敏感数据")

secure_agent = Agent(name="secure_agent", instructions="...")

handoff_with_cleanup = handoff(
    secure_agent,
    on_handoff=on_handoff_cleanup
)
```

## 3. Handoff 配置 API

### 3.1 自定义工具名称和描述

**使用示例：**

```python
# 默认生成的工具名称: transfer_to_billing_agent
# 自定义工具名称
custom_handoff = handoff(
    billing_agent,
    tool_name_override="escalate_to_billing",
    tool_description_override="将复杂的账单问题升级到专业团队"
)
```

**默认值生成规则：**

```python
# 工具名称: transform_string_function_style(f"transfer_to_{agent.name}")
# billing_agent -> transfer_to_billing_agent
# support-agent -> transfer_to_support_agent

# 工具描述: f"Handoff to the {agent.name} agent to handle the request. {agent.handoff_description}"
```

### 3.2 动态启用/禁用

**API 签名：**

```python
is_enabled: bool | Callable[[RunContextWrapper, Agent], bool | Awaitable[bool]]
```

**使用示例：**

```python
# 方式1: 静态禁用
disabled_handoff = handoff(
    agent,
    is_enabled=False
)

# 方式2: 基于上下文动态启用
def should_enable_vip_agent(context: RunContextWrapper, agent: Agent) -> bool:
    """只对VIP用户启用"""
    user_type = context.get("user_type")
    return user_type == "vip"

vip_handoff = handoff(
    vip_agent,
    is_enabled=should_enable_vip_agent
)

# 方式3: 异步检查
async def check_quota_before_handoff(context: RunContextWrapper, agent: Agent) -> bool:
    """检查配额"""
    user_id = context.get("user_id")
    quota = await quota_service.check(user_id)
    return quota > 0

quota_handoff = handoff(
    expert_agent,
    is_enabled=check_quota_before_handoff
)
```

## 4. HandoffInputFilter API

### 4.1 输入过滤器函数

**API 签名：**

```python
HandoffInputFilter = Callable[[HandoffInputData], HandoffInputData | Awaitable[HandoffInputData]]
```

**HandoffInputData 结构：**

```python
@dataclass
class HandoffInputData:
    input_history: str | tuple[TResponseInputItem, ...]  # 初始输入历史
    pre_handoff_items: tuple[RunItem, ...]  # 切换前的项
    new_items: tuple[RunItem, ...]  # 当前turn的新项
    run_context: RunContextWrapper | None  # 运行上下文
    
    def clone(self, **kwargs) -> HandoffInputData:
        """克隆并修改数据"""
```

**功能描述：**
过滤传递给下一个Agent的历史数据，可以删除、修改或添加历史项。

**使用示例：**

```python
# 示例1: 限制历史长度
def limit_history(data: HandoffInputData) -> HandoffInputData:
    """只保留最近5轮对话"""
    if isinstance(data.pre_handoff_items, tuple):
        limited_items = data.pre_handoff_items[-10:]  # 最近5轮（每轮2项）
        return data.clone(pre_handoff_items=limited_items)
    return data

# 示例2: 移除工具调用
def remove_tool_calls(data: HandoffInputData) -> HandoffInputData:
    """移除历史中的工具调用"""
    filtered_items = []
    for item in data.pre_handoff_items:
        if item.get("type") not in ["function_call", "function_call_output"]:
            filtered_items.append(item)
    
    return data.clone(pre_handoff_items=tuple(filtered_items))

# 示例3: 添加系统提示
def add_context_summary(data: HandoffInputData) -> HandoffInputData:
    """添加上下文摘要"""
    summary_item = {
        "type": "message",
        "role": "user",
        "content": f"[系统] 以下是之前的对话摘要: ..."
    }
    
    new_history = (summary_item,) + data.pre_handoff_items
    return data.clone(pre_handoff_items=new_history)

# 示例4: 异步过滤 - API调用
async def summarize_history_with_llm(data: HandoffInputData) -> HandoffInputData:
    """使用LLM总结历史"""
    # 生成摘要
    summary = await llm_summarize(data.pre_handoff_items)
    
    # 替换为摘要
    summary_item = {
        "type": "message",
        "role": "user",
        "content": f"历史摘要: {summary}"
    }
    
    return data.clone(
        pre_handoff_items=(summary_item,),
        input_history=""  # 清空原始输入
    )

# 在handoff中使用
filtered_handoff = handoff(
    agent,
    input_filter=limit_history
)

advanced_handoff = handoff(
    agent,
    input_filter=summarize_history_with_llm
)
```

## 5. Handoff 数据类 API

### 5.1 Handoff 类属性

**核心属性：**

```python
@dataclass
class Handoff:
    tool_name: str  # 工具名称
    tool_description: str  # 工具描述
    input_json_schema: dict  # 输入JSON Schema
    on_invoke_handoff: Callable  # 切换执行函数
    agent_name: str  # 目标Agent名称
    input_filter: HandoffInputFilter | None  # 输入过滤器
    strict_json_schema: bool = True  # 是否严格模式
    is_enabled: bool | Callable = True  # 是否启用
```

### 5.2 Handoff 工具方法

**get_transfer_message():**

```python
def get_transfer_message(self, agent: Agent) -> str:
    """生成切换消息"""
    return json.dumps({"assistant": agent.name})
```

**default_tool_name():**

```python
@classmethod
def default_tool_name(cls, agent: Agent) -> str:
    """生成默认工具名称"""
    return f"transfer_to_{agent.name}"
```

**default_tool_description():**

```python
@classmethod
def default_tool_description(cls, agent: Agent) -> str:
    """生成默认工具描述"""
    return f"Handoff to the {agent.name} agent. {agent.handoff_description or ''}"
```

## 6. 最佳实践

### 6.1 多层切换架构

```python
# 第一层: 主入口Agent
main_agent = Agent(
    name="main",
    instructions="根据用户需求分流到专业团队",
    handoffs=[
        handoff(sales_agent),
        handoff(support_agent),
        handoff(billing_agent)
    ]
)

# 第二层: 支持Agent可以继续切换
support_agent = Agent(
    name="support",
    instructions="处理技术问题，必要时升级",
    handoffs=[
        handoff(senior_support_agent),
        handoff(engineering_agent)
    ]
)

# 第三层: 工程团队Agent
engineering_agent = Agent(
    name="engineering",
    instructions="解决复杂技术问题",
    handoffs=[
        handoff(main_agent)  # 可以返回主Agent
    ]
)
```

### 6.2 带状态的切换

```python
class HandoffState(BaseModel):
    from_agent: str
    timestamp: str
    issue_type: str
    severity: str

def on_handoff_with_state(context: RunContextWrapper, state: HandoffState):
    """记录切换状态"""
    context.set("handoff_history", [
        *context.get("handoff_history", []),
        state.dict()
    ])
    
    # 根据严重程度设置优先级
    if state.severity == "critical":
        context.set("priority", "urgent")

specialist_handoff = handoff(
    specialist_agent,
    on_handoff=on_handoff_with_state,
    input_type=HandoffState
)
```

### 6.3 条件过滤组合

```python
def create_smart_filter(max_items: int, remove_tools: bool):
    """创建可配置的过滤器"""
    def filter_func(data: HandoffInputData) -> HandoffInputData:
        items = list(data.pre_handoff_items)
        
        # 移除工具调用
        if remove_tools:
            items = [
                item for item in items
                if item.get("type") not in ["function_call", "function_call_output"]
            ]
        
        # 限制数量
        items = items[-max_items:]
        
        return data.clone(pre_handoff_items=tuple(items))
    
    return filter_func

optimized_handoff = handoff(
    agent,
    input_filter=create_smart_filter(max_items=10, remove_tools=True)
)
```

Handoffs 模块通过灵活的API设计，为OpenAI Agents提供了强大的多Agent协作能力，实现复杂任务的智能分工。

---

## 数据结构

## 1. 数据结构总览

Handoffs 模块的数据结构定义了Agent间任务委派的机制，包括切换配置、输入数据和过滤器。

### 核心数据结构

```
Handoff (切换配置)
├── tool_name: str
├── tool_description: str
├── input_json_schema: dict
├── on_invoke_handoff: Callable
├── agent_name: str
├── input_filter: HandoffInputFilter
├── strict_json_schema: bool
└── is_enabled: bool | Callable

HandoffInputData (切换数据)
├── input_history: str | tuple
├── pre_handoff_items: tuple[RunItem]
├── new_items: tuple[RunItem]
└── run_context: RunContextWrapper
```

## 2. Handoff 类 UML 图

```mermaid
classDiagram
    class Handoff~TContext,TAgent~ {
        +tool_name: str
        +tool_description: str
        +input_json_schema: dict
        +on_invoke_handoff: Callable
        +agent_name: str
        +input_filter: HandoffInputFilter | None
        +strict_json_schema: bool
        +is_enabled: bool | Callable
        +get_transfer_message(agent) str
        +default_tool_name(agent)$ str
        +default_tool_description(agent)$ str
    }
    
    class HandoffInputData {
        +input_history: str | tuple
        +pre_handoff_items: tuple[RunItem]
        +new_items: tuple[RunItem]
        +run_context: RunContextWrapper | None
        +clone(**kwargs) HandoffInputData
    }
    
    class HandoffInputFilter {
        <<type alias>>
        Callable~HandoffInputData, HandoffInputData~
    }
    
    class Agent {
        +name: str
        +handoffs: list[Handoff]
        +handoff_description: str | None
    }
    
    Agent --> Handoff : 包含多个
    Handoff --> HandoffInputFilter : 可选使用
    HandoffInputFilter --> HandoffInputData : 处理
```

## 3. Handoff 数据结构详解

### 3.1 Handoff 字段说明

```python
@dataclass
class Handoff(Generic[TContext, TAgent]):
    """Agent切换配置"""
    
    tool_name: str
    """工具名称，在模型中显示为可调用的工具"""
    
    tool_description: str
    """工具描述，帮助模型理解何时使用此切换"""
    
    input_json_schema: dict[str, Any]
    """输入参数的JSON Schema，空字典表示无参数"""
    
    on_invoke_handoff: Callable[[RunContextWrapper, str], Awaitable[Agent]]
    """
    切换执行函数
    参数1: RunContextWrapper - 运行上下文
    参数2: str - LLM提供的JSON参数字符串
    返回: Agent - 目标Agent
    """
    
    agent_name: str
    """目标Agent的名称"""
    
    input_filter: HandoffInputFilter | None = None
    """输入数据过滤器，用于修改传递给下一个Agent的历史"""
    
    strict_json_schema: bool = True
    """是否使用严格的JSON Schema模式"""
    
    is_enabled: bool | Callable[[RunContextWrapper, Agent], bool] = True
    """是否启用此切换，可以是布尔值或动态判断函数"""
```

**字段类型和示例：**

| 字段 | 类型 | 示例值 | 说明 |
|------|------|--------|------|
| `tool_name` | `str` | `"transfer_to_billing_agent"` | 函数风格的名称 |
| `tool_description` | `str` | `"Handoff to billing agent..."` | 清晰的功能描述 |
| `input_json_schema` | `dict` | `{"properties": {...}}` | Pydantic生成的Schema |
| `agent_name` | `str` | `"billing_agent"` | 目标Agent标识 |
| `strict_json_schema` | `bool` | `True` | 推荐保持True |

### 3.2 Handoff 工作模式

```mermaid
graph TD
    A[Handoff创建] --> B{有输入参数?}
    
    B -->|是| C[定义input_type]
    B -->|否| D[空Schema]
    
    C --> E[生成JSON Schema]
    D --> E
    
    E --> F{有on_handoff?}
    
    F -->|有输入| G[on_handoff接收context和input]
    F -->|无输入| H[on_handoff仅接收context]
    F -->|无| I[直接返回Agent]
    
    G --> J[包装为on_invoke_handoff]
    H --> J
    I --> J
    
    J --> K{有input_filter?}
    
    K -->|是| L[过滤历史数据]
    K -->|否| M[使用原始历史]
    
    L --> N[传递给目标Agent]
    M --> N
```

## 4. HandoffInputData 数据结构

### 4.1 字段详解

```python
@dataclass(frozen=True)
class HandoffInputData:
    """传递给下一个Agent的数据"""
    
    input_history: str | tuple[TResponseInputItem, ...]
    """
    Runner.run()调用前的输入历史

    - 字符串: 简单文本输入
    - tuple: 结构化的对话历史
    """
    
    pre_handoff_items: tuple[RunItem, ...]
    """
    切换发生前生成的所有项
    包括之前的对话、工具调用等
    """
    
    new_items: tuple[RunItem, ...]
    """
    当前turn生成的新项
    包括触发切换的工具调用和切换结果
    """
    
    run_context: RunContextWrapper | None = None
    """
    切换时的运行上下文
    可用于访问自定义状态
    """

```

**数据流示意：**

```
Timeline:
├── input_history          [初始输入]
├── pre_handoff_items      [之前的所有对话]
│   ├── user message 1
│   ├── assistant message 1
│   ├── tool call 1
│   ├── tool output 1
│   └── ...
├── [HANDOFF TRIGGERED]    [切换点]
└── new_items              [当前turn]
    ├── function_call (handoff)
    └── function_call_output
```

### 4.2 HandoffInputData 类型示例

```python
# 示例1: 简单文本输入
data = HandoffInputData(
    input_history="用户最初的问题",
    pre_handoff_items=(
        {"type": "message", "role": "user", "content": "用户最初的问题"},
        {"type": "message", "role": "assistant", "content": "让我帮你..."},
    ),
    new_items=(
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "transfer_to_billing_agent",
            "arguments": "{}"
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": '{"assistant": "billing_agent"}'
        }
    ),
    run_context=context
)

# 示例2: 结构化历史
data = HandoffInputData(
    input_history=(
        {"type": "message", "role": "user", "content": "初始问题"},
    ),
    pre_handoff_items=(...),
    new_items=(...),
    run_context=context
)
```

### 4.3 clone() 方法

```python
def clone(self, **kwargs: Any) -> HandoffInputData:
    """
    克隆并修改数据
    
    用法:
    new_data = data.clone(
        pre_handoff_items=filtered_items,
        new_items=()
    )
    """
    return dataclasses_replace(self, **kwargs)
```

**使用示例：**

```python
# 场景1: 清空新项
filtered = data.clone(new_items=())

# 场景2: 替换历史
filtered = data.clone(
    pre_handoff_items=(summary_item,),
    input_history=""
)

# 场景3: 修改多个字段
filtered = data.clone(
    pre_handoff_items=limited_items,
    new_items=(),
    run_context=new_context
)
```

## 5. HandoffInputFilter 类型

### 5.1 类型定义

```python
HandoffInputFilter: TypeAlias = Callable[
    [HandoffInputData],
    MaybeAwaitable[HandoffInputData]
]
```

**支持的函数签名：**

```python
# 同步过滤器
def sync_filter(data: HandoffInputData) -> HandoffInputData:
    ...

# 异步过滤器
async def async_filter(data: HandoffInputData) -> HandoffInputData:
    ...
```

### 5.2 过滤器模式 UML 图

```mermaid
classDiagram
    class HandoffInputFilter {
        <<function>>
        +__call__(HandoffInputData) HandoffInputData
    }
    
    class LimitHistoryFilter {
        +max_items: int
        +__call__(data) HandoffInputData
    }
    
    class RemoveToolsFilter {
        +__call__(data) HandoffInputData
    }
    
    class SummarizeFilter {
        +llm_client: LLM
        +__call__(data) async HandoffInputData
    }
    
    class CompositeFilter {
        +filters: list
        +__call__(data) HandoffInputData
    }
    
    HandoffInputFilter <|.. LimitHistoryFilter
    HandoffInputFilter <|.. RemoveToolsFilter
    HandoffInputFilter <|.. SummarizeFilter
    HandoffInputFilter <|.. CompositeFilter
```

## 6. 回调函数类型

### 6.1 OnHandoff 类型别名

```python
# 带输入的回调
OnHandoffWithInput = Callable[
    [RunContextWrapper, THandoffInput],
    Any  # 可以是None或Awaitable[None]
]

# 不带输入的回调
OnHandoffWithoutInput = Callable[
    [RunContextWrapper],
    Any  # 可以是None或Awaitable[None]
]
```

**使用示例：**

```python
# 类型1: 带输入的同步回调
def sync_with_input(ctx: RunContextWrapper, data: MyInput) -> None:
    ctx.set("input_data", data)

# 类型2: 带输入的异步回调
async def async_with_input(ctx: RunContextWrapper, data: MyInput) -> None:
    await save_to_db(data)
    ctx.set("saved", True)

# 类型3: 不带输入的同步回调
def sync_no_input(ctx: RunContextWrapper) -> None:
    ctx.set("transferred", True)

# 类型4: 不带输入的异步回调
async def async_no_input(ctx: RunContextWrapper) -> None:
    await log_handoff(ctx)
```

## 7. 完整的数据流转图

### 7.1 Handoff执行数据流

```mermaid
graph TD
    A[Agent决定切换] --> B[触发Handoff工具]
    B --> C[提取工具参数]
    
    C --> D{有input_type?}
    D -->|是| E[验证JSON参数]
    D -->|否| F[跳过验证]
    
    E --> G{有on_handoff?}
    F --> G
    
    G -->|带输入| H[调用on_handoff<br/>context, validated_input]
    G -->|无输入| I[调用on_handoff<br/>context]
    G -->|无| J[直接获取Agent]
    
    H --> K[返回目标Agent]
    I --> K
    J --> K
    
    K --> L{有input_filter?}
    
    L -->|是| M[创建HandoffInputData]
    L -->|否| N[使用原始数据]
    
    M --> O[调用input_filter<br/>data]
    O --> P[过滤后的数据]
    
    P --> Q[传递给目标Agent]
    N --> Q
    
    Q --> R[目标Agent执行]
```

### 7.2 数据结构关系图

```mermaid
classDiagram
    class Agent {
        +name: str
        +handoffs: list[Handoff]
        +handoff_description: str
    }
    
    class Handoff {
        +tool_name: str
        +agent_name: str
        +input_filter: HandoffInputFilter
        +on_invoke_handoff: Callable
        +is_enabled: bool | Callable
    }
    
    class HandoffInputData {
        +input_history: str | tuple
        +pre_handoff_items: tuple
        +new_items: tuple
        +run_context: RunContextWrapper
        +clone() HandoffInputData
    }
    
    class HandoffInputFilter {
        <<callable>>
        +process(HandoffInputData) HandoffInputData
    }
    
    class RunItem {
        +type: str
        +content: Any
    }
    
    class RunContextWrapper {
        +get(key)
        +set(key, value)
    }
    
    Agent "1" --> "*" Handoff : 配置
    Handoff --> Agent : 指向目标
    Handoff ..> HandoffInputFilter : 可选使用
    HandoffInputFilter --> HandoffInputData : 处理
    HandoffInputData --> "*" RunItem : 包含
    HandoffInputData --> RunContextWrapper : 引用
```

## 8. 类型参数和泛型

### 8.1 Handoff 泛型参数

```python
class Handoff(Generic[TContext, TAgent]):
    """
    TContext: Agent的context类型
    TAgent: 目标Agent类型
    """
```

**类型示例：**

```python
# 场景1: 通用Agent
handoff_1: Handoff[Any, Agent[Any]]

# 场景2: 特定Context类型
class MyContext(TypedDict):
    user_id: str
    session_id: str

handoff_2: Handoff[MyContext, Agent[MyContext]]

# 场景3: 自定义Agent类型
class SpecialAgent(Agent):
    pass

handoff_3: Handoff[Any, SpecialAgent]
```

### 8.2 HandoffInput 泛型

```python
THandoffInput = TypeVar("THandoffInput", default=Any)
```

**使用示例：**

```python
from pydantic import BaseModel

class BillingInput(BaseModel):
    issue_type: str
    account_id: str

# 明确的类型标注
def handle_billing(ctx: RunContextWrapper, input: BillingInput) -> None:
    ...

billing_handoff = handoff(
    billing_agent,
    on_handoff=handle_billing,
    input_type=BillingInput  # 类型推断
)
```

## 9. 数据验证和转换

### 9.1 JSON Schema 生成

```python
from pydantic import BaseModel, TypeAdapter

class TransferData(BaseModel):
    reason: str
    priority: Literal["high", "medium", "low"]

# 自动生成Schema
adapter = TypeAdapter(TransferData)
schema = adapter.json_schema()

# 结果:
# {
#     "properties": {
#         "reason": {"type": "string"},
#         "priority": {"enum": ["high", "medium", "low"]}
#     },
#     "required": ["reason", "priority"]
# }
```

### 9.2 严格模式转换

```python
# 自动转换为OpenAI严格模式
input_json_schema = ensure_strict_json_schema(schema)

# 添加必要的字段:
# - additionalProperties: false
# - 所有字段都required
```

Handoffs 模块通过精心设计的数据结构，实现了灵活而强大的多Agent协作机制，支持复杂的任务委派和数据传递。

---

## 时序图

## 1. 时序图总览

Handoffs 模块的时序图展示了Agent间任务切换的完整流程，包括切换触发、参数验证、数据过滤和目标Agent执行。

### 主要时序流程

| 时序流程 | 参与者 | 触发时机 | 核心操作 |
|---------|--------|---------|---------|
| **Handoff创建** | 用户代码, handoff函数 | Agent初始化 | 配置切换规则 |
| **切换触发** | Runner, Model, Handoff | 模型决策 | 调用切换工具 |
| **参数验证** | Handoff, TypeAdapter | 切换执行 | 验证输入参数 |
| **数据过滤** | HandoffInputFilter | 切换前 | 过滤历史数据 |
| **目标执行** | Runner, Target Agent | 切换后 | 执行目标Agent |

## 2. Handoff 创建时序图

### 2.1 简单切换创建

```mermaid
sequenceDiagram
    participant U as 用户代码
    participant HF as handoff函数
    participant H as Handoff对象
    participant A as Agent
    
    U->>A: 创建目标Agent
    activate A
    A->>A: name="billing_agent"
    A->>A: handoff_description="..."
    A-->>U: billing_agent
    deactivate A
    
    U->>HF: handoff(billing_agent)
    activate HF
    
    Note over HF: 1. 生成工具名称
    HF->>HF: tool_name = default_tool_name(agent)
    HF->>HF: "transfer_to_billing_agent"
    
    Note over HF: 2. 生成工具描述
    HF->>HF: tool_description = default_tool_description(agent)
    
    Note over HF: 3. 空Schema（无参数）
    HF->>HF: input_json_schema = {}
    
    Note over HF: 4. 创建invoke函数
    HF->>HF: 生成_invoke_handoff
    HF->>HF: async def _invoke_handoff(ctx, input_json):<br/>    return agent
    
    Note over HF: 5. 创建Handoff对象
    HF->>H: Handoff(tool_name, tool_description, ...)
    activate H
    H-->>HF: handoff实例
    deactivate H
    
    HF-->>U: Handoff对象
    deactivate HF
    
    U->>A: triage_agent.handoffs = [handoff]
```

**创建流程说明：**

1. **工具命名**：自动生成`transfer_to_{agent_name}`格式
2. **工具描述**：组合Agent名称和handoff_description
3. **Schema生成**：无参数则为空字典
4. **包装函数**：创建统一的invoke函数
5. **返回对象**：完整的Handoff配置

### 2.2 带参数的切换创建

```mermaid
sequenceDiagram
    participant U as 用户代码
    participant HF as handoff函数
    participant TA as TypeAdapter
    participant H as Handoff对象
    
    U->>HF: handoff(agent, on_handoff=func, input_type=MyInput)
    activate HF
    
    Note over HF: 1. 验证函数签名
    HF->>HF: sig = inspect.signature(on_handoff)
    HF->>HF: 检查参数数量 = 2
    
    Note over HF: 2. 创建TypeAdapter
    HF->>TA: TypeAdapter(MyInput)
    activate TA
    TA->>TA: 分析MyInput类型
    TA-->>HF: type_adapter
    deactivate TA
    
    Note over HF: 3. 生成JSON Schema
    HF->>TA: type_adapter.json_schema()
    TA-->>HF: schema_dict
    
    Note over HF: 4. 转换为严格模式
    HF->>HF: ensure_strict_json_schema(schema)
    HF->>HF: 添加additionalProperties等字段
    
    Note over HF: 5. 创建包装函数
    HF->>HF: async def _invoke_handoff(ctx, input_json):
    HF->>HF:     validated = validate_json(input_json, type_adapter)
    HF->>HF:     on_handoff(ctx, validated)
    HF->>HF:     return agent
    
    Note over HF: 6. 创建Handoff
    HF->>H: Handoff(tool_name, ..., input_json_schema=schema)
    H-->>HF: handoff实例
    
    HF-->>U: Handoff对象
    deactivate HF
```

**带参数创建说明：**

1. **签名验证**：确保on_handoff接受2个参数
2. **类型适配器**：创建Pydantic TypeAdapter
3. **Schema生成**：从类型生成JSON Schema
4. **严格模式**：转换为OpenAI严格模式
5. **验证包装**：在invoke中加入验证逻辑

## 3. Handoff 触发时序图

### 3.1 模型决策切换

```mermaid
sequenceDiagram
    participant U as 用户
    participant R as Runner
    participant TA as TriageAgent
    participant M as Model
    participant T as TracingProcessor
    
    U->>R: run(triage_agent, "我的账单有问题")
    activate R
    
    Note over R: 1. 收集handoffs
    R->>TA: 获取agent.handoffs
    TA-->>R: [billing_handoff, support_handoff]
    
    Note over R: 2. 转换为工具定义
    R->>R: 将handoffs转换为Tool
    R->>R: tools = [<br/>  Tool("transfer_to_billing_agent", ...),<br/>  Tool("transfer_to_support_agent", ...)<br/>]
    
    Note over R: 3. 调用模型
    R->>M: get_response(input, tools, ...)
    activate M
    
    Note over M: 模型分析输入
    M->>M: "用户提到账单问题"
    M->>M: "应该切换到billing_agent"
    
    M-->>R: ModelResponse with function_call
    deactivate M
    
    Note over R: 4. 检测到handoff调用
    R->>R: 识别function_call.name = "transfer_to_billing_agent"
    R->>R: 查找对应的Handoff对象
    
    R->>T: on_span_start(handoff_span)
    
    Note over R: 5. 执行handoff
    R->>R: 调用on_invoke_handoff
    R-->>U: 切换到billing_agent继续执行
    deactivate R
```

**触发流程说明：**

1. **工具注册**：Handoffs转换为模型可调用的工具
2. **模型决策**：LLM分析后决定调用切换工具
3. **识别切换**：Runner检测function_call是handoff
4. **执行切换**：调用on_invoke_handoff
5. **继续执行**：在目标Agent上继续

### 3.2 动态启用检查

```mermaid
sequenceDiagram
    participant R as Runner
    participant H as Handoff
    participant Check as is_enabled检查
    participant M as Model
    
    R->>R: 准备工具列表
    
    loop 遍历所有handoffs
        R->>H: 检查is_enabled
        activate H
        
        alt is_enabled是bool
            H->>H: 直接返回True/False
            H-->>R: enabled状态
            
        else is_enabled是Callable
            H->>Check: is_enabled(context, agent)
            activate Check
            
            Check->>Check: 执行自定义逻辑
            Check->>Check: 例如: user_type == "vip"
            
            alt 同步函数
                Check-->>H: bool结果
            else 异步函数
                Check-->>H: awaitable
                H->>H: await result
            end
            deactivate Check
            
            H-->>R: enabled状态
        end
        deactivate H
        
        alt enabled = True
            R->>R: 添加到工具列表
        else enabled = False
            R->>R: 跳过此handoff
        end
    end
    
    R->>M: 传递过滤后的工具列表
```

**动态启用说明：**

1. **条件检查**：支持静态bool和动态函数
2. **异步支持**：可以执行异步检查（如API调用）
3. **工具过滤**：禁用的handoff不暴露给模型
4. **运行时决策**：每次调用时重新评估

## 4. 参数验证时序图

### 4.1 输入参数验证流程

```mermaid
sequenceDiagram
    participant R as Runner
    participant H as Handoff
    participant TA as TypeAdapter
    participant OH as on_handoff回调
    participant A as Agent
    
    R->>H: on_invoke_handoff(ctx, input_json)
    activate H
    
    Note over H: 1. 检查是否需要验证
    alt 有input_type
        Note over H: 2. 验证JSON不为空
        alt input_json is None
            H->>H: 记录错误
            H-->>R: raise ModelBehaviorError
        end
        
        Note over H: 3. 解析和验证JSON
        H->>TA: validate_json(input_json, type_adapter)
        activate TA
        
        alt JSON解析失败
            TA-->>H: JSONDecodeError
            H-->>R: raise ValidationError
        end
        
        alt 类型验证失败
            TA-->>H: ValidationError
            H-->>R: raise ValidationError
        end
        
        TA-->>H: validated_input (Python对象)
        deactivate TA
        
        Note over H: 4. 调用回调函数
        H->>OH: on_handoff(ctx, validated_input)
        activate OH
        
        alt 同步函数
            OH->>OH: 执行逻辑
            OH-->>H: None
        else 异步函数
            OH->>OH: 执行异步逻辑
            OH-->>H: awaitable
            H->>H: await result
        end
        deactivate OH
        
    else 无input_type
        Note over H: 跳过验证
        
        alt 有on_handoff（无参数）
            H->>OH: on_handoff(ctx)
            OH-->>H: None
        end
    end
    
    Note over H: 5. 返回目标Agent
    H->>A: 返回agent对象
    A-->>H: agent
    H-->>R: Agent
    deactivate H
```

**验证流程说明：**

1. **类型检查**：判断是否需要参数验证
2. **空值检查**：确保必需参数不为空
3. **JSON解析**：将字符串解析为对象
4. **类型验证**：使用Pydantic验证类型
5. **回调执行**：调用用户的on_handoff函数

## 5. 数据过滤时序图

### 5.1 input_filter 执行流程

```mermaid
sequenceDiagram
    participant R as Runner
    participant H as Handoff
    participant IF as InputFilter
    participant D as HandoffInputData
    participant NA as NextAgent
    
    Note over R: 准备切换数据
    R->>D: 创建HandoffInputData
    activate D
    D->>D: input_history = 初始输入
    D->>D: pre_handoff_items = 历史对话
    D->>D: new_items = 切换相关项
    D->>D: run_context = 当前上下文
    D-->>R: data对象
    deactivate D
    
    Note over R: 检查是否有过滤器
    R->>H: 获取input_filter
    H-->>R: filter_func或None
    
    alt 有input_filter
        R->>IF: filter_func(data)
        activate IF
        
        Note over IF: 执行过滤逻辑
        
        alt 示例: 限制历史长度
            IF->>IF: items = data.pre_handoff_items
            IF->>IF: limited = items[-10:]
            IF->>D: data.clone(pre_handoff_items=limited)
            D-->>IF: 新的data对象
            
        else 示例: 移除工具调用
            IF->>IF: filtered = [item for item in items<br/>              if item["type"] not in ["function_call"]]
            IF->>D: data.clone(pre_handoff_items=tuple(filtered))
            D-->>IF: 新的data对象
            
        else 示例: 异步LLM总结
            IF->>IF: summary = await llm.summarize(data)
            IF->>IF: summary_item = create_summary_message(summary)
            IF->>D: data.clone(pre_handoff_items=(summary_item,))
            D-->>IF: 新的data对象
        end
        
        IF-->>R: filtered_data
        deactivate IF
        
    else 无input_filter
        R->>R: 使用原始data
    end
    
    Note over R: 传递数据给目标Agent
    R->>NA: run(agent, filtered_data)
    activate NA
    NA->>NA: 使用filtered_data作为输入
    NA-->>R: 执行结果
    deactivate NA
```

**过滤流程说明：**

1. **数据准备**：创建完整的HandoffInputData
2. **过滤器检查**：判断是否配置了filter
3. **执行过滤**：调用filter函数处理数据
4. **数据克隆**：使用clone修改数据
5. **传递结果**：将过滤后数据传给目标Agent

### 5.2 多种过滤器组合

```mermaid
sequenceDiagram
    participant R as Runner
    participant F1 as 过滤器1<br/>限制长度
    participant F2 as 过滤器2<br/>移除工具
    participant F3 as 过滤器3<br/>添加摘要
    participant D as HandoffInputData
    
    R->>D: 原始data
    
    Note over R: 链式过滤
    R->>F1: filter1(data)
    activate F1
    F1->>D: data.clone(pre_handoff_items=limited)
    D-->>F1: data_v1
    F1-->>R: data_v1
    deactivate F1
    
    R->>F2: filter2(data_v1)
    activate F2
    F2->>D: data_v1.clone(pre_handoff_items=no_tools)
    D-->>F2: data_v2
    F2-->>R: data_v2
    deactivate F2
    
    R->>F3: filter3(data_v2)
    activate F3
    F3->>D: data_v2.clone(new_items=with_summary)
    D-->>F3: data_v3
    F3-->>R: data_v3 (最终)
    deactivate F3
```

## 6. 完整的 Handoff 执行时序图

### 6.1 端到端切换流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant R as Runner
    participant TA as TriageAgent
    participant M as Model
    participant H as Handoff
    participant IF as InputFilter
    participant BA as BillingAgent
    participant T as TracingProcessor
    
    U->>R: run(triage_agent, "账单问题")
    activate R
    
    Note over R: 阶段1: 初始Agent执行
    R->>TA: 执行triage_agent
    TA->>M: 调用模型(handoffs作为工具)
    M-->>TA: function_call("transfer_to_billing_agent")
    TA-->>R: 需要执行handoff
    
    Note over R: 阶段2: 追踪开始
    R->>T: on_span_start(handoff_span)
    T-->>R: span已记录
    
    Note over R: 阶段3: 查找Handoff
    R->>R: 查找tool_name匹配的Handoff
    R->>H: 找到billing_handoff
    
    Note over R: 阶段4: 执行切换
    R->>H: on_invoke_handoff(ctx, args_json)
    activate H
    
    alt 有输入参数
        H->>H: 验证参数
        H->>H: 调用on_handoff回调
    end
    
    H-->>R: billing_agent
    deactivate H
    
    Note over R: 阶段5: 数据过滤
    R->>R: 创建HandoffInputData
    
    alt 有input_filter
        R->>IF: filter(handoff_input_data)
        IF-->>R: 过滤后的data
    end
    
    Note over R: 阶段6: 追踪结束
    R->>T: on_span_end(handoff_span, success)
    
    Note over R: 阶段7: 执行目标Agent
    R->>BA: 执行billing_agent(filtered_data)
    activate BA
    BA->>M: 调用模型
    M-->>BA: 账单相关响应
    BA-->>R: 最终结果
    deactivate BA
    
    R-->>U: RunResult
    deactivate R
```

**完整流程总结：**

1. **初始执行**：Triage Agent分析输入
2. **模型决策**：决定切换到Billing Agent
3. **追踪记录**：开始handoff span
4. **切换执行**：验证参数，调用回调
5. **数据过滤**：过滤历史数据
6. **追踪完成**：记录切换成功
7. **目标执行**：在Billing Agent上继续
8. **返回结果**：最终结果给用户

## 7. 多层切换时序图

### 7.1 级联切换流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant R as Runner
    participant L1 as Layer1Agent
    participant L2 as Layer2Agent
    participant L3 as Layer3Agent
    
    U->>R: run(L1Agent, "复杂问题")
    activate R
    
    Note over R,L1: 第一层切换
    R->>L1: 执行L1Agent
    L1-->>R: handoff to L2Agent
    
    R->>R: 执行handoff to L2
    R->>R: 过滤数据
    
    Note over R,L2: 第二层切换
    R->>L2: 执行L2Agent(filtered_data)
    L2-->>R: handoff to L3Agent
    
    R->>R: 执行handoff to L3
    R->>R: 再次过滤数据
    
    Note over R,L3: 第三层执行
    R->>L3: 执行L3Agent(filtered_data)
    L3-->>R: 最终响应
    
    R-->>U: 最终结果
    deactivate R
```

**级联切换说明：**

1. **多层架构**：支持Agent间多次切换
2. **数据传递**：每次切换都可以过滤数据
3. **上下文保持**：RunContext在切换间传递
4. **灵活返回**：可以切换回上层Agent

## 8. 错误处理时序图

### 8.1 参数验证失败

```mermaid
sequenceDiagram
    participant R as Runner
    participant H as Handoff
    participant TA as TypeAdapter
    participant T as TracingProcessor
    
    R->>H: on_invoke_handoff(ctx, invalid_json)
    activate H
    
    H->>TA: validate_json(invalid_json, type_adapter)
    activate TA
    
    Note over TA: JSON解析或验证失败
    TA-->>H: ValidationError
    deactivate TA
    
    H->>T: 记录错误到span
    H->>H: 包装为ModelBehaviorError
    
    H-->>R: raise ModelBehaviorError
    deactivate H
    
    R->>T: on_span_end(handoff_span, error)
    R->>R: 处理错误
    R-->>R: 返回错误给用户或重试
```

Handoffs 模块通过精心设计的时序流程，实现了灵活的多Agent协作机制，支持复杂的任务委派和数据传递场景。

---
