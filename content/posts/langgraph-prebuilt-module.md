---
title: "深入LangGraph预构建组件：高级API与ReAct智能体源码解析"
date: 2025-07-27T13:00:00+08:00
draft: false
featured: true
series: "langgraph-architecture"
tags: ["LangGraph", "Prebuilt", "ReAct", "工具节点", "智能体", "高级API"]
categories: ["langgraph", "AI框架"]
author: "tommie blog"
description: "深度剖析LangGraph预构建组件的设计与实现，从ReAct智能体到工具执行节点的完整高级API解决方案"
toc: true
tocOpen: true
showReadingTime: true
showWordCount: true
weight: 250
slug: "langgraph-prebuilt-module"
---

## 概述

LangGraph预构建模块提供了高级API，用于快速创建和部署智能体和工具。该模块封装了常见的设计模式，如ReAct智能体架构、工具执行节点等，让开发者能够快速构建复杂的AI应用。这些预构建组件的实现原理。

<!--more-->

## 1. Prebuilt模块架构

### 1.1 核心组件关系图

```mermaid
classDiagram
    class create_react_agent {
        <<function>>
        +model: LanguageModelLike
        +tools: Sequence~BaseTool~
        +prompt: Optional~Prompt~
        +state_schema: Optional~StateSchemaType~
        +checkpointer: Optional~Checkpointer~
        +create_react_agent() CompiledStateGraph
    }
    
    class ToolNode {
        +list tools
        +dict fallbacks
        +bool handle_tool_errors
        +invoke(state) dict
        +_execute_tool_calls(messages) list
        +_handle_tool_call(call) ToolMessage
    }
    
    class ValidationNode {
        +Type schema
        +Callable error_formatter
        +invoke(state) dict
        +_validate_calls(messages) list
    }
    
    class AgentState {
        +messages: Annotated~list, add_messages~
        +remaining_steps: NotRequired~RemainingSteps~
    }
    
    class tools_condition {
        <<function>>
        +state: Union~list, dict, BaseModel~
        +messages_key: str
        +tools_condition() Literal
    }
    
    class InjectedState {
        <<Annotation>>
        +inject_state()
    }
    
    class InjectedStore {
        <<Annotation>>
        +inject_store()
    }
    
    create_react_agent --> AgentState : uses
    create_react_agent --> ToolNode : creates
    create_react_agent --> tools_condition : uses
    ToolNode --> InjectedState : supports
    ToolNode --> InjectedStore : supports
    ValidationNode --> ToolNode : extends
    
    style create_react_agent fill:#e1f5fe
    style ToolNode fill:#f3e5f5
    style AgentState fill:#e8f5e8
```

### 1.2 模块架构图

```mermaid
graph TB
    subgraph "LangGraph Prebuilt架构"
        subgraph "智能体层"
            ReactAgent[create_react_agent<br/>ReAct智能体创建器]
            AgentState[AgentState<br/>智能体状态模式]
            Prompt[Prompt<br/>提示词管理]
        end
        
        subgraph "工具层"
            ToolNode[ToolNode<br/>工具执行节点]
            ToolValidator[ValidationNode<br/>工具验证节点]
            ToolCondition[tools_condition<br/>工具路由条件]
        end
        
        subgraph "依赖注入层"
            StateInjection[InjectedState<br/>状态注入]
            StoreInjection[InjectedStore<br/>存储注入]
            DynamicTools[动态工具绑定]
        end
        
        subgraph "执行引擎层"
            StateGraph[StateGraph<br/>状态图]
            Pregel[Pregel执行器]
            Checkpoint[检查点系统]
        end
        
        subgraph "底层服务"
            LangChain[LangChain工具]
            LLMModels[语言模型]
            ExternalAPIs[外部API服务]
        end
    end
    
    %% 连接关系
    ReactAgent --> AgentState
    ReactAgent --> Prompt
    ReactAgent --> ToolNode
    
    ToolNode --> StateInjection
    ToolNode --> StoreInjection
    ToolNode --> DynamicTools
    ToolValidator --> ToolNode
    
    ReactAgent --> StateGraph
    StateGraph --> Pregel
    Pregel --> Checkpoint
    
    ToolNode --> LangChain
    ReactAgent --> LLMModels
    ToolNode --> ExternalAPIs
    
    style ReactAgent fill:#e1f5fe
    style ToolNode fill:#f3e5f5
    style StateGraph fill:#e8f5e8
    style LangChain fill:#fff3e0
```

## 2. create_react_agent：ReAct智能体创建器

### 2.1 ReAct模式工作流

```mermaid
sequenceDiagram
    participant User as 用户
    participant Agent as ReAct智能体
    participant LLM as 语言模型
    participant Tools as 工具节点
    participant State as 状态管理

    Note over User,State: ReAct智能体执行流程
    
    User->>Agent: 用户输入
    Agent->>State: 初始化状态
    State->>Agent: 返回初始状态
    
    loop ReAct循环
        Agent->>LLM: 发送消息和上下文
        LLM->>LLM: 推理(Reasoning)
        LLM->>LLM: 决定行动(Acting)
        LLM->>Agent: 返回响应和工具调用
        
        alt 有工具调用
            Agent->>Tools: 执行工具调用
            Tools->>Tools: 并行执行多个工具
            Tools->>Agent: 返回工具结果
            Agent->>State: 更新状态
            State->>Agent: 确认状态更新
        else 无工具调用
            Agent->>User: 返回最终答案
        end
        
        Note over Agent: 观察(Observing)结果
    end
    
    Agent->>User: 返回完整对话历史
```

### 2.2 create_react_agent核心实现

```python
def create_react_agent(
    model: Union[
        str,
        LanguageModelLike,
        Callable[[StateSchema, Runtime[ContextT]], BaseChatModel],
        Callable[[StateSchema, Runtime[ContextT]], Awaitable[BaseChatModel]],
    ],
    tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
    *,
    prompt: Optional[Prompt] = None,
    response_format: Optional[Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    post_model_hook: Optional[RunnableLike] = None,
    state_schema: Optional[StateSchemaType] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: Optional[str] = None,
    **deprecated_kwargs: Any,
) -> CompiledStateGraph:
    """创建一个在循环中调用工具的智能体图，直到满足停止条件
    
    该函数实现了ReAct（Reasoning and Acting）模式的智能体：
    1. 接收用户输入
    2. 使用语言模型进行推理
    3. 如果需要，调用工具获取信息
    4. 基于工具结果继续推理
    5. 返回最终答案
    
    Args:
        model: 智能体使用的语言模型，支持静态和动态模型选择
            - 静态模型：聊天模型实例或字符串标识符
            - 动态模型：签名为(state, runtime) -> BaseChatModel的可调用对象
        tools: 工具列表或ToolNode实例
        prompt: 可选的LLM提示词，支持多种格式
        response_format: 可选的最终智能体输出模式
        pre_model_hook: 在智能体节点前添加的可选节点
        post_model_hook: 在智能体节点后添加的可选节点  
        state_schema: 定义图状态的可选状态模式
        context_schema: 运行时上下文的可选模式
        checkpointer: 可选的检查点保存器对象
        store: 可选的存储对象
        interrupt_before: 要在之前中断的节点名称列表
        interrupt_after: 要在之后中断的节点名称列表
        debug: 是否启用调试模式的标志
        version: 要创建的图版本
        name: CompiledStateGraph的可选名称
        
    Returns:
        CompiledStateGraph: 编译后的可用于聊天交互的LangChain runnable
    """
    
    # 处理废弃参数
    if (config_schema := deprecated_kwargs.pop("config_schema", MISSING)) is not MISSING:
        warn(
            "`config_schema` is deprecated and will be removed in v2.0.0. "
            "Please use `context_schema` instead.",
            category=LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        if context_schema is None:
            context_schema = config_schema

    # 验证参数
    if deprecated_kwargs:
        raise ValueError(f"Unexpected arguments: {deprecated_kwargs}")
    
    if version not in ("v1", "v2"):
        raise ValueError(f"Version {version} is not supported")
    
    # 处理工具参数
    if isinstance(tools, ToolNode):
        tool_node = tools
        tool_classes = tools.tools_by_name
    else:
        tool_classes = {t.name if isinstance(t, BaseTool) else t.__name__: t for t in tools}
        tool_node = ToolNode(tools) if tools else None
    
    # 确定状态模式
    if state_schema is None:
        if response_format is not None:
            state_schema = cast(
                StateSchemaType,
                AgentStateWithStructuredResponse 
                if not inspect.isclass(context_schema) 
                else AgentStateWithStructuredResponsePydantic
            )
        else:
            state_schema = cast(
                StateSchemaType,
                AgentState 
                if not inspect.isclass(context_schema) 
                else AgentStatePydantic
            )
    
    # 创建状态图
    graph = StateGraph(
        state_schema=state_schema,
        context_schema=context_schema,
    )
    
    # 添加智能体节点
    agent_node = _create_agent_node(
        model=model,
        tools=list(tool_classes.values()) if tool_classes else [],
        prompt=prompt,
        state_schema=state_schema,
    )
    
    # 根据版本创建不同的图结构
    if version == "v1":
        return _create_react_agent_v1(
            graph=graph,
            agent_node=agent_node,
            tool_node=tool_node,
            pre_model_hook=pre_model_hook,
            post_model_hook=post_model_hook,
            response_format=response_format,
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
        )
    else:  # version == "v2"
        return _create_react_agent_v2(
            graph=graph,
            agent_node=agent_node,
            tool_classes=tool_classes,
            pre_model_hook=pre_model_hook,
            post_model_hook=post_model_hook,
            response_format=response_format,
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
        )

def _create_agent_node(
    model: Union[str, LanguageModelLike, Callable],
    tools: list[BaseTool],
    prompt: Optional[Prompt],
    state_schema: StateSchemaType,
) -> RunnableCallable:
    """创建智能体执行节点
    
    Args:
        model: 语言模型
        tools: 工具列表  
        prompt: 提示词
        state_schema: 状态模式
        
    Returns:
        RunnableCallable: 智能体节点
    """
    
    def agent_node(state: StateSchema, *, config: RunnableConfig = None, **kwargs) -> dict:
        """智能体节点执行函数
        
        Args:
            state: 当前图状态
            config: 运行配置
            
        Returns:
            dict: 状态更新
        """
        # 处理动态模型选择
        if callable(model) and not isinstance(model, (str, Runnable)):
            # 动态模型函数
            runtime = Runtime(context=config.get("configurable", {}) if config else {})
            if inspect.iscoroutinefunction(model):
                import asyncio
                current_model = asyncio.run(model(state, runtime))
            else:
                current_model = model(state, runtime)
        else:
            # 静态模型
            current_model = model
            
            # 如果是字符串，解析为模型实例
            if isinstance(current_model, str):
                current_model = _parse_model_string(current_model)
            
            # 绑定工具到模型
            if tools and hasattr(current_model, 'bind_tools'):
                current_model = current_model.bind_tools(tools)
        
        # 构建消息输入
        messages = state.get("messages", [])
        
        # 应用提示词
        if prompt is not None:
            if isinstance(prompt, str):
                system_message = SystemMessage(content=prompt)
                messages = [system_message] + list(messages)
            elif isinstance(prompt, SystemMessage):
                messages = [prompt] + list(messages)
            elif callable(prompt):
                prompt_input = prompt(state)
                if isinstance(prompt_input, str):
                    system_message = SystemMessage(content=prompt_input)
                    messages = [system_message] + list(messages)
                elif isinstance(prompt_input, list):
                    messages = prompt_input + list(messages)
                else:
                    messages = [prompt_input] + list(messages)
            elif hasattr(prompt, 'invoke'):
                # Runnable prompt
                prompt_result = prompt.invoke(state, config)
                if isinstance(prompt_result, str):
                    system_message = SystemMessage(content=prompt_result)
                    messages = [system_message] + list(messages)
                else:
                    messages = [prompt_result] + list(messages)
        
        # 调用模型
        response = current_model.invoke(messages, config)
        
        # 检查剩余步数
        remaining_steps = state.get("remaining_steps")
        if remaining_steps is not None and hasattr(remaining_steps, 'value'):
            if remaining_steps.value <= 1:
                # 步数用尽，强制结束
                if isinstance(response, AIMessage) and response.tool_calls:
                    # 移除工具调用
                    response.tool_calls = []
                    response.content = (
                        response.content + "\n\n(Stopped due to step limit)"
                        if response.content 
                        else "(Stopped due to step limit)"
                    )
        
        return {"messages": [response]}
    
    return RunnableCallable(agent_node, name="agent")

def _create_react_agent_v1(
    graph: StateGraph,
    agent_node: RunnableCallable,
    tool_node: Optional[ToolNode],
    **kwargs
) -> CompiledStateGraph:
    """创建v1版本的ReAct智能体图
    
    v1版本特点：
    - 工具节点处理单个消息
    - 消息中的所有工具调用在工具节点内并行执行
    """
    # 添加前置钩子
    if kwargs.get("pre_model_hook"):
        graph.add_node("pre_hook", kwargs["pre_model_hook"])
        graph.set_entry_point("pre_hook")
        graph.add_edge("pre_hook", "agent")
    else:
        graph.set_entry_point("agent")
    
    # 添加智能体节点
    graph.add_node("agent", agent_node)
    
    # 添加后置钩子
    if kwargs.get("post_model_hook"):
        graph.add_node("post_hook", kwargs["post_model_hook"])
        graph.add_edge("agent", "post_hook")
        
        if tool_node:
            graph.add_node("tools", tool_node)
            graph.add_conditional_edges(
                "post_hook",
                tools_condition,
                {"tools": "tools", END: END}
            )
            graph.add_edge("tools", "agent")
        else:
            graph.set_finish_point("post_hook")
    else:
        if tool_node:
            graph.add_node("tools", tool_node)
            graph.add_conditional_edges(
                "agent",
                tools_condition,
                {"tools": "tools", END: END}
            )
            graph.add_edge("tools", "agent")
        else:
            graph.set_finish_point("agent")
    
    # 添加结构化响应处理
    if kwargs.get("response_format"):
        _add_structured_response_node(graph, kwargs["response_format"])
    
    # 编译图
    return graph.compile(
        checkpointer=kwargs.get("checkpointer"),
        store=kwargs.get("store"),
        interrupt_before=kwargs.get("interrupt_before"),
        interrupt_after=kwargs.get("interrupt_after"),
        debug=kwargs.get("debug", False),
        name=kwargs.get("name"),
    )

def _create_react_agent_v2(
    graph: StateGraph,
    agent_node: RunnableCallable,
    tool_classes: dict[str, BaseTool],
    **kwargs
) -> CompiledStateGraph:
    """创建v2版本的ReAct智能体图
    
    v2版本特点：
    - 工具节点处理单个工具调用
    - 工具调用使用Send API分发到多个工具节点实例
    - 支持更细粒度的工具执行控制
    """
    
    def route_tools(state: StateSchema) -> list[Send]:
        """路由工具调用到不同的工具节点实例"""
        messages = state.get("messages", [])
        if not messages:
            return []
        
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return []
        
        sends = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in tool_classes:
                # 为每个工具调用创建一个Send命令
                sends.append(Send(f"tool_{tool_name}", {
                    "messages": [last_message],
                    "tool_call_id": tool_call["id"]
                }))
            else:
                # 处理无效工具名称
                error_message = ToolMessage(
                    content=f"Error: {tool_name} is not a valid tool",
                    tool_call_id=tool_call["id"]
                )
                sends.append(Send("agent", {"messages": [error_message]}))
        
        return sends
    
    # 添加节点和边
    if kwargs.get("pre_model_hook"):
        graph.add_node("pre_hook", kwargs["pre_model_hook"])
        graph.set_entry_point("pre_hook")
        graph.add_edge("pre_hook", "agent")
    else:
        graph.set_entry_point("agent")
    
    graph.add_node("agent", agent_node)
    
    if kwargs.get("post_model_hook"):
        graph.add_node("post_hook", kwargs["post_model_hook"])
        graph.add_edge("agent", "post_hook")
        
        if tool_classes:
            # 添加工具路由
            graph.add_conditional_edges("post_hook", route_tools)
            
            # 为每个工具添加节点
            for tool_name, tool in tool_classes.items():
                tool_node = ToolNode([tool])
                graph.add_node(f"tool_{tool_name}", tool_node)
                graph.add_edge(f"tool_{tool_name}", "agent")
        else:
            graph.set_finish_point("post_hook")
    else:
        if tool_classes:
            graph.add_conditional_edges("agent", route_tools)
            
            for tool_name, tool in tool_classes.items():
                tool_node = ToolNode([tool])
                graph.add_node(f"tool_{tool_name}", tool_node)
                graph.add_edge(f"tool_{tool_name}", "agent")
        else:
            graph.set_finish_point("agent")
    
    # 添加结构化响应处理
    if kwargs.get("response_format"):
        _add_structured_response_node(graph, kwargs["response_format"])
    
    return graph.compile(
        checkpointer=kwargs.get("checkpointer"),
        store=kwargs.get("store"),
        interrupt_before=kwargs.get("interrupt_before"),
        interrupt_after=kwargs.get("interrupt_after"),
        debug=kwargs.get("debug", False),
        name=kwargs.get("name"),
    )
```

### 2.3 AgentState：智能体状态模式

```python
class AgentState(TypedDict):
    """智能体的状态定义
    
    这个状态模式定义了ReAct智能体运行过程中需要维护的状态信息：
    - messages：对话消息历史
    - remaining_steps：剩余执行步数
    """
    
    messages: Annotated[Sequence[BaseMessage], add_messages]
    """对话消息序列
    
    使用add_messages reducer确保消息正确累积：
    - 新消息会添加到现有消息列表末尾
    - 支持消息去重和更新
    - 自动处理RemoveMessage指令
    """
    
    remaining_steps: NotRequired[RemainingSteps]
    """剩余执行步数（可选）
    
    用于防止智能体无限循环：
    - 每次调用工具后递减
    - 达到0时强制结束执行
    - 支持动态调整
    """

class AgentStatePydantic(BaseModel):
    """Pydantic版本的智能体状态
    
    提供更强的类型验证和序列化支持
    """
    
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps = 25
    
    class Config:
        arbitrary_types_allowed = True

class AgentStateWithStructuredResponse(AgentState):
    """带结构化响应的智能体状态
    
    当智能体需要返回结构化输出时使用
    """
    
    structured_response: StructuredResponse
    """结构化响应数据
    
    可以是dict或BaseModel实例，用于存储
    智能体生成的结构化输出结果
    """

# 状态管理辅助函数
def validate_agent_state(state: dict[str, Any]) -> AgentState:
    """验证智能体状态格式
    
    Args:
        state: 待验证的状态字典
        
    Returns:
        AgentState: 验证后的状态
        
    Raises:
        ValueError: 状态格式不正确时
    """
    if "messages" not in state:
        raise ValueError("Agent state must contain 'messages' key")
    
    messages = state["messages"]
    if not isinstance(messages, (list, tuple)):
        raise ValueError("Messages must be a sequence")
    
    # 验证消息格式
    validated_messages = []
    for msg in messages:
        if not isinstance(msg, BaseMessage):
            # 尝试转换
            validated_messages.extend(convert_to_messages([msg]))
        else:
            validated_messages.append(msg)
    
    result = AgentState(messages=validated_messages)
    
    # 添加可选字段
    if "remaining_steps" in state:
        result["remaining_steps"] = state["remaining_steps"]
    
    if "structured_response" in state:
        result = AgentStateWithStructuredResponse(
            messages=result["messages"],
            structured_response=state["structured_response"]
        )
        if "remaining_steps" in state:
            result["remaining_steps"] = state["remaining_steps"]
    
    return result
```

## 3. ToolNode：工具执行节点

### 3.1 ToolNode核心架构

```python
class ToolNode(RunnableCallable):
    """LangGraph工作流中的工具执行节点
    
    该类实现了以下关键设计模式：
    - 多个工具调用的并行执行以提高效率
    - 可自定义错误消息的健壮错误处理
    - 需要访问图状态的工具的状态注入
    - 需要持久存储的工具的存储注入
    - 用于高级控制流的基于命令的状态更新
    """
    
    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        handle_tool_errors: Optional[bool] = True,
        fallbacks: Optional[dict[str, Union[str, Callable, Runnable]]] = None,
    ) -> None:
        """初始化工具节点
        
        Args:
            tools: 工具序列，支持多种格式
            name: 节点名称
            tags: 节点标签
            handle_tool_errors: 是否处理工具错误
            fallbacks: 工具调用失败时的回退策略
        """
        super().__init__(self._func, name=name, tags=tags, trace=False)
        
        # 处理工具列表
        self.tools_by_name: dict[str, BaseTool] = {}
        for tool in tools:
            if isinstance(tool, BaseTool):
                self.tools_by_name[tool.name] = tool
            elif callable(tool):
                # 转换函数为工具
                tool_obj = create_tool(tool)
                self.tools_by_name[tool_obj.name] = tool_obj
            elif isinstance(tool, dict):
                # 从字典创建工具
                tool_obj = BaseTool.from_dict(tool)
                self.tools_by_name[tool_obj.name] = tool_obj
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
        
        self.handle_tool_errors = handle_tool_errors
        self.fallbacks = fallbacks or {}
        
        # 分析工具的特殊参数需求
        self._analyze_tool_signatures()
    
    def _analyze_tool_signatures(self) -> None:
        """分析工具签名，识别特殊参数需求"""
        self._tools_with_state: dict[str, list[str]] = {}
        self._tools_with_store: dict[str, list[str]] = {}
        
        for tool_name, tool in self.tools_by_name.items():
            # 获取工具函数签名
            if hasattr(tool, 'func'):
                func = tool.func
            else:
                func = tool
            
            try:
                sig = inspect.signature(func)
                type_hints = get_type_hints(func, include_extras=True)
                
                state_params = []
                store_params = []
                
                for param_name, param in sig.parameters.items():
                    param_type = type_hints.get(param_name, param.annotation)
                    
                    # 检查状态注入
                    if self._is_injected_state(param_type):
                        state_params.append(param_name)
                    
                    # 检查存储注入
                    elif self._is_injected_store(param_type):
                        store_params.append(param_name)
                
                if state_params:
                    self._tools_with_state[tool_name] = state_params
                
                if store_params:
                    self._tools_with_store[tool_name] = store_params
                    
            except (ValueError, TypeError):
                # 无法分析签名，跳过
                continue
    
    def _is_injected_state(self, param_type) -> bool:
        """检查参数是否为注入状态类型"""
        return (
            get_origin(param_type) is Annotated and
            any(isinstance(arg, InjectedState) for arg in get_args(param_type)[1:])
        ) or param_type is InjectedState
    
    def _is_injected_store(self, param_type) -> bool:
        """检查参数是否为注入存储类型"""
        return (
            get_origin(param_type) is Annotated and
            any(isinstance(arg, InjectedStore) for arg in get_args(param_type)[1:])
        ) or param_type is InjectedStore
    
    def _func(
        self,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Any:
        """工具节点的主执行函数
        
        Args:
            input: 输入消息或状态
            config: 运行配置
            **kwargs: 额外参数
            
        Returns:
            执行结果
        """
        # 标准化输入格式
        input_type, message_list = self._parse_input(input)
        
        # 提取工具调用
        tool_calls = []
        for message in message_list:
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_calls.extend(message.tool_calls)
        
        if not tool_calls:
            # 无工具调用，返回空结果
            return self._format_output([], input_type)
        
        # 执行工具调用
        tool_messages = self._execute_tool_calls(
            tool_calls=tool_calls,
            config=config,
            state=input if input_type != "list" else None,
            **kwargs
        )
        
        return self._format_output(tool_messages, input_type)
    
    def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        config: RunnableConfig,
        state: Optional[Union[dict[str, Any], BaseModel]] = None,
        **kwargs: Any,
    ) -> list[ToolMessage]:
        """并行执行工具调用
        
        Args:
            tool_calls: 工具调用列表
            config: 运行配置
            state: 图状态（用于状态注入）
            **kwargs: 额外参数
            
        Returns:
            list[ToolMessage]: 工具执行结果消息列表
        """
        # 获取执行器
        executor = get_executor_for_config(config)
        
        # 创建执行任务
        with executor:
            # 为每个工具调用创建Future
            futures = []
            for tool_call in tool_calls:
                future = executor.submit(
                    self._execute_single_tool_call,
                    tool_call=tool_call,
                    config=config,
                    state=state,
                    **kwargs
                )
                futures.append((tool_call, future))
            
            # 收集结果
            tool_messages = []
            for tool_call, future in futures:
                try:
                    result = future.result()
                    tool_messages.append(result)
                except Exception as e:
                    # 处理执行错误
                    error_message = self._handle_tool_error(tool_call, e)
                    tool_messages.append(error_message)
        
        return tool_messages
    
    def _execute_single_tool_call(
        self,
        tool_call: ToolCall,
        config: RunnableConfig,
        state: Optional[Union[dict[str, Any], BaseModel]] = None,
        **kwargs: Any,
    ) -> ToolMessage:
        """执行单个工具调用
        
        Args:
            tool_call: 工具调用信息
            config: 运行配置
            state: 图状态
            **kwargs: 额外参数
            
        Returns:
            ToolMessage: 工具执行结果消息
        """
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call["id"]
        
        # 检查工具是否存在
        if tool_name not in self.tools_by_name:
            available_tools = ", ".join(self.tools_by_name.keys())
            error_content = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                requested_tool=tool_name,
                available_tools=available_tools
            )
            return ToolMessage(
                content=error_content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        
        tool = self.tools_by_name[tool_name]
        
        try:
            # 处理参数注入
            final_args = tool_args.copy()
            
            # 状态注入
            if tool_name in self._tools_with_state and state is not None:
                state_params = self._tools_with_state[tool_name]
                for param_name in state_params:
                    final_args[param_name] = state
            
            # 存储注入
            if tool_name in self._tools_with_store:
                store = kwargs.get('store') or config.get('configurable', {}).get('store')
                if store:
                    store_params = self._tools_with_store[tool_name]
                    for param_name in store_params:
                        final_args[param_name] = store
            
            # 调用工具
            if hasattr(tool, 'ainvoke') and inspect.iscoroutinefunction(tool.ainvoke):
                # 异步工具
                import asyncio
                result = asyncio.run(tool.ainvoke(final_args, config))
            else:
                # 同步工具
                result = tool.invoke(final_args, config)
            
            # 处理结果
            content = msg_content_output(result)
            
            return ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
            
        except Exception as e:
            if self.handle_tool_errors:
                # 生成错误消息
                error_content = TOOL_CALL_ERROR_TEMPLATE.format(error=str(e))
                return ToolMessage(
                    content=error_content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            else:
                # 重新抛出异常
                raise
    
    def _handle_tool_error(self, tool_call: ToolCall, error: Exception) -> ToolMessage:
        """处理工具执行错误
        
        Args:
            tool_call: 失败的工具调用
            error: 错误实例
            
        Returns:
            ToolMessage: 错误消息
        """
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        
        # 检查是否有回退策略
        if tool_name in self.fallbacks:
            fallback = self.fallbacks[tool_name]
            
            try:
                if isinstance(fallback, str):
                    # 字符串回退
                    content = fallback
                elif callable(fallback):
                    # 可调用回退
                    content = fallback(tool_call, error)
                elif hasattr(fallback, 'invoke'):
                    # Runnable回退
                    content = fallback.invoke({
                        "tool_call": tool_call,
                        "error": error
                    })
                else:
                    content = str(fallback)
                
                return ToolMessage(
                    content=msg_content_output(content),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
                
            except Exception as fallback_error:
                # 回退也失败了
                error_content = (
                    f"Tool execution failed: {error}\n"
                    f"Fallback also failed: {fallback_error}"
                )
        else:
            # 没有回退策略
            error_content = TOOL_CALL_ERROR_TEMPLATE.format(error=str(error))
        
        return ToolMessage(
            content=error_content,
            tool_call_id=tool_call_id,
            name=tool_name,
        )
    
    def _parse_input(self, input: Any) -> tuple[str, list[AnyMessage]]:
        """解析输入格式
        
        Args:
            input: 输入数据
            
        Returns:
            tuple: (输入类型, 消息列表)
        """
        if isinstance(input, list):
            return "list", input
        elif isinstance(input, dict):
            if "messages" in input:
                return "dict", input["messages"]
            else:
                raise ValueError("Dictionary input must contain 'messages' key")
        elif isinstance(input, BaseModel):
            if hasattr(input, 'messages'):
                return "dict", input.messages  
            else:
                raise ValueError("BaseModel input must have 'messages' attribute")
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    def _format_output(self, tool_messages: list[ToolMessage], input_type: str) -> Any:
        """格式化输出结果
        
        Args:
            tool_messages: 工具消息列表
            input_type: 输入类型
            
        Returns:
            格式化后的输出
        """
        if input_type == "list":
            return tool_messages
        else:
            return {"messages": tool_messages}

def msg_content_output(output: Any) -> Union[str, list[dict]]:
    """将工具输出转换为有效的消息内容格式
    
    LangChain ToolMessages接受字符串内容或内容块列表。
    此函数确保工具输出正确格式化以供消息消费，
    尽可能保留结构化数据，回退到JSON序列化或字符串转换。
    
    Args:
        output: 工具执行的原始输出，可以是任何类型
        
    Returns:
        Union[str, list[dict]]: 输出的字符串表示或内容块列表
    """
    if isinstance(output, str):
        return output
    elif isinstance(output, (dict, list)):
        # 尝试保持结构化格式
        if isinstance(output, list) and all(
            isinstance(item, dict) and 
            item.get("type") in TOOL_MESSAGE_BLOCK_TYPES 
            for item in output
        ):
            # 已经是有效的内容块格式
            return output
        else:
            # 转换为JSON字符串
            import json
            try:
                return json.dumps(output, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                return str(output)
    else:
        # 其他类型转换为字符串
        return str(output)
```

### 3.2 依赖注入机制

```python
class InjectedState:
    """状态注入注解类
    
    用于标记工具参数需要注入当前图状态
    """
    
    def __init__(self, *, key: Optional[str] = None):
        """初始化状态注入器
        
        Args:
            key: 可选的状态键，如果指定则只注入特定键的值
        """
        self.key = key
    
    def __repr__(self) -> str:
        return f"InjectedState(key={self.key!r})"

class InjectedStore:
    """存储注入注解类
    
    用于标记工具参数需要注入持久存储实例
    """
    
    def __init__(self, *, namespace: Optional[str] = None):
        """初始化存储注入器
        
        Args:
            namespace: 可选的命名空间，用于隔离不同工具的存储
        """
        self.namespace = namespace
    
    def __repr__(self) -> str:
        return f"InjectedStore(namespace={self.namespace!r})"

# 使用示例
def weather_tool_with_state(
    location: str,
    state: Annotated[dict, InjectedState()],
    store: Annotated[BaseStore, InjectedStore(namespace="weather")],
) -> str:
    """带状态和存储注入的天气工具
    
    Args:
        location: 查询位置
        state: 注入的图状态
        store: 注入的持久存储
        
    Returns:
        str: 天气信息
    """
    # 访问图状态
    user_preferences = state.get("user_preferences", {})
    temperature_unit = user_preferences.get("temperature_unit", "celsius")
    
    # 访问持久存储
    cached_weather = store.get(("weather_cache", location))
    if cached_weather and not _is_cache_expired(cached_weather):
        return cached_weather["data"]
    
    # 调用天气API
    weather_data = _fetch_weather(location, temperature_unit)
    
    # 缓存结果
    store.put(("weather_cache", location), {
        "data": weather_data,
        "timestamp": time.time()
    })
    
    return weather_data

def _is_cache_expired(cached_data: dict, max_age: int = 3600) -> bool:
    """检查缓存是否过期"""
    return time.time() - cached_data.get("timestamp", 0) > max_age

def _fetch_weather(location: str, unit: str) -> str:
    """模拟天气API调用"""
    import random
    temp = random.randint(-10, 35) if unit == "celsius" else random.randint(14, 95)
    return f"The weather in {location} is {temp}°{unit[0].upper()}"
```

### 3.3 tools_condition：工具路由条件

```python
def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """工具调用工作流的条件路由函数
    
    此实用函数实现ReAct风格智能体的标准条件逻辑：
    如果最后一条AI消息包含工具调用，则路由到工具执行节点；
    否则，结束工作流。这种模式是大多数工具调用智能体架构的基础。
    
    该函数处理LangGraph应用程序中常用的多种状态格式，
    使其在不同图设计中具有灵活性，同时保持一致的行为。
    
    Args:
        state: 要检查工具调用的当前图状态。支持的格式：
            - 包含messages键的字典（用于StateGraph）
            - 具有messages属性的BaseModel实例
        messages_key: 包含状态中消息列表的键或属性名称。
            这允许使用不同状态模式的图进行自定义。
            默认为"messages"。
            
    Returns:
        如果最后一条AI消息中存在工具调用，则返回"tools"，
        否则返回"__end__"以终止工作流。这些是工具调用条件边的标准路由目标。
        
    Raises:
        ValueError: 如果在提供的状态格式中找不到消息。
        
    Example:
        在ReAct智能体中的基本用法：
        
        ```python
        from langgraph.graph import StateGraph  
        from langgraph.prebuilt import ToolNode, tools_condition
        from typing_extensions import TypedDict
        
        class State(TypedDict):
            messages: list
            
        graph = StateGraph(State)
        graph.add_node("llm", call_model)
        graph.add_node("tools", ToolNode([my_tool]))
        graph.add_conditional_edges(
            "llm",
            tools_condition,  # 路由到"tools"或"__end__"
            {"tools": "tools", "__end__": "__end__"}
        )
        ```
        
        自定义消息键：
        
        ```python
        def custom_condition(state):
            return tools_condition(state, messages_key="chat_history")
        ```
    """
    # 处理不同的状态格式
    if isinstance(state, list):
        # 状态本身就是消息列表
        messages = state
    elif isinstance(state, dict):
        # 字典状态格式
        if messages_key not in state:
            raise ValueError(f"No '{messages_key}' key found in state")
        messages = state[messages_key]
    elif isinstance(state, BaseModel):
        # Pydantic模型状态格式
        if not hasattr(state, messages_key):
            raise ValueError(f"No '{messages_key}' attribute found in state")
        messages = getattr(state, messages_key)
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")
    
    # 检查消息列表
    if not messages:
        return "__end__"
    
    # 获取最后一条消息
    last_message = messages[-1]
    
    # 检查是否为带工具调用的AI消息
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    else:
        return "__end__"

# 扩展版本的条件函数
def advanced_tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    *,
    messages_key: str = "messages",
    max_iterations: Optional[int] = None,
    check_remaining_steps: bool = True,
) -> Literal["tools", "__end__"]:
    """高级工具条件函数，支持更多控制选项
    
    Args:
        state: 图状态
        messages_key: 消息键名
        max_iterations: 最大迭代次数
        check_remaining_steps: 是否检查剩余步数
        
    Returns:
        路由目标："tools"或"__end__"
    """
    # 基础工具调用检查
    basic_result = tools_condition(state, messages_key)
    
    if basic_result == "__end__":
        return "__end__"
    
    # 高级检查
    if isinstance(state, dict):
        # 检查剩余步数
        if check_remaining_steps and "remaining_steps" in state:
            remaining = state["remaining_steps"]
            if hasattr(remaining, 'value') and remaining.value <= 0:
                return "__end__"
        
        # 检查最大迭代次数
        if max_iterations is not None:
            messages = state.get(messages_key, [])
            tool_call_count = sum(
                1 for msg in messages 
                if isinstance(msg, AIMessage) and msg.tool_calls
            )
            if tool_call_count >= max_iterations:
                return "__end__"
    
    return "tools"

# 工具条件的工厂函数
def create_tools_condition(
    *,
    messages_key: str = "messages",
    max_iterations: Optional[int] = None,
    required_tools: Optional[set[str]] = None,
    forbidden_tools: Optional[set[str]] = None,
) -> Callable:
    """创建自定义工具条件函数
    
    Args:
        messages_key: 消息键名
        max_iterations: 最大迭代次数
        required_tools: 必需工具集合（如果指定，只有这些工具被允许）
        forbidden_tools: 禁止工具集合
        
    Returns:
        Callable: 自定义工具条件函数
    """
    
    def custom_condition(state) -> Literal["tools", "__end__"]:
        # 基础检查
        if tools_condition(state, messages_key) == "__end__":
            return "__end__"
        
        # 获取消息
        if isinstance(state, list):
            messages = state
        elif isinstance(state, dict):
            messages = state.get(messages_key, [])
        else:
            messages = getattr(state, messages_key, [])
        
        if not messages:
            return "__end__"
        
        last_message = messages[-1]
        if not (isinstance(last_message, AIMessage) and last_message.tool_calls):
            return "__end__"
        
        # 检查工具限制
        if required_tools or forbidden_tools:
            tool_names = {call["name"] for call in last_message.tool_calls}
            
            if required_tools and not tool_names.issubset(required_tools):
                # 有不被允许的工具
                return "__end__"
            
            if forbidden_tools and tool_names.intersection(forbidden_tools):
                # 有被禁止的工具
                return "__end__"
        
        # 检查迭代次数
        if max_iterations is not None:
            tool_call_count = sum(
                1 for msg in messages
                if isinstance(msg, AIMessage) and msg.tool_calls
            )
            if tool_call_count >= max_iterations:
                return "__end__"
        
        return "tools"
    
    return custom_condition
```

## 4. ValidationNode：工具验证节点

### 4.1 ValidationNode实现

```python
class ValidationNode(RunnableCallable):
    """工具调用验证节点
    
    该节点在工具执行前验证工具调用参数，确保：
    - 参数类型正确
    - 必需参数存在
    - 参数值有效
    - 符合工具的输入模式
    """
    
    def __init__(
        self,
        schemas: dict[str, Union[Type[BaseModel], Type[BaseModelV1]]],
        *,
        name: str = "validation",
        format_error: Optional[Callable[[BaseException, ToolCall, Any], str]] = None,
        handle_validation_errors: bool = True,
    ) -> None:
        """初始化验证节点
        
        Args:
            schemas: 工具名称到验证模式的映射
            name: 节点名称
            format_error: 自定义错误格式化函数
            handle_validation_errors: 是否处理验证错误
        """
        super().__init__(self._func, name=name, trace=False)
        
        self.schemas = schemas
        self.format_error = format_error or _default_format_error
        self.handle_validation_errors = handle_validation_errors
    
    def _func(
        self,
        input: Union[list[AnyMessage], dict[str, Any], BaseModel],
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Any:
        """验证节点执行函数
        
        Args:
            input: 输入消息或状态
            config: 运行配置
            **kwargs: 额外参数
            
        Returns:
            验证结果
        """
        # 解析输入
        input_type, messages = self._parse_input(input)
        
        # 查找需要验证的工具调用
        tool_calls_to_validate = []
        validated_messages = []
        
        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                # 验证工具调用
                new_tool_calls = []
                validation_errors = []
                
                for tool_call in message.tool_calls:
                    try:
                        validated_call = self._validate_tool_call(tool_call)
                        new_tool_calls.append(validated_call)
                    except Exception as e:
                        if self.handle_validation_errors:
                            # 创建错误消息
                            error_msg = ToolMessage(
                                content=self.format_error(
                                    e, tool_call, self.schemas.get(tool_call["name"])
                                ),
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            )
                            validation_errors.append(error_msg)
                        else:
                            raise
                
                # 创建新的AI消息（如果有有效的工具调用）
                if new_tool_calls:
                    new_message = AIMessage(
                        content=message.content,
                        tool_calls=new_tool_calls,
                        id=message.id,
                    )
                    validated_messages.append(new_message)
                
                # 添加验证错误消息
                validated_messages.extend(validation_errors)
                
            else:
                # 非工具调用消息，直接添加
                validated_messages.append(message)
        
        # 格式化输出
        return self._format_output(validated_messages, input_type)
    
    def _validate_tool_call(self, tool_call: ToolCall) -> ToolCall:
        """验证单个工具调用
        
        Args:
            tool_call: 工具调用信息
            
        Returns:
            ToolCall: 验证后的工具调用
            
        Raises:
            ValidationError: 验证失败时
        """
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        
        # 检查是否有对应的模式
        if tool_name not in self.schemas:
            # 没有模式，跳过验证
            return tool_call
        
        schema = self.schemas[tool_name]
        
        try:
            # 使用Pydantic模式验证参数
            if hasattr(schema, 'model_validate'):
                # Pydantic v2
                validated_args = schema.model_validate(tool_args).model_dump()
            else:
                # Pydantic v1
                validated_args = schema(**tool_args).dict()
            
            # 返回验证后的工具调用
            return {
                **tool_call,
                "args": validated_args
            }
            
        except Exception as e:
            # 验证失败
            raise ValidationError(f"Tool '{tool_name}' validation failed: {e}") from e
    
    def _parse_input(self, input: Any) -> tuple[str, list[AnyMessage]]:
        """解析输入格式"""
        if isinstance(input, list):
            return "list", input
        elif isinstance(input, dict):
            return "dict", input.get("messages", [])
        elif isinstance(input, BaseModel):
            return "dict", getattr(input, 'messages', [])
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    def _format_output(self, messages: list[AnyMessage], input_type: str) -> Any:
        """格式化输出"""
        if input_type == "list":
            return messages
        else:
            return {"messages": messages}

def _default_format_error(
    error: BaseException,
    call: ToolCall,
    schema: Union[Type[BaseModel], Type[BaseModelV1]],
) -> str:
    """默认错误格式化函数
    
    Args:
        error: 验证错误
        call: 失败的工具调用
        schema: 验证模式
        
    Returns:
        str: 格式化的错误消息
    """
    tool_name = call["name"]
    
    error_msg = f"Tool '{tool_name}' validation error: {repr(error)}"
    
    # 添加模式信息
    if schema:
        if hasattr(schema, 'model_fields'):
            # Pydantic v2
            fields_info = []
            for field_name, field_info in schema.model_fields.items():
                required = field_info.is_required()
                field_type = field_info.annotation
                fields_info.append(f"  - {field_name}: {field_type} {'(required)' if required else '(optional)'}")
            
            if fields_info:
                error_msg += f"\n\nExpected parameters:\n" + "\n".join(fields_info)
        
        elif hasattr(schema, '__fields__'):
            # Pydantic v1
            fields_info = []
            for field_name, field in schema.__fields__.items():
                required = field.required
                field_type = field.type_
                fields_info.append(f"  - {field_name}: {field_type} {'(required)' if required else '(optional)'}")
            
            if fields_info:
                error_msg += f"\n\nExpected parameters:\n" + "\n".join(fields_info)
    
    error_msg += "\n\nPlease fix the validation errors and try again."
    
    return error_msg

class ValidationError(Exception):
    """工具验证错误"""
    pass

# 验证节点工厂函数
def create_validation_node(
    tools: Sequence[BaseTool],
    *,
    name: str = "validation",
    auto_generate_schemas: bool = True,
    custom_schemas: Optional[dict[str, Type[BaseModel]]] = None,
    format_error: Optional[Callable] = None,
) -> ValidationNode:
    """创建验证节点的工厂函数
    
    Args:
        tools: 工具列表
        name: 节点名称
        auto_generate_schemas: 是否自动生成验证模式
        custom_schemas: 自定义验证模式
        format_error: 自定义错误格式化函数
        
    Returns:
        ValidationNode: 验证节点实例
    """
    schemas = custom_schemas or {}
    
    if auto_generate_schemas:
        for tool in tools:
            if tool.name not in schemas:
                # 尝试从工具生成模式
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    schemas[tool.name] = tool.args_schema
                elif hasattr(tool, '_schema'):
                    schemas[tool.name] = tool._schema
    
    return ValidationNode(
        schemas=schemas,
        name=name,
        format_error=format_error,
    )
```

## 5. 高级特性与扩展

### 5.1 动态工具绑定

```python
class DynamicToolAgent:
    """支持动态工具绑定的智能体"""
    
    def __init__(
        self,
        base_model: BaseChatModel,
        tool_registry: dict[str, BaseTool],
    ):
        self.base_model = base_model
        self.tool_registry = tool_registry
    
    def create_agent_with_tools(
        self,
        tool_names: list[str],
        context: Optional[dict] = None,
    ) -> CompiledStateGraph:
        """根据指定工具动态创建智能体
        
        Args:
            tool_names: 要绑定的工具名称列表
            context: 上下文信息
            
        Returns:
            CompiledStateGraph: 动态创建的智能体图
        """
        # 选择工具
        selected_tools = []
        for name in tool_names:
            if name in self.tool_registry:
                tool = self.tool_registry[name]
                
                # 根据上下文定制工具
                if context and hasattr(tool, 'with_config'):
                    tool = tool.with_config({"context": context})
                
                selected_tools.append(tool)
        
        # 创建动态模型函数
        def dynamic_model(state, runtime):
            """动态模型选择函数"""
            # 根据状态选择合适的模型配置
            model_config = self._select_model_config(state, runtime)
            
            # 绑定选择的工具
            if selected_tools:
                return model_config.bind_tools(selected_tools)
            else:
                return model_config
        
        # 创建智能体
        return create_react_agent(
            model=dynamic_model,
            tools=selected_tools,
            prompt=self._create_dynamic_prompt(tool_names, context),
        )
    
    def _select_model_config(self, state, runtime) -> BaseChatModel:
        """根据状态选择模型配置"""
        # 示例逻辑：根据消息长度选择模型
        messages = state.get("messages", [])
        total_tokens = sum(len(str(msg)) for msg in messages)
        
        if total_tokens > 10000:
            # 长对话使用更强的模型
            return self.base_model.with_config({"model": "gpt-4"})
        else:
            # 短对话使用效率模型
            return self.base_model.with_config({"model": "gpt-3.5-turbo"})
    
    def _create_dynamic_prompt(
        self, 
        tool_names: list[str], 
        context: Optional[dict]
    ) -> str:
        """创建动态提示词"""
        base_prompt = "You are a helpful assistant with access to the following tools:\n"
        
        # 添加工具描述
        for name in tool_names:
            if name in self.tool_registry:
                tool = self.tool_registry[name]
                base_prompt += f"- {name}: {tool.description}\n"
        
        # 添加上下文信息
        if context:
            base_prompt += f"\nContext: {context}\n"
        
        base_prompt += "\nUse tools when appropriate to help answer questions."
        
        return base_prompt

# 工具注册表
class ToolRegistry:
    """工具注册表，支持动态注册和发现"""
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._categories: dict[str, set[str]] = {}
        self._dependencies: dict[str, set[str]] = {}
    
    def register(
        self,
        tool: BaseTool,
        *,
        category: Optional[str] = None,
        dependencies: Optional[list[str]] = None,
    ) -> None:
        """注册工具
        
        Args:
            tool: 要注册的工具
            category: 工具类别
            dependencies: 工具依赖列表
        """
        self._tools[tool.name] = tool
        
        if category:
            if category not in self._categories:
                self._categories[category] = set()
            self._categories[category].add(tool.name)
        
        if dependencies:
            self._dependencies[tool.name] = set(dependencies)
    
    def get_tools_by_category(self, category: str) -> list[BaseTool]:
        """根据类别获取工具"""
        if category not in self._categories:
            return []
        
        return [
            self._tools[name] 
            for name in self._categories[category]
            if name in self._tools
        ]
    
    def resolve_dependencies(self, tool_names: list[str]) -> list[str]:
        """解析工具依赖关系"""
        resolved = set()
        to_resolve = set(tool_names)
        
        while to_resolve:
            tool_name = to_resolve.pop()
            if tool_name in resolved:
                continue
            
            resolved.add(tool_name)
            
            # 添加依赖工具
            if tool_name in self._dependencies:
                for dep in self._dependencies[tool_name]:
                    if dep not in resolved:
                        to_resolve.add(dep)
        
        return list(resolved)
```

### 5.2 工具执行监控

```python
class MonitoredToolNode(ToolNode):
    """带监控的工具节点"""
    
    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
        *,
        monitor_callback: Optional[Callable] = None,
        performance_tracker: Optional["PerformanceTracker"] = None,
        **kwargs
    ):
        super().__init__(tools, **kwargs)
        self.monitor_callback = monitor_callback
        self.performance_tracker = performance_tracker or PerformanceTracker()
    
    def _execute_single_tool_call(
        self,
        tool_call: ToolCall,
        config: RunnableConfig,
        **kwargs
    ) -> ToolMessage:
        """带监控的工具执行"""
        import time
        
        tool_name = tool_call["name"]
        start_time = time.time()
        
        try:
            # 记录开始执行
            if self.monitor_callback:
                self.monitor_callback("tool_start", {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call["id"],
                    "args": tool_call.get("args", {}),
                    "start_time": start_time,
                })
            
            # 执行工具
            result = super()._execute_single_tool_call(tool_call, config, **kwargs)
            
            # 记录执行结果
            end_time = time.time()
            duration = end_time - start_time
            
            self.performance_tracker.record_execution(
                tool_name=tool_name,
                duration=duration,
                success=True,
                result_size=len(str(result.content)) if result.content else 0,
            )
            
            if self.monitor_callback:
                self.monitor_callback("tool_success", {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call["id"],
                    "duration": duration,
                    "result": result,
                })
            
            return result
            
        except Exception as e:
            # 记录错误
            end_time = time.time()
            duration = end_time - start_time
            
            self.performance_tracker.record_execution(
                tool_name=tool_name,
                duration=duration,
                success=False,
                error=str(e),
            )
            
            if self.monitor_callback:
                self.monitor_callback("tool_error", {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call["id"],
                    "duration": duration,
                    "error": e,
                })
            
            raise

class PerformanceTracker:
    """工具性能追踪器"""
    
    def __init__(self):
        self.executions: list[dict] = []
        self.stats: dict[str, dict] = {}
    
    def record_execution(
        self,
        tool_name: str,
        duration: float,
        success: bool,
        result_size: int = 0,
        error: Optional[str] = None,
    ):
        """记录工具执行"""
        execution = {
            "tool_name": tool_name,
            "duration": duration,
            "success": success,
            "result_size": result_size,
            "error": error,
            "timestamp": time.time(),
        }
        
        self.executions.append(execution)
        
        # 更新统计信息
        if tool_name not in self.stats:
            self.stats[tool_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "max_duration": 0.0,
                "min_duration": float('inf'),
            }
        
        stats = self.stats[tool_name]
        stats["total_executions"] += 1
        
        if success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1
        
        stats["total_duration"] += duration
        stats["avg_duration"] = stats["total_duration"] / stats["total_executions"]
        stats["max_duration"] = max(stats["max_duration"], duration)
        stats["min_duration"] = min(stats["min_duration"], duration)
    
    def get_performance_report(self) -> dict:
        """生成性能报告"""
        return {
            "total_executions": len(self.executions),
            "tool_stats": self.stats,
            "recent_failures": [
                ex for ex in self.executions[-10:]
                if not ex["success"]
            ],
        }
```

### 5.3 错误分级与告警策略

```python
class ToolErrorClassifier:
    """工具错误分级：用于区分可重试/不可重试/降级可接受的错误"""
    
    RETRYABLE = (TimeoutError, ConnectionError, OSError)
    NON_RETRYABLE = (ValueError, TypeError)
    
    @staticmethod
    def classify(error: Exception) -> str:
        if isinstance(error, ToolErrorClassifier.RETRYABLE):
            return "retryable"
        if isinstance(error, ToolErrorClassifier.NON_RETRYABLE):
            return "non_retryable"
        return "unknown"

class AlertingGateway:
    """告警网关：将高等级错误和异常QPS/延迟推送到外部告警系统"""
    
    def __init__(self, send_fn: Callable[[str, dict], None]):
        self.send = send_fn
    
    def alert(self, level: str, message: str, **kv):
        payload = {"level": level, "message": message, **kv}
        self.send("tool_alert", payload)

def monitored_fallback(tool_node: MonitoredToolNode, alerting: AlertingGateway):
    """给MonitoredToolNode注入错误分级与告警回调"""
    original_handle = tool_node._handle_tool_error
    
    def _wrapped_handle(call: ToolCall, error: Exception) -> ToolMessage:
        cls = ToolErrorClassifier.classify(error)
        level = "warning" if cls == "retryable" else "critical"
        alerting.alert(level, "tool_execution_failed", tool=call.get("name"), error=str(error), type=cls)
        return original_handle(call, error)
    
    tool_node._handle_tool_error = _wrapped_handle  # 注入包装
    return tool_node
```

要点：
- 将错误按“可重试/不可重试/未知”分级，配合回退策略与指数退避。
- 仅对高等级（critical）错误推送告警，避免告警风暴；对retryable聚合统计上报。
- 与性能追踪合并，形成“错误率、平均延迟、P95”三指标联动的SLO看板。

## 6. 生产级应用案例

### 6.1 法律信息结构化智能体

基于LangGraph在法律领域的实际应用，展示如何构建专业的法律信息处理系统：

```python
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.tools import tool

class LegalEntity(BaseModel):
    """法律实体模型"""
    name: str = Field(description="实体名称")
    type: str = Field(description="实体类型：人员、组织、法条等")
    attributes: Dict[str, str] = Field(default_factory=dict, description="实体属性")
    relationships: List[str] = Field(default_factory=list, description="关联关系")

class LegalDocument(BaseModel):
    """法律文档模型"""
    title: str
    content: str
    entities: List[LegalEntity] = Field(default_factory=list)
    legal_provisions: List[str] = Field(default_factory=list)
    case_references: List[str] = Field(default_factory=list)

@tool
def extract_legal_entities(
    text: str,
    state: Annotated[dict, InjectedState(key="knowledge_base")]
) -> str:
    """提取法律实体和关系"""
    # 使用知识图谱辅助实体识别
    knowledge_base = state.get("knowledge_base", {})
    
    # 实体提取逻辑（结合NER和规则）
    entities = []
    
    # 使用预训练的法律NER模型
    ner_results = _extract_entities_with_ner(text)
    
    # 结合法律知识库进行实体链接
    for entity in ner_results:
        linked_entity = _link_to_knowledge_base(entity, knowledge_base)
        if linked_entity:
            entities.append(linked_entity)
    
    return json.dumps({
        "entities": [entity.dict() for entity in entities],
        "confidence_scores": _calculate_confidence_scores(entities),
    }, ensure_ascii=False, indent=2)

@tool
def analyze_legal_provisions(document_text: str) -> str:
    """分析法律条文和适用性"""
    # 法条识别和解析
    provisions = _identify_legal_provisions(document_text)
    
    # 适用性分析
    applicability = _analyze_applicability(provisions, document_text)
    
    return json.dumps({
        "provisions": provisions,
        "applicability_analysis": applicability,
        "recommendation": _generate_legal_recommendation(provisions, applicability)
    }, ensure_ascii=False, indent=2)

@tool
def structure_legal_information(
    entities: str,
    provisions: str,
    store: Annotated[BaseStore, InjectedStore(namespace="legal_kb")]
) -> str:
    """结构化法律信息并更新知识库"""
    entities_data = json.loads(entities)
    provisions_data = json.loads(provisions)
    
    # 构建结构化法律文档
    structured_doc = LegalDocument(
        title="分析结果",
        content="",
        entities=[LegalEntity(**entity) for entity in entities_data["entities"]],
        legal_provisions=provisions_data["provisions"],
    )
    
    # 更新知识库
    for entity in structured_doc.entities:
        store.put(
            ("entities", entity.name),
            entity.dict()
        )
    
    # 存储案例关联
    case_id = f"case_{int(time.time())}"
    store.put(
        ("cases", case_id),
        {
            "entities": entities_data,
            "provisions": provisions_data,
            "structured_document": structured_doc.dict(),
            "created_at": time.time(),
        }
    )
    
    return structured_doc.model_dump_json(indent=2, ensure_ascii=False)

# 创建专业的法律分析智能体
def create_legal_analysis_agent():
    """创建法律分析智能体"""
    from langchain_openai import ChatOpenAI
    
    # 专业模型配置
    model = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.1,  # 法律分析需要稳定性
    )
    
    # 法律专业工具
    legal_tools = [
        extract_legal_entities,
        analyze_legal_provisions,
        structure_legal_information,
    ]
    
    # 专业提示词
    legal_prompt = """你是一个专业的法律分析助手，具备以下能力：

1. 法律实体识别：准确识别人员、组织、法律条文等实体
2. 条文分析：分析相关法律条文的适用性和影响
3. 信息结构化：将非结构化的法律文本转换为结构化信息

分析时请遵循以下原则：
- 准确性优先：确保法律信息的准确性和完整性
- 引用规范：明确标注法律条文的出处和版本
- 关联分析：识别实体间的法律关系和影响

请使用提供的工具进行专业的法律分析。"""
    
    return create_react_agent(
        model=model,
        tools=legal_tools,
        prompt=legal_prompt,
        checkpointer=PostgresCheckpointSaver.from_conn_string(
            "postgresql://user:pass@localhost/legal_db"
        ),
        store=PostgresStore.from_conn_string(
            "postgresql://user:pass@localhost/legal_kb"
        ),
        name="LegalAnalysisAgent",
        version="v2",
    )

def _extract_entities_with_ner(text: str) -> List[Dict]:
    """使用NER模型提取实体"""
    # 这里使用专业的法律NER模型
    # 实际实现中会调用如spaCy、transformers等库
    entities = []
    
    # 模拟NER结果
    mock_entities = [
        {"text": "最高人民法院", "label": "ORGANIZATION", "start": 0, "end": 6},
        {"text": "民法典", "label": "LAW", "start": 20, "end": 23},
        {"text": "第一百条", "label": "PROVISION", "start": 24, "end": 28},
    ]
    
    return mock_entities

def _link_to_knowledge_base(entity: Dict, knowledge_base: Dict) -> Optional[LegalEntity]:
    """将实体链接到知识库"""
    entity_name = entity["text"]
    entity_type = entity["label"]
    
    # 在知识库中查找匹配实体
    kb_matches = knowledge_base.get(entity_type.lower(), {})
    
    if entity_name in kb_matches:
        kb_info = kb_matches[entity_name]
        return LegalEntity(
            name=entity_name,
            type=entity_type,
            attributes=kb_info.get("attributes", {}),
            relationships=kb_info.get("relationships", [])
        )
    
    return LegalEntity(name=entity_name, type=entity_type)
```

**法律智能体特点**：
- **专业知识集成**：结合法律知识库和NER模型
- **结构化输出**：将非结构化文本转换为标准化数据模型
- **知识图谱支持**：通过实体链接建立法律概念间的关联
- **持久化存储**：自动更新和维护法律知识库

### 6.2 代码生成与调试智能体

展示LangGraph在代码生成领域的强大能力，实现反思式代码生成：

```python
class CodeGenerationState(TypedDict):
    """代码生成状态模式"""
    messages: Annotated[list[BaseMessage], add_messages]
    current_code: Optional[str]
    test_results: Optional[dict]
    error_history: list[str]
    iteration_count: int
    max_iterations: int

@tool
def generate_code(
    requirements: str,
    programming_language: str = "python",
    state: Annotated[dict, InjectedState()],
) -> str:
    """生成代码工具"""
    iteration = state.get("iteration_count", 0)
    error_history = state.get("error_history", [])
    
    # 构建上下文提示
    context_prompt = f"""
要求：{requirements}
语言：{programming_language}

之前的错误（如果有）：
{chr(10).join(error_history[-3:]) if error_history else "无"}

请生成满足要求的代码，注意避免之前的错误。
"""
    
    # 调用代码生成模型
    generated_code = _call_code_generation_model(context_prompt)
    
    return generated_code

@tool
def execute_and_test_code(
    code: str,
    test_cases: Optional[str] = None
) -> str:
    """执行和测试代码"""
    results = {
        "execution_success": False,
        "test_success": False,
        "errors": [],
        "output": "",
        "test_results": {},
    }
    
    try:
        # 安全代码执行环境
        execution_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                # 限制可用的内置函数
            }
        }
        
        # 执行代码
        exec(code, execution_globals)
        results["execution_success"] = True
        
        # 运行测试用例（如果提供）
        if test_cases:
            test_results = _run_test_cases(code, test_cases, execution_globals)
            results["test_results"] = test_results
            results["test_success"] = all(
                test["passed"] for test in test_results.values()
            )
        
    except Exception as e:
        results["errors"].append({
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
    
    return json.dumps(results, ensure_ascii=False, indent=2)

@tool
def reflect_on_errors(
    code: str,
    test_results: str,
    state: Annotated[dict, InjectedState()],
) -> str:
    """反思代码错误并提供改进建议"""
    results = json.loads(test_results)
    current_iteration = state.get("iteration_count", 0)
    
    if results["execution_success"] and results.get("test_success", True):
        return "代码执行成功，无需反思"
    
    # 错误分析
    error_analysis = {
        "iteration": current_iteration,
        "error_types": [],
        "improvement_suggestions": [],
        "code_issues": [],
    }
    
    # 分析执行错误
    for error in results.get("errors", []):
        error_analysis["error_types"].append(error["type"])
        error_analysis["improvement_suggestions"].append(
            _generate_improvement_suggestion(error, code)
        )
    
    # 分析测试失败
    for test_name, test_result in results.get("test_results", {}).items():
        if not test_result["passed"]:
            error_analysis["code_issues"].append({
                "test": test_name,
                "expected": test_result.get("expected"),
                "actual": test_result.get("actual"),
                "suggestion": _suggest_test_fix(test_result, code),
            })
    
    return json.dumps(error_analysis, ensure_ascii=False, indent=2)

def create_code_generation_agent():
    """创建代码生成智能体"""
    graph = StateGraph(CodeGenerationState)
    
    # 代码生成节点
    def code_gen_node(state: CodeGenerationState) -> dict:
        messages = state["messages"]
        requirements = messages[-1].content
        
        code = generate_code.invoke({
            "requirements": requirements,
            "programming_language": "python"
        })
        
        return {
            "current_code": code,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    
    # 测试执行节点
    def test_node(state: CodeGenerationState) -> dict:
        code = state["current_code"]
        test_results = execute_and_test_code.invoke({"code": code})
        
        results_data = json.loads(test_results)
        success = results_data["execution_success"] and results_data.get("test_success", True)
        
        return {
            "test_results": test_results,
            "success": success,
        }
    
    # 反思节点
    def reflection_node(state: CodeGenerationState) -> dict:
        code = state["current_code"]
        test_results = state["test_results"]
        
        reflection = reflect_on_errors.invoke({
            "code": code,
            "test_results": test_results
        })
        
        # 更新错误历史
        error_history = state.get("error_history", [])
        error_history.append(reflection)
        
        return {
            "error_history": error_history,
            "reflection": reflection,
        }
    
    # 构建图结构
    graph.add_node("generate", code_gen_node)
    graph.add_node("test", test_node)
    graph.add_node("reflect", reflection_node)
    
    # 设置流程
    graph.set_entry_point("generate")
    graph.add_edge("generate", "test")
    
    # 条件分支：成功则结束，失败则反思
    graph.add_conditional_edges(
        "test",
        lambda state: END if state.get("success") else (
            "reflect" if state.get("iteration_count", 0) < state.get("max_iterations", 3)
            else END
        ),
        {"reflect": "reflect", END: END}
    )
    
    graph.add_edge("reflect", "generate")
    
    return graph.compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["reflect"],  # 允许人工介入反思过程
    )

def _generate_improvement_suggestion(error: dict, code: str) -> str:
    """生成代码改进建议"""
    error_type = error["type"]
    error_message = error["message"]
    
    suggestions = {
        "SyntaxError": "检查代码语法，特别是括号、引号匹配",
        "NameError": "检查变量名拼写和作用域",
        "TypeError": "检查数据类型和函数参数",
        "ImportError": "检查导入模块是否正确",
    }
    
    return suggestions.get(error_type, f"分析错误: {error_message}")
```

**代码生成智能体特点**：
- **迭代改进**：通过反思机制持续优化代码质量
- **错误学习**：从历史错误中学习，避免重复问题
- **测试驱动**：自动执行测试用例验证代码正确性
- **人工介入**：在关键决策点支持人工审核

### 6.3 完整的智能体应用示例

```python
import asyncio
from typing import Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, ToolNode, InjectedState, InjectedStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# 定义工具
@tool
def weather_tool(location: str) -> str:
    """获取指定位置的天气信息"""
    # 模拟天气API调用
    weather_data = {
        "北京": "晴天，温度25°C",
        "上海": "多云，温度22°C", 
        "广州": "小雨，温度28°C",
    }
    return weather_data.get(location, f"{location}的天气信息暂未获取到")

@tool
def calculator_tool(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全的数学计算
        import ast
        import operator
        
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }
        
        def eval_expr(expr):
            if isinstance(expr, ast.Num):
                return expr.n
            elif isinstance(expr, ast.BinOp):
                return ops[type(expr.op)](eval_expr(expr.left), eval_expr(expr.right))
            else:
                raise TypeError(expr)
        
        result = eval_expr(ast.parse(expression, mode='eval').body)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def search_tool(
    query: str,
    state: Annotated[dict, InjectedState()],
    store: Annotated[InMemoryStore, InjectedStore(namespace="search")],
) -> str:
    """搜索信息工具（带状态和存储注入）"""
    # 从状态获取用户偏好
    user_id = state.get("user_id", "anonymous")
    search_history = store.get(("history", user_id)) or []
    
    # 模拟搜索
    search_results = f"搜索'{query}'的结果：相关信息已找到"
    
    # 保存搜索历史
    search_history.append({
        "query": query,
        "timestamp": time.time(),
    })
    store.put(("history", user_id), search_history[-10:])  # 保留最近10次搜索
    
    return search_results

async def main():
    """主程序演示"""
    
    # 创建模型
    model = ChatAnthropic(model="claude-3-sonnet-20240229")
    
    # 创建检查点保存器和存储
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    
    # 创建智能体
    agent = create_react_agent(
        model=model,
        tools=[weather_tool, calculator_tool, search_tool],
        checkpointer=checkpointer,
        store=store,
        prompt="""你是一个有用的助手，可以：
1. 查询天气信息
2. 进行数学计算  
3. 搜索信息

请根据用户需求选择合适的工具来帮助回答问题。""",
        version="v2",
        name="MultiToolAssistant",
    )
    
    # 配置
    config = {
        "configurable": {
            "thread_id": "demo-session-1",
            "user_id": "user123",
        }
    }
    
    # 测试对话
    test_messages = [
        "北京今天天气怎么样？",
        "帮我计算 15 * 8 + 32",
        "搜索一下人工智能的最新发展",
        "再帮我算一下 (25 + 15) / 4",
    ]
    
    for user_input in test_messages:
        print(f"\n用户: {user_input}")
        print("助手:", end=" ")
        
        # 流式输出
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="updates"
        ):
            for node_name, output in chunk.items():
                if "messages" in output:
                    for msg in output["messages"]:
                        if hasattr(msg, 'content') and msg.content:
                            print(msg.content, end="", flush=True)
        
        print()  # 换行

if __name__ == "__main__":
    asyncio.run(main())
```

### 6.2 最佳实践

1. **工具设计原则**：
   ```python
   # 好的工具设计
   @tool
   def good_weather_tool(
       location: str,
       units: str = "celsius",
       include_forecast: bool = False
   ) -> str:
       """获取天气信息
       
       Args:
           location: 位置名称
           units: 温度单位 (celsius/fahrenheit)
           include_forecast: 是否包含预报
       """
       # 清晰的功能、完整的参数说明、合理的默认值
   ```

2. **错误处理策略**：
   ```python
   # 健壮的工具实现
   @tool  
   def robust_api_tool(endpoint: str) -> str:
       """调用外部API"""
       try:
           response = requests.get(endpoint, timeout=30)
           response.raise_for_status()
           return response.text
       except requests.Timeout:
           return "请求超时，请稍后重试"
       except requests.RequestException as e:
           return f"API调用失败: {e}"
   ```

3. **性能优化**：
   ```python
   # 使用缓存和批量处理
   @tool
   def optimized_batch_tool(items: list[str]) -> str:
       """批量处理项目"""
       # 批量处理而不是逐个处理
       results = process_items_batch(items)
       return json.dumps(results)
   ```

## 7. 总结

LangGraph预构建模块通过封装常见模式和提供高级API，大大简化了智能体应用的开发：

### 7.1 核心优势

- **快速原型**：create_react_agent让ReAct智能体开发变得简单直接
- **灵活工具系统**：ToolNode支持复杂的工具执行和依赖注入
- **生产就绪**：内置错误处理、监控、验证等企业级特性
- **可扩展架构**：清晰的扩展点支持自定义功能

### 7.2 设计亮点

1. **模块化设计**：每个组件职责明确，可独立使用或组合
2. **类型安全**：完整的类型注解确保开发时的类型检查
3. **异步支持**：原生支持异步执行，适合IO密集型应用
4. **监控友好**：内置性能追踪和监控钩子

### 7.3 适用场景

- **聊天机器人**：需要工具调用能力的对话系统
- **任务自动化**：复杂的多步骤任务执行
- **知识问答**：结合外部数据源的智能问答
- **工作流代理**：业务流程的智能化处理

通过深入理解这些预构建组件，开发者能够快速构建高质量的AI应用，同时保持足够的灵活性来满足特定需求。

