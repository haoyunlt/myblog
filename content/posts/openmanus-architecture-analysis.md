---
title: "OpenManus 源码剖析：AI Agent 系统深度解析"
date: 2025-09-09T10:00:00+08:00
draft: false
featured: true
description: "详细解析 OpenManus AI Agent 系统的架构设计、核心组件和执行流程。通过源码分析和流程图，帮助开发者深入理解这个多模态 AI Agent 平台的实现原理"
slug: "openmanus-architecture-analysis"
author: "tommie blog"
categories: ["openmanus", "AI", "源码分析"]
tags: ["OpenManus", "AI Agent", "架构分析", "多模态", "源码解析"]
showComments: true
toc: true
tocOpen: true
showReadingTime: true
showWordCount: true
weight: 1000
---

## 一张图看懂 OpenManus 整体架构

<div class="mermaid-image-container" data-chart-id="architecture-analysis-0">
  <img src="/images/mermaid/architecture-analysis-0.svg" 
       alt="Mermaid Chart architecture-analysis-0" 
       class="mermaid-generated-image"
       loading="lazy"
       style="max-width: 100%; height: auto;"
       onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
  <div class="mermaid-fallback" style="display: none;">
    <details>
      <summary>显示 Mermaid 源码</summary>
      <pre class="mermaid">flowchart TD
    A[main.py] --> B[Manus Agent 创建]
    B --> C[配置加载 Config]
    C --> D[LLM 初始化]
    C --> E[工具集合 ToolCollection]
    C --> F[MCP 客户端连接]

    B --> G[Agent 执行循环]
    G --> H[think: 推理决策]
    H --> I[LLM.ask_tool]
    I --> J{工具调用?}

    J -- 是 --> K[act: 执行工具]
    K --> L[ToolCollection.execute]
    L --> M{工具类型}

    M --> N[本地工具<br/>PythonExecute/BrowserUse/FileOps]
    M --> O[MCP 远程工具<br/>MCPClientTool]
    M --> P[特殊工具<br/>Terminate/AskHuman]

    N --> Q[工具结果 ToolResult]
    O --> Q
    P --> Q

    Q --> R[更新记忆 Memory]
    R --> S{状态检查}

    S -- 继续 --> G
    S -- 完成 --> T[cleanup 清理]

    J -- 否 --> U[直接响应]
    U --> R

    %% Flow 执行路径
    V[run_flow.py] --> W[FlowFactory]
    W --> X[PlanningFlow]
    X --> Y[创建计划 PlanningTool]
    Y --> Z[步骤执行循环]
    Z --> AA[Agent.run]
    AA --> G

    %% 配置和依赖
    BB[config.toml] --> C
    CC[mcp.json] --> F
    DD[Sandbox Docker] --> N
    EE[Browser Context] --> N</pre>
    </details>
  </div>
</div>

## 核心组件深度解析

### 1) 配置系统：统一的配置管理

OpenManus 采用分层配置设计，支持多种 LLM 提供商和运行环境：

```python
// app/config.py - 核心配置类
class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]           # 多 LLM 配置
    sandbox: Optional[SandboxSettings]     # 沙箱环境
    browser_config: Optional[BrowserSettings]  # 浏览器配置
    search_config: Optional[SearchSettings]    # 搜索引擎
    mcp_config: Optional[MCPSettings]      # MCP 服务器
    run_flow_config: Optional[RunflowSettings] # 流程配置
```

**配置加载流程**：
- 单例模式确保全局配置一致性
- 支持 `config.toml` 和 `config.example.toml` 回退
- 动态 LLM 配置覆盖机制
- MCP 服务器配置从 JSON 文件加载

### 2) Agent 体系：分层的智能体架构

#### BaseAgent：抽象基类

```python
class BaseAgent(BaseModel, ABC):
    # 核心属性
    name: str
    description: Optional[str]
    system_prompt: Optional[str]
    next_step_prompt: Optional[str]

    # 依赖组件
    llm: LLM = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = Field(default=AgentState.IDLE)

    # 执行控制
    max_steps: int = Field(default=10)
    current_step: int = Field(default=0)
```

**状态管理**：使用 `state_context` 上下文管理器确保状态转换的原子性和异常安全。

#### ToolCallAgent：工具调用智能体

继承自 `ReActAgent`，实现了完整的工具调用生命周期：

```python
async def think(self) -> bool:
    """推理阶段：决策下一步行动"""
    response = await self.llm.ask_tool(
        messages=self.messages,
        system_msgs=[Message.system_message(self.system_prompt)],
        tools=self.available_tools.to_params(),
        tool_choice=self.tool_choices,
    )
    # 处理工具调用或直接响应

async def act(self) -> str:
    """执行阶段：调用工具并处理结果"""
    for command in self.tool_calls:
        result = await self.execute_tool(command)
        # 更新记忆，处理特殊工具
```

#### Manus：多功能通用智能体

OpenManus 的核心智能体，集成了本地工具和 MCP 远程工具：

```python
class Manus(ToolCallAgent):
    # 工具集合
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),      # Python 代码执行
            BrowserUseTool(),     # 浏览器自动化
            StrReplaceEditor(),   # 文件编辑
            AskHuman(),          # 人机交互
            Terminate(),         # 终止执行
        )
    )

    # MCP 客户端管理
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    connected_servers: Dict[str, str] = Field(default_factory=dict)
```

**MCP 集成机制**：
- 支持 SSE 和 stdio 两种连接方式
- 动态工具发现和注册
- 工具命名冲突处理（前缀机制）
- 连接生命周期管理

### 3) 工具系统：可扩展的能力模块

#### BaseTool：工具抽象基类

```python
class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具逻辑"""

    def to_param(self) -> Dict:
        """转换为 OpenAI 函数调用格式"""
```

#### ToolCollection：工具集合管理

```python
class ToolCollection:
    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    async def execute(self, *, name: str, tool_input: Dict[str, Any] = None) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        return await tool(**tool_input)
```

#### 核心工具类型

**1. 代码执行工具 (PythonExecute)**
- 沙箱环境中安全执行 Python 代码
- 支持 Docker 容器隔离
- 结果捕获和错误处理

**2. 浏览器自动化 (BrowserUseTool)**
- 基于 Playwright 的网页操作
- 支持截图、点击、输入等交互
- 上下文管理和资源清理

**3. 文件操作 (StrReplaceEditor)**
- 文件读写、搜索替换
- 目录遍历和文件管理
- 安全路径验证

**4. MCP 远程工具 (MCPClientTool)**
```python
class MCPClientTool(BaseTool):
    session: Optional[ClientSession] = None
    server_id: str = ""
    original_name: str = ""

    async def execute(self, **kwargs) -> ToolResult:
        result = await self.session.call_tool(self.original_name, kwargs)
        content_str = ", ".join(
            item.text for item in result.content if isinstance(item, TextContent)
        )
        return ToolResult(output=content_str)
```

### 4) 流程编排：PlanningFlow

#### 计划驱动的执行模式

```python
class PlanningFlow(BaseFlow):
    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
```

**执行流程**：

<div class="mermaid-image-container" data-chart-id="architecture-analysis-1">
  <img src="/images/mermaid/architecture-analysis-1.svg" 
       alt="Mermaid Chart architecture-analysis-1" 
       class="mermaid-generated-image"
       loading="lazy"
       style="max-width: 100%; height: auto;"
       onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
  <div class="mermaid-fallback" style="display: none;">
    <details>
      <summary>显示 Mermaid 源码</summary>
      <pre class="mermaid">sequenceDiagram
    autonumber
    participant U as User
    participant PF as PlanningFlow
    participant PT as PlanningTool
    participant LLM as LLM
    participant A as Agent

    U->>PF: execute(input_text)
    PF->>LLM: 创建计划请求
    LLM->>PT: planning 工具调用
    PT-->>PF: 计划创建完成

    loop 执行步骤
        PF->>PT: 获取当前步骤
        PT-->>PF: 步骤信息
        PF->>A: run(step_prompt)
        A-->>PF: 步骤执行结果
        PF->>PT: 标记步骤完成
    end

    PF->>LLM: 生成执行总结
    LLM-->>U: 最终结果</pre>
    </details>
  </div>
</div>

#### 步骤状态管理

```python
class PlanStepStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
```

**状态转换逻辑**：
- 自动发现下一个待执行步骤
- 步骤状态原子性更新
- 支持步骤类型路由到特定 Agent

### 5) LLM 抽象层：统一的模型接口

#### 多模型支持

```python
class LLM:
    def __init__(self, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
        # 支持 Azure OpenAI, OpenAI, AWS Bedrock
        if self.api_type == "azure":
            self.client = AsyncAzureOpenAI(...)
        elif self.api_type == "aws":
            self.client = BedrockClient()
        else:
            self.client = AsyncOpenAI(...)
```

#### Token 计数和限制

```python
class TokenCounter:
    def count_message_tokens(self, messages: List[dict]) -> int:
        """精确计算消息 token 数量，支持图像 token 计算"""

    def count_image(self, image_item: dict) -> int:
        """根据 OpenAI 规则计算图像 token"""
```

**Token 管理特性**：
- 精确的 token 计数（文本 + 图像）
- 累积 token 跟踪和限制
- 自动重试和错误处理
- 流式响应支持

#### 核心 API 方法

```python
async def ask_tool(
    self,
    messages: List[Union[dict, Message]],
    tools: Optional[List[dict]] = None,
    tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
) -> ChatCompletionMessage | None:
    """工具调用接口，支持函数调用"""

async def ask_with_images(
    self,
    messages: List[Union[dict, Message]],
    images: List[Union[str, dict]],
) -> str:
    """多模态接口，支持图像输入"""
```

## 关键执行路径分析

### 1) 单 Agent 执行路径

<div class="mermaid-image-container" data-chart-id="architecture-analysis-2">
  <img src="/images/mermaid/architecture-analysis-2.svg" 
       alt="Mermaid Chart architecture-analysis-2" 
       class="mermaid-generated-image"
       loading="lazy"
       style="max-width: 100%; height: auto;"
       onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
  <div class="mermaid-fallback" style="display: none;">
    <details>
      <summary>显示 Mermaid 源码</summary>
      <pre class="mermaid">flowchart TD
    A[main.py] --> B[Manus.create]
    B --> C[initialize_mcp_servers]
    C --> D[agent.run]
    D --> E[BaseAgent.run]
    E --> F[step 循环]
    F --> G[ToolCallAgent.think]
    G --> H[LLM.ask_tool]
    H --> I[ToolCallAgent.act]
    I --> J[execute_tool]
    J --> K[ToolCollection.execute]
    K --> L[更新 Memory]
    L --> M{max_steps?}
    M -- 否 --> F
    M -- 是 --> N[cleanup]</pre>
    </details>
  </div>
</div>

**关键函数调用链**：
1. `Manus.create()` → 工厂方法创建实例
2. `initialize_mcp_servers()` → 连接远程工具服务器
3. `BaseAgent.run()` → 主执行循环
4. `ToolCallAgent.think()` → LLM 推理决策
5. `ToolCallAgent.act()` → 工具执行
6. `execute_tool()` → 具体工具调用
7. `cleanup()` → 资源清理

### 2) Flow 执行路径

<div class="mermaid-image-container" data-chart-id="architecture-analysis-3">
  <img src="/images/mermaid/architecture-analysis-3.svg" 
       alt="Mermaid Chart architecture-analysis-3" 
       class="mermaid-generated-image"
       loading="lazy"
       style="max-width: 100%; height: auto;"
       onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
  <div class="mermaid-fallback" style="display: none;">
    <details>
      <summary>显示 Mermaid 源码</summary>
      <pre class="mermaid">flowchart TD
    A[run_flow.py] --> B[FlowFactory.create_flow]
    B --> C[PlanningFlow]
    C --> D[flow.execute]
    D --> E[_create_initial_plan]
    E --> F[LLM + PlanningTool]
    F --> G[执行循环]
    G --> H[_get_current_step_info]
    H --> I[get_executor]
    I --> J[_execute_step]
    J --> K[Agent.run]
    K --> L[_mark_step_completed]
    L --> M{更多步骤?}
    M -- 是 --> G
    M -- 否 --> N[_finalize_plan]</pre>
    </details>
  </div>
</div>

**Flow 特有逻辑**：
- 计划创建：LLM 生成结构化计划
- 步骤路由：根据步骤类型选择合适的 Agent
- 状态跟踪：实时更新步骤执行状态
- 上下文传递：计划状态作为 Agent 执行上下文

### 3) MCP 工具调用路径

<div class="mermaid-image-container" data-chart-id="architecture-analysis-4">
  <img src="/images/mermaid/architecture-analysis-4.svg" 
       alt="Mermaid Chart architecture-analysis-4" 
       class="mermaid-generated-image"
       loading="lazy"
       style="max-width: 100%; height: auto;"
       onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
  <div class="mermaid-fallback" style="display: none;">
    <details>
      <summary>显示 Mermaid 源码</summary>
      <pre class="mermaid">sequenceDiagram
    autonumber
    participant A as Agent
    participant TC as ToolCollection
    participant MCT as MCPClientTool
    participant CS as ClientSession
    participant MS as MCP Server

    A->>TC: execute("mcp_server_tool")
    TC->>MCT: execute(**kwargs)
    MCT->>CS: call_tool(original_name, kwargs)
    CS->>MS: JSON-RPC 调用
    MS-->>CS: 工具执行结果
    CS-->>MCT: CallToolResult
    MCT-->>TC: ToolResult
    TC-->>A: 格式化结果</pre>
    </details>
  </div>
</div>

## 源码走读：关键模块详解

### 1) 配置模块 (`app/config.py`)

```python
class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

**设计要点**：
- 线程安全的单例模式
- 支持配置热加载和覆盖
- 多环境配置管理（开发/生产）

### 2) 智能体基类 (`app/agent/base.py`)

```python
@asynccontextmanager
async def state_context(self, new_state: AgentState):
    """Context manager for safe agent state transitions."""
    previous_state = self.state
    self.state = new_state
    try:
        yield
    except Exception as e:
        self.state = AgentState.ERROR
        raise e
    finally:
        self.state = previous_state
```

**设计要点**：
- 状态转换的原子性保证
- 异常安全的状态管理
- 可观测的执行生命周期

### 3) 工具调用智能体 (`app/agent/toolcall.py`)

```python
async def execute_tool(self, command: ToolCall) -> str:
    """Execute a single tool call with robust error handling"""
    try:
        args = json.loads(command.function.arguments or "{}")
        result = await self.available_tools.execute(name=name, tool_input=args)
        await self._handle_special_tool(name=name, result=result)

        if hasattr(result, "base64_image") and result.base64_image:
            self._current_base64_image = result.base64_image

        return f"Observed output of cmd `{name}` executed:\n{str(result)}"
    except Exception as e:
        return f"Error: {str(e)}"
```

**设计要点**：
- 健壮的错误处理机制
- 多模态结果支持（图像）
- 特殊工具的生命周期管理

### 4) MCP 客户端 (`app/tool/mcp.py`)

```python
async def connect_sse(self, server_url: str, server_id: str = "") -> None:
    """Connect to an MCP server using SSE transport."""
    exit_stack = AsyncExitStack()
    self.exit_stacks[server_id] = exit_stack

    streams_context = sse_client(url=server_url)
    streams = await exit_stack.enter_async_context(streams_context)
    session = await exit_stack.enter_async_context(ClientSession(*streams))
    self.sessions[server_id] = session

    await self._initialize_and_list_tools(server_id)
```

**设计要点**：
- 异步上下文管理确保资源清理
- 多服务器连接支持
- 动态工具发现和注册

### 5) LLM 抽象层 (`app/llm.py`)

```python
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((OpenAIError, Exception, ValueError))
)
async def ask_tool(self, messages, tools=None, tool_choice=ToolChoice.AUTO):
    """Ask LLM using functions/tools and return the response."""
    input_tokens = self.count_message_tokens(messages)

    if not self.check_token_limit(input_tokens):
        raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

    response = await self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )

    self.update_token_count(response.usage.prompt_tokens, response.usage.completion_tokens)
    return response.choices[0].message
```

**设计要点**：
- 指数退避重试机制
- 精确的 token 计数和限制
- 多模型统一接口抽象

## 性能优化与最佳实践

### 1) 内存管理
- **Message 限制**：Memory 类自动截断历史消息
- **Token 跟踪**：实时监控 token 使用，防止超限
- **资源清理**：Agent 和工具的异步清理机制

### 2) 并发控制
- **异步设计**：全链路异步，提升并发性能
- **连接池**：MCP 客户端连接复用
- **沙箱隔离**：Docker 容器提供安全隔离

### 3) 错误处理
- **重试机制**：LLM 调用自动重试（指数退避）
- **优雅降级**：工具失败不影响整体流程
- **状态恢复**：Agent 状态异常时的恢复策略

### 4) 可观测性
- **结构化日志**：详细的执行日志和错误追踪
- **Token 统计**：实时 token 使用监控
- **执行追踪**：完整的工具调用链路追踪

## 扩展点与架构演进

### 1) 新工具集成
```python
class CustomTool(BaseTool):
    name: str = "custom_tool"
    description: str = "自定义工具描述"

    async def execute(self, **kwargs) -> ToolResult:
        # 实现具体逻辑
        return ToolResult(output="执行结果")

// 注册到工具集合
manus.available_tools.add_tool(CustomTool())
```

### 2) 新 Agent 类型
```python
class SpecializedAgent(ToolCallAgent):
    name: str = "specialized"
    system_prompt: str = "专门化的系统提示"

    async def think(self) -> bool:
        # 自定义推理逻辑
        return await super().think()
```

### 3) 新 Flow 模式
```python
class CustomFlow(BaseFlow):
    async def execute(self, input_text: str) -> str:
        # 实现自定义执行逻辑
        pass

// 注册到工厂
FlowFactory.flows[FlowType.CUSTOM] = CustomFlow
```

## 关键设计模式与原则

### 1) 设计模式应用

**工厂模式**：
- `FlowFactory` 创建不同类型的执行流程
- `LLM` 类根据配置创建不同的客户端实例

**策略模式**：
- 不同的工具实现统一的 `BaseTool` 接口
- 多种 LLM 提供商的统一抽象

**观察者模式**：
- Agent 状态变化的监听和响应
- 工具执行结果的处理链

**装饰器模式**：
- LLM 调用的重试装饰器
- 工具执行的错误处理装饰器

### 2) 架构原则

**单一职责原则**：
- 每个 Agent 专注特定领域
- 工具类职责明确分离

**开闭原则**：
- 通过继承扩展新的 Agent 类型
- 插件化的工具扩展机制

**依赖倒置原则**：
- Agent 依赖抽象的 LLM 接口
- 工具系统依赖抽象的 BaseTool

**接口隔离原则**：
- 不同类型的工具有专门的接口
- Agent 只依赖需要的工具能力

## 部署与运维考虑

### 1) 容器化部署
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### 2) 配置管理
- 环境变量覆盖配置文件
- 敏感信息使用密钥管理服务
- 配置版本化和回滚机制

### 3) 监控告警
- Agent 执行状态监控
- Token 使用量告警
- 工具调用成功率统计
- 系统资源使用监控

### 4) 日志管理
- 结构化日志输出
- 日志级别动态调整
- 敏感信息脱敏处理
- 日志聚合和分析

## 总结

OpenManus 采用了**分层架构 + 插件化设计**的理念：

1. **配置层**：统一的配置管理，支持多环境部署
2. **抽象层**：LLM、Agent、Tool 的统一抽象
3. **实现层**：具体的工具和智能体实现
4. **编排层**：Flow 提供高级任务编排能力
5. **扩展层**：MCP 协议支持远程工具生态

这种架构设计既保证了**核心功能的稳定性**，又提供了**良好的扩展性**，是构建企业级 AI Agent 系统的优秀实践。通过 MCP 协议的集成，OpenManus 能够无缝接入各种外部工具和服务，真正实现了**"一个 Agent，连接一切"**的愿景。

### 核心优势

- **模块化设计**：清晰的职责分离，便于维护和扩展
- **异步架构**：全链路异步设计，支持高并发场景
- **多模态支持**：文本、图像、代码的统一处理
- **安全隔离**：沙箱环境确保代码执行安全
- **可观测性**：完整的日志、监控和追踪体系
- **生态开放**：MCP 协议支持丰富的工具生态

OpenManus 不仅是一个功能强大的 AI Agent 平台，更是一个展示现代软件架构设计理念的优秀案例。它的设计思想和实现方式，为构建下一代智能应用提供了宝贵的参考。

