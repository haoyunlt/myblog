# OpenAI Agents Python SDK - Tools 模块概览

## 1. 模块职责与边界

Tools 模块是 OpenAI Agents Python SDK 的工具系统核心，提供了统一的工具抽象和多种工具实现。该模块使得智能代理能够执行各种外部操作，从函数调用到复杂的计算机控制任务，极大扩展了代理的能力边界。

### 核心职责

- **工具抽象**：定义统一的工具接口和类型系统
- **函数工具**：将 Python 函数包装为可被 LLM 调用的工具
- **托管工具**：集成 OpenAI 提供的云端工具服务
- **MCP 工具**：支持模型上下文协议的工具集成
- **专用工具**：提供计算机控制、代码执行、文件搜索等专业能力
- **工具安全**：实现工具级的输入输出安全检查
- **动态管理**：支持工具的动态启用/禁用和条件调用

### 工具分类体系

| 工具类型 | 实现类 | 执行位置 | 主要用途 |
|----------|--------|----------|----------|
| 函数工具 | `FunctionTool` | 本地 | Python 函数调用 |
| 文件搜索 | `FileSearchTool` | 云端 | 向量存储检索 |
| 网络搜索 | `WebSearchTool` | 云端 | 实时网络搜索 |
| 计算机控制 | `ComputerTool` | 云端 | 桌面自动化 |
| 代码执行 | `CodeInterpreterTool` | 云端 | 沙箱代码运行 |
| 图像生成 | `ImageGenerationTool` | 云端 | DALL-E 图像创建 |
| 本地 Shell | `LocalShellTool` | 本地 | 系统命令执行 |
| MCP 托管 | `HostedMCPTool` | 云端 | 远程 MCP 服务器 |

### 输入输出接口

**输入：**
- 工具配置参数（名称、描述、参数模式等）
- 执行上下文（`ToolContext` 或 `RunContextWrapper`）
- 工具调用参数（JSON 格式）
- 安全检查规则（输入输出防护）

**输出：**
- 工具执行结果（`FunctionToolResult` 或直接输出）
- 执行项目（`RunItem` 封装）
- 错误信息（异常或错误字符串）

### 上下游依赖关系

**上游调用者：**
- `Agent`：通过 `tools` 配置集成工具
- `RunImpl`：执行引擎调用工具执行逻辑
- `MCPUtil`：MCP 工具集成和管理

**下游依赖：**
- `function_schema`：自动生成 JSON Schema
- `tool_context`：提供执行上下文环境
- `tool_guardrails`：工具级安全检查
- `computer`：计算机控制底层接口
- `tracing`：工具执行追踪和监控

## 2. 模块架构图

```mermaid
flowchart TB
    subgraph "Tools 模块"
        TOOL[Tool 联合类型]
        
        subgraph "函数工具"
            FUNCTIONTOOL[FunctionTool]
            FUNCTOOLRESULT[FunctionToolResult]
            FUNCTOOLDECORATOR[@function_tool 装饰器]
        end
        
        subgraph "托管工具"
            FILESEARCH[FileSearchTool]
            WEBSEARCH[WebSearchTool]
            COMPUTER[ComputerTool]
            CODEINTERPRETER[CodeInterpreterTool]
            IMAGEGEN[ImageGenerationTool]
        end
        
        subgraph "本地工具"
            LOCALSHELL[LocalShellTool]
            LOCALSHELCMD[LocalShellCommandRequest]
        end
        
        subgraph "MCP 工具"
            HOSTEDMCP[HostedMCPTool]
            MCPAPPROVAL[MCPToolApprovalFunction]
            MCPREQUEST[MCPToolApprovalRequest]
        end
        
        subgraph "工具安全"
            TOOLINPUTGUARD[ToolInputGuardrail]
            TOOLOUTPUTGUARD[ToolOutputGuardrail]
            SAFETYCHECKDATA[ComputerToolSafetyCheckData]
        end
        
        subgraph "工具上下文"
            TOOLCONTEXT[ToolContext]
            RUNCONTEXT[RunContextWrapper]
        end
    end
    
    subgraph "支撑系统"
        FUNCTIONSCHEMA[function_schema 模式生成]
        STRICTSCHEMA[strict_schema 严格模式]
        COMPUTER_IMPL[computer 实现层]
        TRACING[tracing 追踪系统]
        GUARDRAILS[guardrails 安全框架]
    end
    
    TOOL --> FUNCTIONTOOL
    TOOL --> FILESEARCH
    TOOL --> WEBSEARCH
    TOOL --> COMPUTER
    TOOL --> CODEINTERPRETER
    TOOL --> IMAGEGEN
    TOOL --> LOCALSHELL
    TOOL --> HOSTEDMCP
    
    FUNCTIONTOOL --> FUNCTOOLRESULT
    FUNCTIONTOOL --> FUNCTOOLDECORATOR
    FUNCTIONTOOL --> TOOLINPUTGUARD
    FUNCTIONTOOL --> TOOLOUTPUTGUARD
    
    HOSTEDMCP --> MCPAPPROVAL
    HOSTEDMCP --> MCPREQUEST
    
    COMPUTER --> SAFETYCHECKDATA
    LOCALSHELL --> LOCALSHELCMD
    
    FUNCTIONTOOL --> TOOLCONTEXT
    FUNCTIONTOOL --> RUNCONTEXT
    
    FUNCTIONTOOL --> FUNCTIONSCHEMA
    FUNCTIONTOOL --> STRICTSCHEMA
    COMPUTER --> COMPUTER_IMPL
    FUNCTIONTOOL --> TRACING
    FUNCTIONTOOL --> GUARDRAILS
    
    style TOOL fill:#e1f5fe
    style FUNCTIONTOOL fill:#f3e5f5
    style HOSTEDMCP fill:#e8f5e8
    style TOOLCONTEXT fill:#fff3e0
```

**架构说明：**

### 工具类型层次

1. **抽象层**：`Tool` 联合类型定义了所有工具的统一接口
2. **实现层**：各种具体工具类实现特定的功能领域
3. **支撑层**：模式生成、安全检查、上下文管理等基础设施
4. **集成层**：与代理系统和执行引擎的集成接口

### 工具执行模式

- **同步执行**：简单工具直接返回结果
- **异步执行**：支持异步 I/O 操作的工具
- **流式执行**：支持渐进式结果返回
- **批量执行**：支持多个工具并发执行

### 安全边界控制

- **输入验证**：JSON Schema 严格模式验证
- **输出过滤**：工具输出安全检查和过滤
- **权限控制**：动态启用/禁用机制
- **执行隔离**：工具执行环境隔离

### 扩展机制

- **函数装饰器**：`@function_tool` 简化函数工具创建
- **MCP 协议**：支持标准化的工具服务集成
- **自定义工具**：通过接口扩展新的工具类型
- **插件化架构**：支持动态工具注册和发现

## 3. 关键算法与流程剖析

### 3.1 函数工具创建算法

```python
def function_tool(
    func: ToolFunction | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    strict_json_schema: bool = True,
    is_enabled: bool | Callable = True,
    **kwargs
) -> Callable:
    """函数工具装饰器的核心实现"""
    
    def decorator(target_func: ToolFunction) -> FunctionTool:
        # 1) 分析函数签名，确定参数类型
        sig = inspect.signature(target_func)
        params = list(sig.parameters.values())
        
        # 2) 检测上下文参数类型
        context_param = None
        func_params = []
        
        for param in params:
            if param.annotation in [RunContextWrapper, ToolContext]:
                context_param = param
            else:
                func_params.append(param)
        
        # 3) 生成 JSON Schema
        schema = function_schema(
            target_func,
            name_override=name_override,
            description_override=description_override,
            docstring_style=DocstringStyle.GOOGLE,
            strict_json_schema=strict_json_schema
        )
        
        # 4) 创建工具调用包装器
        async def tool_invoker(tool_context: ToolContext, args_json: str) -> Any:
            try:
                # 解析 JSON 参数
                args = json.loads(args_json)
                
                # 构建函数调用参数
                call_args = []
                if context_param:
                    if context_param.annotation == ToolContext:
                        call_args.append(tool_context)
                    else:  # RunContextWrapper
                        call_args.append(tool_context.run_context)
                
                # 添加业务参数
                for param in func_params:
                    if param.name in args:
                        call_args.append(args[param.name])
                    elif param.default is not param.empty:
                        call_args.append(param.default)
                    else:
                        raise ValueError(f"Missing required parameter: {param.name}")
                
                # 执行原函数
                if inspect.iscoroutinefunction(target_func):
                    result = await target_func(*call_args)
                else:
                    result = target_func(*call_args)
                
                return result
                
            except Exception as e:
                # 错误处理和日志记录
                logger.error(f"Tool execution failed: {e}")
                return f"Tool execution error: {str(e)}"
        
        # 5) 创建 FunctionTool 实例
        return FunctionTool(
            name=schema["name"],
            description=schema["description"],
            params_json_schema=schema["parameters"],
            on_invoke_tool=tool_invoker,
            strict_json_schema=strict_json_schema,
            is_enabled=is_enabled
        )
    
    # 支持直接调用和装饰器两种用法
    if func is not None:
        return decorator(func)
    return decorator
```

**算法目的：** 自动将 Python 函数转换为可被 LLM 调用的工具，处理参数映射、类型转换和错误处理。

**复杂度分析：**
- 时间复杂度：O(n)，n 为函数参数数量
- 空间复杂度：O(m)，m 为 JSON Schema 大小
- 创建开销：一次性创建，运行时开销极小

**关键设计决策：**
1. **签名分析**：自动识别上下文参数和业务参数
2. **Schema 生成**：基于类型注解自动生成严格的 JSON Schema
3. **错误隔离**：工具执行错误不影响代理主流程
4. **灵活调用**：支持同步和异步函数

### 3.2 工具执行安全检查

```python
async def execute_tool_with_guardrails(
    tool: FunctionTool,
    context: ToolContext,
    args_json: str
) -> FunctionToolResult:
    """带安全检查的工具执行流程"""
    
    # 1) 输入安全检查
    if tool.tool_input_guardrails:
        input_data = ToolInputGuardrailData(
            tool_name=tool.name,
            tool_input=args_json,
            context=context.run_context.context
        )
        
        for guardrail in tool.tool_input_guardrails:
            try:
                with guardrail_span(guardrail.__class__.__name__):
                    result = await _coro.ensure_awaitable(
                        guardrail(input_data, context.run_context, context.agent)
                    )
                    
                    if not result.passed:
                        raise ToolInputGuardrailTripwireTriggered(
                            guardrail_name=guardrail.__class__.__name__,
                            failure_reason=result.failure_reason,
                            tool_name=tool.name
                        )
            except Exception as e:
                logger.error(f"Input guardrail {guardrail.__class__.__name__} failed: {e}")
                raise
    
    # 2) 执行工具
    try:
        with function_span(tool.name) as span:
            raw_output = await tool.on_invoke_tool(context, args_json)
            span.add_input(args_json)
            span.add_output(str(raw_output))
    except Exception as e:
        logger.error(f"Tool {tool.name} execution failed: {e}")
        raise
    
    # 3) 输出安全检查
    if tool.tool_output_guardrails:
        output_data = ToolOutputGuardrailData(
            tool_name=tool.name,
            tool_input=args_json,
            tool_output=str(raw_output),
            context=context.run_context.context
        )
        
        for guardrail in tool.tool_output_guardrails:
            try:
                with guardrail_span(guardrail.__class__.__name__):
                    result = await _coro.ensure_awaitable(
                        guardrail(output_data, context.run_context, context.agent)
                    )
                    
                    if not result.passed:
                        raise ToolOutputGuardrailTripwireTriggered(
                            guardrail_name=guardrail.__class__.__name__,
                            failure_reason=result.failure_reason,
                            tool_name=tool.name
                        )
            except Exception as e:
                logger.error(f"Output guardrail {guardrail.__class__.__name__} failed: {e}")
                raise
    
    # 4) 创建结果对象
    run_item = ToolCallOutputItem(
        tool_name=tool.name,
        input=args_json,
        output=str(raw_output)
    )
    
    return FunctionToolResult(
        tool=tool,
        output=raw_output,
        run_item=run_item
    )
```

**流程目的：** 在工具执行前后进行安全检查，确保工具使用的安全性和合规性。

**安全检查层次：**
1. **输入验证**：参数格式、内容安全、权限检查
2. **执行监控**：执行时间、资源使用、异常处理
3. **输出过滤**：敏感信息过滤、内容审查、格式验证
4. **追踪记录**：完整的执行链路追踪和审计日志

### 3.3 MCP 工具集成算法

```python
class MCPUtil:
    """MCP 工具集成工具类"""
    
    @staticmethod
    async def get_all_function_tools(
        servers: list[MCPServer],
        convert_schemas_to_strict: bool,
        run_context: RunContextWrapper,
        agent: AgentBase
    ) -> list[Tool]:
        """从 MCP 服务器获取所有工具"""
        all_tools = []
        
        for server in servers:
            try:
                # 1) 连接服务器并获取工具列表
                await server.connect()
                mcp_tools = await server.list_tools(run_context, agent)
                
                for mcp_tool in mcp_tools:
                    # 2) 转换 MCP 工具为 FunctionTool
                    function_tool = await MCPUtil._convert_mcp_tool_to_function_tool(
                        server, mcp_tool, convert_schemas_to_strict
                    )
                    all_tools.append(function_tool)
                    
            except Exception as e:
                logger.warning(f"Failed to get tools from MCP server {server.name}: {e}")
                continue  # 跳过失败的服务器，不影响其他服务器
        
        return all_tools
    
    @staticmethod
    async def _convert_mcp_tool_to_function_tool(
        server: MCPServer,
        mcp_tool: MCPTool,
        convert_to_strict: bool
    ) -> FunctionTool:
        """将 MCP 工具转换为 FunctionTool"""
        
        # 1) 处理 JSON Schema
        schema = mcp_tool.inputSchema
        if convert_to_strict and schema:
            try:
                schema = ensure_strict_json_schema(schema)
            except Exception as e:
                logger.warning(f"Failed to convert MCP tool {mcp_tool.name} to strict schema: {e}")
                # 降级使用原始 schema
        
        # 2) 创建工具调用包装器
        async def mcp_tool_invoker(context: ToolContext, args_json: str) -> str:
            try:
                # 解析参数
                args = json.loads(args_json) if args_json else {}
                
                # 调用 MCP 服务器
                with mcp_tools_span(server.name, mcp_tool.name) as span:
                    result = await server.call_tool(mcp_tool.name, args)
                    span.add_input(args_json)
                    span.add_output(result.content)
                
                # 处理结构化内容
                if server.use_structured_content and result.structured_content:
                    # 使用结构化内容
                    return json.dumps(result.structured_content)
                else:
                    # 使用文本内容
                    return str(result.content)
                
            except Exception as e:
                logger.error(f"MCP tool {mcp_tool.name} execution failed: {e}")
                return f"MCP tool execution error: {str(e)}"
        
        # 3) 创建 FunctionTool 实例
        return FunctionTool(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            params_json_schema=schema or {},
            on_invoke_tool=mcp_tool_invoker,
            strict_json_schema=convert_to_strict
        )
```

**集成目的：** 将外部 MCP 服务器的工具无缝集成到代理工具系统中，提供统一的调用接口。

**关键特性：**
1. **协议适配**：MCP 协议到 FunctionTool 的无缝转换
2. **Schema 兼容**：支持严格模式和标准模式的 JSON Schema
3. **错误恢复**：单个服务器失败不影响整体工具集成
4. **结果处理**：支持结构化内容和文本内容两种返回格式

## 4. 工具类型详细说明

### 4.1 函数工具（FunctionTool）

**特点：**
- 最灵活的工具类型，可以包装任意 Python 函数
- 支持自动 JSON Schema 生成
- 提供完整的类型检查和参数验证
- 支持同步和异步函数

**使用场景：**
- 数据库查询和操作
- API 调用和数据获取
- 文件处理和分析
- 业务逻辑计算

**典型实现：**
```python
@function_tool
def calculate_fibonacci(n: int) -> int:
    """计算斐波那契数列的第 n 项
    
    Args:
        n: 要计算的项数，必须是正整数
        
    Returns:
        斐波那契数列的第 n 项
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
```

### 4.2 托管工具（Hosted Tools）

**特点：**
- 由 OpenAI 云端提供的服务
- 无需本地部署和维护
- 具备强大的云端计算能力
- 与 OpenAI 生态深度集成

**主要类型：**

#### FileSearchTool - 文件搜索工具
```python
FileSearchTool(
    vector_store_ids=["vs_123", "vs_456"],
    max_num_results=5,
    ranking_options=RankingOptions(score_threshold=0.8),
    filters=Filters(filename_contains="report")
)
```

#### WebSearchTool - 网络搜索工具
```python
WebSearchTool(
    user_location=UserLocation(country="US", city="San Francisco"),
    search_context_size="high",
    filters=WebSearchToolFilters(domain_includes=["wikipedia.org"])
)
```

#### ComputerTool - 计算机控制工具
```python
ComputerTool(
    display_width_px=1920,
    display_height_px=1080,
    computer_impl=AsyncComputer()  # 或自定义实现
)
```

### 4.3 本地工具（Local Tools）

**特点：**
- 在本地环境执行
- 可以访问本地资源和系统
- 需要考虑安全性和权限控制
- 执行效率高，延迟低

#### LocalShellTool - 本地 Shell 工具
```python
LocalShellTool(
    # 默认配置，支持基本的 shell 命令执行
)

# 使用示例
shell_tool = LocalShellTool()
# 代理可以调用："执行命令 'ls -la' 查看当前目录文件"
```

### 4.4 MCP 工具（MCP Tools）

**特点：**
- 基于模型上下文协议标准
- 支持远程和本地 MCP 服务器
- 提供标准化的工具接口
- 易于扩展和集成第三方服务

#### HostedMCPTool - 托管 MCP 工具
```python
HostedMCPTool(
    tool_config=Mcp(
        server_url="https://mcp-server.example.com",
        api_key="your-api-key"
    ),
    on_approval_request=approval_handler  # 可选的审批处理函数
)
```

## 5. 最佳实践与使用模式

### 5.1 函数工具创建最佳实践

```python
from agents import function_tool
from typing import Optional, List

@function_tool
async def search_products(
    query: str,
    category: Optional[str] = None,
    max_results: int = 10,
    sort_by: str = "relevance"
) -> List[dict]:
    """搜索产品信息
    
    Args:
        query: 搜索关键词，必填
        category: 产品分类，可选
        max_results: 最大结果数量，默认10条
        sort_by: 排序方式，可选值：relevance, price, rating
        
    Returns:
        包含产品信息的字典列表，每个字典包含 id, name, price, rating 字段
    """
    # 实际搜索逻辑
    results = await product_search_api(
        query=query,
        category=category,
        limit=max_results,
        sort=sort_by
    )
    
    return [
        {
            "id": item.id,
            "name": item.name,
            "price": item.price,
            "rating": item.rating
        }
        for item in results
    ]

# 最佳实践要点：
# 1. 详细的文档字符串，说明参数和返回值
# 2. 合理的默认参数，提高易用性
# 3. 类型注解完整，支持自动 Schema 生成
# 4. 返回结构化数据，便于 LLM 理解和处理
```

### 5.2 工具安全防护模式

```python
from agents import function_tool, tool_input_guardrail, tool_output_guardrail

@tool_input_guardrail
def validate_file_path(data, context, agent):
    """验证文件路径的安全性"""
    import os
    args = json.loads(data.tool_input)
    file_path = args.get("file_path", "")
    
    # 检查路径遍历攻击
    if ".." in file_path or file_path.startswith("/"):
        return ToolInputGuardrailResult(
            passed=False,
            failure_reason="不允许访问上级目录或根目录"
        )
    
    # 检查文件扩展名
    allowed_extensions = [".txt", ".json", ".csv", ".md"]
    if not any(file_path.endswith(ext) for ext in allowed_extensions):
        return ToolInputGuardrailResult(
            passed=False,
            failure_reason="只允许访问文本文件"
        )
    
    return ToolInputGuardrailResult(passed=True)

@tool_output_guardrail
def filter_sensitive_content(data, context, agent):
    """过滤输出中的敏感信息"""
    import re
    
    sensitive_patterns = [
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # 信用卡号
        r'\b\d{3}-\d{2}-\d{4}\b',  # 社会安全号
    ]
    
    filtered_output = data.tool_output
    for pattern in sensitive_patterns:
        filtered_output = re.sub(pattern, "[REDACTED]", filtered_output)
    
    if filtered_output != data.tool_output:
        logger.info("Sensitive content filtered from tool output")
    
    return ToolOutputGuardrailResult(
        passed=True,
        modified_output=filtered_output
    )

@function_tool(
    tool_input_guardrails=[validate_file_path],
    tool_output_guardrails=[filter_sensitive_content]
)
def read_file_safe(file_path: str) -> str:
    """安全地读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败：{str(e)}"
```

### 5.3 动态工具管理模式

```python
from agents import Agent, function_tool

class DynamicToolManager:
    """动态工具管理器"""
    
    def __init__(self):
        self.user_permissions = {}
        self.tool_registry = {}
    
    def register_tool(self, tool_name: str, tool_func):
        """注册工具"""
        self.tool_registry[tool_name] = tool_func
    
    def get_enabled_tools(self, user_id: str) -> List[Tool]:
        """获取用户可用的工具列表"""
        user_perms = self.user_permissions.get(user_id, set())
        enabled_tools = []
        
        for tool_name, tool_func in self.tool_registry.items():
            if tool_name in user_perms:
                enabled_tools.append(tool_func)
        
        return enabled_tools

# 使用示例
tool_manager = DynamicToolManager()

@function_tool(
    is_enabled=lambda context, agent: (
        context.context.get("user_role") == "admin"
    )
)
def delete_user(user_id: str) -> str:
    """删除用户（仅管理员可用）"""
    # 删除逻辑
    return f"用户 {user_id} 已删除"

@function_tool(
    is_enabled=lambda context, agent: (
        context.context.get("subscription_level", "free") in ["premium", "enterprise"]
    )
)
def advanced_analytics(query: str) -> str:
    """高级分析功能（仅付费用户可用）"""
    # 分析逻辑
    return f"分析结果：{query}"

# 创建具有条件工具的代理
agent = Agent(
    name="ConditionalAgent",
    tools=[delete_user, advanced_analytics],
    instructions="根据用户权限提供相应的功能"
)
```

### 5.4 工具组合使用模式

```python
@function_tool
async def comprehensive_research(topic: str) -> str:
    """综合研究：结合多个工具进行深度分析"""
    
    # 1. 网络搜索获取基础信息
    web_results = await web_search(f"{topic} overview recent developments")
    
    # 2. 文档检索获取相关资料
    doc_results = await document_search(topic, max_results=5)
    
    # 3. 数据分析
    analysis_data = await analyze_trends(topic)
    
    # 4. 生成综合报告
    report = f"""
    # {topic} 综合研究报告
    
    ## 网络搜索结果摘要
    {web_results[:500]}...
    
    ## 相关文档发现
    - 找到 {len(doc_results)} 篇相关文档
    - 关键主题：{extract_key_themes(doc_results)}
    
    ## 趋势分析
    {analysis_data}
    
    ## 结论与建议
    {generate_recommendations(web_results, doc_results, analysis_data)}
    """
    
    return report

# 工具组合的优势：
# 1. 多数据源整合，提供全面视角
# 2. 自动化工作流，提高效率
# 3. 结构化输出，便于后续处理
```

Tools 模块通过统一的接口和丰富的实现，为智能代理提供了强大的能力扩展机制，支持从简单函数调用到复杂工作流的各种应用场景。
