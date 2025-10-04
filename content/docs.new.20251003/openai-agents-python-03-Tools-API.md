# OpenAI Agents Python SDK - Tools 模块 API 详解

## 1. API 总览

Tools 模块是 OpenAI Agents SDK 的能力扩展核心，提供多种工具类型让代理能够执行实际操作。从简单的Python函数到复杂的托管服务，Tools 模块支持灵活的工具定义和集成。

### API 层次结构

```
Tool (联合类型)
    ├── FunctionTool (函数工具)
    │   └── function_tool() 装饰器/工厂函数
    ├── FileSearchTool (文件搜索工具)
    ├── WebSearchTool (网页搜索工具)
    ├── ComputerTool (计算机控制工具)
    ├── HostedMCPTool (托管MCP工具)
    ├── LocalShellTool (本地Shell工具)
    ├── CodeInterpreterTool (代码解释器工具)
    └── ImageGenerationTool (图像生成工具)
```

### API 分类

| API 类别 | 核心 API | 功能描述 |
|---------|---------|---------|
| **函数工具** | `function_tool()` | Python函数转工具的装饰器/工厂 |
| | `FunctionTool.__init__()` | 手动创建函数工具 |
| **托管工具** | `FileSearchTool()` | 向量存储文件搜索 |
| | `WebSearchTool()` | 网页搜索 |
| | `CodeInterpreterTool()` | 代码执行 |
| | `ImageGenerationTool()` | 图像生成 |
| **高级工具** | `ComputerTool()` | 计算机控制（鼠标、键盘等） |
| | `HostedMCPTool()` | 托管MCP服务器 |
| | `LocalShellTool()` | 本地Shell命令执行 |
| **工具管理** | `Tool` 联合类型 | 统一的工具类型定义 |

## 2. FunctionTool - 函数工具 API

### 2.1 function_tool 装饰器/工厂函数

**API 签名：**
```python
# 装饰器用法（无参数）
@function_tool
def my_tool(param1: str, param2: int) -> str:
    ...

# 装饰器用法（带参数）
@function_tool(
    name_override="custom_name",
    description_override="Custom description",
    strict_mode=True,
    is_enabled=True
)
def my_tool(param1: str, param2: int) -> str:
    ...

# 工厂函数用法
tool = function_tool(my_function, name_override="custom_name")
```

**功能描述：**
将Python函数转换为代理可用的工具。自动从函数签名和文档字符串提取工具元数据。

**请求参数：**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `func` | `Callable \| None` | `None` | 要包装的函数（装饰器自动传递） |
| `name_override` | `str \| None` | `None` | 覆盖工具名称（默认使用函数名） |
| `description_override` | `str \| None` | `None` | 覆盖工具描述（默认使用docstring） |
| `docstring_style` | `DocstringStyle \| None` | `None` | 文档字符串格式（google/numpy/sphinx） |
| `use_docstring_info` | `bool` | `True` | 是否从docstring提取参数描述 |
| `failure_error_function` | `ToolErrorFunction \| None` | `None` | 错误处理函数 |
| `strict_mode` | `bool` | `True` | 使用严格JSON Schema模式 |
| `is_enabled` | `bool \| Callable` | `True` | 工具是否启用（可动态控制） |

**返回结构：**
```python
FunctionTool  # 可直接传递给 Agent(tools=[...])
```

**使用示例：**

```python
from agents import Agent, function_tool, RunContextWrapper

# 基础用法：简单函数
@function_tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 带参数的函数
@function_tool
def calculate(expression: str) -> str:
    """
    计算数学表达式的结果
    
    Args:
        expression: 要计算的数学表达式，例如 "2 + 2"
    
    Returns:
        计算结果的字符串表示
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

# 带上下文的函数
@function_tool
def query_database(
    sql: str,
    context: RunContextWrapper
) -> str:
    """
    执行数据库查询
    
    Args:
        sql: SQL查询语句
        context: 运行上下文，包含数据库连接
    
    Returns:
        查询结果
    """
    db = context.context.database  # 访问用户提供的数据库连接
    results = db.execute(sql)
    return str(results)

# 自定义配置
@function_tool(
    name_override="search_product",
    description_override="在产品数据库中搜索商品",
    strict_mode=True
)
def product_search(keyword: str, limit: int = 10) -> str:
    """搜索商品"""
    # 实际搜索逻辑
    return f"找到 {limit} 个包含'{keyword}'的商品"

# 动态启用/禁用
def is_premium_user(context: RunContextWrapper, agent) -> bool:
    """只有高级用户可以使用此工具"""
    return context.context.get("subscription") == "premium"

@function_tool(is_enabled=is_premium_user)
def premium_feature(param: str) -> str:
    """高级功能，仅限高级用户"""
    return f"执行高级功能: {param}"

# 自定义错误处理
def custom_error_handler(context: RunContextWrapper, error: Exception) -> str:
    """自定义错误消息"""
    return f"操作失败，请稍后重试。技术详情已记录（错误ID: {id(error)}）"

@function_tool(failure_error_function=custom_error_handler)
def risky_operation(data: str) -> str:
    """可能失败的操作"""
    # 可能抛出异常的操作
    return process_data(data)

# 创建代理并使用工具
agent = Agent(
    name="ToolUser",
    instructions="你可以使用多种工具来完成任务。",
    tools=[
        get_current_time,
        calculate,
        query_database,
        product_search,
        premium_feature
    ]
)
```

**函数签名要求：**

**支持的参数类型：**
```python
# 基本类型
def tool(
    text: str,           # 字符串
    number: int,         # 整数
    decimal: float,      # 浮点数
    flag: bool,          # 布尔值
) -> str:
    ...

# 复杂类型
def tool(
    items: list[str],    # 列表
    config: dict,        # 字典
    optional: str | None # 可选参数
) -> str:
    ...

# 带默认值
def tool(
    required: str,
    optional: int = 10,
    flag: bool = False
) -> str:
    ...

# 带上下文（自动注入，不传递给LLM）
def tool(
    param: str,
    context: RunContextWrapper  # 自动注入
) -> str:
    ...

def tool(
    param: str,
    context: ToolContext  # 包含更多信息的上下文
) -> str:
    ...
```

**返回值要求：**
- 必须返回字符串或可转换为字符串的类型
- 返回值会传递回模型作为工具输出

**异常处理：**
```python
@function_tool
def may_fail(param: str) -> str:
    """可能失败的工具"""
    try:
        # 执行操作
        result = risky_operation(param)
        return str(result)
    except ValueError as e:
        # 返回错误消息（模型可以看到）
        return f"参数错误: {e}"
    except Exception as e:
        # 抛出异常会中断执行
        raise RuntimeError(f"严重错误: {e}")
```

### 2.2 FunctionTool 数据类

**数据结构：**
```python
@dataclass
class FunctionTool:
    name: str                    # 工具名称
    description: str             # 工具描述
    params_json_schema: dict     # 参数JSON Schema
    on_invoke_tool: Callable     # 调用函数
    strict_json_schema: bool = True  # 严格模式
    is_enabled: bool | Callable = True  # 是否启用
    tool_input_guardrails: list | None = None  # 输入防护
    tool_output_guardrails: list | None = None  # 输出防护
```

**字段说明：**

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `name` | `str` | 工具名称，展示给LLM |
| `description` | `str` | 工具功能描述，帮助LLM理解用途 |
| `params_json_schema` | `dict` | 参数的JSON Schema定义 |
| `on_invoke_tool` | `Callable` | 实际执行函数 |
| `strict_json_schema` | `bool` | 是否使用严格JSON Schema |
| `is_enabled` | `bool \| Callable` | 工具启用状态或判断函数 |
| `tool_input_guardrails` | `list \| None` | 工具输入防护检查列表 |
| `tool_output_guardrails` | `list \| None` | 工具输出防护检查列表 |

**手动创建示例：**
```python
from agents.tool import FunctionTool
from agents.function_schema import function_schema

def my_function(param1: str, param2: int) -> str:
    return f"{param1}: {param2}"

# 手动创建FunctionTool
tool = FunctionTool(
    name="my_tool",
    description="这是一个示例工具",
    params_json_schema=function_schema(my_function),
    on_invoke_tool=lambda ctx, args_json: my_function(**json.loads(args_json)),
    strict_json_schema=True,
    is_enabled=True
)
```

## 3. 托管工具 API

### 3.1 FileSearchTool - 文件搜索工具

**API 签名：**
```python
@dataclass
class FileSearchTool:
    def __init__(
        self,
        vector_store_ids: list[str],
        max_num_results: int | None = None,
        include_search_results: bool = False,
        ranking_options: RankingOptions | None = None,
        filters: Filters | None = None
    )
```

**功能描述：**
让代理能够搜索向量存储中的文件内容。基于OpenAI的文件搜索服务，支持语义搜索和相关性排序。

**请求参数：**

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `vector_store_ids` | `list[str]` | 是 | - | 向量存储ID列表 |
| `max_num_results` | `int \| None` | 否 | `None` | 返回的最大结果数 |
| `include_search_results` | `bool` | 否 | `False` | 是否在LLM输出中包含搜索结果 |
| `ranking_options` | `RankingOptions \| None` | 否 | `None` | 排序选项配置 |
| `filters` | `Filters \| None` | 否 | `None` | 文件属性过滤器 |

**使用示例：**
```python
from agents import Agent, FileSearchTool

# 创建文件搜索工具
file_search = FileSearchTool(
    vector_store_ids=["vs_abc123", "vs_def456"],
    max_num_results=10,
    include_search_results=True
)

# 创建带文件搜索的代理
agent = Agent(
    name="DocumentAssistant",
    instructions="你可以搜索文档库来回答问题。",
    tools=[file_search]
)

# 代理会自动使用file_search工具查找相关文档
result = await Runner.run(
    agent,
    "请查找关于产品定价的文档"
)
```

**工作原理：**
1. LLM决定需要搜索文档
2. 自动调用文件搜索服务
3. 返回相关文档片段
4. LLM基于搜索结果生成回答

### 3.2 WebSearchTool - 网页搜索工具

**API 签名：**
```python
@dataclass
class WebSearchTool:
    def __init__(
        self,
        user_location: UserLocation | None = None,
        filters: WebSearchToolFilters | None = None,
        search_context_size: Literal["low", "medium", "high"] = "medium"
    )
```

**功能描述：**
让代理能够搜索互联网获取实时信息。支持位置定制和搜索结果过滤。

**请求参数：**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `user_location` | `UserLocation \| None` | `None` | 用户位置（影响搜索结果相关性） |
| `filters` | `WebSearchToolFilters \| None` | `None` | 搜索结果过滤器 |
| `search_context_size` | `"low" \| "medium" \| "high"` | `"medium"` | 搜索上下文大小 |

**使用示例：**
```python
from agents import Agent, WebSearchTool
from openai.types.responses.web_search_tool_param import UserLocation

# 创建网页搜索工具
web_search = WebSearchTool(
    user_location=UserLocation(
        city="Beijing",
        country="CN"
    ),
    search_context_size="high"
)

# 创建搜索代理
agent = Agent(
    name="ResearchAssistant",
    instructions="你可以搜索网页获取最新信息。",
    tools=[web_search]
)

# 代理会自动搜索网页
result = await Runner.run(
    agent,
    "最新的AI技术发展趋势是什么？"
)
```

**search_context_size 说明：**
- `"low"`：快速搜索，较少上下文
- `"medium"`：平衡速度和信息量
- `"high"`：深度搜索，更多上下文

### 3.3 CodeInterpreterTool - 代码解释器工具

**API 签名：**
```python
@dataclass
class CodeInterpreterTool:
    def __init__(
        self,
        tool_config: CodeInterpreter
    )
```

**功能描述：**
让代理能够在沙箱环境中执行Python代码。支持数据分析、计算、可视化等任务。

**使用示例：**
```python
from agents import Agent, CodeInterpreterTool
from openai.types.responses.tool_param import CodeInterpreter

# 创建代码解释器工具
code_interpreter = CodeInterpreterTool(
    tool_config=CodeInterpreter(
        container="default"  # 沙箱容器配置
    )
)

# 创建数据分析代理
agent = Agent(
    name="DataAnalyst",
    instructions="你可以编写和执行Python代码进行数据分析。",
    tools=[code_interpreter]
)

# 代理会自动编写和执行代码
result = await Runner.run(
    agent,
    "请分析这组数据的统计特征：[1, 2, 3, 4, 5, 10, 20, 30]"
)
```

**支持的操作：**
- 数学计算
- 数据处理和分析
- 图表生成
- 文件读写（沙箱内）

### 3.4 ImageGenerationTool - 图像生成工具

**API 签名：**
```python
@dataclass
class ImageGenerationTool:
    def __init__(
        self,
        tool_config: ImageGeneration
    )
```

**功能描述：**
让代理能够生成图像。基于文本描述创建图像。

**使用示例：**
```python
from agents import Agent, ImageGenerationTool
from openai.types.responses.tool_param import ImageGeneration

# 创建图像生成工具
image_gen = ImageGenerationTool(
    tool_config=ImageGeneration()
)

# 创建创意代理
agent = Agent(
    name="CreativeAssistant",
    instructions="你可以根据描述生成图像。",
    tools=[image_gen]
)

# 代理会自动生成图像
result = await Runner.run(
    agent,
    "请创建一张未来城市的概念图"
)
```

## 4. 高级工具 API

### 4.1 ComputerTool - 计算机控制工具

**API 签名：**
```python
@dataclass
class ComputerTool:
    def __init__(
        self,
        computer: Computer | AsyncComputer,
        on_safety_check: Callable[[ComputerToolSafetyCheckData], bool] | None = None
    )
```

**功能描述：**
让代理能够控制计算机，包括鼠标移动、点击、键盘输入、截图等操作。需要实现 `Computer` 或 `AsyncComputer` 接口。

**请求参数：**

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `computer` | `Computer \| AsyncComputer` | 是 | 计算机实现，定义环境和操作 |
| `on_safety_check` | `Callable \| None` | 否 | 安全检查回调函数 |

**使用示例：**
```python
from agents import Agent, ComputerTool
from agents.computer import Computer

class MyComputer(Computer):
    """自定义计算机实现"""
    
    @property
    def display_width_px(self) -> int:
        return 1920
    
    @property
    def display_height_px(self) -> int:
        return 1080
    
    async def screenshot(self) -> bytes:
        """截取屏幕"""
        # 实现截图逻辑
        return screenshot_bytes
    
    async def mouse_move(self, x: int, y: int):
        """移动鼠标"""
        # 实现鼠标移动
        pass
    
    async def mouse_click(self, button: str):
        """点击鼠标"""
        # 实现鼠标点击
        pass
    
    async def keyboard_type(self, text: str):
        """输入文本"""
        # 实现键盘输入
        pass

# 安全检查回调
def safety_check(data: ComputerToolSafetyCheckData) -> bool:
    """检查计算机操作是否安全"""
    action = data.tool_call.action
    
    # 阻止危险操作
    if action == "delete_file":
        return False
    
    # 允许其他操作
    return True

# 创建计算机工具
computer_tool = ComputerTool(
    computer=MyComputer(),
    on_safety_check=safety_check
)

# 创建自动化代理
agent = Agent(
    name="AutomationAgent",
    instructions="你可以控制计算机完成任务。",
    tools=[computer_tool]
)
```

**支持的操作：**
- 鼠标移动和点击
- 键盘输入
- 截图
- 窗口控制

### 4.2 HostedMCPTool - 托管MCP工具

**API 签名：**
```python
@dataclass
class HostedMCPTool:
    def __init__(
        self,
        tool_config: Mcp,
        on_approval_request: MCPToolApprovalFunction | None = None
    )
```

**功能描述：**
让代理能够使用远程MCP（Model Context Protocol）服务器提供的工具。LLM自动列出和调用工具，无需代码往返。

**请求参数：**

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `tool_config` | `Mcp` | 是 | MCP工具配置，包含服务器URL等 |
| `on_approval_request` | `MCPToolApprovalFunction \| None` | 否 | 工具调用批准函数 |

**使用示例：**
```python
from agents import Agent, HostedMCPTool
from openai.types.responses.tool_param import Mcp

# 批准函数
async def approve_mcp_tool(
    request: MCPToolApprovalRequest
) -> MCPToolApprovalFunctionResult:
    """审批MCP工具调用"""
    
    tool_name = request.data.tool_name
    
    # 自动批准某些工具
    if tool_name in ["read_file", "list_directory"]:
        return {"approve": True}
    
    # 拒绝敏感工具
    if tool_name == "delete_file":
        return {
            "approve": False,
            "reason": "删除文件需要人工批准"
        }
    
    # 默认批准
    return {"approve": True}

# 创建托管MCP工具
mcp_tool = HostedMCPTool(
    tool_config=Mcp(
        server_url="https://mcp.example.com",
        api_key="your-api-key"
    ),
    on_approval_request=approve_mcp_tool
)

# 创建集成代理
agent = Agent(
    name="MCPAgent",
    instructions="你可以使用MCP服务器的工具。",
    tools=[mcp_tool]
)
```

### 4.3 LocalShellTool - 本地Shell工具

**API 签名：**
```python
@dataclass
class LocalShellTool:
    def __init__(
        self,
        executor: LocalShellExecutor
    )
```

**功能描述：**
让代理能够执行本地Shell命令。需要提供命令执行器函数。

**请求参数：**

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `executor` | `LocalShellExecutor` | 是 | Shell命令执行函数 |

**使用示例：**
```python
from agents import Agent, LocalShellTool
import subprocess

# 命令执行器
async def shell_executor(
    request: LocalShellCommandRequest
) -> str:
    """执行Shell命令"""
    
    command = request.data.command
    
    # 安全检查
    if any(danger in command for danger in ["rm -rf", "format", "del"]):
        return "错误：命令包含危险操作"
    
    try:
        # 执行命令
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return result.stdout or result.stderr
    
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时"
    except Exception as e:
        return f"错误：{e}"

# 创建本地Shell工具
shell_tool = LocalShellTool(executor=shell_executor)

# 创建系统管理代理
agent = Agent(
    name="SysAdminAgent",
    instructions="你可以执行系统命令完成任务。",
    tools=[shell_tool]
)
```

## 5. 工具防护 API

### 5.1 工具级防护配置

**ToolInputGuardrail：**
```python
from agents.tool_guardrails import ToolInputGuardrail, ToolInputGuardrailResult

class ParameterValidationGuardrail(ToolInputGuardrail):
    """参数验证防护"""
    
    async def run(
        self,
        tool_name: str,
        arguments: dict,
        context: Any
    ) -> ToolInputGuardrailResult:
        """检查工具参数"""
        
        # 检查参数范围
        if "amount" in arguments:
            amount = arguments["amount"]
            if amount > 1000:
                return ToolInputGuardrailResult(
                    output=ToolGuardrailFunctionOutput(
                        behavior=RejectContentBehavior(
                            reject_message="金额超过限制"
                        )
                    )
                )
        
        # 参数合法
        return ToolInputGuardrailResult(
            output=ToolGuardrailFunctionOutput(
                behavior=AllowBehavior()
            )
        )

# 应用到工具
@function_tool
def transfer_money(amount: float, to_account: str) -> str:
    """转账"""
    return f"转账 {amount} 到 {to_account}"

# 手动添加防护
transfer_money.tool_input_guardrails = [ParameterValidationGuardrail()]
```

**ToolOutputGuardrail：**
```python
from agents.tool_guardrails import ToolOutputGuardrail, ToolOutputGuardrailResult

class SensitiveDataGuardrail(ToolOutputGuardrail):
    """敏感数据防护"""
    
    async def run(
        self,
        tool_name: str,
        output: str,
        context: Any
    ) -> ToolOutputGuardrailResult:
        """检查工具输出"""
        
        # 检查敏感信息
        if contains_credit_card(output):
            return ToolOutputGuardrailResult(
                output=ToolGuardrailFunctionOutput(
                    behavior=RejectContentBehavior(
                        reject_message="输出包含信用卡信息"
                    )
                )
            )
        
        # 输出安全
        return ToolOutputGuardrailResult(
            output=ToolGuardrailFunctionOutput(
                behavior=AllowBehavior()
            )
        )
```

## 6. 工具组合与管理

### 6.1 工具集合

```python
# 定义工具集
basic_tools = [
    get_current_time,
    calculate
]

advanced_tools = [
    query_database,
    send_email
]

# 创建不同能力的代理
basic_agent = Agent(
    name="BasicAgent",
    tools=basic_tools
)

advanced_agent = Agent(
    name="AdvancedAgent",
    tools=basic_tools + advanced_tools
)
```

### 6.2 动态工具启用

```python
def tool_availability(context: RunContextWrapper, agent) -> bool:
    """根据上下文动态控制工具可用性"""
    
    # 根据用户权限
    user_role = context.context.get("role")
    if user_role == "admin":
        return True
    
    # 根据时间
    hour = datetime.now().hour
    if 9 <= hour <= 17:  # 工作时间
        return True
    
    return False

@function_tool(is_enabled=tool_availability)
def admin_operation(action: str) -> str:
    """管理员操作"""
    return f"执行管理操作: {action}"
```

Tools 模块通过灵活的工具定义和丰富的工具类型，为 OpenAI Agents 提供了强大的能力扩展机制，支持从简单函数调用到复杂系统集成的各种应用场景。

