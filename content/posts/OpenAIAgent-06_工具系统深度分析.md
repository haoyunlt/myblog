---
title: "OpenAI Agents SDK 工具系统深度分析"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['源码分析', '技术文档']
categories: ['技术分析']
description: "OpenAI Agents SDK 工具系统深度分析的深入技术分析文档"
keywords: ['源码分析', '技术文档']
author: "技术分析师"
weight: 1
---

## 6.1 工具系统架构概览

工具系统是OpenAI Agents SDK的核心功能之一，它允许代理调用外部函数、服务和API来扩展其能力。工具系统采用分层设计，包含：

- **工具接口层**: 定义统一的工具接口
- **函数工具层**: 支持Python函数作为工具
- **内置工具层**: 提供常用的预制工具
- **MCP工具层**: 支持Model Context Protocol
- **工具执行层**: 负责工具的调用和结果处理

## 6.2 工具系统架构图

```mermaid
graph TD
    A[Agent代理] --> B[Tool工具接口]
    B --> C{工具类型}
    
    C --> D[FunctionTool 函数工具]
    C --> E[内置工具]
    C --> F[MCPTool MCP工具]
    
    D --> G[@function_tool 装饰器]
    G --> H[Python函数]
    
    E --> I[FileSearchTool 文件搜索]
    E --> J[CodeInterpreterTool 代码解释器]
    E --> K[WebSearchTool 网络搜索]
    E --> L[ImageGenerationTool 图像生成]
    E --> M[ComputerTool 计算机操作]
    E --> N[LocalShellTool 本地Shell]
    
    F --> O[MCP服务器]
    O --> P[远程工具服务]
    
    B --> Q[ToolContext 工具上下文]
    Q --> R[RunContextWrapper 运行上下文]
    Q --> S[工具调用信息]
    
    style B fill:#e3f2fd
    style G fill:#f3e5f5
    style Q fill:#e8f5e8
```

## 6.3 Tool基础接口分析

### 6.3.1 Tool抽象基类

```python
# 位于 src/agents/tool.py
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

class Tool(ABC):
    """
    所有工具的抽象基类
    
    定义了工具的基本接口，所有具体工具类型都必须实现这个接口
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具的名称，用于LLM识别和调用"""
        pass
    
    @property  
    @abstractmethod
    def description(self) -> str:
        """工具的描述，帮助LLM理解工具的功能和使用场景"""
        pass
    
    @abstractmethod
    async def execute(
        self, 
        context: ToolContext[Any], 
        arguments: str
    ) -> Any:
        """
        执行工具的核心方法
        
        Args:
            context: 工具执行上下文，包含运行时信息
            arguments: 工具参数的JSON字符串
            
        Returns:
            Any: 工具执行结果
        """
        pass
    
    @property
    def is_enabled(self) -> bool | Callable[[RunContextWrapper[Any], AgentBase], MaybeAwaitable[bool]]:
        """
        工具是否启用
        
        可以是布尔值或返回布尔值的函数，支持动态启用/禁用
        """
        return True
```

### 6.3.2 ToolContext工具上下文

```python
# 位于 src/agents/tool_context.py
@dataclass
class ToolContext(Generic[TContext]):
    """
    工具执行的上下文信息
    
    提供工具执行时需要的所有运行时信息和环境
    """
    
    run_context: RunContextWrapper[TContext]      # 运行上下文包装器
    agent: AgentBase                              # 当前执行的代理
    tool_name: str                               # 工具名称
    tool_call_id: str                           # 工具调用ID
    tool_arguments: str                         # 工具参数JSON字符串
    model_provider: ModelProvider               # 模型提供商
    
    @property
    def context(self) -> TContext:
        """获取用户自定义的上下文对象"""
        return self.run_context.context
    
    @property
    def usage(self) -> Usage:
        """获取当前的使用统计信息"""
        return self.run_context.usage
    
    def create_sub_agent(
        self, 
        name: str,
        instructions: str,
        **agent_kwargs
    ) -> Agent[TContext]:
        """
        在工具中创建子代理
        
        这允许工具创建专门的代理来处理特定任务
        """
        return Agent(
            name=name,
            instructions=instructions,
            model_provider=self.model_provider,
            **agent_kwargs
        )
```

## 6.4 FunctionTool详细实现

### 6.4.1 FunctionTool类定义

```python
# 位于 src/agents/tool.py
@dataclass
class FunctionTool(Tool):
    """
    函数工具实现类
    
    将Python函数包装为代理可调用的工具
    支持多种函数签名模式和参数处理
    """
    
    name: str                                        # 工具名称
    description: str                                # 工具描述  
    params_json_schema: dict[str, Any]              # 参数JSON Schema
    on_invoke_tool: Callable[[ToolContext[Any], str], Awaitable[Any]]  # 工具调用处理函数
    strict_json_schema: bool = True                 # 是否使用严格JSON Schema
    is_enabled: bool | Callable[[RunContextWrapper[Any], AgentBase], MaybeAwaitable[bool]] = True  # 启用状态
    
    def __post_init__(self):
        """初始化后处理，确保JSON Schema的严格性"""
        if self.strict_json_schema:
            self.params_json_schema = ensure_strict_json_schema(self.params_json_schema)
    
    async def execute(self, context: ToolContext[Any], arguments: str) -> Any:
        """
        执行函数工具
        
        调用on_invoke_tool处理函数来执行实际的Python函数
        """
        return await self.on_invoke_tool(context, arguments)
    
    def to_openai_format(self) -> dict[str, Any]:
        """
        转换为OpenAI API工具格式
        
        用于向OpenAI API传递工具定义
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_json_schema,
                "strict": self.strict_json_schema,
            }
        }
```

### 6.4.2 function_tool装饰器深度实现

```python
# 位于 src/agents/tool.py
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
    function_tool装饰器的完整实现
    
    支持三种函数签名模式：
    1. 普通函数：func(param1, param2, ...)
    2. 带运行上下文：func(context: RunContextWrapper, param1, param2, ...)
    3. 带工具上下文：func(tool_context: ToolContext, param1, param2, ...)
    """
    
    def decorator(f: ToolFunction[ToolParams]) -> FunctionTool:
        # 1. 分析函数签名
        signature = inspect.signature(f)
        function_name = name_override or f.__name__
        
        # 2. 确定上下文参数类型
        context_info = _analyze_context_parameter(signature)
        
        # 3. 生成工具描述
        description = description_override or _extract_description(f, docstring_style)
        if not description:
            description = f"Execute function {function_name}"
        
        # 4. 生成参数JSON Schema
        try:
            params_schema = function_schema(
                f, 
                strict=strict_json_schema,
                docstring_style=docstring_style
            )
        except Exception as e:
            raise ValueError(f"Failed to generate schema for function {function_name}: {e}")
        
        # 5. 创建工具调用处理函数
        async def on_invoke_tool(tool_context: ToolContext[Any], args_json: str) -> Any:
            """处理工具调用的内部实现"""
            
            try:
                # 解析参数
                if args_json.strip():
                    try:
                        args = json.loads(args_json)
                    except json.JSONDecodeError as e:
                        error_msg = f"Invalid JSON arguments for tool {function_name}: {e}"
                        logger.error(error_msg)
                        return error_msg
                else:
                    args = {}
                
                # 验证参数
                validation_result = _validate_arguments(args, params_schema)
                if not validation_result.valid:
                    error_msg = f"Invalid arguments for tool {function_name}: {validation_result.error}"
                    logger.error(error_msg)
                    return error_msg
                
                # 根据函数签名调用函数
                if context_info.has_context:
                    if context_info.context_type == "run_context":
                        # 传递RunContextWrapper
                        result = f(tool_context.run_context, **args)
                    elif context_info.context_type == "tool_context":
                        # 传递ToolContext
                        result = f(tool_context, **args)
                    else:
                        raise ValueError(f"Unknown context type: {context_info.context_type}")
                else:
                    # 普通函数，不传递上下文
                    result = f(**args)
                
                # 处理异步结果
                if inspect.isawaitable(result):
                    result = await result
                
                return result
                
            except TypeError as e:
                # 参数类型错误
                error_msg = f"Type error in tool {function_name}: {str(e)}"
                logger.error(error_msg)
                return error_msg
                
            except Exception as e:
                # 其他执行错误
                error_msg = f"Error executing tool {function_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # 检查是否有自定义错误处理
                error_handler = getattr(f, '_tool_error_handler', None)
                if error_handler and callable(error_handler):
                    try:
                        handled_result = error_handler(e, tool_context, args)
                        if inspect.isawaitable(handled_result):
                            handled_result = await handled_result
                        return handled_result
                    except Exception:
                        pass  # 错误处理器本身出错，继续返回原始错误
                
                return error_msg
        
        # 6. 创建FunctionTool实例
        return FunctionTool(
            name=function_name,
            description=description,
            params_json_schema=params_schema,
            on_invoke_tool=on_invoke_tool,
            strict_json_schema=strict_json_schema,
            is_enabled=is_enabled,
        )
    
    # 支持直接装饰和参数化装饰
    if func is not None:
        return decorator(func)
    else:
        return decorator

def _analyze_context_parameter(signature: inspect.Signature) -> ContextInfo:
    """
    分析函数签名中的上下文参数
    
    返回上下文信息，包括是否有上下文参数以及类型
    """
    params = list(signature.parameters.values())
    if not params:
        return ContextInfo(has_context=False)
    
    first_param = params[0]
    
    # 检查类型注解
    if hasattr(first_param, 'annotation') and first_param.annotation != inspect.Parameter.empty:
        annotation = first_param.annotation
        
        # 检查是否是RunContextWrapper类型
        if _is_run_context_type(annotation):
            return ContextInfo(has_context=True, context_type="run_context")
        
        # 检查是否是ToolContext类型  
        elif _is_tool_context_type(annotation):
            return ContextInfo(has_context=True, context_type="tool_context")
    
    # 根据参数名称推断
    if first_param.name in ["context", "run_context"]:
        return ContextInfo(has_context=True, context_type="run_context")
    elif first_param.name in ["tool_context", "tool_ctx"]:
        return ContextInfo(has_context=True, context_type="tool_context")
    
    return ContextInfo(has_context=False)

@dataclass
class ContextInfo:
    """上下文参数信息"""
    has_context: bool
    context_type: str | None = None
```

### 6.4.3 函数Schema生成

```python
# 位于 src/agents/function_schema.py
def function_schema(
    func: Callable,
    *,
    strict: bool = True,
    docstring_style: DocstringStyle = "google",
    exclude_first_param: bool = False,
) -> dict[str, Any]:
    """
    从Python函数生成JSON Schema
    
    支持从类型注解、文档字符串等提取参数信息
    """
    
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    
    # 排除上下文参数
    if exclude_first_param and parameters:
        # 检查第一个参数是否是上下文参数
        first_param = parameters[0]
        if _is_context_parameter(first_param):
            parameters = parameters[1:]
    
    # 解析文档字符串
    docstring = inspect.getdoc(func) or ""
    param_docs = _parse_docstring_params(docstring, docstring_style)
    
    # 构建JSON Schema
    schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    
    for param in parameters:
        param_name = param.name
        param_type = param.annotation
        param_default = param.default
        
        # 生成参数schema
        param_schema = _generate_param_schema(
            param_name=param_name,
            param_type=param_type,
            param_default=param_default,
            param_doc=param_docs.get(param_name, ""),
        )
        
        schema["properties"][param_name] = param_schema
        
        # 判断是否为必需参数
        if param_default is inspect.Parameter.empty:
            schema["required"].append(param_name)
    
    # 应用严格模式
    if strict:
        schema = ensure_strict_json_schema(schema)
    
    return schema

def _generate_param_schema(
    param_name: str,
    param_type: Any,
    param_default: Any,
    param_doc: str,
) -> dict[str, Any]:
    """为单个参数生成JSON Schema"""
    
    schema = {"description": param_doc or f"Parameter {param_name}"}
    
    # 处理类型注解
    if param_type == inspect.Parameter.empty:
        # 无类型注解，尝试从默认值推断
        if param_default != inspect.Parameter.empty:
            schema.update(_infer_type_from_value(param_default))
        else:
            schema["type"] = "string"  # 默认为字符串
    else:
        # 根据类型注解生成schema
        schema.update(_type_to_json_schema(param_type))
    
    # 处理默认值
    if param_default != inspect.Parameter.empty:
        schema["default"] = param_default
    
    return schema

def _type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """将Python类型转换为JSON Schema"""
    
    # 基本类型映射
    basic_types = {
        str: {"type": "string"},
        int: {"type": "integer"}, 
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    
    if python_type in basic_types:
        return basic_types[python_type]
    
    # 处理泛型类型
    origin = get_origin(python_type)
    args = get_args(python_type)
    
    if origin is list:
        # List[T] -> array with items schema
        if args:
            item_schema = _type_to_json_schema(args[0])
            return {"type": "array", "items": item_schema}
        else:
            return {"type": "array"}
    
    elif origin is dict:
        # Dict[str, T] -> object with additionalProperties schema
        if len(args) >= 2:
            value_schema = _type_to_json_schema(args[1])
            return {"type": "object", "additionalProperties": value_schema}
        else:
            return {"type": "object"}
    
    elif origin is Union:
        # Union类型 -> oneOf或anyOf
        union_schemas = [_type_to_json_schema(arg) for arg in args if arg is not type(None)]
        
        if len(union_schemas) == 1:
            # Optional[T] case
            return union_schemas[0]
        else:
            return {"anyOf": union_schemas}
    
    # 处理Literal类型
    elif origin is Literal:
        return {"enum": list(args)}
    
    # 处理Annotated类型
    elif origin is Annotated:
        base_type = args[0]
        annotations = args[1:]
        
        base_schema = _type_to_json_schema(base_type)
        
        # 从注解中提取额外信息
        for annotation in annotations:
            if isinstance(annotation, str):
                # 字符串注解作为描述
                base_schema["description"] = annotation
            elif hasattr(annotation, '__dict__'):
                # 其他注解对象的属性
                for key, value in annotation.__dict__.items():
                    if key.startswith('_'):
                        continue
                    base_schema[key] = value
        
        return base_schema
    
    # 处理Pydantic模型
    elif hasattr(python_type, 'model_json_schema'):
        return python_type.model_json_schema()
    
    # 处理dataclass
    elif hasattr(python_type, '__dataclass_fields__'):
        return _dataclass_to_json_schema(python_type)
    
    # 处理Enum
    elif hasattr(python_type, '__members__'):
        return {"enum": list(python_type.__members__.keys())}
    
    # 默认情况
    return {"type": "string"}
```

## 6.5 内置工具详细分析

### 6.5.1 FileSearchTool - 文件搜索工具

```python
# 位于 src/agents/tool.py
@dataclass
class FileSearchTool(Tool):
    """
    文件搜索工具
    
    允许代理在指定文件中搜索信息
    使用OpenAI的文件搜索功能
    """
    
    file_ids: list[str]                              # OpenAI文件ID列表
    filters: Filters | None = None                   # 搜索过滤器
    ranking_options: RankingOptions | None = None    # 排序选项
    description: str = "Search through uploaded files"  # 工具描述
    
    @property
    def name(self) -> str:
        return "file_search"
    
    async def execute(self, context: ToolContext[Any], arguments: str) -> str:
        """执行文件搜索"""
        # 文件搜索通常由模型服务商处理，这里返回配置信息
        return f"File search configured for files: {self.file_ids}"
    
    def to_openai_format(self) -> dict[str, Any]:
        """转换为OpenAI API格式"""
        tool_def = {"type": "file_search"}
        
        if self.file_ids:
            tool_def["file_search"] = {"file_ids": self.file_ids}
            
        if self.filters:
            tool_def["file_search"]["filters"] = self.filters
            
        if self.ranking_options:
            tool_def["file_search"]["ranking_options"] = self.ranking_options
            
        return tool_def
```

### 6.5.2 CodeInterpreterTool - 代码解释器工具

```python
# 位于 src/agents/tool.py
@dataclass
class CodeInterpreterTool(Tool):
    """
    代码解释器工具
    
    允许代理执行Python代码并获取结果
    """
    
    description: str = "Execute Python code and analyze data"
    
    @property
    def name(self) -> str:
        return "code_interpreter"
    
    async def execute(self, context: ToolContext[Any], arguments: str) -> str:
        """执行代码解释器"""
        # 代码执行通常由模型服务商的沙箱环境处理
        return "Code interpreter available for Python execution"
    
    def to_openai_format(self) -> dict[str, Any]:
        """转换为OpenAI API格式"""
        return {"type": "code_interpreter"}
```

### 6.5.3 ComputerTool - 计算机操作工具

```python
# 位于 src/agents/tool.py
@dataclass 
class ComputerTool(Tool):
    """
    计算机操作工具
    
    允许代理执行屏幕截图、点击、输入等计算机操作
    需要配合Computer实例使用
    """
    
    computer: Computer | AsyncComputer               # 计算机操作实例
    description: str = "Control computer screen, keyboard and mouse"
    
    @property
    def name(self) -> str:
        return "computer"
    
    async def execute(self, context: ToolContext[Any], arguments: str) -> str:
        """执行计算机操作"""
        try:
            args = json.loads(arguments) if arguments else {}
            action = args.get("action")
            
            if action == "screenshot":
                # 获取屏幕截图
                if hasattr(self.computer, 'screenshot'):
                    result = await self.computer.screenshot()
                    return f"Screenshot taken: {result}"
                else:
                    return "Screenshot not supported by current computer instance"
            
            elif action == "click":
                # 执行鼠标点击
                x = args.get("coordinate", [0, 0])[0]
                y = args.get("coordinate", [0, 0])[1]
                if hasattr(self.computer, 'click'):
                    await self.computer.click(x, y)
                    return f"Clicked at ({x}, {y})"
                else:
                    return "Click not supported"
            
            elif action == "type":
                # 输入文本
                text = args.get("text", "")
                if hasattr(self.computer, 'type'):
                    await self.computer.type(text)
                    return f"Typed: {text}"
                else:
                    return "Typing not supported"
            
            else:
                return f"Unknown computer action: {action}"
                
        except Exception as e:
            return f"Computer tool error: {str(e)}"
    
    def to_openai_format(self) -> dict[str, Any]:
        """转换为OpenAI API格式"""
        return {
            "type": "computer",
            "computer": {
                "display_width_px": getattr(self.computer, 'display_width', 1920),
                "display_height_px": getattr(self.computer, 'display_height', 1080),
                "display_number": getattr(self.computer, 'display_number', None),
            }
        }
```

### 6.5.4 LocalShellTool - 本地Shell工具

```python
# 位于 src/agents/tool.py
@dataclass
class LocalShellTool(Tool):
    """
    本地Shell命令执行工具
    
    允许代理执行本地系统命令
    需要谨慎使用，建议配合安全防护
    """
    
    executor: LocalShellExecutor                     # Shell执行器
    description: str = "Execute local shell commands"
    
    @property
    def name(self) -> str:
        return "local_shell"
    
    async def execute(self, context: ToolContext[Any], arguments: str) -> str:
        """执行Shell命令"""
        try:
            args = json.loads(arguments) if arguments else {}
            command = args.get("command", "")
            
            if not command:
                return "Error: No command provided"
            
            # 执行命令
            result = await self.executor.execute(
                LocalShellCommandRequest(command=command)
            )
            
            if result.exit_code == 0:
                return f"Command executed successfully:\n{result.stdout}"
            else:
                return f"Command failed (exit code {result.exit_code}):\n{result.stderr}"
                
        except Exception as e:
            return f"Shell execution error: {str(e)}"
    
    def to_openai_format(self) -> dict[str, Any]:
        """转换为OpenAI API格式"""
        return {"type": "local_shell"}

@dataclass
class LocalShellCommandRequest:
    """Shell命令请求"""
    command: str                    # 要执行的命令
    timeout: int = 30              # 超时时间（秒）
    working_directory: str | None = None  # 工作目录

@dataclass  
class LocalShellCommandResult:
    """Shell命令执行结果"""
    exit_code: int                 # 退出码
    stdout: str                    # 标准输出
    stderr: str                    # 标准错误输出
    execution_time: float          # 执行时间

class LocalShellExecutor:
    """本地Shell命令执行器"""
    
    async def execute(self, request: LocalShellCommandRequest) -> LocalShellCommandResult:
        """
        执行Shell命令
        
        使用asyncio.subprocess执行命令，支持超时控制
        """
        import asyncio.subprocess as subprocess
        
        start_time = time.time()
        
        try:
            # 创建子进程
            process = await subprocess.create_subprocess_shell(
                request.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=request.working_directory,
            )
            
            # 等待命令完成（带超时）
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=request.timeout
            )
            
            execution_time = time.time() - start_time
            
            return LocalShellCommandResult(
                exit_code=process.returncode or 0,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                execution_time=execution_time,
            )
            
        except asyncio.TimeoutError:
            # 命令超时，终止进程
            if process:
                process.terminate()
                await process.wait()
            
            execution_time = time.time() - start_time
            return LocalShellCommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {request.timeout} seconds",
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return LocalShellCommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time=execution_time,
            )
```

## 6.6 MCP工具系统分析

### 6.6.1 MCPTool实现

```python
# 位于 src/agents/mcp/__init__.py
@dataclass
class MCPTool(Tool):
    """
    Model Context Protocol (MCP) 工具
    
    支持通过MCP协议调用远程工具服务
    """
    
    server: MCPServer                               # MCP服务器实例  
    tool_name: str                                 # 工具名称
    tool_schema: dict[str, Any]                    # 工具Schema
    description: str                               # 工具描述
    
    @property
    def name(self) -> str:
        return self.tool_name
    
    async def execute(self, context: ToolContext[Any], arguments: str) -> Any:
        """执行MCP工具"""
        try:
            # 调用MCP服务器
            result = await self.server.call_tool(
                name=self.tool_name,
                arguments=json.loads(arguments) if arguments else {}
            )
            
            return result.content
            
        except Exception as e:
            error_msg = f"MCP tool '{self.tool_name}' failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def to_openai_format(self) -> dict[str, Any]:
        """转换为OpenAI API格式"""
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.description,
                "parameters": self.tool_schema,
            }
        }

class MCPServer:
    """MCP服务器连接管理器"""
    
    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self.client: mcp.Client | None = None
        self.session: mcp.ClientSession | None = None
    
    async def connect(self) -> None:
        """连接到MCP服务器"""
        if self.config.transport_type == "stdio":
            # 标准输入输出传输
            transport = mcp.StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env or {},
            )
        elif self.config.transport_type == "sse":
            # Server-Sent Events传输
            transport = mcp.SSEServerParameters(url=self.config.url)
        else:
            raise ValueError(f"Unsupported transport type: {self.config.transport_type}")
        
        # 创建客户端和会话
        self.client = mcp.Client(transport)
        self.session = await self.client.connect()
        
        # 获取服务器信息
        server_info = await self.session.get_server_info()
        logger.info(f"Connected to MCP server: {server_info.name}")
    
    async def get_available_tools(self) -> list[MCPTool]:
        """获取服务器提供的所有工具"""
        if not self.session:
            raise RuntimeError("MCP server not connected")
        
        # 列出可用工具
        tools_response = await self.session.list_tools()
        
        mcp_tools = []
        for tool_info in tools_response.tools:
            mcp_tool = MCPTool(
                server=self,
                tool_name=tool_info.name,
                tool_schema=tool_info.inputSchema,
                description=tool_info.description or f"MCP tool: {tool_info.name}",
            )
            mcp_tools.append(mcp_tool)
        
        return mcp_tools
    
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用MCP工具"""
        if not self.session:
            raise RuntimeError("MCP server not connected")
        
        # 调用工具
        result = await self.session.call_tool(name, arguments)
        return result
    
    async def cleanup(self) -> None:
        """清理连接资源"""
        if self.session:
            await self.session.close()
        if self.client:
            await self.client.close()

@dataclass
class MCPServerConfig:
    """MCP服务器配置"""
    name: str                                       # 服务器名称
    transport_type: Literal["stdio", "sse"]         # 传输类型
    command: str | None = None                      # 命令（stdio传输）
    args: list[str] | None = None                   # 命令参数
    env: dict[str, str] | None = None               # 环境变量
    url: str | None = None                          # URL（SSE传输）
```

## 6.7 工具错误处理和最佳实践

### 6.7.1 工具错误处理策略

```python
# 位于 src/agents/tool.py
def default_tool_error_function(
    error: Exception,
    tool_context: ToolContext[Any],
    arguments: dict[str, Any],
) -> str:
    """
    默认的工具错误处理函数
    
    提供统一的错误处理和日志记录
    """
    
    tool_name = tool_context.tool_name
    error_msg = str(error)
    
    # 根据错误类型提供不同的处理
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON arguments for tool '{tool_name}': {error_msg}"
    
    elif isinstance(error, TypeError):
        return f"Invalid arguments for tool '{tool_name}': {error_msg}"
    
    elif isinstance(error, FileNotFoundError):
        return f"Required file not found for tool '{tool_name}': {error_msg}"
    
    elif isinstance(error, PermissionError):
        return f"Permission denied for tool '{tool_name}': {error_msg}"
    
    elif isinstance(error, TimeoutError):
        return f"Tool '{tool_name}' timed out: {error_msg}"
    
    else:
        # 通用错误处理
        logger.error(f"Tool '{tool_name}' execution failed", exc_info=True)
        return f"Tool '{tool_name}' failed: {error_msg}"

def tool_error_handler(handler_func: Callable[[Exception, ToolContext[Any], dict], str]):
    """
    工具错误处理装饰器
    
    允许为特定工具定义自定义错误处理逻辑
    """
    def decorator(tool_func):
        tool_func._tool_error_handler = handler_func
        return tool_func
    return decorator

# 使用示例
@tool_error_handler(lambda e, ctx, args: f"Custom error handling: {e}")
@function_tool
def risky_operation(value: int) -> int:
    """可能出错的操作"""
    if value < 0:
        raise ValueError("Value must be non-negative")
    return value * 2
```

这个工具系统为OpenAI Agents SDK提供了强大而灵活的扩展能力，支持从简单的Python函数到复杂的外部服务集成，使代理能够与真实世界进行交互。
