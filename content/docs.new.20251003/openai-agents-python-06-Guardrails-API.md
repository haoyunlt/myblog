# OpenAI Agents Python SDK - Guardrails 模块 API 详解

## 1. API 总览

Guardrails 模块提供了输入输出验证和工具级安全控制机制。通过防护栏，开发者可以在Agent执行前后进行检查，确保系统安全和合规。

### API 分类

| API 类别 | 核心 API | 功能描述 |
|---------|---------|---------|
| **Agent级防护** | `@input_guardrail` | 装饰器，创建输入防护 |
| | `@output_guardrail` | 装饰器，创建输出防护 |
| | `InputGuardrail.run()` | 执行输入检查 |
| | `OutputGuardrail.run()` | 执行输出检查 |
| **Tool级防护** | `@tool_input_guardrail` | 装饰器，创建工具输入防护 |
| | `@tool_output_guardrail` | 装饰器，创建工具输出防护 |
| | `ToolInputGuardrail.run()` | 执行工具输入检查 |
| | `ToolOutputGuardrail.run()` | 执行工具输出检查 |
| **行为控制** | `ToolGuardrailFunctionOutput.allow()` | 允许继续执行 |
| | `ToolGuardrailFunctionOutput.reject_content()` | 拒绝但继续 |
| | `ToolGuardrailFunctionOutput.raise_exception()` | 抛出异常终止 |

## 2. Agent级防护 API

### 2.1 @input_guardrail - 输入防护装饰器

**API 签名：**
```python
@input_guardrail
def my_guardrail(
    context: RunContextWrapper[TContext],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    ...

# 或带参数
@input_guardrail(name="custom_name")
async def my_async_guardrail(...) -> GuardrailFunctionOutput:
    ...
```

**功能描述：**
将函数转换为输入防护栏，在Agent执行时并行运行，用于检查输入是否符合要求。

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `context` | `RunContextWrapper[TContext]` | 运行上下文，包含自定义数据 |
| `agent` | `Agent` | 当前执行的Agent |
| `input` | `str \| list[TResponseInputItem]` | 用户输入内容 |

**返回结构：**
```python
@dataclass
class GuardrailFunctionOutput:
    output_info: Any  # 检查结果信息
    tripwire_triggered: bool  # 是否触发熔断
```

**使用示例：**

```python
from agents import input_guardrail, GuardrailFunctionOutput, Agent

# 1. 基础用法 - 主题检查
@input_guardrail
def check_on_topic(context, agent, input):
    """检查输入是否偏离主题"""
    user_text = input if isinstance(input, str) else input[-1].get("content", "")
    
    off_topic_keywords = ["politics", "religion", "adult"]
    is_off_topic = any(kw in user_text.lower() for kw in off_topic_keywords)
    
    if is_off_topic:
        return GuardrailFunctionOutput(
            output_info={"reason": "Off-topic detected"},
            tripwire_triggered=True  # 终止执行
        )
    
    return GuardrailFunctionOutput(
        output_info={"status": "ok"},
        tripwire_triggered=False
    )

# 2. 异步防护 - API调用检查
@input_guardrail(name="content_moderation")
async def moderate_content(context, agent, input):
    """使用外部API进行内容审核"""
    import httpx
    
    text = input if isinstance(input, str) else str(input)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://moderation-api.com/check",
            json={"text": text}
        )
        result = response.json()
    
    if result["flagged"]:
        return GuardrailFunctionOutput(
            output_info=result,
            tripwire_triggered=True
        )
    
    return GuardrailFunctionOutput(
        output_info={"safe": True},
        tripwire_triggered=False
    )

# 3. 上下文相关检查
@input_guardrail
def check_user_quota(context, agent, input):
    """检查用户配额"""
    user_id = context.get("user_id")
    quota = context.get("quota_service").get_remaining_quota(user_id)
    
    if quota <= 0:
        return GuardrailFunctionOutput(
            output_info={"quota": 0, "user_id": user_id},
            tripwire_triggered=True
        )
    
    return GuardrailFunctionOutput(
        output_info={"quota": quota},
        tripwire_triggered=False
    )

# 在Agent中使用
agent = Agent(
    name="SafeAgent",
    instructions="你是一个安全的助手",
    input_guardrails=[
        check_on_topic,
        moderate_content,
        check_user_quota
    ]
)
```

### 2.2 @output_guardrail - 输出防护装饰器

**API 签名：**
```python
@output_guardrail
def my_guardrail(
    context: RunContextWrapper[TContext],
    agent: Agent,
    agent_output: Any
) -> GuardrailFunctionOutput:
    ...
```

**功能描述：**
在Agent执行完成后检查输出是否符合要求，用于验证响应质量和合规性。

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `context` | `RunContextWrapper[TContext]` | 运行上下文 |
| `agent` | `Agent` | 执行的Agent |
| `agent_output` | `Any` | Agent的输出结果 |

**使用示例：**

```python
from agents import output_guardrail, GuardrailFunctionOutput

# 1. 输出长度检查
@output_guardrail
def check_output_length(context, agent, agent_output):
    """确保输出不超过限制"""
    max_length = context.get("max_output_length", 1000)
    output_text = str(agent_output)
    
    if len(output_text) > max_length:
        return GuardrailFunctionOutput(
            output_info={"length": len(output_text), "max": max_length},
            tripwire_triggered=True
        )
    
    return GuardrailFunctionOutput(
        output_info={"length": len(output_text)},
        tripwire_triggered=False
    )

# 2. 敏感信息检测
@output_guardrail
def check_pii_leakage(context, agent, agent_output):
    """检测是否泄露个人信息"""
    import re
    
    output_text = str(agent_output)
    
    # 检测邮箱、手机号等
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
    
    if re.search(email_pattern, output_text) or re.search(phone_pattern, output_text):
        return GuardrailFunctionOutput(
            output_info={"pii_detected": True},
            tripwire_triggered=True
        )
    
    return GuardrailFunctionOutput(
        output_info={"pii_detected": False},
        tripwire_triggered=False
    )

# 3. 结构化输出验证
@output_guardrail
def validate_json_schema(context, agent, agent_output):
    """验证JSON输出格式"""
    import json
    from jsonschema import validate, ValidationError
    
    expected_schema = context.get("output_schema")
    if not expected_schema:
        return GuardrailFunctionOutput(output_info={}, tripwire_triggered=False)
    
    try:
        if isinstance(agent_output, str):
            output_data = json.loads(agent_output)
        else:
            output_data = agent_output
        
        validate(instance=output_data, schema=expected_schema)
        
        return GuardrailFunctionOutput(
            output_info={"valid": True},
            tripwire_triggered=False
        )
    except (json.JSONDecodeError, ValidationError) as e:
        return GuardrailFunctionOutput(
            output_info={"error": str(e)},
            tripwire_triggered=True
        )

# 在Agent中使用
agent = Agent(
    name="ValidatedAgent",
    instructions="输出JSON格式的结果",
    output_guardrails=[
        check_output_length,
        check_pii_leakage,
        validate_json_schema
    ]
)
```

## 3. Tool级防护 API

### 3.1 @tool_input_guardrail - 工具输入防护

**API 签名：**
```python
@tool_input_guardrail
def my_tool_guardrail(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    ...
```

**功能描述：**
在工具函数执行前检查参数是否合法，支持更细粒度的安全控制。

**参数说明：**

```python
@dataclass
class ToolInputGuardrailData:
    context: ToolContext  # 工具上下文（包含args, call_id等）
    agent: Agent  # 当前Agent
```

**返回结构：**
```python
@dataclass
class ToolGuardrailFunctionOutput:
    output_info: Any  # 检查信息
    behavior: AllowBehavior | RejectContentBehavior | RaiseExceptionBehavior
```

**使用示例：**

```python
from agents import tool_input_guardrail, ToolGuardrailFunctionOutput, function_tool

# 1. 参数范围检查
@tool_input_guardrail
def validate_file_path(data):
    """验证文件路径安全性"""
    args = data.context.args
    file_path = args.get("path", "")
    
    # 不允许访问系统目录
    forbidden_paths = ["/etc", "/sys", "/proc", "C:\\Windows"]
    if any(file_path.startswith(fp) for fp in forbidden_paths):
        return ToolGuardrailFunctionOutput.reject_content(
            message="Access to system directories is not allowed",
            output_info={"blocked_path": file_path}
        )
    
    # 不允许路径遍历
    if ".." in file_path:
        return ToolGuardrailFunctionOutput.raise_exception(
            output_info={"reason": "Path traversal detected"}
        )
    
    return ToolGuardrailFunctionOutput.allow(
        output_info={"path": file_path, "safe": True}
    )

# 2. 权限检查
@tool_input_guardrail
async def check_user_permission(data):
    """检查用户是否有权限执行此工具"""
    tool_name = data.context.tool_name
    user_id = data.agent.context.get("user_id")
    
    # 异步查询权限数据库
    has_permission = await permission_service.check(user_id, tool_name)
    
    if not has_permission:
        return ToolGuardrailFunctionOutput.reject_content(
            message=f"User {user_id} does not have permission to use {tool_name}",
            output_info={"user_id": user_id, "tool": tool_name}
        )
    
    return ToolGuardrailFunctionOutput.allow()

# 3. 参数值验证
@tool_input_guardrail
def validate_amount(data):
    """验证金额参数"""
    args = data.context.args
    amount = args.get("amount", 0)
    
    if amount <= 0:
        return ToolGuardrailFunctionOutput.reject_content(
            message="Amount must be positive",
            output_info={"amount": amount}
        )
    
    if amount > 10000:
        return ToolGuardrailFunctionOutput.reject_content(
            message="Amount exceeds maximum limit of 10000",
            output_info={"amount": amount, "max": 10000}
        )
    
    return ToolGuardrailFunctionOutput.allow()

# 在工具中使用
@function_tool(
    input_guardrails=[validate_file_path]
)
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, 'r') as f:
        return f.read()

@function_tool(
    input_guardrails=[check_user_permission, validate_amount]
)
async def transfer_money(amount: float, to_account: str) -> dict:
    """转账操作"""
    # 实际转账逻辑
    return {"status": "success", "amount": amount}
```

### 3.2 @tool_output_guardrail - 工具输出防护

**API 签名：**
```python
@tool_output_guardrail
def my_tool_output_guardrail(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    ...
```

**参数说明：**

```python
@dataclass
class ToolOutputGuardrailData:
    context: ToolContext  # 工具上下文
    agent: Agent  # 当前Agent
    output: Any  # 工具的输出结果
```

**使用示例：**

```python
from agents import tool_output_guardrail, ToolGuardrailFunctionOutput

# 1. 输出大小限制
@tool_output_guardrail
def limit_output_size(data):
    """限制工具输出大小"""
    output = data.output
    output_str = str(output)
    
    max_size = 10000  # 10KB
    if len(output_str) > max_size:
        return ToolGuardrailFunctionOutput.reject_content(
            message=f"Tool output too large ({len(output_str)} bytes), truncated",
            output_info={"size": len(output_str), "max": max_size}
        )
    
    return ToolGuardrailFunctionOutput.allow()

# 2. 敏感数据过滤
@tool_output_guardrail
def filter_sensitive_data(data):
    """过滤输出中的敏感信息"""
    import re
    
    output = str(data.output)
    
    # 替换信用卡号
    output = re.sub(r'\d{4}-\d{4}-\d{4}-\d{4}', '[CARD_NUMBER]', output)
    # 替换API密钥
    output = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[API_KEY]', output)
    
    return ToolGuardrailFunctionOutput.allow(
        output_info={"filtered": True, "output": output}
    )

# 3. 结果验证
@tool_output_guardrail
def validate_api_response(data):
    """验证API响应格式"""
    output = data.output
    
    if not isinstance(output, dict):
        return ToolGuardrailFunctionOutput.reject_content(
            message="Invalid API response format",
            output_info={"type": type(output).__name__}
        )
    
    required_fields = ["status", "data"]
    missing_fields = [f for f in required_fields if f not in output]
    
    if missing_fields:
        return ToolGuardrailFunctionOutput.reject_content(
            message=f"Missing required fields: {missing_fields}",
            output_info={"missing": missing_fields}
        )
    
    return ToolGuardrailFunctionOutput.allow()

# 在工具中使用
@function_tool(
    output_guardrails=[limit_output_size, filter_sensitive_data]
)
async def fetch_user_data(user_id: str) -> dict:
    """获取用户数据"""
    # API调用
    return {"user_id": user_id, "credit_card": "1234-5678-9012-3456"}
```

## 4. 行为控制 API

### 4.1 ToolGuardrailFunctionOutput 工厂方法

**allow() - 允许继续**
```python
@classmethod
def allow(cls, output_info: Any = None) -> ToolGuardrailFunctionOutput:
    """允许工具正常执行"""
```

**reject_content() - 拒绝但继续**
```python
@classmethod
def reject_content(
    cls, 
    message: str, 
    output_info: Any = None
) -> ToolGuardrailFunctionOutput:
    """拒绝工具调用/输出，但继续执行，向模型返回message"""
```

**raise_exception() - 抛出异常**
```python
@classmethod
def raise_exception(cls, output_info: Any = None) -> ToolGuardrailFunctionOutput:
    """抛出异常，终止整个执行流程"""
```

**使用示例：**

```python
@tool_input_guardrail
def security_check(data):
    """三种行为模式示例"""
    args = data.context.args
    action = args.get("action")
    
    # 1. 允许：安全操作
    if action in ["read", "list"]:
        return ToolGuardrailFunctionOutput.allow(
            output_info={"action": action, "safe": True}
        )
    
    # 2. 拒绝但继续：可疑操作，让模型重新思考
    if action in ["write", "update"]:
        return ToolGuardrailFunctionOutput.reject_content(
            message=f"Action '{action}' requires additional confirmation",
            output_info={"action": action, "requires_confirmation": True}
        )
    
    # 3. 抛出异常：危险操作，立即终止
    if action in ["delete", "drop"]:
        return ToolGuardrailFunctionOutput.raise_exception(
            output_info={"action": action, "reason": "Dangerous operation blocked"}
        )
    
    return ToolGuardrailFunctionOutput.allow()
```

## 5. 异常处理

### 5.1 防护异常类型

```python
# Agent级防护异常
class InputGuardrailTripwireTriggered(Exception):
    """输入防护熔断触发"""
    pass

class OutputGuardrailTripwireTriggered(Exception):
    """输出防护熔断触发"""
    pass

# Tool级防护异常
class ToolGuardrailTripwireTriggered(Exception):
    """工具防护熔断触发"""
    pass
```

**处理示例：**

```python
from agents import run, InputGuardrailTripwireTriggered

try:
    result = await run(agent, "危险输入内容")
except InputGuardrailTripwireTriggered as e:
    print(f"输入防护触发: {e}")
    # 记录日志、通知管理员等
```

## 6. 最佳实践

### 6.1 分层防护策略

```python
# Layer 1: Agent级输入防护（粗粒度）
@input_guardrail
def basic_content_filter(context, agent, input):
    """基础内容过滤"""
    # 检查明显的违规内容
    pass

# Layer 2: Agent级输出防护（结果验证）
@output_guardrail
def validate_output_quality(context, agent, output):
    """验证输出质量"""
    # 检查输出是否符合标准
    pass

# Layer 3: Tool级输入防护（细粒度）
@tool_input_guardrail
def validate_tool_params(data):
    """验证工具参数"""
    # 针对特定工具的参数检查
    pass

# Layer 4: Tool级输出防护（数据清洗）
@tool_output_guardrail
def sanitize_tool_output(data):
    """清洗工具输出"""
    # 过滤敏感信息
    pass
```

### 6.2 性能优化

```python
# 使用缓存避免重复检查
from functools import lru_cache

@input_guardrail
def cached_moderation(context, agent, input):
    """使用缓存的内容审核"""
    
    @lru_cache(maxsize=1000)
    def check_content(text_hash):
        # 实际的审核逻辑
        return moderate_api_call(text_hash)
    
    import hashlib
    text = str(input)
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    result = check_content(text_hash)
    return GuardrailFunctionOutput(
        output_info=result,
        tripwire_triggered=result.get("flagged", False)
    )
```

### 6.3 监控和日志

```python
import logging

@tool_input_guardrail
def monitored_guardrail(data):
    """带监控的防护栏"""
    start_time = time.time()
    
    try:
        # 执行检查逻辑
        result = perform_check(data)
        
        # 记录成功
        logging.info(f"Guardrail passed for tool {data.context.tool_name}")
        
        return result
    except Exception as e:
        # 记录失败
        logging.error(f"Guardrail failed: {e}")
        raise
    finally:
        # 记录执行时间
        duration = time.time() - start_time
        logging.debug(f"Guardrail execution time: {duration:.3f}s")
```

Guardrails 模块通过灵活的防护机制和多层次的安全控制，为 OpenAI Agents 提供了强大的安全保障能力。

