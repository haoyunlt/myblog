# OpenAI Agents Python SDK - Handoffs 模块 API 详解

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

