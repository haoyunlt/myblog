# OpenAI Agents Python SDK - Handoffs 模块时序图详解

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

