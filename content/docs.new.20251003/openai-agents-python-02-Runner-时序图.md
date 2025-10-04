# OpenAI Agents Python SDK - Runner 模块时序图详解

## 1. 时序图总览

Runner 模块的时序图展示了代理执行的完整生命周期，从初始化、执行循环到结果返回的各个阶段，以及与其他模块的交互流程。

### 核心时序场景

| 场景类别 | 时序图 | 关键流程 |
|---------|--------|---------|
| **标准执行** | Runner.run() 完整流程 | 初始化、执行循环、结果封装 |
| **流式执行** | Runner.run_streamed() 流程 | 后台任务、事件队列、实时推送 |
| **工具调用** | 工具执行时序 | 工具参数解析、执行、结果处理 |
| **代理切换** | Handoff 切换流程 | 切换请求、输入过滤、新代理执行 |
| **防护检查** | Guardrail 执行时序 | 输入检查、输出检查、tripwire触发 |
| **会话管理** | Session 集成流程 | 历史加载、执行、历史保存 |

## 2. Runner.run() 标准执行时序图

### 场景：完整的代理执行流程

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant Runner as Runner
    participant RunImpl as RunImpl 执行引擎
    participant Session as Session
    participant Guardrails as Guardrails
    participant Agent as Agent
    participant Model as Model
    participant Tools as Tools
    participant Hooks as Lifecycle Hooks
    
    App->>Runner: await run(agent, input, session, config, context)
    
    Runner->>Runner: 验证参数
    Note over Runner: 检查agent、input有效性
    
    Runner->>Hooks: on_run_start(agent, input, context)
    Note over Hooks: 执行开始钩子
    
    Runner->>Session: await get_items()
    Note over Runner,Session: 加载会话历史
    Session-->>Runner: list[TResponseInputItem]
    
    Runner->>Guardrails: 运行输入防护
    Note over Guardrails: 检查用户输入安全性
    
    alt 输入防护触发
        Guardrails-->>Runner: tripwire_triggered=True
        Runner-->>App: 抛出 InputGuardrailTripwireTriggered
    end
    
    Guardrails-->>Runner: 输入检查通过
    
    Runner->>RunImpl: 启动执行循环
    Note over RunImpl: 初始化执行状态<br/>current_turn = 1
    
    loop 执行循环 (最多 max_turns 次)
        RunImpl->>RunImpl: 准备模型输入
        Note over RunImpl: 合并历史、用户输入<br/>应用指令
        
        RunImpl->>Agent: 获取代理配置
        Agent-->>RunImpl: instructions, tools, model
        
        RunImpl->>Model: 调用模型
        Note over Model: 发送完整上下文<br/>等待响应
        
        Model-->>RunImpl: ModelResponse
        Note over Model: 包含消息、工具调用等
        
        RunImpl->>RunImpl: 创建 MessageOutputItem
        
        alt 模型请求工具调用
            RunImpl->>RunImpl: 创建 ToolCallItem
            
            RunImpl->>Tools: 执行工具
            Note over Tools: tool_function(**args)
            
            Tools-->>RunImpl: 工具结果
            
            RunImpl->>Hooks: on_tool_result(tool_name, result)
            
            RunImpl->>RunImpl: 创建 ToolCallOutputItem
            
            RunImpl->>Model: 再次调用模型（带工具结果）
            Model-->>RunImpl: ModelResponse
        
        else 模型请求代理切换
            RunImpl->>RunImpl: 创建 HandoffCallItem
            
            RunImpl->>Agent: 切换到新代理
            Note over Agent: 应用 handoff_input_filter
            
            RunImpl->>RunImpl: 创建 HandoffOutputItem
            
            RunImpl->>Hooks: on_agent_switch(from_agent, to_agent)
            
            Note over RunImpl: 继续循环，使用新代理
        
        else 模型生成最终输出
            RunImpl->>RunImpl: 检测到最终输出
            Note over RunImpl: 退出循环
        end
        
        RunImpl->>RunImpl: current_turn += 1
        
        alt current_turn > max_turns
            RunImpl-->>Runner: 抛出 MaxTurnsExceeded
        end
    end
    
    RunImpl-->>Runner: 返回执行结果
    Note over RunImpl: new_items, raw_responses,<br/>final_output
    
    Runner->>Guardrails: 运行输出防护
    Note over Guardrails: 检查最终输出安全性
    
    alt 输出防护触发
        Guardrails-->>Runner: tripwire_triggered=True
        Runner-->>App: 抛出 OutputGuardrailTripwireTriggered
    end
    
    Guardrails-->>Runner: 输出检查通过
    
    Runner->>Session: await add_items(new_items)
    Note over Runner,Session: 保存新生成的历史
    Session-->>Runner: 保存成功
    
    Runner->>Runner: 封装 RunResult
    Note over Runner: 包含所有执行数据<br/>final_output, new_items等
    
    Runner->>Hooks: on_run_end(result, context)
    Note over Hooks: 执行结束钩子
    
    Runner-->>App: RunResult
    
    App->>App: 处理结果
    Note over App: 访问 final_output,<br/>new_items, usage等
```

**时序图说明：**

### 执行阶段划分

1. **初始化阶段（步骤 1-8）**：
   - 参数验证
   - 触发开始钩子
   - 加载会话历史
   - 执行输入防护检查

2. **执行循环阶段（步骤 9-40）**：
   - 准备模型输入（历史+新输入）
   - 调用模型生成响应
   - 处理工具调用
   - 处理代理切换
   - 检测最终输出

3. **结果处理阶段（步骤 41-52）**：
   - 执行输出防护检查
   - 保存新历史到会话
   - 封装执行结果
   - 触发结束钩子

### 关键决策点

**工具调用决策：**
- 模型返回包含 `tool_calls` → 执行工具 → 再次调用模型
- 模型返回纯文本 → 可能是最终输出

**代理切换决策：**
- 模型请求 handoff → 切换代理 → 继续执行循环
- 新代理接收过滤后的输入

**循环终止条件：**
- 模型生成最终输出 → 正常退出
- 超过 `max_turns` → 抛出异常
- 防护触发 → 抛出异常

## 3. Runner.run_streamed() 流式执行时序图

### 场景：实时事件推送的流式执行

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant Runner as Runner
    participant RunResultStreaming as 流式结果对象
    participant EventQueue as 事件队列
    participant BgTask as 后台执行任务
    participant RunImpl as RunImpl
    participant Model as Model
    participant Tools as Tools
    
    App->>Runner: await run_streamed(agent, input)
    
    Runner->>EventQueue: 创建 asyncio.Queue(maxsize=1000)
    Note over EventQueue: 用于传递流式事件
    
    Runner->>RunResultStreaming: 创建流式结果对象
    Note over RunResultStreaming: current_agent = agent<br/>current_turn = 0<br/>is_complete = False
    
    Runner->>BgTask: 启动后台执行任务
    Note over BgTask: asyncio.create_task(_run_impl_loop())
    
    Runner-->>App: 返回 RunResultStreaming
    Note over Runner: 立即返回，不等待完成
    
    par 后台执行流程
        BgTask->>RunImpl: 启动执行循环
        
        loop 执行循环
            RunImpl->>Model: 调用模型
            
            Model-->>RunImpl: 流式响应事件
            Note over Model: response.audio.delta,<br/>response.output_item.done等
            
            RunImpl->>RunImpl: 转换为 StreamEvent
            
            alt 原始响应事件
                RunImpl->>EventQueue: put_nowait(RawResponsesStreamEvent)
            end
            
            RunImpl->>RunImpl: 处理响应，创建 RunItem
            
            alt 消息输出
                RunImpl->>EventQueue: put_nowait(RunItemStreamEvent(<br/>  name="message_output_created"<br/>))
            
            else 工具调用
                RunImpl->>EventQueue: put_nowait(RunItemStreamEvent(<br/>  name="tool_called"<br/>))
                
                RunImpl->>Tools: 执行工具
                Tools-->>RunImpl: 工具结果
                
                RunImpl->>EventQueue: put_nowait(RunItemStreamEvent(<br/>  name="tool_output"<br/>))
            
            else 代理切换
                RunImpl->>EventQueue: put_nowait(RunItemStreamEvent(<br/>  name="handoff_requested"<br/>))
                
                RunImpl->>RunImpl: 切换到新代理
                
                RunImpl->>EventQueue: put_nowait(AgentUpdatedStreamEvent)
                
                RunImpl->>EventQueue: put_nowait(RunItemStreamEvent(<br/>  name="handoff_occured"<br/>))
            end
            
            alt 检测到最终输出
                RunImpl->>RunImpl: 退出循环
            end
        end
        
        RunImpl->>EventQueue: put_nowait(QueueCompleteSentinel())
        Note over EventQueue: 发送完成信号
        
        BgTask->>RunResultStreaming: 更新 is_complete = True
        BgTask->>RunResultStreaming: 设置 final_output
        
    and 前台事件消费
        App->>RunResultStreaming: async for event in stream_events()
        
        loop 消费事件直到完成
            RunResultStreaming->>EventQueue: await get()
            Note over EventQueue: 阻塞等待新事件
            
            EventQueue-->>RunResultStreaming: StreamEvent
            
            alt 收到 QueueCompleteSentinel
                RunResultStreaming->>RunResultStreaming: 退出循环
            end
            
            RunResultStreaming-->>App: yield StreamEvent
            
            App->>App: 处理事件
            
            alt message_output_created
                App->>App: 增量更新UI文本
                Note over App: 显示打字效果
            
            else tool_called
                App->>App: 显示工具调用状态
                Note over App: "正在使用工具..."
            
            else tool_output
                App->>App: 显示工具结果
            
            else agent_updated
                App->>App: 更新当前代理信息
            end
            
            RunResultStreaming->>EventQueue: task_done()
        end
        
        RunResultStreaming-->>App: 流式迭代结束
    end
    
    App->>RunResultStreaming: 访问 final_output
    RunResultStreaming-->>App: 最终输出结果
    
    App->>App: 完成处理
```

**时序图说明：**

### 并发执行模型

**后台任务：**
- 独立的异步任务执行代理逻辑
- 生成流式事件放入队列
- 完成后发送完成信号

**前台消费：**
- 从队列异步获取事件
- 立即处理和展示
- 收到完成信号后退出

### 事件流转机制

**事件生成：**
1. RunImpl 处理模型响应
2. 创建相应的 RunItem
3. 转换为 StreamEvent
4. 放入异步队列

**事件消费：**
1. 应用调用 `stream_events()`
2. 从队列异步获取事件
3. 通过 `yield` 返回给应用
4. 应用实时处理事件

### 流式执行优势

1. **低延迟**：事件立即推送，无需等待完成
2. **实时反馈**：用户看到增量生成的内容
3. **可中断**：支持通过 `cancel()` 中途取消
4. **资源高效**：使用异步队列，内存占用小

## 4. 工具调用时序图

### 场景：模型请求工具调用并处理结果

```mermaid
sequenceDiagram
    autonumber
    participant RunImpl as RunImpl
    participant Model as Model
    participant ToolRegistry as 工具注册表
    participant ToolGuardrails as 工具防护
    participant ToolFunction as 工具函数
    participant Context as RunContext
    
    RunImpl->>Model: 调用模型（带工具定义）
    Note over Model: tools=[{<br/>  "name": "get_weather",<br/>  "description": "查询天气",<br/>  "parameters": {...}<br/>}]
    
    Model-->>RunImpl: ModelResponse（包含工具调用）
    Note over Model: {<br/>  "tool_calls": [{<br/>    "id": "call_123",<br/>    "name": "get_weather",<br/>    "arguments": '{"city": "Beijing"}'<br/>  }]<br/>}
    
    RunImpl->>RunImpl: 创建 ToolCallItem
    Note over RunImpl: 记录工具调用请求
    
    loop 对每个工具调用
        RunImpl->>ToolRegistry: 查找工具定义
        ToolRegistry-->>RunImpl: Tool 对象
        
        RunImpl->>RunImpl: 解析工具参数
        Note over RunImpl: json.loads(arguments)
        
        alt 参数解析失败
            RunImpl->>RunImpl: 创建错误输出
            Note over RunImpl: 记录参数解析错误
        end
        
        RunImpl->>ToolGuardrails: 运行工具输入防护
        Note over ToolGuardrails: 检查工具参数安全性
        
        alt 工具输入防护触发
            ToolGuardrails-->>RunImpl: reject_content / raise_exception
            
            alt reject_content
                RunImpl->>RunImpl: 创建拒绝消息
                Note over RunImpl: 不执行工具，返回拒绝理由
            
            else raise_exception
                RunImpl-->>RunImpl: 抛出异常
            end
        end
        
        ToolGuardrails-->>RunImpl: 输入检查通过
        
        RunImpl->>ToolFunction: 执行工具函数
        Note over ToolFunction: 传递参数和上下文
        
        alt 工具函数需要上下文
            RunImpl->>Context: 获取用户上下文
            Context-->>ToolFunction: context 对象
        end
        
        ToolFunction->>ToolFunction: 执行实际逻辑
        Note over ToolFunction: 调用API、查询数据库等
        
        alt 工具执行成功
            ToolFunction-->>RunImpl: 返回结果
            
            RunImpl->>ToolGuardrails: 运行工具输出防护
            Note over ToolGuardrails: 检查工具结果安全性
            
            alt 工具输出防护触发
                ToolGuardrails-->>RunImpl: reject_content / allow with modification
                
                alt reject_content
                    RunImpl->>RunImpl: 替换为拒绝消息
                
                else allow with modification
                    RunImpl->>RunImpl: 修改输出内容
                end
            end
            
            ToolGuardrails-->>RunImpl: 输出检查通过
            
            RunImpl->>RunImpl: 创建 ToolCallOutputItem(success=True)
        
        else 工具执行失败
            ToolFunction-->>RunImpl: 抛出异常
            
            RunImpl->>RunImpl: 捕获异常
            RunImpl->>RunImpl: 创建 ToolCallOutputItem(success=False)
            Note over RunImpl: 包含错误信息
        end
    end
    
    RunImpl->>Model: 再次调用模型（带工具结果）
    Note over Model: 传递工具调用的输出<br/>模型理解结果并生成回复
    
    Model-->>RunImpl: ModelResponse（最终回复）
    Note over Model: "根据天气查询结果，<br/>北京今天晴天，22度。"
```

**时序图说明：**

### 工具调用流程

1. **工具定义传递**：模型调用时包含可用工具定义
2. **工具调用请求**：模型返回需要调用的工具和参数
3. **参数验证**：解析和验证工具参数
4. **输入防护**：检查工具参数的安全性
5. **工具执行**：调用实际的工具函数
6. **输出防护**：检查工具结果的安全性
7. **结果传递**：将工具结果返回给模型
8. **生成回复**：模型基于工具结果生成用户可见的回复

### 错误处理机制

**参数解析错误：**
- JSON 解析失败 → 创建错误输出，传递给模型
- 模型可以请求重新调用或放弃

**防护拒绝：**
- `reject_content`：不执行工具，返回拒绝理由
- `raise_exception`：中断整个执行
- `allow with modification`：修改后允许

**执行异常：**
- 工具函数抛出异常 → 捕获并记录
- 创建失败的输出项
- 传递错误信息给模型

## 5. 代理切换时序图

### 场景：主代理切换到专业代理

```mermaid
sequenceDiagram
    autonumber
    participant RunImpl as RunImpl
    participant MainAgent as 主代理
    participant Model as Model
    participant HandoffRegistry as Handoff注册表
    participant InputFilter as 输入过滤器
    participant SpecialistAgent as 专业代理
    participant Hooks as Lifecycle Hooks
    
    RunImpl->>Model: 调用模型（主代理）
    Note over Model: 主代理可访问 handoffs
    
    Model-->>RunImpl: ModelResponse（请求切换）
    Note over Model: {<br/>  "handoff": {<br/>    "target": "ResearchAgent",<br/>    "reason": "需要专业研究"<br/>  }<br/>}
    
    RunImpl->>RunImpl: 创建 HandoffCallItem
    Note over RunImpl: 记录切换请求
    
    RunImpl->>HandoffRegistry: 查找目标代理
    Note over HandoffRegistry: handoffs = [<br/>  ResearchAgent,<br/>  AnalysisAgent<br/>]
    
    HandoffRegistry-->>RunImpl: Handoff 对象
    
    RunImpl->>RunImpl: 获取当前历史
    Note over RunImpl: 包含用户输入和<br/>主代理的所有交互
    
    RunImpl->>InputFilter: 应用输入过滤器
    Note over InputFilter: 过滤或转换历史<br/>handoff_input_filter()
    
    alt 有 Handoff 特定过滤器
        InputFilter->>InputFilter: 使用 Handoff.input_filter
    else 有全局过滤器
        InputFilter->>InputFilter: 使用 RunConfig.handoff_input_filter
    else 无过滤器
        InputFilter->>InputFilter: 保留原始历史
    end
    
    InputFilter-->>RunImpl: 过滤后的输入
    Note over InputFilter: 可能移除了某些消息<br/>或添加了额外上下文
    
    RunImpl->>Hooks: on_agent_switch(MainAgent, SpecialistAgent)
    Note over Hooks: 通知代理切换
    
    RunImpl->>RunImpl: 更新当前代理
    Note over RunImpl: current_agent = SpecialistAgent
    
    RunImpl->>RunImpl: 创建 HandoffOutputItem
    Note over RunImpl: 记录切换完成
    
    RunImpl->>Model: 调用模型（专业代理）
    Note over Model: 使用过滤后的输入<br/>和专业代理的指令
    
    Model-->>RunImpl: ModelResponse（专业回复）
    Note over Model: 专业代理的响应
    
    RunImpl->>RunImpl: 处理响应
    
    alt 专业代理生成最终输出
        RunImpl->>RunImpl: 标记为最终输出
        Note over RunImpl: 退出执行循环
    
    else 专业代理请求工具调用
        RunImpl->>RunImpl: 执行工具调用
        Note over RunImpl: 使用专业代理的工具
    
    else 专业代理请求再次切换
        RunImpl->>RunImpl: 继续切换流程
        Note over RunImpl: 可以切换回主代理<br/>或其他专业代理
    end
```

**时序图说明：**

### 代理切换机制

**切换触发：**
- 模型返回包含 `handoff` 字段
- 指定目标代理名称和切换原因

**输入过滤：**
1. 收集当前所有历史
2. 应用输入过滤器（如果有）
3. 传递过滤后的历史给新代理

**过滤器优先级：**
1. `Handoff.input_filter`（最高）
2. `RunConfig.handoff_input_filter`
3. 无过滤（保留原始历史）

### 输入过滤示例

```python
def handoff_input_filter(handoff_data: HandoffInputData) -> list[TResponseInputItem]:
    """只保留用户消息，移除中间交互"""
    filtered = []
    for item in handoff_data.input:
        if item.get("role") == "user":
            filtered.append(item)
    return filtered

# 效果：
# 原始历史: [user_msg, assistant_msg, tool_call, tool_output, user_msg]
# 过滤后: [user_msg, user_msg]
```

### 切换场景

**专业化分工：**
- 主代理 → 研究代理：需要深度研究
- 主代理 → 分析代理：需要数据分析
- 主代理 → 客服代理：处理客户问题

**任务路由：**
- 根据任务类型自动选择合适的代理
- 每个代理专注特定领域

**权限隔离：**
- 高权限代理 → 低权限代理：执行敏感操作前
- 低权限代理 → 高权限代理：需要提升权限时

## 6. 防护检查时序图

### 场景：输入和输出防护检查

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant Runner as Runner
    participant InputGuardrails as 输入防护列表
    participant IG1 as 内容审核防护
    participant IG2 as PII检测防护
    participant RunImpl as RunImpl
    participant OutputGuardrails as 输出防护列表
    participant OG1 as 敏感信息防护
    participant OG2 as 事实检查防护
    
    App->>Runner: await run(agent, input)
    
    Runner->>InputGuardrails: 运行所有输入防护
    Note over InputGuardrails: config.input_guardrails
    
    par 并行执行输入防护
        InputGuardrails->>IG1: run(input, context)
        IG1->>IG1: 检查不当内容
        
        alt 发现不当内容
            IG1-->>InputGuardrails: InputGuardrailResult(<br/>  tripwire_triggered=True,<br/>  message="包含不当内容"<br/>)
        else 内容正常
            IG1-->>InputGuardrails: InputGuardrailResult(<br/>  tripwire_triggered=False<br/>)
        end
    and
        InputGuardrails->>IG2: run(input, context)
        IG2->>IG2: 检测个人信息
        
        alt 发现PII
            IG2-->>InputGuardrails: InputGuardrailResult(<br/>  tripwire_triggered=True,<br/>  message="包含个人身份信息"<br/>)
        else 无PII
            IG2-->>InputGuardrails: InputGuardrailResult(<br/>  tripwire_triggered=False<br/>)
        end
    end
    
    InputGuardrails->>InputGuardrails: 收集所有结果
    
    alt 任一防护触发tripwire
        InputGuardrails-->>Runner: 防护触发
        Runner->>Runner: 创建 RunErrorDetails
        Runner-->>App: 抛出 InputGuardrailTripwireTriggered
        Note over App: 执行中断，返回错误详情
    end
    
    InputGuardrails-->>Runner: 所有防护通过
    
    Runner->>RunImpl: 继续执行
    Note over RunImpl: 正常的执行循环
    
    RunImpl-->>Runner: 返回执行结果
    Note over Runner: final_output = "这是助手的回复..."
    
    Runner->>OutputGuardrails: 运行所有输出防护
    Note over OutputGuardrails: config.output_guardrails
    
    par 并行执行输出防护
        OutputGuardrails->>OG1: run(final_output, context)
        OG1->>OG1: 检查敏感信息
        
        alt 发现敏感信息
            OG1-->>OutputGuardrails: OutputGuardrailResult(<br/>  tripwire_triggered=True,<br/>  message="输出包含敏感信息"<br/>)
        else 无敏感信息
            OG1-->>OutputGuardrails: OutputGuardrailResult(<br/>  tripwire_triggered=False<br/>)
        end
    and
        OutputGuardrails->>OG2: run(final_output, context)
        OG2->>OG2: 验证事实准确性
        
        alt 事实错误
            OG2-->>OutputGuardrails: OutputGuardrailResult(<br/>  tripwire_triggered=True,<br/>  message="输出包含事实错误"<br/>)
        else 事实正确
            OG2-->>OutputGuardrails: OutputGuardrailResult(<br/>  tripwire_triggered=False<br/>)
        end
    end
    
    OutputGuardrails->>OutputGuardrails: 收集所有结果
    
    alt 任一防护触发tripwire
        OutputGuardrails-->>Runner: 防护触发
        Runner->>Runner: 创建 RunErrorDetails
        Runner-->>App: 抛出 OutputGuardrailTripwireTriggered
        Note over App: 执行中断，返回错误详情
    end
    
    OutputGuardrails-->>Runner: 所有防护通过
    
    Runner->>Runner: 封装 RunResult
    Runner-->>App: 返回 RunResult
    
    App->>App: 处理正常结果
```

**时序图说明：**

### 防护检查机制

**输入防护（执行前）：**
- 在代理执行前检查用户输入
- 可以阻止不安全或不当的输入
- tripwire触发时中断执行

**输出防护（执行后）：**
- 在返回结果前检查最终输出
- 可以阻止不安全或不当的输出
- tripwire触发时中断返回

### 并行执行

**性能优化：**
- 多个防护并行执行
- 使用 `asyncio.gather()` 并发运行
- 减少总体检查时间

**结果聚合：**
- 收集所有防护的结果
- 任一触发tripwire则中断
- 所有结果存储在 `RunResult` 中

### 防护结果处理

**tripwire触发：**
```python
InputGuardrailResult(
    output=GuardrailFunctionOutput(
        tripwire_triggered=True,
        message="输入违反了内容政策"
    )
)
# 抛出 InputGuardrailTripwireTriggered 异常
```

**防护通过：**
```python
InputGuardrailResult(
    output=GuardrailFunctionOutput(
        tripwire_triggered=False
    )
)
# 继续执行
```

**非阻塞警告：**
```python
InputGuardrailResult(
    output=GuardrailFunctionOutput(
        tripwire_triggered=False,
        message="检测到潜在问题，但允许继续"
    )
)
# 记录警告但不中断执行
```

## 7. 完整执行流程总览

```mermaid
flowchart TB
    START([开始执行])
    
    subgraph "初始化阶段"
        INIT[参数验证]
        HOOKS_START[触发 on_run_start]
        LOAD_SESSION[加载会话历史]
        INPUT_GUARD[运行输入防护]
    end
    
    subgraph "执行循环"
        PREP_INPUT[准备模型输入]
        CALL_MODEL[调用模型]
        PROCESS_RESP[处理响应]
        
        CHECK_TYPE{响应类型?}
        
        TOOL_CALL[工具调用]
        TOOL_EXEC[执行工具]
        TOOL_GUARD[工具防护]
        
        HANDOFF[代理切换]
        FILTER_INPUT[输入过滤]
        SWITCH_AGENT[切换代理]
        
        FINAL_OUT[最终输出]
        
        CHECK_TURNS{轮次检查}
    end
    
    subgraph "结果处理"
        OUTPUT_GUARD[运行输出防护]
        SAVE_SESSION[保存会话历史]
        BUILD_RESULT[封装结果]
        HOOKS_END[触发 on_run_end]
    end
    
    END([返回结果])
    
    ERROR_INPUT[输入防护触发]
    ERROR_OUTPUT[输出防护触发]
    ERROR_TURNS[超过最大轮次]
    
    START --> INIT
    INIT --> HOOKS_START
    HOOKS_START --> LOAD_SESSION
    LOAD_SESSION --> INPUT_GUARD
    
    INPUT_GUARD -->|通过| PREP_INPUT
    INPUT_GUARD -->|触发| ERROR_INPUT
    
    PREP_INPUT --> CALL_MODEL
    CALL_MODEL --> PROCESS_RESP
    PROCESS_RESP --> CHECK_TYPE
    
    CHECK_TYPE -->|工具调用| TOOL_CALL
    TOOL_CALL --> TOOL_GUARD
    TOOL_GUARD --> TOOL_EXEC
    TOOL_EXEC --> CALL_MODEL
    
    CHECK_TYPE -->|代理切换| HANDOFF
    HANDOFF --> FILTER_INPUT
    FILTER_INPUT --> SWITCH_AGENT
    SWITCH_AGENT --> PREP_INPUT
    
    CHECK_TYPE -->|最终输出| FINAL_OUT
    FINAL_OUT --> CHECK_TURNS
    
    CHECK_TURNS -->|未超限| OUTPUT_GUARD
    CHECK_TURNS -->|超限| ERROR_TURNS
    
    OUTPUT_GUARD -->|通过| SAVE_SESSION
    OUTPUT_GUARD -->|触发| ERROR_OUTPUT
    
    SAVE_SESSION --> BUILD_RESULT
    BUILD_RESULT --> HOOKS_END
    HOOKS_END --> END
    
    ERROR_INPUT -.-> END
    ERROR_OUTPUT -.-> END
    ERROR_TURNS -.-> END
    
    style START fill:#e8f5e9
    style END fill:#e8f5e9
    style ERROR_INPUT fill:#ffebee
    style ERROR_OUTPUT fill:#ffebee
    style ERROR_TURNS fill:#ffebee
    style CALL_MODEL fill:#e1f5fe
    style TOOL_EXEC fill:#fff3e0
    style SWITCH_AGENT fill:#f3e5f5
```

Runner 模块通过精心设计的时序流程和清晰的执行阶段，为 OpenAI Agents 提供了强大的执行调度能力，支持从简单对话到复杂多代理协作的各种应用场景。

