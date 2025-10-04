# OpenAI Agents Python SDK - Guardrails 模块时序图详解

## 1. 时序图总览

Guardrails 模块的时序图展示了防护栏在Agent和Tool执行过程中的介入时机和处理流程。核心流程包括：输入防护、输出防护、工具防护和异常处理。

### 主要时序流程

| 时序流程 | 参与者 | 触发时机 | 核心操作 |
|---------|--------|---------|---------|
| **输入防护执行** | Runner, InputGuardrails | Agent执行前 | 并行检查输入 |
| **输出防护执行** | Runner, OutputGuardrails | Agent执行后 | 验证输出质量 |
| **工具输入防护** | Runner, ToolInputGuardrails | 工具调用前 | 验证工具参数 |
| **工具输出防护** | Runner, ToolOutputGuardrails | 工具执行后 | 过滤工具输出 |

## 2. Agent级输入防护时序图

### 2.1 输入防护并行执行流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant R as Runner
    participant A as Agent
    participant IG1 as InputGuardrail 1
    participant IG2 as InputGuardrail 2
    participant IG3 as InputGuardrail 3
    participant T as TracingProcessor
    
    U->>R: run(agent, "用户输入")
    activate R
    
    Note over R: 获取Agent的input_guardrails
    R->>R: guardrails = agent.input_guardrails
    
    alt 有输入防护
        Note over R: 并行执行所有输入防护
        
        par 并行执行防护1
            R->>T: on_span_start(guardrail_span_1)
            R->>IG1: run(agent, input, context)
            activate IG1
            IG1->>IG1: 执行检查逻辑
            IG1-->>R: InputGuardrailResult
            deactivate IG1
            R->>T: on_span_end(guardrail_span_1, result)
        and 并行执行防护2
            R->>T: on_span_start(guardrail_span_2)
            R->>IG2: run(agent, input, context)
            activate IG2
            IG2->>IG2: 执行检查逻辑
            IG2-->>R: InputGuardrailResult
            deactivate IG2
            R->>T: on_span_end(guardrail_span_2, result)
        and 并行执行防护3
            R->>T: on_span_start(guardrail_span_3)
            R->>IG3: run(agent, input, context)
            activate IG3
            IG3->>IG3: 执行检查逻辑
            IG3-->>R: InputGuardrailResult
            deactivate IG3
            R->>T: on_span_end(guardrail_span_3, result)
        end
        
        Note over R: 收集所有结果
        R->>R: results = [result1, result2, result3]
        
        Note over R: 检查是否有熔断触发
        loop 遍历results
            R->>R: 检查result.output.tripwire_triggered
            
            alt tripwire触发
                R->>R: 记录触发的guardrail
                R-->>U: raise InputGuardrailTripwireTriggered
                Note over U: 执行终止
            end
        end
    end
    
    Note over R: 所有防护通过，继续执行
    R->>A: 执行Agent逻辑
    A-->>R: Agent输出
    
    R-->>U: RunResult
    deactivate R
```

**流程说明：**

1. **并行启动**：所有输入防护同时开始执行
2. **独立检查**：每个防护独立进行检查逻辑
3. **结果收集**：等待所有防护完成
4. **熔断检查**：任一防护触发熔断则终止
5. **继续执行**：全部通过则继续Agent执行

### 2.2 输入防护详细执行流程

```mermaid
sequenceDiagram
    participant R as Runner
    participant IG as InputGuardrail
    participant Func as GuardrailFunction
    participant Ctx as RunContext
    
    R->>IG: run(agent, input, context)
    activate IG
    
    Note over IG: 1. 验证函数可调用
    IG->>IG: 检查guardrail_function是否callable
    
    alt 不可调用
        IG-->>R: raise UserError
    end
    
    Note over IG: 2. 调用防护函数
    IG->>Func: guardrail_function(context, agent, input)
    activate Func
    
    Note over Func: 执行自定义检查逻辑
    Func->>Ctx: 访问context数据
    Ctx-->>Func: 上下文信息
    
    Func->>Func: 分析输入内容
    Func->>Func: 应用检查规则
    
    Note over Func: 生成检查结果
    Func->>Func: 创建GuardrailFunctionOutput
    
    alt 同步函数
        Func-->>IG: GuardrailFunctionOutput
    else 异步函数
        Func-->>IG: Awaitable[GuardrailFunctionOutput]
        IG->>IG: await output
    end
    deactivate Func
    
    Note over IG: 3. 包装结果
    IG->>IG: 创建InputGuardrailResult
    IG->>IG: result.guardrail = self
    IG->>IG: result.output = output
    
    IG-->>R: InputGuardrailResult
    deactivate IG
```

**执行细节：**

1. **函数验证**：确保防护函数可调用
2. **函数调用**：支持同步和异步函数
3. **上下文访问**：防护函数可访问运行上下文
4. **结果包装**：标准化返回结果

## 3. Agent级输出防护时序图

### 3.1 输出防护执行流程

```mermaid
sequenceDiagram
    participant R as Runner
    participant A as Agent
    participant M as Model
    participant OG as OutputGuardrail
    participant T as TracingProcessor
    
    R->>A: 执行Agent
    A->>M: 调用模型
    M-->>A: 模型响应
    A-->>R: agent_output
    
    Note over R: 检查output_guardrails
    R->>R: guardrails = agent.output_guardrails
    
    alt 有输出防护
        Note over R: 顺序执行输出防护
        
        loop 遍历每个guardrail
            R->>T: on_span_start(guardrail_span)
            
            R->>OG: run(context, agent, agent_output)
            activate OG
            
            Note over OG: 执行输出检查
            OG->>OG: guardrail_function(context, agent, output)
            OG->>OG: 分析输出内容
            OG->>OG: 验证格式和质量
            
            OG-->>R: OutputGuardrailResult
            deactivate OG
            
            R->>T: on_span_end(guardrail_span, result)
            
            Note over R: 检查tripwire
            alt tripwire触发
                R->>R: 记录问题
                R-->>R: raise OutputGuardrailTripwireTriggered
                Note over R: 终止并返回错误
            end
        end
    end
    
    Note over R: 所有输出防护通过
    R->>R: 准备最终结果
    R-->>R: return RunResult(output=agent_output)
```

**流程特点：**

1. **执行时机**：Agent生成输出后
2. **顺序执行**：按防护栏顺序依次检查
3. **快速失败**：任一防护触发即终止
4. **结果返回**：全部通过则返回输出

## 4. Tool级输入防护时序图

### 4.1 工具输入防护完整流程

```mermaid
sequenceDiagram
    participant R as Runner
    participant M as Model
    participant TIG as ToolInputGuardrail
    participant TF as ToolFunction
    participant T as TracingProcessor
    
    M-->>R: function_call(name="search", args={...})
    
    Note over R: 准备执行工具
    R->>R: tool = 查找工具
    R->>R: guardrails = tool.input_guardrails
    
    alt 有输入防护
        Note over R: 创建ToolInputGuardrailData
        R->>R: data = ToolInputGuardrailData(context, agent)
        R->>R: data.context.tool_name = "search"
        R->>R: data.context.args = {...}
        R->>R: data.context.call_id = "call_abc123"
        
        loop 遍历输入防护
            R->>T: on_span_start(tool_guardrail_span)
            
            R->>TIG: run(data)
            activate TIG
            
            Note over TIG: 执行防护检查
            TIG->>TIG: guardrail_function(data)
            TIG->>TIG: 验证参数合法性
            TIG->>TIG: 检查权限
            
            TIG-->>R: ToolGuardrailFunctionOutput
            deactivate TIG
            
            R->>T: on_span_end(tool_guardrail_span, output)
            
            Note over R: 检查behavior
            R->>R: behavior = output.behavior
            
            alt behavior.type == "allow"
                Note over R: 继续下一个防护
                
            else behavior.type == "reject_content"
                Note over R: 拒绝工具调用
                R->>R: 创建工具错误结果
                R->>R: result = {"error": behavior.message}
                R->>M: 返回错误消息
                Note over R: 跳过工具执行
                
            else behavior.type == "raise_exception"
                Note over R: 抛出异常终止
                R-->>R: raise ToolGuardrailTripwireTriggered
            end
        end
    end
    
    Note over R: 所有防护通过，执行工具
    R->>TF: 调用工具函数(args)
    activate TF
    TF->>TF: 执行实际逻辑
    TF-->>R: 工具输出
    deactivate TF
    
    R->>M: 返回工具结果
```

**关键决策点：**

1. **allow行为**：继续检查和执行
2. **reject_content行为**：跳过执行，返回消息
3. **raise_exception行为**：终止整个流程

### 4.2 工具输入防护行为处理

```mermaid
sequenceDiagram
    participant R as Runner
    participant TIG as ToolInputGuardrail
    participant M as Model
    
    R->>TIG: run(data)
    TIG-->>R: ToolGuardrailFunctionOutput
    
    R->>R: 提取behavior
    
    alt behavior: allow
        Note over R: 场景1: 允许执行
        R->>R: 继续处理
        Note over R: 执行工具函数
        
    else behavior: reject_content
        Note over R: 场景2: 拒绝但继续
        R->>R: 创建拒绝消息
        R->>R: tool_result = {<br/>"type": "error",<br/>"message": behavior.message<br/>}
        
        Note over R: 不执行工具，直接返回消息
        R->>M: 返回tool_result
        M->>M: 模型看到错误消息
        M->>M: 可以重新思考或调整参数
        
    else behavior: raise_exception
        Note over R: 场景3: 抛出异常
        R->>R: 记录安全事件
        R-->>R: raise ToolGuardrailTripwireTriggered(...)
        Note over R: 整个run终止
    end
```

## 5. Tool级输出防护时序图

### 5.1 工具输出防护流程

```mermaid
sequenceDiagram
    participant R as Runner
    participant TF as ToolFunction
    participant TOG as ToolOutputGuardrail
    participant M as Model
    
    Note over R: 工具执行完成
    R->>TF: 调用工具
    TF-->>R: tool_output
    
    Note over R: 检查输出防护
    R->>R: guardrails = tool.output_guardrails
    
    alt 有输出防护
        Note over R: 创建ToolOutputGuardrailData
        R->>R: data = ToolOutputGuardrailData(context, agent, output)
        R->>R: data.output = tool_output
        
        loop 遍历输出防护
            R->>TOG: run(data)
            activate TOG
            
            Note over TOG: 检查输出内容
            TOG->>TOG: guardrail_function(data)
            TOG->>TOG: 验证输出格式
            TOG->>TOG: 过滤敏感信息
            TOG->>TOG: 检查输出大小
            
            TOG-->>R: ToolGuardrailFunctionOutput
            deactivate TOG
            
            Note over R: 处理behavior
            alt behavior: allow
                R->>R: 使用原始输出
                
            else behavior: reject_content
                R->>R: 替换为behavior.message
                R->>R: final_output = behavior.message
                
            else behavior: raise_exception
                R-->>R: raise ToolGuardrailTripwireTriggered
            end
        end
    end
    
    Note over R: 返回最终输出给模型
    R->>M: 工具结果
```

**输出处理逻辑：**

1. **allow**：使用工具的原始输出
2. **reject_content**：用消息替换输出
3. **raise_exception**：终止执行

## 6. 防护栏异常处理时序图

### 6.1 熔断触发和异常传播

```mermaid
sequenceDiagram
    participant U as 用户代码
    participant R as Runner
    participant IG as InputGuardrail
    participant T as TracingProcessor
    participant H as ExceptionHandler
    
    U->>R: run(agent, input)
    activate R
    
    R->>IG: run(...)
    activate IG
    IG->>IG: 检测到违规内容
    IG-->>R: GuardrailFunctionOutput(tripwire=True)
    deactivate IG
    
    Note over R: 检测到熔断触发
    R->>R: 记录防护信息
    
    R->>T: on_span_end(guardrail_span, error)
    T->>T: 记录防护失败
    
    R->>H: 创建异常
    H->>H: exception = InputGuardrailTripwireTriggered(<br/>guardrail_name=...,<br/>output_info=...<br/>)
    
    R->>T: on_trace_end(trace, error=exception)
    T->>T: 记录trace失败
    
    R-->>U: raise InputGuardrailTripwireTriggered
    deactivate R
    
    Note over U: 捕获和处理异常
    U->>U: try-except捕获
    U->>U: 记录日志
    U->>U: 用户友好的错误提示
```

**异常处理流程：**

1. **检测触发**：发现tripwire=True
2. **记录追踪**：在trace中记录失败
3. **创建异常**：包含详细信息
4. **传播异常**：向上层抛出
5. **用户处理**：在应用层捕获处理

## 7. 完整的防护执行时序图（端到端）

### 7.1 包含所有防护层的完整流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant R as Runner
    participant IG as InputGuardrails
    participant A as Agent
    participant M as Model
    participant TIG as ToolInputGuardrails
    participant T as Tool
    participant TOG as ToolOutputGuardrails
    participant OG as OutputGuardrails
    
    U->>R: run(agent, "用户输入")
    activate R
    
    Note over R: 阶段1: 输入防护
    R->>IG: 并行执行所有输入防护
    IG-->>R: 所有结果
    
    alt 输入防护触发
        R-->>U: raise InputGuardrailTripwireTriggered
    end
    
    Note over R: 阶段2: Agent执行
    R->>A: 执行Agent
    A->>M: 调用模型
    
    alt 模型决定调用工具
        M-->>A: function_call
        A->>R: 需要执行工具
        
        Note over R: 阶段3: 工具输入防护
        R->>TIG: 检查工具参数
        TIG-->>R: ToolGuardrailFunctionOutput
        
        alt behavior: reject_content
            R->>M: 返回错误消息
            Note over M: 模型重新思考
        else behavior: raise_exception
            R-->>U: raise ToolGuardrailTripwireTriggered
        else behavior: allow
            Note over R: 阶段4: 执行工具
            R->>T: 调用工具函数
            T-->>R: 工具输出
            
            Note over R: 阶段5: 工具输出防护
            R->>TOG: 检查工具输出
            TOG-->>R: ToolGuardrailFunctionOutput
            
            alt behavior: reject_content
                R->>R: 替换输出为message
            else behavior: raise_exception
                R-->>U: raise ToolGuardrailTripwireTriggered
            end
            
            R->>M: 返回工具结果
        end
        
        M->>M: 继续推理
    end
    
    M-->>A: 最终响应
    A-->>R: agent_output
    
    Note over R: 阶段6: 输出防护
    R->>OG: 检查Agent输出
    OG-->>R: OutputGuardrailResult
    
    alt 输出防护触发
        R-->>U: raise OutputGuardrailTripwireTriggered
    end
    
    Note over R: 所有防护通过
    R-->>U: RunResult(output=agent_output)
    deactivate R
```

**完整流程总结：**

1. **输入防护**：Agent执行前的第一道防线
2. **Agent执行**：核心逻辑执行
3. **工具输入防护**：每个工具调用前检查
4. **工具执行**：实际工具功能
5. **工具输出防护**：工具结果后处理
6. **输出防护**：最终结果验证

## 8. 防护栏性能优化时序图

### 8.1 并行执行优化

```mermaid
sequenceDiagram
    participant R as Runner
    participant IG1 as 快速检查
    participant IG2 as 中速检查
    participant IG3 as 慢速API检查
    
    Note over R: 使用asyncio.gather并行执行
    
    par 并行任务1
        R->>IG1: run() - 本地正则检查
        IG1->>IG1: 5ms
        IG1-->>R: 结果1
    and 并行任务2
        R->>IG2: run() - 本地模型推理
        IG2->>IG2: 50ms
        IG2-->>R: 结果2
    and 并行任务3
        R->>IG3: run() - 外部API调用
        IG3->>IG3: 200ms
        IG3-->>R: 结果3
    end
    
    Note over R: 总耗时 = max(5, 50, 200) = 200ms<br/>而非 5 + 50 + 200 = 255ms
```

**优化策略：**

1. **并行执行**：所有输入防护同时开始
2. **快速失败**：任一触发立即终止
3. **缓存结果**：重复检查使用缓存
4. **异步I/O**：API调用不阻塞

Guardrails 模块通过精心设计的时序流程和多层防护机制，为 OpenAI Agents 提供了全面的安全保护能力，确保系统的可靠性和合规性。

