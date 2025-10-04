# LangChain-05-Agents-时序图

## 文档说明

本文档通过详细的时序图展示 **Agents 模块**在各种场景下的执行流程，包括Agent创建、推理-行动循环、工具调用、错误处理、早停机制等复杂交互过程。

---

## 1. Agent 创建场景

### 1.1 create_openai_tools_agent 创建流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Creator as create_openai_tools_agent
    participant Model as ChatOpenAI
    participant Tools as ToolList
    participant Prompt as ChatPromptTemplate
    participant Agent as RunnableAgent
    
    User->>Creator: create_openai_tools_agent(llm, tools, prompt)
    
    Creator->>Model: 验证模型支持工具调用
    alt 模型不支持工具调用
        Model-->>Creator: 抛出 ValueError
        Creator-->>User: Error: Model doesn't support tool calling
    end
    
    Creator->>Tools: 转换工具格式为OpenAI格式
    Tools->>Tools: convert_to_openai_tool(tool) for each tool
    Tools-->>Creator: openai_formatted_tools
    
    Creator->>Model: bind_tools(openai_formatted_tools)
    Model-->>Creator: model_with_tools
    
    Creator->>Prompt: 验证提示模板
    alt 缺少必需占位符
        Prompt-->>Creator: ValueError("Missing agent_scratchpad placeholder")
    end
    
    Creator->>Agent: 创建 RunnableAgent
    Note over Agent: agent = prompt | model_with_tools | output_parser
    
    Agent-->>Creator: runnable_agent
    Creator-->>User: Agent 实例
```

**关键验证步骤**：

1. **模型验证**（步骤 3-5）：
   - 检查模型是否支持 `bind_tools` 方法
   - 验证模型类型（必须是 `BaseChatModel`）
   - 确认工具调用能力

2. **工具格式转换**（步骤 6-8）：
   - 将 LangChain 工具转换为 OpenAI 格式
   - 生成工具的 JSON Schema
   - 处理工具描述和参数验证

3. **提示模板验证**（步骤 11-13）：
   - 检查 `{agent_scratchpad}` 占位符
   - 验证输入变量完整性
   - 确保模板格式正确

---

### 1.2 create_react_agent 创建流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Creator as create_react_agent
    participant LLM as BaseLanguageModel
    participant Tools
    participant Prompt as PromptTemplate
    participant Parser as ReActOutputParser
    participant Agent as ReActAgent
    
    User->>Creator: create_react_agent(llm, tools, prompt)
    
    Creator->>Tools: 构建工具描述
    Tools->>Tools: render_text_description(tools)
    Tools-->>Creator: tools_description = "search: 搜索工具\ncalculator: 计算工具"
    
    Creator->>Prompt: 格式化提示模板
    Note over Prompt: 插入工具列表和工具名称
    Prompt-->>Creator: formatted_prompt
    
    Creator->>Parser: 创建 ReAct 输出解析器
    Parser->>Parser: 设置解析规则<br/>Action: tool_name<br/>Action Input: tool_input<br/>Final Answer: final_answer
    
    Creator->>Agent: 构建 Agent 链
    Note over Agent: chain = prompt | llm | output_parser
    
    Agent-->>Creator: react_agent
    Creator-->>User: ReAct Agent 实例
```

**ReAct 格式说明**：

```text
Question: 用户问题
Thought: 我需要思考如何解决这个问题
Action: search
Action Input: "Python tutorial"
Observation: 找到了相关教程...
Thought: 现在我有了足够信息
Final Answer: 基于搜索结果，这里是Python教程推荐...
```

---

## 2. AgentExecutor 执行场景

### 2.1 完整推理-行动循环

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Executor as AgentExecutor
    participant Agent
    participant Tool1 as SearchTool
    participant Tool2 as CalculatorTool
    participant CB as Callbacks
    
    User->>Executor: invoke({"input": "北京明天天气如何？如果下雨概率>50%推荐室内活动"})
    
    Executor->>CB: on_chain_start()
    Executor->>Executor: 初始化状态<br/>iterations=0, intermediate_steps=[]
    
    rect rgb(240, 248, 255)
        Note over Executor: === 第一轮推理 ===
        Executor->>Agent: plan(intermediate_steps=[], input="...")
        Agent->>Agent: 分析任务：需要查询天气
        Agent-->>Executor: AgentAction(tool="search", tool_input={"query": "北京明天天气"})
        
        Executor->>Tool1: invoke({"query": "北京明天天气"})
        Tool1-->>Executor: "明天北京：小雨，降水概率70%，气温15-22°C"
        
        Executor->>Executor: 添加步骤到 intermediate_steps<br/>iterations=1
    end
    
    rect rgb(248, 255, 248)
        Note over Executor: === 第二轮推理 ===
        Executor->>Agent: plan(intermediate_steps=[...], input="...")
        Agent->>Agent: 分析：降水概率70%>50%，需推荐室内活动
        Agent-->>Executor: AgentAction(tool="search", tool_input={"query": "北京室内活动推荐"})
        
        Executor->>Tool1: invoke({"query": "北京室内活动推荐"})
        Tool1-->>Executor: "推荐：博物馆、购物中心、电影院、健身房..."
        
        Executor->>Executor: 添加步骤到 intermediate_steps<br/>iterations=2
    end
    
    rect rgb(255, 248, 248)
        Note over Executor: === 第三轮推理 ===
        Executor->>Agent: plan(intermediate_steps=[...], input="...")
        Agent->>Agent: 信息充足，可以给出最终答案
        Agent-->>Executor: AgentFinish(return_values={"output": "明天北京小雨..."})
    end
    
    Executor->>CB: on_chain_end()
    Executor-->>User: {"input": "...", "output": "明天北京小雨，降水概率70%...", "intermediate_steps": [...]}
```

**执行步骤详解**：

1. **状态初始化**（步骤 3）：
   - 重置迭代计数器
   - 清空中间步骤列表
   - 记录开始时间

2. **推理循环**（步骤 4-18）：
   - 每轮调用 `agent.plan()` 方法
   - 根据当前状态决定下一步动作
   - 执行动作并收集观察结果

3. **循环终止条件**：
   - Agent 返回 `AgentFinish`
   - 达到最大迭代次数
   - 超过最大执行时间

---

### 2.2 工具调用错误处理

```mermaid
sequenceDiagram
    autonumber
    participant Executor as AgentExecutor
    participant Agent
    participant Tool as WeatherTool
    participant ErrorHandler
    
    Executor->>Agent: plan(intermediate_steps, input)
    Agent-->>Executor: AgentAction(tool="weather", tool_input={"city": "InvalidCity"})
    
    Executor->>Tool: invoke({"city": "InvalidCity"})
    Tool-->>Executor: raise ToolException("城市不存在")
    
    Executor->>ErrorHandler: handle_tool_error(exception)
    
    alt handle_tool_error = True
        ErrorHandler-->>Executor: "ToolException: 城市不存在"
        Executor->>Executor: 将错误作为观察结果添加
        Note over Executor: observation = "Error: 城市不存在"
    else handle_tool_error = False
        ErrorHandler-->>Executor: re-raise ToolException
        Executor-->>Executor: 终止执行，返回错误
    else handle_tool_error = custom_function
        ErrorHandler->>ErrorHandler: custom_handler(exception)
        ErrorHandler-->>Executor: "请提供有效的城市名称"
        Executor->>Executor: 使用自定义错误消息
    end
    
    Executor->>Agent: plan(intermediate_steps + [(action, error_observation)])
    Agent->>Agent: 分析错误，调整策略
    Agent-->>Executor: AgentAction(tool="search", tool_input={"query": "有效城市列表"})
```

**错误处理策略**：

| 策略 | 行为 | 适用场景 |
|-----|------|---------|
| `False` | 抛出异常，终止执行 | 严格模式，不容忍错误 |
| `True` | 返回错误字符串 | 让Agent学习错误信息 |
| 自定义函数 | 智能错误处理 | 复杂错误恢复逻辑 |

---

### 2.3 早停机制触发

```mermaid
sequenceDiagram
    autonumber
    participant Executor as AgentExecutor
    participant Agent
    participant StopHandler as EarlyStoppingHandler
    participant Timer
    
    loop 推理循环 (最多15次)
        Executor->>Timer: 检查执行时间
        Timer-->>Executor: time_elapsed = 45秒
        
        alt 达到时间限制 (max_execution_time=60s)
            Note over Executor: 45s < 60s，继续执行
        else 达到迭代限制 (max_iterations=15)
            Executor->>Executor: iterations = 15，触发早停
            break
        end
        
        Executor->>Agent: plan(intermediate_steps, input)
        Agent-->>Executor: AgentAction(...)
        Note over Executor: 继续执行...
    end
    
    Executor->>StopHandler: 处理早停 (early_stopping_method)
    
    alt method = "force"
        StopHandler-->>Executor: AgentFinish(<br/>  return_values={"output": "达到最大迭代次数"},<br/>  log="强制停止"<br/>)
    else method = "generate"  
        StopHandler->>Agent: plan(..., force_final_answer=True)
        Agent->>Agent: 基于现有信息生成最终答案
        Agent-->>StopHandler: AgentFinish(...)
        StopHandler-->>Executor: 生成的最终答案
    end
    
    Executor-->>Executor: 返回结果 (可能不完整)
```

**早停方法对比**：

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| `"force"` | 确定性，快速 | 可能丢失信息 | 严格时间控制 |  
| `"generate"` | 尽力给出答案 | 可能不准确 | 用户体验优先 |

---

## 3. 不同Agent类型的执行流程

### 3.1 OpenAI Tools Agent 执行

```mermaid
sequenceDiagram
    autonumber
    participant Executor
    participant Agent as OpenAI Tools Agent
    participant Model as ChatOpenAI
    participant Tool as SearchTool
    
    Executor->>Agent: plan(intermediate_steps, input="天气查询")
    
    Agent->>Agent: 构建消息序列<br/>system + human + tool_messages
    
    Agent->>Model: invoke(messages, tools=[search_tool])
    Model->>Model: 模型推理：需要调用搜索工具
    Model-->>Agent: AIMessage(<br/>  content="",<br/>  tool_calls=[{<br/>    "id": "call_123",<br/>    "function": {"name": "search", "arguments": "..."},<br/>    "type": "function"<br/>  }]<br/>)
    
    Agent->>Agent: 解析工具调用
    Agent-->>Executor: AgentAction(<br/>  tool="search",<br/>  tool_input={"query": "天气"},<br/>  tool_call_id="call_123"<br/>)
    
    Executor->>Tool: invoke({"query": "天气"})
    Tool-->>Executor: "今天晴天，25°C"
    
    Executor->>Agent: plan(intermediate_steps + [(action, observation)])
    Agent->>Agent: 构建新消息序列（包含工具结果）
    Agent->>Model: invoke([..., ToolMessage(content="今天晴天", tool_call_id="call_123")])
    Model-->>Agent: AIMessage(content="根据查询结果，今天天气晴朗...")
    
    Agent-->>Executor: AgentFinish(return_values={"output": "今天天气晴朗..."})
```

**工具调用格式**：

```json
{
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function", 
    "function": {
      "name": "search",
      "arguments": "{\"query\": \"weather Beijing\"}"
    }
  }]
}
```

---

### 3.2 ReAct Agent 执行

```mermaid
sequenceDiagram
    autonumber
    participant Executor
    participant Agent as ReAct Agent
    participant LLM as OpenAI LLM
    participant Parser as ReActOutputParser
    participant Tool
    
    Executor->>Agent: plan(intermediate_steps, input="计算2+2*3")
    
    Agent->>Agent: 构建ReAct提示<br/>包含工具描述和中间步骤
    
    Agent->>LLM: predict(prompt)
    LLM-->>Agent: "Thought: 我需要计算2+2*3\nAction: calculator\nAction Input: 2+2*3"
    
    Agent->>Parser: parse(llm_output)
    Parser->>Parser: 解析文本格式<br/>提取Action和Action Input
    
    alt 解析成功
        Parser-->>Agent: AgentAction(tool="calculator", tool_input="2+2*3")
    else 解析失败
        Parser-->>Agent: OutputParserException("Invalid format")
        
        alt handle_parsing_errors=True
            Agent-->>Executor: 返回解析错误信息
        else handle_parsing_errors=False
            Agent-->>Executor: raise OutputParserException
        end
    end
    
    Executor->>Tool: invoke({"expression": "2+2*3"})
    Tool-->>Executor: "8"
    
    Executor->>Agent: plan(intermediate_steps + [(action, "8")])
    Agent->>LLM: predict(prompt_with_observation)
    LLM-->>Agent: "Thought: 我现在知道答案了\nFinal Answer: 2+2*3等于8"
    
    Agent->>Parser: parse(llm_output)
    Parser-->>Agent: AgentFinish(return_values={"output": "2+2*3等于8"})
    
    Agent-->>Executor: AgentFinish(...)
```

**ReAct解析规则**：

```python
# 解析器查找的模式
patterns = {
    "action": r"Action: (.+)",
    "action_input": r"Action Input: (.+)",
    "final_answer": r"Final Answer: (.+)",
    "thought": r"Thought: (.+)"
}
```

---

## 4. 流式执行场景

### 4.1 Agent 流式输出

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Executor
    participant Agent
    participant Tool
    
    User->>Executor: stream({"input": "查询天气并推荐活动"})
    
    loop 推理循环
        Executor->>Agent: plan(...)
        
        alt Agent返回动作
            Agent-->>Executor: AgentAction(tool="search", ...)
            Executor-->>User: yield {"actions": [AgentAction(...)]}
            
            Executor->>Tool: invoke(...)
            Tool-->>Executor: observation
            Executor-->>User: yield {"steps": [AgentStep(action, observation)]}
            
        else Agent返回完成
            Agent-->>Executor: AgentFinish(...)
            Executor-->>User: yield {"output": "最终答案..."}
            break
        end
    end
```

**流式输出示例**：

```python
for chunk in agent_executor.stream({"input": "天气查询"}):
    if "actions" in chunk:
        print(f"🤖 准备执行: {chunk['actions'][0].tool}")
    elif "steps" in chunk:
        print(f"📋 工具返回: {chunk['steps'][0].observation}")
    elif "output" in chunk:
        print(f"✅ 最终答案: {chunk['output']}")
```

---

## 5. 错误恢复场景

### 5.1 解析错误恢复

```mermaid
sequenceDiagram
    autonumber
    participant Executor
    participant Agent
    participant LLM
    participant Parser
    participant ErrorRecovery
    
    Executor->>Agent: plan(intermediate_steps, input)
    Agent->>LLM: 生成响应
    LLM-->>Agent: "I should search for weather\nTool: search\nQuery: Beijing weather"
    
    Agent->>Parser: parse(invalid_format_output)
    Parser-->>Agent: OutputParserException("Expected 'Action:' but got 'Tool:'")
    
    Agent->>ErrorRecovery: handle_parsing_error(exception)
    
    alt 自动修复模式
        ErrorRecovery->>ErrorRecovery: 尝试格式修复<br/>Tool: search → Action: search
        ErrorRecovery->>Parser: parse(corrected_output)
        
        alt 修复成功
            Parser-->>ErrorRecovery: AgentAction(...)
            ErrorRecovery-->>Agent: 修复后的动作
        else 修复失败
            ErrorRecovery-->>Agent: 返回错误信息
        end
        
    else 重新生成模式
        ErrorRecovery->>LLM: 重新生成<br/>附加格式要求
        LLM-->>ErrorRecovery: "Action: search\nAction Input: Beijing weather"
        ErrorRecovery->>Parser: parse(new_output)
        Parser-->>ErrorRecovery: AgentAction(...)
        ErrorRecovery-->>Agent: 新生成的动作
    end
    
    Agent-->>Executor: 最终动作或错误信息
```

**错误恢复策略**：

1. **格式修复**：识别常见格式错误并自动修复
2. **重新生成**：提供更详细的格式说明重新请求
3. **降级处理**：使用简化的解析规则
4. **人工干预**：记录错误等待人工处理

---

### 5.2 工具超时恢复

```mermaid
sequenceDiagram
    autonumber
    participant Executor
    participant Tool as SlowTool
    participant Timeout as TimeoutHandler
    participant Fallback as FallbackTool
    
    Executor->>Tool: invoke({"query": "complex_search"}, timeout=30s)
    
    Tool->>Tool: 执行复杂查询...
    Note over Tool: 30秒后仍在执行
    
    Tool-->>Executor: TimeoutError("Tool execution timeout")
    
    Executor->>Timeout: handle_timeout(tool_name, input)
    
    alt 有备用工具
        Timeout->>Fallback: invoke(simplified_input)
        Fallback-->>Timeout: "简化的搜索结果"
        Timeout-->>Executor: observation = "由于超时，返回简化结果：..."
        
    else 无备用工具
        Timeout-->>Executor: observation = "工具执行超时，请稍后重试"
    end
    
    Executor->>Executor: 添加超时步骤到中间步骤
    Note over Executor: Agent可以基于超时信息调整策略
```

---

## 6. 性能优化场景

### 6.1 中间步骤修剪

```mermaid
sequenceDiagram
    autonumber
    participant Executor
    participant Agent
    participant Trimmer as StepTrimmer
    
    Note over Executor: iterations = 12, intermediate_steps.length = 12
    
    Executor->>Executor: 检查修剪条件<br/>trim_intermediate_steps = 5
    
    Executor->>Trimmer: trim_steps(intermediate_steps, max_count=5)
    
    Trimmer->>Trimmer: 分析步骤重要性<br/>- 最近的步骤（权重高）<br/>- 包含关键工具的步骤<br/>- 有用信息的步骤
    
    Trimmer->>Trimmer: 选择保留步骤<br/>保留最后2步 + 3个重要步骤
    
    Trimmer-->>Executor: trimmed_steps (长度=5)
    
    Executor->>Agent: plan(trimmed_steps, input)
    Note over Agent: Agent基于修剪后的历史进行推理
    
    Agent-->>Executor: 下一步动作
```

**修剪策略**：

```python
def intelligent_trim(steps, max_count):
    if len(steps) <= max_count:
        return steps
    
    # 按重要性评分
    scored_steps = []
    for i, (action, obs) in enumerate(steps):
        score = 0
        
        # 最近的步骤权重更高
        score += (i / len(steps)) * 10
        
        # 关键工具权重更高
        if action.tool in ["search", "calculator"]:
            score += 5
            
        # 有用观察权重更高
        if "error" not in obs.lower() and len(obs) > 10:
            score += 3
            
        scored_steps.append((score, (action, obs)))
    
    # 选择得分最高的步骤
    scored_steps.sort(key=lambda x: x[0], reverse=True)
    return [step for score, step in scored_steps[:max_count]]
```

---

### 6.2 并行工具调用优化

```mermaid
sequenceDiagram
    autonumber
    participant Agent as MultiActionAgent
    participant Executor
    participant Tool1 as WeatherTool
    participant Tool2 as NewsToolq 
    participant Tool3 as TrafficTool
    
    Executor->>Agent: plan(intermediate_steps, input="北京今天天气、新闻、交通状况")
    
    Agent->>Agent: 分析：可以并行获取三类信息
    Agent-->>Executor: [<br/>  AgentAction(tool="weather", ...),<br/>  AgentAction(tool="news", ...),<br/>  AgentAction(tool="traffic", ...)<br/>]
    
    par 并行执行工具
        Executor->>Tool1: invoke({"city": "北京"})
        Tool1-->>Executor: "北京：晴天，25°C"
    and
        Executor->>Tool2: invoke({"location": "北京", "category": "local"})
        Tool2-->>Executor: "今日北京新闻摘要..."
    and  
        Executor->>Tool3: invoke({"city": "北京"})
        Tool3-->>Executor: "当前交通状况良好"
    end
    
    Executor->>Executor: 收集所有结果<br/>构建综合观察
    Note over Executor: observation = "天气：晴天25°C\n新闻：...\n交通：良好"
    
    Executor->>Agent: plan(intermediate_steps + [combined_step])
    Agent-->>Executor: AgentFinish("根据获取的信息，今天北京...")
```

**并行执行优势**：
- 减少总执行时间
- 提高信息获取效率
- 更好的用户体验

---

## 7. 总结

本文档详细展示了 **Agents 模块**的关键执行时序：

1. **Agent创建**：不同类型Agent的创建和验证流程
2. **推理循环**：完整的思考-行动-观察循环
3. **工具调用**：同步和异步工具执行机制
4. **错误处理**：解析错误、工具错误的恢复策略
5. **早停机制**：达到限制时的处理方法
6. **流式执行**：实时输出中间步骤和结果
7. **性能优化**：步骤修剪和并行执行

每张时序图包含：
- 详细的参与者交互过程
- 关键决策点和分支逻辑
- 错误处理和恢复机制
- 性能优化点和最佳实践

这些时序图帮助开发者深入理解Agent系统的复杂执行机制，为构建高效、可靠的智能代理应用提供指导。Agent系统是LangChain中最复杂但也最强大的组件，正确理解其执行流程对成功构建AI应用至关重要。
