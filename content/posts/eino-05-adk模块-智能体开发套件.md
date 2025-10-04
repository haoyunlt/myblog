---
title: "Eino 源码剖析 - ADK 模块：智能体开发套件"
date: 2025-10-04T20:42:31+08:00
draft: false
tags:
  - Eino
  - 源码分析
categories:
  - Eino
  - AI框架
  - Go
series: "eino-source-analysis"
description: "Eino 源码剖析 - Eino 源码剖析 - ADK 模块：智能体开发套件"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# Eino 源码剖析 - ADK 模块：智能体开发套件

## 一、模块概览

### 1.1 模块定位

ADK（Agent Development Kit，智能体开发套件）是 Eino 框架中专门用于构建智能体（Agent）应用的高层抽象模块。它在 Compose 模块的编排能力之上，提供了更符合智能体开发范式的 API 和预制组件。

### 1.2 核心职责

1. **智能体抽象**：提供统一的 Agent 接口，支持多种智能体实现模式
2. **ReAct 模式**：内置 ReAct（Reasoning and Acting）模式的智能体实现
3. **多智能体编排**：支持多个智能体之间的协作和转发
4. **工作流智能体**：提供顺序、并行、循环三种工作流模式
5. **会话管理**：管理智能体运行时的会话状态和上下文
6. **中断和恢复**：支持智能体运行过程的中断和断点恢复
7. **预制智能体**：提供 PlanExecute、Supervisor 等开箱即用的智能体

### 1.3 模块架构

```mermaid
graph TB
    subgraph "ADK 智能体开发套件"
        Agent[Agent 接口]
        Runner[Runner 运行器]
        
        subgraph "智能体实现"
            ChatModelAgent[ChatModelAgent]
            WorkflowAgent[WorkflowAgent]
            FlowAgent[FlowAgent]
        end
        
        subgraph "工作流模式"
            Sequential[顺序执行]
            Parallel[并行执行]
            Loop[循环执行]
        end
        
        subgraph "预制智能体"
            PlanExecute[PlanExecute]
            Supervisor[Supervisor]
        end
        
        subgraph "支持组件"
            Session[会话管理]
            Interrupt[中断机制]
            RunContext[运行上下文]
        end
    end
    
    Agent --> ChatModelAgent
    Agent --> WorkflowAgent
    Agent --> FlowAgent
    
    WorkflowAgent --> Sequential
    WorkflowAgent --> Parallel
    WorkflowAgent --> Loop
    
    Runner --> Agent
    Runner --> Session
    Runner --> Interrupt
    
    PlanExecute --> ChatModelAgent
    Supervisor --> ChatModelAgent
    
    ChatModelAgent --> RunContext
    FlowAgent --> RunContext

    style Agent fill:#e1f5ff
    style Runner fill:#fff4e1
    style ChatModelAgent fill:#e8f5e9
    style WorkflowAgent fill:#f3e5f5
</mermaid>

**架构说明**：

1. **Agent 接口层**：定义了所有智能体必须实现的核心接口（`Run`、`Name`、`Description`）
2. **智能体实现层**：
   - `ChatModelAgent`：基于大模型和工具调用的 ReAct 智能体
   - `WorkflowAgent`：支持顺序/并行/循环模式的工作流智能体
   - `FlowAgent`：智能体包装器，支持多智能体间的转发和协作
3. **运行器层**：`Runner` 负责智能体的启动、会话管理、中断保存等
4. **预制智能体**：基于核心智能体实现的高级智能体模式

---

## 二、核心数据结构

### 2.1 Agent 接口

```go

type Agent interface {
    Name(ctx context.Context) string
    Description(ctx context.Context) string
    Run(ctx context.Context, input *AgentInput, options ...AgentRunOption) *AsyncIterator[*AgentEvent]
}

```

**功能说明**：

- `Name`：返回智能体的唯一名称
- `Description`：返回智能体的功能描述，用于其他智能体判断是否转发任务
- `Run`：运行智能体，返回事件流迭代器

### 2.2 AgentInput

```go

type AgentInput struct {
    Messages        []Message          // 输入消息列表
    EnableStreaming bool               // 是否启用流式输出
}

```

**功能说明**：

智能体的输入参数，包含用户消息和流式控制标志。

### 2.3 AgentEvent

```go

type AgentEvent struct {
    AgentName string              // 事件来源智能体名称
    RunPath   []RunStep           // 执行路径（多智能体场景）
    Output    *AgentOutput        // 智能体输出
    Action    *AgentAction        // 智能体动作
    Err       error               // 错误信息
}

```

**功能说明**：

智能体运行过程中产生的事件，包含输出、动作、错误等信息。

### 2.4 AgentOutput

```go

type AgentOutput struct {
    MessageOutput    *MessageVariant  // 消息输出（文本/流式）
    CustomizedOutput any              // 自定义输出
}

type MessageVariant struct {
    IsStreaming   bool                // 是否流式
    Message       Message             // 完整消息
    MessageStream MessageStream       // 流式消息
    Role          schema.RoleType     // 消息角色
    ToolName      string              // 工具名称（仅 Role=Tool 时）
}

```

**功能说明**：

- `MessageOutput`：标准消息输出，支持完整消息和流式消息
- `CustomizedOutput`：自定义输出，用于特殊场景

### 2.5 AgentAction

```go

type AgentAction struct {
    Exit            bool                      // 退出标志
    Interrupted     *InterruptInfo            // 中断信息
    TransferToAgent *TransferToAgentAction    // 转发到其他智能体
    CustomizedAction any                      // 自定义动作
}

type TransferToAgentAction struct {
    DestAgentName string                     // 目标智能体名称
}

type InterruptInfo struct {
    Data any                                 // 中断数据（用于恢复）
}

```

**功能说明**：

- `Exit`：智能体请求退出
- `Interrupted`：智能体执行被中断，包含恢复所需的状态
- `TransferToAgent`：将控制权转移到其他智能体
- `CustomizedAction`：自定义动作

### 2.6 Runner

```go

type Runner struct {
    a               Agent                    // 待运行的智能体
    enableStreaming bool                     // 是否启用流式
    store           compose.CheckPointStore  // 检查点存储
}

```

**功能说明**：

Runner 是智能体的运行器，负责：

1. 初始化运行上下文
2. 管理会话值（Session Values）
3. 保存和恢复检查点
4. 处理中断事件

---

## 三、核心 API 详解

### 3.1 ChatModelAgent 创建

#### 3.1.1 API 签名

```go

func NewChatModelAgent(ctx context.Context, config *ChatModelAgentConfig) (Agent, error)

```

#### 3.1.2 请求参数

```go

type ChatModelAgentConfig struct {
    Name          string                     // 智能体名称（必填）
    Description   string                     // 智能体描述（必填）
    Instruction   string                     // 系统指令（可选）
    Model         model.ToolCallingChatModel // 大模型（必填）
    ToolsConfig   ToolsConfig                // 工具配置
    GenModelInput GenModelInput              // 输入转换函数（可选）
    Exit          tool.BaseTool              // 退出工具（可选）
    OutputKey     string                     // 输出键（可选）
    MaxIterations int                        // 最大迭代次数（默认 20）
}

type ToolsConfig struct {
    compose.ToolsNodeConfig
    ReturnDirectly map[string]bool           // 工具直接返回标志
}

```

**字段说明**：

- `Name`：智能体的唯一标识，用于多智能体场景中的引用
- `Description`：描述智能体的功能和适用场景，帮助其他智能体判断是否转发任务
- `Instruction`：作为系统提示词，支持 f-string 格式（如 `{Time}`, `{User}`），可从会话中获取值
- `Model`：支持工具调用的大模型
- `ToolsConfig.Tools`：智能体可用的工具列表
- `ToolsConfig.ReturnDirectly`：指定哪些工具调用后直接返回结果
- `GenModelInput`：自定义输入转换函数，默认使用 `defaultGenModelInput`
- `Exit`：退出工具，调用后智能体生成 Exit Action
- `OutputKey`：设置后，智能体输出会保存到会话中（`AddSessionValue(ctx, outputKey, msg.Content)`）
- `MaxIterations`：ReAct 循环的最大次数，防止无限循环

#### 3.1.3 返回值

返回 `Agent` 接口的实现和可能的错误。

#### 3.1.4 核心代码实现

```go

func NewChatModelAgent(ctx context.Context, config *ChatModelAgentConfig) (Agent, error) {
    a := &ChatModelAgent{
        name:          config.Name,
        description:   config.Description,
        model:         config.Model,
        outputKey:     config.OutputKey,
        maxIterations: config.MaxIterations,
    }
    
    // 默认 GenModelInput
    if config.GenModelInput == nil {
        a.genModelInput = defaultGenModelInput
    } else {
        a.genModelInput = config.GenModelInput
    }

    // 构建 ReAct Graph
    if len(config.ToolsConfig.Tools) > 0 {
        conf := &reactConfig{
            model:               a.model,
            toolsConfig:         &config.ToolsConfig.ToolsNodeConfig,
            toolsReturnDirectly: config.ToolsConfig.ReturnDirectly,
            agentName:           a.name,
            maxIterations:       a.maxIterations,
        }
        
        g, err := newReact(ctx, conf)
        if err != nil {
            return nil, err
        }
        
        // 运行 ReAct Graph
        a.run = func(ctx context.Context, input *AgentInput, ...) {
            runnable, _ := g.Compile(ctx, compileOptions...)
            
            msgs, _ := a.genModelInput(ctx, instruction, input)
            
            if input.EnableStreaming {
                msgStream, _ = runnable.Stream(ctx, msgs, opts...)
            } else {
                msg, _ = runnable.Invoke(ctx, msgs, opts...)
            }
            
            // 处理输出和会话保存
            if a.outputKey != "" {
                setOutputToSession(ctx, msg, msgStream, a.outputKey)
            }
        }
    }
    
    return a, nil
}

```

---

### 3.2 Runner 运行

#### 3.2.1 API 签名

```go

func (r *Runner) Run(ctx context.Context, messages []Message, opts ...AgentRunOption) *AsyncIterator[*AgentEvent]
func (r *Runner) Query(ctx context.Context, query string, opts ...AgentRunOption) *AsyncIterator[*AgentEvent]

```

#### 3.2.2 请求参数

- `messages`：输入消息列表（`[]*schema.Message`）
- `query`：文本查询（自动转换为 `UserMessage`）
- `opts`：运行选项，支持：
  - `WithSessionValues(map[string]any)`：设置会话值
  - `WithCheckPointID(string)`：指定检查点 ID
  - `WithChatModelOptions([]model.Option)`：传递给大模型的选项
  - `WithToolOptions([]tool.Option)`：传递给工具的选项

#### 3.2.3 返回值

返回 `*AsyncIterator[*AgentEvent]`，可通过 `Next()` 迭代获取事件。

#### 3.2.4 核心代码实现

```go

func (r *Runner) Run(ctx context.Context, messages []Message, opts ...AgentRunOption) *AsyncIterator[*AgentEvent] {
    o := getCommonOptions(nil, opts...)
    
    fa := toFlowAgent(ctx, r.a)
    
    input := &AgentInput{
        Messages:        messages,
        EnableStreaming: r.enableStreaming,
    }
    
    // 初始化运行上下文
    ctx = ctxWithNewRunCtx(ctx)
    
    // 设置会话值
    AddSessionValues(ctx, o.sessionValues)
    
    // 运行智能体
    iter := fa.Run(ctx, input, opts...)
    
    // 如果配置了 store，处理中断事件
    if r.store != nil {
        niter, gen := NewAsyncIteratorPair[*AgentEvent]()
        go r.handleIter(ctx, iter, gen, o.checkPointID)
        return niter
    }
    
    return iter
}

func (r *Runner) handleIter(ctx context.Context, aIter *AsyncIterator[*AgentEvent],
    gen *AsyncGenerator[*AgentEvent], checkPointID *string) {
    
    var interruptedInfo *InterruptInfo
    for {
        event, ok := aIter.Next()
        if !ok {
            break
        }
        
        // 捕获中断信息
        if event.Action != nil && event.Action.Interrupted != nil {
            interruptedInfo = event.Action.Interrupted
        }
        
        gen.Send(event)
    }
    
    // 保存检查点
    if interruptedInfo != nil && checkPointID != nil {
        saveCheckPoint(ctx, r.store, *checkPointID, getInterruptRunCtx(ctx), interruptedInfo)
    }
}

```

---

### 3.3 智能体中断和恢复

#### 3.3.1 Resume API 签名

```go

func (r *Runner) Resume(ctx context.Context, checkPointID string, opts ...AgentRunOption) (*AsyncIterator[*AgentEvent], error)

```

#### 3.3.2 请求参数

- `checkPointID`：检查点 ID，由之前 Run 时指定
- `opts`：运行选项

#### 3.3.3 返回值

返回 `*AsyncIterator[*AgentEvent]` 和可能的错误。

#### 3.3.4 核心代码实现

```go

func (r *Runner) Resume(ctx context.Context, checkPointID string, opts ...AgentRunOption) (*AsyncIterator[*AgentEvent], error) {
    if r.store == nil {
        return nil, fmt.Errorf("failed to resume: store is nil")
    }
    
    // 从检查点恢复运行上下文和中断信息
    runCtx, info, existed, err := getCheckPoint(ctx, r.store, checkPointID)
    if !existed {
        return nil, fmt.Errorf("checkpoint[%s] is not existed", checkPointID)
    }
    
    // 恢复上下文
    ctx = setRunCtx(ctx, runCtx)
    
    o := getCommonOptions(nil, opts...)
    AddSessionValues(ctx, o.sessionValues)
    
    // 调用智能体的 Resume
    aIter := toFlowAgent(ctx, r.a).Resume(ctx, info, opts...)
    
    // 继续处理中断
    niter, gen := NewAsyncIteratorPair[*AgentEvent]()
    go r.handleIter(ctx, aIter, gen, &checkPointID)
    
    return niter, nil
}

```

---

### 3.4 工作流智能体创建

#### 3.4.1 顺序执行智能体

```go

func NewSequentialAgent(ctx context.Context, config *SequentialAgentConfig) (Agent, error)

type SequentialAgentConfig struct {
    Name        string
    Description string
    SubAgents   []Agent  // 子智能体列表
}

```

**功能说明**：按顺序依次执行子智能体，前一个智能体的输出作为下一个智能体的输入。

#### 3.4.2 并行执行智能体

```go

func NewParallelAgent(ctx context.Context, config *ParallelAgentConfig) (Agent, error)

type ParallelAgentConfig struct {
    Name        string
    Description string
    SubAgents   []Agent
}

```

**功能说明**：并行执行所有子智能体，每个智能体接收相同的输入。

#### 3.4.3 循环执行智能体

```go

func NewLoopAgent(ctx context.Context, config *LoopAgentConfig) (Agent, error)

type LoopAgentConfig struct {
    Name          string
    Description   string
    SubAgents     []Agent
    MaxIterations int  // 最大循环次数（0 表示无限循环）
}

```

**功能说明**：循环执行子智能体序列，直到达到最大迭代次数或某个智能体发出 Exit Action。

---

## 四、ReAct 模式实现

### 4.1 ReAct 模式概述

ReAct（Reasoning and Acting）是一种经典的智能体模式，结合了推理（Reasoning）和行动（Acting）：

1. **推理阶段**：大模型根据用户输入和历史消息，决定下一步行动（调用工具或直接回复）
2. **行动阶段**：如果模型决定调用工具，则执行工具并获取结果
3. **循环迭代**：将工具结果作为新消息加入历史，重新进入推理阶段
4. **终止条件**：模型不再调用工具或达到最大迭代次数

### 4.2 ReAct Graph 构建

```go

func newReact(ctx context.Context, config *reactConfig) (reactGraph, error) {
    // 创建状态生成函数
    genState := func(ctx context.Context) *State {
        return &State{
            ToolGenActions:         map[string]*AgentAction{},
            AgentName:              config.agentName,
            AgentToolInterruptData: make(map[string]*agentToolInterruptInfo),
            RemainingIterations:    config.maxIterations,
        }
    }
    
    // 创建 Graph
    g := compose.NewGraph[[]Message, Message](compose.WithGenLocalState(genState))
    
    // 生成工具信息
    toolsInfo, _ := genToolInfos(ctx, config.toolsConfig)
    
    // 创建支持工具的模型
    chatModel, _ := config.model.WithTools(toolsInfo)
    
    // 创建工具节点
    toolsNode, _ := compose.NewToolNode(ctx, config.toolsConfig)
    
    // 添加 ChatModel 节点（带状态前置处理）
    modelPreHandle := func(ctx context.Context, input []Message, st *State) ([]Message, error) {
        if st.RemainingIterations <= 0 {
            return nil, ErrExceedMaxIterations
        }
        st.RemainingIterations--
        st.Messages = append(st.Messages, input...)
        return st.Messages, nil
    }
    g.AddChatModelNode("ChatModel", chatModel, compose.WithStatePreHandler(modelPreHandle))
    
    // 添加 Tool 节点（带状态前置处理）
    toolPreHandle := func(ctx context.Context, input Message, st *State) (Message, error) {
        if input != nil {
            st.Messages = append(st.Messages, input)
        }
        
        // 检查是否有 ReturnDirectly 工具
        input = st.Messages[len(st.Messages)-1]
        for i := range input.ToolCalls {
            toolName := input.ToolCalls[i].Function.Name
            if config.toolsReturnDirectly[toolName] {
                st.ReturnDirectlyToolCallID = input.ToolCalls[i].ID
            }
        }
        
        return input, nil
    }
    g.AddToolsNode("ToolNode", toolsNode, compose.WithStatePreHandler(toolPreHandle))
    
    // 添加边：START -> ChatModel
    g.AddEdge(compose.START, "ChatModel")
    
    // 添加条件边：ChatModel -> ToolNode / END
    toolCallCheck := func(ctx context.Context, sMsg MessageStream) (string, error) {
        // 检查输出消息是否包含工具调用
        msg, _ := schema.ConcatMessageStream(sMsg)
        if len(msg.ToolCalls) > 0 {
            return "ToolNode", nil
        }
        return compose.END, nil
    }
    g.AddBranch("ChatModel", toolCallCheck)
    
    // 添加边：ToolNode -> ChatModel（循环）
    g.AddEdge("ToolNode", "ChatModel")
    
    return g, nil
}

```

### 4.3 ReAct 状态管理

```go

type State struct {
    Messages                 []Message            // 消息历史
    ReturnDirectlyToolCallID string               // 直接返回的工具调用 ID
    ToolGenActions           map[string]*AgentAction  // 工具生成的动作
    AgentName                string               // 智能体名称
    AgentToolInterruptData   map[string]*agentToolInterruptInfo  // 中断数据
    RemainingIterations      int                  // 剩余迭代次数
}

```

**功能说明**：

- `Messages`：累积所有对话消息（用户输入、模型输出、工具结果）
- `RemainingIterations`：每次进入 ChatModel 节点时递减，为 0 时抛出错误
- `ReturnDirectlyToolCallID`：标记需要直接返回的工具调用
- `ToolGenActions`：工具可以向智能体传递动作（如 Exit、TransferToAgent）

### 4.4 ReAct 执行流程

```mermaid

sequenceDiagram
    participant User
    participant Runner
    participant ChatModelAgent
    participant Graph
    participant ChatModel as ChatModel 节点
    participant ToolNode as ToolNode 节点
    participant LLM as 大语言模型
    participant Tools as 工具集合

    User->>Runner: Run(ctx, messages)
    Runner->>ChatModelAgent: Run(ctx, input)
    ChatModelAgent->>Graph: Compile() → Runnable
    ChatModelAgent->>Graph: Invoke(ctx, messages)
    
    loop ReAct 循环
        Graph->>ChatModel: 执行 ChatModel 节点
        ChatModel->>ChatModel: modelPreHandle（检查迭代次数）
        ChatModel->>LLM: Generate(messages)
        LLM-->>ChatModel: Message（含/不含 ToolCalls）
        
        alt 包含 ToolCalls
            Graph->>ToolNode: 执行 ToolNode 节点
            ToolNode->>ToolNode: toolPreHandle（保存消息）
            ToolNode->>Tools: 调用工具
            Tools-->>ToolNode: 工具结果
            ToolNode-->>Graph: []Message（工具结果消息）
            Graph->>ChatModel: 回到 ChatModel 节点
        else 不包含 ToolCalls
            Graph-->>ChatModelAgent: 输出最终 Message
        end
    end
    
    ChatModelAgent-->>Runner: AgentEvent（Output）
    Runner-->>User: AsyncIterator[*AgentEvent]
</mermaid>

**执行流程说明**：

1. **初始化**：Runner 创建运行上下文，设置会话值
2. **Graph 编译**：ChatModelAgent 将 ReAct Graph 编译为 Runnable
3. **模型推理**：
   - 进入 ChatModel 节点，检查剩余迭代次数
   - 调用大模型生成输出
   - 检查输出是否包含 ToolCalls
4. **工具执行**（如果有 ToolCalls）：
   - 进入 ToolNode 节点
   - 执行工具并收集结果
   - 将工具结果作为新消息添加到历史
   - 返回 ChatModel 节点继续推理
5. **终止**：模型不再调用工具或达到最大迭代次数
6. **输出**：ChatModelAgent 将最终消息包装为 AgentEvent 返回

---

## 五、多智能体协作

### 5.1 FlowAgent 包装器

`FlowAgent` 是对普通 Agent 的包装，提供了多智能体协作的能力：

```go
type flowAgent struct {
    Agent                       // 嵌入原始 Agent
    subAgents     []*flowAgent  // 子智能体列表
    parent        *flowAgent    // 父智能体
    disallowTransferToParent bool  // 禁止转发到父智能体
}
```

**功能说明**：

- 支持智能体之间的转发（TransferToAgent Action）
- 管理父子智能体关系
- 拦截和处理 Exit、TransferToAgent、Interrupted 等动作

### 5.2 智能体转发

```go
func (f *flowAgent) Run(ctx context.Context, input *AgentInput, opts ...AgentRunOption) *AsyncIterator[*AgentEvent] {
    iterator, generator := NewAsyncIteratorPair[*AgentEvent]()
    
    go func() {
        ctx, runCtx := initRunCtx(ctx, f.Name(ctx), input)
        
        iter := f.Agent.Run(ctx, input, opts...)
        
        for {
            event, ok := iter.Next()
            if !ok {
                break
            }
            
            // 处理 TransferToAgent 动作
            if event.Action != nil && event.Action.TransferToAgent != nil {
                destAgentName := event.Action.TransferToAgent.DestAgentName
                
                // 查找目标智能体
                destAgent := f.findAgent(destAgentName)
                if destAgent == nil {
                    generator.Send(&AgentEvent{Err: fmt.Errorf("agent %s not found", destAgentName)})
                    break
                }
                
                // 转发到目标智能体
                destIter := destAgent.Run(ctx, input, opts...)
                for {
                    destEvent, ok := destIter.Next()
                    if !ok {
                        break
                    }
                    generator.Send(destEvent)
                }
                
                break
            }
            
            generator.Send(event)
        }
        
        generator.Close()
    }()
    
    return iterator
}
```

### 5.3 多智能体交互图

```mermaid
graph LR
    subgraph "多智能体系统"
        User[用户]
        Runner[Runner]
        
        subgraph "FlowAgent 包装层"
            FA1[FlowAgent 1]
            FA2[FlowAgent 2]
            FA3[FlowAgent 3]
        end
        
        subgraph "实际智能体"
            A1[ChatModelAgent A]
            A2[ChatModelAgent B]
            A3[WorkflowAgent C]
        end
    end
    
    User -->|Run| Runner
    Runner --> FA1
    FA1 --> A1
    
    FA1 -.->|TransferToAgent| FA2
    FA2 --> A2
    
    FA2 -.->|TransferToAgent| FA3
    FA3 --> A3
    
    FA3 -.->|Exit| FA1
    FA1 -->|AgentEvent| Runner
    Runner -->|AsyncIterator| User

    style FA1 fill:#e1f5ff
    style FA2 fill:#fff4e1
    style FA3 fill:#e8f5e9
</mermaid>

---

## 六、工作流智能体详解

### 6.1 顺序执行模式

```go

func (a *workflowAgent) runSequential(ctx context.Context, input *AgentInput,
    generator *AsyncGenerator[*AgentEvent], intInfo *WorkflowInterruptInfo,
    iterations int, opts ...AgentRunOption) (exit, interrupted bool) {
    
    // 重建运行路径
    var runPath []RunStep
    if intInfo != nil {
        // 从中断点恢复
        i = intInfo.SequentialInterruptIndex
    }
    
    // 依次执行子智能体
    for ; i < len(a.subAgents); i++ {
        subAgent := a.subAgents[i]
        
        var subIterator *AsyncIterator[*AgentEvent]
        if intInfo != nil && i == intInfo.SequentialInterruptIndex {
            // 恢复中断的智能体
            subIterator = subAgent.Resume(ctx, &ResumeInfo{
                EnableStreaming: input.EnableStreaming,
                InterruptInfo:   intInfo.SequentialInterruptInfo,
            }, opts...)
        } else {
            // 正常运行
            subIterator = subAgent.Run(ctx, input, opts...)
        }
        
        // 处理子智能体事件
        var lastActionEvent *AgentEvent
        for {
            event, ok := subIterator.Next()
            if !ok {
                break
            }
            
            // 延迟发送 Action 事件（等待确认是否需要包装）
            if event.Action != nil {
                lastActionEvent = event
                continue
            }
            generator.Send(event)
        }
        
        // 检查最后的 Action 事件
        if lastActionEvent != nil {
            if lastActionEvent.Action.Interrupted != nil {
                // 包装中断信息，包含当前索引
                newEvent := wrapWorkflowInterrupt(lastActionEvent, input, i, iterations)
                generator.Send(newEvent)
                return true, true
            }
            
            if lastActionEvent.Action.Exit {
                generator.Send(lastActionEvent)
                return true, false
            }
        }
    }
    
    return false, false
}

```

### 6.2 并行执行模式

```go

func (a *workflowAgent) runParallel(ctx context.Context, input *AgentInput,
    generator *AsyncGenerator[*AgentEvent], intInfo *WorkflowInterruptInfo, opts ...AgentRunOption) {
    
    runners := getRunners(a.subAgents, input, intInfo, opts...)
    var wg sync.WaitGroup
    interruptMap := make(map[int]*InterruptInfo)
    var mu sync.Mutex
    
    // 并行执行所有子智能体
    for i := 1; i < len(runners); i++ {
        wg.Add(1)
        go func(idx int, runner func(ctx context.Context) *AsyncIterator[*AgentEvent]) {
            defer wg.Done()
            
            iterator := runner(ctx)
            for {
                event, ok := iterator.Next()
                if !ok {
                    break
                }
                if event.Action != nil && event.Action.Interrupted != nil {
                    mu.Lock()
                    interruptMap[idx] = event.Action.Interrupted
                    mu.Unlock()
                    break
                }
                generator.Send(event)
            }
        }(i, runners[i])
    }
    
    // 主线程执行第一个智能体
    runner := runners[0]
    iterator := runner(ctx)
    for {
        event, ok := iterator.Next()
        if !ok {
            break
        }
        if event.Action != nil && event.Action.Interrupted != nil {
            mu.Lock()
            interruptMap[0] = event.Action.Interrupted
            mu.Unlock()
            break
        }
        generator.Send(event)
    }
    
    wg.Wait()
    
    // 如果有中断，包装并发送
    if len(interruptMap) > 0 {
        generator.Send(&AgentEvent{
            AgentName: a.Name(ctx),
            RunPath:   getRunCtx(ctx).RunPath,
            Action: &AgentAction{
                Interrupted: &InterruptInfo{
                    Data: &WorkflowInterruptInfo{
                        OrigInput:             input,
                        ParallelInterruptInfo: interruptMap,
                    },
                },
            },
        })
    }
}

```

### 6.3 循环执行模式

```go

func (a *workflowAgent) runLoop(ctx context.Context, input *AgentInput,
    generator *AsyncGenerator[*AgentEvent], intInfo *WorkflowInterruptInfo, opts ...AgentRunOption) {
    
    var iterations int
    if intInfo != nil {
        iterations = intInfo.LoopIterations
    }
    
    // 循环执行顺序模式
    for iterations < a.maxIterations || a.maxIterations == 0 {
        exit, interrupted := a.runSequential(ctx, input, generator, intInfo, iterations, opts...)
        if interrupted {
            return
        }
        if exit {
            return
        }
        intInfo = nil  // 只在第一次迭代时生效
        iterations++
    }
}

```

---

## 七、会话管理

### 7.1 Session Values

```go

func AddSessionValues(ctx context.Context, values map[string]any)
func AddSessionValue(ctx context.Context, key string, value any)
func GetSessionValues(ctx context.Context) map[string]any
func GetSessionValue(ctx context.Context, key string) (any, bool)

```

**功能说明**：

会话值存储在运行上下文中，用于：

1. 在 Instruction 中通过 f-string 引用（如 `{Time}`, `{User}`）
2. 在智能体之间传递上下文信息
3. 保存智能体输出（通过 `OutputKey` 配置）

### 7.2 RunContext

```go

type runContext struct {
    RootInput      *AgentInput             // 根输入
    SessionValues  map[string]any          // 会话值
    RunPath        []RunStep               // 运行路径
}

```

**功能说明**：

- `RootInput`：保存最初的用户输入，用于恢复时重建上下文
- `SessionValues`：键值对存储，跨智能体共享
- `RunPath`：记录智能体执行路径，便于调试和追踪

---

## 八、预制智能体

### 8.1 PlanExecute 智能体

PlanExecute 是一种"先规划，后执行"的智能体模式：

1. **Planner**：根据用户输入生成执行计划（多个步骤）
2. **Executor**：逐步执行计划中的每个步骤
3. **Replanner**：根据执行结果调整计划

**使用示例**：

```go

import "github.com/cloudwego/eino/adk/prebuilt/planexecute"

pe, _ := planexecute.NewPlanExecute(ctx, &planexecute.Config{
    Planner: plannerAgent,
    Executor: executorAgent,
    MaxIterations: 10,
})

runner := adk.NewRunner(ctx, adk.RunnerConfig{
    Agent: pe,
    EnableStreaming: true,
})

iter := runner.Query(ctx, "帮我预定明天去北京的机票")
for {
    event, ok := iter.Next()
    if !ok {
        break
    }
    // 处理事件
}

```

### 8.2 Supervisor 智能体

Supervisor 是一种"主管-工人"模式的智能体：

1. **Supervisor**：根据用户输入决定调用哪个子智能体
2. **Workers**：执行具体任务的子智能体
3. **Aggregator**：聚合子智能体的输出

**使用示例**：

```go

import "github.com/cloudwego/eino/adk/prebuilt/supervisor"

sv, _ := supervisor.NewSupervisor(ctx, &supervisor.Config{
    Supervisor: supervisorAgent,
    Workers: []adk.Agent{
        codeAgent,
        searchAgent,
        mathAgent,
    },
})

runner := adk.NewRunner(ctx, adk.RunnerConfig{
    Agent: sv,
})

iter := runner.Query(ctx, "帮我搜索今天的天气并生成代码")

```

---

## 九、最佳实践

### 9.1 智能体设计原则

1. **单一职责**：每个智能体应专注于一个明确的任务域
2. **清晰描述**：Description 应详细描述智能体的能力和适用场景
3. **工具选择**：只配置智能体真正需要的工具，避免工具过多导致模型选择困难
4. **迭代限制**：合理设置 MaxIterations，防止无限循环

### 9.2 多智能体协作

1. **转发条件**：明确定义何时转发到其他智能体
2. **退出机制**：提供 Exit Tool 让智能体能够主动退出
3. **层级设计**：合理设计智能体的父子关系，避免循环转发

### 9.3 中断和恢复

1. **检查点策略**：在长时间运行的智能体中启用检查点
2. **状态序列化**：确保自定义状态可以序列化（实现 `GobEncode/GobDecode`）
3. **恢复幂等性**：智能体 Resume 应能正确恢复到中断点，不重复执行已完成的步骤

### 9.4 流式输出

1. **启用流式**：对于长文本生成，启用 `EnableStreaming` 提升用户体验
2. **流式消费**：及时消费 `MessageStream`，避免阻塞
3. **流式关闭**：使用 `SetAutomaticClose()` 确保流式消息自动关闭

### 9.5 性能优化

1. **并行智能体**：对于独立任务，使用 `ParallelAgent` 提升效率
2. **缓存工具结果**：对于幂等工具，可在应用层缓存结果
3. **异步处理**：充分利用 `AsyncIterator` 的异步特性

---

## 十、完整示例

### 10.1 简单 ReAct 智能体

```go

import (
    "context"
    "github.com/cloudwego/eino/adk"
    "github.com/cloudwego/eino/components/model"
    "github.com/cloudwego/eino/components/tool"
    "github.com/cloudwego/eino/schema"
)

func main() {
    ctx := context.Background()
    
    // 创建大模型
    llm := model.NewOpenAIChatModel(...)
    
    // 创建工具
    searchTool := tool.NewSearchTool(...)
    calcTool := tool.NewCalculatorTool(...)
    
    // 创建 ChatModelAgent
    agent, _ := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
        Name:        "assistant",
        Description: "A helpful assistant with search and calculation abilities",
        Instruction: "You are a helpful assistant. Current time: {Time}",
        Model:       llm,
        ToolsConfig: adk.ToolsConfig{
            ToolsNodeConfig: compose.ToolsNodeConfig{
                Tools: []tool.BaseTool{searchTool, calcTool},
            },
        },
        MaxIterations: 10,
    })
    
    // 创建 Runner
    runner := adk.NewRunner(ctx, adk.RunnerConfig{
        Agent:           agent,
        EnableStreaming: true,
    })
    
    // 运行查询
    iter := runner.Query(ctx, "今天北京的天气如何？",
        adk.WithSessionValues(map[string]any{
            "Time": "2025-10-04 10:00:00",
        }))
    
    // 处理事件
    for {
        event, ok := iter.Next()
        if !ok {
            break
        }
        
        if event.Err != nil {
            fmt.Println("Error:", event.Err)
            break
        }
        
        if event.Output != nil && event.Output.MessageOutput != nil {
            msgVariant := event.Output.MessageOutput
            if msgVariant.IsStreaming {
                for {
                    chunk, err := msgVariant.MessageStream.Recv()
                    if err == io.EOF {
                        break
                    }
                    fmt.Print(chunk.Content)
                }
            } else {
                fmt.Println(msgVariant.Message.Content)
            }
        }
    }
}

```

### 10.2 多智能体协作

```go

func main() {
    ctx := context.Background()
    
    // 创建专业智能体
    codeAgent, _ := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
        Name:        "code_agent",
        Description: "Expert in code generation and code review",
        Model:       llm,
        ToolsConfig: adk.ToolsConfig{
            ToolsNodeConfig: compose.ToolsNodeConfig{
                Tools: []tool.BaseTool{codeGenTool, codeReviewTool},
            },
        },
    })
    
    searchAgent, _ := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
        Name:        "search_agent",
        Description: "Expert in web search and information retrieval",
        Model:       llm,
        ToolsConfig: adk.ToolsConfig{
            ToolsNodeConfig: compose.ToolsNodeConfig{
                Tools: []tool.BaseTool{searchTool},
            },
        },
    })
    
    // 创建主管智能体（可以转发任务）
    supervisorAgent, _ := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
        Name:        "supervisor",
        Description: "Main coordinator that delegates tasks",
        Model:       llm,
        ToolsConfig: adk.ToolsConfig{
            ToolsNodeConfig: compose.ToolsNodeConfig{
                Tools: []tool.BaseTool{
                    adk.NewTransferTool("code_agent", codeAgent),
                    adk.NewTransferTool("search_agent", searchAgent),
                },
            },
        },
    })
    
    // 设置子智能体
    supervisorAgent.SetSubAgents(ctx, []adk.Agent{codeAgent, searchAgent})
    
    // 运行
    runner := adk.NewRunner(ctx, adk.RunnerConfig{
        Agent: supervisorAgent,
    })
    
    iter := runner.Query(ctx, "搜索最新的 Go 语言特性并生成示例代码")
    for {
        event, ok := iter.Next()
        if !ok {
            break
        }
        fmt.Printf("Agent: %s, Output: %v\n", event.AgentName, event.Output)
    }
}

```

### 10.3 工作流智能体示例

```go

func main() {
    ctx := context.Background()
    
    // 创建三个子智能体
    step1Agent, _ := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
        Name:        "step1",
        Description: "Data collection",
        Model:       llm,
        ToolsConfig: adk.ToolsConfig{
            ToolsNodeConfig: compose.ToolsNodeConfig{
                Tools: []tool.BaseTool{collectTool},
            },
        },
        OutputKey: "collected_data",
    })
    
    step2Agent, _ := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
        Name:        "step2",
        Description: "Data processing",
        Instruction: "Process the collected data: {collected_data}",
        Model:       llm,
        OutputKey:   "processed_data",
    })
    
    step3Agent, _ := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
        Name:        "step3",
        Description: "Report generation",
        Instruction: "Generate report based on: {processed_data}",
        Model:       llm,
    })
    
    // 创建顺序工作流智能体
    workflow, _ := adk.NewSequentialAgent(ctx, &adk.SequentialAgentConfig{
        Name:        "data_pipeline",
        Description: "Data collection, processing, and reporting pipeline",
        SubAgents:   []adk.Agent{step1Agent, step2Agent, step3Agent},
    })
    
    // 运行工作流
    runner := adk.NewRunner(ctx, adk.RunnerConfig{
        Agent: workflow,
    })
    
    iter := runner.Query(ctx, "生成今日销售报告")
    for {
        event, ok := iter.Next()
        if !ok {
            break
        }
        fmt.Printf("[%s] %v\n", event.AgentName, event.Output)
    }
}

```

---

## 十一、总结

ADK 模块是 Eino 框架中用于构建智能体应用的高层抽象，它提供了：

1. **统一的 Agent 接口**：定义了智能体的核心行为（Run、Name、Description）
2. **ReAct 模式实现**：基于 Compose 模块的 Graph，实现了经典的推理-行动循环
3. **多智能体协作**：通过 FlowAgent 包装器支持智能体间的转发和父子关系
4. **工作流智能体**：提供顺序、并行、循环三种模式，简化多步骤任务的编排
5. **会话管理**：通过 Session Values 在智能体间共享上下文
6. **中断和恢复**：支持长时间运行任务的断点保存和恢复
7. **预制智能体**：提供 PlanExecute、Supervisor 等常见模式的实现

ADK 模块在保持灵活性的同时，大幅降低了智能体开发的复杂度，是构建复杂 LLM 应用的强大工具。
