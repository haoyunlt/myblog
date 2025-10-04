---
title: "Eino 源码剖析 - Flow 模块：预制流程和组件"
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
description: "Eino 源码剖析 - Eino 源码剖析 - Flow 模块：预制流程和组件"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# Eino-06-flow模块-预制流程和组件

## 一、模块概览

### 1.1 模块定位

Flow 模块是 Eino 框架提供的**开箱即用的预制流程和组件集合**。它在 Compose、Components 和 ADK 模块的基础上，提供了经过实践验证的高级组件和应用模式，让开发者能够快速构建生产级别的 LLM 应用。

### 1.2 核心职责

1. **预制智能体（Agent）**：
   - ReAct Agent：实现 Reasoning and Acting 模式
   - Host Multi-Agent：实现主管-专家协作模式
   
2. **高级检索器（Retriever）**：
   - Multi-Query Retriever：多查询重写提升召回
   - Router Retriever：路由到不同检索器并融合结果
   - Parent Retriever：从子文档召回父文档
   
3. **高级索引器（Indexer）**：
   - Parent Indexer：支持父子文档拆分和管理

### 1.3 模块架构

```mermaid
graph TB
    subgraph "Flow 预制流程和组件"
        subgraph "Agent 预制智能体"
            ReactAgent[ReAct Agent]
            HostMultiAgent[Host Multi-Agent]
        end
        
        subgraph "Retriever 高级检索器"
            MultiQueryRetriever[MultiQuery Retriever]
            RouterRetriever[Router Retriever]
            ParentRetriever[Parent Retriever]
        end
        
        subgraph "Indexer 高级索引器"
            ParentIndexer[Parent Indexer]
        end
        
        subgraph "依赖的核心模块"
            Compose[Compose 编排]
            Components[Components 组件]
            ADK[ADK 智能体套件]
        end
    end
    
    ReactAgent --> Compose
    ReactAgent --> Components
    
    HostMultiAgent --> Compose
    HostMultiAgent --> Components
    
    MultiQueryRetriever --> Compose
    MultiQueryRetriever --> Components
    
    RouterRetriever --> Components
    ParentRetriever --> Components
    ParentIndexer --> Components

    style ReactAgent fill:#e1f5ff
    style HostMultiAgent fill:#fff4e1
    style MultiQueryRetriever fill:#e8f5e9
    style RouterRetriever fill:#f3e5f5
    style ParentRetriever fill:#ede7f6
    style ParentIndexer fill:#fce4ec
</mermaid>

**架构说明**：

1. **Agent 层**：提供两种预制的多智能体模式
2. **Retriever 层**：提供三种高级检索策略
3. **Indexer 层**：提供父子文档管理能力
4. **依赖关系**：Flow 模块依赖 Compose、Components 和 ADK，是最上层的预制组件集

---

## 二、ReAct Agent 详解

### 2.1 ReAct 模式概述

ReAct（Reasoning and Acting）是一种经典的智能体模式，通过**推理-行动-观察**的循环来完成任务：

1. **Reasoning（推理）**：大模型分析用户输入和历史，决定下一步行动
2. **Acting（行动）**：执行工具调用
3. **Observation（观察）**：收集工具执行结果
4. **循环**：将结果加入历史，继续推理

### 2.2 ReAct Agent 配置

#### 2.2.1 API 签名

```go

func NewAgent(ctx context.Context, config *AgentConfig) (*Agent, error)

```

#### 2.2.2 配置参数

```go

type AgentConfig struct {
    // 大模型（必填，推荐使用 ToolCallingModel）
    ToolCallingModel model.ToolCallingChatModel
    Model            model.ChatModel  // 已废弃
    
    // 工具配置
    ToolsConfig compose.ToolsNodeConfig
    
    // 消息修改器（可选）
    MessageModifier MessageModifier
    
    // 最大步数（默认 12）
    MaxStep int
    
    // 直接返回的工具
    ToolReturnDirectly map[string]struct{}
    
    // 流式工具调用检查器（可选）
    StreamToolCallChecker func(ctx context.Context, modelOutput *schema.StreamReader[*schema.Message]) (bool, error)
    
    // 自定义节点名称（可选）
    GraphName      string
    ModelNodeName  string
    ToolsNodeName  string
}

type MessageModifier func(ctx context.Context, msgs []*schema.Message) []*schema.Message

```

**字段说明**：

- `ToolCallingModel`：支持工具调用的大模型（推荐）
- `ToolsConfig.Tools`：智能体可用的工具列表
- `MessageModifier`：在模型调用前修改消息，常用于添加系统提示词
- `MaxStep`：Pregel 执行的最大步数，防止无限循环
- `ToolReturnDirectly`：指定哪些工具调用后直接返回结果
- `StreamToolCallChecker`：自定义流式输出中的工具调用检测逻辑

#### 2.2.3 核心代码实现

```go

func NewAgent(ctx context.Context, config *AgentConfig) (*Agent, error) {
    // 生成工具信息
    toolInfos, _ := genToolInfos(ctx, config.ToolsConfig)
    
    // 创建支持工具的模型
    chatModel, _ := agent.ChatModelWithTools(config.Model, config.ToolCallingModel, toolInfos)
    
    // 创建工具节点
    toolsNode, _ := compose.NewToolNode(ctx, &config.ToolsConfig)
    
    // 创建 Graph（带状态管理）
    graph := compose.NewGraph[[]*schema.Message, *schema.Message](
        compose.WithGenLocalState(func(ctx context.Context) *state {
            return &state{Messages: make([]*schema.Message, 0, config.MaxStep+1)}
        }))
    
    // 添加 Model 节点（带前置处理）
    modelPreHandle := func(ctx context.Context, input []*schema.Message, state *state) ([]*schema.Message, error) {
        state.Messages = append(state.Messages, input...)
        
        if config.MessageModifier != nil {
            modifiedInput := make([]*schema.Message, len(state.Messages))
            copy(modifiedInput, state.Messages)
            return config.MessageModifier(ctx, modifiedInput), nil
        }
        
        return state.Messages, nil
    }
    graph.AddChatModelNode("model", chatModel, compose.WithStatePreHandler(modelPreHandle))
    
    // 添加 Tool 节点（带前置处理）
    toolsNodePreHandle := func(ctx context.Context, input *schema.Message, state *state) (*schema.Message, error) {
        if input != nil {
            state.Messages = append(state.Messages, input)
        }
        
        // 检查是否需要直接返回
        lastMsg := state.Messages[len(state.Messages)-1]
        for _, tc := range lastMsg.ToolCalls {
            if _, ok := config.ToolReturnDirectly[tc.Function.Name]; ok {
                state.ReturnDirectlyToolCallID = tc.ID
                break
            }
        }
        
        return lastMsg, nil
    }
    graph.AddToolsNode("tools", toolsNode, compose.WithStatePreHandler(toolsNodePreHandle))
    
    // 添加边
    graph.AddEdge(compose.START, "model")
    
    // 添加条件分支：model -> tools / END
    toolCallCheck := func(ctx context.Context, sMsg *schema.StreamReader[*schema.Message]) (string, error) {
        hasToolCalls := config.StreamToolCallChecker(ctx, sMsg)
        if hasToolCalls {
            return "tools", nil
        }
        return compose.END, nil
    }
    graph.AddBranch("model", toolCallCheck)
    
    // 添加边：tools -> model（循环）
    graph.AddEdge("tools", "model")
    
    // 编译
    runnable, _ := graph.Compile(ctx, compose.WithMaxRunSteps(config.MaxStep))
    
    return &Agent{runnable: runnable}, nil
}

```

### 2.3 ReAct 执行流程

```mermaid

sequenceDiagram
    participant User
    participant ReactAgent
    participant Graph
    participant Model as Model 节点
    participant Tools as Tools 节点
    participant LLM as 大语言模型
    participant ToolExecutor as 工具执行器

    User->>ReactAgent: Generate(ctx, messages)
    ReactAgent->>Graph: Invoke(ctx, messages)
    
    loop ReAct 循环（最多 MaxStep 次）
        Graph->>Model: 执行 Model 节点
        Model->>Model: modelPreHandle（累积历史消息）
        Model->>Model: MessageModifier（修改消息）
        Model->>LLM: Generate(messages)
        LLM-->>Model: Message（含/不含 ToolCalls）
        
        alt 包含 ToolCalls
            Graph->>Tools: 执行 Tools 节点
            Tools->>Tools: toolsNodePreHandle（保存消息）
            Tools->>ToolExecutor: 调用工具
            ToolExecutor-->>Tools: 工具结果
            Tools-->>Graph: []Message（工具结果消息）
            Graph->>Model: 回到 Model 节点
        else 不包含 ToolCalls
            Graph-->>ReactAgent: 输出最终 Message
        end
    end
    
    ReactAgent-->>User: *schema.Message
</mermaid>

**执行流程说明**：

1. **初始化**：创建 Graph 并初始化状态（空消息列表）
2. **Model 节点**：
   - `modelPreHandle`：将输入消息追加到状态的消息历史
   - `MessageModifier`：（可选）修改消息，如添加系统提示词
   - 调用大模型生成输出
3. **条件分支**：
   - 如果输出包含 `ToolCalls`，进入 Tools 节点
   - 否则，结束并返回最终消息
4. **Tools 节点**：
   - `toolsNodePreHandle`：保存模型输出到历史，检查是否需要直接返回
   - 执行工具调用
   - 将工具结果作为新消息返回
5. **循环**：Tools 节点输出回到 Model 节点，继续推理
6. **终止**：达到 MaxStep 或模型不再调用工具

### 2.4 使用示例

```go
import (
    "context"
    "github.com/cloudwego/eino/flow/agent/react"
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
    
    // 创建 ReAct Agent
    agent, _ := react.NewAgent(ctx, &react.AgentConfig{
        ToolCallingModel: llm,
        ToolsConfig: compose.ToolsNodeConfig{
            Tools: []tool.BaseTool{searchTool, calcTool},
        },
        MessageModifier: func(ctx context.Context, msgs []*schema.Message) []*schema.Message {
            // 在消息列表开头添加系统提示词
            systemMsg := schema.SystemMessage("You are a helpful assistant.")
            return append([]*schema.Message{systemMsg}, msgs...)
        },
        MaxStep: 15,
    })
    
    // 使用智能体
    input := []*schema.Message{
        schema.UserMessage("今天北京的天气如何？"),
    }
    
    // 非流式调用
    output, _ := agent.Generate(ctx, input)
    fmt.Println(output.Content)
    
    // 流式调用
    stream, _ := agent.Stream(ctx, input)
    for {
        chunk, err := stream.Recv()
        if err == io.EOF {
            break
        }
        fmt.Print(chunk.Content)
    }
}
```

---

## 三、Host Multi-Agent 详解

### 3.1 Host Multi-Agent 模式概述

Host Multi-Agent 是一种**主管-专家协作模式**，适用于需要多个专业智能体协同完成复杂任务的场景：

1. **Host Agent（主管）**：分析用户输入，决定调用哪个（些）专家智能体
2. **Specialist Agents（专家）**：每个专家负责特定领域的任务
3. **Summarizer（汇总器）**：（可选）当调用多个专家时，汇总他们的输出

### 3.2 Host Multi-Agent 配置

#### 3.2.1 API 签名

```go
func NewMultiAgent(ctx context.Context, config *MultiAgentConfig) (*MultiAgent, error)
```

#### 3.2.2 配置参数

```go
type MultiAgentConfig struct {
    Host        Host              // 主管智能体配置
    Specialists []*Specialist     // 专家智能体列表
    
    Name         string           // 系统名称
    HostNodeName string           // 主管节点名称（默认 "host"）
    
    // 流式工具调用检查器（可选）
    StreamToolCallChecker func(ctx context.Context, modelOutput *schema.StreamReader[*schema.Message]) (bool, error)
    
    // 汇总器（可选，默认简单拼接）
    Summarizer *Summarizer
}

type Host struct {
    ToolCallingModel model.ToolCallingChatModel
    ChatModel        model.ChatModel  // 已废弃
    SystemPrompt     string           // 系统提示词
}

type Specialist struct {
    AgentMeta
    
    // 选项 1：使用 ChatModel
    ChatModel    model.BaseChatModel
    SystemPrompt string
    
    // 选项 2：使用自定义 Invokable/Streamable（如 ReAct Agent）
    Invokable  compose.Invoke[[]*schema.Message, *schema.Message, agent.AgentOption]
    Streamable compose.Stream[[]*schema.Message, *schema.Message, agent.AgentOption]
}

type AgentMeta struct {
    Name        string  // 专家名称（唯一）
    IntendedUse string  // 适用场景描述
}

type Summarizer struct {
    ChatModel    model.BaseChatModel
    SystemPrompt string
}
```

**字段说明**：

- `Host.ToolCallingModel`：主管智能体使用的大模型
- `Host.SystemPrompt`：主管的系统提示词，描述如何选择专家
- `Specialists`：专家智能体列表
  - `Name`：专家的唯一标识
  - `IntendedUse`：描述专家的能力和适用场景，帮助主管做出选择
  - `ChatModel` 或 `Invokable/Streamable`：专家的实现（二选一）
- `Summarizer`：当主管选择多个专家时，用于汇总结果

#### 3.2.3 核心代码实现

```go
func NewMultiAgent(ctx context.Context, config *MultiAgentConfig) (*MultiAgent, error) {
    // 创建 Graph（带状态管理）
    g := compose.NewGraph[[]*schema.Message, *schema.Message](
        compose.WithGenLocalState(func(context.Context) *state { return &state{} }))
    
    // 添加专家答案收集器节点
    g.AddPassthroughNode("specialist_answers_collect")
    
    // 为每个专家创建工具信息
    agentTools := make([]*schema.ToolInfo, 0, len(config.Specialists))
    for _, specialist := range config.Specialists {
        agentTools = append(agentTools, &schema.ToolInfo{
            Name: specialist.Name,
            Desc: specialist.IntendedUse,
            ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
                "reason": {
                    Type: schema.String,
                    Desc: "the reason to call this tool",
                },
            }),
        })
        
        // 添加专家节点
        addSpecialistAgent(specialist, g)
    }
    
    // 创建主管模型（带工具调用能力）
    chatModel, _ := agent.ChatModelWithTools(config.Host.ChatModel, config.Host.ToolCallingModel, agentTools)
    
    // 添加主管节点
    addHostAgent(chatModel, config.Host.SystemPrompt, g, "host")
    
    // 添加条件分支：直接回答 vs 调用专家
    addDirectAnswerBranch("msg2MsgList", g, config.StreamToolCallChecker)
    
    // 添加多专家分支：单个专家 vs 多个专家
    addMultiSpecialistsBranch(g, config.Specialists, config.Summarizer)
    
    // 编译
    runnable, _ := g.Compile(ctx)
    
    return &MultiAgent{runnable: runnable, graph: g}, nil
}
```

### 3.3 Host Multi-Agent 执行流程

```mermaid
sequenceDiagram
    participant User
    participant MultiAgent
    participant Graph
    participant Host as Host 节点
    participant Specialist1 as 专家1 节点
    participant Specialist2 as 专家2 节点
    participant Summarizer as 汇总器节点

    User->>MultiAgent: Generate(ctx, messages)
    MultiAgent->>Graph: Invoke(ctx, messages)
    
    Graph->>Host: 执行 Host 节点
    Host->>Host: 分析用户需求
    
    alt 主管直接回答
        Host-->>Graph: Message（不含 ToolCalls）
        Graph-->>MultiAgent: 输出最终 Message
    else 主管调用单个专家
        Host-->>Graph: Message（含 ToolCall: specialist1）
        Graph->>Specialist1: 执行专家1 节点
        Specialist1-->>Graph: Message（专家1的答案）
        Graph-->>MultiAgent: 输出最终 Message
    else 主管调用多个专家
        Host-->>Graph: Message（含多个 ToolCalls）
        par 并行执行专家
            Graph->>Specialist1: 执行专家1 节点
            Specialist1-->>Graph: Message（答案1）
        and
            Graph->>Specialist2: 执行专家2 节点
            Specialist2-->>Graph: Message（答案2）
        end
        Graph->>Summarizer: 汇总答案
        Summarizer-->>Graph: Message（汇总后的答案）
        Graph-->>MultiAgent: 输出最终 Message
    end
    
    MultiAgent-->>User: *schema.Message
</mermaid>

**执行流程说明**：

1. **Host 节点**：
   - 主管分析用户输入和每个专家的 `IntendedUse`
   - 决定：直接回答、调用单个专家、或调用多个专家
2. **直接回答分支**：
   - 主管输出不含 `ToolCalls` 的消息
   - 直接返回给用户
3. **单个专家分支**：
   - 主管输出含一个 `ToolCall` 的消息
   - 路由到对应的专家节点
   - 专家执行并返回结果
4. **多个专家分支**：
   - 主管输出含多个 `ToolCalls` 的消息
   - 并行执行所有被调用的专家
   - 汇总器收集所有专家的答案并生成最终回答

### 3.4 使用示例

```go

import (
    "context"
    "github.com/cloudwego/eino/flow/agent/multiagent/host"
    "github.com/cloudwego/eino/components/model"
    "github.com/cloudwego/eino/schema"
)

func main() {
    ctx := context.Background()
    
    // 创建大模型
    llm := model.NewOpenAIChatModel(...)
    
    // 创建专家智能体
    codeSpecialist := &host.Specialist{
        AgentMeta: host.AgentMeta{
            Name:        "code_expert",
            IntendedUse: "Expert in code generation, code review, and debugging. Use this when the task involves writing, analyzing, or fixing code.",
        },
        ChatModel:    llm,
        SystemPrompt: "You are a code expert. Help users with programming tasks.",
    }
    
    searchSpecialist := &host.Specialist{
        AgentMeta: host.AgentMeta{
            Name:        "search_expert",
            IntendedUse: "Expert in web search and information retrieval. Use this when the task requires finding information online.",
        },
        ChatModel:    llm,
        SystemPrompt: "You are a search expert. Help users find information.",
    }
    
    mathSpecialist := &host.Specialist{
        AgentMeta: host.AgentMeta{
            Name:        "math_expert",
            IntendedUse: "Expert in mathematics and calculations. Use this when the task involves mathematical reasoning or computations.",
        },
        ChatModel:    llm,
        SystemPrompt: "You are a math expert. Help users solve mathematical problems.",
    }
    
    // 创建 Host Multi-Agent
    multiAgent, _ := host.NewMultiAgent(ctx, &host.MultiAgentConfig{
        Name: "assistant_team",
        Host: host.Host{
            ToolCallingModel: llm,
            SystemPrompt:     "You are a coordinator. Analyze the user's request and decide which expert(s) to consult. You can also answer directly if no expert is needed.",
        },
        Specialists: []*host.Specialist{
            codeSpecialist,
            searchSpecialist,
            mathSpecialist,
        },
        Summarizer: &host.Summarizer{
            ChatModel:    llm,
            SystemPrompt: "Summarize the experts' answers into a cohesive response.",
        },
    })
    
    // 使用多智能体系统
    input := []*schema.Message{
        schema.UserMessage("帮我搜索最新的 Go 语言特性，并生成一个使用新特性的代码示例"),
    }
    
    output, _ := multiAgent.Generate(ctx, input)
    fmt.Println(output.Content)
}

```

---

## 四、Multi-Query Retriever 详解

### 4.1 Multi-Query 模式概述

Multi-Query Retriever 通过**查询重写（Query Rewriting）**来提升检索的召回率：

1. **查询扩展**：将用户的单个查询扩展为多个相似但表述不同的查询
2. **并行检索**：使用扩展后的查询并行检索文档
3. **结果融合**：去重和融合多个检索结果

### 4.2 Multi-Query Retriever 配置

#### 4.2.1 API 签名

```go

func NewRetriever(ctx context.Context, config *Config) (retriever.Retriever, error)

```

#### 4.2.2 配置参数

```go

type Config struct {
    // 查询重写配置（选项 1：使用 LLM）
    RewriteLLM      model.ChatModel
    RewriteTemplate prompt.ChatTemplate
    QueryVar        string
    LLMOutputParser func(context.Context, *schema.Message) ([]string, error)
    
    // 查询重写配置（选项 2：自定义函数）
    RewriteHandler func(ctx context.Context, query string) ([]string, error)
    
    // 最大查询数量（默认 5）
    MaxQueriesNum int
    
    // 原始检索器（必填）
    OrigRetriever retriever.Retriever
    
    // 融合函数（默认按 ID 去重）
    FusionFunc func(ctx context.Context, docs [][]*schema.Document) ([]*schema.Document, error)
}

```

**字段说明**：

- **查询重写方式**：
  - 使用 LLM：配置 `RewriteLLM`、`RewriteTemplate`（可选）、`LLMOutputParser`（可选）
  - 自定义函数：配置 `RewriteHandler`（优先级更高）
- `MaxQueriesNum`：限制重写后的查询数量，超出部分会被截断
- `OrigRetriever`：底层检索器（如向量数据库检索器）
- `FusionFunc`：融合多个检索结果的函数，默认按文档 ID 去重

#### 4.2.3 核心代码实现

```go

func NewRetriever(ctx context.Context, config *Config) (retriever.Retriever, error) {
    // 构建查询重写 Chain
    rewriteChain := compose.NewChain[string, []string]()
    
    if config.RewriteHandler != nil {
        // 使用自定义函数
        rewriteChain.AppendLambda(compose.InvokableLambda(config.RewriteHandler))
    } else {
        // 使用 LLM 进行查询重写
        tpl := config.RewriteTemplate
        if tpl == nil {
            tpl = prompt.FromMessages(schema.Jinja2, schema.UserMessage(defaultRewritePrompt))
        }
        
        parser := config.LLMOutputParser
        if parser == nil {
            parser = func(ctx context.Context, message *schema.Message) ([]string, error) {
                return strings.Split(message.Content, "\n"), nil
            }
        }
        
        rewriteChain.
            AppendLambda(compose.InvokableLambda(func(ctx context.Context, input string) (map[string]any, error) {
                return map[string]any{config.QueryVar: input}, nil
            })).
            AppendChatTemplate(tpl).
            AppendChatModel(config.RewriteLLM).
            AppendLambda(compose.InvokableLambda(parser))
    }
    
    rewriteRunner, _ := rewriteChain.Compile(ctx)
    
    maxQueriesNum := config.MaxQueriesNum
    if maxQueriesNum == 0 {
        maxQueriesNum = 5
    }
    
    fusionFunc := config.FusionFunc
    if fusionFunc == nil {
        fusionFunc = deduplicateFusion  // 默认去重融合
    }
    
    return &multiQueryRetriever{
        queryRunner:   rewriteRunner,
        maxQueriesNum: maxQueriesNum,
        origRetriever: config.OrigRetriever,
        fusionFunc:    fusionFunc,
    }, nil
}

func (m *multiQueryRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
    // 1. 生成多个查询
    queries, _ := m.queryRunner.Invoke(ctx, query)
    if len(queries) > m.maxQueriesNum {
        queries = queries[:m.maxQueriesNum]
    }
    
    // 2. 并行检索
    tasks := make([]*utils.RetrieveTask, len(queries))
    for i := range queries {
        tasks[i] = &utils.RetrieveTask{Retriever: m.origRetriever, Query: queries[i]}
    }
    utils.ConcurrentRetrieveWithCallback(ctx, tasks)
    
    result := make([][]*schema.Document, len(queries))
    for i, task := range tasks {
        result[i] = task.Result
    }
    
    // 3. 融合结果
    fusionDocs, _ := m.fusionFunc(ctx, result)
    return fusionDocs, nil
}

```

### 4.3 Multi-Query 执行流程

```mermaid

sequenceDiagram
    participant User
    participant MultiQueryRetriever
    participant RewriteChain as 查询重写 Chain
    participant LLM as 大语言模型
    participant OrigRetriever as 原始检索器
    participant FusionFunc as 融合函数

    User->>MultiQueryRetriever: Retrieve(ctx, "如何使用 Eino 构建智能体")
    MultiQueryRetriever->>RewriteChain: Invoke(ctx, query)
    
    RewriteChain->>LLM: Generate（提示：生成3个不同版本的查询）
    LLM-->>RewriteChain: Message（含多个查询）
    RewriteChain->>RewriteChain: OutputParser（按换行符分割）
    RewriteChain-->>MultiQueryRetriever: ["如何使用 Eino 构建智能体", "Eino 智能体开发教程", "构建 Eino Agent 的步骤"]
    
    par 并行检索
        MultiQueryRetriever->>OrigRetriever: Retrieve(ctx, "如何使用 Eino 构建智能体")
        OrigRetriever-->>MultiQueryRetriever: [doc1, doc2, doc3]
    and
        MultiQueryRetriever->>OrigRetriever: Retrieve(ctx, "Eino 智能体开发教程")
        OrigRetriever-->>MultiQueryRetriever: [doc2, doc4, doc5]
    and
        MultiQueryRetriever->>OrigRetriever: Retrieve(ctx, "构建 Eino Agent 的步骤")
        OrigRetriever-->>MultiQueryRetriever: [doc1, doc5, doc6]
    end
    
    MultiQueryRetriever->>FusionFunc: Fusion([[doc1, doc2, doc3], [doc2, doc4, doc5], [doc1, doc5, doc6]])
    FusionFunc->>FusionFunc: 按 ID 去重
    FusionFunc-->>MultiQueryRetriever: [doc1, doc2, doc3, doc4, doc5, doc6]
    
    MultiQueryRetriever-->>User: []*schema.Document
</mermaid>

**执行流程说明**：

1. **查询重写**：
   - 使用 LLM 或自定义函数将单个查询扩展为多个查询
   - 限制查询数量不超过 `MaxQueriesNum`
2. **并行检索**：
   - 使用扩展后的每个查询并行调用原始检索器
   - 收集所有检索结果
3. **结果融合**：
   - 调用 `FusionFunc` 融合多个检索结果
   - 默认实现按文档 ID 去重
4. **返回**：返回融合后的文档列表

### 4.4 使用示例

```go
import (
    "context"
    "github.com/cloudwego/eino/flow/retriever/multiquery"
    "github.com/cloudwego/eino/components/model"
    "github.com/cloudwego/eino/components/retriever"
)

func main() {
    ctx := context.Background()
    
    // 创建原始检索器（如向量数据库检索器）
    origRetriever := retriever.NewMilvusRetriever(...)
    
    // 创建 LLM
    llm := model.NewOpenAIChatModel(...)
    
    // 创建 Multi-Query Retriever
    mqRetriever, _ := multiquery.NewRetriever(ctx, &multiquery.Config{
        RewriteLLM:    llm,
        MaxQueriesNum: 3,
        OrigRetriever: origRetriever,
    })
    
    // 使用检索器
    docs, _ := mqRetriever.Retrieve(ctx, "如何使用 Eino 构建智能体")
    for _, doc := range docs {
        fmt.Println(doc.Content)
    }
}
```

---

## 五、Router Retriever 详解

### 5.1 Router 模式概述

Router Retriever 通过**路由机制**将查询分发到不同的检索器，然后融合结果：

1. **路由决策**：根据查询内容决定调用哪些检索器
2. **并行检索**：并行调用选中的检索器
3. **结果融合**：融合多个检索器的结果（默认使用 RRF - Reciprocal Rank Fusion）

### 5.2 Router Retriever 配置

#### 5.2.1 API 签名

```go
func NewRetriever(ctx context.Context, config *Config) (retriever.Retriever, error)
```

#### 5.2.2 配置参数

```go
type Config struct {
    // 检索器映射（必填）
    Retrievers map[string]retriever.Retriever
    
    // 路由函数（可选，默认调用所有检索器）
    Router func(ctx context.Context, query string) ([]string, error)
    
    // 融合函数（可选，默认使用 RRF）
    FusionFunc func(ctx context.Context, result map[string][]*schema.Document) ([]*schema.Document, error)
}
```

**字段说明**：

- `Retrievers`：检索器映射，键为检索器名称，值为检索器实例
- `Router`：路由函数，返回应该调用的检索器名称列表
  - 默认实现：调用所有检索器
- `FusionFunc`：融合函数，输入为 `map[检索器名称][]*Document`，输出为融合后的文档列表
  - 默认实现：RRF（Reciprocal Rank Fusion）算法

#### 5.2.3 核心代码实现

```go
func NewRetriever(ctx context.Context, config *Config) (retriever.Retriever, error) {
    router := config.Router
    if router == nil {
        // 默认路由：调用所有检索器
        var retrieverSet []string
        for k := range config.Retrievers {
            retrieverSet = append(retrieverSet, k)
        }
        router = func(ctx context.Context, query string) ([]string, error) {
            return retrieverSet, nil
        }
    }
    
    fusion := config.FusionFunc
    if fusion == nil {
        fusion = rrf  // 默认使用 RRF 算法
    }
    
    return &routerRetriever{
        retrievers: config.Retrievers,
        router:     router,
        fusionFunc: fusion,
    }, nil
}

func (r *routerRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
    // 1. 路由决策
    retrieverNames, _ := r.router(ctx, query)
    
    // 2. 并行检索
    result := make(map[string][]*schema.Document)
    var mu sync.Mutex
    var wg sync.WaitGroup
    
    for _, name := range retrieverNames {
        ret := r.retrievers[name]
        wg.Add(1)
        go func(n string, r retriever.Retriever) {
            defer wg.Done()
            docs, _ := r.Retrieve(ctx, query, opts...)
            mu.Lock()
            result[n] = docs
            mu.Unlock()
        }(name, ret)
    }
    wg.Wait()
    
    // 3. 融合结果
    fusionDocs, _ := r.fusionFunc(ctx, result)
    return fusionDocs, nil
}

// RRF（Reciprocal Rank Fusion）算法
func rrf(ctx context.Context, result map[string][]*schema.Document) ([]*schema.Document, error) {
    const k = 60
    scoreMap := make(map[string]float64)
    docMap := make(map[string]*schema.Document)
    
    for _, docs := range result {
        for rank, doc := range docs {
            score := 1.0 / float64(rank+k)
            scoreMap[doc.ID] += score
            docMap[doc.ID] = doc
        }
    }
    
    // 按分数排序
    var fusionDocs []*schema.Document
    for id := range scoreMap {
        fusionDocs = append(fusionDocs, docMap[id])
    }
    sort.Slice(fusionDocs, func(i, j int) bool {
        return scoreMap[fusionDocs[i].ID] > scoreMap[fusionDocs[j].ID]
    })
    
    return fusionDocs, nil
}
```

### 5.3 使用示例

```go
import (
    "context"
    "github.com/cloudwego/eino/flow/retriever/router"
    "github.com/cloudwego/eino/components/retriever"
)

func main() {
    ctx := context.Background()
    
    // 创建多个检索器
    vectorRetriever := retriever.NewMilvusRetriever(...)
    fullTextRetriever := retriever.NewElasticsearchRetriever(...)
    
    // 创建 Router Retriever
    routerRetriever, _ := router.NewRetriever(ctx, &router.Config{
        Retrievers: map[string]retriever.Retriever{
            "vector":    vectorRetriever,
            "fulltext":  fullTextRetriever,
        },
        Router: func(ctx context.Context, query string) ([]string, error) {
            // 自定义路由逻辑
            if len(query) < 10 {
                return []string{"fulltext"}, nil
            }
            return []string{"vector", "fulltext"}, nil
        },
    })
    
    // 使用检索器
    docs, _ := routerRetriever.Retrieve(ctx, "Eino 智能体开发")
    for _, doc := range docs {
        fmt.Println(doc.Content)
    }
}
```

---

## 六、Parent Document Retriever 和 Indexer 详解

### 6.1 Parent Document 模式概述

Parent Document 模式用于解决**大文档检索**的问题：

1. **索引阶段**：将大文档切分为小块（子文档），并建立父子关系
2. **检索阶段**：检索子文档，然后返回对应的父文档

**优势**：

- 提升检索精度：小块更容易匹配查询
- 保留完整上下文：返回整个父文档，避免信息缺失

### 6.2 Parent Indexer

#### 6.2.1 API 签名

```go
func NewIndexer(ctx context.Context, config *Config) (indexer.Indexer, error)
```

#### 6.2.2 配置参数

```go
type Config struct {
    // 底层索引器（必填）
    Indexer indexer.Indexer
    
    // 文档拆分器（必填）
    Transformer document.Transformer
    
    // 父文档 ID 键（元数据键名）
    ParentIDKey string
    
    // 子文档 ID 生成器（必填）
    SubIDGenerator func(ctx context.Context, parentID string, num int) ([]string, error)
}
```

**字段说明**：

- `Indexer`：底层索引器（如 Milvus、Elasticsearch）
- `Transformer`：文档拆分器，将大文档切分为小块
- `ParentIDKey`：在子文档的元数据中存储父文档 ID 的键名
- `SubIDGenerator`：为子文档生成唯一 ID 的函数

#### 6.2.3 核心代码实现

```go
func (p *parentIndexer) Store(ctx context.Context, docs []*schema.Document, opts ...indexer.Option) ([]string, error) {
    // 1. 使用 Transformer 拆分文档
    subDocs, _ := p.transformer.Transform(ctx, docs)
    
    // 2. 为每个子文档设置父文档 ID
    currentID := subDocs[0].ID
    startIdx := 0
    for i, subDoc := range subDocs {
        if subDoc.MetaData == nil {
            subDoc.MetaData = make(map[string]interface{})
        }
        subDoc.MetaData[p.parentIDKey] = subDoc.ID  // 保存父文档 ID
        
        if subDoc.ID == currentID {
            continue
        }
        
        // 3. 为同一父文档的子文档生成新 ID
        subIDs, _ := p.subIDGenerator(ctx, subDocs[startIdx].ID, i-startIdx)
        for j := startIdx; j < i; j++ {
            subDocs[j].ID = subIDs[j-startIdx]
        }
        startIdx = i
        currentID = subDoc.ID
    }
    
    // 处理最后一批子文档
    subIDs, _ := p.subIDGenerator(ctx, subDocs[startIdx].ID, len(subDocs)-startIdx)
    for j := startIdx; j < len(subDocs); j++ {
        subDocs[j].ID = subIDs[j-startIdx]
    }
    
    // 4. 存储子文档
    return p.indexer.Store(ctx, subDocs, opts...)
}
```

### 6.3 Parent Retriever

#### 6.3.1 API 签名

```go
func NewRetriever(ctx context.Context, config *Config) (retriever.Retriever, error)
```

#### 6.3.2 配置参数

```go
type Config struct {
    // 底层检索器（必填）
    Retriever retriever.Retriever
    
    // 父文档 ID 键（元数据键名）
    ParentIDKey string
    
    // 父文档获取器（必填）
    OrigDocGetter func(ctx context.Context, ids []string) ([]*schema.Document, error)
}
```

**字段说明**：

- `Retriever`：底层检索器（检索子文档）
- `ParentIDKey`：从子文档元数据中读取父文档 ID 的键名
- `OrigDocGetter`：根据父文档 ID 列表获取父文档的函数

#### 6.3.3 核心代码实现

```go
func (p *parentRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
    // 1. 检索子文档
    subDocs, _ := p.retriever.Retrieve(ctx, query, opts...)
    
    // 2. 提取父文档 ID
    ids := make([]string, 0, len(subDocs))
    for _, subDoc := range subDocs {
        if k, ok := subDoc.MetaData[p.parentIDKey]; ok {
            if s, okk := k.(string); okk && !inList(s, ids) {
                ids = append(ids, s)
            }
        }
    }
    
    // 3. 获取父文档
    return p.origDocGetter(ctx, ids)
}
```

### 6.4 Parent Document 完整流程

```mermaid
sequenceDiagram
    participant User
    participant ParentIndexer
    participant Transformer
    participant Indexer
    participant ParentRetriever
    participant Retriever
    participant DocStore as 文档存储

    Note over User,DocStore: 索引阶段
    User->>ParentIndexer: Store(ctx, [大文档])
    ParentIndexer->>Transformer: Transform（拆分文档）
    Transformer-->>ParentIndexer: [子文档1, 子文档2, 子文档3]
    ParentIndexer->>ParentIndexer: 为每个子文档添加 parent_id 元数据
    ParentIndexer->>ParentIndexer: 为子文档生成新 ID
    ParentIndexer->>Indexer: Store([子文档])
    Indexer-->>ParentIndexer: [子文档 ID]
    ParentIndexer-->>User: [子文档 ID]

    Note over User,DocStore: 检索阶段
    User->>ParentRetriever: Retrieve(ctx, query)
    ParentRetriever->>Retriever: Retrieve（检索子文档）
    Retriever-->>ParentRetriever: [子文档1, 子文档3]
    ParentRetriever->>ParentRetriever: 从元数据提取 parent_id
    ParentRetriever->>DocStore: GetByIDs([parent_id])
    DocStore-->>ParentRetriever: [父文档]
    ParentRetriever-->>User: [父文档]
</mermaid>

### 6.5 使用示例

```go

import (
    "context"
    "fmt"
    "github.com/cloudwego/eino/flow/indexer/parent"
    "github.com/cloudwego/eino/flow/retriever/parent"
    "github.com/cloudwego/eino/components/document"
    "github.com/cloudwego/eino/schema"
)

func main() {
    ctx := context.Background()
    
    // === 索引阶段 ===
    
    // 创建文档拆分器
    textSplitter := document.NewRecursiveCharacterTextSplitter(...)
    
    // 创建底层索引器
    milvusIndexer := indexer.NewMilvusIndexer(...)
    
    // 创建 Parent Indexer
    parentIndexer, _ := parentIndexer.NewIndexer(ctx, &parentIndexer.Config{
        Indexer:     milvusIndexer,
        Transformer: textSplitter,
        ParentIDKey: "parent_id",
        SubIDGenerator: func(ctx context.Context, parentID string, num int) ([]string, error) {
            ids := make([]string, num)
            for i := 0; i < num; i++ {
                ids[i] = fmt.Sprintf("%s_chunk_%d", parentID, i)
            }
            return ids, nil
        },
    })
    
    // 索引大文档
    docs := []*schema.Document{
        {ID: "doc1", Content: "这是一个很长的文档..."},
        {ID: "doc2", Content: "另一个很长的文档..."},
    }
    parentIndexer.Store(ctx, docs)
    
    // === 检索阶段 ===
    
    // 创建底层检索器
    milvusRetriever := retriever.NewMilvusRetriever(...)
    
    // 创建文档存储（用于获取父文档）
    docStore := NewDocumentStore(...)
    
    // 创建 Parent Retriever
    parentRetriever, _ := parentRetriever.NewRetriever(ctx, &parentRetriever.Config{
        Retriever:   milvusRetriever,
        ParentIDKey: "parent_id",
        OrigDocGetter: func(ctx context.Context, ids []string) ([]*schema.Document, error) {
            return docStore.GetByIDs(ctx, ids)
        },
    })
    
    // 检索文档
    docs, _ := parentRetriever.Retrieve(ctx, "查询内容")
    for _, doc := range docs {
        fmt.Println(doc.Content)  // 输出完整的父文档
    }
}

```

---

## 七、最佳实践

### 7.1 ReAct Agent

1. **工具选择**：只配置真正需要的工具，工具过多会影响模型选择准确性
2. **MessageModifier**：使用 MessageModifier 动态添加系统提示词和上下文
3. **MaxStep**：根据任务复杂度合理设置最大步数
4. **StreamToolCallChecker**：针对不同模型（如 Claude）自定义工具调用检测逻辑

### 7.2 Host Multi-Agent

1. **专家定义**：
   - `Name` 应简洁明了
   - `IntendedUse` 应详细描述专家的能力和适用场景
2. **主管提示词**：在 `Host.SystemPrompt` 中明确指导主管如何选择专家
3. **Summarizer**：对于需要多专家协作的场景，配置 Summarizer 提升回答质量
4. **专家实现**：可以使用 ChatModel 或自定义 Invokable/Streamable（如嵌套 ReAct Agent）

### 7.3 Multi-Query Retriever

1. **查询重写**：
   - 默认提示词适用于大多数场景
   - 对于特定领域，自定义 `RewriteTemplate` 可提升效果
2. **MaxQueriesNum**：建议 3-5 个，过多会增加延迟和成本
3. **FusionFunc**：
   - 简单场景使用默认去重即可
   - 复杂场景可实现自定义融合逻辑（如加权融合）

### 7.4 Router Retriever

1. **路由策略**：
   - 简单场景：调用所有检索器
   - 复杂场景：使用 LLM 或规则引擎实现智能路由
2. **融合算法**：
   - RRF 适用于大多数场景
   - 可根据业务需求实现自定义融合算法

### 7.5 Parent Document

1. **文档拆分**：
   - 选择合适的 `Transformer`（如 RecursiveCharacterTextSplitter）
   - 拆分粒度应平衡检索精度和召回率
2. **ID 生成**：确保 `SubIDGenerator` 生成的 ID 唯一且可追溯
3. **父文档存储**：
   - 将父文档存储在高性能数据库（如 MongoDB、PostgreSQL）
   - 实现高效的批量查询接口

---

## 八、总结

Flow 模块是 Eino 框架的**预制组件库**，提供了经过实践验证的高级组件和应用模式：

1. **Agent 层**：
   - ReAct Agent：经典的推理-行动循环模式
   - Host Multi-Agent：主管-专家协作模式
   
2. **Retriever 层**：
   - Multi-Query Retriever：查询重写提升召回
   - Router Retriever：多检索器融合
   - Parent Retriever：父子文档检索
   
3. **Indexer 层**：
   - Parent Indexer：父子文档索引管理

Flow 模块通过组合 Compose、Components 和 ADK 的能力，为开发者提供了开箱即用的解决方案，大幅降低了构建复杂 LLM 应用的门槛。
