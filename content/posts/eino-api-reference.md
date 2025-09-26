---
title: "Eino 框架 API 参考手册"
date: 2024-12-19T10:00:00+08:00
draft: false
tags: ['Eino', 'Go', 'LLM框架', 'CloudWeGo']
categories: ["eino", "技术分析"]
description: "深入分析 Eino 框架 API 参考手册 的技术实现和架构设计"
weight: 40
slug: "eino-api-reference"
---

# Eino 框架 API 参考手册

## 1. 核心 API 概览

Eino 作为 LLM 应用开发框架，主要提供编程 API 而非网络 API。其对外接口主要分为以下几类：

### 1.1 API 分类表

| API 类型 | 协议 | 模块 | 主要用途 | 入口函数 |
|---------|------|------|---------|----------|
| **编排 API** | Go API | `compose` | 构建 LLM 应用流程 | `NewChain`, `NewGraph`, `NewWorkflow` |
| **组件 API** | Go API | `components` | 定义可复用组件 | 各组件的 `New*` 函数 |
| **Agent API** | Go API | `adk` | 构建智能体 | `NewChatModelAgent`, `NewRunner` |
| **流程 API** | Go API | `flow` | 预构建流程 | `react.NewAgent`, `host.NewMultiAgent` |
| **回调 API** | Go API | `callbacks` | 切面与监控 | `NewHandlerBuilder` |

### 1.2 主要入口点

Eino 框架提供以下主要 API 入口点：

| API 类别 | 主要接口 | 功能描述 |
|---------|---------|---------|
| 编排 API | `NewChain[I,O]()` | 创建链式编排 |
| 编排 API | `NewGraph[I,O]()` | 创建图式编排 |
| 编排 API | `NewWorkflow[I,O]()` | 创建工作流编排 |
| 组件 API | `model.BaseChatModel` | 聊天模型接口 |
| 组件 API | `tool.InvokableTool` | 工具组件接口 |
| 组件 API | `prompt.ChatTemplate` | 模板组件接口 |
| 代理 API | `react.NewAgent()` | 创建 ReAct 代理 |
| 流处理 API | `schema.StreamReader[T]` | 流式数据处理 |

### 1.3 核心接口定义

#### 1.3.1 Runnable 可执行接口

```go
// Runnable 是所有可执行对象的核心接口
// 位置: compose/runnable.go:32
type Runnable[I, O any] interface {
    // Invoke 同步执行：单输入 -> 单输出
    Invoke(ctx context.Context, input I, opts ...Option) (output O, err error)
    
    // Stream 流式执行：单输入 -> 流输出
    Stream(ctx context.Context, input I, opts ...Option) (output *schema.StreamReader[O], err error)
    
    // Collect 收集执行：流输入 -> 单输出
    Collect(ctx context.Context, input *schema.StreamReader[I], opts ...Option) (output O, err error)
    
    // Transform 转换执行：流输入 -> 流输出
    Transform(ctx context.Context, input *schema.StreamReader[I], opts ...Option) (output *schema.StreamReader[O], err error)
}
```

**设计目的**：
- 提供统一的执行接口，支持四种数据流模式
- 自动处理流式和非流式间的转换
- 确保类型安全的泛型设计

**调用链路分析**：
1. 用户调用任一执行方法
2. 内部通过 `composableRunnable` 进行类型转换
3. 根据实际实现自动选择最优执行路径
4. 支持流式和非流式间的自动适配

### 1.4 核心 API 清单

#### 1.4.1 Chain API
| 方法 | 签名 | 说明 | 文件位置 |
|------|------|------|----------|
| `NewChain` | `func NewChain[I, O any](opts ...NewGraphOption) *Chain[I, O]` | 创建链式编排 | `compose/chain.go:37` |
| `AppendChatModel` | `func (c *Chain[I, O]) AppendChatModel(chatModel model.BaseChatModel, opts ...GraphAddNodeOpt) *Chain[I, O]` | 添加聊天模型节点 | `compose/chain.go:L150+` |
| `AppendChatTemplate` | `func (c *Chain[I, O]) AppendChatTemplate(chatTemplate prompt.ChatTemplate, opts ...GraphAddNodeOpt) *Chain[I, O]` | 添加聊天模板节点 | `compose/chain.go:L160+` |
| `AppendToolsNode` | `func (c *Chain[I, O]) AppendToolsNode(tools *ToolsNode, opts ...GraphAddNodeOpt) *Chain[I, O]` | 添加工具节点 | `compose/chain.go:L170+` |
| `Compile` | `func (c *Chain[I, O]) Compile(ctx context.Context, opts ...GraphCompileOption) (Runnable[I, O], error)` | 编译为可执行对象 | `compose/chain.go:L500+` |

#### 1.4.2 Graph API
| 方法 | 签名 | 说明 | 文件位置 |
|------|------|------|----------|
| `NewGraph` | `func NewGraph[I, O any](opts ...NewGraphOption) *Graph[I, O]` | 创建图编排 | `compose/generic_graph.go:68` |
| `AddChatModelNode` | `func (g *Graph[I, O]) AddChatModelNode(key string, chatModel model.BaseChatModel, opts ...GraphAddNodeOpt) error` | 添加聊天模型节点 | `compose/graph.go:L200+` |
| `AddEdge` | `func (g *Graph[I, O]) AddEdge(from, to string, opts ...GraphAddEdgeOpt) error` | 添加边 | `compose/graph.go:L300+` |
| `AddBranch` | `func (g *Graph[I, O]) AddBranch(from string, branch *GraphBranch, opts ...GraphAddBranchOpt) error` | 添加分支 | `compose/graph.go:L400+` |
| `Compile` | `func (g *Graph[I, O]) Compile(ctx context.Context, opts ...GraphCompileOption) (Runnable[I, O], error)` | 编译为可执行对象 | `compose/graph.go:L600+` |

#### 1.4.3 Workflow API
| 方法 | 签名 | 说明 | 文件位置 |
|------|------|------|----------|
| `NewWorkflow` | `func NewWorkflow[I, O any](opts ...NewGraphOption) *Workflow[I, O]` | 创建工作流 | `compose/workflow.go:61` |
| `AddChatModelNode` | `func (wf *Workflow[I, O]) AddChatModelNode(key string, chatModel model.BaseChatModel, opts ...GraphAddNodeOpt) *WorkflowNode` | 添加聊天模型节点 | `compose/workflow.go:85` |
| `End` | `func (wf *Workflow[I, O]) End() *WorkflowNode` | 设置结束节点 | `compose/workflow.go:L200+` |
| `Compile` | `func (wf *Workflow[I, O]) Compile(ctx context.Context, opts ...GraphCompileOption) (Runnable[I, O], error)` | 编译为可执行对象 | `compose/workflow.go:81` |

### 1.5 核心执行接口

```go
// Runnable 是所有可执行对象的核心接口
type Runnable[I, O any] interface {
    Invoke(ctx context.Context, input I, opts ...Option) (output O, err error)
    Stream(ctx context.Context, input I, opts ...Option) (output *schema.StreamReader[O], err error)
    Collect(ctx context.Context, input *schema.StreamReader[I], opts ...Option) (output O, err error)
    Transform(ctx context.Context, input *schema.StreamReader[I], opts ...Option) (output *schema.StreamReader[O], err error)
}
```

## 2. Schema 模块 API

### 2.1 消息系统 API

#### 2.1.1 Message 结构体

```go
type Message struct {
    Role    RoleType `json:"role"`    // 消息角色：user、assistant、system、tool
    Content string   `json:"content"` // 消息内容
    
    // 多媒体内容支持
    MultiContent []ChatMessagePart `json:"multi_content,omitempty"`
    
    // 工具调用相关
    ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
    ToolCallID string     `json:"tool_call_id,omitempty"`
    ToolName   string     `json:"tool_name,omitempty"`
    
    // 响应元数据
    ResponseMeta *ResponseMeta `json:"response_meta,omitempty"`
    
    // 推理内容
    ReasoningContent string `json:"reasoning_content,omitempty"`
    
    // 扩展字段
    Extra map[string]any `json:"extra,omitempty"`
}
```

#### 2.1.2 消息创建函数

```go
// SystemMessage 创建系统消息
func SystemMessage(content string) *Message

// UserMessage 创建用户消息  
func UserMessage(content string) *Message

// AssistantMessage 创建助手消息
func AssistantMessage(content string, toolCalls []ToolCall) *Message

// ToolMessage 创建工具消息
func ToolMessage(content string, toolCallID string, opts ...ToolMessageOption) *Message
```

**使用示例：**
```go
// 创建对话消息
messages := []*schema.Message{
    schema.SystemMessage("你是一个有用的助手"),
    schema.UserMessage("今天天气怎么样？"),
    schema.AssistantMessage("我需要调用天气API来获取信息", toolCalls),
    schema.ToolMessage("今天晴天，温度25°C", "call_123", schema.WithToolName("weather")),
}
```

#### 2.1.3 消息处理函数

```go
// ConcatMessages 合并消息流
func ConcatMessages(msgs []*Message) (*Message, error)

// ConcatMessageStream 合并消息流读取器
func ConcatMessageStream(s *StreamReader[*Message]) (*Message, error)
```

**调用链路分析：**
1. `ConcatMessages` 验证消息角色一致性
2. 合并消息内容字符串
3. 处理工具调用合并（按 Index 分组）
4. 合并响应元数据（取最大值）
5. 处理扩展字段合并

### 2.2 流处理 API

#### 2.2.1 StreamReader 接口

```go
type StreamReader[T any] struct {
    // 内部实现
}

// 核心方法
func (sr *StreamReader[T]) Recv() (T, error)
func (sr *StreamReader[T]) Close() error
```

#### 2.2.2 流创建函数

```go
// StreamReaderFromArray 从数组创建流
func StreamReaderFromArray[T any](items []T) *StreamReader[T]

// StreamReaderWithConvert 带转换的流创建
func StreamReaderWithConvert[T, U any](sr *StreamReader[T], convert func(T) (U, error)) *StreamReader[U]

// MergeStreamReaders 合并多个流
func MergeStreamReaders[T any](readers []*StreamReader[T]) *StreamReader[T]
```

**使用示例：**
```go
// 创建消息流
messages := []*schema.Message{
    schema.UserMessage("Hello"),
    schema.AssistantMessage("Hi there!", nil),
}
stream := schema.StreamReaderFromArray(messages)

// 读取流数据
for {
    msg, err := stream.Recv()
    if err == io.EOF {
        break
    }
    if err != nil {
        return err
    }
    fmt.Println(msg.Content)
}
```

### 2.3 工具信息 API

#### 2.3.1 ToolInfo 结构体

```go
type ToolInfo struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Parameters  *jsonschema.Schema     `json:"parameters"`
    Extra       map[string]interface{} `json:"extra,omitempty"`
}
```

#### 2.3.2 ToolCall 结构体

```go
type ToolCall struct {
    Index    *int         `json:"index,omitempty"`    // 流式模式下的索引
    ID       string       `json:"id"`                 // 工具调用ID
    Type     string       `json:"type"`               // 调用类型，默认"function"
    Function FunctionCall `json:"function"`           // 函数调用信息
    Extra    map[string]any `json:"extra,omitempty"`  // 扩展字段
}

type FunctionCall struct {
    Name      string `json:"name,omitempty"`      // 函数名
    Arguments string `json:"arguments,omitempty"` // JSON格式参数
}
```

## 3. Components 模块 API

### 3.1 ChatModel 组件 API

#### 3.1.1 BaseChatModel 接口

```go
type BaseChatModel interface {
    Generate(ctx context.Context, input []*schema.Message, opts ...Option) (*schema.Message, error)
    Stream(ctx context.Context, input []*schema.Message, opts ...Option) (*schema.StreamReader[*schema.Message], error)
}
```

**关键函数调用链路：**

**Generate 方法：**
1. 接收消息数组输入
2. 应用配置选项（温度、最大token等）
3. 调用底层模型API
4. 处理响应并构造 Message
5. 执行回调处理器
6. 返回生成的消息

**Stream 方法：**
1. 创建流式响应通道
2. 启动异步生成协程
3. 逐块发送响应数据
4. 处理流式回调
5. 返回 StreamReader

#### 3.1.2 ToolCallingChatModel 接口

```go
type ToolCallingChatModel interface {
    BaseChatModel
    WithTools(tools []*schema.ToolInfo) (ToolCallingChatModel, error)
}
```

**使用示例：**
```go
// 创建支持工具调用的模型
model, err := openai.NewChatModel(ctx, &openai.ChatModelConfig{
    Model: "gpt-4",
})
if err != nil {
    return err
}

// 绑定工具
toolCallingModel, err := model.WithTools([]*schema.ToolInfo{weatherTool})
if err != nil {
    return err
}

// 生成响应
response, err := toolCallingModel.Generate(ctx, messages)
```

### 3.2 Tool 组件 API

#### 3.2.1 工具接口定义

```go
// BaseTool 基础工具接口
type BaseTool interface {
    Info(ctx context.Context) (*schema.ToolInfo, error)
}

// InvokableTool 可调用工具接口
type InvokableTool interface {
    BaseTool
    InvokableRun(ctx context.Context, argumentsInJSON string, opts ...Option) (string, error)
}

// StreamableTool 流式工具接口
type StreamableTool interface {
    BaseTool
    StreamableRun(ctx context.Context, argumentsInJSON string, opts ...Option) (*schema.StreamReader[string], error)
}
```

#### 3.2.2 工具实现示例

```go
type WeatherTool struct{}

func (w *WeatherTool) Info(ctx context.Context) (*schema.ToolInfo, error) {
    return &schema.ToolInfo{
        Name:        "get_weather",
        Description: "获取指定城市的天气信息",
        Parameters: &jsonschema.Schema{
            Type: "object",
            Properties: map[string]*jsonschema.Schema{
                "city": {
                    Type:        "string",
                    Description: "城市名称",
                },
            },
            Required: []string{"city"},
        },
    }, nil
}

func (w *WeatherTool) InvokableRun(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (string, error) {
    var args struct {
        City string `json:"city"`
    }
    
    err := json.Unmarshal([]byte(argumentsInJSON), &args)
    if err != nil {
        return "", err
    }
    
    // 调用天气API
    weather := getWeatherInfo(args.City)
    return weather, nil
}
```

### 3.3 ChatTemplate 组件 API

#### 3.3.1 ChatTemplate 接口

```go
type ChatTemplate interface {
    Format(ctx context.Context, vs map[string]any, opts ...Option) ([]*schema.Message, error)
}
```

#### 3.3.2 模板创建函数

```go
// FromMessages 从消息模板创建
func FromMessages(formatType schema.FormatType, messages ...schema.MessagesTemplate) (ChatTemplate, error)

// FromString 从字符串创建简单模板
func FromString(formatType schema.FormatType, template string) ChatTemplate
```

**使用示例：**
```go
// 创建 Jinja2 模板
template, err := prompt.FromMessages(schema.Jinja2,
    schema.SystemMessage("你是一个{{role}}助手"),
    schema.MessagesPlaceholder("history", true),
    schema.UserMessage("{{query}}"),
)

// 格式化模板
messages, err := template.Format(ctx, map[string]any{
    "role":    "专业的",
    "query":   "什么是人工智能？",
    "history": previousMessages,
})
```

**关键函数调用链路：**
1. `Format` 方法接收变量映射
2. 根据 FormatType 选择模板引擎
3. 对每个消息模板进行变量替换
4. 处理 MessagesPlaceholder 占位符
5. 返回格式化后的消息数组

### 3.4 其他组件 API

#### 3.4.1 Retriever 组件

```go
type Retriever interface {
    Retrieve(ctx context.Context, query string, opts ...Option) ([]*schema.Document, error)
}
```

#### 3.4.2 Embedding 组件

```go
type Embedder interface {
    EmbedStrings(ctx context.Context, texts []string, opts ...Option) ([][]float64, error)
    EmbedDocuments(ctx context.Context, docs []*schema.Document, opts ...Option) ([][]float64, error)
}
```

#### 3.4.3 Indexer 组件

```go
type Indexer interface {
    Index(ctx context.Context, docs []*schema.Document, opts ...Option) error
}
```

## 4. Compose 模块 API

### 4.1 Graph 编排 API

#### 4.1.1 Graph 创建与配置

```go
// NewGraph 创建新的图编排
func NewGraph[I, O any](opts ...NewGraphOption) *Graph[I, O]

// 图配置选项
func WithGenLocalState[S any](generator func(ctx context.Context) *S) NewGraphOption
func WithMaxRunSteps(steps int) NewGraphOption
func WithNodeTriggerMode(mode NodeTriggerMode) NewGraphOption
```

#### 4.1.2 节点添加 API

```go
// AddChatModelNode 添加聊天模型节点
func (g *Graph[I, O]) AddChatModelNode(key string, node model.BaseChatModel, opts ...GraphAddNodeOpt) error

// AddToolsNode 添加工具节点
func (g *Graph[I, O]) AddToolsNode(key string, node *ToolsNode, opts ...GraphAddNodeOpt) error

// AddChatTemplateNode 添加模板节点
func (g *Graph[I, O]) AddChatTemplateNode(key string, node prompt.ChatTemplate, opts ...GraphAddNodeOpt) error

// AddLambdaNode 添加自定义Lambda节点
func (g *Graph[I, O]) AddLambdaNode(key string, node *Lambda, opts ...GraphAddNodeOpt) error

// AddPassthroughNode 添加透传节点
func (g *Graph[I, O]) AddPassthroughNode(key string, opts ...GraphAddNodeOpt) error
```

#### 4.1.3 边和分支 API

```go
// AddEdge 添加边
func (g *Graph[I, O]) AddEdge(startNode, endNode string) error

// AddBranch 添加分支
func (g *Graph[I, O]) AddBranch(startNode string, branch *GraphBranch) error

// NewGraphBranch 创建图分支
func NewGraphBranch[T any](condition func(context.Context, T) (string, error), endNodes map[string]bool) *GraphBranch

// NewStreamGraphBranch 创建流式图分支
func NewStreamGraphBranch[T any](condition func(context.Context, *schema.StreamReader[T]) (string, error), endNodes map[string]bool) *GraphBranch
```

#### 4.1.4 图编译与执行

```go
// Compile 编译图为可执行对象
func (g *Graph[I, O]) Compile(ctx context.Context, opts ...GraphCompileOption) (Runnable[I, O], error)

// 编译选项
func WithMaxRunSteps(steps int) GraphCompileOption
func WithNodeTriggerMode(mode NodeTriggerMode) GraphCompileOption
func WithGraphName(name string) GraphCompileOption
```

**使用示例：**
```go
// 创建图
graph := compose.NewGraph[map[string]any, *schema.Message]()

// 添加节点
err := graph.AddChatTemplateNode("template", chatTemplate)
err = graph.AddChatModelNode("model", chatModel)
err = graph.AddToolsNode("tools", toolsNode)

// 添加边
err = graph.AddEdge(compose.START, "template")
err = graph.AddEdge("template", "model")

// 添加分支
branch := compose.NewGraphBranch(func(ctx context.Context, msg *schema.Message) (string, error) {
    if len(msg.ToolCalls) > 0 {
        return "tools", nil
    }
    return compose.END, nil
}, map[string]bool{"tools": true, compose.END: true})

err = graph.AddBranch("model", branch)
err = graph.AddEdge("tools", "model")

// 编译并执行
runnable, err := graph.Compile(ctx)
result, err := runnable.Invoke(ctx, map[string]any{"query": "Hello"})
```

### 4.2 Chain 编排 API 详解

#### 4.2.1 基本信息
- **模块**: `compose`
- **入口函数**: `compose.NewChain` (`compose/chain.go:37`)
- **协议**: Go API
- **用途**: 构建线性链式 LLM 应用流程

#### 4.2.2 核心接口

##### NewChain - 创建链式编排
```go
// NewChain create a chain with input/output type.
// 创建具有输入/输出类型的链式编排，支持泛型类型安全
func NewChain[I, O any](opts ...NewGraphOption) *Chain[I, O]
```

**设计目的**: 提供简单的链式编排能力，组件按顺序执行，数据从前一个组件流向后一个组件。

**调用链路径**:
| 深度 | 包/类 | 函数 | 作用 | 备注 |
|---:|---|---|---|---|
| 0 | `compose` | `NewChain` | 创建链实例 | 泛型类型检查 |
| 1 | `compose` | `NewGraph` | 创建底层图结构 | 复用图编排能力 |
| 2 | `compose` | `newGraphFromGeneric` | 初始化图配置 | 类型反射处理 |

##### 主要方法

###### AppendChatModel - 添加聊天模型
```go
// AppendChatModel adds a chat model node to the chain.
// 向链中添加聊天模型节点，用于生成对话回复
func (c *Chain[I, O]) AppendChatModel(chatModel model.BaseChatModel, opts ...GraphAddNodeOpt) *Chain[I, O]
```

**前置条件**: 
- Chain 未编译
- chatModel 实现 `model.BaseChatModel` 接口

**后置条件**:
- 在链中添加新的聊天模型节点
- 自动连接到前一个节点

###### AppendChatTemplate - 添加聊天模板
```go
// AppendChatTemplate adds a chat template node to the chain.
// 向链中添加聊天模板节点，用于格式化输入消息
func (c *Chain[I, O]) AppendChatTemplate(chatTemplate prompt.ChatTemplate, opts ...GraphAddNodeOpt) *Chain[I, O]
```

###### AppendToolsNode - 添加工具节点
```go
// AppendToolsNode adds a tools node to the chain.
// 向链中添加工具节点，用于执行工具调用
func (c *Chain[I, O]) AppendToolsNode(tools *ToolsNode, opts ...GraphAddNodeOpt) *Chain[I, O]
```

###### Compile - 编译链
```go
// Compile compiles the chain into a runnable object.
// 将链编译为可执行对象，进行类型检查和优化
func (c *Chain[I, O]) Compile(ctx context.Context, opts ...GraphCompileOption) (Runnable[I, O], error)
```

**设计目的**: 将构建时的链结构转换为运行时的可执行对象，进行类型安全检查。

**调用链关键路径**:
| 深度 | 包/类 | 函数 | 作用 | 备注 |
|---:|---|---|---|---|
| 0 | `compose` | `Compile` | 编译入口 | 类型检查 |
| 1 | `compose` | `addEndIfNeeded` | 添加结束节点 | 确保链完整性 |
| 2 | `compose` | `compile` | 底层编译逻辑 | 图编译 |
| 3 | `compose` | `buildComposableRunnable` | 构建可执行对象 | 运行时优化 |

#### 4.2.3 使用示例

##### 基础链式编排
```go
// 创建简单的模板->模型链
chain := compose.NewChain[map[string]any, *schema.Message]().
    AppendChatTemplate(template).
    AppendChatModel(model)

// 编译
runnable, err := chain.Compile(ctx)
if err != nil {
    return fmt.Errorf("compilation failed: %w", err)
}

// 执行
result, err := runnable.Invoke(ctx, map[string]any{
    "query": "什么是人工智能？",
})
```

##### 带工具的链式编排
```go
// 创建模板->模型->工具链
chain := compose.NewChain[map[string]any, *schema.Message]().
    AppendChatTemplate(template).
    AppendChatModel(model).
    AppendToolsNode(toolsNode)

runnable, _ := chain.Compile(ctx)
result, _ := runnable.Invoke(ctx, input)
```

##### 流式处理
```go
// 流式执行链
stream, err := runnable.Stream(ctx, input)
if err != nil {
    return err
}

// 处理流式输出
for {
    chunk, err := stream.Recv()
    if err == io.EOF {
        break
    }
    if err != nil {
        return err
    }
    // 处理每个数据块
    fmt.Printf("Chunk: %s\n", chunk.Content)
}
```

#### 4.2.4 错误处理

##### 编译时错误
- `ErrChainCompiled`: 链已编译，无法修改
- 类型不匹配错误: 相邻节点的输入输出类型不兼容
- 节点配置错误: 组件配置不正确

##### 运行时错误
- 组件执行失败
- 超时错误
- 上下文取消

#### 4.2.5 性能特性

##### 执行模式
| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `Invoke` | 同步执行，等待完整结果 | 批处理场景 |
| `Stream` | 流式执行，实时输出 | 交互式对话 |
| `Collect` | 收集流式输入 | 处理流式数据源 |
| `Transform` | 流到流转换 | 数据管道 |

##### 性能预算
- **编译延迟**: < 10ms (简单链)
- **执行开销**: < 1ms (框架层面)
- **内存占用**: 与组件数量线性相关

#### 4.2.6 并发与安全

##### 线程安全
- Chain 构建过程非线程安全
- 编译后的 Runnable 线程安全
- 支持并发执行多个请求

##### 资源管理
- 通过 Context 控制超时和取消
- 自动清理临时资源
- 支持优雅关闭

#### 4.2.7 扩展点

##### 自定义节点
```go
// 添加自定义 Lambda 节点
chain.AppendLambda(func(ctx context.Context, input InputType) (OutputType, error) {
    // 自定义处理逻辑
    return processInput(input), nil
})
```

##### 并行处理
```go
// 添加并行节点
parallel := compose.NewParallel().
    AddLambda("task1", task1).
    AddLambda("task2", task2)

chain.AppendParallel(parallel)
```

##### 分支逻辑
```go
// 添加条件分支
branch := compose.NewChainBranch(func(ctx context.Context, input InputType) (string, error) {
    if shouldGoLeft(input) {
        return "left", nil
    }
    return "right", nil
})

chain.AppendBranch(branch)
```

#### 4.2.8 最佳实践

1. **类型安全**: 使用泛型确保编译时类型检查
2. **错误处理**: 在每个阶段检查错误
3. **资源管理**: 使用 Context 控制生命周期
4. **性能优化**: 对于实时场景使用流式处理
5. **监控**: 使用回调机制添加监控和日志

### 4.3 Workflow 编排 API

#### 4.3.1 Workflow 创建与节点管理

```go
// NewWorkflow 创建新的工作流
func NewWorkflow[I, O any](opts ...NewGraphOption) *Workflow[I, O]

// 节点添加方法
func (wf *Workflow[I, O]) AddChatModelNode(key string, chatModel model.BaseChatModel, opts ...GraphAddNodeOpt) *WorkflowNode
func (wf *Workflow[I, O]) AddLambdaNode(key string, lambda *Lambda, opts ...GraphAddNodeOpt) *WorkflowNode

// End 获取结束节点
func (wf *Workflow[I, O]) End() *WorkflowNode
```

#### 4.3.2 WorkflowNode 输入映射 API

```go
// AddInput 添加输入映射
func (n *WorkflowNode) AddInput(fromNodeKey string, inputs ...*FieldMapping) *WorkflowNode

// AddInputWithOptions 带选项的输入映射
func (n *WorkflowNode) AddInputWithOptions(fromNodeKey string, inputs []*FieldMapping, opts ...WorkflowAddInputOpt) *WorkflowNode

// AddDependency 添加纯依赖关系
func (n *WorkflowNode) AddDependency(fromNodeKey string) *WorkflowNode

// SetStaticValue 设置静态值
func (n *WorkflowNode) SetStaticValue(path FieldPath, value any) *WorkflowNode
```

#### 4.3.3 字段映射 API

```go
// MapFields 创建字段映射
func MapFields(from, to string) *FieldMapping

// ToFieldPath 创建字段路径映射
func ToFieldPath(path FieldPath) *FieldMapping

// WithNoDirectDependency 无直接依赖选项
func WithNoDirectDependency() WorkflowAddInputOpt
```

**使用示例：**
```go
// 创建工作流
wf := compose.NewWorkflow[Input, Output]()

// 添加节点并配置映射
modelNode := wf.AddChatModelNode("model", model).AddInput(compose.START)

processorNode := wf.AddLambdaNode("processor", lambda).
    AddInput("model", compose.MapFields("Content", "Input"))

// 结束节点
wf.End().AddInput("processor")

// 编译执行
runnable, err := wf.Compile(ctx)
```

### 4.4 Lambda 函数 API

#### 4.4.1 Lambda 创建函数

```go
// InvokableLambda 创建可调用Lambda
func InvokableLambda[I, O any](fn func(context.Context, I) (O, error)) *Lambda

// StreamableLambda 创建流式Lambda
func StreamableLambda[I, O any](fn func(context.Context, I) (*schema.StreamReader[O], error)) *Lambda

// CollectableLambda 创建收集式Lambda
func CollectableLambda[I, O any](fn func(context.Context, *schema.StreamReader[I]) (O, error)) *Lambda

// TransformableLambda 创建转换式Lambda
func TransformableLambda[I, O any](fn func(context.Context, *schema.StreamReader[I]) (*schema.StreamReader[O], error)) *Lambda

// AnyLambda 创建任意组合Lambda
func AnyLambda[I, O any](invoke Invoke[I, O], stream Stream[I, O], collect Collect[I, O], transform Transform[I, O]) *Lambda
```

**使用示例：**
```go
// 创建文本处理Lambda
textProcessor := compose.InvokableLambda(func(ctx context.Context, input string) (string, error) {
    return strings.ToUpper(input), nil
})

// 创建流式处理Lambda
streamProcessor := compose.StreamableLambda(func(ctx context.Context, input string) (*schema.StreamReader[string], error) {
    words := strings.Fields(input)
    return schema.StreamReaderFromArray(words), nil
})
```

## 5. ADK 模块 API

### 5.1 Agent 接口 API

#### 5.1.1 核心 Agent 接口

```go
type Agent interface {
    Name(ctx context.Context) string
    Description(ctx context.Context) string
    Run(ctx context.Context, input *AgentInput, options ...AgentRunOption) *AsyncIterator[*AgentEvent]
}

type AgentInput struct {
    Messages        []Message
    EnableStreaming bool
}

type AgentEvent struct {
    AgentName string
    RunPath   []RunStep
    Output    *AgentOutput
    Action    *AgentAction
    Err       error
}
```

#### 5.1.2 Agent 输出与动作

```go
type AgentOutput struct {
    MessageOutput    *MessageVariant
    CustomizedOutput any
}

type AgentAction struct {
    Exit            bool
    Interrupted     *InterruptInfo
    TransferToAgent *TransferToAgentAction
    CustomizedAction any
}
```

### 5.2 ReAct Agent API

#### 5.2.1 ReAct Agent 创建

```go
// NewAgent 创建ReAct代理
func NewAgent(ctx context.Context, config *AgentConfig) (*Agent, error)

type AgentConfig struct {
    // 推荐使用的工具调用模型
    ToolCallingModel model.ToolCallingChatModel
    
    // 已废弃：使用 ToolCallingModel 替代
    Model model.ChatModel
    
    // 工具配置
    ToolsConfig compose.ToolsNodeConfig
    
    // 消息修改器
    MessageModifier MessageModifier
    
    // 最大步数
    MaxStep int
    
    // 直接返回的工具
    ToolReturnDirectly map[string]struct{}
    
    // 流式工具调用检查器
    StreamToolCallChecker func(ctx context.Context, modelOutput *schema.StreamReader[*schema.Message]) (bool, error)
    
    // 图名称配置
    GraphName     string
    ModelNodeName string
    ToolsNodeName string
}
```

#### 5.2.2 ReAct Agent 执行

```go
// Generate 生成响应
func (r *Agent) Generate(ctx context.Context, input []*schema.Message, opts ...agent.AgentOption) (*schema.Message, error)

// Stream 流式生成
func (r *Agent) Stream(ctx context.Context, input []*schema.Message, opts ...agent.AgentOption) (*schema.StreamReader[*schema.Message], error)

// ExportGraph 导出底层图
func (r *Agent) ExportGraph() (compose.AnyGraph, []compose.GraphAddNodeOpt)
```

**使用示例：**
```go
// 创建ReAct代理
agent, err := react.NewAgent(ctx, &react.AgentConfig{
    ToolCallingModel: toolCallingModel,
    ToolsConfig: compose.ToolsNodeConfig{
        Tools: []tool.InvokableTool{weatherTool, calculatorTool},
    },
    MaxStep: 10,
    ToolReturnDirectly: map[string]struct{}{
        "final_answer": {},
    },
})

// 执行对话
response, err := agent.Generate(ctx, []*schema.Message{
    schema.UserMessage("北京今天天气如何？"),
})
```

### 5.3 工具返回控制 API

```go
// SetReturnDirectly 设置工具直接返回
func SetReturnDirectly(ctx context.Context) error

// NewPersonaModifier 创建角色修改器（已废弃）
func NewPersonaModifier(persona string) MessageModifier
```

## 6. Callbacks 模块 API

### 6.1 回调处理器 API

#### 6.1.1 Handler 接口

```go
type Handler interface {
    OnStart(ctx context.Context, info *RunInfo, input CallbackInput) context.Context
    OnEnd(ctx context.Context, info *RunInfo, output CallbackOutput) context.Context
    OnError(ctx context.Context, info *RunInfo, err error) context.Context
    OnStartWithStreamInput(ctx context.Context, info *RunInfo, input CallbackInput) context.Context
    OnEndWithStreamOutput(ctx context.Context, info *RunInfo, output CallbackOutput) context.Context
}

type TimingChecker interface {
    NeedTiming(timing CallbackTiming) bool
}
```

#### 6.1.2 HandlerBuilder API

```go
// NewHandlerBuilder 创建处理器构建器
func NewHandlerBuilder() *HandlerBuilder

// OnStartFn 设置开始回调
func (hb *HandlerBuilder) OnStartFn(fn func(context.Context, *RunInfo, CallbackInput) context.Context) *HandlerBuilder

// OnEndFn 设置结束回调
func (hb *HandlerBuilder) OnEndFn(fn func(context.Context, *RunInfo, CallbackOutput) context.Context) *HandlerBuilder

// OnErrorFn 设置错误回调
func (hb *HandlerBuilder) OnErrorFn(fn func(context.Context, *RunInfo, error) context.Context) *HandlerBuilder

// Build 构建处理器
func (hb *HandlerBuilder) Build() Handler
```

#### 6.1.3 全局回调管理

```go
// AppendGlobalHandlers 添加全局处理器
func AppendGlobalHandlers(handlers ...Handler)

// InitCallbackHandlers 初始化回调处理器（已废弃）
func InitCallbackHandlers(handlers []Handler)
```

**使用示例：**
```go
// 创建回调处理器
handler := callbacks.NewHandlerBuilder().
    OnStartFn(func(ctx context.Context, info *callbacks.RunInfo, input callbacks.CallbackInput) context.Context {
        log.Printf("开始执行组件: %s", info.Name)
        return ctx
    }).
    OnEndFn(func(ctx context.Context, info *callbacks.RunInfo, output callbacks.CallbackOutput) context.Context {
        log.Printf("完成执行组件: %s", info.Name)
        return ctx
    }).
    OnErrorFn(func(ctx context.Context, info *callbacks.RunInfo, err error) context.Context {
        log.Printf("组件执行错误: %s, 错误: %v", info.Name, err)
        return ctx
    }).
    Build()

// 使用回调
result, err := runnable.Invoke(ctx, input, compose.WithCallbacks(handler))
```

## 7. 选项系统 API

### 7.1 通用选项

```go
// WithCallbacks 设置回调处理器
func WithCallbacks(handlers ...callbacks.Handler) Option

// WithTimeout 设置超时时间
func WithTimeout(timeout time.Duration) Option

// WithMaxRunSteps 设置最大运行步数
func WithMaxRunSteps(steps int) Option
```

### 7.2 组件特定选项

```go
// WithChatModelOption 设置聊天模型选项
func WithChatModelOption(opt model.Option) Option

// WithToolOption 设置工具选项
func WithToolOption(opt tool.Option) Option

// WithPromptOption 设置提示选项
func WithPromptOption(opt prompt.Option) Option
```

### 7.3 节点特定选项

```go
// DesignateNode 指定节点
func (opt Option) DesignateNode(nodeKey string) Option

// WithNodeName 设置节点名称
func WithNodeName(name string) GraphAddNodeOpt

// WithStatePreHandler 设置状态前处理器
func WithStatePreHandler[I, S any](handler func(context.Context, I, *S) (I, error)) GraphAddNodeOpt

// WithStatePostHandler 设置状态后处理器
func WithStatePostHandler[O, S any](handler func(context.Context, O, *S) (O, error)) GraphAddNodeOpt
```

## 8. 错误处理 API

### 8.1 中断与恢复

```go
// InterruptError 中断错误
type InterruptError interface {
    error
    GetInterruptInfo() *InterruptInfo
}

// InterruptInfo 中断信息
type InterruptInfo struct {
    State           any
    BeforeNodes     []string
    AfterNodes      []string
    RerunNodes      []string
    RerunNodesExtra map[string]any
    SubGraphs       map[string]*InterruptInfo
}

// ResumeInfo 恢复信息
type ResumeInfo struct {
    CheckpointData map[string]any
    InterruptInfo  *InterruptInfo
}
```

### 8.2 检查点 API

```go
// WithCheckPointID 设置检查点ID
func WithCheckPointID(id string) Option

// WithWriteToCheckPointID 设置写入检查点ID
func WithWriteToCheckPointID(id string) Option

// WithStateModifier 设置状态修改器
func WithStateModifier(modifier StateModifier) Option
```

## 9. 工具节点 API

### 9.1 ToolsNode 创建与配置

```go
// NewToolNode 创建工具节点
func NewToolNode(ctx context.Context, conf *ToolsNodeConfig) (*ToolsNode, error)

type ToolsNodeConfig struct {
    Tools                   []tool.BaseTool
    UnknownToolsHandler     func(ctx context.Context, name, input string) (string, error)
    ExecuteSequentially     bool
    ToolArgumentsHandler    func(ctx context.Context, name, arguments string) (string, error)
}
```

### 9.2 ToolsNode 执行

```go
// Invoke 执行工具调用
func (tn *ToolsNode) Invoke(ctx context.Context, input *schema.Message, opts ...ToolsNodeOption) ([]*schema.Message, error)

// Stream 流式执行工具调用
func (tn *ToolsNode) Stream(ctx context.Context, input *schema.Message, opts ...ToolsNodeOption) (*schema.StreamReader[[]*schema.Message], error)
```

### 9.3 ToolsNode 选项

```go
// WithToolOption 添加工具选项
func WithToolOption(opts ...tool.Option) ToolsNodeOption

// WithToolList 设置工具列表
func WithToolList(tools ...tool.BaseTool) ToolsNodeOption
```

## 10. 最佳实践与示例

### 10.1 完整的应用示例

```go
func main() {
    ctx := context.Background()
    
    // 1. 创建组件
    model, _ := openai.NewChatModel(ctx, &openai.ChatModelConfig{
        Model: "gpt-4",
    })
    
    weatherTool := &WeatherTool{}
    
    // 2. 创建模板
    template, _ := prompt.FromMessages(schema.Jinja2,
        schema.SystemMessage("你是一个有用的助手"),
        schema.UserMessage("{{query}}"),
    )
    
    // 3. 创建工具节点
    toolsNode, _ := compose.NewToolNode(ctx, &compose.ToolsNodeConfig{
        Tools: []tool.BaseTool{weatherTool},
    })
    
    // 4. 构建图
    graph := compose.NewGraph[map[string]any, *schema.Message]()
    graph.AddChatTemplateNode("template", template)
    graph.AddChatModelNode("model", model)
    graph.AddToolsNode("tools", toolsNode)
    
    // 5. 添加边和分支
    graph.AddEdge(compose.START, "template")
    graph.AddEdge("template", "model")
    
    branch := compose.NewGraphBranch(func(ctx context.Context, msg *schema.Message) (string, error) {
        if len(msg.ToolCalls) > 0 {
            return "tools", nil
        }
        return compose.END, nil
    }, map[string]bool{"tools": true, compose.END: true})
    
    graph.AddBranch("model", branch)
    graph.AddEdge("tools", "model")
    
    // 6. 编译并执行
    runnable, _ := graph.Compile(ctx)
    result, _ := runnable.Invoke(ctx, map[string]any{
        "query": "北京今天天气怎么样？",
    })
    
    fmt.Println(result.Content)
}
```

### 10.2 错误处理最佳实践

```go
// 统一错误处理
func handleGraphExecution(ctx context.Context, runnable compose.Runnable[Input, Output], input Input) (Output, error) {
    result, err := runnable.Invoke(ctx, input)
    if err != nil {
        // 检查是否为中断错误
        if interruptErr, ok := err.(compose.InterruptError); ok {
            info := interruptErr.GetInterruptInfo()
            log.Printf("图执行被中断: %+v", info)
            
            // 可以选择恢复执行
            // return resumeExecution(ctx, runnable, info)
        }
        
        return result, fmt.Errorf("图执行失败: %w", err)
    }
    
    return result, nil
}
```

## 11. API 使用模式

### 11.1 基础使用模式

```go
// 1. 创建组件
model := // 实现 model.BaseChatModel 接口
template := // 实现 prompt.ChatTemplate 接口

// 2. 构建链式编排
chain := compose.NewChain[map[string]any, *schema.Message]().
    AppendChatTemplate(template).
    AppendChatModel(model)

// 3. 编译
runnable, err := chain.Compile(ctx)

// 4. 执行
result, err := runnable.Invoke(ctx, map[string]any{"query": "hello"})
```

### 11.2 图编排模式

```go
// 1. 创建图
graph := compose.NewGraph[map[string]any, *schema.Message]()

// 2. 添加节点
graph.AddChatTemplateNode("template", template)
graph.AddChatModelNode("model", model)
graph.AddToolsNode("tools", toolsNode)

// 3. 添加边
graph.AddEdge(compose.START, "template")
graph.AddEdge("template", "model")
graph.AddBranch("model", branch)

// 4. 编译执行
runnable, _ := graph.Compile(ctx)
result, _ := runnable.Invoke(ctx, input)
```

### 11.3 Agent 模式

```go
// 1. 创建 Agent
agent, err := react.NewAgent(ctx, &react.AgentConfig{
    ChatModel: model,
    Tools: tools,
})

// 2. 执行对话
message, err := agent.Generate(ctx, []*schema.Message{
    schema.UserMessage("你好"),
})
```

## 12. 错误处理

### 12.1 常见错误类型

1. **编译错误**: 图结构不合法、类型不匹配
2. **运行时错误**: 组件执行失败、超时
3. **配置错误**: 参数不合法、依赖缺失

### 12.2 错误处理模式

```go
// 编译时错误检查
runnable, err := chain.Compile(ctx)
if err != nil {
    // 处理编译错误
    return fmt.Errorf("compilation failed: %w", err)
}

// 运行时错误处理
result, err := runnable.Invoke(ctx, input, 
    compose.WithTimeout(30*time.Second),
    compose.WithRetry(3),
)
if err != nil {
    // 处理运行时错误
    return fmt.Errorf("execution failed: %w", err)
}
```

## 13. 性能考虑

### 13.1 流式处理
- 使用 `Stream()` 方法获得更好的响应性
- 自动处理流的合并、分发、转换

### 13.2 并发控制
- Graph 支持节点并行执行
- 通过 `NodeTriggerMode` 控制触发模式

### 13.3 资源管理
- 使用 Context 进行超时控制
- 支持优雅关闭和资源清理

## 14. 扩展性

### 14.1 自定义组件
实现对应的组件接口即可集成到编排框架中

### 14.2 自定义回调
通过 `callbacks` 包实现监控、日志、指标收集

### 14.3 自定义 Agent
通过 `adk` 包构建复杂的智能体逻辑

这个 API 参考手册涵盖了 Eino 框架的所有主要接口和使用方法，为开发者提供了完整的 API 使用指南。
