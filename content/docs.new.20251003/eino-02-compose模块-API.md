# Eino-02-Compose模块-API

本文档详细描述 Compose 模块对外提供的所有 API，包括 Runnable 接口、三种编排模式（Chain、Graph、Workflow）的使用方法和核心 API。

---

## 1. Runnable 接口

### 1.1 Runnable 核心接口

**接口定义**:
```go
type Runnable[I, O any] interface {
    // Invoke: 非流输入 => 非流输出
    Invoke(ctx context.Context, input I, opts ...Option) (output O, err error)
    
    // Stream: 非流输入 => 流输出
    Stream(ctx context.Context, input I, opts ...Option) (output *schema.StreamReader[O], err error)
    
    // Collect: 流输入 => 非流输出
    Collect(ctx context.Context, input *schema.StreamReader[I], opts ...Option) (output O, err error)
    
    // Transform: 流输入 => 流输出
    Transform(ctx context.Context, input *schema.StreamReader[I], opts ...Option) (output *schema.StreamReader[O], err error)
}
```

#### 1.1.1 Invoke

**功能说明**: 标准的请求-响应模式，一次性输入，一次性输出。

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| ctx | context.Context | 是 | 上下文 |
| input | I | 是 | 输入数据 |
| opts | ...Option | 否 | 可选配置 |

**返回值**:

| 类型 | 说明 |
|------|------|
| O | 输出数据 |
| error | 错误 |

**使用示例**:
```go
// 示例 1：基本用法
chain := compose.NewChain[string, string]()
chain.AppendChatTemplate("t", template)
chain.AppendChatModel("m", chatModel)

runnable, _ := chain.Compile(ctx)

// 一次性输入，一次性输出
output, err := runnable.Invoke(ctx, "你好")
if err != nil {
    log.Fatal(err)
}
fmt.Println(output)

// 示例 2：带 Callbacks
output, err := runnable.Invoke(ctx, input,
    compose.WithCallbacks(myHandler))

// 示例 3：带组件特定配置
output, err := runnable.Invoke(ctx, input,
    compose.WithChatModelOption(
        model.WithTemperature(0.7),
        model.WithMaxTokens(1000),
    ))
```

**适用场景**:
- 标准的请求-响应模式
- 不需要流式输出
- 数据量不大

---

#### 1.1.2 Stream

**功能说明**: 一次性输入，流式输出，适用于需要实时展示的场景。

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| ctx | context.Context | 是 | 上下文 |
| input | I | 是 | 输入数据 |
| opts | ...Option | 否 | 可选配置 |

**返回值**:

| 类型 | 说明 |
|------|------|
| *StreamReader[O] | 输出流读取器 |
| error | 错误 |

**使用示例**:
```go
// 示例 1：流式接收并打印
stream, err := runnable.Stream(ctx, "讲个笑话")
if err != nil {
    log.Fatal(err)
}
defer stream.Close()

for {
    chunk, err := stream.Recv()
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    fmt.Print(chunk)
}

// 示例 2：使用 ConcatMessageStream 拼接完整输出
stream, _ := runnable.Stream(ctx, input)
fullOutput, _ := schema.ConcatMessageStream(stream)

// 示例 3：流复制到多个处理器
stream, _ := runnable.Stream(ctx, input)
readers := stream.Copy(2)

// 一个用于显示
go func() {
    defer readers[0].Close()
    for {
        chunk, err := readers[0].Recv()
        if err == io.EOF {
            break
        }
        fmt.Print(chunk)
    }
}()

// 一个用于保存
go func() {
    defer readers[1].Close()
    var chunks []string
    for {
        chunk, err := readers[1].Recv()
        if err == io.EOF {
            break
        }
        chunks = append(chunks, chunk)
    }
    saveToFile(chunks)
}()
```

**适用场景**:
- 需要实时展示输出（如聊天）
- 输出内容较多，需要逐步展示
- 需要同时做多种处理（通过 Copy）

---

#### 1.1.3 Collect

**功能说明**: 流式输入，一次性输出，将流数据汇总后处理。

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| ctx | context.Context | 是 | 上下文 |
| input | *StreamReader[I] | 是 | 输入流 |
| opts | ...Option | 否 | 可选配置 |

**返回值**:

| 类型 | 说明 |
|------|------|
| O | 输出数据 |
| error | 错误 |

**使用示例**:
```go
// 示例 1：处理用户的流式输入
inputStream := getUserInputStream()  // 假设从某处获得流
output, err := runnable.Collect(ctx, inputStream)
if err != nil {
    log.Fatal(err)
}
fmt.Println(output)

// 示例 2：从数组创建流输入
chunks := []string{"第一段", "第二段", "第三段"}
inputStream := schema.StreamReaderFromArray(chunks)
output, _ := runnable.Collect(ctx, inputStream)

// 示例 3：管道连接
sr1, sw1 := schema.Pipe[string](10)

go func() {
    defer sw1.Close()
    for i := 0; i < 10; i++ {
        sw1.Send(fmt.Sprintf("chunk-%d", i), nil)
    }
}()

output, _ := runnable.Collect(ctx, sr1)
```

**适用场景**:
- 处理流式输入源
- 需要汇总所有输入后再处理
- 连接多个 Runnable（前一个输出流，后一个接收）

---

#### 1.1.4 Transform

**功能说明**: 流式输入，流式输出，实现端到端的流式处理。

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| ctx | context.Context | 是 | 上下文 |
| input | *StreamReader[I] | 是 | 输入流 |
| opts | ...Option | 否 | 可选配置 |

**返回值**:

| 类型 | 说明 |
|------|------|
| *StreamReader[O] | 输出流 |
| error | 错误 |

**使用示例**:
```go
// 示例 1：流式转换
inputStream := getInputStream()
outputStream, err := runnable.Transform(ctx, inputStream)
if err != nil {
    log.Fatal(err)
}
defer outputStream.Close()

for {
    chunk, err := outputStream.Recv()
    if err == io.EOF {
        break
    }
    processChunk(chunk)
}

// 示例 2：流管道
stream1, _ := runnable1.Stream(ctx, input)
stream2, _ := runnable2.Transform(ctx, stream1)
stream3, _ := runnable3.Transform(ctx, stream2)
// 链式流处理
```

**适用场景**:
- 端到端流式处理
- 连接多个流式处理单元
- 需要最小延迟

---

## 2. Chain API

### 2.1 NewChain

**功能说明**: 创建一个新的 Chain。

**函数签名**:
```go
func NewChain[I, O any](opts ...NewGraphOption) *Chain[I, O]
```

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| opts | ...NewGraphOption | 否 | 可选配置（如 WithState） |

**返回值**:

| 类型 | 说明 |
|------|------|
| *Chain[I, O] | Chain 实例 |

**使用示例**:
```go
// 示例 1：无状态 Chain
chain := compose.NewChain[string, *schema.Message]()

// 示例 2：带状态的 Chain
type MyState struct {
    Counter int
}

chain := compose.NewChain[string, string](
    compose.WithState(func(ctx context.Context) *MyState {
        return &MyState{Counter: 0}
    }),
)
```

---

### 2.2 Chain 添加节点方法

#### 2.2.1 AppendChatModel

**功能说明**: 添加 ChatModel 节点。

**函数签名**:
```go
func (c *Chain[I, O]) AppendChatModel(node model.BaseChatModel, opts ...GraphAddNodeOpt) *Chain[I, O]
```

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| node | model.BaseChatModel | 是 | ChatModel 实例 |
| opts | ...GraphAddNodeOpt | 否 | 节点配置 |

**使用示例**:
```go
chain := compose.NewChain[string, *schema.Message]()

chatModel, _ := openai.NewChatModel(ctx, config)

chain.AppendChatModel(chatModel)

// 带配置
chain.AppendChatModel(chatModel, 
    compose.WithNodeKey("my_model"))  // Chain 专用：指定节点 key
```

#### 2.2.2 AppendChatTemplate

**功能说明**: 添加 ChatTemplate 节点。

**函数签名**:
```go
func (c *Chain[I, O]) AppendChatTemplate(node prompt.ChatTemplate, opts ...GraphAddNodeOpt) *Chain[I, O]
```

**使用示例**:
```go
template := prompt.FromMessages(
    schema.SystemMessage("你是一个助手"),
    schema.UserMessage("{query}"),
)

chain.AppendChatTemplate(template)
```

#### 2.2.3 AppendToolsNode

**功能说明**: 添加 ToolsNode 节点。

**函数签名**:
```go
func (c *Chain[I, O]) AppendToolsNode(node *ToolsNode, opts ...GraphAddNodeOpt) *Chain[I, O]
```

**使用示例**:
```go
toolsNode := compose.NewToolsNode()
toolsNode.RegisterTool(weatherTool, weatherFunc)

chain.AppendToolsNode(toolsNode)
```

#### 2.2.4 AppendRetriever

**功能说明**: 添加 Retriever 节点。

**函数签名**:
```go
func (c *Chain[I, O]) AppendRetriever(node retriever.Retriever, opts ...GraphAddNodeOpt) *Chain[I, O]
```

**使用示例**:
```go
myRetriever, _ := createRetriever()
chain.AppendRetriever(myRetriever)
```

#### 2.2.5 AppendLambda

**功能说明**: 添加 Lambda 节点（自定义处理函数）。

**函数签名**:
```go
func (c *Chain[I, O]) AppendLambda(lambda *Lambda, opts ...GraphAddNodeOpt) *Chain[I, O]
```

**使用示例**:
```go
// 示例 1：简单转换
convertLambda := compose.InvokableLambda(
    func(ctx context.Context, docs []*schema.Document) (string, error) {
        var context strings.Builder
        for _, doc := range docs {
            context.WriteString(doc.Content)
        }
        return context.String(), nil
    })

chain.AppendLambda(convertLambda)

// 示例 2：流式 Lambda
streamLambda := compose.StreamableLambda(
    func(ctx context.Context, input string) (*schema.StreamReader[string], error) {
        sr, sw := schema.Pipe[string](10)
        go func() {
            defer sw.Close()
            words := strings.Fields(input)
            for _, word := range words {
                sw.Send(word, nil)
            }
        }()
        return sr, nil
    })

chain.AppendLambda(streamLambda)
```

#### 2.2.6 AppendGraph

**功能说明**: 添加嵌套的 Graph/Chain。

**函数签名**:
```go
func (c *Chain[I, O]) AppendGraph(graph AnyGraph, opts ...GraphAddNodeOpt) *Chain[I, O]
```

**使用示例**:
```go
// 创建子 Chain
subChain := compose.NewChain[string, string]()
subChain.AppendLambda(lambda1)
subChain.AppendLambda(lambda2)

// 嵌入主 Chain
mainChain := compose.NewChain[map[string]any, string]()
mainChain.AppendLambda(prepareLambda)
mainChain.AppendGraph(subChain)  // 嵌套
mainChain.AppendLambda(finalizeLambda)
```

---

### 2.3 Chain 分支和并行

#### 2.3.1 AppendBranch

**功能说明**: 添加条件分支节点。

**函数签名**:
```go
func (c *Chain[I, O]) AppendBranch(branches ...*ChainBranch) *ChainBranchBuilder[I, O]
```

**使用示例**:
```go
chain := compose.NewChain[string, string]()

// 添加前置节点
chain.AppendLambda(prepareLambda)

// 添加分支
chain.AppendBranch(
    // 分支 1：处理短文本
    compose.NewChainBranch(
        func(ctx context.Context, input string) (bool, error) {
            return len(input) < 100, nil
        },
        shortTextChain,  // 短文本处理链
    ),
    // 分支 2：处理长文本
    compose.NewChainBranch(
        func(ctx context.Context, input string) (bool, error) {
            return len(input) >= 100, nil
        },
        longTextChain,  // 长文本处理链
    ),
).End()  // 结束分支

// 继续添加后续节点
chain.AppendLambda(finalizeLambda)
```

**分支规则**:
- 顺序评估所有分支条件
- 执行第一个条件为 true 的分支
- 如果所有条件都为 false，返回错误

#### 2.3.2 AppendParallel

**功能说明**: 添加并行执行节点。

**函数签名**:
```go
func (c *Chain[I, O]) AppendParallel(branches ...*ParallelBranch) *ParallelBuilder[I, O]
```

**使用示例**:
```go
chain := compose.NewChain[string, map[string]any]()

// 添加并行节点
chain.AppendParallel(
    // 并行分支 1：查询数据库
    compose.NewParallelBranch(dbQueryChain, "db_result"),
    
    // 并行分支 2：调用外部 API
    compose.NewParallelBranch(apiCallChain, "api_result"),
    
    // 并行分支 3：读取缓存
    compose.NewParallelBranch(cacheReadChain, "cache_result"),
).End()  // 等待所有分支完成

// 输出是 map: {
//   "db_result": ...,
//   "api_result": ...,
//   "cache_result": ...
// }

// 继续处理
chain.AppendLambda(mergeLambda)
```

**并行规则**:
- 所有分支同时开始执行
- 等待所有分支完成
- 结果合并为 map[string]any
- 任一分支失败，整体失败

---

### 2.4 Chain 编译

#### 2.4.1 Compile

**功能说明**: 编译 Chain 为可执行的 Runnable。

**函数签名**:
```go
func (c *Chain[I, O]) Compile(ctx context.Context, opts ...GraphCompileOption) (Runnable[I, O], error)
```

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| ctx | context.Context | 是 | 上下文 |
| opts | ...GraphCompileOption | 否 | 编译选项 |

**编译选项**:
```go
// WithMaxRunSteps: 设置最大运行步数（Pregel 模式）
compose.WithMaxRunSteps(10)

// WithRunTimeout: 设置运行超时
compose.WithRunTimeout(30 * time.Second)
```

**使用示例**:
```go
// 示例 1：基本编译
chain := compose.NewChain[string, string]()
// ... 添加节点
runnable, err := chain.Compile(ctx)
if err != nil {
    log.Fatal(err)
}

// 示例 2：带编译选项
runnable, err := chain.Compile(ctx,
    compose.WithMaxRunSteps(10),
    compose.WithRunTimeout(30*time.Second),
)

// 示例 3：缓存编译结果
var runnableCache sync.Map

func getOrCompileChain(key string) (Runnable[I, O], error) {
    if r, ok := runnableCache.Load(key); ok {
        return r.(Runnable[I, O]), nil
    }
    
    chain := buildChain()
    runnable, err := chain.Compile(ctx)
    if err != nil {
        return nil, err
    }
    
    runnableCache.Store(key, runnable)
    return runnable, nil
}
```

---

## 3. Graph API

### 3.1 NewGraph

**功能说明**: 创建一个新的 Graph。

**函数签名**:
```go
func NewGraph[I, O any](opts ...NewGraphOption) *Graph[I, O]
```

**可选配置**:
```go
// WithState: 启用状态管理
compose.WithState(func(ctx context.Context) *MyState {
    return &MyState{}
})
```

**使用示例**:
```go
// 示例 1：无状态 Graph
graph := compose.NewGraph[Input, Output]()

// 示例 2：带状态的 Graph
type AgentState struct {
    Messages     []*schema.Message
    ToolResults  []string
    Iteration    int
}

graph := compose.NewGraph[string, string](
    compose.WithState(func(ctx context.Context) *AgentState {
        return &AgentState{
            Messages:  []*schema.Message{},
            Iteration: 0,
        }
    }),
)
```

---

### 3.2 Graph 添加节点方法

Graph 的添加节点方法与 Chain 类似，但返回 error 而非 *Graph：

```go
// AddChatModelNode 添加 ChatModel 节点
func (g *Graph[I, O]) AddChatModelNode(key string, node model.BaseChatModel, opts ...GraphAddNodeOpt) error

// AddChatTemplateNode 添加 ChatTemplate 节点
func (g *Graph[I, O]) AddChatTemplateNode(key string, node prompt.ChatTemplate, opts ...GraphAddNodeOpt) error

// AddToolsNode 添加 ToolsNode 节点
func (g *Graph[I, O]) AddToolsNode(key string, node *ToolsNode, opts ...GraphAddNodeOpt) error

// AddRetrieverNode 添加 Retriever 节点
func (g *Graph[I, O]) AddRetrieverNode(key string, node retriever.Retriever, opts ...GraphAddNodeOpt) error

// AddLambdaNode 添加 Lambda 节点
func (g *Graph[I, O]) AddLambdaNode(key string, lambda *Lambda, opts ...GraphAddNodeOpt) error

// AddGraphNode 添加嵌套 Graph
func (g *Graph[I, O]) AddGraphNode(key string, graph AnyGraph, opts ...GraphAddNodeOpt) error

// AddPassthroughNode 添加透传节点（不做处理）
func (g *Graph[I, O]) AddPassthroughNode(key string, opts ...GraphAddNodeOpt) error
```

**使用示例**:
```go
graph := compose.NewGraph[Input, Output]()

// 添加各种节点
err := graph.AddChatModelNode("model", chatModel)
if err != nil {
    log.Fatal(err)
}

err = graph.AddToolsNode("tools", toolsNode)
if err != nil {
    log.Fatal(err)
}

// Lambda 节点
lambda := compose.InvokableLambda(func(ctx context.Context, input string) (string, error) {
    return processInput(input), nil
})
err = graph.AddLambdaNode("processor", lambda)
```

---

### 3.3 Graph 连接节点

#### 3.3.1 AddEdge

**功能说明**: 添加边，连接两个节点。

**函数签名**:
```go
func (g *Graph[I, O]) AddEdge(startNode, endNode string, opts ...GraphAddEdgeOpt) error
```

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| startNode | string | 是 | 起始节点 key（可以是 START） |
| endNode | string | 是 | 结束节点 key（可以是 END） |
| opts | ...GraphAddEdgeOpt | 否 | 边配置 |

**边配置选项**:
```go
// 数据边（默认）：传递数据和执行控制
compose.DataEdge()

// 控制边：只传递执行控制，不传递数据
compose.ControlEdge()

// 字段映射：指定字段如何映射
compose.MapFields("SourceField", "TargetField")
```

**使用示例**:
```go
graph := compose.NewGraph[string, string]()

// 添加节点
graph.AddLambdaNode("node1", lambda1)
graph.AddLambdaNode("node2", lambda2)
graph.AddLambdaNode("node3", lambda3)

// 示例 1：简单的边
graph.AddEdge(START, "node1")
graph.AddEdge("node1", "node2")
graph.AddEdge("node2", END)

// 示例 2：控制边（不传递数据）
graph.AddEdge("node1", "node2", compose.ControlEdge())

// 示例 3：字段映射
graph.AddEdge("node1", "node2",
    compose.MapFields("Output", "Input"))

// 示例 4：循环边
graph.AddEdge("tools", "model")  // 工具结果回到模型
```

**关键概念**:
- **START**: 特殊节点，表示 Graph 的输入
- **END**: 特殊节点，表示 Graph 的输出
- **循环边**: 允许数据流回到之前的节点（仅 Pregel 模式）

---

#### 3.3.2 AddBranch

**功能说明**: 添加分支，根据条件路由到不同节点。

**函数签名**:
```go
func (g *Graph[I, O]) AddBranch(startNode string, branch *GraphBranch, opts ...GraphAddEdgeOpt) error
```

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| startNode | string | 是 | 分支起始节点 |
| branch | *GraphBranch | 是 | 分支定义 |
| opts | ...GraphAddEdgeOpt | 否 | 边配置 |

**GraphBranch 创建**:
```go
// NewGraphBranch 创建分支
func NewGraphBranch(
    condition *Lambda,           // 条件判断函数，返回目标节点 key
    pathMap map[string]string,   // 路由映射：条件返回值 -> 目标节点
) *GraphBranch
```

**使用示例**:
```go
graph := compose.NewGraph[Input, Output]()

// 添加节点
graph.AddChatModelNode("model", chatModel)
graph.AddToolsNode("tools", toolsNode)

// 创建分支条件
branchLambda := compose.InvokableLambda(
    func(ctx context.Context, msg *schema.Message) (string, error) {
        if len(msg.ToolCalls) > 0 {
            return "need_tools", nil
        }
        if msg.Content == "" {
            return "error", nil
        }
        return "done", nil
    })

// 添加分支
branch := compose.NewGraphBranch(
    branchLambda,
    map[string]string{
        "need_tools": "tools",  // 需要工具 -> tools 节点
        "done":       END,       // 完成 -> 结束
        "error":      "error_handler",  // 错误 -> 错误处理节点
    },
)

graph.AddBranch("model", branch)
graph.AddEdge("tools", "model")  // 工具完成后回到模型
```

**分支规则**:
- 条件 Lambda 必须返回 string（目标节点 key）
- pathMap 中必须包含所有可能的返回值
- 如果返回值不在 pathMap 中，会报错

---

### 3.4 Graph 状态管理

#### 3.4.1 AddPreHandler

**功能说明**: 添加节点前置处理器，在节点执行前调用。

**函数签名**:
```go
func (g *Graph[I, O]) AddPreHandler(nodeKey string, handler any, opts ...GraphAddNodeOpt) error
```

**使用示例**:
```go
type MyState struct {
    Counter int
}

graph := compose.NewGraph[string, string](
    compose.WithState(func(ctx context.Context) *MyState {
        return &MyState{Counter: 0}
    }),
)

// 添加前置处理器：读取状态
graph.AddPreHandler("node1",
    compose.StatePreHandler(func(ctx context.Context, state *MyState) context.Context {
        log.Printf("当前计数: %d", state.Counter)
        return ctx
    }),
)
```

#### 3.4.2 AddPostHandler

**功能说明**: 添加节点后置处理器，在节点执行后调用。

**函数签名**:
```go
func (g *Graph[I, O]) AddPostHandler(nodeKey string, handler any, opts ...GraphAddNodeOpt) error
```

**使用示例**:
```go
// 添加后置处理器：更新状态
graph.AddPostHandler("node1",
    compose.StatePostHandler(func(ctx context.Context, state *MyState, output any) {
        state.Counter++
        log.Printf("节点执行完成，计数: %d", state.Counter)
    }),
)
```

---

### 3.5 Graph 编译

#### 3.5.1 Compile

**功能说明**: 编译 Graph 为可执行的 Runnable。

**函数签名**:
```go
func (g *Graph[I, O]) Compile(ctx context.Context, opts ...GraphCompileOption) (Runnable[I, O], error)
```

**编译选项**:

```go
// WithMaxRunSteps: 最大迭代次数（Pregel 模式必须设置）
compose.WithMaxRunSteps(10)

// WithRunTimeout: 运行超时
compose.WithRunTimeout(30 * time.Second)

// WithPregelMode: 使用 Pregel 模式（支持循环）
compose.WithPregelMode()

// WithDAGMode: 使用 DAG 模式（不支持循环，性能更好）
compose.WithDAGMode()
```

**使用示例**:
```go
// 示例 1：DAG 模式（默认）
graph := compose.NewGraph[Input, Output]()
// ... 添加节点和边（无循环）
runnable, err := graph.Compile(ctx)

// 示例 2：Pregel 模式（支持循环）
graph := compose.NewGraph[Input, Output]()
// ... 添加节点和边（包含循环）
runnable, err := graph.Compile(ctx,
    compose.WithPregelMode(),
    compose.WithMaxRunSteps(10),  // 必须设置
)

// 示例 3：带超时
runnable, err := graph.Compile(ctx,
    compose.WithRunTimeout(30*time.Second),
)
```

---

## 4. Workflow API

### 4.1 NewWorkflow

**功能说明**: 创建一个新的 Workflow。

**函数签名**:
```go
func NewWorkflow[I, O any](opts ...NewGraphOption) *Workflow[I, O]
```

**使用示例**:
```go
// 创建 Workflow
wf := compose.NewWorkflow[UserInfo, Report]()

// 带状态
type WorkflowState struct {
    IntermediateResults map[string]any
}

wf := compose.NewWorkflow[Input, Output](
    compose.WithState(func(ctx context.Context) *WorkflowState {
        return &WorkflowState{
            IntermediateResults: make(map[string]any),
        }
    }),
)
```

---

### 4.2 Workflow 添加节点

Workflow 的添加节点方法返回 *WorkflowNode，支持链式调用：

```go
// AddChatModelNode 添加 ChatModel 节点
func (wf *Workflow[I, O]) AddChatModelNode(key string, node model.BaseChatModel, opts ...GraphAddNodeOpt) *WorkflowNode

// AddLambdaNode 添加 Lambda 节点
func (wf *Workflow[I, O]) AddLambdaNode(key string, lambda *Lambda, opts ...GraphAddNodeOpt) *WorkflowNode

// ... 其他节点类型类似
```

---

### 4.3 Workflow 依赖和数据映射

#### 4.3.1 AddInput

**功能说明**: 声明节点的输入依赖和字段映射。

**函数签名**:
```go
func (wn *WorkflowNode) AddInput(fromNodeKey string, mappings ...*FieldMapping) *WorkflowNode
```

**参数说明**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| fromNodeKey | string | 是 | 依赖的节点 key（可以是 START） |
| mappings | ...*FieldMapping | 否 | 字段映射规则 |

**FieldMapping 创建**:
```go
// MapFields: 映射字段
compose.MapFields("SourceField", "TargetField")

// MapKey: 映射 map 的 key
compose.MapKey("source_key", "TargetField")

// StaticValue: 设置静态值
compose.StaticValue("TargetField", value)
```

**使用示例**:
```go
type UserInfo struct {
    Name string
    Age  int
}

wf := compose.NewWorkflow[UserInfo, string]()

// 节点 1：处理姓名
wf.AddLambdaNode("greet", greetLambda).
    AddInput(START, compose.MapFields("Name", "Name"))
// UserInfo.Name -> greetLambda 的 Name 参数

// 节点 2：处理年龄
wf.AddLambdaNode("category", categoryLambda).
    AddInput(START, compose.MapFields("Age", "Age"))
// UserInfo.Age -> categoryLambda 的 Age 参数

// 节点 3：合并结果
wf.AddLambdaNode("merge", mergeLambda).
    AddInput("greet", compose.MapFields("Greeting", "Greeting")).
    AddInput("category", compose.MapFields("Category", "Category"))
// greetLambda.Greeting -> mergeLambda.Greeting
// categoryLambda.Category -> mergeLambda.Category

// 输出
wf.End().AddInput("merge")
```

#### 4.3.2 复杂映射示例

```go
wf := compose.NewWorkflow[Input, Output]()

// 示例 1：从 START 映射多个字段
wf.AddLambdaNode("node1", lambda1).
    AddInput(START,
        compose.MapFields("Field1", "Param1"),
        compose.MapFields("Field2", "Param2"),
    )

// 示例 2：从不同节点映射
wf.AddLambdaNode("node3", lambda3).
    AddInput("node1", compose.MapFields("Output1", "Input1")).
    AddInput("node2", compose.MapFields("Output2", "Input2"))

// 示例 3：使用静态值
wf.AddLambdaNode("node4", lambda4).
    AddInput("node3", compose.MapFields("Data", "Data")).
    AddInput(START, compose.StaticValue("Config", myConfig))

// 示例 4：从 map 中取值
wf.AddLambdaNode("node5", lambda5).
    AddInput("parallel_node", compose.MapKey("result_a", "InputA"))
// parallel_node 输出 map，取其中 "result_a" 的值
```

---

### 4.4 Workflow 编译

```go
func (wf *Workflow[I, O]) Compile(ctx context.Context, opts ...GraphCompileOption) (Runnable[I, O], error)
```

**使用示例**:
```go
wf := compose.NewWorkflow[Input, Output]()

// 添加节点和依赖
// ...

// 编译
runnable, err := wf.Compile(ctx)
if err != nil {
    log.Fatal(err)
}

// 执行
output, err := runnable.Invoke(ctx, input)
```

---

## 5. Lambda API

### 5.1 InvokableLambda

**功能说明**: 创建支持 Invoke 模式的 Lambda。

**函数签名**:
```go
func InvokableLambda[I, O, TOption any](
    invoke Invoke[I, O, TOption],
) *Lambda
```

**Invoke 函数签名**:
```go
type Invoke[I, O, TOption any] func(ctx context.Context, input I, opts ...TOption) (O, error)
```

**使用示例**:
```go
// 示例 1：简单转换
toLowerLambda := compose.InvokableLambda(
    func(ctx context.Context, input string) (string, error) {
        return strings.ToLower(input), nil
    })

// 示例 2：复杂处理
processLambda := compose.InvokableLambda(
    func(ctx context.Context, docs []*schema.Document) (map[string]any, error) {
        var content strings.Builder
        for _, doc := range docs {
            content.WriteString(doc.Content)
            content.WriteString("\n\n")
        }
        
        return map[string]any{
            "context": content.String(),
            "count":   len(docs),
        }, nil
    })

// 示例 3：带 Option
type MyOption struct {
    Verbose bool
}

withOptionLambda := compose.InvokableLambda(
    func(ctx context.Context, input string, opts ...MyOption) (string, error) {
        verbose := false
        if len(opts) > 0 {
            verbose = opts[0].Verbose
        }
        
        if verbose {
            log.Printf("处理: %s", input)
        }
        
        return process(input), nil
    })
```

---

### 5.2 StreamableLambda

**功能说明**: 创建支持 Stream 模式的 Lambda。

**函数签名**:
```go
func StreamableLambda[I, O, TOption any](
    stream Stream[I, O, TOption],
) *Lambda
```

**Stream 函数签名**:
```go
type Stream[I, O, TOption any] func(ctx context.Context, input I, opts ...TOption) (*schema.StreamReader[O], error)
```

**使用示例**:
```go
// 示例 1：逐词输出
wordStreamLambda := compose.StreamableLambda(
    func(ctx context.Context, input string) (*schema.StreamReader[string], error) {
        sr, sw := schema.Pipe[string](10)
        
        go func() {
            defer sw.Close()
            words := strings.Fields(input)
            for _, word := range words {
                if sw.Send(word, nil) {
                    return  // 接收端已关闭
                }
                time.Sleep(100 * time.Millisecond)  // 模拟延迟
            }
        }()
        
        return sr, nil
    })

// 示例 2：批处理流
batchStreamLambda := compose.StreamableLambda(
    func(ctx context.Context, items []Item) (*schema.StreamReader[Result], error) {
        sr, sw := schema.Pipe[Result](10)
        
        go func() {
            defer sw.Close()
            for _, item := range items {
                result, err := processItem(item)
                if err != nil {
                    sw.Send(Result{}, err)
                    return
                }
                if sw.Send(result, nil) {
                    return
                }
            }
        }()
        
        return sr, nil
    })
```

---

### 5.3 CollectableLambda

**功能说明**: 创建支持 Collect 模式的 Lambda。

**函数签名**:
```go
func CollectableLambda[I, O, TOption any](
    collect Collect[I, O, TOption],
) *Lambda
```

**使用示例**:
```go
// 汇总流输入
collectLambda := compose.CollectableLambda(
    func(ctx context.Context, input *schema.StreamReader[string]) (string, error) {
        defer input.Close()
        
        var builder strings.Builder
        for {
            chunk, err := input.Recv()
            if err == io.EOF {
                break
            }
            if err != nil {
                return "", err
            }
            builder.WriteString(chunk)
        }
        
        return builder.String(), nil
    })
```

---

### 5.4 TransformableLambda

**功能说明**: 创建支持 Transform 模式的 Lambda。

**函数签名**:
```go
func TransformableLambda[I, O, TOption any](
    transform Transform[I, O, TOption],
) *Lambda
```

**使用示例**:
```go
// 流式转换
transformLambda := compose.TransformableLambda(
    func(ctx context.Context, input *schema.StreamReader[string]) (*schema.StreamReader[string], error) {
        return schema.StreamReaderWithConvert(input, func(s string) (string, error) {
            return strings.ToUpper(s), nil
        }), nil
    })
```

---

## 6. Option API

### 6.1 全局 Option

```go
// WithCallbacks: 设置回调处理器
compose.WithCallbacks(handler)

// WithRunTimeout: 设置运行超时
compose.WithRunTimeout(30 * time.Second)
```

### 6.2 组件类型 Option

```go
// WithChatModelOption: 为所有 ChatModel 节点设置 Option
compose.WithChatModelOption(
    model.WithTemperature(0.7),
    model.WithMaxTokens(1000),
)

// WithRetrieverOption: 为所有 Retriever 节点设置 Option
compose.WithRetrieverOption(
    retriever.WithTopK(5),
)
```

### 6.3 节点 Option

```go
// DesignateNode: 指定 Option 只应用于某个节点
compose.WithCallbacks(handler).DesignateNode("model")
compose.WithChatModelOption(opt).DesignateNode("specific_model")
```

---

**文档版本**: v1.0  
**最后更新**: 2024-12-19  
**适用 Eino 版本**: main 分支（最新版本）

