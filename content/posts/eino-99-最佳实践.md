---
title: "Eino-99-最佳实践"
date: 2025-10-04T20:42:31+08:00
draft: false
tags:
  - Eino
  - 最佳实践
  - 实战经验
  - 源码分析
categories:
  - Eino
  - AI框架
  - Go
series: "eino-source-analysis"
description: "Eino 源码剖析 - Eino-99-最佳实践"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# Eino-99-最佳实践

本文档汇总 Eino 框架的最佳实践、实战经验和常见问题解决方案。

---

## 1. 框架使用最佳实践

### 1.1 编排模式选择

#### Chain vs Graph vs Workflow

```
┌─────────────────────┬──────────┬──────────┬───────────┐
│ 特性                │ Chain    │ Graph    │ Workflow  │
├─────────────────────┼──────────┼──────────┼───────────┤
│ 简单性              │ ★★★★★    │ ★★★      │ ★★★★      │
│ 灵活性              │ ★★       │ ★★★★★    │ ★★★★      │
│ 支持分支            │ ✗        │ ✓        │ ✓         │
│ 支持循环            │ ✗        │ ✓        │ ✗         │
│ 支持并发            │ ✗        │ ✓        │ ✓         │
│ 字段映射            │ ✗        │ ✗        │ ✓         │
│ 学习曲线            │ 低       │ 中       │ 中        │
└─────────────────────┴──────────┴──────────┴───────────┘
```

**选择建议**：

```go
// ✅ 简单顺序流程 → Chain
chain := compose.NewChain[Input, Output]()
chain.AppendChatTemplate("t", template)
chain.AppendChatModel("m", model)

// ✅ 需要条件分支或循环 → Graph
graph := compose.NewGraph[Input, Output]()
graph.AddBranch("node", branch)

// ✅ 需要字段级数据映射 → Workflow
wf := compose.NewWorkflow[Input, Output]()
wf.AddLambdaNode("node", lambda).
    AddInput("prev", compose.MapFields("Field1", "Field2"))
```

---

### 1.2 StreamReader 资源管理

#### ❌ 常见错误

```go
// 错误 1：忘记 Close
sr, sw := schema.Pipe[string](10)
for {
    chunk, err := sr.Recv()
    if err == io.EOF {
        break
    }
}
// 资源泄漏！

// 错误 2：Copy 后继续使用原流
readers := sr.Copy(2)
sr.Recv()  // ❌ 原流已失效

// 错误 3：多个 goroutine 同时 Recv
go func() { sr.Recv() }()
go func() { sr.Recv() }()  // ❌ 不是并发安全的
```

#### ✅ 正确用法

```go
// 正确 1：始终 Close
sr, sw := schema.Pipe[string](10)
defer sr.Close()
defer sw.Close()

// 正确 2：使用 ConcatMessageStream
stream, _ := chatModel.Stream(ctx, messages)
fullMsg, _ := schema.ConcatMessageStream(stream)  // 自动 Close

// 正确 3：Copy 后不再使用原流
readers := sr.Copy(2)
// sr 失效，使用 readers[0] 和 readers[1]

// 正确 4：每个 goroutine 使用独立的 reader
readers := sr.Copy(3)
for i, r := range readers {
    go func(reader *schema.StreamReader[string]) {
        defer reader.Close()
        // 独立处理
    }(r)
}
```

---

### 1.3 Message 最佳实践

#### 使用工厂函数

```go
// ✅ 推荐
msg := schema.UserMessage("Hello")
msg := schema.SystemMessage("You are an assistant")

// ❌ 不推荐
msg := &schema.Message{
    Role:    schema.User,
    Content: "Hello",
}
```

#### 处理多模态内容

```go
// ✅ 推荐：使用 MultiContent
msg := &schema.Message{
    Role: schema.User,
    MultiContent: []schema.ChatMessagePart{
        {Type: schema.ChatMessagePartTypeText, Text: "这是什么？"},
        {
            Type: schema.ChatMessagePartTypeImageURL,
            ImageURL: &schema.ChatMessageImageURL{
                URL: imageURL,
            },
        },
    },
}

// ❌ 避免：混用 Content 和 MultiContent
msg := &schema.Message{
    Role:         schema.User,
    Content:      "这是什么？",  // 会被忽略
    MultiContent: []schema.ChatMessagePart{...},
}
```

---

### 1.4 工具调用最佳实践

#### 工具定义

```go
// ✅ 好的工具定义：详细的描述和示例
weatherTool := &schema.ToolInfo{
    Name: "get_weather",
    Desc: `获取指定城市的当前天气信息。

使用场景：

- 用户询问某个城市的天气
- 用户询问是否需要带伞、穿什么衣服等

示例：

- "北京今天天气怎么样？" → get_weather(city="北京")
- "上海会下雨吗？" → get_weather(city="上海")

返回：天气状况、温度、湿度等信息`,
    ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
        "city": {
            Type:     schema.String,
            Desc:     "城市名称，如：北京、上海、广州",
            Required: true,
        },
    }),
}

// ❌ 不好的工具定义：描述模糊
weatherTool := &schema.ToolInfo{
    Name: "get_weather",
    Desc: "查天气",  // 太简单
}
```

#### 工具调用处理

```go
// ✅ 推荐：使用 ToolsNode
toolsNode := compose.NewToolsNode()
toolsNode.RegisterTool(weatherTool, func(ctx context.Context, args string) (string, error) {
    var params struct {
        City string `json:"city"`
    }
    json.Unmarshal([]byte(args), &params)
    
    // 执行工具逻辑
    weather := getWeatherFromAPI(params.City)
    return weather, nil
})

graph.AddToolsNode("tools", toolsNode)

// ❌ 避免：手动解析 ToolCalls
for _, toolCall := range msg.ToolCalls {
    switch toolCall.Function.Name {
    case "get_weather":
        // 手动处理
    }
}
```

---

## 2. 性能优化

### 2.1 减少内存分配

#### 复用对象

```go
// ✅ 使用对象池
var messagePool = sync.Pool{
    New: func() interface{} {
        return &schema.Message{}
    },
}

func getM

essage() *schema.Message {
    return messagePool.Get().(*schema.Message)
}

func putMessage(msg *schema.Message) {
    *msg = schema.Message{}  // 重置
    messagePool.Put(msg)
}
```

#### 预分配容量

```go
// ✅ 预分配 slice 容量
messages := make([]*schema.Message, 0, 100)

// ✅ 使用 strings.Builder
var sb strings.Builder
sb.Grow(estimatedSize)  // 预分配
for _, chunk := range chunks {
    sb.WriteString(chunk)
}
```

### 2.2 并发优化

#### Graph 并发执行

```go
// ✅ 利用 Graph 的自动并发
graph := compose.NewGraph[Input, Output]()

// 这两个节点会并发执行
graph.AddLambdaNode("node1", lambda1)
graph.AddLambdaNode("node2", lambda2)
graph.AddEdge(START, "node1")
graph.AddEdge(START, "node2")

// node3 等待 node1 和 node2 完成
graph.AddLambdaNode("node3", lambda3)
graph.AddEdge("node1", "node3")
graph.AddEdge("node2", "node3")
```

#### 控制并发数

```go
// ✅ 使用 goroutine pool 控制并发
sem := make(chan struct{}, 10)  // 最多 10 个并发

for _, item := range items {
    sem <- struct{}{}
    go func(item Item) {
        defer func() { <-sem }()
        process(item)
    }(item)
}
```

### 2.3 缓存策略

```go
// ✅ 缓存编译后的 Runnable
var runnableCache sync.Map

func getRunnable(key string) (compose.Runnable[I, O], error) {
    if r, ok := runnableCache.Load(key); ok {
        return r.(compose.Runnable[I, O]), nil
    }
    
    // 编译
    chain := buildChain()
    runnable, err := chain.Compile(ctx)
    if err != nil {
        return nil, err
    }
    
    runnableCache.Store(key, runnable)
    return runnable, nil
}

// ✅ 缓存 Embedding 结果
type EmbeddingCache struct {
    cache *lru.Cache
}

func (c *EmbeddingCache) GetEmbedding(text string) ([]float64, error) {
    if emb, ok := c.cache.Get(text); ok {
        return emb.([]float64), nil
    }
    
    emb, err := embedder.Embed(ctx, text)
    if err != nil {
        return nil, err
    }
    
    c.cache.Add(text, emb)
    return emb, nil
}
```

---

## 3. 错误处理和重试

### 3.1 错误处理模式

```go
// ✅ Lambda 中的错误处理
retryLambda := compose.InvokableLambda(
    func(ctx context.Context, input string) (string, error) {
        maxRetries := 3
        var lastErr error
        
        for i := 0; i < maxRetries; i++ {
            result, err := callExternalAPI(ctx, input)
            if err == nil {
                return result, nil
            }
            
            lastErr = err
            
            // 指数退避
            if i < maxRetries-1 {
                time.Sleep(time.Second * time.Duration(1<<i))
            }
        }
        
        return "", fmt.Errorf("failed after %d retries: %w", maxRetries, lastErr)
    })
```

### 3.2 优雅降级

```go
// ✅ Graph 分支实现降级
graph := compose.NewGraph[Input, Output]()

// 主路径
graph.AddChatModelNode("model", primaryModel)

// 降级判断
fallbackLambda := compose.InvokableLambda(
    func(ctx context.Context, result *schema.Message) (string, error) {
        if result.ResponseMeta != nil &&
           result.ResponseMeta.FinishReason == "error" {
            return "fallback", nil
        }
        return "success", nil
    })

graph.AddBranch("model", compose.NewGraphBranch(
    fallbackLambda,
    map[string]string{
        "success":  END,
        "fallback": "fallback_model",
    },
))

// 降级模型
graph.AddChatModelNode("fallback_model", fallbackModel)
graph.AddEdge("fallback_model", END)
```

---

## 4. 生产环境实践

### 4.1 监控和告警

```go
// ✅ 统一监控 Handler
monitorHandler := callbacks.NewHandlerBuilder().
    OnStartFn(func(ctx context.Context, info *callbacks.RunInfo, input callbacks.CallbackInput) context.Context {
        // 记录开始时间
        return context.WithValue(ctx, "start_time", time.Now())
    }).
    OnEndFn(func(ctx context.Context, info *callbacks.RunInfo, output callbacks.CallbackOutput) context.Context {
        // 计算耗时
        start := ctx.Value("start_time").(time.Time)
        duration := time.Since(start)
        
        // 记录指标
        prometheus.HistogramObserve("eino_node_duration_seconds",
            duration.Seconds(),
            "component", info.Component,
            "node", info.NodeKey,
        )
        
        // 记录 Token 使用
        if msg, ok := output.Output.(*schema.Message); ok {
            if msg.ResponseMeta != nil && msg.ResponseMeta.Usage != nil {
                prometheus.CounterAdd("eino_tokens_total",
                    float64(msg.ResponseMeta.Usage.TotalTokens),
                    "node", info.NodeKey,
                )
            }
        }
        
        return ctx
    }).
    OnErrorFn(func(ctx context.Context, info *callbacks.RunInfo, err error) context.Context {
        // 记录错误
        prometheus.CounterInc("eino_errors_total",
            "component", info.Component,
            "node", info.NodeKey,
            "error", err.Error(),
        )
        
        // 告警
        if isNonRetryableError(err) {
            alerting.Send("Eino node failed", err)
        }
        
        return ctx
    }).
    Build()

// 全局注入
callbacks.AppendGlobalHandlers(monitorHandler)
```

### 4.2 日志和追踪

```go
// ✅ 结构化日志
loggingHandler := callbacks.NewHandlerBuilder().
    OnStartFn(func(ctx context.Context, info *callbacks.RunInfo, input callbacks.CallbackInput) context.Context {
        log.WithFields(logrus.Fields{
            "trace_id":  getTraceID(ctx),
            "component": info.Component,
            "node":      info.NodeKey,
            "input":     summarizeInput(input),
        }).Info("Node started")
        return ctx
    }).
    Build()

// ✅ 分布式追踪
tracingHandler := callbacks.NewHandlerBuilder().
    OnStartFn(func(ctx context.Context, info *callbacks.RunInfo, input callbacks.CallbackInput) context.Context {
        span, ctx := opentracing.StartSpanFromContext(ctx,
            fmt.Sprintf("%s.%s", info.Component, info.NodeKey),
        )
        span.SetTag("component", info.Component)
        span.SetTag("node", info.NodeKey)
        return context.WithValue(ctx, "span", span)
    }).
    OnEndFn(func(ctx context.Context, info *callbacks.RunInfo, output callbacks.CallbackOutput) context.Context {
        if span := ctx.Value("span"); span != nil {
            span.(opentracing.Span).Finish()
        }
        return ctx
    }).
    Build()
```

### 4.3 限流和熔断

```go
// ✅ 限流器
type RateLimitedModel struct {
    model   model.BaseChatModel
    limiter *rate.Limiter
}

func (m *RateLimitedModel) Generate(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error) {
    // 等待令牌
    if err := m.limiter.Wait(ctx); err != nil {
        return nil, err
    }
    
    return m.model.Generate(ctx, input, opts...)
}

// ✅ 熔断器
type CircuitBreakerModel struct {
    model   model.BaseChatModel
    breaker *gobreaker.CircuitBreaker
}

func (m *CircuitBreakerModel) Generate(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error) {
    result, err := m.breaker.Execute(func() (interface{}, error) {
        return m.model.Generate(ctx, input, opts...)
    })
    
    if err != nil {
        return nil, err
    }
    
    return result.(*schema.Message), nil
}
```

---

## 5. 实战案例

### 5.1 案例 1：构建完整的 RAG 应用

```go
func BuildRAGApplication(
    retriever retriever.Retriever,
    chatModel model.BaseChatModel,
) (compose.Runnable[string, string], error) {
    
    graph := compose.NewGraph[string, string]()
    
    // 1. 检索节点
    graph.AddRetrieverNode("retriever", retriever)
    
    // 2. 文档处理节点
    docLambda := compose.InvokableLambda(
        func(ctx context.Context, docs []*schema.Document) (map[string]any, error) {
            var context strings.Builder
            for i, doc := range docs {
                if i > 0 {
                    context.WriteString("\n\n")
                }
                context.WriteString(fmt.Sprintf("文档%d: %s", i+1, doc.Content))
            }
            
            // 传递到下一个节点
            query := ctx.Value("original_query").(string)
            return map[string]any{
                "context": context.String(),
                "query":   query,
            }, nil
        })
    graph.AddLambdaNode("doc_processor", docLambda)
    
    // 3. 模板节点
    template := prompt.FromMessages(
        schema.SystemMessage("你是一个有帮助的AI助手。请根据提供的上下文回答问题。"),
        schema.UserMessage(`上下文：
{context}

问题：{query}

请根据上下文回答问题。如果上下文中没有相关信息，请明确说明。`),
    )
    graph.AddChatTemplateNode("template", template)
    
    // 4. 模型节点
    graph.AddChatModelNode("model", chatModel)
    
    // 5. 提取答案节点
    extractLambda := compose.InvokableLambda(
        func(ctx context.Context, msg *schema.Message) (string, error) {
            return msg.Content, nil
        })
    graph.AddLambdaNode("extract", extractLambda)
    
    // 6. 连接节点
    graph.AddEdge(START, "retriever")
    graph.AddEdge("retriever", "doc_processor")
    graph.AddEdge("doc_processor", "template")
    graph.AddEdge("template", "model")
    graph.AddEdge("model", "extract")
    graph.AddEdge("extract", END)
    
    // 7. 编译
    return graph.Compile(context.Background())
}

// 使用
runnable, _ := BuildRAGApplication(myRetriever, myChatModel)
answer, _ := runnable.Invoke(ctx, "Eino 是什么？",
    compose.WithCallbacks(monitorHandler))
```

### 5.2 案例 2：多轮对话管理

```go
type ConversationManager struct {
    runnable compose.Runnable[*ConversationInput, *schema.Message]
    history  []*schema.Message
    mu       sync.Mutex
}

type ConversationInput struct {
    Query   string
    History []*schema.Message
}

func NewConversationManager(chatModel model.BaseChatModel) (*ConversationManager, error) {
    chain := compose.NewChain[*ConversationInput, *schema.Message]()
    
    // 构建消息
    buildMsgLambda := compose.InvokableLambda(
        func(ctx context.Context, input *ConversationInput) ([]*schema.Message, error) {
            messages := make([]*schema.Message, 0, len(input.History)+1)
            messages = append(messages, input.History...)
            messages = append(messages, schema.UserMessage(input.Query))
            return messages, nil
        })
    
    chain.AppendLambda("build_messages", buildMsgLambda)
    chain.AppendChatModel("model", chatModel)
    
    runnable, err := chain.Compile(context.Background())
    if err != nil {
        return nil, err
    }
    
    return &ConversationManager{
        runnable: runnable,
        history:  []*schema.Message{},
    }, nil
}

func (cm *ConversationManager) Chat(ctx context.Context, query string) (*schema.Message, error) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    input := &ConversationInput{
        Query:   query,
        History: cm.history,
    }
    
    response, err := cm.runnable.Invoke(ctx, input)
    if err != nil {
        return nil, err
    }
    
    // 更新历史
    cm.history = append(cm.history, schema.UserMessage(query))
    cm.history = append(cm.history, response)
    
    // 限制历史长度
    if len(cm.history) > 20 {
        cm.history = cm.history[len(cm.history)-20:]
    }
    
    return response, nil
}
```

---

## 6. 常见问题

### Q1: 如何处理超时？

```go
// ✅ 使用 context 超时
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

result, err := runnable.Invoke(ctx, input)
if errors.Is(err, context.DeadlineExceeded) {
    // 处理超时
}
```

### Q2: 如何调试 Graph 执行？

```go
// ✅ 使用 Callbacks 打印日志
debugHandler := callbacks.NewHandlerBuilder().
    OnStartFn(func(ctx context.Context, info *callbacks.RunInfo, input callbacks.CallbackInput) context.Context {
        fmt.Printf("[%s] 开始执行\n", info.NodeKey)
        return ctx
    }).
    OnEndFn(func(ctx context.Context, info *callbacks.RunInfo, output callbacks.CallbackOutput) context.Context {
        fmt.Printf("[%s] 执行完成\n", info.NodeKey)
        return ctx
    }).
    Build()

runnable.Invoke(ctx, input, compose.WithCallbacks(debugHandler))
```

### Q3: 如何处理大文本？

```go
// ✅ 拆分为多个 chunk
func splitText(text string, chunkSize int) []string {
    var chunks []string
    for i := 0; i < len(text); i += chunkSize {
        end := i + chunkSize
        if end > len(text) {
            end = len(text)
        }
        chunks = append(chunks, text[i:end])
    }
    return chunks
}

// ✅ 使用流式处理
sr, sw := schema.Pipe[string](10)
go func() {
    defer sw.Close()
    chunks := splitText(largeText, 1000)
    for _, chunk := range chunks {
        sw.Send(chunk, nil)
    }
}()
```

---

**文档版本**: v1.0  
**最后更新**: 2024-12-19
