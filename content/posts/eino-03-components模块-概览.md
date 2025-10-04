---
title: "Eino-03-Components模块-概览"
date: 2025-10-04T20:42:31+08:00
draft: false
tags:
  - Eino
  - 架构设计
  - 概览
  - 源码分析
categories:
  - Eino
  - AI框架
  - Go
series: "eino-source-analysis"
description: "Eino 源码剖析 - Eino-03-Components模块-概览"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# Eino-03-Components模块-概览

Components 模块定义了 Eino 框架中所有可组合组件的标准接口，包括模型、工具、检索器、向量化、索引器、文档处理和提示词模板。

---

## 1. Components 模块设计理念

### 1.1 接口抽象

Eino 采用接口为先的设计，每种组件类型都定义了清晰的接口：

```
components/
  ├── model/        - 模型接口（ChatModel）
  ├── tool/         - 工具接口（Tool）
  ├── retriever/    - 检索器接口（Retriever）
  ├── embedding/    - 向量化接口（Embedder）
  ├── indexer/      - 索引器接口（Indexer）
  ├── document/     - 文档处理接口（Loader, Transformer）
  └── prompt/       - 提示词模板接口（ChatTemplate）
```

### 1.2 实现与框架分离

- **接口定义**: 在 `cloudwego/eino` 仓库
- **具体实现**: 在 `cloudwego/eino-ext` 仓库

这种设计使得：

- 框架核心保持轻量
- 实现可以独立演进
- 用户可以自定义实现

---

## 2. 七大组件类型

### 2.1 Model - 大语言模型

**接口定义**:

```go
// BaseChatModel 基础聊天模型接口
type BaseChatModel interface {
    // Generate 生成回复（非流式）
    Generate(ctx context.Context, input []*schema.Message, opts ...Option) (*schema.Message, error)
    
    // Stream 生成回复（流式）
    Stream(ctx context.Context, input []*schema.Message, opts ...Option) (*schema.StreamReader[*schema.Message], error)
}

// ToolCallingChatModel 支持工具调用的模型
type ToolCallingChatModel interface {
    BaseChatModel
    
    // BindTools 绑定工具（已废弃，使用 WithTools）
    BindTools(tools []*schema.ToolInfo) error
    
    // WithTools 返回绑定工具的新模型实例
    WithTools(tools []*schema.ToolInfo) (ToolCallingChatModel, error)
}
```

**典型实现**:

- OpenAI ChatGPT (gpt-4, gpt-3.5-turbo)
- OpenAI o1 (推理模型)
- Anthropic Claude
- Google Gemini
- 国产大模型（通义千问、文心一言等）

**使用示例**:

```go
// 创建模型
chatModel, err := openai.NewChatModel(ctx, &openai.ChatModelConfig{
    APIKey:     "your-api-key",
    Model:      "gpt-4",
})

// 基本对话
messages := []*schema.Message{
    schema.SystemMessage("你是一个有帮助的助手"),
    schema.UserMessage("什么是 Eino？"),
}

response, err := chatModel.Generate(ctx, messages)
fmt.Println(response.Content)

// 流式对话
stream, err := chatModel.Stream(ctx, messages)
defer stream.Close()

for {
    chunk, err := stream.Recv()
    if err == io.EOF {
        break
    }
    fmt.Print(chunk.Content)
}

// 工具调用
tools := []*schema.ToolInfo{weatherTool, calcTool}
modelWithTools, err := chatModel.WithTools(tools)

response, err := modelWithTools.Generate(ctx, messages)
if len(response.ToolCalls) > 0 {
    // 处理工具调用
}
```

---

### 2.2 Tool - 工具

**接口定义**:

```go
// Tool 工具接口
type Tool interface {
    // Info 返回工具元信息
    Info(ctx context.Context) (*schema.ToolInfo, error)
    
    // InvokableRun 执行工具（非流式）
    InvokableRun(ctx context.Context, argumentsInJSON string, opts ...Option) (string, error)
}
```

**ToolsNode - 工具节点**:

```go
// ToolsNode 工具节点，用于在 Graph 中执行工具
type ToolsNode struct {
    // ...
}

// NewToolsNode 创建工具节点
func NewToolsNode(opts ...ToolsNodeOption) *ToolsNode

// RegisterTool 注册工具
func (tn *ToolsNode) RegisterTool(
    toolInfo *schema.ToolInfo,
    toolFunc func(ctx context.Context, argumentsInJSON string) (string, error),
) error
```

**使用示例**:

```go
// 定义工具信息
weatherTool := &schema.ToolInfo{
    Name: "get_weather",
    Desc: "获取指定城市的天气信息",
    ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
        "city": {
            Type:     schema.String,
            Desc:     "城市名称",
            Required: true,
        },
    }),
}

// 创建工具节点
toolsNode := compose.NewToolsNode()

// 注册工具实现
toolsNode.RegisterTool(weatherTool, func(ctx context.Context, args string) (string, error) {
    var params struct {
        City string `json:"city"`
    }
    json.Unmarshal([]byte(args), &params)
    
    // 调用天气 API
    weather := getWeather(params.City)
    return weather, nil
})

// 在 Graph 中使用
graph.AddToolsNode("tools", toolsNode)
```

---

### 2.3 Retriever - 检索器

**接口定义**:

```go
// Retriever 检索器接口
type Retriever interface {
    // Retrieve 检索文档
    Retrieve(ctx context.Context, query string, opts ...Option) ([]*schema.Document, error)
}
```

**典型实现**:

- 向量检索（基于 Embedding）
- 关键词检索（BM25）
- 混合检索
- 多查询检索（MultiQuery）
- Parent Document Retriever
- Router Retriever

**使用示例**:

```go
// 创建向量检索器
vectorStore := createVectorStore()
retriever := vectorstore.NewRetriever(vectorStore,
    retriever.WithTopK(5),
    retriever.WithScoreThreshold(0.7),
)

// 检索文档
docs, err := retriever.Retrieve(ctx, "Eino 是什么？")
for _, doc := range docs {
    fmt.Printf("Score: %.2f, Content: %s\n", doc.Score(), doc.Content)
}

// 在 Graph 中使用
graph.AddRetrieverNode("retriever", retriever)
```

---

### 2.4 Embedding - 向量化

**接口定义**:

```go
// Embedder 向量化接口
type Embedder interface {
    // EmbedStrings 将多个字符串向量化
    EmbedStrings(ctx context.Context, texts []string, opts ...Option) ([][]float64, error)
}
```

**典型实现**:

- OpenAI Embedding (text-embedding-ada-002)
- HuggingFace Sentence Transformers
- 国产 Embedding 模型

**使用示例**:

```go
// 创建 Embedder
embedder, err := openai.NewEmbedder(ctx, &openai.EmbedderConfig{
    APIKey: "your-api-key",
    Model:  "text-embedding-ada-002",
})

// 向量化文本
texts := []string{"Eino 是什么？", "如何使用 Eino？"}
vectors, err := embedder.EmbedStrings(ctx, texts)

// vectors[0] 是第一个文本的向量
fmt.Printf("向量维度: %d\n", len(vectors[0]))

// 在 Graph 中使用
graph.AddEmbeddingNode("embedding", embedder)
```

---

### 2.5 Indexer - 索引器

**接口定义**:

```go
// Indexer 索引器接口
type Indexer interface {
    // Store 存储文档
    Store(ctx context.Context, docs []*schema.Document, opts ...Option) ([]string, error)
}
```

**典型实现**:

- 向量存储索引器
- Parent Document Indexer（存储父子文档）

**使用示例**:

```go
// 创建索引器
vectorStore := createVectorStore()
indexer := vectorstore.NewIndexer(vectorStore, embedder)

// 索引文档
docs := []*schema.Document{
    {ID: "doc1", Content: "Eino 是一个 Go 语言的 LLM 应用开发框架"},
    {ID: "doc2", Content: "Eino 支持 Chain、Graph、Workflow 三种编排模式"},
}

ids, err := indexer.Store(ctx, docs)
fmt.Printf("已索引 %d 个文档\n", len(ids))

// 在 Graph 中使用
graph.AddIndexerNode("indexer", indexer)
```

---

### 2.6 Document - 文档处理

#### 2.6.1 Loader - 文档加载器

**接口定义**:

```go
// Loader 文档加载器接口
type Loader interface {
    // Load 加载文档
    Load(ctx context.Context, src any, opts ...Option) ([]*schema.Document, error)
}
```

**典型实现**:

- 文件加载器（PDF、Word、Markdown）
- 网页加载器
- 数据库加载器
- API 加载器

**使用示例**:

```go
// 创建 PDF 加载器
loader := pdf.NewLoader()

// 加载 PDF 文件
docs, err := loader.Load(ctx, "document.pdf")
for _, doc := range docs {
    fmt.Println(doc.Content)
}

// 在 Graph 中使用
graph.AddLoaderNode("loader", loader)
```

#### 2.6.2 Transformer - 文档转换器

**接口定义**:

```go
// Transformer 文档转换器接口
type Transformer interface {
    // Transform 转换文档
    Transform(ctx context.Context, docs []*schema.Document, opts ...Option) ([]*schema.Document, error)
}
```

**典型实现**:

- 文本分割器（按字符、按 Token）
- 文档清洗器
- 元数据提取器

**使用示例**:

```go
// 创建文本分割器
splitter := textsplitter.NewRecursiveCharacterSplitter(
    textsplitter.WithChunkSize(1000),
    textsplitter.WithChunkOverlap(200),
)

// 分割文档
longDocs := loadLongDocuments()
chunks, err := splitter.Transform(ctx, longDocs)
fmt.Printf("分割为 %d 个块\n", len(chunks))

// 在 Graph 中使用
graph.AddDocumentTransformerNode("splitter", splitter)
```

---

### 2.7 Prompt - 提示词模板

**接口定义**:

```go
// ChatTemplate 聊天模板接口
type ChatTemplate interface {
    // Format 格式化模板
    Format(ctx context.Context, param map[string]any, opts ...Option) ([]*schema.Message, error)
}
```

**使用示例**:

```go
// 创建模板
template := prompt.FromMessages(
    schema.SystemMessage("你是一个{role}"),
    schema.UserMessage(`问题：{query}

上下文：
{context}`),
)

// 格式化
params := map[string]any{
    "role":    "专业的 Go 语言助手",
    "query":   "如何使用 Eino？",
    "context": contextDocs,
}

messages, err := template.Format(ctx, params)

// 在 Graph 中使用
graph.AddChatTemplateNode("template", template)
```

---

## 3. 组件实现规范

### 3.1 接口实现要求

1. **线程安全**: 所有方法必须是线程安全的
2. **上下文感知**: 支持 context 取消和超时
3. **错误处理**: 返回清晰的错误信息
4. **Option 模式**: 使用 Option 模式支持可选配置

### 3.2 Option 模式

```go
// 定义 Option 类型
type Option func(*config)

// 实现具体 Option
func WithTopK(k int) Option {
    return func(c *config) {
        c.topK = k
    }
}

// 在方法中使用
func (r *Retriever) Retrieve(ctx context.Context, query string, opts ...Option) ([]*schema.Document, error) {
    cfg := defaultConfig()
    for _, opt := range opts {
        opt(cfg)
    }
    
    // 使用 cfg.topK
}
```

### 3.3 Callbacks 集成

所有组件都应该支持 Callbacks：

```go
// 在 eino-ext 实现中集成 Callbacks
func (m *ChatModel) Generate(ctx context.Context, input []*schema.Message, opts ...Option) (*schema.Message, error) {
    // Callbacks OnStart
    defer func() {
        // Callbacks OnEnd
    }()
    
    // 实际处理
    return m.doGenerate(ctx, input)
}
```

---

## 4. 自定义组件实现

### 4.1 实现 ChatModel

```go
type MyChatModel struct {
    apiKey string
    model  string
}

func (m *MyChatModel) Generate(ctx context.Context, input []*schema.Message, opts ...Option) (*schema.Message, error) {
    // 1. 处理 Options
    cfg := &config{}
    for _, opt := range opts {
        opt(cfg)
    }
    
    // 2. 调用 API
    response, err := m.callAPI(ctx, input, cfg)
    if err != nil {
        return nil, err
    }
    
    // 3. 转换为 schema.Message
    return &schema.Message{
        Role:         schema.Assistant,
        Content:      response.Content,
        ResponseMeta: &schema.ResponseMeta{
            Usage: &schema.TokenUsage{
                TotalTokens: response.Usage.TotalTokens,
            },
        },
    }, nil
}

func (m *MyChatModel) Stream(ctx context.Context, input []*schema.Message, opts ...Option) (*schema.StreamReader[*schema.Message], error) {
    sr, sw := schema.Pipe[*schema.Message](10)
    
    go func() {
        defer sw.Close()
        
        // 流式调用 API
        stream, err := m.callStreamAPI(ctx, input)
        if err != nil {
            sw.Send(nil, err)
            return
        }
        
        for {
            chunk, err := stream.Recv()
            if err == io.EOF {
                break
            }
            if err != nil {
                sw.Send(nil, err)
                return
            }
            
            msg := &schema.Message{
                Role:    schema.Assistant,
                Content: chunk.Content,
            }
            sw.Send(msg, nil)
        }
    }()
    
    return sr, nil
}
```

### 4.2 实现 Retriever

```go
type MyRetriever struct {
    vectorStore VectorStore
    topK        int
}

func (r *MyRetriever) Retrieve(ctx context.Context, query string, opts ...Option) ([]*schema.Document, error) {
    // 1. 处理 Options
    cfg := &config{topK: r.topK}
    for _, opt := range opts {
        opt(cfg)
    }
    
    // 2. 向量化查询
    queryVector, err := r.embedQuery(ctx, query)
    if err != nil {
        return nil, err
    }
    
    // 3. 检索
    results, err := r.vectorStore.Search(ctx, queryVector, cfg.topK)
    if err != nil {
        return nil, err
    }
    
    // 4. 转换为 schema.Document
    docs := make([]*schema.Document, len(results))
    for i, result := range results {
        docs[i] = &schema.Document{
            ID:      result.ID,
            Content: result.Content,
        }
        docs[i].WithScore(result.Score)
    }
    
    return docs, nil
}
```

---

## 5. 组件组合模式

### 5.1 RAG 模式

```
Retriever -> DocumentTransformer -> ChatTemplate -> ChatModel
```

```go
graph := compose.NewGraph[string, *schema.Message]()

// 1. 检索
graph.AddRetrieverNode("retriever", retriever)

// 2. 文档处理（可选）
graph.AddDocumentTransformerNode("splitter", splitter)

// 3. 构建 Prompt
graph.AddChatTemplateNode("template", template)

// 4. 生成回复
graph.AddChatModelNode("model", chatModel)

// 连接
graph.AddEdge(START, "retriever")
graph.AddEdge("retriever", "template")
graph.AddEdge("template", "model")
graph.AddEdge("model", END)
```

### 5.2 Agent 模式

```
ChatModel <-> ToolsNode (循环)
```

```go
graph := compose.NewGraph[string, *schema.Message]()

// 1. 模型（带工具）
modelWithTools, _ := chatModel.WithTools(tools)
graph.AddChatModelNode("model", modelWithTools)

// 2. 工具节点
toolsNode := compose.NewToolsNode()
// 注册工具...
graph.AddToolsNode("tools", toolsNode)

// 3. 分支判断
graph.AddBranch("model", compose.NewGraphBranch(
    branchLambda,
    map[string]string{
        "tools": "tools",
        "end":   END,
    },
))

// 4. 循环
graph.AddEdge("tools", "model")
```

### 5.3 索引构建模式

```
Loader -> DocumentTransformer -> Embedding -> Indexer
```

```go
graph := compose.NewGraph[string, []string]()

// 1. 加载文档
graph.AddLoaderNode("loader", loader)

// 2. 分割
graph.AddDocumentTransformerNode("splitter", splitter)

// 3. 向量化（在 Indexer 内部）
graph.AddIndexerNode("indexer", indexer)

// 连接
graph.AddEdge(START, "loader")
graph.AddEdge("loader", "splitter")
graph.AddEdge("splitter", "indexer")
graph.AddEdge("indexer", END)
```

---

## 6. 最佳实践

### 6.1 组件选择

1. **ChatModel**: 根据任务选择合适的模型
   - 简单对话: gpt-3.5-turbo
   - 复杂推理: gpt-4, claude-3
   - 工具调用: 选择支持 function calling 的模型

2. **Retriever**: 根据数据特点选择
   - 语义检索: 向量检索
   - 关键词匹配: BM25
   - 混合检索: 结合两者

3. **Embedding**: 根据语言和任务选择
   - 英文: text-embedding-ada-002
   - 中文: 国产 Embedding 模型
   - 多语言: multilingual 模型

### 6.2 性能优化

1. **缓存**: 缓存 Embedding 结果
2. **批处理**: 批量调用 Embedding API
3. **并发**: 使用 Graph 并发执行独立节点
4. **流式**: 对 LLM 调用使用流式模式

### 6.3 错误处理

1. **重试**: 对临时性错误重试
2. **降级**: 主模型失败时使用备用模型
3. **超时**: 设置合理的超时时间
4. **日志**: 记录详细的错误信息

---

**文档版本**: v1.0  
**最后更新**: 2024-12-19  
**适用 Eino 版本**: main 分支（最新版本）
