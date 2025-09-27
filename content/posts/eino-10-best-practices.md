# Eino 实战经验与最佳实践

## 📖 文档概述

本文档汇总了 Eino 框架在生产环境中的实战经验、性能优化技巧、最佳实践模式和常见问题解决方案，帮助开发者构建高性能、可靠的 LLM 应用。

## 🚀 快速上手最佳实践

### 项目结构组织

```
your-eino-project/
├── cmd/                    # 应用入口
│   └── server/
│       └── main.go
├── internal/               # 内部包
│   ├── config/            # 配置管理
│   ├── handlers/          # 业务处理器
│   ├── middleware/        # 中间件
│   └── models/           # 数据模型
├── pkg/                   # 可复用包
│   ├── chains/           # 预定义链
│   ├── components/       # 自定义组件
│   └── utils/           # 工具函数
├── configs/              # 配置文件
├── deployments/          # 部署配置
├── docs/                # 文档
├── examples/            # 示例代码
├── scripts/             # 脚本
├── tests/               # 测试
├── go.mod
├── go.sum
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### 依赖管理最佳实践

```go
// go.mod 示例
module your-company/your-eino-app

go 1.21

require (
    github.com/cloudwego/eino v0.1.0
    github.com/cloudwego/eino-ext v0.1.0
    
    // 基础依赖
    github.com/gin-gonic/gin v1.9.1
    github.com/spf13/viper v1.16.0
    github.com/sirupsen/logrus v1.9.3
    
    // 数据库
    gorm.io/gorm v1.25.4
    gorm.io/driver/postgres v1.5.2
    
    // 缓存
    github.com/redis/go-redis/v9 v9.1.0
    
    // 监控
    github.com/prometheus/client_golang v1.16.0
    go.opentelemetry.io/otel v1.16.0
)
```

## 🏗️ 架构设计最佳实践

### 1. 分层架构设计

```mermaid
graph TB
    subgraph "表现层 (Presentation Layer)"
        API[REST API]
        WS[WebSocket]
        CLI[CLI Interface]
    end
    
    subgraph "业务层 (Business Layer)"
        SVC[Service Layer]
        CHAIN[Chain Orchestration]
        WORKFLOW[Workflow Management]
    end
    
    subgraph "组件层 (Component Layer)"
        MODEL[Model Components]
        TOOL[Tool Components]
        RETRIEVER[Retriever Components]
    end
    
    subgraph "基础设施层 (Infrastructure Layer)"
        DB[Database]
        CACHE[Cache]
        MQ[Message Queue]
        STORAGE[File Storage]
    end
    
    API --> SVC
    WS --> SVC
    CLI --> SVC
    
    SVC --> CHAIN
    SVC --> WORKFLOW
    
    CHAIN --> MODEL
    CHAIN --> TOOL
    CHAIN --> RETRIEVER
    
    MODEL --> DB
    TOOL --> CACHE
    RETRIEVER --> STORAGE
    
    style API fill:#e8f5e8
    style SVC fill:#fff3e0
    style MODEL fill:#f3e5f5
    style DB fill:#e3f2fd
```

### 2. 服务层设计模式

```go
// service/chat_service.go
package service

import (
    "context"
    "fmt"
    "time"
    
    "github.com/cloudwego/eino/compose"
    "github.com/cloudwego/eino/schema"
)

// ChatService 聊天服务接口
type ChatService interface {
    // Chat 处理单轮对话
    Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
    
    // StreamChat 处理流式对话
    StreamChat(ctx context.Context, req *ChatRequest) (<-chan *ChatChunk, error)
    
    // MultiTurnChat 处理多轮对话
    MultiTurnChat(ctx context.Context, req *MultiTurnChatRequest) (*ChatResponse, error)
}

// ChatRequest 聊天请求
type ChatRequest struct {
    UserID    string            `json:"user_id"`
    SessionID string            `json:"session_id"`
    Message   string            `json:"message"`
    Context   map[string]any    `json:"context,omitempty"`
    Options   *ChatOptions      `json:"options,omitempty"`
}

// ChatResponse 聊天响应
type ChatResponse struct {
    MessageID   string            `json:"message_id"`
    Content     string            `json:"content"`
    TokenUsage  *TokenUsage       `json:"token_usage,omitempty"`
    Metadata    map[string]any    `json:"metadata,omitempty"`
    ProcessTime time.Duration     `json:"process_time"`
}

// chatServiceImpl 聊天服务实现
type chatServiceImpl struct {
    chatChain    compose.Runnable[*ChatInput, *schema.Message]
    ragChain     compose.Runnable[*RAGInput, *schema.Message]
    toolChain    compose.Runnable[*ToolInput, *schema.Message]
    
    sessionRepo  SessionRepository
    messageRepo  MessageRepository
    
    logger       Logger
    metrics      Metrics
}

func NewChatService(
    chatChain compose.Runnable[*ChatInput, *schema.Message],
    ragChain compose.Runnable[*RAGInput, *schema.Message],
    toolChain compose.Runnable[*ToolInput, *schema.Message],
    sessionRepo SessionRepository,
    messageRepo MessageRepository,
    logger Logger,
    metrics Metrics,
) ChatService {
    return &chatServiceImpl{
        chatChain:   chatChain,
        ragChain:    ragChain,
        toolChain:   toolChain,
        sessionRepo: sessionRepo,
        messageRepo: messageRepo,
        logger:      logger,
        metrics:     metrics,
    }
}

func (s *chatServiceImpl) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
    startTime := time.Now()
    
    // 记录请求指标
    s.metrics.IncChatRequests(req.UserID)
    
    // 构建输入
    input := &ChatInput{
        UserID:    req.UserID,
        SessionID: req.SessionID,
        Message:   req.Message,
        Context:   req.Context,
    }
    
    // 选择合适的链
    chain, err := s.selectChain(ctx, req)
    if err != nil {
        s.metrics.IncChatErrors("chain_selection_error")
        return nil, fmt.Errorf("failed to select chain: %w", err)
    }
    
    // 执行链
    result, err := chain.Invoke(ctx, input)
    if err != nil {
        s.metrics.IncChatErrors("chain_execution_error")
        s.logger.Error("Chain execution failed", "error", err, "user_id", req.UserID)
        return nil, fmt.Errorf("failed to execute chain: %w", err)
    }
    
    // 保存消息历史
    if err := s.saveMessageHistory(ctx, req, result); err != nil {
        s.logger.Warn("Failed to save message history", "error", err)
    }
    
    processTime := time.Since(startTime)
    s.metrics.ObserveChatLatency(processTime)
    
    return &ChatResponse{
        MessageID:   generateMessageID(),
        Content:     result.Content,
        TokenUsage:  extractTokenUsage(result),
        Metadata:    extractMetadata(result),
        ProcessTime: processTime,
    }, nil
}

func (s *chatServiceImpl) selectChain(ctx context.Context, req *ChatRequest) (compose.Runnable[any, *schema.Message], error) {
    // 根据请求特征选择合适的链
    if s.needsRAG(req) {
        return s.ragChain, nil
    }
    
    if s.needsTools(req) {
        return s.toolChain, nil
    }
    
    return s.chatChain, nil
}
```

### 3. 配置管理最佳实践

```go
// config/config.go
package config

import (
    "fmt"
    "time"
    
    "github.com/spf13/viper"
)

// Config 应用配置
type Config struct {
    Server   ServerConfig   `mapstructure:"server"`
    Database DatabaseConfig `mapstructure:"database"`
    Redis    RedisConfig    `mapstructure:"redis"`
    LLM      LLMConfig      `mapstructure:"llm"`
    RAG      RAGConfig      `mapstructure:"rag"`
    Logging  LoggingConfig  `mapstructure:"logging"`
    Metrics  MetricsConfig  `mapstructure:"metrics"`
}

// ServerConfig 服务器配置
type ServerConfig struct {
    Host         string        `mapstructure:"host"`
    Port         int           `mapstructure:"port"`
    ReadTimeout  time.Duration `mapstructure:"read_timeout"`
    WriteTimeout time.Duration `mapstructure:"write_timeout"`
    IdleTimeout  time.Duration `mapstructure:"idle_timeout"`
}

// LLMConfig 大语言模型配置
type LLMConfig struct {
    Provider    string            `mapstructure:"provider"`
    Model       string            `mapstructure:"model"`
    APIKey      string            `mapstructure:"api_key"`
    BaseURL     string            `mapstructure:"base_url"`
    MaxTokens   int               `mapstructure:"max_tokens"`
    Temperature float64           `mapstructure:"temperature"`
    Timeout     time.Duration     `mapstructure:"timeout"`
    RetryCount  int               `mapstructure:"retry_count"`
    Extra       map[string]any    `mapstructure:"extra"`
}

// RAGConfig RAG配置
type RAGConfig struct {
    VectorStore VectorStoreConfig `mapstructure:"vector_store"`
    Embedding   EmbeddingConfig   `mapstructure:"embedding"`
    Retrieval   RetrievalConfig   `mapstructure:"retrieval"`
}

// LoadConfig 加载配置
func LoadConfig(configPath string) (*Config, error) {
    viper.SetConfigFile(configPath)
    viper.SetConfigType("yaml")
    
    // 设置默认值
    setDefaults()
    
    // 读取环境变量
    viper.AutomaticEnv()
    
    // 读取配置文件
    if err := viper.ReadInConfig(); err != nil {
        return nil, fmt.Errorf("failed to read config file: %w", err)
    }
    
    var config Config
    if err := viper.Unmarshal(&config); err != nil {
        return nil, fmt.Errorf("failed to unmarshal config: %w", err)
    }
    
    // 验证配置
    if err := validateConfig(&config); err != nil {
        return nil, fmt.Errorf("invalid config: %w", err)
    }
    
    return &config, nil
}

func setDefaults() {
    // 服务器默认配置
    viper.SetDefault("server.host", "0.0.0.0")
    viper.SetDefault("server.port", 8080)
    viper.SetDefault("server.read_timeout", "30s")
    viper.SetDefault("server.write_timeout", "30s")
    viper.SetDefault("server.idle_timeout", "120s")
    
    // LLM默认配置
    viper.SetDefault("llm.max_tokens", 4096)
    viper.SetDefault("llm.temperature", 0.7)
    viper.SetDefault("llm.timeout", "60s")
    viper.SetDefault("llm.retry_count", 3)
    
    // RAG默认配置
    viper.SetDefault("rag.retrieval.top_k", 5)
    viper.SetDefault("rag.retrieval.score_threshold", 0.7)
}

func validateConfig(config *Config) error {
    if config.LLM.APIKey == "" {
        return fmt.Errorf("LLM API key is required")
    }
    
    if config.LLM.MaxTokens <= 0 {
        return fmt.Errorf("LLM max tokens must be positive")
    }
    
    if config.RAG.Retrieval.TopK <= 0 {
        return fmt.Errorf("RAG retrieval top_k must be positive")
    }
    
    return nil
}
```

## 🔧 组件开发最佳实践

### 1. 自定义组件开发

```go
// components/custom_retriever.go
package components

import (
    "context"
    "fmt"
    "sort"
    
    "github.com/cloudwego/eino/components/retriever"
    "github.com/cloudwego/eino/schema"
)

// HybridRetriever 混合检索器，结合向量检索和关键词检索
type HybridRetriever struct {
    vectorRetriever   retriever.Retriever
    keywordRetriever  retriever.Retriever
    
    vectorWeight      float64
    keywordWeight     float64
    
    logger           Logger
    metrics          Metrics
}

// HybridRetrieverConfig 混合检索器配置
type HybridRetrieverConfig struct {
    VectorRetriever   retriever.Retriever `validate:"required"`
    KeywordRetriever  retriever.Retriever `validate:"required"`
    VectorWeight      float64             `validate:"min=0,max=1"`
    KeywordWeight     float64             `validate:"min=0,max=1"`
    Logger           Logger
    Metrics          Metrics
}

func NewHybridRetriever(ctx context.Context, config *HybridRetrieverConfig) (*HybridRetriever, error) {
    if err := validateConfig(config); err != nil {
        return nil, fmt.Errorf("invalid config: %w", err)
    }
    
    return &HybridRetriever{
        vectorRetriever:  config.VectorRetriever,
        keywordRetriever: config.KeywordRetriever,
        vectorWeight:     config.VectorWeight,
        keywordWeight:    config.KeywordWeight,
        logger:          config.Logger,
        metrics:         config.Metrics,
    }, nil
}

// Retrieve 实现 retriever.Retriever 接口
func (hr *HybridRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
    startTime := time.Now()
    defer func() {
        hr.metrics.ObserveRetrievalLatency(time.Since(startTime))
    }()
    
    // 解析选项
    options := retriever.GetOptions(opts...)
    
    // 并行执行两种检索
    vectorCh := make(chan retrievalResult, 1)
    keywordCh := make(chan retrievalResult, 1)
    
    // 向量检索
    go func() {
        docs, err := hr.vectorRetriever.Retrieve(ctx, query, opts...)
        vectorCh <- retrievalResult{docs: docs, err: err}
    }()
    
    // 关键词检索
    go func() {
        docs, err := hr.keywordRetriever.Retrieve(ctx, query, opts...)
        keywordCh <- retrievalResult{docs: docs, err: err}
    }()
    
    // 收集结果
    vectorResult := <-vectorCh
    keywordResult := <-keywordCh
    
    // 检查错误
    if vectorResult.err != nil {
        hr.logger.Error("Vector retrieval failed", "error", vectorResult.err)
        hr.metrics.IncRetrievalErrors("vector")
    }
    
    if keywordResult.err != nil {
        hr.logger.Error("Keyword retrieval failed", "error", keywordResult.err)
        hr.metrics.IncRetrievalErrors("keyword")
    }
    
    // 如果两个都失败，返回错误
    if vectorResult.err != nil && keywordResult.err != nil {
        return nil, fmt.Errorf("both retrievals failed: vector=%v, keyword=%v", 
            vectorResult.err, keywordResult.err)
    }
    
    // 合并和重排序结果
    mergedDocs := hr.mergeAndRerank(vectorResult.docs, keywordResult.docs)
    
    // 应用 top_k 限制
    if options.TopK > 0 && len(mergedDocs) > options.TopK {
        mergedDocs = mergedDocs[:options.TopK]
    }
    
    hr.metrics.ObserveRetrievalCount(len(mergedDocs))
    
    return mergedDocs, nil
}

type retrievalResult struct {
    docs []*schema.Document
    err  error
}

// mergeAndRerank 合并和重排序文档
func (hr *HybridRetriever) mergeAndRerank(vectorDocs, keywordDocs []*schema.Document) []*schema.Document {
    // 创建文档映射，避免重复
    docMap := make(map[string]*schema.Document)
    
    // 处理向量检索结果
    for i, doc := range vectorDocs {
        if doc == nil {
            continue
        }
        
        // 计算向量检索分数（基于排名）
        vectorScore := 1.0 - float64(i)/float64(len(vectorDocs))
        
        if existingDoc, exists := docMap[doc.ID]; exists {
            // 文档已存在，更新分数
            existingScore := existingDoc.Score()
            newScore := existingScore + hr.vectorWeight*vectorScore
            existingDoc.WithScore(newScore)
        } else {
            // 新文档
            newDoc := *doc
            newDoc.WithScore(hr.vectorWeight * vectorScore)
            docMap[doc.ID] = &newDoc
        }
    }
    
    // 处理关键词检索结果
    for i, doc := range keywordDocs {
        if doc == nil {
            continue
        }
        
        // 计算关键词检索分数（基于排名）
        keywordScore := 1.0 - float64(i)/float64(len(keywordDocs))
        
        if existingDoc, exists := docMap[doc.ID]; exists {
            // 文档已存在，更新分数
            existingScore := existingDoc.Score()
            newScore := existingScore + hr.keywordWeight*keywordScore
            existingDoc.WithScore(newScore)
        } else {
            // 新文档
            newDoc := *doc
            newDoc.WithScore(hr.keywordWeight * keywordScore)
            docMap[doc.ID] = &newDoc
        }
    }
    
    // 转换为切片并按分数排序
    result := make([]*schema.Document, 0, len(docMap))
    for _, doc := range docMap {
        result = append(result, doc)
    }
    
    sort.Slice(result, func(i, j int) bool {
        return result[i].Score() > result[j].Score()
    })
    
    return result
}

// GetType 返回组件类型
func (hr *HybridRetriever) GetType() string {
    return "HybridRetriever"
}
```

### 2. Lambda 函数最佳实践

```go
// chains/processors.go
package chains

import (
    "context"
    "fmt"
    "strings"
    "time"
    
    "github.com/cloudwego/eino/compose"
    "github.com/cloudwego/eino/schema"
)

// CreateQueryProcessor 创建查询预处理器
func CreateQueryProcessor(config *ProcessorConfig) *compose.Lambda {
    return compose.InvokableLambda(func(ctx context.Context, input string) (string, error) {
        startTime := time.Now()
        defer func() {
            config.Metrics.ObserveProcessingLatency("query_processor", time.Since(startTime))
        }()
        
        // 查询清理
        cleaned := strings.TrimSpace(input)
        if cleaned == "" {
            return "", fmt.Errorf("empty query")
        }
        
        // 查询增强
        enhanced := enhanceQuery(cleaned, config)
        
        config.Logger.Debug("Query processed", 
            "original", input, 
            "enhanced", enhanced)
        
        return enhanced, nil
    }, compose.WithLambdaType("QueryProcessor"))
}

// CreateResponseFormatter 创建响应格式化器
func CreateResponseFormatter(config *ProcessorConfig) *compose.Lambda {
    return compose.InvokableLambda(func(ctx context.Context, input *schema.Message) (*schema.Message, error) {
        startTime := time.Now()
        defer func() {
            config.Metrics.ObserveProcessingLatency("response_formatter", time.Since(startTime))
        }()
        
        // 格式化响应内容
        formatted := formatResponse(input.Content, config)
        
        // 创建新的消息
        result := &schema.Message{
            Role:    input.Role,
            Content: formatted,
            Extra: map[string]any{
                "formatted_at": time.Now(),
                "formatter":    "ResponseFormatter",
            },
        }
        
        // 复制其他字段
        if input.ResponseMeta != nil {
            result.ResponseMeta = input.ResponseMeta
        }
        
        return result, nil
    }, compose.WithLambdaType("ResponseFormatter"))
}

// CreateStreamingProcessor 创建流式处理器
func CreateStreamingProcessor(config *ProcessorConfig) *compose.Lambda {
    return compose.StreamableLambda(func(ctx context.Context, input string) (*schema.StreamReader[string], error) {
        // 创建流式输出
        sr, sw := schema.Pipe[string](10)
        
        go func() {
            defer sw.Close()
            
            // 模拟流式处理
            words := strings.Fields(input)
            for i, word := range words {
                select {
                case <-ctx.Done():
                    return
                default:
                }
                
                // 处理单词
                processed := processWord(word, config)
                
                if sw.Send(processed, nil) {
                    return // 流已关闭
                }
                
                // 模拟处理延迟
                if i < len(words)-1 {
                    time.Sleep(10 * time.Millisecond)
                }
            }
        }()
        
        return sr, nil
    }, compose.WithLambdaType("StreamingProcessor"))
}

// CreateBatchProcessor 创建批处理器
func CreateBatchProcessor(config *ProcessorConfig) *compose.Lambda {
    return compose.CollectableLambda(func(ctx context.Context, input *schema.StreamReader[string]) (string, error) {
        var items []string
        
        // 收集所有流数据
        for {
            item, err := input.Recv()
            if err == io.EOF {
                break
            }
            if err != nil {
                return "", fmt.Errorf("failed to receive stream item: %w", err)
            }
            
            items = append(items, item)
        }
        
        // 批量处理
        result := processBatch(items, config)
        
        return result, nil
    }, compose.WithLambdaType("BatchProcessor"))
}

// 辅助函数
func enhanceQuery(query string, config *ProcessorConfig) string {
    // 实现查询增强逻辑
    enhanced := query
    
    // 添加上下文信息
    if config.AddContext {
        enhanced = fmt.Sprintf("Context: %s\nQuery: %s", config.ContextInfo, enhanced)
    }
    
    // 添加指令
    if config.AddInstructions {
        enhanced = fmt.Sprintf("%s\nInstructions: %s", enhanced, config.Instructions)
    }
    
    return enhanced
}

func formatResponse(content string, config *ProcessorConfig) string {
    // 实现响应格式化逻辑
    formatted := content
    
    // 添加格式化标记
    if config.AddMarkdown {
        formatted = addMarkdownFormatting(formatted)
    }
    
    // 添加元信息
    if config.AddMetaInfo {
        formatted = fmt.Sprintf("%s\n\n---\n*Generated at: %s*", 
            formatted, time.Now().Format("2006-01-02 15:04:05"))
    }
    
    return formatted
}
```

### 3. 错误处理最佳实践

```go
// errors/errors.go
package errors

import (
    "fmt"
    "net/http"
)

// 定义错误类型
type ErrorType string

const (
    ErrorTypeValidation   ErrorType = "validation"
    ErrorTypeNotFound     ErrorType = "not_found"
    ErrorTypeUnauthorized ErrorType = "unauthorized"
    ErrorTypeRateLimit    ErrorType = "rate_limit"
    ErrorTypeInternal     ErrorType = "internal"
    ErrorTypeExternal     ErrorType = "external"
)

// AppError 应用错误
type AppError struct {
    Type       ErrorType         `json:"type"`
    Code       string            `json:"code"`
    Message    string            `json:"message"`
    Details    string            `json:"details,omitempty"`
    Metadata   map[string]any    `json:"metadata,omitempty"`
    Cause      error             `json:"-"`
    HTTPStatus int               `json:"-"`
}

func (e *AppError) Error() string {
    if e.Cause != nil {
        return fmt.Sprintf("%s: %s (caused by: %v)", e.Code, e.Message, e.Cause)
    }
    return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *AppError) Unwrap() error {
    return e.Cause
}

// 错误构造函数
func NewValidationError(code, message string) *AppError {
    return &AppError{
        Type:       ErrorTypeValidation,
        Code:       code,
        Message:    message,
        HTTPStatus: http.StatusBadRequest,
    }
}

func NewNotFoundError(code, message string) *AppError {
    return &AppError{
        Type:       ErrorTypeNotFound,
        Code:       code,
        Message:    message,
        HTTPStatus: http.StatusNotFound,
    }
}

func NewInternalError(code, message string, cause error) *AppError {
    return &AppError{
        Type:       ErrorTypeInternal,
        Code:       code,
        Message:    message,
        Cause:      cause,
        HTTPStatus: http.StatusInternalServerError,
    }
}

func NewExternalError(code, message string, cause error) *AppError {
    return &AppError{
        Type:       ErrorTypeExternal,
        Code:       code,
        Message:    message,
        Cause:      cause,
        HTTPStatus: http.StatusBadGateway,
    }
}

// 错误处理中间件
func ErrorHandlerMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Next()
        
        if len(c.Errors) > 0 {
            err := c.Errors.Last().Err
            
            var appErr *AppError
            if errors.As(err, &appErr) {
                c.JSON(appErr.HTTPStatus, gin.H{
                    "error": appErr,
                })
            } else {
                // 未知错误
                c.JSON(http.StatusInternalServerError, gin.H{
                    "error": NewInternalError("UNKNOWN_ERROR", "An unexpected error occurred", err),
                })
            }
        }
    }
}
```

## 📊 性能优化最佳实践

### 1. 内存管理优化

```go
// optimization/memory.go
package optimization

import (
    "context"
    "sync"
    "time"
    
    "github.com/cloudwego/eino/schema"
)

// ObjectPool 对象池
type ObjectPool[T any] struct {
    pool sync.Pool
    new  func() T
}

func NewObjectPool[T any](newFunc func() T) *ObjectPool[T] {
    return &ObjectPool[T]{
        pool: sync.Pool{
            New: func() any {
                return newFunc()
            },
        },
        new: newFunc,
    }
}

func (p *ObjectPool[T]) Get() T {
    return p.pool.Get().(T)
}

func (p *ObjectPool[T]) Put(obj T) {
    p.pool.Put(obj)
}

// 预定义对象池
var (
    MessagePool = NewObjectPool(func() *schema.Message {
        return &schema.Message{}
    })
    
    DocumentPool = NewObjectPool(func() *schema.Document {
        return &schema.Document{
            MetaData: make(map[string]any),
        }
    })
    
    StringBuilderPool = NewObjectPool(func() *strings.Builder {
        return &strings.Builder{}
    })
)

// 使用示例
func ProcessMessages(messages []*schema.Message) []*schema.Message {
    result := make([]*schema.Message, 0, len(messages))
    
    for _, msg := range messages {
        // 从池中获取对象
        processed := MessagePool.Get()
        defer MessagePool.Put(processed) // 使用完毕后归还
        
        // 重置对象状态
        *processed = schema.Message{
            Role:    msg.Role,
            Content: processContent(msg.Content),
        }
        
        result = append(result, processed)
    }
    
    return result
}

// StreamBuffer 流缓冲区管理
type StreamBuffer[T any] struct {
    buffer   []T
    capacity int
    mu       sync.RWMutex
}

func NewStreamBuffer[T any](capacity int) *StreamBuffer[T] {
    return &StreamBuffer[T]{
        buffer:   make([]T, 0, capacity),
        capacity: capacity,
    }
}

func (sb *StreamBuffer[T]) Add(item T) bool {
    sb.mu.Lock()
    defer sb.mu.Unlock()
    
    if len(sb.buffer) >= sb.capacity {
        return false // 缓冲区已满
    }
    
    sb.buffer = append(sb.buffer, item)
    return true
}

func (sb *StreamBuffer[T]) Flush() []T {
    sb.mu.Lock()
    defer sb.mu.Unlock()
    
    if len(sb.buffer) == 0 {
        return nil
    }
    
    result := make([]T, len(sb.buffer))
    copy(result, sb.buffer)
    sb.buffer = sb.buffer[:0] // 清空但保留容量
    
    return result
}
```

### 2. 并发控制优化

```go
// optimization/concurrency.go
package optimization

import (
    "context"
    "sync"
    "time"
)

// WorkerPool 工作池
type WorkerPool[T any, R any] struct {
    workers    int
    jobCh      chan Job[T, R]
    resultCh   chan Result[R]
    wg         sync.WaitGroup
    ctx        context.Context
    cancel     context.CancelFunc
}

type Job[T any, R any] struct {
    ID   string
    Data T
    Fn   func(context.Context, T) (R, error)
}

type Result[R any] struct {
    ID     string
    Data   R
    Error  error
}

func NewWorkerPool[T any, R any](ctx context.Context, workers int) *WorkerPool[T, R] {
    ctx, cancel := context.WithCancel(ctx)
    
    pool := &WorkerPool[T, R]{
        workers:  workers,
        jobCh:    make(chan Job[T, R], workers*2),
        resultCh: make(chan Result[R], workers*2),
        ctx:      ctx,
        cancel:   cancel,
    }
    
    // 启动工作协程
    for i := 0; i < workers; i++ {
        pool.wg.Add(1)
        go pool.worker()
    }
    
    return pool
}

func (wp *WorkerPool[T, R]) worker() {
    defer wp.wg.Done()
    
    for {
        select {
        case <-wp.ctx.Done():
            return
        case job := <-wp.jobCh:
            result := Result[R]{ID: job.ID}
            result.Data, result.Error = job.Fn(wp.ctx, job.Data)
            
            select {
            case wp.resultCh <- result:
            case <-wp.ctx.Done():
                return
            }
        }
    }
}

func (wp *WorkerPool[T, R]) Submit(job Job[T, R]) bool {
    select {
    case wp.jobCh <- job:
        return true
    case <-wp.ctx.Done():
        return false
    default:
        return false // 队列已满
    }
}

func (wp *WorkerPool[T, R]) Results() <-chan Result[R] {
    return wp.resultCh
}

func (wp *WorkerPool[T, R]) Close() {
    wp.cancel()
    close(wp.jobCh)
    wp.wg.Wait()
    close(wp.resultCh)
}

// RateLimiter 速率限制器
type RateLimiter struct {
    tokens chan struct{}
    ticker *time.Ticker
    done   chan struct{}
}

func NewRateLimiter(rate int, burst int) *RateLimiter {
    rl := &RateLimiter{
        tokens: make(chan struct{}, burst),
        ticker: time.NewTicker(time.Second / time.Duration(rate)),
        done:   make(chan struct{}),
    }
    
    // 初始填充令牌
    for i := 0; i < burst; i++ {
        rl.tokens <- struct{}{}
    }
    
    // 定期添加令牌
    go func() {
        for {
            select {
            case <-rl.ticker.C:
                select {
                case rl.tokens <- struct{}{}:
                default:
                    // 令牌桶已满
                }
            case <-rl.done:
                return
            }
        }
    }()
    
    return rl
}

func (rl *RateLimiter) Wait(ctx context.Context) error {
    select {
    case <-rl.tokens:
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
}

func (rl *RateLimiter) Close() {
    rl.ticker.Stop()
    close(rl.done)
}
```

### 3. 缓存策略优化

```go
// optimization/cache.go
package optimization

import (
    "context"
    "crypto/md5"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

// Cache 缓存接口
type Cache interface {
    Get(ctx context.Context, key string) ([]byte, error)
    Set(ctx context.Context, key string, value []byte, ttl time.Duration) error
    Delete(ctx context.Context, key string) error
    Clear(ctx context.Context) error
}

// MemoryCache 内存缓存实现
type MemoryCache struct {
    data map[string]*cacheItem
    mu   sync.RWMutex
}

type cacheItem struct {
    value     []byte
    expiresAt time.Time
}

func NewMemoryCache() *MemoryCache {
    cache := &MemoryCache{
        data: make(map[string]*cacheItem),
    }
    
    // 启动清理协程
    go cache.cleanup()
    
    return cache
}

func (mc *MemoryCache) Get(ctx context.Context, key string) ([]byte, error) {
    mc.mu.RLock()
    defer mc.mu.RUnlock()
    
    item, exists := mc.data[key]
    if !exists {
        return nil, fmt.Errorf("key not found")
    }
    
    if time.Now().After(item.expiresAt) {
        return nil, fmt.Errorf("key expired")
    }
    
    return item.value, nil
}

func (mc *MemoryCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
    mc.mu.Lock()
    defer mc.mu.Unlock()
    
    mc.data[key] = &cacheItem{
        value:     value,
        expiresAt: time.Now().Add(ttl),
    }
    
    return nil
}

func (mc *MemoryCache) Delete(ctx context.Context, key string) error {
    mc.mu.Lock()
    defer mc.mu.Unlock()
    
    delete(mc.data, key)
    return nil
}

func (mc *MemoryCache) Clear(ctx context.Context) error {
    mc.mu.Lock()
    defer mc.mu.Unlock()
    
    mc.data = make(map[string]*cacheItem)
    return nil
}

func (mc *MemoryCache) cleanup() {
    ticker := time.NewTicker(5 * time.Minute)
    defer ticker.Stop()
    
    for range ticker.C {
        mc.mu.Lock()
        now := time.Now()
        for key, item := range mc.data {
            if now.After(item.expiresAt) {
                delete(mc.data, key)
            }
        }
        mc.mu.Unlock()
    }
}

// CacheManager 缓存管理器
type CacheManager struct {
    cache  Cache
    prefix string
}

func NewCacheManager(cache Cache, prefix string) *CacheManager {
    return &CacheManager{
        cache:  cache,
        prefix: prefix,
    }
}

func (cm *CacheManager) GetOrSet(ctx context.Context, key string, ttl time.Duration, fn func() (any, error)) (any, error) {
    fullKey := cm.prefix + ":" + key
    
    // 尝试从缓存获取
    data, err := cm.cache.Get(ctx, fullKey)
    if err == nil {
        var result any
        if err := json.Unmarshal(data, &result); err == nil {
            return result, nil
        }
    }
    
    // 缓存未命中，执行函数
    result, err := fn()
    if err != nil {
        return nil, err
    }
    
    // 存储到缓存
    data, err = json.Marshal(result)
    if err == nil {
        cm.cache.Set(ctx, fullKey, data, ttl)
    }
    
    return result, nil
}

func (cm *CacheManager) GenerateKey(parts ...string) string {
    h := md5.New()
    for _, part := range parts {
        h.Write([]byte(part))
    }
    return hex.EncodeToString(h.Sum(nil))
}

// 使用示例
func CachedChatCompletion(ctx context.Context, cacheManager *CacheManager, input string) (*schema.Message, error) {
    key := cacheManager.GenerateKey("chat", input)
    
    result, err := cacheManager.GetOrSet(ctx, key, 10*time.Minute, func() (any, error) {
        // 实际的聊天完成逻辑
        return performChatCompletion(ctx, input)
    })
    
    if err != nil {
        return nil, err
    }
    
    return result.(*schema.Message), nil
}
```

## 🔍 监控与可观测性

### 1. 指标收集

```go
// monitoring/metrics.go
package monitoring

import (
    "time"
    
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics 指标收集器
type Metrics struct {
    // 请求指标
    requestsTotal    *prometheus.CounterVec
    requestDuration  *prometheus.HistogramVec
    requestErrors    *prometheus.CounterVec
    
    // 组件指标
    componentCalls   *prometheus.CounterVec
    componentLatency *prometheus.HistogramVec
    componentErrors  *prometheus.CounterVec
    
    // 资源指标
    activeConnections prometheus.Gauge
    memoryUsage      prometheus.Gauge
    goroutineCount   prometheus.Gauge
    
    // 业务指标
    tokenUsage       *prometheus.CounterVec
    cacheHitRate     *prometheus.GaugeVec
    queueLength      *prometheus.GaugeVec
}

func NewMetrics() *Metrics {
    return &Metrics{
        requestsTotal: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "eino_requests_total",
                Help: "Total number of requests",
            },
            []string{"method", "endpoint", "status"},
        ),
        
        requestDuration: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "eino_request_duration_seconds",
                Help:    "Request duration in seconds",
                Buckets: prometheus.DefBuckets,
            },
            []string{"method", "endpoint"},
        ),
        
        requestErrors: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "eino_request_errors_total",
                Help: "Total number of request errors",
            },
            []string{"method", "endpoint", "error_type"},
        ),
        
        componentCalls: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "eino_component_calls_total",
                Help: "Total number of component calls",
            },
            []string{"component_type", "component_name"},
        ),
        
        componentLatency: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "eino_component_latency_seconds",
                Help:    "Component call latency in seconds",
                Buckets: prometheus.DefBuckets,
            },
            []string{"component_type", "component_name"},
        ),
        
        componentErrors: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "eino_component_errors_total",
                Help: "Total number of component errors",
            },
            []string{"component_type", "component_name", "error_type"},
        ),
        
        activeConnections: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "eino_active_connections",
                Help: "Number of active connections",
            },
        ),
        
        memoryUsage: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "eino_memory_usage_bytes",
                Help: "Memory usage in bytes",
            },
        ),
        
        goroutineCount: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "eino_goroutines",
                Help: "Number of goroutines",
            },
        ),
        
        tokenUsage: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "eino_token_usage_total",
                Help: "Total token usage",
            },
            []string{"model", "type"},
        ),
        
        cacheHitRate: promauto.NewGaugeVec(
            prometheus.GaugeOpts{
                Name: "eino_cache_hit_rate",
                Help: "Cache hit rate",
            },
            []string{"cache_type"},
        ),
        
        queueLength: promauto.NewGaugeVec(
            prometheus.GaugeOpts{
                Name: "eino_queue_length",
                Help: "Queue length",
            },
            []string{"queue_name"},
        ),
    }
}

// 指标记录方法
func (m *Metrics) IncRequests(method, endpoint, status string) {
    m.requestsTotal.WithLabelValues(method, endpoint, status).Inc()
}

func (m *Metrics) ObserveRequestDuration(method, endpoint string, duration time.Duration) {
    m.requestDuration.WithLabelValues(method, endpoint).Observe(duration.Seconds())
}

func (m *Metrics) IncRequestErrors(method, endpoint, errorType string) {
    m.requestErrors.WithLabelValues(method, endpoint, errorType).Inc()
}

func (m *Metrics) IncComponentCalls(componentType, componentName string) {
    m.componentCalls.WithLabelValues(componentType, componentName).Inc()
}

func (m *Metrics) ObserveComponentLatency(componentType, componentName string, duration time.Duration) {
    m.componentLatency.WithLabelValues(componentType, componentName).Observe(duration.Seconds())
}

func (m *Metrics) IncComponentErrors(componentType, componentName, errorType string) {
    m.componentErrors.WithLabelValues(componentType, componentName, errorType).Inc()
}

func (m *Metrics) SetActiveConnections(count float64) {
    m.activeConnections.Set(count)
}

func (m *Metrics) SetMemoryUsage(bytes float64) {
    m.memoryUsage.Set(bytes)
}

func (m *Metrics) SetGoroutineCount(count float64) {
    m.goroutineCount.Set(count)
}

func (m *Metrics) IncTokenUsage(model, tokenType string, count float64) {
    m.tokenUsage.WithLabelValues(model, tokenType).Add(count)
}

func (m *Metrics) SetCacheHitRate(cacheType string, rate float64) {
    m.cacheHitRate.WithLabelValues(cacheType).Set(rate)
}

func (m *Metrics) SetQueueLength(queueName string, length float64) {
    m.queueLength.WithLabelValues(queueName).Set(length)
}
```

### 2. 链路追踪

```go
// monitoring/tracing.go
package monitoring

import (
    "context"
    "fmt"
    
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/codes"
    "go.opentelemetry.io/otel/trace"
)

const (
    TracerName = "eino"
)

// TraceableComponent 可追踪的组件接口
type TraceableComponent interface {
    GetTraceInfo() TraceInfo
}

// TraceInfo 追踪信息
type TraceInfo struct {
    ComponentType string
    ComponentName string
    Version       string
    Attributes    map[string]any
}

// TracingMiddleware 追踪中间件
func TracingMiddleware(componentType, componentName string) func(next func(context.Context, any) (any, error)) func(context.Context, any) (any, error) {
    return func(next func(context.Context, any) (any, error)) func(context.Context, any) (any, error) {
        return func(ctx context.Context, input any) (any, error) {
            tracer := otel.Tracer(TracerName)
            
            spanName := fmt.Sprintf("%s.%s", componentType, componentName)
            ctx, span := tracer.Start(ctx, spanName)
            defer span.End()
            
            // 设置基础属性
            span.SetAttributes(
                attribute.String("component.type", componentType),
                attribute.String("component.name", componentName),
                attribute.String("input.type", fmt.Sprintf("%T", input)),
            )
            
            // 执行组件逻辑
            output, err := next(ctx, input)
            
            if err != nil {
                span.RecordError(err)
                span.SetStatus(codes.Error, err.Error())
                span.SetAttributes(
                    attribute.String("error.type", fmt.Sprintf("%T", err)),
                    attribute.String("error.message", err.Error()),
                )
            } else {
                span.SetStatus(codes.Ok, "success")
                span.SetAttributes(
                    attribute.String("output.type", fmt.Sprintf("%T", output)),
                )
            }
            
            return output, err
        }
    }
}

// TraceChainExecution 追踪链执行
func TraceChainExecution(ctx context.Context, chainName string, fn func(context.Context) (any, error)) (any, error) {
    tracer := otel.Tracer(TracerName)
    
    spanName := fmt.Sprintf("chain.%s", chainName)
    ctx, span := tracer.Start(ctx, spanName)
    defer span.End()
    
    span.SetAttributes(
        attribute.String("chain.name", chainName),
        attribute.String("operation", "execute"),
    )
    
    result, err := fn(ctx)
    
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
    } else {
        span.SetStatus(codes.Ok, "success")
    }
    
    return result, err
}

// AddSpanEvent 添加 Span 事件
func AddSpanEvent(ctx context.Context, name string, attributes map[string]any) {
    span := trace.SpanFromContext(ctx)
    if !span.IsRecording() {
        return
    }
    
    attrs := make([]attribute.KeyValue, 0, len(attributes))
    for k, v := range attributes {
        attrs = append(attrs, attribute.String(k, fmt.Sprintf("%v", v)))
    }
    
    span.AddEvent(name, trace.WithAttributes(attrs...))
}

// SetSpanAttributes 设置 Span 属性
func SetSpanAttributes(ctx context.Context, attributes map[string]any) {
    span := trace.SpanFromContext(ctx)
    if !span.IsRecording() {
        return
    }
    
    attrs := make([]attribute.KeyValue, 0, len(attributes))
    for k, v := range attributes {
        attrs = append(attrs, attribute.String(k, fmt.Sprintf("%v", v)))
    }
    
    span.SetAttributes(attrs...)
}
```

### 3. 日志记录

```go
// monitoring/logging.go
package monitoring

import (
    "context"
    "fmt"
    "time"
    
    "github.com/sirupsen/logrus"
    "go.opentelemetry.io/otel/trace"
)

// Logger 日志接口
type Logger interface {
    Debug(msg string, fields ...any)
    Info(msg string, fields ...any)
    Warn(msg string, fields ...any)
    Error(msg string, fields ...any)
    Fatal(msg string, fields ...any)
    
    WithContext(ctx context.Context) Logger
    WithFields(fields map[string]any) Logger
}

// StructuredLogger 结构化日志实现
type StructuredLogger struct {
    logger *logrus.Logger
    fields logrus.Fields
}

func NewStructuredLogger() *StructuredLogger {
    logger := logrus.New()
    logger.SetFormatter(&logrus.JSONFormatter{
        TimestampFormat: time.RFC3339,
        FieldMap: logrus.FieldMap{
            logrus.FieldKeyTime:  "timestamp",
            logrus.FieldKeyLevel: "level",
            logrus.FieldKeyMsg:   "message",
        },
    })
    
    return &StructuredLogger{
        logger: logger,
        fields: make(logrus.Fields),
    }
}

func (sl *StructuredLogger) Debug(msg string, fields ...any) {
    sl.logWithFields(logrus.DebugLevel, msg, fields...)
}

func (sl *StructuredLogger) Info(msg string, fields ...any) {
    sl.logWithFields(logrus.InfoLevel, msg, fields...)
}

func (sl *StructuredLogger) Warn(msg string, fields ...any) {
    sl.logWithFields(logrus.WarnLevel, msg, fields...)
}

func (sl *StructuredLogger) Error(msg string, fields ...any) {
    sl.logWithFields(logrus.ErrorLevel, msg, fields...)
}

func (sl *StructuredLogger) Fatal(msg string, fields ...any) {
    sl.logWithFields(logrus.FatalLevel, msg, fields...)
}

func (sl *StructuredLogger) WithContext(ctx context.Context) Logger {
    newFields := make(logrus.Fields)
    for k, v := range sl.fields {
        newFields[k] = v
    }
    
    // 添加追踪信息
    if span := trace.SpanFromContext(ctx); span.SpanContext().IsValid() {
        newFields["trace_id"] = span.SpanContext().TraceID().String()
        newFields["span_id"] = span.SpanContext().SpanID().String()
    }
    
    return &StructuredLogger{
        logger: sl.logger,
        fields: newFields,
    }
}

func (sl *StructuredLogger) WithFields(fields map[string]any) Logger {
    newFields := make(logrus.Fields)
    for k, v := range sl.fields {
        newFields[k] = v
    }
    for k, v := range fields {
        newFields[k] = v
    }
    
    return &StructuredLogger{
        logger: sl.logger,
        fields: newFields,
    }
}

func (sl *StructuredLogger) logWithFields(level logrus.Level, msg string, fields ...any) {
    entry := sl.logger.WithFields(sl.fields)
    
    // 处理额外字段
    if len(fields) > 0 {
        extraFields := make(logrus.Fields)
        for i := 0; i < len(fields); i += 2 {
            if i+1 < len(fields) {
                key := fmt.Sprintf("%v", fields[i])
                value := fields[i+1]
                extraFields[key] = value
            }
        }
        entry = entry.WithFields(extraFields)
    }
    
    entry.Log(level, msg)
}

// ComponentLogger 组件日志包装器
type ComponentLogger struct {
    logger        Logger
    componentType string
    componentName string
}

func NewComponentLogger(logger Logger, componentType, componentName string) *ComponentLogger {
    return &ComponentLogger{
        logger: logger.WithFields(map[string]any{
            "component_type": componentType,
            "component_name": componentName,
        }),
        componentType: componentType,
        componentName: componentName,
    }
}

func (cl *ComponentLogger) LogExecution(ctx context.Context, operation string, duration time.Duration, err error) {
    logger := cl.logger.WithContext(ctx)
    
    fields := map[string]any{
        "operation": operation,
        "duration":  duration.String(),
    }
    
    if err != nil {
        fields["error"] = err.Error()
        logger.Error("Component execution failed", "fields", fields)
    } else {
        logger.Info("Component execution completed", "fields", fields)
    }
}

func (cl *ComponentLogger) LogInput(ctx context.Context, input any) {
    logger := cl.logger.WithContext(ctx)
    logger.Debug("Component input", 
        "input_type", fmt.Sprintf("%T", input),
        "input_size", getInputSize(input))
}

func (cl *ComponentLogger) LogOutput(ctx context.Context, output any) {
    logger := cl.logger.WithContext(ctx)
    logger.Debug("Component output", 
        "output_type", fmt.Sprintf("%T", output),
        "output_size", getOutputSize(output))
}

func getInputSize(input any) int {
    // 实现输入大小计算逻辑
    return 0
}

func getOutputSize(output any) int {
    // 实现输出大小计算逻辑
    return 0
}
```

## 🚀 部署与运维最佳实践

### 1. Docker 容器化

```dockerfile
# Dockerfile
FROM golang:1.21-alpine AS builder

# 设置工作目录
WORKDIR /app

# 安装依赖
RUN apk add --no-cache git ca-certificates tzdata

# 复制 go mod 文件
COPY go.mod go.sum ./

# 下载依赖
RUN go mod download

# 复制源代码
COPY . .

# 构建应用
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/server

# 运行阶段
FROM alpine:latest

# 安装 ca-certificates
RUN apk --no-cache add ca-certificates tzdata

WORKDIR /root/

# 从构建阶段复制二进制文件
COPY --from=builder /app/main .
COPY --from=builder /app/configs ./configs

# 创建非 root 用户
RUN adduser -D -s /bin/sh appuser
USER appuser

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# 启动应用
CMD ["./main"]
```

### 2. Docker Compose 配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  eino-app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ENV=production
      - LOG_LEVEL=info
      - DATABASE_URL=postgres://user:password@postgres:5432/einodb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./configs:/root/configs:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=einodb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d einodb"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 3. Kubernetes 部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eino-app
  labels:
    app: eino-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: eino-app
  template:
    metadata:
      labels:
        app: eino-app
    spec:
      containers:
      - name: eino-app
        image: your-registry/eino-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: eino-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: eino-secrets
              key: redis-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /root/configs
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: eino-config

---
apiVersion: v1
kind: Service
metadata:
  name: eino-service
spec:
  selector:
    app: eino-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: eino-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: eino-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: eino-service
            port:
              number: 80
```

## 🔒 安全最佳实践

### 1. API 安全

```go
// security/auth.go
package security

import (
    "context"
    "crypto/rand"
    "crypto/subtle"
    "encoding/base64"
    "fmt"
    "strings"
    "time"
    
    "github.com/golang-jwt/jwt/v5"
)

// AuthService 认证服务
type AuthService struct {
    jwtSecret     []byte
    tokenExpiry   time.Duration
    refreshExpiry time.Duration
}

func NewAuthService(secret string, tokenExpiry, refreshExpiry time.Duration) *AuthService {
    return &AuthService{
        jwtSecret:     []byte(secret),
        tokenExpiry:   tokenExpiry,
        refreshExpiry: refreshExpiry,
    }
}

// GenerateToken 生成访问令牌
func (as *AuthService) GenerateToken(userID string, roles []string) (string, error) {
    claims := jwt.MapClaims{
        "user_id": userID,
        "roles":   roles,
        "exp":     time.Now().Add(as.tokenExpiry).Unix(),
        "iat":     time.Now().Unix(),
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(as.jwtSecret)
}

// ValidateToken 验证令牌
func (as *AuthService) ValidateToken(tokenString string) (*jwt.MapClaims, error) {
    token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return as.jwtSecret, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
        return &claims, nil
    }
    
    return nil, fmt.Errorf("invalid token")
}

// RateLimitMiddleware 速率限制中间件
func RateLimitMiddleware(limiter *RateLimiter) gin.HandlerFunc {
    return func(c *gin.Context) {
        clientIP := c.ClientIP()
        
        if err := limiter.Wait(c.Request.Context()); err != nil {
            c.JSON(http.StatusTooManyRequests, gin.H{
                "error": "Rate limit exceeded",
            })
            c.Abort()
            return
        }
        
        c.Next()
    }
}

// APIKeyMiddleware API 密钥验证中间件
func APIKeyMiddleware(validKeys map[string]bool) gin.HandlerFunc {
    return func(c *gin.Context) {
        apiKey := c.GetHeader("X-API-Key")
        if apiKey == "" {
            c.JSON(http.StatusUnauthorized, gin.H{
                "error": "API key required",
            })
            c.Abort()
            return
        }
        
        if !validKeys[apiKey] {
            c.JSON(http.StatusUnauthorized, gin.H{
                "error": "Invalid API key",
            })
            c.Abort()
            return
        }
        
        c.Next()
    }
}

// SecureHeaders 安全头中间件
func SecureHeadersMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Header("X-Content-Type-Options", "nosniff")
        c.Header("X-Frame-Options", "DENY")
        c.Header("X-XSS-Protection", "1; mode=block")
        c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        c.Header("Content-Security-Policy", "default-src 'self'")
        c.Next()
    }
}
```

### 2. 数据加密

```go
// security/encryption.go
package security

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "crypto/sha256"
    "encoding/base64"
    "fmt"
    "io"
    
    "golang.org/x/crypto/pbkdf2"
)

// EncryptionService 加密服务
type EncryptionService struct {
    key []byte
}

func NewEncryptionService(password string, salt []byte) *EncryptionService {
    key := pbkdf2.Key([]byte(password), salt, 10000, 32, sha256.New)
    return &EncryptionService{key: key}
}

// Encrypt 加密数据
func (es *EncryptionService) Encrypt(plaintext []byte) (string, error) {
    block, err := aes.NewCipher(es.key)
    if err != nil {
        return "", err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }
    
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return "", err
    }
    
    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
    return base64.StdEncoding.EncodeToString(ciphertext), nil
}

// Decrypt 解密数据
func (es *EncryptionService) Decrypt(ciphertext string) ([]byte, error) {
    data, err := base64.StdEncoding.DecodeString(ciphertext)
    if err != nil {
        return nil, err
    }
    
    block, err := aes.NewCipher(es.key)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonceSize := gcm.NonceSize()
    if len(data) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }
    
    nonce, ciphertext := data[:nonceSize], data[nonceSize:]
    return gcm.Open(nil, nonce, ciphertext, nil)
}

// SensitiveDataHandler 敏感数据处理器
type SensitiveDataHandler struct {
    encryption *EncryptionService
}

func NewSensitiveDataHandler(encryption *EncryptionService) *SensitiveDataHandler {
    return &SensitiveDataHandler{encryption: encryption}
}

// MaskSensitiveData 脱敏敏感数据
func (sdh *SensitiveDataHandler) MaskSensitiveData(data string, dataType string) string {
    switch dataType {
    case "email":
        return maskEmail(data)
    case "phone":
        return maskPhone(data)
    case "id_card":
        return maskIDCard(data)
    case "credit_card":
        return maskCreditCard(data)
    default:
        return maskDefault(data)
    }
}

func maskEmail(email string) string {
    parts := strings.Split(email, "@")
    if len(parts) != 2 {
        return "***"
    }
    
    username := parts[0]
    domain := parts[1]
    
    if len(username) <= 2 {
        return "***@" + domain
    }
    
    return username[:1] + "***" + username[len(username)-1:] + "@" + domain
}

func maskPhone(phone string) string {
    if len(phone) < 7 {
        return "***"
    }
    
    return phone[:3] + "****" + phone[len(phone)-3:]
}

func maskIDCard(idCard string) string {
    if len(idCard) < 8 {
        return "***"
    }
    
    return idCard[:4] + "**********" + idCard[len(idCard)-4:]
}

func maskCreditCard(card string) string {
    if len(card) < 8 {
        return "***"
    }
    
    return card[:4] + " **** **** " + card[len(card)-4:]
}

func maskDefault(data string) string {
    if len(data) <= 4 {
        return "***"
    }
    
    return data[:2] + "***" + data[len(data)-2:]
}
```

## 📈 性能测试与优化

### 1. 基准测试

```go
// benchmark/benchmark_test.go
package benchmark

import (
    "context"
    "testing"
    "time"
    
    "github.com/cloudwego/eino/compose"
    "github.com/cloudwego/eino/schema"
)

func BenchmarkChainExecution(b *testing.B) {
    ctx := context.Background()
    
    // 创建测试链
    chain := createTestChain()
    runnable, err := chain.Compile(ctx)
    if err != nil {
        b.Fatal(err)
    }
    
    input := "test input"
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _, err := runnable.Invoke(ctx, input)
            if err != nil {
                b.Error(err)
            }
        }
    })
}

func BenchmarkStreamProcessing(b *testing.B) {
    ctx := context.Background()
    
    // 创建流处理链
    chain := createStreamChain()
    runnable, err := chain.Compile(ctx)
    if err != nil {
        b.Fatal(err)
    }
    
    input := "test stream input"
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        stream, err := runnable.Stream(ctx, input)
        if err != nil {
            b.Error(err)
            continue
        }
        
        // 消费流数据
        for {
            _, err := stream.Recv()
            if err == io.EOF {
                break
            }
            if err != nil {
                b.Error(err)
                break
            }
        }
        stream.Close()
    }
}

func BenchmarkConcurrentExecution(b *testing.B) {
    ctx := context.Background()
    
    chain := createTestChain()
    runnable, err := chain.Compile(ctx)
    if err != nil {
        b.Fatal(err)
    }
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            input := fmt.Sprintf("test input %d", time.Now().UnixNano())
            _, err := runnable.Invoke(ctx, input)
            if err != nil {
                b.Error(err)
            }
        }
    })
}

// 性能分析辅助函数
func BenchmarkMemoryAllocation(b *testing.B) {
    b.ReportAllocs()
    
    for i := 0; i < b.N; i++ {
        // 测试内存分配
        msg := &schema.Message{
            Role:    schema.User,
            Content: "test message",
            Extra:   make(map[string]any),
        }
        
        // 模拟处理
        _ = processMessage(msg)
    }
}

func processMessage(msg *schema.Message) *schema.Message {
    return &schema.Message{
        Role:    schema.Assistant,
        Content: "processed: " + msg.Content,
    }
}

func createTestChain() *compose.Chain[string, string] {
    processor := compose.InvokableLambda(func(ctx context.Context, input string) (string, error) {
        return "processed: " + input, nil
    })
    
    return compose.NewChain[string, string]().
        AppendLambda("processor", processor)
}

func createStreamChain() *compose.Chain[string, string] {
    processor := compose.StreamableLambda(func(ctx context.Context, input string) (*schema.StreamReader[string], error) {
        words := strings.Fields(input)
        return schema.StreamReaderFromArray(words), nil
    })
    
    return compose.NewChain[string, string]().
        AppendLambda("processor", processor)
}
```

### 2. 负载测试

```go
// loadtest/loadtest.go
package loadtest

import (
    "context"
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

// LoadTestConfig 负载测试配置
type LoadTestConfig struct {
    Concurrency int           // 并发数
    Duration    time.Duration // 测试持续时间
    RampUp      time.Duration // 预热时间
    Target      string        // 目标地址
}

// LoadTestResult 负载测试结果
type LoadTestResult struct {
    TotalRequests    int64
    SuccessRequests  int64
    FailedRequests   int64
    AverageLatency   time.Duration
    MinLatency       time.Duration
    MaxLatency       time.Duration
    P95Latency       time.Duration
    P99Latency       time.Duration
    RequestsPerSecond float64
}

// LoadTester 负载测试器
type LoadTester struct {
    config  *LoadTestConfig
    client  HTTPClient
    metrics *LoadTestMetrics
}

type HTTPClient interface {
    Do(ctx context.Context, request any) (any, error)
}

type LoadTestMetrics struct {
    totalRequests   int64
    successRequests int64
    failedRequests  int64
    latencies       []time.Duration
    mu              sync.Mutex
}

func NewLoadTester(config *LoadTestConfig, client HTTPClient) *LoadTester {
    return &LoadTester{
        config:  config,
        client:  client,
        metrics: &LoadTestMetrics{},
    }
}

func (lt *LoadTester) Run(ctx context.Context) (*LoadTestResult, error) {
    fmt.Printf("Starting load test with %d concurrent users for %v\n", 
        lt.config.Concurrency, lt.config.Duration)
    
    // 创建工作协程
    var wg sync.WaitGroup
    startTime := time.Now()
    
    // 预热阶段
    if lt.config.RampUp > 0 {
        fmt.Printf("Ramping up for %v\n", lt.config.RampUp)
        time.Sleep(lt.config.RampUp)
    }
    
    // 启动负载测试
    testCtx, cancel := context.WithTimeout(ctx, lt.config.Duration)
    defer cancel()
    
    for i := 0; i < lt.config.Concurrency; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            lt.worker(testCtx, workerID)
        }(i)
    }
    
    wg.Wait()
    endTime := time.Now()
    
    return lt.calculateResults(startTime, endTime), nil
}

func (lt *LoadTester) worker(ctx context.Context, workerID int) {
    for {
        select {
        case <-ctx.Done():
            return
        default:
            lt.executeRequest(ctx)
        }
    }
}

func (lt *LoadTester) executeRequest(ctx context.Context) {
    startTime := time.Now()
    
    atomic.AddInt64(&lt.metrics.totalRequests, 1)
    
    // 执行请求
    _, err := lt.client.Do(ctx, createTestRequest())
    
    latency := time.Since(startTime)
    
    lt.metrics.mu.Lock()
    lt.metrics.latencies = append(lt.metrics.latencies, latency)
    lt.metrics.mu.Unlock()
    
    if err != nil {
        atomic.AddInt64(&lt.metrics.failedRequests, 1)
    } else {
        atomic.AddInt64(&lt.metrics.successRequests, 1)
    }
}

func (lt *LoadTester) calculateResults(startTime, endTime time.Time) *LoadTestResult {
    duration := endTime.Sub(startTime)
    
    lt.metrics.mu.Lock()
    latencies := make([]time.Duration, len(lt.metrics.latencies))
    copy(latencies, lt.metrics.latencies)
    lt.metrics.mu.Unlock()
    
    // 排序延迟数据
    sort.Slice(latencies, func(i, j int) bool {
        return latencies[i] < latencies[j]
    })
    
    var avgLatency time.Duration
    if len(latencies) > 0 {
        var total time.Duration
        for _, lat := range latencies {
            total += lat
        }
        avgLatency = total / time.Duration(len(latencies))
    }
    
    result := &LoadTestResult{
        TotalRequests:     atomic.LoadInt64(&lt.metrics.totalRequests),
        SuccessRequests:   atomic.LoadInt64(&lt.metrics.successRequests),
        FailedRequests:    atomic.LoadInt64(&lt.metrics.failedRequests),
        AverageLatency:    avgLatency,
        RequestsPerSecond: float64(atomic.LoadInt64(&lt.metrics.totalRequests)) / duration.Seconds(),
    }
    
    if len(latencies) > 0 {
        result.MinLatency = latencies[0]
        result.MaxLatency = latencies[len(latencies)-1]
        result.P95Latency = latencies[int(float64(len(latencies))*0.95)]
        result.P99Latency = latencies[int(float64(len(latencies))*0.99)]
    }
    
    return result
}

func createTestRequest() any {
    return map[string]any{
        "message": "test message",
        "user_id": "test_user",
    }
}
```

## 🎯 总结

本文档涵盖了 Eino 框架在生产环境中的最佳实践，包括：

1. **架构设计**: 分层架构、服务设计、配置管理
2. **组件开发**: 自定义组件、Lambda函数、错误处理
3. **性能优化**: 内存管理、并发控制、缓存策略
4. **监控运维**: 指标收集、链路追踪、日志记录
5. **部署运维**: 容器化、Kubernetes、CI/CD
6. **安全实践**: 认证授权、数据加密、安全防护
7. **性能测试**: 基准测试、负载测试、性能分析

遵循这些最佳实践，可以帮助您构建高性能、可靠、安全的 LLM 应用系统。

---

**上一篇**: [关键数据结构与继承关系](eino-09-data-structures.md)

**更新时间**: 2024-12-19 | **文档版本**: v1.0
