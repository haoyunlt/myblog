# Eino-01-Schema模块-API

本文档详细描述 Schema 模块对外提供的所有 API，包括请求/响应结构体、字段说明、入口函数、调用链路和使用示例。

---

## 1. Message API

### 1.1 消息创建 API

#### 1.1.1 SystemMessage

**功能说明**：创建系统角色消息，通常用于设置 AI 助手的行为规则和角色定位。

**函数签名**：
```go
func SystemMessage(content string) *Message
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| content | string | 是 | 系统消息的文本内容 |

**返回值**：
```go
type Message struct {
    Role    RoleType // 固定为 System
    Content string   // 传入的 content 参数
}
```

**核心代码**：
```go
// SystemMessage 创建系统消息
// 用途：设置 AI 助手的角色、规则、行为约束
// 位置：通常放在对话的最开始
func SystemMessage(content string) *Message {
    return &Message{
        Role:    System,
        Content: content,
    }
}
```

**使用示例**：
```go
// 示例 1：设置助手角色
sysMsg := schema.SystemMessage("你是一个专业的 Go 语言编程助手")

// 示例 2：设置行为规则
sysMsg := schema.SystemMessage(`
你是一个有帮助的 AI 助手。请遵循以下规则：
1. 回答要准确、专业
2. 不确定时要明确说明
3. 代码示例要完整可运行
`)

// 示例 3：Few-shot 示例
sysMsg := schema.SystemMessage(`
你是一个情感分析助手。以下是示例：

输入：这个产品真不错！
输出：积极

输入：质量太差了
输出：消极
`)
```

**最佳实践**：
- System 消息应该放在对话的最开始
- 避免在 System 消息中频繁使用用户特定信息
- 可以包含 Few-shot 示例来引导模型行为
- 内容应该清晰、具体，避免模糊的指令

#### 1.1.2 UserMessage

**功能说明**：创建用户角色消息，表示用户的输入或问题。

**函数签名**：
```go
func UserMessage(content string) *Message
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| content | string | 是 | 用户消息的文本内容 |

**返回值**：
```go
type Message struct {
    Role    RoleType // 固定为 User
    Content string   // 传入的 content 参数
}
```

**核心代码**：
```go
// UserMessage 创建用户消息
// 用途：表示用户的输入、问题或指令
// 特点：可以包含变量占位符，配合模板使用
func UserMessage(content string) *Message {
    return &Message{
        Role:    User,
        Content: content,
    }
}
```

**使用示例**：
```go
// 示例 1：简单问题
userMsg := schema.UserMessage("什么是 Eino 框架？")

// 示例 2：带变量的模板消息
userMsg := schema.UserMessage("请帮我查询{city}的天气")

// 示例 3：多行输入
userMsg := schema.UserMessage(`
请分析以下代码的性能问题：

func process(data []int) {
    // 代码内容
}
`)
```

**最佳实践**：
- User 消息通常和 Assistant 消息交替出现
- 可以使用模板变量（配合 Format 方法）
- 对于长文本，考虑拆分或使用 MultiContent

#### 1.1.3 AssistantMessage

**功能说明**：创建助手角色消息，表示 AI 模型的回复，可以包含工具调用。

**函数签名**：
```go
func AssistantMessage(content string, toolCalls []ToolCall) *Message
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| content | string | 是 | 助手回复的文本内容 |
| toolCalls | []ToolCall | 否 | 工具调用列表，可以为 nil |

**返回值**：
```go
type Message struct {
    Role      RoleType   // 固定为 Assistant
    Content   string     // 传入的 content 参数
    ToolCalls []ToolCall // 传入的 toolCalls 参数
}
```

**核心代码**：
```go
// AssistantMessage 创建助手消息
// 用途：表示 AI 模型的回复
// 特点：可以同时包含文本内容和工具调用
func AssistantMessage(content string, toolCalls []ToolCall) *Message {
    return &Message{
        Role:      Assistant,
        Content:   content,
        ToolCalls: toolCalls,
    }
}
```

**使用示例**：
```go
// 示例 1：纯文本回复
assistantMsg := schema.AssistantMessage("Eino 是一个 Go 语言的 LLM 应用开发框架", nil)

// 示例 2：带工具调用的回复
toolCalls := []schema.ToolCall{
    {
        ID:   "call-123",
        Type: "function",
        Function: schema.FunctionCall{
            Name:      "get_weather",
            Arguments: `{"city":"北京"}`,
        },
    },
}
assistantMsg := schema.AssistantMessage("让我查询一下北京的天气", toolCalls)

// 示例 3：多个工具调用
toolCalls := []schema.ToolCall{
    {
        ID:   "call-1",
        Type: "function",
        Function: schema.FunctionCall{
            Name:      "search_web",
            Arguments: `{"query":"Eino framework"}`,
        },
    },
    {
        ID:   "call-2",
        Type: "function",
        Function: schema.FunctionCall{
            Name:      "get_current_time",
            Arguments: `{}`,
        },
    },
}
assistantMsg := schema.AssistantMessage("我需要搜索信息并获取当前时间", toolCalls)
```

**最佳实践**：
- Assistant 消息通常是 ChatModel 的输出
- 如果包含 ToolCalls，通常需要执行工具后继续对话
- Content 和 ToolCalls 可以同时存在

#### 1.1.4 ToolMessage

**功能说明**：创建工具角色消息，表示工具执行的结果。

**函数签名**：
```go
func ToolMessage(content string, toolCallID string, opts ...ToolMessageOption) *Message
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| content | string | 是 | 工具执行结果的文本内容 |
| toolCallID | string | 是 | 对应的工具调用 ID |
| opts | ...ToolMessageOption | 否 | 可选配置，如 WithToolName |

**返回值**：
```go
type Message struct {
    Role       RoleType // 固定为 Tool
    Content    string   // 传入的 content 参数
    ToolCallID string   // 传入的 toolCallID 参数
    ToolName   string   // 通过 WithToolName 设置
}
```

**核心代码**：
```go
// ToolMessage 创建工具消息
// 用途：表示工具执行的结果
// 关联：通过 ToolCallID 关联到 AssistantMessage 的 ToolCall
func ToolMessage(content string, toolCallID string, opts ...ToolMessageOption) *Message {
    o := &toolMessageOptions{}
    for _, opt := range opts {
        opt(o)
    }
    return &Message{
        Role:       Tool,
        Content:    content,
        ToolCallID: toolCallID,
        ToolName:   o.toolName,
    }
}

// WithToolName 设置工具名称
func WithToolName(name string) ToolMessageOption {
    return func(o *toolMessageOptions) {
        o.toolName = name
    }
}
```

**使用示例**：
```go
// 示例 1：简单工具结果
toolMsg := schema.ToolMessage(
    "北京今天晴天，温度 25°C",
    "call-123",
)

// 示例 2：带工具名称
toolMsg := schema.ToolMessage(
    `{"temperature":25,"condition":"sunny"}`,
    "call-123",
    schema.WithToolName("get_weather"),
)

// 示例 3：工具执行失败
toolMsg := schema.ToolMessage(
    "错误：未找到指定城市",
    "call-123",
    schema.WithToolName("get_weather"),
)
```

**最佳实践**：
- ToolCallID 必须与 AssistantMessage 的 ToolCall.ID 对应
- Content 可以是纯文本或 JSON 格式
- 建议使用 WithToolName 明确工具名称
- 工具失败时应该返回错误信息，而不是抛出异常

---

### 1.2 消息模板 API

#### 1.2.1 Message.Format

**功能说明**：使用指定的模板格式渲染消息内容。

**函数签名**：
```go
func (m *Message) Format(ctx context.Context, vs map[string]any, formatType FormatType) ([]*Message, error)
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| ctx | context.Context | 是 | 上下文 |
| vs | map[string]any | 是 | 模板变量字典 |
| formatType | FormatType | 是 | 模板类型（FString/GoTemplate/Jinja2） |

**返回值**：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| []*Message | 消息数组 | 渲染后的消息（通常只有一条） |
| error | 错误 | 渲染失败时返回错误 |

**核心代码**：
```go
// Format 渲染消息模板
// 支持三种模板格式：FString、GoTemplate、Jinja2
// 会同时渲染 Content 和 MultiContent 中的文本
func (m *Message) Format(ctx context.Context, vs map[string]any, formatType FormatType) ([]*Message, error) {
    // 渲染 Content
    c, err := formatContent(m.Content, vs, formatType)
    if err != nil {
        return nil, err
    }
    copied := *m
    copied.Content = c

    // 复制 MultiContent
    if len(m.MultiContent) != 0 {
        copied.MultiContent = make([]ChatMessagePart, len(m.MultiContent))
        copy(copied.MultiContent, m.MultiContent)
    }

    // 渲染 MultiContent 中的文本
    for i, mc := range copied.MultiContent {
        switch mc.Type {
        case ChatMessagePartTypeText:
            nmc, err := formatContent(mc.Text, vs, formatType)
            if err != nil {
                return nil, err
            }
            copied.MultiContent[i].Text = nmc
        // 处理其他类型（ImageURL、AudioURL 等）
        }
    }
    return []*Message{&copied}, nil
}

// formatContent 根据格式类型渲染内容
func formatContent(content string, vs map[string]any, formatType FormatType) (string, error) {
    switch formatType {
    case FString:
        return pyfmt.Fmt(content, vs)  // Python 格式字符串
    case GoTemplate:
        // Go 标准模板
        parsedTmpl, err := template.New("template").Parse(content)
        // ...
    case Jinja2:
        // Jinja2 模板
        tpl, err := jinjaEnv.FromString(content)
        // ...
    }
}
```

**使用示例**：

```go
// 示例 1：FString 格式（Python 风格）
msg := schema.UserMessage("你好，{name}！今天是{date}")
params := map[string]any{
    "name": "Alice",
    "date": "2024-12-19",
}
rendered, _ := msg.Format(ctx, params, schema.FString)
// rendered[0].Content = "你好，Alice！今天是2024-12-19"

// 示例 2：GoTemplate 格式
msg := schema.UserMessage("你好，{{.name}}！{{if .vip}}您是VIP用户{{end}}")
params := map[string]any{
    "name": "Bob",
    "vip":  true,
}
rendered, _ := msg.Format(ctx, params, schema.GoTemplate)
// rendered[0].Content = "你好，Bob！您是VIP用户"

// 示例 3：Jinja2 格式（支持循环）
msg := schema.UserMessage(`
任务列表：
{% for task in tasks %}
- {{ task }}
{% endfor %}
`)
params := map[string]any{
    "tasks": []string{"学习 Eino", "编写代码", "测试功能"},
}
rendered, _ := msg.Format(ctx, params, schema.Jinja2)
```

**调用链路**：
```
用户代码
  ↓
Message.Format(ctx, params, formatType)
  ↓
formatContent(content, params, formatType)
  ↓
根据 formatType 选择：
  - FString: pyfmt.Fmt()
  - GoTemplate: template.Execute()
  - Jinja2: gonja.Execute()
  ↓
返回渲染后的内容
```

**性能要点**：
- FString 最快，适合简单变量替换
- GoTemplate 中等，适合有条件和循环的场景
- Jinja2 最慢但功能最强，适合需要兼容 Python 的场景
- 模板解析有缓存机制（GoTemplate 和 Jinja2）

**最佳实践**：
- 优先使用 FString，除非需要条件或循环
- 模板中避免复杂的逻辑运算
- 使用 Jinja2 时注意禁用了 include、extends 等危险特性

#### 1.2.2 MessagesPlaceholder

**功能说明**：创建消息占位符，用于在模板中插入一组消息。

**函数签名**：
```go
func MessagesPlaceholder(key string, optional bool) MessagesTemplate
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| key | string | 是 | 参数字典中的键名 |
| optional | bool | 是 | 是否可选（true 时键不存在不报错） |

**返回值**：

| 类型 | 说明 |
|------|------|
| MessagesTemplate | 消息模板接口实现 |

**核心代码**：
```go
// MessagesPlaceholder 创建消息占位符
// 用途：在消息模板中插入一组消息（如历史对话）
// 特点：不渲染内容，直接返回参数中的消息数组
func MessagesPlaceholder(key string, optional bool) MessagesTemplate {
    return &messagesPlaceholder{
        key:      key,
        optional: optional,
    }
}

type messagesPlaceholder struct {
    key      string
    optional bool
}

// Format 从参数中提取消息数组
func (p *messagesPlaceholder) Format(ctx context.Context, vs map[string]any, _ FormatType) ([]*Message, error) {
    v, ok := vs[p.key]
    if !ok {
        if p.optional {
            return []*Message{}, nil  // 可选时返回空数组
        }
        return nil, fmt.Errorf("message placeholder format: %s not found", p.key)
    }

    msgs, ok := v.([]*Message)
    if !ok {
        return nil, fmt.Errorf("only messages can be used to format message placeholder")
    }

    return msgs, nil
}
```

**使用示例**：
```go
// 通常与 prompt.FromMessages 配合使用（在 components/prompt 模块）
// 这里展示 MessagesPlaceholder 的独立用法

// 示例 1：插入历史对话
placeholder := schema.MessagesPlaceholder("history", false)
params := map[string]any{
    "history": []*schema.Message{
        schema.UserMessage("你是谁？"),
        schema.AssistantMessage("我是 AI 助手", nil),
    },
}
messages, _ := placeholder.Format(ctx, params, schema.FString)
// messages = [UserMessage, AssistantMessage]

// 示例 2：可选的历史对话
placeholder := schema.MessagesPlaceholder("history", true)
params := map[string]any{} // 不包含 history
messages, _ := placeholder.Format(ctx, params, schema.FString)
// messages = [] (空数组，不报错)

// 示例 3：完整的对话模板构建
// （这个需要 prompt 模块支持，这里只展示概念）
template := []schema.MessagesTemplate{
    schema.SystemMessage("你是一个有帮助的助手"),
    schema.MessagesPlaceholder("history", true),  // 历史对话
    schema.UserMessage("当前问题：{query}"),
}
```

**最佳实践**：
- 用于插入历史对话或上下文消息
- optional=true 适用于首次对话（无历史）的场景
- 确保参数中的值是 `[]*Message` 类型

---

### 1.3 消息拼接 API

#### 1.3.1 ConcatMessages

**功能说明**：将多条消息拼接为一条完整消息，主要用于处理流式消息。

**函数签名**：
```go
func ConcatMessages(msgs []*Message) (*Message, error)
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| msgs | []*Message | 是 | 待拼接的消息数组 |

**返回值**：

| 类型 | 说明 |
|------|------|
| *Message | 拼接后的完整消息 |
| error | 拼接失败时返回错误 |

**核心代码**：
```go
// ConcatMessages 拼接消息
// 约束：
//   1. 所有消息的 Role 必须相同
//   2. 所有消息的 Name 必须相同（如果有）
//   3. ToolCalls 通过 Index 合并
//   4. Content 按顺序拼接
//   5. TokenUsage 取最大值
func ConcatMessages(msgs []*Message) (*Message, error) {
    var (
        contents            []string
        contentLen          int
        reasoningContents   []string
        reasoningContentLen int
        toolCalls           []ToolCall
        ret                 = Message{}
        extraList           = make([]map[string]any, 0, len(msgs))
    )

    for idx, msg := range msgs {
        if msg == nil {
            return nil, fmt.Errorf("unexpected nil chunk in message stream, index: %d", idx)
        }

        // 检查 Role 一致性
        if msg.Role != "" {
            if ret.Role == "" {
                ret.Role = msg.Role
            } else if ret.Role != msg.Role {
                return nil, fmt.Errorf("cannot concat messages with different roles")
            }
        }

        // 收集 Content
        if msg.Content != "" {
            contents = append(contents, msg.Content)
            contentLen += len(msg.Content)
        }

        // 收集 ToolCalls
        if len(msg.ToolCalls) > 0 {
            toolCalls = append(toolCalls, msg.ToolCalls...)
        }

        // 收集 ResponseMeta
        if msg.ResponseMeta != nil {
            // 合并 TokenUsage（取最大值）
            if msg.ResponseMeta.Usage != nil {
                if ret.ResponseMeta.Usage.TotalTokens < msg.ResponseMeta.Usage.TotalTokens {
                    ret.ResponseMeta.Usage = msg.ResponseMeta.Usage
                }
            }
        }
    }

    // 拼接 Content（使用 strings.Builder 优化性能）
    if len(contents) > 0 {
        var sb strings.Builder
        sb.Grow(contentLen)  // 预分配内存
        for _, content := range contents {
            sb.WriteString(content)
        }
        ret.Content = sb.String()
    }

    // 合并 ToolCalls
    if len(toolCalls) > 0 {
        merged, err := concatToolCalls(toolCalls)
        if err != nil {
            return nil, err
        }
        ret.ToolCalls = merged
    }

    return &ret, nil
}

// concatToolCalls 合并工具调用
// 策略：相同 Index 的 ToolCall 合并为一个
func concatToolCalls(chunks []ToolCall) ([]ToolCall, error) {
    var merged []ToolCall
    m := make(map[int][]int)
    
    // 按 Index 分组
    for i := range chunks {
        index := chunks[i].Index
        if index == nil {
            merged = append(merged, chunks[i])
        } else {
            m[*index] = append(m[*index], i)
        }
    }

    // 合并同一 Index 的 ToolCall
    for k, v := range m {
        // 合并 Arguments（字符串拼接）
        var args strings.Builder
        toolID, toolType, toolName := "", "", ""
        
        for _, n := range v {
            chunk := chunks[n]
            if chunk.ID != "" {
                toolID = chunk.ID
            }
            if chunk.Type != "" {
                toolType = chunk.Type
            }
            if chunk.Function.Name != "" {
                toolName = chunk.Function.Name
            }
            args.WriteString(chunk.Function.Arguments)
        }

        merged = append(merged, ToolCall{
            Index: &k,
            ID:    toolID,
            Type:  toolType,
            Function: FunctionCall{
                Name:      toolName,
                Arguments: args.String(),
            },
        })
    }

    // 按 Index 排序
    sort.SliceStable(merged, func(i, j int) bool {
        if merged[i].Index == nil || merged[j].Index == nil {
            return merged[i].Index == nil
        }
        return *merged[i].Index < *merged[j].Index
    })

    return merged, nil
}
```

**使用示例**：
```go
// 示例 1：拼接流式文本消息
chunks := []*schema.Message{
    {Role: schema.Assistant, Content: "Eino "},
    {Role: schema.Assistant, Content: "是一个 "},
    {Role: schema.Assistant, Content: "Go 语言框架"},
}
fullMsg, _ := schema.ConcatMessages(chunks)
// fullMsg.Content = "Eino 是一个 Go 语言框架"

// 示例 2：拼接带工具调用的流式消息
chunks := []*schema.Message{
    {
        Role: schema.Assistant,
        ToolCalls: []schema.ToolCall{
            {Index: intPtr(0), ID: "call-1", Type: "function"},
        },
    },
    {
        Role: schema.Assistant,
        ToolCalls: []schema.ToolCall{
            {Index: intPtr(0), Function: schema.FunctionCall{Name: "get_weather"}},
        },
    },
    {
        Role: schema.Assistant,
        ToolCalls: []schema.ToolCall{
            {Index: intPtr(0), Function: schema.FunctionCall{Arguments: `{"city":"Beijing"}`}},
        },
    },
}
fullMsg, _ := schema.ConcatMessages(chunks)
// fullMsg.ToolCalls[0] = {Index: 0, ID: "call-1", Type: "function", 
//                          Function: {Name: "get_weather", Arguments: `{"city":"Beijing"}`}}

// 示例 3：从 StreamReader 拼接
stream, _ := chatModel.Stream(ctx, messages)
defer stream.Close()

var chunks []*schema.Message
for {
    chunk, err := stream.Recv()
    if err == io.EOF {
        break
    }
    chunks = append(chunks, chunk)
}
fullMsg, _ := schema.ConcatMessages(chunks)
```

**调用链路**：
```
用户代码
  ↓
schema.ConcatMessages(chunks)
  ↓
检查消息一致性（Role、Name 等）
  ↓
拼接 Content（使用 strings.Builder）
  ↓
合并 ToolCalls（调用 concatToolCalls）
  ↓
合并 ResponseMeta
  ↓
返回完整消息
```

**性能要点**：
- 使用 strings.Builder 预分配内存，避免频繁的字符串拼接开销
- 时间复杂度：O(n × m)，n 为消息数，m 为平均内容长度
- 空间复杂度：O(总内容长度)
- 建议：消息数 < 10000 时性能良好

**边界条件**：
- 所有消息的 Role 必须相同，否则返回错误
- 所有消息的 Name、ToolCallID 必须相同（如果有）
- nil 消息会返回错误
- 空数组返回空消息（不报错）

**最佳实践**：
- 主要用于处理 ChatModel 的流式输出
- 确保所有 chunk 来自同一次调用
- 使用 ConcatMessageStream 可以自动处理 StreamReader

#### 1.3.2 ConcatMessageStream

**功能说明**：从 StreamReader 读取并拼接所有消息为一条完整消息。

**函数签名**：
```go
func ConcatMessageStream(s *StreamReader[*Message]) (*Message, error)
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| s | *StreamReader[*Message] | 是 | 消息流读取器 |

**返回值**：

| 类型 | 说明 |
|------|------|
| *Message | 拼接后的完整消息 |
| error | 读取或拼接失败时返回错误 |

**核心代码**：
```go
// ConcatMessageStream 从流中读取并拼接消息
// 自动处理 Close()，避免资源泄漏
func ConcatMessageStream(s *StreamReader[*Message]) (*Message, error) {
    defer s.Close()  // 确保关闭流

    var msgs []*Message
    for {
        msg, err := s.Recv()
        if err != nil {
            if err == io.EOF {
                break  // 正常结束
            }
            return nil, err  // 异常
        }
        msgs = append(msgs, msg)
    }

    return ConcatMessages(msgs)
}
```

**使用示例**：
```go
// 示例 1：处理 ChatModel 的流式输出
stream, _ := chatModel.Stream(ctx, messages)
fullMsg, _ := schema.ConcatMessageStream(stream)
// stream 会自动关闭，无需手动 Close()

// 示例 2：带错误处理
stream, err := chatModel.Stream(ctx, messages)
if err != nil {
    return err
}
fullMsg, err := schema.ConcatMessageStream(stream)
if err != nil {
    return err
}
fmt.Println(fullMsg.Content)

// 对比：手动处理（不推荐）
stream, _ := chatModel.Stream(ctx, messages)
defer stream.Close()  // 需要记得 Close

var chunks []*schema.Message
for {
    chunk, err := stream.Recv()
    if err == io.EOF {
        break
    }
    chunks = append(chunks, chunk)
}
fullMsg, _ := schema.ConcatMessages(chunks)
```

**调用链路**：
```
用户代码
  ↓
schema.ConcatMessageStream(stream)
  ↓
循环调用 stream.Recv()
  ↓
收集所有消息到数组
  ↓
stream.Close() (defer 确保执行)
  ↓
schema.ConcatMessages(msgs)
  ↓
返回完整消息
```

**最佳实践**：
- 优先使用此函数处理流式消息，避免忘记 Close()
- 适用于需要完整消息才能处理的场景
- 如果需要逐 chunk 处理（如实时显示），应手动遍历流

---

## 2. Stream API

### 2.1 流创建 API

#### 2.1.1 Pipe

**功能说明**：创建一对流读写器（StreamReader 和 StreamWriter），用于流式数据传输。

**函数签名**：
```go
func Pipe[T any](cap int) (*StreamReader[T], *StreamWriter[T])
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| cap | int | 是 | 缓冲区容量（建议 5-100） |

**返回值**：

| 类型 | 说明 |
|------|------|
| *StreamReader[T] | 流读取器 |
| *StreamWriter[T] | 流写入器 |

**核心代码**：
```go
// Pipe 创建流管道
// 内部使用 channel 实现，支持缓冲
func Pipe[T any](cap int) (*StreamReader[T], *StreamWriter[T]) {
    stm := newStream[T](cap)
    return stm.asReader(), &StreamWriter[T]{stm: stm}
}

// newStream 创建底层流结构
func newStream[T any](cap int) *stream[T] {
    return &stream[T]{
        items:  make(chan streamItem[T], cap),  // 数据通道（带缓冲）
        closed: make(chan struct{}),            // 关闭信号通道
    }
}

type stream[T any] struct {
    items  chan streamItem[T]
    closed chan struct{}
    
    automaticClose bool
    closedFlag     *uint32
}

type streamItem[T any] struct {
    chunk T
    err   error
}
```

**使用示例**：
```go
// 示例 1：基本用法
sr, sw := schema.Pipe[string](10)

// 发送端（通常在 goroutine 中）
go func() {
    defer sw.Close()  // 确保关闭
    for i := 0; i < 5; i++ {
        sw.Send(fmt.Sprintf("chunk-%d", i), nil)
    }
}()

// 接收端
defer sr.Close()
for {
    chunk, err := sr.Recv()
    if err == io.EOF {
        break
    }
    fmt.Println(chunk)
}

// 示例 2：传递错误
sr, sw := schema.Pipe[string](5)

go func() {
    defer sw.Close()
    for i := 0; i < 3; i++ {
        if i == 2 {
            // 发送错误
            sw.Send("", fmt.Errorf("something wrong"))
            return
        }
        sw.Send(fmt.Sprintf("data-%d", i), nil)
    }
}()

defer sr.Close()
for {
    chunk, err := sr.Recv()
    if err == io.EOF {
        break
    }
    if err != nil {
        fmt.Println("Error:", err)
        break
    }
    fmt.Println(chunk)
}

// 示例 3：接收端提前关闭
sr, sw := schema.Pipe[int](10)

go func() {
    defer sw.Close()
    for i := 0; i < 100; i++ {
        closed := sw.Send(i, nil)
        if closed {
            fmt.Println("Stream closed by receiver")
            return  // 接收端已关闭，停止发送
        }
    }
}()

// 接收端只读取 5 个
count := 0
defer sr.Close()
for {
    _, err := sr.Recv()
    if err == io.EOF {
        break
    }
    count++
    if count >= 5 {
        break  // 提前退出，Close() 会通知发送端
    }
}
```

**调用链路**：
```
用户代码
  ↓
schema.Pipe[T](cap)
  ↓
newStream[T](cap)
  ↓
创建 channel: make(chan streamItem[T], cap)
  ↓
返回 (StreamReader, StreamWriter)
  ↓
发送端: sw.Send() → channel ← sr.Recv() :接收端
```

**性能要点**：
- 缓冲区大小影响性能：
  - 太小（<5）：发送端容易阻塞
  - 太大（>100）：占用内存多
  - 推荐：10-50 之间
- channel 操作的时间复杂度为 O(1)
- 内存占用：cap × sizeof(streamItem[T])

**边界条件**：
- cap 必须 ≥ 0，通常建议 ≥ 5
- 发送端必须 Close()，否则接收端会一直阻塞
- 接收端 Close() 会通知发送端停止发送

**最佳实践**：
- 发送端使用 `defer sw.Close()`
- 接收端使用 `defer sr.Close()`
- 发送端检查 Send() 的返回值，及时停止
- 缓冲区大小根据数据生产/消费速度调整

#### 2.1.2 StreamReaderFromArray

**功能说明**：从数组创建 StreamReader，将数组元素逐个作为流数据。

**函数签名**：
```go
func StreamReaderFromArray[T any](arr []T) *StreamReader[T]
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| arr | []T | 是 | 源数组 |

**返回值**：

| 类型 | 说明 |
|------|------|
| *StreamReader[T] | 流读取器 |

**核心代码**：
```go
// StreamReaderFromArray 从数组创建流
// 特点：不创建 goroutine，性能开销小
// 实现：基于数组的索引迭代
func StreamReaderFromArray[T any](arr []T) *StreamReader[T] {
    return &StreamReader[T]{
        ar:  &arrayReader[T]{arr: arr},
        typ: readerTypeArray,
    }
}

type arrayReader[T any] struct {
    arr   []T
    index int
}

// recv 从数组读取下一个元素
func (ar *arrayReader[T]) recv() (T, error) {
    if ar.index < len(ar.arr) {
        ret := ar.arr[ar.index]
        ar.index++
        return ret, nil
    }
    
    var t T
    return t, io.EOF  // 数组读完
}
```

**使用示例**：
```go
// 示例 1：从字符串数组创建流
arr := []string{"a", "b", "c"}
sr := schema.StreamReaderFromArray(arr)
defer sr.Close()

for {
    chunk, err := sr.Recv()
    if err == io.EOF {
        break
    }
    fmt.Println(chunk)
}
// 输出: a, b, c

// 示例 2：从消息数组创建流
messages := []*schema.Message{
    schema.UserMessage("问题1"),
    schema.AssistantMessage("回答1", nil),
    schema.UserMessage("问题2"),
}
sr := schema.StreamReaderFromArray(messages)
defer sr.Close()

for {
    msg, err := sr.Recv()
    if err == io.EOF {
        break
    }
    fmt.Println(msg.Content)
}

// 示例 3：用于测试
func TestMyFunction(t *testing.T) {
    // 创建测试数据流
    testData := []int{1, 2, 3, 4, 5}
    stream := schema.StreamReaderFromArray(testData)
    
    // 测试函数
    result, err := MyFunction(stream)
    // 断言...
}
```

**性能要点**：
- 不创建 goroutine，无并发开销
- 内存占用小，只有数组本身和一个索引
- 适合数据量不大且已在内存中的场景

**最佳实践**：
- 适用于测试场景（模拟流数据）
- 适用于将批量数据转换为流式处理
- 不适合大数据量（内存占用高）

---

### 2.2 流操作 API

#### 2.2.1 StreamReader.Recv

**功能说明**：从流中接收下一个数据块。

**函数签名**：
```go
func (sr *StreamReader[T]) Recv() (T, error)
```

**返回值**：

| 类型 | 说明 |
|------|------|
| T | 数据块 |
| error | 错误（io.EOF 表示流结束） |

**核心代码**：
```go
// Recv 接收下一个数据块
// 阻塞直到有数据或流关闭
func (sr *StreamReader[T]) Recv() (T, error) {
    switch sr.typ {
    case readerTypeStream:
        return sr.st.recv()
    case readerTypeArray:
        return sr.ar.recv()
    // 其他类型...
    }
}

// stream.recv 从 channel 接收
func (s *stream[T]) recv() (chunk T, err error) {
    item, ok := <-s.items
    if !ok {
        item.err = io.EOF  // channel 已关闭
    }
    return item.chunk, item.err
}
```

**使用示例**：见前面的示例

**最佳实践**：
- 总是检查错误
- 使用 `err == io.EOF` 判断正常结束
- 使用 `defer sr.Close()` 确保资源释放

#### 2.2.2 StreamReader.Close

**功能说明**：关闭流读取器，释放资源。

**函数签名**：
```go
func (sr *StreamReader[T]) Close()
```

**核心代码**：
```go
// Close 关闭流
// 通知发送端停止发送（如果还在发送）
func (sr *StreamReader[T]) Close() {
    switch sr.typ {
    case readerTypeStream:
        sr.st.closeRecv()
    case readerTypeMultiStream:
        sr.msr.close()
    // 其他类型...
    }
}

// stream.closeRecv 关闭接收端
func (s *stream[T]) closeRecv() {
    if s.automaticClose {
        if atomic.CompareAndSwapUint32(s.closedFlag, 0, 1) {
            close(s.closed)
        }
        return
    }
    close(s.closed)  // 通知发送端
}
```

**重要性**：
- 不 Close() 会导致 goroutine 泄漏
- 不 Close() 会导致发送端一直阻塞
- 必须在使用完流后调用

#### 2.2.3 StreamReader.Copy

**功能说明**：创建 n 个独立的流读取器，每个可独立读取。

**函数签名**：
```go
func (sr *StreamReader[T]) Copy(n int) []*StreamReader[T]
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| n | int | 是 | 复制数量（≥2） |

**返回值**：

| 类型 | 说明 |
|------|------|
| []*StreamReader[T] | 独立的流读取器数组 |

**核心代码**：
```go
// Copy 复制流读取器
// 原流失效，返回 n 个新的读取器
// 实现：使用共享链表，避免数据复制
func (sr *StreamReader[T]) Copy(n int) []*StreamReader[T] {
    if n < 2 {
        return []*StreamReader[T]{sr}
    }

    if sr.typ == readerTypeArray {
        // 数组类型直接复制索引
        ret := make([]*StreamReader[T], n)
        for i, ar := range sr.ar.copy(n) {
            ret[i] = &StreamReader[T]{typ: readerTypeArray, ar: ar}
        }
        return ret
    }

    return copyStreamReaders[T](sr, n)
}

// copyStreamReaders 创建子流读取器
// 使用链表共享数据
func copyStreamReaders[T any](sr *StreamReader[T], n int) []*StreamReader[T] {
    cpsr := &parentStreamReader[T]{
        sr:            sr,
        subStreamList: make([]*cpStreamElement[T], n),
        closedNum:     0,
    }

    // 初始化链表尾节点
    elem := &cpStreamElement[T]{}
    for i := range cpsr.subStreamList {
        cpsr.subStreamList[i] = elem
    }

    // 创建子流
    ret := make([]*StreamReader[T], n)
    for i := range ret {
        ret[i] = &StreamReader[T]{
            csr: &childStreamReader[T]{
                parent: cpsr,
                index:  i,
            },
            typ: readerTypeChild,
        }
    }

    return ret
}

type cpStreamElement[T any] struct {
    once sync.Once           // 确保只读取一次原流
    next *cpStreamElement[T] // 下一个元素
    item streamItem[T]       // 数据
}

// peek 读取数据（懒加载）
func (p *parentStreamReader[T]) peek(idx int) (t T, err error) {
    elem := p.subStreamList[idx]
    
    // 使用 sync.Once 确保只从原流读取一次
    elem.once.Do(func() {
        t, err = p.sr.Recv()
        elem.item = streamItem[T]{chunk: t, err: err}
        if err != io.EOF {
            elem.next = &cpStreamElement[T]{}  // 创建下一个节点
            p.subStreamList[idx] = elem.next
        }
    })

    return elem.item.chunk, elem.item.err
}
```

**使用示例**：
```go
// 示例 1：将流发送到多个处理器
sr, sw := schema.Pipe[string](10)

// 发送数据
go func() {
    defer sw.Close()
    for i := 0; i < 10; i++ {
        sw.Send(fmt.Sprintf("data-%d", i), nil)
    }
}()

// 复制为 3 个独立的读取器
readers := sr.Copy(3)
// 注意：sr 此时已失效，不能再使用

// 在不同 goroutine 中处理
var wg sync.WaitGroup
for i, reader := range readers {
    wg.Add(1)
    go func(idx int, r *schema.StreamReader[string]) {
        defer wg.Done()
        defer r.Close()
        
        for {
            chunk, err := r.Recv()
            if err == io.EOF {
                break
            }
            fmt.Printf("Reader %d: %s\n", idx, chunk)
        }
    }(i, reader)
}
wg.Wait()

// 示例 2：复制到 Callbacks 和主流程
func ProcessWithCallback(sr *schema.StreamReader[*schema.Message]) error {
    // 复制流：一个给 callback，一个给主流程
    readers := sr.Copy(2)
    callbackReader := readers[0]
    mainReader := readers[1]

    // callback 处理
    go func() {
        defer callbackReader.Close()
        for {
            msg, err := callbackReader.Recv()
            if err == io.EOF {
                break
            }
            // 发送到 callback
            callbacks.OnEndWithStreamOutput(ctx, msg)
        }
    }()

    // 主流程处理
    defer mainReader.Close()
    return processMainStream(mainReader)
}
```

**性能要点**：
- 使用链表共享数据，不复制实际内容
- 每个子流独立维护读取位置
- sync.Once 确保原流只读取一次
- 内存开销：O(n × 数据量)，n 为复制数量

**最佳实践**：
- 适用于需要多路处理同一流的场景
- 原流在 Copy 后失效，不要再使用
- 每个子流必须独立 Close()
- 建议 n ≤ 10，过多会影响性能

#### 2.2.4 MergeStreamReaders

**功能说明**：合并多个 StreamReader 为一个。

**函数签名**：
```go
func MergeStreamReaders[T any](srs []*StreamReader[T]) *StreamReader[T]
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| srs | []*StreamReader[T] | 是 | 待合并的流数组 |

**返回值**：

| 类型 | 说明 |
|------|------|
| *StreamReader[T] | 合并后的流 |

**核心代码**：
```go
// MergeStreamReaders 合并多个流
// 使用 reflect.Select 从多个 channel 读取
func MergeStreamReaders[T any](srs []*StreamReader[T]) *StreamReader[T] {
    if len(srs) < 1 {
        return nil
    }
    if len(srs) < 2 {
        return srs[0]
    }

    // 收集所有底层 stream
    var ss []*stream[T]
    for _, sr := range srs {
        switch sr.typ {
        case readerTypeStream:
            ss = append(ss, sr.st)
        case readerTypeArray:
            // 数组转为 stream
            ss = append(ss, arrToStream(sr.ar.arr[sr.ar.index:]))
        // 其他类型...
        }
    }

    return &StreamReader[T]{
        typ: readerTypeMultiStream,
        msr: newMultiStreamReader(ss),
    }
}

type multiStreamReader[T any] struct {
    sts        []*stream[T]
    itemsCases []reflect.SelectCase  // 用于 reflect.Select
    nonClosed  []int                 // 未关闭的流索引
}

// recv 从多个流中读取
// 使用 reflect.Select 实现
func (msr *multiStreamReader[T]) recv() (T, error) {
    for len(msr.nonClosed) > 0 {
        var chosen int
        var ok bool
        
        if len(msr.nonClosed) > maxSelectNum {
            // 流数量多时使用 reflect.Select
            var recv reflect.Value
            chosen, recv, ok = reflect.Select(msr.itemsCases)
            if ok {
                item := recv.Interface().(streamItem[T])
                return item.chunk, item.err
            }
        } else {
            // 流数量少时使用优化的 receiveN
            var item *streamItem[T]
            chosen, item, ok = receiveN(msr.nonClosed, msr.sts)
            if ok {
                return item.chunk, item.err
            }
        }

        // 移除已关闭的流
        for i := range msr.nonClosed {
            if msr.nonClosed[i] == chosen {
                msr.nonClosed = append(msr.nonClosed[:i], msr.nonClosed[i+1:]...)
                break
            }
        }
    }

    var t T
    return t, io.EOF  // 所有流都关闭
}
```

**使用示例**：
```go
// 示例 1：合并多个数据源
sr1, sw1 := schema.Pipe[int](5)
sr2, sw2 := schema.Pipe[int](5)
sr3, sw3 := schema.Pipe[int](5)

// 发送数据
go sendData(sw1, []int{1, 2, 3})
go sendData(sw2, []int{4, 5, 6})
go sendData(sw3, []int{7, 8, 9})

// 合并
merged := schema.MergeStreamReaders([]*schema.StreamReader[int]{sr1, sr2, sr3})
defer merged.Close()

// 读取（顺序不确定）
for {
    data, err := merged.Recv()
    if err == io.EOF {
        break
    }
    fmt.Println(data)  // 可能输出: 1, 4, 7, 2, 5, 8, 3, 6, 9 (顺序不定)
}

// 示例 2：合并并行任务的结果
func ParallelProcess(inputs []string) (*schema.StreamReader[string], error) {
    var streams []*schema.StreamReader[string]
    
    for _, input := range inputs {
        sr, sw := schema.Pipe[string](10)
        streams = append(streams, sr)
        
        go func(input string, sw *schema.StreamWriter[string]) {
            defer sw.Close()
            result := process(input)
            sw.Send(result, nil)
        }(input, sw)
    }
    
    return schema.MergeStreamReaders(streams), nil
}
```

**性能要点**：
- 流数量 ≤ 10 时性能好（使用优化的 receiveN）
- 流数量 > 10 时使用 reflect.Select，性能下降
- 建议：如果流很多，考虑分批合并

**最佳实践**：
- 适用于并行任务的结果合并
- 注意数据顺序是不确定的
- 合并后的流必须 Close()

#### 2.2.5 StreamReaderWithConvert

**功能说明**：转换流的数据类型。

**函数签名**：
```go
func StreamReaderWithConvert[T, D any](sr *StreamReader[T], convert func(T) (D, error)) *StreamReader[D]
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| sr | *StreamReader[T] | 是 | 源流 |
| convert | func(T) (D, error) | 是 | 转换函数 |

**返回值**：

| 类型 | 说明 |
|------|------|
| *StreamReader[D] | 转换后的流 |

**核心代码**：
```go
// StreamReaderWithConvert 转换流类型
// 内部维护原流，逐个转换数据
func StreamReaderWithConvert[T, D any](sr *StreamReader[T], convert func(T) (D, error)) *StreamReader[D] {
    c := func(a any) (D, error) {
        return convert(a.(T))
    }
    return newStreamReaderWithConvert(sr, c)
}

type streamReaderWithConvert[T any] struct {
    sr      iStreamReader          // 原流
    convert func(any) (T, error)   // 转换函数
}

// recv 接收并转换数据
func (srw *streamReaderWithConvert[T]) recv() (T, error) {
    for {
        out, err := srw.sr.recvAny()
        if err != nil {
            var t T
            return t, err
        }

        t, err := srw.convert(out)
        if err == nil {
            return t, nil
        }

        // 使用 ErrNoValue 可以跳过某个元素
        if !errors.Is(err, ErrNoValue) {
            return t, err
        }
    }
}
```

**使用示例**：
```go
// 示例 1：提取消息内容
messageStream, _ := chatModel.Stream(ctx, messages)

textStream := schema.StreamReaderWithConvert(messageStream, func(msg *schema.Message) (string, error) {
    if msg.Content == "" {
        return "", schema.ErrNoValue  // 跳过空消息
    }
    return msg.Content, nil
})

defer textStream.Close()
for {
    text, err := textStream.Recv()
    if err == io.EOF {
        break
    }
    fmt.Print(text)
}

// 示例 2：类型转换
intStream := schema.StreamReaderFromArray([]int{1, 2, 3, 4, 5})

stringStream := schema.StreamReaderWithConvert(intStream, func(n int) (string, error) {
    return fmt.Sprintf("num-%d", n), nil
})

// 示例 3：过滤和转换
dataStream, _ := source.GetDataStream()

filteredStream := schema.StreamReaderWithConvert(dataStream, func(data *Data) (*ProcessedData, error) {
    // 过滤条件
    if !data.IsValid() {
        return nil, schema.ErrNoValue  // 跳过无效数据
    }
    
    // 转换
    return &ProcessedData{
        Value: data.Value * 2,
        Time:  time.Now(),
    }, nil
})
```

**最佳实践**：
- 使用 `schema.ErrNoValue` 跳过不需要的元素
- 转换函数应该是纯函数（无副作用）
- 转换函数不应该太耗时，否则影响性能

---

### 2.3 StreamWriter API

#### 2.3.1 StreamWriter.Send

**功能说明**：向流发送数据块。

**函数签名**：
```go
func (sw *StreamWriter[T]) Send(chunk T, err error) (closed bool)
```

**参数说明**：

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| chunk | T | 是 | 数据块 |
| err | error | 否 | 错误（可以为 nil） |

**返回值**：

| 类型 | 说明 |
|------|------|
| bool | true 表示流已关闭（应停止发送） |

**核心代码**：
```go
// Send 发送数据
// 返回 true 表示接收端已关闭，应停止发送
func (sw *StreamWriter[T]) Send(chunk T, err error) (closed bool) {
    return sw.stm.send(chunk, err)
}

// stream.send 实现
func (s *stream[T]) send(chunk T, err error) (closed bool) {
    // 检查流是否已关闭
    select {
    case <-s.closed:
        return true  // 接收端已关闭
    default:
    }

    item := streamItem[T]{chunk, err}

    select {
    case <-s.closed:
        return true  // 再次检查
    case s.items <- item:
        return false  // 发送成功
    }
}
```

**使用示例**：
```go
// 示例 1：基本用法
sr, sw := schema.Pipe[string](10)

go func() {
    defer sw.Close()
    for i := 0; i < 5; i++ {
        closed := sw.Send(fmt.Sprintf("data-%d", i), nil)
        if closed {
            fmt.Println("Stream closed by receiver")
            return
        }
    }
}()

// 示例 2：发送错误
sr, sw := schema.Pipe[int](5)

go func() {
    defer sw.Close()
    for i := 0; i < 10; i++ {
        if i == 5 {
            // 发送错误并停止
            sw.Send(0, fmt.Errorf("error occurred"))
            return
        }
        sw.Send(i, nil)
    }
}()

// 示例 3：生产者-消费者
func Producer(sw *schema.StreamWriter[int], count int) {
    defer sw.Close()
    for i := 0; i < count; i++ {
        // 模拟耗时操作
        time.Sleep(100 * time.Millisecond)
        
        if sw.Send(i, nil) {
            log.Println("Consumer stopped, exiting producer")
            return
        }
    }
}
```

**最佳实践**：
- 检查返回值，及时停止发送
- 使用 `defer sw.Close()`
- 发生错误时，通过 err 参数传递给接收端

#### 2.3.2 StreamWriter.Close

**功能说明**：关闭流写入器，通知接收端流已结束。

**函数签名**：
```go
func (sw *StreamWriter[T]) Close()
```

**核心代码**：
```go
// Close 关闭写入器
// 关闭 channel，通知接收端 EOF
func (sw *StreamWriter[T]) Close() {
    sw.stm.closeSend()
}

// stream.closeSend 实现
func (s *stream[T]) closeSend() {
    close(s.items)  // 关闭 channel
}
```

**重要性**：
- 不 Close() 会导致接收端永远阻塞在 Recv()
- 必须在发送完所有数据后调用
- 使用 `defer sw.Close()` 确保调用

---

## 3. Tool API

### 3.1 ToolInfo 创建

**核心结构**：
```go
type ToolInfo struct {
    Name        string       // 工具名称
    Desc        string       // 工具描述
    Extra       map[string]any // 扩展信息
    *ParamsOneOf             // 参数定义
}
```

**使用示例**：
```go
// 示例 1：无参数工具
tool := &schema.ToolInfo{
    Name: "get_current_time",
    Desc: "获取当前时间",
}

// 示例 2：带参数工具
tool := &schema.ToolInfo{
    Name: "get_weather",
    Desc: "获取指定城市的天气信息",
    ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
        "city": {
            Type:     schema.String,
            Desc:     "城市名称",
            Required: true,
        },
        "unit": {
            Type:     schema.String,
            Desc:     "温度单位",
            Enum:     []string{"celsius", "fahrenheit"},
            Required: false,
        },
    }),
}

// 示例 3：复杂参数工具
tool := &schema.ToolInfo{
    Name: "search_database",
    Desc: "在数据库中搜索",
    ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
        "query": {
            Type:     schema.String,
            Desc:     "搜索查询",
            Required: true,
        },
        "filters": {
            Type: schema.Object,
            Desc: "过滤条件",
            SubParams: map[string]*schema.ParameterInfo{
                "category": {
                    Type: schema.String,
                    Desc: "类别",
                },
                "date_range": {
                    Type: schema.Array,
                    Desc: "日期范围",
                    ElemInfo: &schema.ParameterInfo{
                        Type: schema.String,
                    },
                },
            },
            Required: false,
        },
    }),
}
```

---

## 4. Document API

**核心结构**：
```go
type Document struct {
    ID       string
    Content  string
    MetaData map[string]any
}
```

**辅助方法使用示例**：
```go
doc := &schema.Document{
    ID:      "doc-1",
    Content: "文档内容",
}

// 设置评分
doc.WithScore(0.95)
score := doc.Score()  // 0.95

// 设置向量
doc.WithDenseVector([]float64{0.1, 0.2, 0.3})
vector := doc.DenseVector()

// 设置额外信息
doc.WithExtraInfo("这是一篇技术文档")
info := doc.ExtraInfo()
```

---

**文档版本**: v1.0  
**最后更新**: 2024-12-19  
**适用 Eino 版本**: main 分支（最新版本）

