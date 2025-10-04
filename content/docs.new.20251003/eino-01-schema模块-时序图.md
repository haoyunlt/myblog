# Eino-01-Schema模块-时序图

本文档通过时序图展示 Schema 模块在典型场景下的调用流程和数据流转。

---

## 1. 消息处理时序

### 1.1 创建和格式化消息

#### 1.1.1 基本消息创建流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Factory as 消息工厂函数
    participant Msg as Message对象

    User->>Factory: SystemMessage("你是助手")
    Factory->>Msg: 创建 Message
    Note over Msg: Role = System<br/>Content = "你是助手"
    Factory-->>User: 返回 *Message

    User->>Factory: UserMessage("问题: {query}")
    Factory->>Msg: 创建 Message
    Note over Msg: Role = User<br/>Content = "问题: {query}"
    Factory-->>User: 返回 *Message
```

**流程说明**:
1. 用户调用工厂函数（SystemMessage、UserMessage 等）
2. 工厂函数内部创建 Message 结构体
3. 设置对应的 Role 和 Content
4. 返回 Message 指针

**关键点**:
- 工厂函数不做复杂逻辑，仅创建结构体
- Content 可以包含模板变量（如 `{query}`）
- 创建的 Message 是不可变的（按惯例）

---

#### 1.1.2 消息模板渲染流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Msg as Message
    participant Format as formatContent函数
    participant Engine as 模板引擎
    
    User->>Msg: Format(ctx, params, FString)
    Note over User: params = {"query": "天气"}

    Msg->>Msg: 复制 Message
    Note over Msg: 避免修改原对象

    Msg->>Format: formatContent(Content, params, FString)
    Format->>Engine: pyfmt.Fmt(Content, params)
    Note over Engine: 替换 {query} -> "天气"
    Engine-->>Format: 返回渲染后的内容
    Format-->>Msg: 返回 "问题: 天气"

    Msg->>Msg: 设置 copied.Content
    
    alt MultiContent 不为空
        loop 遍历 MultiContent
            Msg->>Format: 渲染 Text/URL 等
            Format->>Engine: 调用模板引擎
            Engine-->>Format: 返回结果
            Format-->>Msg: 更新 MultiContent[i]
        end
    end

    Msg-->>User: 返回 []*Message
    Note over User: 返回渲染后的消息数组
```

**流程说明**:
1. 用户调用 Message.Format() 传入参数和模板类型
2. Message 创建自身的副本（避免修改原对象）
3. 调用 formatContent 渲染 Content 字段
4. 根据 FormatType 选择模板引擎（pyfmt/template/gonja）
5. 模板引擎替换变量
6. 如果有 MultiContent，遍历渲染每个 Part
7. 返回渲染后的新 Message

**关键点**:
- Format 不修改原 Message
- 支持三种模板格式：FString/GoTemplate/Jinja2
- MultiContent 中的 Text 和 URL 字段都会被渲染

---

### 1.2 流式消息拼接

#### 1.2.1 接收和拼接流式消息

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Stream as StreamReader
    participant Concat as ConcatMessages
    participant Builder as strings.Builder

    User->>Stream: Recv()
    Stream-->>User: chunk1: "Eino "
    
    User->>Stream: Recv()
    Stream-->>User: chunk2: "是一个 "
    
    User->>Stream: Recv()
    Stream-->>User: chunk3: "框架"
    
    User->>Stream: Recv()
    Stream-->>User: io.EOF
    
    Note over User: 收集所有 chunks

    User->>Concat: ConcatMessages(chunks)
    
    Concat->>Concat: 检查消息一致性
    Note over Concat: 验证 Role、Name 等

    Concat->>Builder: strings.Builder.Grow(总长度)
    Note over Builder: 预分配内存

    loop 遍历所有 chunks
        Concat->>Builder: WriteString(chunk.Content)
    end

    Concat->>Builder: builder.String()
    Builder-->>Concat: "Eino 是一个 框架"

    alt 包含 ToolCalls
        Concat->>Concat: concatToolCalls(toolCalls)
        Note over Concat: 按 Index 合并工具调用
    end

    alt 包含 ResponseMeta
        Concat->>Concat: 合并 TokenUsage
        Note over Concat: 取最大值
    end

    Concat->>Concat: 创建完整 Message
    Concat-->>User: 返回完整消息
```

**流程说明**:
1. 用户循环调用 StreamReader.Recv() 接收消息块
2. 遇到 io.EOF 表示流结束
3. 调用 ConcatMessages 拼接所有块
4. 检查消息一致性（Role、Name 必须相同）
5. 使用 strings.Builder 高效拼接 Content
6. 合并 ToolCalls（按 Index 分组）
7. 合并 ResponseMeta（取最大值）
8. 返回完整的 Message

**性能优化**:
- strings.Builder.Grow() 预分配内存，避免多次重新分配
- 时间复杂度：O(n)，n 为总字符数
- 空间复杂度：O(n)

**边界条件**:
- 所有消息的 Role 必须相同
- 所有消息的 Name 必须相同（如果有）
- nil 消息会返回错误

---

#### 1.2.2 工具调用拼接流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Concat as ConcatMessages
    participant ToolFunc as concatToolCalls
    
    User->>Concat: ConcatMessages(chunks)
    Note over User: chunks包含ToolCalls

    Concat->>ToolFunc: concatToolCalls(allToolCalls)
    
    ToolFunc->>ToolFunc: 创建 Index 映射
    Note over ToolFunc: map[Index][]ToolCall

    loop 遍历 ToolCalls
        alt ToolCall.Index == nil
            ToolFunc->>ToolFunc: 直接加入 merged
        else
            ToolFunc->>ToolFunc: 加入 map[Index]
        end
    end

    loop 遍历 Index 映射
        ToolFunc->>ToolFunc: 合并相同 Index 的 ToolCall
        Note over ToolFunc: ID、Type、Name 取第一个<br/>Arguments 字符串拼接
        ToolFunc->>ToolFunc: 添加到 merged
    end

    ToolFunc->>ToolFunc: 按 Index 排序
    Note over ToolFunc: Index 小的在前

    ToolFunc-->>Concat: 返回 merged ToolCalls
    Concat-->>User: 包含完整 ToolCalls 的 Message
```

**流程说明**:
1. 提取所有 ToolCalls
2. 按 Index 分组（nil 的直接加入结果）
3. 相同 Index 的 ToolCall 合并：
   - ID、Type、Name 取第一个非空值
   - Arguments 字符串拼接
4. 按 Index 排序
5. 返回合并后的 ToolCalls

**关键点**:
- Index 用于流式模式下的 chunk 合并
- Arguments 是 JSON 字符串，直接拼接即可
- ID、Type、Name 应该在同一 Index 的所有 chunk 中一致

---

## 2. 流处理时序

### 2.1 创建和使用流

#### 2.1.1 Pipe 创建流

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Pipe as Pipe函数
    participant Stream as stream结构
    participant Channel as Go channel

    User->>Pipe: Pipe[string](10)
    Note over User: 缓冲区大小 10

    Pipe->>Stream: newStream[string](10)
    Stream->>Channel: make(chan streamItem[string], 10)
    Note over Channel: 创建带缓冲的channel

    Stream->>Channel: make(chan struct{})
    Note over Channel: 创建关闭信号channel

    Stream-->>Pipe: 返回 *stream

    Pipe->>Pipe: 创建 StreamReader
    Pipe->>Pipe: 创建 StreamWriter

    Pipe-->>User: 返回 (StreamReader, StreamWriter)
```

**流程说明**:
1. 用户调用 Pipe[T](cap) 创建流
2. 内部创建 stream 结构
3. 创建两个 channel：
   - items: 数据通道（带缓冲）
   - closed: 关闭信号通道
4. 包装为 StreamReader 和 StreamWriter
5. 返回给用户

**关键点**:
- items channel 的缓冲区大小影响性能
- closed channel 用于接收端通知发送端停止

---

#### 2.1.2 发送和接收数据

```mermaid
sequenceDiagram
    autonumber
    participant Sender as 发送端
    participant SW as StreamWriter
    participant Channel as items channel
    participant ClosedChan as closed channel
    participant SR as StreamReader
    participant Receiver as 接收端

    par 发送数据
        Sender->>SW: Send("data1", nil)
        SW->>ClosedChan: select closed
        Note over ClosedChan: 检查是否已关闭
        ClosedChan-->>SW: 未关闭
        SW->>Channel: send streamItem{"data1", nil}
        SW-->>Sender: closed=false
    
        Sender->>SW: Send("data2", nil)
        SW->>Channel: send streamItem{"data2", nil}
        SW-->>Sender: closed=false
    end

    par 接收数据
        Receiver->>SR: Recv()
        SR->>Channel: <- items
        Channel-->>SR: streamItem{"data1", nil}
        SR-->>Receiver: ("data1", nil)

        Receiver->>SR: Recv()
        SR->>Channel: <- items
        Channel-->>SR: streamItem{"data2", nil}
        SR-->>Receiver: ("data2", nil)
    end

    Sender->>SW: Close()
    SW->>Channel: close(items)
    Note over Channel: 关闭数据通道

    Receiver->>SR: Recv()
    SR->>Channel: <- items
    Note over Channel: channel 已关闭
    SR-->>Receiver: (T{}, io.EOF)

    Receiver->>SR: Close()
    SR->>ClosedChan: close(closed)
    Note over ClosedChan: 通知发送端停止
```

**流程说明**:
1. 发送端循环调用 Send()
2. Send() 先检查 closed channel（接收端是否关闭）
3. 如果未关闭，发送数据到 items channel
4. 接收端循环调用 Recv()
5. Recv() 从 items channel 读取数据
6. 发送端完成后调用 Close()，关闭 items channel
7. 接收端收到 io.EOF
8. 接收端调用 Close()，关闭 closed channel

**关键点**:
- 发送和接收可以并发进行
- 发送端必须 Close() 通知结束
- 接收端 Close() 可以提前通知发送端停止

---

### 2.2 流的复制

#### 2.2.1 Copy 创建子流

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant SR as StreamReader
    participant Copy as copyStreamReaders
    participant Parent as parentStreamReader
    participant Child as childStreamReader

    User->>SR: Copy(3)
    Note over User: 创建3个子流

    SR->>Copy: copyStreamReaders(sr, 3)
    
    Copy->>Parent: 创建 parentStreamReader
    Note over Parent: sr = 原流<br/>subStreamList = [elem, elem, elem]

    Copy->>Parent: 初始化链表尾节点
    Note over Parent: elem = &cpStreamElement{}

    loop i = 0 to 2
        Copy->>Child: 创建 childStreamReader
        Note over Child: parent = Parent<br/>index = i
        Copy->>Copy: 包装为 StreamReader
    end

    Copy-->>User: 返回 []*StreamReader
    Note over User: 3个独立的子流

    Note over SR: 原流失效，不可再使用
```

**流程说明**:
1. 用户调用 StreamReader.Copy(n)
2. 创建 parentStreamReader 持有原流
3. 初始化 subStreamList（链表尾节点）
4. 创建 n 个 childStreamReader，每个持有：
   - parent 引用
   - 自己的 index
5. 原流失效

**内存结构**:
```
parentStreamReader
  ├─ sr: 原 StreamReader
  └─ subStreamList: [elem0, elem0, elem0]  (初始都指向同一尾节点)

3 个 childStreamReader:
  - child0: parent=Parent, index=0
  - child1: parent=Parent, index=1
  - child2: parent=Parent, index=2
```

---

#### 2.2.2 子流读取数据（懒加载）

```mermaid
sequenceDiagram
    autonumber
    participant Child1 as 子流1
    participant Child2 as 子流2
    participant Parent as parentStreamReader
    participant Elem as cpStreamElement
    participant Origin as 原流

    Child1->>Parent: peek(0)
    Parent->>Elem: elem.once.Do(...)
    
    Note over Elem: 第一次读取，执行 once

    Elem->>Origin: sr.Recv()
    Origin-->>Elem: ("data1", nil)
    
    Elem->>Elem: 保存到 item
    Note over Elem: item = {"data1", nil}
    
    Elem->>Elem: 创建下一个节点
    Note over Elem: next = &cpStreamElement{}
    
    Elem->>Parent: 更新 subStreamList[0]
    Note over Parent: subStreamList[0] = elem.next

    Parent-->>Child1: ("data1", nil)

    Note over Child2: 同时另一个子流也在读取

    Child2->>Parent: peek(1)
    Parent->>Elem: elem.once.Do(...)
    Note over Elem: once 已执行，直接返回

    Parent-->>Child2: ("data1", nil)
    Note over Parent: subStreamList[1] = elem.next

    Note over Child1,Child2: 两个子流都读到了 "data1"

    Child1->>Parent: peek(0)
    Note over Child1: 读取下一个

    Parent->>Elem: elem.next.once.Do(...)
    Note over Elem: 新节点，首次读取

    Elem->>Origin: sr.Recv()
    Origin-->>Elem: ("data2", nil)

    Parent-->>Child1: ("data2", nil)
```

**流程说明**:
1. 子流调用 parent.peek(index)
2. parent 获取 subStreamList[index] 指向的节点
3. 使用 sync.Once 确保节点只从原流读取一次
4. 首次读取时：
   - 从原流 Recv()
   - 保存数据到 item
   - 创建下一个节点（next）
   - 更新 subStreamList[index] 指向 next
5. 后续子流读取同一节点时，直接返回 item

**关键点**:
- sync.Once 确保原流只读取一次
- 每个子流独立维护读取位置
- 数据通过链表共享，不复制

---

### 2.3 流的合并

#### 2.3.1 MergeStreamReaders 合并流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Merge as MergeStreamReaders
    participant MSR as multiStreamReader
    participant Select as reflect.Select

    User->>Merge: MergeStreamReaders([sr1, sr2, sr3])
    
    Merge->>Merge: 收集所有底层 stream
    Note over Merge: 将 array/convert 等类型<br/>转换为 stream

    Merge->>MSR: newMultiStreamReader(streams)
    
    MSR->>MSR: 初始化 itemsCases
    Note over MSR: 为 reflect.Select 准备

    loop 为每个 stream 创建 SelectCase
        MSR->>MSR: itemsCases[i] = SelectCase{<br/>  Dir: SelectRecv,<br/>  Chan: stream.items<br/>}
    end

    MSR->>MSR: 初始化 nonClosed
    Note over MSR: nonClosed = [0, 1, 2, ...]

    MSR-->>Merge: 返回 multiStreamReader
    
    Merge->>Merge: 包装为 StreamReader
    Merge-->>User: 返回合并后的流

    User->>MSR: Recv()
    
    MSR->>Select: reflect.Select(itemsCases)
    Note over Select: 从多个channel中选择

    Select-->>MSR: (chosen=1, recv, ok=true)
    Note over MSR: 从 stream1 收到数据

    MSR->>MSR: 返回 recv 的数据

    User->>MSR: Recv()
    
    MSR->>Select: reflect.Select(itemsCases)
    Select-->>MSR: (chosen=0, recv, ok=true)
    Note over MSR: 从 stream0 收到数据

    User->>MSR: Recv()
    
    MSR->>Select: reflect.Select(itemsCases)
    Select-->>MSR: (chosen=2, recv, ok=false)
    Note over MSR: stream2 关闭

    MSR->>MSR: 从 nonClosed 移除 2
    Note over MSR: nonClosed = [0, 1]

    Note over User: 继续接收，直到所有流关闭
```

**流程说明**:
1. 用户传入多个 StreamReader
2. 提取所有底层 stream
3. 为每个 stream 创建 reflect.SelectCase
4. 使用 reflect.Select 从多个 channel 读取
5. 哪个 channel 有数据就返回哪个
6. 某个 channel 关闭时，从 nonClosed 列表移除
7. 所有 channel 关闭时返回 io.EOF

**性能特征**:
- 数据顺序不确定（哪个先到返回哪个）
- 流数量 ≤ 10 时使用优化版本（非 reflect）
- 流数量 > 10 时使用 reflect.Select

---

## 3. 工具调用时序

### 3.1 工具定义和调用

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant ToolInfo as ToolInfo
    participant ChatModel as ChatModel
    participant AssistMsg as AssistantMessage
    participant ToolsNode as ToolsNode
    participant ToolMsg as ToolMessage

    User->>ToolInfo: 创建 ToolInfo
    Note over ToolInfo: Name: "get_weather"<br/>Desc: "获取天气"<br/>ParamsOneOf: {...}

    User->>ChatModel: WithTools([toolInfo])
    Note over ChatModel: 绑定工具

    User->>ChatModel: Generate(ctx, messages)
    
    ChatModel->>ChatModel: 根据输入和工具定义生成回复
    Note over ChatModel: 决定是否调用工具

    alt 需要调用工具
        ChatModel->>AssistMsg: 创建带 ToolCalls 的消息
        Note over AssistMsg: ToolCalls: [{<br/>  ID: "call-123",<br/>  Function: {<br/>    Name: "get_weather",<br/>    Arguments: "{\"city\":\"Beijing\"}"<br/>  }<br/>}]
        ChatModel-->>User: 返回 AssistantMessage
        
        User->>ToolsNode: Invoke(ctx, toolCalls)
        
        loop 遍历 ToolCalls
            ToolsNode->>ToolsNode: 查找工具实现
            ToolsNode->>ToolsNode: 执行工具
            Note over ToolsNode: get_weather(city="Beijing")
            ToolsNode->>ToolMsg: 创建 ToolMessage
            Note over ToolMsg: Content: "晴天，25度"<br/>ToolCallID: "call-123"
        end
        
        ToolsNode-->>User: 返回 []*ToolMessage
        
        User->>ChatModel: Generate(ctx, messages + toolMessages)
        Note over User: 将工具结果加入对话
        
        ChatModel-->>User: 返回最终回复
    else 直接回复
        ChatModel->>AssistMsg: 创建纯文本消息
        ChatModel-->>User: 返回 AssistantMessage
    end
```

**流程说明**:
1. 用户创建 ToolInfo 定义工具
2. 通过 WithTools() 绑定到 ChatModel
3. ChatModel 根据输入决定是否调用工具
4. 如果需要调用：
   - 返回包含 ToolCalls 的 AssistantMessage
   - 用户执行工具（通过 ToolsNode）
   - 创建 ToolMessage 包含执行结果
   - 将 ToolMessage 加入对话继续
5. 如果不需要调用：
   - 直接返回文本回复

**关键点**:
- ToolCallID 关联 ToolCall 和 ToolMessage
- Arguments 是 JSON 字符串格式
- 工具结果需要再次传给 ChatModel

---

## 4. 模板渲染时序

### 4.1 多模板格式渲染

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Format as formatContent
    participant FString as pyfmt
    participant GoTpl as text/template
    participant Jinja as gonja

    alt FString 格式
        User->>Format: formatContent(content, vars, FString)
        Format->>FString: pyfmt.Fmt(content, vars)
        Note over FString: 替换 {var} 为实际值
        FString-->>Format: 渲染结果
        Format-->>User: 返回字符串
    end

    alt GoTemplate 格式
        User->>Format: formatContent(content, vars, GoTemplate)
        Format->>GoTpl: template.Parse(content)
        GoTpl-->>Format: 返回 *Template
        Format->>GoTpl: template.Execute(writer, vars)
        Note over GoTpl: 执行模板，支持条件/循环
        GoTpl-->>Format: 写入结果到 writer
        Format-->>User: 返回字符串
    end

    alt Jinja2 格式
        User->>Format: formatContent(content, vars, Jinja2)
        Format->>Jinja: getJinjaEnv()
        Note over Jinja: 获取全局 Jinja 环境
        Jinja-->>Format: 返回 *Environment
        Format->>Jinja: env.FromString(content)
        Jinja-->>Format: 返回 *Template
        Format->>Jinja: template.Execute(vars)
        Note over Jinja: 执行 Jinja2 模板
        Jinja-->>Format: 返回渲染结果
        Format-->>User: 返回字符串
    end
```

**流程说明**:
1. 根据 FormatType 选择模板引擎
2. FString: 使用 pyfmt 库，Python 风格
3. GoTemplate: 使用 Go 标准库
4. Jinja2: 使用 gonja 库，兼容 LangChain
5. 返回渲染后的字符串

**性能对比**:
- FString: 最快（简单替换）
- GoTemplate: 中等（需要解析和执行）
- Jinja2: 最慢（功能最强大）

---

## 5. 完整 RAG 场景时序

### 5.1 检索增强生成完整流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant App as 应用层
    participant Retriever as Retriever
    participant Doc as Document
    participant Template as ChatTemplate
    participant Msg as Message
    participant Model as ChatModel
    participant Stream as StreamReader

    User->>App: 提问 "Eino 是什么？"
    
    App->>Retriever: Retrieve(ctx, "Eino 是什么？")
    Note over Retriever: 向量检索
    
    loop 检索多个文档
        Retriever->>Doc: 创建 Document
        Note over Doc: ID, Content, Score
        Retriever->>Doc: WithScore(0.95)
        Retriever->>Doc: WithDenseVector([...])
    end
    
    Retriever-->>App: 返回 []*Document
    
    App->>App: 提取文档内容构建上下文
    Note over App: context = docs[0].Content +<br/>       docs[1].Content + ...
    
    App->>Template: Format(ctx, params)
    Note over App: params = {<br/>  "context": context,<br/>  "query": "Eino 是什么？"<br/>}
    
    Template->>Msg: 渲染消息模板
    Note over Msg: "根据以下上下文回答：<br/>{context}<br/>问题：{query}"
    
    Template-->>App: 返回 []*Message
    
    App->>Model: Stream(ctx, messages)
    
    Model->>Stream: 创建 StreamReader
    
    Model-->>App: 返回 StreamReader[*Message]
    
    loop 流式接收
        App->>Stream: Recv()
        Stream-->>App: Message chunk
        App->>User: 显示 chunk.Content
    end
    
    Stream-->>App: io.EOF
    
    App->>Stream: Close()
```

**流程说明**:
1. 用户提问
2. Retriever 向量检索相关文档
3. 提取文档内容构建上下文
4. 使用 ChatTemplate 渲染提示词
5. 调用 ChatModel 流式生成回复
6. 逐个接收并显示消息块
7. 关闭流

**涉及的数据结构**:
- Document: 存储检索结果
- Message: 构建对话上下文
- StreamReader[*Message]: 流式接收回复

---

**文档版本**: v1.0  
**最后更新**: 2024-12-19  
**适用 Eino 版本**: main 分支（最新版本）

