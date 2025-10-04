# Eino-02-Compose模块-时序图

本文档通过时序图展示 Compose 模块在典型场景下的编译和执行流程。

---

## 1. Chain 编译和执行时序

### 1.1 Chain 编译流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Chain as Chain
    participant Graph as 内部Graph
    participant Compiler as 编译器
    participant Runner as Runner

    User->>Chain: NewChain[I, O]()
    Chain->>Graph: 创建内部 Graph
    Graph-->>Chain: 返回 Graph

    User->>Chain: AppendChatModel("model", chatModel)
    Chain->>Graph: AddChatModelNode("node-0", chatModel)
    Note over Chain: nodeIdx++
    Chain->>Chain: 记录 preNodeKeys = ["node-0"]

    User->>Chain: AppendLambda("lambda", lambda)
    Chain->>Graph: AddLambdaNode("node-1", lambda)
    Chain->>Graph: AddEdge("node-0", "node-1")
    Chain->>Chain: 更新 preNodeKeys = ["node-1"]

    User->>Chain: Compile(ctx)
    
    Chain->>Chain: addEndIfNeeded()
    Note over Chain: 自动添加 END 边
    
    loop 遍历 preNodeKeys
        Chain->>Graph: AddEdge(nodeKey, END)
    end

    Chain->>Graph: compile(ctx, options)
    
    Graph->>Compiler: 类型检查
    Note over Compiler: 检查节点输入输出类型匹配
    
    Graph->>Compiler: 拓扑排序
    Note over Compiler: 检查是否有环
    
    Graph->>Runner: 创建 Runner
    Note over Runner: 构建执行引擎
    
    Runner-->>Graph: 返回 composableRunnable
    Graph-->>Chain: 返回 composableRunnable
    
    Chain->>Chain: 包装为 Runnable[I, O]
    Chain-->>User: 返回 Runnable
```

**流程说明**:
1. 创建 Chain 时内部创建 Graph
2. 每次 Append 操作添加节点和边
3. Chain 自动维护 preNodeKeys（上一批节点）
4. Compile 时自动添加到 END 的边
5. 调用 Graph 的 compile 方法
6. 进行类型检查和拓扑排序
7. 创建 Runner 执行引擎
8. 返回可执行的 Runnable

**关键点**:
- Chain 是 Graph 的语法糖，内部委托给 Graph
- 自动维护节点顺序，无需手动添加边
- 编译时进行静态类型检查

---

### 1.2 Chain Invoke 执行流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Runnable as Runnable
    participant Runner as Runner
    participant CB as Callbacks
    participant Node1 as Node1
    participant Node2 as Node2

    User->>Runnable: Invoke(ctx, input, opts)
    
    Runnable->>Runner: invoke(ctx, input, opts)
    
    Runner->>Runner: 解析 Options
    Note over Runner: 提取 Callbacks、<br/>组件配置等
    
    Runner->>Runner: 初始化执行上下文
    Note over Runner: 创建 channels、<br/>启动 goroutines
    
    Runner->>CB: 全局 OnStart
    
    Runner->>Node1: 执行 Node1
    activate Node1
    
    Node1->>CB: Node1 OnStart
    Note over Node1: 执行实际逻辑
    Node1->>CB: Node1 OnEnd
    Node1-->>Runner: 返回 output1
    deactivate Node1
    
    Runner->>Runner: 传递数据到 Node2
    Note over Runner: 通过 channel 传递
    
    Runner->>Node2: 执行 Node2
    activate Node2
    
    Node2->>CB: Node2 OnStart
    Note over Node2: 接收 output1<br/>执行实际逻辑
    Node2->>CB: Node2 OnEnd
    Node2-->>Runner: 返回 output2
    deactivate Node2
    
    Runner->>CB: 全局 OnEnd
    Runner-->>Runnable: 返回 output2
    Runnable-->>User: 返回 output2
```

**流程说明**:
1. 用户调用 Runnable.Invoke
2. Runner 解析 Options（Callbacks、配置等）
3. 初始化执行上下文（channels、goroutines）
4. 触发全局 OnStart 回调
5. 顺序执行各个节点
6. 每个节点执行前后触发回调
7. 通过 channel 在节点间传递数据
8. 最后返回输出并触发全局 OnEnd

**性能特点**:
- Chain 是顺序执行，无并发
- 节点间通过 channel 传递数据
- Callbacks 不阻塞主流程

---

## 2. Graph DAG 模式执行时序

### 2.1 Graph 编译流程（DAG 模式）

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Graph as Graph
    participant Validator as 验证器
    participant TopoSort as 拓扑排序
    participant Runner as Runner

    User->>Graph: Compile(ctx, WithDAGMode())
    
    Graph->>Validator: 验证 Graph 结构
    
    Validator->>Validator: 检查节点存在性
    Note over Validator: START/END 必须存在
    
    Validator->>Validator: 检查类型匹配
    Note over Validator: 节点间类型检查
    
    Validator->>Validator: 检查循环
    Note over Validator: DAG 模式不允许循环
    
    alt 发现循环
        Validator-->>Graph: 返回错误
        Graph-->>User: 编译失败
    end
    
    Validator-->>Graph: 验证通过
    
    Graph->>TopoSort: 拓扑排序
    Note over TopoSort: 确定执行顺序
    
    TopoSort->>TopoSort: 计算依赖关系
    Note over TopoSort: predecessors、successors
    
    TopoSort->>TopoSort: 分层
    Note over TopoSort: 同层节点可并发
    
    TopoSort-->>Graph: 返回执行计划
    
    Graph->>Runner: 创建 Runner(DAG)
    Note over Runner: dag=true<br/>eager=false
    
    Runner->>Runner: 构建 channel 网络
    Note over Runner: 为每个节点创建输入/输出 channel
    
    Runner-->>Graph: 返回 Runner
    Graph-->>User: 返回 Runnable
```

**DAG 模式特点**:
- 不允许循环
- 自动并发执行无依赖节点
- 使用拓扑排序确定执行顺序
- 通过 channel 网络传递数据

---

### 2.2 Graph DAG 并发执行时序

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Runner as Runner
    participant NodeA as Node A
    participant NodeB as Node B
    participant NodeC as Node C
    participant NodeD as Node D

    Note over User,NodeD: Graph 结构:<br/>START -> A -> C -> END<br/>START -> B -> C -> END<br/>(A 和 B 并发, C 等待两者)

    User->>Runner: Invoke(ctx, input)
    
    Runner->>Runner: 初始化 channels
    Note over Runner: 为每个节点创建 channel
    
    par 并发执行 A 和 B
        Runner->>NodeA: 启动 goroutine
        activate NodeA
        Note over NodeA: 接收 START 输入
        NodeA->>NodeA: 执行逻辑
        NodeA->>Runner: 发送结果到 channel
        deactivate NodeA
    and
        Runner->>NodeB: 启动 goroutine
        activate NodeB
        Note over NodeB: 接收 START 输入
        NodeB->>NodeB: 执行逻辑
        NodeB->>Runner: 发送结果到 channel
        deactivate NodeB
    end
    
    Note over Runner,NodeC: 等待 A 和 B 完成
    
    Runner->>NodeC: 启动 goroutine
    activate NodeC
    Note over NodeC: 接收 A 和 B 的输出
    NodeC->>NodeC: 合并输入并执行
    NodeC->>Runner: 发送结果到 channel
    deactivate NodeC
    
    Runner->>NodeD: END 节点
    activate NodeD
    Note over NodeD: 收集最终输出
    NodeD-->>Runner: 返回结果
    deactivate NodeD
    
    Runner-->>User: 返回输出
```

**并发特点**:
1. 无依赖的节点自动并发执行
2. 有依赖的节点等待所有前驱完成
3. 使用 channel 进行节点间通信
4. goroutine 数量等于节点数量

**性能优势**:
- 最大化并发度
- 减少总体执行时间
- 自动资源管理

---

## 3. Graph Pregel 模式执行时序

### 3.1 Pregel 迭代执行流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Runner as Runner
    participant Model as Model节点
    participant Branch as Branch判断
    participant Tools as Tools节点
    participant State as Graph State

    Note over User,State: ReAct Agent 示例<br/>模型 -> 判断 -> [工具 | 结束]<br/>工具 -> 模型 (循环)

    User->>Runner: Invoke(ctx, query)
    
    Runner->>State: 初始化状态
    Note over State: iteration = 0<br/>messages = []
    
    Runner->>Runner: 第 1 轮迭代
    
    Runner->>Model: 执行 Model
    activate Model
    Model->>State: 读取 messages
    Model->>Model: 生成回复
    Note over Model: 可能包含 ToolCalls
    Model->>State: 追加 AssistantMessage
    Model-->>Runner: 返回 Message
    deactivate Model
    
    Runner->>Branch: 执行 Branch
    activate Branch
    Branch->>Branch: 判断是否需要工具
    Note over Branch: len(ToolCalls) > 0?
    Branch-->>Runner: 返回 "tools"
    deactivate Branch
    
    Runner->>Tools: 执行 Tools
    activate Tools
    Tools->>Tools: 执行工具
    Note over Tools: 调用实际工具函数
    Tools->>State: 追加 ToolMessages
    Tools-->>Runner: 返回 ToolMessages
    deactivate Tools
    
    Runner->>Runner: 第 2 轮迭代
    Note over Runner: iteration = 1
    
    Runner->>Model: 再次执行 Model
    activate Model
    Model->>State: 读取 messages<br/>(包含工具结果)
    Model->>Model: 生成最终回复
    Model->>State: 追加 AssistantMessage
    Model-->>Runner: 返回 Message
    deactivate Model
    
    Runner->>Branch: 执行 Branch
    activate Branch
    Branch->>Branch: 判断是否需要工具
    Note over Branch: len(ToolCalls) == 0
    Branch-->>Runner: 返回 "end"
    deactivate Branch
    
    Runner->>Runner: 到达 END，停止迭代
    Runner-->>User: 返回最终结果
```

**Pregel 模式特点**:
1. 支持循环边
2. 迭代执行，每轮可以经过多个节点
3. 必须设置 MaxRunSteps 防止无限循环
4. 适合 Agent、工作流等场景

**迭代控制**:
- 达到 MaxRunSteps 时强制停止
- 到达 END 节点时停止
- 发生错误时停止

---

### 3.2 Pregel 最大迭代次数保护

```mermaid
sequenceDiagram
    autonumber
    participant Runner as Runner
    participant Node as 节点
    participant MaxSteps as 最大步数检查

    loop 迭代执行
        Runner->>MaxSteps: 检查 currentStep
        
        alt currentStep >= MaxRunSteps
            MaxSteps-->>Runner: 超出限制
            Runner->>Runner: 抛出错误
            Note over Runner: "reached max run steps"
            Runner-->>User: 返回错误
        else 继续执行
            MaxSteps-->>Runner: 继续
            
            Runner->>Node: 执行节点
            Node-->>Runner: 返回结果
            
            Runner->>Runner: currentStep++
            
            alt 到达 END
                Runner-->>User: 正常结束
            else 继续循环
                Note over Runner: 下一轮迭代
            end
        end
    end
```

**保护机制**:
- 每轮迭代递增计数器
- 达到上限时返回错误
- 避免无限循环导致资源耗尽

---

## 4. Workflow 执行时序

### 4.1 Workflow 依赖解析和执行

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant WF as Workflow
    participant FieldMap as 字段映射器
    participant Node1 as Node1
    participant Node2 as Node2
    participant Node3 as Node3

    Note over User,Node3: Workflow 结构:<br/>START -> Node1 (字段A)<br/>START -> Node2 (字段B)<br/>Node1,Node2 -> Node3

    User->>WF: Invoke(ctx, input)
    
    WF->>WF: 解析依赖关系
    Note over WF: Node1 依赖 START<br/>Node2 依赖 START<br/>Node3 依赖 Node1, Node2
    
    WF->>FieldMap: 解析字段映射
    Note over FieldMap: START.FieldA -> Node1.Input<br/>START.FieldB -> Node2.Input
    
    par 并发执行 Node1 和 Node2
        WF->>Node1: 执行 Node1
        activate Node1
        Node1->>FieldMap: 从 START 提取 FieldA
        FieldMap-->>Node1: FieldA 的值
        Node1->>Node1: 处理
        Node1-->>WF: 返回 Output1
        deactivate Node1
    and
        WF->>Node2: 执行 Node2
        activate Node2
        Node2->>FieldMap: 从 START 提取 FieldB
        FieldMap-->>Node2: FieldB 的值
        Node2->>Node2: 处理
        Node2-->>WF: 返回 Output2
        deactivate Node2
    end
    
    WF->>Node3: 执行 Node3
    activate Node3
    Node3->>FieldMap: 从 Node1 提取字段
    FieldMap-->>Node3: Output1 的某字段
    Node3->>FieldMap: 从 Node2 提取字段
    FieldMap-->>Node3: Output2 的某字段
    Node3->>Node3: 合并并处理
    Node3-->>WF: 返回 Output3
    deactivate Node3
    
    WF-->>User: 返回 Output3
```

**Workflow 特点**:
1. 显式声明依赖关系
2. 支持字段级数据映射
3. 自动并发执行无依赖节点
4. 不支持循环

**字段映射**:
- MapFields: 字段到字段映射
- MapKey: Map 的 key 映射
- StaticValue: 静态值注入

---

## 5. 分支执行时序

### 5.1 Graph 分支路由

```mermaid
sequenceDiagram
    autonumber
    participant Runner as Runner
    participant Source as 源节点
    participant Branch as 分支Lambda
    participant Target1 as 目标节点1
    participant Target2 as 目标节点2
    participant END as END节点

    Runner->>Source: 执行源节点
    Source-->>Runner: 返回 output
    
    Runner->>Branch: 执行分支判断
    Branch->>Branch: 判断逻辑
    Note over Branch: 根据 output 决定路由
    
    alt 条件1满足
        Branch-->>Runner: 返回 "target1"
        Runner->>Target1: 路由到 Target1
        activate Target1
        Target1->>Target1: 处理
        Target1-->>Runner: 返回结果
        deactivate Target1
    else 条件2满足
        Branch-->>Runner: 返回 "target2"
        Runner->>Target2: 路由到 Target2
        activate Target2
        Target2->>Target2: 处理
        Target2-->>Runner: 返回结果
        deactivate Target2
    else 其他
        Branch-->>Runner: 返回 "end"
        Runner->>END: 直接结束
    end
    
    Runner-->>User: 返回最终结果
```

**分支规则**:
- Branch Lambda 返回字符串（目标节点 key）
- pathMap 定义了返回值到节点的映射
- 必须覆盖所有可能的返回值

---

## 6. 流式执行时序

### 6.1 Stream 模式执行

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Runnable as Runnable
    participant Node1 as Node1
    participant Node2 as Node2(流式)
    participant SR as StreamReader

    User->>Runnable: Stream(ctx, input)
    
    Runnable->>Node1: 执行 Node1
    activate Node1
    Node1->>Node1: 处理输入
    Node1-->>Runnable: 返回 output1
    deactivate Node1
    
    Runnable->>Node2: 执行 Node2
    activate Node2
    Node2->>SR: 创建 StreamReader
    Node2->>Node2: 启动 goroutine
    
    Node2-->>Runnable: 返回 StreamReader
    deactivate Node2
    
    Runnable-->>User: 返回 StreamReader
    
    loop 用户读取流
        User->>SR: Recv()
        
        SR->>Node2: 从 channel 读取
        Note over Node2: 后台持续生成 chunks
        Node2-->>SR: 返回 chunk
        
        SR-->>User: 返回 chunk
    end
    
    User->>SR: Close()
    SR->>Node2: 通知关闭
    Note over Node2: 停止生成
```

**流式特点**:
- 支持逐块输出
- 后台 goroutine 持续生成数据
- 通过 channel 实现异步传输

---

### 6.2 流的自动拼接和复制

```mermaid
sequenceDiagram
    autonumber
    participant Graph as Graph
    participant StreamNode as 流式节点
    participant AutoConcat as 自动拼接
    participant NormalNode as 普通节点
    participant AutoCopy as 自动复制
    participant Target1 as 目标1
    participant Target2 as 目标2

    Note over Graph,Target2: 场景1: 流式输出 -> 普通输入

    Graph->>StreamNode: 执行
    StreamNode-->>Graph: StreamReader[T]
    
    Graph->>AutoConcat: 检测到类型不匹配
    Note over AutoConcat: 下游需要 T，不是 StreamReader[T]
    
    AutoConcat->>AutoConcat: 自动拼接流
    Note over AutoConcat: 读取所有 chunks<br/>合并为单个 T
    
    AutoConcat->>NormalNode: 传递拼接后的 T
    NormalNode-->>Graph: 继续执行
    
    Note over Graph,Target2: 场景2: 一个输出 -> 多个目标

    Graph->>StreamNode: 执行
    StreamNode-->>Graph: StreamReader[T]
    
    Graph->>AutoCopy: 检测到多个下游
    Note over AutoCopy: 需要发送到 Target1 和 Target2
    
    AutoCopy->>AutoCopy: 复制流
    Note over AutoCopy: readers = sr.Copy(2)
    
    par 并发传递
        AutoCopy->>Target1: 发送 readers[0]
        Target1-->>Graph: 处理完成
    and
        AutoCopy->>Target2: 发送 readers[1]
        Target2-->>Graph: 处理完成
    end
```

**自动处理**:
- **自动拼接**: 流输出连接普通输入时
- **自动复制**: 一个输出连接多个目标时
- 完全透明，用户无需关心

---

## 7. Callbacks 执行时序

### 7.1 完整的 Callbacks 调用流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Runner as Runner
    participant GlobalCB as 全局Callbacks
    participant NodeCB as 节点Callbacks
    participant Node as 节点

    User->>Runner: Invoke(ctx, input, WithCallbacks(...))
    
    Runner->>Runner: 收集 Callbacks
    Note over Runner: 全局 + 组件类型 + 节点
    
    Runner->>GlobalCB: OnStart(全局)
    GlobalCB-->>Runner: 修改后的 ctx
    
    Runner->>Node: 准备执行节点
    
    Runner->>NodeCB: OnStart(节点)
    NodeCB-->>Runner: 修改后的 ctx
    
    Runner->>Node: 执行节点
    activate Node
    
    alt 执行成功
        Node-->>Runner: 返回 output
        deactivate Node
        
        Runner->>NodeCB: OnEnd(节点)
        NodeCB-->>Runner: ctx
        
        Runner->>GlobalCB: OnEnd(全局)
        GlobalCB-->>Runner: ctx
        
    else 执行失败
        Node-->>Runner: 返回 error
        deactivate Node
        
        Runner->>NodeCB: OnError(节点)
        NodeCB-->>Runner: ctx
        
        Runner->>GlobalCB: OnError(全局)
        GlobalCB-->>Runner: ctx
        
        Runner-->>User: 返回 error
    end
    
    Runner-->>User: 返回 output
```

**Callbacks 顺序**:
1. 全局 OnStart
2. 节点 OnStart
3. 执行节点
4. 节点 OnEnd/OnError
5. 全局 OnEnd/OnError

---

## 8. 状态管理时序

### 8.1 Graph State 读写流程

```mermaid
sequenceDiagram
    autonumber
    participant Runner as Runner
    participant State as State对象
    participant PreHandler as 前置处理器
    participant Node as 节点
    participant PostHandler as 后置处理器

    Runner->>State: 初始化状态
    Note over State: stateGenerator(ctx)
    
    Runner->>PreHandler: 执行前置处理器
    PreHandler->>State: 读取状态（加锁）
    Note over State: 读取 state.Messages等
    PreHandler->>PreHandler: 修改 ctx
    PreHandler-->>Runner: 返回 ctx
    
    Runner->>Node: 执行节点（使用修改后的 ctx）
    Node-->>Runner: 返回 output
    
    Runner->>PostHandler: 执行后置处理器
    PostHandler->>State: 写入状态（加锁）
    Note over State: state.Messages.append(output)
    PostHandler-->>Runner: 完成
    
    Note over Runner,PostHandler: 下一个节点继续这个流程
```

**状态管理特点**:
- State 在所有节点间共享
- 读写通过锁保证线程安全
- PreHandler 读取状态，PostHandler 写入状态
- 适合需要保持上下文的场景（如 Agent）

---

## 9. 错误处理时序

### 9.1 节点错误传播

```mermaid
sequenceDiagram
    autonumber
    participant Runner as Runner
    participant Node1 as Node1
    participant Node2 as Node2
    participant ErrorHandler as 错误处理
    participant User as 用户

    Runner->>Node1: 执行 Node1
    activate Node1
    Node1->>Node1: 处理失败
    Node1-->>Runner: 返回 error
    deactivate Node1
    
    Runner->>ErrorHandler: 触发 OnError
    ErrorHandler->>ErrorHandler: 记录错误
    ErrorHandler->>ErrorHandler: 发送告警（可选）
    ErrorHandler-->>Runner: 返回
    
    Runner->>Runner: 停止执行
    Note over Runner: 不再执行后续节点
    
    Runner->>Runner: 清理资源
    Note over Runner: 关闭 channels、<br/>取消 goroutines
    
    Runner-->>User: 返回 error
```

**错误处理**:
- 任一节点失败，停止整个执行
- 触发 OnError 回调
- 自动清理资源
- 错误信息传播给用户

---

## 10. 性能优化时序示例

### 10.1 并发执行优化

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Graph as Graph(DAG模式)
    participant A as 独立节点A
    participant B as 独立节点B
    participant C as 独立节点C
    participant D as 汇总节点D

    Note over User,D: 优化前: 顺序执行 A->B->C->D<br/>优化后: 并发执行 (A,B,C) -> D

    User->>Graph: Invoke(ctx, input)
    
    par 并发执行 A, B, C
        Graph->>A: 启动 A
        activate A
        Note over A: 耗时 1s
        A-->>Graph: 返回结果A
        deactivate A
    and
        Graph->>B: 启动 B
        activate B
        Note over B: 耗时 1s
        B-->>Graph: 返回结果B
        deactivate B
    and
        Graph->>C: 启动 C
        activate C
        Note over C: 耗时 1s
        C-->>Graph: 返回结果C
        deactivate C
    end
    
    Note over Graph: 总耗时: ~1s (并发)<br/>vs 3s (顺序)
    
    Graph->>D: 执行 D
    activate D
    D->>D: 合并 A, B, C 的结果
    D-->>Graph: 返回最终结果
    deactivate D
    
    Graph-->>User: 返回结果
```

**性能提升**:
- 顺序执行: 1s + 1s + 1s + 处理时间 = 3s+
- 并发执行: max(1s, 1s, 1s) + 处理时间 = 1s+
- 提升 3 倍性能

---

**文档版本**: v1.0  
**最后更新**: 2024-12-19  
**适用 Eino 版本**: main 分支（最新版本）

