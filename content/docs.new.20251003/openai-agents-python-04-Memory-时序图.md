# OpenAI Agents Python SDK - Memory 模块时序图详解

## 1. 时序图总览

Memory 模块的时序图展示了会话历史管理的完整生命周期，从初始化、数据读写到资源清理的各个阶段，以及与 Agent 执行引擎的交互流程。

### 核心时序场景

| 场景类别 | 时序图 | 关键流程 |
|---------|--------|---------|
| **会话初始化** | SQLiteSession 初始化 | 数据库连接、表结构创建 |
| **历史加载** | get_items 查询流程 | SQL 查询、JSON 反序列化、数据返回 |
| **历史保存** | add_items 写入流程 | JSON 序列化、事务提交、时间戳更新 |
| **历史删除** | pop_item 删除流程 | 原子删除、数据返回 |
| **会话清空** | clear_session 清理流程 | 级联删除、资源释放 |
| **与 Agent 集成** | Runner + Session 交互 | 历史加载、执行、结果保存 |
| **云端会话** | OpenAI Conversations 流程 | API 调用、云端存储 |

## 2. SQLiteSession 初始化时序图

### 场景：创建持久化 SQLite 会话

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant SQLiteSession as SQLiteSession
    participant Threading as threading 模块
    participant SQLite as sqlite3 模块
    participant FileSystem as 文件系统
    participant DB as 数据库文件
    
    App->>SQLiteSession: __init__(session_id, db_path="data.db")
    
    SQLiteSession->>SQLiteSession: 保存配置参数
    Note over SQLiteSession: session_id = "user_123"<br/>db_path = "data.db"<br/>sessions_table = "agent_sessions"<br/>messages_table = "agent_messages"
    
    SQLiteSession->>SQLiteSession: 判断数据库模式
    Note over SQLiteSession: _is_memory_db = False<br/>（非内存数据库）
    
    SQLiteSession->>Threading: 创建 threading.local()
    Threading-->>SQLiteSession: _local 对象
    
    SQLiteSession->>Threading: 创建 threading.Lock()
    Threading-->>SQLiteSession: _lock 对象
    
    SQLiteSession->>FileSystem: 检查文件路径
    
    alt 文件不存在
        FileSystem-->>SQLiteSession: 路径可用
    else 文件已存在
        FileSystem-->>SQLiteSession: 文件存在
    end
    
    SQLiteSession->>SQLite: connect(db_path)
    SQLite->>DB: 打开/创建数据库文件
    DB-->>SQLite: 连接成功
    SQLite-->>SQLiteSession: init_conn
    
    SQLiteSession->>SQLite: execute("PRAGMA journal_mode=WAL")
    Note over SQLite: 启用 WAL 模式<br/>提高并发性能
    SQLite-->>SQLiteSession: WAL 模式已启用
    
    SQLiteSession->>SQLiteSession: _init_db_for_connection(init_conn)
    
    SQLiteSession->>SQLite: CREATE TABLE IF NOT EXISTS agent_sessions
    Note over SQLite: 创建会话表
    SQLite->>DB: 执行 DDL
    DB-->>SQLite: 表已创建/已存在
    
    SQLiteSession->>SQLite: CREATE TABLE IF NOT EXISTS agent_messages
    Note over SQLite: 创建消息表<br/>包含外键约束
    SQLite->>DB: 执行 DDL
    DB-->>SQLite: 表已创建/已存在
    
    SQLiteSession->>SQLite: CREATE INDEX IF NOT EXISTS idx_agent_messages_session_id
    Note over SQLite: 创建索引<br/>(session_id, created_at)
    SQLite->>DB: 执行 DDL
    DB-->>SQLite: 索引已创建/已存在
    
    SQLiteSession->>SQLite: commit()
    SQLite->>DB: 提交事务
    DB-->>SQLite: 事务已提交
    
    SQLiteSession->>SQLite: close() init_conn
    SQLite->>DB: 关闭初始化连接
    DB-->>SQLite: 连接已关闭
    
    SQLiteSession-->>App: SQLiteSession 实例
    Note over App: 会话已就绪<br/>可以进行读写操作
```

**时序图说明：**

### 初始化阶段划分

1. **参数配置阶段（步骤 1-5）**：
   - 接收初始化参数
   - 判断数据库模式（内存/文件）
   - 创建线程安全机制（锁和本地存储）

2. **数据库连接阶段（步骤 6-11）**：
   - 检查文件路径可用性
   - 打开或创建数据库文件
   - 启用 WAL 模式提升并发性能

3. **表结构初始化阶段（步骤 12-20）**：
   - 创建会话元数据表
   - 创建消息数据表（带外键）
   - 创建查询索引
   - 提交数据库结构

4. **清理阶段（步骤 21-23）**：
   - 关闭初始化连接
   - 返回可用的会话实例

### 关键设计点

1. **延迟连接**：初始化时不保留连接，实际操作时创建线程本地连接
2. **幂等性**：使用 `IF NOT EXISTS` 确保重复初始化不会出错
3. **WAL 模式**：提高并发读写性能，允许读写同时进行
4. **索引优化**：复合索引 `(session_id, created_at)` 加速查询

### 内存模式差异

**内存模式初始化（db_path=":memory:"）：**
```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant SQLiteSession as SQLiteSession
    participant SQLite as sqlite3 模块
    
    App->>SQLiteSession: __init__(session_id, db_path=":memory:")
    
    SQLiteSession->>SQLiteSession: _is_memory_db = True
    
    SQLiteSession->>SQLite: connect(":memory:", check_same_thread=False)
    Note over SQLite: 创建内存数据库<br/>允许跨线程访问
    SQLite-->>SQLiteSession: _shared_connection
    
    SQLiteSession->>SQLite: execute("PRAGMA journal_mode=WAL")
    SQLiteSession->>SQLiteSession: _init_db_for_connection(_shared_connection)
    
    Note over SQLiteSession: 保留共享连接<br/>不关闭连接
    
    SQLiteSession-->>App: SQLiteSession 实例
```

**内存模式关键差异：**
- 使用共享连接（`_shared_connection`）而非线程本地连接
- 连接在初始化时创建并保留，直到调用 `close()`
- 需要使用全局锁（`_lock`）保护并发访问

## 3. get_items - 历史查询时序图

### 场景：加载会话历史用于 Agent 执行

```mermaid
sequenceDiagram
    autonumber
    participant Runner as Runner.run()
    participant Session as SQLiteSession
    participant AsyncIO as asyncio
    participant ThreadPool as 线程池
    participant SQLite as sqlite3
    participant DB as 数据库
    participant JSON as json 模块
    
    Runner->>Session: await get_items(limit=100)
    Note over Runner: 请求最新100条历史
    
    Session->>Session: 定义 _get_items_sync() 内部函数
    
    Session->>AsyncIO: asyncio.to_thread(_get_items_sync)
    Note over AsyncIO: 转换同步函数为异步执行
    
    AsyncIO->>ThreadPool: 提交到线程池执行
    ThreadPool->>Session: 执行 _get_items_sync()
    
    Session->>Session: _get_connection()
    
    alt 文件模式
        Session->>Session: 检查线程本地连接
        
        alt 连接不存在
            Session->>SQLite: connect(db_path)
            SQLite->>DB: 建立连接
            DB-->>SQLite: 连接对象
            SQLite-->>Session: conn
            Session->>Session: 保存到 _local.connection
            Session->>SQLite: execute("PRAGMA journal_mode=WAL")
        else 连接已存在
            Session->>Session: 返回 _local.connection
        end
    else 内存模式
        Session->>Session: 返回 _shared_connection
    end
    
    Session->>Session: 获取锁（内存模式）或新建锁（文件模式）
    Note over Session: 进入临界区
    
    alt limit 指定
        Session->>SQLite: execute(SELECT ... ORDER BY created_at DESC LIMIT ?)
        Note over SQLite: 查询最新的 N 条记录
        SQLite->>DB: 执行 SQL
        DB-->>SQLite: 结果集（倒序）
        SQLite-->>Session: rows
        
        Session->>Session: rows = list(reversed(rows))
        Note over Session: 反转为时间正序
    else limit 为 None
        Session->>SQLite: execute(SELECT ... ORDER BY created_at ASC)
        Note over SQLite: 查询所有记录（正序）
        SQLite->>DB: 执行 SQL
        DB-->>SQLite: 结果集
        SQLite-->>Session: rows
    end
    
    Session->>Session: 释放锁
    Note over Session: 退出临界区
    
    Session->>Session: items = []
    
    loop 遍历每一行
        Session->>Session: 提取 message_data
        
        Session->>JSON: json.loads(message_data)
        
        alt JSON 解析成功
            JSON-->>Session: item 对象
            Session->>Session: items.append(item)
        else JSON 解析失败
            JSON-->>Session: JSONDecodeError
            Note over Session: 跳过损坏的数据<br/>不中断查询
        end
    end
    
    Session-->>ThreadPool: return items
    ThreadPool-->>AsyncIO: 返回结果
    AsyncIO-->>Session: 异步返回
    Session-->>Runner: list[TResponseInputItem]
    
    Runner->>Runner: 使用历史数据构建上下文
    Note over Runner: 历史加载完成<br/>准备执行 Agent
```

**时序图说明：**

### 查询流程阶段

1. **异步转换阶段（步骤 1-5）**：
   - Runner 调用异步 `get_items()`
   - 使用 `asyncio.to_thread()` 包装同步操作
   - 提交到线程池执行，避免阻塞事件循环

2. **连接获取阶段（步骤 6-14）**：
   - 获取适当的数据库连接
   - 文件模式：获取或创建线程本地连接
   - 内存模式：使用共享连接

3. **SQL 查询阶段（步骤 15-27）**：
   - 根据 `limit` 参数选择查询策略
   - 执行 SQL 查询
   - 处理查询结果的顺序

4. **数据反序列化阶段（步骤 28-38）**：
   - 遍历查询结果
   - JSON 反序列化每一行
   - 错误容忍：跳过损坏的数据

5. **结果返回阶段（步骤 39-42）**：
   - 从线程池返回结果
   - 通过 asyncio 转换为异步返回
   - Runner 接收历史数据

### 查询性能优化

**索引利用：**
```sql
-- 索引定义
CREATE INDEX idx_agent_messages_session_id
    ON agent_messages (session_id, created_at);

-- 查询计划（limit 指定时）
EXPLAIN QUERY PLAN
SELECT message_data FROM agent_messages
WHERE session_id = ?
ORDER BY created_at DESC
LIMIT 100;

-- 结果：使用索引扫描
SEARCH agent_messages USING INDEX idx_agent_messages_session_id (session_id=?)
```

**批量反序列化：**
- 一次性获取所有行，减少数据库交互
- 在内存中批量反序列化，提升效率
- 错误跳过机制保证健壮性

## 4. add_items - 历史保存时序图

### 场景：执行完成后保存对话历史

```mermaid
sequenceDiagram
    autonumber
    participant Runner as Runner.run()
    participant Session as SQLiteSession
    participant AsyncIO as asyncio
    participant ThreadPool as 线程池
    participant JSON as json 模块
    participant SQLite as sqlite3
    participant DB as 数据库
    
    Runner->>Runner: 执行完成，生成新的历史项
    Note over Runner: new_items = [user_msg, assistant_msg, tool_call, tool_output]
    
    Runner->>Session: await add_items(new_items)
    
    Session->>Session: 检查 items 列表
    
    alt items 为空
        Session-->>Runner: return（无操作）
    end
    
    Session->>Session: 定义 _add_items_sync() 内部函数
    
    Session->>AsyncIO: asyncio.to_thread(_add_items_sync)
    AsyncIO->>ThreadPool: 提交到线程池
    ThreadPool->>Session: 执行 _add_items_sync()
    
    Session->>Session: conn = _get_connection()
    Note over Session: 获取数据库连接<br/>（文件模式：线程本地<br/>内存模式：共享连接）
    
    Session->>Session: 获取锁
    Note over Session: 进入临界区<br/>保证事务原子性
    
    Session->>SQLite: BEGIN TRANSACTION
    Note over SQLite: 开始事务
    
    Session->>SQLite: INSERT OR IGNORE INTO agent_sessions<br/>(session_id) VALUES (?)
    Note over Session: 确保会话记录存在
    SQLite->>DB: 执行 INSERT
    
    alt 会话不存在
        DB-->>SQLite: 插入新会话记录
    else 会话已存在
        DB-->>SQLite: 忽略（IGNORE）
    end
    
    Session->>Session: 准备批量数据
    Note over Session: message_data = [<br/>  (session_id, json.dumps(item1)),<br/>  (session_id, json.dumps(item2)),<br/>  ...<br/>]
    
    loop 对每个 item
        Session->>JSON: json.dumps(item)
        JSON-->>Session: JSON 字符串
        Session->>Session: 添加到 message_data 列表
    end
    
    Session->>SQLite: executemany(<br/>  INSERT INTO agent_messages<br/>  (session_id, message_data)<br/>  VALUES (?, ?),<br/>  message_data<br/>)
    Note over SQLite: 批量插入消息记录
    SQLite->>DB: 批量执行 INSERT
    DB-->>SQLite: N 行已插入
    
    Session->>SQLite: UPDATE agent_sessions<br/>SET updated_at = CURRENT_TIMESTAMP<br/>WHERE session_id = ?
    Note over Session: 更新会话时间戳
    SQLite->>DB: 执行 UPDATE
    DB-->>SQLite: 更新成功
    
    Session->>SQLite: COMMIT
    Note over SQLite: 提交事务
    SQLite->>DB: 事务提交
    
    alt 事务成功
        DB-->>SQLite: 提交成功
        SQLite-->>Session: 返回成功
    else 事务失败
        DB-->>SQLite: 错误
        SQLite->>DB: ROLLBACK
        Note over SQLite: 自动回滚
        SQLite-->>Session: 抛出异常
    end
    
    Session->>Session: 释放锁
    Note over Session: 退出临界区
    
    Session-->>ThreadPool: return（无返回值）
    ThreadPool-->>AsyncIO: 返回
    AsyncIO-->>Session: 异步返回
    Session-->>Runner: 完成
    
    Runner->>Runner: 历史保存成功
    Note over Runner: 继续执行后续逻辑
```

**时序图说明：**

### 写入流程阶段

1. **预处理阶段（步骤 1-7）**：
   - Runner 准备要保存的历史项
   - 检查列表是否为空（快速路径）
   - 异步转换并提交到线程池

2. **连接与锁获取阶段（步骤 8-10）**：
   - 获取数据库连接
   - 获取锁保证事务原子性
   - 开始数据库事务

3. **会话确保阶段（步骤 11-16）**：
   - 使用 `INSERT OR IGNORE` 确保会话存在
   - 避免外键约束错误

4. **数据序列化阶段（步骤 17-22）**：
   - 批量序列化所有历史项为 JSON
   - 准备批量插入的数据

5. **批量插入阶段（步骤 23-26）**：
   - 使用 `executemany()` 批量插入
   - 提升插入性能

6. **时间戳更新阶段（步骤 27-29）**：
   - 更新会话的最后更新时间
   - 用于会话管理和查询

7. **事务提交阶段（步骤 30-39）**：
   - 提交事务
   - 失败时自动回滚
   - 释放锁并返回结果

### 事务原子性保证

**成功场景：**
```
BEGIN TRANSACTION
  → INSERT OR IGNORE sessions
  → INSERT messages (批量)
  → UPDATE sessions timestamp
COMMIT
  → 所有操作生效
```

**失败场景：**
```
BEGIN TRANSACTION
  → INSERT OR IGNORE sessions (成功)
  → INSERT messages (失败，如磁盘满)
ROLLBACK
  → 所有操作回滚，数据保持一致
```

### 批量插入性能

**单条插入（低效）：**
```python
for item in items:
    conn.execute("INSERT INTO messages VALUES (?, ?)", 
                 (session_id, json.dumps(item)))
    conn.commit()  # 每次都提交
# 时间复杂度：O(n * (序列化 + SQL + 磁盘I/O + 提交))
```

**批量插入（高效）：**
```python
message_data = [(session_id, json.dumps(item)) for item in items]
conn.executemany("INSERT INTO messages VALUES (?, ?)", message_data)
conn.commit()  # 一次提交
# 时间复杂度：O(n * 序列化 + SQL批处理 + 一次磁盘I/O + 一次提交)
```

## 5. Runner 与 Session 集成时序图

### 场景：完整的对话执行流程（带历史管理）

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant Runner as Runner
    participant Session as Session
    participant Agent as Agent
    participant Model as Model (LLM)
    participant Tools as Tools
    
    App->>Runner: await run(agent, "今天天气怎么样?", session)
    
    Runner->>Session: await get_items()
    Note over Runner,Session: 加载会话历史
    Session->>Session: 查询数据库
    Session-->>Runner: 返回历史列表
    
    Runner->>Runner: 构建完整上下文
    Note over Runner: context = [<br/>  system_message,<br/>  ...history...,<br/>  new_user_message<br/>]
    
    Runner->>Agent: 准备执行
    Agent->>Model: 调用 LLM
    Note over Agent,Model: 传递完整上下文<br/>包含历史对话
    
    Model-->>Agent: 返回响应（包含工具调用）
    Note over Model: {<br/>  "content": "让我查询天气",<br/>  "tool_calls": [{<br/>    "name": "get_weather",<br/>    "arguments": "..."<br/>  }]<br/>}
    
    Agent->>Runner: 处理响应
    Runner->>Runner: 创建 MessageOutputItem
    
    Agent->>Tools: 执行工具调用 get_weather
    Tools->>Tools: 查询天气 API
    Tools-->>Agent: 返回天气数据
    
    Runner->>Runner: 创建 ToolCallItem 和 ToolCallOutputItem
    
    Agent->>Model: 再次调用 LLM（带工具结果）
    Model-->>Agent: 最终响应
    Note over Model: "今天北京晴天，22度"
    
    Runner->>Runner: 创建最终 MessageOutputItem
    
    Runner->>Runner: 收集所有新增的历史项
    Note over Runner: new_items = [<br/>  user_message,<br/>  assistant_message_1,<br/>  tool_call,<br/>  tool_output,<br/>  assistant_message_2<br/>]
    
    Runner->>Session: await add_items(new_items)
    Note over Runner,Session: 保存本轮对话历史
    Session->>Session: 序列化并存储
    Session-->>Runner: 保存成功
    
    Runner-->>App: 返回 RunResult
    Note over App: result.final_output =<br/>"今天北京晴天，22度"
    
    App->>App: 显示结果给用户
```

**时序图说明：**

### 完整执行流程

1. **历史加载阶段（步骤 1-4）**：
   - Runner 启动时从 Session 加载历史
   - 历史用于构建 LLM 的上下文
   - 确保对话连贯性

2. **上下文构建阶段（步骤 5）**：
   - 系统消息（Agent 指令）
   - 历史对话记录
   - 新的用户消息
   - 按时间顺序组织

3. **LLM 交互阶段（步骤 6-10）**：
   - 调用 LLM 生成响应
   - LLM 基于完整历史生成回复
   - 可能包含工具调用请求

4. **工具执行阶段（步骤 11-15）**：
   - 执行工具调用
   - 获取工具结果
   - 创建工具相关的历史项

5. **最终响应阶段（步骤 16-18）**：
   - 带工具结果再次调用 LLM
   - 生成最终用户可见的回复
   - 创建最终响应历史项

6. **历史保存阶段（步骤 19-22）**：
   - 收集本轮所有新增历史
   - 批量保存到 Session
   - 确保数据持久化

7. **结果返回阶段（步骤 23-25）**：
   - 返回执行结果给应用
   - 应用展示给用户
   - 准备下一轮对话

### 历史在对话中的作用

**第一轮对话（无历史）：**
```
Context = [
  {role: "system", content: "你是一个天气助手"},
  {role: "user", content: "今天天气怎么样?"}
]
```

**第二轮对话（有历史）：**
```
Context = [
  {role: "system", content: "你是一个天气助手"},
  {role: "user", content: "今天天气怎么样?"},
  {role: "assistant", content: "今天北京晴天，22度"},
  {role: "user", content: "那明天呢?"}  // 新消息
]

LLM 理解 "明天" 是基于上一轮的 "今天北京" 的上下文
```

## 6. OpenAI Conversations 云端会话时序图

### 场景：使用 OpenAI 云端存储的会话管理

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant Session as OpenAIConversationsSession
    participant OpenAI as OpenAI API
    participant Cloud as OpenAI 云端存储
    
    App->>Session: 创建 OpenAIConversationsSession()
    Note over Session: conversation_id = None<br/>（延迟初始化）
    
    App->>Session: await add_items([user_message])
    
    Session->>Session: await _get_session_id()
    
    alt conversation_id 为 None
        Session->>Session: 调用 start_openai_conversations_session()
        Session->>OpenAI: conversations.create(items=[])
        Note over Session,OpenAI: 创建新的云端会话
        
        OpenAI->>Cloud: 创建会话记录
        Cloud-->>OpenAI: 会话已创建
        OpenAI-->>Session: conversation_id = "conv_abc123"
        
        Session->>Session: _session_id = "conv_abc123"
        Note over Session: 保存会话 ID
    else conversation_id 已存在
        Session->>Session: 返回现有 conversation_id
    end
    
    Session->>OpenAI: conversations.items.create(<br/>  conversation_id="conv_abc123",<br/>  items=[user_message]<br/>)
    Note over Session,OpenAI: 添加历史项到云端
    
    OpenAI->>Cloud: 存储历史项
    Cloud-->>OpenAI: 存储成功
    OpenAI-->>Session: 成功响应
    Session-->>App: 完成
    
    Note over App,Cloud: === 后续对话 ===
    
    App->>Session: await get_items(limit=50)
    
    Session->>Session: await _get_session_id()
    Note over Session: conversation_id = "conv_abc123"<br/>（已存在）
    
    Session->>OpenAI: conversations.items.list(<br/>  conversation_id="conv_abc123",<br/>  limit=50,<br/>  order="desc"<br/>)
    Note over Session,OpenAI: 请求最新50条历史
    
    OpenAI->>Cloud: 查询历史记录
    Cloud-->>OpenAI: 返回历史数据
    
    OpenAI-->>Session: 历史列表（倒序）
    Note over OpenAI,Session: [item_50, item_49, ..., item_1]
    
    Session->>Session: all_items.reverse()
    Note over Session: 反转为时间正序<br/>[item_1, item_2, ..., item_50]
    
    Session-->>App: list[TResponseInputItem]
    
    Note over App,Cloud: === 清空会话 ===
    
    App->>Session: await clear_session()
    
    Session->>Session: await _get_session_id()
    Session->>OpenAI: conversations.delete(<br/>  conversation_id="conv_abc123"<br/>)
    
    OpenAI->>Cloud: 删除会话及所有历史
    Cloud-->>OpenAI: 删除成功
    OpenAI-->>Session: 成功响应
    
    Session->>Session: await _clear_session_id()
    Note over Session: _session_id = None<br/>（重置会话 ID）
    
    Session-->>App: 完成
```

**时序图说明：**

### 云端会话特点

1. **延迟初始化（步骤 1-12）**：
   - 创建实例时不立即创建云端会话
   - 首次使用时才调用 API 创建会话
   - 节省不必要的 API 调用

2. **云端存储（步骤 13-18）**：
   - 所有数据存储在 OpenAI 服务器
   - 无本地数据库或文件
   - 自动处理数据持久化和备份

3. **数据查询（步骤 20-29）**：
   - 通过 API 查询历史数据
   - API 返回倒序数据，需要客户端反转
   - 支持分页和限制查询

4. **会话清理（步骤 31-40）**：
   - 删除云端会话及所有关联数据
   - 重置本地会话 ID
   - 下次使用时会创建新会话

### 云端 vs 本地存储对比

| 特性 | SQLiteSession | OpenAIConversationsSession |
|------|---------------|---------------------------|
| **数据位置** | 本地文件/内存 | OpenAI 云端 |
| **持久化** | 文件模式持久化 | 自动持久化 |
| **网络依赖** | 无 | 需要网络连接 |
| **跨设备** | 不支持 | 支持（同一 conversation_id） |
| **性能** | 本地访问快 | 受网络延迟影响 |
| **存储限制** | 磁盘空间 | OpenAI 配额 |
| **隐私** | 数据完全本地 | 数据存储在 OpenAI |

## 7. 并发场景时序图

### 场景：多线程并发访问 SQLite 会话

```mermaid
sequenceDiagram
    autonumber
    participant Thread1 as 线程 1
    participant Thread2 as 线程 2
    participant Session as SQLiteSession<br/>(文件模式)
    participant Local1 as Thread 1 本地连接
    participant Local2 as Thread 2 本地连接
    participant DB as 数据库文件
    
    par 线程 1 操作
        Thread1->>Session: await get_items()
        Session->>Session: _get_connection()
        Session->>Local1: 检查线程本地连接
        
        alt 连接不存在
            Session->>DB: connect(db_path)
            DB-->>Local1: 新连接
            Local1-->>Session: conn1
        end
        
        Session->>Local1: SELECT ...
        Local1->>DB: 执行查询
        DB-->>Local1: 结果集
        Local1-->>Session: rows
        Session-->>Thread1: 返回历史
        
    and 线程 2 操作（并发）
        Thread2->>Session: await add_items(items)
        Session->>Session: _get_connection()
        Session->>Local2: 检查线程本地连接
        
        alt 连接不存在
            Session->>DB: connect(db_path)
            DB-->>Local2: 新连接
            Local2-->>Session: conn2
        end
        
        Session->>Local2: BEGIN TRANSACTION
        Local2->>DB: 开始事务
        
        Session->>Local2: INSERT ...
        Local2->>DB: 写入数据
        
        Session->>Local2: COMMIT
        Local2->>DB: 提交事务
        
        DB-->>Local2: 成功
        Local2-->>Session: 完成
        Session-->>Thread2: 保存成功
    end
    
    Note over Thread1,DB: WAL 模式允许<br/>读写并发进行<br/>无需等待
```

**并发安全机制：**

1. **线程本地连接**：每个线程有独立的数据库连接，避免连接竞争
2. **WAL 模式**：支持读写并发，读不阻塞写，写不阻塞读
3. **事务隔离**：每个线程的事务独立，互不干扰
4. **自动重试**：SQLite 在锁冲突时自动重试

Memory 模块通过精心设计的时序流程和并发机制，为 OpenAI Agents 提供了高效、可靠的会话历史管理能力，支持从单线程到多线程、从本地到云端的各种应用场景。

