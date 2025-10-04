# OpenAI Agents Python SDK - Memory 模块 API 详解

## 1. API 总览

Memory 模块通过 `Session` 协议提供会话历史管理的标准接口，支持多种存储后端实现。所有 API 均为异步接口，确保在代理执行过程中不会阻塞主流程。

### API 层次结构

```
Session Protocol (接口定义)
    ├── SessionABC (抽象基类)
    │   ├── SQLiteSession (SQLite实现)
    │   └── OpenAIConversationsSession (OpenAI云端实现)
    └── 自定义实现（用户可扩展）
```

### API 分类

| API 类别 | 核心 API | 功能描述 |
|---------|---------|---------|
| **会话初始化** | `SQLiteSession.__init__()` | 创建SQLite会话实例 |
| | `OpenAIConversationsSession.__init__()` | 创建OpenAI云端会话实例 |
| | `start_openai_conversations_session()` | 启动OpenAI会话并获取ID |
| **历史查询** | `get_items()` | 检索会话历史记录 |
| **历史添加** | `add_items()` | 添加新的历史项目 |
| **历史删除** | `pop_item()` | 删除并返回最新项目 |
| | `clear_session()` | 清空会话所有历史 |
| **资源管理** | `close()` | 关闭数据库连接（仅SQLite） |

## 2. Session 协议 API

### 2.1 Session 协议定义

```python
@runtime_checkable
class Session(Protocol):
    """会话存储协议定义"""
    
    session_id: str
    
    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """检索会话历史"""
        ...
    
    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """添加历史项目"""
        ...
    
    async def pop_item(self) -> TResponseInputItem | None:
        """删除最新项目"""
        ...
    
    async def clear_session(self) -> None:
        """清空会话历史"""
        ...
```

**协议特性：**
- **运行时检查**：使用 `@runtime_checkable` 装饰器，支持运行时类型检查
- **协议解耦**：第三方库可实现此协议而无需继承特定基类
- **类型安全**：通过 Protocol 提供静态类型检查支持
- **异步设计**：所有方法均为异步，适配现代异步框架

### 2.2 get_items - 检索会话历史

**API 签名：**
```python
async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]
```

**功能描述：**
检索指定会话的历史记录，支持限制返回数量。始终按时间顺序（从旧到新）返回历史项目。

**请求参数：**

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `limit` | `int \| None` | 否 | `None` | 返回的最大项目数。`None` 表示返回所有历史；指定数值时返回最新的 N 条记录 |

**返回结构：**
```python
list[TResponseInputItem]  # 响应输入项目列表
```

**TResponseInputItem 类型定义：**
```python
# TResponseInputItem 是联合类型，包含以下可能的结构：

# 1. 用户消息
{
    "type": "message",
    "role": "user",
    "content": [
        {
            "type": "input_text",
            "text": "用户输入的文本内容"
        }
    ]
}

# 2. 助手消息
{
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "助手回复的文本内容"
        }
    ]
}

# 3. 工具调用
{
    "type": "function_call",
    "call_id": "call_abc123",
    "name": "get_weather",
    "arguments": "{\"location\": \"Beijing\"}"
}

# 4. 工具结果
{
    "type": "function_call_output",
    "call_id": "call_abc123",
    "output": "{\"temperature\": 25, \"condition\": \"sunny\"}"
}
```

**行为规范：**
1. **时间顺序**：始终返回按创建时间升序排列的历史
2. **限制行为**：当 `limit` 指定时，返回最新的 N 条记录（仍按时间升序）
3. **空会话**：会话无历史时返回空列表 `[]`
4. **数据完整性**：跳过损坏或无法解析的历史项目

**使用示例：**
```python
# 获取所有历史
all_history = await session.get_items()

# 获取最新10条历史
recent_history = await session.get_items(limit=10)

# 遍历历史项目
for item in all_history:
    if item["type"] == "message":
        role = item["role"]
        content = item["content"][0]["text"]
        print(f"{role}: {content}")
```

**异常情况：**
- 数据库连接失败：抛出底层数据库异常
- JSON 解析失败：跳过该项，不中断整个查询
- 会话不存在：返回空列表（不抛出异常）

### 2.3 add_items - 添加历史项目

**API 签名：**
```python
async def add_items(self, items: list[TResponseInputItem]) -> None
```

**功能描述：**
将新的对话项目批量添加到会话历史中。支持原子性添加多个项目，确保历史的一致性。

**请求参数：**

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `items` | `list[TResponseInputItem]` | 是 | - | 要添加的历史项目列表，可包含消息、工具调用、工具结果等 |

**返回值：**
`None` - 无返回值，通过异常传递错误

**请求项目结构示例：**
```python
items = [
    # 用户消息
    {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "今天天气怎么样？"
            }
        ]
    },
    
    # 助手消息（带工具调用）
    {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "让我查询一下天气信息。"
            }
        ]
    },
    
    # 工具调用
    {
        "type": "function_call",
        "call_id": "call_weather_001",
        "name": "get_weather",
        "arguments": "{\"location\": \"current\"}"
    },
    
    # 工具结果
    {
        "type": "function_call_output",
        "call_id": "call_weather_001",
        "output": "{\"temperature\": 22, \"condition\": \"cloudy\", \"humidity\": 65}"
    },
    
    # 最终助手回复
    {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "今天天气多云，温度22度，湿度65%。"
            }
        ]
    }
]

await session.add_items(items)
```

**行为规范：**
1. **原子性**：所有项目作为一个事务添加，要么全部成功，要么全部失败
2. **时间戳**：每个项目添加时自动记录时间戳
3. **会话创建**：如果会话不存在，自动创建会话
4. **空列表处理**：传入空列表时，操作立即返回，不执行任何操作
5. **顺序保证**：项目按列表顺序添加，保持时间顺序

**使用示例：**
```python
# 添加单个用户消息
user_message = {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "Hello"}]
}
await session.add_items([user_message])

# 添加完整的对话轮次
conversation_turn = [
    user_message,
    assistant_message,
    tool_call,
    tool_output,
    final_assistant_message
]
await session.add_items(conversation_turn)
```

**异常情况：**
- 数据库写入失败：抛出数据库异常，事务回滚
- JSON 序列化失败：抛出 `TypeError` 或 `ValueError`
- 无效的项目结构：可能导致后续查询失败

### 2.4 pop_item - 删除最新项目

**API 签名：**
```python
async def pop_item(self) -> TResponseInputItem | None
```

**功能描述：**
原子性地删除并返回会话中最新的历史项目。常用于撤销操作或错误恢复场景。

**请求参数：**
无参数

**返回结构：**
```python
TResponseInputItem | None

# 成功情况：返回被删除的项目
{
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "这是最新的消息"}]
}

# 空会话情况：返回 None
None
```

**行为规范：**
1. **原子操作**：删除和返回在同一事务中完成，保证一致性
2. **最新优先**：始终删除时间戳最新的项目
3. **空会话处理**：会话为空时返回 `None`，不抛出异常
4. **数据损坏**：如果最新项目数据损坏，删除该项目但返回 `None`

**使用示例：**
```python
# 撤销最后一条消息
last_item = await session.pop_item()
if last_item:
    print(f"已删除: {last_item}")
else:
    print("会话为空，无法删除")

# 批量撤销
async def undo_last_n_items(session, n: int):
    """撤销最后 n 个项目"""
    deleted = []
    for _ in range(n):
        item = await session.pop_item()
        if item is None:
            break
        deleted.append(item)
    return deleted

# 撤销最后3个项目
deleted_items = await undo_last_n_items(session, 3)
print(f"共删除 {len(deleted_items)} 个项目")
```

**典型应用场景：**
1. **用户撤销**：用户请求撤销上一轮对话
2. **错误恢复**：工具调用失败后清理历史
3. **重新生成**：删除助手回复，重新生成响应
4. **历史修剪**：删除不需要的历史项目

**异常情况：**
- 数据库连接失败：抛出数据库异常
- 并发删除冲突：可能返回 `None`（项目已被其他操作删除）

### 2.5 clear_session - 清空会话历史

**API 签名：**
```python
async def clear_session(self) -> None
```

**功能描述：**
完全清空指定会话的所有历史记录，包括会话元数据。此操作不可逆。

**请求参数：**
无参数

**返回值：**
`None` - 无返回值

**行为规范：**
1. **完全清除**：删除所有历史项目和会话元数据
2. **级联删除**：相关的所有数据（消息、工具调用等）一并删除
3. **会话删除**：会话本身也会被删除，需要重新创建
4. **空会话处理**：对不存在或已空的会话调用，操作成功但无实际效果
5. **原子操作**：在单个事务中完成，保证一致性

**使用示例：**
```python
# 清空会话历史
await session.clear_session()
print("会话历史已清空")

# 带确认的清空操作
async def clear_with_confirmation(session: Session):
    """带用户确认的清空操作"""
    items = await session.get_items()
    if not items:
        print("会话已为空")
        return
    
    print(f"即将删除 {len(items)} 条历史记录")
    confirm = input("确认清空? (yes/no): ")
    
    if confirm.lower() == "yes":
        await session.clear_session()
        print("会话已清空")
    else:
        print("操作已取消")

# 清空并重新初始化
async def reset_session(session: Session, initial_context: str):
    """重置会话并添加初始上下文"""
    await session.clear_session()
    
    system_message = {
        "type": "message",
        "role": "system",
        "content": [{"type": "text", "text": initial_context}]
    }
    await session.add_items([system_message])
    print("会话已重置")
```

**典型应用场景：**
1. **重新开始**：用户请求开始新对话
2. **隐私保护**：清除敏感对话内容
3. **测试重置**：测试环境重置会话状态
4. **存储优化**：定期清理不需要的历史会话

**异常情况：**
- 数据库连接失败：抛出数据库异常
- 事务失败：抛出异常，部分数据可能仍存在（取决于数据库实现）

## 3. SQLiteSession API

### 3.1 SQLiteSession.__init__ - 初始化 SQLite 会话

**API 签名：**
```python
def __init__(
    self,
    session_id: str,
    db_path: str | Path = ":memory:",
    sessions_table: str = "agent_sessions",
    messages_table: str = "agent_messages",
)
```

**功能描述：**
创建基于 SQLite 的会话存储实例，支持内存数据库和文件数据库两种模式。

**请求参数：**

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `session_id` | `str` | 是 | - | 会话唯一标识符，用于区分不同会话 |
| `db_path` | `str \| Path` | 否 | `":memory:"` | 数据库路径。`:memory:` 表示内存数据库；文件路径表示持久化存储 |
| `sessions_table` | `str` | 否 | `"agent_sessions"` | 会话元数据表名，可自定义避免表名冲突 |
| `messages_table` | `str` | 否 | `"agent_messages"` | 消息数据表名，可自定义避免表名冲突 |

**数据库模式：**

**内存模式 (`db_path=":memory:"`)**：
- 数据存储在内存中，进程结束后丢失
- 使用共享连接避免线程隔离问题
- 适用于临时会话、测试场景
- 性能最优，无磁盘 I/O

**文件模式 (`db_path="/path/to/db.sqlite"`)**：
- 数据持久化到磁盘文件
- 使用线程本地连接提高并发性能
- 适用于生产环境、需要保留历史的场景
- 自动启用 WAL (Write-Ahead Logging) 模式提升性能

**数据库表结构：**

**会话表 (agent_sessions)：**
```sql
CREATE TABLE agent_sessions (
    session_id TEXT PRIMARY KEY,           -- 会话唯一标识
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- 最后更新时间
)
```

**消息表 (agent_messages)：**
```sql
CREATE TABLE agent_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 消息自增ID
    session_id TEXT NOT NULL,              -- 所属会话ID
    message_data TEXT NOT NULL,            -- JSON格式的消息数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
    FOREIGN KEY (session_id) REFERENCES agent_sessions (session_id)
        ON DELETE CASCADE                  -- 级联删除
)

-- 索引
CREATE INDEX idx_agent_messages_session_id
    ON agent_messages (session_id, created_at)
```

**使用示例：**
```python
from agents.memory import SQLiteSession

# 1. 内存数据库（临时会话）
memory_session = SQLiteSession(session_id="temp_session_001")

# 2. 文件数据库（持久化）
file_session = SQLiteSession(
    session_id="user_123_session",
    db_path="./data/conversations.db"
)

# 3. 自定义表名（避免冲突）
custom_session = SQLiteSession(
    session_id="custom_session",
    db_path="shared.db",
    sessions_table="my_sessions",
    messages_table="my_messages"
)

# 4. 使用 Path 对象
from pathlib import Path
path_session = SQLiteSession(
    session_id="path_session",
    db_path=Path.home() / "agent_data" / "conversations.db"
)
```

**线程安全特性：**
1. **内存模式**：使用共享连接 + 全局锁保证线程安全
2. **文件模式**：使用线程本地连接，每个线程独立连接
3. **WAL 模式**：提高并发读写性能，减少锁竞争
4. **事务保护**：所有写操作在事务中执行，保证原子性

**初始化流程：**
```python
# 内部初始化流程（简化版）
def __init__(self, session_id, db_path=":memory:", ...):
    self.session_id = session_id
    self.db_path = db_path
    self._is_memory_db = (str(db_path) == ":memory:")
    
    if self._is_memory_db:
        # 内存模式：创建共享连接
        self._shared_connection = sqlite3.connect(
            ":memory:", 
            check_same_thread=False
        )
        self._shared_connection.execute("PRAGMA journal_mode=WAL")
        self._init_db_for_connection(self._shared_connection)
    else:
        # 文件模式：初始化数据库模式
        init_conn = sqlite3.connect(str(db_path))
        init_conn.execute("PRAGMA journal_mode=WAL")
        self._init_db_for_connection(init_conn)
        init_conn.close()
```

**异常情况：**
- 文件路径无效：抛出 `sqlite3.OperationalError`
- 权限不足：抛出文件系统相关异常
- 表结构冲突：如果表已存在但结构不同，可能导致后续操作失败

### 3.2 SQLiteSession.close - 关闭数据库连接

**API 签名：**
```python
def close(self) -> None
```

**功能描述：**
显式关闭数据库连接，释放资源。对于文件数据库，这是可选的（连接会在对象销毁时自动关闭），但显式关闭是最佳实践。

**请求参数：**
无参数

**返回值：**
`None` - 无返回值

**行为规范：**
1. **内存模式**：关闭共享连接，数据丢失
2. **文件模式**：关闭当前线程的连接，其他线程不受影响
3. **多次调用**：重复调用不会抛出异常
4. **资源清理**：确保数据库文件句柄被释放

**使用示例：**
```python
# 手动管理生命周期
session = SQLiteSession("my_session", db_path="data.db")
try:
    await session.add_items([...])
    items = await session.get_items()
finally:
    session.close()  # 确保连接关闭

# 使用上下文管理器（推荐）
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_session(session_id: str, db_path: str):
    """会话的上下文管理器"""
    session = SQLiteSession(session_id, db_path)
    try:
        yield session
    finally:
        session.close()

# 使用方式
async with managed_session("my_session", "data.db") as session:
    await session.add_items([...])
    items = await session.get_items()
# 自动关闭
```

## 4. OpenAIConversationsSession API

### 4.1 start_openai_conversations_session - 启动云端会话

**API 签名：**
```python
async def start_openai_conversations_session(
    openai_client: AsyncOpenAI | None = None
) -> str
```

**功能描述：**
在 OpenAI 云端创建新的对话会话，返回会话 ID。这是使用 OpenAI Conversations API 的前置步骤。

**请求参数：**

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `openai_client` | `AsyncOpenAI \| None` | 否 | `None` | OpenAI 异步客户端实例。`None` 时使用默认客户端 |

**返回结构：**
```python
str  # OpenAI 会话 ID，格式如 "conv_abc123xyz"
```

**使用示例：**
```python
from openai import AsyncOpenAI
from agents.memory import start_openai_conversations_session

# 使用默认客户端
conversation_id = await start_openai_conversations_session()
print(f"创建的会话ID: {conversation_id}")

# 使用自定义客户端
custom_client = AsyncOpenAI(api_key="your-api-key")
conversation_id = await start_openai_conversations_session(custom_client)
```

**异常情况：**
- API 密钥无效：抛出 `AuthenticationError`
- 网络连接失败：抛出 `APIConnectionError`
- API 限流：抛出 `RateLimitError`

### 4.2 OpenAIConversationsSession.__init__ - 初始化云端会话

**API 签名：**
```python
def __init__(
    self,
    *,
    conversation_id: str | None = None,
    openai_client: AsyncOpenAI | None = None,
)
```

**功能描述：**
创建基于 OpenAI Conversations API 的会话实例，历史数据存储在 OpenAI 云端。

**请求参数：**

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `conversation_id` | `str \| None` | 否 | `None` | 已存在的会话 ID。`None` 时首次调用 API 会自动创建 |
| `openai_client` | `AsyncOpenAI \| None` | 否 | `None` | OpenAI 异步客户端。`None` 时使用默认客户端 |

**使用示例：**
```python
from agents.memory import OpenAIConversationsSession
from openai import AsyncOpenAI

# 1. 创建新会话（自动分配 ID）
session = OpenAIConversationsSession()

# 2. 使用已存在的会话 ID
existing_session = OpenAIConversationsSession(
    conversation_id="conv_abc123xyz"
)

# 3. 使用自定义客户端
custom_client = AsyncOpenAI(api_key="your-api-key")
custom_session = OpenAIConversationsSession(
    openai_client=custom_client
)

# 4. 完整配置
full_session = OpenAIConversationsSession(
    conversation_id="conv_existing",
    openai_client=custom_client
)
```

**内部实现特性：**
1. **延迟初始化**：会话 ID 在首次使用时才创建（如果未提供）
2. **自动客户端**：未提供客户端时自动使用默认配置
3. **云端存储**：所有数据存储在 OpenAI 服务器，不占用本地存储
4. **跨设备同步**：同一 conversation_id 可在不同设备/进程中访问

**适用场景：**
- 需要跨设备/进程共享会话历史
- 不想管理本地数据库
- 使用 OpenAI 的高级对话管理功能
- 云端备份和持久化需求

## 5. API 调用链路分析

### 5.1 get_items 调用链路

```
应用代码
    ↓
Runner.run() / RealtimeSession
    ↓
session.get_items(limit)
    ↓
┌─────────────────────────────────────┐
│ SQLiteSession.get_items()           │
│   ↓                                 │
│ asyncio.to_thread(_get_items_sync)  │
│   ↓                                 │
│ _get_connection()                   │
│   ↓                                 │
│ SQL SELECT with ORDER BY and LIMIT  │
│   ↓                                 │
│ json.loads() for each row           │
│   ↓                                 │
│ return list[TResponseInputItem]     │
└─────────────────────────────────────┘
    ↓
返回历史数据给调用者
```

**核心代码片段：**
```python
async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
    """SQLiteSession.get_items 核心实现"""
    
    def _get_items_sync():
        conn = self._get_connection()  # 获取数据库连接
        
        with self._lock if self._is_memory_db else threading.Lock():
            if limit is None:
                # 查询所有历史
                cursor = conn.execute(
                    f"""
                    SELECT message_data FROM {self.messages_table}
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (self.session_id,),
                )
            else:
                # 查询最新N条
                cursor = conn.execute(
                    f"""
                    SELECT message_data FROM {self.messages_table}
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (self.session_id, limit),
                )
            
            rows = cursor.fetchall()
            
            # 反转结果以保持时间顺序
            if limit is not None:
                rows = list(reversed(rows))
            
            # JSON 解析
            items = []
            for (message_data,) in rows:
                try:
                    item = json.loads(message_data)
                    items.append(item)
                except json.JSONDecodeError:
                    continue  # 跳过损坏的数据
            
            return items
    
    # 在线程池中执行同步数据库操作
    return await asyncio.to_thread(_get_items_sync)
```

**关键设计点：**
1. **异步转同步**：使用 `asyncio.to_thread()` 包装同步数据库操作
2. **线程安全**：内存模式使用全局锁，文件模式使用线程本地连接
3. **查询优化**：使用索引 `(session_id, created_at)` 加速查询
4. **错误容忍**：JSON 解析失败时跳过该项，不中断查询

### 5.2 add_items 调用链路

```
应用代码
    ↓
Runner.run() 执行后
    ↓
session.add_items(new_items)
    ↓
┌────────────────────────────────────────┐
│ SQLiteSession.add_items()              │
│   ↓                                    │
│ asyncio.to_thread(_add_items_sync)     │
│   ↓                                    │
│ _get_connection()                      │
│   ↓                                    │
│ BEGIN TRANSACTION                      │
│   ↓                                    │
│ INSERT OR IGNORE INTO sessions         │
│   ↓                                    │
│ json.dumps() for each item             │
│   ↓                                    │
│ INSERT INTO messages (batch)           │
│   ↓                                    │
│ UPDATE sessions SET updated_at         │
│   ↓                                    │
│ COMMIT                                 │
└────────────────────────────────────────┘
    ↓
返回成功（无返回值）
```

**核心代码片段：**
```python
async def add_items(self, items: list[TResponseInputItem]) -> None:
    """SQLiteSession.add_items 核心实现"""
    
    if not items:
        return  # 空列表快速返回
    
    def _add_items_sync():
        conn = self._get_connection()
        
        with self._lock if self._is_memory_db else threading.Lock():
            # 1. 确保会话存在
            conn.execute(
                f"""
                INSERT OR IGNORE INTO {self.sessions_table} (session_id) 
                VALUES (?)
                """,
                (self.session_id,),
            )
            
            # 2. 批量插入消息
            message_data = [
                (self.session_id, json.dumps(item)) 
                for item in items
            ]
            conn.executemany(
                f"""
                INSERT INTO {self.messages_table} (session_id, message_data) 
                VALUES (?, ?)
                """,
                message_data,
            )
            
            # 3. 更新会话时间戳
            conn.execute(
                f"""
                UPDATE {self.sessions_table}
                SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (self.session_id,),
            )
            
            # 4. 提交事务
            conn.commit()
    
    await asyncio.to_thread(_add_items_sync)
```

**关键设计点：**
1. **原子性**：所有操作在单个事务中完成
2. **批量插入**：使用 `executemany()` 批量插入提升性能
3. **自动创建会话**：使用 `INSERT OR IGNORE` 自动创建会话
4. **时间戳更新**：自动更新会话的最后更新时间

### 5.3 OpenAI Conversations API 调用链路

```
应用代码
    ↓
session.add_items(items)
    ↓
┌─────────────────────────────────────────────┐
│ OpenAIConversationsSession.add_items()      │
│   ↓                                         │
│ _get_session_id()                           │
│   ├── session_id 已存在？返回               │
│   └── 未存在？调用 API 创建                 │
│       ↓                                     │
│       openai_client.conversations.create()  │
│   ↓                                         │
│ openai_client.conversations.items.create()  │
│   ↓                                         │
│   [HTTP POST to OpenAI API]                 │
│   ↓                                         │
│   OpenAI 服务器存储数据                     │
└─────────────────────────────────────────────┘
    ↓
返回成功
```

**核心代码片段：**
```python
async def add_items(self, items: list[TResponseInputItem]) -> None:
    """OpenAIConversationsSession.add_items 核心实现"""
    
    # 1. 获取或创建会话 ID
    session_id = await self._get_session_id()
    
    # 2. 调用 OpenAI API 添加项目
    await self._openai_client.conversations.items.create(
        conversation_id=session_id,
        items=items,
    )

async def _get_session_id(self) -> str:
    """延迟初始化会话 ID"""
    if self._session_id is None:
        # 调用 API 创建新会话
        self._session_id = await start_openai_conversations_session(
            self._openai_client
        )
    return self._session_id
```

**关键设计点：**
1. **延迟初始化**：首次使用时才创建云端会话
2. **API 封装**：完全封装 OpenAI API 细节
3. **无本地存储**：所有数据存储在云端
4. **网络依赖**：所有操作需要网络连接

## 6. API 使用最佳实践

### 6.1 会话生命周期管理

```python
from agents.memory import SQLiteSession
from contextlib import asynccontextmanager

@asynccontextmanager
async def create_persistent_session(user_id: str, db_path: str):
    """创建持久化会话的上下文管理器"""
    session_id = f"user_{user_id}_session"
    session = SQLiteSession(session_id, db_path=db_path)
    
    try:
        # 初始化时加载历史
        history = await session.get_items(limit=100)
        print(f"加载了 {len(history)} 条历史记录")
        
        yield session
        
    finally:
        # 清理资源
        session.close()
        print("会话已关闭")

# 使用方式
async def chat_with_memory(user_id: str):
    async with create_persistent_session(
        user_id, 
        "data/conversations.db"
    ) as session:
        # 进行对话
        result = await Runner.run(agent, "Hello", session=session)
        print(result.final_output)
```

### 6.2 历史管理策略

```python
async def manage_conversation_history(session: Session, max_items: int = 50):
    """管理会话历史，防止历史过长"""
    
    # 获取当前历史数量
    all_items = await session.get_items()
    
    if len(all_items) > max_items:
        # 保留最新的 max_items 条
        items_to_remove = len(all_items) - max_items
        
        print(f"历史过长，移除最旧的 {items_to_remove} 条记录")
        
        # 创建新会话ID（重新开始）
        await session.clear_session()
        
        # 保留最新的历史
        recent_items = all_items[-max_items:]
        await session.add_items(recent_items)
        
        print(f"历史已修剪，保留 {max_items} 条记录")
```

### 6.3 错误处理和重试

```python
import asyncio
from typing import TypeVar

T = TypeVar('T')

async def retry_session_operation(
    operation,
    max_retries: int = 3,
    delay: float = 1.0
) -> T:
    """会话操作的重试包装器"""
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"操作失败，{delay}秒后重试: {e}")
                await asyncio.sleep(delay)
                delay *= 2  # 指数退避
            else:
                print(f"操作失败，已达最大重试次数: {e}")
    
    raise last_error

# 使用方式
async def robust_add_items(session: Session, items: list):
    """带重试的添加项目操作"""
    await retry_session_operation(
        lambda: session.add_items(items),
        max_retries=3,
        delay=1.0
    )
```

### 6.4 多会话管理

```python
from typing import Dict
from agents.memory import Session, SQLiteSession

class SessionManager:
    """多会话管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.active_sessions: Dict[str, Session] = {}
    
    def get_session(self, user_id: str) -> Session:
        """获取或创建用户会话"""
        if user_id not in self.active_sessions:
            session_id = f"user_{user_id}"
            self.active_sessions[user_id] = SQLiteSession(
                session_id,
                db_path=self.db_path
            )
        return self.active_sessions[user_id]
    
    def close_session(self, user_id: str):
        """关闭指定用户的会话"""
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            if isinstance(session, SQLiteSession):
                session.close()
            del self.active_sessions[user_id]
    
    def close_all(self):
        """关闭所有会话"""
        for user_id in list(self.active_sessions.keys()):
            self.close_session(user_id)

# 使用方式
manager = SessionManager("data/multi_user.db")

async def handle_user_request(user_id: str, message: str):
    """处理用户请求"""
    session = manager.get_session(user_id)
    result = await Runner.run(agent, message, session=session)
    return result.final_output
```

Memory 模块通过简洁的 API 设计和灵活的实现选择，为 OpenAI Agents 提供了强大的会话历史管理能力，支持从临时内存存储到云端持久化的多种应用场景。

