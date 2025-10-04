---
title: "LangGraph-04-checkpoint-postgres"
date: 2025-10-04T21:26:31+08:00
draft: false
tags:
  - LangGraph
  - 架构设计
  - 概览
  - 源码分析
categories:
  - LangGraph
  - AI框架
  - Python
series: "langgraph-source-analysis"
description: "LangGraph 源码剖析 - 04-checkpoint-postgres"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true

---

# LangGraph-04-checkpoint-postgres

## 模块概览

## 一、模块职责

checkpoint-postgres模块是LangGraph检查点系统的PostgreSQL实现，提供生产级的状态持久化能力。该模块基于`BaseCheckpointSaver`接口，使用PostgreSQL数据库存储和检索图执行状态。

### 1.1 核心能力

1. **持久化存储**
   - 使用PostgreSQL数据库存储检查点
   - 支持多线程（thread）会话管理
   - 自动数据库表创建和迁移

2. **高性能优化**
   - Pipeline模式：批量执行SQL语句
   - Blob分离存储：大对象单独存储
   - 连接池支持：复用数据库连接

3. **完整功能支持**
   - 检查点CRUD操作
   - Pending writes存储
   - 时间旅行查询
   - Thread级别删除

4. **异步支持**
   - AsyncPostgresSaver：异步API
   - 完全兼容asyncio
   - 高并发场景优化

## 二、输入与输出

### 2.1 PostgresSaver输入

**初始化参数**

- `conn`：数据库连接或连接池
- `pipe`：Pipeline实例（可选）
- `serde`：序列化器（可选，默认JsonPlusSerializer）

**操作输入**

- `config`：RunnableConfig包含thread_id等
- `checkpoint`：Checkpoint字典
- `metadata`：CheckpointMetadata字典
- `writes`：待存储的写操作列表

### 2.2 PostgresSaver输出

**查询输出**

- `CheckpointTuple`：检查点元组
- `Iterator[CheckpointTuple]`：检查点列表
- `RunnableConfig`：更新后的配置

**状态输出**

- 存储成功：返回包含checkpoint_id的config
- 查询成功：返回完整的CheckpointTuple
- 列表查询：返回CheckpointTuple迭代器

### 2.3 上下游依赖

**上游（依赖方）**

- LangGraph core：使用PostgresSaver作为checkpointer
- 用户应用：直接实例化PostgresSaver
- create_react_agent等高级API

**下游（被依赖）**

- `langgraph-checkpoint`：继承BaseCheckpointSaver
- `psycopg`：PostgreSQL驱动（Psycopg 3）
- PostgreSQL数据库：实际存储引擎

## 三、模块架构

### 3.1 整体架构图

```mermaid
flowchart TB
    subgraph "用户层"
        A1[create_react_agent<br/>with checkpointer]
        A2[CompiledStateGraph<br/>with checkpointer]
        A3[直接使用<br/>PostgresSaver]
    end
    
    subgraph "PostgresSaver层"
        B1[PostgresSaver<br/>同步API]
        B2[AsyncPostgresSaver<br/>异步API]
        B3[ShallowPostgresSaver<br/>浅复制版本]
    end
    
    subgraph "BasePostgresSaver层"
        C1[BasePostgresSaver<br/>共享逻辑]
        C2[SQL语句定义<br/>MIGRATIONS/SELECT/INSERT/UPDATE]
        C3[序列化/反序列化<br/>_dump_blobs/_load_blobs]
    end
    
    subgraph "连接管理层"
        D1[Connection<br/>单连接]
        D2[ConnectionPool<br/>连接池]
        D3[Pipeline<br/>批量执行]
        D4[_internal<br/>连接管理工具]
    end
    
    subgraph "数据库层"
        E1[(checkpoints表<br/>主检查点数据)]
        E2[(checkpoint_blobs表<br/>大对象存储)]
        E3[(checkpoint_writes表<br/>待写入数据)]
        E4[(checkpoint_migrations表<br/>版本管理)]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A3 --> B2
    
    B1 --> C1
    B2 --> C1
    B3 --> C1
    
    C1 --> C2
    C1 --> C3
    
    B1 --> D1
    B1 --> D2
    B1 --> D3
    B2 --> D1
    B2 --> D2
    
    D1 --> D4
    D2 --> D4
    D3 --> D4
    
    D4 --> E1
    D4 --> E2
    D4 --> E3
    D4 --> E4
    
    style B1 fill:#e1f5ff
    style B2 fill:#e1f5ff
    style C1 fill:#ffe1e1
    style E1 fill:#fff4e1
```

### 3.2 架构说明

#### 3.2.1 图意概述

checkpoint-postgres模块采用分层架构，从上到下依次是用户层、PostgresSaver层、BasePostgresSaver层、连接管理层和数据库层。各层职责清晰，实现了同步/异步API的统一抽象。

#### 3.2.2 关键组件

**PostgresSaver**：同步实现

```python
class PostgresSaver(BasePostgresSaver):
    """使用PostgreSQL存储检查点"""
    
    def __init__(
        self,
        conn: Conn,
        pipe: Pipeline | None = None,
        serde: SerializerProtocol | None = None,
    ):
        super().__init__(serde=serde)
        self.conn = conn
        self.pipe = pipe
        self.lock = threading.Lock()
    
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """获取检查点"""
        ...
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """保存检查点"""
        ...
```

**工作原理**：

1. 通过conn参数接收数据库连接
2. 使用lock确保线程安全
3. 支持Pipeline模式批量执行SQL
4. 调用BasePostgresSaver的共享逻辑

**AsyncPostgresSaver**：异步实现

```python
class AsyncPostgresSaver(BasePostgresSaver):
    """异步PostgreSQL检查点存储器"""
    
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """异步获取检查点"""
        ...
    
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """异步保存检查点"""
        ...
```

**工作原理**：

1. 使用AsyncConnection和AsyncConnectionPool
2. 完全兼容asyncio
3. 共享BasePostgresSaver的SQL和序列化逻辑
4. 提供与PostgresSaver相同的接口

**BasePostgresSaver**：共享基类

```python
class BasePostgresSaver(BaseCheckpointSaver):
    """PostgreSQL检查点存储器的共享逻辑"""
    
    # SQL语句定义
    MIGRATIONS: list[str]  # 数据库迁移脚本
    SELECT_SQL: str  # 查询检查点
    UPSERT_CHECKPOINTS_SQL: str  # 插入/更新检查点
    UPSERT_CHECKPOINT_BLOBS_SQL: str  # 插入/更新blob
    INSERT_CHECKPOINT_WRITES_SQL: str  # 插入writes
    
    # 序列化方法
    def _dump_blobs(self, ...): ...
    def _load_blobs(self, ...): ...
    def _dump_writes(self, ...): ...
    def _load_writes(self, ...): ...
```

**职责**：

- 定义所有SQL语句
- 实现序列化/反序列化逻辑
- 提供数据库迁移脚本
- 被PostgresSaver和AsyncPostgresSaver继承

**连接管理**：

```python
# 单连接模式
with Connection.connect(
    DB_URI,
    autocommit=True,
    row_factory=dict_row
) as conn:
    checkpointer = PostgresSaver(conn)

# 连接池模式
from psycopg_pool import ConnectionPool

pool = ConnectionPool(DB_URI, open=True)
checkpointer = PostgresSaver(pool)

# Pipeline模式
with Connection.connect(DB_URI) as conn:
    with conn.pipeline() as pipe:
        checkpointer = PostgresSaver(conn, pipe)
```

**特点**：

- 单连接：简单场景，一次一个操作
- 连接池：高并发场景，自动管理连接
- Pipeline：批量操作，减少网络往返

#### 3.2.3 边界与约束

**连接要求**

- 必须设置`autocommit=True`：setup()方法需要立即提交
- 必须设置`row_factory=dict_row`：代码使用字典访问
- 缺少这些设置会导致运行时错误

**Pipeline限制**

- 不能与ConnectionPool同时使用
- 只能用于单个Connection
- 需要手动调用sync()提交

**线程安全**

- PostgresSaver使用threading.Lock保护
- 同一实例可以多线程访问
- 但每次只能有一个操作执行

**数据库兼容性**

- 需要PostgreSQL 9.5+（支持UPSERT）
- 需要JSON/JSONB类型支持
- 需要UUID扩展（可选）

#### 3.2.4 异常处理与回退

**数据库连接失败**

```python
try:
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
except psycopg.OperationalError as e:
    print(f"数据库连接失败: {e}")
    # 回退：使用InMemorySaver
    checkpointer = InMemorySaver()
```

**表不存在**

```python
# 首次使用必须调用setup()
checkpointer = PostgresSaver(conn)
checkpointer.setup()  # 创建表和运行迁移

# 如果忘记调用setup()，操作会失败
# psycopg.errors.UndefinedTable: relation "checkpoints" does not exist
```

**序列化失败**

```python
# 某些对象无法序列化
try:
    checkpointer.put(config, checkpoint, metadata, new_versions)
except Exception as e:
    print(f"序列化失败: {e}")
    # 回退：只保存可序列化的部分
    checkpoint_copy = {k: v for k, v in checkpoint.items() if is_serializable(v)}
    checkpointer.put(config, checkpoint_copy, metadata, new_versions)
```

**并发冲突**

```python
# 多个进程同时更新同一个checkpoint
# PostgreSQL使用UPSERT，后写入的会覆盖
# 使用乐观锁可以检测冲突：
if current_checkpoint_id != expected_checkpoint_id:
    raise ConcurrentModificationError()
```

#### 3.2.5 性能与容量

**存储策略**

- **Inline存储**：原始类型（str, int, float, bool, None）存储在checkpoints表
- **Blob存储**：复杂对象存储在checkpoint_blobs表
- **目的**：减少主表大小，加快查询速度

**查询性能**

```sql
-- 查询最新检查点：带索引，O(log n)
SELECT * FROM checkpoints
WHERE thread_id = ? AND checkpoint_ns = ?
ORDER BY checkpoint_id DESC LIMIT 1;

-- 索引策略
CREATE INDEX idx_checkpoints_thread_id ON checkpoints(thread_id, checkpoint_ns, checkpoint_id DESC);
```

**写入性能**

- Pipeline模式：批量写入，减少网络往返
- 单次put()调用：2-3个SQL语句（checkpoints + blobs + writes）
- 使用Pipeline：一次网络往返完成所有写入

**容量估算**

```python
# 单个检查点大小估算
checkpoint_size = (
    len(json.dumps(checkpoint))  # 主数据
    + len(json.dumps(metadata))   # 元数据
    + sum(len(serialize(v)) for v in blobs.values())  # Blob数据
)

# 示例：
# - 10条消息，每条1KB：~10KB
# - Metadata：~1KB
# - 总计：~11KB/检查点

# 100个用户，每人100个检查点：
total_storage = 100 * 100 * 11KB = 110MB
```

**优化建议**

1. 定期清理旧检查点
2. 使用消息修剪减少历史大小
3. 考虑使用ShallowPostgresSaver（不复制channel_values）
4. 使用连接池减少连接开销

#### 3.2.6 版本兼容与演进

**数据库迁移机制**

```python
MIGRATIONS = [
    # v0: 创建初始表
    """
    CREATE TABLE IF NOT EXISTS checkpoint_migrations (...);
    """,
    # v1: 添加新字段
    """
    ALTER TABLE checkpoints ADD COLUMN parent_checkpoint_id UUID;
    """,
    # v2: 添加索引
    """
    CREATE INDEX idx_parent ON checkpoints(parent_checkpoint_id);
    """,
]

def setup(self):
    """运行所有未应用的迁移"""
    current_version = get_current_version()
    for v in range(current_version + 1, len(MIGRATIONS)):
        execute_migration(MIGRATIONS[v])
        mark_migration_complete(v)
```

**向后兼容**

- 新版本代码兼容旧版本数据库schema
- 使用`checkpoint["v"]`字段标识检查点版本
- 读取时自动迁移pending_sends（v3 -> v4）

**未来演进**

- 支持分布式锁（避免并发冲突）
- 支持分区表（提高大规模性能）
- 支持自动归档（移动旧数据到冷存储）

## 四、数据库Schema

### 4.1 数据库表结构

```mermaid
erDiagram
    checkpoints ||--o{ checkpoint_blobs : contains
    checkpoints ||--o{ checkpoint_writes : contains
    checkpoints ||--o| checkpoints : parent
    checkpoint_migrations ||--o{ checkpoints : tracks
    
    checkpoints {
        text thread_id PK
        text checkpoint_ns PK
        uuid checkpoint_id PK
        uuid parent_checkpoint_id FK
        jsonb checkpoint
        jsonb metadata
        timestamp created_at
    }
    
    checkpoint_blobs {
        text thread_id PK
        text checkpoint_ns PK
        text channel PK
        int version PK
        text type
        bytea blob
    }
    
    checkpoint_writes {
        text thread_id PK
        text checkpoint_ns PK
        uuid checkpoint_id PK
        text task_id PK
        text task_path PK
        int idx PK
        text channel
        text type
        bytea blob
    }
    
    checkpoint_migrations {
        int v PK
        timestamp created_at
    }
```

**表说明**：

**checkpoints**：主检查点表

- `thread_id`：线程ID，会话标识
- `checkpoint_ns`：检查点命名空间
- `checkpoint_id`：检查点唯一ID（UUID）
- `parent_checkpoint_id`：父检查点ID（用于时间旅行）
- `checkpoint`：检查点数据（JSONB，包含inline值）
- `metadata`：元数据（JSONB）

**checkpoint_blobs**：大对象表

- `thread_id`, `checkpoint_ns`：关联到checkpoints
- `channel`：channel名称（如"messages"）
- `version`：channel版本号
- `type`：序列化类型
- `blob`：二进制数据

**checkpoint_writes**：待写入表

- `thread_id`, `checkpoint_ns`, `checkpoint_id`：关联到checkpoints
- `task_id`：任务ID
- `task_path`：任务路径
- `idx`：写入索引（WRITES_IDX_MAP）
- `channel`：目标channel
- `type`：序列化类型
- `blob`：二进制数据

**checkpoint_migrations**：迁移版本表

- `v`：迁移版本号
- `created_at`：迁移时间

### 4.2 索引策略

```sql
-- 主键索引（自动创建）
CREATE UNIQUE INDEX pk_checkpoints
ON checkpoints(thread_id, checkpoint_ns, checkpoint_id);

-- 查询最新检查点的索引
CREATE INDEX idx_checkpoints_latest
ON checkpoints(thread_id, checkpoint_ns, checkpoint_id DESC);

-- 父检查点索引（时间旅行）
CREATE INDEX idx_checkpoints_parent
ON checkpoints(parent_checkpoint_id);

-- Blob索引
CREATE UNIQUE INDEX pk_checkpoint_blobs
ON checkpoint_blobs(thread_id, checkpoint_ns, channel, version);

-- Writes索引
CREATE UNIQUE INDEX pk_checkpoint_writes
ON checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx);
```

## 五、生命周期

### 5.1 初始化阶段

```python
# 方式1：使用连接字符串
DB_URI = "postgres://user:pass@localhost:5432/db"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 首次使用时创建表
    # 使用checkpointer...

# 方式2：使用现有连接
from psycopg import Connection
from psycopg.rows import dict_row

with Connection.connect(
    DB_URI,
    autocommit=True,
    row_factory=dict_row
) as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    # 使用checkpointer...

# 方式3：使用连接池
from psycopg_pool import ConnectionPool

pool = ConnectionPool(DB_URI, open=True)
checkpointer = PostgresSaver(pool)
checkpointer.setup()
# 使用checkpointer...
# 程序退出时关闭连接池
pool.close()
```

### 5.2 运行时阶段

```python
# 创建Agent
agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
)

# 执行对话
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke(
    {"messages": [HumanMessage("Hello")]},
    config
)

# checkpointer在后台自动：
# 1. 每个超步后调用put()保存检查点
# 2. 恢复时调用get_tuple()加载检查点
# 3. 需要时调用put_writes()保存中间写入
```

### 5.3 清理阶段

```python
# 删除特定线程的所有数据
checkpointer.delete_thread("user-123")

# 批量清理旧数据（自定义）
def cleanup_old_checkpoints(checkpointer, days=30):
    """删除30天前的检查点"""
    with checkpointer._cursor() as cur:
        cur.execute("""
            DELETE FROM checkpoints
            WHERE created_at < NOW() - INTERVAL '%s days'
        """, (days,))

# 关闭连接
if isinstance(checkpointer.conn, ConnectionPool):
    checkpointer.conn.close()
```

## 六、核心算法与流程

### 6.1 Blob分离存储算法

```python
def put(self, config, checkpoint, metadata, new_versions):
    """
    保存检查点，智能分离inline和blob数据
    
    算法：

    1. 复制checkpoint，避免修改原始数据
    2. 遍历channel_values
    3. 原始类型：保留在checkpoint中（inline）
    4. 复杂对象：移到blob_values字典
    5. 分别存储到checkpoints和checkpoint_blobs表
    """
    # 步骤1：准备数据
    copy = checkpoint.copy()
    copy["channel_values"] = copy["channel_values"].copy()
    
    # 步骤2：分离inline和blob
    blob_values = {}
    for k, v in checkpoint["channel_values"].items():
        if v is None or isinstance(v, (str, int, float, bool)):
            # 原始类型：保留在主表
            pass
        else:
            # 复杂对象：移到blob表
            blob_values[k] = copy["channel_values"].pop(k)
    
    # 步骤3：存储
    with self._cursor(pipeline=True) as cur:
        # 先存储blobs
        if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
            cur.executemany(
                UPSERT_CHECKPOINT_BLOBS_SQL,
                self._dump_blobs(thread_id, checkpoint_ns, blob_values, blob_versions)
            )
        
        # 再存储主检查点
        cur.execute(
            UPSERT_CHECKPOINTS_SQL,
            (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
             Jsonb(copy), Jsonb(metadata))
        )
    
    return next_config

```

**算法说明**：

- **时间复杂度**：O(N)，N为channel_values数量
- **空间复杂度**：O(N)，需要复制数据
- **优化效果**：大幅减少主表大小，提高查询性能

### 6.2 Pipeline批量执行算法

```python
def put_writes(self, config, writes, task_id, task_path=""):
    """
    使用Pipeline批量存储writes
    
    算法：

    1. 准备所有SQL参数
    2. 使用executemany批量执行
    3. Pipeline模式下只需一次网络往返
    """
    # 步骤1：序列化所有writes
    serialized_writes = self._dump_writes(
        thread_id, checkpoint_ns, checkpoint_id,
        task_id, task_path, writes
    )
    
    # 步骤2：批量执行
    with self._cursor(pipeline=True) as cur:
        cur.executemany(
            INSERT_CHECKPOINT_WRITES_SQL,
            serialized_writes
        )
    # Pipeline模式：sync()在context manager退出时自动调用

```

**算法说明**：

- **网络往返**：Pipeline模式下仅1次，非Pipeline模式N次
- **性能提升**：10个writes，Pipeline快约10倍
- **适用场景**：多个相关操作需要一起执行

## 七、最佳实践

### 7.1 连接管理

**生产环境：使用连接池**

```python
from psycopg_pool import ConnectionPool

# 创建连接池
pool = ConnectionPool(
    DB_URI,
    min_size=5,
    max_size=20,
    open=True
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()

# 使用checkpointer...

# 应用退出时关闭
pool.close()
```

**开发环境：使用单连接**

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    # 开发测试...
```

### 7.2 错误处理

```python
import psycopg

try:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
except psycopg.OperationalError:
    print("数据库不可用，使用内存存储")
    checkpointer = InMemorySaver()
except psycopg.errors.InsufficientPrivilege:
    print("权限不足，无法创建表")
    raise

try:
    checkpointer.put(config, checkpoint, metadata, new_versions)
except psycopg.errors.UniqueViolation:
    print("检查点已存在，可能是并发写入")
except psycopg.errors.SerializationFailure:
    print("序列化失败，重试...")
    # 重试逻辑
```

### 7.3 性能优化

**使用Pipeline**

```python
# 批量操作时启用Pipeline
with Connection.connect(DB_URI) as conn:
    with conn.pipeline() as pipe:
        checkpointer = PostgresSaver(conn, pipe)
        
        # 批量保存检查点
        for checkpoint in checkpoints:
            checkpointer.put(config, checkpoint, metadata, versions)
        
        # 一次性提交所有操作
        pipe.sync()
```

**定期清理**

```python
def cleanup_old_data(checkpointer, days=30):
    """定期清理旧数据"""
    with checkpointer._cursor() as cur:
        # 删除旧检查点
        cur.execute("""
            DELETE FROM checkpoints
            WHERE created_at < NOW() - INTERVAL '%s days'
        """, (days,))
        
        # 删除孤立的blobs
        cur.execute("""
            DELETE FROM checkpoint_blobs
            WHERE (thread_id, checkpoint_ns) NOT IN (
                SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints
            )
        """)
```

## 八、总结

checkpoint-postgres模块通过以下设计实现了生产级的检查点存储：

1. **分层架构**：清晰的职责划分，同步/异步统一抽象
2. **智能存储**：Blob分离减少主表大小，提高查询性能
3. **高性能**：Pipeline批量执行，连接池复用连接
4. **可靠性**：完整的迁移机制，向后兼容
5. **易用性**：简单的API，自动表创建

这使得LangGraph可以在生产环境中稳定运行，支持大规模应用。

---

## API接口

## 一、API总览

checkpoint-postgres模块提供以下API：

1. **PostgresSaver**：同步检查点存储器
2. **AsyncPostgresSaver**：异步检查点存储器  
3. **ShallowPostgresSaver / AsyncShallowPostgresSaver**：浅复制版本
4. **BasePostgresSaver**：共享基类（通常不直接使用）

## 二、PostgresSaver API

### 2.1 基本信息

- **名称**：`PostgresSaver`
- **继承**：`BasePostgresSaver` → `BaseCheckpointSaver`
- **作用**：使用PostgreSQL数据库同步存储检查点
- **线程安全**：是（使用threading.Lock）

### 2.2 初始化方法

```python
def __init__(
    self,
    conn: Conn,
    pipe: Pipeline | None = None,
    serde: SerializerProtocol | None = None,
) -> None:
    """初始化PostgresSaver"""
```

**参数**：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `conn` | `Connection \| ConnectionPool` | ✓ | - | 数据库连接或连接池 |
| `pipe` | `Pipeline \| None` | ✗ | `None` | Pipeline实例（批量执行） |
| `serde` | `SerializerProtocol \| None` | ✗ | `None` | 序列化器（默认JsonPlusSerializer） |

**注意事项**：

- 如果`conn`是ConnectionPool，`pipe`必须为None
- Connection必须设置`autocommit=True`和`row_factory=dict_row`

### 2.3 from_conn_string

```python
@classmethod
@contextmanager
def from_conn_string(
    cls,
    conn_string: str,
    *,
    pipeline: bool = False,
) -> Iterator[PostgresSaver]:
    """从连接字符串创建PostgresSaver"""
```

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `conn_string` | `str` | ✓ | PostgreSQL连接字符串 |
| `pipeline` | `bool` | ✗ | 是否使用Pipeline模式 |

**返回**：`PostgresSaver`实例（作为上下文管理器）

**使用示例**：

```python
DB_URI = "postgres://user:pass@localhost:5432/db?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    # 使用checkpointer...
```

### 2.4 setup

```python
def setup(self) -> None:
    """创建数据库表并运行迁移"""
```

**功能**：

- 创建checkpoints、checkpoint_blobs、checkpoint_writes等表
- 运行所有未应用的数据库迁移
- **必须**在首次使用前调用

**示例**：

```python
checkpointer = PostgresSaver(conn)
checkpointer.setup()  # 创建表
```

### 2.5 get_tuple

```python
def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
    """获取检查点元组"""
```

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `config` | `RunnableConfig` | ✓ | 包含thread_id、checkpoint_id等 |

**返回**：`CheckpointTuple | None`

**逻辑**：

- 如果config包含`checkpoint_id`：返回该特定检查点
- 否则：返回该线程的最新检查点

**示例**：

```python
# 获取最新检查点
config = {"configurable": {"thread_id": "1"}}
checkpoint_tuple = checkpointer.get_tuple(config)

# 获取特定检查点
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"
    }
}
checkpoint_tuple = checkpointer.get_tuple(config)
```

### 2.6 list

```python
def list(
    self,
    config: RunnableConfig | None,
    *,
    filter: dict[str, Any] | None = None,
    before: RunnableConfig | None = None,
    limit: int | None = None,
) -> Iterator[CheckpointTuple]:
    """列出检查点"""
```

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `config` | `RunnableConfig \| None` | ✓ | 包含thread_id |
| `filter` | `dict \| None` | ✗ | 元数据过滤条件 |
| `before` | `RunnableConfig \| None` | ✗ | 返回此检查点之前的 |
| `limit` | `int \| None` | ✗ | 最大返回数量 |

**返回**：`Iterator[CheckpointTuple]`（按checkpoint_id降序）

**示例**：

```python
# 列出最近10个检查点
config = {"configurable": {"thread_id": "1"}}
checkpoints = list(checkpointer.list(config, limit=10))

# 列出特定检查点之前的
before = {"configurable": {"checkpoint_id": "..."}}
checkpoints = list(checkpointer.list(config, before=before))

# 按元数据过滤
filter_dict = {"source": "user"}
checkpoints = list(checkpointer.list(config, filter=filter_dict))
```

### 2.7 put

```python
def put(
    self,
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    new_versions: ChannelVersions,
) -> RunnableConfig:
    """保存检查点"""
```

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `config` | `RunnableConfig` | ✓ | 配置（包含thread_id等） |
| `checkpoint` | `Checkpoint` | ✓ | 检查点数据 |
| `metadata` | `CheckpointMetadata` | ✓ | 元数据 |
| `new_versions` | `ChannelVersions` | ✓ | 新channel版本 |

**返回**：`RunnableConfig`（包含生成的checkpoint_id）

**执行逻辑**：

1. 分离inline值和blob值
2. 保存blobs到checkpoint_blobs表
3. 保存检查点到checkpoints表
4. 返回包含checkpoint_id的config

**示例**：

```python
config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
checkpoint = {
    "v": 4,
    "id": "abc123",
    "ts": "2024-01-01T00:00:00Z",
    "channel_values": {"messages": [...], "key": "value"},
    "channel_versions": {"messages": "1", "key": "2"},
    "versions_seen": {},
}
metadata = {"source": "user", "step": 1}
new_versions = {"messages": "1", "key": "2"}

saved_config = checkpointer.put(config, checkpoint, metadata, new_versions)
print(saved_config)  # {'configurable': {'thread_id': '1', 'checkpoint_id': 'abc123'}}
```

### 2.8 put_writes

```python
def put_writes(
    self,
    config: RunnableConfig,
    writes: Sequence[tuple[str, Any]],
    task_id: str,
    task_path: str = "",
) -> None:
    """保存待写入数据"""
```

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `config` | `RunnableConfig` | ✓ | 包含thread_id、checkpoint_id |
| `writes` | `Sequence[tuple[str, Any]]` | ✓ | 待写入的(channel, value)列表 |
| `task_id` | `str` | ✓ | 任务ID |
| `task_path` | `str` | ✗ | 任务路径 |

**功能**：存储节点成功执行后的写入，用于pending writes机制

**示例**：

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "abc123"}}
writes = [
    ("messages", ToolMessage(...)),
    ("key", "value"),
]
checkpointer.put_writes(config, writes, task_id="task_1")
```

### 2.9 delete_thread

```python
def delete_thread(self, thread_id: str) -> None:
    """删除线程的所有数据"""
```

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `thread_id` | `str` | ✓ | 线程ID |

**功能**：删除该线程的所有检查点、blobs和writes

**示例**：

```python
checkpointer.delete_thread("user-123")
```

## 三、AsyncPostgresSaver API

### 3.1 基本信息

- **名称**：`AsyncPostgresSaver`
- **继承**：`BasePostgresSaver` → `BaseCheckpointSaver`
- **作用**：异步版本的PostgresSaver
- **并发安全**：是（使用asyncio.Lock）

### 3.2 初始化方法

```python
def __init__(
    self,
    conn: AsyncConnection | AsyncConnectionPool,
    pipe: AsyncPipeline | None = None,
    serde: SerializerProtocol | None = None,
) -> None:
    """初始化AsyncPostgresSaver"""
```

**参数**：与PostgresSaver相同，但使用异步类型

### 3.3 from_conn_string

```python
@classmethod
@asynccontextmanager
async def from_conn_string(
    cls,
    conn_string: str,
    *,
    pipeline: bool = False,
    serde: SerializerProtocol | None = None,
) -> AsyncIterator[AsyncPostgresSaver]:
    """异步创建AsyncPostgresSaver"""
```

**使用示例**：

```python
DB_URI = "postgres://user:pass@localhost:5432/db"
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    # 使用checkpointer...
```

### 3.4 异步方法

所有方法都有异步版本（添加`a`前缀）：

| 同步方法 | 异步方法 | 说明 |
|---|---|---|
| `setup()` | `async setup()` | 创建表和迁移 |
| `get_tuple()` | `async aget_tuple()` | 获取检查点 |
| `list()` | `alist()` | 列出检查点（返回AsyncIterator） |
| `put()` | `async aput()` | 保存检查点 |
| `put_writes()` | `async aput_writes()` | 保存writes |
| `delete_thread()` | `async adelete_thread()` | 删除线程 |

**使用示例**：

```python
async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        
        # 获取检查点
        config = {"configurable": {"thread_id": "1"}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        
        # 列出检查点
        checkpoints = [c async for c in checkpointer.alist(config, limit=10)]
        
        # 保存检查点
        saved_config = await checkpointer.aput(config, checkpoint, metadata, versions)
        
        # 删除线程
        await checkpointer.adelete_thread("user-123")

asyncio.run(main())
```

## 四、ShallowPostgresSaver API

### 4.1 基本信息

- **名称**：`ShallowPostgresSaver` / `AsyncShallowPostgresSaver`
- **作用**：浅复制版本，不完全复制channel_values
- **适用场景**：减少存储空间，仅保存channel版本引用

### 4.2 特点

**与PostgresSaver的区别**：

- 不复制整个channel_values
- 只保存channel版本号
- 节省存储空间
- 查询时需要追溯到父检查点

**使用场景**：

- 检查点非常频繁
- channel_values很大
- 对读取性能要求不高

**示例**：

```python
from langgraph.checkpoint.postgres.shallow import ShallowPostgresSaver

checkpointer = ShallowPostgresSaver(conn)
checkpointer.setup()

# API与PostgresSaver完全相同
checkpointer.put(config, checkpoint, metadata, versions)
```

## 五、BasePostgresSaver API

### 5.1 基本信息

- **名称**：`BasePostgresSaver`
- **作用**：PostgresSaver和AsyncPostgresSaver的共享基类
- **用途**：通常不直接使用，提供共享逻辑

### 5.2 共享属性

```python
class BasePostgresSaver(BaseCheckpointSaver[str]):
    # SQL语句
    SELECT_SQL: str
    SELECT_PENDING_SENDS_SQL: str
    UPSERT_CHECKPOINTS_SQL: str
    UPSERT_CHECKPOINT_BLOBS_SQL: str
    INSERT_CHECKPOINT_WRITES_SQL: str
    UPSERT_CHECKPOINT_WRITES_SQL: str
    
    # 迁移脚本
    MIGRATIONS: list[str]
```

### 5.3 共享方法

**序列化方法**：

```python
def _dump_blobs(
    self,
    thread_id: str,
    checkpoint_ns: str,
    values: dict[str, Any],
    versions: dict[str, str],
) -> Iterator[tuple]:
    """序列化blobs为数据库行"""

def _load_blobs(
    self,
    blob_values: list | None,
) -> dict[str, Any]:
    """反序列化blobs"""

def _dump_writes(
    self,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    task_path: str,
    writes: Sequence[tuple[str, Any]],
) -> Iterator[tuple]:
    """序列化writes为数据库行"""

def _load_writes(
    self,
    writes: list | None,
) -> list[tuple[str, Any]]:
    """反序列化writes"""
```

**查询构建方法**：

```python
def _search_where(
    self,
    config: RunnableConfig | None,
    filter: dict[str, Any] | None,
    before: RunnableConfig | None,
) -> tuple[str, tuple]:
    """构建WHERE子句和参数"""
```

## 六、API使用模式

### 6.1 基本使用模式

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgres://user:pass@localhost:5432/db"

# 模式1：使用from_conn_string
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    
    # 使用checkpointer
    agent = create_react_agent(model, tools, checkpointer=checkpointer)
    result = agent.invoke(input, config)

# 模式2：使用自定义连接
from psycopg import Connection
from psycopg.rows import dict_row

with Connection.connect(
    DB_URI,
    autocommit=True,
    row_factory=dict_row
) as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    
    # 使用checkpointer...

# 模式3：使用连接池
from psycopg_pool import ConnectionPool

pool = ConnectionPool(DB_URI, min_size=5, max_size=20, open=True)
checkpointer = PostgresSaver(pool)
checkpointer.setup()

# 使用checkpointer...

pool.close()  # 应用退出时关闭
```

### 6.2 Pipeline模式

```python
# 批量操作时使用Pipeline
with Connection.connect(DB_URI) as conn:
    with conn.pipeline() as pipe:
        checkpointer = PostgresSaver(conn, pipe)
        checkpointer.setup()
        
        # 批量保存
        for checkpoint in checkpoints:
            checkpointer.put(config, checkpoint, metadata, versions)
        
        # 一次性提交
        pipe.sync()
```

### 6.3 异步模式

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def main():
    DB_URI = "postgres://user:pass@localhost:5432/db"
    
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        
        # 异步使用
        config = {"configurable": {"thread_id": "1"}}
        checkpoint = await checkpointer.aget_tuple(config)
        
        # 列出检查点
        async for ckpt in checkpointer.alist(config, limit=10):
            print(ckpt)

asyncio.run(main())
```

### 6.4 错误处理模式

```python
import psycopg

try:
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
except psycopg.OperationalError as e:
    print(f"数据库连接失败: {e}")
    # 回退到InMemorySaver
    from langgraph.checkpoint.memory import InMemorySaver
    checkpointer = InMemorySaver()

try:
    checkpointer.put(config, checkpoint, metadata, versions)
except psycopg.errors.UniqueViolation:
    print("检查点已存在")
except psycopg.errors.SerializationFailure:
    print("序列化失败，重试...")
```

## 七、API最佳实践

### 7.1 连接管理

**生产环境使用连接池**：

```python
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    DB_URI,
    min_size=5,  # 最小连接数
    max_size=20,  # 最大连接数
    timeout=30,  # 获取连接超时
    open=True,
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()

# 应用退出时关闭
import atexit
atexit.register(pool.close)
```

**开发环境使用单连接**：

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    # 开发测试...
```

### 7.2 性能优化

**使用Pipeline批量操作**：

```python
with conn.pipeline() as pipe:
    checkpointer = PostgresSaver(conn, pipe)
    
    # 批量保存（减少网络往返）
    for i in range(100):
        checkpointer.put(config, checkpoints[i], metadata, versions)
    
    pipe.sync()  # 一次性提交
```

**异步提高并发**：

```python
async def save_many(checkpointer, checkpoints):
    tasks = [
        checkpointer.aput(config, ckpt, metadata, versions)
        for ckpt in checkpoints
    ]
    await asyncio.gather(*tasks)
```

### 7.3 数据清理

**定期清理旧数据**：

```python
def cleanup_old_checkpoints(checkpointer, days=30):
    """删除30天前的检查点"""
    with checkpointer._cursor() as cur:
        cur.execute("""
            DELETE FROM checkpoints
            WHERE created_at < NOW() - INTERVAL '%s days'
        """, (days,))
        
        # 清理孤立的blobs和writes
        cur.execute("""
            DELETE FROM checkpoint_blobs
            WHERE (thread_id, checkpoint_ns) NOT IN (
                SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints
            )
        """)
        
        cur.execute("""
            DELETE FROM checkpoint_writes
            WHERE (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints
            )
        """)
```

### 7.4 监控与日志

**添加日志**：

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingPostgresSaver(PostgresSaver):
    def put(self, config, checkpoint, metadata, new_versions):
        logger.info(f"Saving checkpoint for thread {config['configurable']['thread_id']}")
        result = super().put(config, checkpoint, metadata, new_versions)
        logger.info(f"Checkpoint saved: {result['configurable']['checkpoint_id']}")
        return result
```

**性能监控**：

```python
import time

class TimingPostgresSaver(PostgresSaver):
    def put(self, config, checkpoint, metadata, new_versions):
        start = time.time()
        result = super().put(config, checkpoint, metadata, new_versions)
        duration = time.time() - start
        print(f"put() took {duration:.3f}s")
        return result
```

## 八、总结

checkpoint-postgres模块的API设计体现了以下特点：

1. **简洁易用**：from_conn_string快速创建
2. **灵活配置**：支持单连接、连接池、Pipeline多种模式
3. **同步异步**：完整的异步API支持
4. **生产就绪**：完善的错误处理和性能优化
5. **可扩展**：清晰的继承结构，易于定制

通过合理使用这些API，可以构建稳定高效的LangGraph应用。

---

## 数据结构

## 一、数据结构总览

checkpoint-postgres模块的数据结构分为三个层次：

1. **Python数据结构**：Python类和TypedDict
2. **数据库Schema**：PostgreSQL表结构
3. **序列化格式**：数据库中的存储格式

## 二、Python数据结构

### 2.1 PostgresSaver类

```python
class PostgresSaver(BasePostgresSaver):
    """PostgreSQL检查点存储器"""
    
    conn: Connection | ConnectionPool
    pipe: Pipeline | None
    lock: threading.Lock
    supports_pipeline: bool
```

**UML类图**：

```mermaid
classDiagram
    class BaseCheckpointSaver {
        <<abstract>>
        +serde: SerializerProtocol
        +get_tuple(config) CheckpointTuple
        +list(config) Iterator
        +put(config, checkpoint, metadata, versions) RunnableConfig
        +put_writes(config, writes, task_id)
    }
    
    class BasePostgresSaver {
        <<abstract>>
        +SELECT_SQL: str
        +MIGRATIONS: list~str~
        +_dump_blobs(...) Iterator
        +_load_blobs(...) dict
        +_dump_writes(...) Iterator
        +_load_writes(...) list
    }
    
    class PostgresSaver {
        +conn: Connection | ConnectionPool
        +pipe: Pipeline | None
        +lock: threading.Lock
        +supports_pipeline: bool
        +setup()
        +get_tuple(config) CheckpointTuple
        +list(config) Iterator
        +put(config, ...) RunnableConfig
        +put_writes(config, writes, task_id)
        +delete_thread(thread_id)
        +_cursor() Iterator~Cursor~
    }
    
    class AsyncPostgresSaver {
        +conn: AsyncConnection | AsyncConnectionPool
        +pipe: AsyncPipeline | None
        +lock: asyncio.Lock
        +loop: AbstractEventLoop
        +async setup()
        +async aget_tuple(config) CheckpointTuple
        +alist(config) AsyncIterator
        +async aput(config, ...) RunnableConfig
        +async aput_writes(config, writes, task_id)
        +async adelete_thread(thread_id)
        +_cursor() AsyncIterator~AsyncCursor~
    }
    
    class ShallowPostgresSaver {
        +浅复制版本
    }
    
    class AsyncShallowPostgresSaver {
        +异步浅复制版本
    }
    
    BaseCheckpointSaver <|-- BasePostgresSaver
    BasePostgresSaver <|-- PostgresSaver
    BasePostgresSaver <|-- AsyncPostgresSaver
    BasePostgresSaver <|-- ShallowPostgresSaver
    BasePostgresSaver <|-- AsyncShallowPostgresSaver
```

**字段说明**：

**conn**：数据库连接

- 类型：`Connection | ConnectionPool`
- 作用：执行SQL操作
- Connection：单连接，适用于单线程/低并发
- ConnectionPool：连接池，适用于高并发

**pipe**：Pipeline实例

- 类型：`Pipeline | None`
- 作用：批量执行SQL，减少网络往返
- 限制：不能与ConnectionPool同时使用

**lock**：线程锁

- 类型：`threading.Lock`（同步）或`asyncio.Lock`（异步）
- 作用：确保线程/协程安全
- 保护：同一时间只能有一个操作访问连接

**supports_pipeline**：Pipeline支持

- 类型：`bool`
- 作用：指示psycopg是否支持Pipeline模式
- 检测：通过`Capabilities().has_pipeline()`

### 2.2 连接类型

```mermaid
classDiagram
    class Connection {
        +execute(sql, params)
        +cursor() Cursor
        +pipeline() Pipeline
        +commit()
        +rollback()
        +close()
    }
    
    class ConnectionPool {
        +getconn() Connection
        +putconn(conn)
        +min_size: int
        +max_size: int
        +timeout: float
        +open()
        +close()
    }
    
    class Pipeline {
        +sync()
        +__enter__()
        +__exit__()
    }
    
    class Cursor {
        +execute(sql, params)
        +executemany(sql, params_seq)
        +fetchone() dict
        +fetchall() list~dict~
        +close()
    }
    
    Connection --> Cursor : creates
    Connection --> Pipeline : creates
    ConnectionPool --> Connection : manages
```

**Connection**：单个数据库连接

- 适用场景：单线程应用、测试
- 特点：简单直接
- 限制：同时只能执行一个操作

**ConnectionPool**：连接池

- 适用场景：多线程应用、高并发
- 特点：自动管理连接、复用连接
- 配置：min_size（最小连接数）、max_size（最大连接数）

**Pipeline**：批量执行模式

- 适用场景：需要执行多个SQL语句
- 特点：减少网络往返、提高性能
- 使用：with conn.pipeline() as pipe

**Cursor**：数据库游标

- 作用：执行SQL、获取结果
- 类型：DictRow（返回dict）或TupleRow（返回tuple）
- 要求：必须使用dict_row工厂

## 三、数据库Schema

### 3.1 数据库表结构

```mermaid
erDiagram
    checkpoints {
        text thread_id PK
        text checkpoint_ns PK
        text checkpoint_id PK
        text parent_checkpoint_id FK
        text type
        jsonb checkpoint
        jsonb metadata
    }
    
    checkpoint_blobs {
        text thread_id PK
        text checkpoint_ns PK
        text channel PK
        text version PK
        text type
        bytea blob
    }
    
    checkpoint_writes {
        text thread_id PK
        text checkpoint_ns PK
        text checkpoint_id PK
        text task_id PK
        text task_path PK
        int idx PK
        text channel
        text type
        bytea blob
    }
    
    checkpoint_migrations {
        int v PK
    }
    
    checkpoints ||--o{ checkpoint_blobs : "thread_id, checkpoint_ns"
    checkpoints ||--o{ checkpoint_writes : "thread_id, checkpoint_ns, checkpoint_id"
    checkpoints ||--o| checkpoints : "parent_checkpoint_id"
```

### 3.2 checkpoints表

```sql
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
```

**字段详解**：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `thread_id` | TEXT | PK, NOT NULL | 线程ID，会话标识符 |
| `checkpoint_ns` | TEXT | PK, NOT NULL, DEFAULT '' | 检查点命名空间 |
| `checkpoint_id` | TEXT | PK, NOT NULL | 检查点唯一ID（UUID） |
| `parent_checkpoint_id` | TEXT | FK, NULL | 父检查点ID，用于时间旅行 |
| `type` | TEXT | NULL | 类型标识（保留字段） |
| `checkpoint` | JSONB | NOT NULL | 检查点数据（JSON格式） |
| `metadata` | JSONB | NOT NULL, DEFAULT '{}' | 元数据（JSON格式） |

**checkpoint字段结构**：

```json
{
  "v": 4,
  "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
  "ts": "2024-07-31T20:14:19.804150+00:00",
  "channel_values": {
    "my_key": "inline_value",
    // 复杂对象不在这里，在checkpoint_blobs表
  },
  "channel_versions": {
    "__start__": "2",
    "my_key": "3",
    "messages": "5"
  },
  "versions_seen": {
    "__input__": {},
    "__start__": {"__start__": "1"},
    "node": {"start:node": "2"}
  },
  "pending_sends": []
}
```

**metadata字段结构**：

```json
{
  "source": "input",
  "step": 1,
  "writes": {"key": "value"},
  "user_id": "123",
  // 任意用户定义的元数据
}
```

**索引**：

```sql
-- 主键索引（自动创建）
CREATE UNIQUE INDEX checkpoints_pkey
ON checkpoints(thread_id, checkpoint_ns, checkpoint_id);

-- 查询最新检查点
CREATE INDEX checkpoints_thread_id_idx
ON checkpoints(thread_id);

-- 父检查点索引（时间旅行）
CREATE INDEX checkpoints_parent_idx
ON checkpoints(parent_checkpoint_id);
```

### 3.3 checkpoint_blobs表

```sql
CREATE TABLE checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
```

**字段详解**：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `thread_id` | TEXT | PK, NOT NULL | 对应checkpoints.thread_id |
| `checkpoint_ns` | TEXT | PK, NOT NULL, DEFAULT '' | 对应checkpoints.checkpoint_ns |
| `channel` | TEXT | PK, NOT NULL | Channel名称（如"messages"） |
| `version` | TEXT | PK, NOT NULL | Channel版本号 |
| `type` | TEXT | NOT NULL | 序列化类型（如"json"） |
| `blob` | BYTEA | NULL | 二进制数据 |

**为什么需要blob表**：

- **性能优化**：分离大对象，减少主表大小
- **查询加速**：查询检查点元数据时不加载大对象
- **存储效率**：BYTEA存储二进制数据比JSONB更紧凑

**存储策略**：

- 原始类型（str, int, float, bool, None）：存储在checkpoints.checkpoint.channel_values
- 复杂对象（list, dict, BaseModel等）：存储在checkpoint_blobs.blob

**索引**：

```sql
CREATE UNIQUE INDEX checkpoint_blobs_pkey
ON checkpoint_blobs(thread_id, checkpoint_ns, channel, version);

CREATE INDEX checkpoint_blobs_thread_id_idx
ON checkpoint_blobs(thread_id);
```

### 3.4 checkpoint_writes表

```sql
CREATE TABLE checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
```

**字段详解**：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `thread_id` | TEXT | PK, NOT NULL | 线程ID |
| `checkpoint_ns` | TEXT | PK, NOT NULL | 命名空间 |
| `checkpoint_id` | TEXT | PK, NOT NULL | 检查点ID |
| `task_id` | TEXT | PK, NOT NULL | 任务ID |
| `task_path` | TEXT | PK, NOT NULL, DEFAULT '' | 任务路径（子图） |
| `idx` | INTEGER | PK, NOT NULL | 写入索引（WRITES_IDX_MAP） |
| `channel` | TEXT | NOT NULL | 目标channel |
| `type` | TEXT | NULL | 序列化类型 |
| `blob` | BYTEA | NOT NULL | 序列化数据 |

**用途**：存储pending writes（节点成功但图执行失败时的写入）

**写入类型（idx）**：

```python
WRITES_IDX_MAP = {
    "error": 0,       # 错误输出
    "info": 1,        # 信息输出
    "tool_calls": 2,  # 工具调用
    "writes": 3,      # 一般写入
}
```

**索引**：

```sql
CREATE UNIQUE INDEX checkpoint_writes_pkey
ON checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx);

CREATE INDEX checkpoint_writes_thread_id_idx
ON checkpoint_writes(thread_id);
```

### 3.5 checkpoint_migrations表

```sql
CREATE TABLE checkpoint_migrations (
    v INTEGER PRIMARY KEY
);
```

**用途**：跟踪已应用的数据库迁移版本

**使用**：

```python
# 查询当前版本
SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1;

# 记录新迁移
INSERT INTO checkpoint_migrations (v) VALUES (5);
```

## 四、序列化格式

### 4.1 Checkpoint序列化

```python
# Python对象
checkpoint = {
    "v": 4,
    "id": "abc123",
    "ts": "2024-01-01T00:00:00Z",
    "channel_values": {
        "messages": [AIMessage(...), ToolMessage(...)],  # 复杂对象
        "key": "value",  # 原始类型
        "count": 42,  # 原始类型
    },
    "channel_versions": {"messages": "5", "key": "3", "count": "2"},
    "versions_seen": {},
}

# 存储到checkpoints表
stored_checkpoint = {
    "v": 4,
    "id": "abc123",
    "ts": "2024-01-01T00:00:00Z",
    "channel_values": {
        "key": "value",  # 原始类型保留
        "count": 42,
        // messages被移除，存储到checkpoint_blobs
    },
    "channel_versions": {"messages": "5", "key": "3", "count": "2"},
    "versions_seen": {},
}

# 存储到checkpoint_blobs表
# thread_id | checkpoint_ns | channel  | version | type | blob
# thread-1  | ""            | messages | 5       | json | <serialized_messages>
```

### 4.2 Blob序列化

**序列化流程**：

```python
def _dump_blobs(thread_id, checkpoint_ns, values, versions):
    for channel, value in values.items():
        version = versions[channel]
        
        # 使用serde序列化
        type_name, serialized_value = serde.dumps(value)
        
        yield (
            thread_id,
            checkpoint_ns,
            channel,
            version,
            type_name,  # "json", "msgpack", etc.
            serialized_value,  # bytes
        )
```

**反序列化流程**：

```python
def _load_blobs(blob_values):
    result = {}
    for row in blob_values:
        channel = row[0].decode()
        type_name = row[1].decode()
        blob = row[2]
        
        # 使用serde反序列化
        value = serde.loads((type_name, blob))
        result[channel] = value
    
    return result
```

### 4.3 Writes序列化

**序列化**：

```python
def _dump_writes(thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, writes):
    for idx, (channel, value) in enumerate(writes):
        type_name, serialized_value = serde.dumps(value)
        
        # 映射channel到idx
        write_idx = WRITES_IDX_MAP.get(channel, 3)  # 默认为"writes"
        
        yield (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_id,
            task_path,
            write_idx,
            channel,
            type_name,
            serialized_value,
        )
```

**反序列化**：

```python
def _load_writes(writes):
    if not writes:
        return []
    
    result = []
    for row in writes:
        task_id = row[0].decode()
        channel = row[1].decode()
        type_name = row[2].decode()
        blob = row[3]
        
        value = serde.loads((type_name, blob))
        result.append((channel, value))
    
    return result
```

## 五、数据结构关系图

```mermaid
graph TB
    subgraph "Python层"
        A1[PostgresSaver]
        A2[AsyncPostgresSaver]
        A3[BasePostgresSaver]
    end
    
    subgraph "连接层"
        B1[Connection]
        B2[ConnectionPool]
        B3[Pipeline]
        B4[Cursor]
    end
    
    subgraph "数据库层"
        C1[(checkpoints表)]
        C2[(checkpoint_blobs表)]
        C3[(checkpoint_writes表)]
        C4[(checkpoint_migrations表)]
    end
    
    subgraph "序列化层"
        D1[SerializerProtocol]
        D2[JsonPlusSerializer]
        D3[JSONB格式]
        D4[BYTEA格式]
    end
    
    A1 --> A3
    A2 --> A3
    A1 --> B1
    A1 --> B2
    A1 --> B3
    A2 --> B1
    A2 --> B2
    
    B1 --> B4
    B2 --> B1
    B3 --> B4
    
    B4 --> C1
    B4 --> C2
    B4 --> C3
    B4 --> C4
    
    A3 --> D1
    D1 --> D2
    D2 --> D3
    D2 --> D4
    
    C1 --> D3
    C2 --> D4
    C3 --> D4
    
    style A3 fill:#e1f5ff
    style C1 fill:#ffe1e1
    style D2 fill:#fff4e1
```

## 六、数据结构演进

### 6.1 版本历史

**v0 → v1**：添加checkpoint_migrations表

```sql
CREATE TABLE checkpoint_migrations (v INTEGER PRIMARY KEY);
```

**v1 → v2**：创建checkpoints表

```sql
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
```

**v2 → v3**：创建checkpoint_blobs表

```sql
CREATE TABLE checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
```

**v3 → v4**：创建checkpoint_writes表

```sql
CREATE TABLE checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
```

**v4 → v5**：blob字段允许NULL

```sql
ALTER TABLE checkpoint_blobs ALTER COLUMN blob DROP NOT NULL;
```

**v6 → v9**：添加索引

```sql
CREATE INDEX CONCURRENTLY checkpoints_thread_id_idx ON checkpoints(thread_id);
CREATE INDEX CONCURRENTLY checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
CREATE INDEX CONCURRENTLY checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
```

**v9 → v10**：添加task_path字段

```sql
ALTER TABLE checkpoint_writes ADD COLUMN task_path TEXT NOT NULL DEFAULT '';
```

### 6.2 向后兼容策略

**读取旧版本检查点**：

```python
def _load_checkpoint_tuple(self, value):
    checkpoint = value["checkpoint"]
    
    # 检测版本
    if checkpoint.get("v", 1) < 4:
        # 迁移pending_sends
        if value["parent_checkpoint_id"]:
            self._migrate_pending_sends(...)
    
    # 加载blobs
    channel_values = {
        **checkpoint.get("channel_values", {}),
        **self._load_blobs(value["channel_values"]),
    }
    
    return CheckpointTuple(...)
```

**自动迁移**：

- setup()方法检测当前版本
- 只运行未应用的迁移
- 记录迁移版本到checkpoint_migrations

## 七、存储容量估算

### 7.1 单个检查点大小

**checkpoints表**：

```python
# 基础字段：thread_id(50) + checkpoint_ns(10) + checkpoint_id(36) + parent(36) = 132 bytes
# checkpoint JSONB：
#   - 结构：~200 bytes
#   - inline channel_values：~100 bytes/channel
#   - 总计：~500 bytes
# metadata JSONB：~100 bytes
# 总计：~750 bytes/检查点
```

**checkpoint_blobs表**：

```python
# 每个blob：
#   - 基础字段：~150 bytes
#   - blob数据：取决于内容大小
#   - 示例：10条消息，每条1KB = ~10KB
# 总计：~10KB/blob
```

**checkpoint_writes表**：

```python
# 每个write：
#   - 基础字段：~200 bytes
#   - blob数据：取决于写入内容
#   - 示例：1个ToolMessage，~1KB
# 总计：~1.2KB/write
```

### 7.2 容量估算示例

**场景：100个用户，每人100个检查点**

```python
# 每个检查点：
checkpoint_size = 750  # checkpoints表
blob_size = 10 * 1024  # 10条消息
write_size = 2 * 1200  # 2个writes

total_per_checkpoint = checkpoint_size + blob_size + write_size
# = 750 + 10240 + 2400 = 13,390 bytes ≈ 13KB

# 总存储：
total_users = 100
checkpoints_per_user = 100
total_storage = total_users * checkpoints_per_user * total_per_checkpoint
# = 100 * 100 * 13KB = 130MB
```

**优化建议**：

1. 定期清理旧检查点（如保留最近30天）
2. 使用消息修剪减少blob大小
3. 考虑使用ShallowPostgresSaver
4. 启用数据库压缩

## 八、总结

checkpoint-postgres的数据结构设计体现了以下原则：

1. **分层存储**：主表+Blob表，优化查询性能
2. **灵活序列化**：支持任意Python对象
3. **完整关系**：通过外键维护数据一致性
4. **版本演进**：通过迁移机制平滑升级
5. **存储优化**：Blob分离、索引优化、压缩支持

通过合理的数据结构设计，checkpoint-postgres实现了高性能和可扩展的检查点存储。

---

## 时序图

## 一、时序图总览

本文档提供checkpoint-postgres模块的详细时序图，涵盖：

1. **初始化与setup**：创建连接和数据库表
2. **检查点保存（put）**：完整的保存流程
3. **检查点加载（get_tuple）**：查询和反序列化
4. **检查点列表（list）**：批量查询
5. **Writes保存（put_writes）**：Pending writes存储
6. **Pipeline批量操作**：高性能批量执行
7. **线程删除（delete_thread）**：清理数据

## 二、初始化与setup流程

### 2.1 完整初始化时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户代码
    participant Saver as PostgresSaver
    participant Conn as Connection
    participant DB as PostgreSQL
    
    Note over User,DB: === 场景1：使用from_conn_string ===
    
    User->>Saver: from_conn_string(DB_URI)
    activate Saver
    
    Saver->>Conn: Connection.connect(DB_URI, autocommit=True, row_factory=dict_row)
    activate Conn
    Conn->>DB: 建立TCP连接
    activate DB
    DB-->>Conn: 连接成功
    Conn-->>Saver: conn
    deactivate Conn
    
    Saver->>Saver: __init__(conn, pipe=None, serde=None)
    Note right of Saver: 初始化：<br/>- self.conn = conn<br/>- self.lock = threading.Lock()<br/>- self.serde = JsonPlusSerializer()
    
    Saver-->>User: PostgresSaver实例
    
    Note over User,DB: === 步骤2：setup()创建表 ===
    
    User->>Saver: setup()
    activate Saver
    
    Saver->>Saver: _cursor()
    Note right of Saver: 获取cursor，加锁
    
    Saver->>DB: CREATE TABLE checkpoint_migrations (v INTEGER PRIMARY KEY)
    DB-->>Saver: OK
    
    Saver->>DB: SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1
    DB-->>Saver: current_version (e.g., 5)
    
    Note over Saver: 检测需要运行的迁移
    
    loop 未应用的迁移 (version 6 to 10)
        Saver->>DB: 执行迁移SQL (e.g., CREATE INDEX ...)
        DB-->>Saver: OK
        
        Saver->>DB: INSERT INTO checkpoint_migrations (v) VALUES (?)
        DB-->>Saver: OK
    end
    
    Saver-->>User: setup完成
    deactivate Saver
    deactivate DB
    
    Note over User,DB: === 场景2：使用ConnectionPool ===
    
    User->>User: pool = ConnectionPool(DB_URI, min_size=5, max_size=20)
    User->>Saver: PostgresSaver(pool)
    activate Saver
    
    Saver->>Saver: __init__(pool, pipe=None, serde=None)
    Note right of Saver: 存储连接池引用<br/>不立即创建连接
    
    Saver-->>User: PostgresSaver实例
    
    User->>Saver: setup()
    Saver->>Saver: _cursor()
    Saver->>Conn: pool.getconn()
    activate Conn
    Conn-->>Saver: conn
    
    Note over Saver: 使用连接执行setup<br/>（SQL执行流程同上）
    
    Saver->>Conn: pool.putconn(conn)
    Conn-->>Saver: 连接归还到池
    deactivate Conn
    
    Saver-->>User: setup完成
    deactivate Saver
```

### 2.2 文字说明

#### 2.2.1 图意概述

该时序图展示了PostgresSaver的完整初始化流程，包括创建连接、初始化实例和运行数据库迁移的全过程。支持两种连接模式：单连接和连接池。

#### 2.2.2 关键步骤

**from_conn_string**：

- 创建Connection并设置必要参数
- autocommit=True：确保setup()能提交表创建
- row_factory=dict_row：支持字典访问

**setup()方法**：

1. 创建checkpoint_migrations表
2. 查询当前版本号
3. 运行所有未应用的迁移
4. 记录新版本到migrations表

**迁移顺序**：

```python
MIGRATIONS = [
    # 0: 创建migrations表
    # 1: 创建checkpoints表
    # 2: 创建checkpoint_blobs表
    # 3: 创建checkpoint_writes表
    # 4: 修改blob字段
    # 5: no-op
    # 6-8: 创建索引
    # 9: 添加task_path字段
]
```

#### 2.2.3 边界与异常

**连接失败**：

```python
try:
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
except psycopg.OperationalError as e:
    # 数据库不可达
    # 回退到InMemorySaver
```

**权限不足**：

```python
try:
    checkpointer.setup()
except psycopg.errors.InsufficientPrivilege:
    # 用户没有CREATE TABLE权限
```

**迁移失败**：

- 如果某个迁移失败，会抛出异常
- 已执行的迁移已提交（autocommit=True）
- 需要手动修复数据库后重试

## 三、检查点保存流程

### 3.1 put方法详细时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as Agent/用户
    participant Saver as PostgresSaver
    participant Base as BasePostgresSaver
    participant Serde as JsonPlusSerializer
    participant Cursor as Cursor
    participant DB as PostgreSQL
    
    User->>Saver: put(config, checkpoint, metadata, new_versions)
    activate Saver
    
    Note over Saver: === 步骤1：准备数据 ===
    Saver->>Saver: 提取config信息
    Note right of Saver: thread_id = "user-123"<br/>checkpoint_ns = ""<br/>checkpoint_id = "abc123"<br/>parent_checkpoint_id = "xyz789"
    
    Saver->>Saver: 复制checkpoint
    Note right of Saver: copy = checkpoint.copy()<br/>copy["channel_values"] = copy["channel_values"].copy()
    
    Note over Saver: === 步骤2：分离inline和blob ===
    Saver->>Saver: 遍历channel_values
    
    loop 每个channel
        alt value是原始类型
            Note right of Saver: None, str, int, float, bool<br/>保留在copy["channel_values"]
        else value是复杂对象
            Note right of Saver: list, dict, BaseModel等<br/>移到blob_values
        end
    end
    
    Note right of Saver: 结果：<br/>copy["channel_values"] = {<br/>  "key": "value",<br/>  "count": 42<br/>}<br/>blob_values = {<br/>  "messages": [AIMessage, ToolMessage]<br/>}
    
    Note over Saver: === 步骤3：序列化blobs ===
    Saver->>Base: _dump_blobs(thread_id, checkpoint_ns, blob_values, blob_versions)
    activate Base
    
    loop 每个blob
        Base->>Serde: dumps(blob_value)
        activate Serde
        Serde->>Serde: 序列化对象
        Note right of Serde: 1. 检查是否JSON可序列化<br/>2. 否则使用ormsgpack<br/>3. 返回(type, bytes)
        Serde-->>Base: (type, serialized_bytes)
        deactivate Serde
        
        Base->>Base: 构建行元组
        Note right of Base: (thread_id, checkpoint_ns,<br/> channel, version,<br/> type, blob)
    end
    
    Base-->>Saver: Iterator[blob_rows]
    deactivate Base
    
    Note over Saver: === 步骤4：执行SQL（Pipeline模式） ===
    Saver->>Saver: _cursor(pipeline=True)
    Saver->>Cursor: 获取cursor
    activate Cursor
    
    alt 有blobs需要保存
        Saver->>Cursor: executemany(UPSERT_CHECKPOINT_BLOBS_SQL, blob_rows)
        Note right of Cursor: INSERT INTO checkpoint_blobs<br/>VALUES (%s, %s, %s, %s, %s, %s)<br/>ON CONFLICT DO NOTHING
        
        loop 每个blob_row
            Cursor->>DB: INSERT (batch)
            Note right of DB: 批量插入，一次网络往返
        end
        DB-->>Cursor: OK
    end
    
    Note over Saver: === 步骤5：保存主检查点 ===
    Saver->>Cursor: execute(UPSERT_CHECKPOINTS_SQL, (...))
    Note right of Cursor: INSERT INTO checkpoints<br/>VALUES (%s, %s, %s, %s, %s, %s)<br/>ON CONFLICT DO UPDATE
    
    Cursor->>DB: INSERT/UPDATE checkpoint
    Note right of DB: checkpoint = Jsonb(copy)<br/>metadata = Jsonb(metadata)
    DB-->>Cursor: OK
    
    alt Pipeline模式
        Saver->>Cursor: pipe.sync()
        Note right of Cursor: 提交所有批量操作<br/>一次网络往返
        Cursor->>DB: SYNC
        DB-->>Cursor: 所有操作完成
    end
    
    Cursor-->>Saver: 执行完成
    deactivate Cursor
    
    Note over Saver: === 步骤6：返回新config ===
    Saver->>Saver: 构建next_config
    Note right of Saver: {<br/>  "configurable": {<br/>    "thread_id": "user-123",<br/>    "checkpoint_ns": "",<br/>    "checkpoint_id": "abc123"<br/>  }<br/>}
    
    Saver-->>User: next_config
    deactivate Saver
```

### 3.2 文字说明

#### 3.2.1 图意概述

该时序图展示了检查点保存的完整流程，包括数据准备、inline/blob分离、序列化、SQL执行等步骤。使用Pipeline模式实现高性能批量写入。

#### 3.2.2 关键算法

**Inline/Blob分离**：

```python
def separate_inline_and_blobs(channel_values):
    inline_values = {}
    blob_values = {}
    
    for k, v in channel_values.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            inline_values[k] = v  # 保留在主表
        else:
            blob_values[k] = v  # 移到blob表
    
    return inline_values, blob_values
```

**优势**：

- 主表更小，查询更快
- 大对象单独存储，按需加载
- BYTEA比JSONB更紧凑

**Pipeline批量执行**：

```python
with self._cursor(pipeline=True) as cur:
    # 批量插入blobs
    cur.executemany(UPSERT_CHECKPOINT_BLOBS_SQL, blob_rows)
    
    # 插入主检查点
    cur.execute(UPSERT_CHECKPOINTS_SQL, checkpoint_row)
    
    # 一次性提交（context manager退出时自动调用pipe.sync()）
```

**性能提升**：

- 非Pipeline：N+1次网络往返（N个blobs + 1个checkpoint）
- Pipeline：1次网络往返
- 提升倍数：约等于blobs数量

#### 3.2.3 边界与异常

**序列化失败**：

```python
try:
    type_name, serialized = serde.dumps(value)
except Exception as e:
    # 对象无法序列化
    # 可以尝试：
    # 1. 转换为可序列化格式
    # 2. 跳过该channel
    # 3. 使用自定义序列化器
```

**唯一约束冲突**：

```python
# UPSERT语句自动处理
# ON CONFLICT DO UPDATE：更新现有检查点
# ON CONFLICT DO NOTHING：跳过已存在的blob
```

**事务保证**：

- autocommit=True：每个语句立即提交
- Pipeline：整个pipeline是原子的
- 如果中途失败，已提交的不会回滚

## 四、检查点加载流程

### 4.1 get_tuple方法时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as Agent/用户
    participant Saver as PostgresSaver
    participant Base as BasePostgresSaver
    participant Serde as JsonPlusSerializer
    participant Cursor as Cursor
    participant DB as PostgreSQL
    
    User->>Saver: get_tuple(config)
    activate Saver
    
    Note over Saver: === 步骤1：解析config ===
    Saver->>Saver: 提取配置
    Note right of Saver: thread_id = config["configurable"]["thread_id"]<br/>checkpoint_id = get_checkpoint_id(config)<br/>checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
    
    Note over Saver: === 步骤2：构建SQL ===
    alt 指定checkpoint_id
        Note right of Saver: WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
    else 未指定checkpoint_id
        Note right of Saver: WHERE thread_id = ? AND checkpoint_ns = ?<br/>ORDER BY checkpoint_id DESC LIMIT 1
    end
    
    Note over Saver: === 步骤3：执行查询 ===
    Saver->>Saver: _cursor()
    Saver->>Cursor: execute(SELECT_SQL + where, args)
    activate Cursor
    
    Note over Cursor: SELECT包含：<br/>1. checkpoints表的所有字段<br/>2. 关联的checkpoint_blobs（子查询）<br/>3. 关联的checkpoint_writes（子查询）
    
    Cursor->>DB: 执行复杂JOIN查询
    activate DB
    
    DB->>DB: 查询checkpoints表
    DB->>DB: 关联checkpoint_blobs表
    DB->>DB: 关联checkpoint_writes表
    DB->>DB: 聚合结果
    
    DB-->>Cursor: 结果行（DictRow）
    deactivate DB
    
    Cursor->>Cursor: fetchone()
    Cursor-->>Saver: value (dict)
    Note right of Cursor: {<br/>  "thread_id": "user-123",<br/>  "checkpoint": {...},<br/>  "channel_values": [[...]],<br/>  "pending_writes": [[...]],<br/>  ...<br/>}
    deactivate Cursor
    
    alt value为None
        Saver-->>User: None（检查点不存在）
    end
    
    Note over Saver: === 步骤4：迁移pending_sends（如果需要） ===
    alt checkpoint版本 < 4 且有parent
        Saver->>Cursor: execute(SELECT_PENDING_SENDS_SQL, ...)
        activate Cursor
        Cursor->>DB: 查询pending sends
        DB-->>Cursor: sends数据
        Cursor-->>Saver: sends
        deactivate Cursor
        
        Saver->>Base: _migrate_pending_sends(sends, checkpoint, channel_values)
        activate Base
        Base->>Base: 将sends迁移到新格式
        Base-->>Saver: 迁移完成
        deactivate Base
    end
    
    Note over Saver: === 步骤5：反序列化 ===
    Saver->>Base: _load_checkpoint_tuple(value)
    activate Base
    
    Base->>Base: 解析checkpoint字段
    Note right of Base: checkpoint = value["checkpoint"]<br/>metadata = value["metadata"]
    
    Base->>Base: _load_blobs(value["channel_values"])
    activate Base
    
    loop 每个blob
        Base->>Serde: loads((type, blob))
        activate Serde
        Serde->>Serde: 反序列化
        Note right of Serde: 1. 根据type选择反序列化器<br/>2. JSON或ormsgpack<br/>3. 返回Python对象
        Serde-->>Base: 反序列化对象
        deactivate Serde
    end
    
    Base-->>Base: blobs dict
    deactivate Base
    
    Base->>Base: 合并channel_values
    Note right of Base: channel_values = {<br/>  **checkpoint["channel_values"],<br/>  **loaded_blobs<br/>}
    
    Base->>Base: _load_writes(value["pending_writes"])
    activate Base
    
    loop 每个write
        Base->>Serde: loads((type, blob))
        Serde-->>Base: write value
    end
    
    Base-->>Base: writes list
    deactivate Base
    
    Base->>Base: 构建CheckpointTuple
    Note right of Base: CheckpointTuple(<br/>  config={...},<br/>  checkpoint={...},<br/>  metadata={...},<br/>  parent_config={...},<br/>  pending_writes=[...]<br/>)
    
    Base-->>Saver: CheckpointTuple
    deactivate Base
    
    Saver-->>User: CheckpointTuple
    deactivate Saver
```

### 4.2 文字说明

#### 4.2.1 图意概述

该时序图展示了检查点加载的完整流程，包括SQL查询、关联查询、pending_sends迁移、反序列化等步骤。

#### 4.2.2 关键SQL

**SELECT_SQL**：

```sql
SELECT
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        -- 子查询：加载blobs
        SELECT array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
        FROM jsonb_each_text(checkpoint -> 'channel_versions')
        INNER JOIN checkpoint_blobs bl
            ON bl.thread_id = checkpoints.thread_id
            AND bl.checkpoint_ns = checkpoints.checkpoint_ns
            AND bl.channel = jsonb_each_text.key
            AND bl.version = jsonb_each_text.value
    ) AS channel_values,
    (
        -- 子查询：加载pending writes
        SELECT array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] ORDER BY cw.task_id, cw.idx)
        FROM checkpoint_writes cw
        WHERE cw.thread_id = checkpoints.thread_id
            AND cw.checkpoint_ns = checkpoints.checkpoint_ns
            AND cw.checkpoint_id = checkpoints.checkpoint_id
    ) AS pending_writes
FROM checkpoints
WHERE ...
```

**优势**：

- 一次查询获取所有数据
- 避免N+1查询问题
- 数据库端完成JOIN，减少网络传输

#### 4.2.3 pending_sends迁移

**为什么需要迁移**：

- v3检查点使用不同的pending_sends格式
- v4改为使用checkpoint_writes表
- 加载v3检查点时需要动态迁移

**迁移逻辑**：

```python
if checkpoint["v"] < 4 and parent_checkpoint_id:
    # 查询父检查点的pending sends
    sends = query_pending_sends(parent_checkpoint_id)
    
    # 迁移到新格式
    migrate_pending_sends(sends, checkpoint, channel_values)
```

#### 4.2.4 性能考虑

**索引优化**：

```sql
-- 查询最新检查点
CREATE INDEX checkpoints_thread_id_idx ON checkpoints(thread_id);

-- 加速JOIN
CREATE INDEX checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
CREATE INDEX checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
```

**查询性能**：

- 单个检查点查询：<10ms（有索引）
- 包含10个blobs：<50ms
- 瓶颈：反序列化（不是SQL）

## 五、Pipeline批量操作流程

### 5.1 Pipeline模式时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Saver as PostgresSaver
    participant Pipeline as Pipeline
    participant Conn as Connection
    participant DB as PostgreSQL
    
    Note over User,DB: === 创建Pipeline模式的Saver ===
    
    User->>Conn: Connection.connect(DB_URI)
    Conn-->>User: conn
    
    User->>Conn: conn.pipeline()
    activate Conn
    Conn->>Pipeline: 创建Pipeline实例
    activate Pipeline
    Pipeline-->>Conn: pipe
    Conn-->>User: pipe
    
    User->>Saver: PostgresSaver(conn, pipe)
    activate Saver
    Saver-->>User: saver
    
    Note over User,DB: === 批量保存检查点 ===
    
    loop 10个检查点
        User->>Saver: put(config, checkpoint_i, metadata, versions)
        
        Saver->>Saver: 分离inline/blob
        Saver->>Saver: 序列化blobs
        
        Note over Saver: 执行SQL，但不立即发送
        Saver->>Pipeline: executemany(UPSERT_CHECKPOINT_BLOBS_SQL, blobs)
        Note right of Pipeline: 添加到批次，不发送
        
        Saver->>Pipeline: execute(UPSERT_CHECKPOINTS_SQL, checkpoint)
        Note right of Pipeline: 添加到批次，不发送
        
        Saver-->>User: next_config
    end
    
    Note over User,DB: === 一次性提交所有操作 ===
    
    User->>Pipeline: sync()
    Note right of Pipeline: 将所有批次的SQL语句<br/>一次性发送到数据库
    
    Pipeline->>DB: 发送批量SQL（一次网络往返）
    activate DB
    Note right of DB: 执行所有SQL：<br/>- 30个INSERT blobs<br/>- 10个INSERT checkpoints
    DB-->>Pipeline: 所有操作完成
    deactivate DB
    
    Pipeline-->>User: sync完成
    
    deactivate Pipeline
    deactivate Conn
    deactivate Saver
    
    Note over User,DB: === 性能对比 ===
    
    Note over User: 非Pipeline模式：<br/>10个检查点 × 4次往返/检查点 = 40次往返
    
    Note over User: Pipeline模式：<br/>1次往返
    
    Note over User: 性能提升：约40倍（网络延迟为主要瓶颈时）
```

### 5.2 文字说明

#### 5.2.1 图意概述

该时序图展示了Pipeline模式如何通过批量执行SQL来减少网络往返，从而大幅提升性能。

#### 5.2.2 Pipeline工作原理

**批处理机制**：

```python
with conn.pipeline() as pipe:
    checkpointer = PostgresSaver(conn, pipe)
    
    # 所有SQL都添加到批次，不立即发送
    checkpointer.put(config1, ...)
    checkpointer.put(config2, ...)
    checkpointer.put(config3, ...)
    
    # context manager退出时自动调用pipe.sync()
    # 一次性发送所有SQL
```

**性能提升**：

- 减少网络往返
- 减少TCP开销
- 数据库端可以优化执行计划

**适用场景**：

- 批量保存检查点
- 批量保存writes
- 初始化/迁移数据

## 六、总结

checkpoint-postgres模块的时序图展示了：

1. **初始化**：连接管理和数据库迁移
2. **保存**：Inline/Blob分离和Pipeline批量执行
3. **加载**：复杂JOIN查询和反序列化
4. **Pipeline**：批量操作性能优化

通过这些时序图，可以深入理解checkpoint-postgres的工作机制，为性能优化和问题排查提供指导。

---
