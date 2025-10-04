# LangGraph-04-checkpoint-postgres-概览

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

