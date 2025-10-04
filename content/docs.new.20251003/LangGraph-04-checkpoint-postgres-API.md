# LangGraph-04-checkpoint-postgres-API

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

