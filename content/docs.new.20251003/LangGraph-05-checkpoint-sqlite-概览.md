# LangGraph-05-checkpoint-sqlite-概览

## 一、模块职责

checkpoint-sqlite模块是LangGraph检查点系统的SQLite实现，提供轻量级的本地状态持久化能力。该模块与checkpoint-postgres模块功能相同，但使用SQLite数据库，适用于单机部署、开发测试和嵌入式场景。

### 1.1 核心能力

1. **轻量级存储**：无需独立数据库服务，文件即数据库
2. **零配置**：无需安装配置数据库服务器
3. **完整功能**：与PostgreSQL版本功能相同
4. **跨平台**：支持Windows、Linux、macOS

## 二、与PostgreSQL版本的区别

### 2.1 相同之处

- API完全相同
- 数据库schema相同
- 序列化机制相同
- 支持所有检查点功能

### 2.2 不同之处

| 特性 | PostgreSQL | SQLite |
|---|---|---|
| 部署 | 需要独立服务器 | 本地文件 |
| 并发 | 高并发支持 | 写并发受限 |
| 连接 | 需要网络连接 | 本地I/O |
| 性能 | 高性能 | 中等性能 |
| 适用场景 | 生产环境、多实例 | 开发测试、单机部署 |

## 三、使用示例

### 3.1 基本使用

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 方式1：使用文件数据库
DB_PATH = "./checkpoints.db"
with SqliteSaver.from_conn_string(f"file:{DB_PATH}") as checkpointer:
    checkpointer.setup()
    
    # 创建Agent
    agent = create_react_agent(model, tools, checkpointer=checkpointer)
    result = agent.invoke(input, config)

# 方式2：使用内存数据库（测试）
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    checkpointer.setup()
    # 使用checkpointer...
```

### 3.2 异步使用

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    async with AsyncSqliteSaver.from_conn_string("./checkpoints.db") as checkpointer:
        await checkpointer.setup()
        
        config = {"configurable": {"thread_id": "1"}}
        checkpoint = await checkpointer.aget_tuple(config)
        print(checkpoint)

asyncio.run(main())
```

## 四、API参考

### 4.1 SqliteSaver

```python
class SqliteSaver(BaseSqliteSaver):
    """SQLite检查点存储器（同步）"""
    
    def __init__(self, conn: sqlite3.Connection, serde: SerializerProtocol | None = None):
        """初始化
        
        Args:
            conn: SQLite连接
            serde: 序列化器（默认JsonPlusSerializer）
        """
    
    @classmethod
    @contextmanager
    def from_conn_string(cls, conn_string: str) -> Iterator[SqliteSaver]:
        """从连接字符串创建"""
    
    def setup(self) -> None:
        """创建表并运行迁移"""
    
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """获取检查点"""
    
    def list(self, config: RunnableConfig | None, **kwargs) -> Iterator[CheckpointTuple]:
        """列出检查点"""
    
    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        """保存检查点"""
    
    def put_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        """保存writes"""
    
    def delete_thread(self, thread_id: str) -> None:
        """删除线程"""
```

### 4.2 AsyncSqliteSaver

```python
class AsyncSqliteSaver(BaseSqliteSaver):
    """SQLite检查点存储器（异步）"""
    
    async def setup(self) -> None:
        """异步创建表"""
    
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """异步获取检查点"""
    
    def alist(self, config: RunnableConfig | None, **kwargs) -> AsyncIterator[CheckpointTuple]:
        """异步列出检查点"""
    
    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        """异步保存检查点"""
    
    async def aput_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        """异步保存writes"""
    
    async def adelete_thread(self, thread_id: str) -> None:
        """异步删除线程"""
```

## 五、数据库Schema

与checkpoint-postgres完全相同：

- `checkpoints`：主检查点表
- `checkpoint_blobs`：大对象表
- `checkpoint_writes`：待写入表
- `checkpoint_migrations`：迁移版本表

## 六、最佳实践

### 6.1 开发环境

```python
# 使用内存数据库（快速测试）
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    checkpointer.setup()
    # 测试...

# 使用文件数据库（持久化测试）
with SqliteSaver.from_conn_string("./test.db") as checkpointer:
    checkpointer.setup()
    # 测试...
```

### 6.2 生产环境（单机）

```python
from pathlib import Path

# 创建数据目录
DB_DIR = Path("./data")
DB_DIR.mkdir(exist_ok=True)

DB_PATH = DB_DIR / "checkpoints.db"
with SqliteSaver.from_conn_string(f"file:{DB_PATH}") as checkpointer:
    checkpointer.setup()
    
    # 生产应用...
```

### 6.3 并发控制

```python
# SQLite写并发受限，使用锁保护
import threading

lock = threading.Lock()

def save_checkpoint(checkpointer, config, checkpoint, metadata, versions):
    with lock:
        checkpointer.put(config, checkpoint, metadata, versions)
```

### 6.4 性能优化

```python
import sqlite3

# 启用WAL模式（提高并发）
conn = sqlite3.connect("./checkpoints.db")
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("PRAGMA cache_size=10000")

checkpointer = SqliteSaver(conn)
```

## 七、迁移指南

### 7.1 从InMemorySaver迁移

```python
# 之前：
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# 之后：
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("./checkpoints.db") as checkpointer:
    checkpointer.setup()
```

### 7.2 迁移到PostgreSQL

当应用规模增长，需要迁移到PostgreSQL：

1. **导出SQLite数据**
2. **创建PostgreSQL数据库**
3. **导入数据**
4. **切换checkpointer**

```python
# 新代码
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
```

## 八、限制与注意事项

### 8.1 写并发

- SQLite同时只能有一个写操作
- 多个进程同时写入会导致`SQLITE_BUSY`错误
- 解决：使用锁或切换到PostgreSQL

### 8.2 数据库大小

- SQLite单文件，大小不宜过大（建议<10GB）
- 定期清理旧检查点
- 大规模应用使用PostgreSQL

### 8.3 备份

```python
import shutil

# 简单备份：复制文件
shutil.copy("checkpoints.db", "checkpoints.db.backup")

# 在线备份（SQLite 3.27+）
import sqlite3
src = sqlite3.connect("checkpoints.db")
dst = sqlite3.connect("checkpoints.db.backup")
src.backup(dst)
```

## 九、总结

checkpoint-sqlite模块提供了轻量级的检查点存储方案：

**优势**：
- 零配置，开箱即用
- 部署简单，单个文件
- API与PostgreSQL完全相同
- 适合开发测试和单机部署

**劣势**：
- 写并发受限
- 不适合大规模应用
- 不支持分布式

**适用场景**：
- 开发和测试
- 单机应用
- 嵌入式系统
- 原型验证

如需更高性能和并发，请使用checkpoint-postgres模块。

