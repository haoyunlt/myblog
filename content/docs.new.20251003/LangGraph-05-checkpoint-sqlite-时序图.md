# LangGraph-05-checkpoint-sqlite-时序图

## 一、时序图总览

checkpoint-sqlite的时序图与checkpoint-postgres基本相同，只是数据库操作从PostgreSQL换成SQLite。

详细的时序图请参考：`LangGraph-04-checkpoint-postgres-时序图.md`

## 二、主要差异

### 2.1 连接管理

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Saver as SqliteSaver
    participant Conn as sqlite3.Connection
    participant DB as SQLite数据库文件
    
    User->>Saver: from_conn_string("./checkpoints.db")
    activate Saver
    
    Saver->>Conn: sqlite3.connect("./checkpoints.db")
    activate Conn
    Conn->>DB: 打开/创建数据库文件
    activate DB
    DB-->>Conn: 连接成功
    Conn-->>Saver: conn
    deactivate Conn
    
    Saver->>Saver: __init__(conn)
    Note right of Saver: 初始化：<br/>- self.conn = conn<br/>- self.lock = threading.Lock()
    
    Saver-->>User: SqliteSaver实例
    
    User->>Saver: setup()
    Saver->>DB: 创建表和索引
    DB-->>Saver: OK
    
    deactivate Saver
    deactivate DB
```

**关键差异**：
- 无需网络连接，直接操作本地文件
- 连接更快（<1ms vs 10-50ms for PostgreSQL）
- 无ConnectionPool（SQLite本身不支持）

### 2.2 写操作串行化

```mermaid
sequenceDiagram
    autonumber
    participant Thread1 as 线程1
    participant Thread2 as 线程2
    participant Lock as threading.Lock
    participant Saver as SqliteSaver
    participant DB as SQLite
    
    Thread1->>Lock: acquire()
    Lock-->>Thread1: 获得锁
    
    Thread1->>Saver: put(config, checkpoint, ...)
    activate Saver
    Saver->>DB: BEGIN TRANSACTION
    Saver->>DB: INSERT blobs
    Saver->>DB: INSERT checkpoint
    Saver->>DB: COMMIT
    DB-->>Saver: OK
    Saver-->>Thread1: success
    deactivate Saver
    
    Thread1->>Lock: release()
    
    Note over Thread2: 等待锁释放
    
    Thread2->>Lock: acquire()
    Lock-->>Thread2: 获得锁
    
    Thread2->>Saver: put(config, checkpoint, ...)
    activate Saver
    Note over Saver,DB: 执行写操作...
    Saver-->>Thread2: success
    deactivate Saver
    
    Thread2->>Lock: release()
```

**关键差异**：
- SQLite写操作必须串行（数据库级别的锁）
- 应用层使用threading.Lock避免竞争
- 多个写操作会排队等待

### 2.3 查询操作

查询操作与PostgreSQL相同，但SQL语法略有差异：

```sql
-- PostgreSQL使用JSONB操作符
SELECT checkpoint -> 'channel_values' FROM checkpoints;

-- SQLite使用JSON函数
SELECT json_extract(checkpoint, '$.channel_values') FROM checkpoints;
```

## 三、性能对比

### 3.1 操作延迟

| 操作 | PostgreSQL | SQLite |
|---|---|---|
| 连接 | 10-50ms（网络） | <1ms（本地） |
| 单次put | 5-10ms | 2-5ms |
| 单次get | 3-5ms | 1-3ms |
| 批量put（10个） | 20ms（Pipeline） | 20-50ms（串行） |
| list（100个） | 50ms | 30-50ms |

### 3.2 并发性能

```
PostgreSQL:
- 读并发：高（多连接）
- 写并发：高（行级锁）
- 适合：多实例、高并发

SQLite:
- 读并发：中等（多读者单写者）
- 写并发：低（数据库锁）
- 适合：单实例、低并发
```

## 四、最佳实践时序

### 4.1 批量操作优化

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Saver as SqliteSaver
    participant DB as SQLite
    
    User->>Saver: 开始批量操作
    
    Saver->>DB: BEGIN TRANSACTION
    activate DB
    
    loop 10个检查点
        Saver->>DB: INSERT blobs
        Saver->>DB: INSERT checkpoint
    end
    
    Saver->>DB: COMMIT
    DB-->>Saver: OK
    deactivate DB
    
    Saver-->>User: 批量操作完成
    
    Note over User: 时间：~50ms<br/>vs 单独操作：~200ms
```

**优化策略**：
```python
conn = sqlite3.connect("checkpoints.db")
conn.execute("BEGIN TRANSACTION")

for checkpoint in checkpoints:
    checkpointer.put(config, checkpoint, metadata, versions)

conn.execute("COMMIT")
```

### 4.2 并发读取

```mermaid
sequenceDiagram
    autonumber
    participant Thread1 as 读线程1
    participant Thread2 as 读线程2
    participant Thread3 as 读线程3
    participant DB as SQLite
    
    par 并发读取
        Thread1->>DB: SELECT checkpoint WHERE thread_id=1
        DB-->>Thread1: result1
    and
        Thread2->>DB: SELECT checkpoint WHERE thread_id=2
        DB-->>Thread2: result2
    and
        Thread3->>DB: SELECT checkpoint WHERE thread_id=3
        DB-->>Thread3: result3
    end
    
    Note over DB: SQLite支持多个并发读取<br/>（WAL模式下）
```

## 五、总结

checkpoint-sqlite的时序流程与checkpoint-postgres相似，主要差异在于：

1. **连接更简单**：无需网络，直接打开文件
2. **写操作串行**：需要应用层锁保护
3. **性能差异**：单次操作更快，批量操作较慢
4. **适用场景**：开发测试、单机部署

详细的时序图和代码示例请参考`LangGraph-04-checkpoint-postgres-时序图.md`。

