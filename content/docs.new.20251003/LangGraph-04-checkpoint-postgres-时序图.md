# LangGraph-04-checkpoint-postgres-时序图

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

