# LangGraph-05-checkpoint-sqlite-数据结构

## 一、数据结构总览

checkpoint-sqlite的数据结构与checkpoint-postgres完全相同，只是底层数据库从PostgreSQL换成SQLite。

详细的数据结构说明请参考：`LangGraph-04-checkpoint-postgres-数据结构.md`

## 二、数据库Schema

### 2.1 表结构（与PostgreSQL相同）

```sql
-- checkpoints表
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB NOT NULL,
    metadata BLOB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- checkpoint_blobs表
CREATE TABLE checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

-- checkpoint_writes表
CREATE TABLE checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- checkpoint_migrations表
CREATE TABLE checkpoint_migrations (
    v INTEGER PRIMARY KEY
);
```

### 2.2 SQLite vs PostgreSQL差异

| 特性 | PostgreSQL | SQLite |
|---|---|---|
| JSON类型 | JSONB（高效） | BLOB（序列化JSON） |
| 数组类型 | ARRAY | 不支持，使用JOIN |
| UUID类型 | UUID | TEXT |
| 子查询性能 | 高 | 中等 |
| 索引类型 | B-tree, GiST等 | B-tree |

## 三、SQLite特定优化

### 3.1 索引策略

```sql
-- 主键索引（自动创建）
CREATE UNIQUE INDEX checkpoints_pkey 
ON checkpoints(thread_id, checkpoint_ns, checkpoint_id);

-- 查询优化索引
CREATE INDEX checkpoints_thread_id_idx 
ON checkpoints(thread_id);

CREATE INDEX checkpoint_blobs_thread_id_idx 
ON checkpoint_blobs(thread_id);

CREATE INDEX checkpoint_writes_thread_id_idx 
ON checkpoint_writes(thread_id);
```

### 3.2 性能配置

```python
import sqlite3

conn = sqlite3.connect("checkpoints.db")

# WAL模式（提高并发）
conn.execute("PRAGMA journal_mode=WAL")

# 同步模式
conn.execute("PRAGMA synchronous=NORMAL")

# 缓存大小（负数表示KB，正数表示页数）
conn.execute("PRAGMA cache_size=-64000")  # 64MB

# 临时存储
conn.execute("PRAGMA temp_store=MEMORY")

# 内存映射（加速读取）
conn.execute("PRAGMA mmap_size=268435456")  # 256MB
```

## 四、存储容量估算

与PostgreSQL相同的估算方法，但SQLite的存储效率略低：

```python
# 单个检查点
checkpoint_size = 800  # vs 750 in PostgreSQL（BLOB vs JSONB）
blob_size = 10 * 1024  # 相同
write_size = 2 * 1200  # 相同

total_per_checkpoint = checkpoint_size + blob_size + write_size
# ≈ 13.2KB per checkpoint

# 100个用户，每人100个检查点
total_storage = 100 * 100 * 13.2KB = 132MB

# 加上索引和开销
total_with_overhead = 132MB * 1.3 = 171MB
```

## 五、数据迁移

### 5.1 从SQLite迁移到PostgreSQL

```python
import sqlite3
import psycopg

# 连接两个数据库
sqlite_conn = sqlite3.connect("checkpoints.db")
pg_conn = psycopg.connect(PG_URI)

# 迁移数据
for table in ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]:
    # 从SQLite读取
    rows = sqlite_conn.execute(f"SELECT * FROM {table}").fetchall()
    
    # 写入PostgreSQL
    with pg_conn.cursor() as cur:
        for row in rows:
            cur.execute(f"INSERT INTO {table} VALUES ({','.join(['%s']*len(row))})", row)

pg_conn.commit()
```

## 六、总结

checkpoint-sqlite使用与PostgreSQL相同的数据结构，但底层实现有差异。适合单机部署和开发测试，生产环境建议使用PostgreSQL。

