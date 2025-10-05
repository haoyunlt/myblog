---
title: "ClickHouse-20-实战经验与最佳实践"
date: 2024-12-28T10:20:00+08:00
series: ["ClickHouse源码剖析"]
categories: ['ClickHouse']
tags: ['ClickHouse', '源码剖析', '列式数据库']
description: "ClickHouse 实战经验与最佳实践模块源码剖析 - 详细分析实战经验与最佳实践模块的架构设计、核心功能和实现机制"
---


# ClickHouse-20-实战经验与最佳实践

## 框架使用示例

### 示例 1：使用 TCP 原生客户端

```bash
# 安装 clickhouse-client
curl https://clickhouse.com/ | sh

# 连接到 ClickHouse
./clickhouse client --host 127.0.0.1 --port 9000 --user default --password ''

# 执行查询
SELECT * FROM system.tables LIMIT 10;

# 插入数据
INSERT INTO my_table (id, name, value) VALUES (1, 'test', 100);

# 批量插入
INSERT INTO my_table SELECT number, concat('name_', toString(number)), number * 10 FROM numbers(1000000);
```

### 示例 2：使用 HTTP 接口

```python
import requests
import json

# 配置
url = 'http://localhost:8123/'
user = 'default'
password = ''

# 执行查询
def execute_query(sql):
    params = {
        'user': user,
        'password': password,
        'database': 'default',
        'query': sql
    }
    response = requests.post(url, params=params)
    return response.text

# SELECT 查询
result = execute_query('SELECT * FROM system.tables FORMAT JSONEachRow')
for line in result.strip().split('\n'):
    print(json.loads(line))

# INSERT 查询
data = "1\tAlice\t100\n2\tBob\t200\n"
sql = "INSERT INTO my_table FORMAT TabSeparated"
requests.post(url, params={'query': sql}, data=data)

# 使用会话
session_id = 'my_session_001'
params = {
    'session_id': session_id,
    'session_timeout': 3600
}

# 创建临时表
execute_query('CREATE TEMPORARY TABLE temp AS SELECT 1')
# 在同一会话中使用临时表
execute_query('SELECT * FROM temp')
```

### 示例 3：使用 Python 客户端（clickhouse-driver）

```python
from clickhouse_driver import Client

# 创建客户端
client = Client(
    host='localhost',
    port=9000,
    user='default',
    password='',
    database='default',
    settings={'max_threads': 8}
)

# 简单查询
result = client.execute('SELECT * FROM system.tables LIMIT 5')
for row in result:
    print(row)

# 参数化查询
result = client.execute(
    'SELECT * FROM system.tables WHERE database = %(db)s',
    {'db': 'system'}
)

# 批量插入
data = [
    (1, 'Alice', 100),
    (2, 'Bob', 200),
    (3, 'Charlie', 300)
]
client.execute('INSERT INTO my_table (id, name, value) VALUES', data)

# 流式查询（处理大结果集）
with client.execute_iter('SELECT * FROM large_table') as result:
    for row in result:
        process_row(row)

# 异步批量插入
from clickhouse_driver import Client
from concurrent.futures import ThreadPoolExecutor

def insert_batch(batch):
    client = Client(host='localhost')
    client.execute('INSERT INTO my_table VALUES', batch)

batches = [data[i:i+10000] for i in range(0, len(data), 10000)]
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(insert_batch, batches)
```

### 示例 4：使用 JDBC 驱动（MySQL 协议）

```java
import java.sql.*;

public class ClickHouseExample {
    public static void main(String[] args) {
        String url = "jdbc:clickhouse://localhost:9004/default";
        String user = "default";
        String password = "";
        
        try (Connection conn = DriverManager.getConnection(url, user, password)) {
            // 查询数据
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery("SELECT * FROM system.tables LIMIT 10");
            
            while (rs.next()) {
                System.out.println(
                    rs.getString("database") + "." + rs.getString("name")
                );
            }
            
            // 预处理语句
            String sql = "INSERT INTO my_table (id, name, value) VALUES (?, ?, ?)";
            PreparedStatement pstmt = conn.prepareStatement(sql);
            
            for (int i = 0; i < 1000; i++) {
                pstmt.setInt(1, i);
                pstmt.setString(2, "name_" + i);
                pstmt.setInt(3, i * 10);
                pstmt.addBatch();
                
                if (i % 100 == 0) {
                    pstmt.executeBatch();
                }
            }
            pstmt.executeBatch();
            
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```

### 示例 5：分布式查询

```sql
-- 创建分布式表
CREATE TABLE events_local ON CLUSTER my_cluster
(
    event_date Date,
    user_id UInt64,
    event_type String,
    value Float64
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/events_local', '{replica}')
PARTITION BY toYYYYMM(event_date)
ORDER BY (event_date, user_id);

-- 创建分布式表（全局视图）
CREATE TABLE events_distributed ON CLUSTER my_cluster AS events_local
ENGINE = Distributed(my_cluster, default, events_local, rand());

-- 插入数据（自动分片）
INSERT INTO events_distributed SELECT
    today() - number % 30,
    number % 10000,
    ['click', 'view', 'purchase'][number % 3 + 1],
    rand() % 1000 / 10.0
FROM numbers(1000000);

-- 查询数据（自动聚合）
SELECT
    event_type,
    count() as cnt,
    avg(value) as avg_value
FROM events_distributed
WHERE event_date >= today() - 7
GROUP BY event_type
ORDER BY cnt DESC;
```

## 最佳实践

### 表设计最佳实践

#### 1. 选择合适的表引擎

```sql
-- OLAP 分析场景：MergeTree
CREATE TABLE analytics_events
(
    date Date,
    user_id UInt64,
    event String,
    value Float64
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

-- 需要实时更新：ReplacingMergeTree
CREATE TABLE user_profiles
(
    user_id UInt64,
    name String,
    age UInt8,
    update_time DateTime
)
ENGINE = ReplacingMergeTree(update_time)
ORDER BY user_id;

-- 累加场景：SummingMergeTree
CREATE TABLE user_stats
(
    date Date,
    user_id UInt64,
    events UInt64,
    revenue Float64
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

-- 聚合预计算：AggregatingMergeTree
CREATE TABLE metrics_agg
(
    date Date,
    metric_name String,
    value AggregateFunction(quantiles(0.5, 0.9, 0.99), Float64)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, metric_name);

-- 分布式场景：Distributed + Replicated
CREATE TABLE events_replicated ON CLUSTER cluster
(
    date Date,
    data String
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/events', '{replica}')
PARTITION BY toYYYYMM(date)
ORDER BY date;
```

#### 2. 优化 ORDER BY 键

```sql
-- 不好的设计（高基数列在前）
CREATE TABLE bad_order
(
    user_id UInt64,        -- 高基数
    date Date,             -- 低基数
    event String
)
ENGINE = MergeTree()
ORDER BY (user_id, date);  -- 索引效率低

-- 好的设计（低基数列在前）
CREATE TABLE good_order
(
    date Date,             -- 低基数（查询常用过滤）
    event String,          -- 中基数
    user_id UInt64         -- 高基数
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, event, user_id);  -- 索引效率高

-- 根据查询模式优化
-- 如果常查：WHERE date = X AND event = Y
-- 则：ORDER BY (date, event, user_id)
```

#### 3. 合理使用分区

```sql
-- 按月分区（推荐用于时间序列数据）
PARTITION BY toYYYYMM(date)

-- 按天分区（仅用于数据量巨大的场景）
PARTITION BY toDate(date)

-- 多维分区（慎用）
PARTITION BY (toYYYYMM(date), event_type)

-- 注意事项
-- - 分区数不宜过多（< 1000）
-- - 每个分区至少几 GB 数据
-- - 可以按分区删除数据（ALTER TABLE DROP PARTITION）
```

#### 4. 使用物化视图加速查询

```sql
-- 原始表
CREATE TABLE events
(
    date Date,
    user_id UInt64,
    event String,
    value Float64
)
ENGINE = MergeTree()
ORDER BY (date, user_id);

-- 物化视图（实时聚合）
CREATE MATERIALIZED VIEW events_daily_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, event)
AS SELECT
    date,
    event,
    count() as event_count,
    sum(value) as total_value
FROM events
GROUP BY date, event;

-- 查询物化视图（速度快）
SELECT * FROM events_daily_stats WHERE date = today();
```

### 查询优化最佳实践

#### 1. 使用 PREWHERE 过滤

```sql
-- 不好（WHERE 在读取所有列后过滤）
SELECT user_id, event, data
FROM events
WHERE date = today() AND event = 'click';

-- 好（PREWHERE 先过滤，减少读取列数）
SELECT user_id, event, data
FROM events
PREWHERE date = today() AND event = 'click';

-- ClickHouse 会自动优化简单条件
-- 但复杂查询手动指定 PREWHERE 更好
```

#### 2. 投影优化

```sql
-- 不好（读取不必要的列）
SELECT * FROM large_table WHERE id = 123;

-- 好（只读取需要的列）
SELECT id, name, value FROM large_table WHERE id = 123;

-- 列式存储的优势：只读取需要的列
```

#### 3. 合理使用聚合

```sql
-- 不好（全表聚合）
SELECT event, count() FROM events GROUP BY event;

-- 好（先过滤再聚合）
SELECT event, count()
FROM events
WHERE date >= today() - 7
GROUP BY event;

-- 更好（使用物化视图）
SELECT event, sum(event_count)
FROM events_daily_stats
WHERE date >= today() - 7
GROUP BY event;
```

#### 4. JOIN 优化

```sql
-- 确保小表在右侧（构建哈希表）
SELECT *
FROM large_table AS l
INNER JOIN small_table AS s ON l.id = s.id;

-- 使用 GLOBAL JOIN 在分布式查询中
SELECT *
FROM distributed_table AS d
GLOBAL INNER JOIN small_table AS s ON d.id = s.id;

-- 使用字典替代 JOIN（更高效）
CREATE DICTIONARY user_dict
(
    user_id UInt64,
    name String
)
PRIMARY KEY user_id
SOURCE(CLICKHOUSE(TABLE 'users'))
LIFETIME(3600);

SELECT
    user_id,
    dictGet('user_dict', 'name', user_id) AS user_name
FROM events;
```

### 写入优化最佳实践

#### 1. 批量写入

```python
# 不好（逐行插入）
for row in data:
    client.execute('INSERT INTO table VALUES', [row])  # 慢

# 好（批量插入）
batch_size = 10000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    client.execute('INSERT INTO table VALUES', batch)  # 快

# 更好（使用 Buffer 引擎）
CREATE TABLE events_buffer AS events
ENGINE = Buffer(default, events, 16, 10, 100, 10000, 1000000, 10000000, 100000000);

-- 写入 Buffer 表（异步刷写到主表）
INSERT INTO events_buffer VALUES (...);
```

#### 2. 异步插入

```sql
-- 启用异步插入
SET async_insert = 1;
SET wait_for_async_insert = 0;  -- 不等待插入完成

-- 插入立即返回，后台异步写入
INSERT INTO table VALUES (...);

-- 配置参数
SET async_insert_max_data_size = 1000000;  -- 批次大小
SET async_insert_busy_timeout_ms = 1000;   -- 最大等待时间
```

#### 3. 去重策略

```sql
-- 使用 ReplacingMergeTree 自动去重
CREATE TABLE dedup_table
(
    id UInt64,
    data String,
    version UInt64
)
ENGINE = ReplacingMergeTree(version)
ORDER BY id;

-- 使用 insert_deduplication_token 幂等插入
INSERT INTO table
SETTINGS insert_deduplication_token = 'unique_token_123'
VALUES (...);

-- 重复执行不会插入重复数据
```

### 分布式部署最佳实践

#### 1. 集群配置

```xml
<!-- config.xml -->
<clickhouse>
    <remote_servers>
        <my_cluster>
            <!-- 分片 1 -->
            <shard>
                <replica>
                    <host>node1</host>
                    <port>9000</port>
                </replica>
                <replica>
                    <host>node2</host>
                    <port>9000</port>
                </replica>
            </shard>
            
            <!-- 分片 2 -->
            <shard>
                <replica>
                    <host>node3</host>
                    <port>9000</port>
                </replica>
                <replica>
                    <host>node4</host>
                    <port>9000</port>
                </replica>
            </shard>
        </my_cluster>
    </remote_servers>
    
    <!-- ZooKeeper 配置 -->
    <zookeeper>
        <node>
            <host>zk1</host>
            <port>2181</port>
        </node>
        <node>
            <host>zk2</host>
            <port>2181</port>
        </node>
        <node>
            <host>zk3</host>
            <port>2181</port>
        </node>
    </zookeeper>
</clickhouse>
```

#### 2. 副本同步

```sql
-- 创建复制表
CREATE TABLE replicated_table ON CLUSTER my_cluster
(
    date Date,
    data String
)
ENGINE = ReplicatedMergeTree(
    '/clickhouse/tables/{shard}/replicated_table',
    '{replica}'
)
PARTITION BY toYYYYMM(date)
ORDER BY date;

-- 数据自动同步到副本
INSERT INTO replicated_table VALUES (today(), 'test');

-- 监控副本状态
SELECT * FROM system.replicas WHERE table = 'replicated_table';
```

#### 3. 负载均衡

```python
from random import choice

nodes = [
    'http://node1:8123',
    'http://node2:8123',
    'http://node3:8123'
]

# 随机选择节点
def execute_query(sql):
    node = choice(nodes)
    response = requests.post(node, params={'query': sql})
    return response.text

# 或使用 Distributed 表（ClickHouse 内置负载均衡）
SELECT * FROM events_distributed;  -- 自动分发到各个分片
```

### 性能监控与调优

#### 1. 关键监控指标

```sql
-- 查看当前查询
SELECT
    query_id,
    user,
    elapsed,
    read_rows,
    memory_usage
FROM system.processes
ORDER BY elapsed DESC;

-- 查询历史
SELECT
    query,
    type,
    query_duration_ms,
    read_rows,
    result_rows,
    memory_usage
FROM system.query_log
WHERE event_date = today()
ORDER BY query_duration_ms DESC
LIMIT 10;

-- 慢查询分析
SELECT
    normalized_query_hash,
    any(query) as example_query,
    count() as times,
    avg(query_duration_ms) as avg_duration,
    max(query_duration_ms) as max_duration
FROM system.query_log
WHERE event_date = today()
  AND type = 'QueryFinish'
  AND query_duration_ms > 1000
GROUP BY normalized_query_hash
ORDER BY avg_duration DESC;

-- 表大小统计
SELECT
    table,
    formatReadableSize(sum(bytes)) as size,
    sum(rows) as rows,
    count() as parts
FROM system.parts
WHERE active
GROUP BY table
ORDER BY sum(bytes) DESC;

-- 后台合并监控
SELECT * FROM system.merges;
SELECT * FROM system.mutations;
```

#### 2. 使用 EXPLAIN 分析查询

```sql
-- 查看执行计划
EXPLAIN SELECT * FROM events WHERE date = today();

-- 查看管道
EXPLAIN PIPELINE SELECT * FROM events WHERE date = today();

-- 查看优化后的 SQL
EXPLAIN SYNTAX SELECT * FROM events WHERE 1=1 AND date = today();

-- 查看预估性能
EXPLAIN ESTIMATE SELECT * FROM events WHERE date >= today() - 7;
```

#### 3. 配置优化

```xml
<!-- 性能相关配置 -->
<clickhouse>
    <!-- 内存限制 -->
    <max_memory_usage>10000000000</max_memory_usage>
    <max_memory_usage_for_user>20000000000</max_memory_usage_for_user>
    <max_server_memory_usage>40000000000</max_server_memory_usage>
    
    <!-- 并发控制 -->
    <max_concurrent_queries>100</max_concurrent_queries>
    <max_threads>8</max_threads>
    
    <!-- 后台任务 -->
    <background_pool_size>16</background_pool_size>
    <background_merges_mutations_concurrency_ratio>2</background_merges_mutations_concurrency_ratio>
    
    <!-- 缓存 -->
    <mark_cache_size>5368709120</mark_cache_size>
    <uncompressed_cache_size>8589934592</uncompressed_cache_size>
</clickhouse>
```

## 具体案例

### 案例 1：实时用户行为分析系统

**场景**: 电商网站需要实时分析用户行为，支持秒级查询

**方案设计**

```sql
-- 1. 原始事件表
CREATE TABLE user_events
(
    event_time DateTime,
    event_date Date MATERIALIZED toDate(event_time),
    user_id UInt64,
    session_id String,
    event_type LowCardinality(String),
    page_url String,
    referrer String,
    device_type LowCardinality(String),
    properties String  -- JSON 属性
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (event_date, event_type, user_id, event_time);

-- 2. 实时统计物化视图
CREATE MATERIALIZED VIEW user_events_hourly_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (event_date, event_hour, event_type)
AS SELECT
    event_date,
    toHour(event_time) as event_hour,
    event_type,
    uniq(user_id) as unique_users,
    uniq(session_id) as unique_sessions,
    count() as event_count
FROM user_events
GROUP BY event_date, event_hour, event_type;

-- 3. 漏斗分析表
CREATE TABLE funnel_analysis
(
    date Date,
    funnel_name String,
    step UInt8,
    step_name String,
    user_count UInt64
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, funnel_name, step);

-- 4. 查询示例
-- 实时 PV/UV
SELECT
    event_type,
    sum(event_count) as pv,
    sum(unique_users) as uv
FROM user_events_hourly_mv
WHERE event_date = today()
GROUP BY event_type;

-- 用户路径分析
SELECT
    arrayStringConcat(groupArray(event_type), ' -> ') as user_path,
    count() as path_count
FROM (
    SELECT
        user_id,
        session_id,
        groupArray(event_type) as event_type
    FROM user_events
    WHERE event_date = today()
    GROUP BY user_id, session_id
)
GROUP BY user_path
ORDER BY path_count DESC
LIMIT 10;
```

**性能优化**
- 使用 LowCardinality 类型降低存储
- 按日期分区，便于查询和删除历史数据
- 物化视图预聚合，查询秒级返回
- ORDER BY 键根据查询模式优化

### 案例 2：日志分析系统

**场景**: 收集应用日志，支持全文搜索和统计分析

**方案设计**

```sql
-- 1. 日志表
CREATE TABLE application_logs
(
    timestamp DateTime64(3),
    log_date Date MATERIALIZED toDate(timestamp),
    application LowCardinality(String),
    level LowCardinality(String),
    logger String,
    message String,
    exception String,
    trace_id String,
    span_id String,
    host LowCardinality(String),
    INDEX message_idx message TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 1
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(log_date)
ORDER BY (log_date, application, level, timestamp)
TTL log_date + INTERVAL 30 DAY;  -- 自动删除 30 天前的日志

-- 2. 错误统计视图
CREATE MATERIALIZED VIEW error_stats_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(log_date)
ORDER BY (log_date, log_hour, application, level)
AS SELECT
    log_date,
    toHour(timestamp) as log_hour,
    application,
    level,
    count() as log_count
FROM application_logs
WHERE level IN ('ERROR', 'FATAL')
GROUP BY log_date, log_hour, application, level;

-- 3. 查询示例
-- 全文搜索
SELECT *
FROM application_logs
WHERE log_date = today()
  AND message LIKE '%exception%'
ORDER BY timestamp DESC
LIMIT 100;

-- 错误趋势
SELECT
    log_hour,
    application,
    sum(log_count) as errors
FROM error_stats_mv
WHERE log_date >= today() - 7
GROUP BY log_hour, application
ORDER BY log_hour, application;

-- TraceID 追踪
SELECT
    timestamp,
    application,
    level,
    message
FROM application_logs
WHERE trace_id = 'abc123'
ORDER BY timestamp;
```

**性能优化**
- 使用 tokenbf_v1 索引加速全文搜索
- TTL 自动删除历史数据
- LowCardinality 优化高频字段
- 错误日志单独统计，查询更快

### 案例 3：IoT 传感器数据存储

**场景**: 收集百万级传感器数据，支持时序查询和聚合

**方案设计**

```sql
-- 1. 原始数据表
CREATE TABLE sensor_data
(
    sensor_id UInt64,
    timestamp DateTime,
    metric_name LowCardinality(String),
    value Float64
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (sensor_id, metric_name, timestamp);

-- 2. 分钟级聚合
CREATE MATERIALIZED VIEW sensor_data_minute_mv
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(minute)
ORDER BY (sensor_id, metric_name, minute)
AS SELECT
    sensor_id,
    metric_name,
    toStartOfMinute(timestamp) as minute,
    avgState(value) as avg_value,
    minState(value) as min_value,
    maxState(value) as max_value,
    quantilesState(0.5, 0.9, 0.99)(value) as quantiles_value
FROM sensor_data
GROUP BY sensor_id, metric_name, minute;

-- 3. 小时级聚合
CREATE MATERIALIZED VIEW sensor_data_hour_mv
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (sensor_id, metric_name, hour)
AS SELECT
    sensor_id,
    metric_name,
    toStartOfHour(minute) as hour,
    avgMerge(avg_value) as avg_value,
    minMerge(min_value) as min_value,
    maxMerge(max_value) as max_value,
    quantilesMerge(quantiles_value) as quantiles_value
FROM sensor_data_minute_mv
GROUP BY sensor_id, metric_name, hour;

-- 4. 查询示例
-- 实时查询（分钟粒度）
SELECT
    minute,
    avgMerge(avg_value) as avg,
    minMerge(min_value) as min,
    maxMerge(max_value) as max,
    quantilesMerge(0.99)(quantiles_value) as p99
FROM sensor_data_minute_mv
WHERE sensor_id = 12345
  AND metric_name = 'temperature'
  AND minute >= now() - INTERVAL 1 HOUR
GROUP BY minute
ORDER BY minute;

-- 历史查询（小时粒度）
SELECT
    hour,
    avg(avg_value) as avg_temperature
FROM sensor_data_hour_mv
WHERE sensor_id = 12345
  AND metric_name = 'temperature'
  AND hour >= now() - INTERVAL 30 DAY
GROUP BY hour
ORDER BY hour;
```

**性能优化**
- 多级聚合：原始数据 → 分钟 → 小时 → 天
- AggregatingMergeTree 保存聚合状态
- 按天分区，历史数据可以快速删除
- 查询不同粒度的数据使用对应的物化视图

## 总结

通过以上实战经验和最佳实践，可以：

1. **正确使用 ClickHouse**：选择合适的表引擎、优化表结构
2. **提升查询性能**：使用物化视图、优化 SQL、合理使用索引
3. **高效写入数据**：批量写入、异步插入、去重策略
4. **分布式部署**：集群配置、副本同步、负载均衡
5. **监控与调优**：关键指标监控、慢查询分析、配置优化
6. **实际案例参考**：用户行为分析、日志分析、IoT 数据存储

ClickHouse 适合 OLAP 场景，通过合理的设计和优化，可以实现极高的查询性能和数据吞吐量。关键是理解其架构特点，根据业务场景选择合适的方案。

