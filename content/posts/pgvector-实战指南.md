---
title: "pgvector 实战指南"
date: 2025-10-04T20:42:31+08:00
draft: false
tags:
  - pgvector
  - 源码分析
categories:
  - PostgreSQL
  - 向量检索
  - 数据库
series: "pgvector-source-analysis"
description: "pgvector 源码剖析 - pgvector 实战指南"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# pgvector 实战指南

本文档提供 pgvector 在实际生产环境中的使用指南,包括常见场景、性能优化、故障排查和最佳实践。

## 目录

1. [快速入门](#1-快速入门)
2. [常见应用场景](#2-常见应用场景)
3. [性能调优实战](#3-性能调优实战)
4. [生产环境部署](#4-生产环境部署)
5. [故障排查](#5-故障排查)
6. [最佳实践清单](#6-最佳实践清单)

---

## 1. 快速入门

### 1.1 安装与初始化

```bash
# 编译安装(Linux/Mac)
cd /tmp
git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# 或使用 Docker
docker pull pgvector/pgvector:pg18-bookworm
```

```sql
-- 创建扩展
CREATE EXTENSION vector;

-- 验证安装
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

### 1.2 第一个向量表

```sql
-- 场景:存储文档 embedding(OpenAI ada-002, 1536 维)
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    title text NOT NULL,
    content text NOT NULL,
    embedding vector(1536),
    created_at timestamptz DEFAULT now()
);

-- 插入示例数据
INSERT INTO documents (title, content, embedding) VALUES
('PostgreSQL Tutorial', 'Learn PostgreSQL basics...',
 '[0.1, 0.2, ..., 0.9]'::vector(1536)),
('Vector Search Guide', 'How to use pgvector...',
 '[0.2, 0.3, ..., 0.8]'::vector(1536));

-- 相似度查询
SELECT
    id,
    title,
    embedding <=> '[0.15, 0.25, ..., 0.85]'::vector(1536) AS distance
FROM documents
ORDER BY distance
LIMIT 5;
```

### 1.3 创建索引(10万行以上推荐)

```sql
-- 选择 1:HNSW(查询性能优先)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- 选择 2:IVFFlat(构建速度优先,内存受限)
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 使用索引查询
SET hnsw.ef_search = 100;  -- 提高召回率
SELECT * FROM documents
ORDER BY embedding <=> '[...]'::vector(1536)
LIMIT 10;
```

---

## 2. 常见应用场景

### 2.1 语义搜索

#### 场景描述
在文档库中查找与用户查询语义相似的内容。

#### 表结构

```sql
CREATE TABLE articles (
    id bigserial PRIMARY KEY,
    title text NOT NULL,
    content text NOT NULL,
    author_id bigint,
    category text,
    embedding vector(1536),
    published_at timestamptz
);

-- 索引
CREATE INDEX ON articles USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX ON articles (category);
CREATE INDEX ON articles (author_id);
CREATE INDEX ON articles (published_at DESC);
```

#### 查询示例

```sql
-- 1. 简单相似度查询
SELECT id, title, embedding <=> $query_embedding AS similarity
FROM articles
ORDER BY embedding <=> $query_embedding
LIMIT 20;

-- 2. 带过滤条件的查询
SET hnsw.iterative_scan = strict_order;
SELECT id, title
FROM articles
WHERE category = 'technology'
  AND published_at > now() - interval '30 days'
ORDER BY embedding <=> $query_embedding
LIMIT 10;

-- 3. 混合排序(语义 + 时间)
WITH semantic_results AS (
    SELECT id, title, published_at,
           embedding <=> $query_embedding AS sim_score
    FROM articles
    ORDER BY sim_score
    LIMIT 100
)
SELECT id, title,
       -- 混合得分:70% 语义,30% 时间新鲜度
       (0.7 * sim_score + 0.3 * EXTRACT(epoch FROM now() - published_at) / 86400) AS final_score
FROM semantic_results
ORDER BY final_score
LIMIT 10;
```

### 2.2 推荐系统

#### 场景描述
基于用户历史行为和物品向量的协同过滤推荐。

#### 表结构

```sql
CREATE TABLE items (
    id bigserial PRIMARY KEY,
    name text NOT NULL,
    category text,
    price numeric(10, 2),
    embedding vector(128),
    popularity_score float
);

CREATE TABLE user_interactions (
    user_id bigint,
    item_id bigint,
    interaction_type text,  -- view, click, purchase
    weight float,
    created_at timestamptz DEFAULT now(),
    PRIMARY KEY (user_id, item_id, created_at)
);

-- 索引
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
CREATE INDEX ON user_interactions (user_id, created_at DESC);
```

#### 推荐逻辑

```sql
-- 1. 计算用户兴趣向量(加权平均)
CREATE OR REPLACE FUNCTION get_user_interest_vector(p_user_id bigint)
RETURNS vector AS $$
    SELECT
        -- 使用加权平均计算用户兴趣
        CASE
            WHEN COUNT(*) > 0 THEN
                (SELECT AVG(i.embedding * u.weight)
                 FROM user_interactions u
                 JOIN items i ON i.id = u.item_id
                 WHERE u.user_id = p_user_id
                   AND u.created_at > now() - interval '90 days')
            ELSE NULL
        END
    FROM user_interactions
    WHERE user_id = p_user_id;
$$ LANGUAGE sql;

-- 2. 生成推荐
WITH user_vector AS (
    SELECT get_user_interest_vector($user_id) AS vec
),
candidates AS (
    SELECT i.id, i.name, i.category, i.price,
           i.embedding <-> (SELECT vec FROM user_vector) AS distance,
           i.popularity_score
    FROM items i
    WHERE i.id NOT IN (
        -- 排除用户已交互的物品
        SELECT item_id FROM user_interactions
        WHERE user_id = $user_id
          AND created_at > now() - interval '30 days'
    )
    ORDER BY distance
    LIMIT 100
)
SELECT id, name, category, price,
       -- 综合得分:距离 + 热度
       (1 - distance / 10.0) * 0.7 + popularity_score * 0.3 AS score
FROM candidates
ORDER BY score DESC
LIMIT 20;
```

### 2.3 图像相似搜索

#### 场景描述
在图像库中查找视觉相似的图片。

#### 表结构

```sql
CREATE TABLE images (
    id bigserial PRIMARY KEY,
    filename text NOT NULL,
    url text NOT NULL,
    width int,
    height int,
    file_size bigint,
    embedding vector(512),  -- ResNet/CLIP 特征
    uploaded_at timestamptz DEFAULT now(),
    user_id bigint
);

-- 半精度索引(节省空间)
CREATE INDEX ON images USING hnsw
((embedding::halfvec(512)) halfvec_l2_ops);

-- 分区表(按上传时间)
CREATE TABLE images (
    ...
) PARTITION BY RANGE (uploaded_at);

CREATE TABLE images_2024_q1 PARTITION OF images
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
```

#### 查询示例

```sql
-- 1. 以图搜图
SELECT id, filename, url,
       embedding::halfvec(512) <-> $query_embedding::halfvec(512) AS distance
FROM images
ORDER BY embedding::halfvec(512) <-> $query_embedding::halfvec(512)
LIMIT 20;

-- 2. 去重(查找相似图片)
WITH duplicates AS (
    SELECT DISTINCT ON (a.id)
        a.id AS image_id,
        b.id AS duplicate_id,
        a.embedding <-> b.embedding AS distance
    FROM images a
    JOIN images b ON a.id < b.id  -- 避免重复比较
    WHERE a.embedding <-> b.embedding < 0.1  -- 阈值
)
SELECT image_id, array_agg(duplicate_id) AS duplicates
FROM duplicates
GROUP BY image_id;

-- 3. 聚类分析
-- 使用 K-means 对图片聚类
SELECT
    cluster_id,
    COUNT(*) AS image_count,
    AVG(embedding) AS cluster_center
FROM (
    SELECT
        id,
        embedding,
        -- 使用 K-means(需要扩展或外部工具)
        kmeans(embedding, 50) AS cluster_id
    FROM images
) clustered
GROUP BY cluster_id
ORDER BY image_count DESC;
```

### 2.4 问答系统(RAG)

#### 场景描述
检索增强生成(Retrieval-Augmented Generation),先检索相关文档,再生成答案。

#### 表结构

```sql
CREATE TABLE knowledge_base (
    id bigserial PRIMARY KEY,
    document_id bigint,
    chunk_text text NOT NULL,
    chunk_index int,
    metadata jsonb,
    embedding vector(1536),
    created_at timestamptz DEFAULT now()
);

CREATE INDEX ON knowledge_base USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON knowledge_base (document_id);
CREATE INDEX ON knowledge_base USING gin (metadata);
```

#### RAG 流程

```sql
-- Step 1:检索相关文档块
WITH relevant_chunks AS (
    SELECT
        id,
        document_id,
        chunk_text,
        metadata,
        1 - (embedding <=> $query_embedding) AS relevance_score
    FROM knowledge_base
    WHERE metadata @> '{"source": "official_docs"}'  -- 过滤数据源
    ORDER BY embedding <=> $query_embedding
    LIMIT 5
)
SELECT
    json_agg(
        json_build_object(
            'text', chunk_text,
            'score', relevance_score,
            'metadata', metadata
        ) ORDER BY relevance_score DESC
    ) AS context
FROM relevant_chunks;

-- Step 2:将 context 和 query 传给 LLM 生成答案(应用层)

-- Step 3:存储对话历史
CREATE TABLE conversations (
    id bigserial PRIMARY KEY,
    user_id bigint,
    question text,
    answer text,
    context_chunks bigint[],  -- 引用的文档块 ID
    created_at timestamptz DEFAULT now()
);
```

---

## 3. 性能调优实战

### 3.1 索引参数调优

#### HNSW 参数实验

```sql
-- 准备测试数据
CREATE TABLE test_vectors AS
SELECT
    id,
    ARRAY(SELECT random() FROM generate_series(1, 1536))::vector(1536) AS embedding
FROM generate_series(1, 100000) id;

-- 测试不同参数组合
-- 组合 1:默认参数(快速构建)
CREATE INDEX idx_default ON test_vectors USING hnsw (embedding vector_l2_ops);

-- 组合 2:高质量(构建慢,查询快)
CREATE INDEX idx_quality ON test_vectors USING hnsw (embedding vector_l2_ops)
WITH (m = 32, ef_construction = 200);

-- 测试查询性能
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM test_vectors
ORDER BY embedding <-> '[...]'::vector(1536)
LIMIT 10;

-- 比较召回率
WITH true_nn AS (
    -- 精确最近邻(顺序扫描)
    SELECT id FROM test_vectors
    ORDER BY embedding <-> $query
    LIMIT 10
),
approx_nn AS (
    -- 近似最近邻(索引扫描)
    SELECT id FROM test_vectors
    ORDER BY embedding <-> $query
    LIMIT 10
)
SELECT
    COUNT(*) FILTER (WHERE a.id = t.id) * 1.0 / 10 AS recall
FROM true_nn t
LEFT JOIN approx_nn a USING (id);
```

#### IVFFlat lists 参数选择

```sql
-- 根据数据规模计算 lists
CREATE OR REPLACE FUNCTION calculate_optimal_lists(table_name text)
RETURNS int AS $$
DECLARE
    row_count bigint;
    optimal_lists int;
BEGIN
    -- 获取行数
    EXECUTE format('SELECT COUNT(*) FROM %I', table_name) INTO row_count;
    
    -- 计算最优 lists
    IF row_count < 1000000 THEN
        optimal_lists := GREATEST(row_count / 1000, 10);
    ELSE
        optimal_lists := CEIL(SQRT(row_count));
    END IF;
    
    RETURN optimal_lists;
END;
$$ LANGUAGE plpgsql;

-- 使用
SELECT calculate_optimal_lists('documents');  -- 输出:100(假设 10万行)

-- 创建索引
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
```

### 3.2 批量加载优化

```sql
-- 场景:导入 100万个向量

-- 方法 1:COPY(最快,推荐)
-- 准备 CSV 文件:id,embedding
-- 格式:"1","[0.1,0.2,...,0.9]"

\timing on
COPY documents (id, embedding)
FROM '/path/to/vectors.csv'
WITH (FORMAT CSV, HEADER true);
-- 时间:约 30-60 秒(取决于维度和硬件)

-- 方法 2:批量 INSERT
BEGIN;
INSERT INTO documents (embedding) VALUES
('[...]'::vector(1536)),
('[...]'::vector(1536)),
...  -- 批量 1000-10000 行
;
COMMIT;

-- 方法 3:使用 unnest(最灵活)
INSERT INTO documents (embedding)
SELECT unnest($1::vector(1536)[]);
-- 应用层传递向量数组

-- 加载后创建索引(比先创建索引快 3-5 倍)
SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 7;
CREATE INDEX CONCURRENTLY ON documents USING hnsw (embedding vector_l2_ops);
```

### 3.3 查询性能优化

#### 慢查询诊断

```sql
-- 启用慢查询日志
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- 1秒
SELECT pg_reload_conf();

-- 查看慢查询
SELECT
    query,
    calls,
    total_exec_time / 1000 / 60 AS total_min,
    mean_exec_time AS avg_ms,
    stddev_exec_time,
    max_exec_time
FROM pg_stat_statements
WHERE query LIKE '%<->%' OR query LIKE '%<=>%'
ORDER BY total_exec_time DESC
LIMIT 20;
```

#### 优化策略

```sql
-- 问题 1:索引未使用
-- 检查
EXPLAIN SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 10;
-- 如果显示 Seq Scan,强制使用索引
BEGIN;
SET LOCAL enable_seqscan = off;
EXPLAIN SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 10;
COMMIT;

-- 问题 2:过滤条件导致结果不足
-- 使用迭代扫描
SET hnsw.iterative_scan = strict_order;
SET hnsw.max_scan_tuples = 50000;
SELECT * FROM items
WHERE category = 'rare_category'  -- 选择性高的条件
ORDER BY embedding <-> '[...]'
LIMIT 10;

-- 问题 3:返回结果过少
-- 增大 ef_search
SET hnsw.ef_search = 200;
SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 100;

-- 问题 4:查询超时
-- 设置语句超时
SET statement_timeout = '5s';
SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 10;
```

### 3.4 内存优化

```sql
-- 检查当前配置
SHOW shared_buffers;
SHOW work_mem;
SHOW maintenance_work_mem;

-- 推荐配置(16GB 内存服务器)
ALTER SYSTEM SET shared_buffers = '4GB';        -- 25% 内存
ALTER SYSTEM SET work_mem = '64MB';             -- 查询排序
ALTER SYSTEM SET maintenance_work_mem = '2GB';  -- 索引构建
ALTER SYSTEM SET effective_cache_size = '12GB'; -- 75% 内存
SELECT pg_reload_conf();

-- 监控缓冲区命中率
SELECT
    datname,
    blks_hit,
    blks_read,
    round(blks_hit::numeric / NULLIF(blks_hit + blks_read, 0) * 100, 2) AS cache_hit_ratio
FROM pg_stat_database
WHERE datname = current_database();
-- 目标:cache_hit_ratio > 99%
```

---

## 4. 生产环境部署

### 4.1 容量规划

```sql
-- 估算存储需求
CREATE OR REPLACE FUNCTION estimate_storage(
    num_vectors bigint,
    dimensions int,
    index_type text  -- 'hnsw' or 'ivfflat'
) RETURNS text AS $$
DECLARE
    vector_size bigint;
    index_size bigint;
    total_size bigint;
BEGIN
    -- 向量数据大小
    vector_size := num_vectors * (8 + dimensions * 4);
    
    -- 索引大小估算
    IF index_type = 'hnsw' THEN
        -- HNSW:约 1.5-2x 向量大小
        index_size := vector_size * 1.75;
    ELSIF index_type = 'ivfflat' THEN
        -- IVFFlat:约 1.1-1.3x 向量大小
        index_size := vector_size * 1.2;
    END IF;
    
    total_size := vector_size + index_size;
    
    RETURN format(
        'Vectors: %s, Index: %s, Total: %s',
        pg_size_pretty(vector_size),
        pg_size_pretty(index_size),
        pg_size_pretty(total_size)
    );
END;
$$ LANGUAGE plpgsql;

-- 使用示例
SELECT estimate_storage(10000000, 1536, 'hnsw');
-- 输出:Vectors: 58 GB, Index: 102 GB, Total: 160 GB
```

### 4.2 高可用配置

```sql
-- 主从复制配置

-- 主库:postgresql.conf
wal_level = replica
max_wal_senders = 10
wal_keep_size = 1GB

-- 从库:recovery.conf (PG 12+: postgresql.auto.conf)
primary_conninfo = 'host=primary_host port=5432 user=replicator password=xxx'
hot_standby = on

-- 验证复制状态
-- 主库
SELECT * FROM pg_stat_replication;

-- 从库
SELECT * FROM pg_stat_wal_receiver;

-- 只读查询(从库)
SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 10;
```

### 4.3 备份与恢复

```bash
# 逻辑备份(pg_dump)
pg_dump -h localhost -U postgres -d mydb \
    --format=custom --file=mydb_backup.dump

# 恢复
pg_restore -h localhost -U postgres -d mydb_new mydb_backup.dump

# 物理备份(pg_basebackup)
pg_basebackup -h localhost -U replicator \
    -D /var/lib/postgresql/backup -Fp -Xs -P

# PITR (时间点恢复)
# 1. 启用归档
archive_mode = on
archive_command = 'cp %p /archive/%f'

# 2. 基础备份 + WAL 归档
# 3. 恢复到指定时间
restore_command = 'cp /archive/%f %p'
recovery_target_time = '2024-01-04 10:00:00'
```

---

## 5. 故障排查

### 5.1 常见问题

#### 问题 1:索引构建失败

```
ERROR:  memory required is 8192 MB, maintenance_work_mem is 64 MB
```

**解决方案**:

```sql
-- 增大 maintenance_work_mem
SET maintenance_work_mem = '8GB';
CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 1000);
```

#### 问题 2:查询返回结果少于 LIMIT

```sql
-- 查询
SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 100;
-- 实际返回 30 行

-- 原因:ef_search 太小
-- 解决
SET hnsw.ef_search = 200;  -- 增大候选列表

-- 或启用迭代扫描
SET hnsw.iterative_scan = relaxed_order;
```

#### 问题 3:维度不匹配

```
ERROR:  expected 1536 dimensions, not 768
```

**解决方案**:

```sql
-- 检查表定义
\d+ documents;

-- 修正向量维度
UPDATE documents SET embedding = pad_or_truncate(embedding, 1536);

-- 或重新生成 embedding
```

#### 问题 4:NaN/Infinity 错误

```
ERROR:  NaN not allowed in vector
```

**解决方案**:

```sql
-- 检测异常值
SELECT id, embedding
FROM items
WHERE embedding::text ~ '(NaN|Infinity)';

-- 清理
DELETE FROM items WHERE embedding::text ~ '(NaN|Infinity)';

-- 预防:在应用层验证
```

### 5.2 性能问题诊断

```sql
-- 检查索引是否存在
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'items' AND indexdef LIKE '%hnsw%';

-- 检查索引使用情况
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'items';

-- 检查索引膨胀
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes
WHERE schemaname = 'public';

-- 检查锁等待
SELECT
    pid,
    usename,
    pg_blocking_pids(pid) AS blocked_by,
    query
FROM pg_stat_activity
WHERE waiting = true;
```

---

## 6. 最佳实践清单

### ✅ 数据建模

- [ ] 根据应用场景选择合适的向量类型(vector/halfvec/sparsevec)
- [ ] 使用类型修饰符指定固定维度:`vector(1536)`
- [ ] 为大表添加分区(按时间或类别)
- [ ] 使用 NOT NULL 约束防止空向量
- [ ] 添加合适的业务索引(B-tree、GIN)配合向量搜索

### ✅ 索引策略

- [ ] 10万行以上数据才创建向量索引
- [ ] 先加载数据,后创建索引
- [ ] 使用 CONCURRENTLY 避免阻塞写入
- [ ] HNSW 适合查询性能优先,IVFFlat 适合构建速度优先
- [ ] 定期 REINDEX 清理索引膨胀

### ✅ 查询优化

- [ ] 始终使用 `ORDER BY distance LIMIT n` 触发索引
- [ ] 根据召回率需求调整 ef_search/probes
- [ ] 过滤条件使用迭代扫描或部分索引
- [ ] 使用 EXPLAIN ANALYZE 验证查询计划
- [ ] 监控慢查询并优化

### ✅ 性能调优

- [ ] shared_buffers 设置为内存的 25%
- [ ] maintenance_work_mem 至少 2GB(构建索引时)
- [ ] work_mem 设置为 64MB-256MB
- [ ] 启用并行构建(max_parallel_maintenance_workers)
- [ ] 使用 COPY 批量加载数据

### ✅ 生产环境

- [ ] 配置主从复制实现高可用
- [ ] 定期备份(pg_dump/pg_basebackup)
- [ ] 启用 WAL 归档支持 PITR
- [ ] 监控磁盘空间和内存使用
- [ ] 设置合理的连接池大小

### ✅ 监控与告警

- [ ] 监控索引大小和增长趋势
- [ ] 监控查询 P95/P99 延迟
- [ ] 监控缓冲区命中率(目标 > 99%)
- [ ] 监控慢查询数量
- [ ] 设置磁盘空间告警(< 20% 剩余)

---

## 附录

### A. 性能基准测试

```sql
-- 生成测试数据
CREATE TABLE benchmark (
    id serial PRIMARY KEY,
    embedding vector(1536)
);

INSERT INTO benchmark (embedding)
SELECT ARRAY(SELECT random() FROM generate_series(1, 1536))::vector(1536)
FROM generate_series(1, 1000000);

-- 创建索引
\timing on
CREATE INDEX ON benchmark USING hnsw (embedding vector_l2_ops);
-- 记录构建时间

-- 测试查询性能
SELECT * FROM benchmark
ORDER BY embedding <-> (
    SELECT ARRAY(SELECT random() FROM generate_series(1, 1536))::vector(1536)
)
LIMIT 10;
-- 记录查询时间
```

### B. 迁移脚本模板

```sql
-- 从其他向量数据库迁移到 pgvector

-- Step 1:创建目标表
CREATE TABLE vectors_new (
    id bigserial PRIMARY KEY,
    metadata jsonb,
    embedding vector(1536)
);

-- Step 2:导出源数据为 CSV
-- (在源系统执行)

-- Step 3:导入到 PostgreSQL
\copy vectors_new (id, metadata, embedding) FROM '/path/to/vectors.csv' CSV HEADER;

-- Step 4:创建索引
CREATE INDEX CONCURRENTLY ON vectors_new USING hnsw (embedding vector_l2_ops);

-- Step 5:验证数据
SELECT COUNT(*) FROM vectors_new;
SELECT * FROM vectors_new LIMIT 5;
```

---

**文档版本**: 1.0  
**最后更新**: 2025-01-04  
**适用版本**: pgvector v0.8.1, PostgreSQL 13+
