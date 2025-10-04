---
title: "pgvector-02-HalfVec模块-概览"
date: 2025-10-04T20:42:31+08:00
draft: false
tags:
  - pgvector
  - 架构设计
  - 概览
  - 源码分析
categories:
  - PostgreSQL
  - 向量检索
  - 数据库
series: "pgvector-source-analysis"
description: "pgvector 源码剖析 - pgvector-02-HalfVec模块-概览"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# pgvector-02-HalfVec模块-概览

## 模块概述

HalfVec 模块实现了半精度浮点向量(half-precision vector)数据类型,使用 16 位浮点数(half/float16)存储向量元素。相比标准 Vector 类型(32 位浮点),HalfVec 可以节省 50% 的存储空间和内存,同时保持合理的精度,适用于内存受限和存储优化场景。

### 核心特性

- **存储优化**:每个元素 2 字节,相比 Vector 节省 50% 空间
- **精度权衡**:有效精度约 3-4 位十进制数字(相比单精度 6-7 位)
- **数值范围**:±65504(相比单精度 ±3.4e38)
- **向量维度**:最多 16,000 维(与 Vector 相同)

### 使用场景

**适用场景**:

- 嵌入维度较高(如 1536 维)且存储空间紧张
- 向量值范围在 ±65504 之间
- 可接受轻微精度损失
- 需要索引更高维度(HNSW 支持最多 4000 维 halfvec)

**不适用场景**:

- 需要高精度计算
- 向量值超出半精度范围
- 需要精确的数值稳定性

## 数据结构

### HalfVector 结构体

```c
typedef struct HalfVector
{
    int32 vl_len_;     // varlena 头部(总字节数)
    int16 dim;         // 维度
    int16 unused;      // 保留字段
    half x[FLEXIBLE_ARRAY_MEMBER];  // 半精度浮点数组
} HalfVector;
```

### 内存布局

| 字段 | 类型 | 偏移 | 大小 | 说明 |
|------|------|------|------|------|
| vl_len_ | int32 | 0 | 4 bytes | varlena 头部 |
| dim | int16 | 4 | 2 bytes | 向量维度 |
| unused | int16 | 6 | 2 bytes | 保留字段 |
| x | half[] | 8 | dim × 2 bytes | 半精度数组 |

### 总大小计算

```c
#define HALFVEC_SIZE(_dim) (offsetof(HalfVector, x) + sizeof(half) * (_dim))

// 示例:
// 128 维: 8 + 128×2 = 264 bytes
// 1536 维: 8 + 1536×2 = 3080 bytes
// 相比 Vector:
// 128 维 Vector: 8 + 128×4 = 520 bytes (HalfVec 节省 49%)
// 1536 维 Vector: 8 + 1536×4 = 6152 bytes (HalfVec 节省 50%)
```

## 半精度浮点数实现

### 两种实现方式

pgvector 根据平台支持选择半精度实现:

#### 1. F16C 指令集(x86-64 首选)

```c
#if defined(__F16C__)
#define F16C_SUPPORT
#endif

// 使用硬件指令转换
float HalfToFloat4(half value)
{
    // _cvtsh_ss:half → float(单条指令)
    return _cvtsh_ss(value);
}

half Float4ToHalf(float value)
{
    // _cvtss_sh:float → half(单条指令)
    return _cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT);
}
```

**优势**:

- 硬件加速,极快(1-2 CPU 周期)
- 遵循 IEEE 754 标准
- 正确处理舍入和特殊值

#### 2. _Float16 类型(编译器支持)

```c
#elif defined(__FLT16_MAX__)
#define FLT16_SUPPORT
#define half _Float16

// 编译器自动转换
float HalfToFloat4(half value)
{
    return (float) value;  // 编译器生成优化代码
}

half Float4ToHalf(float value)
{
    return (half) value;
}
```

**优势**:

- 跨平台支持(ARM、RISC-V)
- 编译器优化
- 类型安全

#### 3. 软件模拟(后备方案)

```c
#else
#define half uint16

// 手动位操作转换
float HalfToFloat4(half value)
{
    uint16 h = value;
    uint32 sign = (h & 0x8000) << 16;
    uint32 exponent = (h & 0x7C00) >> 10;
    uint32 mantissa = h & 0x03FF;
    
    if (exponent == 0)
    {
        // 零或非规格化数
        if (mantissa == 0)
            return *(float *)&sign;  // ±0
        
        // 非规格化数转换
        ...
    }
    else if (exponent == 31)
    {
        // 无穷大或 NaN
        uint32 result = sign | 0x7F800000 | (mantissa << 13);
        return *(float *)&result;
    }
    else
    {
        // 规格化数
        uint32 result = sign | ((exponent + 112) << 23) | (mantissa << 13);
        return *(float *)&result;
    }
}
```

### 数值特性

| 特性 | Half(FP16) | Float(FP32) |
|------|------------|-------------|
| **符号位** | 1 bit | 1 bit |
| **指数位** | 5 bits | 8 bits |
| **尾数位** | 10 bits | 23 bits |
| **最大值** | 65504 | 3.4e38 |
| **最小正值** | 6.1e-5 | 1.2e-38 |
| **精度** | ~3 位十进制 | ~7 位十进制 |
| **特殊值** | ±0, ±Inf, NaN | ±0, ±Inf, NaN |

## 主要 API 函数

### 1. I/O 函数

```c
// 文本输入:与 vector_in 类似,但转换为 half
Datum halfvec_in(PG_FUNCTION_ARGS);

// 文本输出:转换为 float 后格式化
Datum halfvec_out(PG_FUNCTION_ARGS);

// 二进制输入/输出
Datum halfvec_recv(PG_FUNCTION_ARGS);
Datum halfvec_send(PG_FUNCTION_ARGS);
```

### 2. 类型转换

```c
// HalfVec → Vector(提升精度)
Datum halfvec_to_vector(PG_FUNCTION_ARGS);

// Vector → HalfVec(降低精度)
Datum vector_to_halfvec(PG_FUNCTION_ARGS);

// Array → HalfVec
Datum array_to_halfvec(PG_FUNCTION_ARGS);

// HalfVec → Array
Datum halfvec_to_float4(PG_FUNCTION_ARGS);
```

### 3. 距离函数

HalfVec 支持与 Vector 相同的距离函数:

```c
// L2 距离
Datum halfvec_l2_distance(PG_FUNCTION_ARGS);

// 内积
Datum halfvec_inner_product(PG_FUNCTION_ARGS);

// 余弦距离
Datum halfvec_cosine_distance(PG_FUNCTION_ARGS);

// L1 距离
Datum halfvec_l1_distance(PG_FUNCTION_ARGS);
```

**实现策略**:先转换为 float 计算,确保精度

```c
Datum halfvec_l2_distance(PG_FUNCTION_ARGS)
{
    HalfVector *a = PG_GETARG_HALFVEC_P(0);
    HalfVector *b = PG_GETARG_HALFVEC_P(1);
    float distance = 0.0;
    
    CheckDims(a, b);
    
    // 转换为 float 计算,避免精度损失
    for (int i = 0; i < a->dim; i++)
    {
        float ax = HalfToFloat4(a->x[i]);
        float bx = HalfToFloat4(b->x[i]);
        float diff = ax - bx;
        distance += diff * diff;
    }
    
    PG_RETURN_FLOAT8(sqrt(distance));
}
```

### 4. 向量运算

```c
// 加减乘、拼接(v0.7.0+)
Datum halfvec_add(PG_FUNCTION_ARGS);
Datum halfvec_sub(PG_FUNCTION_ARGS);
Datum halfvec_mul(PG_FUNCTION_ARGS);
Datum halfvec_concat(PG_FUNCTION_ARGS);

// 归一化、范数
Datum halfvec_l2_normalize(PG_FUNCTION_ARGS);
Datum halfvec_l2_norm(PG_FUNCTION_ARGS);

// 子向量
Datum halfvec_subvector(PG_FUNCTION_ARGS);
```

## 使用示例

### 基本使用

```sql
-- 1. 创建表
CREATE TABLE embeddings (
    id bigserial PRIMARY KEY,
    content text,
    embedding halfvec(1536)  -- OpenAI ada-002 embedding
);

-- 2. 插入数据
INSERT INTO embeddings (content, embedding) VALUES
('PostgreSQL database', '[0.1, 0.2, ..., 0.9]'::halfvec(1536)),
('Vector search', '[0.2, 0.3, ..., 0.8]'::halfvec(1536));

-- 3. 查询
SELECT * FROM embeddings
ORDER BY embedding <-> '[0.15, 0.25, ..., 0.85]'::halfvec(1536)
LIMIT 5;
```

### 索引使用

```sql
-- HalfVec 支持 HNSW 和 IVFFlat 索引
-- 使用 halfvec_l2_ops 等操作符类

-- HNSW 索引(最多 4000 维)
CREATE INDEX ON embeddings USING hnsw (embedding halfvec_l2_ops);

-- IVFFlat 索引(最多 4000 维)
CREATE INDEX ON embeddings USING ivfflat (embedding halfvec_l2_ops)
WITH (lists = 100);

-- 查询自动使用索引
SELECT * FROM embeddings
ORDER BY embedding <=> '[...]'::halfvec(1536)
LIMIT 10;
```

### 半精度索引(Expression Index)

对 Vector 列创建半精度索引,节省索引空间:

```sql
-- 表定义使用 Vector
CREATE TABLE items (
    id bigserial PRIMARY KEY,
    embedding vector(1536)  -- 单精度存储
);

-- 索引使用半精度
CREATE INDEX ON items USING hnsw
((embedding::halfvec(1536)) halfvec_l2_ops);

-- 查询时转换为半精度
SELECT * FROM items
ORDER BY embedding::halfvec(1536) <-> '[...]'::halfvec(1536)
LIMIT 5;
```

**优势**:

- 索引大小减半
- 查询速度略快(缓存友好)
- 召回率略降(通常 < 1%)

### 类型转换

```sql
-- HalfVec ↔ Vector
SELECT '[1,2,3]'::halfvec(3)::vector(3);  -- 提升精度
SELECT '[1,2,3]'::vector(3)::halfvec(3);  -- 降低精度

-- HalfVec ↔ Array
SELECT '[1,2,3]'::halfvec(3)::float4[];
SELECT ARRAY[1.0, 2.0, 3.0]::halfvec(3);

-- 批量转换
UPDATE items SET embedding_half = embedding::halfvec(1536);
```

## 性能与精度权衡

### 存储空间对比

| 维度 | Vector 大小 | HalfVec 大小 | 节省 |
|------|-------------|--------------|------|
| 128 | 520 B | 264 B | 49% |
| 384 | 1544 B | 776 B | 50% |
| 768 | 3080 B | 1544 B | 50% |
| 1536 | 6152 B | 3080 B | 50% |

### 索引大小对比(100万向量)

| 维度 | 索引类型 | Vector | HalfVec | 节省 |
|------|----------|--------|---------|------|
| 1536 | HNSW | 12 GB | 6.5 GB | 46% |
| 1536 | IVFFlat | 8 GB | 4.5 GB | 44% |

### 精度影响

**距离计算误差**:

```sql
-- 测试半精度转换误差
WITH test AS (
    SELECT
        v1::vector(128) AS v_full,
        v1::halfvec(128)::vector(128) AS v_half
    FROM generate_series(1, 1000) AS v1
)
SELECT
    AVG(ABS(v_full <-> v_query - v_half <-> v_query)) AS avg_error,
    MAX(ABS(v_full <-> v_query - v_half <-> v_query)) AS max_error
FROM test;

-- 典型结果:
-- avg_error: 0.0001-0.001
-- max_error: 0.001-0.01
```

**召回率影响**(经验数据):

- HNSW 索引:召回率下降 < 1%
- IVFFlat 索引:召回率下降 1-2%

### 性能对比

| 操作 | Vector | HalfVec | 比较 |
|------|--------|---------|------|
| **距离计算** | 1.0x | 1.1x | HalfVec 略慢(转换开销) |
| **索引构建** | 1.0x | 0.8x | HalfVec 快 20%(I/O 少) |
| **查询 QPS** | 1.0x | 1.1x | HalfVec 略快(缓存友好) |
| **内存占用** | 1.0x | 0.5x | HalfVec 节省 50% |

## 最佳实践

### 1. 何时使用 HalfVec

**推荐使用**:

```sql
-- ✅ 高维嵌入(1536 维)
CREATE TABLE openai_embeddings (
    embedding halfvec(1536)  -- 节省 3KB/向量
);

-- ✅ 大规模数据集(百万级以上)
CREATE TABLE large_dataset (
    embedding halfvec(768)   -- 节省大量存储
);

-- ✅ 半精度索引
CREATE INDEX ON items USING hnsw
((embedding::halfvec(768)) halfvec_l2_ops);
```

**不推荐使用**:

```sql
-- ❌ 低维向量(存储节省不明显)
CREATE TABLE small_vectors (
    embedding halfvec(3)  -- 只节省 6 字节
);

-- ❌ 需要高精度
CREATE TABLE high_precision (
    embedding halfvec(128)  -- 科学计算不适用
);

-- ❌ 极端值
INSERT INTO items (embedding) VALUES
('[100000, -50000, ...]'::halfvec);  -- 超出半精度范围
```

### 2. 迁移策略

```sql
-- 从 Vector 迁移到 HalfVec

-- 方案 1:添加新列,逐步迁移
ALTER TABLE items ADD COLUMN embedding_half halfvec(1536);
UPDATE items SET embedding_half = embedding::halfvec(1536);

-- 创建新索引
CREATE INDEX CONCURRENTLY items_embedding_half_idx
ON items USING hnsw (embedding_half halfvec_l2_ops);

-- 验证查询结果一致性
-- 删除旧列和索引

-- 方案 2:直接转换(需要停机)
ALTER TABLE items ALTER COLUMN embedding TYPE halfvec(1536)
USING embedding::halfvec(1536);
```

### 3. 监控与调优

```sql
-- 检查存储空间
SELECT
    pg_size_pretty(pg_relation_size('items')) AS table_size,
    pg_size_pretty(pg_relation_size('items_embedding_idx')) AS index_size;

-- 比较查询性能
EXPLAIN ANALYZE
SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 10;

-- 比较半精度索引性能
EXPLAIN ANALYZE
SELECT * FROM items
ORDER BY embedding::halfvec(768) <-> '[...]'::halfvec(768)
LIMIT 10;
```

## 限制与注意事项

### 1. 数值范围限制

```sql
-- ⚠️ 超出范围的值会溢出
SELECT '[70000, -80000, 0.5]'::halfvec(3);  -- 错误:超出范围

-- ✅ 正确用法:归一化到合理范围
SELECT l2_normalize('[70000, -80000, 0.5]'::vector)::halfvec(3);
```

### 2. 精度损失

```sql
-- ⚠️ 小数精度有限
SELECT '[0.123456, 0.654321]'::halfvec(2)::vector(2);
-- 结果:[0.1235, 0.6543](仅保留 3-4 位有效数字)
```

### 3. 索引维度限制

- **HNSW**:最多 4000 维(相比 Vector 的 2000 维)
- **IVFFlat**:最多 4000 维

### 4. 不支持的操作

- 某些高级数学函数(如对数、指数)
- 需要扩展精度的科学计算

---

**文档版本**: 1.0  
**最后更新**: 2025-01-04  
**对应源文件**: src/halfvec.c, src/halfvec.h, src/halfutils.c
