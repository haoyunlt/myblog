---
title: "MySQL Server 源码剖析 - 框架使用示例、实战经验和最佳实践"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 最佳实践
  - 实战经验
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - MySQL Server 源码剖析 - 框架使用示例、实战经验和最佳实践"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# MySQL Server 源码剖析 - 框架使用示例、实战经验和最佳实践

## 一、框架使用示例

### 1.1 开发自定义存储引擎

MySQL 的插件式存储引擎架构允许开发者实现自定义存储引擎。以下是一个最小化示例。

#### 1.1.1 Handler 接口实现

```cpp
// 文件：storage/example/ha_example.h
#include "sql/handler.h"

class ha_example : public handler {
  THR_LOCK_DATA lock;      // 表锁
  EXAMPLE_SHARE *share;    // 共享数据
  
public:
  ha_example(handlerton *hton, TABLE_SHARE *table_arg);
  ~ha_example() override = default;
  
  // 必须实现的接口
  int open(const char *name, int mode, uint test_if_locked) override;
  int close(void) override;
  int rnd_init(bool scan) override;
  int rnd_next(uchar *buf) override;
  int rnd_pos(uchar *buf, uchar *pos) override;
  void position(const uchar *record) override;
  int info(uint flag) override;
  
  // 写操作接口
  int write_row(uchar *buf) override;
  int update_row(const uchar *old_data, uchar *new_data) override;
  int delete_row(const uchar *buf) override;
  
  // 索引操作接口
  int index_init(uint idx, bool sorted) override;
  int index_read_map(uchar *buf, const uchar *key,
                     key_part_map keypart_map,
                     enum ha_rkey_function find_flag) override;
  int index_next(uchar *buf) override;
  int index_end() override;
  
  // 元数据操作
  int create(const char *name, TABLE *form,
             HA_CREATE_INFO *create_info) override;
  int delete_table(const char *name) override;
  int rename_table(const char *from, const char *to) override;
  
  // 特性标志
  ulonglong table_flags() const override {
    return HA_BINLOG_ROW_CAPABLE |  // 支持基于行的复制
           HA_REC_NOT_IN_SEQ |       // 记录不按顺序存储
           HA_CAN_GEOMETRY;          // 支持空间数据
  }
  
  ulong index_flags(uint inx, uint part, bool all_parts) const override {
    return HA_READ_NEXT |            // 支持索引顺序读取
           HA_READ_RANGE;            // 支持范围查询
  }
};
```

**关键函数说明**：

**1. `open()` - 打开表**
```cpp
int ha_example::open(const char *name, int mode, uint test_if_locked) {
  // 1. 分配共享结构（多个连接共享同一表）
  if (!(share = get_share(name)))
    return 1; // 错误：无法获取共享数据
    
  // 2. 初始化表锁
  thr_lock_data_init(&share->lock, &lock, nullptr);
  
  // 3. 打开数据文件
  char data_file_name[FN_REFLEN];
  fn_format(data_file_name, name, "", ".exd", MY_REPLACE_EXT);
  
  if ((data_file = my_open(data_file_name, O_RDWR | O_CREAT, MYF(0))) == -1)
    return my_errno(); // 错误：无法打开文件
    
  return 0; // 成功
}
```

**2. `rnd_next()` - 全表扫描读取下一行**
```cpp
int ha_example::rnd_next(uchar *buf) {
  // 从当前位置读取一行数据
  int rc;
  
  // （此处省略：文件 I/O 操作读取记录）
  rc = read_record_from_file(buf);
  
  if (rc == 0) {
    // 成功读取，更新统计信息
    stats.records++;
    return 0;
  } else if (rc == HA_ERR_END_OF_FILE) {
    // 到达文件末尾
    return HA_ERR_END_OF_FILE;
  } else {
    // 读取错误
    return rc;
  }
}
```

**3. `write_row()` - 插入行**
```cpp
int ha_example::write_row(uchar *buf) {
  ha_statistic_increment(&SSV::ha_write_count);
  
  // 1. 检查唯一约束
  // （此处省略：唯一键冲突检查）
  
  // 2. 分配新行 ID
  my_off_t pos = allocate_row_position();
  
  // 3. 写入数据文件
  if (my_pwrite(data_file, buf, table->s->reclength, 
                pos, MYF(MY_NABP)))
    return errno_to_handler_error(errno);
    
  // 4. 更新索引
  // （此处省略：索引维护）
  
  return 0; // 成功
}
```

**4. `index_read_map()` - 索引查找**
```cpp
int ha_example::index_read_map(uchar *buf, const uchar *key,
                                key_part_map keypart_map,
                                enum ha_rkey_function find_flag) {
  // 1. 在索引中查找键
  // （此处省略：B+ 树查找逻辑）
  my_off_t pos = index_search(active_index, key, keypart_map, find_flag);
  
  if (pos == (my_off_t)-1)
    return HA_ERR_KEY_NOT_FOUND; // 未找到
    
  // 2. 根据位置读取记录
  if (my_pread(data_file, buf, table->s->reclength, pos, MYF(MY_NABP)))
    return errno_to_handler_error(errno);
    
  return 0; // 成功
}
```

#### 1.1.2 Handlerton 注册

```cpp
// 文件：storage/example/ha_example.cc

static handler *example_create_handler(handlerton *hton,
                                        TABLE_SHARE *table, 
                                        bool partitioned,
                                        MEM_ROOT *mem_root) {
  return new (mem_root) ha_example(hton, table);
}

static int example_init_func(void *p) {
  handlerton *example_hton = (handlerton *)p;
  
  // 设置存储引擎特性标志
  example_hton->state = SHOW_OPTION_YES;
  example_hton->db_type = DB_TYPE_EXAMPLE;
  example_hton->create = example_create_handler;
  
  // 事务接口（可选）
  example_hton->commit = nullptr;  // 不支持事务
  example_hton->rollback = nullptr;
  
  // DDL 接口
  example_hton->drop_database = nullptr;
  example_hton->panic = nullptr;
  
  return 0;
}

// 插件声明
mysql_declare_plugin(example) {
  MYSQL_STORAGE_ENGINE_PLUGIN,
  &example_storage_engine,
  "EXAMPLE",
  "MySQL AB",
  "Example storage engine",
  PLUGIN_LICENSE_GPL,
  example_init_func,     // 插件初始化
  nullptr,               // 插件检查卸载
  nullptr,               // 插件卸载
  0x0001,                // 版本 0.1
  nullptr,               // 状态变量
  nullptr,               // 系统变量
  nullptr,               // 保留字段
  0,                     // 标志
}
mysql_declare_plugin_end;
```

**使用示例**：
```sql
-- 创建使用自定义引擎的表
CREATE TABLE test_table (
  id INT PRIMARY KEY,
  name VARCHAR(50)
) ENGINE=EXAMPLE;

-- 插入数据
INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob');

-- 查询数据
SELECT * FROM test_table WHERE id = 1;
```

### 1.2 开发 UDF（用户自定义函数）

UDF 允许扩展 MySQL 的函数库。

#### 1.2.1 简单 UDF - 字符串反转

```cpp
// 文件：plugin/udf_examples/udf_reverse.cc

#include <mysql/plugin.h>
#include <mysql/service_mysql_alloc.h>
#include <cstring>

extern "C" {

// UDF 初始化
bool reverse_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
  // 1. 检查参数数量
  if (args->arg_count != 1) {
    strcpy(message, "reverse() requires exactly one argument");
    return true; // 初始化失败
  }
  
  // 2. 检查参数类型
  if (args->arg_type[0] != STRING_RESULT) {
    strcpy(message, "reverse() requires a string argument");
    return true;
  }
  
  // 3. 分配结果缓冲区
  initid->max_length = args->lengths[0];
  
  return false; // 初始化成功
}

// UDF 主函数
char *reverse_func(UDF_INIT *initid, UDF_ARGS *args,
                   char *result, unsigned long *length,
                   char *is_null, char *error) {
  // 1. 处理 NULL 输入
  if (!args->args[0]) {
    *is_null = 1;
    return nullptr;
  }
  
  // 2. 获取输入字符串
  const char *input = args->args[0];
  unsigned long input_len = args->lengths[0];
  
  // 3. 反转字符串
  char *output = (char *)malloc(input_len + 1);
  for (unsigned long i = 0; i < input_len; i++) {
    output[i] = input[input_len - 1 - i];
  }
  output[input_len] = '\0';
  
  // 4. 设置返回值
  *length = input_len;
  strcpy(result, output);
  free(output);
  
  return result;
}

// UDF 清理
void reverse_deinit(UDF_INIT *initid) {
  // 释放资源
  // （此处省略：清理分配的内存）
}

} // extern "C"
```

**注册 UDF**：
```sql
CREATE FUNCTION reverse RETURNS STRING SONAME 'udf_reverse.so';
```

**使用示例**：
```sql
SELECT reverse('hello');  -- 返回 'olleh'
SELECT reverse(name) FROM users;
```

#### 1.2.2 聚合 UDF - 自定义 SUM

```cpp
// 聚合 UDF 需要额外的 add/clear/remove 函数

extern "C" {

// 初始化
bool mysum_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
  if (args->arg_count != 1) {
    strcpy(message, "mysum() requires one argument");
    return true;
  }
  
  // 分配累加器内存
  initid->ptr = (char *)malloc(sizeof(double));
  *(double *)initid->ptr = 0.0;
  
  return false;
}

// 清空累加器
void mysum_clear(UDF_INIT *initid, char *is_null, char *error) {
  *(double *)initid->ptr = 0.0;
}

// 添加值到累加器
void mysum_add(UDF_INIT *initid, UDF_ARGS *args,
               char *is_null, char *error) {
  if (args->args[0]) {  // 非 NULL
    double value = *((double *)args->args[0]);
    *(double *)initid->ptr += value;
  }
}

// 返回最终结果
double mysum_func(UDF_INIT *initid, UDF_ARGS *args,
                  char *is_null, char *error) {
  return *(double *)initid->ptr;
}

// 清理
void mysum_deinit(UDF_INIT *initid) {
  free(initid->ptr);
}

} // extern "C"
```

**使用示例**：
```sql
SELECT mysum(salary) FROM employees GROUP BY department;
```

### 1.3 使用 Performance Schema 进行性能分析

#### 1.3.1 识别慢查询

```sql
-- 启用语句监控
UPDATE performance_schema.setup_instruments 
SET ENABLED = 'YES', TIMED = 'YES' 
WHERE NAME LIKE 'statement/%';

UPDATE performance_schema.setup_consumers 
SET ENABLED = 'YES' 
WHERE NAME LIKE 'events_statements%';

-- 查询最慢的 10 条 SQL（按总耗时）
SELECT 
  DIGEST_TEXT,
  COUNT_STAR AS exec_count,
  SUM_TIMER_WAIT/1000000000 AS total_ms,
  AVG_TIMER_WAIT/1000000000 AS avg_ms,
  MAX_TIMER_WAIT/1000000000 AS max_ms
FROM 
  performance_schema.events_statements_summary_by_digest
WHERE 
  SCHEMA_NAME NOT IN ('mysql', 'performance_schema', 'information_schema')
ORDER BY 
  SUM_TIMER_WAIT DESC
LIMIT 10;
```

**输出示例**：
```
+---------------------------------------------+------------+----------+---------+---------+
| DIGEST_TEXT                                 | exec_count | total_ms | avg_ms  | max_ms  |
+---------------------------------------------+------------+----------+---------+---------+
| SELECT * FROM `orders` WHERE `user_id` = ?  |      15234 | 45678.90 |    3.00 |   12.50 |
| UPDATE `users` SET `last_login` = ? ...     |       8965 | 23456.78 |    2.62 |    8.90 |
+---------------------------------------------+------------+----------+---------+---------+
```

#### 1.3.2 分析表访问模式

```sql
-- 查询读写最频繁的表
SELECT 
  OBJECT_SCHEMA,
  OBJECT_NAME,
  COUNT_READ,
  COUNT_WRITE,
  SUM_TIMER_READ/1000000000 AS read_ms,
  SUM_TIMER_WRITE/1000000000 AS write_ms
FROM 
  performance_schema.table_io_waits_summary_by_table
WHERE 
  OBJECT_SCHEMA NOT IN ('mysql', 'performance_schema')
ORDER BY 
  (SUM_TIMER_READ + SUM_TIMER_WRITE) DESC
LIMIT 10;
```

#### 1.3.3 诊断锁等待

```sql
-- 查询当前锁等待
SELECT 
  r.trx_id AS waiting_trx_id,
  r.trx_mysql_thread_id AS waiting_thread,
  r.trx_query AS waiting_query,
  b.trx_id AS blocking_trx_id,
  b.trx_mysql_thread_id AS blocking_thread,
  b.trx_query AS blocking_query
FROM 
  information_schema.innodb_lock_waits w
INNER JOIN information_schema.innodb_trx b 
  ON b.trx_id = w.blocking_trx_id
INNER JOIN information_schema.innodb_trx r 
  ON r.trx_id = w.requesting_trx_id;
```

## 二、实战经验

### 2.1 高并发场景优化

#### 2.1.1 连接池优化

**问题**：频繁创建/销毁连接导致性能下降

**解决方案**：
```ini
# my.cnf 配置
[mysqld]
# 使用线程池（MySQL Enterprise）
thread_handling = pool-of-threads
thread_pool_size = 16  # CPU 核心数
thread_pool_max_threads = 1000

# 或使用传统连接池 + 合理的最大连接数
max_connections = 500
max_user_connections = 400
thread_cache_size = 128  # 缓存线程，避免频繁创建
```

**应用层连接池配置**（以 HikariCP 为例）：
```java
HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:mysql://localhost:3306/mydb");
config.setMaximumPoolSize(50);        // 最大连接数
config.setMinimumIdle(10);            // 最小空闲连接
config.setConnectionTimeout(30000);   // 连接超时 30s
config.setIdleTimeout(600000);        // 空闲超时 10min
config.setMaxLifetime(1800000);       // 连接最大生命周期 30min

HikariDataSource ds = new HikariDataSource(config);
```

**最佳实践**：
- 应用层连接池大小 = 核心数 × 2 + 有效磁盘数
- MySQL `max_connections` = 所有应用实例连接池大小之和 × 1.2
- 使用 `wait_timeout` 自动清理僵尸连接

#### 2.1.2 热点行更新优化

**问题**：大量并发更新同一行导致锁等待

**场景**：商品库存扣减、账户余额更新

**方案 1：乐观锁**
```sql
-- 1. 读取当前版本
SELECT id, stock, version FROM products WHERE id = 123;

-- 2. 更新时检查版本
UPDATE products 
SET stock = stock - 1, version = version + 1 
WHERE id = 123 AND version = 10;  -- 原版本号

-- 3. 检查影响行数，如果为 0 则重试
```

**方案 2：队列化**
```
应用层维护内存队列 → 后台线程批量写入数据库
```

**方案 3：分库分表（极端场景）**
```
将单行拆分为多行（如库存拆分为 10 份），随机更新其中一份
```

#### 2.1.3 大事务拆分

**问题**：长事务持有锁时间过长，阻塞其他事务

**示例**：批量删除 1000万 条数据
```sql
-- ❌ 错误做法：一次性删除
DELETE FROM logs WHERE created_at < '2023-01-01';
-- 可能执行数小时，持有表锁/MDL 锁

-- ✅ 正确做法：分批删除
DELIMITER $$
CREATE PROCEDURE batch_delete()
BEGIN
  DECLARE rows INT DEFAULT 1;
  
  WHILE rows > 0 DO
    DELETE FROM logs 
    WHERE created_at < '2023-01-01' 
    LIMIT 10000;  -- 每批 1万
    
    SET rows = ROW_COUNT();
    
    -- 提交释放锁
    COMMIT;
    
    -- 短暂休眠，降低系统负载
    DO SLEEP(0.1);
  END WHILE;
END$$
DELIMITER ;

CALL batch_delete();
```

### 2.2 索引设计最佳实践

#### 2.2.1 联合索引顺序

**原则**：区分度高的列放在前面，等值查询列放在前面

**场景**：用户表查询
```sql
-- 查询 SQL
SELECT * FROM users 
WHERE status = 'active' AND city = 'Beijing' AND age > 25
ORDER BY created_at DESC
LIMIT 20;

-- ❌ 错误索引
INDEX idx_1(age, status, city)  -- age 范围查询，后续列无法使用

-- ✅ 正确索引
INDEX idx_2(status, city, age, created_at)
-- status 等值 → city 等值 → age 范围 → created_at 排序
```

**区分度计算**：
```sql
-- 计算列的选择性
SELECT 
  COUNT(DISTINCT status) / COUNT(*) AS status_sel,
  COUNT(DISTINCT city) / COUNT(*) AS city_sel,
  COUNT(DISTINCT age) / COUNT(*) AS age_sel
FROM users;

-- 选择性越高（接近 1），区分度越好
```

#### 2.2.2 覆盖索引

**原则**：索引包含所有查询列，避免回表

**示例**：
```sql
-- 原查询（需要回表）
SELECT id, name, age FROM users WHERE status = 'active';
INDEX idx_status(status)  -- 只包含 status

-- 优化后（覆盖索引）
INDEX idx_status_covering(status, id, name, age)
-- 或使用 InnoDB 二级索引自动包含主键的特性
INDEX idx_status(status)  -- 自动包含主键 id
SELECT id, status FROM users WHERE status = 'active';  -- 覆盖
```

**验证**：
```sql
EXPLAIN SELECT id, status FROM users WHERE status = 'active';
-- Extra: Using index  ← 表示覆盖索引
```

#### 2.2.3 前缀索引

**原则**：对于长字符串列，使用前缀索引节省空间

**示例**：
```sql
-- 分析前缀区分度
SELECT 
  COUNT(DISTINCT LEFT(email, 5)) / COUNT(*) AS prefix_5,
  COUNT(DISTINCT LEFT(email, 10)) / COUNT(*) AS prefix_10,
  COUNT(DISTINCT LEFT(email, 15)) / COUNT(*) AS prefix_15,
  COUNT(DISTINCT email) / COUNT(*) AS full
FROM users;

-- 选择合适的前缀长度（接近全列区分度）
CREATE INDEX idx_email_prefix ON users(email(10));
```

**注意**：前缀索引不支持 `ORDER BY` 和覆盖索引。

### 2.3 查询优化实战

#### 2.3.1 避免 SELECT *

**问题**：传输不必要的数据，无法利用覆盖索引

```sql
-- ❌ 错误
SELECT * FROM users WHERE id = 123;

-- ✅ 正确
SELECT id, name, email FROM users WHERE id = 123;
```

**例外**：如果确实需要所有列，使用 `SELECT *` 避免列变更时修改代码。

#### 2.3.2 子查询改写为 JOIN

**场景**：关联查询

```sql
-- ❌ 慢查询（相关子查询）
SELECT u.name, 
       (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) AS order_count
FROM users u;

-- ✅ 优化后（LEFT JOIN + GROUP BY）
SELECT u.name, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.name;
```

#### 2.3.3 LIMIT 深分页优化

**问题**：`LIMIT 1000000, 20` 需要扫描 1000020 行

```sql
-- ❌ 深分页慢查询
SELECT * FROM users ORDER BY id LIMIT 1000000, 20;

-- ✅ 优化 1：记录上次最大 ID
SELECT * FROM users WHERE id > 999980 ORDER BY id LIMIT 20;

-- ✅ 优化 2：延迟关联
SELECT u.* FROM users u
INNER JOIN (
  SELECT id FROM users ORDER BY id LIMIT 1000000, 20
) t ON u.id = t.id;
```

### 2.4 表结构设计最佳实践

#### 2.4.1 范式与反范式权衡

**第三范式（3NF）**：消除传递依赖
```sql
-- ❌ 不符合 3NF（冗余）
CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT,
  user_name VARCHAR(50),  -- 冗余：依赖于 user_id
  user_email VARCHAR(100) -- 冗余
);

-- ✅ 符合 3NF
CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  email VARCHAR(100)
);
```

**反范式优化（牺牲范式换取性能）**：
```sql
-- 在 orders 表冗余 user_name，避免 JOIN
CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT,
  user_name VARCHAR(50),  -- 冗余，但减少 JOIN
  amount DECIMAL(10, 2)
);

-- 适用场景：
-- 1. user_name 很少变更
-- 2. 查询频繁需要 user_name
-- 3. JOIN 成为性能瓶颈
```

#### 2.4.2 字段类型选择

| 场景 | ❌ 不推荐 | ✅ 推荐 | 原因 |
|------|---------|---------|------|
| 主键 | `VARCHAR(36)` (UUID) | `BIGINT` (雪花 ID) | INT 比较和存储更高效 |
| 状态 | `VARCHAR(20)` | `TINYINT` + 枚举映射 | 节省空间，索引更高效 |
| 金额 | `FLOAT` | `DECIMAL(10,2)` | 避免精度丢失 |
| 时间 | `VARCHAR(20)` | `DATETIME/TIMESTAMP` | 支持时间函数和索引 |
| IP地址 | `VARCHAR(15)` | `INT UNSIGNED` (INET_ATON/INET_NTOA) | 节省空间 |

#### 2.4.3 分区表设计

**适用场景**：
- 时间序列数据（日志、订单）
- 数据生命周期管理（自动删除历史数据）

**示例**：按月分区
```sql
CREATE TABLE orders (
  id BIGINT PRIMARY KEY,
  user_id BIGINT,
  amount DECIMAL(10, 2),
  created_at DATETIME
)
PARTITION BY RANGE (YEAR(created_at) * 100 + MONTH(created_at)) (
  PARTITION p202301 VALUES LESS THAN (202302),
  PARTITION p202302 VALUES LESS THAN (202303),
  PARTITION p202303 VALUES LESS THAN (202304),
  -- ...
  PARTITION p_future VALUES LESS THAN MAXVALUE
);
```

**维护**：
```sql
-- 删除旧分区（瞬间完成，无需扫描数据）
ALTER TABLE orders DROP PARTITION p202301;

-- 添加新分区
ALTER TABLE orders ADD PARTITION (
  PARTITION p202404 VALUES LESS THAN (202405)
);
```

### 2.5 复制与高可用

#### 2.5.1 主从复制配置

**主库配置** (`my.cnf`)：
```ini
[mysqld]
server-id = 1
log-bin = mysql-bin
binlog_format = ROW                # 推荐 ROW 格式
binlog_row_image = MINIMAL         # 只记录变更列
sync_binlog = 1                    # 每次事务刷盘（安全）
expire_logs_days = 7               # Binlog 保留 7 天

# GTID 复制（推荐）
gtid_mode = ON
enforce_gtid_consistency = ON
```

**从库配置** (`my.cnf`)：
```ini
[mysqld]
server-id = 2
relay-log = relay-bin
relay_log_recovery = ON            # 崩溃后自动恢复中继日志
read_only = ON                     # 从库只读
super_read_only = ON               # 限制 SUPER 权限用户写入
```

**建立复制**：
```sql
-- 主库：创建复制用户
CREATE USER 'repl'@'%' IDENTIFIED BY 'password';
GRANT REPLICATION SLAVE ON *.* TO 'repl'@'%';

-- 从库：配置主库信息（GTID 模式）
CHANGE REPLICATION SOURCE TO
  SOURCE_HOST='master_host',
  SOURCE_PORT=3306,
  SOURCE_USER='repl',
  SOURCE_PASSWORD='password',
  SOURCE_AUTO_POSITION=1;  -- 使用 GTID 自动定位

-- 启动复制
START REPLICA;

-- 检查状态
SHOW REPLICA STATUS\G
-- Replica_IO_Running: Yes
-- Replica_SQL_Running: Yes
-- Seconds_Behind_Source: 0  ← 复制延迟
```

#### 2.5.2 半同步复制

**提高数据安全性**：至少一个从库确认接收 Binlog 后主库才返回

```sql
-- 主库安装插件
INSTALL PLUGIN rpl_semi_sync_source SONAME 'semisync_source.so';
SET GLOBAL rpl_semi_sync_source_enabled = ON;
SET GLOBAL rpl_semi_sync_source_timeout = 1000;  -- 1秒超时

-- 从库安装插件
INSTALL PLUGIN rpl_semi_sync_replica SONAME 'semisync_replica.so';
SET GLOBAL rpl_semi_sync_replica_enabled = ON;
```

**监控**：
```sql
SHOW STATUS LIKE 'Rpl_semi_sync%';
-- Rpl_semi_sync_source_clients: 2  ← 半同步从库数量
-- Rpl_semi_sync_source_status: ON  ← 半同步状态
```

#### 2.5.3 故障切换

**手动切换**：
```sql
-- 1. 停止主库写入
SET GLOBAL read_only = ON;

-- 2. 等待从库追上主库
-- 在从库检查：
SHOW REPLICA STATUS\G
-- Seconds_Behind_Source: 0
-- Executed_Gtid_Set: <应与主库一致>

-- 3. 提升从库为新主库
STOP REPLICA;
RESET REPLICA ALL;
SET GLOBAL read_only = OFF;

-- 4. 其他从库指向新主库
CHANGE REPLICATION SOURCE TO SOURCE_HOST='new_master_host', ...;
START REPLICA;
```

**自动故障切换工具**：
- **MHA (Master High Availability)**：Perl 实现，成熟稳定
- **Orchestrator**：Go 实现，可视化界面
- **MySQL Router + Group Replication**：官方方案

## 三、常见陷阱与避坑指南

### 3.1 隐式类型转换

**问题**：索引失效

```sql
-- phone 列类型为 VARCHAR(20)，有索引

-- ❌ 错误：WHERE 子句中使用数字，发生隐式转换
SELECT * FROM users WHERE phone = 13800138000;
-- 等价于：WHERE CAST(phone AS UNSIGNED) = 13800138000
-- 结果：索引失效，全表扫描

-- ✅ 正确：使用字符串
SELECT * FROM users WHERE phone = '13800138000';
```

**验证**：
```sql
EXPLAIN SELECT * FROM users WHERE phone = 13800138000;
-- type: ALL  ← 全表扫描

EXPLAIN SELECT * FROM users WHERE phone = '13800138000';
-- type: ref  ← 使用索引
```

### 3.2 OR 条件索引失效

**问题**：OR 连接的列如果不都有索引，则所有索引失效

```sql
-- status 有索引，city 无索引

-- ❌ 索引失效
SELECT * FROM users WHERE status = 'active' OR city = 'Beijing';

-- ✅ 优化方案 1：UNION
SELECT * FROM users WHERE status = 'active'
UNION
SELECT * FROM users WHERE city = 'Beijing';

-- ✅ 优化方案 2：为 city 也建索引
CREATE INDEX idx_city ON users(city);
```

### 3.3 LIKE 前缀通配符

**问题**：`LIKE '%xxx'` 无法使用索引

```sql
-- ❌ 索引失效
SELECT * FROM users WHERE name LIKE '%张%';

-- ✅ 如果只需前缀匹配，去掉前导 %
SELECT * FROM users WHERE name LIKE '张%';

-- ✅ 如果必须全文搜索，使用全文索引
ALTER TABLE users ADD FULLTEXT INDEX idx_name_fulltext(name);
SELECT * FROM users WHERE MATCH(name) AGAINST('张' IN NATURAL LANGUAGE MODE);
```

### 3.4 函数操作列

**问题**：在列上使用函数导致索引失效

```sql
-- ❌ 索引失效
SELECT * FROM users WHERE YEAR(created_at) = 2023;

-- ✅ 改写为范围查询
SELECT * FROM users 
WHERE created_at >= '2023-01-01' AND created_at < '2024-01-01';
```

### 3.5 JOIN 字符集不一致

**问题**：JOIN 的列字符集不同导致索引失效

```sql
-- t1.name: utf8mb4_general_ci
-- t2.user_name: utf8mb4_unicode_ci

-- ❌ 字符集转换，索引失效
SELECT * FROM t1 JOIN t2 ON t1.name = t2.user_name;

-- ✅ 统一字符集
ALTER TABLE t2 MODIFY user_name VARCHAR(50) 
CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
```

## 四、性能监控与诊断

### 4.1 关键指标监控

```sql
-- 1. QPS (Queries Per Second)
SHOW GLOBAL STATUS LIKE 'Questions';
-- 计算：(当前值 - 上次值) / 时间间隔

-- 2. TPS (Transactions Per Second)
SHOW GLOBAL STATUS LIKE 'Com_commit';
SHOW GLOBAL STATUS LIKE 'Com_rollback';

-- 3. 连接数
SHOW GLOBAL STATUS LIKE 'Threads_connected';
SHOW GLOBAL STATUS LIKE 'Max_used_connections';

-- 4. InnoDB Buffer Pool 命中率
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read_requests';
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_reads';
-- 命中率 = (read_requests - reads) / read_requests
-- 目标：> 99%

-- 5. 慢查询
SHOW GLOBAL STATUS LIKE 'Slow_queries';

-- 6. 表锁等待
SHOW GLOBAL STATUS LIKE 'Table_locks_waited';

-- 7. 临时表
SHOW GLOBAL STATUS LIKE 'Created_tmp_disk_tables';
SHOW GLOBAL STATUS LIKE 'Created_tmp_tables';
-- 磁盘临时表比例应 < 25%
```

### 4.2 定位慢查询根因

**步骤**：

1. **开启慢查询日志**（已在配置文件设置）

2. **使用 pt-query-digest 分析**：
```bash
pt-query-digest /var/log/mysql/slow.log > slow_query_report.txt
```

3. **EXPLAIN 分析具体查询**：
```sql
EXPLAIN FORMAT=JSON 
SELECT * FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.status = 'pending';
```

4. **查看实际执行统计**（MySQL 8.0+）：
```sql
EXPLAIN ANALYZE
SELECT * FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.status = 'pending';
```

输出示例：
```
-> Nested loop inner join  (cost=125.5 rows=100) (actual time=0.15..2.34 rows=85 loops=1)
    -> Index lookup on o using idx_status (status='pending')  (cost=45.2 rows=100) (actual time=0.08..0.82 rows=85 loops=1)
    -> Single-row index lookup on u using PRIMARY (id=o.user_id)  (cost=0.25 rows=1) (actual time=0.015..0.016 rows=1 loops=85)
```

关键信息：
- `cost`：优化器估算成本
- `rows`：预计行数
- `actual time`：实际执行时间
- `loops`：执行次数

### 4.3 死锁诊断

```sql
-- 查看最近一次死锁
SHOW ENGINE INNODB STATUS\G

-- 输出中查找 LATEST DETECTED DEADLOCK 部分
```

**死锁日志示例**：
```
------------------------
LATEST DETECTED DEADLOCK
------------------------
2025-01-10 10:30:15

*** (1) TRANSACTION:
TRANSACTION 12345, ACTIVE 2 sec starting index read
mysql tables in use 1, locked 1
LOCK WAIT 3 lock struct(s), heap size 1136, 2 row lock(s)
MySQL thread id 10, OS thread handle 140123456, query id 5678 localhost root updating
UPDATE accounts SET balance = balance - 100 WHERE id = 1

*** (1) HOLDS THE LOCK(S):
RECORD LOCKS space id 0 page no 3 n bits 72 index PRIMARY of table `test`.`accounts` trx id 12345 lock_mode X locks rec but not gap
Record lock, heap no 2 PHYSICAL RECORD: n_fields 4; ...

*** (1) WAITING FOR THIS LOCK TO BE GRANTED:
RECORD LOCKS space id 0 page no 3 n bits 72 index PRIMARY of table `test`.`accounts` trx id 12345 lock_mode X locks rec but not gap waiting
Record lock, heap no 3 PHYSICAL RECORD: n_fields 4; ...

*** (2) TRANSACTION:
TRANSACTION 12346, ACTIVE 1 sec starting index read
mysql tables in use 1, locked 1
3 lock struct(s), heap size 1136, 2 row lock(s)
MySQL thread id 11, OS thread handle 140123457, query id 5679 localhost root updating
UPDATE accounts SET balance = balance + 100 WHERE id = 2

*** (2) HOLDS THE LOCK(S):
RECORD LOCKS space id 0 page no 3 n bits 72 index PRIMARY of table `test`.`accounts` trx id 12346 lock_mode X locks rec but not gap
Record lock, heap no 3 PHYSICAL RECORD: n_fields 4; ...

*** (2) WAITING FOR THIS LOCK TO BE GRANTED:
RECORD LOCKS space id 0 page no 3 n bits 72 index PRIMARY of table `test`.`accounts` trx id 12346 lock_mode X locks rec but not gap waiting
Record lock, heap no 2 PHYSICAL RECORD: n_fields 4; ...

*** WE ROLL BACK TRANSACTION (1)
```

**分析**：
- 事务 1 持有 id=2 的锁，等待 id=1 的锁
- 事务 2 持有 id=1 的锁，等待 id=2 的锁
- 形成循环等待，InnoDB 回滚事务 1

**避免死锁**：
1. 按相同顺序访问表和行
2. 缩短事务时间
3. 使用较低的隔离级别（如 READ COMMITTED）
4. 为表添加合理索引避免锁范围过大

## 五、安全最佳实践

### 5.1 权限最小化

```sql
-- ❌ 错误：授予过高权限
GRANT ALL PRIVILEGES ON *.* TO 'app_user'@'%';

-- ✅ 正确：只授予必需权限
GRANT SELECT, INSERT, UPDATE, DELETE ON app_db.* TO 'app_user'@'%';

-- ✅ 只读用户
GRANT SELECT ON app_db.* TO 'readonly_user'@'%';

-- ✅ 限制来源 IP
GRANT SELECT, INSERT, UPDATE, DELETE ON app_db.* TO 'app_user'@'10.0.1.%';
```

### 5.2 SQL 注入防御

**参数化查询**（以 Python 为例）：
```python
import mysql.connector

# ❌ 错误：字符串拼接（SQL 注入风险）
user_input = "admin' OR '1'='1"
query = f"SELECT * FROM users WHERE username = '{user_input}'"
cursor.execute(query)

# ✅ 正确：参数化查询
user_input = "admin' OR '1'='1"
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, (user_input,))
```

### 5.3 数据加密

**传输加密（SSL/TLS）**：
```ini
# my.cnf
[mysqld]
require_secure_transport = ON
ssl-ca = /path/to/ca.pem
ssl-cert = /path/to/server-cert.pem
ssl-key = /path/to/server-key.pem
```

**静态数据加密（InnoDB Tablespace Encryption）**：
```sql
-- 启用 Keyring 插件（存储加密密钥）
INSTALL PLUGIN keyring_file SONAME 'keyring_file.so';

-- 创建加密表
CREATE TABLE sensitive_data (
  id INT PRIMARY KEY,
  ssn VARCHAR(11),
  credit_card VARCHAR(16)
) ENCRYPTION='Y';

-- 加密现有表
ALTER TABLE sensitive_data ENCRYPTION='Y';
```

## 六、总结

本文档涵盖了 MySQL Server 的框架使用示例、实战经验和最佳实践，关键要点：

**开发扩展**：
- 自定义存储引擎：实现 Handler 接口
- UDF：扩展函数库
- 插件机制：灵活扩展功能

**性能优化**：
- 连接池：减少连接开销
- 索引设计：联合索引顺序、覆盖索引、前缀索引
- 查询优化：避免子查询、深分页优化
- 事务拆分：避免长事务

**高可用**：
- 主从复制：异步/半同步
- GTID：简化故障切换
- 自动故障切换工具

**监控诊断**：
- 关键指标：QPS、TPS、连接数、Buffer Pool 命中率
- 慢查询分析：pt-query-digest
- 死锁诊断：SHOW ENGINE INNODB STATUS

**安全**：
- 权限最小化
- 参数化查询防 SQL 注入
- SSL/TLS 传输加密
- Tablespace 静态数据加密

实践中应根据具体场景灵活应用这些原则和技巧。

