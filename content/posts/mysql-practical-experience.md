---
title: "MySQL源码实战经验与最佳实践：从理论到生产环境的完整指南"
date: 2024-05-11T21:00:00+08:00
draft: false
featured: true
series: "mysql-architecture"
tags: ["MySQL", "实战经验", "最佳实践", "性能优化", "故障排查", "运维经验"]
categories: ["mysql", "数据库系统"]
author: "tommie blog"
description: "MySQL数据库系统实战经验总结，涵盖性能优化、故障排查、运维管理等生产环境最佳实践"
image: "/images/articles/mysql-practical-experience.svg"
toc: true
tocOpen: true
showReadingTime: true
showWordCount: true
weight: 460
slug: "mysql-practical-experience"
---

## 概述

基于对MySQL源码的深入理解，本文总结了在生产环境中的实战经验和最佳实践。从性能优化到故障排查，从架构设计到运维管理，提供全方位的实用指导。

<!--more-->

## 1. 性能优化实战经验

### 1.1 连接层优化

#### 1.1.1 连接池配置优化

```sql
-- 连接相关参数优化
SET GLOBAL max_connections = 1000;              -- 最大连接数
SET GLOBAL max_connect_errors = 100000;         -- 最大连接错误数
SET GLOBAL connect_timeout = 10;                -- 连接超时时间
SET GLOBAL interactive_timeout = 28800;         -- 交互超时时间
SET GLOBAL wait_timeout = 28800;                -- 等待超时时间

-- 线程池配置（MySQL 8.0+）
SET GLOBAL thread_pool_size = 16;               -- 线程池大小
SET GLOBAL thread_pool_max_threads = 2000;      -- 最大线程数
SET GLOBAL thread_pool_stall_limit = 6;         -- 停滞限制
SET GLOBAL thread_pool_oversubscribe = 3;       -- 过度订阅
```

**实战经验**：
- 连接数设置为CPU核心数的8-12倍
- 在高并发场景下启用线程池
- 监控连接使用率，避免连接耗尽

#### 1.1.2 网络优化配置

```sql
-- 网络缓冲区优化
SET GLOBAL net_buffer_length = 32768;           -- 网络缓冲区长度
SET GLOBAL max_allowed_packet = 1073741824;     -- 最大数据包大小
SET GLOBAL net_read_timeout = 30;               -- 网络读超时
SET GLOBAL net_write_timeout = 60;              -- 网络写超时

-- TCP相关优化
SET GLOBAL back_log = 512;                      -- TCP监听队列大小
```

### 1.2 SQL层优化

#### 1.2.1 查询缓存优化

```sql
-- 查询缓存配置（MySQL 5.7及以下）
SET GLOBAL query_cache_type = ON;
SET GLOBAL query_cache_size = 268435456;        -- 256MB
SET GLOBAL query_cache_limit = 2097152;         -- 2MB
SET GLOBAL query_cache_min_res_unit = 4096;     -- 4KB

-- 监控查询缓存效果
SHOW STATUS LIKE 'Qcache%';
```

**实战经验**：
- MySQL 8.0已移除查询缓存，建议使用应用层缓存
- 查询缓存在写多读少场景下效果不佳
- 使用Redis等外部缓存替代查询缓存

#### 1.2.2 优化器配置

```sql
-- 优化器相关参数
SET GLOBAL optimizer_search_depth = 62;         -- 优化器搜索深度
SET GLOBAL optimizer_prune_level = 1;           -- 优化器剪枝级别
SET GLOBAL optimizer_switch = 'index_merge=on,index_merge_union=on,index_merge_sort_union=on,index_merge_intersection=on,engine_condition_pushdown=on,index_condition_pushdown=on,mrr=on,mrr_cost_based=on,block_nested_loop=on,batched_key_access=off,materialization=on,semijoin=on,loosescan=on,firstmatch=on,duplicateweedout=on,subquery_materialization_cost_based=on,use_index_extensions=on,condition_fanout_filter=on,derived_merge=on';

-- 成本模型配置（MySQL 5.7+）
SELECT * FROM mysql.server_cost;
SELECT * FROM mysql.engine_cost;
```

#### 1.2.3 排序和分组优化

```sql
-- 排序相关参数
SET GLOBAL sort_buffer_size = 2097152;          -- 2MB排序缓冲区
SET GLOBAL max_sort_length = 1024;              -- 最大排序长度
SET GLOBAL max_length_for_sort_data = 1024;     -- 排序数据最大长度

-- 分组相关参数
SET GLOBAL tmp_table_size = 134217728;          -- 128MB临时表大小
SET GLOBAL max_heap_table_size = 134217728;     -- 128MB堆表大小
```

### 1.3 存储引擎层优化

#### 1.3.1 InnoDB缓冲池优化

```sql
-- InnoDB缓冲池配置
SET GLOBAL innodb_buffer_pool_size = 8589934592;    -- 8GB（物理内存的70-80%）
SET GLOBAL innodb_buffer_pool_instances = 8;        -- 缓冲池实例数
SET GLOBAL innodb_buffer_pool_chunk_size = 134217728; -- 128MB块大小

-- 缓冲池预热
SET GLOBAL innodb_buffer_pool_dump_at_shutdown = ON;
SET GLOBAL innodb_buffer_pool_load_at_startup = ON;
SET GLOBAL innodb_buffer_pool_dump_pct = 25;        -- 转储25%的页面

-- 监控缓冲池状态
SHOW ENGINE INNODB STATUS\G
SELECT * FROM information_schema.INNODB_BUFFER_POOL_STATS;
```

**实战经验**：
- 缓冲池大小设置为物理内存的70-80%
- 实例数设置为CPU核心数，但不超过64
- 启用缓冲池转储和加载，加快重启后预热

#### 1.3.2 InnoDB日志优化

```sql
-- Redo日志配置
SET GLOBAL innodb_log_file_size = 1073741824;       -- 1GB日志文件大小
SET GLOBAL innodb_log_files_in_group = 2;           -- 日志文件数量
SET GLOBAL innodb_log_buffer_size = 67108864;       -- 64MB日志缓冲区
SET GLOBAL innodb_flush_log_at_trx_commit = 1;      -- 事务提交时刷新日志

-- 日志刷新优化
SET GLOBAL innodb_flush_method = 'O_DIRECT';        -- 直接I/O
SET GLOBAL innodb_io_capacity = 2000;               -- I/O容量
SET GLOBAL innodb_io_capacity_max = 4000;           -- 最大I/O容量
```

#### 1.3.3 InnoDB并发控制

```sql
-- 并发配置
SET GLOBAL innodb_thread_concurrency = 0;           -- 线程并发数（0表示无限制）
SET GLOBAL innodb_concurrency_tickets = 5000;       -- 并发票据数
SET GLOBAL innodb_commit_concurrency = 0;           -- 提交并发数

-- 锁等待配置
SET GLOBAL innodb_lock_wait_timeout = 50;           -- 锁等待超时时间
SET GLOBAL innodb_deadlock_detect = ON;             -- 死锁检测
SET GLOBAL innodb_print_all_deadlocks = ON;         -- 打印所有死锁
```

### 1.4 索引优化实战

#### 1.4.1 索引设计原则

```sql
-- 1. 复合索引设计示例
CREATE TABLE user_orders (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    order_date DATE NOT NULL,
    status TINYINT NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    
    -- 根据查询模式设计复合索引
    INDEX idx_user_date_status (user_id, order_date, status),
    INDEX idx_date_status (order_date, status),
    INDEX idx_status_amount (status, amount)
) ENGINE=InnoDB;

-- 2. 前缀索引示例
CREATE TABLE articles (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    
    -- 对长字段使用前缀索引
    INDEX idx_title_prefix (title(50))
) ENGINE=InnoDB;

-- 3. 函数索引示例（MySQL 8.0+）
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 函数索引支持复杂查询
    INDEX idx_email_domain ((SUBSTRING_INDEX(email, '@', -1))),
    INDEX idx_created_year ((YEAR(created_at)))
) ENGINE=InnoDB;
```

**实战经验**：
- 复合索引字段顺序：等值查询 > 范围查询 > 排序字段
- 前缀索引长度选择：保证选择性在95%以上
- 避免过多索引，影响写入性能

#### 1.4.2 索引监控和分析

```sql
-- 1. 索引使用情况分析
SELECT 
    t.TABLE_SCHEMA,
    t.TABLE_NAME,
    s.INDEX_NAME,
    s.COLUMN_NAME,
    s.SEQ_IN_INDEX,
    s.CARDINALITY,
    ROUND(((s.CARDINALITY / t.TABLE_ROWS) * 100), 2) AS selectivity
FROM 
    information_schema.STATISTICS s
    INNER JOIN information_schema.TABLES t 
        ON s.TABLE_SCHEMA = t.TABLE_SCHEMA 
        AND s.TABLE_NAME = t.TABLE_NAME
WHERE 
    t.TABLE_SCHEMA NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
    AND t.TABLE_ROWS > 0
ORDER BY 
    t.TABLE_SCHEMA, t.TABLE_NAME, s.INDEX_NAME, s.SEQ_IN_INDEX;

-- 2. 未使用的索引检测
SELECT 
    object_schema,
    object_name,
    index_name
FROM 
    performance_schema.table_io_waits_summary_by_index_usage 
WHERE 
    index_name IS NOT NULL
    AND count_star = 0
    AND object_schema NOT IN ('mysql', 'performance_schema', 'information_schema', 'sys')
ORDER BY 
    object_schema, object_name;

-- 3. 重复索引检测
SELECT 
    a.TABLE_SCHEMA,
    a.TABLE_NAME,
    a.INDEX_NAME as index1,
    b.INDEX_NAME as index2,
    a.COLUMN_NAME
FROM 
    information_schema.STATISTICS a
    JOIN information_schema.STATISTICS b 
        ON a.TABLE_SCHEMA = b.TABLE_SCHEMA
        AND a.TABLE_NAME = b.TABLE_NAME
        AND a.SEQ_IN_INDEX = b.SEQ_IN_INDEX
        AND a.COLUMN_NAME = b.COLUMN_NAME
        AND a.INDEX_NAME != b.INDEX_NAME
WHERE 
    a.TABLE_SCHEMA NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
ORDER BY 
    a.TABLE_SCHEMA, a.TABLE_NAME, a.INDEX_NAME;
```

## 2. 故障排查实战

### 2.1 性能问题诊断

#### 2.1.1 慢查询分析

```sql
-- 1. 启用慢查询日志
SET GLOBAL slow_query_log = ON;
SET GLOBAL slow_query_log_file = '/var/log/mysql/slow.log';
SET GLOBAL long_query_time = 1.0;               -- 1秒以上的查询
SET GLOBAL log_queries_not_using_indexes = ON;  -- 记录未使用索引的查询
SET GLOBAL min_examined_row_limit = 1000;       -- 最小检查行数

-- 2. 分析慢查询
-- 使用mysqldumpslow工具
-- mysqldumpslow -s t -t 10 /var/log/mysql/slow.log

-- 3. Performance Schema分析
SELECT 
    DIGEST_TEXT,
    COUNT_STAR,
    AVG_TIMER_WAIT/1000000000 AS avg_time_sec,
    MAX_TIMER_WAIT/1000000000 AS max_time_sec,
    SUM_ROWS_EXAMINED,
    SUM_ROWS_SENT,
    SUM_CREATED_TMP_TABLES,
    SUM_CREATED_TMP_DISK_TABLES
FROM 
    performance_schema.events_statements_summary_by_digest 
ORDER BY 
    AVG_TIMER_WAIT DESC 
LIMIT 10;
```

#### 2.1.2 锁等待分析

```sql
-- 1. 当前锁等待情况
SELECT 
    r.trx_id waiting_trx_id,
    r.trx_mysql_thread_id waiting_thread,
    r.trx_query waiting_query,
    b.trx_id blocking_trx_id,
    b.trx_mysql_thread_id blocking_thread,
    b.trx_query blocking_query,
    bl.lock_mode,
    bl.lock_type,
    bl.lock_table,
    bl.lock_index
FROM 
    information_schema.innodb_lock_waits w
    INNER JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
    INNER JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id
    INNER JOIN information_schema.innodb_locks bl ON bl.lock_id = w.blocking_lock_id;

-- 2. 死锁信息查看
SHOW ENGINE INNODB STATUS\G

-- 3. 元数据锁等待（MySQL 5.7+）
SELECT 
    object_type,
    object_schema,
    object_name,
    lock_type,
    lock_duration,
    lock_status,
    source
FROM 
    performance_schema.metadata_locks
WHERE 
    lock_status = 'PENDING';
```

#### 2.1.3 连接问题诊断

```sql
-- 1. 连接状态统计
SELECT 
    command,
    COUNT(*) as count,
    AVG(time) as avg_time
FROM 
    information_schema.processlist 
GROUP BY 
    command
ORDER BY 
    count DESC;

-- 2. 长时间运行的查询
SELECT 
    id,
    user,
    host,
    db,
    command,
    time,
    state,
    LEFT(info, 100) as query_snippet
FROM 
    information_schema.processlist 
WHERE 
    command != 'Sleep' 
    AND time > 60
ORDER BY 
    time DESC;

-- 3. 连接错误统计
SHOW STATUS LIKE 'Connection_errors%';
SHOW STATUS LIKE 'Aborted%';
```

### 2.2 存储引擎问题诊断

#### 2.2.1 InnoDB状态监控

```sql
-- 1. InnoDB引擎状态
SHOW ENGINE INNODB STATUS\G

-- 2. 缓冲池状态
SELECT 
    pool_id,
    pool_size,
    free_buffers,
    database_pages,
    old_database_pages,
    modified_database_pages,
    pending_decompress,
    pending_reads,
    pending_flush_lru,
    pending_flush_list
FROM 
    information_schema.innodb_buffer_pool_stats;

-- 3. 事务状态
SELECT 
    trx_id,
    trx_state,
    trx_started,
    trx_requested_lock_id,
    trx_wait_started,
    trx_weight,
    trx_mysql_thread_id,
    trx_query
FROM 
    information_schema.innodb_trx
ORDER BY 
    trx_started;
```

#### 2.2.2 表空间和数据文件监控

```sql
-- 1. 表空间使用情况
SELECT 
    tablespace_name,
    file_name,
    file_type,
    file_size/1024/1024 as size_mb,
    allocated_size/1024/1024 as allocated_mb,
    autoextend_size/1024/1024 as autoextend_mb,
    data_free/1024/1024 as free_mb
FROM 
    information_schema.files
WHERE 
    file_type = 'TABLESPACE';

-- 2. 表大小统计
SELECT 
    table_schema,
    table_name,
    ROUND(((data_length + index_length) / 1024 / 1024), 2) as size_mb,
    ROUND((data_free / 1024 / 1024), 2) as free_mb,
    table_rows,
    avg_row_length,
    auto_increment
FROM 
    information_schema.tables
WHERE 
    table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
ORDER BY 
    (data_length + index_length) DESC
LIMIT 20;
```

## 3. 架构设计最佳实践

### 3.1 读写分离架构

#### 3.1.1 主从复制配置

```sql
-- 主库配置
[mysqld]
server-id = 1
log-bin = mysql-bin
binlog-format = ROW
gtid-mode = ON
enforce-gtid-consistency = ON
log-slave-updates = ON
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1

-- 从库配置
[mysqld]
server-id = 2
relay-log = relay-bin
read-only = ON
super-read-only = ON
gtid-mode = ON
enforce-gtid-consistency = ON
log-slave-updates = ON
slave-parallel-type = LOGICAL_CLOCK
slave-parallel-workers = 8
```

#### 3.1.2 读写分离实现

```python
# Python读写分离示例
import pymysql
from typing import Dict, Any

class MySQLCluster:
    def __init__(self, master_config: Dict[str, Any], slave_configs: list):
        self.master_config = master_config
        self.slave_configs = slave_configs
        self.master_conn = None
        self.slave_conns = []
        self.current_slave_index = 0
        
    def get_master_connection(self):
        """获取主库连接"""
        if not self.master_conn or not self.master_conn.open:
            self.master_conn = pymysql.connect(**self.master_config)
        return self.master_conn
    
    def get_slave_connection(self):
        """获取从库连接（轮询）"""
        if not self.slave_conns:
            for config in self.slave_configs:
                conn = pymysql.connect(**config)
                self.slave_conns.append(conn)
        
        # 简单轮询策略
        conn = self.slave_conns[self.current_slave_index]
        self.current_slave_index = (self.current_slave_index + 1) % len(self.slave_conns)
        
        if not conn.open:
            conn = pymysql.connect(**self.slave_configs[self.current_slave_index])
            self.slave_conns[self.current_slave_index] = conn
            
        return conn
    
    def execute_write(self, sql: str, params=None):
        """执行写操作"""
        conn = self.get_master_connection()
        with conn.cursor() as cursor:
            return cursor.execute(sql, params)
    
    def execute_read(self, sql: str, params=None):
        """执行读操作"""
        conn = self.get_slave_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
```

### 3.2 分库分表策略

#### 3.2.1 水平分表实现

```sql
-- 按时间分表示例
CREATE TABLE user_orders_202401 (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    order_date DATE NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    status TINYINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_user_id (user_id),
    INDEX idx_order_date (order_date),
    INDEX idx_status (status)
) ENGINE=InnoDB;

-- 按哈希分表示例
CREATE TABLE user_profiles_0 (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    nickname VARCHAR(50) NOT NULL,
    avatar VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE KEY uk_user_id (user_id)
) ENGINE=InnoDB;

-- 分表路由函数
DELIMITER //
CREATE FUNCTION get_table_suffix(user_id BIGINT) 
RETURNS VARCHAR(10)
READS SQL DATA
DETERMINISTIC
BEGIN
    RETURN CONCAT('_', user_id % 16);
END //
DELIMITER ;
```

#### 3.2.2 分库分表中间件配置

```yaml
# ShardingSphere配置示例
dataSources:
  ds_0:
    url: jdbc:mysql://192.168.1.101:3306/order_db_0
    username: root
    password: password
  ds_1:
    url: jdbc:mysql://192.168.1.102:3306/order_db_1
    username: root
    password: password

shardingRule:
  tables:
    user_orders:
      actualDataNodes: ds_${0..1}.user_orders_${0..15}
      databaseStrategy:
        inline:
          shardingColumn: user_id
          algorithmExpression: ds_${user_id % 2}
      tableStrategy:
        inline:
          shardingColumn: user_id
          algorithmExpression: user_orders_${user_id % 16}
      keyGenerator:
        type: SNOWFLAKE
        column: id
```

### 3.3 高可用架构

#### 3.3.1 MySQL Group Replication

```sql
-- Group Replication配置
[mysqld]
server-id = 1
gtid-mode = ON
enforce-gtid-consistency = ON
binlog-checksum = NONE
log-bin = binlog
log-slave-updates = ON
binlog-format = ROW
master-info-repository = TABLE
relay-log-info-repository = TABLE
transaction-write-set-extraction = XXHASH64

# Group Replication配置
loose-group_replication_group_name = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
loose-group_replication_start_on_boot = OFF
loose-group_replication_local_address = "192.168.1.101:33061"
loose-group_replication_group_seeds = "192.168.1.101:33061,192.168.1.102:33061,192.168.1.103:33061"
loose-group_replication_bootstrap_group = OFF
loose-group_replication_single_primary_mode = ON
```

#### 3.3.2 故障切换脚本

```bash
#!/bin/bash
# MySQL故障切换脚本

MASTER_HOST="192.168.1.101"
SLAVE_HOST="192.168.1.102"
VIP="192.168.1.100"
MYSQL_USER="root"
MYSQL_PASS="password"

# 检查主库状态
check_master() {
    mysql -h$MASTER_HOST -u$MYSQL_USER -p$MYSQL_PASS -e "SELECT 1" >/dev/null 2>&1
    return $?
}

# 提升从库为主库
promote_slave() {
    echo "Promoting slave to master..."
    mysql -h$SLAVE_HOST -u$MYSQL_USER -p$MYSQL_PASS -e "
        STOP SLAVE;
        RESET SLAVE ALL;
        SET GLOBAL read_only = OFF;
        SET GLOBAL super_read_only = OFF;
    "
    
    # 切换VIP
    ssh root@$SLAVE_HOST "ip addr add $VIP/24 dev eth0"
    ssh root@$MASTER_HOST "ip addr del $VIP/24 dev eth0" 2>/dev/null
    
    echo "Failover completed"
}

# 主循环
while true; do
    if ! check_master; then
        echo "Master is down, starting failover..."
        promote_slave
        break
    fi
    sleep 5
done
```

## 4. 运维管理最佳实践

### 4.1 备份策略

#### 4.1.1 物理备份脚本

```bash
#!/bin/bash
# MySQL物理备份脚本

BACKUP_DIR="/backup/mysql"
MYSQL_USER="backup"
MYSQL_PASS="backup_password"
RETENTION_DAYS=7

# 创建备份目录
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$BACKUP_DATE"
mkdir -p $BACKUP_PATH

# 执行备份
echo "Starting backup at $(date)"
xtrabackup --backup \
    --user=$MYSQL_USER \
    --password=$MYSQL_PASS \
    --target-dir=$BACKUP_PATH \
    --compress \
    --compress-threads=4 \
    --parallel=4

if [ $? -eq 0 ]; then
    echo "Backup completed successfully"
    
    # 清理过期备份
    find $BACKUP_DIR -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;
    
    # 发送通知
    echo "MySQL backup completed at $(date)" | mail -s "MySQL Backup Success" admin@company.com
else
    echo "Backup failed"
    echo "MySQL backup failed at $(date)" | mail -s "MySQL Backup Failed" admin@company.com
    exit 1
fi
```

#### 4.1.2 逻辑备份脚本

```bash
#!/bin/bash
# MySQL逻辑备份脚本

BACKUP_DIR="/backup/mysql/logical"
MYSQL_USER="backup"
MYSQL_PASS="backup_password"
DATABASES="app_db user_db order_db"

# 创建备份目录
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$BACKUP_DATE"
mkdir -p $BACKUP_PATH

# 备份每个数据库
for db in $DATABASES; do
    echo "Backing up database: $db"
    mysqldump \
        --user=$MYSQL_USER \
        --password=$MYSQL_PASS \
        --single-transaction \
        --routines \
        --triggers \
        --events \
        --hex-blob \
        --master-data=2 \
        --flush-logs \
        $db | gzip > $BACKUP_PATH/${db}_${BACKUP_DATE}.sql.gz
    
    if [ $? -ne 0 ]; then
        echo "Failed to backup database: $db"
        exit 1
    fi
done

echo "All databases backed up successfully"
```

### 4.2 监控告警

#### 4.2.1 Prometheus监控配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mysql'
    static_configs:
      - targets: ['192.168.1.101:9104', '192.168.1.102:9104']
    scrape_interval: 5s
    metrics_path: /metrics

rule_files:
  - "mysql_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

```yaml
# mysql_rules.yml
groups:
- name: mysql
  rules:
  - alert: MySQLDown
    expr: mysql_up == 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "MySQL instance is down"
      description: "MySQL instance {{ $labels.instance }} has been down for more than 0 minutes."

  - alert: MySQLSlowQueries
    expr: rate(mysql_global_status_slow_queries[5m]) > 0.2
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "MySQL slow queries"
      description: "MySQL instance {{ $labels.instance }} has slow query rate of {{ $value }} per second."

  - alert: MySQLConnectionsHigh
    expr: mysql_global_status_threads_connected / mysql_global_variables_max_connections * 100 > 80
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "MySQL connection usage high"
      description: "MySQL instance {{ $labels.instance }} connection usage is {{ $value }}%."

  - alert: MySQLInnoDBLogWaits
    expr: rate(mysql_global_status_innodb_log_waits[5m]) > 10
    for: 0m
    labels:
      severity: warning
    annotations:
      summary: "MySQL InnoDB log waits"
      description: "MySQL instance {{ $labels.instance }} has InnoDB log waits rate of {{ $value }} per second."
```

#### 4.2.2 自定义监控脚本

```python
#!/usr/bin/env python3
# MySQL健康检查脚本

import pymysql
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class MySQLMonitor:
    def __init__(self, config):
        self.config = config
        self.connection = None
        
    def connect(self):
        """连接MySQL"""
        try:
            self.connection = pymysql.connect(**self.config['mysql'])
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def check_replication(self):
        """检查主从复制状态"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SHOW SLAVE STATUS")
                result = cursor.fetchone()
                
                if not result:
                    return True, "Not a slave server"
                
                io_running = result[10]  # Slave_IO_Running
                sql_running = result[11]  # Slave_SQL_Running
                seconds_behind = result[32]  # Seconds_Behind_Master
                
                if io_running != 'Yes' or sql_running != 'Yes':
                    return False, f"Replication stopped: IO={io_running}, SQL={sql_running}"
                
                if seconds_behind and seconds_behind > 300:  # 5分钟延迟
                    return False, f"Replication lag: {seconds_behind} seconds"
                
                return True, "Replication OK"
                
        except Exception as e:
            return False, f"Check replication failed: {e}"
    
    def check_connections(self):
        """检查连接数"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
                connected = int(cursor.fetchone()[1])
                
                cursor.execute("SHOW VARIABLES LIKE 'max_connections'")
                max_connections = int(cursor.fetchone()[1])
                
                usage_pct = (connected / max_connections) * 100
                
                if usage_pct > 80:
                    return False, f"High connection usage: {usage_pct:.1f}%"
                
                return True, f"Connection usage: {usage_pct:.1f}%"
                
        except Exception as e:
            return False, f"Check connections failed: {e}"
    
    def check_slow_queries(self):
        """检查慢查询"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SHOW STATUS LIKE 'Slow_queries'")
                slow_queries = int(cursor.fetchone()[1])
                
                cursor.execute("SHOW STATUS LIKE 'Queries'")
                total_queries = int(cursor.fetchone()[1])
                
                if total_queries > 0:
                    slow_pct = (slow_queries / total_queries) * 100
                    if slow_pct > 5:  # 5%慢查询率
                        return False, f"High slow query rate: {slow_pct:.2f}%"
                
                return True, f"Slow query rate: {slow_pct:.2f}%"
                
        except Exception as e:
            return False, f"Check slow queries failed: {e}"
    
    def send_alert(self, subject, message):
        """发送告警邮件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from']
            msg['To'] = ', '.join(self.config['email']['to'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.config['email']['smtp_server'])
            server.starttls()
            server.login(self.config['email']['username'], self.config['email']['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"Alert sent: {subject}")
            
        except Exception as e:
            print(f"Failed to send alert: {e}")
    
    def run_checks(self):
        """运行所有检查"""
        if not self.connect():
            self.send_alert("MySQL Connection Failed", "Cannot connect to MySQL server")
            return
        
        checks = [
            ('Replication', self.check_replication),
            ('Connections', self.check_connections),
            ('Slow Queries', self.check_slow_queries)
        ]
        
        for check_name, check_func in checks:
            success, message = check_func()
            if not success:
                self.send_alert(f"MySQL {check_name} Alert", message)
            else:
                print(f"{check_name}: {message}")

# 配置
config = {
    'mysql': {
        'host': '192.168.1.101',
        'user': 'monitor',
        'password': 'monitor_password',
        'database': 'mysql'
    },
    'email': {
        'smtp_server': 'smtp.company.com',
        'from': 'mysql-monitor@company.com',
        'to': ['dba@company.com', 'ops@company.com'],
        'username': 'mysql-monitor@company.com',
        'password': 'email_password'
    }
}

if __name__ == '__main__':
    monitor = MySQLMonitor(config)
    while True:
        monitor.run_checks()
        time.sleep(60)  # 每分钟检查一次
```

### 4.3 容量规划

#### 4.3.1 存储容量预测

```sql
-- 表增长趋势分析
SELECT 
    table_schema,
    table_name,
    ROUND(((data_length + index_length) / 1024 / 1024), 2) as current_size_mb,
    table_rows,
    ROUND((data_length / table_rows), 2) as avg_row_size,
    auto_increment,
    create_time,
    update_time,
    DATEDIFF(NOW(), create_time) as table_age_days,
    ROUND((table_rows / DATEDIFF(NOW(), create_time)), 0) as avg_rows_per_day
FROM 
    information_schema.tables
WHERE 
    table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
    AND table_rows > 0
    AND create_time IS NOT NULL
ORDER BY 
    (data_length + index_length) DESC;

-- 预测未来6个月的存储需求
SELECT 
    table_schema,
    table_name,
    current_size_mb,
    avg_rows_per_day,
    avg_row_size,
    ROUND((avg_rows_per_day * 180 * avg_row_size / 1024 / 1024), 2) as predicted_growth_mb,
    ROUND((current_size_mb + (avg_rows_per_day * 180 * avg_row_size / 1024 / 1024)), 2) as predicted_size_mb
FROM (
    SELECT 
        table_schema,
        table_name,
        ROUND(((data_length + index_length) / 1024 / 1024), 2) as current_size_mb,
        ROUND((table_rows / DATEDIFF(NOW(), create_time)), 0) as avg_rows_per_day,
        ROUND((data_length / table_rows), 2) as avg_row_size
    FROM 
        information_schema.tables
    WHERE 
        table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
        AND table_rows > 0
        AND create_time IS NOT NULL
        AND DATEDIFF(NOW(), create_time) > 30
) t
ORDER BY 
    predicted_growth_mb DESC;
```

## 5. 安全最佳实践

### 5.1 权限管理

#### 5.1.1 最小权限原则

```sql
-- 1. 创建专用用户
CREATE USER 'app_user'@'192.168.1.%' IDENTIFIED BY 'strong_password';
CREATE USER 'readonly_user'@'192.168.1.%' IDENTIFIED BY 'readonly_password';
CREATE USER 'backup_user'@'localhost' IDENTIFIED BY 'backup_password';

-- 2. 分配最小权限
-- 应用用户权限
GRANT SELECT, INSERT, UPDATE, DELETE ON app_db.* TO 'app_user'@'192.168.1.%';
GRANT EXECUTE ON app_db.* TO 'app_user'@'192.168.1.%';

-- 只读用户权限
GRANT SELECT ON app_db.* TO 'readonly_user'@'192.168.1.%';

-- 备份用户权限
GRANT SELECT, LOCK TABLES, SHOW VIEW, EVENT, TRIGGER ON *.* TO 'backup_user'@'localhost';
GRANT REPLICATION CLIENT ON *.* TO 'backup_user'@'localhost';

-- 3. 定期审计权限
SELECT 
    user,
    host,
    db,
    table_name,
    table_priv,
    column_priv
FROM 
    mysql.tables_priv
ORDER BY 
    user, host, db;
```

#### 5.1.2 密码策略

```sql
-- 密码验证插件配置
INSTALL PLUGIN validate_password SONAME 'validate_password.so';

-- 密码策略设置
SET GLOBAL validate_password.policy = 'STRONG';
SET GLOBAL validate_password.length = 12;
SET GLOBAL validate_password.mixed_case_count = 1;
SET GLOBAL validate_password.number_count = 1;
SET GLOBAL validate_password.special_char_count = 1;

-- 密码过期策略
ALTER USER 'app_user'@'192.168.1.%' PASSWORD EXPIRE INTERVAL 90 DAY;
SET GLOBAL default_password_lifetime = 90;
```

### 5.2 网络安全

#### 5.2.1 SSL/TLS配置

```sql
-- 生成SSL证书
-- openssl req -newkey rsa:2048 -days 3600 -nodes -keyout server-key.pem -out server-req.pem
-- openssl rsa -in server-key.pem -out server-key.pem
-- openssl x509 -req -in server-req.pem -days 3600 -CA ca.pem -CAkey ca-key.pem -set_serial 01 -out server-cert.pem

-- MySQL配置
[mysqld]
ssl-ca=/etc/mysql/ssl/ca.pem
ssl-cert=/etc/mysql/ssl/server-cert.pem
ssl-key=/etc/mysql/ssl/server-key.pem
require_secure_transport=ON

-- 强制SSL连接
ALTER USER 'app_user'@'192.168.1.%' REQUIRE SSL;
```

#### 5.2.2 防火墙配置

```bash
# iptables规则
# 只允许特定IP访问MySQL端口
iptables -A INPUT -p tcp --dport 3306 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 3306 -j DROP

# 限制连接频率
iptables -A INPUT -p tcp --dport 3306 -m connlimit --connlimit-above 10 -j DROP
iptables -A INPUT -p tcp --dport 3306 -m recent --set --name mysql
iptables -A INPUT -p tcp --dport 3306 -m recent --update --seconds 60 --hitcount 10 --name mysql -j DROP
```

## 6. 总结

### 6.1 核心实战经验

1. **性能优化**
   - 合理配置缓冲池大小和实例数
   - 根据业务特点设计索引策略
   - 监控慢查询和锁等待情况
   - 定期分析和优化SQL语句

2. **故障排查**
   - 建立完善的监控体系
   - 掌握常用的诊断工具和命令
   - 制定标准的故障处理流程
   - 定期进行故障演练

3. **架构设计**
   - 根据业务规模选择合适的架构
   - 考虑数据一致性和可用性的平衡
   - 设计合理的分库分表策略
   - 建立完善的备份和恢复机制

4. **运维管理**
   - 自动化日常运维任务
   - 建立完善的监控告警体系
   - 定期进行容量规划
   - 重视安全管理和权限控制

### 6.2 持续改进建议

1. **技术跟进**
   - 关注MySQL新版本特性
   - 学习新的优化技术和工具
   - 参与社区讨论和经验分享

2. **流程优化**
   - 不断完善运维流程
   - 提高自动化程度
   - 建立知识库和文档体系

3. **团队建设**
   - 提升团队技术水平
   - 建立轮岗和培训机制
   - 加强跨团队协作

通过这些实战经验和最佳实践，可以有效提升MySQL数据库系统的性能、稳定性和可维护性，为业务发展提供强有力的数据支撑。
