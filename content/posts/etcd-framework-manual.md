---
title: "etcd 框架使用手册"
date: 2025-04-15T14:00:00+08:00
draft: false
featured: true
description: "etcd 分布式键值存储框架完整使用手册，涵盖安装、配置、API使用、集群管理等核心功能"
slug: "etcd-framework-manual"
author: "tommie blog"
categories: ["etcd", "框架手册"]
tags: ["etcd", "分布式", "键值存储", "使用手册"]
showComments: true
toc: true
tocOpen: true
showReadingTime: true
showWordCount: true
showBreadCrumbs: true
showPostNavLinks: true
useHugoToc: true
showRssButtonInSectionTermList: true
weight: 430
---

# etcd 框架使用手册

etcd 是一个分布式、可靠的键值存储系统，专为分布式系统中最关键的数据而设计。本手册将全面介绍 etcd 的安装、配置、使用和最佳实践。

## 1. 框架概述

### 1.1 etcd 特性

- **简单**：定义良好的用户友好 API (gRPC)
- **安全**：自动 TLS 和可选的客户端证书认证
- **快速**：基准测试显示 10,000 次/秒写入
- **可靠**：使用 Raft 算法实现分布式一致性

### 1.2 核心概念

- **键值存储**：存储任意键值对数据
- **监听机制**：监听键的变化并获得通知
- **租约机制**：为键设置 TTL（生存时间）
- **事务支持**：原子性的多操作事务
- **集群管理**：动态添加/删除节点

## 2. 安装与部署

### 2.1 单机安装

```bash
# 下载 etcd
ETCD_VER=v3.5.10
DOWNLOAD_URL=https://github.com/etcd-io/etcd/releases/download
curl -L ${DOWNLOAD_URL}/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz -o etcd-${ETCD_VER}-linux-amd64.tar.gz
tar xzvf etcd-${ETCD_VER}-linux-amd64.tar.gz
cd etcd-${ETCD_VER}-linux-amd64
sudo cp etcd etcdctl /usr/local/bin/

# 启动 etcd
etcd
```

### 2.2 集群部署

#### 静态配置集群

```bash
# 节点1
etcd --name infra1 \
  --initial-advertise-peer-urls http://10.0.1.10:2380 \
  --listen-peer-urls http://10.0.1.10:2380 \
  --advertise-client-urls http://10.0.1.10:2379 \
  --listen-client-urls http://10.0.1.10:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster 'infra1=http://10.0.1.10:2380,infra2=http://10.0.1.11:2380,infra3=http://10.0.1.12:2380' \
  --initial-cluster-state new

# 节点2
etcd --name infra2 \
  --initial-advertise-peer-urls http://10.0.1.11:2380 \
  --listen-peer-urls http://10.0.1.11:2380 \
  --advertise-client-urls http://10.0.1.11:2379 \
  --listen-client-urls http://10.0.1.11:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster 'infra1=http://10.0.1.10:2380,infra2=http://10.0.1.11:2380,infra3=http://10.0.1.12:2380' \
  --initial-cluster-state new

# 节点3
etcd --name infra3 \
  --initial-advertise-peer-urls http://10.0.1.12:2380 \
  --listen-peer-urls http://10.0.1.12:2380 \
  --advertise-client-urls http://10.0.1.12:2379 \
  --listen-client-urls http://10.0.1.12:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster 'infra1=http://10.0.1.10:2380,infra2=http://10.0.1.11:2380,infra3=http://10.0.1.12:2380' \
  --initial-cluster-state new
```

### 2.3 配置文件

```yaml
# etcd.conf.yml
name: 'infra1'
data-dir: /var/lib/etcd
wal-dir: /var/lib/etcd/wal
snapshot-count: 10000
heartbeat-interval: 100
election-timeout: 1000
quota-backend-bytes: 0
listen-peer-urls: http://10.0.1.10:2380
listen-client-urls: http://10.0.1.10:2379,http://127.0.0.1:2379
max-snapshots: 5
max-wals: 5
cors:
initial-advertise-peer-urls: http://10.0.1.10:2380
advertise-client-urls: http://10.0.1.10:2379
discovery:
discovery-fallback: 'proxy'
discovery-proxy:
discovery-srv:
initial-cluster: 'infra1=http://10.0.1.10:2380,infra2=http://10.0.1.11:2380,infra3=http://10.0.1.12:2380'
initial-cluster-token: 'etcd-cluster'
initial-cluster-state: 'new'
strict-reconfig-check: false
enable-v2: true
enable-pprof: true
proxy: 'off'
proxy-failure-wait: 5000
proxy-refresh-interval: 30000
proxy-dial-timeout: 1000
proxy-write-timeout: 5000
proxy-read-timeout: 0
client-transport-security:
  cert-file:
  key-file:
  client-cert-auth: false
  trusted-ca-file:
  auto-tls: false
peer-transport-security:
  cert-file:
  key-file:
  client-cert-auth: false
  trusted-ca-file:
  auto-tls: false
debug: false
logger: zap
log-outputs: [stderr]
log-level: info
```

## 3. 客户端使用

### 3.1 etcdctl 命令行工具

#### 基本操作

```bash
# 设置键值
etcdctl put /mykey "Hello World"

# 获取键值
etcdctl get /mykey

# 获取键值范围
etcdctl get /mykey /mykey2

# 获取前缀匹配的所有键
etcdctl get --prefix /my

# 删除键
etcdctl del /mykey

# 删除前缀匹配的所有键
etcdctl del --prefix /my

# 监听键的变化
etcdctl watch /mykey

# 监听前缀的变化
etcdctl watch --prefix /my
```

#### 事务操作

```bash
# 事务：如果 key1 的值是 "value1"，则设置 key2 为 "value2"
etcdctl txn --interactive
compares:
key1 = "value1"

success requests (get, put, del):
put key2 value2

failure requests (get, put, del):
put key2 failed
```

#### 租约操作

```bash
# 创建租约（TTL=60秒）
etcdctl lease grant 60

# 使用租约设置键值
etcdctl put --lease=<lease_id> /mykey "Hello World"

# 续约
etcdctl lease keep-alive <lease_id>

# 撤销租约
etcdctl lease revoke <lease_id>

# 查看租约信息
etcdctl lease timetolive <lease_id>
```

### 3.2 Go 客户端库

#### 基本连接

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "go.etcd.io/etcd/clientv3"
)

func main() {
    cli, err := clientv3.New(clientv3.Config{
        Endpoints:   []string{"localhost:2379"},
        DialTimeout: 5 * time.Second,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer cli.Close()

    // 基本操作示例
    basicOperations(cli)
}

func basicOperations(cli *clientv3.Client) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // Put
    _, err := cli.Put(ctx, "/mykey", "Hello World")
    if err != nil {
        log.Fatal(err)
    }

    // Get
    resp, err := cli.Get(ctx, "/mykey")
    if err != nil {
        log.Fatal(err)
    }
    for _, kv := range resp.Kvs {
        fmt.Printf("Key: %s, Value: %s\n", kv.Key, kv.Value)
    }

    // Delete
    _, err = cli.Delete(ctx, "/mykey")
    if err != nil {
        log.Fatal(err)
    }
}
```

#### 监听机制

```go
func watchExample(cli *clientv3.Client) {
    ctx := context.Background()
    
    // 监听单个键
    watchChan := cli.Watch(ctx, "/mykey")
    
    // 监听前缀
    // watchChan := cli.Watch(ctx, "/my", clientv3.WithPrefix())
    
    for watchResp := range watchChan {
        for _, event := range watchResp.Events {
            fmt.Printf("Event Type: %s, Key: %s, Value: %s\n", 
                event.Type, event.Kv.Key, event.Kv.Value)
        }
    }
}
```

#### 租约使用

```go
func leaseExample(cli *clientv3.Client) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // 创建租约
    resp, err := cli.Grant(ctx, 60)
    if err != nil {
        log.Fatal(err)
    }
    leaseID := resp.ID

    // 使用租约设置键值
    _, err = cli.Put(ctx, "/mykey", "Hello World", clientv3.WithLease(leaseID))
    if err != nil {
        log.Fatal(err)
    }

    // 续约
    ch, kaerr := cli.KeepAlive(context.TODO(), leaseID)
    if kaerr != nil {
        log.Fatal(kaerr)
    }

    // 处理续约响应
    go func() {
        for ka := range ch {
            fmt.Printf("TTL: %d\n", ka.TTL)
        }
    }()
}
```

#### 事务操作

```go
func transactionExample(cli *clientv3.Client) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // 事务：如果 key1 不存在，则创建 key1 和 key2
    txn := cli.Txn(ctx)
    txnResp, err := txn.If(
        clientv3.Compare(clientv3.CreateRevision("/key1"), "=", 0),
    ).Then(
        clientv3.OpPut("/key1", "value1"),
        clientv3.OpPut("/key2", "value2"),
    ).Else(
        clientv3.OpGet("/key1"),
    ).Commit()

    if err != nil {
        log.Fatal(err)
    }

    if txnResp.Succeeded {
        fmt.Println("Transaction succeeded")
    } else {
        fmt.Println("Transaction failed")
    }
}
```

## 4. 集群管理

### 4.1 成员管理

```bash
# 查看集群成员
etcdctl member list

# 添加成员
etcdctl member add infra4 --peer-urls=http://10.0.1.13:2380

# 移除成员
etcdctl member remove <member_id>

# 更新成员
etcdctl member update <member_id> --peer-urls=http://10.0.1.13:2380
```

### 4.2 集群状态

```bash
# 查看集群健康状态
etcdctl endpoint health

# 查看集群状态
etcdctl endpoint status

# 查看集群性能
etcdctl check perf
```

### 4.3 数据备份与恢复

```bash
# 备份数据
etcdctl snapshot save backup.db

# 查看快照状态
etcdctl snapshot status backup.db

# 恢复数据
etcdctl snapshot restore backup.db \
  --name infra1 \
  --initial-cluster infra1=http://10.0.1.10:2380,infra2=http://10.0.1.11:2380,infra3=http://10.0.1.12:2380 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-advertise-peer-urls http://10.0.1.10:2380
```

## 5. 安全配置

### 5.1 TLS 配置

```bash
# 生成 CA 证书
cfssl gencert -initca ca-csr.json | cfssljson -bare ca

# 生成服务器证书
cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json -profile=server server-csr.json | cfssljson -bare server

# 生成客户端证书
cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json -profile=client client-csr.json | cfssljson -bare client

# 启动带 TLS 的 etcd
etcd --name infra1 \
  --cert-file=server.pem \
  --key-file=server-key.pem \
  --trusted-ca-file=ca.pem \
  --client-cert-auth \
  --peer-cert-file=server.pem \
  --peer-key-file=server-key.pem \
  --peer-trusted-ca-file=ca.pem \
  --peer-client-cert-auth
```

### 5.2 认证与授权

```bash
# 启用认证
etcdctl auth enable

# 创建用户
etcdctl user add myuser

# 创建角色
etcdctl role add myrole

# 为角色授权
etcdctl role grant-permission myrole readwrite /mykey

# 将用户分配给角色
etcdctl user grant-role myuser myrole

# 使用认证
etcdctl --user myuser:password get /mykey
```

## 6. 性能调优

### 6.1 硬件建议

- **CPU**：2-4 核心
- **内存**：8GB+
- **磁盘**：SSD，独立的 WAL 目录
- **网络**：1Gbps+，低延迟

### 6.2 配置优化

```yaml
# 性能相关配置
heartbeat-interval: 100          # 心跳间隔（毫秒）
election-timeout: 1000           # 选举超时（毫秒）
snapshot-count: 100000           # 快照触发阈值
max-snapshots: 5                 # 保留快照数量
max-wals: 5                      # 保留 WAL 文件数量
quota-backend-bytes: 2147483648  # 后端存储配额（2GB）
auto-compaction-retention: "1h"  # 自动压缩保留时间
auto-compaction-mode: periodic   # 自动压缩模式
```

### 6.3 监控指标

```bash
# 关键监控指标
etcd_server_proposals_committed_total    # 提交的提案总数
etcd_server_proposals_applied_total      # 应用的提案总数
etcd_server_proposals_pending            # 待处理的提案数
etcd_disk_wal_fsync_duration_seconds     # WAL fsync 延迟
etcd_disk_backend_commit_duration_seconds # 后端提交延迟
etcd_mvcc_db_total_size_in_bytes         # 数据库总大小
etcd_network_peer_round_trip_time_seconds # 节点间往返时间
```

## 7. 故障排查

### 7.1 常见问题

#### 集群分裂
```bash
# 检查集群状态
etcdctl endpoint status --cluster

# 检查网络连通性
etcdctl endpoint health --cluster
```

#### 磁盘空间不足
```bash
# 检查数据库大小
etcdctl endpoint status

# 压缩历史版本
etcdctl compact <revision>

# 整理碎片
etcdctl defrag
```

#### 性能问题
```bash
# 检查性能
etcdctl check perf

# 查看慢查询
etcdctl get --prefix / --limit=1000 --print-value-only=false
```

### 7.2 日志分析

```bash
# 查看 etcd 日志
journalctl -u etcd -f

# 常见错误模式
grep "failed to send" /var/log/etcd.log
grep "election timeout" /var/log/etcd.log
grep "disk space" /var/log/etcd.log
```

## 8. 最佳实践

### 8.1 部署建议

1. **奇数节点**：使用 3、5、7 个节点
2. **地理分布**：跨可用区部署
3. **资源隔离**：独立的磁盘和网络
4. **监控告警**：完善的监控体系

### 8.2 使用建议

1. **键设计**：使用有意义的前缀
2. **值大小**：控制在 1MB 以内
3. **事务使用**：合理使用事务保证一致性
4. **连接管理**：复用客户端连接

### 8.3 运维建议

1. **定期备份**：自动化备份策略
2. **版本升级**：渐进式升级
3. **容量规划**：监控增长趋势
4. **安全加固**：启用 TLS 和认证

## 9. 集成示例

### 9.1 服务发现

```go
// 服务注册
func registerService(cli *clientv3.Client, serviceName, serviceAddr string) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // 创建租约
    resp, err := cli.Grant(ctx, 30)
    if err != nil {
        return err
    }

    // 注册服务
    key := fmt.Sprintf("/services/%s/%s", serviceName, serviceAddr)
    _, err = cli.Put(ctx, key, serviceAddr, clientv3.WithLease(resp.ID))
    if err != nil {
        return err
    }

    // 续约
    ch, kaerr := cli.KeepAlive(context.TODO(), resp.ID)
    if kaerr != nil {
        return kaerr
    }

    go func() {
        for ka := range ch {
            // 处理续约响应
            _ = ka
        }
    }()

    return nil
}

// 服务发现
func discoverServices(cli *clientv3.Client, serviceName string) ([]string, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    resp, err := cli.Get(ctx, fmt.Sprintf("/services/%s/", serviceName), clientv3.WithPrefix())
    if err != nil {
        return nil, err
    }

    var services []string
    for _, kv := range resp.Kvs {
        services = append(services, string(kv.Value))
    }

    return services, nil
}
```

### 9.2 分布式锁

```go
import "go.etcd.io/etcd/clientv3/concurrency"

func distributedLock(cli *clientv3.Client) error {
    // 创建会话
    session, err := concurrency.NewSession(cli)
    if err != nil {
        return err
    }
    defer session.Close()

    // 创建互斥锁
    mutex := concurrency.NewMutex(session, "/my-lock")

    // 获取锁
    ctx := context.Background()
    if err := mutex.Lock(ctx); err != nil {
        return err
    }

    // 执行临界区代码
    fmt.Println("Acquired lock, doing work...")
    time.Sleep(5 * time.Second)

    // 释放锁
    if err := mutex.Unlock(ctx); err != nil {
        return err
    }

    return nil
}
```

### 9.3 配置管理

```go
type ConfigManager struct {
    cli    *clientv3.Client
    prefix string
}

func NewConfigManager(cli *clientv3.Client, prefix string) *ConfigManager {
    return &ConfigManager{
        cli:    cli,
        prefix: prefix,
    }
}

func (cm *ConfigManager) Set(key, value string) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    _, err := cm.cli.Put(ctx, cm.prefix+key, value)
    return err
}

func (cm *ConfigManager) Get(key string) (string, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    resp, err := cm.cli.Get(ctx, cm.prefix+key)
    if err != nil {
        return "", err
    }

    if len(resp.Kvs) == 0 {
        return "", fmt.Errorf("key not found")
    }

    return string(resp.Kvs[0].Value), nil
}

func (cm *ConfigManager) Watch(key string, callback func(string, string)) {
    watchChan := cm.cli.Watch(context.Background(), cm.prefix+key)
    
    for watchResp := range watchChan {
        for _, event := range watchResp.Events {
            callback(string(event.Kv.Key), string(event.Kv.Value))
        }
    }
}
```

## 10. 总结

etcd 是一个功能强大的分布式键值存储系统，适用于配置管理、服务发现、分布式锁等场景。通过本手册的学习，你应该能够：

1. 正确安装和配置 etcd 集群
2. 使用各种客户端工具操作 etcd
3. 实现常见的分布式系统模式
4. 进行性能调优和故障排查
5. 遵循最佳实践确保系统稳定性

在实际使用中，建议根据具体业务需求选择合适的配置和使用模式，并建立完善的监控和运维体系。
