---
title: "Milvus-04-QueryCoord-概览"
date: 2025-10-04T20:42:30+08:00
draft: false
tags:
  - Milvus
  - 架构设计
  - 概览
  - 源码分析
categories:
  - Milvus
  - 向量数据库
  - 分布式系统
series: "milvus-source-analysis"
description: "Milvus 源码剖析 - Milvus-04-QueryCoord-概览"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# Milvus-04-QueryCoord-概览

## 1. 模块概述

### 1.1 职责定义

QueryCoord（查询协调器）负责管理查询节点、协调数据加载和查询任务分配。

**核心职责**：

1. **QueryNode管理**
   - 节点注册与心跳监控
   - 节点资源统计
   - 故障检测与恢复

2. **Collection加载管理**
   - LoadCollection/ReleaseCollection
   - Segment分配到QueryNode
   - 副本管理（Replica）

3. **负载均衡**
   - Segment自动迁移
   - Channel负载均衡
   - 节点间负载均衡

4. **Handoff协调**
   - Growing → Sealed Segment切换
   - 通知QueryNode加载新Segment
   - 卸载旧Segment

5. **查询任务协调**
   - ShardLeader管理
   - 查询路由信息提供
   - DQL Channel管理

### 1.2 架构图

```mermaid
flowchart TB
    subgraph QueryCoord["QueryCoord Server"]
        API[gRPC API Layer]
        
        subgraph Managers["管理器"]
            NM[NodeManager<br/>QueryNode管理]
            TM[TargetManager<br/>目标状态管理]
            DM[DistManager<br/>实际分布管理]
            Meta[Meta<br/>元数据]
        end
        
        subgraph Schedulers["调度器"]
            JS[JobScheduler<br/>Job调度]
            TS[TaskScheduler<br/>Task调度]
            DC[DistController<br/>分布控制器]
        end
        
        subgraph Checkers["检查器"]
            CC[ChannelChecker<br/>Channel检查]
            SC[SegmentChecker<br/>Segment检查]
            BC[BalanceChecker<br/>负载均衡检查]
        end
        
        subgraph Observers["观察者"]
            CO[CollectionObserver<br/>Collection观察]
            TO[TargetObserver<br/>目标观察]
            RO[ReplicaObserver<br/>副本观察]
        end
        
        API --> JS
        JS --> TS
        TS --> DC
        
        DC --> NM
        DC --> TM
        DC --> DM
        
        CC --> DM
        SC --> DM
        BC --> DM
        
        CO --> TM
        TO --> TM
        RO --> Meta
    end
    
    subgraph External["外部组件"]
        QN[QueryNode集群]
        ETCD[etcd]
        DC_ext[DataCoord]
    end
    
    NM -.->|监控| QN
    DC -.->|分配任务| QN
    Meta <-->|元数据| ETCD
    TM -.->|Handoff通知| DC_ext
    
    style QueryCoord fill:#e1f5ff,stroke:#333,stroke-width:2px
```

### 1.3 核心概念

**Target（目标状态）**：

- Collection应该加载哪些Segment
- 每个Segment应该在哪些节点上（副本数）

**Distribution（实际分布）**：

- 当前各QueryNode实际加载的Segment
- 实时收集QueryNode心跳获取

**Reconcile（协调）**：

- 对比Target与Distribution的差异
- 生成LoadSegment/ReleaseSegment任务
- 驱动Distribution收敛到Target

```mermaid
flowchart LR
    Target[Target<br/>目标状态] -->|Diff| Reconcile[Reconcile<br/>协调器]
    Dist[Distribution<br/>实际状态] -->|Diff| Reconcile
    
    Reconcile -->|LoadSegment| QN[QueryNode]
    Reconcile -->|ReleaseSegment| QN
    
    QN -->|心跳上报| Dist
    
    style Target fill:#e6ffe6
    style Dist fill:#ffe6e6
    style Reconcile fill:#e6f3ff
```

---

## 2. 核心流程

### 2.1 LoadCollection流程

```mermaid
sequenceDiagram
    autonumber
    participant P as Proxy
    participant QC as QueryCoord
    participant TM as TargetManager
    participant DC as DistController
    participant QN as QueryNode
    
    P->>QC: LoadCollection(collection, replica=2)
    QC->>QC: 创建LoadCollectionJob
    QC->>TM: 更新Target(collection, segments)
    
    loop 协调循环
        TM->>DC: GetTarget(collection)
        DC->>DC: GetDistribution(collection)
        DC->>DC: Diff(Target, Distribution)
        
        alt Segment未加载
            DC->>QN: LoadSegment(segmentID, replica1)
            DC->>QN: LoadSegment(segmentID, replica2)
        end
        
        QN->>QN: 从Object Storage加载Segment
        QN->>DC: 心跳上报(已加载segmentID)
        DC->>DC: 更新Distribution
        
        DC->>DC: 检查Target==Distribution?
        alt 已收敛
            DC-->>QC: LoadCollection完成
        end
    end
    
    QC-->>P: Status(Success, 进度100%)
```

### 2.2 Handoff流程

**触发场景**：DataNode Flush完成，Segment从Growing变为Sealed

```mermaid
sequenceDiagram
    participant DC as DataCoord
    participant QC as QueryCoord
    participant TO as TargetObserver
    participant TM as TargetManager
    participant Checker as SegmentChecker
    participant QN as QueryNode
    
    DC->>QC: NotifySegmentSeal(segmentID)
    QC->>TO: 接收通知
    TO->>TM: 更新Target(+sealed_segment)
    
    Checker->>Checker: 定期检查(每1秒)
    Checker->>TM: GetTarget()
    Checker->>Checker: Diff(Target, Dist)
    
    Checker->>QN: LoadSegment(sealed_segment)
    QN->>QN: 加载Sealed Segment
    QN-->>Checker: 心跳上报
    
    Checker->>TM: 更新Target(-growing_segment)
    Checker->>QN: ReleaseSegment(growing_segment)
    QN->>QN: 卸载Growing Segment
```

### 2.3 负载均衡流程

**触发条件**：

- 节点间Segment数量差异>20%
- 节点内存使用率>80%
- 手动触发Balance API

```mermaid
flowchart TD
    Start[BalanceChecker<br/>每分钟扫描] --> CalcScore[计算每个节点负载分数]
    CalcScore --> CheckImbalance{不平衡?}
    
    CheckImbalance -->|否| End[结束]
    CheckImbalance -->|是| SelectSegment[选择要迁移的Segment]
    
    SelectSegment --> SelectTarget[选择目标节点]
    SelectTarget --> GenPlan[生成Balance Plan]
    
    GenPlan --> Load[目标节点LoadSegment]
    Load --> Release[源节点ReleaseSegment]
    
    Release --> Verify{验证迁移成功}
    Verify -->|成功| UpdateMeta[更新元数据]
    Verify -->|失败| Rollback[回滚]
    
    UpdateMeta --> End
    Rollback --> End
    
    style CheckImbalance fill:#ffe6e6
    style GenPlan fill:#e6f3ff
```

---

## 3. 关键设计

### 3.1 副本管理（Replica）

**目的**：提高查询可用性和吞吐量

```mermaid
flowchart LR
    C[Collection<br/>4 Shards] --> R1[Replica1]
    C --> R2[Replica2]
    
    R1 --> S1[Shard0→QN1]
    R1 --> S2[Shard1→QN2]
    R1 --> S3[Shard2→QN3]
    R1 --> S4[Shard3→QN4]
    
    R2 --> S5[Shard0→QN5]
    R2 --> S6[Shard1→QN6]
    R2 --> S7[Shard2→QN7]
    R2 --> S8[Shard3→QN8]
    
    style R1 fill:#e6ffe6
    style R2 fill:#ffe6e6
```

**负载均衡时选择副本**：

```go
// Proxy查询时选择副本
func selectReplica(replicas []*Replica) *Replica {
    // 轮询或基于负载选择
    return replicas[rand.Intn(len(replicas))]
}
```

### 3.2 Task与Job

**Job**：用户级操作（LoadCollection、ReleaseCollection）
**Task**：具体执行单元（LoadSegment、ReleaseSegment）

```
LoadCollectionJob
  ├── LoadSegmentTask (Segment1 → QN1)
  ├── LoadSegmentTask (Segment1 → QN2)  # 副本2
  ├── LoadSegmentTask (Segment2 → QN1)
  └── ...
```

### 3.3 Resource Group

**资源隔离**：将QueryNode划分到不同资源组

```yaml
ResourceGroups:

  - Name: "rg_high_priority"
    Nodes: [QN1, QN2, QN3]
    Capacity: 100GB
    
  - Name: "rg_low_priority"
    Nodes: [QN4, QN5]
    Capacity: 50GB

# Collection可指定ResourceGroup
LoadCollection(collection="important", resource_group="rg_high_priority")
```

---

## 4. 性能与容量

### 4.1 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **LoadCollection延迟** | 数据量/加载速度 | 10GB约10-30秒 |
| **Handoff延迟** | P99: 5秒 | Segment切换 |
| **Balance延迟** | 数据量/带宽 | 1GB约5-10秒 |
| **HeartBeat处理** | >10000 QPS | 来自QueryNode |

### 4.2 容量规划

| 维度 | 容量 | 说明 |
|------|------|------|
| **QueryNode数量** | 1000 | 单QueryCoord |
| **Loaded Collection数量** | 100 | 并发加载 |
| **Segment数量** | 100万 | 内存占用约5GB |

---

## 5. 配置参数

```yaml
queryCoord:
  # 心跳配置
  heartbeatAvailableTime: 10000  # QueryNode心跳超时(ms)
  
  # 负载均衡
  balanceIntervalSeconds: 60     # Balance检查周期
  balanceSegmentCntThreshold: 100  # 触发Balance的Segment差异阈值
  
  # 任务调度
  taskExecutionCap: 256          # 最大并发Task数
  
  # Handoff
  handoffSegmentNum: 4           # 每次Handoff的Segment数
```

---

**相关文档**：

- [Milvus-00-总览.md](./Milvus-00-总览.md)
- [Milvus-06-QueryNode-概览.md](./Milvus-06-QueryNode-概览.md) *(待生成)*
