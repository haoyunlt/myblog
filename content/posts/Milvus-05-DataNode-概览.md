---
title: "Milvus-05-DataNode-概览"
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
description: "Milvus 源码剖析 - Milvus-05-DataNode-概览"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# Milvus-05-DataNode-概览

## 1. 模块概述

### 1.1 职责定义

DataNode负责数据消费、缓存和持久化，是Milvus数据写入路径的执行者。

**核心职责**：

1. **消息消费**
   - 订阅DML Channel（Pulsar/Kafka）
   - 消费InsertMsg/DeleteMsg
   - 维护消费进度（Checkpoint）

2. **数据缓存**
   - Growing Segment内存缓存
   - 构建内存索引（inverted index）
   - 处理Delete操作（Bloom Filter）

3. **数据Flush**
   - 接收DataCoord Flush命令
   - 序列化数据为Binlog格式
   - 写入Object Storage（S3/MinIO）

4. **Compaction执行**
   - 合并小Segment（MergeCompaction）
   - 清理删除数据（MixCompaction）
   - 数据聚类（ClusteringCompaction）

5. **时间同步**
   - 定期上报TimeTick到DataCoord
   - 保证数据可见性

### 1.2 架构图

```mermaid
flowchart TB
    subgraph DataNode["DataNode"]
        subgraph FlowGraph["FlowGraph 流式处理"]
            MsgStream[MsgStream Node<br/>消息消费]
            Filter[Filter Node<br/>消息过滤]
            Insert[Insert Node<br/>数据写入]
            Delete[Delete Node<br/>删除处理]
            TT[TimeTick Node<br/>时间同步]
        end
        
        subgraph Storage["存储层"]
            InsertBuf[Insert Buffer<br/>写入缓冲]
            DeleteBuf[Delete Buffer<br/>删除缓冲]
            BF[Bloom Filter<br/>删除标记]
        end
        
        subgraph Background["后台任务"]
            FM[Flush Manager<br/>Flush管理]
            CM[Compaction Manager<br/>Compaction管理]
            SYNC[Sync Manager<br/>同步管理]
        end
        
        MsgStream --> Filter
        Filter --> Insert
        Filter --> Delete
        Filter --> TT
        
        Insert --> InsertBuf
        Delete --> DeleteBuf
        Delete --> BF
        
        FM --> InsertBuf
        FM --> DeleteBuf
        CM --> FM
    end
    
    subgraph External["外部组件"]
        MQ[MessageQueue<br/>Pulsar/Kafka]
        OS[Object Storage<br/>S3/MinIO]
        DC[DataCoord]
    end
    
    MsgStream -.->|Subscribe| MQ
    FM -.->|Write Binlog| OS
    CM -.->|Read/Write| OS
    TT -.->|Report| DC
    
    style DataNode fill:#e1f5ff,stroke:#333,stroke-width:2px
```

### 1.3 核心数据流

```mermaid
sequenceDiagram
    participant MQ as MessageQueue
    participant DN as DataNode
    participant Buf as Insert Buffer
    participant OS as Object Storage
    participant DC as DataCoord
    
    loop 持续消费
        MQ->>DN: InsertMsg
        DN->>Buf: 缓存数据
        DN->>DN: 构建内存索引
    end
    
    alt Segment满 或 超时
        DC->>DN: FlushSegments命令
        DN->>Buf: 序列化数据
        DN->>OS: 写入Binlog
        DN->>DC: Flush完成通知
    end
```

---

## 2. 核心流程

### 2.1 FlowGraph机制

**FlowGraph**：流式数据处理框架

```go
// FlowGraph节点接口
type Node interface {
    Name() string
    Operate(in []Msg) []Msg
}

// 示例：Insert Node
type insertNode struct {
    insertBuffer *InsertBuffer
}

func (n *insertNode) Operate(in []Msg) []Msg {
    insertMsg := in[0].(*InsertMsg)
    
    // 写入缓冲区
    n.insertBuffer.Buffer(insertMsg.Data)
    
    // 传递给下游节点
    return []Msg{insertMsg}
}
```

**节点拓扑**：

```
MsgStream → Filter → Insert → Delete → TimeTick
                      ↓         ↓        ↓
                   Buffer    Buffer   Report
```

### 2.2 Flush流程

```mermaid
sequenceDiagram
    participant DC as DataCoord
    participant DN as DataNode
    participant Buf as Buffer
    participant BW as BinlogWriter
    participant OS as Object Storage
    
    DC->>DN: FlushSegments([seg1, seg2])
    
    loop 每个Segment
        DN->>Buf: GetData(segmentID)
        Buf-->>DN: FieldData
        
        DN->>BW: Serialize(FieldData)
        BW-->>DN: Binlog Bytes
        
        DN->>OS: Upload(binlog_path, data)
        OS-->>DN: Success
        
        DN->>OS: Upload(statslog_path, stats)
        DN->>OS: Upload(deltalog_path, deletes)
    end
    
    DN->>DC: FlushCompleted([seg1, seg2], binlog_paths)
```

**Binlog格式**：

```
InsertLog:

  - 字段1.binlog (Int64字段)
  - 字段2.binlog (FloatVector字段)
  - ...

StatsLog:

  - segment_stats.json (行数、大小等)

DeltaLog:

  - delta.binlog (删除的PrimaryKey列表)

```

### 2.3 Compaction流程

**MixCompaction（清理删除数据）**：

```mermaid
flowchart LR
    Input[输入Segment<br/>1000行，200行已删除] --> Read[读取Binlog]
    Read --> Filter[过滤已删除行]
    Filter --> Merge[合并数据]
    Merge --> Output[输出Segment<br/>800行]
    
    style Input fill:#ffe6e6
    style Output fill:#e6ffe6
```

**MergeCompaction（合并小文件）**：

```
输入: [Seg1(100MB), Seg2(50MB), Seg3(80MB)]
输出: [Seg_Merged(230MB)]
```

---

## 3. 关键设计

### 3.1 Insert Buffer

**目的**：缓存数据，减少Flush频率

```go
type InsertBuffer struct {
    // Channel -> SegmentID -> FieldData
    data map[string]map[int64]*FieldDataBuffer
    
    mu sync.RWMutex
}

// 写入数据
func (buf *InsertBuffer) Buffer(segmentID int64, fieldData *FieldData) {
    buf.mu.Lock()
    defer buf.mu.Unlock()
    
    if buf.data[segmentID] == nil {
        buf.data[segmentID] = NewFieldDataBuffer()
    }
    
    buf.data[segmentID].Append(fieldData)
}

// Flush时序列化
func (buf *InsertBuffer) Serialize(segmentID int64) ([]byte, error) {
    buf.mu.RLock()
    defer buf.mu.RUnlock()
    
    return buf.data[segmentID].MarshalBinlog()
}
```

### 3.2 Delete Buffer与Bloom Filter

**Delete处理**：

```go
type DeleteBuffer struct {
    // SegmentID -> PrimaryKeys
    deletes map[int64]*roaring.Bitmap
    
    // Bloom Filter：快速判断PK是否被删除
    bloomFilters map[int64]*BloomFilter
}

// 记录删除
func (buf *DeleteBuffer) Delete(segmentID int64, pk int64) {
    buf.deletes[segmentID].Add(pk)
    buf.bloomFilters[segmentID].Add(pk)
}

// 查询时过滤
func (buf *DeleteBuffer) IsDeleted(segmentID int64, pk int64) bool {
    // 先用Bloom Filter快速判断
    if !buf.bloomFilters[segmentID].Test(pk) {
        return false
    }
    
    // 再用精确数据确认
    return buf.deletes[segmentID].Contains(pk)
}
```

### 3.3 Checkpoint机制

**目的**：记录消费进度，支持故障恢复

```go
type Checkpoint struct {
    Channel  string
    Position msgstream.Position  // MessageQueue Position
    Timestamp uint64
}

// 定期持久化Checkpoint
func (dn *DataNode) saveCheckpoint() {
    for channel, pos := range dn.positions {
        checkpoint := &Checkpoint{
            Channel:   channel,
            Position:  pos,
            Timestamp: dn.latestTimestamp,
        }
        
        dn.etcd.Save(checkpointKey(channel), checkpoint)
    }
}

// 启动时恢复
func (dn *DataNode) recoverFromCheckpoint() {
    checkpoint := dn.etcd.Load(checkpointKey(channel))
    dn.msgStream.Seek(checkpoint.Position)
}
```

---

## 4. 性能与容量

### 4.1 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **Insert吞吐** | 10-50MB/s/node | 取决于CPU和内存 |
| **Flush延迟** | 1-5秒/GB | 取决于Object Storage带宽 |
| **Compaction吞吐** | 100GB/小时/node | CPU密集 |
| **内存占用** | 数据量*1.5 | 包含索引 |

### 4.2 容量规划

| 维度 | 容量 | 说明 |
|------|------|------|
| **Channel数量/节点** | 10-50 | 取决于CPU核数 |
| **Growing Segment缓存** | <内存50% | 避免OOM |
| **Compaction并发** | CPU核数*0.5 | CPU密集 |

---

## 5. 配置参数

```yaml
dataNode:
  # FlowGraph配置
  flowGraph:
    maxQueueLength: 1024      # 节点间队列长度
    maxParallelism: 1024      # 最大并行度
    
  # Flush配置
  flush:
    insertBufSize: 16777216   # 16MB
    deleteBufSize: 16777216   # 16MB
    
  # Compaction配置
  compaction:
    enabled: true
    memoryRatio: 0.5          # 最大使用50%内存
```

---

**相关文档**：

- [Milvus-00-总览.md](./Milvus-00-总览.md)
- [Milvus-03-DataCoord-概览.md](./Milvus-03-DataCoord-概览.md)
