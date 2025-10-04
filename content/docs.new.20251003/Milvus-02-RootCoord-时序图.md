# Milvus-02-RootCoord-时序图

## 1. CreateCollection时序图

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as Proxy
    participant RC as RootCoord
    participant SCH as Scheduler
    participant TSO as TSOAllocator
    participant ID as IDAllocator
    participant Meta as MetaTable
    participant ETCD as etcd
    participant DC as DataCoord
    participant QC as QueryCoord
    
    C->>P: CreateCollection(name, schema)
    P->>RC: CreateCollection RPC
    RC->>RC: 健康检查
    RC->>SCH: AddTask(createCollectionTask)
    
    SCH->>SCH: 等待DDL锁
    SCH->>SCH: 解析Schema
    SCH->>SCH: 参数校验
    
    SCH->>ID: AllocOne()
    ID->>ETCD: CompareAndSwap(idKey)
    ETCD-->>ID: OK
    ID-->>SCH: collectionID=123
    
    SCH->>TSO: GenerateTSO(1)
    TSO-->>SCH: timestamp
    
    SCH->>SCH: 创建VirtualChannels
    
    SCH->>Meta: AddCollection(collection)
    Meta->>ETCD: Save(/collection/123)
    ETCD-->>Meta: OK
    Meta->>Meta: 更新内存缓存
    Meta-->>SCH: Success
    
    par 异步广播
        SCH->>DC: BroadcastAlteredCollection
        DC->>DC: InvalidateCache
        DC-->>SCH: ACK
    and
        SCH->>QC: BroadcastAlteredCollection
        QC->>QC: InvalidateCache
        QC-->>SCH: ACK
    and
        SCH->>P: InvalidateCollectionMetaCache
        P->>P: RemoveCache
        P-->>SCH: ACK
    end
    
    SCH-->>RC: Task完成
    RC-->>P: Status(Success)
    P-->>C: Status(Success)
```

**时序图说明**：

1. **步骤1-4**：Client通过Proxy发起CreateCollection请求
2. **步骤5-7**：任务进入DDL Scheduler，等待串行执行
3. **步骤8-11**：分配CollectionID（从etcd批量预分配）
4. **步骤12-13**：分配创建时间戳
5. **步骤14**：创建VirtualChannels（从Channel池分配）
6. **步骤15-19**：持久化元数据到etcd，更新内存缓存
7. **步骤20-29**：并发广播失效通知到所有组件
8. **步骤30-32**：返回成功状态

---

## 2. AllocTimestamp时序图

```mermaid
sequenceDiagram
    autonumber
    participant P as Proxy
    participant RC as RootCoord
    participant TSO as TSOAllocator
    participant ETCD as etcd
    
    loop 每50ms
        TSO->>ETCD: Save(tsoKey, lastPhysical)
        Note over TSO: 定期持久化物理时间
    end
    
    P->>RC: AllocTimestamp(count=10)
    RC->>TSO: GenerateTSO(10)
    
    TSO->>TSO: 获取当前时间(UnixMilli)
    
    alt 物理时间前进
        TSO->>TSO: lastPhysical=currentTime
        TSO->>TSO: lastLogical=0
    end
    
    alt 逻辑计数器溢出
        TSO->>TSO: sleep(1ms)
        TSO->>TSO: lastPhysical=新时间
        TSO->>TSO: lastLogical=0
    end
    
    TSO->>TSO: ts = lastPhysical<<18 | lastLogical
    TSO->>TSO: lastLogical += 10
    
    TSO-->>RC: ts
    RC-->>P: Timestamp(ts, count=10)
    
    Note over P: 使用范围[ts, ts+9]
```

**TSO生成机制**：

```
时间轴：
  t0: physical=1000, logical=0      → TSO=1000<<18|0   = 262144000
  t1: physical=1000, logical=10     → TSO=1000<<18|10  = 262144010
  t2: physical=1000, logical=262143 → TSO=1000<<18|262143
  t3: physical=1001, logical=0      → TSO=1001<<18|0   = 262406144

特性：
- 单调递增：保证分布式顺序
- 高精度：毫秒+逻辑计数器
- 高性能：本地生成，减少etcd访问
```

---

## 3. DescribeCollection时序图

```mermaid
sequenceDiagram
    autonumber
    participant P as Proxy
    participant RC as RootCoord
    participant Meta as MetaTable
    participant ETCD as etcd
    
    P->>RC: DescribeCollection(name="docs")
    RC->>RC: 健康检查
    
    RC->>Meta: GetCollectionByName("default", "docs", ts)
    
    alt 内存缓存命中
        Meta-->>RC: Collection(from cache)
    else 缓存未命中
        Meta->>ETCD: Load(/collection/*)
        ETCD-->>Meta: CollectionMeta
        Meta->>Meta: 反序列化+缓存
        Meta-->>RC: Collection
    end
    
    RC->>RC: 构造Response
    RC-->>P: DescribeCollectionResponse
```

**缓存策略**：

```
Cache Key: dbID + collectionName
Cache Value: *model.Collection

失效时机：
1. DDL操作（CreateCollection/DropCollection/AlterCollection）
2. 接收到InvalidateCollectionMetaCache通知
3. 查询时发现版本不匹配（etcd比缓存新）

性能：
- Cache Hit: P99 < 5ms
- Cache Miss: P99 < 20ms (包含etcd查询)
```

---

## 4. CreatePartition时序图

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RC as RootCoord
    participant SCH as Scheduler
    participant ID as IDAllocator
    participant Meta as MetaTable
    participant ETCD as etcd
    
    C->>RC: CreatePartition(collection, partition)
    RC->>SCH: AddTask(createPartitionTask)
    
    SCH->>Meta: GetCollectionByName(collection)
    Meta-->>SCH: Collection
    
    SCH->>SCH: 检查分区数量限制
    alt 超过限制
        SCH-->>RC: Error(MaxPartitionNum)
        RC-->>C: Status(Error)
    end
    
    SCH->>ID: AllocOne()
    ID-->>SCH: partitionID
    
    SCH->>Meta: AddPartition(partition)
    Meta->>Meta: 更新Collection.Partitions
    Meta->>ETCD: Save(/collection/123/partition/456)
    ETCD-->>Meta: OK
    Meta-->>SCH: Success
    
    SCH->>RC: 广播失效通知
    SCH-->>RC: Task完成
    RC-->>C: Status(Success)
```

---

## 5. DropCollection时序图

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RC as RootCoord
    participant SCH as Scheduler
    participant Meta as MetaTable
    participant ETCD as etcd
    participant DC as DataCoord
    participant QC as QueryCoord
    
    C->>RC: DropCollection(collection)
    RC->>SCH: AddTask(dropCollectionTask)
    
    SCH->>Meta: GetCollectionByName(collection)
    Meta-->>SCH: Collection
    
    SCH->>Meta: ChangeCollectionState(Dropping)
    Meta->>ETCD: Update(state=Dropping)
    ETCD-->>Meta: OK
    
    par 通知Release
        SCH->>QC: ReleaseCollection(collectionID)
        QC->>QC: 卸载所有Segment
        QC-->>SCH: Success
    and
        SCH->>DC: ReleaseCollection(collectionID)
        DC->>DC: 标记Segment为Dropped
        DC-->>SCH: Success
    end
    
    SCH->>Meta: DropCollection(collectionID)
    Meta->>ETCD: Delete(/collection/123)
    ETCD-->>Meta: OK
    Meta->>Meta: 从缓存移除
    Meta-->>SCH: Success
    
    SCH->>SCH: 触发GC
    Note over SCH: 异步清理Binlog文件
    
    SCH-->>RC: Task完成
    RC-->>C: Status(Success)
```

**状态转换**：

```
CollectionCreated → CollectionDropping → CollectionDropped

步骤：
1. 标记为Dropping（不可见）
2. 通知QueryCoord/DataCoord释放资源
3. 删除etcd元数据
4. 异步GC清理Object Storage文件
```

---

## 6. ShowCollections时序图

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RC as RootCoord
    participant SCH as Scheduler
    participant Meta as MetaTable
    
    C->>RC: ShowCollections(dbName)
    RC->>SCH: AddTask(showCollectionTask)
    
    SCH->>Meta: ListCollections(dbName, ts)
    Meta->>Meta: 过滤：state==Created
    Meta-->>SCH: [coll1, coll2, ...]
    
    SCH->>SCH: 构造Response
    SCH-->>RC: [names, ids, timestamps]
    RC-->>C: ShowCollectionsResponse
```

---

## 7. 时间旅行（Time Travel）

```mermaid
sequenceDiagram
    participant C as Client
    participant RC as RootCoord
    participant Meta as MetaTable
    participant ETCD as etcd
    
    Note over C: 查询历史状态
    C->>RC: DescribeCollection(name, ts=100)
    RC->>Meta: GetCollectionByName(name, ts=100)
    
    Meta->>ETCD: LoadWithRevision(key, revision)
    ETCD-->>Meta: HistoricalData(ts=100)
    Meta-->>RC: Collection(历史状态)
    RC-->>C: DescribeCollectionResponse
    
    Note over C: 查询当前状态
    C->>RC: DescribeCollection(name, ts=MaxTimestamp)
    RC->>Meta: GetCollectionByName(name, ts=Max)
    Meta-->>RC: Collection(当前状态)
    RC-->>C: DescribeCollectionResponse
```

**时间旅行用途**：

1. **MVCC查询**：查询历史版本的Collection
2. **一致性保证**：使用特定Timestamp保证读一致性
3. **调试与审计**：回溯DDL操作历史

**实现机制**：

```go
// etcd支持基于Revision的查询
// Timestamp → Revision映射（单调递增）

func (m *MetaTable) GetCollectionByName(dbName, name string, ts Timestamp) (*Collection, error) {
    if ts == MaxTimestamp {
        // 查询最新版本（从缓存）
        return m.cache[dbName][name], nil
    }
    
    // 查询历史版本（从etcd）
    revision := m.timestampToRevision(ts)
    data := m.etcdCli.Get(ctx, collectionKey, clientv3.WithRev(revision))
    
    return parseCollection(data), nil
}
```

---

## 8. DDL串行化机制

```mermaid
sequenceDiagram
    participant T1 as DDL Task1
    participant T2 as DDL Task2
    participant T3 as DDL Task3
    participant SCH as Scheduler
    participant Lock as DdlTsLockManager
    
    par 并发提交
        T1->>SCH: AddTask(createCollection)
        T2->>SCH: AddTask(dropCollection)
        T3->>SCH: AddTask(createPartition)
    end
    
    SCH->>SCH: taskQueue接收任务
    
    loop 串行执行
        SCH->>Lock: Lock()
        Lock-->>SCH: Acquired
        
        SCH->>T1: Execute()
        T1->>T1: 执行DDL
        T1-->>SCH: Success
        
        SCH->>Lock: Unlock()
        
        SCH->>Lock: Lock()
        SCH->>T2: Execute()
        T2-->>SCH: Success
        SCH->>Lock: Unlock()
        
        SCH->>Lock: Lock()
        SCH->>T3: Execute()
        T3-->>SCH: Success
        SCH->>Lock: Unlock()
    end
```

**串行化原因**：

1. **元数据一致性**：避免并发修改导致冲突
2. **Timestamp顺序性**：保证DDL操作的全局顺序
3. **简化实现**：无需复杂的并发控制

**性能影响**：

- DDL操作频率低（通常<1/秒）
- 串行化对系统吞吐影响小
- DML/DQL操作不受影响（并发执行）

---

**相关文档**：
- [Milvus-02-RootCoord-概览.md](./Milvus-02-RootCoord-概览.md)
- [Milvus-02-RootCoord-API.md](./Milvus-02-RootCoord-API.md)
- [Milvus-02-RootCoord-数据结构.md](./Milvus-02-RootCoord-数据结构.md)

