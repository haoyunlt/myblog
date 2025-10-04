# Milvus-01-Proxy-时序图

本文档提供Proxy模块核心API的详细时序图，展示各类请求从接收到响应的完整调用链路。

## 1. CreateCollection 时序图

### 1.1 完整流程

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as Proxy
    participant Auth as 鉴权模块
    participant RL as 限流器
    participant DDL as DDL Queue
    participant MC as MetaCache
    participant RC as RootCoord
    participant E as etcd
    participant DC as DataCoord
    participant QC as QueryCoord
    
    C->>P: CreateCollection(name, schema)
    Note over P: gRPC接收请求
    
    P->>P: 健康检查(GetStateCode)
    alt Proxy状态异常
        P-->>C: Status(Abnormal)
    end
    
    P->>Auth: CheckPrivilege(user, CreateCollection)
    Auth->>RC: ListPolicy(user, role)
    RC-->>Auth: Policies
    Auth-->>P: 权限验证结果
    alt 权限不足
        P-->>C: Status(PermissionDenied)
    end
    
    P->>RL: Check(db, collection, DDL, 1)
    RL->>RL: TokenBucket.TryAlloc(1)
    alt 触发限流
        RL-->>P: RateLimitExceeded
        P-->>C: Status(RateLimitExceeded)
    end
    
    P->>DDL: createCollectionTask入队
    Note over DDL: 串行执行，等待前序DDL完成
    
    DDL->>DDL: PreExecute
    DDL->>DDL: 解析Schema(protobuf)
    DDL->>DDL: 参数校验(name、fields、pk)
    alt Schema无效
        DDL-->>P: InvalidSchema
        P-->>C: Status(InvalidSchema)
    end
    
    DDL->>DDL: Execute
    DDL->>RC: CreateCollection RPC
    Note over RC: 核心处理
    RC->>RC: 分配CollectionID
    RC->>RC: 创建VirtualChannels
    RC->>E: 持久化元数据(CollectionInfo)
    E-->>RC: OK
    RC->>RC: 注册Collection到TSO
    RC-->>DDL: Status(Success, collectionID)
    
    DDL->>MC: 更新MetaCache(collectionID, schema)
    
    par 异步广播
        RC->>DC: InvalidateCollectionMetaCache
        DC->>DC: 移除缓存
        DC-->>RC: ACK
    and
        RC->>QC: InvalidateCollectionMetaCache
        QC->>QC: 移除缓存
        QC-->>RC: ACK
    and
        RC->>P: InvalidateCollectionMetaCache
        P->>MC: RemoveCollection(name)
        P-->>RC: ACK
    end
    
    DDL->>DDL: PostExecute
    DDL->>DDL: 构造返回结果
    DDL-->>P: 任务完成通知
    P-->>C: Status(Success)
```

### 1.2 时序图关键点说明

**步骤1-4：请求接收与健康检查**
- Proxy通过gRPC接收客户端请求
- 检查Proxy状态（Healthy/Initializing/Abnormal）
- 状态异常时直接返回错误，避免级联失败

**步骤5-8：权限验证**
- 调用鉴权模块检查用户权限
- 鉴权模块从RootCoord获取用户的策略（Policy）
- 检查是否有`CreateCollection`权限
- 权限不足时返回`PermissionDenied`错误

**步骤9-13：限流检查**
- 限流器基于TokenBucket算法
- DDL操作通常配额较大，很少触发限流
- 限流触发时返回`RateLimitExceeded`，建议客户端退避重试

**步骤14-15：任务入队**
- 创建`createCollectionTask`对象
- 进入DDL队列（串行执行）
- DDL队列保证元数据操作的顺序性

**步骤16-22：PreExecute阶段**
- 解析Schema（从protobuf bytes反序列化）
- 校验参数：
  - Collection名称合法性（长度、字符集）
  - 字段定义完整性（必须有主键、向量字段）
  - 向量维度有效性（≤32768）
  - Shard数量合理性（1-64）

**步骤23-31：Execute阶段（核心）**
- 调用RootCoord的`CreateCollection` RPC
- RootCoord执行：
  1. 分配全局唯一CollectionID
  2. 创建VirtualChannels（数量=ShardsNum）
  3. 持久化元数据到etcd
  4. 注册到TSO服务（分配时间戳）
- 返回成功状态和CollectionID

**步骤32：MetaCache更新**
- Proxy更新本地MetaCache
- 后续请求可直接从缓存读取，无需调用RootCoord

**步骤33-44：异步广播失效通知**
- RootCoord并发向所有Proxy、DataCoord、QueryCoord广播
- 各组件收到通知后移除本地缓存
- 保证分布式缓存一致性

**步骤45-48：PostExecute与返回**
- 构造返回结果
- 通知等待者任务完成
- 返回Status给客户端

### 1.3 异常场景

```mermaid
sequenceDiagram
    participant C as Client
    participant P as Proxy
    participant DDL as DDL Queue
    participant RC as RootCoord
    participant E as etcd
    
    C->>P: CreateCollection(name="existing")
    P->>DDL: 任务入队
    DDL->>RC: CreateCollection RPC
    RC->>E: 检查Collection是否存在
    E-->>RC: 已存在
    RC-->>DDL: Status(CollectionAlreadyExists)
    DDL-->>P: 任务完成(失败)
    P-->>C: Status(CollectionAlreadyExists, "collection already exists")
    
    Note over C: 客户端处理错误
    alt 幂等处理
        C->>C: 忽略错误（已存在即成功）
    else 重命名
        C->>P: CreateCollection(name="existing_v2")
    end
```

---

## 2. Insert 时序图

### 2.1 完整流程

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as Proxy
    participant DML as DML Queue
    participant MC as MetaCache
    participant ID as IDAllocator
    participant DC as DataCoord
    participant CH as ChannelMgr
    participant MQ as MessageQueue
    participant DN as DataNode
    participant OS as Object Storage
    
    C->>P: Insert(collection, fields_data)
    
    P->>P: 健康检查
    P->>P: 鉴权与限流(同CreateCollection)
    
    P->>DML: insertTask入队(并发)
    Note over DML: DML队列支持并发执行
    
    DML->>DML: PreExecute
    DML->>MC: GetCollectionInfo(db, name)
    alt Cache Hit
        MC-->>DML: CollectionInfo(from cache)
    else Cache Miss
        MC->>P: RootCoord.DescribeCollection
        P-->>MC: CollectionInfo
        MC->>MC: 更新缓存
        MC-->>DML: CollectionInfo
    end
    
    DML->>DML: 参数校验(字段数量、类型匹配)
    alt 字段不匹配
        DML-->>P: InvalidFieldData
        P-->>C: Status(InvalidFieldData)
    end
    
    DML->>DML: Execute
    
    alt Schema.AutoID == true
        DML->>ID: AllocIDs(NumRows)
        ID->>P: RootCoord.AllocID(count)
        P-->>ID: [idBegin, idEnd]
        ID-->>DML: [idBegin, idEnd]
        DML->>DML: fillAutoID(fields_data, idBegin, idEnd)
    end
    
    DML->>DML: hashPrimaryKeys(fields_data)
    Note over DML: 按PK哈希分配到不同Shard
    DML->>DML: groupByHash(fields_data, ShardNum)
    
    DML->>CH: getVChannels(collectionID)
    CH->>MC: GetCollectionInfo(collectionID)
    MC-->>CH: VChannels列表
    CH-->>DML: ["vchan_0", "vchan_1", ...]
    
    loop 每个Shard
        DML->>DC: AssignSegmentID(channel, NumRows)
        DC->>DC: 查找或创建Segment
        DC-->>DML: SegmentID
        
        DML->>DML: 构造InsertMsg
        Note over DML: InsertMsg包含collectionID、partitionID、segmentID、fields_data
        
        DML->>CH: getOrCreateDMLStream(collectionID)
        CH-->>DML: MsgStream
        
        DML->>MQ: Produce(InsertMsg)
        Note over MQ: Pulsar/Kafka消息队列
        MQ-->>DML: MessageID
    end
    
    par 异步消费（DataNode）
        MQ->>DN: 订阅消息（按Channel）
        DN->>DN: 消费InsertMsg
        DN->>DN: 数据缓存到内存
        DN->>DN: 构建内存索引(Growing Segment)
        
        DN->>DN: 判断Segment是否满
        alt Segment满(大小>512MB或行数>100万)
            DN->>DN: Seal Segment
            DN->>DC: 报告Segment状态(Sealed)
            DC->>DN: FlushSegment命令
            DN->>DN: 序列化数据
            DN->>OS: 写入Binlog(insert_log)
            DN->>OS: 写入Statslog(stats_log)
            DN->>OS: 写入Deltalog(delete_log)
            DN->>DC: Flush完成
            DC->>DC: 更新Segment状态(Flushed)
        end
    end
    
    DML->>DML: PostExecute
    DML->>DML: 构造MutationResult
    Note over DML: 包含IDs、Timestamp、InsertCnt
    DML-->>P: 任务完成
    
    P-->>C: MutationResult(IDs, Timestamp)
    Note over C: 客户端可用Timestamp进行一致性查询
```

### 2.2 时序图关键点说明

**步骤1-5：请求接收与前置检查**
- 与CreateCollection类似的健康检查、鉴权、限流
- DML任务进入并发队列（与DDL串行不同）

**步骤6-13：MetaCache查询**
- 优先从本地缓存获取Collection元信息
- Cache Miss时调用RootCoord并更新缓存
- 缓存命中率通常>95%

**步骤14-17：参数校验**
- 检查字段数量是否匹配Schema
- 检查字段类型是否匹配（Int64/Float/Vector等）
- 检查向量维度是否正确
- 检查主键是否存在（非AutoID场景）

**步骤18-24：AutoID处理**
- 如果主键字段标记为AutoID
- 调用IDAllocator分配唯一ID
- IDAllocator从RootCoord批量分配（每次1000个）
- 填充ID到主键字段

**步骤25-27：数据分片**
- 按主键哈希值分配到不同Shard
- 哈希算法：`hash(primaryKey) % ShardNum`
- 保证相同主键总是路由到同一Shard（Upsert/Delete一致性）

**步骤28-31：Channel查询**
- 获取Collection的VirtualChannel列表
- 每个Shard对应一个VChannel
- VChannel映射到PChannel（物理通道）

**步骤32-41：消息发布**
- 为每个Shard生成InsertMsg
- 调用DataCoord分配SegmentID
- 构造消息并发布到MessageQueue
- 返回MessageID作为确认

**步骤42-53：异步消费（DataNode）**
- DataNode订阅对应的Channel
- 消费InsertMsg并缓存数据
- 构建内存索引（Growing Segment）
- Segment满时触发Flush：
  1. Seal Segment（不再接受新数据）
  2. 序列化数据为Binlog
  3. 写入Object Storage
  4. 通知DataCoord更新状态

**步骤54-58：返回结果**
- 构造MutationResult
- 包含插入的主键ID列表
- 包含操作Timestamp（用于一致性查询）

### 2.3 数据分片示意

```mermaid
flowchart LR
    Data[原始数据<br/>1000行] --> Hash{按PK哈希}
    Hash --> Shard0[Shard0<br/>250行]
    Hash --> Shard1[Shard1<br/>300行]
    Hash --> Shard2[Shard2<br/>200行]
    Hash --> Shard3[Shard3<br/>250行]
    
    Shard0 --> VC0[VChannel_0]
    Shard1 --> VC1[VChannel_1]
    Shard2 --> VC2[VChannel_2]
    Shard3 --> VC3[VChannel_3]
    
    VC0 --> DN0[DataNode0]
    VC1 --> DN1[DataNode1]
    VC2 --> DN2[DataNode2]
    VC3 --> DN0
    
    DN0 --> Seg1[Segment1]
    DN1 --> Seg2[Segment2]
    DN2 --> Seg3[Segment3]
    DN0 --> Seg4[Segment4]
```

---

## 3. Search 时序图

### 3.1 完整流程

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as Proxy
    participant DQL as DQL Queue
    participant MC as MetaCache
    participant QC as QueryCoord
    participant LB as LoadBalancer
    participant QN1 as QueryNode1
    participant QN2 as QueryNode2
    participant MQ as MessageQueue
    participant OS as Object Storage
    
    C->>P: Search(vectors, topK, filter)
    
    P->>P: 健康检查、鉴权、限流
    
    P->>DQL: searchTask入队(并发)
    
    DQL->>DQL: PreExecute
    DQL->>MC: GetCollectionInfo(db, name)
    MC-->>DQL: CollectionInfo(schema, shardNum)
    
    DQL->>DQL: 解析SearchParams(topK, metricType, nprobe)
    DQL->>DQL: 校验向量维度
    alt 维度不匹配
        DQL-->>P: InvalidVectorDimension
        P-->>C: Status(InvalidVectorDimension)
    end
    
    DQL->>MC: GetShards(db, name)
    alt Cache Miss
        MC->>QC: GetShardLeaders(collectionID)
        QC-->>MC: ShardLeaders
        MC->>MC: 更新缓存
    end
    MC-->>DQL: ShardLeaders
    Note over DQL: [{channel:"vchan_0", leader:QN1, nodes:[QN1,QN3]}, ...]
    
    DQL->>DQL: Execute
    
    par 并发查询所有Shard
        DQL->>LB: SelectNode(shard0.nodes)
        LB->>LB: 负载均衡算法(LookAside/RoundRobin)
        LB-->>DQL: NodeID=QN1
        
        DQL->>QN1: Search(vectors, shard0)
        
        QN1->>QN1: 查询内存Segment（Growing）
        QN1->>MQ: 订阅DML Channel
        MQ-->>QN1: 最新InsertMsg/DeleteMsg
        QN1->>QN1: 实时数据过滤
        
        QN1->>QN1: 查询磁盘Segment（Historical）
        QN1->>OS: 读取Segment元数据
        OS-->>QN1: SegmentInfo
        QN1->>OS: 读取索引文件(HNSW/IVF)
        OS-->>QN1: IndexData
        QN1->>QN1: 加载到内存(带LRU缓存)
        
        QN1->>QN1: 向量检索(HNSW.Search/IVF.Search)
        QN1->>QN1: 标量过滤(expr:"age>18 and city=='BJ'")
        QN1->>QN1: 局部TopK排序
        QN1-->>DQL: SearchResult(topK, scores, IDs)
        
    and
        DQL->>LB: SelectNode(shard1.nodes)
        LB-->>DQL: NodeID=QN2
        
        DQL->>QN2: Search(vectors, shard1)
        QN2->>QN2: 同QN1处理流程
        QN2-->>DQL: SearchResult(topK, scores, IDs)
    end
    
    DQL->>DQL: 归并结果(全局TopK)
    Note over DQL: 使用最小堆合并各Shard的TopK
    
    DQL->>DQL: 构造SearchResults
    Note over DQL: 包含IDs、Scores、OutputFields
    
    DQL->>DQL: PostExecute
    DQL-->>P: 任务完成
    
    P-->>C: SearchResults
```

### 3.2 时序图关键点说明

**步骤1-5：请求接收与前置检查**
- 同Insert流程

**步骤6-11：元数据查询**
- 获取Collection Schema（校验向量维度）
- 解析搜索参数（topK、metric_type、nprobe等）

**步骤12-19：Shard信息查询**
- 从MetaCache获取Shard领导者信息
- Cache Miss时查询QueryCoord
- ShardLeaders包含：
  - ChannelName：DML Channel名称
  - LeaderID：主节点ID
  - NodeIDs：所有副本节点ID列表

**步骤20-46：并发查询**
- Proxy并发向所有Shard发起查询
- 每个Shard独立执行：
  1. **负载均衡**：从副本节点中选择一个
  2. **Growing查询**：查询内存中的增量数据
  3. **Historical查询**：查询已刷新的磁盘数据
  4. **向量检索**：使用索引（HNSW/IVF）加速
  5. **标量过滤**：应用过滤表达式
  6. **局部TopK**：返回该Shard的TopK结果

**步骤47-50：结果归并**
- 使用最小堆（Min Heap）归并各Shard结果
- 时间复杂度：O(K * log(M))，K=TopK，M=Shard数*TopK
- 保证全局TopK准确性

**步骤51-55：返回结果**
- 构造SearchResults
- 包含：主键ID、距离分数、输出字段数据

### 3.3 负载均衡决策

```mermaid
flowchart TD
    Start[收到Search请求] --> GetShards[获取Shard信息]
    GetShards --> CheckReplicas{副本数量}
    
    CheckReplicas -->|单副本| DirectCall[直接调用唯一节点]
    CheckReplicas -->|多副本| LBStrategy{负载均衡策略}
    
    LBStrategy -->|RoundRobin| RR[轮询选择节点]
    LBStrategy -->|LookAside| LA[查询节点负载]
    
    LA --> GetMetrics[获取节点Metrics]
    GetMetrics --> CalcScore[计算负载分数<br/>CPU*0.5+Mem*0.3+Queue*0.2]
    CalcScore --> SelectMin[选择分数最低节点]
    
    RR --> CallNode[调用QueryNode]
    DirectCall --> CallNode
    SelectMin --> CallNode
    
    CallNode --> CheckResult{调用是否成功}
    CheckResult -->|成功| ReturnResult[返回结果]
    CheckResult -->|失败| Retry{重试次数<3}
    Retry -->|是| LBStrategy
    Retry -->|否| ReturnError[返回错误]
```

### 3.4 查询并发度

```mermaid
gantt
    title Search并发执行时间线
    dateFormat SSS
    axisFormat %L ms
    
    section Proxy
    接收请求           :done, p1, 000, 2ms
    解析参数           :done, p2, after p1, 3ms
    任务入队           :done, p3, after p2, 1ms
    
    section Shard0
    负载均衡          :active, s01, 006, 2ms
    RPC调用           :active, s02, after s01, 5ms
    向量检索          :active, s03, after s02, 50ms
    返回结果          :active, s04, after s03, 2ms
    
    section Shard1
    负载均衡          :active, s11, 006, 2ms
    RPC调用           :active, s12, after s11, 5ms
    向量检索          :active, s13, after s12, 45ms
    返回结果          :active, s14, after s13, 2ms
    
    section Shard2
    负载均衡          :active, s21, 006, 2ms
    RPC调用           :active, s22, after s21, 5ms
    向量检索          :active, s23, after s22, 55ms
    返回结果          :active, s24, after s23, 2ms
    
    section Proxy
    结果归并          :crit, p4, 070, 5ms
    返回客户端        :crit, p5, after p4, 2ms
```

**说明**：
- 并发查询3个Shard，总耗时取决于最慢的Shard（约65ms）
- 串行查询总耗时 = 50+45+55 = 150ms
- 并发带来2-3倍性能提升

---

## 4. Delete 时序图

### 4.1 完整流程

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as Proxy
    participant DML as DML Queue
    participant MC as MetaCache
    participant Parser as ExprParser
    participant CH as ChannelMgr
    participant MQ as MessageQueue
    participant DN as DataNode
    
    C->>P: Delete(expr="id in [1,2,3]")
    
    P->>DML: deleteTask入队
    
    DML->>DML: PreExecute
    DML->>MC: GetCollectionInfo(db, name)
    MC-->>DML: CollectionInfo
    
    DML->>Parser: ParseExpression(expr)
    Note over Parser: 解析删除条件，提取主键列表
    Parser-->>DML: PrimaryKeys=[1, 2, 3]
    
    DML->>DML: Execute
    
    DML->>DML: hashPrimaryKeys(PrimaryKeys)
    DML->>DML: groupByHash(PrimaryKeys, ShardNum)
    Note over DML: 按哈希分配到不同Shard
    
    loop 每个Shard
        DML->>DML: 构造DeleteMsg
        Note over DML: 包含CollectionID、PartitionID、PrimaryKeys、Timestamp
        
        DML->>CH: getOrCreateDMLStream(collectionID)
        CH-->>DML: MsgStream
        
        DML->>MQ: Produce(DeleteMsg)
        MQ-->>DML: MessageID
    end
    
    par 异步消费（DataNode）
        MQ->>DN: 订阅DeleteMsg
        DN->>DN: 消费DeleteMsg
        DN->>DN: 更新Bloom Filter（标记删除）
        DN->>DN: 写入Delete Buffer
        
        alt Growing Segment
            DN->>DN: 直接从内存删除
        else Historical Segment
            DN->>DN: 记录Delete Delta
            DN->>DN: 等待Compaction物理删除
        end
        
        DN->>DN: 判断是否触发Compaction
        alt Delete比例>20%
            DN->>DN: 触发MixCompaction
            Note over DN: 合并Delete Delta，释放空间
        end
    end
    
    DML->>DML: PostExecute
    DML->>DML: 构造MutationResult(DeleteCnt, Timestamp)
    DML-->>P: 任务完成
    
    P-->>C: MutationResult
```

### 4.2 删除机制说明

**逻辑删除**：
- Delete操作仅标记删除，不立即物理删除
- 使用Bloom Filter快速判断数据是否被删除
- 查询时过滤已删除数据

**物理删除**：
- 由Compaction任务执行
- 合并Delete Delta到Segment
- 释放存储空间

**删除流程**：
```
逻辑删除（立即） → 查询过滤（实时） → 物理删除（异步）
```

---

## 5. Query 时序图

### 5.1 完整流程

Query与Search类似，但有以下区别：

| 特性 | Search | Query |
|------|--------|-------|
| **查询方式** | 向量近邻搜索 | 标量过滤查询 |
| **输入** | 向量+过滤条件 | 仅过滤条件（expr） |
| **输出** | TopK结果 | 所有匹配结果 |
| **性能** | 取决于索引 | 取决于过滤条件选择性 |

```mermaid
sequenceDiagram
    participant C as Client
    participant P as Proxy
    participant DQL as DQL Queue
    participant MC as MetaCache
    participant QN as QueryNode
    
    C->>P: Query(expr="age>18 and city=='BJ'")
    
    P->>DQL: queryTask入队
    
    DQL->>MC: GetCollectionInfo & GetShards
    
    par 并发查询
        DQL->>QN: Query(expr, shard0)
        QN->>QN: 标量索引扫描
        QN->>QN: 过滤数据
        QN-->>DQL: QueryResult
    and
        DQL->>QN: Query(expr, shard1)
        QN-->>DQL: QueryResult
    end
    
    DQL->>DQL: 合并结果（无需TopK排序）
    DQL-->>P: 任务完成
    P-->>C: QueryResults
```

---

## 6. 总结与性能分析

### 6.1 API延迟对比

| API | P50 | P95 | P99 | 瓶颈 |
|-----|-----|-----|-----|------|
| **CreateCollection** | 50ms | 100ms | 200ms | etcd写入 |
| **Insert** | 5ms | 20ms | 50ms | Message Queue |
| **Search** | 20ms | 80ms | 150ms | 向量检索 |
| **Query** | 10ms | 40ms | 100ms | 标量索引 |
| **Delete** | 5ms | 15ms | 30ms | Message Queue |

### 6.2 并发度

| 队列类型 | 并发度 | 说明 |
|---------|--------|------|
| **DDL Queue** | 串行（1） | 保证元数据一致性 |
| **DML Queue** | 并发（可配置） | 默认64并发 |
| **DQL Queue** | 并发（可配置） | 默认64并发 |

### 6.3 优化建议

**Insert优化**：
- 批量插入（1000-5000行/批）
- 预计算HashKeys
- 使用流式API（FixedWidthInsert）

**Search优化**：
- 合理设置TopK（≤1000）
- 使用标量过滤缩小范围
- 批量查询（≤10个向量/批）

**Delete优化**：
- 批量删除（≤1000个ID/批）
- 定期触发Compaction释放空间

---

**相关文档**：
- [Milvus-01-Proxy-概览.md](./Milvus-01-Proxy-概览.md)
- [Milvus-01-Proxy-API.md](./Milvus-01-Proxy-API.md)
- [Milvus-01-Proxy-数据结构.md](./Milvus-01-Proxy-数据结构.md)

