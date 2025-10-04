# LangGraph-01-checkpoint-概览

## 一、模块职责

checkpoint模块是LangGraph持久化系统的基础抽象层，定义了检查点保存、恢复和查询的标准接口。该模块提供以下核心能力：

1. **检查点抽象**：定义`Checkpoint`数据结构，表示图在特定时刻的完整状态快照
2. **存储接口**：通过`BaseCheckpointSaver`定义统一的持久化接口
3. **序列化协议**：提供`SerializerProtocol`和`JsonPlusSerializer`实现状态的序列化与反序列化
4. **内存实现**：提供`InMemorySaver`作为开发和测试的默认实现
5. **存储服务**：提供`BaseStore`接口用于跨线程的长期数据存储
6. **缓存机制**：提供`BaseCache`接口用于节点级别的计算缓存

## 二、输入与输出

### 2.1 输入

**检查点保存（put操作）**
- `RunnableConfig`：包含thread_id、checkpoint_ns等配置
- `Checkpoint`：待保存的检查点数据
- `CheckpointMetadata`：检查点元数据（来源、步数等）
- `ChannelVersions`：新生成的通道版本号

**检查点检索（get操作）**
- `RunnableConfig`：指定要检索的thread_id和可选的checkpoint_id
- `filter`：元数据过滤条件（可选）
- `before`：时间范围限制（可选）
- `limit`：返回结果数量限制（可选）

**待处理写入（put_writes操作）**
- `RunnableConfig`：配置信息
- `writes`：待保存的写入操作列表
- `task_id`：任务标识符
- `task_path`：任务路径

### 2.2 输出

**检查点保存**
- 返回更新后的`RunnableConfig`，包含新保存的checkpoint_id

**检查点检索**
- `CheckpointTuple`：包含checkpoint、metadata、config和pending_writes的完整元组
- 如果不存在则返回None

**检查点列表**
- `Iterator[CheckpointTuple]`：满足条件的检查点元组迭代器，按时间倒序排列

### 2.3 上下游依赖

**上游（依赖方）**
- `langgraph.pregel`：执行引擎，每个超级步调用checkpoint保存状态
- `langgraph.graph.state`：StateGraph编译时注入检查点存储器
- `prebuilt.chat_agent_executor`：ReAct Agent使用检查点实现记忆

**下游（被依赖）**
- 无直接依赖，作为最底层模块

**外部依赖**
- `langchain_core.runnables`：使用RunnableConfig类型
- `ormsgpack`：高效的二进制序列化库
- `langchain_core.load.serializable`：LangChain对象序列化

## 三、模块架构

### 3.1 架构图

```mermaid
flowchart TB
    subgraph "接口层"
        A1[BaseCheckpointSaver<br/>检查点存储基类]
        A2[SerializerProtocol<br/>序列化协议]
        A3[BaseStore<br/>存储接口]
        A4[BaseCache<br/>缓存接口]
    end
    
    subgraph "序列化层"
        B1[JsonPlusSerializer<br/>扩展JSON序列化器]
        B2[EncryptedSerializer<br/>加密序列化器]
        B3[SerDe Types<br/>序列化类型]
    end
    
    subgraph "实现层"
        C1[InMemorySaver<br/>内存存储]
        C2[InMemoryStore<br/>内存存储服务]
        C3[InMemoryCache<br/>内存缓存]
    end
    
    subgraph "数据结构"
        D1[Checkpoint<br/>检查点快照]
        D2[CheckpointMetadata<br/>元数据]
        D3[CheckpointTuple<br/>完整元组]
        D4[Item<br/>存储项]
    end
    
    subgraph "工具模块"
        E1[UUID生成<br/>checkpoint ID]
        E2[版本管理<br/>ChannelVersions]
        E3[批处理<br/>Batch Operations]
    end
    
    A1 -.implements.-> C1
    A2 -.implements.-> B1
    A2 -.implements.-> B2
    A3 -.implements.-> C2
    A4 -.implements.-> C3
    
    A1 --> A2
    A1 --> D1
    A1 --> D2
    A1 --> D3
    A1 --> E1
    A1 --> E2
    
    A3 --> D4
    A3 --> E3
    
    B1 --> B3
    B2 --> B1
    
    C1 --> B1
    C2 --> B1
    
    style A1 fill:#ffe1e1
    style B1 fill:#fff4e1
    style C1 fill:#e1f5ff
    style D1 fill:#e1ffe1
```

### 3.2 架构说明

#### 3.2.1 图意概述

checkpoint模块采用分层设计，从下至上分别是数据结构层、接口层、序列化层和实现层。核心设计原则是面向接口编程，通过抽象基类定义契约，具体实现可灵活替换。

#### 3.2.2 关键接口

**BaseCheckpointSaver**

定义检查点存储的核心接口，包括：
- `get_tuple(config)`: 检索单个检查点
- `list(config, filter, before, limit)`: 列出检查点历史
- `put(config, checkpoint, metadata, new_versions)`: 保存检查点
- `put_writes(config, writes, task_id)`: 保存待处理写入
- `delete_thread(thread_id)`: 删除线程的所有检查点

每个方法都提供同步和异步版本（带a前缀）。

**SerializerProtocol**

定义序列化和反序列化的协议：
- `dumps(obj)`: 序列化为bytes
- `dumps_typed(obj)`: 序列化并保留类型信息，返回(type_name, bytes)
- `loads(data)`: 从bytes反序列化
- `loads_typed(data)`: 从(type_name, bytes)反序列化并还原类型

**BaseStore**

定义跨线程的长期存储接口：
- `get(namespace, key)`: 获取单个项
- `put(namespace, key, value)`: 存储单个项
- `search(namespace_prefix)`: 搜索匹配的项
- `delete(namespace, key)`: 删除单个项
- `batch(operations)`: 批量操作

支持命名空间（namespace）和分层键（hierarchical key），适合存储用户偏好、知识库等长期数据。

#### 3.2.3 边界与约束

**并发约束**
- 每个thread_id的检查点读写应当串行化，避免冲突
- 不同thread_id之间可以完全并行
- InMemorySaver使用内存锁，数据库实现使用事务

**幂等性约束**
- `put`操作：相同checkpoint_id的重复写入应当幂等
- `put_writes`操作：通过(task_id, write_idx)作为唯一键，重复写入会被忽略

**版本约束**
- checkpoint_id必须单调递增（通过UUID v6保证）
- channel版本号必须单调递增（通过字符串比较）
- 不支持回滚到旧版本，只能fork新分支

**大小限制**
- InMemorySaver无大小限制，但受进程内存限制
- 数据库实现受BLOB字段大小限制（通常几MB到几GB）
- 建议单个检查点大小控制在10MB以内

#### 3.2.4 异常处理与回退

**序列化失败**
- 捕获序列化异常，包装为SerializationError
- 提供fallback到pickle（如果启用pickle_fallback）
- 记录详细的对象类型和路径信息

**存储失败**
- 数据库连接失败：抛出ConnectionError，由上层重试
- 磁盘空间不足：抛出IOError，由上层处理
- 版本冲突：抛出ValueError，表示并发写入冲突

**数据损坏**
- 反序列化失败时抛出异常，不返回损坏数据
- 提供checksum校验（如果存储后端支持）
- 建议定期备份关键检查点

#### 3.2.5 性能与容量

**InMemorySaver性能**
- 写入：O(1)，纯内存操作
- 读取：O(1)，基于dict查找
- 列表：O(N)，N为检查点数量
- 内存占用：所有检查点常驻内存

**序列化性能**
- JsonPlusSerializer使用ormsgpack，比标准JSON快10倍以上
- 支持增量序列化（仅序列化变化的通道）
- 典型检查点（10个通道，每个1KB）序列化耗时<1ms

**可扩展性**
- 单个线程支持数千个检查点
- 支持数十万并发线程（取决于后端）
- 通过分片和索引提升大规模场景性能

#### 3.2.6 版本兼容与演进

**检查点格式版本**
- 当前格式版本：v1
- 通过`Checkpoint.v`字段标识
- 未来版本变更会提供迁移工具

**序列化版本**
- 使用类型标签(type_name)标识序列化格式
- 支持多版本共存（通过type_name路由）
- 向后兼容：新版本可读取旧格式

**API演进**
- 接口方法保持稳定，新增方法向后兼容
- 废弃方法通过`@deprecated`装饰器标记
- 主要版本升级提供迁移指南

## 四、生命周期

### 4.1 初始化阶段

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

**步骤**：
1. 实例化存储后端（InMemorySaver、PostgresSaver等）
2. 初始化序列化器（默认JsonPlusSerializer）
3. 建立连接（数据库后端）或初始化数据结构（内存后端）

### 4.2 运行阶段

**检查点保存流程**

```mermaid
sequenceDiagram
    autonumber
    participant Engine as Pregel引擎
    participant Saver as CheckpointSaver
    participant Serde as Serializer
    participant Storage as 存储后端
    
    Engine->>Saver: put(config, checkpoint, metadata, versions)
    Saver->>Saver: 生成checkpoint_id (UUID v6)
    Saver->>Serde: dumps_typed(checkpoint)
    Serde-->>Saver: (type, bytes)
    Saver->>Serde: dumps_typed(metadata)
    Serde-->>Saver: (type, bytes)
    Saver->>Storage: 写入(thread_id, checkpoint_id, data)
    Storage-->>Saver: OK
    Saver-->>Engine: updated_config (with checkpoint_id)
```

**检查点恢复流程**

```mermaid
sequenceDiagram
    autonumber
    participant Engine as Pregel引擎
    participant Saver as CheckpointSaver
    participant Storage as 存储后端
    participant Serde as Serializer
    
    Engine->>Saver: get_tuple(config)
    Saver->>Storage: 查询(thread_id, checkpoint_id)
    Storage-->>Saver: raw_data or None
    alt 找到检查点
        Saver->>Serde: loads_typed(checkpoint_data)
        Serde-->>Saver: checkpoint
        Saver->>Serde: loads_typed(metadata_data)
        Serde-->>Saver: metadata
        Saver->>Storage: 查询pending_writes
        Storage-->>Saver: writes
        Saver-->>Engine: CheckpointTuple
    else 未找到
        Saver-->>Engine: None
    end
```

### 4.3 清理阶段

```python
checkpointer.delete_thread("thread-123")

checkpointer.close()
```

**步骤**：
1. 调用`delete_thread()`删除不再需要的线程数据
2. 关闭存储后端连接（数据库后端）
3. 释放内存资源（内存后端）

## 五、核心算法与流程

### 5.1 UUID v6生成算法

```python
import uuid
import time

def uuid6(node: int | None = None, clock_seq: int | None = None) -> uuid.UUID:
    """
    生成UUID v6，包含时间戳信息，保证单调递增
    
    参数：
        node: 节点标识符（MAC地址）
        clock_seq: 时钟序列号
    
    返回：
        UUID对象，格式符合RFC 4122 v6草案
    """
    # 获取当前时间戳（100纳秒单位，自1582-10-15以来）
    nanoseconds = int(time.time() * 1e7) + 0x01b21dd213814000
    
    # 时间戳的高32位
    time_high = (nanoseconds >> 28) & 0xffffffff
    # 时间戳的中16位
    time_mid = (nanoseconds >> 12) & 0xffff
    # 时间戳的低12位 + 版本号(0110)
    time_low = ((nanoseconds & 0xfff) | 0x6000)
    
    # 时钟序列（14位）
    if clock_seq is None:
        import random
        clock_seq = random.getrandbits(14)
    clock_seq_low = clock_seq & 0xff
    clock_seq_high = ((clock_seq >> 8) & 0x3f) | 0x80  # variant位
    
    # 节点标识（48位）
    if node is None:
        node = uuid.getnode()
    
    # 组装UUID
    return uuid.UUID(fields=(
        time_high, time_mid, time_low,
        clock_seq_high, clock_seq_low, node
    ))
```

**算法说明**：
- UUID v6将时间戳放在最高位，自然保证了字典序即时间序
- 时间精度为100纳秒，同一毫秒内可生成10000个不同ID
- 包含节点信息，可用于分布式环境下去重

**复杂度**：
- 时间复杂度：O(1)
- 空间复杂度：O(1)
- 生成速度：每秒数百万个

### 5.2 检查点过滤算法

```python
def list(
    self,
    config: RunnableConfig | None,
    filter: dict[str, Any] | None = None,
    before: RunnableConfig | None = None,
    limit: int | None = None,
) -> Iterator[CheckpointTuple]:
    """
    列出满足条件的检查点
    
    参数：
        config: 基础配置（thread_id, checkpoint_ns）
        filter: 元数据过滤器，如{"source": "loop", "step": 5}
        before: 只返回此检查点之前的检查点
        limit: 最多返回的数量
    
    返回：
        检查点元组的迭代器，按时间倒序
    """
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
    before_id = get_checkpoint_id(before) if before else None
    
    # 获取所有检查点ID，按时间倒序
    checkpoint_ids = sorted(
        self.storage[thread_id][checkpoint_ns].keys(),
        reverse=True
    )
    
    count = 0
    for checkpoint_id in checkpoint_ids:
        # 应用before过滤
        if before_id and checkpoint_id >= before_id:
            continue
        
        # 加载检查点
        checkpoint, metadata_bytes, parent_id = \
            self.storage[thread_id][checkpoint_ns][checkpoint_id]
        metadata = self.serde.loads_typed(metadata_bytes)
        
        # 应用元数据过滤
        if filter:
            if not all(metadata.get(k) == v for k, v in filter.items()):
                continue
        
        # 检查limit
        if limit is not None and count >= limit:
            break
        count += 1
        
        yield CheckpointTuple(
            config={"configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }},
            checkpoint=self.serde.loads_typed(checkpoint),
            metadata=metadata,
            parent_config=make_parent_config(parent_id) if parent_id else None,
            pending_writes=self._load_pending_writes(
                thread_id, checkpoint_ns, checkpoint_id
            ),
        )
```

**算法说明**：
- 使用排序后的checkpoint_id列表（已按时间排序）
- 顺序扫描并应用各种过滤条件
- 懒加载检查点数据，只在通过初步过滤后才反序列化

**复杂度**：
- 时间复杂度：O(N)，N为检查点总数
- 空间复杂度：O(1)，流式返回
- 优化：可通过索引加速元数据过滤

### 5.3 通道版本比较算法

```python
def should_execute_node(
    node: str,
    versions_seen: dict[str, ChannelVersions],
    current_versions: ChannelVersions,
    triggers: list[str],
) -> bool:
    """
    判断节点是否应当执行
    
    参数：
        node: 节点名称
        versions_seen: 节点已见过的通道版本
        current_versions: 当前的通道版本
        triggers: 节点的触发器通道列表
    
    返回：
        True表示应当执行，False表示跳过
    """
    node_seen = versions_seen.get(node, {})
    
    for trigger_channel in triggers:
        current_version = current_versions.get(trigger_channel)
        seen_version = node_seen.get(trigger_channel)
        
        # 如果通道有新版本，且节点未见过，则执行
        if current_version is not None:
            if seen_version is None:
                return True
            if compare_versions(current_version, seen_version) > 0:
                return True
    
    return False

def compare_versions(v1: str | int | float, v2: str | int | float) -> int:
    """
    比较两个版本号
    
    返回：
        > 0: v1 > v2
        = 0: v1 == v2
        < 0: v1 < v2
    """
    # 字符串版本格式："<sequence>.<hash>"
    if isinstance(v1, str) and isinstance(v2, str):
        seq1 = int(v1.split(".")[0])
        seq2 = int(v2.split(".")[0])
        return seq1 - seq2
    # 数值版本直接比较
    elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return 1 if v1 > v2 else (-1 if v1 < v2 else 0)
    # 类型不匹配，转换为字符串比较
    else:
        return 1 if str(v1) > str(v2) else (-1 if str(v1) < str(v2) else 0)
```

**算法说明**：
- 版本号格式：`<sequence>.<random_hash>`，如"00000032.0.7234819283748"`
- sequence部分保证单调递增
- random_hash部分用于打破并发时的平局
- 通过比较sequence部分即可确定新旧关系

**复杂度**：
- 时间复杂度：O(T)，T为触发器数量（通常<10）
- 空间复杂度：O(1)

## 六、关键代码片段

### 6.1 InMemorySaver.put实现

```python
def put(
    self,
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    new_versions: ChannelVersions,
) -> RunnableConfig:
    """保存检查点到内存存储"""
    # 复制检查点，避免修改原始数据
    c = checkpoint.copy()
    
    # 提取配置信息
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
    
    # 分离通道值，单独存储（支持大对象）
    values: dict[str, Any] = c.pop("channel_values")
    
    # 将通道值序列化为blobs
    for k, v in new_versions.items():
        self.blobs[(thread_id, checkpoint_ns, k, v)] = (
            self.serde.dumps_typed(values[k]) if k in values 
            else ("empty", b"")
        )
    
    # 存储检查点元数据
    self.storage[thread_id][checkpoint_ns].update({
        checkpoint["id"]: (
            self.serde.dumps_typed(c),  # 检查点（不含通道值）
            self.serde.dumps_typed(
                get_checkpoint_metadata(config, metadata)
            ),  # 元数据
            config["configurable"].get("checkpoint_id"),  # 父检查点ID
        )
    })
    
    # 返回包含新checkpoint_id的配置
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
        }
    }
```

**代码说明**：
- 通道值单独存储为blobs，支持大对象和按需加载
- 使用`dumps_typed`保留类型信息，确保正确反序列化
- 存储父检查点ID，支持检查点链和分支
- 操作是内存级别的，极快但不持久

### 6.2 JsonPlusSerializer序列化

```python
def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
    """序列化对象并保留类型信息"""
    if isinstance(obj, (bytes, bytearray)):
        # bytes直接返回，无需编码
        return "bytes", bytes(obj)
    
    elif isinstance(obj, Serializable):
        # LangChain对象使用其自带的序列化方法
        return "json", ormsgpack.packb(
            obj.to_json(),
            option=ormsgpack.OPT_SERIALIZE_NUMPY | ormsgpack.OPT_NON_STR_KEYS,
            default=self._default,
        )
    
    else:
        # 其他对象使用ormsgpack
        try:
            return "msgpack", ormsgpack.packb(
                obj,
                option=ormsgpack.OPT_SERIALIZE_NUMPY | ormsgpack.OPT_NON_STR_KEYS,
                default=self._default,
            )
        except (TypeError, ormsgpack.EncodeError) as e:
            # 如果启用了pickle fallback
            if self.pickle_fallback:
                import pickle
                return "pickle", pickle.dumps(obj)
            raise

def _default(self, obj: Any) -> dict[str, Any]:
    """为不可直接序列化的对象提供编码方案"""
    # Pydantic模型
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return self._encode_constructor_args(
            obj.__class__,
            method=(None, "model_construct"),
            kwargs=obj.model_dump()
        )
    
    # datetime对象
    elif isinstance(obj, datetime):
        return self._encode_constructor_args(
            datetime,
            method="fromisoformat",
            args=(obj.isoformat(),)
        )
    
    # UUID对象
    elif isinstance(obj, UUID):
        return self._encode_constructor_args(
            UUID,
            args=(obj.hex,)
        )
    
    # ... 更多类型处理
    
    else:
        raise TypeError(f"不支持序列化类型: {type(obj)}")

def _encode_constructor_args(
    self,
    constructor: Callable,
    method: str | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
) -> dict[str, Any]:
    """编码构造器调用为JSON-safe的字典"""
    return {
        "lc": 2,  # LangChain序列化协议版本
        "type": "constructor",
        "id": (*constructor.__module__.split("."), constructor.__name__),
        "method": method,
        "args": args,
        "kwargs": kwargs,
    }
```

**代码说明**：
- 使用类型标签区分不同序列化格式（bytes、json、msgpack、pickle）
- ormsgpack比标准JSON快约10倍，且支持更多类型
- `_default`提供自定义类型的序列化方案
- 编码为构造器调用形式，反序列化时重新构造对象

## 七、最佳实践

### 7.1 检查点存储选择

**开发和测试环境**
```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```
- 优点：零配置，速度快
- 缺点：不持久，进程重启后丢失
- 适用：单元测试、演示、原型开发

**生产环境（单机）**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```
- 优点：文件持久化，轻量级
- 缺点：不支持真正的并发写入
- 适用：单机部署、个人项目

**生产环境（分布式）**
```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)
```
- 优点：支持并发，支持分布式
- 缺点：需要额外的数据库服务
- 适用：多实例部署、企业应用

### 7.2 检查点清理策略

**按时间清理**
```python
from datetime import datetime, timedelta

def cleanup_old_checkpoints(checkpointer, days=30):
    """删除30天前的检查点"""
    cutoff = datetime.now() - timedelta(days=days)
    
    for thread_tuple in checkpointer.list(None):
        metadata = thread_tuple.metadata
        if "ts" in metadata:
            ts = datetime.fromisoformat(metadata["ts"])
            if ts < cutoff:
                thread_id = thread_tuple.config["configurable"]["thread_id"]
                checkpointer.delete_thread(thread_id)
```

**按数量清理**
```python
def keep_latest_n_checkpoints(checkpointer, thread_id, n=10):
    """只保留最新的N个检查点"""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = list(checkpointer.list(config))
    
    if len(checkpoints) > n:
        # checkpoints已按时间倒序排列
        for checkpoint_tuple in checkpoints[n:]:
            checkpoint_id = checkpoint_tuple.config["configurable"]["checkpoint_id"]
            # 注意：BaseCheckpointSaver没有删除单个检查点的方法
            # 需要自行实现或使用自定义的Saver
```

### 7.3 大状态处理

**使用引用而非直接存储**
```python
from typing import TypedDict

class State(TypedDict):
    messages: list
    large_data_ref: str  # 存储对象存储的URL或ID

def process_node(state: State) -> dict:
    # 从对象存储加载大数据
    large_data = s3_client.get_object(state["large_data_ref"])
    
    # 处理...
    result = process(large_data)
    
    # 将结果上传到对象存储
    new_ref = s3_client.put_object(result)
    
    return {"large_data_ref": new_ref}
```

**分离热数据和冷数据**
```python
class State(TypedDict):
    # 热数据：频繁访问，存储在检查点中
    current_step: int
    recent_messages: list  # 只保留最近10条
    
    # 冷数据：不常访问，存储在Store中
    full_history_key: str  # 指向Store的键

def node_with_history(state: State, runtime: Runtime) -> dict:
    # 需要时从Store加载完整历史
    if runtime.store:
        full_history = runtime.store.get(
            ("history",),
            state["full_history_key"]
        )
```

### 7.4 检查点分支与合并

**创建检查点分支**
```python
def fork_checkpoint(app, thread_id, checkpoint_id, new_thread_id):
    """从指定检查点创建新分支"""
    # 加载源检查点
    source_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }
    source_tuple = app.checkpointer.get_tuple(source_config)
    
    # 创建新线程，使用源检查点的状态
    new_config = {
        "configurable": {
            "thread_id": new_thread_id,
            "checkpoint_ns": "",
        }
    }
    app.checkpointer.put(
        new_config,
        source_tuple.checkpoint,
        {**source_tuple.metadata, "source": "fork"},
        source_tuple.checkpoint["channel_versions"],
    )
    
    return new_thread_id
```

**合并多个分支（手动）**
```python
def merge_branches(app, thread_ids, strategy="latest"):
    """合并多个分支的状态"""
    # 加载所有分支的最新检查点
    checkpoints = []
    for thread_id in thread_ids:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = app.checkpointer.get_tuple(config)
        if checkpoint_tuple:
            checkpoints.append(checkpoint_tuple)
    
    if strategy == "latest":
        # 选择最新的检查点
        latest = max(checkpoints, key=lambda cp: cp.checkpoint["id"])
        return latest.config["configurable"]["thread_id"]
    
    elif strategy == "manual":
        # 手动合并状态（需要业务逻辑）
        merged_state = manual_merge_logic(checkpoints)
        new_thread_id = f"merged-{uuid.uuid4().hex[:8]}"
        # 保存合并后的状态...
        return new_thread_id
```

## 八、示例场景

### 8.1 检查点恢复

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

# 创建图并配置检查点
graph = StateGraph(State)
# ... 添加节点和边 ...
checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 第一次运行，到一半失败
config = {"configurable": {"thread_id": "user-123"}}
try:
    result = app.invoke({"input": "hello"}, config)
except Exception as e:
    print(f"执行失败: {e}")

# 检查当前状态
snapshot = app.get_state(config)
print(f"当前步数: {snapshot.next}")
print(f"当前值: {snapshot.values}")

# 从中断点继续执行
result = app.invoke(None, config)
print(f"执行完成: {result}")
```

### 8.2 时间旅行与调试

```python
# 列出所有历史检查点
config = {"configurable": {"thread_id": "user-123"}}
for checkpoint_tuple in app.checkpointer.list(config):
    print(f"Step {checkpoint_tuple.metadata['step']}: "
          f"{checkpoint_tuple.checkpoint['id']}")

# 回到特定检查点
historical_config = {
    "configurable": {
        "thread_id": "user-123",
        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
    }
}
historical_snapshot = app.get_state(historical_config)
print(f"历史状态: {historical_snapshot.values}")

# 从历史检查点继续执行（创建新分支）
new_result = app.invoke({"correction": "fix something"}, historical_config)
```

### 8.3 多会话管理

```python
# 用户A的会话
config_a = {"configurable": {"thread_id": "user-a-session-1"}}
app.invoke({"messages": ["Hi"]}, config_a)

# 用户B的会话
config_b = {"configurable": {"thread_id": "user-b-session-1"}}
app.invoke({"messages": ["Hello"]}, config_b)

# 列出用户A的所有会话
sessions_a = list(app.checkpointer.list(config_a))
print(f"用户A有{len(sessions_a)}个检查点")

# 删除用户B的会话
app.checkpointer.delete_thread("user-b-session-1")
```

通过checkpoint模块的灵活设计，LangGraph实现了强大的状态持久化能力，为构建可靠的长时间运行Agent系统奠定了基础。

