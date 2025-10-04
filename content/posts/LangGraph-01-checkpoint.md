---
title: "LangGraph-01-checkpoint"
date: 2025-10-04T21:26:31+08:00
draft: false
tags:
  - LangGraph
  - 架构设计
  - 概览
  - 源码分析
categories:
  - LangGraph
  - AI框架
  - Python
series: "langgraph-source-analysis"
description: "LangGraph 源码剖析 - 01-checkpoint"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true

---

# LangGraph-01-checkpoint

## 模块概览

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

---

## API接口

## 一、API概览

checkpoint模块对外提供以下核心API：

| API类/接口 | 类型 | 功能描述 |
|-----------|------|---------|
| BaseCheckpointSaver | 抽象基类 | 定义检查点保存和恢复的标准接口 |
| InMemorySaver | 实现类 | 基于内存的检查点存储实现 |
| JsonPlusSerializer | 工具类 | 扩展的JSON序列化器 |
| BaseStore | 抽象基类 | 跨线程的长期存储接口 |
| BaseCache | 抽象基类 | 节点级别的缓存接口 |

## 二、BaseCheckpointSaver API

### 2.1 get_tuple

#### 基本信息

- **方法名称**: `get_tuple`
- **协议**: 同步方法
- **幂等性**: 是（多次调用返回相同结果）

#### 请求结构体

```python
def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
    pass
```

**RunnableConfig结构**：

```python
class RunnableConfig(TypedDict, total=False):
    """
    运行配置，用于指定要检索的检查点
    """
    configurable: Required[dict[str, Any]]
```

| 字段 | 类型 | 必填 | 默认值 | 约束 | 说明 |
|------|------|------|--------|------|------|
| configurable.thread_id | str | 是 | 无 | 非空字符串 | 线程标识符，用于隔离不同会话的检查点 |
| configurable.checkpoint_ns | str | 否 | "" | 任意字符串 | 检查点命名空间，用于子图隔离 |
| configurable.checkpoint_id | str | 否 | None | UUID字符串 | 指定检查点ID，不提供则返回最新 |

#### 响应结构体

```python
class CheckpointTuple(NamedTuple):
    """
    完整的检查点元组，包含所有相关信息
    """
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| config | RunnableConfig | 是 | 该检查点的完整配置（含thread_id和checkpoint_id） |
| checkpoint | Checkpoint | 是 | 检查点数据，包含通道值和版本信息 |
| metadata | CheckpointMetadata | 是 | 元数据，包含来源(source)、步数(step)等 |
| parent_config | RunnableConfig \| None | 否 | 父检查点的配置，用于追溯历史 |
| pending_writes | list[PendingWrite] \| None | 否 | 待处理的写入操作（节点执行失败时） |

#### 入口函数与核心代码

```python
class BaseCheckpointSaver(Generic[V]):
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """
        根据配置检索检查点元组
        
        功能：

        1. 从config中提取thread_id和checkpoint_id
        2. 调用具体实现的存储后端获取数据
        3. 反序列化检查点和元数据
        4. 组装完整的CheckpointTuple返回
        
        参数：
            config: 包含thread_id和可选checkpoint_id的配置
            
        返回：
            CheckpointTuple或None（未找到时）
        """
        raise NotImplementedError

```

**InMemorySaver实现**：

```python
class InMemorySaver(BaseCheckpointSaver[str]):
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        # 1. 提取配置参数
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        
        # 2. 获取checkpoint_id（如果指定）
        if checkpoint_id := get_checkpoint_id(config):
            # 3. 从存储中查找指定检查点
            if saved := self.storage[thread_id][checkpoint_ns].get(checkpoint_id):
                checkpoint, metadata, parent_checkpoint_id = saved
                writes = self.writes[(thread_id, checkpoint_ns, checkpoint_id)].values()
                
                # 4. 反序列化数据
                checkpoint_: Checkpoint = self.serde.loads_typed(checkpoint)
                
                # 5. 加载通道值（blobs）
                channel_values = self._load_blobs(
                    thread_id, checkpoint_ns, checkpoint_["channel_versions"]
                )
                
                # 6. 组装CheckpointTuple
                return CheckpointTuple(
                    config=config,
                    checkpoint={**checkpoint_, "channel_values": channel_values},
                    metadata=self.serde.loads_typed(metadata),
                    pending_writes=[
                        (id, c, self.serde.loads_typed(v))
                        for id, c, v, _ in writes
                    ],
                    parent_config=self._make_parent_config(
                        thread_id, checkpoint_ns, parent_checkpoint_id
                    ) if parent_checkpoint_id else None,
                )
        else:
            # 7. 未指定checkpoint_id，返回最新的检查点
            if checkpoints := self.storage[thread_id][checkpoint_ns]:
                checkpoint_id = max(checkpoints.keys())  # 最大ID即最新
                # ... 同上逻辑 ...
        
        return None  # 未找到任何检查点
    
    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        """加载通道值"""
        channel_values: dict[str, Any] = {}
        for k, v in versions.items():
            kk = (thread_id, checkpoint_ns, k, v)
            if kk in self.blobs:
                vv = self.blobs[kk]
                if vv[0] != "empty":
                    channel_values[k] = self.serde.loads_typed(vv)
        return channel_values
```

#### 调用链路

```
Pregel._invoke()
  └─> Pregel._prepare_state()
       └─> BaseCheckpointSaver.get_tuple()
            └─> InMemorySaver.get_tuple()
                 ├─> self.storage[thread_id][checkpoint_ns][checkpoint_id]
                 ├─> self.serde.loads_typed(checkpoint)
                 ├─> self._load_blobs()
                 │    └─> self.blobs[(thread_id, checkpoint_ns, k, v)]
                 └─> return CheckpointTuple(...)
```

#### 异常与性能

**异常情况**：

- `KeyError`: thread_id不存在时（返回None而非抛异常）
- `SerializationError`: 反序列化失败时
- `MemoryError`: 内存不足时（极少见）

**性能要点**：

- 时间复杂度：O(1) - 基于dict的直接查找
- 空间复杂度：O(S)，S为检查点大小
- 优化建议：对于大状态，考虑延迟加载通道值

---

### 2.2 list

#### 基本信息

- **方法名称**: `list`
- **协议**: 同步方法，返回迭代器
- **幂等性**: 是

#### 请求结构体

```python
def list(
    self,
    config: RunnableConfig | None,
    *,
    filter: dict[str, Any] | None = None,
    before: RunnableConfig | None = None,
    limit: int | None = None,
) -> Iterator[CheckpointTuple]:
    pass
```

| 字段 | 类型 | 必填 | 默认值 | 约束 | 说明 |
|------|------|------|--------|------|------|
| config | RunnableConfig \| None | 否 | None | - | 基础配置，提供thread_id过滤，None表示所有线程 |
| filter | dict[str, Any] \| None | 否 | None | 键值对 | 元数据过滤器，如{"source": "loop"} |
| before | RunnableConfig \| None | 否 | None | - | 只返回此检查点之前的检查点 |
| limit | int \| None | 否 | None | 正整数 | 最多返回的检查点数量 |

#### 响应结构体

返回`Iterator[CheckpointTuple]`，每个元素为检查点元组，按时间倒序排列。

#### 入口函数与核心代码

```python
class InMemorySaver(BaseCheckpointSaver[str]):
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """
        列出满足条件的检查点
        
        功能：

        1. 确定要扫描的线程范围（单个或所有）
        2. 遍历每个线程的检查点，按ID倒序
        3. 应用before过滤
        4. 应用元数据过滤
        5. 应用limit限制
        6. 懒加载和yield检查点
        """
        # 1. 确定线程范围
        thread_ids = (
            (config["configurable"]["thread_id"],) if config
            else self.storage.keys()
        )
        
        config_checkpoint_ns = (
            config["configurable"].get("checkpoint_ns") if config else None
        )
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        
        # 2. 遍历线程
        for thread_id in thread_ids:
            for checkpoint_ns in self.storage[thread_id].keys():
                # 2.1 命名空间过滤
                if (config_checkpoint_ns is not None
                    and checkpoint_ns != config_checkpoint_ns):
                    continue
                
                # 2.2 遍历检查点（倒序）
                for checkpoint_id, (checkpoint, metadata_b, parent_checkpoint_id) \
                        in sorted(
                            self.storage[thread_id][checkpoint_ns].items(),
                            key=lambda x: x[0],
                            reverse=True,
                        ):
                    # 3. checkpoint_id过滤
                    if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                        continue
                    
                    # 4. before过滤
                    if (before
                        and (before_checkpoint_id := get_checkpoint_id(before))
                        and checkpoint_id >= before_checkpoint_id):
                        continue
                    
                    # 5. 元数据过滤
                    metadata = self.serde.loads_typed(metadata_b)
                    if filter and not all(
                        query_value == metadata.get(query_key)
                        for query_key, query_value in filter.items()
                    ):
                        continue
                    
                    # 6. limit限制
                    if limit is not None and limit <= 0:
                        break
                    elif limit is not None:
                        limit -= 1
                    
                    # 7. 加载并yield
                    writes = self.writes[(thread_id, checkpoint_ns, checkpoint_id)].values()
                    checkpoint_: Checkpoint = self.serde.loads_typed(checkpoint)
                    
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                        checkpoint={
                            **checkpoint_,
                            "channel_values": self._load_blobs(
                                thread_id, checkpoint_ns, checkpoint_["channel_versions"]
                            ),
                        },
                        metadata=metadata,
                        parent_config=self._make_parent_config(
                            thread_id, checkpoint_ns, parent_checkpoint_id
                        ) if parent_checkpoint_id else None,
                        pending_writes=[
                            (id, c, self.serde.loads_typed(v))
                            for id, c, v, _ in writes
                        ],
                    )

```

#### 调用链路

```
用户代码
  └─> app.checkpointer.list(config, filter=..., before=..., limit=...)
       └─> InMemorySaver.list()
            ├─> 遍历self.storage[thread_id][checkpoint_ns]
            ├─> 应用各种过滤条件
            ├─> self.serde.loads_typed()
            └─> yield CheckpointTuple(...)
```

#### 异常与性能

**异常情况**：

- 一般不抛异常，返回空迭代器
- `SerializationError`: 反序列化失败时跳过该检查点

**性能要点**：

- 时间复杂度：O(N)，N为检查点总数
- 空间复杂度：O(1)，流式返回
- 优化：通过索引加速元数据过滤（数据库后端）

---

### 2.3 put

#### 基本信息

- **方法名称**: `put`
- **协议**: 同步方法
- **幂等性**: 是（相同checkpoint_id重复写入结果一致）

#### 请求结构体

```python
def put(
    self,
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    new_versions: ChannelVersions,
) -> RunnableConfig:
    pass
```

**Checkpoint结构**：

```python
class Checkpoint(TypedDict):
    """状态快照"""
    v: int  # 格式版本，当前为1
    id: str  # UUID，唯一且单调递增
    ts: str  # ISO 8601时间戳
    channel_values: dict[str, Any]  # 通道值
    channel_versions: ChannelVersions  # 通道版本
    versions_seen: dict[str, ChannelVersions]  # 节点已见版本
    updated_channels: list[str] | None  # 本次更新的通道
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| v | int | 是 | 检查点格式版本号，当前固定为1 |
| id | str | 是 | UUID v6字符串，保证单调递增 |
| ts | str | 是 | 创建时间戳，ISO 8601格式 |
| channel_values | dict[str, Any] | 是 | 所有通道的当前值 |
| channel_versions | ChannelVersions | 是 | 各通道的版本号 |
| versions_seen | dict[str, ChannelVersions] | 是 | 记录每个节点看到的通道版本 |
| updated_channels | list[str] \| None | 否 | 本次更新的通道列表，用于优化 |

**CheckpointMetadata结构**：

```python
class CheckpointMetadata(TypedDict, total=False):
    """检查点元数据"""
    source: Literal["input", "loop", "update", "fork"]
    step: int
    parents: dict[str, str]
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| source | Literal | 否 | 来源：input(初始)/loop(循环)/update(手动)/fork(分支) |
| step | int | 否 | 步数，从-1（input）开始计数 |
| parents | dict[str, str] | 否 | 父检查点映射，namespace -> checkpoint_id |

#### 响应结构体

返回`RunnableConfig`，包含新保存的checkpoint_id：

```python
{
    "configurable": {
        "thread_id": "thread-123",
        "checkpoint_ns": "",
        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
    }
}
```

#### 入口函数与核心代码

```python
class InMemorySaver(BaseCheckpointSaver[str]):
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        保存检查点到内存
        
        功能：

        1. 复制检查点，避免修改原数据
        2. 分离通道值，单独存储为blobs
        3. 序列化检查点和元数据
        4. 更新存储字典
        5. 返回包含checkpoint_id的配置
        """
        # 1. 复制并提取信息
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        values: dict[str, Any] = c.pop("channel_values")
        
        # 2. 存储通道值为blobs
        for k, v in new_versions.items():
            self.blobs[(thread_id, checkpoint_ns, k, v)] = (
                self.serde.dumps_typed(values[k]) if k in values
                else ("empty", b"")
            )
        
        # 3. 序列化并存储检查点
        self.storage[thread_id][checkpoint_ns].update({
            checkpoint["id"]: (
                self.serde.dumps_typed(c),  # 检查点（不含values）
                self.serde.dumps_typed(
                    get_checkpoint_metadata(config, metadata)
                ),  # 元数据
                config["configurable"].get("checkpoint_id"),  # 父ID
            )
        })
        
        # 4. 返回新配置
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

```

#### 调用链路

```
Pregel._loop()
  └─> Pregel._checkpoint()
       └─> BaseCheckpointSaver.put()
            └─> InMemorySaver.put()
                 ├─> checkpoint.copy()
                 ├─> 遍历new_versions
                 │    └─> self.serde.dumps_typed(values[k])
                 │         └─> self.blobs[(thread_id, ns, k, v)] = data
                 ├─> self.serde.dumps_typed(checkpoint)
                 ├─> self.serde.dumps_typed(metadata)
                 └─> self.storage[thread_id][ns][id] = data
```

#### 异常与性能

**异常情况**：

- `SerializationError`: 序列化失败（对象不可序列化）
- `MemoryError`: 内存不足
- `KeyError`: config缺少必需字段

**性能要点**：

- 时间复杂度：O(C)，C为通道数量
- 空间复杂度：O(S)，S为状态大小
- 优化：大对象使用引用，减少序列化开销

---

### 2.4 put_writes

#### 基本信息

- **方法名称**: `put_writes`
- **协议**: 同步方法
- **幂等性**: 是（相同task_id和write_idx的写入会被忽略）

#### 请求结构体

```python
def put_writes(
    self,
    config: RunnableConfig,
    writes: Sequence[tuple[str, Any]],
    task_id: str,
    task_path: str = "",
) -> None:
    pass
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| config | RunnableConfig | 是 | - | 包含thread_id和checkpoint_id |
| writes | Sequence[tuple[str, Any]] | 是 | - | 写入列表，每项为(channel, value) |
| task_id | str | 是 | - | 任务标识符（通常为节点名） |
| task_path | str | 否 | "" | 任务路径（子图场景） |

#### 响应结构体

无返回值（None）

#### 入口函数与核心代码

```python
class InMemorySaver(BaseCheckpointSaver[str]):
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """
        保存待处理写入
        
        功能：

        1. 提取配置信息
        2. 遍历写入列表
        3. 使用(task_id, write_idx)作为唯一键
        4. 序列化并存储（避免重复）
        
        用途：
        - 节点执行失败时，保存成功节点的写入
        - 恢复时直接应用这些写入，避免重新执行
        """
        # 1. 提取信息
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        outer_key = (thread_id, checkpoint_ns, checkpoint_id)
        
        # 2. 获取已有写入
        outer_writes_ = self.writes.get(outer_key)
        
        # 3. 遍历新写入
        for idx, (c, v) in enumerate(writes):
            # 3.1 构造唯一键
            inner_key = (task_id, WRITES_IDX_MAP.get(c, idx))
            
            # 3.2 检查是否已存在（幂等性）
            if inner_key[1] >= 0 and outer_writes_ and inner_key in outer_writes_:
                continue
            
            # 3.3 序列化并存储
            self.writes[outer_key][inner_key] = (
                task_id,
                c,  # channel
                self.serde.dumps_typed(v),  # value
                task_path,
            )

```

**WRITES_IDX_MAP说明**：

```python
WRITES_IDX_MAP: dict[str, int] = {
    ERROR: -1,  # 错误写入，优先级最高
    INTERRUPT: -2,  # 中断写入
}
```

特殊通道（ERROR、INTERRUPT）使用负索引，确保唯一性和优先级。

#### 调用链路

```
Pregel._loop()
  └─> 捕获节点执行异常
       └─> collect_successful_writes()
            └─> BaseCheckpointSaver.put_writes()
                 └─> InMemorySaver.put_writes()
                      ├─> 遍历writes
                      ├─> self.serde.dumps_typed(value)
                      └─> self.writes[outer_key][inner_key] = data
```

#### 异常与性能

**异常情况**：

- `SerializationError`: 写入值序列化失败
- `KeyError`: config缺少checkpoint_id

**性能要点**：

- 时间复杂度：O(W)，W为写入数量
- 空间复杂度：O(W)
- 幂等性保证：通过(task_id, write_idx)唯一键

---

### 2.5 delete_thread

#### 基本信息

- **方法名称**: `delete_thread`
- **协议**: 同步方法
- **幂等性**: 是

#### 请求结构体

```python
def delete_thread(self, thread_id: str) -> None:
    pass
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| thread_id | str | 是 | 要删除的线程标识符 |

#### 响应结构体

无返回值（None）

#### 入口函数与核心代码

```python
class InMemorySaver(BaseCheckpointSaver[str]):
    def delete_thread(self, thread_id: str) -> None:
        """
        删除线程的所有数据
        
        功能：

        1. 删除storage中的所有检查点
        2. 删除writes中的所有待处理写入
        3. 删除blobs中的所有通道值
        """
        # 1. 删除检查点
        if thread_id in self.storage:
            del self.storage[thread_id]
        
        # 2. 删除待处理写入
        for k in list(self.writes.keys()):
            if k[0] == thread_id:
                del self.writes[k]
        
        # 3. 删除blobs
        for k in list(self.blobs.keys()):
            if k[0] == thread_id:
                del self.blobs[k]

```

#### 调用链路

```
用户代码
  └─> app.checkpointer.delete_thread("thread-123")
       └─> InMemorySaver.delete_thread()
            ├─> del self.storage[thread_id]
            ├─> 遍历self.writes，删除匹配项
            └─> 遍历self.blobs，删除匹配项
```

#### 异常与性能

**异常情况**：

- 一般不抛异常（幂等）
- `MemoryError`: 内存操作失败（极少见）

**性能要点**：

- 时间复杂度：O(W + B)，W为写入数，B为blob数
- 可优化为O(1)：使用单独的字典存储每个thread的数据

---

## 三、异步API

所有同步API都有对应的异步版本，方法名前缀为`a`：

| 同步方法 | 异步方法 | 说明 |
|---------|---------|------|
| get_tuple | aget_tuple | 异步检索检查点 |
| list | alist | 异步列出检查点（AsyncIterator） |
| put | aput | 异步保存检查点 |
| put_writes | aput_writes | 异步保存写入 |
| delete_thread | adelete_thread | 异步删除线程 |

**InMemorySaver的异步实现**（简单包装）：

```python
async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
    """异步版本直接调用同步方法（内存操作无阻塞）"""
    return self.get_tuple(config)

async def alist(
    self,
    config: RunnableConfig | None,
    *,
    filter: dict[str, Any] | None = None,
    before: RunnableConfig | None = None,
    limit: int | None = None,
) -> AsyncIterator[CheckpointTuple]:
    """异步迭代器，yield同步方法的结果"""
    for item in self.list(config, filter=filter, before=before, limit=limit):
        yield item
```

**数据库后端的异步实现**（真正异步）：

```python
# PostgresSaver中的异步方法使用asyncpg
async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
    async with self.conn.cursor() as cur:
        await cur.execute(
            "SELECT checkpoint, metadata, parent_checkpoint_id "
            "FROM checkpoints WHERE thread_id = %s AND checkpoint_id = %s",
            (thread_id, checkpoint_id),
        )
        row = await cur.fetchone()
        # ... 处理结果 ...
```

## 四、总结

checkpoint模块的API设计遵循以下原则：

1. **简洁性**：核心接口只有5个方法，易于理解和实现
2. **一致性**：所有方法都有同步和异步版本
3. **可扩展性**：通过继承BaseCheckpointSaver实现自定义后端
4. **幂等性**：重复调用不会产生副作用
5. **类型安全**：使用TypedDict和NamedTuple提供类型提示

通过这些API，LangGraph实现了强大的状态持久化能力，为构建可靠的Agent系统提供了坚实基础。

---

## 数据结构

## 一、核心数据结构概览

checkpoint模块定义了以下核心数据结构：

| 数据结构 | 类型 | 用途 |
|---------|------|------|
| Checkpoint | TypedDict | 完整的状态快照 |
| CheckpointMetadata | TypedDict | 检查点的元数据 |
| CheckpointTuple | NamedTuple | 检查点的完整元组 |
| ChannelVersions | dict | 通道版本映射 |
| PendingWrite | tuple | 待处理的写入操作 |
| Item | dataclass | Store中的存储项 |

## 二、UML类图

### 2.1 核心数据结构关系图

```mermaid
classDiagram
    class Checkpoint {
        +int v
        +str id
        +str ts
        +dict~str,Any~ channel_values
        +ChannelVersions channel_versions
        +dict~str,ChannelVersions~ versions_seen
        +list~str~ updated_channels
    }
    
    class CheckpointMetadata {
        +Literal source
        +int step
        +dict~str,str~ parents
    }
    
    class CheckpointTuple {
        +RunnableConfig config
        +Checkpoint checkpoint
        +CheckpointMetadata metadata
        +RunnableConfig parent_config
        +list~PendingWrite~ pending_writes
    }
    
    class BaseCheckpointSaver {
        <<abstract>>
        +SerializerProtocol serde
        +get_tuple(config) CheckpointTuple
        +list(config, filter, before, limit) Iterator
        +put(config, checkpoint, metadata, versions) RunnableConfig
        +put_writes(config, writes, task_id) None
        +delete_thread(thread_id) None
    }
    
    class InMemorySaver {
        +defaultdict storage
        +defaultdict writes
        +dict blobs
        +get_tuple(config) CheckpointTuple
        +list(config, filter, before, limit) Iterator
        +put(config, checkpoint, metadata, versions) RunnableConfig
        +_load_blobs(thread_id, ns, versions) dict
    }
    
    class SerializerProtocol {
        <<interface>>
        +dumps(obj) bytes
        +loads(data) Any
        +dumps_typed(obj) tuple
        +loads_typed(data) Any
    }
    
    class JsonPlusSerializer {
        +bool pickle_fallback
        +dumps(obj) bytes
        +loads(data) Any
        +dumps_typed(obj) tuple~str,bytes~
        +loads_typed(data) Any
        +_default(obj) dict
        +_encode_constructor_args() dict
    }
    
    CheckpointTuple *-- Checkpoint
    CheckpointTuple *-- CheckpointMetadata
    CheckpointTuple o-- RunnableConfig
    
    BaseCheckpointSaver ..> CheckpointTuple : creates
    BaseCheckpointSaver ..> Checkpoint : stores
    BaseCheckpointSaver *-- SerializerProtocol
    
    InMemorySaver --|> BaseCheckpointSaver
    JsonPlusSerializer ..|> SerializerProtocol
    
    InMemorySaver ..> JsonPlusSerializer : uses
```

### 2.2 Store存储系统类图

```mermaid
classDiagram
    class Item {
        +tuple namespace
        +str key
        +dict value
        +datetime created_at
        +datetime updated_at
    }
    
    class BaseStore {
        <<abstract>>
        +get(namespace, key) Item
        +put(namespace, key, value) None
        +delete(namespace, key) None
        +search(namespace_prefix) list~Item~
        +batch(operations) list
    }
    
    class InMemoryStore {
        +dict _data
        +get(namespace, key) Item
        +put(namespace, key, value) None
        +delete(namespace, key) None
        +search(namespace_prefix) list~Item~
    }
    
    class BaseCache {
        <<abstract>>
        +get(namespace, key) Any
        +put(namespace, key, value, ttl) None
        +delete(namespace, key) None
        +clear(namespaces) None
    }
    
    class InMemoryCache {
        +dict _cache
        +dict _expiry
        +get(namespace, key) Any
        +put(namespace, key, value, ttl) None
    }
    
    BaseStore ..> Item : returns
    InMemoryStore --|> BaseStore
    InMemoryStore ..> Item : stores
    
    InMemoryCache --|> BaseCache
```

### 2.3 图意说明

#### 核心数据结构关系

**Checkpoint与CheckpointTuple**

- `CheckpointTuple`是`Checkpoint`的容器，包含配置、元数据和待处理写入
- `CheckpointTuple`用于对外传递，`Checkpoint`用于内部存储
- 通过`parent_config`字段形成检查点链

**BaseCheckpointSaver与实现类**

- `BaseCheckpointSaver`定义抽象接口，不同后端继承实现
- 所有实现类都依赖`SerializerProtocol`进行序列化
- `InMemorySaver`是最简单的实现，用于开发和测试

**序列化层**

- `SerializerProtocol`定义序列化契约
- `JsonPlusSerializer`实现扩展的JSON序列化，支持LangChain对象
- 序列化器可插拔，支持自定义实现

#### Store存储系统

**Item数据项**

- 表示Store中的单个存储项
- 包含命名空间、键、值和时间戳
- 支持层级命名空间（tuple形式）

**BaseStore抽象**

- 定义跨线程的长期存储接口
- 支持CRUD操作和前缀搜索
- 不同于检查点存储（检查点按线程隔离，Store全局共享）

## 三、详细数据结构定义

### 3.1 Checkpoint

```python
class Checkpoint(TypedDict):
    """
    状态快照，表示图在特定时刻的完整状态
    
    设计要点：

    1. 包含所有通道的值和版本信息
    2. 记录每个节点看到的版本，用于确定下一步执行
    3. 支持增量序列化（只序列化变化的通道）
    """
    
    v: int
    """
    检查点格式版本号
    
    当前值：1
    用途：支持未来格式升级和向后兼容
    """
    
    id: str
    """
    检查点唯一标识符
    
    格式：UUID v6字符串（如"1ef4f797-8335-6428-8001-8a1503f9b875"）
    特性：
    - 包含时间戳，保证单调递增
    - 字典序即时间序
    - 可用于分布式环境去重
    """
    
    ts: str
    """
    创建时间戳
    
    格式：ISO 8601（如"2024-07-31T20:14:19.804150+00:00"）
    精度：微秒级
    时区：UTC
    """
    
    channel_values: dict[str, Any]
    """
    所有通道的当前值
    
    键：通道名称（如"messages", "user_input"）
    值：通道的当前值（经过reducer聚合）
    
    注意：
    - 存储时可能被分离为blobs（InMemorySaver）
    - 支持延迟加载（数据库后端）
    """
    
    channel_versions: ChannelVersions
    """
    所有通道的当前版本
    
    类型：dict[str, str | int | float]
    格式：版本号可以是字符串"<seq>.<hash>"或数值
    
    用途：
    - 版本比较，确定通道是否有更新
    - 实现乐观并发控制
    """
    
    versions_seen: dict[str, ChannelVersions]
    """
    每个节点已看到的通道版本
    
    结构：
    {
        "node1": {"channel_a": "00000001.0.123", "channel_b": "00000002.0.456"},
        "node2": {"channel_a": "00000001.0.123"},
        ...
    }
    
    用途：
    - 确定哪些节点需要在下一步执行
    - 节点只在其触发器通道有新版本时执行
    """
    
    updated_channels: list[str] | None
    """
    本次更新的通道列表（可选）
    
    用途：
    - 优化节点调度，只检查更新过的通道
    - 与trigger_to_nodes映射配合使用
    
    示例：["messages", "agent_scratchpad"]
    """

```

**字段映射规则**：

| 内部存储 | 序列化格式 | 传输格式 |
|---------|-----------|---------|
| channel_values | 分离为blobs（InMemorySaver） | 合并到checkpoint |
| channel_versions | 字符串或数值 | 统一为字符串 |
| id | UUID对象 | UUID字符串 |
| ts | datetime对象 | ISO 8601字符串 |

**版本演进**：

- v1（当前版本）：初始格式
- v2（未来）：可能增加压缩、增量更新等特性

### 3.2 CheckpointMetadata

```python
class CheckpointMetadata(TypedDict, total=False):
    """
    检查点元数据
    
    设计原则：

    - total=False表示所有字段都是可选的
    - 可扩展，允许添加自定义字段
    """
    
    source: Literal["input", "loop", "update", "fork"]
    """
    检查点来源
    
    取值说明：
    - input: 首次调用时创建（step=-1）
    - loop: 正常超级步创建（step>=0）
    - update: 手动调用update_state创建
    - fork: 从其他检查点分支创建
    
    用途：区分检查点的创建场景
    """
    
    step: int
    """
    执行步数
    
    计数规则：
    - -1: input检查点
    -  0: 第一个loop检查点
    -  N: 第N个loop检查点
    
    用途：
    - 追踪执行进度
    - 调试和可视化
    - 过滤检查点（如只看前10步）
    """
    
    parents: dict[str, str]
    """
    父检查点映射
    
    结构：
    {
        "": "parent-checkpoint-id",  # 主图的父检查点
        "subgraph1": "subgraph-checkpoint-id",  # 子图的父检查点
    }
    
    键：checkpoint_ns（命名空间）
    值：父检查点的ID
    
    用途：
    - 支持多层级图的检查点关系
    - 追溯完整的执行历史
    """

```

**扩展字段示例**：

```python
metadata = {
    "source": "loop",
    "step": 5,
    "parents": {},
    # 自定义字段
    "user_id": "user-123",
    "session_id": "session-456",
    "execution_time_ms": 234,
    "tokens_used": 1500,
}
```

### 3.3 CheckpointTuple

```python
class CheckpointTuple(NamedTuple):
    """
    完整的检查点元组
    
    设计目的：

    - 将检查点及其上下文信息打包为不可变对象
    - 提供类型安全的字段访问
    """
    
    config: RunnableConfig
    """
    检查点的完整配置
    
    必含字段：
    - configurable.thread_id
    - configurable.checkpoint_ns
    - configurable.checkpoint_id
    
    示例：
    {
        "configurable": {
            "thread_id": "thread-123",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-...",
        }
    }
    """
    
    checkpoint: Checkpoint
    """
    检查点数据
    
    注意：channel_values已加载（不再是blobs）
    """
    
    metadata: CheckpointMetadata
    """
    检查点元数据
    
    可能包含自定义字段
    """
    
    parent_config: RunnableConfig | None = None
    """
    父检查点的配置
    
    用途：
    - 追溯检查点链
    - 实现时间旅行
    - 支持分支合并
    
    为None表示：
    - 这是第一个检查点
    - 或者是孤立的检查点（如fork后）
    """
    
    pending_writes: list[PendingWrite] | None = None
    """
    待处理的写入操作
    
    类型：list[tuple[str, str, Any]]
    格式：[(task_id, channel, value), ...]
    
    场景：
    - 某个节点执行失败
    - 其他成功节点的写入被保存为pending_writes
    - 恢复时直接应用这些写入，不重新执行成功的节点
    
    示例：
    [
        ("agent", "messages", AIMessage(...)),
        ("tools", "tool_results", [...]),
    ]
    """

```

**使用示例**：

```python
# 获取检查点元组
checkpoint_tuple = checkpointer.get_tuple(config)

# 访问字段
print(f"Step: {checkpoint_tuple.metadata['step']}")
print(f"Values: {checkpoint_tuple.checkpoint['channel_values']}")

# 追溯父检查点
if checkpoint_tuple.parent_config:
    parent_tuple = checkpointer.get_tuple(checkpoint_tuple.parent_config)
    print(f"Parent step: {parent_tuple.metadata['step']}")

# 检查是否有待处理写入
if checkpoint_tuple.pending_writes:
    print(f"有{len(checkpoint_tuple.pending_writes)}个待处理写入")
```

### 3.4 ChannelVersions

```python
ChannelVersions = dict[str, Union[str, int, float]]
"""
通道版本映射

键：通道名称
值：版本号（字符串、整数或浮点数）

字符串格式："<sequence>.<random_hash>"

- sequence: 单调递增的序列号（32位整数）
- random_hash: 随机哈希（16位浮点数）

示例：
{
    "messages": "00000005.0.7234819283748",
    "agent_scratchpad": "00000003.0.2837461928374",
    "user_input": 3,  # 简单数值版本
}
"""
```

**版本比较规则**：

```python
def compare_versions(v1: str | int | float, v2: str | int | float) -> int:
    """
    返回：
    > 0: v1 > v2 (v1更新)
    = 0: v1 == v2 (相同版本)
    < 0: v1 < v2 (v2更新)
    """
    if isinstance(v1, str) and isinstance(v2, str):
        # 比较sequence部分
        seq1 = int(v1.split(".")[0])
        seq2 = int(v2.split(".")[0])
        return seq1 - seq2
    else:
        # 数值直接比较
        return 1 if v1 > v2 else (-1 if v1 < v2 else 0)
```

### 3.5 PendingWrite

```python
PendingWrite = tuple[str, str, Any]
"""
待处理的写入操作

结构：

- [0] task_id: 任务标识符（通常是节点名）
- [1] channel: 目标通道名
- [2] value: 要写入的值

示例：
("agent", "messages", AIMessage(content="Hello"))
("tools", "tool_results", ["Result1", "Result2"])
("error_handler", ERROR, Exception("Failed"))
"""
```

**特殊通道**：

```python
# 定义在langgraph._internal._constants中
ERROR = "__error__"  # 错误通道
INTERRUPT = "__interrupt__"  # 中断通道

# 这些通道的写入具有特殊语义
WRITES_IDX_MAP: dict[str, int] = {
    ERROR: -1,  # 错误写入，优先级最高
    INTERRUPT: -2,  # 中断写入
}
```

### 3.6 Item (Store数据项)

```python
@dataclass
class Item:
    """
    Store中的存储项
    
    Store与Checkpoint的区别：

    - Checkpoint: 按线程隔离，记录执行状态
    - Store: 全局共享，存储长期数据
    """
    
    namespace: tuple[str, ...]
    """
    命名空间（层级）
    
    示例：
    - ("users",): 用户数据
    - ("users", "user-123", "preferences"): 用户偏好
    - ("knowledge_base", "documents"): 知识库文档
    
    设计：
    - 使用tuple支持层级结构
    - 支持前缀搜索
    """
    
    key: str
    """
    存储键
    
    在命名空间内唯一标识一个项
    
    示例：
    - "user-123"
    - "doc-456"
    - "preference-theme"
    """
    
    value: dict[str, Any]
    """
    存储值
    
    类型：必须是可JSON序列化的字典
    
    示例：
    {
        "name": "Alice",
        "email": "alice@example.com",
        "preferences": {
            "theme": "dark",
            "language": "en"
        }
    }
    """
    
    created_at: datetime
    """
    创建时间
    
    格式：datetime对象（UTC时区）
    自动设置：首次创建时
    """
    
    updated_at: datetime
    """
    最后更新时间
    
    格式：datetime对象（UTC时区）
    自动更新：每次put操作时
    """

```

**使用示例**：

```python
from langgraph.store.memory import InMemoryStore
from datetime import datetime

store = InMemoryStore()

# 存储用户数据
store.put(
    namespace=("users",),
    key="user-123",
    value={
        "name": "Alice",
        "email": "alice@example.com",
    }
)

# 检索
item = store.get(
    namespace=("users",),
    key="user-123",
)
print(item.value["name"])  # "Alice"
print(item.created_at)  # datetime对象

# 搜索命名空间
items = store.search(namespace_prefix=("users",))
for item in items:
    print(f"{item.key}: {item.value['name']}")
```

## 四、数据结构关系与生命周期

### 4.1 检查点创建流程

```mermaid
stateDiagram-v2
    [*] --> CreateCheckpoint: Pregel执行超级步
    
    CreateCheckpoint --> GenerateID: 生成UUID v6
    GenerateID --> CollectChannels: 收集通道值和版本
    CollectChannels --> BuildCheckpoint: 构建Checkpoint对象
    BuildCheckpoint --> AddMetadata: 添加元数据
    AddMetadata --> SerializeData: 序列化数据
    SerializeData --> StoreCheckpoint: 存储到后端
    StoreCheckpoint --> [*]
    
    note right of CreateCheckpoint
        触发条件：

        - 超级步完成
        - 手动update_state
        - 创建分支
    end note
    
    note right of BuildCheckpoint
        包含：
        - channel_values
        - channel_versions
        - versions_seen
        - updated_channels
    end note

```

### 4.2 检查点检索流程

```mermaid
stateDiagram-v2
    [*] --> ParseConfig: 解析配置
    ParseConfig --> QueryStorage: 查询存储后端
    
    QueryStorage --> CheckResult: 检查结果
    CheckResult --> NotFound: 未找到
    CheckResult --> Found: 找到
    
    Found --> LoadData: 加载序列化数据
    LoadData --> Deserialize: 反序列化
    Deserialize --> LoadBlobs: 加载通道值
    LoadBlobs --> LoadWrites: 加载待处理写入
    LoadWrites --> BuildTuple: 构建CheckpointTuple
    BuildTuple --> [*]
    
    NotFound --> [*]: 返回None
    
    note right of LoadBlobs
        分离存储时需要：

        - 根据channel_versions
        - 加载每个通道的blob
        - 反序列化通道值
    end note

```

### 4.3 数据流转图

```mermaid
flowchart LR
    subgraph "执行时"
        A[GraphState] --> B[Checkpoint]
        B --> C[Serialized Data]
        C --> D[Storage Backend]
    end
    
    subgraph "恢复时"
        D --> E[Serialized Data]
        E --> F[Checkpoint]
        F --> G[GraphState]
    end
    
    subgraph "序列化层"
        H[JsonPlusSerializer]
        C -.serialize.-> H
        H -.deserialize.-> E
    end
    
    style B fill:#ffe1e1
    style F fill:#ffe1e1
    style H fill:#fff4e1
```

## 五、字段约束与验证

### 5.1 Checkpoint字段约束

| 字段 | 约束 | 验证方法 |
|------|------|---------|
| v | 必须为1 | 类型检查 |
| id | 必须是有效的UUID字符串 | UUID.validate() |
| ts | 必须是ISO 8601格式 | datetime.fromisoformat() |
| channel_values | 必须可序列化 | 序列化器测试 |
| channel_versions | 版本号必须可比较 | compare_versions() |
| versions_seen | 键必须是节点名 | 与图定义对比 |

### 5.2 CheckpointMetadata字段约束

| 字段 | 约束 | 默认值 |
|------|------|--------|
| source | 必须是枚举值之一 | "loop" |
| step | 必须>=−1 | -1 |
| parents | 值必须是有效的checkpoint_id | {} |

### 5.3 Item字段约束

| 字段 | 约束 | 验证方法 |
|------|------|---------|
| namespace | 非空tuple | len(namespace) > 0 |
| key | 非空字符串 | len(key) > 0 |
| value | 可JSON序列化的dict | json.dumps(value) |
| created_at | UTC datetime | timezone检查 |
| updated_at | >= created_at | 时间比较 |

## 六、性能优化建议

### 6.1 减少Checkpoint大小

**问题**：大的channel_values导致序列化和存储慢

**优化**：

1. 使用引用而非直接存储大对象
2. 定期清理不需要的通道值
3. 使用压缩序列化器

**示例**：

```python
class State(TypedDict):
    # 不推荐：直接存储大对象
    # large_data: bytes  # 可能有几MB
    
    # 推荐：使用引用
    large_data_url: str  # S3/GCS URL
    large_data_hash: str  # 用于验证
```

### 6.2 优化版本比较

**问题**：字符串版本比较需要split和int转换

**优化**：

1. 缓存解析后的sequence值
2. 使用数值版本（如果不需要random_hash）

**示例**：

```python
# InMemorySaver使用简单数值版本
def get_next_version(self, current: str | None, channel: None) -> str:
    if current is None:
        current_v = 0
    else:
        current_v = int(current.split(".")[0])
    next_v = current_v + 1
    return f"{next_v:032}.{random.random():016}"

# 优化：直接使用整数
def get_next_version_optimized(self, current: int | None) -> int:
    return (current or 0) + 1
```

### 6.3 延迟加载Blobs

**问题**：加载整个检查点时，可能不需要所有通道值

**优化**：

1. 只在访问时加载blob
2. 使用代理对象包装channel_values

**示例**：

```python
class LazyChannelValues:
    def __init__(self, loader, versions):
        self._loader = loader
        self._versions = versions
        self._cache = {}
    
    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = self._loader(key, self._versions[key])
        return self._cache[key]
```

## 七、总结

checkpoint模块的数据结构设计体现了以下原则：

1. **不可变性**：使用TypedDict和NamedTuple，确保数据不被意外修改
2. **类型安全**：充分利用Python类型提示，便于IDE支持和静态检查
3. **可扩展性**：元数据支持自定义字段，便于未来扩展
4. **性能优化**：支持分离存储、延迟加载等优化手段
5. **向后兼容**：通过版本字段支持格式演进

通过这些精心设计的数据结构，checkpoint模块为LangGraph提供了可靠、高效的状态持久化能力。

---
