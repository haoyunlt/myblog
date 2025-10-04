# LangGraph-01-checkpoint-API

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

