# LangGraph-02-langgraph-API

## 一、API概览

langgraph核心模块对外提供以下主要API：

| API类/方法 | 类型 | 功能描述 |
|-----------|------|---------|
| StateGraph | 类 | 状态图构建器 |
| MessageGraph | 类 | 消息图构建器（StateGraph的特化版本） |
| CompiledStateGraph | 类 | 编译后的可执行图 |
| add_messages | 函数 | 消息列表的reducer函数 |
| START | 常量 | 图的入口节点标识 |
| END | 常量 | 图的出口节点标识 |
| Send | 类 | 动态任务创建 |

## 二、StateGraph API

### 2.1 StateGraph.__init__

#### 基本信息

- **方法名称**: `__init__`
- **协议**: 构造函数
- **幂等性**: N/A

#### 请求结构体

```python
def __init__(
    self,
    state_schema: Type[StateT] | StateSchemaType,
    context_schema: Type[ContextT] | None = None,
    input_schema: Type[InputT] | None = None,
    output_schema: Type[OutputT] | None = None,
) -> None:
    pass
```

| 字段 | 类型 | 必填 | 默认值 | 约束 | 说明 |
|------|------|------|--------|------|------|
| state_schema | Type[StateT] \| StateSchemaType | 是 | 无 | TypedDict或Pydantic模型 | 状态的结构定义 |
| context_schema | Type[ContextT] \| None | 否 | None | 任意类型 | 运行时上下文的结构定义 |
| input_schema | Type[InputT] \| None | 否 | None | TypedDict或Pydantic模型 | 输入数据的结构定义 |
| output_schema | Type[OutputT] \| None | 否 | None | TypedDict或Pydantic模型 | 输出数据的结构定义 |

#### 响应结构体

返回`StateGraph`实例。

#### 入口函数与核心代码

```python
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """状态图构建器"""
    
    def __init__(
        self,
        state_schema: Type[StateT] | StateSchemaType,
        context_schema: Type[ContextT] | None = None,
        input_schema: Type[InputT] | None = None,
        output_schema: Type[OutputT] | None = None,
    ) -> None:
        """
        初始化状态图构建器
        
        功能：
        1. 解析state_schema，提取字段和reducer
        2. 初始化节点和边的容器
        3. 设置输入输出schema
        
        参数：
            state_schema: 状态结构，支持TypedDict或Pydantic模型
            context_schema: 运行时上下文结构
            input_schema: 输入结构，默认与state_schema相同
            output_schema: 输出结构，默认与state_schema相同
        """
        # 1. 解析state_schema
        if is_typeddict(state_schema):
            # TypedDict: 使用get_type_hints提取字段
            self.schemas = {state_schema: get_type_hints(state_schema)}
        elif isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            # Pydantic: 使用model_fields提取字段
            self.schemas = {state_schema: state_schema.model_fields}
        else:
            # 其他类型
            self.schemas = {state_schema: {}}
        
        # 2. 提取reducer（从Annotated类型）
        self.channels = {}
        for key, field_type in self.schemas[state_schema].items():
            if get_origin(field_type) is Annotated:
                # Annotated[list, add_messages]
                args = get_args(field_type)
                base_type = args[0]
                reducer = args[1] if len(args) > 1 else None
            else:
                base_type = field_type
                reducer = None
            
            # 根据reducer选择通道类型
            if reducer:
                self.channels[key] = BinaryOperatorAggregate(base_type, reducer)
            else:
                self.channels[key] = LastValue(base_type)
        
        # 3. 初始化节点和边
        self.nodes: dict[str, StateNode] = {}
        self.edges: set[tuple[str, str]] = set()
        self.branches: dict[str, dict[str, BranchSpec]] = defaultdict(dict)
        
        # 4. 设置schema
        self.state_schema = state_schema
        self.context_schema = context_schema
        self.input_schema = input_schema or state_schema
        self.output_schema = output_schema or state_schema
        
        # 5. 设置入口和出口
        self._entry_point: str | None = None
        self._finish_points: set[str] = set()
```

**state_schema解析示例**：

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages

# 示例1：TypedDict with reducer
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 使用add_messages reducer
    counter: Annotated[int, operator.add]  # 使用operator.add reducer
    name: str  # 无reducer，使用LastValue

# 解析结果：
# channels = {
#     "messages": BinaryOperatorAggregate(list, add_messages),
#     "counter": BinaryOperatorAggregate(int, operator.add),
#     "name": LastValue(str),
# }

# 示例2：Pydantic模型
from pydantic import BaseModel

class State(BaseModel):
    messages: Annotated[list, add_messages] = []
    counter: int = 0
    name: str = ""
```

#### 调用链路

```
用户代码
  └─> StateGraph(State)
       └─> __init__()
            ├─> 解析state_schema
            │    ├─> is_typeddict() 或 issubclass(BaseModel)
            │    └─> get_type_hints() 或 model_fields
            ├─> 提取Annotated中的reducer
            │    ├─> get_origin() == Annotated
            │    └─> get_args()
            ├─> 创建通道
            │    ├─> BinaryOperatorAggregate (有reducer)
            │    └─> LastValue (无reducer)
            └─> 初始化容器（nodes, edges, branches）
```

#### 异常与性能

**异常情况**：
- `TypeError`: state_schema类型不支持
- `ValueError`: reducer函数签名不正确

**性能要点**：
- 时间复杂度：O(F)，F为state_schema的字段数
- 空间复杂度：O(F)
- 只在初始化时执行一次

---

### 2.2 add_node

#### 基本信息

- **方法名称**: `add_node`
- **协议**: 实例方法
- **幂等性**: 否（重复添加同名节点会覆盖）

#### 请求结构体

```python
def add_node(
    self, 
    node: str, 
    action: Callable[[StateT], dict] | Runnable,
) -> Self:
    pass
```

| 字段 | 类型 | 必填 | 约束 | 说明 |
|------|------|------|------|------|
| node | str | 是 | 非空字符串，不能是START/END | 节点名称，用于引用 |
| action | Callable \| Runnable | 是 | 必须接收State，返回dict | 节点的执行函数 |

#### 响应结构体

返回`Self`（支持链式调用）。

#### 入口函数与核心代码

```python
def add_node(
    self, 
    node: str, 
    action: Callable[[StateT], dict] | Runnable,
) -> Self:
    """
    添加节点到图
    
    功能：
    1. 验证节点名称
    2. 将action包装为StateNode
    3. 存储到nodes字典
    
    参数：
        node: 节点名称
        action: 节点函数，接收State，返回State的部分更新
        
    返回：
        Self，支持链式调用
        
    示例：
        graph.add_node("agent", agent_func)
              .add_node("tools", tool_func)
    """
    # 1. 验证节点名称
    if node in (START, END):
        raise ValueError(f"节点名称不能为 {START} 或 {END}")
    
    if not isinstance(node, str) or not node:
        raise ValueError("节点名称必须为非空字符串")
    
    # 2. 包装为StateNode
    if callable(action):
        # 普通函数：包装为RunnableCallable
        runnable = RunnableCallable(action, name=node)
    elif isinstance(action, Runnable):
        # 已经是Runnable
        runnable = action
    else:
        raise TypeError(f"action必须是Callable或Runnable，得到 {type(action)}")
    
    # 3. 创建StateNode
    state_node = StateNode(
        runnable=runnable,
        input=self.state_schema,
        output=dict,  # 节点返回dict类型
        metadata={"name": node},
    )
    
    # 4. 存储
    self.nodes[node] = state_node
    
    return self
```

**节点函数签名**：

```python
# 基础签名
def node_func(state: State) -> dict:
    """
    参数：
        state: 当前状态（只读）
        
    返回：
        dict: 状态的部分更新
    """
    return {"key": new_value}

# 高级签名：注入runtime
def node_func(state: State, runtime: Runtime[Context]) -> dict:
    """
    参数：
        state: 当前状态
        runtime: 运行时对象，包含context、store等
        
    返回：
        dict: 状态的部分更新
    """
    user_id = runtime.context.user_id
    return {"key": new_value}
```

#### 调用链路

```
用户代码
  └─> graph.add_node("agent", agent_func)
       └─> StateGraph.add_node()
            ├─> 验证节点名称
            ├─> 包装action为Runnable
            │    └─> RunnableCallable(action, name=node)
            ├─> 创建StateNode
            │    └─> StateNode(runnable, input, output, metadata)
            └─> 存储到self.nodes[node]
```

#### 异常与性能

**异常情况**：
- `ValueError`: 节点名称为START或END
- `ValueError`: 节点名称为空
- `TypeError`: action类型不支持

**性能要点**：
- 时间复杂度：O(1)
- 空间复杂度：O(1)

---

### 2.3 add_edge

#### 基本信息

- **方法名称**: `add_edge`
- **协议**: 实例方法
- **幂等性**: 是（重复添加相同边无影响）

#### 请求结构体

```python
def add_edge(self, start_key: str, end_key: str) -> Self:
    pass
```

| 字段 | 类型 | 必填 | 约束 | 说明 |
|------|------|------|------|------|
| start_key | str | 是 | 必须是已添加的节点或START | 起始节点 |
| end_key | str | 是 | 必须是已添加的节点或END | 目标节点 |

#### 响应结构体

返回`Self`（支持链式调用）。

#### 入口函数与核心代码

```python
def add_edge(self, start_key: str, end_key: str) -> Self:
    """
    添加固定边
    
    功能：
    1. 验证节点存在性
    2. 添加边到edges集合
    3. 如果start_key是START，设置为入口点
    4. 如果end_key是END，添加到finish_points
    
    参数：
        start_key: 起始节点名称（或START）
        end_key: 目标节点名称（或END）
        
    返回：
        Self，支持链式调用
    """
    # 1. 验证起始节点
    if start_key != START and start_key not in self.nodes:
        raise ValueError(f"起始节点 '{start_key}' 未定义")
    
    # 2. 验证目标节点
    if end_key != END and end_key not in self.nodes:
        raise ValueError(f"目标节点 '{end_key}' 未定义")
    
    # 3. 处理START
    if start_key == START:
        if self._entry_point is not None:
            raise ValueError(f"已设置入口点为 '{self._entry_point}'")
        self._entry_point = end_key
    
    # 4. 处理END
    if end_key == END:
        self._finish_points.add(start_key)
    
    # 5. 添加边
    self.edges.add((start_key, end_key))
    
    return self
```

**边的类型**：

```python
# 固定边：确定性转换
graph.add_edge(START, "node1")
graph.add_edge("node1", "node2")
graph.add_edge("node2", END)

# 形成线性流程：START -> node1 -> node2 -> END
```

#### 调用链路

```
用户代码
  └─> graph.add_edge("node1", "node2")
       └─> StateGraph.add_edge()
            ├─> 验证start_key存在
            ├─> 验证end_key存在
            ├─> 处理特殊节点（START/END）
            │    ├─> START: 设置_entry_point
            │    └─> END: 添加到_finish_points
            └─> 添加到self.edges集合
```

#### 异常与性能

**异常情况**：
- `ValueError`: 节点未定义
- `ValueError`: 重复设置入口点

**性能要点**：
- 时间复杂度：O(1)
- 空间复杂度：O(1)

---

### 2.4 add_conditional_edge

#### 基本信息

- **方法名称**: `add_conditional_edge`
- **协议**: 实例方法
- **幂等性**: 否（会覆盖已有的条件边）

#### 请求结构体

```python
def add_conditional_edge(
    self,
    source: str,
    path: Callable[[StateT], str] | Callable[[StateT], list[str]],
    path_map: dict[str, str] | None = None,
    then: str | None = None,
) -> Self:
    pass
```

| 字段 | 类型 | 必填 | 约束 | 说明 |
|------|------|------|------|------|
| source | str | 是 | 必须是已添加的节点或START | 源节点 |
| path | Callable | 是 | 接收State，返回str或list[str] | 路由函数 |
| path_map | dict[str, str] \| None | 否 | None | 路由值到节点名的映射 |
| then | str \| None | 否 | None | 所有分支汇聚后的下一个节点 |

#### 响应结构体

返回`Self`（支持链式调用）。

#### 入口函数与核心代码

```python
def add_conditional_edge(
    self,
    source: str,
    path: Callable[[StateT], str] | Callable[[StateT], list[str]],
    path_map: dict[str, str] | None = None,
    then: str | None = None,
) -> Self:
    """
    添加条件边
    
    功能：
    1. 验证源节点
    2. 包装path函数为Runnable
    3. 创建BranchSpec
    4. 存储到branches字典
    
    参数：
        source: 源节点名称
        path: 路由函数，接收State，返回目标节点名
        path_map: 路由值映射，将path返回值映射到实际节点名
        then: 汇聚节点，所有分支执行完后的下一步
        
    返回：
        Self，支持链式调用
        
    示例：
        def should_continue(state: State) -> str:
            if state["count"] > 10:
                return "end"
            return "continue"
        
        graph.add_conditional_edge(
            "process",
            should_continue,
            {
                "continue": "process",  # 循环
                "end": END,  # 结束
            }
        )
    """
    # 1. 验证源节点
    if source != START and source not in self.nodes:
        raise ValueError(f"源节点 '{source}' 未定义")
    
    # 2. 包装path函数
    if callable(path):
        path_runnable = RunnableCallable(path, name=f"{source}_router")
    else:
        raise TypeError("path必须是Callable")
    
    # 3. 验证path_map
    if path_map:
        for target in path_map.values():
            if target != END and target not in self.nodes:
                raise ValueError(f"目标节点 '{target}' 未定义")
    
    # 4. 验证then
    if then and then not in self.nodes and then != END:
        raise ValueError(f"汇聚节点 '{then}' 未定义")
    
    # 5. 创建BranchSpec
    branch_spec = BranchSpec(
        path=path_runnable,
        path_map=path_map or {},
        then=then,
    )
    
    # 6. 存储（每个源节点只能有一个条件边，但可以有多个分支）
    if source in self.branches and self.branches[source]:
        # 已有条件边，覆盖
        pass
    
    self.branches[source]["__default__"] = branch_spec
    
    return self
```

**路由函数类型**：

```python
# 类型1：返回单个目标
def route_single(state: State) -> str:
    if state["success"]:
        return "next_node"
    return END

# 类型2：返回多个目标（并行执行）
def route_multiple(state: State) -> list[str]:
    targets = []
    if state["need_process"]:
        targets.append("process")
    if state["need_save"]:
        targets.append("save")
    return targets

# 类型3：使用Send动态创建任务
def route_dynamic(state: State) -> dict:
    items = state["items"]
    return {
        "__send__": [Send("process", {"item": item}) for item in items]
    }
```

#### 调用链路

```
用户代码
  └─> graph.add_conditional_edge("node1", route_func, path_map)
       └─> StateGraph.add_conditional_edge()
            ├─> 验证source节点
            ├─> 包装path为Runnable
            │    └─> RunnableCallable(path, name="...")
            ├─> 验证path_map中的目标节点
            ├─> 验证then节点（如果提供）
            ├─> 创建BranchSpec
            │    └─> BranchSpec(path, path_map, then)
            └─> 存储到self.branches[source]
```

#### 异常与性能

**异常情况**：
- `ValueError`: 源节点未定义
- `ValueError`: path_map中的目标节点未定义
- `TypeError`: path不是Callable

**性能要点**：
- 时间复杂度：O(M)，M为path_map的大小
- 空间复杂度：O(M)

---

### 2.5 compile

#### 基本信息

- **方法名称**: `compile`
- **协议**: 实例方法
- **幂等性**: 是（多次编译结果相同）

#### 请求结构体

```python
def compile(
    self,
    checkpointer: Checkpointer = None,
    *,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    interrupt_before: All | list[str] | None = None,
    interrupt_after: All | list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
    pass
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| checkpointer | Checkpointer | 否 | None | 检查点存储器，用于持久化 |
| cache | BaseCache \| None | 否 | None | 缓存实例，用于节点级缓存 |
| store | BaseStore \| None | 否 | None | 存储实例，用于跨线程数据 |
| interrupt_before | All \| list[str] \| None | 否 | None | 在指定节点前中断 |
| interrupt_after | All \| list[str] \| None | 否 | None | 在指定节点后中断 |
| debug | bool | 否 | False | 是否启用调试模式 |
| name | str \| None | 否 | None | 编译后图的名称 |

#### 响应结构体

返回`CompiledStateGraph`实例，实现Runnable接口。

#### 入口函数与核心代码

```python
def compile(
    self,
    checkpointer: Checkpointer = None,
    *,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    interrupt_before: All | list[str] | None = None,
    interrupt_after: All | list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
) -> CompiledStateGraph:
    """
    编译状态图为可执行对象
    
    功能：
    1. 验证图结构的完整性
    2. 为状态字段分配通道
    3. 转换节点为PregelNode
    4. 建立触发器关系
    5. 创建Pregel执行引擎
    6. 包装为CompiledStateGraph
    
    参数：
        checkpointer: 检查点存储器
        cache: 节点缓存
        store: 跨线程存储
        interrupt_before: 中断点（执行前）
        interrupt_after: 中断点（执行后）
        debug: 调试模式
        name: 图名称
        
    返回：
        CompiledStateGraph: 可执行的图对象
    """
    # 步骤1：验证图结构
    self.validate(
        interrupt=(
            (interrupt_before if interrupt_before != "*" else [])
            + (interrupt_after if interrupt_after != "*" else [])
        )
    )
    
    # 步骤2：准备输出通道
    output_channels = self._get_output_channels()
    
    # 步骤3：获取通道定义
    channels = self._get_channels(checkpointer)
    
    # 步骤4：转换节点为PregelNode
    compiled_nodes = {}
    for node_name, state_node in self.nodes.items():
        # 4.1 创建输入读取器
        input_reader = ChannelRead(
            channels=list(self.schemas[self.state_schema].keys()),
            fresh=False,
        )
        
        # 4.2 创建输出写入器
        output_writer = ChannelWrite(
            writes=[
                ChannelWriteEntry(ch) 
                for ch in self.schemas[self.state_schema].keys()
            ],
            require_at_least_one_of=list(self.schemas[self.state_schema].keys()),
        )
        
        # 4.3 组装PregelNode
        compiled_nodes[node_name] = PregelNode(
            channels=list(self.schemas[self.state_schema].keys()),
            triggers=[],  # 稍后填充
            bound=RunnableSequence(
                input_reader,
                state_node.runnable,
                output_writer,
            ),
        )
    
    # 步骤5：建立触发器关系
    trigger_to_nodes = defaultdict(list)
    
    for start, end in self.edges:
        if end != END:
            # end节点被start的输出触发
            for channel in self._get_node_output_channels(start):
                trigger_to_nodes[channel].append(end)
                compiled_nodes[end].triggers.append(channel)
    
    for source, branches in self.branches.items():
        for branch_name, branch_spec in branches.items():
            if branch_spec.path_map:
                for target in branch_spec.path_map.values():
                    if target != END:
                        for channel in self._get_node_output_channels(source):
                            trigger_to_nodes[channel].append(target)
                            compiled_nodes[target].triggers.append(channel)
    
    # 步骤6：创建Pregel实例
    pregel = Pregel(
        nodes=compiled_nodes,
        channels=channels,
        input_channels=self.input_schema or "__input__",
        output_channels=output_channels,
        stream_channels=output_channels,
        interrupt_before_nodes=interrupt_before or [],
        interrupt_after_nodes=interrupt_after or [],
        trigger_to_nodes=dict(trigger_to_nodes),
        debug=debug,
        checkpointer=checkpointer,
        store=store,
        cache=cache,
    )
    
    # 步骤7：包装为CompiledStateGraph
    return CompiledStateGraph(
        pregel=pregel,
        builder=self,
        name=name or self.__class__.__name__,
        context_schema=self.context_schema,
    )

def validate(self, interrupt: list[str] | None = None) -> None:
    """
    验证图结构
    
    检查项：
    1. 必须有入口点（START边或set_entry_point）
    2. 必须有出口点（END边或set_finish_point）
    3. 所有节点都可达（从START出发）
    4. 没有孤立节点（无入边也无出边）
    5. 边引用的节点都已定义
    6. 中断点引用的节点都已定义
    """
    # 1. 检查入口点
    if self._entry_point is None:
        raise ValueError("必须通过 add_edge(START, ...) 或 set_entry_point(...) 设置入口点")
    
    # 2. 检查出口点
    if not self._finish_points:
        raise ValueError("必须通过 add_edge(..., END) 或 set_finish_point(...) 设置出口点")
    
    # 3. 检查可达性（BFS）
    visited = set()
    queue = deque([self._entry_point])
    
    while queue:
        node = queue.popleft()
        if node in visited or node == END:
            continue
        visited.add(node)
        
        # 添加固定边的目标
        for start, end in self.edges:
            if start == node and end not in visited:
                queue.append(end)
        
        # 添加条件边的目标
        if node in self.branches:
            for branch_spec in self.branches[node].values():
                if branch_spec.path_map:
                    for target in branch_spec.path_map.values():
                        if target != END and target not in visited:
                            queue.append(target)
    
    # 4. 检查孤立节点
    all_nodes = set(self.nodes.keys())
    unreachable = all_nodes - visited
    if unreachable:
        raise ValueError(f"以下节点不可达: {unreachable}")
    
    # 5. 检查中断点
    if interrupt:
        for node in interrupt:
            if node not in self.nodes:
                raise ValueError(f"中断点 '{node}' 未定义")
```

#### 调用链路

```
用户代码
  └─> app = graph.compile(checkpointer=...)
       └─> StateGraph.compile()
            ├─> validate()
            │    ├─> 检查入口点
            │    ├─> 检查出口点
            │    ├─> BFS检查可达性
            │    └─> 检查孤立节点
            ├─> _get_output_channels()
            ├─> _get_channels()
            │    └─> 为每个状态字段创建通道
            ├─> 转换节点
            │    └─> 为每个节点创建PregelNode
            │         ├─> ChannelRead（输入）
            │         ├─> 节点runnable（处理）
            │         └─> ChannelWrite（输出）
            ├─> 建立触发器关系
            │    └─> 分析边和分支，确定triggers
            ├─> 创建Pregel实例
            │    └─> Pregel(nodes, channels, ...)
            └─> 包装为CompiledStateGraph
                 └─> CompiledStateGraph(pregel, builder, ...)
```

#### 异常与性能

**异常情况**：
- `ValueError`: 图结构不完整（无入口/出口）
- `ValueError`: 存在不可达节点
- `ValueError`: 中断点节点未定义

**性能要点**：
- 时间复杂度：O(N + E)，N为节点数，E为边数
- 空间复杂度：O(N + E)
- 只在编译时执行一次，运行时不再验证

---

## 三、CompiledStateGraph API

### 3.1 invoke

#### 基本信息

- **方法名称**: `invoke`
- **协议**: 同步方法，实现Runnable接口
- **幂等性**: 否（有副作用：状态变化、检查点保存）

#### 请求结构体

```python
def invoke(
    self,
    input: InputT,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> OutputT:
    pass
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| input | InputT | 是 | 输入数据，类型由input_schema定义 |
| config | RunnableConfig \| None | 否 | 运行配置（thread_id、recursion_limit等） |
| context | ContextT \| None | 否（kwargs） | 运行时上下文 |

#### 响应结构体

返回`OutputT`，类型由output_schema定义，通常是State的全部或部分字段。

#### 入口函数与核心代码

```python
def invoke(
    self,
    input: InputT,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> OutputT:
    """
    同步执行图
    
    功能：
    1. 合并config和kwargs
    2. 注入context（如果提供）
    3. 调用Pregel.invoke执行
    4. 返回最终状态
    
    参数：
        input: 初始输入
        config: 运行配置
        **kwargs: 可包含context等
        
    返回：
        最终状态（根据output_schema）
    """
    # 1. 准备config
    config = ensure_config(config)
    
    # 2. 处理context
    if "context" in kwargs:
        context = kwargs.pop("context")
        if self.context_schema:
            # 验证context类型
            if not isinstance(context, self.context_schema):
                raise TypeError(f"context必须是 {self.context_schema}")
        # 注入到config
        config = merge_configs(config, {"configurable": {"__context__": context}})
    
    # 3. 调用Pregel执行
    result = self.pregel.invoke(input, config, **kwargs)
    
    return result
```

#### 调用链路

```
用户代码
  └─> app.invoke({"messages": ["Hello"]}, config)
       └─> CompiledStateGraph.invoke()
            ├─> ensure_config()
            ├─> 处理context（如果有）
            └─> Pregel.invoke()
                 ├─> 初始化或恢复状态
                 ├─> 超级步循环
                 │    ├─> prepare_next_tasks()
                 │    ├─> 并行执行节点
                 │    ├─> apply_writes()
                 │    └─> 保存检查点
                 └─> 返回最终状态
```

---

### 3.2 stream

#### 基本信息

- **方法名称**: `stream`
- **协议**: 同步方法，返回迭代器
- **幂等性**: 否

#### 请求结构体

```python
def stream(
    self,
    input: InputT,
    config: RunnableConfig | None = None,
    *,
    stream_mode: StreamMode = "values",
    **kwargs: Any,
) -> Iterator[OutputT | tuple[str, OutputT]]:
    pass
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| input | InputT | 是 | - | 输入数据 |
| config | RunnableConfig \| None | 否 | None | 运行配置 |
| stream_mode | StreamMode | 否 | "values" | 流式模式：values/updates/debug |

**stream_mode选项**：
- `"values"`: 每个超级步后的完整状态
- `"updates"`: 每个节点的状态更新
- `"debug"`: 调试信息（节点输入输出、时间等）

#### 响应结构体

返回迭代器，每次yield一个输出。

#### 入口函数与核心代码

```python
def stream(
    self,
    input: InputT,
    config: RunnableConfig | None = None,
    *,
    stream_mode: StreamMode = "values",
    **kwargs: Any,
) -> Iterator[OutputT | tuple[str, OutputT]]:
    """
    流式执行图
    
    功能：
    1. 配置流式模式
    2. 调用Pregel.stream执行
    3. Yield中间和最终结果
    
    参数：
        input: 初始输入
        config: 运行配置
        stream_mode: 流式模式
        
    返回：
        迭代器，根据stream_mode返回不同格式
    """
    config = ensure_config(config)
    
    # 注入stream_mode到config
    config = merge_configs(config, {"configurable": {"__stream_mode__": stream_mode}})
    
    # 调用Pregel.stream
    for chunk in self.pregel.stream(input, config, stream_mode=stream_mode, **kwargs):
        yield chunk
```

**使用示例**：

```python
# 模式1：values - 完整状态
for state in app.stream(input, config, stream_mode="values"):
    print(f"Current state: {state}")

# 模式2：updates - 增量更新
for node_name, update in app.stream(input, config, stream_mode="updates"):
    print(f"Node {node_name} updated: {update}")

# 模式3：debug - 调试信息
for debug_info in app.stream(input, config, stream_mode="debug"):
    print(f"Type: {debug_info['type']}")
    print(f"Node: {debug_info['node']}")
    print(f"Input: {debug_info['payload']['input']}")
    print(f"Output: {debug_info['payload']['output']}")
```

---

### 3.3 get_state

#### 基本信息

- **方法名称**: `get_state`
- **协议**: 同步方法
- **幂等性**: 是

#### 请求结构体

```python
def get_state(
    self,
    config: RunnableConfig,
) -> StateSnapshot:
    pass
```

#### 响应结构体

```python
class StateSnapshot(NamedTuple):
    """状态快照"""
    values: dict  # 当前状态值
    next: tuple[str, ...]  # 下一步将执行的节点
    config: RunnableConfig  # 快照的配置
    metadata: CheckpointMetadata  # 元数据
    created_at: str | None  # 创建时间
    parent_config: RunnableConfig | None  # 父快照配置
    tasks: tuple[PregelTask, ...]  # 待执行任务
```

#### 入口函数与核心代码

```python
def get_state(self, config: RunnableConfig) -> StateSnapshot:
    """
    获取当前状态快照
    
    功能：
    1. 从检查点加载状态
    2. 准备下一步任务（不执行）
    3. 构建StateSnapshot返回
    
    参数：
        config: 配置（thread_id和可选的checkpoint_id）
        
    返回：
        StateSnapshot: 状态快照
    """
    # 1. 加载检查点
    checkpoint_tuple = self.checkpointer.get_tuple(config)
    if not checkpoint_tuple:
        return StateSnapshot(
            values={},
            next=(),
            config=config,
            metadata={},
            created_at=None,
            parent_config=None,
            tasks=(),
        )
    
    # 2. 恢复通道
    channels = channels_from_checkpoint(
        checkpoint_tuple.checkpoint,
        self.pregel.channels,
    )
    
    # 3. 准备下一步任务（预览，不执行）
    tasks = prepare_next_tasks(
        checkpoint_tuple.checkpoint,
        checkpoint_tuple.pending_writes or [],
        self.pregel.nodes,
        channels,
        config=config,
        step=checkpoint_tuple.metadata.get("step", 0) + 1,
        stop=config.get("recursion_limit", 25),
        for_execution=False,  # 不执行，只预览
    )
    
    # 4. 读取当前值
    values = read_channels(channels, self.pregel.output_channels)
    
    # 5. 构建快照
    return StateSnapshot(
        values=values,
        next=tuple(task.name for task in tasks.values()),
        config=checkpoint_tuple.config,
        metadata=checkpoint_tuple.metadata,
        created_at=checkpoint_tuple.checkpoint.get("ts"),
        parent_config=checkpoint_tuple.parent_config,
        tasks=tuple(tasks.values()),
    )
```

---

### 3.4 update_state

#### 基本信息

- **方法名称**: `update_state`
- **协议**: 同步方法
- **幂等性**: 否（修改状态）

#### 请求结构体

```python
def update_state(
    self,
    config: RunnableConfig,
    values: dict | Sequence[dict],
    as_node: str | None = None,
) -> RunnableConfig:
    pass
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| config | RunnableConfig | 是 | 目标检查点配置 |
| values | dict \| Sequence[dict] | 是 | 要更新的值 |
| as_node | str \| None | 否 | 模拟为指定节点的更新 |

#### 响应结构体

返回`RunnableConfig`，包含新创建的checkpoint_id。

#### 入口函数与核心代码

```python
def update_state(
    self,
    config: RunnableConfig,
    values: dict | Sequence[dict],
    as_node: str | None = None,
) -> RunnableConfig:
    """
    手动更新状态
    
    功能：
    1. 加载当前检查点
    2. 应用更新到通道
    3. 创建新检查点（source="update"）
    4. 返回新配置
    
    参数：
        config: 目标配置
        values: 更新值，单个dict或列表
        as_node: 模拟节点更新（影响versions_seen）
        
    返回：
        新检查点的配置
    """
    # 1. 加载检查点
    checkpoint_tuple = self.checkpointer.get_tuple(config)
    if not checkpoint_tuple:
        raise ValueError("未找到检查点")
    
    # 2. 恢复通道
    channels = channels_from_checkpoint(
        checkpoint_tuple.checkpoint,
        self.pregel.channels,
    )
    
    # 3. 应用更新
    if not isinstance(values, list):
        values = [values]
    
    task = PregelTaskWrites(
        path=(),
        name=as_node or "__update__",
        writes=[(k, v) for update in values for k, v in update.items()],
        triggers=[],
    )
    
    updated_channels = apply_writes(
        checkpoint_tuple.checkpoint,
        channels,
        [task],
        self.checkpointer.get_next_version,
    )
    
    # 4. 更新versions_seen
    if as_node:
        checkpoint_tuple.checkpoint["versions_seen"][as_node] = {
            ch: checkpoint_tuple.checkpoint["channel_versions"][ch]
            for ch in updated_channels
        }
    
    # 5. 创建新检查点
    new_checkpoint = create_checkpoint(
        checkpoint_tuple.checkpoint,
        channels,
        checkpoint_tuple.metadata.get("step", 0),
    )
    
    # 6. 保存
    new_config = self.checkpointer.put(
        config,
        new_checkpoint,
        {"source": "update", "step": checkpoint_tuple.metadata.get("step", 0)},
        {ch: new_checkpoint["channel_versions"][ch] for ch in updated_channels},
    )
    
    return new_config
```

---

## 四、工具函数API

### 4.1 add_messages

#### 基本信息

- **函数名称**: `add_messages`
- **用途**: 消息列表的reducer函数
- **幂等性**: 否（累加消息）

#### 函数签名

```python
def add_messages(
    left: Sequence[AnyMessage],
    right: Sequence[AnyMessage] | AnyMessage,
) -> list[AnyMessage]:
    """
    合并消息列表
    
    功能：
    1. 将right中的消息添加到left
    2. 支持RemoveMessage删除消息
    3. 支持按ID更新消息
    
    参数：
        left: 现有消息列表
        right: 新消息或消息列表
        
    返回：
        合并后的消息列表
    """
    # 1. 统一为列表
    if not isinstance(right, list):
        right = [right]
    
    # 2. 复制left
    messages = list(left)
    
    # 3. 处理每个新消息
    for msg in right:
        if isinstance(msg, RemoveMessage):
            # 删除消息
            if msg.id == REMOVE_ALL_MESSAGES:
                messages.clear()
            else:
                messages = [m for m in messages if m.id != msg.id]
        else:
            # 添加或更新消息
            existing_idx = next(
                (i for i, m in enumerate(messages) if m.id == msg.id),
                None
            )
            if existing_idx is not None:
                # 更新现有消息
                messages[existing_idx] = msg
            else:
                # 添加新消息
                messages.append(msg)
    
    return messages
```

**使用示例**：

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 在节点中使用
def agent(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # 会被add_messages合并
```

---

## 五、总结

langgraph核心模块的API设计体现了以下特点：

1. **声明式API**：StateGraph提供简洁的图构建接口
2. **类型安全**：充分利用Python类型提示
3. **灵活的控制流**：支持固定边、条件边、动态任务
4. **Runnable接口**：编译后的图实现标准Runnable接口
5. **流式支持**：多种流式模式满足不同需求
6. **状态管理**：get_state和update_state支持精确控制

通过这些精心设计的API，开发者可以构建出强大而灵活的Agent系统。

