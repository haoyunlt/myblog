# OpenAI Agents Python SDK - Tools 模块时序图详解

## 1. 时序图总览

Tools 模块的时序图展示了工具从定义、注册到执行的完整生命周期，以及不同类型工具的执行流程和防护机制。

### 核心时序场景

| 场景类别 | 时序图 | 关键流程 |
|---------|--------|---------|
| **工具定义** | function_tool 装饰器流程 | 函数分析、Schema生成、工具创建 |
| **工具注册** | Agent 工具聚合流程 | 工具收集、去重、传递给模型 |
| **函数工具执行** | FunctionTool 调用流程 | 参数解析、防护检查、函数执行 |
| **托管工具执行** | FileSearchTool 调用流程 | 搜索请求、向量检索、结果返回 |
| **计算机工具执行** | ComputerTool 调用流程 | 操作请求、安全检查、执行操作 |
| **MCP工具执行** | HostedMCPTool 调用流程 | 工具列表、批准请求、远程执行 |

## 2. 工具定义时序图

### 场景：使用 function_tool 装饰器定义工具

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者代码
    participant Decorator as @function_tool装饰器
    participant Inspector as inspect模块
    participant SchemaGen as function_schema
    participant FuncTool as FunctionTool实例
    
    Dev->>Decorator: @function_tool<br/>def my_tool(param: str) -> str
    
    Decorator->>Inspector: inspect.signature(my_tool)
    Note over Inspector: 提取函数签名
    Inspector-->>Decorator: 参数列表、类型注解
    
    Decorator->>Inspector: inspect.getdoc(my_tool)
    Note over Inspector: 提取文档字符串
    Inspector-->>Decorator: "这是工具描述..."
    
    Decorator->>SchemaGen: function_schema(my_tool)
    Note over SchemaGen: 生成JSON Schema
    
    SchemaGen->>SchemaGen: 分析参数类型
    Note over SchemaGen: str → {"type": "string"}<br/>int → {"type": "integer"}
    
    SchemaGen->>SchemaGen: 解析文档字符串
    Note over SchemaGen: 提取参数描述<br/>Args: param: 参数说明
    
    SchemaGen->>SchemaGen: 检测上下文参数
    Note over SchemaGen: 识别RunContextWrapper<br/>或ToolContext参数
    
    SchemaGen-->>Decorator: params_json_schema
    Note over SchemaGen: {<br/>  "type": "object",<br/>  "properties": {...},<br/>  "required": [...]<br/>}
    
    Decorator->>Decorator: 创建包装函数
    Note over Decorator: on_invoke_tool =<br/>  lambda ctx, args: ...
    
    Decorator->>FuncTool: FunctionTool(<br/>  name="my_tool",<br/>  description="...",<br/>  params_json_schema={...},<br/>  on_invoke_tool=wrapper<br/>)
    
    FuncTool->>FuncTool: __post_init__()
    Note over FuncTool: 确保严格JSON Schema<br/>ensure_strict_json_schema()
    
    FuncTool-->>Dev: 返回 FunctionTool 实例
    
    Dev->>Dev: agent = Agent(tools=[my_tool])
    Note over Dev: 工具可直接传递给Agent
```

**时序图说明：**

### 工具定义阶段

1. **函数签名分析（步骤 1-4）**：
   - 使用 `inspect` 模块提取函数签名
   - 获取参数名称、类型注解、默认值

2. **文档提取（步骤 5-7）**：
   - 提取函数的文档字符串
   - 用作工具描述

3. **Schema 生成（步骤 8-15）**：
   - 根据类型注解生成 JSON Schema
   - 解析文档字符串获取参数描述
   - 识别并过滤上下文参数

4. **包装函数创建（步骤 16-17）**：
   - 创建调用包装器
   - 处理参数解析和上下文注入

5. **工具实例化（步骤 18-21）**：
   - 创建 `FunctionTool` 实例
   - 应用严格 JSON Schema
   - 返回可用的工具

### 类型映射规则

**Python类型 → JSON Schema：**
```python
str        → {"type": "string"}
int        → {"type": "integer"}
float      → {"type": "number"}
bool       → {"type": "boolean"}
list[str]  → {"type": "array", "items": {"type": "string"}}
dict       → {"type": "object"}
str | None → {"type": ["string", "null"]}  # 可选参数
```

## 3. 工具注册时序图

### 场景：Agent 聚合和准备工具

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant Agent as Agent实例
    participant Aggregator as 工具聚合器
    participant MCPServer as MCP服务器
    participant ToolRegistry as 工具注册表
    participant Model as 模型
    
    Dev->>Agent: Agent(<br/>  tools=[tool1, tool2],<br/>  mcp_servers=[mcp1]<br/>)
    
    Agent->>Aggregator: 启动工具聚合
    
    Aggregator->>Aggregator: 收集直接工具
    Note over Aggregator: tools=[tool1, tool2]
    
    Aggregator->>MCPServer: 连接 MCP 服务器
    Note over MCPServer: 建立连接
    
    Aggregator->>MCPServer: list_tools()
    Note over MCPServer: 请求工具列表
    
    MCPServer-->>Aggregator: [mcp_tool1, mcp_tool2]
    Note over MCPServer: 返回可用工具
    
    Aggregator->>Aggregator: 转换 MCP 工具
    Note over Aggregator: 包装为 FunctionTool
    
    Aggregator->>Aggregator: 检查工具启用状态
    
    loop 对每个工具
        Aggregator->>Aggregator: 检查 is_enabled
        
        alt is_enabled = True
            Aggregator->>Aggregator: 添加到活动列表
        
        else is_enabled = Callable
            Aggregator->>Aggregator: 暂时添加（运行时检查）
        
        else is_enabled = False
            Aggregator->>Aggregator: 跳过此工具
        end
    end
    
    Aggregator->>Aggregator: 去重工具
    Note over Aggregator: 按name去重
    
    Aggregator->>ToolRegistry: 注册所有工具
    Note over ToolRegistry: 建立 name → tool 映射
    
    Aggregator-->>Agent: 工具列表已就绪
    
    Agent->>Model: 准备模型调用
    Note over Agent: 将工具转换为<br/>模型可理解的格式
    
    Agent->>Agent: 转换工具定义
    Note over Agent: FunctionTool →<br/>{<br/>  "type": "function",<br/>  "function": {<br/>    "name": "tool_name",<br/>    "description": "...",<br/>    "parameters": {...}<br/>  }<br/>}
    
    Agent-->>Model: tools=[...]
    Note over Model: 模型知道可用工具
```

**时序图说明：**

### 工具聚合流程

1. **直接工具收集（步骤 1-4）**：
   - 收集 `Agent(tools=[...])` 中的工具
   - 验证工具类型

2. **MCP 工具集成（步骤 5-11）**：
   - 连接 MCP 服务器
   - 列出远程工具
   - 转换为本地工具格式

3. **启用状态检查（步骤 12-20）**：
   - 检查每个工具的 `is_enabled`
   - 过滤禁用的工具
   - 保留动态启用的工具

4. **去重和注册（步骤 21-24）**：
   - 按工具名称去重
   - 注册到工具注册表

5. **模型格式转换（步骤 25-31）**：
   - 转换为模型 API 格式
   - 传递给模型

### 工具定义格式

**发送给模型的格式：**
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名称"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "温度单位"
        }
      },
      "required": ["city"],
      "additionalProperties": false
    }
  }
}
```

## 4. FunctionTool 执行时序图

### 场景：完整的函数工具调用流程

```mermaid
sequenceDiagram
    autonumber
    participant Model as 模型
    participant RunImpl as RunImpl执行引擎
    participant ToolRegistry as 工具注册表
    participant InputGuard as 输入防护
    participant Enabled as 启用检查
    participant Wrapper as 包装函数
    participant UserFunc as 用户函数
    participant OutputGuard as 输出防护
    participant Result as 结果处理
    
    Model-->>RunImpl: ModelResponse<br/>{tool_calls: [{<br/>  name: "get_weather",<br/>  arguments: '{"city":"Beijing"}'<br/>}]}
    
    RunImpl->>ToolRegistry: 查找工具 "get_weather"
    ToolRegistry-->>RunImpl: FunctionTool实例
    
    RunImpl->>RunImpl: 解析参数 JSON
    Note over RunImpl: json.loads(arguments)
    
    alt JSON 解析失败
        RunImpl->>Result: 创建错误输出
        Note over Result: "参数格式错误"
        Result-->>Model: 返回错误消息
    end
    
    RunImpl->>RunImpl: 验证参数类型
    Note over RunImpl: 根据 params_json_schema<br/>验证参数
    
    alt 参数验证失败
        RunImpl->>Result: 创建验证错误
        Note over Result: "参数不符合要求"
        Result-->>Model: 返回错误消息
    end
    
    RunImpl->>InputGuard: 运行输入防护
    Note over InputGuard: tool_input_guardrails
    
    loop 对每个输入防护
        InputGuard->>InputGuard: 检查参数安全性
        
        alt 防护触发拒绝
            InputGuard-->>RunImpl: RejectContentBehavior
            RunImpl->>Result: 创建拒绝输出
            Result-->>Model: 返回拒绝消息
            Note over Model: 不执行工具
        
        else 防护触发异常
            InputGuard-->>RunImpl: RaiseExceptionBehavior
            RunImpl-->>RunImpl: 抛出异常
            Note over RunImpl: 中断整个执行
        end
    end
    
    InputGuard-->>RunImpl: AllowBehavior（通过）
    
    RunImpl->>Enabled: 检查工具启用状态
    
    alt is_enabled = Callable
        Enabled->>Enabled: 调用启用检查函数
        Note over Enabled: is_enabled(context, agent)
        
        alt 返回 False
            Enabled-->>RunImpl: 工具未启用
            RunImpl->>Result: 创建禁用消息
            Result-->>Model: "工具当前不可用"
        end
    end
    
    Enabled-->>RunImpl: 工具已启用
    
    RunImpl->>Wrapper: 调用 on_invoke_tool(ctx, args_json)
    
    Wrapper->>Wrapper: 准备函数参数
    Note over Wrapper: 解析 JSON<br/>注入上下文（如需要）
    
    Wrapper->>UserFunc: 调用用户函数<br/>my_tool(city="Beijing")
    
    UserFunc->>UserFunc: 执行业务逻辑
    Note over UserFunc: 调用API、查询数据库等
    
    alt 函数执行成功
        UserFunc-->>Wrapper: 返回结果<br/>"北京：晴天，22°C"
        
        Wrapper->>Wrapper: 转换为字符串
        Note over Wrapper: str(result)
        
        Wrapper-->>RunImpl: 工具输出字符串
        
        RunImpl->>OutputGuard: 运行输出防护
        Note over OutputGuard: tool_output_guardrails
        
        loop 对每个输出防护
            OutputGuard->>OutputGuard: 检查输出安全性
            
            alt 防护修改内容
                OutputGuard-->>RunImpl: AllowBehavior<br/>(modified_content="...")
                RunImpl->>RunImpl: 使用修改后的内容
            
            else 防护拒绝内容
                OutputGuard-->>RunImpl: RejectContentBehavior
                RunImpl->>Result: 创建拒绝输出
            end
        end
        
        OutputGuard-->>RunImpl: AllowBehavior（通过）
        
        RunImpl->>Result: 创建 FunctionToolResult
        Note over Result: tool, output, run_item
        
        RunImpl->>Result: 创建 ToolCallOutputItem
        Note over Result: success=True<br/>output="..."
        
        Result-->>Model: 返回工具结果
    
    else 函数执行失败
        UserFunc-->>Wrapper: 抛出异常
        
        Wrapper->>Wrapper: 捕获异常
        
        alt 有自定义错误处理
            Wrapper->>Wrapper: 调用 failure_error_function
            Wrapper-->>RunImpl: 错误消息字符串
        
        else 使用默认错误处理
            Wrapper-->>RunImpl: "工具执行失败: {error}"
        end
        
        RunImpl->>Result: 创建 ToolCallOutputItem
        Note over Result: success=False<br/>output="错误消息"
        
        Result-->>Model: 返回错误结果
    end
    
    Model->>Model: 理解工具结果
    Note over Model: 基于结果生成回复
```

**时序图说明：**

### 工具执行阶段

1. **参数处理（步骤 1-11）**：
   - 解析参数 JSON
   - 验证参数类型和格式
   - 处理解析或验证错误

2. **输入防护（步骤 12-23）**：
   - 运行所有输入防护
   - 处理拒绝或异常行为
   - 通过后继续执行

3. **启用检查（步骤 24-32）**：
   - 检查工具启用状态
   - 动态启用检查
   - 处理禁用情况

4. **函数执行（步骤 33-43）**：
   - 准备函数参数
   - 注入上下文（如需要）
   - 调用实际函数
   - 执行业务逻辑

5. **输出防护（步骤 44-56）**：
   - 运行所有输出防护
   - 处理内容修改或拒绝
   - 通过后返回结果

6. **结果封装（步骤 57-66）**：
   - 创建工具结果对象
   - 创建运行项
   - 返回给模型

### 错误处理路径

**参数错误：**
```
JSON 解析失败 → 错误消息 → 返回模型
参数验证失败 → 错误消息 → 返回模型
```

**防护拒绝：**
```
输入防护拒绝 → 拒绝消息 → 返回模型（不执行）
输出防护拒绝 → 拒绝消息 → 返回模型
```

**执行异常：**
```
函数抛出异常 → 捕获并转换 → 错误结果 → 返回模型
防护抛出异常 → 传播异常 → 中断整个执行
```

## 5. FileSearchTool 执行时序图

### 场景：托管文件搜索工具调用

```mermaid
sequenceDiagram
    autonumber
    participant Model as 模型
    participant RunImpl as RunImpl
    participant FileSearch as FileSearchTool
    participant OpenAI as OpenAI API
    participant VectorStore as 向量存储服务
    
    Model-->>RunImpl: 决定搜索文档
    Note over Model: 分析用户问题<br/>需要文档支持
    
    RunImpl->>FileSearch: 准备搜索请求
    Note over FileSearch: 提取搜索查询
    
    RunImpl->>OpenAI: file_search.query(<br/>  vector_store_ids=[...],<br/>  query="产品定价策略",<br/>  max_results=10<br/>)
    
    OpenAI->>VectorStore: 向量搜索
    Note over VectorStore: 语义搜索相关文档
    
    VectorStore->>VectorStore: 计算相似度
    Note over VectorStore: 查询嵌入 vs 文档嵌入
    
    VectorStore->>VectorStore: 应用过滤器
    Note over VectorStore: filters: {<br/>  metadata: {...},<br/>  file_ids: [...]<br/>}
    
    VectorStore->>VectorStore: 排序结果
    Note over VectorStore: 按相关性排序<br/>应用 ranking_options
    
    VectorStore-->>OpenAI: 搜索结果
    Note over VectorStore: [<br/>  {file_id, content, score},<br/>  ...<br/>]
    
    OpenAI->>OpenAI: 格式化结果
    Note over OpenAI: 准备文档片段
    
    alt include_search_results = True
        OpenAI-->>RunImpl: 搜索结果 + 文档内容
        Note over OpenAI: 在响应中包含完整结果
    else include_search_results = False
        OpenAI-->>RunImpl: 仅搜索摘要
        Note over OpenAI: 隐式使用结果
    end
    
    RunImpl->>RunImpl: 创建 ToolCallOutputItem
    Note over RunImpl: 记录搜索结果
    
    RunImpl->>Model: 提供搜索结果
    Note over RunImpl: 作为工具输出返回
    
    Model->>Model: 基于文档生成回答
    Note over Model: 引用搜索到的文档<br/>回答用户问题
    
    Model-->>RunImpl: 最终回复
    Note over Model: "根据文档，<br/>产品定价策略是..."
```

**时序图说明：**

### 文件搜索流程

1. **搜索决策（步骤 1-3）**：
   - 模型判断需要搜索文档
   - 准备搜索查询

2. **向量搜索（步骤 4-11）**：
   - 调用 OpenAI API
   - 向量存储进行语义搜索
   - 计算查询与文档的相似度

3. **结果过滤和排序（步骤 12-15）**：
   - 应用元数据过滤器
   - 按相关性排序
   - 限制结果数量

4. **结果返回（步骤 16-24）**：
   - 格式化搜索结果
   - 根据配置返回内容
   - 模型基于结果生成回答

### 搜索配置影响

**max_num_results：**
- 控制返回结果数量
- 影响上下文大小和精确度

**include_search_results：**
- `True`：在输出中包含完整结果
- `False`：隐式使用，不显示给用户

**ranking_options：**
- 自定义排序算法
- 设置相关性阈值

## 6. ComputerTool 执行时序图

### 场景：计算机控制工具执行操作

```mermaid
sequenceDiagram
    autonumber
    participant Model as 模型
    participant RunImpl as RunImpl
    participant ComputerTool as ComputerTool
    participant SafetyCheck as 安全检查回调
    participant Computer as Computer实现
    participant OS as 操作系统
    
    Model-->>RunImpl: 请求计算机操作
    Note over Model: {<br/>  action: "mouse_move",<br/>  x: 500,<br/>  y: 300<br/>}
    
    RunImpl->>ComputerTool: 处理操作请求
    
    ComputerTool->>ComputerTool: 解析操作类型
    Note over ComputerTool: 识别操作：<br/>mouse_move, click,<br/>type, screenshot
    
    alt 有安全检查回调
        ComputerTool->>SafetyCheck: on_safety_check(data)
        Note over SafetyCheck: 检查操作是否安全
        
        SafetyCheck->>SafetyCheck: 评估操作
        Note over SafetyCheck: 检查坐标范围<br/>检查操作类型<br/>检查频率
        
        alt 操作被拒绝
            SafetyCheck-->>ComputerTool: return False
            ComputerTool-->>RunImpl: 操作被拒绝
            RunImpl-->>Model: "安全检查失败"
        end
        
        SafetyCheck-->>ComputerTool: return True（允许）
    end
    
    ComputerTool->>Computer: 调用相应方法
    
    alt 操作类型: screenshot
        Computer->>OS: 截取屏幕
        OS-->>Computer: 屏幕图像数据
        Computer-->>ComputerTool: bytes (PNG/JPEG)
        
    else 操作类型: mouse_move
        Computer->>Computer: 验证坐标
        Note over Computer: x < display_width_px<br/>y < display_height_px
        
        Computer->>OS: 移动鼠标指针
        OS-->>Computer: 操作完成
        Computer-->>ComputerTool: 成功
        
    else 操作类型: mouse_click
        Computer->>OS: 执行鼠标点击
        OS-->>Computer: 操作完成
        Computer-->>ComputerTool: 成功
        
    else 操作类型: keyboard_type
        Computer->>OS: 模拟键盘输入
        Note over OS: 输入文本内容
        OS-->>Computer: 操作完成
        Computer-->>ComputerTool: 成功
    end
    
    ComputerTool->>ComputerTool: 记录操作结果
    
    ComputerTool-->>RunImpl: 操作结果
    Note over ComputerTool: 成功消息或错误
    
    RunImpl->>RunImpl: 创建 ToolCallOutputItem
    
    RunImpl-->>Model: 返回结果
    Note over Model: "操作已完成"<br/>或 "截图已获取"
    
    Model->>Model: 理解操作结果
    Note over Model: 决定下一步操作<br/>或生成回复
```

**时序图说明：**

### 计算机操作流程

1. **操作解析（步骤 1-4）**：
   - 模型请求计算机操作
   - 解析操作类型和参数

2. **安全检查（步骤 5-14）**：
   - 调用安全检查回调
   - 评估操作的安全性
   - 拒绝危险操作

3. **操作执行（步骤 15-34）**：
   - 根据操作类型调用相应方法
   - 与操作系统交互
   - 返回操作结果

4. **结果处理（步骤 35-40）**：
   - 记录操作日志
   - 创建运行项
   - 返回给模型

### 支持的操作类型

**鼠标操作：**
- `mouse_move(x, y)`: 移动鼠标
- `mouse_click(button)`: 点击鼠标
- `mouse_drag(from_x, from_y, to_x, to_y)`: 拖动

**键盘操作：**
- `keyboard_type(text)`: 输入文本
- `keyboard_press(key)`: 按键

**屏幕操作：**
- `screenshot()`: 截取屏幕
- `screenshot_region(x, y, width, height)`: 区域截图

## 7. HostedMCPTool 执行时序图

### 场景：托管 MCP 工具调用和批准流程

```mermaid
sequenceDiagram
    autonumber
    participant Model as 模型
    participant RunImpl as RunImpl
    participant MCPTool as HostedMCPTool
    participant Approval as 批准回调
    participant MCPServer as MCP服务器
    
    Model-->>RunImpl: 请求使用 MCP 工具
    Note over Model: 决定调用远程工具
    
    RunImpl->>MCPTool: 处理工具调用
    
    MCPTool->>MCPServer: 发送工具调用请求
    Note over MCPServer: {<br/>  tool_name: "read_file",<br/>  arguments: {<br/>    path: "/data/file.txt"<br/>  }<br/>}
    
    MCPServer->>MCPServer: 处理请求
    
    alt 服务器要求批准
        MCPServer-->>MCPTool: ApprovalRequired
        Note over MCPServer: {<br/>  approval_needed: true,<br/>  tool_name: "read_file",<br/>  reason: "访问文件系统"<br/>}
        
        MCPTool->>MCPTool: 检查批准回调
        
        alt 有批准回调
            MCPTool->>Approval: on_approval_request(request)
            Note over Approval: 评估请求安全性
            
            Approval->>Approval: 检查工具名称
            Note over Approval: 白名单检查
            
            Approval->>Approval: 检查参数
            Note over Approval: 路径安全性检查
            
            Approval->>Approval: 检查用户权限
            Note over Approval: 权限验证
            
            alt 批准请求
                Approval-->>MCPTool: {<br/>  approve: True<br/>}
                
                MCPTool->>MCPServer: 发送批准
                Note over MCPServer: 继续执行工具
                
                MCPServer->>MCPServer: 执行工具逻辑
                Note over MCPServer: 读取文件内容
                
                MCPServer-->>MCPTool: 工具结果
                Note over MCPServer: {<br/>  success: true,<br/>  output: "文件内容..."<br/>}
                
                MCPTool-->>RunImpl: 返回结果
                RunImpl-->>Model: 工具输出
            
            else 拒绝请求
                Approval-->>MCPTool: {<br/>  approve: False,<br/>  reason: "权限不足"<br/>}
                
                MCPTool->>MCPServer: 发送拒绝
                Note over MCPServer: 取消工具执行
                
                MCPServer-->>MCPTool: 拒绝确认
                
                MCPTool-->>RunImpl: 工具被拒绝
                Note over MCPTool: 返回拒绝原因
                
                RunImpl-->>Model: "工具调用被拒绝：权限不足"
            end
        
        else 无批准回调
            MCPTool-->>RunImpl: 需要手动批准
            Note over MCPTool: 返回批准请求<br/>等待用户处理
            
            RunImpl-->>Model: 返回批准请求
            Note over Model: 通知用户需要批准
        end
    
    else 服务器直接执行
        MCPServer->>MCPServer: 执行工具
        Note over MCPServer: 不需要批准的工具
        
        MCPServer-->>MCPTool: 工具结果
        MCPTool-->>RunImpl: 返回结果
        RunImpl-->>Model: 工具输出
    end
```

**时序图说明：**

### MCP 工具调用流程

1. **请求发送（步骤 1-4）**：
   - 模型请求 MCP 工具
   - 发送到远程服务器

2. **批准检查（步骤 5-13）**：
   - 服务器判断是否需要批准
   - 返回批准请求

3. **批准决策（步骤 14-30）**：
   - 调用批准回调
   - 评估请求安全性
   - 做出批准或拒绝决定

4. **工具执行（步骤 31-42）**：
   - 批准后执行工具
   - 拒绝则取消执行
   - 返回结果或错误

### 批准决策逻辑

**自动批准：**
```python
# 白名单工具
whitelist = ["read_file", "list_directory"]
if tool_name in whitelist:
    return {"approve": True}
```

**条件批准：**
```python
# 检查路径安全性
if "path" in arguments:
    if is_safe_path(arguments["path"]):
        return {"approve": True}
    else:
        return {"approve": False, "reason": "路径不安全"}
```

**需要人工批准：**
```python
# 敏感操作
sensitive_tools = ["delete_file", "modify_config"]
if tool_name in sensitive_tools:
    # 不提供批准回调，需要用户手动批准
    return None
```

## 8. 工具执行流程总览

```mermaid
flowchart TB
    START([模型决策使用工具])
    
    subgraph "工具查找"
        LOOKUP[查找工具定义]
        FOUND{找到工具?}
        ERROR_NOTFOUND[返回未找到错误]
    end
    
    subgraph "参数处理"
        PARSE[解析参数JSON]
        VALIDATE[验证参数类型]
        ERROR_PARAM[返回参数错误]
    end
    
    subgraph "防护检查"
        INPUT_GUARD[运行输入防护]
        GUARD_DECISION{防护结果?}
        ERROR_GUARD[返回拒绝消息]
    end
    
    subgraph "启用检查"
        ENABLED_CHECK[检查启用状态]
        IS_ENABLED{工具启用?}
        ERROR_DISABLED[返回禁用消息]
    end
    
    subgraph "工具执行"
        EXECUTE[执行工具函数]
        SUCCESS{执行成功?}
        OUTPUT_GUARD[运行输出防护]
        OUTPUT_DECISION{防护结果?}
    end
    
    subgraph "结果处理"
        CREATE_RESULT[创建FunctionToolResult]
        CREATE_ITEM[创建ToolCallOutputItem]
        RETURN[返回给模型]
    end
    
    END([模型处理结果])
    
    START --> LOOKUP
    LOOKUP --> FOUND
    
    FOUND -->|找到| PARSE
    FOUND -->|未找到| ERROR_NOTFOUND
    ERROR_NOTFOUND --> END
    
    PARSE --> VALIDATE
    VALIDATE -->|成功| INPUT_GUARD
    VALIDATE -->|失败| ERROR_PARAM
    ERROR_PARAM --> END
    
    INPUT_GUARD --> GUARD_DECISION
    GUARD_DECISION -->|通过| ENABLED_CHECK
    GUARD_DECISION -->|拒绝| ERROR_GUARD
    ERROR_GUARD --> END
    
    ENABLED_CHECK --> IS_ENABLED
    IS_ENABLED -->|启用| EXECUTE
    IS_ENABLED -->|禁用| ERROR_DISABLED
    ERROR_DISABLED --> END
    
    EXECUTE --> SUCCESS
    SUCCESS -->|成功| OUTPUT_GUARD
    SUCCESS -->|失败| CREATE_ITEM
    
    OUTPUT_GUARD --> OUTPUT_DECISION
    OUTPUT_DECISION -->|通过| CREATE_RESULT
    OUTPUT_DECISION -->|拒绝| ERROR_GUARD
    
    CREATE_RESULT --> CREATE_ITEM
    CREATE_ITEM --> RETURN
    RETURN --> END
    
    style START fill:#e8f5e9
    style END fill:#e8f5e9
    style ERROR_NOTFOUND fill:#ffebee
    style ERROR_PARAM fill:#ffebee
    style ERROR_GUARD fill:#ffebee
    style ERROR_DISABLED fill:#ffebee
    style EXECUTE fill:#e1f5fe
```

Tools 模块通过精心设计的时序流程和完善的防护机制，为 OpenAI Agents 提供了安全、可靠的工具执行能力，支持从简单函数调用到复杂系统集成的各种应用场景。

