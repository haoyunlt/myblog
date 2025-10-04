# OpenAI Agents Python SDK - Models 模块时序图详解

## 1. 时序图总览

Models 模块的时序图展示了模型选择、调用、响应处理的完整流程。核心流程包括：模型提供商初始化、模型实例获取、标准调用、流式调用。

### 主要时序流程

| 时序流程 | 参与者 | 核心操作 |
|---------|--------|---------|
| **模型提供商初始化** | MultiProvider, OpenAIProvider | 配置提供商映射 |
| **模型实例获取** | MultiProvider, ModelProvider, Model | 路由并创建模型 |
| **标准模型调用** | Agent, Model, LLM API | 完整请求响应 |
| **流式模型调用** | Agent, Model, LLM API | 流式数据处理 |
| **多提供商路由** | MultiProvider, ProviderMap | 前缀匹配与查找 |

## 2. 模型提供商初始化时序图

### 2.1 MultiProvider 初始化流程

```mermaid
sequenceDiagram
    participant U as 用户代码
    participant MP as MultiProvider
    participant PM as MultiProviderMap
    participant OP as OpenAIProvider
    participant Client as AsyncOpenAI
    
    U->>MP: __init__(provider_map, openai_api_key, ...)
    activate MP
    
    Note over MP: 保存自定义provider_map
    MP->>MP: self.provider_map = provider_map
    
    Note over MP: 初始化OpenAI Provider
    MP->>OP: __init__(api_key, base_url, ...)
    activate OP
    
    alt 提供了openai_client
        OP->>OP: self.client = openai_client
    else 未提供
        OP->>Client: AsyncOpenAI(api_key, base_url, ...)
        Client-->>OP: client实例
        OP->>OP: self.client = client
    end
    
    OP->>OP: self.use_responses = use_responses
    OP->>OP: self._models = {}
    OP-->>MP: OpenAIProvider实例
    deactivate OP
    
    MP->>MP: self.openai_provider = openai_provider
    MP->>MP: self._fallback_providers = {}
    
    MP-->>U: MultiProvider实例
    deactivate MP
```

**流程说明：**

1. **创建MultiProvider**：用户传入配置参数
2. **保存自定义映射**：存储provider_map用于前缀路由
3. **初始化OpenAI Provider**：
   - 如果提供了自定义客户端，直接使用
   - 否则创建新的AsyncOpenAI客户端
4. **初始化缓存**：准备fallback providers字典
5. **返回实例**：MultiProvider准备就绪

### 2.2 自定义提供商注册流程

```mermaid
sequenceDiagram
    participant U as 用户代码
    participant PM as MultiProviderMap
    participant MP as MultiProvider
    participant CP as CustomProvider
    
    U->>PM: MultiProviderMap()
    activate PM
    PM->>PM: self._mapping = {}
    PM-->>U: map实例
    deactivate PM
    
    U->>CP: CustomProvider(config)
    activate CP
    CP->>CP: 初始化自定义配置
    CP-->>U: provider实例
    deactivate CP
    
    U->>PM: add_provider("custom", provider)
    activate PM
    PM->>PM: self._mapping["custom"] = provider
    PM-->>U: None
    deactivate PM
    
    U->>MP: MultiProvider(provider_map=map)
    activate MP
    MP->>MP: self.provider_map = map
    MP-->>U: MultiProvider实例
    deactivate MP
```

**流程说明：**

1. **创建映射对象**：初始化空的提供商映射
2. **创建自定义提供商**：实现ModelProvider接口
3. **注册提供商**：添加前缀到映射
4. **创建MultiProvider**：使用自定义映射

## 3. 模型实例获取时序图

### 3.1 标准模型获取流程

```mermaid
sequenceDiagram
    participant A as Agent
    participant MP as MultiProvider
    participant PM as MultiProviderMap
    participant OP as OpenAIProvider
    participant M as Model
    
    A->>MP: get_model("gpt-4o")
    activate MP
    
    Note over MP: 解析模型名称
    MP->>MP: _get_prefix_and_model_name("gpt-4o")
    MP->>MP: prefix=None, name="gpt-4o"
    
    Note over MP: 检查自定义映射
    alt provider_map存在且有前缀
        MP->>PM: get_provider(prefix)
        PM-->>MP: None (无匹配)
    end
    
    Note over MP: 使用fallback
    MP->>MP: _get_fallback_provider(None)
    MP->>MP: return self.openai_provider
    
    Note over MP: 从提供商获取模型
    MP->>OP: get_model("gpt-4o")
    activate OP
    
    alt 模型已缓存
        OP->>OP: 从_models缓存获取
    else 首次获取
        alt use_responses=True
            OP->>M: OpenAIResponsesModel(client, "gpt-4o")
        else
            OP->>M: OpenAIChatCompletionsModel(client, "gpt-4o")
        end
        M-->>OP: model实例
        OP->>OP: _models["gpt-4o"] = model
    end
    
    OP-->>MP: Model实例
    deactivate OP
    
    MP-->>A: Model实例
    deactivate MP
```

**流程说明：**

1. **接收请求**：Agent请求模型实例
2. **解析名称**：提取前缀和实际名称
3. **查找提供商**：
   - 先检查自定义映射
   - 再使用fallback提供商
4. **获取模型**：
   - 检查缓存
   - 创建新实例（根据use_responses选择）
5. **返回模型**：准备好的模型实例

### 3.2 带前缀的模型获取流程

```mermaid
sequenceDiagram
    participant A as Agent
    participant MP as MultiProvider
    participant PM as MultiProviderMap
    participant LP as LitellmProvider
    participant M as LitellmModel
    
    A->>MP: get_model("litellm/claude-3-opus")
    activate MP
    
    Note over MP: 解析模型名称
    MP->>MP: _get_prefix_and_model_name()
    MP->>MP: prefix="litellm", name="claude-3-opus"
    
    Note over MP: 检查自定义映射
    alt provider_map存在
        MP->>PM: get_provider("litellm")
        alt 映射中存在
            PM-->>MP: LitellmProvider
        else 映射中不存在
            PM-->>MP: None
        end
    end
    
    alt 自定义映射中未找到
        Note over MP: 使用fallback创建
        MP->>MP: _get_fallback_provider("litellm")
        
        alt litellm在_fallback_providers缓存
            MP->>MP: return _fallback_providers["litellm"]
        else 首次创建
            MP->>MP: _create_fallback_provider("litellm")
            MP->>LP: LitellmProvider()
            LP-->>MP: provider实例
            MP->>MP: _fallback_providers["litellm"] = provider
        end
    end
    
    Note over MP: 从提供商获取模型
    MP->>LP: get_model("claude-3-opus")
    activate LP
    LP->>M: 创建LitellmModel
    M-->>LP: model实例
    LP-->>MP: Model实例
    deactivate LP
    
    MP-->>A: Model实例
    deactivate MP
```

**流程说明：**

1. **解析前缀**：识别"litellm"前缀
2. **查找提供商**：
   - 先查自定义映射
   - 再查fallback缓存
   - 最后动态创建
3. **获取模型**：从对应提供商获取
4. **返回实例**：返回实际模型

## 4. 标准模型调用时序图

### 4.1 完整的get_response流程

```mermaid
sequenceDiagram
    participant R as Runner
    participant M as Model
    participant Conv as Converter
    participant API as LLM API
    participant Parser as ResponseParser
    
    R->>M: get_response(instructions, input, settings, tools, ...)
    activate M
    
    Note over M: 1. 准备请求数据
    M->>Conv: 转换为API格式
    activate Conv
    Conv->>Conv: 格式化system_instructions
    Conv->>Conv: 转换input items
    Conv->>Conv: 转换tools定义
    Conv->>Conv: 应用model_settings
    Conv-->>M: API请求数据
    deactivate Conv
    
    Note over M: 2. 调用LLM API
    M->>API: POST /chat/completions
    activate API
    
    Note over API: LLM处理
    API->>API: 理解输入
    API->>API: 生成响应
    
    alt 需要调用工具
        API->>API: 生成function_call
    else 普通响应
        API->>API: 生成文本
    end
    
    API-->>M: 原始响应
    deactivate API
    
    Note over M: 3. 解析响应
    M->>Parser: parse_response(raw_response)
    activate Parser
    
    Parser->>Parser: 提取response_id
    Parser->>Parser: 解析output items
    
    loop 每个output item
        alt message类型
            Parser->>Parser: 创建ResponseTextItem
        else function_call类型
            Parser->>Parser: 创建ResponseFunctionCallItem
        else audio类型
            Parser->>Parser: 创建ResponseAudioItem
        end
    end
    
    Parser->>Parser: 提取usage统计
    Parser->>Parser: 保存raw_response
    
    Parser-->>M: ModelResponse
    deactivate Parser
    
    Note over M: 4. 返回标准化响应
    M-->>R: ModelResponse(response_id, output, usage, raw)
    deactivate M
```

**流程说明：**

1. **准备请求**：
   - 格式化系统指令
   - 转换输入项为API格式
   - 转换工具定义
   - 应用模型设置
2. **调用API**：发送HTTP请求到LLM服务
3. **解析响应**：
   - 提取响应ID
   - 解析输出项（文本、工具调用等）
   - 统计Token使用
4. **返回结果**：标准化的ModelResponse

### 4.2 带追踪的模型调用流程

```mermaid
sequenceDiagram
    participant R as Runner
    participant T as TracingProcessor
    participant M as Model
    participant API as LLM API
    
    R->>R: 检查tracing配置
    
    alt tracing.is_disabled() = False
        R->>T: on_span_start(generation_span)
        activate T
        T->>T: 记录span开始
        T-->>R: None
        deactivate T
    end
    
    R->>M: get_response(..., tracing=tracing)
    activate M
    
    alt tracing.include_data() = True
        M->>M: 记录完整输入数据
    else
        M->>M: 只记录元数据
    end
    
    M->>API: 调用API
    API-->>M: 响应
    
    alt tracing.include_data() = True
        M->>M: 记录完整输出数据
    else
        M->>M: 只记录元数据
    end
    
    M-->>R: ModelResponse
    deactivate M
    
    alt tracing.is_disabled() = False
        R->>T: on_span_end(generation_span, response)
        activate T
        T->>T: 记录span结束
        T->>T: 计算耗时
        T->>T: 记录usage
        T-->>R: None
        deactivate T
    end
```

**追踪流程说明：**

1. **检查追踪配置**：确定追踪级别
2. **记录开始**：创建generation span
3. **条件记录**：根据include_data决定记录详细程度
4. **记录结束**：完成span，记录统计信息

## 5. 流式模型调用时序图

### 5.1 stream_response流程

```mermaid
sequenceDiagram
    participant R as Runner
    participant M as Model
    participant API as LLM API Stream
    participant H as EventHandler
    participant A as Agent
    
    R->>M: stream_response(instructions, input, settings, ...)
    activate M
    
    Note over M: 准备流式请求
    M->>M: request_data["stream"] = True
    
    M->>API: POST /chat/completions (stream)
    activate API
    
    Note over API: 开始流式生成
    
    loop 流式chunk
        API-->>M: SSE chunk
        
        M->>M: 解析chunk
        
        alt text.delta事件
            M->>M: 创建ResponseTextDelta
            M->>H: yield event
            H->>A: 更新UI（增量文本）
            
        else function_call_arguments.delta
            M->>M: 创建FunctionCallArgsDelta
            M->>H: yield event
            H->>H: 累积参数字符串
            
        else output_item.done
            M->>M: 创建OutputItemDone
            M->>H: yield event
            H->>H: 完成输出项
            
        else audio.delta
            M->>M: 创建AudioDelta
            M->>H: yield event
            H->>A: 播放音频片段
            
        else response.done
            M->>M: 创建ResponseDone
            M->>H: yield event
            H->>H: 生成最终ModelResponse
        end
    end
    
    API-->>M: Stream结束
    deactivate API
    
    M-->>R: AsyncIterator完成
    deactivate M
```

**流式处理说明：**

1. **启用流式**：设置stream=True
2. **接收chunks**：持续接收SSE事件
3. **分类处理**：
   - 文本增量：实时显示
   - 工具参数：累积解析
   - 输出完成：标记结束
   - 音频数据：实时播放
4. **完成流式**：发送done事件

### 5.2 流式事件聚合流程

```mermaid
sequenceDiagram
    participant S as StreamHandler
    participant B as Buffer
    participant P as Parser
    
    Note over S: 初始化
    S->>B: 创建文本缓冲区
    S->>B: 创建参数缓冲区
    
    loop 处理流式事件
        S->>S: 接收event
        
        alt response.text.delta
            S->>B: append_text(delta, content_index)
            B->>B: text_buffers[index] += delta
            
        else response.function_call_arguments.delta
            S->>B: append_args(delta, call_id)
            B->>B: args_buffers[call_id] += delta
            
        else response.output_item.done
            S->>B: get_complete_text(content_index)
            B-->>S: 完整文本
            
            alt 是function_call
                S->>B: get_complete_args(call_id)
                B-->>S: 完整参数字符串
                
                S->>P: parse_json(args_string)
                P-->>S: 参数对象
            end
            
            S->>S: 创建完整OutputItem
            
        else response.done
            S->>S: 生成最终ModelResponse
            S->>S: 清空所有缓冲区
        end
    end
```

**聚合流程说明：**

1. **初始化缓冲区**：为文本和参数创建缓冲
2. **累积增量**：将delta添加到对应缓冲区
3. **完成输出项**：
   - 获取完整内容
   - 解析JSON参数
   - 创建完整对象
4. **清理缓冲**：响应完成后清空

## 6. 多提供商路由时序图

### 6.1 完整路由流程

```mermaid
sequenceDiagram
    participant A as Agent
    participant MP as MultiProvider
    participant PM as ProviderMap
    participant Cache as FallbackCache
    participant P1 as OpenAIProvider
    participant P2 as CustomProvider
    participant M as Model
    
    A->>MP: get_model("custom/my-model")
    activate MP
    
    Note over MP: Step 1: 解析模型名称
    MP->>MP: _get_prefix_and_model_name()
    MP->>MP: prefix="custom", name="my-model"
    
    Note over MP: Step 2: 检查自定义映射
    alt provider_map存在
        MP->>PM: get_provider("custom")
        activate PM
        
        alt 前缀已注册
            PM->>PM: return _mapping["custom"]
            PM-->>MP: CustomProvider
            deactivate PM
            
            Note over MP: 使用自定义提供商
            MP->>P2: get_model("my-model")
            activate P2
            P2->>M: 创建CustomModel
            M-->>P2: model实例
            P2-->>MP: Model
            deactivate P2
            
        else 前缀未注册
            PM-->>MP: None
            deactivate PM
            
            Note over MP: Step 3: 使用fallback
            MP->>Cache: 查找_fallback_providers["custom"]
            
            alt 缓存存在
                Cache-->>MP: CustomProvider
            else 需要创建
                MP->>MP: _create_fallback_provider("custom")
                MP->>P2: CustomProvider()
                P2-->>MP: provider
                MP->>Cache: _fallback_providers["custom"] = provider
            end
            
            MP->>P2: get_model("my-model")
            P2->>M: 创建Model
            M-->>P2: model实例
            P2-->>MP: Model
        end
        
    else provider_map为None
        Note over MP: 直接使用fallback
        MP->>MP: _get_fallback_provider("custom")
        
        alt prefix=None或"openai"
            MP->>P1: get_model(name)
            P1-->>MP: OpenAIModel
        else 其他prefix
            MP->>Cache: 查找或创建provider
            Cache-->>MP: Provider
            MP->>P2: get_model(name)
            P2-->>MP: Model
        end
    end
    
    MP-->>A: Model实例
    deactivate MP
```

**路由决策流程：**

1. **解析阶段**：
   - 分离前缀和模型名
   - 无前缀视为"openai"
2. **查找阶段**：
   - 优先查自定义映射
   - 再查fallback缓存
   - 最后动态创建
3. **获取模型**：调用提供商的get_model
4. **返回实例**：返回准备好的Model

## 7. 错误处理时序图

### 7.1 模型调用失败处理

```mermaid
sequenceDiagram
    participant R as Runner
    participant M as Model
    participant API as LLM API
    participant E as ErrorHandler
    
    R->>M: get_response(...)
    activate M
    
    M->>API: 调用API
    activate API
    
    alt API调用失败
        API-->>M: HTTPError/APIError
        deactivate API
        
        M->>E: 处理异常
        activate E
        
        alt RateLimitError
            E->>E: 记录限流错误
            E->>E: 返回重试建议
            E-->>M: RateLimitException
            
        else AuthenticationError
            E->>E: 记录认证错误
            E-->>M: AuthException
            
        else TimeoutError
            E->>E: 记录超时
            E-->>M: TimeoutException
            
        else 其他错误
            E->>E: 记录通用错误
            E-->>M: ModelException
        end
        
        deactivate E
        
        M-->>R: raise Exception
        deactivate M
        
        R->>R: 捕获异常
        R->>R: 记录到trace
        R-->>R: 返回错误结果或重试
        
    else API调用成功
        API-->>M: 响应数据
        deactivate API
        M-->>R: ModelResponse
        deactivate M
    end
```

**错误处理说明：**

1. **捕获异常**：识别API错误类型
2. **分类处理**：
   - 限流：建议重试
   - 认证：检查密钥
   - 超时：调整timeout
   - 其他：记录详情
3. **传播异常**：向上层返回
4. **记录追踪**：在trace中记录错误

## 8. 模型切换时序图

### 8.1 运行时模型切换

```mermaid
sequenceDiagram
    participant U as 用户
    participant R as Runner
    participant RC as RunConfig
    participant MP as MultiProvider
    participant M1 as GPT-4o
    participant M2 as GPT-4o-mini
    
    Note over U: 首次运行
    U->>R: run(agent, input)
    activate R
    
    R->>R: 使用agent.model
    R->>MP: get_model("gpt-4o")
    MP-->>R: M1实例
    
    R->>M1: get_response(...)
    M1-->>R: 响应
    R-->>U: 结果
    deactivate R
    
    Note over U: 切换到更快的模型
    U->>RC: RunConfig(model="gpt-4o-mini")
    U->>R: run(agent, input, config=config)
    activate R
    
    R->>RC: 检查config.model
    R->>R: 覆盖agent.model
    
    R->>MP: get_model("gpt-4o-mini")
    MP-->>R: M2实例
    
    R->>M2: get_response(...)
    M2-->>R: 响应
    R-->>U: 结果
    deactivate R
```

**切换流程说明：**

1. **默认模型**：使用Agent配置的模型
2. **配置覆盖**：RunConfig.model覆盖默认值
3. **获取新模型**：从Provider获取
4. **使用新模型**：本次运行使用新模型

## 9. 模型调用完整时序图（端到端）

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as Agent
    participant R as Runner
    participant MP as MultiProvider
    participant OP as OpenAIProvider
    participant M as OpenAIResponsesModel
    participant API as OpenAI API
    participant T as TracingProcessor
    
    Note over U: 1. 创建Agent
    U->>A: Agent(name, model="gpt-4o", ...)
    A->>A: 保存配置
    
    Note over U: 2. 运行Agent
    U->>R: run(agent, "user input")
    activate R
    
    Note over R: 3. 获取模型实例
    R->>MP: get_model("gpt-4o")
    activate MP
    MP->>MP: 解析名称(None, "gpt-4o")
    MP->>OP: get_model("gpt-4o")
    activate OP
    OP->>M: 创建或获取缓存
    M-->>OP: model实例
    OP-->>MP: model实例
    deactivate OP
    MP-->>R: model实例
    deactivate MP
    
    Note over R: 4. 开始追踪
    R->>T: on_span_start(generation_span)
    T-->>R: None
    
    Note over R: 5. 调用模型
    R->>M: get_response(instructions, input, settings, ...)
    activate M
    
    M->>M: 准备请求数据
    M->>API: POST /v1/responses
    activate API
    API->>API: LLM处理
    API-->>M: 原始响应
    deactivate API
    
    M->>M: 解析响应
    M-->>R: ModelResponse(output, usage, ...)
    deactivate M
    
    Note over R: 6. 结束追踪
    R->>T: on_span_end(generation_span, response)
    T-->>R: None
    
    Note over R: 7. 处理响应
    alt 包含tool_calls
        R->>R: 执行工具
        R->>M: get_response(结果, ...)
        Note over R: 继续turn循环
    else 纯文本响应
        R->>R: 完成执行
    end
    
    R-->>U: RunResult
    deactivate R
```

**端到端流程总结：**

1. **Agent创建**：配置模型名称
2. **Runner启动**：初始化执行环境
3. **模型获取**：通过Provider获取Model实例
4. **追踪开始**：记录generation span
5. **模型调用**：发送请求并解析响应
6. **追踪结束**：记录统计信息
7. **响应处理**：执行工具或返回结果
8. **返回用户**：完整的RunResult

Models 模块通过清晰的时序流程实现了模型的灵活选择、统一调用和标准化响应，为 OpenAI Agents 提供了强大的LLM集成基础。
