# Kitex-04-Generic-时序图

## 1. Generic创建与初始化时序图

```mermaid
sequenceDiagram
    autonumber
    participant APP as 应用代码
    participant FACTORY as 工厂函数
    participant PROVIDER as DescriptorProvider
    participant CODEC as 编解码器
    participant GENERIC as Generic实例
    participant CACHE as 描述符缓存
    
    APP->>FACTORY: JSONThriftGeneric(provider, opts)
    Note over APP,FACTORY: 创建JSON Thrift泛型实例
    
    FACTORY->>PROVIDER: 验证描述符提供器
    PROVIDER->>PROVIDER: 检查IDL文件路径
    PROVIDER-->>FACTORY: 返回验证结果
    
    FACTORY->>CODEC: newJsonThriftCodec(provider, opts)
    CODEC->>CODEC: 初始化编解码器配置
    CODEC->>CODEC: 设置DynamicGo选项
    CODEC->>PROVIDER: 预加载常用服务描述符
    
    loop 预加载服务描述符
        PROVIDER->>CACHE: 解析IDL文件
        CACHE->>CACHE: 构建服务描述符
        CACHE-->>PROVIDER: 缓存描述符
    end
    
    PROVIDER-->>CODEC: 返回预加载结果
    CODEC-->>FACTORY: 返回编解码器实例
    
    FACTORY->>GENERIC: 创建jsonThriftGeneric实例
    GENERIC->>GENERIC: 设置codec字段
    GENERIC-->>FACTORY: 返回Generic实例
    
    FACTORY-->>APP: 返回Generic接口
    
    Note over APP,CACHE: Generic实例创建完成，可用于客户端或服务端
```

### 创建时序说明

**1. 工厂函数调用阶段（步骤1-4）**
- 应用代码调用具体的Generic工厂函数
- 验证描述符提供器的有效性
- 检查IDL文件路径和内容的正确性

**2. 编解码器初始化阶段（步骤5-12）**
- 创建对应的编解码器实例
- 设置DynamicGo等性能优化选项
- 预加载常用的服务描述符到缓存

**3. Generic实例创建阶段（步骤13-16）**
- 创建具体的Generic实现实例
- 设置编解码器引用
- 返回Generic接口供后续使用

## 2. 泛型客户端调用时序图

```mermaid
sequenceDiagram
    autonumber
    participant APP as 应用代码
    participant CLIENT as GenericClient
    participant GENERIC as Generic实例
    participant CODEC as 编解码器
    participant PROVIDER as DescriptorProvider
    participant KCLIENT as KitexClient
    participant SERVER as 服务端
    
    APP->>CLIENT: GenericCall(ctx, method, request, opts)
    Note over APP,CLIENT: 执行泛型调用
    
    CLIENT->>CLIENT: 设置调用选项
    CLIENT->>CLIENT: 处理二进制泛型特殊逻辑
    
    alt HTTP泛型动态方法名
        CLIENT->>GENERIC: getMethodFunc(request)
        GENERIC->>GENERIC: 从HTTP请求中提取方法名
        GENERIC-->>CLIENT: 返回动态方法名
    end
    
    CLIENT->>CLIENT: 构建Args和Result对象
    CLIENT->>CLIENT: 获取方法信息
    
    alt 方法信息存在
        CLIENT->>CLIENT: 创建具体的Args和Result
        CLIENT->>CLIENT: 设置Method和Request字段
    else 方法信息不存在
        CLIENT->>CLIENT: 创建空的Args和Result
    end
    
    CLIENT->>KCLIENT: Call(ctx, method, args, result)
    Note over KCLIENT: 底层RPC调用
    
    KCLIENT->>GENERIC: 序列化请求
    GENERIC->>CODEC: Marshal(ctx, msg, out)
    
    CODEC->>PROVIDER: Provide(serviceName, methodName)
    alt 缓存命中
        PROVIDER-->>CODEC: 返回缓存的描述符
    else 缓存未命中
        PROVIDER->>PROVIDER: 解析IDL获取描述符
        PROVIDER->>PROVIDER: 缓存新描述符
        PROVIDER-->>CODEC: 返回新描述符
    end
    
    CODEC->>CODEC: 根据描述符序列化数据
    alt 使用DynamicGo
        CODEC->>CODEC: 高性能动态序列化
    else 使用传统方式
        CODEC->>CODEC: 反射序列化
    end
    
    CODEC-->>GENERIC: 返回序列化结果
    GENERIC-->>KCLIENT: 返回序列化数据
    
    KCLIENT->>SERVER: 发送RPC请求
    SERVER-->>KCLIENT: 返回RPC响应
    
    KCLIENT->>GENERIC: 反序列化响应
    GENERIC->>CODEC: Unmarshal(ctx, msg, in)
    CODEC->>CODEC: 根据描述符反序列化数据
    CODEC-->>GENERIC: 返回反序列化结果
    GENERIC-->>KCLIENT: 返回响应对象
    
    KCLIENT-->>CLIENT: 返回调用结果
    CLIENT->>CLIENT: 提取Success字段
    CLIENT-->>APP: 返回响应数据
```

### 客户端调用时序说明

**1. 调用准备阶段（步骤1-12）**
- 设置调用级别的配置选项
- 处理不同泛型类型的特殊逻辑
- 动态获取方法名（HTTP泛型）
- 构建泛型参数和结果容器

**2. 序列化阶段（步骤13-24）**
- 获取服务和方法的描述符信息
- 优先使用缓存的描述符，提高性能
- 根据配置选择高性能或传统序列化方式
- 将请求数据序列化为网络传输格式

**3. 网络传输阶段（步骤25-26）**
- 通过底层RPC客户端发送请求
- 接收服务端返回的响应数据

**4. 反序列化阶段（步骤27-32）**
- 使用相同的描述符反序列化响应
- 将网络数据转换为业务对象
- 提取业务结果返回给应用代码

## 3. 泛型服务端处理时序图

```mermaid
sequenceDiagram
    autonumber
    participant CLIENT as 客户端
    participant SERVER as 服务端框架
    participant HANDLER as callHandler
    participant SERVICEV2 as ServiceV2
    participant BUSINESS as 业务逻辑
    participant CODEC as 编解码器
    participant PROVIDER as DescriptorProvider
    
    CLIENT->>SERVER: 发送RPC请求
    SERVER->>SERVER: 接收网络数据
    
    SERVER->>CODEC: Unmarshal(ctx, msg, in)
    Note over CODEC: 反序列化请求
    
    CODEC->>PROVIDER: Provide(serviceName, methodName)
    PROVIDER-->>CODEC: 返回服务描述符
    
    CODEC->>CODEC: 根据描述符反序列化
    CODEC-->>SERVER: 返回Args对象
    
    SERVER->>HANDLER: callHandler(ctx, handler, arg, result)
    Note over HANDLER: 调用业务处理器
    
    HANDLER->>HANDLER: 提取Args和Result
    HANDLER->>HANDLER: 获取RPC信息
    HANDLER->>HANDLER: 提取服务名和方法名
    
    alt ServiceV2处理器
        HANDLER->>SERVICEV2: 检查GenericCall是否实现
        alt GenericCall已实现
            HANDLER->>SERVICEV2: GenericCall(ctx, service, method, request)
            
            SERVICEV2->>BUSINESS: 根据服务名和方法名路由
            Note over BUSINESS: 执行具体业务逻辑
            
            alt Echo服务
                BUSINESS->>BUSINESS: 处理Echo逻辑
            else Calc服务
                BUSINESS->>BUSINESS: 处理计算逻辑
            else 其他服务
                BUSINESS->>BUSINESS: 处理对应业务逻辑
            end
            
            BUSINESS-->>SERVICEV2: 返回业务结果
            SERVICEV2-->>HANDLER: 返回处理结果
            HANDLER->>HANDLER: 设置realResult.Success
        else GenericCall未实现
            HANDLER-->>SERVER: 返回未实现错误
        end
    else Service v1处理器
        HANDLER->>SERVICEV2: GenericCall(ctx, method, request)
        SERVICEV2->>BUSINESS: 根据方法名路由
        BUSINESS-->>SERVICEV2: 返回业务结果
        SERVICEV2-->>HANDLER: 返回处理结果
        HANDLER->>HANDLER: 设置realResult.Success
    end
    
    HANDLER-->>SERVER: 返回处理结果
    
    SERVER->>CODEC: Marshal(ctx, msg, out)
    Note over CODEC: 序列化响应
    
    CODEC->>PROVIDER: Provide(serviceName, methodName)
    PROVIDER-->>CODEC: 返回服务描述符
    
    CODEC->>CODEC: 根据描述符序列化响应
    CODEC-->>SERVER: 返回序列化数据
    
    SERVER->>CLIENT: 发送响应数据
```

### 服务端处理时序说明

**1. 请求接收阶段（步骤1-6）**
- 服务端接收客户端的RPC请求
- 使用编解码器反序列化请求数据
- 获取服务描述符进行数据解析
- 构建Args参数对象

**2. 业务路由阶段（步骤7-12）**
- 调用callHandler进行请求分发
- 提取RPC调用的元信息
- 获取服务名和方法名用于路由

**3. 业务处理阶段（步骤13-26）**
- 根据处理器类型选择调用方式
- ServiceV2支持多服务路由
- 根据服务名和方法名路由到具体业务逻辑
- 执行业务处理并返回结果

**4. 响应发送阶段（步骤27-33）**
- 使用编解码器序列化响应数据
- 获取相同的服务描述符保证一致性
- 将响应数据发送回客户端

## 4. HTTP泛型映射时序图

```mermaid
sequenceDiagram
    autonumber
    participant HTTPCLIENT as HTTP客户端
    participant GATEWAY as API网关
    participant HTTPCODEC as HTTP编解码器
    participant ROUTER as 路由器
    participant MAPPER as 参数映射器
    participant RPCCLIENT as RPC客户端
    participant RPCSERVER as RPC服务端
    
    HTTPCLIENT->>GATEWAY: HTTP请求
    Note over HTTPCLIENT,GATEWAY: POST /api/v1/echo
    
    GATEWAY->>HTTPCODEC: Decode(ctx, msg, in)
    Note over HTTPCODEC: HTTP到RPC转换
    
    HTTPCODEC->>HTTPCODEC: 解析HTTP请求
    HTTPCODEC->>HTTPCODEC: 提取Method、Path、Headers、Body
    
    HTTPCODEC->>ROUTER: Match(method, path)
    ROUTER->>ROUTER: 查找路由规则
    
    alt 路由匹配成功
        ROUTER-->>HTTPCODEC: 返回路由信息
        HTTPCODEC->>MAPPER: mapRequestToArgs(request, route)
        
        MAPPER->>MAPPER: 提取路径参数
        MAPPER->>MAPPER: 提取查询参数
        MAPPER->>MAPPER: 解析请求体
        MAPPER->>MAPPER: 构建RPC参数
        MAPPER-->>HTTPCODEC: 返回RPC参数
        
        HTTPCODEC->>HTTPCODEC: 设置消息数据
        HTTPCODEC-->>GATEWAY: 返回RPC消息
    else 路由匹配失败
        ROUTER-->>HTTPCODEC: 返回404错误
        HTTPCODEC-->>GATEWAY: 返回路由错误
    end
    
    GATEWAY->>RPCCLIENT: 发送RPC请求
    RPCCLIENT->>RPCSERVER: 转发RPC调用
    RPCSERVER-->>RPCCLIENT: 返回RPC响应
    RPCCLIENT-->>GATEWAY: 返回RPC结果
    
    GATEWAY->>HTTPCODEC: Encode(ctx, msg, out)
    Note over HTTPCODEC: RPC到HTTP转换
    
    HTTPCODEC->>HTTPCODEC: 获取RPC结果
    HTTPCODEC->>HTTPCODEC: 构建HTTP响应
    
    alt 成功响应
        HTTPCODEC->>HTTPCODEC: 设置200状态码
        HTTPCODEC->>HTTPCODEC: 序列化响应体
    else 错误响应
        HTTPCODEC->>HTTPCODEC: 设置错误状态码
        HTTPCODEC->>HTTPCODEC: 构建错误响应
    end
    
    HTTPCODEC->>HTTPCODEC: 设置响应头
    HTTPCODEC-->>GATEWAY: 返回HTTP响应数据
    
    GATEWAY->>HTTPCLIENT: 发送HTTP响应
```

### HTTP映射时序说明

**1. HTTP请求解析阶段（步骤1-6）**
- HTTP客户端发送RESTful API请求
- API网关接收HTTP请求
- HTTP编解码器解析请求的各个组成部分

**2. 路由匹配阶段（步骤7-16）**
- 根据HTTP方法和路径进行路由匹配
- 成功匹配时提取路由参数和查询参数
- 解析请求体并构建RPC调用参数
- 失败时返回404错误

**3. RPC调用阶段（步骤17-20）**
- 将HTTP请求转换为RPC调用
- 通过RPC客户端发送到目标服务
- 接收RPC服务的处理结果

**4. HTTP响应构建阶段（步骤21-30）**
- 将RPC结果转换为HTTP响应
- 根据结果类型设置相应的状态码
- 序列化响应体并设置响应头
- 发送HTTP响应给客户端

## 5. 描述符缓存管理时序图

```mermaid
sequenceDiagram
    autonumber
    participant CODEC as 编解码器
    participant PROVIDER as DescriptorProvider
    participant CACHE as 描述符缓存
    participant PARSER as IDL解析器
    participant FILESYSTEM as 文件系统
    participant MONITOR as 文件监控器
    
    Note over CODEC,MONITOR: 首次访问描述符
    CODEC->>PROVIDER: Provide(serviceName, methodName)
    PROVIDER->>CACHE: 检查缓存
    
    alt 缓存命中
        CACHE-->>PROVIDER: 返回缓存的描述符
        PROVIDER-->>CODEC: 返回描述符
    else 缓存未命中
        PROVIDER->>PROVIDER: 获取读锁
        PROVIDER->>CACHE: 双重检查缓存
        
        alt 仍然未命中
            PROVIDER->>PROVIDER: 升级为写锁
            PROVIDER->>FILESYSTEM: 读取IDL文件
            FILESYSTEM-->>PROVIDER: 返回文件内容
            
            PROVIDER->>PARSER: 解析IDL内容
            PARSER->>PARSER: 词法分析
            PARSER->>PARSER: 语法分析
            PARSER->>PARSER: 构建描述符树
            PARSER-->>PROVIDER: 返回服务描述符
            
            PROVIDER->>CACHE: 缓存新描述符
            PROVIDER->>PROVIDER: 释放写锁
            PROVIDER-->>CODEC: 返回描述符
        else 其他线程已加载
            PROVIDER->>PROVIDER: 释放读锁
            PROVIDER->>CACHE: 获取缓存描述符
            CACHE-->>PROVIDER: 返回描述符
            PROVIDER-->>CODEC: 返回描述符
        end
    end
    
    Note over CODEC,MONITOR: 文件变更监控
    MONITOR->>FILESYSTEM: 监控IDL文件变更
    
    alt 文件发生变更
        FILESYSTEM->>MONITOR: 通知文件变更事件
        MONITOR->>PROVIDER: 触发缓存失效
        PROVIDER->>CACHE: 清理相关缓存
        CACHE->>CACHE: 删除过期描述符
        CACHE-->>PROVIDER: 缓存清理完成
        PROVIDER-->>MONITOR: 处理完成
    end
    
    Note over CODEC,MONITOR: 后续访问使用新缓存
    CODEC->>PROVIDER: Provide(serviceName, methodName)
    PROVIDER->>CACHE: 检查缓存（已失效）
    CACHE-->>PROVIDER: 缓存未命中
    
    Note over PROVIDER: 重新加载最新的IDL文件
```

### 描述符缓存管理说明

**1. 缓存查找阶段（步骤1-6）**
- 编解码器请求服务描述符
- 优先检查内存缓存
- 缓存命中时直接返回，提高性能

**2. 缓存加载阶段（步骤7-20）**
- 使用读写锁保证并发安全
- 双重检查避免重复加载
- 解析IDL文件构建描述符
- 将新描述符缓存供后续使用

**3. 文件监控阶段（步骤21-30）**
- 监控IDL文件的变更事件
- 文件变更时主动失效相关缓存
- 确保使用最新的IDL定义

**4. 缓存更新阶段（步骤31-35）**
- 后续访问时重新加载最新文件
- 更新缓存内容
- 保证服务定义的一致性

## 6. 流式调用处理时序图

```mermaid
sequenceDiagram
    autonumber
    participant CLIENT as 客户端
    participant GCLIENT as GenericClient
    participant STREAM as 流式客户端
    participant SERVER as 服务端
    participant GSERVER as GenericServer
    participant HANDLER as 流式处理器
    participant BUSINESS as 业务逻辑
    
    Note over CLIENT,BUSINESS: 双向流式调用示例
    CLIENT->>GCLIENT: BidirectionalStreaming(ctx, method, opts)
    GCLIENT->>GCLIENT: 创建流式调用选项
    GCLIENT->>STREAM: 创建双向流客户端
    STREAM-->>GCLIENT: 返回流客户端实例
    GCLIENT-->>CLIENT: 返回流接口
    
    Note over CLIENT,BUSINESS: 建立流式连接
    CLIENT->>STREAM: 建立流式连接
    STREAM->>SERVER: 发起流式连接请求
    SERVER->>GSERVER: 接受流式连接
    GSERVER->>HANDLER: 创建流式处理器
    HANDLER-->>GSERVER: 返回处理器实例
    GSERVER-->>SERVER: 连接建立成功
    SERVER-->>STREAM: 确认连接建立
    STREAM-->>CLIENT: 流连接可用
    
    Note over CLIENT,BUSINESS: 流式数据交换
    loop 流式数据交换
        CLIENT->>STREAM: Send(data)
        STREAM->>SERVER: 发送流数据
        SERVER->>GSERVER: 接收流数据
        GSERVER->>HANDLER: OnMessage(data)
        HANDLER->>BUSINESS: 处理业务数据
        BUSINESS->>BUSINESS: 执行业务逻辑
        BUSINESS-->>HANDLER: 返回处理结果
        HANDLER->>HANDLER: 构建响应数据
        HANDLER->>GSERVER: Send(response)
        GSERVER->>SERVER: 发送响应数据
        SERVER->>STREAM: 转发响应
        STREAM->>CLIENT: Recv()接收响应
        CLIENT->>CLIENT: 处理响应数据
    end
    
    Note over CLIENT,BUSINESS: 关闭流连接
    CLIENT->>STREAM: Close()
    STREAM->>SERVER: 发送关闭信号
    SERVER->>GSERVER: 处理关闭事件
    GSERVER->>HANDLER: OnClose()
    HANDLER->>BUSINESS: 清理业务资源
    BUSINESS-->>HANDLER: 清理完成
    HANDLER-->>GSERVER: 关闭处理完成
    GSERVER-->>SERVER: 流关闭完成
    SERVER-->>STREAM: 确认关闭
    STREAM-->>CLIENT: 流已关闭
```

### 流式调用处理说明

**1. 流连接建立阶段（步骤1-12）**
- 客户端创建流式调用客户端
- 建立到服务端的流式连接
- 服务端创建对应的流处理器
- 确认流连接建立成功

**2. 数据交换阶段（步骤13-25）**
- 客户端和服务端进行双向数据交换
- 支持并发的发送和接收操作
- 业务逻辑处理流式数据
- 实时响应和数据传输

**3. 流关闭阶段（步骤26-35）**
- 客户端主动关闭流连接
- 服务端处理关闭事件
- 清理流相关的业务资源
- 确认流连接完全关闭

## 时序图总结

这些时序图展示了Generic模块的完整工作流程：

1. **Generic创建**：从工厂函数到实例化的完整过程，包含描述符预加载
2. **客户端调用**：泛型客户端的完整调用链路，包含动态编解码
3. **服务端处理**：泛型服务的请求处理流程，支持多服务路由
4. **HTTP映射**：HTTP请求到RPC调用的完整转换过程
5. **描述符缓存**：IDL描述符的缓存管理和文件监控机制
6. **流式调用**：流式泛型调用的建立、数据交换和关闭过程

每个时序图都包含了详细的步骤说明和关键节点分析，帮助开发者理解Generic模块的内部工作机制、性能优化点和扩展方式。
