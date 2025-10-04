# Kitex-02-Server-时序图

## 1. 服务器创建与初始化时序图

```mermaid
sequenceDiagram
    autonumber
    participant APP as 业务应用
    participant SERVER as server包
    participant SERVERIMPL as server实现
    participant OPTIONS as Options
    participant SERVICES as services管理器
    participant MIDDLEWARE as 中间件系统
    
    APP->>SERVER: NewServer(opts...)
    Note over APP,SERVER: 创建服务器实例
    
    SERVER->>OPTIONS: internal_server.NewOptions(opts)
    OPTIONS->>OPTIONS: 解析配置选项
    OPTIONS->>OPTIONS: 设置默认值
    OPTIONS->>OPTIONS: 验证配置有效性
    OPTIONS-->>SERVER: 返回配置对象
    
    SERVER->>SERVICES: newServices()
    SERVICES->>SERVICES: 初始化服务映射表
    SERVICES->>SERVICES: 创建未知服务处理器
    SERVICES-->>SERVER: 返回服务管理器
    
    SERVER->>SERVERIMPL: 创建server实例
    SERVERIMPL->>SERVERIMPL: 设置opt和svcs
    SERVERIMPL->>SERVERIMPL: 初始化状态标记
    SERVERIMPL-->>SERVER: 返回server实例
    
    SERVER-->>APP: 返回Server接口
```

### 创建时序说明

**1. 配置解析阶段（步骤1-4）**
- 业务应用调用NewServer创建服务器
- 解析传入的配置选项，设置默认值
- 验证配置的有效性和兼容性
- 构建完整的Options配置对象

**2. 组件初始化阶段（步骤5-8）**
- 创建服务管理器，初始化服务映射表
- 设置未知服务和降级服务处理器
- 准备中间件系统和事件总线

**3. 实例创建阶段（步骤9-12）**
- 创建server实现实例
- 设置配置和服务管理器引用
- 初始化状态标记和同步原语
- 返回Server接口给业务应用

## 2. 服务注册时序图

```mermaid
sequenceDiagram
    autonumber
    participant APP as 业务应用
    participant SERVER as Server实例
    participant SERVICES as services管理器
    participant SERVICE as service实例
    participant SVCINFO as ServiceInfo
    participant HANDLER as 业务Handler
    
    APP->>SERVER: RegisterService(svcInfo, handler, opts...)
    Note over APP,SERVER: 注册业务服务
    
    SERVER->>SERVER: 获取互斥锁
    SERVER->>SERVER: 检查运行状态
    
    alt 服务器正在运行
        SERVER-->>APP: panic("service cannot be registered while server is running")
    else 服务器未运行
        SERVER->>SERVER: 验证svcInfo非空
        SERVER->>SERVER: 验证handler非空
        
        alt 参数验证失败
            SERVER-->>APP: panic("参数错误")
        else 参数验证成功
            SERVER->>SERVER: NewRegisterOptions(opts)
            Note over SERVER: 解析注册选项
            
            SERVER->>SERVICES: addService(svcInfo, handler, registerOpts)
            
            SERVICES->>SERVICES: 检查服务名冲突
            alt 服务已存在且非降级服务
                SERVICES-->>SERVER: 返回冲突错误
                SERVER-->>APP: panic("服务冲突")
            else 可以注册服务
                SERVICES->>SERVICE: newService(svcInfo, handler)
                SERVICE->>SVCINFO: 存储服务信息
                SERVICE->>HANDLER: 存储处理器引用
                SERVICE-->>SERVICES: 返回service实例
                
                alt 是降级服务
                    SERVICES->>SERVICES: 设置为fallbackSvc
                else 普通服务
                    SERVICES->>SERVICES: 存储到knownSvcMap
                end
                
                SERVICES-->>SERVER: 注册成功
            end
            
            SERVER->>SERVER: 释放互斥锁
            SERVER-->>APP: 返回nil
        end
    end
```

### 服务注册时序说明

**1. 状态检查阶段（步骤1-6）**
- 获取互斥锁保护并发安全
- 检查服务器运行状态，运行中不允许注册
- 验证服务信息和处理器的有效性

**2. 参数验证阶段（步骤7-10）**
- 严格验证ServiceInfo和Handler非空
- 解析注册选项，支持降级服务等特殊配置
- 参数无效时直接panic快速失败

**3. 服务存储阶段（步骤11-20）**
- 检查服务名冲突，防止重复注册
- 创建service实例，封装服务信息和处理器
- 根据注册选项决定存储位置（普通服务或降级服务）
- 更新服务映射表，支持快速查找

## 3. 服务器启动时序图

```mermaid
sequenceDiagram
    autonumber
    participant APP as 业务应用
    participant SERVER as Server实例
    participant INIT as 初始化组件
    participant MIDDLEWARE as 中间件系统
    participant REMOTESVR as RemoteServer
    participant NETWORK as 网络监听器
    participant REGISTRY as 注册中心
    participant HOOKS as 启动钩子
    participant PROFILER as 性能分析器
    
    APP->>SERVER: Run()
    Note over APP,SERVER: 启动服务器
    
    SERVER->>SERVER: 设置运行状态
    SERVER->>SERVER: 获取互斥锁
    SERVER->>SERVER: 设置isRun=true
    SERVER->>SERVER: 释放互斥锁
    
    SERVER->>INIT: init()
    Note over INIT: 服务器初始化
    
    alt 已经初始化
        INIT-->>SERVER: 直接返回
    else 未初始化
        INIT->>INIT: 设置isInit=true
        INIT->>INIT: fillContext()创建基础上下文
        
        alt 诊断服务存在
            INIT->>INIT: 注册诊断探测函数
        end
        
        INIT->>INIT: backup.Init()初始化会话备份
        INIT->>MIDDLEWARE: buildInvokeChain()
        
        MIDDLEWARE->>MIDDLEWARE: buildMiddlewares()
        MIDDLEWARE->>MIDDLEWARE: 初始化一元中间件
        MIDDLEWARE->>MIDDLEWARE: 初始化流式中间件
        MIDDLEWARE->>MIDDLEWARE: 构建中间件栈
        MIDDLEWARE-->>INIT: 返回调用链
        
        INIT-->>SERVER: 初始化完成
    end
    
    SERVER->>SERVER: check()配置检查
    
    alt 代理配置存在
        SERVER->>SERVER: 处理代理地址替换
    end
    
    SERVER->>SERVER: registerDebugInfo()
    SERVER->>SERVER: richRemoteOption()
    SERVER->>SERVER: newSvrTransHandler()
    
    SERVER->>REMOTESVR: NewServer(RemoteOpt, transHdlr)
    REMOTESVR->>REMOTESVR: 创建传输服务器
    REMOTESVR->>REMOTESVR: 配置网络选项
    REMOTESVR-->>SERVER: 返回服务器实例
    
    alt 性能分析器配置存在
        SERVER->>PROFILER: 启动profiler协程
        PROFILER->>PROFILER: 异步运行性能分析
    end
    
    SERVER->>NETWORK: svr.Start()
    NETWORK->>NETWORK: 绑定监听地址
    NETWORK->>NETWORK: 开始接受连接
    NETWORK-->>SERVER: 返回错误通道
    
    alt 启动失败
        SERVER-->>APP: 返回启动错误
    else 启动成功
        SERVER->>HOOKS: 执行启动钩子
        loop 所有启动钩子
            HOOKS->>HOOKS: 并发执行钩子函数
        end
        
        SERVER->>REGISTRY: buildRegistryInfo()
        SERVER->>REGISTRY: 注册服务到注册中心
        
        SERVER->>SERVER: waitExit(errCh)
        Note over SERVER: 阻塞等待退出信号
        
        alt 收到退出信号或错误
            SERVER->>SERVER: Stop()
            SERVER-->>APP: 返回退出结果
        end
    end
```

### 启动时序说明

**1. 状态设置阶段（步骤1-5）**
- 设置服务器运行状态标记
- 使用互斥锁保护状态变更
- 防止重复启动和并发问题

**2. 初始化阶段（步骤6-18）**
- 检查是否已初始化，支持幂等调用
- 创建基础上下文，包含事件总线和队列
- 构建完整的中间件调用链
- 初始化会话备份和诊断服务

**3. 网络启动阶段（步骤19-28）**
- 创建传输处理器和远程服务器
- 配置网络监听参数和协议选项
- 启动网络监听，开始接受客户端连接
- 可选启动性能分析器

**4. 服务注册阶段（步骤29-35）**
- 执行用户定义的启动钩子函数
- 构建服务注册信息
- 向注册中心注册服务实例
- 阻塞等待退出信号

## 4. 请求处理时序图

```mermaid
sequenceDiagram
    autonumber
    participant CLIENT as 客户端
    participant NETWORK as 网络层
    participant REMOTESVR as RemoteServer
    participant TRANSHANDLER as TransHandler
    participant ENDPOINT as Endpoint链
    participant MIDDLEWARE as 中间件栈
    participant SERVICES as 服务管理器
    participant SERVICE as 目标服务
    participant HANDLER as 业务Handler
    
    CLIENT->>NETWORK: 发送RPC请求
    NETWORK->>REMOTESVR: 接收网络数据
    REMOTESVR->>TRANSHANDLER: OnMessage()
    
    TRANSHANDLER->>TRANSHANDLER: 解析协议头
    TRANSHANDLER->>TRANSHANDLER: 构建RPCInfo
    TRANSHANDLER->>TRANSHANDLER: 解码请求消息
    
    TRANSHANDLER->>ENDPOINT: 调用端点链
    ENDPOINT->>MIDDLEWARE: 执行中间件栈
    
    Note over MIDDLEWARE: 中间件按顺序执行
    loop 中间件链
        MIDDLEWARE->>MIDDLEWARE: 执行前置处理
        alt 中间件拦截
            MIDDLEWARE-->>ENDPOINT: 返回拦截结果
        else 继续执行
            MIDDLEWARE->>MIDDLEWARE: 传递到下一个中间件
        end
    end
    
    MIDDLEWARE->>SERVICES: 路由到服务管理器
    SERVICES->>SERVICES: 根据服务名查找服务
    
    alt 服务不存在
        SERVICES->>SERVICES: 尝试未知服务处理器
        alt 未知服务处理器存在
            SERVICES->>SERVICE: 动态创建服务实例
        else 无处理器
            SERVICES-->>MIDDLEWARE: 返回服务不存在错误
        end
    else 服务存在
        SERVICES->>SERVICE: 获取目标服务
    end
    
    SERVICE->>SERVICE: 根据方法名获取处理器
    alt 方法不存在
        SERVICE->>SERVICE: 使用未知方法处理器
    else 方法存在
        SERVICE->>HANDLER: 调用业务处理器
    end
    
    HANDLER->>HANDLER: 执行业务逻辑
    HANDLER-->>SERVICE: 返回处理结果
    SERVICE-->>SERVICES: 返回结果
    SERVICES-->>MIDDLEWARE: 返回结果
    
    Note over MIDDLEWARE: 中间件逆序执行后置处理
    loop 中间件链（逆序）
        MIDDLEWARE->>MIDDLEWARE: 执行后置处理
        MIDDLEWARE->>MIDDLEWARE: 处理响应和错误
    end
    
    MIDDLEWARE-->>ENDPOINT: 返回最终结果
    ENDPOINT-->>TRANSHANDLER: 返回结果
    
    TRANSHANDLER->>TRANSHANDLER: 编码响应消息
    TRANSHANDLER->>TRANSHANDLER: 设置响应头
    TRANSHANDLER-->>REMOTESVR: 返回编码结果
    
    REMOTESVR->>NETWORK: 发送响应数据
    NETWORK->>CLIENT: 响应到达客户端
```

### 请求处理时序说明

**1. 网络接收阶段（步骤1-6）**
- 客户端发送RPC请求到服务器
- 网络层接收数据并传递给RemoteServer
- TransHandler解析协议头和请求消息
- 构建RPCInfo包含调用元信息

**2. 中间件处理阶段（步骤7-12）**
- 按配置顺序执行中间件栈
- 支持中间件拦截和短路返回
- 处理认证、限流、日志等横切关注点
- 传递请求到服务路由层

**3. 服务路由阶段（步骤13-22）**
- 根据服务名从映射表中查找服务
- 支持未知服务的动态处理
- 根据方法名获取具体的处理器
- 处理方法不存在的异常情况

**4. 业务处理阶段（步骤23-26）**
- 调用业务Handler执行具体逻辑
- 处理业务异常和返回结果
- 支持同步和异步处理模式

**5. 响应返回阶段（步骤27-35）**
- 中间件逆序执行后置处理
- 编码响应消息和设置响应头
- 通过网络层发送响应给客户端
- 完成一次完整的RPC调用

## 5. 服务器停止时序图

```mermaid
sequenceDiagram
    autonumber
    participant APP as 业务应用
    participant SERVER as Server实例
    participant HOOKS as 停机钩子
    participant REGISTRY as 注册中心
    participant REMOTESVR as RemoteServer
    participant NETWORK as 网络监听器
    participant CONN as 活跃连接
    
    APP->>SERVER: Stop()
    Note over APP,SERVER: 停止服务器
    
    SERVER->>SERVER: stopped.Do()
    Note over SERVER: 确保只停止一次
    
    SERVER->>SERVER: 获取互斥锁
    
    SERVER->>HOOKS: 执行停机钩子
    Note over HOOKS: 用户自定义清理逻辑
    
    loop 所有停机钩子
        HOOKS->>HOOKS: 执行钩子函数
        alt 钩子执行失败
            HOOKS->>HOOKS: 记录错误继续执行
        end
    end
    HOOKS-->>SERVER: 钩子执行完成
    
    alt 服务已注册到注册中心
        SERVER->>REGISTRY: Deregister(RegistryInfo)
        REGISTRY->>REGISTRY: 从注册中心移除服务
        REGISTRY-->>SERVER: 注销完成
        SERVER->>SERVER: 清空RegistryInfo
    end
    
    alt RemoteServer存在
        SERVER->>REMOTESVR: svr.Stop()
        REMOTESVR->>NETWORK: 停止网络监听
        NETWORK->>NETWORK: 关闭监听套接字
        NETWORK-->>REMOTESVR: 监听停止
        
        REMOTESVR->>CONN: 优雅关闭活跃连接
        loop 所有活跃连接
            CONN->>CONN: 等待请求处理完成
            CONN->>CONN: 关闭连接
        end
        CONN-->>REMOTESVR: 连接清理完成
        
        REMOTESVR->>REMOTESVR: 清理内部资源
        REMOTESVR-->>SERVER: 停止完成
        
        SERVER->>SERVER: 清空svr引用
    end
    
    SERVER->>SERVER: 释放互斥锁
    SERVER-->>APP: 返回停止结果
```

### 停止时序说明

**1. 停止保护阶段（步骤1-4）**
- 使用sync.Once确保Stop方法只执行一次
- 获取互斥锁保护停止过程
- 防止并发停止导致的资源竞争

**2. 钩子执行阶段（步骤5-9）**
- 执行用户注册的停机钩子函数
- 支持自定义的资源清理逻辑
- 即使某个钩子失败也继续执行其他钩子

**3. 服务注销阶段（步骤10-14）**
- 从注册中心注销服务实例
- 防止新的客户端发现和连接服务
- 清理注册信息，标记服务不可用

**4. 网络停止阶段（步骤15-25）**
- 停止网络监听，不再接受新连接
- 优雅关闭活跃连接，等待请求处理完成
- 清理网络资源和内部状态
- 确保所有连接正确关闭

**5. 资源清理阶段（步骤26-28）**
- 清空RemoteServer引用
- 释放互斥锁
- 返回停止结果给调用方

## 6. 中间件执行时序图

```mermaid
sequenceDiagram
    autonumber
    participant REQUEST as 请求
    participant CHAIN as 中间件链
    participant MW1 as 超时中间件
    participant MW2 as 认证中间件
    participant MW3 as 限流中间件
    participant MW4 as 日志中间件
    participant CORE as 核心中间件
    participant BUSINESS as 业务逻辑
    
    REQUEST->>CHAIN: 进入中间件链
    
    CHAIN->>MW1: 执行超时中间件
    MW1->>MW1: 设置请求超时
    MW1->>MW1: 启动超时检查
    
    MW1->>MW2: 调用下一个中间件
    MW2->>MW2: 验证客户端认证
    alt 认证失败
        MW2-->>MW1: 返回认证错误
        MW1-->>CHAIN: 返回错误响应
        CHAIN-->>REQUEST: 返回认证失败
    else 认证成功
        MW2->>MW3: 调用下一个中间件
        
        MW3->>MW3: 检查限流规则
        alt 触发限流
            MW3-->>MW2: 返回限流错误
            MW2-->>MW1: 传递错误
            MW1-->>CHAIN: 返回限流响应
            CHAIN-->>REQUEST: 返回限流错误
        else 未触发限流
            MW3->>MW4: 调用下一个中间件
            
            MW4->>MW4: 记录请求日志
            MW4->>MW4: 开始计时
            
            MW4->>CORE: 调用核心中间件
            CORE->>CORE: 执行ACL检查
            CORE->>CORE: 路由到业务服务
            
            CORE->>BUSINESS: 调用业务逻辑
            BUSINESS->>BUSINESS: 执行业务处理
            BUSINESS-->>CORE: 返回业务结果
            
            CORE->>CORE: 处理业务异常
            CORE-->>MW4: 返回处理结果
            
            MW4->>MW4: 记录响应日志
            MW4->>MW4: 计算处理耗时
            MW4-->>MW3: 返回结果
            
            MW3->>MW3: 更新限流统计
            MW3-->>MW2: 返回结果
            
            MW2->>MW2: 清理认证上下文
            MW2-->>MW1: 返回结果
            
            MW1->>MW1: 取消超时检查
            MW1-->>CHAIN: 返回最终结果
        end
    end
    
    CHAIN-->>REQUEST: 返回响应
```

### 中间件执行时序说明

**1. 中间件顺序执行（步骤1-16）**
- 按配置顺序依次执行中间件
- 每个中间件可以选择继续或短路返回
- 支持条件执行和错误处理

**2. 核心业务处理（步骤17-22）**
- 核心中间件执行ACL检查和服务路由
- 调用实际的业务处理逻辑
- 处理业务异常和错误转换

**3. 中间件逆序清理（步骤23-32）**
- 按相反顺序执行中间件的后置处理
- 清理资源和更新统计信息
- 确保所有中间件正确完成处理

**4. 错误处理机制**
- 任何中间件都可以短路返回错误
- 错误会逐层向上传播
- 支持错误转换和包装

## 时序图总结

这些时序图展示了Kitex Server模块的完整生命周期：

1. **创建初始化**：展示了服务器从创建到可用的完整过程
2. **服务注册**：展示了业务服务注册和管理的详细流程
3. **服务器启动**：展示了网络监听和服务注册的启动过程
4. **请求处理**：展示了从请求接收到响应返回的完整链路
5. **服务器停止**：展示了优雅停机和资源清理过程
6. **中间件执行**：展示了中间件链的执行顺序和错误处理

每个时序图都包含了详细的步骤说明和关键节点分析，帮助开发者理解Server模块的内部工作机制、扩展点和最佳实践。
