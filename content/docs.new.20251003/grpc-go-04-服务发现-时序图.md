# gRPC-Go 服务发现模块时序图文档

## 时序图概览

本文档详细描述了 gRPC-Go 服务发现模块的各种时序流程，包括解析器注册、初始化、名称解析、状态更新、错误处理等核心场景。每个时序图都配有详细的文字说明，帮助理解服务发现的完整工作流程。

## 核心时序图列表

1. **解析器注册时序图** - 解析器的注册和查找流程
2. **解析器初始化时序图** - 解析器实例的创建和启动
3. **DNS解析时序图** - DNS名称解析的完整流程
4. **地址更新时序图** - 解析结果向客户端的通知流程
5. **立即解析时序图** - ResolveNow触发的解析流程
6. **解析错误处理时序图** - 解析失败的错误处理机制
7. **解析器关闭时序图** - 解析器的清理和关闭流程

---

## 1. 解析器注册时序图

```mermaid
sequenceDiagram
    autonumber
    participant Init as init()
    participant Builder as ResolverBuilder
    participant Reg as Register
    participant Map as GlobalMap
    participant Get as Get
    participant Dial as grpc.Dial
    
    Note over Init,Map: 初始化阶段
    Init->>Builder: 创建 Builder 实例
    Init->>Reg: Register(builder)
    Reg->>Builder: Scheme()
    Builder-->>Reg: "dns"
    Reg->>Map: m["dns"] = builder
    
    Note over Dial,Map: 连接创建阶段
    Dial->>Dial: 解析 target URL
    Dial->>Get: Get(scheme)
    Get->>Map: 查找 m["dns"]
    Map-->>Get: 返回 builder
    Get-->>Dial: 返回 Builder
```

---

## 2. 解析器初始化时序图

```mermaid
sequenceDiagram
    autonumber
    participant Dial as grpc.Dial
    participant CC as ClientConn
    participant Builder as Builder
    participant Resolver as Resolver
    participant Watcher as Watcher
    
    Dial->>CC: 创建 ClientConn
    CC->>Builder: Build(target, cc, opts)
    Builder->>Resolver: 创建 Resolver 实例
    Resolver->>Resolver: 初始化内部状态
    Resolver->>Watcher: 启动监控协程
    Watcher->>Watcher: 进入监控循环
    Resolver->>Resolver: ResolveNow()
    Resolver->>Resolver: 触发首次解析
    Builder-->>CC: 返回 Resolver
    CC->>CC: 保存 Resolver 实例
```

---

## 3. DNS解析时序图

```mermaid
sequenceDiagram
    autonumber
    participant W as Watcher
    participant R as Resolver
    participant DNS as DNS Server
    participant CC as ClientConn
    participant LB as Balancer
    
    W->>R: 定时触发或手动触发
    R->>R: 解析主机名和端口
    R->>DNS: LookupHost(hostname)
    DNS->>DNS: 查询DNS记录
    
    alt DNS查询成功
        DNS-->>R: 返回IP地址列表
        R->>R: 构建 Address 列表
        R->>R: 构建 State 对象
        R->>CC: UpdateState(state)
        CC->>CC: 验证状态有效性
        CC->>LB: 通知负载均衡器
        LB->>LB: 更新后端地址
        CC-->>R: 更新成功
    else DNS查询失败
        DNS-->>R: 返回错误
        R->>CC: ReportError(err)
        CC->>CC: 记录错误
        CC->>LB: 保持当前状态
    end
```

---

## 4. 地址更新时序图

```mermaid
sequenceDiagram
    autonumber
    participant R as Resolver
    participant CC as ClientConn
    participant LB as Balancer
    participant Picker as Picker
    participant SC as SubConn
    
    R->>R: 检测到地址变化
    R->>CC: UpdateState(newState)
    CC->>CC: 比较新旧地址
    
    alt 地址有变化
        CC->>LB: UpdateClientConnState(state)
        LB->>LB: 处理地址变更
        
        loop 新增地址
            LB->>SC: NewSubConn(addr)
            SC->>SC: 建立连接
        end
        
        loop 删除地址
            LB->>SC: RemoveSubConn(sc)
            SC->>SC: 关闭连接
        end
        
        LB->>Picker: regeneratePicker()
        LB->>CC: UpdateState(newState)
        CC-->>R: 更新完成
    else 地址无变化
        CC->>CC: 忽略更新
        CC-->>R: 更新完成
    end
```

---

## 5. 立即解析时序图

```mermaid
sequenceDiagram
    autonumber
    participant App as Application
    participant CC as ClientConn
    participant LB as Balancer
    participant R as Resolver
    participant W as Watcher
    
    App->>CC: Connect() 或重试
    CC->>LB: ExitIdle()
    LB->>CC: ResolveNow()
    CC->>R: ResolveNow(opts)
    
    R->>R: 检查解析间隔
    
    alt 距离上次解析足够久
        R->>W: 发送解析信号
        Note over W: 通道非阻塞发送
        W->>W: 接收信号
        W->>R: 执行解析
        R->>R: lookup()
        R->>CC: UpdateState(state)
    else 距离上次解析太近
        R->>R: 忽略本次请求
    end
```

---

## 6. 解析错误处理时序图

```mermaid
sequenceDiagram
    autonumber
    participant R as Resolver
    participant CC as ClientConn
    participant LB as Balancer
    participant Retry as RetryLogic
    
    R->>R: 执行解析
    R->>R: 解析失败
    R->>CC: ReportError(err)
    CC->>CC: 记录错误日志
    
    alt 首次解析失败
        CC->>LB: ResolverError(err)
        LB->>LB: 进入错误状态
        CC->>Retry: 启动退避重试
    else 非首次解析失败
        CC->>LB: ResolverError(err)
        LB->>LB: 保持现有连接
        CC->>Retry: 继续退避重试
    end
    
    Retry->>Retry: 等待退避时间
    Retry->>CC: 触发重试
    CC->>R: ResolveNow()
```

---

## 7. 解析器关闭时序图

```mermaid
sequenceDiagram
    autonumber
    participant App as Application
    participant CC as ClientConn
    participant R as Resolver
    participant W as Watcher
    
    App->>CC: Close()
    CC->>R: Close()
    R->>R: 设置关闭标志
    R->>W: 发送关闭信号
    W->>W: 退出监控循环
    W->>W: 清理资源
    R->>R: 取消定时器
    R->>R: 清理内部状态
    R-->>CC: 关闭完成
    CC-->>App: 连接已关闭
```

这些时序图展示了 gRPC-Go 服务发现模块在各种场景下的完整工作流程，帮助开发者理解名称解析的内部机制，为服务发现集成和故障排查提供指导。
