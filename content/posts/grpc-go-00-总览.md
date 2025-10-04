---
title: "gRPC-Go 源码剖析总览"
date: 2025-10-04T20:42:31+08:00
draft: false
tags:
  - Go
  - 源码剖析
  - 架构分析
  - 源码分析
categories:
  - Go
  - 编程语言
  - 运行时
series: "go-source-analysis"
description: "Go 源码剖析 - gRPC-Go 源码剖析总览"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# gRPC-Go 源码剖析总览

## 0. 摘要

### 项目目标与核心能力
gRPC-Go 是 Google 开发的高性能、开源的通用 RPC 框架在 Go 语言上的实现。该项目提供完整的 gRPC 客户端和服务端功能，支持 HTTP/2 协议、流式通信、负载均衡、服务发现、认证授权等企业级特性。

**核心能力边界：**
- 基于 HTTP/2 的高性能 RPC 通信
- 支持 Unary、Client Streaming、Server Streaming、Bidirectional Streaming 四种调用模式
- 内置多种负载均衡策略（round_robin、pick_first、weighted_round_robin 等）
- 可插拔的服务发现机制（DNS、passthrough、unix socket 等）
- 完整的认证体系（TLS、OAuth2、JWT、ALTS 等）
- 丰富的拦截器机制支持中间件扩展
- 内置健康检查、指标监控、链路追踪能力

**非目标：**
- 不提供服务注册中心实现（依赖外部服务发现）
- 不包含业务逻辑处理框架
- 不提供数据持久化能力

### 运行环境与部署形态
- **语言版本：** Go 1.24.0+
- **核心依赖：** golang.org/x/net、google.golang.org/protobuf、google.golang.org/genproto
- **部署形态：** 
  - 库形式集成到应用程序中
  - 支持单体应用内 RPC 调用
  - 支持微服务间通信
  - 可作为 sidecar 代理组件

## 1. 整体架构图

```mermaid
flowchart TB
    subgraph "Client Side"
        App1[Application Code]
        CC[ClientConn<br/>连接管理]
        Resolver[Resolver<br/>服务发现]
        Balancer[Balancer<br/>负载均衡]
        Picker[Picker<br/>连接选择]
        Transport1[Transport<br/>传输层]
    end
    
    subgraph "Network"
        HTTP2[HTTP/2 Protocol<br/>over TCP/TLS]
    end
    
    subgraph "Server Side"
        Transport2[Transport<br/>传输层]
        Server[Server<br/>服务管理]
        Handler[Method Handler<br/>请求处理]
        App2[Application Code]
    end
    
    subgraph "Cross-Cutting Concerns"
        Interceptor[Interceptors<br/>拦截器链]
        Credentials[Credentials<br/>认证凭证]
        Encoding[Encoding<br/>编码压缩]
        Metadata[Metadata<br/>元数据]
        Status[Status<br/>状态码]
        Health[Health Check<br/>健康检查]
    end
    
    App1 --> CC
    CC --> Resolver
    CC --> Balancer
    Balancer --> Picker
    Picker --> Transport1
    Transport1 --> HTTP2
    HTTP2 --> Transport2
    Transport2 --> Server
    Server --> Handler
    Handler --> App2
    
    CC -.-> Interceptor
    Server -.-> Interceptor
    Transport1 -.-> Credentials
    Transport2 -.-> Credentials
    Transport1 -.-> Encoding
    Transport2 -.-> Encoding
    CC -.-> Metadata
    Server -.-> Metadata
    Handler -.-> Status
    Server -.-> Health
```

**图解与要点：**

1. **客户端架构层次：**
   - `ClientConn` 作为客户端核心，管理连接生命周期
   - `Resolver` 负责将服务名解析为具体地址列表
   - `Balancer` 根据负载均衡策略管理多个连接
   - `Picker` 为每个 RPC 请求选择合适的连接
   - `Transport` 处理底层 HTTP/2 协议通信

2. **服务端架构层次：**
   - `Transport` 接收并解析 HTTP/2 请求
   - `Server` 管理服务注册和请求路由
   - `Handler` 执行具体的 RPC 方法调用

3. **横切关注点：**
   - `Interceptors` 提供请求/响应拦截能力
   - `Credentials` 处理各种认证机制
   - `Encoding` 支持多种序列化和压缩方式
   - `Metadata` 传递请求上下文信息
   - `Status` 统一错误码和状态管理
   - `Health` 提供服务健康状态检查

4. **数据流与控制流：**
   - 控制流：应用代码 → ClientConn → Balancer → Picker → Transport
   - 数据流：应用数据经过编码、压缩后通过 HTTP/2 传输
   - 错误流：各层异常通过 Status 统一处理并向上传播

5. **并发与扩展性：**
   - ClientConn 支持多 goroutine 并发调用
   - Server 为每个 RPC 请求分配独立 goroutine
   - Balancer 和 Resolver 异步更新连接状态
   - Transport 层支持多路复用和流控

## 2. 全局时序图（主要业务闭环）

```mermaid
sequenceDiagram
    autonumber
    participant App as Application
    participant CC as ClientConn
    participant R as Resolver
    participant B as Balancer
    participant P as Picker
    participant T1 as Client Transport
    participant Net as Network
    participant T2 as Server Transport
    participant S as Server
    participant H as Handler

    Note over App,H: 1. 连接建立阶段
    App->>CC: grpc.NewClient(target, opts)
    CC->>R: 启动服务发现
    R-->>CC: 地址更新通知
    CC->>B: UpdateClientConnState
    B->>P: 创建连接选择器
    B-->>CC: Picker 就绪

    Note over App,H: 2. RPC 调用阶段
    App->>CC: Invoke(method, req)
    CC->>P: Pick(rpcInfo)
    P-->>CC: 返回选中连接
    CC->>T1: 发送请求
    T1->>Net: HTTP/2 HEADERS + DATA
    Net->>T2: 接收请求
    T2->>S: 路由到对应服务
    S->>H: 调用业务方法
    H-->>S: 返回响应
    S-->>T2: 发送响应
    T2->>Net: HTTP/2 HEADERS + DATA
    Net->>T1: 接收响应
    T1-->>CC: 响应数据
    CC-->>App: 返回结果

    Note over App,H: 3. 连接维护阶段
    R->>R: 定期健康检查
    R-->>CC: 地址状态更新
    CC->>B: 连接状态变更
    B->>P: 更新连接选择策略
```

**图解与要点：**

1. **连接建立阶段（步骤1-6）：**
   - 应用调用 `grpc.NewClient()` 创建客户端连接
   - `Resolver` 根据 target 进行服务发现，解析出后端地址列表
   - `Balancer` 接收地址更新，建立到后端的物理连接
   - `Picker` 根据负载均衡策略准备连接选择逻辑

2. **RPC调用阶段（步骤7-16）：**
   - 应用发起 RPC 调用，`ClientConn` 通过 `Picker` 选择连接
   - `Transport` 层将请求编码为 HTTP/2 帧并发送
   - 服务端 `Transport` 接收请求并路由到对应的 `Handler`
   - `Handler` 执行业务逻辑并返回响应
   - 响应沿相同路径返回给应用

3. **连接维护阶段（步骤17-20）：**
   - `Resolver` 持续监控后端地址变化
   - `Balancer` 根据连接健康状态动态调整负载均衡策略
   - 支持连接池管理、故障转移、熔断等高可用特性

4. **关键边界条件：**
   - 连接超时：默认20秒连接建立超时
   - 请求超时：支持 context.WithTimeout 控制
   - 并发控制：客户端支持多 goroutine 并发调用
   - 流控管理：HTTP/2 层面的窗口大小控制
   - 错误重试：支持可配置的重试策略

## 3. 模块边界与交互图

### 核心模块清单

| 序号 | 模块名称 | 目录路径 | 主要职责 | 对外API |
|------|----------|----------|----------|---------|
| 01 | 客户端连接 | clientconn.go | 客户端连接管理、RPC调用入口 | NewClient, Dial, Invoke |
| 02 | 服务端 | server.go | 服务端管理、请求处理 | NewServer, RegisterService, Serve |
| 03 | 负载均衡 | balancer/ | 连接选择、负载均衡策略 | Register, Builder, Picker |
| 04 | 服务发现 | resolver/ | 服务名解析、地址发现 | Register, Builder, Resolver |
| 05 | 认证凭证 | credentials/ | 认证授权、传输安全 | NewTLS, NewOAuth, NewJWT |
| 06 | 编码压缩 | encoding/ | 消息序列化、数据压缩 | RegisterCodec, RegisterCompressor |
| 07 | 元数据 | metadata/ | 请求上下文、Header传递 | New, FromContext, AppendToOutgoing |
| 08 | 状态码 | status/, codes/ | 错误码定义、状态管理 | New, Error, FromError |
| 09 | 拦截器 | interceptor.go | 请求拦截、中间件链 | UnaryInterceptor, StreamInterceptor |
| 10 | 健康检查 | health/ | 服务健康状态检查 | NewServer, Check, Watch |
| 11 | 传输层 | internal/transport/ | HTTP/2协议处理、连接管理 | NewClientTransport, NewServerTransport |
| 12 | 流处理 | stream.go | 流式RPC、双向通信 | ClientStream, ServerStream |

### 模块交互矩阵

| 调用方 → 被调方 | 客户端连接 | 服务端 | 负载均衡 | 服务发现 | 认证凭证 | 编码压缩 | 元数据 | 状态码 | 拦截器 | 健康检查 | 传输层 |
|----------------|------------|--------|----------|----------|----------|----------|--------|--------|--------|----------|--------|
| **客户端连接** | - | - | 同步调用 | 同步调用 | 同步调用 | 同步调用 | 同步调用 | 同步调用 | 同步调用 | - | 同步调用 |
| **服务端** | - | - | - | - | 同步调用 | 同步调用 | 同步调用 | 同步调用 | 同步调用 | 同步调用 | 同步调用 |
| **负载均衡** | 异步回调 | - | - | - | - | - | - | 同步调用 | - | - | 同步调用 |
| **服务发现** | 异步回调 | - | - | - | - | - | - | 同步调用 | - | 同步调用 | - |
| **认证凭证** | - | - | - | - | - | - | 同步调用 | 同步调用 | - | - | - |
| **编码压缩** | - | - | - | - | - | - | - | 同步调用 | - | - | - |
| **元数据** | - | - | - | - | - | - | - | - | - | - | - |
| **状态码** | - | - | - | - | - | - | - | - | - | - | - |
| **拦截器** | 同步调用 | 同步调用 | - | - | 同步调用 | 同步调用 | 同步调用 | 同步调用 | - | - | - |
| **健康检查** | - | - | - | - | - | - | 同步调用 | 同步调用 | - | - | - |
| **传输层** | 异步回调 | 异步回调 | - | - | 同步调用 | 同步调用 | 同步调用 | 同步调用 | - | - | - |

**交互说明：**

1. **同步调用：** 直接函数调用，调用方等待被调方返回结果
2. **异步回调：** 通过接口回调或 channel 通信，非阻塞调用
3. **事件驱动：** 基于状态变化触发的异步通知机制

## 4. 关键设计与权衡

### 数据一致性与事务边界

1. **连接状态一致性：**
   - 采用最终一致性模型，Resolver 和 Balancer 异步更新连接状态
   - 通过版本号机制避免状态更新竞争
   - 连接失败时支持快速故障转移

2. **请求幂等性：**
   - 框架层面不保证幂等性，由应用层处理
   - 支持请求ID传递用于幂等性判断
   - 重试机制可配置幂等安全的方法

### 并发控制策略

1. **客户端并发：**
   - `ClientConn` 支持多 goroutine 并发调用
   - 使用读写锁保护连接状态变更
   - `Picker` 通过原子操作实现无锁连接选择

2. **服务端并发：**
   - 每个 RPC 请求分配独立 goroutine
   - 通过 `sync.Pool` 复用 goroutine 和内存对象
   - 支持最大并发连接数和请求数限制

3. **传输层并发：**
   - HTTP/2 多路复用支持单连接并发请求
   - 流控机制防止内存过度消耗
   - 连接池管理避免连接数过多

### 性能关键路径与优化

1. **热路径优化：**
   - RPC 调用路径：`App → ClientConn → Picker → Transport`（P95 < 1ms）
   - 连接选择使用轮询或一致性哈希，避免复杂计算
   - 编码解码支持零拷贝和内存池复用

2. **内存管理：**
   - 使用 `mem.Buffer` 统一内存分配
   - 支持共享缓冲池减少 GC 压力
   - 流式传输支持背压控制

3. **网络I/O优化：**
   - HTTP/2 连接复用减少握手开销
   - 支持 TCP keepalive 和连接预热
   - 批量发送和接收减少系统调用

### 可观测性指标

1. **关键指标：**
   - RPC 调用 QPS、延迟分布（P50/P95/P99）
   - 连接数、活跃连接数、连接建立失败率
   - 消息大小分布、编码解码耗时
   - 错误率按状态码分类统计

2. **监控集成：**
   - 支持 OpenTelemetry 标准
   - 内置 Prometheus 指标导出
   - 集成 gRPC channelz 调试接口

### 配置项与可变参数

1. **连接配置：**
   - `MaxReceiveMessageSize`：最大接收消息大小（默认4MB）
   - `MaxSendMessageSize`：最大发送消息大小（默认无限制）
   - `InitialWindowSize`：HTTP/2 流初始窗口大小（默认64KB）
   - `ConnectTimeout`：连接建立超时（默认20秒）

2. **负载均衡配置：**
   - 健康检查间隔、超时设置
   - 连接数上限、请求重试次数
   - 熔断阈值、恢复时间窗口

3. **安全配置：**
   - TLS 证书、密钥文件路径
   - 支持的加密套件、协议版本
   - 客户端认证模式（单向/双向）

## 5. 典型使用示例与最佳实践

### 示例1：最小可运行入口

**客户端示例：**
```go
func main() {
    conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatal("连接失败:", err)
    }
    defer conn.Close()
    
    client := pb.NewGreeterClient(conn)
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()
    
    resp, err := client.SayHello(ctx, &pb.HelloRequest{Name: "World"})
    if err != nil {
        log.Fatal("调用失败:", err)
    }
    log.Printf("响应: %s", resp.Message)
}
```

**服务端示例：**
```go
type server struct {
    pb.UnimplementedGreeterServer
}

func (s *server) SayHello(ctx context.Context, req *pb.HelloRequest) (*pb.HelloReply, error) {
    return &pb.HelloReply{Message: "Hello " + req.Name}, nil
}

func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatal("监听失败:", err)
    }
    
    s := grpc.NewServer()
    pb.RegisterGreeterServer(s, &server{})
    
    if err := s.Serve(lis); err != nil {
        log.Fatal("服务启动失败:", err)
    }
}
```

### 示例2：生产环境配置

```go
func newProductionClient(target string) (*grpc.ClientConn, error) {
    creds, err := credentials.NewClientTLSFromFile("ca-cert.pem", "")
    if err != nil {
        return nil, err
    }
    
    return grpc.NewClient(target,
        grpc.WithTransportCredentials(creds),
        grpc.WithDefaultServiceConfig(`{
            "loadBalancingPolicy": "round_robin",
            "healthCheckConfig": {
                "serviceName": "grpc.health.v1.Health"
            },
            "retryPolicy": {
                "maxAttempts": 3,
                "initialBackoff": "0.1s",
                "maxBackoff": "1s",
                "backoffMultiplier": 2,
                "retryableStatusCodes": ["UNAVAILABLE"]
            }
        }`),
        grpc.WithStatsHandler(&ocgrpc.ClientHandler{}),
        grpc.WithUnaryInterceptor(grpc_middleware.ChainUnaryClient(
            grpc_retry.UnaryClientInterceptor(),
            grpc_opentracing.UnaryClientInterceptor(),
        )),
    )
}

func newProductionServer() *grpc.Server {
    creds, err := credentials.NewServerTLSFromFile("server-cert.pem", "server-key.pem")
    if err != nil {
        log.Fatal(err)
    }
    
    s := grpc.NewServer(
        grpc.Creds(creds),
        grpc.MaxRecvMsgSize(4*1024*1024),
        grpc.MaxSendMsgSize(4*1024*1024),
        grpc.KeepaliveParams(keepalive.ServerParameters{
            MaxConnectionIdle: 15 * time.Second,
            Timeout:          5 * time.Second,
        }),
        grpc.StatsHandler(&ocgrpc.ServerHandler{}),
        grpc.UnaryInterceptor(grpc_middleware.ChainUnaryServer(
            grpc_recovery.UnaryServerInterceptor(),
            grpc_auth.UnaryServerInterceptor(authFunc),
            grpc_opentracing.UnaryServerInterceptor(),
        )),
    )
    
    // 注册健康检查
    healthServer := health.NewServer()
    grpc_health_v1.RegisterHealthServer(s, healthServer)
    healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
    
    return s
}
```

### 示例3：规模化部署注意事项

1. **连接管理：**
   - 客户端使用连接池，避免频繁建立连接
   - 设置合理的连接超时和keepalive参数
   - 监控连接数和连接建立失败率

2. **负载均衡：**
   - 根据后端容量选择合适的负载均衡策略
   - 配置健康检查避免请求发送到不健康实例
   - 使用断路器模式处理级联故障

3. **监控告警：**
   - 监控 RPC 成功率、延迟分布
   - 设置连接数、错误率阈值告警
   - 集成分布式链路追踪系统

4. **容量规划：**
   - 根据 QPS 和消息大小规划带宽
   - 考虑 HTTP/2 多路复用的连接数需求
   - 预留足够的文件描述符和内存资源

5. **安全加固：**
   - 启用 TLS 加密传输
   - 实现细粒度的认证授权
   - 定期轮换证书和密钥
   - 限制客户端访问频率和消息大小

---

## gRPC-Go 核心机制与生产实践

### 客户端连接管理

- **ClientConn 生命周期**：
  - **IDLE**：初始状态，未建立连接。
  - **CONNECTING**：正在建立连接，执行服务发现和负载均衡。
  - **READY**：连接就绪，可以发送请求。
  - **TRANSIENT_FAILURE**：临时故障，等待重连（指数退避）。
  - **SHUTDOWN**：连接已关闭。

- **连接池优化**：
  - 复用 ClientConn：一个 target 共享一个连接，HTTP/2 多路复用降低连接开销。
  - Keepalive 配置：`Time: 10s`（心跳间隔），`Timeout: 3s`（超时），`PermitWithoutStream: true`（无流时允许 ping）。
  - 连接状态监控：`conn.GetState()` 检查状态，避免使用失效连接。

- **重连策略**：
  - 指数退避：`BaseDelay: 1s`，`Multiplier: 1.6`，`Jitter: 0.2`，`MaxDelay: 120s`。
  - 最小连接超时：`MinConnectTimeout: 5s`。

### 负载均衡机制

- **Resolver（服务发现）**：
  - **DNS Resolver**：解析 SRV 记录获取服务端地址列表。
  - **Passthrough Resolver**：直接使用目标地址，不进行解析。
  - **自定义 Resolver**：集成 etcd、Consul、Nacos 等注册中心。

- **Balancer（负载均衡策略）**：
  - **pick_first**：选择第一个可用地址，适用于单实例或主备场景。
  - **round_robin**：轮询所有地址，均匀分配请求。
  - **weighted_round_robin**：根据权重分配，支持服务端反馈权重。
  - **自定义 Balancer**：实现一致性哈希、最少连接数等策略。

- **Picker（连接选择）**：
  - 每次 RPC 调用时，Picker 从 SubConn 池中选择一个连接。
  - SubConn 表示到单个后端的 HTTP/2 连接。

- **健康检查集成**：
  - 启用健康检查：`grpc.WithDefaultServiceConfig({"healthCheckConfig": {"serviceName": ""})`。
  - 自动剔除不健康实例，避免请求失败。

### 服务端架构

- **Server 初始化**：
  - 注册服务处理器：`pb.RegisterXXXServer(s, &handler{})`。
  - 配置选项：最大消息大小、keepalive、拦截器、凭证等。

- **请求处理流程**：
  - 接收 HTTP/2 请求 → 解析 Frame → 提取 RPC 元数据 → 执行拦截器链 → 调用业务处理器 → 编码响应 → 发送 Frame。

- **并发处理**：
  - 每个 RPC 请求在独立 goroutine 中处理。
  - 使用 sync.Pool 复用 buffer，减少内存分配。
  - 流量控制：HTTP/2 窗口机制防止接收端过载。

- **优雅关闭**：
  - `GracefulStop()`：停止接收新连接，等待现有请求完成。
  - 设置超时：配合 context 控制最大等待时间。

### 流式通信

- **四种流模式**：
  - **Unary**：一请求一响应，最常用。
  - **Client Streaming**：客户端发送多个消息，服务端返回一个响应（批量上传）。
  - **Server Streaming**：客户端发送一个请求，服务端返回多个响应（推送通知）。
  - **Bidirectional Streaming**：双向流式，全双工通信（聊天、实时协作）。

- **流控制**：
  - HTTP/2 流量控制：连接级窗口和流级窗口限制发送速率。
  - 接收窗口更新：`WINDOW_UPDATE` Frame 通知对端增加窗口。

- **流式 API 使用**：
  - 服务端流式：`stream.Send(msg)` 发送多个消息，客户端 `Recv()` 循环接收直到 `io.EOF`。
  - 客户端流式：客户端 `Send()` 循环发送，服务端 `Recv()` 接收直到 `io.EOF`，最后 `SendAndClose()`。
  - 双向流式：两端都调用 `Send()` 和 `Recv()`，独立收发。

### 拦截器机制

- **Unary 拦截器**：
  - 签名：`func(ctx, req, info, handler) (resp, error)`。
  - 链式调用：使用 `grpc_middleware.ChainUnaryServer()` 组合多个拦截器。
  - 执行顺序：按添加顺序依次执行，最后调用实际 handler。

- **Stream 拦截器**：
  - 签名：`func(srv, ss, info, handler) error`。
  - 包装 ServerStream：拦截 `SendMsg()` 和 `RecvMsg()` 实现消息级控制。

- **常见用途**：
  - 认证授权：验证 JWT Token、API Key。
  - 日志记录：记录请求/响应、耗时、错误。
  - 指标收集：记录 QPS、延迟分布、错误率。
  - 限流熔断：按 IP/用户限制请求速率，熔断不健康服务。
  - 链路追踪：集成 OpenTelemetry、Jaeger。

### 元数据（Metadata）

- **Metadata 用途**：
  - 传递认证信息：Token、API Key。
  - 传递链路追踪 ID：Trace ID、Span ID。
  - 传递自定义头：用户 ID、租户 ID。

- **客户端发送**：
  - `md := metadata.Pairs("key", "value")`。
  - `ctx := metadata.NewOutgoingContext(ctx, md)`。
  - 发起 RPC 调用时使用该 ctx。

- **服务端接收**：
  - `md, ok := metadata.FromIncomingContext(ctx)`。
  - `values := md.Get("key")`。

- **服务端发送（响应头/尾）**：
  - `grpc.SendHeader(ctx, md)`：发送响应头。
  - `grpc.SetTrailer(ctx, md)`：设置响应尾。

### 错误处理

- **Status Code**：
  - **OK**：成功。
  - **INVALID_ARGUMENT**：参数无效。
  - **NOT_FOUND**：资源不存在。
  - **PERMISSION_DENIED**：权限不足。
  - **UNAUTHENTICATED**：未认证。
  - **UNAVAILABLE**：服务不可用（临时故障，可重试）。
  - **DEADLINE_EXCEEDED**：超时。
  - **INTERNAL**：内部错误。

- **错误返回**：
  - `status.Errorf(codes.InvalidArgument, "invalid param: %v", err)`。
  - `status.Error(codes.NotFound, "resource not found")`。

- **错误处理**：
  - 客户端检查：`st, ok := status.FromError(err)`，判断 `st.Code()`。
  - 重试策略：对 `UNAVAILABLE`、`DEADLINE_EXCEEDED` 重试，避免重试 `INVALID_ARGUMENT`。

### 认证与安全

- **TLS 加密**：
  - 服务端：`creds, _ := credentials.NewServerTLSFromFile(certFile, keyFile)`。
  - 客户端：`creds, _ := credentials.NewClientTLSFromFile(caFile, serverName)`。
  - 双向 TLS：客户端和服务端都验证证书。

- **Token 认证**：
  - 实现 `credentials.PerRPCCredentials` 接口。
  - `GetRequestMetadata()` 返回认证头（如 `authorization: Bearer <token>`）。
  - 服务端拦截器验证 Token。

- **OAuth2 集成**：
  - 使用 `golang.org/x/oauth2` 生成 Token。
  - `grpc.WithPerRPCCredentials(oauth.NewOauthAccess(token))`。

### 性能优化

- **消息压缩**：
  - 启用 gzip：`grpc.UseCompressor(gzip.Name)`。
  - 适用于大消息，减少传输时间，但增加 CPU 开销。

- **连接复用**：
  - HTTP/2 多路复用：单连接承载多个并发流，减少连接建立开销。
  - 避免频繁创建 ClientConn，使用单例或连接池。

- **批量请求**：
  - 使用 Client Streaming 或 Bidirectional Streaming 批量发送请求，减少网络往返。

- **预分配 Buffer**：
  - 使用 `sync.Pool` 缓存 buffer，避免频繁分配。

- **减少序列化开销**：
  - 使用 Protobuf 二进制编码（比 JSON 快 5-10 倍）。
  - 避免嵌套过深的消息结构。

### 监控与可观测性

- **指标收集**：
  - 集成 Prometheus：使用 `go-grpc-prometheus` 导出指标。
  - 关键指标：`grpc_server_handled_total`（请求数）、`grpc_server_handling_seconds`（延迟）、`grpc_server_msg_received_total`（接收消息数）。

- **链路追踪**：
  - 集成 OpenTelemetry：拦截器中提取/注入 Trace Context。
  - 记录 RPC 调用的完整链路。

- **日志**：
  - 结构化日志：记录 method、status、duration、error。
  - 脱敏：避免记录敏感信息（密码、Token）。

### 常见问题排查

- **连接失败**：检查网络、DNS 解析、防火墙、TLS 证书。
- **请求超时**：调整 context 超时、检查服务端处理时间、网络延迟。
- **负载不均**：检查负载均衡策略、健康检查配置、后端权重。
- **内存泄漏**：检查流未关闭、Context 未取消、goroutine 泄漏。
- **CPU 高**：检查序列化/反序列化、压缩、大量并发请求。

