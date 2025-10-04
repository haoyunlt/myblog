# gRPC-Go 服务发现模块 API 文档

## API 概览

服务发现模块提供了完整的名称解析框架，负责将服务名称转换为可连接的网络地址列表。该模块支持多种解析策略（DNS、直连、Unix Socket等），并能动态监控地址变化，为负载均衡器提供实时的后端实例信息。

## 核心 API 列表

### 解析器管理 API
- `Register()` - 注册解析器构建器
- `Get()` - 获取已注册的解析器构建器
- `SetDefaultScheme()` - 设置默认解析协议
- `GetDefaultScheme()` - 获取默认解析协议

### 解析器接口 API
- `Builder.Build()` - 创建解析器实例
- `Builder.Scheme()` - 获取解析器协议名
- `Resolver.ResolveNow()` - 立即触发解析
- `Resolver.Close()` - 关闭解析器

### 状态更新 API
- `ClientConn.UpdateState()` - 更新解析状态
- `ClientConn.ReportError()` - 报告解析错误
- `ClientConn.ParseServiceConfig()` - 解析服务配置

---

## API 详细规格

### 1. Register

#### 基本信息
- **名称：** `Register`
- **签名：** `func Register(b Builder)`
- **功能：** 注册解析器构建器到全局注册表
- **幂等性：** 否，后注册的会覆盖先注册的同名构建器

#### 请求参数

```go
// Builder 解析器构建器接口
type Builder interface {
    // Build 创建新的解析器实例
    Build(target Target, cc ClientConn, opts BuildOptions) (Resolver, error)
    // Scheme 返回解析器支持的协议名
    Scheme() string
}
```

**参数说明表**

| 参数 | 类型 | 必填 | 约束 | 说明 |
|------|------|:----:|------|------|
| b | Builder | 是 | 实现 Builder 接口 | 解析器构建器 |

#### 入口函数实现

```go
func Register(b Builder) {
    // 使用 Scheme 作为键注册构建器
    m[b.Scheme()] = b
}
```

---

### 2. Builder.Build

#### 基本信息
- **名称：** `Build`
- **签名：** `func Build(target Target, cc ClientConn, opts BuildOptions) (Resolver, error)`
- **功能：** 创建解析器实例并开始解析

#### 请求参数

```go
// Target 解析目标
type Target struct {
    URL url.URL  // 解析的目标URL
}

// ClientConn 客户端连接回调接口
type ClientConn interface {
    UpdateState(State) error
    ReportError(error)
    ParseServiceConfig(serviceConfigJSON string) *serviceconfig.ParseResult
}

// BuildOptions 构建选项
type BuildOptions struct {
    DialCreds      credentials.TransportCredentials
    CredsBundle    credentials.Bundle
    Dialer         func(context.Context, string) (net.Conn, error)
    Authority      string
}
```

**参数说明表**

| 参数 | 类型 | 必填 | 约束 | 说明 |
|------|------|:----:|------|------|
| target | Target | 是 | 有效的目标地址 | 解析目标 |
| cc | ClientConn | 是 | 有效的回调接口 | 客户端连接回调 |
| opts | BuildOptions | 是 | 构建选项 | 解析器构建选项 |

#### 响应结果

```go
// Resolver 解析器接口
type Resolver interface {
    ResolveNow(ResolveNowOptions)
    Close()
}
```

---

### 3. Resolver.ResolveNow

#### 基本信息
- **名称：** `ResolveNow`
- **签名：** `func ResolveNow(opts ResolveNowOptions)`
- **功能：** 立即触发一次名称解析
- **幂等性：** 是，多次调用安全

#### 请求参数

```go
// ResolveNowOptions 解析选项
type ResolveNowOptions struct{}
```

#### 入口函数实现（DNS示例）

```go
func (d *dnsResolver) ResolveNow(opts resolver.ResolveNowOptions) {
    select {
    case d.rn <- struct{}{}:
    default:
    }
}
```

---

### 4. ClientConn.UpdateState

#### 基本信息
- **名称：** `UpdateState`
- **签名：** `func UpdateState(s State) error`
- **功能：** 更新解析状态，通知新的地址列表

#### 请求参数

```go
// State 解析器状态
type State struct {
    Addresses     []Address                     // 地址列表
    ServiceConfig *serviceconfig.ParseResult    // 服务配置
    Attributes    *attributes.Attributes        // 属性信息
}

// Address 网络地址
type Address struct {
    Addr       string                    // 网络地址
    ServerName string                    // 服务器名称
    Attributes *attributes.Attributes    // 地址属性
    Metadata   any                       // 元数据
}
```

**参数说明表**

| 参数 | 类型 | 必填 | 约束 | 说明 |
|------|------|:----:|------|------|
| s | State | 是 | 非空状态 | 解析器状态 |

---

## 内置解析器

### DNS 解析器

```go
// DNS 解析器实现
type dnsBuilder struct{}

func (b *dnsBuilder) Build(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOptions) (resolver.Resolver, error) {
    host, port, err := parseTarget(target.Endpoint())
    if err != nil {
        return nil, err
    }
    
    d := &dnsResolver{
        host: host,
        port: port,
        cc:   cc,
        rn:   make(chan struct{}, 1),
    }
    
    // 启动解析协程
    go d.watcher()
    
    // 立即执行一次解析
    d.ResolveNow(resolver.ResolveNowOptions{})
    
    return d, nil
}
```

### Passthrough 解析器

```go
// 直连解析器实现
type passthroughBuilder struct{}

func (b *passthroughBuilder) Build(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOptions) (resolver.Resolver, error) {
    addr := resolver.Address{Addr: target.Endpoint()}
    cc.UpdateState(resolver.State{Addresses: []resolver.Address{addr}})
    return &passthroughResolver{}, nil
}
```

## 使用示例

### 注册自定义解析器

```go
package main

import (
    "google.golang.org/grpc/resolver"
)

type customResolver struct {
    cc resolver.ClientConn
}

type customBuilder struct{}

func (b *customBuilder) Build(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOptions) (resolver.Resolver, error) {
    r := &customResolver{cc: cc}
    
    // 执行自定义解析逻辑
    go r.start()
    
    return r, nil
}

func (b *customBuilder) Scheme() string {
    return "custom"
}

func (r *customResolver) ResolveNow(opts resolver.ResolveNowOptions) {
    // 触发立即解析
}

func (r *customResolver) Close() {
    // 清理资源
}

func init() {
    resolver.Register(&customBuilder{})
}
```

### 使用解析器

```go
func main() {
    // 使用 DNS 解析器
    conn, err := grpc.Dial(
        "dns:///example.com:50051",
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
    
    // 使用自定义解析器
    conn2, err := grpc.Dial(
        "custom:///my-service",
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
}
```

## 最佳实践

1. **解析器选择**
   - DNS：适用于标准服务发现
   - Passthrough：适用于直连场景
   - 自定义：集成服务注册中心

2. **性能优化**
   - 缓存解析结果
   - 合理设置解析间隔
   - 异步解析不阻塞

3. **错误处理**
   - 解析失败时的降级策略
   - 重试机制和退避算法
   - 及时报告解析错误

4. **动态更新**
   - 监听地址变化
   - 增量更新地址列表
   - 平滑切换后端实例
