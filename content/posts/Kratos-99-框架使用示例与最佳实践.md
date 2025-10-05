---
title: "Kratos-99-框架使用示例与最佳实践"
date: 2025-10-05T16:00:00+08:00
categories:
  - Kratos
tags: ["Kratos", "最佳实践", "使用示例", "项目架构", "性能优化", "监控", "安全性", "测试"]
series: ["Kratos源码剖析"]
description: "提供Kratos框架的实际使用示例和生产环境最佳实践，包括项目结构、配置管理、错误处理、性能优化、监控告警等实战经验。"
draft: false
---

# Kratos-99-框架使用示例与最佳实践

## 基础使用示例

### 创建简单的 HTTP 服务

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/go-kratos/kratos/v2"
    "github.com/go-kratos/kratos/v2/transport/http"
)

func main() {
    // 创建 HTTP 服务器
    httpSrv := http.NewServer(
        http.Address(":8000"),
        http.Timeout(30*time.Second),
    )
    
    // 注册路由
    httpSrv.HandleFunc("/hello", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, Kratos!")
    })
    
    // 创建应用实例
    app := kratos.New(
        kratos.Name("hello-service"),
        kratos.Version("v1.0.0"),
        kratos.Server(httpSrv),
    )
    
    // 启动应用
    if err := app.Run(); err != nil {
        log.Fatal("Failed to run app:", err)
    }
}
```

### 创建 gRPC 服务

```go
package main

import (
    "context"
    "log"

    "github.com/go-kratos/kratos/v2"
    "github.com/go-kratos/kratos/v2/transport/grpc"
    pb "your-project/api/helloworld/v1"
)

type GreeterService struct {
    pb.UnimplementedGreeterServer
}

func (s *GreeterService) SayHello(ctx context.Context, req *pb.HelloRequest) (*pb.HelloReply, error) {
    return &pb.HelloReply{
        Message: "Hello " + req.Name,
    }, nil
}

func main() {
    // 创建 gRPC 服务器
    grpcSrv := grpc.NewServer(
        grpc.Address(":9000"),
    )
    
    // 注册服务
    pb.RegisterGreeterServer(grpcSrv, &GreeterService{})
    
    // 创建应用实例
    app := kratos.New(
        kratos.Name("greeter-service"),
        kratos.Version("v1.0.0"),
        kratos.Server(grpcSrv),
    )
    
    if err := app.Run(); err != nil {
        log.Fatal("Failed to run app:", err)
    }
}
```

### 集成配置管理

```go
package main

import (
    "github.com/go-kratos/kratos/v2"
    "github.com/go-kratos/kratos/v2/config"
    "github.com/go-kratos/kratos/v2/config/file"
    "github.com/go-kratos/kratos/v2/transport/http"
)

type Config struct {
    Server struct {
        HTTP struct {
            Addr    string `json:"addr"`
            Timeout int    `json:"timeout"`
        } `json:"http"`
    } `json:"server"`
}

func main() {
    // 加载配置
    c := config.New(
        config.WithSource(
            file.NewSource("configs/config.yaml"),
        ),
    )
    if err := c.Load(); err != nil {
        panic(err)
    }
    
    var cfg Config
    if err := c.Scan(&cfg); err != nil {
        panic(err)
    }
    
    // 使用配置创建服务器
    httpSrv := http.NewServer(
        http.Address(cfg.Server.HTTP.Addr),
        http.Timeout(time.Duration(cfg.Server.HTTP.Timeout)*time.Second),
    )
    
    app := kratos.New(
        kratos.Name("config-demo"),
        kratos.Server(httpSrv),
    )
    
    if err := app.Run(); err != nil {
        panic(err)
    }
}
```

### 中间件使用示例

```go
package main

import (
    "context"
    "time"

    "github.com/go-kratos/kratos/v2"
    "github.com/go-kratos/kratos/v2/log"
    "github.com/go-kratos/kratos/v2/middleware"
    "github.com/go-kratos/kratos/v2/middleware/logging"
    "github.com/go-kratos/kratos/v2/middleware/recovery"
    "github.com/go-kratos/kratos/v2/middleware/validate"
    "github.com/go-kratos/kratos/v2/transport/http"
)

func main() {
    logger := log.DefaultLogger
    
    // 创建中间件链
    httpSrv := http.NewServer(
        http.Address(":8000"),
        http.Middleware(
            recovery.Recovery(),              // 异常恢复
            logging.Server(logger),          // 访问日志
            validate.Validator(),            // 参数校验
        ),
    )
    
    app := kratos.New(
        kratos.Name("middleware-demo"),
        kratos.Logger(logger),
        kratos.Server(httpSrv),
    )
    
    if err := app.Run(); err != nil {
        panic(err)
    }
}
```

## 实战经验

### 1. 应用架构设计

**推荐目录结构：**
```
your-project/
├── api/                    # API 定义（protobuf）
├── cmd/                    # 程序入口
├── configs/                # 配置文件
├── internal/               # 私有代码
│   ├── biz/               # 业务逻辑层
│   ├── data/              # 数据访问层
│   ├── service/           # 服务层
│   └── server/            # 服务器配置
├── third_party/           # 第三方依赖
└── go.mod
```

**分层架构原则：**
- **API 层**: 定义对外接口，使用 protobuf
- **Service 层**: 实现 gRPC/HTTP 服务接口
- **Business 层**: 核心业务逻辑，领域模型
- **Data 层**: 数据访问，外部服务调用

### 2. 配置管理最佳实践

**配置文件组织：**
```yaml
# config.yaml
server:
  http:
    addr: 0.0.0.0:8000
    timeout: 1s
  grpc:
    addr: 0.0.0.0:9000
    timeout: 1s

data:
  database:
    driver: mysql
    source: user:password@tcp(127.0.0.1:3306)/db?charset=utf8mb4
  redis:
    addr: 127.0.0.1:6379
    password: ""
    db: 0

registry:
  consul:
    address: 127.0.0.1:8500
    scheme: http
```

**配置热重载：**
```go
func watchConfig(c config.Config) {
    if err := c.Watch("database", func(key string, value config.Value) {
        // 数据库配置变更处理
        var dbConf DatabaseConfig
        if err := value.Scan(&dbConf); err == nil {
            // 重新初始化数据库连接
            updateDatabaseConnection(dbConf)
        }
    }); err != nil {
        log.Error("Failed to watch config:", err)
    }
}
```

### 3. 错误处理策略

**错误定义：**
```go
// errors.proto
enum ErrorReason {
  // 用户相关错误
  USER_NOT_FOUND = 0;
  USER_ALREADY_EXISTS = 1;
  
  // 业务相关错误  
  INSUFFICIENT_BALANCE = 100;
  ORDER_CANCELLED = 101;
}
```

**错误处理：**
```go
import (
    "github.com/go-kratos/kratos/v2/errors"
)

func (s *UserService) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    user, err := s.userRepo.GetByID(ctx, req.Id)
    if err != nil {
        if errors.IsNotFound(err) {
            return nil, errors.NotFound("USER_NOT_FOUND", "用户不存在")
        }
        return nil, errors.Internal("INTERNAL_ERROR", "内部服务错误")
    }
    return user, nil
}
```

### 4. 中间件开发与使用

**自定义中间件：**
```go
func AuthMiddleware(secret string) middleware.Middleware {
    return func(handler middleware.Handler) middleware.Handler {
        return func(ctx context.Context, req interface{}) (interface{}, error) {
            // 从上下文获取传输信息
            if tr, ok := transport.FromServerContext(ctx); ok {
                token := tr.RequestHeader().Get("Authorization")
                if token == "" {
                    return nil, errors.Unauthorized("MISSING_TOKEN", "缺少认证令牌")
                }
                
                // 验证 token
                if !validateToken(token, secret) {
                    return nil, errors.Unauthorized("INVALID_TOKEN", "无效的认证令牌")
                }
                
                // 将用户信息注入上下文
                ctx = context.WithValue(ctx, "user_id", getUserIDFromToken(token))
            }
            
            return handler(ctx, req)
        }
    }
}
```

### 5. 服务注册与发现

**Consul 集成：**
```go
import (
    "github.com/go-kratos/kratos/contrib/registry/consul/v2"
    "github.com/hashicorp/consul/api"
)

func main() {
    // 创建 Consul 客户端
    consulClient, err := api.NewClient(api.DefaultConfig())
    if err != nil {
        panic(err)
    }
    
    // 创建注册器
    registry := consul.New(consulClient)
    
    // 创建应用
    app := kratos.New(
        kratos.Name("user-service"),
        kratos.Version("v1.0.0"),
        kratos.Metadata(map[string]string{
            "env": "production",
        }),
        kratos.Server(httpSrv, grpcSrv),
        kratos.Registrar(registry),
    )
    
    if err := app.Run(); err != nil {
        panic(err)
    }
}
```

## 最佳实践总结

### 1. 项目结构最佳实践

**使用 Kratos CLI 工具：**
```bash
# 创建新项目
kratos new helloworld

# 生成 API 代码
kratos proto add api/helloworld/v1/helloworld.proto
kratos proto client api/helloworld/v1/helloworld.proto
kratos proto server api/helloworld/v1/helloworld.proto
```

**依赖注入模式：**
```go
// 使用 wire 进行依赖注入
//+build wireinject

func wireApp(*conf.Server, *conf.Data, log.Logger) (*kratos.App, func(), error) {
    panic(wire.Build(server.ProviderSet, data.ProviderSet, biz.ProviderSet, service.ProviderSet, newApp))
}
```

### 2. 性能优化建议

**连接池配置：**
```go
// HTTP 客户端连接池
httpClient := http.NewClient(
    http.WithTimeout(30*time.Second),
    http.WithTransport(&http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
    }),
)

// gRPC 客户端连接池
grpcClient := grpc.NewClient(
    grpc.WithTimeout(30*time.Second),
    grpc.WithPoolSize(10),
)
```

**缓存策略：**
```go
// 在业务层使用缓存
func (uc *UserUsecase) GetUser(ctx context.Context, id int64) (*User, error) {
    // 先查缓存
    if user, err := uc.cache.Get(ctx, fmt.Sprintf("user:%d", id)); err == nil {
        return user, nil
    }
    
    // 缓存未命中，查数据库
    user, err := uc.userRepo.GetByID(ctx, id)
    if err != nil {
        return nil, err
    }
    
    // 写入缓存
    uc.cache.Set(ctx, fmt.Sprintf("user:%d", id), user, 5*time.Minute)
    return user, nil
}
```

### 3. 监控与可观测性

**Prometheus 指标：**
```go
import "github.com/go-kratos/kratos/v2/middleware/metrics"

httpSrv := http.NewServer(
    http.Address(":8000"),
    http.Middleware(
        metrics.Server(
            metrics.WithSeconds(prometheus.NewHistogramVec(prometheus.HistogramOpts{
                Namespace: "server",
                Subsystem: "requests",
                Name:      "duration_ms",
                Help:      "server requests duration(ms).",
                Buckets:   []float64{5, 10, 25, 50, 100, 250, 500, 1000},
            }, []string{"kind", "operation"})),
        ),
    ),
)
```

**链路追踪：**
```go
import "github.com/go-kratos/kratos/v2/middleware/tracing"

httpSrv := http.NewServer(
    http.Address(":8000"),  
    http.Middleware(
        tracing.Server(),
    ),
)
```

### 4. 安全性考虑

**TLS 配置：**
```go
// HTTPS 服务器
cert, err := tls.LoadX509KeyPair("server.crt", "server.key")
if err != nil {
    panic(err)
}

httpSrv := http.NewServer(
    http.Address(":8443"),
    http.TLSConfig(&tls.Config{
        Certificates: []tls.Certificate{cert},
    }),
)
```

**输入验证：**
```go
// 使用 validate 标签
type CreateUserRequest struct {
    Name  string `json:"name" validate:"required,min=2,max=50"`
    Email string `json:"email" validate:"required,email"`
    Age   int    `json:"age" validate:"min=0,max=120"`
}
```

### 5. 测试策略

**单元测试：**
```go
func TestUserService_GetUser(t *testing.T) {
    ctrl := gomock.NewController(t)
    defer ctrl.Finish()
    
    mockRepo := mocks.NewMockUserRepo(ctrl)
    service := NewUserService(mockRepo)
    
    mockRepo.EXPECT().
        GetByID(gomock.Any(), int64(1)).
        Return(&User{ID: 1, Name: "test"}, nil)
    
    user, err := service.GetUser(context.Background(), &pb.GetUserRequest{Id: 1})
    assert.NoError(t, err)
    assert.Equal(t, "test", user.Name)
}
```

**集成测试：**
```go
func TestHTTPServer(t *testing.T) {
    httpSrv := http.NewServer()
    app := kratos.New(kratos.Server(httpSrv))
    
    go func() {
        app.Run()
    }()
    defer app.Stop()
    
    time.Sleep(time.Second) // 等待服务启动
    
    resp, err := http.Get("http://localhost:8000/health")
    assert.NoError(t, err)
    assert.Equal(t, 200, resp.StatusCode)
}
```

通过遵循这些最佳实践，可以构建出稳定、高性能、可维护的 Kratos 微服务应用。
