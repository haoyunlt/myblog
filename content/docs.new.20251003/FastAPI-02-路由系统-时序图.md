# FastAPI-02-路由系统-时序图

> **文档版本**: v1.0  
> **FastAPI 版本**: 0.118.0  
> **创建日期**: 2025年10月4日

---

## 📋 目录

1. [时序图概览](#时序图概览)
2. [路由注册流程](#路由注册流程)
3. [路由匹配流程](#路由匹配流程)
4. [子路由包含流程](#子路由包含流程)
5. [路径参数解析流程](#路径参数解析流程)
6. [请求处理完整流程](#请求处理完整流程)
7. [WebSocket路由处理流程](#websocket路由处理流程)

---

## 时序图概览

### 核心流程清单

| # | 流程名称 | 参与组件 | 复杂度 | 频率 |
|---|---------|----------|--------|------|
| 1 | 路由注册流程 | APIRouter, APIRoute, Dependant | ⭐⭐ | 启动时 |
| 2 | 路由匹配流程 | Router, Route, Path Regex | ⭐⭐⭐ | 每个请求 |
| 3 | 子路由包含流程 | APIRouter, APIRoute | ⭐⭐ | 启动时 |
| 4 | 路径参数解析 | Convertor, Path Regex | ⭐⭐ | 每个请求 |
| 5 | 请求处理完整流程 | 所有组件 | ⭐⭐⭐⭐ | 每个请求 |
| 6 | WebSocket处理流程 | APIWebSocketRoute, Dependant | ⭐⭐⭐ | WS连接 |

---

## 路由注册流程

### 1.1 使用装饰器注册路由

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant Decorator as @router.get()
    participant Router as APIRouter
    participant Route as APIRoute
    participant Dep as get_dependant()
    participant Compile as compile_path()
    
    Dev->>Decorator: @router.get("/users/{id}")
    Note over Dev: 定义路由处理函数
    Decorator->>Router: api_route(path, methods=["GET"])
    Router->>Router: add_api_route()
    Note over Router: 合并配置（tags, dependencies）
    
    Router->>Route: 创建APIRoute实例
    Route->>Dep: get_dependant(call=endpoint)
    Dep->>Dep: 分析函数签名
    Note over Dep: 提取参数、类型、默认值
    Dep-->>Route: 返回Dependant树
    
    Route->>Compile: compile_path("/users/{id}")
    Compile->>Compile: 生成正则表达式
    Note over Compile: path_regex, param_convertors
    Compile-->>Route: 返回编译结果
    
    Route->>Route: get_route_handler()
    Note over Route: 创建ASGI应用
    Route-->>Router: 返回Route实例
    
    Router->>Router: routes.append(route)
    Router-->>Decorator: 完成注册
    Decorator-->>Dev: 返回原函数
```

**时序图说明**：
1. **图意概述**: 展示使用装饰器注册路由的完整流程，包括依赖解析和路径编译
2. **关键字段**: `dependant`存储所有参数信息；`path_regex`用于路径匹配
3. **边界条件**: 函数签名错误会在get_dependant()阶段报错；路径格式错误在compile_path()阶段报错
4. **异常路径**: 路径格式错误抛出ValueError；参数类型不支持抛出FastAPIError
5. **性能假设**: 路由注册在启动时完成一次，O(n)复杂度，n为参数数量
6. **版本兼容**: FastAPI 0.100+支持所有类型注解

### 1.2 直接调用add_api_route()注册

```mermaid
sequenceDiagram
    autonumber
    participant User as 调用方
    participant Router as APIRouter
    participant Route as APIRoute
    
    User->>Router: add_api_route("/items", endpoint)
    Router->>Router: 获取默认配置
    Note over Router: self.tags, self.dependencies等
    Router->>Router: 合并用户配置
    Note over Router: tags = user_tags or self.tags
    
    Router->>Route: APIRoute(prefix + path, ...)
    Note over Route: 完整路径 = prefix + path
    Route->>Route: __init__()
    Note over Route: 构建依赖树、编译路径
    Route-->>Router: 返回route实例
    
    Router->>Router: self.routes.append(route)
    Router-->>User: 完成
```

**时序图说明**：
1. **图意概述**: 直接调用add_api_route()的简化流程
2. **关键点**: prefix会自动与path拼接；tags和dependencies会合并
3. **边界条件**: prefix为空时直接使用path；tags为None时使用空列表
4. **性能**: O(1)时间复杂度，仅做列表append操作

---

## 路由匹配流程

### 2.1 完整路由匹配流程

```mermaid
sequenceDiagram
    autonumber
    participant Request as 请求
    participant App as FastAPI App
    participant Router as Router
    participant Route as APIRoute
    participant Regex as path_regex
    participant Conv as Convertor
    
    Request->>App: HTTP GET /users/123
    App->>Router: route(scope)
    Note over Router: scope = {"path": "/users/123", ...}
    
    loop 遍历routes
        Router->>Route: matches(scope)
        Route->>Regex: regex.match(path)
        Regex-->>Route: match对象 or None
        
        alt 路径匹配成功
            Route->>Route: 检查HTTP方法
            alt 方法匹配成功
                Route->>Conv: 提取并转换参数
                Note over Conv: "123" -> 123 (int)
                Conv-->>Route: {"user_id": 123}
                Route-->>Router: (Match.FULL, scope)
                Note over Router: 更新scope["path_params"]
            else 方法不匹配
                Route-->>Router: (Match.NONE, {})
            end
        else 路径不匹配
            Route-->>Router: (Match.NONE, {})
        end
    end
    
    Router-->>App: 匹配结果
    
    alt 匹配成功
        App->>Route: 调用endpoint
    else 未匹配
        App->>App: 404 Not Found
    end
```

**时序图说明**：
1. **图意概述**: 展示请求到达后，路由匹配的完整流程，包括路径匹配和参数提取
2. **关键字段**: scope["path"]用于匹配；scope["path_params"]存储提取的参数
3. **边界条件**: 按注册顺序匹配，先匹配到的优先；无匹配返回404
4. **异常路径**: 路径匹配成功但参数转换失败，继续尝试下一个路由
5. **性能假设**: 路由数量n，平均匹配时间O(n)；静态路由O(1)
6. **优化点**: 静态路由应放在参数路由之前，可减少匹配次数

### 2.2 路径参数类型转换

```mermaid
sequenceDiagram
    autonumber
    participant Route as APIRoute
    participant Conv as Convertor
    participant Type as 类型系统
    
    Route->>Conv: convert("123")
    Conv->>Conv: 识别转换器类型
    Note over Conv: IntConvertor, FloatConvertor等
    
    alt IntConvertor
        Conv->>Type: int("123")
        Type-->>Conv: 123
    else FloatConvertor
        Conv->>Type: float("3.14")
        Type-->>Conv: 3.14
    else UUIDConvertor
        Conv->>Type: UUID("...")
        Type-->>Conv: UUID对象
    else PathConvertor
        Conv-->>Conv: 直接返回字符串
    end
    
    Conv-->>Route: 转换后的值
```

**时序图说明**：
1. **图意概述**: 路径参数的类型转换过程
2. **关键点**: 每种参数类型有对应的Convertor
3. **异常路径**: 转换失败抛出ValueError
4. **性能**: O(1)时间复杂度

---

## 子路由包含流程

### 3.1 include_router()完整流程

```mermaid
sequenceDiagram
    autonumber
    participant App as FastAPI
    participant MainRouter as 主路由器
    participant SubRouter as 子路由器
    participant Routes as routes列表
    participant Route as APIRoute
    
    App->>MainRouter: include_router(sub, prefix="/api")
    MainRouter->>MainRouter: 验证prefix格式
    Note over MainRouter: 必须以/开头，不以/结尾
    
    MainRouter->>SubRouter: 获取routes
    SubRouter-->>MainRouter: 返回路由列表
    
    loop 遍历子路由器的每个路由
        MainRouter->>Route: 获取路由信息
        MainRouter->>MainRouter: 叠加配置
        Note over MainRouter: 合并prefix, tags, dependencies
        
        MainRouter->>MainRouter: add_api_route()
        Note over MainRouter: prefix="/api" + route.path="/users"<br/>= "/api/users"
        
        MainRouter->>Routes: 创建新路由并添加
        Note over Routes: 新路由包含合并后的配置
    end
    
    MainRouter-->>App: 完成包含
```

**时序图说明**：
1. **图意概述**: 展示子路由包含的配置合并和路由复制过程
2. **关键字段**: prefix叠加拼接；tags和dependencies列表合并；responses字典合并
3. **边界条件**: 可以多层嵌套包含；空prefix有效
4. **异常路径**: prefix格式错误抛出AssertionError
5. **性能假设**: 子路由数量m，时间复杂度O(m)
6. **设计理由**: 通过复制路由实现配置继承，而不是运行时动态计算

### 3.2 多层嵌套包含

```mermaid
sequenceDiagram
    autonumber
    participant App as FastAPI
    participant ApiRouter as api_router
    participant UsersRouter as users_router
    participant Route as 实际路由
    
    UsersRouter->>UsersRouter: add_api_route("/", endpoint)
    Note over UsersRouter: 路径: "/"
    
    ApiRouter->>UsersRouter: include_router(prefix="/users")
    ApiRouter->>ApiRouter: 创建新路由
    Note over ApiRouter: 路径: "/users" + "/" = "/users"
    
    App->>ApiRouter: include_router(prefix="/api/v1")
    App->>App: 创建新路由
    Note over App: 路径: "/api/v1" + "/users" = "/api/v1/users"
    
    Note over Route: 最终路径: /api/v1/users
```

**时序图说明**：
1. **图意概述**: 多层路由嵌套时的prefix叠加过程
2. **关键点**: 每层include_router都会重新注册路由，叠加prefix
3. **边界条件**: 理论上支持无限层嵌套
4. **性能**: 嵌套层数k，路由数n，总复杂度O(k*n)

---

## 路径参数解析流程

### 4.1 compile_path()路径编译

```mermaid
sequenceDiagram
    autonumber
    participant Router as APIRouter
    participant Compile as compile_path()
    participant Regex as re模块
    participant Conv as Convertor工厂
    
    Router->>Compile: compile_path("/users/{user_id:int}/posts/{post_id}")
    Compile->>Compile: 解析路径模式
    Note over Compile: 识别参数：user_id, post_id
    
    Compile->>Conv: 创建IntConvertor (user_id)
    Conv-->>Compile: IntConvertor实例
    
    Compile->>Conv: 创建StringConvertor (post_id)
    Conv-->>Compile: StringConvertor实例
    
    Compile->>Compile: 构建正则表达式
    Note over Compile: ^/users/(?P<user_id>[0-9]+)/posts/(?P<post_id>[^/]+)$
    
    Compile->>Regex: re.compile(pattern)
    Regex-->>Compile: Pattern对象
    
    Compile-->>Router: (regex, format, convertors)
    Note over Router: format="/users/{user_id}/posts/{post_id}"<br/>convertors={"user_id": IntConvertor, ...}
```

**时序图说明**：
1. **图意概述**: 路径编译过程，将路径模式转换为正则表达式和参数转换器
2. **关键字段**: regex用于匹配；convertors用于类型转换
3. **边界条件**: 支持嵌套参数；支持自定义转换器
4. **性能**: 编译在启动时完成，O(m)复杂度，m为参数数量

### 4.2 请求时参数提取

```mermaid
sequenceDiagram
    autonumber
    participant Request as 请求
    participant Route as APIRoute
    participant Regex as path_regex
    participant Conv as Convertor
    participant Scope as scope字典
    
    Request->>Route: /users/123/posts/456
    Route->>Regex: regex.match("/users/123/posts/456")
    Regex->>Regex: 匹配各个捕获组
    Regex-->>Route: match.groupdict()
    Note over Route: {"user_id": "123", "post_id": "456"}
    
    loop 遍历每个参数
        Route->>Conv: convert("123")
        Conv->>Conv: int("123")
        Conv-->>Route: 123
    end
    
    Route->>Scope: 更新path_params
    Scope->>Scope: {"user_id": 123, "post_id": "456"}
    Scope-->>Route: 完成
```

---

## 请求处理完整流程

### 5.1 从请求到响应的完整链路

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant Server as ASGI Server
    participant App as FastAPI
    participant Router as Router
    participant Route as APIRoute
    participant Handler as get_route_handler()
    participant Deps as solve_dependencies()
    participant Endpoint as 端点函数
    participant Valid as 响应验证
    participant Response as Response
    
    Client->>Server: HTTP Request
    Server->>App: ASGI(scope, receive, send)
    App->>Router: route(scope)
    
    Router->>Route: matches(scope)
    Route->>Route: 路径匹配 + 参数提取
    Route-->>Router: Match.FULL + scope
    Router-->>App: 返回matched route
    
    App->>Handler: 调用ASGI app
    Handler->>Deps: solve_dependencies(request, dependant)
    Deps->>Deps: 递归解析依赖树
    Note over Deps: 提取参数、调用依赖函数
    Deps-->>Handler: values字典
    
    Handler->>Endpoint: endpoint(**values)
    Endpoint->>Endpoint: 执行业务逻辑
    Endpoint-->>Handler: 返回结果
    
    alt 返回Response对象
        Handler-->>App: 直接返回
    else 返回其他对象
        Handler->>Valid: 验证响应模型
        Valid->>Valid: Pydantic验证
        Valid-->>Handler: 验证后的数据
        Handler->>Response: 创建Response
        Response-->>Handler: Response对象
    end
    
    Handler-->>App: Response
    App-->>Server: Response
    Server-->>Client: HTTP Response
```

**时序图说明**：
1. **图意概述**: 展示从客户端请求到服务端响应的完整处理链路
2. **关键字段**: scope传递请求信息；values存储解析后的参数
3. **边界条件**: 依赖解析失败返回422；业务逻辑异常返回500
4. **异常路径**: 验证失败→RequestValidationError→422响应
5. **性能假设**: 依赖数量d，参数数量p，复杂度O(d+p)
6. **优化点**: 依赖缓存可减少重复计算；响应模型验证可选

### 5.2 依赖注入详细流程

```mermaid
sequenceDiagram
    autonumber
    participant Handler as Route Handler
    participant Solve as solve_dependencies()
    participant Dep1 as 依赖1
    participant Dep2 as 依赖2 (嵌套)
    participant Cache as 依赖缓存
    
    Handler->>Solve: solve_dependencies(dependant)
    Solve->>Solve: 遍历依赖树
    
    loop 处理每个依赖
        Solve->>Cache: 检查缓存
        alt 缓存命中
            Cache-->>Solve: 返回缓存值
        else 缓存未命中
            Solve->>Dep1: 调用依赖函数
            
            alt 依赖有子依赖
                Dep1->>Solve: 递归解析子依赖
                Solve->>Dep2: 调用子依赖
                Dep2-->>Solve: 子依赖结果
                Solve-->>Dep1: 传入子依赖结果
            end
            
            Dep1-->>Solve: 依赖结果
            Solve->>Cache: 缓存结果
        end
    end
    
    Solve-->>Handler: 所有依赖的值
```

---

## WebSocket路由处理流程

### 6.1 WebSocket连接建立与处理

```mermaid
sequenceDiagram
    autonumber
    participant Client as WebSocket客户端
    participant App as FastAPI
    participant Route as APIWebSocketRoute
    participant Deps as solve_dependencies()
    participant Endpoint as WebSocket端点
    participant WS as WebSocket连接
    
    Client->>App: WebSocket连接请求
    App->>Route: matches(scope)
    Route->>Route: 路径匹配
    Route-->>App: Match.FULL
    
    App->>Route: 调用websocket_app
    Route->>Deps: solve_dependencies(websocket)
    Deps->>Deps: 解析依赖
    Deps-->>Route: values
    
    Route->>Endpoint: endpoint(websocket, **values)
    Endpoint->>WS: await websocket.accept()
    WS-->>Client: 连接建立
    
    loop WebSocket通信
        Client->>WS: 发送消息
        WS->>Endpoint: await websocket.receive_text()
        Endpoint->>Endpoint: 处理消息
        Endpoint->>WS: await websocket.send_text()
        WS-->>Client: 响应消息
    end
    
    alt 正常关闭
        Client->>WS: 关闭连接
        WS->>Endpoint: 触发disconnect
        Endpoint->>Endpoint: 清理资源
    else 异常关闭
        Endpoint->>Endpoint: 捕获异常
        Endpoint->>WS: 关闭连接
    end
    
    Endpoint-->>Route: 完成
    Route-->>App: 完成
```

**时序图说明**：
1. **图意概述**: WebSocket从连接建立到关闭的完整生命周期
2. **关键字段**: websocket对象贯穿整个生命周期；依赖在连接建立时解析一次
3. **边界条件**: 依赖解析失败拒绝连接；消息处理异常关闭连接
4. **异常路径**: 连接被拒绝→WebSocketDisconnect；消息格式错误→关闭连接
5. **性能假设**: 连接保持期间，依赖不会重新解析
6. **资源管理**: yield依赖在连接关闭时自动清理

---

## 📊 时序图总结

### 核心流程对比

| 流程 | 执行时机 | 频率 | 复杂度 | 性能影响 |
|------|----------|------|--------|----------|
| 路由注册 | 应用启动 | 一次 | O(n) | 无 |
| 路由匹配 | 每个请求 | 高频 | O(r) | 中 |
| 参数提取 | 匹配成功后 | 高频 | O(p) | 低 |
| 依赖解析 | 每个请求 | 高频 | O(d) | 高 |
| 响应验证 | 返回响应时 | 高频 | O(f) | 中 |

*r=路由数量, p=参数数量, d=依赖数量, f=响应字段数量*

### 性能优化建议

1. **路由匹配优化**
   - ✅ 静态路由放在参数路由之前
   - ✅ 减少路由总数
   - ✅ 使用精确匹配而非模糊匹配

2. **依赖解析优化**
   - ✅ 启用依赖缓存
   - ✅ 减少依赖层级
   - ✅ 避免在依赖中执行IO操作

3. **响应验证优化**
   - ✅ 仅在开发环境启用response_model
   - ✅ 使用exclude_unset减少验证字段
   - ✅ 对大响应使用StreamingResponse

---

## 📚 相关文档

- [FastAPI-02-路由系统-概览](./FastAPI-02-路由系统-概览.md) - 路由系统架构
- [FastAPI-02-路由系统-API](./FastAPI-02-路由系统-API.md) - 路由API详解
- [FastAPI-02-路由系统-数据结构](./FastAPI-02-路由系统-数据结构.md) - 路由数据结构
- [FastAPI-03-依赖注入-时序图](./FastAPI-03-依赖注入-时序图.md) - 依赖解析详细流程

---

*本文档生成于 2025年10月4日，基于 FastAPI 0.118.0*

