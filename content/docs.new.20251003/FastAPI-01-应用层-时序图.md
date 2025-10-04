# FastAPI 源码剖析 - 01 应用层 - 时序图

## 概述

本文档详细展示 FastAPI 应用层各个关键流程的时序图，包括应用初始化、路由注册、请求处理等核心流程。

---

## 1. 应用初始化时序

### 完整初始化流程

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant FastAPI as FastAPI 类
    participant Starlette as Starlette 父类
    participant Router as APIRouter
    participant ExcHandlers as 异常处理器
    participant Setup as setup 方法
    
    Dev->>FastAPI: FastAPI(title, version, ...)
    Note over Dev: 创建应用实例
    
    FastAPI->>FastAPI: 存储 OpenAPI 配置
    Note over FastAPI: title, version, description<br/>contact, license_info 等
    
    FastAPI->>Starlette: super().__init__()
    Note over Starlette: 初始化父类
    
    Starlette->>Starlette: 初始化基础属性
    Note over Starlette: debug, routes<br/>middleware, state
    
    Starlette-->>FastAPI: 初始化完成
    
    FastAPI->>Router: 创建 APIRouter 实例
    Note over Router: self.router = APIRouter(<br/>  dependencies=dependencies,<br/>  default_response_class=...<br/>)
    
    Router-->>FastAPI: router 实例
    
    FastAPI->>Router: 创建 webhooks router
    Note over Router: self.webhooks = APIRouter()
    
    FastAPI->>FastAPI: 初始化 OpenAPI schema
    Note over FastAPI: self.openapi_schema = None<br/>(延迟生成)
    
    FastAPI->>ExcHandlers: 注册默认异常处理器
    Note over ExcHandlers: HTTPException<br/>RequestValidationError<br/>WebSocketRequestValidationError
    
    ExcHandlers-->>FastAPI: 注册完成
    
    FastAPI->>Setup: self.setup()
    Note over Setup: 设置文档路由
    
    Setup->>Setup: 注册 OpenAPI JSON 路由
    Note over Setup: GET /openapi.json
    
    Setup->>Setup: 注册 Swagger UI 路由
    Note over Setup: GET /docs
    
    Setup->>Setup: 注册 ReDoc 路由
    Note over Setup: GET /redoc
    
    Setup->>Setup: 注册 OAuth2 重定向路由
    Note over Setup: GET /docs/oauth2-redirect
    
    Setup-->>FastAPI: 设置完成
    
    FastAPI-->>Dev: 返回应用实例
    
    Note over Dev,FastAPI: 应用初始化完成，可以注册路由
```

### 初始化步骤说明

**步骤 1-2：创建实例**
- 开发者调用 `FastAPI()` 构造函数
- 传递 OpenAPI 配置参数（title、version 等）

**步骤 3-7：父类初始化**
- 调用 Starlette 的 `__init__` 方法
- 初始化 ASGI 应用基础属性
- 设置 debug 模式、routes、middleware、state 等

**步骤 8-11：创建路由器**
- 创建内部 `APIRouter` 实例（`self.router`）
- 创建 webhooks 路由器（`self.webhooks`）
- 路由器继承应用级配置（dependencies、default_response_class 等）

**步骤 12-13：OpenAPI 初始化**
- 将 `openapi_schema` 设置为 `None`
- 采用延迟生成策略（首次访问时生成）

**步骤 14-16：注册默认异常处理器**
- `HTTPException` → `http_exception_handler`
- `RequestValidationError` → `request_validation_exception_handler`
- `WebSocketRequestValidationError` → `websocket_request_validation_exception_handler`

**步骤 17-25：设置文档路由**
- 调用 `setup()` 方法
- 根据配置注册文档相关路由：
  - `/openapi.json`：返回 OpenAPI JSON schema
  - `/docs`：Swagger UI 文档界面
  - `/redoc`：ReDoc 文档界面
  - `/docs/oauth2-redirect`：OAuth2 重定向页面

---

## 2. 路由注册时序

### 使用装饰器注册路由

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant Decorator as @app.get() 装饰器
    participant App as FastAPI 实例
    participant Router as APIRouter
    participant Route as APIRoute
    participant Dependant as get_dependant
    
    Dev->>Decorator: @app.get("/items/{item_id}")
    Note over Dev: def read_item(item_id: int)
    
    Decorator->>App: app.get(path="/items/{item_id}", ...)
    Note over App: 调用 HTTP 方法路由装饰器
    
    App->>App: api_route(path, methods=["GET"], ...)
    Note over App: 转到通用路由装饰器
    
    App->>App: 创建装饰器函数
    Note over App: def decorator(func)
    
    App-->>Decorator: 返回装饰器
    
    Decorator->>Decorator: decorator(read_item)
    Note over Decorator: 应用装饰器到函数
    
    Decorator->>Router: router.add_api_route()
    Note over Router: 传递路径、函数、配置
    
    Router->>Route: 创建 APIRoute 实例
    Note over Route: APIRoute(<br/>  path="/items/{item_id}",<br/>  endpoint=read_item,<br/>  methods=["GET"],<br/>  ...<br/>)
    
    Route->>Route: 编译路径正则
    Note over Route: compile_path(path)<br/>提取路径参数
    
    Route->>Dependant: get_dependant()
    Note over Dependant: 分析函数签名<br/>构建依赖树
    
    Dependant->>Dependant: 提取参数类型
    Note over Dependant: item_id: int → Path 参数
    
    Dependant->>Dependant: 递归解析依赖
    Note over Dependant: 构建 Dependant 对象
    
    Dependant-->>Route: 返回 Dependant 对象
    
    Route->>Route: 生成 OpenAPI metadata
    Note over Route: 参数、响应、描述等
    
    Route-->>Router: 返回 APIRoute
    
    Router->>Router: 添加到路由列表
    Note over Router: self.routes.append(route)
    
    Router-->>Decorator: 注册完成
    
    Decorator-->>Dev: 返回原函数
    Note over Dev: 函数未被修改<br/>仅注册了路由
```

### 直接调用方法注册

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant App as FastAPI
    participant Router as APIRouter
    
    Dev->>App: app.add_api_route()
    Note over Dev: path="/users/"<br/>endpoint=list_users<br/>methods=["GET"]
    
    App->>Router: router.add_api_route()
    Note over Router: 委托给内部路由器
    
    Router->>Router: 创建 APIRoute
    Note over Router: 同装饰器流程
    
    Router->>Router: 添加到路由列表
    
    Router-->>App: 完成
    App-->>Dev: 完成
```

---

## 3. 路由包含时序（include_router）

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant App as FastAPI
    participant MainRouter as 主路由器
    participant SubRouter as 子路由器
    participant Route as 子路由
    
    Dev->>SubRouter: 创建子路由器
    Note over SubRouter: users_router = APIRouter()
    
    Dev->>SubRouter: 注册子路由
    Note over SubRouter: @users_router.get("/")
    
    SubRouter->>SubRouter: 添加路由到列表
    
    Dev->>App: app.include_router()
    Note over Dev: router=users_router<br/>prefix="/users"<br/>tags=["users"]
    
    App->>MainRouter: router.include_router()
    Note over MainRouter: 委托给主路由器
    
    MainRouter->>SubRouter: 获取子路由列表
    SubRouter-->>MainRouter: routes
    
    loop 遍历子路由
        MainRouter->>Route: 获取路由信息
        Note over Route: path, endpoint, methods
        
        MainRouter->>MainRouter: 合并配置
        Note over MainRouter: 添加 prefix<br/>合并 tags<br/>合并 dependencies
        
        MainRouter->>MainRouter: 创建新路由
        Note over MainRouter: 基于合并后的配置<br/>创建 APIRoute
        
        MainRouter->>MainRouter: 添加到主路由列表
    end
    
    MainRouter-->>App: 包含完成
    App-->>Dev: 完成
    
    Note over Dev,MainRouter: 现在可以访问<br/>GET /users/ → list_users()
```

---

## 4. 中间件注册时序

### 使用 add_middleware

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant App as FastAPI
    participant Starlette as Starlette
    participant MiddlewareStack as 中间件列表
    
    Dev->>App: app.add_middleware()
    Note over Dev: CORSMiddleware<br/>allow_origins=["*"]
    
    App->>Starlette: super().add_middleware()
    Note over Starlette: 调用父类方法
    
    Starlette->>MiddlewareStack: 添加到列表
    Note over MiddlewareStack: Middleware(<br/>  cls=CORSMiddleware,<br/>  kwargs={...}<br/>)
    
    MiddlewareStack-->>Starlette: 完成
    Starlette-->>App: 完成
    App-->>Dev: 完成
    
    Note over Dev,MiddlewareStack: 中间件会在首次请求时<br/>构建中间件栈
```

### 使用 @app.middleware 装饰器

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant Decorator as @app.middleware
    participant App as FastAPI
    participant BaseHTTPMiddleware as BaseHTTPMiddleware
    
    Dev->>Decorator: @app.middleware("http")
    Note over Dev: async def my_middleware(...)
    
    Decorator->>App: app.middleware("http")
    
    App->>App: 创建装饰器函数
    
    App-->>Decorator: 返回装饰器
    
    Decorator->>Decorator: decorator(my_middleware)
    
    Decorator->>App: app.add_middleware()
    Note over App: BaseHTTPMiddleware<br/>dispatch=my_middleware
    
    App->>BaseHTTPMiddleware: 包装函数
    Note over BaseHTTPMiddleware: 将函数转为中间件类
    
    BaseHTTPMiddleware-->>App: 完成
    App-->>Decorator: 完成
    Decorator-->>Dev: 返回原函数
```

---

## 5. 异常处理器注册时序

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant Decorator as @app.exception_handler
    participant App as FastAPI
    participant Starlette as Starlette
    participant Handlers as 异常处理器字典
    
    Dev->>Decorator: @app.exception_handler(CustomException)
    Note over Dev: async def handler(request, exc)
    
    Decorator->>App: app.exception_handler(CustomException)
    
    App->>App: 创建装饰器函数
    
    App-->>Decorator: 返回装饰器
    
    Decorator->>Decorator: decorator(handler)
    
    Decorator->>App: app.add_exception_handler()
    Note over App: CustomException, handler
    
    App->>Starlette: super().add_exception_handler()
    
    Starlette->>Handlers: 添加到字典
    Note over Handlers: exception_handlers[CustomException] = handler
    
    Handlers-->>Starlette: 完成
    Starlette-->>App: 完成
    App-->>Decorator: 完成
    Decorator-->>Dev: 返回原函数
```

---

## 6. OpenAPI Schema 生成时序

### 首次访问文档时生成

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant App as FastAPI
    participant DocsRoute as /docs 路由
    participant OpenAPIMethod as app.openapi()
    participant GetOpenAPI as get_openapi()
    participant Router as 路由器
    participant Routes as 路由列表
    participant Pydantic as Pydantic
    
    Client->>App: GET /docs
    
    App->>DocsRoute: 匹配文档路由
    
    DocsRoute->>DocsRoute: swagger_ui_html()
    Note over DocsRoute: 需要 OpenAPI URL
    
    DocsRoute->>OpenAPIMethod: app.openapi()
    
    OpenAPIMethod->>OpenAPIMethod: 检查缓存
    Note over OpenAPIMethod: if self.openapi_schema
    
    alt 已缓存
        OpenAPIMethod-->>DocsRoute: 返回缓存
    else 未缓存
        OpenAPIMethod->>GetOpenAPI: get_openapi()
        Note over GetOpenAPI: 传递应用配置
        
        GetOpenAPI->>Router: 获取所有路由
        Router-->>GetOpenAPI: self.routes
        
        loop 遍历路由
            GetOpenAPI->>Routes: 获取路由信息
            Note over Routes: path, methods<br/>parameters, responses
            
            GetOpenAPI->>Pydantic: 生成 JSON Schema
            Note over Pydantic: 从 Pydantic 模型<br/>生成 schema
            
            Pydantic-->>GetOpenAPI: JSON Schema
            
            GetOpenAPI->>GetOpenAPI: 构建 Path Item
            Note over GetOpenAPI: OpenAPI path object
        end
        
        GetOpenAPI->>GetOpenAPI: 构建完整 schema
        Note over GetOpenAPI: {<br/>  "openapi": "3.1.0",<br/>  "info": {...},<br/>  "paths": {...},<br/>  "components": {...}<br/>}
        
        GetOpenAPI-->>OpenAPIMethod: 返回 schema
        
        OpenAPIMethod->>OpenAPIMethod: 缓存 schema
        Note over OpenAPIMethod: self.openapi_schema = schema
        
        OpenAPIMethod-->>DocsRoute: 返回 schema
    end
    
    DocsRoute->>DocsRoute: 生成 Swagger UI HTML
    Note over DocsRoute: 嵌入 OpenAPI URL
    
    DocsRoute-->>App: HTML 响应
    App-->>Client: 返回文档页面
```

---

## 7. 完整请求处理时序

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant ASGI as ASGI 服务器
    participant App as FastAPI.__call__
    participant MiddlewareStack as 中间件栈
    participant Router as 路由器
    participant Route as 路由处理
    participant Deps as 依赖注入
    participant Handler as 路由函数
    participant Response as 响应生成
    
    Client->>ASGI: HTTP 请求
    
    ASGI->>App: (scope, receive, send)
    Note over ASGI: ASGI 三元组
    
    alt 首次请求
        App->>App: build_middleware_stack()
        Note over App: 构建中间件栈<br/>（洋葱模型）
    end
    
    App->>MiddlewareStack: 执行中间件
    Note over MiddlewareStack: 按顺序执行<br/>before 部分
    
    MiddlewareStack->>Router: 路由匹配
    Note over Router: 遍历路由列表<br/>匹配路径和方法
    
    Router->>Route: 找到匹配路由
    
    Route->>Deps: solve_dependencies()
    Note over Deps: 解析并执行依赖项
    
    Deps->>Deps: 提取参数
    Note over Deps: 路径、查询、请求体等
    
    Deps->>Deps: Pydantic 验证
    
    alt 验证失败
        Deps-->>Client: 422 验证错误
    else 验证成功
        Deps-->>Route: 已验证参数
        
        Route->>Handler: 调用处理函数
        Note over Handler: handler(**params, **deps)
        
        alt 异步函数
            Handler->>Handler: await 执行
        else 同步函数
            Handler->>Handler: 线程池执行
        end
        
        Handler-->>Route: 返回值
        
        Route->>Response: 生成响应
        Note over Response: 验证响应模型<br/>序列化 JSON
        
        Response-->>MiddlewareStack: 响应对象
        
        MiddlewareStack->>MiddlewareStack: 执行中间件
        Note over MiddlewareStack: 按相反顺序<br/>执行 after 部分
        
        MiddlewareStack-->>App: 最终响应
        
        App-->>ASGI: 响应
        ASGI-->>Client: HTTP 响应
    end
```

---

## 8. 应用生命周期时序

### 使用 Lifespan 上下文管理器

```mermaid
sequenceDiagram
    autonumber
    participant Server as ASGI 服务器
    participant App as FastAPI
    participant Lifespan as lifespan 上下文
    participant Resources as 资源管理
    participant Router as 路由处理
    
    Server->>App: 启动应用
    
    App->>Lifespan: __aenter__()
    Note over Lifespan: 进入 async with 块
    
    Lifespan->>Resources: 初始化资源
    Note over Resources: 创建数据库连接池<br/>连接 Redis<br/>加载配置
    
    Resources->>App: app.state.db = pool
    Note over App: 存储到应用状态
    
    Lifespan-->>App: 启动完成
    
    Note over App,Router: 应用运行中<br/>处理请求
    
    loop 处理请求
        Server->>Router: 请求
        Router->>Resources: 使用资源
        Note over Resources: app.state.db
        Resources-->>Router: 响应
        Router-->>Server: 响应
    end
    
    Server->>App: 关闭应用
    Note over Server: SIGTERM/SIGINT
    
    App->>Lifespan: __aexit__()
    Note over Lifespan: 离开 async with 块
    
    Lifespan->>Resources: 清理资源
    Note over Resources: 关闭数据库连接<br/>关闭 Redis<br/>保存状态
    
    Resources-->>Lifespan: 清理完成
    
    Lifespan-->>App: 关闭完成
    
    App-->>Server: 退出
```

### 已废弃的 on_event 方式

```mermaid
sequenceDiagram
    autonumber
    participant Server as ASGI 服务器
    participant App as FastAPI
    participant StartupHandlers as startup 处理器列表
    participant ShutdownHandlers as shutdown 处理器列表
    
    Server->>App: 启动应用
    
    App->>StartupHandlers: 执行所有 startup 事件
    
    loop 遍历处理器
        StartupHandlers->>StartupHandlers: await handler()
        Note over StartupHandlers: 按注册顺序执行
    end
    
    StartupHandlers-->>App: 启动完成
    
    Note over App: 应用运行中
    
    Server->>App: 关闭应用
    
    App->>ShutdownHandlers: 执行所有 shutdown 事件
    
    loop 遍历处理器
        ShutdownHandlers->>ShutdownHandlers: await handler()
        Note over ShutdownHandlers: 按注册顺序执行
    end
    
    ShutdownHandlers-->>App: 关闭完成
    
    App-->>Server: 退出
```

---

## 9. 中间件栈构建时序

### 首次请求时构建

```mermaid
sequenceDiagram
    autonumber
    participant Request as 首次请求
    participant App as FastAPI
    participant Build as build_middleware_stack
    participant MiddlewareList as 中间件列表
    participant Stack as 中间件栈
    participant Router as 路由器
    
    Request->>App: __call__(scope, receive, send)
    
    App->>App: 检查中间件栈
    Note over App: if not hasattr(self, '_stack')
    
    App->>Build: build_middleware_stack()
    
    Build->>MiddlewareList: 获取中间件列表
    Note over MiddlewareList: self.user_middleware +<br/>default middlewares
    
    Build->>Build: 添加默认中间件
    Note over Build: ServerErrorMiddleware<br/>ExceptionMiddleware<br/>AsyncExitStackMiddleware
    
    Build->>Router: app = self.router
    Note over Router: 最内层是路由器
    
    loop 反向遍历中间件
        Build->>Stack: 包装上一层
        Note over Stack: app = Middleware(app, ...)
    end
    
    Build-->>App: 返回中间件栈
    
    App->>App: 缓存中间件栈
    Note over App: self._stack = stack
    
    App->>Stack: 执行请求
    Note over Stack: await stack(scope, receive, send)
    
    Stack-->>App: 响应
    App-->>Request: 返回响应
    
    Note over App: 后续请求直接使用缓存的栈
```

---

## 总结

FastAPI 应用层的时序流程包括：

1. **初始化**：创建 FastAPI 实例，设置配置，注册默认处理器
2. **路由注册**：通过装饰器或方法注册路由，构建依赖树
3. **路由包含**：合并子路由器到主应用
4. **中间件注册**：添加中间件到列表
5. **异常处理器**：注册自定义异常处理器
6. **OpenAPI 生成**：延迟生成并缓存 schema
7. **请求处理**：完整的请求-响应生命周期
8. **应用生命周期**：启动和关闭资源管理
9. **中间件栈**：首次请求时构建洋葱模型

这些时序图展示了 FastAPI 应用层的核心工作流程和组件交互。

