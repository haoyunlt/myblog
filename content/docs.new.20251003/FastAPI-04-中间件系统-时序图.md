# FastAPI-04-中间件系统-时序图

> **文档版本**: v1.0  
> **FastAPI 版本**: 0.118.0  
> **创建日期**: 2025年10月4日

---

## 📋 目录

1. [时序图概览](#时序图概览)
2. [中间件栈构建流程](#中间件栈构建流程)
3. [中间件注册流程](#中间件注册流程)
4. [请求通过中间件链流程](#请求通过中间件链流程)
5. [AsyncExitStack生命周期](#asyncexitstack生命周期)
6. [CORS预检请求流程](#cors预检请求流程)
7. [GZIP压缩流程](#gzip压缩流程)

---

## 时序图概览

### 核心流程清单

| # | 流程名称 | 执行时机 | 复杂度 | 频率 |
|---|---------|----------|--------|------|
| 1 | 中间件栈构建 | 应用启动 | ⭐⭐ | 一次 |
| 2 | 中间件注册 | 启动前配置 | ⭐ | 多次 |
| 3 | 请求处理链 | 每个请求 | ⭐⭐⭐⭐ | 高频 |
| 4 | AsyncExitStack管理 | 每个请求 | ⭐⭐⭐ | 高频 |
| 5 | CORS预检 | OPTIONS请求 | ⭐⭐ | 中频 |
| 6 | GZIP压缩 | 符合条件的响应 | ⭐⭐ | 高频 |

---

## 中间件栈构建流程

### 1.1 build_middleware_stack()完整流程

```mermaid
sequenceDiagram
    autonumber
    participant App as FastAPI
    participant Build as build_middleware_stack()
    participant UserMW as user_middleware列表
    participant Stack as 中间件栈
    participant AsyncMW as AsyncExitStackMiddleware
    participant Router as Router
    
    App->>Build: build_middleware_stack()
    Build->>Router: app = self.router
    Note over Build: 从最内层开始
    
    Build->>AsyncMW: 创建AsyncExitStackMiddleware
    AsyncMW-->>Build: app = AsyncExitStackMiddleware(app)
    
    Build->>UserMW: reversed(self.user_middleware)
    Note over UserMW: 逆序遍历用户中间件
    
    loop 每个用户中间件（逆序）
        Build->>Build: middleware = user_middleware[i]
        Build->>Stack: app = middleware.cls(app, **options)
        Note over Stack: 嵌套包装
    end
    
    Build->>Stack: app = ExceptionMiddleware(app)
    Build->>Stack: app = ServerErrorMiddleware(app)
    
    Build-->>App: 返回最外层app
```

**时序图说明**：
1. **图意概述**: 展示中间件栈的构建过程，从内到外逐层包装
2. **关键字段**: app变量不断被新的中间件包装；user_middleware逆序遍历
3. **边界条件**: user_middleware可以为空；Router是最内层
4. **执行顺序**: Router → AsyncExitStack → 用户中间件(逆序) → 异常处理
5. **性能假设**: 构建在启动时完成一次，O(n)复杂度，n为中间件数量
6. **设计理由**: 通过嵌套调用实现洋葱模型；逆序确保后添加的先执行

### 1.2 中间件嵌套结构

```mermaid
graph TD
    A[构建开始] --> B[app = Router]
    B --> C["app = AsyncExitStackMiddleware(app)"]
    C --> D["app = UserMiddleware3(app)"]
    D --> E["app = UserMiddleware2(app)"]
    E --> F["app = UserMiddleware1(app)"]
    F --> G["app = ExceptionMiddleware(app)"]
    G --> H["app = ServerErrorMiddleware(app)"]
    H --> I[构建完成]
    
    style B fill:#bbf
    style C fill:#f9f
    style H fill:#fbb
```

---

## 中间件注册流程

### 2.1 add_middleware()注册流程

```mermaid
sequenceDiagram
    autonumber
    participant User as 开发者
    participant App as FastAPI
    participant List as user_middleware
    participant MW as Middleware配置
    
    User->>App: add_middleware(CORSMiddleware, allow_origins=[...])
    App->>MW: 创建Middleware(cls=CORSMiddleware, options={...})
    App->>List: insert(0, middleware)
    Note over List: 插入到列表开头
    
    User->>App: add_middleware(GZIPMiddleware, minimum_size=1000)
    App->>MW: 创建Middleware(cls=GZIPMiddleware, options={...})
    App->>List: insert(0, middleware)
    Note over List: 列表: [GZIP, CORS]
    
    Note over App: 实际执行顺序: GZIP → CORS
```

**时序图说明**：
1. **图意概述**: 展示中间件注册时的配置存储过程
2. **关键点**: 使用`insert(0)`而不是`append()`，确保后添加的先执行
3. **边界条件**: 可以多次添加同一个中间件类（不同配置）
4. **性能**: O(1)时间复杂度（列表insert操作）

---

## 请求通过中间件链流程

### 3.1 完整中间件链执行

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant Server as ServerErrorMiddleware
    participant Exception as ExceptionMiddleware
    participant GZIP as GZIPMiddleware
    participant CORS as CORSMiddleware
    participant AsyncStack as AsyncExitStackMiddleware
    participant Router as Router
    participant Endpoint as 端点函数
    
    Client->>Server: HTTP Request
    Note over Client,Server: 请求阶段（从外到内）
    
    Server->>Exception: __call__(scope, receive, send)
    Exception->>GZIP: __call__(scope, receive, send)
    GZIP->>CORS: __call__(scope, receive, send)
    CORS->>AsyncStack: __call__(scope, receive, send)
    AsyncStack->>AsyncStack: 创建AsyncExitStack
    AsyncStack->>Router: __call__(scope, receive, send)
    Router->>Endpoint: 路由匹配并调用
    
    Note over Endpoint: 执行业务逻辑
    
    Endpoint-->>Router: 返回响应数据
    Note over Endpoint,Client: 响应阶段（从内到外）
    
    Router-->>AsyncStack: Response
    AsyncStack->>AsyncStack: 清理AsyncExitStack
    AsyncStack-->>CORS: Response
    CORS->>CORS: 添加CORS头
    CORS-->>GZIP: Response
    GZIP->>GZIP: 压缩响应体
    GZIP-->>Exception: Response
    Exception-->>Server: Response
    Server-->>Client: HTTP Response
```

**时序图说明**：
1. **图意概述**: 展示请求从外到内穿过中间件链，响应从内到外返回的完整流程
2. **关键阶段**: 请求阶段（外→内）；业务逻辑；响应阶段（内→外）
3. **边界条件**: 任何中间件可以短路返回；异常会被ExceptionMiddleware捕获
4. **异常路径**: 异常 → ExceptionMiddleware → 转换为HTTP响应 → 返回客户端
5. **性能假设**: 中间件数量n，时间复杂度O(n)
6. **设计理由**: 洋葱模型确保每个中间件都能处理请求和响应

### 3.2 中间件短路返回

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant MW1 as Middleware 1
    participant MW2 as Middleware 2
    participant MW3 as Middleware 3
    participant App as Application
    
    Client->>MW1: Request
    MW1->>MW2: 转发
    MW2->>MW2: 检查条件
    
    alt 条件不满足（如认证失败）
        MW2-->>MW1: 401 Unauthorized
        Note over MW2: 短路返回，不调用下一层
        MW1-->>Client: 401 Unauthorized
    else 条件满足
        MW2->>MW3: 转发
        MW3->>App: 转发
        App-->>MW3: Response
        MW3-->>MW2: Response
        MW2-->>MW1: Response
        MW1-->>Client: Response
    end
```

---

## AsyncExitStack生命周期

### 4.1 AsyncExitStackMiddleware完整流程

```mermaid
sequenceDiagram
    autonumber
    participant Request as 请求
    participant AESM as AsyncExitStackMiddleware
    participant Stack as AsyncExitStack
    participant Scope as scope字典
    participant Next as 下一层中间件
    participant Deps as yield依赖
    
    Request->>AESM: __call__(scope, receive, send)
    AESM->>Stack: async with AsyncExitStack()
    Stack->>Stack: __aenter__()
    Note over Stack: 创建stack实例
    
    AESM->>Scope: scope["fastapi_astack"] = stack
    AESM->>Next: await self.app(scope, receive, send)
    
    Next->>Deps: 解析yield依赖
    Deps->>Stack: stack.enter_async_context(dependency)
    Note over Stack: 注册清理回调
    Deps-->>Next: yield的值
    
    Next->>Next: 执行业务逻辑
    Next-->>AESM: 响应完成
    
    AESM->>Stack: __aexit__()
    Note over Stack: 触发清理
    
    loop 逆序清理依赖
        Stack->>Deps: 调用finally块
        Deps->>Deps: 释放资源
        Deps-->>Stack: 完成
    end
    
    Stack-->>AESM: 清理完成
    AESM-->>Request: 响应返回
```

**时序图说明**：
1. **图意概述**: 展示AsyncExitStack的完整生命周期，从创建到清理
2. **关键字段**: scope["fastapi_astack"]存储stack；依赖注册到stack
3. **边界条件**: 即使发生异常，__aexit__()也会执行；清理逆序进行
4. **异常路径**: 异常 → __aexit__()仍然执行 → 清理完成后异常继续传播
5. **性能假设**: 清理操作应该快速完成
6. **设计理由**: 确保资源正确释放，防止内存泄漏

### 4.2 多个yield依赖的清理顺序

```mermaid
sequenceDiagram
    autonumber
    participant Stack as AsyncExitStack
    participant Dep1 as yield依赖1
    participant Dep2 as yield依赖2
    participant Dep3 as yield依赖3
    
    Note over Stack: 注入阶段（FIFO）
    Stack->>Dep1: enter_async_context(dep1)
    Dep1-->>Stack: 注册清理1
    Stack->>Dep2: enter_async_context(dep2)
    Dep2-->>Stack: 注册清理2
    Stack->>Dep3: enter_async_context(dep3)
    Dep3-->>Stack: 注册清理3
    
    Note over Stack: 清理阶段（LIFO）
    Stack->>Dep3: 清理依赖3
    Stack->>Dep2: 清理依赖2
    Stack->>Dep1: 清理依赖1
```

---

## CORS预检请求流程

### 5.1 OPTIONS预检请求处理

```mermaid
sequenceDiagram
    autonumber
    participant Browser as 浏览器
    participant CORS as CORSMiddleware
    participant Config as CORS配置
    participant App as 下一层应用
    
    Browser->>CORS: OPTIONS /api/users
    Note over Browser: Origin: https://example.com<br/>Access-Control-Request-Method: POST<br/>Access-Control-Request-Headers: Content-Type
    
    CORS->>CORS: 识别为预检请求
    Note over CORS: method == "OPTIONS" &&<br/>Access-Control-Request-Method存在
    
    CORS->>Config: 检查allow_origins
    
    alt Origin不在允许列表
        CORS-->>Browser: 403 Forbidden
        Note over CORS: 不添加CORS头
    else Origin允许
        CORS->>Config: 检查allow_methods
        
        alt Method不允许
            CORS-->>Browser: 403 Forbidden
        else Method允许
            CORS->>Config: 检查allow_headers
            
            alt Headers不允许
                CORS-->>Browser: 403 Forbidden
            else Headers允许
                CORS->>CORS: 构建预检响应
                Note over CORS: Access-Control-Allow-Origin<br/>Access-Control-Allow-Methods<br/>Access-Control-Allow-Headers<br/>Access-Control-Max-Age
                CORS-->>Browser: 200 OK + CORS Headers
            end
        end
    end
```

**时序图说明**：
1. **图意概述**: 展示CORS预检请求的完整验证和响应流程
2. **关键字段**: Origin、Access-Control-Request-Method、Access-Control-Request-Headers
3. **边界条件**: 预检请求直接返回，不会到达应用层
4. **异常路径**: 任何验证失败都返回403
5. **性能假设**: 预检请求通常占总请求的10-20%（取决于max_age配置）
6. **设计理由**: 浏览器缓存预检结果，减少不必要的请求

### 5.2 实际请求的CORS处理

```mermaid
sequenceDiagram
    autonumber
    participant Browser as 浏览器
    participant CORS as CORSMiddleware
    participant App as 应用
    
    Browser->>CORS: GET /api/users
    Note over Browser: Origin: https://example.com
    
    CORS->>App: 转发请求
    App->>App: 处理请求
    App-->>CORS: Response
    
    CORS->>CORS: 检查Origin
    alt Origin允许
        CORS->>CORS: 添加CORS响应头
        Note over CORS: Access-Control-Allow-Origin<br/>Access-Control-Allow-Credentials<br/>Access-Control-Expose-Headers
        CORS-->>Browser: Response + CORS Headers
    else Origin不允许
        CORS-->>Browser: Response (无CORS头)
        Note over Browser: 浏览器会阻止访问响应
    end
```

---

## GZIP压缩流程

### 6.1 GZIP中间件处理流程

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用
    participant GZIP as GZIPMiddleware
    participant Config as 配置
    participant Compress as gzip.compress()
    participant Client as 客户端
    
    App->>GZIP: Response + body
    GZIP->>GZIP: 检查Accept-Encoding
    Note over GZIP: request.headers["Accept-Encoding"]
    
    alt 不支持gzip
        GZIP-->>Client: 原样返回
    else 支持gzip
        GZIP->>GZIP: 检查Content-Type
        
        alt 不可压缩类型（如image/jpeg）
            GZIP-->>Client: 原样返回
        else 可压缩类型
            GZIP->>GZIP: 检查响应体大小
            Note over GZIP: len(body)
            
            alt < minimum_size
                GZIP-->>Client: 原样返回
                Note over GZIP: 太小，不值得压缩
            else >= minimum_size
                GZIP->>Compress: gzip.compress(body, compresslevel)
                Compress-->>GZIP: 压缩后的body
                
                GZIP->>GZIP: 修改响应头
                Note over GZIP: Content-Encoding: gzip<br/>移除Content-Length<br/>添加Vary: Accept-Encoding
                
                GZIP-->>Client: 压缩后的Response
            end
        end
    end
```

**时序图说明**：
1. **图意概述**: 展示GZIP压缩的完整决策和执行流程
2. **关键字段**: Accept-Encoding决定是否压缩；minimum_size控制压缩阈值
3. **边界条件**: 小响应不压缩；不可压缩类型跳过；客户端不支持时跳过
4. **异常路径**: 压缩失败返回原始响应
5. **性能假设**: 压缩级别越高，CPU消耗越大；通常能减少60-80%传输量
6. **设计理由**: 平衡CPU消耗和带宽节省

### 6.2 压缩率对比

```mermaid
graph LR
    A[原始响应 100KB] --> B{GZIP压缩}
    B -->|compresslevel=1| C[~50KB 压缩率50%]
    B -->|compresslevel=5| D[~30KB 压缩率70%]
    B -->|compresslevel=9| E[~25KB 压缩率75%]
```

---

## 📊 时序图总结

### 核心流程对比

| 流程 | 执行时机 | 复杂度 | 频率 | 性能影响 |
|------|----------|--------|------|----------|
| 中间件栈构建 | 应用启动 | O(n) | 一次 | 无 |
| 中间件注册 | 配置阶段 | O(1) | 多次 | 无 |
| 请求处理链 | 每个请求 | O(n) | 高频 | 高 |
| AsyncExitStack | 每个请求 | O(d) | 高频 | 中 |
| CORS预检 | OPTIONS请求 | O(1) | 中频 | 低 |
| GZIP压缩 | 大响应 | O(m) | 高频 | 中 |

*n=中间件数量, d=yield依赖数量, m=响应体大小*

### 性能优化建议

1. **减少中间件数量**
   - ✅ 合并功能相似的中间件
   - ✅ 移除不必要的中间件
   - ⚠️ 中间件数量直接影响每个请求的处理时间

2. **CORS配置优化**
   - ✅ 增加max_age减少预检请求频率
   - ✅ 生产环境明确指定allow_origins
   - ⚠️ 避免使用正则表达式匹配（性能较差）

3. **GZIP配置优化**
   - ✅ 设置合理的minimum_size（推荐1000-2000字节）
   - ✅ 使用中等压缩级别（5-6）
   - ⚠️ 对已压缩内容（图片、视频）禁用压缩

4. **AsyncExitStack优化**
   - ✅ 清理代码应该快速执行
   - ✅ 避免在清理代码中执行IO操作
   - ⚠️ yield依赖清理异常应该被捕获

### 中间件顺序最佳实践

```python
# 推荐的中间件添加顺序
app = FastAPI()

# 1. 安全相关（最外层）
app.add_middleware(TrustedHostMiddleware, ...)
app.add_middleware(HTTPSRedirectMiddleware)

# 2. CORS（在压缩之前）
app.add_middleware(CORSMiddleware, ...)

# 3. 压缩（最后，压缩所有响应）
app.add_middleware(GZIPMiddleware, ...)

# 4. 自定义中间件
@app.middleware("http")
async def custom_middleware(request, call_next):
    response = await call_next(request)
    return response
```

---

## 📚 相关文档

- [FastAPI-04-中间件系统-概览](./FastAPI-04-中间件系统-概览.md) - 中间件系统架构
- [FastAPI-04-中间件系统-API](./FastAPI-04-中间件系统-API.md) - 中间件API详解
- [FastAPI-04-中间件系统-数据结构](./FastAPI-04-中间件系统-数据结构.md) - 中间件数据结构
- [FastAPI-03-依赖注入-时序图](./FastAPI-03-依赖注入-时序图.md) - yield依赖详细流程

---

*本文档生成于 2025年10月4日，基于 FastAPI 0.118.0*

