# FastAPI-03-依赖注入-时序图

> **文档版本**: v1.0  
> **FastAPI 版本**: 0.118.0  
> **创建日期**: 2025年10月4日

---

## 📋 目录

1. [时序图概览](#时序图概览)
2. [依赖树构建流程](#依赖树构建流程)
3. [依赖解析执行流程](#依赖解析执行流程)
4. [依赖缓存流程](#依赖缓存流程)
5. [yield依赖生命周期](#yield依赖生命周期)
6. [安全依赖验证流程](#安全依赖验证流程)
7. [完整请求处理流程](#完整请求处理流程)

---

## 时序图概览

### 核心流程清单

| # | 流程名称 | 执行时机 | 复杂度 | 频率 |
|---|---------|----------|--------|------|
| 1 | 依赖树构建 | 路由注册时 | ⭐⭐⭐ | 一次 |
| 2 | 依赖解析执行 | 每个请求 | ⭐⭐⭐⭐ | 高频 |
| 3 | 依赖缓存 | 每个请求 | ⭐⭐ | 高频 |
| 4 | yield依赖管理 | 每个请求 | ⭐⭐⭐ | 中频 |
| 5 | 安全依赖验证 | 有安全需求时 | ⭐⭐⭐ | 中频 |

---

## 依赖树构建流程

### 1.1 get_dependant()构建依赖树

```mermaid
sequenceDiagram
    autonumber
    participant Route as APIRoute
    participant GetDep as get_dependant()
    participant Inspect as inspect模块
    participant Dep as Dependant
    participant SubDep as get_sub_dependant()
    
    Route->>GetDep: get_dependant(path, call=endpoint)
    GetDep->>Inspect: inspect.signature(endpoint)
    Inspect-->>GetDep: 函数签名
    
    GetDep->>Dep: 创建Dependant()
    Note over Dep: 初始化空的参数列表
    
    loop 遍历函数参数
        GetDep->>GetDep: 分析参数类型和注解
        
        alt 是Depends依赖
            GetDep->>SubDep: get_parameterless_sub_dependant()
            SubDep->>SubDep: 递归调用get_dependant()
            SubDep-->>GetDep: 子Dependant
            GetDep->>Dep: dependencies.append(子Dependant)
        else 是Security依赖
            GetDep->>GetDep: 创建SecurityRequirement
            GetDep->>Dep: security_requirements.append()
        else 是Path参数
            GetDep->>GetDep: 创建ModelField
            GetDep->>Dep: path_params.append()
        else 是Query参数
            GetDep->>Dep: query_params.append()
        else 是Header参数
            GetDep->>Dep: header_params.append()
        else 是Cookie参数
            GetDep->>Dep: cookie_params.append()
        else 是Body参数
            GetDep->>Dep: body_params.append()
        else 是Request/WebSocket等
            GetDep->>Dep: 设置特殊参数名
            Note over Dep: request_param_name = "request"
        end
    end
    
    GetDep-->>Route: 返回Dependant树
```

**时序图说明**：
1. **图意概述**: 展示依赖树构建的完整过程，从函数签名分析到Dependant对象创建
2. **关键字段**: signature包含所有参数信息；Dependant各类params列表分类存储参数
3. **边界条件**: 参数可以没有类型注解（使用默认类型）；可以没有依赖
4. **异常路径**: 不支持的参数类型抛出FastAPIError
5. **性能假设**: 参数数量n，子依赖数量d，复杂度O(n+d)
6. **设计理由**: 在启动时构建依赖树，避免运行时重复解析

### 1.2 递归构建子依赖树

```mermaid
sequenceDiagram
    autonumber
    participant Root as 根Dependant
    participant Get1 as get_dependant(依赖1)
    participant Dep1 as 子Dependant1
    participant Get2 as get_dependant(依赖2)
    participant Dep2 as 子Dependant2
    
    Root->>Get1: 解析依赖1函数
    Get1->>Dep1: 创建Dependant
    Dep1->>Get2: 发现依赖1有子依赖
    Get2->>Dep2: 创建子Dependant2
    Dep2-->>Dep1: 返回
    Dep1-->>Root: 返回完整子树
    
    Note over Root: 最终形成多层依赖树
```

**递归示例**：
```python
# 依赖3（叶子节点）
async def get_config():
    return {"key": "value"}

# 依赖2（中间节点）
async def get_db(config=Depends(get_config)):
    return Database(config)

# 依赖1（根节点的子节点）
async def get_current_user(db=Depends(get_db)):
    return await db.get_user()

# 根节点
@app.get("/users/me")
async def read_user(user=Depends(get_current_user)):
    return user
```

**依赖树结构**：
```
read_user (根Dependant)
└── get_current_user (子Dependant)
    └── get_db (孙Dependant)
        └── get_config (曾孙Dependant)
```

---

## 依赖解析执行流程

### 2.1 solve_dependencies()完整流程

```mermaid
sequenceDiagram
    autonumber
    participant Handler as Route Handler
    participant Solve as solve_dependencies()
    participant Cache as 依赖缓存
    participant Extract as 参数提取
    participant Call as 依赖函数调用
    participant Validate as 参数验证
    participant Stack as AsyncExitStack
    
    Handler->>Solve: solve_dependencies(request, dependant)
    Solve->>Solve: 初始化values字典
    Note over Solve: values = {}
    
    loop 遍历所有子依赖
        Solve->>Cache: 检查缓存 use_cache=True?
        
        alt 缓存命中
            Cache-->>Solve: 返回缓存值
        else 缓存未命中
            Solve->>Solve: 递归解析子依赖的子依赖
            Note over Solve: 深度优先遍历
            
            Solve->>Extract: 提取依赖函数的参数
            Extract->>Extract: 从request中提取
            Note over Extract: query, header, cookie等
            Extract-->>Solve: 参数字典
            
            Solve->>Validate: 验证参数
            alt 验证失败
                Validate-->>Solve: 返回errors
            else 验证成功
                Validate-->>Solve: 验证后的值
                
                Solve->>Call: 调用依赖函数(**params)
                
                alt 是yield依赖
                    Call->>Stack: 注册清理回调
                    Stack-->>Solve: yield的值
                else 普通依赖
                    Call-->>Solve: 返回值
                end
                
                Solve->>Cache: 缓存结果
            end
        end
        
        Solve->>Solve: 将结果加入values
    end
    
    Solve->>Extract: 提取endpoint自身的参数
    Extract-->>Solve: path_params, query_params等
    
    Solve-->>Handler: (values, errors, background_tasks, sub_response)
```

**时序图说明**：
1. **图意概述**: 展示依赖解析的完整执行流程，包括缓存、参数提取、验证、调用
2. **关键字段**: values字典存储所有依赖的结果；errors列表收集验证错误
3. **边界条件**: 依赖可以嵌套任意层；参数验证可能失败
4. **异常路径**: 验证失败收集到errors，不中断后续依赖；依赖函数异常直接抛出
5. **性能假设**: 依赖数量d，参数数量p，复杂度O(d*p)，有缓存时O(d+p)
6. **设计理由**: 深度优先遍历确保子依赖先于父依赖执行

### 2.2 参数提取与验证

```mermaid
sequenceDiagram
    autonumber
    participant Solve as solve_dependencies()
    participant Request as Request对象
    participant Query as query_params
    participant Header as header_params
    participant Path as path_params
    participant Body as body_params
    participant Pydantic as Pydantic验证
    
    Solve->>Request: 获取原始数据
    
    Solve->>Path: 提取路径参数
    Path->>Request: scope["path_params"]
    Request-->>Path: {"user_id": "123"}
    Path->>Pydantic: 验证并转换
    Pydantic-->>Path: {"user_id": 123}
    
    Solve->>Query: 提取查询参数
    Query->>Request: request.query_params
    Request-->>Query: {"limit": "10"}
    Query->>Pydantic: 验证并转换
    Pydantic-->>Query: {"limit": 10}
    
    Solve->>Header: 提取请求头
    Header->>Request: request.headers
    Request-->>Header: {"authorization": "Bearer ..."}
    Header->>Pydantic: 验证
    Pydantic-->>Header: {"authorization": "Bearer ..."}
    
    Solve->>Body: 提取请求体
    Body->>Request: await request.json()
    Request-->>Body: {"name": "Item"}
    Body->>Pydantic: 验证Pydantic模型
    Pydantic-->>Body: Item(name="Item")
    
    Solve->>Solve: 合并所有参数
    Note over Solve: values = {path_params + query_params + ...}
```

---

## 依赖缓存流程

### 3.1 缓存检查与存储

```mermaid
sequenceDiagram
    autonumber
    participant Solve as solve_dependencies()
    participant Dep as Dependant
    participant Cache as dependency_cache
    participant Call as 依赖函数
    
    Solve->>Dep: 获取依赖信息
    Dep->>Dep: 检查use_cache
    
    alt use_cache=True
        Dep->>Dep: 生成cache_key
        Note over Dep: (call, security_scopes)
        
        Solve->>Cache: cache.get(cache_key)
        
        alt 缓存存在
            Cache-->>Solve: 返回缓存值
            Note over Solve: 跳过函数调用
        else 缓存不存在
            Solve->>Call: 调用依赖函数
            Call-->>Solve: 返回结果
            Solve->>Cache: cache[cache_key] = result
        end
    else use_cache=False
        Solve->>Call: 直接调用依赖函数
        Call-->>Solve: 返回结果
        Note over Solve: 不使用缓存
    end
    
    Solve-->>Solve: 继续处理下一个依赖
```

**时序图说明**：
1. **图意概述**: 展示依赖缓存的检查和存储机制
2. **关键字段**: cache_key由(call, security_scopes)组成，确保唯一性
3. **边界条件**: use_cache=False时跳过缓存；每个请求有独立的缓存字典
4. **性能假设**: 缓存查找O(1)；可以显著减少重复计算
5. **设计理由**: 缓存范围限定在单个请求内，避免跨请求污染

### 3.2 缓存键生成

```mermaid
graph LR
    A[Dependant] --> B[cache_key生成]
    B --> C[call: 函数对象]
    B --> D[security_scopes: 元组]
    C --> E["id(function)"]
    D --> F["tuple(sorted(scopes))"]
    E --> G["cache_key = (call, scopes)"]
    F --> G
```

---

## yield依赖生命周期

### 4.1 yield依赖的完整生命周期

```mermaid
sequenceDiagram
    autonumber
    participant Request as 请求到达
    participant Solve as solve_dependencies()
    participant Stack as AsyncExitStack
    participant Dep as yield依赖
    participant Endpoint as 端点函数
    participant Response as 响应返回
    participant Cleanup as 清理阶段
    
    Request->>Solve: 开始解析依赖
    Solve->>Stack: 创建AsyncExitStack
    
    Solve->>Dep: 调用yield依赖
    Dep->>Dep: 执行yield之前的代码
    Note over Dep: 初始化资源（如数据库连接）
    Dep-->>Solve: yield的值
    Solve->>Stack: enter_async_context(依赖)
    Note over Stack: 注册清理回调
    
    Solve->>Endpoint: 调用端点函数(yield的值)
    Endpoint->>Endpoint: 执行业务逻辑
    Endpoint-->>Solve: 返回响应数据
    
    Solve-->>Response: 构建Response对象
    Response->>Response: 发送响应给客户端
    
    Response->>Cleanup: 触发清理
    Cleanup->>Stack: __aexit__()
    Stack->>Dep: 执行yield之后的代码
    Note over Dep: 清理资源（如关闭数据库连接）
    Dep-->>Stack: 完成清理
    Stack-->>Cleanup: 完成
```

**时序图说明**：
1. **图意概述**: 展示yield依赖的完整生命周期，从资源初始化到清理
2. **关键字段**: AsyncExitStack管理所有yield依赖的清理；yield的值传递给endpoint
3. **边界条件**: yield之后的代码保证在响应发送后执行；异常也会触发清理
4. **异常路径**: endpoint异常→仍然执行清理代码；yield后代码异常→记录日志但不影响响应
5. **性能假设**: 清理代码应该快速执行，避免阻塞其他请求
6. **设计理由**: 使用AsyncExitStack确保资源正确释放，即使发生异常

### 4.2 多个yield依赖的执行顺序

```mermaid
sequenceDiagram
    autonumber
    participant Solve as solve_dependencies()
    participant Dep1 as yield依赖1
    participant Dep2 as yield依赖2
    participant Dep3 as yield依赖3
    participant Endpoint as 端点
    participant Stack as AsyncExitStack
    
    Note over Solve: 依赖注入阶段
    Solve->>Dep1: 调用依赖1
    Dep1->>Dep1: yield值1
    Solve->>Dep2: 调用依赖2
    Dep2->>Dep2: yield值2
    Solve->>Dep3: 调用依赖3
    Dep3->>Dep3: yield值3
    
    Solve->>Endpoint: 执行endpoint
    Endpoint-->>Solve: 返回响应
    
    Note over Solve: 清理阶段（逆序）
    Stack->>Dep3: 清理依赖3
    Stack->>Dep2: 清理依赖2
    Stack->>Dep1: 清理依赖1
```

**执行顺序说明**：
- **注入顺序**: 按依赖树深度优先顺序（先注入子依赖，后注入父依赖）
- **清理顺序**: 与注入顺序相反（后进先出，LIFO）
- **设计理由**: 确保依赖关系正确（如先创建连接池，后创建连接；先关闭连接，后关闭连接池）

### 4.3 yield依赖示例

```python
from fastapi import Depends
from typing import AsyncIterator

# yield依赖
async def get_db() -> AsyncIterator[Database]:
    # yield之前：初始化资源
    db = Database()
    await db.connect()
    
    try:
        # yield：提供资源
        yield db
    finally:
        # yield之后：清理资源
        await db.close()

# 使用yield依赖
@app.get("/users")
async def list_users(db: Database = Depends(get_db)):
    return await db.query("SELECT * FROM users")
    # 函数返回后，get_db的finally块会自动执行
```

---

## 安全依赖验证流程

### 5.1 Security()依赖验证

```mermaid
sequenceDiagram
    autonumber
    participant Solve as solve_dependencies()
    participant SecReq as SecurityRequirement
    participant Scheme as OAuth2PasswordBearer
    participant Request as Request
    participant Token as Token验证
    participant Scopes as Scopes检查
    
    Solve->>SecReq: 获取安全需求
    SecReq->>Scheme: 获取security_scheme
    
    Scheme->>Request: 提取凭证
    Note over Scheme: Authorization: Bearer <token>
    Request-->>Scheme: token字符串
    
    Scheme->>Token: 验证token格式
    alt token无效
        Token-->>Scheme: 抛出HTTPException(401)
    else token有效
        Token-->>Scheme: token字符串
    end
    
    Scheme->>Scopes: 检查作用域
    alt 需要特定scopes
        Scopes->>Scopes: 比较required vs actual
        alt scopes不足
            Scopes-->>Scheme: 抛出HTTPException(403)
        else scopes满足
            Scopes-->>Scheme: 通过
        end
    else 不需要scopes
        Scopes-->>Scheme: 通过
    end
    
    Scheme-->>Solve: 返回token（或用户信息）
    Solve->>Solve: 将token加入values
```

**时序图说明**：
1. **图意概述**: 展示安全依赖的验证流程，包括凭证提取和权限检查
2. **关键字段**: token从Authorization header提取；scopes用于权限检查
3. **边界条件**: token缺失返回401；scopes不足返回403
4. **异常路径**: 验证失败抛出HTTPException，中断请求处理
5. **性能假设**: token验证可能涉及数据库查询或外部API调用
6. **设计理由**: 安全验证优先执行，失败则快速返回错误

### 5.2 OAuth2 scopes检查

```mermaid
graph TD
    A[请求到达] --> B{提取token}
    B -->|失败| C[401 Unauthorized]
    B -->|成功| D{解析token scopes}
    D --> E{检查required scopes}
    E -->|scopes不足| F[403 Forbidden]
    E -->|scopes满足| G[继续处理请求]
    G --> H[调用endpoint]
```

---

## 完整请求处理流程

### 6.1 从请求到响应的依赖注入全流程

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant App as FastAPI
    participant Route as APIRoute
    participant Solve as solve_dependencies()
    participant Cache as 依赖缓存
    participant Dep as 依赖函数
    participant Stack as AsyncExitStack
    participant Endpoint as 端点函数
    participant Valid as 响应验证
    participant Response as Response
    
    Client->>App: HTTP Request
    App->>Route: 匹配路由
    Route->>Route: 创建AsyncExitStack
    Route->>Solve: solve_dependencies(dependant)
    
    loop 遍历依赖树（深度优先）
        Solve->>Cache: 检查缓存
        alt 缓存未命中
            Solve->>Dep: 递归解析子依赖
            Dep->>Dep: 提取参数并验证
            
            alt 是yield依赖
                Dep->>Stack: 注册清理回调
                Dep-->>Solve: yield的值
            else 普通依赖
                Dep-->>Solve: 返回值
            end
            
            Solve->>Cache: 缓存结果
        end
    end
    
    Solve->>Endpoint: 调用endpoint(**values)
    Endpoint->>Endpoint: 执行业务逻辑
    Endpoint-->>Solve: 返回结果
    
    Solve->>Valid: 验证响应模型
    Valid-->>Solve: 验证后的数据
    
    Solve->>Response: 创建Response
    Response-->>Client: 发送HTTP Response
    
    Response->>Stack: 触发清理
    loop 逆序清理yield依赖
        Stack->>Dep: 执行finally块
        Dep->>Dep: 释放资源
    end
```

**完整流程说明**：
1. **图意概述**: 展示包含依赖注入的完整请求处理流程
2. **关键阶段**: 路由匹配→依赖解析→endpoint执行→响应验证→资源清理
3. **边界条件**: 任何阶段失败都会跳过后续步骤，直接返回错误响应
4. **异常路径**: 验证失败→422；安全验证失败→401/403；业务逻辑异常→500
5. **性能假设**: 依赖缓存可以显著提升性能；yield依赖清理应该快速
6. **设计理由**: 分阶段处理，清晰的职责划分；使用AsyncExitStack确保资源正确释放

---

## 📊 时序图总结

### 核心流程对比

| 流程 | 执行时机 | 复杂度 | 频率 | 性能影响 |
|------|----------|--------|------|----------|
| 依赖树构建 | 启动时 | O(n+d) | 一次 | 无 |
| 依赖解析 | 每个请求 | O(d*p) | 高频 | 高 |
| 缓存检查 | 每个依赖 | O(1) | 高频 | 低 |
| yield清理 | 请求结束 | O(y) | 中频 | 低 |
| 安全验证 | 有安全需求时 | O(1) | 中频 | 中 |

*n=参数数量, d=依赖数量, p=参数数量, y=yield依赖数量*

### 性能优化建议

1. **依赖缓存**
   - ✅ 默认启用依赖缓存（use_cache=True）
   - ✅ 仅对无副作用的依赖启用缓存
   - ⚠️ 有状态依赖禁用缓存（use_cache=False）

2. **依赖层级**
   - ✅ 减少依赖嵌套层级
   - ✅ 将公共依赖提取到路由器级别
   - ⚠️ 避免循环依赖

3. **参数验证**
   - ✅ 使用Pydantic的验证缓存
   - ✅ 对简单类型使用原生Python类型
   - ⚠️ 复杂模型考虑使用orm_mode

4. **yield依赖**
   - ✅ 清理代码应该快速执行
   - ✅ 避免在清理代码中执行IO操作
   - ⚠️ 清理异常应该被捕获和记录

---

## 📚 相关文档

- [FastAPI-03-依赖注入-概览](./FastAPI-03-依赖注入-概览.md) - 依赖注入机制概述
- [FastAPI-03-依赖注入-API](./FastAPI-03-依赖注入-API.md) - Depends()等API详解
- [FastAPI-03-依赖注入-数据结构](./FastAPI-03-依赖注入-数据结构.md) - Dependant详解
- [FastAPI-02-路由系统-时序图](./FastAPI-02-路由系统-时序图.md) - 路由处理流程

---

*本文档生成于 2025年10月4日，基于 FastAPI 0.118.0*

