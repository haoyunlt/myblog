# FastAPI 源码剖析 - 01 应用层 - API

## 模块对外 API 总览

`applications.py` 模块对外提供的核心 API 是 `FastAPI` 类及其方法。主要包括：

### 应用创建与配置
- `FastAPI.__init__()` - 创建 FastAPI 应用实例
- `FastAPI.setup()` - 设置文档路由

### 路由注册（HTTP 方法）
- `FastAPI.get()` - 注册 GET 路由
- `FastAPI.post()` - 注册 POST 路由
- `FastAPI.put()` - 注册 PUT 路由
- `FastAPI.delete()` - 注册 DELETE 路由
- `FastAPI.patch()` - 注册 PATCH 路由
- `FastAPI.options()` - 注册 OPTIONS 路由
- `FastAPI.head()` - 注册 HEAD 路由
- `FastAPI.trace()` - 注册 TRACE 路由

### 路由注册（通用）
- `FastAPI.api_route()` - 通用路由注册装饰器
- `FastAPI.add_api_route()` - 添加API路由（方法调用方式）
- `FastAPI.add_api_websocket_route()` - 添加 WebSocket 路由
- `FastAPI.websocket()` - 注册 WebSocket 路由装饰器
- `FastAPI.include_router()` - 包含 APIRouter

### 中间件管理
- `FastAPI.add_middleware()` - 添加中间件
- `FastAPI.middleware()` - 中间件装饰器
- `FastAPI.build_middleware_stack()` - 构建中间件栈

### 异常处理
- `FastAPI.exception_handler()` - 异常处理装饰器
- `FastAPI.add_exception_handler()` - 添加异常处理器

### OpenAPI 文档
- `FastAPI.openapi()` - 生成 OpenAPI schema
- `FastAPI.setup()` - 设置文档路由

### 生命周期管理
- `FastAPI.on_event()` - 生命周期事件装饰器（已废弃）
- `lifespan` 参数 - 生命周期上下文管理器（推荐）

---

## API 详细规格

## 1. FastAPI.__init__() - 应用初始化

### 基本信息
- **名称**：`FastAPI.__init__()`
- **类型**：构造方法
- **用途**：创建 FastAPI 应用实例并进行初始化配置

### 请求结构体（初始化参数）

```python
def __init__(
    self,
    *,
    debug: bool = False,
    routes: Optional[List[BaseRoute]] = None,
    title: str = "FastAPI",
    summary: Optional[str] = None,
    description: str = "",
    version: str = "0.1.0",
    openapi_url: Optional[str] = "/openapi.json",
    openapi_tags: Optional[List[Dict[str, Any]]] = None,
    servers: Optional[List[Dict[str, Union[str, Any]]]] = None,
    dependencies: Optional[Sequence[Depends]] = None,
    default_response_class: Type[Response] = Default(JSONResponse),
    docs_url: Optional[str] = "/docs",
    redoc_url: Optional[str] = "/redoc",
    swagger_ui_oauth2_redirect_url: Optional[str] = "/docs/oauth2-redirect",
    swagger_ui_init_oauth: Optional[Dict[str, Any]] = None,
    middleware: Optional[Sequence[Middleware]] = None,
    exception_handlers: Optional[Dict[Union[int, Type[Exception]], Callable]] = None,
    on_startup: Optional[Sequence[Callable[[], Any]]] = None,
    on_shutdown: Optional[Sequence[Callable[[], Any]]] = None,
    lifespan: Optional[Lifespan[AppType]] = None,
    terms_of_service: Optional[str] = None,
    contact: Optional[Dict[str, Union[str, Any]]] = None,
    license_info: Optional[Dict[str, Union[str, Any]]] = None,
    openapi_prefix: str = "",
    root_path: str = "",
    root_path_in_servers: bool = True,
    responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
    callbacks: Optional[List[BaseRoute]] = None,
    webhooks: Optional[routing.APIRouter] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    swagger_ui_parameters: Optional[Dict[str, Any]] = None,
    generate_unique_id_function: Callable[[routing.APIRoute], str] = Default(generate_unique_id),
    separate_input_output_schemas: bool = True,
) -> None:
```

### 参数详细说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| debug | bool | 否 | False | 是否启用调试模式，启用后会返回详细的错误堆栈 |
| routes | List[BaseRoute] | 否 | None | 预定义的路由列表（不推荐使用，应使用装饰器） |
| title | str | 否 | "FastAPI" | API 标题，显示在 OpenAPI 文档中 |
| summary | str | 否 | None | API 简短摘要 |
| description | str | 否 | "" | API 详细描述，支持 Markdown |
| version | str | 否 | "0.1.0" | API 版本号 |
| openapi_url | str | 否 | "/openapi.json" | OpenAPI JSON 文档的 URL，设为 None 可禁用 |
| openapi_tags | List[Dict] | 否 | None | OpenAPI 标签元数据 |
| servers | List[Dict] | 否 | None | API 服务器列表 |
| dependencies | Sequence[Depends] | 否 | None | 全局依赖项，应用于所有路由 |
| default_response_class | Type[Response] | 否 | JSONResponse | 默认响应类 |
| docs_url | str | 否 | "/docs" | Swagger UI 文档 URL，设为 None 可禁用 |
| redoc_url | str | 否 | "/redoc" | ReDoc 文档 URL，设为 None 可禁用 |
| swagger_ui_oauth2_redirect_url | str | 否 | "/docs/oauth2-redirect" | OAuth2 重定向 URL |
| swagger_ui_init_oauth | Dict | 否 | None | Swagger UI OAuth2 初始化参数 |
| middleware | Sequence[Middleware] | 否 | None | 预定义的中间件列表 |
| exception_handlers | Dict | 否 | None | 异常处理器字典 |
| on_startup | Sequence[Callable] | 否 | None | 启动事件处理器列表（已废弃） |
| on_shutdown | Sequence[Callable] | 否 | None | 关闭事件处理器列表（已废弃） |
| lifespan | Lifespan | 否 | None | 生命周期上下文管理器（推荐） |
| terms_of_service | str | 否 | None | 服务条款 URL |
| contact | Dict | 否 | None | 联系信息（name, url, email） |
| license_info | Dict | 否 | None | 许可证信息（name, url） |
| root_path | str | 否 | "" | 应用的根路径前缀 |
| root_path_in_servers | bool | 否 | True | 是否在 servers 列表中包含 root_path |
| responses | Dict | 否 | None | 全局响应定义 |
| callbacks | List[BaseRoute] | 否 | None | API 回调定义 |
| webhooks | APIRouter | 否 | None | Webhook 路由 |
| deprecated | bool | 否 | None | 是否标记整个 API 为废弃 |
| include_in_schema | bool | 否 | True | 是否包含在 OpenAPI schema 中 |
| swagger_ui_parameters | Dict | 否 | None | 自定义 Swagger UI 参数 |
| generate_unique_id_function | Callable | 否 | generate_unique_id | 生成唯一 operation ID 的函数 |
| separate_input_output_schemas | bool | 否 | True | 是否分离输入输出 schema（OpenAPI 3.1+） |

### 核心代码与逻辑

```python
def __init__(self, ...):
    # 初始化父类 Starlette
    super().__init__(
        debug=debug,
        routes=routes or [],
        middleware=middleware,
        exception_handlers=exception_handlers,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        lifespan=lifespan,
    )
    
    # 存储 OpenAPI 相关配置
    self.title = title
    self.summary = summary
    self.description = description
    self.version = version
    self.openapi_version = "3.1.0"
    self.terms_of_service = terms_of_service
    self.contact = contact
    self.license_info = license_info
    
    # 文档 URL 配置
    self.openapi_url = openapi_url
    self.docs_url = docs_url
    self.redoc_url = redoc_url
    self.swagger_ui_oauth2_redirect_url = swagger_ui_oauth2_redirect_url
    self.swagger_ui_init_oauth = swagger_ui_init_oauth
    self.swagger_ui_parameters = swagger_ui_parameters
    
    # 创建内部 APIRouter
    self.router: routing.APIRouter = routing.APIRouter(
        routes=routes or [],
        redirect_slashes=redirect_slashes,
        dependency_overrides_provider=self,
        on_startup=on_startup or [],
        on_shutdown=on_shutdown or [],
        lifespan=lifespan,
        default_response_class=default_response_class,
        dependencies=dependencies,
        callbacks=callbacks,
        deprecated=deprecated,
        include_in_schema=include_in_schema,
        responses=responses,
        generate_unique_id_function=generate_unique_id_function,
    )
    
    # Webhooks 路由
    self.webhooks = webhooks or routing.APIRouter()
    
    # 初始化 OpenAPI schema（延迟生成）
    self.openapi_schema: Optional[Dict[str, Any]] = None
    
    # 注册默认异常处理器
    self.add_exception_handler(HTTPException, http_exception_handler)
    self.add_exception_handler(
        RequestValidationError, request_validation_exception_handler
    )
    self.add_exception_handler(
        WebSocketRequestValidationError,
        websocket_request_validation_exception_handler,
    )
    
    # 设置文档路由
    self.setup()
```

**逻辑说明**：

1. **继承初始化**：调用父类 `Starlette.__init__()`，传递基础配置（debug、routes、middleware 等）

2. **存储 OpenAPI 配置**：保存所有与 OpenAPI 文档相关的配置参数，这些参数在生成文档时使用

3. **创建内部路由器**：创建 `APIRouter` 实例作为 `self.router`，所有路由注册操作委托给这个路由器

4. **初始化 Webhooks**：创建独立的 Webhooks 路由器，用于定义 API 回调

5. **延迟 OpenAPI 生成**：`self.openapi_schema` 初始化为 `None`，在首次访问文档时才生成

6. **注册默认异常处理器**：
   - `HTTPException` → 返回 JSON 格式的错误响应
   - `RequestValidationError` → 返回 422 验证错误，包含详细错误信息
   - `WebSocketRequestValidationError` → WebSocket 验证错误处理

7. **设置文档路由**：调用 `self.setup()` 注册文档相关的路由（`/docs`、`/redoc`、`/openapi.json`）

### 使用示例

```python
from fastapi import FastAPI

# 基础使用
app = FastAPI()

# 完整配置
app = FastAPI(
    title="我的 API",
    description="这是一个示例 API",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    contact={
        "name": "API Support",
        "url": "https://example.com/support",
        "email": "support@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# 禁用文档
app_no_docs = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# 使用生命周期管理器
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print("Application startup")
    yield
    # 关闭时执行
    print("Application shutdown")

app_with_lifespan = FastAPI(lifespan=lifespan)
```

---

## 2. FastAPI.get() / post() / put() / delete() 等 - HTTP 路由装饰器

### 基本信息
- **名称**：`FastAPI.get()` / `post()` / `put()` / `delete()` / `patch()` / `options()` / `head()` / `trace()`
- **类型**：装饰器方法
- **用途**：注册特定 HTTP 方法的路由处理函数

### 请求结构体（参数）

```python
def get(
    self,
    path: str,
    *,
    response_model: Any = Default(None),
    status_code: Optional[int] = None,
    tags: Optional[List[Union[str, Enum]]] = None,
    dependencies: Optional[Sequence[Depends]] = None,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    response_description: str = "Successful Response",
    responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
    deprecated: Optional[bool] = None,
    operation_id: Optional[str] = None,
    response_model_include: Optional[IncEx] = None,
    response_model_exclude: Optional[IncEx] = None,
    response_model_by_alias: bool = True,
    response_model_exclude_unset: bool = False,
    response_model_exclude_defaults: bool = False,
    response_model_exclude_none: bool = False,
    include_in_schema: bool = True,
    response_class: Type[Response] = Default(JSONResponse),
    name: Optional[str] = None,
    openapi_extra: Optional[Dict[str, Any]] = None,
    generate_unique_id_function: Callable[[routing.APIRoute], str] = Default(generate_unique_id),
) -> Callable[[DecoratedCallable], DecoratedCallable]:
```

### 参数详细说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| path | str | 是 | - | URL 路径，支持路径参数（如 `/items/{item_id}`） |
| response_model | Any | 否 | None | 响应数据模型，用于序列化和文档生成 |
| status_code | int | 否 | None | 默认响应状态码（GET: 200, POST: 201） |
| tags | List[str/Enum] | 否 | None | OpenAPI 标签，用于文档分组 |
| dependencies | Sequence[Depends] | 否 | None | 路由级依赖项 |
| summary | str | 否 | None | 路由摘要，显示在文档中 |
| description | str | 否 | None | 路由详细描述，支持 Markdown |
| response_description | str | 否 | "Successful Response" | 成功响应的描述 |
| responses | Dict | 否 | None | 额外的响应定义（不同状态码） |
| deprecated | bool | 否 | None | 是否标记为废弃 |
| operation_id | str | 否 | None | OpenAPI operation ID，默认自动生成 |
| response_model_include | IncEx | 否 | None | 响应模型包含的字段 |
| response_model_exclude | IncEx | 否 | None | 响应模型排除的字段 |
| response_model_by_alias | bool | 否 | True | 是否使用字段别名 |
| response_model_exclude_unset | bool | 否 | False | 是否排除未设置的字段 |
| response_model_exclude_defaults | bool | 否 | False | 是否排除默认值字段 |
| response_model_exclude_none | bool | 否 | False | 是否排除 None 值字段 |
| include_in_schema | bool | 否 | True | 是否包含在 OpenAPI schema 中 |
| response_class | Type[Response] | 否 | JSONResponse | 响应类（JSONResponse、HTMLResponse 等） |
| name | str | 否 | None | 路由名称，用于 URL 反向解析 |
| openapi_extra | Dict | 否 | None | 额外的 OpenAPI 属性 |
| generate_unique_id_function | Callable | 否 | generate_unique_id | 生成唯一 ID 的函数 |

### 核心代码与逻辑

```python
def get(self, path: str, **kwargs) -> Callable:
    # 调用通用的 api_route 方法，指定 methods=["GET"]
    return self.api_route(path, methods=["GET"], **kwargs)

def post(self, path: str, **kwargs) -> Callable:
    # 调用通用的 api_route 方法，指定 methods=["POST"]
    return self.api_route(path, methods=["POST"], **kwargs)

def api_route(self, path: str, *, methods: Optional[List[str]] = None, **kwargs) -> Callable:
    def decorator(func: DecoratedCallable) -> DecoratedCallable:
        # 委托给内部路由器处理
        self.router.add_api_route(
            path,
            func,
            methods=methods,
            **kwargs
        )
        # 返回原函数，保持装饰器透明性
        return func
    
    return decorator
```

**逻辑说明**：

1. **HTTP 方法路由**：`get()`、`post()` 等方法都是 `api_route()` 的便捷封装，预设了 `methods` 参数

2. **装饰器模式**：返回一个装饰器函数，该装饰器：
   - 接收被装饰的函数
   - 调用 `self.router.add_api_route()` 注册路由
   - 返回原函数（保持函数不变，仅注册路由）

3. **委托给路由器**：实际的路由注册逻辑在 `APIRouter.add_api_route()` 中实现

4. **参数传递**：所有路由配置参数（response_model、dependencies 等）通过 `**kwargs` 传递给路由器

### 使用示例

```python
from fastapi import FastAPI, Path, Query
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 基础 GET 路由
@app.get("/")
async def root():
    return {"message": "Hello World"}

# 带路径参数的 GET 路由
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

# 带查询参数的 GET 路由
@app.get("/items/")
async def list_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# POST 路由带请求体
class Item(BaseModel):
    name: str
    price: float

@app.post("/items/", response_model=Item, status_code=201)
async def create_item(item: Item):
    return item

# 带响应模型和文档的 GET 路由
@app.get(
    "/items/{item_id}",
    response_model=Item,
    tags=["items"],
    summary="Get an item",
    description="Retrieve an item by its ID",
    response_description="The requested item",
)
async def read_item_detailed(
    item_id: int = Path(..., description="The ID of the item to get")
):
    return {"name": "Item", "price": 10.5}

# PUT 路由
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}

# DELETE 路由
@app.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: int):
    # 返回 204 No Content
    return None

# 排除未设置字段的响应
@app.get("/items/", response_model=List[Item], response_model_exclude_unset=True)
async def list_items_optimized():
    return [{"name": "Item 1"}, {"name": "Item 2", "price": 20.0}]
```

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者代码
    participant App as FastAPI 实例
    participant Router as APIRouter
    participant Route as APIRoute
    
    Dev->>App: @app.get("/items/{id}")
    Note over Dev: def read_item(id: int)
    
    App->>App: 调用 api_route()
    Note over App: methods=["GET"]
    
    App->>Router: router.add_api_route()
    Note over Router: 传递路径、函数、参数
    
    Router->>Route: 创建 APIRoute 实例
    Note over Route: 解析路径参数<br/>提取依赖树<br/>生成 OpenAPI schema
    
    Route-->>Router: 返回 APIRoute 对象
    Router->>Router: 添加到路由列表
    Router-->>App: 注册完成
    App-->>Dev: 返回原函数
    
    Note over Dev,Route: 后续请求处理
    
    participant Client as 客户端
    Client->>App: GET /items/42
    App->>Router: 路由匹配
    Router->>Route: 找到匹配的路由
    Route->>Route: 执行依赖注入
    Route->>Route: 调用处理函数
    Note over Route: read_item(id=42)
    Route-->>Client: 返回响应
```

### 时序图说明

**注册阶段（步骤 1-9）**：
1. 开发者使用 `@app.get()` 装饰器
2. FastAPI 调用 `api_route()` 方法，设置 `methods=["GET"]`
3. `api_route()` 返回装饰器函数
4. 装饰器函数调用 `router.add_api_route()`
5. APIRouter 创建 `APIRoute` 实例：
   - 编译路径正则表达式
   - 解析依赖树
   - 生成 OpenAPI schema
6. APIRoute 对象添加到路由列表
7. 返回原函数给开发者

**请求处理阶段（步骤 10-15）**：
1. 客户端发送 GET 请求
2. FastAPI 接收请求并委托给 Router
3. Router 遍历路由列表，使用正则匹配路径
4. 找到匹配的 APIRoute 对象
5. APIRoute 执行依赖注入、参数提取、数据验证
6. 调用处理函数并返回响应

---

## 3. FastAPI.include_router() - 包含路由模块

### 基本信息
- **名称**：`FastAPI.include_router()`
- **类型**：方法
- **用途**：将 `APIRouter` 实例包含到 FastAPI 应用中，用于模块化路由管理

### 请求结构体（参数）

```python
def include_router(
    self,
    router: routing.APIRouter,
    *,
    prefix: str = "",
    tags: Optional[List[Union[str, Enum]]] = None,
    dependencies: Optional[Sequence[Depends]] = None,
    responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
    deprecated: Optional[bool] = None,
    include_in_schema: bool = True,
    default_response_class: Type[Response] = Default(JSONResponse),
    callbacks: Optional[List[BaseRoute]] = None,
    generate_unique_id_function: Callable[[routing.APIRoute], str] = Default(generate_unique_id),
) -> None:
```

### 参数详细说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| router | APIRouter | 是 | - | 要包含的 APIRouter 实例 |
| prefix | str | 否 | "" | URL 路径前缀（如 `/api/v1`） |
| tags | List[str/Enum] | 否 | None | 为该路由器的所有路由添加标签 |
| dependencies | Sequence[Depends] | 否 | None | 为该路由器的所有路由添加依赖项 |
| responses | Dict | 否 | None | 为该路由器的所有路由添加响应定义 |
| deprecated | bool | 否 | None | 标记该路由器的所有路由为废弃 |
| include_in_schema | bool | 否 | True | 是否包含在 OpenAPI schema 中 |
| default_response_class | Type[Response] | 否 | JSONResponse | 默认响应类 |
| callbacks | List[BaseRoute] | 否 | None | API 回调定义 |
| generate_unique_id_function | Callable | 否 | generate_unique_id | 生成唯一 ID 的函数 |

### 核心代码与逻辑

```python
def include_router(
    self,
    router: routing.APIRouter,
    *,
    prefix: str = "",
    tags: Optional[List[Union[str, Enum]]] = None,
    dependencies: Optional[Sequence[Depends]] = None,
    **kwargs
) -> None:
    # 直接委托给内部路由器
    self.router.include_router(
        router,
        prefix=prefix,
        tags=tags,
        dependencies=dependencies,
        **kwargs
    )
```

**逻辑说明**：

1. **委托模式**：FastAPI 的 `include_router()` 直接委托给 `self.router.include_router()`

2. **路径前缀**：`prefix` 参数会添加到路由器中所有路由的路径前面

3. **标签继承**：`tags` 参数会添加到路由器中所有路由的标签列表

4. **依赖项继承**：`dependencies` 参数会添加到路由器中所有路由的依赖项列表

5. **级联配置**：子路由器的配置会与主应用的配置合并

### 使用示例

```python
from fastapi import FastAPI, APIRouter, Depends

app = FastAPI()

# 用户模块路由
users_router = APIRouter()

@users_router.get("/")
async def list_users():
    return [{"id": 1, "name": "User 1"}]

@users_router.get("/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id, "name": f"User {user_id}"}

@users_router.post("/")
async def create_user(user: dict):
    return user

# 商品模块路由
items_router = APIRouter()

@items_router.get("/")
async def list_items():
    return [{"id": 1, "name": "Item 1"}]

@items_router.get("/{item_id}")
async def get_item(item_id: int):
    return {"id": item_id, "name": f"Item {item_id}"}

# 包含路由模块
app.include_router(
    users_router,
    prefix="/users",
    tags=["users"],
)

app.include_router(
    items_router,
    prefix="/items",
    tags=["items"],
)

# 现在可以访问:
# GET /users/       -> list_users()
# GET /users/1      -> get_user(1)
# POST /users/      -> create_user()
# GET /items/       -> list_items()
# GET /items/1      -> get_item(1)

# 带依赖项的路由包含
async def verify_token(token: str = Header(...)):
    if token != "secret":
        raise HTTPException(401, "Invalid token")
    return token

admin_router = APIRouter()

@admin_router.get("/dashboard")
async def admin_dashboard():
    return {"message": "Admin dashboard"}

app.include_router(
    admin_router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(verify_token)],  # 所有 admin 路由都需要 token
)
```

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 开发者
    participant App as FastAPI
    participant MainRouter as 主路由器
    participant SubRouter as 子路由器
    participant Routes as 路由列表
    
    Dev->>SubRouter: 创建 APIRouter()
    Dev->>SubRouter: @router.get("/items")
    Note over SubRouter: 注册子路由
    
    Dev->>App: app.include_router(router)
    Note over Dev: prefix="/api"
    
    App->>MainRouter: router.include_router()
    MainRouter->>SubRouter: 获取子路由列表
    SubRouter-->>MainRouter: 返回路由列表
    
    loop 遍历子路由
        MainRouter->>MainRouter: 添加前缀到路径
        Note over MainRouter: /api + /items = /api/items
        MainRouter->>MainRouter: 合并标签和依赖项
        MainRouter->>Routes: 添加路由到主列表
    end
    
    MainRouter-->>App: 完成包含
    App-->>Dev: 返回
    
    Note over Dev,Routes: 后续请求处理
    
    participant Client as 客户端
    Client->>App: GET /api/items
    App->>MainRouter: 路由匹配
    MainRouter->>Routes: 遍历查找
    Routes->>Routes: 匹配 /api/items
    Routes->>SubRouter: 调用原始处理函数
    SubRouter-->>Client: 返回响应
```

---

## 4. FastAPI.middleware() - 中间件装饰器

### 基本信息
- **名称**：`FastAPI.middleware()`
- **类型**：装饰器方法
- **用途**：注册 HTTP 中间件，用于处理所有请求和响应

### 请求结构体（参数）

```python
def middleware(
    self,
    middleware_type: str,
) -> Callable[[DecoratedCallable], DecoratedCallable]:
```

### 参数详细说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| middleware_type | str | 是 | - | 中间件类型，目前仅支持 "http" |

### 核心代码与逻辑

```python
def middleware(self, middleware_type: str) -> Callable:
    def decorator(func: DecoratedCallable) -> DecoratedCallable:
        # 使用 BaseHTTPMiddleware 包装函数
        self.add_middleware(BaseHTTPMiddleware, dispatch=func)
        return func
    
    return decorator
```

**逻辑说明**：

1. **装饰器模式**：返回装饰器函数，接收中间件处理函数

2. **BaseHTTPMiddleware 包装**：将函数包装为 `BaseHTTPMiddleware` 实例

3. **dispatch 参数**：中间件函数作为 `dispatch` 参数传递给 `BaseHTTPMiddleware`

4. **中间件签名**：中间件函数必须接收 `(request, call_next)` 参数

5. **洋葱模型**：中间件按注册顺序反向执行（后注册的先执行 before 部分）

### 使用示例

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

# 添加处理时间中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 添加请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"收到请求: {request.method} {request.url}")
    response = await call_next(request)
    print(f"返回响应: {response.status_code}")
    return response

# 添加认证中间件
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/admin"):
        token = request.headers.get("Authorization")
        if not token or token != "Bearer secret":
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"}
            )
    response = await call_next(request)
    return response

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/admin/dashboard")
async def admin_dashboard():
    return {"message": "Admin Dashboard"}
```

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant MW1 as 日志中间件
    participant MW2 as 认证中间件
    participant MW3 as 计时中间件
    participant Route as 路由处理
    
    Client->>MW1: 请求
    Note over MW1: 记录请求日志
    MW1->>MW2: call_next(request)
    
    Note over MW2: 检查认证
    alt 认证失败
        MW2-->>Client: 401 Unauthorized
    else 认证通过
        MW2->>MW3: call_next(request)
        Note over MW3: 记录开始时间
        MW3->>Route: call_next(request)
        
        Route->>Route: 执行业务逻辑
        Route-->>MW3: 返回响应
        
        Note over MW3: 计算耗时<br/>添加 X-Process-Time 头部
        MW3-->>MW2: 返回响应
        MW2-->>MW1: 返回响应
        
        Note over MW1: 记录响应日志
        MW1-->>Client: 返回响应
    end
```

### 中间件执行顺序说明

**注册顺序**：
```python
@app.middleware("http")  # 第1个注册
async def middleware_1(request, call_next):
    ...

@app.middleware("http")  # 第2个注册
async def middleware_2(request, call_next):
    ...

@app.middleware("http")  # 第3个注册
async def middleware_3(request, call_next):
    ...
```

**执行顺序（洋葱模型）**：
```
请求 → middleware_3 (before) → middleware_2 (before) → middleware_1 (before) 
     → 路由处理 
     → middleware_1 (after) → middleware_2 (after) → middleware_3 (after) → 响应
```

---

## 5. FastAPI.exception_handler() - 异常处理装饰器

### 基本信息
- **名称**：`FastAPI.exception_handler()`
- **类型**：装饰器方法
- **用途**：注册全局异常处理器，捕获特定异常并返回自定义响应

### 请求结构体（参数）

```python
def exception_handler(
    self,
    exc_class_or_status_code: Union[int, Type[Exception]],
) -> Callable[[DecoratedCallable], DecoratedCallable]:
```

### 参数详细说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| exc_class_or_status_code | Union[int, Type[Exception]] | 是 | - | 异常类或 HTTP 状态码 |

### 核心代码与逻辑

```python
def exception_handler(
    self,
    exc_class_or_status_code: Union[int, Type[Exception]],
) -> Callable:
    def decorator(func: DecoratedCallable) -> DecoratedCallable:
        # 注册异常处理器
        self.add_exception_handler(exc_class_or_status_code, func)
        return func
    
    return decorator
```

**逻辑说明**：

1. **装饰器模式**：返回装饰器函数，接收异常处理函数

2. **异常匹配**：
   - **异常类**：匹配该类及其子类的异常
   - **状态码**：匹配该状态码的 HTTPException

3. **处理器签名**：异常处理函数必须接收 `(request, exc)` 参数

4. **优先级**：精确匹配的处理器优先于基类处理器

### 使用示例

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# 自定义异常
class CustomException(Exception):
    def __init__(self, message: str):
        self.message = message

# 注册自定义异常处理器
@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=500,
        content={"detail": exc.message}
    )

# 注册特定状态码处理器
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Resource not found: {request.url.path}"}
    )

# 全局异常处理器（捕获所有未处理异常）
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# 使用异常
@app.get("/custom-error")
async def trigger_custom_error():
    raise CustomException("This is a custom error")

@app.get("/not-found")
async def trigger_not_found():
    raise HTTPException(status_code=404, detail="Item not found")

@app.get("/server-error")
async def trigger_server_error():
    raise ValueError("Unexpected error")  # 被全局处理器捕获
```

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant App as FastAPI
    participant Route as 路由处理
    participant ExcMiddleware as 异常中间件
    participant Handler as 异常处理器
    
    Client->>App: GET /custom-error
    App->>Route: 执行路由函数
    Route->>Route: raise CustomException()
    Route-->>ExcMiddleware: 抛出异常
    
    ExcMiddleware->>ExcMiddleware: 捕获异常
    ExcMiddleware->>ExcMiddleware: 查找匹配的处理器
    Note over ExcMiddleware: 匹配 CustomException
    
    ExcMiddleware->>Handler: 调用异常处理器
    Note over Handler: custom_exception_handler(request, exc)
    
    Handler->>Handler: 构造 JSONResponse
    Handler-->>ExcMiddleware: 返回响应
    ExcMiddleware-->>Client: 500 + JSON
```

---

## 6. FastAPI.openapi() - 生成 OpenAPI Schema

### 基本信息
- **名称**：`FastAPI.openapi()`
- **类型**：方法
- **用途**：生成 OpenAPI 3.1 规范的 JSON Schema

### 响应结构体

```python
def openapi(self) -> Dict[str, Any]:
```

**返回值**：OpenAPI 3.1 规范的字典

### 核心代码与逻辑

```python
def openapi(self) -> Dict[str, Any]:
    # 检查是否已缓存
    if not self.openapi_schema:
        # 调用 get_openapi 函数生成 schema
        self.openapi_schema = get_openapi(
            title=self.title,
            version=self.version,
            openapi_version=self.openapi_version,
            summary=self.summary,
            description=self.description,
            terms_of_service=self.terms_of_service,
            contact=self.contact,
            license_info=self.license_info,
            routes=self.routes,  # 所有路由
            webhooks=self.webhooks.routes,  # Webhooks 路由
            tags=self.openapi_tags,
            servers=self.servers,
            separate_input_output_schemas=self.separate_input_output_schemas,
        )
    return self.openapi_schema
```

**逻辑说明**：

1. **延迟生成**：首次调用时生成，之后返回缓存

2. **缓存策略**：生成后存储在 `self.openapi_schema` 中

3. **生成流程**：
   - 遍历所有路由
   - 提取路径参数、查询参数、请求体、响应模型
   - 生成 JSON Schema
   - 构造符合 OpenAPI 3.1 规范的字典

4. **可定制**：可以重写此方法自定义 OpenAPI 生成逻辑

### 使用示例

```python
from fastapi import FastAPI

app = FastAPI(
    title="我的 API",
    version="1.0.0",
    description="这是一个示例 API",
)

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# 获取 OpenAPI schema
openapi_schema = app.openapi()

# 打印 schema
import json
print(json.dumps(openapi_schema, indent=2))

# 输出示例:
# {
#   "openapi": "3.1.0",
#   "info": {
#     "title": "我的 API",
#     "version": "1.0.0",
#     "description": "这是一个示例 API"
#   },
#   "paths": {
#     "/items/{item_id}": {
#       "get": {
#         "summary": "Read Item",
#         "operationId": "read_item_items__item_id__get",
#         "parameters": [
#           {
#             "name": "item_id",
#             "in": "path",
#             "required": true,
#             "schema": {"type": "integer"}
#           },
#           {
#             "name": "q",
#             "in": "query",
#             "required": false,
#             "schema": {"type": "string"}
#           }
#         ],
#         "responses": { ... }
#       }
#     }
#   }
# }

# 自定义 OpenAPI 生成
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="自定义 API",
        version="2.0.0",
        routes=app.routes,
    )
    
    # 自定义修改
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

---

## 7. FastAPI.setup() - 设置文档路由

### 基本信息
- **名称**：`FastAPI.setup()`
- **类型**：方法
- **用途**：注册 OpenAPI 文档相关的路由（`/docs`、`/redoc`、`/openapi.json`）

### 核心代码与逻辑

```python
def setup(self) -> None:
    # 注册 OpenAPI JSON 路由
    if self.openapi_url:
        @self.get(self.openapi_url, include_in_schema=False)
        async def openapi() -> Dict[str, Any]:
            return self.openapi()
    
    # 注册 Swagger UI 路由
    if self.docs_url:
        @self.get(self.docs_url, include_in_schema=False)
        async def swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url=self.openapi_url,
                title=self.title + " - Swagger UI",
                oauth2_redirect_url=self.swagger_ui_oauth2_redirect_url,
                init_oauth=self.swagger_ui_init_oauth,
                swagger_ui_parameters=self.swagger_ui_parameters,
            )
    
    # 注册 OAuth2 重定向路由
    if self.swagger_ui_oauth2_redirect_url:
        @self.get(self.swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect():
            return get_swagger_ui_oauth2_redirect_html()
    
    # 注册 ReDoc 路由
    if self.redoc_url:
        @self.get(self.redoc_url, include_in_schema=False)
        async def redoc_html():
            return get_redoc_html(
                openapi_url=self.openapi_url,
                title=self.title + " - ReDoc",
            )
```

**逻辑说明**：

1. **条件注册**：只有当相应 URL 不为 `None` 时才注册路由

2. **exclude_from_schema**：文档路由不包含在 OpenAPI schema 中（避免递归）

3. **OpenAPI JSON**：返回 `self.openapi()` 生成的 schema

4. **Swagger UI**：返回 HTML 页面，嵌入 Swagger UI JavaScript

5. **ReDoc**：返回 HTML 页面，嵌入 ReDoc JavaScript

### 使用示例

```python
from fastapi import FastAPI

# 默认配置
app = FastAPI()
# 自动注册:
# GET /docs          -> Swagger UI
# GET /redoc         -> ReDoc
# GET /openapi.json  -> OpenAPI JSON

# 自定义文档 URL
app_custom = FastAPI(
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
# 注册:
# GET /api/docs          -> Swagger UI
# GET /api/redoc         -> ReDoc
# GET /api/openapi.json  -> OpenAPI JSON

# 禁用文档
app_no_docs = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
# 不注册任何文档路由

# 自定义 Swagger UI
app_custom_swagger = FastAPI(
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,  # 隐藏模型
        "docExpansion": "none",          # 默认折叠所有操作
        "filter": True,                   # 显示过滤框
    }
)
```

---

## 总结

FastAPI 应用层（`applications.py`）提供的 API 主要分为以下几类：

1. **应用初始化**：`FastAPI.__init__()` - 配置应用参数
2. **路由注册**：`get()` / `post()` / `put()` / `delete()` 等 - 注册 HTTP 路由
3. **路由模块化**：`include_router()` - 包含子路由器
4. **中间件管理**：`middleware()` / `add_middleware()` - 注册中间件
5. **异常处理**：`exception_handler()` - 注册异常处理器
6. **文档生成**：`openapi()` / `setup()` - 生成和提供 API 文档

这些 API 构成了 FastAPI 框架的核心接口，支持灵活、模块化、声明式的 Web API 开发。

