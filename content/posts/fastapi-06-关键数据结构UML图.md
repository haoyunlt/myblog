---
title: "FastAPI 源码剖析 - 关键数据结构 UML 图"
date: 2025-09-28T00:47:17+08:00
draft: false
tags: ['Python', 'Web框架', '源码分析', 'FastAPI', 'API', '架构设计']
categories: ['Python框架', 'FastAPI']
description: "FastAPI 源码剖析 - 关键数据结构 UML 图的深入技术分析文档"
keywords: ['Python', 'Web框架', '源码分析', 'FastAPI', 'API', '架构设计']
author: "技术分析师"
weight: 1
---

## 1. 整体数据结构关系图

```mermaid
classDiagram
    %% FastAPI 核心类
    class FastAPI {
        +title: str
        +version: str
        +openapi_url: str
        +docs_url: str
        +redoc_url: str
        +router: APIRouter
        +dependency_overrides: Dict
        +openapi(): Dict
        +get(path): Callable
        +post(path): Callable
        +include_router(router): None
    }
    
    %% 路由系统
    class APIRouter {
        +prefix: str
        +tags: List[str]
        +dependencies: List[Depends]
        +routes: List[BaseRoute]
        +add_api_route(): None
        +get(): Callable
        +post(): Callable
        +api_route(): Callable
    }
    
    class APIRoute {
        +path: str
        +endpoint: Callable
        +methods: Set[str]
        +dependant: Dependant
        +response_model: Type
        +status_code: int
        +get_route_handler(): Callable
        +matches(): Tuple[Match, Scope]
    }
    
    %% 依赖注入系统
    class Dependant {
        +call: Callable
        +name: str
        +path_params: List[ModelField]
        +query_params: List[ModelField]
        +header_params: List[ModelField]
        +cookie_params: List[ModelField]
        +body_params: List[ModelField]
        +dependencies: List[Dependant]
        +security_requirements: List[SecurityRequirement]
        +use_cache: bool
    }
    
    class Depends {
        +dependency: Callable
        +use_cache: bool
    }
    
    %% 安全系统
    class SecurityBase {
        <<abstract>>
        +type_: str
        +scheme_name: str
        +auto_error: bool
        +__call__(request): Any
        +get_openapi_security_scheme(): Dict
    }
    
    class SecurityRequirement {
        +security_scheme: SecurityBase
        +scopes: List[str]
    }
    
    class SecurityScopes {
        +scopes: List[str]
        +scope_str: str
    }
    
    %% 参数处理
    class Param {
        <<abstract>>
        +default: Any
        +annotation: Any
        +alias: str
        +description: str
    }
    
    class Query {
        +default: Any
        +max_length: int
        +min_length: int
        +regex: str
    }
    
    class Path {
        +default: Any
        +ge: float
        +le: float
        +gt: float
        +lt: float
    }
    
    class Header {
        +default: Any
        +convert_underscores: bool
    }
    
    class Body {
        +default: Any
        +embed: bool
        +media_type: str
    }
    
    %% 关系定义
    FastAPI --> APIRouter : contains
    APIRouter --> APIRoute : creates
    APIRoute --> Dependant : analyzes
    Dependant --> SecurityRequirement : has
    SecurityRequirement --> SecurityBase : uses
    Dependant --> Depends : contains
    
    Param <|-- Query
    Param <|-- Path  
    Param <|-- Header
    Param <|-- Body
```

## 2. FastAPI 应用类详细结构

```mermaid
classDiagram
    %% Starlette 基类
    class Starlette {
        +debug: bool
        +routes: List[BaseRoute]
        +middleware: List[Middleware]
        +exception_handlers: Dict
        +on_startup: List[Callable]
        +on_shutdown: List[Callable]
        +state: State
        +__call__(scope, receive, send): None
    }
    
    %% FastAPI 主类
    class FastAPI {
        %% 基本属性
        +title: str
        +summary: str
        +description: str
        +version: str
        +terms_of_service: str
        +contact: Dict
        +license_info: Dict
        
        %% OpenAPI 相关
        +openapi_url: str
        +openapi_tags: List[Dict]
        +openapi_version: str
        +openapi_schema: Dict
        +openapi_external_docs: Dict
        
        %% 文档相关
        +docs_url: str
        +redoc_url: str
        +swagger_ui_oauth2_redirect_url: str
        +swagger_ui_init_oauth: Dict
        +swagger_ui_parameters: Dict
        
        %% 核心组件
        +router: APIRouter
        +dependency_overrides: Dict
        +webhooks: APIRouter
        +user_middleware: List[Middleware]
        +middleware_stack: ASGIApp
        
        %% 配置
        +separate_input_output_schemas: bool
        +generate_unique_id_function: Callable
        +default_response_class: Type[Response]
        
        %% 方法
        +openapi(): Dict
        +setup(): None
        +add_api_route(): None
        +include_router(): None
        +add_middleware(): None
        +exception_handler(): Callable
        +on_event(): Callable
        +get(): Callable
        +post(): Callable
        +put(): Callable
        +delete(): Callable
        +patch(): Callable
        +head(): Callable
        +options(): Callable
        +trace(): Callable
        +websocket(): Callable
    }
    
    Starlette <|-- FastAPI
```

## 3. 路由系统类图

```mermaid
classDiagram
    %% Starlette 路由基类
    class BaseRoute {
        <<abstract>>
        +matches(scope): Tuple[Match, Scope]
        +__call__(scope, receive, send): None
    }
    
    class Route {
        +path: str
        +endpoint: Callable
        +methods: Set[str]
        +name: str
        +include_in_schema: bool
        +path_regex: Pattern
        +path_format: str
        +param_convertors: Dict
        +matches(scope): Tuple[Match, Scope]
        +url_path_for(name, **params): URLPath
    }
    
    class Mount {
        +path: str
        +app: ASGIApp
        +name: str
        +matches(scope): Tuple[Match, Scope]
    }
    
    class WebSocketRoute {
        +path: str
        +endpoint: Callable
        +name: str
        +matches(scope): Tuple[Match, Scope]
    }
    
    %% FastAPI 扩展的路由类
    class APIRoute {
        %% 基本属性
        +path: str
        +endpoint: Callable
        +methods: Set[str]
        +name: str
        +summary: str
        +description: str
        +tags: List[str]
        +deprecated: bool
        
        %% 响应相关
        +response_model: Type
        +response_class: Type[Response]
        +status_code: int
        +response_model_include: IncEx
        +response_model_exclude: IncEx
        +response_model_by_alias: bool
        +response_model_exclude_unset: bool
        +response_model_exclude_defaults: bool
        +response_model_exclude_none: bool
        +responses: Dict
        
        %% 依赖和安全
        +dependant: Dependant
        +body_field: ModelField
        +secure_cloned_response_field: ModelField
        +dependency_overrides_provider: Any
        +callbacks: List[BaseRoute]
        
        %% OpenAPI
        +operation_id: str
        +unique_id: str
        +include_in_schema: bool
        +openapi_extra: Dict
        +generate_unique_id_function: Callable
        
        %% 方法
        +get_route_handler(): Callable
        +matches(scope): Tuple[Match, Scope]
    }
    
    class APIWebSocketRoute {
        +path: str
        +endpoint: Callable
        +name: str
        +dependant: Dependant
        +dependencies: List[Depends]
        +dependency_overrides_provider: Any
        +matches(scope): Tuple[Match, Scope]
    }
    
    class APIRouter {
        %% 配置属性
        +prefix: str
        +tags: List[str]
        +dependencies: List[Depends]
        +default_response_class: Type[Response]
        +responses: Dict
        +callbacks: List[BaseRoute]
        +deprecated: bool
        +include_in_schema: bool
        +generate_unique_id_function: Callable
        
        %% 路由管理
        +routes: List[BaseRoute]
        +route_class: Type[APIRoute]
        +dependency_overrides_provider: Any
        
        %% 生命周期
        +on_startup: List[Callable]
        +on_shutdown: List[Callable]
        +lifespan: Lifespan
        
        %% 方法
        +add_api_route(): None
        +api_route(): Callable
        +get(): Callable
        +post(): Callable
        +put(): Callable
        +delete(): Callable
        +patch(): Callable
        +head(): Callable
        +options(): Callable
        +trace(): Callable
        +websocket(): Callable
        +include_router(): None
        +add_websocket_route(): None
        +on_event(): Callable
    }
    
    %% 继承关系
    BaseRoute <|-- Route
    BaseRoute <|-- Mount
    BaseRoute <|-- WebSocketRoute
    Route <|-- APIRoute
    WebSocketRoute <|-- APIWebSocketRoute
    
    %% 组合关系
    APIRouter --> APIRoute : creates
    APIRouter --> APIWebSocketRoute : creates
    APIRoute --> Dependant : contains
```

## 4. 依赖注入系统类图

```mermaid
classDiagram
    %% 核心依赖类
    class Dependant {
        %% 基本信息
        +call: Callable
        +name: str
        +path: str
        +use_cache: bool
        +cache_key: Tuple
        
        %% 参数分类
        +path_params: List[ModelField]
        +query_params: List[ModelField]
        +header_params: List[ModelField]
        +cookie_params: List[ModelField]
        +body_params: List[ModelField]
        
        %% 依赖关系
        +dependencies: List[Dependant]
        +security_requirements: List[SecurityRequirement]
        +security_scopes: List[str]
        
        %% 特殊参数
        +request_param_name: str
        +websocket_param_name: str
        +http_connection_param_name: str
        +response_param_name: str
        +background_tasks_param_name: str
        +security_scopes_param_name: str
        
        %% 方法
        +__post_init__(): None
    }
    
    class Depends {
        +dependency: Callable
        +use_cache: bool
        +__init__(dependency, use_cache): None
        +__repr__(): str
    }
    
    class SolvedDependency {
        +values: Dict[str, Any]
        +errors: List[ErrorWrapper]
        +background_tasks: BackgroundTasks
        +response: Response
        +dependency_cache: Dict
    }
    
    %% 安全相关
    class SecurityRequirement {
        +security_scheme: SecurityBase
        +scopes: List[str]
    }
    
    class SecurityScopes {
        +scopes: List[str]
        +scope_str: str
        +__str__(): str
        +__repr__(): str
    }
    
    %% Pydantic 模型字段
    class ModelField {
        <<external>>
        +name: str
        +type_: Type
        +field_info: FieldInfo
        +default: Any
        +required: bool
        +alias: str
        +validate(value, values, loc): Tuple
    }
    
    %% 参数基类
    class FieldInfo {
        <<external>>
        +default: Any
        +default_factory: Callable
        +annotation: Any
        +alias: str
        +title: str
        +description: str
    }
    
    %% 关系
    Dependant --> SecurityRequirement : has
    Dependant --> ModelField : contains
    SecurityRequirement --> SecurityBase : references
    Dependant --> Depends : processes
    ModelField --> FieldInfo : contains
```

## 5. 参数处理系统类图

```mermaid
classDiagram
    %% Pydantic FieldInfo 基类
    class FieldInfo {
        <<external>>
        +default: Any
        +default_factory: Callable
        +annotation: Any
        +alias: str
        +alias_priority: int
        +validation_alias: str
        +serialization_alias: str
        +title: str
        +description: str
        +examples: List[Any]
        +deprecated: bool
        +include_in_schema: bool
        +json_schema_extra: Dict
    }
    
    %% FastAPI 参数基类
    class Param {
        +in_: ParamTypes
        +default: Any
        +annotation: Any
        +alias: str
        +title: str
        +description: str
        +gt: float
        +ge: float
        +lt: float
        +le: float
        +min_length: int
        +max_length: int
        +pattern: str
        +examples: List[Any]
        +deprecated: bool
        +include_in_schema: bool
    }
    
    %% 具体参数类型
    class Query {
        +in_: ParamTypes.query
    }
    
    class Path {
        +in_: ParamTypes.path
    }
    
    class Header {
        +in_: ParamTypes.header
        +convert_underscores: bool
    }
    
    class Cookie {
        +in_: ParamTypes.cookie
    }
    
    class Body {
        +in_: ParamTypes.body
        +embed: bool
        +media_type: str
    }
    
    class Form {
        +in_: ParamTypes.form
        +media_type: str
    }
    
    class File {
        +in_: ParamTypes.file
        +media_type: str
    }
    
    %% 枚举类型
    class ParamTypes {
        <<enumeration>>
        +query
        +path
        +header
        +cookie
        +body
        +form
        +file
    }
    
    %% 继承关系
    FieldInfo <|-- Param
    Param <|-- Query
    Param <|-- Path
    Param <|-- Header
    Param <|-- Cookie
    Param <|-- Body
    Param <|-- Form
    Param <|-- File
    
    %% 关联关系
    Param --> ParamTypes : uses
```

## 6. 安全认证系统类图

```mermaid
classDiagram
    %% 安全基类
    class SecurityBase {
        <<abstract>>
        +type_: str
        +scheme_name: str
        +description: str
        +auto_error: bool
        +__init__(scheme_name, description, auto_error): None
        +__call__(request): Any
        +get_openapi_security_scheme(): Dict
    }
    
    %% HTTP 认证基类
    class HTTPBase {
        +scheme: str
        +bearer_format: str
        +get_openapi_security_scheme(): Dict
    }
    
    %% API Key 认证基类  
    class APIKeyBase {
        +name: str
        +get_openapi_security_scheme(): Dict
    }
    
    %% OAuth2 认证基类
    class OAuth2 {
        +flows: OAuthFlows
        +scheme_name: str
        +scopes: Dict[str, str]
        +auto_error: bool
        +get_openapi_security_scheme(): Dict
    }
    
    %% OpenID Connect
    class OpenIdConnect {
        +openIdConnectUrl: str
        +get_openapi_security_scheme(): Dict
    }
    
    %% HTTP 认证具体实现
    class HTTPBasic {
        +scheme: str = "Basic"
        +__call__(request): HTTPBasicCredentials
    }
    
    class HTTPBearer {
        +scheme: str = "Bearer"
        +bearer_format: str
        +__call__(request): HTTPAuthorizationCredentials
    }
    
    class HTTPDigest {
        +scheme: str = "Digest"
        +__call__(request): HTTPAuthorizationCredentials
    }
    
    %% API Key 具体实现
    class APIKeyHeader {
        +name: str
        +__call__(request): str
    }
    
    class APIKeyCookie {
        +name: str
        +__call__(request): str
    }
    
    class APIKeyQuery {
        +name: str
        +__call__(request): str
    }
    
    %% OAuth2 具体实现
    class OAuth2PasswordBearer {
        +tokenUrl: str
        +scopes: Dict[str, str]
        +__call__(request): str
    }
    
    class OAuth2AuthorizationCodeBearer {
        +authorizationUrl: str
        +tokenUrl: str
        +scopes: Dict[str, str]
        +__call__(request): str
    }
    
    %% 认证凭据类
    class HTTPBasicCredentials {
        +username: str
        +password: str
    }
    
    class HTTPAuthorizationCredentials {
        +scheme: str
        +credentials: str
        +__str__(): str
    }
    
    %% OAuth2 表单
    class OAuth2PasswordRequestForm {
        +grant_type: str
        +username: str
        +password: str
        +scope: str
        +client_id: str
        +client_secret: str
        +scopes: List[str]
    }
    
    %% 继承关系
    SecurityBase <|-- HTTPBase
    SecurityBase <|-- APIKeyBase
    SecurityBase <|-- OAuth2
    SecurityBase <|-- OpenIdConnect
    
    HTTPBase <|-- HTTPBasic
    HTTPBase <|-- HTTPBearer
    HTTPBase <|-- HTTPDigest
    
    APIKeyBase <|-- APIKeyHeader
    APIKeyBase <|-- APIKeyCookie
    APIKeyBase <|-- APIKeyQuery
    
    OAuth2 <|-- OAuth2PasswordBearer
    OAuth2 <|-- OAuth2AuthorizationCodeBearer
    
    %% 关联关系
    HTTPBasic ..> HTTPBasicCredentials : creates
    HTTPBearer ..> HTTPAuthorizationCredentials : creates
    HTTPDigest ..> HTTPAuthorizationCredentials : creates
```

## 7. 中间件系统类图

```mermaid
classDiagram
    %% Starlette 中间件基类
    class BaseHTTPMiddleware {
        <<abstract>>
        +app: ASGIApp
        +dispatch(request, call_next): Response
        +__call__(scope, receive, send): None
    }
    
    class Middleware {
        +cls: Type
        +args: Tuple
        +kwargs: Dict
    }
    
    %% FastAPI 中间件
    class CORSMiddleware {
        +app: ASGIApp
        +allow_origins: List[str]
        +allow_methods: List[str]
        +allow_headers: List[str]
        +allow_credentials: bool
        +allow_origin_regex: str
        +expose_headers: List[str]
        +max_age: int
        +__call__(scope, receive, send): None
    }
    
    class GZipMiddleware {
        +app: ASGIApp
        +minimum_size: int
        +compresslevel: int
        +__call__(scope, receive, send): None
    }
    
    class HTTPSRedirectMiddleware {
        +app: ASGIApp
        +__call__(scope, receive, send): None
    }
    
    class TrustedHostMiddleware {
        +app: ASGIApp
        +allowed_hosts: List[str]
        +www_redirect: bool
        +__call__(scope, receive, send): None
    }
    
    class WSGIMiddleware {
        +wsgi: WSGIApp
        +executor: ThreadPoolExecutor
        +__call__(scope, receive, send): None
    }
    
    %% 自定义中间件基类
    class RequestResponseMiddleware {
        <<abstract>>
        +dispatch(request, call_next): Response
    }
    
    %% 继承关系
    BaseHTTPMiddleware <|-- RequestResponseMiddleware
    BaseHTTPMiddleware <|-- CORSMiddleware
    BaseHTTPMiddleware <|-- GZipMiddleware
    BaseHTTPMiddleware <|-- HTTPSRedirectMiddleware
    BaseHTTPMiddleware <|-- TrustedHostMiddleware
    BaseHTTPMiddleware <|-- WSGIMiddleware
```

## 8. 请求响应处理类图

```mermaid
classDiagram
    %% Starlette 请求响应基类
    class Request {
        +method: str
        +url: URL
        +headers: Headers
        +query_params: QueryParams
        +path_params: Dict[str, Any]
        +cookies: Dict[str, str]
        +client: Address
        +scope: Scope
        +body(): bytes
        +json(): Any
        +form(): FormData
        +stream(): AsyncIterator[bytes]
    }
    
    class WebSocket {
        +url: URL
        +headers: Headers
        +query_params: QueryParams
        +path_params: Dict[str, Any]
        +client: Address
        +scope: Scope
        +accept(): None
        +close(): None
        +send_text(): None
        +send_bytes(): None
        +send_json(): None
        +receive_text(): str
        +receive_bytes(): bytes
        +receive_json(): Any
    }
    
    class Response {
        +status_code: int
        +headers: MutableHeaders
        +media_type: str
        +body: bytes
        +charset: str
        +background: BackgroundTask
        +render(content): bytes
        +init_headers(): None
    }
    
    %% FastAPI 扩展响应类
    class JSONResponse {
        +media_type: str = "application/json"
        +render(content): bytes
    }
    
    class HTMLResponse {
        +media_type: str = "text/html"
        +render(content): bytes
    }
    
    class PlainTextResponse {
        +media_type: str = "text/plain"
        +render(content): bytes
    }
    
    class RedirectResponse {
        +status_code: int = 307
    }
    
    class StreamingResponse {
        +media_type: str
        +background: BackgroundTask
        +body_iterator: AsyncIterator[bytes]
    }
    
    class FileResponse {
        +path: str
        +media_type: str
        +filename: str
        +background: BackgroundTask
    }
    
    class ORJSONResponse {
        +media_type: str = "application/json"
        +render(content): bytes
    }
    
    class UJSONResponse {
        +media_type: str = "application/json"  
        +render(content): bytes
    }
    
    %% 数据结构
    class UploadFile {
        +filename: str
        +content_type: str
        +file: BinaryIO
        +size: int
        +read(): bytes
        +write(): int
        +seek(): int
        +close(): None
    }
    
    class BackgroundTasks {
        +tasks: List[BackgroundTask]
        +add_task(): None
        +__call__(): None
    }
    
    class BackgroundTask {
        +func: Callable
        +args: Tuple
        +kwargs: Dict
        +__call__(): None
    }
    
    %% 继承关系
    Response <|-- JSONResponse
    Response <|-- HTMLResponse
    Response <|-- PlainTextResponse
    Response <|-- RedirectResponse
    Response <|-- StreamingResponse
    Response <|-- FileResponse
    JSONResponse <|-- ORJSONResponse
    JSONResponse <|-- UJSONResponse
    
    %% 关联关系
    BackgroundTasks --> BackgroundTask : contains
    Response --> BackgroundTasks : may have
```

## 9. OpenAPI 集成系统类图

```mermaid
classDiagram
    %% OpenAPI 核心类
    class OpenAPI {
        +openapi: str
        +info: Info
        +servers: List[Server]
        +paths: Paths
        +components: Components
        +security: List[SecurityRequirement]
        +tags: List[Tag]
        +externalDocs: ExternalDocumentation
    }
    
    class Info {
        +title: str
        +version: str
        +description: str
        +summary: str
        +termsOfService: str
        +contact: Contact
        +license: License
    }
    
    class Components {
        +schemas: Dict[str, Schema]
        +responses: Dict[str, Response]
        +parameters: Dict[str, Parameter]
        +examples: Dict[str, Example]
        +requestBodies: Dict[str, RequestBody]
        +headers: Dict[str, Header]
        +securitySchemes: Dict[str, SecurityScheme]
        +links: Dict[str, Link]
        +callbacks: Dict[str, Callback]
    }
    
    class PathItem {
        +summary: str
        +description: str
        +get: Operation
        +post: Operation
        +put: Operation
        +delete: Operation
        +options: Operation
        +head: Operation
        +patch: Operation
        +trace: Operation
        +servers: List[Server]
        +parameters: List[Parameter]
    }
    
    class Operation {
        +tags: List[str]
        +summary: str
        +description: str
        +externalDocs: ExternalDocumentation
        +operationId: str
        +parameters: List[Parameter]
        +requestBody: RequestBody
        +responses: Responses
        +callbacks: Dict[str, Callback]
        +deprecated: bool
        +security: List[SecurityRequirement]
        +servers: List[Server]
    }
    
    class Parameter {
        +name: str
        +in: str
        +description: str
        +required: bool
        +deprecated: bool
        +allowEmptyValue: bool
        +style: str
        +explode: bool
        +allowReserved: bool
        +schema: Schema
        +example: Any
        +examples: Dict[str, Example]
    }
    
    class RequestBody {
        +description: str
        +content: Dict[str, MediaType]
        +required: bool
    }
    
    class MediaType {
        +schema: Schema
        +example: Any
        +examples: Dict[str, Example]
        +encoding: Dict[str, Encoding]
    }
    
    class Schema {
        +title: str
        +type: str
        +format: str
        +description: str
        +default: Any
        +example: Any
        +examples: List[Any]
        +required: List[str]
        +properties: Dict[str, Schema]
        +additionalProperties: Union[bool, Schema]
        +items: Schema
        +allOf: List[Schema]
        +oneOf: List[Schema]
        +anyOf: List[Schema]
    }
    
    class SecurityScheme {
        +type: str
        +description: str
        +name: str
        +in: str
        +scheme: str
        +bearerFormat: str
        +flows: OAuthFlows
        +openIdConnectUrl: str
    }
    
    %% 关系
    OpenAPI --> Info : contains
    OpenAPI --> Components : contains
    OpenAPI --> PathItem : contains
    PathItem --> Operation : contains  
    Operation --> Parameter : contains
    Operation --> RequestBody : contains
    RequestBody --> MediaType : contains
    MediaType --> Schema : contains
    Components --> SecurityScheme : contains
```

## 10. 异常处理系统类图

```mermaid
classDiagram
    %% 标准异常基类
    class Exception {
        <<built-in>>
        +args: Tuple
        +__str__(): str
    }
    
    %% Starlette 异常
    class HTTPException {
        +status_code: int
        +detail: Any
        +headers: Dict[str, str]
    }
    
    %% FastAPI 异常
    class FastAPIError {
        +__init__(message): None
    }
    
    class RequestValidationError {
        +errors: List[ErrorWrapper]
        +body: Any
        +__init__(errors, body): None
        +errors(): List[Dict[str, Any]]
    }
    
    class ResponseValidationError {
        +errors: List[ErrorWrapper]
        +__init__(errors): None
    }
    
    class WebSocketRequestValidationError {
        +errors: List[ErrorWrapper]
        +__init__(errors): None
    }
    
    class WebSocketException {
        +code: int
        +reason: str
        +__init__(code, reason): None
    }
    
    %% 错误处理相关
    class ErrorWrapper {
        +exc: Exception
        +loc: Tuple[Union[str, int], ...]
        +__init__(exc, loc): None
        +__repr__(): str
    }
    
    class ValidationError {
        <<external>>
        +errors: List[Dict]
        +model: Type[BaseModel]
    }
    
    %% 继承关系
    Exception <|-- FastAPIError
    Exception <|-- HTTPException
    Exception <|-- RequestValidationError
    Exception <|-- ResponseValidationError  
    Exception <|-- WebSocketRequestValidationError
    Exception <|-- WebSocketException
    
    %% 关联关系
    RequestValidationError --> ErrorWrapper : contains
    ResponseValidationError --> ErrorWrapper : contains
    WebSocketRequestValidationError --> ErrorWrapper : contains
    ErrorWrapper --> Exception : wraps
```

这些 UML 图展示了 FastAPI 的核心数据结构和它们之间的关系，帮助理解整个框架的架构设计：

1. **分层设计**：从 Starlette 基础到 FastAPI 扩展的清晰分层
2. **组合模式**：大量使用组合而非继承来构建复杂功能  
3. **类型安全**：通过 Pydantic 模型确保类型安全
4. **依赖注入**：通过 Dependant 和相关类实现复杂的依赖管理
5. **可扩展性**：通过抽象基类和接口设计支持自定义扩展

这些结构图为深入理解 FastAPI 的实现原理提供了重要参考。
