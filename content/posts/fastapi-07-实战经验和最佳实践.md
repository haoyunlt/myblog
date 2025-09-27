---
title: "FastAPI 源码剖析 - 实战经验和最佳实践"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['Python', 'Web框架', '源码分析', 'FastAPI', '最佳实践', 'API']
categories: ['Python框架', 'FastAPI']
description: "FastAPI 源码剖析 - 实战经验和最佳实践的深入技术分析文档"
keywords: ['Python', 'Web框架', '源码分析', 'FastAPI', '最佳实践', 'API']
author: "技术分析师"
weight: 1
---

## 1. 项目结构最佳实践

### 1.1 推荐的项目结构

```
my_fastapi_project/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 应用入口
│   ├── core/                   # 核心配置
│   │   ├── __init__.py
│   │   ├── config.py          # 配置管理
│   │   ├── security.py        # 安全配置
│   │   └── database.py        # 数据库配置
│   ├── api/                    # API 路由
│   │   ├── __init__.py
│   │   ├── deps.py            # 依赖函数
│   │   └── v1/                # API 版本管理
│   │       ├── __init__.py
│   │       ├── api.py         # 路由聚合
│   │       ├── endpoints/     # 具体端点
│   │       │   ├── __init__.py
│   │       │   ├── users.py
│   │       │   ├── items.py
│   │       │   └── auth.py
│   ├── models/                 # 数据模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── item.py
│   │   └── base.py
│   ├── schemas/               # Pydantic 模式
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── item.py
│   │   └── token.py
│   ├── crud/                  # CRUD 操作
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── user.py
│   │   └── item.py
│   ├── services/              # 业务逻辑
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   └── user_service.py
│   ├── middleware/            # 自定义中间件
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── timing.py
│   └── utils/                 # 工具函数
│       ├── __init__.py
│       ├── security.py
│       └── common.py
├── tests/                     # 测试文件
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_main.py
│   └── api/
│       └── test_users.py
├── alembic/                   # 数据库迁移
├── requirements.txt
├── pyproject.toml
├── .env
├── .gitignore
└── README.md
```

### 1.2 核心配置管理

```python
# app/core/config.py
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """
    应用配置类
    
    使用 Pydantic BaseSettings 进行配置管理的最佳实践：
    1. 类型安全的配置
    2. 自动环境变量映射
    3. 验证和转换
    4. 缓存配置实例
    """
    
    # 基本配置
    PROJECT_NAME: str = "FastAPI Project"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # 服务器配置
    HOST: str = "localhost"
    PORT: int = 8000
    
    # 数据库配置
    DATABASE_URL: str = "sqlite:///./test.db"
    ASYNC_DATABASE_URL: Optional[str] = None
    
    # 安全配置
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # CORS 配置
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # Redis 配置
    REDIS_URL: str = "redis://localhost:6379"
    
    # 外部 API 配置
    EXTERNAL_API_KEY: Optional[str] = None
    EXTERNAL_API_URL: Optional[str] = None
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @validator("DATABASE_URL", pre=True)
    def validate_database_url(cls, v: str) -> str:
        """验证数据库 URL 格式"""
        if not v.startswith(("sqlite://", "postgresql://", "mysql://")):
            raise ValueError("Invalid database URL format")
        return v
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """解析 CORS 来源列表"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        """Pydantic 配置"""
        env_file = ".env"                    # 环境变量文件
        env_file_encoding = "utf-8"          # 文件编码
        case_sensitive = True                # 环境变量名大小写敏感

@lru_cache()
def get_settings() -> Settings:
    """
    获取配置实例
    
    使用 @lru_cache() 确保配置只被加载一次，提高性能
    
    Returns:
        Settings: 配置实例
    """
    return Settings()

# 全局配置实例
settings = get_settings()
```

### 1.3 应用初始化最佳实践

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
import logging
import time

from app.core.config import settings
from app.api.v1.api import api_router
from app.core.database import engine, Base
from app.middleware.logging import LoggingMiddleware
from app.middleware.timing import TimingMiddleware

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    使用 async context manager 管理应用的启动和关闭
    这是现代 FastAPI 推荐的方式，替代 on_startup 和 on_shutdown
    """
    # 启动时执行
    logger.info("🚀 Starting application...")
    
    # 创建数据库表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("📊 Database tables created")
    
    # 可以在这里添加其他启动任务
    # - 缓存预热
    # - 外部服务连接检查
    # - 定时任务启动
    
    logger.info("✅ Application started successfully")
    
    yield  # 应用运行期间
    
    # 关闭时执行
    logger.info("🔄 Shutting down application...")
    
    # 清理资源
    await engine.dispose()
    logger.info("🗑️ Resources cleaned up")
    
    logger.info("🛑 Application shutdown complete")

# 创建 FastAPI 应用实例
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if not settings.DEBUG else "/openapi.json",
    docs_url="/docs" if settings.DEBUG else None,      # 生产环境可禁用文档
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,  # 使用生命周期管理
    description="""
    一个完整的 FastAPI 应用示例 🚀
    
    ## 功能特性
    
    * **用户管理**: 用户注册、登录、信息管理
    * **认证授权**: JWT Token 认证，权限控制
    * **CRUD 操作**: 完整的增删改查功能
    * **数据验证**: 基于 Pydantic 的数据验证
    * **API 文档**: 自动生成的交互式 API 文档
    """,
)

# === 中间件配置 ===

# CORS 中间件（必须在最前面）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
    expose_headers=["X-Process-Time"],  # 暴露自定义头部
)

# 压缩中间件
app.add_middleware(
    GZipMiddleware, 
    minimum_size=1000  # 只压缩大于 1KB 的响应
)

# 自定义中间件
app.add_middleware(TimingMiddleware)    # 请求计时
app.add_middleware(LoggingMiddleware)   # 请求日志

# === 异常处理 ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP 异常处理器
    
    统一处理 HTTP 异常，提供一致的错误响应格式
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": time.time(),
                "path": str(request.url),
            }
        },
        headers=getattr(exc, "headers", None),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    请求验证异常处理器
    
    处理 Pydantic 验证错误，提供详细的错误信息
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "ValidationError",
                "code": 422,
                "message": "请求数据验证失败",
                "details": exc.errors(),
                "timestamp": time.time(),
                "path": str(request.url),
            }
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    通用异常处理器
    
    处理所有未捕获的异常，避免应用崩溃
    """
    logger.exception(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "InternalServerError",
                "code": 500,
                "message": "服务器内部错误" if not settings.DEBUG else str(exc),
                "timestamp": time.time(),
                "path": str(request.url),
            }
        },
    )

# === 路由注册 ===
app.include_router(api_router, prefix=settings.API_V1_STR)

# === 健康检查端点 ===
@app.get("/health", tags=["监控"])
async def health_check():
    """
    健康检查端点
    
    用于监控系统检查应用状态
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "environment": "development" if settings.DEBUG else "production",
    }

# 根路径重定向到文档
@app.get("/", include_in_schema=False)
async def root():
    """根路径，重定向到 API 文档"""
    return {"message": "Welcome to FastAPI!", "docs_url": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
```

## 2. 依赖注入最佳实践

### 2.1 通用依赖函数

```python
# app/api/deps.py
from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.crud.user import user_crud

# OAuth2 方案
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    scopes={
        "read": "Read access",
        "write": "Write access", 
        "admin": "Administrative access"
    }
)

# === 数据库依赖 ===
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    数据库会话依赖
    
    使用异步生成器确保会话正确关闭
    
    Yields:
        AsyncSession: 数据库会话
        
    Examples:
        @app.get("/users/")
        async def list_users(db: AsyncSession = Depends(get_db)):
            return await user_crud.get_multi(db)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()  # 自动提交
        except Exception:
            await session.rollback()  # 出错时回滚
            raise
        finally:
            await session.close()

# === 认证依赖 ===
async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    获取当前用户
    
    从 JWT token 中解析用户信息
    
    Args:
        db: 数据库会话
        token: JWT 访问令牌
        
    Returns:
        User: 当前用户对象
        
    Raises:
        HTTPException: 认证失败时抛出 401 错误
        
    Examples:
        @app.get("/users/me")
        async def get_me(current_user: User = Depends(get_current_user)):
            return current_user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await user_crud.get(db, id=user_id)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    获取当前活跃用户
    
    检查用户是否被禁用
    
    Args:
        current_user: 当前用户
        
    Returns:
        User: 活跃用户
        
    Raises:
        HTTPException: 用户被禁用时抛出 400 错误
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户账户已被禁用"
        )
    return current_user

# === 权限依赖 ===
def require_permissions(*required_permissions: str):
    """
    权限检查依赖工厂
    
    创建需要特定权限的依赖函数
    
    Args:
        *required_permissions: 所需权限列表
        
    Returns:
        Callable: 权限检查依赖函数
        
    Examples:
        # 需要管理员权限
        require_admin = require_permissions("admin")
        
        @app.delete("/users/{user_id}")
        async def delete_user(
            user_id: int,
            current_user: User = Depends(require_admin)
        ):
            pass
    """
    def permission_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        """检查用户权限"""
        user_permissions = set(current_user.permissions)
        required_permissions_set = set(required_permissions)
        
        if not required_permissions_set.issubset(user_permissions):
            missing_permissions = required_permissions_set - user_permissions
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"缺少权限: {', '.join(missing_permissions)}"
            )
        
        return current_user
    
    return permission_checker

# === 分页依赖 ===
class PaginationParams:
    """分页参数类"""
    
    def __init__(
        self,
        skip: int = 0,      # 跳过记录数
        limit: int = 100,   # 每页记录数
    ):
        # 验证参数
        if skip < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="skip 参数不能小于 0"
            )
        
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit 参数必须在 1-1000 之间"
            )
        
        self.skip = skip
        self.limit = limit

def get_pagination_params(
    skip: int = 0,
    limit: int = 100,
) -> PaginationParams:
    """
    获取分页参数
    
    Args:
        skip: 跳过的记录数
        limit: 每页记录数，最大 1000
        
    Returns:
        PaginationParams: 分页参数对象
        
    Examples:
        @app.get("/users/")
        async def list_users(
            pagination: PaginationParams = Depends(get_pagination_params)
        ):
            return await user_crud.get_multi(
                db, skip=pagination.skip, limit=pagination.limit
            )
    """
    return PaginationParams(skip=skip, limit=limit)

# === 缓存依赖 ===
class CacheService:
    """缓存服务类"""
    
    def __init__(self):
        self.cache = {}  # 简单内存缓存，生产环境应使用 Redis
    
    async def get(self, key: str) -> Optional[str]:
        """获取缓存值"""
        return self.cache.get(key)
    
    async def set(self, key: str, value: str, expire: int = 3600) -> None:
        """设置缓存值"""
        self.cache[key] = value
    
    async def delete(self, key: str) -> None:
        """删除缓存"""
        self.cache.pop(key, None)

# 缓存服务单例
cache_service = CacheService()

async def get_cache_service() -> CacheService:
    """获取缓存服务依赖"""
    return cache_service

# === 预设的常用依赖 ===

# 管理员权限
require_admin = require_permissions("admin")

# 写权限  
require_write = require_permissions("write")

# 读写权限
require_read_write = require_permissions("read", "write")
```

### 2.2 高级依赖模式

```python
# app/api/advanced_deps.py
from typing import List, Dict, Any, Optional, TypeVar, Generic, Callable
from fastapi import Depends, HTTPException, Query, Path
from pydantic import BaseModel
import functools

T = TypeVar('T')

class DependencyCache(Generic[T]):
    """依赖缓存装饰器"""
    
    def __init__(self):
        self._cache: Dict[str, T] = {}
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # 生成缓存键
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            if cache_key not in self._cache:
                self._cache[cache_key] = await func(*args, **kwargs)
            
            return self._cache[cache_key]
        
        return wrapper

# 条件依赖
def conditional_dependency(
    condition: bool,
    true_dependency: Callable,
    false_dependency: Callable
):
    """
    条件依赖
    
    根据条件选择不同的依赖
    
    Args:
        condition: 判断条件
        true_dependency: 条件为真时的依赖
        false_dependency: 条件为假时的依赖
        
    Returns:
        选择的依赖函数
    """
    return Depends(true_dependency if condition else false_dependency)

# 组合依赖
class CombinedDependency:
    """组合多个依赖的结果"""
    
    def __init__(self, **dependencies):
        self.dependencies = dependencies
    
    def __getattr__(self, name: str):
        if name in self.dependencies:
            return self.dependencies[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def combine_dependencies(**dep_mapping) -> CombinedDependency:
    """
    组合多个依赖
    
    将多个依赖的结果组合到一个对象中
    
    Args:
        **dep_mapping: 依赖映射，格式为 name=Depends(dependency)
        
    Returns:
        CombinedDependency: 组合依赖对象
        
    Examples:
        combined = combine_dependencies(
            user=Depends(get_current_user),
            db=Depends(get_db),
            cache=Depends(get_cache_service)
        )
        
        @app.get("/complex-endpoint")
        async def complex_operation(deps: CombinedDependency = Depends(combined)):
            user = deps.user
            db = deps.db
            cache = deps.cache
            # 使用组合的依赖
    """
    async def dependency(**kwargs):
        return CombinedDependency(**kwargs)
    
    # 动态设置依赖参数
    import inspect
    sig = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=dep)
        for name, dep in dep_mapping.items()
    ])
    dependency.__signature__ = sig
    
    return dependency

# 资源锁定依赖
class ResourceLock:
    """资源锁定管理"""
    
    def __init__(self):
        self._locks: Dict[str, bool] = {}
    
    async def acquire(self, resource_id: str) -> bool:
        """获取资源锁"""
        if resource_id in self._locks:
            return False
        self._locks[resource_id] = True
        return True
    
    async def release(self, resource_id: str) -> None:
        """释放资源锁"""
        self._locks.pop(resource_id, None)

resource_lock = ResourceLock()

def require_resource_lock(resource_id_param: str = "id"):
    """
    需要资源锁的依赖
    
    Args:
        resource_id_param: 资源ID参数名
        
    Returns:
        依赖函数
    """
    def dependency(request_data: dict = None, **path_params):
        resource_id = path_params.get(resource_id_param)
        if not resource_id:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required parameter: {resource_id_param}"
            )
        
        if not resource_lock.acquire(resource_id):
            raise HTTPException(
                status_code=423,  # Locked
                detail=f"Resource {resource_id} is currently locked"
            )
        
        # 这里应该在请求结束后自动释放锁
        # 实际实现需要更复杂的生命周期管理
        return resource_id
    
    return Depends(dependency)
```

## 3. 模型和Schema设计模式

### 3.1 分层模型设计

```python
# app/models/base.py
from sqlalchemy import Column, Integer, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class BaseModel(Base):
    """数据库模型基类"""
    
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True, comment="主键ID")
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    is_active = Column(Boolean, default=True, comment="是否活跃")

# app/models/user.py  
from sqlalchemy import Column, String, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship

from .base import BaseModel

class User(BaseModel):
    """用户模型"""
    
    __tablename__ = "users"
    __table_args__ = {'comment': '用户表'}
    
    # 基本信息
    email = Column(String(100), unique=True, index=True, nullable=False, comment="邮箱")
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    hashed_password = Column(String(255), nullable=False, comment="密码哈希")
    
    # 个人信息
    full_name = Column(String(100), comment="全名")
    phone = Column(String(20), comment="手机号")
    avatar_url = Column(String(500), comment="头像URL")
    bio = Column(Text, comment="个人简介")
    
    # 状态
    is_active = Column(Boolean, default=True, comment="是否活跃")
    is_verified = Column(Boolean, default=False, comment="是否验证")
    is_superuser = Column(Boolean, default=False, comment="是否超级用户")
    
    # 关系
    items = relationship("Item", back_populates="owner", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

# app/schemas/base.py
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

DataType = TypeVar("DataType")

class BaseSchema(BaseModel):
    """Pydantic 模式基类"""
    
    model_config = ConfigDict(
        # 允许使用 ORM 对象
        from_attributes=True,
        # 验证赋值
        validate_assignment=True,
        # 使用枚举值
        use_enum_values=True,
        # 严格模式
        strict=False,
    )

class TimestampMixin(BaseModel):
    """时间戳混入类"""
    
    created_at: Optional[datetime] = Field(None, description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")

class IDMixin(BaseModel):
    """ID混入类"""
    
    id: Optional[int] = Field(None, description="主键ID", ge=1)

# 通用响应模式
class ResponseBase(BaseSchema, Generic[DataType]):
    """通用响应基类"""
    
    success: bool = Field(True, description="是否成功")
    message: str = Field("操作成功", description="响应消息")
    data: Optional[DataType] = Field(None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")

class PaginatedResponse(ResponseBase[List[DataType]]):
    """分页响应模式"""
    
    total: int = Field(description="总记录数", ge=0)
    page: int = Field(description="当前页码", ge=1)
    per_page: int = Field(description="每页记录数", ge=1, le=1000)
    pages: int = Field(description="总页数", ge=0)

# app/schemas/user.py
from typing import List, Optional
from pydantic import EmailStr, Field, validator
import re

from .base import BaseSchema, IDMixin, TimestampMixin

# 用户基础模式
class UserBase(BaseSchema):
    """用户基础模式"""
    
    email: EmailStr = Field(description="邮箱地址")
    username: str = Field(description="用户名", min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, description="全名", max_length=100)
    phone: Optional[str] = Field(None, description="手机号", regex=r"^1[3-9]\d{9}$")
    bio: Optional[str] = Field(None, description="个人简介", max_length=500)
    
    @validator("username")
    def validate_username(cls, v):
        """验证用户名格式"""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("用户名只能包含字母、数字、下划线和短横线")
        return v

class UserCreate(UserBase):
    """创建用户模式"""
    
    password: str = Field(description="密码", min_length=8)
    password_confirm: str = Field(description="确认密码")
    
    @validator("password")
    def validate_password(cls, v):
        """验证密码强度"""
        if not re.search(r"[A-Z]", v):
            raise ValueError("密码必须包含至少一个大写字母")
        if not re.search(r"[a-z]", v):
            raise ValueError("密码必须包含至少一个小写字母")
        if not re.search(r"\d", v):
            raise ValueError("密码必须包含至少一个数字")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("密码必须包含至少一个特殊字符")
        return v
    
    @validator("password_confirm")
    def validate_password_match(cls, v, values):
        """验证密码确认"""
        if "password" in values and v != values["password"]:
            raise ValueError("两次输入的密码不一致")
        return v

class UserUpdate(BaseSchema):
    """更新用户模式"""
    
    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, regex=r"^1[3-9]\d{9}$")
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = Field(None, max_length=500)

class UserResponse(UserBase, IDMixin, TimestampMixin):
    """用户响应模式"""
    
    is_active: bool = Field(description="是否活跃")
    is_verified: bool = Field(description="是否验证")
    is_superuser: bool = Field(description="是否超级用户")
    avatar_url: Optional[str] = Field(None, description="头像URL")

class UserDetail(UserResponse):
    """用户详情模式（包含关联数据）"""
    
    items_count: Optional[int] = Field(None, description="用户项目数量")
    last_login: Optional[datetime] = Field(None, description="最后登录时间")

# 认证相关模式
class Token(BaseSchema):
    """访问令牌模式"""
    
    access_token: str = Field(description="访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(description="过期时间（秒）")
    refresh_token: Optional[str] = Field(None, description="刷新令牌")

class TokenData(BaseSchema):
    """令牌数据模式"""
    
    user_id: Optional[int] = None
    username: Optional[str] = None
    scopes: List[str] = []
```

### 3.2 模型工厂模式

```python
# app/utils/model_factory.py
from typing import Type, TypeVar, Dict, Any, Optional, List
from faker import Faker
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.item import Item

T = TypeVar('T')

class ModelFactory:
    """模型工厂基类"""
    
    def __init__(self, model_class: Type[T]):
        self.model_class = model_class
        self.faker = Faker('zh_CN')  # 中文本地化
    
    def create(self, db: Session, **kwargs) -> T:
        """创建模型实例"""
        defaults = self.get_defaults()
        defaults.update(kwargs)
        
        instance = self.model_class(**defaults)
        db.add(instance)
        db.commit()
        db.refresh(instance)
        
        return instance
    
    def create_batch(self, db: Session, count: int, **kwargs) -> List[T]:
        """批量创建模型实例"""
        instances = []
        for _ in range(count):
            instance = self.create(db, **kwargs)
            instances.append(instance)
        return instances
    
    def get_defaults(self) -> Dict[str, Any]:
        """获取默认值，子类重写"""
        return {}

class UserFactory(ModelFactory):
    """用户工厂"""
    
    def __init__(self):
        super().__init__(User)
    
    def get_defaults(self) -> Dict[str, Any]:
        return {
            'email': self.faker.email(),
            'username': self.faker.user_name(),
            'full_name': self.faker.name(),
            'hashed_password': 'hashed_password',  # 实际应用中应该使用真实的哈希密码
            'phone': self.faker.phone_number(),
            'bio': self.faker.text(max_nb_chars=200),
        }
    
    def create_admin(self, db: Session, **kwargs) -> User:
        """创建管理员用户"""
        defaults = {
            'is_superuser': True,
            'is_verified': True,
            'username': 'admin',
            'email': 'admin@example.com',
        }
        defaults.update(kwargs)
        return self.create(db, **defaults)

# 工厂实例
user_factory = UserFactory()

# 使用示例
def create_test_data(db: Session):
    """创建测试数据"""
    # 创建管理员
    admin = user_factory.create_admin(db)
    
    # 批量创建普通用户
    users = user_factory.create_batch(db, count=10)
    
    return {"admin": admin, "users": users}
```

## 4. API设计最佳实践

### 4.1 RESTful API 设计

```python
# app/api/v1/endpoints/users.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user, require_admin, PaginationParams
from app.crud.user import user_crud
from app.schemas.user import UserCreate, UserUpdate, UserResponse, UserDetail
from app.schemas.base import ResponseBase, PaginatedResponse
from app.models.user import User

router = APIRouter()

@router.get(
    "/",
    response_model=PaginatedResponse[UserResponse],
    summary="获取用户列表",
    description="""
    获取用户列表，支持分页和筛选
    
    - **skip**: 跳过的记录数
    - **limit**: 每页记录数（最大1000）
    - **search**: 搜索关键词（用户名或邮箱）
    - **is_active**: 筛选活跃状态
    """,
    response_description="用户列表",
    tags=["用户管理"],
)
async def list_users(
    *,
    db: Session = Depends(get_db),
    pagination: PaginationParams = Depends(),
    search: Optional[str] = Query(None, description="搜索关键词"),
    is_active: Optional[bool] = Query(None, description="是否活跃"),
    current_user: User = Depends(require_admin),  # 需要管理员权限
) -> PaginatedResponse[UserResponse]:
    """
    获取用户列表
    
    需要管理员权限才能访问
    支持按用户名、邮箱搜索和状态筛选
    """
    # 构建筛选条件
    filters = {}
    if search:
        filters["search"] = search
    if is_active is not None:
        filters["is_active"] = is_active
    
    # 获取数据
    users, total = await user_crud.get_multi_with_count(
        db, 
        skip=pagination.skip, 
        limit=pagination.limit,
        **filters
    )
    
    # 计算分页信息
    total_pages = (total + pagination.limit - 1) // pagination.limit
    current_page = (pagination.skip // pagination.limit) + 1
    
    return PaginatedResponse[UserResponse](
        data=[UserResponse.model_validate(user) for user in users],
        total=total,
        page=current_page,
        per_page=pagination.limit,
        pages=total_pages,
        message=f"成功获取 {len(users)} 个用户"
    )

@router.get(
    "/{user_id}",
    response_model=ResponseBase[UserDetail],
    summary="获取用户详情",
    description="根据用户ID获取用户详细信息",
    responses={
        404: {"description": "用户不存在"},
        403: {"description": "权限不足"},
    },
    tags=["用户管理"],
)
async def get_user(
    user_id: int = Path(description="用户ID", ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> ResponseBase[UserDetail]:
    """
    获取用户详情
    
    用户只能查看自己的详细信息，管理员可以查看所有用户
    """
    user = await user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 权限检查：只能查看自己的信息，除非是管理员
    if user.id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限查看此用户信息"
        )
    
    # 获取扩展信息
    user_detail = await user_crud.get_user_detail(db, user_id=user_id)
    
    return ResponseBase[UserDetail](
        data=UserDetail.model_validate(user_detail),
        message="成功获取用户详情"
    )

@router.post(
    "/",
    response_model=ResponseBase[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="创建用户",
    description="创建新用户账户",
    responses={
        201: {"description": "用户创建成功"},
        400: {"description": "请求数据无效"},
        409: {"description": "用户已存在"},
    },
    tags=["用户管理"],
)
async def create_user(
    *,
    user_in: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),  # 需要管理员权限
) -> ResponseBase[UserResponse]:
    """
    创建新用户
    
    需要管理员权限
    会自动检查用户名和邮箱的唯一性
    """
    # 检查用户是否已存在
    existing_user = await user_crud.get_by_email(db, email=user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="邮箱已被使用"
        )
    
    existing_user = await user_crud.get_by_username(db, username=user_in.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已被使用"
        )
    
    # 创建用户
    user = await user_crud.create(db, obj_in=user_in)
    
    return ResponseBase[UserResponse](
        data=UserResponse.model_validate(user),
        message="用户创建成功"
    )

@router.put(
    "/{user_id}",
    response_model=ResponseBase[UserResponse],
    summary="更新用户信息",
    description="更新指定用户的信息",
    tags=["用户管理"],
)
async def update_user(
    *,
    user_id: int = Path(description="用户ID", ge=1),
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> ResponseBase[UserResponse]:
    """
    更新用户信息
    
    用户只能更新自己的信息，管理员可以更新所有用户
    """
    user = await user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 权限检查
    if user.id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限修改此用户信息"
        )
    
    # 更新用户
    user = await user_crud.update(db, db_obj=user, obj_in=user_in)
    
    return ResponseBase[UserResponse](
        data=UserResponse.model_validate(user),
        message="用户信息更新成功"
    )

@router.delete(
    "/{user_id}",
    response_model=ResponseBase[None],
    summary="删除用户",
    description="删除指定用户（软删除）",
    tags=["用户管理"],
)
async def delete_user(
    *,
    user_id: int = Path(description="用户ID", ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),  # 需要管理员权限
) -> ResponseBase[None]:
    """
    删除用户
    
    需要管理员权限
    执行软删除，将用户状态设为非活跃
    """
    user = await user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能删除自己的账户"
        )
    
    # 软删除：设置为非活跃状态
    await user_crud.update(
        db, 
        db_obj=user, 
        obj_in={"is_active": False}
    )
    
    return ResponseBase[None](
        message="用户删除成功"
    )

# 当前用户相关端点
@router.get(
    "/me/profile",
    response_model=ResponseBase[UserDetail],
    summary="获取当前用户信息",
    description="获取当前登录用户的详细信息",
    tags=["当前用户"],
)
async def get_current_user_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> ResponseBase[UserDetail]:
    """获取当前用户的详细信息"""
    user_detail = await user_crud.get_user_detail(db, user_id=current_user.id)
    
    return ResponseBase[UserDetail](
        data=UserDetail.model_validate(user_detail),
        message="成功获取用户信息"
    )

@router.put(
    "/me/profile",
    response_model=ResponseBase[UserResponse],
    summary="更新当前用户信息",
    description="更新当前登录用户的个人信息",
    tags=["当前用户"],
)
async def update_current_user_profile(
    *,
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> ResponseBase[UserResponse]:
    """更新当前用户的个人信息"""
    user = await user_crud.update(db, db_obj=current_user, obj_in=user_in)
    
    return ResponseBase[UserResponse](
        data=UserResponse.model_validate(user),
        message="个人信息更新成功"
    )
```

## 5. 中间件开发最佳实践

### 5.1 自定义中间件示例

```python
# app/middleware/logging.py
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    
    记录每个请求的详细信息：
    - 请求ID（用于追踪）
    - 请求方法和路径
    - 客户端IP地址
    - 用户代理
    - 处理时间
    - 响应状态码
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成唯一请求ID
        request_id = str(uuid.uuid4())[:8]
        
        # 在请求对象中存储请求ID，供后续使用
        request.state.request_id = request_id
        
        # 记录请求开始
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.info(
            f"[{request_id}] 请求开始: {request.method} {request.url} "
            f"客户端: {client_ip} 用户代理: {user_agent}"
        )
        
        # 处理请求
        response = await call_next(request)
        
        # 记录请求结束
        process_time = time.time() - start_time
        
        # 根据状态码选择日志级别
        if response.status_code >= 400:
            log_level = logging.ERROR if response.status_code >= 500 else logging.WARNING
            logger.log(
                log_level,
                f"[{request_id}] 请求完成: {request.method} {request.url} "
                f"状态码: {response.status_code} 耗时: {process_time:.4f}s"
            )
        else:
            logger.info(
                f"[{request_id}] 请求完成: {request.method} {request.url} "
                f"状态码: {response.status_code} 耗时: {process_time:.4f}s"
            )
        
        # 在响应头中添加请求ID和处理时间
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

# app/middleware/timing.py
class TimingMiddleware(BaseHTTPMiddleware):
    """
    请求计时中间件
    
    测量和记录请求处理时间
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        process_time = time.perf_counter() - start_time
        
        # 添加计时头部
        response.headers["X-Process-Time"] = str(process_time)
        
        # 记录慢请求
        if process_time > 1.0:  # 超过1秒的请求
            logger.warning(
                f"慢请求警告: {request.method} {request.url} "
                f"耗时: {process_time:.4f}s"
            )
        
        return response

# app/middleware/rate_limit.py
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import HTTPException, status

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    简单的限流中间件
    
    基于客户端IP的请求频率限制
    """
    
    def __init__(
        self, 
        app,
        calls: int = 100,           # 允许的请求次数
        period: int = 3600,         # 时间窗口（秒）
        exclude_paths: list = None  # 排除的路径
    ):
        super().__init__(app)
        self.calls = calls
        self.period = timedelta(seconds=period)
        self.exclude_paths = exclude_paths or []
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 检查是否排除此路径
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.now()
        
        # 清理过期记录
        cutoff = now - self.period
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if req_time > cutoff
        ]
        
        # 检查是否超过限制
        if len(self.requests[client_ip]) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="请求过于频繁，请稍后再试",
                headers={
                    "Retry-After": str(self.period.total_seconds())
                }
            )
        
        # 记录当前请求
        self.requests[client_ip].append(now)
        
        return await call_next(request)

# app/middleware/security.py
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    安全头部中间件
    
    添加常用的安全HTTP头部
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # 添加安全头部
        security_headers = {
            "X-Content-Type-Options": "nosniff",           # 防止MIME类型嗅探
            "X-Frame-Options": "DENY",                      # 防止点击劫持
            "X-XSS-Protection": "1; mode=block",           # XSS保护
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",  # HTTPS强制
            "Referrer-Policy": "strict-origin-when-cross-origin",  # 引荐来源策略
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",  # 权限策略
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
```

## 6. 测试最佳实践

### 6.1 测试配置和工具

```python
# tests/conftest.py
import asyncio
from typing import AsyncGenerator, Generator
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.api.deps import get_db
from app.core.config import settings
from app.models.base import Base
from app.utils.model_factory import user_factory

# 测试数据库引擎
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
test_engine = create_async_engine(
    TEST_DATABASE_URL, 
    echo=False,
    future=True
)

TestSessionLocal = sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def setup_database():
    """设置测试数据库"""
    # 创建表
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # 清理
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await test_engine.dispose()

@pytest.fixture
async def db_session(setup_database) -> AsyncGenerator[AsyncSession, None]:
    """创建数据库会话"""
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

@pytest.fixture
def override_get_db(db_session: AsyncSession):
    """覆盖数据库依赖"""
    async def _override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()

@pytest.fixture
async def client(override_get_db) -> AsyncGenerator[AsyncClient, None]:
    """创建测试客户端"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def admin_user(db_session: AsyncSession):
    """创建管理员用户"""
    return user_factory.create_admin(db_session)

@pytest.fixture
async def regular_user(db_session: AsyncSession):
    """创建普通用户"""
    return user_factory.create(db_session)

@pytest.fixture
async def admin_token(admin_user) -> str:
    """获取管理员令牌"""
    from app.core.security import create_access_token
    return create_access_token(data={"sub": str(admin_user.id)})

@pytest.fixture
async def user_token(regular_user) -> str:
    """获取普通用户令牌"""
    from app.core.security import create_access_token
    return create_access_token(data={"sub": str(regular_user.id)})

@pytest.fixture
def auth_headers(admin_token: str) -> dict:
    """获取认证头部"""
    return {"Authorization": f"Bearer {admin_token}"}
```

### 6.2 API 测试示例

```python
# tests/api/test_users.py
import pytest
from httpx import AsyncClient
from fastapi import status

from app.models.user import User

class TestUserAPI:
    """用户API测试"""
    
    async def test_create_user_success(
        self, 
        client: AsyncClient, 
        auth_headers: dict
    ):
        """测试成功创建用户"""
        user_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "full_name": "New User",
            "password": "Password123!",
            "password_confirm": "Password123!"
        }
        
        response = await client.post(
            "/api/v1/users/", 
            json=user_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["success"] is True
        assert data["data"]["email"] == user_data["email"]
        assert data["data"]["username"] == user_data["username"]
        assert "password" not in data["data"]  # 确保密码不会返回
    
    async def test_create_user_duplicate_email(
        self, 
        client: AsyncClient, 
        auth_headers: dict,
        regular_user: User
    ):
        """测试创建重复邮箱用户"""
        user_data = {
            "email": regular_user.email,  # 使用已存在的邮箱
            "username": "newuser",
            "password": "Password123!",
            "password_confirm": "Password123!"
        }
        
        response = await client.post(
            "/api/v1/users/",
            json=user_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_409_CONFLICT
        data = response.json()
        assert "邮箱已被使用" in data["error"]["message"]
    
    async def test_create_user_weak_password(
        self, 
        client: AsyncClient, 
        auth_headers: dict
    ):
        """测试弱密码验证"""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "weak",  # 弱密码
            "password_confirm": "weak"
        }
        
        response = await client.post(
            "/api/v1/users/",
            json=user_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert any("password" in str(error) for error in data["error"]["details"])
    
    async def test_get_user_list(
        self, 
        client: AsyncClient, 
        auth_headers: dict,
        regular_user: User
    ):
        """测试获取用户列表"""
        response = await client.get(
            "/api/v1/users/",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        assert data["total"] >= 1
        assert data["page"] == 1
    
    async def test_get_user_list_with_search(
        self, 
        client: AsyncClient, 
        auth_headers: dict,
        regular_user: User
    ):
        """测试搜索用户"""
        response = await client.get(
            f"/api/v1/users/?search={regular_user.username}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1
        assert any(
            user["username"] == regular_user.username 
            for user in data["data"]
        )
    
    async def test_get_user_detail_success(
        self, 
        client: AsyncClient, 
        auth_headers: dict,
        regular_user: User
    ):
        """测试获取用户详情"""
        response = await client.get(
            f"/api/v1/users/{regular_user.id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == regular_user.id
        assert data["data"]["username"] == regular_user.username
    
    async def test_get_user_detail_not_found(
        self, 
        client: AsyncClient, 
        auth_headers: dict
    ):
        """测试获取不存在的用户"""
        response = await client.get(
            "/api/v1/users/99999",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "用户不存在" in data["error"]["message"]
    
    async def test_update_user_success(
        self, 
        client: AsyncClient, 
        auth_headers: dict,
        regular_user: User
    ):
        """测试更新用户信息"""
        update_data = {
            "full_name": "Updated Name",
            "bio": "Updated bio"
        }
        
        response = await client.put(
            f"/api/v1/users/{regular_user.id}",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["data"]["full_name"] == update_data["full_name"]
        assert data["data"]["bio"] == update_data["bio"]
    
    async def test_delete_user_success(
        self, 
        client: AsyncClient, 
        auth_headers: dict,
        db_session,
        user_factory
    ):
        """测试删除用户"""
        # 创建一个要删除的用户
        user_to_delete = user_factory.create(db_session)
        
        response = await client.delete(
            f"/api/v1/users/{user_to_delete.id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "删除成功" in data["message"]
    
    async def test_unauthorized_access(self, client: AsyncClient):
        """测试未授权访问"""
        response = await client.get("/api/v1/users/")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    async def test_permission_denied(
        self, 
        client: AsyncClient,
        user_token: str  # 普通用户token
    ):
        """测试权限不足"""
        headers = {"Authorization": f"Bearer {user_token}"}
        
        response = await client.get(
            "/api/v1/users/",
            headers=headers
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
```

## 7. 性能优化实践

### 7.1 数据库查询优化

```python
# app/crud/optimized.py
from typing import List, Optional, Dict, Any
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload, joinedload, contains_eager
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.item import Item

class OptimizedUserCRUD:
    """优化的用户CRUD操作"""
    
    async def get_users_with_items_count(
        self, 
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取用户列表并包含其项目数量
        
        使用 subquery 避免 N+1 查询问题
        """
        # 子查询计算每个用户的项目数量
        items_count_subq = (
            select(
                Item.owner_id,
                func.count(Item.id).label("items_count")
            )
            .group_by(Item.owner_id)
            .subquery()
        )
        
        # 主查询
        query = (
            select(
                User,
                func.coalesce(items_count_subq.c.items_count, 0).label("items_count")
            )
            .outerjoin(items_count_subq, User.id == items_count_subq.c.owner_id)
            .offset(skip)
            .limit(limit)
        )
        
        result = await db.execute(query)
        
        return [
            {
                "user": user,
                "items_count": items_count
            }
            for user, items_count in result.all()
        ]
    
    async def get_user_with_items_eager(
        self, 
        db: AsyncSession, 
        user_id: int
    ) -> Optional[User]:
        """
        预加载用户的所有项目
        
        使用 selectinload 避免懒加载导致的额外查询
        """
        query = (
            select(User)
            .options(selectinload(User.items))  # 预加载关联项目
            .where(User.id == user_id)
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def search_users_optimized(
        self,
        db: AsyncSession,
        search_term: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        优化的用户搜索
        
        使用数据库的全文搜索功能（如果支持）
        或使用索引优化的 LIKE 查询
        """
        search_pattern = f"%{search_term}%"
        
        query = (
            select(User)
            .where(
                or_(
                    User.username.ilike(search_pattern),
                    User.email.ilike(search_pattern),
                    User.full_name.ilike(search_pattern)
                )
            )
            .order_by(
                # 优先显示用户名匹配的结果
                func.case(
                    (User.username.ilike(search_pattern), 1),
                    (User.email.ilike(search_pattern), 2),
                    else_=3
                )
            )
            .offset(skip)
            .limit(limit)
        )
        
        result = await db.execute(query)
        return result.scalars().all()

# 缓存优化
from functools import lru_cache
import json
from typing import Union

class CachedUserService:
    """带缓存的用户服务"""
    
    def __init__(self, cache_service):
        self.cache = cache_service
    
    async def get_user_by_id(
        self, 
        db: AsyncSession, 
        user_id: int
    ) -> Optional[User]:
        """
        带缓存的获取用户
        
        首先尝试从缓存获取，缓存未命中时查询数据库
        """
        cache_key = f"user:{user_id}"
        
        # 尝试从缓存获取
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            user_data = json.loads(cached_data)
            return User(**user_data)
        
        # 从数据库查询
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if user:
            # 缓存结果
            user_dict = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                # ... 其他字段
            }
            await self.cache.set(
                cache_key, 
                json.dumps(user_dict), 
                expire=3600  # 1小时过期
            )
        
        return user
    
    async def invalidate_user_cache(self, user_id: int) -> None:
        """使用户缓存失效"""
        cache_key = f"user:{user_id}"
        await self.cache.delete(cache_key)

# 批量操作优化
class BatchUserOperations:
    """批量用户操作"""
    
    async def bulk_create_users(
        self,
        db: AsyncSession,
        users_data: List[dict],
        batch_size: int = 1000
    ) -> List[User]:
        """
        批量创建用户
        
        使用批量插入提高性能
        """
        created_users = []
        
        # 分批处理，避免单次操作过大
        for i in range(0, len(users_data), batch_size):
            batch = users_data[i:i + batch_size]
            
            # 创建用户对象
            users = [User(**user_data) for user_data in batch]
            
            # 批量添加
            db.add_all(users)
            await db.commit()
            
            # 刷新以获取ID
            for user in users:
                await db.refresh(user)
            
            created_users.extend(users)
        
        return created_users
    
    async def bulk_update_users(
        self,
        db: AsyncSession,
        updates: List[Dict[str, Any]]
    ) -> None:
        """
        批量更新用户
        
        使用 bulk_update_mappings 提高性能
        """
        if not updates:
            return
        
        # 使用 SQLAlchemy 的批量更新
        await db.execute(
            User.__table__.update(),
            updates
        )
        await db.commit()
```

### 7.2 异步和并发优化

```python
# app/services/async_service.py
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import httpx

class AsyncUserService:
    """异步用户服务"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def enrich_users_data(
        self, 
        users: List[User]
    ) -> List[Dict[str, Any]]:
        """
        并发丰富用户数据
        
        同时从多个外部服务获取用户的额外信息
        """
        tasks = []
        for user in users:
            task = self._enrich_single_user(user)
            tasks.append(task)
        
        # 并发执行所有任务
        enriched_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, data in enumerate(enriched_data):
            if isinstance(data, Exception):
                # 处理异常，使用默认值
                results.append({
                    "user": users[i],
                    "avatar_url": None,
                    "social_data": {},
                    "error": str(data)
                })
            else:
                results.append(data)
        
        return results
    
    async def _enrich_single_user(self, user: User) -> Dict[str, Any]:
        """为单个用户获取额外数据"""
        # 并发获取多个外部数据
        avatar_task = self._get_user_avatar(user.email)
        social_task = self._get_social_data(user.username)
        
        avatar_url, social_data = await asyncio.gather(
            avatar_task, 
            social_task,
            return_exceptions=True
        )
        
        return {
            "user": user,
            "avatar_url": avatar_url if not isinstance(avatar_url, Exception) else None,
            "social_data": social_data if not isinstance(social_data, Exception) else {},
        }
    
    async def _get_user_avatar(self, email: str) -> Optional[str]:
        """从 Gravatar 获取用户头像"""
        import hashlib
        
        # 计算邮箱的 MD5 哈希
        email_hash = hashlib.md5(email.lower().encode()).hexdigest()
        gravatar_url = f"https://www.gravatar.com/avatar/{email_hash}?d=404"
        
        try:
            response = await self.http_client.head(gravatar_url, timeout=2.0)
            if response.status_code == 200:
                return gravatar_url.replace("?d=404", "?s=200")  # 返回200px大小
        except:
            pass
        
        return None
    
    async def _get_social_data(self, username: str) -> Dict[str, Any]:
        """获取用户社交数据（示例）"""
        try:
            # 模拟从社交平台API获取数据
            await asyncio.sleep(0.1)  # 模拟网络延迟
            return {
                "followers": 100,
                "following": 50,
                "posts": 25
            }
        except:
            return {}
    
    async def batch_send_emails(
        self, 
        email_tasks: List[Dict[str, Any]]
    ) -> List[bool]:
        """
        批量发送邮件
        
        使用信号量限制并发数量
        """
        semaphore = asyncio.Semaphore(5)  # 最多同时发送5封邮件
        
        async def send_single_email(email_data: Dict[str, Any]) -> bool:
            async with semaphore:
                return await self._send_email(email_data)
        
        # 创建所有邮件发送任务
        tasks = [send_single_email(email_data) for email_data in email_tasks]
        
        # 并发执行，收集结果
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        success_results = []
        for result in results:
            if isinstance(result, Exception):
                success_results.append(False)
            else:
                success_results.append(result)
        
        return success_results
    
    async def _send_email(self, email_data: Dict[str, Any]) -> bool:
        """发送单封邮件"""
        try:
            # 模拟发送邮件
            await asyncio.sleep(0.5)  # 模拟发送时间
            
            # 这里应该是实际的邮件发送逻辑
            # 例如使用 aiosmtplib 或调用邮件服务API
            
            return True
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return False
    
    async def close(self):
        """清理资源"""
        await self.http_client.aclose()
        self.executor.shutdown(wait=True)

# 使用示例
async def optimize_user_operations():
    """优化用户操作的使用示例"""
    async_service = AsyncUserService()
    
    try:
        # 获取用户列表
        users = await get_users_from_db()
        
        # 并发丰富用户数据
        enriched_users = await async_service.enrich_users_data(users)
        
        # 准备邮件数据
        email_tasks = [
            {
                "to": user_data["user"].email,
                "subject": "Welcome!",
                "body": f"Hello {user_data['user'].username}!"
            }
            for user_data in enriched_users
        ]
        
        # 批量发送邮件
        email_results = await async_service.batch_send_emails(email_tasks)
        
        return {
            "users_processed": len(enriched_users),
            "emails_sent": sum(email_results)
        }
    
    finally:
        await async_service.close()
```

## 8. 总结

本文档提供了 FastAPI 开发的全面最佳实践，涵盖了从项目结构到性能优化的各个方面。这些实践基于以下原则：

1. **代码组织**：清晰的分层架构和模块划分
2. **类型安全**：充分利用 Python 类型提示
3. **异步优先**：使用异步编程提高性能
4. **测试驱动**：完善的测试覆盖
5. **安全考虑**：认证、授权和数据验证
6. **性能优化**：数据库查询优化和并发处理
7. **可维护性**：文档完善，代码简洁

遵循这些最佳实践可以帮助开发者构建高质量、高性能和易维护的 FastAPI 应用。
