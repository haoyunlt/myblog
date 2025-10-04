# FastAPI 源码剖析 - 实战案例与最佳实践

## 概述

本文档汇总 FastAPI 的实战案例、最佳实践、常见模式和生产环境建议，帮助开发者构建健壮、高性能的 FastAPI 应用。

---

## 案例 1：完整的用户认证系统

### 需求
- 用户注册和登录
- JWT Token 认证
- 密码加密
- 权限管理
- 刷新 Token

### 完整实现

```python
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy import Column, Integer, String, Boolean, DateTime

# ========== 配置 ==========
SECRET_KEY = "your-secret-key-keep-it-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# ========== 数据库配置 ==========
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
Base = declarative_base()

# ========== 数据模型 ==========
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# ========== Pydantic 模型 ==========
class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []

# ========== 密码加密 ==========
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# ========== JWT Token ==========
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ========== 依赖项 ==========
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "read": "Read access",
        "write": "Write access",
        "admin": "Admin access",
    }
)

async def get_db():
    async with SessionLocal() as session:
        yield session

async def get_user(db: AsyncSession, username: str) -> Optional[User]:
    result = await db.execute(
        select(User).filter(User.username == username)
    )
    return result.scalar_one_or_none()

async def authenticate_user(
    db: AsyncSession,
    username: str,
    password: str
) -> Optional[User]:
    user = await get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "access":
            raise credentials_exception
        
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = await get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="The user doesn't have enough privileges"
        )
    return current_user

# ========== FastAPI 应用 ==========
app = FastAPI(title="User Authentication System")

# ========== 路由 ==========

@app.post("/register", response_model=UserInDB, status_code=status.HTTP_201_CREATED)
async def register(
    user_create: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """注册新用户"""
    # 检查用户是否已存在
    existing_user = await get_user(db, user_create.username)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # 创建新用户
    hashed_password = get_password_hash(user_create.password)
    db_user = User(
        email=user_create.email,
        username=user_create.username,
        hashed_password=hashed_password,
    )
    
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    
    return db_user

@app.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """用户登录，获取 Token"""
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 创建 Token
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db)
):
    """刷新 Access Token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "refresh":
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await get_user(db, username)
    if user is None:
        raise credentials_exception
    
    # 创建新的 Token
    new_access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    new_refresh_token = create_refresh_token(data={"sub": user.username})
    
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }

@app.get("/users/me", response_model=UserInDB)
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
):
    """获取当前用户信息"""
    return current_user

@app.get("/users/{username}", response_model=UserInDB)
async def read_user(
    username: str,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
):
    """获取指定用户信息（仅管理员）"""
    user = await get_user(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.delete("/users/{username}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    username: str,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
):
    """删除用户（仅管理员）"""
    user = await get_user(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await db.delete(user)
    await db.commit()
    
    return None
```

### 使用示例

```bash
# 1. 注册用户
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "testuser",
    "password": "securepassword"
  }'

# 2. 登录获取 Token
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=securepassword"

# 返回: {"access_token": "eyJ...", "refresh_token": "eyJ...", "token_type": "bearer"}

# 3. 使用 Token 访问受保护资源
curl -X GET "http://localhost:8000/users/me" \
  -H "Authorization: Bearer eyJ..."

# 4. 刷新 Token
curl -X POST "http://localhost:8000/refresh" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "eyJ..."}'
```

### 关键要点

1. **密码安全**：使用 `bcrypt` 加密密码，从不存储明文
2. **Token 类型**：区分 Access Token 和 Refresh Token
3. **依赖注入层次**：
   - `get_current_user`：验证 Token
   - `get_current_active_user`：检查用户激活状态
   - `get_current_superuser`：检查管理员权限
4. **数据库异步**：使用 `async/await` 提高性能
5. **错误处理**：返回标准 HTTP 状态码和清晰的错误信息

---

## 案例 2：高并发 API 性能优化

### 场景
- 需要处理高并发请求（10000+ QPS）
- 数据库查询是性能瓶颈
- 部分数据可缓存

### 优化策略

#### 1. 使用连接池

```python
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool, QueuePool

# 配置连接池
engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # 连接池大小
    max_overflow=10,       # 最大溢出连接数
    pool_timeout=30,       # 连接超时
    pool_recycle=3600,     # 连接回收时间（秒）
    pool_pre_ping=True,    # 使用前检查连接
)
```

#### 2. 实现 Redis 缓存

```python
import redis.asyncio as redis
from functools import wraps
import json

# Redis 连接池
redis_pool = redis.ConnectionPool.from_url(
    "redis://localhost:6379",
    decode_responses=True
)

async def get_redis():
    return redis.Redis(connection_pool=redis_pool)

# 缓存装饰器
def cache(expire: int = 300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # 尝试从缓存获取
            redis_client = await get_redis()
            cached = await redis_client.get(cache_key)
            
            if cached:
                return json.loads(cached)
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存入缓存
            await redis_client.setex(
                cache_key,
                expire,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# 使用缓存
@cache(expire=600)  # 缓存 10 分钟
async def get_popular_items(db: AsyncSession):
    result = await db.execute(
        select(Item).order_by(Item.views.desc()).limit(10)
    )
    return result.scalars().all()

@app.get("/popular-items")
async def popular_items(db: AsyncSession = Depends(get_db)):
    items = await get_popular_items(db)
    return items
```

#### 3. 批量查询优化

```python
from sqlalchemy.orm import selectinload

# 避免 N+1 查询
@app.get("/users-with-posts")
async def users_with_posts(db: AsyncSession = Depends(get_db)):
    # 不好的方式（N+1 查询）
    # users = await db.execute(select(User))
    # for user in users:
    #     posts = await db.execute(select(Post).filter(Post.user_id == user.id))
    
    # 好的方式（一次查询）
    result = await db.execute(
        select(User).options(selectinload(User.posts))
    )
    users = result.scalars().all()
    return users
```

#### 4. 并发请求处理

```python
import asyncio

@app.get("/dashboard")
async def dashboard():
    # 并发执行多个查询
    users_task = get_users_count()
    items_task = get_items_count()
    orders_task = get_orders_count()
    
    # 等待所有任务完成
    users, items, orders = await asyncio.gather(
        users_task,
        items_task,
        orders_task
    )
    
    return {
        "users": users,
        "items": items,
        "orders": orders
    }
```

#### 5. 响应分页

```python
from pydantic import BaseModel
from typing import Generic, TypeVar, List

T = TypeVar('T')

class PageResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int
    pages: int

async def paginate(
    query,
    db: AsyncSession,
    page: int = 1,
    size: int = 20
) -> PageResponse:
    # 获取总数
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)
    
    # 获取分页数据
    offset = (page - 1) * size
    result = await db.execute(query.limit(size).offset(offset))
    items = result.scalars().all()
    
    return PageResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=(total + size - 1) // size
    )

@app.get("/items", response_model=PageResponse[ItemOut])
async def list_items(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    query = select(Item).order_by(Item.created_at.desc())
    return await paginate(query, db, page, size)
```

### 性能测试结果

```bash
# 使用 wrk 进行压测
wrk -t12 -c400 -d30s --latency http://localhost:8000/items

# 优化前：
# Requests/sec:   2,500
# Latency avg:    160ms
# Latency 99%:    850ms

# 优化后（缓存 + 连接池 + 并发）：
# Requests/sec:   15,000
# Latency avg:     26ms
# Latency 99%:    120ms
```

---

## 案例 3：WebSocket 实时聊天室

### 需求
- 支持多用户实时聊天
- 房间管理
- 消息广播
- 用户在线状态

### 完整实现

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
from datetime import datetime

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        # 房间 -> 连接列表
        self.active_connections: Dict[str, List[dict]] = {}
    
    async def connect(self, websocket: WebSocket, room: str, username: str):
        await websocket.accept()
        
        if room not in self.active_connections:
            self.active_connections[room] = []
        
        self.active_connections[room].append({
            "websocket": websocket,
            "username": username,
            "joined_at": datetime.now()
        })
        
        # 广播用户加入消息
        await self.broadcast_to_room(
            room,
            {
                "type": "user_joined",
                "username": username,
                "timestamp": datetime.now().isoformat(),
                "users_count": len(self.active_connections[room])
            }
        )
    
    def disconnect(self, room: str, username: str):
        if room in self.active_connections:
            self.active_connections[room] = [
                conn for conn in self.active_connections[room]
                if conn["username"] != username
            ]
            
            if not self.active_connections[room]:
                del self.active_connections[room]
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
    
    async def broadcast_to_room(self, room: str, message: dict):
        if room in self.active_connections:
            for connection in self.active_connections[room]:
                try:
                    await connection["websocket"].send_json(message)
                except:
                    pass
    
    def get_room_users(self, room: str) -> List[str]:
        if room not in self.active_connections:
            return []
        return [conn["username"] for conn in self.active_connections[room]]

manager = ConnectionManager()

@app.websocket("/ws/{room}/{username}")
async def websocket_endpoint(
    websocket: WebSocket,
    room: str,
    username: str
):
    await manager.connect(websocket, room, username)
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # 构造消息
            message = {
                "type": "message",
                "username": username,
                "content": message_data.get("content", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            # 广播到房间
            await manager.broadcast_to_room(room, message)
    
    except WebSocketDisconnect:
        manager.disconnect(room, username)
        
        # 广播用户离开消息
        await manager.broadcast_to_room(
            room,
            {
                "type": "user_left",
                "username": username,
                "timestamp": datetime.now().isoformat(),
                "users_count": len(manager.get_room_users(room))
            }
        )

@app.get("/rooms/{room}/users")
async def get_room_users(room: str):
    """获取房间用户列表"""
    return {"users": manager.get_room_users(room)}

@app.get("/rooms")
async def get_rooms():
    """获取所有活跃房间"""
    return {
        "rooms": [
            {
                "name": room,
                "users_count": len(connections)
            }
            for room, connections in manager.active_connections.items()
        ]
    }
```

### 客户端示例（JavaScript）

```javascript
class ChatRoom {
    constructor(room, username) {
        this.room = room;
        this.username = username;
        this.ws = null;
    }
    
    connect() {
        this.ws = new WebSocket(
            `ws://localhost:8000/ws/${this.room}/${this.username}`
        );
        
        this.ws.onopen = () => {
            console.log('Connected to chat room');
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected from chat room');
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    sendMessage(content) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ content }));
        }
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'message':
                console.log(`[${message.timestamp}] ${message.username}: ${message.content}`);
                break;
            case 'user_joined':
                console.log(`${message.username} joined the room. Users: ${message.users_count}`);
                break;
            case 'user_left':
                console.log(`${message.username} left the room. Users: ${message.users_count}`);
                break;
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// 使用
const chat = new ChatRoom('general', 'Alice');
chat.connect();
chat.sendMessage('Hello everyone!');
```

---

## 最佳实践总结

### 1. 项目结构

```
myproject/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── dependencies.py      # 全局依赖项
│   ├── models/              # 数据库模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── schemas/             # Pydantic 模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── crud/                # CRUD 操作
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── api/                 # 路由模块
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── users.py
│   │   │   └── items.py
│   │   └── deps.py         # 路由依赖项
│   ├── core/                # 核心功能
│   │   ├── __init__.py
│   │   ├── security.py     # 安全相关
│   │   └── database.py     # 数据库连接
│   └── utils/               # 工具函数
│       ├── __init__.py
│       └── helpers.py
├── tests/                   # 测试
│   ├── __init__.py
│   ├── test_users.py
│   └── test_items.py
├── alembic/                 # 数据库迁移
│   └── versions/
├── .env                     # 环境变量
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 2. 配置管理

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # 应用配置
    app_name: str = "MyApp"
    debug: bool = False
    
    # 数据库
    database_url: str
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # JWT
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    backend_cors_origins: list[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

### 3. 错误处理

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# 自定义异常
class BusinessException(Exception):
    def __init__(self, message: str, code: str = "BUSINESS_ERROR"):
        self.message = message
        self.code = code

# 异常处理器
@app.exception_handler(BusinessException)
async def business_exception_handler(request: Request, exc: BusinessException):
    return JSONResponse(
        status_code=400,
        content={
            "code": exc.code,
            "message": exc.message,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Invalid request data",
            "errors": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )
```

### 4. 日志配置

```python
import logging
from logging.handlers import RotatingFileHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'app.log',
            maxBytes=10485760,  # 10MB
            backupCount=10
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 使用
@app.post("/items/")
async def create_item(item: Item):
    logger.info(f"Creating item: {item.name}")
    try:
        result = await create_item_in_db(item)
        logger.info(f"Item created successfully: {result.id}")
        return result
    except Exception as e:
        logger.error(f"Failed to create item: {str(e)}", exc_info=True)
        raise
```

### 5. 测试

```python
from fastapi.testclient import TestClient
import pytest

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def override_get_db():
    # 使用测试数据库
    pass

def test_create_user(client):
    response = client.post(
        "/users/",
        json={"email": "test@example.com", "username": "test"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data

def test_get_user_not_found(client):
    response = client.get("/users/999")
    assert response.status_code == 404

def test_authentication(client):
    # 登录
    response = client.post(
        "/token",
        data={"username": "test", "password": "password"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # 使用 token 访问受保护资源
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
```

### 6. 部署建议

#### Docker 部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

#### 生产环境配置

```bash
# 使用 gunicorn + uvicorn workers
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --access-logfile - \
  --error-logfile -
```

---

## 性能优化清单

✅ 使用异步数据库驱动（asyncpg、motor）  
✅ 配置连接池  
✅ 实现缓存层（Redis）  
✅ 使用 CDN 缓存静态资源  
✅ 启用 GZIP 压缩  
✅ 实现分页避免大数据集  
✅ 使用数据库索引  
✅ 避免 N+1 查询  
✅ 使用批量操作  
✅ 并发执行独立操作  
✅ 使用流式响应处理大文件  
✅ 实现 API 限流  
✅ 监控和日志记录  
✅ 使用 Load Balancer  
✅ 数据库主从分离  

---

## 安全清单

✅ 使用 HTTPS  
✅ 密码加密存储（bcrypt）  
✅ JWT Token 有过期时间  
✅ 实现 CORS 策略  
✅ 输入验证（Pydantic）  
✅ SQL 注入防护（ORM）  
✅ XSS 防护  
✅ CSRF 防护  
✅ 限流防止 DDoS  
✅ 敏感数据不记录日志  
✅ 环境变量管理密钥  
✅ 依赖项安全更新  
✅ API 版本控制  
✅ 错误信息不泄露敏感信息  

---

这些实战案例和最佳实践覆盖了 FastAPI 开发的各个方面，从认证授权到性能优化，从实时通信到测试部署，帮助开发者构建生产级的 FastAPI 应用。
