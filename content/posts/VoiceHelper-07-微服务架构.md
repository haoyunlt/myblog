---
title: "VoiceHelper-07-微服务架构"
date: 2025-01-08T11:10:00+08:00
draft: false
tags:
  - 源码剖析
  - 架构分析
  - 源码分析
  - 微服务
  - 分布式系统
  - Kubernetes
categories:
  - VoiceHelper
  - 系统架构
description: "源码剖析 - VoiceHelper 微服务架构设计与实现"
author: "源码分析"
weight: 507
ShowToc: true
TocOpen: true
---

# VoiceHelper-07-微服务架构

## 概览

VoiceHelper 采用微服务架构，将核心功能拆分为 4 个独立服务，每个服务职责单一、可独立部署和扩展。

### 微服务清单

| 服务名称 | 端口 | 职责 | 技术栈 |
|---|---|---|---|
| auth-service | 8081 | 用户认证、JWT、SSO | Go + PostgreSQL |
| document-service | 8082 | 文档管理、元数据 | Go + PostgreSQL + MinIO |
| session-service | 8083 | 会话管理、上下文 | Go + PostgreSQL + Redis |
| notification-service | 8084 | 通知推送、WebSocket | Go + Redis |

### 架构图

```mermaid
flowchart TB
    subgraph Client["客户端层"]
        WEB[Web前端]
        MOBILE[移动端]
    end

    subgraph Gateway["网关层"]
        GATEWAY[API网关<br/>:8080]
    end

    subgraph Services["微服务层"]
        AUTH[认证服务<br/>:8081]
        DOC[文档服务<br/>:8082]
        SESSION[会话服务<br/>:8083]
        NOTIFY[通知服务<br/>:8084]
    end

    subgraph Storage["存储层"]
        PG[(PostgreSQL)]
        REDIS[(Redis)]
        MINIO[(MinIO)]
    end

    subgraph Discovery["服务发现"]
        CONSUL[Consul]
    end

    WEB --> GATEWAY
    MOBILE --> GATEWAY
    
    GATEWAY --> AUTH
    GATEWAY --> DOC
    GATEWAY --> SESSION
    GATEWAY --> NOTIFY
    
    AUTH --> PG
    AUTH --> REDIS
    
    DOC --> PG
    DOC --> MINIO
    
    SESSION --> PG
    SESSION --> REDIS
    
    NOTIFY --> REDIS
    
    AUTH -.注册.-> CONSUL
    DOC -.注册.-> CONSUL
    SESSION -.注册.-> CONSUL
    NOTIFY -.注册.-> CONSUL

    style Gateway fill:#e3f2fd
    style Services fill:#f3e5f5
    style Storage fill:#fff3e0
    style Discovery fill:#e8f5e9
```

---

## 1. 认证服务（auth-service）

### 1.1 服务职责

- **用户注册**：用户名/密码注册、邮箱验证
- **用户登录**：JWT Token 签发
- **Token 刷新**：延长会话有效期
- **SSO 集成**：微信登录、企业 SSO
- **Token 验证**：为其他服务提供 Token 校验
- **用户资料**：获取用户信息

### 1.2 核心 API

#### 1.2.1 用户注册

**接口**：`POST /api/v1/auth/register`

**请求体**：
```go
type RegisterRequest struct {
    Username string `json:"username" binding:"required,min=3,max=32"`
    Password string `json:"password" binding:"required,min=8"`
    Email    string `json:"email" binding:"required,email"`
}
```

**响应体**：
```go
type RegisterResponse struct {
    UserID   string    `json:"user_id"`
    Username string    `json:"username"`
    Email    string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
}
```

**核心代码**：
```go
func (h *AuthHandler) Register(c *gin.Context) {
    var req RegisterRequest
    // 1. 解析请求
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    
    // 2. 密码哈希
    hashedPassword, err := bcrypt.GenerateFromPassword(
        []byte(req.Password), 
        bcrypt.DefaultCost,
    )
    
    // 3. 创建用户
    user := &model.User{
        Username:     req.Username,
        Email:        req.Email,
        PasswordHash: string(hashedPassword),
        Status:       "active",
    }
    
    // 4. 存储到数据库
    if err := h.authService.CreateUser(c.Request.Context(), user); err != nil {
        c.JSON(500, gin.H{"error": "创建用户失败"})
        return
    }
    
    // 5. 返回结果
    c.JSON(201, RegisterResponse{
        UserID:    user.ID,
        Username:  user.Username,
        Email:     user.Email,
        CreatedAt: user.CreatedAt,
    })
}
```

#### 1.2.2 用户登录

**接口**：`POST /api/v1/auth/login`

**请求体**：
```go
type LoginRequest struct {
    Username string `json:"username" binding:"required"`
    Password string `json:"password" binding:"required"`
}
```

**响应体**：
```go
type LoginResponse struct {
    AccessToken  string `json:"access_token"`
    RefreshToken string `json:"refresh_token"`
    ExpiresIn    int64  `json:"expires_in"`  // 秒
}
```

**核心代码**：
```go
func (h *AuthHandler) Login(c *gin.Context) {
    var req LoginRequest
    // 1. 解析请求
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    
    // 2. 查找用户
    user, err := h.authService.GetUserByUsername(c.Request.Context(), req.Username)
    if err != nil {
        c.JSON(401, gin.H{"error": "用户名或密码错误"})
        return
    }
    
    // 3. 验证密码
    if err := bcrypt.CompareHashAndPassword(
        []byte(user.PasswordHash),
        []byte(req.Password),
    ); err != nil {
        c.JSON(401, gin.H{"error": "用户名或密码错误"})
        return
    }
    
    // 4. 生成 JWT Token
    accessToken, err := h.authService.GenerateAccessToken(user)
    refreshToken, err := h.authService.GenerateRefreshToken(user)
    
    // 5. 返回结果
    c.JSON(200, LoginResponse{
        AccessToken:  accessToken,
        RefreshToken: refreshToken,
        ExpiresIn:    3600,  // 1小时
    })
}
```

#### 1.2.3 Token 验证

**接口**：`POST /api/v1/auth/validate`

**请求头**：
```
Authorization: Bearer <token>
```

**响应体**：
```go
type ValidateResponse struct {
    Valid  bool   `json:"valid"`
    UserID string `json:"user_id"`
    Error  string `json:"error,omitempty"`
}
```

**核心代码**：
```go
func (h *AuthHandler) ValidateToken(c *gin.Context) {
    // 1. 提取 Token
    tokenString := c.GetHeader("Authorization")
    tokenString = strings.TrimPrefix(tokenString, "Bearer ")
    
    // 2. 解析 Token
    token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
        return []byte(h.jwtSecret), nil
    })
    
    if err != nil || !token.Valid {
        c.JSON(200, ValidateResponse{
            Valid: false,
            Error: "Token 无效",
        })
        return
    }
    
    // 3. 提取 Claims
    claims := token.Claims.(jwt.MapClaims)
    userID := claims["user_id"].(string)
    
    // 4. 返回结果
    c.JSON(200, ValidateResponse{
        Valid:  true,
        UserID: userID,
    })
}
```

### 1.3 服务注册（Consul）

```go
func main() {
    // 服务配置
    serviceName := "auth-service"
    serviceID := fmt.Sprintf("%s-%d", serviceName, os.Getpid())
    
    // 注册到 Consul
    consulRegistry, _ := discovery.NewConsulRegistry(
        "localhost:8500",
        &discovery.RegistryConfig{
            ServiceName: serviceName,
            ServiceID:   serviceID,
            Host:        "localhost",
            Port:        8081,
            Tags:        []string{"v0.3", "auth", "microservice"},
            HealthCheck: &api.AgentServiceCheck{
                HTTP:     "http://localhost:8081/health",
                Interval: "10s",
                Timeout:  "3s",
            },
        },
    )
    
    consulRegistry.Register()
}
```

---

## 2. 文档服务（document-service）

### 2.1 服务职责

- **文档上传**：支持 PDF、Word、Markdown
- **元数据管理**：标题、标签、分类
- **文件存储**：MinIO 对象存储
- **文档检索**：按标题、标签搜索

### 2.2 核心 API

#### 2.2.1 上传文档

**接口**：`POST /api/v1/documents`

**请求**：`multipart/form-data`
- `file`: 文件内容
- `title`: 文档标题
- `tags`: 标签（逗号分隔）

**响应体**：
```go
type UploadResponse struct {
    DocumentID string    `json:"document_id"`
    FileName   string    `json:"file_name"`
    FileSize   int64     `json:"file_size"`
    UploadedAt time.Time `json:"uploaded_at"`
}
```

**核心代码**：
```go
func (h *DocumentHandler) Upload(c *gin.Context) {
    // 1. 解析文件
    file, header, err := c.Request.FormFile("file")
    if err != nil {
        c.JSON(400, gin.H{"error": "文件解析失败"})
        return
    }
    defer file.Close()
    
    // 2. 生成文档 ID
    documentID := uuid.New().String()
    
    // 3. 上传到 MinIO
    objectName := fmt.Sprintf("documents/%s/%s", documentID, header.Filename)
    _, err = h.minioClient.PutObject(
        c.Request.Context(),
        "voicehelper",
        objectName,
        file,
        header.Size,
        minio.PutObjectOptions{ContentType: header.Header.Get("Content-Type")},
    )
    
    // 4. 保存元数据到数据库
    doc := &model.Document{
        ID:       documentID,
        FileName: header.Filename,
        FileSize: header.Size,
        FilePath: objectName,
        Title:    c.PostForm("title"),
        Tags:     strings.Split(c.PostForm("tags"), ","),
        UserID:   c.GetString("user_id"),
    }
    h.docService.Create(c.Request.Context(), doc)
    
    // 5. 返回结果
    c.JSON(201, UploadResponse{
        DocumentID: documentID,
        FileName:   header.Filename,
        FileSize:   header.Size,
        UploadedAt: time.Now(),
    })
}
```

#### 2.2.2 获取文档

**接口**：`GET /api/v1/documents/:id`

**响应体**：
```go
type DocumentResponse struct {
    ID         string    `json:"id"`
    FileName   string    `json:"file_name"`
    FileSize   int64     `json:"file_size"`
    Title      string    `json:"title"`
    Tags       []string  `json:"tags"`
    CreatedAt  time.Time `json:"created_at"`
    DownloadURL string   `json:"download_url"`
}
```

**核心代码**：
```go
func (h *DocumentHandler) Get(c *gin.Context) {
    documentID := c.Param("id")
    
    // 1. 查询数据库
    doc, err := h.docService.GetByID(c.Request.Context(), documentID)
    if err != nil {
        c.JSON(404, gin.H{"error": "文档不存在"})
        return
    }
    
    // 2. 生成预签名 URL（有效期 1 小时）
    presignedURL, _ := h.minioClient.PresignedGetObject(
        c.Request.Context(),
        "voicehelper",
        doc.FilePath,
        time.Hour,
        nil,
    )
    
    // 3. 返回结果
    c.JSON(200, DocumentResponse{
        ID:          doc.ID,
        FileName:    doc.FileName,
        FileSize:    doc.FileSize,
        Title:       doc.Title,
        Tags:        doc.Tags,
        CreatedAt:   doc.CreatedAt,
        DownloadURL: presignedURL.String(),
    })
}
```

---

## 3. 会话服务（session-service）

### 3.1 服务职责

- **会话创建**：创建新的对话会话
- **消息存储**：持久化聊天消息
- **上下文管理**：维护对话上下文（Redis 缓存）
- **会话检索**：按用户、时间范围查询

### 3.2 核心 API

#### 3.2.1 创建会话

**接口**：`POST /api/v1/sessions`

**请求体**：
```go
type CreateSessionRequest struct {
    Title string `json:"title"`
}
```

**响应体**：
```go
type SessionResponse struct {
    SessionID string    `json:"session_id"`
    Title     string    `json:"title"`
    CreatedAt time.Time `json:"created_at"`
}
```

**核心代码**：
```go
func (h *SessionHandler) Create(c *gin.Context) {
    var req CreateSessionRequest
    c.ShouldBindJSON(&req)
    
    // 1. 创建会话
    session := &model.Session{
        ID:        uuid.New().String(),
        UserID:    c.GetString("user_id"),
        Title:     req.Title,
        Status:    "active",
        CreatedAt: time.Now(),
    }
    
    // 2. 存储到数据库
    h.sessionService.Create(c.Request.Context(), session)
    
    // 3. 初始化 Redis 上下文
    contextKey := fmt.Sprintf("session:context:%s", session.ID)
    h.redisClient.Set(c.Request.Context(), contextKey, "[]", 24*time.Hour)
    
    // 4. 返回结果
    c.JSON(201, SessionResponse{
        SessionID: session.ID,
        Title:     session.Title,
        CreatedAt: session.CreatedAt,
    })
}
```

#### 3.2.2 添加消息

**接口**：`POST /api/v1/sessions/:id/messages`

**请求体**：
```go
type AddMessageRequest struct {
    Role    string `json:"role"`    // user / assistant
    Content string `json:"content"`
}
```

**核心代码**：
```go
func (h *SessionHandler) AddMessage(c *gin.Context) {
    sessionID := c.Param("id")
    var req AddMessageRequest
    c.ShouldBindJSON(&req)
    
    // 1. 创建消息
    message := &model.Message{
        ID:        uuid.New().String(),
        SessionID: sessionID,
        Role:      req.Role,
        Content:   req.Content,
        CreatedAt: time.Now(),
    }
    
    // 2. 存储到数据库
    h.messageService.Create(c.Request.Context(), message)
    
    // 3. 更新 Redis 上下文
    contextKey := fmt.Sprintf("session:context:%s", sessionID)
    contextJSON, _ := h.redisClient.Get(c.Request.Context(), contextKey).Result()
    
    var context []map[string]string
    json.Unmarshal([]byte(contextJSON), &context)
    
    context = append(context, map[string]string{
        "role":    req.Role,
        "content": req.Content,
    })
    
    // 保留最近 10 条消息
    if len(context) > 10 {
        context = context[len(context)-10:]
    }
    
    updatedJSON, _ := json.Marshal(context)
    h.redisClient.Set(c.Request.Context(), contextKey, updatedJSON, 24*time.Hour)
    
    // 4. 返回结果
    c.JSON(201, message)
}
```

#### 3.2.3 获取上下文

**接口**：`GET /api/v1/sessions/:id/context`

**响应体**：
```go
type ContextResponse struct {
    SessionID string              `json:"session_id"`
    Messages  []map[string]string `json:"messages"`
}
```

**核心代码**：
```go
func (h *SessionHandler) GetContext(c *gin.Context) {
    sessionID := c.Param("id")
    
    // 1. 从 Redis 读取上下文
    contextKey := fmt.Sprintf("session:context:%s", sessionID)
    contextJSON, err := h.redisClient.Get(c.Request.Context(), contextKey).Result()
    
    if err != nil {
        // 如果 Redis 中没有，从数据库加载
        messages, _ := h.messageService.GetBySessionID(
            c.Request.Context(), 
            sessionID,
            10,  // 最近 10 条
        )
        
        var context []map[string]string
        for _, msg := range messages {
            context = append(context, map[string]string{
                "role":    msg.Role,
                "content": msg.Content,
            })
        }
        
        contextJSON, _ = json.Marshal(context)
        h.redisClient.Set(c.Request.Context(), contextKey, contextJSON, 24*time.Hour)
    }
    
    // 2. 返回结果
    var messages []map[string]string
    json.Unmarshal([]byte(contextJSON), &messages)
    
    c.JSON(200, ContextResponse{
        SessionID: sessionID,
        Messages:  messages,
    })
}
```

---

## 4. 通知服务（notification-service）

### 4.1 服务职责

- **WebSocket 连接管理**：维护客户端连接
- **事件推送**：实时推送通知
- **订阅管理**：用户订阅特定主题

### 4.2 核心 API

#### 4.2.1 WebSocket 连接

**接口**：`GET /api/v1/notifications/ws`

**连接参数**：`?token=<jwt_token>`

**核心代码**：
```go
func (h *NotificationHandler) HandleWebSocket(c *gin.Context) {
    // 1. 升级为 WebSocket
    conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
    if err != nil {
        return
    }
    defer conn.Close()
    
    // 2. 验证 Token
    token := c.Query("token")
    claims, err := h.authService.ValidateToken(token)
    if err != nil {
        conn.WriteMessage(websocket.TextMessage, []byte(`{"error":"认证失败"}`))
        return
    }
    userID := claims["user_id"].(string)
    
    // 3. 注册连接
    clientID := uuid.New().String()
    h.connectionManager.Register(clientID, userID, conn)
    
    // 4. 监听消息
    for {
        _, message, err := conn.ReadMessage()
        if err != nil {
            h.connectionManager.Unregister(clientID)
            break
        }
        
        // 处理订阅/取消订阅
        var msg map[string]string
        json.Unmarshal(message, &msg)
        
        switch msg["type"] {
        case "subscribe":
            h.connectionManager.Subscribe(clientID, msg["topic"])
        case "unsubscribe":
            h.connectionManager.Unsubscribe(clientID, msg["topic"])
        }
    }
}
```

#### 4.2.2 推送通知

**接口**：`POST /api/v1/notifications/send`

**请求体**：
```go
type SendNotificationRequest struct {
    UserID  string                 `json:"user_id"`
    Topic   string                 `json:"topic"`
    Type    string                 `json:"type"`
    Title   string                 `json:"title"`
    Content string                 `json:"content"`
    Data    map[string]interface{} `json:"data"`
}
```

**核心代码**：
```go
func (h *NotificationHandler) Send(c *gin.Context) {
    var req SendNotificationRequest
    c.ShouldBindJSON(&req)
    
    // 1. 构建通知消息
    notification := map[string]interface{}{
        "id":      uuid.New().String(),
        "type":    req.Type,
        "title":   req.Title,
        "content": req.Content,
        "data":    req.Data,
        "time":    time.Now().Unix(),
    }
    
    notificationJSON, _ := json.Marshal(notification)
    
    // 2. 查找用户的所有连接
    connections := h.connectionManager.GetUserConnections(req.UserID)
    
    // 3. 推送给所有订阅了该主题的连接
    for _, conn := range connections {
        if conn.IsSubscribed(req.Topic) {
            conn.WriteMessage(websocket.TextMessage, notificationJSON)
        }
    }
    
    // 4. 如果用户不在线，存储到 Redis
    if len(connections) == 0 {
        key := fmt.Sprintf("notifications:pending:%s", req.UserID)
        h.redisClient.RPush(c.Request.Context(), key, notificationJSON)
        h.redisClient.Expire(c.Request.Context(), key, 7*24*time.Hour)  // 7天过期
    }
    
    c.JSON(200, gin.H{"status": "sent"})
}
```

---

## 微服务间通信

### 时序图：完整流程

```mermaid
sequenceDiagram
    autonumber
    participant C as 客户端
    participant G as API网关
    participant A as 认证服务
    participant S as 会话服务
    participant N as 通知服务

    C->>G: POST /api/v1/auth/login
    G->>A: 转发登录请求
    A->>A: 验证用户名密码
    A->>A: 生成 JWT Token
    A-->>G: 返回 Token
    G-->>C: 返回 Token
    
    C->>G: POST /api/v1/sessions<br/>(携带 Token)
    G->>A: 验证 Token
    A-->>G: Token 有效
    G->>S: 创建会话
    S->>S: 生成会话 ID
    S-->>G: 返回会话信息
    G-->>C: 返回会话 ID
    
    G->>N: 推送通知<br/>"会话已创建"
    N->>N: 查找用户连接
    N->>C: WebSocket 推送
```

---

## 部署架构

### Docker Compose 配置

```yaml
version: '3.8'

services:
  auth-service:
    build: ./services/auth-service
    ports:
      - "8081:8081"
    environment:
      - JWT_SECRET=${JWT_SECRET}
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/auth_db
      - CONSUL_ADDR=consul:8500
    depends_on:
      - postgres
      - consul

  document-service:
    build: ./services/document-service
    ports:
      - "8082:8082"
    environment:
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/doc_db
    depends_on:
      - postgres
      - minio

  session-service:
    build: ./services/session-service
    ports:
      - "8083:8083"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/session_db
      - REDIS_URL=redis:6379
    depends_on:
      - postgres
      - redis

  notification-service:
    build: ./services/notification-service
    ports:
      - "8084:8084"
    environment:
      - REDIS_URL=redis:6379
    depends_on:
      - redis

  consul:
    image: consul:1.15
    ports:
      - "8500:8500"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  minio_data:
```

---

**文档版本**：v1.0  
**最后更新**：2025-01-08  
**维护者**：VoiceHelper 团队

