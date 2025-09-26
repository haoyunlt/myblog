---
title: "代码样式测试 - 黑色背景主题"
date: 2024-01-04T10:00:00+08:00
draft: false
tags: ["测试", "代码样式", "主题"]
categories: ["开发"]
ShowToc: true
TocOpen: true
---

# 代码样式测试

测试Go、Python、JavaScript、Shell脚本的黑色背景代码块效果。

## Go语言代码

```go
package main

import (
    "fmt"
    "net/http"
    "log"
    "time"
)

// User 用户结构体
type User struct {
    ID       int    `json:"id"`
    Name     string `json:"name"`
    Email    string `json:"email"`
    Created  time.Time `json:"created"`
}

// NewUser 创建新用户
func NewUser(name, email string) *User {
    return &User{
        Name:    name,
        Email:   email,
        Created: time.Now(),
    }
}

func main() {
    // 创建HTTP服务器
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        user := NewUser("张三", "zhangsan@example.com")
        fmt.Fprintf(w, "Hello, %s! Email: %s", user.Name, user.Email)
    })
    
    fmt.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Python代码

```python
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class User:
    """用户数据类"""
    id: int
    name: str
    email: str
    created: datetime

class UserService:
    """用户服务类"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_users(self) -> List[User]:
        """获取用户列表"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        async with self.session.get(f"{self.base_url}/users") as response:
            data = await response.json()
            return [User(**item) for item in data]
    
    def fibonacci(self, n: int) -> int:
        """计算斐波那契数列"""
        if n <= 1:
            return n
        return self.fibonacci(n-1) + self.fibonacci(n-2)

# 使用示例
async def main():
    async with UserService("https://api.example.com") as service:
        users = await service.get_users()
        for user in users:
            print(f"User: {user.name} ({user.email})")
        
        # 计算斐波那契数列
        for i in range(10):
            result = service.fibonacci(i)
            print(f"fibonacci({i}) = {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## JavaScript代码

```javascript
// 导入模块
import axios from 'axios';
import { EventEmitter } from 'events';

/**
 * 用户管理类
 */
class UserManager extends EventEmitter {
    constructor(apiUrl) {
        super();
        this.apiUrl = apiUrl;
        this.users = new Map();
        this.cache = new WeakMap();
    }

    /**
     * 异步获取用户数据
     * @param {number} userId - 用户ID
     * @returns {Promise<Object>} 用户对象
     */
    async fetchUser(userId) {
        try {
            const response = await axios.get(`${this.apiUrl}/users/${userId}`);
            const user = response.data;
            
            // 缓存用户数据
            this.users.set(userId, user);
            this.emit('userFetched', user);
            
            return user;
        } catch (error) {
            console.error(`Error fetching user ${userId}:`, error.message);
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * 批量处理用户
     * @param {number[]} userIds - 用户ID数组
     */
    async processUsers(userIds) {
        const promises = userIds.map(id => this.fetchUser(id));
        const users = await Promise.allSettled(promises);
        
        const successful = users
            .filter(result => result.status === 'fulfilled')
            .map(result => result.value);
        
        const failed = users
            .filter(result => result.status === 'rejected')
            .map(result => result.reason);
        
        console.log(`Successfully processed: ${successful.length} users`);
        console.log(`Failed: ${failed.length} users`);
        
        return { successful, failed };
    }

    // 使用箭头函数和解构赋值
    getUserStats = () => {
        const { size } = this.users;
        const activeUsers = [...this.users.values()]
            .filter(user => user.active)
            .length;
        
        return {
            total: size,
            active: activeUsers,
            inactive: size - activeUsers
        };
    };
}

// 使用示例
const userManager = new UserManager('https://api.example.com');

userManager.on('userFetched', (user) => {
    console.log(`User fetched: ${user.name}`);
});

userManager.on('error', (error) => {
    console.error('UserManager error:', error);
});

// 异步执行
(async () => {
    try {
        const userIds = [1, 2, 3, 4, 5];
        const result = await userManager.processUsers(userIds);
        
        const stats = userManager.getUserStats();
        console.log('User statistics:', stats);
    } catch (error) {
        console.error('Main execution error:', error);
    }
})();
```

## Shell脚本

```bash
#!/bin/bash

# 设置脚本选项
set -euo pipefail

# 全局变量
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="${SCRIPT_DIR}/deploy.log"
readonly CONFIG_FILE="${SCRIPT_DIR}/config.env"

# 颜色定义
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# 日志函数
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# 检查依赖
check_dependencies() {
    local deps=("docker" "docker-compose" "git" "curl")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "ERROR" "Missing dependencies: ${missing_deps[*]}"
        return 1
    fi
    
    log "INFO" "All dependencies are satisfied"
    return 0
}

# 加载配置
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        # shellcheck source=/dev/null
        source "$CONFIG_FILE"
        log "INFO" "Configuration loaded from $CONFIG_FILE"
    else
        log "WARN" "Configuration file not found: $CONFIG_FILE"
        # 设置默认值
        export APP_NAME="${APP_NAME:-myapp}"
        export APP_VERSION="${APP_VERSION:-latest}"
        export DEPLOY_ENV="${DEPLOY_ENV:-development}"
    fi
}

# 部署应用
deploy_application() {
    local app_name="$1"
    local version="$2"
    local environment="$3"
    
    log "INFO" "Starting deployment of $app_name:$version to $environment"
    
    # 拉取最新代码
    if git pull origin main; then
        log "INFO" "Code updated successfully"
    else
        log "ERROR" "Failed to update code"
        return 1
    fi
    
    # 构建Docker镜像
    if docker build -t "$app_name:$version" .; then
        log "INFO" "Docker image built successfully"
    else
        log "ERROR" "Failed to build Docker image"
        return 1
    fi
    
    # 部署服务
    if docker-compose up -d; then
        log "INFO" "Application deployed successfully"
    else
        log "ERROR" "Failed to deploy application"
        return 1
    fi
    
    # 健康检查
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:8080/health &> /dev/null; then
            log "INFO" "Health check passed (attempt $attempt)"
            break
        else
            log "WARN" "Health check failed (attempt $attempt/$max_attempts)"
            sleep 10
            ((attempt++))
        fi
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        log "ERROR" "Health check failed after $max_attempts attempts"
        return 1
    fi
    
    log "INFO" "Deployment completed successfully"
    return 0
}

# 清理函数
cleanup() {
    log "INFO" "Performing cleanup..."
    docker system prune -f
    log "INFO" "Cleanup completed"
}

# 主函数
main() {
    log "INFO" "Starting deployment script"
    
    # 设置清理陷阱
    trap cleanup EXIT
    
    # 检查依赖
    if ! check_dependencies; then
        exit 1
    fi
    
    # 加载配置
    load_config
    
    # 部署应用
    if deploy_application "$APP_NAME" "$APP_VERSION" "$DEPLOY_ENV"; then
        log "INFO" "Deployment script completed successfully"
        exit 0
    else
        log "ERROR" "Deployment script failed"
        exit 1
    fi
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

## 其他语言对比

### CSS代码（保持浅色背景）

```css
/* 响应式布局 */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
}

.card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
}
```

### HTML代码（保持浅色背景）

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="代码样式测试页面">
    <title>代码样式测试</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header class="header">
        <nav class="navigation">
            <h1>我的博客</h1>
            <ul class="nav-list">
                <li><a href="#home">首页</a></li>
                <li><a href="#about">关于</a></li>
                <li><a href="#contact">联系</a></li>
            </ul>
        </nav>
    </header>
    
    <main class="main-content">
        <article class="article">
            <h2>代码样式展示</h2>
            <p>这里展示了不同编程语言的代码高亮效果。</p>
        </article>
    </main>
    
    <footer class="footer">
        <p>&copy; 2024 我的博客. 保留所有权利.</p>
    </footer>
</body>
</html>
```

## 样式特点

### 黑色背景语言
- **Go、Python、JavaScript、Shell** 使用深色主题
- 背景色：`#1e1e1e` (VS Code深色主题)
- 文字色：`#d4d4d4` (浅灰色)
- 边框色：`#404040` (深灰色)

### 字体选择
- **主要字体**：JetBrains Mono (支持连字符)
- **备选字体**：Fira Code, SF Mono, Monaco, Cascadia Code
- **字体大小**：0.9rem (与正文协调)
- **行高**：1.6 (提高可读性)
- **字重**：400 (正常)

### 语法高亮颜色
- **关键字**：`#569cd6` (蓝色)
- **字符串**：`#ce9178` (橙色)  
- **注释**：`#6a9955` (绿色)
- **函数名**：`#dcdcaa` (黄色)
- **类名**：`#4ec9b0` (青色)
- **数字**：`#b5cea8` (浅绿色)

这样的配色方案提供了优秀的对比度和可读性，特别适合长时间阅读代码。