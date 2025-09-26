---
title: "Nginx 实战经验和最佳实践"
date: 2024-12-19T10:00:00+08:00
draft: false
tags: ['Nginx', 'Web服务器', '反向代理', 'C语言']
categories: ["nginx", "技术分析"]
description: "深入分析 Nginx 实战经验和最佳实践 的技术实现和架构设计"
weight: 520
slug: "nginx-best_practices"
---

# Nginx 实战经验和最佳实践

## 1. 概述

本文档汇总了Nginx开发和运维中的实战经验、性能优化技巧、常见问题解决方案以及最佳实践，帮助开发者和运维人员更好地使用和扩展Nginx。

## 2. 架构设计最佳实践

### 2.1 进程模型优化

#### Worker进程数量配置

```nginx
# 推荐配置：worker进程数等于CPU核心数
worker_processes auto;

# 或者手动指定
worker_processes 4;

# 绑定worker进程到特定CPU核心
worker_cpu_affinity auto;
# 或手动指定
worker_cpu_affinity 0001 0010 0100 1000;
```

**最佳实践**:
- 通常设置为CPU核心数
- 对于I/O密集型应用可以适当增加
- 避免设置过多导致上下文切换开销

#### 连接数优化

```nginx
# 每个worker进程的最大连接数
worker_connections 65535;

# 系统级别文件描述符限制
worker_rlimit_nofile 65535;

# 启用高效的事件模型
events {
    use epoll;  # Linux
    # use kqueue;  # BSD/macOS
    
    # 允许一个worker同时接受多个连接
    multi_accept on;
    
    # 禁用accept锁（单worker时）
    accept_mutex off;
}
```

### 2.2 内存管理优化

#### 缓冲区配置

```nginx
# 客户端请求体缓冲区
client_body_buffer_size 128k;
client_max_body_size 10m;
client_body_timeout 60s;

# 客户端请求头缓冲区
client_header_buffer_size 1k;
large_client_header_buffers 4 4k;
client_header_timeout 60s;

# 输出缓冲区
output_buffers 1 32k;
postpone_output 1460;

# 代理缓冲区
proxy_buffering on;
proxy_buffer_size 4k;
proxy_buffers 8 4k;
proxy_busy_buffers_size 8k;
```

#### 内存池优化

```c
// 自定义模块中的内存池使用
static ngx_int_t
my_module_handler(ngx_http_request_t *r)
{
    ngx_pool_t *pool;
    
    // 使用请求池进行内存分配
    pool = r->pool;
    
    // 分配内存，请求结束时自动释放
    char *buffer = ngx_palloc(pool, 1024);
    if (buffer == NULL) {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }
    
    // 注册清理函数
    ngx_pool_cleanup_t *cln = ngx_pool_cleanup_add(pool, 0);
    if (cln == NULL) {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }
    
    cln->handler = my_cleanup_handler;
    cln->data = cleanup_data;
    
    return NGX_OK;
}
```

### 2.3 I/O优化

#### 文件I/O优化

```nginx
# 启用sendfile
sendfile on;

# 启用TCP_NOPUSH（Linux）或TCP_CORK
tcp_nopush on;

# 启用TCP_NODELAY
tcp_nodelay on;

# 文件AIO（Linux）
aio on;
aio_write on;

# 直接I/O阈值
directio 512k;

# 文件缓存
open_file_cache max=1000 inactive=20s;
open_file_cache_valid 30s;
open_file_cache_min_uses 2;
open_file_cache_errors on;
```

#### 网络I/O优化

```nginx
# Keep-Alive连接
keepalive_timeout 65;
keepalive_requests 100;

# 发送超时
send_timeout 60s;

# 接收超时
client_body_timeout 60s;
client_header_timeout 60s;

# 重置连接而不是正常关闭
reset_timedout_connection on;
```

## 3. 性能调优实践

### 3.1 系统级优化

#### 内核参数调优

```bash
# /etc/sysctl.conf

# 网络参数
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_tw_recycle = 0

# 文件描述符
fs.file-max = 6815744
fs.nr_open = 6815744

# 内存管理
vm.swappiness = 1
vm.overcommit_memory = 1
```

#### 系统限制调优

```bash
# /etc/security/limits.conf
nginx soft nofile 65535
nginx hard nofile 65535
nginx soft nproc 65535
nginx hard nproc 65535

# systemd服务限制
# /etc/systemd/system/nginx.service.d/override.conf
[Service]
LimitNOFILE=65535
LimitNPROC=65535
```

### 3.2 Nginx配置优化

#### 核心配置优化

```nginx
# nginx.conf

# 用户和组
user nginx;

# 进程优先级
worker_priority -5;

# 错误日志级别（生产环境）
error_log /var/log/nginx/error.log warn;

# 访问日志优化
access_log /var/log/nginx/access.log main buffer=64k flush=5s;

# 或者关闭访问日志（高性能场景）
# access_log off;

# 服务器令牌
server_tokens off;

# 字符集
charset utf-8;

# MIME类型
include /etc/nginx/mime.types;
default_type application/octet-stream;
```

#### HTTP配置优化

```nginx
http {
    # 隐藏版本信息
    server_tokens off;
    
    # 压缩配置
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;
    
    # 静态文件缓存
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
    }
    
    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
}
```

### 3.3 缓存策略优化

#### 代理缓存配置

```nginx
# 缓存路径配置
proxy_cache_path /var/cache/nginx/proxy
                 levels=1:2
                 keys_zone=proxy_cache:10m
                 max_size=1g
                 inactive=60m
                 use_temp_path=off;

server {
    location / {
        proxy_pass http://backend;
        
        # 缓存配置
        proxy_cache proxy_cache;
        proxy_cache_key $scheme$proxy_host$request_uri;
        proxy_cache_valid 200 302 10m;
        proxy_cache_valid 404 1m;
        
        # 缓存控制
        proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
        proxy_cache_background_update on;
        proxy_cache_lock on;
        
        # 缓存头
        add_header X-Cache-Status $upstream_cache_status;
    }
}
```

#### FastCGI缓存配置

```nginx
# FastCGI缓存路径
fastcgi_cache_path /var/cache/nginx/fastcgi
                   levels=1:2
                   keys_zone=fastcgi_cache:10m
                   max_size=1g
                   inactive=60m;

server {
    location ~ \.php$ {
        fastcgi_pass 127.0.0.1:9000;
        fastcgi_index index.php;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        include fastcgi_params;
        
        # FastCGI缓存
        fastcgi_cache fastcgi_cache;
        fastcgi_cache_key $scheme$request_method$host$request_uri;
        fastcgi_cache_valid 200 10m;
        fastcgi_cache_valid 404 1m;
        
        # 缓存控制
        fastcgi_cache_use_stale error timeout updating http_500;
        fastcgi_cache_background_update on;
        fastcgi_cache_lock on;
        
        # 缓存绕过
        fastcgi_cache_bypass $cookie_nocache $arg_nocache;
        fastcgi_no_cache $cookie_nocache $arg_nocache;
    }
}
```

## 4. 负载均衡最佳实践

### 4.1 上游服务器配置

```nginx
# 基本负载均衡
upstream backend {
    server 192.168.1.10:8080 weight=3;
    server 192.168.1.11:8080 weight=2;
    server 192.168.1.12:8080 weight=1;
    server 192.168.1.13:8080 backup;
    
    # 负载均衡方法
    # least_conn;  # 最少连接
    # ip_hash;     # IP哈希
    # hash $request_uri;  # 自定义哈希
    
    # 健康检查
    server 192.168.1.10:8080 max_fails=3 fail_timeout=30s;
    
    # 连接保持
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

server {
    location / {
        proxy_pass http://backend;
        
        # 代理头设置
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 连接和超时
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # HTTP版本
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

### 4.2 高可用配置

```nginx
# 主备配置
upstream primary {
    server 192.168.1.10:8080;
    server 192.168.1.11:8080 backup;
}

# 故障转移配置
upstream failover {
    server 192.168.1.10:8080 max_fails=2 fail_timeout=10s;
    server 192.168.1.11:8080 max_fails=2 fail_timeout=10s;
    server 192.168.1.12:8080 backup;
}

# 会话保持
upstream session_sticky {
    ip_hash;
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
    server 192.168.1.12:8080;
}
```

## 5. 安全最佳实践

### 5.1 基础安全配置

```nginx
# 隐藏服务器信息
server_tokens off;
more_set_headers "Server: MyServer";

# 限制请求方法
if ($request_method !~ ^(GET|HEAD|POST)$ ) {
    return 405;
}

# 防止点击劫持
add_header X-Frame-Options DENY;

# 防止MIME类型嗅探
add_header X-Content-Type-Options nosniff;

# XSS保护
add_header X-XSS-Protection "1; mode=block";

# HSTS
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload";

# CSP
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'";
```

### 5.2 访问控制

```nginx
# IP白名单
location /admin {
    allow 192.168.1.0/24;
    allow 10.0.0.0/8;
    deny all;
    
    # 其他配置...
}

# 基于用户代理的限制
if ($http_user_agent ~* (bot|crawler|spider)) {
    return 403;
}

# 防止SQL注入
location ~ \.(sql|bak|inc|old)$ {
    deny all;
}

# 防止访问敏感文件
location ~ /\. {
    deny all;
}

location ~ ~$ {
    deny all;
}
```

### 5.3 速率限制

```nginx
# 定义限制区域
limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_conn_zone $binary_remote_addr zone=conn:10m;

server {
    # 应用限制
    location /login {
        limit_req zone=login burst=3 nodelay;
        limit_conn conn 1;
        
        # 其他配置...
    }
    
    location /api {
        limit_req zone=api burst=20 nodelay;
        limit_conn conn 10;
        
        # 其他配置...
    }
}
```

## 6. SSL/TLS最佳实践

### 6.1 SSL配置优化

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;
    
    # 证书配置
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # SSL协议版本
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # 密码套件
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # DH参数
    ssl_dhparam /path/to/dhparam.pem;
    
    # ECDH曲线
    ssl_ecdh_curve secp384r1;
    
    # 会话缓存
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    
    # OCSP装订
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /path/to/chain.crt;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;
}
```

### 6.2 HTTP/2配置

```nginx
server {
    listen 443 ssl http2;
    
    # HTTP/2推送
    location / {
        http2_push /css/style.css;
        http2_push /js/script.js;
        
        # 其他配置...
    }
    
    # HTTP/2服务器推送预加载
    location ~ \.(css|js)$ {
        add_header Link "</css/style.css>; rel=preload; as=style, </js/script.js>; rel=preload; as=script";
        expires 1y;
    }
}
```

## 7. 监控和日志最佳实践

### 7.1 日志配置

```nginx
# 自定义日志格式
log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                '$status $body_bytes_sent "$http_referer" '
                '"$http_user_agent" "$http_x_forwarded_for" '
                'rt=$request_time uct="$upstream_connect_time" '
                'uht="$upstream_header_time" urt="$upstream_response_time"';

# 访问日志
access_log /var/log/nginx/access.log main buffer=64k flush=5s;

# 错误日志
error_log /var/log/nginx/error.log warn;

# 特定位置的日志
location /api {
    access_log /var/log/nginx/api.log main;
    error_log /var/log/nginx/api_error.log;
    
    # 其他配置...
}
```

### 7.2 状态监控

```nginx
# 启用状态模块
location /nginx_status {
    stub_status on;
    allow 127.0.0.1;
    allow 192.168.1.0/24;
    deny all;
}

# 自定义状态页面
location /status {
    access_log off;
    return 200 "Nginx is running\n";
    add_header Content-Type text/plain;
}
```

### 7.3 健康检查

```nginx
# 应用健康检查
location /health {
    access_log off;
    return 200 "healthy\n";
    add_header Content-Type text/plain;
}

# 上游健康检查
upstream backend {
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
    
    # 使用第三方模块进行健康检查
    # health_check interval=5s fails=3 passes=2;
}
```

## 8. 模块开发最佳实践

### 8.1 模块结构设计

```c
// 模块配置结构
typedef struct {
    ngx_flag_t  enable;
    ngx_str_t   name;
    ngx_uint_t  timeout;
    ngx_array_t *values;
} ngx_my_module_conf_t;

// 模块上下文
typedef struct {
    ngx_str_t   key;
    ngx_str_t   value;
    time_t      expire;
} ngx_my_module_ctx_t;

// 模块定义
static ngx_command_t ngx_my_module_commands[] = {
    {
        ngx_string("my_enable"),
        NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_FLAG,
        ngx_conf_set_flag_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_my_module_conf_t, enable),
        NULL
    },
    ngx_null_command
};

static ngx_http_module_t ngx_my_module_ctx = {
    NULL,                          /* preconfiguration */
    ngx_my_module_init,            /* postconfiguration */
    NULL,                          /* create main configuration */
    NULL,                          /* init main configuration */
    NULL,                          /* create server configuration */
    NULL,                          /* merge server configuration */
    ngx_my_module_create_conf,     /* create location configuration */
    ngx_my_module_merge_conf       /* merge location configuration */
};

ngx_module_t ngx_my_module = {
    NGX_MODULE_V1,
    &ngx_my_module_ctx,            /* module context */
    ngx_my_module_commands,        /* module directives */
    NGX_HTTP_MODULE,               /* module type */
    NULL,                          /* init master */
    NULL,                          /* init module */
    NULL,                          /* init process */
    NULL,                          /* init thread */
    NULL,                          /* exit thread */
    NULL,                          /* exit process */
    NULL,                          /* exit master */
    NGX_MODULE_V1_PADDING
};
```

### 8.2 错误处理

```c
static ngx_int_t
ngx_my_module_handler(ngx_http_request_t *r)
{
    ngx_int_t                  rc;
    ngx_my_module_conf_t      *conf;
    ngx_my_module_ctx_t       *ctx;
    
    // 获取配置
    conf = ngx_http_get_module_loc_conf(r, ngx_my_module);
    if (!conf->enable) {
        return NGX_DECLINED;
    }
    
    // 获取或创建上下文
    ctx = ngx_http_get_module_ctx(r, ngx_my_module);
    if (ctx == NULL) {
        ctx = ngx_pcalloc(r->pool, sizeof(ngx_my_module_ctx_t));
        if (ctx == NULL) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "failed to allocate context");
            return NGX_HTTP_INTERNAL_SERVER_ERROR;
        }
        ngx_http_set_ctx(r, ctx, ngx_my_module);
    }
    
    // 业务逻辑处理
    rc = ngx_my_module_process(r, ctx, conf);
    if (rc != NGX_OK) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "processing failed: %i", rc);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }
    
    return NGX_OK;
}
```

### 8.3 内存管理

```c
static ngx_int_t
ngx_my_module_process(ngx_http_request_t *r, ngx_my_module_ctx_t *ctx,
    ngx_my_module_conf_t *conf)
{
    ngx_pool_cleanup_t  *cln;
    ngx_buf_t           *b;
    ngx_chain_t         *out;
    
    // 注册清理函数
    cln = ngx_pool_cleanup_add(r->pool, 0);
    if (cln == NULL) {
        return NGX_ERROR;
    }
    
    cln->handler = ngx_my_module_cleanup;
    cln->data = ctx;
    
    // 分配缓冲区
    b = ngx_create_temp_buf(r->pool, 1024);
    if (b == NULL) {
        return NGX_ERROR;
    }
    
    // 填充数据
    b->last = ngx_snprintf(b->pos, 1024, "Hello from my module!");
    b->last_buf = 1;
    b->last_in_chain = 1;
    
    // 创建输出链
    out = ngx_alloc_chain_link(r->pool);
    if (out == NULL) {
        return NGX_ERROR;
    }
    
    out->buf = b;
    out->next = NULL;
    
    // 发送响应
    r->headers_out.status = NGX_HTTP_OK;
    r->headers_out.content_length_n = b->last - b->pos;
    
    ngx_http_send_header(r);
    return ngx_http_output_filter(r, out);
}

static void
ngx_my_module_cleanup(void *data)
{
    ngx_my_module_ctx_t *ctx = data;
    
    // 清理资源
    if (ctx->key.data) {
        // 清理操作
    }
}
```

## 9. 运维最佳实践

### 9.1 部署策略

#### 蓝绿部署

```bash
#!/bin/bash
# 蓝绿部署脚本

BLUE_CONFIG="/etc/nginx/sites-available/blue"
GREEN_CONFIG="/etc/nginx/sites-available/green"
CURRENT_LINK="/etc/nginx/sites-enabled/current"

# 检查新版本
if [ "$1" = "green" ]; then
    NEW_CONFIG=$GREEN_CONFIG
    OLD_CONFIG=$BLUE_CONFIG
else
    NEW_CONFIG=$BLUE_CONFIG
    OLD_CONFIG=$GREEN_CONFIG
fi

# 测试新配置
nginx -t -c $NEW_CONFIG
if [ $? -ne 0 ]; then
    echo "Configuration test failed"
    exit 1
fi

# 切换配置
ln -sf $NEW_CONFIG $CURRENT_LINK

# 重载配置
nginx -s reload

echo "Switched to $1 environment"
```

#### 滚动更新

```bash
#!/bin/bash
# 滚动更新脚本

SERVERS=("server1" "server2" "server3")
NEW_VERSION="1.2.0"

for server in "${SERVERS[@]}"; do
    echo "Updating $server..."
    
    # 从负载均衡器移除
    curl -X POST "http://lb/api/servers/$server/disable"
    
    # 等待连接排空
    sleep 30
    
    # 更新服务器
    ssh $server "
        systemctl stop nginx
        yum update nginx-$NEW_VERSION
        systemctl start nginx
    "
    
    # 健康检查
    for i in {1..10}; do
        if curl -f "http://$server/health"; then
            break
        fi
        sleep 5
    done
    
    # 重新加入负载均衡器
    curl -X POST "http://lb/api/servers/$server/enable"
    
    echo "$server updated successfully"
done
```

### 9.2 备份和恢复

```bash
#!/bin/bash
# 配置备份脚本

BACKUP_DIR="/backup/nginx"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR/$DATE

# 备份配置文件
cp -r /etc/nginx $BACKUP_DIR/$DATE/

# 备份SSL证书
cp -r /etc/ssl/nginx $BACKUP_DIR/$DATE/

# 备份自定义模块
cp -r /usr/local/nginx/modules $BACKUP_DIR/$DATE/

# 创建备份信息
cat > $BACKUP_DIR/$DATE/info.txt << EOF
Backup Date: $(date)
Nginx Version: $(nginx -v 2>&1)
System: $(uname -a)
EOF

# 压缩备份
tar -czf $BACKUP_DIR/nginx_backup_$DATE.tar.gz -C $BACKUP_DIR $DATE

# 清理旧备份（保留30天）
find $BACKUP_DIR -name "nginx_backup_*.tar.gz" -mtime +30 -delete

echo "Backup completed: nginx_backup_$DATE.tar.gz"
```

### 9.3 监控脚本

```bash
#!/bin/bash
# Nginx监控脚本

NGINX_STATUS_URL="http://localhost/nginx_status"
LOG_FILE="/var/log/nginx_monitor.log"
ALERT_EMAIL="admin@example.com"

# 检查Nginx进程
check_process() {
    if ! pgrep nginx > /dev/null; then
        echo "$(date): Nginx process not running" >> $LOG_FILE
        systemctl start nginx
        echo "Nginx restarted" | mail -s "Nginx Alert" $ALERT_EMAIL
    fi
}

# 检查响应时间
check_response_time() {
    RESPONSE_TIME=$(curl -o /dev/null -s -w "%{time_total}" http://localhost/)
    if (( $(echo "$RESPONSE_TIME > 5.0" | bc -l) )); then
        echo "$(date): High response time: $RESPONSE_TIME" >> $LOG_FILE
        echo "High response time: $RESPONSE_TIME" | mail -s "Nginx Performance Alert" $ALERT_EMAIL
    fi
}

# 检查连接数
check_connections() {
    ACTIVE_CONNECTIONS=$(curl -s $NGINX_STATUS_URL | grep "Active connections" | awk '{print $3}')
    if [ "$ACTIVE_CONNECTIONS" -gt 1000 ]; then
        echo "$(date): High connection count: $ACTIVE_CONNECTIONS" >> $LOG_FILE
        echo "High connection count: $ACTIVE_CONNECTIONS" | mail -s "Nginx Connection Alert" $ALERT_EMAIL
    fi
}

# 检查错误日志
check_error_log() {
    ERROR_COUNT=$(tail -100 /var/log/nginx/error.log | grep "$(date '+%Y/%m/%d %H:')" | wc -l)
    if [ "$ERROR_COUNT" -gt 10 ]; then
        echo "$(date): High error count: $ERROR_COUNT" >> $LOG_FILE
        tail -20 /var/log/nginx/error.log | mail -s "Nginx Error Alert" $ALERT_EMAIL
    fi
}

# 执行检查
check_process
check_response_time
check_connections
check_error_log
```

## 10. 故障排除指南

### 10.1 常见问题诊断

#### 性能问题排查

```bash
# 1. 检查系统资源
top
iostat -x 1
netstat -i

# 2. 检查Nginx状态
curl http://localhost/nginx_status

# 3. 分析访问日志
tail -f /var/log/nginx/access.log | grep "HTTP/1.1\" 5"

# 4. 分析错误日志
tail -f /var/log/nginx/error.log

# 5. 检查网络连接
ss -tuln | grep :80
ss -s

# 6. 检查文件描述符使用
lsof -p $(pgrep nginx)
cat /proc/$(pgrep nginx | head -1)/limits
```

#### 配置问题排查

```bash
# 测试配置语法
nginx -t

# 检查配置文件包含关系
nginx -T

# 查看编译参数
nginx -V

# 检查模块加载
nginx -V 2>&1 | grep -o with-[a-z_]*

# 调试配置
nginx -t -c /etc/nginx/nginx.conf -g "error_log stderr debug;"
```

### 10.2 性能调优检查清单

```markdown
## 系统级检查
- [ ] 内核参数优化
- [ ] 文件描述符限制
- [ ] 网络参数调优
- [ ] 磁盘I/O优化

## Nginx配置检查
- [ ] Worker进程数配置
- [ ] 连接数配置
- [ ] 缓冲区大小
- [ ] 超时设置
- [ ] 压缩配置
- [ ] 缓存配置

## 应用层检查
- [ ] 静态文件优化
- [ ] 代理配置优化
- [ ] SSL/TLS配置
- [ ] HTTP/2启用
- [ ] 负载均衡配置

## 监控和日志
- [ ] 访问日志配置
- [ ] 错误日志级别
- [ ] 状态监控启用
- [ ] 性能指标收集
```

## 11. 安全检查清单

### 11.1 基础安全配置

```markdown
## 服务器安全
- [ ] 隐藏版本信息
- [ ] 限制请求方法
- [ ] 设置安全头
- [ ] 配置HTTPS
- [ ] 启用HSTS

## 访问控制
- [ ] IP白名单/黑名单
- [ ] 速率限制
- [ ] 请求大小限制
- [ ] 用户代理过滤

## 文件安全
- [ ] 隐藏敏感文件
- [ ] 禁止执行上传文件
- [ ] 设置正确的文件权限
- [ ] 定期更新SSL证书

## 日志和监控
- [ ] 启用访问日志
- [ ] 监控异常访问
- [ ] 设置告警机制
- [ ] 定期安全审计
```

### 11.2 SSL/TLS安全配置

```nginx
# 安全的SSL配置模板
server {
    listen 443 ssl http2;
    server_name example.com;
    
    # 证书配置
    ssl_certificate /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;
    
    # 安全协议
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # 安全特性
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # 安全头
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
}
```

## 12. 总结

### 12.1 核心原则

1. **性能优先**: 针对具体场景进行优化
2. **安全第一**: 多层防护，纵深防御
3. **可维护性**: 配置清晰，文档完善
4. **监控完善**: 全面监控，及时告警
5. **持续改进**: 定期评估，持续优化

### 12.2 关键要点

- **合理配置**: 根据实际负载调整参数
- **定期维护**: 及时更新，定期备份
- **监控告警**: 建立完善的监控体系
- **安全防护**: 多重安全措施并举
- **性能调优**: 持续优化，追求极致

### 12.3 发展趋势

- **云原生**: 容器化部署，微服务架构
- **自动化**: 自动化运维，智能化管理
- **可观测性**: 全链路监控，智能分析
- **安全增强**: 零信任架构，AI安全防护

通过遵循这些最佳实践，可以构建高性能、高可用、安全可靠的Nginx服务，满足现代Web应用的需求。
