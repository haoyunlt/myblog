# Envoy使用示例和最佳实践

## 概述

本文档提供了Envoy的实际使用示例和生产环境的最佳实践，帮助开发者和运维人员更好地理解和应用Envoy代理。

## 基础配置示例

### 1. 简单的HTTP代理

```yaml
# envoy-simple-proxy.yaml
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address: 
        address: 0.0.0.0
        port_value: 10000
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          codec_type: AUTO
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  cluster: service_backend
          http_filters:
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router

  clusters:
  - name: service_backend
    connect_timeout: 0.25s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    load_assignment:
      cluster_name: service_backend
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: httpbin.org
                port_value: 80

admin:
  address:
    socket_address:
      address: 127.0.0.1
      port_value: 9901
```

**功能说明**：
- 在端口10000上监听HTTP请求
- 将所有请求转发到httpbin.org
- 提供管理接口在端口9901

**启动命令**：
```bash
envoy -c envoy-simple-proxy.yaml
```

### 2. HTTPS终止和重写

```yaml
# envoy-tls-termination.yaml  
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 10000
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          access_log:
          - name: envoy.access_loggers.stdout
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
          http_filters:
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["example.com"]
              routes:
              - match:
                  prefix: "/api/v1/"
                route:
                  prefix_rewrite: "/v1/"
                  cluster: api_service
              - match:
                  prefix: "/static/"
                route:
                  cluster: static_service
              - match:
                  prefix: "/"
                route:
                  cluster: web_service
      transport_socket:
        name: envoy.transport_sockets.tls
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext
          common_tls_context:
            tls_certificates:
            - certificate_chain:
                filename: "/etc/ssl/certs/example.com.crt"
              private_key:
                filename: "/etc/ssl/private/example.com.key"

  clusters:
  - name: api_service
    connect_timeout: 0.25s
    type: STRICT_DNS
    load_assignment:
      cluster_name: api_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: api.internal
                port_value: 8080

  - name: static_service
    connect_timeout: 0.25s
    type: STRICT_DNS
    load_assignment:
      cluster_name: static_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: cdn.internal
                port_value: 8080

  - name: web_service
    connect_timeout: 0.25s
    type: STRICT_DNS
    load_assignment:
      cluster_name: web_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: web.internal
                port_value: 8080

admin:
  address:
    socket_address:
      address: 127.0.0.1
      port_value: 9901
```

**功能说明**：
- TLS/SSL终止
- 基于路径的路由分发
- URL重写功能
- 访问日志记录

## 高级配置示例

### 3. 微服务网关配置

```yaml
# envoy-microservices-gateway.yaml
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 8080
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          access_log:
          - name: envoy.access_loggers.file
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.file.v3.FileAccessLog
              path: "/var/log/envoy/access.log"
              format: |
                [%START_TIME%] "%REQ(:METHOD)% %REQ(X-ENVOY-ORIGINAL-PATH?:PATH)% %PROTOCOL%"
                %RESPONSE_CODE% %RESPONSE_FLAGS% %BYTES_RECEIVED% %BYTES_SENT%
                %DURATION% %RESP(X-ENVOY-UPSTREAM-SERVICE-TIME)% "%REQ(X-FORWARDED-FOR)%"
                "%REQ(USER-AGENT)%" "%REQ(X-REQUEST-ID)%" "%REQ(:AUTHORITY)%" "%UPSTREAM_HOST%"
          http_filters:
          # CORS过滤器
          - name: envoy.filters.http.cors
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.cors.v3.Cors
          # JWT认证过滤器
          - name: envoy.filters.http.jwt_authn
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.jwt_authn.v3.JwtAuthentication
              providers:
                jwt_provider:
                  issuer: "https://auth.example.com"
                  audiences: ["api.example.com"]
                  remote_jwks:
                    http_uri:
                      uri: "https://auth.example.com/.well-known/jwks.json"
                      cluster: auth_service
                      timeout: 5s
                    cache_duration: 300s
              rules:
              - match:
                  prefix: "/api/"
                requires:
                  provider_name: "jwt_provider"
          # 限流过滤器
          - name: envoy.filters.http.local_ratelimit
            typed_config:
              "@type": type.googleapis.com/udpa.type.v1.TypedStruct
              type_url: type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit
              value:
                stat_prefix: http_local_rate_limiter
                token_bucket:
                  max_tokens: 10000
                  tokens_per_fill: 1000
                  fill_interval: 1s
                filter_enabled:
                  runtime_key: local_rate_limit_enabled
                  default_value:
                    numerator: 100
                    denominator: HUNDRED
                filter_enforced:
                  runtime_key: local_rate_limit_enforced
                  default_value:
                    numerator: 100
                    denominator: HUNDRED
          # 路由过滤器
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
          route_config:
            name: local_route
            virtual_hosts:
            - name: api_gateway
              domains: ["api.example.com", "localhost:8080"]
              routes:
              # 用户服务路由
              - match:
                  prefix: "/api/users"
                route:
                  cluster: user_service
                  timeout: 30s
                  retry_policy:
                    retry_on: "5xx,gateway-error,connect-failure,refused-stream"
                    num_retries: 3
                    per_try_timeout: 10s
              # 订单服务路由  
              - match:
                  prefix: "/api/orders"
                route:
                  cluster: order_service
                  timeout: 30s
                  retry_policy:
                    retry_on: "5xx,gateway-error,connect-failure,refused-stream"
                    num_retries: 3
              # 支付服务路由
              - match:
                  prefix: "/api/payments"
                route:
                  cluster: payment_service
                  timeout: 60s
              # 健康检查路由
              - match:
                  path: "/health"
                route:
                  cluster: health_check
              cors:
                allow_origin_string_match:
                - prefix: "https://"
                - exact: "http://localhost:3000"
                allow_methods: "GET, POST, PUT, DELETE, OPTIONS"
                allow_headers: "authorization,content-type,x-requested-with"
                max_age: "86400"

  clusters:
  # 用户服务集群
  - name: user_service
    connect_timeout: 0.25s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: user_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: user-service
                port_value: 8080
        - endpoint:
            address:
              socket_address:
                address: user-service-2
                port_value: 8080
    health_checks:
    - timeout: 1s
      interval: 5s
      unhealthy_threshold: 3
      healthy_threshold: 2
      http_health_check:
        path: "/health"
        expected_statuses:
        - start: 200
          end: 300

  # 订单服务集群
  - name: order_service
    connect_timeout: 0.25s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    lb_policy: LEAST_REQUEST
    load_assignment:
      cluster_name: order_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: order-service
                port_value: 8080

  # 支付服务集群
  - name: payment_service
    connect_timeout: 0.25s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: payment_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: payment-service
                port_value: 8080
    circuit_breakers:
      thresholds:
      - priority: DEFAULT
        max_connections: 100
        max_pending_requests: 100
        max_requests: 200
        max_retries: 3

  # 认证服务集群
  - name: auth_service
    connect_timeout: 5s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    load_assignment:
      cluster_name: auth_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: auth.example.com
                port_value: 443
    transport_socket:
      name: envoy.transport_sockets.tls
      typed_config:
        "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext
        sni: auth.example.com

  # 健康检查集群
  - name: health_check
    connect_timeout: 0.25s
    type: STATIC
    load_assignment:
      cluster_name: health_check
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 127.0.0.1
                port_value: 8080

admin:
  address:
    socket_address:
      address: 127.0.0.1
      port_value: 9901
```

**功能说明**：
- JWT认证保护API端点
- 基于路径的微服务路由
- 限流保护
- 健康检查和重试策略
- CORS支持
- 熔断器配置

### 4. 动态配置(xDS)示例

#### 控制平面配置

```yaml
# envoy-xds-bootstrap.yaml
node:
  id: "envoy-node-1"
  cluster: "production"
  metadata:
    version: "1.0"

dynamic_resources:
  lds_config:
    resource_api_version: V3
    api_config_source:
      api_type: GRPC
      transport_api_version: V3
      grpc_services:
      - envoy_grpc:
          cluster_name: xds_cluster
      set_node_on_first_message_only: true
      
  cds_config:
    resource_api_version: V3
    api_config_source:
      api_type: GRPC
      transport_api_version: V3
      grpc_services:
      - envoy_grpc:
          cluster_name: xds_cluster
      set_node_on_first_message_only: true

static_resources:
  clusters:
  - name: xds_cluster
    connect_timeout: 0.25s
    type: LOGICAL_DNS
    http2_protocol_options: {}
    dns_lookup_family: V4_ONLY
    load_assignment:
      cluster_name: xds_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: control-plane
                port_value: 18000

admin:
  address:
    socket_address:
      address: 127.0.0.1
      port_value: 9901
```

#### Python控制平面示例

```python
#!/usr/bin/env python3
"""
简单的xDS控制平面实现示例
"""

import grpc
from concurrent import futures
import time
import threading
import logging

# Envoy API imports
from envoy.service.discovery.v3 import discovery_service_pb2_grpc
from envoy.service.discovery.v3 import discovery_service_pb2
from envoy.config.listener.v3 import listener_pb2
from envoy.config.cluster.v3 import cluster_pb2
from envoy.config.core.v3 import config_core_pb2
from envoy.config.core.v3 import address_pb2
from envoy.extensions.filters.network.http_connection_manager.v3 import http_connection_manager_pb2
from envoy.extensions.filters.http.router.v3 import router_pb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ControlPlaneServer:
    """xDS控制平面服务器实现"""
    
    def __init__(self):
        self.version = "1"
        self.listeners_cache = {}
        self.clusters_cache = {}
        self.nonce = 0
        self._build_initial_config()
    
    def _build_initial_config(self):
        """构建初始配置"""
        
        # 创建监听器配置
        listener = listener_pb2.Listener()
        listener.name = "listener_0"
        listener.address.socket_address.address = "0.0.0.0"
        listener.address.socket_address.port_value = 10000
        
        # HTTP连接管理器配置
        hcm = http_connection_manager_pb2.HttpConnectionManager()
        hcm.stat_prefix = "ingress_http"
        hcm.codec_type = http_connection_manager_pb2.HttpConnectionManager.AUTO
        
        # 路由配置
        route_config = hcm.route_config
        route_config.name = "local_route"
        
        vhost = route_config.virtual_hosts.add()
        vhost.name = "service"
        vhost.domains[:] = ["*"]
        
        route = vhost.routes.add()
        route.match.prefix = "/"
        route.route.cluster = "service_backend"
        
        # HTTP过滤器
        router_filter = hcm.http_filters.add()
        router_filter.name = "envoy.filters.http.router"
        router_filter.typed_config.Pack(router_pb2.Router())
        
        # 添加过滤器链
        filter_chain = listener.filter_chains.add()
        filter_config = filter_chain.filters.add()
        filter_config.name = "envoy.filters.network.http_connection_manager"
        filter_config.typed_config.Pack(hcm)
        
        self.listeners_cache[listener.name] = listener
        
        # 创建集群配置
        cluster = cluster_pb2.Cluster()
        cluster.name = "service_backend"
        cluster.type = cluster_pb2.Cluster.LOGICAL_DNS
        cluster.connect_timeout.seconds = 0
        cluster.connect_timeout.nanos = 250000000
        cluster.dns_lookup_family = cluster_pb2.Cluster.V4_ONLY
        
        # 负载均衡配置
        endpoint = cluster.load_assignment.endpoints.add()
        lb_endpoint = endpoint.lb_endpoints.add()
        lb_endpoint.endpoint.address.socket_address.address = "httpbin.org"
        lb_endpoint.endpoint.address.socket_address.port_value = 80
        cluster.load_assignment.cluster_name = cluster.name
        
        self.clusters_cache[cluster.name] = cluster
    
    def get_next_nonce(self):
        """获取下一个nonce"""
        self.nonce += 1
        return str(self.nonce)


class AggregatedDiscoveryService(discovery_service_pb2_grpc.AggregatedDiscoveryServiceServicer):
    """聚合发现服务实现"""
    
    def __init__(self, control_plane):
        self.control_plane = control_plane
        
    def StreamAggregatedResources(self, request_iterator, context):
        """流式聚合资源发现"""
        logger.info("新的ADS连接建立")
        
        # 发送初始配置
        for request in request_iterator:
            logger.info(f"收到请求: {request.type_url}")
            
            if request.type_url == "type.googleapis.com/envoy.config.listener.v3.Listener":
                # 发送监听器配置
                response = discovery_service_pb2.DiscoveryResponse()
                response.version_info = self.control_plane.version
                response.type_url = request.type_url
                response.nonce = self.control_plane.get_next_nonce()
                
                for listener in self.control_plane.listeners_cache.values():
                    resource = response.resources.add()
                    resource.Pack(listener)
                
                logger.info(f"发送监听器配置，版本: {response.version_info}")
                yield response
                
            elif request.type_url == "type.googleapis.com/envoy.config.cluster.v3.Cluster":
                # 发送集群配置
                response = discovery_service_pb2.DiscoveryResponse()
                response.version_info = self.control_plane.version
                response.type_url = request.type_url
                response.nonce = self.control_plane.get_next_nonce()
                
                for cluster in self.control_plane.clusters_cache.values():
                    resource = response.resources.add()
                    resource.Pack(cluster)
                
                logger.info(f"发送集群配置，版本: {response.version_info}")
                yield response


def main():
    """主函数"""
    # 创建控制平面
    control_plane = ControlPlaneServer()
    
    # 创建gRPC服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # 注册服务
    ads_service = AggregatedDiscoveryService(control_plane)
    discovery_service_pb2_grpc.add_AggregatedDiscoveryServiceServicer_to_server(
        ads_service, server
    )
    
    # 监听端口
    listen_addr = '[::]:18000'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"控制平面服务器启动，监听地址: {listen_addr}")
    server.start()
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("正在关闭服务器...")
        server.stop(0)


if __name__ == "__main__":
    main()
```

**功能说明**：
- 动态监听器发现(LDS)
- 动态集群发现(CDS)  
- gRPC流式API
- 版本控制和增量更新

## Docker化部署

### 5. Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  envoy:
    image: envoyproxy/envoy:v1.28-latest
    ports:
      - "8080:8080"
      - "9901:9901"
    volumes:
      - ./envoy.yaml:/etc/envoy/envoy.yaml:ro
      - ./certs:/etc/ssl/certs:ro
    command: ["envoy", "-c", "/etc/envoy/envoy.yaml", "--log-level", "info"]
    depends_on:
      - web-service
      - api-service

  web-service:
    image: nginx:alpine
    ports:
      - "8081:80"
    volumes:
      - ./web:/usr/share/nginx/html:ro

  api-service:
    image: httpbin/httpbin
    ports:
      - "8082:80"

  control-plane:
    build: 
      context: .
      dockerfile: Dockerfile.control-plane
    ports:
      - "18000:18000"
    environment:
      - LOG_LEVEL=info
```

### Dockerfile示例

```dockerfile
# Dockerfile.control-plane
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY control_plane.py .
COPY protos/ ./protos/

# 暴露端口
EXPOSE 18000

# 启动命令
CMD ["python", "control_plane.py"]
```

## 最佳实践

### 1. 性能优化

#### 工作线程配置

```yaml
# 基于CPU核数配置工作线程
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 8080
    # 启用端口重用以提高性能
    reuse_port: true
    # 设置连接平衡配置
    connection_balance_config:
      exact_balance: {}
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          # 设置连接缓冲区限制
          per_connection_buffer_limit_bytes: 32768
          # 启用HTTP/2
          http2_protocol_options:
            max_concurrent_streams: 100
          # 配置访问日志
          access_log:
          - name: envoy.access_loggers.file
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.file.v3.FileAccessLog
              path: "/var/log/envoy/access.log"
```

#### 连接池优化

```yaml
# 集群连接池配置
clusters:
- name: service_backend
  connect_timeout: 0.25s
  type: LOGICAL_DNS
  # 连接池设置
  circuit_breakers:
    thresholds:
    - priority: DEFAULT
      max_connections: 1024
      max_pending_requests: 1024
      max_requests: 1024
      max_retries: 3
    - priority: HIGH
      max_connections: 2048
      max_pending_requests: 2048
      max_requests: 2048
      max_retries: 5
  # HTTP连接池配置
  typed_extension_protocol_options:
    envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
      "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
      explicit_http_config:
        http2_protocol_options:
          max_concurrent_streams: 100
          initial_stream_window_size: 65536
          initial_connection_window_size: 1048576
```

### 2. 安全配置

#### TLS最佳实践

```yaml
# HTTPS监听器配置
listeners:
- name: https_listener
  address:
    socket_address:
      address: 0.0.0.0
      port_value: 443
  filter_chains:
  - filters:
    - name: envoy.filters.network.http_connection_manager
      # ... HTTP连接管理器配置 ...
    transport_socket:
      name: envoy.transport_sockets.tls
      typed_config:
        "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext
        common_tls_context:
          # TLS证书配置
          tls_certificates:
          - certificate_chain:
              filename: "/etc/ssl/certs/server.crt"
            private_key:
              filename: "/etc/ssl/private/server.key"
          # TLS参数
          tls_params:
            tls_minimum_protocol_version: TLSv1_2
            tls_maximum_protocol_version: TLSv1_3
            cipher_suites:
            - "ECDHE-ECDSA-AES256-GCM-SHA384"
            - "ECDHE-RSA-AES256-GCM-SHA384" 
            - "ECDHE-ECDSA-CHACHA20-POLY1305"
            - "ECDHE-RSA-CHACHA20-POLY1305"
            - "ECDHE-ECDSA-AES128-GCM-SHA256"
            - "ECDHE-RSA-AES128-GCM-SHA256"
          # ALPN协议协商
          alpn_protocols: 
          - "h2"
          - "http/1.1"
```

#### 安全头配置

```yaml
# 安全HTTP头过滤器
http_filters:
- name: envoy.filters.http.local_response
  typed_config:
    "@type": type.googleapis.com/udpa.type.v1.TypedStruct
    type_url: type.googleapis.com/envoy.extensions.filters.http.local_response.v3.LocalResponse
    value:
      mappers:
      - filter:
          status_code_filter:
            comparison:
              op: EQ
              value:
                default_value: 200
                runtime_key: response_code
        headers_to_add:
        - header:
            key: "X-Content-Type-Options"
            value: "nosniff"
        - header:
            key: "X-Frame-Options" 
            value: "DENY"
        - header:
            key: "X-XSS-Protection"
            value: "1; mode=block"
        - header:
            key: "Strict-Transport-Security"
            value: "max-age=63072000; includeSubDomains; preload"
        - header:
            key: "Referrer-Policy"
            value: "strict-origin-when-cross-origin"
```

### 3. 可观测性配置

#### 统计指标配置

```yaml
stats_config:
  # 统计标签配置
  stats_tags:
  - tag_name: "method"
    regex: "^http\\.(.+?)\\.downstream_rq_(.+?)$"
    fixed_value: "\\1"
  - tag_name: "status"
    regex: "^http\\.(.+?)\\.downstream_rq_(\\d{3})$"  
    fixed_value: "\\2"
  # 直方图桶配置
  histogram_bucket_settings:
  - match:
      name: "http.downstream_rq_time"
    buckets: [0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
  - match:  
      name: "http.upstream_rq_time"
    buckets: [0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

# Prometheus统计接收器
stats_sinks:
- name: envoy.stat_sinks.statsd
  typed_config:
    "@type": type.googleapis.com/envoy.config.metrics.v3.StatsdSink
    address:
      socket_address:
        address: 127.0.0.1
        port_value: 8125
    prefix: "envoy"
```

#### OpenTelemetry追踪配置

```yaml
tracing:
  http:
    name: envoy.tracers.opentelemetry
    typed_config:
      "@type": type.googleapis.com/envoy.config.trace.v3.OpenTelemetryConfig
      grpc_service:
        envoy_grpc:
          cluster_name: jaeger
        timeout: 0.25s
      service_name: "envoy-gateway"
      
# 追踪集群
clusters:
- name: jaeger
  type: LOGICAL_DNS
  dns_lookup_family: V4_ONLY
  load_assignment:
    cluster_name: jaeger
    endpoints:
    - lb_endpoints:
      - endpoint:
          address:
            socket_address:
              address: jaeger-collector
              port_value: 14250
```

### 4. 容器化部署最佳实践

#### Kubernetes配置

```yaml
# envoy-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: envoy-config
  namespace: default
data:
  envoy.yaml: |
    # Envoy配置内容...

---
# envoy-deployment.yaml  
apiVersion: apps/v1
kind: Deployment
metadata:
  name: envoy-gateway
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: envoy-gateway
  template:
    metadata:
      labels:
        app: envoy-gateway
    spec:
      containers:
      - name: envoy
        image: envoyproxy/envoy:v1.28-latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9901
          name: admin
        volumeMounts:
        - name: config-volume
          mountPath: /etc/envoy
          readOnly: true
        # 资源限制
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m" 
            memory: "512Mi"
        # 健康检查
        livenessProbe:
          httpGet:
            path: /ready
            port: 9901
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9901
          initialDelaySeconds: 5
          periodSeconds: 5
        # 优雅关闭
        lifecycle:
          preStop:
            exec:
              command: 
              - /bin/sh
              - -c
              - "wget -qO- http://localhost:9901/healthcheck/fail && sleep 15"
      volumes:
      - name: config-volume
        configMap:
          name: envoy-config

---
# envoy-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: envoy-gateway-service
  namespace: default
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  - port: 9901
    targetPort: 9901
    protocol: TCP
    name: admin
  selector:
    app: envoy-gateway
```

### 5. 监控和告警

#### Prometheus监控配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'envoy'
  static_configs:
  - targets: ['envoy:9901']
  metrics_path: '/stats/prometheus'
  scrape_interval: 5s

- job_name: 'envoy-admin'
  static_configs:
  - targets: ['envoy:9901']
  metrics_path: '/server_info'
  scrape_interval: 30s
```

#### Grafana仪表板示例

```json
{
  "dashboard": {
    "title": "Envoy Proxy Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(envoy_http_downstream_rq_total[1m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(envoy_http_downstream_rq_time_bucket[1m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(envoy_http_downstream_rq_time_bucket[1m]))", 
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Upstream Health",
        "type": "stat",
        "targets": [
          {
            "expr": "envoy_cluster_upstream_healthy",
            "legendFormat": "{{cluster_name}}"
          }
        ]
      }
    ]
  }
}
```

## 故障排查指南

### 1. 常见问题和解决方案

#### 配置验证

```bash
# 验证Envoy配置
envoy --mode validate -c envoy.yaml

# 检查配置语法
envoy -c envoy.yaml --log-level debug --log-format '[%Y-%m-%d %T.%e][%t][%l][%n] %v'
```

#### 日志分析

```bash
# 启用详细日志
envoy -c envoy.yaml --log-level trace --component-log-level upstream:debug,connection:debug

# 实时查看访问日志
tail -f /var/log/envoy/access.log | jq '.'

# 分析错误模式
grep -E "(5\d\d|ERROR|WARN)" /var/log/envoy/envoy.log | tail -20
```

#### 管理接口调试

```bash
# 检查集群状态
curl http://localhost:9901/clusters

# 查看监听器状态  
curl http://localhost:9901/listeners

# 检查统计信息
curl http://localhost:9901/stats | grep http

# 查看配置转储
curl http://localhost:9901/config_dump

# 检查证书信息
curl http://localhost:9901/certs
```

### 2. 性能调优指南

#### 系统级优化

```bash
# 增加文件描述符限制
ulimit -n 65536

# 调整TCP参数
echo 'net.core.somaxconn = 8192' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 8192' >> /etc/sysctl.conf
sysctl -p
```

#### Envoy配置优化

```yaml
# 性能优化配置
static_resources:
  listeners:
  - name: optimized_listener
    # 启用端口重用
    reuse_port: true
    # 设置TCP backlog
    tcp_backlog_size: 8192
    # 优化连接平衡
    connection_balance_config:
      exact_balance: {}
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          # 减少缓冲区
          per_connection_buffer_limit_bytes: 1048576
          # 优化HTTP/2设置
          http2_protocol_options:
            max_concurrent_streams: 1000
            initial_stream_window_size: 268435456
            initial_connection_window_size: 268435456
          # 禁用不必要的功能
          normalize_path: false
          strip_any_host_port: false
```

## 总结

本文档提供了Envoy的实用配置示例和最佳实践，涵盖了：

1. **基础到高级的配置示例**: 从简单代理到微服务网关
2. **动态配置**: xDS控制平面实现
3. **容器化部署**: Docker和Kubernetes配置
4. **安全配置**: TLS、认证、授权最佳实践
5. **可观测性**: 监控、日志、追踪配置
6. **性能优化**: 系统和应用级别的调优
7. **故障排查**: 常见问题的诊断和解决方案

这些实践经验可以帮助您在生产环境中更好地部署和运维Envoy代理。
