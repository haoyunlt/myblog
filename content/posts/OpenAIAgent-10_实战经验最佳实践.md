---
title: "OpenAI Agents SDK 实战经验与最佳实践"
date: 2025-09-28T00:47:17+08:00
draft: false
tags: ['源码分析', '技术文档', '最佳实践']
categories: ['技术分析']
description: "OpenAI Agents SDK 实战经验与最佳实践的深入技术分析文档"
keywords: ['源码分析', '技术文档', '最佳实践']
author: "技术分析师"
weight: 1
---

## 10.1 架构设计最佳实践

### 10.1.1 代理设计原则

**单一职责原则**
每个代理应该专注于一个特定的领域或任务，避免创建"万能代理"。

```python
# ❌ 不好的做法：万能代理
mega_agent = Agent(
    name="Super Agent",
    instructions="You can do everything: answer questions, write code, analyze data, send emails, book flights...",
    tools=[email_tool, code_tool, data_tool, booking_tool, weather_tool, ...]
)

# ✅ 好的做法：专门化代理
customer_service_agent = Agent(
    name="Customer Service",
    instructions="You are a helpful customer service representative. Focus on resolving customer issues and providing support.",
    tools=[ticket_system_tool, knowledge_base_tool, escalation_tool]
)

technical_support_agent = Agent(
    name="Technical Support", 
    instructions="You are a technical support specialist. Help with technical issues and troubleshooting.",
    tools=[system_diagnostics_tool, log_analysis_tool, deployment_tool]
)

triage_agent = Agent(
    name="Support Triage",
    instructions="Route customer requests to the appropriate specialized agent based on the issue type.",
    handoffs=[customer_service_agent, technical_support_agent]
)
```

**层次化代理设计**
使用分层的代理结构，从通用到专门，实现清晰的责任分工。

```python
# 实战案例：电商客服系统
class ECommerceCustomerService:
    def __init__(self):
        # 专门代理
        self.order_agent = Agent(
            name="Order Specialist",
            instructions="Handle order-related inquiries: status, modifications, cancellations.",
            tools=[order_lookup_tool, order_modify_tool, shipping_tracker_tool]
        )
        
        self.product_agent = Agent(
            name="Product Specialist", 
            instructions="Answer product questions: features, compatibility, recommendations.",
            tools=[product_catalog_tool, inventory_tool, recommendation_engine_tool]
        )
        
        self.billing_agent = Agent(
            name="Billing Specialist",
            instructions="Handle billing and payment issues: refunds, payment failures, invoices.",
            tools=[payment_processor_tool, refund_tool, invoice_generator_tool]
        )
        
        # 主路由代理
        self.main_agent = Agent(
            name="Customer Service Router",
            instructions="""
            You are the main customer service agent. Analyze customer requests and:
            1. For order-related issues, handoff to Order Specialist
            2. For product questions, handoff to Product Specialist  
            3. For billing/payment issues, handoff to Billing Specialist
            4. For simple greetings or general info, handle directly
            """,
            handoffs=[self.order_agent, self.product_agent, self.billing_agent],
            tools=[general_info_tool, store_hours_tool]
        )
```

### 10.1.2 工具设计最佳实践

**幂等性设计**
工具应该设计为幂等的，多次调用相同参数产生相同结果。

```python
@function_tool
def get_user_info(user_id: str) -> UserInfo:
    """
    获取用户信息 - 幂等操作
    
    多次调用返回相同结果，不会产生副作用
    """
    return database.get_user(user_id)

@function_tool
def create_user_safely(
    email: str, 
    name: str,
    phone: str | None = None
) -> UserInfo:
    """
    安全创建用户 - 处理重复调用
    
    如果用户已存在，返回现有用户而不是报错
    """
    existing_user = database.find_user_by_email(email)
    if existing_user:
        logger.info(f"User {email} already exists, returning existing user")
        return existing_user
    
    return database.create_user(email=email, name=name, phone=phone)

@function_tool
def update_user_email(
    user_id: str, 
    new_email: str,
    confirmation_required: bool = True
) -> UpdateResult:
    """
    更新用户邮箱 - 带确认机制
    
    包含状态检查，避免无效更新
    """
    user = database.get_user(user_id)
    if user.email == new_email:
        return UpdateResult(success=True, message="Email unchanged", user=user)
    
    if confirmation_required and not user.email_verified:
        return UpdateResult(
            success=False, 
            message="Please verify current email before changing",
            requires_verification=True
        )
    
    updated_user = database.update_user_email(user_id, new_email)
    return UpdateResult(success=True, message="Email updated", user=updated_user)
```

**错误处理和优雅降级**

```python
@function_tool
def robust_weather_lookup(
    city: str,
    country_code: str = "US"
) -> WeatherInfo:
    """
    健壮的天气查询工具 - 多重错误处理
    """
    
    try:
        # 主要API服务
        result = primary_weather_api.get_weather(city, country_code)
        return WeatherInfo(
            temperature=result.temperature,
            conditions=result.conditions,
            source="primary_api",
            reliability="high"
        )
        
    except APIRateLimitError:
        # 备用API服务
        try:
            result = backup_weather_api.get_weather(city, country_code)
            return WeatherInfo(
                temperature=result.temp,
                conditions=result.weather,
                source="backup_api", 
                reliability="medium"
            )
        except Exception as backup_error:
            logger.warning(f"Backup weather API failed: {backup_error}")
            
    except APIKeyError:
        return WeatherInfo(
            error="Weather service temporarily unavailable due to authentication issue",
            source="error",
            reliability="none"
        )
        
    except Exception as e:
        logger.error(f"Weather lookup failed for {city}: {e}")
        
        # 使用缓存的数据作为最后手段
        cached_data = weather_cache.get(f"{city}_{country_code}")
        if cached_data and not cached_data.is_expired():
            return WeatherInfo(
                temperature=cached_data.temperature,
                conditions=cached_data.conditions,
                source="cache",
                reliability="low",
                note="Data from cache due to service unavailability"
            )
        
        # 最终降级：返回通用信息
        return WeatherInfo(
            error=f"Unable to get weather for {city}. Please try again later.",
            source="error",
            reliability="none"
        )
```

## 10.2 性能优化实战

### 10.2.1 批处理和并行执行

```python
class OptimizedAgentWorkflow:
    """优化的代理工作流示例"""
    
    @function_tool
    async def batch_user_lookup(
        user_ids: List[str]
    ) -> List[UserInfo]:
        """
        批量用户查询 - 减少数据库往返
        """
        # 批量查询而不是循环单次查询
        users = await database.get_users_batch(user_ids)
        return users
    
    @function_tool  
    async def parallel_data_enrichment(
        user_id: str
    ) -> EnrichedUserData:
        """
        并行数据增强 - 同时调用多个API
        """
        
        # 并行执行多个独立的API调用
        user_task = database.get_user(user_id)
        orders_task = order_service.get_user_orders(user_id)
        preferences_task = preferences_service.get_user_preferences(user_id)
        activity_task = activity_service.get_recent_activity(user_id)
        
        user, orders, preferences, activity = await asyncio.gather(
            user_task, orders_task, preferences_task, activity_task,
            return_exceptions=True
        )
        
        # 处理部分失败的情况
        enriched_data = EnrichedUserData(user=user)
        
        if not isinstance(orders, Exception):
            enriched_data.orders = orders
        if not isinstance(preferences, Exception):
            enriched_data.preferences = preferences  
        if not isinstance(activity, Exception):
            enriched_data.recent_activity = activity
        
        return enriched_data
```

### 10.2.2 智能缓存策略

```python
class CacheOptimizedTools:
    """带缓存优化的工具集"""
    
    def __init__(self):
        # 多层缓存：内存 + Redis
        self.memory_cache = {}
        self.redis_cache = redis.Redis()
        
    @function_tool
    async def cached_api_call(
        self,
        endpoint: str,
        params: Dict[str, Any],
        ttl: int = 300  # 5分钟缓存
    ) -> ApiResponse:
        """
        带缓存的API调用
        """
        
        # 生成缓存键
        cache_key = self._generate_cache_key(endpoint, params)
        
        # L1: 内存缓存查找
        if cache_key in self.memory_cache:
            cached_data, timestamp = self.memory_cache[cache_key]
            if time.time() - timestamp < ttl:
                return cached_data
            else:
                del self.memory_cache[cache_key]
        
        # L2: Redis缓存查找
        cached_json = await self.redis_cache.get(cache_key)
        if cached_json:
            try:
                cached_data = ApiResponse.from_json(cached_json)
                # 同时更新内存缓存
                self.memory_cache[cache_key] = (cached_data, time.time())
                return cached_data
            except Exception:
                await self.redis_cache.delete(cache_key)
        
        # 缓存未命中，执行实际API调用
        response = await self._make_api_call(endpoint, params)
        
        # 更新两级缓存
        self.memory_cache[cache_key] = (response, time.time())
        await self.redis_cache.setex(
            cache_key, 
            ttl, 
            response.to_json()
        )
        
        return response
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """生成一致的缓存键"""
        sorted_params = sorted(params.items())
        params_str = "&".join(f"{k}={v}" for k, v in sorted_params)
        return f"api_cache:{endpoint}:{hashlib.md5(params_str.encode()).hexdigest()}"
```

### 10.2.3 连接池和资源管理

```python
class ResourceOptimizedAgent:
    """资源优化的代理实现"""
    
    def __init__(self):
        # 数据库连接池
        self.db_pool = asyncpg.create_pool(
            database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # HTTP会话复用
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=10)
        )
    
    @function_tool
    async def efficient_database_query(
        self,
        query: str,
        params: List[Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        高效的数据库查询 - 使用连接池
        """
        
        async with self.db_pool.acquire() as connection:
            async with connection.transaction():
                result = await connection.fetch(query, *(params or []))
                return [dict(row) for row in result]
    
    @function_tool
    async def efficient_http_request(
        self,
        url: str,
        method: str = "GET",
        headers: Dict[str, str] | None = None,
        data: Any = None
    ) -> HttpResponse:
        """
        高效的HTTP请求 - 复用连接
        """
        
        async with self.http_session.request(
            method, url, headers=headers, json=data
        ) as response:
            content = await response.text()
            return HttpResponse(
                status_code=response.status,
                headers=dict(response.headers),
                content=content
            )
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.db_pool
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口 - 清理资源"""
        await self.db_pool.close()
        await self.http_session.close()
```

## 10.3 安全性最佳实践

### 10.3.1 输入验证和清洗

```python
@input_guardrail
async def comprehensive_input_validation(
    context: RunContextWrapper,
    agent: Agent,
    input_data: str | List[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """
    综合输入验证防护
    """
    
    # 将输入转换为文本进行分析
    if isinstance(input_data, list):
        text = "\n".join(
            item.get("content", "") if isinstance(item, dict) else str(item)
            for item in input_data
        )
    else:
        text = input_data
    
    # 1. SQL注入检测
    sql_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
        r"(UNION\s+SELECT)",
        r"(OR\s+1\s*=\s*1)",
        r"(;\s*(DROP|DELETE))",
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailFunctionOutput(
                tripwire_triggered=True,
                message="Potential SQL injection attempt detected",
                severity="high",
                metadata={"pattern_matched": pattern, "input_sample": text[:100]}
            )
    
    # 2. XSS检测
    xss_patterns = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
    ]
    
    for pattern in xss_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailFunctionOutput(
                tripwire_triggered=True,
                message="Potential XSS attempt detected",
                severity="high",
                metadata={"pattern_matched": pattern}
            )
    
    # 3. 敏感信息检测
    sensitive_patterns = {
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}-\d{3}-\d{4}\b",
    }
    
    detected_sensitive = []
    for info_type, pattern in sensitive_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            detected_sensitive.append({
                "type": info_type,
                "count": len(matches)
            })
    
    if detected_sensitive:
        return GuardrailFunctionOutput(
            tripwire_triggered=False,  # 警告但不阻止
            message="Sensitive information detected in input",
            severity="medium",
            metadata={"sensitive_info": detected_sensitive}
        )
    
    # 4. 内容长度检查
    if len(text) > 50000:  # 50KB限制
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            message="Input exceeds maximum allowed length",
            severity="medium",
            metadata={"input_length": len(text), "max_length": 50000}
        )
    
    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        message="Input validation passed"
    )
```

### 10.3.2 权限控制和访问管理

```python
class SecureAgentSystem:
    """安全的代理系统实现"""
    
    def __init__(self):
        self.permission_manager = PermissionManager()
        self.audit_logger = AuditLogger()
    
    @function_tool
    async def secure_database_operation(
        self,
        context: ToolContext,
        operation: str,
        table: str,
        data: Dict[str, Any] | None = None,
        user_id: str | None = None
    ) -> OperationResult:
        """
        安全的数据库操作工具
        """
        
        # 1. 用户认证
        if not user_id:
            user_id = context.context.get("current_user_id")
            
        if not user_id:
            await self.audit_logger.log_security_event(
                event_type="unauthorized_access_attempt",
                details={"operation": operation, "table": table},
                severity="high"
            )
            return OperationResult(
                success=False,
                error="User authentication required",
                error_code="AUTH_REQUIRED"
            )
        
        # 2. 权限检查
        required_permission = f"database.{table}.{operation}"
        if not await self.permission_manager.check_permission(user_id, required_permission):
            await self.audit_logger.log_security_event(
                event_type="permission_denied",
                user_id=user_id,
                details={
                    "operation": operation,
                    "table": table,
                    "required_permission": required_permission
                },
                severity="medium"
            )
            return OperationResult(
                success=False,
                error=f"Insufficient permissions for {operation} on {table}",
                error_code="PERMISSION_DENIED"
            )
        
        # 3. 数据验证和清洗
        if data:
            validation_result = await self._validate_data(table, data)
            if not validation_result.valid:
                return OperationResult(
                    success=False,
                    error=f"Data validation failed: {validation_result.error}",
                    error_code="VALIDATION_FAILED"
                )
        
        # 4. 执行操作（在事务中）
        try:
            async with database.transaction():
                result = await self._execute_database_operation(
                    operation, table, data
                )
                
                # 5. 审计日志
                await self.audit_logger.log_database_operation(
                    user_id=user_id,
                    operation=operation,
                    table=table,
                    data_hash=hashlib.sha256(
                        json.dumps(data, sort_keys=True).encode()
                    ).hexdigest() if data else None,
                    success=True
                )
                
                return OperationResult(
                    success=True,
                    data=result,
                    operation_id=str(uuid.uuid4())
                )
                
        except Exception as e:
            await self.audit_logger.log_database_operation(
                user_id=user_id,
                operation=operation,
                table=table,
                error=str(e),
                success=False
            )
            
            return OperationResult(
                success=False,
                error=f"Database operation failed: {str(e)}",
                error_code="OPERATION_FAILED"
            )

class PermissionManager:
    """权限管理器"""
    
    async def check_permission(self, user_id: str, permission: str) -> bool:
        """检查用户权限"""
        user_permissions = await self._get_user_permissions(user_id)
        return permission in user_permissions or self._check_wildcard_permissions(
            user_permissions, permission
        )
    
    def _check_wildcard_permissions(
        self, 
        user_permissions: Set[str], 
        required_permission: str
    ) -> bool:
        """检查通配符权限"""
        permission_parts = required_permission.split(".")
        
        for i in range(len(permission_parts)):
            wildcard_permission = ".".join(permission_parts[:i + 1]) + ".*"
            if wildcard_permission in user_permissions:
                return True
                
        return "admin.*" in user_permissions

class AuditLogger:
    """审计日志记录器"""
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: str | None = None,
        details: Dict[str, Any] | None = None,
        severity: str = "medium"
    ) -> None:
        """记录安全事件"""
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details or {},
            "severity": severity,
            "source": "agents_security_system"
        }
        
        # 写入安全日志
        await self._write_to_security_log(event)
        
        # 高危事件实时告警
        if severity == "high":
            await self._send_security_alert(event)
```

## 10.4 监控和可观测性实战

### 10.4.1 自定义追踪处理器

```python
class ProductionTracingProcessor(TracingProcessor):
    """生产环境追踪处理器"""
    
    def __init__(self, config: TracingConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
    async def process_span_start(self, span: Span[Any]) -> None:
        """处理跨度开始 - 收集指标"""
        
        # 记录开始时间指标
        self.metrics_collector.record_span_start(
            span_type=type(span.span_data).__name__,
            agent_name=getattr(span.span_data, "agent_name", "unknown"),
            trace_id=span.span_data.trace_id
        )
    
    async def process_span_end(self, span: Span[Any]) -> None:
        """处理跨度结束 - 性能分析和告警"""
        
        duration = self._calculate_duration(span)
        
        # 记录持续时间指标
        self.metrics_collector.record_span_duration(
            span_type=type(span.span_data).__name__,
            duration=duration,
            success="error" not in span.span_data.tags
        )
        
        # 检查性能阈值
        if duration > self.config.slow_span_threshold:
            await self._handle_slow_span(span, duration)
        
        # 检查错误
        if "error" in span.span_data.tags:
            await self._handle_error_span(span)
    
    async def _handle_slow_span(self, span: Span[Any], duration: float) -> None:
        """处理慢查询跨度"""
        
        alert = {
            "type": "performance_degradation",
            "severity": "warning",
            "message": f"Slow span detected: {span.span_data.name}",
            "duration": duration,
            "threshold": self.config.slow_span_threshold,
            "trace_id": span.span_data.trace_id,
            "span_id": span.span_data.span_id,
            "metadata": span.span_data.metadata
        }
        
        await self.alert_manager.send_alert(alert)
    
    async def _handle_error_span(self, span: Span[Any]) -> None:
        """处理错误跨度"""
        
        error_rate = await self.metrics_collector.get_recent_error_rate(
            span_type=type(span.span_data).__name__,
            time_window=300  # 5分钟窗口
        )
        
        if error_rate > self.config.error_rate_threshold:
            alert = {
                "type": "high_error_rate",
                "severity": "critical",
                "message": f"High error rate detected: {error_rate:.2%}",
                "span_type": type(span.span_data).__name__,
                "error_rate": error_rate,
                "threshold": self.config.error_rate_threshold,
                "recent_errors": await self._get_recent_error_samples(span)
            }
            
            await self.alert_manager.send_alert(alert)

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        # 使用时序数据库存储指标
        self.timeseries_db = InfluxDBClient()
        
    async def record_span_duration(
        self,
        span_type: str,
        duration: float,
        success: bool,
        tags: Dict[str, str] | None = None
    ) -> None:
        """记录跨度持续时间"""
        
        measurement = {
            "measurement": "agent_span_duration",
            "tags": {
                "span_type": span_type,
                "success": str(success),
                **(tags or {})
            },
            "fields": {
                "duration_seconds": duration
            },
            "time": datetime.utcnow()
        }
        
        await self.timeseries_db.write_point(measurement)
    
    async def get_recent_error_rate(
        self,
        span_type: str,
        time_window: int = 300
    ) -> float:
        """获取最近的错误率"""
        
        query = f"""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN success = 'false' THEN 1 END) as errors
        FROM agent_span_duration 
        WHERE span_type = '{span_type}'
        AND time >= now() - {time_window}s
        """
        
        result = await self.timeseries_db.query(query)
        
        if result and len(result) > 0:
            total = result[0].get("total", 0)
            errors = result[0].get("errors", 0)
            return errors / total if total > 0 else 0.0
        
        return 0.0
```

### 10.4.2 健康检查和状态监控

```python
class AgentHealthMonitor:
    """代理健康监控器"""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.health_checks = {}
        self.last_check_time = {}
        
    async def register_health_check(
        self,
        agent_name: str,
        check_function: Callable[[], Awaitable[HealthStatus]]
    ) -> None:
        """注册健康检查函数"""
        self.health_checks[agent_name] = check_function
    
    async def check_all_agents_health(self) -> Dict[str, HealthStatus]:
        """检查所有代理的健康状态"""
        
        health_results = {}
        
        for agent in self.agents:
            try:
                health_status = await self._check_agent_health(agent)
                health_results[agent.name] = health_status
                
            except Exception as e:
                health_results[agent.name] = HealthStatus(
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    last_check=datetime.utcnow(),
                    checks={}
                )
        
        return health_results
    
    async def _check_agent_health(self, agent: Agent) -> HealthStatus:
        """检查单个代理的健康状态"""
        
        checks = {}
        overall_status = "healthy"
        
        # 1. 基础连接检查
        model_check = await self._check_model_connectivity(agent)
        checks["model_connectivity"] = model_check
        if model_check["status"] != "ok":
            overall_status = "degraded"
        
        # 2. 工具可用性检查
        tools_check = await self._check_tools_availability(agent)
        checks["tools_availability"] = tools_check
        if tools_check["status"] != "ok":
            overall_status = "degraded"
        
        # 3. 自定义健康检查
        if agent.name in self.health_checks:
            custom_check = await self.health_checks[agent.name]()
            checks["custom"] = {
                "status": "ok" if custom_check.is_healthy else "error",
                "message": custom_check.message,
                "details": custom_check.details
            }
            if not custom_check.is_healthy:
                overall_status = "unhealthy"
        
        # 4. 性能指标检查
        performance_check = await self._check_performance_metrics(agent)
        checks["performance"] = performance_check
        if performance_check["status"] == "warning":
            overall_status = max(overall_status, "degraded")
        
        return HealthStatus(
            status=overall_status,
            message=self._generate_health_summary(checks),
            last_check=datetime.utcnow(),
            checks=checks
        )
    
    async def _check_model_connectivity(self, agent: Agent) -> Dict[str, Any]:
        """检查模型连接性"""
        
        try:
            # 发送简单的测试请求
            test_result = await Runner.run(
                agent.clone(
                    instructions="Reply with exactly 'OK' and nothing else.",
                    tools=[],  # 移除工具以简化测试
                    handoffs=[],
                    input_guardrails=[],
                    output_guardrails=[]
                ),
                input="Health check",
                max_turns=1
            )
            
            response_time = test_result.context_wrapper.usage.total_tokens / 1000  # 估算
            
            if "OK" in str(test_result.final_output):
                return {
                    "status": "ok",
                    "response_time": response_time,
                    "message": "Model responding normally"
                }
            else:
                return {
                    "status": "warning", 
                    "response_time": response_time,
                    "message": "Model response unexpected",
                    "actual_response": str(test_result.final_output)[:100]
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Model connectivity failed: {str(e)}"
            }
    
    async def _check_tools_availability(self, agent: Agent) -> Dict[str, Any]:
        """检查工具可用性"""
        
        unavailable_tools = []
        
        for tool in agent.tools:
            try:
                # 尝试基本的工具调用（如果工具支持健康检查）
                if hasattr(tool, 'health_check'):
                    is_healthy = await tool.health_check()
                    if not is_healthy:
                        unavailable_tools.append(tool.name)
                        
            except Exception as e:
                unavailable_tools.append(f"{tool.name} ({str(e)})")
        
        if unavailable_tools:
            return {
                "status": "error" if len(unavailable_tools) == len(agent.tools) else "warning",
                "message": f"Some tools unavailable: {', '.join(unavailable_tools)}",
                "unavailable_tools": unavailable_tools,
                "total_tools": len(agent.tools)
            }
        else:
            return {
                "status": "ok",
                "message": "All tools available",
                "total_tools": len(agent.tools)
            }

@dataclass
class HealthStatus:
    """健康状态数据类"""
    status: Literal["healthy", "degraded", "unhealthy"]
    message: str
    last_check: datetime
    checks: Dict[str, Any]
    
    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"
```

这些实战经验和最佳实践涵盖了OpenAI Agents SDK在生产环境中的关键考虑因素，包括架构设计、性能优化、安全性、监控等各个方面，为开发者提供了实用的指导。
