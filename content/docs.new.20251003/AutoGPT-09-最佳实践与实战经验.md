---
title: "AutoGPT-09-最佳实践与实战经验：平台开发运维实战指南"
date: 2025-01-27T18:00:00+08:00
draft: false
featured: true
series: "autogpt-best-practices"
tags: ["AutoGPT", "最佳实践", "实战经验", "开发指南", "运维指南", "性能优化"]
categories: ["autogpt", "最佳实践"]
description: "汇总AutoGPT平台开发和运维的最佳实践，包括代码规范、架构设计、性能优化、安全防护和运维经验"
weight: 190
slug: "AutoGPT-09-最佳实践与实战经验"
---

## 概述

本文档汇总了AutoGPT平台在开发、部署、运维过程中积累的最佳实践和实战经验。通过这些经过验证的方法和策略，可以帮助开发者和运维人员更好地理解和使用AutoGPT平台，避免常见的陷阱，提高开发效率和系统稳定性。

<!--more-->

## 1. 开发最佳实践

### 1.1 代码规范与质量

#### 1.1.1 Python代码规范

**基础规范**：
```python
# 遵循PEP 8规范
# 使用类型注解提高代码可读性和IDE支持

from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    """
    用户配置模型
    
    遵循以下规范：
    1. 类名使用PascalCase
    2. 函数和变量使用snake_case
    3. 常量使用UPPER_CASE
    4. 私有成员使用下划线前缀
    """
    
    id: str = Field(..., description="用户唯一标识符")
    email: str = Field(..., description="用户邮箱地址")
    created_at: datetime = Field(..., description="创建时间")
    preferences: Dict[str, Union[str, int, bool]] = Field(
        default_factory=dict, 
        description="用户偏好设置"
    )
    
    def get_display_name(self) -> str:
        """
        获取用户显示名称
        
        Returns:
            str: 用户显示名称
        """
        return self.email.split('@')[0]
    
    def _validate_preferences(self) -> bool:
        """私有方法：验证偏好设置"""
        # 实现验证逻辑
        return True

# 常量定义
DEFAULT_PAGE_SIZE = 20
MAX_RETRY_ATTEMPTS = 3
CACHE_EXPIRY_SECONDS = 3600
```

**错误处理规范**：
```python
import logging
from typing import Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class AutoGPTError(Exception):
    """AutoGPT基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ValidationError(AutoGPTError):
    """数据验证异常"""
    pass

class ExecutionError(AutoGPTError):
    """执行异常"""
    pass

async def execute_graph_safely(graph_id: str, inputs: Dict) -> Dict:
    """
    安全执行图的示例
    
    最佳实践：
    1. 明确的异常类型
    2. 详细的错误日志
    3. 优雅的错误处理
    4. 资源清理
    """
    try:
        logger.info(f"开始执行图: {graph_id}")
        
        # 验证输入
        if not graph_id:
            raise ValidationError("图ID不能为空", "INVALID_GRAPH_ID")
        
        # 执行逻辑
        result = await _execute_graph_internal(graph_id, inputs)
        
        logger.info(f"图执行成功: {graph_id}")
        return result
        
    except ValidationError as e:
        logger.error(f"输入验证失败: {e.message}")
        raise
    except ExecutionError as e:
        logger.error(f"图执行失败: {e.message}")
        raise
    except Exception as e:
        logger.error(f"未知错误: {str(e)}", exc_info=True)
        raise ExecutionError(f"图执行异常: {str(e)}", "EXECUTION_FAILED")

@asynccontextmanager
async def database_transaction():
    """数据库事务上下文管理器"""
    transaction = None
    try:
        transaction = await db.begin()
        yield transaction
        await transaction.commit()
    except Exception as e:
        if transaction:
            await transaction.rollback()
        logger.error(f"数据库事务失败: {e}")
        raise
```

#### 1.1.2 TypeScript/JavaScript代码规范

**React组件规范**：
```typescript
// 使用TypeScript提供类型安全
import React, { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';

// 接口定义
interface GraphExecutionProps {
  graphId: string;
  onExecutionComplete?: (result: ExecutionResult) => void;
  className?: string;
}

interface ExecutionResult {
  id: string;
  status: 'completed' | 'failed' | 'cancelled';
  outputs: Record<string, unknown>;
  error?: string;
}

// 组件实现
export const GraphExecution: React.FC<GraphExecutionProps> = ({
  graphId,
  onExecutionComplete,
  className = ''
}) => {
  // 状态管理
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionId, setExecutionId] = useState<string | null>(null);
  
  // 数据获取
  const { data: graph, isLoading } = useQuery({
    queryKey: ['graph', graphId],
    queryFn: () => fetchGraph(graphId),
    enabled: !!graphId
  });
  
  // 执行变更
  const executeMutation = useMutation({
    mutationFn: executeGraph,
    onSuccess: (result) => {
      setIsExecuting(false);
      onExecutionComplete?.(result);
    },
    onError: (error) => {
      setIsExecuting(false);
      console.error('执行失败:', error);
    }
  });
  
  // 事件处理
  const handleExecute = useCallback(async () => {
    if (!graph || isExecuting) return;
    
    setIsExecuting(true);
    try {
      const result = await executeMutation.mutateAsync({
        graphId,
        inputs: {}
      });
      setExecutionId(result.id);
    } catch (error) {
      // 错误已在mutation中处理
    }
  }, [graph, isExecuting, executeMutation, graphId]);
  
  // 副作用
  useEffect(() => {
    // 清理逻辑
    return () => {
      if (executionId) {
        // 取消执行
        cancelExecution(executionId);
      }
    };
  }, [executionId]);
  
  if (isLoading) {
    return <div className="loading">加载中...</div>;
  }
  
  return (
    <div className={`graph-execution ${className}`}>
      <h3>{graph?.name}</h3>
      <button 
        onClick={handleExecute}
        disabled={isExecuting}
        className="execute-button"
      >
        {isExecuting ? '执行中...' : '执行'}
      </button>
    </div>
  );
};

// 工具函数
async function fetchGraph(graphId: string): Promise<Graph> {
  const response = await fetch(`/api/graphs/${graphId}`);
  if (!response.ok) {
    throw new Error(`获取图失败: ${response.statusText}`);
  }
  return response.json();
}

async function executeGraph(params: {
  graphId: string;
  inputs: Record<string, unknown>;
}): Promise<ExecutionResult> {
  const response = await fetch('/api/executions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  
  if (!response.ok) {
    throw new Error(`执行失败: ${response.statusText}`);
  }
  
  return response.json();
}
```

### 1.2 架构设计最佳实践

#### 1.2.1 模块化设计原则

**单一职责原则**：
```python
# 好的例子：职责单一的服务类
class UserAuthenticationService:
    """用户认证服务 - 只负责认证相关逻辑"""
    
    def __init__(self, jwt_service: JWTService, user_repository: UserRepository):
        self.jwt_service = jwt_service
        self.user_repository = user_repository
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """认证用户"""
        user = await self.user_repository.get_by_email(email)
        if user and self._verify_password(password, user.password_hash):
            return user
        return None
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        # 密码验证逻辑
        pass

class UserProfileService:
    """用户配置服务 - 只负责配置管理"""
    
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository
    
    async def update_profile(self, user_id: str, profile_data: Dict) -> User:
        """更新用户配置"""
        # 配置更新逻辑
        pass

# 避免的例子：职责混乱的服务类
class UserService:  # 不好的设计
    """用户服务 - 职责过多"""
    
    async def authenticate_user(self, email: str, password: str):
        pass
    
    async def update_profile(self, user_id: str, profile_data: Dict):
        pass
    
    async def send_notification(self, user_id: str, message: str):
        pass
    
    async def calculate_usage_cost(self, user_id: str):
        pass
```

**依赖注入模式**：
```python
from abc import ABC, abstractmethod
from typing import Protocol

# 定义接口
class EmailServiceProtocol(Protocol):
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        ...

class UserRepositoryProtocol(Protocol):
    async def get_by_id(self, user_id: str) -> Optional[User]:
        ...
    
    async def save(self, user: User) -> User:
        ...

# 实现类
class SMTPEmailService:
    """SMTP邮件服务实现"""
    
    def __init__(self, smtp_config: SMTPConfig):
        self.smtp_config = smtp_config
    
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        # SMTP发送实现
        pass

class DatabaseUserRepository:
    """数据库用户仓储实现"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        # 数据库查询实现
        pass

# 服务类使用依赖注入
class NotificationService:
    """通知服务"""
    
    def __init__(
        self, 
        email_service: EmailServiceProtocol,
        user_repository: UserRepositoryProtocol
    ):
        self.email_service = email_service
        self.user_repository = user_repository
    
    async def notify_user(self, user_id: str, message: str) -> bool:
        """通知用户"""
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            return False
        
        return await self.email_service.send_email(
            to=user.email,
            subject="AutoGPT通知",
            body=message
        )

# 依赖注入容器配置
def create_notification_service() -> NotificationService:
    """创建通知服务实例"""
    smtp_config = SMTPConfig.from_env()
    email_service = SMTPEmailService(smtp_config)
    
    db_connection = create_db_connection()
    user_repository = DatabaseUserRepository(db_connection)
    
    return NotificationService(email_service, user_repository)
```

#### 1.2.2 异步编程最佳实践

**异步函数设计**：
```python
import asyncio
from typing import List, Dict, Any
from contextlib import asynccontextmanager

class GraphExecutionService:
    """图执行服务 - 异步编程最佳实践"""
    
    def __init__(self, max_concurrent_executions: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent_executions)
        self.active_executions: Dict[str, asyncio.Task] = {}
    
    async def execute_graph(self, graph_id: str, inputs: Dict) -> str:
        """执行图 - 返回执行ID"""
        execution_id = generate_execution_id()
        
        # 使用信号量控制并发数
        async with self.semaphore:
            task = asyncio.create_task(
                self._execute_graph_internal(execution_id, graph_id, inputs)
            )
            self.active_executions[execution_id] = task
            
            # 设置任务完成回调
            task.add_done_callback(
                lambda t: self.active_executions.pop(execution_id, None)
            )
        
        return execution_id
    
    async def _execute_graph_internal(
        self, 
        execution_id: str, 
        graph_id: str, 
        inputs: Dict
    ) -> Dict:
        """内部执行逻辑"""
        try:
            # 获取图定义
            graph = await self._get_graph(graph_id)
            
            # 并发执行节点
            results = await self._execute_nodes_concurrently(
                execution_id, 
                graph.nodes, 
                inputs
            )
            
            return {"execution_id": execution_id, "results": results}
            
        except Exception as e:
            logger.error(f"图执行失败 {execution_id}: {e}")
            raise
    
    async def _execute_nodes_concurrently(
        self, 
        execution_id: str, 
        nodes: List[Node], 
        inputs: Dict
    ) -> Dict:
        """并发执行节点"""
        # 构建依赖图
        dependency_graph = self._build_dependency_graph(nodes)
        
        # 按拓扑顺序执行
        results = {}
        executed_nodes = set()
        
        while len(executed_nodes) < len(nodes):
            # 找到可以执行的节点（依赖已满足）
            ready_nodes = [
                node for node in nodes
                if node.id not in executed_nodes
                and all(dep in executed_nodes for dep in dependency_graph.get(node.id, []))
            ]
            
            if not ready_nodes:
                raise ExecutionError("检测到循环依赖")
            
            # 并发执行就绪的节点
            tasks = [
                self._execute_node(execution_id, node, inputs, results)
                for node in ready_nodes
            ]
            
            node_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for node, result in zip(ready_nodes, node_results):
                if isinstance(result, Exception):
                    raise ExecutionError(f"节点执行失败 {node.id}: {result}")
                
                results[node.id] = result
                executed_nodes.add(node.id)
        
        return results
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        task = self.active_executions.get(execution_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        return False
    
    @asynccontextmanager
    async def execution_context(self, execution_id: str):
        """执行上下文管理器"""
        logger.info(f"开始执行 {execution_id}")
        start_time = asyncio.get_event_loop().time()
        
        try:
            yield
        finally:
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            logger.info(f"执行完成 {execution_id}, 耗时: {duration:.2f}s")
```

### 1.3 测试最佳实践

#### 1.3.1 单元测试

**测试结构和命名**：
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

class TestUserAuthenticationService:
    """用户认证服务测试类
    
    测试命名规范：test_<method_name>_<scenario>_<expected_result>
    """
    
    @pytest.fixture
    def mock_jwt_service(self):
        """JWT服务模拟对象"""
        return Mock(spec=JWTService)
    
    @pytest.fixture
    def mock_user_repository(self):
        """用户仓储模拟对象"""
        return AsyncMock(spec=UserRepository)
    
    @pytest.fixture
    def auth_service(self, mock_jwt_service, mock_user_repository):
        """认证服务实例"""
        return UserAuthenticationService(mock_jwt_service, mock_user_repository)
    
    async def test_authenticate_user_valid_credentials_returns_user(
        self, 
        auth_service, 
        mock_user_repository
    ):
        """测试：有效凭据认证返回用户"""
        # Arrange
        email = "test@example.com"
        password = "password123"
        expected_user = User(id="user1", email=email)
        
        mock_user_repository.get_by_email.return_value = expected_user
        
        # Act
        result = await auth_service.authenticate_user(email, password)
        
        # Assert
        assert result == expected_user
        mock_user_repository.get_by_email.assert_called_once_with(email)
    
    async def test_authenticate_user_invalid_credentials_returns_none(
        self, 
        auth_service, 
        mock_user_repository
    ):
        """测试：无效凭据认证返回None"""
        # Arrange
        email = "test@example.com"
        password = "wrong_password"
        
        mock_user_repository.get_by_email.return_value = None
        
        # Act
        result = await auth_service.authenticate_user(email, password)
        
        # Assert
        assert result is None
    
    async def test_authenticate_user_database_error_raises_exception(
        self, 
        auth_service, 
        mock_user_repository
    ):
        """测试：数据库错误抛出异常"""
        # Arrange
        email = "test@example.com"
        password = "password123"
        
        mock_user_repository.get_by_email.side_effect = DatabaseError("连接失败")
        
        # Act & Assert
        with pytest.raises(DatabaseError):
            await auth_service.authenticate_user(email, password)

# 参数化测试
@pytest.mark.parametrize("email,password,expected", [
    ("valid@example.com", "password123", True),
    ("", "password123", False),
    ("valid@example.com", "", False),
    ("invalid-email", "password123", False),
])
async def test_validate_credentials_various_inputs(email, password, expected):
    """测试：各种输入的凭据验证"""
    result = validate_credentials(email, password)
    assert result == expected

# 集成测试
@pytest.mark.integration
class TestGraphExecutionIntegration:
    """图执行集成测试"""
    
    @pytest.fixture
    async def test_database(self):
        """测试数据库"""
        # 创建测试数据库连接
        db = await create_test_database()
        yield db
        await cleanup_test_database(db)
    
    async def test_complete_graph_execution_flow(self, test_database):
        """测试：完整的图执行流程"""
        # 创建测试图
        graph = await create_test_graph(test_database)
        
        # 执行图
        execution_service = GraphExecutionService(test_database)
        execution_id = await execution_service.execute_graph(
            graph.id, 
            {"input": "test_value"}
        )
        
        # 等待执行完成
        result = await wait_for_execution_completion(execution_id)
        
        # 验证结果
        assert result.status == "completed"
        assert "output" in result.outputs
```

#### 1.3.2 端到端测试

**Playwright测试示例**：
```typescript
// tests/e2e/graph-execution.spec.ts
import { test, expect } from '@playwright/test';

test.describe('图执行功能', () => {
  test.beforeEach(async ({ page }) => {
    // 登录测试用户
    await page.goto('/login');
    await page.fill('[data-testid="email-input"]', 'test@example.com');
    await page.fill('[data-testid="password-input"]', 'password123');
    await page.click('[data-testid="login-button"]');
    
    // 等待跳转到主页
    await expect(page).toHaveURL('/dashboard');
  });

  test('应该能够创建和执行简单图', async ({ page }) => {
    // 导航到图编辑器
    await page.click('[data-testid="create-graph-button"]');
    await expect(page).toHaveURL('/graphs/new');
    
    // 添加节点
    await page.click('[data-testid="add-node-button"]');
    await page.selectOption('[data-testid="node-type-select"]', 'text-input');
    await page.click('[data-testid="confirm-add-node"]');
    
    // 配置节点
    await page.fill('[data-testid="node-input-text"]', 'Hello, World!');
    
    // 保存图
    await page.fill('[data-testid="graph-name-input"]', '测试图');
    await page.click('[data-testid="save-graph-button"]');
    
    // 等待保存成功
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    
    // 执行图
    await page.click('[data-testid="execute-graph-button"]');
    
    // 等待执行完成
    await expect(page.locator('[data-testid="execution-status"]')).toHaveText('已完成');
    
    // 验证输出
    const output = await page.textContent('[data-testid="execution-output"]');
    expect(output).toContain('Hello, World!');
  });

  test('应该能够处理执行错误', async ({ page }) => {
    // 创建会失败的图
    await page.goto('/graphs/new');
    await page.click('[data-testid="add-node-button"]');
    await page.selectOption('[data-testid="node-type-select"]', 'api-call');
    
    // 配置无效的API调用
    await page.fill('[data-testid="api-url-input"]', 'invalid-url');
    
    // 保存并执行
    await page.fill('[data-testid="graph-name-input"]', '错误测试图');
    await page.click('[data-testid="save-graph-button"]');
    await page.click('[data-testid="execute-graph-button"]');
    
    // 验证错误处理
    await expect(page.locator('[data-testid="execution-status"]')).toHaveText('失败');
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
  });
});
```

## 2. 性能优化最佳实践

### 2.1 数据库优化

#### 2.1.1 查询优化

**索引策略**：
```sql
-- 用户表优化
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at);
CREATE INDEX CONCURRENTLY idx_users_active ON users(is_active) WHERE is_active = true;

-- 图表优化
CREATE INDEX CONCURRENTLY idx_graphs_user_id ON graphs(user_id);
CREATE INDEX CONCURRENTLY idx_graphs_user_active ON graphs(user_id, is_active) 
  WHERE is_active = true;

-- 执行表优化
CREATE INDEX CONCURRENTLY idx_executions_user_status ON graph_executions(user_id, status);
CREATE INDEX CONCURRENTLY idx_executions_created_at ON graph_executions(created_at);
CREATE INDEX CONCURRENTLY idx_executions_graph_id ON graph_executions(graph_id);

-- 复合索引优化
CREATE INDEX CONCURRENTLY idx_executions_user_created ON graph_executions(user_id, created_at DESC);
```

**查询优化示例**：
```python
class OptimizedGraphRepository:
    """优化的图仓储实现"""
    
    async def get_user_graphs_with_stats(
        self, 
        user_id: str, 
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict]:
        """
        获取用户图列表及统计信息
        
        优化策略：
        1. 使用单个查询获取所有需要的数据
        2. 避免N+1查询问题
        3. 使用适当的索引
        """
        query = """
        SELECT 
            g.id,
            g.name,
            g.description,
            g.created_at,
            g.updated_at,
            COUNT(ge.id) as execution_count,
            MAX(ge.created_at) as last_execution_at,
            AVG(CASE WHEN ge.status = 'completed' THEN 1.0 ELSE 0.0 END) as success_rate
        FROM graphs g
        LEFT JOIN graph_executions ge ON g.id = ge.graph_id
        WHERE g.user_id = $1 AND g.is_active = true
        GROUP BY g.id, g.name, g.description, g.created_at, g.updated_at
        ORDER BY g.updated_at DESC
        LIMIT $2 OFFSET $3
        """
        
        return await self.db.fetch_all(query, user_id, limit, offset)
    
    async def get_execution_history_paginated(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[str] = None
    ) -> Dict:
        """
        分页获取执行历史
        
        优化策略：
        1. 使用游标分页提高大数据集性能
        2. 并行执行数据查询和计数查询
        3. 使用适当的WHERE条件和索引
        """
        offset = (page - 1) * page_size
        
        # 构建WHERE条件
        where_conditions = ["user_id = $1"]
        params = [user_id]
        
        if status_filter:
            where_conditions.append("status = $2")
            params.append(status_filter)
        
        where_clause = " AND ".join(where_conditions)
        
        # 并行执行数据查询和计数查询
        data_query = f"""
        SELECT id, graph_id, status, created_at, started_at, ended_at
        FROM graph_executions
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """
        
        count_query = f"""
        SELECT COUNT(*) as total
        FROM graph_executions
        WHERE {where_clause}
        """
        
        # 使用asyncio.gather并行执行
        data_task = self.db.fetch_all(data_query, *params, page_size, offset)
        count_task = self.db.fetch_one(count_query, *params)
        
        data_results, count_result = await asyncio.gather(data_task, count_task)
        
        return {
            "data": data_results,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": count_result["total"],
                "total_pages": (count_result["total"] + page_size - 1) // page_size
            }
        }
```

#### 2.1.2 连接池优化

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

class DatabaseManager:
    """数据库管理器 - 连接池优化"""
    
    def __init__(self, database_url: str):
        # 连接池配置
        self.engine = create_async_engine(
            database_url,
            # 连接池设置
            poolclass=QueuePool,
            pool_size=20,          # 连接池大小
            max_overflow=30,       # 最大溢出连接数
            pool_timeout=30,       # 获取连接超时时间
            pool_recycle=3600,     # 连接回收时间（秒）
            pool_pre_ping=True,    # 连接前ping测试
            
            # 连接参数
            connect_args={
                "server_settings": {
                    "application_name": "autogpt_backend",
                    "jit": "off"  # 关闭JIT以提高小查询性能
                }
            },
            
            # 执行选项
            execution_options={
                "isolation_level": "READ_COMMITTED"
            }
        )
        
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def get_session(self) -> AsyncSession:
        """获取数据库会话"""
        return self.async_session()
    
    async def close(self):
        """关闭数据库连接"""
        await self.engine.dispose()
```

### 2.2 缓存策略优化

#### 2.2.1 多层缓存架构

```python
from typing import Optional, Any, Dict
import json
import hashlib
from datetime import timedelta

class MultiLevelCache:
    """多层缓存实现"""
    
    def __init__(
        self,
        redis_client,
        local_cache_size: int = 1000,
        default_ttl: int = 3600
    ):
        self.redis = redis_client
        self.local_cache = {}  # 简化的本地缓存
        self.local_cache_size = local_cache_size
        self.default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        查找顺序：本地缓存 -> Redis -> 数据库
        """
        # 1. 检查本地缓存
        if key in self.local_cache:
            return self.local_cache[key]
        
        # 2. 检查Redis缓存
        redis_value = await self.redis.get(key)
        if redis_value:
            try:
                value = json.loads(redis_value)
                # 更新本地缓存
                self._update_local_cache(key, value)
                return value
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存值
        
        同时更新本地缓存和Redis
        """
        ttl = ttl or self.default_ttl
        
        try:
            # 序列化值
            serialized_value = json.dumps(value, default=str)
            
            # 更新Redis
            await self.redis.setex(key, ttl, serialized_value)
            
            # 更新本地缓存
            self._update_local_cache(key, value)
            
            return True
        except Exception as e:
            logger.error(f"缓存设置失败: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        # 删除本地缓存
        self.local_cache.pop(key, None)
        
        # 删除Redis缓存
        return await self.redis.delete(key) > 0
    
    def _update_local_cache(self, key: str, value: Any):
        """更新本地缓存"""
        # 简单的LRU实现
        if len(self.local_cache) >= self.local_cache_size:
            # 删除最旧的条目
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
        
        self.local_cache[key] = value

class CacheService:
    """缓存服务 - 业务层缓存封装"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """获取用户配置（带缓存）"""
        cache_key = f"user_profile:{user_id}"
        
        # 尝试从缓存获取
        cached_profile = await self.cache.get(cache_key)
        if cached_profile:
            return cached_profile
        
        # 从数据库获取
        profile = await self._fetch_user_profile_from_db(user_id)
        if profile:
            # 缓存用户配置（1小时）
            await self.cache.set(cache_key, profile, ttl=3600)
        
        return profile
    
    async def invalidate_user_cache(self, user_id: str):
        """使用户缓存失效"""
        patterns = [
            f"user_profile:{user_id}",
            f"user_graphs:{user_id}",
            f"user_executions:{user_id}",
        ]
        
        for pattern in patterns:
            await self.cache.delete(pattern)
    
    def cache_key_for_graph_list(
        self, 
        user_id: str, 
        page: int, 
        filters: Dict
    ) -> str:
        """生成图列表的缓存键"""
        # 创建过滤器的哈希
        filter_hash = hashlib.md5(
            json.dumps(filters, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"graph_list:{user_id}:{page}:{filter_hash}"
```

#### 2.2.2 缓存预热和更新策略

```python
class CacheWarmupService:
    """缓存预热服务"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
    
    async def warmup_user_data(self, user_id: str):
        """预热用户数据"""
        tasks = [
            self._warmup_user_profile(user_id),
            self._warmup_user_graphs(user_id),
            self._warmup_recent_executions(user_id),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _warmup_user_profile(self, user_id: str):
        """预热用户配置"""
        await self.cache_service.get_user_profile(user_id)
    
    async def _warmup_user_graphs(self, user_id: str):
        """预热用户图列表"""
        # 预热前几页的图列表
        for page in range(1, 4):
            await self.cache_service.get_user_graphs(
                user_id, 
                page=page, 
                filters={}
            )
    
    async def schedule_cache_refresh(self):
        """定期刷新缓存"""
        while True:
            try:
                # 获取活跃用户列表
                active_users = await self._get_active_users()
                
                # 并发预热活跃用户数据
                semaphore = asyncio.Semaphore(10)  # 限制并发数
                
                async def warmup_with_semaphore(user_id: str):
                    async with semaphore:
                        await self.warmup_user_data(user_id)
                
                tasks = [
                    warmup_with_semaphore(user_id) 
                    for user_id in active_users
                ]
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # 等待下次刷新
                await asyncio.sleep(300)  # 5分钟
                
            except Exception as e:
                logger.error(f"缓存刷新失败: {e}")
                await asyncio.sleep(60)  # 错误时等待1分钟
```

### 2.3 前端性能优化

#### 2.3.1 React组件优化

```typescript
import React, { memo, useMemo, useCallback, useState } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';

// 使用memo优化组件重渲染
const GraphListItem = memo<{
  graph: Graph;
  onExecute: (graphId: string) => void;
  onEdit: (graphId: string) => void;
}>(({ graph, onExecute, onEdit }) => {
  // 使用useCallback缓存事件处理函数
  const handleExecute = useCallback(() => {
    onExecute(graph.id);
  }, [graph.id, onExecute]);
  
  const handleEdit = useCallback(() => {
    onEdit(graph.id);
  }, [graph.id, onEdit]);
  
  return (
    <div className="graph-item">
      <h3>{graph.name}</h3>
      <p>{graph.description}</p>
      <div className="actions">
        <button onClick={handleExecute}>执行</button>
        <button onClick={handleEdit}>编辑</button>
      </div>
    </div>
  );
});

// 虚拟滚动优化大列表
const VirtualizedGraphList: React.FC<{
  graphs: Graph[];
  onExecute: (graphId: string) => void;
  onEdit: (graphId: string) => void;
}> = ({ graphs, onExecute, onEdit }) => {
  const parentRef = React.useRef<HTMLDivElement>(null);
  
  const virtualizer = useVirtualizer({
    count: graphs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 120, // 估算每项高度
    overscan: 5, // 预渲染项数
  });
  
  return (
    <div
      ref={parentRef}
      className="graph-list-container"
      style={{ height: '600px', overflow: 'auto' }}
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualItem) => {
          const graph = graphs[virtualItem.index];
          return (
            <div
              key={virtualItem.key}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualItem.size}px`,
                transform: `translateY(${virtualItem.start}px)`,
              }}
            >
              <GraphListItem
                graph={graph}
                onExecute={onExecute}
                onEdit={onEdit}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};

// 使用useMemo优化计算
const GraphDashboard: React.FC = () => {
  const [graphs, setGraphs] = useState<Graph[]>([]);
  const [filter, setFilter] = useState('');
  
  // 使用useMemo缓存过滤结果
  const filteredGraphs = useMemo(() => {
    if (!filter) return graphs;
    
    return graphs.filter(graph =>
      graph.name.toLowerCase().includes(filter.toLowerCase()) ||
      graph.description.toLowerCase().includes(filter.toLowerCase())
    );
  }, [graphs, filter]);
  
  // 使用useMemo缓存统计数据
  const stats = useMemo(() => {
    return {
      total: graphs.length,
      active: graphs.filter(g => g.isActive).length,
      templates: graphs.filter(g => g.isTemplate).length,
    };
  }, [graphs]);
  
  const handleExecute = useCallback((graphId: string) => {
    // 执行逻辑
  }, []);
  
  const handleEdit = useCallback((graphId: string) => {
    // 编辑逻辑
  }, []);
  
  return (
    <div className="dashboard">
      <div className="stats">
        <div>总计: {stats.total}</div>
        <div>活跃: {stats.active}</div>
        <div>模板: {stats.templates}</div>
      </div>
      
      <input
        type="text"
        placeholder="搜索图..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />
      
      <VirtualizedGraphList
        graphs={filteredGraphs}
        onExecute={handleExecute}
        onEdit={handleEdit}
      />
    </div>
  );
};
```

#### 2.3.2 代码分割和懒加载

```typescript
import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';

// 懒加载组件
const GraphEditor = lazy(() => import('./components/GraphEditor'));
const GraphList = lazy(() => import('./components/GraphList'));
const Settings = lazy(() => import('./components/Settings'));
const Analytics = lazy(() => import('./components/Analytics'));

// 加载组件
const LoadingSpinner = () => (
  <div className="loading-spinner">
    <div className="spinner" />
    <p>加载中...</p>
  </div>
);

// 错误边界组件
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  
  static getDerivedStateFromError(error: Error) {
    return { hasError: true };
  }
  
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('组件错误:', error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>出错了</h2>
          <p>页面加载失败，请刷新重试。</p>
        </div>
      );
    }
    
    return this.props.children;
  }
}

// 应用路由
const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/" element={
          <Suspense fallback={<LoadingSpinner />}>
            <GraphList />
          </Suspense>
        } />
        
        <Route path="/editor/:graphId?" element={
          <Suspense fallback={<LoadingSpinner />}>
            <GraphEditor />
          </Suspense>
        } />
        
        <Route path="/settings" element={
          <Suspense fallback={<LoadingSpinner />}>
            <Settings />
          </Suspense>
        } />
        
        <Route path="/analytics" element={
          <Suspense fallback={<LoadingSpinner />}>
            <Analytics />
          </Suspense>
        } />
      </Routes>
    </ErrorBoundary>
  );
};

// 预加载关键路由
const preloadRoutes = () => {
  // 在空闲时间预加载常用组件
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      import('./components/GraphEditor');
      import('./components/Settings');
    });
  }
};

// 在应用启动时预加载
export { App, preloadRoutes };
```

## 3. 安全最佳实践

### 3.1 认证和授权

#### 3.1.1 JWT安全实践

```python
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional

class SecureJWTService:
    """安全的JWT服务实现"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        if len(secret_key) < 32:
            raise ValueError("JWT密钥长度至少32字符")
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(minutes=15)  # 短期访问令牌
        self.refresh_token_expire = timedelta(days=7)     # 长期刷新令牌
    
    def create_access_token(self, user_id: str, permissions: List[str]) -> str:
        """创建访问令牌"""
        now = datetime.utcnow()
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self.access_token_expire,
            "type": "access",
            "permissions": permissions,
            "jti": secrets.token_urlsafe(16),  # JWT ID防止重放攻击
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """创建刷新令牌"""
        now = datetime.utcnow()
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self.refresh_token_expire,
            "type": "refresh",
            "jti": secrets.token_urlsafe(16),
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict]:
        """验证令牌"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_signature": True,
                }
            )
            
            # 验证令牌类型
            if payload.get("type") != token_type:
                return None
            
            # 检查令牌是否在黑名单中
            if await self._is_token_blacklisted(payload.get("jti")):
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT令牌已过期")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"JWT令牌无效: {e}")
            return None
    
    async def blacklist_token(self, jti: str):
        """将令牌加入黑名单"""
        # 使用Redis存储黑名单，设置过期时间
        await redis_client.setex(
            f"blacklist:{jti}",
            self.refresh_token_expire.total_seconds(),
            "1"
        )
    
    async def _is_token_blacklisted(self, jti: str) -> bool:
        """检查令牌是否在黑名单中"""
        return await redis_client.exists(f"blacklist:{jti}")

# 权限装饰器
def require_permissions(*required_permissions: str):
    """权限检查装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 从请求上下文获取用户权限
            user_permissions = get_current_user_permissions()
            
            # 检查权限
            if not all(perm in user_permissions for perm in required_permissions):
                raise HTTPException(
                    status_code=403,
                    detail="权限不足"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# 使用示例
@require_permissions("graph:execute", "user:read")
async def execute_user_graph(graph_id: str, user_id: str):
    """执行用户图 - 需要图执行和用户读取权限"""
    pass
```

#### 3.1.2 输入验证和清理

```python
from pydantic import BaseModel, Field, validator
import re
import html
from typing import Optional, List

class SecureUserInput(BaseModel):
    """安全的用户输入模型"""
    
    email: str = Field(..., max_length=254)
    name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = Field(None, max_length=200)
    
    @validator('email')
    def validate_email(cls, v):
        """验证邮箱格式"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('邮箱格式无效')
        return v.lower().strip()
    
    @validator('name')
    def validate_name(cls, v):
        """验证姓名"""
        if v is None:
            return v
        
        # 移除HTML标签
        v = html.escape(v.strip())
        
        # 检查特殊字符
        if re.search(r'[<>"\']', v):
            raise ValueError('姓名包含无效字符')
        
        return v
    
    @validator('bio')
    def validate_bio(cls, v):
        """验证个人简介"""
        if v is None:
            return v
        
        # HTML转义
        v = html.escape(v.strip())
        
        # 检查恶意内容
        malicious_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('内容包含不安全字符')
        
        return v
    
    @validator('website')
    def validate_website(cls, v):
        """验证网站URL"""
        if v is None:
            return v
        
        # URL格式验证
        url_pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        if not re.match(url_pattern, v):
            raise ValueError('网站URL格式无效')
        
        # 检查恶意URL
        if any(domain in v.lower() for domain in ['malware.com', 'phishing.net']):
            raise ValueError('不允许的域名')
        
        return v

class SQLInjectionPrevention:
    """SQL注入防护"""
    
    @staticmethod
    def sanitize_search_query(query: str) -> str:
        """清理搜索查询"""
        if not query:
            return ""
        
        # 移除危险字符
        dangerous_chars = [';', '--', '/*', '*/', 'xp_', 'sp_']
        for char in dangerous_chars:
            query = query.replace(char, '')
        
        # 限制长度
        query = query[:100]
        
        # HTML转义
        query = html.escape(query.strip())
        
        return query
    
    @staticmethod
    def validate_order_by(order_by: str, allowed_columns: List[str]) -> str:
        """验证ORDER BY参数"""
        if order_by not in allowed_columns:
            raise ValueError(f"无效的排序字段: {order_by}")
        
        return order_by

# 使用参数化查询
async def get_user_graphs_safe(
    user_id: str, 
    search_query: Optional[str] = None,
    order_by: str = "created_at"
) -> List[Dict]:
    """安全的用户图查询"""
    
    # 验证输入
    if not user_id or not isinstance(user_id, str):
        raise ValueError("无效的用户ID")
    
    # 清理搜索查询
    if search_query:
        search_query = SQLInjectionPrevention.sanitize_search_query(search_query)
    
    # 验证排序字段
    allowed_order_columns = ["created_at", "updated_at", "name"]
    order_by = SQLInjectionPrevention.validate_order_by(order_by, allowed_order_columns)
    
    # 使用参数化查询
    if search_query:
        query = f"""
        SELECT id, name, description, created_at
        FROM graphs
        WHERE user_id = $1 
          AND is_active = true
          AND (name ILIKE $2 OR description ILIKE $2)
        ORDER BY {order_by} DESC
        """
        return await db.fetch_all(query, user_id, f"%{search_query}%")
    else:
        query = f"""
        SELECT id, name, description, created_at
        FROM graphs
        WHERE user_id = $1 AND is_active = true
        ORDER BY {order_by} DESC
        """
        return await db.fetch_all(query, user_id)
```

### 3.2 数据保护

#### 3.2.1 敏感数据加密

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from typing import Union

class DataEncryption:
    """数据加密服务"""
    
    def __init__(self, master_key: str):
        self.master_key = master_key.encode()
        self._fernet = self._create_fernet()
    
    def _create_fernet(self) -> Fernet:
        """创建Fernet加密器"""
        # 使用PBKDF2派生密钥
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'autogpt_salt',  # 在生产环境中应使用随机盐
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """加密数据"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self._fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self._fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
        except Exception as e:
            logger.error(f"解密失败: {e}")
            raise ValueError("数据解密失败")

class SecureCredentialsStore:
    """安全凭据存储"""
    
    def __init__(self, encryption_service: DataEncryption):
        self.encryption = encryption_service
    
    async def store_api_key(
        self, 
        user_id: str, 
        provider: str, 
        api_key: str
    ) -> str:
        """存储API密钥"""
        # 加密API密钥
        encrypted_key = self.encryption.encrypt(api_key)
        
        # 生成密钥ID
        key_id = secrets.token_urlsafe(16)
        
        # 存储到数据库
        await db.execute(
            """
            INSERT INTO user_credentials (id, user_id, provider, encrypted_data, created_at)
            VALUES ($1, $2, $3, $4, $5)
            """,
            key_id, user_id, provider, encrypted_key, datetime.utcnow()
        )
        
        return key_id
    
    async def retrieve_api_key(self, user_id: str, key_id: str) -> Optional[str]:
        """检索API密钥"""
        result = await db.fetch_one(
            """
            SELECT encrypted_data
            FROM user_credentials
            WHERE id = $1 AND user_id = $2
            """,
            key_id, user_id
        )
        
        if not result:
            return None
        
        try:
            return self.encryption.decrypt(result['encrypted_data'])
        except ValueError:
            logger.error(f"无法解密凭据: {key_id}")
            return None
    
    async def delete_api_key(self, user_id: str, key_id: str) -> bool:
        """删除API密钥"""
        result = await db.execute(
            """
            DELETE FROM user_credentials
            WHERE id = $1 AND user_id = $2
            """,
            key_id, user_id
        )
        
        return result > 0

# 数据脱敏
class DataMasking:
    """数据脱敏工具"""
    
    @staticmethod
    def mask_email(email: str) -> str:
        """脱敏邮箱地址"""
        if '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        if len(local) <= 2:
            masked_local = local
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
        
        return f"{masked_local}@{domain}"
    
    @staticmethod
    def mask_api_key(api_key: str) -> str:
        """脱敏API密钥"""
        if len(api_key) <= 8:
            return '*' * len(api_key)
        
        return api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
    
    @staticmethod
    def mask_phone(phone: str) -> str:
        """脱敏电话号码"""
        if len(phone) <= 4:
            return '*' * len(phone)
        
        return phone[:3] + '*' * (len(phone) - 6) + phone[-3:]
```

## 4. 运维最佳实践

### 4.1 监控和告警

#### 4.1.1 应用性能监控

```python
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from typing import Dict, Any

# Prometheus指标定义
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections_total',
    'Active connections'
)

EXECUTION_COUNT = Counter(
    'graph_executions_total',
    'Total graph executions',
    ['status', 'user_id']
)

EXECUTION_DURATION = Histogram(
    'graph_execution_duration_seconds',
    'Graph execution duration',
    ['graph_id']
)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """记录HTTP请求指标"""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_execution(self, graph_id: str, status: str, user_id: str, duration: float):
        """记录图执行指标"""
        EXECUTION_COUNT.labels(
            status=status,
            user_id=user_id
        ).inc()
        
        EXECUTION_DURATION.labels(
            graph_id=graph_id
        ).observe(duration)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_usage_percent': cpu_percent,
            'memory_usage_percent': memory.percent,
            'memory_available_bytes': memory.available,
            'disk_usage_percent': disk.percent,
            'disk_free_bytes': disk.free,
            'uptime_seconds': time.time() - self.start_time,
        }
    
    async def collect_application_metrics(self) -> Dict[str, Any]:
        """收集应用指标"""
        # 数据库连接池状态
        db_pool_stats = await self._get_db_pool_stats()
        
        # Redis连接状态
        redis_stats = await self._get_redis_stats()
        
        # 活跃执行数量
        active_executions = await self._get_active_executions_count()
        
        return {
            'database': db_pool_stats,
            'redis': redis_stats,
            'active_executions': active_executions,
        }

# FastAPI中间件集成
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class MetricsMiddleware(BaseHTTPMiddleware):
    """指标收集中间件"""
    
    def __init__(self, app, monitor: PerformanceMonitor):
        super().__init__(app)
        self.monitor = monitor
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 处理请求
        response = await call_next(request)
        
        # 记录指标
        duration = time.time() - start_time
        self.monitor.record_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration
        )
        
        return response

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查数据库连接
        await db.execute("SELECT 1")
        
        # 检查Redis连接
        await redis.ping()
        
        # 检查系统资源
        system_metrics = monitor.get_system_metrics()
        
        # 判断健康状态
        is_healthy = (
            system_metrics['cpu_usage_percent'] < 90 and
            system_metrics['memory_usage_percent'] < 90 and
            system_metrics['disk_usage_percent'] < 90
        )
        
        status = "healthy" if is_healthy else "degraded"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": system_metrics
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.get("/metrics")
async def metrics():
    """Prometheus指标端点"""
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

#### 4.1.2 告警规则配置

```yaml
# prometheus/alerts.yml
groups:
  - name: autogpt_alerts
    rules:
      # 高错误率告警
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "高错误率检测"
          description: "5分钟内HTTP 5xx错误率超过10%"
      
      # 响应时间过长告警
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "响应时间过长"
          description: "95%的请求响应时间超过2秒"
      
      # 系统资源告警
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "CPU使用率过高"
          description: "CPU使用率持续5分钟超过80%"
      
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "内存使用率过高"
          description: "内存使用率持续3分钟超过85%"
      
      # 数据库连接告警
      - alert: DatabaseConnectionIssue
        expr: up{job="autogpt-backend"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "数据库连接异常"
          description: "无法连接到数据库"
      
      # 执行队列积压告警
      - alert: ExecutionQueueBacklog
        expr: execution_queue_size > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "执行队列积压"
          description: "执行队列中待处理任务超过100个"
```

### 4.2 日志管理

#### 4.2.1 结构化日志实践

```python
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredLogger:
    """结构化日志器"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 配置处理器
        handler = logging.StreamHandler()
        formatter = StructuredFormatter()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self._log(logging.INFO, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self._log(logging.ERROR, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self._log(logging.WARNING, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """内部日志方法"""
        extra = {
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'autogpt-backend',
            'version': '1.0.0',
            **kwargs
        }
        
        self.logger.log(level, message, extra=extra)

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': getattr(record, 'timestamp', datetime.utcnow().isoformat()),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # 添加额外字段
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process']:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)

# 业务日志记录
class BusinessLogger:
    """业务日志记录器"""
    
    def __init__(self):
        self.logger = StructuredLogger('business')
    
    def log_user_action(
        self, 
        user_id: str, 
        action: str, 
        resource_type: str,
        resource_id: str,
        result: str,
        **kwargs
    ):
        """记录用户操作日志"""
        self.logger.info(
            f"用户操作: {action}",
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            result=result,
            **kwargs
        )
    
    def log_graph_execution(
        self,
        execution_id: str,
        graph_id: str,
        user_id: str,
        status: str,
        duration: Optional[float] = None,
        error: Optional[str] = None
    ):
        """记录图执行日志"""
        log_data = {
            'execution_id': execution_id,
            'graph_id': graph_id,
            'user_id': user_id,
            'status': status,
        }
        
        if duration is not None:
            log_data['duration_seconds'] = duration
        
        if error:
            log_data['error'] = error
            self.logger.error(f"图执行失败: {execution_id}", **log_data)
        else:
            self.logger.info(f"图执行{status}: {execution_id}", **log_data)
    
    def log_api_call(
        self,
        provider: str,
        endpoint: str,
        user_id: str,
        status_code: int,
        duration: float,
        cost: Optional[int] = None
    ):
        """记录API调用日志"""
        self.logger.info(
            f"API调用: {provider}/{endpoint}",
            provider=provider,
            endpoint=endpoint,
            user_id=user_id,
            status_code=status_code,
            duration_seconds=duration,
            cost_cents=cost
        )

# 使用示例
business_logger = BusinessLogger()

async def execute_graph_with_logging(graph_id: str, user_id: str, inputs: Dict):
    """带日志的图执行"""
    execution_id = generate_execution_id()
    start_time = time.time()
    
    try:
        business_logger.log_user_action(
            user_id=user_id,
            action="execute_graph",
            resource_type="graph",
            resource_id=graph_id,
            result="started",
            execution_id=execution_id
        )
        
        # 执行图
        result = await _execute_graph_internal(graph_id, inputs)
        
        duration = time.time() - start_time
        business_logger.log_graph_execution(
            execution_id=execution_id,
            graph_id=graph_id,
            user_id=user_id,
            status="completed",
            duration=duration
        )
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        business_logger.log_graph_execution(
            execution_id=execution_id,
            graph_id=graph_id,
            user_id=user_id,
            status="failed",
            duration=duration,
            error=str(e)
        )
        raise
```

### 4.3 部署和发布

#### 4.3.1 Docker化部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd --create-home --shell /bin/bash autogpt
RUN chown -R autogpt:autogpt /app
USER autogpt

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/autogpt
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=autogpt
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### 4.3.2 Kubernetes部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autogpt-backend
  labels:
    app: autogpt-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autogpt-backend
  template:
    metadata:
      labels:
        app: autogpt-backend
    spec:
      containers:
      - name: backend
        image: autogpt/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: autogpt-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: autogpt-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: autogpt-backend-service
spec:
  selector:
    app: autogpt-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autogpt-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.autogpt.com
    secretName: autogpt-tls
  rules:
  - host: api.autogpt.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: autogpt-backend-service
            port:
              number: 80
```

## 总结

本文档汇总了AutoGPT平台在开发、性能优化、安全防护和运维管理方面的最佳实践。这些经过验证的方法和策略可以帮助团队：

1. **提高代码质量**：通过规范的代码风格、完善的测试和错误处理
2. **优化系统性能**：通过数据库优化、缓存策略和前端优化
3. **增强安全防护**：通过多层安全机制、输入验证和数据加密
4. **改善运维效率**：通过监控告警、日志管理和自动化部署

遵循这些最佳实践，可以构建更加稳定、安全、高效的AutoGPT平台，为用户提供优质的AI智能体开发和运行体验。

---
