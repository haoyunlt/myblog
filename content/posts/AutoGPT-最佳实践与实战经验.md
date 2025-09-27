---
title: "AutoGPT平台最佳实践与实战经验"
date: 2024-12-19T16:00:00+08:00
draft: false
tags: ['AutoGPT', '最佳实践', '实战经验', '案例分析']
categories: ["autogpt", "实践指南"]
description: "AutoGPT平台开发和使用的最佳实践，包含实战案例、性能优化、安全建议和故障排除"
weight: 190
slug: "AutoGPT-最佳实践与实战经验"
---

# AutoGPT平台最佳实践与实战经验

## 概述

本文档汇总了AutoGPT平台开发、部署和使用过程中的最佳实践和实战经验。通过具体案例分析、性能优化技巧、安全建议和故障排除方法，帮助开发者和用户更好地使用AutoGPT平台。

## 开发最佳实践

### 1. Block开发最佳实践

#### 1.1 Block设计原则

**单一职责原则**
```python
# ✅ 好的实践：单一职责的Block
class EmailSenderBlock(Block):
    """专门负责发送邮件的Block"""
    
    class Input(BaseModel):
        to: str = Field(..., description="收件人邮箱")
        subject: str = Field(..., description="邮件主题")
        content: str = Field(..., description="邮件内容")
    
    class Output(BaseModel):
        success: bool = Field(..., description="发送是否成功")
        message_id: str = Field(..., description="邮件ID")
    
    async def run(self, input_data: Input, **kwargs) -> AsyncGenerator[Output, None]:
        # 专注于邮件发送逻辑
        result = await self.send_email(input_data)
        yield "result", self.Output(
            success=result.success,
            message_id=result.message_id
        )

# ❌ 避免：职责过多的Block
class EmailAndSMSBlock(Block):
    """不好的设计：同时处理邮件和短信"""
    # 违反单一职责原则，应该拆分为两个Block
```

**输入验证和错误处理**
```python
class DataProcessorBlock(Block):
    """数据处理Block的最佳实践"""
    
    class Input(BaseModel):
        data: List[dict] = Field(..., description="待处理数据")
        operation: str = Field(..., description="操作类型")
        
        @field_validator('data')
        @classmethod
        def validate_data(cls, v):
            if not v:
                raise ValueError("数据不能为空")
            if len(v) > 1000:
                raise ValueError("数据量过大，请分批处理")
            return v
        
        @field_validator('operation')
        @classmethod
        def validate_operation(cls, v):
            allowed_ops = ['filter', 'transform', 'aggregate']
            if v not in allowed_ops:
                raise ValueError(f"不支持的操作类型: {v}")
            return v
    
    async def run(self, input_data: Input, **kwargs) -> AsyncGenerator[tuple[str, Any], None]:
        try:
            # 详细的执行日志
            logger.info(f"开始处理 {len(input_data.data)} 条数据")
            
            # 分批处理大数据集
            batch_size = 100
            results = []
            
            for i in range(0, len(input_data.data), batch_size):
                batch = input_data.data[i:i + batch_size]
                batch_result = await self.process_batch(batch, input_data.operation)
                results.extend(batch_result)
                
                # 进度反馈
                progress = (i + len(batch)) / len(input_data.data) * 100
                yield "progress", {"percentage": progress, "processed": i + len(batch)}
            
            yield "result", {"data": results, "total": len(results)}
            
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            yield "error", f"处理失败: {str(e)}"
    
    async def process_batch(self, batch: List[dict], operation: str) -> List[dict]:
        """批处理逻辑"""
        # 实现具体的批处理逻辑
        pass
```

#### 1.2 异步和并发处理

**正确使用异步操作**
```python
class APICallBlock(Block):
    """API调用Block的异步最佳实践"""
    
    async def run(self, input_data: Input, **kwargs) -> AsyncGenerator[tuple[str, Any], None]:
        # ✅ 使用异步HTTP客户端
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    input_data.url,
                    json=input_data.payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        yield "success", result
                    else:
                        yield "error", f"API调用失败: {response.status}"
                        
            except asyncio.TimeoutError:
                yield "error", "API调用超时"
            except Exception as e:
                yield "error", f"API调用异常: {str(e)}"

class ConcurrentProcessorBlock(Block):
    """并发处理Block的最佳实践"""
    
    async def run(self, input_data: Input, **kwargs) -> AsyncGenerator[tuple[str, Any], None]:
        # ✅ 控制并发数量，避免资源耗尽
        semaphore = asyncio.Semaphore(10)  # 最多10个并发
        
        async def process_item(item):
            async with semaphore:
                return await self.process_single_item(item)
        
        # 批量并发处理
        tasks = [process_item(item) for item in input_data.items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Item {i}: {str(result)}")
            else:
                successful_results.append(result)
        
        yield "results", successful_results
        if errors:
            yield "errors", errors
```

#### 1.3 资源管理和清理

**正确的资源管理**
```python
class FileProcessorBlock(Block):
    """文件处理Block的资源管理最佳实践"""
    
    async def run(self, input_data: Input, **kwargs) -> AsyncGenerator[tuple[str, Any], None]:
        temp_files = []
        try:
            # 创建临时文件
            temp_file = await self.create_temp_file()
            temp_files.append(temp_file)
            
            # 处理文件
            result = await self.process_file(temp_file, input_data)
            yield "result", result
            
        except Exception as e:
            yield "error", str(e)
        finally:
            # ✅ 确保资源清理
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as cleanup_error:
                    logger.warning(f"清理临时文件失败: {cleanup_error}")

class DatabaseBlock(Block):
    """数据库操作Block的连接管理"""
    
    async def run(self, input_data: Input, **kwargs) -> AsyncGenerator[tuple[str, Any], None]:
        # ✅ 使用连接池和上下文管理器
        async with self.get_db_connection() as conn:
            try:
                async with conn.transaction():
                    result = await self.execute_query(conn, input_data.query)
                    yield "result", result
            except Exception as e:
                # 事务会自动回滚
                yield "error", f"数据库操作失败: {str(e)}"
```

### 2. 图设计最佳实践

#### 2.1 图结构设计

**合理的图结构**
```python
# ✅ 好的图设计：清晰的数据流
"""
输入节点 → 数据验证 → 数据处理 → 结果输出
    ↓
错误处理 → 日志记录
"""

# 图设计原则：
# 1. 单向数据流，避免循环依赖
# 2. 错误处理节点处理异常情况
# 3. 日志节点记录执行过程
# 4. 输出节点统一结果格式

class GraphDesignHelper:
    """图设计辅助工具"""
    
    @staticmethod
    def validate_graph_structure(graph: Graph) -> List[str]:
        """验证图结构的最佳实践"""
        issues = []
        
        # 检查是否有入口节点
        entry_nodes = graph.get_entry_nodes()
        if not entry_nodes:
            issues.append("图缺少入口节点")
        
        # 检查是否有出口节点
        exit_nodes = graph.get_exit_nodes()
        if not exit_nodes:
            issues.append("图缺少出口节点")
        
        # 检查循环依赖
        if GraphDesignHelper.has_cycles(graph):
            issues.append("图存在循环依赖")
        
        # 检查孤立节点
        isolated_nodes = GraphDesignHelper.find_isolated_nodes(graph)
        if isolated_nodes:
            issues.append(f"发现孤立节点: {isolated_nodes}")
        
        # 检查错误处理
        has_error_handling = any(
            'error' in node.block_id.lower() 
            for node in graph.nodes
        )
        if not has_error_handling:
            issues.append("建议添加错误处理节点")
        
        return issues
    
    @staticmethod
    def suggest_optimizations(graph: Graph) -> List[str]:
        """图优化建议"""
        suggestions = []
        
        # 检查并行处理机会
        parallel_opportunities = GraphDesignHelper.find_parallel_opportunities(graph)
        if parallel_opportunities:
            suggestions.append("可以并行处理的节点组合")
        
        # 检查缓存机会
        cacheable_nodes = GraphDesignHelper.find_cacheable_nodes(graph)
        if cacheable_nodes:
            suggestions.append("建议为重复计算节点添加缓存")
        
        # 检查资源使用
        resource_intensive_nodes = GraphDesignHelper.find_resource_intensive_nodes(graph)
        if resource_intensive_nodes:
            suggestions.append("资源密集型节点建议添加资源限制")
        
        return suggestions
```

#### 2.2 错误处理策略

**完善的错误处理**
```python
class ErrorHandlingStrategy:
    """错误处理策略最佳实践"""
    
    @staticmethod
    def create_error_handling_subgraph() -> List[Node]:
        """创建标准的错误处理子图"""
        return [
            Node(
                id="error_detector",
                block_id="ErrorDetectorBlock",
                label="错误检测",
                input_default={"error_types": ["validation", "execution", "timeout"]}
            ),
            Node(
                id="error_logger",
                block_id="LoggerBlock", 
                label="错误日志",
                input_default={"level": "ERROR"}
            ),
            Node(
                id="error_notifier",
                block_id="NotificationBlock",
                label="错误通知",
                input_default={"channels": ["email", "slack"]}
            ),
            Node(
                id="fallback_handler",
                block_id="FallbackBlock",
                label="降级处理",
                input_default={"strategy": "default_response"}
            )
        ]
    
    @staticmethod
    def create_retry_mechanism() -> Node:
        """创建重试机制节点"""
        return Node(
            id="retry_handler",
            block_id="RetryBlock",
            label="重试处理",
            input_default={
                "max_retries": 3,
                "backoff_strategy": "exponential",
                "retry_conditions": ["timeout", "rate_limit", "server_error"]
            }
        )
```

### 3. 代码质量最佳实践

#### 3.1 代码规范

**代码风格和文档**
```python
class ExampleBlock(Block):
    """
    示例Block的文档最佳实践
    
    这个Block演示了如何编写高质量的Block代码，包括：
    - 完整的类型注解
    - 详细的文档字符串
    - 合理的错误处理
    - 清晰的日志记录
    
    使用示例:
        block = ExampleBlock()
        async for output_name, output_data in block.run(input_data):
            print(f"{output_name}: {output_data}")
    
    注意事项:
        - 输入数据必须包含required_field字段
        - 处理大数据时会自动分批
        - 失败时会返回详细的错误信息
    """
    
    # 类型注解
    id: str = "example_block"
    name: str = "示例Block"
    description: str = "用于演示最佳实践的示例Block"
    
    class Input(BaseModel):
        """
        输入数据模型
        
        字段说明:
            required_field: 必需字段，不能为空
            optional_field: 可选字段，有默认值
            config: 配置参数，影响处理行为
        """
        required_field: str = Field(
            ..., 
            description="必需的输入字段",
            min_length=1,
            max_length=1000
        )
        optional_field: Optional[str] = Field(
            None,
            description="可选的输入字段"
        )
        config: dict = Field(
            default_factory=dict,
            description="配置参数"
        )
    
    class Output(BaseModel):
        """
        输出数据模型
        
        字段说明:
            result: 处理结果
            metadata: 处理元数据
            stats: 处理统计信息
        """
        result: Any = Field(..., description="处理结果")
        metadata: dict = Field(default_factory=dict, description="元数据")
        stats: dict = Field(default_factory=dict, description="统计信息")
    
    async def run(
        self, 
        input_data: Input, 
        **kwargs
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """
        执行Block逻辑
        
        参数:
            input_data: 验证过的输入数据
            **kwargs: 额外的执行参数，可能包含：
                - user_id: 用户ID
                - execution_id: 执行ID
                - credentials: 凭据信息
        
        生成:
            tuple[str, Any]: (输出名称, 输出数据) 元组
        
        异常:
            ValidationError: 输入数据验证失败
            ProcessingError: 处理过程中的错误
        """
        # 获取执行上下文
        user_id = kwargs.get('user_id')
        execution_id = kwargs.get('execution_id')
        
        # 记录开始日志
        logger.info(
            f"开始执行Block {self.id}",
            extra={
                "user_id": user_id,
                "execution_id": execution_id,
                "input_size": len(str(input_data.required_field))
            }
        )
        
        try:
            # 执行处理逻辑
            start_time = time.time()
            result = await self._process_data(input_data)
            processing_time = time.time() - start_time
            
            # 构建输出
            output = self.Output(
                result=result,
                metadata={
                    "processing_time": processing_time,
                    "input_length": len(input_data.required_field),
                    "timestamp": datetime.utcnow().isoformat()
                },
                stats={
                    "success": True,
                    "error_count": 0
                }
            )
            
            # 记录成功日志
            logger.info(
                f"Block {self.id} 执行成功",
                extra={
                    "processing_time": processing_time,
                    "output_size": len(str(result))
                }
            )
            
            yield "result", output
            
        except Exception as e:
            # 记录错误日志
            logger.error(
                f"Block {self.id} 执行失败: {str(e)}",
                extra={
                    "user_id": user_id,
                    "execution_id": execution_id,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            # 返回错误信息
            yield "error", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _process_data(self, input_data: Input) -> Any:
        """
        私有方法：处理数据的核心逻辑
        
        参数:
            input_data: 输入数据
            
        返回:
            Any: 处理结果
        """
        # 实现具体的处理逻辑
        pass
```

#### 3.2 测试最佳实践

**完整的测试覆盖**
```python
import pytest
import asyncio
from unittest.mock import Mock, patch
from your_block import ExampleBlock

class TestExampleBlock:
    """ExampleBlock的测试用例"""
    
    @pytest.fixture
    def block(self):
        """测试用的Block实例"""
        return ExampleBlock()
    
    @pytest.fixture
    def valid_input(self):
        """有效的输入数据"""
        return ExampleBlock.Input(
            required_field="test data",
            optional_field="optional",
            config={"setting": "value"}
        )
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, block, valid_input):
        """测试成功执行的情况"""
        results = []
        async for output_name, output_data in block.run(valid_input):
            results.append((output_name, output_data))
        
        # 验证输出
        assert len(results) == 1
        output_name, output_data = results[0]
        assert output_name == "result"
        assert isinstance(output_data, ExampleBlock.Output)
        assert output_data.stats["success"] is True
    
    @pytest.mark.asyncio
    async def test_input_validation(self, block):
        """测试输入验证"""
        # 测试必需字段缺失
        with pytest.raises(ValidationError):
            invalid_input = ExampleBlock.Input(
                required_field="",  # 空字符串应该失败
                optional_field="test"
            )
    
    @pytest.mark.asyncio
    async def test_error_handling(self, block, valid_input):
        """测试错误处理"""
        # 模拟处理过程中的异常
        with patch.object(block, '_process_data', side_effect=Exception("Test error")):
            results = []
            async for output_name, output_data in block.run(valid_input):
                results.append((output_name, output_data))
            
            # 验证错误输出
            assert len(results) == 1
            output_name, output_data = results[0]
            assert output_name == "error"
            assert "Test error" in output_data["error_message"]
    
    @pytest.mark.asyncio
    async def test_performance(self, block, valid_input):
        """测试性能要求"""
        import time
        
        start_time = time.time()
        results = []
        async for output_name, output_data in block.run(valid_input):
            results.append((output_name, output_data))
        end_time = time.time()
        
        # 验证执行时间在合理范围内
        execution_time = end_time - start_time
        assert execution_time < 5.0  # 应该在5秒内完成
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, block, valid_input):
        """测试并发执行"""
        # 创建多个并发任务
        tasks = []
        for i in range(10):
            task_input = ExampleBlock.Input(
                required_field=f"test data {i}",
                config={"task_id": i}
            )
            task = asyncio.create_task(self._run_block(block, task_input))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        # 验证所有任务都成功完成
        assert len(results) == 10
        for result in results:
            assert result is not None
    
    async def _run_block(self, block, input_data):
        """辅助方法：运行Block并返回结果"""
        async for output_name, output_data in block.run(input_data):
            if output_name == "result":
                return output_data
        return None

# 集成测试
class TestBlockIntegration:
    """Block集成测试"""
    
    @pytest.mark.asyncio
    async def test_block_in_graph(self):
        """测试Block在图中的执行"""
        # 创建包含ExampleBlock的测试图
        graph = create_test_graph_with_example_block()
        
        # 执行图
        execution_result = await execute_graph(graph, test_inputs)
        
        # 验证执行结果
        assert execution_result.status == "completed"
        assert "result" in execution_result.outputs
    
    @pytest.mark.asyncio
    async def test_block_with_credentials(self):
        """测试需要凭据的Block"""
        # 模拟凭据
        mock_credentials = {
            "api_provider": Mock(spec=["get_token", "is_valid"])
        }
        mock_credentials["api_provider"].is_valid.return_value = True
        mock_credentials["api_provider"].get_token.return_value = "test_token"
        
        # 使用凭据执行Block
        block = ExampleBlock()
        input_data = ExampleBlock.Input(required_field="test")
        
        results = []
        async for output_name, output_data in block.run(
            input_data, 
            credentials=mock_credentials
        ):
            results.append((output_name, output_data))
        
        # 验证凭据被正确使用
        mock_credentials["api_provider"].is_valid.assert_called_once()
```

## 架构设计原则

### 1. 微服务架构原则

#### 1.1 服务拆分策略

**按业务能力拆分**
```python
# ✅ 好的服务拆分
"""
用户服务 (UserService)
- 用户注册、登录、信息管理
- 权限和角色管理

图服务 (GraphService)  
- 图的创建、编辑、版本管理
- 图结构验证和优化

执行服务 (ExecutionService)
- 图执行调度和管理
- 执行状态跟踪

Block服务 (BlockService)
- Block注册和管理
- Block元数据和schema

通知服务 (NotificationService)
- 邮件、短信、推送通知
- 通知模板和规则管理
"""

class ServiceBoundaryPrinciples:
    """服务边界设计原则"""
    
    @staticmethod
    def evaluate_service_design(service_name: str, responsibilities: List[str]) -> dict:
        """评估服务设计的合理性"""
        evaluation = {
            "service_name": service_name,
            "score": 0,
            "issues": [],
            "suggestions": []
        }
        
        # 检查单一职责
        if len(responsibilities) > 5:
            evaluation["issues"].append("职责过多，建议拆分")
            evaluation["score"] -= 20
        
        # 检查职责相关性
        if not ServiceBoundaryPrinciples._are_responsibilities_related(responsibilities):
            evaluation["issues"].append("职责关联性不强")
            evaluation["score"] -= 15
        
        # 检查数据一致性
        if ServiceBoundaryPrinciples._requires_distributed_transaction(responsibilities):
            evaluation["issues"].append("可能需要分布式事务")
            evaluation["suggestions"].append("考虑使用Saga模式")
        
        return evaluation
```

#### 1.2 服务间通信

**异步消息通信**
```python
class EventDrivenCommunication:
    """事件驱动的服务间通信"""
    
    def __init__(self, message_broker):
        self.broker = message_broker
        self.event_handlers = {}
    
    async def publish_event(self, event_type: str, event_data: dict):
        """发布事件"""
        event = {
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.utcnow().isoformat(),
            "event_id": str(uuid.uuid4())
        }
        
        await self.broker.publish(
            exchange="events",
            routing_key=event_type,
            message=json.dumps(event)
        )
        
        logger.info(f"Published event: {event_type}")
    
    async def subscribe_to_event(self, event_type: str, handler: callable):
        """订阅事件"""
        self.event_handlers[event_type] = handler
        
        await self.broker.subscribe(
            queue=f"{event_type}_queue",
            callback=self._handle_event
        )
    
    async def _handle_event(self, message):
        """处理接收到的事件"""
        try:
            event = json.loads(message.body)
            event_type = event["event_type"]
            
            if event_type in self.event_handlers:
                handler = self.event_handlers[event_type]
                await handler(event["event_data"])
            
            await message.ack()
            
        except Exception as e:
            logger.error(f"Event handling failed: {e}")
            await message.nack(requeue=True)

# 使用示例
class GraphService:
    """图服务的事件处理示例"""
    
    def __init__(self):
        self.event_bus = EventDrivenCommunication(message_broker)
        self._setup_event_handlers()
    
    async def _setup_event_handlers(self):
        """设置事件处理器"""
        await self.event_bus.subscribe_to_event(
            "user.deleted", 
            self._handle_user_deleted
        )
        await self.event_bus.subscribe_to_event(
            "execution.completed",
            self._handle_execution_completed
        )
    
    async def _handle_user_deleted(self, event_data):
        """处理用户删除事件"""
        user_id = event_data["user_id"]
        
        # 删除用户的所有图
        await self.delete_user_graphs(user_id)
        
        # 发布图删除事件
        await self.event_bus.publish_event(
            "graphs.user_deleted",
            {"user_id": user_id, "deleted_count": deleted_count}
        )
    
    async def create_graph(self, graph_data):
        """创建图并发布事件"""
        graph = await self._create_graph_in_db(graph_data)
        
        # 发布图创建事件
        await self.event_bus.publish_event(
            "graph.created",
            {
                "graph_id": graph.id,
                "user_id": graph.user_id,
                "graph_name": graph.name
            }
        )
        
        return graph
```

### 2. 数据一致性策略

#### 2.1 分布式事务处理

**Saga模式实现**
```python
class SagaOrchestrator:
    """Saga事务协调器"""
    
    def __init__(self):
        self.steps = []
        self.compensations = []
    
    def add_step(self, action: callable, compensation: callable):
        """添加事务步骤"""
        self.steps.append(action)
        self.compensations.append(compensation)
    
    async def execute(self) -> bool:
        """执行Saga事务"""
        executed_steps = []
        
        try:
            # 顺序执行所有步骤
            for i, step in enumerate(self.steps):
                result = await step()
                executed_steps.append(i)
                
                if not result:
                    raise Exception(f"Step {i} failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Saga execution failed: {e}")
            
            # 执行补偿操作
            await self._compensate(executed_steps)
            return False
    
    async def _compensate(self, executed_steps: List[int]):
        """执行补偿操作"""
        # 按相反顺序执行补偿
        for step_index in reversed(executed_steps):
            try:
                compensation = self.compensations[step_index]
                await compensation()
            except Exception as e:
                logger.error(f"Compensation {step_index} failed: {e}")

# 使用示例：用户注册的分布式事务
class UserRegistrationSaga:
    """用户注册的Saga事务"""
    
    async def register_user(self, user_data: dict) -> bool:
        saga = SagaOrchestrator()
        
        # 步骤1：创建用户账户
        saga.add_step(
            action=lambda: self._create_user_account(user_data),
            compensation=lambda: self._delete_user_account(user_data["user_id"])
        )
        
        # 步骤2：初始化用户积分
        saga.add_step(
            action=lambda: self._initialize_user_credits(user_data["user_id"]),
            compensation=lambda: self._delete_user_credits(user_data["user_id"])
        )
        
        # 步骤3：发送欢迎邮件
        saga.add_step(
            action=lambda: self._send_welcome_email(user_data["email"]),
            compensation=lambda: self._send_cancellation_email(user_data["email"])
        )
        
        # 步骤4：创建默认工作空间
        saga.add_step(
            action=lambda: self._create_default_workspace(user_data["user_id"]),
            compensation=lambda: self._delete_user_workspace(user_data["user_id"])
        )
        
        return await saga.execute()
```

#### 2.2 最终一致性处理

**事件溯源和CQRS**
```python
class EventStore:
    """事件存储"""
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
    
    async def append_event(self, stream_id: str, event: dict):
        """追加事件到流"""
        event_with_metadata = {
            **event,
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "stream_id": stream_id
        }
        
        await self.storage.append(stream_id, event_with_metadata)
        
        # 发布事件到消息总线
        await self._publish_event(event_with_metadata)
    
    async def get_events(self, stream_id: str, from_version: int = 0) -> List[dict]:
        """获取流中的事件"""
        return await self.storage.get_events(stream_id, from_version)
    
    async def _publish_event(self, event: dict):
        """发布事件到消息总线"""
        # 实现事件发布逻辑
        pass

class GraphAggregate:
    """图聚合根"""
    
    def __init__(self, graph_id: str):
        self.graph_id = graph_id
        self.version = 0
        self.uncommitted_events = []
        self.state = GraphState()
    
    def create_graph(self, name: str, user_id: str):
        """创建图"""
        event = {
            "event_type": "GraphCreated",
            "data": {
                "graph_id": self.graph_id,
                "name": name,
                "user_id": user_id
            }
        }
        
        self._apply_event(event)
        self.uncommitted_events.append(event)
    
    def add_node(self, node_data: dict):
        """添加节点"""
        event = {
            "event_type": "NodeAdded",
            "data": {
                "graph_id": self.graph_id,
                "node_id": node_data["id"],
                "node_data": node_data
            }
        }
        
        self._apply_event(event)
        self.uncommitted_events.append(event)
    
    def _apply_event(self, event: dict):
        """应用事件到状态"""
        event_type = event["event_type"]
        data = event["data"]
        
        if event_type == "GraphCreated":
            self.state.name = data["name"]
            self.state.user_id = data["user_id"]
            self.state.created_at = datetime.utcnow()
        
        elif event_type == "NodeAdded":
            self.state.nodes[data["node_id"]] = data["node_data"]
        
        self.version += 1
    
    async def save(self, event_store: EventStore):
        """保存未提交的事件"""
        for event in self.uncommitted_events:
            await event_store.append_event(self.graph_id, event)
        
        self.uncommitted_events.clear()

class GraphProjection:
    """图投影（读模型）"""
    
    def __init__(self, read_db):
        self.read_db = read_db
    
    async def handle_graph_created(self, event_data: dict):
        """处理图创建事件"""
        graph_record = {
            "id": event_data["graph_id"],
            "name": event_data["name"],
            "user_id": event_data["user_id"],
            "created_at": datetime.utcnow(),
            "node_count": 0
        }
        
        await self.read_db.insert("graphs", graph_record)
    
    async def handle_node_added(self, event_data: dict):
        """处理节点添加事件"""
        await self.read_db.increment(
            "graphs",
            {"id": event_data["graph_id"]},
            {"node_count": 1}
        )
        
        # 更新节点索引
        node_record = {
            "id": event_data["node_id"],
            "graph_id": event_data["graph_id"],
            **event_data["node_data"]
        }
        
        await self.read_db.insert("nodes", node_record)
```

## 性能优化实战

### 1. 数据库优化

#### 1.1 查询优化

**索引策略**
```sql
-- ✅ 高效的索引设计

-- 1. 复合索引：按查询频率和选择性排序
CREATE INDEX idx_graph_user_active ON graphs(user_id, is_active, created_at DESC);

-- 2. 部分索引：只为活跃数据创建索引
CREATE INDEX idx_active_executions ON graph_executions(status, created_at) 
WHERE status IN ('queued', 'running');

-- 3. 表达式索引：为计算字段创建索引
CREATE INDEX idx_execution_duration ON graph_executions(
    EXTRACT(EPOCH FROM (ended_at - started_at))
) WHERE ended_at IS NOT NULL;

-- 4. 覆盖索引：包含查询所需的所有字段
CREATE INDEX idx_graph_list_covering ON graphs(user_id, is_active) 
INCLUDE (id, name, created_at, updated_at);
```

**查询优化技巧**
```python
class OptimizedGraphRepository:
    """优化的图数据访问层"""
    
    async def get_user_graphs_paginated(
        self, 
        user_id: str, 
        page: int = 1, 
        page_size: int = 20,
        include_inactive: bool = False
    ) -> dict:
        """分页获取用户图列表（优化版本）"""
        
        # ✅ 使用覆盖索引避免回表查询
        base_query = """
        SELECT id, name, description, is_active, created_at, updated_at,
               (SELECT COUNT(*) FROM nodes WHERE graph_id = g.id) as node_count
        FROM graphs g
        WHERE user_id = $1
        """
        
        conditions = []
        params = [user_id]
        
        if not include_inactive:
            conditions.append("AND is_active = $2")
            params.append(True)
        
        # ✅ 使用游标分页而不是OFFSET
        if page > 1:
            # 假设前端传递了last_created_at
            conditions.append("AND created_at < $" + str(len(params) + 1))
            params.append(last_created_at)
        
        query = base_query + " ".join(conditions) + """
        ORDER BY created_at DESC
        LIMIT $""" + str(len(params) + 1)
        params.append(page_size)
        
        # 执行查询
        graphs = await self.db.fetch(query, *params)
        
        # ✅ 批量获取相关数据，避免N+1查询
        graph_ids = [g['id'] for g in graphs]
        if graph_ids:
            # 批量获取执行统计
            exec_stats = await self._get_execution_stats_batch(graph_ids)
            
            # 合并数据
            for graph in graphs:
                graph['execution_stats'] = exec_stats.get(graph['id'], {})
        
        return {
            'graphs': graphs,
            'has_more': len(graphs) == page_size
        }
    
    async def _get_execution_stats_batch(self, graph_ids: List[str]) -> dict:
        """批量获取执行统计"""
        query = """
        SELECT 
            graph_id,
            COUNT(*) as total_executions,
            COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
            AVG(EXTRACT(EPOCH FROM (ended_at - started_at))) FILTER (WHERE ended_at IS NOT NULL) as avg_duration
        FROM graph_executions
        WHERE graph_id = ANY($1)
        GROUP BY graph_id
        """
        
        results = await self.db.fetch(query, graph_ids)
        return {r['graph_id']: dict(r) for r in results}
```

#### 1.2 连接池优化

**连接池配置**
```python
class DatabaseConnectionManager:
    """数据库连接管理器"""
    
    def __init__(self):
        self.pools = {}
    
    async def create_pool(self, database_url: str, **kwargs):
        """创建优化的连接池"""
        pool_config = {
            # ✅ 连接池大小配置
            'min_size': 5,          # 最小连接数
            'max_size': 20,         # 最大连接数
            
            # ✅ 连接超时配置
            'command_timeout': 30,   # 命令超时
            'server_settings': {
                'application_name': 'autogpt_platform',
                'tcp_keepalives_idle': '600',
                'tcp_keepalives_interval': '30',
                'tcp_keepalives_count': '3',
            },
            
            # ✅ 连接验证
            'init': self._init_connection,
            
            **kwargs
        }
        
        pool = await asyncpg.create_pool(database_url, **pool_config)
        self.pools['default'] = pool
        return pool
    
    async def _init_connection(self, conn):
        """初始化连接"""
        # 设置连接级别的配置
        await conn.execute("SET timezone = 'UTC'")
        await conn.execute("SET statement_timeout = '30s'")
        
        # 预编译常用查询
        await conn.prepare("SELECT * FROM graphs WHERE id = $1")
        await conn.prepare("SELECT * FROM users WHERE id = $1")
    
    async def get_connection(self, pool_name: str = 'default'):
        """获取连接"""
        pool = self.pools.get(pool_name)
        if not pool:
            raise ValueError(f"Pool {pool_name} not found")
        
        return pool.acquire()
    
    async def execute_with_retry(self, query: str, *args, max_retries: int = 3):
        """带重试的查询执行"""
        for attempt in range(max_retries):
            try:
                async with self.get_connection() as conn:
                    return await conn.fetch(query, *args)
            
            except (asyncpg.ConnectionDoesNotExistError, 
                    asyncpg.InterfaceError) as e:
                if attempt == max_retries - 1:
                    raise
                
                logger.warning(f"Database connection error, retrying: {e}")
                await asyncio.sleep(2 ** attempt)  # 指数退避
```

### 2. 缓存策略

#### 2.1 多级缓存

**缓存层次设计**
```python
class MultiLevelCache:
    """多级缓存实现"""
    
    def __init__(self):
        # L1: 内存缓存（最快）
        self.l1_cache = {}
        self.l1_max_size = 1000
        self.l1_ttl = 300  # 5分钟
        
        # L2: Redis缓存（中等速度）
        self.l2_cache = None  # Redis客户端
        self.l2_ttl = 3600    # 1小时
        
        # L3: 数据库（最慢但最可靠）
        self.l3_cache = None  # 数据库连接
    
    async def get(self, key: str, fetch_func: callable = None):
        """多级缓存获取"""
        # L1缓存检查
        l1_value = self._get_from_l1(key)
        if l1_value is not None:
            return l1_value
        
        # L2缓存检查
        l2_value = await self._get_from_l2(key)
        if l2_value is not None:
            # 回填L1缓存
            self._set_to_l1(key, l2_value)
            return l2_value
        
        # L3缓存或原始数据源
        if fetch_func:
            l3_value = await fetch_func()
            if l3_value is not None:
                # 回填所有缓存层
                await self._set_to_l2(key, l3_value)
                self._set_to_l1(key, l3_value)
                return l3_value
        
        return None
    
    async def set(self, key: str, value: any, ttl: int = None):
        """设置多级缓存"""
        # 设置所有缓存层
        self._set_to_l1(key, value)
        await self._set_to_l2(key, value, ttl or self.l2_ttl)
    
    async def invalidate(self, key: str):
        """失效所有缓存层"""
        self._remove_from_l1(key)
        await self._remove_from_l2(key)
    
    def _get_from_l1(self, key: str):
        """从L1缓存获取"""
        entry = self.l1_cache.get(key)
        if entry and time.time() - entry['timestamp'] < self.l1_ttl:
            return entry['value']
        elif entry:
            # 过期删除
            del self.l1_cache[key]
        return None
    
    def _set_to_l1(self, key: str, value: any):
        """设置L1缓存"""
        # LRU淘汰
        if len(self.l1_cache) >= self.l1_max_size:
            oldest_key = min(
                self.l1_cache.keys(),
                key=lambda k: self.l1_cache[k]['timestamp']
            )
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

# 使用示例
class CachedGraphService:
    """带缓存的图服务"""
    
    def __init__(self):
        self.cache = MultiLevelCache()
        self.db = DatabaseConnectionManager()
    
    async def get_graph(self, graph_id: str) -> dict:
        """获取图（带缓存）"""
        cache_key = f"graph:{graph_id}"
        
        return await self.cache.get(
            cache_key,
            fetch_func=lambda: self._fetch_graph_from_db(graph_id)
        )
    
    async def update_graph(self, graph_id: str, updates: dict) -> dict:
        """更新图并失效缓存"""
        # 更新数据库
        updated_graph = await self._update_graph_in_db(graph_id, updates)
        
        # 失效相关缓存
        await self.cache.invalidate(f"graph:{graph_id}")
        await self.cache.invalidate(f"user_graphs:{updated_graph['user_id']}")
        
        # 预热缓存
        await self.cache.set(f"graph:{graph_id}", updated_graph)
        
        return updated_graph
```

#### 2.2 缓存预热和失效策略

**智能缓存管理**
```python
class CacheWarmupManager:
    """缓存预热管理器"""
    
    def __init__(self, cache_service, analytics_service):
        self.cache = cache_service
        self.analytics = analytics_service
    
    async def warmup_popular_content(self):
        """预热热门内容"""
        # 获取热门图列表
        popular_graphs = await self.analytics.get_popular_graphs(limit=100)
        
        # 并发预热
        tasks = []
        for graph_id in popular_graphs:
            task = asyncio.create_task(self._warmup_graph(graph_id))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _warmup_graph(self, graph_id: str):
        """预热单个图的缓存"""
        try:
            # 预热图基本信息
            await self.cache.get(f"graph:{graph_id}")
            
            # 预热图的节点信息
            await self.cache.get(f"graph_nodes:{graph_id}")
            
            # 预热最近的执行记录
            await self.cache.get(f"graph_executions:{graph_id}")
            
        except Exception as e:
            logger.warning(f"Failed to warmup graph {graph_id}: {e}")
    
    async def schedule_cache_cleanup(self):
        """定期缓存清理"""
        while True:
            try:
                # 清理过期的缓存项
                await self._cleanup_expired_cache()
                
                # 清理低命中率的缓存项
                await self._cleanup_low_hit_rate_cache()
                
                # 等待下次清理
                await asyncio.sleep(3600)  # 每小时清理一次
                
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
                await asyncio.sleep(300)  # 出错时5分钟后重试

class CacheInvalidationStrategy:
    """缓存失效策略"""
    
    def __init__(self, cache_service, event_bus):
        self.cache = cache_service
        self.event_bus = event_bus
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """设置事件处理器"""
        self.event_bus.subscribe("graph.updated", self._handle_graph_updated)
        self.event_bus.subscribe("user.updated", self._handle_user_updated)
        self.event_bus.subscribe("execution.completed", self._handle_execution_completed)
    
    async def _handle_graph_updated(self, event_data: dict):
        """处理图更新事件"""
        graph_id = event_data["graph_id"]
        user_id = event_data["user_id"]
        
        # 失效相关缓存
        invalidation_keys = [
            f"graph:{graph_id}",
            f"graph_nodes:{graph_id}",
            f"user_graphs:{user_id}",
            f"graph_stats:{graph_id}"
        ]
        
        for key in invalidation_keys:
            await self.cache.invalidate(key)
    
    async def _handle_execution_completed(self, event_data: dict):
        """处理执行完成事件"""
        graph_id = event_data["graph_id"]
        user_id = event_data["user_id"]
        
        # 失效统计相关的缓存
        await self.cache.invalidate(f"graph_stats:{graph_id}")
        await self.cache.invalidate(f"user_stats:{user_id}")
        await self.cache.invalidate(f"execution_history:{graph_id}")
```

### 3. 异步处理优化

#### 3.1 任务队列优化

**高效的任务调度**
```python
class OptimizedTaskQueue:
    """优化的任务队列"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.worker_pools = {}
        self.metrics = TaskQueueMetrics()
    
    async def enqueue_task(
        self, 
        task_type: str, 
        task_data: dict, 
        priority: int = 0,
        delay: int = 0
    ):
        """入队任务"""
        task = {
            "id": str(uuid.uuid4()),
            "type": task_type,
            "data": task_data,
            "priority": priority,
            "created_at": time.time(),
            "execute_at": time.time() + delay
        }
        
        # 根据优先级选择队列
        queue_name = f"tasks:{task_type}:priority_{priority}"
        
        if delay > 0:
            # 延迟任务放入延迟队列
            await self.redis.zadd(
                "delayed_tasks",
                {json.dumps(task): task["execute_at"]}
            )
        else:
            # 立即执行的任务
            await self.redis.lpush(queue_name, json.dumps(task))
        
        self.metrics.task_enqueued(task_type, priority)
        return task["id"]
    
    async def start_worker_pool(self, task_type: str, worker_count: int = 5):
        """启动工作进程池"""
        workers = []
        for i in range(worker_count):
            worker = TaskWorker(
                worker_id=f"{task_type}_worker_{i}",
                task_type=task_type,
                redis_client=self.redis,
                metrics=self.metrics
            )
            workers.append(worker)
            asyncio.create_task(worker.start())
        
        self.worker_pools[task_type] = workers
    
    async def start_delayed_task_processor(self):
        """启动延迟任务处理器"""
        while True:
            try:
                current_time = time.time()
                
                # 获取到期的延迟任务
                ready_tasks = await self.redis.zrangebyscore(
                    "delayed_tasks",
                    0,
                    current_time,
                    withscores=True
                )
                
                for task_data, score in ready_tasks:
                    task = json.loads(task_data)
                    
                    # 移动到对应的执行队列
                    queue_name = f"tasks:{task['type']}:priority_{task['priority']}"
                    await self.redis.lpush(queue_name, task_data)
                    
                    # 从延迟队列中移除
                    await self.redis.zrem("delayed_tasks", task_data)
                
                await asyncio.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"Delayed task processor error: {e}")
                await asyncio.sleep(5)

class TaskWorker:
    """任务工作进程"""
    
    def __init__(self, worker_id: str, task_type: str, redis_client, metrics):
        self.worker_id = worker_id
        self.task_type = task_type
        self.redis = redis_client
        self.metrics = metrics
        self.is_running = False
    
    async def start(self):
        """启动工作进程"""
        self.is_running = True
        logger.info(f"Worker {self.worker_id} started")
        
        while self.is_running:
            try:
                # 按优先级获取任务
                task_data = await self._get_next_task()
                
                if task_data:
                    await self._process_task(task_data)
                else:
                    # 没有任务时短暂休眠
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _get_next_task(self) -> dict:
        """获取下一个任务（按优先级）"""
        # 按优先级顺序检查队列
        for priority in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
            queue_name = f"tasks:{self.task_type}:priority_{priority}"
            
            task_data = await self.redis.brpop(queue_name, timeout=1)
            if task_data:
                return json.loads(task_data[1])
        
        return None
    
    async def _process_task(self, task_data: dict):
        """处理任务"""
        task_id = task_data["id"]
        start_time = time.time()
        
        try:
            # 记录任务开始
            self.metrics.task_started(self.task_type, task_id)
            
            # 执行任务
            result = await self._execute_task(task_data)
            
            # 记录成功
            duration = time.time() - start_time
            self.metrics.task_completed(self.task_type, task_id, duration)
            
            logger.info(f"Task {task_id} completed in {duration:.2f}s")
            
        except Exception as e:
            # 记录失败
            duration = time.time() - start_time
            self.metrics.task_failed(self.task_type, task_id, duration, str(e))
            
            # 重试逻辑
            await self._handle_task_failure(task_data, e)
    
    async def _execute_task(self, task_data: dict):
        """执行具体任务"""
        task_type = task_data["type"]
        task_payload = task_data["data"]
        
        # 根据任务类型分发到不同的处理器
        if task_type == "graph_execution":
            return await self._execute_graph_task(task_payload)
        elif task_type == "notification":
            return await self._execute_notification_task(task_payload)
        elif task_type == "data_processing":
            return await self._execute_data_processing_task(task_payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
```

## 安全防护策略

### 1. 认证和授权

#### 1.1 多层认证体系

**JWT + API Key双重认证**
```python
class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET")
        self.api_key_store = APIKeyStore()
        self.rate_limiter = RateLimiter()
    
    async def authenticate_request(self, request: Request) -> AuthContext:
        """请求认证"""
        auth_context = AuthContext()
        
        # 1. JWT认证
        jwt_token = self._extract_jwt_token(request)
        if jwt_token:
            jwt_payload = await self._validate_jwt_token(jwt_token)
            if jwt_payload:
                auth_context.user_id = jwt_payload["user_id"]
                auth_context.user_role = jwt_payload["role"]
                auth_context.auth_method = "jwt"
        
        # 2. API Key认证（备选或补充）
        api_key = self._extract_api_key(request)
        if api_key and not auth_context.user_id:
            api_key_info = await self.api_key_store.validate_key(api_key)
            if api_key_info:
                auth_context.user_id = api_key_info["user_id"]
                auth_context.user_role = api_key_info["role"]
                auth_context.auth_method = "api_key"
                auth_context.api_key_id = api_key_info["id"]
        
        # 3. 限流检查
        if auth_context.user_id:
            rate_limit_key = f"user:{auth_context.user_id}"
            if not await self.rate_limiter.check_limit(rate_limit_key):
                raise RateLimitExceededError("Rate limit exceeded")
        
        return auth_context
    
    async def authorize_action(
        self, 
        auth_context: AuthContext, 
        resource: str, 
        action: str
    ) -> bool:
        """动作授权"""
        # 管理员拥有所有权限
        if auth_context.user_role == "admin":
            return True
        
        # 检查资源所有权
        if resource.startswith("graph:"):
            graph_id = resource.split(":")[1]
            graph_owner = await self._get_graph_owner(graph_id)
            
            if graph_owner == auth_context.user_id:
                return True
            
            # 检查共享权限
            if action == "read":
                return await self._check_graph_sharing(graph_id, auth_context.user_id)
        
        # 检查基于角色的权限
        return await self._check_rbac_permission(
            auth_context.user_role, 
            resource, 
            action
        )

class SecureAPIKeyManager:
    """安全的API Key管理"""
    
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
    
    async def generate_api_key(self, user_id: str, permissions: List[str]) -> dict:
        """生成API Key"""
        # 生成随机密钥
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # 创建密钥记录
        api_key_record = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "key_hash": key_hash,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "last_used_at": None,
            "is_active": True,
            "usage_count": 0
        }
        
        # 存储到数据库（只存储哈希）
        await self._store_api_key_record(api_key_record)
        
        # 返回完整密钥（仅此一次）
        return {
            "api_key": f"agpt_{raw_key}",
            "key_id": api_key_record["id"],
            "permissions": permissions
        }
    
    async def validate_api_key(self, api_key: str) -> dict:
        """验证API Key"""
        if not api_key.startswith("agpt_"):
            return None
        
        raw_key = api_key[5:]  # 移除前缀
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # 查找密钥记录
        key_record = await self._get_api_key_by_hash(key_hash)
        
        if not key_record or not key_record["is_active"]:
            return None
        
        # 更新使用统计
        await self._update_key_usage(key_record["id"])
        
        return {
            "user_id": key_record["user_id"],
            "permissions": key_record["permissions"],
            "key_id": key_record["id"]
        }
```

#### 1.2 权限控制模型

**基于角色和资源的访问控制（RBAC + ABAC）**
```python
class PermissionManager:
    """权限管理器"""
    
    def __init__(self):
        self.role_permissions = self._load_role_permissions()
        self.resource_policies = self._load_resource_policies()
    
    def _load_role_permissions(self) -> dict:
        """加载角色权限配置"""
        return {
            "admin": ["*"],  # 管理员拥有所有权限
            "user": [
                "graph:create",
                "graph:read:own",
                "graph:update:own", 
                "graph:delete:own",
                "execution:create:own",
                "execution:read:own"
            ],
            "viewer": [
                "graph:read:shared",
                "execution:read:shared"
            ]
        }
    
    def _load_resource_policies(self) -> dict:
        """加载资源策略"""
        return {
            "graph": {
                "ownership_field": "user_id",
                "sharing_enabled": True,
                "public_read": False
            },
            "execution": {
                "ownership_field": "user_id", 
                "sharing_enabled": True,
                "public_read": False
            },
            "user": {
                "ownership_field": "id",
                "sharing_enabled": False,
                "public_read": False
            }
        }
    
    async def check_permission(
        self,
        user_id: str,
        user_role: str,
        resource_type: str,
        resource_id: str,
        action: str
    ) -> bool:
        """检查权限"""
        
        # 1. 检查角色权限
        role_perms = self.role_permissions.get(user_role, [])
        
        # 管理员拥有所有权限
        if "*" in role_perms:
            return True
        
        # 2. 构建权限字符串
        permission_patterns = [
            f"{resource_type}:{action}",
            f"{resource_type}:{action}:own",
            f"{resource_type}:{action}:shared"
        ]
        
        # 3. 检查基本权限
        for pattern in permission_patterns:
            if pattern in role_perms:
                # 如果是own权限，需要检查所有权
                if pattern.endswith(":own"):
                    return await self._check_ownership(
                        user_id, resource_type, resource_id
                    )
                
                # 如果是shared权限，需要检查共享
                elif pattern.endswith(":shared"):
                    return await self._check_sharing(
                        user_id, resource_type, resource_id
                    )
                
                # 基本权限直接通过
                else:
                    return True
        
        return False
    
    async def _check_ownership(
        self, 
        user_id: str, 
        resource_type: str, 
        resource_id: str
    ) -> bool:
        """检查资源所有权"""
        policy = self.resource_policies.get(resource_type)
        if not policy:
            return False
        
        ownership_field = policy["ownership_field"]
        
        # 查询资源所有者
        resource = await self._get_resource(resource_type, resource_id)
        if not resource:
            return False
        
        return resource.get(ownership_field) == user_id
    
    async def _check_sharing(
        self,
        user_id: str,
        resource_type: str, 
        resource_id: str
    ) -> bool:
        """检查资源共享权限"""
        policy = self.resource_policies.get(resource_type)
        if not policy or not policy["sharing_enabled"]:
            return False
        
        # 检查是否在共享列表中
        return await self._is_resource_shared_with_user(
            resource_type, resource_id, user_id
        )

# 权限装饰器
def require_permission(resource_type: str, action: str):
    """权限检查装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 从请求中获取认证信息
            request = kwargs.get('request') or args[0]
            auth_context = getattr(request.state, 'auth_context', None)
            
            if not auth_context or not auth_context.user_id:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # 获取资源ID
            resource_id = kwargs.get('resource_id') or kwargs.get('graph_id') or kwargs.get('id')
            
            # 检查权限
            permission_manager = PermissionManager()
            has_permission = await permission_manager.check_permission(
                user_id=auth_context.user_id,
                user_role=auth_context.user_role,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action
            )
            
            if not has_permission:
                raise HTTPException(status_code=403, detail="Permission denied")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# 使用示例
@require_permission("graph", "update")
async def update_graph(graph_id: str, updates: dict, request: Request):
    """更新图（需要更新权限）"""
    # 实现更新逻辑
    pass
```

### 2. 数据安全

#### 2.1 敏感数据加密

**端到端加密方案**
```python
class DataEncryptionManager:
    """数据加密管理器"""
    
    def __init__(self):
        # 主密钥（从环境变量或密钥管理服务获取）
        self.master_key = self._get_master_key()
        self.fernet = Fernet(self.master_key)
        
        # 字段级加密配置
        self.encrypted_fields = {
            "users": ["email", "phone"],
            "credentials": ["access_token", "refresh_token", "api_key"],
            "graphs": ["sensitive_data"],
            "executions": ["input_data", "output_data"]
        }
    
    def _get_master_key(self) -> bytes:
        """获取主加密密钥"""
        key_str = os.getenv("ENCRYPTION_MASTER_KEY")
        if not key_str:
            # 在生产环境中，应该从密钥管理服务获取
            raise ValueError("Master encryption key not configured")
        
        return base64.urlsafe_b64decode(key_str.encode())
    
    def encrypt_field(self, table: str, field: str, value: str) -> str:
        """加密字段值"""
        if not self._should_encrypt_field(table, field):
            return value
        
        if not value:
            return value
        
        # 加密数据
        encrypted_bytes = self.fernet.encrypt(value.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
    
    def decrypt_field(self, table: str, field: str, encrypted_value: str) -> str:
        """解密字段值"""
        if not self._should_encrypt_field(table, field):
            return encrypted_value
        
        if not encrypted_value:
            return encrypted_value
        
        try:
            # 解密数据
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode('utf-8'))
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decrypt field {table}.{field}: {e}")
            return encrypted_value
    
    def _should_encrypt_field(self, table: str, field: str) -> bool:
        """检查字段是否需要加密"""
        return field in self.encrypted_fields.get(table, [])
    
    def encrypt_record(self, table: str, record: dict) -> dict:
        """加密记录中的敏感字段"""
        encrypted_record = record.copy()
        
        for field, value in record.items():
            if isinstance(value, str):
                encrypted_record[field] = self.encrypt_field(table, field, value)
        
        return encrypted_record
    
    def decrypt_record(self, table: str, record: dict) -> dict:
        """解密记录中的敏感字段"""
        decrypted_record = record.copy()
        
        for field, value in record.items():
            if isinstance(value, str):
                decrypted_record[field] = self.decrypt_field(table, field, value)
        
        return decrypted_record

class SecureCredentialsStore:
    """安全凭据存储"""
    
    def __init__(self):
        self.encryption_manager = DataEncryptionManager()
        self.db = DatabaseManager()
    
    async def store_credentials(
        self, 
        user_id: str, 
        provider: str, 
        credentials: dict
    ) -> str:
        """存储加密的凭据"""
        
        # 生成凭据ID
        credential_id = str(uuid.uuid4())
        
        # 加密敏感字段
        encrypted_credentials = {}
        for key, value in credentials.items():
            if key in ["access_token", "refresh_token", "api_key", "client_secret"]:
                encrypted_credentials[key] = self.encryption_manager.encrypt_field(
                    "credentials", key, str(value)
                )
            else:
                encrypted_credentials[key] = value
        
        # 存储到数据库
        record = {
            "id": credential_id,
            "user_id": user_id,
            "provider": provider,
            "credentials": json.dumps(encrypted_credentials),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await self.db.insert("user_credentials", record)
        
        # 记录审计日志
        await self._log_credential_access(
            user_id, credential_id, "store", provider
        )
        
        return credential_id
    
    async def retrieve_credentials(
        self, 
        user_id: str, 
        credential_id: str
    ) -> dict:
        """检索并解密凭据"""
        
        # 从数据库获取
        record = await self.db.get_one(
            "user_credentials",
            {"id": credential_id, "user_id": user_id}
        )
        
        if not record:
            return None
        
        # 解密凭据
        encrypted_credentials = json.loads(record["credentials"])
        decrypted_credentials = {}
        
        for key, value in encrypted_credentials.items():
            if key in ["access_token", "refresh_token", "api_key", "client_secret"]:
                decrypted_credentials[key] = self.encryption_manager.decrypt_field(
                    "credentials", key, value
                )
            else:
                decrypted_credentials[key] = value
        
        # 记录审计日志
        await self._log_credential_access(
            user_id, credential_id, "retrieve", record["provider"]
        )
        
        return decrypted_credentials
    
    async def _log_credential_access(
        self, 
        user_id: str, 
        credential_id: str, 
        action: str, 
        provider: str
    ):
        """记录凭据访问审计日志"""
        audit_log = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "credential_id": credential_id,
            "action": action,
            "provider": provider,
            "timestamp": datetime.utcnow(),
            "ip_address": self._get_current_ip(),
            "user_agent": self._get_current_user_agent()
        }
        
        await self.db.insert("credential_audit_log", audit_log)
```

#### 2.2 输入验证和防护

**全面的输入验证**
```python
class InputValidator:
    """输入验证器"""
    
    def __init__(self):
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$(){}[\]\\]",
            r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)\b",
        ]
    
    def validate_and_sanitize(self, data: dict, schema: dict) -> dict:
        """验证和清理输入数据"""
        validated_data = {}
        
        for field, value in data.items():
            field_schema = schema.get(field, {})
            
            # 类型验证
            expected_type = field_schema.get("type")
            if expected_type and not self._validate_type(value, expected_type):
                raise ValidationError(f"Field {field} has invalid type")
            
            # 长度验证
            if isinstance(value, str):
                min_length = field_schema.get("min_length", 0)
                max_length = field_schema.get("max_length", 10000)
                
                if len(value) < min_length:
                    raise ValidationError(f"Field {field} is too short")
                if len(value) > max_length:
                    raise ValidationError(f"Field {field} is too long")
            
            # 安全验证
            if isinstance(value, str):
                value = self._sanitize_string(value, field_schema)
            
            validated_data[field] = value
        
        return validated_data
    
    def _sanitize_string(self, value: str, field_schema: dict) -> str:
        """清理字符串输入"""
        # HTML转义
        if field_schema.get("escape_html", True):
            value = html.escape(value)
        
        # SQL注入检测
        if self._detect_sql_injection(value):
            raise SecurityError("Potential SQL injection detected")
        
        # XSS检测
        if self._detect_xss(value):
            raise SecurityError("Potential XSS attack detected")
        
        # 命令注入检测
        if field_schema.get("check_command_injection", False):
            if self._detect_command_injection(value):
                raise SecurityError("Potential command injection detected")
        
        return value
    
    def _detect_sql_injection(self, value: str) -> bool:
        """检测SQL注入"""
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    def _detect_xss(self, value: str) -> bool:
        """检测XSS攻击"""
        for pattern in self.xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    def _detect_command_injection(self, value: str) -> bool:
        """检测命令注入"""
        for pattern in self.command_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False

# 安全中间件
class SecurityMiddleware:
    """安全中间件"""
    
    def __init__(self):
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter()
        self.ip_whitelist = self._load_ip_whitelist()
        self.blocked_ips = self._load_blocked_ips()
    
    async def __call__(self, request: Request, call_next):
        """中间件处理"""
        
        # 1. IP检查
        client_ip = self._get_client_ip(request)
        if client_ip in self.blocked_ips:
            raise HTTPException(status_code=403, detail="IP blocked")
        
        # 2. 限流检查
        if not await self.rate_limiter.check_limit(f"ip:{client_ip}"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # 3. 请求大小检查
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=413, detail="Request too large")
        
        # 4. 安全头检查
        self._validate_security_headers(request)
        
        # 执行请求
        response = await call_next(request)
        
        # 5. 添加安全响应头
        self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """添加安全响应头"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
```

## 总结

通过以上最佳实践和实战经验，AutoGPT平台能够实现：

### 开发效率提升
1. **标准化开发流程**：统一的Block开发规范和测试框架
2. **代码质量保障**：完整的代码审查和质量检查机制
3. **快速问题定位**：详细的日志记录和错误处理策略

### 系统性能优化
1. **数据库优化**：合理的索引设计和查询优化
2. **缓存策略**：多级缓存和智能失效机制
3. **异步处理**：高效的任务队列和并发控制

### 安全防护加强
1. **多层认证**：JWT + API Key双重认证体系
2. **权限控制**：细粒度的RBAC + ABAC权限模型
3. **数据保护**：端到端加密和输入验证防护

### 运维监控完善
1. **实时监控**：全面的系统指标和业务指标监控
2. **故障处理**：自动化的故障检测和恢复机制
3. **性能分析**：详细的性能分析和优化建议

这些最佳实践为AutoGPT平台的稳定运行和持续发展提供了坚实的基础。
