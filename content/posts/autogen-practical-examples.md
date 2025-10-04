---
title: "AutoGen实战案例集：从入门到企业级应用"
date: 2025-05-01T07:00:00+08:00
draft: false
featured: true
series: "autogen-architecture"
tags: ["AutoGen", "实战案例", "使用示例", "最佳实践", "企业应用"]
categories: ["autogen", "实战指南"]
author: "Architecture Analysis"
description: "全面的AutoGen框架使用示例，从基础入门到复杂企业级应用的完整实战案例"
toc: true
tocOpen: true
showReadingTime: true
showWordCount: true
weight: 170
slug: "autogen-practical-examples"
---

## 概述

本文档提供AutoGen框架的完整实战案例，涵盖从基础使用到企业级应用的各种场景，帮助开发者快速掌握框架的使用方法和最佳实践。

## 1. 基础入门案例

### 1.1 简单对话代理

```python
"""
案例1：创建一个简单的对话代理
功能：实现基本的问答对话
适用场景：快速原型开发、学习框架基础
"""

import asyncio
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_agentchat import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def simple_chat_example():
    """简单对话示例"""
    
    # 1. 创建运行时
    runtime = SingleThreadedAgentRuntime()
    
    # 2. 配置模型客户端
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key="your_openai_api_key",
        base_url="https://api.openai.com/v1"
    )
    
    # 3. 创建助手代理
    assistant = AssistantAgent(
        name="helpful_assistant",
        model_client=model_client,
        description="一个有用的AI助手，能够回答各种问题",
        system_message="""你是一个专业的AI助手。请遵循以下原则：

        1. 提供准确、有用的信息
        2. 保持友好和专业的语调
        3. 如果不确定答案，请诚实说明
        4. 尽量提供具体的例子和建议
        """
    )
    
    # 4. 注册代理到运行时
    await assistant.register(runtime, "AssistantAgent", lambda: assistant)
    
    # 5. 启动运行时
    runtime_context = runtime.start()
    
    try:
        # 6. 发送消息并获取响应
        questions = [
            "你好，请介绍一下你自己",
            "什么是机器学习？",
            "请推荐一些Python学习资源",
            "如何提高编程技能？"
        ]
        
        for question in questions:
            print(f"\n用户: {question}")
            
            response = await runtime.send_message(
                question,
                AgentId("AssistantAgent", "default")
            )
            
            print(f"助手: {response}")
            
    finally:
        # 7. 清理资源
        await runtime_context.stop()

# 运行示例
if __name__ == "__main__":
    asyncio.run(simple_chat_example())
```

### 1.2 代码执行代理

```python
"""
案例2：创建代码执行代理
功能：执行Python代码并返回结果
适用场景：数据分析、计算任务、代码验证
"""

import asyncio
import subprocess
import tempfile
import os
from typing import Any
from autogen_core import SingleThreadedAgentRuntime, AgentId, MessageContext
from autogen_agentchat import RoutedAgent

class CodeExecutorAgent(RoutedAgent):
    """代码执行代理"""
    
    def __init__(self, name: str = "code_executor"):
        super().__init__(
            AgentId("CodeExecutor", name),
            "Python代码执行代理，可以安全地执行Python代码"
        )
        
        # 允许的模块列表（安全考虑）
        self.allowed_modules = {
            'math', 'statistics', 'datetime', 'json', 'csv',
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'requests'
        }
    
    async def handle_code_request(self, code: str, context: MessageContext) -> dict:
        """处理代码执行请求"""
        
        # 1. 代码安全检查
        if not self._is_code_safe(code):
            return {
                "success": False,
                "error": "代码包含不安全的操作",
                "output": "",
                "execution_time": 0
            }
        
        # 2. 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 3. 执行代码
            start_time = asyncio.get_event_loop().time()
            
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=30  # 30秒超时
            )
            
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            # 4. 返回结果
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else "",
                "execution_time": execution_time,
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "代码执行超时（30秒）",
                "output": "",
                "execution_time": 30
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"执行异常: {str(e)}",
                "output": "",
                "execution_time": 0
            }
        finally:
            # 5. 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _is_code_safe(self, code: str) -> bool:
        """检查代码安全性"""
        
        # 危险关键词检查
        dangerous_keywords = [
            'import os', 'import sys', 'import subprocess',
            'exec(', 'eval(', '__import__',
            'open(', 'file(', 'input(',
            'raw_input(', 'compile('
        ]
        
        code_lower = code.lower()
        for keyword in dangerous_keywords:
            if keyword in code_lower:
                return False
        
        return True

async def code_execution_example():
    """代码执行示例"""
    
    # 1. 创建运行时和代理
    runtime = SingleThreadedAgentRuntime()
    code_executor = CodeExecutorAgent("main")
    
    # 2. 注册代理
    await code_executor.register(runtime, "CodeExecutor", lambda: code_executor)
    
    # 3. 启动运行时
    runtime_context = runtime.start()
    
    try:
        # 4. 测试代码执行
        test_codes = [
            # 简单计算
            """
print("Hello, AutoGen!")
result = 2 + 3 * 4
print(f"计算结果: {result}")
            """,
            
            # 数学运算
            """
import math
numbers = [1, 2, 3, 4, 5]
mean = sum(numbers) / len(numbers)
std_dev = math.sqrt(sum((x - mean) ** 2 for x in numbers) / len(numbers))
print(f"平均值: {mean}")
print(f"标准差: {std_dev}")
            """,
            
            # 数据处理
            """
data = [
    {"name": "Alice", "age": 25, "score": 85},
    {"name": "Bob", "age": 30, "score": 92},
    {"name": "Charlie", "age": 35, "score": 78}
]

# 计算平均分
avg_score = sum(item["score"] for item in data) / len(data)
print(f"平均分: {avg_score:.2f}")

# 找出最高分
best_student = max(data, key=lambda x: x["score"])
print(f"最高分学生: {best_student['name']} ({best_student['score']}分)")
            """
        ]
        
        for i, code in enumerate(test_codes, 1):
            print(f"\n=== 测试 {i} ===")
            print("代码:")
            print(code.strip())
            
            result = await runtime.send_message(
                code,
                AgentId("CodeExecutor", "main")
            )
            
            print(f"\n执行结果:")
            print(f"成功: {result['success']}")
            print(f"输出: {result['output']}")
            if result['error']:
                print(f"错误: {result['error']}")
            print(f"执行时间: {result['execution_time']:.3f}秒")
            
    finally:
        await runtime_context.stop()

# 运行示例
if __name__ == "__main__":
    asyncio.run(code_execution_example())
```

## 2. 多代理协作案例

### 2.1 研究团队协作

```python
"""
案例3：研究团队协作系统
功能：模拟研究团队进行文献调研、分析和报告生成
适用场景：学术研究、市场调研、技术分析
"""

import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from autogen_core import SingleThreadedAgentRuntime, AgentId, MessageContext
from autogen_agentchat import RoutedAgent, AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

@dataclass
class ResearchTask:
    """研究任务"""
    topic: str
    requirements: List[str]
    deadline: str
    priority: str

@dataclass
class ResearchResult:
    """研究结果"""
    topic: str
    findings: List[str]
    sources: List[str]
    confidence: float
    researcher: str

class ResearchCoordinator(RoutedAgent):
    """研究协调员"""
    
    def __init__(self):
        super().__init__(
            AgentId("ResearchCoordinator", "main"),
            "研究协调员，负责分配任务和整合结果"
        )
        self.active_tasks: Dict[str, ResearchTask] = {}
        self.completed_results: List[ResearchResult] = []
    
    async def handle_research_request(self, request: dict, context: MessageContext) -> dict:
        """处理研究请求"""
        
        topic = request.get("topic")
        requirements = request.get("requirements", [])
        
        # 1. 创建研究任务
        task = ResearchTask(
            topic=topic,
            requirements=requirements,
            deadline="7天",
            priority="高"
        )
        
        self.active_tasks[topic] = task
        
        # 2. 分配给研究员
        researchers = ["researcher_1", "researcher_2", "researcher_3"]
        
        # 为每个研究员分配子任务
        subtasks = self._split_research_task(task, len(researchers))
        
        results = []
        for i, researcher in enumerate(researchers):
            subtask_request = {
                "topic": subtasks[i]["topic"],
                "focus_area": subtasks[i]["focus_area"],
                "requirements": subtasks[i]["requirements"]
            }
            
            # 发送任务给研究员
            result = await context.send_message(
                subtask_request,
                AgentId("Researcher", researcher)
            )
            
            results.append(result)
        
        # 3. 整合结果
        final_report = self._integrate_results(task, results)
        
        return {
            "topic": topic,
            "report": final_report,
            "researchers_involved": researchers,
            "completion_time": "模拟完成时间"
        }
    
    def _split_research_task(self, task: ResearchTask, num_researchers: int) -> List[dict]:
        """将研究任务分解为子任务"""
        
        focus_areas = [
            "理论基础和概念定义",
            "当前技术现状和趋势",
            "实际应用案例和效果"
        ]
        
        subtasks = []
        for i in range(num_researchers):
            subtasks.append({
                "topic": f"{task.topic} - {focus_areas[i % len(focus_areas)]}",
                "focus_area": focus_areas[i % len(focus_areas)],
                "requirements": task.requirements
            })
        
        return subtasks
    
    def _integrate_results(self, task: ResearchTask, results: List[dict]) -> str:
        """整合研究结果"""
        
        report_sections = []
        
        # 1. 执行摘要
        report_sections.append("# 研究报告：" + task.topic)
        report_sections.append("\n## 执行摘要")
        report_sections.append(f"本报告针对'{task.topic}'进行了全面研究，涉及理论基础、技术现状和实际应用等多个方面。")
        
        # 2. 各研究员的发现
        for i, result in enumerate(results, 1):
            report_sections.append(f"\n## 研究发现 {i}")
            report_sections.append(f"**研究重点**: {result.get('focus_area', '未指定')}")
            report_sections.append(f"**主要发现**: {result.get('findings', '无')}")
            report_sections.append(f"**参考来源**: {result.get('sources', '无')}")
        
        # 3. 综合结论
        report_sections.append("\n## 综合结论")
        report_sections.append("基于以上研究发现，我们得出以下结论...")
        
        return "\n".join(report_sections)

class Researcher(RoutedAgent):
    """研究员代理"""
    
    def __init__(self, name: str, specialty: str):
        super().__init__(
            AgentId("Researcher", name),
            f"专业研究员 - {specialty}"
        )
        self.name = name
        self.specialty = specialty
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key="your_openai_api_key"
        )
    
    async def handle_research_subtask(self, request: dict, context: MessageContext) -> dict:
        """处理研究子任务"""
        
        topic = request.get("topic")
        focus_area = request.get("focus_area")
        requirements = request.get("requirements", [])
        
        # 1. 构建研究提示
        prompt = f"""
作为专业研究员，请针对以下主题进行深入研究：

主题: {topic}
研究重点: {focus_area}
具体要求: {', '.join(requirements)}

请提供：

1. 关键概念和定义
2. 主要发现和观点
3. 相关数据和统计
4. 参考来源和文献
5. 研究结论和建议

请确保信息准确、全面且具有参考价值。
        """
        
        # 2. 调用LLM进行研究
        try:
            # 这里应该调用实际的LLM API
            # response = await self.model_client.create_completion([{"role": "user", "content": prompt}])
            
            # 模拟研究结果
            findings = [
                f"针对{focus_area}的深入分析显示...",
                f"当前{topic}领域的主要趋势包括...",
                f"实证研究表明{topic}在实际应用中..."
            ]
            
            sources = [
                "学术期刊文章 (2023)",
                "行业报告和白皮书",
                "专家访谈和调研数据"
            ]
            
            return {
                "researcher": self.name,
                "topic": topic,
                "focus_area": focus_area,
                "findings": findings,
                "sources": sources,
                "confidence": 0.85,
                "completion_time": "2小时"
            }
            
        except Exception as e:
            return {
                "researcher": self.name,
                "error": f"研究过程中发生错误: {str(e)}",
                "findings": [],
                "sources": [],
                "confidence": 0.0
            }

async def research_team_example():
    """研究团队协作示例"""
    
    # 1. 创建运行时
    runtime = SingleThreadedAgentRuntime()
    
    # 2. 创建研究团队
    coordinator = ResearchCoordinator()
    
    researchers = [
        Researcher("researcher_1", "人工智能"),
        Researcher("researcher_2", "数据科学"),
        Researcher("researcher_3", "软件工程")
    ]
    
    # 3. 注册代理
    await coordinator.register(runtime, "ResearchCoordinator", lambda: coordinator)
    
    for researcher in researchers:
        await researcher.register(
            runtime,
            "Researcher",
            lambda r=researcher: r
        )
    
    # 4. 启动运行时
    runtime_context = runtime.start()
    
    try:
        # 5. 发起研究请求
        research_requests = [
            {
                "topic": "大语言模型在企业应用中的实践",
                "requirements": [
                    "分析技术可行性",
                    "评估实施成本",
                    "识别潜在风险",
                    "提供实施建议"
                ]
            },
            {
                "topic": "多代理系统的设计模式",
                "requirements": [
                    "梳理设计原则",
                    "分析架构模式",
                    "总结最佳实践",
                    "提供案例研究"
                ]
            }
        ]
        
        for request in research_requests:
            print(f"\n=== 研究请求: {request['topic']} ===")
            
            result = await runtime.send_message(
                request,
                AgentId("ResearchCoordinator", "main")
            )
            
            print(f"研究完成!")
            print(f"参与研究员: {', '.join(result['researchers_involved'])}")
            print(f"报告预览:")
            print(result['report'][:500] + "..." if len(result['report']) > 500 else result['report'])
            
    finally:
        await runtime_context.stop()

# 运行示例
if __name__ == "__main__":
    asyncio.run(research_team_example())
```

### 2.2 客服系统案例

```python
"""
案例4：智能客服系统
功能：多层级客服代理协作处理用户问题
适用场景：企业客服、技术支持、售后服务
"""

import asyncio
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from autogen_core import SingleThreadedAgentRuntime, AgentId, MessageContext
from autogen_agentchat import RoutedAgent

class TicketPriority(Enum):
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    URGENT = "紧急"

class TicketStatus(Enum):
    OPEN = "开放"
    IN_PROGRESS = "处理中"
    RESOLVED = "已解决"
    CLOSED = "已关闭"

@dataclass
class CustomerTicket:
    """客服工单"""
    id: str
    customer_id: str
    title: str
    description: str
    category: str
    priority: TicketPriority
    status: TicketStatus
    created_at: datetime
    assigned_agent: Optional[str] = None
    resolution: Optional[str] = None

class CustomerServiceRouter(RoutedAgent):
    """客服路由代理"""
    
    def __init__(self):
        super().__init__(
            AgentId("CustomerServiceRouter", "main"),
            "客服路由代理，负责分类和分配客户问题"
        )
        
        # 问题分类规则
        self.category_rules = {
            "技术支持": ["bug", "错误", "故障", "无法使用", "技术问题"],
            "账户问题": ["登录", "密码", "账户", "权限", "访问"],
            "订单问题": ["订单", "支付", "退款", "发货", "物流"],
            "产品咨询": ["功能", "价格", "方案", "咨询", "了解"],
            "投诉建议": ["投诉", "建议", "不满", "改进", "反馈"]
        }
        
        # 代理专长映射
        self.agent_specialties = {
            "tech_support": "技术支持",
            "account_manager": "账户问题",
            "order_specialist": "订单问题",
            "product_consultant": "产品咨询",
            "complaint_handler": "投诉建议"
        }
    
    async def handle_customer_inquiry(self, inquiry: dict, context: MessageContext) -> dict:
        """处理客户咨询"""
        
        # 1. 创建工单
        ticket = CustomerTicket(
            id=f"T{datetime.now().strftime('%Y%m%d%H%M%S')}",
            customer_id=inquiry.get("customer_id", "unknown"),
            title=inquiry.get("title", ""),
            description=inquiry.get("description", ""),
            category=self._classify_inquiry(inquiry.get("description", "")),
            priority=self._determine_priority(inquiry.get("description", "")),
            status=TicketStatus.OPEN,
            created_at=datetime.now()
        )
        
        # 2. 分配给专门代理
        assigned_agent = self._assign_agent(ticket.category)
        ticket.assigned_agent = assigned_agent
        ticket.status = TicketStatus.IN_PROGRESS
        
        print(f"工单 {ticket.id} 已创建并分配给 {assigned_agent}")
        print(f"分类: {ticket.category}, 优先级: {ticket.priority.value}")
        
        # 3. 转发给专门代理处理
        agent_request = {
            "ticket_id": ticket.id,
            "customer_id": ticket.customer_id,
            "title": ticket.title,
            "description": ticket.description,
            "category": ticket.category,
            "priority": ticket.priority.value
        }
        
        try:
            # 发送给专门代理
            agent_response = await context.send_message(
                agent_request,
                AgentId("CustomerServiceAgent", assigned_agent)
            )
            
            # 4. 更新工单状态
            if agent_response.get("resolved", False):
                ticket.status = TicketStatus.RESOLVED
                ticket.resolution = agent_response.get("resolution", "")
            
            return {
                "ticket_id": ticket.id,
                "status": ticket.status.value,
                "assigned_agent": assigned_agent,
                "response": agent_response.get("response", ""),
                "resolution": ticket.resolution,
                "processing_time": agent_response.get("processing_time", 0)
            }
            
        except Exception as e:
            return {
                "ticket_id": ticket.id,
                "status": "处理失败",
                "error": str(e)
            }
    
    def _classify_inquiry(self, description: str) -> str:
        """分类客户咨询"""
        description_lower = description.lower()
        
        for category, keywords in self.category_rules.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return "产品咨询"  # 默认分类
    
    def _determine_priority(self, description: str) -> TicketPriority:
        """确定优先级"""
        description_lower = description.lower()
        
        urgent_keywords = ["紧急", "严重", "无法使用", "系统崩溃"]
        high_keywords = ["重要", "影响业务", "客户投诉"]
        
        if any(keyword in description_lower for keyword in urgent_keywords):
            return TicketPriority.URGENT
        elif any(keyword in description_lower for keyword in high_keywords):
            return TicketPriority.HIGH
        else:
            return TicketPriority.MEDIUM
    
    def _assign_agent(self, category: str) -> str:
        """分配代理"""
        for agent_id, specialty in self.agent_specialties.items():
            if specialty == category:
                return agent_id
        return "general_agent"  # 默认代理

class CustomerServiceAgent(RoutedAgent):
    """专门客服代理"""
    
    def __init__(self, agent_id: str, specialty: str):
        super().__init__(
            AgentId("CustomerServiceAgent", agent_id),
            f"专门客服代理 - {specialty}"
        )
        self.agent_id = agent_id
        self.specialty = specialty
        
        # 知识库（简化版）
        self.knowledge_base = {
            "技术支持": {
                "常见问题": [
                    "清除浏览器缓存",
                    "检查网络连接",
                    "更新到最新版本",
                    "重启应用程序"
                ],
                "解决方案": {
                    "登录问题": "请检查用户名和密码是否正确，如果忘记密码请使用找回密码功能",
                    "功能异常": "请尝试刷新页面或重启应用，如果问题持续请联系技术支持"
                }
            },
            "账户问题": {
                "常见问题": [
                    "密码重置",
                    "账户锁定",
                    "权限申请",
                    "个人信息修改"
                ],
                "解决方案": {
                    "密码重置": "请点击登录页面的'忘记密码'链接，按照提示操作",
                    "账户锁定": "账户锁定通常是由于多次错误登录导致，请联系管理员解锁"
                }
            },
            "订单问题": {
                "常见问题": [
                    "订单查询",
                    "支付问题",
                    "退款申请",
                    "物流跟踪"
                ],
                "解决方案": {
                    "支付失败": "请检查银行卡余额和网络连接，或尝试其他支付方式",
                    "退款申请": "退款申请需要3-5个工作日处理，请保持手机畅通"
                }
            }
        }
    
    async def handle_ticket(self, request: dict, context: MessageContext) -> dict:
        """处理客服工单"""
        
        ticket_id = request.get("ticket_id")
        description = request.get("description", "")
        category = request.get("category", "")
        priority = request.get("priority", "")
        
        print(f"代理 {self.agent_id} 正在处理工单 {ticket_id}")
        
        # 1. 分析问题
        analysis = self._analyze_problem(description, category)
        
        # 2. 查找解决方案
        solution = self._find_solution(analysis, category)
        
        # 3. 生成响应
        response = self._generate_response(analysis, solution, priority)
        
        # 4. 判断是否需要升级
        needs_escalation = self._needs_escalation(analysis, priority)
        
        if needs_escalation:
            # 升级到高级代理
            escalation_result = await self._escalate_ticket(request, context)
            return escalation_result
        
        return {
            "ticket_id": ticket_id,
            "agent_id": self.agent_id,
            "response": response,
            "solution": solution,
            "resolved": solution is not None,
            "processing_time": 2.5,  # 模拟处理时间
            "satisfaction_score": 4.2  # 模拟满意度评分
        }
    
    def _analyze_problem(self, description: str, category: str) -> dict:
        """分析问题"""
        
        # 简化的问题分析
        keywords = description.lower().split()
        
        analysis = {
            "category": category,
            "keywords": keywords,
            "complexity": "简单" if len(keywords) < 10 else "复杂",
            "sentiment": "中性"  # 简化的情感分析
        }
        
        # 检测负面情绪
        negative_words = ["不满", "愤怒", "失望", "糟糕", "差劲"]
        if any(word in description for word in negative_words):
            analysis["sentiment"] = "负面"
        
        return analysis
    
    def _find_solution(self, analysis: dict, category: str) -> Optional[str]:
        """查找解决方案"""
        
        if category not in self.knowledge_base:
            return None
        
        kb = self.knowledge_base[category]
        keywords = analysis["keywords"]
        
        # 匹配解决方案
        for problem, solution in kb["解决方案"].items():
            if any(keyword in problem.lower() for keyword in keywords):
                return solution
        
        # 返回通用建议
        if "常见问题" in kb:
            return f"建议尝试以下方法：{', '.join(kb['常见问题'][:2])}"
        
        return None
    
    def _generate_response(self, analysis: dict, solution: Optional[str], priority: str) -> str:
        """生成响应"""
        
        # 根据情感调整语调
        if analysis["sentiment"] == "负面":
            greeting = "非常抱歉给您带来了不便，我会尽快为您解决这个问题。"
        else:
            greeting = "感谢您的咨询，我很乐意为您提供帮助。"
        
        if solution:
            response = f"{greeting}\n\n针对您的问题，建议您尝试以下解决方案：\n{solution}"
            
            if priority in ["高", "紧急"]:
                response += "\n\n由于您的问题比较紧急，如果以上方案无法解决，请立即联系我们的技术支持热线。"
        else:
            response = f"{greeting}\n\n您的问题比较特殊，我需要进一步了解详情。请您提供更多信息，或者我可以为您转接到专门的技术专家。"
        
        response += "\n\n如果还有其他问题，请随时联系我们。祝您使用愉快！"
        
        return response
    
    def _needs_escalation(self, analysis: dict, priority: str) -> bool:
        """判断是否需要升级"""
        
        # 升级条件
        if priority == "紧急":
            return True
        
        if analysis["complexity"] == "复杂" and analysis["sentiment"] == "负面":
            return True
        
        return False
    
    async def _escalate_ticket(self, request: dict, context: MessageContext) -> dict:
        """升级工单到高级代理"""
        
        escalation_request = {
            **request,
            "escalated_from": self.agent_id,
            "escalation_reason": "问题复杂或优先级高"
        }
        
        try:
            # 转发给高级代理
            result = await context.send_message(
                escalation_request,
                AgentId("SeniorCustomerServiceAgent", "senior_agent")
            )
            
            return {
                **result,
                "escalated": True,
                "escalated_from": self.agent_id
            }
            
        except Exception as e:
            return {
                "ticket_id": request.get("ticket_id"),
                "error": f"升级失败: {str(e)}",
                "resolved": False
            }

class SeniorCustomerServiceAgent(RoutedAgent):
    """高级客服代理"""
    
    def __init__(self):
        super().__init__(
            AgentId("SeniorCustomerServiceAgent", "senior_agent"),
            "高级客服代理，处理复杂和紧急问题"
        )
    
    async def handle_escalated_ticket(self, request: dict, context: MessageContext) -> dict:
        """处理升级的工单"""
        
        ticket_id = request.get("ticket_id")
        escalated_from = request.get("escalated_from")
        
        print(f"高级代理接收到从 {escalated_from} 升级的工单 {ticket_id}")
        
        # 高级处理逻辑（简化）
        response = f"""
感谢您的耐心等待。我是高级客服专员，已经接手您的问题。

我已经详细了解了您的情况，这确实是一个需要特别关注的问题。我将为您提供以下专业解决方案：

1. 立即为您开通绿色通道，优先处理您的问题
2. 安排技术专家进行一对一支持
3. 在24小时内给您明确的解决方案
4. 全程跟踪处理进度，确保问题得到彻底解决

同时，作为对您遇到问题的补偿，我们将为您提供额外的服务优惠。

请您保持联系方式畅通，我们会主动与您联系跟进处理进度。
        """
        
        return {
            "ticket_id": ticket_id,
            "agent_id": "senior_agent",
            "response": response.strip(),
            "resolved": True,
            "escalated": True,
            "processing_time": 5.0,
            "satisfaction_score": 4.8,
            "follow_up_required": True
        }

async def customer_service_example():
    """客服系统示例"""
    
    # 1. 创建运行时
    runtime = SingleThreadedAgentRuntime()
    
    # 2. 创建客服团队
    router = CustomerServiceRouter()
    
    # 专门代理
    agents = [
        CustomerServiceAgent("tech_support", "技术支持"),
        CustomerServiceAgent("account_manager", "账户问题"),
        CustomerServiceAgent("order_specialist", "订单问题"),
        CustomerServiceAgent("product_consultant", "产品咨询")
    ]
    
    senior_agent = SeniorCustomerServiceAgent()
    
    # 3. 注册代理
    await router.register(runtime, "CustomerServiceRouter", lambda: router)
    
    for agent in agents:
        await agent.register(runtime, "CustomerServiceAgent", lambda a=agent: a)
    
    await senior_agent.register(runtime, "SeniorCustomerServiceAgent", lambda: senior_agent)
    
    # 4. 启动运行时
    runtime_context = runtime.start()
    
    try:
        # 5. 模拟客户咨询
        customer_inquiries = [
            {
                "customer_id": "C001",
                "title": "登录问题",
                "description": "我无法登录系统，一直提示密码错误，但我确定密码是正确的"
            },
            {
                "customer_id": "C002",
                "title": "订单支付失败",
                "description": "我的订单支付一直失败，已经尝试了多张银行卡都不行，很着急"
            },
            {
                "customer_id": "C003",
                "title": "系统故障投诉",
                "description": "系统经常崩溃，严重影响我的工作，我对你们的服务非常不满意，要求立即解决"
            },
            {
                "customer_id": "C004",
                "title": "产品功能咨询",
                "description": "想了解一下你们产品的高级功能和价格方案"
            }
        ]
        
        for inquiry in customer_inquiries:
            print(f"\n=== 客户咨询: {inquiry['title']} ===")
            print(f"客户ID: {inquiry['customer_id']}")
            print(f"问题描述: {inquiry['description']}")
            
            result = await runtime.send_message(
                inquiry,
                AgentId("CustomerServiceRouter", "main")
            )
            
            print(f"\n处理结果:")
            print(f"工单ID: {result['ticket_id']}")
            print(f"处理状态: {result['status']}")
            print(f"负责代理: {result['assigned_agent']}")
            if result.get('escalated'):
                print(f"已升级处理: 是")
            print(f"客服回复: {result['response']}")
            print(f"处理时间: {result.get('processing_time', 0):.1f}秒")
            
    finally:
        await runtime_context.stop()

# 运行示例
if __name__ == "__main__":
    asyncio.run(customer_service_example())
```

## 3. 企业级应用案例

### 3.1 智能运维系统

```python
"""
案例5：智能运维系统
功能：自动化监控、故障诊断和修复
适用场景：DevOps、系统运维、故障管理
"""

import asyncio
import json
import random
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from autogen_core import SingleThreadedAgentRuntime, AgentId, MessageContext
from autogen_agentchat import RoutedAgent

class AlertSeverity(Enum):
    INFO = "信息"
    WARNING = "警告"
    ERROR = "错误"
    CRITICAL = "严重"

class SystemStatus(Enum):
    HEALTHY = "健康"
    WARNING = "警告"
    ERROR = "错误"
    DOWN = "宕机"

@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    response_time: float
    error_rate: float

@dataclass
class Alert:
    """告警信息"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source_system: str
    timestamp: datetime
    metrics: Optional[SystemMetrics] = None
    resolved: bool = False
    resolution: Optional[str] = None

class MonitoringAgent(RoutedAgent):
    """监控代理"""
    
    def __init__(self):
        super().__init__(
            AgentId("MonitoringAgent", "main"),
            "系统监控代理，负责收集和分析系统指标"
        )
        
        # 监控的系统列表
        self.monitored_systems = [
            "web-server-01", "web-server-02", "database-01",
            "redis-cluster", "message-queue", "api-gateway"
        ]
        
        # 告警阈值
        self.thresholds = {
            "cpu_usage": {"warning": 70, "critical": 90},
            "memory_usage": {"warning": 80, "critical": 95},
            "disk_usage": {"warning": 85, "critical": 95},
            "response_time": {"warning": 1000, "critical": 3000},  # ms
            "error_rate": {"warning": 5, "critical": 10}  # %
        }
        
        self.active_alerts: Dict[str, Alert] = {}
    
    async def handle_monitoring_request(self, request: dict, context: MessageContext) -> dict:
        """处理监控请求"""
        
        action = request.get("action", "collect_metrics")
        
        if action == "collect_metrics":
            return await self._collect_all_metrics(context)
        elif action == "check_alerts":
            return await self._check_and_process_alerts(context)
        elif action == "get_system_status":
            return self._get_system_status()
        else:
            return {"error": f"未知操作: {action}"}
    
    async def _collect_all_metrics(self, context: MessageContext) -> dict:
        """收集所有系统指标"""
        
        all_metrics = {}
        alerts_generated = []
        
        for system in self.monitored_systems:
            # 模拟收集指标
            metrics = self._simulate_metrics(system)
            all_metrics[system] = asdict(metrics)
            
            # 检查是否需要生成告警
            alerts = self._check_thresholds(system, metrics)
            alerts_generated.extend(alerts)
        
        # 处理新告警
        for alert in alerts_generated:
            await self._process_alert(alert, context)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "systems_monitored": len(self.monitored_systems),
            "metrics_collected": all_metrics,
            "alerts_generated": len(alerts_generated),
            "active_alerts": len(self.active_alerts)
        }
    
    def _simulate_metrics(self, system: str) -> SystemMetrics:
        """模拟系统指标收集"""
        
        # 根据系统类型生成不同的指标模式
        base_metrics = {
            "web-server": {"cpu": 45, "memory": 60, "disk": 30},
            "database": {"cpu": 70, "memory": 85, "disk": 60},
            "redis": {"cpu": 30, "memory": 40, "disk": 20},
            "message-queue": {"cpu": 50, "memory": 55, "disk": 25},
            "api-gateway": {"cpu": 40, "memory": 50, "disk": 35}
        }
        
        # 确定系统类型
        system_type = "web-server"
        for key in base_metrics:
            if key in system:
                system_type = key
                break
        
        base = base_metrics[system_type]
        
        # 添加随机波动
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=max(0, min(100, base["cpu"] + random.uniform(-15, 25))),
            memory_usage=max(0, min(100, base["memory"] + random.uniform(-10, 20))),
            disk_usage=max(0, min(100, base["disk"] + random.uniform(-5, 15))),
            network_io=random.uniform(10, 100),  # MB/s
            response_time=random.uniform(100, 2000),  # ms
            error_rate=random.uniform(0, 8)  # %
        )
    
    def _check_thresholds(self, system: str, metrics: SystemMetrics) -> List[Alert]:
        """检查指标阈值"""
        
        alerts = []
        
        # 检查各项指标
        checks = [
            ("cpu_usage", metrics.cpu_usage, "CPU使用率"),
            ("memory_usage", metrics.memory_usage, "内存使用率"),
            ("disk_usage", metrics.disk_usage, "磁盘使用率"),
            ("response_time", metrics.response_time, "响应时间"),
            ("error_rate", metrics.error_rate, "错误率")
        ]
        
        for metric_name, value, display_name in checks:
            threshold = self.thresholds[metric_name]
            
            if value >= threshold["critical"]:
                severity = AlertSeverity.CRITICAL
                description = f"{display_name}达到严重水平: {value:.2f}"
            elif value >= threshold["warning"]:
                severity = AlertSeverity.WARNING
                description = f"{display_name}超过警告阈值: {value:.2f}"
            else:
                continue
            
            alert_id = f"{system}_{metric_name}_{int(datetime.now().timestamp())}"
            
            alert = Alert(
                id=alert_id,
                title=f"{system} {display_name}告警",
                description=description,
                severity=severity,
                source_system=system,
                timestamp=datetime.now(),
                metrics=metrics
            )
            
            alerts.append(alert)
        
        return alerts
    
    async def _process_alert(self, alert: Alert, context: MessageContext) -> None:
        """处理告警"""
        
        # 添加到活跃告警列表
        self.active_alerts[alert.id] = alert
        
        print(f"🚨 新告警: {alert.title} ({alert.severity.value})")
        
        # 发送给故障诊断代理
        diagnostic_request = {
            "alert_id": alert.id,
            "alert": asdict(alert),
            "action": "diagnose"
        }
        
        try:
            await context.send_message(
                diagnostic_request,
                AgentId("DiagnosticAgent", "main")
            )
        except Exception as e:
            print(f"发送诊断请求失败: {e}")
    
    def _get_system_status(self) -> dict:
        """获取系统整体状态"""
        
        system_statuses = {}
        
        for system in self.monitored_systems:
            # 检查该系统的活跃告警
            system_alerts = [
                alert for alert in self.active_alerts.values()
                if alert.source_system == system and not alert.resolved
            ]
            
            if not system_alerts:
                status = SystemStatus.HEALTHY
            elif any(alert.severity == AlertSeverity.CRITICAL for alert in system_alerts):
                status = SystemStatus.DOWN
            elif any(alert.severity == AlertSeverity.ERROR for alert in system_alerts):
                status = SystemStatus.ERROR
            else:
                status = SystemStatus.WARNING
            
            system_statuses[system] = {
                "status": status.value,
                "active_alerts": len(system_alerts),
                "last_check": datetime.now().isoformat()
            }
        
        return {
            "overall_status": self._calculate_overall_status(system_statuses),
            "systems": system_statuses,
            "total_active_alerts": len([a for a in self.active_alerts.values() if not a.resolved])
        }
    
    def _calculate_overall_status(self, system_statuses: dict) -> str:
        """计算整体状态"""
        
        statuses = [info["status"] for info in system_statuses.values()]
        
        if "宕机" in statuses:
            return "严重"
        elif "错误" in statuses:
            return "错误"
        elif "警告" in statuses:
            return "警告"
        else:
            return "健康"

class DiagnosticAgent(RoutedAgent):
    """故障诊断代理"""
    
    def __init__(self):
        super().__init__(
            AgentId("DiagnosticAgent", "main"),
            "故障诊断代理，分析告警并提供解决方案"
        )
        
        # 诊断规则库
        self.diagnostic_rules = {
            "cpu_usage": {
                "possible_causes": [
                    "进程占用CPU过高",
                    "系统负载过大",
                    "死循环或无限递归",
                    "资源竞争"
                ],
                "diagnostic_steps": [
                    "检查top命令输出，找出CPU占用最高的进程",
                    "分析进程行为，确定是否正常",
                    "检查系统负载和并发连接数",
                    "查看系统日志中的异常信息"
                ]
            },
            "memory_usage": {
                "possible_causes": [
                    "内存泄漏",
                    "缓存过大",
                    "进程内存占用异常",
                    "系统内存不足"
                ],
                "diagnostic_steps": [
                    "使用free命令查看内存使用情况",
                    "检查各进程内存占用",
                    "分析内存增长趋势",
                    "检查是否存在内存泄漏"
                ]
            },
            "response_time": {
                "possible_causes": [
                    "数据库查询慢",
                    "网络延迟",
                    "服务器负载高",
                    "代码性能问题"
                ],
                "diagnostic_steps": [
                    "检查数据库慢查询日志",
                    "分析网络连接状况",
                    "查看服务器资源使用情况",
                    "检查应用程序性能指标"
                ]
            }
        }
    
    async def handle_diagnostic_request(self, request: dict, context: MessageContext) -> dict:
        """处理诊断请求"""
        
        alert_data = request.get("alert", {})
        alert_id = request.get("alert_id")
        
        print(f"🔍 开始诊断告警: {alert_id}")
        
        # 1. 分析告警
        analysis = self._analyze_alert(alert_data)
        
        # 2. 执行诊断步骤
        diagnostic_result = await self._perform_diagnosis(analysis, context)
        
        # 3. 生成修复建议
        recommendations = self._generate_recommendations(analysis, diagnostic_result)
        
        # 4. 判断是否需要自动修复
        auto_fix_needed = self._should_auto_fix(alert_data, diagnostic_result)
        
        result = {
            "alert_id": alert_id,
            "analysis": analysis,
            "diagnostic_result": diagnostic_result,
            "recommendations": recommendations,
            "auto_fix_needed": auto_fix_needed,
            "confidence": diagnostic_result.get("confidence", 0.7)
        }
        
        # 5. 如果需要自动修复，发送给修复代理
        if auto_fix_needed and diagnostic_result.get("confidence", 0) > 0.8:
            await self._request_auto_fix(result, context)
        
        return result
    
    def _analyze_alert(self, alert_data: dict) -> dict:
        """分析告警"""
        
        # 提取关键信息
        severity = alert_data.get("severity", "WARNING")
        source_system = alert_data.get("source_system", "")
        description = alert_data.get("description", "")
        
        # 确定问题类型
        problem_type = "unknown"
        for metric_type in self.diagnostic_rules.keys():
            if metric_type in description.lower():
                problem_type = metric_type
                break
        
        analysis = {
            "problem_type": problem_type,
            "severity": severity,
            "source_system": source_system,
            "description": description,
            "timestamp": alert_data.get("timestamp"),
            "affected_metrics": self._extract_metrics(alert_data)
        }
        
        return analysis
    
    def _extract_metrics(self, alert_data: dict) -> dict:
        """提取指标信息"""
        
        metrics = alert_data.get("metrics", {})
        if not metrics:
            return {}
        
        return {
            "cpu_usage": metrics.get("cpu_usage", 0),
            "memory_usage": metrics.get("memory_usage", 0),
            "disk_usage": metrics.get("disk_usage", 0),
            "response_time": metrics.get("response_time", 0),
            "error_rate": metrics.get("error_rate", 0)
        }
    
    async def _perform_diagnosis(self, analysis: dict, context: MessageContext) -> dict:
        """执行诊断"""
        
        problem_type = analysis["problem_type"]
        
        if problem_type not in self.diagnostic_rules:
            return {
                "status": "无法诊断",
                "reason": f"未知问题类型: {problem_type}",
                "confidence": 0.1
            }
        
        rules = self.diagnostic_rules[problem_type]
        
        # 模拟执行诊断步骤
        diagnostic_results = []
        
        for step in rules["diagnostic_steps"]:
            # 模拟诊断步骤执行
            result = await self._simulate_diagnostic_step(step, analysis)
            diagnostic_results.append({
                "step": step,
                "result": result,
                "success": result.get("success", True)
            })
        
        # 计算诊断置信度
        success_rate = sum(1 for r in diagnostic_results if r["success"]) / len(diagnostic_results)
        confidence = min(0.9, success_rate * 0.8 + 0.2)
        
        return {
            "status": "诊断完成",
            "possible_causes": rules["possible_causes"],
            "diagnostic_steps": diagnostic_results,
            "confidence": confidence,
            "root_cause": self._determine_root_cause(diagnostic_results, rules["possible_causes"])
        }
    
    async def _simulate_diagnostic_step(self, step: str, analysis: dict) -> dict:
        """模拟诊断步骤执行"""
        
        # 模拟不同诊断步骤的结果
        if "top命令" in step or "进程" in step:
            return {
                "success": True,
                "finding": "发现java进程CPU占用率达到85%",
                "details": "PID 1234的java进程持续占用大量CPU资源"
            }
        elif "内存" in step:
            return {
                "success": True,
                "finding": "内存使用率持续增长",
                "details": "过去1小时内存使用率从60%增长到90%"
            }
        elif "数据库" in step:
            return {
                "success": True,
                "finding": "发现3个慢查询",
                "details": "SELECT查询平均执行时间超过2秒"
            }
        else:
            return {
                "success": random.choice([True, False]),
                "finding": "检查完成",
                "details": f"执行步骤: {step}"
            }
    
    def _determine_root_cause(self, diagnostic_results: List[dict], possible_causes: List[str]) -> str:
        """确定根本原因"""
        
        # 简化的根因分析
        successful_findings = [
            r["result"]["finding"] for r in diagnostic_results
            if r["success"] and "发现" in r["result"]["finding"]
        ]
        
        if successful_findings:
            return f"根据诊断结果，主要问题是: {successful_findings[0]}"
        else:
            return f"可能的原因: {possible_causes[0]}"
    
    def _generate_recommendations(self, analysis: dict, diagnostic_result: dict) -> List[str]:
        """生成修复建议"""
        
        problem_type = analysis["problem_type"]
        root_cause = diagnostic_result.get("root_cause", "")
        
        recommendations = []
        
        if "cpu" in problem_type.lower() or "CPU" in root_cause:
            recommendations.extend([
                "重启占用CPU过高的进程",
                "优化应用程序代码",
                "增加服务器CPU资源",
                "实施负载均衡"
            ])
        
        if "memory" in problem_type.lower() or "内存" in root_cause:
            recommendations.extend([
                "重启内存占用异常的服务",
                "清理系统缓存",
                "检查并修复内存泄漏",
                "增加系统内存"
            ])
        
        if "response_time" in problem_type.lower() or "查询" in root_cause:
            recommendations.extend([
                "优化数据库查询",
                "添加数据库索引",
                "启用查询缓存",
                "优化网络配置"
            ])
        
        # 通用建议
        recommendations.extend([
            "监控系统资源使用情况",
            "检查系统日志",
            "联系相关技术人员"
        ])
        
        return recommendations[:5]  # 返回前5个建议
    
    def _should_auto_fix(self, alert_data: dict, diagnostic_result: dict) -> bool:
        """判断是否应该自动修复"""
        
        # 自动修复条件
        confidence = diagnostic_result.get("confidence", 0)
        severity = alert_data.get("severity", "WARNING")
        
        # 高置信度且非严重告警才考虑自动修复
        if confidence > 0.8 and severity != "CRITICAL":
            return True
        
        return False
    
    async def _request_auto_fix(self, diagnostic_result: dict, context: MessageContext) -> None:
        """请求自动修复"""
        
        fix_request = {
            "alert_id": diagnostic_result["alert_id"],
            "diagnostic_result": diagnostic_result,
            "action": "auto_fix"
        }
        
        try:
            await context.send_message(
                fix_request,
                AgentId("AutoFixAgent", "main")
            )
            print(f"✨ 已请求自动修复: {diagnostic_result['alert_id']}")
        except Exception as e:
            print(f"请求自动修复失败: {e}")

class AutoFixAgent(RoutedAgent):
    """自动修复代理"""
    
    def __init__(self):
        super().__init__(
            AgentId("AutoFixAgent", "main"),
            "自动修复代理，执行安全的自动化修复操作"
        )
        
        # 自动修复操作库
        self.fix_operations = {
            "restart_service": {
                "description": "重启服务",
                "risk_level": "medium",
                "commands": ["systemctl restart {service}"]
            },
            "clear_cache": {
                "description": "清理缓存",
                "risk_level": "low",
                "commands": ["echo 3 > /proc/sys/vm/drop_caches"]
            },
            "kill_process": {
                "description": "终止进程",
                "risk_level": "high",
                "commands": ["kill -9 {pid}"]
            },
            "scale_service": {
                "description": "扩容服务",
                "risk_level": "low",
                "commands": ["kubectl scale deployment {deployment} --replicas={replicas}"]
            }
        }
    
    async def handle_auto_fix_request(self, request: dict, context: MessageContext) -> dict:
        """处理自动修复请求"""
        
        alert_id = request.get("alert_id")
        diagnostic_result = request.get("diagnostic_result", {})
        
        print(f"🔧 开始自动修复: {alert_id}")
        
        # 1. 选择修复操作
        fix_plan = self._create_fix_plan(diagnostic_result)
        
        if not fix_plan:
            return {
                "alert_id": alert_id,
                "status": "无法自动修复",
                "reason": "未找到合适的修复操作"
            }
        
        # 2. 执行修复操作
        execution_results = []
        
        for operation in fix_plan:
            result = await self._execute_fix_operation(operation)
            execution_results.append(result)
            
            # 如果操作失败，停止后续操作
            if not result["success"]:
                break
        
        # 3. 验证修复效果
        verification_result = await self._verify_fix(alert_id, context)
        
        return {
            "alert_id": alert_id,
            "status": "修复完成" if all(r["success"] for r in execution_results) else "修复失败",
            "fix_plan": fix_plan,
            "execution_results": execution_results,
            "verification": verification_result,
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_fix_plan(self, diagnostic_result: dict) -> List[dict]:
        """创建修复计划"""
        
        analysis = diagnostic_result.get("analysis", {})
        recommendations = diagnostic_result.get("recommendations", [])
        problem_type = analysis.get("problem_type", "")
        
        fix_plan = []
        
        # 根据问题类型选择修复操作
        if "cpu" in problem_type:
            # CPU问题的修复策略
            if "进程" in str(recommendations):
                fix_plan.append({
                    "operation": "restart_service",
                    "target": analysis.get("source_system", ""),
                    "parameters": {"service": "application"},
                    "description": "重启占用CPU过高的服务"
                })
        
        elif "memory" in problem_type:
            # 内存问题的修复策略
            fix_plan.extend([
                {
                    "operation": "clear_cache",
                    "target": "system",
                    "parameters": {},
                    "description": "清理系统缓存"
                },
                {
                    "operation": "restart_service",
                    "target": analysis.get("source_system", ""),
                    "parameters": {"service": "application"},
                    "description": "重启内存占用异常的服务"
                }
            ])
        
        elif "response_time" in problem_type:
            # 响应时间问题的修复策略
            fix_plan.append({
                "operation": "scale_service",
                "target": analysis.get("source_system", ""),
                "parameters": {"deployment": "web-app", "replicas": 3},
                "description": "扩容服务以改善响应时间"
            })
        
        return fix_plan
    
    async def _execute_fix_operation(self, operation: dict) -> dict:
        """执行修复操作"""
        
        op_type = operation["operation"]
        target = operation["target"]
        description = operation["description"]
        
        print(f"  执行操作: {description}")
        
        # 模拟操作执行
        await asyncio.sleep(1)  # 模拟执行时间
        
        # 模拟执行结果
        success = random.choice([True, True, True, False])  # 75%成功率
        
        if success:
            result = {
                "success": True,
                "operation": op_type,
                "target": target,
                "description": description,
                "output": f"操作成功执行: {description}",
                "execution_time": 1.2
            }
        else:
            result = {
                "success": False,
                "operation": op_type,
                "target": target,
                "description": description,
                "error": f"操作执行失败: 权限不足或目标不可用",
                "execution_time": 0.8
            }
        
        return result
    
    async def _verify_fix(self, alert_id: str, context: MessageContext) -> dict:
        """验证修复效果"""
        
        print(f"  验证修复效果...")
        
        # 等待系统稳定
        await asyncio.sleep(2)
        
        # 请求监控代理重新检查
        try:
            verification_request = {
                "action": "collect_metrics",
                "target_alert": alert_id
            }
            
            result = await context.send_message(
                verification_request,
                AgentId("MonitoringAgent", "main")
            )
            
            # 简化的验证逻辑
            alerts_count = result.get("active_alerts", 0)
            
            return {
                "success": alerts_count == 0,
                "active_alerts": alerts_count,
                "message": "修复验证完成" if alerts_count == 0 else "仍有活跃告警",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"验证失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

async def intelligent_ops_example():
    """智能运维系统示例"""
    
    # 1. 创建运行时
    runtime = SingleThreadedAgentRuntime()
    
    # 2. 创建运维团队
    monitoring_agent = MonitoringAgent()
    diagnostic_agent = DiagnosticAgent()
    autofix_agent = AutoFixAgent()
    
    # 3. 注册代理
    await monitoring_agent.register(runtime, "MonitoringAgent", lambda: monitoring_agent)
    await diagnostic_agent.register(runtime, "DiagnosticAgent", lambda: diagnostic_agent)
    await autofix_agent.register(runtime, "AutoFixAgent", lambda: autofix_agent)
    
    # 4. 启动运行时
    runtime_context = runtime.start()
    
    try:
        print("🚀 智能运维系统启动")
        print("=" * 50)
        
        # 5. 模拟运维场景
        scenarios = [
            {"action": "get_system_status"},
            {"action": "collect_metrics"},
            {"action": "check_alerts"}
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📊 场景 {i}: {scenario['action']}")
            print("-" * 30)
            
            result = await runtime.send_message(
                scenario,
                AgentId("MonitoringAgent", "main")
            )
            
            if scenario["action"] == "get_system_status":
                print(f"整体状态: {result['overall_status']}")
                print(f"监控系统数: {len(result['systems'])}")
                print(f"活跃告警数: {result['total_active_alerts']}")
                
                for system, status in result['systems'].items():
                    print(f"  {system}: {status['status']}")
            
            elif scenario["action"] == "collect_metrics":
                print(f"指标收集完成:")
                print(f"  监控系统: {result['systems_monitored']}")
                print(f"  生成告警: {result['alerts_generated']}")
                print(f"  活跃告警: {result['active_alerts']}")
            
            # 等待一段时间再执行下一个场景
            await asyncio.sleep(2)
        
        print(f"\n✅ 运维演示完成")
        
    finally:
        await runtime_context.stop()

# 运行示例
if __name__ == "__main__":
    asyncio.run(intelligent_ops_example())
```

## 4. 总结和最佳实践

### 4.1 框架使用最佳实践

```python
"""
AutoGen框架使用最佳实践总结
"""

# 1. 代理设计原则
class BestPracticeAgent(RoutedAgent):
    """最佳实践代理示例"""
    
    def __init__(self, name: str, specialty: str):
        super().__init__(
            AgentId("BestPractice", name),
            f"最佳实践代理 - {specialty}"
        )
        
        # ✅ 明确的职责定义
        self.specialty = specialty
        self.capabilities = self._define_capabilities()
        
        # ✅ 配置化设计
        self.config = self._load_config()
        
        # ✅ 状态管理
        self.state = {}
        self.metrics = {"processed": 0, "errors": 0}
        
        # ✅ 日志记录
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def _define_capabilities(self) -> List[str]:
        """定义代理能力"""
        return [
            "处理特定类型消息",
            "维护内部状态",
            "记录处理指标",
            "错误处理和恢复"
        ]
    
    def _load_config(self) -> dict:
        """加载配置"""
        return {
            "timeout": 30,
            "retry_count": 3,
            "batch_size": 10,
            "enable_metrics": True
        }
    
    async def handle_message(self, message: dict, context: MessageContext) -> dict:
        """消息处理最佳实践"""
        
        start_time = time.time()
        
        try:
            # ✅ 输入验证
            self._validate_input(message)
            
            # ✅ 业务逻辑处理
            result = await self._process_business_logic(message, context)
            
            # ✅ 输出验证
            self._validate_output(result)
            
            # ✅ 成功指标记录
            self.metrics["processed"] += 1
            processing_time = time.time() - start_time
            
            self.logger.info(f"消息处理成功: {processing_time:.3f}s")
            
            return result
            
        except ValidationError as e:
            # ✅ 特定异常处理
            self.metrics["errors"] += 1
            self.logger.warning(f"输入验证失败: {e}")
            raise
            
        except Exception as e:
            # ✅ 通用异常处理
            self.metrics["errors"] += 1
            self.logger.error(f"消息处理异常: {e}")
            raise
    
    def _validate_input(self, message: dict) -> None:
        """输入验证"""
        if not isinstance(message, dict):
            raise ValidationError("消息必须是字典类型")
        
        required_fields = ["type", "content"]
        for field in required_fields:
            if field not in message:
                raise ValidationError(f"缺少必需字段: {field}")
    
    async def _process_business_logic(self, message: dict, context: MessageContext) -> dict:
        """业务逻辑处理"""
        # 实现具体的业务逻辑
        return {"status": "processed", "result": "success"}
    
    def _validate_output(self, result: dict) -> None:
        """输出验证"""
        if not isinstance(result, dict):
            raise ValidationError("结果必须是字典类型")

# 2. 错误处理最佳实践
class ErrorHandlingBestPractices:
    """错误处理最佳实践"""
    
    @staticmethod
    async def robust_message_sending(runtime: AgentRuntime, message: Any, recipient: AgentId) -> dict:
        """健壮的消息发送"""
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                result = await runtime.send_message(message, recipient)
                return {"success": True, "result": result, "attempts": attempt + 1}
                
            except TimeoutError as e:
                if attempt == max_retries - 1:
                    return {"success": False, "error": "超时", "attempts": attempt + 1}
                await asyncio.sleep(base_delay * (2 ** attempt))
                
            except AgentNotFoundException as e:
                return {"success": False, "error": "代理未找到", "attempts": attempt + 1}
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"success": False, "error": str(e), "attempts": attempt + 1}
                await asyncio.sleep(base_delay * (2 ** attempt))
        
        return {"success": False, "error": "未知错误", "attempts": max_retries}

# 3. 性能优化最佳实践
class PerformanceOptimization:
    """性能优化最佳实践"""
    
    @staticmethod
    async def batch_processing_example(runtime: AgentRuntime, messages: List[Any], recipient: AgentId):
        """批量处理示例"""
        
        batch_size = 10
        results = []
        
        # 分批处理
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            
            # 并发处理批次内的消息
            tasks = [
                runtime.send_message(msg, recipient)
                for msg in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    @staticmethod
    async def connection_pooling_example():
        """连接池示例"""
        
        # 使用连接池管理资源
        class ConnectionPool:
            def __init__(self, max_connections: int = 10):
                self.max_connections = max_connections
                self.available_connections = asyncio.Queue(maxsize=max_connections)
                self.total_connections = 0
            
            async def get_connection(self):
                if self.available_connections.empty() and self.total_connections < self.max_connections:
                    # 创建新连接
                    connection = await self._create_connection()
                    self.total_connections += 1
                    return connection
                else:
                    # 等待可用连接
                    return await self.available_connections.get()
            
            async def return_connection(self, connection):
                await self.available_connections.put(connection)
            
            async def _create_connection(self):
                # 模拟创建连接
                return {"id": self.total_connections, "created_at": datetime.now()}

# 4. 监控和日志最佳实践
class MonitoringBestPractices:
    """监控和日志最佳实践"""
    
    @staticmethod
    def setup_logging():
        """设置日志"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('autogen.log'),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def setup_metrics():
        """设置指标收集"""
        
        # 使用Prometheus客户端库
        from prometheus_client import Counter, Histogram, Gauge
        
        metrics = {
            "messages_processed": Counter('autogen_messages_processed_total', 'Total processed messages'),
            "processing_time": Histogram('autogen_processing_time_seconds', 'Message processing time'),
            "active_agents": Gauge('autogen_active_agents', 'Number of active agents')
        }
        
        return metrics

# 5. 配置管理最佳实践
class ConfigurationBestPractices:
    """配置管理最佳实践"""
    
    @staticmethod
    def load_configuration():
        """加载配置"""
        
        import os
        from typing import Optional
        
        class Config:
            # 环境变量配置
            OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
            MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
            
            # 运行时配置
            MAX_CONCURRENT_AGENTS: int = int(os.getenv("MAX_CONCURRENT_AGENTS", "100"))
            MESSAGE_TIMEOUT: int = int(os.getenv("MESSAGE_TIMEOUT", "30"))
            
            # 日志配置
            LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
            LOG_FILE: str = os.getenv("LOG_FILE", "autogen.log")
            
            @classmethod
            def validate(cls) -> bool:
                """验证配置"""
                if not cls.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY is required")
                
                if cls.MAX_CONCURRENT_AGENTS <= 0:
                    raise ValueError("MAX_CONCURRENT_AGENTS must be positive")
                
                return True
        
        return Config

# 6. 测试最佳实践
class TestingBestPractices:
    """测试最佳实践"""
    
    @staticmethod
    async def test_agent_behavior():
        """代理行为测试"""
        
        import unittest
        from unittest.mock import Mock, AsyncMock
        
        class TestMyAgent(unittest.IsolatedAsyncioTestCase):
            
            async def asyncSetUp(self):
                self.runtime = Mock()
                self.agent = BestPracticeAgent("test", "testing")
            
            async def test_message_processing(self):
                # 准备测试数据
                message = {"type": "test", "content": "hello"}
                context = Mock()
                
                # 执行测试
                result = await self.agent.handle_message(message, context)
                
                # 验证结果
                self.assertEqual(result["status"], "processed")
                self.assertEqual(self.agent.metrics["processed"], 1)
            
            async def test_error_handling(self):
                # 测试错误处理
                invalid_message = {"invalid": "data"}
                context = Mock()
                
                with self.assertRaises(ValidationError):
                    await self.agent.handle_message(invalid_message, context)
                
                self.assertEqual(self.agent.metrics["errors"], 1)

print("""
🎯 AutoGen框架使用最佳实践总结:

1. 代理设计原则:
   - 单一职责原则
   - 明确的输入输出定义
   - 完善的错误处理
   - 状态管理和指标收集

2. 错误处理:
   - 分层异常处理
   - 重试机制
   - 优雅降级
   - 详细的错误日志

3. 性能优化:
   - 批量处理
   - 连接池管理
   - 异步并发
   - 资源复用

4. 监控和日志:
   - 结构化日志
   - 关键指标收集
   - 分布式追踪
   - 告警机制

5. 配置管理:
   - 环境变量配置
   - 配置验证
   - 分环境配置
   - 热更新支持

6. 测试策略:
   - 单元测试
   - 集成测试
   - 性能测试
   - 混沌工程

遵循这些最佳实践，可以构建出高质量、高性能、高可靠性的AutoGen应用系统。
""")
```

通过这些完整的实战案例，开发者可以：

1. **快速上手**：从简单对话代理开始学习框架基础
2. **理解协作**：通过多代理协作案例掌握复杂系统设计
3. **企业应用**：学习如何构建企业级的智能系统
4. **最佳实践**：掌握框架使用的最佳实践和优化技巧

这些案例涵盖了从基础使用到高级应用的各个层面，为不同水平的开发者提供了完整的学习路径。

---
