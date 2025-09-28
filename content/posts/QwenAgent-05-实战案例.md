---
title: "10 - 实战案例与最佳实践"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['源码分析', '技术文档', '最佳实践']
categories: ['qwenagent', '技术分析']
description: "10 - 实战案例与最佳实践的深入技术分析文档"
keywords: ['源码分析', '技术文档', '最佳实践']
author: "技术分析师"
weight: 1
---

## 📝 概述

本文档通过丰富的实战案例，展示Qwen-Agent框架的实际应用场景，并提供最佳实践指南。从基础使用到高级定制，从单一功能到复合应用，帮助开发者深入理解框架的能力和使用技巧。

## 🚀 基础应用案例

### 案例1：智能客服机器人

#### 业务需求
- 回答常见问题
- 查询订单状态  
- 处理退换货申请
- 智能转人工客服

#### 实现方案

```python
import os
import json
from typing import List, Dict
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI

# 1. 自定义订单查询工具
@register_tool('order_query')
class OrderQueryTool(BaseTool):
    """订单查询工具"""
    description = '查询用户的订单信息，包括订单状态、物流信息等'
    parameters = {
        'type': 'object',
        'properties': {
            'order_id': {
                'type': 'string',
                'description': '订单号，格式如：ORD20240101001'
            },
            'phone': {
                'type': 'string', 
                'description': '下单时使用的手机号码'
            }
        },
        'required': ['order_id']
    }
    
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # 模拟订单数据库连接
        self.order_db = self._init_order_db()
    
    def _init_order_db(self) -> Dict:
        """初始化模拟订单数据"""
        return {
            'ORD20240101001': {
                'status': '已发货',
                'items': [{'name': 'iPhone 15', 'quantity': 1, 'price': 5999}],
                'total': 5999,
                'shipping': {
                    'company': '顺丰快递',
                    'tracking_no': 'SF1234567890',
                    'status': '运输中',
                    'estimated_delivery': '2024-01-05'
                },
                'customer_phone': '138****8888'
            },
            'ORD20240101002': {
                'status': '已完成',
                'items': [{'name': 'MacBook Pro', 'quantity': 1, 'price': 12999}],
                'total': 12999,
                'shipping': {
                    'company': '京东物流',
                    'tracking_no': 'JD9876543210',
                    'status': '已签收',
                    'delivery_date': '2024-01-03'
                },
                'customer_phone': '139****9999'
            }
        }
    
    def call(self, params: str, **kwargs) -> str:
        """执行订单查询"""
        try:
            params_dict = self._verify_json_format_args(params)
            order_id = params_dict['order_id']
            phone = params_dict.get('phone', '')
            
            # 查询订单信息
            if order_id not in self.order_db:
                return f"未找到订单号为 {order_id} 的订单，请检查订单号是否正确。"
            
            order = self.order_db[order_id]
            
            # 如果提供了手机号，验证身份
            if phone and not order['customer_phone'].endswith(phone[-4:]):
                return "手机号码验证失败，请确认您输入的是下单时使用的手机号码。"
            
            # 格式化返回结果
            result = f"""
📦 订单信息查询结果

订单号：{order_id}
订单状态：{order['status']}
订单总额：¥{order['total']}

商品清单：
"""
            for item in order['items']:
                result += f"• {item['name']} × {item['quantity']} - ¥{item['price']}\n"
            
            if 'shipping' in order:
                shipping = order['shipping']
                result += f"""
🚚 物流信息：
快递公司：{shipping['company']}
运单号：{shipping['tracking_no']}
物流状态：{shipping['status']}
"""
                if 'estimated_delivery' in shipping:
                    result += f"预计送达：{shipping['estimated_delivery']}\n"
                elif 'delivery_date' in shipping:
                    result += f"签收时间：{shipping['delivery_date']}\n"
            
            return result.strip()
            
        except Exception as e:
            return f"查询订单时发生错误：{str(e)}"

# 2. 知识库工具（FAQ）
@register_tool('faq_search')
class FAQSearchTool(BaseTool):
    """常见问题搜索工具"""
    description = '搜索常见问题的答案，支持退换货、配送、支付等问题'
    parameters = {
        'type': 'object',
        'properties': {
            'question': {
                'type': 'string',
                'description': '用户的问题描述'
            },
            'category': {
                'type': 'string',
                'enum': ['退换货', '配送', '支付', '产品', '其他'],
                'description': '问题分类'
            }
        },
        'required': ['question']
    }
    
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.faq_db = self._load_faq_data()
    
    def _load_faq_data(self) -> List[Dict]:
        """加载FAQ数据"""
        return [
            {
                'category': '退换货',
                'question': '如何申请退货？',
                'answer': '您可以在订单详情页点击"申请退货"，或者联系客服。退货条件：商品完好无损，在7天无理由退货期内。'
            },
            {
                'category': '配送',
                'question': '多久可以收到货？',
                'answer': '一般情况下，现货商品1-3个工作日内发货，快递配送通常需要1-5个工作日，具体时间以物流公司为准。'
            },
            {
                'category': '支付',
                'question': '支持哪些支付方式？',
                'answer': '我们支持微信支付、支付宝、银行卡支付、花呗分期等多种支付方式。'
            },
            {
                'category': '产品',
                'question': 'iPhone 15有哪些颜色？',
                'answer': 'iPhone 15提供以下颜色选择：粉色、黄色、绿色、蓝色、黑色。不同型号可能颜色选择略有差异。'
            }
        ]
    
    def call(self, params: str, **kwargs) -> str:
        """搜索FAQ答案"""
        try:
            params_dict = self._verify_json_format_args(params)
            question = params_dict['question'].lower()
            category = params_dict.get('category', '')
            
            # 简单的关键词匹配
            matches = []
            for faq in self.faq_db:
                if category and faq['category'] != category:
                    continue
                    
                # 检查问题关键词匹配
                if any(keyword in question for keyword in ['退货', '退款', '换货']):
                    if '退换货' in faq['category']:
                        matches.append(faq)
                elif any(keyword in question for keyword in ['配送', '发货', '快递', '收货']):
                    if '配送' in faq['category']:
                        matches.append(faq)
                elif any(keyword in question for keyword in ['支付', '付款', '花呗']):
                    if '支付' in faq['category']:
                        matches.append(faq)
                elif any(keyword in question for keyword in ['颜色', 'iphone', '产品']):
                    if '产品' in faq['category']:
                        matches.append(faq)
            
            if not matches:
                return "抱歉，我没有找到相关的答案。您可以联系人工客服获得更详细的帮助。"
            
            # 返回最匹配的答案
            best_match = matches[0]
            return f"💡 关于「{best_match['question']}」\n\n{best_match['answer']}"
            
        except Exception as e:
            return f"搜索FAQ时发生错误：{str(e)}"

# 3. 人工客服转接工具
@register_tool('transfer_to_human')
class TransferToHumanTool(BaseTool):
    """转人工客服工具"""
    description = '当无法解决用户问题时，转接到人工客服'
    parameters = {
        'type': 'object',
        'properties': {
            'reason': {
                'type': 'string',
                'description': '转接原因'
            },
            'urgency': {
                'type': 'string',
                'enum': ['低', '中', '高'],
                'description': '紧急程度'
            }
        },
        'required': ['reason']
    }
    
    def call(self, params: str, **kwargs) -> str:
        """执行转人工客服"""
        try:
            params_dict = self._verify_json_format_args(params)
            reason = params_dict['reason']
            urgency = params_dict.get('urgency', '中')
            
            # 生成转接工单
            ticket_id = f"TICKET{hash(reason) % 100000:05d}"
            
            result = f"""
🎧 正在为您转接人工客服...

工单号：{ticket_id}
转接原因：{reason}
优先级：{urgency}

预计等待时间：
• 高优先级：1-3分钟
• 中优先级：3-10分钟  
• 低优先级：10-30分钟

请稍候，人工客服将尽快为您服务！
"""
            return result.strip()
            
        except Exception as e:
            return f"转接人工客服时发生错误：{str(e)}"

def create_customer_service_bot():
    """创建智能客服机器人"""
    
    # LLM配置
    llm_cfg = {
        'model': 'qwen3-235b-a22b',
        'model_type': 'qwen_dashscope',
        'generate_cfg': {
            'top_p': 0.8,
            'temperature': 0.3,  # 较低的温度保证回复的一致性
            'max_input_tokens': 6000
        }
    }
    
    # 系统提示
    system_message = '''你是一个专业的智能客服助手，名叫"小Q"。你的任务是帮助用户解决购物相关的问题。

🎯 服务原则：
1. 态度友好、耐心、专业
2. 准确理解用户需求
3. 优先使用工具查询真实信息
4. 无法解决时及时转接人工客服

🛠️ 可用工具：
- order_query：查询订单状态和物流信息
- faq_search：搜索常见问题答案  
- transfer_to_human：转接人工客服

💡 回复风格：
- 使用友好的称呼（如"亲"、"您"）
- 适当使用表情符号增加亲和力
- 结构化展示查询结果
- 主动提供相关建议

🚨 转人工情况：
- 复杂的技术问题
- 投诉和纠纷处理
- 特殊政策咨询
- 用户明确要求

请始终保持专业和热情的服务态度！'''
    
    # 工具列表
    tools = [
        'order_query',
        'faq_search', 
        'transfer_to_human'
    ]
    
    # 创建客服机器人
    customer_service = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=system_message,
        name='智能客服小Q',
        description='专业的购物客服助手，可以查询订单、解答问题、转接人工'
    )
    
    return customer_service

# 4. 启动客服系统
def start_customer_service():
    """启动智能客服系统"""
    
    bot = create_customer_service_bot()
    
    # 配置Web界面
    chatbot_config = {
        'user.name': '客户',
        'user.avatar': '👤',
        'agent.avatar': '🤖',
        'input.placeholder': '请描述您遇到的问题，我来帮您解决～',
        'prompt.suggestions': [
            '我想查询订单ORD20240101001的状态',
            '如何申请退货？',
            '支持哪些支付方式？',
            'iPhone 15有哪些颜色可选？',
            '转人工客服'
        ]
    }
    
    # 启动Web界面
    web_ui = WebUI(bot, chatbot_config=chatbot_config)
    web_ui.run(server_name='0.0.0.0', server_port=7860)

if __name__ == '__main__':
    start_customer_service()
```

#### 实际使用效果

```python
# 测试智能客服功能
def test_customer_service():
    bot = create_customer_service_bot()
    
    test_cases = [
        "你好，我想查询订单ORD20240101001的状态",
        "iPhone 15都有什么颜色？",
        "我想申请退货，需要什么条件？",  
        "这个问题比较复杂，请帮我转人工客服"
    ]
    
    for query in test_cases:
        print(f"\n👤 用户：{query}")
        print("🤖 客服小Q：")
        
        messages = [{'role': 'user', 'content': query}]
        response_text = ""
        
        for response in bot.run(messages):
            if response:
                response_text = response[-1].get('content', '')
        
        print(response_text)
        print("-" * 50)
```

### 案例2：代码助手Agent

#### 业务需求
- 代码审查和优化建议
- 自动化测试用例生成
- 文档生成
- 代码重构建议

#### 实现方案

```python
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import ast
import subprocess
import tempfile
import os

@register_tool('code_analyzer')
class CodeAnalyzerTool(BaseTool):
    """代码分析工具"""
    description = '分析Python代码质量，提供优化建议和潜在问题检测'
    parameters = {
        'type': 'object',
        'properties': {
            'code': {
                'type': 'string',
                'description': '要分析的Python代码'
            },
            'analysis_type': {
                'type': 'string',
                'enum': ['syntax', 'style', 'complexity', 'security', 'all'],
                'description': '分析类型'
            }
        },
        'required': ['code']
    }
    
    def call(self, params: str, **kwargs) -> str:
        """执行代码分析"""
        try:
            params_dict = self._verify_json_format_args(params)
            code = params_dict['code']
            analysis_type = params_dict.get('analysis_type', 'all')
            
            results = []
            
            # 语法检查
            if analysis_type in ['syntax', 'all']:
                syntax_result = self._check_syntax(code)
                if syntax_result:
                    results.append(f"🔍 语法检查:\n{syntax_result}")
            
            # 代码风格检查
            if analysis_type in ['style', 'all']:
                style_result = self._check_style(code)
                if style_result:
                    results.append(f"🎨 代码风格:\n{style_result}")
            
            # 复杂度分析
            if analysis_type in ['complexity', 'all']:
                complexity_result = self._check_complexity(code)
                if complexity_result:
                    results.append(f"📊 复杂度分析:\n{complexity_result}")
            
            # 安全检查
            if analysis_type in ['security', 'all']:
                security_result = self._check_security(code)
                if security_result:
                    results.append(f"🔒 安全检查:\n{security_result}")
            
            if not results:
                return "✅ 代码分析完成，未发现明显问题！"
            
            return "\n\n".join(results)
            
        except Exception as e:
            return f"代码分析时发生错误：{str(e)}"
    
    def _check_syntax(self, code: str) -> str:
        """语法检查"""
        try:
            ast.parse(code)
            return ""
        except SyntaxError as e:
            return f"语法错误：第{e.lineno}行，{e.msg}"
    
    def _check_style(self, code: str) -> str:
        """代码风格检查"""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # 检查行长度
            if len(line) > 88:
                issues.append(f"第{i}行过长({len(line)}字符)，建议不超过88字符")
            
            # 检查缩进
            if line.startswith(' ') and not line.startswith('    '):
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces % 4 != 0:
                    issues.append(f"第{i}行缩进不规范，应使用4的倍数个空格")
            
            # 检查import顺序
            if line.strip().startswith('from ') and i > 1:
                prev_line = lines[i-2].strip()
                if prev_line.startswith('import ') and not prev_line.startswith('from '):
                    issues.append(f"第{i}行：from import应该在import语句之后")
        
        return "\n".join(issues) if issues else ""
    
    def _check_complexity(self, code: str) -> str:
        """复杂度分析"""
        try:
            tree = ast.parse(code)
            complexity_analyzer = ComplexityAnalyzer()
            complexity_analyzer.visit(tree)
            return complexity_analyzer.get_report()
        except:
            return ""
    
    def _check_security(self, code: str) -> str:
        """基础安全检查"""
        issues = []
        dangerous_functions = ['eval', 'exec', 'compile', '__import__']
        
        for func in dangerous_functions:
            if func in code:
                issues.append(f"发现潜在风险函数：{func}()")
        
        if 'subprocess' in code and 'shell=True' in code:
            issues.append("使用subprocess时启用shell=True可能存在命令注入风险")
        
        return "\n".join(issues) if issues else ""

class ComplexityAnalyzer(ast.NodeVisitor):
    """代码复杂度分析器"""
    
    def __init__(self):
        self.functions = {}
        self.current_function = None
        
    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.functions[node.name] = {
            'lines': node.end_lineno - node.lineno + 1,
            'complexity': 1  # 基础复杂度
        }
        self.generic_visit(node)
        self.current_function = None
    
    def visit_If(self, node):
        if self.current_function:
            self.functions[self.current_function]['complexity'] += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        if self.current_function:
            self.functions[self.current_function]['complexity'] += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        if self.current_function:
            self.functions[self.current_function]['complexity'] += 1
        self.generic_visit(node)
    
    def get_report(self) -> str:
        if not self.functions:
            return ""
        
        report = []
        for func_name, metrics in self.functions.items():
            complexity = metrics['complexity']
            lines = metrics['lines']
            
            if complexity > 10:
                report.append(f"函数 {func_name}：圈复杂度过高({complexity})，建议重构")
            elif lines > 50:
                report.append(f"函数 {func_name}：函数过长({lines}行)，建议拆分")
        
        return "\n".join(report)

@register_tool('test_generator')
class TestGeneratorTool(BaseTool):
    """测试用例生成工具"""
    description = '为Python函数自动生成单元测试用例'
    parameters = {
        'type': 'object',
        'properties': {
            'function_code': {
                'type': 'string',
                'description': '要生成测试的函数代码'
            },
            'test_framework': {
                'type': 'string',
                'enum': ['unittest', 'pytest'],
                'description': '测试框架类型'
            }
        },
        'required': ['function_code']
    }
    
    def call(self, params: str, **kwargs) -> str:
        """生成测试用例"""
        try:
            params_dict = self._verify_json_format_args(params)
            function_code = params_dict['function_code']
            framework = params_dict.get('test_framework', 'unittest')
            
            # 解析函数信息
            tree = ast.parse(function_code)
            function_info = self._extract_function_info(tree)
            
            if not function_info:
                return "未检测到有效的函数定义"
            
            # 生成测试用例
            if framework == 'unittest':
                test_code = self._generate_unittest(function_info, function_code)
            else:
                test_code = self._generate_pytest(function_info, function_code)
            
            return f"🧪 自动生成的测试用例：\n\n```python\n{test_code}\n```"
            
        except Exception as e:
            return f"生成测试用例时发生错误：{str(e)}"
    
    def _extract_function_info(self, tree):
        """提取函数信息"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'docstring': ast.get_docstring(node) or ""
                }
        return None
    
    def _generate_unittest(self, func_info, original_code) -> str:
        """生成unittest测试用例"""
        func_name = func_info['name']
        class_name = f"Test{func_name.capitalize()}"
        
        test_template = f'''import unittest
{original_code}

class {class_name}(unittest.TestCase):
    """测试 {func_name} 函数"""
    
    def test_{func_name}_basic(self):
        """基本功能测试"""
        # TODO: 添加基本测试用例
        result = {func_name}()  # 根据函数参数调整
        self.assertIsNotNone(result)
    
    def test_{func_name}_edge_cases(self):
        """边界条件测试"""
        # TODO: 测试边界条件
        pass
    
    def test_{func_name}_invalid_input(self):
        """无效输入测试"""
        # TODO: 测试异常输入
        with self.assertRaises(ValueError):
            {func_name}(None)  # 根据实际情况调整

if __name__ == '__main__':
    unittest.main()
'''
        return test_template
    
    def _generate_pytest(self, func_info, original_code) -> str:
        """生成pytest测试用例"""
        func_name = func_info['name']
        
        test_template = f'''import pytest
{original_code}

class Test{func_name.capitalize()}:
    """测试 {func_name} 函数"""
    
    def test_{func_name}_basic(self):
        """基本功能测试"""
        # TODO: 添加基本测试用例
        result = {func_name}()  # 根据函数参数调整
        assert result is not None
    
    def test_{func_name}_edge_cases(self):
        """边界条件测试"""
        # TODO: 测试边界条件
        pass
    
    def test_{func_name}_invalid_input(self):
        """无效输入测试"""
        # TODO: 测试异常输入
        with pytest.raises(ValueError):
            {func_name}(None)  # 根据实际情况调整
    
    @pytest.mark.parametrize("input_value,expected", [
        # TODO: 添加参数化测试数据
        (1, 1),
        (2, 2),
    ])
    def test_{func_name}_parametrized(self, input_value, expected):
        """参数化测试"""
        result = {func_name}(input_value)
        assert result == expected
'''
        return test_template

def create_code_assistant():
    """创建代码助手"""
    
    llm_cfg = {
        'model': 'qwen3-235b-a22b',
        'model_type': 'qwen_dashscope',
        'generate_cfg': {
            'top_p': 0.8,
            'temperature': 0.2,  # 代码相关任务使用较低温度
            'max_input_tokens': 8000
        }
    }
    
    system_message = '''你是一个专业的代码助手，专门帮助开发者进行代码审查、优化和测试。

🎯 核心能力：
1. 代码质量分析和改进建议
2. 自动生成测试用例
3. 代码重构建议
4. 性能优化指导
5. 最佳实践推荐

🛠️ 可用工具：
- code_analyzer：分析代码质量，检测潜在问题
- test_generator：自动生成单元测试用例
- code_interpreter：执行代码并查看结果

💡 工作流程：
1. 理解用户的代码需求
2. 使用工具进行深入分析
3. 提供具体的改进建议
4. 生成相关的测试用例
5. 验证代码的正确性

📋 回复格式：
- 使用代码块展示代码
- 提供清晰的解释和建议
- 标注关键改进点
- 给出实用的最佳实践

请始终以专业、准确、实用的方式帮助开发者提升代码质量！'''
    
    tools = [
        'code_analyzer',
        'test_generator', 
        'code_interpreter'
    ]
    
    return Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=system_message,
        name='代码助手',
        description='专业的代码审查和优化助手'
    )

# 使用示例
def demo_code_assistant():
    """代码助手使用演示"""
    assistant = create_code_assistant()
    
    sample_code = '''
def calculate_average(numbers):
    sum = 0
    for i in range(len(numbers)):
        sum = sum + numbers[i]
    return sum / len(numbers)
'''
    
    query = f"请帮我分析这段代码并提供优化建议：\n{sample_code}"
    
    messages = [{'role': 'user', 'content': query}]
    
    print("🤖 代码助手分析结果：")
    for response in assistant.run(messages):
        if response:
            print(response[-1].get('content', ''))
```

## 🏢 企业级应用案例

### 案例3：智能文档处理系统

#### 业务场景
- 合同文档审查
- 财务报表分析
- 政策文档解读
- 多语言文档翻译

```python
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import pandas as pd
import json
import re
from pathlib import Path

@register_tool('contract_analyzer')
class ContractAnalyzerTool(BaseTool):
    """合同分析工具"""
    description = '分析合同文档，提取关键信息和风险点'
    parameters = {
        'type': 'object',
        'properties': {
            'contract_text': {
                'type': 'string',
                'description': '合同文本内容'
            },
            'analysis_focus': {
                'type': 'string',
                'enum': ['全面分析', '风险评估', '关键条款', '财务条款'],
                'description': '分析重点'
            }
        },
        'required': ['contract_text']
    }
    
    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = self._verify_json_format_args(params)
            contract_text = params_dict['contract_text']
            focus = params_dict.get('analysis_focus', '全面分析')
            
            analysis_result = {
                'basic_info': self._extract_basic_info(contract_text),
                'key_terms': self._extract_key_terms(contract_text),
                'risk_assessment': self._assess_risks(contract_text),
                'financial_terms': self._extract_financial_terms(contract_text)
            }
            
            return self._format_analysis_result(analysis_result, focus)
            
        except Exception as e:
            return f"合同分析错误：{str(e)}"
    
    def _extract_basic_info(self, text: str) -> dict:
        """提取基本信息"""
        info = {}
        
        # 提取日期
        date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})'
        dates = re.findall(date_pattern, text)
        if dates:
            info['contract_date'] = dates[0]
        
        # 提取金额
        amount_pattern = r'([￥¥$]\s*[\d,]+\.?\d*[万千百十元美元]?|[\d,]+\.?\d*\s*[万千百十元美元])'
        amounts = re.findall(amount_pattern, text)
        if amounts:
            info['amounts'] = amounts[:3]  # 取前3个金额
        
        # 提取期限
        term_pattern = r'(\d+\s*[年月日天周]|[一二三四五六七八九十]+\s*[年月日天周])'
        terms = re.findall(term_pattern, text)
        if terms:
            info['terms'] = terms[:2]
        
        return info
    
    def _extract_key_terms(self, text: str) -> list:
        """提取关键条款"""
        key_terms = []
        
        # 常见关键条款关键词
        keywords = [
            '违约责任', '付款条件', '交付时间', '质量标准', 
            '保密条款', '知识产权', '争议解决', '终止条件'
        ]
        
        for keyword in keywords:
            if keyword in text:
                # 提取包含关键词的句子
                sentences = text.split('。')
                for sentence in sentences:
                    if keyword in sentence and len(sentence.strip()) > 10:
                        key_terms.append(f"{keyword}：{sentence.strip()[:100]}...")
                        break
        
        return key_terms
    
    def _assess_risks(self, text: str) -> list:
        """风险评估"""
        risks = []
        
        # 高风险关键词
        high_risk_keywords = [
            '不承担责任', '免责', '不可抗力', '单方面解除',
            '不退还', '最终解释权', '甲方有权'
        ]
        
        for keyword in high_risk_keywords:
            if keyword in text:
                risks.append(f"⚠️ 发现风险条款：包含'{keyword}'")
        
        # 检查是否缺少重要条款
        important_clauses = ['违约责任', '争议解决', '付款方式']
        for clause in important_clauses:
            if clause not in text:
                risks.append(f"⚠️ 缺少重要条款：{clause}")
        
        return risks
    
    def _extract_financial_terms(self, text: str) -> dict:
        """提取财务条款"""
        financial = {}
        
        # 提取付款方式
        payment_methods = ['银行转账', '支票', '现金', '信用证', '电汇']
        for method in payment_methods:
            if method in text:
                financial['payment_method'] = method
                break
        
        # 提取付款期限
        payment_terms = re.findall(r'(\d+日内付款|\d+天内支付)', text)
        if payment_terms:
            financial['payment_terms'] = payment_terms[0]
        
        return financial
    
    def _format_analysis_result(self, analysis: dict, focus: str) -> str:
        """格式化分析结果"""
        result = "📋 合同分析报告\n\n"
        
        if focus in ['全面分析', '关键条款']:
            result += "📊 基本信息：\n"
            for key, value in analysis['basic_info'].items():
                result += f"• {key}: {value}\n"
            result += "\n"
        
        if focus in ['全面分析', '关键条款']:
            result += "🔑 关键条款：\n"
            for term in analysis['key_terms']:
                result += f"• {term}\n"
            result += "\n"
        
        if focus in ['全面分析', '风险评估']:
            result += "⚠️ 风险评估：\n"
            if analysis['risk_assessment']:
                for risk in analysis['risk_assessment']:
                    result += f"{risk}\n"
            else:
                result += "✅ 未发现明显风险点\n"
            result += "\n"
        
        if focus in ['全面分析', '财务条款']:
            result += "💰 财务条款：\n"
            for key, value in analysis['financial_terms'].items():
                result += f"• {key}: {value}\n"
        
        return result

@register_tool('document_summarizer')
class DocumentSummarizerTool(BaseTool):
    """文档摘要工具"""
    description = '生成文档摘要，支持多种摘要类型'
    parameters = {
        'type': 'object',
        'properties': {
            'document_text': {
                'type': 'string',
                'description': '文档内容'
            },
            'summary_type': {
                'type': 'string',
                'enum': ['执行摘要', '详细摘要', '要点摘要', '结构化摘要'],
                'description': '摘要类型'
            },
            'max_length': {
                'type': 'integer',
                'description': '摘要最大长度（字符数）',
                'default': 500
            }
        },
        'required': ['document_text']
    }
    
    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = self._verify_json_format_args(params)
            document_text = params_dict['document_text']
            summary_type = params_dict.get('summary_type', '执行摘要')
            max_length = params_dict.get('max_length', 500)
            
            if summary_type == '执行摘要':
                return self._create_executive_summary(document_text, max_length)
            elif summary_type == '详细摘要':
                return self._create_detailed_summary(document_text, max_length)
            elif summary_type == '要点摘要':
                return self._create_bullet_summary(document_text)
            elif summary_type == '结构化摘要':
                return self._create_structured_summary(document_text)
                
        except Exception as e:
            return f"文档摘要生成错误：{str(e)}"
    
    def _create_executive_summary(self, text: str, max_length: int) -> str:
        """创建执行摘要"""
        # 简单的执行摘要生成逻辑
        sentences = text.split('。')
        important_sentences = []
        
        # 选择包含关键词的句子
        key_indicators = ['重要', '关键', '主要', '核心', '目标', '结果', '建议']
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in key_indicators):
                important_sentences.append(sentence.strip())
            if len('。'.join(important_sentences)) > max_length:
                break
        
        summary = '。'.join(important_sentences[:3])
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return f"📋 执行摘要：\n{summary}"
    
    def _create_bullet_summary(self, text: str) -> str:
        """创建要点摘要"""
        paragraphs = text.split('\n\n')
        bullet_points = []
        
        for para in paragraphs:
            if len(para.strip()) > 20:  # 忽略太短的段落
                # 提取段落的核心内容
                sentences = para.split('。')
                if sentences:
                    main_sentence = sentences[0].strip()
                    if len(main_sentence) > 10:
                        bullet_points.append(f"• {main_sentence[:100]}...")
        
        return f"🔸 要点摘要：\n" + '\n'.join(bullet_points[:5])
    
    def _create_structured_summary(self, text: str) -> str:
        """创建结构化摘要"""
        return f"""📊 结构化摘要：

🎯 主要内容：
{text[:200]}...

🔑 关键信息：
• 文档长度：约{len(text)}字符
• 段落数量：{len(text.split('\\n\\n'))}段
• 主要关键词：{self._extract_keywords(text)}

💡 摘要建议：
基于文档内容，建议关注重点信息和关键决策点。
"""
    
    def _extract_keywords(self, text: str) -> str:
        """提取关键词"""
        # 简单的关键词提取
        common_words = set(['的', '是', '和', '在', '有', '了', '为', '与', '及', '等'])
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        word_freq = {}
        
        for word in words:
            if len(word) >= 2 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 返回频率最高的前5个词
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return ', '.join([word for word, _ in top_words])

def create_document_processor():
    """创建文档处理系统"""
    
    llm_cfg = {
        'model': 'qwen3-235b-a22b',
        'model_type': 'qwen_dashscope',
        'generate_cfg': {
            'top_p': 0.7,
            'temperature': 0.3,
            'max_input_tokens': 12000  # 支持更长的文档
        }
    }
    
    system_message = '''你是一个专业的智能文档处理助手，擅长分析各类商业文档。

🎯 核心功能：
1. 合同文档深度分析和风险评估
2. 财务报表数据提取和分析
3. 政策文档解读和要点提取
4. 多格式文档摘要生成

🛠️ 专业工具：
- contract_analyzer：深度分析合同条款和风险点
- document_summarizer：生成多种类型的文档摘要
- doc_parser：解析各种格式的文档文件

💼 服务标准：
- 准确识别关键信息和风险点
- 提供结构化的分析报告
- 给出专业的建议和改进意见
- 保护商业信息的机密性

📋 分析维度：
- 合规性检查
- 风险点识别
- 关键条款提取
- 财务影响分析
- 法律风险评估

请始终保持专业、准确、保密的服务态度！'''
    
    tools = [
        'contract_analyzer',
        'document_summarizer',
        'doc_parser'
    ]
    
    return Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=system_message,
        name='智能文档处理助手',
        description='专业的合同分析和文档处理助手'
    )
```

### 案例4：多模态内容创作助手

#### 应用场景
- 社交媒体内容创作
- 产品文案生成
- 图文并茂的报告制作
- 多媒体内容策划

```python
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem

@register_tool('content_planner')
class ContentPlannerTool(BaseTool):
    """内容策划工具"""
    description = '制定内容创作计划，包括主题、结构和素材需求'
    parameters = {
        'type': 'object',
        'properties': {
            'topic': {
                'type': 'string',
                'description': '内容主题'
            },
            'content_type': {
                'type': 'string',
                'enum': ['社交媒体', '产品介绍', '技术文章', '营销文案', '教程指南'],
                'description': '内容类型'
            },
            'target_audience': {
                'type': 'string',
                'description': '目标受众'
            },
            'platform': {
                'type': 'string',
                'enum': ['微信公众号', '知乎', '小红书', '抖音', 'B站', '官网'],
                'description': '发布平台'
            }
        },
        'required': ['topic', 'content_type']
    }
    
    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = self._verify_json_format_args(params)
            topic = params_dict['topic']
            content_type = params_dict['content_type']
            audience = params_dict.get('target_audience', '通用用户')
            platform = params_dict.get('platform', '通用平台')
            
            plan = self._create_content_plan(topic, content_type, audience, platform)
            return self._format_plan(plan)
            
        except Exception as e:
            return f"内容策划错误：{str(e)}"
    
    def _create_content_plan(self, topic, content_type, audience, platform):
        """创建内容计划"""
        
        # 平台特性配置
        platform_specs = {
            '微信公众号': {'word_limit': 3000, 'style': '深度文章', 'media': '图文'},
            '知乎': {'word_limit': 2000, 'style': '专业分析', 'media': '图文+数据'},
            '小红书': {'word_limit': 1000, 'style': '生活化', 'media': '精美图片+简洁文字'},
            '抖音': {'word_limit': 50, 'style': '短视频脚本', 'media': '视频+字幕'},
            'B站': {'word_limit': 500, 'style': '视频介绍', 'media': '视频+封面'},
            '官网': {'word_limit': 1500, 'style': '专业介绍', 'media': '图文+图表'}
        }
        
        specs = platform_specs.get(platform, {'word_limit': 1500, 'style': '通用', 'media': '图文'})
        
        # 内容结构模板
        structure_templates = {
            '社交媒体': ['吸引眼球的开头', '核心内容阐述', '互动引导', '行动召唤'],
            '产品介绍': ['产品亮点概述', '功能详细介绍', '使用场景展示', '购买引导'],
            '技术文章': ['问题背景', '技术方案', '实现步骤', '总结和展望'],
            '营销文案': ['痛点分析', '解决方案', '产品优势', '优惠信息'],
            '教程指南': ['前置知识', '步骤分解', '注意事项', '进阶建议']
        }
        
        structure = structure_templates.get(content_type, ['引言', '主体', '结论'])
        
        return {
            'topic': topic,
            'content_type': content_type,
            'audience': audience,
            'platform': platform,
            'specs': specs,
            'structure': structure,
            'keywords': self._generate_keywords(topic, content_type),
            'media_suggestions': self._suggest_media(topic, content_type, platform)
        }
    
    def _generate_keywords(self, topic, content_type):
        """生成相关关键词"""
        # 简化的关键词生成逻辑
        base_keywords = topic.split()
        
        type_keywords = {
            '社交媒体': ['分享', '互动', '话题', '讨论'],
            '产品介绍': ['功能', '特性', '优势', '应用'],
            '技术文章': ['方法', '实现', '原理', '最佳实践'],
            '营销文案': ['优惠', '限时', '专业', '信赖'],
            '教程指南': ['教程', '步骤', '方法', '技巧']
        }
        
        extended_keywords = base_keywords + type_keywords.get(content_type, [])
        return extended_keywords[:8]
    
    def _suggest_media(self, topic, content_type, platform):
        """建议媒体素材"""
        suggestions = []
        
        if platform in ['微信公众号', '知乎', '官网']:
            suggestions.extend(['配图', '图表', 'GIF动图'])
        elif platform == '小红书':
            suggestions.extend(['精美配图', '拼图', '产品图'])
        elif platform in ['抖音', 'B站']:
            suggestions.extend(['短视频', '动画', '字幕'])
        
        if content_type == '产品介绍':
            suggestions.extend(['产品截图', '对比图', '使用场景图'])
        elif content_type == '技术文章':
            suggestions.extend(['代码截图', '架构图', '流程图'])
        
        return list(set(suggestions))
    
    def _format_plan(self, plan):
        """格式化计划输出"""
        result = f"""📋 内容创作计划

🎯 基本信息：
• 主题：{plan['topic']}
• 类型：{plan['content_type']}
• 受众：{plan['audience']}
• 平台：{plan['platform']}

📊 平台规格：
• 建议字数：{plan['specs']['word_limit']}字以内
• 内容风格：{plan['specs']['style']}
• 媒体形式：{plan['specs']['media']}

📝 内容结构：
"""
        for i, section in enumerate(plan['structure'], 1):
            result += f"{i}. {section}\n"
        
        result += f"""
🏷️ 关键词建议：
{', '.join(plan['keywords'])}

🎨 媒体素材建议：
• {' • '.join(plan['media_suggestions'])}

💡 创作提示：
根据{plan['platform']}的特点，建议采用{plan['specs']['style']}的风格，
重点突出{plan['topic']}的核心价值点，确保内容对{plan['audience']}有吸引力。
"""
        return result

@register_tool('copywriter')
class CopywriterTool(BaseTool):
    """文案写作工具"""
    description = '根据需求生成各种类型的营销文案'
    parameters = {
        'type': 'object',
        'properties': {
            'product_info': {
                'type': 'string',
                'description': '产品信息描述'
            },
            'copy_type': {
                'type': 'string',
                'enum': ['标题', '简介', '详细描述', '广告文案', '推广语'],
                'description': '文案类型'
            },
            'tone': {
                'type': 'string',
                'enum': ['专业', '亲切', '激情', '理性', '时尚'],
                'description': '文案语调'
            },
            'length': {
                'type': 'string',
                'enum': ['简短', '中等', '详细'],
                'description': '文案长度'
            }
        },
        'required': ['product_info', 'copy_type']
    }
    
    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = self._verify_json_format_args(params)
            product_info = params_dict['product_info']
            copy_type = params_dict['copy_type']
            tone = params_dict.get('tone', '专业')
            length = params_dict.get('length', '中等')
            
            copy_text = self._generate_copy(product_info, copy_type, tone, length)
            return copy_text
            
        except Exception as e:
            return f"文案生成错误：{str(e)}"
    
    def _generate_copy(self, product_info, copy_type, tone, length):
        """生成文案"""
        
        # 根据文案类型生成模板
        if copy_type == '标题':
            return self._generate_title(product_info, tone)
        elif copy_type == '简介':
            return self._generate_intro(product_info, tone, length)
        elif copy_type == '详细描述':
            return self._generate_description(product_info, tone, length)
        elif copy_type == '广告文案':
            return self._generate_ad_copy(product_info, tone)
        elif copy_type == '推广语':
            return self._generate_slogan(product_info, tone)
    
    def _generate_title(self, product_info, tone):
        """生成标题"""
        tone_templates = {
            '专业': [
                "{}：专业级解决方案",
                "{}，行业领先技术",
                "专业{}，值得信赖"
            ],
            '亲切': [
                "{}，让生活更美好",
                "用心打造的{}",
                "{}，温暖每一天"
            ],
            '激情': [
                "震撼来袭！{}",
                "{}，点燃你的激情！",
                "不可错过的{}"
            ]
        }
        
        templates = tone_templates.get(tone, tone_templates['专业'])
        # 简化处理，实际应用中会更智能
        product_name = product_info.split('：')[0] if '：' in product_info else product_info[:10]
        
        return f"📝 标题建议：\n" + "\n".join([template.format(product_name) for template in templates])
    
    def _generate_intro(self, product_info, tone, length):
        """生成简介"""
        length_limits = {'简短': 50, '中等': 100, '详细': 200}
        limit = length_limits[length]
        
        intro = f"基于{product_info}的核心功能和特点，"
        
        if tone == '专业':
            intro += "采用先进技术，为用户提供可靠的解决方案。"
        elif tone == '亲切':
            intro += "用心设计每一个细节，让您的体验更加舒适。"
        elif tone == '激情':
            intro += "突破传统界限，为您带来前所未有的使用体验！"
        
        if len(intro) > limit:
            intro = intro[:limit] + "..."
        
        return f"📝 产品简介：\n{intro}"
    
    def _generate_ad_copy(self, product_info, tone):
        """生成广告文案"""
        return f"""📝 广告文案：

【标题】{product_info}的核心卖点
【正文】
{product_info}，{tone}的选择。
立即体验，感受不一样的品质！

【行动召唤】
限时优惠，立即购买！
"""

def create_content_creator():
    """创建内容创作助手"""
    
    llm_cfg = {
        'model': 'qwen3-235b-a22b', 
        'model_type': 'qwen_dashscope',
        'generate_cfg': {
            'top_p': 0.9,  # 创作类任务使用较高的创造性
            'temperature': 0.8,
            'max_input_tokens': 8000
        }
    }
    
    system_message = '''你是一个专业的多模态内容创作助手，擅长策划和创作各种类型的内容。

🎨 创作能力：
1. 内容策划和结构设计
2. 多平台适配的文案创作
3. 图文并茂的内容制作
4. 营销文案和推广语创作
5. 多媒体内容脚本编写

🛠️ 专业工具：
- content_planner：制定详细的内容创作计划
- copywriter：生成各类营销文案和推广文字
- image_gen：生成配套的视觉素材

🎯 创作原则：
- 内容有价值，能解决用户问题
- 语言生动有趣，符合目标受众
- 结构清晰，逻辑性强
- 视觉效果佳，图文搭配合理
- 符合平台特性和传播规律

📊 服务流程：
1. 理解创作需求和目标
2. 制定内容策划方案
3. 创作优质的文字内容
4. 建议配套的视觉素材
5. 提供发布和推广建议

让我们一起创作出色的内容吧！'''
    
    tools = [
        'content_planner',
        'copywriter',
        'image_gen'
    ]
    
    return Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=system_message,
        name='内容创作助手',
        description='专业的多模态内容创作和营销文案助手'
    )

# 使用演示
def demo_content_creator():
    """内容创作助手演示"""
    creator = create_content_creator()
    
    query = """我需要为我们公司的新产品"智能家居控制系统"制作一篇微信公众号文章，
    目标受众是25-40岁的科技爱好者和家庭用户。请帮我制定创作计划并生成文案。"""
    
    messages = [{'role': 'user', 'content': query}]
    
    print("🎨 内容创作助手回复：")
    for response in creator.run(messages):
        if response:
            print(response[-1].get('content', ''))
```

## 🔧 自定义Agent开发指南

### 开发步骤详解

#### 1. 继承合适的基类

```python
from qwen_agent import Agent
from qwen_agent.agents import Assistant, FnCallAgent

# 选择1：继承基础Agent类（最大自由度）
class CustomAgent(Agent):
    def _run(self, messages, **kwargs):
        # 实现自定义逻辑
        pass

# 选择2：继承FnCallAgent（支持工具调用）
class CustomFnCallAgent(FnCallAgent):
    def _run(self, messages, **kwargs):
        # 在工具调用基础上添加自定义逻辑
        pass

# 选择3：继承Assistant（支持RAG+工具调用）
class CustomAssistant(Assistant):
    def _run(self, messages, **kwargs):
        # 在RAG+工具调用基础上添加自定义逻辑
        pass
```

#### 2. 自定义消息处理逻辑

```python
class SpecializedAgent(FnCallAgent):
    """专门化的Agent示例"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversation_history = []
        self.user_preferences = {}
    
    def _run(self, messages, **kwargs):
        """自定义消息处理逻辑"""
        
        # 1. 预处理：提取用户偏好
        self._extract_user_preferences(messages)
        
        # 2. 上下文增强：添加历史对话信息
        enhanced_messages = self._enhance_with_context(messages)
        
        # 3. 调用父类处理逻辑
        response_generator = super()._run(enhanced_messages, **kwargs)
        
        # 4. 后处理：优化响应内容
        for responses in response_generator:
            processed_responses = self._post_process_responses(responses)
            self.conversation_history.extend(processed_responses)
            yield processed_responses
    
    def _extract_user_preferences(self, messages):
        """提取用户偏好信息"""
        latest_user_message = None
        for msg in reversed(messages):
            if msg.role == 'user':
                latest_user_message = msg
                break
        
        if latest_user_message:
            content = latest_user_message.get_text_content()
            # 简单的偏好提取逻辑
            if '我喜欢' in content:
                preferences = content.split('我喜欢')[1].split('，')[0]
                self.user_preferences['likes'] = preferences
    
    def _enhance_with_context(self, messages):
        """使用历史对话增强上下文"""
        if self.conversation_history:
            # 添加最近的对话历史
            recent_history = self.conversation_history[-3:]  # 最近3轮对话
            context_summary = self._summarize_history(recent_history)
            
            # 将历史摘要添加到系统消息中
            if messages and messages[0].role == 'system':
                messages[0].content += f"\n\n历史对话要点：{context_summary}"
            
        return messages
    
    def _summarize_history(self, history):
        """总结历史对话"""
        # 简化的历史总结逻辑
        topics = []
        for msg in history:
            if msg.role == 'user':
                content = msg.get_text_content()[:50]
                topics.append(content)
        return "，".join(topics)
    
    def _post_process_responses(self, responses):
        """后处理响应"""
        processed = []
        for response in responses:
            # 添加个性化元素
            if response.role == 'assistant' and self.user_preferences.get('likes'):
                content = response.content
                if isinstance(content, str):
                    content += f"\n\n💡 基于您喜欢{self.user_preferences['likes']}，我还推荐..."
                    response.content = content
            processed.append(response)
        return processed
```

#### 3. 集成自定义工具

```python
@register_tool('personality_analyzer')
class PersonalityAnalyzerTool(BaseTool):
    """个性分析工具"""
    description = '分析用户对话中体现的个性特征'
    parameters = {
        'type': 'object',
        'properties': {
            'conversation_text': {
                'type': 'string',
                'description': '对话文本内容'
            }
        },
        'required': ['conversation_text']
    }
    
    def call(self, params: str, **kwargs) -> str:
        params_dict = self._verify_json_format_args(params)
        text = params_dict['conversation_text']
        
        # 简单的个性分析逻辑
        personality_traits = self._analyze_personality(text)
        return f"个性分析结果：{personality_traits}"
    
    def _analyze_personality(self, text: str) -> str:
        """分析个性特征"""
        traits = []
        
        if any(word in text for word in ['详细', '具体', '准确']):
            traits.append('注重细节')
        if any(word in text for word in ['快速', '简单', '直接']):
            traits.append('追求效率')
        if any(word in text for word in ['创新', '新颖', '不同']):
            traits.append('富有创造力')
        
        return '、'.join(traits) if traits else '个性特征不明显'

class PersonalizedAgent(FnCallAgent):
    """个性化Agent"""
    
    def __init__(self, **kwargs):
        # 添加个性分析工具
        if 'function_list' not in kwargs:
            kwargs['function_list'] = []
        kwargs['function_list'].append('personality_analyzer')
        
        super().__init__(**kwargs)
        self.personality_profile = {}
    
    def _run(self, messages, **kwargs):
        """加入个性化处理"""
        
        # 定期更新个性档案
        if len(messages) % 5 == 0:  # 每5轮对话分析一次
            self._update_personality_profile(messages)
        
        # 根据个性档案调整系统消息
        if self.personality_profile:
            messages = self._personalize_system_message(messages)
        
        return super()._run(messages, **kwargs)
    
    def _update_personality_profile(self, messages):
        """更新个性档案"""
        conversation_text = "\n".join([
            msg.get_text_content() for msg in messages[-10:] 
            if msg.role == 'user'
        ])
        
        # 调用个性分析工具
        if conversation_text:
            analysis_result = self._call_tool(
                'personality_analyzer',
                {'conversation_text': conversation_text}
            )
            self.personality_profile['last_analysis'] = analysis_result
```

## 📈 性能优化最佳实践

### 1. LLM调用优化

```python
# ✅ 推荐：合理设置生成配置
llm_cfg = {
    'model': 'qwen3-235b-a22b',
    'generate_cfg': {
        'max_input_tokens': 6000,  # 控制输入长度
        'max_retries': 3,          # 合理的重试次数
        'top_p': 0.8,              # 平衡创造性和一致性
        'temperature': 0.3,        # 对话任务使用较低温度
    }
}

# ❌ 避免：过长的输入导致性能问题
# 不要将整个文档直接作为输入，应该先进行摘要或分段处理
```

### 2. 工具调用优化

```python
class OptimizedTool(BaseTool):
    """优化的工具示例"""
    
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.cache = {}  # 添加缓存机制
        self.batch_size = cfg.get('batch_size', 10) if cfg else 10
    
    def call(self, params: str, **kwargs) -> str:
        """优化的工具调用"""
        
        # 1. 参数缓存检查
        cache_key = self._generate_cache_key(params)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 2. 批量处理
        params_dict = self._verify_json_format_args(params)
        if 'batch_data' in params_dict:
            return self._batch_process(params_dict['batch_data'])
        
        # 3. 正常处理
        result = self._process_single(params_dict)
        
        # 4. 缓存结果
        self.cache[cache_key] = result
        return result
    
    def _batch_process(self, batch_data):
        """批量处理提升效率"""
        results = []
        for i in range(0, len(batch_data), self.batch_size):
            batch = batch_data[i:i+self.batch_size]
            batch_result = self._process_batch(batch)
            results.extend(batch_result)
        return results
```

### 3. 内存管理优化

```python
class MemoryEfficientAgent(FnCallAgent):
    """内存高效的Agent"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_history_length = kwargs.get('max_history_length', 50)
    
    def _run(self, messages, **kwargs):
        """优化内存使用"""
        
        # 1. 历史消息截断
        if len(messages) > self.max_history_length:
            # 保留系统消息和最近的对话
            system_messages = [m for m in messages if m.role == 'system']
            recent_messages = messages[-self.max_history_length:]
            messages = system_messages + recent_messages
        
        # 2. 大文件处理优化
        messages = self._optimize_large_content(messages)
        
        # 3. 调用父类处理
        for responses in super()._run(messages, **kwargs):
            yield responses
    
    def _optimize_large_content(self, messages):
        """优化大内容处理"""
        optimized = []
        for msg in messages:
            if isinstance(msg.content, str) and len(msg.content) > 10000:
                # 大内容摘要处理
                summary = msg.content[:1000] + "\n...(内容过长，已摘要)...\n" + msg.content[-1000:]
                msg.content = summary
            optimized.append(msg)
        return optimized
```

## 🚨 常见问题与解决方案

### 问题1：工具调用失败

```python
# 问题：工具调用参数格式错误
# 解决方案：增强参数验证

class RobustTool(BaseTool):
    def call(self, params: str, **kwargs) -> str:
        try:
            # 1. 多种格式兼容
            if isinstance(params, dict):
                params_dict = params
            else:
                params_dict = self._verify_json_format_args(params, strict_json=False)
            
            # 2. 参数默认值处理
            required_params = ['param1', 'param2']
            for param in required_params:
                if param not in params_dict:
                    return f"缺少必需参数：{param}"
            
            # 3. 参数类型转换
            params_dict = self._convert_param_types(params_dict)
            
            return self._execute_tool(params_dict)
            
        except Exception as e:
            return f"工具执行错误：{str(e)}\n请检查参数格式是否正确"
    
    def _convert_param_types(self, params_dict):
        """参数类型转换"""
        converted = {}
        for key, value in params_dict.items():
            # 尝试转换常见类型
            if isinstance(value, str) and value.isdigit():
                converted[key] = int(value)
            elif isinstance(value, str) and value.replace('.', '').isdigit():
                converted[key] = float(value)
            else:
                converted[key] = value
        return converted
```

### 问题2：响应延迟过高

```python
# 解决方案：流式处理优化

class FastResponseAgent(FnCallAgent):
    def _run(self, messages, **kwargs):
        """快速响应优化"""
        
        # 1. 预处理加速
        messages = self._quick_preprocess(messages)
        
        # 2. 并发工具调用（概念性，实际需要异步支持）
        for responses in super()._run(messages, **kwargs):
            # 3. 实时流式输出
            if responses:
                # 立即输出部分结果
                partial_response = self._extract_partial_response(responses)
                if partial_response:
                    yield partial_response
            
            yield responses
    
    def _quick_preprocess(self, messages):
        """快速预处理"""
        # 简化预处理逻辑，减少延迟
        return messages
    
    def _extract_partial_response(self, responses):
        """提取部分响应用于实时显示"""
        for response in responses:
            if response.role == 'assistant' and response.content:
                text = response.get_text_content()
                if len(text) > 50:  # 有足够内容时提前输出
                    return [response]
        return []
```

### 问题3：多轮对话上下文丢失

```python
# 解决方案：上下文管理优化

class ContextAwareAgent(Assistant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context_manager = ContextManager()
    
    def _run(self, messages, **kwargs):
        """上下文感知处理"""
        
        # 1. 上下文恢复
        enhanced_messages = self.context_manager.enhance_context(messages)
        
        # 2. 处理消息
        for responses in super()._run(enhanced_messages, **kwargs):
            # 3. 上下文更新
            self.context_manager.update_context(responses)
            yield responses

class ContextManager:
    """上下文管理器"""
    
    def __init__(self, max_context_length=20):
        self.conversation_history = []
        self.key_information = {}
        self.max_length = max_context_length
    
    def enhance_context(self, messages):
        """增强上下文信息"""
        # 添加关键信息到系统消息
        if self.key_information:
            context_info = self._format_key_information()
            # 将上下文信息添加到消息中
            if messages and messages[0].role == 'system':
                messages[0].content += f"\n\n上下文信息：{context_info}"
        
        return messages
    
    def update_context(self, responses):
        """更新上下文"""
        for response in responses:
            # 提取关键信息
            if response.role == 'assistant':
                key_info = self._extract_key_information(response)
                self.key_information.update(key_info)
        
        # 维护历史长度
        self.conversation_history.extend(responses)
        if len(self.conversation_history) > self.max_length:
            self.conversation_history = self.conversation_history[-self.max_length:]
    
    def _extract_key_information(self, message):
        """提取关键信息"""
        # 简化的关键信息提取
        key_info = {}
        content = message.get_text_content()
        
        # 提取数字信息
        numbers = re.findall(r'\d+', content)
        if numbers:
            key_info['numbers'] = numbers[:3]
        
        # 提取重要概念
        important_words = ['重要', '关键', '核心', '主要']
        for word in important_words:
            if word in content:
                key_info['importance'] = word
                break
        
        return key_info
    
    def _format_key_information(self):
        """格式化关键信息"""
        info_parts = []
        for key, value in self.key_information.items():
            if isinstance(value, list):
                info_parts.append(f"{key}: {', '.join(map(str, value))}")
            else:
                info_parts.append(f"{key}: {value}")
        
        return "；".join(info_parts)
```

## 🎯 实战总结与建议

### 开发建议

1. **从简单开始**：先使用内置Agent，熟悉框架后再自定义
2. **工具优先**：优先开发好用的工具，Agent的能力很大程度上取决于工具质量
3. **流式处理**：始终考虑用户体验，使用流式输出提升响应性
4. **错误处理**：完善的异常处理是生产环境的必需品
5. **性能监控**：添加必要的日志和监控，便于问题排查

### 最佳实践

1. **配置分离**：将LLM配置、工具配置等分离，便于管理
2. **模块化设计**：工具和Agent都应该模块化，易于复用和维护
3. **测试驱动**：为关键功能编写测试用例
4. **文档完善**：为自定义工具和Agent编写详细文档
5. **版本控制**：使用Git等工具管理代码版本

### 性能调优

1. **缓存策略**：合理使用缓存减少重复计算
2. **批量处理**：对于大量数据，使用批量处理提升效率
3. **资源管理**：及时释放不必要的资源
4. **并发优化**：在支持的场景下使用并发处理
5. **监控告警**：设置性能监控和告警机制

通过这些实战案例和最佳实践，开发者可以更好地理解和使用Qwen-Agent框架，构建出高质量的AI应用。

---

*本实战案例文档提供了从基础到高级的完整开发指南，帮助开发者快速掌握Qwen-Agent框架的使用技巧和最佳实践。*
