---
title: "OpenAI Agents Python SDK 框架使用示例"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['Python', '源码分析']
categories: ['Python']
description: "OpenAI Agents Python SDK 框架使用示例的深入技术分析文档"
keywords: ['Python', '源码分析']
author: "技术分析师"
weight: 1
---

## 2.1 基础使用示例

### 2.1.1 Hello World 示例

最简单的代理使用方式：

```python
from agents import Agent, Runner

# 创建代理实例
agent = Agent(
    name="Assistant",                                    # 代理名称
    instructions="You are a helpful assistant"          # 系统提示词
)

# 同步执行
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# 异步执行
async def main():
    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)

import asyncio
asyncio.run(main())
```

**执行结果分析**：
- `Agent`类是代理的基本定义，包含名称和指令
- `Runner.run_sync`/`Runner.run`是执行代理的入口方法
- `result.final_output`包含代理的最终输出结果

### 2.1.2 函数工具集成示例

展示如何为代理添加自定义函数工具：

```python
import asyncio
from typing import Annotated
from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool

# 定义返回值的数据结构
class Weather(BaseModel):
    city: str = Field(description="城市名称")
    temperature_range: str = Field(description="温度范围（摄氏度）")
    conditions: str = Field(description="天气状况")

# 使用装饰器定义工具函数
@function_tool
def get_weather(
    city: Annotated[str, "要查询天气的城市名称"]
) -> Weather:
    """获取指定城市的当前天气信息。
    
    Args:
        city: 要查询天气的城市名称
        
    Returns:
        Weather: 包含城市、温度范围和天气状况的Weather对象
    """
    print(f"[调试] get_weather 被调用，城市: {city}")
    return Weather(
        city=city, 
        temperature_range="14-20°C", 
        conditions="晴朗有风"
    )

# 创建带工具的代理
agent = Agent(
    name="天气助手",
    instructions="你是一个有用的天气查询助手。",
    tools=[get_weather]  # 添加工具到代理
)

async def main():
    # 执行查询
    result = await Runner.run(agent, input="东京的天气如何？")
    print(result.final_output)
    # 输出: 东京的天气是晴朗有风，温度范围在14-20°C。

if __name__ == "__main__":
    asyncio.run(main())
```

**关键技术点分析**：
1. **`@function_tool`装饰器**：将普通Python函数转换为代理可调用的工具
2. **类型注解**：使用`Annotated`为参数添加描述，帮助LLM理解参数用途
3. **Pydantic模型**：定义结构化的返回值，确保数据一致性
4. **工具集成**：通过`tools`参数将函数工具添加到代理中

### 2.1.3 多代理协作示例（Handoffs）

展示代理间的任务委托和切换：

```python
import asyncio
from agents import Agent, Runner

# 创建西班牙语专家代理
spanish_agent = Agent(
    name="Spanish Expert",
    instructions="你只说西班牙语，回答要简洁。",
    handoff_description="专门处理西班牙语请求的助手。"
)

# 创建英语专家代理
english_agent = Agent(
    name="English Expert", 
    instructions="你只说英语，回答要简洁。",
    handoff_description="专门处理英语请求的助手。"
)

# 创建分流代理（根据语言将请求转发给专门的代理）
triage_agent = Agent(
    name="Language Triage",
    instructions="根据请求的语言，将任务委托给相应的语言专家代理。",
    handoffs=[spanish_agent, english_agent]  # 可切换的目标代理列表
)

async def main():
    # 测试西班牙语请求
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print("西班牙语响应:", result.final_output)
    # 输出: ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?
    
    # 测试英语请求
    result = await Runner.run(triage_agent, input="Hello, how are you?") 
    print("英语响应:", result.final_output)
    # 输出: Hello! I'm doing well, thank you for asking. How are you today?

if __name__ == "__main__":
    asyncio.run(main())
```

**代理切换机制分析**：
1. **`handoffs`参数**：定义可切换的目标代理列表
2. **`handoff_description`**：为每个代理提供描述，帮助分流代理选择合适的目标
3. **自动路由**：框架会自动根据LLM的判断进行代理切换
4. **上下文传递**：切换时会保持对话上下文

## 2.2 高级功能示例

### 2.2.1 生命周期钩子示例

展示如何监控代理执行过程：

```python
import asyncio
import random
from typing import Any, Optional
from pydantic import BaseModel
from agents import (
    Agent, RunContextWrapper, RunHooks, Runner, 
    Tool, Usage, function_tool
)
from agents.items import ModelResponse, TResponseInputItem

class ExampleHooks(RunHooks):
    """自定义生命周期钩子类，用于监控代理执行过程"""
    
    def __init__(self):
        self.event_counter = 0

    def _usage_to_str(self, usage: Usage) -> str:
        """将使用情况转换为可读字符串"""
        return (f"{usage.requests} 个请求, {usage.input_tokens} 输入token, "
                f"{usage.output_tokens} 输出token, {usage.total_tokens} 总token")

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        """代理开始执行时的回调"""
        self.event_counter += 1
        print(f"### {self.event_counter}: 代理 {agent.name} 开始执行. "
              f"使用情况: {self._usage_to_str(context.usage)}")

    async def on_llm_start(
        self,
        context: RunContextWrapper,
        agent: Agent, 
        system_prompt: Optional[str],
        input_items: list[TResponseInputItem],
    ) -> None:
        """LLM开始调用时的回调"""
        self.event_counter += 1
        print(f"### {self.event_counter}: LLM 开始调用. "
              f"使用情况: {self._usage_to_str(context.usage)}")

    async def on_llm_end(
        self, context: RunContextWrapper, agent: Agent, response: ModelResponse
    ) -> None:
        """LLM调用结束时的回调"""
        self.event_counter += 1
        print(f"### {self.event_counter}: LLM 调用结束. "
              f"使用情况: {self._usage_to_str(context.usage)}")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        """工具开始执行时的回调"""
        self.event_counter += 1
        print(f"### {self.event_counter}: 工具 {tool.name} 开始执行. "
              f"参数: {context.tool_arguments}. "  # type: ignore[attr-defined]
              f"使用情况: {self._usage_to_str(context.usage)}")

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        """工具执行结束时的回调"""
        self.event_counter += 1
        print(f"### {self.event_counter}: 工具 {tool.name} 执行完成. "
              f"结果: {result}. "
              f"使用情况: {self._usage_to_str(context.usage)}")

    async def on_handoff(
        self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent
    ) -> None:
        """代理切换时的回调"""
        self.event_counter += 1
        print(f"### {self.event_counter}: 代理切换 从 {from_agent.name} 到 {to_agent.name}. "
              f"使用情况: {self._usage_to_str(context.usage)}")

# 创建钩子实例
hooks = ExampleHooks()

# 定义工具函数
@function_tool
def random_number(max: int) -> int:
    """生成0到max之间的随机数（包含边界）"""
    return random.randint(0, max)

@function_tool  
def multiply_by_two(x: int) -> int:
    """将输入数字乘以2"""
    return x * 2

# 定义输出数据结构
class FinalResult(BaseModel):
    number: int

# 创建乘法代理
multiply_agent = Agent(
    name="乘法代理",
    instructions="将数字乘以2，然后返回最终结果。",
    tools=[multiply_by_two],
    output_type=FinalResult,  # 指定结构化输出类型
)

# 创建启动代理  
start_agent = Agent(
    name="启动代理",
    instructions="生成一个随机数。如果是偶数就停止，如果是奇数就交给乘法代理处理。",
    tools=[random_number],
    output_type=FinalResult,
    handoffs=[multiply_agent],  # 可切换到乘法代理
)

async def main() -> None:
    # 使用钩子执行代理
    result = await Runner.run(
        start_agent,
        hooks=hooks,  # 传入生命周期钩子
        input="生成一个0到250之间的随机数。",
    )
    print(f"最终结果: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
```

**生命周期钩子执行示例输出**：
```
### 1: 代理 启动代理 开始执行. 使用情况: 0 个请求, 0 输入token, 0 输出token, 0 总token
### 2: LLM 开始调用. 使用情况: 0 个请求, 0 输入token, 0 输出token, 0 总token  
### 3: LLM 调用结束. 使用情况: 1 个请求, 143 输入token, 15 输出token, 158 总token
### 4: 工具 random_number 开始执行. 参数: {"max":250}. 使用情况: 1 个请求, 143 输入token, 15 输出token, 158 总token
### 5: 工具 random_number 执行完成. 结果: 107. 使用情况: 1 个请求, 143 输入token, 15 输出token, 158 总token
### 6: LLM 开始调用. 使用情况: 1 个请求, 143 输入token, 15 输出token, 158 总token
### 7: LLM 调用结束. 使用情况: 2 个请求, 310 输入token, 29 输出token, 339 总token
### 8: 代理切换 从 启动代理 到 乘法代理. 使用情况: 2 个请求, 310 输入token, 29 输出token, 339 总token
### 9: 代理 乘法代理 开始执行. 使用情况: 2 个请求, 310 输入token, 29 输出token, 339 总token
### 10: LLM 开始调用. 使用情况: 2 个请求, 310 输入token, 29 输出token, 339 总token
### 11: LLM 调用结束. 使用情况: 3 个请求, 472 输入token, 45 输出token, 517 总token
### 12: 工具 multiply_by_two 开始执行. 参数: {"x":107}. 使用情况: 3 个请求, 472 输入token, 45 输出token, 517 总token
### 13: 工具 multiply_by_two 执行完成. 结果: 214. 使用情况: 3 个请求, 472 输入token, 45 输出token, 517 总token
### 14: LLM 开始调用. 使用情况: 3 个请求, 472 输入token, 45 输出token, 517 总token
### 15: LLM 调用结束. 使用情况: 4 个请求, 660 输入token, 56 输出token, 716 总token
### 16: 代理 乘法代理 执行结束，输出: number=214. 使用情况: 4 个请求, 660 输入token, 56 输出token, 716 总token
最终结果: number=214
```

### 2.2.2 会话管理示例

展示如何使用会话实现对话历史的自动管理：

```python
import asyncio
from agents import Agent, Runner, SQLiteSession

# 创建代理
agent = Agent(
    name="助手",
    instructions="回答要简洁明了。",
)

async def main():
    # 创建会话实例（使用SQLite持久化存储）
    session = SQLiteSession("conversation_123", "conversations.db")
    
    # 第一轮对话
    result = await Runner.run(
        agent,
        "金门大桥在哪个城市？",
        session=session  # 传入会话实例
    )
    print("第一轮:", result.final_output)  # "旧金山"
    
    # 第二轮对话 - 代理自动记住前面的上下文
    result = await Runner.run(
        agent,
        "它在哪个州？",  # 这里的"它"指的是金门大桥
        session=session
    )  
    print("第二轮:", result.final_output)  # "加利福尼亚州"
    
    # 第三轮对话
    result = await Runner.run(
        agent,
        "那个州的人口是多少？",
        session=session
    )
    print("第三轮:", result.final_output)  # "约3900万人"

if __name__ == "__main__":
    asyncio.run(main())
```

**会话管理特点分析**：
1. **自动历史管理**：无需手动处理对话历史，框架自动维护
2. **持久化存储**：支持SQLite、Redis等多种存储方式
3. **会话隔离**：不同session_id的对话完全独立
4. **上下文连续性**：代理能够理解前面对话中的指代关系

### 2.2.3 流式输出示例

展示如何实时获取代理的执行过程和输出：

```python
import asyncio
from agents import Agent, Runner, function_tool

@function_tool
def calculate_fibonacci(n: int) -> int:
    """计算斐波那契数列的第n项"""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

agent = Agent(
    name="数学助手",
    instructions="你是一个数学计算助手，可以进行各种数学运算。",
    tools=[calculate_fibonacci]
)

async def main():
    # 使用流式运行
    streamed_result = Runner.run_streamed(
        agent, 
        input="请计算斐波那契数列的第10项，并解释计算过程。"
    )
    
    # 实时处理流式事件
    async for event in streamed_result.stream_events():
        if hasattr(event, 'data'):
            print(f"事件类型: {type(event).__name__}")
            
            # 处理不同类型的流式事件
            if hasattr(event, 'item') and hasattr(event.item, 'agent'):
                print(f"  代理: {event.item.agent.name}")
            
            if hasattr(event, 'name'):
                print(f"  事件名称: {event.name}")
                
        # 可以在这里实现实时UI更新等功能
    
    # 获取最终完整结果
    final_result = await streamed_result.get_final_result()
    print(f"\n最终结果: {final_result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 2.3 最佳实践示例

### 2.3.1 结构化输出示例

使用Pydantic模型定义结构化输出：

```python
import asyncio
from typing import List
from pydantic import BaseModel, Field
from agents import Agent, Runner

class AnalysisResult(BaseModel):
    """文本分析结果"""
    sentiment: str = Field(description="情感倾向: positive/negative/neutral")
    confidence: float = Field(description="置信度 (0.0-1.0)", ge=0.0, le=1.0)
    key_themes: List[str] = Field(description="关键主题列表")
    summary: str = Field(description="内容摘要", max_length=200)

# 创建文本分析代理
analyzer_agent = Agent(
    name="文本分析师",
    instructions="""
    你是一个专业的文本分析师。请分析给定的文本内容，
    确定其情感倾向、关键主题，并生成简洁的摘要。
    """,
    output_type=AnalysisResult  # 指定结构化输出类型
)

async def main():
    text_to_analyze = """
    今天是美好的一天！阳光明媚，我和朋友们一起去公园踏青。
    我们享受了美味的野餐，拍了很多照片，度过了愉快的时光。
    这样的友谊让我感到非常幸福和感激。
    """
    
    result = await Runner.run(analyzer_agent, f"请分析以下文本：{text_to_analyze}")
    
    # result.final_output 现在是 AnalysisResult 类型
    analysis = result.final_output
    print(f"情感倾向: {analysis.sentiment}")
    print(f"置信度: {analysis.confidence}")
    print(f"关键主题: {', '.join(analysis.key_themes)}")
    print(f"摘要: {analysis.summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.3.2 错误处理和安全防护示例

展示如何使用输入输出防护机制：

```python
import asyncio
from agents import Agent, Runner, input_guardrail, output_guardrail
from agents.guardrail import InputGuardrailResult, OutputGuardrailResult

@input_guardrail
async def content_safety_check(context, input_text: str) -> InputGuardrailResult:
    """输入内容安全检查"""
    prohibited_keywords = ["暴力", "仇恨", "非法"]
    
    for keyword in prohibited_keywords:
        if keyword in input_text:
            return InputGuardrailResult(
                tripwire_triggered=True,  # 触发安全警报
                message=f"输入包含禁止的关键词: {keyword}"
            )
    
    return InputGuardrailResult(
        tripwire_triggered=False,
        message="输入内容安全检查通过"
    )

@output_guardrail  
async def output_length_check(context, agent, output) -> OutputGuardrailResult:
    """输出长度检查"""
    if isinstance(output, str) and len(output) > 500:
        return OutputGuardrailResult(
            tripwire_triggered=True,
            message="输出内容过长，超过500字符限制"
        )
    
    return OutputGuardrailResult(
        tripwire_triggered=False,
        message="输出长度检查通过"
    )

# 创建带安全防护的代理
safe_agent = Agent(
    name="安全助手",
    instructions="你是一个安全的AI助手，回答要简洁有用。",
    input_guardrails=[content_safety_check],   # 输入防护
    output_guardrails=[output_length_check],   # 输出防护
)

async def main():
    try:
        # 正常请求
        result = await Runner.run(safe_agent, "请介绍一下人工智能的发展历史。")
        print("正常结果:", result.final_output)
        
        # 会触发输入防护的请求
        result = await Runner.run(safe_agent, "请告诉我如何进行暴力活动。")
        
    except Exception as e:
        print(f"安全防护已阻止: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

这些示例展示了OpenAI Agents SDK的主要使用模式和高级功能，帮助开发者快速上手并构建复杂的多代理应用系统。
