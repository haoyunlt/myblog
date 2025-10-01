---
title: "LiveKit Agents 框架使用示例"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['WebRTC', 'LiveKit', '语音处理', '实时通信']
categories: ['AI语音助手']
description: "LiveKit Agents 框架使用示例的深入技术分析文档"
keywords: ['WebRTC', 'LiveKit', '语音处理', '实时通信']
author: "技术分析师"
weight: 1
---

## 1. 基础语音代理示例

### 1.1 简单语音助手

```python
import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero

load_dotenv()
logger = logging.getLogger("basic-agent")

class MyAgent(Agent):
    """
    基础语音代理类
    
    功能说明：
    - 继承自Agent基类，提供语音交互能力
    - 支持自定义指令和工具函数
    - 自动处理用户输入和生成响应
    """
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "你的名字是Kelly。你通过语音与用户交互。"
                "请保持回复简洁明了。"
                "不要使用表情符号、星号、markdown或其他特殊字符。"
                "你很好奇、友好，有幽默感。"
                "你会用英语与用户交流"
            ),
        )

    async def on_enter(self):
        """
        代理进入会话时的回调函数
        
        功能说明：
        - 当代理被添加到会话时自动调用
        - 根据代理指令生成初始回复
        - 启动与用户的对话流程
        """
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, 
        context: RunContext, 
        location: str, 
        latitude: str, 
        longitude: str
    ):
        """
        天气查询工具函数
        
        参数说明：
        - context: 运行时上下文，包含会话状态和用户数据
        - location: 用户询问的地点名称
        - latitude: 地点的纬度坐标（由LLM自动估算）
        - longitude: 地点的经度坐标（由LLM自动估算）
        
        返回值：
        - str: 天气信息描述
        
        使用场景：
        当用户询问天气相关信息时被LLM自动调用
        """
        logger.info(f"查询{location}的天气信息")
        
        # 这里可以调用真实的天气API
        # 示例返回固定值
        return "今天天气晴朗，温度70华氏度。"

async def entrypoint(ctx: JobContext):
    """
    应用程序入口点函数
    
    参数说明：
    - ctx: 任务上下文，包含房间连接和配置信息
    
    功能说明：
    - 配置并启动代理会话
    - 设置语音组件（STT、LLM、TTS、VAD）
    - 建立与LiveKit房间的连接
    """
    # 设置日志上下文字段
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # 创建代理会话，配置各个组件
    session = AgentSession(
        vad=silero.VAD.load(),                    # 语音活动检测
        llm=openai.LLM(model="gpt-4o-mini"),     # 大语言模型
        stt=deepgram.STT(model="nova-3"),        # 语音转文本
        tts=elevenlabs.TTS(),                    # 文本转语音
        preemptive_generation=True,               # 预先生成响应
        resume_false_interruption=True,           # 恢复错误中断
    )

    # 启动会话
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
    )

if __name__ == "__main__":
    # 运行应用程序
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### 1.2 运行方式

```bash
# 开发模式 - 支持热重载
python basic_agent.py dev

# 控制台测试模式 - 终端音频测试
python basic_agent.py console

# 生产模式 - 稳定运行
python basic_agent.py start
```

## 2. 动态工具创建示例

### 2.1 三种工具创建方式

```python
import logging
import random
from enum import Enum
from typing import Literal
from pydantic import BaseModel
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    FunctionTool,
    JobContext,
    ModelSettings,
    function_tool,
)

class DynamicToolAgent(Agent):
    """
    动态工具创建代理
    
    功能说明：
    - 演示三种不同的工具创建方式
    - 支持运行时动态添加和修改工具
    - 展示工具的灵活使用模式
    """
    
    def __init__(self, instructions: str, tools: list[FunctionTool]) -> None:
        super().__init__(instructions=instructions, tools=tools)

    async def llm_node(
        self, 
        chat_ctx: ChatContext, 
        tools: list[FunctionTool], 
        model_settings: ModelSettings
    ):
        """
        自定义LLM节点处理函数
        
        参数说明：
        - chat_ctx: 聊天上下文，包含对话历史
        - tools: 当前可用的工具列表
        - model_settings: 模型设置参数
        
        功能说明：
        - 方式3：为本次调用临时添加工具
        - 在不修改代理配置的情况下动态扩展功能
        """
        
        # 方式3: 为本次LLM调用添加临时工具
        async def _get_weather(location: str) -> str:
            """临时天气查询工具"""
            return f"{location}的天气是晴天。"

        # 将临时工具添加到工具列表
        tools.append(
            function_tool(
                _get_weather,
                name="get_weather",
                description="获取指定地点的天气信息",
            )
        )

        # 调用默认的LLM节点处理
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

async def _get_course_list_from_db() -> list[str]:
    """
    模拟数据库查询函数
    
    返回值：
    - list[str]: 课程列表
    
    功能说明：
    在实际应用中，这里应该连接真实的数据库
    """
    return [
        "应用数学",
        "数据科学", 
        "机器学习",
        "深度学习",
        "语音代理",
    ]

async def entrypoint(ctx: JobContext):
    """动态工具创建示例入口点"""
    
    # 方式1: 代理创建时定义工具
    courses = await _get_course_list_from_db()
    
    # 动态创建枚举类型（LLM会自动识别）
    CourseType = Enum("CourseType", {c.replace(" ", "_"): c for c in courses})
    
    # 使用Pydantic模型定义复杂参数类型
    class CourseInfo(BaseModel):
        """
        课程信息模型
        
        属性说明：
        - course: 课程类型，从动态枚举中选择
        - location: 上课方式，在线或面授
        """
        course: CourseType  # type: ignore
        location: Literal["online", "in-person"]
    
    async def _get_course_info(info: CourseInfo) -> str:
        """
        获取课程信息工具函数
        
        参数说明：
        - info: 课程信息对象，包含课程类型和地点
        
        返回值：
        - str: 课程详细信息
        """
        logger.info(f"get_course_info调用: {info}")
        return f"想象一下关于{info.course}的课程。"
    
    # 创建代理，初始化时包含基础工具
    agent = DynamicToolAgent(
        instructions="你是一个有用的助手，可以回答问题并帮助完成任务。",
        tools=[
            function_tool(
                _get_course_info,
                name="get_course_info", 
                description="获取课程信息",
            )
        ],
    )
    
    # 方式2: 代理创建后使用update_tools()更新工具
    async def _random_number() -> int:
        """
        随机数生成工具
        
        返回值：
        - int: 0-100之间的随机整数
        """
        num = random.randint(0, 100)
        logger.info(f"random_number调用: {num}")
        return num
    
    # 更新代理工具集，添加新工具
    await agent.update_tools(
        agent.tools + [
            function_tool(
                _random_number, 
                name="random_number", 
                description="生成随机数"
            )
        ]
    )
    
    # 创建并启动会话
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(use_realtime=True),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
    )
    await session.start(agent, room=ctx.room)
```

## 3. 实时API集成示例

### 3.1 OpenAI Realtime API

```python
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import openai

class RealtimeAgent(Agent):
    """
    实时API代理
    
    功能说明：
    - 使用OpenAI的实时API进行低延迟对话
    - 支持流式音频处理
    - 提供更自然的对话体验
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="你是一个友好的语音助手。",
            # 使用实时模型替代传统的LLM + TTS组合
            llm=openai.realtime.RealtimeModel(
                voice="coral",           # 语音类型
                temperature=0.8,         # 创造性参数
                max_response_output_tokens=4096,  # 最大输出长度
            ),
        )
    
    async def on_enter(self):
        """代理进入时生成欢迎消息"""
        await self.session.generate_reply(
            instructions="向用户问好并提供帮助。"
        )

async def entrypoint(ctx: JobContext):
    """实时API示例入口点"""
    
    # 使用实时API时，不需要单独配置STT和TTS
    session = AgentSession(
        # 实时模型已包含语音处理能力
        llm=openai.realtime.RealtimeModel(voice="coral")
    )
    
    await session.start(
        agent=RealtimeAgent(),
        room=ctx.room,
    )
```

## 4. 多代理切换示例

### 4.1 代理切换机制

```python
from livekit.agents import Agent, AgentSession, JobContext, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero, deepgram

class IntroAgent(Agent):
    """
    介绍代理 - 收集用户信息
    
    功能说明：
    - 作为对话的入口点
    - 收集用户的基本信息
    - 完成后切换到其他专门代理
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "你是一个故事讲述者。你的目标是收集一些用户信息"
                "来让故事个性化和引人入胜。询问用户的姓名和来自哪里。"
            )
        )

    async def on_enter(self):
        """进入时开始收集信息"""
        await self.session.generate_reply(
            instructions="问候用户并收集信息"
        )

    @function_tool
    async def information_gathered(
        self,
        context: RunContext,
        name: str,
        location: str,
    ):
        """
        信息收集完成工具
        
        参数说明：
        - context: 运行时上下文
        - name: 用户姓名
        - location: 用户所在地
        
        返回值：
        - tuple: (下一个代理, 切换消息)
        
        功能说明：
        当收集到足够信息时，切换到故事代理
        """
        # 保存用户数据到上下文
        context.userdata.name = name
        context.userdata.location = location
        
        # 创建并返回下一个代理
        story_agent = StoryAgent(name, location)
        return story_agent, "让我们开始故事吧！"

class StoryAgent(Agent):
    """
    故事代理 - 讲述个性化故事
    
    功能说明：
    - 使用收集的用户信息
    - 讲述个性化故事
    - 可以切换到实时API获得更好的体验
    """
    
    def __init__(self, name: str, location: str) -> None:
        super().__init__(
            instructions=(
                f"你是一个故事讲述者。使用用户信息让故事个性化。"
                f"用户名字是{name}，来自{location}。"
            ),
            # 切换到实时模型获得更好的语音体验
            llm=openai.realtime.RealtimeModel(voice="echo"),
        )

    async def on_enter(self):
        """进入时开始讲故事"""
        await self.session.generate_reply()

@dataclass 
class StoryData:
    """用户故事数据结构"""
    name: str | None = None
    location: str | None = None

async def entrypoint(ctx: JobContext):
    """多代理切换示例入口点"""
    
    # 创建用户数据对象
    userdata = StoryData()
    
    # 创建支持用户数据的会话
    session = AgentSession[StoryData](
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="echo"),
        userdata=userdata,
    )
    
    # 从介绍代理开始
    await session.start(
        agent=IntroAgent(),
        room=ctx.room,
    )
```

## 5. 工具函数高级用法

### 5.1 参数类型注解

```python
from enum import Enum
from typing import Annotated, Literal
from livekit.agents import Agent, function_tool

class RoomName(str, Enum):
    """房间名称枚举"""
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom" 
    KITCHEN = "kitchen"
    GARAGE = "garage"

class AdvancedToolAgent(Agent):
    """高级工具使用代理"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="你是一个智能家居控制助手。"
        )

    @function_tool
    async def control_light(
        self,
        room: RoomName,                              # 枚举类型参数
        action: Literal["on", "off", "dim"],         # 字面量类型
        brightness: Annotated[int, "亮度 (0-100)"] = 100,  # 带注解的参数
    ):
        """
        智能灯光控制工具
        
        参数说明：
        - room: 房间名称，从预定义枚举中选择
        - action: 操作类型，开启/关闭/调暗
        - brightness: 亮度级别，0-100之间的整数
        
        返回值：
        - str: 操作结果描述
        """
        if action == "on":
            return f"{room.value}的灯已开启，亮度{brightness}%"
        elif action == "off":
            return f"{room.value}的灯已关闭"
        else:  # dim
            return f"{room.value}的灯已调暗到{brightness}%"

    @function_tool
    async def get_number_info(
        self,
        value: Annotated[float, "数值参数，支持小数"]
    ):
        """
        数值信息获取工具
        
        功能说明：
        - 展示如何使用Annotated类型提供参数说明
        - LLM可以理解参数的具体含义和约束
        """
        return f"数值是{value}。"
```

### 5.2 静默工具调用

```python
from livekit.agents import Agent, FunctionToolsExecutedEvent, function_tool

class SilentToolAgent(Agent):
    """
    静默工具调用代理
    
    功能说明：
    - 演示如何执行不生成回复的工具函数
    - 控制何时生成语音响应
    - 适用于后台操作和状态更新
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="你是一个语音代理。当用户要求开灯时调用turn_on_light函数。"
        )
        self.light_on = False

    @function_tool
    async def turn_on_light(self):
        """
        开灯工具 - 静默执行
        
        功能说明：
        - 没有返回值的工具不会自动生成回复
        - 适用于后台状态更新操作
        """
        self.light_on = True
        logger.info("灯已开启")
        # 没有返回值，不会生成语音回复

    @function_tool 
    async def turn_off_light(self):
        """
        关灯工具 - 生成回复
        
        功能说明：
        - 有返回值的工具会自动生成语音回复
        - 用户会听到确认消息
        """
        self.light_on = False
        logger.info("灯已关闭")
        return "灯已关闭"  # 有返回值，会生成语音

    async def on_function_tools_executed(self, event: FunctionToolsExecutedEvent):
        """
        工具执行后的事件处理
        
        参数说明：
        - event: 工具执行事件，包含执行结果
        
        功能说明：
        - 可以在此处决定是否取消自动回复
        - 实现复杂的回复控制逻辑
        """
        # 如果需要，可以取消工具回复
        # event.cancel_tool_reply()
        pass
```

## 6. 环境配置

### 6.1 环境变量设置

```bash
# .env文件内容
# OpenAI配置
OPENAI_API_KEY=your_openai_api_key

# Deepgram配置  
DEEPGRAM_API_KEY=your_deepgram_api_key

# ElevenLabs配置
ELEVEN_API_KEY=your_eleven_api_key

# LiveKit配置
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

### 6.2 依赖安装

```bash
# 安装核心库和常用插件
pip install "livekit-agents[openai,silero,deepgram,elevenlabs,turn-detector]~=1.0"

# 或者分别安装
pip install livekit-agents
pip install livekit-plugins-openai
pip install livekit-plugins-deepgram
pip install livekit-plugins-elevenlabs
pip install livekit-plugins-silero
```

## 7. 调试和监控

### 7.1 日志配置

```python
import logging
from livekit.agents import JobContext, metrics, MetricsCollectedEvent

# 配置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my-agent")

async def entrypoint(ctx: JobContext):
    # 设置日志上下文
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "user_id": "user_123",
    }
    
    # 使用指标收集器
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        """
        指标收集事件处理
        
        功能说明：
        - 实时记录和显示性能指标
        - 收集使用统计信息
        - 支持自定义监控逻辑
        """
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        """记录使用统计"""
        summary = usage_collector.get_summary()
        logger.info(f"使用统计: {summary}")
    
    # 注册关闭回调
    ctx.add_shutdown_callback(log_usage)
```

### 7.2 错误处理

```python
from livekit.agents.llm import ToolError, StopResponse

@function_tool
async def risky_operation(param: str):
    """
    可能出错的操作示例
    
    功能说明：
    - 演示工具函数中的错误处理
    - 使用专门的异常类型
    """
    try:
        # 执行可能失败的操作
        result = perform_operation(param)
        return result
    except ValueError as e:
        # 抛出ToolError，LLM可以看到错误信息
        raise ToolError(f"参数错误: {e}")
    except Exception as e:
        # 严重错误时停止响应
        logger.error(f"操作失败: {e}")
        raise StopResponse()
```

这些示例展示了LiveKit Agents框架的核心用法和高级特性。每个示例都包含详细的注释和说明，帮助开发者理解框架的工作原理和最佳实践。
