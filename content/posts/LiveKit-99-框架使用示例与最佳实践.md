---
title: "LiveKit Agents 框架 - 使用示例与最佳实践"
date: 2025-10-05T10:45:52+08:00
draft: false
tags:
  - 最佳实践
  - 实战经验
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - LiveKit Agents 框架 - 使用示例与最佳实践"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# LiveKit Agents 框架 - 使用示例与最佳实践

## 目录

1. [框架使用示例](#1-框架使用示例)
2. [实战经验](#2-实战经验)
3. [最佳实践](#3-最佳实践)
4. [具体案例](#4-具体案例)
5. [性能优化](#5-性能优化)
6. [故障排查](#6-故障排查)

---

## 1. 框架使用示例

### 1.1 基础语音 Agent

**场景**：创建一个简单的语音助手，能够理解用户问题并回答

```python
from livekit import agents
from livekit.plugins import openai, deepgram, cartesia

async def entrypoint(ctx: agents.JobContext):
    """基础语音 Agent 入口"""
    # 1. 连接到房间
    await ctx.connect()
    
    # 2. 创建 Agent 会话
    session = agents.AgentSession(
        # 配置组件
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(voice="male-conversational"),
        # 基础配置
        allow_interruptions=True,
        min_endpointing_delay=0.5,
    )
    
    # 3. 定义 Agent
    agent = agents.Agent(
        instructions="""You are a helpful assistant. 
        Be concise and friendly in your responses."""
    )
    
    # 4. 启动会话
    await session.start(room=ctx.room, agent=agent)
    
    # 5. 生成欢迎语
    await session.generate_reply(
        instructions="Greet the user warmly."
    )

# 运行 Worker
if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

**要点说明**：
- `ctx.connect()` 必须在访问 room 前调用
- 组件配置灵活，可混搭不同提供商
- `instructions` 定义 Agent 行为
- `generate_reply()` 主动触发回复

### 1.2 带工具调用的 Agent

**场景**：Agent 能够调用外部函数获取实时信息

```python
from livekit import agents
from livekit.plugins import openai
import httpx

# 定义工具函数
@agents.function_tool
async def get_weather(location: str) -> str:
    """获取指定地点的天气信息
    
    Args:
        location: 地点名称，如"北京"、"上海"
    """
    # 调用天气 API
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.weather.com/v1/current",
            params={"location": location}
        )
        data = response.json()
    
    return f"{location}当前天气：{data['weather']}, 温度{data['temp']}°C"

@agents.function_tool
async def search_web(query: str) -> str:
    """在网络上搜索信息
    
    Args:
        query: 搜索关键词
    """
    # 调用搜索 API
    return f"关于'{query}'的搜索结果：..."

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    # 创建会话
    session = agents.AgentSession(
        llm=openai.LLM(model="gpt-4o"),
        tts="openai-tts",
    )
    
    # 定义 Agent，注册工具
    class AssistantAgent(agents.Agent):
        def __init__(self):
            super().__init__(
                instructions="""You are a helpful assistant with access to weather and search tools.
                Use tools when needed to provide accurate information.""",
                tools=[get_weather, search_web],
            )
    
    await session.start(room=ctx.room, agent=AssistantAgent())

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

**要点说明**：
- `@agents.function_tool` 自动生成工具定义
- Docstring 作为工具描述，LLM 根据描述决定何时调用
- 工具函数可以是同步或异步
- 工具返回值会自动加入对话上下文

### 1.3 多 Agent 转接场景

**场景**：前台接待 Agent 可以将用户转接到专业 Agent

```python
from livekit import agents
from livekit.plugins import openai

# 前台 Agent
class ReceptionAgent(agents.Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a receptionist. 
            Ask what the user needs and transfer them to the right department."""
        )
        
    @agents.function_tool
    async def transfer_to_sales(self) -> str:
        """转接到销售部门"""
        # 获取当前运行上下文
        ctx = agents.get_current_job_context()
        session = self._activity._session
        
        # 停止当前 Agent
        await session.stop()
        
        # 启动销售 Agent
        await session.start(
            room=ctx.room,
            agent=SalesAgent(),
        )
        
        return "正在为您转接到销售部门..."
    
    @agents.function_tool
    async def transfer_to_support(self) -> str:
        """转接到技术支持"""
        ctx = agents.get_current_job_context()
        session = self._activity._session
        
        await session.stop()
        await session.start(
            room=ctx.room,
            agent=SupportAgent(),
        )
        
        return "正在为您转接到技术支持..."

# 销售 Agent
class SalesAgent(agents.Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a sales agent. 
            Provide product information and pricing."""
        )

# 技术支持 Agent
class SupportAgent(agents.Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a technical support agent. 
            Help users troubleshoot issues."""
        )

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = agents.AgentSession(
        llm=openai.LLM(model="gpt-4o"),
        tts="openai-tts",
    )
    
    # 从前台 Agent 开始
    await session.start(
        room=ctx.room,
        agent=ReceptionAgent(),
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

**要点说明**：
- 通过工具函数实现 Agent 切换
- `session.stop()` + `session.start()` 切换 Agent
- 对话历史自动延续
- 可以实现复杂的多 Agent 协作流程

### 1.4 自定义音频处理

**场景**：在音频输入输出链路中插入自定义处理

```python
from livekit import agents, rtc
from livekit.agents import voice

# 自定义音频输入（添加降噪）
class NoiseReductionInput(voice.io.AudioInput):
    def __init__(self, source: voice.io.AudioInput):
        super().__init__(label="NoiseReduction", source=source)
        self._buffer = []
    
    async def __anext__(self) -> rtc.AudioFrame:
        # 从上游获取音频帧
        frame = await self.source.__anext__()
        
        # 应用降噪算法
        processed_frame = self._apply_noise_reduction(frame)
        
        return processed_frame
    
    def _apply_noise_reduction(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        # 降噪逻辑实现
        # （此处省略具体算法）
        return frame

# 自定义音频输出（添加音效）
class AudioEffectsOutput(voice.io.AudioOutput):
    def __init__(self, next_in_chain: voice.io.AudioOutput | None = None):
        super().__init__(
            label="AudioEffects",
            next_in_chain=next_in_chain,
            capabilities=voice.io.AudioOutputCapabilities(pause=True),
        )
    
    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        # 应用音效
        processed_frame = self._apply_effects(frame)
        
        # 传递给下一层
        if self.next_in_chain:
            await self.next_in_chain.capture_frame(processed_frame)
    
    def flush(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.flush()
    
    def clear_buffer(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.clear_buffer()
    
    def _apply_effects(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        # 音效处理逻辑
        # （此处省略具体实现）
        return frame

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = agents.AgentSession(
        llm="gpt-4o",
        tts="openai-tts",
    )
    
    # 自定义 I/O
    # 需要在 start() 之前设置
    from livekit.agents.voice import room_io
    
    # 创建 RoomIO
    room_io_instance = room_io.RoomIO(
        room=ctx.room,
        agent_session=session,
    )
    
    # 包装输入
    original_input = room_io_instance._audio_input
    session.input.audio = NoiseReductionInput(original_input)
    
    # 包装输出
    original_output = room_io_instance._audio_output
    session.output.audio = AudioEffectsOutput(next_in_chain=original_output)
    
    await session.start(
        room=ctx.room,
        agent=agents.Agent(instructions="You are a helpful assistant.")
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

**要点说明**：
- AudioInput/AudioOutput 支持链式组合
- 自定义处理插入到默认 I/O 链路中
- 必须在 `session.start()` 前设置自定义 I/O
- 可实现降噪、混响、变调等音效

### 1.5 Realtime API 使用

**场景**：使用 OpenAI Realtime API 实现端到端低延迟对话

```python
from livekit import agents
from livekit.plugins import openai

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    # 使用 Realtime Model（无需单独配置 STT/TTS）
    session = agents.AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral",
            temperature=0.8,
            instructions="You are a helpful assistant.",
        ),
    )
    
    # 定义 Agent 并注册工具
    @agents.function_tool
    async def get_time() -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    agent = agents.Agent(
        instructions="You are a helpful assistant with access to time info.",
        tools=[get_time],
    )
    
    await session.start(room=ctx.room, agent=agent)

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

**要点说明**：
- Realtime API 直接处理音频，无需 STT/TTS
- 服务端轮次检测，延迟更低
- 支持工具调用
- 适合对延迟敏感的场景

---

## 2. 实战经验

### 2.1 组件选择

#### STT 选择

| 提供商 | 优势 | 劣势 | 适用场景 |
|--------|------|------|---------|
| **Deepgram** | 低延迟、多语言、高准确率 | 成本较高 | 生产环境、多语言 |
| **AssemblyAI** | 强大的说话人分离 | 延迟稍高 | 多人对话场景 |
| **OpenAI Whisper** | 开源、成本低 | 延迟高、需本地部署 | 离线场景、成本敏感 |
| **Google STT** | 多语言支持好 | 需 GCP 账户 | Google 生态 |

**推荐配置**：

```python
# 生产环境：低延迟优先
stt=deepgram.STT(
    model="nova-3",
    language="multi",  # 自动语言检测
    interim_results=True,
)

# 多人对话：说话人分离
stt=assemblyai.STT(
    model="best",
    diarization=True,
    speakers_expected=2,
)

# 成本优化：降级组合
from livekit.agents import stt as stt_module

stt=stt_module.FallbackAdapter(
    stt=[
        deepgram.STT(),  # 主服务
        openai.STT(),    # 备用服务
    ]
)
```

#### LLM 选择

| 模型 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **GPT-4o** | 智能、多模态 | 成本高 | 复杂推理、生产环境 |
| **GPT-4o-mini** | 性价比高 | 能力稍弱 | 简单对话、成本敏感 |
| **Claude 3.5** | 长上下文、安全 | API 限制 | 文档分析、合规要求 |
| **Gemini** | 免费额度高 | 稳定性 | 开发测试 |

**推荐配置**：

```python
# 生产环境：平衡性能与成本
llm=openai.LLM(
    model="gpt-4o-mini",
    temperature=0.8,
)

# 复杂场景：高智能
llm=openai.LLM(
    model="gpt-4o",
    temperature=0.7,
)

# Realtime API：超低延迟
llm=openai.realtime.RealtimeModel(
    voice="coral",
    temperature=0.8,
)
```

#### TTS 选择

| 提供商 | 优势 | 劣势 | 适用场景 |
|--------|------|------|---------|
| **ElevenLabs** | 音质最佳、情感丰富 | 成本最高、延迟高 | 高端产品 |
| **Cartesia** | 低延迟、流式 | 音色较少 | 实时对话 |
| **OpenAI TTS** | 音质好、支持多语言 | 延迟中等 | 通用场景 |
| **PlayAI** | 性价比高 | 音质一般 | 成本敏感 |

**推荐配置**：

```python
# 实时对话：低延迟优先
tts=cartesia.TTS(
    voice="male-conversational",
    model="sonic-english",
)

# 高端产品：音质优先
tts=elevenlabs.TTS(
    voice="Adam",
    model="eleven_turbo_v2",
)

# 通用场景：平衡
tts=openai.TTS(
    voice="ash",
    model="tts-1-hd",
)
```

### 2.2 预热策略

**问题**：首次请求延迟高，影响用户体验

**解决方案**：使用 `prewarm_fnc` 预加载资源

```python
def prewarm(proc: agents.JobProcess):
    """预热函数，在进程启动时执行一次"""
    # 1. 加载 VAD 模型
    from livekit.plugins import silero
    proc.userdata["vad"] = silero.VAD.load()
    
    # 2. 预热 LLM 连接
    llm_instance = openai.LLM()
    llm_instance.prewarm()
    proc.userdata["llm"] = llm_instance
    
    # 3. 预热 TTS 连接
    tts_instance = cartesia.TTS()
    tts_instance.prewarm()
    proc.userdata["tts"] = tts_instance

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    # 使用预热的组件
    session = agents.AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=ctx.proc.userdata["llm"],
        tts=ctx.proc.userdata["tts"],
    )
    
    await session.start(
        room=ctx.room,
        agent=agents.Agent(instructions="You are a helpful assistant.")
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,  # 注册预热函数
        )
    )
```

**效果**：
- 首次请求延迟降低 50-70%
- 模型常驻内存，后续请求更快
- 进程池复用，避免重复加载

### 2.3 错误处理

**问题**：组件服务不稳定，导致会话中断

**解决方案**：使用 FallbackAdapter + 错误监听

```python
from livekit import agents
from livekit.agents import stt, tts, llm

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    # 1. 配置降级适配器
    stt_adapter = stt.FallbackAdapter(
        stt=[
            deepgram.STT(),      # 主服务
            assemblyai.STT(),    # 备用1
            openai.STT(),        # 备用2
        ]
    )
    
    tts_adapter = tts.FallbackAdapter(
        tts=[
            cartesia.TTS(),
            openai.TTS(),
        ]
    )
    
    llm_adapter = llm.FallbackAdapter(
        llm=[
            openai.LLM(model="gpt-4o"),
            openai.LLM(model="gpt-4o-mini"),
        ]
    )
    
    # 2. 监听可用性变更
    @stt_adapter.on("stt_availability_changed")
    def on_stt_change(event):
        logger.warning(
            f"STT availability changed: {event.available}",
            extra={"stt": stt_adapter.wrapped_stt.label}
        )
    
    # 3. 监听错误事件
    session = agents.AgentSession(
        stt=stt_adapter,
        llm=llm_adapter,
        tts=tts_adapter,
    )
    
    @session.on("error")
    def on_error(event: agents.ErrorEvent):
        logger.error(
            f"Session error: {event.error}",
            exc_info=event.error,
        )
    
    await session.start(
        room=ctx.room,
        agent=agents.Agent(instructions="You are a helpful assistant.")
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

**效果**：
- 主服务故障时自动切换
- 用户无感知
- 提高系统可用性

### 2.4 指标收集

**问题**：无法了解 Agent 使用情况和成本

**解决方案**：监听 metrics_collected 事件

```python
from livekit import agents
from livekit.agents import metrics

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = agents.AgentSession(
        llm="gpt-4o",
        stt="deepgram",
        tts="openai-tts",
    )
    
    # 创建使用量收集器
    usage_collector = metrics.UsageCollector()
    
    # 监听指标事件
    @session.on("metrics_collected")
    def on_metrics(event: agents.MetricsCollectedEvent):
        # 1. 实时记录指标
        metrics.log_metrics(event.metrics)
        
        # 2. 累积使用量
        usage_collector.collect(event.metrics)
        
        # 3. 发送到监控系统
        # send_to_monitoring_system(event.metrics)
    
    # 会话结束时输出总使用量
    async def log_final_usage():
        summary = usage_collector.get_summary()
        logger.info(
            "Session usage summary",
            extra={
                "stt_duration": summary.stt_duration,
                "llm_tokens": summary.llm_tokens,
                "tts_characters": summary.tts_characters,
                "estimated_cost": summary.estimated_cost,
            }
        )
    
    ctx.add_shutdown_callback(log_final_usage)
    
    await session.start(
        room=ctx.room,
        agent=agents.Agent(instructions="You are a helpful assistant.")
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

**收集的指标**：
- **STT**: 识别时长、字符数
- **LLM**: Token 使用量、延迟
- **TTS**: 合成字符数、延迟
- **估算成本**: 基于定价计算

---

## 3. 最佳实践

### 3.1 指令设计

**好的指令**：

```python
instructions="""你是一个专业的客服助手。

**角色定位**：
- 礼貌、耐心、专业
- 主动询问用户需求
- 提供清晰的解决方案

**对话原则**：
1. 简洁回复，每次不超过3句话
2. 使用友好的语气，避免过于正式
3. 遇到不确定的问题，诚实告知用户

**工具使用**：
- 查询订单：使用 check_order 工具
- 查询物流：使用 track_shipping 工具
- 如无法解决，转接人工客服

**禁止事项**：
- 不要提供医疗、法律建议
- 不要泄露系统内部信息
- 不要承诺无法兑现的事情
"""
```

**避免的指令**：

```python
# ❌ 过于简单
instructions="You are a helpful assistant."

# ❌ 过于冗长
instructions="""你是一个助手，你需要...(5000字的详细说明)"""

# ❌ 自相矛盾
instructions="""你必须简洁，但要详细解释所有内容。"""
```

### 3.2 工具函数设计

**好的工具函数**：

```python
@agents.function_tool
async def check_order_status(order_id: str) -> str:
    """查询订单状态
    
    Args:
        order_id: 订单号，格式为 ORD-XXXXXX
    
    Returns:
        订单状态信息，包含：状态、预计送达时间、物流信息
    
    Example:
        order_id="ORD-123456" -> "订单状态：配送中，预计今天18:00送达"
    """
    # 1. 参数验证
    if not order_id.startswith("ORD-"):
        return "订单号格式错误，应为 ORD-XXXXXX"
    
    # 2. 调用后端 API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.example.com/orders/{order_id}"
            )
            data = response.json()
    except Exception as e:
        # 3. 友好的错误提示
        return f"查询订单时出错：{str(e)}，请稍后重试"
    
    # 4. 格式化返回
    status = data['status']
    eta = data['estimated_delivery']
    return f"订单状态：{status}，预计送达：{eta}"
```

**避免的工具函数**：

```python
# ❌ 缺少描述
@agents.function_tool
async def foo(x):
    return x

# ❌ 抛出原始异常
@agents.function_tool
async def bar(order_id: str) -> str:
    # 崩溃会导致会话中断
    data = httpx.get(f"https://api.../orders/{order_id}").json()
    return data['status']

# ❌ 返回过于详细
@agents.function_tool
async def baz(query: str) -> str:
    # 返回整个数据库内容
    return json.dumps(database.query_all(), indent=2)
```

### 3.3 对话历史管理

**问题**：对话历史过长，导致 Token 费用高、延迟大

**解决方案**：限制历史长度或使用摘要

```python
from livekit import agents, llm

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = agents.AgentSession(
        llm="gpt-4o",
        tts="openai-tts",
    )
    
    # 方案1：限制历史长度
    @session.on("conversation_item_added")
    def on_item_added(event: agents.ConversationItemAddedEvent):
        # 保留最近 20 条消息
        MAX_MESSAGES = 20
        if len(session.history.messages) > MAX_MESSAGES:
            # 保留系统消息 + 最近消息
            system_msgs = [
                msg for msg in session.history.messages 
                if msg.role == "system"
            ]
            recent_msgs = session.history.messages[-MAX_MESSAGES:]
            session.history.messages = system_msgs + recent_msgs
    
    # 方案2：定期摘要
    async def summarize_history():
        if len(session.history.messages) > 30:
            # 生成摘要
            summary_prompt = llm.ChatContext.empty()
            summary_prompt.append(
                llm.ChatMessage(
                    role="system",
                    content="Summarize the following conversation:"
                )
            )
            # 添加要摘要的消息
            for msg in session.history.messages[:-10]:
                summary_prompt.append(msg)
            
            # 调用 LLM 生成摘要
            llm_instance = openai.LLM()
            summary_stream = llm_instance.chat(
                chat_ctx=summary_prompt
            )
            summary_text = ""
            async for chunk in summary_stream:
                summary_text += chunk.choices[0].delta.content or ""
            
            # 替换历史消息
            session.history.messages = [
                llm.ChatMessage(role="system", content=f"Previous conversation summary: {summary_text}"),
                *session.history.messages[-10:]  # 保留最近10条
            ]
    
    # 定期调用摘要
    import asyncio
    async def periodic_summary():
        while True:
            await asyncio.sleep(300)  # 每5分钟
            await summarize_history()
    
    asyncio.create_task(periodic_summary())
    
    await session.start(
        room=ctx.room,
        agent=agents.Agent(instructions="You are a helpful assistant.")
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

---

## 4. 具体案例

### 4.1 客服机器人

**需求**：
- 处理常见问题咨询
- 查询订单状态
- 转接人工客服

**实现**：

```python
from livekit import agents
from livekit.plugins import openai, deepgram, cartesia
import httpx

# 订单查询工具
@agents.function_tool
async def check_order(order_id: str) -> str:
    """查询订单状态"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.example.com/orders/{order_id}"
        )
        data = response.json()
    return f"订单{order_id}：{data['status']}，预计{data['eta']}送达"

# 转人工工具
@agents.function_tool
async def transfer_to_human() -> str:
    """转接人工客服"""
    ctx = agents.get_current_job_context()
    # 实际应用中，这里会触发转接逻辑
    # 例如：发送通知给客服系统
    return "正在为您转接人工客服，请稍候..."

class CustomerServiceAgent(agents.Agent):
    def __init__(self):
        super().__init__(
            instructions="""你是一个专业的客服助手。

**职责**：
- 回答产品咨询
- 查询订单状态
- 处理退换货

**原则**：
- 简洁友好
- 主动询问
- 无法解决时转人工

**工具使用**：
- 查订单：check_order
- 转人工：transfer_to_human
""",
            tools=[check_order, transfer_to_human],
        )

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = agents.AgentSession(
        stt=deepgram.STT(model="nova-3", language="zh"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(voice="female-conversational"),
        # 配置打断
        allow_interruptions=True,
        min_interruption_duration=0.3,
    )
    
    await session.start(
        room=ctx.room,
        agent=CustomerServiceAgent(),
    )
    
    # 欢迎语
    await session.generate_reply(
        instructions="用友好的语气欢迎客户，询问需要什么帮助"
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

### 4.2 教育辅导助手

**需求**：
- 解答学科问题
- 提供学习建议
- 推荐学习资源

**实现**：

```python
from livekit import agents
from livekit.plugins import openai

# 知识库搜索
@agents.function_tool
async def search_knowledge(topic: str, subject: str) -> str:
    """在知识库中搜索相关内容
    
    Args:
        topic: 主题，如"二次方程"
        subject: 学科，如"数学"
    """
    # 实际应用中连接向量数据库
    return f"关于{subject}的{topic}：...(知识内容)"

# 练习题推荐
@agents.function_tool
async def recommend_exercises(topic: str, difficulty: str) -> str:
    """推荐练习题
    
    Args:
        topic: 主题
        difficulty: 难度，easy/medium/hard
    """
    return f"推荐{topic}的{difficulty}练习题：..."

class TutorAgent(agents.Agent):
    def __init__(self):
        super().__init__(
            instructions="""你是一个耐心的教育辅导助手。

**教学原则**：
1. 启发式教学，不直接给答案
2. 从简单到复杂，循序渐进
3. 鼓励学生独立思考

**交互方式**：
- 用简单的语言解释概念
- 举生活化的例子
- 提供练习题巩固

**工具使用**：
- 搜索知识：search_knowledge
- 推荐练习：recommend_exercises
""",
            tools=[search_knowledge, recommend_exercises],
        )

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = agents.AgentSession(
        llm=openai.LLM(model="gpt-4o"),  # 高智能模型
        tts="openai-tts",
        # 允许学生打断
        allow_interruptions=True,
    )
    
    await session.start(
        room=ctx.room,
        agent=TutorAgent(),
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

### 4.3 语音控制系统

**需求**：
- 控制智能家居设备
- 查询设备状态
- 设置定时任务

**实现**：

```python
from livekit import agents
from livekit.plugins import openai
import aiohttp

# 设备控制工具
@agents.function_tool
async def control_light(
    room: str,
    action: str,
    brightness: int = 100
) -> str:
    """控制灯光
    
    Args:
        room: 房间名称，如"客厅"、"卧室"
        action: 操作，"on"或"off"
        brightness: 亮度，0-100
    """
    async with aiohttp.ClientSession() as session:
        await session.post(
            "https://smarthome.example.com/api/light",
            json={
                "room": room,
                "action": action,
                "brightness": brightness
            }
        )
    return f"{room}的灯已{action}，亮度{brightness}%"

@agents.function_tool
async def set_temperature(room: str, temperature: float) -> str:
    """设置空调温度
    
    Args:
        room: 房间名称
        temperature: 目标温度（摄氏度）
    """
    async with aiohttp.ClientSession() as session:
        await session.post(
            "https://smarthome.example.com/api/ac",
            json={"room": room, "temperature": temperature}
        )
    return f"{room}的空调已设置为{temperature}°C"

class SmartHomeAgent(agents.Agent):
    def __init__(self):
        super().__init__(
            instructions="""你是一个智能家居助手。

**功能**：
- 控制灯光、空调、窗帘等设备
- 查询设备状态
- 设置自动化场景

**交互**：
- 简短确认
- 主动提醒
- 安全检查

**工具**：
- 灯光：control_light
- 温度：set_temperature
""",
            tools=[control_light, set_temperature],
        )

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = agents.AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        tts="openai-tts",
        # 快速响应
        min_endpointing_delay=0.3,
        max_endpointing_delay=1.5,
    )
    
    await session.start(
        room=ctx.room,
        agent=SmartHomeAgent(),
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

---

## 5. 性能优化

### 5.1 降低延迟

**启用预生成**：

```python
session = agents.AgentSession(
    llm="gpt-4o",
    tts="openai-tts",
    # 在端点检测期间即开始推理
    preemptive_generation=True,
)
```

**效果**：响应延迟降低 30-50%

**调整端点检测**：

```python
session = agents.AgentSession(
    llm="gpt-4o",
    tts="openai-tts",
    # 更快的端点检测
    min_endpointing_delay=0.3,  # 默认0.5
    max_endpointing_delay=2.0,  # 默认6.0
)
```

**使用 Realtime API**：

```python
session = agents.AgentSession(
    llm=openai.realtime.RealtimeModel(),  # 端到端低延迟
)
```

### 5.2 优化成本

**使用更小的模型**：

```python
session = agents.AgentSession(
    # gpt-4o-mini 成本仅为 gpt-4o 的 1/15
    llm=openai.LLM(model="gpt-4o-mini"),
    tts="openai-tts",
)
```

**限制 Token 使用**：

```python
# 限制历史长度
MAX_HISTORY = 20

@session.on("conversation_item_added")
def on_item_added(event):
    if len(session.history.messages) > MAX_HISTORY:
        session.history.messages = session.history.messages[-MAX_HISTORY:]
```

**批量处理**：

```python
# 对于非实时场景，批量处理多个请求
async def batch_process(requests: list[str]):
    llm_instance = openai.LLM()
    
    # 并发处理
    tasks = [
        llm_instance.chat(
            chat_ctx=llm.ChatContext.empty().append(
                llm.ChatMessage(role="user", content=req)
            )
        )
        for req in requests
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

### 5.3 优化资源使用

**进程池配置**：

```python
agents.cli.run_app(
    agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        # 根据负载动态调整
        num_idle_processes=4,  # 预热4个进程
        load_threshold=0.7,    # 负载超过70%标记为FULL
    )
)
```

**内存限制**：

```python
agents.cli.run_app(
    agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        job_memory_warn_mb=500,   # 超过500MB警告
        job_memory_limit_mb=1000, # 超过1GB杀死进程
    )
)
```

---

## 6. 故障排查

### 6.1 常见问题

#### 问题1：Agent 不响应

**症状**：用户说话后 Agent 没有回复

**排查步骤**：

1. 检查 Room 连接状态
```python
logger.info(f"Room connected: {ctx.room.is_connected}")
logger.info(f"Participants: {len(ctx.room.remote_participants)}")
```

2. 检查 STT 是否正常
```python
@session.on("user_input_transcribed")
def on_transcript(event):
    logger.info(f"User said: {event.transcript}")
```

3. 检查 LLM 推理
```python
@session.on("speech_created")
def on_speech(event):
    logger.info(f"Agent responding: {event.speech_handle.id}")
```

4. 检查 TTS 合成
```python
@session.output.audio.on("playback_finished")
def on_playback(event):
    logger.info(f"Playback finished: {event.playback_position}s")
```

#### 问题2：频繁打断

**症状**：Agent 说话时经常被误打断

**解决方案**：

```python
session = agents.AgentSession(
    llm="gpt-4o",
    tts="openai-tts",
    # 提高打断阈值
    min_interruption_duration=0.8,  # 增加到0.8秒
    min_interruption_words=2,       # 至少2个词
    # 启用误打断恢复
    false_interruption_timeout=2.0,
    resume_false_interruption=True,
)
```

#### 问题3：响应延迟高

**症状**：Agent 回复很慢

**排查步骤**：

1. 检查端点检测配置
```python
# 端点检测延迟过长
session = agents.AgentSession(
    min_endpointing_delay=0.3,  # 减小
    max_endpointing_delay=2.0,  # 减小
)
```

2. 启用预生成
```python
session = agents.AgentSession(
    preemptive_generation=True,
)
```

3. 使用更快的模型
```python
session = agents.AgentSession(
    llm=openai.realtime.RealtimeModel(),  # 或 gpt-4o-mini
    tts=cartesia.TTS(),  # 低延迟 TTS
)
```

### 6.2 调试技巧

**使用 CLI 模式调试**：

```bash
# 在终端模式运行，便于调试
python agent.py console
```

**启用详细日志**：

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

**使用本地模拟**：

```python
# 模拟房间，无需真实的 LiveKit 服务端
if __name__ == "__main__":
    worker = agents.Worker(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
    
    async def main():
        await worker.run()
        # 模拟任务
        await worker.simulate_job(
            agents.SimulateJobInfo(
                room="test-room",
                participant_identity="test-user",
            )
        )
    
    asyncio.run(main())
```

**追踪请求链路**：

```python
# 启用 OpenTelemetry 追踪
from livekit.agents import telemetry

# 配置追踪导出器
# （此处省略具体配置）

# 查看 Span 追踪
```

---

**本文档版本**：基于 LiveKit Agents SDK 主分支（2025-01-04）生成  
**下一步**：查看具体模块文档深入学习

