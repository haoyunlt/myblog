---
title: "LiveKit Agents 框架 - Voice 模块 API 详解"
date: 2025-10-05T10:45:52+08:00
draft: false
tags:
  - API设计
  - 接口文档
  - 源码分析
categories:
  - AI应用
description: "源码剖析 - LiveKit Agents 框架 - Voice 模块 API 详解"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# LiveKit Agents 框架 - Voice 模块 API 详解

## 目录

1. [AgentSession API](#1-agentsession-api)
2. [Agent API](#2-agent-api)
3. [RoomIO API](#3-roomio-api)
4. [SpeechHandle API](#4-speechhandle-api)
5. [事件 API](#5-事件-api)

---

## 1. AgentSession API

### 1.1 AgentSession.__init__()

#### 基本信息

- **名称**: `AgentSession.__init__()`
- **职责**: 创建 Agent 会话实例，配置运行时组件
- **返回**: AgentSession 实例

#### 请求参数结构

```python
def __init__(
    self,
    *,
    turn_detection: NotGivenOr[TurnDetectionMode] = NOT_GIVEN,
    stt: NotGivenOr[stt.STT | STTModels | str] = NOT_GIVEN,
    vad: NotGivenOr[vad.VAD] = NOT_GIVEN,
    llm: NotGivenOr[llm.LLM | llm.RealtimeModel | LLMModels | str] = NOT_GIVEN,
    tts: NotGivenOr[tts.TTS | TTSModels | str] = NOT_GIVEN,
    mcp_servers: NotGivenOr[list[mcp.MCPServer]] = NOT_GIVEN,
    userdata: NotGivenOr[Userdata_T] = NOT_GIVEN,
    allow_interruptions: bool = True,
    discard_audio_if_uninterruptible: bool = True,
    min_interruption_duration: float = 0.5,
    min_interruption_words: int = 0,
    min_endpointing_delay: float = 0.5,
    max_endpointing_delay: float = 6.0,
    max_tool_steps: int = 3,
    video_sampler: NotGivenOr[_VideoSampler | None] = NOT_GIVEN,
    user_away_timeout: float | None = 15.0,
    false_interruption_timeout: float | None = 2.0,
    resume_false_interruption: bool = True,
    min_consecutive_speech_delay: float = 0.0,
    use_tts_aligned_transcript: NotGivenOr[bool] = NOT_GIVEN,
    tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None] = NOT_GIVEN,
    preemptive_generation: bool = False,
    conn_options: NotGivenOr[SessionConnectOptions] = NOT_GIVEN,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None
```

#### 参数说明表

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|--------|------|
| `turn_detection` | `TurnDetectionMode` | 否 | NOT_GIVEN | 轮次检测模式："stt"/"vad"/"realtime_llm"/"manual" |
| `stt` | `STT \| str` | 否 | NOT_GIVEN | 语音识别组件或模型名 |
| `vad` | `VAD` | 否 | NOT_GIVEN | 语音活动检测组件 |
| `llm` | `LLM \| RealtimeModel \| str` | 否 | NOT_GIVEN | 大语言模型或模型名 |
| `tts` | `TTS \| str` | 否 | NOT_GIVEN | 语音合成组件或模型名 |
| `mcp_servers` | `list[MCPServer]` | 否 | NOT_GIVEN | MCP 服务器列表 |
| `userdata` | `T` | 否 | NOT_GIVEN | 用户自定义数据 |
| `allow_interruptions` | `bool` | 否 | True | 是否允许用户打断 Agent |
| `discard_audio_if_uninterruptible` | `bool` | 否 | True | Agent 不可打断时是否丢弃音频 |
| `min_interruption_duration` | `float` | 否 | 0.5 | 最小打断时长（秒） |
| `min_interruption_words` | `int` | 否 | 0 | 最小打断词数（需启用 STT） |
| `min_endpointing_delay` | `float` | 否 | 0.5 | 最小端点延迟（秒） |
| `max_endpointing_delay` | `float` | 否 | 6.0 | 最大端点延迟（秒） |
| `max_tool_steps` | `int` | 否 | 3 | 最大连续工具调用次数 |
| `video_sampler` | `_VideoSampler` | 否 | VoiceActivityVideoSampler | 视频采样器 |
| `user_away_timeout` | `float \| None` | 否 | 15.0 | 用户离开超时（秒），None 禁用 |
| `false_interruption_timeout` | `float \| None` | 否 | 2.0 | 误打断超时（秒），None 禁用 |
| `resume_false_interruption` | `bool` | 否 | True | 是否恢复误打断的语音 |
| `min_consecutive_speech_delay` | `float` | 否 | 0.0 | 连续语音最小间隔（秒） |
| `use_tts_aligned_transcript` | `bool` | 否 | NOT_GIVEN | 是否使用 TTS 对齐转录 |
| `tts_text_transforms` | `Sequence[TextTransforms]` | 否 | ["filter_markdown", "filter_emoji"] | TTS 文本转换列表 |
| `preemptive_generation` | `bool` | 否 | False | 是否启用预生成优化 |
| `conn_options` | `SessionConnectOptions` | 否 | NOT_GIVEN | 连接选项 |
| `loop` | `AbstractEventLoop` | 否 | None | 事件循环 |

#### 核心代码示例

```python
from livekit import agents
from livekit.plugins import openai, deepgram, cartesia, silero

# 1. 基础配置
session = agents.AgentSession(
    llm=openai.LLM(model="gpt-4o-mini"),
    stt=deepgram.STT(model="nova-3"),
    tts=cartesia.TTS(voice="male-conversational"),
)

# 2. 完整配置
session = agents.AgentSession(
    # 组件配置
    vad=silero.VAD(),
    stt=deepgram.STT(model="nova-3", language="multi"),
    llm=openai.LLM(model="gpt-4o"),
    tts=cartesia.TTS(voice="male-conversational"),
    
    # 轮次检测
    turn_detection="vad",  # 使用 VAD 检测
    
    # 打断配置
    allow_interruptions=True,
    min_interruption_duration=0.3,
    min_interruption_words=2,
    
    # 端点检测
    min_endpointing_delay=0.5,
    max_endpointing_delay=3.0,
    
    # 性能优化
    preemptive_generation=True,
    
    # 用户状态
    user_away_timeout=15.0,
    
    # 误打断处理
    false_interruption_timeout=2.0,
    resume_false_interruption=True,
)
```

#### 异常与边界

- **组件冲突**: 如果同时提供 `llm` 和 `mcp_servers`，确保 LLM 支持工具调用
- **轮次检测**: 如果指定 `turn_detection="vad"` 但未提供 `vad`，会自动降级
- **超时配置**: `min_endpointing_delay` 不应大于 `max_endpointing_delay`

---

### 1.2 AgentSession.start()

#### 基本信息

- **名称**: `AgentSession.start()`
- **职责**: 启动 Agent 会话，连接 I/O 并开始运行
- **返回**: `None` 或 `RunResult`（如果 `capture_run=True`）

#### 请求参数结构

```python
async def start(
    self,
    agent: Agent,
    *,
    capture_run: bool = False,
    room: NotGivenOr[rtc.Room] = NOT_GIVEN,
    room_input_options: NotGivenOr[room_io.RoomInputOptions] = NOT_GIVEN,
    room_output_options: NotGivenOr[room_io.RoomOutputOptions] = NOT_GIVEN,
) -> RunResult | None
```

#### 参数说明表

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|--------|------|
| `agent` | `Agent` | 是 | - | Agent 定义（指令、工具等） |
| `capture_run` | `bool` | 否 | False | 是否捕获运行结果 |
| `room` | `rtc.Room` | 否 | NOT_GIVEN | LiveKit Room 连接 |
| `room_input_options` | `RoomInputOptions` | 否 | NOT_GIVEN | 房间输入配置 |
| `room_output_options` | `RoomOutputOptions` | 否 | NOT_GIVEN | 房间输出配置 |

#### 核心代码示例

```python
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = agents.AgentSession(
        llm="gpt-4o",
        tts="openai-tts",
    )
    
    # 1. 基础启动
    await session.start(
        room=ctx.room,
        agent=agents.Agent(
            instructions="You are a helpful assistant."
        )
    )
    
    # 2. 自定义 I/O 配置
    await session.start(
        room=ctx.room,
        agent=agents.Agent(
            instructions="You are a helpful assistant."
        ),
        room_input_options=agents.voice.room_io.RoomInputOptions(
            audio_enabled=True,
            video_enabled=False,
            audio_sample_rate=16000,
        ),
        room_output_options=agents.voice.room_io.RoomOutputOptions(
            audio_enabled=True,
            transcription_enabled=True,
        ),
    )
    
    # 3. 捕获运行结果
    run_result = await session.start(
        room=ctx.room,
        agent=agents.Agent(
            instructions="You are a helpful assistant."
        ),
        capture_run=True,
    )
    
    # 等待完成
    await run_result.wait()
    print(f"Final output: {run_result.output}")
```

#### 调用链与上层适配

```python
# AgentSession.start() 内部调用流程：

async def start(self, agent, ...):
    # 1. 检查是否已启动
    if self._started:
        return None
    
    # 2. 初始化 Agent
    self._agent = agent
    self._update_agent_state("initializing")
    
    # 3. 设置 I/O
    if is_given(room) and not self._room_io:
        # 创建 RoomIO
        self._room_io = room_io.RoomIO(
            room=room,
            agent_session=self,
            input_options=room_input_options,
            output_options=room_output_options,
        )
        # 启动 RoomIO
        await self._room_io.start()
    
    # 4. 创建 AgentActivity
    await self._update_activity(self._agent, wait_on_enter=False)
    
    # 5. 启动音视频转发任务
    if self.input.audio:
        self._forward_audio_atask = asyncio.create_task(
            self._forward_audio_task()
        )
    
    # 6. 更新状态
    self._started = True
    self._update_agent_state("listening")
```

#### 异常与边界

- **重复启动**: 调用多次 `start()` 只有第一次生效
- **Room 连接**: 如果提供 `room`，必须已经调用 `ctx.connect()`
- **I/O 冲突**: 如果手动设置了 `session.input.audio`，`room_input_options.audio_enabled` 会被忽略

---

### 1.3 AgentSession.generate_reply()

#### 基本信息

- **名称**: `AgentSession.generate_reply()`
- **职责**: 主动触发 Agent 生成回复
- **返回**: `None`

#### 请求参数结构

```python
def generate_reply(
    self,
    *,
    user_input: str | None = None,
    instructions: str | None = None,
) -> None
```

#### 参数说明表

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|--------|------|
| `user_input` | `str` | 否 | None | 用户输入文本（模拟用户说话） |
| `instructions` | `str` | 否 | None | 临时指令（覆盖 Agent 的 instructions） |

#### 核心代码示例

```python
# 1. 生成欢迎语
await session.generate_reply(
    instructions="Greet the user warmly and introduce yourself."
)

# 2. 模拟用户输入
await session.generate_reply(
    user_input="What's the weather like today?"
)

# 3. 带特定指令生成回复
await session.generate_reply(
    user_input="Tell me about Python",
    instructions="Explain Python in simple terms for beginners."
)
```

#### 调用链与上层适配

```python
# generate_reply() 内部流程：

def generate_reply(self, *, user_input=None, instructions=None):
    # 1. 构造用户消息
    if user_input:
        user_msg = llm.ChatMessage(
            role="user",
            content=user_input,
        )
        self._chat_ctx.append(user_msg)
    
    # 2. 创建 SpeechHandle
    speech_handle = SpeechHandle(
        id=utils.shortuuid(),
        allow_interruptions=self._opts.allow_interruptions,
    )
    
    # 3. 触发 Agent 推理任务
    asyncio.create_task(
        self._activity._pipeline_reply_task(
            speech_handle=speech_handle,
            chat_ctx=self._chat_ctx.copy(),
            tools=self._agent._tools,
            model_settings=ModelSettings(),
            new_message=user_msg if user_input else None,
            instructions=instructions,
        )
    )
```

---

### 1.4 AgentSession.stop()

#### 基本信息

- **名称**: `AgentSession.stop()`
- **职责**: 停止当前 Agent，可选地启动新 Agent
- **返回**: `None`

#### 请求参数结构

```python
async def stop(
    self,
    *,
    agent: Agent | None = None,
) -> None
```

#### 参数说明表

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|--------|------|
| `agent` | `Agent` | 否 | None | 新 Agent（如果提供，停止后立即启动） |

#### 核心代码示例

```python
# 1. 停止当前 Agent
await session.stop()

# 2. 切换到新 Agent
await session.stop(
    agent=NewAgent()
)

# 3. 在工具函数中切换 Agent
class ReceptionAgent(agents.Agent):
    @agents.function_tool
    async def transfer_to_sales(self):
        """转接到销售部门"""
        session = self._activity._session
        await session.stop(agent=SalesAgent())
        return "Transferring to sales..."
```

---

## 2. Agent API

### 2.1 Agent.__init__()

#### 基本信息

- **名称**: `Agent.__init__()`
- **职责**: 定义 Agent 的行为和能力
- **返回**: Agent 实例

#### 请求参数结构

```python
def __init__(
    self,
    *,
    instructions: str,
    chat_ctx: NotGivenOr[llm.ChatContext | None] = NOT_GIVEN,
    tools: list[llm.FunctionTool | llm.RawFunctionTool] | None = None,
    turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
    stt: NotGivenOr[stt.STT | STTModels | str | None] = NOT_GIVEN,
    vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
    llm: NotGivenOr[llm.LLM | llm.RealtimeModel | LLMModels | str | None] = NOT_GIVEN,
    tts: NotGivenOr[tts.TTS | TTSModels | str | None] = NOT_GIVEN,
    mcp_servers: NotGivenOr[list[mcp.MCPServer] | None] = NOT_GIVEN,
    allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    min_consecutive_speech_delay: NotGivenOr[float] = NOT_GIVEN,
    use_tts_aligned_transcript: NotGivenOr[bool] = NOT_GIVEN,
    min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
    max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
) -> None
```

#### 参数说明表

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|--------|------|
| `instructions` | `str` | 是 | - | Agent 指令（系统提示词） |
| `chat_ctx` | `ChatContext` | 否 | NOT_GIVEN | 初始对话上下文 |
| `tools` | `list[FunctionTool]` | 否 | None | 工具函数列表 |
| 其他 | - | 否 | NOT_GIVEN | 覆盖 AgentSession 的配置 |

#### 核心代码示例

```python
# 1. 基础 Agent
agent = agents.Agent(
    instructions="You are a helpful assistant."
)

# 2. 带工具的 Agent
@agents.function_tool
async def get_weather(location: str) -> str:
    """获取天气信息"""
    return f"{location}: Sunny, 25°C"

agent = agents.Agent(
    instructions="You are a weather assistant.",
    tools=[get_weather],
)

# 3. 继承 Agent 类
class CustomAgent(agents.Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a custom agent.",
        )
    
    @agents.function_tool
    async def custom_tool(self, param: str) -> str:
        """自定义工具"""
        return f"Result: {param}"

# 4. 覆盖会话配置
agent = agents.Agent(
    instructions="You are fast.",
    min_endpointing_delay=0.3,  # 更快的端点检测
    allow_interruptions=False,   # 不允许打断
)
```

---

## 3. RoomIO API

### 3.1 RoomInputOptions

#### 数据结构

```python
@dataclass
class RoomInputOptions:
    audio_enabled: NotGivenOr[bool] = NOT_GIVEN
    video_enabled: bool = False
    text_enabled: NotGivenOr[bool] = NOT_GIVEN
    participant_identity: NotGivenOr[str] = NOT_GIVEN
    audio_sample_rate: int = 48000
    audio_num_channels: int = 1
    noise_cancellation: NotGivenOr[noise_cancellation.BVC | None] = NOT_GIVEN
    pre_connect_audio_enabled: bool = False
    pre_connect_audio_timeout: float = 10.0
```

#### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `audio_enabled` | `bool` | NOT_GIVEN | 是否启用音频输入 |
| `video_enabled` | `bool` | False | 是否启用视频输入 |
| `text_enabled` | `bool` | NOT_GIVEN | 是否启用文本输入 |
| `participant_identity` | `str` | NOT_GIVEN | 指定参与者身份 |
| `audio_sample_rate` | `int` | 48000 | 音频采样率 |
| `audio_num_channels` | `int` | 1 | 音频通道数 |
| `noise_cancellation` | `BVC` | NOT_GIVEN | Krisp 噪声消除 |
| `pre_connect_audio_enabled` | `bool` | False | 预连接音频缓存 |
| `pre_connect_audio_timeout` | `float` | 10.0 | 预连接超时 |

### 3.2 RoomOutputOptions

#### 数据结构

```python
@dataclass
class RoomOutputOptions:
    audio_enabled: NotGivenOr[bool] = NOT_GIVEN
    transcription_enabled: bool = False
    sync_transcription: bool = True
    transcription_speed: float = 1.0
```

#### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `audio_enabled` | `bool` | NOT_GIVEN | 是否启用音频输出 |
| `transcription_enabled` | `bool` | False | 是否启用转录输出 |
| `sync_transcription` | `bool` | True | 转录是否与音频同步 |
| `transcription_speed` | `float` | 1.0 | 转录同步速度倍率 |

---

## 4. SpeechHandle API

### 4.1 SpeechHandle 属性

#### 数据结构

```python
class SpeechHandle:
    @property
    def id(self) -> str:
        """语音句柄唯一 ID"""
    
    @property
    def is_final(self) -> bool:
        """是否已完成"""
    
    @property
    def allow_interruptions(self) -> bool:
        """是否允许打断"""
    
    @property
    def interrupted(self) -> bool:
        """是否被打断"""
```

### 4.2 SpeechHandle.interrupt()

#### 基本信息

- **名称**: `SpeechHandle.interrupt()`
- **职责**: 打断当前语音播放
- **返回**: `None`

#### 核心代码示例

```python
@session.on("speech_created")
def on_speech_created(event: agents.SpeechCreatedEvent):
    speech = event.speech_handle
    
    # 3 秒后自动打断
    async def auto_interrupt():
        await asyncio.sleep(3.0)
        if not speech.is_final:
            await speech.interrupt()
    
    asyncio.create_task(auto_interrupt())
```

---

## 5. 事件 API

### 5.1 事件类型

| 事件名 | 事件类 | 说明 |
|--------|--------|------|
| `agent_state_changed` | `AgentStateChangedEvent` | Agent 状态变更 |
| `user_state_changed` | `UserStateChangedEvent` | 用户状态变更 |
| `speech_created` | `SpeechCreatedEvent` | Agent 开始说话 |
| `user_input_transcribed` | `UserInputTranscribedEvent` | 用户输入转录完成 |
| `conversation_item_added` | `ConversationItemAddedEvent` | 对话项添加 |
| `function_tools_executed` | `FunctionToolsExecutedEvent` | 工具执行完成 |
| `metrics_collected` | `MetricsCollectedEvent` | 指标收集 |
| `agent_false_interruption` | `AgentFalseInterruptionEvent` | 误打断检测 |
| `close` | `CloseEvent` | 会话关闭 |
| `error` | `ErrorEvent` | 错误发生 |

### 5.2 事件监听示例

```python
# 监听 Agent 状态变更
@session.on("agent_state_changed")
def on_agent_state_changed(event: agents.AgentStateChangedEvent):
    print(f"Agent state: {event.state}")

# 监听用户输入
@session.on("user_input_transcribed")
def on_user_input(event: agents.UserInputTranscribedEvent):
    print(f"User said: {event.transcript}")

# 监听语音创建
@session.on("speech_created")
def on_speech_created(event: agents.SpeechCreatedEvent):
    print(f"Agent speaking: {event.speech_handle.id}")

# 监听指标
@session.on("metrics_collected")
def on_metrics(event: agents.MetricsCollectedEvent):
    metrics.log_metrics(event.metrics)

# 监听错误
@session.on("error")
def on_error(event: agents.ErrorEvent):
    logger.error(f"Session error: {event.error}")
```

---

**本文档版本**：基于 LiveKit Agents SDK 主分支（2025-01-04）生成  
**下一步**：查看时序图文档了解详细调用流程

