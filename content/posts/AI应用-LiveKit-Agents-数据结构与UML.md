---
title: "LiveKit Agents 数据结构与UML图"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['WebRTC', '实时通信', 'LiveKit', '语音处理', '架构设计']
categories: ['AI应用']
description: "LiveKit Agents 数据结构与UML图的深入技术分析文档"
keywords: ['WebRTC', '实时通信', 'LiveKit', '语音处理', '架构设计']
author: "技术分析师"
weight: 1
---

## 1. 核心数据结构概览

LiveKit Agents框架包含多个层次的数据结构，从基础类型到复杂的业务对象。本文档详细分析这些数据结构的设计和关系。

### 1.1 数据结构层次图

```mermaid
graph TB
    subgraph "基础类型层 (Basic Types)"
        NotGiven[NotGiven<br/>未给定类型]
        APIOptions[APIConnectOptions<br/>API连接选项]
        EventTypes[EventTypes<br/>事件类型定义]
    end
    
    subgraph "配置类型层 (Configuration Types)"
        WorkerOptions[WorkerOptions<br/>工作配置]
        VoiceOptions[VoiceOptions<br/>语音配置]
        ModelSettings[ModelSettings<br/>模型设置]
        SessionOptions[SessionConnectOptions<br/>会话连接选项]
    end
    
    subgraph "核心业务层 (Core Business)"
        Agent[Agent<br/>代理类]
        AgentSession[AgentSession<br/>代理会话]
        JobContext[JobContext<br/>任务上下文]
        ChatContext[ChatContext<br/>聊天上下文]
    end
    
    subgraph "事件数据层 (Event Data)"
        UserEvent[UserInputTranscribedEvent<br/>用户输入事件]
        AgentEvent[AgentStateChangedEvent<br/>代理状态事件]
        MetricsEvent[MetricsCollectedEvent<br/>指标收集事件]
    end
    
    subgraph "工具系统层 (Tool System)"
        FunctionTool[FunctionTool<br/>函数工具]
        ToolError[ToolError<br/>工具错误]
        RunContext[RunContext<br/>运行上下文]
    end
    
    subgraph "I/O数据层 (I/O Data)"
        AudioFrame[AudioFrame<br/>音频帧]
        VideoFrame[VideoFrame<br/>视频帧]
        ChatMessage[ChatMessage<br/>聊天消息]
    end
```

## 2. 核心类UML图

### 2.1 Agent类族UML图

```mermaid
classDiagram
    class Agent {
        -_instructions: str
        -_tools: list[FunctionTool]
        -_chat_ctx: ChatContext
        -_stt: STT | None
        -_llm: LLM | None
        -_tts: TTS | None
        -_vad: VAD | None
        -_activity: AgentActivity | None
        
        +__init__(instructions: str, **kwargs)
        +label: str
        +tools: list[FunctionTool]
        +update_tools(tools: list[FunctionTool]) -> None
        +on_enter() -> None
        +on_exit() -> None
        +llm_node(chat_ctx, tools, settings) -> LLMStream
        +transcription_node(input, settings) -> TextOutput
    }
    
    class ModelSettings {
        +tool_choice: ToolChoice
        +temperature: float
        +max_tokens: int
        +top_p: float
        
        +__init__(**kwargs)
        +to_dict() -> dict
    }
    
    class AgentActivity {
        -_agent: Agent
        -_session: AgentSession
        -_chat_ctx: ChatContext
        -_current_speech: SpeechHandle | None
        -_generating: bool
        -_tool_calling: bool
        
        +__init__(agent: Agent, session: AgentSession)
        +handle_user_input(text: str) -> None
        +generate_reply(**kwargs) -> None
        +interrupt() -> None
        +_pipeline_reply_task(**kwargs) -> None
        +_execute_tools(calls: list) -> list[str]
    }
    
    class AgentTask {
        <<enumeration>>
        LISTENING
        THINKING
        SPEAKING
        TOOL_CALLING
    }
    
    Agent ||--|| AgentActivity : creates
    Agent --* ModelSettings : uses
    AgentActivity --* AgentTask : has_state
    Agent ..> ChatContext : manages
```

### 2.2 AgentSession类族UML图

```mermaid
classDiagram
    class AgentSession~Userdata_T~ {
        -_stt: STT | None
        -_vad: VAD | None
        -_llm: LLM | None
        -_tts: TTS | None
        -_agent: Agent | None
        -_activity: AgentActivity | None
        -_user_state: UserState
        -_agent_state: AgentState
        -_userdata: Userdata_T | None
        -_input: AgentInput
        -_output: AgentOutput
        -_room_io: RoomIO | None
        
        +__init__(**voice_options)
        +start(agent: Agent, room: Room) -> None
        +generate_reply(**kwargs) -> None
        +interrupt() -> None
        +userdata: Userdata_T
        +current_agent: Agent
        +user_state: UserState
        +agent_state: AgentState
        +history: ChatContext
        +run(user_input: str) -> RunResult
    }
    
    class VoiceOptions {
        +allow_interruptions: bool
        +min_interruption_duration: float
        +min_endpointing_delay: float
        +max_endpointing_delay: float
        +max_tool_steps: int
        +preemptive_generation: bool
        +user_away_timeout: float | None
        +false_interruption_timeout: float | None
        
        +__init__(**kwargs)
        +validate() -> None
    }
    
    class UserState {
        <<enumeration>>
        LISTENING
        SPEAKING
        AWAY
    }
    
    class AgentState {
        <<enumeration>>
        INITIALIZING
        LISTENING
        THINKING
        SPEAKING
        TOOL_CALLING
    }
    
    class EventEmitter~T~ {
        <<interface>>
        +on(event: str, callback: Callable) -> None
        +emit(event: str, *args) -> None
        +off(event: str, callback: Callable) -> None
    }
    
    AgentSession --|> EventEmitter : extends
    AgentSession --* VoiceOptions : configured_with
    AgentSession --* UserState : has_state
    AgentSession --* AgentState : has_state
    AgentSession ||--|| Agent : manages
```

### 2.3 工具系统UML图

```mermaid
classDiagram
    class FunctionTool {
        <<interface>>
        +__livekit_tool_info: _FunctionToolInfo
        +__call__(*args, **kwargs) -> Any
    }
    
    class _FunctionToolInfo {
        +name: str
        +description: str | None
        +parameters: dict[str, Any]
        
        +__init__(name: str, description: str)
        +to_schema() -> dict
    }
    
    class RawFunctionDescription {
        +name: str
        +description: str
        +parameters: dict[str, Any]
        
        +__init__(**kwargs)
        +validate() -> None
    }
    
    class ToolChoice {
        <<union>>
        AUTO
        REQUIRED  
        NONE
        NamedToolChoice
    }
    
    class NamedToolChoice {
        +type: Literal["function"]
        +function: Function
    }
    
    class Function {
        +name: str
    }
    
    class ToolError {
        -_message: str
        
        +__init__(message: str)
        +message: str
    }
    
    class StopResponse {
        +__init__()
    }
    
    class RunContext {
        +userdata: Any
        +session: AgentSession
        +agent: Agent
        
        +__init__(session: AgentSession)
    }
    
    FunctionTool --* _FunctionToolInfo : contains
    FunctionTool ..> RawFunctionDescription : alternative_to
    ToolChoice --* NamedToolChoice : includes
    NamedToolChoice --* Function : contains
    ToolError --|> Exception : extends
    StopResponse --|> Exception : extends
    FunctionTool ..> RunContext : receives
```

### 2.4 聊天上下文UML图

```mermaid
classDiagram
    class ChatContext {
        -_messages: list[ChatMessage]
        -_tools: list[FunctionTool]
        -_metadata: dict[str, Any]
        
        +__init__(messages, tools, metadata)
        +empty(tools: list) -> ChatContext
        +copy(**overrides) -> ChatContext
        +append(role, content, **kwargs) -> ChatContext
        +messages: list[ChatMessage]
        +tools: list[FunctionTool]
        +metadata: dict[str, Any]
    }
    
    class ChatMessage {
        +role: ChatRole
        +content: ChatContent
        +name: str | None
        +tool_calls: list[FunctionCall] | None
        +tool_call_id: str | None
        
        +__init__(**kwargs)
        +to_dict() -> dict
    }
    
    class ChatRole {
        <<enumeration>>
        SYSTEM
        USER
        ASSISTANT
        TOOL
    }
    
    class ChatContent {
        <<union>>
        str
        list[ChatContentPart]
    }
    
    class ChatContentPart {
        <<abstract>>
        +type: str
    }
    
    class TextContentPart {
        +type: Literal["text"]
        +text: str
        
        +__init__(text: str)
    }
    
    class ImageContentPart {
        +type: Literal["image"]
        +image_url: ImageURL
        
        +__init__(image_url: ImageURL)
    }
    
    class FunctionCall {
        +id: str
        +function: FunctionCallData
        
        +__init__(id: str, function: FunctionCallData)
    }
    
    class FunctionCallData {
        +name: str
        +arguments: str
        
        +__init__(name: str, arguments: str)
        +parse_arguments() -> dict
    }
    
    class FunctionCallOutput {
        +call_id: str
        +content: str
        
        +__init__(call_id: str, content: str)
    }
    
    ChatContext --* ChatMessage : contains
    ChatMessage --* ChatRole : has
    ChatMessage --* ChatContent : has
    ChatContent --* ChatContentPart : may_contain
    ChatContentPart <|-- TextContentPart
    ChatContentPart <|-- ImageContentPart
    ChatMessage --* FunctionCall : may_have
    FunctionCall --* FunctionCallData : contains
    ChatContext ..> FunctionCallOutput : processes
```

## 3. 事件系统数据结构

### 3.1 事件类型UML图

```mermaid
classDiagram
    class AgentEvent {
        <<abstract>>
        +timestamp: float
        
        +__init__()
    }
    
    class UserInputTranscribedEvent {
        +text: str
        +confidence: float
        +is_final: bool
        +participant: RemoteParticipant | None
        
        +__init__(text: str, **kwargs)
    }
    
    class AgentStateChangedEvent {
        +previous_state: AgentState
        +current_state: AgentState
        
        +__init__(previous: AgentState, current: AgentState)
    }
    
    class AgentFalseInterruptionEvent {
        +duration: float
        +reason: str
        
        +__init__(duration: float, reason: str)
    }
    
    class SpeechCreatedEvent {
        +speech_handle: SpeechHandle
        +source: str
        
        +__init__(handle: SpeechHandle, source: str)
    }
    
    class MetricsCollectedEvent {
        +metrics: dict[str, Any]
        +timestamp: float
        
        +__init__(metrics: dict)
    }
    
    class FunctionToolsExecutedEvent {
        +tool_calls: list[FunctionCall]
        +tool_results: list[str]
        +execution_time: float
        
        +__init__(calls: list, results: list)
        +cancel_tool_reply() -> None
    }
    
    class CloseEvent {
        +reason: CloseReason
        +message: str | None
        
        +__init__(reason: CloseReason, message: str)
    }
    
    class CloseReason {
        <<enumeration>>
        USER_DISCONNECTED
        AGENT_ERROR
        ROOM_CLOSED
        TIMEOUT
        MANUAL
    }
    
    class ErrorEvent {
        +error: Exception
        +recoverable: bool
        
        +__init__(error: Exception, recoverable: bool)
    }
    
    AgentEvent <|-- UserInputTranscribedEvent
    AgentEvent <|-- AgentStateChangedEvent
    AgentEvent <|-- AgentFalseInterruptionEvent
    AgentEvent <|-- SpeechCreatedEvent
    AgentEvent <|-- MetricsCollectedEvent
    AgentEvent <|-- FunctionToolsExecutedEvent
    AgentEvent <|-- CloseEvent
    AgentEvent <|-- ErrorEvent
    CloseEvent --* CloseReason : has
```

### 3.2 运行结果数据结构

```mermaid
classDiagram
    class RunResult~T~ {
        -_user_input: str
        -_events: list[RunEvent]
        -_output_type: type[T] | None
        -_done: bool
        -_error: Exception | None
        
        +__init__(user_input: str, output_type: type[T])
        +user_input: str
        +events: list[RunEvent]
        +done() -> bool
        +wait_for_completion() -> T
        +expect: EventAssert
    }
    
    class RunEvent {
        +type: str
        +data: Any
        +timestamp: float
        
        +__init__(type: str, data: Any)
    }
    
    class EventAssert {
        -_result: RunResult
        -_current_index: int
        
        +__init__(result: RunResult)
        +next_event() -> EventRangeAssert
        +skip_next_event_if(**filters) -> EventAssert
        +no_more_events() -> None
    }
    
    class EventRangeAssert {
        -_events: list[RunEvent]
        
        +__init__(events: list[RunEvent])
        +is_message(role: ChatRole) -> EventRangeAssert
        +is_function_call(name: str) -> EventRangeAssert
        +is_function_call_output() -> EventRangeAssert
        +judge(llm: LLM, intent: str) -> None
    }
    
    class ChatMessageEvent {
        +role: ChatRole
        +content: str
        +tool_calls: list[FunctionCall] | None
        
        +__init__(**kwargs)
    }
    
    class FunctionCallEvent {
        +function_name: str
        +arguments: dict[str, Any]
        +call_id: str
        
        +__init__(**kwargs)
    }
    
    class FunctionCallOutputEvent {
        +call_id: str
        +output: str
        
        +__init__(**kwargs)
    }
    
    class AgentHandoffEvent {
        +from_agent: Agent
        +to_agent: Agent
        +message: str | None
        
        +__init__(from_agent, to_agent, message)
    }
    
    RunResult --* RunEvent : contains
    RunResult ||--|| EventAssert : provides
    EventAssert ||--|| EventRangeAssert : creates
    RunEvent <|-- ChatMessageEvent
    RunEvent <|-- FunctionCallEvent
    RunEvent <|-- FunctionCallOutputEvent
    RunEvent <|-- AgentHandoffEvent
```

## 4. I/O数据结构

### 4.1 音视频数据结构

```mermaid
classDiagram
    class AudioFrame {
        +data: bytes
        +sample_rate: int
        +num_channels: int
        +samples_per_channel: int
        +timestamp: float
        
        +__init__(data: bytes, **kwargs)
        +duration() -> float
        +remix_and_resample(**kwargs) -> AudioFrame
    }
    
    class VideoFrame {
        +width: int
        +height: int
        +format: VideoPixelFormat
        +data: bytes
        +timestamp: float
        
        +__init__(**kwargs)
        +to_image() -> PIL.Image
        +resize(width: int, height: int) -> VideoFrame
    }
    
    class AudioInput {
        -_frames: asyncio.Queue[AudioFrame]
        -_closed: bool
        
        +__init__()
        +push_frame(frame: AudioFrame) -> None
        +aclose() -> None
        +__aiter__() -> AsyncIterator[AudioFrame]
    }
    
    class AudioOutput {
        -_track: LocalAudioTrack | None
        -_closed: bool
        
        +__init__()
        +push_frame(frame: AudioFrame) -> None
        +aclose() -> None
    }
    
    class VideoInput {
        -_frames: asyncio.Queue[VideoFrame]
        -_closed: bool
        
        +__init__()
        +push_frame(frame: VideoFrame) -> None
        +aclose() -> None
        +__aiter__() -> AsyncIterator[VideoFrame]
    }
    
    class TextOutput {
        -_segments: list[str]
        -_closed: bool
        
        +__init__()
        +push_text(text: str) -> None
        +aclose() -> None
    }
    
    AudioInput --* AudioFrame : processes
    AudioOutput --* AudioFrame : processes
    VideoInput --* VideoFrame : processes
```

### 4.2 房间I/O配置结构

```mermaid
classDiagram
    class RoomInputOptions {
        +text_enabled: bool
        +audio_enabled: bool
        +video_enabled: bool
        +audio_sample_rate: int
        +audio_num_channels: int
        +noise_cancellation: NoiseCancellationOptions | None
        +text_input_cb: TextInputCallback
        +participant_kinds: list[ParticipantKind]
        +participant_identity: str | None
        +pre_connect_audio: bool
        +pre_connect_audio_timeout: float
        +close_on_disconnect: bool
        +delete_room_on_close: bool
        
        +__init__(**kwargs)
        +validate() -> None
    }
    
    class RoomOutputOptions {
        +transcription_enabled: bool
        +audio_enabled: bool
        +audio_sample_rate: int
        +audio_num_channels: int
        +audio_publish_options: TrackPublishOptions
        
        +__init__(**kwargs)
        +validate() -> None
    }
    
    class TextInputEvent {
        +text: str
        +info: TextStreamInfo
        +participant: RemoteParticipant
        
        +__init__(text: str, **kwargs)
    }
    
    class TextInputCallback {
        <<interface>>
        +__call__(session: AgentSession, event: TextInputEvent) -> None
    }
    
    RoomInputOptions --* TextInputCallback : uses
    TextInputCallback ..> TextInputEvent : receives
```

## 5. 配置和选项数据结构

### 5.1 Worker配置结构

```mermaid
classDiagram
    class WorkerOptions {
        +entrypoint_fnc: Callable[[JobContext], Awaitable[None]] | None
        +prewarm_fnc: Callable[[JobProcess], Any] | None
        +request_fnc: Callable[[JobRequest], Awaitable[None]] | None
        +initialize_process_fnc: Callable[[JobProcess], Any] | None
        +worker_type: WorkerType
        +permissions: WorkerPermissions | None
        +max_retry: int
        +ws_url: str | None
        +api_key: str | None
        +api_secret: str | None
        +port: int | None
        +host: str | None
        
        +__init__(**kwargs)
        +validate() -> None
    }
    
    class WorkerType {
        <<enumeration>>
        ROOM
        PUBLISHER
    }
    
    class WorkerPermissions {
        +can_publish: bool
        +can_subscribe: bool
        +can_publish_data: bool
        +hidden: bool
        +recorder: bool
        
        +__init__(**kwargs)
    }
    
    class JobExecutorType {
        <<enumeration>>
        PROCESS
        THREAD
    }
    
    class AutoSubscribe {
        <<enumeration>>
        SUBSCRIBE_ALL
        SUBSCRIBE_NONE
        AUDIO_ONLY
        VIDEO_ONLY
    }
    
    WorkerOptions --* WorkerType : has
    WorkerOptions --* WorkerPermissions : may_have
    WorkerOptions ..> JobExecutorType : uses
    WorkerOptions ..> AutoSubscribe : uses
```

### 5.2 API连接选项

```mermaid
classDiagram
    class APIConnectOptions {
        +timeout: float
        +max_retry: int
        +retry_interval: float
        +headers: dict[str, str] | None
        
        +__init__(**kwargs)
        +to_dict() -> dict
    }
    
    class SessionConnectOptions {
        +stt: APIConnectOptions | None
        +llm: APIConnectOptions | None
        +tts: APIConnectOptions | None
        
        +__init__(**kwargs)
        +get_stt_options() -> APIConnectOptions
        +get_llm_options() -> APIConnectOptions
        +get_tts_options() -> APIConnectOptions
    }
    
    class NotGiven {
        +__str__() -> str
        +__repr__() -> str
        +__bool__() -> bool
    }
    
    class NotGivenOr~T~ {
        <<type alias>>
        T | NotGiven
    }
    
    SessionConnectOptions --* APIConnectOptions : contains
    APIConnectOptions ..> NotGivenOr : uses_pattern
```

## 6. 数据结构关系总图

```mermaid
erDiagram
    Agent ||--|| AgentSession : manages
    AgentSession ||--o{ AgentActivity : creates
    AgentActivity ||--o{ SpeechHandle : manages
    
    Agent ||--o{ FunctionTool : has
    Agent ||--|| ChatContext : maintains
    ChatContext ||--o{ ChatMessage : contains
    
    AgentSession ||--|| RoomIO : uses
    RoomIO ||--o{ AudioInput : manages
    RoomIO ||--o{ AudioOutput : manages
    RoomIO ||--o{ VideoInput : manages
    
    JobContext ||--|| Worker : created_by
    Worker ||--|| WorkerOptions : configured_with
    
    AgentSession ||--o{ AgentEvent : emits
    AgentEvent ||--o{ RunEvent : becomes
    RunResult ||--o{ RunEvent : collects
    
    FunctionTool ||--|| _FunctionToolInfo : contains
    ChatMessage ||--o{ FunctionCall : may_have
    FunctionCall ||--|| FunctionCallData : contains
```

## 7. 数据流向图

```mermaid
flowchart TB
    subgraph "输入数据流"
        UserAudio[用户音频] --> AudioFrame[AudioFrame]
        UserText[用户文本] --> TextInput[TextInputEvent]
        UserVideo[用户视频] --> VideoFrame[VideoFrame]
    end
    
    subgraph "处理数据流"
        AudioFrame --> STTResult[STT结果]
        STTResult --> ChatMessage[ChatMessage]
        ChatMessage --> LLMResponse[LLM响应]
        LLMResponse --> FunctionCall[FunctionCall]
        FunctionCall --> FunctionResult[函数结果]
        FunctionResult --> TTSInput[TTS输入]
    end
    
    subgraph "输出数据流"
        TTSInput --> AudioOutput[音频输出]
        LLMResponse --> TextOutput[文本输出]
        ProcessingEvents[处理事件] --> AgentEvent[AgentEvent]
    end
    
    subgraph "状态数据流"
        UserState[用户状态] --> StateChange[状态变化]
        AgentState[代理状态] --> StateChange
        StateChange --> AgentStateChangedEvent[状态变化事件]
    end
    
    subgraph "配置数据流"
        WorkerOptions[Worker配置] --> JobContext[任务上下文]
        VoiceOptions[语音配置] --> AgentSession[代理会话]
        RoomOptions[房间配置] --> RoomIO[房间I/O]
    end
```

这个数据结构与UML文档详细展示了LiveKit Agents框架中所有重要数据结构的设计、关系和数据流向。每个UML图都包含了详细的属性和方法定义，帮助开发者理解框架的内部架构和数据组织方式。
