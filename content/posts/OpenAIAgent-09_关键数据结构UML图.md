# OpenAI Agents SDK 关键数据结构UML图

## 9.1 核心数据结构概览

OpenAI Agents SDK的数据结构设计遵循清晰的分层架构，主要包含以下几个层次：

- **执行层数据结构**: RunResult, RunConfig, RunContextWrapper等
- **代理层数据结构**: Agent, AgentBase, ModelSettings等  
- **工具层数据结构**: FunctionTool, ToolContext, Handoff等
- **消息层数据结构**: RunItem, ModelResponse, TResponseInputItem等
- **会话层数据结构**: Session, HandoffInputData等
- **追踪层数据结构**: Trace, Span, SpanData等

## 9.2 核心执行层UML类图

```mermaid
classDiagram
    class Runner {
        +run(agent: Agent, input: str) RunResult
        +run_sync(agent: Agent, input: str) RunResult  
        +run_streamed(agent: Agent, input: str) RunResultStreaming
    }
    
    class AgentRunner {
        -_validate_run_hooks(hooks) RunHooks
        -_run_single_turn(...) SingleStepResult
        -_get_new_response(...) ModelResponse
        -_run_input_guardrails(...) List[InputGuardrailResult]
        -_run_output_guardrails(...) List[OutputGuardrailResult]
    }
    
    class RunResult {
        +input: str | List[TResponseInputItem]
        +new_items: List[RunItem]
        +raw_responses: List[ModelResponse]
        +final_output: Any
        +input_guardrail_results: List[InputGuardrailResult]
        +output_guardrail_results: List[OutputGuardrailResult]
        +context_wrapper: RunContextWrapper
        +to_input_list() List[TResponseInputItem]
    }
    
    class RunConfig {
        +model: str | Model | None
        +model_provider: ModelProvider
        +model_settings: ModelSettings | None
        +handoff_input_filter: HandoffInputFilter | None
        +input_guardrails: List[InputGuardrail]
        +output_guardrails: List[OutputGuardrail]
        +tracing_disabled: bool
        +trace_include_sensitive_data: bool
        +workflow_name: str
        +trace_id: str | None
        +group_id: str | None
        +trace_metadata: Dict[str, Any] | None
        +session_input_callback: SessionInputCallback | None
        +call_model_input_filter: CallModelInputFilter | None
    }
    
    class RunContextWrapper~TContext~ {
        +context: TContext
        +usage: Usage
        +original_input: str | List[TResponseInputItem]
        +pre_handoff_items: List[RunItem]
        +new_items: List[RunItem]
        +current_agent: Agent
        +tool_name: str | None
        +tool_call_id: str | None
        +tool_arguments: str | None
        +update_for_handoff(data: HandoffInputData)
    }
    
    Runner --> AgentRunner : delegates to
    AgentRunner --> RunResult : creates
    AgentRunner --> RunConfig : uses
    RunResult --> RunContextWrapper : contains
```

## 9.3 Agent和模型层UML类图

```mermaid
classDiagram
    class AgentBase~TContext~ {
        +name: str
        +handoff_description: str | None
        +tools: List[Tool]
        +mcp_servers: List[MCPServer]
        +mcp_config: MCPConfig
        +get_mcp_tools(context) List[Tool]
        +get_all_tools(context) List[Tool]
    }
    
    class Agent~TContext~ {
        +instructions: str | Callable | None
        +prompt: Prompt | DynamicPromptFunction | None
        +handoffs: List[Agent | Handoff]
        +model: str | Model | None
        +model_settings: ModelSettings
        +input_guardrails: List[InputGuardrail]
        +output_guardrails: List[OutputGuardrail]
        +output_type: type[Any] | AgentOutputSchemaBase | None
        +hooks: AgentHooks | None
        +tool_use_behavior: str | StopAtTools | ToolsToFinalOutputFunction
        +reset_tool_choice: bool
        +clone(**kwargs) Agent[TContext]
        +as_tool(...) Tool
        +get_system_prompt(context) str | None
        +get_prompt(context) ResponsePromptParam | None
    }
    
    class ModelSettings {
        +temperature: float | None
        +max_tokens: int | None
        +top_p: float | None
        +frequency_penalty: float | None
        +presence_penalty: float | None
        +tool_choice: str | None
        +parallel_tool_calls: bool | None
        +reasoning: ReasoningSettings | None
        +logprobs: bool | None
        +top_logprobs: int | None
        +user: str | None
        +resolve(override: ModelSettings | None) ModelSettings
    }
    
    class Model {
        <<interface>>
        +get_response(...) ModelResponse
        +stream_response(...) AsyncIterator[ResponseEvent]
        +name: str
    }
    
    class ModelProvider {
        <<interface>>
        +get_model(name: str | None) Model
        +supports_model(name: str) bool
    }
    
    class MultiProvider {
        -_custom_providers: Dict[str, ModelProvider]
        -_openai_provider: OpenAIProvider
        -_litellm_provider: LiteLLMProvider | None
        +register_provider(pattern: str, provider: ModelProvider)
        +get_model(name: str | None) Model
        +supports_model(name: str) bool
    }
    
    AgentBase <|-- Agent : extends
    Agent --> ModelSettings : contains
    Agent --> Tool : contains multiple
    Agent --> Handoff : contains multiple
    Model <|.. OpenAIResponsesModel : implements
    Model <|.. OpenAIChatCompletionsModel : implements
    Model <|.. LitellmModel : implements
    ModelProvider <|.. MultiProvider : implements
    ModelProvider <|.. OpenAIProvider : implements
    MultiProvider --> ModelProvider : aggregates
```

## 9.4 工具系统UML类图

```mermaid
classDiagram
    class Tool {
        <<interface>>
        +name: str
        +description: str
        +execute(context: ToolContext, arguments: str) Any
        +is_enabled: bool | Callable
    }
    
    class FunctionTool {
        +name: str
        +description: str
        +params_json_schema: Dict[str, Any]
        +on_invoke_tool: Callable
        +strict_json_schema: bool
        +is_enabled: bool | Callable
        +execute(context: ToolContext, arguments: str) Any
        +to_openai_format() Dict[str, Any]
    }
    
    class ToolContext~TContext~ {
        +run_context: RunContextWrapper[TContext]
        +agent: AgentBase
        +tool_name: str
        +tool_call_id: str
        +tool_arguments: str
        +model_provider: ModelProvider
        +context: TContext
        +usage: Usage
        +create_sub_agent(name, instructions, **kwargs) Agent[TContext]
    }
    
    class MCPTool {
        +server: MCPServer
        +tool_name: str
        +tool_schema: Dict[str, Any]
        +description: str
        +execute(context: ToolContext, arguments: str) Any
        +to_openai_format() Dict[str, Any]
    }
    
    class FileSearchTool {
        +file_ids: List[str]
        +filters: Filters | None
        +ranking_options: RankingOptions | None
        +description: str
        +execute(context: ToolContext, arguments: str) str
        +to_openai_format() Dict[str, Any]
    }
    
    class CodeInterpreterTool {
        +description: str
        +execute(context: ToolContext, arguments: str) str
        +to_openai_format() Dict[str, Any]
    }
    
    class ComputerTool {
        +computer: Computer | AsyncComputer
        +description: str
        +execute(context: ToolContext, arguments: str) str
        +to_openai_format() Dict[str, Any]
    }
    
    class LocalShellTool {
        +executor: LocalShellExecutor
        +description: str
        +execute(context: ToolContext, arguments: str) str
        +to_openai_format() Dict[str, Any]
    }
    
    Tool <|.. FunctionTool : implements
    Tool <|.. MCPTool : implements
    Tool <|.. FileSearchTool : implements
    Tool <|.. CodeInterpreterTool : implements
    Tool <|.. ComputerTool : implements
    Tool <|.. LocalShellTool : implements
    Tool --> ToolContext : uses
```

## 9.5 代理协作层UML类图

```mermaid
classDiagram
    class Handoff~TContext, TAgent~ {
        +tool_name: str
        +tool_description: str
        +input_json_schema: Dict[str, Any]
        +on_invoke_handoff: Callable
        +agent_name: str
        +strict_json_schema: bool
        +input_filter: HandoffInputFilter | None
        +is_enabled: bool | Callable
        +to_openai_format() Dict[str, Any]
    }
    
    class HandoffInputData {
        +input_history: str | Tuple[TResponseInputItem, ...]
        +pre_handoff_items: Tuple[RunItem, ...]
        +new_items: Tuple[RunItem, ...]
        +run_context: RunContextWrapper | None
        +clone(**kwargs) HandoffInputData
    }
    
    class InputGuardrail~TContext~ {
        <<interface>>
        +check(context, agent, input_data) GuardrailFunctionOutput
        +name: str
        +get_name() str
    }
    
    class OutputGuardrail~TContext~ {
        <<interface>>
        +check(context, agent, output) GuardrailFunctionOutput
        +name: str
        +get_name() str
    }
    
    class GuardrailFunctionOutput {
        +tripwire_triggered: bool
        +message: str
        +severity: Literal["low", "medium", "high"]
        +metadata: Dict[str, Any]
    }
    
    class InputGuardrailResult {
        +guardrail: InputGuardrail
        +output: GuardrailFunctionOutput
        +execution_time: float
    }
    
    class OutputGuardrailResult {
        +guardrail: OutputGuardrail
        +output: GuardrailFunctionOutput
        +execution_time: float
    }
    
    Handoff --> HandoffInputData : processes
    InputGuardrail --> GuardrailFunctionOutput : returns
    OutputGuardrail --> GuardrailFunctionOutput : returns
    InputGuardrailResult --> InputGuardrail : contains
    InputGuardrailResult --> GuardrailFunctionOutput : contains
    OutputGuardrailResult --> OutputGuardrail : contains
    OutputGuardrailResult --> GuardrailFunctionOutput : contains
```

## 9.6 消息和项目层UML类图

```mermaid
classDiagram
    class RunItem {
        <<interface>>
        +to_input_item() TResponseInputItem
        +agent: Agent
    }
    
    class MessageOutputItem {
        +raw_item: ResponseOutputMessage
        +agent: Agent
        +to_input_item() TResponseInputItem
    }
    
    class ToolCallItem {
        +raw_item: ToolCallItemTypes
        +agent: Agent
        +name: str
        +call_id: str
        +arguments: str
        +to_input_item() TResponseInputItem
    }
    
    class ToolCallOutputItem {
        +call_id: str
        +output: Any
        +agent: Agent
        +to_input_item() TResponseInputItem
    }
    
    class HandoffCallItem {
        +raw_item: ResponseOutputHandoffCall
        +agent: Agent
        +handoff_name: str
        +arguments: str
        +to_input_item() TResponseInputItem
    }
    
    class HandoffOutputItem {
        +handoff_name: str
        +result: str
        +agent: Agent
        +to_input_item() TResponseInputItem
    }
    
    class ReasoningItem {
        +raw_item: ResponseReasoningItem
        +agent: Agent
        +to_input_item() TResponseInputItem
    }
    
    class ModelResponse {
        +output: List[TResponseOutputItem]
        +usage: Usage
        +response_id: str | None
        +to_input_items() List[TResponseInputItem]
    }
    
    class Usage {
        +requests: int
        +input_tokens: int
        +output_tokens: int
        +total_tokens: int
        +input_tokens_details: Any | None
        +output_tokens_details: Any | None
        +add(other: Usage)
    }
    
    RunItem <|.. MessageOutputItem : implements
    RunItem <|.. ToolCallItem : implements
    RunItem <|.. ToolCallOutputItem : implements
    RunItem <|.. HandoffCallItem : implements
    RunItem <|.. HandoffOutputItem : implements
    RunItem <|.. ReasoningItem : implements
    ModelResponse --> Usage : contains
```

## 9.7 会话管理UML类图

```mermaid
classDiagram
    class SessionABC {
        <<interface>>
        +get_items(limit: int | None) List[dict]
        +add_items(items: List[dict])
        +pop_item() dict | None
        +clear_session()
    }
    
    class SQLiteSession {
        -session_id: str
        -db_path: str
        -table_name: str
        -_lock: asyncio.Lock
        -_ensure_table_exists()
        -_get_connection() AsyncContextManager
        +get_items(limit: int | None) List[dict]
        +add_items(items: List[dict])
        +pop_item() dict | None
        +clear_session()
        +get_session_stats() Dict[str, Any]
    }
    
    class RedisSession {
        -session_id: str
        -redis_client: redis.Redis
        -key: str
        -ttl: int | None
        +from_url(session_id: str, url: str) RedisSession
        -_create_default_client() redis.Redis
        +get_items(limit: int | None) List[dict]
        +add_items(items: List[dict])
        +pop_item() dict | None
        +clear_session()
        +get_session_info() Dict[str, Any]
    }
    
    class OpenAIConversationsSession {
        -conversation_id: str
        -client: AsyncOpenAI
        +get_items(limit: int | None) List[dict]
        +add_items(items: List[dict])
        +pop_item() dict | None
        +clear_session()
    }
    
    SessionABC <|.. SQLiteSession : implements
    SessionABC <|.. RedisSession : implements  
    SessionABC <|.. OpenAIConversationsSession : implements
```

## 9.8 追踪系统UML类图

```mermaid
classDiagram
    class Trace {
        +trace_id: str
        +workflow_name: str
        +group_id: str | None
        +metadata: Dict[str, Any]
        +start_time: datetime
        +end_time: datetime | None
        +root_spans: List[Span]
        -_is_current: bool
        +start(mark_as_current: bool) Trace
        +finish(reset_current: bool)
    }
    
    class Span~TSpanData~ {
        +span_data: TSpanData
        -_is_current: bool
        -_children: List[Span]
        -_parent: Span | None
        +start(mark_as_current: bool) Span[TSpanData]
        +finish(reset_current: bool)
        +add_child(child: Span)
        +set_error(error: Exception | str)
    }
    
    class SpanData {
        +span_id: str
        +name: str
        +start_time: datetime
        +end_time: datetime | None
        +parent_span_id: str | None
        +trace_id: str
        +metadata: Dict[str, Any]
        +tags: Dict[str, str]
    }
    
    class AgentSpanData {
        +agent_name: str
        +handoffs: List[str]
        +output_type: str
        +tools: List[str]
        +error: str | None
    }
    
    class GenerationSpanData {
        +model: str
        +provider: str
        +input: Any
        +output: Any
        +usage: Dict[str, int] | None
        +settings: Dict[str, Any]
    }
    
    class FunctionSpanData {
        +function_name: str
        +arguments: Dict[str, Any]
        +result: Any
        +execution_time: float
    }
    
    class TracingProcessor {
        <<interface>>
        +process_span_start(span: Span)
        +process_span_end(span: Span)
        +process_trace_complete(trace: Trace)
    }
    
    class ConsoleTracingProcessor {
        -logger: logging.Logger
        -include_data: bool
        +process_span_start(span: Span)
        +process_span_end(span: Span)
        +process_trace_complete(trace: Trace)
    }
    
    class FileTracingProcessor {
        -file_path: Path
        -format: str
        -include_sensitive_data: bool
        -_write_event(event_type: str, data: Any)
        -_serialize_data(data: Any) Any
        +process_span_start(span: Span)
        +process_span_end(span: Span)
        +process_trace_complete(trace: Trace)
    }
    
    Trace --> Span : contains multiple
    Span --> SpanData : contains
    SpanData <|-- AgentSpanData : extends
    SpanData <|-- GenerationSpanData : extends  
    SpanData <|-- FunctionSpanData : extends
    TracingProcessor <|.. ConsoleTracingProcessor : implements
    TracingProcessor <|.. FileTracingProcessor : implements
```

## 9.9 数据结构关系总览图

```mermaid
classDiagram
    direction TB
    
    class Runner {
        +run() RunResult
    }
    
    class Agent {
        +name: str
        +tools: List[Tool]
        +handoffs: List[Handoff]
    }
    
    class Tool {
        <<interface>>
        +execute()
    }
    
    class Session {
        <<interface>>
        +get_items()
        +add_items()
    }
    
    class Trace {
        +workflow_name: str
        +spans: List[Span]
    }
    
    class RunResult {
        +final_output: Any
        +new_items: List[RunItem]
    }
    
    class ModelResponse {
        +output: List[Item]
        +usage: Usage
    }
    
    class RunContextWrapper {
        +context: Any
        +usage: Usage
    }
    
    Runner --> Agent : executes
    Runner --> RunResult : returns
    Runner --> Session : uses
    Runner --> Trace : creates
    Agent --> Tool : contains
    Agent --> Handoff : contains
    RunResult --> RunItem : contains
    RunResult --> ModelResponse : contains
    RunResult --> RunContextWrapper : contains
    
    style Runner fill:#e3f2fd
    style Agent fill:#f3e5f5
    style Tool fill:#e8f5e8
    style Session fill:#fff3e0
    style Trace fill:#fce4ec
```

## 9.10 数据流向图

```mermaid
flowchart TD
    A[用户输入] --> B[Runner.run]
    B --> C[准备RunConfig]
    C --> D[创建RunContextWrapper]
    D --> E[初始化Trace]
    
    E --> F[代理执行循环]
    F --> G[获取Agent工具]
    G --> H[调用Model.get_response]
    H --> I[处理ModelResponse]
    
    I --> J{响应类型?}
    J -->|工具调用| K[执行Tool.execute]
    J -->|代理切换| L[处理Handoff]
    J -->|最终输出| M[应用OutputGuardrail]
    
    K --> N[创建ToolCallOutputItem]
    N --> F
    
    L --> O[应用HandoffInputFilter]
    O --> P[切换到新Agent]
    P --> F
    
    M --> Q[创建RunResult]
    Q --> R[保存到Session]
    R --> S[完成Trace]
    S --> T[返回结果给用户]
    
    style A fill:#e1f5fe
    style T fill:#e8f5e8
    style F fill:#fff3e0
    style J fill:#f3e5f5
```

这些UML图和数据结构分析展示了OpenAI Agents SDK的完整架构设计，体现了其高度模块化、类型安全和扩展性强的特点。通过这些图表，开发者可以更好地理解框架的内部工作原理和各组件间的关系。
