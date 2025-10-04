# OpenAI Agents Python SDK - Models 模块 API 详解

## 1. API 总览

Models 模块提供了模型抽象层，支持多种LLM提供商的统一接口。通过 `Model` 和 `ModelProvider` 接口，开发者可以灵活地切换和扩展不同的语言模型。

### API 层次结构

```
ModelProvider (提供商接口)
    ├── MultiProvider (多提供商路由)
    ├── OpenAIProvider (OpenAI实现)
    └── 自定义Provider (可扩展)

Model (模型接口)
    ├── get_response() (标准响应)
    └── stream_response() (流式响应)

ModelSettings (模型配置)
    └── 温度、token限制等参数
```

### API 分类

| API 类别 | 核心 API | 功能描述 |
|---------|---------|---------|
| **模型接口** | `Model.get_response()` | 获取模型标准响应 |
| | `Model.stream_response()` | 获取模型流式响应 |
| **提供商接口** | `ModelProvider.get_model()` | 根据名称获取模型实例 |
| **多提供商** | `MultiProvider.__init__()` | 创建多提供商路由 |
| | `MultiProvider.get_model()` | 路由到对应提供商 |
| **提供商映射** | `MultiProviderMap.add_provider()` | 添加自定义提供商 |
| | `MultiProviderMap.get_provider()` | 查找提供商 |

## 2. Model 接口 API

### 2.1 Model.get_response - 获取标准响应

**API 签名：**
```python
@abc.abstractmethod
async def get_response(
    self,
    system_instructions: str | None,
    input: str | list[TResponseInputItem],
    model_settings: ModelSettings,
    tools: list[Tool],
    output_schema: AgentOutputSchemaBase | None,
    handoffs: list[Handoff],
    tracing: ModelTracing,
    *,
    previous_response_id: str | None,
    conversation_id: str | None,
    prompt: ResponsePromptParam | None,
) -> ModelResponse
```

**功能描述：**
从语言模型获取完整响应。这是模型交互的核心方法，所有模型实现必须提供此方法。

**请求参数：**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `system_instructions` | `str \| None` | 系统指令（Agent的instructions） |
| `input` | `str \| list[TResponseInputItem]` | 输入内容（用户消息、历史等） |
| `model_settings` | `ModelSettings` | 模型参数配置 |
| `tools` | `list[Tool]` | 可用工具列表 |
| `output_schema` | `AgentOutputSchemaBase \| None` | 输出结构化Schema |
| `handoffs` | `list[Handoff]` | 可用的代理切换列表 |
| `tracing` | `ModelTracing` | 追踪配置 |
| `previous_response_id` | `str \| None` | 上一个响应ID（OpenAI Responses API） |
| `conversation_id` | `str \| None` | 对话ID（服务器端状态） |
| `prompt` | `ResponsePromptParam \| None` | 提示词配置 |

**返回结构：**
```python
@dataclass
class ModelResponse:
    response_id: str | None  # 响应ID
    output: list  # 输出项列表（消息、工具调用等）
    usage: Usage  # Token使用统计
    raw_response: dict  # 原始响应数据
```

**实现示例：**
```python
from agents.models import Model, ModelResponse
from agents import Usage

class CustomModel(Model):
    """自定义模型实现"""
    
    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> ModelResponse:
        """实现模型调用逻辑"""
        
        # 1. 准备请求数据
        request_data = {
            "messages": self._prepare_messages(system_instructions, input),
            "temperature": model_settings.temperature,
            "max_tokens": model_settings.max_tokens,
            "tools": self._convert_tools(tools),
        }
        
        # 2. 调用实际的LLM API
        response = await self._call_llm_api(request_data)
        
        # 3. 解析响应
        output_items = self._parse_output(response)
        
        # 4. 统计Token使用
        usage = Usage(
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=response.get("usage", {}).get("completion_tokens", 0),
            total_tokens=response.get("usage", {}).get("total_tokens", 0)
        )
        
        # 5. 返回标准化响应
        return ModelResponse(
            response_id=response.get("id"),
            output=output_items,
            usage=usage,
            raw_response=response
        )
```

### 2.2 Model.stream_response - 获取流式响应

**API 签名：**
```python
@abc.abstractmethod
def stream_response(
    self,
    system_instructions: str | None,
    input: str | list[TResponseInputItem],
    model_settings: ModelSettings,
    tools: list[Tool],
    output_schema: AgentOutputSchemaBase | None,
    handoffs: list[Handoff],
    tracing: ModelTracing,
    *,
    previous_response_id: str | None,
    conversation_id: str | None,
    prompt: ResponsePromptParam | None,
) -> AsyncIterator[TResponseStreamEvent]
```

**功能描述：**
从语言模型获取流式响应。适用于需要实时反馈的场景。

**参数说明：**
与 `get_response()` 完全相同。

**返回结构：**
```python
AsyncIterator[TResponseStreamEvent]  # 异步生成器，产生流式事件
```

**流式事件类型：**
- `response.output_item.done`: 输出项完成
- `response.audio.delta`: 音频增量数据
- `response.function_call_arguments.delta`: 工具调用参数增量
- 其他OpenAI Responses API事件

**实现示例：**
```python
class CustomModel(Model):
    """支持流式响应的模型"""
    
    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """实现流式响应"""
        
        # 准备请求
        request_data = self._prepare_request(
            system_instructions, input, model_settings, tools
        )
        request_data["stream"] = True
        
        # 调用流式API
        async for chunk in self._call_llm_stream_api(request_data):
            # 转换为标准事件格式
            event = self._convert_to_response_event(chunk)
            yield event
```

## 3. ModelProvider 接口 API

### 3.1 ModelProvider.get_model - 获取模型实例

**API 签名：**
```python
@abc.abstractmethod
def get_model(self, model_name: str | None) -> Model
```

**功能描述：**
根据模型名称获取对应的模型实例。这是提供商的核心方法。

**请求参数：**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `model_name` | `str \| None` | 模型名称（如"gpt-4o"） |

**返回结构：**
```python
Model  # 实现了Model接口的实例
```

**实现示例：**
```python
from agents.models import ModelProvider, Model

class CustomProvider(ModelProvider):
    """自定义模型提供商"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self._model_cache = {}
    
    def get_model(self, model_name: str | None) -> Model:
        """根据名称返回模型实例"""
        
        # 默认模型
        if model_name is None:
            model_name = "default-model"
        
        # 缓存模型实例
        if model_name not in self._model_cache:
            self._model_cache[model_name] = CustomModel(
                name=model_name,
                api_key=self.api_key,
                base_url=self.base_url
            )
        
        return self._model_cache[model_name]
```

## 4. MultiProvider API

### 4.1 MultiProvider.__init__ - 创建多提供商路由

**API 签名：**
```python
def __init__(
    self,
    *,
    provider_map: MultiProviderMap | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    openai_organization: str | None = None,
    openai_project: str | None = None,
    openai_use_responses: bool | None = None,
)
```

**功能描述：**
创建支持多个模型提供商的路由器。根据模型名称前缀自动选择对应的提供商。

**请求参数：**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `provider_map` | `MultiProviderMap \| None` | `None` | 自定义提供商映射 |
| `openai_api_key` | `str \| None` | `None` | OpenAI API密钥 |
| `openai_base_url` | `str \| None` | `None` | OpenAI API基础URL |
| `openai_client` | `AsyncOpenAI \| None` | `None` | 自定义OpenAI客户端 |
| `openai_organization` | `str \| None` | `None` | OpenAI组织ID |
| `openai_project` | `str \| None` | `None` | OpenAI项目ID |
| `openai_use_responses` | `bool \| None` | `None` | 是否使用Responses API |

**默认路由规则：**
```python
# 无前缀或"openai/"前缀 -> OpenAIProvider
"gpt-4o" -> OpenAIProvider
"openai/gpt-4o" -> OpenAIProvider

# "litellm/"前缀 -> LitellmProvider
"litellm/claude-3-opus" -> LitellmProvider

# 自定义前缀 -> 自定义Provider
"custom/my-model" -> CustomProvider (需要在provider_map中配置)
```

**使用示例：**

```python
from agents.models import MultiProvider
from openai import AsyncOpenAI

# 基础用法（默认OpenAI）
provider = MultiProvider()

# 自定义OpenAI配置
provider = MultiProvider(
    openai_api_key="your-api-key",
    openai_base_url="https://api.openai.com/v1"
)

# 使用自定义客户端
custom_client = AsyncOpenAI(
    api_key="your-key",
    base_url="https://custom.api.com"
)
provider = MultiProvider(openai_client=custom_client)

# 获取模型
model = provider.get_model("gpt-4o")
model_litellm = provider.get_model("litellm/claude-3-opus")
```

### 4.2 MultiProvider.get_model - 路由到提供商

**API 签名：**
```python
def get_model(self, model_name: str | None) -> Model
```

**功能描述：**
根据模型名称的前缀路由到对应的提供商，然后获取模型实例。

**路由逻辑：**
```python
# 解析前缀
"openai/gpt-4o" -> prefix="openai", name="gpt-4o"
"litellm/claude" -> prefix="litellm", name="claude"
"gpt-4o" -> prefix=None, name="gpt-4o"

# 查找提供商
if prefix in provider_map:
    return provider_map[prefix].get_model(name)
else:
    return fallback_provider.get_model(name)
```

**使用示例：**
```python
provider = MultiProvider()

# OpenAI模型
gpt4 = provider.get_model("gpt-4o")
gpt4_explicit = provider.get_model("openai/gpt-4o")

# LiteLLM模型
claude = provider.get_model("litellm/claude-3-opus")
gemini = provider.get_model("litellm/gemini-pro")

# 使用模型
response = await gpt4.get_response(
    system_instructions="你是一个助手",
    input="Hello",
    model_settings=ModelSettings(),
    tools=[],
    output_schema=None,
    handoffs=[],
    tracing=ModelTracing.ENABLED
)
```

## 5. MultiProviderMap API

### 5.1 添加自定义提供商

**API 方法：**
```python
def add_provider(self, prefix: str, provider: ModelProvider)
def get_provider(self, prefix: str) -> ModelProvider | None
def has_prefix(self, prefix: str) -> bool
def remove_provider(self, prefix: str)
```

**使用示例：**

```python
from agents.models import MultiProvider, MultiProviderMap, ModelProvider

# 创建自定义提供商
class MyCustomProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return MyCustomModel(model_name)

# 创建提供商映射
provider_map = MultiProviderMap()
provider_map.add_provider("custom", MyCustomProvider())
provider_map.add_provider("azure", AzureProvider())

# 使用自定义映射创建MultiProvider
provider = MultiProvider(provider_map=provider_map)

# 现在可以使用自定义前缀
custom_model = provider.get_model("custom/my-model")
azure_model = provider.get_model("azure/gpt-4")
```

### 5.2 完整的自定义提供商示例

```python
from agents.models import ModelProvider, Model, MultiProvider, MultiProviderMap
from agents import ModelResponse, Usage, ModelSettings
import httpx

class HuggingFaceModel(Model):
    """Hugging Face模型实现"""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co/models"
    
    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list,
        output_schema,
        handoffs: list,
        tracing,
        **kwargs
    ) -> ModelResponse:
        """调用Hugging Face API"""
        
        # 准备请求
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/{self.model_name}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "inputs": self._format_input(system_instructions, input),
                    "parameters": {
                        "temperature": model_settings.temperature,
                        "max_new_tokens": model_settings.max_tokens,
                    }
                }
            )
        
        # 解析响应
        data = response.json()
        output_text = data[0]["generated_text"]
        
        return ModelResponse(
            response_id=None,
            output=[{"type": "message", "content": output_text}],
            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            raw_response=data
        )
    
    async def stream_response(self, **kwargs):
        """流式响应（简化版）"""
        response = await self.get_response(**kwargs)
        yield response.output[0]

class HuggingFaceProvider(ModelProvider):
    """Hugging Face提供商"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_model(self, model_name: str | None) -> Model:
        """返回Hugging Face模型"""
        if model_name is None:
            model_name = "gpt2"
        return HuggingFaceModel(model_name, self.api_key)

# 注册到MultiProvider
provider_map = MultiProviderMap()
provider_map.add_provider("hf", HuggingFaceProvider(api_key="your-key"))

provider = MultiProvider(provider_map=provider_map)

# 使用Hugging Face模型
agent = Agent(
    name="HFAgent",
    model="hf/gpt2",  # 使用自定义前缀
    instructions="你是一个助手"
)
```

## 6. ModelTracing 枚举

### 6.1 追踪级别

**枚举定义：**
```python
class ModelTracing(enum.Enum):
    DISABLED = 0              # 完全禁用追踪
    ENABLED = 1               # 启用追踪，包含所有数据
    ENABLED_WITHOUT_DATA = 2  # 启用追踪，但不包含输入输出数据
```

**方法：**
```python
def is_disabled(self) -> bool:
    """是否禁用追踪"""
    return self == ModelTracing.DISABLED

def include_data(self) -> bool:
    """是否包含数据"""
    return self == ModelTracing.ENABLED
```

**使用示例：**
```python
from agents.models import ModelTracing

# 完全启用追踪
tracing = ModelTracing.ENABLED

# 启用追踪但不记录敏感数据
tracing = ModelTracing.ENABLED_WITHOUT_DATA

# 禁用追踪
tracing = ModelTracing.DISABLED

# 在模型调用中使用
response = await model.get_response(
    ...,
    tracing=tracing
)
```

## 7. 最佳实践

### 7.1 多模型支持

```python
from agents import Agent, RunConfig
from agents.models import MultiProvider

# 创建支持多模型的提供商
provider = MultiProvider()

# 为不同任务使用不同模型
fast_agent = Agent(
    name="FastAgent",
    model="gpt-4o-mini",
    instructions="快速响应"
)

smart_agent = Agent(
    name="SmartAgent",
    model="gpt-4o",
    instructions="深度思考"
)

# 在运行时切换模型
config = RunConfig(
    model="gpt-4o",  # 覆盖代理的模型
    model_provider=provider
)
```

### 7.2 模型回退策略

```python
class FallbackModel(Model):
    """支持回退的模型"""
    
    def __init__(self, primary: Model, fallback: Model):
        self.primary = primary
        self.fallback = fallback
    
    async def get_response(self, **kwargs) -> ModelResponse:
        """尝试主模型，失败时使用备用模型"""
        try:
            return await self.primary.get_response(**kwargs)
        except Exception as e:
            logger.warning(f"Primary model failed: {e}, using fallback")
            return await self.fallback.get_response(**kwargs)
```

Models 模块通过清晰的接口抽象和灵活的提供商机制，为 OpenAI Agents 提供了强大的模型集成能力，支持多种LLM的无缝切换和扩展。

