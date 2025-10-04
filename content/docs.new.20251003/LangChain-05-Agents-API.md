# LangChain-05-Agents-API

## 文档说明

本文档详细描述 **Agents 模块**的对外 API，包括 `AgentExecutor`、各种Agent类型、工具调用、推理循环等核心接口的所有公开方法、参数规格和最佳实践。

---

## 1. AgentExecutor 核心 API

### 1.1 创建 AgentExecutor

#### 基本信息
- **类名**：`AgentExecutor`
- **功能**：代理执行器，管理Agent的推理-行动循环
- **核心职责**：工具调用、步骤管理、错误处理、结果收集

#### 构造参数

```python
class AgentExecutor(Chain):
    def __init__(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent, Runnable],
        tools: Sequence[BaseTool],
        return_intermediate_steps: bool = False,
        max_iterations: Optional[int] = 15,
        max_execution_time: Optional[float] = None,
        early_stopping_method: str = "force",
        handle_parsing_errors: Union[bool, str, Callable[[OutputParserException], str]] = False,
        trim_intermediate_steps: Union[int, Callable[[List[Tuple[AgentAction, str]]], List[Tuple[AgentAction, str]]]] = -1,
        **kwargs: Any,
    ):
        """代理执行器构造函数。"""
```

**参数说明**：

| 参数 | 类型 | 必填 | 默认 | 说明 |
|-----|------|-----|------|------|
| agent | `BaseSingleActionAgent \| BaseMultiActionAgent \| Runnable` | 是 | - | 代理实例 |
| tools | `Sequence[BaseTool]` | 是 | - | 可用工具列表 |
| return_intermediate_steps | `bool` | 否 | `False` | 是否返回中间步骤 |
| max_iterations | `int` | 否 | `15` | 最大迭代次数 |
| max_execution_time | `float` | 否 | `None` | 最大执行时间（秒） |
| early_stopping_method | `str` | 否 | `"force"` | 早停策略：`"force"` 或 `"generate"` |
| handle_parsing_errors | `Union[bool, str, Callable]` | 否 | `False` | 解析错误处理策略 |
| trim_intermediate_steps | `Union[int, Callable]` | 否 | `-1` | 中间步骤修剪策略 |

#### 使用示例

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# 创建工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气信息。"""
    return f"{city}的天气是晴天，温度25°C"

@tool
def search_web(query: str) -> str:
    """搜索网页信息。"""
    return f"关于'{query}'的搜索结果..."

tools = [get_weather, search_web]

# 创建模型
model = ChatOpenAI(model="gpt-4", temperature=0)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 创建Agent
agent = create_openai_tools_agent(model, tools, prompt)

# 创建AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    max_iterations=10,
    max_execution_time=60.0
)
```

---

### 1.2 invoke - 同步执行

#### 基本信息
- **方法签名**：`invoke(inputs: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]`
- **功能**：执行代理推理-行动循环，返回最终结果
- **执行模式**：同步阻塞

#### 请求参数

```python
def invoke(
    self,
    inputs: Dict[str, Any],
    config: Optional[RunnableConfig] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """同步执行代理。"""
```

**输入格式**：

| 字段 | 类型 | 必填 | 说明 |
|-----|------|-----|------|
| input | `str` | 是 | 用户问题或任务描述 |
| chat_history | `List[BaseMessage]` | 否 | 聊天历史（可选） |
| intermediate_steps | `List[Tuple[AgentAction, str]]` | 否 | 之前的中间步骤 |

#### 响应结构

```python
# 返回字典结构
{
    "input": str,                    # 原始输入
    "output": str,                   # 最终输出
    "intermediate_steps": List[      # 中间步骤（如果启用）
        Tuple[AgentAction, str]      # (动作, 观察结果)
    ]
}
```

#### 核心执行流程

```python
def invoke(self, inputs: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """执行代理推理循环。"""
    # 1. 初始化
    inputs = self.prep_inputs(inputs)
    intermediate_steps: List[Tuple[AgentAction, str]] = []
    iterations = 0
    time_elapsed = 0.0
    start_time = time.time()
    
    # 2. 推理-行动循环
    while self._should_continue(iterations, time_elapsed):
        # 构建Agent输入
        agent_inputs = self._construct_scratchpad(intermediate_steps)
        agent_inputs.update(inputs)
        
        # Agent推理
        output = self.agent.plan(
            intermediate_steps=intermediate_steps,
            **agent_inputs
        )
        
        # 检查是否完成
        if isinstance(output, AgentFinish):
            return self._return(
                output, 
                intermediate_steps if self.return_intermediate_steps else []
            )
        
        # 执行动作
        if isinstance(output, AgentAction):
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = self._take_action(output, **tool_run_kwargs)
            intermediate_steps.append((output, observation))
        
        iterations += 1
        time_elapsed = time.time() - start_time
    
    # 3. 达到限制时的处理
    return self._early_stopping_handler(iterations, time_elapsed, intermediate_steps)
```

#### 使用示例

```python
# 执行代理任务
result = agent_executor.invoke({
    "input": "北京的天气怎么样？如果天气好的话，推荐一些户外活动"
})

print("输入:", result["input"])
print("输出:", result["output"])

# 查看中间步骤
if "intermediate_steps" in result:
    for i, (action, observation) in enumerate(result["intermediate_steps"]):
        print(f"\n步骤 {i+1}:")
        print(f"  动作: {action.tool} - {action.tool_input}")
        print(f"  观察: {observation}")
```

---

### 1.3 stream - 流式执行

#### 基本信息
- **方法签名**：`stream(inputs: Dict[str, Any]) -> Iterator[Dict[str, Any]]`
- **功能**：流式执行代理，实时返回中间步骤
- **适用场景**：需要实时显示推理过程的应用

#### 使用示例

```python
# 流式执行
for chunk in agent_executor.stream({"input": "帮我查询天气并推荐活动"}):
    if "actions" in chunk:
        for action in chunk["actions"]:
            print(f"🤖 思考: 使用工具 {action.tool}")
            print(f"   参数: {action.tool_input}")
    
    if "steps" in chunk:
        for step in chunk["steps"]:
            print(f"📋 观察: {step.observation}")
    
    if "output" in chunk:
        print(f"✅ 最终答案: {chunk['output']}")
```

#### 流式输出格式

```python
# 流式输出的chunk格式
{
    "actions": [AgentAction],     # 当前步骤的动作
    "steps": [AgentStep],         # 完成的步骤
    "messages": [BaseMessage],    # 消息更新
    "output": str                 # 最终输出（最后一个chunk）
}
```

---

## 2. Agent创建函数 API

### 2.1 create_openai_tools_agent

#### 基本信息
- **功能**：创建使用OpenAI工具调用的Agent
- **适用场景**：GPT-4等支持工具调用的模型
- **优势**：结构化工具调用，准确性高

#### 方法签名

```python
def create_openai_tools_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    *,
    tools_renderer: ToolsRenderer = render_text_description,
    **kwargs: Any,
) -> Runnable[Union[Dict, BaseMessage], Union[AgentAction, AgentFinish]]:
    """创建OpenAI工具Agent。"""
```

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|-----|------|-----|------|
| llm | `BaseChatModel` | 是 | 支持工具调用的聊天模型 |
| tools | `Sequence[BaseTool]` | 是 | 可用工具列表 |
| prompt | `ChatPromptTemplate` | 是 | 提示模板 |
| tools_renderer | `ToolsRenderer` | 否 | 工具描述渲染器 |

#### 提示模板要求

```python
# 必须包含的占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # 必需：中间步骤占位符
])
```

#### 使用示例

```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 模型（必须支持工具调用）
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 工具
tools = [get_weather, search_web]

# 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to answer questions."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 创建Agent
agent = create_openai_tools_agent(llm, tools, prompt)

# 创建执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用
result = agent_executor.invoke({"input": "北京天气如何？"})
```

---

### 2.2 create_react_agent

#### 基本信息
- **功能**：创建ReAct（推理+行动）风格的Agent
- **适用场景**：不支持工具调用的模型，通过文本生成控制
- **特点**：使用思考-行动-观察的循环模式

#### 方法签名

```python
def create_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    *,
    tools_renderer: ToolsRenderer = render_text_description,
    stop_sequence: Optional[List[str]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """创建ReAct Agent。"""
```

#### ReAct提示模板示例

```python
from langchain import hub

# 使用Hub中的ReAct提示模板
prompt = hub.pull("hwchase17/react")

# 自定义ReAct提示模板
custom_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
```

#### 使用示例

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import OpenAI  # 注意：使用传统LLM

# 传统LLM模型
llm = OpenAI(temperature=0)

# 创建ReAct Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用
result = agent_executor.invoke({"input": "What's the weather in Beijing?"})
```

---

### 2.3 create_structured_chat_agent

#### 基本信息
- **功能**：创建结构化聊天Agent，支持复杂输入格式
- **适用场景**：需要结构化工具输入的复杂任务
- **特点**：JSON格式的工具调用

#### 使用示例

```python
from langchain.agents import create_structured_chat_agent

# 结构化工具示例
@tool
def complex_search(query: str, filters: dict, max_results: int = 10) -> str:
    """复杂搜索工具，支持结构化参数。"""
    return f"搜索'{query}'，过滤器：{filters}，结果数：{max_results}"

# 创建结构化Agent
agent = create_structured_chat_agent(llm, tools, prompt)
```

---

## 3. Agent动作数据结构 API

### 3.1 AgentAction

#### 基本信息
- **功能**：表示Agent决定执行的动作
- **用途**：工具调用的标准化表示

#### 数据结构

```python
class AgentAction(NamedTuple):
    """Agent动作。"""
    tool: str                    # 工具名称
    tool_input: Union[str, Dict] # 工具输入
    log: str                     # 推理日志
    
    # 可选字段
    message_log: List[BaseMessage] = []  # 消息日志
    tool_call_id: Optional[str] = None   # 工具调用ID
```

#### 创建示例

```python
# 创建Agent动作
action = AgentAction(
    tool="get_weather",
    tool_input={"city": "Beijing"},
    log="我需要查询北京的天气信息"
)

# 访问字段
print(f"工具: {action.tool}")
print(f"输入: {action.tool_input}")
print(f"推理: {action.log}")
```

---

### 3.2 AgentFinish

#### 基本信息
- **功能**：表示Agent完成任务的最终结果
- **用途**：推理循环的终止条件

#### 数据结构

```python
class AgentFinish(NamedTuple):
    """Agent完成。"""
    return_values: Dict[str, Any]  # 返回值
    log: str                       # 推理日志
    
    # 可选字段
    message_log: List[BaseMessage] = []  # 消息日志
```

#### 使用示例

```python
# Agent完成
finish = AgentFinish(
    return_values={"output": "北京今天天气晴朗，温度25°C，适合户外活动"},
    log="我已经获取了天气信息并给出了建议"
)
```

---

### 3.3 AgentStep

#### 基本信息
- **功能**：表示Agent执行的完整步骤（动作+观察）
- **用途**：记录推理过程的历史

#### 数据结构

```python
class AgentStep(NamedTuple):
    """Agent步骤。"""
    action: AgentAction  # 执行的动作
    observation: str     # 观察结果
```

---

## 4. 错误处理 API

### 4.1 解析错误处理

#### 配置选项

```python
# 1. 忽略解析错误
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=False  # 抛出异常
)

# 2. 返回默认错误消息
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True  # 返回 "Invalid Format"
)

# 3. 自定义错误消息
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors="解析失败，请重新格式化输出"
)

# 4. 自定义错误处理函数
def custom_error_handler(error: OutputParserException) -> str:
    return f"解析错误: {error}，请使用正确的格式"

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=custom_error_handler
)
```

---

### 4.2 工具调用错误处理

#### 工具错误处理策略

```python
# 在工具中使用ToolException
from langchain_core.tools import ToolException

@tool
def risky_tool(input_data: str) -> str:
    """可能出错的工具。"""
    if not input_data:
        raise ToolException("输入不能为空")
    
    try:
        result = process_data(input_data)
        return result
    except Exception as e:
        raise ToolException(f"处理失败: {e}")

# Agent会捕获ToolException并继续执行
```

---

## 5. 性能优化 API

### 5.1 中间步骤修剪

#### 配置选项

```python
# 1. 保留最后N个步骤
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    trim_intermediate_steps=5  # 只保留最后5个步骤
)

# 2. 自定义修剪函数
def custom_trim(steps: List[Tuple[AgentAction, str]]) -> List[Tuple[AgentAction, str]]:
    """保留重要步骤，移除冗余信息。"""
    important_steps = []
    for action, observation in steps:
        # 保留工具调用步骤
        if action.tool in ["search", "calculate"]:
            important_steps.append((action, observation[:200]))  # 截断观察
    return important_steps[-3:]  # 最多保留3个重要步骤

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    trim_intermediate_steps=custom_trim
)
```

---

### 5.2 早停策略

#### 配置选项

```python
# 强制停止（默认）
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    early_stopping_method="force",  # 达到限制时强制返回
    max_iterations=10
)

# 生成式停止
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    early_stopping_method="generate",  # 让Agent生成最终答案
    max_iterations=10
)
```

---

## 6. 高级用法 API

### 6.1 自定义Agent

#### 继承BaseSingleActionAgent

```python
from langchain.agents import BaseSingleActionAgent

class CustomAgent(BaseSingleActionAgent):
    """自定义Agent实现。"""
    
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.tool_names = [tool.name for tool in tools]
    
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """规划下一步动作。"""
        # 构建提示
        prompt = self._construct_prompt(intermediate_steps, **kwargs)
        
        # LLM推理
        response = self.llm.predict(prompt)
        
        # 解析响应
        return self._parse_response(response)
    
    def _construct_prompt(self, steps: List[Tuple[AgentAction, str]], **kwargs) -> str:
        """构建提示。"""
        # 自定义提示构建逻辑
        pass
    
    def _parse_response(self, response: str) -> Union[AgentAction, AgentFinish]:
        """解析LLM响应。"""
        # 自定义解析逻辑
        pass
    
    @property
    def input_keys(self) -> List[str]:
        return ["input"]
    
    @property
    def return_values(self) -> List[str]:
        return ["output"]
```

---

## 7. 总结

本文档详细描述了 **Agents 模块**的核心 API：

### 主要组件
1. **AgentExecutor**：代理执行器，管理推理循环
2. **Agent创建函数**：create_openai_tools_agent、create_react_agent等
3. **数据结构**：AgentAction、AgentFinish、AgentStep
4. **错误处理**：解析错误、工具错误的处理策略
5. **性能优化**：中间步骤修剪、早停策略

### 核心方法
1. **invoke/stream**：同步/流式执行代理
2. **工具调用**：结构化工具调用和错误处理
3. **推理循环**：思考-行动-观察的完整流程

每个 API 均包含：
- 完整的请求/响应结构
- 详细的参数说明和配置选项
- 实际使用示例和最佳实践
- 错误处理和性能优化建议

Agent系统是LangChain最复杂的模块之一，正确理解和使用这些API对构建智能代理应用至关重要。
