# LangChain-09-Memory-API

## 文档说明

本文档详细描述 **Memory 模块**的对外 API，包括对话记忆、缓冲区管理、向量存储记忆、摘要记忆等核心接口的所有公开方法和参数规格。

---

## 1. BaseMemory 核心 API

### 1.1 基础接口

#### 基本信息
- **类名**：`BaseMemory`
- **功能**：所有记忆系统的抽象基类
- **核心职责**：存储对话历史、管理上下文、提供记忆检索

#### 核心方法

```python
class BaseMemory(Serializable, ABC):
    """记忆基类。"""

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """返回记忆变量列表。"""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载记忆变量。"""

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """保存对话上下文。"""

    def clear(self) -> None:
        """清除记忆内容。"""
        pass
```

**方法详解**：

| 方法 | 参数 | 返回类型 | 说明 |
|-----|------|---------|------|
| memory_variables | 属性 | `List[str]` | 记忆系统提供的变量名列表 |
| load_memory_variables | `inputs: Dict[str, Any]` | `Dict[str, Any]` | 根据输入加载相关记忆 |
| save_context | `inputs: Dict`, `outputs: Dict` | `None` | 保存对话轮次到记忆 |
| clear | 无 | `None` | 清空所有记忆内容 |

---

## 2. ConversationBufferMemory API

### 2.1 基础缓冲区记忆

#### 基本信息
- **功能**：存储完整的对话历史
- **特点**：简单直接，保留所有对话内容
- **适用场景**：短对话、需要完整上下文的场景

#### 构造参数

```python
class ConversationBufferMemory(BaseChatMemory):
    def __init__(
        self,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        **kwargs: Any
    ):
        """对话缓冲区记忆构造函数。"""
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| human_prefix | `str` | `"Human"` | 人类消息前缀 |
| ai_prefix | `str` | `"AI"` | AI消息前缀 |
| memory_key | `str` | `"history"` | 记忆变量的键名 |
| return_messages | `bool` | `False` | 是否返回消息对象而非字符串 |
| input_key | `str` | `None` | 指定输入键（多输入时使用） |
| output_key | `str` | `None` | 指定输出键（多输出时使用） |

#### 使用示例

```python
from langchain.memory import ConversationBufferMemory

# 创建缓冲区记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 保存对话
memory.save_context(
    {"input": "你好，我是小明"},
    {"output": "你好小明！很高兴认识你。"}
)

memory.save_context(
    {"input": "今天天气怎么样？"},
    {"output": "今天天气很好，阳光明媚。"}
)

# 加载记忆
memory_vars = memory.load_memory_variables({})
print(memory_vars["chat_history"])
# [
#   HumanMessage(content="你好，我是小明"),
#   AIMessage(content="你好小明！很高兴认识你。"),
#   HumanMessage(content="今天天气怎么样？"),
#   AIMessage(content="今天天气很好，阳光明媚。")
# ]
```

#### 核心实现

```python
def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """加载记忆变量。"""
    if self.return_messages:
        # 返回消息对象列表
        return {self.memory_key: self.chat_memory.messages}
    else:
        # 返回格式化的字符串
        buffer = self.buffer
        return {self.memory_key: buffer}

def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
    """保存对话上下文。"""
    input_str = inputs[self.input_key or list(inputs.keys())[0]]
    output_str = outputs[self.output_key or list(outputs.keys())[0]]

    # 添加到聊天记忆
    self.chat_memory.add_user_message(input_str)
    self.chat_memory.add_ai_message(output_str)

@property
def buffer(self) -> str:
    """获取格式化的缓冲区内容。"""
    return get_buffer_string(
        self.chat_memory.messages,
        human_prefix=self.human_prefix,
        ai_prefix=self.ai_prefix
    )
```

---

## 3. ConversationBufferWindowMemory API

### 3.1 窗口缓冲区记忆

#### 基本信息
- **功能**：只保留最近的K轮对话
- **特点**：固定窗口大小，自动淘汰旧对话
- **适用场景**：长对话、内存限制场景

#### 构造参数

```python
class ConversationBufferWindowMemory(BaseChatMemory):
    def __init__(
        self,
        k: int = 5,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False,
        **kwargs: Any
    ):
        """窗口缓冲区记忆构造函数。"""
        super().__init__(**kwargs)
        self.k = k  # 窗口大小
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
```

#### 使用示例

```python
from langchain.memory import ConversationBufferWindowMemory

# 创建窗口记忆（只保留最近3轮对话）
memory = ConversationBufferWindowMemory(
    k=3,  # 保留3轮对话（6条消息）
    memory_key="chat_history",
    return_messages=True
)

# 添加多轮对话
conversations = [
    ("第1轮：你好", "你好！"),
    ("第2轮：天气", "今天天气很好"),
    ("第3轮：时间", "现在是下午3点"),
    ("第4轮：计划", "我们来制定计划吧"),
    ("第5轮：总结", "让我总结一下")
]

for human_msg, ai_msg in conversations:
    memory.save_context({"input": human_msg}, {"output": ai_msg})

# 检查记忆内容（只有最近3轮）
memory_vars = memory.load_memory_variables({})
messages = memory_vars["chat_history"]
print(f"保留的消息数量: {len(messages)}")  # 6条消息（3轮对话）

for msg in messages:
    print(f"{msg.__class__.__name__}: {msg.content}")
# HumanMessage: 第3轮：时间
# AIMessage: 现在是下午3点
# HumanMessage: 第4轮：计划
# AIMessage: 我们来制定计划吧
# HumanMessage: 第5轮：总结
# AIMessage: 让我总结一下
```

#### 窗口管理实现

```python
def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
    """保存上下文并维护窗口大小。"""
    # 添加新消息
    super().save_context(inputs, outputs)

    # 维护窗口大小
    self._prune_messages()

def _prune_messages(self) -> None:
    """修剪消息以维护窗口大小。"""
    messages = self.chat_memory.messages

    # 计算应保留的消息数量（k轮对话 = 2*k条消息）
    max_messages = 2 * self.k

    if len(messages) > max_messages:
        # 只保留最新的消息
        self.chat_memory.messages = messages[-max_messages:]

@property
def buffer(self) -> str:
    """获取窗口内的缓冲区内容。"""
    return get_buffer_string(
        self.chat_memory.messages,
        human_prefix=self.human_prefix,
        ai_prefix=self.ai_prefix
    )
```

---

## 4. ConversationSummaryMemory API

### 4.1 摘要记忆

#### 基本信息
- **功能**：将对话历史压缩为摘要
- **特点**：节省内存，保留关键信息
- **适用场景**：长期对话、内存敏感应用

#### 构造参数

```python
class ConversationSummaryMemory(BaseChatMemory):
    def __init__(
        self,
        llm: BaseLanguageModel,
        memory_key: str = "history",
        return_messages: bool = False,
        buffer: str = "",
        prompt: BasePromptTemplate = SUMMARY_PROMPT,
        **kwargs: Any
    ):
        """摘要记忆构造函数。"""
        super().__init__(**kwargs)
        self.llm = llm
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.buffer = buffer
        self.prompt = prompt
```

#### 使用示例

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAI

# 创建摘要记忆
llm = OpenAI(temperature=0)
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_summary"
)

# 添加对话历史
memory.save_context(
    {"input": "我想了解机器学习"},
    {"output": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。"}
)

memory.save_context(
    {"input": "有哪些主要的机器学习算法？"},
    {"output": "主要包括监督学习（如决策树、随机森林）、无监督学习（如聚类、降维）和强化学习。"}
)

memory.save_context(
    {"input": "监督学习和无监督学习的区别是什么？"},
    {"output": "监督学习使用标注数据训练模型，无监督学习从未标注数据中发现模式。"}
)

# 获取摘要
memory_vars = memory.load_memory_variables({})
print(memory_vars["chat_summary"])
# "用户询问了机器学习的基础概念，AI解释了机器学习的定义、主要算法分类，
#  以及监督学习和无监督学习的区别。对话涵盖了机器学习的核心概念。"
```

#### 摘要生成实现

```python
def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
    """保存上下文并更新摘要。"""
    # 添加新消息到临时缓冲区
    super().save_context(inputs, outputs)

    # 生成新的摘要
    self._update_summary()

def _update_summary(self) -> None:
    """更新对话摘要。"""
    messages = self.chat_memory.messages

    if len(messages) >= 2:  # 至少有一轮对话
        # 构建摘要提示
        new_lines = get_buffer_string(messages)

        if self.buffer:
            # 有现有摘要，进行增量更新
            prompt_input = {
                "summary": self.buffer,
                "new_lines": new_lines
            }
            prompt = self.prompt
        else:
            # 首次生成摘要
            prompt_input = {"new_lines": new_lines}
            prompt = SUMMARY_PROMPT

        # 调用LLM生成摘要
        self.buffer = self.llm.predict(prompt.format(**prompt_input))

        # 清空消息缓冲区（已经摘要化）
        self.chat_memory.clear()

def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """加载摘要记忆。"""
    if self.return_messages:
        # 将摘要转换为系统消息
        if self.buffer:
            return {self.memory_key: [SystemMessage(content=self.buffer)]}
        else:
            return {self.memory_key: []}
    else:
        return {self.memory_key: self.buffer}
```

#### 默认摘要提示

```python
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template="""
请简洁地总结以下对话内容，保留关键信息：

现有摘要：
{summary}

新的对话内容：
{new_lines}

新的摘要：
""".strip()
)
```

---

## 5. ConversationTokenBufferMemory API

### 5.1 令牌缓冲区记忆

#### 基本信息
- **功能**：基于令牌数量限制记忆大小
- **特点**：精确控制记忆的令牌消耗
- **适用场景**：API成本敏感、有严格令牌限制的场景

#### 构造参数

```python
class ConversationTokenBufferMemory(BaseChatMemory):
    def __init__(
        self,
        llm: BaseLanguageModel,
        max_token_limit: int = 2000,
        return_messages: bool = False,
        memory_key: str = "history",
        **kwargs: Any
    ):
        """令牌缓冲区记忆构造函数。"""
        super().__init__(**kwargs)
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.return_messages = return_messages
        self.memory_key = memory_key
```

#### 使用示例

```python
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import OpenAI

# 创建令牌限制记忆
llm = OpenAI()
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=100,  # 限制100个令牌
    memory_key="chat_history"
)

# 添加对话（会自动管理令牌数量）
memory.save_context(
    {"input": "请详细介绍一下深度学习的发展历史和主要里程碑"},
    {"output": "深度学习起源于1940年代的感知机概念，经历了多次起伏..."}
)

# 检查当前令牌使用情况
current_tokens = memory._get_current_token_count()
print(f"当前令牌数: {current_tokens}/{memory.max_token_limit}")

# 继续添加对话，超出限制时会自动删除旧消息
memory.save_context(
    {"input": "深度学习有哪些主要应用领域？"},
    {"output": "深度学习在计算机视觉、自然语言处理、语音识别等领域有广泛应用..."}
)
```

#### 令牌管理实现

```python
def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
    """保存上下文并管理令牌限制。"""
    # 添加新消息
    super().save_context(inputs, outputs)

    # 修剪消息以符合令牌限制
    self._prune_messages_to_token_limit()

def _prune_messages_to_token_limit(self) -> None:
    """修剪消息以满足令牌限制。"""
    while self._get_current_token_count() > self.max_token_limit:
        if len(self.chat_memory.messages) <= 2:
            # 至少保留一轮对话
            break

        # 删除最旧的消息对（人类+AI）
        self.chat_memory.messages = self.chat_memory.messages[2:]

def _get_current_token_count(self) -> int:
    """计算当前消息的令牌数量。"""
    buffer = get_buffer_string(self.chat_memory.messages)
    return self.llm.get_num_tokens(buffer)

@property
def buffer(self) -> str:
    """获取当前缓冲区内容。"""
    return get_buffer_string(self.chat_memory.messages)
```

---

## 6. VectorStoreRetrieverMemory API

### 6.1 向量存储记忆

#### 基本信息
- **功能**：使用向量存储进行语义记忆检索
- **特点**：基于相似性检索相关历史
- **适用场景**：长期记忆、语义相关的上下文检索

#### 构造参数

```python
class VectorStoreRetrieverMemory(BaseMemory):
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        memory_key: str = "history",
        input_key: Optional[str] = None,
        return_docs: bool = False,
        **kwargs: Any
    ):
        """向量存储检索记忆构造函数。"""
        self.retriever = retriever
        self.memory_key = memory_key
        self.input_key = input_key
        self.return_docs = return_docs
```

#### 使用示例

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 创建向量存储记忆
memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="relevant_history"
)

# 保存对话历史
memory.save_context(
    {"input": "我对机器学习很感兴趣"},
    {"output": "机器学习是一个很有前景的领域，建议从基础算法开始学习"}
)

memory.save_context(
    {"input": "Python有哪些机器学习库？"},
    {"output": "主要有scikit-learn、TensorFlow、PyTorch等优秀的库"}
)

memory.save_context(
    {"input": "今天天气真好"},
    {"output": "是的，适合出去散步"}
)

# 基于查询检索相关历史
relevant_memory = memory.load_memory_variables(
    {"input": "推荐一些深度学习资源"}
)
print(relevant_memory["relevant_history"])
# 会返回与"深度学习"相关的历史对话，如机器学习和Python库的讨论
```

#### 向量检索实现

```python
def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
    """保存对话到向量存储。"""
    input_str = inputs[self.input_key or list(inputs.keys())[0]]
    output_str = outputs[list(outputs.keys())[0]]

    # 构建文档内容
    document_content = f"Human: {input_str}\nAI: {output_str}"

    # 添加到向量存储
    self.retriever.vectorstore.add_texts(
        texts=[document_content],
        metadatas=[{
            "input": input_str,
            "output": output_str,
            "timestamp": time.time()
        }]
    )

def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """基于输入检索相关记忆。"""
    query = inputs[self.input_key or list(inputs.keys())[0]]

    # 检索相关文档
    docs = self.retriever.get_relevant_documents(query)

    if self.return_docs:
        return {self.memory_key: docs}
    else:
        # 格式化为字符串
        memory_content = "\n\n".join([doc.page_content for doc in docs])
        return {self.memory_key: memory_content}

@property
def memory_variables(self) -> List[str]:
    """返回记忆变量列表。"""
    return [self.memory_key]

def clear(self) -> None:
    """清除向量存储中的所有记忆。"""
    # 注意：这会删除向量存储中的所有文档
    if hasattr(self.retriever.vectorstore, 'delete'):
        self.retriever.vectorstore.delete()
```

---

## 7. ConversationEntityMemory API

### 7.1 实体记忆

#### 基本信息
- **功能**：提取和记住对话中的实体信息
- **特点**：结构化存储实体及其属性
- **适用场景**：需要记住人物、地点、事件等实体信息的对话

#### 构造参数

```python
class ConversationEntityMemory(BaseChatMemory):
    def __init__(
        self,
        llm: BaseLanguageModel,
        entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT,
        entity_summarization_prompt: BasePromptTemplate = ENTITY_SUMMARIZATION_PROMPT,
        entity_cache: Optional[List[str]] = None,
        k: int = 3,
        memory_key: str = "entities",
        **kwargs: Any
    ):
        """实体记忆构造函数。"""
        super().__init__(**kwargs)
        self.llm = llm
        self.entity_extraction_prompt = entity_extraction_prompt
        self.entity_summarization_prompt = entity_summarization_prompt
        self.entity_cache = entity_cache or []
        self.k = k  # 返回的相关实体数量
        self.memory_key = memory_key
        self.entity_store: Dict[str, str] = {}  # 实体存储
```

#### 使用示例

```python
from langchain.memory import ConversationEntityMemory
from langchain_openai import OpenAI

# 创建实体记忆
llm = OpenAI(temperature=0)
memory = ConversationEntityMemory(
    llm=llm,
    memory_key="entity_info"
)

# 保存包含实体的对话
memory.save_context(
    {"input": "我叫张三，今年30岁，住在北京，在阿里巴巴工作"},
    {"output": "很高兴认识你张三！你在阿里巴巴做什么工作呢？"}
)

memory.save_context(
    {"input": "我是一名软件工程师，主要负责后端开发"},
    {"output": "软件工程师是个很有前景的职业，后端开发需要掌握哪些技术呢？"}
)

# 检索与特定输入相关的实体信息
entity_info = memory.load_memory_variables(
    {"input": "张三的工作经历如何？"}
)
print(entity_info["entity_info"])
# "张三: 30岁，住在北京，在阿里巴巴担任软件工程师，负责后端开发"
```

#### 实体提取和管理

```python
def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
    """保存上下文并提取实体。"""
    # 保存到聊天记忆
    super().save_context(inputs, outputs)

    # 提取新实体
    input_str = inputs[list(inputs.keys())[0]]
    output_str = outputs[list(outputs.keys())[0]]

    # 从输入和输出中提取实体
    text = f"{input_str}\n{output_str}"
    entities = self._extract_entities(text)

    # 更新实体存储
    for entity in entities:
        self._update_entity_info(entity, text)

def _extract_entities(self, text: str) -> List[str]:
    """从文本中提取实体。"""
    prompt = self.entity_extraction_prompt.format(text=text)
    result = self.llm.predict(prompt)

    # 解析LLM返回的实体列表
    entities = [entity.strip() for entity in result.split(',') if entity.strip()]
    return entities

def _update_entity_info(self, entity: str, context: str) -> None:
    """更新实体信息。"""
    if entity in self.entity_store:
        # 更新现有实体信息
        existing_info = self.entity_store[entity]
        prompt = self.entity_summarization_prompt.format(
            entity=entity,
            existing_info=existing_info,
            new_context=context
        )
        updated_info = self.llm.predict(prompt)
        self.entity_store[entity] = updated_info
    else:
        # 创建新实体信息
        prompt = f"根据以下上下文，总结关于{entity}的信息：\n{context}"
        entity_info = self.llm.predict(prompt)
        self.entity_store[entity] = entity_info

def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """加载相关实体信息。"""
    input_str = inputs[list(inputs.keys())[0]]

    # 从输入中提取实体
    relevant_entities = self._extract_entities(input_str)

    # 获取相关实体信息
    entity_summaries = []
    for entity in relevant_entities[:self.k]:
        if entity in self.entity_store:
            entity_summaries.append(f"{entity}: {self.entity_store[entity]}")

    return {self.memory_key: "\n".join(entity_summaries)}
```

---

## 8. 组合记忆 API

### 8.1 CombinedMemory

#### 基本信息
- **功能**：组合多种记忆类型
- **特点**：同时使用多个记忆系统
- **适用场景**：需要不同类型记忆互补的复杂应用

#### 使用示例

```python
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
    VectorStoreRetrieverMemory
)

# 创建多个记忆组件
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="conversation_summary"
)

vector_memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="relevant_context"
)

# 组合记忆
combined_memory = CombinedMemory(
    memories=[buffer_memory, summary_memory, vector_memory]
)

# 使用组合记忆
combined_memory.save_context(
    {"input": "我想学习深度学习"},
    {"output": "深度学习是机器学习的一个重要分支..."}
)

# 获取所有记忆类型的信息
all_memory = combined_memory.load_memory_variables({
    "input": "有什么深度学习的学习建议吗？"
})

print("聊天历史:", all_memory["chat_history"])
print("对话摘要:", all_memory["conversation_summary"])
print("相关上下文:", all_memory["relevant_context"])
```

#### 组合记忆实现

```python
class CombinedMemory(BaseMemory):
    """组合多个记忆系统。"""

    def __init__(self, memories: List[BaseMemory]):
        self.memories = memories

    @property
    def memory_variables(self) -> List[str]:
        """返回所有记忆的变量列表。"""
        variables = []
        for memory in self.memories:
            variables.extend(memory.memory_variables)
        return variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载所有记忆的变量。"""
        memory_data = {}
        for memory in self.memories:
            memory_vars = memory.load_memory_variables(inputs)
            memory_data.update(memory_vars)
        return memory_data

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """保存上下文到所有记忆。"""
        for memory in self.memories:
            memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """清除所有记忆。"""
        for memory in self.memories:
            memory.clear()
```

---

## 9. 记忆管理工具 API

### 9.1 记忆统计和监控

```python
class MemoryManager:
    """记忆管理器。"""

    def __init__(self, memory: BaseMemory):
        self.memory = memory
        self.stats = {
            "total_contexts_saved": 0,
            "total_memory_loads": 0,
            "memory_size_bytes": 0,
            "last_accessed": None
        }

    def save_context_with_stats(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """保存上下文并更新统计。"""
        start_time = time.time()

        # 保存上下文
        self.memory.save_context(inputs, outputs)

        # 更新统计
        self.stats["total_contexts_saved"] += 1
        self.stats["last_accessed"] = time.time()
        self.stats["save_time"] = time.time() - start_time

        # 估算内存大小
        self._update_memory_size()

    def load_memory_with_stats(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """加载记忆并更新统计。"""
        start_time = time.time()

        # 加载记忆
        memory_vars = self.memory.load_memory_variables(inputs)

        # 更新统计
        self.stats["total_memory_loads"] += 1
        self.stats["last_accessed"] = time.time()
        self.stats["load_time"] = time.time() - start_time

        return memory_vars

    def _update_memory_size(self) -> None:
        """更新记忆大小估算。"""
        if hasattr(self.memory, 'buffer'):
            self.stats["memory_size_bytes"] = len(self.memory.buffer.encode('utf-8'))
        elif hasattr(self.memory, 'chat_memory'):
            total_size = 0
            for msg in self.memory.chat_memory.messages:
                total_size += len(msg.content.encode('utf-8'))
            self.stats["memory_size_bytes"] = total_size

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息。"""
        return {
            **self.stats,
            "memory_type": type(self.memory).__name__,
            "memory_variables": self.memory.memory_variables
        }

    def optimize_memory(self) -> None:
        """优化记忆性能。"""
        if isinstance(self.memory, ConversationBufferMemory):
            # 检查是否需要转换为窗口记忆
            if hasattr(self.memory, 'chat_memory'):
                message_count = len(self.memory.chat_memory.messages)
                if message_count > 100:  # 消息过多
                    print("建议使用ConversationBufferWindowMemory以提高性能")

        elif isinstance(self.memory, ConversationSummaryMemory):
            # 检查摘要是否过长
            if len(self.memory.buffer) > 2000:
                print("摘要过长，建议重新生成或分段摘要")
```

---

## 10. 最佳实践与配置

### 10.1 记忆类型选择指南

| 场景 | 推荐记忆类型 | 配置建议 |
|-----|-------------|---------|
| 短对话 | `ConversationBufferMemory` | 简单直接，保留完整历史 |
| 长对话 | `ConversationBufferWindowMemory` | k=5-10，平衡性能和上下文 |
| 成本敏感 | `ConversationTokenBufferMemory` | 根据模型定价设置token限制 |
| 长期记忆 | `ConversationSummaryMemory` | 使用高质量LLM生成摘要 |
| 语义检索 | `VectorStoreRetrieverMemory` | 选择合适的embedding模型 |
| 实体追踪 | `ConversationEntityMemory` | 适用于客服、个人助手等场景 |
| 复杂应用 | `CombinedMemory` | 组合多种记忆类型 |

### 10.2 性能优化配置

```python
def create_optimized_memory(
    conversation_length: str,
    cost_sensitivity: str,
    semantic_search: bool = False
) -> BaseMemory:
    """根据需求创建优化的记忆配置。"""

    if conversation_length == "short" and cost_sensitivity == "low":
        # 短对话，成本不敏感
        return ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )

    elif conversation_length == "long" and cost_sensitivity == "high":
        # 长对话，成本敏感
        return ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=1000,
            memory_key="history"
        )

    elif semantic_search:
        # 需要语义检索
        return VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="relevant_history"
        )

    else:
        # 默认配置：窗口记忆
        return ConversationBufferWindowMemory(
            k=5,
            memory_key="history",
            return_messages=True
        )

# 使用示例
memory = create_optimized_memory(
    conversation_length="long",
    cost_sensitivity="high",
    semantic_search=False
)
```

---

## 11. 总结

本文档详细描述了 **Memory 模块**的核心 API：

### 主要记忆类型
1. **ConversationBufferMemory**：完整对话历史存储
2. **ConversationBufferWindowMemory**：固定窗口大小的记忆
3. **ConversationSummaryMemory**：基于LLM的对话摘要
4. **ConversationTokenBufferMemory**：基于令牌限制的记忆
5. **VectorStoreRetrieverMemory**：基于向量检索的语义记忆
6. **ConversationEntityMemory**：实体提取和追踪记忆

### 核心功能
1. **上下文管理**：save_context和load_memory_variables
2. **记忆检索**：基于输入检索相关历史信息
3. **内存优化**：不同策略的内存使用优化
4. **组合使用**：CombinedMemory支持多种记忆类型组合

每个 API 均包含：
- 完整的构造参数和配置选项
- 详细的使用示例和最佳实践
- 核心实现逻辑和算法说明
- 性能优化建议和选择指南

Memory 模块是构建有状态对话系统的关键组件，正确选择和配置记忆类型对提高对话质量和系统性能至关重要。
