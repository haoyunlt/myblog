---
title: "LangChain 源码剖析 - 使用示例与最佳实践"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['源码分析', '技术文档', '最佳实践']
categories: ['技术分析']
description: "LangChain 源码剖析 - 使用示例与最佳实践的深入技术分析文档"
keywords: ['源码分析', '技术文档', '最佳实践']
author: "技术分析师"
weight: 1
---

## 1. 框架基础使用示例

### 1.1 简单的LLM调用

```python
"""
基本的语言模型调用示例
演示如何使用LangChain进行基础的LLM交互
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# 创建提示模板
# PromptTemplate是LangChain中用于结构化提示的核心类
# 它允许动态插入变量，确保提示的一致性和可重用性
prompt = PromptTemplate(
    input_variables=["topic"],  # 定义输入变量
    template="请简要解释 {topic} 的概念。"  # 模板字符串，{topic}会被替换
)

# 初始化聊天模型
# ChatOpenAI是BaseChatModel的实现，提供与OpenAI ChatGPT的接口
model = ChatOpenAI(
    model="gpt-3.5-turbo",  # 指定模型版本
    temperature=0.7,        # 控制输出的随机性 (0-1)
    max_tokens=150         # 限制生成的最大token数
)

# 创建处理链
# 使用LCEL (LangChain Expression Language) 语法将组件连接
# "|" 操作符创建RunnableSequence，实现数据的管道式处理
chain = prompt | model

# 执行调用
# invoke方法是Runnable接口的核心方法，执行整个处理链
result = chain.invoke({"topic": "机器学习"})
print(f"模型回复: {result.content}")

# 异步执行示例
import asyncio

async def async_example():
    """异步执行示例，适用于需要处理大量并发请求的场景"""
    result = await chain.ainvoke({"topic": "深度学习"})
    print(f"异步回复: {result.content}")

# 批量处理示例
topics = ["自然语言处理", "计算机视觉", "强化学习"]
# batch方法自动并行处理多个输入，提高处理效率
batch_results = chain.batch([{"topic": topic} for topic in topics])
for i, result in enumerate(batch_results):
    print(f"话题 {topics[i]}: {result.content}")
```

### 1.2 构建RAG（检索增强生成）系统

```python
"""
构建检索增强生成(RAG)系统的完整示例
展示了文档加载、向量存储、检索和生成的完整流程
"""
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 1. 准备文档数据
documents = [
    Document(
        page_content="LangChain是一个用于构建LLM应用的框架，提供了丰富的工具链。",
        metadata={"source": "langchain_intro.md", "page": 1}
    ),
    Document(
        page_content="Runnable是LangChain的核心接口，所有组件都实现了这个接口。",
        metadata={"source": "langchain_core.md", "page": 2}
    ),
    Document(
        page_content="向量存储用于存储和检索嵌入向量，支持相似性搜索。",
        metadata={"source": "vectorstore.md", "page": 3}
    )
]

# 2. 创建向量存储
# OpenAIEmbeddings将文本转换为向量表示
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# InMemoryVectorStore是一个内存中的向量存储实现
# 适用于小规模数据或原型开发
vectorstore = InMemoryVectorStore(embedding=embeddings)

# 添加文档到向量存储
# 这个过程会自动计算每个文档的嵌入向量并存储
document_ids = vectorstore.add_documents(documents)
print(f"添加了 {len(document_ids)} 个文档到向量存储")

# 3. 创建检索器
# as_retriever方法将向量存储转换为检索器
# 检索器实现了BaseRetriever接口，提供统一的检索API
retriever = vectorstore.as_retriever(
    search_type="similarity",    # 使用相似性搜索
    search_kwargs={"k": 2}      # 返回最相似的2个文档
)

# 4. 构建RAG提示模板
# ChatPromptTemplate用于构造聊天格式的提示
rag_prompt = ChatPromptTemplate.from_template("""
基于以下上下文回答问题:

上下文:
{context}

问题: {question}

请基于上下文提供准确的回答。如果上下文中没有相关信息，请明确说明。
""")

# 5. 创建模型
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 6. 构建RAG链
# RunnableParallel允许并行执行多个操作
# RunnablePassthrough传递原始输入不做修改
rag_chain = (
    RunnableParallel({
        "context": retriever,           # 检索相关文档
        "question": RunnablePassthrough()  # 直接传递问题
    })
    | rag_prompt                        # 构建提示
    | model                            # 生成回答
    | StrOutputParser()                # 提取文本输出
)

# 7. 测试RAG系统
question = "什么是Runnable接口？"
answer = rag_chain.invoke(question)
print(f"问题: {question}")
print(f"回答: {answer}")

# 8. 流式输出RAG结果
print("\n流式输出:")
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
print("\n")
```

### 1.3 构建Agent系统

```python
"""
构建智能Agent系统示例
展示了工具定义、Agent创建和执行的完整流程
"""
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import requests
import json

# 1. 定义工具函数
@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称，如"北京"、"上海"等
        
    Returns:
        包含温度、湿度、天气描述的JSON字符串
        
    Raises:
        Exception: 当API调用失败时抛出异常
    """
    # 模拟天气API调用
    weather_data = {
        "北京": {"temperature": "15°C", "humidity": "60%", "description": "晴天"},
        "上海": {"temperature": "18°C", "humidity": "70%", "description": "多云"},
        "深圳": {"temperature": "25°C", "humidity": "80%", "description": "小雨"}
    }
    
    if city in weather_data:
        return json.dumps(weather_data[city], ensure_ascii=False)
    else:
        return json.dumps({"error": f"暂无{city}的天气信息"}, ensure_ascii=False)

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式的值
    
    Args:
        expression: 数学表达式，如"2+3*4"、"10/2"等
        
    Returns:
        计算结果的字符串表示
        
    Raises:
        ValueError: 当表达式无效时抛出异常
    """
    try:
        # 安全的数学表达式计算
        # 注意：在生产环境中应该使用更安全的计算方法
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("表达式包含不允许的字符")
            
        result = eval(expression)  # 仅用于演示，生产环境应避免使用eval
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 2. 创建支持工具调用的模型
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1
)

# 3. 绑定工具到模型
# bind_tools方法将工具绑定到模型，使模型能够调用这些工具
tools = [get_weather, calculate]
model_with_tools = model.bind_tools(tools)

# 4. 定义Agent执行逻辑
def run_agent(query: str, max_iterations: int = 5):
    """
    运行Agent处理用户查询
    
    Args:
        query: 用户查询
        max_iterations: 最大迭代次数，防止无限循环
        
    Returns:
        最终的回答结果
    """
    messages = [
        SystemMessage(content="""你是一个有用的助手，可以获取天气信息和进行数学计算。
当用户询问天气时，使用get_weather工具获取信息。
当用户需要计算时，使用calculate工具进行计算。
请根据工具返回的结果给出友好的回答。"""),
        HumanMessage(content=query)
    ]
    
    iteration = 0
    while iteration < max_iterations:
        print(f"\n--- 迭代 {iteration + 1} ---")
        
        # 模型生成回应（可能包含工具调用）
        response = model_with_tools.invoke(messages)
        messages.append(response)
        
        print(f"模型回应: {response.content}")
        
        # 检查是否有工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # 执行工具调用
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_call_id = tool_call['id']
                
                print(f"调用工具: {tool_name}, 参数: {tool_args}")
                
                # 根据工具名称执行对应的工具
                if tool_name == "get_weather":
                    result = get_weather.invoke(tool_args['city'])
                elif tool_name == "calculate":
                    result = calculate.invoke(tool_args['expression'])
                else:
                    result = f"未知工具: {tool_name}"
                
                print(f"工具结果: {result}")
                
                # 将工具结果添加到消息历史
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call_id
                ))
        else:
            # 没有工具调用，返回最终答案
            return response.content
            
        iteration += 1
    
    return "达到最大迭代次数，无法完成任务"

# 5. 测试Agent
test_queries = [
    "北京今天天气怎么样？",
    "计算 15 * 8 + 25 的结果",
    "上海和深圳今天哪里天气更好？"
]

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"用户查询: {query}")
    answer = run_agent(query)
    print(f"最终回答: {answer}")
```

## 2. 高级使用模式

### 2.1 自定义Runnable组件

```python
"""
创建自定义Runnable组件的高级示例
展示如何扩展LangChain的核心功能
"""
from typing import Any, Dict, List, Optional
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.callbacks import CallbackManagerForChainRun
import time
import logging

class TextPreprocessor(Runnable[str, str]):
    """
    自定义文本预处理器
    实现Runnable接口，提供统一的调用方式
    """
    
    def __init__(self, operations: List[str] = None):
        """
        初始化预处理器
        
        Args:
            operations: 预处理操作列表，支持 'lowercase', 'strip', 'remove_punctuation'
        """
        super().__init__()
        self.operations = operations or ['lowercase', 'strip']
        
    def invoke(
        self, 
        input: str, 
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> str:
        """
        执行文本预处理
        
        Args:
            input: 输入文本
            config: 运行时配置
            **kwargs: 额外参数
            
        Returns:
            处理后的文本
        """
        # 获取回调管理器用于追踪执行过程
        callback_manager = None
        if config and config.get('callbacks'):
            from langchain_core.callbacks import CallbackManager
            callback_manager = CallbackManager(config['callbacks'])
        
        # 记录开始执行
        if callback_manager:
            callback_manager.on_chain_start(
                serialized={'name': self.__class__.__name__},
                inputs={'text': input},
                **kwargs
            )
        
        result = input
        
        # 执行预处理操作
        for operation in self.operations:
            if operation == 'lowercase':
                result = result.lower()
            elif operation == 'strip':
                result = result.strip()
            elif operation == 'remove_punctuation':
                import string
                result = result.translate(str.maketrans('', '', string.punctuation))
        
        # 记录执行完成
        if callback_manager:
            callback_manager.on_chain_end(
                outputs={'processed_text': result},
                **kwargs
            )
            
        return result

class CacheRunnable(Runnable[Any, Any]):
    """
    带缓存功能的Runnable包装器
    演示如何为现有组件添加缓存功能
    """
    
    def __init__(self, runnable: Runnable, cache_ttl: int = 300):
        """
        初始化缓存包装器
        
        Args:
            runnable: 被包装的Runnable组件
            cache_ttl: 缓存过期时间（秒）
        """
        super().__init__()
        self.runnable = runnable
        self.cache_ttl = cache_ttl
        self.cache = {}  # 简单的内存缓存
        
    def _get_cache_key(self, input: Any) -> str:
        """生成缓存键"""
        import hashlib
        input_str = str(input)
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def invoke(
        self, 
        input: Any, 
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Any:
        """
        带缓存的执行
        """
        cache_key = self._get_cache_key(input)
        current_time = time.time()
        
        # 检查缓存
        if cache_key in self.cache:
            cached_result, cache_time = self.cache[cache_key]
            if current_time - cache_time < self.cache_ttl:
                logging.info(f"缓存命中: {cache_key}")
                return cached_result
            else:
                # 缓存过期，删除旧缓存
                del self.cache[cache_key]
        
        # 执行实际计算
        result = self.runnable.invoke(input, config, **kwargs)
        
        # 存储到缓存
        self.cache[cache_key] = (result, current_time)
        logging.info(f"结果已缓存: {cache_key}")
        
        return result

# 使用示例
preprocessor = TextPreprocessor(['lowercase', 'strip', 'remove_punctuation'])
cached_preprocessor = CacheRunnable(preprocessor, cache_ttl=60)

# 创建处理链
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(
    template="分析以下文本的情感: {text}",
    input_variables=["text"]
)

# 组合自定义组件
analysis_chain = (
    {"text": cached_preprocessor}  # 预处理文本
    | prompt                       # 构建提示
    | model                       # 模型分析
)

# 测试
test_text = "  This is A GREAT day!!! "
result = analysis_chain.invoke(test_text)
print(f"原文: {test_text}")
print(f"分析结果: {result.content}")
```

### 2.2 错误处理和监控

```python
"""
错误处理和监控最佳实践示例
展示如何构建健壮的生产级应用
"""
from langchain_core.runnables import RunnableWithFallbacks, RunnableBranch
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
import logging
import traceback
from datetime import datetime

class MonitoringCallbackHandler(BaseCallbackHandler):
    """
    自定义监控回调处理器
    用于收集性能指标和错误信息
    """
    
    def __init__(self):
        """初始化监控处理器"""
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'total_duration': 0
        }
        self.errors = []
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM开始执行时的回调"""
        self.start_time = datetime.now()
        self.metrics['total_calls'] += 1
        logging.info(f"LLM调用开始: {serialized.get('name', 'Unknown')}")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """LLM执行完成时的回调"""
        if hasattr(self, 'start_time'):
            duration = (datetime.now() - self.start_time).total_seconds()
            self.metrics['total_duration'] += duration
            
        self.metrics['successful_calls'] += 1
        
        # 统计token使用量
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.metrics['total_tokens'] += token_usage.get('total_tokens', 0)
        
        logging.info(f"LLM调用成功完成，耗时: {duration:.2f}秒")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """LLM执行出错时的回调"""
        self.metrics['failed_calls'] += 1
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        self.errors.append(error_info)
        logging.error(f"LLM调用失败: {error}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        success_rate = (
            self.metrics['successful_calls'] / self.metrics['total_calls'] 
            if self.metrics['total_calls'] > 0 else 0
        )
        avg_duration = (
            self.metrics['total_duration'] / self.metrics['successful_calls']
            if self.metrics['successful_calls'] > 0 else 0
        )
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'average_duration': avg_duration,
            'recent_errors': self.errors[-5:]  # 最近5个错误
        }

# 创建健壮的处理链
def create_robust_chain():
    """
    创建带有错误处理和监控的健壮处理链
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    
    # 定义输出格式
    class SentimentAnalysis(BaseModel):
        """情感分析结果"""
        sentiment: str = Field(description="情感分类: positive, negative, neutral")
        confidence: float = Field(description="置信度 (0-1)")
        reasoning: str = Field(description="分析理由")
    
    # 创建解析器
    parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)
    
    # 主模型
    primary_model = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        timeout=30  # 设置超时
    )
    
    # 备用模型（更便宜但可能效果略差）
    fallback_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        timeout=30
    )
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的情感分析专家。请分析给定文本的情感倾向。"),
        ("human", "分析以下文本的情感:\n{text}\n\n{format_instructions}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    # 构建主处理链
    primary_chain = prompt | primary_model | parser
    
    # 构建备用链（简化版本）
    fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", "分析文本情感，回答 positive/negative/neutral"),
        ("human", "{text}")
    ])
    
    fallback_chain = fallback_prompt | fallback_model
    
    # 创建带错误恢复的链
    robust_chain = RunnableWithFallbacks(
        runnable=primary_chain,
        fallbacks=[fallback_chain],
        exception_to_check=(Exception,)  # 捕获所有异常
    )
    
    return robust_chain

# 使用示例
def run_with_monitoring():
    """运行带监控的处理链"""
    # 创建监控处理器
    monitor = MonitoringCallbackHandler()
    
    # 创建健壮的链
    chain = create_robust_chain()
    
    # 测试数据
    test_texts = [
        "我非常喜欢这个产品！",
        "这个服务真的很糟糕。",
        "还可以吧，没什么特别的。",
        "今天天气不错。",  # 中性文本
        "",  # 空文本，可能导致错误
    ]
    
    results = []
    
    for text in test_texts:
        try:
            print(f"\n分析文本: '{text}'")
            
            # 使用监控配置
            config = {
                "callbacks": [monitor]
            }
            
            result = chain.invoke({"text": text}, config=config)
            results.append({"text": text, "result": result, "success": True})
            print(f"结果: {result}")
            
        except Exception as e:
            results.append({"text": text, "error": str(e), "success": False})
            print(f"处理失败: {e}")
    
    # 输出监控指标
    print("\n" + "="*50)
    print("监控指标:")
    metrics = monitor.get_metrics()
    for key, value in metrics.items():
        if key != 'recent_errors':
            print(f"{key}: {value}")
    
    if metrics['recent_errors']:
        print("\n最近的错误:")
        for error in metrics['recent_errors']:
            print(f"- {error['timestamp']}: {error['error_type']} - {error['error_message']}")
    
    return results

# 运行监控示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_with_monitoring()
```

## 3. 生产环境最佳实践

### 3.1 配置管理

```python
"""
生产环境配置管理最佳实践
"""
import os
from typing import Optional
from pydantic import BaseSettings, Field

class LangChainConfig(BaseSettings):
    """
    LangChain应用配置类
    使用Pydantic进行配置验证和管理
    """
    
    # API密钥配置
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # 模型配置
    default_model: str = Field("gpt-3.5-turbo", env="DEFAULT_MODEL")
    max_tokens: int = Field(1000, env="MAX_TOKENS")
    temperature: float = Field(0.7, env="TEMPERATURE")
    
    # 性能配置
    max_concurrency: int = Field(5, env="MAX_CONCURRENCY")
    timeout_seconds: int = Field(30, env="TIMEOUT_SECONDS")
    max_retries: int = Field(3, env="MAX_RETRIES")
    
    # 缓存配置
    enable_cache: bool = Field(True, env="ENABLE_CACHE")
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    
    # 日志配置
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # 向量存储配置
    vector_store_type: str = Field("faiss", env="VECTOR_STORE_TYPE")
    vector_store_path: str = Field("./vector_store", env="VECTOR_STORE_PATH")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# 全局配置实例
config = LangChainConfig()

# 配置验证函数
def validate_config():
    """验证配置的有效性"""
    errors = []
    
    if not config.openai_api_key:
        errors.append("OpenAI API key is required")
    
    if config.temperature < 0 or config.temperature > 1:
        errors.append("Temperature must be between 0 and 1")
    
    if config.max_tokens <= 0:
        errors.append("Max tokens must be positive")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    print("Configuration validation passed")

# 使用配置创建组件
def create_configured_model():
    """使用配置创建模型实例"""
    from langchain_openai import ChatOpenAI
    
    return ChatOpenAI(
        openai_api_key=config.openai_api_key,
        model=config.default_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries
    )
```

### 3.2 性能优化策略

```python
"""
性能优化策略和技巧
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import asyncio
import time

class PerformanceOptimizer:
    """
    性能优化工具类
    """
    
    def __init__(self, max_workers: int = 5):
        """
        初始化优化器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_embedding(text: str) -> str:
        """
        缓存的嵌入计算
        使用LRU缓存避免重复计算相同文本的嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入结果的字符串表示
        """
        # 模拟嵌入计算
        time.sleep(0.1)  # 模拟API调用延迟
        return f"embedding_of_{hash(text)}"
    
    def batch_process_with_concurrency(self, items: list, process_func, batch_size: int = 10):
        """
        并发批处理
        
        Args:
            items: 要处理的项目列表
            process_func: 处理函数
            batch_size: 批次大小
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # 提交并发任务
            futures = {
                self.executor.submit(process_func, item): item 
                for item in batch
            }
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    print(f"处理失败 {futures[future]}: {e}")
                    results.append(None)
        
        return results
    
    async def async_batch_process(self, items: list, async_func, concurrency_limit: int = 5):
        """
        异步批处理
        
        Args:
            items: 要处理的项目列表
            async_func: 异步处理函数
            concurrency_limit: 并发限制
            
        Returns:
            处理结果列表
        """
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def limited_process(item):
            async with semaphore:
                return await async_func(item)
        
        tasks = [limited_process(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

# 使用示例
def demonstrate_performance_optimization():
    """演示性能优化技巧"""
    
    # 1. 缓存优化
    print("=== 缓存优化测试 ===")
    optimizer = PerformanceOptimizer()
    
    # 第一次调用 - 需要计算
    start_time = time.time()
    result1 = optimizer.cached_embedding("hello world")
    first_call_time = time.time() - start_time
    print(f"首次调用耗时: {first_call_time:.3f}秒")
    
    # 第二次调用 - 使用缓存
    start_time = time.time()
    result2 = optimizer.cached_embedding("hello world")
    second_call_time = time.time() - start_time
    print(f"缓存调用耗时: {second_call_time:.3f}秒")
    print(f"性能提升: {first_call_time/second_call_time:.1f}x")
    
    # 2. 批处理优化
    print("\n=== 批处理优化测试 ===")
    items = [f"text_{i}" for i in range(50)]
    
    # 串行处理
    start_time = time.time()
    serial_results = [optimizer.cached_embedding.cache_clear() or 
                     optimizer.cached_embedding(item) for item in items]
    serial_time = time.time() - start_time
    
    # 并发处理
    start_time = time.time()
    concurrent_results = optimizer.batch_process_with_concurrency(
        items, optimizer.cached_embedding, batch_size=10
    )
    concurrent_time = time.time() - start_time
    
    print(f"串行处理耗时: {serial_time:.3f}秒")
    print(f"并发处理耗时: {concurrent_time:.3f}秒")
    print(f"性能提升: {serial_time/concurrent_time:.1f}x")

# 运行优化演示
if __name__ == "__main__":
    demonstrate_performance_optimization()
```

### 3.3 安全最佳实践

```python
"""
安全最佳实践示例
"""
import re
import hashlib
from typing import List, Dict, Any
from langchain_core.callbacks import BaseCallbackHandler

class SecurityValidator:
    """
    安全验证器
    用于验证用户输入和模型输出的安全性
    """
    
    # 危险模式列表
    DANGEROUS_PATTERNS = [
        r'eval\s*\(',           # eval函数调用
        r'exec\s*\(',           # exec函数调用
        r'__import__\s*\(',     # import函数调用
        r'getattr\s*\(',        # getattr函数调用
        r'setattr\s*\(',        # setattr函数调用
        r'delattr\s*\(',        # delattr函数调用
        r'globals\s*\(',        # globals函数调用
        r'locals\s*\(',         # locals函数调用
        r'open\s*\(',           # 文件操作
        r'file\s*\(',           # 文件操作
        r'input\s*\(',          # 用户输入
        r'raw_input\s*\(',      # 用户输入
    ]
    
    # 敏感信息模式
    SENSITIVE_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',           # 信用卡号
        r'\b\d{3}-\d{2}-\d{4}\b',                                 # SSN
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',                          # IP地址
    ]
    
    @classmethod
    def validate_input(cls, text: str) -> Dict[str, Any]:
        """
        验证用户输入的安全性
        
        Args:
            text: 用户输入文本
            
        Returns:
            包含验证结果的字典
        """
        result = {
            'is_safe': True,
            'warnings': [],
            'blocked_patterns': []
        }
        
        # 检查危险模式
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                result['is_safe'] = False
                result['blocked_patterns'].append(pattern)
        
        # 检查敏感信息
        for pattern in cls.SENSITIVE_PATTERNS:
            if re.search(pattern, text):
                result['warnings'].append(f"检测到可能的敏感信息: {pattern}")
        
        # 检查输入长度
        if len(text) > 10000:  # 限制输入长度
            result['warnings'].append("输入文本过长，可能影响性能")
        
        return result
    
    @classmethod
    def sanitize_output(cls, text: str) -> str:
        """
        清理模型输出，移除敏感信息
        
        Args:
            text: 模型输出文本
            
        Returns:
            清理后的文本
        """
        sanitized = text
        
        # 替换敏感信息
        for pattern in cls.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        return sanitized

class SecurityCallbackHandler(BaseCallbackHandler):
    """
    安全回调处理器
    监控和记录安全相关事件
    """
    
    def __init__(self):
        """初始化安全处理器"""
        self.security_events = []
        self.validator = SecurityValidator()
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """在LLM开始前验证输入"""
        for prompt in prompts:
            validation_result = self.validator.validate_input(prompt)
            
            if not validation_result['is_safe']:
                event = {
                    'type': 'SECURITY_VIOLATION',
                    'message': 'Dangerous pattern detected in input',
                    'patterns': validation_result['blocked_patterns'],
                    'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest()[:16]
                }
                self.security_events.append(event)
                raise SecurityError(f"输入包含危险模式: {validation_result['blocked_patterns']}")
            
            if validation_result['warnings']:
                event = {
                    'type': 'SECURITY_WARNING',
                    'message': 'Sensitive information detected',
                    'warnings': validation_result['warnings'],
                    'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest()[:16]
                }
                self.security_events.append(event)
    
    def on_llm_end(self, response, **kwargs) -> None:
        """在LLM结束后清理输出"""
        if hasattr(response, 'generations'):
            for generation in response.generations:
                for gen in generation:
                    if hasattr(gen, 'text'):
                        # 清理敏感信息
                        original_text = gen.text
                        sanitized_text = self.validator.sanitize_output(original_text)
                        
                        if original_text != sanitized_text:
                            event = {
                                'type': 'OUTPUT_SANITIZED',
                                'message': 'Sensitive information removed from output'
                            }
                            self.security_events.append(event)
                            gen.text = sanitized_text
    
    def get_security_report(self) -> Dict[str, Any]:
        """获取安全报告"""
        return {
            'total_events': len(self.security_events),
            'events_by_type': {
                event_type: len([e for e in self.security_events if e['type'] == event_type])
                for event_type in ['SECURITY_VIOLATION', 'SECURITY_WARNING', 'OUTPUT_SANITIZED']
            },
            'recent_events': self.security_events[-10:]  # 最近10个事件
        }

class SecurityError(Exception):
    """安全相关异常"""
    pass

# 安全使用示例
def create_secure_chain():
    """创建安全的处理链"""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    
    # 创建安全回调处理器
    security_handler = SecurityCallbackHandler()
    
    # 创建模型
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        callbacks=[security_handler]
    )
    
    # 创建安全的提示模板
    safe_prompt = PromptTemplate(
        input_variables=["query"],
        template="""请回答以下问题，不要包含任何可执行代码或敏感信息：

问题: {query}

回答:"""
    )
    
    # 创建安全链
    secure_chain = safe_prompt | model
    
    return secure_chain, security_handler

# 测试安全功能
def test_security():
    """测试安全功能"""
    chain, security_handler = create_secure_chain()
    
    # 测试用例
    test_cases = [
        "什么是机器学习？",                    # 安全查询
        "帮我写一段Python代码",                # 普通查询
        "eval('print(1)')",                   # 危险查询
        "我的邮箱是test@example.com",         # 包含敏感信息的查询
    ]
    
    for query in test_cases:
        print(f"\n测试查询: {query}")
        try:
            result = chain.invoke({"query": query})
            print(f"回答: {result.content}")
        except SecurityError as e:
            print(f"安全检查失败: {e}")
        except Exception as e:
            print(f"其他错误: {e}")
    
    # 输出安全报告
    print("\n" + "="*50)
    print("安全报告:")
    report = security_handler.get_security_report()
    for key, value in report.items():
        if key != 'recent_events':
            print(f"{key}: {value}")
    
    if report['recent_events']:
        print("\n最近的安全事件:")
        for event in report['recent_events']:
            print(f"- {event['type']}: {event['message']}")

# 运行安全测试
if __name__ == "__main__":
    test_security()
```

## 4. 总结

本文档展示了LangChain的各种使用模式和最佳实践：

### 4.1 核心原则
1. **统一接口**: 使用Runnable接口实现一致的组件调用
2. **组合式设计**: 通过LCEL实现灵活的组件组合
3. **错误处理**: 实现完善的异常处理和容错机制
4. **性能优化**: 合理使用缓存、并发和批处理
5. **安全第一**: 始终验证输入和清理输出

### 4.2 开发建议
1. **渐进式开发**: 从简单用例开始，逐步增加复杂性
2. **监控完备**: 实施全面的监控和日志记录
3. **测试驱动**: 为所有关键功能编写单元测试
4. **配置管理**: 使用环境变量和配置文件管理应用设置
5. **文档齐全**: 为所有自定义组件编写详细文档

这些示例和最佳实践为构建生产级的LangChain应用提供了坚实的基础。
