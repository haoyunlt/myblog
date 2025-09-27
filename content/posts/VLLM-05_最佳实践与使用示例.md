---
title: "VLLM最佳实践与使用示例"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['Python', 'LLM', 'AI推理', '最佳实践', 'vLLM']
categories: ['AI推理']
description: "VLLM最佳实践与使用示例的深入技术分析文档"
keywords: ['Python', 'LLM', 'AI推理', '最佳实践', 'vLLM']
author: "技术分析师"
weight: 1
---

## 1. 框架使用示例

### 1.1 基础文本生成

#### 简单文本生成示例

```python
"""
基础文本生成示例
功能：展示最基本的VLLM使用方法
适用场景：快速原型开发、概念验证
"""

from vllm import LLM, SamplingParams

# 1. 创建LLM实例
# model: 指定要使用的模型路径或HuggingFace模型ID
# tensor_parallel_size: 张量并行大小（GPU数量）
# gpu_memory_utilization: GPU内存使用率（0.0-1.0）
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",  # 模型路径
    tensor_parallel_size=1,            # 单GPU
    gpu_memory_utilization=0.85,       # 85%内存利用率
    trust_remote_code=True,            # 信任远程代码
    max_model_len=4096,               # 最大序列长度
)

# 2. 设置采样参数
# temperature: 控制生成随机性（0.0-2.0，越高越随机）
# top_p: 核采样参数（0.0-1.0）
# max_tokens: 最大生成token数
sampling_params = SamplingParams(
    temperature=0.8,          # 温度参数
    top_p=0.95,              # 核采样
    max_tokens=256,          # 最大生成长度
    stop=["<|endoftext|>"],  # 停止标记
    repetition_penalty=1.1,   # 重复惩罚
)

# 3. 准备输入提示
prompts = [
    "Explain the concept of artificial intelligence in simple terms:",
    "Write a short story about a robot learning to paint:",
    "What are the key differences between machine learning and deep learning?",
]

# 4. 执行生成
# generate()方法会自动批处理多个提示以提高效率
outputs = llm.generate(prompts, sampling_params)

# 5. 处理输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)
```

#### 高级配置示例

```python
"""
高级配置示例
功能：展示VLLM的高级配置选项
适用场景：生产环境部署、性能调优
"""

from vllm import LLM, SamplingParams
from vllm.config import ModelConfig, CacheConfig, ParallelConfig
import torch

# 创建高级配置的LLM实例
llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    
    # 并行配置
    tensor_parallel_size=2,           # 2个GPU张量并行
    pipeline_parallel_size=1,         # 管道并行（暂时设为1）
    
    # 内存管理
    gpu_memory_utilization=0.90,      # 90%内存利用率
    swap_space=4,                     # 4GB交换空间
    cpu_offload_gb=0,                # CPU卸载（GB）
    
    # 性能优化
    enforce_eager=False,              # 启用CUDA Graph
    disable_custom_all_reduce=False,  # 启用自定义all-reduce
    
    # 精度设置
    dtype='float16',                  # 使用FP16精度
    quantization="awq",              # AWQ量化（如果模型支持）
    
    # 缓存配置
    enable_prefix_caching=True,       # 启用前缀缓存
    block_size=16,                   # KV缓存块大小
    
    # 其他选项
    seed=42,                         # 随机种子
    revision=None,                   # 模型版本
)

# 复杂采样参数配置
sampling_params = SamplingParams(
    # 基础参数
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=512,
    min_tokens=10,
    
    # 停止条件
    stop=["Human:", "Assistant:", "\n\n"],
    ignore_eos=False,
    
    # 惩罚机制
    presence_penalty=0.1,            # 存在惩罚
    frequency_penalty=0.2,           # 频率惩罚
    repetition_penalty=1.15,         # 重复惩罚
    
    # 日志概率
    logprobs=5,                      # 返回前5个token的概率
    prompt_logprobs=3,               # prompt的概率信息
    
    # 高级选项
    n=1,                            # 生成序列数量
    best_of=1,                      # 最佳序列选择（V1暂不支持）
    use_beam_search=False,          # 不使用束搜索
    
    # logit偏置（可以调整特定token的概率）
    logit_bias={
        50256: -100.0,              # 降低EOS token概率
    }
)

# 批量处理提示
prompts = [
    "Analyze the pros and cons of renewable energy sources:",
    "Describe the process of photosynthesis in detail:",
    "Explain quantum computing and its potential applications:",
]

# 生成文本
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

# 详细输出分析
for output in outputs:
    print(f"Request ID: {output.request_id}")
    print(f"Prompt tokens: {len(output.prompt_token_ids) if output.prompt_token_ids else 0}")
    
    for i, completion in enumerate(output.outputs):
        print(f"Completion {i+1}:")
        print(f"  Text: {completion.text}")
        print(f"  Tokens: {len(completion.token_ids)}")
        print(f"  Cumulative logprob: {completion.cumulative_logprob}")
        print(f"  Finish reason: {completion.finish_reason}")
        
        # 显示logprob信息
        if completion.logprobs:
            print(f"  Top logprobs for last token:")
            last_logprobs = completion.logprobs[-1]
            for token_id, logprob in last_logprobs.items():
                print(f"    Token {token_id}: {logprob.logprob:.4f}")
    
    print("-" * 60)
```

### 1.2 对话系统示例

#### 基础对话示例

```python
"""
对话系统示例
功能：演示如何使用VLLM构建对话系统
适用场景：聊天机器人、问答系统
"""

from vllm import LLM, SamplingParams

class VLLMChatBot:
    """基于VLLM的聊天机器人类"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        初始化聊天机器人
        
        Args:
            model_name: 聊天模型名称
        """
        # 初始化LLM实例
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            max_model_len=4096,
        )
        
        # 对话专用采样参数
        self.sampling_params = SamplingParams(
            temperature=0.7,              # 适中的随机性
            top_p=0.9,                   # 核采样
            max_tokens=512,              # 对话回复长度
            stop=["<|endoftext|>", "Human:", "[/INST]"],  # 停止标记
            repetition_penalty=1.1,      # 避免重复
        )
        
        # 系统提示词
        self.system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature."""
        
        # 对话历史
        self.conversation_history = []
    
    def chat(self, user_input: str) -> str:
        """
        处理用户输入并生成回复
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            机器人回复
        """
        # 添加用户输入到历史
        self.conversation_history.append(f"Human: {user_input}")
        
        # 构建完整的对话提示
        conversation_prompt = self._build_conversation_prompt()
        
        # 生成回复
        outputs = self.llm.generate([conversation_prompt], self.sampling_params)
        assistant_reply = outputs[0].outputs[0].text.strip()
        
        # 添加助手回复到历史
        self.conversation_history.append(f"Assistant: {assistant_reply}")
        
        return assistant_reply
    
    def _build_conversation_prompt(self) -> str:
        """构建对话提示词"""
        # 格式化对话历史
        conversation = "\n".join(self.conversation_history)
        
        # 构建完整提示
        prompt = f"""<s>[INST] <<SYS>>
{self.system_prompt}
<</SYS>>

{conversation}
Assistant: [/INST]"""
        
        return prompt
    
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history.clear()
    
    def get_conversation_history(self) -> list[str]:
        """获取对话历史"""
        return self.conversation_history.copy()

# 使用示例
def main():
    # 创建聊天机器人
    chatbot = VLLMChatBot()
    
    # 模拟对话
    print("VLLM ChatBot initialized. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
            
        try:
            # 获取机器人回复
            response = chatbot.chat(user_input)
            print(f"Bot: {response}")
            print()
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()
```

#### 多轮对话批处理示例

```python
"""
多轮对话批处理示例
功能：高效处理多个对话会话
适用场景：客服系统、批量对话处理
"""

from vllm import LLM, SamplingParams
from typing import List, Dict, Any
import asyncio
from dataclasses import dataclass
import time

@dataclass
class ConversationSession:
    """对话会话数据类"""
    session_id: str
    messages: List[Dict[str, str]]
    last_update: float
    
class BatchChatProcessor:
    """批量对话处理器"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        # 初始化VLLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["Human:", "Assistant:"],
        )
        
        # 会话管理
        self.sessions: Dict[str, ConversationSession] = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                messages=[],
                last_update=time.time()
            )
        
        self.sessions[session_id].messages.append({
            "role": role,
            "content": content
        })
        self.sessions[session_id].last_update = time.time()
    
    def format_conversation(self, session: ConversationSession) -> str:
        """格式化对话为提示词"""
        formatted_messages = []
        for msg in session.messages:
            if msg["role"] == "user":
                formatted_messages.append(f"Human: {msg['content']}")
            elif msg["role"] == "assistant":
                formatted_messages.append(f"Assistant: {msg['content']}")
        
        # 添加新的Assistant提示
        formatted_messages.append("Assistant:")
        return "\n".join(formatted_messages)
    
    def process_batch(self, session_ids: List[str]) -> Dict[str, str]:
        """批量处理多个会话"""
        # 准备批量输入
        prompts = []
        valid_session_ids = []
        
        for session_id in session_ids:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                prompt = self.format_conversation(session)
                prompts.append(prompt)
                valid_session_ids.append(session_id)
        
        if not prompts:
            return {}
        
        # 批量生成
        start_time = time.time()
        outputs = self.llm.generate(prompts, self.sampling_params)
        generation_time = time.time() - start_time
        
        # 收集结果
        results = {}
        for i, output in enumerate(outputs):
            session_id = valid_session_ids[i]
            response = output.outputs[0].text.strip()
            
            # 更新会话历史
            self.add_message(session_id, "assistant", response)
            results[session_id] = response
        
        print(f"Processed {len(prompts)} conversations in {generation_time:.2f}s")
        return results
    
    def cleanup_old_sessions(self, max_age_seconds: int = 3600):
        """清理旧会话"""
        current_time = time.time()
        old_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.last_update > max_age_seconds
        ]
        
        for session_id in old_sessions:
            del self.sessions[session_id]
        
        print(f"Cleaned up {len(old_sessions)} old sessions")

# 使用示例
def demo_batch_chat():
    processor = BatchChatProcessor()
    
    # 模拟多个用户会话
    test_conversations = [
        ("user1", "What is machine learning?"),
        ("user2", "How does photosynthesis work?"),
        ("user3", "Explain quantum computing."),
        ("user1", "Can you give me a simple example?"),
        ("user2", "What about cellular respiration?"),
    ]
    
    # 添加消息到会话
    for session_id, message in test_conversations:
        processor.add_message(session_id, "user", message)
    
    # 批量处理所有会话
    results = processor.process_batch(["user1", "user2", "user3"])
    
    # 显示结果
    for session_id, response in results.items():
        print(f"Session {session_id}: {response}")
        print("-" * 40)

if __name__ == "__main__":
    demo_batch_chat()
```

### 1.3 多模态处理示例

#### 图像理解示例

```python
"""
多模态图像理解示例
功能：使用VLLM处理图像+文本输入
适用场景：图像描述、视觉问答、图像分析
"""

from vllm import LLM, SamplingParams
from PIL import Image
import torch
import requests
from io import BytesIO

class MultiModalVLLM:
    """多模态VLLM处理器"""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        """
        初始化多模态模型
        
        Args:
            model_name: 多模态模型名称
        """
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            max_model_len=4096,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.2,              # 较低温度保证准确性
            top_p=0.9,
            max_tokens=512,
            stop=["<|endoftext|>"],
        )
    
    def load_image_from_url(self, url: str) -> Image.Image:
        """从URL加载图像"""
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    
    def load_image_from_path(self, path: str) -> Image.Image:
        """从本地路径加载图像"""
        return Image.open(path)
    
    def analyze_image(self, image: Image.Image, question: str) -> str:
        """
        分析图像并回答问题
        
        Args:
            image: PIL图像对象
            question: 关于图像的问题
            
        Returns:
            模型的回答
        """
        # 构建多模态输入
        prompt = {
            "prompt": f"USER: <image>\n{question}\nASSISTANT:",
            "multi_modal_data": {
                "image": image
            }
        }
        
        # 生成回答
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def describe_image(self, image: Image.Image) -> str:
        """生成图像描述"""
        return self.analyze_image(image, "Describe this image in detail.")
    
    def answer_visual_question(self, image: Image.Image, question: str) -> str:
        """回答视觉问题"""
        return self.analyze_image(image, question)
    
    def batch_analyze_images(self, image_question_pairs: list) -> list:
        """批量分析多个图像"""
        prompts = []
        for image, question in image_question_pairs:
            prompt = {
                "prompt": f"USER: <image>\n{question}\nASSISTANT:",
                "multi_modal_data": {
                    "image": image
                }
            }
            prompts.append(prompt)
        
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

# 使用示例
def demo_multimodal():
    # 初始化多模态处理器
    mm_llm = MultiModalVLLM()
    
    # 示例1: 图像描述
    print("=== 图像描述示例 ===")
    try:
        # 加载示例图像
        image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4"
        image = mm_llm.load_image_from_url(image_url)
        
        # 生成描述
        description = mm_llm.describe_image(image)
        print(f"图像描述: {description}")
        
    except Exception as e:
        print(f"图像加载失败: {e}")
    
    # 示例2: 视觉问答
    print("\n=== 视觉问答示例 ===")
    questions = [
        "What is the main subject of this image?",
        "What colors are prominent in this image?",
        "What is the mood or atmosphere of this image?",
    ]
    
    for question in questions:
        try:
            answer = mm_llm.answer_visual_question(image, question)
            print(f"问题: {question}")
            print(f"回答: {answer}")
            print("-" * 40)
        except Exception as e:
            print(f"处理问题失败: {e}")

if __name__ == "__main__":
    demo_multimodal()
```

### 1.4 嵌入和检索示例

#### 文本嵌入示例

```python
"""
文本嵌入和检索示例
功能：使用VLLM生成文本嵌入向量
适用场景：语义搜索、相似度计算、RAG系统
"""

from vllm import LLM, PoolingParams
import numpy as np
from typing import List
import faiss
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """文档片段数据类"""
    text: str
    embedding: np.ndarray
    source: str
    chunk_id: int

class VLLMEmbeddingService:
    """VLLM嵌入服务"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        初始化嵌入模型
        
        Args:
            model_name: 嵌入模型名称
        """
        # 注意：嵌入模型需要使用pooling runner
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            runner="pooling",  # 重要：指定为pooling模式
        )
        
        self.pooling_params = PoolingParams(
            pooling_type="mean",      # 平均池化
            normalize=True,           # 标准化向量
        )
        
        # 文档存储
        self.documents: List[DocumentChunk] = []
        self.index = None
    
    def encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        编码文本为嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        outputs = self.llm.embed(texts, pooling_params=self.pooling_params)
        embeddings = []
        
        for output in outputs:
            # 转换为numpy数组
            embedding = output.outputs.data.cpu().numpy()
            embeddings.append(embedding)
        
        return embeddings
    
    def add_documents(self, texts: List[str], sources: List[str] = None):
        """
        添加文档到检索库
        
        Args:
            texts: 文档文本列表
            sources: 文档来源列表
        """
        if sources is None:
            sources = [f"doc_{i}" for i in range(len(texts))]
        
        # 生成嵌入
        embeddings = self.encode_texts(texts)
        
        # 存储文档片段
        for i, (text, embedding, source) in enumerate(zip(texts, embeddings, sources)):
            chunk = DocumentChunk(
                text=text,
                embedding=embedding,
                source=source,
                chunk_id=len(self.documents)
            )
            self.documents.append(chunk)
        
        # 重建索引
        self._build_index()
    
    def _build_index(self):
        """构建FAISS检索索引"""
        if not self.documents:
            return
        
        # 获取嵌入维度
        embedding_dim = self.documents[0].embedding.shape[0]
        
        # 创建FAISS索引
        self.index = faiss.IndexFlatIP(embedding_dim)  # 内积相似度
        
        # 添加所有嵌入到索引
        embeddings = np.vstack([doc.embedding for doc in self.documents])
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            相关文档片段列表
        """
        if self.index is None:
            return []
        
        # 编码查询
        query_embedding = self.encode_texts([query])[0]
        
        # 搜索
        scores, indices = self.index.search(
            query_embedding.astype('float32').reshape(1, -1), 
            top_k
        )
        
        # 返回结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(doc)
        
        return results
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数（0-1）
        """
        embeddings = self.encode_texts([text1, text2])
        emb1, emb2 = embeddings[0], embeddings[1]
        
        # 计算余弦相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

# 使用示例
def demo_embedding_search():
    # 初始化嵌入服务
    embedding_service = VLLMEmbeddingService()
    
    # 示例文档库
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing deals with the interaction between computers and human language.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Reinforcement learning is about training agents to make decisions in an environment.",
        "Data science involves extracting insights and knowledge from data using various techniques.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Blockchain is a distributed ledger technology that ensures security and transparency.",
    ]
    
    sources = [f"textbook_chapter_{i+1}" for i in range(len(documents))]
    
    # 添加文档到检索库
    print("Adding documents to search index...")
    embedding_service.add_documents(documents, sources)
    
    # 执行搜索
    queries = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Tell me about data analysis techniques",
    ]
    
    for query in queries:
        print(f"\n查询: '{query}'")
        print("=" * 50)
        
        results = embedding_service.search(query, top_k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"{i}. [来源: {doc.source}]")
            print(f"   内容: {doc.text}")
            print()
    
    # 相似度计算示例
    print("=== 相似度计算示例 ===")
    text_pairs = [
        ("machine learning", "artificial intelligence"),
        ("deep learning", "neural networks"),
        ("cloud computing", "data science"),
    ]
    
    for text1, text2 in text_pairs:
        similarity = embedding_service.compute_similarity(text1, text2)
        print(f"'{text1}' 和 '{text2}' 的相似度: {similarity:.4f}")

if __name__ == "__main__":
    demo_embedding_search()
```

## 2. 性能调优最佳实践

### 2.1 内存优化策略

```python
"""
内存优化最佳实践
功能：展示如何优化VLLM的内存使用
适用场景：大模型部署、资源受限环境
"""

from vllm import LLM, SamplingParams
import torch
import psutil
import GPUtil

class MemoryOptimizedVLLM:
    """内存优化的VLLM配置"""
    
    def __init__(self, model_name: str):
        # 获取系统信息
        self.gpu_memory_gb = self._get_gpu_memory()
        self.cpu_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"GPU内存: {self.gpu_memory_gb:.1f} GB")
        print(f"CPU内存: {self.cpu_memory_gb:.1f} GB")
        
        # 根据硬件配置优化参数
        optimal_config = self._calculate_optimal_config()
        
        # 创建优化后的LLM实例
        self.llm = LLM(
            model=model_name,
            **optimal_config
        )
    
    def _get_gpu_memory(self) -> float:
        """获取GPU内存大小"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryTotal / 1024  # MB转GB
        except:
            pass
        return 8.0  # 默认值
    
    def _calculate_optimal_config(self) -> dict:
        """计算最优配置参数"""
        config = {}
        
        # GPU内存利用率优化
        if self.gpu_memory_gb >= 24:
            # 高端GPU（RTX 4090, A100等）
            config.update({
                "gpu_memory_utilization": 0.90,
                "tensor_parallel_size": 1,
                "max_model_len": 8192,
                "block_size": 32,
                "swap_space": 8,
            })
        elif self.gpu_memory_gb >= 16:
            # 中端GPU（RTX 4080, A40等）
            config.update({
                "gpu_memory_utilization": 0.85,
                "tensor_parallel_size": 1,
                "max_model_len": 4096,
                "block_size": 16,
                "swap_space": 4,
            })
        elif self.gpu_memory_gb >= 8:
            # 入门级GPU（RTX 3080等）
            config.update({
                "gpu_memory_utilization": 0.80,
                "tensor_parallel_size": 1,
                "max_model_len": 2048,
                "block_size": 16,
                "swap_space": 2,
                "dtype": "float16",  # 使用FP16节省内存
            })
        else:
            # 低端GPU或CPU
            config.update({
                "gpu_memory_utilization": 0.75,
                "tensor_parallel_size": 1,
                "max_model_len": 1024,
                "block_size": 8,
                "swap_space": 1,
                "dtype": "float16",
                "cpu_offload_gb": 2,  # 启用CPU卸载
            })
        
        # 通用优化设置
        config.update({
            "enable_prefix_caching": True,    # 启用前缀缓存
            "enforce_eager": False,          # 启用CUDA Graph
            "disable_custom_all_reduce": False,
        })
        
        return config
    
    def monitor_memory_usage(self):
        """监控内存使用情况"""
        # GPU内存
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_used = gpu.memoryUsed / 1024  # MB转GB
                gpu_util = gpu.memoryUtil * 100
                print(f"GPU内存使用: {gpu_used:.1f}/{self.gpu_memory_gb:.1f} GB ({gpu_util:.1f}%)")
        except:
            pass
        
        # CPU内存
        cpu_memory = psutil.virtual_memory()
        cpu_used = cpu_memory.used / (1024**3)
        cpu_util = cpu_memory.percent
        print(f"CPU内存使用: {cpu_used:.1f}/{self.cpu_memory_gb:.1f} GB ({cpu_util:.1f}%)")
        
        # PyTorch GPU内存
        if torch.cuda.is_available():
            torch_allocated = torch.cuda.memory_allocated() / (1024**3)
            torch_cached = torch.cuda.memory_reserved() / (1024**3)
            print(f"PyTorch GPU内存: 已分配 {torch_allocated:.1f} GB, 已缓存 {torch_cached:.1f} GB")

# 内存监控装饰器
def monitor_memory(func):
    """内存监控装饰器"""
    def wrapper(*args, **kwargs):
        print(f"执行前内存状态:")
        if hasattr(args[0], 'monitor_memory_usage'):
            args[0].monitor_memory_usage()
        
        result = func(*args, **kwargs)
        
        print(f"执行后内存状态:")
        if hasattr(args[0], 'monitor_memory_usage'):
            args[0].monitor_memory_usage()
        
        return result
    return wrapper

# 使用示例
def demo_memory_optimization():
    print("=== 内存优化演示 ===")
    
    # 创建内存优化的VLLM实例
    optimized_llm = MemoryOptimizedVLLM("microsoft/DialoGPT-small")
    
    # 监控内存使用
    optimized_llm.monitor_memory_usage()
    
    # 配置采样参数（内存友好）
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=128,  # 较短的输出长度
        top_p=0.9,
    )
    
    # 小批量测试
    prompts = [
        "What is the future of AI?",
        "Explain machine learning briefly.",
    ]
    
    print("\n开始生成...")
    outputs = optimized_llm.llm.generate(prompts, sampling_params)
    
    # 显示结果
    for i, output in enumerate(outputs):
        print(f"输出 {i+1}: {output.outputs[0].text}")
    
    # 最终内存状态
    print("\n最终内存状态:")
    optimized_llm.monitor_memory_usage()

if __name__ == "__main__":
    demo_memory_optimization()
```

### 2.2 批处理优化

```python
"""
批处理优化最佳实践
功能：优化批处理性能，提高吞吐量
适用场景：批量推理、生产环境部署
"""

from vllm import LLM, SamplingParams
import time
import threading
from queue import Queue
from typing import List, Tuple
from dataclasses import dataclass
import uuid

@dataclass
class BatchRequest:
    """批处理请求"""
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    timestamp: float
    callback: callable = None

class OptimizedBatchProcessor:
    """优化的批处理器"""
    
    def __init__(
        self, 
        model_name: str,
        max_batch_size: int = 32,
        batch_timeout: float = 0.1,  # 批处理超时时间（秒）
    ):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            # 批处理优化配置
            max_num_seqs=max_batch_size,              # 最大并发序列数
            max_num_batched_tokens=8192,              # 最大批处理token数
            enable_prefix_caching=True,               # 启用前缀缓存
        )
        
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        
        # 请求队列和批处理线程
        self.request_queue = Queue()
        self.response_callbacks = {}
        self.running = False
        self.batch_thread = None
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "total_tokens": 0,
            "total_time": 0,
        }
    
    def start(self):
        """启动批处理线程"""
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processing_loop)
        self.batch_thread.start()
        print("批处理器已启动")
    
    def stop(self):
        """停止批处理器"""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join()
        print("批处理器已停止")
    
    def submit_request(
        self, 
        prompt: str, 
        sampling_params: SamplingParams = None,
        callback: callable = None
    ) -> str:
        """
        提交请求
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            callback: 完成回调函数
            
        Returns:
            请求ID
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.8,
                max_tokens=256,
                top_p=0.9,
            )
        
        request_id = str(uuid.uuid4())
        request = BatchRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            timestamp=time.time(),
            callback=callback
        )
        
        self.request_queue.put(request)
        return request_id
    
    def _collect_batch(self) -> List[BatchRequest]:
        """收集一个批次的请求"""
        batch = []
        deadline = time.time() + self.batch_timeout
        
        # 获取第一个请求（阻塞等待）
        try:
            first_request = self.request_queue.get(timeout=self.batch_timeout)
            batch.append(first_request)
        except:
            return batch
        
        # 收集更多请求直到达到批次大小或超时
        while (
            len(batch) < self.max_batch_size and 
            time.time() < deadline and 
            not self.request_queue.empty()
        ):
            try:
                request = self.request_queue.get_nowait()
                batch.append(request)
            except:
                break
        
        return batch
    
    def _process_batch(self, batch: List[BatchRequest]) -> None:
        """处理一个批次的请求"""
        if not batch:
            return
        
        start_time = time.time()
        
        # 准备批处理输入
        prompts = [req.prompt for req in batch]
        
        # 统一采样参数（简化处理，实际可以支持不同参数）
        # 这里使用第一个请求的参数
        sampling_params = batch[0].sampling_params
        
        try:
            # 执行批量生成
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            
            # 处理结果
            for request, output in zip(batch, outputs):
                response_text = output.outputs[0].text
                
                # 调用回调函数
                if request.callback:
                    request.callback(request.request_id, response_text)
                
                # 更新统计
                self.stats["total_tokens"] += len(output.outputs[0].token_ids)
            
            # 更新批处理统计
            batch_time = time.time() - start_time
            self.stats["total_requests"] += len(batch)
            self.stats["total_batches"] += 1
            self.stats["total_time"] += batch_time
            
            print(f"处理批次: {len(batch)}个请求, 用时: {batch_time:.3f}s")
            
        except Exception as e:
            print(f"批处理错误: {e}")
            # 通知所有请求失败
            for request in batch:
                if request.callback:
                    request.callback(request.request_id, f"Error: {e}")
    
    def _batch_processing_loop(self):
        """批处理主循环"""
        print("批处理循环启动")
        
        while self.running:
            try:
                # 收集批次
                batch = self._collect_batch()
                
                if batch:
                    # 处理批次
                    self._process_batch(batch)
                else:
                    # 没有请求时短暂休眠
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"批处理循环错误: {e}")
                time.sleep(0.1)
        
        print("批处理循环结束")
    
    def get_stats(self) -> dict:
        """获取性能统计"""
        stats = self.stats.copy()
        if stats["total_time"] > 0:
            stats["throughput"] = stats["total_tokens"] / stats["total_time"]
            stats["requests_per_second"] = stats["total_requests"] / stats["total_time"]
            stats["avg_batch_size"] = stats["total_requests"] / max(stats["total_batches"], 1)
        return stats

# 使用示例
def demo_batch_optimization():
    print("=== 批处理优化演示 ===")
    
    # 创建批处理器
    batch_processor = OptimizedBatchProcessor(
        model_name="microsoft/DialoGPT-small",
        max_batch_size=8,
        batch_timeout=0.5,
    )
    
    # 结果收集
    results = {}
    
    def result_callback(request_id: str, response: str):
        """结果回调函数"""
        results[request_id] = response
        print(f"请求 {request_id[:8]}... 完成")
    
    # 启动批处理器
    batch_processor.start()
    
    try:
        # 提交多个请求
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How does deep learning work?",
            "What are neural networks?",
            "Describe natural language processing.",
            "What is computer vision?",
            "Explain reinforcement learning.",
            "What is data science?",
        ]
        
        request_ids = []
        start_time = time.time()
        
        # 提交所有请求
        for prompt in test_prompts:
            request_id = batch_processor.submit_request(
                prompt=prompt,
                callback=result_callback
            )
            request_ids.append(request_id)
            print(f"提交请求: {request_id[:8]}...")
        
        # 等待所有请求完成
        print("等待所有请求完成...")
        while len(results) < len(request_ids):
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        
        # 显示结果
        print(f"\n所有请求完成，总耗时: {total_time:.2f}秒")
        
        for i, request_id in enumerate(request_ids):
            if request_id in results:
                print(f"请求{i+1}: {results[request_id][:100]}...")
        
        # 显示性能统计
        stats = batch_processor.get_stats()
        print(f"\n性能统计:")
        print(f"  总请求数: {stats['total_requests']}")
        print(f"  总批次数: {stats['total_batches']}")
        print(f"  平均批次大小: {stats.get('avg_batch_size', 0):.1f}")
        print(f"  吞吐量: {stats.get('throughput', 0):.1f} tokens/秒")
        print(f"  请求率: {stats.get('requests_per_second', 0):.1f} 请求/秒")
        
    finally:
        # 停止批处理器
        batch_processor.stop()

if __name__ == "__main__":
    demo_batch_optimization()
```

## 3. 部署和生产环境最佳实践

### 3.1 生产环境配置

```python
"""
生产环境部署最佳实践
功能：展示生产级VLLM部署配置
适用场景：高可用服务部署、企业级应用
"""

import os
import logging
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
import signal
import sys
from typing import Optional
import json

class ProductionVLLMService:
    """生产环境VLLM服务"""
    
    def __init__(self, config_path: str):
        """
        从配置文件初始化生产服务
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.llm = None
        self._setup_logging()
        self._setup_signal_handlers()
        self._initialize_llm()
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 环境变量覆盖
        for key, value in config.items():
            env_key = f"VLLM_{key.upper()}"
            if env_key in os.environ:
                config[key] = os.environ[env_key]
        
        return config
    
    def _setup_logging(self):
        """配置日志系统"""
        log_level = self.config.get("log_level", "INFO")
        log_file = self.config.get("log_file", "vllm_service.log")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging configured")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        signal.signal(signal.SIGTERM, self._graceful_shutdown)
        signal.signal(signal.SIGINT, self._graceful_shutdown)
    
    def _graceful_shutdown(self, signum, frame):
        """优雅关闭服务"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        # 清理资源
        if self.llm:
            # 这里可以添加模型状态保存逻辑
            pass
        sys.exit(0)
    
    def _initialize_llm(self):
        """初始化LLM实例"""
        try:
            # 生产级配置
            self.llm = LLM(
                model=self.config["model_path"],
                tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
                pipeline_parallel_size=self.config.get("pipeline_parallel_size", 1),
                
                # 内存优化
                gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.85),
                swap_space=self.config.get("swap_space", 4),
                cpu_offload_gb=self.config.get("cpu_offload_gb", 0),
                
                # 性能优化
                enforce_eager=self.config.get("enforce_eager", False),
                disable_custom_all_reduce=self.config.get("disable_custom_all_reduce", False),
                enable_prefix_caching=self.config.get("enable_prefix_caching", True),
                
                # 安全设置
                trust_remote_code=self.config.get("trust_remote_code", False),
                revision=self.config.get("model_revision", None),
                
                # 限制设置
                max_model_len=self.config.get("max_model_len", 4096),
                max_num_seqs=self.config.get("max_batch_size", 64),
                max_num_batched_tokens=self.config.get("max_batch_tokens", 8192),
            )
            
            self.logger.info("LLM initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        try:
            # 默认采样参数
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 512),
                stop=kwargs.get("stop", []),
            )
            
            # 生成
            outputs = self.llm.generate([prompt], sampling_params)
            result = outputs[0].outputs[0].text
            
            self.logger.info(f"Generated response for prompt length: {len(prompt)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def health_check(self) -> dict:
        """健康检查"""
        try:
            # 简单的健康检查生成
            test_prompt = "Hello"
            sampling_params = SamplingParams(max_tokens=5, temperature=0)
            outputs = self.llm.generate([test_prompt], sampling_params)
            
            return {
                "status": "healthy",
                "model": self.config["model_path"],
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# 生产配置示例 (production_config.json)
PRODUCTION_CONFIG = {
    "model_path": "meta-llama/Llama-2-7b-hf",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.85,
    "max_batch_size": 64,
    "max_batch_tokens": 8192,
    "enable_prefix_caching": True,
    "log_level": "INFO",
    "log_file": "/var/log/vllm/service.log",
    "trust_remote_code": False,
    "max_model_len": 4096
}

# 使用示例
def deploy_production_service():
    # 创建配置文件
    with open("production_config.json", "w") as f:
        json.dump(PRODUCTION_CONFIG, f, indent=2)
    
    # 启动服务
    service = ProductionVLLMService("production_config.json")
    
    # 测试服务
    print("=== 生产服务测试 ===")
    health = service.health_check()
    print(f"健康状态: {health}")
    
    if health["status"] == "healthy":
        response = service.generate(
            "What are the key considerations for deploying AI models in production?",
            temperature=0.7,
            max_tokens=256
        )
        print(f"生成响应: {response}")

if __name__ == "__main__":
    deploy_production_service()
```

### 3.2 错误处理和监控

```python
"""
错误处理和监控最佳实践
功能：展示如何实现完善的错误处理和监控
适用场景：生产环境的稳定性保障
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import GPUtil
from vllm import LLM, SamplingParams

@dataclass
class MetricData:
    """监控指标数据"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

class VLLMMonitor:
    """VLLM监控系统"""
    
    def __init__(self, llm: LLM, collection_interval: float = 10.0):
        self.llm = llm
        self.collection_interval = collection_interval
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Callable] = {}
        self.running = False
        self.monitor_thread = None
        
        self.logger = logging.getLogger(__name__)
        
    def register_alert(self, metric_name: str, threshold: float, callback: Callable):
        """注册告警回调"""
        self.alerts[metric_name] = {
            "threshold": threshold,
            "callback": callback
        }
    
    def start_monitoring(self):
        """启动监控"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        self.logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """监控主循环"""
        while self.running:
            try:
                # 收集系统指标
                self._collect_system_metrics()
                
                # 收集GPU指标
                self._collect_gpu_metrics()
                
                # 检查告警
                self._check_alerts()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(1)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        current_time = time.time()
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        self.metrics["cpu_utilization"].append(
            MetricData(current_time, cpu_percent)
        )
        
        # 内存使用
        memory = psutil.virtual_memory()
        self.metrics["memory_utilization"].append(
            MetricData(current_time, memory.percent)
        )
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        self.metrics["disk_utilization"].append(
            MetricData(current_time, disk.percent)
        )
    
    def _collect_gpu_metrics(self):
        """收集GPU指标"""
        try:
            gpus = GPUtil.getGPUs()
            current_time = time.time()
            
            for i, gpu in enumerate(gpus):
                labels = {"gpu_id": str(i)}
                
                self.metrics["gpu_utilization"].append(
                    MetricData(current_time, gpu.load * 100, labels)
                )
                
                self.metrics["gpu_memory_utilization"].append(
                    MetricData(current_time, gpu.memoryUtil * 100, labels)
                )
                
                self.metrics["gpu_temperature"].append(
                    MetricData(current_time, gpu.temperature, labels)
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to collect GPU metrics: {e}")
    
    def _check_alerts(self):
        """检查告警条件"""
        for metric_name, alert_config in self.alerts.items():
            if metric_name in self.metrics and self.metrics[metric_name]:
                latest_metric = self.metrics[metric_name][-1]
                
                if latest_metric.value > alert_config["threshold"]:
                    try:
                        alert_config["callback"](metric_name, latest_metric)
                    except Exception as e:
                        self.logger.error(f"Alert callback error: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Dict]:
        """获取指标摘要"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                recent_values = [v.value for v in list(values)[-10:]]  # 最近10个值
                summary[metric_name] = {
                    "current": values[-1].value,
                    "avg": sum(recent_values) / len(recent_values),
                    "min": min(recent_values),
                    "max": max(recent_values),
                    "count": len(values)
                }
        
        return summary

class RobustVLLMService:
    """健壮的VLLM服务"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = None
        self.monitor = None
        self.error_counts = defaultdict(int)
        self.circuit_breaker_open = False
        self.last_error_time = 0
        
        self.logger = logging.getLogger(__name__)
        
        # 错误阈值配置
        self.max_errors = 5
        self.error_window = 300  # 5分钟
        self.circuit_breaker_timeout = 60  # 1分钟
        
    def initialize(self):
        """初始化服务"""
        try:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
            )
            
            # 启动监控
            self.monitor = VLLMMonitor(self.llm)
            self._setup_alerts()
            self.monitor.start_monitoring()
            
            self.logger.info("Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service: {e}")
            raise
    
    def _setup_alerts(self):
        """设置告警"""
        def gpu_memory_alert(metric_name, metric_data):
            self.logger.warning(f"GPU memory usage high: {metric_data.value:.1f}%")
        
        def gpu_temperature_alert(metric_name, metric_data):
            self.logger.warning(f"GPU temperature high: {metric_data.value:.1f}°C")
        
        # 注册告警
        self.monitor.register_alert("gpu_memory_utilization", 90.0, gpu_memory_alert)
        self.monitor.register_alert("gpu_temperature", 80.0, gpu_temperature_alert)
    
    def _circuit_breaker_check(self):
        """断路器检查"""
        current_time = time.time()
        
        # 清理过期错误记录
        recent_errors = sum(
            1 for error_time in self.error_counts.values()
            if current_time - error_time < self.error_window
        )
        
        # 检查是否应该开启断路器
        if recent_errors >= self.max_errors:
            self.circuit_breaker_open = True
            self.last_error_time = current_time
            self.logger.warning("Circuit breaker opened due to high error rate")
        
        # 检查是否可以关闭断路器
        if (self.circuit_breaker_open and 
            current_time - self.last_error_time > self.circuit_breaker_timeout):
            self.circuit_breaker_open = False
            self.error_counts.clear()
            self.logger.info("Circuit breaker closed, service restored")
    
    def generate_with_retry(
        self, 
        prompt: str, 
        max_retries: int = 3,
        **kwargs
    ) -> Optional[str]:
        """带重试的文本生成"""
        
        # 断路器检查
        self._circuit_breaker_check()
        if self.circuit_breaker_open:
            raise Exception("Service temporarily unavailable (circuit breaker open)")
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                sampling_params = SamplingParams(
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 512),
                    top_p=kwargs.get("top_p", 0.9),
                )
                
                outputs = self.llm.generate([prompt], sampling_params)
                result = outputs[0].outputs[0].text
                
                # 成功时重置错误计数
                if attempt > 0:
                    self.logger.info(f"Generation succeeded after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                self.error_counts[time.time()] = time.time()
                
                self.logger.warning(
                    f"Generation attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    time.sleep(wait_time)
        
        # 所有重试都失败
        self.logger.error(f"Generation failed after {max_retries} attempts")
        raise last_exception
    
    def get_service_status(self) -> Dict:
        """获取服务状态"""
        status = {
            "circuit_breaker_open": self.circuit_breaker_open,
            "error_count": len(self.error_counts),
            "last_error_time": self.last_error_time,
            "service_healthy": not self.circuit_breaker_open
        }
        
        if self.monitor:
            status["metrics"] = self.monitor.get_metrics_summary()
        
        return status
    
    def shutdown(self):
        """关闭服务"""
        if self.monitor:
            self.monitor.stop_monitoring()
        self.logger.info("Service shutdown complete")

# 使用示例
def demo_robust_service():
    print("=== 健壮服务演示 ===")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建服务
    service = RobustVLLMService("microsoft/DialoGPT-small")
    
    try:
        # 初始化服务
        service.initialize()
        
        # 测试正常操作
        print("测试正常生成...")
        response = service.generate_with_retry(
            "What are the benefits of robust system design?",
            temperature=0.7,
            max_tokens=128
        )
        print(f"响应: {response}")
        
        # 获取服务状态
        status = service.get_service_status()
        print(f"服务状态: {status}")
        
        # 等待一段时间收集监控数据
        print("收集监控数据...")
        time.sleep(15)
        
        # 显示最终状态
        final_status = service.get_service_status()
        print(f"最终状态: {final_status}")
        
    except Exception as e:
        print(f"服务运行错误: {e}")
    
    finally:
        # 关闭服务
        service.shutdown()

if __name__ == "__main__":
    demo_robust_service()
```

## 4. 实战经验总结

### 4.1 性能优化经验

#### 内存优化要点

1. **GPU内存利用率设置**：
   - 生产环境建议设置为0.85-0.90
   - 开发环境可设置为0.7-0.8留出调试空间
   - 避免设置过高导致OOM错误

2. **KV缓存优化**：
   - 启用前缀缓存可显著提高重复前缀的处理效率
   - 合理设置block_size，通常16-32为最佳
   - 监控缓存命中率，低于50%时考虑调整策略

3. **批处理策略**：
   - 动态批处理比静态批处理效率更高
   - 批次大小与GPU内存成正比
   - 考虑请求长度的方差，避免内存浪费

#### 吞吐量优化

1. **并行策略选择**：
   - 7B以下模型：单GPU即可
   - 7B-13B模型：考虑张量并行
   - 13B以上模型：张量并行 + 管道并行

2. **CUDA Graph优化**：
   - 默认开启CUDA Graph可提升10-30%性能
   - 某些模型或配置下可能不稳定，需要测试

### 4.2 部署经验

#### 模型选择

1. **推理专用模型**：
   - 优先选择针对推理优化的模型变体
   - 考虑量化版本以节省内存和提升速度
   - 评估模型质量与资源消耗的平衡

2. **硬件适配**：
   - 不同GPU架构的性能差异显著
   - 考虑模型大小与GPU内存的匹配
   - 网络带宽对多GPU部署很重要

#### 生产环境配置

1. **资源管理**：
   - 预留系统资源，避免竞争
   - 设置合理的超时和重试机制
   - 实施熔断器模式防止雪崩

2. **监控和告警**：
   - 监控GPU利用率、内存使用、温度
   - 设置关键指标的阈值告警
   - 记录详细的性能和错误日志

### 4.3 故障排除指南

#### 常见问题解决

1. **OOM错误**：
   ```python
   # 降低GPU内存利用率
   llm = LLM(gpu_memory_utilization=0.7)
   
   # 减少批处理大小
   llm = LLM(max_num_seqs=16)
   
   # 启用CPU卸载
   llm = LLM(cpu_offload_gb=2)
   ```

2. **性能问题**：
   ```python
   # 启用前缀缓存
   llm = LLM(enable_prefix_caching=True)
   
   # 使用CUDA Graph
   llm = LLM(enforce_eager=False)
   
   # 优化数据类型
   llm = LLM(dtype="float16")
   ```

3. **稳定性问题**：
   ```python
   # 保守的内存设置
   llm = LLM(
       gpu_memory_utilization=0.8,
       swap_space=4,
       enforce_eager=True  # 如果CUDA Graph不稳定
   )
   ```

这些最佳实践和使用示例为VLLM的高效使用提供了全面的指导，涵盖了从基础使用到生产部署的各个方面。通过这些实战经验，开发者可以更好地理解和应用VLLM框架。
