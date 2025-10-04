# LangChain-08-VectorStores-Retrievers-API

## 文档说明

本文档详细描述 **VectorStores 和 Retrievers 模块**的对外 API，包括向量存储、相似性搜索、检索策略、文档管理等核心接口的所有公开方法和参数规格。

---

## 1. VectorStore 核心 API

### 1.1 基础接口

#### 基本信息
- **类名**：`VectorStore`
- **功能**：向量存储的抽象基类，提供文档存储和相似性搜索能力
- **核心职责**：向量化存储、相似性计算、文档检索

#### 核心方法

```python
class VectorStore(ABC):
    """向量存储抽象基类。"""

    @abstractmethod
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """添加文本到向量存储。"""

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """相似性搜索。"""

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """带分数的相似性搜索。"""

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """最大边际相关性搜索。"""
```

**方法详解**：

| 方法 | 参数 | 返回类型 | 说明 |
|-----|------|---------|------|
| add_texts | `texts`, `metadatas` | `List[str]` | 添加文本并返回文档ID |
| similarity_search | `query`, `k` | `List[Document]` | 返回最相似的k个文档 |
| similarity_search_with_score | `query`, `k` | `List[Tuple[Document, float]]` | 返回文档和相似度分数 |
| max_marginal_relevance_search | `query`, `k`, `fetch_k`, `lambda_mult` | `List[Document]` | MMR搜索，平衡相关性和多样性 |

---

### 1.2 add_texts - 文档添加

#### 基本信息
- **功能**：向向量存储添加文本文档
- **处理流程**：文本向量化 → 存储向量 → 返回文档ID

#### 方法签名

```python
def add_texts(
    self,
    texts: Iterable[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[str]:
    """添加文本到向量存储。

    Args:
        texts: 要添加的文本列表
        metadatas: 每个文本的元数据（可选）
        ids: 自定义文档ID（可选）
        **kwargs: 额外参数

    Returns:
        添加文档的ID列表
    """
```

#### 使用示例

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# 添加文档
texts = [
    "LangChain是一个用于构建LLM应用的框架",
    "向量数据库用于存储和检索高维向量",
    "RAG结合了检索和生成技术"
]

metadatas = [
    {"source": "doc1", "category": "framework"},
    {"source": "doc2", "category": "database"},
    {"source": "doc3", "category": "technique"}
]

# 执行添加
doc_ids = vectorstore.add_texts(
    texts=texts,
    metadatas=metadatas
)

print(f"添加了 {len(doc_ids)} 个文档")
print(f"文档ID: {doc_ids}")
```

#### 入口函数实现

```python
def add_texts(
    self,
    texts: Iterable[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[str]:
    """添加文本的核心实现。"""

    # 1. 参数验证
    texts_list = list(texts)
    if not texts_list:
        return []

    if metadatas is not None and len(metadatas) != len(texts_list):
        raise ValueError("metadatas长度必须与texts相同")

    # 2. 生成文档ID
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts_list]
    elif len(ids) != len(texts_list):
        raise ValueError("ids长度必须与texts相同")

    # 3. 向量化文本
    embeddings = self._embedding_function.embed_documents(texts_list)

    # 4. 构建文档对象
    documents = []
    for i, text in enumerate(texts_list):
        doc = Document(
            page_content=text,
            metadata=metadatas[i] if metadatas else {}
        )
        documents.append(doc)

    # 5. 存储到向量数据库
    self._add_vectors(
        vectors=embeddings,
        documents=documents,
        ids=ids,
        **kwargs
    )

    return ids
```

---

### 1.3 similarity_search - 相似性搜索

#### 基本信息
- **功能**：基于查询文本找到最相似的文档
- **算法**：余弦相似度、欧几里得距离等

#### 方法签名

```python
def similarity_search(
    self,
    query: str,
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[Document]:
    """相似性搜索。

    Args:
        query: 查询文本
        k: 返回文档数量
        filter: 元数据过滤条件
        **kwargs: 额外参数

    Returns:
        最相似的k个文档
    """
```

#### 使用示例

```python
# 执行相似性搜索
query = "什么是向量数据库？"
results = vectorstore.similarity_search(
    query=query,
    k=3,
    filter={"category": "database"}  # 只搜索数据库相关文档
)

for i, doc in enumerate(results):
    print(f"结果 {i+1}:")
    print(f"  内容: {doc.page_content}")
    print(f"  元数据: {doc.metadata}")
    print()
```

#### 核心实现

```python
def similarity_search(
    self,
    query: str,
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[Document]:
    """相似性搜索实现。"""

    # 1. 向量化查询
    query_embedding = self._embedding_function.embed_query(query)

    # 2. 执行向量搜索
    results = self._similarity_search_by_vector(
        embedding=query_embedding,
        k=k,
        filter=filter,
        **kwargs
    )

    return results

def _similarity_search_by_vector(
    self,
    embedding: List[float],
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[Document]:
    """基于向量的相似性搜索。"""

    # 1. 应用过滤条件
    candidate_docs = self._apply_filter(filter) if filter else self._get_all_docs()

    # 2. 计算相似度
    similarities = []
    for doc_id, doc_vector, document in candidate_docs:
        similarity = self._compute_similarity(embedding, doc_vector)
        similarities.append((similarity, document))

    # 3. 排序并返回Top-K
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in similarities[:k]]
```

---

### 1.4 similarity_search_with_score - 带分数搜索

#### 基本信息
- **功能**：返回文档及其相似度分数
- **用途**：需要了解匹配置信度的场景

#### 使用示例

```python
# 带分数的搜索
results_with_scores = vectorstore.similarity_search_with_score(
    query="LangChain框架",
    k=3
)

for doc, score in results_with_scores:
    print(f"相似度: {score:.4f}")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print("-" * 50)
```

#### 分数解释

| 相似度范围 | 解释 | 建议使用 |
|-----------|------|---------|
| 0.9 - 1.0 | 极高相似度 | 直接使用 |
| 0.7 - 0.9 | 高相似度 | 推荐使用 |
| 0.5 - 0.7 | 中等相似度 | 谨慎使用 |
| 0.3 - 0.5 | 低相似度 | 可能不相关 |
| 0.0 - 0.3 | 极低相似度 | 通常不使用 |

---

### 1.5 max_marginal_relevance_search - MMR搜索

#### 基本信息
- **功能**：最大边际相关性搜索，平衡相关性和多样性
- **算法**：MMR = λ × Sim(q,d) - (1-λ) × max Sim(d,d')

#### 方法签名

```python
def max_marginal_relevance_search(
    self,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[Document]:
    """MMR搜索。

    Args:
        query: 查询文本
        k: 最终返回文档数
        fetch_k: 初始候选文档数
        lambda_mult: 相关性vs多样性权重 (0-1)
        filter: 过滤条件

    Returns:
        平衡相关性和多样性的文档列表
    """
```

#### 参数说明

| 参数 | 范围 | 说明 | 推荐值 |
|-----|------|------|--------|
| lambda_mult | 0.0 - 1.0 | 相关性权重，越大越关注相关性 | 0.5 |
| fetch_k | k - 100 | 初始候选数，越大多样性越好 | 20 |

#### 使用示例

```python
# MMR搜索 - 平衡相关性和多样性
mmr_results = vectorstore.max_marginal_relevance_search(
    query="机器学习算法",
    k=5,           # 最终返回5个文档
    fetch_k=20,    # 从20个候选中选择
    lambda_mult=0.7  # 更关注相关性
)

print("MMR搜索结果（平衡相关性和多样性）:")
for i, doc in enumerate(mmr_results):
    print(f"{i+1}. {doc.page_content[:100]}...")
```

#### MMR算法实现

```python
def max_marginal_relevance_search(
    self,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
) -> List[Document]:
    """MMR搜索实现。"""

    # 1. 获取初始候选文档
    candidates = self.similarity_search_with_score(
        query=query,
        k=fetch_k,
        **kwargs
    )

    if not candidates:
        return []

    # 2. 向量化查询
    query_embedding = self._embedding_function.embed_query(query)

    # 3. MMR选择算法
    selected_docs = []
    selected_embeddings = []
    remaining_candidates = candidates.copy()

    # 选择第一个文档（最相似的）
    first_doc, first_score = remaining_candidates.pop(0)
    selected_docs.append(first_doc)
    first_embedding = self._get_document_embedding(first_doc)
    selected_embeddings.append(first_embedding)

    # 迭代选择剩余文档
    for _ in range(min(k - 1, len(remaining_candidates))):
        best_score = float('-inf')
        best_idx = -1

        for i, (candidate_doc, relevance_score) in enumerate(remaining_candidates):
            candidate_embedding = self._get_document_embedding(candidate_doc)

            # 计算与已选文档的最大相似度
            max_similarity = max([
                self._compute_similarity(candidate_embedding, selected_emb)
                for selected_emb in selected_embeddings
            ])

            # MMR分数计算
            mmr_score = (
                lambda_mult * relevance_score -
                (1 - lambda_mult) * max_similarity
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        # 选择最佳候选
        if best_idx >= 0:
            selected_doc, _ = remaining_candidates.pop(best_idx)
            selected_docs.append(selected_doc)
            selected_embeddings.append(
                self._get_document_embedding(selected_doc)
            )

    return selected_docs
```

---

## 2. BaseRetriever 核心 API

### 2.1 基础接口

#### 基本信息
- **类名**：`BaseRetriever`
- **功能**：文档检索器的抽象基类
- **与VectorStore关系**：更高层的检索抽象

#### 核心方法

```python
class BaseRetriever(RunnableSerializable[RetrieverInput, RetrieverOutput]):
    """检索器基类。"""

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """获取相关文档。"""

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """异步获取相关文档。"""

    @abstractmethod
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """子类需实现的核心检索方法。"""
```

---

### 2.2 VectorStoreRetriever

#### 基本信息
- **功能**：基于VectorStore的检索器实现
- **特性**：支持多种搜索类型和参数配置

#### 创建方式

```python
# 方式1: 从VectorStore创建
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# 方式2: 直接创建
from langchain_core.vectorstores import VectorStoreRetriever

retriever = VectorStoreRetriever(
    vectorstore=vectorstore,
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 6,
        "score_threshold": 0.5
    }
)
```

#### 搜索类型配置

| 搜索类型 | 说明 | 主要参数 |
|---------|------|---------|
| `"similarity"` | 标准相似性搜索 | `k`: 返回文档数 |
| `"similarity_score_threshold"` | 带分数阈值的搜索 | `k`, `score_threshold` |
| `"mmr"` | 最大边际相关性搜索 | `k`, `fetch_k`, `lambda_mult` |

#### 使用示例

```python
# 不同搜索类型的检索器
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,
        "score_threshold": 0.8  # 只返回相似度>0.8的文档
    }
)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)

# 使用检索器
query = "LangChain的主要特性"

# 相似性检索
docs1 = similarity_retriever.get_relevant_documents(query)
print(f"相似性检索: {len(docs1)} 个文档")

# 阈值检索
docs2 = threshold_retriever.get_relevant_documents(query)
print(f"阈值检索: {len(docs2)} 个文档")

# MMR检索
docs3 = mmr_retriever.get_relevant_documents(query)
print(f"MMR检索: {len(docs3)} 个文档")
```

---

## 3. 具体向量存储实现 API

### 3.1 Chroma向量存储

#### 基本信息
- **特点**：开源、轻量级、支持本地和云端
- **适用场景**：开发测试、中小规模应用

#### 创建和配置

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 基础创建
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # 持久化目录
)

# 高级配置
vectorstore = Chroma(
    collection_name="advanced_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"description": "高级配置示例"},
    client_settings={
        "chroma_db_impl": "duckdb+parquet",
        "persist_directory": "./chroma_db"
    }
)
```

#### Chroma特有方法

```python
# 获取集合信息
collection_info = vectorstore._collection.count()
print(f"文档数量: {collection_info}")

# 删除文档
vectorstore.delete(ids=["doc1", "doc2"])

# 更新文档
vectorstore.update_document(
    document_id="doc1",
    document=Document(
        page_content="更新后的内容",
        metadata={"updated": True}
    )
)

# 持久化数据
vectorstore.persist()
```

---

### 3.2 FAISS向量存储

#### 基本信息
- **特点**：Facebook开源，高性能，支持大规模数据
- **适用场景**：生产环境、大规模向量搜索

#### 创建和配置

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 从文档创建
embeddings = OpenAIEmbeddings()
texts = ["文档1", "文档2", "文档3"]

vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=[{"id": i} for i in range(len(texts))]
)

# 保存和加载
vectorstore.save_local("faiss_index")
new_vectorstore = FAISS.load_local("faiss_index", embeddings)
```

#### FAISS特有功能

```python
# 合并索引
other_vectorstore = FAISS.from_texts(["新文档"], embeddings)
vectorstore.merge_from(other_vectorstore)

# 获取索引统计
print(f"向量维度: {vectorstore.index.d}")
print(f"文档数量: {vectorstore.index.ntotal}")

# 自定义距离计算
import faiss

# 创建不同类型的索引
flat_index = faiss.IndexFlatL2(768)  # L2距离
ip_index = faiss.IndexFlatIP(768)    # 内积
```

---

## 4. 文档管理 API

### 4.1 文档添加和更新

#### 批量添加文档

```python
from langchain_core.documents import Document

# 创建文档对象
documents = [
    Document(
        page_content="LangChain是一个强大的框架",
        metadata={
            "source": "doc1.txt",
            "category": "framework",
            "author": "LangChain Team",
            "created_at": "2024-01-01"
        }
    ),
    Document(
        page_content="向量数据库存储高维向量",
        metadata={
            "source": "doc2.txt",
            "category": "database",
            "author": "Vector Team",
            "created_at": "2024-01-02"
        }
    )
]

# 添加文档
doc_ids = vectorstore.add_documents(documents)
print(f"添加了 {len(doc_ids)} 个文档")
```

#### 更新文档

```python
# 更新文档内容
updated_doc = Document(
    page_content="LangChain是一个用于构建LLM应用的强大框架",
    metadata={
        "source": "doc1.txt",
        "category": "framework",
        "author": "LangChain Team",
        "updated_at": "2024-01-15"
    }
)

vectorstore.update_document(
    document_id=doc_ids[0],
    document=updated_doc
)
```

---

### 4.2 文档删除和清理

#### 删除特定文档

```python
# 按ID删除
vectorstore.delete(ids=["doc1", "doc2"])

# 按过滤条件删除
vectorstore.delete(filter={"category": "outdated"})

# 删除所有文档
vectorstore.delete()  # 清空整个集合
```

#### 清理和维护

```python
# 压缩索引（FAISS）
vectorstore.index.train(training_vectors)

# 重建索引
vectorstore.rebuild_index()

# 获取存储统计
stats = vectorstore.get_stats()
print(f"文档数量: {stats['doc_count']}")
print(f"索引大小: {stats['index_size_mb']} MB")
print(f"平均向量维度: {stats['avg_vector_dim']}")
```

---

## 5. 高级搜索功能 API

### 5.1 混合搜索

#### 基本信息
- **功能**：结合向量搜索和关键词搜索
- **优势**：提高搜索准确性和召回率

#### 实现示例

```python
class HybridRetriever(BaseRetriever):
    """混合检索器：向量搜索 + 关键词搜索。"""

    def __init__(
        self,
        vectorstore: VectorStore,
        keyword_retriever: BaseRetriever,
        alpha: float = 0.5
    ):
        self.vectorstore = vectorstore
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha  # 向量搜索权重

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 向量搜索
        vector_docs = self.vectorstore.similarity_search_with_score(
            query, k=10
        )

        # 关键词搜索
        keyword_docs = self.keyword_retriever.get_relevant_documents(query)

        # 合并和重排序
        return self._merge_results(vector_docs, keyword_docs)

    def _merge_results(
        self,
        vector_results: List[Tuple[Document, float]],
        keyword_results: List[Document]
    ) -> List[Document]:
        """合并向量和关键词搜索结果。"""
        # 实现RRF（Reciprocal Rank Fusion）算法
        doc_scores = {}

        # 处理向量搜索结果
        for rank, (doc, score) in enumerate(vector_results):
            doc_key = doc.page_content
            rrf_score = self.alpha / (60 + rank + 1)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + rrf_score

        # 处理关键词搜索结果
        for rank, doc in enumerate(keyword_results):
            doc_key = doc.page_content
            rrf_score = (1 - self.alpha) / (60 + rank + 1)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + rrf_score

        # 排序并返回
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc_content, _ in sorted_docs[:5]]
```

---

### 5.2 多向量搜索

#### 基本信息
- **功能**：使用多个向量表示进行搜索
- **应用**：多模态搜索、多角度匹配

#### 实现示例

```python
class MultiVectorRetriever(BaseRetriever):
    """多向量检索器。"""

    def __init__(
        self,
        vectorstores: List[VectorStore],
        weights: Optional[List[float]] = None
    ):
        self.vectorstores = vectorstores
        self.weights = weights or [1.0] * len(vectorstores)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        all_results = []

        # 从每个向量存储搜索
        for i, vectorstore in enumerate(self.vectorstores):
            results = vectorstore.similarity_search_with_score(query, k=10)

            # 应用权重
            weighted_results = [
                (doc, score * self.weights[i])
                for doc, score in results
            ]
            all_results.extend(weighted_results)

        # 合并相同文档的分数
        doc_scores = {}
        for doc, score in all_results:
            doc_key = doc.page_content
            if doc_key in doc_scores:
                doc_scores[doc_key] = (doc_scores[doc_key][0], doc_scores[doc_key][1] + score)
            else:
                doc_scores[doc_key] = (doc, score)

        # 排序返回
        sorted_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_results[:5]]
```

---

## 6. 性能优化 API

### 6.1 批量操作

#### 批量添加优化

```python
def add_documents_batch(
    vectorstore: VectorStore,
    documents: List[Document],
    batch_size: int = 100
) -> List[str]:
    """批量添加文档，优化性能。"""
    all_ids = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        batch_ids = vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )
        all_ids.extend(batch_ids)

        print(f"已处理 {min(i + batch_size, len(documents))}/{len(documents)} 个文档")

    return all_ids
```

#### 异步批量操作

```python
import asyncio
from typing import AsyncIterator

async def add_documents_async(
    vectorstore: VectorStore,
    documents: AsyncIterator[Document],
    batch_size: int = 100
) -> None:
    """异步批量添加文档。"""
    batch = []

    async for doc in documents:
        batch.append(doc)

        if len(batch) >= batch_size:
            # 处理当前批次
            await asyncio.create_task(
                _process_batch_async(vectorstore, batch)
            )
            batch = []

    # 处理剩余文档
    if batch:
        await _process_batch_async(vectorstore, batch)

async def _process_batch_async(
    vectorstore: VectorStore,
    batch: List[Document]
) -> None:
    """异步处理单个批次。"""
    texts = [doc.page_content for doc in batch]
    metadatas = [doc.metadata for doc in batch]

    # 如果向量存储支持异步操作
    if hasattr(vectorstore, 'aadd_texts'):
        await vectorstore.aadd_texts(texts, metadatas)
    else:
        # 在线程池中执行同步操作
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            vectorstore.add_texts,
            texts,
            metadatas
        )
```

---

### 6.2 缓存和预计算

#### 查询缓存

```python
import hashlib
from functools import lru_cache

class CachedVectorStore:
    """带缓存的向量存储包装器。"""

    def __init__(self, vectorstore: VectorStore, cache_size: int = 128):
        self.vectorstore = vectorstore
        self.cache_size = cache_size
        self._similarity_search = lru_cache(maxsize=cache_size)(
            self._similarity_search_impl
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        """带缓存的相似性搜索。"""
        # 创建缓存键
        cache_key = self._create_cache_key(query, k, kwargs)
        return self._similarity_search(cache_key, query, k, **kwargs)

    def _create_cache_key(
        self,
        query: str,
        k: int,
        kwargs: Dict[str, Any]
    ) -> str:
        """创建缓存键。"""
        content = f"{query}:{k}:{sorted(kwargs.items())}"
        return hashlib.md5(content.encode()).hexdigest()

    def _similarity_search_impl(
        self,
        cache_key: str,  # 用于缓存
        query: str,
        k: int,
        **kwargs: Any
    ) -> List[Document]:
        """实际的搜索实现。"""
        return self.vectorstore.similarity_search(query, k, **kwargs)

    def clear_cache(self) -> None:
        """清空缓存。"""
        self._similarity_search.cache_clear()

    def get_cache_info(self) -> dict:
        """获取缓存统计。"""
        info = self._similarity_search.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize,
            "currsize": info.currsize,
            "hit_rate": info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
        }
```

---

## 7. 总结

本文档详细描述了 **VectorStores 和 Retrievers 模块**的核心 API：

### 主要组件
1. **VectorStore**：向量存储基类，提供文档存储和搜索能力
2. **BaseRetriever**：检索器基类，更高层的检索抽象
3. **具体实现**：Chroma、FAISS等向量存储的具体实现
4. **高级功能**：混合搜索、多向量搜索、MMR等

### 核心方法
1. **add_texts/add_documents**：文档添加和向量化存储
2. **similarity_search**：基于向量相似度的文档检索
3. **max_marginal_relevance_search**：平衡相关性和多样性的MMR搜索
4. **get_relevant_documents**：检索器的统一检索接口

### 性能优化
1. **批量操作**：提高大规模文档处理效率
2. **异步处理**：支持非阻塞的文档操作
3. **缓存机制**：减少重复计算，提高查询速度
4. **索引优化**：针对不同场景的索引策略

每个 API 均包含：
- 完整的方法签名和参数说明
- 详细的使用示例和最佳实践
- 性能优化建议和配置选项
- 错误处理和异常情况说明

VectorStores 和 Retrievers 是RAG（检索增强生成）系统的核心基础设施，正确理解和使用这些API对构建高效的知识检索系统至关重要。
