# LangChain-08-VectorStores-Retrievers-时序图

## 文档说明

本文档通过详细的时序图展示 **VectorStores 和 Retrievers 模块**在各种场景下的执行流程，包括文档添加、向量化、相似性搜索、MMR算法、检索器使用等复杂交互过程。

---

## 1. 文档添加场景

### 1.1 VectorStore 文档添加流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant VectorStore
    participant Embeddings as EmbeddingFunction
    participant Index as VectorIndex
    participant DocStore as DocumentStore

    User->>VectorStore: add_texts(["文档1", "文档2"], metadatas)

    VectorStore->>VectorStore: 参数验证<br/>检查texts和metadatas长度匹配

    VectorStore->>VectorStore: 生成文档ID<br/>ids = [uuid4(), uuid4()]

    VectorStore->>Embeddings: embed_documents(["文档1", "文档2"])
    Embeddings->>Embeddings: 调用OpenAI API<br/>或本地模型进行向量化
    Embeddings-->>VectorStore: [[0.1, 0.2, ...], [0.3, 0.4, ...]]

    VectorStore->>VectorStore: 构建Document对象<br/>page_content + metadata

    par 并行存储
        VectorStore->>Index: add_vectors(embeddings, ids)
        Index->>Index: 构建向量索引<br/>更新相似性搜索结构
        Index-->>VectorStore: 索引更新完成
    and
        VectorStore->>DocStore: store_documents(docs, ids)
        DocStore->>DocStore: 存储文档内容和元数据
        DocStore-->>VectorStore: 文档存储完成
    end

    VectorStore->>VectorStore: 更新统计信息<br/>文档数量、索引大小等

    VectorStore-->>User: 返回文档ID列表<br/>["id1", "id2"]
```

**关键步骤详解**：

1. **参数验证**（步骤 2）：
   - 检查texts列表不为空
   - 验证metadatas长度与texts匹配
   - 处理可选的自定义IDs

2. **向量化处理**（步骤 4-6）：
   - 批量调用embedding函数
   - 处理API限制和重试逻辑
   - 缓存向量化结果

3. **并行存储**（步骤 8-13）：
   - 向量索引和文档存储并行进行
   - 提高整体处理性能
   - 确保数据一致性

**性能特征**：
- 批量向量化：减少API调用次数
- 并行存储：提高I/O效率
- 索引优化：支持增量更新

---

### 1.2 FAISS 向量添加流程

```mermaid
sequenceDiagram
    autonumber
    participant VectorStore as FAISS
    participant Index as FAISSIndex
    participant DocStore as InMemoryDocstore
    participant Mapping as IndexToDocMapping

    VectorStore->>VectorStore: 接收向量和文档数据

    VectorStore->>Index: 检查索引维度匹配<br/>验证向量维度与索引一致

    alt 维度不匹配
        Index-->>VectorStore: raise ValueError("维度不匹配")
    end

    VectorStore->>Index: 准备向量数据<br/>转换为numpy.float32格式

    VectorStore->>Index: index.add(vector_array)
    Index->>Index: 更新FAISS索引结构<br/>重新计算聚类中心（如果需要）

    Index->>Index: 获取新添加向量的索引位置<br/>start_idx = 之前的ntotal
    Index-->>VectorStore: 返回索引位置范围

    loop 为每个文档建立映射
        VectorStore->>Mapping: index_to_docstore_id[idx] = doc_id
        VectorStore->>DocStore: add_document(doc_id, document)
        DocStore->>DocStore: 存储文档到内存字典
    end

    VectorStore->>VectorStore: 更新统计信息<br/>total_vectors++, last_added_time

    VectorStore-->>VectorStore: 添加完成
```

**FAISS特有处理**：

```python
# 向量数据预处理
def prepare_vectors_for_faiss(vectors: List[List[float]]) -> np.ndarray:
    """为FAISS准备向量数据。"""
    import numpy as np

    # 转换为numpy数组
    vector_array = np.array(vectors, dtype=np.float32)

    # L2归一化（如果需要）
    if self._normalize_L2:
        faiss.normalize_L2(vector_array)

    return vector_array

# 索引类型对应的处理
def handle_index_type(index_type: str, vectors: np.ndarray):
    if index_type == "IndexIVFFlat":
        # 需要训练聚类中心
        if not index.is_trained:
            index.train(vectors)
    elif index_type == "IndexHNSW":
        # 层次化导航小世界图
        # 无需特殊训练
        pass
```

---

## 2. 相似性搜索场景

### 2.1 标准相似性搜索流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant VectorStore
    participant Embeddings
    participant Index
    participant Filter as MetadataFilter
    participant Ranker as ResultRanker

    User->>VectorStore: similarity_search("什么是AI?", k=5, filter={"category": "tech"})

    VectorStore->>Embeddings: embed_query("什么是AI?")
    Embeddings->>Embeddings: 向量化查询文本<br/>调用embedding模型
    Embeddings-->>VectorStore: query_vector = [0.2, 0.3, ...]

    VectorStore->>Index: search_by_vector(query_vector, k=20)
    Note over Index: 获取更多候选文档<br/>为后续过滤预留空间

    Index->>Index: 计算向量相似度<br/>使用余弦相似度或欧几里得距离
    Index->>Index: 排序候选结果<br/>按相似度降序排列
    Index-->>VectorStore: [(doc1, 0.95), (doc2, 0.87), ...]

    alt 有过滤条件
        VectorStore->>Filter: 应用元数据过滤<br/>filter={"category": "tech"}

        loop 遍历每个候选文档
            Filter->>Filter: 检查doc.metadata["category"] == "tech"
            alt 匹配过滤条件
                Filter->>Filter: 保留文档
            else 不匹配
                Filter->>Filter: 丢弃文档
            end
        end

        Filter-->>VectorStore: 过滤后的文档列表
    end

    VectorStore->>Ranker: 最终排序和截取<br/>取前k=5个结果
    Ranker-->>VectorStore: 最终结果列表

    VectorStore-->>User: [Document1, Document2, Document3, Document4, Document5]
```

**相似度计算详解**：

```python
def compute_similarity_scores(
    query_vector: List[float],
    candidate_vectors: List[List[float]],
    distance_strategy: str = "cosine"
) -> List[float]:
    """计算相似度分数。"""

    if distance_strategy == "cosine":
        scores = []
        query_norm = np.linalg.norm(query_vector)

        for candidate_vector in candidate_vectors:
            candidate_norm = np.linalg.norm(candidate_vector)

            if query_norm == 0 or candidate_norm == 0:
                scores.append(0.0)
            else:
                dot_product = np.dot(query_vector, candidate_vector)
                cosine_sim = dot_product / (query_norm * candidate_norm)
                scores.append(cosine_sim)

        return scores

    elif distance_strategy == "euclidean":
        scores = []
        for candidate_vector in candidate_vectors:
            distance = np.linalg.norm(
                np.array(query_vector) - np.array(candidate_vector)
            )
            # 转换距离为相似度分数
            similarity = 1.0 / (1.0 + distance)
            scores.append(similarity)

        return scores
```

---

### 2.2 带分数阈值的搜索流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant VectorStore
    participant Embeddings
    participant Index
    participant Threshold as ScoreThreshold

    User->>VectorStore: similarity_search_with_score("机器学习", k=10, score_threshold=0.8)

    VectorStore->>Embeddings: embed_query("机器学习")
    Embeddings-->>VectorStore: query_vector

    VectorStore->>Index: search_with_scores(query_vector, k=50)
    Note over Index: 获取更多候选以应对阈值过滤

    Index-->>VectorStore: [(doc1, 0.95), (doc2, 0.85), (doc3, 0.75), ...]

    VectorStore->>Threshold: 应用分数阈值过滤<br/>score_threshold=0.8

    loop 遍历搜索结果
        Threshold->>Threshold: 检查 score >= 0.8

        alt 分数达到阈值
            Threshold->>Threshold: 保留文档<br/>(doc1, 0.95) ✓<br/>(doc2, 0.85) ✓
        else 分数低于阈值
            Threshold->>Threshold: 丢弃文档<br/>(doc3, 0.75) ✗
        end
    end

    Threshold-->>VectorStore: 过滤后结果<br/>只包含高质量匹配

    VectorStore->>VectorStore: 截取前k个结果<br/>最多返回10个文档

    VectorStore-->>User: [(Document1, 0.95), (Document2, 0.85), ...]
```

**阈值过滤优势**：
- 保证结果质量：只返回高相关性文档
- 动态结果数：实际返回数可能少于k
- 避免噪声：过滤掉低质量匹配

---

## 3. MMR搜索场景

### 3.1 最大边际相关性搜索流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant VectorStore
    participant Embeddings
    participant Index
    participant MMR as MMRSelector
    participant Calculator as SimilarityCalculator

    User->>VectorStore: max_marginal_relevance_search("深度学习", k=5, fetch_k=20, lambda_mult=0.7)

    VectorStore->>Embeddings: embed_query("深度学习")
    Embeddings-->>VectorStore: query_vector

    VectorStore->>Index: similarity_search_with_score(query_vector, k=20)
    Index-->>VectorStore: 20个候选文档及分数<br/>[(doc1, 0.95), (doc2, 0.90), ..., (doc20, 0.60)]

    VectorStore->>MMR: 启动MMR选择算法<br/>lambda_mult=0.7 (更重视相关性)

    MMR->>MMR: 选择第一个文档<br/>最相关的文档 (doc1, 0.95)
    MMR->>MMR: selected_docs = [doc1]<br/>selected_vectors = [vec1]

    loop 选择剩余4个文档
        MMR->>MMR: 遍历剩余候选文档

        loop 计算每个候选的MMR分数
            MMR->>Calculator: 计算与查询的相关性<br/>relevance_score = cosine_sim(query, candidate)

            MMR->>Calculator: 计算与已选文档的最大相似度

            loop 遍历已选文档
                Calculator->>Calculator: sim = cosine_sim(candidate, selected_doc)
                Calculator->>Calculator: max_similarity = max(max_similarity, sim)
            end

            MMR->>MMR: 计算MMR分数<br/>mmr_score = λ × relevance - (1-λ) × max_similarity<br/>= 0.7 × 0.85 - 0.3 × 0.6 = 0.415
        end

        MMR->>MMR: 选择MMR分数最高的候选<br/>平衡相关性和多样性
        MMR->>MMR: 添加到已选列表<br/>selected_docs.append(best_candidate)
    end

    MMR-->>VectorStore: 返回5个平衡的文档<br/>[doc1, doc5, doc3, doc8, doc12]

    VectorStore-->>User: 最终MMR结果<br/>既相关又多样化
```

**MMR算法核心**：

```python
def calculate_mmr_score(
    relevance_score: float,
    max_similarity_to_selected: float,
    lambda_mult: float
) -> float:
    """计算MMR分数。

    Args:
        relevance_score: 与查询的相关性分数 (0-1)
        max_similarity_to_selected: 与已选文档的最大相似度 (0-1)
        lambda_mult: 相关性权重 (0-1)

    Returns:
        MMR分数，越高越好
    """
    return (
        lambda_mult * relevance_score -
        (1.0 - lambda_mult) * max_similarity_to_selected
    )

# lambda_mult参数影响：
# - lambda_mult = 1.0: 完全基于相关性（忽略多样性）
# - lambda_mult = 0.0: 完全基于多样性（忽略相关性）
# - lambda_mult = 0.5: 平衡相关性和多样性
# - lambda_mult = 0.7: 更重视相关性
```

---

## 4. Retriever使用场景

### 4.1 VectorStoreRetriever 检索流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Retriever as VectorStoreRetriever
    participant VectorStore
    participant Callbacks as CallbackManager

    User->>Retriever: get_relevant_documents("LangChain框架特性")

    Retriever->>Callbacks: on_retriever_start(query="LangChain框架特性")

    Retriever->>Retriever: 检查搜索类型和参数<br/>search_type="similarity"<br/>search_kwargs={"k": 6}

    alt search_type == "similarity"
        Retriever->>VectorStore: similarity_search(query, k=6)
        VectorStore-->>Retriever: 相似性搜索结果

    else search_type == "similarity_score_threshold"
        Retriever->>VectorStore: similarity_search_with_score(query, **search_kwargs)
        VectorStore->>VectorStore: 应用分数阈值过滤<br/>score_threshold=0.8
        VectorStore-->>Retriever: 过滤后的结果

    else search_type == "mmr"
        Retriever->>VectorStore: max_marginal_relevance_search(query, **search_kwargs)
        VectorStore->>VectorStore: MMR算法选择<br/>lambda_mult=0.5, fetch_k=20
        VectorStore-->>Retriever: MMR结果
    end

    Retriever->>Callbacks: on_retriever_end(documents=results)

    Retriever-->>User: List[Document] - 检索到的相关文档
```

**检索器配置示例**：

```python
# 不同搜索类型的配置
configs = {
    "similarity": {
        "search_type": "similarity",
        "search_kwargs": {"k": 6}
    },
    "threshold": {
        "search_type": "similarity_score_threshold",
        "search_kwargs": {
            "k": 10,
            "score_threshold": 0.8
        }
    },
    "mmr": {
        "search_type": "mmr",
        "search_kwargs": {
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    }
}

# 创建不同类型的检索器
for name, config in configs.items():
    retriever = vectorstore.as_retriever(**config)
    print(f"{name} retriever created")
```

---

### 4.2 EnsembleRetriever 集成检索流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Ensemble as EnsembleRetriever
    participant VectorRet as VectorRetriever
    participant KeywordRet as KeywordRetriever
    participant Fusion as RankFusion

    User->>Ensemble: get_relevant_documents("Python机器学习库")

    Ensemble->>Ensemble: 检查权重配置<br/>weights=[0.7, 0.3] (向量70%, 关键词30%)

    par 并行检索
        Ensemble->>VectorRet: get_relevant_documents("Python机器学习库")
        VectorRet->>VectorRet: 向量相似性搜索
        VectorRet-->>Ensemble: vector_results = [doc1, doc3, doc5, doc2, doc7]
    and
        Ensemble->>KeywordRet: get_relevant_documents("Python机器学习库")
        KeywordRet->>KeywordRet: 关键词匹配搜索<br/>TF-IDF或BM25算法
        KeywordRet-->>Ensemble: keyword_results = [doc2, doc1, doc6, doc4, doc8]
    end

    Ensemble->>Fusion: 执行排序融合算法<br/>Reciprocal Rank Fusion (RRF)

    Fusion->>Fusion: 计算每个文档的RRF分数

    loop 处理向量搜索结果
        Fusion->>Fusion: 文档rank=1: RRF += 0.7/(60+1) = 0.0115<br/>文档rank=2: RRF += 0.7/(60+2) = 0.0113
    end

    loop 处理关键词搜索结果
        Fusion->>Fusion: 文档rank=1: RRF += 0.3/(60+1) = 0.0049<br/>文档rank=2: RRF += 0.3/(60+2) = 0.0048
    end

    Fusion->>Fusion: 合并相同文档的分数<br/>doc1: 0.0113 + 0.0048 = 0.0161<br/>doc2: 0.0108 + 0.0049 = 0.0157

    Fusion->>Fusion: 按最终RRF分数排序
    Fusion-->>Ensemble: 融合后的排序结果

    Ensemble-->>User: 最终检索结果<br/>结合了向量和关键词搜索的优势
```

**RRF算法实现**：

```python
def reciprocal_rank_fusion(
    doc_lists: List[List[Document]],
    weights: List[float],
    c: int = 60
) -> List[Document]:
    """倒数排名融合算法。

    Args:
        doc_lists: 多个检索器的结果列表
        weights: 各检索器的权重
        c: RRF常数，通常为60

    Returns:
        融合后的文档列表
    """
    doc_scores = {}

    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list, 1):
            doc_key = doc.page_content  # 使用内容作为唯一标识
            rrf_score = weight / (c + rank)

            if doc_key in doc_scores:
                doc_scores[doc_key][1] += rrf_score
            else:
                doc_scores[doc_key] = [doc, rrf_score]

    # 按分数排序
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs]
```

---

## 5. 缓存优化场景

### 5.1 向量缓存命中流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant VectorStore
    participant Cache as VectorCache
    participant Embeddings
    participant Hasher

    User->>VectorStore: similarity_search("什么是机器学习？")

    VectorStore->>Hasher: 生成查询缓存键<br/>hash("什么是机器学习？")
    Hasher-->>VectorStore: cache_key = "ml_query_abc123"

    VectorStore->>Cache: get_cached_vector(cache_key)

    alt 缓存命中
        Cache->>Cache: 检查缓存项是否过期<br/>TTL = 3600秒

        alt 未过期
            Cache->>Cache: 更新LRU访问顺序<br/>hit_count++
            Cache-->>VectorStore: cached_vector = [0.1, 0.2, ...]

            VectorStore->>VectorStore: 使用缓存向量进行搜索<br/>跳过embedding调用
            VectorStore-->>User: 搜索结果 (快速返回, ~10ms)

        else 已过期
            Cache->>Cache: 删除过期缓存项<br/>miss_count++
            Cache-->>VectorStore: None
        end

    else 缓存未命中
        Cache->>Cache: miss_count++
        Cache-->>VectorStore: None

        VectorStore->>Embeddings: embed_query("什么是机器学习？")
        Embeddings-->>VectorStore: query_vector = [0.1, 0.2, ...]

        VectorStore->>Cache: 缓存新向量<br/>put(cache_key, query_vector, ttl=3600)

        Cache->>Cache: 检查缓存容量<br/>如果满了则淘汰LRU项
        Cache->>Cache: 存储新向量并更新访问顺序

        VectorStore->>VectorStore: 执行向量搜索
        VectorStore-->>User: 搜索结果 (较慢, ~200ms)
    end
```

**缓存性能对比**：

| 场景 | 响应时间 | 说明 |
|-----|---------|------|
| 缓存命中 | ~10ms | 直接使用缓存向量 |
| 缓存未命中 | ~200ms | 需要调用embedding API |
| 首次查询 | ~200ms | 必须生成向量 |
| 重复查询 | ~10ms | 从缓存获取向量 |

---

### 5.2 批量操作优化流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant BatchProcessor
    participant VectorStore
    participant Embeddings
    participant Index

    User->>BatchProcessor: add_documents_batch(1000_documents, batch_size=100)

    BatchProcessor->>BatchProcessor: 将文档分批<br/>10个批次，每批100个文档

    loop 处理每个批次 (10次)
        BatchProcessor->>BatchProcessor: 准备批次数据<br/>batch = documents[i:i+100]

        BatchProcessor->>Embeddings: embed_documents(batch_texts)
        Note over Embeddings: 批量调用embedding API<br/>减少网络开销
        Embeddings-->>BatchProcessor: batch_embeddings (100个向量)

        BatchProcessor->>VectorStore: add_texts_with_embeddings(texts, embeddings, metadatas)

        VectorStore->>Index: 批量添加向量<br/>index.add(embedding_matrix)
        Index->>Index: 批量更新索引结构<br/>比逐个添加更高效
        Index-->>VectorStore: 批次添加完成

        VectorStore-->>BatchProcessor: batch_ids = ["id1", ..., "id100"]

        BatchProcessor->>BatchProcessor: 更新进度<br/>已处理: (i+1)*100 / 1000
        BatchProcessor-->>User: 进度通知: "已处理 100/1000 文档"
    end

    BatchProcessor->>BatchProcessor: 合并所有批次ID
    BatchProcessor-->>User: 完成！总共添加1000个文档<br/>all_document_ids
```

**批量优化效果**：

```python
# 性能对比示例
import time

def add_documents_individually(vectorstore, documents):
    """逐个添加文档。"""
    start_time = time.time()
    ids = []

    for doc in documents:
        doc_ids = vectorstore.add_texts([doc.page_content], [doc.metadata])
        ids.extend(doc_ids)

    return ids, time.time() - start_time

def add_documents_batch(vectorstore, documents, batch_size=100):
    """批量添加文档。"""
    start_time = time.time()
    all_ids = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        batch_ids = vectorstore.add_texts(texts, metadatas)
        all_ids.extend(batch_ids)

    return all_ids, time.time() - start_time

# 1000个文档的性能对比：
# 逐个添加：~120秒
# 批量添加：~15秒
# 性能提升：8倍
```

---

## 6. 异步操作场景

### 6.1 异步文档添加流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant AsyncProcessor
    participant VectorStore
    participant AsyncEmbeddings
    participant TaskPool as AsyncTaskPool

    User->>AsyncProcessor: await add_documents_async(documents_stream)

    AsyncProcessor->>AsyncProcessor: 创建异步任务池<br/>max_concurrent_tasks = 5

    loop 处理文档流
        AsyncProcessor->>AsyncProcessor: 收集批次文档<br/>batch_size = 50

        AsyncProcessor->>TaskPool: 创建异步任务<br/>process_batch_async(batch)

        par 并发处理多个批次
            TaskPool->>AsyncEmbeddings: await embed_documents_async(batch1)
            AsyncEmbeddings-->>TaskPool: embeddings1
        and
            TaskPool->>AsyncEmbeddings: await embed_documents_async(batch2)
            AsyncEmbeddings-->>TaskPool: embeddings2
        and
            TaskPool->>AsyncEmbeddings: await embed_documents_async(batch3)
            AsyncEmbeddings-->>TaskPool: embeddings3
        end

        par 并发存储
            TaskPool->>VectorStore: await add_vectors_async(embeddings1, docs1)
        and
            TaskPool->>VectorStore: await add_vectors_async(embeddings2, docs2)
        and
            TaskPool->>VectorStore: await add_vectors_async(embeddings3, docs3)
        end

        TaskPool-->>AsyncProcessor: 批次处理完成
        AsyncProcessor-->>User: yield 进度更新
    end

    AsyncProcessor-->>User: 所有文档处理完成
```

**异步处理优势**：
- 并发向量化：同时处理多个批次
- 非阻塞I/O：不阻塞主线程
- 资源利用：充分利用网络和计算资源
- 实时反馈：流式处理进度更新

---

## 7. 错误处理和恢复场景

### 7.1 搜索失败恢复流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant VectorStore
    participant Index
    participant FallbackStore as BackupVectorStore
    participant ErrorHandler

    User->>VectorStore: similarity_search("查询内容")

    VectorStore->>Index: 执行向量搜索
    Index-->>VectorStore: IndexError("索引损坏")

    VectorStore->>ErrorHandler: 处理搜索错误

    ErrorHandler->>ErrorHandler: 分析错误类型<br/>IndexError -> 索引问题

    alt 有备份存储
        ErrorHandler->>FallbackStore: 尝试备份存储搜索<br/>similarity_search(query)

        alt 备份搜索成功
            FallbackStore-->>ErrorHandler: 搜索结果
            ErrorHandler->>ErrorHandler: 记录主存储故障<br/>触发修复任务
            ErrorHandler-->>VectorStore: 返回备份结果
            VectorStore-->>User: 搜索结果 (来自备份)

        else 备份也失败
            FallbackStore-->>ErrorHandler: 备份搜索也失败
            ErrorHandler->>ErrorHandler: 降级处理<br/>返回空结果或缓存结果
            ErrorHandler-->>VectorStore: 降级结果
            VectorStore-->>User: 有限的搜索结果
        end

    else 无备份存储
        ErrorHandler->>ErrorHandler: 尝试索引重建<br/>或返回错误信息

        alt 能够快速修复
            ErrorHandler->>Index: 重建损坏的索引部分
            Index-->>ErrorHandler: 修复完成
            ErrorHandler->>VectorStore: 重试搜索
            VectorStore-->>User: 搜索结果 (延迟返回)

        else 无法快速修复
            ErrorHandler-->>VectorStore: 返回用户友好错误
            VectorStore-->>User: "搜索服务暂时不可用，请稍后重试"
        end
    end
```

**错误恢复策略**：

```python
class VectorStoreErrorHandler:
    """向量存储错误处理器。"""

    def __init__(self, fallback_store=None, max_retries=3):
        self.fallback_store = fallback_store
        self.max_retries = max_retries
        self.error_stats = defaultdict(int)

    async def handle_search_error(
        self,
        error: Exception,
        query: str,
        search_func: Callable,
        **kwargs
    ) -> List[Document]:
        """处理搜索错误。"""
        error_type = type(error).__name__
        self.error_stats[error_type] += 1

        if isinstance(error, (ConnectionError, TimeoutError)):
            # 网络错误 - 重试
            return await self._retry_with_backoff(search_func, query, **kwargs)

        elif isinstance(error, IndexError):
            # 索引错误 - 使用备份
            if self.fallback_store:
                return await self.fallback_store.asimilarity_search(query, **kwargs)
            else:
                raise VectorStoreException("索引损坏且无备份存储")

        elif isinstance(error, MemoryError):
            # 内存错误 - 降级处理
            kwargs['k'] = min(kwargs.get('k', 4), 2)  # 减少返回数量
            return await self._retry_with_backoff(search_func, query, **kwargs)

        else:
            # 未知错误 - 直接抛出
            raise error

    async def _retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """带退避的重试。"""
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(2 ** attempt)  # 指数退避
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                continue
```

---

## 8. 总结

本文档详细展示了 **VectorStores 和 Retrievers 模块**的关键执行时序：

1. **文档管理**：文档添加、向量化、存储的完整流程
2. **相似性搜索**：标准搜索、阈值过滤、分数计算机制
3. **MMR算法**：平衡相关性和多样性的智能选择过程
4. **检索器使用**：不同类型检索器的调用和配置流程
5. **性能优化**：缓存命中、批量处理、异步操作
6. **错误处理**：搜索失败的恢复和降级策略

每张时序图包含：
- 详细的参与者交互过程
- 关键算法和计算步骤
- 性能优化点和最佳实践
- 错误处理和恢复机制
- 实际代码示例和配置方法

这些时序图帮助开发者深入理解向量存储和检索系统的内部工作机制，为构建高效、可靠的RAG（检索增强生成）应用提供指导。VectorStores和Retrievers是现代AI应用的核心基础设施，正确理解其执行流程对构建智能知识检索系统至关重要。
