---
title: "VoiceHelper-06-GraphRAG服务"
date: 2025-10-10T10:06:00+08:00
draft: false
tags: ["VoiceHelper", "GraphRAG", "知识图谱", "向量检索", "Neo4j", "FAISS"]
categories: ["VoiceHelper", "源码剖析"]
description: "VoiceHelper GraphRAG服务详细设计，包含文档摄取、实体提取、图谱构建、混合检索、智能问答完整实现与优化策略"
series: ["VoiceHelper源码剖析"]
weight: 6
---

# VoiceHelper-06-GraphRAG服务

## 一、模块概览

### 1.1 职责定位

GraphRAG服务是VoiceHelper的核心算法服务之一,专注于构建和检索知识图谱,提供企业级的文档理解和智能问答能力。

**核心职责**:
- **文档摄取**:解析多种格式文档,进行语义分块和向量化
- **实体提取**:从文本中提取实体和关系,支持NER+LLM双路径
- **图谱构建**:在Neo4j中构建知识图谱,支持社区检测和图谱分析
- **混合检索**:整合向量检索、图谱检索、BM25检索,提供高质量结果
- **智能问答**:基于检索增强生成(RAG)提供精准答案

**技术特性**:
- FastAPI异步框架,高并发处理
- 语义分块,保持文本语义完整性
- 查询改写,提升召回率
- RRF融合+CrossEncoder重排,提升精排准确性
- 支持社区检测,发现知识聚类
- 语义缓存,加速重复查询

### 1.2 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      GraphRAG Service                           │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application                                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │   Routes      │  │  Middleware   │  │  Lifespan     │      │
│  │  (routes.py)  │  │  (CORS, Log)  │  │  Management   │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│  Core Modules                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Ingest Pipeline                                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │   Document   │→ │   Chunker    │→ │   Vector     │ │   │
│  │  │   Processor  │  │  (Semantic)  │  │   Store      │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  │         ↓                                               │   │
│  │  ┌──────────────┐  ┌──────────────┐                   │   │
│  │  │   Entity     │→ │    Graph     │                   │   │
│  │  │   Extractor  │  │   Builder    │                   │   │
│  │  └──────────────┘  └──────────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Retrieval Pipeline                                     │   │
│  │  ┌──────────────┐                                       │   │
│  │  │    Query     │                                       │   │
│  │  │   Rewriter   │                                       │   │
│  │  └───────┬──────┘                                       │   │
│  │          ↓                                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │   Vector     │  │    Graph     │  │    BM25      │ │   │
│  │  │  Retriever   │  │  Retriever   │  │  Retriever   │ │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │   │
│  │         └──────────────────┼──────────────────┘         │   │
│  │                            ↓                             │   │
│  │                  ┌──────────────────┐                   │   │
│  │                  │   RRF Fusion     │                   │   │
│  │                  │     Ranker       │                   │   │
│  │                  └─────────┬────────┘                   │   │
│  │                            ↓                             │   │
│  │                  ┌──────────────────┐                   │   │
│  │                  │  CrossEncoder    │                   │   │
│  │                  │    Reranker      │                   │   │
│  │                  └──────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Community Detection                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐                   │   │
│  │  │   Louvain/   │→ │   Subgraph   │                   │   │
│  │  │    Leiden    │  │   Analysis   │                   │   │
│  │  └──────────────┘  └──────────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  External Dependencies                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Neo4j   │  │  FAISS   │  │  Redis   │  │   LLM    │      │
│  │ (Graph)  │  │ (Vector) │  │ (Cache)  │  │ Router   │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

**设计特点**:
- **分层架构**:API层、服务层、数据层清晰分离
- **管道模式**:摄取和检索均采用管道设计,模块可插拔
- **异步并发**:多路检索并行执行,RRF融合
- **降级策略**:各环节都有fallback,确保鲁棒性

---

## 二、API规格说明

### 2.1 文档摄取API

#### POST /api/v1/ingest

**功能描述**:
摄取文档内容,进行分块、实体提取、图谱构建和向量索引。支持后台异步处理。

**请求参数**:
```python
class IngestRequest(BaseModel):
    content: str              # 文档内容(必填)
    doc_type: str = "text"    # 文档类型: text/markdown/pdf/html
    title: Optional[str]      # 文档标题
    metadata: Optional[Dict]  # 元数据(可选)
    build_graph: bool = True  # 是否构建图谱
```

**响应格式**:
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "doc_id": "doc_1697012345678",
        "status": "processing",
        "message": "文档摄取任务已创建",
        "elapsed_time": 0.123
    }
}
```

**核心代码**:
```python
@router.post("/ingest")
async def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
    # 初始化服务组件
    entity_extractor = EntityRelationExtractor(llm_router_url=llm_router_url)
    graph_builder = Neo4jManager(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    ingest_service = IngestService(
        entity_extractor=entity_extractor,
        graph_builder=graph_builder
    )
    
    doc_id = request.title or f"doc_{int(time.time() * 1000)}"
    
    # 后台异步处理
    async def process_ingest_task():
        result = await ingest_service.ingest_document(
            content=request.content,
            doc_type=request.doc_type,
            doc_id=doc_id,
            metadata=request.metadata or {},
            build_graph=request.build_graph
        )
        # 处理结果记录
    
    background_tasks.add_task(process_ingest_task)
    return success_response({...})
```

**调用链路**:
```
1. POST /api/v1/ingest
   ↓
2. IngestService.ingest_document()
   ├─→ DocumentProcessor.process_document()      # 文档解析和分块
   │   ├─→ _parse_document()                     # 格式解析
   │   └─→ _semantic_chunk()                     # 语义分块
   ├─→ VectorStore.add_texts()                   # 向量索引
   └─→ _build_graph()                            # 图谱构建
       ├─→ EntityExtractor.extract()             # 实体提取
       │   ├─→ _extract_with_ner()               # NER路径
       │   └─→ _extract_with_llm()               # LLM路径
       ├─→ Neo4jManager.batch_create_entities()  # 批量创建实体
       └─→ Neo4jManager.batch_create_relations() # 批量创建关系
```

**最佳实践**:
- 大文档建议分批摄取,避免超时
- 使用`build_graph=False`可跳过图谱构建,加快处理速度
- 为文档添加丰富的元数据,便于后续过滤检索

---

### 2.2 智能检索API

#### POST /api/v1/query

**功能描述**:
执行混合检索,整合向量、图谱、BM25三路召回,经RRF融合和CrossEncoder重排后返回相关文档片段。

**请求参数**:
```python
class QueryRequest(BaseModel):
    query: str                         # 查询文本(必填)
    mode: str = "hybrid"               # 检索模式: vector_only/graph_only/hybrid
    top_k: int = Field(10, ge=1, le=100)  # 返回结果数量
```

**响应格式**:
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "query": "什么是GraphRAG?",
        "mode": "hybrid",
        "results": [
            {
                "document_id": "doc-123",
                "chunk_id": "doc-123_5",
                "text": "GraphRAG是一种结合知识图谱的检索增强生成技术...",
                "score": 0.95,
                "source": "vector",
                "metadata": {"doc_type": "pdf", "page": 5}
            }
        ],
        "total": 10,
        "elapsed_time": 0.28,
        "stats": {
            "total_queries": 142,
            "avg_latency_ms": 265
        }
    }
}
```

**核心代码**:
```python
@router.post("/query")
async def query_documents(request: QueryRequest):
    # 获取智能检索器(懒加载)
    retriever = get_intelligent_retriever()
    
    # 根据mode设置权重
    weights = {
        'vector_only': {'vector': 1.0, 'graph': 0.0, 'bm25': 0.0},
        'graph_only': {'vector': 0.0, 'graph': 1.0, 'bm25': 0.0},
        'hybrid': {'vector': 0.5, 'graph': 0.3, 'bm25': 0.2}
    }[request.mode]
    
    # 执行智能检索
    results = await retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        use_rewrite=True,    # 查询改写
        use_rerank=True,     # CrossEncoder重排
        weights=weights
    )
    
    response_results = [r.to_dict() for r in results]
    return success_response({...})
```

**调用链路**:
```
1. POST /api/v1/query
   ↓
2. IntelligentRetriever.retrieve()
   ├─→ _rewrite_query()                        # 查询改写
   │   └─→ QueryRewriter.rewrite()
   │       ├─→ _apply_synonyms()               # 同义词替换
   │       ├─→ _expand_query()                 # 查询扩展
   │       └─→ _decompose_query()              # 问题分解
   ├─→ _multi_recall()                         # 并行多路召回
   │   ├─→ _vector_retrieve()                  # 向量检索(FAISS)
   │   ├─→ _graph_retrieve()                   # 图谱检索(Neo4j)
   │   └─→ _bm25_retrieve()                    # BM25检索
   ├─→ _fusion_ranking()                       # RRF融合
   │   └─→ RRFFusionRanker.fuse()
   │       └─→ 计算RRF分数: weight/(k+rank)
   └─→ _rerank_results()                       # CrossEncoder重排
       └─→ CrossEncoderReranker.rerank()
           └─→ model.predict(query-doc pairs)
```

**最佳实践**:
- 一般场景使用`mode=hybrid`,平衡召回和精度
- 纯语义匹配场景使用`mode=vector_only`
- 需要推理和多跳查询时使用`mode=graph_only`
- `top_k`设置为5-10即可满足大部分需求

---

### 2.3 社区检测API

#### POST /api/v1/community/detect

**功能描述**:
在知识图谱中执行社区检测,发现实体聚类,支持Louvain和Leiden算法。

**请求参数**:
```python
class CommunityDetectRequest(BaseModel):
    algorithm: str = "louvain"          # 算法: leiden/louvain
    resolution: float = Field(1.0, ge=0.1, le=10.0)  # 分辨率参数
```

**响应格式**:
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "algorithm": "louvain",
        "communities": [
            {
                "community_id": 0,
                "size": 15,
                "nodes": ["实体1", "实体2", ...],
                "summary": "包含: 实体1, 实体2 等15个节点",
                "llm_summary": "这个社区主要讨论机器学习相关概念..."
            }
        ],
        "stats": {
            "total_communities": 5,
            "total_nodes": 100,
            "total_edges": 250,
            "avg_community_size": 20.0
        },
        "modularity": 0.82,
        "elapsed_time": 1.234
    }
}
```

**核心代码**:
```python
@router.post("/community/detect")
async def detect_communities(request: CommunityDetectRequest):
    # 初始化组件
    graph_manager = Neo4jManager(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    detector = CommunityDetector(graph_manager)
    
    # 执行社区检测
    result = await detector.detect_communities(
        algorithm=request.algorithm,
        resolution=request.resolution,
        min_community_size=3
    )
    
    # 为前5个社区生成LLM摘要(可选)
    if llm_router_url:
        extractor = EntityRelationExtractor(llm_router_url=llm_router_url)
        for community in result["communities"][:5]:
            nodes_str = ", ".join(community["nodes"][:10])
            summary = await extractor._generate_community_summary(nodes_str)
            community["llm_summary"] = summary
    
    return success_response({...})
```

**调用链路**:
```
1. POST /api/v1/community/detect
   ↓
2. CommunityDetector.detect_communities()
   ├─→ graph_manager.export_to_networkx()      # 导出NetworkX图
   ├─→ _detect_communities_sync()              # 社区检测(在executor中运行)
   │   ├─→ Louvain算法 (python-louvain)
   │   ├─→ Leiden算法 (leidenalg)
   │   └─→ Label Propagation (NetworkX内置)
   ├─→ _format_communities()                   # 格式化社区
   ├─→ nx.algorithms.community.modularity()    # 计算模块度
   └─→ _compute_stats()                        # 统计信息
```

**最佳实践**:
- Louvain算法速度快,适合大规模图谱
- Leiden算法质量更高,适合中小规模图谱
- `resolution`参数控制社区粒度:值越大,社区越小越多
- 定期执行社区检测,发现知识结构变化

---

### 2.4 图谱统计API

#### GET /api/v1/stats

**功能描述**:
获取知识图谱的统计信息,包括节点数、关系数、类型分布等。

**响应格式**:
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "total_nodes": 1523,
        "total_relationships": 4287,
        "node_labels": {
            "Entity": 1200,
            "Concept": 323
        },
        "relationship_types": {
            "RELATED_TO": 2100,
            "BELONGS_TO": 1200,
            "IS_A": 987
        }
    }
}
```

**核心代码**:
```python
@router.get("/stats")
async def get_graph_stats():
    graph_manager = Neo4jManager(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    stats = await graph_manager.get_stats()
    return success_response(stats)
```

---

## 三、核心功能实现

### 3.1 文档摄取与处理

#### 3.1.1 文档处理器

`DocumentProcessor`负责解析文档和语义分块。

**关键特性**:
- 支持多种格式:text、markdown、html、pdf
- 语义分块:保持句子和段落完整性
- 自适应块大小:根据文本特征动态调整
- 块重叠:提升检索召回率

**核心代码**:
```python
class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 500,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 50
    ):
        # 初始化参数
    
    async def process_document(
        self,
        content: str,
        doc_type: str = "text",
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        # 1. 解析文档
        parsed_content = await self._parse_document(content, doc_type)
        
        # 2. 语义分块
        chunks = await self._semantic_chunk(parsed_content, doc_id, metadata)
        
        return chunks
    
    async def _parse_document(self, content: str, doc_type: str) -> str:
        if doc_type == "markdown":
            # 移除markdown标记
            text = re.sub(r'```[^\n]*\n.*?```', '', content, flags=re.DOTALL)
            text = re.sub(r'`[^`]*`', '', text)
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            return text.strip()
        elif doc_type == "html":
            # 移除HTML标签
            text = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()
        # 其他格式处理...
    
    def _create_chunks(
        self,
        text: str,
        sentences: List[str],
        doc_id: Optional[str],
        metadata: Optional[Dict]
    ) -> List[DocumentChunk]:
        chunks = []
        current_chunk_sentences = []
        current_chunk_size = 0
        
        for sentence in sentences:
            # 检查是否需要创建新块
            if current_chunk_size >= self.chunk_size or \
               current_chunk_size + len(sentence) > self.max_chunk_size:
                # 创建块
                chunk_text = '。'.join(current_chunk_sentences)
                chunk = DocumentChunk(...)
                chunks.append(chunk)
                
                # 保留重叠句子
                current_chunk_sentences = current_chunk_sentences[-self.chunk_overlap:]
            
            current_chunk_sentences.append(sentence)
            current_chunk_size += len(sentence)
        
        return chunks
```

**数据流图**:
```
文档内容
   ↓
格式解析 (markdown/html/pdf → 纯文本)
   ↓
句子分割 (正则分割)
   ↓
语义分块 (根据chunk_size和语义边界)
   ↓
DocumentChunk列表
```

---

#### 3.1.2 实体关系提取

`EntityRelationExtractor`支持NER和LLM双路径提取。

**核心代码**:
```python
class EntityRelationExtractor:
    def __init__(
        self,
        llm_router_url: str,
        use_ner: bool = True,
        use_llm: bool = True
    ):
        # 加载spaCy NER模型
        if use_ner:
            import spacy
            self.nlp = spacy.load("zh_core_web_lg")
    
    async def extract(
        self,
        text: str,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        entities = []
        relations = []
        
        # 路径1: NER提取
        if self.use_ner:
            ner_entities = await self._extract_with_ner(text)
            entities.extend(ner_entities)
        
        # 路径2: LLM提取
        if self.use_llm:
            llm_results = await self._extract_with_llm(text)
            entities.extend(llm_results.get("entities", []))
            relations.extend(llm_results.get("relations", []))
        
        # 去重和过滤
        entities = self._deduplicate_entities(entities)
        relations = self._deduplicate_relations(relations)
        entities = [e for e in entities if e.get("confidence", 1.0) >= min_confidence]
        
        return {"entities": entities, "relations": relations}
    
    async def _extract_with_llm(self, text: str) -> Dict:
        # 构建提示词
        prompt = f"""从以下文本中提取实体和关系。

文本: {text[:1000]}

要求:
1. 提取所有重要实体（人物、地点、组织、概念等）
2. 提取实体之间的关系
3. 用JSON格式返回

格式:
{{
    "entities": [{{"name": "实体名", "type": "类型", "description": "描述", "confidence": 0.9}}],
    "relations": [{{"source": "实体1", "relation": "关系", "target": "实体2", "confidence": 0.8}}]
}}"""
        
        # 调用LLM Router Service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.llm_router_url}/api/v1/chat",
                json={
                    "messages": [
                        {"role": "system", "content": "你是一个专业的知识图谱构建助手。返回JSON格式。"},
                        {"role": "user", "content": prompt}
                    ],
                    "task_type": "reasoning",
                    "temperature": 0.3
                }
            )
        
        result = response.json()
        content = result.get("data", {}).get("content", "{}")
        return json.loads(content)
```

**双路径融合策略**:
- NER路径:快速,适合识别常见实体类型
- LLM路径:准确,能理解复杂语义和领域特定实体
- 去重合并:相同实体保留更高置信度的结果

---

#### 3.1.3 图谱构建

`Neo4jManager`负责将实体和关系写入Neo4j。

**核心代码**:
```python
class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    async def batch_create_entities(self, entities: List[Dict]) -> List[str]:
        async with self.driver.session() as session:
            result = await session.run(
                """
                UNWIND $entities as entity
                MERGE (e:Entity {name: entity.name})
                SET e.type = entity.type
                SET e += entity.properties
                SET e.updated_at = datetime()
                RETURN id(e) as entity_id
                """,
                entities=entities
            )
            
            entity_ids = []
            async for record in result:
                entity_ids.append(str(record["entity_id"]))
        
        return entity_ids
    
    async def batch_create_relations(self, relations: List[Dict]) -> int:
        async with self.driver.session() as session:
            result = await session.run(
                """
                UNWIND $relations as rel
                MATCH (a:Entity {name: rel.source})
                MATCH (b:Entity {name: rel.target})
                MERGE (a)-[r:RELATION {type: rel.relation}]->(b)
                SET r.updated_at = datetime()
                RETURN count(r) as count
                """,
                relations=relations
            )
            
            record = await result.single()
            return record["count"] if record else 0
```

**批量操作优化**:
- 使用`UNWIND`批量处理,减少网络往返
- `MERGE`确保幂等性,重复摄取不会重复创建
- 添加时间戳,支持增量更新

---

### 3.2 智能检索系统

#### 3.2.1 查询改写

`QueryRewriter`提升召回率。

**核心策略**:
1. **同义词替换**:基于词典替换同义词
2. **查询扩展**:使用LLM添加相关术语
3. **问题分解**:将复杂问题分解为子问题

**核心代码**:
```python
class QueryRewriter:
    async def rewrite_query(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> RewrittenQuery:
        rewritten_queries = [query]  # 始终包含原始查询
        
        # 1. 同义词替换
        if self.enable_synonym:
            synonym_queries = self._apply_synonyms(query)
            rewritten_queries.extend(synonym_queries[:1])
        
        # 2. 查询扩展
        if self.enable_expansion and self.llm_client:
            expanded_queries = await self._expand_query(query, context)
            rewritten_queries.extend(expanded_queries[:1])
        
        # 3. 问题分解
        if self.enable_decomposition and self._is_complex_query(query):
            decomposed_queries = await self._decompose_query(query, context)
            rewritten_queries.extend(decomposed_queries[:1])
        
        # 去重并限制数量
        rewritten_queries = list(dict.fromkeys(rewritten_queries))[:self.max_rewrites + 1]
        
        return RewrittenQuery(original=query, rewritten=rewritten_queries, ...)
```

---

#### 3.2.2 混合检索

`IntelligentRetriever`整合三路召回。

**核心代码**:
```python
class IntelligentRetriever:
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        weights: Optional[Dict[str, float]] = None
    ) -> List[RetrievalResult]:
        # 默认权重
        if weights is None:
            weights = {'vector': 0.5, 'graph': 0.3, 'bm25': 0.2}
        
        # 1. 查询改写
        queries = await self._rewrite_query(query, use_rewrite)
        
        # 2. 并行多路召回
        recall_results = await self._multi_recall(queries, query, top_k, weights, filters=None)
        vector_results = recall_results['vector']
        graph_results = recall_results['graph']
        bm25_results = recall_results['bm25']
        
        # 3. RRF融合
        fused_results = await self._fusion_ranking(
            vector_results, graph_results, bm25_results,
            weights, top_k
        )
        
        # 4. CrossEncoder重排
        final_results = await self._rerank_results(query, fused_results, top_k, use_rerank)
        
        return final_results
    
    async def _multi_recall(self, queries, original_query, top_k, weights, filters):
        recall_top_k = top_k * 4  # 召回4倍候选
        
        tasks = []
        if weights.get('vector', 0) > 0:
            tasks.append(self._vector_retrieve(queries, recall_top_k, filters))
        else:
            tasks.append(self._empty_results())
        
        if weights.get('graph', 0) > 0:
            tasks.append(self._graph_retrieve(original_query, recall_top_k, filters))
        else:
            tasks.append(self._empty_results())
        
        if weights.get('bm25', 0) > 0:
            tasks.append(self._bm25_retrieve(original_query, recall_top_k, filters))
        else:
            tasks.append(self._empty_results())
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'vector': results[0] if not isinstance(results[0], Exception) else [],
            'graph': results[1] if not isinstance(results[1], Exception) else [],
            'bm25': results[2] if not isinstance(results[2], Exception) else []
        }
```

**检索流程图**:
```
用户查询
   ↓
查询改写 (3-5个变体)
   ↓
┌───────────────┬───────────────┬───────────────┐
│               │               │               │
向量检索        图谱检索        BM25检索
(FAISS)        (Neo4j)        (Rank-BM25)
│               │               │
└───────────────┴───────────────┴───────────────┘
                 ↓
         RRF融合排序 (Top-K*2候选)
                 ↓
         CrossEncoder重排 (Top-K)
                 ↓
              最终结果
```

---

#### 3.2.3 RRF融合排序

`RRFFusionRanker`实现Reciprocal Rank Fusion算法。

**核心代码**:
```python
class RRFFusionRanker:
    def __init__(self, k: int = 60):
        self.k = k  # RRF参数,论文推荐值60
    
    async def fuse(
        self,
        results_lists: List[Tuple[List[RetrievalResult], float]],
        top_k: int
    ) -> List[RetrievalResult]:
        # 收集所有文档的RRF分数
        rrf_scores = defaultdict(float)
        doc_map = {}  # chunk_id -> RetrievalResult
        
        for results, weight in results_lists:
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id
                
                # RRF公式: score = weight / (k + rank)
                rrf_score = weight / (self.k + rank)
                rrf_scores[chunk_id] += rrf_score
                
                if chunk_id not in doc_map:
                    doc_map[chunk_id] = result
        
        # 按RRF分数排序
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 返回Top-K
        fused_results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            result = doc_map[chunk_id]
            result.score = score
            fused_results.append(result)
        
        return fused_results
```

**RRF优势**:
- 无需归一化各路分数
- 对异构排序结果鲁棒
- 平衡不同来源的贡献

---

#### 3.2.4 CrossEncoder重排

`CrossEncoderReranker`使用深度模型精排。

**核心代码**:
```python
class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        # 构造query-document对
        pairs = [[query, c.text[:self.max_length]] for c in candidates]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 按分数排序
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 更新分数并返回Top-K
        reranked = []
        for candidate, score in scored_candidates[:top_k]:
            candidate.score = float(score)
            reranked.append(candidate)
        
        return reranked
```

**重排效果**:
- CrossEncoder比BiEncoder更准确(+15% MRR)
- 适合精排Top-20~50候选
- 计算成本较高,不适合粗排

---

### 3.3 社区检测

#### 3.3.1 社区检测算法

`CommunityDetector`支持多种算法。

**核心代码**:
```python
class CommunityDetector:
    async def detect_communities(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
        min_community_size: int = 3
    ) -> Dict[str, Any]:
        # 1. 从Neo4j导出NetworkX图
        G = await self.graph_manager.export_to_networkx()
        
        # 2. 执行社区检测(在executor中运行,避免阻塞)
        loop = asyncio.get_event_loop()
        communities_dict = await loop.run_in_executor(
            None,
            self._detect_communities_sync,
            G,
            algorithm,
            resolution
        )
        
        # 3. 转换为社区列表
        communities = self._format_communities(communities_dict, min_community_size)
        
        # 4. 计算模块度
        modularity = nx.algorithms.community.modularity(
            G,
            [set(c["nodes"]) for c in communities]
        )
        
        # 5. 统计信息
        stats = self._compute_stats(communities, G)
        
        return {
            "communities": communities,
            "stats": stats,
            "modularity": modularity
        }
    
    def _detect_communities_sync(self, G, algorithm, resolution):
        if algorithm == "louvain":
            import community as community_louvain
            return community_louvain.best_partition(G, resolution=resolution)
        elif algorithm == "leiden":
            import leidenalg, igraph
            ig_graph = igraph.Graph.from_networkx(G)
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution
            )
            # 转换为字典格式
            communities = {}
            for i, community in enumerate(partition):
                for node_idx in community:
                    node_name = list(G.nodes())[node_idx]
                    communities[node_name] = i
            return communities
        else:  # label_propagation
            communities_generator = nx.algorithms.community.label_propagation_communities(G)
            communities = {}
            for i, community_set in enumerate(communities_generator):
                for node in community_set:
                    communities[node] = i
            return communities
```

**算法对比**:
| 算法 | 速度 | 质量 | 适用场景 |
|------|------|------|---------|
| Louvain | 快 | 中 | 大规模图谱(>10万节点) |
| Leiden | 中 | 高 | 中小规模图谱(<10万节点) |
| Label Propagation | 快 | 低 | 快速原型,对质量要求不高 |

---

## 四、关键数据结构

### 4.1 核心数据模型UML

```
┌─────────────────────────────────────────────────────────────────┐
│                        IngestRequest                            │
├─────────────────────────────────────────────────────────────────┤
│ + content: str                                                  │
│ + doc_type: str = "text"                                        │
│ + title: Optional[str]                                          │
│ + metadata: Optional[Dict]                                      │
│ + build_graph: bool = True                                      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ processes
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        DocumentChunk                            │
├─────────────────────────────────────────────────────────────────┤
│ + text: str                                                     │
│ + chunk_id: str                                                 │
│ + chunk_index: int                                              │
│ + start_pos: int                                                │
│ + end_pos: int                                                  │
│ + metadata: Dict[str, Any]                                      │
│ + semantic_score: float                                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ extracts
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Entity / Relation                        │
├─────────────────────────────────────────────────────────────────┤
│ Entity:                                                         │
│   + name: str                                                   │
│   + type: str                                                   │
│   + description: str                                            │
│   + confidence: float                                           │
│   + source: str (ner/llm)                                       │
│                                                                 │
│ Relation:                                                       │
│   + source: str                                                 │
│   + relation: str                                               │
│   + target: str                                                 │
│   + confidence: float                                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ builds
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Knowledge Graph                          │
├─────────────────────────────────────────────────────────────────┤
│ Nodes (Neo4j):                                                  │
│   - Entity {name, type, properties, updated_at}                 │
│                                                                 │
│ Relationships (Neo4j):                                          │
│   - RELATION {type, properties, updated_at}                     │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                        QueryRequest                             │
├─────────────────────────────────────────────────────────────────┤
│ + query: str                                                    │
│ + mode: str = "hybrid"                                          │
│ + top_k: int = 10                                               │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ rewrites
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        RewrittenQuery                           │
├─────────────────────────────────────────────────────────────────┤
│ + original: str                                                 │
│ + rewritten: List[str]                                          │
│ + strategy: str                                                 │
│ + confidence: float                                             │
│ + metadata: Dict                                                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ retrieves
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        RetrievalResult                          │
├─────────────────────────────────────────────────────────────────┤
│ + document_id: str                                              │
│ + chunk_id: str                                                 │
│ + text: str                                                     │
│ + score: float                                                  │
│ + source: str (vector/graph/bm25)                               │
│ + metadata: Dict[str, Any]                                      │
│                                                                 │
│ + to_dict() → Dict                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 字段说明

#### DocumentChunk
- `text`: 文档块文本内容
- `chunk_id`: 唯一标识,格式为`{doc_id}_{chunk_index}`
- `chunk_index`: 在文档中的序号
- `start_pos`/`end_pos`: 在原文中的位置
- `metadata`: 元数据,包含`doc_id`、`num_sentences`等
- `semantic_score`: 语义完整性分数,1.0表示完整语义块

#### Entity
- `name`: 实体名称,去重的唯一标识
- `type`: 实体类型,如Person、Organization、Concept等
- `description`: 实体描述(LLM提取时提供)
- `confidence`: 置信度(0-1),NER通常0.7,LLM通常0.9
- `source`: 提取来源,ner或llm

#### RetrievalResult
- `document_id`: 所属文档ID
- `chunk_id`: 文档块ID
- `text`: 检索到的文本内容
- `score`: 相关性分数,经过RRF融合或CrossEncoder打分
- `source`: 来源,vector(向量)、graph(图谱)、bm25(关键词)
- `metadata`: 包含原文档的元数据

---

## 五、上游调用链路

### 5.1 Gateway → GraphRAG完整调用链路

#### 5.1.1 Web前端文档上传 → GraphRAG摄取全流程

```mermaid
sequenceDiagram
    autonumber
    participant Browser as 浏览器
    participant WebApp as Web前端
    participant Gateway as API网关<br/>:8080
    participant AuthMW as 认证中间件
    participant DocHandler as 文档Handler
    participant DocSvc as 文档服务<br/>:8082
    participant MinIO as MinIO存储
    participant MQ as 消息队列<br/>(Redis Pub/Sub)
    participant GraphRAG as GraphRAG服务<br/>:8001
    participant Neo4j as Neo4j图数据库
    participant FAISS as FAISS向量库
    
    Note over Browser,FAISS: 阶段1: 文档上传
    Browser->>WebApp: 用户选择文件<br/>点击上传
    WebApp->>Gateway: POST /api/v01/documents/upload<br/>Cookie: access_token<br/>multipart/form-data
    
    Gateway->>AuthMW: 认证检查
    AuthMW->>AuthMW: 提取user_id、tenant_id
    AuthMW-->>Gateway: 认证通过
    
    Gateway->>DocHandler: 路由到文档处理器
    DocHandler->>DocHandler: 文件验证<br/>(类型、大小、MIME)
    DocHandler->>MinIO: PUT /documents/{path}
    MinIO-->>DocHandler: 存储成功
    
    DocHandler->>DocHandler: 保存元数据到PostgreSQL<br/>status='uploaded'
    DocHandler->>MQ: PUBLISH document.uploaded<br/>{doc_id, path, tenant_id}
    DocHandler-->>Gateway: 200 OK<br/>{doc_id, status:"uploaded"}
    Gateway-->>WebApp: 上传成功
    WebApp-->>Browser: 显示:上传成功,处理中
    
    Note over MQ,FAISS: 阶段2: GraphRAG异步摄取
    MQ-->>GraphRAG: SUBSCRIBE document.uploaded
    GraphRAG->>GraphRAG: 消费消息<br/>提取doc_id和path
    
    GraphRAG->>MinIO: GET /documents/{path}
    MinIO-->>GraphRAG: 文件内容stream
    
    GraphRAG->>GraphRAG: POST /api/v1/ingest (内部API)<br/>{content, doc_type, doc_id}
    
    Note over GraphRAG: 阶段2.1: 文档解析与分块
    GraphRAG->>GraphRAG: DocumentProcessor.process_document()
    GraphRAG->>GraphRAG: _parse_document()<br/>格式转换(PDF/DOCX→文本)
    GraphRAG->>GraphRAG: _semantic_chunk()<br/>chunk_size=500, overlap=50
    GraphRAG->>GraphRAG: 生成120个DocumentChunk
    
    Note over GraphRAG,FAISS: 阶段2.2: 向量索引
    GraphRAG->>GraphRAG: VectorStore.add_texts(chunks)
    GraphRAG->>GraphRAG: BGE-M3 embedding<br/>batch_size=32
    GraphRAG->>FAISS: add_vectors(embeddings)
    FAISS->>FAISS: IVF索引更新<br/>nlist=100, nprobe=10
    FAISS-->>GraphRAG: 索引完成
    
    Note over GraphRAG,Neo4j: 阶段2.3: 实体提取与图谱构建
    loop 对每个chunk (batch_size=10)
        GraphRAG->>GraphRAG: EntityExtractor.extract(chunk.text)
        
        par 双路径提取
            GraphRAG->>GraphRAG: _extract_with_ner()<br/>spaCy NER模型
        and
            GraphRAG->>GraphRAG: _extract_with_llm()<br/>调用LLM Router
            GraphRAG->>LLMRouter: POST /api/v1/chat<br/>task_type:reasoning<br/>prompt:实体提取
            LLMRouter-->>GraphRAG: {entities:[], relations:[]}
        end
        
        GraphRAG->>GraphRAG: 去重合并实体<br/>_deduplicate_entities()
    end
    
    GraphRAG->>Neo4j: batch_create_entities(all_entities)<br/>MERGE (e:Entity {...})
    Neo4j-->>GraphRAG: 45个实体已创建
    
    GraphRAG->>Neo4j: batch_create_relations(all_relations)<br/>MERGE (a)-[:RELATION]->(b)
    Neo4j-->>GraphRAG: 67个关系已创建
    
    GraphRAG->>DocSvc: 更新文档状态<br/>PUT /documents/{doc_id}/status
    DocSvc->>PostgreSQL: UPDATE documents<br/>SET status='completed',<br/>chunks_count=120,<br/>entities_count=45
    
    GraphRAG->>MQ: PUBLISH document.processed<br/>{doc_id, status:'completed'}
    
    Note over WebApp,Browser: 阶段3: 前端轮询状态
    loop 每2秒轮询
        WebApp->>Gateway: GET /api/v01/documents/{doc_id}/status
        Gateway->>DocSvc: 查询文档状态
        DocSvc->>PostgreSQL: SELECT status, chunks_count<br/>FROM documents WHERE id=$1
        
        alt 处理完成
            PostgreSQL-->>DocSvc: {status:'completed', chunks:120}
            DocSvc-->>Gateway: 处理完成
            Gateway-->>WebApp: 完成响应
            WebApp-->>Browser: 显示:处理完成✓<br/>120个块,45个实体
        else 处理中
            PostgreSQL-->>DocSvc: {status:'processing'}
            DocSvc-->>Gateway: 处理中
            Gateway-->>WebApp: 处理中
            WebApp-->>Browser: 显示:处理中...
        end
    end
```

#### 5.1.2 调用链路说明

**阶段1: 文档上传（同步）**
1. 用户在浏览器选择文件并上传
2. Web前端发送multipart/form-data请求到网关
3. 网关认证中间件验证Cookie中的access_token
4. 文档Handler验证文件（类型白名单、大小限制、MIME类型）
5. 上传到MinIO对象存储（服务端加密）
6. 元数据保存到PostgreSQL（status='uploaded'）
7. 发布消息到Redis Pub/Sub队列
8. 立即返回200 OK给前端

**关键代码（文档Handler）**：
```go
// backend/internal/handlers/document_handler.go
func (h *DocumentHandler) UploadDocument(c *gin.Context) {
    // 1. 获取认证信息
    userID, _ := c.Get("user_id")
    tenantID, _ := c.Get("tenant_id")

    // 2. 解析multipart表单（最大50MB）
    if err := c.Request.ParseMultipartForm(50 << 20); err != nil {
        c.JSON(400, gin.H{"error": "File too large"})
        return
    }

    // 3. 获取文件
    file, header, err := c.Request.FormFile("file")
    defer file.Close()

    // 4. 文件验证
    ext := filepath.Ext(header.Filename)
    if !isAllowedExtension(ext) { // .pdf, .docx, .txt, .md
        c.JSON(400, gin.H{"error": "Unsupported file type"})
        return
    }

    // 5. 上传到MinIO
    docID := uuid.New().String()
    storagePath := fmt.Sprintf("%s/%s/%s", tenantID, docID, header.Filename)
    
    _, err = h.minioClient.PutObject(
        context.Background(),
        "documents",
        storagePath,
        file,
        header.Size,
        minio.PutObjectOptions{
            ContentType: mimeType,
            ServerSideEncryption: encrypt.NewSSE(), // 加密
        },
    )

    // 6. 保存元数据到PostgreSQL
    doc := &Document{
        ID:          docID,
        UserID:      userID.(string),
        TenantID:    tenantID.(string),
        Filename:    header.Filename,
        Size:        header.Size,
        StoragePath: storagePath,
        Status:      "uploaded",
        CreatedAt:   time.Now(),
    }
    h.db.Create(doc)

    // 7. 发布消息到队列（触发异步处理）
    h.pubsub.Publish("document.uploaded", map[string]interface{}{
        "doc_id":    docID,
        "tenant_id": tenantID,
        "path":      storagePath,
    })

    // 8. 立即返回
    c.JSON(200, gin.H{
        "document_id": docID,
        "status":      "uploaded",
        "message":     "File uploaded, processing in background",
    })
}
```

**阶段2: GraphRAG异步摄取（后台处理）**
1. GraphRAG服务订阅`document.uploaded`主题
2. 接收到消息后，从MinIO下载文件内容
3. **文档解析**：根据文件类型解析（PDF用pypdf，DOCX用python-docx）
4. **语义分块**：按句子边界分割，chunk_size=500字符，overlap=50
5. **向量化**：使用BGE-M3模型生成768维向量，batch_size=32
6. **FAISS索引**：添加向量到FAISS索引（IVF算法）
7. **实体提取**：NER（spaCy）和LLM（通过LLM Router）双路径
8. **图谱构建**：批量创建实体节点和关系边到Neo4j
9. 更新文档状态为'completed'
10. 发布`document.processed`消息

**关键代码（GraphRAG摄取服务）**：
```python
# algo/graphrag-service/core/ingest/ingest_service.py
class IngestService:
    async def ingest_document(
        self,
        content: str,
        doc_type: str,
        doc_id: str,
        metadata: Dict,
        build_graph: bool = True
    ) -> Dict:
        start_time = time.time()
        
        # 1. 文档处理和分块
        chunks = await self.document_processor.process_document(
            content=content,
            doc_type=doc_type,
            doc_id=doc_id,
            metadata=metadata
        )
        logger.info(f"文档分块完成: {len(chunks)}个块")
        
        # 2. 向量索引
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        await self.vector_store.add_texts(
            texts=chunk_texts,
            ids=chunk_ids,
            metadatas=[chunk.metadata for chunk in chunks]
        )
        logger.info(f"向量索引完成: {len(chunks)}个向量")
        
        # 3. 构建图谱（可选）
        entities_count = 0
        relations_count = 0
        
        if build_graph:
            all_entities = []
            all_relations = []
            
            # 批量提取实体（每10个chunk一批）
            for i in range(0, len(chunks), 10):
                batch_chunks = chunks[i:i+10]
                for chunk in batch_chunks:
                    # 实体提取（NER + LLM双路径）
                    extraction_result = await self.entity_extractor.extract(
                        text=chunk.text,
                        min_confidence=0.5
                    )
                    all_entities.extend(extraction_result["entities"])
                    all_relations.extend(extraction_result["relations"])
            
            # 去重
            all_entities = self._deduplicate_entities(all_entities)
            all_relations = self._deduplicate_relations(all_relations)
            
            # 批量创建实体节点
            entity_ids = await self.graph_builder.batch_create_entities(
                all_entities
            )
            entities_count = len(entity_ids)
            
            # 批量创建关系边
            relations_count = await self.graph_builder.batch_create_relations(
                all_relations
            )
            
            logger.info(f"图谱构建完成: {entities_count}个实体, {relations_count}个关系")
        
        elapsed_time = time.time() - start_time
        
        return {
            "doc_id": doc_id,
            "status": "completed",
            "chunks_count": len(chunks),
            "entities_count": entities_count,
            "relationships_count": relations_count,
            "elapsed_time": round(elapsed_time, 2)
        }
```

**阶段3: 前端轮询状态（定时查询）**
1. 前端每2秒轮询文档状态
2. 网关转发请求到文档服务
3. 文档服务查询PostgreSQL获取最新状态
4. 根据status字段返回不同状态：
   - `uploaded`: 已上传，等待处理
   - `processing`: 正在处理（摄取中）
   - `completed`: 处理完成
   - `failed`: 处理失败
5. 前端根据状态更新UI

**性能指标**：
- 文件上传：P50 ~500ms/MB，P95 ~1.5s/MB
- 文档摄取：~10秒/1000字（含图谱构建）
- 向量化：~300 chunks/s (batch_size=32)
- 实体提取：~50 chunks/min（LLM调用为瓶颈）
- 图谱构建：~1000 entities/s（批量写入）

---

### 5.2 Web前端智能问答 → GraphRAG检索全流程

```mermaid
sequenceDiagram
    autonumber
    participant Browser as 浏览器
    participant WebApp as Web前端
    participant Gateway as API网关<br/>:8080
    participant AuthMW as 认证中间件
    participant ChatHandler as 聊天Handler
    participant GraphRAG as GraphRAG服务<br/>:8001
    participant QueryRewriter as 查询改写
    participant VectorRet as 向量检索器
    participant GraphRet as 图谱检索器
    participant BM25Ret as BM25检索器
    participant FusionRanker as RRF融合器
    participant Reranker as CrossEncoder重排器
    participant FAISS as FAISS向量库
    participant Neo4j as Neo4j图数据库
    participant LLMRouter as LLM Router<br/>:8005
    
    Note over Browser,LLMRouter: 用户智能问答流程
    Browser->>WebApp: 用户输入问题<br/>"GraphRAG如何工作?"
    WebApp->>Gateway: POST /api/v01/chat/stream<br/>Cookie: access_token<br/>{query, session_id}
    
    Gateway->>AuthMW: 认证检查
    AuthMW-->>Gateway: user_id提取成功
    
    Gateway->>ChatHandler: 路由到聊天处理器
    ChatHandler->>ChatHandler: 验证会话所有权
    
    Note over ChatHandler,GraphRAG: 调用GraphRAG检索
    ChatHandler->>GraphRAG: POST /api/v1/query<br/>{query:"GraphRAG如何工作?",<br/>mode:"hybrid", top_k:5}
    
    Note over GraphRAG: 阶段1: 查询改写
    GraphRAG->>QueryRewriter: rewrite_query(query)
    QueryRewriter->>QueryRewriter: _apply_synonyms()<br/>"GraphRAG"→["GraphRAG","图检索增强"]
    QueryRewriter->>QueryRewriter: _expand_query()<br/>添加相关术语
    QueryRewriter->>QueryRewriter: _decompose_query()<br/>分解子问题
    QueryRewriter-->>GraphRAG: [原始查询, 变体1, 变体2]
    
    Note over GraphRAG,Neo4j: 阶段2: 并行多路召回
    par 向量检索
        GraphRAG->>VectorRet: search(queries, top_k=20)
        VectorRet->>VectorRet: embed_queries()<br/>BGE-M3 embedding
        VectorRet->>FAISS: search(query_vectors, k=20)
        FAISS->>FAISS: IVF近似搜索<br/>L2距离计算
        FAISS-->>VectorRet: 20个候选<br/>(score>0.7)
        VectorRet-->>GraphRAG: vector_results[20]
        
    and 图谱检索
        GraphRAG->>GraphRet: search(query, top_k=20)
        GraphRet->>Neo4j: MATCH (e:Entity)<br/>WHERE e.name CONTAINS $query<br/>RETURN e, relationships
        Neo4j->>Neo4j: 全文索引搜索<br/>多跳关系遍历
        Neo4j-->>GraphRet: 15个相关节点
        GraphRet-->>GraphRAG: graph_results[15]
        
    and BM25检索
        GraphRAG->>BM25Ret: search(query, top_k=20)
        BM25Ret->>BM25Ret: BM25算法<br/>TF-IDF计算<br/>内存倒排索引
        BM25Ret-->>GraphRAG: bm25_results[20]
    end
    
    Note over GraphRAG: 阶段3: RRF融合排序
    GraphRAG->>FusionRanker: fuse(<br/>  [(vector_results, 0.5),<br/>   (graph_results, 0.3),<br/>   (bm25_results, 0.2)],<br/>  top_k=10)
    FusionRanker->>FusionRanker: 计算RRF分数<br/>score = Σ(weight/(k+rank))<br/>k=60
    FusionRanker->>FusionRanker: 按RRF分数排序<br/>去重合并
    FusionRanker-->>GraphRAG: fused_results[10]
    
    Note over GraphRAG: 阶段4: CrossEncoder重排
    GraphRAG->>Reranker: rerank(query, fused_results, top_k=5)
    Reranker->>Reranker: 构造[query, doc]对<br/>10对
    Reranker->>Reranker: model.predict(pairs)<br/>cross-encoder/ms-marco-MiniLM
    Reranker->>Reranker: 按相关性分数排序
    Reranker-->>GraphRAG: reranked_results[5]
    
    GraphRAG-->>ChatHandler: {results:[5个最相关文档块],<br/>elapsed_time:265ms}
    
    Note over ChatHandler,LLMRouter: 阶段5: LLM生成答案
    ChatHandler->>ChatHandler: 构造Prompt<br/>context=检索结果<br/>question=用户问题
    ChatHandler->>LLMRouter: POST /api/v1/chat/stream<br/>{messages, task_type:"chat"}
    
    loop 流式生成
        LLMRouter-->>ChatHandler: SSE: data:{content:"..."}
        ChatHandler-->>Gateway: 转发SSE
        Gateway-->>WebApp: SSE流
        WebApp->>WebApp: 逐字累积内容
        WebApp-->>Browser: 实时渲染答案
    end
    
    LLMRouter-->>ChatHandler: SSE: data:{type:"done"}
    ChatHandler-->>Gateway: 完成信号
    Gateway-->>WebApp: 流结束
    WebApp-->>Browser: 显示完整答案
```

#### 5.2.1 调用链路详细说明

**阶段1: 查询改写（提升召回率）**

查询改写的目的是生成多个语义相似的查询变体，提高召回率。

**关键代码（QueryRewriter）**：
```python
# algo/graphrag-service/core/retrieval/query_rewriter.py
class QueryRewriter:
    async def rewrite_query(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> RewrittenQuery:
        rewritten_queries = [query]  # 始终包含原始查询
        
        # 策略1: 同义词替换
        if self.enable_synonym:
            synonym_queries = self._apply_synonyms(query)
            rewritten_queries.extend(synonym_queries[:1])
        
        # 策略2: 查询扩展（使用LLM）
        if self.enable_expansion and self.llm_client:
            expanded_queries = await self._expand_query(query, context)
            rewritten_queries.extend(expanded_queries[:1])
        
        # 策略3: 问题分解（复杂问题分解为子问题）
        if self.enable_decomposition and self._is_complex_query(query):
            decomposed_queries = await self._decompose_query(query, context)
            rewritten_queries.extend(decomposed_queries[:1])
        
        # 去重并限制数量（最多5个变体）
        rewritten_queries = list(dict.fromkeys(rewritten_queries))[:self.max_rewrites + 1]
        
        return RewrittenQuery(
            original=query,
            rewritten=rewritten_queries,
            strategy="hybrid",
            confidence=0.9,
            metadata={"count": len(rewritten_queries)}
        )
    
    def _apply_synonyms(self, query: str) -> List[str]:
        """同义词替换"""
        # 基于预定义词典替换
        synonyms = self.synonym_dict.get(query, [])
        return [query.replace(word, syn) for word, syn in synonyms.items()]
    
    async def _expand_query(self, query: str, context: Optional[Dict]) -> List[str]:
        """查询扩展（使用LLM生成相关术语）"""
        prompt = f"""
        原始查询: {query}
        
        请生成1个语义相似的查询变体，添加相关术语。
        只返回变体查询，不要解释。
        """
        
        # 调用LLM Router
        response = await self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        expanded = response.get("content", "").strip()
        return [expanded] if expanded and expanded != query else []
```

**查询改写示例**：
```
原始查询: "GraphRAG如何工作?"

改写变体:
1. "GraphRAG如何工作?" (原始)
2. "图检索增强生成如何工作?" (同义词替换)
3. "GraphRAG的工作原理和流程是什么?" (查询扩展)
4. "什么是GraphRAG?" + "GraphRAG的核心功能有哪些?" (问题分解)
```

**阶段2: 并行多路召回（异步并发）**

三路检索器并行执行，提高响应速度。

**关键代码（IntelligentRetriever）**：
```python
# algo/graphrag-service/core/retrieval/intelligent_retriever.py
class IntelligentRetriever:
    async def _multi_recall(
        self,
        queries: List[str],
        original_query: str,
        top_k: int,
        weights: Dict[str, float],
        filters: Optional[Dict]
    ) -> Dict[str, List[RetrievalResult]]:
        """并行多路召回"""
        recall_top_k = top_k * 4  # 召回4倍候选(20个)
        
        # 构建任务列表
        tasks = []
        if weights.get('vector', 0) > 0:
            tasks.append(self._vector_retrieve(queries, recall_top_k, filters))
        else:
            tasks.append(self._empty_results())
        
        if weights.get('graph', 0) > 0:
            tasks.append(self._graph_retrieve(original_query, recall_top_k, filters))
        else:
            tasks.append(self._empty_results())
        
        if weights.get('bm25', 0) > 0:
            tasks.append(self._bm25_retrieve(original_query, recall_top_k, filters))
        else:
            tasks.append(self._empty_results())
        
        # 并行执行（asyncio.gather）
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'vector': results[0] if not isinstance(results[0], Exception) else [],
            'graph': results[1] if not isinstance(results[1], Exception) else [],
            'bm25': results[2] if not isinstance(results[2], Exception) else []
        }
    
    async def _vector_retrieve(
        self,
        queries: List[str],
        top_k: int,
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """向量检索"""
        # 1. 对所有查询变体生成向量
        query_vectors = await self.vector_retriever.embed_queries(queries)
        
        # 2. FAISS搜索
        all_results = []
        for query_vector in query_vectors:
            results = await self.vector_retriever.search(
                query_vector,
                k=top_k,
                filters=filters
            )
            all_results.extend(results)
        
        # 3. 去重并按分数排序
        seen = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.score, reverse=True):
            if result.chunk_id not in seen:
                seen.add(result.chunk_id)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
```

**三路检索对比**：
| 检索路径 | 算法 | 优势 | 劣势 | 适用场景 |
|---------|------|------|------|---------|
| 向量检索 | FAISS IVF | 语义理解强 | 对关键词不敏感 | 语义相似匹配 |
| 图谱检索 | Neo4j Cypher | 推理能力强 | 覆盖面窄 | 多跳关系查询 |
| BM25检索 | TF-IDF | 关键词匹配准 | 无语义理解 | 精确关键词搜索 |

**阶段3: RRF融合排序（加权融合）**

RRF (Reciprocal Rank Fusion) 算法不依赖具体分数，只看排名。

**关键代码（RRFFusionRanker）**：
```python
# algo/graphrag-service/core/retrieval/fusion_ranker.py
class RRFFusionRanker:
    def __init__(self, k: int = 60):
        self.k = k  # RRF参数,论文推荐值60
    
    async def fuse(
        self,
        results_lists: List[Tuple[List[RetrievalResult], float]],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        RRF融合算法
        
        公式: RRF_score(d) = Σ(weight_i / (k + rank_i(d)))
        
        参数:
            results_lists: [(results, weight), ...] 
            top_k: 返回Top-K结果
        """
        # 收集所有文档的RRF分数
        rrf_scores = defaultdict(float)
        doc_map = {}  # chunk_id -> RetrievalResult
        
        for results, weight in results_lists:
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id
                
                # RRF公式: score = weight / (k + rank)
                rrf_score = weight / (self.k + rank)
                rrf_scores[chunk_id] += rrf_score
                
                # 保存文档对象（首次出现）
                if chunk_id not in doc_map:
                    doc_map[chunk_id] = result
        
        # 按RRF分数排序
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 返回Top-K
        fused_results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            result = doc_map[chunk_id]
            result.score = score  # 更新为RRF分数
            fused_results.append(result)
        
        return fused_results
```

**RRF融合示例**：
```
假设查询"GraphRAG如何工作?"的三路召回结果:

向量检索(权重0.5):
  rank1: doc_A (score=0.95)
  rank2: doc_B (score=0.90)
  rank3: doc_C (score=0.85)

图谱检索(权重0.3):
  rank1: doc_B (score=0.92)
  rank2: doc_D (score=0.88)
  rank3: doc_A (score=0.80)

BM25检索(权重0.2):
  rank1: doc_C (score=0.88)
  rank2: doc_A (score=0.85)
  rank3: doc_E (score=0.82)

RRF融合计算(k=60):
  doc_A: 0.5/(60+1) + 0.3/(60+3) + 0.2/(60+2) = 0.008197 + 0.004762 + 0.003226 = 0.016185
  doc_B: 0.5/(60+2) + 0.3/(60+1) = 0.008065 + 0.004918 = 0.012983
  doc_C: 0.5/(60+3) + 0.2/(60+1) = 0.007937 + 0.003279 = 0.011216
  doc_D: 0.3/(60+2) = 0.004839
  doc_E: 0.2/(60+3) = 0.003175

排序结果: doc_A > doc_B > doc_C > doc_D > doc_E
```

**阶段4: CrossEncoder重排（精排）**

CrossEncoder使用深度模型对[query, document]对进行相关性打分。

**关键代码（CrossEncoderReranker）**：
```python
# algo/graphrag-service/core/retrieval/reranker.py
class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.max_length = 512  # 最大输入长度
    
    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        CrossEncoder重排序
        
        步骤:
        1. 构造[query, document]对
        2. 使用CrossEncoder模型计算相关性分数
        3. 按分数排序并返回Top-K
        """
        # 1. 构造query-document对（截断文本）
        pairs = [[query, c.text[:self.max_length]] for c in candidates]
        
        # 2. 计算相关性分数（批量预测）
        scores = self.model.predict(pairs)
        
        # 3. 按分数排序
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 更新分数并返回Top-K
        reranked = []
        for candidate, score in scored_candidates[:top_k]:
            candidate.score = float(score)  # 更新为CrossEncoder分数
            candidate.source += "_reranked"  # 标记已重排
            reranked.append(candidate)
        
        return reranked
```

**CrossEncoder vs BiEncoder**：
| 特性 | BiEncoder | CrossEncoder |
|------|-----------|--------------|
| 架构 | 独立编码query和doc | 联合编码[query,doc] |
| 速度 | 快（可预计算doc向量） | 慢（需实时计算） |
| 精度 | 中等 | 高（+15% MRR） |
| 适用 | 粗排（Top-100） | 精排（Top-10） |

**阶段5: LLM生成答案（流式响应）**

基于检索结果构造Prompt，调用LLM生成答案。

**关键代码（ChatHandler）**：
```python
# 构造Prompt
context_text = "\n\n".join([
    f"文档{i+1}: {result.text}" 
    for i, result in enumerate(retrieval_results)
])

prompt = f"""
基于以下检索到的文档内容回答问题。

文档内容:
{context_text}

问题: {user_query}

要求:
1. 基于文档内容回答,不要编造
2. 如果文档中没有相关信息,说"文档中未找到相关信息"
3. 引用文档编号说明来源

答案:
"""

# 调用LLM Router流式生成
response = await llm_router_client.chat_stream(
    messages=[
        {"role": "system", "content": "你是一个专业的知识问答助手。"},
        {"role": "user", "content": prompt}
    ],
    task_type="chat",
    temperature=0.7,
    max_tokens=500
)

# 流式转发给前端
async for chunk in response:
    yield f"data: {json.dumps(chunk)}\n\n"
```

**性能指标**：
- 查询改写：~50ms
- 并行召回：P50 ~120ms，P95 ~200ms
  - 向量检索：~30ms
  - 图谱检索：~80ms
  - BM25检索：~10ms
- RRF融合：~5ms
- CrossEncoder重排：~40ms（10个候选）
- **总延迟**：P50 ~215ms，P95 ~295ms
- **召回率**：92%（Top-10）
- **Top-5准确率**：87%

---

## 六、时序图

### 6.1 文档摄取时序图

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI Router
    participant IngestService
    participant DocProcessor as DocumentProcessor
    participant EntityExtractor
    participant Neo4jManager
    participant VectorStore

    Client->>API: POST /api/v1/ingest
    API->>IngestService: ingest_document(content, doc_type, doc_id)
    
    IngestService->>DocProcessor: process_document(content, doc_type)
    DocProcessor->>DocProcessor: _parse_document(content, doc_type)
    DocProcessor->>DocProcessor: _semantic_chunk(parsed_content)
    DocProcessor-->>IngestService: List[DocumentChunk]
    
    IngestService->>VectorStore: add_texts(chunks)
    VectorStore-->>IngestService: indexed
    
    loop 对每个chunk
        IngestService->>EntityExtractor: extract(chunk.text)
        EntityExtractor->>EntityExtractor: _extract_with_ner(text)
        EntityExtractor->>EntityExtractor: _extract_with_llm(text)
        EntityExtractor-->>IngestService: {entities, relations}
    end
    
    IngestService->>Neo4jManager: batch_create_entities(all_entities)
    Neo4jManager-->>IngestService: entity_ids
    
    IngestService->>Neo4jManager: batch_create_relations(all_relations)
    Neo4jManager-->>IngestService: relation_count
    
    IngestService-->>API: {status, chunks_count, entities_count}
    API-->>Client: 200 OK {doc_id, status: "processing"}
```

### 5.2 智能检索时序图

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Retriever as IntelligentRetriever
    participant QueryRewriter
    participant VectorRetriever
    participant GraphRetriever
    participant BM25Retriever
    participant FusionRanker as RRFFusionRanker
    participant Reranker as CrossEncoderReranker

    Client->>API: POST /api/v1/query {query, mode, top_k}
    API->>Retriever: retrieve(query, top_k, use_rewrite, use_rerank)
    
    Retriever->>QueryRewriter: rewrite(query)
    QueryRewriter->>QueryRewriter: _apply_synonyms()
    QueryRewriter->>QueryRewriter: _expand_query()
    QueryRewriter->>QueryRewriter: _decompose_query()
    QueryRewriter-->>Retriever: [query, variant1, variant2, ...]
    
    par 并行多路召回
        Retriever->>VectorRetriever: search(queries, top_k*4)
        VectorRetriever-->>Retriever: vector_results
    and
        Retriever->>GraphRetriever: search(query, top_k*4)
        GraphRetriever-->>Retriever: graph_results
    and
        Retriever->>BM25Retriever: search(query, top_k*4)
        BM25Retriever-->>Retriever: bm25_results
    end
    
    Retriever->>FusionRanker: fuse([(vector_results, 0.5), (graph_results, 0.3), (bm25_results, 0.2)], top_k*2)
    FusionRanker->>FusionRanker: 计算RRF分数: weight/(k+rank)
    FusionRanker-->>Retriever: fused_results (Top-K*2)
    
    Retriever->>Reranker: rerank(query, fused_results, top_k)
    Reranker->>Reranker: model.predict([[query, doc.text], ...])
    Reranker-->>Retriever: reranked_results (Top-K)
    
    Retriever-->>API: List[RetrievalResult]
    API-->>Client: 200 OK {results, elapsed_time, stats}
```

### 5.3 社区检测时序图

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Detector as CommunityDetector
    participant Neo4jManager
    participant NetworkX
    participant LLMService

    Client->>API: POST /api/v1/community/detect {algorithm, resolution}
    API->>Detector: detect_communities(algorithm, resolution)
    
    Detector->>Neo4jManager: export_to_networkx()
    Neo4jManager->>Neo4jManager: 查询所有节点和边
    Neo4jManager-->>Detector: NetworkX Graph (G)
    
    Detector->>NetworkX: 在executor中执行算法(避免阻塞)
    alt Louvain算法
        NetworkX->>NetworkX: community_louvain.best_partition(G)
    else Leiden算法
        NetworkX->>NetworkX: leidenalg.find_partition(G)
    else Label Propagation
        NetworkX->>NetworkX: label_propagation_communities(G)
    end
    NetworkX-->>Detector: communities_dict {node: community_id}
    
    Detector->>Detector: _format_communities(communities_dict)
    Detector->>NetworkX: compute modularity
    NetworkX-->>Detector: modularity_score
    
    opt 生成LLM摘要
        loop 对前5个社区
            Detector->>LLMService: 生成社区摘要(nodes_str)
            LLMService-->>Detector: llm_summary
        end
    end
    
    Detector-->>API: {communities, stats, modularity}
    API-->>Client: 200 OK {communities, stats, modularity, elapsed_time}
```

---

## 六、架构图与交互图

### 6.1 GraphRAG服务内部架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  /ingest │  │  /query  │  │/community│  │  /stats  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
└───────┼─────────────┼─────────────┼─────────────┼──────────────┘
        │             │             │             │
┌───────┼─────────────┼─────────────┼─────────────┼──────────────┐
│       │             │             │             │  Service     │
│       ↓             ↓             ↓             ↓  Layer       │
│  ┌─────────┐  ┌─────────────┐  ┌─────────┐  ┌─────────┐      │
│  │ Ingest  │  │ Intelligent │  │Community│  │ Graph   │      │
│  │ Service │  │  Retriever  │  │Detector │  │ Manager │      │
│  └────┬────┘  └──────┬──────┘  └────┬────┘  └────┬────┘      │
│       │              │              │            │             │
│       │    ┌─────────┴────────┐     │            │             │
│       │    │                  │     │            │             │
│  ┌────┴───┐│  ┌──────┐  ┌────┴─┐  ┌┴───────┐   │             │
│  │Document││  │Query │  │Fusion│  │NetworkX│   │             │
│  │Process.││  │Rewrit│  │Ranker│  │ Algos  │   │             │
│  └────┬───┘│  └──────┘  └──────┘  └────────┘   │             │
│       │    │                                     │             │
│  ┌────┴───┐│  ┌────────────────────────────┐   │             │
│  │ Entity ││  │  Vector / Graph / BM25     │   │             │
│  │Extract.││  │       Retrievers           │   │             │
│  └────┬───┘│  └────────────────────────────┘   │             │
│       │    │                  │                 │             │
│       │    └──────────────────┼─────────────────┘             │
└───────┼───────────────────────┼───────────────────────────────┘
        │                       │
┌───────┼───────────────────────┼───────────────────────────────┐
│       │                       │             Data Layer        │
│       ↓                       ↓                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │  Neo4j  │  │  FAISS  │  │  Redis  │  │   LLM   │         │
│  │(Graph)  │  │(Vector) │  │ (Cache) │  │ Router  │         │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 模块交互关系详解

#### 6.2.1 GraphRAG服务与其他模块的交互拓扑图

```mermaid
graph TB
    subgraph "前端层"
        WebFE[Web前端<br/>Next.js]
        AdminFE[Admin后台<br/>Flask]
    end
    
    subgraph "网关层"
        Gateway[API网关:8080<br/>Gin Engine]
        AuthMW[认证中间件]
        RateLimit[限流中间件]
    end
    
    subgraph "微服务层"
        DocSvc[文档服务:8082<br/>Document Service]
        SessionSvc[会话服务:8083<br/>Session Service]
    end
    
    subgraph "GraphRAG服务层 :8001"
        direction TB
        GraphRAG[GraphRAG API Layer<br/>FastAPI]
        
        subgraph "Ingest Pipeline"
            DocProc[DocumentProcessor]
            EntityExt[EntityExtractor]
            GraphBuilder[Neo4jManager]
        end
        
        subgraph "Retrieval Pipeline"
            QueryRW[QueryRewriter]
            VectorRet[VectorRetriever]
            GraphRet[GraphRetriever]
            BM25Ret[BM25Retriever]
            FusionRanker[RRFFusionRanker]
            Reranker[CrossEncoderReranker]
        end
        
        GraphRAG --> DocProc
        GraphRAG --> EntityExt
        GraphRAG --> QueryRW
        DocProc --> GraphBuilder
        EntityExt --> GraphBuilder
        QueryRW --> VectorRet
        QueryRW --> GraphRet
        QueryRW --> BM25Ret
        VectorRet --> FusionRanker
        GraphRet --> FusionRanker
        BM25Ret --> FusionRanker
        FusionRanker --> Reranker
    end
    
    subgraph "算法服务层"
        LLMRouter[LLM Router:8005<br/>智能路由]
    end
    
    subgraph "存储层"
        MinIO[(MinIO<br/>对象存储)]
        PostgreSQL[(PostgreSQL<br/>元数据)]
        Neo4j[(Neo4j<br/>知识图谱)]
        FAISS[(FAISS<br/>向量索引)]
        RedisMQ[(Redis<br/>消息队列)]
    end
    
    %% 前端到网关
    WebFE -->|1.上传文档| Gateway
    WebFE -->|2.智能问答| Gateway
    AdminFE -->|监控管理| Gateway
    
    %% 网关到微服务
    Gateway --> AuthMW
    Gateway --> RateLimit
    Gateway -->|路由| DocSvc
    Gateway -->|路由| SessionSvc
    Gateway -.->|直接代理<br/>/graphrag/*| GraphRAG
    
    %% 微服务到存储
    DocSvc -->|存储文件| MinIO
    DocSvc -->|元数据| PostgreSQL
    DocSvc -->|发布事件| RedisMQ
    
    %% GraphRAG订阅和处理
    RedisMQ -.->|订阅document.uploaded| GraphRAG
    GraphRAG -->|下载文件| MinIO
    GraphRAG -->|更新状态| DocSvc
    
    %% GraphRAG到存储层
    GraphRAG -->|向量索引| FAISS
    GraphBuilder -->|图谱构建| Neo4j
    VectorRet -->|向量检索| FAISS
    GraphRet -->|图谱查询| Neo4j
    
    %% GraphRAG到LLM
    EntityExt -->|实体提取| LLMRouter
    QueryRW -->|查询扩展| LLMRouter
    
    %% LLM返回
    LLMRouter -.->|返回检索结果| Gateway
    Gateway -.->|SSE流式响应| WebFE
    
    classDef frontend fill:#e1f5ff,stroke:#01579b
    classDef gateway fill:#fff9c4,stroke:#f57f17
    classDef service fill:#f3e5f5,stroke:#4a148c
    classDef graphrag fill:#e8f5e9,stroke:#1b5e20
    classDef algo fill:#fce4ec,stroke:#880e4f
    classDef storage fill:#fff3e0,stroke:#e65100
    
    class WebFE,AdminFE frontend
    class Gateway,AuthMW,RateLimit gateway
    class DocSvc,SessionSvc service
    class GraphRAG,DocProc,EntityExt,GraphBuilder,QueryRW,VectorRet,GraphRet,BM25Ret,FusionRanker,Reranker graphrag
    class LLMRouter algo
    class MinIO,PostgreSQL,Neo4j,FAISS,RedisMQ storage
```

#### 6.2.2 交互场景详细说明

##### **场景1: 文档摄取流程**

| 步骤 | 模块A | 交互类型 | 模块B | 数据流 | 协议/格式 |
|------|-------|----------|-------|--------|-----------|
| 1 | Web前端 | HTTP POST | API网关 | multipart/form-data | HTTP/1.1 |
| 2 | API网关 | 认证检查 | 认证中间件 | Cookie: access_token | JWT Token |
| 3 | 认证中间件 | 提取身份 | Gateway Context | user_id, tenant_id | Context传递 |
| 4 | API网关 | 路由转发 | 文档服务 | 文件流+元数据 | HTTP代理 |
| 5 | 文档服务 | 文件存储 | MinIO | PutObject请求 | S3协议 |
| 6 | 文档服务 | 保存元数据 | PostgreSQL | INSERT语句 | SQL |
| 7 | 文档服务 | 发布事件 | Redis Pub/Sub | document.uploaded | JSON消息 |
| 8 | 文档服务 | 响应前端 | API网关 | {doc_id, status} | JSON |
| 9 | GraphRAG | 订阅消息 | Redis Pub/Sub | 接收事件 | JSON消息 |
| 10 | GraphRAG | 下载文件 | MinIO | GetObject请求 | S3协议 |
| 11 | GraphRAG | 文档解析 | DocumentProcessor | 内部调用 | Python函数 |
| 12 | GraphRAG | 向量化 | FAISS | add_vectors() | NumPy数组 |
| 13 | GraphRAG | 实体提取 | LLM Router | POST /api/v1/chat | HTTP+JSON |
| 14 | GraphRAG | 图谱构建 | Neo4j | Cypher MERGE | Bolt协议 |
| 15 | GraphRAG | 更新状态 | 文档服务 | PUT /status | HTTP+JSON |

**关键交互说明**：

1. **异步解耦**：文档服务通过消息队列与GraphRAG解耦，上传立即返回，后台异步处理
2. **事件驱动**：使用Redis Pub/Sub实现事件驱动架构，支持多个订阅者
3. **协议适配**：MinIO使用S3兼容协议，Neo4j使用Bolt二进制协议
4. **错误处理**：每个环节都有超时和重试机制，失败后更新文档状态为'failed'

**性能优化**：
- 文件流式传输，不在内存中完整加载
- 向量化批量处理（batch_size=32）
- 图谱批量写入（UNWIND批量MERGE）
- 实体提取并发处理（10个chunk一批）

---

##### **场景2: 智能检索流程**

| 步骤 | 模块A | 交互类型 | 模块B | 数据流 | 延迟(ms) |
|------|-------|----------|-------|--------|----------|
| 1 | Web前端 | HTTP POST | API网关 | {query, session_id} | ~10 |
| 2 | API网关 | 认证检查 | 认证中间件 | JWT验证 | ~5 |
| 3 | API网关 | 路由转发 | 聊天Handler | 请求代理 | ~2 |
| 4 | 聊天Handler | HTTP POST | GraphRAG /query | {query, mode, top_k} | ~220 |
| 5 | GraphRAG | 查询改写 | QueryRewriter | 内部调用 | ~50 |
| 6 | GraphRAG | 并行召回 | 3个检索器 | asyncio.gather() | ~120 |
| 6a | - VectorRetriever | 向量搜索 | FAISS | search(vectors, k) | ~30 |
| 6b | - GraphRetriever | 图谱查询 | Neo4j | MATCH Cypher | ~80 |
| 6c | - BM25Retriever | 关键词搜索 | 内存索引 | BM25算法 | ~10 |
| 7 | GraphRAG | RRF融合 | FusionRanker | 加权融合 | ~5 |
| 8 | GraphRAG | CrossEncoder重排 | Reranker | model.predict() | ~40 |
| 9 | GraphRAG | 返回结果 | 聊天Handler | Top-5文档块 | ~3 |
| 10 | 聊天Handler | 构造Prompt | LLM Router | {messages, context} | ~5 |
| 11 | LLM Router | 流式生成 | OpenAI/Claude | SSE流 | ~2000 |
| 12 | 聊天Handler | 转发SSE | API网关 | 流式代理 | ~1 |
| 13 | API网关 | 转发SSE | Web前端 | data: {...} | ~1 |

**关键交互说明**：

1. **并行优化**：三路检索器使用`asyncio.gather()`并发执行，总延迟取决于最慢的路径
2. **流式响应**：LLM生成阶段使用SSE流式传输，用户体验更好（TTFT ~500ms）
3. **缓存策略**：
   - 查询改写结果缓存（TTL 5分钟）
   - 向量检索结果缓存（TTL 10分钟）
   - LLM响应缓存（TTL 1小时）
4. **降级策略**：
   - 向量检索失败 → 降级到BM25检索
   - 图谱检索失败 → 权重转移到向量检索
   - CrossEncoder重排失败 → 使用RRF融合结果

**延迟分解**：
```
总延迟 ~2.3秒 = 
  前端到网关(10ms) + 
  认证(5ms) + 
  GraphRAG检索(220ms) + 
  LLM生成(2000ms) + 
  网络传输(5ms)
```

---

##### **场景3: 实体提取与图谱构建**

| 组件 | 调用关系 | 数据流 | 批量策略 |
|------|----------|--------|----------|
| EntityExtractor | 调用 LLM Router | {text, prompt} | batch_size=10 chunks |
| EntityExtractor | 调用 spaCy NER | {text} | 单chunk处理 |
| EntityExtractor | 返回 | {entities:[], relations:[]} | 去重后合并 |
| Neo4jManager | batch_create_entities() | UNWIND + MERGE | 批量写入100个 |
| Neo4jManager | batch_create_relations() | UNWIND + MERGE | 批量写入100个 |

**双路径实体提取**：

```python
# 路径1: NER提取（快速、覆盖常见实体）
ner_entities = spacy_nlp(text).ents  # Person, Organization, Location, Date

# 路径2: LLM提取（准确、覆盖领域实体）
llm_prompt = """
从以下文本中提取实体和关系。
实体类型: Person, Organization, Concept, Technology, Method
关系类型: belongs_to, uses, related_to, implements
"""
llm_entities = await llm_router.chat(prompt)  # 深度理解语义

# 融合策略: 
# - 相同实体名，保留LLM结果（confidence更高）
# - 不同实体名，合并两路结果
# - 关系仅LLM提取（NER无法提取关系）
```

**图谱批量写入优化**：

```cypher
-- 批量创建实体（使用UNWIND）
UNWIND $entities as entity
MERGE (e:Entity {name: entity.name})
SET e.type = entity.type,
    e.description = entity.description,
    e.confidence = entity.confidence,
    e.updated_at = datetime()
RETURN id(e) as entity_id

-- 批量创建关系
UNWIND $relations as rel
MATCH (a:Entity {name: rel.source})
MATCH (b:Entity {name: rel.target})
MERGE (a)-[r:RELATION {type: rel.relation}]->(b)
SET r.confidence = rel.confidence,
    r.updated_at = datetime()
RETURN count(r) as count
```

**性能对比**：
| 方式 | 100个实体耗时 | 1000个实体耗时 | 吞吐量 |
|------|--------------|----------------|--------|
| 逐个CREATE | ~5秒 | ~50秒 | 20 entities/s |
| 批量UNWIND+MERGE | ~0.1秒 | ~1秒 | 1000 entities/s |
| **提升** | **50倍** | **50倍** | **50倍** |

---

#### 6.2.3 模块依赖矩阵

| GraphRAG模块 ↓ \\ 依赖 → | LLM Router | Neo4j | FAISS | Redis | MinIO | PostgreSQL | Document Service |
|------------------------|-----------|-------|-------|-------|-------|------------|------------------|
| **DocumentProcessor** | ❌ | ❌ | ❌ | ❌ | ✅读取 | ❌ | ❌ |
| **EntityExtractor** | ✅调用 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Neo4jManager** | ❌ | ✅读写 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **VectorRetriever** | ❌ | ❌ | ✅读写 | ✅缓存 | ❌ | ❌ | ❌ |
| **GraphRetriever** | ❌ | ✅只读 | ❌ | ✅缓存 | ❌ | ❌ | ❌ |
| **BM25Retriever** | ❌ | ❌ | ❌ | ✅缓存 | ❌ | ❌ | ❌ |
| **QueryRewriter** | ✅可选 | ❌ | ❌ | ✅缓存 | ❌ | ❌ | ❌ |
| **IngestService** | ✅调用 | ✅写入 | ✅写入 | ✅订阅 | ✅读取 | ❌ | ✅更新状态 |
| **IntelligentRetriever** | ❌ | ❌ | ❌ | ✅缓存 | ❌ | ❌ | ❌ |

**依赖说明**：
- ✅ **强依赖**：模块无法工作如果依赖不可用
- ✅可选 **可选依赖**：模块可降级运行如果依赖不可用
- ❌ **无依赖**：模块不使用该依赖

**降级策略**：
1. **LLM Router不可用**：
   - EntityExtractor → 只使用NER路径（准确率降低15%）
   - QueryRewriter → 禁用查询扩展（召回率降低10%）
   
2. **Neo4j不可用**：
   - GraphRetriever → 返回空结果
   - 权重自动调整：vector=0.7, bm25=0.3
   
3. **FAISS不可用**：
   - VectorRetriever → 降级到BM25检索
   - 召回率降低20%
   
4. **Redis不可用**：
   - 禁用缓存，所有请求直接查询
   - 延迟增加50-100ms

---

#### 6.2.4 数据流详解

**文档摄取数据流**：

```
┌─────────────────┐
│ 文档文件 (PDF)  │  10MB
└────────┬────────┘
         │ 1. 上传
         ↓
┌─────────────────┐
│  MinIO存储      │  原始文件
└────────┬────────┘
         │ 2. 下载
         ↓
┌─────────────────┐
│  文档解析       │  纯文本 ~50KB
└────────┬────────┘
         │ 3. 分块
         ↓
┌─────────────────┐
│  120个Chunk     │  每块~500字符
└────────┬────────┘
         │ 4. 双路处理
         ├─────────────────┬───────────────┐
         ↓                 ↓               ↓
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  向量化         │ │  实体提取       │ │  元数据         │
│  BGE-M3         │ │  NER+LLM        │ │  保存           │
│  120×768维      │ │  45个实体       │ │  PostgreSQL     │
└────────┬────────┘ └────────┬────────┘ └─────────────────┘
         │                   │
         ↓                   ↓
┌─────────────────┐ ┌─────────────────┐
│  FAISS索引      │ │  Neo4j图谱      │
│  IVF nlist=100  │ │  45节点+67边    │
└─────────────────┘ └─────────────────┘
```

**智能检索数据流**：

```
┌─────────────────┐
│ 用户查询        │  "GraphRAG如何工作?"
└────────┬────────┘
         │ 1. 改写
         ↓
┌─────────────────┐
│ 3个查询变体     │  [原始, 同义词, 扩展]
└────────┬────────┘
         │ 2. 并行召回(Top-20)
         ├──────────────┬──────────────┐
         ↓              ↓              ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ 向量检索    │  │ 图谱检索    │  │ BM25检索    │
│ 20个结果    │  │ 15个结果    │  │ 20个结果    │
│ L2距离排序  │  │ 相关度排序  │  │ TF-IDF排序  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┴────────────────┘
                        │ 3. RRF融合(去重)
                        ↓
                ┌───────────────┐
                │ 10个候选      │  权重融合排序
                └───────┬───────┘
                        │ 4. CrossEncoder重排
                        ↓
                ┌───────────────┐
                │ Top-5结果     │  精准相关性分数
                └───────┬───────┘
                        │ 5. 构造Prompt
                        ↓
                ┌───────────────┐
                │ LLM生成       │  基于检索结果回答
                └───────┬───────┘
                        │ 6. SSE流式返回
                        ↓
                ┌───────────────┐
                │ 前端渲染      │  逐字显示答案
                └───────────────┘
```

**数据大小对比**：
| 阶段 | 数据类型 | 大小 | 传输协议 |
|------|----------|------|----------|
| 用户查询 | 文本 | ~100 bytes | HTTP JSON |
| 查询变体 | 文本数组 | ~300 bytes | 内存 |
| 向量化 | float32数组 | 768×4=3KB | 内存 |
| FAISS召回 | 20个chunk | 20×500字符=10KB | 内存/文件 |
| Neo4j召回 | 15个节点+边 | ~5KB | Bolt协议 |
| BM25召回 | 20个chunk | ~10KB | 内存 |
| RRF融合结果 | 10个chunk | ~5KB | 内存 |
| CrossEncoder输入 | 10对[query,doc] | ~5KB | GPU内存 |
| Top-5结果 | 5个chunk | ~2.5KB | HTTP JSON |
| LLM Prompt | 文本 | ~3KB | HTTP JSON |
| LLM响应 | SSE流 | ~1-2KB | SSE |

---

**交互协议栈**：

```
应用层:  HTTP/1.1, SSE, WebSocket
         JSON, MessagePack
         
传输层:  TCP (HTTP, Bolt)
         Redis Protocol
         S3 Protocol
         
网络层:  IPv4/IPv6
         
数据链路层: Ethernet
```

---

## 七、性能优化与最佳实践

### 7.1 摄取性能优化

**批量处理**:
```python
# 使用BatchDocumentProcessor并发处理多个文档
batch_processor = BatchDocumentProcessor(document_processor)
all_chunks = await batch_processor.process_documents(documents, batch_size=10)
```

**异步摄取**:
```python
# 使用FastAPI的BackgroundTasks异步处理
@router.post("/ingest")
async def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_ingest_task)
    return success_response({"status": "processing"})
```

**Neo4j批量操作**:
```python
# 使用UNWIND批量创建,而非逐个创建
await graph_builder.batch_create_entities(entities)  # ✅
# 而不是:
# for entity in entities:
#     await graph_builder.create_entity(entity)  # ❌
```

### 7.2 检索性能优化

**缓存策略**:
- 查询级缓存:缓存完全相同的查询
- 语义缓存:缓存语义相似的查询(基于embedding)
- 结果缓存:缓存Top-K结果,TTL 5-10分钟

**并行召回**:
```python
# 使用asyncio.gather并行执行三路检索
tasks = [
    self._vector_retrieve(queries, top_k),
    self._graph_retrieve(query, top_k),
    self._bm25_retrieve(query, top_k)
]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Top-K优化**:
- 召回阶段:取4倍候选(Top-40)
- 融合阶段:保留2倍候选(Top-20)
- 重排阶段:返回Top-K(Top-10)

### 7.3 图谱设计最佳实践

**索引优化**:
```cypher
// 为常用查询字段创建索引
CREATE INDEX entity_name_idx FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_idx FOR (e:Entity) ON (e.type);
```

**实体去重**:
```python
# 使用MERGE而非CREATE,确保实体唯一
MERGE (e:Entity {name: $name})
SET e.type = $type
```

**关系权重**:
```python
# 为关系添加权重,支持加权图算法
MERGE (a)-[r:RELATION {type: $relation_type}]->(b)
SET r.weight = $confidence
```

### 7.4 监控与调试

**关键指标**:
- 摄取速度:文档/秒,块/秒
- 检索延迟:P50、P95、P99
- 召回率:向量/图谱/BM25各路召回数
- 融合效果:RRF后排序变化
- 重排提升:CrossEncoder前后Top-5变化

**日志记录**:
```python
logger.info(
    "检索完成",
    query=query[:50],
    vector_count=len(vector_results),
    graph_count=len(graph_results),
    bm25_count=len(bm25_results),
    fused_count=len(fused_results),
    final_count=len(final_results),
    elapsed_ms=int(elapsed * 1000)
)
```

---

## 八、配置与部署

### 8.1 环境变量

```bash
# Neo4j配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# FAISS索引路径
FAISS_INDEX_PATH=data/faiss_index

# BM25索引路径
BM25_INDEX_PATH=data/bm25_index

# Redis配置
REDIS_URL=redis://localhost:6379

# LLM Router Service
LLM_ROUTER_URL=http://localhost:8005

# 服务端口
PORT=8001
```

### 8.2 Docker部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 8.3 依赖安装

```txt
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
neo4j==5.13.0
faiss-cpu==1.7.4
rank-bm25==0.2.2
sentence-transformers==2.2.2
spacy==3.7.2
httpx==0.25.0
networkx==3.2.1
python-louvain==0.16
leidenalg==0.10.1
scikit-learn==1.3.2
numpy==1.26.2
```

---

## 九、总结

### 9.1 核心优势

1. **混合检索**:整合向量、图谱、BM25,召回率>90%
2. **智能重排**:RRF融合+CrossEncoder,Top-5准确率>85%
3. **语义分块**:保持文本完整性,提升检索质量
4. **双路径提取**:NER+LLM,实体识别F1>0.85
5. **社区检测**:发现知识聚类,支持多种算法

### 9.2 性能指标

| 指标 | 目标 | 实际 |
|------|------|------|
| 检索延迟(P95) | < 300ms | 265ms |
| 召回率 | > 90% | 92% |
| Top-5准确率 | > 85% | 87% |
| 摄取速度 | > 100 chunks/s | 120 chunks/s |
| 实体提取F1 | > 0.85 | 0.87 |

### 9.3 未来优化方向

1. **增量更新**:支持文档增量摄取和图谱增量更新
2. **多模态检索**:支持图像、表格等多模态内容
3. **实时更新**:支持流式文档摄取
4. **分布式部署**:支持FAISS分片和Neo4j集群
5. **查询优化器**:基于统计信息自动调整权重

---


## 十、总结与展望


### 10.1 技术亮点

1. **异步解耦架构**：
   - 文档上传立即返回，后台异步处理
   - 使用Redis Pub/Sub实现事件驱动
   - 前端轮询状态，用户体验友好

2. **混合检索系统**：
   - 向量检索（语义理解）+ 图谱检索（推理能力）+ BM25检索（关键词匹配）
   - RRF融合算法无需归一化分数
   - CrossEncoder重排提升15% MRR

3. **双路径实体提取**：
   - NER路径：快速、覆盖常见实体
   - LLM路径：准确、覆盖领域实体
   - 融合策略：保留更高置信度结果

4. **批量优化**：
   - 向量化批量处理（batch_size=32）
   - 图谱批量写入（UNWIND+MERGE，50倍提升）
   - 实体提取并发处理（10个chunk一批）

5. **降级策略**：
   - LLM Router不可用 → 只使用NER
   - Neo4j不可用 → 权重转移到向量检索
   - FAISS不可用 → 降级到BM25检索
   - Redis不可用 → 禁用缓存

### 10.2 优化方向

1. **增量更新**：
   - 支持文档增量摄取（只处理变更部分）
   - 图谱增量更新（避免重复创建实体）

2. **多模态检索**：
   - 支持图像检索（CLIP模型）
   - 支持表格检索（Table-BERT）
   - 支持音频检索（Whisper embedding）

3. **实时更新**：
   - 支持流式文档摄取（WebSocket推送）
   - 实时图谱更新（CDC + Kafka）

4. **分布式部署**：
   - FAISS分片（Shard by tenant_id）
   - Neo4j集群（Causal Cluster）
   - 向量检索服务独立扩容

5. **查询优化器**：
   - 基于统计信息自动调整权重
   - 根据召回率动态调整Top-K
   - 基于历史查询优化查询改写策略


