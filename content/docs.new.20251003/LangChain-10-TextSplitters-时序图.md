# LangChain-10-TextSplitters-时序图

## 文档说明

本文档通过详细的时序图展示 **Text Splitters 模块**在各种场景下的执行流程，包括文本分割、块管理、分隔符选择、令牌处理、文档结构化等复杂交互过程。

---

## 1. 基础分割场景

### 1.1 CharacterTextSplitter 基础分割流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Splitter as CharacterTextSplitter
    participant Validator as ConfigValidator
    participant Processor as TextProcessor
    participant Merger as ChunkMerger

    User->>Splitter: split_text("段落1\n\n段落2\n\n段落3")

    Splitter->>Validator: 验证配置参数<br/>chunk_size=200, chunk_overlap=50, separator="\n\n"

    Validator->>Validator: 检查参数有效性<br/>chunk_overlap < chunk_size ✓<br/>separator不为空 ✓

    Validator-->>Splitter: 配置验证通过

    Splitter->>Processor: 按分隔符分割文本<br/>text.split("\n\n")

    Processor->>Processor: 执行文本分割<br/>splits = ["段落1", "段落2", "段落3"]

    alt keep_separator = True
        Processor->>Processor: 重新添加分隔符<br/>["段落1\n\n", "段落2\n\n", "段落3"]
    end

    Processor-->>Splitter: 初始分割结果<br/>splits = ["段落1", "段落2", "段落3"]

    Splitter->>Merger: 合并小块并处理重叠<br/>_merge_splits(splits, separator)

    Merger->>Merger: 检查每个分割块的长度<br/>length_function(split) for split in splits

    loop 处理每个分割块
        Merger->>Merger: 当前块长度 = len("段落1") = 3

        alt 块长度 < chunk_size
            Merger->>Merger: 块大小合适，保留
        else 块长度 >= chunk_size
            Merger->>Merger: 块过大，需要进一步分割<br/>（CharacterTextSplitter无递归处理）
        end
    end

    Merger->>Merger: 合并相邻小块<br/>确保块大小接近chunk_size<br/>同时处理chunk_overlap重叠

    Merger-->>Splitter: 最终合并结果<br/>["段落1", "段落2", "段落3"]

    Splitter->>Splitter: 应用后处理<br/>strip_whitespace, add_start_index

    Splitter-->>User: 分割结果<br/>["段落1", "段落2", "段落3"]
```

**关键步骤说明**：

1. **配置验证**（步骤 2-4）：
   - 检查chunk_size和chunk_overlap的合理性
   - 验证分隔符的有效性
   - 确保length_function可调用

2. **文本分割**（步骤 5-8）：
   - 使用指定分隔符分割文本
   - 可选择保留分隔符在结果中
   - 处理正则表达式分隔符

3. **块合并**（步骤 9-16）：
   - 检查每个块的长度
   - 合并过小的相邻块
   - 处理块间重叠

---

### 1.2 RecursiveCharacterTextSplitter 递归分割流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Splitter as RecursiveCharacterTextSplitter
    participant Strategy as SeparatorStrategy
    participant Processor as RecursiveProcessor
    participant Merger as ChunkMerger

    User->>Splitter: split_text("# 标题\n\n段落内容...\n\n另一个很长的段落...")

    Splitter->>Strategy: 获取分隔符优先级列表<br/>separators = ["\n\n", "\n", " ", ""]

    Strategy-->>Splitter: 分隔符策略已准备

    Splitter->>Processor: 启动递归分割<br/>_recursive_split(text, separators)

    Processor->>Processor: 选择最佳分隔符<br/>遍历separators找到第一个存在于文本中的

    Processor->>Processor: 找到分隔符: "\n\n"<br/>因为文本包含段落分隔符

    Processor->>Processor: 使用"\n\n"分割文本<br/>splits = ["# 标题", "段落内容...", "另一个很长的段落..."]

    loop 处理每个分割块
        Processor->>Processor: 检查块大小<br/>current_split = "另一个很长的段落..."

        alt 块大小 <= chunk_size
            Processor->>Processor: 块大小合适<br/>添加到good_splits
        else 块大小 > chunk_size
            Processor->>Processor: 块过大，需要递归处理<br/>remaining_separators = ["\n", " ", ""]

            alt 还有剩余分隔符
                Processor->>Processor: 递归调用<br/>_recursive_split(oversized_split, remaining_separators)

                Processor->>Processor: 使用下一个分隔符"\n"<br/>进一步分割过大的块

                Processor->>Processor: 如果仍然过大，继续使用" "分隔符<br/>最终使用""（字符级分割）

            else 无剩余分隔符
                Processor->>Processor: 强制添加过大块<br/>（无法进一步分割）
            end
        end
    end

    Processor->>Merger: 合并good_splits<br/>_merge_splits(good_splits, current_separator)

    Merger->>Merger: 智能合并相邻块<br/>考虑chunk_size和chunk_overlap

    Merger-->>Processor: 当前层级合并结果

    Processor->>Processor: 收集所有层级的结果<br/>final_chunks.extend(merged_chunks)

    Processor-->>Splitter: 递归分割完成<br/>所有块都满足大小要求

    Splitter-->>User: 最终分割结果<br/>["# 标题", "段落内容...", "分割后的块1", "分割后的块2", ...]
```

**递归算法核心**：

```python
def _recursive_split_logic(text: str, separators: List[str]) -> List[str]:
    """递归分割逻辑伪代码。"""

    # 1. 选择分隔符
    separator = None
    for sep in separators:
        if sep in text:
            separator = sep
            remaining_separators = separators[separators.index(sep) + 1:]
            break

    if separator is None:
        return [text]  # 无法分割

    # 2. 分割文本
    splits = text.split(separator)

    # 3. 处理每个分割块
    final_chunks = []
    good_splits = []

    for split in splits:
        if len(split) <= chunk_size:
            good_splits.append(split)
        else:
            # 先处理已收集的好块
            if good_splits:
                merged = merge_splits(good_splits, separator)
                final_chunks.extend(merged)
                good_splits = []

            # 递归处理过大的块
            if remaining_separators:
                recursive_result = _recursive_split_logic(split, remaining_separators)
                final_chunks.extend(recursive_result)
            else:
                final_chunks.append(split)  # 无法进一步分割

    # 处理剩余的好块
    if good_splits:
        merged = merge_splits(good_splits, separator)
        final_chunks.extend(merged)

    return final_chunks
```

---

## 2. 令牌分割场景

### 2.1 TokenTextSplitter 令牌处理流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Splitter as TokenTextSplitter
    participant Tokenizer as TikTokenEncoder
    participant Calculator as TokenCalculator
    participant Decoder as TokenDecoder
    participant Cache as TokenCache

    User->>Splitter: split_text("这是一个需要分割的长文本...")

    Splitter->>Cache: 检查文本是否已缓存<br/>cache_key = hash(text)

    alt 缓存命中
        Cache-->>Splitter: 返回缓存的令牌数据<br/>tokens = [cached_tokens]
    else 缓存未命中
        Splitter->>Tokenizer: 编码文本为令牌<br/>encode(text, allowed_special, disallowed_special)

        Tokenizer->>Tokenizer: 使用cl100k_base编码器<br/>处理中文和英文混合文本

        Tokenizer-->>Splitter: 令牌ID列表<br/>tokens = [123, 456, 789, ...]

        Splitter->>Cache: 缓存令牌结果<br/>put(cache_key, tokens)
    end

    Splitter->>Calculator: 计算分割参数<br/>total_tokens = len(tokens)<br/>chunk_size = 100, chunk_overlap = 20

    Calculator->>Calculator: 计算分割点<br/>start_idx = 0<br/>end_idx = min(100, total_tokens)

    loop 分割令牌序列
        Calculator->>Calculator: 提取当前块的令牌<br/>chunk_tokens = tokens[start_idx:end_idx]

        Calculator->>Decoder: 解码令牌为文本<br/>decode(chunk_tokens)

        Decoder->>Decoder: 将令牌ID转换回文本<br/>处理特殊令牌和编码问题

        Decoder-->>Calculator: 解码后的文本块<br/>chunk_text = "这是一个需要..."

        Calculator->>Calculator: 验证文本块质量<br/>检查是否为空或只有空白

        alt 文本块有效
            Calculator->>Calculator: 添加到结果列表<br/>chunks.append(chunk_text)
        else 文本块无效
            Calculator->>Calculator: 跳过无效块
        end

        Calculator->>Calculator: 计算下一个块的起始位置<br/>start_idx = end_idx - chunk_overlap<br/>end_idx = min(start_idx + chunk_size, total_tokens)

        alt 还有剩余令牌
            Calculator->>Calculator: 继续下一轮分割
        else 所有令牌已处理
            Calculator->>Calculator: 退出循环
        end
    end

    Calculator-->>Splitter: 分割完成<br/>chunks = ["块1", "块2", "块3", ...]

    Splitter->>Splitter: 更新统计信息<br/>total_texts_processed++<br/>total_tokens_counted += len(tokens)

    Splitter-->>User: 令牌分割结果<br/>每个块都精确控制在令牌限制内
```

**令牌处理特点**：

1. **精确控制**：每个块的令牌数量严格控制在指定范围内
2. **编码感知**：理解不同模型的令牌编码差异
3. **缓存优化**：缓存令牌化结果，提高重复处理效率
4. **重叠处理**：在令牌级别处理块间重叠

---

## 3. 专用分割场景

### 3.1 MarkdownHeaderTextSplitter 结构化分割流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Splitter as MarkdownHeaderTextSplitter
    participant Parser as MarkdownParser
    participant Structure as DocumentStructure
    participant Builder as DocumentBuilder

    User->>Splitter: split_text("# 主标题\n\n内容1\n\n## 二级标题\n\n内容2\n\n### 三级标题\n\n内容3")

    Splitter->>Parser: 解析Markdown结构<br/>headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]

    Parser->>Parser: 逐行扫描文本<br/>识别标题行和内容行

    loop 处理每一行
        Parser->>Parser: 检查行是否为标题<br/>line.startswith("#")

        alt 是标题行
            Parser->>Structure: 创建标题对象<br/>header = Header(level=1, text="主标题", line_num=1)

            Structure->>Structure: 添加到标题层次结构<br/>headers.append(header)

            Structure->>Structure: 更新当前上下文<br/>current_header = header

        else 是内容行
            Parser->>Structure: 添加内容到当前段落<br/>current_section.content += line
        end
    end

    Parser-->>Splitter: 解析完成<br/>结构化数据 = {headers: [...], sections: [...]}

    Splitter->>Builder: 构建文档块<br/>_aggregate_lines_to_chunks(structured_data)

    Builder->>Builder: 为每个段落创建Document对象

    loop 处理每个段落
        Builder->>Builder: 获取段落内容和所属标题<br/>section = {content: "内容1", header: "主标题"}

        Builder->>Builder: 构建元数据<br/>metadata = {<br/>  "Header 1": "主标题",<br/>  "start_index": 0,<br/>  "section_level": 1<br/>}

        Builder->>Builder: 创建Document对象<br/>doc = Document(page_content=content, metadata=metadata)

        Builder->>Builder: 添加到结果列表<br/>documents.append(doc)
    end

    Builder-->>Splitter: 文档构建完成<br/>documents = [doc1, doc2, doc3, ...]

    Splitter->>Splitter: 应用层次信息<br/>为每个文档添加完整的标题层次路径

    loop 处理标题层次
        Splitter->>Splitter: 计算标题路径<br/>"主标题 > 二级标题 > 三级标题"

        Splitter->>Splitter: 更新文档元数据<br/>添加完整的层次信息
    end

    Splitter-->>User: 结构化分割结果<br/>[<br/>  Document(content="内容1", metadata={"Header 1": "主标题"}),<br/>  Document(content="内容2", metadata={"Header 1": "主标题", "Header 2": "二级标题"}),<br/>  Document(content="内容3", metadata={"Header 1": "主标题", "Header 2": "二级标题", "Header 3": "三级标题"})<br/>]
```

**Markdown结构化处理**：

```python
class MarkdownStructureProcessor:
    def process_headers(self, text: str) -> List[HeaderInfo]:
        """处理标题结构。"""
        headers = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines):
            # 检测标题
            if line.strip().startswith('#'):
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break

                title = line[level:].strip()
                headers.append(HeaderInfo(
                    level=level,
                    title=title,
                    line_number=line_num,
                    start_index=sum(len(l) + 1 for l in lines[:line_num])
                ))

        return headers

    def build_hierarchy(self, headers: List[HeaderInfo]) -> Dict[str, Any]:
        """构建标题层次结构。"""
        hierarchy = {}
        stack = [hierarchy]

        for header in headers:
            # 调整栈深度
            while len(stack) > header.level:
                stack.pop()

            # 添加当前标题
            current_level = stack[-1]
            current_level[header.title] = {
                "level": header.level,
                "line_number": header.line_number,
                "children": {}
            }

            # 推入下一层
            stack.append(current_level[header.title]["children"])

        return hierarchy
```

---

### 3.2 PythonCodeTextSplitter 代码结构分割流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Splitter as PythonCodeTextSplitter
    participant AST as ASTParser
    participant Analyzer as CodeAnalyzer
    participant Segmenter as CodeSegmenter

    User->>Splitter: split_text("class MyClass:\n    def method1(self):\n        pass\n\ndef function1():\n    return True")

    Splitter->>AST: 解析Python代码结构<br/>ast.parse(code_text)

    AST->>AST: 构建抽象语法树<br/>识别类、函数、导入等结构

    AST-->>Splitter: AST节点树<br/>nodes = [ClassDef, FunctionDef, ...]

    Splitter->>Analyzer: 分析代码结构<br/>extract_code_elements(ast_nodes)

    Analyzer->>Analyzer: 遍历AST节点<br/>提取结构化信息

    loop 处理每个AST节点
        Analyzer->>Analyzer: 识别节点类型<br/>node_type = ClassDef | FunctionDef | Import

        alt 类定义节点
            Analyzer->>Analyzer: 提取类信息<br/>class_info = {<br/>  name: "MyClass",<br/>  start_line: 1,<br/>  end_line: 3,<br/>  methods: ["method1"]<br/>}

        else 函数定义节点
            Analyzer->>Analyzer: 提取函数信息<br/>function_info = {<br/>  name: "function1",<br/>  start_line: 5,<br/>  end_line: 6,<br/>  parameters: [],<br/>  is_method: False<br/>}

        else 导入节点
            Analyzer->>Analyzer: 提取导入信息<br/>import_info = {<br/>  module: "os",<br/>  items: ["path"],<br/>  line: 1<br/>}
        end
    end

    Analyzer-->>Splitter: 结构化代码信息<br/>code_structure = {<br/>  classes: [...],<br/>  functions: [...],<br/>  imports: [...]<br/>}

    Splitter->>Segmenter: 基于结构分割代码<br/>segment_by_structure(code_text, code_structure)

    Segmenter->>Segmenter: 使用Python特定分隔符<br/>separators = ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]

    Segmenter->>Segmenter: 应用递归分割策略<br/>优先保持代码结构完整性

    loop 处理代码段
        Segmenter->>Segmenter: 检查代码段完整性<br/>确保类和函数不被截断

        Segmenter->>Segmenter: 计算代码段大小<br/>考虑缩进和语法结构

        alt 代码段过大
            Segmenter->>Segmenter: 在合适的位置分割<br/>优先在函数边界分割
        else 代码段合适
            Segmenter->>Segmenter: 保持代码段完整
        end
    end

    Segmenter->>Segmenter: 添加代码元数据<br/>为每个段添加结构信息

    Segmenter-->>Splitter: 分割完成<br/>code_chunks = [<br/>  "class MyClass:\n    def method1(self):\n        pass",<br/>  "def function1():\n    return True"<br/>]

    Splitter-->>User: 代码分割结果<br/>保持语法完整性的代码块
```

**Python代码分割特点**：

1. **语法感知**：理解Python语法结构，避免破坏代码完整性
2. **结构优先**：优先在类、函数边界进行分割
3. **缩进保持**：保持Python代码的缩进结构
4. **元数据丰富**：为每个代码块添加结构化元数据

---

## 4. 文档处理场景

### 4.1 create_documents 文档创建流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Splitter as TextSplitter
    participant Processor as DocumentProcessor
    participant MetadataManager
    participant IndexManager

    User->>Splitter: create_documents(<br/>  texts=["文档1内容", "文档2内容"],<br/>  metadatas=[{"source": "doc1.txt"}, {"source": "doc2.txt"}]<br/>)

    Splitter->>Splitter: 验证输入参数<br/>检查texts和metadatas长度匹配

    alt 参数验证失败
        Splitter-->>User: raise ValueError("参数不匹配")
    end

    Splitter->>Processor: 开始批量文档处理<br/>process_documents(texts, metadatas)

    loop 处理每个文档
        Processor->>Processor: 获取当前文档<br/>text = texts[i], metadata = metadatas[i]

        Processor->>Splitter: 分割当前文档<br/>split_text(text)

        Splitter->>Splitter: 执行文本分割逻辑<br/>chunks = ["块1", "块2", "块3"]

        Splitter-->>Processor: 返回文档块<br/>text_chunks

        loop 处理每个文档块
            Processor->>MetadataManager: 构建块元数据<br/>build_chunk_metadata(original_metadata, chunk_index)

            MetadataManager->>MetadataManager: 复制原始元数据<br/>chunk_metadata = metadata.copy()

            alt add_start_index = True
                MetadataManager->>IndexManager: 计算起始索引<br/>start_index = calculate_start_index(chunk, original_text)

                IndexManager-->>MetadataManager: start_index = 150

                MetadataManager->>MetadataManager: 添加索引信息<br/>chunk_metadata["start_index"] = 150
            end

            MetadataManager->>MetadataManager: 添加块特定信息<br/>chunk_metadata.update({<br/>  "chunk_index": chunk_idx,<br/>  "total_chunks": len(chunks),<br/>  "chunk_length": len(chunk)<br/>})

            MetadataManager-->>Processor: 完整的块元数据

            Processor->>Processor: 创建Document对象<br/>doc = Document(<br/>  page_content=chunk,<br/>  metadata=chunk_metadata<br/>)

            Processor->>Processor: 添加到结果列表<br/>all_documents.append(doc)
        end
    end

    Processor-->>Splitter: 所有文档处理完成<br/>documents = [doc1, doc2, doc3, ...]

    Splitter->>Splitter: 更新分割统计<br/>total_documents_created += len(documents)<br/>total_original_texts += len(texts)

    Splitter-->>User: 文档创建完成<br/>List[Document] 包含所有分割后的文档块
```

**文档创建优化**：

```python
def create_documents_optimized(
    self,
    texts: List[str],
    metadatas: Optional[List[dict]] = None
) -> List[Document]:
    """优化的文档创建流程。"""

    # 1. 参数预处理
    if metadatas is None:
        metadatas = [{}] * len(texts)

    if len(texts) != len(metadatas):
        raise ValueError("texts和metadatas长度必须相同")

    # 2. 批量处理优化
    all_documents = []

    # 并行处理大批量文档
    if len(texts) > 100:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for text, metadata in zip(texts, metadatas):
                future = executor.submit(self._process_single_document, text, metadata)
                futures.append(future)

            for future in futures:
                documents = future.result()
                all_documents.extend(documents)
    else:
        # 串行处理小批量
        for text, metadata in zip(texts, metadatas):
            documents = self._process_single_document(text, metadata)
            all_documents.extend(documents)

    return all_documents

def _process_single_document(self, text: str, metadata: dict) -> List[Document]:
    """处理单个文档。"""
    chunks = self.split_text(text)
    documents = []

    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy()

        # 添加块信息
        chunk_metadata.update({
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_length": len(chunk)
        })

        # 添加起始索引（如果需要）
        if self._add_start_index:
            start_index = self._calculate_start_index(text, chunk, i)
            chunk_metadata["start_index"] = start_index

        documents.append(Document(
            page_content=chunk,
            metadata=chunk_metadata
        ))

    return documents
```

---

### 4.2 split_documents 文档分割流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Splitter
    participant DocumentProcessor
    participant MetadataPreserver
    participant ResultAggregator

    User->>Splitter: split_documents([<br/>  Document(content="长文档1", metadata={"source": "file1.txt"}),<br/>  Document(content="长文档2", metadata={"source": "file2.txt"})<br/>])

    Splitter->>DocumentProcessor: 开始批量文档分割<br/>process_document_list(documents)

    loop 处理每个文档
        DocumentProcessor->>DocumentProcessor: 提取文档内容和元数据<br/>content = doc.page_content<br/>original_metadata = doc.metadata

        DocumentProcessor->>Splitter: 分割文档内容<br/>split_text(content)

        Splitter->>Splitter: 执行分割逻辑<br/>chunks = self._split_implementation(content)

        Splitter-->>DocumentProcessor: 返回文本块<br/>text_chunks = ["块1", "块2", ...]

        DocumentProcessor->>MetadataPreserver: 准备元数据保持<br/>prepare_metadata_preservation(original_metadata, len(chunks))

        loop 为每个块创建文档
            MetadataPreserver->>MetadataPreserver: 创建块元数据<br/>chunk_metadata = original_metadata.copy()

            MetadataPreserver->>MetadataPreserver: 添加分割信息<br/>chunk_metadata.update({<br/>  "chunk_index": idx,<br/>  "parent_document": original_metadata.get("source"),<br/>  "split_timestamp": time.now()<br/>})

            alt 需要保持文档关系
                MetadataPreserver->>MetadataPreserver: 添加关系信息<br/>chunk_metadata.update({<br/>  "parent_doc_id": generate_doc_id(original_doc),<br/>  "sibling_chunks": len(chunks)<br/>})
            end

            MetadataPreserver->>DocumentProcessor: 创建新文档<br/>new_doc = Document(<br/>  page_content=chunk,<br/>  metadata=chunk_metadata<br/>)

            DocumentProcessor->>ResultAggregator: 添加到结果集<br/>add_document(new_doc)
        end
    end

    ResultAggregator->>ResultAggregator: 整理最终结果<br/>按原始文档顺序排列分割后的文档

    ResultAggregator->>ResultAggregator: 验证结果完整性<br/>检查是否有遗漏或重复

    ResultAggregator-->>Splitter: 分割完成<br/>split_documents = [doc1_chunk1, doc1_chunk2, doc2_chunk1, ...]

    Splitter->>Splitter: 更新处理统计<br/>documents_processed += len(original_docs)<br/>chunks_created += len(split_documents)

    Splitter-->>User: 文档分割结果<br/>List[Document] 保持原有元数据并添加分割信息
```

---

## 5. 性能优化场景

### 5.1 批量分割优化流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant BatchProcessor
    participant Splitter
    participant Cache as SplitterCache
    participant Pool as ThreadPool
    participant Monitor as PerformanceMonitor

    User->>BatchProcessor: batch_split_texts([text1, text2, ..., text1000])

    BatchProcessor->>Monitor: 开始性能监控<br/>start_batch_operation()

    BatchProcessor->>BatchProcessor: 分析批量大小<br/>batch_size = 1000, 决定处理策略

    alt 大批量处理 (>100)
        BatchProcessor->>Pool: 创建线程池<br/>ThreadPoolExecutor(max_workers=4)

        BatchProcessor->>BatchProcessor: 将文本分组<br/>groups = [texts[0:250], texts[250:500], ...]

        par 并行处理各组
            BatchProcessor->>Pool: 提交任务组1<br/>submit(process_group, group1)
            Pool->>Splitter: 处理group1文本

            loop 处理组内文本
                Splitter->>Cache: 检查缓存<br/>cache_key = hash(text + config)

                alt 缓存命中
                    Cache-->>Splitter: 返回缓存结果<br/>cached_chunks
                    Splitter->>Monitor: 记录缓存命中<br/>cache_hit++
                else 缓存未命中
                    Splitter->>Splitter: 执行实际分割<br/>split_text(text)
                    Splitter->>Cache: 缓存结果<br/>put(cache_key, chunks)
                    Splitter->>Monitor: 记录缓存未命中<br/>cache_miss++
                end
            end

            Pool-->>BatchProcessor: 组1处理完成<br/>group1_results
        and
            BatchProcessor->>Pool: 提交任务组2<br/>submit(process_group, group2)
            Pool-->>BatchProcessor: 组2处理完成<br/>group2_results
        and
            BatchProcessor->>Pool: 提交任务组3<br/>submit(process_group, group3)
            Pool-->>BatchProcessor: 组3处理完成<br/>group3_results
        and
            BatchProcessor->>Pool: 提交任务组4<br/>submit(process_group, group4)
            Pool-->>BatchProcessor: 组4处理完成<br/>group4_results
        end

        BatchProcessor->>BatchProcessor: 合并所有结果<br/>all_results = group1 + group2 + group3 + group4

    else 小批量处理 (<=100)
        BatchProcessor->>Splitter: 串行处理<br/>sequential_processing(texts)

        loop 逐个处理文本
            Splitter->>Cache: 检查缓存
            Splitter->>Splitter: 分割文本
            Splitter->>Monitor: 记录处理时间
        end

        Splitter-->>BatchProcessor: 串行结果<br/>sequential_results
    end

    BatchProcessor->>Monitor: 结束性能监控<br/>end_batch_operation()

    Monitor->>Monitor: 计算性能指标<br/>throughput = total_texts / total_time<br/>cache_hit_rate = hits / (hits + misses)<br/>average_chunk_size = total_chunks / total_texts

    Monitor-->>BatchProcessor: 性能报告<br/>{<br/>  "throughput": "500 texts/sec",<br/>  "cache_hit_rate": "75%",<br/>  "total_chunks": 2500<br/>}

    BatchProcessor-->>User: 批量处理完成<br/>results + performance_stats
```

**批量优化策略**：

| 批量大小 | 处理策略 | 并发数 | 缓存策略 | 预期性能 |
|---------|---------|--------|---------|---------|
| 1-10 | 串行处理 | 1 | 基础缓存 | 100% |
| 11-100 | 串行+缓存 | 1 | 智能缓存 | 300% |
| 101-1000 | 并行处理 | 4 | 分布式缓存 | 800% |
| 1000+ | 分批+并行 | 8 | 预加载缓存 | 1500% |

---

### 5.2 智能缓存管理流程

```mermaid
sequenceDiagram
    autonumber
    participant Splitter
    participant CacheManager
    participant LRUCache
    participant HashCalculator
    participant StatsCollector

    Splitter->>CacheManager: 请求分割结果<br/>get_or_split(text, config)

    CacheManager->>HashCalculator: 生成缓存键<br/>calculate_cache_key(text, config)

    HashCalculator->>HashCalculator: 计算文本哈希<br/>text_hash = md5(text)<br/>config_hash = md5(str(config))

    HashCalculator-->>CacheManager: cache_key = f"{text_hash}_{config_hash}"

    CacheManager->>LRUCache: 查找缓存<br/>get(cache_key)

    alt 缓存命中
        LRUCache->>LRUCache: 更新访问时间<br/>move_to_end(cache_key)

        LRUCache-->>CacheManager: 缓存结果<br/>cached_chunks

        CacheManager->>StatsCollector: 记录命中<br/>record_cache_hit(cache_key, hit_time)

        CacheManager-->>Splitter: 返回缓存结果<br/>chunks (< 1ms)

    else 缓存未命中
        LRUCache-->>CacheManager: None

        CacheManager->>Splitter: 执行实际分割<br/>perform_split(text, config)

        Splitter->>Splitter: 分割处理<br/>split_text_implementation()

        Splitter-->>CacheManager: 分割结果<br/>chunks (10-100ms)

        CacheManager->>CacheManager: 评估缓存价值<br/>should_cache(text_length, split_time, chunk_count)

        alt 值得缓存
            CacheManager->>LRUCache: 存储结果<br/>put(cache_key, chunks, ttl=3600)

            LRUCache->>LRUCache: 检查容量限制<br/>if len(cache) > max_size: evict_lru()

            alt 需要淘汰
                LRUCache->>LRUCache: 淘汰最久未使用项<br/>evicted_key = popitem(last=False)

                LRUCache->>StatsCollector: 记录淘汰<br/>record_eviction(evicted_key)
            end

        else 不值得缓存
            CacheManager->>StatsCollector: 记录跳过<br/>record_cache_skip(reason="too_small")
        end

        CacheManager->>StatsCollector: 记录未命中<br/>record_cache_miss(cache_key, split_time)

        CacheManager-->>Splitter: 返回分割结果<br/>chunks
    end

    StatsCollector->>StatsCollector: 更新统计信息<br/>hit_rate = hits / (hits + misses)<br/>average_split_time = total_time / operations<br/>cache_efficiency = saved_time / total_time
```

**缓存策略优化**：

```python
class IntelligentCacheManager:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_saved_time": 0.0
        }

    def should_cache(self, text_length: int, split_time: float, chunk_count: int) -> bool:
        """智能缓存决策。"""
        # 缓存策略：
        # 1. 长文本优先缓存（处理时间长）
        # 2. 复杂分割优先缓存（块数多）
        # 3. 避免缓存一次性文本

        if text_length < 100:  # 太短，不值得缓存
            return False

        if split_time < 0.01:  # 处理太快，缓存收益小
            return False

        if chunk_count < 2:  # 分割结果简单
            return False

        # 预估缓存价值
        cache_value_score = (
            text_length * 0.001 +  # 文本长度权重
            split_time * 100 +     # 处理时间权重
            chunk_count * 10       # 复杂度权重
        )

        return cache_value_score > 50  # 阈值

    def evict_intelligently(self) -> None:
        """智能淘汰策略。"""
        if len(self.cache) <= self.max_size:
            return

        # 按访问频率和时间综合评分
        candidates = []
        for key, entry in self.cache.items():
            score = (
                entry.access_count * 0.3 +  # 访问频率
                (time.time() - entry.last_access) * -0.001 +  # 最近访问时间
                entry.cache_value * 0.7  # 缓存价值
            )
            candidates.append((key, score))

        # 淘汰评分最低的项
        candidates.sort(key=lambda x: x[1])
        to_evict = candidates[:len(self.cache) - self.max_size + 1]

        for key, _ in to_evict:
            del self.cache[key]
            self.stats["evictions"] += 1
```

---

## 6. 总结

本文档详细展示了 **Text Splitters 模块**的关键执行时序：

1. **基础分割**：CharacterTextSplitter和RecursiveCharacterTextSplitter的分割策略和递归处理
2. **令牌处理**：TokenTextSplitter的精确令牌控制和编码处理机制
3. **专用分割**：MarkdownHeaderTextSplitter和PythonCodeTextSplitter的结构化处理
4. **文档处理**：create_documents和split_documents的批量处理流程
5. **性能优化**：批量处理、智能缓存和并发优化策略

每张时序图包含：
- 详细的参与者交互过程
- 关键算法和处理逻辑
- 性能优化点和缓存策略
- 错误处理和边界情况
- 统计信息收集和监控

这些时序图帮助开发者深入理解文本分割系统的内部工作机制，为构建高效、可靠的文档处理管道提供指导。Text Splitters是RAG系统和文档处理应用的基础组件，正确理解其执行流程对提高文档处理质量和系统性能至关重要。

通过递归分割、智能缓存、批量优化等技术，Text Splitters模块能够高效处理各种类型和规模的文本数据，为下游的向量化、检索和生成任务提供优质的输入。
