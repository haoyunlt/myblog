# LangChain-10-TextSplitters-API

## 文档说明

本文档详细描述 **Text Splitters 模块**的对外 API，包括文本分割、块管理、重叠处理、分隔符策略等核心接口的所有公开方法和参数规格。

---

## 1. TextSplitter 基础 API

### 1.1 基础接口

#### 基本信息
- **类名**：`TextSplitter`
- **功能**：文本分割的抽象基类
- **核心职责**：将长文本分割为适合处理的小块

#### 核心方法

```python
class TextSplitter(ABC):
    """文本分割器基类。"""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ):
        """初始化文本分割器。"""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """分割文本为字符串列表。"""

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[Document]:
        """创建文档对象列表。"""

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表。"""
```

**方法详解**：

| 方法 | 参数 | 返回类型 | 说明 |
|-----|------|---------|------|
| split_text | `text: str` | `List[str]` | 将文本分割为字符串块列表 |
| create_documents | `texts: List[str]`, `metadatas: List[dict]` | `List[Document]` | 创建带元数据的文档对象 |
| split_documents | `documents: List[Document]` | `List[Document]` | 分割现有文档列表 |

#### 构造参数详解

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| chunk_size | `int` | `4000` | 每个块的最大大小 |
| chunk_overlap | `int` | `200` | 块之间的重叠字符数 |
| length_function | `Callable` | `len` | 计算文本长度的函数 |
| keep_separator | `bool` | `False` | 是否保留分隔符 |
| add_start_index | `bool` | `False` | 是否添加起始索引到元数据 |
| strip_whitespace | `bool` | `True` | 是否去除空白字符 |

---

## 2. CharacterTextSplitter API

### 2.1 字符分割器

#### 基本信息
- **功能**：基于指定分隔符分割文本
- **特点**：简单直接，适用于结构化文本
- **适用场景**：有明确分隔符的文本（如段落、句子）

#### 构造参数

```python
class CharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ):
        """字符文本分割器构造函数。"""
```

#### 使用示例

```python
from langchain_text_splitters import CharacterTextSplitter

# 基础用法 - 按段落分割
text = """
第一段内容。这是一个完整的段落，包含了相关的信息。

第二段内容。这是另一个段落，讨论不同的主题。

第三段内容。最后一个段落，总结前面的内容。
"""

splitter = CharacterTextSplitter(
    separator="\n\n",  # 按双换行分割
    chunk_size=100,    # 每块最大100字符
    chunk_overlap=20,  # 重叠20字符
    length_function=len,
    is_separator_regex=False
)

chunks = splitter.split_text(text)
print(f"分割后的块数: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"块 {i+1}: {repr(chunk)}")

# 输出:
# 块 1: '第一段内容。这是一个完整的段落，包含了相关的信息。'
# 块 2: '第二段内容。这是另一个段落，讨论不同的主题。'
# 块 3: '第三段内容。最后一个段落，总结前面的内容。'
```

#### 正则表达式分割

```python
import re

# 使用正则表达式分割
text = "句子1。句子2！句子3？句子4。"

regex_splitter = CharacterTextSplitter(
    separator=r'[。！？]',  # 按中文标点分割
    chunk_size=50,
    chunk_overlap=5,
    is_separator_regex=True,
    keep_separator=True  # 保留分隔符
)

chunks = regex_splitter.split_text(text)
print("正则分割结果:")
for chunk in chunks:
    print(f"- {repr(chunk)}")

# 输出:
# - '句子1。'
# - '句子2！'
# - '句子3？'
# - '句子4。'
```

#### 核心实现

```python
def split_text(self, text: str) -> List[str]:
    """分割文本实现。"""
    # 按分隔符分割
    if self.is_separator_regex:
        splits = re.split(self.separator, text)
    else:
        splits = text.split(self.separator)

    # 处理分隔符保留
    if self.keep_separator and not self.is_separator_regex:
        # 重新添加分隔符
        result = []
        for i, split in enumerate(splits[:-1]):
            result.append(split + self.separator)
        if splits:
            result.append(splits[-1])
        splits = result

    # 合并小块并处理重叠
    return self._merge_splits(splits, self.separator)
```

---

## 3. RecursiveCharacterTextSplitter API

### 3.1 递归字符分割器

#### 基本信息
- **功能**：递归使用多个分隔符分割文本
- **特点**：智能选择最佳分隔符，保持语义完整性
- **适用场景**：通用文本分割，特别是代码和结构化文档

#### 构造参数

```python
class RecursiveCharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ):
        """递归字符文本分割器构造函数。"""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or self._get_default_separators()
        self._is_separator_regex = is_separator_regex
```

#### 默认分隔符优先级

```python
def _get_default_separators(self) -> List[str]:
    """获取默认分隔符列表（按优先级排序）。"""
    return [
        "\n\n",    # 段落分隔
        "\n",      # 行分隔
        " ",       # 词分隔
        "",        # 字符分隔
    ]
```

#### 使用示例

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 长文本示例
long_text = """
# 标题

这是第一段内容。它包含了一些重要的信息，需要被正确地分割。

这是第二段内容。它继续前面的讨论，并添加了新的观点。

## 子标题

这里是一个列表：
- 项目1：描述内容
- 项目2：更多描述
- 项目3：最后的描述

结论段落包含了总结性的内容。
"""

# 创建递归分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(long_text)
print(f"递归分割结果 - 块数: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n块 {i+1} ({len(chunk)} 字符):")
    print("-" * 40)
    print(chunk)
    print("-" * 40)
```

#### 自定义分隔符策略

```python
# 代码分割器
code_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\nclass ",    # 类定义
        "\n\ndef ",     # 函数定义
        "\n\n",         # 段落
        "\n",           # 行
        " ",            # 空格
        "",             # 字符
    ],
    chunk_size=500,
    chunk_overlap=50,
    keep_separator=True
)

# Markdown分割器
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n## ",        # 二级标题
        "\n### ",       # 三级标题
        "\n\n",         # 段落
        "\n",           # 行
        " ",            # 空格
        "",             # 字符
    ],
    chunk_size=300,
    chunk_overlap=30
)
```

#### 核心递归算法

```python
def split_text(self, text: str) -> List[str]:
    """递归分割文本。"""
    final_chunks = []

    # 选择合适的分隔符
    separator = self._separators[-1]  # 默认使用最后一个
    new_separators = []

    for i, _s in enumerate(self._separators):
        if _s == "":
            separator = _s
            break
        if re.search(_s, text) if self._is_separator_regex else _s in text:
            separator = _s
            new_separators = self._separators[i + 1:]
            break

    # 使用选定的分隔符分割
    splits = self._split_text_with_regex(text, separator) if self._is_separator_regex else text.split(separator)

    # 处理每个分割块
    good_splits = []
    for s in splits:
        if self._length_function(s) < self._chunk_size:
            good_splits.append(s)
        else:
            if good_splits:
                merged_text = self._merge_splits(good_splits, separator)
                final_chunks.extend(merged_text)
                good_splits = []

            # 递归处理过大的块
            if not new_separators:
                final_chunks.append(s)
            else:
                other_info = self.split_text(s)
                final_chunks.extend(other_info)

    # 处理剩余的块
    if good_splits:
        merged_text = self._merge_splits(good_splits, separator)
        final_chunks.extend(merged_text)

    return final_chunks
```

---

## 4. TokenTextSplitter API

### 4.1 令牌分割器

#### 基本信息
- **功能**：基于令牌数量分割文本
- **特点**：精确控制令牌数量，适用于LLM输入
- **适用场景**：需要严格控制令牌数的应用

#### 构造参数

```python
class TokenTextSplitter(TextSplitter):
    def __init__(
        self,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ):
        """令牌文本分割器构造函数。"""
```

#### 使用示例

```python
from langchain_text_splitters import TokenTextSplitter

# 创建令牌分割器
token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # GPT-4 编码
    chunk_size=100,               # 100个令牌
    chunk_overlap=20              # 20个令牌重叠
)

text = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
"""

chunks = token_splitter.split_text(text)
print(f"令牌分割结果 - 块数: {len(chunks)}")

for i, chunk in enumerate(chunks):
    token_count = token_splitter.count_tokens(chunk)
    print(f"块 {i+1} ({token_count} 令牌): {chunk}")
```

#### 不同编码器的使用

```python
# GPT-3.5/GPT-4 编码器
gpt4_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=2000,
    chunk_overlap=200
)

# GPT-3 编码器
gpt3_splitter = TokenTextSplitter(
    encoding_name="p50k_base",
    chunk_size=1500,
    chunk_overlap=150
)

# Claude 编码器
claude_splitter = TokenTextSplitter(
    encoding_name="gpt2",  # 近似
    chunk_size=1800,
    chunk_overlap=180
)

# 比较不同编码器的令牌计数
test_text = "这是一个测试文本，用来比较不同编码器的令牌计数差异。"

print("令牌计数比较:")
print(f"GPT-4: {gpt4_splitter.count_tokens(test_text)} 令牌")
print(f"GPT-3: {gpt3_splitter.count_tokens(test_text)} 令牌")
print(f"近似: {claude_splitter.count_tokens(test_text)} 令牌")
```

#### 令牌计数实现

```python
def count_tokens(self, text: str) -> int:
    """计算文本的令牌数量。"""
    return len(self._tokenizer.encode(text))

def split_text(self, text: str) -> List[str]:
    """基于令牌分割文本。"""
    splits = []
    input_ids = self._tokenizer.encode(text)

    start_idx = 0
    cur_idx = min(start_idx + self._chunk_size, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]

    while start_idx < len(input_ids):
        chunk_text = self._tokenizer.decode(chunk_ids)
        splits.append(chunk_text)

        # 计算下一个块的起始位置（考虑重叠）
        start_idx += self._chunk_size - self._chunk_overlap
        cur_idx = min(start_idx + self._chunk_size, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return splits
```

---

## 5. 专用分割器 API

### 5.1 MarkdownHeaderTextSplitter

#### 基本信息
- **功能**：基于Markdown标题层次分割文本
- **特点**：保持文档结构，支持标题层次
- **适用场景**：Markdown文档、技术文档

#### 使用示例

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# 主标题

这是主标题下的内容。

## 二级标题1

这是二级标题1的内容。

### 三级标题1.1

这是三级标题1.1的内容。

### 三级标题1.2

这是三级标题1.2的内容。

## 二级标题2

这是二级标题2的内容。
"""

# 定义标题分割规则
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_header_splits = markdown_splitter.split_text(markdown_text)

for split in md_header_splits:
    print(f"内容: {split.page_content}")
    print(f"元数据: {split.metadata}")
    print("-" * 50)
```

#### 与其他分割器组合

```python
# 先按标题分割，再按字符数分割
chunk_size = 200
chunk_overlap = 30

# 第一步：按标题分割
md_header_splits = markdown_splitter.split_text(markdown_text)

# 第二步：对长段落进行字符分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

final_splits = text_splitter.split_documents(md_header_splits)

print(f"最终分割结果: {len(final_splits)} 个块")
for i, split in enumerate(final_splits):
    print(f"\n块 {i+1}:")
    print(f"内容: {split.page_content[:100]}...")
    print(f"元数据: {split.metadata}")
```

---

### 5.2 PythonCodeTextSplitter

#### 基本信息
- **功能**：专门用于Python代码的分割
- **特点**：理解Python语法结构
- **适用场景**：代码文档、代码分析

#### 使用示例

```python
from langchain_text_splitters import PythonCodeTextSplitter

python_code = '''
class DataProcessor:
    """数据处理器类。"""

    def __init__(self, config):
        self.config = config
        self.data = []

    def load_data(self, file_path):
        """加载数据文件。"""
        with open(file_path, 'r') as f:
            self.data = f.readlines()
        return len(self.data)

    def process_data(self):
        """处理数据。"""
        processed = []
        for line in self.data:
            # 清理数据
            cleaned = line.strip()
            if cleaned:
                processed.append(cleaned.upper())
        return processed

def main():
    """主函数。"""
    processor = DataProcessor({"debug": True})
    processor.load_data("data.txt")
    result = processor.process_data()
    print(f"处理了 {len(result)} 条数据")

if __name__ == "__main__":
    main()
'''

python_splitter = PythonCodeTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

python_chunks = python_splitter.split_text(python_code)

print(f"Python代码分割结果: {len(python_chunks)} 个块")
for i, chunk in enumerate(python_chunks):
    print(f"\n代码块 {i+1}:")
    print("-" * 40)
    print(chunk)
    print("-" * 40)
```

---

### 5.3 HTMLHeaderTextSplitter

#### 基本信息
- **功能**：基于HTML标题标签分割文本
- **特点**：保持HTML文档结构
- **适用场景**：网页内容、HTML文档

#### 使用示例

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_text = """
<html>
<body>
    <h1>网站主标题</h1>
    <p>这是网站的主要介绍内容。</p>

    <h2>产品介绍</h2>
    <p>我们的产品具有以下特点：</p>
    <ul>
        <li>高性能</li>
        <li>易使用</li>
        <li>可扩展</li>
    </ul>

    <h3>技术规格</h3>
    <p>详细的技术规格信息。</p>

    <h2>联系我们</h2>
    <p>如有疑问，请联系我们。</p>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

html_header_splits = html_splitter.split_text(html_text)

for split in html_header_splits:
    print(f"内容: {split.page_content}")
    print(f"元数据: {split.metadata}")
    print("-" * 50)
```

---

## 6. 文档处理 API

### 6.1 文档创建和分割

#### create_documents 方法

```python
def create_documents(
    self,
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
) -> List[Document]:
    """从文本列表创建文档对象。"""

    # 使用示例
    texts = [
        "第一个文档的内容...",
        "第二个文档的内容...",
        "第三个文档的内容..."
    ]

    metadatas = [
        {"source": "doc1.txt", "author": "张三"},
        {"source": "doc2.txt", "author": "李四"},
        {"source": "doc3.txt", "author": "王五"}
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    documents = splitter.create_documents(texts, metadatas)

    print(f"创建了 {len(documents)} 个文档")
    for doc in documents:
        print(f"内容: {doc.page_content[:50]}...")
        print(f"元数据: {doc.metadata}")
```

#### split_documents 方法

```python
def split_documents(self, documents: List[Document]) -> List[Document]:
    """分割现有文档列表。"""

    # 使用示例
    from langchain_core.documents import Document

    # 创建原始文档
    original_docs = [
        Document(
            page_content="这是一个很长的文档内容..." * 100,
            metadata={"source": "long_doc.txt", "type": "article"}
        ),
        Document(
            page_content="另一个长文档的内容..." * 80,
            metadata={"source": "another_doc.txt", "type": "report"}
        )
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True  # 添加起始索引
    )

    split_docs = splitter.split_documents(original_docs)

    print(f"原始文档数: {len(original_docs)}")
    print(f"分割后文档数: {len(split_docs)}")

    for i, doc in enumerate(split_docs[:3]):  # 显示前3个
        print(f"\n分割文档 {i+1}:")
        print(f"内容: {doc.page_content[:100]}...")
        print(f"元数据: {doc.metadata}")
```

---

## 7. 高级配置和优化

### 7.1 自定义长度函数

```python
import tiktoken

def token_length_function(text: str) -> int:
    """基于GPT令牌的长度函数。"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def chinese_char_length(text: str) -> int:
    """中文字符计数函数。"""
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    english_chars = sum(1 for char in text if char.isalpha() and not ('\u4e00' <= char <= '\u9fff'))
    return chinese_chars * 2 + english_chars  # 中文字符权重更高

# 使用自定义长度函数
custom_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=token_length_function  # 使用令牌计数
)

chinese_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=chinese_char_length  # 中文优化
)
```

### 7.2 分割策略优化

```python
class SmartTextSplitter:
    """智能文本分割器。"""

    def __init__(self, target_chunk_size: int = 1000):
        self.target_chunk_size = target_chunk_size
        self.splitters = {
            'markdown': MarkdownHeaderTextSplitter([
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]),
            'code': PythonCodeTextSplitter(
                chunk_size=target_chunk_size,
                chunk_overlap=100
            ),
            'general': RecursiveCharacterTextSplitter(
                chunk_size=target_chunk_size,
                chunk_overlap=100
            )
        }

    def detect_text_type(self, text: str) -> str:
        """检测文本类型。"""
        if text.count('#') > 3 and '##' in text:
            return 'markdown'
        elif 'def ' in text and 'class ' in text and ':' in text:
            return 'code'
        else:
            return 'general'

    def smart_split(self, text: str) -> List[str]:
        """智能分割文本。"""
        text_type = self.detect_text_type(text)
        splitter = self.splitters[text_type]

        if text_type == 'markdown':
            # Markdown需要两步分割
            header_splits = splitter.split_text(text)
            final_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.target_chunk_size,
                chunk_overlap=100
            )
            return final_splitter.split_documents(header_splits)
        else:
            return splitter.split_text(text)

# 使用智能分割器
smart_splitter = SmartTextSplitter(target_chunk_size=800)

# 测试不同类型的文本
markdown_text = "# 标题\n\n内容..."
code_text = "def function():\n    pass"
general_text = "这是一般的文本内容。"

for text_type, text in [('Markdown', markdown_text), ('Code', code_text), ('General', general_text)]:
    chunks = smart_splitter.smart_split(text)
    print(f"{text_type} 文本分割结果: {len(chunks)} 个块")
```

---

## 8. 性能监控和调优

### 8.1 分割性能分析

```python
import time
from typing import Dict, Any

class SplitterPerformanceAnalyzer:
    """分割器性能分析器。"""

    def __init__(self):
        self.metrics = {
            'split_times': [],
            'chunk_counts': [],
            'chunk_sizes': [],
            'overlap_ratios': []
        }

    def analyze_splitter(
        self,
        splitter: TextSplitter,
        texts: List[str],
        test_name: str = "default"
    ) -> Dict[str, Any]:
        """分析分割器性能。"""
        start_time = time.time()

        all_chunks = []
        for text in texts:
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)

        end_time = time.time()

        # 计算统计信息
        split_time = end_time - start_time
        chunk_count = len(all_chunks)
        avg_chunk_size = sum(len(chunk) for chunk in all_chunks) / chunk_count if chunk_count > 0 else 0

        # 计算重叠比率
        total_original_length = sum(len(text) for text in texts)
        total_chunk_length = sum(len(chunk) for chunk in all_chunks)
        overlap_ratio = (total_chunk_length - total_original_length) / total_original_length if total_original_length > 0 else 0

        metrics = {
            'test_name': test_name,
            'split_time': split_time,
            'chunk_count': chunk_count,
            'avg_chunk_size': avg_chunk_size,
            'overlap_ratio': overlap_ratio,
            'throughput': len(texts) / split_time if split_time > 0 else 0,
            'chunks_per_text': chunk_count / len(texts) if texts else 0
        }

        return metrics

    def compare_splitters(self, splitters: Dict[str, TextSplitter], test_texts: List[str]):
        """比较多个分割器的性能。"""
        results = {}

        for name, splitter in splitters.items():
            results[name] = self.analyze_splitter(splitter, test_texts, name)

        # 打印比较结果
        print("分割器性能比较:")
        print("-" * 80)
        print(f"{'分割器':<20} {'时间(s)':<10} {'块数':<8} {'平均大小':<10} {'重叠率':<10} {'吞吐量':<10}")
        print("-" * 80)

        for name, metrics in results.items():
            print(f"{name:<20} {metrics['split_time']:<10.3f} {metrics['chunk_count']:<8} "
                  f"{metrics['avg_chunk_size']:<10.1f} {metrics['overlap_ratio']:<10.2%} "
                  f"{metrics['throughput']:<10.1f}")

        return results

# 使用性能分析器
analyzer = SplitterPerformanceAnalyzer()

# 准备测试数据
test_texts = [
    "这是一个测试文本..." * 100,
    "另一个测试文本..." * 150,
    "第三个测试文本..." * 80
]

# 准备不同的分割器
splitters_to_test = {
    "Character": CharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    "Recursive": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    "Token": TokenTextSplitter(chunk_size=100, chunk_overlap=10)
}

# 执行性能比较
performance_results = analyzer.compare_splitters(splitters_to_test, test_texts)
```

---

## 9. 最佳实践和配置指南

### 9.1 分割器选择指南

| 文本类型 | 推荐分割器 | 配置建议 | 使用场景 |
|---------|-----------|---------|---------|
| 通用文本 | `RecursiveCharacterTextSplitter` | chunk_size=1000, overlap=100 | 大多数文本处理 |
| Markdown | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | 先按标题，再按大小 | 技术文档、博客 |
| 代码 | `PythonCodeTextSplitter` | chunk_size=800, overlap=100 | 代码分析、文档 |
| HTML | `HTMLHeaderTextSplitter` | 按标签层次分割 | 网页内容提取 |
| 令牌敏感 | `TokenTextSplitter` | 根据模型限制设置 | LLM输入准备 |

### 9.2 参数调优建议

```python
def get_optimal_splitter_config(
    text_type: str,
    target_model: str,
    use_case: str
) -> Dict[str, Any]:
    """获取最优分割器配置。"""

    configs = {
        # RAG应用配置
        "rag": {
            "general": {
                "splitter": RecursiveCharacterTextSplitter,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", "。", "！", "？", " ", ""]
            },
            "technical": {
                "splitter": RecursiveCharacterTextSplitter,
                "chunk_size": 1500,
                "chunk_overlap": 300,
                "separators": ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            }
        },

        # 摘要应用配置
        "summarization": {
            "general": {
                "splitter": RecursiveCharacterTextSplitter,
                "chunk_size": 2000,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n", " ", ""]
            }
        },

        # 问答应用配置
        "qa": {
            "general": {
                "splitter": RecursiveCharacterTextSplitter,
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n", "。", " ", ""]
            }
        }
    }

    return configs.get(use_case, {}).get(text_type, configs["rag"]["general"])

# 使用配置生成器
config = get_optimal_splitter_config(
    text_type="general",
    target_model="gpt-3.5-turbo",
    use_case="rag"
)

optimal_splitter = config["splitter"](
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"],
    separators=config.get("separators")
)
```

---

## 10. 总结

本文档详细描述了 **Text Splitters 模块**的核心 API：

### 主要分割器类型
1. **TextSplitter**：抽象基类，定义通用接口
2. **CharacterTextSplitter**：基于字符分隔符的简单分割
3. **RecursiveCharacterTextSplitter**：递归多分隔符智能分割
4. **TokenTextSplitter**：基于令牌数量的精确分割
5. **专用分割器**：Markdown、Python代码、HTML等专门分割器

### 核心功能
1. **文本分割**：split_text方法将长文本分割为小块
2. **文档处理**：create_documents和split_documents处理文档对象
3. **重叠控制**：chunk_overlap参数控制块间重叠
4. **长度控制**：自定义length_function精确控制块大小

### 配置参数
1. **chunk_size**：控制每个块的最大大小
2. **chunk_overlap**：控制块间重叠程度
3. **separators**：定义分割优先级策略
4. **length_function**：自定义长度计算方法

每个 API 均包含：
- 完整的构造参数和配置选项
- 详细的使用示例和最佳实践
- 性能分析和优化建议
- 不同场景的配置指南

Text Splitters 模块是文档处理和RAG系统的基础组件，正确选择和配置分割器对提高下游任务的效果至关重要。
