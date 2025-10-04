# LangChain-06-OutputParsers-API

## 文档说明

本文档详细描述 **Output Parsers 模块**的对外 API，包括各种解析器类型、结构化输出、流式解析、错误处理等核心接口的所有公开方法和参数规格。

---

## 1. BaseOutputParser 核心 API

### 1.1 基础接口

#### 基本信息
- **类名**：`BaseOutputParser[T]`
- **功能**：所有输出解析器的抽象基类
- **泛型参数**：`T` 表示解析后的目标类型

#### 核心方法

```python
class BaseOutputParser(Generic[T], Runnable[Union[str, BaseMessage], T]):
    """输出解析器基类。"""

    def parse(self, text: str) -> T:
        """解析文本为目标类型。"""
        raise NotImplementedError

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> T:
        """解析生成结果。"""
        return self.parse(result[0].text)

    def get_format_instructions(self) -> str:
        """获取格式说明。"""
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """解析器类型标识。"""
        raise NotImplementedError
```

**字段表**：

| 方法 | 参数 | 返回类型 | 说明 |
|-----|------|---------|------|
| parse | `text: str` | `T` | 解析文本为目标类型 |
| parse_result | `result: List[Generation]` | `T` | 解析LLM生成结果 |
| get_format_instructions | 无 | `str` | 获取格式说明给LLM |
| _type | 属性 | `str` | 解析器类型标识 |

#### 使用示例

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class ListOutputParser(BaseOutputParser[List[str]]):
    """列表输出解析器示例。"""

    def parse(self, text: str) -> List[str]:
        """解析文本为字符串列表。"""
        lines = text.strip().split('\n')
        return [line.strip('- ').strip() for line in lines if line.strip()]

    def get_format_instructions(self) -> str:
        """格式说明。"""
        return "请以列表格式输出，每行一个项目，前面加破折号，例如：\n- 项目1\n- 项目2\n- 项目3"

    @property
    def _type(self) -> str:
        return "list"

# 使用
parser = ListOutputParser()
result = parser.parse("- 苹果\n- 香蕉\n- 橙子")
print(result)  # ['苹果', '香蕉', '橙子']
```

---

## 2. 核心解析器 API

### 2.1 StrOutputParser - 字符串解析器

#### 基本信息
- **功能**：返回原始字符串，最简单的解析器
- **适用场景**：不需要特殊处理的文本输出

#### 方法签名

```python
class StrOutputParser(BaseOutputParser[str]):
    """字符串输出解析器。"""

    def parse(self, text: str) -> str:
        """直接返回文本。"""
        return text
```

#### 使用示例

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 创建链
prompt = ChatPromptTemplate.from_template("写一首关于{topic}的诗")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser

# 使用
result = chain.invoke({"topic": "春天"})
print(type(result))  # <class 'str'>
print(result)        # "春风拂面暖人心..."
```

---

### 2.2 JsonOutputParser - JSON解析器

#### 基本信息
- **功能**：解析JSON格式的输出
- **支持特性**：部分解析、错误恢复、Schema验证

#### 方法签名

```python
class JsonOutputParser(BaseOutputParser[Any]):
    """JSON输出解析器。"""

    def __init__(self, pydantic_object: Optional[Type[BaseModel]] = None):
        self.pydantic_object = pydantic_object

    def parse(self, text: str) -> Any:
        """解析JSON文本。"""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            return self._handle_json_error(text, e)

    def get_format_instructions(self) -> str:
        """JSON格式说明。"""
        if self.pydantic_object:
            schema = self.pydantic_object.schema()
            return f"请以JSON格式回答，遵循以下schema：\n```json\n{json.dumps(schema, indent=2, ensure_ascii=False)}\n```"
        return "请以有效的JSON格式回答"
```

#### 使用示例

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 定义数据模型
class Person(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")

# 创建解析器
parser = JsonOutputParser(pydantic_object=Person)

# 使用
text = '{"name": "张三", "age": 30, "occupation": "程序员"}'
result = parser.parse(text)
print(result)  # {'name': '张三', 'age': 30, 'occupation': '程序员'}

# 获取格式说明
instructions = parser.get_format_instructions()
print(instructions)
```

---

### 2.3 PydanticOutputParser - Pydantic解析器

#### 基本信息
- **功能**：解析为Pydantic模型对象，提供类型验证
- **优势**：强类型检查、自动验证、IDE支持

#### 方法签名

```python
class PydanticOutputParser(BaseOutputParser[T]):
    """Pydantic输出解析器。"""

    def __init__(self, pydantic_object: Type[T]):
        self.pydantic_object = pydantic_object

    def parse(self, text: str) -> T:
        """解析为Pydantic对象。"""
        try:
            json_data = json.loads(text.strip())
            return self.pydantic_object.parse_obj(json_data)
        except Exception as e:
            raise OutputParserException(f"Failed to parse: {e}")

    def get_format_instructions(self) -> str:
        """获取格式说明。"""
        schema = self.pydantic_object.schema()
        return self._format_schema_instructions(schema)
```

#### 使用示例

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

class BookReview(BaseModel):
    """书评模型。"""
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    rating: int = Field(description="评分(1-5)", ge=1, le=5)
    summary: str = Field(description="评论摘要")
    tags: List[str] = Field(description="标签列表")

    @validator('rating')
    def validate_rating(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('评分必须在1-5之间')
        return v

# 创建解析器
parser = PydanticOutputParser(pydantic_object=BookReview)

# 使用
text = '''
{
    "title": "Python编程",
    "author": "张三",
    "rating": 5,
    "summary": "很棒的Python入门书籍",
    "tags": ["编程", "Python", "入门"]
}
'''

result = parser.parse(text)
print(type(result))     # <class '__main__.BookReview'>
print(result.title)     # "Python编程"
print(result.tags)      # ["编程", "Python", "入门"]

# 验证会自动进行
invalid_text = '{"title": "Test", "rating": 10}'  # 评分超出范围
try:
    parser.parse(invalid_text)
except OutputParserException as e:
    print(f"解析失败: {e}")
```

---

### 2.4 ListOutputParser - 列表解析器

#### 基本信息
- **功能**：解析文本为字符串列表
- **支持格式**：多种列表格式（逗号分隔、换行分隔、数字编号等）

#### 方法签名

```python
class ListOutputParser(BaseOutputParser[List[str]]):
    """列表输出解析器。"""

    def __init__(self, delimiter: str = "\n"):
        self.delimiter = delimiter

    def parse(self, text: str) -> List[str]:
        """解析文本为列表。"""
        if not text.strip():
            return []

        # 处理多种列表格式
        lines = text.strip().split(self.delimiter)
        result = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 移除列表标记
            line = re.sub(r'^[\d\w]*[.)]\s*', '', line)  # 移除 "1. " 或 "a) "
            line = re.sub(r'^[-*+]\s*', '', line)        # 移除 "- " 或 "* "

            if line:
                result.append(line)

        return result
```

#### 使用示例

```python
from langchain_core.output_parsers import ListOutputParser

# 默认换行分隔
parser = ListOutputParser()

# 解析不同格式的列表
text1 = """
1. 学习Python基础
2. 练习算法题
3. 做项目实践
"""

text2 = """
- 买菜
- 做饭
- 洗碗
"""

text3 = "苹果, 香蕉, 橙子"

result1 = parser.parse(text1)
print(result1)  # ['学习Python基础', '练习算法题', '做项目实践']

result2 = parser.parse(text2)
print(result2)  # ['买菜', '做饭', '洗碗']

# 逗号分隔
comma_parser = ListOutputParser(delimiter=",")
result3 = comma_parser.parse(text3)
print(result3)  # ['苹果', '香蕉', '橙子']
```

---

## 3. 高级解析器 API

### 3.1 XMLOutputParser - XML解析器

#### 基本信息
- **功能**：解析XML格式的结构化输出
- **优势**：支持嵌套结构、属性解析

#### 方法签名

```python
class XMLOutputParser(BaseOutputParser[Dict[str, Any]]):
    """XML输出解析器。"""

    def __init__(self, tags: Optional[List[str]] = None):
        self.tags = tags or []

    def parse(self, text: str) -> Dict[str, Any]:
        """解析XML文本。"""
        try:
            root = ET.fromstring(f"<root>{text}</root>")
            return self._parse_element(root)
        except ET.ParseError as e:
            raise OutputParserException(f"XML解析失败: {e}")
```

#### 使用示例

```python
from langchain_core.output_parsers import XMLOutputParser

parser = XMLOutputParser()

xml_text = """
<person>
    <name>张三</name>
    <age>30</age>
    <address>
        <city>北京</city>
        <district>海淀区</district>
    </address>
</person>
"""

result = parser.parse(xml_text)
print(result)
# {
#   'person': {
#     'name': '张三',
#     'age': '30',
#     'address': {
#       'city': '北京',
#       'district': '海淀区'
#     }
#   }
# }
```

---

### 3.2 EnumOutputParser - 枚举解析器

#### 基本信息
- **功能**：解析为预定义的枚举值
- **优势**：限制输出范围，提高准确性

#### 方法签名

```python
class EnumOutputParser(BaseOutputParser[str]):
    """枚举输出解析器。"""

    def __init__(self, enum: Enum):
        self.enum = enum
        self.enum_values = [e.value for e in enum]

    def parse(self, text: str) -> str:
        """解析为枚举值。"""
        text = text.strip().lower()

        # 精确匹配
        for value in self.enum_values:
            if text == value.lower():
                return value

        # 模糊匹配
        for value in self.enum_values:
            if value.lower() in text or text in value.lower():
                return value

        raise OutputParserException(f"无法匹配枚举值: {text}, 可选值: {self.enum_values}")
```

#### 使用示例

```python
from enum import Enum
from langchain_core.output_parsers import EnumOutputParser

class Sentiment(Enum):
    POSITIVE = "积极"
    NEGATIVE = "消极"
    NEUTRAL = "中性"

parser = EnumOutputParser(enum=Sentiment)

# 测试解析
result1 = parser.parse("积极")           # "积极"
result2 = parser.parse("这是积极的情感")    # "积极" (模糊匹配)
result3 = parser.parse("neutral")      # 抛出异常，不在枚举中

print(parser.get_format_instructions())
# 输出: "请从以下选项中选择: 积极, 消极, 中性"
```

---

## 4. 流式解析器 API

### 4.1 BaseTransformOutputParser

#### 基本信息
- **功能**：支持流式输入的转换解析器
- **适用场景**：实时处理流式LLM输出

#### 方法签名

```python
class BaseTransformOutputParser(BaseOutputParser[T]):
    """流式转换解析器基类。"""

    def transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[T]:
        """转换流式输入。"""
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield from self._transform_chunk(self._extract_content(chunk))
            else:
                yield from self._transform_chunk(chunk)

    def _transform_chunk(self, chunk: str) -> Iterator[T]:
        """转换单个chunk。"""
        raise NotImplementedError
```

---

### 4.2 JsonStreamOutputParser

#### 基本信息
- **功能**：流式解析JSON输出
- **特性**：增量解析、部分对象支持

#### 使用示例

```python
from langchain_core.output_parsers import JsonStreamOutputParser

parser = JsonStreamOutputParser()

# 模拟流式输入
json_chunks = [
    '{"name":',
    ' "张三",',
    ' "age": 30,',
    ' "skills": ["Python"',
    ', "JavaScript"]}'
]

# 流式解析
for chunk in json_chunks:
    try:
        partial_result = parser.parse_partial(chunk)
        if partial_result:
            print(f"部分结果: {partial_result}")
    except:
        continue

# 最终解析
final_result = parser.parse(''.join(json_chunks))
print(f"完整结果: {final_result}")
```

---

## 5. 错误处理与恢复 API

### 5.1 OutputParserException

#### 基本信息
- **功能**：输出解析异常的标准类型
- **包含信息**：错误消息、原始文本、恢复建议

#### 数据结构

```python
class OutputParserException(ValueError):
    """输出解析异常。"""

    def __init__(
        self,
        error: str,
        observation: str = "",
        llm_output: str = "",
        send_to_llm: bool = False
    ):
        super().__init__(error)
        self.error = error
        self.observation = observation
        self.llm_output = llm_output
        self.send_to_llm = send_to_llm
```

#### 使用示例

```python
def safe_parse_with_retry(parser, text, max_retries=3):
    """安全解析，支持重试。"""
    for attempt in range(max_retries):
        try:
            return parser.parse(text)
        except OutputParserException as e:
            if attempt == max_retries - 1:
                raise

            # 记录错误并尝试修复
            print(f"解析失败 (第{attempt+1}次): {e.error}")

            # 简单的文本清理
            text = text.strip()
            if not text.startswith('{') and '{' in text:
                text = text[text.find('{'):]
            if not text.endswith('}') and '}' in text:
                text = text[:text.rfind('}')+1]

    return None
```

---

### 5.2 解析修复策略

#### 自动修复机制

```python
class RobustJsonParser(JsonOutputParser):
    """健壮的JSON解析器，支持自动修复。"""

    def parse(self, text: str) -> Any:
        """解析JSON，支持自动修复。"""
        original_text = text

        # 尝试直接解析
        try:
            return super().parse(text)
        except json.JSONDecodeError:
            pass

        # 修复常见问题
        fixes = [
            self._fix_trailing_comma,
            self._fix_unquoted_keys,
            self._fix_incomplete_json,
            self._extract_json_block
        ]

        for fix_func in fixes:
            try:
                fixed_text = fix_func(text)
                return json.loads(fixed_text)
            except:
                continue

        # 所有修复都失败
        raise OutputParserException(
            f"无法解析JSON: {original_text}",
            observation="请确保输出是有效的JSON格式"
        )

    def _fix_trailing_comma(self, text: str) -> str:
        """修复尾随逗号。"""
        return re.sub(r',(\s*[}\]])', r'\1', text)

    def _fix_unquoted_keys(self, text: str) -> str:
        """修复未加引号的键。"""
        return re.sub(r'(\w+):', r'"\1":', text)

    def _extract_json_block(self, text: str) -> str:
        """提取JSON代码块。"""
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1)

        # 寻找第一个完整的JSON对象
        brace_count = 0
        start_idx = text.find('{')
        if start_idx == -1:
            return text

        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]

        return text
```

---

## 6. 链式组合 API

### 6.1 多解析器组合

#### 使用示例

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import Union, Dict, List

class FlexibleOutputParser(BaseOutputParser[Union[Dict, List, str]]):
    """灵活输出解析器，自动识别格式。"""

    def __init__(self):
        self.json_parser = JsonOutputParser()
        self.list_parser = ListOutputParser()
        self.str_parser = StrOutputParser()

    def parse(self, text: str) -> Union[Dict, List, str]:
        """自动识别格式并解析。"""
        text = text.strip()

        # 尝试JSON解析
        if text.startswith(('{', '[')):
            try:
                return self.json_parser.parse(text)
            except:
                pass

        # 尝试列表解析
        if any(text.startswith(marker) for marker in ['1.', '-', '*', '•']):
            try:
                result = self.list_parser.parse(text)
                if len(result) > 1:  # 确实是列表
                    return result
            except:
                pass

        # 回退到字符串
        return self.str_parser.parse(text)

    def get_format_instructions(self) -> str:
        return """请根据内容选择合适的格式：
- JSON对象或数组：使用标准JSON格式
- 列表：使用编号或破折号格式
- 普通文本：直接输出文本内容"""

# 使用示例
parser = FlexibleOutputParser()

# 自动识别不同格式
result1 = parser.parse('{"name": "张三", "age": 30}')        # Dict
result2 = parser.parse('1. 苹果\n2. 香蕉\n3. 橙子')           # List
result3 = parser.parse('这是一段普通文本')                     # str

print(f"类型1: {type(result1)}, 值: {result1}")
print(f"类型2: {type(result2)}, 值: {result2}")
print(f"类型3: {type(result3)}, 值: {result3}")
```

---

### 6.2 解析器链组合

```python
class ParsedOutputChain:
    """解析器链组合。"""

    def __init__(self, parsers: List[BaseOutputParser]):
        self.parsers = parsers

    def parse(self, text: str) -> List[Any]:
        """使用多个解析器解析同一文本。"""
        results = []
        for parser in self.parsers:
            try:
                result = parser.parse(text)
                results.append({
                    'parser_type': parser._type,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'parser_type': parser._type,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        return results

# 使用
parsers = [JsonOutputParser(), ListOutputParser(), StrOutputParser()]
chain = ParsedOutputChain(parsers)

text = '["苹果", "香蕉", "橙子"]'
results = chain.parse(text)

for result in results:
    print(f"{result['parser_type']}: {result['success']} - {result['result']}")
```

---

## 7. 最佳实践与配置

### 7.1 解析器选择指南

| 输出类型 | 推荐解析器 | 场景 |
|---------|-----------|------|
| 简单文本 | `StrOutputParser` | 直接文本输出 |
| JSON数据 | `JsonOutputParser` | 结构化数据 |
| 强类型对象 | `PydanticOutputParser` | 需要验证的结构化数据 |
| 字符串列表 | `ListOutputParser` | 清单、选项等 |
| 枚举值 | `EnumOutputParser` | 分类、选择题 |
| XML数据 | `XMLOutputParser` | 层次化数据 |
| 流式数据 | `BaseTransformOutputParser` | 实时处理 |

### 7.2 错误处理配置

```python
class ParserConfig:
    """解析器配置。"""

    def __init__(
        self,
        enable_auto_fix: bool = True,
        max_retries: int = 3,
        fallback_to_string: bool = True,
        log_errors: bool = True
    ):
        self.enable_auto_fix = enable_auto_fix
        self.max_retries = max_retries
        self.fallback_to_string = fallback_to_string
        self.log_errors = log_errors

def create_robust_parser(parser_class, config: ParserConfig = None):
    """创建健壮的解析器。"""
    config = config or ParserConfig()

    class RobustParser(parser_class):
        def parse(self, text: str):
            for attempt in range(config.max_retries):
                try:
                    return super().parse(text)
                except Exception as e:
                    if config.log_errors:
                        print(f"解析失败 (第{attempt+1}次): {e}")

                    if attempt == config.max_retries - 1:
                        if config.fallback_to_string:
                            return text  # 回退到原始字符串
                        raise e

    return RobustParser()
```

---

## 8. 总结

本文档详细描述了 **Output Parsers 模块**的核心 API：

### 主要解析器类型
1. **基础解析器**：StrOutputParser、JsonOutputParser、PydanticOutputParser
2. **高级解析器**：XMLOutputParser、EnumOutputParser、ListOutputParser
3. **流式解析器**：BaseTransformOutputParser及其实现
4. **组合解析器**：多解析器链式组合

### 核心功能
1. **格式转换**：文本到结构化数据的转换
2. **类型验证**：基于Pydantic的强类型检查
3. **错误处理**：自动修复和降级策略
4. **流式处理**：实时解析流式输出
5. **格式说明**：为LLM提供输出格式指导

每个 API 均包含：
- 完整的方法签名和参数说明
- 详细的使用示例和最佳实践
- 错误处理和恢复机制
- 性能考虑和优化建议

Output Parsers 是连接LLM原始输出和应用程序结构化数据的重要桥梁，正确选择和使用解析器对提高应用程序的可靠性和用户体验至关重要。
