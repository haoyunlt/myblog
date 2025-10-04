# LangChain-06-OutputParsers-时序图

## 文档说明

本文档通过详细的时序图展示 **Output Parsers 模块**在各种场景下的执行流程，包括不同解析器的工作机制、错误处理、流式解析、缓存优化等复杂交互过程。

---

## 1. 基础解析场景

### 1.1 StrOutputParser 简单解析流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Chain as LLMChain
    participant Model as ChatOpenAI
    participant Parser as StrOutputParser

    User->>Chain: invoke({"topic": "AI"})
    Chain->>Model: generate(prompt)
    Model-->>Chain: AIMessage(content="AI是人工智能的缩写...")

    Chain->>Parser: parse(message.content)
    Parser->>Parser: 直接返回字符串内容
    Parser-->>Chain: "AI是人工智能的缩写..."

    Chain-->>User: "AI是人工智能的缩写..."
```

**关键特点**：
- 最简单的解析器，无需任何转换
- 直接返回LLM输出的文本内容
- 解析时间 < 1ms，几乎无性能开销
- 适用于不需要结构化的文本输出场景

---

### 1.2 JsonOutputParser 标准解析流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Chain
    participant Model
    participant Parser as JsonOutputParser
    participant Validator as JSONValidator

    User->>Chain: invoke({"query": "创建用户信息"})

    Chain->>Model: generate(prompt + format_instructions)
    Note over Model: 提示包含JSON格式说明
    Model-->>Chain: AIMessage(content='{"name": "张三", "age": 30, "email": "zhang@example.com"}')

    Chain->>Parser: parse(message.content)

    Parser->>Parser: 提取JSON内容<br/>去除markdown标记

    Parser->>Validator: json.loads(cleaned_content)

    alt JSON有效
        Validator-->>Parser: {"name": "张三", "age": 30, "email": "zhang@example.com"}

        alt 有Pydantic模型
            Parser->>Parser: 验证数据结构
            Parser->>Parser: 转换为Python对象
        end

        Parser-->>Chain: 解析后的对象
    else JSON无效
        Validator-->>Parser: json.JSONDecodeError
        Parser->>Parser: 尝试自动修复

        alt 修复成功
            Parser->>Validator: json.loads(fixed_content)
            Validator-->>Parser: 修复后的对象
            Parser-->>Chain: 解析结果
        else 修复失败
            Parser-->>Chain: raise OutputParserException
        end
    end

    Chain-->>User: 最终解析结果
```

**解析步骤详解**：

1. **内容清理**（步骤 5）：
   - 移除markdown代码块标记（```json）
   - 去除前后空白字符
   - 处理转义字符

2. **JSON验证**（步骤 6-7）：
   - 使用标准`json.loads()`解析
   - 检查语法正确性
   - 验证数据类型

3. **错误修复**（步骤 12-17）：
   - 修复尾随逗号
   - 补全缺失的引号
   - 修复常见格式错误

---

## 2. Pydantic解析场景

### 2.1 PydanticOutputParser 强类型解析

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Parser as PydanticOutputParser
    participant Schema as PydanticModel
    participant Validator as PydanticValidator
    participant Cache as SchemaCache

    User->>Parser: parse('{"name": "张三", "age": 30, "email": "invalid-email"}')

    Parser->>Cache: 获取缓存的Schema
    alt Schema已缓存
        Cache-->>Parser: cached_schema
    else 首次使用
        Parser->>Schema: generate_schema()
        Schema-->>Parser: json_schema
        Parser->>Cache: 缓存Schema
    end

    Parser->>Parser: json.loads(text) → raw_data

    Parser->>Validator: Schema.parse_obj(raw_data)

    Validator->>Validator: 验证字段类型<br/>name: str ✓<br/>age: int ✓<br/>email: str (格式检查)

    alt 验证成功
        Validator->>Schema: 创建模型实例
        Schema-->>Validator: UserModel(name="张三", age=30, email="...")
        Validator-->>Parser: validated_object
        Parser-->>User: UserModel实例
    else 验证失败
        Validator-->>Parser: ValidationError([{<br/>  "loc": ["email"],<br/>  "msg": "field required",<br/>  "type": "value_error.email"<br/>}])

        Parser->>Parser: 格式化错误信息
        Parser-->>User: raise OutputParserException(<br/>  "字段验证失败: email格式不正确"<br/>)
    end
```

**验证过程详解**：

```python
# 示例Pydantic模型
class UserProfile(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    age: int = Field(ge=0, le=150)
    email: str = Field(regex=r'^[^@]+@[^@]+\.[^@]+$')

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('姓名不能为空')
        return v.strip()

# 验证流程
# 1. 类型检查：确保字段类型正确
# 2. 约束验证：检查Field中定义的约束
# 3. 自定义验证：执行@validator装饰的方法
# 4. 对象构造：创建最终的Pydantic实例
```

---

## 3. 列表解析场景

### 3.1 ListOutputParser 多格式识别

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Parser as ListOutputParser
    participant Recognizer as FormatRecognizer
    participant Cleaner as TextCleaner

    User->>Parser: parse("1. 苹果\n2. 香蕉\n- 橙子\n* 葡萄")

    Parser->>Recognizer: 识别列表格式
    Recognizer->>Recognizer: 检测多种格式<br/>数字编号: "1. 苹果"<br/>破折号: "- 橙子"<br/>星号: "* 葡萄"

    Recognizer-->>Parser: format_patterns = [numbered, dashed, starred]

    Parser->>Parser: 按分隔符分割文本<br/>lines = text.split('\n')

    loop 遍历每一行
        Parser->>Cleaner: clean_line(line)

        Cleaner->>Cleaner: 应用清理规则<br/>移除编号: "1. " → ""<br/>移除符号: "- " → ""<br/>移除符号: "* " → ""

        Cleaner-->>Parser: cleaned_item

        alt 清理后非空
            Parser->>Parser: 添加到结果列表
        else 清理后为空
            Parser->>Parser: 跳过空项
        end
    end

    Parser-->>User: ["苹果", "香蕉", "橙子", "葡萄"]
```

**格式识别规则**：

```python
# 支持的列表格式
PATTERNS = [
    r'^\s*\d+[.)]\s*(.+)$',      # "1. 项目" 或 "1) 项目"
    r'^\s*[a-zA-Z][.)]\s*(.+)$', # "a. 项目" 或 "a) 项目"
    r'^\s*[-*+•]\s*(.+)$',       # "- 项目", "* 项目", "+ 项目", "• 项目"
    r'^\s*>\s*(.+)$',            # "> 项目" (引用格式)
    r'^\s*\(\d+\)\s*(.+)$',      # "(1) 项目"
]

# 清理步骤
# 1. 去除前后空白
# 2. 移除列表标记符号
# 3. 再次去除空白
# 4. 检查是否为空项
```

---

## 4. 错误处理与恢复场景

### 4.1 JSON解析错误自动修复

```mermaid
sequenceDiagram
    autonumber
    participant Parser as JsonOutputParser
    participant Fixer as AutoFixer
    participant Validator as JSONValidator
    participant Logger

    Parser->>Parser: 接收原始文本<br/>'{"name": "张三", "age": 30,}'

    Parser->>Validator: 尝试直接解析
    Validator-->>Parser: JSONDecodeError("尾随逗号")

    Parser->>Logger: 记录解析失败
    Parser->>Fixer: 启动自动修复流程

    rect rgb(255, 248, 248)
        Note over Fixer: === 修复策略1: 移除尾随逗号 ===
        Fixer->>Fixer: regex替换: ',\s*}' → '}'
        Fixer->>Fixer: 修复后: '{"name": "张三", "age": 30}'

        Fixer->>Validator: 验证修复结果
        alt 修复成功
            Validator-->>Fixer: 解析成功
            Fixer-->>Parser: {"name": "张三", "age": 30}
        else 仍有错误
            Validator-->>Fixer: 仍然失败
        end
    end

    alt 第一次修复成功
        Parser-->>Parser: 返回修复后结果
    else 需要更多修复
        rect rgb(248, 255, 248)
            Note over Fixer: === 修复策略2: 补全引号 ===
            Fixer->>Fixer: 检测未加引号的键
            Fixer->>Fixer: regex替换: '(\w+):' → '"$1":'
            Fixer->>Validator: 再次验证
        end

        alt 第二次修复成功
            Parser-->>Parser: 返回修复后结果
        else 所有修复都失败
            Parser->>Parser: raise OutputParserException(<br/>  "无法修复JSON格式错误",<br/>  send_to_llm=True<br/>)
        end
    end
```

**自动修复策略**：

```python
class JsonAutoFixer:
    """JSON自动修复器。"""

    FIXES = [
        # 1. 移除尾随逗号
        (r',(\s*[}\]])', r'\1'),

        # 2. 为键添加引号
        (r'(\w+):', r'"\1":'),

        # 3. 修复单引号为双引号
        (r"'([^']*)'", r'"\1"'),

        # 4. 移除注释
        (r'//.*', ''),
        (r'/\*.*?\*/', ''),

        # 5. 修复布尔值
        (r'\bTrue\b', 'true'),
        (r'\bFalse\b', 'false'),
        (r'\bNone\b', 'null'),
    ]

    def fix(self, text: str) -> str:
        """应用所有修复策略。"""
        for pattern, replacement in self.FIXES:
            text = re.sub(pattern, replacement, text, flags=re.DOTALL)
        return text
```

---

### 4.2 解析异常处理流程

```mermaid
sequenceDiagram
    autonumber
    participant Chain as LLMChain
    participant Parser
    participant ErrorHandler
    participant LLM as LanguageModel
    participant User

    Chain->>Parser: parse(invalid_output)
    Parser-->>Chain: OutputParserException(<br/>  error="JSON格式错误",<br/>  observation="输出不是有效JSON",<br/>  send_to_llm=True<br/>)

    Chain->>ErrorHandler: handle_parsing_error(exception)

    alt send_to_llm = True (重试模式)
        ErrorHandler->>ErrorHandler: 构建错误提示<br/>"之前的输出格式不正确，请重新生成"

        ErrorHandler->>LLM: 重新生成<br/>prompt + error_message + format_instructions
        LLM-->>ErrorHandler: 新的输出

        ErrorHandler->>Parser: parse(new_output)

        alt 重新解析成功
            Parser-->>ErrorHandler: 解析结果
            ErrorHandler-->>Chain: 成功结果
        else 仍然失败
            Parser-->>ErrorHandler: 再次失败
            ErrorHandler->>ErrorHandler: 记录重试失败
            ErrorHandler-->>Chain: 最终错误
        end

    else send_to_llm = False (直接失败)
        ErrorHandler-->>Chain: 返回错误信息
    end

    alt 处理成功
        Chain-->>User: 解析结果
    else 处理失败
        Chain-->>User: 友好的错误消息<br/>"抱歉，无法解析输出格式"
    end
```

**错误处理配置**：

```python
# 不同的错误处理策略
class ErrorHandlingStrategy:
    STRICT = "strict"        # 严格模式，直接抛出异常
    RETRY = "retry"          # 重试模式，重新请求LLM
    FALLBACK = "fallback"    # 回退模式，返回原始文本
    IGNORE = "ignore"        # 忽略模式，返回None

# 配置示例
parser = JsonOutputParser(
    error_handling=ErrorHandlingStrategy.RETRY,
    max_retries=3,
    fallback_to_string=True
)
```

---

## 5. 流式解析场景

### 5.1 JsonStreamOutputParser 实时解析

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Stream as StreamingLLM
    participant Parser as JsonStreamParser
    participant State as ParseState
    participant Buffer as ChunkBuffer

    User->>Stream: astream("生成用户信息JSON")

    loop 流式输出
        Stream->>Parser: chunk = '{"name":'
        Parser->>Buffer: 添加到缓冲区
        Buffer->>State: 更新解析状态<br/>检测到对象开始 '{'

        State->>State: brace_depth = 1<br/>in_string = false
        State-->>Parser: 状态：对象开始
        Parser-->>User: yield {"status": "parsing", "partial": null}

        Stream->>Parser: chunk = ' "张三",'
        Parser->>Buffer: 继续累积
        State->>State: 检测到字符串值<br/>当前键: "name"
        Parser-->>User: yield {"status": "parsing", "partial": {"name": "张三"}}

        Stream->>Parser: chunk = ' "age": 30'
        Parser->>Buffer: 继续累积
        State->>State: 添加新键值对
        Parser-->>User: yield {"status": "parsing", "partial": {"name": "张三", "age": 30}}

        Stream->>Parser: chunk = '}'
        Parser->>Buffer: 完成对象
        State->>State: brace_depth = 0<br/>对象完成

        Parser->>Parser: 验证完整JSON
        alt JSON完整且有效
            Parser-->>User: yield {<br/>  "status": "complete",<br/>  "result": {"name": "张三", "age": 30}<br/>}
        else JSON不完整
            Parser-->>User: yield {"status": "error", "error": "不完整的JSON"}
        end
    end
```

**流式解析状态管理**：

```python
class StreamParseState:
    def __init__(self):
        self.buffer = ""
        self.brace_depth = 0      # 大括号深度
        self.bracket_depth = 0    # 中括号深度
        self.in_string = False    # 是否在字符串中
        self.escape_next = False  # 下一个字符是否转义
        self.current_objects = [] # 当前解析的对象栈

    def process_char(self, char: str) -> Optional[dict]:
        """处理单个字符，返回完成的对象（如果有）。"""
        self.buffer += char

        if self.escape_next:
            self.escape_next = False
            return None

        if char == '\\' and self.in_string:
            self.escape_next = True
            return None

        if char == '"':
            self.in_string = not self.in_string
            return None

        if not self.in_string:
            if char == '{':
                self.brace_depth += 1
            elif char == '}':
                self.brace_depth -= 1
                if self.brace_depth == 0:
                    # 尝试解析完整对象
                    try:
                        obj = json.loads(self.buffer)
                        self.buffer = ""  # 清空缓冲区
                        return obj
                    except json.JSONDecodeError:
                        pass  # 继续累积

        return None
```

---

## 6. 缓存优化场景

### 6.1 解析结果缓存机制

```mermaid
sequenceDiagram
    autonumber
    participant Parser
    participant Cache as ParseCache
    participant Hasher
    participant Storage as CacheStorage

    Parser->>Hasher: 生成缓存键<br/>hash(parser_type + input_text)
    Hasher-->>Parser: cache_key = "json_abc123"

    Parser->>Cache: get(cache_key)

    alt 缓存命中
        Cache->>Storage: retrieve(cache_key)
        Storage-->>Cache: CacheEntry(value, created_at, ttl)

        Cache->>Cache: 检查是否过期
        alt 未过期
            Cache-->>Parser: cached_result
            Parser->>Parser: 更新统计：缓存命中
            Parser-->>Parser: 直接返回缓存结果 (< 1ms)
        else 已过期
            Cache->>Storage: remove(cache_key)
            Cache-->>Parser: None (缓存未命中)
        end

    else 缓存未命中
        Cache-->>Parser: None

        Parser->>Parser: 执行实际解析逻辑 (10-50ms)
        Parser->>Parser: parse_result = actual_parse(text)

        Parser->>Cache: put(cache_key, parse_result, ttl=3600)
        Cache->>Storage: store(cache_key, CacheEntry(...))

        Parser->>Parser: 更新统计：缓存未命中
        Parser-->>Parser: 返回解析结果
    end
```

**缓存性能优化**：

```python
class ParseCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []  # LRU跟踪
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    def _generate_key(self, parser_type: str, text: str) -> str:
        """生成缓存键。"""
        # 对长文本进行哈希以节省内存
        if len(text) > 1000:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            return f"{parser_type}:{text_hash}"
        else:
            return f"{parser_type}:{text}"

    def _evict_lru(self) -> None:
        """淘汰最久未使用的缓存项。"""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)
            self._stats["evictions"] += 1
```

---

## 7. 性能监控场景

### 7.1 解析器性能追踪

```mermaid
sequenceDiagram
    autonumber
    participant Parser
    participant Monitor as PerformanceMonitor
    participant Metrics as MetricsCollector
    participant Alert as AlertSystem

    Parser->>Monitor: 开始解析<br/>start_parse(parser_type, input_length)
    Monitor->>Monitor: 记录开始时间<br/>start_time = time.now()

    Parser->>Parser: 执行解析逻辑
    Note over Parser: 实际解析过程<br/>可能包含多个步骤

    alt 解析成功
        Parser->>Monitor: 解析完成<br/>end_parse(success=True, result_type)
        Monitor->>Monitor: 计算执行时间<br/>duration = time.now() - start_time

        Monitor->>Metrics: 记录成功指标<br/>success_count++<br/>total_time += duration<br/>input_lengths.append(input_length)

        Monitor->>Monitor: 检查性能阈值
        alt 执行时间 > 慢查询阈值
            Monitor->>Alert: 触发慢解析告警<br/>"JsonParser解析耗时500ms"
        end

    else 解析失败
        Parser->>Monitor: 解析失败<br/>end_parse(success=False, error_type)
        Monitor->>Monitor: 计算执行时间

        Monitor->>Metrics: 记录失败指标<br/>error_count++<br/>error_types[error_type]++

        Monitor->>Monitor: 检查错误率
        alt 错误率 > 阈值
            Monitor->>Alert: 触发错误率告警<br/>"JsonParser错误率超过10%"
        end
    end

    Monitor->>Metrics: 更新聚合统计<br/>计算平均时间、成功率等

    alt 定期报告 (每分钟)
        Monitor->>Monitor: 生成性能报告
        Monitor->>Alert: 发送统计报告<br/>包含各解析器性能数据
    end
```

**性能指标收集**：

```python
class ParserPerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(ParserMetrics)
        self.alert_thresholds = {
            "slow_parse_ms": 1000,      # 慢解析阈值
            "error_rate": 0.1,          # 错误率阈值
            "memory_usage_mb": 100      # 内存使用阈值
        }

    def record_parse(self, parser_type: str, duration: float,
                    success: bool, input_length: int, **kwargs):
        """记录解析性能。"""
        metrics = self.metrics[parser_type]
        metrics.record_parse(duration, success, input_length, **kwargs)

        # 检查告警条件
        self._check_alerts(parser_type, metrics)

    def _check_alerts(self, parser_type: str, metrics: ParserMetrics):
        """检查告警条件。"""
        # 慢解析告警
        if metrics.last_parse_time > self.alert_thresholds["slow_parse_ms"]:
            self._send_alert(f"慢解析: {parser_type} 耗时 {metrics.last_parse_time}ms")

        # 错误率告警
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            self._send_alert(f"高错误率: {parser_type} 错误率 {metrics.error_rate:.2%}")
```

---

## 8. 解析器组合场景

### 8.1 多解析器链式处理

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Chain as ParserChain
    participant Parser1 as JsonParser
    participant Parser2 as ListParser
    participant Parser3 as StrParser
    participant Selector as FormatSelector

    User->>Chain: parse(ambiguous_text)
    Chain->>Selector: 检测输入格式<br/>analyze_format(text)

    Selector->>Selector: 格式启发式检测<br/>- 是否以 { 或 [ 开头？<br/>- 是否包含列表标记？<br/>- 是否为纯文本？

    alt 检测为JSON格式
        Selector-->>Chain: format = "json", confidence = 0.9
        Chain->>Parser1: parse(text)

        alt JSON解析成功
            Parser1-->>Chain: json_result
            Chain-->>User: json_result
        else JSON解析失败
            Parser1-->>Chain: OutputParserException
            Chain->>Parser3: 回退到字符串解析
            Parser3-->>Chain: str_result
            Chain-->>User: str_result
        end

    else 检测为列表格式
        Selector-->>Chain: format = "list", confidence = 0.8
        Chain->>Parser2: parse(text)

        alt 列表解析成功
            Parser2-->>Chain: list_result
            Chain-->>User: list_result
        else 列表解析失败
            Parser2-->>Chain: OutputParserException
            Chain->>Parser3: 回退到字符串解析
            Parser3-->>Chain: str_result
            Chain-->>User: str_result
        end

    else 格式不明确
        Selector-->>Chain: format = "unknown", confidence = 0.3

        rect rgb(255, 248, 248)
            Note over Chain: === 尝试所有解析器 ===
            par 并行尝试
                Chain->>Parser1: parse(text)
                Parser1-->>Chain: json_result 或 exception
            and
                Chain->>Parser2: parse(text)
                Parser2-->>Chain: list_result 或 exception
            and
                Chain->>Parser3: parse(text)
                Parser3-->>Chain: str_result (总是成功)
            end
        end

        Chain->>Chain: 选择最佳结果<br/>优先级: JSON > List > String
        Chain-->>User: best_result
    end
```

---

## 9. 总结

本文档详细展示了 **Output Parsers 模块**的关键执行时序：

1. **基础解析**：StrOutputParser、JsonOutputParser的标准解析流程
2. **强类型解析**：PydanticOutputParser的验证和类型转换机制
3. **列表解析**：ListOutputParser的多格式识别和清理过程
4. **错误处理**：自动修复、重试机制、降级策略
5. **流式解析**：实时处理流式输出的状态管理
6. **缓存优化**：解析结果缓存的命中和淘汰机制
7. **性能监控**：解析器执行的指标收集和告警
8. **解析器组合**：多解析器的智能选择和回退策略

每张时序图包含：
- 详细的执行步骤和参与者交互
- 关键决策点和分支处理逻辑
- 错误处理和恢复机制
- 性能优化点和监控策略

这些时序图帮助开发者深入理解输出解析系统的内部工作机制，为构建高效、可靠的LLM输出处理管道提供指导。Output Parsers是连接原始LLM输出和结构化应用数据的关键桥梁，正确理解其执行流程对提高应用程序的稳定性和用户体验至关重要。
