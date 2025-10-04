# LangChain-04-Prompts-时序图

## 文档说明

本文档通过详细的时序图展示 **Prompts 模块**在各种场景下的执行流程，包括模板创建、变量绑定、格式化、消息构建、少样本学习等。

---

## 1. 基础模板创建

### 1.1 PromptTemplate.from_template 创建流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant PT as PromptTemplate
    participant Extractor as VariableExtractor
    participant Validator as TemplateValidator

    User->>PT: from_template("Hello {name}, you are {age}")
    PT->>Extractor: extract_variables(template)

    alt f-string格式
        Extractor->>Extractor: 正则匹配 \{([^}]+)\}
        Extractor-->>PT: ["name", "age"]
    else jinja2格式
        Extractor->>Extractor: jinja2.meta.find_undeclared_variables
        Extractor-->>PT: ["name", "age"]
    end

    PT->>Validator: validate_template(template, variables)
    Validator->>Validator: 检查语法错误
    Validator-->>PT: 验证通过

    PT->>PT: 创建实例
    PT-->>User: PromptTemplate(template="...", input_variables=["name", "age"])
```

**关键步骤说明**：

1. **变量提取**（步骤 2-6）：
   - f-string：使用正则表达式 `\{([^}]+)\}` 匹配
   - Jinja2：使用 AST 分析提取未声明变量
   - Mustache：解析 `{{variable}}` 语法

2. **模板验证**（步骤 7-9）：
   - 语法检查：确保模板格式正确
   - 变量一致性：确保提取的变量存在于模板中
   - 格式安全性：防止代码注入（特别是 f-string）

**性能特征**：
- 变量提取：O(n)，n 为模板长度
- 模板验证：O(1) 到 O(n)
- 创建开销：约 1-5ms

---

### 1.2 ChatPromptTemplate.from_messages 创建流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant CPT as ChatPromptTemplate
    participant Converter
    participant MSG as MessageTemplate
    participant Extractor

    User->>CPT: from_messages([("system", "You are {role}"), ("human", "{input}")])

    loop 处理每个消息
        CPT->>Converter: convert_message(("system", "You are {role}"))

        Converter->>MSG: SystemMessagePromptTemplate.from_template("You are {role}")
        MSG->>Extractor: extract_variables("You are {role}")
        Extractor-->>MSG: ["role"]
        MSG-->>Converter: SystemMessagePromptTemplate(variables=["role"])

        Converter-->>CPT: message_template + variables
    end

    CPT->>CPT: 合并所有变量: ["role", "input"]
    CPT->>CPT: 创建实例
    CPT-->>User: ChatPromptTemplate(messages=[...], input_variables=["role", "input"])
```

**消息转换规则**：

| 输入格式 | 转换结果 |
|---------|---------|
| `("system", "text")` | `SystemMessagePromptTemplate` |
| `("human", "text")` | `HumanMessagePromptTemplate` |
| `("ai", "text")` | `AIMessagePromptTemplate` |
| `SystemMessage(...)` | 包装为对应的模板 |
| `MessagesPlaceholder(...)` | 直接使用 |

---

## 2. 模板格式化场景

### 2.1 PromptTemplate.invoke 格式化

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant PT as PromptTemplate
    participant Validator
    participant Formatter
    participant PV as StringPromptValue

    User->>PT: invoke({"name": "Alice", "age": 30})

    PT->>PT: 合并部分变量: {**partial_variables, **input}
    PT->>Validator: validate_variables(merged_input, input_variables)

    alt 缺少必需变量
        Validator-->>PT: raise KeyError("Missing variables: ...")
    else 变量完整
        Validator-->>PT: 验证通过
    end

    PT->>Formatter: format(template, merged_input)

    alt f-string格式
        Formatter->>Formatter: template.format(**merged_input)
    else jinja2格式
        Formatter->>Formatter: jinja2_template.render(**merged_input)
    else mustache格式
        Formatter->>Formatter: pystache.render(template, merged_input)
    end

    Formatter-->>PT: "Hello Alice, you are 30"
    PT->>PV: StringPromptValue("Hello Alice, you are 30")
    PT-->>User: StringPromptValue
```

**错误处理场景**：

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant PT as PromptTemplate
    participant Validator

    User->>PT: invoke({"name": "Alice"})  # 缺少 age
    PT->>Validator: validate_variables({"name": "Alice"}, ["name", "age"])
    Validator->>Validator: 检查: {"name", "age"} - {"name"} = {"age"}
    Validator-->>PT: KeyError("Missing variables: {'age'}")
    PT-->>User: raise KeyError
```

---

### 2.2 ChatPromptTemplate.invoke 消息格式化

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant CPT as ChatPromptTemplate
    participant Loop as MessageLoop
    participant MT as MessageTemplate
    participant MP as MessagesPlaceholder
    participant CV as ChatPromptValue

    User->>CPT: invoke({"role": "assistant", "input": "Hi", "history": [...]})

    CPT->>Loop: 遍历消息模板

    loop 处理每个消息模板
        alt 普通消息模板
            Loop->>MT: format_messages(role="assistant")
            MT->>MT: format("You are {role}") -> "You are assistant"
            MT-->>Loop: [SystemMessage("You are assistant")]
        else 消息占位符
            Loop->>MP: format_messages(history=[...])
            MP->>MP: 获取变量 "history"
            MP-->>Loop: [HumanMessage("..."), AIMessage("...")]
        end
    end

    Loop-->>CPT: all_messages = [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    CPT->>CV: ChatPromptValue(all_messages)
    CPT-->>User: ChatPromptValue
```

**MessagesPlaceholder 处理逻辑**：

```mermaid
sequenceDiagram
    autonumber
    participant MP as MessagesPlaceholder
    participant Input

    alt optional=False 且变量不存在
        MP->>Input: get("chat_history")
        Input-->>MP: None
        MP-->>MP: raise KeyError("Missing required variable")
    else optional=True 且变量不存在
        MP->>Input: get("chat_history")
        Input-->>MP: None
        MP-->>MP: return []
    else 变量存在
        MP->>Input: get("chat_history")
        alt 单个消息
            Input-->>MP: BaseMessage
            MP-->>MP: return [BaseMessage]
        else 消息列表
            Input-->>MP: [Message1, Message2, ...]
            MP-->>MP: return [Message1, Message2, ...]
        end
    end
```

---

## 3. 部分变量绑定场景

### 3.1 partial 方法执行流程

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant PT1 as Original Template
    participant PT2 as New Template

    User->>PT1: partial(role="assistant", language="English")

    PT1->>PT1: 合并部分变量<br/>{...existing_partial, role="assistant", language="English"}

    PT1->>PT1: 更新输入变量列表<br/>input_variables - partial_variables.keys()
    Note over PT1: 原来: ["role", "language", "task"]<br/>现在: ["task"]

    PT1->>PT2: 创建新实例<br/>相同template, 新的变量配置
    PT2-->>PT1: new_template
    PT1-->>User: new_template(input_variables=["task"])

    User->>PT2: invoke({"task": "translate"})
    PT2->>PT2: 使用合并变量: {role="assistant", language="English", task="translate"}
    PT2-->>User: StringPromptValue(formatted_text)
```

**变量管理逻辑**：

```python
# 原始模板
original_vars = {"role", "language", "task", "input"}
partial_vars = {"role": "assistant"}
input_vars = original_vars - set(partial_vars.keys())
# input_vars = {"language", "task", "input"}

# 再次部分绑定
new_partial_vars = {"role": "assistant", "language": "English"}
new_input_vars = original_vars - set(new_partial_vars.keys())
# new_input_vars = {"task", "input"}
```

---

## 4. 少样本学习场景

### 4.1 FewShotPromptTemplate 格式化

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant FST as FewShotPromptTemplate
    participant Selector as ExampleSelector
    participant ExampleTemplate
    participant Formatter

    User->>FST: invoke({"word": "big"})

    alt 使用固定示例
        FST->>FST: 使用 self.examples
    else 使用示例选择器
        FST->>Selector: select_examples({"word": "big"})
        Selector->>Selector: 计算相似度/长度/其他策略
        Selector-->>FST: selected_examples
    end

    FST->>FST: 构建完整提示

    loop 格式化每个示例
        FST->>ExampleTemplate: format(example)
        ExampleTemplate-->>FST: "Input: happy\nOutput: sad"
    end

    FST->>Formatter: 组装最终提示
    Note over Formatter: prefix +<br/>example1 + separator +<br/>example2 + separator +<br/>suffix

    Formatter-->>FST: formatted_prompt
    FST-->>User: StringPromptValue(formatted_prompt)
```

**完整示例格式化结果**：

```text
Find the opposite of the given word:

Input: happy
Output: sad

Input: tall
Output: short

Input: hot
Output: cold

Input: big
Output:
```

---

### 4.2 语义相似度示例选择

```mermaid
sequenceDiagram
    autonumber
    participant FST as FewShotPromptTemplate
    participant Selector as SemanticSimilarityExampleSelector
    participant VS as VectorStore
    participant Embeddings

    FST->>Selector: select_examples({"input": "excited"})

    Selector->>Selector: 构建查询字符串<br/>"input: excited"

    Selector->>Embeddings: embed_query("input: excited")
    Embeddings-->>Selector: query_vector

    Selector->>VS: similarity_search(query_vector, k=2)
    VS->>VS: 计算余弦相似度
    VS-->>Selector: [doc1, doc2]  # 按相似度排序

    Selector->>Selector: 提取元数据
    Selector-->>FST: [{"input": "happy", "output": "joyful"}, {"input": "glad", "output": "pleased"}]
```

**相似度计算过程**：

1. **查询向量化**：`"input: excited"` → `[0.1, 0.3, -0.2, ...]`
2. **候选匹配**：与所有示例向量计算相似度
3. **排序选择**：返回最相似的 k 个示例

**性能特征**：
- 向量化：10-50ms
- 相似度搜索：1-10ms（取决于示例数量）
- 总延迟：20-100ms

---

## 5. 模板组合场景

### 5.1 PipelinePromptTemplate 管道执行

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Pipeline as PipelinePromptTemplate
    participant Stage1 as IntroTemplate
    participant Stage2 as MainTemplate
    participant Final as FinalTemplate

    User->>Pipeline: invoke({"topic": "AI", "style": "casual"})

    Pipeline->>Stage1: format(topic="AI")
    Stage1-->>Pipeline: intro_text = "Let's talk about AI..."

    Pipeline->>Stage2: format(style="casual", intro=intro_text)
    Stage2-->>Pipeline: main_content = "So, AI is pretty cool..."

    Pipeline->>Final: format(intro=intro_text, main=main_content)
    Final-->>Pipeline: final_prompt

    Pipeline-->>User: StringPromptValue(final_prompt)
```

**管道配置示例**：

```python
pipeline = PipelinePromptTemplate(
    final_prompt=PromptTemplate.from_template("{intro}\n\n{main}\n\nConclusion: {conclusion}"),
    pipeline_prompts=[
        ("intro", PromptTemplate.from_template("Let's discuss {topic}")),
        ("main", PromptTemplate.from_template("In a {style} tone: {detailed_content}")),
        ("conclusion", PromptTemplate.from_template("To summarize {topic}"))
    ]
)
```

---

## 6. 高级格式化场景

### 6.1 Jinja2 复杂模板格式化

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant PT as PromptTemplate
    participant Jinja2 as Jinja2Engine
    participant AST as Template AST

    User->>PT: invoke({"users": [{"name": "Alice"}, {"name": "Bob"}], "task": "greet"})

    PT->>Jinja2: render(template, variables)
    Jinja2->>AST: parse("Hello {% for user in users %}{{ user.name }}{% endfor %}")
    AST-->>Jinja2: parsed_template

    Jinja2->>Jinja2: 执行模板逻辑

    loop 处理循环
        Jinja2->>Jinja2: 遍历 users
        Note over Jinja2: user = {"name": "Alice"}
        Jinja2->>Jinja2: 渲染 {{ user.name }} -> "Alice"

        Note over Jinja2: user = {"name": "Bob"}
        Jinja2->>Jinja2: 渲染 {{ user.name }} -> "Bob"
    end

    Jinja2-->>PT: "Hello AliceBob"
    PT-->>User: StringPromptValue("Hello AliceBob")
```

**Jinja2 特性支持**：

| 特性 | 语法示例 | 用途 |
|-----|---------|------|
| 变量 | `{{ name }}` | 输出变量值 |
| 条件 | `{% if condition %}...{% endif %}` | 条件渲染 |
| 循环 | `{% for item in list %}...{% endfor %}` | 遍历列表 |
| 过滤器 | `{{ name\|upper }}` | 文本转换 |
| 宏 | `{% macro func() %}...{% endmacro %}` | 可重用片段 |

---

### 6.2 条件模板选择

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Selector as TemplateSelector
    participant Casual as CasualTemplate
    participant Formal as FormalTemplate
    participant Business as BusinessTemplate

    User->>Selector: select_template({"tone": "business", "urgency": "high"})

    Selector->>Selector: 分析输入参数
    Note over Selector: tone="business" + urgency="high"

    alt tone == "casual"
        Selector->>Casual: 选择休闲模板
        Casual-->>Selector: "Hey! {message}"
    else tone == "formal"
        Selector->>Formal: 选择正式模板
        Formal-->>Selector: "Dear Sir/Madam, {message}"
    else tone == "business" and urgency == "high"
        Selector->>Business: 选择商务紧急模板
        Business-->>Selector: "URGENT: {message}. Please respond ASAP."
    end

    Selector-->>User: selected_template
```

**动态模板选择逻辑**：

```python
def select_template(context: Dict[str, Any]) -> PromptTemplate:
    """根据上下文选择合适的模板。"""
    tone = context.get("tone", "neutral")
    urgency = context.get("urgency", "normal")
    audience = context.get("audience", "general")

    if urgency == "high":
        return urgent_templates[tone]
    elif audience == "technical":
        return technical_templates[tone]
    else:
        return standard_templates[tone]
```

---

## 7. 错误处理场景

### 7.1 变量缺失错误处理

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant PT as PromptTemplate
    participant Validator
    participant ErrorHandler

    User->>PT: invoke({"name": "Alice"})  # 缺少 age 变量
    PT->>Validator: validate_variables({"name": "Alice"}, ["name", "age"])

    Validator->>Validator: 计算缺失变量
    Note over Validator: required = {"name", "age"}<br/>provided = {"name"}<br/>missing = {"age"}

    Validator->>ErrorHandler: 构建错误信息
    ErrorHandler->>ErrorHandler: format_missing_variables_error({"age"})
    ErrorHandler-->>Validator: "Missing required variables: {'age'}"

    Validator-->>PT: raise KeyError("Missing required variables: {'age'}")
    PT-->>User: KeyError
```

### 7.2 模板格式错误处理

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant PT as PromptTemplate
    participant Formatter
    participant ErrorHandler

    User->>PT: from_template("Hello {name")  # 缺少右括号
    PT->>Formatter: validate_template("Hello {name")

    Formatter->>Formatter: 尝试解析模板
    Note over Formatter: str.format() 测试

    Formatter->>ErrorHandler: 捕获 ValueError
    ErrorHandler->>ErrorHandler: 分析错误类型
    ErrorHandler-->>Formatter: "Invalid template syntax: unmatched '{'"

    Formatter-->>PT: raise ValueError("Invalid template syntax")
    PT-->>User: ValueError
```

---

## 8. 性能优化场景

### 8.1 模板缓存使用

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Factory as TemplateFactory
    participant Cache as TemplateCache
    participant PT as PromptTemplate

    User->>Factory: get_template("greeting", "Hello {name}")
    Factory->>Cache: get(template_key="greeting")

    alt 缓存命中
        Cache-->>Factory: cached_template
        Factory-->>User: cached_template (快速返回)
    else 缓存未命中
        Cache-->>Factory: None
        Factory->>PT: from_template("Hello {name}")
        PT-->>Factory: new_template
        Factory->>Cache: put("greeting", new_template)
        Factory-->>User: new_template
    end

    User->>User: 后续使用缓存模板 (避免重复创建)
```

**缓存策略**：

- **LRU 淘汰**：最久未使用的模板被移除
- **大小限制**：默认缓存 128 个模板
- **键生成**：基于模板内容和格式的哈希值

### 8.2 批量格式化优化

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant BatchFormatter
    participant PT as PromptTemplate
    participant Pool as ThreadPool

    User->>BatchFormatter: batch_format(template, [input1, input2, input3, ...])

    BatchFormatter->>Pool: 提交批量任务

    par 并行格式化
        Pool->>PT: format(input1)
        PT-->>Pool: result1
    and
        Pool->>PT: format(input2)
        PT-->>Pool: result2
    and
        Pool->>PT: format(input3)
        PT-->>Pool: result3
    end

    Pool-->>BatchFormatter: [result1, result2, result3, ...]
    BatchFormatter-->>User: batch_results
```

**性能对比**：

| 方法 | 100个输入耗时 | 内存使用 |
|-----|------------|---------|
| 顺序格式化 | 1000ms | 低 |
| 并行格式化 | 200ms | 中等 |
| 批量优化 | 150ms | 高 |

---

## 9. 总结

本文档详细展示了 **Prompts 模块**的关键执行时序：

1. **模板创建**：from_template、from_messages 的变量提取和验证
2. **格式化流程**：invoke 方法的完整执行链路
3. **部分绑定**：partial 方法的变量管理
4. **少样本学习**：示例选择和格式化的完整流程
5. **模板组合**：PipelinePromptTemplate 的管道执行
6. **高级特性**：Jinja2 复杂模板和条件选择
7. **错误处理**：变量缺失和格式错误的处理机制
8. **性能优化**：缓存策略和批量处理

每张时序图包含：
- 详细的参与者和交互步骤
- 关键决策点和分支逻辑
- 错误处理和边界条件
- 性能特征和优化建议
- 实际使用场景和最佳实践

这些时序图帮助开发者深入理解提示工程的内部机制，为构建复杂的提示系统提供指导。
