---
title: "vLLM-10-InputsOutputs模块-时序图"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 时序图
  - 流程分析
  - 源码分析
categories:
  - vLLM
description: "源码剖析 - vLLM-10-InputsOutputs模块-时序图"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# vLLM-10-InputsOutputs模块-时序图

## 时序图概览

本文档展示 InputsOutputs 模块在不同场景下的处理时序，涵盖：

| 场景 | 输入类型 | 参与方 | 关键特征 |
|------|----------|--------|----------|
| 文本输入处理 | TextPrompt | Preprocessor + Tokenizer | 分词和验证 |
| Token 输入处理 | TokensPrompt | Preprocessor + Validator | 格式验证 |
| 多模态输入处理 | TextPrompt + 多模态 | Preprocessor + MMProcessor | 多模态融合 |
| 嵌入输入处理 | EmbedsPrompt | Preprocessor + Validator | 维度验证 |
| 批量输入处理 | 批量 prompt | BatchProcessor | 并行处理 |
| 编码器-解码器处理 | Enc-Dec Prompt | Preprocessor | 双路处理 |

---

## 场景 1：文本输入预处理流程

### 业务场景
用户提交纯文本输入，需要分词并转换为模型可接受的格式。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Entrypoints as Entrypoints 层
    participant Preprocessor as InputPreprocessor
    participant Parser as parse_singleton_prompt
    participant Tokenizer as Tokenizer
    participant Cache as 前缀缓存
    participant Engine as Engine
    
    Note over User,Engine: 文本输入的完整预处理流程
    
    rect rgb(255, 245, 235)
        Note over User,Engine: 1. 输入接收阶段
        User->>Entrypoints: 提交文本请求
        Note over User: prompt = "Explain quantum computing"
        Entrypoints->>Preprocessor: preprocess(prompt)
        Preprocessor->>Parser: parse_singleton_prompt(prompt)
        Parser->>Parser: 识别类型为 "str"
        Parser-->>Preprocessor: ParsedStrPrompt{type:"str", content:"Explain..."}
    end
    
    rect rgb(235, 245, 255)
        Note over Preprocessor,Cache: 2. 分词处理阶段
        Preprocessor->>Preprocessor: _process_text(parsed_content)
        
        alt 启用前缀缓存
            Preprocessor->>Cache: 查询缓存 hash(prompt)
            alt 缓存命中
                Cache-->>Preprocessor: 返回缓存的 token_ids
                Note over Preprocessor: 跳过分词步骤
            else 缓存未命中
                Preprocessor->>Tokenizer: encode(text)
                Tokenizer->>Tokenizer: 执行分词算法
                Tokenizer-->>Preprocessor: token_ids [1, 12345, 67890, ...]
                Preprocessor->>Cache: 存储缓存 (hash, token_ids)
            end
        else 未启用缓存
            Preprocessor->>Tokenizer: encode(text)
            Tokenizer->>Tokenizer: 分词处理
            Tokenizer-->>Preprocessor: token_ids
        end
        
        Preprocessor->>Preprocessor: 构建 TokenInputs
    end
    
    rect rgb(245, 255, 235)
        Note over Preprocessor,Engine: 3. 输出构建阶段
        Preprocessor->>Preprocessor: _build_decoder_only_llm_inputs()
        Preprocessor->>Preprocessor: 生成 cache_salt（如果需要）
        
        Note over Preprocessor: 创建 DecoderOnlyInputs:<br/>prompt_token_ids: [1, 12345, ...]<br/>cache_salt: "text_abc123"
        
        Preprocessor-->>Entrypoints: DecoderOnlyInputs
        Entrypoints->>Engine: add_request(processed_inputs)
        Engine-->>Entrypoints: request_id
        Entrypoints-->>User: 请求已接收
    end
```

### 关键要点说明

1. **类型识别**：`parse_singleton_prompt` 根据输入类型（str、dict等）准确识别格式
2. **缓存优化**：前缀缓存可以显著减少重复文本的分词开销
3. **错误处理**：分词过程中的编码错误会被捕获并报告
4. **内存管理**：大型文本会被分块处理以避免内存溢出

### 性能特征

- **分词延迟**：10-50ms（取决于文本长度和分词器类型）
- **缓存命中率**：70-90%（在相似请求场景下）
- **内存使用**：O(sequence_length) for token storage

---

## 场景 2：Token 输入验证和处理

### 业务场景
用户直接提供预分词的 Token ID 序列，需要验证格式并传递给模型。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Entrypoints as Entrypoints 层
    participant Preprocessor as InputPreprocessor
    participant Parser as parse_singleton_prompt
    participant Validator as Token 验证器
    participant Engine as Engine
    
    Note over User,Engine: Token 输入的处理和验证流程
    
    rect rgb(255, 245, 235)
        Note over User,Engine: 1. Token 输入接收
        User->>Entrypoints: 提交 TokensPrompt
        Note over User: {<br/>  "prompt_token_ids": [1, 2, 3, 4, 5],<br/>  "prompt": "Hello world"<br/>}
        Entrypoints->>Preprocessor: preprocess(tokens_prompt)
        Preprocessor->>Parser: parse_singleton_prompt(tokens_prompt)
        Parser->>Parser: 检查字典包含 "prompt_token_ids"
        Parser->>Parser: 识别类型为 "tokens"
        Parser-->>Preprocessor: ParsedTokensPrompt{type:"tokens", content:TokensPrompt}
    end
    
    rect rgb(235, 245, 255)
        Note over Preprocessor,Validator: 2. Token 验证阶段
        Preprocessor->>Preprocessor: _process_tokens(parsed_content)
        
        Preprocessor->>Validator: 验证 token_ids 格式
        
        rect rgb(230, 255, 230)
            Note over Validator: Token 验证规则
            Validator->>Validator: 检查 token_ids 非空
            Validator->>Validator: 检查所有元素为 int 类型
            Validator->>Validator: 检查 token_ids 范围有效
            Validator->>Validator: 检查序列长度 <= MAX_LENGTH
            
            alt 验证失败
                Validator-->>Preprocessor: ValidationError
                Preprocessor-->>Entrypoints: 抛出异常
                Entrypoints-->>User: 400 Bad Request
            else 验证通过
                Validator-->>Preprocessor: 验证成功
            end
        end
        
        alt 包含 token_type_ids
            Preprocessor->>Validator: 验证 token_type_ids 长度匹配
            Validator->>Validator: len(token_type_ids) == len(prompt_token_ids)
            Validator-->>Preprocessor: 验证结果
        end
        
        Preprocessor->>Preprocessor: 构建 TokenInputs
        Note over Preprocessor: TokenInputs{<br/>  prompt_token_ids: [1,2,3,4,5],<br/>  prompt: "Hello world"<br/>}
    end
    
    rect rgb(245, 255, 235)
        Note over Preprocessor,Engine: 3. 输出构建阶段
        Preprocessor->>Preprocessor: _build_decoder_only_llm_inputs()
        
        Note over Preprocessor: 创建 DecoderOnlyInputs:<br/>直接使用已验证的 token_ids
        
        Preprocessor-->>Entrypoints: DecoderOnlyInputs
        Entrypoints->>Engine: add_request(processed_inputs)
        Engine-->>Entrypoints: request_id
        Entrypoints-->>User: 请求处理成功
    end
```

### Token 验证详细流程

```mermaid
graph TB
    subgraph "Token 验证管道"
        Input["Token IDs 输入"]
        
        subgraph "基础验证"
            EmptyCheck["空值检查"]
            TypeCheck["类型检查"]
            RangeCheck["范围检查"]
        end
        
        subgraph "长度验证"
            LengthCheck["序列长度检查"]
            MaxCheck["最大长度限制"]
        end
        
        subgraph "可选验证"
            TokenTypeCheck["Token Type IDs 验证"]
            SemanticCheck["语义合理性检查"]
        end
        
        subgraph "输出"
            Success["验证通过"]
            Error["验证失败"]
        end
    end
    
    Input --> EmptyCheck
    EmptyCheck --> TypeCheck
    TypeCheck --> RangeCheck
    RangeCheck --> LengthCheck
    LengthCheck --> MaxCheck
    MaxCheck --> TokenTypeCheck
    TokenTypeCheck --> SemanticCheck
    
    SemanticCheck --> Success
    
    EmptyCheck -.-> Error
    TypeCheck -.-> Error
    RangeCheck -.-> Error
    LengthCheck -.-> Error
    MaxCheck -.-> Error
    TokenTypeCheck -.-> Error
```

---

## 场景 3：多模态输入融合处理

### 业务场景
用户提交包含文本和图像的多模态输入，需要分别处理并融合。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Entrypoints as Entrypoints 层
    participant Preprocessor as InputPreprocessor
    participant Tokenizer as Tokenizer
    participant MMProcessor as 多模态处理器
    participant ImageProcessor as 图像处理器
    participant FeatureFusion as 特征融合器
    participant Engine as Engine
    
    Note over User,Engine: 多模态输入的完整处理流程
    
    rect rgb(255, 245, 235)
        Note over User,Engine: 1. 多模态输入接收
        User->>Entrypoints: 提交多模态请求
        Note over User: {<br/>  "prompt": "Describe this image",<br/>  "multi_modal_data": {"image": [image_data]}<br/>}
        Entrypoints->>Preprocessor: preprocess(multimodal_prompt)
        Preprocessor->>Preprocessor: parse_singleton_prompt() → "text" 类型
        Preprocessor->>Preprocessor: _process_text(parsed_content)
    end
    
    rect rgb(235, 245, 255)
        Note over Preprocessor,FeatureFusion: 2. 并行处理阶段
        
        par 文本处理
            Preprocessor->>Tokenizer: encode(text_prompt)
            Tokenizer->>Tokenizer: 分词："Describe this image"
            Tokenizer-->>Preprocessor: text_token_ids [1, 2345, 678, 910]
        and 图像处理
            Preprocessor->>MMProcessor: process_multimodal_data(image_data)
            MMProcessor->>ImageProcessor: 处理原始图像数据
            
            rect rgb(230, 255, 230)
                Note over ImageProcessor: 图像预处理管道
                ImageProcessor->>ImageProcessor: 格式转换 (PIL → Tensor)
                ImageProcessor->>ImageProcessor: 尺寸调整 (resize to 224x224)
                ImageProcessor->>ImageProcessor: 归一化 (normalize pixel values)
                ImageProcessor->>ImageProcessor: 特征提取 (CNN backbone)
                Note over ImageProcessor: image_features: [batch, 196, 768]
            end
            
            ImageProcessor-->>MMProcessor: processed_image_features
            MMProcessor-->>Preprocessor: image_token_embeddings
        end
        
        Note over Preprocessor: 文本 tokens: [1, 2345, 678, 910]<br/>图像 features: [196, 768]
    end
    
    rect rgb(245, 255, 235)
        Note over Preprocessor,Engine: 3. 特征融合阶段
        Preprocessor->>FeatureFusion: 融合文本和图像特征
        
        alt 序列拼接策略
            FeatureFusion->>FeatureFusion: 图像 token 插入文本序列
            Note over FeatureFusion: [1, <img_1>, <img_2>, ..., <img_196>, 2345, 678, 910]
        else 交错融合策略
            FeatureFusion->>FeatureFusion: 文本和图像特征交错排列
            Note over FeatureFusion: [1, <img_1>, 2345, <img_2>, 678, <img_3>, ...]
        else 注意力融合策略
            FeatureFusion->>FeatureFusion: 通过注意力机制融合
            Note over FeatureFusion: cross_attention(text_features, image_features)
        end
        
        FeatureFusion-->>Preprocessor: fused_multimodal_inputs
        
        Preprocessor->>Preprocessor: 构建 MultiModalInputs
        Note over Preprocessor: MultiModalInputs{<br/>  prompt_token_ids: [...],<br/>  multi_modal_data: processed_data<br/>}
    end
    
    rect rgb(255, 235, 245)
        Note over Preprocessor,Engine: 4. 输出构建阶段
        Preprocessor->>Preprocessor: _build_decoder_only_llm_inputs()
        
        Note over Preprocessor: 创建 DecoderOnlyInputs:<br/>包含融合后的多模态表示
        
        Preprocessor-->>Entrypoints: DecoderOnlyInputs (多模态)
        Entrypoints->>Engine: add_request(multimodal_inputs)
        Engine-->>Entrypoints: request_id
        Entrypoints-->>User: 多模态请求处理成功
    end
```

### 多模态数据处理详解

```mermaid
graph TB
    subgraph "多模态处理管道"
        RawInput["原始多模态输入"]
        
        subgraph "图像处理"
            ImageInput["图像数据"]
            ImageResize["尺寸调整"]
            ImageNorm["像素归一化"]
            ImageEmbed["特征提取"]
            ImageTokens["图像 Tokens"]
        end
        
        subgraph "音频处理"
            AudioInput["音频数据"]
            AudioSTFT["短时傅里叶变换"]
            AudioMel["梅尔频谱"]
            AudioEmbed["音频嵌入"]
            AudioTokens["音频 Tokens"]
        end
        
        subgraph "文本处理"
            TextInput["文本数据"]
            Tokenization["分词"]
            TextEmbed["文本嵌入"]
            TextTokens["文本 Tokens"]
        end
        
        subgraph "融合策略"
            Concatenation["序列拼接"]
            Interleaving["交错融合"]
            CrossAttention["交叉注意力"]
            FusedOutput["融合输出"]
        end
    end
    
    RawInput --> ImageInput
    RawInput --> AudioInput
    RawInput --> TextInput
    
    ImageInput --> ImageResize --> ImageNorm --> ImageEmbed --> ImageTokens
    AudioInput --> AudioSTFT --> AudioMel --> AudioEmbed --> AudioTokens
    TextInput --> Tokenization --> TextEmbed --> TextTokens
    
    ImageTokens --> Concatenation
    AudioTokens --> Concatenation
    TextTokens --> Concatenation
    
    ImageTokens --> Interleaving
    AudioTokens --> Interleaving
    TextTokens --> Interleaving
    
    ImageTokens --> CrossAttention
    AudioTokens --> CrossAttention
    TextTokens --> CrossAttention
    
    Concatenation --> FusedOutput
    Interleaving --> FusedOutput
    CrossAttention --> FusedOutput
```

---

## 场景 4：嵌入输入直接处理

### 业务场景
用户直接提供预计算的嵌入向量，跳过 token embedding lookup。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Entrypoints as Entrypoints 层
    participant Preprocessor as InputPreprocessor
    participant Parser as parse_singleton_prompt
    participant Validator as 嵌入验证器
    participant Engine as Engine
    
    Note over User,Engine: 嵌入输入的直接处理流程
    
    rect rgb(255, 245, 235)
        Note over User,Engine: 1. 嵌入输入接收
        User->>Entrypoints: 提交 EmbedsPrompt
        Note over User: {<br/>  "inputs_embeds": tensor([10, 4096]),<br/>  "cache_salt": "custom_embeds"<br/>}
        Entrypoints->>Preprocessor: preprocess(embeds_prompt)
        Preprocessor->>Parser: parse_singleton_prompt(embeds_prompt)
        Parser->>Parser: 检查字典包含 "inputs_embeds"
        Parser->>Parser: 识别类型为 "embeds"
        Parser-->>Preprocessor: ParsedEmbedsPrompt{type:"embeds", content:EmbedsPrompt}
    end
    
    rect rgb(235, 245, 255)
        Note over Preprocessor,Validator: 2. 嵌入验证阶段
        Preprocessor->>Preprocessor: _process_embeds(parsed_content)
        
        Preprocessor->>Validator: 验证 inputs_embeds 张量
        
        rect rgb(230, 255, 230)
            Note over Validator: 嵌入验证规则
            Validator->>Validator: 检查张量维度 == 2D
            Note over Validator: shape: [seq_len, hidden_size]
            
            Validator->>Validator: 检查序列长度 > 0
            Validator->>Validator: 检查隐层维度匹配模型配置
            Note over Validator: hidden_size == model.config.hidden_size
            
            Validator->>Validator: 检查数值有效性
            Note over Validator: 无 NaN、Inf 值
            
            Validator->>Validator: 检查数据类型
            Note over Validator: 支持 float32、float16、bfloat16
            
            alt 验证失败
                Validator-->>Preprocessor: ValidationError
                Preprocessor-->>Entrypoints: 抛出异常
                Entrypoints-->>User: 400 Bad Request
            else 验证通过
                Validator-->>Preprocessor: 验证成功
            end
        end
        
        Preprocessor->>Preprocessor: 构建 EmbedsInputs
        Note over Preprocessor: EmbedsInputs{<br/>  inputs_embeds: tensor([10, 4096]),<br/>  cache_salt: "custom_embeds"<br/>}
    end
    
    rect rgb(245, 255, 235)
        Note over Preprocessor,Engine: 3. 直接传递阶段
        Preprocessor->>Preprocessor: _build_decoder_only_llm_inputs()
        
        Note over Preprocessor: 创建 DecoderOnlyInputs:<br/>直接使用 inputs_embeds，跳过 embedding lookup
        
        Preprocessor-->>Entrypoints: DecoderOnlyInputs (嵌入)
        Entrypoints->>Engine: add_request(embeds_inputs)
        Engine->>Engine: 跳过 embedding layer，直接进入 transformer
        Engine-->>Entrypoints: request_id
        Entrypoints-->>User: 嵌入输入处理成功
    end
```

### 嵌入验证详细规则

```mermaid
graph TB
    subgraph "嵌入验证管道"
        EmbedInput["输入嵌入张量"]
        
        subgraph "形状验证"
            DimCheck["维度检查 (必须是2D)"]
            SeqLenCheck["序列长度检查 (>0)"]
            HiddenCheck["隐层维度检查"]
        end
        
        subgraph "数值验证"
            NaNCheck["NaN 值检查"]
            InfCheck["无穷值检查"]
            RangeCheck["数值范围验证"]
        end
        
        subgraph "类型验证"
            DTypeCheck["数据类型验证"]
            DeviceCheck["设备兼容性检查"]
            ContiguousCheck["内存连续性检查"]
        end
        
        subgraph "结果"
            ValidEmbeds["有效嵌入"]
            InvalidEmbeds["无效嵌入"]
        end
    end
    
    EmbedInput --> DimCheck
    DimCheck --> SeqLenCheck
    SeqLenCheck --> HiddenCheck
    HiddenCheck --> NaNCheck
    NaNCheck --> InfCheck
    InfCheck --> RangeCheck
    RangeCheck --> DTypeCheck
    DTypeCheck --> DeviceCheck
    DeviceCheck --> ContiguousCheck
    
    ContiguousCheck --> ValidEmbeds
    
    DimCheck -.-> InvalidEmbeds
    SeqLenCheck -.-> InvalidEmbeds
    HiddenCheck -.-> InvalidEmbeds
    NaNCheck -.-> InvalidEmbeds
    InfCheck -.-> InvalidEmbeds
    RangeCheck -.-> InvalidEmbeds
    DTypeCheck -.-> InvalidEmbeds
    DeviceCheck -.-> InvalidEmbeds
```

---

## 场景 5：批量输入并行处理

### 业务场景
用户提交多个输入进行批量处理，需要并行预处理以提高吞吐量。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant BatchAPI as 批量 API
    participant ThreadPool as 线程池
    participant Preprocessor1 as Preprocessor 1
    participant Preprocessor2 as Preprocessor 2
    participant Preprocessor3 as Preprocessor 3
    participant Coordinator as 协调器
    participant Engine as Engine
    
    Note over User,Engine: 批量输入的并行处理流程
    
    rect rgb(255, 245, 235)
        Note over User,Engine: 1. 批量输入接收
        User->>BatchAPI: 提交批量请求
        Note over User: [<br/>  "Explain AI",<br/>  "What is ML?",<br/>  "Describe DL"<br/>]
        BatchAPI->>BatchAPI: parse_and_batch_prompt(batch_prompts)
        BatchAPI->>BatchAPI: 分割为独立任务
        
        Note over BatchAPI: 创建 3 个并行任务:<br/>Task1: "Explain AI"<br/>Task2: "What is ML?"<br/>Task3: "Describe DL"
    end
    
    rect rgb(235, 245, 255)
        Note over ThreadPool,Coordinator: 2. 并行处理阶段
        BatchAPI->>ThreadPool: 提交并行任务
        
        par 任务 1 处理
            ThreadPool->>Preprocessor1: preprocess("Explain AI")
            Preprocessor1->>Preprocessor1: 分词和处理
            Preprocessor1-->>Coordinator: DecoderOnlyInputs_1
        and 任务 2 处理
            ThreadPool->>Preprocessor2: preprocess("What is ML?") 
            Preprocessor2->>Preprocessor2: 分词和处理
            Preprocessor2-->>Coordinator: DecoderOnlyInputs_2
        and 任务 3 处理
            ThreadPool->>Preprocessor3: preprocess("Describe DL")
            Preprocessor3->>Preprocessor3: 分词和处理
            Preprocessor3-->>Coordinator: DecoderOnlyInputs_3
        end
        
        Note over Coordinator: 等待所有任务完成
    end
    
    rect rgb(245, 255, 235)
        Note over Coordinator,Engine: 3. 结果聚合阶段
        Coordinator->>Coordinator: 聚合处理结果
        Coordinator->>Coordinator: 检查处理错误
        
        alt 所有任务成功
            Coordinator->>Engine: batch_add_requests([inputs_1, inputs_2, inputs_3])
            Engine->>Engine: 为每个输入创建独立请求
            Engine-->>Coordinator: [request_id_1, request_id_2, request_id_3]
            Coordinator-->>BatchAPI: 批量请求 ID
            BatchAPI-->>User: 批量处理成功
        else 部分任务失败
            Coordinator->>Coordinator: 标记失败的任务
            Coordinator-->>BatchAPI: 部分成功结果 + 错误列表
            BatchAPI-->>User: 207 Multi-Status (部分成功)
        end
    end
```

### 批量处理性能优化

```mermaid
graph TB
    subgraph "批量处理优化策略"
        BatchInput["批量输入"]
        
        subgraph "负载均衡"
            TaskSplit["任务分割"]
            LoadBalance["负载均衡"]
            ThreadAssign["线程分配"]
        end
        
        subgraph "缓存优化"
            SharedCache["共享缓存"]
            PrefixShare["前缀共享"]
            TokenReuse["Token 复用"]
        end
        
        subgraph "内存优化"
            MemoryPool["内存池"]
            BatchAlloc["批量分配"]
            LazyLoad["延迟加载"]
        end
        
        subgraph "并发控制"
            Semaphore["信号量控制"]
            RateLimit["速率限制"]
            BackPressure["背压机制"]
        end
        
        OptimizedOutput["优化的批量输出"]
    end
    
    BatchInput --> TaskSplit
    TaskSplit --> LoadBalance
    LoadBalance --> ThreadAssign
    
    BatchInput --> SharedCache
    SharedCache --> PrefixShare
    PrefixShare --> TokenReuse
    
    BatchInput --> MemoryPool
    MemoryPool --> BatchAlloc
    BatchAlloc --> LazyLoad
    
    BatchInput --> Semaphore
    Semaphore --> RateLimit
    RateLimit --> BackPressure
    
    ThreadAssign --> OptimizedOutput
    TokenReuse --> OptimizedOutput
    LazyLoad --> OptimizedOutput
    BackPressure --> OptimizedOutput
```

---

## 场景 6：编码器-解码器输入处理

### 业务场景
Seq2Seq 模型（如 T5、BART）需要分别处理编码器和解码器输入。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Entrypoints as Entrypoints 层
    participant Preprocessor as InputPreprocessor
    participant Parser as parse_singleton_prompt
    participant EncTokenizer as 编码器分词器
    participant DecTokenizer as 解码器分词器
    participant Engine as Engine
    
    Note over User,Engine: 编码器-解码器输入的处理流程
    
    rect rgb(255, 245, 235)
        Note over User,Engine: 1. 编码器-解码器输入接收
        User->>Entrypoints: 提交 Seq2Seq 请求
        Note over User: {<br/>  "encoder_prompt": "translate English to German: Hello",<br/>  "decoder_prompt": "Hallo"<br/>}
        Entrypoints->>Preprocessor: preprocess(enc_dec_prompt)
        Preprocessor->>Parser: is_explicit_encoder_decoder_prompt()
        Parser-->>Preprocessor: True (确认为编码器-解码器格式)
        Preprocessor->>Preprocessor: _process_encoder_decoder_prompt()
    end
    
    rect rgb(235, 245, 255)
        Note over Preprocessor,DecTokenizer: 2. 双路并行处理
        
        par 编码器输入处理
            Preprocessor->>Preprocessor: _prompt_to_llm_inputs(encoder_prompt)
            Preprocessor->>EncTokenizer: encode("translate English to German: Hello")
            EncTokenizer->>EncTokenizer: 添加特殊 token [CLS] ... [SEP]
            EncTokenizer-->>Preprocessor: encoder_token_ids [1, 567, 89, ..., 2]
            
            Note over Preprocessor: encoder_inputs = {<br/>  prompt_token_ids: [1, 567, 89, ..., 2]<br/>}
        and 解码器输入处理
            alt 提供了解码器 prompt
                Preprocessor->>Preprocessor: _prompt_to_llm_inputs(decoder_prompt)
                Preprocessor->>DecTokenizer: encode("Hallo")
                DecTokenizer->>DecTokenizer: 添加起始 token [BOS] Hallo
                DecTokenizer-->>Preprocessor: decoder_token_ids [0, 12345]
                
                Note over Preprocessor: decoder_inputs = {<br/>  prompt_token_ids: [0, 12345]<br/>}
            else 无解码器 prompt（自由生成）
                Note over Preprocessor: decoder_inputs = None<br/>（模型将从 [BOS] 开始生成）
            end
        end
        
        Note over Preprocessor: 编码器和解码器输入已分别处理完成
    end
    
    rect rgb(245, 255, 235)
        Note over Preprocessor,Engine: 3. 编码器-解码器输入构建
        Preprocessor->>Preprocessor: _build_enc_dec_llm_inputs()
        
        Preprocessor->>Preprocessor: 构建 EncoderDecoderInputs
        Note over Preprocessor: EncoderDecoderInputs {<br/>  encoder_prompt_token_ids: [1, 567, 89, ..., 2],<br/>  decoder_prompt_token_ids: [0, 12345],<br/>  encoder_multi_modal_data: None,<br/>  decoder_multi_modal_data: None<br/>}
        
        Preprocessor-->>Entrypoints: EncoderDecoderInputs
        Entrypoints->>Engine: add_request(enc_dec_inputs)
        Engine->>Engine: 初始化编码器-解码器推理状态
        Engine-->>Entrypoints: request_id
        Entrypoints-->>User: Seq2Seq 请求处理成功
    end
```

### 编码器-解码器处理架构

```mermaid
graph TB
    subgraph "编码器-解码器处理架构"
        Input["Explicit Enc-Dec Prompt"]
        
        subgraph "输入分离"
            EncInput["编码器输入"]
            DecInput["解码器输入"]
        end
        
        subgraph "编码器路径"
            EncParser["编码器解析"]
            EncTokenizer["编码器分词"]
            EncValidation["编码器验证"]
            EncOutput["编码器输出"]
        end
        
        subgraph "解码器路径"
            DecParser["解码器解析"]
            DecTokenizer["解码器分词"]
            DecValidation["解码器验证"]
            DecOutput["解码器输出"]
        end
        
        subgraph "融合层"
            InputMerger["输入合并器"]
            CrossValidation["交叉验证"]
            FinalOutput["最终输出"]
        end
    end
    
    Input --> EncInput
    Input --> DecInput
    
    EncInput --> EncParser --> EncTokenizer --> EncValidation --> EncOutput
    DecInput --> DecParser --> DecTokenizer --> DecValidation --> DecOutput
    
    EncOutput --> InputMerger
    DecOutput --> InputMerger
    
    InputMerger --> CrossValidation --> FinalOutput
```

---

## 错误处理和恢复时序

### 输入验证失败处理

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Preprocessor as InputPreprocessor
    participant Validator as 验证器
    participant ErrorHandler as 错误处理器
    participant Logger as 日志系统
    
    rect rgb(255, 235, 235)
        Note over User,Logger: 错误场景：无效输入处理
        User->>Preprocessor: 提交无效输入
        Note over User: 错误示例：空的 token_ids 或无效维度的嵌入
        
        Preprocessor->>Validator: 验证输入格式
        
        alt Token 验证失败
            Validator->>Validator: 发现 token_ids 为空
            Validator-->>ErrorHandler: ValidationError("empty token_ids")
        else 嵌入验证失败
            Validator->>Validator: 发现嵌入维度不匹配
            Validator-->>ErrorHandler: ValidationError("dimension mismatch")
        else 多模态验证失败
            Validator->>Validator: 发现图像格式不支持
            Validator-->>ErrorHandler: ValidationError("unsupported image format")
        end
        
        ErrorHandler->>Logger: 记录验证错误
        ErrorHandler->>ErrorHandler: 生成用户友好的错误消息
        ErrorHandler-->>Preprocessor: 格式化的错误响应
        Preprocessor-->>User: 400 Bad Request + 详细错误信息
    end
```

---

## 性能监控时序

### 处理性能指标收集

```mermaid
sequenceDiagram
    autonumber
    participant Request as 请求
    participant Preprocessor as InputPreprocessor
    participant Metrics as 性能监控
    participant Dashboard as 监控面板
    
    rect rgb(235, 255, 235)
        Note over Request,Dashboard: 性能监控流程
        Request->>Preprocessor: 开始处理
        Preprocessor->>Metrics: 开始计时 (start_time)
        
        Preprocessor->>Preprocessor: 执行预处理逻辑
        
        alt 处理成功
            Preprocessor->>Metrics: 记录成功 (end_time, success=True)
            Metrics->>Metrics: 计算处理延迟
            Metrics->>Metrics: 更新成功率统计
        else 处理失败
            Preprocessor->>Metrics: 记录失败 (end_time, success=False, error_type)
            Metrics->>Metrics: 更新错误率统计
            Metrics->>Metrics: 分类错误类型
        end
        
        Metrics->>Dashboard: 更新实时指标
        Note over Dashboard: 显示：<br/>- 平均处理延迟<br/>- 成功率/错误率<br/>- 吞吐量<br/>- 错误分布
    end
```

---

## 使用示例总结

### 常见处理模式

```python
# 1. 简单文本处理
text_prompt = "Explain machine learning"
processed = preprocessor.preprocess(text_prompt)
# → DecoderOnlyInputs

# 2. 预分词输入
tokens_prompt = TokensPrompt(prompt_token_ids=[1, 2, 3, 4])
processed = preprocessor.preprocess(tokens_prompt)
# → DecoderOnlyInputs (跳过分词)

# 3. 多模态输入
mm_prompt = TextPrompt(
    prompt="Describe this image",
    multi_modal_data={"image": [image_data]}
)
processed = preprocessor.preprocess(mm_prompt)
# → DecoderOnlyInputs (含多模态数据)

# 4. 编码器-解码器
enc_dec_prompt = build_explicit_enc_dec_prompt(
    encoder_prompt="translate: Hello",
    decoder_prompt="Bonjour"
)
processed = preprocessor.preprocess(enc_dec_prompt)
# → EncoderDecoderInputs

# 5. 批量处理
batch_prompts = ["Hello", "Hi", "Good morning"]
parsed_batch = parse_and_batch_prompt(batch_prompts)
processed_batch = [preprocessor.preprocess(p) for p in parsed_batch]
```

---

## 总结

InputsOutputs 模块的时序图展示了：

1. **多格式支持**：文本、Token、嵌入、多模态、编码器-解码器等格式的统一处理
2. **验证机制**：完整的输入验证和错误处理流程
3. **并行处理**：批量输入的高效并行处理策略
4. **性能优化**：缓存、内存管理、负载均衡等优化机制
5. **错误恢复**：健壮的错误处理和用户友好的错误报告

**关键设计要点**：
- **统一接口**：不同输入格式通过统一的预处理接口处理
- **类型安全**：严格的类型检查和验证机制
- **性能优化**：缓存、并行处理、内存优化
- **可扩展性**：模块化设计支持新的输入类型和处理策略

通过这些时序图，可以深入理解 vLLM 输入处理系统的工作机制和优化策略。
