# VLLM关键流程时序图分析

## 1. 总体请求处理时序图

### 1.1 完整推理流程

```mermaid
sequenceDiagram
    participant User as 用户应用
    participant LLM as LLM类
    participant Proc as Processor
    participant Engine as EngineCore
    participant Sched as Scheduler
    participant KVMgr as KVCacheManager
    participant Exec as ModelExecutor
    participant Worker as GPUWorker
    participant Model as 模型
    participant Attn as AttentionBackend

    Note over User,Attn: 1. 请求初始化阶段
    User->>LLM: generate(prompts, sampling_params)
    LLM->>LLM: validate_inputs()
    LLM->>Proc: process_requests()
    
    Note over Proc: 输入预处理
    Proc->>Proc: validate_params()
    Proc->>Proc: tokenize_prompts()
    Proc->>Proc: process_multimodal_data()
    Proc->>Engine: add_requests()
    
    Note over Engine,Attn: 2. 调度与执行循环
    loop 推理循环 [直到所有请求完成]
        Engine->>Sched: schedule()
        
        Note over Sched,KVMgr: 调度决策阶段
        Sched->>KVMgr: check_available_memory()
        KVMgr-->>Sched: memory_status
        Sched->>Sched: select_requests_to_run()
        Sched->>KVMgr: allocate_kv_cache()
        KVMgr-->>Sched: allocated_blocks
        
        Note over Sched,Attn: 模型执行阶段  
        Sched->>Exec: execute_model_batch()
        Exec->>Worker: prepare_inputs()
        Worker->>Worker: build_input_tensors()
        
        Worker->>Model: forward()
        Model->>Attn: attention_forward()
        
        Note over Attn: PagedAttention计算
        Attn->>Attn: paged_attention_v2()
        Attn-->>Model: attention_output
        
        Model->>Model: feedforward()
        Model-->>Worker: logits
        
        Worker->>Worker: sample_tokens()
        Worker-->>Exec: new_tokens
        Exec-->>Sched: execution_results
        
        Note over Sched: 状态更新阶段
        Sched->>Sched: update_request_states()
        Sched->>KVMgr: update_kv_cache()
        Sched-->>Engine: scheduler_output
    end
    
    Note over Engine,LLM: 3. 结果返回阶段
    Engine->>Engine: collect_finished_requests()
    Engine-->>LLM: request_outputs
    LLM->>LLM: post_process_outputs()
    LLM-->>User: generated_texts
```

**关键阶段说明**：

1. **请求初始化阶段（1-3秒）**：
   - 输入验证和预处理
   - Token化和多模态数据处理
   - 请求对象创建

2. **调度与执行循环（主要耗时）**：
   - 每个循环处理一个或多个推理步骤
   - 包含调度决策、内存管理、模型执行
   - 动态调整批次大小和内存分配

3. **结果返回阶段（<1秒）**：
   - 收集完成的请求
   - 后处理和格式化输出

## 2. 详细子流程分析

### 2.1 输入处理时序图

```mermaid
sequenceDiagram
    participant LLM as LLM类
    participant Proc as Processor
    participant PreProc as InputPreprocessor
    participant Tokenizer as Tokenizer
    participant MMProc as MultiModalProcessor
    
    Note over LLM,MMProc: 输入预处理详细流程
    LLM->>Proc: process_request(prompt, params)
    
    Proc->>Proc: _validate_params(params)
    alt 参数验证失败
        Proc-->>LLM: ValidationError
    else 参数验证成功
        Proc->>PreProc: preprocess(prompt, lora_request)
        
        alt 文本输入
            PreProc->>Tokenizer: encode(prompt)
            Tokenizer-->>PreProc: token_ids
        else Token输入
            PreProc->>PreProc: validate_token_ids()
        end
        
        alt 包含多模态数据
            PreProc->>MMProc: process_multimodal_data()
            MMProc->>MMProc: process_images()
            MMProc->>MMProc: process_audio()
            MMProc->>MMProc: extract_features()
            MMProc-->>PreProc: mm_inputs
        end
        
        PreProc->>PreProc: combine_inputs()
        PreProc-->>Proc: ProcessorInputs
        Proc-->>LLM: processed_request
    end
```

**处理步骤详解**：

1. **参数验证（~1ms）**：
   - 检查采样参数范围
   - 验证结构化输出配置
   - 确认token ID有效性

2. **文本预处理（1-10ms）**：
   - 分词处理
   - 特殊token添加
   - 序列长度检查

3. **多模态处理（10-100ms）**：
   - 图像特征提取
   - 音频预处理
   - 特征融合

### 2.2 调度器详细时序图

```mermaid
sequenceDiagram
    participant Engine as EngineCore
    participant Sched as Scheduler
    participant ReqQueue as RequestQueue
    participant KVMgr as KVCacheManager
    participant BlockPool as BlockPool
    participant PrefixCache as PrefixCache
    
    Note over Engine,PrefixCache: 调度器内部流程
    Engine->>Sched: schedule()
    
    Note over Sched,PrefixCache: 1. 收集待调度请求
    Sched->>ReqQueue: get_waiting_requests()
    ReqQueue-->>Sched: waiting_requests
    Sched->>ReqQueue: get_running_requests()
    ReqQueue-->>Sched: running_requests
    
    Note over Sched,PrefixCache: 2. 内存预算计算
    Sched->>KVMgr: get_available_blocks()
    KVMgr->>BlockPool: get_free_blocks()
    BlockPool-->>KVMgr: free_count
    KVMgr-->>Sched: available_memory
    
    Note over Sched,PrefixCache: 3. 请求选择和资源分配
    loop 对每个待处理请求
        Sched->>Sched: estimate_memory_cost(request)
        
        alt 内存足够
            Sched->>KVMgr: allocate_blocks(request)
            KVMgr->>PrefixCache: find_reusable_blocks()
            PrefixCache-->>KVMgr: cached_blocks
            KVMgr->>BlockPool: allocate_new_blocks()
            BlockPool-->>KVMgr: new_blocks
            KVMgr-->>Sched: allocated_blocks
            Sched->>Sched: add_to_batch(request)
        else 内存不足
            Sched->>Sched: try_preemption()
            alt 可以抢占
                Sched->>KVMgr: free_blocks(preempted_request)
                Sched->>ReqQueue: move_to_waiting(preempted_request)
            else 无法抢占
                Sched->>Sched: skip_request()
            end
        end
    end
    
    Note over Sched,PrefixCache: 4. 生成调度输出
    Sched->>Sched: create_scheduler_output()
    Sched-->>Engine: SchedulerOutput
```

**调度策略说明**：

1. **内存预算管理**：
   - 基于可用GPU内存动态调整批次大小
   - 考虑KV缓存、激活值、梯度的内存需求
   - 预留安全边际防止OOM

2. **抢占式调度**：
   - 优先级高的请求可以抢占资源
   - 支持部分完成请求的暂停和恢复
   - 最小化重新计算开销

3. **前缀缓存优化**：
   - 识别相同前缀的请求
   - 共享KV缓存块
   - 减少重复计算

### 2.3 KV缓存管理时序图

```mermaid
sequenceDiagram
    participant Sched as Scheduler
    participant KVMgr as KVCacheManager
    participant BlockPool as BlockPool  
    participant PrefixCache as PrefixCache
    participant HashFunc as BlockHasher
    participant GPU as GPU内存
    
    Note over Sched,GPU: KV缓存分配流程
    Sched->>KVMgr: allocate_blocks(seq_id, num_tokens)
    
    Note over KVMgr,GPU: 1. 计算资源需求
    KVMgr->>KVMgr: calculate_blocks_needed(num_tokens)
    KVMgr->>HashFunc: compute_prefix_hash(seq_id)
    HashFunc-->>KVMgr: prefix_hash
    
    Note over KVMgr,GPU: 2. 检查前缀缓存
    KVMgr->>PrefixCache: get_cached_blocks(prefix_hash)
    PrefixCache-->>KVMgr: cached_block_ids
    
    Note over KVMgr,GPU: 3. 分配新块
    KVMgr->>BlockPool: allocate(num_needed_blocks)
    loop 对每个需要的块
        BlockPool->>BlockPool: get_free_block()
        alt 有可用块
            BlockPool->>GPU: allocate_gpu_memory(block_id)
            GPU-->>BlockPool: memory_ptr
            BlockPool-->>KVMgr: block_id
        else 无可用块
            BlockPool->>BlockPool: trigger_eviction()
            alt 驱逐成功
                BlockPool->>GPU: free_gpu_memory(evicted_block)
                BlockPool->>GPU: allocate_gpu_memory(block_id)
                GPU-->>BlockPool: memory_ptr
                BlockPool-->>KVMgr: block_id
            else 驱逐失败
                BlockPool-->>KVMgr: OutOfMemoryError
            end
        end
    end
    
    Note over KVMgr,GPU: 4. 更新映射关系
    KVMgr->>KVMgr: update_seq_to_blocks_mapping()
    KVMgr->>PrefixCache: update_cache(prefix_hash, blocks)
    KVMgr-->>Sched: allocated_block_ids
```

**缓存管理策略**：

1. **分页内存管理**：
   - 固定大小的内存页（通常16-64 tokens）
   - 动态分配和释放
   - 支持非连续内存布局

2. **前缀缓存共享**：
   - 基于内容哈希的缓存键
   - 引用计数管理
   - LRU驱逐策略

3. **内存驱逐机制**：
   - 优先驱逐未激活的缓存
   - 支持CPU-GPU内存交换
   - 最小化重新计算代价

### 2.4 模型执行时序图

```mermaid
sequenceDiagram
    participant Sched as Scheduler
    participant Exec as ModelExecutor
    participant Worker as GPUWorker
    participant Runner as ModelRunner
    participant Model as 模型
    participant Attn as AttentionBackend
    participant KVCache as KVCache
    participant Sampler as Sampler
    
    Note over Sched,Sampler: 模型执行详细流程
    Sched->>Exec: execute_model(scheduler_output)
    
    Note over Exec,Sampler: 1. 输入准备阶段
    Exec->>Worker: prepare_model_inputs()
    Worker->>Worker: build_input_tensors()
    Worker->>Worker: prepare_attention_metadata()
    Worker->>KVCache: get_kv_cache_refs()
    KVCache-->>Worker: cache_references
    
    Note over Exec,Sampler: 2. 分布式协调
    alt 多GPU并行
        Exec->>Exec: broadcast_inputs_to_workers()
        Exec->>Exec: synchronize_workers()
    end
    
    Note over Exec,Sampler: 3. 模型前向传播
    Worker->>Runner: execute_model()
    Runner->>Model: forward()
    
    Note over Model,Sampler: 3.1 嵌入层
    Model->>Model: embedding_forward()
    
    Note over Model,Sampler: 3.2 Transformer层循环
    loop 对每个Transformer层
        Model->>Attn: attention_forward()
        
        Note over Attn,KVCache: PagedAttention计算
        Attn->>KVCache: get_kv_blocks()
        KVCache-->>Attn: key_cache, value_cache
        Attn->>Attn: scaled_dot_product_attention()
        Attn->>Attn: apply_rotary_embedding()
        Attn->>Attn: paged_attention_kernel()
        Attn-->>Model: attention_output
        
        Model->>Model: feedforward()
        Model->>Model: layer_norm()
    end
    
    Note over Model,Sampler: 3.3 输出层
    Model->>Model: output_projection()
    Model-->>Runner: logits
    
    Note over Runner,Sampler: 4. 采样和后处理
    Runner->>Sampler: sample_tokens()
    Sampler->>Sampler: apply_temperature()
    Sampler->>Sampler: top_k_top_p_filtering()
    Sampler->>Sampler: multinomial_sampling()
    Sampler-->>Runner: sampled_tokens
    
    Runner->>Runner: update_sequences()
    Runner-->>Worker: model_output
    Worker-->>Exec: execution_result
    
    Note over Exec,Sampler: 5. 分布式聚合
    alt 多GPU并行
        Exec->>Exec: gather_outputs_from_workers()
        Exec->>Exec: aggregate_results()
    end
    
    Exec-->>Sched: ModelRunnerOutput
```

**模型执行关键点**：

1. **输入准备（1-5ms）**：
   - 构建输入张量批次
   - 准备注意力掩码
   - 获取KV缓存引用

2. **前向传播（10-1000ms）**：
   - 嵌入层计算
   - 多层Transformer处理  
   - PagedAttention高效计算

3. **采样后处理（1-10ms）**：
   - 温度缩放
   - Top-k/Top-p过滤
   - 多项式采样

### 2.5 PagedAttention详细时序图

```mermaid
sequenceDiagram
    participant Model as Transformer层
    participant Attn as AttentionBackend
    participant KVMgr as KVCacheManager
    participant BlockTable as 块表
    participant GPU as GPU内核
    
    Note over Model,GPU: PagedAttention核心流程
    Model->>Attn: attention_forward(query, key, value, metadata)
    
    Note over Attn,GPU: 1. 缓存管理
    Attn->>KVMgr: update_kv_cache(key, value)
    KVMgr->>BlockTable: get_block_tables()
    BlockTable-->>KVMgr: block_mappings
    KVMgr->>KVMgr: write_to_cache_blocks()
    KVMgr-->>Attn: cache_updated
    
    Note over Attn,GPU: 2. 注意力计算
    Attn->>Attn: prepare_attention_inputs()
    Attn->>GPU: launch_paged_attention_kernel()
    
    Note over GPU: 2.1 内核执行细节
    GPU->>GPU: load_query_from_global_mem()
    loop 对每个注意力头
        loop 对每个序列
            GPU->>GPU: load_keys_from_paged_cache()
            GPU->>GPU: compute_qk_scores()
            GPU->>GPU: apply_softmax()
            GPU->>GPU: load_values_from_paged_cache()
            GPU->>GPU: compute_weighted_values()
        end
        GPU->>GPU: reduce_across_sequences()
    end
    GPU->>GPU: store_output_to_global_mem()
    
    GPU-->>Attn: attention_output
    Attn-->>Model: output_tensor
```

**PagedAttention优化点**：

1. **内存访问优化**：
   - 合并全局内存访问
   - 最大化共享内存利用
   - 减少内存带宽瓶颈

2. **并行策略**：
   - 头并行：不同注意力头并行计算
   - 序列并行：长序列分块并行
   - 批并行：多个序列同时处理

3. **缓存友好性**：
   - 页表索引优化
   - 块内数据布局优化
   - 预取策略

## 3. 异步处理时序图

### 3.1 异步LLM引擎

```mermaid
sequenceDiagram
    participant User as 用户应用
    participant AsyncLLM as AsyncLLMEngine
    participant Engine as EngineCore
    participant BgLoop as 后台循环
    participant Queue as 请求队列
    
    Note over User,Queue: 异步处理架构
    User->>AsyncLLM: generate_async(prompt)
    AsyncLLM->>Queue: add_request()
    AsyncLLM-->>User: Future[RequestOutput]
    
    Note over BgLoop,Queue: 后台处理循环
    loop 后台执行循环
        BgLoop->>Queue: get_pending_requests()
        Queue-->>BgLoop: request_batch
        BgLoop->>Engine: process_batch()
        Engine-->>BgLoop: outputs
        BgLoop->>BgLoop: complete_futures()
        BgLoop->>User: notify_completion()
    end
    
    User->>User: await_results()
    User->>AsyncLLM: get_results()
```

### 3.2 流式输出时序图

```mermaid
sequenceDiagram
    participant User as 用户应用
    participant Stream as StreamingIterator
    participant Engine as EngineCore
    participant Buffer as 输出缓冲
    
    Note over User,Buffer: 流式输出流程
    User->>Stream: generate_stream(prompt)
    Stream->>Engine: start_generation()
    
    loop 流式生成循环
        Engine->>Engine: generate_next_token()
        Engine->>Buffer: append_token()
        Buffer->>Stream: notify_new_token()
        Stream-->>User: yield token
        User->>User: process_partial_result()
    end
    
    Engine->>Buffer: mark_complete()
    Buffer->>Stream: notify_completion()
    Stream-->>User: final_result
```

## 4. 错误处理和恢复时序图

### 4.1 OOM错误处理

```mermaid
sequenceDiagram
    participant Sched as Scheduler
    participant KVMgr as KVCacheManager  
    participant BlockPool as BlockPool
    participant Preemption as 抢占管理器
    
    Note over Sched,Preemption: OOM错误处理流程
    Sched->>KVMgr: allocate_blocks()
    KVMgr->>BlockPool: allocate()
    BlockPool-->>KVMgr: OutOfMemoryError
    
    KVMgr->>Preemption: trigger_preemption()
    Preemption->>Preemption: select_preemption_candidates()
    Preemption->>KVMgr: free_blocks(preempted_requests)
    KVMgr->>BlockPool: release_blocks()
    
    BlockPool-->>KVMgr: blocks_freed
    KVMgr->>BlockPool: retry_allocation()
    alt 重试成功
        BlockPool-->>KVMgr: allocated_blocks
        KVMgr-->>Sched: allocation_success
    else 重试失败
        KVMgr-->>Sched: allocation_failed
        Sched->>Sched: defer_request()
    end
```

### 4.2 请求超时处理

```mermaid
sequenceDiagram
    participant Timer as 超时定时器
    participant Engine as EngineCore
    participant Request as 请求对象
    participant Cleanup as 清理服务
    
    Note over Timer,Cleanup: 超时处理机制
    Timer->>Timer: check_request_timeouts()
    Timer->>Engine: get_running_requests()
    Engine-->>Timer: request_list
    
    loop 检查每个请求
        Timer->>Request: check_timeout()
        alt 请求超时
            Timer->>Engine: abort_request()
            Engine->>Cleanup: cleanup_request_resources()
            Cleanup->>Cleanup: free_kv_cache()
            Cleanup->>Cleanup: release_gpu_memory()
            Cleanup-->>Engine: cleanup_complete
            Engine-->>Timer: request_aborted
        else 请求未超时
            Timer->>Timer: continue_monitoring()
        end
    end
```

## 5. 性能监控时序图

### 5.1 指标收集流程

```mermaid
sequenceDiagram
    participant Engine as EngineCore
    participant Collector as MetricCollector
    participant Prometheus as Prometheus
    participant Dashboard as 监控面板
    
    Note over Engine,Dashboard: 性能监控流程
    loop 指标收集循环
        Engine->>Collector: collect_metrics()
        Collector->>Collector: aggregate_request_metrics()
        Collector->>Collector: calculate_throughput()
        Collector->>Collector: measure_latency()
        Collector->>Prometheus: push_metrics()
        Prometheus-->>Dashboard: update_dashboard
    end
    
    Dashboard->>Dashboard: display_realtime_metrics()
    Dashboard->>Dashboard: trigger_alerts_if_needed()
```

这些时序图展现了VLLM在不同场景下的执行流程，帮助开发者理解系统的动态行为和性能特征。通过这些图表，可以：

1. **识别性能瓶颈**：找出耗时最多的操作环节
2. **优化执行顺序**：调整组件间的交互时机
3. **设计容错机制**：理解错误传播路径
4. **监控系统健康**：建立有效的观测体系

每个时序图都标注了关键的时间节点和资源消耗，为系统调优提供了重要参考。
