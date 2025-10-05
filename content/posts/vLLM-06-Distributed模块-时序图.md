---
title: "vLLM-06-Distributed模块-时序图"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 时序图
  - 流程分析
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - vLLM-06-Distributed模块-时序图"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# vLLM-06-Distributed模块-时序图

## 时序图概览

本文档展示 Distributed 模块在不同场景下的通信时序，涵盖：

| 场景 | 通信模式 | 参与方 | 关键特征 |
|------|----------|--------|----------|
| TP AllReduce | 集合通信 | 所有 TP rank | 行并行层聚合 |
| TP AllGather | 集合通信 | 所有 TP rank | 列并行层收集 |
| Expert 并行 | All2All | 所有 EP rank | MoE 模型通信 |
| 分离式 Prefill | 点对点 | Prefill ↔ Decode | KV cache 传输 |
| 分布式初始化 | 组播 | 所有 rank | 进程组建立 |
| Pipeline 并行 | 点对点 | 相邻 PP stage | 流水线传输 |

---

## 场景 1：Tensor 并行 AllReduce 通信

### 业务场景
行并行层（如 Attention Output、FFN Down Projection）的输出需要在所有 TP rank 之间聚合。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用层
    participant Rank0 as TP Rank 0
    participant Rank1 as TP Rank 1  
    participant Rank2 as TP Rank 2
    participant Rank3 as TP Rank 3
    participant NCCL as NCCL 集合通信
    
    Note over App,NCCL: 行并行层计算完成，每个 TP rank 有部分输出
    
    rect rgb(255, 245, 235)
        Note over App,NCCL: 1. 准备阶段
        App->>Rank0: tensor_model_parallel_all_reduce(output_0)
        App->>Rank1: tensor_model_parallel_all_reduce(output_1)
        App->>Rank2: tensor_model_parallel_all_reduce(output_2)
        App->>Rank3: tensor_model_parallel_all_reduce(output_3)
        
        Rank0->>Rank0: 获取 TP 通信组
        Rank1->>Rank1: 获取 TP 通信组
        Rank2->>Rank2: 获取 TP 通信组
        Rank3->>Rank3: 获取 TP 通信组
    end
    
    rect rgb(235, 245, 255)
        Note over Rank0,NCCL: 2. AllReduce 执行阶段
        Rank0->>NCCL: all_reduce(output_0, group=tp_group)
        Rank1->>NCCL: all_reduce(output_1, group=tp_group)
        Rank2->>NCCL: all_reduce(output_2, group=tp_group)
        Rank3->>NCCL: all_reduce(output_3, group=tp_group)
        
        rect rgb(230, 255, 230)
            Note over NCCL: Ring AllReduce 算法
            NCCL->>NCCL: Phase 1: Reduce-Scatter
            Note over NCCL: 每个 rank 收到总和的 1/N
            NCCL->>NCCL: Phase 2: All-Gather  
            Note over NCCL: 每个 rank 收到完整总和
            Note over NCCL: result = output_0 + output_1 + output_2 + output_3
        end
        
        NCCL-->>Rank0: aggregated_result
        NCCL-->>Rank1: aggregated_result (相同)
        NCCL-->>Rank2: aggregated_result (相同)
        NCCL-->>Rank3: aggregated_result (相同)
    end
    
    rect rgb(245, 255, 235)
        Note over App,NCCL: 3. 完成阶段
        Rank0-->>App: aggregated_result
        Rank1-->>App: aggregated_result
        Rank2-->>App: aggregated_result
        Rank3-->>App: aggregated_result
        
        Note over App: 所有 TP rank 现在有相同的聚合结果
        Note over App: 可以继续下一层计算
    end
```

### 关键要点说明

1. **同步性**：所有 TP rank 必须同时调用 `all_reduce`，这是同步操作
2. **NCCL 优化**：使用 Ring AllReduce 算法，通信复杂度为 O(N)，带宽利用率接近最优
3. **内存效率**：原地操作，不需要额外内存分配
4. **错误处理**：如果任何 rank 失败，整个操作会超时并报错

### 性能特征

- **延迟**：10-100 μs（取决于张量大小）
- **带宽利用率**：90%+（NCCL 优化）
- **扩展性**：随 TP size 线性扩展

---

## 场景 2：Tensor 并行 AllGather 通信

### 业务场景
列并行层（如 QKV Projection、FFN Up Projection）需要收集所有 TP rank 的分片输出。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用层
    participant Rank0 as TP Rank 0<br/>(分片 0)
    participant Rank1 as TP Rank 1<br/>(分片 1)
    participant Rank2 as TP Rank 2<br/>(分片 2)
    participant Rank3 as TP Rank 3<br/>(分片 3)
    participant NCCL as NCCL 集合通信
    
    Note over App,NCCL: 列并行层计算完成，每个 TP rank 有不同的分片
    
    rect rgb(255, 245, 235)
        Note over App,NCCL: 1. 准备阶段
        App->>Rank0: tensor_model_parallel_all_gather(shard_0, dim=-1)
        App->>Rank1: tensor_model_parallel_all_gather(shard_1, dim=-1)
        App->>Rank2: tensor_model_parallel_all_gather(shard_2, dim=-1)
        App->>Rank3: tensor_model_parallel_all_gather(shard_3, dim=-1)
        
        Note over Rank0,Rank3: 每个 rank 的分片形状：[batch, seq_len, hidden_size/4]
    end
    
    rect rgb(235, 245, 255)
        Note over Rank0,NCCL: 2. AllGather 执行阶段
        Rank0->>NCCL: all_gather([shard_0], group=tp_group)
        Rank1->>NCCL: all_gather([shard_1], group=tp_group)
        Rank2->>NCCL: all_gather([shard_2], group=tp_group)
        Rank3->>NCCL: all_gather([shard_3], group=tp_group)
        
        rect rgb(230, 255, 230)
            Note over NCCL: Ring AllGather 算法
            NCCL->>NCCL: Phase 1: 初始化收集列表
            Note over NCCL: gather_list = [empty] * world_size
            NCCL->>NCCL: Phase 2: 执行 all_gather
            Note over NCCL: 每个 rank 贡献自己的分片
            NCCL->>NCCL: Phase 3: 拼接结果
            Note over NCCL: torch.cat([shard_0, shard_1, shard_2, shard_3], dim=-1)
        end
        
        NCCL-->>Rank0: [shard_0, shard_1, shard_2, shard_3] (拼接后)
        NCCL-->>Rank1: [shard_0, shard_1, shard_2, shard_3] (相同)
        NCCL-->>Rank2: [shard_0, shard_1, shard_2, shard_3] (相同)
        NCCL-->>Rank3: [shard_0, shard_1, shard_2, shard_3] (相同)
    end
    
    rect rgb(245, 255, 235)
        Note over App,NCCL: 3. 完成阶段
        Rank0-->>App: complete_tensor [batch, seq_len, hidden_size]
        Rank1-->>App: complete_tensor (相同)
        Rank2-->>App: complete_tensor (相同)
        Rank3-->>App: complete_tensor (相同)
        
        Note over App: 所有 TP rank 现在有完整的张量
        Note over App: 可以继续计算（如 Attention）
    end
```

### QKV Projection 的具体示例

```mermaid
graph TB
    subgraph "输入阶段"
        Input["Hidden States<br/>[batch, seq_len, 4096]"]
    end
    
    subgraph "列并行计算"
        QKV0["TP Rank 0<br/>Q[:,:,:1024], K[:,:,:1024], V[:,:,:1024]"]
        QKV1["TP Rank 1<br/>Q[:,:,1024:2048], K[:,:,1024:2048], V[:,:,1024:2048]"]
        QKV2["TP Rank 2<br/>Q[:,:,2048:3072], K[:,:,2048:3072], V[:,:,2048:3072]"]
        QKV3["TP Rank 3<br/>Q[:,:,3072:4096], K[:,:,3072:4096], V[:,:,3072:4096]"]
    end
    
    subgraph "AllGather 收集"
        Gather["All2All Gather<br/>dim=-1 拼接"]
    end
    
    subgraph "输出阶段"
        Output["Complete QKV<br/>[batch, seq_len, 4096*3]"]
    end
    
    Input --> QKV0
    Input --> QKV1
    Input --> QKV2
    Input --> QKV3
    
    QKV0 --> Gather
    QKV1 --> Gather
    QKV2 --> Gather
    QKV3 --> Gather
    
    Gather --> Output
```

---

## 场景 3：Expert 并行 All2All 通信

### 业务场景
MoE（Mixture of Experts）模型中，token 需要根据路由决策分发到不同的 expert 进行计算。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant Router as 路由层
    participant Rank0 as EP Rank 0<br/>(Expert 0,1)
    participant Rank1 as EP Rank 1<br/>(Expert 2,3)
    participant Rank2 as EP Rank 2<br/>(Expert 4,5)
    participant Rank3 as EP Rank 3<br/>(Expert 6,7)
    participant All2All as All2All 通信
    
    Note over Router,All2All: 输入：每个 rank 有所有 token，需要分发到对应 expert
    
    rect rgb(255, 245, 235)
        Note over Router,All2All: 1. 路由决策阶段
        Router->>Rank0: router_logits → expert_ids, routing_weights
        Router->>Rank1: router_logits → expert_ids, routing_weights
        Router->>Rank2: router_logits → expert_ids, routing_weights
        Router->>Rank3: router_logits → expert_ids, routing_weights
        
        Note over Rank0: token_0 → Expert 2, token_1 → Expert 0, ...
        Note over Rank1: token_0 → Expert 2, token_1 → Expert 0, ...
        Note over Rank2: token_0 → Expert 2, token_1 → Expert 0, ...
        Note over Rank3: token_0 → Expert 2, token_1 → Expert 0, ...
    end
    
    rect rgb(235, 245, 255)
        Note over Rank0,All2All: 2. All2All Scatter 阶段
        Rank0->>All2All: scatter(tokens_for_experts, expert_capacity)
        Rank1->>All2All: scatter(tokens_for_experts, expert_capacity)
        Rank2->>All2All: scatter(tokens_for_experts, expert_capacity)
        Rank3->>All2All: scatter(tokens_for_experts, expert_capacity)
        
        rect rgb(230, 255, 230)
            Note over All2All: All2All Scatter 通信
            All2All->>All2All: Rank0 → Rank1: Expert 2,3 的 tokens
            All2All->>All2All: Rank0 → Rank2: Expert 4,5 的 tokens
            All2All->>All2All: Rank1 → Rank0: Expert 0,1 的 tokens
            All2All->>All2All: ... (所有 rank 间的 token 交换)
        end
        
        All2All-->>Rank0: Expert 0,1 分配的所有 tokens
        All2All-->>Rank1: Expert 2,3 分配的所有 tokens
        All2All-->>Rank2: Expert 4,5 分配的所有 tokens
        All2All-->>Rank3: Expert 6,7 分配的所有 tokens
    end
    
    rect rgb(245, 255, 235)
        Note over Rank0,All2All: 3. Expert 计算阶段
        Rank0->>Rank0: Expert 0 FFN(assigned_tokens)
        Rank0->>Rank0: Expert 1 FFN(assigned_tokens)
        Rank1->>Rank1: Expert 2 FFN(assigned_tokens)
        Rank1->>Rank1: Expert 3 FFN(assigned_tokens)
        Rank2->>Rank2: Expert 4 FFN(assigned_tokens)
        Rank2->>Rank2: Expert 5 FFN(assigned_tokens)
        Rank3->>Rank3: Expert 6 FFN(assigned_tokens)
        Rank3->>Rank3: Expert 7 FFN(assigned_tokens)
        
        Note over Rank0,Rank3: 并行计算，每个 Expert 处理分配给它的 tokens
    end
    
    rect rgb(255, 235, 245)
        Note over Rank0,All2All: 4. All2All Combine 阶段
        Rank0->>All2All: combine(expert_outputs)
        Rank1->>All2All: combine(expert_outputs)
        Rank2->>All2All: combine(expert_outputs)
        Rank3->>All2All: combine(expert_outputs)
        
        rect rgb(230, 255, 230)
            Note over All2All: All2All Combine 通信
            All2All->>All2All: 将 expert 输出发送回原始 rank
            All2All->>All2All: Rank1 → Rank0: Expert 2,3 的输出
            All2All->>All2All: Rank2 → Rank0: Expert 4,5 的输出
            All2All->>All2All: ... (reverse communication)
        end
        
        All2All-->>Rank0: 所有 expert 对该 rank tokens 的输出
        All2All-->>Rank1: 所有 expert 对该 rank tokens 的输出
        All2All-->>Rank2: 所有 expert 对该 rank tokens 的输出
        All2All-->>Rank3: 所有 expert 对该 rank tokens 的输出
    end
    
    rect rgb(235, 255, 245)
        Note over Router,All2All: 5. 输出合并阶段
        Rank0->>Rank0: 按 routing_weights 加权合并
        Rank1->>Rank1: 按 routing_weights 加权合并
        Rank2->>Rank2: 按 routing_weights 加权合并
        Rank3->>Rank3: 按 routing_weights 加权合并
        
        Rank0-->>Router: final_output (所有 tokens 的最终结果)
        Rank1-->>Router: final_output
        Rank2-->>Router: final_output
        Rank3-->>Router: final_output
    end
```

### Expert 路由决策详解

```mermaid
graph TB
    subgraph "路由决策"
        Input["Input Tokens<br/>[batch*seq_len, hidden_size]"]
        Gate["Router/Gate<br/>Linear(hidden_size, num_experts)"]
        Softmax["Softmax + TopK"]
        ExpertIds["Expert IDs<br/>[batch*seq_len, topk]"]
        Weights["Routing Weights<br/>[batch*seq_len, topk]"]
    end
    
    subgraph "Token 分发映射"
        Map["Token → Expert 映射"]
        Rank0Map["Rank 0: Expert 0,1 的 tokens"]
        Rank1Map["Rank 1: Expert 2,3 的 tokens"]
        Rank2Map["Rank 2: Expert 4,5 的 tokens"]
        Rank3Map["Rank 3: Expert 6,7 的 tokens"]
    end
    
    Input --> Gate
    Gate --> Softmax
    Softmax --> ExpertIds
    Softmax --> Weights
    
    ExpertIds --> Map
    Weights --> Map
    
    Map --> Rank0Map
    Map --> Rank1Map
    Map --> Rank2Map
    Map --> Rank3Map
```

---

## 场景 4：分离式 Prefill KV Cache 传输

### 业务场景
在分离式架构中，Prefill Worker 处理长 prompt 并生成 KV cache，需要传输给 Decode Worker 进行后续 token 生成。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant PrefillWorker as Prefill Worker
    participant KVPipe as KV Pipe<br/>(共享内存/网络)
    participant KVBuffer as KV Lookup Buffer
    participant DecodeWorker as Decode Worker
    
    Note over Client,DecodeWorker: 场景：长 prompt 处理 + 流式生成
    
    rect rgb(255, 245, 235)
        Note over Client,DecodeWorker: 1. 请求处理阶段
        Client->>PrefillWorker: 长 prompt 请求 (2048 tokens)
        Client->>DecodeWorker: 同步发送请求 ID
        
        PrefillWorker->>PrefillWorker: 检查 prompt 长度 > threshold
        DecodeWorker->>DecodeWorker: 等待 KV cache 传输
    end
    
    rect rgb(235, 245, 255)
        Note over PrefillWorker,KVBuffer: 2. Prefill 计算阶段
        PrefillWorker->>PrefillWorker: 处理长 prompt (2048 tokens)
        PrefillWorker->>PrefillWorker: 逐层计算 Attention
        PrefillWorker->>PrefillWorker: 生成 KV cache
        Note over PrefillWorker: KV cache shape: [num_layers, batch, num_heads, seq_len, head_dim]
        PrefillWorker->>PrefillWorker: 计算最后一个 token 的 hidden states
    end
    
    rect rgb(245, 255, 235)
        Note over PrefillWorker,KVBuffer: 3. KV Cache 传输阶段
        PrefillWorker->>KVPipe: send_tensor(kv_cache) # 每层单独发送
        loop 每一层
            PrefillWorker->>KVPipe: send_tensor(layer_i_kv_cache)
            KVPipe->>KVBuffer: insert(request_id, layer_i, kv_cache)
        end
        
        PrefillWorker->>KVPipe: send_tensor(hidden_states) # 最后的隐状态
        KVPipe->>KVBuffer: insert(request_id, "hidden", hidden_states)
        
        PrefillWorker->>KVPipe: send_tensor(None) # 传输完成标记
        KVPipe->>KVBuffer: mark_complete(request_id)
    end
    
    rect rgb(255, 235, 245)
        Note over KVBuffer,DecodeWorker: 4. KV Cache 接收阶段
        DecodeWorker->>KVBuffer: query(request_id) # 轮询检查
        KVBuffer-->>DecodeWorker: status: "transferring"
        
        Note over DecodeWorker: 等待传输完成...
        
        DecodeWorker->>KVBuffer: query(request_id)
        KVBuffer-->>DecodeWorker: status: "complete"
        
        loop 每一层
            DecodeWorker->>KVBuffer: drop_select(request_id, layer_i)
            KVBuffer-->>DecodeWorker: layer_i_kv_cache
            DecodeWorker->>DecodeWorker: load_kv_cache(layer_i, kv_cache)
        end
        
        DecodeWorker->>KVBuffer: drop_select(request_id, "hidden")
        KVBuffer-->>DecodeWorker: hidden_states
    end
    
    rect rgb(235, 255, 245)
        Note over DecodeWorker,Client: 5. Decode 生成阶段
        DecodeWorker->>DecodeWorker: 使用接收的 KV cache
        DecodeWorker->>DecodeWorker: forward_decode(next_token_id)
        DecodeWorker->>DecodeWorker: 更新 KV cache (append new token)
        DecodeWorker->>DecodeWorker: 生成下一个 token
        
        DecodeWorker-->>Client: 流式输出 token_1
        
        loop 继续生成
            DecodeWorker->>DecodeWorker: forward_decode(token_i)
            DecodeWorker-->>Client: 流式输出 token_i
        end
        
        DecodeWorker-->>Client: [EOS] 生成完成
    end
```

### KV Cache 传输详细流程

```mermaid
graph TB
    subgraph "Prefill Worker"
        P1["长 Prompt 输入<br/>[1, 2048, 4096]"]
        P2["逐层 Attention"]
        P3["生成 KV Cache<br/>[layers, 1, heads, 2048, head_dim]"]
        P4["序列化 KV Cache"]
        P5["发送到 KV Pipe"]
    end
    
    subgraph "KV Transfer Infrastructure"
        Pipe["KV Pipe<br/>(共享内存/TCP)"]
        Buffer["KV Lookup Buffer<br/>Key: request_id<br/>Value: kv_data"]
    end
    
    subgraph "Decode Worker"
        D1["接收 KV Cache"]
        D2["反序列化并加载"]
        D3["初始化 Attention 状态"]
        D4["生成第一个新 token"]
        D5["更新 KV Cache"]
        D6["流式输出"]
    end
    
    P1 --> P2 --> P3 --> P4 --> P5
    P5 --> Pipe --> Buffer
    Buffer --> D1 --> D2 --> D3 --> D4 --> D5 --> D6
```

---

## 场景 5：分布式环境初始化

### 业务场景
vLLM 启动时需要初始化分布式环境，建立各种并行组。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant Master as Master Process
    participant Rank0 as Worker Rank 0
    participant Rank1 as Worker Rank 1
    participant Rank2 as Worker Rank 2
    participant Rank3 as Worker Rank 3
    participant ProcessGroupMgr as ProcessGroup Manager
    
    Note over Master,ProcessGroupMgr: 场景：8 GPU，TP=2, PP=2, DP=2
    
    rect rgb(255, 245, 235)
        Note over Master,ProcessGroupMgr: 1. 全局初始化阶段
        Master->>Master: 解析配置 (world_size=8, tp=2, pp=2, dp=2)
        Master->>Master: 计算进程组拓扑
        
        Master->>Rank0: spawn_worker(rank=0, world_size=8)
        Master->>Rank1: spawn_worker(rank=1, world_size=8)
        Master->>Rank2: spawn_worker(rank=2, world_size=8)
        Master->>Rank3: spawn_worker(rank=3, world_size=8)
        Note over Master: ... (启动所有 8 个 worker)
        
        Rank0->>ProcessGroupMgr: torch.distributed.init_process_group()
        Rank1->>ProcessGroupMgr: torch.distributed.init_process_group()
        Rank2->>ProcessGroupMgr: torch.distributed.init_process_group()
        Rank3->>ProcessGroupMgr: torch.distributed.init_process_group()
        
        ProcessGroupMgr-->>Rank0: 全局通信组建立
        ProcessGroupMgr-->>Rank1: 全局通信组建立
        ProcessGroupMgr-->>Rank2: 全局通信组建立
        ProcessGroupMgr-->>Rank3: 全局通信组建立
    end
    
    rect rgb(235, 245, 255)
        Note over Rank0,ProcessGroupMgr: 2. Tensor Parallel 组创建
        Rank0->>ProcessGroupMgr: new_group([0, 1]) # TP group 0
        Rank1->>ProcessGroupMgr: new_group([0, 1]) # TP group 0
        Rank2->>ProcessGroupMgr: new_group([2, 3]) # TP group 1
        Rank3->>ProcessGroupMgr: new_group([2, 3]) # TP group 1
        
        ProcessGroupMgr-->>Rank0: tp_group = group([0,1])
        ProcessGroupMgr-->>Rank1: tp_group = group([0,1])
        ProcessGroupMgr-->>Rank2: tp_group = group([2,3])
        ProcessGroupMgr-->>Rank3: tp_group = group([2,3])
        
        Note over Rank0,Rank3: 每个 rank 知道自己的 TP neighbors
    end
    
    rect rgb(245, 255, 235)
        Note over Rank0,ProcessGroupMgr: 3. Pipeline Parallel 组创建
        Rank0->>ProcessGroupMgr: new_group([0, 2]) # PP group 0
        Rank1->>ProcessGroupMgr: new_group([1, 3]) # PP group 1
        Rank2->>ProcessGroupMgr: new_group([0, 2]) # PP group 0
        Rank3->>ProcessGroupMgr: new_group([1, 3]) # PP group 1
        
        ProcessGroupMgr-->>Rank0: pp_group = group([0,2])
        ProcessGroupMgr-->>Rank1: pp_group = group([1,3])
        ProcessGroupMgr-->>Rank2: pp_group = group([0,2])
        ProcessGroupMgr-->>Rank3: pp_group = group([1,3])
        
        Note over Rank0,Rank3: Pipeline 连接：Rank0→Rank2, Rank1→Rank3
    end
    
    rect rgb(255, 235, 245)
        Note over Rank0,ProcessGroupMgr: 4. Data Parallel 组创建
        Rank0->>ProcessGroupMgr: new_group([0, 4]) # DP group 0
        Rank1->>ProcessGroupMgr: new_group([1, 5]) # DP group 1
        Rank2->>ProcessGroupMgr: new_group([2, 6]) # DP group 2
        Rank3->>ProcessGroupMgr: new_group([3, 7]) # DP group 3
        
        ProcessGroupMgr-->>Rank0: dp_group = group([0,4])
        ProcessGroupMgr-->>Rank1: dp_group = group([1,5])
        ProcessGroupMgr-->>Rank2: dp_group = group([2,6])
        ProcessGroupMgr-->>Rank3: dp_group = group([3,7])
    end
    
    rect rgb(235, 255, 245)
        Note over Rank0,ProcessGroupMgr: 5. 设备通信器初始化
        Rank0->>Rank0: create_device_communicator(cuda:0, tp_group)
        Rank1->>Rank1: create_device_communicator(cuda:1, tp_group)
        Rank2->>Rank2: create_device_communicator(cuda:2, tp_group)
        Rank3->>Rank3: create_device_communicator(cuda:3, tp_group)
        
        Note over Rank0,Rank3: 初始化 NCCL 后端，建立 GPU 直连
        
        Rank0->>Rank1: NCCL handshake (TP group)
        Rank2->>Rank3: NCCL handshake (TP group)
        
        Note over Rank0,Rank1: TP group 0 ready
        Note over Rank2,Rank3: TP group 1 ready
    end
    
    rect rgb(245, 235, 255)
        Note over Master,ProcessGroupMgr: 6. 初始化验证
        Rank0->>Rank1: test_all_reduce(dummy_tensor)
        Rank1-->>Rank0: success
        
        Rank2->>Rank3: test_all_reduce(dummy_tensor)
        Rank3-->>Rank2: success
        
        Rank0-->>Master: initialization_complete
        Rank1-->>Master: initialization_complete
        Rank2-->>Master: initialization_complete
        Rank3-->>Master: initialization_complete
        
        Master->>Master: 所有 worker 就绪，开始模型加载
    end
```

### 进程组拓扑图

```mermaid
graph TB
    subgraph "8 GPU 拓扑 (TP=2, PP=2, DP=2)"
        subgraph "TP Groups"
            TP0["TP Group 0<br/>Rank 0,1"]
            TP1["TP Group 1<br/>Rank 2,3"]
            TP2["TP Group 2<br/>Rank 4,5"]
            TP3["TP Group 3<br/>Rank 6,7"]
        end
        
        subgraph "PP Groups"
            PP0["PP Group 0<br/>Rank 0,2,4,6"]
            PP1["PP Group 1<br/>Rank 1,3,5,7"]
        end
        
        subgraph "DP Groups"
            DP0["DP Group 0<br/>Rank 0,4"]
            DP1["DP Group 1<br/>Rank 1,5"]
            DP2["DP Group 2<br/>Rank 2,6"]
            DP3["DP Group 3<br/>Rank 3,7"]
        end
    end
    
    subgraph "通信模式"
        AllReduce["AllReduce<br/>(TP Groups)"]
        SendRecv["Send/Recv<br/>(PP Groups)"]
        AllGather["AllGather<br/>(DP Groups)"]
    end
    
    TP0 --> AllReduce
    TP1 --> AllReduce
    PP0 --> SendRecv
    PP1 --> SendRecv
    DP0 --> AllGather
    DP1 --> AllGather
```

---

## 场景 6：Pipeline 并行通信

### 业务场景
Pipeline 并行中，不同 stage 之间需要传递中间激活值。

### 时序图

```mermaid
sequenceDiagram
    autonumber
    participant Input as 输入数据
    participant Stage0 as PP Stage 0<br/>(Rank 0,1)
    participant Stage1 as PP Stage 1<br/>(Rank 2,3)
    participant Stage2 as PP Stage 2<br/>(Rank 4,5)
    participant Stage3 as PP Stage 3<br/>(Rank 6,7)
    participant Output as 输出结果
    
    Note over Input,Output: Pipeline 并行：4 stage，每个 stage 2 个 TP rank
    
    rect rgb(255, 245, 235)
        Note over Input,Output: 微批次 1 前向传播
        Input->>Stage0: microbatch_1 [batch, seq_len, hidden]
        
        Stage0->>Stage0: 内部 TP 计算 (AllReduce/AllGather)
        Stage0->>Stage1: send(activations_1)
        
        Stage1->>Stage1: 内部 TP 计算
        Stage1->>Stage2: send(activations_1)
        
        Stage2->>Stage2: 内部 TP 计算
        Stage2->>Stage3: send(activations_1)
        
        Stage3->>Stage3: 内部 TP 计算
        Stage3-->>Output: output_1
    end
    
    rect rgb(235, 245, 255)
        Note over Input,Output: 微批次 2 前向传播 (流水线并行)
        Input->>Stage0: microbatch_2
        
        Stage0->>Stage0: 计算 microbatch_2
        Stage1->>Stage1: 计算 microbatch_1 (接收自 Stage0)
        
        Stage0->>Stage1: send(activations_2)
        Stage1->>Stage2: send(activations_1)
        
        Stage1->>Stage1: 计算 microbatch_2
        Stage2->>Stage2: 计算 microbatch_1
        
        Stage1->>Stage2: send(activations_2)
        Stage2->>Stage3: send(activations_1)
        
        Stage2->>Stage2: 计算 microbatch_2
        Stage3->>Stage3: 计算 microbatch_1
        Stage3-->>Output: output_1
        
        Stage2->>Stage3: send(activations_2)
        Stage3->>Stage3: 计算 microbatch_2
        Stage3-->>Output: output_2
    end
    
    rect rgb(245, 255, 235)
        Note over Input,Output: 反向传播阶段
        Output->>Stage3: grad_output_2
        Stage3->>Stage3: 反向计算 microbatch_2
        Stage3->>Stage2: send(grad_activations_2)
        
        Stage2->>Stage2: 反向计算 microbatch_2
        Stage2->>Stage1: send(grad_activations_2)
        
        Stage1->>Stage1: 反向计算 microbatch_2
        Stage1->>Stage0: send(grad_activations_2)
        
        Stage0->>Stage0: 反向计算 microbatch_2
        
        Note over Stage0,Stage3: 同时处理多个微批次的反向传播
    end
```

### Pipeline 调度详解

```mermaid
gantt
    title Pipeline 并行调度时序
    dateFormat X
    axisFormat %L
    
    section Stage 0
    MB1 Forward    :active, mb1f0, 0, 100
    MB2 Forward    :active, mb2f0, 100, 200
    MB3 Forward    :active, mb3f0, 200, 300
    MB1 Backward   :crit, mb1b0, 600, 700
    MB2 Backward   :crit, mb2b0, 700, 800
    
    section Stage 1
    MB1 Forward    :active, mb1f1, 100, 200
    MB2 Forward    :active, mb2f1, 200, 300
    MB3 Forward    :active, mb3f1, 300, 400
    MB1 Backward   :crit, mb1b1, 500, 600
    MB2 Backward   :crit, mb2b1, 600, 700
    
    section Stage 2
    MB1 Forward    :active, mb1f2, 200, 300
    MB2 Forward    :active, mb2f2, 300, 400
    MB3 Forward    :active, mb3f2, 400, 500
    MB1 Backward   :crit, mb1b2, 400, 500
    MB2 Backward   :crit, mb2b2, 500, 600
    
    section Stage 3
    MB1 Forward    :active, mb1f3, 300, 400
    MB2 Forward    :active, mb2f3, 400, 500
    MB3 Forward    :active, mb3f3, 500, 600
    MB1 Backward   :crit, mb1b3, 300, 400
    MB2 Backward   :crit, mb2b3, 400, 500
```

---

## 性能优化时序分析

### 通信与计算重叠

```mermaid
gantt
    title 通信计算重叠优化
    dateFormat X
    axisFormat %L
    
    section GPU 计算
    Layer 1 Compute    :active, l1c, 0, 100
    Layer 2 Compute    :active, l2c, 150, 250
    Layer 3 Compute    :active, l3c, 300, 400
    Layer 4 Compute    :active, l4c, 450, 550
    
    section TP 通信
    Layer 1 AllReduce  :crit, l1ar, 80, 130
    Layer 2 AllGather  :crit, l2ag, 230, 280
    Layer 3 AllReduce  :crit, l3ar, 380, 430
    Layer 4 AllGather  :crit, l4ag, 530, 580
    
    section 优化效果
    重叠计算          :done, overlap, 100, 500
    总体加速          :milestone, speedup, 580
```

---

## 错误处理时序

### 通信超时和恢复

```mermaid
sequenceDiagram
    autonumber
    participant Rank0 as Rank 0
    participant Rank1 as Rank 1 (故障)
    participant Rank2 as Rank 2
    participant Rank3 as Rank 3
    participant Monitor as 故障监控
    
    rect rgb(255, 235, 235)
        Note over Rank0,Monitor: 正常通信阶段
        Rank0->>Rank1: all_reduce(tensor) 
        Rank0->>Rank2: all_reduce(tensor)
        Rank0->>Rank3: all_reduce(tensor)
        
        Note over Rank1: GPU 故障！
        Rank1->>Rank1: 进程崩溃
        
        Note over Rank0,Rank3: 等待 all_reduce 完成...
        Note over Rank0,Rank3: 超时！(30s)
    end
    
    rect rgb(255, 245, 235)
        Note over Rank0,Monitor: 故障检测阶段
        Rank0->>Monitor: 报告通信超时
        Rank2->>Monitor: 报告通信超时
        Rank3->>Monitor: 报告通信超时
        
        Monitor->>Monitor: 检测到 Rank 1 故障
        Monitor->>Monitor: 决定恢复策略
    end
    
    rect rgb(235, 245, 255)
        Note over Rank0,Monitor: 故障恢复阶段
        Monitor->>Rank0: 重启 Rank 1 进程
        Monitor->>Rank2: 重建通信组
        Monitor->>Rank3: 重建通信组
        
        Note over Rank1: 新进程启动
        Rank1->>Rank0: 重新加入 TP group
        Rank1->>Rank2: 重新加入 TP group
        Rank1->>Rank3: 重新加入 TP group
        
        Note over Rank0,Rank3: 从检查点恢复状态
        Note over Rank0,Rank3: 继续训练/推理
    end
```

---

## 总结

Distributed 模块的时序图展示了：

1. **基础通信原语**：AllReduce、AllGather、All2All 的详细执行流程
2. **复杂场景**：Expert 并行、分离式 Prefill、Pipeline 并行的多方协调
3. **系统初始化**：分布式环境建立的完整过程
4. **性能优化**：通信计算重叠、故障恢复机制

**关键设计要点**：
- **同步协调**：确保所有参与方在正确时机执行通信
- **错误处理**：超时检测、故障恢复、状态一致性
- **性能优化**：重叠执行、缓冲区管理、算法选择
- **可扩展性**：支持不同规模和拓扑的分布式部署

通过这些时序图，可以深入理解 vLLM 分布式系统的运行机制和优化策略。
