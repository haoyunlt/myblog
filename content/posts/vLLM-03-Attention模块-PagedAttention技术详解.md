---
title: "vLLM-03-Attention模块-PagedAttention技术详解"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - vLLM-03-Attention模块-PagedAttention技术详解"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# vLLM-03-Attention模块-PagedAttention技术详解

## PagedAttention 核心原理

### 传统 Attention 的内存问题

在传统的 Transformer Attention 实现中，KV 缓存采用连续内存分配：

```python
# 传统方式：为每个序列预分配固定大小的 KV 缓存
kv_cache_shape = (batch_size, max_seq_len, num_heads, head_dim)
k_cache = torch.zeros(kv_cache_shape, device="cuda")  # Key 缓存
v_cache = torch.zeros(kv_cache_shape, device="cuda")  # Value 缓存
```

**问题**：
1. **内存浪费**：必须按 `max_seq_len` 预分配，实际使用可能远小于此
2. **无法共享**：不同请求即使有相同 prompt 也无法共享内存
3. **内存碎片**：不同长度的序列难以高效打包

**示例**：
- 模型：Llama-2-7B（32 层，32 heads，128 head_dim）
- 最大长度：4096 tokens
- 数据类型：FP16
- 单序列 KV 缓存：`2 × 4096 × 32 × 32 × 128 × 2 bytes = 1GB`
- 实际使用（100 tokens）：`2 × 100 × 32 × 32 × 128 × 2 bytes = 25MB`
- **浪费率：97.5%**

### PagedAttention 解决方案

PagedAttention 借鉴操作系统虚拟内存的分页机制：

```python
# PagedAttention 方式：按块（block）分配
block_size = 16  # 每个块存储 16 个 token 的 KV
num_blocks = 1000  # 物理块池

# KV 缓存按块组织
kv_cache_shape = (num_blocks, block_size, num_heads, head_dim)
k_cache = torch.zeros(kv_cache_shape, device="cuda")
v_cache = torch.zeros(kv_cache_shape, device="cuda")

# 块表：逻辑块 → 物理块的映射
block_tables = {
    "request_1": [0, 5, 12, ...],  # 请求 1 使用块 0, 5, 12, ...
    "request_2": [1, 5, 8, ...],   # 请求 2 使用块 1, 5, 8, ...（块 5 共享）
}
```

**优势**：
1. **按需分配**：只分配实际需要的块
2. **内存共享**：多个请求可共享相同的块（Prefix Caching）
3. **灵活增长**：decode 阶段动态增加块
4. **高利用率**：接近 100% 内存利用率

## 架构与实现

### 1. 块（Block）设计

```python
class KVCacheBlock:
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id        # 物理块 ID
        self.ref_cnt = 0                # 引用计数
        self.block_hash = None          # 块哈希（用于 Prefix Caching）
        self.is_null = False            # 是否为空块
        
        # 链表指针（用于 LRU）
        self.prev = None
        self.next = None
```

**块大小选择**：
- 太小（如 4）：管理开销大，映射表占用内存多
- 太大（如 64）：内存碎片增加，浪费率高
- **推荐：16**，在开销和碎片间取得平衡

### 2. 块表（Block Table）

块表维护逻辑地址到物理地址的映射：

```python
# 逻辑地址：token 位置
token_position = 35

# 物理地址：块 ID + 块内偏移
block_index = token_position // block_size  # 35 // 16 = 2（第 2 个逻辑块）
block_offset = token_position % block_size   # 35 % 16 = 3（块内第 3 个位置）
physical_block_id = block_table[block_index]  # 查询块表
```

**块表结构**：
```python
# 二维张量：(batch_size, max_num_blocks_per_seq)
block_tables = torch.tensor([
    [10, 25, 37, -1, -1, ...],  # 请求 0：使用物理块 10, 25, 37
    [12, 25, 41, 53, -1, ...],  # 请求 1：使用物理块 12, 25, 41, 53
    ...
], device="cuda")
```

### 3. 前向传播流程

#### 3.1 Prefill 阶段（处理 prompt）

```python
def write_to_paged_cache(
    key: torch.Tensor,      # [num_tokens, num_heads, head_dim]
    value: torch.Tensor,    # [num_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_dim]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens] 每个 token 的物理槽位
):
    # 核心操作：将 KV 写入对应的物理槽位
    for i, slot_idx in enumerate(slot_mapping):
        block_id = slot_idx // block_size
        offset = slot_idx % block_size
        key_cache[block_id, offset] = key[i]
        value_cache[block_id, offset] = value[i]
```

**slot_mapping 计算**：
```python
# 假设请求使用块 [10, 25]，当前处理 35 个 token
block_table = [10, 25]
slot_mapping = []
for token_pos in range(35):
    block_idx = token_pos // 16
    offset = token_pos % 16
    physical_block = block_table[block_idx]
    slot_mapping.append(physical_block * 16 + offset)

# slot_mapping = [160, 161, ..., 175,  # 块 10
#                 400, 401, ..., 402]   # 块 25
```

#### 3.2 Decode 阶段（生成 token）

```python
def paged_attention_decode(
    query: torch.Tensor,       # [num_seqs, num_heads, head_dim]
    key_cache: torch.Tensor,   # [num_blocks, block_size, num_heads, head_dim]
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks]
    seq_lens: torch.Tensor,      # [num_seqs]
) -> torch.Tensor:             # [num_seqs, num_heads, head_dim]
    
    num_seqs, num_heads, head_dim = query.shape
    output = torch.empty_like(query)
    
    # 对每个序列并行计算
    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx]
        block_table = block_tables[seq_idx]
        
        # 计算注意力
        attn_scores = []
        for token_pos in range(seq_len):
            block_idx = token_pos // block_size
            offset = token_pos % block_size
            physical_block = block_table[block_idx]
            
            # 获取 Key
            key = key_cache[physical_block, offset]
            
            # 计算注意力分数
            score = torch.dot(query[seq_idx], key) / sqrt(head_dim)
            attn_scores.append(score)
        
        # Softmax
        attn_weights = torch.softmax(torch.stack(attn_scores), dim=0)
        
        # 加权求和 Value
        attn_output = 0
        for token_pos in range(seq_len):
            block_idx = token_pos // block_size
            offset = token_pos % block_size
            physical_block = block_table[block_idx]
            value = value_cache[physical_block, offset]
            attn_output += attn_weights[token_pos] * value
        
        output[seq_idx] = attn_output
    
    return output
```

### 4. CUDA 内核优化

实际实现中，PagedAttention 使用高度优化的 CUDA 内核：

```cuda
// PagedAttention V1 内核（简化版）
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,  // [num_blocks, block_size, num_heads, head_size]
    const scalar_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const float scale
) {
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int seq_len = seq_lens[seq_idx];
    
    // 使用 shared memory 存储中间结果
    __shared__ float qk_max;
    __shared__ float exp_sum;
    
    // 1. 计算 QK^T
    float qk = 0.0f;
    for (int token_pos = threadIdx.x; token_pos < seq_len; token_pos += blockDim.x) {
        // 查找物理块
        int block_idx = token_pos / BLOCK_SIZE;
        int offset = token_pos % BLOCK_SIZE;
        int physical_block = block_tables[seq_idx * max_num_blocks + block_idx];
        
        // 加载 Key
        const scalar_t* key = k_cache + physical_block * block_size * num_heads * head_size
                              + offset * num_heads * head_size
                              + head_idx * head_size;
        
        // 点积
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < HEAD_SIZE; i++) {
            score += float(q[i]) * float(key[i]);
        }
        qk = score * scale;
    }
    
    // 2. Softmax（分两步：max + exp_sum）
    // （此处省略 reduction 逻辑）
    
    // 3. 加权求和 Value
    // （此处省略实现）
}
```

**优化技术**：
- **Shared Memory**：减少全局内存访问
- **Warp-level Reduction**：高效的并行归约
- **Vectorized Load**：批量加载数据
- **Register Tiling**：复用寄存器数据

### 5. Prefix Caching

Prefix Caching 是 PagedAttention 的重要应用：

```python
class PrefixCache:
    def __init__(self):
        # 哈希表：block_hash -> KVCacheBlock
        self.hash_to_block: dict[int, KVCacheBlock] = {}
        # LRU 队列
        self.lru_queue: deque[KVCacheBlock] = deque()
    
    def find_cached_blocks(self, token_ids: list[int]) -> list[KVCacheBlock]:
        """查找缓存的块"""
        cached_blocks = []
        for i in range(0, len(token_ids), block_size):
            # 计算块哈希
            chunk = token_ids[i:i+block_size]
            block_hash = hash(tuple(chunk))
            
            if block_hash in self.hash_to_block:
                block = self.hash_to_block[block_hash]
                block.ref_cnt += 1
                cached_blocks.append(block)
            else:
                break  # 未命中，停止查找
        
        return cached_blocks
    
    def cache_block(self, block: KVCacheBlock, token_ids: list[int]):
        """缓存块"""
        block_hash = hash(tuple(token_ids))
        self.hash_to_block[block_hash] = block
        self.lru_queue.append(block)
    
    def evict_lru(self) -> KVCacheBlock:
        """驱逐 LRU 块"""
        while self.lru_queue:
            block = self.lru_queue.popleft()
            if block.ref_cnt == 0:
                del self.hash_to_block[block.block_hash]
                return block
        return None
```

**缓存策略**：
- **Hash-based 查找**：O(1) 查找缓存块
- **LRU 驱逐**：优先驱逐最久未使用的块
- **引用计数**：防止驱逐正在使用的块

## 性能分析

### 1. 内存利用率

**传统方法**：
- 预分配：`batch_size × max_seq_len`
- 实际使用：`sum(actual_seq_lens)`
- 利用率：通常 20-40%

**PagedAttention**：
- 分配：`ceil(sum(actual_seq_lens) / block_size)` 个块
- 利用率：接近 100%（仅块内碎片 <6%）

**示例**：
- 批次：8 个请求，平均长度 500 tokens
- 最大长度：2048 tokens
- 块大小：16

传统：`8 × 2048 = 16384` tokens 空间
PagedAttention：`ceil(8 × 500 / 16) = 250` 块 = `4000` tokens 空间

**节省：75.6%**

### 2. 计算开销

**Prefill 阶段**：
- 传统：O(n²) attention
- PagedAttention：O(n²) + O(n) 写入开销
- 额外开销：<5%（写入操作被内存访问掩盖）

**Decode 阶段**：
- 传统：O(n) attention（n = 序列长度）
- PagedAttention：O(n) + O(n/block_size) 块查找
- 额外开销：<2%（块查找开销极小）

### 3. Prefix Caching 收益

**缓存命中时**：
- 跳过已缓存 token 的计算
- TTFT 降低：40-60%
- 吞吐提升：20-40%（取决于命中率）

**示例**：
- System prompt：500 tokens
- 缓存命中率：80%
- TTFT：从 800ms 降至 320ms
- 节省计算：500 / (500 + 100) ≈ 83%

## 代码示例

### 完整 PagedAttention 使用

```python
import torch
from vllm.attention.ops.paged_attn import PagedAttention

# 初始化参数
num_seqs = 4
num_heads = 32
head_dim = 128
block_size = 16
max_seq_len = 256
num_blocks = 100

# 创建 KV 缓存
kv_cache_shape = PagedAttention.get_kv_cache_shape(
    num_blocks, block_size, num_heads, head_dim)
key_cache = torch.zeros(kv_cache_shape, dtype=torch.float16, device="cuda")
value_cache = torch.zeros(kv_cache_shape, dtype=torch.float16, device="cuda")

# Prefill：写入 KV
key = torch.randn(100, num_heads, head_dim, device="cuda")
value = torch.randn(100, num_heads, head_dim, device="cuda")
slot_mapping = torch.arange(100, device="cuda")  # 前 100 个槽位

PagedAttention.write_to_paged_cache(
    key, value, key_cache, value_cache, slot_mapping,
    kv_cache_dtype="auto", k_scale=1.0, v_scale=1.0)

# Decode：计算注意力
query = torch.randn(num_seqs, num_heads, head_dim, device="cuda")
block_tables = torch.tensor([
    [0, 1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
], device="cuda")
seq_lens = torch.tensor([25, 30, 35, 40], device="cuda")

output = PagedAttention.forward_decode(
    query, key_cache, value_cache, block_tables, seq_lens,
    max_seq_len=max_seq_len, kv_cache_dtype="auto",
    num_kv_heads=num_heads, scale=1.0 / (head_dim ** 0.5),
    alibi_slopes=None, k_scale=1.0, v_scale=1.0)

print(output.shape)  # [4, 32, 128]
```

## 总结

PagedAttention 是 vLLM 的核心创新，通过借鉴操作系统虚拟内存的分页机制，实现了：

1. **高内存利用率**：接近 100%，相比传统方法提升 4-5 倍
2. **灵活内存管理**：动态分配，无需预知序列长度
3. **前缀共享**：跨请求共享相同 prompt 的 KV 缓存
4. **低计算开销**：额外开销 <5%

这些优化使得 vLLM 能够在相同硬件上支持更大的批次和更长的序列，显著提升推理吞吐量和降低成本。
