---
title: "vLLM-10-CSRC模块-核心内核"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 源码分析
categories:
  - vLLM
description: "源码剖析 - vLLM-10-CSRC模块-核心内核"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# vLLM-10-CSRC模块-核心内核

## 核心内核实现详解

CSRC 模块包含了 vLLM 的核心计算内核实现，本文档详细分析各类关键内核的设计原理、实现细节和性能优化策略。

## PagedAttention 内核详解

### 内核设计原理

PagedAttention 是 vLLM 的核心创新，通过分页内存管理实现高效的注意力计算。

```cpp
// PagedAttention V1 内核主函数
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,           // 输出 [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // 查询 [num_seqs, num_heads, head_size]  
    const scalar_t* __restrict__ k_cache, // Key缓存 [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache, // Value缓存 [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,               // KV头数
    const float scale,                    // 缩放因子
    const int* __restrict__ block_tables, // 块表 [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,     // 序列长度 [num_seqs]
    const int max_num_blocks_per_seq,     // 每序列最大块数
    const float* __restrict__ alibi_slopes, // ALiBi斜率
    const int q_stride,                   // Q张量步长
    const int kv_block_stride,            // KV块步长
    const int kv_head_stride              // KV头步长
) {
    // 1) 线程和块索引计算
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int num_heads = gridDim.y;
    const int thread_idx = threadIdx.x;
    
    // 2) 序列长度和块数检查
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;
    
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 3) 共享内存分配
    extern __shared__ char shared_mem[];
    scalar_t* q_vec = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* k_vec = q_vec + HEAD_SIZE;
    float* qk_max = reinterpret_cast<float*>(k_vec + HEAD_SIZE);
    float* exp_sums = qk_max + NUM_THREADS;
    
    // 4) 加载查询向量到共享内存
    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
    for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
        q_vec[i] = q_ptr[i];
    }
    __syncthreads();
    
    // 5) 计算注意力权重和输出
    float max_logit = -FLT_MAX;
    float exp_sum = 0.0f;
    
    // 第一遍：计算最大值和指数和
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block_number = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        const int block_offset = physical_block_number * kv_block_stride;
        
        // 计算当前块的注意力分数
        float block_max = compute_block_attention_scores(
            q_vec, k_cache + block_offset, head_idx, thread_idx,
            HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, scale, alibi_slopes);
        
        max_logit = fmaxf(max_logit, block_max);
    }
    
    // 使用 warp reduce 找到全局最大值
    max_logit = warp_reduce_max(max_logit);
    if (thread_idx == 0) {
        qk_max[0] = max_logit;
    }
    __syncthreads();
    max_logit = qk_max[0];
    
    // 第二遍：计算注意力权重和加权和
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block_number = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        
        // 计算注意力权重并累积到输出
        compute_attention_weights_and_values(
            q_vec, k_cache, v_cache, out, physical_block_number,
            seq_idx, head_idx, block_idx, thread_idx,
            HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, scale, max_logit,
            &exp_sum, alibi_slopes);
    }
    
    // 6) 归一化输出
    normalize_attention_output(out, seq_idx, head_idx, thread_idx, 
                              HEAD_SIZE, NUM_THREADS, exp_sum);
}
```

### 关键优化技术

#### 1. 分页内存布局优化

```cpp
// KV缓存的分页布局设计
template<typename T>
struct PagedKVCache {
    // Key缓存布局: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    // 其中 x 是向量化因子（如16字节对齐）
    T* key_cache;
    
    // Value缓存布局: [num_blocks, num_kv_heads, head_size, block_size]  
    T* value_cache;
    
    // 块表: 逻辑块到物理块的映射
    int* block_tables;
    
    // 内存访问优化函数
    __device__ __forceinline__ T* get_key_ptr(
        int physical_block, int head_idx, int token_idx, int dim_idx) {
        constexpr int X = 16 / sizeof(T);  // 向量化因子
        const int block_offset = physical_block * num_kv_heads * (head_size / X) * block_size * X;
        const int head_offset = head_idx * (head_size / X) * block_size * X;
        const int token_offset = token_idx * X;
        const int dim_offset = (dim_idx / X) * block_size * X + (dim_idx % X);
        
        return key_cache + block_offset + head_offset + dim_offset + token_offset;
    }
};
```

#### 2. 共享内存优化策略

```cpp
// 共享内存使用优化
template<int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__device__ void optimize_shared_memory_usage() {
    // 共享内存布局计算
    constexpr int q_vec_size = HEAD_SIZE * sizeof(scalar_t);
    constexpr int k_vec_size = HEAD_SIZE * sizeof(scalar_t);  
    constexpr int logits_size = BLOCK_SIZE * sizeof(float);
    constexpr int reduction_size = NUM_THREADS * sizeof(float);
    
    constexpr int total_shared_mem = q_vec_size + k_vec_size + logits_size + reduction_size;
    static_assert(total_shared_mem <= 48 * 1024, "Shared memory usage exceeds limit");
    
    // 内存银行冲突避免
    constexpr int BANK_SIZE = 32;  // 32个bank
    constexpr int bank_offset = (HEAD_SIZE % BANK_SIZE != 0) ? 1 : 0;
    
    // 添加填充避免银行冲突
    extern __shared__ char shared_mem[];
    scalar_t* q_vec = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* k_vec = q_vec + HEAD_SIZE + bank_offset;
    float* logits = reinterpret_cast<float*>(k_vec + HEAD_SIZE + bank_offset);
}
```

#### 3. 数值稳定的 Softmax 实现

```cpp
// 数值稳定的在线Softmax算法
template<int BLOCK_SIZE, int NUM_THREADS>
__device__ void stable_online_softmax(
    float* logits,           // 当前块的logits
    float* max_val,          // 全局最大值
    float* exp_sum,          // 指数和
    float* output,           // 累积输出
    const float* values,     // 当前块的values
    int valid_tokens         // 有效token数
) {
    const int thread_idx = threadIdx.x;
    
    // 1) 计算当前块的最大值
    float local_max = -FLT_MAX;
    for (int i = thread_idx; i < valid_tokens; i += NUM_THREADS) {
        local_max = fmaxf(local_max, logits[i]);
    }
    
    // Warp内reduction求最大值
    local_max = warp_reduce_max(local_max);
    
    // 2) 更新全局最大值和重新缩放
    float old_max = *max_val;
    float new_max = fmaxf(old_max, local_max);
    float scale_factor = expf(old_max - new_max);
    
    // 重新缩放之前的exp_sum和output
    *exp_sum *= scale_factor;
    for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
        output[i] *= scale_factor;
    }
    
    // 3) 计算当前块的贡献
    float block_exp_sum = 0.0f;
    for (int i = thread_idx; i < valid_tokens; i += NUM_THREADS) {
        float exp_val = expf(logits[i] - new_max);
        logits[i] = exp_val;  // 存储指数值用于后续计算
        block_exp_sum += exp_val;
    }
    
    // Warp内reduction求和
    block_exp_sum = warp_reduce_sum(block_exp_sum);
    
    // 4) 累积到输出
    *exp_sum += block_exp_sum;
    *max_val = new_max;
    
    // 加权累积values到output
    for (int head_dim = thread_idx; head_dim < HEAD_SIZE; head_dim += NUM_THREADS) {
        float weighted_sum = 0.0f;
        for (int token_idx = 0; token_idx < valid_tokens; ++token_idx) {
            weighted_sum += logits[token_idx] * values[token_idx * HEAD_SIZE + head_dim];
        }
        output[head_dim] += weighted_sum;
    }
    
    __syncthreads();
}
```

## FlashAttention 内核实现

### 内核架构设计

FlashAttention 通过分块计算和在线Softmax算法实现内存高效的注意力计算。

```cpp
// FlashAttention 前向传播内核
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__global__ void flash_attention_forward_kernel(
    T* __restrict__ out,              // 输出 [batch, seqlen, num_heads, head_size]
    const T* __restrict__ q,          // 查询 [batch, seqlen, num_heads, head_size]
    const T* __restrict__ k,          // 键 [batch, seqlen, num_kv_heads, head_size]
    const T* __restrict__ v,          // 值 [batch, seqlen, num_kv_heads, head_size]
    float* __restrict__ softmax_lse,  // Softmax LSE [batch, num_heads, seqlen]
    const float softmax_scale,        // Softmax缩放因子
    const int seqlen_q,               // 查询序列长度
    const int seqlen_k,               // 键序列长度
    const bool is_causal              // 是否因果掩码
) {
    // 1) 线程块和线程索引
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int m_block = blockIdx.z;
    
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / 32;
    const int lane_idx = thread_idx % 32;
    
    // 2) 计算当前块的行范围
    const int m_start = m_block * BLOCK_M;
    const int m_end = min(m_start + BLOCK_M, seqlen_q);
    
    // 3) 共享内存分配
    extern __shared__ char shared_mem[];
    T* q_shared = reinterpret_cast<T*>(shared_mem);
    T* k_shared = q_shared + BLOCK_M * head_size;
    T* v_shared = k_shared + BLOCK_N * head_size;
    float* qk_shared = reinterpret_cast<float*>(v_shared + BLOCK_N * head_size);
    
    // 4) 加载Q块到共享内存
    load_q_block_to_shared(q, q_shared, batch_idx, head_idx, m_start, 
                          thread_idx, NUM_THREADS, head_size);
    
    // 5) 初始化输出累积器
    float max_val[BLOCK_M];
    float exp_sum[BLOCK_M];  
    T output_acc[BLOCK_M * head_size];
    
    for (int i = 0; i < BLOCK_M; ++i) {
        max_val[i] = -FLT_MAX;
        exp_sum[i] = 0.0f;
    }
    memset(output_acc, 0, sizeof(output_acc));
    
    // 6) 遍历K/V块
    for (int n_block = 0; n_block * BLOCK_N < seqlen_k; ++n_block) {
        const int n_start = n_block * BLOCK_N;
        const int n_end = min(n_start + BLOCK_N, seqlen_k);
        
        // 加载K/V块到共享内存
        load_kv_block_to_shared(k, v, k_shared, v_shared, batch_idx, head_idx,
                               n_start, thread_idx, NUM_THREADS, head_size);
        __syncthreads();
        
        // 计算Q@K^T
        compute_qk_scores(q_shared, k_shared, qk_shared, m_start, m_end,
                         n_start, n_end, thread_idx, softmax_scale, is_causal);
        __syncthreads();
        
        // 在线Softmax更新
        online_softmax_update(qk_shared, v_shared, output_acc, max_val, exp_sum,
                             m_start, m_end, n_start, n_end, thread_idx);
        __syncthreads();
    }
    
    // 7) 最后归一化和写回输出
    normalize_and_write_output(output_acc, out, softmax_lse, max_val, exp_sum,
                              batch_idx, head_idx, m_start, m_end, thread_idx);
}
```

### FlashAttention 优化细节

#### 1. 分块策略优化

```cpp
// 自适应分块大小选择
template<typename T>
__host__ void select_optimal_block_sizes(
    int seqlen, int head_size, int num_heads,
    int& block_m, int& block_n, int& block_k
) {
    // 基于共享内存限制计算最优块大小
    constexpr int SHARED_MEM_LIMIT = 48 * 1024;  // 48KB 共享内存
    const int element_size = sizeof(T);
    
    // 计算不同组件的内存需求
    auto calc_shared_mem = [&](int bm, int bn) -> int {
        int q_mem = bm * head_size * element_size;
        int kv_mem = 2 * bn * head_size * element_size; 
        int qk_mem = bm * bn * sizeof(float);
        return q_mem + kv_mem + qk_mem;
    };
    
    // 搜索最优块大小
    int best_throughput = 0;
    for (int bm = 16; bm <= 128; bm *= 2) {
        for (int bn = 16; bn <= 128; bn *= 2) {
            if (calc_shared_mem(bm, bn) <= SHARED_MEM_LIMIT) {
                int estimated_throughput = estimate_throughput(bm, bn, seqlen, head_size);
                if (estimated_throughput > best_throughput) {
                    best_throughput = estimated_throughput;
                    block_m = bm;
                    block_n = bn;
                }
            }
        }
    }
    
    block_k = head_size;  // K维度通常等于head_size
}
```

#### 2. 内存合并访问优化

```cpp
// 向量化内存访问
template<typename T, int VEC_SIZE>
__device__ void vectorized_load_store(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int elements,
    int thread_idx,
    int num_threads
) {
    using VecType = typename VectorType<T, VEC_SIZE>::type;
    
    const VecType* vec_src = reinterpret_cast<const VecType*>(src);
    VecType* vec_dst = reinterpret_cast<VecType*>(dst);
    
    const int vec_elements = elements / VEC_SIZE;
    
    // 向量化加载，确保内存合并
    for (int i = thread_idx; i < vec_elements; i += num_threads) {
        vec_dst[i] = vec_src[i];
    }
    
    // 处理剩余元素
    const int remaining = elements % VEC_SIZE;
    if (remaining > 0 && thread_idx < remaining) {
        const int base_idx = vec_elements * VEC_SIZE;
        dst[base_idx + thread_idx] = src[base_idx + thread_idx];
    }
}
```

## 量化内核实现

### AWQ INT4 量化内核

```cpp
// AWQ INT4 GEMM 内核实现
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int NUM_THREADS>
__global__ void awq_gemm_kernel(
    half* __restrict__ out,              // 输出 [M, N]
    const half* __restrict__ input,      // FP16输入 [M, K]
    const uint32_t* __restrict__ qweight,// INT4量化权重 [K/8, N]
    const uint32_t* __restrict__ qzeros, // INT4量化零点 [K/group_size, N/8]
    const half* __restrict__ scales,     // FP16缩放因子 [K/group_size, N]
    const int M, const int N, const int K,
    const int group_size,                // 量化分组大小
    const int split_k_iters              // Split-K迭代数
) {
    // 1) 线程块索引计算
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int split_k = blockIdx.z;
    
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / 32;
    const int lane_idx = thread_idx % 32;
    
    // 2) 计算当前块的范围
    const int m_start = block_m * BLOCK_SIZE_M;
    const int n_start = block_n * BLOCK_SIZE_N;
    const int k_start = split_k * (K / split_k_iters);
    const int k_end = (split_k == split_k_iters - 1) ? K : (split_k + 1) * (K / split_k_iters);
    
    // 3) 共享内存分配
    extern __shared__ char shared_mem[];
    half* input_shared = reinterpret_cast<half*>(shared_mem);
    uint32_t* weight_shared = reinterpret_cast<uint32_t*>(input_shared + BLOCK_SIZE_M * BLOCK_SIZE_K);
    half* dequant_shared = reinterpret_cast<half*>(weight_shared + BLOCK_SIZE_K * BLOCK_SIZE_N / 8);
    
    // 4) 初始化累积器
    float acc[BLOCK_SIZE_M * BLOCK_SIZE_N / NUM_THREADS] = {0.0f};
    
    // 5) 主循环：处理K维度的块
    for (int k_block = k_start; k_block < k_end; k_block += BLOCK_SIZE_K) {
        const int k_actual = min(BLOCK_SIZE_K, k_end - k_block);
        
        // 加载输入到共享内存
        load_input_block(input, input_shared, m_start, k_block, 
                        thread_idx, M, K, BLOCK_SIZE_M, k_actual);
        
        // 加载量化权重到共享内存  
        load_quantized_weights(qweight, weight_shared, k_block, n_start,
                              thread_idx, K, N, BLOCK_SIZE_K, BLOCK_SIZE_N);
        __syncthreads();
        
        // 反量化权重
        dequantize_weights_int4(weight_shared, dequant_shared, qzeros, scales,
                               k_block, n_start, thread_idx, group_size,
                               K, N, BLOCK_SIZE_K, BLOCK_SIZE_N);
        __syncthreads();
        
        // 执行矩阵乘法累积
        gemm_accumulate(input_shared, dequant_shared, acc,
                       thread_idx, BLOCK_SIZE_M, BLOCK_SIZE_N, k_actual);
        __syncthreads();
    }
    
    // 6) 写回输出结果
    write_output_block(acc, out, m_start, n_start, thread_idx,
                      M, N, BLOCK_SIZE_M, BLOCK_SIZE_N);
}
```

### INT4 反量化优化

```cpp
// 高效的INT4反量化实现
__device__ __forceinline__ void dequantize_int4_to_fp16(
    const uint32_t packed_weights,      // 8个INT4权重打包在uint32_t中
    half* dequantized,                  // 输出8个FP16值
    const uint32_t packed_zeros,        // 打包的零点
    const half* scales,                 // 缩放因子
    const int group_idx                 // 量化组索引
) {
    // 1) 解包INT4权重
    uint8_t weights[8];
    weights[0] = (packed_weights >> 0) & 0xF;
    weights[1] = (packed_weights >> 4) & 0xF;
    weights[2] = (packed_weights >> 8) & 0xF;
    weights[3] = (packed_weights >> 12) & 0xF;
    weights[4] = (packed_weights >> 16) & 0xF;
    weights[5] = (packed_weights >> 20) & 0xF;
    weights[6] = (packed_weights >> 24) & 0xF;
    weights[7] = (packed_weights >> 28) & 0xF;
    
    // 2) 解包零点
    uint8_t zeros[8];
    zeros[0] = (packed_zeros >> 0) & 0xF;
    zeros[1] = (packed_zeros >> 4) & 0xF;
    zeros[2] = (packed_zeros >> 8) & 0xF;
    zeros[3] = (packed_zeros >> 12) & 0xF;
    zeros[4] = (packed_zeros >> 16) & 0xF;
    zeros[5] = (packed_zeros >> 20) & 0xF;
    zeros[6] = (packed_zeros >> 24) & 0xF;
    zeros[7] = (packed_zeros >> 28) & 0xF;
    
    // 3) 向量化反量化
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float dequant_val = (static_cast<float>(weights[i]) - static_cast<float>(zeros[i])) 
                           * __half2float(scales[group_idx]);
        dequantized[i] = __float2half(dequant_val);
    }
}
```

## MOE (Mixture of Experts) 内核

### 专家路由内核

```cpp
// MOE 专家路由和分发内核
template<int NUM_EXPERTS, int TOP_K, int HIDDEN_SIZE>
__global__ void moe_routing_kernel(
    float* __restrict__ expert_weights,     // 专家权重 [num_tokens, num_experts]
    int* __restrict__ selected_experts,     // 选中的专家 [num_tokens, top_k]
    float* __restrict__ routing_weights,    // 路由权重 [num_tokens, top_k]
    int* __restrict__ expert_tokens,        // 每个专家的token数 [num_experts]
    int* __restrict__ token_to_expert_map,  // token到专家的映射
    const int num_tokens                    // token总数
) {
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_idx = threadIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    // 1) 加载当前token的专家权重到共享内存
    extern __shared__ float shared_weights[];
    float* token_weights = shared_weights + thread_idx * NUM_EXPERTS;
    
    for (int expert_idx = 0; expert_idx < NUM_EXPERTS; ++expert_idx) {
        token_weights[expert_idx] = expert_weights[token_idx * NUM_EXPERTS + expert_idx];
    }
    
    // 2) Top-K选择算法
    int top_experts[TOP_K];
    float top_weights[TOP_K];
    
    // 初始化为最小值
    for (int k = 0; k < TOP_K; ++k) {
        top_experts[k] = -1;
        top_weights[k] = -FLT_MAX;
    }
    
    // 找到Top-K专家
    for (int expert_idx = 0; expert_idx < NUM_EXPERTS; ++expert_idx) {
        float weight = token_weights[expert_idx];
        
        // 插入排序维护Top-K
        for (int k = 0; k < TOP_K; ++k) {
            if (weight > top_weights[k]) {
                // 向后移动较小的权重
                for (int j = TOP_K - 1; j > k; --j) {
                    top_weights[j] = top_weights[j - 1];
                    top_experts[j] = top_experts[j - 1];
                }
                top_weights[k] = weight;
                top_experts[k] = expert_idx;
                break;
            }
        }
    }
    
    // 3) Softmax归一化路由权重
    float sum_exp = 0.0f;
    for (int k = 0; k < TOP_K; ++k) {
        top_weights[k] = expf(top_weights[k]);
        sum_exp += top_weights[k];
    }
    
    for (int k = 0; k < TOP_K; ++k) {
        top_weights[k] /= sum_exp;
        
        // 保存结果
        selected_experts[token_idx * TOP_K + k] = top_experts[k];
        routing_weights[token_idx * TOP_K + k] = top_weights[k];
        
        // 原子递增专家token计数
        if (top_experts[k] >= 0) {
            atomicAdd(&expert_tokens[top_experts[k]], 1);
        }
    }
}
```

### MOE 专家计算内核

```cpp
// MOE 专家前馈网络计算
template<int EXPERT_HIDDEN_SIZE, int BLOCK_SIZE>
__global__ void moe_expert_ffn_kernel(
    float* __restrict__ output,             // 输出 [num_expert_tokens, hidden_size]
    const float* __restrict__ input,        // 输入 [num_expert_tokens, hidden_size]
    const float* __restrict__ gate_weight,  // 门控权重 [hidden_size, expert_hidden_size]
    const float* __restrict__ up_weight,    // 上升权重 [hidden_size, expert_hidden_size]  
    const float* __restrict__ down_weight,  // 下降权重 [expert_hidden_size, hidden_size]
    const int num_expert_tokens,            // 当前专家的token数
    const int hidden_size                   // 隐藏层大小
) {
    const int token_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    
    if (token_idx >= num_expert_tokens) return;
    
    // 1) 共享内存分配
    extern __shared__ float shared_mem[];
    float* gate_output = shared_mem;
    float* up_output = gate_output + EXPERT_HIDDEN_SIZE;
    float* gated_output = up_output + EXPERT_HIDDEN_SIZE;
    
    // 2) 计算门控和上升投影
    const float* token_input = input + token_idx * hidden_size;
    
    // Gate projection: input @ gate_weight
    for (int i = thread_idx; i < EXPERT_HIDDEN_SIZE; i += BLOCK_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += token_input[j] * gate_weight[j * EXPERT_HIDDEN_SIZE + i];
        }
        gate_output[i] = sum;
    }
    
    // Up projection: input @ up_weight  
    for (int i = thread_idx; i < EXPERT_HIDDEN_SIZE; i += BLOCK_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += token_input[j] * up_weight[j * EXPERT_HIDDEN_SIZE + i];
        }
        up_output[i] = sum;
    }
    __syncthreads();
    
    // 3) 应用激活函数和门控机制
    for (int i = thread_idx; i < EXPERT_HIDDEN_SIZE; i += BLOCK_SIZE) {
        // SwiGLU激活: gate_output * sigmoid(gate_output) * up_output
        float gate_val = gate_output[i];
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
        gated_output[i] = gate_val * sigmoid_gate * up_output[i];
    }
    __syncthreads();
    
    // 4) 下降投影
    float* token_output = output + token_idx * hidden_size;
    for (int i = thread_idx; i < hidden_size; i += BLOCK_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < EXPERT_HIDDEN_SIZE; ++j) {
            sum += gated_output[j] * down_weight[j * hidden_size + i];
        }
        token_output[i] = sum;
    }
}
```

## 采样内核实现

### Top-K Top-P 采样内核

```cpp
// 高效的 Top-K Top-P 采样内核
template<int VOCAB_SIZE, int BLOCK_SIZE>
__global__ void top_k_top_p_sampling_kernel(
    int* __restrict__ output_tokens,        // 输出token [batch_size]
    const float* __restrict__ logits,       // 输入logits [batch_size, vocab_size]
    const float* __restrict__ temperatures, // 温度参数 [batch_size]
    const int* __restrict__ top_k_values,   // Top-K值 [batch_size]
    const float* __restrict__ top_p_values, // Top-P值 [batch_size]
    curandState* __restrict__ rand_states,  // 随机数状态
    const int batch_size,                   // 批大小
    const int vocab_size                    // 词汇表大小
) {
    const int batch_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // 1) 加载当前批次的参数
    const float temperature = temperatures[batch_idx];
    const int top_k = top_k_values[batch_idx];
    const float top_p = top_p_values[batch_idx];
    const float* batch_logits = logits + batch_idx * vocab_size;
    
    // 2) 共享内存分配
    extern __shared__ char shared_mem[];
    float* probs = reinterpret_cast<float*>(shared_mem);
    int* indices = reinterpret_cast<int*>(probs + vocab_size);
    float* prefix_sums = reinterpret_cast<float*>(indices + vocab_size);
    
    // 3) 应用温度缩放并计算概率
    float max_logit = -FLT_MAX;
    
    // 找最大logit值
    for (int i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
        float logit = batch_logits[i] / temperature;
        probs[i] = logit;
        max_logit = fmaxf(max_logit, logit);
    }
    
    // 块内reduction找全局最大值
    __shared__ float shared_max;
    max_logit = block_reduce_max(max_logit, thread_idx, BLOCK_SIZE);
    if (thread_idx == 0) shared_max = max_logit;
    __syncthreads();
    max_logit = shared_max;
    
    // 计算稳定的softmax概率
    float sum_exp = 0.0f;
    for (int i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
        float prob = expf(probs[i] - max_logit);
        probs[i] = prob;
        sum_exp += prob;
        indices[i] = i;  // 初始化索引
    }
    
    // 归一化概率
    sum_exp = block_reduce_sum(sum_exp, thread_idx, BLOCK_SIZE);
    __shared__ float shared_sum;
    if (thread_idx == 0) shared_sum = sum_exp;
    __syncthreads();
    sum_exp = shared_sum;
    
    for (int i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
        probs[i] /= sum_exp;
    }
    __syncthreads();
    
    // 4) Top-K 过滤
    if (top_k < vocab_size) {
        // 使用 bitonic sort 进行部分排序
        bitonic_sort_top_k(probs, indices, vocab_size, top_k, thread_idx, BLOCK_SIZE);
        __syncthreads();
        
        // 清零非Top-K的概率
        for (int i = thread_idx + top_k; i < vocab_size; i += BLOCK_SIZE) {
            probs[indices[i]] = 0.0f;
        }
        __syncthreads();
    }
    
    // 5) Top-P 过滤
    if (top_p < 1.0f) {
        // 计算累积概率
        prefix_sum_scan(probs, prefix_sums, vocab_size, thread_idx, BLOCK_SIZE);
        __syncthreads();
        
        // 找到累积概率超过top_p的截止点
        int cutoff_idx = vocab_size;
        for (int i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
            if (prefix_sums[i] >= top_p && i < cutoff_idx) {
                cutoff_idx = i + 1;  // 包含当前token
            }
        }
        
        // 块内reduction找最小截止索引
        cutoff_idx = block_reduce_min(cutoff_idx, thread_idx, BLOCK_SIZE);
        __shared__ int shared_cutoff;
        if (thread_idx == 0) shared_cutoff = cutoff_idx;
        __syncthreads();
        cutoff_idx = shared_cutoff;
        
        // 清零超过cutoff的概率
        for (int i = thread_idx + cutoff_idx; i < vocab_size; i += BLOCK_SIZE) {
            probs[i] = 0.0f;
        }
        __syncthreads();
        
        // 重新归一化
        float filtered_sum = 0.0f;
        for (int i = thread_idx; i < cutoff_idx; i += BLOCK_SIZE) {
            filtered_sum += probs[i];
        }
        filtered_sum = block_reduce_sum(filtered_sum, thread_idx, BLOCK_SIZE);
        if (thread_idx == 0) shared_sum = filtered_sum;
        __syncthreads();
        filtered_sum = shared_sum;
        
        for (int i = thread_idx; i < cutoff_idx; i += BLOCK_SIZE) {
            probs[i] /= filtered_sum;
        }
        __syncthreads();
    }
    
    // 6) 多项式采样
    if (thread_idx == 0) {
        curandState* rand_state = &rand_states[batch_idx];
        float random_val = curand_uniform(rand_state);
        
        // 根据累积概率采样
        float cumulative = 0.0f;
        int sampled_token = vocab_size - 1;  // 默认最后一个token
        
        for (int i = 0; i < vocab_size; ++i) {
            cumulative += probs[i];
            if (cumulative >= random_val) {
                sampled_token = i;
                break;
            }
        }
        
        output_tokens[batch_idx] = sampled_token;
    }
}
```

## 性能优化技术总结

### 1. 内存访问优化
- **合并访问**：确保相邻线程访问连续内存
- **向量化加载**：使用128位向量指令
- **共享内存缓存**：缓存频繁访问的数据
- **Bank冲突避免**：优化共享内存布局

### 2. 计算优化
- **指令级并行**：充分利用GPU的指令流水线
- **寄存器优化**：减少寄存器使用提高占用率
- **分支减少**：避免线程束内的分支发散
- **数学函数优化**：使用快速数学库函数

### 3. 同步优化
- **减少同步点**：最小化__syncthreads()调用
- **Warp级同步**：使用warp shuffle减少同步开销
- **异步执行**：重叠计算和内存传输

### 4. 数值稳定性
- **在线算法**：避免中间结果的数值溢出
- **混合精度**：平衡精度和性能
- **梯度裁剪**：防止梯度爆炸

这些核心内核的优化实现为vLLM提供了高性能的计算基础，支撑了整个推理框架的高效运行。
