---
title: "PyTorch-08-数据结构UML与交互图"
date: 2025-10-05T12:57:00+08:00
draft: false
tags:
  - 数据结构
  - UML
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - PyTorch-08-数据结构UML与交互图"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# PyTorch-08-数据结构UML与交互图

## 核心数据结构关系图

### 完整系统UML类图

```mermaid
classDiagram
    %% c10核心层
    class intrusive_ptr_target {
        <<interface>>
        +std::atomic~uint64_t~ combined_refcount_
        +uint32_t refcount()
        +uint32_t weakcount()
        +void incref()
        +void decref()
    }
    
    class TensorImpl {
        -Storage storage_
        -SizesAndStrides sizes_and_strides_
        -int64_t storage_offset_
        -int64_t numel_
        -TypeMeta data_type_
        -optional~Device~ device_opt_
        -DispatchKeySet key_set_
        -VariableVersion version_counter_
        -unique_ptr~AutogradMetaInterface~ autograd_meta_
        -unique_ptr~ExtraMeta~ extra_meta_
        -PyObjectSlot pyobj_slot_
        
        +IntArrayRef sizes()
        +IntArrayRef strides()
        +int64_t dim()
        +int64_t numel()
        +ScalarType dtype()
        +Device device()
        +Storage storage()
        +bool is_contiguous()
        +void set_sizes_contiguous(IntArrayRef)
        +TensorImpl* as_view()
    }
    
    class StorageImpl {
        -DataPtr data_ptr_
        -SymInt size_bytes_
        -Allocator* allocator_
        -bool resizable_
        -bool received_cuda_
        
        +const DataPtr& data_ptr()
        +DataPtr& mutable_data_ptr()
        +size_t nbytes()
        +bool resizable()
        +void set_nbytes(size_t)
        +void reset()
    }
    
    class Allocator {
        <<interface>>
        +virtual DataPtr allocate(size_t) = 0
        +virtual void copy_data(void*, const void*, size_t) = 0
        +virtual DeleterFnPtr raw_deleter()
        +DataPtr clone(const void*, size_t)
    }
    
    class CPUAllocator {
        +DataPtr allocate(size_t) override
        +void copy_data(void*, const void*, size_t) override
    }
    
    class CUDACachingAllocator {
        -DeviceAllocator base_allocator_
        -std::vector~Block*~ free_blocks_
        -std::unordered_set~Block*~ active_blocks_
        
        +DataPtr allocate(size_t) override
        +void free_block(Block*)
        +void empty_cache()
        +DeviceStats get_stats()
    }
    
    class DispatchKeySet {
        -uint64_t repr_
        
        +bool has(DispatchKey)
        +DispatchKeySet add(DispatchKey)
        +DispatchKeySet remove(DispatchKey)
        +DispatchKey highestPriorityTypeId()
        +int getDispatchTableIndexForDispatchKeySet()
    }
    
    %% ATen层
    class Tensor {
        -intrusive_ptr~TensorImpl~ impl_
        
        +Tensor add(const Tensor&, const Scalar&)
        +Tensor& add_(const Tensor&, const Scalar&)
        +Tensor matmul(const Tensor&)
        +Tensor view(IntArrayRef)
        +Tensor transpose(int64_t, int64_t)
        +void backward()
        +Tensor grad()
    }
    
    class TensorBase {
        -intrusive_ptr~TensorImpl~ impl_
        
        +IntArrayRef sizes()
        +IntArrayRef strides()
        +int64_t numel()
        +ScalarType dtype()
        +Device device()
        +bool is_same(const TensorBase&)
    }
    
    class Dispatcher {
        -std::array~OperatorEntry~ operators_
        -std::vector~BackendFallbackKernel~ backend_fallback_kernels_
        
        +static Dispatcher& singleton()
        +template~class Return, class... Args~
         Return call(const TypedOperatorHandle~Return(Args...)>&, Args...)
        +RegistrationHandleRAII registerDef(FunctionSchema)
        +RegistrationHandleRAII registerImpl(OperatorHandle, DispatchKey, KernelFunction)
    }
    
    class OperatorEntry {
        -FunctionSchema schema_
        -std::array~AnnotatedKernel~ dispatchTable_
        -DispatchKeyExtractor dispatchKeyExtractor_
        
        +const KernelFunction& lookup(DispatchKeySet)
        +bool hasKernelForDispatchKey(DispatchKey)
        +void updateDispatchTable(DispatchKey, KernelFunction)
    }
    
    class TensorIterator {
        -SmallVector~OperandInfo~ operands_
        -DimVector shape_
        -int64_t numel_
        -bool is_reduction_
        -bool all_ops_same_shape_
        
        +void add_output(const Tensor&)
        +void add_input(const Tensor&)
        +TensorIteratorConfig& build()
        +char* data_ptr(int arg)
        +void for_each(loop2d_t loop)
    }
    
    %% Autograd层
    class Node {
        <<interface>>
        +edge_list next_edges_
        +std::vector~SavedVariable~ saved_variables_
        +uint64_t sequence_nr_
        
        +virtual variable_list apply(variable_list&&) = 0
        +virtual std::string name() = 0
        +void save_variables(TensorList)
    }
    
    class AddBackward0 {
        +variable_list apply(variable_list&&) override
        +std::string name() override
    }
    
    class MulBackward0 {
        +SavedVariable saved_self
        +SavedVariable saved_other
        
        +variable_list apply(variable_list&&) override
        +std::string name() override
    }
    
    class AutogradMeta {
        -Variable grad_
        -std::shared_ptr~Node~ grad_fn_
        -std::weak_ptr~Node~ grad_accumulator_
        -std::vector~std::shared_ptr~FunctionPreHook~~ hooks_
        -bool requires_grad_
        -bool is_view_
        
        +void set_requires_grad(bool, TensorImpl*)
        +bool requires_grad()
        +Variable& mutable_grad()
        +const std::shared_ptr~Node~& grad_fn()
    }
    
    class Engine {
        -std::vector~std::shared_ptr~ReadyQueue~~ ready_queues_
        -std::vector~std::thread~ workers_
        
        +static Engine& get_default_engine()
        +variable_list execute(const edge_list&, const variable_list&, bool, bool, const edge_list&)
        +void thread_main(std::shared_ptr~GraphTask~)
    }
    
    %% torch.nn层
    class Module {
        -OrderedDict~std::string, std::shared_ptr~Module~~ _modules
        -OrderedDict~std::string, Parameter~ _parameters
        -OrderedDict~std::string, Tensor~ _buffers
        -bool training_
        
        +virtual Tensor forward(TensorList) = 0
        +Tensor operator()(TensorList)
        +void train(bool)
        +void eval()
        +std::vector~Tensor~ parameters()
        +void to(Device)
    }
    
    class Linear {
        +Parameter weight
        +Parameter bias
        +int64_t in_features
        +int64_t out_features
        
        +Tensor forward(const Tensor&) override
        +void reset_parameters()
    }
    
    class Conv2d {
        +Parameter weight
        +Parameter bias
        +int64_t in_channels
        +int64_t out_channels
        +std::pair~int64_t, int64_t~ kernel_size
        
        +Tensor forward(const Tensor&) override
    }
    
    %% 继承关系
    intrusive_ptr_target <|-- TensorImpl
    intrusive_ptr_target <|-- StorageImpl
    Allocator <|-- CPUAllocator
    Allocator <|-- CUDACachingAllocator
    TensorBase <|-- Tensor
    Node <|-- AddBackward0
    Node <|-- MulBackward0
    Module <|-- Linear
    Module <|-- Conv2d
    
    %% 组合关系
    TensorImpl *-- StorageImpl : storage_
    TensorImpl *-- DispatchKeySet : key_set_
    TensorImpl *-- AutogradMeta : autograd_meta_
    StorageImpl *-- Allocator : allocator_
    Tensor *-- TensorImpl : impl_
    Dispatcher *-- OperatorEntry : operators_
    AutogradMeta *-- Node : grad_fn_
    
    %% 依赖关系
    TensorIterator ..> Tensor : operates on
    Engine ..> Node : executes
    Linear ..> Tensor : processes
    Conv2d ..> Tensor : processes
```

### 模块交互状态图

```mermaid
stateDiagram-v2
    [*] --> TensorCreated
    
    TensorCreated --> WithGrad : requires_grad=True
    TensorCreated --> WithoutGrad : requires_grad=False
    
    WithGrad --> ComputationGraph : 参与运算
    WithoutGrad --> SimpleComputation : 参与运算
    
    ComputationGraph --> GraphBuilt : 构建完成
    SimpleComputation --> ResultReady : 计算完成
    
    GraphBuilt --> BackwardReady : 调用backward()
    BackwardReady --> GradientsComputed : 梯度计算完成
    
    GradientsComputed --> OptimizationStep : 调用optimizer.step()
    OptimizationStep --> ParametersUpdated : 参数更新完成
    
    ParametersUpdated --> [*] : 训练步骤结束
    ResultReady --> [*] : 推理完成
    
    note right of ComputationGraph
        构建autograd图
        创建Function节点
        保存前向上下文
    end note
    
    note right of BackwardReady
        启动Engine
        遍历计算图
        计算梯度
    end note
```

### 内存管理生命周期图

```mermaid
stateDiagram-v2
    [*] --> Allocated : Allocator::allocate()
    
    Allocated --> InUse : 创建Tensor
    InUse --> Shared : 多个Tensor共享
    InUse --> Released : 引用计数=0
    
    Shared --> InUse : 其他引用释放
    Shared --> Released : 所有引用释放
    
    Released --> Cached : CachingAllocator缓存
    Released --> Deallocated : 直接释放
    
    Cached --> InUse : 缓存命中复用
    Cached --> Deallocated : 内存压力释放
    
    Deallocated --> [*]
    
    note right of Cached
        GPU内存池
        减少cudaMalloc调用
        按大小分桶管理
    end note
```

## 关键交互时序图

### 训练一个batch的完整时序

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant DataLoader as DataLoader
    participant Model as nn.Module
    participant Tensor as Tensor
    participant Autograd as Autograd Engine
    participant Optimizer as Optimizer
    participant Allocator as GPU Allocator
    
    App->>DataLoader: next(iter(dataloader))
    DataLoader->>DataLoader: 加载批次数据
    DataLoader->>Allocator: 分配GPU内存
    Allocator-->>DataLoader: 返回内存地址
    DataLoader-->>App: batch数据
    
    App->>Model: model(batch_input)
    
    loop 前向传播
        Model->>Tensor: 各种算子调用
        Tensor->>Tensor: 构建autograd图
        Note over Tensor: 创建Function节点<br/>保存前向上下文
    end
    
    Model-->>App: loss值
    
    App->>Autograd: loss.backward()
    Autograd->>Autograd: 创建GraphTask
    Autograd->>Autograd: 拓扑排序
    
    loop 反向传播
        Autograd->>Autograd: 执行Function::apply()
        Note over Autograd: 计算梯度<br/>传播给输入
    end
    
    Autograd->>Tensor: 累积梯度到.grad
    Autograd-->>App: 反向传播完成
    
    App->>Optimizer: optimizer.step()
    Optimizer->>Tensor: 更新参数
    Optimizer-->>App: 参数更新完成
    
    App->>Model: model.zero_grad()
    Model->>Tensor: 清零梯度
```

### 算子分发详细时序

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户调用
    participant PyBind as Python绑定
    participant Dispatcher as Dispatcher
    participant OpEntry as OperatorEntry
    participant Kernel as 具体Kernel
    participant CUDA as CUDA Runtime
    
    User->>PyBind: tensor.add(other)
    PyBind->>PyBind: 解析Python参数
    PyBind->>Dispatcher: at::add(self, other, alpha)
    
    Dispatcher->>Dispatcher: 提取DispatchKeySet
    Note over Dispatcher: self.key_set() | other.key_set()<br/>{CUDA, AutogradCUDA}
    
    Dispatcher->>OpEntry: lookup(DispatchKeySet)
    OpEntry->>OpEntry: getDispatchTableIndex()
    Note over OpEntry: 位运算计算索引<br/>查表获取KernelFunction
    
    OpEntry-->>Dispatcher: KernelFunction
    Dispatcher->>Kernel: call(op, DispatchKeySet, args)
    
    alt Autograd模式
        Kernel->>Kernel: AutogradCUDA内核
        Note over Kernel: 记录AddBackward节点<br/>保存前向上下文
        Kernel->>Dispatcher: redispatch去除Autograd键
        Dispatcher->>Kernel: CUDA内核
    else 推理模式
        Kernel->>Kernel: CUDA内核
    end
    
    Kernel->>CUDA: cudaLaunchKernel()
    CUDA->>CUDA: GPU执行kernel
    CUDA-->>Kernel: 执行完成
    
    Kernel-->>Dispatcher: 返回结果
    Dispatcher-->>PyBind: 返回结果
    PyBind-->>User: Python Tensor对象
```

### 内存分配与回收时序

```mermaid
sequenceDiagram
    autonumber
    participant Tensor as Tensor创建
    participant Storage as StorageImpl
    participant Allocator as CUDACachingAllocator
    participant Pool as 内存池
    participant CUDA as CUDA Driver
    participant GC as 垃圾回收
    
    Tensor->>Storage: 创建StorageImpl
    Storage->>Allocator: allocate(nbytes)
    
    Allocator->>Pool: 查找空闲块
    alt 缓存命中
        Pool-->>Allocator: 返回缓存块
    else 缓存未命中
        Allocator->>CUDA: cudaMalloc(nbytes)
        CUDA-->>Allocator: device_ptr
        Allocator->>Pool: 记录新分配块
    end
    
    Allocator-->>Storage: DataPtr
    Storage-->>Tensor: 完成创建
    
    Note over Tensor: 使用张量...
    
    Tensor->>GC: 引用计数=0
    GC->>Storage: ~StorageImpl()
    Storage->>Allocator: DataPtr析构
    Allocator->>Pool: 回收块到空闲列表
    
    Note over Pool: 延迟释放<br/>等待内存压力
    
    opt 内存压力大时
        Pool->>CUDA: cudaFree(device_ptr)
        CUDA-->>Pool: 释放完成
    end
```

## 性能瓶颈分析图

### CPU时间分布图

```mermaid
pie title CPU时间分布 (训练1个epoch)
    "CUDA Kernels" : 65.2
    "Python解释器" : 12.3
    "内存管理" : 8.7
    "Dispatcher分发" : 6.1
    "Autograd引擎" : 4.2
    "数据加载" : 2.8
    "其他" : 0.7
```

### 内存使用分布图

```mermaid
pie title GPU内存使用分布
    "模型参数" : 35.5
    "激活值" : 28.2
    "梯度" : 15.8
    "优化器状态" : 12.3
    "临时缓冲区" : 6.1
    "内存池开销" : 2.1
```

### 调用频次热力图

```mermaid
graph TD
    A[at::add - 45.2%] --> A1[add_cpu: 12.3%]
    A --> A2[add_cuda: 32.9%]
    
    B[at::matmul - 23.1%] --> B1[mm_cpu: 3.2%]
    B --> B2[mm_cuda: 19.9%]
    
    C[at::conv2d - 18.7%] --> C1[cudnn_conv: 18.7%]
    
    D[autograd::backward - 8.9%] --> D1[Engine::execute: 8.9%]
    
    E[其他算子 - 4.1%]
    
    style A fill:#ff6b6b
    style B fill:#ff8e53
    style C fill:#ff8e53
    style D fill:#4ecdc4
    style E fill:#45b7d1
```

## 数据流向图

### 训练数据流

```mermaid
flowchart LR
    subgraph 数据加载
        A1[原始数据] --> A2[DataLoader]
        A2 --> A3[批次数据]
        A3 --> A4[GPU内存]
    end
    
    subgraph 前向传播
        A4 --> B1[Input Layer]
        B1 --> B2[Hidden Layers]
        B2 --> B3[Output Layer]
        B3 --> B4[Loss Function]
    end
    
    subgraph 反向传播
        B4 --> C1[Loss.backward]
        C1 --> C2[Autograd Engine]
        C2 --> C3[梯度计算]
        C3 --> C4[梯度累积]
    end
    
    subgraph 参数更新
        C4 --> D1[Optimizer]
        D1 --> D2[参数更新]
        D2 --> D3[梯度清零]
    end
    
    D3 --> A2
    
    style A4 fill:#e1f5ff
    style B4 fill:#e8f5e9
    style C2 fill:#fff4e1
    style D1 fill:#fce4ec
```

### 内存数据流

```mermaid
flowchart TD
    subgraph CPU内存
        A1[Python对象] --> A2[numpy arrays]
        A2 --> A3[torch.Tensor]
    end
    
    subgraph GPU内存
        B1[Device Tensor]
        B2[Kernel输入]
        B3[Kernel输出]
        B4[缓存池]
    end
    
    subgraph 计算单元
        C1[CUDA Cores]
        C2[Tensor Cores]
        C3[cuDNN]
    end
    
    A3 --> B1 : to(device)
    B1 --> B2 : 内存布局转换
    B2 --> C1 : 一般计算
    B2 --> C2 : 混合精度
    B2 --> C3 : 卷积/RNN
    
    C1 --> B3
    C2 --> B3
    C3 --> B3
    
    B3 --> B1 : 结果存储
    B1 --> B4 : 释放到缓存
    B4 --> B1 : 缓存复用
    
    B1 --> A3 : to('cpu')
```

---

**文档版本**: v1.0  
**最后更新**: 2025-01-01
