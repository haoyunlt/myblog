# LevelDB 整体架构分析

## 1. 架构概述

LevelDB采用LSM-Tree（Log-Structured Merge-Tree）架构，将随机写入转换为顺序写入，通过多级存储结构实现高性能的键值存储。整个系统设计遵循"写入优化"的原则，牺牲一定的读取性能来换取卓越的写入性能。

## 2. 核心架构组件

### 2.1 分层架构视图

```mermaid
graph TB
    subgraph "用户接口层"
        A[DB Interface] --> B[DBImpl]
    end
    
    subgraph "内存存储层"
        B --> C[MemTable]
        B --> D[Immutable MemTable]
    end
    
    subgraph "持久化层"
        B --> E[WAL Log]
        B --> F[SSTable Files]
    end
    
    subgraph "元数据管理层"
        B --> G[VersionSet]
        G --> H[Version]
        G --> I[MANIFEST]
    end
    
    subgraph "工具层"
        J[TableCache] --> F
        K[BlockCache] --> F
        L[Comparator] --> C
        L --> F
        M[FilterPolicy] --> F
    end
    
    subgraph "系统层"
        N[Env] --> E
        N --> F
        N --> I
    end
```

### 2.2 数据流向图

```mermaid
flowchart TD
    A[写入请求] --> B{检查MemTable空间}
    B -->|空间足够| C[写入WAL日志]
    C --> D[写入MemTable]
    D --> E[返回成功]
    
    B -->|空间不足| F[切换MemTable]
    F --> G[启动后台压缩]
    G --> C
    
    H[读取请求] --> I[查找MemTable]
    I -->|找到| J[返回结果]
    I -->|未找到| K[查找Immutable MemTable]
    K -->|找到| J
    K -->|未找到| L[查找SSTable]
    L --> M[Level 0]
    M -->|未找到| N[Level 1]
    N -->|未找到| O[Level N]
    O --> P[返回NotFound/结果]
```

## 3. 核心模块详解

### 3.1 内存存储模块

#### MemTable架构
```mermaid
classDiagram
    class MemTable {
        +SkipList~char*, KeyComparator~ table_
        +Arena arena_
        +int refs_
        +Add(seq, type, key, value)
        +Get(key, value, status)
        +NewIterator()
        +ApproximateMemoryUsage()
    }
    
    class SkipList {
        +Node* head_
        +Random rnd_
        +KeyComparator compare_
        +Insert(key)
        +Contains(key)
        +NewIterator()
    }
    
    class Arena {
        +vector~char*~ blocks_
        +size_t alloc_ptr_
        +size_t alloc_bytes_remaining_
        +Allocate(bytes)
        +AllocateAligned(bytes)
    }
    
    MemTable --> SkipList
    MemTable --> Arena
    SkipList --> Arena
```

**MemTable特性分析**:
- **数据结构**: 使用跳表（SkipList）实现，支持高效的有序插入和查找
- **内存管理**: Arena内存池避免频繁的malloc/free操作
- **并发控制**: 支持单写多读，通过引用计数管理生命周期
- **内存限制**: 默认4MB大小，超过后切换为Immutable状态

### 3.2 持久化存储模块

#### SSTable文件结构
```mermaid
graph TD
    subgraph "SSTable文件格式"
        A[Data Block 1] --> B[Data Block 2]
        B --> C[Data Block N]
        C --> D[MetaIndex Block]
        D --> E[Index Block]
        E --> F[Footer]
    end
    
    subgraph "Block内部结构"
        G[Record 1] --> H[Record 2]
        H --> I[Record N]
        I --> J[Restart Points]
        J --> K[Restart Count]
        K --> L[CRC32]
    end
    
    subgraph "记录格式"
        M[Shared Key Len] --> N[Unshared Key Len]
        N --> O[Value Len]
        O --> P[Unshared Key]
        P --> Q[Value]
    end
```

#### 多级存储结构
```mermaid
graph LR
    subgraph "Level 0"
        A[File 1] 
        B[File 2]
        C[File 3]
        D[File 4]
    end
    
    subgraph "Level 1"
        E[File A]
        F[File B] 
        G[File C]
    end
    
    subgraph "Level 2"
        H[File X]
        I[File Y]
        J[File Z]
        K[File W]
    end
    
    A -.-> E
    A -.-> F
    B -.-> E
    C -.-> F
    C -.-> G
    
    E -.-> H
    F -.-> I
    F -.-> J
    G -.-> K
    
    style A fill:#ff9999
    style B fill:#ff9999
    style C fill:#ff9999
    style D fill:#ff9999
    style E fill:#99ccff
    style F fill:#99ccff
    style G fill:#99ccff
    style H fill:#99ff99
    style I fill:#99ff99
    style J fill:#99ff99
    style K fill:#99ff99
```

**存储特性分析**:
- **Level 0**: 文件可能重叠，直接从MemTable压缩得到
- **Level 1+**: 文件不重叠，有序排列，便于查找
- **容量递增**: 每层容量是上一层的10倍（默认配置）
- **压缩策略**: 自动触发多级压缩，保持系统性能

### 3.3 版本管理模块

#### Version和VersionSet关系
```mermaid
classDiagram
    class VersionSet {
        +string dbname_
        +Options* options_
        +TableCache* table_cache_
        +uint64_t next_file_number_
        +uint64_t manifest_file_number_
        +uint64_t last_sequence_
        +Version* current_
        +LogAndApply(edit)
        +NewFileNumber()
        +PickCompaction()
    }
    
    class Version {
        +VersionSet* vset_
        +Version* next_
        +Version* prev_
        +vector~FileMetaData*~ files_[kNumLevels]
        +int refs_
        +Get(options, key, value)
        +AddIterators(options, iters)
        +UpdateStats(stats)
    }
    
    class VersionEdit {
        +string comparator_name_
        +uint64_t log_number_
        +uint64_t next_file_number_
        +SequenceNumber last_sequence_
        +vector~pair~int, InternalKey~~ compact_pointers_
        +set~pair~int, uint64_t~~ deleted_files_
        +vector~pair~int, FileMetaData~~ new_files_
        +EncodeTo(dst)
        +DecodeFrom(src)
    }
    
    class FileMetaData {
        +uint64_t number
        +uint64_t file_size
        +InternalKey smallest
        +InternalKey largest
        +int refs
        +int allowed_seeks
    }
    
    VersionSet --> Version : manages
    Version --> FileMetaData : contains
    VersionSet --> VersionEdit : applies
    VersionEdit --> FileMetaData : modifies
```

### 3.4 压缩管理模块

#### 压缩触发机制
```mermaid
graph TD
    A[写入操作] --> B{MemTable满？}
    B -->|是| C[Minor Compaction]
    B -->|否| D[继续写入]
    
    C --> E[创建Level 0 SSTable]
    E --> F{Level 0文件数 > 4？}
    F -->|是| G[触发Major Compaction]
    F -->|否| H[等待下次写入]
    
    G --> I{选择压缩Level}
    I --> J[Level N Size > Limit？]
    J -->|是| K[压缩Level N到N+1]
    J -->|否| L[检查下一Level]
    
    K --> M[选择重叠文件]
    M --> N[合并排序写入新文件]
    N --> O[更新Version]
    O --> P[删除旧文件]
```

#### 压缩算法流程
```mermaid
sequenceDiagram
    participant BG as 后台线程
    participant VS as VersionSet  
    participant C as Compaction
    participant TC as TableCache
    participant FS as 文件系统
    
    BG->>VS: PickCompaction()
    VS-->>BG: 返回Compaction对象
    
    alt 有压缩任务
        BG->>C: 获取输入文件列表
        BG->>TC: 创建合并迭代器
        
        loop 遍历所有键值对
            BG->>BG: 检查是否应该输出
            BG->>FS: 写入新SSTable文件
        end
        
        BG->>VS: LogAndApply(VersionEdit)
        VS->>FS: 更新MANIFEST文件
        BG->>FS: 删除旧文件
    else 无压缩任务
        BG->>BG: 等待下次调度
    end
```

## 4. 关键算法分析

### 4.1 跳表算法
跳表是MemTable的核心数据结构，提供O(log n)的查找性能：

```cpp
// 文件: db/skiplist.h
template<typename Key, class Comparator>
class SkipList {
private:
    struct Node {
        Key const key;
        
        Node* Next(int n) {
            return reinterpret_cast<Node*>(next_[n].load(std::memory_order_acquire));
        }
        
        void SetNext(int n, Node* x) {
            next_[n].store(x, std::memory_order_release);
        }
        
        std::atomic<Node*> next_[1];  // 实际大小由层数决定
    };
    
    Node* head_;
    std::atomic<int> max_height_;
    Random rnd_;
    
public:
    void Insert(const Key& key) {
        Node* prev[kMaxHeight];
        Node* x = FindGreaterOrEqual(key, prev);
        
        int height = RandomHeight();
        if (height > GetMaxHeight()) {
            for (int i = GetMaxHeight(); i < height; i++) {
                prev[i] = head_;
            }
            max_height_.store(height, std::memory_order_relaxed);
        }
        
        x = NewNode(key, height);
        for (int i = 0; i < height; i++) {
            x->SetNext(i, prev[i]->Next(i));
            prev[i]->SetNext(i, x);
        }
    }
};
```

### 4.2 LSM压缩算法
LevelDB的压缩算法确保各Level的大小控制在合理范围：

```cpp
// 文件: db/version_set.cc
Compaction* VersionSet::PickCompaction() {
    Compaction* c;
    int level;
    
    // 基于大小的压缩
    const bool size_compaction = (current_->compaction_score_ >= 1);
    // 基于查找的压缩  
    const bool seek_compaction = (current_->file_to_compact_ != nullptr);
    
    if (size_compaction) {
        level = current_->compaction_level_;
        c = new Compaction(&options_, level);
        
        // 选择要压缩的文件
        for (size_t i = 0; i < current_->files_[level].size(); i++) {
            FileMetaData* f = current_->files_[level][i];
            if (compact_pointer_[level].empty() ||
                icmp_.Compare(f->largest.Encode(), compact_pointer_[level]) > 0) {
                c->inputs_[0].push_back(f);
                break;
            }
        }
    } else if (seek_compaction) {
        level = current_->file_to_compact_level_;
        c = new Compaction(&options_, level);
        c->inputs_[0].push_back(current_->file_to_compact_);
    } else {
        return nullptr;
    }
    
    SetupOtherInputs(c);
    return c;
}
```

## 5. 性能优化设计

### 5.1 写入路径优化
```mermaid
graph LR
    A[用户写入] --> B[WriteBatch批量化]
    B --> C[WAL顺序写入]
    C --> D[MemTable内存写入] 
    D --> E[后台异步压缩]
    
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#FFB6C1
```

### 5.2 读取路径优化  
```mermaid
graph TD
    A[用户读取] --> B[MemTable查找]
    B -->|命中| F[返回结果]
    B -->|未命中| C[Immutable MemTable查找]
    C -->|命中| F
    C -->|未命中| D[布隆过滤器检测]
    D -->|可能存在| E[SSTable查找]
    D -->|一定不存在| G[返回NotFound]
    E --> H[Block Cache查找]
    H -->|命中| F
    H -->|未命中| I[磁盘读取]
    I --> F
    
    style B fill:#90EE90
    style C fill:#90EE90  
    style D fill:#FFB6C1
    style H fill:#87CEEB
```

## 6. 并发控制机制

### 6.1 读写并发
- **写入串行化**: 所有写入操作通过单一队列串行化执行
- **读写并行**: 读操作不阻塞写操作，通过MVCC机制实现
- **快照隔离**: 使用序列号实现快照隔离级别

### 6.2 后台压缩并发
```mermaid
sequenceDiagram
    participant W as 写入线程
    participant M as MemTable
    participant B as 后台线程
    participant S as SSTable
    
    W->>M: 写入数据
    W->>W: 检查MemTable大小
    
    alt MemTable满
        W->>M: 切换为Immutable
        W->>M: 创建新MemTable
        W->>B: 通知压缩
    end
    
    par 并行执行
        W->>M: 继续写入新MemTable
    and
        B->>M: 读取Immutable MemTable
        B->>S: 创建新SSTable
        B->>B: 删除Immutable MemTable
        B->>B: 检查是否需要Major压缩
    end
```

## 7. 容错和恢复机制

### 7.1 WAL日志恢复
```mermaid
flowchart TD
    A[数据库启动] --> B[扫描WAL文件]
    B --> C[按序列号排序]
    C --> D[重放日志记录]
    D --> E{记录类型？}
    E -->|Put| F[插入MemTable]
    E -->|Delete| G[标记删除MemTable]
    F --> H[继续下一条记录]
    G --> H
    H --> I{日志结束？}
    I -->|否| D
    I -->|是| J[创建新日志文件]
    J --> K[启动完成]
```

### 7.2 MANIFEST恢复
- **版本历史**: MANIFEST记录所有版本变更
- **检查点**: 定期写入完整版本信息
- **增量恢复**: 从最近检查点开始应用增量变更

## 8. 架构优势分析

### 8.1 写入优势
- **顺序写入**: WAL和SSTable都是顺序写入，充分利用磁盘性能
- **批量写入**: WriteBatch减少系统调用开销
- **异步压缩**: 压缩在后台执行，不阻塞前台写入

### 8.2 存储优势
- **分层存储**: 热数据在上层，冷数据在下层
- **增量压缩**: 只压缩重叠的文件，减少IO
- **空间回收**: 自动清理过期数据和文件

### 8.3 查询优势
- **内存优先**: 新数据在MemTable中，查询效率高
- **索引优化**: 每个SSTable都有索引，支持快速定位
- **缓存机制**: 多级缓存提升随机读性能

这种架构设计使得LevelDB在保持简单性的同时，实现了出色的写入性能和合理的读取性能，特别适合写多读少的应用场景。
