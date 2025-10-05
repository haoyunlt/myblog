---
title: "CPython-06-内存管理-深度补充"
date: 2025-10-05T01:01:58+08:00
draft: false
tags:
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - CPython-06-内存管理-深度补充"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# CPython-06-内存管理-深度补充

## 一、垃圾回收核心API源码剖析

### 1.1 collect_with_callback - 分代垃圾回收入口

```c
// Modules/gcmodule.c (约1200行起)

static Py_ssize_t collect_with_callback(PyThreadState *tstate, int generation)
{
    Py_ssize_t result, collected, uncollectable;
    PyGC_Head unreachable;  // 不可达对象链表
    PyGC_Head finalizers;   // 带__del__的对象
    PyGC_Head *young;       // 待回收代
    PyGC_Head *old;         // 晋升目标代
    
    // 1. 合并年轻代到当前代
    if (generation < NUM_GENERATIONS - 1) {
        merge_queues(young, old);
    }
    
    // 2. 标记可达对象
    update_refs(young);      // 初始化gc_refs
    subtract_refs(young);    // 减去内部引用
    
    // 3. 查找不可达对象
    move_unreachable(young, &unreachable);
    
    // 4. 分离finalizer对象
    move_legacy_finalizers(&unreachable, &finalizers);
    move_legacy_finalizer_reachable(&finalizers);
    
    // 5. 清理不可达对象
    collected = gc_list_size(&unreachable);
    delete_garbage(tstate, &unreachable, old);
    
    // 6. 处理finalizers
    handle_legacy_finalizers(tstate, &finalizers, old);
    
    return collected;
}
```

### 1.2 update_refs - 初始化引用计数

```c
// Modules/gcmodule.c

static void update_refs(PyGC_Head *containers)
{
    PyGC_Head *gc = GC_NEXT(containers);
    
    // 遍历所有容器对象
    for (; gc != containers; gc = GC_NEXT(gc)) {
        PyObject *op = FROM_GC(gc);
        
        #ifdef Py_GIL_DISABLED
        // 自由线程构建：使用原子操作
        Py_ssize_t refcnt = _Py_atomic_load_ssize_relaxed(&op->ob_refcnt);
        #else
        // 标准构建：直接读取
        Py_ssize_t refcnt = Py_REFCNT(op);
        #endif
        
        // 将引用计数复制到gc_refs
        // 这是初始"外部"引用数
        _PyGCHead_SET_REFS(gc, refcnt);
    }
}

// 减去内部引用
static void subtract_refs(PyGC_Head *containers)
{
    traverseproc traverse;
    PyGC_Head *gc = GC_NEXT(containers);
    
    for (; gc != containers; gc = GC_NEXT(gc)) {
        PyObject *op = FROM_GC(gc);
        traverse = Py_TYPE(op)->tp_traverse;
        
        // 调用tp_traverse访问所有引用的对象
        (void) traverse(op,
                       (visitproc)visit_decref,
                       (void *)containers);
    }
}

// 访问函数：减少被引用对象的gc_refs
static int visit_decref(PyObject *op, void *parent)
{
    if (_PyObject_IS_GC(op)) {
        PyGC_Head *gc = AS_GC(op);
        
        // 如果对象在本代中，减少其gc_refs
        if (gc_is_collecting(gc)) {
            Py_ssize_t refs = _PyGCHead_REFS(gc);
            if (refs > 0) {
                _PyGCHead_SET_REFS(gc, refs - 1);
            }
        }
    }
    return 0;
}
```

### 1.3 move_unreachable - 标记-清除算法

```c
// Modules/gcmodule.c

static void move_unreachable(PyGC_Head *young, PyGC_Head *unreachable)
{
    // 使用三色标记算法
    // 白色（gc_refs==0）：潜在垃圾
    // 灰色（gc_refs>0但未扫描）：可达但未处理
    // 黑色（gc_refs>0已扫描）：确定可达
    
    PyGC_Head *gc = GC_NEXT(young);
    
    while (gc != young) {
        PyGC_Head *next = GC_NEXT(gc);
        
        if (_PyGCHead_REFS(gc) == 0) {
            // gc_refs为0：白色对象，移到unreachable
            GC_UNLINK(gc);
            GC_LINK(gc, unreachable);
            
            // 标记为TENTATIVELY_UNREACHABLE
            _PyGCHead_SET_REFS(gc, GC_TENTATIVELY_UNREACHABLE);
        }
        else {
            // gc_refs > 0：灰色对象，从它开始扫描
            traverse_and_mark(gc);
        }
        
        gc = next;
    }
}

// 递归标记可达对象
static void traverse_and_mark(PyGC_Head *gc)
{
    PyObject *op = FROM_GC(gc);
    traverseproc traverse = Py_TYPE(op)->tp_traverse;
    
    // 标记为黑色
    _PyGCHead_SET_REFS(gc, GC_REACHABLE);
    
    // 遍历所有引用，标记它们为可达
    (void) traverse(op, (visitproc)visit_reachable, NULL);
}

static int visit_reachable(PyObject *op, void *arg)
{
    if (!_PyObject_IS_GC(op)) {
        return 0;
    }
    
    PyGC_Head *gc = AS_GC(op);
    Py_ssize_t refs = _PyGCHead_REFS(gc);
    
    if (refs == GC_TENTATIVELY_UNREACHABLE) {
        // 之前标记为不可达，现在发现可达
        // 将其移回young代
        GC_UNLINK(gc);
        GC_LINK(gc, young);
        _PyGCHead_SET_REFS(gc, 1);
        
        // 递归扫描
        traverse_and_mark(gc);
    }
    else if (refs == 0) {
        // 首次发现可达
        _PyGCHead_SET_REFS(gc, 1);
    }
    
    return 0;
}
```

## 二、内存分配器完整剖析

### 2.1 PyObject_Malloc - 内存分配入口

```c
// Objects/obmalloc.c

void* PyObject_Malloc(size_t size)
{
    #ifdef Py_GIL_DISABLED
    // 自由线程构建：使用mimalloc
    return _PyObject_Malloc_Mimalloc(size);
    #else
    // 标准构建：使用pymalloc
    return _PyObject_Malloc_Pymalloc(size);
    #endif
}

// pymalloc分配器
static void* _PyObject_Malloc_Pymalloc(size_t nbytes)
{
    // 1. 检查大小
    if (nbytes == 0) {
        return NULL;
    }
    
    if (nbytes > SMALL_REQUEST_THRESHOLD) {
        // 大对象：直接使用系统malloc
        return PyMem_RawMalloc(nbytes);
    }
    
    // 2. 计算size class
    size_t size = (nbytes - 1) >> ALIGNMENT_SHIFT;
    poolp pool;
    block *bp;
    
    // 3. 从线程本地缓存获取
    poolp *usedpools = get_usedpools();
    pool = usedpools[size];
    
    if (pool != pool->nextpool) {
        // 有可用的pool
        bp = pool->freeblock;
        
        if (bp != NULL) {
            // pool中有空闲块
            pool->freeblock = *(block **)bp;
            
            if (--pool->ref.count == 0) {
                // pool变空，移到空pool链表
                unlink_pool(pool, usedpools);
            }
            
            return (void *)bp;
        }
        
        // pool满了，尝试获取新块
        bp = allocate_from_new_pool(pool);
        if (bp != NULL) {
            return (void *)bp;
        }
    }
    
    // 4. 分配新arena
    arena_object *arenaobj = new_arena();
    if (arenaobj == NULL) {
        return NULL;
    }
    
    // 5. 从新arena分配pool
    pool = arenaobj->freepools;
    arenaobj->freepools = pool->nextpool;
    
    // 初始化pool
    init_pool(pool, size);
    
    // 6. 从新pool分配
    bp = pool->freeblock;
    pool->freeblock = *(block **)bp;
    
    return (void *)bp;
}
```

### 2.2 PyObject_Free - 内存释放

```c
// Objects/obmalloc.c

void PyObject_Free(void *ptr)
{
    if (ptr == NULL) {
        return;
    }
    
    #ifdef Py_GIL_DISABLED
    _PyObject_Free_Mimalloc(ptr);
    return;
    #endif
    
    // 1. 定位pool
    poolp pool = POOL_ADDR(ptr);
    
    // 2. 检查pool header
    if (!address_in_range(ptr, pool)) {
        // 不在pymalloc管理的内存中，使用系统free
        PyMem_RawFree(ptr);
        return;
    }
    
    // 3. 释放到pool的freeblock链表
    block *lastfree = pool->freeblock;
    *(block **)ptr = lastfree;
    pool->freeblock = (block *)ptr;
    
    // 4. 更新pool状态
    if (lastfree == NULL) {
        // pool从满变为部分满
        insert_to_usedpool(pool);
    }
    
    size_t size = pool->szidx;
    
    if (++pool->ref.count == pool->maxnextoffset) {
        // pool变空
        
        // 从usedpools移除
        unlink_pool(pool, get_usedpools());
        
        // 归还给arena
        pool->nextpool = pool->arenaindex->freepools;
        pool->arenaindex->freepools = pool;
        
        // 检查arena是否完全空闲
        arena_object *arenaobj = pool->arenaindex;
        arenaobj->nfreepools++;
        
        if (arenaobj->nfreepools == arenaobj->ntotalpools) {
            // arena完全空闲，释放给系统
            free_arena(arenaobj);
        }
    }
}
```

## 三、UML类图

### 3.1 GC系统类图

```mermaid
classDiagram
    class PyGC_Head {
        +PyGC_Head* _gc_next
        +PyGC_Head* _gc_prev
        +Py_ssize_t _gc_refs
        +uintptr_t _gc_next_masked
    }

    class GC_Generation {
        +PyGC_Head head
        +int threshold
        +int count
    }

    class PyObject_GC {
        +PyObject_HEAD
        +PyGC_Head gc
        +具体对象数据
    }

    class GCState {
        +GC_Generation generations[3]
        +GC_Generation permanent_generation
        +Py_ssize_t long_lived_total
        +int enabled
        +int collecting
        +PyObject* garbage
        +PyObject* callbacks
    }

    class Arena {
        +uintptr_t address
        +poolp freepools
        +size_t nfreepools
        +size_t ntotalpools
    }

    class Pool {
        +poolp nextpool
        +poolp prevpool
        +uint szidx
        +uint nextoffset
        +uint maxnextoffset
        +block* freeblock
        +struct arena_object* arenaindex
    }

    class Block {
        +block* next
        +uint8_t data[]
    }

    GCState "1" *-- "4" GC_Generation: generations
    GC_Generation "1" *-- "*" PyGC_Head: head
    PyObject_GC --|> PyGC_Head: contains
    Arena "1" *-- "*" Pool: freepools
    Pool "1" *-- "*" Block: freeblock
```

### 3.2 内存分配器层次UML

```mermaid
classDiagram
    class MemoryAPI {
        <<interface>>
        +malloc(size)
        +calloc(n, size)
        +realloc(ptr, size)
        +free(ptr)
    }

    class PyMem_RawAPI {
        +PyMem_RawMalloc()
        +PyMem_RawCalloc()
        +PyMem_RawRealloc()
        +PyMem_RawFree()
    }

    class PyMem_API {
        +PyMem_Malloc()
        +PyMem_Calloc()
        +PyMem_Realloc()
        +PyMem_Free()
    }

    class PyObject_API {
        +PyObject_Malloc()
        +PyObject_Calloc()
        +PyObject_Realloc()
        +PyObject_Free()
    }

    class Pymalloc {
        +arenas: Arena[]
        +usedpools: Pool[]
        +size_classes[64]
        +allocate_block()
        +free_block()
    }

    class Mimalloc {
        +heap: mi_heap_t
        +thread_local_cache
        +mi_malloc()
        +mi_free()
    }

    class SystemMalloc {
        +malloc()
        +free()
        +mmap()
        +munmap()
    }

    MemoryAPI <|.. PyMem_RawAPI
    PyMem_RawAPI <|-- PyMem_API
    PyMem_API <|-- PyObject_API
    PyObject_API --> Pymalloc: 标准构建
    PyObject_API --> Mimalloc: 自由线程构建
    Pymalloc --> SystemMalloc: 大对象
    Mimalloc --> SystemMalloc: 底层
```

## 四、详细时序图

### 4.1 垃圾回收完整流程

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant Alloc as 内存分配器
    participant GC as 垃圾回收器
    participant Gen0 as 第0代
    participant Gen1 as 第1代
    participant Gen2 as 第2代
    participant Finalizer as Finalizer处理器

    App->>Alloc: PyObject_GC_New()
    Alloc->>Gen0: 添加对象到第0代
    Gen0->>Gen0: count++
    
    alt count > threshold
        Gen0->>GC: 触发垃圾回收
        
        GC->>Gen0: update_refs()
        Note over Gen0: 初始化gc_refs<br/>= ob_refcnt
        
        GC->>Gen0: subtract_refs()
        Note over Gen0: 遍历对象<br/>减去内部引用
        
        GC->>Gen0: move_unreachable()
        Note over Gen0: 三色标记算法<br/>白色→unreachable<br/>灰色→扫描<br/>黑色→可达
        
        GC->>Finalizer: move_legacy_finalizers()
        Finalizer->>Finalizer: 分离带__del__的对象
        
        GC->>Gen0: delete_garbage()
        Note over Gen0: 调用tp_clear<br/>清除循环引用
        
        GC->>Gen0: 调用tp_dealloc
        Note over Gen0: 释放内存
        
        GC->>Gen1: 晋升幸存对象
        Note over Gen1: 合并到上一代
        
        alt 触发第1代回收
            GC->>Gen1: collect_generation(1)
            Gen1->>Gen2: 晋升幸存对象
        end
        
        Finalizer->>App: 调用__del__方法
        
        GC-->>App: 返回回收对象数
    end
```

### 4.2 对象分配与释放时序

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用代码
    participant API as PyObject_Malloc
    participant Cache as 线程缓存
    participant Pool as Pool管理器
    participant Arena as Arena管理器
    participant System as 系统malloc

    App->>API: 分配64字节
    API->>API: 计算size class
    Note over API: size = (64-1) >> 3 = 7
    
    API->>Cache: 查找usedpools[7]
    
    alt pool有空闲块
        Cache-->>API: pool
        API->>Pool: 获取freeblock
        Pool-->>API: block指针
        API-->>App: 返回内存
        
    else pool已满
        Cache->>Arena: 请求新pool
        
        alt arena有空闲pool
            Arena-->>Cache: 分配pool
            Cache->>Pool: 初始化pool
            Pool-->>API: block指针
            API-->>App: 返回内存
            
        else 需要新arena
            Arena->>System: mmap(256KB)
            System-->>Arena: arena地址
            Arena->>Arena: 初始化pools
            Arena-->>Cache: 分配pool
            Cache->>Pool: 初始化pool
            Pool-->>API: block指针
            API-->>App: 返回内存
        end
    end
    
    Note over App: 对象使用中...
    
    App->>API: PyObject_Free(ptr)
    API->>Pool: 定位pool
    Pool-->>API: pool指针
    
    API->>Pool: 归还block到freeblock链
    Pool->>Pool: ref.count++
    
    alt pool变空
        Pool->>Arena: 归还pool
        Arena->>Arena: nfreepools++
        
        alt arena完全空闲
            Arena->>System: munmap(arena)
            Note over System: 释放256KB给系统
        end
    end
```

### 4.3 循环引用检测时序

```mermaid
sequenceDiagram
    autonumber
    participant A as 对象A
    participant B as 对象B
    participant C as 对象C
    participant GC as 垃圾回收器
    participant Unreachable as unreachable链表

    Note over A,C: 初始状态<br/>A.refcnt=2, B.refcnt=1, C.refcnt=1<br/>A→B, B→C, C→A (循环)
    
    GC->>A: update_refs()
    Note over A: A.gc_refs = 2
    GC->>B: update_refs()
    Note over B: B.gc_refs = 1
    GC->>C: update_refs()
    Note over C: C.gc_refs = 1
    
    GC->>A: subtract_refs()
    A->>B: visit(B)
    Note over B: B.gc_refs--<br/>= 0
    
    GC->>B: subtract_refs()
    B->>C: visit(C)
    Note over C: C.gc_refs--<br/>= 0
    
    GC->>C: subtract_refs()
    C->>A: visit(A)
    Note over A: A.gc_refs--<br/>= 1
    
    Note over A,C: 减法后状态<br/>A.gc_refs=1 (有外部引用)<br/>B.gc_refs=0, C.gc_refs=0
    
    GC->>GC: move_unreachable()
    
    GC->>A: 检查gc_refs
    Note over A: gc_refs=1 > 0<br/>标记为可达
    GC->>A: traverse_and_mark()
    A->>B: visit(B)
    B->>Unreachable: 从unreachable移回
    Note over B: 标记为可达
    B->>C: visit(C)
    C->>Unreachable: 从unreachable移回
    Note over C: 标记为可达
    C->>A: visit(A)
    Note over A: 已标记，跳过
    
    GC->>Unreachable: 检查unreachable链表
    Note over Unreachable: 空！所有对象可达
    
    Note over A,C: 结果：A有外部引用<br/>A→B→C循环可达<br/>无对象被回收
```

## 五、完整函数调用链

### 5.1 对象创建到GC跟踪

```
PyList_New()                           // Objects/listobject.c:179
  └─> _PyObject_GC_New()              // Modules/gcmodule.c:2285
        ├─> PyObject_Malloc()          // Objects/obmalloc.c:702
        │     └─> _PyObject_Malloc()    // Objects/obmalloc.c:2188
        │           └─> pymalloc_alloc() // Objects/obmalloc.c:1485
        │                 ├─> usedpool_nextblock() // Objects/obmalloc.c:1425
        │                 └─> allocate_from_new_pool() // Objects/obmalloc.c:1397
        │
        └─> _PyObject_GC_Link()        // Modules/gcmodule.c:2256
              └─> gc_list_append()      // Modules/gcmodule.c:194
                    └─> _PyGCHead_SET_NEXT() // Include/internal/pycore_gc.h:45
```

### 5.2 垃圾回收触发链

```
PyObject_GC_New()                      // Modules/gcmodule.c:2285
  └─> _PyObject_GC_Alloc()            // Modules/gcmodule.c:2265
        └─> gc_alloc()                 // Modules/gcmodule.c:2241
              ├─> generation->count++
              │
              └─> _PyObject_GC_TRACK() // Include/internal/pycore_gc.h:145
                    └─> [检查阈值]
                          └─> _PyGC_Collect()      // Modules/gcmodule.c:1450
                                └─> collect_with_callback() // Modules/gcmodule.c:1332
                                      ├─> gc_collect_main()     // Modules/gcmodule.c:1250
                                      │     ├─> update_refs()        // Modules/gcmodule.c:520
                                      │     ├─> subtract_refs()      // Modules/gcmodule.c:531
                                      │     ├─> move_unreachable()   // Modules/gcmodule.c:635
                                      │     ├─> move_legacy_finalizers() // Modules/gcmodule.c:774
                                      │     ├─> delete_garbage()     // Modules/gcmodule.c:1103
                                      │     │     └─> clear_weakrefs()      // Modules/gcmodule.c:1065
                                      │     │           └─> PyObject_ClearWeakRefs() // Objects/weakrefobject.c:956
                                      │     │
                                      │     └─> handle_legacy_finalizers() // Modules/gcmodule.c:1189
                                      │           └─> Py_DECREF()           // Include/object.h:604
                                      │                 └─> tp_dealloc()
                                      │
                                      └─> invoke_gc_callback()  // Modules/gcmodule.c:1307
```

### 5.3 对象释放链

```
Py_DECREF(obj)                        // Include/object.h:604
  └─> [--obj->ob_refcnt == 0]
        └─> _Py_Dealloc()             // Objects/object.c:2394
              └─> destructor dealloc = Py_TYPE(obj)->tp_dealloc
                    └─> list_dealloc()        // Objects/listobject.c:369
                          ├─> PyObject_GC_UnTrack() // Modules/gcmodule.c:2301
                          │     └─> _PyObject_GC_UNTRACK() // Include/internal/pycore_gc.h:163
                          │           └─> gc_list_remove()   // Modules/gcmodule.c:202
                          │
                          ├─> [清理列表项]
                          │     └─> Py_XDECREF(item)   // [递归]
                          │
                          └─> PyObject_GC_Del()      // Modules/gcmodule.c:2323
                                └─> PyObject_Free()    // Objects/obmalloc.c:737
                                      └─> pymalloc_free()  // Objects/obmalloc.c:1688
                                            ├─> pool_is_in_list()  // Objects/obmalloc.c:1554
                                            ├─> *(block **)p = pool->freeblock
                                            └─> pool->freeblock = (block *)p
```

## 六、架构图

### 6.1 内存管理整体架构

```mermaid
flowchart TB
    subgraph Application["应用层"]
        PyListNew["PyList_New()"]
        PyDictNew["PyDict_New()"]
        PyObjectNew["PyObject_New()"]
    end
    
    subgraph PublicAPI["公共API层"]
        PyObjectMalloc["PyObject_Malloc()"]
        PyObjectFree["PyObject_Free()"]
        PyMemMalloc["PyMem_Malloc()"]
        PyMemFree["PyMem_Free()"]
    end
    
    subgraph Allocator["分配器层"]
        direction LR
        
        subgraph Pymalloc["Pymalloc (标准构建)"]
            SmallObject["小对象<br/>(≤512字节)"]
            LargeObject["大对象<br/>(>512字节)"]
        end
        
        subgraph Mimalloc["Mimalloc (自由线程)"]
            MiHeap["线程本地堆"]
            MiSegment["段管理"]
        end
    end
    
    subgraph PoolMgr["Pool管理"]
        UsedPools["usedpools[64]<br/>(按size class)"]
        EmptyPools["empty pools"]
        FullPools["full pools"]
    end
    
    subgraph ArenaMgr["Arena管理"]
        Arenas["arenas数组"]
        UnusedArenas["unused_arena_objects"]
        UsableArenas["usable_arenas"]
    end
    
    subgraph System["系统层"]
        Malloc["malloc/free"]
        Mmap["mmap/munmap"]
    end
    
    subgraph GC["垃圾回收器"]
        Gen0["第0代<br/>(新对象)"]
        Gen1["第1代<br/>(中年对象)"]
        Gen2["第2代<br/>(老年对象)"]
        Permanent["永久代"]
    end
    
    Application --> PublicAPI
    PyObjectNew -.GC跟踪.-> GC
    
    PublicAPI --> Allocator
    
    SmallObject --> PoolMgr
    LargeObject --> System
    
    MiHeap --> System
    
    PoolMgr --> ArenaMgr
    ArenaMgr --> Mmap
    
    Gen0 -.晋升.-> Gen1
    Gen1 -.晋升.-> Gen2
    Gen2 -.晋升.-> Permanent
    
    GC -.回收.-> PyObjectFree
    
    style Application fill:#90EE90
    style PublicAPI fill:#FFD700
    style Allocator fill:#87CEEB
    style PoolMgr fill:#DDA0DD
    style ArenaMgr fill:#F0E68C
    style System fill:#FFB6C1
    style GC fill:#FFA07A
```

### 6.2 Pymalloc三层结构

```mermaid
flowchart TB
    subgraph Layer1["Arena层 (256KB)"]
        Arena1["Arena 1"]
        Arena2["Arena 2"]
        Arena3["Arena 3"]
    end
    
    subgraph Layer2["Pool层 (4KB)"]
        direction TB
        subgraph Arena1Detail["Arena 1 内部"]
            Pool1_1["Pool[0]<br/>8字节块"]
            Pool1_2["Pool[1]<br/>16字节块"]
            Pool1_3["Pool[2]<br/>24字节块"]
            Pool1_N["Pool[N]<br/>..."]
        end
    end
    
    subgraph Layer3["Block层"]
        direction TB
        subgraph PoolDetail["Pool 内部"]
            Block1["Block 1"]
            Block2["Block 2"]
            Block3["Block 3"]
            FreeList["freeblock链表"]
            BlockN["Block N"]
        end
    end
    
    Arena1 -.包含64个pools.-> Pool1_1
    Pool1_1 -.包含多个相同大小的blocks.-> Block1
    Block1 --> FreeList
    Block2 --> FreeList
    FreeList --> Block3
    
    style Layer1 fill:#FFE4B5
    style Layer2 fill:#E0FFFF
    style Layer3 fill:#F0E68C
```

## 七、性能优化技巧

### 7.1 对象池技术

```c
// Objects/listobject.c

#ifndef PyList_MAXFREELIST
#  define PyList_MAXFREELIST 80
#endif

static PyListObject *free_list[PyList_MAXFREELIST];
static int numfree = 0;

PyObject* PyList_New(Py_ssize_t size)
{
    PyListObject *op;
    
    // 从对象池获取
    if (numfree) {
        numfree--;
        op = free_list[numfree];
        _Py_NewReference((PyObject *)op);
    }
    else {
        // 对象池空，分配新对象
        op = PyObject_GC_New(PyListObject, &PyList_Type);
        if (op == NULL)
            return NULL;
    }
    
    // 初始化...
    return (PyObject *) op;
}

static void list_dealloc(PyListObject *op)
{
    // 归还到对象池
    if (numfree < PyList_MAXFREELIST && PyList_CheckExact(op)) {
        free_list[numfree++] = op;
    }
    else {
        // 对象池满，真正释放
        PyObject_GC_Del(op);
    }
}
```

### 7.2 GC调优建议

```python
import gc

# 1. 调整GC阈值
gc.set_threshold(
    700,    # gen0阈值（默认700）
    10,     # gen1阈值（默认10）
    10      # gen2阈值（默认10）
)

# 2. 长生命周期对象优化
large_data = create_large_data_structure()
gc.collect()  # 触发full GC
# large_data现在在第2代，减少扫描频率

# 3. 关键路径禁用GC
gc.disable()
# 执行关键计算
result = compute_intensive_task()
gc.enable()

# 4. 手动清理循环引用
class Node:
    def __init__(self):
        self.ref = None
    
    def __del__(self):
        self.ref = None  # 显式打破循环

# 5. 监控GC统计
stats = gc.get_stats()
print(f"Gen0 collections: {stats[0]['collections']}")
print(f"Gen0 collected: {stats[0]['collected']}")
print(f"Gen0 uncollectable: {stats[0]['uncollectable']}")
```

---

本文档详细剖析了CPython的内存管理与垃圾回收机制，涵盖引用计数、分代GC、内存分配器等核心技术。理解这些机制对于编写高性能Python代码和优化内存使用至关重要。

---

## 深度补充文档

本模块的详细API源码分析、完整UML图、详细时序图、完整函数调用链和架构图请参阅：

**[CPython-06-深度补充文档](CPython-06-*-深度补充.md)**

深度补充内容包括：
- ✅ 核心API完整源码剖析（带详细注释）
- ✅ 多层次UML类图（数据结构、关系图）
- ✅ 完整执行流程时序图
- ✅ 端到端函数调用链追踪
- ✅ 模块内部架构流程图
- ✅ 性能优化技术详解
- ✅ 最佳实践与调试技巧


