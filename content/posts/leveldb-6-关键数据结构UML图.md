---
title: "LevelDB 关键数据结构UML图详解"
date: 2025-09-28T00:47:17+08:00
draft: false
tags: ['源码分析', '技术文档', '架构设计']
categories: ['技术分析']
description: "LevelDB 关键数据结构UML图详解的深入技术分析文档"
keywords: ['源码分析', '技术文档', '架构设计']
author: "技术分析师"
weight: 1
---

## 1. 整体类关系图

```mermaid
classDiagram
    class DB {
        <<abstract>>
        +Open(options, name, dbptr) Status
        +Put(options, key, value) Status
        +Get(options, key, value) Status  
        +Delete(options, key) Status
        +Write(options, batch) Status
        +NewIterator(options) Iterator*
        +GetSnapshot() Snapshot*
        +ReleaseSnapshot(snapshot)
        +GetProperty(property, value) bool
        +GetApproximateSizes(ranges, n, sizes)
        +CompactRange(begin, end)
    }
    
    class DBImpl {
        -Env* env_
        -InternalKeyComparator internal_comparator_
        -Options options_
        -string dbname_
        -TableCache* table_cache_
        -FileLock* db_lock_
        -Mutex mutex_
        -MemTable* mem_
        -MemTable* imm_
        -WritableFile* logfile_
        -log::Writer* log_
        -VersionSet* versions_
        -deque~Writer*~ writers_
        -SnapshotList snapshots_
        +Put(options, key, value) Status
        +Get(options, key, value) Status
        +Delete(options, key) Status
        +Write(options, batch) Status
        +NewIterator(options) Iterator*
        +GetSnapshot() Snapshot*
        -MakeRoomForWrite(force) Status
        -CompactMemTable()
        -BackgroundCompaction()
        -NewDB() Status
        -Recover(edit, save_manifest) Status
    }
    
    class MemTable {
        -SkipList~char*, KeyComparator~ table_
        -Arena arena_
        -KeyComparator comparator_
        -int refs_
        +Add(seq, type, key, value)
        +Get(key, value, status) bool
        +NewIterator() Iterator*
        +ApproximateMemoryUsage() size_t
        +Ref()
        +Unref()
    }
    
    class SkipList {
        -Node* head_
        -atomic~int~ max_height_
        -Random rnd_
        -Comparator compare_
        -Arena* arena_
        +Insert(key)
        +Contains(key) bool
        +NewIterator() Iterator*
        -FindGreaterOrEqual(key, prev) Node*
        -FindLessThan(key) Node*
        -RandomHeight() int
    }
    
    class VersionSet {
        -string dbname_
        -Options* options_
        -TableCache* table_cache_
        -InternalKeyComparator icmp_
        -uint64_t next_file_number_
        -uint64_t manifest_file_number_
        -uint64_t last_sequence_
        -Version* current_
        -WritableFile* descriptor_file_
        -log::Writer* descriptor_log_
        +LogAndApply(edit, mutex) Status
        +Recover(save_manifest) Status
        +NewFileNumber() uint64_t
        +PickCompaction() Compaction*
        +MakeInputIterator(c) Iterator*
        +NeedsCompaction() bool
    }
    
    class Version {
        -VersionSet* vset_
        -Version* next_
        -Version* prev_
        -vector~FileMetaData*~ files_[kNumLevels]
        -int refs_
        -FileMetaData* file_to_compact_
        -double compaction_score_
        +Get(options, key, value, stats) Status
        +AddIterators(options, iters)
        +UpdateStats(stats) bool
        +Ref()
        +Unref()
        +NumFiles(level) int
        +OverlapInLevel(level, smallest, largest) bool
    }
    
    class WriteBatch {
        -string rep_
        +Put(key, value)
        +Delete(key)
        +Clear()
        +ApproximateSize() size_t
        +Append(source)
        +Iterate(handler) Status
    }
    
    DB <|-- DBImpl
    DBImpl --> MemTable
    DBImpl --> VersionSet
    DBImpl --> WriteBatch
    MemTable --> SkipList
    VersionSet --> Version
```

## 2. 存储层次结构图

```mermaid
classDiagram
    class StorageLevel {
        <<abstract>>
        +Get(key) Status
        +Put(key, value) Status
        +NewIterator() Iterator*
    }
    
    class MemTable {
        -SkipList~char*, KeyComparator~ table_
        -Arena arena_
        +Get(key, value, status) bool
        +Add(seq, type, key, value)
        +NewIterator() Iterator*
        +ApproximateMemoryUsage() size_t
    }
    
    class ImmutableMemTable {
        -SkipList~char*, KeyComparator~ table_
        -Arena arena_
        +Get(key, value, status) bool
        +NewIterator() Iterator*
        +ApproximateMemoryUsage() size_t
    }
    
    class Level0Files {
        -vector~FileMetaData*~ files_
        +Get(key, value) Status
        +AddIterators(iters)
        +OverlapsWithRange(smallest, largest) bool
    }
    
    class LevelNFiles {
        -vector~FileMetaData*~ files_
        +Get(key, value) Status
        +AddIterators(iters)  
        +FindFile(key) FileMetaData*
        +OverlapsWithRange(smallest, largest) bool
    }
    
    class SSTable {
        -RandomAccessFile* file_
        -Block* index_block_
        -FilterBlockReader* filter_
        -Cache* block_cache_
        +Open(options, file, size, table) Status
        +NewIterator(options) Iterator*
        +InternalGet(options, key, arg, callback) Status
        +ApproximateOffsetOf(key) uint64_t
    }
    
    class TableCache {
        -Cache* cache_
        -Env* env_
        -string dbname_
        -Options* options_
        +NewIterator(options, number, size, tableptr) Iterator*
        +Get(options, number, size, key, arg, callback) Status
        +FindTable(number, size, handle) Status
        +Evict(number)
    }
    
    StorageLevel <|-- MemTable
    StorageLevel <|-- ImmutableMemTable
    StorageLevel <|-- Level0Files
    StorageLevel <|-- LevelNFiles
    
    Level0Files --> SSTable
    LevelNFiles --> SSTable
    SSTable --> TableCache
    TableCache --> SSTable
```

## 3. 迭代器继承体系

```mermaid
classDiagram
    class Iterator {
        <<abstract>>
        +Valid() bool
        +SeekToFirst()
        +SeekToLast()
        +Seek(target)
        +Next()
        +Prev()
        +key() Slice
        +value() Slice
        +status() Status
    }
    
    class MemTableIterator {
        -SkipList::Iterator iter_
        -string tmp_
        +Valid() bool
        +key() Slice
        +value() Slice
        +Next()
        +Prev()
        +Seek(k)
        +SeekToFirst()
        +SeekToLast()
    }
    
    class Block::Iter {
        -Comparator* comparator_
        -const char* data_
        -uint32_t restarts_
        -uint32_t num_restarts_
        -uint32_t current_
        -uint32_t restart_index_
        -string key_
        -Slice value_
        +Valid() bool
        +key() Slice
        +value() Slice
        +Next()
        +Prev()
        +Seek(target)
        -SeekToRestartPoint(index)
        -ParseNextKey() bool
    }
    
    class TwoLevelIterator {
        -Iterator* index_iter_
        -Iterator* data_iter_
        -BlockFunction block_function_
        -void* arg_
        -ReadOptions options_
        +Valid() bool
        +key() Slice
        +value() Slice
        +Next()
        +Prev()
        +Seek(target)
        +SeekToFirst()
        +SeekToLast()
        -InitDataBlock()
        -SetDataIterator(iter)
    }
    
    class MergingIterator {
        -Comparator* comparator_
        -IteratorWrapper* children_
        -int n_
        -IteratorWrapper* current_
        -Direction direction_
        +Valid() bool
        +key() Slice
        +value() Slice
        +Next()
        +Prev()
        +Seek(target)
        +SeekToFirst()
        +SeekToLast()
        -FindSmallest()
        -FindLargest()
    }
    
    class DBIterator {
        -DBImpl* db_
        -Iterator* iter_
        -SequenceNumber sequence_
        -Status status_
        -Slice saved_key_
        -Slice saved_value_
        -Direction direction_
        +Valid() bool
        +key() Slice
        +value() Slice  
        +Next()
        +Prev()
        +Seek(target)
        +SeekToFirst()
        +SeekToLast()
        -FindNextUserEntry(skipping)
        -FindPrevUserEntry()
    }
    
    Iterator <|-- MemTableIterator
    Iterator <|-- Block::Iter
    Iterator <|-- TwoLevelIterator
    Iterator <|-- MergingIterator
    Iterator <|-- DBIterator
```

## 4. 日志系统结构图

```mermaid
classDiagram
    class log::Writer {
        -WritableFile* dest_
        -int block_offset_
        -uint32_t type_crc_[kMaxRecordType + 1]
        +AddRecord(slice) Status
        -EmitPhysicalRecord(type, ptr, length) Status
    }
    
    class log::Reader {
        -SequentialFile* file_
        -Reporter* reporter_
        -bool checksum_
        -char* backing_store_
        -Slice buffer_
        -bool eof_
        -uint64_t last_record_offset_
        -uint64_t end_of_buffer_offset_
        -uint64_t initial_offset_
        +ReadRecord(record, scratch) bool
        -SkipToInitialBlock() bool
        -ReadPhysicalRecord(type) unsigned int
        -ReportCorruption(bytes, reason)
        -ReportDrop(bytes, reason)
    }
    
    class LogFormat {
        <<enumeration>>
        kZeroType
        kFullType
        kFirstType
        kMiddleType
        kLastType
    }
    
    class RecordHeader {
        +uint32_t checksum
        +uint16_t length
        +uint8_t type
    }
    
    log::Writer --> LogFormat
    log::Reader --> LogFormat
    log::Writer --> RecordHeader
    log::Reader --> RecordHeader
```

## 5. 压缩系统结构图

```mermaid
classDiagram
    class Compaction {
        -int level_
        -uint64_t max_output_file_size_
        -Version* input_version_
        -VersionEdit edit_
        -vector~FileMetaData*~ inputs_[2]
        -vector~FileMetaData*~ grandparents_
        -size_t grandparent_index_
        -bool seen_key_
        -int64_t overlapped_bytes_
        +level() int
        +edit() VersionEdit*
        +num_input_files(which) int
        +input(which, i) FileMetaData*
        +MaxOutputFileSize() uint64_t
        +IsTrivialMove() bool
        +AddInputDeletions(edit)
        +IsBaseLevelForKey(user_key) bool
        +ShouldStopBefore(internal_key) bool
        +ReleaseInputs()
    }
    
    class CompactionState {
        -Compaction* compaction
        -SequenceNumber smallest_snapshot
        -vector~Output~ outputs
        -WritableFile* outfile
        -TableBuilder* builder
        -uint64_t total_bytes
        +current_output() Output*
    }
    
    class VersionEdit {
        -string comparator_name_
        -uint64_t log_number_
        -uint64_t prev_log_number_  
        -uint64_t next_file_number_
        -SequenceNumber last_sequence_
        -bool has_comparator_
        -bool has_log_number_
        -bool has_prev_log_number_
        -bool has_next_file_number_
        -bool has_last_sequence_
        -vector~pair~int, InternalKey~~ compact_pointers_
        -DeletedFileSet deleted_files_
        -vector~pair~int, FileMetaData~~ new_files_
        +SetComparatorName(name)
        +SetLogNumber(num)
        +SetNextFile(num)
        +SetLastSequence(seq)
        +AddFile(level, file, smallest, largest)
        +DeleteFile(level, file)
        +EncodeTo(dst)
        +DecodeFrom(src) Status
    }
    
    class FileMetaData {
        +int refs
        +int allowed_seeks
        +uint64_t number
        +uint64_t file_size
        +InternalKey smallest
        +InternalKey largest
    }
    
    Compaction --> VersionEdit
    Compaction --> FileMetaData
    CompactionState --> Compaction
    VersionEdit --> FileMetaData
```

## 6. 缓存系统结构图

```mermaid
classDiagram
    class Cache {
        <<abstract>>
        +Insert(key, value, charge, deleter) Handle*
        +Lookup(key) Handle*
        +Release(handle)
        +Value(handle) void*
        +Erase(key)
        +NewId() uint64_t
        +Prune()
        +TotalCharge() size_t
    }
    
    class LRUCache {
        -HashTable table_
        -LRUHandle lru_
        -LRUHandle in_use_
        -size_t usage_
        -size_t capacity_
        -Mutex mutex_
        +Insert(key, value, charge, deleter) Handle*
        +Lookup(key) Handle*
        +Release(handle)
        +Erase(key)
        +Prune()
        +TotalCharge() size_t
        -FinishErase(e) bool
        -Unref(e)
        -LRU_Remove(e)
        -LRU_Append(list, e)
        -Ref(e)
    }
    
    class LRUHandle {
        +void* value
        +void (*deleter)(const Slice&, void* value)
        +LRUHandle* next_hash
        +LRUHandle* next
        +LRUHandle* prev  
        +size_t charge
        +size_t key_length
        +bool in_cache
        +uint32_t refs
        +uint32_t hash
        +char key_data[1]
        +key() Slice
    }
    
    class HashTable {
        -LRUHandle** list_
        -uint32_t length_
        -uint32_t elems_
        +Lookup(key, hash) LRUHandle*
        +Insert(h) LRUHandle*
        +Remove(key, hash) LRUHandle*
        -FindPointer(key, hash) LRUHandle**
        -Resize()
    }
    
    class TableCache {
        -Cache* cache_
        -Env* env_
        -string dbname_
        -Options* options_
        +NewIterator(options, file_number, file_size, tableptr) Iterator*
        +Get(options, file_number, file_size, key, arg, saver) Status
        +FindTable(file_number, file_size, handle) Status
        +Evict(file_number)
    }
    
    Cache <|-- LRUCache
    LRUCache --> LRUHandle
    LRUCache --> HashTable
    HashTable --> LRUHandle
    TableCache --> Cache
```

## 7. 过滤器系统结构图

```mermaid
classDiagram
    class FilterPolicy {
        <<abstract>>
        +Name() const char*
        +CreateFilter(keys, n, dst)
        +KeyMayMatch(key, filter) bool
    }
    
    class BloomFilterPolicy {
        -size_t bits_per_key_
        -size_t k_
        +CreateFilter(keys, n, dst)
        +KeyMayMatch(key, filter) bool
        +Name() const char*
        -BloomHash(key) uint32_t
    }
    
    class FilterBlockBuilder {
        -FilterPolicy* policy_
        -string keys_
        -vector~size_t~ start_
        -string result_
        -vector~Slice~ tmp_keys_
        -vector~uint32_t~ filter_offsets_
        +StartBlock(block_offset)
        +AddKey(key)
        +Finish() Slice
        -GenerateFilter()
    }
    
    class FilterBlockReader {
        -FilterPolicy* policy_
        -const char* data_
        -const char* offset_
        -size_t num_
        -size_t base_lg_
        +KeyMayMatch(block_offset, key) bool
    }
    
    class InternalFilterPolicy {
        -FilterPolicy* user_policy_
        +CreateFilter(keys, n, dst)
        +KeyMayMatch(key, filter) bool
        +Name() const char*
    }
    
    FilterPolicy <|-- BloomFilterPolicy
    FilterPolicy <|-- InternalFilterPolicy
    FilterBlockBuilder --> FilterPolicy
    FilterBlockReader --> FilterPolicy
    InternalFilterPolicy --> FilterPolicy
```

## 8. 环境抽象层结构图

```mermaid
classDiagram
    class Env {
        <<abstract>>
        +NewSequentialFile(fname, result) Status
        +NewRandomAccessFile(fname, result) Status
        +NewWritableFile(fname, result) Status
        +NewAppendableFile(fname, result) Status
        +FileExists(fname) bool
        +GetChildren(dir, result) Status
        +DeleteFile(fname) Status
        +CreateDir(dirname) Status
        +DeleteDir(dirname) Status
        +GetFileSize(fname, size) Status
        +RenameFile(src, target) Status
        +LockFile(fname, lock) Status
        +UnlockFile(lock) Status
        +Schedule(function, arg)
        +StartThread(function, arg)
        +GetTestDirectory(result) Status
        +NewLogger(fname, result) Status
        +NowMicros() uint64_t
        +SleepForMicroseconds(micros)
    }
    
    class PosixEnv {
        -BackgroundWorkItemQueue queue_
        -Mutex mu_
        -CondVar bgsignal_
        -bool started_bgthread_
        +NewSequentialFile(fname, result) Status
        +NewRandomAccessFile(fname, result) Status
        +NewWritableFile(fname, result) Status
        +FileExists(fname) bool
        +GetChildren(dir, result) Status
        +DeleteFile(fname) Status
        +CreateDir(dirname) Status
        +RenameFile(src, target) Status
        +LockFile(fname, lock) Status
        +UnlockFile(lock) Status
        +Schedule(function, arg)
        +GetTestDirectory(result) Status
        +NewLogger(fname, result) Status
        +NowMicros() uint64_t
        +SleepForMicroseconds(micros)
        -PthreadCall(label, result)
        -BGThread()
    }
    
    class RandomAccessFile {
        <<abstract>>
        +Read(offset, n, result, scratch) Status
    }
    
    class SequentialFile {
        <<abstract>>
        +Read(n, result, scratch) Status
        +Skip(n) Status
    }
    
    class WritableFile {
        <<abstract>>
        +Append(data) Status
        +Close() Status
        +Flush() Status
        +Sync() Status
    }
    
    class FileLock {
        <<abstract>>
    }
    
    class Logger {
        <<abstract>>
        +Logv(format, ap)
    }
    
    Env <|-- PosixEnv
    Env --> RandomAccessFile
    Env --> SequentialFile  
    Env --> WritableFile
    Env --> FileLock
    Env --> Logger
```

## 9. 关键数据结构详细说明

### 9.1 DBImpl核心成员详解

```cpp
class DBImpl : public DB {
 private:
  // 常量，构造后不变
  Env* const env_;                              // 环境抽象层
  const InternalKeyComparator internal_comparator_; // 内部键比较器
  const InternalFilterPolicy internal_filter_policy_; // 内部过滤器策略
  const Options options_;                       // 配置选项
  const std::string dbname_;                    // 数据库名称
  
  // 缓存和锁
  TableCache* const table_cache_;               // SSTable缓存
  FileLock* db_lock_;                          // 文件锁
  
  // 状态保护 (mutex_保护)
  port::Mutex mutex_;                          // 主要互斥锁
  std::atomic<bool> shutting_down_;            // 关闭标志
  port::CondVar background_work_finished_signal_; // 后台工作完成信号
  
  // 内存表
  MemTable* mem_;                              // 当前内存表
  MemTable* imm_;                              // 不可变内存表
  std::atomic<bool> has_imm_;                  // 是否有不可变内存表
  
  // 日志
  WritableFile* logfile_;                      // 日志文件
  uint64_t logfile_number_;                    // 日志文件编号
  log::Writer* log_;                           // 日志写入器
  
  // 写入管理
  std::deque<Writer*> writers_;                // 写入者队列
  WriteBatch* tmp_batch_;                      // 临时批处理
  
  // 快照管理
  SnapshotList snapshots_;                     // 快照列表
  
  // 版本管理
  VersionSet* const versions_;                 // 版本集合
  
  // 后台任务管理
  bool background_compaction_scheduled_;       // 是否调度了后台压缩
  ManualCompaction* manual_compaction_;        // 手动压缩
  
  // 错误处理
  Status bg_error_;                           // 后台错误
  
  // 统计信息
  CompactionStats stats_[config::kNumLevels];  // 各级压缩统计
};
```

### 9.2 MemTable内存布局

```cpp
class MemTable {
 private:
  // 键比较器包装
  struct KeyComparator {
    const InternalKeyComparator comparator;
    int operator()(const char* a, const char* b) const;
  };

  typedef SkipList<const char*, KeyComparator> Table;

  KeyComparator comparator_;  // 比较器实例
  int refs_;                  // 引用计数
  Arena arena_;               // 内存分配器
  Table table_;               // 跳表存储

  // 内存中的记录格式：
  // [internal_key_size:varint32][internal_key:internal_key_size]
  // [value_size:varint32][value:value_size]
  //
  // internal_key格式：
  // [user_key][sequence_number:7bytes][type:1byte]
};
```

### 9.3 SSTable在内存中的表示

```cpp
struct Table::Rep {
  Options options;              // 选项配置
  Status status;                // 状态
  RandomAccessFile* file;       // 文件句柄
  uint64_t cache_id;           // 缓存ID
  
  FilterBlockReader* filter;    // 过滤器读取器
  const char* filter_data;      // 过滤器数据
  
  BlockHandle metaindex_handle; // 元索引块句柄
  Block* index_block;          // 索引块
  
  // 文件布局：
  // [data blocks...]
  // [filter block (optional)]
  // [metaindex block]  
  // [index block]
  // [footer: 48 bytes]
};
```

### 9.4 Version版本信息结构

```cpp
class Version {
 private:
  VersionSet* vset_;                    // 所属版本集
  Version* next_;                       // 链表下一个节点  
  Version* prev_;                       // 链表上一个节点
  int refs_;                            // 引用计数

  // 每个级别的文件列表
  std::vector<FileMetaData*> files_[config::kNumLevels];

  // 压缩统计和触发信息
  FileMetaData* file_to_compact_;       // 基于查找统计需要压缩的文件
  int file_to_compact_level_;           // 对应级别
  double compaction_score_;             // 压缩评分
  int compaction_level_;                // 需要压缩的级别
  
  // 文件组织原则：
  // Level 0: 文件可能重叠，按文件编号排序
  // Level 1+: 文件不重叠，按键范围排序
};
```

## 10. 数据结构性能特性

### 10.1 时间复杂度对比表

| 数据结构 | 插入 | 查找 | 删除 | 遍历 | 空间复杂度 |
|----------|------|------|------|------|------------|
| SkipList | O(log n) | O(log n) | O(log n) | O(n) | O(n) |
| SSTable Index | N/A | O(log m) | N/A | O(m) | O(m) |
| LRU Cache | O(1) | O(1) | O(1) | N/A | O(k) |
| Bloom Filter | O(k) | O(k) | N/A | N/A | O(m) |
| Version Files | N/A | O(log f) | N/A | O(f) | O(f) |

注：n=键数量，m=块数量，k=哈希函数数量，f=文件数量

### 10.2 内存使用分析

```cpp
// MemTable内存开销估算
sizeof(MemTable) = 48字节 (对象本身)
+ sizeof(Arena) ≈ 32字节 (Arena对象)  
+ sizeof(SkipList) ≈ 56字节 (跳表对象)
+ 数据存储 ≈ 1.33 * 原始数据大小 (跳表平均指针开销)
+ Arena块开销 ≈ 每4KB块8字节额外开销

// SSTable内存开销估算  
索引块大小 ≈ 文件数量 * 40字节 / 块
过滤器大小 ≈ 键数量 * 10位 / 8 (布隆过滤器)
块缓存开销 ≈ 缓存的块数量 * (块大小 + 64字节LRUHandle)
```

这些UML图和数据结构分析为理解LevelDB的内部工作机制提供了详细的蓝图，有助于深入掌握其设计思想和实现细节。
