# LevelDB 实战经验和最佳实践

## 1. 性能调优最佳实践

### 1.1 写入性能优化

#### 批量写入策略
```cpp
// ❌ 错误做法：频繁的单个写入
for (int i = 0; i < 10000; i++) {
    std::string key = "key" + std::to_string(i);
    std::string value = "value" + std::to_string(i);
    db->Put(leveldb::WriteOptions(), key, value);  // 每次都需要获取锁和写日志
}

// ✅ 正确做法：使用WriteBatch批量写入
leveldb::WriteBatch batch;
for (int i = 0; i < 10000; i++) {
    std::string key = "key" + std::to_string(i);
    std::string value = "value" + std::to_string(i);
    batch.Put(key, value);
    
    // 每1000条提交一次，避免单个batch过大
    if (i % 1000 == 999) {
        db->Write(leveldb::WriteOptions(), &batch);
        batch.Clear();
    }
}
// 提交剩余的数据
if (batch.ApproximateSize() > 0) {
    db->Write(leveldb::WriteOptions(), &batch);
}

// 性能提升：批量写入比单个写入快10-50倍
```

#### 异步写入配置
```cpp
// 高吞吐场景的写入配置
leveldb::WriteOptions write_options;
write_options.sync = false;  // 不强制fsync，提升写入性能

// 关键数据的写入配置
leveldb::WriteOptions safe_write_options;
safe_write_options.sync = true;  // 强制fsync，确保数据持久化

// 实际应用中的策略：
class DatabaseWriter {
private:
    leveldb::DB* db_;
    std::atomic<int> pending_writes_{0};
    
public:
    void WriteBatch(const std::vector<std::pair<std::string, std::string>>& data) {
        leveldb::WriteBatch batch;
        for (const auto& kv : data) {
            batch.Put(kv.first, kv.second);
        }
        
        leveldb::WriteOptions options;
        // 每100次写入或积累超过1MB时执行一次同步
        if (++pending_writes_ % 100 == 0 || batch.ApproximateSize() > 1024 * 1024) {
            options.sync = true;
            pending_writes_ = 0;
        } else {
            options.sync = false;
        }
        
        db_->Write(options, &batch);
    }
};
```

#### MemTable大小优化
```cpp
leveldb::Options options;

// 默认配置：4MB MemTable
options.write_buffer_size = 4 * 1024 * 1024;

// 高写入负载场景：增大MemTable
options.write_buffer_size = 64 * 1024 * 1024;  // 64MB
// 优点：减少Minor Compaction频率，提高写入吞吐量  
// 缺点：占用更多内存，恢复时间变长

// 内存受限场景：减小MemTable  
options.write_buffer_size = 1 * 1024 * 1024;   // 1MB
// 优点：减少内存占用
// 缺点：频繁触发压缩，可能影响写入性能

// 监控MemTable切换频率
void MonitorMemTableSwitching() {
    std::string stats;
    db->GetProperty("leveldb.stats", &stats);
    
    // 查看统计信息中的Level 0文件数
    // 如果Level 0文件频繁增长，说明MemTable切换过于频繁
    std::cout << "Database Stats:\n" << stats << std::endl;
}
```

### 1.2 读取性能优化

#### Block Cache配置
```cpp
#include "leveldb/cache.h"

leveldb::Options options;

// 为随机读负载配置大容量缓存
leveldb::Cache* cache = leveldb::NewLRUCache(512 * 1024 * 1024);  // 512MB
options.block_cache = cache;

// 为顺序读负载配置小容量缓存  
leveldb::Cache* small_cache = leveldb::NewLRUCache(16 * 1024 * 1024);  // 16MB
options.block_cache = small_cache;

// 实际场景中的动态缓存配置
class AdaptiveCacheManager {
private:
    leveldb::Cache* cache_;
    std::atomic<uint64_t> hit_count_{0};
    std::atomic<uint64_t> miss_count_{0};
    
public:
    AdaptiveCacheManager(size_t initial_size) 
        : cache_(leveldb::NewLRUCache(initial_size)) {}
    
    void RecordCacheAccess(bool hit) {
        if (hit) {
            hit_count_++;
        } else {
            miss_count_++;
        }
        
        // 每10000次访问后分析缓存效率
        uint64_t total = hit_count_ + miss_count_;
        if (total % 10000 == 0) {
            double hit_rate = static_cast<double>(hit_count_) / total;
            std::cout << "Cache hit rate: " << hit_rate * 100 << "%" << std::endl;
            
            // 根据命中率调整缓存策略
            if (hit_rate < 0.7) {  // 命中率低于70%
                std::cout << "Consider increasing cache size or using bloom filters" << std::endl;
            }
        }
    }
};
```

#### 布隆过滤器优化
```cpp
#include "leveldb/filter_policy.h"

// 为点查询优化配置布隆过滤器
const leveldb::FilterPolicy* filter = leveldb::NewBloomFilterPolicy(10);
options.filter_policy = filter;

// 不同场景的过滤器配置
class FilterPolicyManager {
public:
    // 高精度过滤器：适合点查询频繁的场景
    static const leveldb::FilterPolicy* CreateHighPrecisionFilter() {
        return leveldb::NewBloomFilterPolicy(16);  // 每个键16位，假阳性率约0.1%
    }
    
    // 标准过滤器：平衡性能和空间
    static const leveldb::FilterPolicy* CreateStandardFilter() {
        return leveldb::NewBloomFilterPolicy(10);  // 每个键10位，假阳性率约1%
    }
    
    // 节省空间过滤器：适合存储空间受限的场景
    static const leveldb::FilterPolicy* CreateCompactFilter() {
        return leveldb::NewBloomFilterPolicy(6);   // 每个键6位，假阳性率约5%
    }
    
    // 根据访问模式选择过滤器
    static const leveldb::FilterPolicy* SelectFilter(double point_query_ratio) {
        if (point_query_ratio > 0.8) {
            return CreateHighPrecisionFilter();
        } else if (point_query_ratio > 0.3) {
            return CreateStandardFilter();
        } else {
            return CreateCompactFilter();
        }
    }
};
```

#### 迭代器使用最佳实践
```cpp
// ❌ 错误做法：每次查询都创建新迭代器
std::vector<std::string> GetRangeValues(const std::string& start_key, 
                                       const std::string& end_key) {
    std::vector<std::string> results;
    for (std::string key = start_key; key <= end_key; /* increment key */) {
        leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
        it->Seek(key);
        if (it->Valid()) {
            results.push_back(it->value().ToString());
        }
        delete it;  // 频繁创建销毁，性能很差
    }
    return results;
}

// ✅ 正确做法：重用迭代器进行范围扫描
std::vector<std::string> GetRangeValuesOptimized(const std::string& start_key,
                                                const std::string& end_key) {
    std::vector<std::string> results;
    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
    
    for (it->Seek(start_key); it->Valid() && it->key().ToString() <= end_key; it->Next()) {
        results.push_back(it->value().ToString());
    }
    
    delete it;
    return results;
}

// 高级迭代器使用模式
class IteratorPool {
private:
    std::queue<leveldb::Iterator*> available_iterators_;
    std::mutex mutex_;
    leveldb::DB* db_;
    
public:
    IteratorPool(leveldb::DB* db, size_t pool_size) : db_(db) {
        for (size_t i = 0; i < pool_size; ++i) {
            available_iterators_.push(db_->NewIterator(leveldb::ReadOptions()));
        }
    }
    
    leveldb::Iterator* AcquireIterator() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!available_iterators_.empty()) {
            auto it = available_iterators_.front();
            available_iterators_.pop();
            return it;
        }
        return db_->NewIterator(leveldb::ReadOptions());
    }
    
    void ReleaseIterator(leveldb::Iterator* it) {
        std::lock_guard<std::mutex> lock(mutex_);
        available_iterators_.push(it);
    }
};
```

### 1.3 存储空间优化

#### 压缩配置优化
```cpp
leveldb::Options options;

// 高压缩率配置（适合存储空间敏感的场景）
options.compression = leveldb::kSnappyCompression;  // 默认Snappy压缩
options.block_size = 16 * 1024;  // 减小块大小提高压缩率

// 高性能配置（适合CPU敏感的场景）
options.compression = leveldb::kNoCompression;  // 关闭压缩
options.block_size = 64 * 1024;  // 增大块大小减少索引开销

// 动态压缩策略
class AdaptiveCompression {
public:
    static leveldb::CompressionType SelectCompression(
        double cpu_usage, 
        double storage_usage,
        size_t data_size) {
        
        // CPU使用率高且数据量大时，考虑关闭压缩
        if (cpu_usage > 0.8 && data_size > 100 * 1024 * 1024) {
            return leveldb::kNoCompression;
        }
        
        // 存储空间紧张时，启用压缩
        if (storage_usage > 0.9) {
            return leveldb::kSnappyCompression;
        }
        
        // 默认使用Snappy压缩
        return leveldb::kSnappyCompression;
    }
};
```

#### 手动压缩策略
```cpp
class CompactionManager {
private:
    leveldb::DB* db_;
    std::thread compaction_thread_;
    std::atomic<bool> running_{true};
    
public:
    CompactionManager(leveldb::DB* db) : db_(db) {
        compaction_thread_ = std::thread([this]() {
            CompactionWorker();
        });
    }
    
    void CompactionWorker() {
        while (running_) {
            std::string stats;
            if (db_->GetProperty("leveldb.stats", &stats)) {
                // 解析统计信息，判断是否需要压缩
                if (ShouldTriggerCompaction(stats)) {
                    // 执行全量压缩
                    db_->CompactRange(nullptr, nullptr);
                    std::cout << "Manual compaction completed" << std::endl;
                }
            }
            
            // 每小时检查一次
            std::this_thread::sleep_for(std::chrono::hours(1));
        }
    }
    
private:
    bool ShouldTriggerCompaction(const std::string& stats) {
        // 简单的压缩触发逻辑：检查Level 0文件数
        size_t level0_files = ExtractLevel0FileCount(stats);
        return level0_files > 8;  // 超过8个Level 0文件时触发压缩
    }
    
    size_t ExtractLevel0FileCount(const std::string& stats) {
        // 解析统计信息中的Level 0文件数
        std::regex pattern(R"(Level\s+0:\s+(\d+)\s+files)");
        std::smatch matches;
        if (std::regex_search(stats, matches, pattern)) {
            return std::stoi(matches[1]);
        }
        return 0;
    }
};
```

## 2. 架构设计模式实践

### 2.1 多数据库实例管理

```cpp
class DatabaseManager {
private:
    std::map<std::string, std::unique_ptr<leveldb::DB>> databases_;
    std::map<std::string, std::unique_ptr<leveldb::Cache>> caches_;
    std::mutex mutex_;
    
public:
    leveldb::DB* GetDatabase(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = databases_.find(name);
        if (it != databases_.end()) {
            return it->second.get();
        }
        
        // 创建新的数据库实例
        leveldb::DB* db;
        leveldb::Options options = CreateOptimizedOptions(name);
        leveldb::Status status = leveldb::DB::Open(options, name, &db);
        
        if (status.ok()) {
            databases_[name] = std::unique_ptr<leveldb::DB>(db);
            return db;
        }
        
        return nullptr;
    }
    
private:
    leveldb::Options CreateOptimizedOptions(const std::string& db_name) {
        leveldb::Options options;
        options.create_if_missing = true;
        
        // 为每个数据库创建独立的缓存
        auto cache = std::make_unique<leveldb::Cache*>(
            leveldb::NewLRUCache(128 * 1024 * 1024));  // 128MB
        caches_[db_name] = std::move(cache);
        options.block_cache = caches_[db_name].get();
        
        // 根据数据库名称配置不同的参数
        if (db_name.find("logs") != std::string::npos) {
            // 日志数据库：优化写入性能
            options.write_buffer_size = 64 * 1024 * 1024;  // 64MB
            options.max_file_size = 128 * 1024 * 1024;     // 128MB
        } else if (db_name.find("index") != std::string::npos) {
            // 索引数据库：优化读取性能
            options.block_cache = leveldb::NewLRUCache(512 * 1024 * 1024);  // 512MB
            options.filter_policy = leveldb::NewBloomFilterPolicy(16);
        }
        
        return options;
    }
};
```

### 2.2 数据分片策略

```cpp
class ShardedDatabase {
private:
    std::vector<std::unique_ptr<leveldb::DB>> shards_;
    size_t shard_count_;
    
public:
    ShardedDatabase(const std::string& base_path, size_t shard_count) 
        : shard_count_(shard_count) {
        
        for (size_t i = 0; i < shard_count_; ++i) {
            std::string shard_path = base_path + "/shard_" + std::to_string(i);
            leveldb::DB* db;
            leveldb::Options options;
            options.create_if_missing = true;
            
            // 为每个分片配置独立的缓存
            options.block_cache = leveldb::NewLRUCache(64 * 1024 * 1024);  // 64MB per shard
            
            leveldb::Status status = leveldb::DB::Open(options, shard_path, &db);
            if (status.ok()) {
                shards_.emplace_back(db);
            }
        }
    }
    
    leveldb::Status Put(const std::string& key, const std::string& value) {
        size_t shard_index = HashKey(key) % shard_count_;
        return shards_[shard_index]->Put(leveldb::WriteOptions(), key, value);
    }
    
    leveldb::Status Get(const std::string& key, std::string* value) {
        size_t shard_index = HashKey(key) % shard_count_;
        return shards_[shard_index]->Get(leveldb::ReadOptions(), key, value);
    }
    
    // 范围查询需要查询所有分片
    std::vector<std::pair<std::string, std::string>> GetRange(
        const std::string& start_key, const std::string& end_key) {
        
        std::vector<std::pair<std::string, std::string>> results;
        
        for (auto& shard : shards_) {
            leveldb::Iterator* it = shard->NewIterator(leveldb::ReadOptions());
            
            for (it->Seek(start_key); 
                 it->Valid() && it->key().ToString() <= end_key; 
                 it->Next()) {
                results.emplace_back(it->key().ToString(), it->value().ToString());
            }
            
            delete it;
        }
        
        // 合并和排序结果
        std::sort(results.begin(), results.end());
        return results;
    }
    
private:
    size_t HashKey(const std::string& key) {
        // 使用简单的哈希函数
        size_t hash = 0;
        for (char c : key) {
            hash = hash * 31 + c;
        }
        return hash;
    }
};
```

### 2.3 读写分离架构

```cpp
class ReadWriteSeparatedDB {
private:
    leveldb::DB* write_db_;      // 主写数据库
    std::vector<leveldb::DB*> read_dbs_;  // 多个只读副本
    std::atomic<size_t> read_round_robin_{0};
    
    // 异步同步线程
    std::thread sync_thread_;
    std::atomic<bool> running_{true};
    
public:
    ReadWriteSeparatedDB(const std::string& write_path, 
                        const std::vector<std::string>& read_paths) {
        // 打开主写数据库
        leveldb::Options write_options;
        write_options.create_if_missing = true;
        leveldb::DB::Open(write_options, write_path, &write_db_);
        
        // 打开只读副本
        for (const auto& path : read_paths) {
            leveldb::DB* read_db;
            leveldb::Options read_options;
            read_options.create_if_missing = false;  // 只读，不创建
            if (leveldb::DB::Open(read_options, path, &read_db).ok()) {
                read_dbs_.push_back(read_db);
            }
        }
        
        // 启动同步线程
        sync_thread_ = std::thread([this]() { SyncWorker(); });
    }
    
    leveldb::Status Put(const std::string& key, const std::string& value) {
        return write_db_->Put(leveldb::WriteOptions(), key, value);
    }
    
    leveldb::Status Get(const std::string& key, std::string* value) {
        if (read_dbs_.empty()) {
            return write_db_->Get(leveldb::ReadOptions(), key, value);
        }
        
        // 轮询选择读取副本
        size_t index = read_round_robin_.fetch_add(1) % read_dbs_.size();
        return read_dbs_[index]->Get(leveldb::ReadOptions(), key, value);
    }
    
private:
    void SyncWorker() {
        while (running_) {
            // 简化的同步逻辑：定期触发压缩确保数据同步
            for (auto* read_db : read_dbs_) {
                read_db->CompactRange(nullptr, nullptr);
            }
            
            std::this_thread::sleep_for(std::chrono::minutes(5));
        }
    }
};
```

## 3. 错误处理和监控实践

### 3.1 全面的错误处理

```cpp
class RobustDBWrapper {
private:
    leveldb::DB* db_;
    std::string db_path_;
    leveldb::Options options_;
    
    // 错误统计
    std::atomic<uint64_t> read_errors_{0};
    std::atomic<uint64_t> write_errors_{0};
    std::atomic<uint64_t> corruption_errors_{0};
    
public:
    enum class ErrorSeverity {
        INFO,
        WARNING, 
        ERROR,
        CRITICAL
    };
    
    leveldb::Status Put(const std::string& key, const std::string& value) {
        leveldb::Status status = db_->Put(leveldb::WriteOptions(), key, value);
        
        if (!status.ok()) {
            write_errors_++;
            HandleError("Put operation failed", status, ErrorSeverity::ERROR);
            
            // 对于严重错误，尝试恢复
            if (status.IsCorruption()) {
                corruption_errors_++;
                AttemptRepair();
            }
        }
        
        return status;
    }
    
    leveldb::Status Get(const std::string& key, std::string* value) {
        leveldb::Status status = db_->Get(leveldb::ReadOptions(), key, value);
        
        if (!status.ok() && !status.IsNotFound()) {
            read_errors_++;
            HandleError("Get operation failed", status, ErrorSeverity::WARNING);
        }
        
        return status;
    }
    
private:
    void HandleError(const std::string& operation, 
                    const leveldb::Status& status, 
                    ErrorSeverity severity) {
        
        std::string error_msg = operation + ": " + status.ToString();
        
        // 记录日志
        LogError(error_msg, severity);
        
        // 发送监控指标
        ReportMetric("leveldb_error", 1, {
            {"operation", operation},
            {"severity", SeverityToString(severity)},
            {"error_type", GetErrorType(status)}
        });
        
        // 严重错误时发送告警
        if (severity == ErrorSeverity::CRITICAL) {
            SendAlert(error_msg);
        }
    }
    
    void AttemptRepair() {
        std::cout << "Attempting to repair database at: " << db_path_ << std::endl;
        
        // 关闭当前数据库
        delete db_;
        db_ = nullptr;
        
        // 尝试修复
        leveldb::Status repair_status = leveldb::RepairDB(db_path_, options_);
        if (repair_status.ok()) {
            // 重新打开数据库
            leveldb::Status open_status = leveldb::DB::Open(options_, db_path_, &db_);
            if (open_status.ok()) {
                std::cout << "Database repair successful" << std::endl;
            } else {
                std::cerr << "Failed to reopen repaired database: " 
                         << open_status.ToString() << std::endl;
            }
        } else {
            std::cerr << "Database repair failed: " 
                     << repair_status.ToString() << std::endl;
        }
    }
    
    std::string GetErrorType(const leveldb::Status& status) {
        if (status.IsNotFound()) return "not_found";
        if (status.IsCorruption()) return "corruption";
        if (status.IsIOError()) return "io_error";
        if (status.IsInvalidArgument()) return "invalid_argument";
        return "unknown";
    }
};
```

### 3.2 性能监控系统

```cpp
class DatabaseMonitor {
private:
    leveldb::DB* db_;
    std::thread monitor_thread_;
    std::atomic<bool> running_{true};
    
    // 性能计数器
    std::atomic<uint64_t> total_reads_{0};
    std::atomic<uint64_t> total_writes_{0};
    std::atomic<uint64_t> cache_hits_{0};
    std::atomic<uint64_t> cache_misses_{0};
    
    // 性能历史记录
    std::deque<double> read_latency_history_;
    std::deque<double> write_latency_history_;
    std::mutex history_mutex_;
    
public:
    DatabaseMonitor(leveldb::DB* db) : db_(db) {
        monitor_thread_ = std::thread([this]() { MonitorWorker(); });
    }
    
    void RecordRead(double latency_ms) {
        total_reads_++;
        
        std::lock_guard<std::mutex> lock(history_mutex_);
        read_latency_history_.push_back(latency_ms);
        
        // 保留最近1000次记录
        if (read_latency_history_.size() > 1000) {
            read_latency_history_.pop_front();
        }
    }
    
    void RecordWrite(double latency_ms) {
        total_writes_++;
        
        std::lock_guard<std::mutex> lock(history_mutex_);
        write_latency_history_.push_back(latency_ms);
        
        if (write_latency_history_.size() > 1000) {
            write_latency_history_.pop_front();
        }
    }
    
    void RecordCacheAccess(bool hit) {
        if (hit) {
            cache_hits_++;
        } else {
            cache_misses_++;
        }
    }
    
private:
    void MonitorWorker() {
        while (running_) {
            CollectMetrics();
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
    }
    
    void CollectMetrics() {
        // 收集LevelDB内部统计信息
        std::string stats;
        if (db_->GetProperty("leveldb.stats", &stats)) {
            ParseAndReportStats(stats);
        }
        
        // 收集内存使用信息
        std::string memory_usage;
        if (db_->GetProperty("leveldb.approximate-memory-usage", &memory_usage)) {
            ReportMemoryUsage(std::stoull(memory_usage));
        }
        
        // 收集性能指标
        ReportPerformanceMetrics();
    }
    
    void ReportPerformanceMetrics() {
        std::lock_guard<std::mutex> lock(history_mutex_);
        
        if (!read_latency_history_.empty()) {
            double avg_read_latency = std::accumulate(
                read_latency_history_.begin(), 
                read_latency_history_.end(), 
                0.0) / read_latency_history_.size();
            
            ReportMetric("avg_read_latency_ms", avg_read_latency);
        }
        
        if (!write_latency_history_.empty()) {
            double avg_write_latency = std::accumulate(
                write_latency_history_.begin(),
                write_latency_history_.end(),
                0.0) / write_latency_history_.size();
                
            ReportMetric("avg_write_latency_ms", avg_write_latency);
        }
        
        // 缓存命中率
        uint64_t total_cache_accesses = cache_hits_ + cache_misses_;
        if (total_cache_accesses > 0) {
            double hit_rate = static_cast<double>(cache_hits_) / total_cache_accesses;
            ReportMetric("cache_hit_rate", hit_rate);
        }
        
        // QPS
        static auto last_time = std::chrono::steady_clock::now();
        static uint64_t last_reads = 0;
        static uint64_t last_writes = 0;
        
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_time);
        
        if (duration.count() > 0) {
            double read_qps = static_cast<double>(total_reads_ - last_reads) / duration.count();
            double write_qps = static_cast<double>(total_writes_ - last_writes) / duration.count();
            
            ReportMetric("read_qps", read_qps);
            ReportMetric("write_qps", write_qps);
            
            last_time = now;
            last_reads = total_reads_;
            last_writes = total_writes_;
        }
    }
    
    void ParseAndReportStats(const std::string& stats) {
        // 解析leveldb内部统计信息
        std::istringstream iss(stats);
        std::string line;
        
        while (std::getline(iss, line)) {
            // 解析Level信息
            if (line.find("Level") == 0) {
                std::regex level_pattern(R"(Level\s+(\d+):\s+(\d+)\s+files,\s+(\d+)\s+MB)");
                std::smatch matches;
                
                if (std::regex_search(line, matches, level_pattern)) {
                    int level = std::stoi(matches[1]);
                    int files = std::stoi(matches[2]);
                    int size_mb = std::stoi(matches[3]);
                    
                    ReportMetric("level_files", files, {{"level", std::to_string(level)}});
                    ReportMetric("level_size_mb", size_mb, {{"level", std::to_string(level)}});
                }
            }
        }
    }
    
    void ReportMetric(const std::string& name, double value, 
                     const std::map<std::string, std::string>& tags = {}) {
        // 发送指标到监控系统（如Prometheus、StatsD等）
        std::cout << "Metric: " << name << " = " << value;
        for (const auto& tag : tags) {
            std::cout << " " << tag.first << "=" << tag.second;
        }
        std::cout << std::endl;
    }
};
```

## 4. 生产环境部署指南

### 4.1 配置模板

```cpp
class ProductionConfigBuilder {
public:
    // 高并发读写场景配置
    static leveldb::Options CreateHighConcurrencyConfig() {
        leveldb::Options options;
        
        // 基本配置
        options.create_if_missing = true;
        options.error_if_exists = false;
        options.paranoid_checks = true;  // 生产环境建议开启
        
        // 内存配置
        options.write_buffer_size = 32 * 1024 * 1024;  // 32MB
        options.max_open_files = 2000;  // 增加打开文件数限制
        
        // 缓存配置
        options.block_cache = leveldb::NewLRUCache(256 * 1024 * 1024);  // 256MB
        options.block_size = 32 * 1024;  // 32KB块大小
        
        // 过滤器配置
        options.filter_policy = leveldb::NewBloomFilterPolicy(10);
        
        // 压缩配置
        options.compression = leveldb::kSnappyCompression;
        options.max_file_size = 64 * 1024 * 1024;  // 64MB文件大小
        
        return options;
    }
    
    // 大数据批处理场景配置
    static leveldb::Options CreateBatchProcessingConfig() {
        leveldb::Options options;
        
        options.create_if_missing = true;
        options.paranoid_checks = false;  // 批处理可以关闭以提升性能
        
        // 大内存配置
        options.write_buffer_size = 128 * 1024 * 1024;  // 128MB
        options.max_open_files = 5000;
        
        // 较小的缓存（批处理通常是顺序访问）
        options.block_cache = leveldb::NewLRUCache(64 * 1024 * 1024);  // 64MB
        options.block_size = 64 * 1024;  // 64KB
        
        // 更大的文件大小
        options.max_file_size = 256 * 1024 * 1024;  // 256MB
        
        // 压缩配置
        options.compression = leveldb::kSnappyCompression;
        
        return options;
    }
    
    // 存储空间受限场景配置
    static leveldb::Options CreateSpaceConstrainedConfig() {
        leveldb::Options options;
        
        options.create_if_missing = true;
        options.paranoid_checks = true;
        
        // 小内存配置
        options.write_buffer_size = 8 * 1024 * 1024;   // 8MB
        options.max_open_files = 500;
        
        // 小缓存
        options.block_cache = leveldb::NewLRUCache(32 * 1024 * 1024);  // 32MB
        options.block_size = 16 * 1024;  // 16KB
        
        // 高压缩率配置
        options.compression = leveldb::kSnappyCompression;
        options.max_file_size = 32 * 1024 * 1024;  // 32MB
        options.block_restart_interval = 32;  // 更大的重启间隔提高压缩率
        
        // 布隆过滤器节省IO
        options.filter_policy = leveldb::NewBloomFilterPolicy(12);
        
        return options;
    }
};
```

### 4.2 部署检查清单

```cpp
class DeploymentChecker {
public:
    struct CheckResult {
        bool passed;
        std::string message;
        std::string suggestion;
    };
    
    static std::vector<CheckResult> RunDeploymentChecks(
        const std::string& db_path, 
        const leveldb::Options& options) {
        
        std::vector<CheckResult> results;
        
        // 检查磁盘空间
        results.push_back(CheckDiskSpace(db_path));
        
        // 检查文件描述符限制
        results.push_back(CheckFileDescriptorLimit(options.max_open_files));
        
        // 检查内存配置
        results.push_back(CheckMemoryConfiguration(options));
        
        // 检查权限
        results.push_back(CheckPermissions(db_path));
        
        // 检查性能相关配置
        results.push_back(CheckPerformanceConfiguration(options));
        
        return results;
    }
    
private:
    static CheckResult CheckDiskSpace(const std::string& db_path) {
        // 检查可用磁盘空间
        struct statvfs stat;
        if (statvfs(db_path.c_str(), &stat) == 0) {
            uint64_t available_bytes = stat.f_bavail * stat.f_frsize;
            uint64_t min_required = 10LL * 1024 * 1024 * 1024;  // 10GB
            
            if (available_bytes < min_required) {
                return {false, "Insufficient disk space", 
                       "Ensure at least 10GB free space for database growth"};
            }
        }
        
        return {true, "Disk space check passed", ""};
    }
    
    static CheckResult CheckFileDescriptorLimit(int max_open_files) {
        struct rlimit limit;
        if (getrlimit(RLIMIT_NOFILE, &limit) == 0) {
            if (limit.rlim_cur < max_open_files + 100) {  // +100 for other files
                return {false, "File descriptor limit too low",
                       "Increase ulimit -n to at least " + std::to_string(max_open_files + 100)};
            }
        }
        
        return {true, "File descriptor limit check passed", ""};
    }
    
    static CheckResult CheckMemoryConfiguration(const leveldb::Options& options) {
        // 估算内存使用
        size_t estimated_memory = 
            options.write_buffer_size +  // MemTable
            options.write_buffer_size +  // Immutable MemTable
            (options.block_cache ? options.block_cache->TotalCharge() : 0);
        
        // 获取系统可用内存
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        uint64_t total_memory = pages * page_size;
        
        if (estimated_memory > total_memory * 0.8) {  // 不超过80%系统内存
            return {false, "Memory configuration too high",
                   "Reduce write_buffer_size or block_cache size"};
        }
        
        return {true, "Memory configuration check passed", ""};
    }
};
```

### 4.3 容灾备份策略

```cpp
class BackupManager {
private:
    leveldb::DB* db_;
    std::string backup_dir_;
    std::thread backup_thread_;
    std::atomic<bool> running_{true};
    
public:
    BackupManager(leveldb::DB* db, const std::string& backup_dir) 
        : db_(db), backup_dir_(backup_dir) {
        
        // 创建备份目录
        std::filesystem::create_directories(backup_dir_);
        
        // 启动定期备份线程
        backup_thread_ = std::thread([this]() { BackupWorker(); });
    }
    
    // 创建快照备份
    bool CreateSnapshot() {
        auto timestamp = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(timestamp);
        
        std::stringstream ss;
        ss << backup_dir_ << "/snapshot_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        std::string snapshot_path = ss.str();
        
        try {
            // 获取数据库快照
            const leveldb::Snapshot* snapshot = db_->GetSnapshot();
            
            // 创建备份数据库
            leveldb::DB* backup_db;
            leveldb::Options options;
            options.create_if_missing = true;
            
            leveldb::Status status = leveldb::DB::Open(options, snapshot_path, &backup_db);
            if (!status.ok()) {
                db_->ReleaseSnapshot(snapshot);
                return false;
            }
            
            // 使用快照创建迭代器复制数据
            leveldb::ReadOptions read_options;
            read_options.snapshot = snapshot;
            leveldb::Iterator* it = db_->NewIterator(read_options);
            
            leveldb::WriteBatch batch;
            size_t batch_size = 0;
            const size_t max_batch_size = 10 * 1024 * 1024;  // 10MB批次
            
            for (it->SeekToFirst(); it->Valid(); it->Next()) {
                batch.Put(it->key(), it->value());
                batch_size += it->key().size() + it->value().size();
                
                if (batch_size >= max_batch_size) {
                    backup_db->Write(leveldb::WriteOptions(), &batch);
                    batch.Clear();
                    batch_size = 0;
                }
            }
            
            // 写入剩余数据
            if (batch_size > 0) {
                backup_db->Write(leveldb::WriteOptions(), &batch);
            }
            
            delete it;
            delete backup_db;
            db_->ReleaseSnapshot(snapshot);
            
            std::cout << "Backup created successfully: " << snapshot_path << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Backup failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // 恢复备份
    bool RestoreFromBackup(const std::string& backup_path, const std::string& restore_path) {
        try {
            // 打开备份数据库
            leveldb::DB* backup_db;
            leveldb::Options options;
            leveldb::Status status = leveldb::DB::Open(options, backup_path, &backup_db);
            if (!status.ok()) {
                return false;
            }
            
            // 创建恢复目标数据库
            leveldb::DB* restore_db;
            options.create_if_missing = true;
            options.error_if_exists = true;  // 确保不覆盖现有数据
            
            status = leveldb::DB::Open(options, restore_path, &restore_db);
            if (!status.ok()) {
                delete backup_db;
                return false;
            }
            
            // 复制所有数据
            leveldb::Iterator* it = backup_db->NewIterator(leveldb::ReadOptions());
            leveldb::WriteBatch batch;
            size_t batch_size = 0;
            
            for (it->SeekToFirst(); it->Valid(); it->Next()) {
                batch.Put(it->key(), it->value());
                batch_size += it->key().size() + it->value().size();
                
                if (batch_size >= 10 * 1024 * 1024) {  // 10MB批次
                    restore_db->Write(leveldb::WriteOptions(), &batch);
                    batch.Clear();
                    batch_size = 0;
                }
            }
            
            if (batch_size > 0) {
                restore_db->Write(leveldb::WriteOptions(), &batch);
            }
            
            delete it;
            delete backup_db;
            delete restore_db;
            
            std::cout << "Restore completed successfully" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Restore failed: " << e.what() << std::endl;
            return false;
        }
    }
    
private:
    void BackupWorker() {
        while (running_) {
            CreateSnapshot();
            CleanupOldBackups();
            
            // 每6小时备份一次
            std::this_thread::sleep_for(std::chrono::hours(6));
        }
    }
    
    void CleanupOldBackups() {
        // 保留最近7天的备份
        auto cutoff_time = std::chrono::system_clock::now() - std::chrono::hours(24 * 7);
        
        for (const auto& entry : std::filesystem::directory_iterator(backup_dir_)) {
            if (entry.is_directory()) {
                auto file_time = std::chrono::file_clock::to_sys(entry.last_write_time());
                if (file_time < cutoff_time) {
                    std::filesystem::remove_all(entry.path());
                    std::cout << "Removed old backup: " << entry.path() << std::endl;
                }
            }
        }
    }
};
```

## 5. 总结

通过以上实战经验和最佳实践，我们可以看到：

1. **性能调优**：合理配置MemTable大小、Block Cache、布隆过滤器等参数对性能至关重要
2. **架构设计**：分片、读写分离等设计模式可以有效提升系统的可扩展性
3. **监控运维**：完善的错误处理、性能监控和备份策略是生产环境稳定运行的基础
4. **配置管理**：根据不同的使用场景选择合适的配置参数，避免一刀切的配置方式

LevelDB作为一个高性能的嵌入式数据库，在正确使用的前提下能够为应用提供可靠、高效的数据存储服务。掌握这些最佳实践对于在生产环境中成功部署和运维LevelDB至关重要。
