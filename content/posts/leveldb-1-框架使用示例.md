# LevelDB 框架使用示例

## 概述

LevelDB是一个由Google开发的快速键值存储库，提供了有序映射从字符串键到字符串值。它采用LSM-Tree（Log-Structured Merge-Tree）架构，优化了写入性能。

## 基本API使用示例

### 1. 基本的增删改查操作

```cpp
#include <iostream>
#include <cassert>
#include "leveldb/db.h"

int main() {
    leveldb::DB* db;
    leveldb::Options options;
    leveldb::Status status;

    // 配置选项
    options.create_if_missing = true;  // 如果数据库不存在则创建
    
    // 打开数据库
    status = leveldb::DB::Open(options, "/tmp/testdb", &db);
    assert(status.ok());

    // 写入数据
    status = db->Put(leveldb::WriteOptions(), "key1", "value1");
    assert(status.ok());
    
    // 读取数据
    std::string value;
    status = db->Get(leveldb::ReadOptions(), "key1", &value);
    assert(status.ok());
    std::cout << "读取到的值: " << value << std::endl;
    
    // 删除数据
    status = db->Delete(leveldb::WriteOptions(), "key1");
    assert(status.ok());
    
    // 验证删除
    status = db->Get(leveldb::ReadOptions(), "key1", &value);
    assert(status.IsNotFound());
    
    delete db;
    return 0;
}
```

### 2. 批量操作示例

```cpp
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

void batch_operations_example() {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    
    leveldb::Status status = leveldb::DB::Open(options, "/tmp/batchdb", &db);
    assert(status.ok());
    
    // 创建批量操作
    leveldb::WriteBatch batch;
    batch.Put("key1", "value1");
    batch.Put("key2", "value2"); 
    batch.Put("key3", "value3");
    batch.Delete("old_key");  // 删除可能不存在的键
    
    // 原子性执行所有操作
    status = db->Write(leveldb::WriteOptions(), &batch);
    assert(status.ok());
    
    delete db;
}
```

### 3. 迭代器遍历示例

```cpp
#include "leveldb/db.h"

void iterator_example() {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    
    leveldb::Status status = leveldb::DB::Open(options, "/tmp/iterdb", &db);
    assert(status.ok());
    
    // 插入一些测试数据
    db->Put(leveldb::WriteOptions(), "apple", "fruit");
    db->Put(leveldb::WriteOptions(), "banana", "fruit");
    db->Put(leveldb::WriteOptions(), "carrot", "vegetable");
    
    // 创建迭代器
    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
    
    // 遍历所有键值对
    std::cout << "正向遍历:" << std::endl;
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::cout << it->key().ToString() << " => " << it->value().ToString() << std::endl;
    }
    
    // 反向遍历  
    std::cout << "\n反向遍历:" << std::endl;
    for (it->SeekToLast(); it->Valid(); it->Prev()) {
        std::cout << it->key().ToString() << " => " << it->value().ToString() << std::endl;
    }
    
    // 定位到特定键
    it->Seek("b");
    if (it->Valid()) {
        std::cout << "\n从'b'开始的第一个键: " << it->key().ToString() << std::endl;
    }
    
    delete it;
    delete db;
}
```

### 4. 快照（Snapshot）使用示例

```cpp
#include "leveldb/db.h"

void snapshot_example() {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    
    leveldb::Status status = leveldb::DB::Open(options, "/tmp/snapdb", &db);
    assert(status.ok());
    
    // 写入初始数据
    db->Put(leveldb::WriteOptions(), "key1", "initial_value");
    
    // 创建快照
    const leveldb::Snapshot* snapshot = db->GetSnapshot();
    
    // 修改数据
    db->Put(leveldb::WriteOptions(), "key1", "modified_value");
    
    // 从快照读取（看到的是旧值）
    leveldb::ReadOptions read_options;
    read_options.snapshot = snapshot;
    std::string value;
    
    status = db->Get(read_options, "key1", &value);
    assert(status.ok());
    std::cout << "快照中的值: " << value << std::endl;  // 输出: initial_value
    
    // 正常读取（看到的是新值）
    status = db->Get(leveldb::ReadOptions(), "key1", &value);
    assert(status.ok());
    std::cout << "当前值: " << value << std::endl;  // 输出: modified_value
    
    // 释放快照
    db->ReleaseSnapshot(snapshot);
    delete db;
}
```

### 5. 配置选项示例

```cpp
#include "leveldb/db.h"
#include "leveldb/cache.h"
#include "leveldb/filter_policy.h"

void advanced_options_example() {
    leveldb::DB* db;
    leveldb::Options options;
    
    // 基本设置
    options.create_if_missing = true;
    options.error_if_exists = false;
    options.paranoid_checks = true;
    
    // 性能调优参数
    options.write_buffer_size = 64 * 1024 * 1024;  // 64MB写入缓冲区
    options.max_open_files = 500;
    options.block_size = 64 * 1024;  // 64KB块大小
    options.block_restart_interval = 32;
    options.max_file_size = 64 * 1024 * 1024;  // 64MB最大文件大小
    
    // 压缩设置
    options.compression = leveldb::kSnappyCompression;
    
    // 缓存设置
    leveldb::Cache* cache = leveldb::NewLRUCache(256 * 1024 * 1024);  // 256MB缓存
    options.block_cache = cache;
    
    // 布隆过滤器
    const leveldb::FilterPolicy* filter = leveldb::NewBloomFilterPolicy(10);
    options.filter_policy = filter;
    
    leveldb::Status status = leveldb::DB::Open(options, "/tmp/advanceddb", &db);
    if (!status.ok()) {
        std::cerr << "打开数据库失败: " << status.ToString() << std::endl;
        return;
    }
    
    // 执行一些操作...
    db->Put(leveldb::WriteOptions(), "test_key", "test_value");
    
    // 清理资源
    delete db;
    delete cache;
    delete filter;
}
```

### 6. 范围查询示例

```cpp
#include "leveldb/db.h"

void range_query_example() {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    
    leveldb::DB::Open(options, "/tmp/rangedb", &db);
    
    // 插入有序数据
    for (int i = 0; i < 100; i++) {
        std::string key = "key" + std::to_string(i);
        std::string value = "value" + std::to_string(i);
        db->Put(leveldb::WriteOptions(), key, value);
    }
    
    // 范围查询: 查询key10到key20之间的数据
    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
    
    std::cout << "key10到key20范围内的数据:" << std::endl;
    for (it->Seek("key10"); it->Valid() && it->key().ToString() <= "key20"; it->Next()) {
        std::cout << it->key().ToString() << " => " << it->value().ToString() << std::endl;
    }
    
    delete it;
    delete db;
}
```

### 7. 错误处理示例

```cpp
#include "leveldb/db.h"

void error_handling_example() {
    leveldb::DB* db;
    leveldb::Options options;
    leveldb::Status status;
    
    // 尝试打开不存在的数据库且不创建
    options.create_if_missing = false;
    status = leveldb::DB::Open(options, "/tmp/nonexistentdb", &db);
    
    if (!status.ok()) {
        std::cout << "预期的错误: " << status.ToString() << std::endl;
        
        if (status.IsNotFound()) {
            std::cout << "数据库文件不存在" << std::endl;
        } else if (status.IsCorruption()) {
            std::cout << "数据库文件损坏" << std::endl;
        } else if (status.IsIOError()) {
            std::cout << "IO错误" << std::endl;
        }
    }
    
    // 正确打开数据库
    options.create_if_missing = true;
    status = leveldb::DB::Open(options, "/tmp/errordb", &db);
    assert(status.ok());
    
    // 读取不存在的键
    std::string value;
    status = db->Get(leveldb::ReadOptions(), "nonexistent_key", &value);
    if (status.IsNotFound()) {
        std::cout << "键不存在（这是正常情况）" << std::endl;
    }
    
    delete db;
}
```

### 8. 数据库属性查询示例

```cpp
#include "leveldb/db.h"

void property_query_example() {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    
    leveldb::DB::Open(options, "/tmp/propdb", &db);
    
    // 插入一些数据
    for (int i = 0; i < 1000; i++) {
        std::string key = "key" + std::to_string(i);
        std::string value = "value" + std::to_string(i);
        db->Put(leveldb::WriteOptions(), key, value);
    }
    
    // 查询数据库统计信息
    std::string stats;
    if (db->GetProperty("leveldb.stats", &stats)) {
        std::cout << "数据库统计信息:\n" << stats << std::endl;
    }
    
    // 查询SSTable信息
    std::string sstables;
    if (db->GetProperty("leveldb.sstables", &sstables)) {
        std::cout << "SSTable信息:\n" << sstables << std::endl;
    }
    
    // 查询大概的内存使用量
    std::string memory_usage;
    if (db->GetProperty("leveldb.approximate-memory-usage", &memory_usage)) {
        std::cout << "大概内存使用量: " << memory_usage << " bytes" << std::endl;
    }
    
    delete db;
}
```

### 9. 手动压缩示例

```cpp
#include "leveldb/db.h"

void compaction_example() {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    
    leveldb::DB::Open(options, "/tmp/compactdb", &db);
    
    // 写入大量数据
    for (int i = 0; i < 10000; i++) {
        std::string key = "key" + std::to_string(i);
        std::string value = std::string(1000, 'a' + (i % 26));  // 1KB值
        db->Put(leveldb::WriteOptions(), key, value);
    }
    
    // 删除一半数据
    for (int i = 0; i < 5000; i++) {
        std::string key = "key" + std::to_string(i);
        db->Delete(leveldb::WriteOptions(), key);
    }
    
    std::cout << "压缩前的数据库状态:" << std::endl;
    std::string stats;
    if (db->GetProperty("leveldb.stats", &stats)) {
        std::cout << stats << std::endl;
    }
    
    // 手动触发全量压缩
    db->CompactRange(nullptr, nullptr);
    
    std::cout << "压缩后的数据库状态:" << std::endl;
    if (db->GetProperty("leveldb.stats", &stats)) {
        std::cout << stats << std::endl;
    }
    
    delete db;
}
```

## 编译和链接

### CMake示例

```cmake
cmake_minimum_required(VERSION 3.10)
project(LevelDBExample)

set(CMAKE_CXX_STANDARD 14)

# 查找LevelDB库
find_package(PkgConfig REQUIRED)
pkg_check_modules(LEVELDB REQUIRED leveldb)

# 添加可执行文件
add_executable(leveldb_example main.cpp)

# 链接LevelDB库
target_link_libraries(leveldb_example ${LEVELDB_LIBRARIES})
target_include_directories(leveldb_example PRIVATE ${LEVELDB_INCLUDE_DIRS})
target_compile_options(leveldb_example PRIVATE ${LEVELDB_CFLAGS_OTHER})
```

### Makefile示例

```makefile
CXX = g++
CXXFLAGS = -std=c++14 -Wall -O2
LIBS = -lleveldb -lsnappy -lpthread

all: leveldb_example

leveldb_example: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f leveldb_example

.PHONY: all clean
```

## 性能优化建议

### 1. 写入优化
- 使用WriteBatch进行批量写入
- 合理设置write_buffer_size
- 考虑设置sync=false以提高写入性能（但会有数据丢失风险）

### 2. 读取优化  
- 使用布隆过滤器减少磁盘访问
- 调整block_cache大小
- 使用快照避免读写冲突

### 3. 存储优化
- 启用压缩（默认Snappy）
- 定期进行手动压缩
- 合理设置max_file_size

这些示例展示了LevelDB的基本使用方法和高级特性。在实际项目中，应根据具体需求调整配置参数以获得最佳性能。
