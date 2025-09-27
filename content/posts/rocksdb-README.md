# RocksDB 源码深度剖析系列

## 概述

本系列文档旨在帮助开发者由浅入深地精通RocksDB的源代码逻辑，通过详细的架构图、时序图、UML图和代码分析，全面理解这个高性能键值存储引擎的内部实现。

## 文档结构

### 1. [整体架构篇](./01-overall-architecture.md)
- RocksDB整体架构设计
- 核心组件关系图
- 系统交互时序图
- 模块间依赖关系

### 2. [API接口篇](./02-api-analysis.md)
- 对外API详细说明
- 调用链路分析
- 关键函数源码解析
- 使用示例

### 3. [核心模块篇](./03-core-modules/)
- [数据库引擎模块](./03-core-modules/db-engine.md)
- [存储引擎模块](./03-core-modules/storage-engine.md)
- [内存表模块](./03-core-modules/memtable.md)
- [压缩模块](./03-core-modules/compaction.md)
- [缓存模块](./03-core-modules/cache.md)
- [文件系统模块](./03-core-modules/file-system.md)
- [监控模块](./03-core-modules/monitoring.md)

### 4. [数据结构篇](./04-data-structures.md)
- 核心数据结构UML图
- 数据结构详细说明
- 内存布局分析

### 5. [实战经验篇](./05-best-practices.md)
- 性能优化实践
- 配置调优指南
- 常见问题解决方案
- 实际案例分析

## RocksDB简介

RocksDB是Facebook开发的高性能键值存储引擎，基于Google的LevelDB构建，采用LSM-Tree（Log-Structured Merge Tree）架构设计。它具有以下特点：

- **高性能**：针对SSD优化，支持多线程压缩
- **可扩展**：支持存储TB级数据
- **灵活配置**：提供丰富的配置选项
- **事务支持**：支持ACID事务
- **多语言绑定**：支持C++、Java、Python等多种语言

## 核心概念

### LSM-Tree架构
- **MemTable**：内存中的写缓冲区
- **SST文件**：磁盘上的有序字符串表
- **压缩**：后台合并和整理数据的过程

### 列族（Column Family）
- 逻辑上的数据分区
- 独立的配置和压缩策略
- 支持原子操作

### 快照（Snapshot）
- 数据库在某个时间点的一致性视图
- 支持多版本并发控制（MVCC）

## 开始阅读

建议按照以下顺序阅读文档：

1. 首先阅读[整体架构篇](./01-overall-architecture.md)，了解RocksDB的整体设计
2. 然后阅读[API接口篇](./02-api-analysis.md)，掌握如何使用RocksDB
3. 深入学习[核心模块篇](./03-core-modules/)，理解各个模块的实现细节
4. 参考[数据结构篇](./04-data-structures.md)，了解内部数据组织
5. 最后阅读[实战经验篇](./05-best-practices.md)，学习最佳实践

## 源码版本

本分析基于RocksDB主分支最新版本，涵盖了最新的特性和优化。

## 贡献

欢迎提交Issue和PR来完善这个文档系列。
