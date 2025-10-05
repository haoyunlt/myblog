---
title: "MongoDB-11-索引模块-数据结构"
date: 2025-10-05T12:12:00Z
draft: false
tags: ["MongoDB", "源码剖析", "索引模块", "数据结构", "B树"]
categories: ["mongodb"]
series: ["MongoDB源码剖析"]
weight: 15
description: "MongoDB索引模块数据结构详解：索引管理器、索引目录、B树结构等核心数据结构"
---

# MongoDB-11-索引模块-数据结构

## 1. 核心数据结构概览

索引模块的数据结构设计采用分层架构，从上层的索引描述符到底层的存储结构，形成了完整的索引数据管理体系。

### 1.1 数据结构分类

- **描述符层：** IndexDescriptor、IndexCatalogEntry
- **访问方法层：** IndexAccessMethod及其子类
- **键值处理层：** KeyString、BSONObjSet
- **存储接口层：** SortedDataInterface、RecordId
- **构建器层：** IndexBuilder、BulkBuilder

## 2. 核心类图

### 2.1 索引描述符类图

```mermaid
classDiagram
    class IndexDescriptor {
        -BSONObj _keyPattern
        -string _indexName
        -BSONObj _infoObj
        -IndexVersion _version
        -bool _isIdIndex
        -bool _isSparse
        -bool _isUnique
        -bool _isPartial
        -PartialFilterExpression _partialFilterExpression
        +keyPattern() BSONObj
        +indexName() string
        +unique() bool
        +sparse() bool
        +partial() bool
        +getAccessMethodName() string
        +compareIndexOptions() bool
    }

    class IndexCatalogEntry {
        -shared_ptr~IndexDescriptor~ _descriptor
        -unique_ptr~IndexAccessMethod~ _accessMethod
        -shared_ptr~Ident~ _ident
        -bool _isReady
        -bool _isFrozen
        -Timestamp _readyTimestamp
        -BSONObj _multikeyPaths
        +descriptor() IndexDescriptor*
        +accessMethod() IndexAccessMethod*
        +isReady() bool
        +setIsReady(bool ready)
        +ident() Ident*
    }

    class PartialFilterExpression {
        -unique_ptr~MatchExpression~ _expression
        -BSONObj _serializedExpression
        +getFilter() MatchExpression*
        +serialize() BSONObj
        +matches(BSONObj doc) bool
    }

    class Ident {
        -string _identString
        -NamespaceString _nss
        -string _indexName
        +toString() string
        +ns() NamespaceString
        +getIdent() string
    }

    IndexCatalogEntry --> IndexDescriptor : contains
    IndexDescriptor --> PartialFilterExpression : uses
    IndexCatalogEntry --> Ident : references
```

### 2.2 索引访问方法类图

```mermaid
classDiagram
    class IndexAccessMethod {
        <<abstract>>
        +insert(opCtx, records)* Status
        +remove(opCtx, obj, loc)* void
        +update(opCtx, oldDoc, newDoc)* Status
        +initializeAsEmpty()* Status
        +validate(opCtx, options)* IndexValidateResults
        +numKeys(opCtx)* int64_t
        +getSpaceUsedBytes(opCtx)* long long
        +make(opCtx, entry, ident)$ unique_ptr~IndexAccessMethod~
    }

    class SortedDataIndexAccessMethod {
        <<abstract>>
        -unique_ptr~SortedDataInterface~ _newInterface
        -IndexCatalogEntry* _btreeState
        +getKeys(obj, keys)* void
        +getSortedDataInterface() SortedDataInterface*
        +setIndexIsMultikey(paths) void
    }

    class BtreeAccessMethod {
        +BtreeAccessMethod(entry, interface)
        +getKeys(obj, keys) void
        +insert(opCtx, records) Status
        +remove(opCtx, obj, loc) void
        -_extractKeys(obj, keySet) void
        -_handleArrayField(elem, keys) void
    }

    class HashAccessMethod {
        +HashAccessMethod(entry, interface)
        +getKeys(obj, keys) void
        -_hash(elem) long long
        -_makeSingleKey(elem, hashValue) BSONObj
    }

    class S2AccessMethod {
        -unique_ptr~S2IndexingParams~ _params
        +S2AccessMethod(entry, interface)
        +getKeys(obj, keys) void
        -_getS2Keys(geoElement, keys) void
        -_getCellsForGeometry(elem, cells) void
    }

    class FTSAccessMethod {
        -unique_ptr~FTSSpec~ _ftsSpec
        +FTSAccessMethod(entry, interface)
        +getKeys(obj, keys) void
        -_scoreDocument(doc) TermFrequencyMap
        -_extractTerms(text, terms) void
    }

    class WildcardAccessMethod {
        -BSONObj _wildcardProjection
        +WildcardAccessMethod(entry, interface)
        +getKeys(obj, keys) void
        -_extractAllPaths(path, elem, keys) void
        -_applyProjection(doc, projection) BSONObj
    }

    class TextIndexAccessMethod {
        -unique_ptr~FTSSpec~ _ftsSpec
        +TextIndexAccessMethod(entry, interface)
        +getKeys(obj, keys) void
        -_tokenizeText(text, tokens) void
        -_buildIndexKeys(tokens, keys) void
    }

    IndexAccessMethod <|-- SortedDataIndexAccessMethod
    SortedDataIndexAccessMethod <|-- BtreeAccessMethod
    SortedDataIndexAccessMethod <|-- HashAccessMethod
    SortedDataIndexAccessMethod <|-- S2AccessMethod
    SortedDataIndexAccessMethod <|-- FTSAccessMethod
    SortedDataIndexAccessMethod <|-- WildcardAccessMethod
    SortedDataIndexAccessMethod <|-- TextIndexAccessMethod
```

### 2.3 索引键值处理类图

```mermaid
classDiagram
    class KeyString {
        <<namespace>>
        +Value
        +Builder
        +Discriminator
    }

    class KeyStringValue {
        -SharedBuffer _buffer
        -size_t _size
        -TypeBits _typeBits
        +getBuffer() SharedBuffer
        +getSize() size_t
        +getTypeBits() TypeBits
        +compare(other) int
        +isEmpty() bool
    }

    class KeyStringBuilder {
        -BufBuilder _buffer
        -TypeBitsBuilder _typeBits
        -Ordering _order
        -Version _version
        +appendBSONElement(elem) Builder&
        +appendString(str) Builder&
        +appendNumber(num) Builder&
        +appendRecordId(rid) Builder&
        +getValueCopy() Value
        +release() Value
    }

    class BSONObjSet {
        -set~BSONObj~ _objects
        -BSONObj::Comparator _comparator
        +insert(obj) pair~iterator, bool~
        +find(obj) iterator
        +erase(obj) size_t
        +size() size_t
        +empty() bool
    }

    class RecordId {
        -int64_t _id
        -string _str
        -Type _type
        +RecordId(id)
        +RecordId(str)
        +isLong() bool
        +isStr() bool
        +getLong() int64_t
        +getStr() string
        +compare(other) int
    }

    class MultikeyPaths {
        -vector~set~size_t~~ _paths
        +operator[](index) set~size_t~&
        +size() size_t
        +empty() bool
        +clear() void
    }

    KeyString ..> KeyStringValue : creates
    KeyString ..> KeyStringBuilder : creates
    BSONObjSet --> BSONObj : contains
    KeyStringValue --> RecordId : references
    KeyStringBuilder --> RecordId : appends
```

### 2.4 存储接口类图

```mermaid
classDiagram
    class SortedDataInterface {
        <<abstract>>
        +insertKeys(keys, recordIds)* Status
        +removeKeys(keys, recordIds)* void
        +unindex(key, recordId)* void
        +newCursor(opCtx)* unique_ptr~Cursor~
        +makeBulkBuilder(opCtx)* unique_ptr~BulkBuilder~
        +truncate(opCtx)* Status
        +isEmpty(opCtx)* bool
    }

    class SortedDataBuilderInterface {
        <<abstract>>
        +addKey(key, recordId)* Status
        +commit(dupsAllowed)* Status
        +numKeys()* int64_t
    }

    class Cursor {
        <<abstract>>
        +setEndPosition(key, inclusive)* void
        +seek(key, inclusive)* boost::optional~Entry~
        +next()* boost::optional~Entry~
        +seekExact(key)* boost::optional~Entry~
        +save()* void
        +restore()* void
    }

    class Entry {
        +key KeyString::Value
        +recordId RecordId
        +Entry(k, rid)
    }

    class IndexValidateResults {
        +valid bool
        +warnings vector~string~
        +errors vector~string~
        +keysTraversed int64_t
        +numRecords int64_t
    }

    class CompactOptions {
        +dryRun bool
        +paddingFactor double
        +paddingBytes int
        +validateDocuments bool
    }

    SortedDataInterface ..> SortedDataBuilderInterface : creates
    SortedDataInterface ..> Cursor : creates
    Cursor --> Entry : returns
    SortedDataInterface ..> IndexValidateResults : produces
    SortedDataInterface ..> CompactOptions : uses
```

### 2.5 索引构建器类图

```mermaid
classDiagram
    class IndexBuilder {
        -CollectionPtr _collection
        -IndexCatalogEntry* _entry
        -string _tempDir
        -size_t _maxMemoryUsage
        -size_t _docsProcessed
        +buildIndex(opCtx, background) Status
        +backgroundBuild(opCtx) Status
        +foregroundBuild(opCtx) Status
        -_scanAndSort(opCtx, sorter) Status
        -_insertSortedKeys(opCtx, sorter) Status
    }

    class MultiIndexBuilder {
        -vector~IndexBuilder*~ _builders
        -ThreadPool _threadPool
        +addBuilder(builder) void
        +buildAllIndexes(opCtx) Status
        -_buildIndexParallel(builder) void
    }

    class BackgroundIndexBuilder {
        -IndexBuildOptions _options
        -BackgroundOperation _backgroundOp
        +startBuild(opCtx) Status
        +resumeBuild(opCtx) Status
        +abortBuild(opCtx) Status
        -_phase1Scan(opCtx) Status
        -_phase2ProcessDeltas(opCtx) Status
        -_phase3Finalize(opCtx) Status
    }

    class IndexBuildCoordinator {
        -map~BuildId, IndexBuild~ _indexBuilds
        -ReplicationCoordinator* _replCoord
        +startIndexBuild(opCtx, spec) BuildId
        +commitIndexBuild(opCtx, buildId) Status
        +abortIndexBuild(opCtx, buildId) Status
        +getIndexBuildState(buildId) IndexBuildState
    }

    class Sorter {
        <<template>>
        -string _tempDir
        -size_t _maxMemoryUsage
        -File _tempFile
        -Settings _settings
        +add(key, value) void
        +done() Iterator*
        +numDataBytes() size_t
    }

    class BSONObjSorter {
        +BSONObjSorter(options)
        +add(key, recordId) void
        +done() Iterator*
    }

    IndexBuilder --> Sorter : uses
    MultiIndexBuilder --> IndexBuilder : manages
    BackgroundIndexBuilder --> IndexBuilder : extends
    IndexBuildCoordinator --> BackgroundIndexBuilder : orchestrates
    IndexBuilder ..> BSONObjSorter : creates
    Sorter <|-- BSONObjSorter
```

## 3. 索引类型特定数据结构

### 3.1 地理索引数据结构

```mermaid
classDiagram
    class S2IndexingParams {
        +int coarsestIndexedLevel
        +int finestIndexedLevel
        +int maxCellsInCovering
        +double radiusOfEarthInRadians
        +S2IndexingParams()
        +configureCoverer(coverer) void
    }

    class S2CellId {
        +uint64 id
        +int level
        +S2CellId(id)
        +S2CellId(point)
        +parent(level) S2CellId
        +child(pos) S2CellId
        +isLeaf() bool
        +toString() string
    }

    class GeoHash {
        +string _hash
        +BSONElement _type
        +double _lng
        +double _lat
        +GeoHash(lng, lat, precision)
        +getHash() string
        +getNeighbors() vector~GeoHash~
    }

    class Point {
        +double x
        +double y
        +Point(x, y)
        +distanceTo(other) double
        +toString() string
    }

    class S2Region {
        <<abstract>>
        +Contains(point)* bool
        +IntersectsCap(cap)* bool
        +GetCapBound()* S2Cap
    }

    class S2Polygon {
        +vector~S2Loop~ loops
        +S2Polygon(loops)
        +Contains(point) bool
        +Intersects(other) bool
        +GetArea() double
    }

    S2IndexingParams ..> S2CellId : configures
    S2Region <|-- S2Polygon
    Point ..> GeoHash : converts
```

### 3.2 全文索引数据结构

```mermaid
classDiagram
    class FTSSpec {
        -BSONObj _indexSpec
        -LanguageMap _languageMap
        -map~string, double~ _weights
        -string _defaultLanguage
        +FTSSpec(spec)
        +scoreDocument(doc) TermFrequencyMap
        +getLanguage(doc) string
        +getWeight(field) double
    }

    class TermFrequencyMap {
        -map~string, int~ _terms
        +addTerm(term, frequency) void
        +getFrequency(term) int
        +getTotalTerms() size_t
        +getTerms() vector~string~
    }

    class FTSLanguage {
        -string _languageCode
        -unique_ptr~Stemmer~ _stemmer
        -set~string~ _stopWords
        +FTSLanguage(code)
        +stem(word) string
        +isStopWord(word) bool
        +tokenize(text) vector~string~
    }

    class Stemmer {
        <<abstract>>
        +stem(word)* string
        +getStemmedForm(word)* string
    }

    class TextIndexEntry {
        +string term
        +double score
        +RecordId recordId
        +TextIndexEntry(term, score, id)
        +compare(other) int
    }

    FTSSpec --> TermFrequencyMap : produces
    FTSSpec --> FTSLanguage : uses
    FTSLanguage --> Stemmer : contains
    FTSSpec ..> TextIndexEntry : creates
```

### 3.3 哈希索引数据结构

```mermaid
classDiagram
    class HashAccessMethod {
        +static hashElement(elem, seed) long long
        +static makeSingleKey(elem, hashValue, keyPattern) BSONObj
        -_hasher ExpressionHash
    }

    class ExpressionHash {
        -int _seed
        -HashAlgorithm _algorithm
        +ExpressionHash(seed)
        +hash(elem) long long
        +hashString(str) long long
        +hashNumber(num) long long
    }

    class HashSeed {
        +static const int DEFAULT_SEED
        +static getSeed(indexSpec) int
        +static isValidSeed(seed) bool
    }

    class HashedIndexKey {
        +long long hashValue
        +BSONElement originalValue
        +HashedIndexKey(hash, original)
        +toBSON() BSONObj
        +compare(other) int
    }

    HashAccessMethod --> ExpressionHash : uses
    HashAccessMethod --> HashSeed : uses
    HashAccessMethod ..> HashedIndexKey : creates
```

## 4. 索引构建过程数据结构

### 4.1 构建状态管理

```mermaid
classDiagram
    class IndexBuildState {
        <<enumeration>>
        SETUP
        IN_PROGRESS
        DRAINING
        READY
        ABORTED
        COMMITTED
    }

    class IndexBuildBlock {
        -OperationContext* _opCtx
        -IndexCatalogEntry* _entry
        -bool _acquired
        +IndexBuildBlock(opCtx, entry)
        +acquire() void
        +release() void
        +isAcquired() bool
    }

    class IndexBuildsManager {
        -map~BuildId, IndexBuildInfo~ _builds
        -mutex _mutex
        +startBuild(buildId, info) void
        +getBuild(buildId) IndexBuildInfo*
        +finishBuild(buildId) void
        +abortBuild(buildId) void
    }

    class IndexBuildInfo {
        +BuildId buildId
        +NamespaceString nss
        +vector~BSONObj~ indexSpecs
        +IndexBuildState state
        +Timestamp startTimestamp
        +unique_ptr~IndexBuilder~ builder
    }

    class BuildId {
        +UUID id
        +BuildId()
        +BuildId(uuid)
        +toString() string
        +operator==(other) bool
    }

    IndexBuildsManager --> IndexBuildInfo : manages
    IndexBuildInfo --> IndexBuildState : has
    IndexBuildInfo --> BuildId : identified by
    IndexBuildBlock --> IndexCatalogEntry : locks
```

### 4.2 外部排序数据结构

```mermaid
classDiagram
    class SorterOptions {
        +string tempDir
        +size_t maxMemoryUsageBytes
        +bool extSortAllowed
        +size_t dbNameSize
        +size_t sorterFileNameMaxLen
        +SorterOptions()
        +TempDir(path) SorterOptions&
        +MaxMemoryUsageBytes(bytes) SorterOptions&
    }

    class SorterStats {
        +size_t memUsed
        +size_t spilledRanges
        +size_t numSpills
        +size_t totalDataSizeBytes
        +Microseconds timeSpentSpilling
        +SorterStats()
        +addSpill(dataSize, time) void
    }

    class SorterFile {
        -string _path
        -File _file
        -bool _isTemp
        +SorterFile(path, temp)
        +open() void
        +close() void
        +write(data, size) void
        +read(buffer, size) size_t
    }

    class SorterIterator {
        <<template>>
        +more() bool
        +next() pair~Key, Value~
        +openSource() void
        +closeSource() void
    }

    class BSONObjSorterIterator {
        -unique_ptr~SorterFile~ _file
        -BSONObj _current
        +more() bool
        +next() pair~BSONObj, RecordId~
        -_advance() void
    }

    SorterOptions --> SorterStats : configures
    Sorter --> SorterFile : uses
    Sorter ..> SorterIterator : creates
    SorterIterator <|-- BSONObjSorterIterator
```

## 5. 索引验证数据结构

### 5.1 验证结果和选项

```mermaid
classDiagram
    class ValidationOptions {
        +bool full
        +bool background
        +bool metadata
        +bool repair
        +ValidationLevel level
        +ValidationOptions()
        +enforceFastCount() bool
    }

    class ValidationLevel {
        <<enumeration>>
        OFF
        MODERATE
        STRICT
    }

    class ValidateResults {
        +bool valid
        +bool repaired
        +vector~string~ warnings
        +vector~string~ errors
        +vector~string~ extraIndexEntries
        +vector~string~ missingIndexEntries
        +BSONObjBuilder details
        +ValidateResults()
        +addError(msg) void
        +addWarning(msg) void
    }

    class IndexConsistency {
        -map~RecordId, BSONObjSet~ _missingIndexEntries
        -multiset~KeyString::Value~ _extraIndexEntries
        -size_t _numKeys
        -size_t _numRecords
        +addDocKey(key, recordId) void
        +addIndexKey(key, recordId) void
        +repairMissingIndexEntries(opCtx) int
        +repairExtraIndexEntries(opCtx) int
    }

    ValidationOptions --> ValidationLevel : uses
    ValidateResults --> ValidationOptions : based on
    IndexConsistency --> ValidateResults : produces
```

## 6. 字段映射和约束

### 6.1 字段到索引的映射关系

| 数据结构 | 主要字段 | 数据类型 | 约束条件 | 说明 |
|---------|---------|----------|----------|------|
| IndexDescriptor | _keyPattern | BSONObj | 非空，最大32字段 | 索引键模式定义 |
| IndexDescriptor | _indexName | string | 非空，最大127字符 | 索引唯一名称 |
| IndexCatalogEntry | _isReady | bool | - | 索引是否就绪 |
| KeyString::Value | _buffer | SharedBuffer | 最大1024字节 | 索引键的二进制表示 |
| RecordId | _id | int64_t | > 0 | 记录标识符 |
| MultikeyPaths | _paths | vector<set<size_t>> | 最大32个路径 | 多键路径集合 |

### 6.2 版本兼容性映射

| 索引版本 | 数据格式 | 支持特性 | 兼容性 |
|---------|---------|----------|--------|
| v1 | 旧格式 | 基础B树 | 已废弃 |
| v2 | 新格式 | 所有索引类型 | 当前版本 |
| v3 | 未来格式 | 列式索引 | 计划中 |

---

**文档版本：** v1.0  
**生成时间：** 2025-10-05  
**适用版本：** MongoDB 8.0+
