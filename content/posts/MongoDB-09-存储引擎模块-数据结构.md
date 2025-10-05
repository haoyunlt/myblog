---
title: "MongoDB-09-存储引擎模块-数据结构"
date: 2025-10-05T12:10:00Z
draft: false
tags: ["MongoDB", "源码剖析", "存储引擎", "数据结构", "WiredTiger"]
categories: ["mongodb"]
series: ["MongoDB源码剖析"]
weight: 11
description: "MongoDB存储引擎模块数据结构详解：存储接口、事务对象、数据管理的核心数据结构"
---

# MongoDB-09-存储引擎模块-数据结构

## 1. 核心数据结构概览

存储引擎模块的数据结构设计实现了MongoDB的持久化存储功能，支持多种存储引擎（主要是WiredTiger）的统一抽象。数据结构按功能分为几个主要层次：

### 1.1 数据结构分类

- **存储引擎抽象层：** StorageEngine、KVEngine
- **记录存储层：** RecordStore、SeekableRecordCursor
- **事务管理层：** RecoveryUnit、Change
- **索引存储层：** SortedDataInterface、SortedDataBuilderInterface
- **目录管理层：** KVCatalog、WiredTigerCatalog
- **缓存管理层：** KVDrop pendings、SizeStorer

## 2. 核心类图

### 2.1 存储引擎抽象层类图

```mermaid
classDiagram
    class StorageEngine {
        <<abstract>>
        +loadCatalog(opCtx, stableTimestamp) void
        +listDatabases(out) void
        +createRecordStore(opCtx, nss, options) Status
        +getRecordStore(opCtx, nss, ident, options) unique_ptr~RecordStore~
        +createSortedDataInterface(opCtx, collOptions, ident, desc) Status
        +getSortedDataInterface(opCtx, collOptions, ident, desc) unique_ptr~SortedDataInterface~
        +newRecoveryUnit() unique_ptr~RecoveryUnit~
        +createCheckpoint(opCtx) Status
        +getStableTimestamp() Timestamp
        +setStableTimestamp(timestamp, force) void
        +beginBackup(opCtx) StatusWith~vector~string~~
        +endBackup(opCtx) void
        +isDurable() bool
        +isEphemeral() bool
        +supportsCappedCollections() bool
        +supportsDocLocking() bool
    }

    class KVEngine {
        <<abstract>>
        +getAllIdents(opCtx) vector~string~
        +recoverToStableTimestamp(opCtx) Status
        +cleanShutdown() void
        +flushAllFiles(opCtx, callerHoldsReadLock) Status
        +beginNonBlockingBackup(opCtx, options) StatusWith~StorageEngine::BackupInformation~
        +endNonBlockingBackup(opCtx) void
        +disableIncrementalBackup(opCtx) Status
        +getMetadata(opCtx, ident) BSONObj
        +setMetadata(opCtx, ident, metadata) Status
    }

    class WiredTigerKVEngine {
        -WT_CONNECTION* _conn
        -unique_ptr~WiredTigerSessionCache~ _sessionCache
        -unique_ptr~WiredTigerCatalog~ _catalog
        -string _canonicalName
        -string _path
        -bool _durable
        -bool _ephemeral
        -WiredTigerEventHandler _eventHandler
        -unique_ptr~WiredTigerSizeStorer~ _sizeStorer
        -mutable Mutex _dropPendingQueueMutex
        -WiredTigerDropPendingQueue _dropPendingQueue
        +WiredTigerKVEngine(canonicalName, path, cs, extraOpenOptions, cacheMB, maxHistoryFileSizeMB, ephemeral, repair)
        +loadCatalog(opCtx, stableTimestamp) void
        +createRecordStore(opCtx, nss, options) Status
        +getRecordStore(opCtx, nss, ident, options) unique_ptr~RecordStore~
        +createCheckpoint(opCtx) Status
        +flushAllFiles(opCtx, callerHoldsReadLock) Status
        +getConnection() WT_CONNECTION*
        +getSession(opCtx) WT_SESSION*
        +releaseSession(session) void
        -_initConnection(options) Status
        -_createIdent(nss, options) string
        -_postCreateCollection(opCtx, ident, metadata) void
    }

    class DevNullKVEngine {
        -map~string, shared_ptr~DevNullRecordStore~~ _recordStores
        -map~string, shared_ptr~DevNullSortedDataInterface~~ _sortedDataInterfaces
        -mutable Mutex _mutex
        +loadCatalog(opCtx, stableTimestamp) void
        +createRecordStore(opCtx, nss, options) Status
        +getRecordStore(opCtx, nss, ident, options) unique_ptr~RecordStore~
        +newRecoveryUnit() unique_ptr~RecoveryUnit~
        +isDurable() bool
        +isEphemeral() bool
    }

    StorageEngine <|-- KVEngine
    KVEngine <|-- WiredTigerKVEngine
    KVEngine <|-- DevNullKVEngine
    WiredTigerKVEngine --> WiredTigerCatalog : manages
    WiredTigerKVEngine --> WiredTigerSessionCache : uses
```

### 2.2 记录存储层类图

```mermaid
classDiagram
    class RecordStore {
        <<abstract>>
        +insertRecord(opCtx, data, len, out) Status
        +insertRecords(opCtx, records, timestamps) Status
        +updateRecord(opCtx, recordId, data, len) Status
        +deleteRecord(opCtx, recordId) Status
        +dataFor(opCtx, recordId) RecordData
        +findRecord(opCtx, recordId, out) bool
        +getCursor(opCtx, forward) unique_ptr~SeekableRecordCursor~
        +numRecords(opCtx) int64_t
        +dataSize(opCtx) int64_t
        +storageSize(opCtx) int64_t
        +isCapped() bool
        +isOplog() bool
        +setCappedCallback(cb) void
        +validate(opCtx, results, output) Status
        +doDeleteRecord(opCtx, recordId) void
    }

    class WiredTigerRecordStore {
        -WiredTigerKVEngine* _kvEngine
        -string _nss
        -string _ident
        -string _tableURI
        -bool _isOplog
        -bool _isCapped
        -shared_ptr~CappedCallback~ _cappedCallback
        -mutable AtomicWord~long long~ _numRecords
        -mutable AtomicWord~long long~ _dataSize
        -unique_ptr~WiredTigerSizeStorer~ _sizeStorer
        -string _sizeStorerUri
        -AtomicWord~long long~ _nextRecordId
        +WiredTigerRecordStore(kvEngine, opCtx, nss, ident, isOplog, cappedCallback)
        +insertRecord(opCtx, data, len, out) Status
        +updateRecord(opCtx, recordId, data, len) Status
        +deleteRecord(opCtx, recordId) Status
        +dataFor(opCtx, recordId) RecordData
        +getCursor(opCtx, forward) unique_ptr~SeekableRecordCursor~
        +validate(opCtx, results, output) Status
        -_generateOplogRecordId(opCtx) RecordId
        -_nextRecordId() RecordId
        -_increaseDataSize(amount) void
        -_decreaseDataSize(amount) void
    }

    class DevNullRecordStore {
        -string _ns
        -mutable Mutex _recordsMutex
        -map~RecordId, string~ _records
        -AtomicWord~long long~ _nextRecordId
        +insertRecord(opCtx, data, len, out) Status
        +deleteRecord(opCtx, recordId) Status
        +dataFor(opCtx, recordId) RecordData
        +getCursor(opCtx, forward) unique_ptr~SeekableRecordCursor~
        +numRecords(opCtx) int64_t
        +isCapped() bool
        +isOplog() bool
    }

    class SeekableRecordCursor {
        <<abstract>>
        +next() boost::optional~Record~
        +seekExact(id) boost::optional~Record~
        +seekNear(id) boost::optional~Record~
        +save() void
        +restore() bool
        +detachFromOperationContext() void
        +reattachToOperationContext(opCtx) void
    }

    class WiredTigerRecordStoreCursor {
        -WiredTigerRecoveryUnit* _ru
        -string _uri
        -WT_CURSOR* _cursor
        -bool _forward
        -bool _positioned
        -RecordId _lastReturnedId
        +WiredTigerRecordStoreCursor(opCtx, rs, forward)
        +next() boost::optional~Record~
        +seekExact(id) boost::optional~Record~
        +save() void
        +restore() bool
        -_getNext() boost::optional~Record~
        -_skipToNext() void
    }

    class RecordData {
        -SharedBuffer _buffer
        -int _size
        +RecordData()
        +RecordData(data, size)
        +data() const char*
        +size() int
        +isNull() bool
        +getOwned() RecordData
        +toBson() BSONObj
    }

    class RecordId {
        -int64_t _id
        -string _str
        -Type _type
        +RecordId()
        +RecordId(id)
        +RecordId(str)
        +repr() int64_t
        +isLong() bool
        +isStr() bool
        +toString() string
        +compare(other) int
        +operator<(other) bool
        +operator==(other) bool
        +isValid() bool
    }

    RecordStore <|-- WiredTigerRecordStore
    RecordStore <|-- DevNullRecordStore
    RecordStore --> SeekableRecordCursor : creates
    SeekableRecordCursor <|-- WiredTigerRecordStoreCursor
    RecordStore --> RecordData : returns
    RecordStore --> RecordId : uses
    WiredTigerRecordStore --> WiredTigerKVEngine : uses
```

### 2.3 事务管理层类图

```mermaid
classDiagram
    class RecoveryUnit {
        <<abstract>>
        +beginUnitOfWork(opCtx) void
        +commitUnitOfWork() void
        +abortUnitOfWork() void
        +waitUntilDurable(opCtx) bool
        +waitUntilUnjournaledWritesDurable(opCtx) bool
        +abandonSnapshot() void
        +setTimestamp(timestamp) Status
        +getTimestamp() Timestamp
        +registerChange(change) void
        +onRollback(callback) void
        +onCommit(callback) void
        +inUnitOfWork() bool
        +hasUncommittedWrites() bool
        +setReadOnce(readOnce) void
        +getReadOnce() bool
    }

    class WiredTigerRecoveryUnit {
        -WT_SESSION* _session
        -WiredTigerKVEngine* _kvEngine
        -WiredTigerSessionCache* _sessionCache
        -bool _inUnitOfWork
        -bool _hasUncommittedWrites
        -bool _isTimestamped
        -Timestamp _commitTimestamp
        -Timestamp _durableTimestamp
        -Timestamp _readAtTimestamp
        -bool _roundUpToIncludeEndOfBatch
        -vector~unique_ptr~Change~~ _changes
        -vector~function~void()~~ _onCommitCallbacks
        -vector~function~void()~~ _onRollbackCallbacks
        -map~string, WT_CURSOR*~ _cursors
        -int _myTransactionCount
        +WiredTigerRecoveryUnit(sessionCache, kvEngine)
        +beginUnitOfWork(opCtx) void
        +commitUnitOfWork() void
        +abortUnitOfWork() void
        +setTimestamp(timestamp) Status
        +getSession() WT_SESSION*
        +getCursor(uri, forRecordStore) WT_CURSOR*
        +waitUntilDurable(opCtx) bool
        -_isTimestamped() bool
        -_prepareForSnapshot(opCtx) void
        -_txnClose(commit) void
        -_getTransactionConfigString() string
    }

    class DevNullRecoveryUnit {
        -bool _inUnitOfWork
        -vector~unique_ptr~Change~~ _changes
        -vector~function~void()~~ _onCommitCallbacks
        -vector~function~void()~~ _onRollbackCallbacks
        +beginUnitOfWork(opCtx) void
        +commitUnitOfWork() void
        +abortUnitOfWork() void
        +waitUntilDurable(opCtx) bool
        +inUnitOfWork() bool
    }

    class Change {
        <<abstract>>
        +commit(commitTimestamp) void
        +rollback() void
    }

    class RecordStoreChange {
        -RecordStore* _rs
        -RecordId _recordId
        -ChangeType _type
        -string _oldData
        -string _newData
        +RecordStoreChange(rs, recordId, type, oldData, newData)
        +commit(commitTimestamp) void
        +rollback() void
    }

    class IndexChange {
        -SortedDataInterface* _interface
        -KeyString::Value _key
        -RecordId _recordId
        -ChangeType _type
        +IndexChange(interface, key, recordId, type)
        +commit(commitTimestamp) void
        +rollback() void
    }

    class OperationContext {
        -unique_ptr~RecoveryUnit~ _recoveryUnit
        -ServiceContext* _serviceContext
        -LogicalSessionId _lsid
        -TxnNumber _txnNumber
        -bool _inMultiDocumentTransaction
        -Timestamp _readTimestamp
        +getRecoveryUnit() RecoveryUnit*
        +setRecoveryUnit(ru, tokensToLinkFrom) void
        +writeConflictRetry(opStr, ns, f) auto
        +checkForInterrupt() void
        +getDeadline() Date_t
        +hasDeadline() bool
    }

    RecoveryUnit <|-- WiredTigerRecoveryUnit
    RecoveryUnit <|-- DevNullRecoveryUnit
    RecoveryUnit --> Change : manages
    Change <|-- RecordStoreChange
    Change <|-- IndexChange
    OperationContext --> RecoveryUnit : contains
    WiredTigerRecoveryUnit --> WiredTigerKVEngine : uses
```

### 2.4 索引存储层类图

```mermaid
classDiagram
    class SortedDataInterface {
        <<abstract>>
        +insertKeys(opCtx, keys, dupsAllowed) Status
        +removeKeys(opCtx, keys, dupsAllowed) void
        +unindex(opCtx, key, recordId, dupsAllowed) void
        +newCursor(opCtx, forward) unique_ptr~Cursor~
        +newRandomCursor(opCtx) unique_ptr~Cursor~
        +makeBulkBuilder(opCtx, dupsAllowed) unique_ptr~BulkBuilder~
        +truncate(opCtx) Status
        +isEmpty(opCtx) bool
        +getSpaceUsedBytes(opCtx) long long
        +getFreeStorageBytes(opCtx) long long
        +fullValidate(opCtx, keysOut, sizeOut) Status
        +dupKeyCheck(opCtx, key) Status
        +isUnique() bool
        +getKeyStringVersion() KeyString::Version
        +getOrdering() Ordering
    }

    class WiredTigerIndexUnique {
        -WiredTigerKVEngine* _kvEngine
        -string _uri
        -string _tableURI
        -KeyString::Version _keyStringVersion
        -Ordering _ordering
        -string _indexName
        -BSONObj _keyPattern
        -BSONObj _collation
        +WiredTigerIndexUnique(opCtx, uri, desc, keyFormat, isLogged)
        +insertKeys(opCtx, keys, dupsAllowed) Status
        +removeKeys(opCtx, keys, dupsAllowed) void
        +newCursor(opCtx, forward) unique_ptr~Cursor~
        +makeBulkBuilder(opCtx, dupsAllowed) unique_ptr~BulkBuilder~
        +isUnique() bool
        -_keyExists(opCtx, key) bool
        -_insertKey(opCtx, key, recordId) Status
        -_removeKey(opCtx, key, recordId) void
    }

    class WiredTigerIndexStandard {
        -WiredTigerKVEngine* _kvEngine
        -string _uri
        -string _tableURI
        -KeyString::Version _keyStringVersion
        -Ordering _ordering
        +WiredTigerIndexStandard(opCtx, uri, desc, keyFormat, isLogged)
        +insertKeys(opCtx, keys, dupsAllowed) Status
        +removeKeys(opCtx, keys, dupsAllowed) void
        +newCursor(opCtx, forward) unique_ptr~Cursor~
        +makeBulkBuilder(opCtx, dupsAllowed) unique_ptr~BulkBuilder~
        +isUnique() bool
    }

    class SortedDataBuilderInterface {
        <<abstract>>
        +addKey(key, recordId) Status
        +commit(dupsAllowed) Status
        +numKeys() int64_t
    }

    class WiredTigerIndexBulkBuilder {
        -WiredTigerKVEngine* _kvEngine
        -WT_SESSION* _session
        -WT_CURSOR* _cursor
        -string _uri
        -bool _dupsAllowed
        -int64_t _keysInserted
        -Ordering _ordering
        +WiredTigerIndexBulkBuilder(kvEngine, uri, dupsAllowed, ordering)
        +addKey(key, recordId) Status
        +commit(dupsAllowed) Status
        +numKeys() int64_t
        -_insertKey(key, recordId) Status
    }

    class SortedDataCursor {
        <<abstract>>
        +setEndPosition(endKey, endKeyInclusive) void
        +seek(key, requestedInfo) boost::optional~IndexKeyEntry~
        +seekForPrev(key, requestedInfo) boost::optional~IndexKeyEntry~
        +next(requestedInfo) boost::optional~IndexKeyEntry~
        +prev(requestedInfo) boost::optional~IndexKeyEntry~
        +save() void
        +restore() void
        +detachFromOperationContext() void
        +reattachToOperationContext(opCtx) void
    }

    class WiredTigerIndexCursor {
        -WT_CURSOR* _cursor
        -WiredTigerKVEngine* _kvEngine
        -string _uri
        -bool _forward
        -bool _positioned
        -Ordering _ordering
        -KeyString::Version _keyStringVersion
        +WiredTigerIndexCursor(kvEngine, uri, forward, ordering, keyStringVersion)
        +seek(key, requestedInfo) boost::optional~IndexKeyEntry~
        +next(requestedInfo) boost::optional~IndexKeyEntry~
        +save() void
        +restore() void
        -_getNext(requestedInfo) boost::optional~IndexKeyEntry~
        -_skipToNext() void
    }

    SortedDataInterface <|-- WiredTigerIndexUnique
    SortedDataInterface <|-- WiredTigerIndexStandard
    SortedDataInterface --> SortedDataBuilderInterface : creates
    SortedDataBuilderInterface <|-- WiredTigerIndexBulkBuilder
    SortedDataInterface --> SortedDataCursor : creates
    SortedDataCursor <|-- WiredTigerIndexCursor
    WiredTigerIndexUnique --> WiredTigerKVEngine : uses
    WiredTigerIndexStandard --> WiredTigerKVEngine : uses
```

### 2.5 目录管理层类图

```mermaid
classDiagram
    class KVCatalog {
        <<abstract>>
        +newEntry(opCtx, nss, ident, md) Status
        +putEntry(opCtx, nss, ident, md) Status
        +getEntry(nss) Entry
        +getAllEntries() vector~Entry~
        +dropEntry(opCtx, nss) Status
        +renameEntry(opCtx, fromNss, toNss, ident) Status
        +getIndexIdent(opCtx, nss, indexName) string
        +isCollectionIdent(ident) bool
        +isIndexIdent(ident) bool
        +getMetaData(opCtx, ns) BSONObj
        +setMetaData(opCtx, ns, md) Status
    }

    class WiredTigerCatalog {
        -WiredTigerKVEngine* _kvEngine
        -mutable Mutex _catalogLock
        -StringMap~Entry~ _catalog
        -string _catalogUri
        -bool _rand
        +WiredTigerCatalog(kvEngine, isLogged, isEphemeral, directoryPerDb)
        +init(opCtx) Status
        +newEntry(opCtx, nss, ident, md) Status
        +getEntry(nss) Entry
        +dropEntry(opCtx, nss) Status
        +getIndexIdent(opCtx, nss, indexName) string
        +isUserDataIdent(ident) bool
        +getMetaData(opCtx, ns) BSONObj
        +putMetaData(opCtx, ns, md) Status
        +getAllIdentsForDB(db) list~string~
        +getAllIdents() vector~string~
        -_loadCatalog(opCtx) Status
        -_saveCatalogToIndex(opCtx) Status
        -_generateNextIdent() string
    }

    class Entry {
        +string ident
        +BSONObj metadata
        +Entry()
        +Entry(ident, metadata)
    }

    class WiredTigerSizeStorer {
        -WiredTigerKVEngine* _kvEngine
        -string _uri
        -mutable Mutex _bufferMutex
        -map~string, SizeInfo~ _buffer
        -AtomicWord~long long~ _loadedFromDisk
        -unique_ptr~WiredTigerSession~ _session
        +WiredTigerSizeStorer(kvEngine, uri)
        +onCreate(uri, numRecords, dataSize) void
        +onDestroy(uri) void
        +store(uri, numRecords, dataSize) void
        +load(uri, numRecords, dataSize) void
        +flush() void
        +fillCache() void
        -_checkMagicNumber() void
        -_writeToBuffer(uri, numRecords, dataSize) void
    }

    class SizeInfo {
        +AtomicWord~long long~ numRecords
        +AtomicWord~long long~ dataSize
        +bool dirty
        +SizeInfo()
        +SizeInfo(numRecords, dataSize)
    }

    class WiredTigerDropPendingQueue {
        -mutable Mutex _mutex
        -condition_variable _condVar
        -bool _shuttingDown
        -list~WiredTigerDropPendingQueueEntry~ _dropPendingQueue
        +addDropPendingIdent(opTime, nss, ident) void
        +dropIdentsOlderThan(opCtx, ts) void
        +getAllDropPendingIdents() vector~string~
        +clearDropPendingState() void
        +getEarliestDropTimestamp() Timestamp
        -_dropIdent(opCtx, entry) void
    }

    KVCatalog <|-- WiredTigerCatalog
    WiredTigerCatalog --> Entry : manages
    WiredTigerCatalog --> WiredTigerKVEngine : uses
    WiredTigerKVEngine --> WiredTigerSizeStorer : uses
    WiredTigerSizeStorer --> SizeInfo : contains
    WiredTigerKVEngine --> WiredTigerDropPendingQueue : uses
```

## 3. 会话和连接管理

### 3.1 会话管理类图

```mermaid
classDiagram
    class WiredTigerSessionCache {
        -WiredTigerKVEngine* _engine
        -WT_CONNECTION* _conn
        -mutable Mutex _sessionsMutex
        -vector~WiredTigerSession*~ _sessions
        -size_t _maxSessions
        -AtomicWord~int~ _currentSessions
        -mutable Mutex _tableCacheMutex
        -StringMap~uint64_t~ _tableCreationTime
        +WiredTigerSessionCache(conn, engine)
        +getSession() WiredTigerSession*
        +releaseSession(session) void
        +closeAll() void
        +shuttingDown() void
        +snapshotManager() SnapshotManager*
        +isEphemeral() bool
        -_getSession(opCtx) WiredTigerSession*
        -_closeExpiredIdleSessions(idleExpireAfterSeconds) void
    }

    class WiredTigerSession {
        -WT_SESSION* _session
        -WiredTigerSessionCache* _cache
        -uint64_t _epoch
        -mutable Mutex _cursorsMutex
        -StringMap~WT_CURSOR*~ _cursors
        -int _cursorsOut
        -bool _idle
        +WiredTigerSession(conn, cache, epoch)
        +getSession() WT_SESSION*
        +getCursor(uri, overwrite, config) WT_CURSOR*
        +releaseCursor(id, cursor) void
        +closeCursor(uri) void
        +closeAllCursors() void
        +getIdleExpireTime() uint64_t
        +markIdle() void
        +markActive() void
        -_openCursor(uri, overwrite, config) WT_CURSOR*
    }

    class WiredTigerSnapshotManager {
        -WiredTigerKVEngine* _engine
        -WT_CONNECTION* _conn
        -mutable Mutex _mutex
        -bool _inShutdown
        -map~Timestamp, WT_SESSION*~ _committedSnapshots
        +WiredTigerSnapshotManager(engine, conn)
        +setCommittedSnapshot(timestamp) void
        +cleanupUnneededSnapshots() void
        +getMinSnapshotForNextCommittedRead() Timestamp
        +takeGlobalLock(opCtx) void
        +shutdown() void
        -_makeSnapshot(session, timestamp) void
    }

    class WiredTigerOplogManager {
        -WiredTigerKVEngine* _engine
        -unique_ptr~WiredTigerRecordStore~ _oplogRecordStore
        -mutable Mutex _oplogReadTimestampMutex
        -Timestamp _oplogMaxAtStartup
        -Timestamp _oplogReadTimestamp
        -AtomicWord~unsigned long long~ _oplogCounter
        +WiredTigerOplogManager(engine)
        +start(opCtx, oplogNS) void
        +halt() void
        +getOplogReadTimestamp() Timestamp
        +setOplogReadTimestamp(timestamp) void
        +getOplogMaxAtStartup() Timestamp
        +isRunning() bool
        -_oplogJournalThreadLoop() void
        -_updateOplogReadTimestamp(opCtx) void
    }

    WiredTigerKVEngine --> WiredTigerSessionCache : contains
    WiredTigerSessionCache --> WiredTigerSession : manages
    WiredTigerKVEngine --> WiredTigerSnapshotManager : uses
    WiredTigerKVEngine --> WiredTigerOplogManager : uses
    WiredTigerRecoveryUnit --> WiredTigerSession : uses
```

## 4. 存储特性和配置

### 4.1 存储配置类图

```mermaid
classDiagram
    class CollectionOptions {
        +bool capped
        +long long cappedSize
        +long long cappedMaxDocs
        +BSONObj storageEngine
        +BSONObj indexOptionDefaults
        +ValidationOptions validator
        +bool autoIndexId
        +BSONObj collation
        +boost::optional~UUID~ uuid
        +BSONObj encryptionInformation
        +CollectionOptions()
        +toBSON() BSONObj
        +fromBSON(obj) CollectionOptions
        +getStorageEngineOptions(engineName) BSONObj
    }

    class CappedCallback {
        <<abstract>>
        +aboutToDeleteCapped(opCtx, recordId, dataPtr) bool
        +cappedTruncateAfter(opCtx, recordId, inclusive) void
    }

    class RemoveSaver {
        -string _root
        -string _file
        -boost::filesystem::path _path
        -unique_ptr~boost::filesystem::ofstream~ _out
        +RemoveSaver(type, ns, why)
        +goingToDelete(obj) void
        +~RemoveSaver()
        -_findFileName() void
    }

    class IndexDescriptor {
        -BSONObj _keyPattern
        -BSONObj _infoObj
        -string _indexName
        -bool _isIdIndex
        -bool _isSparse
        -bool _isUnique
        -bool _isPartial
        -IndexVersion _version
        +IndexDescriptor(collection, accessMethodName, infoObj)
        +keyPattern() BSONObj
        +indexName() string
        +unique() bool
        +sparse() bool
        +partial() bool
        +getAccessMethodName() string
        +getEntry() IndexCatalogEntry*
        +areIndexOptionsEquivalent(other) bool
    }

    class WiredTigerKVEngineOptions {
        +bool directoryPerDB
        +bool journalCompressor
        +string engineConfig
        +string collectionConfig
        +string indexConfig
        +int cacheSizeGB
        +int maxCacheOverflowSizeGB
        +bool ephemeral
        +bool repair
        +WiredTigerKVEngineOptions()
        +setDirectoryPerDB(directoryPerDB) void
        +setCacheSizeGB(cacheSizeGB) void
        +getConfigString() string
    }

    class WiredTigerFileVersion {
        +enum Version
        +static const Version kLatestVersion
        +static hasStartupWarningsLogged() bool
        +static startupWarningsLogged() void
        +static shouldLog(warnType) bool
        +static getMinimumIndexVersion() IndexVersion
    }

    WiredTigerKVEngine --> CollectionOptions : uses
    RecordStore --> CappedCallback : uses
    WiredTigerKVEngine --> WiredTigerKVEngineOptions : configured by
    SortedDataInterface --> IndexDescriptor : uses
    WiredTigerKVEngine --> WiredTigerFileVersion : validates
```

## 5. 字段映射和约束

### 5.1 主要数据结构字段映射

| 数据结构 | 主要字段 | 数据类型 | 约束条件 | 说明 |
|---------|---------|----------|----------|------|
| WiredTigerKVEngine | _conn | WT_CONNECTION* | 非空 | WiredTiger数据库连接 |
| WiredTigerRecordStore | _tableURI | string | 非空，格式"table:xxx" | WiredTiger表URI |
| RecordId | _id | int64_t | > 0或特殊值 | 记录唯一标识符 |
| RecoveryUnit | _inUnitOfWork | bool | - | 是否在事务中 |
| WiredTigerSession | _session | WT_SESSION* | 非空 | WiredTiger会话句柄 |
| CollectionOptions | cappedSize | long long | >= 0 | 固定集合大小限制 |

### 5.2 存储引擎约束

| 组件 | 约束类型 | 限制值 | 说明 |
|------|---------|--------|------|
| WiredTiger | 最大文档大小 | 16MB | 单个BSON文档最大大小 |
| WiredTiger | 最大键长度 | 1024字节 | 索引键最大长度 |
| WiredTiger | 最大集合数 | 无限制 | 理论上无限制 |
| RecordStore | 固定集合大小 | 最小4KB | 固定集合最小大小 |
| Transaction | 最大事务大小 | 256MB | 单个事务最大修改数据量 |
| SessionCache | 最大会话数 | 1000 | 默认最大会话缓存数量 |

### 5.3 内存使用模式

| 组件 | 内存模式 | 默认大小 | 扩展策略 |
|------|---------|----------|----------|
| WiredTiger缓存 | 固定+动态 | 50%可用内存 | 自动调整 |
| 会话缓存 | 对象池 | 按需分配 | LRU淘汰 |
| 游标缓存 | 线程本地 | 按需创建 | 生命周期管理 |
| 元数据缓存 | 固定大小 | 64MB | 不扩展 |

---

**文档版本：** v1.0  
**生成时间：** 2025-10-05  
**适用版本：** MongoDB 8.0+
