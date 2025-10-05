---
title: "MongoDB-12-复制模块-数据结构"
date: 2025-10-05T12:13:30Z
draft: false
tags: ["MongoDB", "源码剖析", "复制模块", "数据结构", "副本集"]
categories: ["mongodb"]
series: ["MongoDB源码剖析"]
weight: 19
description: "MongoDB复制模块数据结构详解：复制协调器、Oplog管理器、选举管理器等核心数据结构"
---

# MongoDB-12-复制模块-数据结构

## 1. 核心数据结构概览

复制模块的数据结构设计实现了MongoDB副本集的完整功能，包括数据复制、故障转移、一致性保证等核心特性。数据结构按功能分为几个主要层次：

### 1.1 数据结构分类

- **协调器层：** ReplicationCoordinator、TopologyCoordinator
- **配置管理层：** ReplSetConfig、MemberConfig
- **时间戳管理层：** OpTime、OpTimeAndWallTime
- **同步处理层：** InitialSyncer、OplogApplier、OplogFetcher
- **存储接口层：** StorageInterface、ReplicationConsistencyMarkers
- **状态管理层：** MemberState、ReplicationMode

## 2. 核心类图

### 2.1 复制协调器类图

```mermaid
classDiagram
    class ReplicationCoordinator {
        <<abstract>>
        +startup(opCtx, storageEngine) void
        +shutdown(opCtx, shutdownBuilder) void
        +getReplicationMode() ReplicationMode
        +awaitReplication(opCtx, opTime, writeConcern) StatusAndDuration
        +stepUp(opCtx, obj, force, skipDryRun) StatusWith~OpTime~
        +stepDown(opCtx, force, waitTime, stepdownTime) Status
        +processReplSetReconfig(opCtx, args, resultObj) Status
        +setFollowerMode(newState) Status
        +isInPrimaryOrSecondaryState() bool
        +getMyLastAppliedOpTime() OpTime
        +getMyLastDurableOpTime() OpTime
    }

    class ReplicationCoordinatorImpl {
        -unique_ptr~TopologyCoordinator~ _topCoord
        -unique_ptr~ReplicationExecutor~ _replExecutor
        -ReplSetConfig _rsConfig
        -MemberState _memberState
        -OpTime _lastCommittedOpTime
        -vector~MemberData~ _memberData
        -ReplicationWaiterList _replicationWaiterList
        -BackgroundSync* _bgsync
        -mutable Mutex _mutex
        +startup(opCtx, storageEngine) void
        +awaitReplication(opCtx, opTime, writeConcern) StatusAndDuration
        +stepUp(opCtx, obj, force, skipDryRun) StatusWith~OpTime~
        -_validateWriteConcern(writeConcern) Status
        -_isOpTimeReplicatedEnough(opTime, writeConcern) bool
        -_wakeReadyWaiters(opTime) void
    }

    class TopologyCoordinator {
        -Role _role
        -long long _leaderTerm
        -Date_t _electionTime
        -Date_t _electionTimeoutWhen
        -ReplSetConfig _rsConfig
        -vector~MemberData~ _memberData
        -OpTime _lastCommittedOpTime
        +processWinElection(electionOpTime) void
        +processLoseElection() void
        +processHeartbeatResponse(now, networkTime, target, response) HeartbeatResponseAction
        +checkShouldStandForElection(now) bool
        +getRole() Role
        +getTerm() long long
        +setElectionTimeoutOffset(offset) void
        -_shouldStartElection(now) bool
        -_findMemberIndexByHostAndPort(host) int
    }

    class ReplicationExecutor {
        -ThreadPool _threadPool
        -executor::TaskExecutor* _executor
        -NetworkInterface* _networkInterface
        +scheduleWork(callback) StatusWith~CallbackHandle~
        +scheduleWorkAt(when, callback) StatusWith~CallbackHandle~
        +scheduleRemoteCommand(request, callback) StatusWith~CallbackHandle~
        +cancel(handle) void
        +wait(handle) void
        +shutdown() void
    }

    ReplicationCoordinator <|-- ReplicationCoordinatorImpl
    ReplicationCoordinatorImpl --> TopologyCoordinator : contains
    ReplicationCoordinatorImpl --> ReplicationExecutor : uses
    TopologyCoordinator --> MemberData : manages
```

### 2.2 副本集配置类图

```mermaid
classDiagram
    class ReplSetConfig {
        -BSONObj _raw
        -string _replSetName
        -ConfigVersionAndTerm _versionAndTerm
        -vector~MemberConfig~ _members
        -WriteConcernOptions _defaultWriteConcern
        -ReplicaSetTag _heartbeatTimeoutPeriod
        -ReplicaSetTag _electionTimeoutPeriod
        -bool _chainingAllowed
        -long long _heartbeatIntervalMillis
        +getReplSetName() string
        +getConfigVersion() long long
        +getConfigTerm() long long
        +getNumMembers() int
        +getMemberAt(i) MemberConfig&
        +findMemberByHostAndPort(host) MemberConfig*
        +findMemberById(id) MemberConfig*
        +validate() Status
        +isValidMajority(memberIds) bool
        +getMajorityVoteCount() int
        +getWriteMajority() int
    }

    class MemberConfig {
        -int _id
        -HostAndPort _host
        -int _priority
        -Seconds _slaveDelay
        -bool _hidden
        -bool _buildIndexes
        -int _votes
        -ReplicaSetTagConfig _tags
        -bool _arbiterOnly
        -Seconds _secondaryDelaySecs
        +getId() int
        +getHostAndPort() HostAndPort
        +getPriority() int
        +isVoter() bool
        +isArbiter() bool
        +isHidden() bool
        +shouldBuildIndexes() bool
        +getSlaveDelay() Seconds
        +hasTags() bool
        +getTags() ReplicaSetTagConfig
    }

    class ConfigVersionAndTerm {
        -long long version
        -long long term
        +ConfigVersionAndTerm(v, t)
        +getVersion() long long
        +getTerm() long long
        +operator<(other) bool
        +operator==(other) bool
        +toString() string
    }

    class ReplicaSetTagConfig {
        -map~string, string~ _tags
        +ReplicaSetTagConfig()
        +addTag(key, value) void
        +hasTag(key) bool
        +getTag(key) string
        +getAllTags() map~string, string~
        +matches(pattern) bool
        +toBSON() BSONObj
    }

    class WriteConcernOptions {
        -string wMode
        -int wNumNodes
        -int wTimeout
        -bool jValue
        -bool fsync
        +WriteConcernOptions()
        +WriteConcernOptions(w, timeout, j, fsync)
        +isMajority() bool
        +shouldWaitForOtherNodes() bool
        +isValid() bool
        +toBSON() BSONObj
    }

    ReplSetConfig --> MemberConfig : contains
    ReplSetConfig --> ConfigVersionAndTerm : has
    ReplSetConfig --> WriteConcernOptions : contains
    MemberConfig --> ReplicaSetTagConfig : has
```

### 2.3 时间戳和操作时间类图

```mermaid
classDiagram
    class OpTime {
        -Timestamp _timestamp
        -long long _term
        +OpTime()
        +OpTime(timestamp, term)
        +getTimestamp() Timestamp
        +getTerm() long long
        +isNull() bool
        +operator<(other) bool
        +operator==(other) bool
        +operator<=(other) bool
        +toString() string
        +toBSON() BSONObj
        +fromBSON(obj) OpTime
    }

    class OpTimeAndWallTime {
        -OpTime opTime
        -Date_t wallTime
        +OpTimeAndWallTime()
        +OpTimeAndWallTime(opTime, wallTime)
        +getOpTime() OpTime
        +getWallTime() Date_t
        +toString() string
        +toBSON() BSONObj
    }

    class Timestamp {
        -uint64_t _value
        +Timestamp()
        +Timestamp(seconds, increment)
        +getSeconds() uint32_t
        +getInc() uint32_t
        +isNull() bool
        +operator<(other) bool
        +operator==(other) bool
        +toString() string
        +toBSONElement(fieldName) BSONElement
    }

    class LastCommittedOpTimeTracker {
        -OpTime _lastCommittedOpTime
        -mutable Mutex _mutex
        -stdx::condition_variable _opTimeChangedCV
        +updateLastCommittedOpTime(opTime) void
        +getLastCommittedOpTime() OpTime
        +waitForOpTime(opTime, timeout) Status
        +notifyAll() void
    }

    OpTime --> Timestamp : contains
    OpTimeAndWallTime --> OpTime : contains
    LastCommittedOpTimeTracker --> OpTime : tracks
```

### 2.4 成员状态和数据类图

```mermaid
classDiagram
    class MemberState {
        -int _state
        +static PRIMARY int
        +static SECONDARY int
        +static ROLLBACK int
        +static RECOVERING int
        +static STARTUP int
        +static STARTUP2 int
        +static UNKNOWN int
        +static DOWN int
        +static REMOVED int
        +MemberState()
        +MemberState(state)
        +primary() bool
        +secondary() bool
        +rollback() bool
        +recovering() bool
        +readable() bool
        +writable() bool
        +toString() string
        +operator==(other) bool
    }

    class MemberData {
        -int _configIndex
        -Date_t _lastUpdateTime
        -OpTime _lastAppliedOpTime
        -OpTime _lastDurableOpTime
        -Date_t _lastAppliedWallTime
        -Date_t _lastDurableWallTime
        -MemberState _state
        -Date_t _electionTime
        -string _syncSource
        -bool _isSelf
        -HostAndPort _hostAndPort
        +getConfigIndex() int
        +getLastAppliedOpTime() OpTime
        +getLastDurableOpTime() OpTime
        +getState() MemberState
        +isSelf() bool
        +getHostAndPort() HostAndPort
        +setUpValues(now, state, electionTime, appliedOpTime, durableOpTime) void
        +setDownValues(now, reason) void
        +getLastUpdateTime() Date_t
        +getSyncSource() string
    }

    class ReplicationMode {
        <<enumeration>>
        kNone
        kPrimary
        kSecondary
        kInitialSync
    }

    class Role {
        <<enumeration>>
        kFollower
        kCandidate
        kLeader
    }

    class HeartbeatResponseAction {
        -ActionType _action
        -string _reason
        +makeNoAction() HeartbeatResponseAction
        +makeElectAction() HeartbeatResponseAction
        +makeStepDownRemotePrimaryAction(primaryIndex) HeartbeatResponseAction
        +getAction() ActionType
        +getReason() string
    }

    MemberData --> MemberState : has
    MemberData --> OpTime : tracks
    TopologyCoordinator --> Role : has
    ReplicationCoordinator --> ReplicationMode : reports
```

### 2.5 初始同步数据结构类图

```mermaid
classDiagram
    class InitialSyncer {
        -InitialSyncState _state
        -uint32_t _attempts
        -uint32_t _maxAttempts
        -string _currentPhase
        -size_t _totalBytesToClone
        -size_t _totalBytesCloned
        -executor::TaskExecutor* _executor
        -ReplicationProcess* _replProcess
        -StorageInterface* _storage
        -mutable Mutex _mutex
        +startup(opCtx, maxAttempts) StatusWith~OpTimeAndWallTime~
        +shutdown() void
        +getState() InitialSyncState
        +getInitialSyncProgress() BSONObj
        -_runInitialSync(opCtx) StatusWith~OpTimeAndWallTime~
        -_chooseSyncSource(opCtx) HostAndPort
        -_cloneDatabases(opCtx, client) Status
        -_applyOplog(opCtx, client, beginTimestamp) StatusWith~OpTimeAndWallTime~
    }

    class InitialSyncState {
        <<enumeration>>
        kInactive
        kRunning
        kComplete
        kFailed
    }

    class DatabaseCloner {
        -string _dbName
        -vector~CollectionCloner~ _collectionCloners
        -executor::TaskExecutor* _executor
        -StorageInterface* _storage
        -DBClientConnection* _client
        -BaseCloner::ClonerExecStatus _status
        +run() Status
        +getStats() DatabaseClonerStats
        -_listCollections() StatusWith~vector~BSONObj~~
        -_createCollections(collections) Status
        -_cloneCollections() Status
    }

    class CollectionCloner {
        -NamespaceString _nss
        -CollectionOptions _options
        -BSONObj _idIndexSpec
        -vector~BSONObj~ _nonIdIndexSpecs
        -unique_ptr~CollectionBulkLoader~ _collectionLoader
        -executor::TaskExecutor* _executor
        -size_t _documentsCloned
        -size_t _bytesCloned
        +run() Status
        +getStats() CollectionClonerStats
        -_createCollection() Status
        -_listIndexes() Status
        -_cloneDocuments() Status
        -_createIndexes() Status
    }

    class BaseCloner {
        <<abstract>>
        -ReplSyncSharedData* _sharedData
        -DBClientConnection* _client
        -StorageInterface* _storageInterface
        -ThreadPool* _dbPool
        -HostAndPort _source
        -bool _active
        -Status _status
        +run() Status
        +runOnExecutorEvent(executor) Future~void~
        +isActive() bool
        +getStatus() Status
        #runStages() AfterStageBehavior
    }

    class ReplSyncSharedData {
        -OpTime _beginOpTime
        -OpTime _stopOpTime
        -int _totalBytesCopied
        -mutable Mutex _mutex
        +setBeginOpTime(opTime) void
        +getBeginOpTime() OpTime
        +setStopOpTime(opTime) void
        +getStopOpTime() OpTime
        +addBytesCopied(bytes) void
        +getTotalBytesCopied() int
    }

    InitialSyncer --> InitialSyncState : has
    InitialSyncer --> DatabaseCloner : uses
    DatabaseCloner --> CollectionCloner : contains
    DatabaseCloner --|> BaseCloner : extends
    CollectionCloner --|> BaseCloner : extends
    BaseCloner --> ReplSyncSharedData : uses
```

### 2.6 Oplog处理数据结构类图

```mermaid
classDiagram
    class OplogEntry {
        -BSONObj _raw
        -OpTime _opTime
        -OpTypeEnum _opType
        -NamespaceString _nss
        -BSONObj _o
        -boost::optional~BSONObj~ _o2
        -boost::optional~BSONObj~ _preImageOpTime
        -boost::optional~BSONObj~ _postImageOpTime
        -boost::optional~long long~ _version
        +getOpTime() OpTime
        +getOpType() OpTypeEnum
        +getNss() NamespaceString
        +getObject() BSONObj
        +getObject2() boost::optional~BSONObj~
        +isCommand() bool
        +isCrudOpType() bool
        +toBSON() BSONObj
        +fromBSON(obj) OplogEntry
    }

    class OpTypeEnum {
        <<enumeration>>
        kInsert
        kDelete
        kUpdate
        kCommand
        kNoop
    }

    class OplogApplier {
        -ApplyMode _applyMode
        -vector~MultiApplier::OperationPtrs~ _writerPool
        -ReplicationConsistencyMarkers* _consistencyMarkers
        -StorageInterface* _storage
        -OplogApplierStats _stats
        +applyOplogBatch(opCtx, entries) StatusWith~OpTime~
        +getStats() OplogApplierStats
        +setApplyMode(mode) void
        -_applyOplogEntryOrGroupedInserts(opCtx, entry, mode) Status
        -_applyCommand(opCtx, entry, mode) Status
        -_applyInsert(opCtx, entry, mode) Status
        -_applyUpdate(opCtx, entry, mode) Status
        -_applyDelete(opCtx, entry, mode) Status
    }

    class ApplyMode {
        <<enumeration>>
        kInitialSync
        kSecondary
        kRecovering
    }

    class OplogFetcher {
        -HostAndPort _source
        -OpTime _lastFetched
        -unique_ptr~DBClientConnection~ _conn
        -executor::TaskExecutor* _executor
        -OplogFetcher::OnOplogEntryBatchFn _onOplogEntryBatchFn
        -int _batchSize
        -Milliseconds _maxNetworkTimeoutMS
        +startup() Status
        +shutdown() void
        +getSource() HostAndPort
        +getLastOpTimeFetched() OpTime
        -_runQuery(opCtx) Status
        -_processBatch(batch) Status
        -_createNewCursor() StatusWith~unique_ptr~DBClientCursor~~
    }

    class OplogApplierStats {
        +long long appliedOps
        +long long totalOps
        +Date_t startTime
        +Date_t endTime
        +long long insertedDocuments
        +long long updatedDocuments
        +long long deletedDocuments
        +OplogApplierStats()
        +toBSON() BSONObj
    }

    OplogEntry --> OpTypeEnum : has
    OplogApplier --> ApplyMode : uses
    OplogApplier --> OplogApplierStats : produces
    OplogApplier --> OplogEntry : processes
    OplogFetcher --> OplogEntry : fetches
```

## 3. 存储接口数据结构

### 3.1 存储接口和一致性标记类图

```mermaid
classDiagram
    class StorageInterface {
        <<abstract>>
        +insertDocument(opCtx, nss, doc, term) Status
        +insertDocuments(opCtx, nss, docs) Status
        +deleteByFilter(opCtx, nss, filter) Status
        +findSingleton(opCtx, nss) StatusWith~BSONObj~
        +createOplog(opCtx, nss) Status
        +createCollection(opCtx, nss, options) Status
        +dropDatabase(opCtx, db) Status
        +getCollectionSize(opCtx, nss) StatusWith~size_t~
        +getCollectionCount(opCtx, nss) StatusWith~size_t~
        +setInitialSyncId(opCtx, id) Status
        +getInitialSyncId(opCtx) StatusWith~UUID~
    }

    class StorageInterfaceImpl {
        +insertDocument(opCtx, nss, doc, term) Status
        +insertDocuments(opCtx, nss, docs) Status  
        +createOplog(opCtx, nss) Status
        +findSingleton(opCtx, nss) StatusWith~BSONObj~
        +getCollectionSize(opCtx, nss) StatusWith~size_t~
        -_insertDocumentToCollection(opCtx, coll, doc) Status
        -_findOrCreateCollection(opCtx, nss) StatusWith~Collection*~
    }

    class ReplicationConsistencyMarkers {
        <<abstract>>
        +getMinValid(opCtx) StatusWith~OpTime~
        +setMinValid(opCtx, minValid) Status
        +getAppliedThrough(opCtx) StatusWith~OpTime~
        +setAppliedThrough(opCtx, opTime) Status
        +getInitialSyncFlag(opCtx) StatusWith~bool~
        +setInitialSyncFlag(opCtx) Status
        +clearInitialSyncFlag(opCtx) Status
        +getOplogTruncateAfterPoint(opCtx) StatusWith~Timestamp~
        +setOplogTruncateAfterPoint(opCtx, timestamp) Status
    }

    class ReplicationConsistencyMarkersImpl {
        -StorageInterface* _storageInterface
        -NamespaceString _minValidNss
        -NamespaceString _oplogTruncateAfterPointNss
        +getMinValid(opCtx) StatusWith~OpTime~
        +setMinValid(opCtx, minValid) Status
        +getAppliedThrough(opCtx) StatusWith~OpTime~
        +setAppliedThrough(opCtx, opTime) Status
        -_updateMinValidDocument(opCtx, updateSpec) Status
        -_readMinValidDocument(opCtx) StatusWith~BSONObj~
    }

    class TimestampedBSONObj {
        +BSONObj obj
        +Timestamp timestamp
        +TimestampedBSONObj(o, ts)
    }

    class MinValidDocument {
        +Timestamp minValidTimestamp
        +long long minValidTerm  
        +boost::optional~OpTime~ appliedThrough
        +boost::optional~bool~ initialSyncFlag
        +boost::optional~ObjectId~ _id
        +toBSON() BSONObj
        +fromBSON(obj) MinValidDocument
    }

    StorageInterface <|-- StorageInterfaceImpl
    ReplicationConsistencyMarkers <|-- ReplicationConsistencyMarkersImpl
    ReplicationConsistencyMarkersImpl --> StorageInterface : uses
    StorageInterface --> TimestampeDBSONObj : uses
    ReplicationConsistencyMarkers --> MinValidDocument : manages
```

## 4. 网络和心跳数据结构

### 4.1 心跳和网络通信类图

```mermaid
classDiagram
    class ReplSetHeartbeatArgs {
        -string _setName
        -long long _configVersion
        -long long _configTerm
        -long long _senderId
        -OpTime _lastCommittedOpTime
        -OpTime _lastAppliedOpTime
        -HostAndPort _senderHost
        +getSetName() string
        +getConfigVersion() long long
        +getConfigTerm() long long
        +getSenderId() long long
        +getLastCommittedOpTime() OpTime
        +getLastAppliedOpTime() OpTime
        +getSenderHost() HostAndPort
        +toBSON() BSONObj
        +fromBSON(obj) ReplSetHeartbeatArgs
    }

    class ReplSetHeartbeatResponse {
        -string _setName
        -long long _configVersion
        -long long _configTerm
        -MemberState _state
        -OpTime _lastCommittedOpTime
        -OpTime _lastAppliedOpTime
        -OpTime _lastDurableOpTime
        -Date_t _electionTime
        -HostAndPort _syncSource
        -bool _hasConfig
        -ReplSetConfig _config
        +getSetName() string
        +getState() MemberState
        +getElectionTime() Date_t
        +getAppliedOpTime() OpTime
        +getDurableOpTime() OpTime
        +getSyncSource() HostAndPort
        +hasConfig() bool
        +getConfig() ReplSetConfig
        +toBSON() BSONObj
        +fromBSON(obj) ReplSetHeartbeatResponse
    }

    class ReplSetRequestVotesArgs {
        -long long _term
        -long long _candidateId
        -long long _configVersion
        -long long _configTerm
        -string _setName
        -OpTime _lastAppliedOpTime
        -OpTime _lastDurableOpTime
        -bool _dryRun
        +getTerm() long long
        +getCandidateId() long long
        +getSetName() string
        +getLastAppliedOpTime() OpTime
        +isDryRun() bool
        +toBSON() BSONObj
        +fromBSON(obj) ReplSetRequestVotesArgs
    }

    class ReplSetRequestVotesResponse {
        -long long _term
        -bool _voteGranted
        -string _reason
        +getTerm() long long
        +getVoteGranted() bool
        +getReason() string
        +toBSON() BSONObj
        +fromBSON(obj) ReplSetRequestVotesResponse
    }

    class UpdatePositionArgs {
        -long long _cfgVersion
        -long long _memberId
        -OpTime _appliedOpTime
        -OpTime _durableOpTime
        -Date_t _appliedWallTime
        -Date_t _durableWallTime
        +getConfigVersion() long long
        +getMemberId() long long
        +getAppliedOpTime() OpTime
        +getDurableOpTime() OpTime
        +toBSON() BSONObj
        +fromBSON(obj) UpdatePositionArgs
    }

    ReplSetHeartbeatArgs --> OpTime : contains
    ReplSetHeartbeatResponse --> MemberState : has
    ReplSetHeartbeatResponse --> OpTime : contains
    ReplSetRequestVotesArgs --> OpTime : contains
    ReplSetRequestVotesResponse --> VoteGranted : indicates
    UpdatePositionArgs --> OpTime : tracks
```

## 5. 选举和投票数据结构

### 5.1 选举过程数据结构类图

```mermaid
classDiagram
    class VoteRequester {
        -executor::TaskExecutor* _executor
        -ReplSetConfig _rsConfig
        -long long _term
        -OpTime _lastAppliedOpTime
        -OpTime _lastDurableOpTime
        -int _primaryIndex
        -bool _dryRun
        -VoteRequester::Algorithm _algorithm
        -vector~RemoteCommandCallbackArgs~ _responses
        +start(executor, args, onCompletion) Status
        +cancel() void
        +getResponses() vector~RemoteCommandCallbackArgs~
        -_requestVoteCallback(args) void
        -_onRequestVoteResponse(cbData) void
    }

    class FreshnessChecker {
        -executor::TaskExecutor* _executor
        -ReplSetConfig _rsConfig
        -int _candidateIndex
        -OpTime _lastAppliedOpTime
        -vector~HostAndPort~ _targets
        -FreshnessChecker::Algorithm _algorithm
        +start(executor, args, onCompletion) Status
        +cancel() void
        -_processFreshnessResponse(cbData) void
        -_isElectable(memberIndex, response) bool
    }

    class ElectCmdRunner {
        -executor::TaskExecutor* _executor
        -ReplSetConfig _rsConfig
        -int _primaryIndex
        -string _whyElectable
        -ElectCmdRunner::Algorithm _algorithm
        +start(executor, args, onCompletion) Status
        +cancel() void
        -_processElectResponse(cbData) void
    }

    class ScatterGatherAlgorithm {
        <<abstract>>
        -executor::TaskExecutor* _executor
        -vector~RemoteCommandRequest~ _requests
        -ScatterGatherRunner::RunnerState _state
        +start() Status
        +cancel() void
        +hasReceivedSufficientResponses() bool
        #_processSingleResponse(cbData) void
    }

    class Algorithm {
        <<enumeration>>
        kElectV1
        kDryRun
        kPriorityTakeover
        kStepDownRemotePrimary
    }

    VoteRequester --> Algorithm : uses
    FreshnessChecker --> Algorithm : uses
    ElectCmdRunner --> Algorithm : uses
    VoteRequester --|> ScatterGatherAlgorithm : extends
    FreshnessChecker --|> ScatterGatherAlgorithm : extends
    ElectCmdRunner --|> ScatterGatherAlgorithm : extends
```

## 6. 字段映射和约束

### 6.1 主要数据结构字段映射

| 数据结构 | 主要字段 | 数据类型 | 约束条件 | 说明 |
|---------|---------|----------|----------|------|
| OpTime | _timestamp | Timestamp | 非空 | 操作时间戳 |
| OpTime | _term | long long | >= -1 | 选举任期，-1表示PV0 |
| ReplSetConfig | _replSetName | string | 非空，最大127字符 | 副本集名称 |
| ReplSetConfig | _members | vector<MemberConfig> | 1-50个成员 | 成员配置列表 |
| MemberConfig | _id | int | 0-255 | 成员唯一ID |
| MemberConfig | _priority | int | 0-1000 | 选举优先级 |
| MemberConfig | _votes | int | 0-1 | 投票权重 |
| WriteConcernOptions | wNumNodes | int | >= 0 | 需要确认的节点数 |
| WriteConcernOptions | wTimeout | int | >= 0 | 等待超时时间(毫秒) |

### 6.2 状态转换约束

| 状态类型 | 允许的转换 | 限制条件 |
|---------|-----------|----------|
| MemberState | PRIMARY → SECONDARY | 只能通过stepDown |
| MemberState | SECONDARY → PRIMARY | 需要选举胜利 |
| MemberState | STARTUP → SECONDARY | 初始同步完成 |
| InitialSyncState | kInactive → kRunning | 调用startup() |
| InitialSyncState | kRunning → kComplete | 同步成功完成 |
| InitialSyncState | kRunning → kFailed | 超过最大重试次数 |

### 6.3 版本兼容性

| 协议版本 | 支持特性 | 兼容性 |
|---------|---------|--------|
| PV0 | 旧版本选举协议 | 已废弃 |
| PV1 | Raft-based选举 | 当前标准 |
| v4.4+ | 多数写关注 | 向后兼容 |
| v5.0+ | 快照读取 | 向后兼容 |

---

**文档版本：** v1.0  
**生成时间：** 2025-10-05  
**适用版本：** MongoDB 8.0+
