---
title: "Apache Pulsar 关键数据结构 UML 分析"
date: 2025-09-28T00:47:17+08:00
draft: false
tags: ['Java', 'Apache Pulsar', '消息队列', '流处理', '架构设计']
categories: ['消息队列']
description: "Apache Pulsar 关键数据结构 UML 分析的深入技术分析文档"
keywords: ['Java', 'Apache Pulsar', '消息队列', '流处理', '架构设计']
author: "技术分析师"
weight: 1
---

## 1. 概述

本文档深入分析 Apache Pulsar 中的关键数据结构，通过 UML 类图和详细说明帮助读者理解各个组件之间的关系和设计模式。

## 2. 消息相关数据结构

### 2.1 Message 消息体系 UML 图

```mermaid
classDiagram
    class Message~T~ {
        <<interface>>
        +getKey() String
        +getValue() T
        +getMessageId() MessageId
        +getPublishTime() long
        +getEventTime() long
        +getProperties() Map~String, String~
        +hasKey() boolean
        +hasBase64EncodedKey() boolean
        +hasOrderingKey() boolean
        +getReaderSchema() Optional~Schema~T~~
    }
    
    class MessageImpl~T~ {
        -msgMetadata MessageMetadata
        -payload ByteBuf
        -messageId MessageIdImpl
        -cnx ClientCnx
        -schema Schema~T~
        -schemaState SchemaState
        +getKey() String
        +getValue() T
        +getDataBuffer() ByteBuf
        +getMessageBuilder() MessageMetadata
    }
    
    class MessageMetadata {
        -producerName String
        -sequenceId long
        -publishTime long
        -properties List~KeyValue~
        -partitionKey String
        -orderingKey ByteString
        -deliverAtTime long
        -eventTime long
        -compressionType CompressionType
        -uncompressedSize int
        -numMessagesInBatch int
        +getProducerName() String
        +getSequenceId() long
        +getPublishTime() long
        +hasPartitionKey() boolean
    }
    
    class MessageId {
        <<interface>>
        +toByteArray() byte[]
        +compareTo(MessageId) int
    }
    
    class MessageIdImpl {
        -ledgerId long
        -entryId long
        -partitionIndex int
        -batchIndex int
        -batchSize int
        +getLedgerId() long
        +getEntryId() long
        +getPartitionIndex() int
        +getBatchIndex() int
        +toString() String
        +equals(Object) boolean
        +hashCode() int
    }
    
    class TopicMessageImpl~T~ {
        -topicName String
        -topicPartitionName TopicName
        -ownerConsumer String
        +getTopicName() String
        +getOwnerConsumer() String
    }

    Message~T~ <|-- MessageImpl~T~
    Message~T~ <|-- TopicMessageImpl~T~
    MessageImpl~T~ *-- MessageMetadata : contains
    MessageImpl~T~ *-- MessageIdImpl : contains
    MessageId <|-- MessageIdImpl
    MessageImpl~T~ <-- TopicMessageImpl~T~ : wraps
```

### 2.2 Position 位置体系 UML 图

```mermaid
classDiagram
    class Position {
        <<interface>>
        +compareTo(Position) int
        +getNext() Position
    }
    
    class PositionImpl {
        -ledgerId long
        -entryId long
        +getLedgerId() long
        +getEntryId() long
        +compareTo(Position) int
        +getNext() Position
        +toString() String
        +create(long, long)$ PositionImpl
    }
    
    class Range~Position~ {
        -lowerBound Position
        -upperBound Position
        +lowerEndpoint() Position
        +upperEndpoint() Position
        +contains(Position) boolean
        +isEmpty() boolean
        +span(Range) Range
    }
    
    Position <|-- PositionImpl
    Range~Position~ o-- Position : bounds
```

## 3. 客户端核心数据结构

### 3.1 Producer 生产者体系 UML 图

```mermaid
classDiagram
    class Producer~T~ {
        <<interface>>
        +getTopic() String
        +getProducerName() String
        +send(T) MessageId
        +sendAsync(T) CompletableFuture~MessageId~
        +newMessage() TypedMessageBuilder~T~
        +flush() void
        +close() void
        +getStats() ProducerStats
        +getLastSequenceId() long
        +isConnected() boolean
    }
    
    class ProducerBase~T~ {
        <<abstract>>
        #producerCreatedFuture CompletableFuture~Producer~T~~
        #conf ProducerConfigurationData
        #schema Schema~T~
        #interceptors ProducerInterceptors
        +send(T) MessageId
        +sendAsync(T) CompletableFuture~MessageId~
        +newMessage() TypedMessageBuilder~T~
        #internalSendAsync(Message) CompletableFuture~MessageId~
        #triggerFlush() void
    }
    
    class ProducerImpl~T~ {
        -client PulsarClientImpl
        -topic String
        -producerName String
        -producerId long
        -sequenceIdGenerator AtomicLong
        -pendingMessages Queue~OpSendMsg~
        -semaphore Semaphore
        -batchMessageContainer BatchMessageContainer
        -stats ProducerStatsImpl
        +sendAsync(Message, SendCallback) void
        -processOpSendMsg(OpSendMsg) void
        -batchMessageAndSend(boolean) void
        -canAddToBatch(MessageImpl) boolean
        -doBatchSendAndAdd(MessageImpl, SendCallback, ByteBuf) void
    }
    
    class PartitionedProducerImpl~T~ {
        -producers List~ProducerImpl~T~~
        -routerPolicy MessageRouter
        -topicMetadata TopicMetadata
        +sendAsync(Message, SendCallback) void
        -getProducer(Message) ProducerImpl~T~
        -routeMessageToPartition(Message) int
    }
    
    class ProducerConfigurationData {
        -topicName String
        -producerName String
        -sendTimeoutMs long
        -blockIfQueueFull boolean
        -maxPendingMessages int
        -messageRoutingMode MessageRoutingMode
        -hashingScheme HashingScheme
        -cryptoKeyReader CryptoKeyReader
        -encryptionKeys Set~String~
        -compressionType CompressionType
        -batchingEnabled boolean
        -batchingMaxMessages int
        -batchingMaxPublishDelayMicros long
        -batchingMaxBytes int
        -chunkingEnabled boolean
    }

    Producer~T~ <|-- ProducerBase~T~
    ProducerBase~T~ <|-- ProducerImpl~T~
    ProducerBase~T~ <|-- PartitionedProducerImpl~T~
    ProducerImpl~T~ *-- ProducerConfigurationData : config
    PartitionedProducerImpl~T~ o-- ProducerImpl~T~ : contains
```

### 3.2 Consumer 消费者体系 UML 图

```mermaid
classDiagram
    class Consumer~T~ {
        <<interface>>
        +getTopic() String
        +getSubscription() String
        +receive() Message~T~
        +receiveAsync() CompletableFuture~Message~T~~
        +batchReceive() Messages~T~
        +acknowledge(Message) void
        +acknowledgeAsync(Message) CompletableFuture~Void~
        +acknowledgeCumulative(Message) void
        +negativeAcknowledge(Message) void
        +close() void
        +seek(MessageId) void
        +getStats() ConsumerStats
        +isConnected() boolean
    }
    
    class ConsumerBase~T~ {
        <<abstract>>
        #client PulsarClientImpl
        #subscription String
        #conf ConsumerConfigurationData
        #schema Schema~T~
        #interceptors ConsumerInterceptors
        #unAckedMessageTracker UnAckedMessageTracker
        #acknowledgementsGroupingTracker AcknowledgmentsGroupingTracker
        +receive(long, TimeUnit) Message~T~
        +acknowledge(Message) void
        +negativeAcknowledge(Message) void
        #internalReceive(long, TimeUnit) Message~T~
        #internalReceiveAsync() CompletableFuture~Message~T~~
        #beforeConsume(Message) Message~T~
        #processMessage(Message) void
    }
    
    class ConsumerImpl~T~ {
        -consumerId long
        -consumerName String
        -incomingMessages BlockingQueue~Message~T~~
        -pendingReceives ConcurrentLinkedQueue~CompletableFuture~Message~T~~~
        -availablePermits AtomicInteger
        -subscription Subscription
        -startMessageId MessageId
        -lastDequeuedMessageId MessageId
        -lastMessageIdInBroker MessageId
        -duringSeek AtomicBoolean
        -seekMessageId MessageId
        -startMessageRollbackDurationInSec int
        +internalReceive(long, TimeUnit) Message~T~
        +internalReceiveAsync() CompletableFuture~Message~T~~
        +increaseAvailablePermits(int) void
        +messageReceived(CommandMessage, ByteBuf, ClientCnx) void
    }
    
    class MultiTopicsConsumerImpl~T~ {
        -consumers ConcurrentHashMap~String, ConsumerImpl~T~~
        -topicsPattern Pattern
        -topics List~String~
        -allTopicPartitionsNumber AtomicInteger
        -pausedConsumers AtomicInteger
        -sharedQueueResumeThreshold int
        -maxReceiverQueueSize int
        +subscribe(String) CompletableFuture~Void~
        +unsubscribeAsync(String) CompletableFuture~Void~
        -messageReceived(ConsumerImpl, Message) void
        -receiveMessageFromConsumer(ConsumerImpl, boolean) void
    }
    
    class ConsumerConfigurationData {
        -topics Set~String~
        -topicsPattern Pattern
        -subscriptionName String
        -subscriptionType SubscriptionType
        -subscriptionInitialPosition SubscriptionInitialPosition
        -messageListener MessageListener~T~
        -consumerEventListener ConsumerEventListener
        -receiverQueueSize int
        -acknowledgementsGroupTimeMicros long
        -maxAcknowledgmentGroupSize int
        -negativeAckRedeliveryDelayMicros long
        -maxTotalReceiverQueueSizeAcrossPartitions int
        -consumerName String
        -ackTimeoutMillis long
        -tickDurationMillis long
        -priorityLevel int
        -maxPendingChuckedMessage int
        -cryptoKeyReader CryptoKeyReader
        -readCompacted boolean
        -subscriptionProperties Map~String, String~
        -patternAutoDiscoveryPeriod int
        -regexSubscriptionMode RegexSubscriptionMode
        -deadLetterPolicy DeadLetterPolicy
        -retryEnable boolean
        -batchReceivePolicy BatchReceivePolicy
    }

    Consumer~T~ <|-- ConsumerBase~T~
    ConsumerBase~T~ <|-- ConsumerImpl~T~
    ConsumerBase~T~ <|-- MultiTopicsConsumerImpl~T~
    ConsumerImpl~T~ *-- ConsumerConfigurationData : config
    MultiTopicsConsumerImpl~T~ o-- ConsumerImpl~T~ : contains
```

## 4. Broker 核心数据结构

### 4.1 Topic 主题体系 UML 图

```mermaid
classDiagram
    class Topic {
        <<interface>>
        +getName() String
        +addProducer(Producer, CompletableFuture) void
        +removeProducer(Producer) void
        +subscribe(String, Consumer, boolean) CompletableFuture~Consumer~
        +unsubscribe(Consumer) CompletableFuture~Void~
        +publishMessage(ByteBuf, PublishCallback) void
        +getStats(boolean, boolean, boolean) CompletableFuture~TopicStatsImpl~
        +getInternalStats(boolean) CompletableFuture~PersistentTopicInternalStats~
        +close() CompletableFuture~Void~
        +isActive() boolean
        +getLastPosition() Position
        +isFenced() boolean
    }
    
    class AbstractTopic {
        <<abstract>>
        #topic String
        #producers ConcurrentHashMap~String, Producer~
        #brokerService BrokerService
        #lock ReentrantReadWriteLock
        #isFenced boolean
        #topicPolicies HierarchyTopicPolicies
        #lastActive long
        #hasBatchMessagePublished boolean
        #isEncryptionRequired boolean
        #topicPublishRateLimiter PublishRateLimiter
        #resourceGroupPublishLimiter ResourceGroupPublishLimiter
        #bytesInCounter LongAdder
        #msgInCounter LongAdder
        +addProducer(Producer, CompletableFuture) void
        +removeProducer(Producer) void
        +checkPublishRate() boolean
        +updatePublishRateLimiter() void
        +incrementPublishCount(int, long) void
    }
    
    class PersistentTopic {
        -ledger ManagedLedger
        -subscriptions Map~String, PersistentSubscription~
        -replicators Map~String, Replicator~
        -shadowReplicators Map~String, Replicator~
        -dispatchRateLimiter Optional~DispatchRateLimiter~
        -subscribeRateLimiter Optional~SubscribeRateLimiter~
        -compactor Optional~Compactor~
        -topicCompactionService TopicCompactionService
        +publishMessage(ByteBuf, PublishCallback) void
        +subscribe(String, Consumer, boolean) CompletableFuture~Consumer~
        +addComplete(Position, ByteBuf, Object) void
        +addFailed(ManagedLedgerException, Object) void
        -createSubscription(String, CommandSubscribe, boolean, boolean) PersistentSubscription
        -checkBacklogQuota() void
    }
    
    class NonPersistentTopic {
        -subscriptions ConcurrentHashMap~String, NonPersistentSubscription~
        -replicators ConcurrentHashMap~String, NonPersistentReplicator~
        +publishMessage(ByteBuf, PublishCallback) void
        +subscribe(String, Consumer, boolean) CompletableFuture~Consumer~
        -createSubscription(String, CommandSubscribe) NonPersistentSubscription
    }

    Topic <|-- AbstractTopic
    AbstractTopic <|-- PersistentTopic
    AbstractTopic <|-- NonPersistentTopic
    PersistentTopic *-- ManagedLedger : contains
    PersistentTopic o-- PersistentSubscription : manages
```

### 4.2 Subscription 订阅体系 UML 图

```mermaid
classDiagram
    class Subscription {
        <<interface>>
        +getTopic() Topic
        +getName() String
        +getType() SubType
        +addConsumer(Consumer) CompletableFuture~Void~
        +removeConsumer(Consumer, boolean) CompletableFuture~Void~
        +consumerFlow(Consumer, int) void
        +acknowledgeMessage(List~Position~, AckType, Map~String, Long~) CompletableFuture~Void~
        +getNumberOfEntriesInBacklog(boolean) CompletableFuture~Long~
        +close() CompletableFuture~Void~
        +delete() CompletableFuture~Void~
        +disconnect() CompletableFuture~Void~
        +getStats(GetStatsOptions) SubscriptionStatsImpl
    }
    
    class AbstractSubscription {
        <<abstract>>
        #topic AbstractTopic
        #subName String
        #cursor ManagedCursor
        #IS_FENCED_UPDATER AtomicIntegerFieldUpdater
        #isFenced int
        #recentlyJoinedConsumers Map~Consumer, Long~
        #subscriptionProperties Map~String, String~
        +getName() String
        +getTopic() Topic
        +isCursorActive() boolean
        +checkAndApplyReachedEndOfTopicOrTopicMigration(List) boolean
        +getNumberOfEntriesInBacklog(boolean) CompletableFuture~Long~
        #acknowledgeMessage(Position, AckType, Map) CompletableFuture~Void~
    }
    
    class PersistentSubscription {
        -dispatcher Dispatcher
        -subscriptionProperties ConcurrentHashMap~String, String~
        -lastExpiredTimestamp long
        -totalNonContiguousDeletedMessagesRange int
        -subscriptionStatsUnsafeMode boolean
        +addConsumer(Consumer) CompletableFuture~Void~
        +removeConsumer(Consumer, boolean) CompletableFuture~Void~
        +consumerFlow(Consumer, int) void
        +acknowledgeMessage(List, AckType, Map) CompletableFuture~Void~
        +expireMessages(int) void
        +redeliverUnacknowledgedMessages(Consumer, List) void
        -createDispatcher(Consumer) Dispatcher
        -checkAndApplyReachedEndOfTopic() void
    }
    
    class PersistentDispatcherMultipleConsumers {
        -consumerList List~Consumer~
        -consumerSet ConcurrentHashMap~Consumer, Boolean~
        -partitionedTopicConsumers ConcurrentHashMap~Consumer, String~
        -readType ReadType
        -sendInProgress AtomicBoolean
        -readFailureBackoff Backoff
        -totalAvailablePermits int
        -messagesToReplay ConcurrentLinkedQueue~MessageId~
        +addConsumer(Consumer) synchronized CompletableFuture~Void~
        +removeConsumer(Consumer) synchronized CompletableFuture~Void~
        +consumerFlow(Consumer, int) void
        +canUnsubscribe(Consumer) boolean
        -readMoreEntries() void
        -dispatchMessagesToConsumers(List) void
    }
    
    class PersistentDispatcherSingleActiveConsumer {
        -activeConsumer Consumer
        -readOnActiveConsumerTask ScheduledFuture
        +addConsumer(Consumer) CompletableFuture~Void~
        +removeConsumer(Consumer) CompletableFuture~Void~
        +consumerFlow(Consumer, int) void
        -pickAndScheduleActiveConsumer() void
        -scheduleReadOnActiveConsumer() void
        -cancelPendingRead() void
    }

    Subscription <|-- AbstractSubscription
    AbstractSubscription <|-- PersistentSubscription
    PersistentSubscription *-- Dispatcher : contains
    Dispatcher <|-- PersistentDispatcherMultipleConsumers
    Dispatcher <|-- PersistentDispatcherSingleActiveConsumer
```

## 5. 存储相关数据结构

### 5.1 ManagedLedger 存储体系 UML 图

```mermaid
classDiagram
    class ManagedLedger {
        <<interface>>
        +getName() String
        +asyncAddEntry(ByteBuf, AddEntryCallback, Object) void
        +asyncOpenCursor(String, OpenCursorCallback, Object) void
        +asyncDeleteCursor(String, DeleteCursorCallback, Object) void
        +getCursors() List~ManagedCursor~
        +getLastConfirmedEntry() Position
        +getFirstPosition() Position
        +closeAsync() CompletableFuture~Void~
        +deleteAsync() CompletableFuture~Void~
        +terminateAsync() CompletableFuture~Position~
        +isTerminated() boolean
        +getEstimatedBacklogSize() long
    }
    
    class ManagedLedgerImpl {
        -bookKeeper BookKeeper
        -name String
        -ledgerMetadata Map~String, byte[]~
        -config ManagedLedgerConfig
        -store MetaStore
        -ledgers NavigableMap~Long, LedgerInfo~
        -currentLedger LedgerHandle
        -cursors ManagedCursorContainer
        -activeCursors ActiveManagedCursorContainer
        -entryCache EntryCache
        -lastConfirmedEntry Position
        -state State
        -clock Clock
        +asyncAddEntry(ByteBuf, AddEntryCallback, Object) void
        +asyncOpenCursor(String, OpenCursorCallback, Object) void
        -createLedgerAfterClosed() CompletableFuture~Void~
        -rollCurrentLedgerIfFull() void
        +initialize(ManagedLedgerInitializeLedgerCallback, Object) void
        -scheduledExecutor OrderedScheduler
        -mbean ManagedLedgerMBeanImpl
    }
    
    class ManagedCursor {
        <<interface>>
        +getName() String
        +readEntries(int) List~Entry~
        +asyncReadEntries(int, ReadEntriesCallback, Object) void
        +asyncMarkDelete(Position, MarkDeleteCallback, Object) void
        +asyncDelete(List, DeleteCallback, Object) void
        +seek(Position, boolean) void
        +rewind() void
        +getReadPosition() Position
        +getMarkDeletedPosition() Position
        +hasMoreEntries() boolean
        +getNumberOfEntries() long
        +getTotalSize() long
        +close() void
    }
    
    class ManagedCursorImpl {
        -bookkeeper BookKeeper
        -ledger ManagedLedgerImpl
        -name String
        -markDeletePosition Position
        -readPosition Position
        -lock ReadWriteLock
        -state State
        -lastActive long
        -cursorProperties Map~String, String~
        -individualDeletedMessages RangeSet~Position~
        -batchDeletedIndexes RangeSet~Position~
        +readEntries(int) List~Entry~
        +asyncReadEntries(int, ReadEntriesCallback, Object) void
        +asyncMarkDelete(Position, MarkDeleteCallback, Object) void
        +seek(Position, boolean) void
        +rewind() void
        -persistPositionToMetaStore(Position, MarkDeleteCallback, Object) void
        -setAcknowledgedPosition(Position) Position
    }
    
    class Entry {
        <<interface>>
        +getLedgerId() long
        +getEntryId() long
        +getPosition() Position
        +getLength() int
        +getData() ByteBuf
        +getDataBuffer() ByteBuf
        +release() int
        +retain() Entry
        +retain(int) Entry
    }
    
    class EntryImpl {
        -ledgerId long
        -entryId long
        -data ByteBuf
        +getLedgerId() long
        +getEntryId() long
        +getLength() int
        +getData() ByteBuf
        +release() int
        +toString() String
        +create(LedgerEntry, int)$ EntryImpl
        +create(long, long, ByteBuf)$ EntryImpl
    }

    ManagedLedger <|-- ManagedLedgerImpl
    ManagedCursor <|-- ManagedCursorImpl
    Entry <|-- EntryImpl
    ManagedLedgerImpl *-- ManagedCursorImpl : manages
    ManagedLedgerImpl *-- EntryCache : uses
    ManagedCursorImpl --> Entry : reads
```

## 6. 网络通信数据结构

### 6.1 网络连接体系 UML 图

```mermaid
classDiagram
    class ClientCnx {
        -channel Channel
        -state State
        -connectionHandler ConnectionHandler
        -pendingRequests ConcurrentHashMap~Long, CompletableFuture~
        -producers ConcurrentHashMap~Long, ProducerImpl~
        -consumers ConcurrentHashMap~Long, ConsumerImpl~
        -remoteAddress SocketAddress
        -proxyToTargetBrokerAddress String
        -lastDataReceivedTime long
        -operationTimeoutMs long
        +sendRequestWithId(ByteBuf, long) CompletableFuture~Void~
        +newLookup(ByteBuf, long) CompletableFuture~LookupDataResult~
        +newConsumer(Topic, Subscription, long, long, String, boolean, InitialPosition, SchemaInfo, boolean) CompletableFuture~Void~
        +newProducer(String, long, String, boolean, Map, SchemaInfo, long, boolean, TxnID, long, Optional) CompletableFuture~Void~
        +removeProducer(long) CompletableFuture~Void~
        +removeConsumer(long) CompletableFuture~Void~
        -handleResponse(ByteBuf) void
        -handleActiveConsumerChange(CommandActiveConsumerChange) void
        -handleMessage(CommandMessage, ByteBuf) void
    }
    
    class ServerCnx {
        -service BrokerService
        -cnx SocketAddress
        -authState AuthenticationState
        -authRole String
        -authenticationData AuthenticationDataSource
        -originalPrincipal String
        -originalAuthData AuthenticationDataSource
        -originalAuthMethod String
        -producers ConcurrentHashMap~Long, Producer~
        -consumers ConcurrentHashMap~Long, Consumer~
        -remoteAddress SocketAddress
        -authMethod String
        -supportsTopicWatchers boolean
        +handleConnect(CommandConnect) void
        +handleSubscribe(CommandSubscribe) void
        +handleProducer(CommandProducer) void
        +handleSend(CommandSend, ByteBuf) void
        +handleAck(CommandAck) void
        +handleFlow(CommandFlow) void
        +handleUnsubscribe(CommandUnsubscribe) void
        +handleCloseProducer(CommandCloseProducer) void
        +handleCloseConsumer(CommandCloseConsumer) void
        -checkAuth(CompletableFuture, String) void
        -isTopicOperationAllowed(TopicName, TopicOperation) CompletableFuture~Boolean~
    }
    
    class PulsarRequestIdGenerator {
        -requestId AtomicLong
        +nextId() long
    }
    
    class Commands {
        +newConnect(String, String, String) ByteBuf
        +newSubscribe(String, String, long, long, SubType, int, String) ByteBuf
        +newProducer(String, long, String, boolean, Map) ByteBuf
        +newSend(long, long, int, ChecksumType, MessageMetadata, ByteBuf) ByteBuf
        +newAck(long, AckType, List, ValidationError, Map) ByteBuf
        +newFlow(long, int) ByteBuf
        +newCloseProducer(long, long) ByteBuf
        +newCloseConsumer(long, long) ByteBuf
        +parseMessageMetadata(ByteBuf) MessageMetadata
        +parseBrokerEntryMetadataIfExist(ByteBuf) BrokerEntryMetadata
    }

    ClientCnx *-- PulsarRequestIdGenerator : uses
    ServerCnx --> Commands : uses
    ClientCnx --> Commands : uses
```

## 7. 配置数据结构

### 7.1 配置体系 UML 图

```mermaid
classDiagram
    class ServiceConfiguration {
        -clusterName String
        -brokerServicePort Optional~Integer~
        -brokerServicePortTls Optional~Integer~
        -webServicePort Optional~Integer~
        -webServicePortTls Optional~Integer~
        -bindAddress String
        -advertisedAddress String
        -numIOThreads int
        -numHttpServerThreads int
        -zookeeperServers String
        -configurationStoreServers String
        -authenticationEnabled boolean
        -authorizationEnabled boolean
        -superUserRoles Set~String~
        -managedLedgerDefaultEnsembleSize int
        -managedLedgerDefaultWriteQuorum int
        -managedLedgerDefaultAckQuorum int
        -managedLedgerMaxEntriesPerLedger int
        -managedLedgerMaxSizePerLedgerMB int
        -loadBalancerEnabled boolean
        +getClusterName() String
        +getBrokerServicePort() Optional~Integer~
        +isAuthenticationEnabled() boolean
        +getManagedLedgerDefaultEnsembleSize() int
    }
    
    class ClientConfigurationData {
        -serviceUrl String
        -authPluginClassName String
        -authParams String
        -authParamMap Map~String, String~
        -operationTimeoutMs long
        -statsIntervalSeconds long
        -numIoThreads int
        -numListenerThreads int
        -connectionsPerBroker int
        -useTls boolean
        -tlsTrustCertsFilePath String
        -tlsAllowInsecureConnection boolean
        -tlsHostnameVerificationEnable boolean
        -concurrentLookupRequest int
        -maxLookupRequest int
        -maxLookupRedirects int
        -maxNumberOfRejectedRequestPerConnection int
        -keepAliveIntervalSeconds int
        -connectionTimeoutMs int
        -requestTimeoutMs long
        +getServiceUrl() String
        +getOperationTimeoutMs() long
        +getNumIoThreads() int
    }
    
    class ProducerConfigurationData {
        -topicName String
        -producerName String
        -sendTimeoutMs long
        -blockIfQueueFull boolean
        -maxPendingMessages int
        -maxPendingMessagesAcrossPartitions int
        -messageRoutingMode MessageRoutingMode
        -hashingScheme HashingScheme
        -compressionType CompressionType
        -batchingEnabled boolean
        -batchingMaxMessages int
        -batchingMaxPublishDelayMicros long
        -batchingMaxBytes int
        -batchingPartitionSwitchFrequencyByPublishDelay int
        -cryptoKeyReader CryptoKeyReader
        -encryptionKeys Set~String~
        -chunkingEnabled boolean
        -chunkMaxMessageSize int
        +getTopicName() String
        +getSendTimeoutMs() long
        +isBatchingEnabled() boolean
    }
    
    class ConsumerConfigurationData {
        -topics Set~String~
        -topicsPattern Pattern
        -subscriptionName String
        -subscriptionType SubscriptionType
        -subscriptionInitialPosition SubscriptionInitialPosition
        -messageListener MessageListener
        -consumerEventListener ConsumerEventListener
        -receiverQueueSize int
        -acknowledgementsGroupTimeMicros long
        -maxAcknowledgmentGroupSize int
        -negativeAckRedeliveryDelayMicros long
        -maxTotalReceiverQueueSizeAcrossPartitions int
        -consumerName String
        -ackTimeoutMillis long
        -priorityLevel int
        -cryptoKeyReader CryptoKeyReader
        -readCompacted boolean
        -subscriptionProperties Map~String, String~
        -patternAutoDiscoveryPeriod int
        -regexSubscriptionMode RegexSubscriptionMode
        -deadLetterPolicy DeadLetterPolicy
        -retryEnable boolean
        -batchReceivePolicy BatchReceivePolicy
        +getTopics() Set~String~
        +getSubscriptionName() String
        +getSubscriptionType() SubscriptionType
    }

    ServiceConfiguration --> "uses" ProducerConfigurationData : validates
    ServiceConfiguration --> "uses" ConsumerConfigurationData : validates
    ClientConfigurationData --> "creates" ProducerConfigurationData : factory
    ClientConfigurationData --> "creates" ConsumerConfigurationData : factory
```

## 8. 统计监控数据结构

### 8.1 统计体系 UML 图

```mermaid
classDiagram
    class TopicStats {
        <<interface>>
        +getName() String
        +getPublishers() List~PublisherStatsImpl~
        +getSubscriptions() Map~String, SubscriptionStatsImpl~
        +getReplication() Map~String, ReplicatorStatsImpl~
        +getInboundConnections() List~String~
        +getOutboundConnections() List~String~
        +getMsgRateIn() double
        +getMsgRateOut() double
        +getMsgThroughputIn() double
        +getMsgThroughputOut() double
        +getAverageMsgSize() double
        +getStorageSize() long
        +getBacklogSize() long
    }
    
    class TopicStatsImpl {
        +msgRateIn double
        +msgThroughputIn double
        +msgRateOut double
        +msgThroughputOut double
        +averageMsgSize double
        +storageSize long
        +backlogSize long
        +publishers List~PublisherStatsImpl~
        +subscriptions Map~String, SubscriptionStatsImpl~
        +replication Map~String, ReplicatorStatsImpl~
        +deduplicationStatus String
        +nonContiguousDeletedMessagesRanges int
        +nonContiguousDeletedMessagesRangesSerializedSize int
        +delayedMessageIndexSizeInBytes long
        +bucketDelayedIndexStats BucketDelayedDeliveryTrackerStats
    }
    
    class PublisherStatsImpl {
        +msgRateIn double
        +msgThroughputIn double
        +averageMsgSize double
        +producerId long
        +producerName String
        +address String
        +connectedSince String
        +clientVersion String
        +metadata Map~String, String~
        +accessMode ProducerAccessMode
        +chunkingEnabled boolean
    }
    
    class SubscriptionStatsImpl {
        +msgRateOut double
        +msgThroughputOut double
        +msgBacklog long
        +backlogSize long
        +msgRateRedeliver double
        +type String
        +msgRateExpired double
        +totalMsgExpired long
        +lastExpiredTimestamp long
        +lastMarkDeleteAdvancedTimestamp long
        +lastConsumedFlowTimestamp long
        +lastConsumedTimestamp long
        +lastAckedTimestamp long
        +consumersAfterMarkDeletePosition Map~String, String~
        +nonContiguousDeletedMessagesRanges int
        +nonContiguousDeletedMessagesRangesSerializedSize int
        +delayedMessageIndexSizeInBytes long
        +bucketDelayedIndexStats BucketDelayedDeliveryTrackerStats
        +filterProcessedMsgCount long
        +filterAcceptedMsgCount long
        +filterRejectedMsgCount long
        +filterRescheduledMsgCount long
        +durable boolean
        +replicated boolean
        +consumers List~ConsumerStatsImpl~
    }
    
    class ConsumerStatsImpl {
        +msgRateOut double
        +msgThroughputOut double
        +msgRateRedeliver double
        +consumerName String
        +availablePermits int
        +unackedMessages int
        +avgMessagesPerEntry double
        +blockedConsumerOnUnackedMsgs boolean
        +address String
        +connectedSince String
        +clientVersion String
        +lastAckedTimestamp long
        +lastConsumedTimestamp long
        +keyHashRanges List~String~
        +metadata Map~String, String~
    }

    TopicStats <|-- TopicStatsImpl
    TopicStatsImpl o-- PublisherStatsImpl
    TopicStatsImpl o-- SubscriptionStatsImpl
    SubscriptionStatsImpl o-- ConsumerStatsImpl
```

## 9. Schema 数据结构

### 9.1 Schema 体系 UML 图

```mermaid
classDiagram
    class Schema~T~ {
        <<interface>>
        +encode(T) byte[]
        +decode(byte[]) T
        +decode(byte[], byte[]) T
        +getSchemaInfo() SchemaInfo
        +validate(byte[]) boolean
        +supportSchemaVersioning() boolean
        +setSchemaInfoProvider(SchemaInfoProvider) void
        +requireFetchingSchemaInfo() boolean
        +configureSchemaInfo(String, String, SchemaInfo) void
        +clone() Schema~T~
    }
    
    class SchemaInfo {
        -name String
        -schema byte[]
        -type SchemaType
        -timestamp long
        -properties Map~String, String~
        +getName() String
        +getSchema() byte[]
        +getType() SchemaType
        +getTimestamp() long
        +getProperties() Map~String, String~
        +toString() String
    }
    
    class AvroSchema~T~ {
        -avroSchema org.apache.avro.Schema
        -schemaInfo SchemaInfo
        -reader GenericDatumReader~GenericRecord~
        -writer GenericDatumWriter~GenericRecord~
        +encode(T) byte[]
        +decode(byte[]) T
        +getSchemaInfo() SchemaInfo
        +validate(byte[]) boolean
        +getAvroSchema() org.apache.avro.Schema
    }
    
    class JsonSchema~T~ {
        -objectMapper ObjectMapper
        -pojo Class~T~
        -schemaInfo SchemaInfo
        +encode(T) byte[]
        +decode(byte[]) T
        +getSchemaInfo() SchemaInfo
        +validate(byte[]) boolean
    }
    
    class ProtobufSchema~T~ {
        -protoMessage T
        -schemaInfo SchemaInfo
        +encode(T) byte[]
        +decode(byte[]) T
        +getSchemaInfo() SchemaInfo
        +validate(byte[]) boolean
        +parseFrom(byte[]) T
    }
    
    class PrimitiveSchema~T~ {
        -schemaInfo SchemaInfo
        +encode(T) byte[]
        +decode(byte[]) T
        +getSchemaInfo() SchemaInfo
        +validate(byte[]) boolean
    }
    
    class KeyValueSchema~K,V~ {
        -keySchema Schema~K~
        -valueSchema Schema~V~
        -keyValueEncodingType KeyValueEncodingType
        -schemaInfo SchemaInfo
        +encode(KeyValue) byte[]
        +decode(byte[]) KeyValue~K,V~
        +getSchemaInfo() SchemaInfo
    }

    Schema~T~ <|-- AvroSchema~T~
    Schema~T~ <|-- JsonSchema~T~
    Schema~T~ <|-- ProtobufSchema~T~
    Schema~T~ <|-- PrimitiveSchema~T~
    Schema~T~ <|-- KeyValueSchema~K,V~
    Schema~T~ *-- SchemaInfo : contains
```

## 10. 总结

本文档通过详细的 UML 类图展示了 Apache Pulsar 中的关键数据结构和它们之间的关系。这些数据结构构成了 Pulsar 的核心架构，理解它们的设计和关系对于：

1. **系统设计**：帮助理解 Pulsar 的整体架构设计思路
2. **性能优化**：识别性能瓶颈和优化点  
3. **功能扩展**：基于现有结构进行功能扩展和定制
4. **问题排查**：深入理解系统行为，快速定位问题
5. **代码贡献**：为 Pulsar 社区贡献代码和功能

每个数据结构都有其特定的职责和设计考虑，它们通过清晰的接口和继承关系协同工作，构建了一个高性能、可扩展的分布式消息系统。
