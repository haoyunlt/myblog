---
title: "Apache Pulsar 最佳实践和实战经验"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['Java', 'Apache Pulsar', '消息队列', '流处理', '最佳实践']
categories: ['pulsar', '消息队列']
description: "Apache Pulsar 最佳实践和实战经验的深入技术分析文档"
keywords: ['Java', 'Apache Pulsar', '消息队列', '流处理', '最佳实践']
author: "技术分析师"
weight: 1
---

## 1. 概述

本文档基于生产环境的实际经验，总结了使用 Apache Pulsar 的最佳实践、常见问题及解决方案，帮助开发者和运维人员更好地部署、配置和使用 Pulsar。

## 2. 架构设计最佳实践

### 2.1 集群规划和部署

#### 2.1.1 硬件配置建议

**Broker 节点配置**：
```yaml
# 生产环境推荐配置
CPU: 16-32 核心
内存: 64-128GB RAM
网络: 10Gbps
存储: SSD 用于日志和缓存

# 关键配置参数
# broker.conf
numIOThreads=16
numHttpServerThreads=16
managedLedgerDefaultEnsembleSize=3
managedLedgerDefaultWriteQuorum=3
managedLedgerDefaultAckQuorum=2
managedLedgerCacheSizeMB=8192
```

**BookKeeper 节点配置**：
```yaml
# 硬件配置
CPU: 8-16 核心
内存: 32-64GB RAM
Journal 磁盘: 高速 SSD，独立磁盘
数据磁盘: SSD 或高速 HDD

# bookkeeper.conf
journalDirectories=/mnt/journal
ledgerDirectories=/mnt/data1,/mnt/data2,/mnt/data3
dbStorage_writeCacheMaxSizeMb=2048
dbStorage_readAheadCacheMaxSizeMb=1024
nettyMaxFrameSizeBytes=52428800
```

#### 2.1.2 网络和防火墙配置

```bash
# Pulsar 端口规划
# Broker
6650    # Pulsar 二进制协议
6651    # Pulsar TLS 协议  
8080    # HTTP 管理接口
8443    # HTTPS 管理接口

# BookKeeper
3181    # Bookie 端口
8000    # Bookie HTTP 端口

# ZooKeeper
2181    # 客户端连接
2888    # Follower 连接 Leader
3888    # Leader 选举

# 防火墙规则示例
iptables -A INPUT -p tcp --dport 6650 -j ACCEPT
iptables -A INPUT -p tcp --dport 6651 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
iptables -A INPUT -p tcp --dport 3181 -j ACCEPT
```

### 2.2 主题设计模式

#### 2.2.1 主题命名规范

```java
// 推荐的主题命名规范
// 格式：{租户}/{命名空间}/{应用}-{功能}-{环境}
"company/production/user-events-v1"
"company/production/order-processing-prod" 
"company/staging/notification-service-test"

// 避免的命名方式
"topic1"  // 无语义
"user_events_prod_v1_final"  // 过长且混乱
"UserEvents"  // 使用大写字母
```

#### 2.2.2 分区策略

```java
/**
 * 分区数量计算公式
 * 分区数 = (目标吞吐量 MB/s) / (单分区吞吐量 MB/s)
 * 
 * 生产环境建议：
 * - 高吞吐量主题：16-64 个分区
 * - 中等吞吐量主题：4-16 个分区  
 * - 低吞吐量主题：1-4 个分区
 */

// 创建分区主题
pulsar-admin topics create-partitioned-topic \
    --partitions 16 \
    persistent://company/production/high-throughput-topic

// 动态扩展分区（谨慎使用）
pulsar-admin topics update-partitioned-topic \
    --partitions 32 \
    persistent://company/production/high-throughput-topic
```

#### 2.2.3 消息路由策略

```java
// 按业务键路由 - 保证顺序性
Producer<OrderEvent> producer = client.newProducer(Schema.AVRO(OrderEvent.class))
    .topic("persistent://ecommerce/orders/order-events")
    .messageRouter(new MessageRouter() {
        @Override
        public int choosePartition(Message<?> msg, TopicMetadata metadata) {
            // 根据订单ID路由到固定分区，保证同一订单的消息有序
            String orderId = msg.getKey();
            return Math.abs(orderId.hashCode()) % metadata.numPartitions();
        }
    })
    .create();

// 轮询路由 - 最大化吞吐量
Producer<UserEvent> producer = client.newProducer(Schema.JSON(UserEvent.class))
    .topic("persistent://company/events/user-events") 
    .messageRouter(MessageRouter.RoundRobinPartition)
    .create();
```

## 3. 生产者最佳实践

### 3.1 性能优化配置

```java
public class OptimizedProducerExample {
    
    public static Producer<String> createHighPerformanceProducer(PulsarClient client) {
        return client.newProducer(Schema.STRING)
            .topic("high-performance-topic")
            .producerName("optimized-producer-" + UUID.randomUUID())
            
            // 批量发送优化
            .enableBatching(true)
            .batchingMaxMessages(1000)               // 批量消息数量
            .batchingMaxPublishDelay(10, TimeUnit.MILLISECONDS)  // 批量延迟
            .batchingMaxBytes(128 * 1024)            // 批量大小 128KB
            .batcherBuilder(BatcherBuilder.DEFAULT)  // 使用默认批处理器
            
            // 压缩配置
            .compressionType(CompressionType.LZ4)    // LZ4 压缩，平衡性能和压缩率
            
            // 发送配置
            .sendTimeout(30, TimeUnit.SECONDS)       // 发送超时
            .blockIfQueueFull(true)                  // 队列满时阻塞而不是抛异常
            .maxPendingMessages(10000)               // 最大挂起消息数
            
            // 异步发送配置
            .accessMode(ProducerAccessMode.Shared)   // 允许多个生产者
            .create();
    }
    
    public static void demonstrateBestPractices() throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .ioThreads(Runtime.getRuntime().availableProcessors())  // IO 线程数
            .connectionsPerBroker(1)                 // 每个 Broker 连接数
            .operationTimeout(30, TimeUnit.SECONDS)
            .build();
        
        Producer<String> producer = createHighPerformanceProducer(client);
        
        // 使用异步发送以获得最佳性能
        List<CompletableFuture<MessageId>> futures = new ArrayList<>();
        
        for (int i = 0; i < 10000; i++) {
            String message = "Message-" + i;
            String key = "Key-" + (i % 100);  // 用于分区路由
            
            CompletableFuture<MessageId> future = producer.newMessage()
                .key(key)
                .value(message)
                .property("timestamp", String.valueOf(System.currentTimeMillis()))
                .property("source", "producer-demo")
                .sendAsync();
                
            futures.add(future);
            
            // 每1000条消息检查一次发送状态
            if (i % 1000 == 0) {
                try {
                    // 等待前面的消息发送完成
                    CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get(5, TimeUnit.SECONDS);
                    futures.clear();
                    System.out.println("Sent " + (i + 1) + " messages");
                } catch (TimeoutException e) {
                    System.err.println("发送超时，检查网络和 Broker 状态");
                }
            }
        }
        
        // 确保所有消息都发送完成
        producer.flush();
        
        producer.close();
        client.close();
    }
}
```

### 3.2 错误处理和重试机制

```java
public class RobustProducerExample {
    
    private static final Logger log = LoggerFactory.getLogger(RobustProducerExample.class);
    
    public static void demonstrateErrorHandling(PulsarClient client) {
        Producer<String> producer = null;
        
        try {
            producer = client.newProducer(Schema.STRING)
                .topic("robust-topic")
                .sendTimeout(10, TimeUnit.SECONDS)
                .create();
            
            for (int i = 0; i < 1000; i++) {
                sendMessageWithRetry(producer, "Message-" + i, 3);
            }
            
        } catch (PulsarClientException e) {
            log.error("Failed to create producer", e);
        } finally {
            if (producer != null) {
                try {
                    producer.close();
                } catch (PulsarClientException e) {
                    log.error("Failed to close producer", e);
                }
            }
        }
    }
    
    /**
     * 带重试机制的消息发送
     */
    private static void sendMessageWithRetry(Producer<String> producer, String message, int maxRetries) {
        int retryCount = 0;
        
        while (retryCount <= maxRetries) {
            try {
                producer.sendAsync(message)
                    .thenAccept(messageId -> {
                        log.debug("消息发送成功: {} -> {}", message, messageId);
                    })
                    .exceptionally(ex -> {
                        handleSendException(ex, message, retryCount);
                        return null;
                    });
                return; // 发送成功，退出重试循环
                
            } catch (Exception e) {
                retryCount++;
                log.warn("消息发送失败 (重试 {}/{}): {}", retryCount, maxRetries, e.getMessage());
                
                if (retryCount > maxRetries) {
                    log.error("消息发送最终失败: {}", message, e);
                    // 可以将失败的消息写入死信队列或错误日志
                    handleFinalSendFailure(message, e);
                    break;
                } else {
                    // 指数退避重试
                    try {
                        Thread.sleep(1000L * retryCount);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }
    }
    
    private static void handleSendException(Throwable ex, String message, int retryCount) {
        if (ex instanceof PulsarClientException.TimeoutException) {
            log.warn("发送超时: {}", message);
        } else if (ex instanceof PulsarClientException.ProducerQueueIsFullError) {
            log.warn("生产者队列已满: {}", message);
        } else if (ex instanceof PulsarClientException.TopicDoesNotExistException) {
            log.error("主题不存在: {}", message);
        } else {
            log.error("未知发送错误: {}", message, ex);
        }
    }
    
    private static void handleFinalSendFailure(String message, Throwable e) {
        // 实现失败消息处理逻辑
        // 例如：写入数据库、发送到死信队列、记录错误日志等
        log.error("消息最终发送失败，需要人工处理: message={}", message, e);
    }
}
```

## 4. 消费者最佳实践

### 4.1 消费模式选择

```java
public class ConsumerPatternExamples {
    
    /**
     * 独占订阅模式 - 保证消息顺序
     * 适用场景：需要严格消息顺序的业务场景
     */
    public static Consumer<String> createExclusiveConsumer(PulsarClient client) throws PulsarClientException {
        return client.newConsumer(Schema.STRING)
            .topic("order-processing")
            .subscriptionName("order-processor")
            .subscriptionType(SubscriptionType.Exclusive)  // 独占模式
            .subscriptionInitialPosition(SubscriptionInitialPosition.Earliest)
            .receiverQueueSize(1000)
            .ackTimeout(30, TimeUnit.SECONDS)
            .subscribe();
    }
    
    /**
     * 共享订阅模式 - 最大化吞吐量
     * 适用场景：高吞吐量，不需要严格顺序的业务场景
     */
    public static Consumer<String> createSharedConsumer(PulsarClient client) throws PulsarClientException {
        return client.newConsumer(Schema.STRING)
            .topic("user-events")
            .subscriptionName("user-event-processors")
            .subscriptionType(SubscriptionType.Shared)     // 共享模式
            .receiverQueueSize(1000)
            .maxUnackedMessages(5000)                      // 最大未确认消息数
            .ackTimeout(60, TimeUnit.SECONDS)
            .negativeAckRedeliveryDelay(1, TimeUnit.MINUTES)
            .subscribe();
    }
    
    /**
     * Key_Shared 订阅模式 - 按键分区的并行处理
     * 适用场景：需要按键保证顺序，但允许不同键并行处理
     */
    public static Consumer<UserEvent> createKeySharedConsumer(PulsarClient client) throws PulsarClientException {
        return client.newConsumer(Schema.JSON(UserEvent.class))
            .topic("user-activity")
            .subscriptionName("user-activity-processors")
            .subscriptionType(SubscriptionType.Key_Shared)  // Key_Shared 模式
            .keySharedPolicy(KeySharedPolicy.autoSplitHashRange())
            .receiverQueueSize(1000)
            .subscribe();
    }
}
```

### 4.2 高效的消息处理

```java
public class EfficientMessageProcessing {
    
    private static final Logger log = LoggerFactory.getLogger(EfficientMessageProcessing.class);
    private final ExecutorService processingExecutor;
    private final Semaphore processingPermits;
    
    public EfficientMessageProcessing(int processingThreads) {
        this.processingExecutor = Executors.newFixedThreadPool(processingThreads);
        this.processingPermits = new Semaphore(processingThreads * 2); // 允许一定的缓冲
    }
    
    /**
     * 批量消息处理 - 提高处理效率
     */
    public void processBatchMessages(PulsarClient client) throws PulsarClientException {
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("batch-processing-topic")
            .subscriptionName("batch-processor")
            .subscriptionType(SubscriptionType.Shared)
            .batchReceivePolicy(BatchReceivePolicy.builder()
                .maxNumMessages(100)                        // 批量大小
                .maxNumBytes(1024 * 1024)                   // 1MB
                .timeout(100, TimeUnit.MILLISECONDS)        // 批量超时
                .build())
            .subscribe();
        
        while (!Thread.currentThread().isInterrupted()) {
            try {
                Messages<String> messages = consumer.batchReceive();
                if (messages.size() > 0) {
                    processBatchMessagesAsync(consumer, messages);
                }
            } catch (PulsarClientException e) {
                log.error("Failed to receive batch messages", e);
                Thread.sleep(1000); // 短暂休眠后重试
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        
        consumer.close();
    }
    
    private void processBatchMessagesAsync(Consumer<String> consumer, Messages<String> messages) {
        try {
            processingPermits.acquire();
            
            processingExecutor.submit(() -> {
                try {
                    List<Message<String>> successMessages = new ArrayList<>();
                    List<Message<String>> failedMessages = new ArrayList<>();
                    
                    for (Message<String> message : messages) {
                        try {
                            // 处理单个消息
                            processMessage(message);
                            successMessages.add(message);
                        } catch (Exception e) {
                            log.error("处理消息失败: {}", message.getMessageId(), e);
                            failedMessages.add(message);
                        }
                    }
                    
                    // 批量确认成功的消息
                    if (!successMessages.isEmpty()) {
                        Message<String> lastSuccess = successMessages.get(successMessages.size() - 1);
                        consumer.acknowledge(lastSuccess);
                        log.debug("批量确认 {} 条消息", successMessages.size());
                    }
                    
                    // 拒绝失败的消息，触发重试
                    for (Message<String> failedMessage : failedMessages) {
                        consumer.negativeAcknowledge(failedMessage);
                    }
                    
                } finally {
                    processingPermits.release();
                }
            });
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    /**
     * 消息处理业务逻辑
     */
    private void processMessage(Message<String> message) throws Exception {
        String content = message.getValue();
        String messageKey = message.getKey();
        Map<String, String> properties = message.getProperties();
        
        // 实现具体的业务逻辑
        log.debug("处理消息: key={}, content={}, properties={}", messageKey, content, properties);
        
        // 模拟处理时间
        Thread.sleep(10);
        
        // 可能抛出业务异常
        if (content.contains("error")) {
            throw new BusinessException("业务处理失败: " + content);
        }
    }
    
    /**
     * 实现消息监听器模式
     */
    public Consumer<String> createAsyncConsumer(PulsarClient client) throws PulsarClientException {
        return client.newConsumer(Schema.STRING)
            .topic("async-processing")
            .subscriptionName("async-processor")
            .subscriptionType(SubscriptionType.Shared)
            .messageListener(new MessageListener<String>() {
                @Override
                public void received(Consumer<String> consumer, Message<String> message) {
                    processingExecutor.submit(() -> {
                        try {
                            processMessage(message);
                            consumer.acknowledge(message);
                        } catch (Exception e) {
                            log.error("异步处理消息失败", e);
                            consumer.negativeAcknowledge(message);
                        }
                    });
                }
            })
            .subscribe();
    }
    
    public void shutdown() {
        processingExecutor.shutdown();
        try {
            if (!processingExecutor.awaitTermination(30, TimeUnit.SECONDS)) {
                processingExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            processingExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    private static class BusinessException extends Exception {
        public BusinessException(String message) {
            super(message);
        }
    }
}
```

## 5. Schema 管理最佳实践

### 5.1 Schema 演化策略

```java
public class SchemaEvolutionExample {
    
    // 版本1：初始用户事件结构
    public static class UserEventV1 {
        private String userId;
        private String eventType;
        private long timestamp;
        
        // 构造函数、getter、setter省略
    }
    
    // 版本2：添加新字段（向后兼容）
    public static class UserEventV2 {
        private String userId;
        private String eventType;
        private long timestamp;
        private String deviceId;      // 新增字段
        private Map<String, String> metadata; // 新增字段
        
        // 构造函数、getter、setter省略
    }
    
    /**
     * 演示 Schema 演化的最佳实践
     */
    public static void demonstrateSchemaEvolution(PulsarClient client) throws Exception {
        
        // 生产者使用新版本 Schema
        Schema<UserEventV2> producerSchema = Schema.AVRO(UserEventV2.class);
        Producer<UserEventV2> producer = client.newProducer(producerSchema)
            .topic("user-events-with-schema")
            .create();
        
        // 发送新版本消息
        UserEventV2 event = new UserEventV2();
        event.setUserId("user123");
        event.setEventType("login");
        event.setTimestamp(System.currentTimeMillis());
        event.setDeviceId("mobile-001");
        event.setMetadata(Map.of("ip", "192.168.1.1", "userAgent", "Chrome"));
        
        producer.send(event);
        
        // 旧版本消费者仍能正常工作（向后兼容）
        Schema<UserEventV1> consumerSchema = Schema.AVRO(UserEventV1.class);
        Consumer<UserEventV1> consumer = client.newConsumer(consumerSchema)
            .topic("user-events-with-schema")
            .subscriptionName("legacy-processor")
            .subscribe();
        
        Message<UserEventV1> message = consumer.receive();
        UserEventV1 receivedEvent = message.getValue();
        // 新字段会被忽略，旧字段正常解析
        
        consumer.acknowledge(message);
        
        producer.close();
        consumer.close();
    }
    
    /**
     * Schema 兼容性验证
     */
    public static void validateSchemaCompatibility() {
        // 建议在 CI/CD 流程中添加 Schema 兼容性检查
        SchemaInfo oldSchemaInfo = SchemaInfo.builder()
            .name("UserEvent")
            .type(SchemaType.AVRO)
            .schema(getAvroSchema(UserEventV1.class))
            .build();
            
        SchemaInfo newSchemaInfo = SchemaInfo.builder()
            .name("UserEvent")
            .type(SchemaType.AVRO)
            .schema(getAvroSchema(UserEventV2.class))
            .build();
        
        // 使用 Avro 的 Schema 兼容性检查
        boolean isCompatible = checkSchemaCompatibility(oldSchemaInfo, newSchemaInfo);
        
        if (!isCompatible) {
            throw new RuntimeException("Schema 不兼容，需要创建新的主题或调整 Schema");
        }
    }
    
    private static byte[] getAvroSchema(Class<?> clazz) {
        // 实现获取 Avro Schema 的逻辑
        return new byte[0]; // 简化示例
    }
    
    private static boolean checkSchemaCompatibility(SchemaInfo oldSchema, SchemaInfo newSchema) {
        // 实现 Schema 兼容性检查逻辑
        return true; // 简化示例
    }
}
```

## 6. 运维监控最佳实践

### 6.1 关键监控指标

```yaml
# Prometheus 监控配置示例
groups:
- name: pulsar-broker
  rules:
  # 吞吐量监控
  - alert: PulsarLowThroughput
    expr: rate(pulsar_in_messages_total[5m]) < 100
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Pulsar throughput is low"
      description: "Messages per second: {{ $value }}"

  # 延迟监控
  - alert: PulsarHighLatency
    expr: histogram_quantile(0.95, rate(pulsar_publish_latency_seconds_bucket[5m])) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Pulsar publish latency is high"
      description: "95th percentile latency: {{ $value }}s"

  # 积压监控
  - alert: PulsarHighBacklog
    expr: pulsar_subscription_back_log > 10000
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "High message backlog"
      description: "Subscription {{ $labels.subscription }} has {{ $value }} messages in backlog"
      
  # 连接监控
  - alert: PulsarHighConnections
    expr: pulsar_active_connections > 1000
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High number of connections"
      description: "Active connections: {{ $value }}"
```

### 6.2 性能调优参数

```properties
# Broker 性能调优配置
# conf/broker.conf

# IO 和线程配置
numIOThreads=32                                    # IO 线程数，建议为 CPU 核心数
numHttpServerThreads=16                            # HTTP 服务线程数
numExecutorThreadPoolSize=32                       # 执行器线程池大小

# 内存和缓存配置
managedLedgerCacheSizeMB=8192                     # ML 缓存大小，建议为堆内存的 10-20%
managedLedgerCacheEvictionWatermark=0.9           # 缓存淘汰水位
managedLedgerMaxEntriesPerLedger=50000            # 每个 Ledger 最大条目数
managedLedgerMaxSizePerLedgerMB=1024              # 每个 Ledger 最大大小 1GB

# 批量配置
managedLedgerMinLedgerRolloverTimeMinutes=10      # Ledger 滚动最小时间
managedLedgerMaxLedgerRolloverTimeMinutes=240     # Ledger 滚动最大时间

# 负载均衡配置
loadBalancerEnabled=true
loadBalancerPlacementStrategy=weightedRandomSelection
loadBalancerReportUpdateThresholdPercentage=10
loadBalancerReportUpdateMaxIntervalMinutes=15

# 消息去重配置
brokerDeduplicationEnabled=true
brokerDeduplicationMaxNumberOfProducers=10000
brokerDeduplicationEntriesInterval=1000

# 压缩配置
brokerServiceCompactionThresholdInBytes=104857600  # 100MB
brokerServiceCompactionMonitorIntervalInSeconds=60

# 连接和超时配置
keepAliveIntervalSeconds=30
brokerClientTimeoutMs=30000
backlogQuotaCheckIntervalInSeconds=60

# 认证和授权
authenticationEnabled=true
authorizationEnabled=true
superUserRoles=admin,superuser
authenticationProviders=org.apache.pulsar.broker.authentication.AuthenticationProviderToken
```

```properties
# BookKeeper 性能调优配置  
# conf/bookkeeper.conf

# Journal 配置
journalDirectories=/mnt/fast-ssd/journal          # Journal 独立高速磁盘
journalWriteBufferSizeKB=1024                     # Journal 写缓冲区 1MB
journalSyncData=true                               # 强制 fsync
journalAdaptiveGroupWrites=true                   # 自适应批量写入
journalMaxGroupWaitMSec=2                         # 最大批量等待时间

# Ledger 存储配置
ledgerDirectories=/mnt/ssd1/data,/mnt/ssd2/data,/mnt/ssd3/data
indexDirectories=/mnt/ssd1/index,/mnt/ssd2/index,/mnt/ssd3/index

# DbLedgerStorage 配置（推荐）
ledgerStorageClass=org.apache.bookkeeper.bookie.storage.ldb.DbLedgerStorage
dbStorage_writeCacheMaxSizeMb=2048                # 写缓存 2GB
dbStorage_readAheadCacheMaxSizeMb=1024            # 预读缓存 1GB
dbStorage_rocksDB_blockCacheSize=1073741824       # RocksDB block cache 1GB
dbStorage_rocksDB_writeBufferSizeMB=256           # RocksDB write buffer 256MB
dbStorage_rocksDB_sstSizeInMB=64                  # SST 文件大小 64MB
dbStorage_rocksDB_blockSize=65536                 # 块大小 64KB
dbStorage_rocksDB_bloomFilterBitsPerKey=10        # Bloom filter 配置
dbStorage_rocksDB_numLevels=4                     # LSM 层数

# 垃圾回收配置
gcWaitTime=300000                                 # GC 等待时间 5分钟
gcOverreplicatedLedgerWaitTime=86400000          # 过复制 Ledger GC 等待时间 1天
minorCompactionThreshold=0.2                     # 小压缩阈值
majorCompactionThreshold=0.8                     # 大压缩阈值
minorCompactionInterval=3600                     # 小压缩间隔 1小时
majorCompactionInterval=86400                    # 大压缩间隔 1天

# 网络配置
nettyMaxFrameSizeBytes=52428800                  # 最大帧大小 50MB
serverTcpNoDelay=true                            # TCP_NODELAY
serverTcpKeepAlive=true                          # TCP_KEEPALIVE

# 内存配置
sortedLedgerStorageEnabled=true
skipListSizeLimit=67108864                       # SkipList 大小限制 64MB
skipListChunkSizeEntry=8192                      # SkipList 块大小
```

## 7. 故障排查和解决方案

### 7.1 常见问题诊断

```bash
#!/bin/bash
# Pulsar 健康检查脚本

echo "=== Pulsar 集群健康检查 ==="

# 检查 Broker 状态
echo "1. 检查 Broker 状态"
pulsar-admin brokers healthcheck

# 检查主题列表
echo "2. 检查主题状态"
pulsar-admin topics list public/default

# 检查订阅状态
echo "3. 检查订阅积压"
for topic in $(pulsar-admin topics list public/default); do
    echo "Topic: $topic"
    pulsar-admin topics stats $topic | jq '.subscriptions | to_entries[] | "\(.key): \(.value.msgBacklog) messages"'
done

# 检查 Bookie 状态
echo "4. 检查 BookKeeper 状态"
bookkeeper shell listbookies -rw
bookkeeper shell listbookies -ro

# 检查磁盘使用率
echo "5. 检查磁盘使用率"
df -h | grep -E "(journal|ledger|data)"

# 检查网络连接
echo "6. 检查网络连接"
netstat -tlnp | grep -E "(6650|6651|3181)"

# 检查 JVM 堆内存使用
echo "7. 检查 JVM 内存使用"
jps | grep -E "(PulsarBrokerStarter|BookieServer)" | while read pid name; do
    echo "$name (PID: $pid)"
    jstat -gc $pid
done
```

### 7.2 常见故障处理

```java
/**
 * 常见异常处理指南
 */
public class TroubleshootingGuide {
    
    /**
     * 处理 TopicDoesNotExistException
     */
    public void handleTopicNotExists(PulsarClient client, String topicName) {
        try {
            // 自动创建主题
            Producer<String> producer = client.newProducer(Schema.STRING)
                .topic(topicName)
                .create();
            
            log.info("主题创建成功: {}", topicName);
            producer.close();
            
        } catch (PulsarClientException e) {
            if (e instanceof PulsarClientException.AuthorizationException) {
                log.error("没有权限创建主题: {}", topicName);
                // 联系管理员创建主题
            } else if (e instanceof PulsarClientException.NotAllowedException) {
                log.error("禁止自动创建主题: {}", topicName);
                // 手动创建主题: pulsar-admin topics create persistent://tenant/namespace/topic
            }
        }
    }
    
    /**
     * 处理连接异常
     */
    public PulsarClient createResilientClient(String serviceUrl) {
        return PulsarClient.builder()
            .serviceUrl(serviceUrl)
            .operationTimeout(30, TimeUnit.SECONDS)
            .connectionTimeout(10, TimeUnit.SECONDS)
            .keepAliveInterval(30, TimeUnit.SECONDS)
            .maxLookupRedirects(20)
            .maxLookupRequests(50000)
            .maxNumberOfRejectedRequestPerConnection(50)
            // 启用连接池
            .connectionsPerBroker(1)
            // 启用故障转移
            .enableTransaction(false)  // 根据需要启用
            .build();
    }
    
    /**
     * 处理消费延迟
     */
    public void diagnoseConsumerLag(String topic, String subscription) {
        try {
            // 检查订阅积压
            String command = String.format(
                "pulsar-admin topics stats-internal %s", topic);
            Process process = Runtime.getRuntime().exec(command);
            
            // 分析输出（简化示例）
            log.info("检查消费延迟，执行命令: {}", command);
            
            // 可能的解决方案：
            // 1. 增加消费者实例数量
            // 2. 优化消息处理逻辑
            // 3. 调整接收队列大小
            // 4. 使用批量接收
            
        } catch (IOException e) {
            log.error("执行诊断命令失败", e);
        }
    }
    
    /**
     * 处理 OOM 问题
     */
    public void handleOutOfMemory() {
        // OOM 排查步骤：
        // 1. 检查 JVM 堆内存配置
        // 2. 检查 DirectMemory 使用情况
        // 3. 检查 ManagedLedger Cache 配置
        // 4. 检查是否有内存泄漏
        
        log.info("OOM 排查建议:");
        log.info("1. 调整 JVM 参数: -Xmx32g -XX:MaxDirectMemorySize=32g");
        log.info("2. 降低缓存大小: managedLedgerCacheSizeMB=4096");
        log.info("3. 启用 G1GC: -XX:+UseG1GC -XX:MaxGCPauseMillis=200");
        log.info("4. 监控内存使用: jstat -gc <pid> 5s");
    }
}
```

## 8. 安全最佳实践

### 8.1 认证和授权配置

```bash
# 1. 生成 JWT 密钥
pulsar tokens create-secret-key --output token-secret-key.txt

# 2. 生成管理员 Token
pulsar tokens create --secret-key file:///path/to/token-secret-key.txt \
    --subject admin

# 3. 生成应用 Token
pulsar tokens create --secret-key file:///path/to/token-secret-key.txt \
    --subject app-producer

# 4. Broker 配置
cat >> conf/broker.conf << EOF
# 启用认证
authenticationEnabled=true
authenticationProviders=org.apache.pulsar.broker.authentication.AuthenticationProviderToken
tokenSecretKey=file:///path/to/token-secret-key.txt

# 启用授权
authorizationEnabled=true
superUserRoles=admin

# TLS 配置
tlsEnabled=true
tlsCertificateFilePath=/path/to/broker.cert.pem
tlsKeyFilePath=/path/to/broker.key-pk8.pem
tlsTrustCertsFilePath=/path/to/ca.cert.pem
EOF
```

### 8.2 TLS 加密配置

```java
public class SecurePulsarClient {
    
    public static PulsarClient createSecureClient() throws PulsarClientException {
        return PulsarClient.builder()
            .serviceUrl("pulsar+ssl://pulsar-cluster.example.com:6651")
            .authentication("org.apache.pulsar.client.impl.auth.AuthenticationToken", 
                "token:eyJhbGciOiJIUzI1NiJ9...")
            
            // TLS 配置
            .enableTls(true)
            .tlsTrustCertsFilePath("/path/to/ca.cert.pem")
            .tlsAllowInsecureConnection(false)
            .tlsHostnameVerificationEnable(true)
            
            // 性能配置
            .operationTimeout(30, TimeUnit.SECONDS)
            .connectionsPerBroker(1)
            .build();
    }
    
    public static void demonstrateSecureProduction() throws Exception {
        PulsarClient client = createSecureClient();
        
        // 创建加密生产者
        Producer<String> producer = client.newProducer(Schema.STRING)
            .topic("persistent://secure-tenant/secure-ns/encrypted-topic")
            .cryptoKeyReader(CryptoKeyReader.builder()
                .defaultCryptoKeyReader("file:///path/to/public.key")
                .build())
            .addEncryptionKey("key1")
            .cryptoFailureAction(ProducerCryptoFailureAction.SEND)  // 失败时的动作
            .create();
        
        // 发送加密消息
        producer.send("This message will be encrypted");
        
        producer.close();
        client.close();
    }
}
```

## 9. 性能测试和基准测试

### 9.1 性能测试脚本

```bash
#!/bin/bash
# Pulsar 性能测试脚本

echo "=== Pulsar 性能测试 ==="

# 测试参数
TOPIC="persistent://public/default/perf-test"
PRODUCERS=10
CONSUMERS=10
MESSAGES=1000000
MESSAGE_SIZE=1024

# 清理旧测试数据
echo "清理测试环境..."
pulsar-admin topics delete $TOPIC 2>/dev/null || true
pulsar-admin topics create $TOPIC

# 生产者性能测试
echo "开始生产者性能测试..."
pulsar-perf produce $TOPIC \
    --rate 10000 \
    --num-producers $PRODUCERS \
    --num-messages $MESSAGES \
    --message-size $MESSAGE_SIZE \
    --batch-max-messages 100 \
    --batch-max-publish-delay 10ms

# 消费者性能测试  
echo "开始消费者性能测试..."
pulsar-perf consume $TOPIC \
    --subscription-name perf-test-sub \
    --subscription-type Shared \
    --num-consumers $CONSUMERS \
    --receiver-queue-size 1000

# 端到端延迟测试
echo "端到端延迟测试..."
pulsar-perf latency $TOPIC \
    --rate 1000 \
    --num-messages 100000 \
    --message-size 1024

echo "性能测试完成"
```

### 9.2 性能调优检查清单

```yaml
# 性能调优检查清单
performance_checklist:
  infrastructure:
    - "[ ] 使用 SSD 存储"
    - "[ ] 网络带宽 >= 10Gbps"
    - "[ ] 足够的 CPU 核心数"
    - "[ ] 足够的内存容量"
    
  broker_config:
    - "[ ] numIOThreads = CPU 核心数"
    - "[ ] managedLedgerCacheSizeMB 合理设置"
    - "[ ] 启用 DirectMemory"
    - "[ ] 合理的 GC 配置"
    
  bookkeeper_config:
    - "[ ] Journal 使用独立高速磁盘"
    - "[ ] 启用 DbLedgerStorage"
    - "[ ] 合理的缓存配置"
    - "[ ] 定期压缩配置"
    
  client_config:
    - "[ ] 启用消息批量发送"
    - "[ ] 使用合适的压缩算法"
    - "[ ] 异步发送和接收"
    - "[ ] 连接池配置"
    
  application_design:
    - "[ ] 合理的分区策略"
    - "[ ] 避免热点分区"
    - "[ ] 批量处理消息"
    - "[ ] 合理的重试策略"
```

## 10. 总结

通过以上最佳实践的应用，可以确保 Apache Pulsar 在生产环境中稳定、高效地运行。关键要点包括：

1. **架构设计**：合理的集群规划和硬件配置
2. **客户端优化**：正确使用生产者和消费者 API
3. **Schema 管理**：维护 Schema 的向后兼容性
4. **监控告警**：全面的监控体系和及时的告警
5. **故障处理**：快速诊断和解决常见问题
6. **安全配置**：完善的认证、授权和加密
7. **性能调优**：持续的性能监控和优化

定期回顾和更新这些最佳实践，根据实际业务需求和技术发展进行调整，是保证系统长期稳定运行的关键。
