---
title: "Apache Pulsar 框架使用示例"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['消息队列', '流处理', 'Apache Pulsar', 'Java']
categories: ['pulsar', '消息队列']
description: "Apache Pulsar 框架使用示例的深入技术分析文档"
keywords: ['消息队列', '流处理', 'Apache Pulsar', 'Java']
author: "技术分析师"
weight: 1
---

## 1. 环境准备

### 1.1 快速启动 Standalone 模式

```bash
# 下载并启动 Pulsar Standalone
bin/pulsar standalone

# 验证服务状态
curl http://localhost:8080/admin/v2/clusters
```

### 1.2 Docker 方式启动

```bash
# 启动 Pulsar standalone 容器
docker run -it \
  -p 6650:6650 \
  -p 8080:8080 \
  --mount source=pulsardata,target=/pulsar/data \
  --mount source=pulsarconf,target=/pulsar/conf \
  apachepulsar/pulsar:latest \
  bin/pulsar standalone
```

## 2. 基础使用示例

### 2.1 Java 客户端基础使用

#### 2.1.1 项目依赖配置

```xml
<!-- Maven 依赖配置 -->
<dependencies>
    <dependency>
        <groupId>org.apache.pulsar</groupId>
        <artifactId>pulsar-client</artifactId>
        <version>4.2.0</version>
    </dependency>
    <dependency>
        <groupId>org.apache.pulsar</groupId>
        <artifactId>pulsar-client-admin</artifactId>
        <version>4.2.0</version>
    </dependency>
</dependencies>
```

#### 2.1.2 简单生产者示例

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**

 * 简单消息生产者示例
 * 演示如何创建生产者并发送消息到 Pulsar 主题

 */
public class SimpleProducer {
    private static final Logger log = LoggerFactory.getLogger(SimpleProducer.class);
    
    public static void main(String[] args) throws Exception {
        // 1. 创建 Pulsar 客户端
        // serviceUrl: Pulsar broker 的服务地址
        // operationTimeout: 操作超时时间
        // connectionsPerBroker: 每个 broker 的连接数
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .operationTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
                .connectionsPerBroker(5)
                .build();
        
        try {
            // 2. 创建生产者
            // topic: 目标主题名称
            // producerName: 生产者名称（可选，系统会自动生成）
            // sendTimeout: 发送超时时间
            // maxPendingMessages: 最大待发送消息数
            Producer<String> producer = client.newProducer(Schema.STRING)
                    .topic("simple-topic")
                    .producerName("simple-producer")
                    .sendTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
                    .maxPendingMessages(100)
                    .create();
            
            // 3. 发送消息
            for (int i = 0; i < 10; i++) {
                String message = "Hello Pulsar! Message " + i;
                
                // 同步发送消息
                // send() 方法会阻塞直到消息被 broker 确认
                MessageId msgId = producer.send(message);
                log.info("发送消息成功: {} -> MessageId: {}", message, msgId);
            }
            
            // 4. 关闭生产者
            producer.close();
        } finally {
            // 5. 关闭客户端
            client.close();
        }
    }
}
```

#### 2.1.3 异步生产者示例

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**

 * 异步消息生产者示例
 * 演示如何使用异步 API 提高消息发送性能

 */
public class AsyncProducer {
    private static final Logger log = LoggerFactory.getLogger(AsyncProducer.class);
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
        
        try {
            // 创建生产者配置
            Producer<String> producer = client.newProducer(Schema.STRING)
                    .topic("async-topic")
                    .producerName("async-producer")
                    // 启用批量发送以提高性能
                    .batchingMaxMessages(100)
                    .batchingMaxPublishDelay(100, TimeUnit.MILLISECONDS)
                    // 启用压缩减少网络传输
                    .compressionType(CompressionType.LZ4)
                    .create();
            
            // 异步发送多条消息
            for (int i = 0; i < 100; i++) {
                String message = "Async message " + i;
                
                // 异步发送消息
                // sendAsync() 立即返回 CompletableFuture，不阻塞当前线程
                CompletableFuture<MessageId> future = producer.sendAsync(message);
                
                // 设置回调处理发送结果
                final int messageIndex = i;
                future.thenAccept(messageId -> {
                    log.info("消息 {} 发送成功，MessageId: {}", messageIndex, messageId);
                }).exceptionally(ex -> {
                    log.error("消息 {} 发送失败: {}", messageIndex, ex.getMessage());
                    return null;
                });
            }
            
            // 等待所有消息发送完成
            producer.flush();
            Thread.sleep(2000); // 等待回调完成
            
            producer.close();
        } finally {
            client.close();
        }
    }
}
```

#### 2.1.4 简单消费者示例

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**

 * 简单消息消费者示例
 * 演示如何创建消费者并从 Pulsar 主题接收消息

 */
public class SimpleConsumer {
    private static final Logger log = LoggerFactory.getLogger(SimpleConsumer.class);
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
        
        try {
            // 创建消费者
            // topic: 订阅的主题
            // subscriptionName: 订阅名称，用于跟踪消费进度
            // subscriptionType: 订阅类型（Exclusive, Shared, Failover, Key_Shared）
            Consumer<String> consumer = client.newConsumer(Schema.STRING)
                    .topic("simple-topic")
                    .subscriptionName("simple-subscription")
                    .subscriptionType(SubscriptionType.Exclusive)
                    // 设置接收队列大小，预拉取消息数量
                    .receiverQueueSize(100)
                    // 设置确认超时时间，超时未确认的消息会重新投递
                    .ackTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
                    .subscribe();
            
            // 接收并处理消息
            while (true) {
                try {
                    // 阻塞等待接收消息，超时时间 5 秒
                    Message<String> message = consumer.receive(5, java.util.concurrent.TimeUnit.SECONDS);
                    
                    if (message != null) {
                        // 处理消息内容
                        String content = message.getValue();
                        MessageId messageId = message.getMessageId();
                        
                        log.info("接收到消息: {} -> MessageId: {}", content, messageId);
                        log.info("消息属性: {}", message.getProperties());
                        log.info("发布时间: {}", message.getPublishTime());
                        
                        // 确认消息处理完成
                        // 确认后该消息不会再次投递给这个订阅
                        consumer.acknowledge(message);
                        
                        // 检查是否需要退出
                        if (content.contains("exit")) {
                            log.info("收到退出指令，停止消费");
                            break;
                        }
                    } else {
                        log.info("接收消息超时，继续等待...");
                    }
                } catch (PulsarClientException e) {
                    log.error("接收消息失败: {}", e.getMessage());
                }
            }
            
            consumer.close();
        } finally {
            client.close();
        }
    }
}
```

#### 2.1.5 异步消费者示例

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**

 * 异步消息消费者示例
 * 演示如何使用消息监听器异步处理消息

 */
public class AsyncConsumer {
    private static final Logger log = LoggerFactory.getLogger(AsyncConsumer.class);
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
        
        CountDownLatch latch = new CountDownLatch(1);
        
        try {
            // 创建消费者并设置消息监听器
            Consumer<String> consumer = client.newConsumer(Schema.STRING)
                    .topic("async-topic")
                    .subscriptionName("async-subscription")
                    .subscriptionType(SubscriptionType.Shared)
                    .receiverQueueSize(100)
                    // 设置消息监听器，消息到达时自动调用
                    .messageListener(new MessageListener<String>() {
                        private int messageCount = 0;
                        
                        @Override
                        public void received(Consumer<String> consumer, Message<String> message) {
                            try {
                                // 处理消息
                                String content = message.getValue();
                                messageCount++;
                                
                                log.info("异步接收消息 #{}: {} -> MessageId: {}",
                                    messageCount, content, message.getMessageId());
                                
                                // 模拟消息处理时间
                                Thread.sleep(100);
                                
                                // 确认消息
                                consumer.acknowledge(message);
                                
                                // 处理完 20 条消息后退出
                                if (messageCount >= 20) {
                                    latch.countDown();
                                }
                                
                            } catch (Exception e) {
                                log.error("处理消息失败: {}", e.getMessage(), e);
                                // 拒绝消息，消息会重新投递
                                consumer.negativeAcknowledge(message);
                            }
                        }
                    })
                    .subscribe();
            
            log.info("开始监听消息，等待消息到达...");
            
            // 等待处理完成
            latch.await(30, TimeUnit.SECONDS);
            
            consumer.close();
        } finally {
            client.close();
        }
    }
}
```

### 2.2 批量操作示例

#### 2.2.1 批量消息生产者

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.TimeUnit;

/**

 * 批量消息生产者示例
 * 演示如何配置和使用批量发送功能提高性能

 */
public class BatchProducer {
    private static final Logger log = LoggerFactory.getLogger(BatchProducer.class);
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
        
        try {
            Producer<String> producer = client.newProducer(Schema.STRING)
                    .topic("batch-topic")
                    .producerName("batch-producer")
                    // 启用批量发送
                    .enableBatching(true)
                    // 批量消息最大条数
                    .batchingMaxMessages(50)
                    // 批量发送最大延迟时间
                    .batchingMaxPublishDelay(100, TimeUnit.MILLISECONDS)
                    // 批量消息最大字节数
                    .batchingMaxBytes(128 * 1024)
                    // 批量分区策略
                    .batcherBuilder(BatcherBuilder.DEFAULT)
                    // 压缩类型
                    .compressionType(CompressionType.LZ4)
                    .create();
            
            long startTime = System.currentTimeMillis();
            
            // 发送大量消息测试批量性能
            for (int i = 0; i < 1000; i++) {
                String message = String.format("Batch message %d - timestamp: %d",
                    i, System.currentTimeMillis());
                
                // 异步发送以提高性能
                producer.sendAsync(message)
                    .thenAccept(messageId -> {
                        if (i % 100 == 0) {
                            log.info("发送消息 {} 成功: {}", i, messageId);
                        }
                    })
                    .exceptionally(ex -> {
                        log.error("发送消息 {} 失败: {}", i, ex.getMessage());
                        return null;
                    });
                
                // 每发送 100 条消息强制刷新一次
                if (i % 100 == 0 && i > 0) {
                    producer.flush();
                    log.info("已发送 {} 条消息，强制刷新批量缓冲区", i);
                }
            }
            
            // 确保所有消息都发送完成
            producer.flush();
            
            long endTime = System.currentTimeMillis();
            log.info("批量发送 1000 条消息完成，耗时: {} ms", endTime - startTime);
            
            producer.close();
        } finally {
            client.close();
        }
    }
}
```

#### 2.2.2 批量消息消费者

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.TimeUnit;

/**

 * 批量消息消费者示例
 * 演示如何使用批量接收 API 提高消费性能

 */
public class BatchConsumer {
    private static final Logger log = LoggerFactory.getLogger(BatchConsumer.class);
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
        
        try {
            Consumer<String> consumer = client.newConsumer(Schema.STRING)
                    .topic("batch-topic")
                    .subscriptionName("batch-subscription")
                    .subscriptionType(SubscriptionType.Shared)
                    // 设置较大的接收队列以提高吞吐量
                    .receiverQueueSize(1000)
                    .subscribe();
            
            int totalMessages = 0;
            long startTime = System.currentTimeMillis();
            
            while (totalMessages < 1000) {
                try {
                    // 批量接收消息
                    // maxNumMessages: 批量接收的最大消息数量
                    // maxSizeBytes: 批量接收的最大字节数
                    // timeout: 批量接收的超时时间
                    Messages<String> messages = consumer.batchReceive();
                    
                    if (messages.size() > 0) {
                        log.info("批量接收到 {} 条消息", messages.size());
                        
                        // 处理批量消息
                        for (Message<String> message : messages) {
                            String content = message.getValue();
                            MessageId messageId = message.getMessageId();
                            
                            // 这里可以进行批量业务处理
                            // 例如：批量插入数据库、批量调用外部服务等
                            if (totalMessages % 100 == 0) {
                                log.info("处理消息: {} -> {}", content, messageId);
                            }
                            
                            totalMessages++;
                        }
                        
                        // 批量确认所有消息
                        // 注意：只需要确认最后一条消息即可
                        Message<String> lastMessage = null;
                        for (Message<String> message : messages) {
                            lastMessage = message;
                        }
                        if (lastMessage != null) {
                            consumer.acknowledge(lastMessage);
                        }
                        
                        log.info("批量确认完成，已处理消息总数: {}", totalMessages);
                    } else {
                        log.info("未接收到消息，继续等待...");
                    }
                    
                } catch (PulsarClientException e) {
                    log.error("批量接收消息失败: {}", e.getMessage());
                }
            }
            
            long endTime = System.currentTimeMillis();
            log.info("批量消费 {} 条消息完成，耗时: {} ms", totalMessages, endTime - startTime);
            
            consumer.close();
        } finally {
            client.close();
        }
    }
}
```

### 2.3 高级特性示例

#### 2.3.1 消息路由和分区

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**

 * 消息路由和分区示例
 * 演示如何使用自定义消息路由策略控制消息分发

 */
public class PartitionedProducer {
    private static final Logger log = LoggerFactory.getLogger(PartitionedProducer.class);
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
        
        try {
            // 创建分区主题的生产者
            Producer<String> producer = client.newProducer(Schema.STRING)
                    .topic("partitioned-topic")
                    .producerName("partitioned-producer")
                    // 自定义消息路由器
                    .messageRouter(new MessageRouter() {
                        @Override
                        public int choosePartition(Message<?> msg, TopicMetadata metadata) {
                            // 根据消息键的哈希值选择分区
                            String key = msg.getKey();
                            if (key != null) {
                                return Math.abs(key.hashCode()) % metadata.numPartitions();
                            }
                            // 如果没有键，使用轮询方式
                            return -1;
                        }
                    })
                    .create();
            
            // 发送带有不同键的消息到不同分区
            String[] keys = {"user-1", "user-2", "user-3", "user-4", "user-5"};
            
            for (int i = 0; i < 50; i++) {
                String key = keys[i % keys.length];
                String message = String.format("Message %d for %s", i, key);
                
                // 使用类型化消息构建器设置消息属性
                MessageId messageId = producer.newMessage()
                        .key(key)  // 设置消息键用于分区路由
                        .value(message)  // 设置消息内容
                        .property("user-id", key)  // 设置自定义属性
                        .property("message-type", "data")
                        .eventTime(System.currentTimeMillis())  // 设置事件时间
                        .send();
                
                log.info("发送消息到分区: key={}, message={}, messageId={}",
                    key, message, messageId);
            }
            
            producer.close();
        } finally {
            client.close();
        }
    }
}
```

#### 2.3.2 消息过滤和选择器

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**

 * 消息过滤和选择器示例
 * 演示如何使用消息选择器过滤消息

 */
public class MessageFilterConsumer {
    private static final Logger log = LoggerFactory.getLogger(MessageFilterConsumer.class);
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
        
        try {
            // 创建使用消息选择器的消费者
            Consumer<String> consumer = client.newConsumer(Schema.STRING)
                    .topic("filtered-topic")
                    .subscriptionName("filtered-subscription")
                    .subscriptionType(SubscriptionType.Shared)
                    // 使用 SQL 表达式过滤消息
                    // 只接收 user-type 属性为 'premium' 的消息
                    .subscriptionProperties(java.util.Map.of(
                        "subscriptionType", "filtered"
                    ))
                    .subscribe();
            
            // 创建生产者发送测试消息
            Producer<String> producer = client.newProducer(Schema.STRING)
                    .topic("filtered-topic")
                    .create();
            
            // 发送不同类型的消息
            String[] userTypes = {"premium", "standard", "premium", "basic", "premium"};
            
            for (int i = 0; i < 10; i++) {
                String userType = userTypes[i % userTypes.length];
                String message = String.format("Message %d for %s user", i, userType);
                
                producer.newMessage()
                        .value(message)
                        .property("user-type", userType)
                        .property("priority", userType.equals("premium") ? "high" : "normal")
                        .sendAsync()
                        .thenAccept(messageId -> {
                            log.info("发送消息: type={}, message={}, messageId={}",
                                userType, message, messageId);
                        });
            }
            
            // 等待消息发送完成
            producer.flush();
            
            // 接收过滤后的消息
            int receivedCount = 0;
            while (receivedCount < 5) {  // 预期接收 5 条 premium 消息
                try {
                    Message<String> message = consumer.receive(10, java.util.concurrent.TimeUnit.SECONDS);
                    
                    if (message != null) {
                        String content = message.getValue();
                        String userType = message.getProperty("user-type");
                        String priority = message.getProperty("priority");
                        
                        log.info("接收到过滤消息: content={}, user-type={}, priority={}",
                            content, userType, priority);
                        
                        consumer.acknowledge(message);
                        receivedCount++;
                    }
                } catch (PulsarClientException e) {
                    log.error("接收消息失败: {}", e.getMessage());
                    break;
                }
            }
            
            producer.close();
            consumer.close();
        } finally {
            client.close();
        }
    }
}
```

#### 2.3.3 死信队列和重试机制

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.TimeUnit;

/**

 * 死信队列和重试机制示例
 * 演示如何配置死信队列处理消息处理失败的情况

 */
public class DeadLetterQueueExample {
    private static final Logger log = LoggerFactory.getLogger(DeadLetterQueueExample.class);
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
        
        try {
            // 创建配置了死信队列的消费者
            Consumer<String> consumer = client.newConsumer(Schema.STRING)
                    .topic("retry-topic")
                    .subscriptionName("retry-subscription")
                    .subscriptionType(SubscriptionType.Shared)
                    // 配置死信队列策略
                    .deadLetterPolicy(DeadLetterPolicy.builder()
                            .maxRedeliverCount(3)  // 最大重试次数
                            .deadLetterTopic("retry-topic-dlq")  // 死信队列主题
                            .retryLetterTopic("retry-topic-retry")  // 重试队列主题
                            .build())
                    // 设置确认超时时间
                    .ackTimeout(10, TimeUnit.SECONDS)
                    // 设置消息监听器
                    .messageListener((consumer1, message) -> {
                        try {
                            String content = message.getValue();
                            int redeliveryCount = message.getRedeliveryCount();
                            
                            log.info("处理消息: content={}, redeliveryCount={}, messageId={}",
                                content, redeliveryCount, message.getMessageId());
                            
                            // 模拟处理失败的情况
                            if (content.contains("error")) {
                                log.error("消息处理失败，模拟异常: {}", content);
                                throw new RuntimeException("模拟处理异常");
                            }
                            
                            // 正常处理完成
                            consumer1.acknowledge(message);
                            log.info("消息处理成功并确认: {}", content);
                            
                        } catch (Exception e) {
                            log.error("消息处理异常: {}", e.getMessage());
                            // 拒绝消息，触发重试机制
                            consumer1.negativeAcknowledge(message);
                        }
                    })
                    .subscribe();
            
            // 创建生产者发送测试消息
            Producer<String> producer = client.newProducer(Schema.STRING)
                    .topic("retry-topic")
                    .create();
            
            // 发送正常消息和错误消息
            producer.send("正常消息 1");
            producer.send("包含 error 的消息");  // 这条消息会触发重试
            producer.send("正常消息 2");
            producer.send("另一条包含 error 的消息");  // 这条消息也会触发重试
            producer.send("正常消息 3");
            
            log.info("测试消息发送完成，等待处理结果...");
            
            // 等待消息处理完成
            Thread.sleep(30000);
            
            // 创建死信队列消费者查看最终失败的消息
            Consumer<String> dlqConsumer = client.newConsumer(Schema.STRING)
                    .topic("retry-topic-dlq")
                    .subscriptionName("dlq-subscription")
                    .subscriptionType(SubscriptionType.Shared)
                    .subscribe();
            
            log.info("检查死信队列中的消息...");
            
            while (true) {
                try {
                    Message<String> dlqMessage = dlqConsumer.receive(5, TimeUnit.SECONDS);
                    if (dlqMessage != null) {
                        log.info("死信队列消息: content={}, originalMessageId={}, properties={}",
                            dlqMessage.getValue(),
                            dlqMessage.getProperty("REAL_TOPIC"),
                            dlqMessage.getProperties());
                        dlqConsumer.acknowledge(dlqMessage);
                    } else {
                        log.info("没有更多死信队列消息");
                        break;
                    }
                } catch (PulsarClientException e) {
                    log.info("死信队列消息接收完成");
                    break;
                }
            }
            
            producer.close();
            consumer.close();
            dlqConsumer.close();
        } finally {
            client.close();
        }
    }
}
```

### 2.4 Schema 和序列化示例

#### 2.4.1 Avro Schema 示例

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.apache.pulsar.client.api.schema.Field;
import org.apache.pulsar.client.api.schema.RecordSchemaBuilder;
import org.apache.pulsar.client.api.schema.SchemaBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**

 * Avro Schema 示例
 * 演示如何使用结构化数据模式进行序列化和反序列化

 */
public class AvroSchemaExample {
    private static final Logger log = LoggerFactory.getLogger(AvroSchemaExample.class);
    
    // 定义用户数据结构
    public static class User {
        private String name;
        private int age;
        private String email;
        private long timestamp;
        
        // 构造函数
        public User() {}
        
        public User(String name, int age, String email, long timestamp) {
            this.name = name;
            this.age = age;
            this.email = email;
            this.timestamp = timestamp;
        }
        
        // Getter 和 Setter 方法
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        
        public int getAge() { return age; }
        public void setAge(int age) { this.age = age; }
        
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        
        public long getTimestamp() { return timestamp; }
        public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
        
        @Override
        public String toString() {
            return String.format("User{name='%s', age=%d, email='%s', timestamp=%d}",
                name, age, email, timestamp);
        }
    }
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
        
        try {
            // 方式1：使用自动生成的 Avro Schema
            Schema<User> autoSchema = Schema.AVRO(User.class);
            
            // 方式2：手动构建 Avro Schema
            RecordSchemaBuilder recordSchemaBuilder = SchemaBuilder.record("User");
            recordSchemaBuilder.field("name").type(SchemaType.STRING);
            recordSchemaBuilder.field("age").type(SchemaType.INT32);
            recordSchemaBuilder.field("email").type(SchemaType.STRING);
            recordSchemaBuilder.field("timestamp").type(SchemaType.INT64);
            Schema<User> manualSchema = Schema.generic(recordSchemaBuilder.build(SchemaType.AVRO));
            
            // 使用自动生成的 Schema 创建生产者
            Producer<User> producer = client.newProducer(autoSchema)
                    .topic("avro-topic")
                    .producerName("avro-producer")
                    .create();
            
            // 创建消费者
            Consumer<User> consumer = client.newConsumer(autoSchema)
                    .topic("avro-topic")
                    .subscriptionName("avro-subscription")
                    .subscriptionType(SubscriptionType.Exclusive)
                    .subscribe();
            
            // 发送结构化数据
            for (int i = 0; i < 5; i++) {
                User user = new User(
                    "User" + i,
                    25 + i,
                    "user" + i + "@example.com",
                    System.currentTimeMillis()
                );
                
                MessageId messageId = producer.newMessage()
                        .value(user)
                        .property("user-id", String.valueOf(i))
                        .send();
                
                log.info("发送用户数据: {} -> MessageId: {}", user, messageId);
            }
            
            // 接收结构化数据
            for (int i = 0; i < 5; i++) {
                Message<User> message = consumer.receive(10, java.util.concurrent.TimeUnit.SECONDS);
                
                if (message != null) {
                    User receivedUser = message.getValue();
                    String userId = message.getProperty("user-id");
                    
                    log.info("接收到用户数据: {} -> 属性: user-id={}, MessageId: {}",
                        receivedUser, userId, message.getMessageId());
                    
                    consumer.acknowledge(message);
                }
            }
            
            producer.close();
            consumer.close();
        } finally {
            client.close();
        }
    }
}
```

### 2.5 事务支持示例

#### 2.5.1 事务消息示例

```java
package org.apache.pulsar.examples;

import org.apache.pulsar.client.api.*;
import org.apache.pulsar.client.api.transaction.Transaction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.TimeUnit;

/**

 * 事务消息示例
 * 演示如何使用事务保证消息的原子性操作

 */
public class TransactionExample {
    private static final Logger log = LoggerFactory.getLogger(TransactionExample.class);
    
    public static void main(String[] args) throws Exception {
        // 启用事务的客户端配置
        PulsarClient client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .enableTransaction(true)  // 启用事务支持
                .build();
        
        try {
            // 创建生产者
            Producer<String> producer = client.newProducer(Schema.STRING)
                    .topic("transaction-topic")
                    .producerName("transaction-producer")
                    .sendTimeout(10, TimeUnit.SECONDS)
                    .create();
            
            // 创建消费者
            Consumer<String> consumer = client.newConsumer(Schema.STRING)
                    .topic("transaction-topic")
                    .subscriptionName("transaction-subscription")
                    .subscriptionType(SubscriptionType.Exclusive)
                    .subscribe();
            
            // 示例1：成功的事务
            log.info("=== 执行成功的事务 ===");
            executeSuccessfulTransaction(client, producer);
            
            // 等待消息发送完成
            Thread.sleep(1000);
            
            // 示例2：失败的事务（回滚）
            log.info("=== 执行失败的事务（回滚） ===");
            executeFailedTransaction(client, producer);
            
            // 等待事务处理完成
            Thread.sleep(1000);
            
            // 消费消息验证事务效果
            log.info("=== 消费消息验证事务效果 ===");
            consumeMessages(consumer);
            
            producer.close();
            consumer.close();
        } finally {
            client.close();
        }
    }
    
    /**

     * 执行成功的事务
     */
    private static void executeSuccessfulTransaction(PulsarClient client, Producer<String> producer)
            throws Exception {
        // 开始事务
        Transaction transaction = client.newTransaction()
                .withTransactionTimeout(30, TimeUnit.SECONDS)
                .build()
                .get();
        
        try {
            log.info("开始事务: {}", transaction.getTxnID());
            
            // 在事务中发送多条消息
            for (int i = 0; i < 3; i++) {
                String message = "Transaction message " + i;
                
                MessageId messageId = producer.newMessage(transaction)
                        .value(message)
                        .property("transaction-id", transaction.getTxnID().toString())
                        .property("batch", "success")
                        .send();
                
                log.info("在事务中发送消息: {} -> MessageId: {}", message, messageId);
            }
            
            // 提交事务
            transaction.commit().get();
            log.info("事务提交成功: {}", transaction.getTxnID());
            
        } catch (Exception e) {
            log.error("事务执行失败，回滚: {}", e.getMessage());
            transaction.abort().get();
        }
    }
    
    /**
     * 执行失败的事务（模拟回滚）
     */
    private static void executeFailedTransaction(PulsarClient client, Producer<String> producer)
            throws Exception {
        // 开始事务
        Transaction transaction = client.newTransaction()
                .withTransactionTimeout(30, TimeUnit.SECONDS)
                .build()
                .get();
        
        try {
            log.info("开始事务: {}", transaction.getTxnID());
            
            // 在事务中发送消息
            for (int i = 0; i < 3; i++) {
                String message = "Failed transaction message " + i;
                
                MessageId messageId = producer.newMessage(transaction)
                        .value(message)
                        .property("transaction-id", transaction.getTxnID().toString())
                        .property("batch", "failed")
                        .send();
                
                log.info("在事务中发送消息: {} -> MessageId: {}", message, messageId);
                
                // 模拟在第2条消息后发生异常
                if (i == 1) {
                    throw new RuntimeException("模拟事务处理异常");
                }
            }
            
            // 提交事务
            transaction.commit().get();
            log.info("事务提交成功: {}", transaction.getTxnID());
            
        } catch (Exception e) {
            log.error("事务执行失败，回滚: {} -> {}", transaction.getTxnID(), e.getMessage());
            // 回滚事务
            transaction.abort().get();
            log.info("事务回滚完成: {}", transaction.getTxnID());
        }
    }
    
    /**
     * 消费消息验证事务效果
     */
    private static void consumeMessages(Consumer<String> consumer) throws Exception {
        int messageCount = 0;
        
        while (messageCount < 10) {  // 最多等待 10 条消息
            try {
                Message<String> message = consumer.receive(5, TimeUnit.SECONDS);
                
                if (message != null) {
                    String content = message.getValue();
                    String transactionId = message.getProperty("transaction-id");
                    String batch = message.getProperty("batch");
                    
                    log.info("接收到消息: content={}, transaction-id={}, batch={}, messageId={}",
                        content, transactionId, batch, message.getMessageId());
                    
                    consumer.acknowledge(message);
                    messageCount++;
                } else {
                    log.info("没有更多消息，退出消费");
                    break;
                }
                
            } catch (PulsarClientException e) {
                log.info("消费消息完成");
                break;
            }
        }
        
        log.info("总共接收到 {} 条消息（只有成功事务的消息会被消费）", messageCount);
    }

}
```

这些示例展示了 Apache Pulsar 的主要使用场景和特性。从基础的生产和消费，到高级的事务、Schema、分区路由等功能，为用户提供了全面的使用指南。
