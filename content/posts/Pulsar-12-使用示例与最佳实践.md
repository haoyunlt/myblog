# Pulsar-12-使用示例与最佳实践

## 概述

本文档汇总了 Apache Pulsar 在实际应用中的典型使用场景、完整代码示例、性能调优策略和最佳实践建议。内容覆盖从基础入门到生产环境部署的全流程。

---

## 一、基础使用示例

### 1.1 简单的生产者与消费者

#### 生产者示例

```java
import org.apache.pulsar.client.api.*;

public class SimpleProducerExample {
    public static void main(String[] args) throws Exception {
        // 创建 Pulsar Client
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // 创建 Producer
        Producer<String> producer = client.newProducer(Schema.STRING)
            .topic("persistent://public/default/my-topic")
            .create();
        
        // 发送消息
        for (int i = 0; i < 10; i++) {
            String message = "Hello Pulsar " + i;
            MessageId msgId = producer.send(message);
            System.out.println("Published message: " + message + ", ID: " + msgId);
        }
        
        // 关闭资源
        producer.close();
        client.close();
    }
}
```

**说明**：
- `serviceUrl`：Broker 地址，支持 `pulsar://`（明文）和 `pulsar+ssl://`（TLS）
- `topic`：完整主题名，格式为 `{persistent|non-persistent}://{tenant}/{namespace}/{topic}`
- `Schema.STRING`：自动序列化/反序列化字符串
- `send()`：同步发送，阻塞等待确认

#### 消费者示例

```java
import org.apache.pulsar.client.api.*;

public class SimpleConsumerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // 创建 Consumer
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/my-topic")
            .subscriptionName("my-subscription")
            .subscriptionType(SubscriptionType.Shared)
            .subscribe();
        
        // 接收消息
        while (true) {
            Message<String> msg = consumer.receive();
            try {
                System.out.println("Received: " + msg.getValue());
                consumer.acknowledge(msg);
            } catch (Exception e) {
                consumer.negativeAcknowledge(msg);
            }
        }
    }
}
```

**说明**：
- `subscriptionName`：订阅名，标识消费者组
- `subscriptionType`：
  - `Exclusive`：独占，单消费者
  - `Shared`：共享，多消费者负载均衡
  - `Failover`：故障转移，主备模式
  - `Key_Shared`：按 Key 分区共享
- `acknowledge()`：确认消息处理成功
- `negativeAcknowledge()`：标记失败，立即重试

---

### 1.2 异步发送与接收

#### 异步生产者

```java
public class AsyncProducerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        Producer<String> producer = client.newProducer(Schema.STRING)
            .topic("persistent://public/default/async-topic")
            .sendTimeout(10, TimeUnit.SECONDS)
            .create();
        
        // 异步发送消息
        List<CompletableFuture<MessageId>> futures = new ArrayList<>();
        
        for (int i = 0; i < 100; i++) {
            String message = "Async message " + i;
            
            CompletableFuture<MessageId> future = producer.sendAsync(message)
                .thenApply(msgId -> {
                    System.out.println("Published: " + message + ", ID: " + msgId);
                    return msgId;
                })
                .exceptionally(ex -> {
                    System.err.println("Failed to publish: " + message);
                    return null;
                });
            
            futures.add(future);
        }
        
        // 等待所有消息发送完成
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
        
        // 关闭资源
        producer.close();
        client.close();
    }
}
```

**优势**：
- 非阻塞，提升吞吐量 10-100 倍
- 支持批量并发发送
- 适合高性能场景

#### 异步消费者

```java
public class AsyncConsumerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/async-topic")
            .subscriptionName("async-sub")
            .subscribe();
        
        // 异步接收和处理消息
        while (true) {
            consumer.receiveAsync()
                .thenCompose(msg -> {
                    // 异步处理消息
                    return processMessageAsync(msg)
                        .thenCompose(result -> consumer.acknowledgeAsync(msg));
                })
                .exceptionally(ex -> {
                    System.err.println("Failed to process message: " + ex.getMessage());
                    return null;
                });
        }
    }
    
    private static CompletableFuture<Void> processMessageAsync(Message<String> msg) {
        return CompletableFuture.supplyAsync(() -> {
            System.out.println("Processing: " + msg.getValue());
            // 业务处理逻辑
            return null;
        });
    }
}
```

---

### 1.3 使用 MessageListener

```java
public class MessageListenerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/listener-topic")
            .subscriptionName("listener-sub")
            .messageListener((cons, msg) -> {
                try {
                    System.out.println("Received: " + msg.getValue());
                    // 业务处理
                    cons.acknowledge(msg);
                } catch (Exception e) {
                    cons.negativeAcknowledge(msg);
                }
            })
            .subscribe();
        
        // 主线程阻塞，监听器在后台线程处理消息
        Thread.sleep(Long.MAX_VALUE);
    }
}
```

**说明**：
- `messageListener`：注册消息监听器，自动接收消息
- 监听器在独立线程池执行，不阻塞主线程
- 适合事件驱动架构

---

## 二、进阶使用场景

### 2.1 分区主题

#### 分区生产者

```java
public class PartitionedProducerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // 创建分区主题（需要先通过 Admin API 创建）
        String partitionedTopic = "persistent://public/default/partitioned-topic";
        
        Producer<String> producer = client.newProducer(Schema.STRING)
            .topic(partitionedTopic)
            // 分区路由策略
            .messageRoutingMode(MessageRoutingMode.CustomPartition)
            .messageRouter(new MessageRouter() {
                @Override
                public int choosePartition(Message<?> msg, TopicMetadata metadata) {
                    // 自定义路由逻辑：根据 Key 哈希
                    String key = msg.getKey();
                    if (key != null) {
                        return Math.abs(key.hashCode()) % metadata.numPartitions();
                    }
                    // 无 Key，轮询
                    return metadata.numPartitions() - 1;
                }
            })
            .create();
        
        // 发送消息到不同分区
        for (int i = 0; i < 100; i++) {
            String key = "user-" + (i % 10);  // 10 个用户
            producer.newMessage()
                .key(key)
                .value("Message " + i + " from " + key)
                .send();
        }
        
        producer.close();
        client.close();
    }
}
```

**分区策略**：
- `RoundRobinPartition`：轮询，均匀分布
- `SinglePartition`：单分区，保证顺序
- `CustomPartition`：自定义路由

#### 分区消费者

```java
public class PartitionedConsumerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        String partitionedTopic = "persistent://public/default/partitioned-topic";
        
        // 方式 1：自动订阅所有分区（推荐）
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic(partitionedTopic)
            .subscriptionName("partitioned-sub")
            .subscriptionType(SubscriptionType.Shared)
            .subscribe();
        
        // 方式 2：订阅指定分区
        Consumer<String> partition0Consumer = client.newConsumer(Schema.STRING)
            .topic(partitionedTopic + "-partition-0")
            .subscriptionName("partition-0-sub")
            .subscribe();
        
        // 接收消息
        while (true) {
            Message<String> msg = consumer.receive();
            System.out.println("Received: " + msg.getValue() + " from partition-" + 
                msg.getTopicName());
            consumer.acknowledge(msg);
        }
    }
}
```

---

### 2.2 多主题订阅

```java
public class MultiTopicsConsumerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // 订阅多个主题
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topics(Arrays.asList(
                "persistent://public/default/topic-1",
                "persistent://public/default/topic-2",
                "persistent://public/default/topic-3"
            ))
            .subscriptionName("multi-topic-sub")
            .subscribe();
        
        // 或使用正则表达式订阅
        Consumer<String> regexConsumer = client.newConsumer(Schema.STRING)
            .topicsPattern("persistent://public/default/topic-.*")
            .subscriptionName("regex-sub")
            .subscribe();
        
        while (true) {
            Message<String> msg = consumer.receive();
            System.out.println("Received from " + msg.getTopicName() + ": " + msg.getValue());
            consumer.acknowledge(msg);
        }
    }
}
```

**使用场景**：
- 聚合多个数据源
- 简化消费者管理
- 动态订阅新主题（正则模式）

---

### 2.3 Schema 与序列化

#### 使用 Avro Schema

```java
import org.apache.pulsar.client.api.Schema;
import org.apache.pulsar.client.api.schema.SchemaDefinition;

// 定义数据类
@Data
@AllArgsConstructor
@NoArgsConstructor
public class User {
    private String name;
    private int age;
    private String email;
}

public class AvroSchemaExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // 定义 Avro Schema
        Schema<User> userSchema = Schema.AVRO(User.class);
        
        // 生产者
        Producer<User> producer = client.newProducer(userSchema)
            .topic("persistent://public/default/user-topic")
            .create();
        
        // 发送对象
        User user = new User("Alice", 30, "alice@example.com");
        producer.send(user);
        
        // 消费者
        Consumer<User> consumer = client.newConsumer(userSchema)
            .topic("persistent://public/default/user-topic")
            .subscriptionName("user-sub")
            .subscribe();
        
        Message<User> msg = consumer.receive();
        User receivedUser = msg.getValue();
        System.out.println("Received user: " + receivedUser);
        
        consumer.acknowledge(msg);
        producer.close();
        consumer.close();
        client.close();
    }
}
```

#### 使用 JSON Schema

```java
public class JsonSchemaExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // JSON Schema
        Schema<User> jsonSchema = Schema.JSON(User.class);
        
        Producer<User> producer = client.newProducer(jsonSchema)
            .topic("persistent://public/default/json-topic")
            .create();
        
        producer.send(new User("Bob", 25, "bob@example.com"));
        
        producer.close();
        client.close();
    }
}
```

#### 自定义 Schema

```java
public class CustomSchemaExample {
    
    // 自定义 Schema 实现
    public static class CustomUserSchema implements Schema<User> {
        
        @Override
        public byte[] encode(User user) {
            // 自定义序列化逻辑
            String json = String.format("{\"name\":\"%s\",\"age\":%d,\"email\":\"%s\"}",
                user.getName(), user.getAge(), user.getEmail());
            return json.getBytes(StandardCharsets.UTF_8);
        }
        
        @Override
        public User decode(byte[] bytes) {
            // 自定义反序列化逻辑
            String json = new String(bytes, StandardCharsets.UTF_8);
            // 解析 JSON（简化示例，实际应使用 JSON 库）
            return parseJson(json);
        }
        
        @Override
        public SchemaInfo getSchemaInfo() {
            return SchemaInfo.builder()
                .name("CustomUserSchema")
                .type(SchemaType.BYTES)
                .schema(new byte[0])
                .build();
        }
        
        private User parseJson(String json) {
            // 简化的 JSON 解析逻辑
            // 实际应使用 Jackson 或 Gson
            return new User("", 0, "");
        }
    }
    
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        Schema<User> customSchema = new CustomUserSchema();
        
        Producer<User> producer = client.newProducer(customSchema)
            .topic("persistent://public/default/custom-schema-topic")
            .create();
        
        producer.send(new User("Charlie", 35, "charlie@example.com"));
        
        producer.close();
        client.close();
    }
}
```

---

### 2.4 消息属性与键

```java
public class MessagePropertiesExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        Producer<String> producer = client.newProducer(Schema.STRING)
            .topic("persistent://public/default/properties-topic")
            .create();
        
        // 发送带属性的消息
        producer.newMessage()
            .value("Message with properties")
            .key("order-123")                           // 消息 Key
            .property("source", "web-app")              // 自定义属性
            .property("priority", "high")
            .eventTime(System.currentTimeMillis())      // 事件时间
            .send();
        
        // 消费者读取属性
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/properties-topic")
            .subscriptionName("properties-sub")
            .subscribe();
        
        Message<String> msg = consumer.receive();
        
        System.out.println("Value: " + msg.getValue());
        System.out.println("Key: " + msg.getKey());
        System.out.println("Properties: " + msg.getProperties());
        System.out.println("Source: " + msg.getProperty("source"));
        System.out.println("Event time: " + msg.getEventTime());
        
        consumer.acknowledge(msg);
        producer.close();
        consumer.close();
        client.close();
    }
}
```

**使用场景**：
- `key`：分区路由、Key_Shared 订阅、消息去重
- `properties`：传递元数据（来源、优先级、追踪 ID 等）
- `eventTime`：事件溯源、时间窗口聚合

---

### 2.5 消息重试与死信队列

```java
public class RetryAndDLQExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // 配置死信队列策略
        DeadLetterPolicy deadLetterPolicy = DeadLetterPolicy.builder()
            .maxRedeliverCount(3)                           // 最大重试 3 次
            .deadLetterTopic("persistent://public/default/my-topic-DLQ")
            .build();
        
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/my-topic")
            .subscriptionName("retry-sub")
            .subscriptionType(SubscriptionType.Shared)
            .ackTimeout(60, TimeUnit.SECONDS)              // 60 秒未确认自动重试
            .negativeAckRedeliveryDelay(1, TimeUnit.MINUTES)  // 否定确认 1 分钟后重试
            .deadLetterPolicy(deadLetterPolicy)
            .subscribe();
        
        while (true) {
            Message<String> msg = consumer.receive();
            try {
                processMessage(msg);
                consumer.acknowledge(msg);
            } catch (Exception e) {
                System.err.println("Failed to process message, will retry: " + e.getMessage());
                // 立即重试
                consumer.negativeAcknowledge(msg);
            }
        }
    }
    
    private static void processMessage(Message<String> msg) throws Exception {
        // 业务处理逻辑
        if (msg.getValue().contains("error")) {
            throw new Exception("Processing error");
        }
        System.out.println("Processed: " + msg.getValue());
    }
}

// 死信队列消费者
public class DLQConsumerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // 订阅死信队列
        Consumer<String> dlqConsumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/my-topic-DLQ")
            .subscriptionName("dlq-sub")
            .subscribe();
        
        while (true) {
            Message<String> msg = dlqConsumer.receive();
            System.err.println("Dead letter message: " + msg.getValue());
            System.err.println("Redelivery count: " + msg.getRedeliveryCount());
            
            // 记录到日志、发送告警、人工处理等
            dlqConsumer.acknowledge(msg);
        }
    }
}
```

**重试机制**：
1. **Ack Timeout**：未确认消息自动重新投递
2. **Negative Ack**：显式标记失败，立即重试
3. **Dead Letter Queue**：超过最大重试次数，路由到 DLQ

---

### 2.6 批量接收

```java
public class BatchReceiveExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // 配置批量接收策略
        BatchReceivePolicy batchReceivePolicy = BatchReceivePolicy.builder()
            .maxNumMessages(100)                    // 最多 100 条消息
            .maxNumBytes(10 * 1024 * 1024)         // 最多 10 MB
            .timeout(100, TimeUnit.MILLISECONDS)    // 最多等待 100 ms
            .build();
        
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/batch-topic")
            .subscriptionName("batch-sub")
            .batchReceivePolicy(batchReceivePolicy)
            .subscribe();
        
        while (true) {
            // 批量接收
            Messages<String> messages = consumer.batchReceive();
            
            System.out.println("Received batch size: " + messages.size());
            
            // 处理批量消息
            for (Message<String> msg : messages) {
                System.out.println("Processing: " + msg.getValue());
            }
            
            // 批量确认
            consumer.acknowledge(messages);
        }
    }
}
```

**优势**：
- 减少 `receive()` 调用次数
- 提升吞吐量 5-10 倍
- 适合批量处理场景（批量写入数据库、批量计算等）

---

### 2.7 Seek 操作（重置消费位置）

```java
public class SeekExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/seek-topic")
            .subscriptionName("seek-sub")
            .subscribe();
        
        // 消费一些消息
        for (int i = 0; i < 10; i++) {
            Message<String> msg = consumer.receive();
            System.out.println("Received: " + msg.getValue());
            consumer.acknowledge(msg);
        }
        
        // Seek 到最早位置
        consumer.seek(MessageId.earliest);
        System.out.println("Seeked to earliest, re-consuming...");
        
        for (int i = 0; i < 5; i++) {
            Message<String> msg = consumer.receive();
            System.out.println("Re-received: " + msg.getValue());
            consumer.acknowledge(msg);
        }
        
        // Seek 到指定时间戳（重新消费过去 1 小时的消息）
        long oneHourAgo = System.currentTimeMillis() - TimeUnit.HOURS.toMillis(1);
        consumer.seek(oneHourAgo);
        
        // Seek 到指定 MessageId
        MessageId specificMessageId = // ... 从某处获取
        consumer.seek(specificMessageId);
        
        consumer.close();
        client.close();
    }
}
```

**使用场景**：
- 回溯历史数据
- 故障恢复后重新处理
- 数据回填和修复

---

### 2.8 Reader 模式（无订阅读取）

```java
public class ReaderExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        // 创建 Reader（从最早位置开始读）
        Reader<String> reader = client.newReader(Schema.STRING)
            .topic("persistent://public/default/reader-topic")
            .startMessageId(MessageId.earliest)
            .create();
        
        // 读取消息
        while (reader.hasMessageAvailable()) {
            Message<String> msg = reader.readNext();
            System.out.println("Read: " + msg.getValue());
            // Reader 不需要 acknowledge
        }
        
        reader.close();
        client.close();
    }
}
```

**Reader vs Consumer**：
- Reader：无订阅，手动控制位置，适合一次性读取、数据导出
- Consumer：有订阅，自动管理位置，适合持续消费、多消费者负载均衡

---

## 三、事务支持

### 3.1 事务性生产与消费

```java
public class TransactionExample {
    public static void main(String[] args) throws Exception {
        // 启用事务
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .enableTransaction(true)
            .build();
        
        Producer<String> producer = client.newProducer(Schema.STRING)
            .topic("persistent://public/default/txn-topic")
            .sendTimeout(0, TimeUnit.SECONDS)  // 禁用超时（事务内必须）
            .create();
        
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/txn-input")
            .subscriptionName("txn-sub")
            .subscribe();
        
        // 开启事务
        Transaction txn = client.newTransaction()
            .withTransactionTimeout(5, TimeUnit.MINUTES)
            .build()
            .get();
        
        try {
            // 1. 事务性消费
            Message<String> inputMsg = consumer.receive();
            consumer.acknowledgeAsync(inputMsg.getMessageId(), txn).get();
            
            // 2. 业务处理
            String result = processMessage(inputMsg.getValue());
            
            // 3. 事务性生产
            producer.newMessage(txn)
                .value(result)
                .sendAsync()
                .get();
            
            // 4. 提交事务
            txn.commit().get();
            System.out.println("Transaction committed");
            
        } catch (Exception e) {
            // 5. 回滚事务
            txn.abort().get();
            System.err.println("Transaction aborted: " + e.getMessage());
        }
        
        producer.close();
        consumer.close();
        client.close();
    }
    
    private static String processMessage(String input) {
        return "Processed: " + input;
    }
}
```

**事务保证**：
- **Exactly-Once 语义**：消息恰好处理一次
- **原子性**：多个操作要么全部成功，要么全部失败
- **跨分区/跨主题**：支持多个 Producer 和 Consumer 参与同一事务

### 3.2 跨主题事务

```java
public class CrossTopicTransactionExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .enableTransaction(true)
            .build();
        
        Producer<String> producer1 = client.newProducer(Schema.STRING)
            .topic("persistent://public/default/txn-topic-1")
            .sendTimeout(0, TimeUnit.SECONDS)
            .create();
        
        Producer<String> producer2 = client.newProducer(Schema.STRING)
            .topic("persistent://public/default/txn-topic-2")
            .sendTimeout(0, TimeUnit.SECONDS)
            .create();
        
        Consumer<String> consumer = client.newConsumer(Schema.STRING)
            .topic("persistent://public/default/txn-input")
            .subscriptionName("cross-topic-txn-sub")
            .subscribe();
        
        Transaction txn = client.newTransaction().build().get();
        
        try {
            // 消费输入
            Message<String> inputMsg = consumer.receive();
            consumer.acknowledgeAsync(inputMsg.getMessageId(), txn).get();
            
            // 跨主题生产
            producer1.newMessage(txn).value("Output 1").sendAsync().get();
            producer2.newMessage(txn).value("Output 2").sendAsync().get();
            
            // 提交事务
            txn.commit().get();
            
        } catch (Exception e) {
            txn.abort().get();
        }
        
        producer1.close();
        producer2.close();
        consumer.close();
        client.close();
    }
}
```

---

## 四、性能优化最佳实践

### 4.1 高吞吐量场景

```java
public class HighThroughputProducerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .ioThreads(Runtime.getRuntime().availableProcessors())
            .listenerThreads(Runtime.getRuntime().availableProcessors())
            .connectionsPerBroker(5)
            .build();
        
        Producer<byte[]> producer = client.newProducer()
            .topic("persistent://public/default/high-throughput-topic")
            // 启用批量
            .enableBatching(true)
            .batchingMaxPublishDelay(10, TimeUnit.MILLISECONDS)
            .batchingMaxMessages(1000)
            .batchingMaxBytes(128 * 1024)
            // 启用压缩
            .compressionType(CompressionType.LZ4)
            // 增大队列
            .maxPendingMessages(50000)
            .blockIfQueueFull(true)
            // 超时配置
            .sendTimeout(30, TimeUnit.SECONDS)
            .create();
        
        // 异步批量发送
        ExecutorService executor = Executors.newFixedThreadPool(10);
        AtomicLong totalSent = new AtomicLong(0);
        
        for (int i = 0; i < 10; i++) {
            executor.submit(() -> {
                for (int j = 0; j < 100000; j++) {
                    byte[] data = ("Message " + j).getBytes();
                    producer.sendAsync(data).thenRun(() -> {
                        long sent = totalSent.incrementAndGet();
                        if (sent % 10000 == 0) {
                            System.out.println("Sent: " + sent);
                        }
                    });
                }
            });
        }
        
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);
        
        producer.flush();
        producer.close();
        client.close();
    }
}
```

**优化要点**：
- **批量发送**：10-100 倍吞吐量提升
- **压缩**：减少网络传输，LZ4 平衡性能和压缩率
- **异步发送**：非阻塞，充分利用 CPU
- **多线程并发**：提升并发度
- **增大队列**：缓冲突发流量

### 4.2 低延迟场景

```java
public class LowLatencyProducerExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        Producer<byte[]> producer = client.newProducer()
            .topic("persistent://public/default/low-latency-topic")
            // 禁用批量（减少延迟）
            .enableBatching(false)
            // 禁用压缩
            .compressionType(CompressionType.NONE)
            // 短超时
            .sendTimeout(5, TimeUnit.SECONDS)
            .create();
        
        // 同步发送（确保消息立即发送）
        byte[] data = "Low latency message".getBytes();
        MessageId msgId = producer.send(data);
        System.out.println("Sent with low latency: " + msgId);
        
        producer.close();
        client.close();
    }
}
```

**优化要点**：
- **禁用批量**：避免批量延迟
- **同步发送**：立即发送，不等待批量
- **禁用压缩**：减少 CPU 开销

### 4.3 大消息处理

```java
public class LargeMessageExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .build();
        
        Producer<byte[]> producer = client.newProducer()
            .topic("persistent://public/default/large-message-topic")
            // 禁用批量（大消息不适合批量）
            .enableBatching(false)
            // 启用分块（Chunking）
            .enableChunking(true)
            .maxMessageSize(10 * 1024 * 1024)  // 10 MB
            .create();
        
        // 发送大消息
        byte[] largeData = new byte[5 * 1024 * 1024];  // 5 MB
        Arrays.fill(largeData, (byte) 'A');
        
        MessageId msgId = producer.send(largeData);
        System.out.println("Sent large message: " + msgId);
        
        // 消费者自动处理分块
        Consumer<byte[]> consumer = client.newConsumer()
            .topic("persistent://public/default/large-message-topic")
            .subscriptionName("large-msg-sub")
            .subscribe();
        
        Message<byte[]> msg = consumer.receive();
        System.out.println("Received large message size: " + msg.getData().length);
        consumer.acknowledge(msg);
        
        producer.close();
        consumer.close();
        client.close();
    }
}
```

**大消息最佳实践**：
- **启用分块（Chunking）**：自动拆分和组装
- **使用对象存储**：大文件存储到 S3/GCS，消息仅传递引用
- **限制消息大小**：建议 < 1 MB，避免影响性能

---

## 五、生产环境部署建议

### 5.1 客户端配置优化

```java
public class ProductionClientConfig {
    public static PulsarClient createProductionClient() throws Exception {
        return PulsarClient.builder()
            // Broker 地址（支持多个）
            .serviceUrl("pulsar://broker1:6650,broker2:6650,broker3:6650")
            // 线程池配置
            .ioThreads(Runtime.getRuntime().availableProcessors() * 2)
            .listenerThreads(Runtime.getRuntime().availableProcessors())
            // 连接配置
            .connectionsPerBroker(3)
            .connectionTimeout(10, TimeUnit.SECONDS)
            .operationTimeout(30, TimeUnit.SECONDS)
            .keepAliveInterval(30, TimeUnit.SECONDS)
            // 重试配置
            .maxNumberOfRejectedRequestPerConnection(50)
            .maxLookupRequests(50000)
            // 内存限制
            .memoryLimit(64L * 1024 * 1024, SizeUnit.MEGA_BYTES)
            // TLS 配置（生产环境推荐）
            .serviceUrl("pulsar+ssl://broker1:6651")
            .tlsTrustCertsFilePath("/path/to/ca.cert.pem")
            .enableTlsHostnameVerification(true)
            // 认证配置
            .authentication(AuthenticationFactory.token("eyJhbGciOiJIUzI1NiJ9..."))
            .build();
    }
}
```

### 5.2 监控与可观测性

```java
public class MonitoringExample {
    public static void main(String[] args) throws Exception {
        PulsarClient client = PulsarClient.builder()
            .serviceUrl("pulsar://localhost:6650")
            .statsIntervalSeconds(60)  // 每 60 秒输出统计
            .build();
        
        Producer<String> producer = client.newProducer(Schema.STRING)
            .topic("persistent://public/default/monitored-topic")
            .create();
        
        // 发送消息
        for (int i = 0; i < 1000; i++) {
            producer.sendAsync("Message " + i);
        }
        
        // 获取统计信息
        ProducerStats stats = producer.getStats();
        System.out.println("Total messages sent: " + stats.getNumMsgsSent());
        System.out.println("Total bytes sent: " + stats.getTotalBytesSent());
        System.out.println("Send latency P50: " + stats.getSendLatencyMillis50Percentile());
        System.out.println("Send latency P99: " + stats.getSendLatencyMillis99Percentile());
        System.out.println("Pending queue size: " + stats.getPendingQueueSize());
        
        producer.close();
        client.close();
    }
}
```

**关键指标**：
- **生产者**：发送速率、延迟分位数、待发送队列大小
- **消费者**：接收速率、确认速率、接收队列大小、未确认消息数
- **客户端**：连接数、重连次数、操作超时次数

### 5.3 故障处理与重试

```java
public class FaultTolerantProducerExample {
    private static final int MAX_RETRIES = 3;
    private static final long RETRY_DELAY_MS = 1000;
    
    public static void main(String[] args) {
        PulsarClient client = null;
        Producer<String> producer = null;
        
        try {
            client = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .build();
            
            producer = client.newProducer(Schema.STRING)
                .topic("persistent://public/default/fault-tolerant-topic")
                .sendTimeout(10, TimeUnit.SECONDS)
                .create();
            
            // 发送消息（带重试）
            sendWithRetry(producer, "Important message");
            
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
        } finally {
            // 优雅关闭
            closeGracefully(producer, client);
        }
    }
    
    private static void sendWithRetry(Producer<String> producer, String message) {
        int attempt = 0;
        while (attempt < MAX_RETRIES) {
            try {
                MessageId msgId = producer.send(message);
                System.out.println("Message sent successfully: " + msgId);
                return;
            } catch (PulsarClientException e) {
                attempt++;
                System.err.println("Send failed (attempt " + attempt + "): " + e.getMessage());
                
                if (attempt < MAX_RETRIES) {
                    try {
                        Thread.sleep(RETRY_DELAY_MS * attempt);  // 指数退避
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException(ie);
                    }
                }
            }
        }
        throw new RuntimeException("Failed to send message after " + MAX_RETRIES + " attempts");
    }
    
    private static void closeGracefully(Producer<?> producer, PulsarClient client) {
        if (producer != null) {
            try {
                producer.flush();
                producer.close();
            } catch (PulsarClientException e) {
                System.err.println("Error closing producer: " + e.getMessage());
            }
        }
        
        if (client != null) {
            try {
                client.close();
            } catch (PulsarClientException e) {
                System.err.println("Error closing client: " + e.getMessage());
            }
        }
    }
}
```

### 5.4 资源管理与连接池

```java
public class ConnectionPoolExample {
    private static final PulsarClient sharedClient;
    private static final Map<String, Producer<?>> producerCache = new ConcurrentHashMap<>();
    
    static {
        try {
            sharedClient = PulsarClient.builder()
                .serviceUrl("pulsar://localhost:6650")
                .ioThreads(8)
                .build();
        } catch (PulsarClientException e) {
            throw new RuntimeException("Failed to create Pulsar client", e);
        }
    }
    
    public static <T> Producer<T> getProducer(String topic, Schema<T> schema) throws PulsarClientException {
        return (Producer<T>) producerCache.computeIfAbsent(topic, t -> {
            try {
                return sharedClient.newProducer(schema)
                    .topic(t)
                    .create();
            } catch (PulsarClientException e) {
                throw new RuntimeException("Failed to create producer for topic: " + t, e);
            }
        });
    }
    
    public static void shutdown() {
        // 关闭所有 Producer
        producerCache.values().forEach(producer -> {
            try {
                producer.close();
            } catch (PulsarClientException e) {
                System.err.println("Error closing producer: " + e.getMessage());
            }
        });
        
        // 关闭 Client
        try {
            sharedClient.close();
        } catch (PulsarClientException e) {
            System.err.println("Error closing client: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) throws Exception {
        // 复用 Producer
        Producer<String> producer1 = getProducer("topic-1", Schema.STRING);
        Producer<String> producer2 = getProducer("topic-2", Schema.STRING);
        Producer<String> producer1Again = getProducer("topic-1", Schema.STRING);
        
        assert producer1 == producer1Again;  // 同一实例
        
        producer1.send("Message to topic-1");
        producer2.send("Message to topic-2");
        
        // JVM 关闭时调用
        Runtime.getRuntime().addShutdownHook(new Thread(ConnectionPoolExample::shutdown));
    }
}
```

**资源管理最佳实践**：
- **单例 PulsarClient**：进程内共享，避免重复创建
- **Producer/Consumer 缓存**：复用实例，减少连接开销
- **优雅关闭**：JVM 关闭时释放资源
- **连接池**：限制连接数，避免资源耗尽

---

## 六、常见问题与故障排查

### 6.1 消息丢失排查

**问题**：Producer 发送成功，但 Consumer 未收到消息

**排查步骤**：
1. 检查订阅类型和 Consumer 数量
2. 查看 Broker 日志是否有错误
3. 检查积压消息数量：`pulsar-admin topics stats persistent://public/default/my-topic`
4. 验证消息确认是否正常

**解决方案**：
```java
// 确保消息持久化
Producer<String> producer = client.newProducer(Schema.STRING)
    .topic("persistent://public/default/my-topic")  // 使用 persistent
    .sendTimeout(30, TimeUnit.SECONDS)
    .create();

// 确保消息确认
consumer.acknowledgeAsync(msg).get();  // 等待确认完成
```

### 6.2 消息重复排查

**问题**：Consumer 收到重复消息

**原因**：
- Consumer 未及时确认（Ack Timeout）
- Consumer 重启或网络抖动
- Broker 故障切换

**解决方案**：
```java
// 业务侧幂等处理
Set<String> processedMessageIds = new ConcurrentHashSet<>();

consumer.receiveAsync().thenAccept(msg -> {
    String messageId = msg.getMessageId().toString();
    
    if (processedMessageIds.contains(messageId)) {
        // 重复消息，直接确认
        consumer.acknowledgeAsync(msg);
        return;
    }
    
    // 处理消息
    processMessage(msg);
    
    // 记录已处理
    processedMessageIds.add(messageId);
    consumer.acknowledgeAsync(msg);
});
```

### 6.3 延迟过高排查

**问题**：消息端到端延迟 > 100 ms

**排查步骤**：
1. 检查生产者批量延迟：`batchingMaxPublishDelay`
2. 检查消费者接收队列：`receiverQueueSize`
3. 检查网络延迟：`ping broker-host`
4. 检查 Broker 负载：CPU、内存、磁盘 IO

**优化方案**：
```java
// 低延迟生产者配置
Producer<String> producer = client.newProducer(Schema.STRING)
    .topic("my-topic")
    .enableBatching(false)              // 禁用批量
    .compressionType(CompressionType.NONE)  // 禁用压缩
    .create();

// 低延迟消费者配置
Consumer<String> consumer = client.newConsumer(Schema.STRING)
    .topic("my-topic")
    .subscriptionName("my-sub")
    .receiverQueueSize(100)            // 减小接收队列
    .subscribe();
```

---

## 七、总结

本文档涵盖了 Apache Pulsar 的主要使用场景和最佳实践：

1. **基础使用**：简单生产者/消费者、异步 API、消息监听器
2. **进阶功能**：分区主题、多主题订阅、Schema、消息属性、重试与 DLQ
3. **事务支持**：Exactly-Once 语义、跨主题事务
4. **性能优化**：高吞吐量配置、低延迟配置、大消息处理
5. **生产部署**：客户端配置、监控、故障处理、资源管理
6. **故障排查**：消息丢失、消息重复、延迟过高

**关键要点**：
- 根据场景选择合适的订阅类型和配置
- 生产环境启用 TLS 和认证
- 实施监控和告警
- 设计幂等处理逻辑
- 优雅关闭和资源管理

**进一步学习**：
- 官方文档：https://pulsar.apache.org/docs/
- 源码剖析：参考本系列其他文档
- 社区实践：Pulsar Summit 演讲、博客文章

---

**文档版本**：v1.0  
**对应 Pulsar 版本**：4.2.0-SNAPSHOT  
**最后更新**：2025-10-05

