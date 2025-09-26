---
title: "Kafka Connect 框架：Worker/Connector/Task 与数据管道实现要点"
date: 2024-10-22T12:00:00+08:00
draft: false
featured: true
series: "kafka-architecture"
tags: ["Kafka Connect", "Connector", "Task", "Source", "Sink", "源码分析"]
categories: ["kafka", "数据集成"]
author: "kafka connect team"
description: "Kafka Connect 框架的运行模型与实现精要，覆盖 Worker、Connector、Task、Converter/Transfomer 的关键路径与边界条件"
showToc: true
tocOpen: true
showReadingTime: true
showWordCount: true
weight: 490
slug: "kafka-connect-framework"
---

## 概述

Kafka Connect 提供可扩展的数据集成框架，支持 Source 与 Sink 连接器的分布式运行。本文补充关键函数核心代码、调用链、时序与类结构图，并合并与其它文档的相似内容，保持中性描述。

<!--more-->

## 1. 运行时组件架构

```mermaid
graph TB
  subgraph Runtime
    W[Worker]
    HER[Herder]
    REST[Connect REST]
    OFF[OffsetBackingStore]
  end
  subgraph Connector
    C[Connector]
    T[Task]
    TR[Transformation]
    CV[Converter]
  end
  subgraph Kafka
    P[Producer]
    CO[Consumer]
    TOP[Topics]
  end

  REST --> HER
  HER --> W
  W --> C
  C --> T
  T --> CV
  T --> TR
  T --> P
  T --> CO
  CO --> TOP
  P --> TOP
  OFF --> W
```

## 2. 关键函数核心代码与说明（精要）

```java
// WorkerTask 拉取与投递（Sink 任务摘要）
public void execute() {
  while (active.get()) {
    List<SinkRecord> batch = consumer.poll(batchSize, pollTimeoutMs);
    if (!batch.isEmpty()) {
      List<SinkRecord> transformed = applyTransformations(batch);
      connectorTask.put(transformed);
      commitOffsetsIfNeeded();
    }
  }
}
```

- 功能：按批拉取数据，应用变换并交给 `SinkTask.put`，按策略提交偏移量。

```java
// Source 任务轮询（摘要）
public void execute() {
  while (active.get()) {
    List<SourceRecord> records = connectorTask.poll();
    if (records != null && !records.isEmpty()) {
      List<ProducerRecord<byte[], byte[]>> serialized = convert(records);
      producer.send(serialized);
      commitSourceOffsets(records);
    }
  }
}
```

- 功能：从外部系统拉取生成 `SourceRecord`，经转换后写入 Kafka 并记录 Source 偏移。

```java
// Converter 序列化（摘要）
public byte[] fromConnectData(String topic, Schema schema, Object value) {
  if (schema == null) return serializeNull(value);
  return serializer.serialize(topic, dataConverter.convert(schema, value));
}

public Object toConnectData(String topic, byte[] value) {
  if (value == null) return null;
  return dataConverter.toConnectValue(topic, value);
}
```

- 功能：在 Connect 内部数据模型与字节表示之间转换，常见实现有 JSON/Avro/Protobuf。

```java
// 偏移存储（摘要）
public void put(ConnectorTaskId taskId, Map<ByteBuffer, ByteBuffer> offsets) {
  KafkaBasedLog log = topicLog(taskId);
  offsets.forEach((k,v) -> log.send(k, v));
}
```

- 功能：将任务偏移写入 `__consumer_offsets` 或专用偏移主题（分布式模式）。

## 3. 调用链（Source 与 Sink）

```mermaid
flowchart LR
  REST[REST] --> Herder[Herder.createConnector]
  Herder --> Worker[Worker.startConnector]
  Worker --> Task[WorkerTask.execute]
  Task --> SourcePoll[SourceTask.poll]
  SourcePoll --> Convert[Converter.fromConnectData]
  Convert --> Producer
  Producer --> Kafka
```

```mermaid
flowchart LR
  REST[REST] --> Herder[Herder.createConnector]
  Herder --> Worker[Worker.startConnector]
  Worker --> Task[WorkerSinkTask.execute]
  Task --> Consumer[Consumer.poll]
  Consumer --> Transform[Transformation.apply]
  Transform --> SinkPut[SinkTask.put]
  SinkPut --> Commit[commitOffsets]
```

## 4. 时序图（批处理与提交）

```mermaid
sequenceDiagram
  participant H as Herder
  participant W as Worker
  participant T as Task
  participant K as Kafka

  H->>W: startConnector
  W->>T: startTask
  loop 周期性
    T->>K: poll()/send()
    T->>T: 转换/写入/调用put
    alt 提交时机到达
      T->>K: commitSync(offsets)
    end
  end
```

## 5. 类结构图与继承关系（简化）

```mermaid
classDiagram
  class Worker
  class Herder
  class Connector
  class SourceConnector
  class SinkConnector
  class Task
  class SourceTask
  class SinkTask
  class Converter
  class Transformation

  Worker --> Connector
  Connector <|-- SourceConnector
  Connector <|-- SinkConnector
  Task <|-- SourceTask
  Task <|-- SinkTask
  Task --> Converter
  Task --> Transformation
```

