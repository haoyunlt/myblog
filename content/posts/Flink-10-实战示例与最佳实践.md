# Flink-10-实战示例与最佳实践

## 一、流处理基础示例

### 1.1 WordCount - 流式词频统计

这是Flink最经典的入门示例，展示了流处理的基本概念。

**场景描述**：实时统计来自Socket的文本流中每个单词的出现次数，每5秒输出一次结果。

**完整代码**：

```java
public class StreamingWordCount {
    
    public static void main(String[] args) throws Exception {
        // 1. 创建执行环境
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // 2. 配置检查点（生产环境必须）
        env.enableCheckpointing(60000); // 60秒检查点间隔
        env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
        
        // 3. 从Socket读取文本流
        DataStream<String> text = env.socketTextStream("localhost", 9999);
        
        // 4. 数据转换：分词、计数、聚合
        DataStream<Tuple2<String, Integer>> wordCounts = text
            // 按行分词
            .flatMap(new FlatMapFunction<String, Tuple2<String, Integer>>() {
                @Override
                public void flatMap(String value, Collector<Tuple2<String, Integer>> out) {
                    for (String word : value.split("\\s+")) {
                        if (word.length() > 0) {
                            out.collect(Tuple2.of(word.toLowerCase(), 1));
                        }
                    }
                }
            })
            // 按单词分组
            .keyBy(tuple -> tuple.f0)
            // 5秒滚动窗口
            .window(TumblingProcessingTimeWindows.of(Time.seconds(5)))
            // 求和
            .sum(1);
        
        // 5. 输出结果
        wordCounts.print();
        
        // 6. 执行作业
        env.execute("Streaming WordCount");
    }
}
```

**关键点说明**：

1. **环境创建**：`StreamExecutionEnvironment`是所有流处理作业的入口
2. **检查点配置**：生产环境必须启用检查点保证容错
3. **数据源**：`socketTextStream`从Socket读取数据，生产环境应使用Kafka等
4. **转换算子**：flatMap分词，keyBy分组，window开窗，sum聚合
5. **执行触发**：调用`execute()`才真正提交作业执行

**测试方法**：
```bash
# 启动netcat作为数据源
nc -lk 9999

# 输入测试数据
hello world
hello flink
flink streaming
```

### 1.2 实时用户行为分析

**场景描述**：分析用户行为日志，统计每个用户最近1小时的PV、UV、行为类型分布。

**数据模型**：

```java
// 用户行为事件
public class UserBehavior {
    public long userId;      // 用户ID
    public long itemId;      // 商品ID
    public String behavior;  // 行为类型：pv/cart/fav/buy
    public long timestamp;   // 时间戳
    
    // 构造函数、Getter/Setter省略
}

// 统计结果
public class UserStatistics {
    public long userId;
    public long windowEnd;
    public long pv;          // 页面浏览数
    public Set<Long> items;  // 浏览的商品集合（用于UV）
    public Map<String, Long> behaviorCount; // 各行为类型计数
    
    public long getUv() {
        return items.size();
    }
}
```

**完整实现**：

```java
public class UserBehaviorAnalysis {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
        env.setParallelism(4);
        
        // 配置检查点
        env.enableCheckpointing(60000);
        CheckpointConfig config = env.getCheckpointConfig();
        config.setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
        config.setMinPauseBetweenCheckpoints(30000);
        config.setCheckpointTimeout(600000);
        config.setExternalizedCheckpointCleanup(
            ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
        
        // 从Kafka读取数据
        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            "user-behavior",
            new SimpleStringSchema(),
            getKafkaProperties());
        consumer.setStartFromLatest();
        
        DataStream<String> source = env.addSource(consumer);
        
        // 解析JSON并分配Watermark
        DataStream<UserBehavior> behaviors = source
            .map(json -> JSON.parseObject(json, UserBehavior.class))
            .assignTimestampsAndWatermarks(
                WatermarkStrategy.<UserBehavior>forBoundedOutOfOrderness(Duration.ofSeconds(10))
                    .withTimestampAssigner((event, timestamp) -> event.timestamp));
        
        // 按用户分组，1小时滑动窗口（每5分钟滑动）
        DataStream<UserStatistics> statistics = behaviors
            .keyBy(behavior -> behavior.userId)
            .window(SlidingEventTimeWindows.of(Time.hours(1), Time.minutes(5)))
            .aggregate(new UserBehaviorAggregateFunction());
        
        // 输出到Kafka
        FlinkKafkaProducer<UserStatistics> producer = new FlinkKafkaProducer<>(
            "user-statistics",
            new UserStatisticsSerializationSchema(),
            getKafkaProperties(),
            FlinkKafkaProducer.Semantic.EXACTLY_ONCE);
        
        statistics.addSink(producer);
        
        env.execute("User Behavior Analysis");
    }
    
    // 聚合函数实现
    public static class UserBehaviorAggregateFunction 
            implements AggregateFunction<UserBehavior, UserStatistics, UserStatistics> {
        
        @Override
        public UserStatistics createAccumulator() {
            UserStatistics acc = new UserStatistics();
            acc.items = new HashSet<>();
            acc.behaviorCount = new HashMap<>();
            return acc;
        }
        
        @Override
        public UserStatistics add(UserBehavior behavior, UserStatistics acc) {
            acc.userId = behavior.userId;
            
            // 统计PV
            if ("pv".equals(behavior.behavior)) {
                acc.pv++;
                acc.items.add(behavior.itemId); // 用于UV统计
            }
            
            // 统计各行为类型
            acc.behaviorCount.merge(behavior.behavior, 1L, Long::sum);
            
            return acc;
        }
        
        @Override
        public UserStatistics getResult(UserStatistics acc) {
            return acc;
        }
        
        @Override
        public UserStatistics merge(UserStatistics a, UserStatistics b) {
            a.pv += b.pv;
            a.items.addAll(b.items);
            b.behaviorCount.forEach((k, v) -> a.behaviorCount.merge(k, v, Long::sum));
            return a;
        }
    }
    
    private static Properties getKafkaProperties() {
        Properties props = new Properties();
        props.setProperty("bootstrap.servers", "localhost:9092");
        props.setProperty("group.id", "flink-user-behavior");
        props.setProperty("transaction.timeout.ms", "900000"); // 15分钟
        return props;
    }
}
```

**关键点说明**：

1. **Event Time**：使用事件时间而非处理时间，保证乱序数据正确处理
2. **Watermark**：允许10秒乱序，超过Watermark的数据会被丢弃或进入侧输出流
3. **滑动窗口**：1小时窗口，每5分钟计算一次，实现近实时统计
4. **AggregateFunction**：增量聚合，内存效率高，适合大窗口
5. **Exactly-Once**：Kafka Producer配置精确一次语义

## 二、状态管理最佳实践

### 2.1 Keyed State使用

**场景**：实现用户会话跟踪，记录每个用户的最后活动时间和累计行为次数。

```java
public class SessionTracker extends KeyedProcessFunction<Long, UserBehavior, SessionInfo> {
    
    // ValueState：存储用户最后活动时间
    private transient ValueState<Long> lastActiveTime;
    
    // ValueState：存储累计行为次数
    private transient ValueState<Long> behaviorCount;
    
    // MapState：存储各行为类型的计数
    private transient MapState<String, Long> behaviorTypeCount;
    
    @Override
    public void open(Configuration parameters) {
        // 注册状态
        ValueStateDescriptor<Long> lastActiveDescriptor = 
            new ValueStateDescriptor<>("lastActiveTime", Long.class);
        lastActiveTime = getRuntimeContext().getState(lastActiveDescriptor);
        
        ValueStateDescriptor<Long> countDescriptor = 
            new ValueStateDescriptor<>("behaviorCount", Long.class);
        behaviorCount = getRuntimeContext().getState(countDescriptor);
        
        MapStateDescriptor<String, Long> typeCountDescriptor =
            new MapStateDescriptor<>("behaviorTypeCount", String.class, Long.class);
        behaviorTypeCount = getRuntimeContext().getMapState(typeCountDescriptor);
    }
    
    @Override
    public void processElement(
            UserBehavior behavior,
            Context ctx,
            Collector<SessionInfo> out) throws Exception {
        
        // 更新最后活动时间
        lastActiveTime.update(behavior.timestamp);
        
        // 累加行为次数
        Long count = behaviorCount.value();
        behaviorCount.update(count == null ? 1 : count + 1);
        
        // 更新行为类型计数
        Long typeCount = behaviorTypeCount.get(behavior.behavior);
        behaviorTypeCount.put(behavior.behavior, typeCount == null ? 1 : typeCount + 1);
        
        // 注册30分钟后的定时器（会话超时）
        long sessionTimeout = behavior.timestamp + 30 * 60 * 1000;
        ctx.timerService().registerEventTimeTimer(sessionTimeout);
    }
    
    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<SessionInfo> out) 
            throws Exception {
        // 会话超时，输出会话信息并清理状态
        SessionInfo info = new SessionInfo();
        info.userId = ctx.getCurrentKey();
        info.lastActiveTime = lastActiveTime.value();
        info.totalBehaviors = behaviorCount.value();
        info.behaviorBreakdown = new HashMap<>();
        
        for (Map.Entry<String, Long> entry : behaviorTypeCount.entries()) {
            info.behaviorBreakdown.put(entry.getKey(), entry.getValue());
        }
        
        out.collect(info);
        
        // 清理状态
        lastActiveTime.clear();
        behaviorCount.clear();
        behaviorTypeCount.clear();
    }
}
```

**最佳实践**：

1. **选择合适的State类型**：
   - ValueState：单一值，如计数器、标志位
   - ListState：列表，如事件缓冲区
   - MapState：键值对，如分组统计
   - ReducingState/AggregatingState：需要合并逻辑的聚合

2. **状态清理**：
   - 使用定时器自动清理过期状态
   - 设置状态TTL（StateTtlConfig）
   - 避免状态无限增长导致OOM

3. **状态大小控制**：
   - 避免在状态中存储大对象
   - 使用压缩（如Snappy）
   - 定期清理不再需要的状态

### 2.2 状态TTL配置

```java
public class StateWithTTL extends KeyedProcessFunction<Long, Event, Result> {
    
    private transient ValueState<UserInfo> userInfoState;
    
    @Override
    public void open(Configuration parameters) {
        // 配置状态TTL
        StateTtlConfig ttlConfig = StateTtlConfig
            .newBuilder(Time.hours(24)) // 24小时TTL
            .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite) // 创建和写入时更新
            .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired) // 永不返回过期数据
            .cleanupFullSnapshot() // 全量快照时清理
            .cleanupIncrementally(1000, true) // 增量清理，每次处理1000条
            .build();
        
        ValueStateDescriptor<UserInfo> descriptor = 
            new ValueStateDescriptor<>("userInfo", UserInfo.class);
        descriptor.enableTimeToLive(ttlConfig);
        
        userInfoState = getRuntimeContext().getState(descriptor);
    }
    
    @Override
    public void processElement(Event event, Context ctx, Collector<Result> out) 
            throws Exception {
        // 状态会自动过期，过期后get()返回null
        UserInfo info = userInfoState.value();
        
        if (info == null) {
            info = new UserInfo();
            info.userId = event.userId;
        }
        
        // 更新状态（自动更新TTL）
        info.lastEvent = event;
        info.eventCount++;
        userInfoState.update(info);
        
        out.collect(new Result(info));
    }
}
```

### 2.3 Operator State使用

Operator State适用于与Key无关的状态，如Kafka消费者的offset。

```java
public class BufferedSink implements SinkFunction<Event>, CheckpointedFunction {
    
    private transient ListState<Event> checkpointedState;
    private List<Event> bufferedEvents;
    private final int bufferSize = 100;
    
    @Override
    public void invoke(Event event, Context context) {
        bufferedEvents.add(event);
        
        if (bufferedEvents.size() >= bufferSize) {
            flush();
        }
    }
    
    @Override
    public void snapshotState(FunctionSnapshotContext context) throws Exception {
        // 检查点时保存缓冲区
        checkpointedState.clear();
        checkpointedState.addAll(bufferedEvents);
    }
    
    @Override
    public void initializeState(FunctionInitializationContext context) throws Exception {
        // 恢复或初始化状态
        ListStateDescriptor<Event> descriptor = 
            new ListStateDescriptor<>("buffered-events", Event.class);
        checkpointedState = context.getOperatorStateStore().getListState(descriptor);
        
        bufferedEvents = new ArrayList<>();
        
        if (context.isRestored()) {
            // 从检查点恢复
            for (Event event : checkpointedState.get()) {
                bufferedEvents.add(event);
            }
        }
    }
    
    private void flush() {
        // 批量写入外部系统
        // ...
        bufferedEvents.clear();
    }
}
```

## 三、窗口处理最佳实践

### 3.1 复杂窗口场景

**场景**：实时交易监控，检测5分钟内交易额超过阈值的用户。

```java
public class TradingMonitor {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
        
        DataStream<Trade> trades = env
            .addSource(new TradeSource())
            .assignTimestampsAndWatermarks(
                WatermarkStrategy.<Trade>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((trade, ts) -> trade.timestamp));
        
        // 按用户分组，5分钟滚动窗口
        DataStream<Alert> alerts = trades
            .keyBy(trade -> trade.userId)
            .window(TumblingEventTimeWindows.of(Time.minutes(5)))
            // 使用ProcessWindowFunction获取窗口元数据
            .process(new ProcessWindowFunction<Trade, Alert, Long, TimeWindow>() {
                
                @Override
                public void process(
                        Long userId,
                        Context context,
                        Iterable<Trade> trades,
                        Collector<Alert> out) {
                    
                    double totalAmount = 0;
                    int tradeCount = 0;
                    
                    for (Trade trade : trades) {
                        totalAmount += trade.amount;
                        tradeCount++;
                    }
                    
                    // 检测异常
                    if (totalAmount > 100000) { // 阈值10万
                        Alert alert = new Alert();
                        alert.userId = userId;
                        alert.windowStart = context.window().getStart();
                        alert.windowEnd = context.window().getEnd();
                        alert.totalAmount = totalAmount;
                        alert.tradeCount = tradeCount;
                        alert.avgAmount = totalAmount / tradeCount;
                        alert.alertTime = System.currentTimeMillis();
                        
                        out.collect(alert);
                    }
                }
            });
        
        alerts.print();
        env.execute("Trading Monitor");
    }
}
```

### 3.2 会话窗口

会话窗口根据数据间隔动态确定窗口边界，适合用户会话分析。

```java
public class SessionAnalysis {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        DataStream<UserAction> actions = env.addSource(new UserActionSource());
        
        // 按用户分组，30分钟不活动则会话结束
        DataStream<SessionSummary> sessions = actions
            .keyBy(action -> action.userId)
            .window(EventTimeSessionWindows.withGap(Time.minutes(30)))
            .aggregate(
                new AggregateFunction<UserAction, SessionAccumulator, SessionSummary>() {
                    
                    @Override
                    public SessionAccumulator createAccumulator() {
                        return new SessionAccumulator();
                    }
                    
                    @Override
                    public SessionAccumulator add(UserAction action, SessionAccumulator acc) {
                        if (acc.sessionStart == 0) {
                            acc.sessionStart = action.timestamp;
                        }
                        acc.sessionEnd = action.timestamp;
                        acc.actionCount++;
                        acc.actions.add(action);
                        return acc;
                    }
                    
                    @Override
                    public SessionSummary getResult(SessionAccumulator acc) {
                        SessionSummary summary = new SessionSummary();
                        summary.userId = acc.actions.get(0).userId;
                        summary.sessionStart = acc.sessionStart;
                        summary.sessionEnd = acc.sessionEnd;
                        summary.duration = acc.sessionEnd - acc.sessionStart;
                        summary.actionCount = acc.actionCount;
                        return summary;
                    }
                    
                    @Override
                    public SessionAccumulator merge(SessionAccumulator a, SessionAccumulator b) {
                        a.sessionStart = Math.min(a.sessionStart, b.sessionStart);
                        a.sessionEnd = Math.max(a.sessionEnd, b.sessionEnd);
                        a.actionCount += b.actionCount;
                        a.actions.addAll(b.actions);
                        return a;
                    }
                });
        
        sessions.print();
        env.execute("Session Analysis");
    }
}
```

### 3.3 自定义触发器

实现提前触发：窗口未结束但已有足够数据时提前输出。

```java
public class EarlyTrigger extends Trigger<Object, TimeWindow> {
    
    private final long interval;
    private final ReducingStateDescriptor<Long> countDescriptor;
    
    public EarlyTrigger(long interval) {
        this.interval = interval;
        this.countDescriptor = new ReducingStateDescriptor<>(
            "count", (a, b) -> a + b, Long.class);
    }
    
    @Override
    public TriggerResult onElement(
            Object element,
            long timestamp,
            TimeWindow window,
            TriggerContext ctx) throws Exception {
        
        // 累加计数
        ReducingState<Long> count = ctx.getPartitionedState(countDescriptor);
        count.add(1L);
        
        // 注册窗口结束定时器
        ctx.registerEventTimeTimer(window.maxTimestamp());
        
        // 注册周期性提前触发定时器
        long fireTimestamp = timestamp - (timestamp % interval) + interval;
        if (fireTimestamp < window.maxTimestamp()) {
            ctx.registerEventTimeTimer(fireTimestamp);
        }
        
        return TriggerResult.CONTINUE;
    }
    
    @Override
    public TriggerResult onEventTime(long time, TimeWindow window, TriggerContext ctx) 
            throws Exception {
        if (time == window.maxTimestamp()) {
            // 窗口结束，触发并清除状态
            return TriggerResult.FIRE_AND_PURGE;
        } else {
            // 周期性提前触发，不清除状态
            return TriggerResult.FIRE;
        }
    }
    
    @Override
    public TriggerResult onProcessingTime(long time, TimeWindow window, TriggerContext ctx) {
        return TriggerResult.CONTINUE;
    }
    
    @Override
    public void clear(TimeWindow window, TriggerContext ctx) throws Exception {
        ctx.deleteEventTimeTimer(window.maxTimestamp());
        ctx.getPartitionedState(countDescriptor).clear();
    }
}

// 使用自定义触发器
DataStream<Result> result = input
    .keyBy(...)
    .window(TumblingEventTimeWindows.of(Time.minutes(10)))
    .trigger(new EarlyTrigger(Time.minutes(1).toMilliseconds())) // 每1分钟提前触发
    .aggregate(new MyAggregateFunction());
```

## 四、性能优化实践

### 4.1 Operator Chain优化

**原理**：将多个算子链接在一起，在同一线程中执行，避免序列化和网络传输。

```java
// 默认会自动Chain
DataStream<String> result = input
    .map(new MapFunction1())    // 可以Chain
    .filter(new FilterFunction1())  // 可以Chain
    .map(new MapFunction2());   // 可以Chain

// 禁用Chain
DataStream<String> result = input
    .map(new MapFunction1()).disableChaining()  // 断开Chain
    .filter(new FilterFunction1())
    .map(new MapFunction2());

// 开始新Chain
DataStream<String> result = input
    .map(new MapFunction1())
    .filter(new FilterFunction1()).startNewChain()  // 从这里开始新Chain
    .map(new MapFunction2());

// Slot共享组
DataStream<String> result = input
    .map(new MapFunction1()).slotSharingGroup("group1")
    .filter(new FilterFunction1()).slotSharingGroup("group2");
```

**最佳实践**：
- 默认让Flink自动Chain，通常是最优的
- 仅在特殊情况下手动控制：如需要独立扩缩容的算子
- 合理使用Slot共享组隔离资源密集型算子

### 4.2 并行度设置

```java
// 全局并行度
env.setParallelism(8);

// 算子级并行度
DataStream<String> result = input
    .map(new MapFunction()).setParallelism(4)   // CPU密集型，低并行度
    .filter(new FilterFunction()).setParallelism(8)  // 快速过滤，继承默认
    .keyBy(...)
    .window(...)
    .aggregate(new AggFunc()).setParallelism(16);  // 状态密集型，高并行度
```

**最佳实践**：
- 根据算子特点设置并行度：
  - Source/Sink：根据分区数设置（Kafka分区数）
  - 无状态算子：CPU核心数的1-2倍
  - 有状态算子：考虑状态大小，避免单个Task状态过大
  - 窗口算子：较高并行度，避免数据倾斜
- 生产环境并行度建议：4-32之间，根据负载调整

### 4.3 背压处理

**监控背压**：
```java
// Web UI查看背压指标
// Metrics: backPressuredTimeMsPerSecond, busyTimeMsPerSecond, idleTimeMsPerSecond
```

**处理策略**：

1. **增加资源**：
```java
// 增加并行度
env.setParallelism(16); // 之前是8

// 增加内存
taskmanager.memory.process.size: 8gb // 之前是4gb
```

2. **优化算子**：
```java
// 使用高效的序列化器
env.getConfig().enableForceKryo(); // 避免Kryo
env.getConfig().registerTypeWithKryoSerializer(MyClass.class, MySerializer.class);

// 对象复用（减少GC）
env.getConfig().enableObjectReuse();

// 使用RichFunction避免重复初始化
public class MyMapper extends RichMapFunction<IN, OUT> {
    private transient SomeResource resource;
    
    @Override
    public void open(Configuration parameters) {
        // 初始化一次
        resource = new SomeResource();
    }
    
    @Override
    public OUT map(IN value) {
        return resource.process(value);
    }
}
```

3. **异步I/O**：
```java
// 异步查询外部系统，避免阻塞
DataStream<Result> result = input
    .keyBy(...)
    .asyncWaitOrdered(new AsyncDatabaseRequest(), 1000, TimeUnit.MILLISECONDS, 100);

class AsyncDatabaseRequest extends RichAsyncFunction<Event, Result> {
    
    private transient DatabaseClient client;
    
    @Override
    public void open(Configuration parameters) {
        client = new DatabaseClient();
    }
    
    @Override
    public void asyncInvoke(Event event, ResultFuture<Result> resultFuture) {
        CompletableFuture<Result> future = client.queryAsync(event.id);
        
        future.whenComplete((result, error) -> {
            if (error != null) {
                resultFuture.completeExceptionally(error);
            } else {
                resultFuture.complete(Collections.singleton(result));
            }
        });
    }
}
```

### 4.4 内存调优

**TaskManager内存配置**：
```yaml
taskmanager.memory.process.size: 4g
taskmanager.memory.flink.size: 3.5g

# 详细配置
taskmanager.memory.framework.heap.size: 128m
taskmanager.memory.task.heap.size: 1g
taskmanager.memory.managed.size: 1.5g
taskmanager.memory.task.off-heap.size: 256m
taskmanager.memory.network.fraction: 0.1
taskmanager.memory.network.min: 64mb
taskmanager.memory.network.max: 1gb
taskmanager.memory.jvm-metaspace.size: 256m
taskmanager.memory.jvm-overhead.fraction: 0.1
```

**RocksDB状态后端调优**：
```java
// 使用RocksDB状态后端
EmbeddedRocksDBStateBackend backend = new EmbeddedRocksDBStateBackend();

// 启用增量检查点
backend.enableIncrementalCheckpointing(true);

// 配置RocksDB选项
RocksDBOptions options = new RocksDBOptions();
options.setMaxBackgroundJobs(4);
options.setMaxOpenFiles(10000);

// 配置Block Cache大小
options.setBlockCacheSize(256 * 1024 * 1024); // 256MB

// 配置Write Buffer
options.setWriteBufferSize(64 * 1024 * 1024); // 64MB
options.setMaxWriteBufferNumber(3);

backend.setRocksDBOptions(options);

env.setStateBackend(backend);
```

## 五、生产环境部署

### 5.1 高可用配置

**ZooKeeper HA**：
```yaml
high-availability: zookeeper
high-availability.zookeeper.quorum: zk1:2181,zk2:2181,zk3:2181
high-availability.zookeeper.path.root: /flink
high-availability.cluster-id: /my-cluster
high-availability.storageDir: hdfs:///flink/ha
```

**Kubernetes HA**：
```yaml
high-availability: kubernetes
high-availability.cluster-id: my-flink-cluster
kubernetes.cluster-id: my-k8s-cluster
high-availability.storageDir: hdfs:///flink/ha
```

### 5.2 检查点配置

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// 基础配置
env.enableCheckpointing(60000); // 60秒间隔
CheckpointConfig config = env.getCheckpointConfig();

// 检查点模式
config.setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);

// 超时时间
config.setCheckpointTimeout(600000); // 10分钟

// 最小间隔
config.setMinPauseBetweenCheckpoints(30000); // 30秒

// 最大并发
config.setMaxConcurrentCheckpoints(1);

// 外部化检查点
config.setExternalizedCheckpointCleanup(
    ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);

// 允许失败检查点
config.setTolerableCheckpointFailureNumber(3);

// Unaligned Checkpoint
config.enableUnalignedCheckpoints(true);
config.setAlignedCheckpointTimeout(Duration.ofSeconds(30));

// 状态后端
env.setStateBackend(new HashMapStateBackend());
config.setCheckpointStorage("hdfs:///flink/checkpoints");
```

### 5.3 监控指标

**Prometheus配置**：
```yaml
metrics.reporter.prom.class: org.apache.flink.metrics.prometheus.PrometheusReporter
metrics.reporter.prom.port: 9249
metrics.reporters: prom

# 指标范围
metrics.scope.jm: flink_jobmanager
metrics.scope.jm.job: flink_jobmanager_job
metrics.scope.tm: flink_taskmanager
metrics.scope.tm.job: flink_taskmanager_job
metrics.scope.task: flink_task
metrics.scope.operator: flink_operator
```

**关键指标**：
```
# 检查点指标
flink_jobmanager_job_lastCheckpointDuration
flink_jobmanager_job_lastCheckpointSize
flink_jobmanager_job_numberOfFailedCheckpoints
flink_jobmanager_job_numberOfCompletedCheckpoints

# 吞吐指标
flink_taskmanager_job_task_numRecordsInPerSecond
flink_taskmanager_job_task_numRecordsOutPerSecond

# 延迟指标
flink_taskmanager_job_task_operator_currentInputWatermark
flink_taskmanager_job_task_operator_currentOutputWatermark

# 背压指标
flink_taskmanager_job_task_backPressuredTimeMsPerSecond
flink_taskmanager_job_task_busyTimeMsPerSecond

# 内存指标
flink_taskmanager_Status_JVM_Memory_Heap_Used
flink_taskmanager_Status_JVM_Memory_NonHeap_Used
```

### 5.4 故障排查

**常见问题及解决方案**：

1. **检查点超时**：
   - 增大检查点超时时间
   - 使用增量检查点
   - 优化状态大小

2. **背压严重**：
   - 增加并行度
   - 优化慢算子
   - 使用异步I/O

3. **状态过大**：
   - 使用RocksDB状态后端
   - 设置状态TTL
   - 优化状态结构

4. **OOM**：
   - 增加TaskManager内存
   - 调整内存分配比例
   - 检查状态泄漏

5. **数据倾斜**：
   - 添加随机前缀打散Key
   - 使用两阶段聚合
   - 调整窗口策略

## 六、端到端案例

### 6.1 实时数据仓库

构建基于Flink的实时数据仓库，支持实时OLAP查询。

**架构设计**：
```
Kafka → Flink → Doris/ClickHouse
         ↓
     State Backend (RocksDB)
```

**实现**：
```java
public class RealtimeDataWarehouse {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
        env.setParallelism(8);
        
        // 配置检查点
        env.enableCheckpointing(300000); // 5分钟
        CheckpointConfig config = env.getCheckpointConfig();
        config.setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
        config.setExternalizedCheckpointCleanup(
            ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
        
        // 从Kafka读取原始数据
        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            "ods-topic",
            new SimpleStringSchema(),
            getKafkaProperties());
        
        DataStream<String> source = env.addSource(consumer);
        
        // ETL处理
        DataStream<DimTable> dimStream = source
            .filter(line -> !line.isEmpty())
            .map(new JsonToObjectMapper())
            .assignTimestampsAndWatermarks(
                WatermarkStrategy.<DimTable>forBoundedOutOfOrderness(Duration.ofSeconds(10))
                    .withTimestampAssigner((event, ts) -> event.timestamp));
        
        // 维度表Join（使用Broadcast State）
        MapStateDescriptor<String, DimData> dimDescriptor = new MapStateDescriptor<>(
            "dim-data", String.class, DimData.class);
        
        BroadcastStream<DimData> dimBroadcast = env
            .addSource(new DimTableSource())
            .broadcast(dimDescriptor);
        
        DataStream<EnrichedFact> enriched = dimStream
            .connect(dimBroadcast)
            .process(new DimJoinProcessFunction(dimDescriptor));
        
        // 实时聚合
        DataStream<AggResult> aggregated = enriched
            .keyBy(fact -> fact.getDimensionKey())
            .window(TumblingEventTimeWindows.of(Time.minutes(5)))
            .aggregate(new MetricsAggregateFunction());
        
        // 写入OLAP数据库
        aggregated.addSink(new DorisSink<>(getDorisConfig()));
        
        env.execute("Realtime Data Warehouse");
    }
    
    // 维度Join函数
    static class DimJoinProcessFunction extends BroadcastProcessFunction<
            DimTable, DimData, EnrichedFact> {
        
        private final MapStateDescriptor<String, DimData> descriptor;
        
        public DimJoinProcessFunction(MapStateDescriptor<String, DimData> descriptor) {
            this.descriptor = descriptor;
        }
        
        @Override
        public void processElement(
                DimTable fact,
                ReadOnlyContext ctx,
                Collector<EnrichedFact> out) throws Exception {
            
            // 从Broadcast State读取维度数据
            ReadOnlyBroadcastState<String, DimData> dimState = 
                ctx.getBroadcastState(descriptor);
            
            DimData dim = dimState.get(fact.dimKey);
            
            if (dim != null) {
                EnrichedFact enriched = new EnrichedFact();
                enriched.setFact(fact);
                enriched.setDim(dim);
                out.collect(enriched);
            }
        }
        
        @Override
        public void processBroadcastElement(
                DimData dim,
                Context ctx,
                Collector<EnrichedFact> out) throws Exception {
            // 更新Broadcast State
            ctx.getBroadcastState(descriptor).put(dim.key, dim);
        }
    }
}
```

### 6.2 实时风控系统

实时检测欺诈交易。

```java
public class FraudDetectionSystem {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        DataStream<Transaction> transactions = env
            .addSource(new TransactionSource())
            .assignTimestampsAndWatermarks(...);
        
        // 规则引擎
        DataStream<Alert> alerts = transactions
            .keyBy(tx -> tx.userId)
            .process(new FraudDetectionFunction());
        
        // 输出告警
        alerts.addSink(new AlertSink());
        
        env.execute("Fraud Detection");
    }
    
    static class FraudDetectionFunction extends KeyedProcessFunction<
            Long, Transaction, Alert> {
        
        // 规则1：5分钟内超过3笔大额交易
        private transient ValueState<Integer> largeTransactionCount;
        
        // 规则2：连续小额交易后突然大额
        private transient ListState<Double> recentAmounts;
        
        @Override
        public void open(Configuration parameters) {
            largeTransactionCount = getRuntimeContext().getState(
                new ValueStateDescriptor<>("large-tx-count", Integer.class));
            
            recentAmounts = getRuntimeContext().getListState(
                new ListStateDescriptor<>("recent-amounts", Double.class));
        }
        
        @Override
        public void processElement(
                Transaction tx,
                Context ctx,
                Collector<Alert> out) throws Exception {
            
            // 规则1：大额交易计数
            if (tx.amount > 10000) {
                Integer count = largeTransactionCount.value();
                count = (count == null ? 0 : count) + 1;
                largeTransactionCount.update(count);
                
                if (count > 3) {
                    out.collect(new Alert("频繁大额交易", tx));
                }
                
                // 5分钟后重置计数
                ctx.timerService().registerEventTimeTimer(
                    tx.timestamp + 5 * 60 * 1000);
            }
            
            // 规则2：交易模式异常
            List<Double> amounts = new ArrayList<>();
            for (Double amt : recentAmounts.get()) {
                amounts.add(amt);
            }
            amounts.add(tx.amount);
            
            if (amounts.size() > 5) {
                // 检测前5笔都是小额，当前是大额
                boolean allSmall = amounts.subList(0, 5).stream()
                    .allMatch(amt -> amt < 100);
                boolean currentLarge = tx.amount > 10000;
                
                if (allSmall && currentLarge) {
                    out.collect(new Alert("交易模式异常", tx));
                }
                
                amounts.remove(0); // 保持窗口大小
            }
            
            recentAmounts.update(amounts);
        }
        
        @Override
        public void onTimer(long timestamp, OnTimerContext ctx, Collector<Alert> out) {
            // 重置计数器
            largeTransactionCount.clear();
        }
    }
}
```

## 七、总结

本文档提供了Flink在实际生产环境中的最佳实践，包括：

1. **基础示例**：从WordCount到复杂的用户行为分析
2. **状态管理**：Keyed State、Operator State、TTL配置
3. **窗口处理**：滚动、滑动、会话窗口及自定义触发器
4. **性能优化**：Operator Chain、并行度、背压处理、内存调优
5. **生产部署**：HA配置、检查点、监控、故障排查
6. **端到端案例**：实时数仓、风控系统

关键要点：
- 始终启用检查点保证容错
- 合理设置并行度和内存配置
- 使用Event Time处理乱序数据
- 监控关键指标及时发现问题
- 根据场景选择合适的状态后端
- 优化算子链和序列化提升性能

