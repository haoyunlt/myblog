# Kitex-03-Remote-时序图

## 1. 消息创建与回收时序图

```mermaid
sequenceDiagram
    autonumber
    participant CLIENT as 客户端代码
    participant POOL as 对象池
    participant MSG as Message实例
    participant TRANSINFO as TransInfo
    participant CODEC as 编解码器
    
    CLIENT->>POOL: NewMessage(data, ri, msgType, role)
    Note over CLIENT,POOL: 创建新消息
    
    POOL->>MSG: messagePool.Get()
    MSG->>MSG: 初始化基本字段
    MSG->>MSG: data = data, rpcInfo = ri
    MSG->>MSG: msgType = msgType, rpcRole = role
    
    MSG->>POOL: transInfoPool.Get()
    POOL->>TRANSINFO: 返回TransInfo实例
    TRANSINFO->>TRANSINFO: 初始化strInfo和intInfo
    TRANSINFO-->>MSG: 设置transInfo字段
    
    MSG-->>CLIENT: 返回Message接口
    
    CLIENT->>MSG: 设置PayloadCodec
    CLIENT->>MSG: 添加Tags信息
    CLIENT->>TRANSINFO: PutTransStrInfo(headers)
    
    Note over CLIENT,CODEC: 使用消息进行编解码
    CLIENT->>CODEC: Encode(ctx, msg, buffer)
    CODEC->>MSG: 读取消息字段
    CODEC->>TRANSINFO: 读取传输信息
    CODEC-->>CLIENT: 编码完成
    
    Note over CLIENT,POOL: 回收消息
    CLIENT->>MSG: Recycle()
    MSG->>MSG: zero()清理所有字段
    MSG->>TRANSINFO: transInfo.Recycle()
    TRANSINFO->>TRANSINFO: zero()清理映射
    TRANSINFO->>POOL: transInfoPool.Put(transInfo)
    MSG->>POOL: messagePool.Put(msg)
```

### 消息生命周期说明

**1. 创建阶段（步骤1-8）**
- 客户端调用NewMessage创建消息实例
- 从messagePool对象池获取message实例
- 初始化消息的基本字段（数据、RPC信息、类型、角色）
- 从transInfoPool获取TransInfo实例并关联

**2. 配置阶段（步骤9-12）**
- 设置负载编解码器
- 添加用户自定义标签
- 设置传输层元信息（如HTTP头）

**3. 使用阶段（步骤13-16）**
- 编解码器读取消息字段进行序列化
- 传输信息用于协议头构建
- 完成网络传输

**4. 回收阶段（步骤17-22）**
- 调用Recycle方法回收消息
- zero方法清理所有字段防止内存泄漏
- 递归回收TransInfo实例
- 将对象放回对象池供下次使用

## 2. 客户端消息发送时序图

```mermaid
sequenceDiagram
    autonumber
    participant CLIENT as 客户端
    participant REMOTECLI as RemoteClient
    participant CONNPOOL as 连接池
    participant CONN as 网络连接
    participant PIPELINE as TransPipeline
    participant OUTBOUND as 出站处理器
    participant TRANSHANDLER as TransHandler
    participant CODEC as 编解码器
    participant BUFFER as ByteBuffer
    participant SERVER as 服务端
    
    CLIENT->>REMOTECLI: Send(ctx, ri, req)
    Note over CLIENT,REMOTECLI: 发送RPC请求
    
    REMOTECLI->>CONNPOOL: GetConn(ctx, dialer, ri)
    CONNPOOL->>CONNPOOL: 检查连接池
    
    alt 池中有可用连接
        CONNPOOL-->>REMOTECLI: 返回现有连接
    else 需要创建新连接
        CONNPOOL->>CONN: Dialer.DialContext()
        CONN-->>CONNPOOL: 新连接建立
        CONNPOOL->>CONNPOOL: 设置连接选项
        CONNPOOL-->>REMOTECLI: 返回新连接
    end
    
    REMOTECLI->>PIPELINE: Write(ctx, conn, req)
    Note over PIPELINE: 执行出站处理器链
    
    loop 所有出站处理器
        PIPELINE->>OUTBOUND: Write(ctx, conn, req)
        OUTBOUND->>OUTBOUND: 处理消息（如加密、压缩）
        alt 处理失败
            OUTBOUND-->>PIPELINE: 返回错误
            PIPELINE-->>REMOTECLI: 传播错误
        else 处理成功
            OUTBOUND-->>PIPELINE: 继续下一个处理器
        end
    end
    
    PIPELINE->>TRANSHANDLER: Write(ctx, conn, req)
    TRANSHANDLER->>CODEC: Encode(ctx, req, buffer)
    
    CODEC->>CODEC: 编码元信息
    CODEC->>CODEC: 编码负载数据
    CODEC->>BUFFER: 写入编码结果
    CODEC-->>TRANSHANDLER: 编码完成
    
    TRANSHANDLER->>CONN: Write(buffer.Bytes())
    CONN->>SERVER: 发送网络数据
    
    alt 发送成功
        CONN-->>TRANSHANDLER: 返回成功
        TRANSHANDLER-->>REMOTECLI: 发送完成
        REMOTECLI-->>CLIENT: 返回成功
    else 发送失败
        CONN-->>TRANSHANDLER: 返回网络错误
        TRANSHANDLER-->>REMOTECLI: 传播错误
        REMOTECLI->>CONNPOOL: ReleaseConn(err, ri)
        CONNPOOL->>CONN: Close()关闭连接
        REMOTECLI-->>CLIENT: 返回错误
    end
```

### 发送流程说明

**1. 连接获取阶段（步骤1-8）**
- 客户端发起发送请求
- 从连接池获取到目标服务的连接
- 优先使用池中现有连接，无可用连接时创建新连接
- 设置连接的超时和缓冲区参数

**2. 管道处理阶段（步骤9-18）**
- 进入传输管道的出站处理流程
- 依次执行所有出站处理器（如加密、压缩、监控）
- 任何处理器失败都会中断管道执行
- 最后由TransHandler执行实际的网络写入

**3. 编码发送阶段（步骤19-27）**
- TransHandler调用编解码器编码消息
- 编码器分别处理元信息和负载数据
- 将编码结果写入网络连接
- 数据通过网络发送到服务端

**4. 结果处理阶段（步骤28-35）**
- 根据发送结果决定连接的处理方式
- 发送成功时保持连接供后续复用
- 发送失败时关闭连接并从池中移除
- 向客户端返回最终的发送结果

## 3. 服务端消息接收时序图

```mermaid
sequenceDiagram
    autonumber
    participant CLIENT as 客户端
    participant NETWORK as 网络层
    participant LISTENER as 监听器
    participant CONN as 连接
    participant TRANSHANDLER as TransHandler
    participant CODEC as 编解码器
    participant BUFFER as ByteBuffer
    participant PIPELINE as TransPipeline
    participant INBOUND as 入站处理器
    participant ENDPOINT as Endpoint
    participant BUSINESS as 业务逻辑
    
    CLIENT->>NETWORK: 发送请求数据
    NETWORK->>LISTENER: 接收网络数据
    LISTENER->>CONN: 建立连接
    
    CONN->>TRANSHANDLER: OnActive(ctx, conn)
    Note over TRANSHANDLER: 连接激活回调
    TRANSHANDLER->>TRANSHANDLER: 初始化连接状态
    TRANSHANDLER-->>CONN: 激活完成
    
    NETWORK->>TRANSHANDLER: OnRead(ctx, conn)
    Note over TRANSHANDLER: 有数据可读
    
    TRANSHANDLER->>TRANSHANDLER: Read(ctx, conn, msg)
    TRANSHANDLER->>BUFFER: 从连接读取数据
    TRANSHANDLER->>CODEC: Decode(ctx, msg, buffer)
    
    CODEC->>CODEC: 解码元信息
    CODEC->>CODEC: 解码负载数据
    CODEC->>CODEC: 设置消息字段
    CODEC-->>TRANSHANDLER: 解码完成
    
    TRANSHANDLER->>PIPELINE: OnMessage(ctx, args, result)
    Note over PIPELINE: 执行入站处理器链
    
    loop 所有入站处理器
        PIPELINE->>INBOUND: OnMessage(ctx, args, result)
        INBOUND->>INBOUND: 处理消息（如解密、验证）
        alt 处理失败
            INBOUND-->>PIPELINE: 返回错误
            PIPELINE-->>TRANSHANDLER: 传播错误
        else 处理成功
            INBOUND-->>PIPELINE: 继续下一个处理器
        end
    end
    
    PIPELINE->>ENDPOINT: Call(ctx, args, result)
    ENDPOINT->>BUSINESS: 调用业务方法
    BUSINESS->>BUSINESS: 执行业务逻辑
    BUSINESS-->>ENDPOINT: 返回处理结果
    ENDPOINT-->>PIPELINE: 返回结果
    
    Note over PIPELINE,TRANSHANDLER: 发送响应
    PIPELINE->>TRANSHANDLER: Write(ctx, conn, result)
    TRANSHANDLER->>CODEC: Encode(ctx, result, buffer)
    CODEC-->>TRANSHANDLER: 编码完成
    TRANSHANDLER->>CONN: Write(buffer.Bytes())
    CONN->>CLIENT: 发送响应数据
    
    alt 连接异常
        CONN->>TRANSHANDLER: OnError(ctx, err, conn)
        TRANSHANDLER->>TRANSHANDLER: 错误处理逻辑
        TRANSHANDLER->>CONN: Close()
    end
    
    alt 连接关闭
        CONN->>TRANSHANDLER: OnInactive(ctx, conn)
        TRANSHANDLER->>TRANSHANDLER: 清理连接资源
    end
```

### 接收处理说明

**1. 连接建立阶段（步骤1-6）**
- 客户端发送请求到服务端
- 网络监听器接收连接请求
- TransHandler的OnActive回调处理连接激活
- 初始化连接相关的状态和资源

**2. 数据读取阶段（步骤7-13）**
- OnRead回调通知有数据可读
- TransHandler从连接读取原始数据
- 调用编解码器解码消息
- 设置Message对象的各个字段

**3. 管道处理阶段（步骤14-24）**
- 进入传输管道的入站处理流程
- 依次执行所有入站处理器（如解密、验证、日志）
- 处理器可以修改消息内容或中断处理
- 最终调用业务Endpoint处理请求

**4. 业务处理阶段（步骤25-29）**
- Endpoint调用具体的业务方法
- 执行业务逻辑并生成响应结果
- 将结果封装到响应Message中

**5. 响应发送阶段（步骤30-35）**
- 通过相同的管道发送响应消息
- 编码响应并写入网络连接
- 将响应数据发送回客户端

**6. 异常处理阶段（步骤36-42）**
- OnError处理各种网络和协议错误
- OnInactive处理连接关闭事件
- 正确清理连接相关的资源

## 4. 传输管道处理时序图

```mermaid
sequenceDiagram
    autonumber
    participant REQUEST as 请求
    participant PIPELINE as TransPipeline
    participant OUT1 as 出站处理器1
    participant OUT2 as 出站处理器2
    participant OUT3 as 出站处理器3
    participant NET as 网络处理器
    participant NETWORK as 网络层
    participant IN1 as 入站处理器1
    participant IN2 as 入站处理器2
    participant IN3 as 入站处理器3
    participant BUSINESS as 业务处理
    
    Note over REQUEST,BUSINESS: 出站处理流程（发送消息）
    REQUEST->>PIPELINE: Write(ctx, conn, msg)
    
    PIPELINE->>OUT1: Write(ctx, conn, msg)
    OUT1->>OUT1: 日志记录处理
    alt 处理成功
        OUT1->>OUT2: Write(ctx, conn, msg)
        OUT2->>OUT2: 压缩处理
        alt 压缩成功
            OUT2->>OUT3: Write(ctx, conn, msg)
            OUT3->>OUT3: 加密处理
            alt 加密成功
                OUT3->>NET: Write(ctx, conn, msg)
                NET->>NET: 编码消息
                NET->>NETWORK: 发送到网络
                NETWORK-->>NET: 发送完成
                NET-->>OUT3: 返回成功
                OUT3-->>OUT2: 返回成功
                OUT2-->>OUT1: 返回成功
                OUT1-->>PIPELINE: 返回成功
            else 加密失败
                OUT3-->>OUT2: 返回加密错误
                OUT2-->>OUT1: 传播错误
                OUT1-->>PIPELINE: 传播错误
            end
        else 压缩失败
            OUT2-->>OUT1: 返回压缩错误
            OUT1-->>PIPELINE: 传播错误
        end
    else 日志处理失败
        OUT1-->>PIPELINE: 返回日志错误
    end
    
    PIPELINE-->>REQUEST: 返回最终结果
    
    Note over REQUEST,BUSINESS: 入站处理流程（接收消息）
    NETWORK->>NET: OnMessage(ctx, args, result)
    NET->>PIPELINE: OnMessage(ctx, args, result)
    
    PIPELINE->>IN1: OnMessage(ctx, args, result)
    IN1->>IN1: 解密处理
    alt 解密成功
        IN1->>IN2: OnMessage(ctx, args, result)
        IN2->>IN2: 解压缩处理
        alt 解压缩成功
            IN2->>IN3: OnMessage(ctx, args, result)
            IN3->>IN3: 验证处理
            alt 验证成功
                IN3->>BUSINESS: 调用业务逻辑
                BUSINESS->>BUSINESS: 执行业务方法
                BUSINESS-->>IN3: 返回业务结果
                IN3-->>IN2: 返回结果
                IN2-->>IN1: 返回结果
                IN1-->>PIPELINE: 返回结果
            else 验证失败
                IN3-->>IN2: 返回验证错误
                IN2-->>IN1: 传播错误
                IN1-->>PIPELINE: 传播错误
            end
        else 解压缩失败
            IN2-->>IN1: 返回解压缩错误
            IN1-->>PIPELINE: 传播错误
        end
    else 解密失败
        IN1-->>PIPELINE: 返回解密错误
    end
    
    PIPELINE-->>NET: 返回最终结果
```

### 管道处理说明

**1. 出站处理流程（步骤1-23）**
- 请求进入传输管道进行出站处理
- 按顺序执行出站处理器：日志→压缩→加密
- 每个处理器都可能失败并中断后续处理
- 最终由网络处理器完成实际的网络发送
- 错误会逐层向上传播到管道调用方

**2. 入站处理流程（步骤24-47）**
- 网络数据进入管道进行入站处理
- 按顺序执行入站处理器：解密→解压缩→验证
- 处理顺序与出站相反，实现对称处理
- 最终调用业务逻辑处理请求
- 业务结果逐层返回到网络层

**3. 错误处理机制**
- 任何处理器失败都会中断管道执行
- 错误信息包含失败的具体原因和位置
- 支持错误恢复和降级处理策略
- 错误统计和监控便于问题排查

## 5. 连接池管理时序图

```mermaid
sequenceDiagram
    autonumber
    participant CLIENT as 客户端
    participant CONNPOOL as 连接池
    participant POOLMAP as 连接映射
    participant DIALER as 拨号器
    participant CONN as 网络连接
    participant MONITOR as 连接监控
    participant CLEANER as 清理器
    
    Note over CLIENT,CLEANER: 获取连接流程
    CLIENT->>CONNPOOL: Get(ctx, network, address, opt)
    CONNPOOL->>POOLMAP: getPool(address)
    
    alt 地址池不存在
        POOLMAP->>POOLMAP: 创建新的地址池
        POOLMAP-->>CONNPOOL: 返回新池
    else 地址池存在
        POOLMAP-->>CONNPOOL: 返回现有池
    end
    
    CONNPOOL->>POOLMAP: tryGet()尝试获取连接
    
    alt 池中有可用连接
        POOLMAP->>POOLMAP: 检查连接健康状态
        alt 连接健康
            POOLMAP-->>CONNPOOL: 返回可用连接
            CONNPOOL-->>CLIENT: 返回连接
        else 连接不健康
            POOLMAP->>CONN: Close()关闭无效连接
            POOLMAP->>POOLMAP: 继续尝试下一个连接
        end
    else 池中无可用连接
        CONNPOOL->>DIALER: DialContext(ctx, network, address)
        DIALER->>CONN: 建立TCP连接
        
        alt 连接建立成功
            CONN-->>DIALER: 连接建立完成
            DIALER-->>CONNPOOL: 返回新连接
            CONNPOOL->>CONNPOOL: setConnOptions(conn, opt)
            CONNPOOL->>MONITOR: 注册连接监控
            CONNPOOL-->>CLIENT: 返回新连接
        else 连接建立失败
            DIALER-->>CONNPOOL: 返回连接错误
            CONNPOOL-->>CLIENT: 返回错误
        end
    end
    
    Note over CLIENT,CLEANER: 使用连接
    CLIENT->>CONN: 执行RPC调用
    CONN-->>CLIENT: 调用完成
    
    Note over CLIENT,CLEANER: 归还连接流程
    CLIENT->>CONNPOOL: Put(conn, err)
    
    alt 调用有错误
        CONNPOOL->>CONN: Close()关闭连接
        CONNPOOL->>MONITOR: 更新错误统计
    else 调用成功
        CONNPOOL->>CONNPOOL: isConnReusable(conn)
        alt 连接可复用
            CONNPOOL->>POOLMAP: put(conn)放回池中
            CONNPOOL->>MONITOR: 更新成功统计
        else 连接不可复用
            CONNPOOL->>CONN: Close()关闭连接
        end
    end
    
    Note over CLIENT,CLEANER: 后台清理流程
    CLEANER->>POOLMAP: 定期扫描连接池
    loop 所有地址池
        CLEANER->>POOLMAP: 检查空闲连接
        alt 连接空闲超时
            CLEANER->>CONN: Close()关闭超时连接
            CLEANER->>POOLMAP: 从池中移除
        end
        
        alt 池为空且长时间未使用
            CLEANER->>POOLMAP: 删除整个地址池
        end
    end
    
    CLEANER->>MONITOR: 更新清理统计
```

### 连接池管理说明

**1. 连接获取流程（步骤1-21）**
- 客户端请求获取到指定地址的连接
- 根据地址获取对应的连接池，不存在则创建
- 优先从池中获取可用连接，检查连接健康状态
- 池中无可用连接时通过Dialer创建新连接
- 设置连接参数并注册监控

**2. 连接使用阶段（步骤22-24）**
- 客户端使用连接执行RPC调用
- 连接承载实际的网络通信
- 记录连接的使用统计信息

**3. 连接归还流程（步骤25-35）**
- 客户端归还连接到连接池
- 根据调用结果决定连接的处理方式
- 有错误时直接关闭连接
- 无错误时检查连接可复用性，决定是否放回池中

**4. 后台清理流程（步骤36-46）**
- 定期扫描所有连接池
- 清理空闲超时的连接
- 删除长时间未使用的地址池
- 更新清理统计信息

**5. 监控统计**
- 连接创建、复用、关闭的统计
- 连接池大小和使用率监控
- 错误率和性能指标统计
- 支持连接池健康状态报告

## 6. 编解码处理时序图

```mermaid
sequenceDiagram
    autonumber
    participant CLIENT as 客户端
    participant CODEC as Codec
    participant METACODER as MetaCoder
    participant PAYLOADCODEC as PayloadCodec
    participant BUFFER as ByteBuffer
    participant COMPRESSOR as 压缩器
    participant SERVER as 服务端
    
    Note over CLIENT,SERVER: 编码流程
    CLIENT->>CODEC: Encode(ctx, msg, out)
    CODEC->>CODEC: 检查MetaEncoder支持
    
    alt 支持MetaEncoder
        CODEC->>METACODER: EncodeMetaAndPayload(ctx, msg, out, me)
        METACODER->>METACODER: 编码协议头信息
        METACODER->>PAYLOADCODEC: EncodePayload(ctx, msg, out)
        
        alt 需要压缩
            PAYLOADCODEC->>COMPRESSOR: 压缩负载数据
            COMPRESSOR-->>PAYLOADCODEC: 返回压缩结果
        end
        
        PAYLOADCODEC->>BUFFER: 写入负载数据
        PAYLOADCODEC-->>METACODER: 负载编码完成
        METACODER->>BUFFER: 写入完整消息
        METACODER-->>CODEC: 编码完成
    else 不支持MetaEncoder
        CODEC->>CODEC: encodeMeta(ctx, msg, out)
        CODEC->>BUFFER: 写入协议头
        
        CODEC->>PAYLOADCODEC: Marshal(ctx, msg, out)
        
        alt 需要压缩
            PAYLOADCODEC->>COMPRESSOR: 压缩负载数据
            COMPRESSOR-->>PAYLOADCODEC: 返回压缩结果
        end
        
        PAYLOADCODEC->>BUFFER: 写入负载数据
        PAYLOADCODEC-->>CODEC: 编码完成
    end
    
    CODEC-->>CLIENT: 返回编码结果
    CLIENT->>SERVER: 发送编码数据
    
    Note over CLIENT,SERVER: 解码流程
    SERVER->>CODEC: Decode(ctx, msg, in)
    CODEC->>CODEC: 检查MetaDecoder支持
    
    alt 支持MetaDecoder
        CODEC->>METACODER: DecodeMeta(ctx, msg, in)
        METACODER->>BUFFER: 读取协议头信息
        METACODER->>METACODER: 解析协议版本和类型
        METACODER->>METACODER: 设置消息元信息
        METACODER-->>CODEC: 元信息解码完成
        
        CODEC->>PAYLOADCODEC: DecodePayload(ctx, msg, in)
        PAYLOADCODEC->>BUFFER: 读取负载数据
        
        alt 数据被压缩
            PAYLOADCODEC->>COMPRESSOR: 解压缩负载数据
            COMPRESSOR-->>PAYLOADCODEC: 返回解压结果
        end
        
        PAYLOADCODEC->>PAYLOADCODEC: 反序列化业务对象
        PAYLOADCODEC-->>CODEC: 负载解码完成
    else 不支持MetaDecoder
        CODEC->>CODEC: decodeMeta(ctx, msg, in)
        CODEC->>BUFFER: 读取协议头
        CODEC->>CODEC: 解析元信息
        
        CODEC->>PAYLOADCODEC: Unmarshal(ctx, msg, in)
        PAYLOADCODEC->>BUFFER: 读取负载数据
        
        alt 数据被压缩
            PAYLOADCODEC->>COMPRESSOR: 解压缩负载数据
            COMPRESSOR-->>PAYLOADCODEC: 返回解压结果
        end
        
        PAYLOADCODEC->>PAYLOADCODEC: 反序列化业务对象
        PAYLOADCODEC-->>CODEC: 解码完成
    end
    
    CODEC-->>SERVER: 返回解码结果
```

### 编解码处理说明

**1. 编码流程（步骤1-20）**
- 客户端调用Codec进行消息编码
- 检查是否支持MetaEncoder接口
- 支持时使用统一的EncodeMetaAndPayload方法
- 不支持时分别编码元信息和负载
- 根据配置决定是否压缩负载数据
- 将完整的编码结果写入ByteBuffer

**2. 解码流程（步骤21-40）**
- 服务端调用Codec进行消息解码
- 检查是否支持MetaDecoder接口
- 首先解码协议头和元信息
- 然后解码负载数据
- 根据协议头信息决定是否解压缩
- 反序列化得到最终的业务对象

**3. 压缩处理**
- 编码时根据配置和数据大小决定是否压缩
- 支持多种压缩算法（Gzip、Snappy等）
- 解码时根据协议头标识进行相应解压缩
- 压缩可以显著减少网络传输数据量

**4. 协议适配**
- 支持不同协议的元信息格式
- 兼容协议版本升级和降级
- 处理协议特定的编解码逻辑
- 提供协议无关的统一接口

## 时序图总结

这些时序图展示了Remote模块的完整工作流程：

1. **消息生命周期**：从创建到回收的完整过程，展示了对象池的优化机制
2. **客户端发送**：从连接获取到数据发送的完整链路，包含管道处理
3. **服务端接收**：从数据接收到业务处理的完整流程，包含异常处理
4. **传输管道**：出站和入站处理器的执行顺序和错误传播机制
5. **连接池管理**：连接的获取、使用、归还和清理的完整生命周期
6. **编解码处理**：消息序列化和反序列化的详细过程，包含压缩处理

每个时序图都包含了详细的步骤说明和关键节点分析，帮助开发者理解Remote模块的内部工作机制、扩展点和性能优化策略。
