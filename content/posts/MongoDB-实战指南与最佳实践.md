# MongoDB 源码剖析 - 实战指南与最佳实践

## 实战指南总览

基于对MongoDB源码的深入分析，本指南将理论知识转化为实际可操作的最佳实践，帮助开发者在生产环境中更好地运用MongoDB。

## 框架使用示例

### 1. MongoDB C++驱动程序使用示例

```cpp
/**
 * MongoDB C++驱动程序高级使用示例
 * 展示如何正确使用MongoDB的核心API
 */
#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/pool.hpp>
#include <mongocxx/uri.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>

class MongoDBManager {
private:
    mongocxx::instance _instance{};  // 全局初始化
    std::unique_ptr<mongocxx::pool> _pool;
    
public:
    /**
     * 初始化MongoDB连接池
     * 对应源码：src/mongo/client/connection_pool.h
     */
    bool initialize(const std::string& uri_string) {
        try {
            // 配置连接选项
            mongocxx::uri uri{uri_string};
            mongocxx::options::client client_options;
            mongocxx::options::pool pool_options;
            
            // 连接池配置（对应TransportLayer配置）
            pool_options.max_size(100);           // 最大连接数
            pool_options.min_size(10);            // 最小连接数
            pool_options.max_idle_time(std::chrono::minutes(5)); // 空闲超时
            
            // SSL配置（对应SSLManager）
            mongocxx::options::ssl ssl_options;
            ssl_options.allow_invalid_certificates(false);
            ssl_options.ca_file("/path/to/ca.pem");
            ssl_options.certificate_key_file("/path/to/client.pem");
            client_options.ssl_opts(ssl_options);
            
            // 创建连接池
            _pool = std::make_unique<mongocxx::pool>(uri, client_options, pool_options);
            
            // 测试连接
            auto client = _pool->acquire();
            auto ping_result = (*client)["admin"].run_command(
                bsoncxx::builder::stream::document{} << "ping" << 1 
                << bsoncxx::builder::stream::finalize);
                
            return true;
            
        } catch (const std::exception& ex) {
            std::cerr << "MongoDB连接失败: " << ex.what() << std::endl;
            return false;
        }
    }
    
    /**
     * 高效的文档插入操作
     * 对应源码：src/mongo/db/ops/write_ops.cpp
     */
    bool insertDocuments(const std::string& db_name,
                        const std::string& collection_name,
                        const std::vector<bsoncxx::document::view>& documents) {
        try {
            auto client = _pool->acquire();
            auto collection = (*client)[db_name][collection_name];
            
            // 批量插入（对应Collection::insertDocument的批量版本）
            mongocxx::options::insert insert_options;
            insert_options.ordered(false);  // 无序插入，提高并发性能
            
            auto result = collection.insert_many(documents, insert_options);
            
            if (result) {
                std::cout << "插入文档数量: " << result->inserted_count() << std::endl;
                return true;
            }
            
            return false;
            
        } catch (const mongocxx::bulk_write_exception& ex) {
            // 处理批量写入异常
            std::cerr << "批量插入错误: " << ex.what() << std::endl;
            for (const auto& error : ex.write_errors()) {
                std::cerr << "错误索引: " << error.index() 
                         << ", 错误码: " << error.code()
                         << ", 消息: " << error.message() << std::endl;
            }
            return false;
        }
    }
    
    /**
     * 优化的查询操作
     * 对应源码：src/mongo/db/query/query_planner.cpp
     */
    std::vector<bsoncxx::document::value> performOptimizedQuery(
        const std::string& db_name,
        const std::string& collection_name,
        const bsoncxx::document::view& filter,
        const bsoncxx::document::view& sort = {},
        int limit = 0) {
        
        std::vector<bsoncxx::document::value> results;
        
        try {
            auto client = _pool->acquire();
            auto collection = (*client)[db_name][collection_name];
            
            // 构建查询选项
            mongocxx::options::find find_options;
            
            if (!sort.empty()) {
                find_options.sort(sort);
            }
            
            if (limit > 0) {
                find_options.limit(limit);
            }
            
            // 设置读偏好（对应源码中的ReadPreference）
            mongocxx::read_preference rp;
            rp.mode(mongocxx::read_preference::read_mode::k_secondary_preferred);
            find_options.read_preference(rp);
            
            // 启用查询计划缓存
            find_options.allow_disk_use(true);
            
            // 执行查询
            auto cursor = collection.find(filter, find_options);
            
            for (auto&& doc : cursor) {
                results.push_back(bsoncxx::document::value{doc});
            }
            
        } catch (const std::exception& ex) {
            std::cerr << "查询执行失败: " << ex.what() << std::endl;
        }
        
        return results;
    }
    
    /**
     * 事务处理示例
     * 对应源码：src/mongo/db/transaction/transaction_participant.cpp
     */
    bool performTransaction(std::function<bool(mongocxx::client_session&)> transaction_func) {
        try {
            auto client = _pool->acquire();
            auto session = client->start_session();
            
            // 事务选项配置
            mongocxx::options::transaction transaction_options;
            transaction_options.write_concern(mongocxx::write_concern{});
            transaction_options.read_concern(mongocxx::read_concern{});
            transaction_options.read_preference(mongocxx::read_preference{});
            
            // 执行事务
            session.with_transaction([&](mongocxx::client_session* session) {
                return transaction_func(*session);
            }, transaction_options);
            
            return true;
            
        } catch (const std::exception& ex) {
            std::cerr << "事务执行失败: " << ex.what() << std::endl;
            return false;
        }
    }
};

// 使用示例
int main() {
    MongoDBManager mongo_mgr;
    
    // 1. 初始化连接
    if (!mongo_mgr.initialize("mongodb://localhost:27017/?ssl=true&replicaSet=rs0")) {
        return 1;
    }
    
    // 2. 批量插入示例
    std::vector<bsoncxx::document::value> documents;
    for (int i = 0; i < 1000; ++i) {
        documents.push_back(
            bsoncxx::builder::stream::document{}
            << "_id" << bsoncxx::oid{}
            << "user_id" << i
            << "name" << ("User " + std::to_string(i))
            << "email" << ("user" + std::to_string(i) + "@example.com")
            << "created_at" << bsoncxx::types::b_date{std::chrono::system_clock::now()}
            << "status" << "active"
            << bsoncxx::builder::stream::finalize
        );
    }
    
    mongo_mgr.insertDocuments("myapp", "users", 
        std::vector<bsoncxx::document::view>(documents.begin(), documents.end()));
    
    // 3. 优化查询示例
    auto filter = bsoncxx::builder::stream::document{}
        << "status" << "active"
        << "user_id" << bsoncxx::builder::stream::open_document
            << "$gte" << 100
            << "$lte" << 200
        << bsoncxx::builder::stream::close_document
        << bsoncxx::builder::stream::finalize;
        
    auto sort = bsoncxx::builder::stream::document{}
        << "created_at" << -1
        << bsoncxx::builder::stream::finalize;
    
    auto results = mongo_mgr.performOptimizedQuery("myapp", "users", 
                                                  filter.view(), sort.view(), 50);
    
    std::cout << "查询到 " << results.size() << " 个结果" << std::endl;
    
    return 0;
}
```

### 2. JavaScript/Node.js驱动程序使用示例

```javascript
/**
 * MongoDB Node.js驱动程序高级使用示例
 * 展示生产环境最佳实践
 */
const { MongoClient, GridFSBucket } = require('mongodb');

class MongoDBService {
    constructor() {
        this.client = null;
        this.db = null;
    }
    
    /**
     * 连接初始化（对应Grid::init）
     */
    async initialize() {
        try {
            // 生产环境连接配置
            const uri = process.env.MONGODB_URI || 'mongodb://localhost:27017';
            const options = {
                // 连接池配置（对应ConnectionPool）
                maxPoolSize: 100,
                minPoolSize: 5,
                maxIdleTimeMS: 300000,  // 5分钟
                
                // 连接超时配置
                connectTimeoutMS: 10000,
                socketTimeoutMS: 45000,
                
                // 重试配置
                retryWrites: true,
                retryReads: true,
                
                // 读写关注配置
                readPreference: 'secondaryPreferred',
                writeConcern: {
                    w: 'majority',
                    j: true,
                    wtimeout: 10000
                },
                
                // 压缩配置（对应MessageCompressor）
                compressors: ['snappy', 'zlib'],
                
                // SSL配置
                tls: true,
                tlsAllowInvalidCertificates: false,
                tlsCAFile: './certs/ca.pem',
                tlsCertificateKeyFile: './certs/client.pem'
            };
            
            this.client = new MongoClient(uri, options);
            await this.client.connect();
            
            // 验证连接
            await this.client.db('admin').admin().ping();
            console.log('MongoDB连接成功');
            
            this.db = this.client.db(process.env.DB_NAME || 'myapp');
            
            // 创建索引
            await this.createIndexes();
            
        } catch (error) {
            console.error('MongoDB连接失败:', error);
            throw error;
        }
    }
    
    /**
     * 创建优化索引（对应IndexCatalog）
     */
    async createIndexes() {
        try {
            const collections = {
                users: [
                    // 复合索引，遵循ESR规则
                    { key: { status: 1, email: 1, createdAt: -1 } },
                    // 部分索引，减少存储开销
                    { 
                        key: { phoneNumber: 1 }, 
                        options: { 
                            sparse: true,
                            partialFilterExpression: { phoneNumber: { $exists: true } }
                        }
                    },
                    // 哈希索引，适合等值查询
                    { key: { userId: 'hashed' } }
                ],
                orders: [
                    // 复合索引支持多种查询模式
                    { key: { customerId: 1, status: 1, createdAt: -1 } },
                    { key: { orderId: 1 }, options: { unique: true } },
                    // 地理空间索引
                    { key: { location: '2dsphere' } }
                ],
                logs: [
                    // TTL索引，自动清理过期数据
                    { key: { createdAt: 1 }, options: { expireAfterSeconds: 86400 } }
                ]
            };
            
            for (const [collName, indexes] of Object.entries(collections)) {
                const collection = this.db.collection(collName);
                for (const index of indexes) {
                    await collection.createIndex(index.key, index.options || {});
                }
                console.log(`${collName}集合索引创建完成`);
            }
            
        } catch (error) {
            console.error('索引创建失败:', error);
        }
    }
    
    /**
     * 高性能批量写入（对应Collection::insertDocument批量版本）
     */
    async bulkWrite(collectionName, operations, options = {}) {
        try {
            const collection = this.db.collection(collectionName);
            
            // 批量写入配置
            const bulkOptions = {
                ordered: false,        // 无序执行，提高性能
                writeConcern: {
                    w: 'majority',
                    j: true,
                    wtimeout: 10000
                },
                ...options
            };
            
            const result = await collection.bulkWrite(operations, bulkOptions);
            
            return {
                success: true,
                insertedCount: result.insertedCount,
                modifiedCount: result.modifiedCount,
                deletedCount: result.deletedCount,
                upsertedCount: result.upsertedCount
            };
            
        } catch (error) {
            console.error('批量写入失败:', error);
            return { success: false, error: error.message };
        }
    }
    
    /**
     * 聚合查询优化（对应ClusterPipeline）
     */
    async performAggregation(collectionName, pipeline, options = {}) {
        try {
            const collection = this.db.collection(collectionName);
            
            // 聚合选项优化
            const aggOptions = {
                allowDiskUse: true,    // 允许使用磁盘，处理大数据集
                batchSize: 1000,       // 批量大小优化
                readPreference: 'secondaryPreferred',  // 从副本读取
                ...options
            };
            
            // 在管道开始添加索引提示
            if (options.hint) {
                pipeline.unshift({ $hint: options.hint });
            }
            
            const cursor = collection.aggregate(pipeline, aggOptions);
            const results = await cursor.toArray();
            
            return results;
            
        } catch (error) {
            console.error('聚合查询失败:', error);
            throw error;
        }
    }
    
    /**
     * 事务处理（对应TransactionParticipant）
     */
    async withTransaction(operations) {
        const session = this.client.startSession();
        
        try {
            await session.withTransaction(async () => {
                for (const operation of operations) {
                    await operation(session);
                }
            }, {
                readPreference: 'primary',
                readConcern: { level: 'snapshot' },
                writeConcern: { w: 'majority', j: true }
            });
            
            return { success: true };
            
        } catch (error) {
            console.error('事务执行失败:', error);
            return { success: false, error: error.message };
        } finally {
            await session.endSession();
        }
    }
    
    /**
     * 大文件存储（GridFS）
     */
    async storeFile(filename, fileStream, metadata = {}) {
        try {
            const bucket = new GridFSBucket(this.db);
            
            return new Promise((resolve, reject) => {
                const uploadStream = bucket.openUploadStream(filename, {
                    metadata: metadata
                });
                
                uploadStream.on('error', reject);
                uploadStream.on('finish', (file) => {
                    resolve({ fileId: file._id, filename: file.filename });
                });
                
                fileStream.pipe(uploadStream);
            });
            
        } catch (error) {
            console.error('文件存储失败:', error);
            throw error;
        }
    }
    
    /**
     * 性能监控和健康检查
     */
    async getHealthStatus() {
        try {
            const admin = this.db.admin();
            
            // 服务器状态
            const serverStatus = await admin.serverStatus();
            
            // 数据库统计
            const dbStats = await this.db.stats();
            
            // 连接信息
            const connectionStatus = await admin.command({ connectionStatus: 1 });
            
            return {
                serverInfo: {
                    version: serverStatus.version,
                    uptime: serverStatus.uptime,
                    connections: serverStatus.connections
                },
                database: {
                    collections: dbStats.collections,
                    dataSize: dbStats.dataSize,
                    indexSize: dbStats.indexSize
                },
                connection: {
                    user: connectionStatus.authInfo.authenticatedUsers[0]?.user,
                    roles: connectionStatus.authInfo.authenticatedUserRoles
                }
            };
            
        } catch (error) {
            console.error('健康检查失败:', error);
            return { error: error.message };
        }
    }
    
    /**
     * 优雅关闭连接
     */
    async close() {
        if (this.client) {
            await this.client.close();
            console.log('MongoDB连接已关闭');
        }
    }
}

// 使用示例
async function main() {
    const mongoService = new MongoDBService();
    
    try {
        // 初始化连接
        await mongoService.initialize();
        
        // 1. 批量写入示例
        const operations = [];
        for (let i = 0; i < 10000; i++) {
            operations.push({
                insertOne: {
                    document: {
                        userId: i,
                        name: `User ${i}`,
                        email: `user${i}@example.com`,
                        status: i % 2 === 0 ? 'active' : 'inactive',
                        createdAt: new Date(),
                        metadata: {
                            source: 'batch_import',
                            version: 1
                        }
                    }
                }
            });
        }
        
        const writeResult = await mongoService.bulkWrite('users', operations);
        console.log('批量写入结果:', writeResult);
        
        // 2. 聚合查询示例
        const pipeline = [
            { $match: { status: 'active' } },
            { $group: {
                _id: '$status',
                count: { $sum: 1 },
                avgUserId: { $avg: '$userId' }
            }},
            { $sort: { count: -1 } }
        ];
        
        const aggResult = await mongoService.performAggregation('users', pipeline);
        console.log('聚合查询结果:', aggResult);
        
        // 3. 事务示例
        const transactionResult = await mongoService.withTransaction([
            async (session) => {
                await mongoService.db.collection('users').updateOne(
                    { userId: 1 },
                    { $set: { lastActive: new Date() } },
                    { session }
                );
            },
            async (session) => {
                await mongoService.db.collection('logs').insertOne({
                    userId: 1,
                    action: 'login',
                    timestamp: new Date()
                }, { session });
            }
        ]);
        
        console.log('事务执行结果:', transactionResult);
        
        // 4. 健康检查
        const health = await mongoService.getHealthStatus();
        console.log('系统健康状态:', health);
        
    } catch (error) {
        console.error('操作失败:', error);
    } finally {
        await mongoService.close();
    }
}

// 错误处理和进程信号监听
process.on('SIGINT', async () => {
    console.log('收到SIGINT信号，正在关闭连接...');
    if (global.mongoService) {
        await global.mongoService.close();
    }
    process.exit(0);
});

if (require.main === module) {
    main();
}

module.exports = MongoDBService;
```

## 生产环境部署最佳实践

### 1. 副本集部署配置

```yaml
# docker-compose.yml - MongoDB副本集部署
version: '3.8'
services:
  mongo-primary:
    image: mongo:7.0
    container_name: mongo-primary
    ports:
      - "27017:27017"
    volumes:
      - mongo-primary-data:/data/db
      - ./mongo-keyfile:/data/keyfile:ro
      - ./ssl:/data/ssl:ro
    command: >
      mongod --replSet rs0 
             --bind_ip_all 
             --auth 
             --keyFile /data/keyfile
             --sslMode requireSSL
             --sslPEMKeyFile /data/ssl/mongodb.pem
             --sslCAFile /data/ssl/ca.pem
             --oplogSize 1024
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: secretpassword
    networks:
      - mongo-network
    
  mongo-secondary1:
    image: mongo:7.0
    container_name: mongo-secondary1
    ports:
      - "27018:27017"
    volumes:
      - mongo-secondary1-data:/data/db
      - ./mongo-keyfile:/data/keyfile:ro
      - ./ssl:/data/ssl:ro
    command: >
      mongod --replSet rs0 
             --bind_ip_all 
             --auth 
             --keyFile /data/keyfile
             --sslMode requireSSL
             --sslPEMKeyFile /data/ssl/mongodb.pem
             --sslCAFile /data/ssl/ca.pem
             --oplogSize 1024
    depends_on:
      - mongo-primary
    networks:
      - mongo-network
      
  mongo-secondary2:
    image: mongo:7.0
    container_name: mongo-secondary2
    ports:
      - "27019:27017"
    volumes:
      - mongo-secondary2-data:/data/db
      - ./mongo-keyfile:/data/keyfile:ro
      - ./ssl:/data/ssl:ro
    command: >
      mongod --replSet rs0 
             --bind_ip_all 
             --auth 
             --keyFile /data/keyfile
             --sslMode requireSSL
             --sslPEMKeyFile /data/ssl/mongodb.pem
             --sslCAFile /data/ssl/ca.pem
             --oplogSize 1024
    depends_on:
      - mongo-primary
    networks:
      - mongo-network

volumes:
  mongo-primary-data:
  mongo-secondary1-data:
  mongo-secondary2-data:

networks:
  mongo-network:
    driver: bridge
```

```bash
#!/bin/bash
# 副本集初始化脚本

# 生成keyfile用于内部认证
openssl rand -base64 756 > mongo-keyfile
chmod 400 mongo-keyfile

# 启动服务
docker-compose up -d

# 等待服务启动
sleep 30

# 初始化副本集
docker exec -it mongo-primary mongosh --eval "
rs.initiate({
  _id: 'rs0',
  members: [
    { _id: 0, host: 'mongo-primary:27017', priority: 2 },
    { _id: 1, host: 'mongo-secondary1:27017', priority: 1 },
    { _id: 2, host: 'mongo-secondary2:27017', priority: 1 }
  ]
});

// 创建管理员用户
db = db.getSiblingDB('admin');
db.createUser({
  user: 'admin',
  pwd: 'secretpassword',
  roles: ['root']
});

// 创建应用用户
db = db.getSiblingDB('myapp');
db.createUser({
  user: 'appuser',
  pwd: 'apppassword',
  roles: ['readWrite']
});
"

echo "副本集初始化完成"
```

### 2. 分片集群部署配置

```yaml
# 分片集群部署配置
version: '3.8'
services:
  # 配置服务器副本集
  config1:
    image: mongo:7.0
    command: mongod --configsvr --replSet configReplSet --port 27017 --dbpath /data/db
    volumes:
      - config1:/data/db
    networks:
      - mongo-shard-network

  config2:
    image: mongo:7.0
    command: mongod --configsvr --replSet configReplSet --port 27017 --dbpath /data/db
    volumes:
      - config2:/data/db
    networks:
      - mongo-shard-network

  config3:
    image: mongo:7.0
    command: mongod --configsvr --replSet configReplSet --port 27017 --dbpath /data/db
    volumes:
      - config3:/data/db
    networks:
      - mongo-shard-network

  # 分片1副本集
  shard1-primary:
    image: mongo:7.0
    command: mongod --shardsvr --replSet shard1ReplSet --port 27017 --dbpath /data/db
    volumes:
      - shard1-primary:/data/db
    networks:
      - mongo-shard-network

  shard1-secondary:
    image: mongo:7.0
    command: mongod --shardsvr --replSet shard1ReplSet --port 27017 --dbpath /data/db
    volumes:
      - shard1-secondary:/data/db
    networks:
      - mongo-shard-network

  # 分片2副本集
  shard2-primary:
    image: mongo:7.0
    command: mongod --shardsvr --replSet shard2ReplSet --port 27017 --dbpath /data/db
    volumes:
      - shard2-primary:/data/db
    networks:
      - mongo-shard-network

  shard2-secondary:
    image: mongo:7.0
    command: mongod --shardsvr --replSet shard2ReplSet --port 27017 --dbpath /data/db
    volumes:
      - shard2-secondary:/data/db
    networks:
      - mongo-shard-network

  # mongos路由器
  mongos1:
    image: mongo:7.0
    command: mongos --configdb configReplSet/config1:27017,config2:27017,config3:27017 --port 27017
    ports:
      - "27017:27017"
    depends_on:
      - config1
      - config2  
      - config3
      - shard1-primary
      - shard2-primary
    networks:
      - mongo-shard-network

  mongos2:
    image: mongo:7.0
    command: mongos --configdb configReplSet/config1:27017,config2:27017,config3:27017 --port 27017
    ports:
      - "27018:27017"
    depends_on:
      - config1
      - config2
      - config3
      - shard1-primary
      - shard2-primary
    networks:
      - mongo-shard-network

volumes:
  config1:
  config2:
  config3:
  shard1-primary:
  shard1-secondary:
  shard2-primary:
  shard2-secondary:

networks:
  mongo-shard-network:
    driver: bridge
```

### 3. 监控和告警配置

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  # MongoDB导出器，用于Prometheus监控
  mongodb-exporter:
    image: percona/mongodb_exporter:0.40
    ports:
      - "9216:9216"
    environment:
      - MONGODB_URI=mongodb://monitor:password@mongo-primary:27017
    command:
      - '--mongodb.uri=mongodb://monitor:password@mongo-primary:27017'
      - '--mongodb.collstats-colls=myapp.users,myapp.orders'
      - '--mongodb.indexstats-colls=myapp.users,myapp.orders'
      - '--web.listen-address=:9216'
    networks:
      - mongo-network
      - monitoring-network

  # Prometheus监控服务
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    networks:
      - monitoring-network

  # Grafana可视化服务
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - monitoring-network

  # AlertManager告警服务
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    networks:
      - monitoring-network

volumes:
  prometheus-data:
  grafana-data:

networks:
  monitoring-network:
    driver: bridge
  mongo-network:
    external: true
```

```yaml
# prometheus.yml - Prometheus配置
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "mongodb-rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb-exporter:9216']
    scrape_interval: 30s
    metrics_path: /metrics
```

## 性能调优实战

### 1. 索引优化策略

```javascript
/**
 * 索引性能分析和优化工具
 */
class IndexOptimizer {
    constructor(db) {
        this.db = db;
    }
    
    /**
     * 分析集合的索引使用情况
     */
    async analyzeIndexUsage(collectionName) {
        const collection = this.db.collection(collectionName);
        
        // 获取索引统计信息
        const indexStats = await collection.aggregate([
            { $indexStats: {} }
        ]).toArray();
        
        // 获取当前索引
        const indexes = await collection.indexes();
        
        // 分析结果
        const analysis = {
            totalIndexes: indexes.length,
            unusedIndexes: [],
            lowUsageIndexes: [],
            recommendations: []
        };
        
        for (const stat of indexStats) {
            const usage = stat.accesses.ops;
            const indexName = stat.name;
            
            if (usage === 0) {
                analysis.unusedIndexes.push(indexName);
                analysis.recommendations.push(
                    `考虑删除未使用的索引: ${indexName}`
                );
            } else if (usage < 100) {
                analysis.lowUsageIndexes.push({
                    name: indexName,
                    usage: usage
                });
            }
        }
        
        return analysis;
    }
    
    /**
     * 基于查询模式推荐索引
     */
    async recommendIndexes(collectionName, queryPatterns) {
        const recommendations = [];
        
        for (const pattern of queryPatterns) {
            const explanation = await this.explainQuery(collectionName, pattern);
            
            if (explanation.executionStats.executionSuccess && 
                explanation.executionStats.totalDocsExamined > 
                explanation.executionStats.totalDocsReturned * 10) {
                
                // 如果扫描的文档数远大于返回的文档数，建议添加索引
                const fields = this.extractIndexFields(pattern);
                recommendations.push({
                    query: pattern,
                    suggestedIndex: fields,
                    reason: '查询效率低，建议添加复合索引'
                });
            }
        }
        
        return recommendations;
    }
    
    /**
     * 查询执行计划分析
     */
    async explainQuery(collectionName, query, options = {}) {
        const collection = this.db.collection(collectionName);
        return await collection.find(query, options).explain('executionStats');
    }
    
    /**
     * 从查询模式中提取索引字段
     */
    extractIndexFields(query) {
        const fields = {};
        
        // 简化的字段提取逻辑
        for (const [key, value] of Object.entries(query)) {
            if (key.startsWith('$')) continue;
            
            if (typeof value === 'object' && value !== null) {
                if (value.$gt !== undefined || value.$gte !== undefined ||
                    value.$lt !== undefined || value.$lte !== undefined) {
                    fields[key] = 1; // 范围查询
                } else if (value.$in !== undefined) {
                    fields[key] = 1; // IN查询
                } else {
                    fields[key] = 1; // 等值查询
                }
            } else {
                fields[key] = 1; // 等值查询
            }
        }
        
        return fields;
    }
    
    /**
     * 创建优化的复合索引
     */
    async createOptimizedIndex(collectionName, fields, options = {}) {
        const collection = this.db.collection(collectionName);
        
        // 按照ESR规则排序字段：Equality, Sort, Range
        const sortedFields = this.sortFieldsByESR(fields);
        
        const indexOptions = {
            background: true,  // 后台创建，不阻塞数据库操作
            ...options
        };
        
        try {
            const result = await collection.createIndex(sortedFields, indexOptions);
            return { success: true, indexName: result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    /**
     * 按ESR规则排序字段
     */
    sortFieldsByESR(fields) {
        // 简化实现：相等性查询优先，然后是排序，最后是范围
        const equality = {};
        const sort = {};
        const range = {};
        
        for (const [field, direction] of Object.entries(fields)) {
            // 这里需要根据实际的查询模式来分类
            // 简化处理：假设都是相等性查询
            equality[field] = direction;
        }
        
        return { ...equality, ...sort, ...range };
    }
}

// 使用示例
async function optimizeIndexes() {
    const { MongoClient } = require('mongodb');
    const client = new MongoClient('mongodb://localhost:27017');
    await client.connect();
    const db = client.db('myapp');
    
    const optimizer = new IndexOptimizer(db);
    
    // 分析现有索引使用情况
    const analysis = await optimizer.analyzeIndexUsage('users');
    console.log('索引使用分析:', analysis);
    
    // 基于查询模式推荐索引
    const queryPatterns = [
        { status: 'active', age: { $gte: 18 } },
        { email: 'user@example.com' },
        { createdAt: { $gte: new Date('2023-01-01') } }
    ];
    
    const recommendations = await optimizer.recommendIndexes('users', queryPatterns);
    console.log('索引推荐:', recommendations);
    
    // 创建优化索引
    for (const rec of recommendations) {
        const result = await optimizer.createOptimizedIndex('users', rec.suggestedIndex);
        console.log('创建索引结果:', result);
    }
    
    await client.close();
}
```

### 2. 查询性能优化

```javascript
/**
 * 查询性能优化工具类
 */
class QueryOptimizer {
    constructor(db) {
        this.db = db;
        this.slowQueryThreshold = 100; // 慢查询阈值：100ms
    }
    
    /**
     * 启用查询性能分析
     */
    async enableProfiling(level = 2, slowms = 100) {
        try {
            await this.db.admin().profiling.setLevel(level, { slowms });
            console.log(`查询分析已启用，慢查询阈值: ${slowms}ms`);
        } catch (error) {
            console.error('启用查询分析失败:', error);
        }
    }
    
    /**
     * 分析慢查询
     */
    async analyzeSlowQueries(limit = 10) {
        try {
            const profilerCollection = this.db.collection('system.profile');
            
            const slowQueries = await profilerCollection.find({
                op: { $in: ['query', 'update', 'remove'] },
                millis: { $gte: this.slowQueryThreshold }
            })
            .sort({ ts: -1 })
            .limit(limit)
            .toArray();
            
            const analysis = slowQueries.map(query => ({
                operation: query.op,
                namespace: query.ns,
                duration: query.millis,
                command: query.command,
                executionStats: query.execStats,
                timestamp: query.ts,
                suggestion: this.generateOptimizationSuggestion(query)
            }));
            
            return analysis;
            
        } catch (error) {
            console.error('慢查询分析失败:', error);
            return [];
        }
    }
    
    /**
     * 生成优化建议
     */
    generateOptimizationSuggestion(query) {
        const suggestions = [];
        
        // 检查是否使用了索引
        if (query.execStats && query.execStats.stage === 'COLLSCAN') {
            suggestions.push('查询进行了全表扫描，建议添加索引');
        }
        
        // 检查扫描效率
        if (query.execStats) {
            const examined = query.execStats.totalDocsExamined || 0;
            const returned = query.execStats.totalDocsReturned || 0;
            
            if (examined > returned * 10) {
                suggestions.push('查询效率低，扫描了过多文档，建议优化查询条件或索引');
            }
        }
        
        // 检查排序操作
        if (query.command && query.command.sort && query.execStats.stage !== 'IXSCAN') {
            suggestions.push('排序操作未使用索引，建议创建支持排序的索引');
        }
        
        return suggestions;
    }
    
    /**
     * 查询执行计划对比
     */
    async compareQueryPlans(collectionName, query, indexes) {
        const collection = this.db.collection(collectionName);
        const results = [];
        
        for (const index of indexes) {
            try {
                // 创建临时索引
                await collection.createIndex(index, { background: true });
                
                // 执行查询并获取执行计划
                const explanation = await collection.find(query).explain('executionStats');
                
                results.push({
                    index: index,
                    executionTime: explanation.executionStats.executionTimeMillis,
                    docsExamined: explanation.executionStats.totalDocsExamined,
                    docsReturned: explanation.executionStats.totalDocsReturned,
                    efficiency: explanation.executionStats.totalDocsReturned / 
                               (explanation.executionStats.totalDocsExamined || 1)
                });
                
            } catch (error) {
                console.error(`索引测试失败 ${JSON.stringify(index)}:`, error);
            }
        }
        
        // 按执行时间排序
        results.sort((a, b) => a.executionTime - b.executionTime);
        
        return results;
    }
    
    /**
     * 聚合管道优化
     */
    optimizeAggregationPipeline(pipeline) {
        const optimized = [...pipeline];
        
        // 优化策略1：将$match操作移到管道开始
        const matchStages = [];
        const otherStages = [];
        
        for (const stage of optimized) {
            if (stage.$match) {
                matchStages.push(stage);
            } else {
                otherStages.push(stage);
            }
        }
        
        // 优化策略2：合并相邻的$match操作
        const combinedMatch = matchStages.reduce((combined, stage) => {
            return { ...combined, ...stage.$match };
        }, {});
        
        const finalPipeline = [];
        if (Object.keys(combinedMatch).length > 0) {
            finalPipeline.push({ $match: combinedMatch });
        }
        finalPipeline.push(...otherStages);
        
        return {
            original: pipeline,
            optimized: finalPipeline,
            optimizations: [
                'Match操作已移至管道开始',
                '相邻的Match操作已合并'
            ]
        };
    }
    
    /**
     * 查询结果缓存
     */
    async getCachedQuery(cacheKey, queryFunction, ttl = 300) {
        const cacheCollection = this.db.collection('query_cache');
        
        // 尝试从缓存获取
        const cached = await cacheCollection.findOne({
            key: cacheKey,
            expiredAt: { $gt: new Date() }
        });
        
        if (cached) {
            return { data: cached.data, fromCache: true };
        }
        
        // 执行查询
        const result = await queryFunction();
        
        // 存储到缓存
        await cacheCollection.replaceOne(
            { key: cacheKey },
            {
                key: cacheKey,
                data: result,
                createdAt: new Date(),
                expiredAt: new Date(Date.now() + ttl * 1000)
            },
            { upsert: true }
        );
        
        return { data: result, fromCache: false };
    }
}

// 使用示例
async function performQueryOptimization() {
    const { MongoClient } = require('mongodb');
    const client = new MongoClient('mongodb://localhost:27017');
    await client.connect();
    const db = client.db('myapp');
    
    const optimizer = new QueryOptimizer(db);
    
    // 1. 启用查询分析
    await optimizer.enableProfiling(2, 100);
    
    // 等待一段时间收集查询数据
    console.log('等待收集查询数据...');
    setTimeout(async () => {
        // 2. 分析慢查询
        const slowQueries = await optimizer.analyzeSlowQueries(10);
        console.log('慢查询分析结果:', JSON.stringify(slowQueries, null, 2));
        
        // 3. 测试不同索引的性能
        const testIndexes = [
            { status: 1 },
            { status: 1, createdAt: -1 },
            { status: 1, email: 1, createdAt: -1 }
        ];
        
        const comparison = await optimizer.compareQueryPlans('users', 
            { status: 'active', email: /gmail\.com$/ }, testIndexes);
        console.log('索引性能对比:', comparison);
        
        // 4. 优化聚合管道
        const originalPipeline = [
            { $project: { name: 1, email: 1, status: 1 } },
            { $match: { status: 'active' } },
            { $match: { email: /gmail\.com$/ } },
            { $sort: { name: 1 } }
        ];
        
        const optimizationResult = optimizer.optimizeAggregationPipeline(originalPipeline);
        console.log('管道优化结果:', optimizationResult);
        
        await client.close();
    }, 5000);
}
```

## 安全加固实践

### 1. 认证和授权配置

```javascript
/**
 * MongoDB安全配置最佳实践
 */
class SecurityManager {
    constructor(adminDb) {
        this.adminDb = adminDb;
    }
    
    /**
     * 创建安全的用户角色体系
     */
    async setupSecurityRoles() {
        try {
            // 创建自定义角色：应用只读角色
            await this.adminDb.runCommand({
                createRole: 'appReadOnly',
                privileges: [
                    {
                        resource: { db: 'myapp', collection: '' },
                        actions: ['find', 'listIndexes', 'listCollections']
                    }
                ],
                roles: []
            });
            
            // 创建自定义角色：应用读写角色
            await this.adminDb.runCommand({
                createRole: 'appReadWrite',
                privileges: [
                    {
                        resource: { db: 'myapp', collection: '' },
                        actions: [
                            'find', 'insert', 'update', 'remove',
                            'createIndex', 'listIndexes', 'listCollections'
                        ]
                    }
                ],
                roles: ['appReadOnly']
            });
            
            // 创建自定义角色：应用管理员角色
            await this.adminDb.runCommand({
                createRole: 'appAdmin',
                privileges: [
                    {
                        resource: { db: 'myapp', collection: '' },
                        actions: ['*']
                    }
                ],
                roles: ['appReadWrite', 'dbAdmin']
            });
            
            console.log('安全角色创建完成');
            
        } catch (error) {
            console.error('角色创建失败:', error);
        }
    }
    
    /**
     * 创建应用用户
     */
    async createApplicationUsers() {
        try {
            const users = [
                {
                    user: 'app-read',
                    pwd: this.generateSecurePassword(),
                    roles: ['appReadOnly']
                },
                {
                    user: 'app-write', 
                    pwd: this.generateSecurePassword(),
                    roles: ['appReadWrite']
                },
                {
                    user: 'app-admin',
                    pwd: this.generateSecurePassword(),
                    roles: ['appAdmin']
                }
            ];
            
            for (const user of users) {
                await this.adminDb.runCommand({
                    createUser: user.user,
                    pwd: user.pwd,
                    roles: user.roles
                });
                
                console.log(`用户创建成功: ${user.user}`);
                console.log(`密码: ${user.pwd}`);
            }
            
        } catch (error) {
            console.error('用户创建失败:', error);
        }
    }
    
    /**
     * 生成安全密码
     */
    generateSecurePassword(length = 24) {
        const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*';
        let password = '';
        for (let i = 0; i < length; i++) {
            password += charset.charAt(Math.floor(Math.random() * charset.length));
        }
        return password;
    }
    
    /**
     * 配置网络访问控制
     */
    async configureNetworkSecurity() {
        // 在MongoDB配置文件中设置
        const securityConfig = {
            net: {
                bindIp: '127.0.0.1,10.0.0.0/8', // 限制绑定IP
                port: 27017,
                ssl: {
                    mode: 'requireSSL',
                    PEMKeyFile: '/etc/ssl/mongodb.pem',
                    CAFile: '/etc/ssl/ca.pem',
                    allowInvalidCertificates: false,
                    allowInvalidHostnames: false
                }
            },
            security: {
                authorization: 'enabled',
                keyFile: '/etc/mongodb/keyfile',
                clusterAuthMode: 'x509'
            }
        };
        
        console.log('网络安全配置:', JSON.stringify(securityConfig, null, 2));
        return securityConfig;
    }
    
    /**
     * 审计配置
     */
    async setupAuditing() {
        const auditConfig = {
            auditLog: {
                destination: 'file',
                format: 'JSON',
                path: '/var/log/mongodb/audit.json',
                filter: {
                    atype: {
                        $in: [
                            'authenticate', 'authCheck', 'createUser',
                            'dropUser', 'createRole', 'dropRole',
                            'createIndex', 'dropIndex', 'createCollection',
                            'dropCollection', 'insert', 'update', 'remove'
                        ]
                    }
                }
            }
        };
        
        console.log('审计配置:', JSON.stringify(auditConfig, null, 2));
        return auditConfig;
    }
}
```

### 2. 数据加密配置

```bash
#!/bin/bash
# MongoDB数据加密配置脚本

# 1. 生成加密密钥
echo "生成主密钥..."
openssl rand -base64 32 > /etc/mongodb/encryption-keyfile
chmod 600 /etc/mongodb/encryption-keyfile
chown mongodb:mongodb /etc/mongodb/encryption-keyfile

# 2. 生成SSL证书
echo "生成SSL证书..."
mkdir -p /etc/mongodb/ssl

# 创建CA私钥和证书
openssl genrsa -out /etc/mongodb/ssl/ca-key.pem 4096
openssl req -new -x509 -days 365 -key /etc/mongodb/ssl/ca-key.pem \
    -out /etc/mongodb/ssl/ca.pem \
    -subj "/CN=MongoDB-CA"

# 创建服务器证书
openssl genrsa -out /etc/mongodb/ssl/server-key.pem 4096
openssl req -new -key /etc/mongodb/ssl/server-key.pem \
    -out /etc/mongodb/ssl/server.csr \
    -subj "/CN=mongodb.example.com"

# 用CA签名服务器证书
openssl x509 -req -days 365 -in /etc/mongodb/ssl/server.csr \
    -CA /etc/mongodb/ssl/ca.pem \
    -CAkey /etc/mongodb/ssl/ca-key.pem \
    -CAcreateserial \
    -out /etc/mongodb/ssl/server.pem

# 合并服务器证书和私钥
cat /etc/mongodb/ssl/server.pem /etc/mongodb/ssl/server-key.pem > \
    /etc/mongodb/ssl/mongodb.pem

# 设置权限
chmod 600 /etc/mongodb/ssl/*
chown -R mongodb:mongodb /etc/mongodb/ssl/

echo "SSL证书生成完成"
```

```yaml
# MongoDB加密配置文件
# /etc/mongod.conf
systemLog:
  destination: file
  logAppend: true
  path: /var/log/mongodb/mongod.log
  logRotate: reopen

storage:
  dbPath: /var/lib/mongodb
  journal:
    enabled: true
  engine: wiredTiger
  wiredTiger:
    engineConfig:
      journalCompressor: snappy
      directoryForIndexes: false
    collectionConfig:
      blockCompressor: snappy
    indexConfig:
      prefixCompression: true
  # 静态数据加密
  encryptionKeyFile: /etc/mongodb/encryption-keyfile

processManagement:
  fork: true
  pidFilePath: /var/run/mongodb/mongod.pid

net:
  port: 27017
  bindIp: 127.0.0.1
  ssl:
    mode: requireSSL
    PEMKeyFile: /etc/mongodb/ssl/mongodb.pem
    CAFile: /etc/mongodb/ssl/ca.pem
    allowInvalidCertificates: false
    allowInvalidHostnames: false

security:
  authorization: enabled
  keyFile: /etc/mongodb/keyfile
  clusterAuthMode: x509

replication:
  replSetName: rs0

# 审计日志
auditLog:
  destination: file
  format: JSON
  path: /var/log/mongodb/audit.json
```

## 故障排查和恢复

### 1. 常见故障诊断工具

```javascript
/**
 * MongoDB故障诊断工具
 */
class DiagnosticTool {
    constructor(client) {
        this.client = client;
        this.admin = client.db('admin').admin();
    }
    
    /**
     * 系统健康检查
     */
    async performHealthCheck() {
        const healthReport = {
            timestamp: new Date(),
            server: {},
            replication: {},
            sharding: {},
            performance: {},
            issues: []
        };
        
        try {
            // 1. 服务器状态检查
            const serverStatus = await this.admin.serverStatus();
            healthReport.server = {
                version: serverStatus.version,
                uptime: serverStatus.uptime,
                connections: serverStatus.connections,
                network: serverStatus.network,
                opcounters: serverStatus.opcounters,
                mem: serverStatus.mem,
                globalLock: serverStatus.globalLock
            };
            
            // 检查连接数
            if (serverStatus.connections.current > serverStatus.connections.available * 0.8) {
                healthReport.issues.push({
                    level: 'warning',
                    message: '连接数接近上限',
                    current: serverStatus.connections.current,
                    available: serverStatus.connections.available
                });
            }
            
            // 检查内存使用
            if (serverStatus.mem.resident > serverStatus.mem.virtual * 0.9) {
                healthReport.issues.push({
                    level: 'critical',
                    message: '内存使用率过高',
                    resident: serverStatus.mem.resident,
                    virtual: serverStatus.mem.virtual
                });
            }
            
            // 2. 副本集状态检查
            try {
                const replStatus = await this.admin.command({ replSetGetStatus: 1 });
                healthReport.replication = {
                    set: replStatus.set,
                    members: replStatus.members.map(member => ({
                        name: member.name,
                        state: member.stateStr,
                        health: member.health,
                        uptime: member.uptime,
                        optime: member.optime,
                        lag: member.optimeDate ? 
                            (new Date() - member.optimeDate) / 1000 : null
                    }))
                };
                
                // 检查副本延迟
                for (const member of healthReport.replication.members) {
                    if (member.lag > 10) { // 超过10秒延迟
                        healthReport.issues.push({
                            level: 'warning',
                            message: `副本 ${member.name} 延迟过高`,
                            lag: member.lag
                        });
                    }
                }
                
            } catch (error) {
                // 可能不是副本集环境
                healthReport.replication.error = 'Not a replica set or access denied';
            }
            
            // 3. 分片状态检查
            try {
                const shardStatus = await this.admin.command({ listShards: 1 });
                healthReport.sharding = {
                    shards: shardStatus.shards,
                    isSharded: true
                };
                
            } catch (error) {
                healthReport.sharding = { isSharded: false };
            }
            
            // 4. 性能指标检查
            const dbStats = await this.client.db().admin().listDatabases();
            healthReport.performance = {
                databases: dbStats.databases.length,
                totalSize: dbStats.totalSize
            };
            
        } catch (error) {
            healthReport.issues.push({
                level: 'critical',
                message: 'Health check failed',
                error: error.message
            });
        }
        
        return healthReport;
    }
    
    /**
     * 慢操作检测
     */
    async detectSlowOperations() {
        try {
            const currentOps = await this.admin.command({
                currentOp: true,
                $or: [
                    { "active": true, "secs_running": { $gte: 5 } },
                    { "locks": { $exists: true } }
                ]
            });
            
            const slowOps = currentOps.inprog
                .filter(op => op.secs_running >= 5)
                .map(op => ({
                    opid: op.opid,
                    operation: op.op,
                    namespace: op.ns,
                    runningTime: op.secs_running,
                    command: op.command,
                    client: op.client,
                    desc: op.desc
                }));
                
            return slowOps;
            
        } catch (error) {
            console.error('慢操作检测失败:', error);
            return [];
        }
    }
    
    /**
     * 锁等待检测
     */
    async detectLockWaiting() {
        try {
            const currentOps = await this.admin.command({
                currentOp: true,
                waitingForLock: true
            });
            
            return currentOps.inprog.map(op => ({
                opid: op.opid,
                operation: op.op,
                namespace: op.ns,
                waitingTime: op.secs_running,
                lockType: op.lockStats,
                client: op.client
            }));
            
        } catch (error) {
            console.error('锁等待检测失败:', error);
            return [];
        }
    }
    
    /**
     * 索引使用分析
     */
    async analyzeIndexUsage() {
        const databases = await this.admin.listDatabases();
        const analysis = {};
        
        for (const dbInfo of databases.databases) {
            if (dbInfo.name === 'admin' || dbInfo.name === 'local' || dbInfo.name === 'config') {
                continue;
            }
            
            const db = this.client.db(dbInfo.name);
            const collections = await db.listCollections().toArray();
            analysis[dbInfo.name] = {};
            
            for (const collInfo of collections) {
                const collection = db.collection(collInfo.name);
                
                try {
                    const indexStats = await collection.aggregate([
                        { $indexStats: {} }
                    ]).toArray();
                    
                    analysis[dbInfo.name][collInfo.name] = indexStats.map(stat => ({
                        name: stat.name,
                        accesses: stat.accesses.ops,
                        since: stat.accesses.since,
                        spec: stat.spec
                    }));
                    
                } catch (error) {
                    analysis[dbInfo.name][collInfo.name] = { error: error.message };
                }
            }
        }
        
        return analysis;
    }
    
    /**
     * 生成诊断报告
     */
    async generateDiagnosticReport() {
        console.log('开始生成MongoDB诊断报告...');
        
        const report = {
            timestamp: new Date(),
            healthCheck: await this.performHealthCheck(),
            slowOperations: await this.detectSlowOperations(),
            lockWaiting: await this.detectLockWaiting(),
            indexUsage: await this.analyzeIndexUsage()
        };
        
        // 生成建议
        report.recommendations = this.generateRecommendations(report);
        
        return report;
    }
    
    /**
     * 生成优化建议
     */
    generateRecommendations(report) {
        const recommendations = [];
        
        // 基于健康检查的建议
        for (const issue of report.healthCheck.issues) {
            if (issue.level === 'critical') {
                recommendations.push({
                    priority: 'high',
                    category: 'performance',
                    message: issue.message,
                    action: this.getActionForIssue(issue)
                });
            }
        }
        
        // 基于慢操作的建议
        if (report.slowOperations.length > 0) {
            recommendations.push({
                priority: 'medium',
                category: 'performance',
                message: `发现 ${report.slowOperations.length} 个慢操作`,
                action: '检查查询是否可以优化，考虑添加索引'
            });
        }
        
        // 基于索引使用的建议
        for (const [dbName, collections] of Object.entries(report.indexUsage)) {
            for (const [collName, indexes] of Object.entries(collections)) {
                if (Array.isArray(indexes)) {
                    const unusedIndexes = indexes.filter(idx => idx.accesses === 0);
                    if (unusedIndexes.length > 0) {
                        recommendations.push({
                            priority: 'low',
                            category: 'maintenance',
                            message: `${dbName}.${collName} 有 ${unusedIndexes.length} 个未使用的索引`,
                            action: '考虑删除未使用的索引以节省存储空间'
                        });
                    }
                }
            }
        }
        
        return recommendations;
    }
    
    /**
     * 获取问题对应的行动建议
     */
    getActionForIssue(issue) {
        const actions = {
            '连接数接近上限': '增加最大连接数配置或优化连接池使用',
            '内存使用率过高': '增加物理内存或优化查询减少内存使用',
            '副本延迟过高': '检查网络连接和副本集配置，考虑增加副本集成员'
        };
        
        return actions[issue.message] || '请查阅MongoDB文档获取详细解决方案';
    }
}

// 使用示例
async function runDiagnostics() {
    const { MongoClient } = require('mongodb');
    const client = new MongoClient('mongodb://localhost:27017');
    
    try {
        await client.connect();
        const diagnostic = new DiagnosticTool(client);
        
        const report = await diagnostic.generateDiagnosticReport();
        
        console.log('=== MongoDB诊断报告 ===');
        console.log(JSON.stringify(report, null, 2));
        
        // 保存报告到文件
        const fs = require('fs');
        const filename = `mongodb-diagnostic-${Date.now()}.json`;
        fs.writeFileSync(filename, JSON.stringify(report, null, 2));
        console.log(`诊断报告已保存到: ${filename}`);
        
    } catch (error) {
        console.error('诊断失败:', error);
    } finally {
        await client.close();
    }
}
```

### 2. 备份和恢复策略

```bash
#!/bin/bash
# MongoDB备份和恢复脚本

# 配置参数
MONGO_HOST="localhost:27017"
MONGO_AUTH_DB="admin"
MONGO_USER="backup_user"
MONGO_PASSWORD="backup_password"
BACKUP_DIR="/backup/mongodb"
RETENTION_DAYS=7
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR/$DATE

# 1. 完整备份（使用mongodump）
echo "开始完整备份..."
mongodump \
    --host $MONGO_HOST \
    --authenticationDatabase $MONGO_AUTH_DB \
    --username $MONGO_USER \
    --password $MONGO_PASSWORD \
    --out $BACKUP_DIR/$DATE/complete \
    --gzip \
    --oplog

if [ $? -eq 0 ]; then
    echo "完整备份成功: $BACKUP_DIR/$DATE/complete"
else
    echo "完整备份失败"
    exit 1
fi

# 2. 增量备份（基于oplog）
if [ -f "$BACKUP_DIR/last_oplog_timestamp" ]; then
    LAST_TIMESTAMP=$(cat $BACKUP_DIR/last_oplog_timestamp)
    echo "开始增量备份，从时间戳: $LAST_TIMESTAMP"
    
    mongodump \
        --host $MONGO_HOST \
        --authenticationDatabase $MONGO_AUTH_DB \
        --username $MONGO_USER \
        --password $MONGO_PASSWORD \
        --db local \
        --collection oplog.rs \
        --query "{'ts': {\$gt: $LAST_TIMESTAMP}}" \
        --out $BACKUP_DIR/$DATE/incremental \
        --gzip
fi

# 记录当前oplog时间戳
mongo --host $MONGO_HOST \
      --authenticationDatabase $MONGO_AUTH_DB \
      --username $MONGO_USER \
      --password $MONGO_PASSWORD \
      --quiet \
      --eval "db.runCommand({isMaster: 1}).lastWrite.opTime.ts.toString()" \
      > $BACKUP_DIR/last_oplog_timestamp

# 3. 压缩备份
echo "压缩备份文件..."
cd $BACKUP_DIR
tar -czf mongodb_backup_$DATE.tar.gz $DATE/
rm -rf $DATE/

# 4. 上传到云存储（示例：AWS S3）
if command -v aws &> /dev/null; then
    echo "上传备份到S3..."
    aws s3 cp mongodb_backup_$DATE.tar.gz s3://my-mongodb-backups/
    
    if [ $? -eq 0 ]; then
        echo "备份已上传到S3"
    else
        echo "S3上传失败"
    fi
fi

# 5. 清理过期备份
echo "清理过期备份..."
find $BACKUP_DIR -name "mongodb_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "备份脚本执行完成"

# 恢复脚本函数
restore_mongodb() {
    local BACKUP_FILE=$1
    local TARGET_HOST=${2:-"localhost:27017"}
    local RESTORE_DIR="/tmp/mongodb_restore_$$"
    
    echo "开始恢复MongoDB数据库..."
    echo "备份文件: $BACKUP_FILE"
    echo "目标主机: $TARGET_HOST"
    
    # 创建临时恢复目录
    mkdir -p $RESTORE_DIR
    
    # 解压备份文件
    echo "解压备份文件..."
    tar -xzf $BACKUP_FILE -C $RESTORE_DIR
    
    # 获取解压后的目录名
    EXTRACT_DIR=$(find $RESTORE_DIR -maxdepth 1 -type d -name "*_*" | head -1)
    
    if [ ! -d "$EXTRACT_DIR/complete" ]; then
        echo "错误：备份文件格式不正确"
        rm -rf $RESTORE_DIR
        exit 1
    fi
    
    # 恢复数据
    echo "恢复数据库数据..."
    mongorestore \
        --host $TARGET_HOST \
        --authenticationDatabase $MONGO_AUTH_DB \
        --username $MONGO_USER \
        --password $MONGO_PASSWORD \
        --drop \
        --oplogReplay \
        --gzip \
        $EXTRACT_DIR/complete
    
    if [ $? -eq 0 ]; then
        echo "数据恢复成功"
    else
        echo "数据恢复失败"
        rm -rf $RESTORE_DIR
        exit 1
    fi
    
    # 清理临时文件
    rm -rf $RESTORE_DIR
    echo "恢复完成"
}

# Point-in-Time 恢复函数
point_in_time_restore() {
    local BACKUP_FILE=$1
    local TARGET_TIMESTAMP=$2
    local TARGET_HOST=${3:-"localhost:27017"}
    
    echo "执行时间点恢复..."
    echo "目标时间戳: $TARGET_TIMESTAMP"
    
    # 首先执行完整恢复
    restore_mongodb $BACKUP_FILE $TARGET_HOST
    
    # 然后应用oplog到指定时间点
    echo "应用oplog到指定时间点..."
    mongorestore \
        --host $TARGET_HOST \
        --authenticationDatabase $MONGO_AUTH_DB \
        --username $MONGO_USER \
        --password $MONGO_PASSWORD \
        --oplogReplay \
        --oplogLimit $TARGET_TIMESTAMP \
        --dir /path/to/oplog/dump
    
    echo "时间点恢复完成"
}

# 使用示例
# ./backup_restore.sh
# restore_mongodb "/backup/mongodb/mongodb_backup_20231120_120000.tar.gz"
# point_in_time_restore "/backup/mongodb/mongodb_backup_20231120_120000.tar.gz" "1700472000"
```

## 最终总结

本MongoDB源码剖析系列文档通过深入分析源码实现，为您提供了：

1. **完整的架构理解**: 从底层BSON格式到上层分片系统的全方位解析
2. **实用的API指南**: 详细的代码示例和最佳实践
3. **生产环境经验**: 实际部署、监控、调优的完整方案
4. **故障处理能力**: 系统化的诊断和恢复策略

通过掌握这些知识，您将能够：
- 更好地设计MongoDB应用架构
- 优化查询性能和索引策略
- 构建高可用的MongoDB集群
- 有效地监控和维护生产系统
- 快速诊断和解决各种故障

MongoDB作为一个成熟的分布式数据库系统，其源码体现了许多优秀的设计理念和工程实践。希望这份源码剖析能够帮助您在技术道路上更进一步。

---

*完成时间：2024年*
*文档版本：v1.0*
*适用MongoDB版本：7.0+*
