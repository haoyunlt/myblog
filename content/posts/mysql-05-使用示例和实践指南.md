---
title: "MySQL Router Mock Server - 使用示例和实践指南"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['源码分析', '技术文档']
categories: ['mysql', '技术分析']
description: "MySQL Router Mock Server - 使用示例和实践指南的深入技术分析文档"
keywords: ['源码分析', '技术文档']
author: "技术分析师"
weight: 1
---

## 1. 框架使用示例

### 1.1 基本使用流程

```bash
# 1. 编译Mock Server
cd mysql-server/router
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make mysql_server_mock

# 2. 创建配置文件
cat > simple_test.js << 'EOF'
({
  "handshake": {
    "greeting": {
      "server_version": "8.0.0-mock-test",
      "connection_id": 1,
      "exec_time": 0
    },
    "auth": {
      "username": "testuser",
      "password": "testpass"
    }
  },
  "stmts": [
    {
      "stmt": "SELECT @@version_comment",
      "result": {
        "columns": [{"name": "@@version_comment", "type": "STRING"}],
        "rows": [["MySQL Router Mock Server"]]
      }
    }
  ]
})
EOF

# 3. 启动Mock Server
./mysql_server_mock --port=3306 --filename=simple_test.js --verbose

# 4. 使用MySQL客户端连接测试
mysql -h127.0.0.1 -P3306 -utestuser -ptestpass -e "SELECT @@version_comment"
```

### 1.2 命令行参数详解

```bash
# 完整的命令行参数示例
./mysql_server_mock \
  --filename=/path/to/test_scenario.js \        # 测试脚本文件
  --bind-address=0.0.0.0 \                      # 绑定地址
  --port=3306 \                                 # Classic协议端口
  --xport=33060 \                               # X协议端口
  --socket=/tmp/mysql_mock.sock \               # Unix socket路径
  --http-port=8080 \                            # REST API端口
  --module-prefix=/usr/local/share/mock_modules \ # JS模块路径
  --ssl-mode=PREFERRED \                        # SSL模式
  --ssl-cert=/path/to/server.pem \             # SSL证书
  --ssl-key=/path/to/server-key.pem \          # SSL私钥
  --ssl-ca=/path/to/ca.pem \                   # CA证书
  --verbose \                                   # 详细日志
  --logging-folder=/var/log/mysql_mock         # 日志目录
```

### 1.3 Docker容器部署示例

```dockerfile
# Dockerfile for MySQL Mock Server
FROM ubuntu:22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libssl-dev \
    libduktape-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制源码并编译
COPY mysql-server /src/mysql-server
WORKDIR /src/mysql-server/router/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make mysql_server_mock && \
    cp mysql_server_mock /usr/local/bin/

# 创建运行用户
RUN useradd -r -s /bin/false mysql_mock

# 复制配置文件
COPY test_scenarios/ /etc/mysql_mock/

# 暴露端口
EXPOSE 3306 33060 8080

# 启动脚本
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER mysql_mock
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
```

```bash
#!/bin/bash
# docker-entrypoint.sh

# 默认配置
MOCK_FILE="${MOCK_FILE:-/etc/mysql_mock/default.js}"
BIND_ADDRESS="${BIND_ADDRESS:-0.0.0.0}"
PORT="${PORT:-3306}"
XPORT="${XPORT:-33060}"
HTTP_PORT="${HTTP_PORT:-8080}"

# 启动Mock Server
exec mysql_server_mock \
  --filename="$MOCK_FILE" \
  --bind-address="$BIND_ADDRESS" \
  --port="$PORT" \
  --xport="$XPORT" \
  --http-port="$HTTP_PORT" \
  --verbose
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  mysql-mock:
    build: .
    ports:
      - "3306:3306"
      - "33060:33060" 
      - "8080:8080"
    volumes:
      - ./scenarios:/etc/mysql_mock
      - ./logs:/var/log/mysql_mock
    environment:
      - MOCK_FILE=/etc/mysql_mock/integration_test.js
      - PORT=3306
      - VERBOSE=true
    networks:
      - test_network

  test_client:
    image: mysql:8.0
    depends_on:
      - mysql-mock
    command: >
      sh -c "
        sleep 10 &&
        mysql -hmysql-mock -P3306 -utestuser -ptestpass 
          -e 'SELECT @@version; SHOW DATABASES;'
      "
    networks:
      - test_network

networks:
  test_network:
```

## 2. 详细API说明和使用示例

### 2.1 JavaScript配置API详解

#### 2.1.1 握手配置API

```javascript
/**
 * @brief 完整的握手配置示例
 */
({
  "handshake": {
    // 服务器问候消息配置
    "greeting": {
      "server_version": "8.0.28-mock",      // 服务器版本字符串
      "connection_id": 42,                  // 连接ID（可动态生成）
      "capabilities": 0x807fffff,           // 服务器能力标志位
      "status_flags": 0x0002,              // 服务器状态标志
      "character_set": 8,                   // 默认字符集(latin1_swedish_ci)
      "auth_method": "caching_sha2_password", // 认证方法
      "nonce": "abcdefghij0123456789",      // 认证随机数(20字节)
      "exec_time": 1000                     // 握手延迟(微秒)
    },
    
    // 认证配置
    "auth": {
      "username": "admin",                  // 期望的用户名
      "password": "secret123",              // 期望的密码
      "auth_method_name": "mysql_native_password", // 强制使用的认证方法
      
      // 客户端证书验证(可选)
      "certificate": {
        "issuer": "CN=Test CA,O=Example Corp,C=US",
        "subject": "CN=test-client,O=Example Corp,C=US"
      }
    },
    
    // 错误响应(可选，用于测试认证失败)
    "error": {
      "code": 1045,
      "message": "Access denied for user 'test'@'localhost'",
      "sql_state": "28000"
    }
  }
})
```

#### 2.1.2 动态握手处理

```javascript
/**
 * @brief 动态握手处理示例
 */
({
  "handshake": function(is_greeting) {
    // 根据请求类型返回不同配置
    if (is_greeting) {
      return {
        "greeting": {
          "server_version": "8.0.0-dynamic-" + new Date().getTime(),
          "connection_id": Math.floor(Math.random() * 1000000),
          "exec_time": Math.random() < 0.1 ? 5000 : 0  // 10%概率延迟5ms
        }
      };
    } else {
      // 认证阶段
      var current_hour = new Date().getHours();
      if (current_hour < 9 || current_hour > 17) {
        // 工作时间外拒绝连接
        return {
          "error": {
            "code": 1226,
            "message": "User access denied outside business hours",
            "sql_state": "42000"
          }
        };
      }
      
      return {
        "auth": {
          "username": "business_user",
          "password": "work_password"
        }
      };
    }
  }
})
```

### 2.2 语句处理API详解

#### 2.2.1 静态响应配置

```javascript
/**
 * @brief 静态语句响应配置
 */
({
  "stmts": [
    {
      "stmt": "SELECT @@version",
      "exec_time": 500,  // 500微秒执行时间
      "result": {
        "columns": [{
          "catalog": "def",
          "schema": "",
          "table": "",
          "orig_table": "",
          "name": "@@version",
          "orig_name": "",
          "character_set": 33,  // utf8_general_ci
          "length": 255,
          "type": "VAR_STRING",
          "flags": 1,           // NOT_NULL
          "decimals": 31
        }],
        "rows": [["8.0.28-mock-test"]],
        "affected_rows": 0,
        "last_insert_id": 0,
        "status": 0,
        "warning_count": 0
      }
    },
    
    {
      "stmt": "INSERT INTO test_table VALUES (1, 'test')",
      "ok": {
        "affected_rows": 1,
        "last_insert_id": 1,
        "status": 0x0002,      // SERVER_STATUS_AUTOCOMMIT
        "warning_count": 0,
        "message": "",
        "session_trackers": [
          {
            "type": "system_variable",
            "name": "autocommit",
            "value": "ON"
          }
        ]
      }
    },
    
    {
      "stmt": "SELECT * FROM nonexistent_table",
      "error": {
        "code": 1146,
        "message": "Table 'test.nonexistent_table' doesn't exist",
        "sql_state": "42S02"
      }
    }
  ]
})
```

#### 2.2.2 动态语句处理

```javascript
/**
 * @brief 动态语句处理示例
 */
({
  "stmts": function(stmt) {
    // 模拟数据库状态
    if (!mysqld.global.table_created) {
      mysqld.global.table_created = false;
      mysqld.global.row_count = 0;
    }
    
    // DDL语句处理
    if (stmt.match(/^CREATE TABLE/i)) {
      mysqld.global.table_created = true;
      return {
        "ok": {
          "affected_rows": 0,
          "message": "Table created successfully"
        }
      };
    }
    
    // DML语句处理
    if (stmt.match(/^INSERT INTO/i)) {
      if (!mysqld.global.table_created) {
        return {
          "error": {
            "code": 1146,
            "message": "Table doesn't exist",
            "sql_state": "42S02"
          }
        };
      }
      
      mysqld.global.row_count++;
      return {
        "ok": {
          "affected_rows": 1,
          "last_insert_id": mysqld.global.row_count
        }
      };
    }
    
    // 查询语句处理
    if (stmt.match(/^SELECT COUNT\(\*\) FROM/i)) {
      return {
        "result": {
          "columns": [{"name": "COUNT(*)", "type": "LONGLONG"}],
          "rows": [[mysqld.global.row_count.toString()]]
        }
      };
    }
    
    // 事务处理
    if (stmt.match(/^(BEGIN|START TRANSACTION)/i)) {
      mysqld.global.in_transaction = true;
      return {"ok": {}};
    }
    
    if (stmt.match(/^(COMMIT|ROLLBACK)/i)) {
      mysqld.global.in_transaction = false;
      return {
        "ok": {
          "status": stmt.match(/ROLLBACK/i) ? 0 : 0x0002  // AUTOCOMMIT flag
        }
      };
    }
    
    // 默认错误响应
    return {
      "error": {
        "code": 1064,
        "message": "Unsupported statement: " + stmt,
        "sql_state": "42000"
      }
    };
  }
})
```

#### 2.2.3 复杂查询模拟

```javascript
/**
 * @brief 复杂查询处理示例
 */
({
  "stmts": function(stmt) {
    // 解析SQL语句
    var parsed = parseSQL(stmt);
    
    if (parsed.type === 'SELECT') {
      // 模拟不同的查询结果
      if (parsed.from === 'users') {
        return generateUserData(parsed);
      } else if (parsed.from === 'orders') {
        return generateOrderData(parsed);
      } else if (parsed.from === 'performance_schema.replication_group_members') {
        // 模拟Group Replication状态
        return {
          "result": {
            "columns": [
              {"name": "member_id", "type": "STRING"},
              {"name": "member_host", "type": "STRING"},
              {"name": "member_port", "type": "LONG"},
              {"name": "member_state", "type": "STRING"},
              {"name": "member_role", "type": "STRING"}
            ],
            "rows": [
              ["550fa9ee-a1f8-4b6d-9bfe-c03ed5d30654", "mysql-node1", 3306, "ONLINE", "PRIMARY"],
              ["6091e3d1-b2f2-4c5d-a3e4-d04f6e7a8b9c", "mysql-node2", 3306, "ONLINE", "SECONDARY"],
              ["7182f4e2-c3f3-5d6e-b4f5-e15g7f8a9c0d", "mysql-node3", 3306, "ONLINE", "SECONDARY"]
            ]
          }
        };
      }
    }
    
    // 辅助函数：简单SQL解析
    function parseSQL(sql) {
      var normalized = sql.toLowerCase().trim();
      if (normalized.startsWith('select')) {
        var fromMatch = normalized.match(/from\s+([^\s\;]+)/);
        return {
          type: 'SELECT',
          from: fromMatch ? fromMatch[1] : null
        };
      }
      return {type: 'UNKNOWN'};
    }
    
    // 生成用户数据
    function generateUserData(parsed) {
      var users = [
        [1, "admin", "admin@example.com", "2023-01-01 10:00:00"],
        [2, "user1", "user1@example.com", "2023-01-02 11:00:00"],
        [3, "user2", "user2@example.com", "2023-01-03 12:00:00"]
      ];
      
      return {
        "result": {
          "columns": [
            {"name": "id", "type": "LONG"},
            {"name": "username", "type": "STRING"},
            {"name": "email", "type": "STRING"},
            {"name": "created_at", "type": "DATETIME"}
          ],
          "rows": users
        }
      };
    }
    
    // 生成订单数据
    function generateOrderData(parsed) {
      return {
        "result": {
          "columns": [
            {"name": "order_id", "type": "LONG"},
            {"name": "user_id", "type": "LONG"},
            {"name": "amount", "type": "DECIMAL"},
            {"name": "status", "type": "STRING"}
          ],
          "rows": [
            [1001, 1, "99.99", "completed"],
            [1002, 2, "149.99", "pending"],
            [1003, 1, "79.99", "completed"]
          ]
        }
      };
    }
    
    return {
      "error": {
        "code": 1146,
        "message": "Table '" + (parsed.from || 'unknown') + "' doesn't exist",
        "sql_state": "42S02"
      }
    };
  }
})
```

### 2.3 REST API使用示例

#### 2.3.1 HTTP接口配置

```bash
# 启动支持REST API的Mock Server
./mysql_server_mock \
  --filename=test.js \
  --port=3306 \
  --http-port=8080 \
  --verbose
```

```javascript
// JavaScript中通过REST API动态修改行为
({
  "stmts": function(stmt) {
    // 检查全局状态
    var maintenance_mode = mysqld.global.maintenance_mode || false;
    
    if (maintenance_mode && !stmt.match(/^(SHOW|SELECT @@)/i)) {
      return {
        "error": {
          "code": 1105,
          "message": "Server is in maintenance mode",
          "sql_state": "HY000"
        }
      };
    }
    
    // 正常处理
    return handle_normal_statement(stmt);
  }
})
```

#### 2.3.2 REST API客户端示例

```python
#!/usr/bin/env python3
"""
MySQL Mock Server REST API客户端示例
"""
import requests
import json
import time

class MockServerClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        
    def set_global_variable(self, key, value):
        """设置全局变量"""
        url = f"{self.base_url}/api/v1/mock/globals/{key}"
        response = requests.put(url, json={"value": value})
        return response.status_code == 200
    
    def get_global_variable(self, key):
        """获取全局变量"""
        url = f"{self.base_url}/api/v1/mock/globals/{key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["value"]
        return None
    
    def enable_maintenance_mode(self):
        """启用维护模式"""
        return self.set_global_variable("maintenance_mode", True)
    
    def disable_maintenance_mode(self):
        """禁用维护模式"""
        return self.set_global_variable("maintenance_mode", False)
    
    def get_connection_stats(self):
        """获取连接统计"""
        url = f"{self.base_url}/api/v1/mock/stats/connections"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None

# 使用示例
def main():
    client = MockServerClient()
    
    print("启用维护模式...")
    client.enable_maintenance_mode()
    
    print("等待5秒...")
    time.sleep(5)
    
    print("禁用维护模式...")
    client.disable_maintenance_mode()
    
    stats = client.get_connection_stats()
    if stats:
        print(f"当前连接数: {stats['active_connections']}")
        print(f"总连接数: {stats['total_connections']}")

if __name__ == "__main__":
    main()
```

```curl
# 使用curl操作REST API

# 获取服务器状态
curl -X GET http://localhost:8080/api/v1/mock/status

# 设置全局变量
curl -X PUT http://localhost:8080/api/v1/mock/globals/test_mode \
  -H "Content-Type: application/json" \
  -d '{"value": true}'

# 获取连接统计
curl -X GET http://localhost:8080/api/v1/mock/stats/connections

# 重新加载配置
curl -X POST http://localhost:8080/api/v1/mock/reload

# 获取所有全局变量
curl -X GET http://localhost:8080/api/v1/mock/globals
```

## 3. 测试场景实战案例

### 3.1 MySQL Router Bootstrap测试

```javascript
/**
 * @brief MySQL Router Bootstrap场景测试
 */
({
  "handshake": {
    "greeting": {
      "server_version": "8.0.28",
      "connection_id": 10
    },
    "auth": {
      "username": "root",
      "password": "root_password"
    }
  },
  
  "stmts": [
    // 检查schema版本
    {
      "stmt": "SELECT * FROM mysql_innodb_cluster_metadata.schema_version",
      "result": {
        "columns": [
          {"name": "major", "type": "LONG"},
          {"name": "minor", "type": "LONG"},
          {"name": "patch", "type": "LONG"}
        ],
        "rows": [["2", "1", "0"]]
      }
    },
    
    // 检查集群存在
    {
      "stmt": "SELECT ((SELECT count(*) FROM mysql_innodb_cluster_metadata.clusters) <= 1 AND (SELECT count(*) FROM mysql_innodb_cluster_metadata.replicasets) <= 1) as has_one_replicaset, (SELECT attributes->>'$.adopted' FROM mysql_innodb_cluster_metadata.clusters) as adopted",
      "result": {
        "columns": [
          {"name": "has_one_replicaset", "type": "LONG"},
          {"name": "adopted", "type": "STRING"}
        ],
        "rows": [["1", "null"]]
      }
    },
    
    // 获取集群信息
    {
      "stmt": "SELECT c.cluster_id, c.cluster_name, c.description, c.options, c.attributes FROM mysql_innodb_cluster_metadata.clusters c",
      "result": {
        "columns": [
          {"name": "cluster_id", "type": "STRING"},
          {"name": "cluster_name", "type": "STRING"},
          {"name": "description", "type": "STRING"},
          {"name": "options", "type": "JSON"},
          {"name": "attributes", "type": "JSON"}
        ],
        "rows": [
          ["550fa9ee-a1f8-4b6d-9bfe-c03ed5d30654", "testCluster", "Test InnoDB Cluster", "{}", "{}"]
        ]
      }
    },
    
    // 获取复制集信息
    {
      "stmt.regex": "SELECT R.replicaset_name, I.mysql_server_uuid, H.location, I.addresses.*",
      "result": {
        "columns": [
          {"name": "replicaset_name", "type": "STRING"},
          {"name": "mysql_server_uuid", "type": "STRING"},
          {"name": "location", "type": "STRING"},
          {"name": "addresses", "type": "JSON"}
        ],
        "rows": [
          ["default", "550fa9ee-a1f8-4b6d-9bfe-c03ed5d30654", "", "{\"mysqlClassic\": \"localhost:3306\", \"mysqlX\": \"localhost:33060\"}"],
          ["default", "6091e3d1-b2f2-4c5d-a3e4-d04f6e7a8b9c", "", "{\"mysqlClassic\": \"localhost:3307\", \"mysqlX\": \"localhost:33070\"}"],
          ["default", "7182f4e2-c3f3-5d6e-b4f5-e15g7f8a9c0d", "", "{\"mysqlClassic\": \"localhost:3308\", \"mysqlX\": \"localhost:33080\"}"]
        ]
      }
    }
  ]
})
```

### 3.2 Group Replication状态模拟

```javascript
/**
 * @brief Group Replication状态变化模拟
 */
({
  "stmts": function(stmt) {
    // 初始化集群状态
    if (!mysqld.global.group_members) {
      mysqld.global.group_members = [
        {
          uuid: "550fa9ee-a1f8-4b6d-9bfe-c03ed5d30654",
          host: "mysql-node1", 
          port: 3306,
          state: "ONLINE",
          role: "PRIMARY"
        },
        {
          uuid: "6091e3d1-b2f2-4c5d-a3e4-d04f6e7a8b9c",
          host: "mysql-node2",
          port: 3306, 
          state: "ONLINE",
          role: "SECONDARY"
        },
        {
          uuid: "7182f4e2-c3f3-5d6e-b4f5-e15g7f8a9c0d",
          host: "mysql-node3",
          port: 3306,
          state: "RECOVERING", // 模拟恢复状态
          role: "SECONDARY"
        }
      ];
      mysqld.global.primary_member = "550fa9ee-a1f8-4b6d-9bfe-c03ed5d30654";
      mysqld.global.failover_count = 0;
    }
    
    // 查询Group Replication成员
    if (stmt.match(/SELECT.*FROM performance_schema\.replication_group_members/i)) {
      // 模拟节点状态变化
      simulateNodeStateChanges();
      
      var rows = mysqld.global.group_members.map(function(member) {
        return [
          member.uuid,
          member.host,
          member.port,
          member.state,
          member.role === "PRIMARY" ? "1" : "0"
        ];
      });
      
      return {
        "result": {
          "columns": [
            {"name": "member_id", "type": "STRING"},
            {"name": "member_host", "type": "STRING"}, 
            {"name": "member_port", "type": "LONG"},
            {"name": "member_state", "type": "STRING"},
            {"name": "@@group_replication_single_primary_mode", "type": "STRING"}
          ],
          "rows": rows
        }
      };
    }
    
    // 查询Primary成员
    if (stmt.match(/show status like 'group_replication_primary_member'/i)) {
      return {
        "result": {
          "columns": [
            {"name": "Variable_name", "type": "STRING"},
            {"name": "Value", "type": "STRING"}
          ],
          "rows": [["group_replication_primary_member", mysqld.global.primary_member]]
        }
      };
    }
    
    // 模拟节点状态变化的函数
    function simulateNodeStateChanges() {
      var now = new Date().getTime();
      
      // 每30秒模拟一次状态变化
      if (!mysqld.global.last_state_change || (now - mysqld.global.last_state_change) > 30000) {
        mysqld.global.last_state_change = now;
        
        // 随机事件模拟
        var event = Math.random();
        
        if (event < 0.1) {
          // 10%概率：Primary故障切换
          simulateFailover();
        } else if (event < 0.2) {
          // 10%概率：节点恢复
          simulateNodeRecovery();
        } else if (event < 0.25) {
          // 5%概率：节点故障
          simulateNodeFailure();
        }
      }
    }
    
    function simulateFailover() {
      // 找到当前Primary
      var primaryIndex = mysqld.global.group_members.findIndex(
        function(m) { return m.role === "PRIMARY"; }
      );
      
      if (primaryIndex !== -1) {
        // 当前Primary变为不可用
        mysqld.global.group_members[primaryIndex].state = "ERROR";
        mysqld.global.group_members[primaryIndex].role = "SECONDARY";
        
        // 选择新的Primary（第一个ONLINE的SECONDARY）
        for (var i = 0; i < mysqld.global.group_members.length; i++) {
          if (mysqld.global.group_members[i].state === "ONLINE" && 
              mysqld.global.group_members[i].role === "SECONDARY") {
            mysqld.global.group_members[i].role = "PRIMARY";
            mysqld.global.primary_member = mysqld.global.group_members[i].uuid;
            mysqld.global.failover_count++;
            break;
          }
        }
      }
    }
    
    function simulateNodeRecovery() {
      // 找到ERROR状态的节点进行恢复
      for (var i = 0; i < mysqld.global.group_members.length; i++) {
        if (mysqld.global.group_members[i].state === "ERROR") {
          mysqld.global.group_members[i].state = "RECOVERING";
          
          // 延迟恢复到ONLINE状态
          setTimeout(function(index) {
            return function() {
              if (mysqld.global.group_members[index]) {
                mysqld.global.group_members[index].state = "ONLINE";
              }
            };
          }(i), 10000); // 10秒后恢复
          break;
        }
      }
    }
    
    function simulateNodeFailure() {
      // 随机选择一个SECONDARY节点故障
      var secondaries = mysqld.global.group_members.filter(
        function(m) { return m.role === "SECONDARY" && m.state === "ONLINE"; }
      );
      
      if (secondaries.length > 0) {
        var randomSecondary = secondaries[Math.floor(Math.random() * secondaries.length)];
        randomSecondary.state = "ERROR";
      }
    }
    
    return {
      "error": {
        "code": 1064,
        "message": "Unknown statement: " + stmt,
        "sql_state": "42000"
      }
    };
  }
})
```

### 3.3 性能测试场景

```javascript
/**
 * @brief 性能测试场景配置
 */
({
  "stmts": function(stmt) {
    // 初始化性能计数器
    if (!mysqld.global.perf_counters) {
      mysqld.global.perf_counters = {
        queries_total: 0,
        queries_per_second: 0,
        slow_queries: 0,
        last_second: Math.floor(Date.now() / 1000)
      };
    }
    
    var counters = mysqld.global.perf_counters;
    var current_second = Math.floor(Date.now() / 1000);
    
    // 重置每秒计数器
    if (current_second !== counters.last_second) {
      counters.queries_per_second = 0;
      counters.last_second = current_second;
    }
    
    counters.queries_total++;
    counters.queries_per_second++;
    
    // 模拟慢查询
    var is_slow_query = stmt.match(/SELECT.*SLEEP\(|SELECT.*BENCHMARK\(|SELECT.*HEAVY_COMPUTATION/i);
    var exec_time = 0;
    
    if (is_slow_query) {
      counters.slow_queries++;
      exec_time = 2000000; // 2秒
    } else if (stmt.match(/SELECT/i)) {
      exec_time = Math.random() * 10000; // 0-10ms随机延迟
    }
    
    // 性能统计查询
    if (stmt.match(/SHOW.*STATUS.*LIKE.*Queries/i)) {
      return {
        "exec_time": exec_time,
        "result": {
          "columns": [
            {"name": "Variable_name", "type": "STRING"},
            {"name": "Value", "type": "STRING"}
          ],
          "rows": [
            ["Queries", counters.queries_total.toString()],
            ["Queries_per_second", counters.queries_per_second.toString()],
            ["Slow_queries", counters.slow_queries.toString()]
          ]
        }
      };
    }
    
    // 模拟大结果集查询
    if (stmt.match(/SELECT.*FROM.*large_table/i)) {
      var row_count = 10000;
      var rows = [];
      
      for (var i = 1; i <= row_count; i++) {
        rows.push([
          i.toString(),
          "data_" + i,
          (Math.random() * 1000).toFixed(2),
          new Date().toISOString()
        ]);
      }
      
      return {
        "exec_time": 50000, // 50ms
        "result": {
          "columns": [
            {"name": "id", "type": "LONG"},
            {"name": "name", "type": "STRING"},
            {"name": "value", "type": "DECIMAL"},
            {"name": "created_at", "type": "DATETIME"}
          ],
          "rows": rows,
          "affected_rows": row_count
        }
      };
    }
    
    // 模拟连接池压力测试
    if (stmt.match(/SELECT.*CONNECTION_ID\(\)/i)) {
      return {
        "exec_time": exec_time,
        "result": {
          "columns": [{"name": "CONNECTION_ID()", "type": "LONG"}],
          "rows": [[Math.floor(Math.random() * 1000000).toString()]]
        }
      };
    }
    
    // 默认响应
    return {
      "exec_time": exec_time,
      "ok": {
        "affected_rows": Math.floor(Math.random() * 10),
        "message": "Query executed successfully"
      }
    };
  }
})
```

## 4. 集成测试实践

### 4.1 自动化测试框架

```python
#!/usr/bin/env python3
"""
MySQL Mock Server 自动化测试框架
"""
import subprocess
import time
import mysql.connector
import threading
import json
import os
from contextlib import contextmanager

class MockServerTestFramework:
    def __init__(self, mock_binary_path="./mysql_server_mock"):
        self.mock_binary = mock_binary_path
        self.mock_process = None
        self.test_results = []
        
    @contextmanager
    def mock_server(self, config_file, port=3306, **kwargs):
        """启动Mock Server的上下文管理器"""
        cmd = [
            self.mock_binary,
            f"--filename={config_file}",
            f"--port={port}",
            "--verbose"
        ]
        
        # 添加额外参数
        for key, value in kwargs.items():
            cmd.append(f"--{key.replace('_', '-')}={value}")
        
        try:
            print(f"启动Mock Server: {' '.join(cmd)}")
            self.mock_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 等待服务器启动
            time.sleep(2)
            
            if self.mock_process.poll() is not None:
                stdout, stderr = self.mock_process.communicate()
                raise RuntimeError(f"Mock server failed to start: {stderr.decode()}")
            
            yield port
            
        finally:
            if self.mock_process:
                self.mock_process.terminate()
                self.mock_process.wait(timeout=10)
                self.mock_process = None
    
    def create_mysql_connection(self, port, user="testuser", password="testpass", 
                              database=None, **kwargs):
        """创建MySQL连接"""
        config = {
            'host': 'localhost',
            'port': port,
            'user': user,
            'password': password,
            'autocommit': True,
            **kwargs
        }
        
        if database:
            config['database'] = database
            
        return mysql.connector.connect(**config)
    
    def run_test_case(self, test_func, config_file, description=""):
        """运行单个测试用例"""
        start_time = time.time()
        result = {
            'test': test_func.__name__,
            'description': description,
            'status': 'FAILED',
            'duration': 0,
            'error': None
        }
        
        try:
            with self.mock_server(config_file) as port:
                test_func(port)
            result['status'] = 'PASSED'
        except Exception as e:
            result['error'] = str(e)
        finally:
            result['duration'] = time.time() - start_time
            self.test_results.append(result)
            
        print(f"测试 {test_func.__name__}: {result['status']} "
              f"({result['duration']:.2f}s)")
        if result['error']:
            print(f"  错误: {result['error']}")
        
        return result['status'] == 'PASSED'
    
    def generate_report(self, output_file="test_report.json"):
        """生成测试报告"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['status'] == 'PASSED')
        failed_tests = total_tests - passed_tests
        
        report = {
            'summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'results': self.test_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n测试报告已生成: {output_file}")
        print(f"测试总数: {total_tests}, 通过: {passed_tests}, 失败: {failed_tests}")
        print(f"成功率: {report['summary']['success_rate']:.1f}%")
        
        return report

# 测试用例示例
def test_basic_connection(port):
    """测试基本连接"""
    framework = MockServerTestFramework()
    conn = framework.create_mysql_connection(port)
    cursor = conn.cursor()
    
    cursor.execute("SELECT @@version")
    result = cursor.fetchone()
    
    assert result[0] == "8.0.28-mock-test", f"Unexpected version: {result[0]}"
    
    conn.close()

def test_authentication_failure(port):
    """测试认证失败"""
    framework = MockServerTestFramework()
    
    try:
        conn = framework.create_mysql_connection(port, user="wronguser", password="wrongpass")
        conn.close()
        assert False, "Should have failed authentication"
    except mysql.connector.Error as e:
        assert e.errno == 1045, f"Expected error 1045, got {e.errno}"

def test_query_execution(port):
    """测试查询执行"""
    framework = MockServerTestFramework()
    conn = framework.create_mysql_connection(port)
    cursor = conn.cursor()
    
    # 测试INSERT
    cursor.execute("INSERT INTO test_table VALUES (1, 'test')")
    assert cursor.rowcount == 1
    
    # 测试SELECT
    cursor.execute("SELECT COUNT(*) FROM test_table")
    result = cursor.fetchone()
    assert result[0] > 0
    
    conn.close()

def test_error_handling(port):
    """测试错误处理"""
    framework = MockServerTestFramework()
    conn = framework.create_mysql_connection(port)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM nonexistent_table")
        assert False, "Should have raised an error"
    except mysql.connector.Error as e:
        assert e.errno == 1146, f"Expected error 1146, got {e.errno}"
    
    conn.close()

# 性能测试
def test_concurrent_connections(port):
    """测试并发连接"""
    framework = MockServerTestFramework()
    
    def worker_thread(thread_id):
        try:
            conn = framework.create_mysql_connection(port)
            cursor = conn.cursor()
            
            for i in range(10):
                cursor.execute("SELECT CONNECTION_ID()")
                cursor.fetchone()
            
            conn.close()
        except Exception as e:
            raise RuntimeError(f"Thread {thread_id} failed: {e}")
    
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker_thread, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

# 主测试流程
def main():
    # 创建测试配置文件
    test_config = {
        "handshake": {
            "greeting": {"server_version": "8.0.28-mock-test"},
            "auth": {"username": "testuser", "password": "testpass"}
        },
        "stmts": [
            {
                "stmt": "SELECT @@version",
                "result": {
                    "columns": [{"name": "@@version", "type": "STRING"}],
                    "rows": [["8.0.28-mock-test"]]
                }
            },
            {
                "stmt": "INSERT INTO test_table VALUES (1, 'test')",
                "ok": {"affected_rows": 1}
            },
            {
                "stmt": "SELECT COUNT(*) FROM test_table", 
                "result": {
                    "columns": [{"name": "COUNT(*)", "type": "LONGLONG"}],
                    "rows": [["1"]]
                }
            },
            {
                "stmt": "SELECT * FROM nonexistent_table",
                "error": {
                    "code": 1146,
                    "message": "Table 'test.nonexistent_table' doesn't exist",
                    "sql_state": "42S02"
                }
            },
            {
                "stmt": "SELECT CONNECTION_ID()",
                "result": {
                    "columns": [{"name": "CONNECTION_ID()", "type": "LONG"}], 
                    "rows": [["42"]]
                }
            }
        ]
    }
    
    config_file = "test_config.js"
    with open(config_file, 'w') as f:
        f.write(f"({json.dumps(test_config, indent=2)})")
    
    # 运行测试
    framework = MockServerTestFramework()
    
    try:
        framework.run_test_case(test_basic_connection, config_file, "基本连接测试")
        framework.run_test_case(test_query_execution, config_file, "查询执行测试") 
        framework.run_test_case(test_error_handling, config_file, "错误处理测试")
        framework.run_test_case(test_concurrent_connections, config_file, "并发连接测试")
        
        # 认证失败测试需要特殊配置
        auth_fail_config = test_config.copy()
        auth_fail_config["handshake"]["auth"] = {"username": "admin", "password": "secret"}
        
        auth_config_file = "auth_test_config.js" 
        with open(auth_config_file, 'w') as f:
            f.write(f"({json.dumps(auth_fail_config, indent=2)})")
        
        framework.run_test_case(test_authentication_failure, auth_config_file, "认证失败测试")
        
    finally:
        # 清理配置文件
        if os.path.exists(config_file):
            os.remove(config_file)
        if os.path.exists(auth_config_file):
            os.remove(auth_config_file)
    
    # 生成报告
    framework.generate_report()

if __name__ == "__main__":
    main()
```

### 4.2 CI/CD集成示例

```yaml
# .github/workflows/mock-server-test.yml
name: Mock Server Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        compiler: [gcc, clang]
        build_type: [Debug, Release]
        
    steps:
    - uses: actions/checkout@v2
      with:
        submodules: recursive
    
    - name: Install Dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y \
          build-essential \
          cmake \
          libssl-dev \
          libduktape-dev \
          python3-pip \
          mysql-client
        pip3 install mysql-connector-python
    
    - name: Build Mock Server
      run: |
        cd router
        mkdir build && cd build
        cmake .. \
          -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} \
          -DCMAKE_CXX_COMPILER=${{ matrix.compiler == 'clang' && 'clang++' || 'g++' }}
        make mysql_server_mock -j$(nproc)
    
    - name: Run Unit Tests
      run: |
        cd router/build
        ctest --output-on-failure -R mock_server
    
    - name: Run Integration Tests  
      run: |
        cd router/build
        python3 ../../tests/integration/mock_server_test.py
    
    - name: Run Performance Tests
      run: |
        cd router/build
        python3 ../../tests/performance/mock_server_perf.py
        
    - name: Upload Test Results
      uses: actions/upload-artifact@v2
      if: always()
      with:
        name: test-results-${{ matrix.compiler }}-${{ matrix.build_type }}
        path: |
          router/build/test_report.json
          router/build/performance_report.json
```

```dockerfile
# 测试容器Dockerfile
FROM mysql:8.0 AS mysql-base

FROM ubuntu:22.04

# 复制MySQL客户端
COPY --from=mysql-base /usr/bin/mysql /usr/bin/mysql
COPY --from=mysql-base /usr/lib/x86_64-linux-gnu/libmysqlclient.so* /usr/lib/x86_64-linux-gnu/

# 安装测试依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && pip3 install mysql-connector-python pytest \
    && rm -rf /var/lib/apt/lists/*

# 复制Mock Server二进制
COPY mysql_server_mock /usr/local/bin/
COPY test_scenarios/ /opt/test_scenarios/
COPY tests/ /opt/tests/

WORKDIR /opt/tests
CMD ["python3", "-m", "pytest", ".", "-v"]
```

## 5. 总结

MySQL Router Mock Server通过其精心设计的架构和丰富的功能，为MySQL生态系统的测试提供了强大的支持。本文档详细分析了其：

### 5.1 核心价值

1. **测试效率提升**：将测试setup时间从20秒降低到毫秒级
2. **错误场景覆盖**：轻松模拟各种异常情况
3. **隔离性测试**：消除外部依赖，确保测试结果稳定
4. **成本降低**：减少测试环境的资源需求

### 5.2 技术亮点

1. **现代C++设计**：充分利用C++17/20特性，代码简洁高效
2. **异步IO架构**：基于io_context的高性能网络处理
3. **JavaScript集成**：提供灵活的脚本化配置能力
4. **协议完整性**：完整实现MySQL Client/Server协议

### 5.3 最佳实践总结

1. **RAII资源管理**：确保异常安全和资源正确释放
2. **对象池化**：优化频繁创建销毁的性能开销
3. **错误码vs异常**：在合适的场景选择合适的错误处理方式
4. **线程安全设计**：WaitableMonitor等工具类提供安全的并发访问

### 5.4 应用场景

1. **MySQL Router测试**：Bootstrap、配置验证、故障切换等
2. **应用程序测试**：数据库层集成测试、性能测试
3. **CI/CD流水线**：自动化测试环境搭建
4. **培训和演示**：MySQL协议学习、故障模拟演示

### 5.5 扩展方向

1. **协议支持扩展**：可添加更多数据库协议支持
2. **云原生集成**：Kubernetes Operator、服务网格集成
3. **监控集成**：Prometheus指标、分布式追踪支持
4. **配置管理**：热更新、版本控制、模板化配置

通过深入理解MySQL Router Mock Server的架构设计和实现细节，开发者可以：

- 更好地利用其功能进行高效测试
- 参与项目的开发和改进
- 借鉴其设计思想用于其他项目
- 构建更加可靠的测试基础设施

这个Mock Server项目体现了现代软件工程的最佳实践，为整个MySQL生态系统的质量保证做出了重要贡献。
