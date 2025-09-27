---
title: "Dify API参考手册：完整的接口文档与调用指南"
date: 2025-01-27T23:00:00+08:00
draft: false
featured: true
series: "dify-architecture"
tags: ["Dify", "API文档", "接口参考", "调用指南", "开发文档"]
categories: ["dify", "API文档"]
description: "Dify平台的完整API参考手册，包含所有接口的详细说明、参数定义、示例代码和最佳实践"
toc: true
tocOpen: true
showReadingTime: true
showWordCount: true
weight: 40
slug: "dify-api-reference"
---

## 概述

本文档提供Dify平台的完整API参考，包含Service API、Console API和Web API的详细说明、调用示例和最佳实践。

<!--more-->

## 1. API概览

### 1.1 API分类

Dify平台提供三类API接口：

| API类型 | 基础路径 | 目标用户 | 认证方式 | 主要功能 |
|---------|----------|----------|----------|----------|
| Service API | `/v1` | 外部开发者 | API Key | 应用调用、数据检索 |
| Console API | `/console/api` | 管理员 | Session | 应用管理、系统配置 |
| Web API | `/api` | 前端应用 | Token | 用户交互、界面数据 |

### 1.2 认证方式

#### Service API认证
```http
Authorization: Bearer {your-api-key}
Content-Type: application/json
```

#### Console API认证
```http
Cookie: session={session-id}
Content-Type: application/json
```

#### Web API认证
```http
Authorization: Bearer {user-token}
Content-Type: application/json
```

### 1.3 响应格式

所有API响应都采用统一的JSON格式：

```json
{
  "event": "message",
  "message_id": "uuid",
  "conversation_id": "uuid", 
  "answer": "响应内容",
  "metadata": {
    "usage": {
      "prompt_tokens": 100,
      "completion_tokens": 50,
      "total_tokens": 150
    }
  },
  "created_at": 1640995200
}
```

## 2. Service API详解

### 2.1 聊天消息接口

#### 发送聊天消息

**接口地址**: `POST /v1/chat-messages`

**功能说明**: 向聊天应用发送消息并获取AI回复

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| inputs | object | 是 | 输入变量字典 |
| query | string | 是 | 用户查询内容 |
| response_mode | string | 否 | 响应模式：streaming/blocking |
| conversation_id | string | 否 | 对话ID，续接对话时提供 |
| user | string | 是 | 用户唯一标识 |
| files | array | 否 | 文件列表，支持多模态输入 |

**请求示例**:

```bash
curl -X POST 'https://api.dify.ai/v1/chat-messages' \
--header 'Authorization: Bearer {api-key}' \
--header 'Content-Type: application/json' \
--data-raw '{
    "inputs": {
        "name": "张三"
    },
    "query": "你好，我想了解产品信息",
    "response_mode": "streaming",
    "conversation_id": "",
    "user": "user-123",
    "files": []
}'
```

**流式响应示例**:

```json
data: {"event": "message", "message_id": "msg-123", "conversation_id": "conv-456", "answer": "你好", "created_at": 1640995200}

data: {"event": "message", "message_id": "msg-123", "conversation_id": "conv-456", "answer": "张三", "created_at": 1640995201}

data: {"event": "message_end", "message_id": "msg-123", "conversation_id": "conv-456", "metadata": {"usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}}, "created_at": 1640995202}
```

**阻塞响应示例**:

```json
{
  "message_id": "msg-123",
  "conversation_id": "conv-456",
  "mode": "chat",
  "answer": "你好张三！我很乐意为您介绍我们的产品信息...",
  "metadata": {
    "usage": {
      "prompt_tokens": 50,
      "completion_tokens": 30,
      "total_tokens": 80
    }
  },
  "created_at": 1640995200
}
```

#### 获取对话历史

**接口地址**: `GET /v1/messages`

**功能说明**: 获取指定对话的消息历史

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| conversation_id | string | 是 | 对话ID |
| user | string | 是 | 用户标识 |
| first_id | string | 否 | 分页起始消息ID |
| limit | integer | 否 | 返回数量限制，默认20 |

**请求示例**:

```bash
curl -X GET 'https://api.dify.ai/v1/messages?conversation_id=conv-456&user=user-123&limit=20' \
--header 'Authorization: Bearer {api-key}'
```

**响应示例**:

```json
{
  "limit": 20,
  "has_more": false,
  "data": [
    {
      "id": "msg-123",
      "conversation_id": "conv-456",
      "inputs": {"name": "张三"},
      "query": "你好，我想了解产品信息",
      "answer": "你好张三！我很乐意为您介绍我们的产品信息...",
      "message_files": [],
      "feedback": null,
      "retriever_resources": [],
      "created_at": 1640995200
    }
  ]
}
```

### 2.2 文本补全接口

#### 创建文本补全

**接口地址**: `POST /v1/completions`

**功能说明**: 向补全应用发送提示并获取AI补全

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| inputs | object | 是 | 输入变量字典 |
| response_mode | string | 否 | 响应模式：streaming/blocking |
| user | string | 是 | 用户唯一标识 |
| files | array | 否 | 文件列表 |

**请求示例**:

```bash
curl -X POST 'https://api.dify.ai/v1/completions' \
--header 'Authorization: Bearer {api-key}' \
--header 'Content-Type: application/json' \
--data-raw '{
    "inputs": {
        "topic": "人工智能",
        "style": "专业"
    },
    "response_mode": "blocking",
    "user": "user-123"
}'
```

**响应示例**:

```json
{
  "message_id": "msg-789",
  "mode": "completion",
  "answer": "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支...",
  "metadata": {
    "usage": {
      "prompt_tokens": 30,
      "completion_tokens": 200,
      "total_tokens": 230
    }
  },
  "created_at": 1640995200
}
```

### 2.3 工作流执行接口

#### 运行工作流

**接口地址**: `POST /v1/workflows/run`

**功能说明**: 执行指定的工作流

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| inputs | object | 是 | 工作流输入变量 |
| response_mode | string | 否 | 响应模式：streaming/blocking |
| user | string | 是 | 用户唯一标识 |
| files | array | 否 | 文件列表 |

**请求示例**:

```bash
curl -X POST 'https://api.dify.ai/v1/workflows/run' \
--header 'Authorization: Bearer {api-key}' \
--header 'Content-Type: application/json' \
--data-raw '{
    "inputs": {
        "user_input": "分析这个文档的主要内容",
        "document_url": "https://example.com/doc.pdf"
    },
    "response_mode": "streaming",
    "user": "user-123"
}'
```

**流式响应示例**:

```json
data: {"event": "workflow_started", "workflow_run_id": "run-123", "created_at": 1640995200}

data: {"event": "node_started", "workflow_run_id": "run-123", "node_id": "node-1", "node_type": "llm", "created_at": 1640995201}

data: {"event": "node_finished", "workflow_run_id": "run-123", "node_id": "node-1", "data": {"outputs": {"result": "文档分析结果..."}}, "created_at": 1640995202}

data: {"event": "workflow_finished", "workflow_run_id": "run-123", "data": {"outputs": {"final_result": "完整分析报告..."}}, "created_at": 1640995203}
```

### 2.4 文件上传接口

#### 上传文件

**接口地址**: `POST /v1/files/upload`

**功能说明**: 上传文件用于多模态输入

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| file | file | 是 | 要上传的文件 |
| user | string | 是 | 用户唯一标识 |

**请求示例**:

```bash
curl -X POST 'https://api.dify.ai/v1/files/upload' \
--header 'Authorization: Bearer {api-key}' \
--form 'file=@"/path/to/file.pdf"' \
--form 'user="user-123"'
```

**响应示例**:

```json
{
  "id": "file-123",
  "name": "document.pdf",
  "size": 1024000,
  "extension": "pdf",
  "mime_type": "application/pdf",
  "created_by": "user-123",
  "created_at": 1640995200
}
```

### 2.5 应用信息接口

#### 获取应用配置

**接口地址**: `GET /v1/parameters`

**功能说明**: 获取应用的用户输入参数配置

**请求示例**:

```bash
curl -X GET 'https://api.dify.ai/v1/parameters' \
--header 'Authorization: Bearer {api-key}'
```

**响应示例**:

```json
{
  "opening_statement": "欢迎使用我们的AI助手！",
  "suggested_questions": [
    "如何使用这个产品？",
    "有什么新功能？",
    "如何联系客服？"
  ],
  "suggested_questions_after_answer": {
    "enabled": true
  },
  "speech_to_text": {
    "enabled": false
  },
  "text_to_speech": {
    "enabled": false
  },
  "retriever_resource": {
    "enabled": true
  },
  "annotation_reply": {
    "enabled": false
  },
  "user_input_form": [
    {
      "text-input": {
        "label": "姓名",
        "variable": "name",
        "required": true,
        "max_length": 50
      }
    }
  ],
  "file_upload": {
    "image": {
      "enabled": true,
      "number_limits": 3,
      "detail": "high",
      "transfer_methods": ["remote_url", "local_file"]
    }
  }
}
```

## 3. Console API详解

### 3.1 应用管理接口

#### 获取应用列表

**接口地址**: `GET /console/api/apps`

**功能说明**: 获取当前用户的应用列表

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| page | integer | 否 | 页码，默认1 |
| limit | integer | 否 | 每页数量，默认20 |
| search | string | 否 | 搜索关键词 |

**请求示例**:

```bash
curl -X GET 'https://api.dify.ai/console/api/apps?page=1&limit=20' \
--header 'Cookie: session={session-id}'
```

**响应示例**:

```json
{
  "data": [
    {
      "id": "app-123",
      "name": "智能客服",
      "description": "基于AI的智能客服系统",
      "mode": "chat",
      "status": "normal",
      "enable_site": true,
      "enable_api": true,
      "api_rpm": 1000,
      "api_rph": 10000,
      "is_demo": false,
      "model_config": {
        "provider": "openai",
        "model_id": "gpt-4",
        "configs": {
          "temperature": 0.1
        }
      },
      "created_at": 1640995200,
      "updated_at": 1640995200
    }
  ],
  "has_more": false,
  "limit": 20,
  "page": 1,
  "total": 1
}
```

#### 创建应用

**接口地址**: `POST /console/api/apps`

**功能说明**: 创建新的应用

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| name | string | 是 | 应用名称 |
| mode | string | 是 | 应用模式：chat/completion/workflow |
| description | string | 否 | 应用描述 |
| icon | string | 否 | 应用图标 |
| icon_background | string | 否 | 图标背景色 |

**请求示例**:

```bash
curl -X POST 'https://api.dify.ai/console/api/apps' \
--header 'Cookie: session={session-id}' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "新的聊天应用",
    "mode": "chat",
    "description": "这是一个新的聊天应用",
    "icon": "🤖",
    "icon_background": "#FFEAD5"
}'
```

**响应示例**:

```json
{
  "id": "app-456",
  "name": "新的聊天应用",
  "description": "这是一个新的聊天应用",
  "mode": "chat",
  "status": "normal",
  "enable_site": false,
  "enable_api": false,
  "icon": "🤖",
  "icon_background": "#FFEAD5",
  "created_at": 1640995200
}
```

### 3.2 数据集管理接口

#### 获取数据集列表

**接口地址**: `GET /console/api/datasets`

**功能说明**: 获取数据集列表

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| page | integer | 否 | 页码，默认1 |
| limit | integer | 否 | 每页数量，默认20 |
| search | string | 否 | 搜索关键词 |

**请求示例**:

```bash
curl -X GET 'https://api.dify.ai/console/api/datasets?page=1&limit=20' \
--header 'Cookie: session={session-id}'
```

**响应示例**:

```json
{
  "data": [
    {
      "id": "dataset-123",
      "name": "产品知识库",
      "description": "包含所有产品相关信息",
      "provider": "vendor",
      "permission": "only_me",
      "data_source_type": "upload_file",
      "indexing_technique": "high_quality",
      "app_count": 3,
      "document_count": 25,
      "word_count": 50000,
      "created_by": "user-123",
      "created_at": 1640995200,
      "updated_at": 1640995200
    }
  ],
  "has_more": false,
  "limit": 20,
  "page": 1,
  "total": 1
}
```

#### 创建数据集

**接口地址**: `POST /console/api/datasets`

**功能说明**: 创建新的数据集

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| name | string | 是 | 数据集名称 |
| description | string | 否 | 数据集描述 |
| indexing_technique | string | 是 | 索引技术：high_quality/economy |
| permission | string | 是 | 权限：only_me/all_team_members |

**请求示例**:

```bash
curl -X POST 'https://api.dify.ai/console/api/datasets' \
--header 'Cookie: session={session-id}' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "新知识库",
    "description": "用于存储客服相关文档",
    "indexing_technique": "high_quality",
    "permission": "all_team_members"
}'
```

**响应示例**:

```json
{
  "id": "dataset-456",
  "name": "新知识库",
  "description": "用于存储客服相关文档",
  "provider": "vendor",
  "permission": "all_team_members",
  "indexing_technique": "high_quality",
  "created_at": 1640995200
}
```

### 3.3 文档管理接口

#### 上传文档

**接口地址**: `POST /console/api/datasets/{dataset_id}/document/create_by_file`

**功能说明**: 向数据集上传文档

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| data | string | 是 | 文档处理规则JSON字符串 |
| file | file | 是 | 要上传的文件 |

**请求示例**:

```bash
curl -X POST 'https://api.dify.ai/console/api/datasets/dataset-123/document/create_by_file' \
--header 'Cookie: session={session-id}' \
--form 'data="{\"name\":\"产品手册\",\"text_preprocessing_rule\":{\"mode\":\"automatic\"},\"indexing_technique\":\"high_quality\"}"' \
--form 'file=@"/path/to/manual.pdf"'
```

**响应示例**:

```json
{
  "document": {
    "id": "doc-123",
    "name": "产品手册",
    "character_count": 5000,
    "tokens": 1200,
    "indexing_status": "parsing",
    "processing_started_at": 1640995200,
    "parsing_completed_at": null,
    "cleaning_completed_at": null,
    "splitting_completed_at": null,
    "completed_at": null,
    "paused_at": null,
    "error": null,
    "stopped_at": null,
    "indexing_latency": null,
    "created_at": 1640995200,
    "updated_at": 1640995200
  },
  "batch": "batch-123"
}
```

## 4. Web API详解

### 4.1 用户认证接口

#### 用户登录

**接口地址**: `POST /api/login`

**功能说明**: 用户登录获取访问令牌

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| email | string | 是 | 用户邮箱 |
| password | string | 是 | 用户密码 |
| remember_me | boolean | 否 | 是否记住登录状态 |

**请求示例**:

```bash
curl -X POST 'https://api.dify.ai/api/login' \
--header 'Content-Type: application/json' \
--data-raw '{
    "email": "user@example.com",
    "password": "password123",
    "remember_me": true
}'
```

**响应示例**:

```json
{
  "result": "success",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "expires_in": 3600
  }
}
```

### 4.2 对话管理接口

#### 获取对话列表

**接口地址**: `GET /api/conversations`

**功能说明**: 获取用户的对话列表

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| app_id | string | 是 | 应用ID |
| page | integer | 否 | 页码，默认1 |
| limit | integer | 否 | 每页数量，默认20 |

**请求示例**:

```bash
curl -X GET 'https://api.dify.ai/api/conversations?app_id=app-123&page=1&limit=20' \
--header 'Authorization: Bearer {user-token}'
```

**响应示例**:

```json
{
  "data": [
    {
      "id": "conv-123",
      "name": "关于产品的咨询",
      "inputs": {"name": "张三"},
      "status": "normal",
      "introduction": "你好，我想了解产品信息",
      "created_at": 1640995200,
      "updated_at": 1640995200
    }
  ],
  "has_more": false,
  "limit": 20,
  "page": 1,
  "total": 1
}
```

#### 重命名对话

**接口地址**: `POST /api/conversations/{conversation_id}/name`

**功能说明**: 重命名指定对话

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| name | string | 是 | 新的对话名称 |

**请求示例**:

```bash
curl -X POST 'https://api.dify.ai/api/conversations/conv-123/name' \
--header 'Authorization: Bearer {user-token}' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "产品咨询对话"
}'
```

**响应示例**:

```json
{
  "result": "success",
  "data": {
    "id": "conv-123",
    "name": "产品咨询对话",
    "updated_at": 1640995300
  }
}
```

## 5. 错误处理

### 5.1 错误响应格式

所有API错误都采用统一的响应格式：

```json
{
  "code": "invalid_param",
  "message": "参数验证失败",
  "status": 400
}
```

### 5.2 常见错误码

| 错误码 | HTTP状态码 | 说明 |
|--------|------------|------|
| invalid_api_key | 401 | API密钥无效或已过期 |
| unauthorized | 401 | 未授权访问 |
| forbidden | 403 | 权限不足 |
| not_found | 404 | 资源不存在 |
| invalid_param | 400 | 参数验证失败 |
| rate_limit_exceeded | 429 | 请求频率超限 |
| quota_exceeded | 429 | 配额已用完 |
| app_unavailable | 503 | 应用不可用 |
| provider_not_initialize | 503 | 模型提供商未初始化 |
| model_currently_not_support | 503 | 模型当前不支持 |
| completion_request_error | 500 | 补全请求错误 |
| internal_server_error | 500 | 内部服务器错误 |

### 5.3 错误处理最佳实践

```javascript
// JavaScript错误处理示例
async function callDifyAPI(endpoint, options) {
  try {
    const response = await fetch(endpoint, {
      ...options,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        ...options.headers
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      
      switch (errorData.code) {
        case 'rate_limit_exceeded':
          // 实现退避重试
          await new Promise(resolve => setTimeout(resolve, 1000));
          return callDifyAPI(endpoint, options);
          
        case 'invalid_api_key':
          // 刷新API密钥
          throw new Error('API密钥无效，请检查配置');
          
        case 'quota_exceeded':
          // 处理配额超限
          throw new Error('API配额已用完，请升级套餐');
          
        default:
          throw new Error(`API调用失败: ${errorData.message}`);
      }
    }

    return await response.json();
    
  } catch (error) {
    console.error('API调用错误:', error);
    throw error;
  }
}
```

## 6. SDK和代码示例

### 6.1 Python SDK示例

```python
import requests
import json
from typing import Generator, Dict, Any

class DifyClient:
    """Dify API客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.dify.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_messages(
        self, 
        inputs: Dict[str, Any],
        query: str,
        user: str,
        conversation_id: str = "",
        response_mode: str = "streaming",
        files: list = None
    ) -> Generator[Dict[str, Any], None, None]:
        """发送聊天消息"""
        
        url = f"{self.base_url}/v1/chat-messages"
        data = {
            "inputs": inputs,
            "query": query,
            "response_mode": response_mode,
            "conversation_id": conversation_id,
            "user": user,
            "files": files or []
        }
        
        if response_mode == "streaming":
            # 流式响应
            response = requests.post(
                url, 
                headers=self.headers, 
                json=data, 
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            yield data
                        except json.JSONDecodeError:
                            continue
        else:
            # 阻塞响应
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            yield response.json()
    
    def completions(
        self,
        inputs: Dict[str, Any],
        user: str,
        response_mode: str = "blocking",
        files: list = None
    ) -> Dict[str, Any]:
        """创建文本补全"""
        
        url = f"{self.base_url}/v1/completions"
        data = {
            "inputs": inputs,
            "response_mode": response_mode,
            "user": user,
            "files": files or []
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def upload_file(self, file_path: str, user: str) -> Dict[str, Any]:
        """上传文件"""
        
        url = f"{self.base_url}/v1/files/upload"
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'user': user}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json()

# 使用示例
client = DifyClient(api_key="your-api-key")

# 聊天对话
for chunk in client.chat_messages(
    inputs={"name": "张三"},
    query="你好，请介绍一下你的功能",
    user="user-123"
):
    if chunk.get("event") == "message":
        print(chunk.get("answer", ""), end="", flush=True)
    elif chunk.get("event") == "message_end":
        print(f"\n使用令牌: {chunk.get('metadata', {}).get('usage', {})}")
```

### 6.2 JavaScript SDK示例

```javascript
class DifyClient {
  constructor(apiKey, baseUrl = 'https://api.dify.ai') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    };
  }

  async *chatMessages(options) {
    const {
      inputs,
      query,
      user,
      conversationId = '',
      responseMode = 'streaming',
      files = []
    } = options;

    const url = `${this.baseUrl}/v1/chat-messages`;
    const data = {
      inputs,
      query,
      response_mode: responseMode,
      conversation_id: conversationId,
      user,
      files
    };

    if (responseMode === 'streaming') {
      const response = await fetch(url, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                yield data;
              } catch (e) {
                // 忽略解析错误
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } else {
      const response = await fetch(url, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      yield await response.json();
    }
  }

  async completions(options) {
    const {
      inputs,
      user,
      responseMode = 'blocking',
      files = []
    } = options;

    const url = `${this.baseUrl}/v1/completions`;
    const data = {
      inputs,
      response_mode: responseMode,
      user,
      files
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async uploadFile(file, user) {
    const url = `${this.baseUrl}/v1/files/upload`;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user', user);

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
}

// 使用示例
const client = new DifyClient('your-api-key');

// 聊天对话
async function chat() {
  try {
    for await (const chunk of client.chatMessages({
      inputs: { name: '张三' },
      query: '你好，请介绍一下你的功能',
      user: 'user-123'
    })) {
      if (chunk.event === 'message') {
        process.stdout.write(chunk.answer || '');
      } else if (chunk.event === 'message_end') {
        console.log(`\n使用令牌: ${JSON.stringify(chunk.metadata?.usage)}`);
      }
    }
  } catch (error) {
    console.error('聊天错误:', error);
  }
}

chat();
```

## 7. 最佳实践

### 7.1 API调用优化

#### 连接池管理

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OptimizedDifyClient:
    """优化的Dify客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.dify.ai"):
        self.api_key = api_key
        self.base_url = base_url
        
        # 创建会话
        self.session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # 配置适配器
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置默认头部
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DifyClient/1.0"
        })
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
```

#### 缓存策略

```python
import hashlib
import json
from functools import wraps
from typing import Dict, Any, Optional

class APICache:
    """API响应缓存"""
    
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
    
    def get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        content = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """设置缓存"""
        self.cache[key] = (value, time.time())

def cached_api_call(cache: APICache, ttl: int = 3600):
    """API调用缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 生成缓存键
            cache_key = cache.get_cache_key(func.__name__, kwargs)
            
            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行API调用
            result = func(self, *args, **kwargs)
            
            # 设置缓存
            cache.set(cache_key, result)
            
            return result
        return wrapper
    return decorator
```

### 7.2 错误处理和重试

```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries:
                        raise e
                    
                    # 计算延迟时间
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    
                    # 特殊处理429错误
                    if hasattr(e, 'response') and e.response.status_code == 429:
                        retry_after = e.response.headers.get('Retry-After')
                        if retry_after:
                            delay = max(delay, int(retry_after))
                    
                    time.sleep(delay)
            
        return wrapper
    return decorator

class RobustDifyClient(DifyClient):
    """健壮的Dify客户端"""
    
    @retry_with_backoff(max_retries=3)
    def chat_messages(self, *args, **kwargs):
        """带重试的聊天消息"""
        return super().chat_messages(*args, **kwargs)
    
    @retry_with_backoff(max_retries=3)
    def completions(self, *args, **kwargs):
        """带重试的文本补全"""
        return super().completions(*args, **kwargs)
```

### 7.3 监控和日志

```python
import logging
import time
from contextlib import contextmanager

class APIMonitor:
    """API监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger('dify_api')
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0
        }
    
    @contextmanager
    def monitor_request(self, endpoint: str, params: dict):
        """监控API请求"""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            self.logger.info(f"API请求开始: {endpoint}", extra={
                'endpoint': endpoint,
                'params': params
            })
            
            yield
            
            # 请求成功
            response_time = time.time() - start_time
            self.metrics['successful_requests'] += 1
            self.metrics['total_response_time'] += response_time
            
            self.logger.info(f"API请求成功: {endpoint}", extra={
                'endpoint': endpoint,
                'response_time': response_time
            })
            
        except Exception as e:
            # 请求失败
            response_time = time.time() - start_time
            self.metrics['failed_requests'] += 1
            
            self.logger.error(f"API请求失败: {endpoint}", extra={
                'endpoint': endpoint,
                'error': str(e),
                'response_time': response_time
            })
            
            raise
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total = self.metrics['total_requests']
        if total == 0:
            return self.metrics
        
        return {
            **self.metrics,
            'success_rate': self.metrics['successful_requests'] / total,
            'failure_rate': self.metrics['failed_requests'] / total,
            'average_response_time': self.metrics['total_response_time'] / total
        }

class MonitoredDifyClient(DifyClient):
    """带监控的Dify客户端"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = APIMonitor()
    
    def chat_messages(self, *args, **kwargs):
        """带监控的聊天消息"""
        with self.monitor.monitor_request('chat_messages', kwargs):
            return super().chat_messages(*args, **kwargs)
```

## 8. 总结

本API参考手册提供了Dify平台的完整接口文档，包括：

### 8.1 核心特性

1. **三层API架构**：Service API、Console API、Web API
2. **统一认证机制**：API Key、Session、Token认证
3. **流式响应支持**：实时流式输出和阻塞响应
4. **完整错误处理**：统一错误格式和错误码
5. **多语言SDK**：Python、JavaScript等语言支持

### 8.2 最佳实践

1. **连接池管理**：提高API调用性能
2. **缓存策略**：减少重复请求
3. **重试机制**：处理网络异常和限流
4. **监控日志**：跟踪API使用情况
5. **错误处理**：优雅处理各种异常情况

### 8.3 开发建议

1. **使用SDK**：推荐使用官方或社区SDK
2. **遵循限制**：注意API频率和配额限制
3. **错误处理**：实现完善的错误处理逻辑
4. **性能优化**：使用连接池和缓存机制
5. **监控告警**：建立API调用监控体系

通过这套完整的API参考手册，开发者可以快速集成Dify平台的各种功能，构建强大的AI应用。

---

*最后更新时间：2025-01-27*  
*文档版本：v1.0*  
*维护者：Dify API团队*
