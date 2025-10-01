---
title: "VoiceHelper 框架使用示例与集成指南"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['源码分析', '技术文档']
categories: ['AI语音助手']
description: "VoiceHelper 框架使用示例与集成指南的深入技术分析文档"
keywords: ['源码分析', '技术文档']
author: "技术分析师"
weight: 1
---

## 📋 概述

本文档提供了VoiceHelper框架的详细使用示例，涵盖快速部署、SDK集成、多平台开发和实际应用场景，帮助开发者快速上手并深度集成VoiceHelper平台。

## 🚀 快速开始

### 1. Docker一键部署

```bash
# 克隆项目
git clone https://github.com/voicehelper/voicehelper.git
cd voicehelper

# 配置环境变量
cp shared/configs/env.example .env

# 编辑环境配置
vim .env
```

#### 环境配置示例

```bash
# .env 配置文件
# 基础配置
NODE_ENV=production
LOG_LEVEL=info

# 数据库配置
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=voicehelper
POSTGRES_USER=voicehelper_user
POSTGRES_PASSWORD=your_secure_password

# Redis配置
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Neo4j配置（图数据库）
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# MinIO配置（对象存储）
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=your_minio_password

# AI服务API密钥
OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_key

# 服务端口配置
BACKEND_PORT=8080
ALGO_PORT=8070
FRONTEND_PORT=3000

# JWT配置
JWT_SECRET=your_jwt_secret_key
JWT_EXPIRES_IN=24h

# 文件上传配置
MAX_FILE_SIZE=100MB
ALLOWED_FILE_TYPES=pdf,docx,txt,md,html
```

#### 启动服务

```bash
# 生产环境部署
docker-compose -f docker-compose.prod.yml up -d

# 开发环境部署
docker-compose -f docker-compose.dev.yml up -d

# 查看服务状态
docker-compose ps

# 实时查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
docker-compose logs -f algo
docker-compose logs -f frontend
```

#### 健康检查

```bash
# 检查后端服务
curl http://localhost:8080/health

# 检查算法服务
curl http://localhost:8070/health

# 检查前端服务
curl http://localhost:3000
```

### 2. 开发环境手动搭建

#### Go后端服务

```bash
cd backend

# 安装依赖
go mod download

# 设置开发环境变量
export GO_ENV=development
export DB_HOST=localhost
export DB_PORT=5432
export REDIS_HOST=localhost
export REDIS_PORT=6379

# 运行数据库迁移
make migrate-up

# 启动开发服务器
make dev

# 或者直接运行
go run cmd/gateway/main.go
```

#### Python算法服务

```bash
cd algo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export PYTHONPATH=${PWD}
export ALGO_ENV=development

# 启动开发服务器
python app/main.py

# 或使用uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8070 --reload
```

#### Next.js前端服务

```bash
cd platforms/web

# 安装依赖
npm install

# 设置环境变量
cp .env.example .env.local
# 编辑 .env.local 文件

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
npm start
```

## 📱 SDK集成示例

### JavaScript/TypeScript SDK

#### 安装和初始化

```bash
npm install @voicehelper/javascript-sdk
```

```typescript
import { VoiceHelperClient, VoiceHelperConfig } from '@voicehelper/javascript-sdk';

const config: VoiceHelperConfig = {
  apiKey: 'your-api-key',
  baseURL: 'https://api.voicehelper.ai',
  timeout: 30000,
  retryAttempts: 3,
  enableLogging: true
};

const client = new VoiceHelperClient(config);
```

#### 基础聊天功能

```typescript
// 简单文本聊天
async function simpleChat() {
  try {
    const response = await client.chat.send({
      message: '你好，我想了解人工智能的基本概念',
      conversation_id: 'conv_123',
      model: 'gpt-3.5-turbo'
    });
    
    console.log('AI回复:', response.data.message);
    console.log('参考资料:', response.data.references);
  } catch (error) {
    console.error('聊天失败:', error);
  }
}

// 流式聊天
async function streamingChat() {
  try {
    const stream = await client.chat.createStream({
      message: '请详细介绍机器学习的发展历程',
      conversation_id: 'conv_456',
      retrieval_config: {
        mode: 'hybrid',
        top_k: 10,
        collection: 'ai_knowledge'
      }
    });

    console.log('开始接收回复...');
    
    for await (const chunk of stream) {
      switch (chunk.type) {
        case 'retrieval_start':
          console.log('🔍 开始检索相关资料...');
          break;
          
        case 'retrieval_result':
          console.log(`📚 找到 ${chunk.data.results.length} 条相关资料`);
          break;
          
        case 'generation_start':
          console.log('🤖 AI开始生成回复...');
          break;
          
        case 'generation_chunk':
          process.stdout.write(chunk.data.text);
          break;
          
        case 'generation_done':
          console.log(`\n✅ 回复完成，耗时: ${chunk.data.total_time_ms}ms`);
          console.log('📖 参考资料:', chunk.data.context_sources);
          break;
          
        case 'error':
          console.error('❌ 错误:', chunk.data.error);
          break;
      }
    }
  } catch (error) {
    console.error('流式聊天失败:', error);
  }
}
```

#### 文档管理功能

```typescript
// 文档上传入库
async function uploadDocuments() {
  try {
    const files = [
      {
        filename: 'company_handbook.pdf',
        content: await fs.readFile('path/to/handbook.pdf'),
        contentType: 'application/pdf'
      },
      {
        filename: 'product_guide.docx',
        content: await fs.readFile('path/to/guide.docx'),
        contentType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
      }
    ];

    const ingestResult = await client.documents.ingest({
      files,
      collection_name: 'company_knowledge',
      chunk_size: 1000,
      chunk_overlap: 200,
      metadata: {
        department: 'HR',
        category: 'policy',
        version: '2024.1'
      }
    });

    console.log('入库任务ID:', ingestResult.task_id);

    // 查询入库进度
    const interval = setInterval(async () => {
      try {
        const status = await client.documents.getTaskStatus(ingestResult.task_id);
        console.log(`处理进度: ${status.progress}% - ${status.status}`);

        if (status.status === 'completed') {
          console.log('✅ 文档入库完成!');
          console.log('处理结果:', status.result);
          clearInterval(interval);
        } else if (status.status === 'failed') {
          console.error('❌ 文档入库失败:', status.error);
          clearInterval(interval);
        }
      } catch (error) {
        console.error('查询状态失败:', error);
        clearInterval(interval);
      }
    }, 2000);

  } catch (error) {
    console.error('文档上传失败:', error);
  }
}

// 文档搜索
async function searchDocuments() {
  try {
    const searchResult = await client.documents.search({
      query: '请假政策',
      collection_name: 'company_knowledge',
      top_k: 5,
      filters: {
        department: 'HR',
        category: 'policy'
      }
    });

    console.log(`找到 ${searchResult.total_found} 条相关文档:`);
    searchResult.results.forEach((result, index) => {
      console.log(`${index + 1}. ${result.title}`);
      console.log(`   相关度: ${(result.score * 100).toFixed(1)}%`);
      console.log(`   摘要: ${result.content.substring(0, 100)}...`);
    });
  } catch (error) {
    console.error('文档搜索失败:', error);
  }
}
```

#### 语音交互功能

```typescript
// 语音聊天类
class VoiceChat {
  private client: VoiceHelperClient;
  private ws: WebSocket | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;

  constructor(client: VoiceHelperClient) {
    this.client = client;
  }

  async startVoiceSession(options?: {
    language?: string;
    voice_id?: string;
    conversation_id?: string;
  }) {
    try {
      // 建立WebSocket连接
      this.ws = await this.client.voice.connect({
        language: options?.language || 'zh-CN',
        voice_id: options?.voice_id || 'zh-CN-XiaoxiaoNeural',
        conversation_id: options?.conversation_id
      });

      // 监听语音响应
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleVoiceMessage(data);
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket错误:', error);
      };

      this.ws.onclose = () => {
        console.log('语音会话已结束');
        this.cleanup();
      };

      console.log('✅ 语音会话已建立');
    } catch (error) {
      console.error('建立语音会话失败:', error);
    }
  }

  private handleVoiceMessage(data: any) {
    switch (data.type) {
      case 'session_initialized':
        console.log('🎙️ 语音会话初始化完成:', data.session_id);
        break;

      case 'asr_partial':
        // 实时显示语音识别结果
        this.updateTranscript(data.text, false);
        break;

      case 'asr_final':
        // 最终语音识别结果
        console.log('🎯 识别完成:', data.text);
        this.updateTranscript(data.text, true);
        break;

      case 'processing_start':
        console.log('🤔 AI正在思考...');
        break;

      case 'llm_response_chunk':
        // 实时显示AI文本回复
        this.displayResponse(data.text, false);
        break;

      case 'llm_response_final':
        console.log('💬 AI回复完成');
        this.displayResponse(data.text, true);
        if (data.references) {
          console.log('📚 参考资料:', data.references);
        }
        break;

      case 'tts_start':
        console.log('🔊 开始语音合成...');
        break;

      case 'tts_audio':
        // 播放TTS音频
        this.playAudio(data.audio_data, data.audio_format);
        break;

      case 'tts_complete':
        console.log('🎵 语音播放完成');
        break;

      case 'error':
        console.error('❌ 语音处理错误:', data.error);
        break;
    }
  }

  async startRecording() {
    try {
      // 请求麦克风权限
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      // 创建录音器
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 64000
      });

      // 处理录音数据
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && this.ws) {
          this.sendAudioData(event.data);
        }
      };

      // 开始录音，每100ms发送一次数据
      this.mediaRecorder.start(100);
      console.log('🎙️ 开始录音...');

    } catch (error) {
      console.error('启动录音失败:', error);
    }
  }

  stopRecording() {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.mediaRecorder = null;
      console.log('⏹️ 录音已停止');
    }
  }

  private async sendAudioData(audioBlob: Blob) {
    try {
      const arrayBuffer = await audioBlob.arrayBuffer();
      const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'audio_chunk',
          audio_chunk: base64,
          timestamp: Date.now()
        }));
      }
    } catch (error) {
      console.error('发送音频数据失败:', error);
    }
  }

  private async playAudio(base64Audio: string, format: string) {
    try {
      // 解码base64音频
      const audioData = atob(base64Audio);
      const arrayBuffer = new ArrayBuffer(audioData.length);
      const view = new Uint8Array(arrayBuffer);
      
      for (let i = 0; i < audioData.length; i++) {
        view[i] = audioData.charCodeAt(i);
      }

      // 创建音频上下文
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      }

      // 解码并播放音频
      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audioContext.destination);
      source.start();

    } catch (error) {
      console.error('音频播放失败:', error);
    }
  }

  private updateTranscript(text: string, isFinal: boolean) {
    const element = document.getElementById('transcript');
    if (element) {
      element.textContent = text;
      element.className = isFinal ? 'final' : 'partial';
    }
  }

  private displayResponse(text: string, isFinal: boolean) {
    const element = document.getElementById('response');
    if (element) {
      if (isFinal) {
        element.textContent = text;
      } else {
        element.textContent += text;
      }
    }
  }

  private cleanup() {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
    }
    if (this.audioContext) {
      this.audioContext.close();
    }
    if (this.ws) {
      this.ws.close();
    }
  }
}

// 使用示例
const voiceChat = new VoiceChat(client);

// 按钮事件
document.getElementById('start-voice')?.addEventListener('click', () => {
  voiceChat.startVoiceSession({
    language: 'zh-CN',
    conversation_id: 'voice_conv_123'
  });
});

document.getElementById('start-recording')?.addEventListener('click', () => {
  voiceChat.startRecording();
});

document.getElementById('stop-recording')?.addEventListener('click', () => {
  voiceChat.stopRecording();
});
```

### Python SDK集成

#### 安装和基础配置

```bash
pip install voicehelper-sdk
```

```python
from voicehelper_sdk import VoiceHelperClient, VoiceHelperConfig
import asyncio
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 客户端配置
config = VoiceHelperConfig(
    api_key="your-api-key",
    base_url="https://api.voicehelper.ai",
    timeout=30.0,
    max_retries=3,
    enable_logging=True
)

client = VoiceHelperClient(config)
```

#### 异步聊天功能

```python
async def async_chat_example():
    """异步聊天示例"""
    try:
        # 简单聊天
        response = await client.chat.send_message(
            message="请介绍一下深度学习的基本原理",
            conversation_id="conv_python_123",
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        print("AI回复:", response.message)
        print("响应时间:", response.response_time_ms, "ms")
        print("使用模型:", response.model_used)
        
        if response.references:
            print("\n参考资料:")
            for i, ref in enumerate(response.references, 1):
                print(f"{i}. {ref.title}")
                print(f"   来源: {ref.source}")
                print(f"   相关度: {ref.relevance_score:.2f}")
        
    except Exception as e:
        logger.error(f"聊天失败: {e}")

# 流式聊天
async def streaming_chat_example():
    """流式聊天示例"""
    try:
        stream = client.chat.create_stream(
            message="详细解释什么是Transformer架构",
            conversation_id="conv_stream_456", 
            retrieval_config={
                "mode": "hybrid",
                "top_k": 8,
                "collection": "ai_papers"
            }
        )
        
        print("开始接收流式回复...")
        full_response = ""
        
        async for chunk in stream:
            if chunk.type == "retrieval_start":
                print("🔍 开始检索...")
                
            elif chunk.type == "retrieval_result":
                results = chunk.data.get("results", [])
                print(f"📚 检索到 {len(results)} 条相关文档")
                
            elif chunk.type == "generation_start":
                print("🤖 开始生成回复...")
                
            elif chunk.type == "generation_chunk":
                text = chunk.data.get("text", "")
                print(text, end="", flush=True)
                full_response += text
                
            elif chunk.type == "generation_done":
                print(f"\n\n✅ 生成完成!")
                print(f"总耗时: {chunk.data.get('total_time_ms')}ms")
                print(f"生成token数: {chunk.data.get('token_count', 0)}")
                
            elif chunk.type == "error":
                print(f"❌ 错误: {chunk.data.get('error')}")
                break
                
        print(f"\n完整回复长度: {len(full_response)} 字符")
        
    except Exception as e:
        logger.error(f"流式聊天失败: {e}")

# 批量文档处理
async def batch_document_processing():
    """批量文档处理示例"""
    try:
        # 准备文档文件
        documents = []
        
        # 从文件系统读取文档
        import os
        doc_folder = "path/to/documents"
        
        for filename in os.listdir(doc_folder):
            if filename.endswith(('.pdf', '.docx', '.txt')):
                file_path = os.path.join(doc_folder, filename)
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                documents.append({
                    "filename": filename,
                    "content": content,
                    "metadata": {
                        "source": "company_docs",
                        "upload_date": "2024-01-15",
                        "department": "product"
                    }
                })
        
        print(f"准备处理 {len(documents)} 个文档...")
        
        # 批量入库
        ingest_result = await client.documents.batch_ingest(
            files=documents,
            collection_name="company_knowledge_v2",
            chunk_size=1200,
            chunk_overlap=150,
            processing_options={
                "enable_ocr": True,
                "extract_images": True,
                "enable_table_extraction": True
            }
        )
        
        print(f"批量入库任务已启动: {ingest_result.task_id}")
        
        # 监控处理进度
        while True:
            status = await client.documents.get_task_status(ingest_result.task_id)
            
            print(f"进度: {status.progress}% - {status.status}")
            print(f"已处理: {status.processed_files}/{status.total_files} 个文件")
            
            if status.status == "completed":
                print("🎉 批量处理完成!")
                result = status.result
                print(f"成功处理: {result.get('documents_processed', 0)} 个文档")
                print(f"生成分块: {result.get('chunks_created', 0)} 个")
                print(f"索引向量: {result.get('vectors_indexed', 0)} 个")
                print(f"处理时间: {result.get('processing_time_seconds', 0)} 秒")
                break
                
            elif status.status == "failed":
                print(f"❌ 批量处理失败: {status.error}")
                break
                
            elif status.status == "processing":
                # 显示详细进度
                if hasattr(status, 'detailed_progress'):
                    for file_progress in status.detailed_progress:
                        print(f"  - {file_progress['filename']}: {file_progress['stage']}")
            
            await asyncio.sleep(3)  # 每3秒检查一次
            
    except Exception as e:
        logger.error(f"批量文档处理失败: {e}")

# 智能问答系统
class IntelligentQASystem:
    """智能问答系统"""
    
    def __init__(self, client: VoiceHelperClient):
        self.client = client
        self.conversation_history = {}
    
    async def answer_question(
        self,
        question: str,
        user_id: str,
        context: dict = None
    ) -> dict:
        """回答问题"""
        try:
            # 获取或创建对话历史
            conversation_id = f"qa_{user_id}"
            
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
            
            # 构建增强的问题上下文
            enhanced_question = self._build_enhanced_question(
                question, 
                self.conversation_history[conversation_id],
                context
            )
            
            # 调用VoiceHelper进行问答
            response = await self.client.chat.send_message(
                message=enhanced_question,
                conversation_id=conversation_id,
                retrieval_config={
                    "mode": "graph",  # 使用图检索模式
                    "top_k": 5,
                    "enable_reasoning": True,
                    "collection": "knowledge_base"
                },
                generation_config={
                    "temperature": 0.3,
                    "max_tokens": 1000,
                    "enable_citations": True
                }
            )
            
            # 保存对话历史
            self.conversation_history[conversation_id].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response.message}
            ])
            
            # 限制历史长度
            if len(self.conversation_history[conversation_id]) > 20:
                self.conversation_history[conversation_id] = \
                    self.conversation_history[conversation_id][-20:]
            
            # 构建结构化回复
            structured_response = {
                "answer": response.message,
                "confidence": response.confidence_score,
                "sources": [
                    {
                        "title": ref.title,
                        "content": ref.content[:200] + "..." if len(ref.content) > 200 else ref.content,
                        "relevance": ref.relevance_score,
                        "source": ref.source
                    }
                    for ref in response.references[:3]
                ],
                "response_time_ms": response.response_time_ms,
                "follow_up_questions": self._generate_follow_up_questions(question, response.message)
            }
            
            return structured_response
            
        except Exception as e:
            logger.error(f"问答失败: {e}")
            return {
                "answer": "抱歉，我暂时无法回答这个问题，请稍后重试。",
                "error": str(e)
            }
    
    def _build_enhanced_question(
        self, 
        question: str, 
        history: list,
        context: dict = None
    ) -> str:
        """构建增强问题上下文"""
        context_parts = [question]
        
        # 添加对话历史上下文
        if history:
            recent_history = history[-4:]  # 最近2轮对话
            history_context = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_history
            ])
            context_parts.append(f"对话历史:\n{history_context}")
        
        # 添加额外上下文
        if context:
            context_info = []
            for key, value in context.items():
                context_info.append(f"{key}: {value}")
            context_parts.append(f"上下文信息:\n" + "\n".join(context_info))
        
        return "\n\n".join(context_parts)
    
    def _generate_follow_up_questions(self, original_question: str, answer: str) -> list:
        """生成相关问题建议"""
        # 这里可以集成更复杂的问题生成逻辑
        follow_ups = [
            "能否提供更多细节？",
            "有什么实际应用案例吗？", 
            "还有其他相关的概念吗？"
        ]
        return follow_ups

# 使用示例
async def main():
    """主函数示例"""
    try:
        # 基础聊天
        await async_chat_example()
        
        # 流式聊天
        await streaming_chat_example()
        
        # 批量文档处理
        await batch_document_processing()
        
        # 智能问答系统
        qa_system = IntelligentQASystem(client)
        
        result = await qa_system.answer_question(
            question="什么是GraphRAG？它与传统RAG有什么区别？",
            user_id="user_123",
            context={
                "domain": "AI技术",
                "expertise_level": "intermediate"
            }
        )
        
        print("\n智能问答结果:")
        print(f"回答: {result['answer']}")
        print(f"置信度: {result['confidence']}")
        print(f"响应时间: {result['response_time_ms']}ms")
        print("参考来源:")
        for source in result['sources']:
            print(f"  - {source['title']} (相关度: {source['relevance']})")
        
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🌐 多平台集成示例

### React Web应用集成

```jsx
// React Hook集成示例
import React, { useState, useEffect, useCallback } from 'react';
import { VoiceHelperClient } from '@voicehelper/javascript-sdk';

// 自定义Hook
const useVoiceHelper = (apiKey) => {
  const [client, setClient] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initClient = async () => {
      try {
        const voiceHelperClient = new VoiceHelperClient({
          apiKey,
          baseURL: process.env.REACT_APP_VOICEHELPER_API_URL
        });

        await voiceHelperClient.connect();
        setClient(voiceHelperClient);
        setIsConnected(true);
        setError(null);
      } catch (err) {
        setError(err.message);
        setIsConnected(false);
      }
    };

    if (apiKey) {
      initClient();
    }
  }, [apiKey]);

  return { client, isConnected, error };
};

// 聊天组件
const ChatComponent = ({ apiKey, conversationId }) => {
  const { client, isConnected, error } = useVoiceHelper(apiKey);
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = useCallback(async () => {
    if (!client || !inputText.trim()) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputText,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const stream = await client.chat.createStream({
        message: inputText,
        conversation_id: conversationId,
        retrieval_config: {
          mode: 'hybrid',
          top_k: 5
        }
      });

      let aiMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: '',
        references: [],
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);

      for await (const chunk of stream) {
        if (chunk.type === 'generation_chunk') {
          aiMessage.content += chunk.data.text;
          setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = { ...aiMessage };
            return updated;
          });
        } else if (chunk.type === 'retrieval_result') {
          aiMessage.references = chunk.data.results;
        }
      }
    } catch (err) {
      console.error('发送消息失败:', err);
    } finally {
      setIsLoading(false);
    }
  }, [client, inputText, conversationId]);

  if (error) {
    return <div className="error">连接失败: {error}</div>;
  }

  if (!isConnected) {
    return <div className="loading">连接中...</div>;
  }

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map(message => (
          <div key={message.id} className={`message ${message.role}`}>
            <div className="content">{message.content}</div>
            {message.references && message.references.length > 0 && (
              <div className="references">
                <h4>参考资料:</h4>
                {message.references.map((ref, index) => (
                  <div key={index} className="reference">
                    <strong>{ref.title}</strong>
                    <p>{ref.content.substring(0, 100)}...</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      
      <div className="input-area">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="输入消息..."
          disabled={isLoading}
        />
        <button onClick={sendMessage} disabled={isLoading || !inputText.trim()}>
          {isLoading ? '发送中...' : '发送'}
        </button>
      </div>
    </div>
  );
};

// 主应用
const App = () => {
  const [apiKey, setApiKey] = useState('');
  const [conversationId] = useState(`conv_${Date.now()}`);

  return (
    <div className="app">
      <header>
        <h1>VoiceHelper 聊天示例</h1>
        <input
          type="text"
          placeholder="输入API密钥"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
        />
      </header>
      
      {apiKey && (
        <ChatComponent 
          apiKey={apiKey} 
          conversationId={conversationId}
        />
      )}
    </div>
  );
};

export default App;
```

### 微信小程序集成

```javascript
// 微信小程序集成示例
// app.js
App({
  globalData: {
    voiceHelperConfig: {
      apiKey: 'your-api-key',
      baseURL: 'https://api.voicehelper.ai',
      timeout: 30000
    },
    userInfo: null
  },

  onLaunch() {
    // 初始化VoiceHelper客户端
    this.initVoiceHelper();
    
    // 获取用户信息
    this.getUserInfo();
  },

  async initVoiceHelper() {
    try {
      const { VoiceHelperMiniProgram } = require('./utils/voicehelper-sdk.js');
      
      this.voiceHelper = new VoiceHelperMiniProgram(this.globalData.voiceHelperConfig);
      await this.voiceHelper.initialize();
      
      console.log('VoiceHelper初始化成功');
    } catch (error) {
      console.error('VoiceHelper初始化失败:', error);
      wx.showToast({
        title: '服务初始化失败',
        icon: 'none'
      });
    }
  },

  getUserInfo() {
    wx.getUserProfile({
      desc: '用于个性化服务',
      success: (res) => {
        this.globalData.userInfo = res.userInfo;
      },
      fail: (err) => {
        console.log('获取用户信息失败:', err);
      }
    });
  }
});

// pages/chat/chat.js
Page({
  data: {
    messages: [],
    inputText: '',
    isRecording: false,
    isConnected: false,
    currentTranscript: '',
    scrollIntoView: ''
  },

  onLoad(options) {
    this.conversationId = `miniprogram_${Date.now()}`;
    this.connectVoiceHelper();
  },

  async connectVoiceHelper() {
    try {
      const app = getApp();
      
      if (!app.voiceHelper) {
        await app.initVoiceHelper();
      }

      // 建立WebSocket连接
      await app.voiceHelper.connect({
        conversation_id: this.conversationId
      });

      this.setData({ isConnected: true });

      // 监听语音识别结果
      app.voiceHelper.onASRResult((result) => {
        this.setData({
          currentTranscript: result.text
        });

        if (result.is_final) {
          this.addMessage('user', result.text);
        }
      });

      // 监听AI回复
      app.voiceHelper.onAIResponse((response) => {
        this.addMessage('assistant', response.text, response.references);
        
        // 播放TTS音频
        if (response.audio_data) {
          this.playTTSAudio(response.audio_data);
        }
      });

      // 监听连接状态
      app.voiceHelper.onConnectionChange((status) => {
        this.setData({ isConnected: status === 'connected' });
      });

    } catch (error) {
      console.error('连接VoiceHelper失败:', error);
      wx.showToast({
        title: '连接服务失败',
        icon: 'none'
      });
    }
  },

  // 发送文本消息
  async sendTextMessage() {
    const message = this.data.inputText.trim();
    if (!message) return;

    this.addMessage('user', message);
    this.setData({ inputText: '' });

    try {
      const app = getApp();
      await app.voiceHelper.sendMessage(message);
    } catch (error) {
      console.error('发送消息失败:', error);
      wx.showToast({
        title: '发送失败',
        icon: 'none'
      });
    }
  },

  // 开始录音
  startRecording() {
    if (!this.data.isConnected) {
      wx.showToast({
        title: '请先连接服务',
        icon: 'none'
      });
      return;
    }

    wx.authorize({
      scope: 'scope.record',
      success: () => {
        this.recorderManager = wx.getRecorderManager();
        
        this.recorderManager.onStart(() => {
          this.setData({ 
            isRecording: true,
            currentTranscript: '正在录音...'
          });
        });

        this.recorderManager.onFrameRecorded((res) => {
          // 实时发送音频帧
          const app = getApp();
          app.voiceHelper.sendAudioFrame(res.frameBuffer);
        });

        this.recorderManager.onStop(() => {
          this.setData({ 
            isRecording: false,
            currentTranscript: ''
          });
        });

        // 开始录音
        this.recorderManager.start({
          duration: 60000,
          sampleRate: 16000,
          numberOfChannels: 1,
          encodeBitRate: 48000,
          format: 'mp3',
          frameSize: 50
        });
      },
      fail: () => {
        wx.showToast({
          title: '需要录音权限',
          icon: 'none'
        });
      }
    });
  },

  // 停止录音
  stopRecording() {
    if (this.recorderManager) {
      this.recorderManager.stop();
    }
  },

  // 播放TTS音频
  playTTSAudio(base64AudioData) {
    try {
      // 保存音频文件
      const fs = wx.getFileSystemManager();
      const audioPath = `${wx.env.USER_DATA_PATH}/tts_${Date.now()}.mp3`;
      
      fs.writeFile({
        filePath: audioPath,
        data: base64AudioData,
        encoding: 'base64',
        success: () => {
          // 播放音频
          const innerAudioContext = wx.createInnerAudioContext();
          innerAudioContext.src = audioPath;
          innerAudioContext.play();
          
          innerAudioContext.onEnded(() => {
            // 播放完成后删除临时文件
            fs.unlink({
              filePath: audioPath,
              success: () => console.log('临时音频文件已删除'),
              fail: (err) => console.warn('删除临时文件失败:', err)
            });
          });
        },
        fail: (err) => {
          console.error('保存音频文件失败:', err);
        }
      });
    } catch (error) {
      console.error('播放TTS音频失败:', error);
    }
  },

  // 添加消息
  addMessage(role, content, references = null) {
    const message = {
      id: Date.now(),
      role,
      content,
      references,
      timestamp: new Date().toLocaleTimeString()
    };

    const messages = this.data.messages;
    messages.push(message);

    this.setData({
      messages,
      scrollIntoView: `msg-${message.id}`
    });
  },

  // 输入框变化
  onInputChange(e) {
    this.setData({
      inputText: e.detail.value
    });
  }
});
```

```xml
<!-- pages/chat/chat.wxml -->
<view class="chat-container">
  <!-- 连接状态 -->
  <view class="status-bar">
    <text class="status {{isConnected ? 'connected' : 'disconnected'}}">
      {{isConnected ? '已连接' : '未连接'}}
    </text>
  </view>

  <!-- 消息列表 -->
  <scroll-view class="messages" scroll-y="true" scroll-into-view="{{scrollIntoView}}">
    <view wx:for="{{messages}}" wx:key="id" id="msg-{{item.id}}" class="message {{item.role}}">
      <view class="message-content">
        <text>{{item.content}}</text>
      </view>
      
      <view wx:if="{{item.references}}" class="references">
        <text class="references-title">参考资料:</text>
        <view wx:for="{{item.references}}" wx:key="index" wx:for-item="ref" class="reference">
          <text class="ref-title">{{ref.title}}</text>
          <text class="ref-content">{{ref.content}}</text>
        </view>
      </view>
      
      <text class="timestamp">{{item.timestamp}}</text>
    </view>
  </scroll-view>

  <!-- 实时转录显示 -->
  <view wx:if="{{currentTranscript}}" class="transcript">
    <text>{{currentTranscript}}</text>
  </view>

  <!-- 输入区域 -->
  <view class="input-area">
    <input 
      type="text" 
      placeholder="输入消息..."
      value="{{inputText}}"
      bindinput="onInputChange"
      bindconfirm="sendTextMessage"
      disabled="{{!isConnected}}"
    />
    
    <button 
      class="send-btn" 
      bindtap="sendTextMessage"
      disabled="{{!isConnected || !inputText}}"
    >
      发送
    </button>
  </view>

  <!-- 语音控制按钮 -->
  <view class="voice-controls">
    <button 
      class="voice-btn {{isRecording ? 'recording' : ''}}"
      bindtouchstart="startRecording"
      bindtouchend="stopRecording"
      disabled="{{!isConnected}}"
    >
      {{isRecording ? '录音中...' : '按住说话'}}
    </button>
  </view>
</view>
```

### React Native集成

```javascript
// React Native集成示例
import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  StyleSheet,
  Platform
} from 'react-native';
import { VoiceHelperReactNative } from '@voicehelper/react-native-sdk';

const ChatScreen = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentTranscript, setCurrentTranscript] = useState('');
  
  const voiceHelperRef = useRef(null);
  const scrollViewRef = useRef(null);

  useEffect(() => {
    initializeVoiceHelper();
    
    return () => {
      if (voiceHelperRef.current) {
        voiceHelperRef.current.disconnect();
      }
    };
  }, []);

  const initializeVoiceHelper = async () => {
    try {
      const voiceHelper = new VoiceHelperReactNative({
        apiKey: 'your-api-key',
        baseURL: 'https://api.voicehelper.ai',
        timeout: 30000
      });

      await voiceHelper.initialize();
      voiceHelperRef.current = voiceHelper;

      // 设置事件监听
      voiceHelper.onConnectionChange((status) => {
        setIsConnected(status === 'connected');
      });

      voiceHelper.onASRResult((result) => {
        setCurrentTranscript(result.text);
        
        if (result.is_final) {
          addMessage('user', result.text);
          setCurrentTranscript('');
        }
      });

      voiceHelper.onAIResponse((response) => {
        addMessage('assistant', response.text, response.references);
      });

      voiceHelper.onError((error) => {
        Alert.alert('错误', error.message);
      });

      // 连接服务
      await voiceHelper.connect({
        conversation_id: `rn_${Date.now()}`
      });

    } catch (error) {
      console.error('初始化失败:', error);
      Alert.alert('初始化失败', error.message);
    }
  };

  const sendTextMessage = async () => {
    if (!inputText.trim() || !isConnected) return;

    addMessage('user', inputText);
    setInputText('');

    try {
      await voiceHelperRef.current.sendMessage(inputText);
    } catch (error) {
      console.error('发送消息失败:', error);
      Alert.alert('发送失败', error.message);
    }
  };

  const startRecording = async () => {
    try {
      await voiceHelperRef.current.startRecording({
        language: 'zh-CN',
        enableVAD: true,
        enableNoiseReduction: true
      });
      setIsRecording(true);
    } catch (error) {
      console.error('开始录音失败:', error);
      Alert.alert('录音失败', error.message);
    }
  };

  const stopRecording = async () => {
    try {
      await voiceHelperRef.current.stopRecording();
      setIsRecording(false);
    } catch (error) {
      console.error('停止录音失败:', error);
    }
  };

  const addMessage = (role, content, references = null) => {
    const message = {
      id: Date.now(),
      role,
      content,
      references,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, message]);
    
    // 滚动到底部
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  };

  return (
    <View style={styles.container}>
      {/* 状态栏 */}
      <View style={styles.statusBar}>
        <Text style={[styles.statusText, { color: isConnected ? 'green' : 'red' }]}>
          {isConnected ? '已连接' : '未连接'}
        </Text>
      </View>

      {/* 消息列表 */}
      <ScrollView 
        ref={scrollViewRef}
        style={styles.messagesList}
        contentContainerStyle={styles.messagesContent}
      >
        {messages.map(message => (
          <View key={message.id} style={[
            styles.messageBubble,
            message.role === 'user' ? styles.userMessage : styles.aiMessage
          ]}>
            <Text style={[
              styles.messageText,
              { color: message.role === 'user' ? 'white' : 'black' }
            ]}>
              {message.content}
            </Text>
            
            {message.references && message.references.length > 0 && (
              <View style={styles.references}>
                <Text style={styles.referencesTitle}>参考资料:</Text>
                {message.references.map((ref, index) => (
                  <View key={index} style={styles.reference}>
                    <Text style={styles.refTitle}>{ref.title}</Text>
                    <Text style={styles.refContent}>
                      {ref.content.substring(0, 100)}...
                    </Text>
                  </View>
                ))}
              </View>
            )}
            
            <Text style={styles.timestamp}>{message.timestamp}</Text>
          </View>
        ))}
      </ScrollView>

      {/* 实时转录显示 */}
      {currentTranscript ? (
        <View style={styles.transcriptBar}>
          <Text style={styles.transcriptText}>{currentTranscript}</Text>
        </View>
      ) : null}

      {/* 输入区域 */}
      <View style={styles.inputArea}>
        <TextInput
          style={styles.textInput}
          value={inputText}
          onChangeText={setInputText}
          placeholder="输入消息..."
          editable={isConnected}
          onSubmitEditing={sendTextMessage}
        />
        
        <TouchableOpacity
          style={[styles.sendButton, { opacity: (!isConnected || !inputText) ? 0.5 : 1 }]}
          onPress={sendTextMessage}
          disabled={!isConnected || !inputText}
        >
          <Text style={styles.sendButtonText}>发送</Text>
        </TouchableOpacity>
      </View>

      {/* 语音控制 */}
      <View style={styles.voiceControls}>
        <TouchableOpacity
          style={[
            styles.voiceButton,
            { 
              backgroundColor: isRecording ? '#FF4444' : '#007AFF',
              opacity: !isConnected ? 0.5 : 1 
            }
          ]}
          onPressIn={startRecording}
          onPressOut={stopRecording}
          disabled={!isConnected}
        >
          <Text style={styles.voiceButtonText}>
            {isRecording ? '松开结束' : '按住说话'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5'
  },
  statusBar: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0'
  },
  statusText: {
    fontSize: 14,
    fontWeight: 'bold'
  },
  messagesList: {
    flex: 1,
    paddingHorizontal: 16
  },
  messagesContent: {
    paddingVertical: 16
  },
  messageBubble: {
    marginBottom: 12,
    padding: 12,
    borderRadius: 12,
    maxWidth: '80%'
  },
  userMessage: {
    backgroundColor: '#007AFF',
    alignSelf: 'flex-end'
  },
  aiMessage: {
    backgroundColor: 'white',
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#E0E0E0'
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22
  },
  references: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.3)'
  },
  referencesTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    color: 'rgba(0,0,0,0.6)',
    marginBottom: 4
  },
  reference: {
    marginBottom: 4
  },
  refTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    color: 'rgba(0,0,0,0.8)'
  },
  refContent: {
    fontSize: 11,
    color: 'rgba(0,0,0,0.6)'
  },
  timestamp: {
    fontSize: 11,
    color: 'rgba(0,0,0,0.5)',
    marginTop: 4
  },
  transcriptBar: {
    backgroundColor: '#FFF3CD',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0'
  },
  transcriptText: {
    fontSize: 14,
    color: '#856404',
    fontStyle: 'italic'
  },
  inputArea: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: 'white',
    alignItems: 'center'
  },
  textInput: {
    flex: 1,
    height: 40,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    borderRadius: 20,
    paddingHorizontal: 16,
    fontSize: 16,
    marginRight: 8
  },
  sendButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20
  },
  sendButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold'
  },
  voiceControls: {
    paddingHorizontal: 16,
    paddingVertical: 16,
    backgroundColor: 'white',
    alignItems: 'center'
  },
  voiceButton: {
    width: 120,
    height: 120,
    borderRadius: 60,
    justifyContent: 'center',
    alignItems: 'center'
  },
  voiceButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center'
  }
});

export default ChatScreen;
```

---

## 🎯 总结

本框架使用指南提供了VoiceHelper平台的全面集成方案：

### 🚀 部署选项
- **Docker一键部署**: 生产环境快速启动
- **开发环境搭建**: 本地开发和调试
- **云原生部署**: Kubernetes集群部署

### 🔧 SDK支持  
- **JavaScript/TypeScript**: Web应用和Node.js服务
- **Python**: 服务端应用和AI工作流
- **React Native**: 跨平台移动应用
- **微信小程序**: 原生小程序开发

### 📱 多平台覆盖
- **Web应用**: React、Vue、Angular等主流框架
- **移动应用**: iOS、Android原生和跨平台
- **桌面应用**: Electron、Tauri等桌面框架
- **小程序**: 微信、支付宝、百度等小程序平台

### 🌟 核心能力
- **多模态交互**: 文本和语音无缝切换
- **智能检索**: RAG和GraphRAG双重加持  
- **实时通信**: WebSocket低延迟交互
- **企业级**: 高可用、高性能、高安全

通过本指南的示例代码和最佳实践，开发者可以快速集成VoiceHelper平台，构建功能强大的智能对话应用。
