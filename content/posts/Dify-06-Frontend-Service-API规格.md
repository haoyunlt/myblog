---
title: "Dify-06-Frontend Service API规格"
date: 2025-10-05T01:01:58+08:00
draft: false
tags:
  - Dify
  - API设计
  - 接口文档
  - 源码分析
categories:
  - Dify
  - AI应用开发
series: "dify-source-analysis"
description: "Dify 源码剖析 - Dify-06-Frontend Service API规格"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# Dify-06-Frontend Service API规格

## 摘要

本文档详细说明 Dify Frontend 的 Service 层 API，这些 API 封装了与 Backend 的交互逻辑，提供类型安全的接口供前端组件调用。

### Service层职责

| 职责 | 说明 |
|------|------|
| **HTTP封装** | 封装fetch/axios，统一请求格式和错误处理 |
| **类型定义** | 提供TypeScript类型定义，确保类型安全 |
| **认证管理** | 自动添加Token，处理Token刷新 |
| **错误处理** | 统一错误格式，提供友好的错误提示 |
| **缓存策略** | 配合SWR实现数据缓存和重验证 |
| **SSE处理** | 处理Server-Sent Events流式响应 |

### 核心Service模块

- `base.ts` - 基础HTTP封装
- `apps.ts` - 应用管理
- `datasets.ts` - 知识库管理
- `workflow.ts` - 工作流管理
- `debug.ts` - 调试和日志
- `common.ts` - 通用功能（文件上传、模型列表等）
- `plugins.ts` - 插件管理
- `tools.ts` - 工具管理

---

## 一、基础Service（base.ts）

### 1.1 HTTP请求封装

#### 1.1.1 request()

**功能**：基础HTTP请求方法

**函数签名**：
```typescript
async function request<T>(
  url: string,
  options: RequestOptions = {},
  otherOptions?: IOtherOptions
): Promise<T>
```

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| url | string | 请求URL（相对路径） |
| options | RequestOptions | Fetch选项 |
| otherOptions | IOtherOptions | 自定义选项（认证、错误处理等） |

```typescript
interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  headers?: Record<string, string>
  body?: any
  params?: Record<string, any>      // URL查询参数
}

interface IOtherOptions {
  isPublicAPI?: boolean              // 是否使用Public API
  bodyStringify?: boolean            // 是否序列化body
  silent?: boolean                   // 是否静默错误（不弹Toast）
  needAllResponseContent?: boolean   // 是否返回完整响应
  deleteContentType?: boolean        // 是否删除Content-Type头
}
```

**返回值**：Promise<T>，泛型T为响应数据类型

**核心代码**：
```typescript
export const request = async <T>(
  url: string,
  options = {},
  otherOptions?: IOtherOptions
): Promise<T> => {
  try {
    // 1. 获取Token
    const token = await getAccessToken(otherOptions?.isPublicAPI)
    
    // 2. 构建请求选项
    const baseOptions = getBaseOptions()
    const mergedOptions = {
      ...baseOptions,
      ...options,
      headers: {
        ...baseOptions.headers,
        ...options.headers,
        'Authorization': `Bearer ${token}`,
      },
    }
    
    // 3. 构建完整URL
    const urlPrefix = otherOptions?.isPublicAPI 
      ? PUBLIC_API_PREFIX 
      : API_PREFIX
    const fullUrl = `${urlPrefix}${url}`
    
    // 4. 发送请求
    const response = await baseFetch<T>(fullUrl, mergedOptions, otherOptions)
    
    return response
  } catch (error) {
    // 5. 错误处理
    if (error.status === 401) {
      // Token过期，刷新Token并重试
      await refreshAccessTokenOrRelogin()
      return request<T>(url, options, otherOptions)
    }
    
    // 6. 显示错误提示
    if (!otherOptions?.silent) {
      Toast.notify({
        type: 'error',
        message: error.message || 'Request failed'
      })
    }
    
    throw error
  }
}
```

**使用示例**：
```typescript
// GET请求
const apps = await request<App[]>('/apps', {
  method: 'GET',
  params: { page: 1, limit: 30 }
})

// POST请求
const newApp = await request<App>('/apps', {
  method: 'POST',
  body: {
    name: 'My App',
    mode: 'chat'
  }
})
```

---

#### 1.1.2 ssePost()

**功能**：发送SSE请求，处理流式响应

**函数签名**：
```typescript
async function ssePost(
  url: string,
  fetchOptions: FetchOptionType,
  otherOptions: IOtherOptions
): Promise<void>
```

**参数**：

```typescript
interface IOtherOptions {
  onData?: (message: string, isFirst: boolean, moreInfo: IOnDataMoreInfo) => void
  onThought?: (thought: ThoughtItem) => void
  onMessageEnd?: (messageEnd: MessageEnd) => void
  onWorkflowStarted?: (data: WorkflowStartedResponse) => void
  onNodeStarted?: (data: NodeStartedResponse) => void
  onNodeFinished?: (data: NodeFinishedResponse) => void
  onCompleted?: (hasError?: boolean, errorMessage?: string) => void
  onError?: (error: string, code?: string) => void
  getAbortController?: (controller: AbortController) => void
}
```

**核心代码**：
```typescript
export const ssePost = async (
  url: string,
  fetchOptions: FetchOptionType,
  otherOptions: IOtherOptions
) => {
  const {
    onData,
    onCompleted,
    onError,
    onWorkflowStarted,
    onNodeStarted,
    onNodeFinished,
    getAbortController,
  } = otherOptions
  
  // 1. 创建AbortController
  const abortController = new AbortController()
  getAbortController?.(abortController)
  
  // 2. 构建请求选项
  const token = await getAccessToken()
  const options = {
    method: 'POST',
    signal: abortController.signal,
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(fetchOptions.body),
  }
  
  // 3. 发送请求
  const response = await globalThis.fetch(url, options)
  
  // 4. 处理SSE流
  const reader = response.body?.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  
  function read() {
    reader?.read().then((result) => {
      if (result.done) {
        onCompleted?.()
        return
      }
      
      // 5. 解析SSE数据
      buffer += decoder.decode(result.value, { stream: true })
      const lines = buffer.split('\n')
      
      lines.forEach((line) => {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.substring(6))
          
          // 6. 根据事件类型调用回调
          switch (data.event) {
            case 'message':
              onData?.(data.answer, false, {
                conversationId: data.conversation_id,
                messageId: data.id,
              })
              break
            case 'workflow_started':
              onWorkflowStarted?.(data)
              break
            case 'node_started':
              onNodeStarted?.(data)
              break
            case 'node_finished':
              onNodeFinished?.(data)
              break
            // ... 其他事件类型
          }
        }
      })
      
      buffer = lines[lines.length - 1]
      read()
    })
  }
  
  read()
}
```

**使用示例**：
```typescript
// 发送消息（流式）
ssePost('/chat-messages', {
  body: {
    query: 'Hello',
    user: 'user-123',
    response_mode: 'streaming'
  }
}, {
  onData: (message, isFirst, moreInfo) => {
    console.log('收到消息片段:', message)
  },
  onMessageEnd: (data) => {
    console.log('消息结束:', data.metadata)
  },
  onError: (error) => {
    console.error('错误:', error)
  },
  onCompleted: () => {
    console.log('完成')
  }
})
```

---

#### 1.1.3 简化方法

**get()**：GET请求
```typescript
export const get = <T>(url: string, options = {}, otherOptions?: IOtherOptions) => {
  return request<T>(url, { ...options, method: 'GET' }, otherOptions)
}
```

**post()**：POST请求
```typescript
export const post = <T>(url: string, options = {}, otherOptions?: IOtherOptions) => {
  return request<T>(url, { ...options, method: 'POST' }, otherOptions)
}
```

**put()**：PUT请求
```typescript
export const put = <T>(url: string, options = {}, otherOptions?: IOtherOptions) => {
  return request<T>(url, { ...options, method: 'PUT' }, otherOptions)
}
```

**del()**：DELETE请求
```typescript
export const del = <T>(url: string, options = {}, otherOptions?: IOtherOptions) => {
  return request<T>(url, { ...options, method: 'DELETE' }, otherOptions)
}
```

---

### 1.2 文件上传

#### upload()

**功能**：上传文件

**函数签名**：
```typescript
async function upload(
  options: UploadOptions,
  isPublicAPI?: boolean,
  url?: string,
  searchParams?: string
): Promise<UploadResponse>
```

**参数**：
```typescript
interface UploadOptions {
  xhr: XMLHttpRequest
  data: FormData
  onprogress: (event: ProgressEvent) => void
}

interface UploadResponse {
  id: string
  name: string
  size: number
  extension: string
  mime_type: string
  created_at: number
}
```

**核心代码**：
```typescript
export const upload = async (
  options: any,
  isPublicAPI?: boolean,
  url?: string,
  searchParams?: string
): Promise<any> => {
  const urlPrefix = isPublicAPI ? PUBLIC_API_PREFIX : API_PREFIX
  const token = await getAccessToken(isPublicAPI)
  
  const defaultOptions = {
    method: 'POST',
    url: (url ? `${urlPrefix}${url}` : `${urlPrefix}/files/upload`) + (searchParams || ''),
    headers: {
      Authorization: `Bearer ${token}`,
    },
  }
  
  return new Promise((resolve, reject) => {
    const xhr = options.xhr
    xhr.open(defaultOptions.method, defaultOptions.url)
    
    for (const key in defaultOptions.headers)
      xhr.setRequestHeader(key, defaultOptions.headers[key])
    
    xhr.withCredentials = true
    xhr.responseType = 'json'
    
    xhr.onreadystatechange = function () {
      if (xhr.readyState === 4) {
        if (xhr.status === 201)
          resolve(xhr.response)
        else
          reject(xhr)
      }
    }
    
    xhr.upload.onprogress = options.onprogress
    xhr.send(options.data)
  })
}
```

**使用示例**：
```typescript
// 上传文件
const formData = new FormData()
formData.append('file', file)

const xhr = new XMLHttpRequest()

const result = await upload({
  xhr,
  data: formData,
  onprogress: (event) => {
    const progress = (event.loaded / event.total) * 100
    console.log(`上传进度: ${progress}%`)
  }
})

console.log('文件ID:', result.id)
```

---

## 二、应用管理Service（apps.ts）

### 2.1 应用CRUD

#### 2.1.1 fetchAppList()

**功能**：获取应用列表

**函数签名**：
```typescript
async function fetchAppList(): Promise<{ data: App[] }>
```

**返回类型**：
```typescript
interface App {
  id: string
  name: string
  mode: AppMode
  icon: string
  icon_background: string
  description: string
  // ... 其他字段
}
```

**实现**：
```typescript
export const fetchAppList = () => {
  return get<{ data: App[] }>('/apps', {
    params: { page: 1, limit: 100, name: '' }
  })
}
```

**配合SWR使用**：
```typescript
import useSWR from 'swr'

export const useAppList = () => {
  const { data, error, isLoading, mutate } = useSWR<{ data: App[] }>(
    '/apps',
    fetchAppList
  )
  
  return {
    apps: data?.data,
    isLoading,
    isError: error,
    refresh: mutate,
  }
}

// 在组件中使用
function AppsPage() {
  const { apps, isLoading, refresh } = useAppList()
  
  if (isLoading) return <Loading />
  
  return (
    <div>
      {apps?.map(app => <AppCard key={app.id} app={app} />)}
      <button onClick={refresh}>刷新</button>
    </div>
  )
}
```

---

#### 2.1.2 createApp()

**功能**：创建应用

**函数签名**：
```typescript
async function createApp(body: CreateAppBody): Promise<App>
```

**参数类型**：
```typescript
interface CreateAppBody {
  name: string
  mode: AppMode
  description?: string
  icon?: string
  icon_background?: string
}
```

**实现**：
```typescript
export const createApp = (body: CreateAppBody) => {
  return post<App>('/apps', { body })
}
```

**使用示例**：
```typescript
async function handleCreateApp() {
  try {
    const newApp = await createApp({
      name: 'My New App',
      mode: 'chat',
      description: 'A chat application'
    })
    
    console.log('创建成功:', newApp.id)
    
    // 刷新应用列表
    mutate('/apps')
    
    // 跳转到应用配置页
    router.push(`/app/${newApp.id}/configuration`)
  } catch (error) {
    console.error('创建失败:', error)
  }
}
```

---

#### 2.1.3 updateAppInfo()

**功能**：更新应用信息

**函数签名**：
```typescript
async function updateAppInfo(
  appID: string,
  body: UpdateAppBody
): Promise<App>
```

**实现**：
```typescript
export const updateAppInfo = (appID: string, body: UpdateAppBody) => {
  return put<App>(`/apps/${appID}`, { body })
}
```

---

#### 2.1.4 deleteApp()

**功能**：删除应用

**函数签名**：
```typescript
async function deleteApp(appID: string): Promise<{ result: 'success' }>
```

**实现**：
```typescript
export const deleteApp = (appID: string) => {
  return del<{ result: 'success' }>(`/apps/${appID}`)
}
```

**使用示例**：
```typescript
async function handleDeleteApp(appId: string) {
  // 1. 确认对话框
  const confirmed = await confirm({
    title: '删除应用',
    message: '确定要删除此应用吗？此操作不可撤销。'
  })
  
  if (!confirmed) return
  
  try {
    // 2. 调用删除API
    await deleteApp(appId)
    
    // 3. 刷新列表
    mutate('/apps')
    
    Toast.notify({ type: 'success', message: '删除成功' })
  } catch (error) {
    Toast.notify({ type: 'error', message: '删除失败' })
  }
}
```

---

### 2.2 应用配置

#### 2.2.1 fetchAppDetail()

**功能**：获取应用详情

**函数签名**：
```typescript
async function fetchAppDetail(appID: string): Promise<App>
```

**实现**：
```typescript
export const fetchAppDetail = (appID: string) => {
  return get<App>(`/apps/${appID}`)
}

// 配合SWR
export const useAppDetail = (appID: string) => {
  const { data, error, isLoading, mutate } = useSWR<App>(
    appID ? `/apps/${appID}` : null,
    () => fetchAppDetail(appID)
  )
  
  return {
    app: data,
    isLoading,
    isError: error,
    refresh: mutate,
  }
}
```

---

#### 2.2.2 updateAppModelConfig()

**功能**：更新应用模型配置

**函数签名**：
```typescript
async function updateAppModelConfig(
  appID: string,
  body: ModelConfig
): Promise<{ result: 'success' }>
```

**参数类型**：
```typescript
interface ModelConfig {
  model: {
    provider: string
    name: string
    mode: 'chat' | 'completion'
    completion_params: {
      temperature: number
      top_p: number
      max_tokens: number
      presence_penalty: number
      frequency_penalty: number
    }
  }
  user_input_form: UserInputForm[]
  pre_prompt?: string
  agent_mode?: {
    enabled: boolean
    tools: Tool[]
    strategy: 'function_call' | 'react'
  }
  dataset_configs?: DatasetConfig
  // ... 其他配置
}
```

**实现**：
```typescript
export const updateAppModelConfig = (appID: string, body: ModelConfig) => {
  return post<{ result: 'success' }>(`/apps/${appID}/model-config`, { body })
}
```

---

### 2.3 应用调试

#### 2.3.1 sendChatMessage()

**功能**：发送聊天消息（调试模式）

**函数签名**：
```typescript
function sendChatMessage(
  appId: string,
  body: ChatMessageBody,
  callbacks: ChatMessageCallbacks
): AbortController
```

**参数类型**：
```typescript
interface ChatMessageBody {
  query: string
  inputs: Record<string, any>
  conversation_id?: string
  parent_message_id?: string
}

interface ChatMessageCallbacks {
  onData: (message: string, isFirst: boolean) => void
  onMessageEnd: (data: MessageEnd) => void
  onError: (error: string) => void
  onCompleted: () => void
}
```

**实现**：
```typescript
export const sendChatMessage = (
  appId: string,
  body: ChatMessageBody,
  callbacks: ChatMessageCallbacks
): AbortController => {
  const abortController = new AbortController()
  
  ssePost(`/apps/${appId}/chat-messages`, {
    body,
  }, {
    onData: callbacks.onData,
    onMessageEnd: callbacks.onMessageEnd,
    onError: callbacks.onError,
    onCompleted: callbacks.onCompleted,
    getAbortController: (controller) => {
      // 返回controller供外部使用
    }
  })
  
  return abortController
}
```

**使用示例**：
```typescript
function ChatDebugPanel({ appId }: { appId: string }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [currentMessage, setCurrentMessage] = useState('')
  const abortControllerRef = useRef<AbortController>()
  
  const handleSend = (query: string) => {
    const tempMessage: Message = {
      id: 'temp',
      role: 'assistant',
      content: '',
    }
    
    setMessages(prev => [...prev, tempMessage])
    
    const controller = sendChatMessage(
      appId,
      {
        query,
        inputs: {},
      },
      {
        onData: (message, isFirst) => {
          if (isFirst) {
            setCurrentMessage(message)
          } else {
            setCurrentMessage(prev => prev + message)
          }
        },
        onMessageEnd: (data) => {
          setMessages(prev => [
            ...prev.slice(0, -1),
            {
              id: data.id,
              role: 'assistant',
              content: currentMessage,
              metadata: data.metadata,
            }
          ])
          setCurrentMessage('')
        },
        onError: (error) => {
          Toast.notify({ type: 'error', message: error })
        },
        onCompleted: () => {
          console.log('完成')
        }
      }
    )
    
    abortControllerRef.current = controller
  }
  
  const handleStop = () => {
    abortControllerRef.current?.abort()
  }
  
  return (
    <div>
      {/* 消息列表 */}
      <div className="messages">
        {messages.map(msg => (
          <MessageItem key={msg.id} message={msg} />
        ))}
        {currentMessage && (
          <MessageItem message={{ content: currentMessage, role: 'assistant' }} />
        )}
      </div>
      
      {/* 输入框 */}
      <ChatInput onSend={handleSend} onStop={handleStop} />
    </div>
  )
}
```

---

## 三、知识库Service（datasets.ts）

### 3.1 知识库CRUD

#### 3.1.1 fetchDatasets()

**功能**：获取知识库列表

**函数签名**：
```typescript
async function fetchDatasets(params: FetchDatasetsParams): Promise<DatasetsListResponse>
```

**参数类型**：
```typescript
interface FetchDatasetsParams {
  page: number
  limit: number
}

interface DatasetsListResponse {
  data: Dataset[]
  total: number
  page: number
  limit: number
  has_more: boolean
}
```

**实现**：
```typescript
export const fetchDatasets = (params: FetchDatasetsParams) => {
  return get<DatasetsListResponse>('/datasets', { params })
}
```

---

#### 3.1.2 createDataset()

**功能**：创建知识库

**函数签名**：
```typescript
async function createDataset(body: CreateDatasetBody): Promise<Dataset>
```

**参数类型**：
```typescript
interface CreateDatasetBody {
  name: string
  indexing_technique: 'high_quality' | 'economy'
  permission: 'only_me' | 'all_team_members'
  embedding_model?: string
  embedding_model_provider?: string
  retrieval_model?: RetrievalModel
}
```

**实现**：
```typescript
export const createDataset = (body: CreateDatasetBody) => {
  return post<Dataset>('/datasets', { body })
}
```

---

### 3.2 文档管理

#### 3.2.1 fetchDocuments()

**功能**：获取文档列表

**函数签名**：
```typescript
async function fetchDocuments(
  datasetId: string,
  params: FetchDocumentsParams
): Promise<DocumentsListResponse>
```

**实现**：
```typescript
export const fetchDocuments = (datasetId: string, params: FetchDocumentsParams) => {
  return get<DocumentsListResponse>(`/datasets/${datasetId}/documents`, { params })
}
```

---

#### 3.2.2 uploadDocument()

**功能**：上传文档

**函数签名**：
```typescript
async function uploadDocument(
  datasetId: string,
  file: File,
  options: UploadDocumentOptions,
  onProgress: (progress: number) => void
): Promise<UploadDocumentResponse>
```

**参数类型**：
```typescript
interface UploadDocumentOptions {
  indexing_technique: 'high_quality' | 'economy'
  process_rule: {
    mode: 'automatic' | 'custom'
    rules?: ProcessRules
  }
}

interface UploadDocumentResponse {
  document: Document
  batch: string
}
```

**实现**：
```typescript
export const uploadDocument = async (
  datasetId: string,
  file: File,
  options: UploadDocumentOptions,
  onProgress: (progress: number) => void
): Promise<UploadDocumentResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('indexing_technique', options.indexing_technique)
  formData.append('process_rule', JSON.stringify(options.process_rule))
  
  const xhr = new XMLHttpRequest()
  
  return upload({
    xhr,
    data: formData,
    onprogress: (event) => {
      const progress = (event.loaded / event.total) * 100
      onProgress(progress)
    }
  }, false, `/datasets/${datasetId}/document/create_by_file`)
}
```

**使用示例**：
```typescript
function DocumentUpload({ datasetId }: { datasetId: string }) {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  
  const handleUpload = async (file: File) => {
    setUploading(true)
    setProgress(0)
    
    try {
      const result = await uploadDocument(
        datasetId,
        file,
        {
          indexing_technique: 'high_quality',
          process_rule: {
            mode: 'automatic'
          }
        },
        (progress) => {
          setProgress(progress)
        }
      )
      
      console.log('上传成功:', result.document.id)
      console.log('批次ID:', result.batch)
      
      // 开始轮询索引状态
      pollIndexingStatus(result.batch)
      
      Toast.notify({ type: 'success', message: '上传成功，正在索引...' })
    } catch (error) {
      Toast.notify({ type: 'error', message: '上传失败' })
    } finally {
      setUploading(false)
    }
  }
  
  return (
    <div>
      <input
        type="file"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) handleUpload(file)
        }}
        disabled={uploading}
      />
      {uploading && <Progress value={progress} />}
    </div>
  )
}
```

---

#### 3.2.3 fetchDocumentIndexingStatus()

**功能**：查询文档索引状态

**函数签名**：
```typescript
async function fetchDocumentIndexingStatus(
  datasetId: string,
  batch: string
): Promise<IndexingStatusResponse>
```

**返回类型**：
```typescript
interface IndexingStatusResponse {
  data: Array<{
    id: string
    indexing_status: IndexingStatus
    processing_started_at?: number
    completed_at?: number
    error?: string
  }>
}

type IndexingStatus = 
  | 'queuing'
  | 'parsing'
  | 'cleaning'
  | 'splitting'
  | 'indexing'
  | 'completed'
  | 'error'
```

**实现**：
```typescript
export const fetchDocumentIndexingStatus = (datasetId: string, batch: string) => {
  return get<IndexingStatusResponse>(
    `/datasets/${datasetId}/documents/${batch}/indexing-status`
  )
}
```

**轮询索引状态**：
```typescript
async function pollIndexingStatus(batch: string) {
  const maxAttempts = 100
  let attempts = 0
  
  const poll = async () => {
    if (attempts >= maxAttempts) {
      Toast.notify({ type: 'error', message: '索引超时' })
      return
    }
    
    const result = await fetchDocumentIndexingStatus(datasetId, batch)
    const document = result.data[0]
    
    if (document.indexing_status === 'completed') {
      Toast.notify({ type: 'success', message: '索引完成' })
      mutate(`/datasets/${datasetId}/documents`)
    } else if (document.indexing_status === 'error') {
      Toast.notify({ type: 'error', message: `索引失败: ${document.error}` })
    } else {
      // 继续轮询
      attempts++
      setTimeout(poll, 3000)
    }
  }
  
  poll()
}
```

---

### 3.3 检索测试

#### fetchDocumentRetrievalRecords()

**功能**：检索测试

**函数签名**：
```typescript
async function fetchDocumentRetrievalRecords(
  datasetId: string,
  body: RetrievalTestBody
): Promise<RetrievalRecordsResponse>
```

**参数类型**：
```typescript
interface RetrievalTestBody {
  query: string
  retrieval_model: {
    search_method: 'semantic_search' | 'full_text_search' | 'hybrid_search'
    reranking_enable: boolean
    top_k: number
    score_threshold?: number
  }
}
```

**实现**：
```typescript
export const fetchDocumentRetrievalRecords = (
  datasetId: string,
  body: RetrievalTestBody
) => {
  return post<RetrievalRecordsResponse>(`/datasets/${datasetId}/retrieve`, { body })
}
```

**使用示例**：
```typescript
function RetrievalTestPanel({ datasetId }: { datasetId: string }) {
  const [query, setQuery] = useState('')
  const [records, setRecords] = useState<RetrievalRecord[]>([])
  const [loading, setLoading] = useState(false)
  
  const handleTest = async () => {
    setLoading(true)
    
    try {
      const result = await fetchDocumentRetrievalRecords(datasetId, {
        query,
        retrieval_model: {
          search_method: 'hybrid_search',
          reranking_enable: true,
          top_k: 5,
          score_threshold: 0.3
        }
      })
      
      setRecords(result.records)
    } catch (error) {
      Toast.notify({ type: 'error', message: '检索失败' })
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="输入查询文本"
      />
      <button onClick={handleTest} disabled={loading}>
        {loading ? '检索中...' : '测试检索'}
      </button>
      
      <div className="results">
        {records.map((record, index) => (
          <RetrievalRecordCard
            key={index}
            record={record}
            rank={index + 1}
          />
        ))}
      </div>
    </div>
  )
}
```

---

## 四、工作流Service（workflow.ts）

### 4.1 工作流配置

#### 4.1.1 fetchWorkflowDraft()

**功能**：获取工作流草稿

**函数签名**：
```typescript
async function fetchWorkflowDraft(appId: string): Promise<WorkflowDraft>
```

**返回类型**：
```typescript
interface WorkflowDraft {
  graph: {
    nodes: Node[]
    edges: Edge[]
  }
  features: WorkflowFeatures
  environment_variables: EnvironmentVariable[]
}
```

**实现**：
```typescript
export const fetchWorkflowDraft = (appId: string) => {
  return get<WorkflowDraft>(`/apps/${appId}/workflows/draft`)
}
```

---

#### 4.1.2 saveWorkflowDraft()

**功能**：保存工作流草稿

**函数签名**：
```typescript
async function saveWorkflowDraft(
  appId: string,
  body: SaveWorkflowDraftBody
): Promise<SaveWorkflowDraftResponse>
```

**参数类型**：
```typescript
interface SaveWorkflowDraftBody {
  graph: {
    nodes: Node[]
    edges: Edge[]
  }
  features?: WorkflowFeatures
  environment_variables?: EnvironmentVariable[]
}
```

**实现**：
```typescript
export const saveWorkflowDraft = (appId: string, body: SaveWorkflowDraftBody) => {
  return post<SaveWorkflowDraftResponse>(`/apps/${appId}/workflows/draft`, { body })
}
```

**使用示例**：
```typescript
function WorkflowEditor({ appId }: { appId: string }) {
  const store = useWorkflowStore()  // Zustand store
  const [saving, setSaving] = useState(false)
  
  const handleSave = async () => {
    setSaving(true)
    
    try {
      const result = await saveWorkflowDraft(appId, {
        graph: {
          nodes: store.nodes,
          edges: store.edges
        },
        features: store.features,
        environment_variables: store.environmentVariables
      })
      
      Toast.notify({ type: 'success', message: '保存成功' })
      
      // 更新最后保存时间
      store.setLastSavedAt(result.updated_at)
    } catch (error) {
      Toast.notify({ type: 'error', message: '保存失败' })
    } finally {
      setSaving(false)
    }
  }
  
  // 自动保存
  useEffect(() => {
    const timer = setInterval(() => {
      if (store.hasUnsavedChanges) {
        handleSave()
      }
    }, 30000) // 30秒自动保存
    
    return () => clearInterval(timer)
  }, [store.hasUnsavedChanges])
  
  return (
    <div>
      <button onClick={handleSave} disabled={saving}>
        {saving ? '保存中...' : '保存'}
      </button>
      {/* 工作流画布 */}
      <WorkflowCanvas />
    </div>
  )
}
```

---

### 4.2 工作流执行

#### runWorkflow()

**功能**：执行工作流（调试模式）

**函数签名**：
```typescript
function runWorkflow(
  appId: string,
  body: RunWorkflowBody,
  callbacks: WorkflowCallbacks
): AbortController
```

**参数类型**：
```typescript
interface RunWorkflowBody {
  inputs: Record<string, any>
  files?: File[]
}

interface WorkflowCallbacks {
  onWorkflowStarted: (data: WorkflowStartedResponse) => void
  onNodeStarted: (data: NodeStartedResponse) => void
  onNodeFinished: (data: NodeFinishedResponse) => void
  onWorkflowFinished: (data: WorkflowFinishedResponse) => void
  onError: (error: string) => void
}
```

**实现**：
```typescript
export const runWorkflow = (
  appId: string,
  body: RunWorkflowBody,
  callbacks: WorkflowCallbacks
): AbortController => {
  const abortController = new AbortController()
  
  ssePost(`/apps/${appId}/workflows/run`, {
    body,
  }, {
    onWorkflowStarted: callbacks.onWorkflowStarted,
    onNodeStarted: callbacks.onNodeStarted,
    onNodeFinished: callbacks.onNodeFinished,
    onWorkflowFinished: callbacks.onWorkflowFinished,
    onError: callbacks.onError,
    getAbortController: (controller) => {
      // 存储controller
    }
  })
  
  return abortController
}
```

---

## 五、通用Service（common.ts）

### 5.1 模型管理

#### fetchModelProviders()

**功能**：获取模型供应商列表

**函数签名**：
```typescript
async function fetchModelProviders(): Promise<{ data: ModelProvider[] }>
```

**实现**：
```typescript
export const fetchModelProviders = () => {
  return get<{ data: ModelProvider[] }>('/workspaces/current/model-providers')
}
```

---

### 5.2 文件管理

#### fetchFiles()

**功能**：获取文件列表

**函数签名**：
```typescript
async function fetchFiles(): Promise<{ data: FileItem[] }>
```

**实现**：
```typescript
export const fetchFiles = () => {
  return get<{ data: FileItem[] }>('/files')
}
```

---

## 六、自定义Hooks

### 6.1 useApps()

**功能**：应用列表Hook

```typescript
export function useApps() {
  const { data, error, isLoading, mutate } = useSWR<{ data: App[] }>(
    '/apps',
    fetchAppList,
    {
      revalidateOnFocus: false,
      dedupingInterval: 60000, // 1分钟去重
    }
  )
  
  const createApp = useCallback(async (body: CreateAppBody) => {
    const newApp = await createAppApi(body)
    mutate() // 刷新列表
    return newApp
  }, [mutate])
  
  const deleteApp = useCallback(async (appId: string) => {
    await deleteAppApi(appId)
    mutate() // 刷新列表
  }, [mutate])
  
  return {
    apps: data?.data,
    isLoading,
    isError: error,
    createApp,
    deleteApp,
    refresh: mutate,
  }
}
```

---

### 6.2 useDatasets()

**功能**：知识库列表Hook

```typescript
export function useDatasets() {
  const [page, setPage] = useState(1)
  const limit = 30
  
  const { data, error, isLoading, mutate } = useSWR<DatasetsListResponse>(
    `/datasets?page=${page}&limit=${limit}`,
    () => fetchDatasets({ page, limit })
  )
  
  return {
    datasets: data?.data,
    total: data?.total,
    page,
    setPage,
    isLoading,
    isError: error,
    refresh: mutate,
  }
}
```

---

## 七、最佳实践

### 7.1 错误处理

```typescript
async function handleApiCall() {
  try {
    const result = await someApiCall()
    return result
  } catch (error) {
    // 1. 根据错误类型处理
    if (error.status === 401) {
      // Token过期，已自动刷新并重试
      return
    }
    
    if (error.status === 429) {
      // 速率限制
      Toast.notify({
        type: 'warning',
        message: '请求过于频繁，请稍后再试'
      })
      return
    }
    
    // 2. 通用错误提示
    Toast.notify({
      type: 'error',
      message: error.message || '操作失败'
    })
    
    // 3. 记录错误到Sentry
    Sentry.captureException(error)
    
    throw error
  }
}
```

---

### 7.2 防抖和节流

```typescript
import { useDebounce } from 'ahooks'

function SearchInput() {
  const [keyword, setKeyword] = useState('')
  const debouncedKeyword = useDebounce(keyword, { wait: 500 })
  
  // 仅当防抖后的值变化时才请求
  const { data } = useSWR(
    debouncedKeyword ? `/search?q=${debouncedKeyword}` : null,
    fetchSearch
  )
  
  return (
    <input
      value={keyword}
      onChange={(e) => setKeyword(e.target.value)}
      placeholder="搜索..."
    />
  )
}
```

---

### 7.3 乐观更新

```typescript
async function handleLikeMessage(messageId: string) {
  // 1. 乐观更新本地状态
  mutate(
    `/messages/${messageId}`,
    { ...currentMessage, liked: true },
    false  // 不触发重验证
  )
  
  try {
    // 2. 发送API请求
    await likeMessage(messageId)
  } catch (error) {
    // 3. 失败时回滚
    mutate(
      `/messages/${messageId}`,
      { ...currentMessage, liked: false }
    )
    
    Toast.notify({ type: 'error', message: '点赞失败' })
  }
}
```

---

### 7.4 请求取消

```typescript
function SearchComponent() {
  const abortControllerRef = useRef<AbortController>()
  
  const handleSearch = (query: string) => {
    // 取消之前的请求
    abortControllerRef.current?.abort()
    
    // 创建新的AbortController
    const controller = new AbortController()
    abortControllerRef.current = controller
    
    // 发送请求
    fetch(`/search?q=${query}`, {
      signal: controller.signal
    })
      .then(response => response.json())
      .then(data => {
        // 处理结果
      })
      .catch(error => {
        if (error.name === 'AbortError') {
          console.log('请求已取消')
        }
      })
  }
  
  // 组件卸载时取消请求
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])
}
```

---

## 八、性能优化

### 8.1 请求合并

```typescript
// 使用SWR的mutate合并多个请求
async function batchUpdateApps(updates: Array<{ id: string; data: any }>) {
  // 1. 批量发送请求
  const promises = updates.map(update =>
    updateApp(update.id, update.data)
  )
  
  // 2. 等待全部完成
  await Promise.all(promises)
  
  // 3. 一次性刷新列表
  mutate('/apps')
}
```

---

### 8.2 缓存策略

```typescript
// 配置SWR缓存
const { data } = useSWR('/apps', fetchAppList, {
  revalidateOnFocus: true,        // 窗口聚焦时重验证
  revalidateOnReconnect: true,    // 网络恢复时重验证
  dedupingInterval: 60000,        // 60秒内去重
  refreshInterval: 300000,        // 5分钟自动刷新
  errorRetryCount: 3,             // 错误重试3次
  errorRetryInterval: 5000,       // 重试间隔5秒
  onErrorRetry: (error, key, config, revalidate, { retryCount }) => {
    // 404不重试
    if (error.status === 404) return
    
    // 最多重试3次
    if (retryCount >= 3) return
    
    // 指数退避
    setTimeout(() => revalidate({ retryCount }), 5000 * (retryCount + 1))
  }
})
```

---

## 附录

### A. Service模块索引

| 模块 | 文件 | 主要功能 |
|------|------|----------|
| **Base** | `base.ts` | HTTP封装、认证、错误处理 |
| **Apps** | `apps.ts` | 应用CRUD、配置、调试 |
| **Datasets** | `datasets.ts` | 知识库CRUD、文档管理、检索 |
| **Workflow** | `workflow.ts` | 工作流配置、执行、日志 |
| **Debug** | `debug.ts` | 应用调试、日志查询 |
| **Common** | `common.ts` | 文件上传、模型列表 |
| **Plugins** | `plugins.ts` | 插件安装、配置 |
| **Tools** | `tools.ts` | 工具创建、测试 |

### B. 相关资源

- **TypeScript文档**：https://www.typescriptlang.org/docs
- **SWR文档**：https://swr.vercel.app
- **Fetch API**：https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API

---

**文档版本**：v1.0  
**生成日期**：2025-10-04  
**维护者**：Frontend Team

