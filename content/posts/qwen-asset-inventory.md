# Qwen-Agent 项目资产清单

## 项目概述
- **项目名称**: Qwen-Agent
- **版本**: 0.0.30
- **语言**: Python
- **许可证**: Apache License 2.0
- **描述**: 基于 Qwen 大语言模型的智能体框架，支持指令跟随、工具使用、规划和记忆能力

## 核心模块结构

| 模块 | 路径 | 职责 | 关键文件 |
|------|------|------|----------|
| **核心代理** | `qwen_agent/` | 框架核心入口 | `agent.py`, `multi_agent_hub.py` |
| **智能体实现** | `qwen_agent/agents/` | 各种智能体实现 | `assistant.py`, `fncall_agent.py`, `react_chat.py` |
| **大语言模型** | `qwen_agent/llm/` | LLM 接口与实现 | `base.py`, `qwen_dashscope.py`, `oai.py` |
| **工具系统** | `qwen_agent/tools/` | 工具注册与执行 | `base.py`, `code_interpreter.py`, `web_search.py` |
| **图形界面** | `qwen_agent/gui/` | Web UI 界面 | `web_ui.py`, `gradio_utils.py` |
| **记忆系统** | `qwen_agent/memory/` | 对话记忆管理 | `memory.py` |
| **工具库** | `qwen_agent/utils/` | 通用工具函数 | `utils.py`, `str_processing.py` |
| **服务端** | `qwen_server/` | 服务器实现 | `assistant_server.py`, `workstation_server.py` |

## 依赖分析

### 核心依赖 (install_requires)
| 依赖包 | 版本要求 | 用途 |
|--------|----------|------|
| `dashscope` | >=1.11.0 | 阿里云 DashScope API 客户端 |
| `openai` | - | OpenAI API 兼容接口 |
| `pydantic` | >=2.3.0 | 数据验证和序列化 |
| `tiktoken` | - | 文本分词器 |
| `requests` | - | HTTP 请求库 |
| `json5` | - | JSON5 格式解析 |
| `jsonlines` | - | JSON Lines 格式处理 |
| `jsonschema` | - | JSON Schema 验证 |

### 可选依赖 (extras_require)

#### RAG 支持 (`[rag]`)
- `rank_bm25`: BM25 检索算法
- `jieba`: 中文分词
- `beautifulsoup4`: HTML 解析
- `pdfminer.six`, `pdfplumber`: PDF 文档解析
- `python-docx`, `python-pptx`: Office 文档解析
- `pandas`, `tabulate`: 数据处理与表格展示

#### 代码解释器 (`[code_interpreter]`)
- `fastapi`, `uvicorn`: Web 服务框架
- `jupyter`: Jupyter 内核支持
- `matplotlib`, `seaborn`: 数据可视化
- `numpy`, `pandas`, `sympy`: 科学计算

#### GUI 支持 (`[gui]`)
- `gradio==5.23.1`: Web UI 框架
- `modelscope_studio==1.1.7`: ModelScope 工作室组件

#### MCP 支持 (`[mcp]`)
- `mcp`: Model Context Protocol 支持

## 配置项与环境变量

| 环境变量 | 必需 | 默认值 | 说明 |
|----------|------|--------|------|
| `DASHSCOPE_API_KEY` | 是* | - | DashScope API 密钥 |
| `OPENAI_API_KEY` | 否 | - | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | 否 | - | OpenAI API 基础 URL |

*注：使用 DashScope 服务时必需

## 外部系统依赖

| 系统 | 类型 | 用途 | 配置 |
|------|------|------|------|
| **DashScope** | 云服务 | Qwen 模型 API | API Key 认证 |
| **OpenAI API** | 云服务/自部署 | 兼容 API 服务 | API Key + Base URL |
| **vLLM** | 自部署 | 高性能推理服务 | HTTP 接口 |
| **Ollama** | 本地部署 | 本地模型服务 | HTTP 接口 |
| **Jupyter** | 本地服务 | 代码执行环境 | 内核管理 |

## 主要功能特性

### 智能体类型
1. **Assistant**: 通用助手，支持工具调用和文件读取
2. **FnCallAgent**: 函数调用专用智能体
3. **ReActChat**: 基于 ReAct 模式的对话智能体
4. **GroupChat**: 多智能体群聊
5. **DocQA**: 文档问答智能体
6. **WriteFromScratch**: 写作助手

### 工具系统
1. **代码解释器**: Python 代码执行
2. **文档解析**: PDF/Word/PPT 解析
3. **网络搜索**: 网页搜索与提取
4. **图像生成**: AI 图像生成
5. **天气查询**: 高德地图天气 API
6. **RAG 检索**: 文档检索与问答
7. **MCP 工具**: Model Context Protocol 工具集成

### 模型支持
1. **Qwen 系列**: qwen-max, qwen-plus, qwen2.5 等
2. **OpenAI 兼容**: GPT-3.5, GPT-4 等
3. **本地部署**: vLLM, Ollama 等
4. **多模态**: Qwen-VL, Qwen-Audio 等

## 项目规模统计

```
总文件数: ~150+ 个 Python 文件
代码行数: ~20,000+ 行
测试文件: ~20 个测试文件
示例文件: ~30 个示例文件
文档文件: ~10 个文档文件
```

## 验收清单
- [x] 项目结构分析完成
- [x] 依赖关系梳理完成  
- [x] 配置项识别完成
- [x] 外部系统依赖分析完成
- [x] 功能特性清单完成