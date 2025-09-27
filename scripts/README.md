# 博客文章处理脚本

此目录包含用于批量处理博客文章的自动化脚本，主要功能是标准化markdown文件格式以符合Hugo静态网站生成器的要求。

## 功能概述

### 🎯 主要功能

1. **添加Hugo Front Matter**: 为没有元数据的文章自动添加Hugo前置元数据
2. **删除目录**: 移除文章中的目录部分（Hugo会自动生成）
3. **格式化修复**: 确保markdown格式符合linter要求

### 📁 脚本文件

- `process-blog-posts.py` - 核心处理脚本（Python）
- `process-posts.sh` - 便捷的shell包装器
- `README.md` - 本说明文档

## 🚀 快速开始

### 1. 试运行（推荐）

在实际处理之前，先进行试运行查看会处理哪些文件：

```bash
# 查看所有将被处理的文件
./scripts/process-posts.sh --dry-run

# 或者使用Python脚本
python3 scripts/process-blog-posts.py --dry-run
```

### 2. 处理单个文件

```bash
# 使用shell脚本（推荐）
./scripts/process-posts.sh example.md

# 或者指定完整选项
./scripts/process-posts.sh --file example.md

# 或者使用Python脚本
python3 scripts/process-blog-posts.py --file example.md
```

### 3. 批量处理所有文件

```bash
# 使用shell脚本（推荐）
./scripts/process-posts.sh --all

# 或者使用Python脚本
python3 scripts/process-blog-posts.py
```

## 📝 处理效果示例

### Front Matter 添加

**处理前:**
```markdown
# Python装饰器分析

## 概述
这是一篇关于Python装饰器的文章...
```

**处理后:**
```markdown
---
title: "Python装饰器分析"
date: 2024-12-20T10:00:00+08:00
draft: false
tags: ["Python", "源码分析"]
categories: ["Python"]
description: "Python装饰器分析的深入技术分析文档"
keywords: ["Python", "源码分析"]
author: "技术分析师"
weight: 1
---

## 概述
这是一篇关于Python装饰器的文章...
```

### 目录删除

**处理前:**
```markdown
## 目录

1. [概述](#概述)
2. [核心原理](#核心原理)
3. [实战示例](#实战示例)

## 概述
文章内容...
```

**处理后:**
```markdown
## 概述
文章内容...
```

## 🔧 智能检测功能

### 分类和标签自动检测

脚本会根据文件名和标题智能检测文章的分类和标签：

| 文件名包含 | 自动分类 | 自动标签 |
|-----------|----------|----------|
| `python` | Python | Python, 源码分析 |
| `fastapi` | FastAPI, Python框架 | FastAPI, Python, Web框架, API |
| `chatwoot` | 聊天助手 | Chatwoot, Ruby, Rails, 客服系统 |
| `kubernetes` | 容器编排 | Kubernetes, Go, 容器编排, DevOps |
| `docker` | 容器化 | Docker, Go, 容器化, DevOps |
| `envoy` | 代理服务器 | Envoy, C++, 代理, 微服务 |

### 目录模式匹配

支持多种目录格式的自动检测和删除：

- 标准目录: `## 目录`
- 文档结构: `## 📚 文档结构概览`
- 文档目录: `## 📖 文档目录`
- 表格形式的目录

## ⚙️ 高级用法

### Python脚本直接使用

```bash
# 查看帮助
python3 scripts/process-blog-posts.py --help

# 指定自定义posts目录
python3 scripts/process-blog-posts.py /path/to/posts

# 试运行
python3 scripts/process-blog-posts.py --dry-run

# 处理指定文件
python3 scripts/process-blog-posts.py --file example.md
```

### Shell脚本参数

```bash
# 查看帮助
./scripts/process-posts.sh --help

# 各种处理方式
./scripts/process-posts.sh --dry-run      # 试运行
./scripts/process-posts.sh --all          # 处理所有文件
./scripts/process-posts.sh --file test.md # 处理指定文件
./scripts/process-posts.sh test.md        # 简化语法
```

## 🛡️ 安全特性

- **备份建议**: 处理前建议备份重要文件
- **试运行模式**: 提供`--dry-run`选项预览将要进行的更改
- **错误处理**: 单个文件处理失败不会影响其他文件
- **确认机制**: shell脚本在批量处理前会要求确认

## 📊 处理统计

脚本运行后会显示详细的处理统计信息：

```
找到 342 个markdown文件

处理文件: example.md
    ✓ 添加Hugo front matter
    ✓ 删除目录部分: ## 目录...
    ✅ 文件已更新

============================
处理完成!
成功处理: 150 个文件
处理失败: 2 个文件
```

## 🔍 故障排除

### 常见问题

1. **权限错误**: 确保脚本有执行权限
   ```bash
   chmod +x scripts/process-posts.sh
   ```

2. **Python环境**: 确保使用Python 3.6+
   ```bash
   python3 --version
   ```

3. **文件路径**: 确保在项目根目录运行脚本
   ```bash
   pwd  # 应该显示项目根目录
   ls content/posts  # 应该能看到markdown文件
   ```

### 调试模式

如果遇到问题，可以编辑Python脚本添加更多调试信息，或者单独处理问题文件：

```bash
python3 scripts/process-blog-posts.py --file problematic-file.md
```

## 📝 扩展和定制

### 添加新的分类映射

编辑`process-blog-posts.py`中的`category_mapping`和`tag_mapping`字典：

```python
self.category_mapping = {
    'your-keyword': ['Your Category'],
    # ...
}

self.tag_mapping = {
    'your-keyword': ['Your', 'Tags'],
    # ...
}
```

### 添加新的目录模式

在`toc_patterns`列表中添加新的正则表达式：

```python
self.toc_patterns = [
    (r'^## 你的目录格式\s*\n(.*?)(?=^##)', re.MULTILINE | re.DOTALL),
    # ...
]
```

## 🤝 贡献

如果发现bug或有改进建议，请：

1. 查看现有issues
2. 创建详细的bug报告或功能请求
3. 提交PR时请包含测试用例

## 📄 许可证

此脚本遵循项目的开源许可证条款。
