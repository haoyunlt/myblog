#!/usr/bin/env python3
"""
文章格式综合检查脚本使用说明

这个脚本整合了多个检查工具的功能，专门用于检查Hugo博客文章格式问题。

## 功能特性

✅ **YAML Front Matter 验证**
   - 检查 YAML 语法错误
   - 验证必需字段 (title, date)
   - 检查推荐字段 (tags, categories, description, draft)
   - 验证日期格式兼容性

✅ **文件名格式检查**
   - 检测空格和特殊字符
   - 验证文件扩展名
   - 检查文件名长度

✅ **内容结构分析**
   - 检查文章长度
   - 验证标题层级结构
   - 分析内容组织

✅ **Mermaid 图表检查**
   - 验证代码块完整性
   - 检查图表类型声明
   - 验证语法结构

✅ **Markdown 语法验证**
   - 检查代码块配对
   - 验证链接格式
   - 检查图片 alt 文本

✅ **项目分类匹配**
   - 自动提取项目名
   - 验证 categories 字段匹配

## 使用方法

### 基本用法
python3 scripts/check-article-format.py content/posts

### 显示详细信息
python3 scripts/check-article-format.py content/posts --verbose

### 只显示错误
python3 scripts/check-article-format.py content/posts --no-warnings

### 检查单个目录
python3 scripts/check-article-format.py content/posts/mongodb

## 输出说明

**✅ 完全符合** - 文件没有任何问题
**⚠️ 有警告** - 文件有警告但不影响Hugo构建
**❌ 有错误** - 文件有错误，可能影响Hugo构建

## 退出码

- 0: 成功，没有错误
- 1: 发现错误，需要修复
- 2: 仅有警告 (使用 --no-warnings 时返回0)

## 兼容性

- 无需额外依赖，内置YAML解析器
- 兼容 Python 3.6+
- 支持所有Hugo日期格式
- 处理各种Front Matter格式

## 示例输出

```
🔍 检查 439 个 Markdown 文件...

❌ posts/example.md
   错误: 缺少必需字段: title

================================================================================
📊 文章格式检查摘要
================================================================================
总文件数:      439
✅ 完全符合:   400
⚠️  有警告:     38
❌ 有错误:     1
================================================================================

📈 通过率: 99.8% (无致命错误)

🎯 Hugo 兼容性评估:
   ⚠️  1 个文件存在问题，可能影响 Hugo 构建

📋 问题类型统计（前10项）:
   - 代码块格式: 45 次
   - 标题结构: 23 次
   - 链接格式: 12 次

🔧 修复建议:
   1. 优先修复标记为 '错误' 的问题，这些会影响 Hugo 构建
   2. 处理 '警告' 问题以提升文章质量
```

## 与其他脚本的关系

这个脚本整合了以下工具的功能：
- `validate-hugo-content.py` - YAML和字段验证
- `check-posts-format.sh` - 文件名和基础格式
- `check-hugo-posts.py` - Hugo兼容性检查
- `comprehensive-check.sh` 的文章格式部分

可以替代这些单独的检查脚本，提供统一的检查体验。
"""
