# Hugo博客维护脚本说明

本目录包含了用于维护Hugo博客的各种脚本工具。

## 脚本列表

### 1. `check-article-format.py` - 文章格式综合检查脚本 ⭐**推荐**

用于检查Hugo博客文章格式问题的综合工具，整合了多个检查功能。

#### 功能特性
- ✅ YAML Front Matter 完整性检查
- ✅ 必需字段和推荐字段验证  
- ✅ 文件名格式检查
- ✅ 内容结构分析
- ✅ Mermaid图表语法检查
- ✅ Markdown语法验证
- ✅ 链接和图片检查
- ✅ 项目分类匹配验证

#### 使用方法

```bash
# 基本检查
python3 scripts/check-article-format.py content/posts

# 显示详细信息
python3 scripts/check-article-format.py content/posts --verbose

# 只显示错误
python3 scripts/check-article-format.py content/posts --no-warnings

# 查看帮助
python3 scripts/check-article-format.py --help
```

#### 输出示例

```
🔍 检查 439 个 Markdown 文件...

⚠️  posts/example.md
   警告: categories 应包含项目名 'MongoDB'

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
   ✅ 所有文件都符合 Hugo 基本要求，可以正常构建
```

### 2. `comprehensive-check.sh` - 综合环境检查

检查整个博客环境，包括Hugo环境、Git状态、项目结构等。

```bash
./scripts/comprehensive-check.sh
```

### 3. `auto-fix-issues.sh` - 自动修复问题

基于检查结果自动修复常见问题。

```bash
./scripts/auto-fix-issues.sh
```

### 4. 构建和部署脚本

- `build.sh` - 基础构建脚本
- `build-mobile-optimized.sh` - 移动端优化构建
- `deploy-enhanced.sh` - 增强部署脚本

### 5. 内容处理脚本

- `new-post.sh` - 创建新文章
- `add-front-matter.py` - 添加Front Matter
- `process-blog-posts.py` - 批量处理文章
- `merge-module-docs.py` - 合并模块文档

### 6. 优化和修复脚本

- `optimize-images.sh` - 图片优化
- `fix-mermaid-html.py` - 修复Mermaid HTML格式
- `fix-mermaid-rendering.py` - 修复Mermaid渲染问题

## 最佳实践

### 日常维护流程

1. **检查文章格式**
   ```bash
   python3 scripts/check-article-format.py content/posts
   ```

2. **修复发现的问题**
   ```bash
   ./scripts/auto-fix-issues.sh
   ```

3. **综合环境检查**
   ```bash
   ./scripts/comprehensive-check.sh
   ```

4. **测试构建**
   ```bash
   ./scripts/build.sh
   ```

### 写作规范

推荐的文章格式：
```markdown
---
title: "文章标题"
date: 2025-10-05T12:00:00Z
categories: ["项目名"]
tags: ["标签1", "标签2"]
description: "文章描述"
draft: false
---

# 文章标题

文章内容...

```python
# 代码块需要语言标识
print("Hello World")
```
```

### 自动化建议

可以将格式检查添加到Git pre-commit hook中：

```bash
#!/bin/sh
# .git/hooks/pre-commit

echo "🔍 检查文章格式..."
python3 scripts/check-article-format.py content/posts --no-warnings

if [ $? -ne 0 ]; then
    echo "❌ 发现格式问题，请先修复"
    exit 1
fi

echo "✅ 格式检查通过"
```

## 已废弃的脚本

以下脚本已被 `check-article-format.py` 替代，已删除：
- ~~`validate-hugo-content.py`~~ - 功能已整合
- ~~`check-hugo-posts.py`~~ - 功能已整合  
- ~~`check-posts-format.sh`~~ - 功能已整合

## 故障排查

### 常见问题

1. **依赖问题**
   - 所有Python脚本都避免了外部依赖
   - 内置YAML解析器，无需安装额外包

2. **权限问题**
   ```bash
   chmod +x scripts/*.sh
   chmod +x scripts/*.py
   ```

3. **编码问题**
   - 确保所有文件使用UTF-8编码
   - 检查是否有二进制文件混在markdown目录中

### 获取帮助

如果遇到问题，可以：
1. 查看脚本的详细输出（使用 `--verbose` 参数）
2. 使用 `--help` 查看使用说明
3. 检查Hugo服务器的构建日志

---

**注意**: 运行修复脚本前建议先备份文件或提交到Git仓库。