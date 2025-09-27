# Hugo博客维护脚本说明

本目录包含了用于维护Hugo博客的各种脚本工具。

## 脚本列表

### 1. `maintain-links.py` - 链接维护脚本

用于检查和修复博客中的内部链接格式问题。

#### 功能特性
- ✅ 自动检测相对路径链接（`./filename.md`）
- ✅ 转换为正确的Hugo URL格式（`/posts/filename/`）
- ✅ 验证目标文章是否存在
- ✅ 支持预览模式（不修改文件）
- ✅ 详细的修复报告

#### 使用方法

```bash
# 预览模式 - 检查但不修改文件
python3 scripts/maintain-links.py --dry-run

# 修复模式 - 实际修改文件
python3 scripts/maintain-links.py

# 指定文章目录
python3 scripts/maintain-links.py --posts-dir content/posts

# 查看帮助
python3 scripts/maintain-links.py --help
```

#### 输出示例

```
🚀 Hugo博客链接维护工具
✏️  运行模式: 修复模式
==================================================
📚 发现 346 篇文章
📝 检查 346 个文件...

📄 处理: dify-documentation-index.md
  ✅ 修复: dify-development-guide.md -> /posts/dify-development-guide/
  ✅ 修复: dify-api-reference.md -> /posts/dify-api-reference/

==================================================
📊 处理结果:
  检查文件数: 346
  修复文件数: 1
  修复链接数: 2

✅ 修复完成!
```

### 2. 其他可用脚本

- `build.sh` - 构建博客
- `dev.sh` - 启动开发服务器
- `new-post.sh` - 创建新文章

## 最佳实践

### 日常维护流程

1. **检查链接状态**
   ```bash
   python3 scripts/maintain-links.py --dry-run
   ```

2. **修复发现的问题**
   ```bash
   python3 scripts/maintain-links.py
   ```

3. **测试修复结果**
   ```bash
   ./scripts/dev.sh
   # 访问 http://localhost:1313 测试链接
   ```

### 写作时的链接规范

推荐的内部链接格式：
```markdown
✅ 正确格式
[文章标题](/posts/article-slug/)

❌ 避免使用
[文章标题](./article-slug.md)
[文章标题](article-slug.md)
```

### 自动化建议

可以将链接检查添加到Git pre-commit hook中：

```bash
#!/bin/sh
# .git/hooks/pre-commit

echo "🔍 检查链接格式..."
python3 scripts/maintain-links.py --dry-run

if [ $? -ne 0 ]; then
    echo "❌ 发现链接问题，请先修复"
    exit 1
fi

echo "✅ 链接检查通过"
```

## 故障排查

### 常见问题

1. **编码错误**
   - 确保所有文件使用UTF-8编码
   - 检查是否有二进制文件混在markdown目录中

2. **链接指向不存在的文件**
   - 脚本会警告但不会修复
   - 需要手动检查文件名是否正确
   - 确认目标文章是否已发布

3. **脚本权限问题**
   ```bash
   chmod +x scripts/maintain-links.py
   ```

### 获取帮助

如果遇到问题，可以：
1. 查看脚本的详细输出
2. 使用 `--dry-run` 模式先预览
3. 检查Hugo服务器的构建日志

---

**注意**: 运行修复脚本前建议先备份文件或提交到Git仓库。