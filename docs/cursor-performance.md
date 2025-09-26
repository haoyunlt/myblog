# Cursor 性能优化指南

> 更新时间：2025-09-26

## 🎯 优化目标

- 提升 Cursor 编辑器响应速度
- 减少内存占用
- 优化文件索引和搜索性能
- 改善大文件处理体验

## 📋 已实施的优化措施

### 1. `.cursorignore` 文件优化

**忽略规则统计：** 222条规则，304行配置

**主要忽略内容：**
- Hugo 构建产物：`/public/`, `/resources/`
- 主题文件：`/themes/PaperMod/`
- 大型文档：`*-完整文档.md`, `*-源码剖析.md`
- 二进制文件：字体、图片、媒体文件
- 临时文件：日志、缓存、测试产物
- 开发工具文件：IDE配置、版本控制

### 2. Cursor 工作区配置

**配置文件：** `.cursor/settings.json`

**关键优化设置：**
```json
{
  "search.exclude": {
    "**/public/**": true,
    "**/themes/PaperMod/**": true,
    "**/*-完整文档.md": true
  },
  "files.watcherExclude": {
    "**/public/**": true,
    "**/*.log": true
  },
  "editor.largeFileOptimizations": true,
  "files.maxMemoryForLargeFilesMB": 4096,
  "editor.semanticHighlighting.enabled": false,
  "git.autorefresh": false
}
```

### 3. 自动化优化脚本

**脚本位置：** `scripts/cursor-optimize.sh`

**功能：**
- 清理 Hugo 构建产物
- 删除临时文件和日志
- 统计大文件
- 显示项目统计信息
- 提供性能建议

## 🚀 使用方法

### 运行优化脚本
```bash
./scripts/cursor-optimize.sh
```

### 手动清理（如需要）
```bash
# 清理Hugo构建产物
rm -rf public/* resources/_gen/*

# 清理临时文件
find . -name "*.tmp" -delete
find . -name "*.log" -path "*/deploy/*" -delete
```

## 📊 性能指标

### 项目统计
- **总文件数：** 242个（排除.git/public/themes）
- **内容文件数：** 219个Markdown文件
- **大文件数：** 10个超过50KB的文档

### 大文件列表
```
108K  dify-agent-module.md
104K  RocksDB-源码剖析_完整文档.md
104K  kubernetes-apiserver-source-analysis.md
104K  autogen-core-analysis.md
100K  autogen-advanced-patterns.md
```

## 💡 使用建议

### 日常维护
1. **定期运行优化脚本**（建议每周一次）
2. **重启Cursor**以应用新配置
3. **关闭不必要的扩展**
4. **避免同时打开多个大文件**

### 编辑大文件时
1. 使用 `@文件名` 精确引用
2. 避免全项目搜索
3. 利用文件分段功能
4. 考虑将超大文档拆分

### 搜索优化
- 使用具体的搜索范围
- 避免在 `content/posts/` 中进行全文搜索
- 优先使用文件名搜索

## 🔧 故障排除

### 如果Cursor仍然缓慢
1. 检查是否有未忽略的大文件
2. 确认配置文件是否生效
3. 重启Cursor编辑器
4. 检查系统内存使用情况

### 配置不生效
1. 确认 `.cursor/settings.json` 文件存在
2. 检查JSON语法是否正确
3. 重启Cursor以重新加载配置

## 📈 预期效果

实施这些优化后，你应该体验到：
- ✅ 文件搜索速度提升
- ✅ 编辑器启动更快
- ✅ 内存占用减少
- ✅ 大文件处理更流畅
- ✅ 整体响应性改善

---

*如有问题或建议，请查看项目根目录的其他文档或运行优化脚本获取更多信息。*
