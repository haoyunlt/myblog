#!/bin/bash

# 创建新文章的脚本

# 检查参数
if [ $# -eq 0 ]; then
    echo "📝 创建新文章"
    echo "用法: $0 <文章标题>"
    echo "示例: $0 \"我的第一篇文章\""
    exit 1
fi

# 获取文章标题
title="$1"

# 生成文件名（转换为小写，替换空格为连字符）
filename=$(echo "$title" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9\u4e00-\u9fa5]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g')

# 如果文件名为空，使用时间戳
if [ -z "$filename" ]; then
    filename="post-$(date +%Y%m%d-%H%M%S)"
fi

# 文件路径
filepath="content/posts/${filename}.md"

# 检查文件是否已存在
if [ -f "$filepath" ]; then
    echo "❌ 文件已存在: $filepath"
    exit 1
fi

# 获取当前时间
current_time=$(date +"%Y-%m-%dT%H:%M:%S+08:00")

# 创建文章内容
cat > "$filepath" << EOF
---
title: "$title"
date: $current_time
draft: true
tags: []
categories: []
description: ""
---

## 简介

在这里写文章的简介...

## 正文

在这里开始写你的文章内容...

### 小标题

更多内容...

## 总结

总结文章的要点...

---

*最后更新：$(date +"%Y年%m月%d日")*
EOF

echo "✅ 新文章创建成功！"
echo "📄 文件路径: $filepath"
echo "📝 请编辑文章内容，完成后将 draft: true 改为 draft: false"
echo ""
echo "💡 提示:"
echo "   - 使用 ./scripts/dev.sh 启动开发服务器预览"
echo "   - 使用 hugo server -D 可以预览草稿文章"

# 如果安装了编辑器，自动打开文件
if command -v code &> /dev/null; then
    echo "🚀 正在用VS Code打开文件..."
    code "$filepath"
elif command -v vim &> /dev/null; then
    echo "🚀 正在用Vim打开文件..."
    vim "$filepath"
fi
