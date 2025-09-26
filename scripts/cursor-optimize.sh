#!/bin/bash

# Cursor性能优化脚本
# 用于清理临时文件和优化工作区性能

set -e

echo "🚀 开始Cursor性能优化..."

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "📁 项目根目录: $PROJECT_ROOT"

# 1. 清理Hugo构建产物
echo "🧹 清理Hugo构建产物..."
if [ -d "public" ]; then
    rm -rf public/*
    echo "   ✅ 清理 public/ 目录"
fi

if [ -d "resources" ]; then
    rm -rf resources/_gen/*
    echo "   ✅ 清理 resources/_gen/ 目录"
fi

if [ -f ".hugo_build.lock" ]; then
    rm -f .hugo_build.lock
    echo "   ✅ 删除 .hugo_build.lock"
fi

# 2. 清理临时文件
echo "🧹 清理临时文件..."
find . -name "*.tmp" -type f -delete 2>/dev/null || true
find . -name "*.temp" -type f -delete 2>/dev/null || true
find . -name "*.log" -path "*/deploy/*" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true
echo "   ✅ 清理临时文件完成"

# 3. 清理测试文件
echo "🧹 清理测试产物..."
find . -name "test-*.log" -delete 2>/dev/null || true
find . -name "*.test.log" -delete 2>/dev/null || true
if [ -d "test-results" ]; then
    rm -rf test-results
    echo "   ✅ 清理 test-results/ 目录"
fi
if [ -d "test-output" ]; then
    rm -rf test-output
    echo "   ✅ 清理 test-output/ 目录"
fi

# 4. 统计大文件
echo "📊 检查大文件（>50KB）..."
echo "   前10个最大的内容文件："
du -sh content/posts/* 2>/dev/null | sort -hr | head -10 | sed 's/^/   /'

# 5. 显示.cursorignore统计
echo "📋 .cursorignore 统计："
if [ -f ".cursorignore" ]; then
    IGNORE_LINES=$(wc -l < .cursorignore)
    IGNORE_RULES=$(grep -v '^#' .cursorignore | grep -v '^$' | wc -l)
    echo "   总行数: $IGNORE_LINES"
    echo "   忽略规则数: $IGNORE_RULES"
else
    echo "   ⚠️  .cursorignore 文件不存在"
fi

# 6. 检查Cursor配置
echo "⚙️  检查Cursor配置..."
if [ -f ".cursor/settings.json" ]; then
    echo "   ✅ Cursor配置文件存在"
else
    echo "   ⚠️  Cursor配置文件不存在，建议创建"
fi

# 7. 内存使用建议
echo "💡 性能优化建议："
echo "   1. 大文件已在.cursorignore中排除"
echo "   2. 建议重启Cursor以应用新配置"
echo "   3. 如遇性能问题，考虑关闭不必要的扩展"
echo "   4. 定期运行此脚本清理临时文件"

# 8. 显示项目统计
echo "📈 项目统计："
TOTAL_FILES=$(find . -type f -not -path "./.git/*" -not -path "./public/*" -not -path "./themes/*" | wc -l)
CONTENT_FILES=$(find content -name "*.md" | wc -l)
echo "   总文件数（排除.git/public/themes）: $TOTAL_FILES"
echo "   内容文件数: $CONTENT_FILES"

echo "✨ Cursor性能优化完成！"
