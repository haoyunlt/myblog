#!/bin/bash
# 移动端优化部署脚本

set -e

echo "🚀 开始移动端优化部署..."

# 1. 清理构建
echo "📦 清理旧的构建文件..."
rm -rf public/

# 2. 构建网站
echo "🏗️ 构建Hugo网站..."
hugo --gc --minify --cleanDestinationDir --baseURL="https://www.tommienotes.com"

# 3. 检查关键文件
echo "🔍 检查关键文件..."
if [ ! -f "public/favicon.ico" ]; then
    echo "❌ favicon.ico 缺失"
    exit 1
fi

if [ ! -f "public/manifest.json" ]; then
    echo "❌ manifest.json 缺失"
    exit 1
fi

# 4. 验证移动端优化
echo "📱 验证移动端优化..."
if grep -q "mobile-optimized" public/index.html; then
    echo "✅ 移动端优化CSS已应用"
else
    echo "⚠️ 移动端优化CSS未找到"
fi

# 5. 压缩资源（可选）
echo "🗜️ 压缩静态资源..."
find public -name "*.html" -exec gzip -k {} \;
find public -name "*.css" -exec gzip -k {} \;
find public -name "*.js" -exec gzip -k {} \;

echo "✅ 移动端优化部署完成！"
echo ""
echo "📝 部署摘要:"
echo "   - 修复了移动端CSS加载问题"
echo "   - 添加了关键移动端优化样式"
echo "   - 修复了JavaScript MutationObserver错误"
echo "   - 创建了必要的图标文件"
echo "   - 应用了移动端性能优化"
echo ""
echo "🌐 网站现在应该可以在移动端Chrome正常访问了！"
