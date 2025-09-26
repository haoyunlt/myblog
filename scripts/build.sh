#!/bin/bash

# Hugo博客构建脚本

echo "🔨 开始构建Hugo博客..."

# 检查Hugo是否安装
if ! command -v hugo &> /dev/null; then
    echo "❌ Hugo未安装，请先安装Hugo"
    exit 1
fi

# 清理之前的构建
if [ -d "public" ]; then
    echo "🧹 清理之前的构建文件..."
    rm -rf public
fi

# 更新主题
echo "🔄 更新主题..."
git submodule update --remote --merge

# 构建网站
echo "📦 构建网站..."
hugo --gc --minify

# 检查构建结果
if [ $? -eq 0 ]; then
    echo "✅ 构建成功！"
    echo "📁 构建文件位于: public/"
    
    # 显示构建统计
    if [ -d "public" ]; then
        file_count=$(find public -type f | wc -l)
        dir_size=$(du -sh public | cut -f1)
        echo "📊 构建统计:"
        echo "   - 文件数量: $file_count"
        echo "   - 总大小: $dir_size"
    fi
else
    echo "❌ 构建失败！"
    exit 1
fi
