#!/bin/bash

# Hugo博客开发脚本

echo "🚀 启动Hugo开发服务器..."

# 检查Hugo是否安装
if ! command -v hugo &> /dev/null; then
    echo "❌ Hugo未安装，请先安装Hugo"
    echo "macOS: brew install hugo"
    echo "其他系统请参考: https://gohugo.io/installation/"
    exit 1
fi

# 检查主题是否存在
if [ ! -d "themes/PaperMod" ]; then
    echo "❌ PaperMod主题未找到"
    echo "请运行: git submodule update --init --recursive"
    exit 1
fi

# 启动开发服务器
echo "📝 开发服务器启动中..."
echo "🌐 访问地址: http://localhost:1313"
echo "📁 管理后台: http://localhost:1313/admin"
echo "🔄 文件变更将自动重新加载"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

hugo server \
    --buildDrafts \
    --buildFuture \
    --disableFastRender \
    --ignoreCache \
    --watch \
    --port 1313 \
    --bind 0.0.0.0
