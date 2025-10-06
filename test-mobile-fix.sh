#!/bin/bash

# 移动端崩溃修复验证脚本
# 用于快速启动本地服务器并进行测试

set -e

echo "=========================================="
echo "移动端崩溃修复 - 本地测试脚本"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. 检查Hugo是否安装
echo -e "${YELLOW}[1/5] 检查Hugo是否安装...${NC}"
if ! command -v hugo &> /dev/null; then
    echo -e "${RED}错误: Hugo未安装${NC}"
    echo "请先安装Hugo: brew install hugo"
    exit 1
fi
echo -e "${GREEN}✓ Hugo已安装: $(hugo version | head -n 1)${NC}"
echo ""

# 2. 清理旧构建
echo -e "${YELLOW}[2/5] 清理旧构建...${NC}"
rm -rf public/
echo -e "${GREEN}✓ 旧构建已清理${NC}"
echo ""

# 3. 重新构建网站
echo -e "${YELLOW}[3/5] 重新构建网站...${NC}"
if hugo --cleanDestinationDir --minify; then
    echo -e "${GREEN}✓ 构建成功${NC}"
else
    echo -e "${RED}✗ 构建失败${NC}"
    exit 1
fi
echo ""

# 4. 检查关键文件
echo -e "${YELLOW}[4/5] 检查关键文件...${NC}"
MOBILE_PERF_JS="public/js/mobile-performance.js"
INDEX_HTML="public/index.html"

if [ -f "$MOBILE_PERF_JS" ]; then
    SIZE=$(du -h "$MOBILE_PERF_JS" | cut -f1)
    echo -e "${GREEN}✓ mobile-performance.js 存在 (大小: $SIZE)${NC}"
else
    echo -e "${RED}✗ mobile-performance.js 不存在${NC}"
fi

if [ -f "$INDEX_HTML" ]; then
    SIZE=$(du -h "$INDEX_HTML" | cut -f1)
    echo -e "${GREEN}✓ index.html 存在 (大小: $SIZE)${NC}"
else
    echo -e "${RED}✗ index.html 不存在${NC}"
fi
echo ""

# 5. 获取本地IP地址
echo -e "${YELLOW}[5/5] 准备启动本地服务器...${NC}"
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
PORT=1313

echo ""
echo "=========================================="
echo -e "${GREEN}准备就绪！${NC}"
echo "=========================================="
echo ""
echo "本地测试地址:"
echo -e "  桌面端: ${GREEN}http://localhost:$PORT${NC}"
echo -e "  移动端: ${GREEN}http://$LOCAL_IP:$PORT${NC}"
echo ""
echo "测试步骤:"
echo "  1. 使用移动设备浏览器访问上述地址"
echo "  2. 或使用Chrome DevTools模拟移动设备 (F12 -> 设备工具栏)"
echo "  3. 检查以下内容:"
echo "     - 页面是否正常加载（不崩溃）"
echo "     - 首屏加载时间 < 5秒"
echo "     - 图片懒加载是否正常"
echo "     - Mermaid图表是否渲染"
echo "     - 打开Console查看是否有错误"
echo ""
echo "Chrome DevTools性能测试:"
echo "  1. F12 打开开发者工具"
echo "  2. 切换到Performance标签"
echo "  3. 点击录制按钮"
echo "  4. 刷新页面"
echo "  5. 停止录制，查看:"
echo "     - FCP (First Contentful Paint) < 2s"
echo "     - LCP (Largest Contentful Paint) < 3s"
echo "     - 内存占用 < 150MB"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""
echo "=========================================="
echo "正在启动Hugo服务器..."
echo "=========================================="
echo ""

# 启动Hugo服务器
hugo server \
    --bind 0.0.0.0 \
    --port $PORT \
    --disableFastRender \
    --navigateToChanged \
    --verbose

# 如果服务器停止
echo ""
echo "服务器已停止"

