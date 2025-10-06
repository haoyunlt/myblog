#!/bin/bash

# MutationObserver修复验证脚本

echo "=================================================="
echo "MutationObserver TypeError 修复验证"
echo "=================================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查构建
echo -e "${YELLOW}[1/3] 检查最新构建...${NC}"
if [ -f "public/index.html" ]; then
    BUILD_TIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" public/index.html)
    echo -e "${GREEN}✓ 构建存在 (时间: $BUILD_TIME)${NC}"
else
    echo -e "${RED}✗ 需要重新构建${NC}"
    echo "运行: hugo --cleanDestinationDir"
    exit 1
fi
echo ""

# 检查关键修复
echo -e "${YELLOW}[2/3] 验证关键修复...${NC}"

# 检查extend_head.html
if grep -q "function initImageEnhancements()" layouts/partials/extend_head.html; then
    echo -e "${GREEN}✓ initImageEnhancements 函数已添加${NC}"
else
    echo -e "${RED}✗ initImageEnhancements 函数未找到${NC}"
fi

if grep -q "document.addEventListener('DOMContentLoaded', initImageEnhancements)" layouts/partials/extend_head.html; then
    echo -e "${GREEN}✓ DOMContentLoaded 监听器已正确设置${NC}"
else
    echo -e "${RED}✗ DOMContentLoaded 监听器未找到${NC}"
fi

if grep -q "if (!document.body)" layouts/partials/extend_head.html; then
    echo -e "${GREEN}✓ document.body 检查已添加${NC}"
else
    echo -e "${RED}✗ document.body 检查未找到${NC}"
fi
echo ""

# 启动测试服务器
echo -e "${YELLOW}[3/3] 准备启动测试服务器...${NC}"
echo ""
echo "测试步骤："
echo "  1. 服务器启动后，访问问题页面："
echo "     http://localhost:1313/posts/fastchat-02-serve模块-controller详细分析/"
echo ""
echo "  2. 打开Chrome DevTools (F12)"
echo ""
echo "  3. 检查Console，应该看到："
echo -e "     ${GREEN}✓ [Mobile-Perf] 移动端轻量级优化系统 v3.0 启动${NC}"
echo -e "     ${GREEN}✓ [Image-Enhancement] DOM已就绪，立即启动图片增强${NC}"
echo -e "     ${GREEN}✓ [Image-Enhancement] 图片动态监控已启动（仅桌面端）${NC}"
echo ""
echo "  4. 不应该看到："
echo -e "     ${RED}✗ TypeError: Failed to execute 'observe' on 'MutationObserver'${NC}"
echo ""
echo "  5. 测试其他页面确保无回归问题"
echo ""
echo "=================================================="
echo "按 Ctrl+C 停止服务器"
echo "=================================================="
echo ""

# 启动服务器
hugo server --port 1313 --disableFastRender

