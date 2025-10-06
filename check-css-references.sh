#!/bin/bash

# CSS引用检查脚本

echo "=================================================="
echo "CSS文件引用检查"
echo "=================================================="
echo ""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 1. 检查CSS文件是否存在
echo -e "${YELLOW}[1/4] 检查CSS文件...${NC}"
if [ -f "public/css/extended/mobile-performance.css" ]; then
    SIZE=$(du -h public/css/extended/mobile-performance.css | cut -f1)
    echo -e "${GREEN}✓ mobile-performance.css 存在 (大小: $SIZE)${NC}"
else
    echo -e "${RED}✗ mobile-performance.css 不存在${NC}"
    echo "路径应该是: public/css/extended/mobile-performance.css"
fi

if [ -f "static/css/extended/mobile-performance.css" ]; then
    echo -e "${GREEN}✓ 源文件存在于 static/css/extended/${NC}"
else
    echo -e "${YELLOW}⚠ 源文件不在 static/css/extended/ 中${NC}"
fi
echo ""

# 2. 检查HTML中的引用
echo -e "${YELLOW}[2/4] 检查HTML引用...${NC}"
CORRECT_REF=$(grep -r 'css/extended/mobile-performance.css' public/*.html 2>/dev/null | wc -l | tr -d ' ')
WRONG_REF=$(grep -r 'assets/css/mobile-performance.css' public/*.html 2>/dev/null | wc -l | tr -d ' ')

if [ "$CORRECT_REF" -gt 0 ]; then
    echo -e "${GREEN}✓ 找到 $CORRECT_REF 个正确引用: /css/extended/mobile-performance.css${NC}"
else
    echo -e "${YELLOW}⚠ 未找到正确引用${NC}"
fi

if [ "$WRONG_REF" -gt 0 ]; then
    echo -e "${RED}✗ 找到 $WRONG_REF 个错误引用: /assets/css/mobile-performance.css${NC}"
    echo "错误文件:"
    grep -r 'assets/css/mobile-performance.css' public/*.html 2>/dev/null | cut -d':' -f1 | sort -u
else
    echo -e "${GREEN}✓ 无错误引用${NC}"
fi
echo ""

# 3. 检查模板文件
echo -e "${YELLOW}[3/4] 检查模板文件...${NC}"
echo "layouts/partials/extend_head.html:"
grep 'mobile-performance.css' layouts/partials/extend_head.html

echo ""
echo "layouts/partials/mobile-head.html:"
grep 'mobile-performance.css' layouts/partials/mobile-head.html
echo ""

# 4. 测试CSS可访问性
echo -e "${YELLOW}[4/4] 建议测试步骤...${NC}"
echo ""
echo "本地测试:"
echo "  1. 启动服务器: hugo server"
echo "  2. 访问: http://localhost:1313"
echo "  3. 打开DevTools (F12)"
echo "  4. 检查Network标签页"
echo "  5. 查找 mobile-performance.css"
echo "  6. 确认状态码为 200"
echo ""
echo "线上测试:"
echo "  1. 部署最新版本"
echo "  2. 清除浏览器缓存 (Ctrl+Shift+Delete)"
echo "  3. 硬刷新页面 (Ctrl+Shift+R)"
echo "  4. 检查 https://www.tommienotes.com/css/extended/mobile-performance.css"
echo "  5. 应该返回CSS内容，而不是404"
echo ""
echo "如果仍然404:"
echo "  - 检查CDN缓存是否已更新"
echo "  - 检查Nginx配置是否正确"
echo "  - 检查文件权限"
echo ""
echo "=================================================="
echo "检查完成"
echo "=================================================="

