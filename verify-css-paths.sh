#!/bin/bash

echo "=== 验证 CSS 路径配置 ==="
echo ""

# 1. 检查源文件是否存在
echo "1. 检查源文件:"
if [ -f "static/css/extended/mobile-performance.css" ]; then
    echo "✅ static/css/extended/mobile-performance.css 存在"
else
    echo "❌ static/css/extended/mobile-performance.css 不存在"
fi

if [ -f "assets/css/extended/mobile-performance.css" ]; then
    echo "✅ assets/css/extended/mobile-performance.css 存在"
else
    echo "❌ assets/css/extended/mobile-performance.css 不存在"
fi

echo ""

# 2. 检查 HTML 模板引用
echo "2. 检查 HTML 模板引用:"
echo "layouts/partials/extend_head.html:"
grep "mobile-performance.css" layouts/partials/extend_head.html | head -1
echo ""
echo "layouts/partials/mobile-head.html:"
grep "mobile-performance.css" layouts/partials/mobile-head.html | head -2

echo ""

# 3. 检查 nginx 配置
echo "3. 检查 nginx.conf:"
grep "mobile-performance.css" deploy/nginx.conf

echo ""

# 4. 正确路径应该是
echo "4. ✅ 正确路径:"
echo "   - /css/extended/mobile-performance.css (用于浏览器访问)"
echo "   - static/css/extended/mobile-performance.css (源文件)"
echo ""
echo "5. ❌ 错误路径:"
echo "   - /assets/css/mobile-performance.css (错误!)"

echo ""
echo "=== 验证完成 ==="
