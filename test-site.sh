#!/bin/bash

# Hugo博客功能测试脚本
echo "🚀 开始测试Hugo博客功能..."

BASE_URL="http://localhost:1313"

# 测试函数
test_url() {
    local url="$1"
    local name="$2"
    local status_code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$status_code" = "200" ]; then
        echo "✅ $name - 测试通过 ($status_code)"
        return 0
    else
        echo "❌ $name - 测试失败 ($status_code)"
        return 1
    fi
}

# 测试主要页面
echo ""
echo "📄 测试主要页面..."
test_url "$BASE_URL/" "首页"
test_url "$BASE_URL/about/" "关于页面"
test_url "$BASE_URL/archives/" "归档页面"
test_url "$BASE_URL/search/" "搜索页面"

# 测试文章页面
echo ""
echo "📝 测试文章页面..."
test_url "$BASE_URL/posts/hello-world/" "Hello World文章"
test_url "$BASE_URL/posts/markdown-features-demo/" "Markdown功能演示"
test_url "$BASE_URL/posts/mermaid-test/" "Mermaid图表测试"
test_url "$BASE_URL/posts/code-style-test/" "代码样式测试"

# 测试资源文件
echo ""
echo "🎨 测试资源文件..."
test_url "$BASE_URL/assets/css/stylesheet.bb644850ea46e4d102f1b3dde2fb79b828837ea4af71dd2995e87861ad20a93e.css" "CSS样式文件"

# 测试RSS和搜索
echo ""
echo "🔍 测试RSS和搜索功能..."
test_url "$BASE_URL/index.xml" "RSS订阅"
test_url "$BASE_URL/index.json" "搜索索引"

# 测试分类和标签
echo ""
echo "🏷️ 测试分类和标签..."
test_url "$BASE_URL/categories/" "分类页面"
test_url "$BASE_URL/tags/" "标签页面"

echo ""
echo "🎉 测试完成！"
echo ""
echo "📋 访问地址："
echo "   🏠 首页: $BASE_URL/"
echo "   📚 归档: $BASE_URL/archives/"
echo "   🔍 搜索: $BASE_URL/search/"
echo "   📖 关于: $BASE_URL/about/"
echo ""
echo "🧪 测试页面："
echo "   📝 Markdown演示: $BASE_URL/posts/markdown-features-demo/"
echo "   📊 Mermaid图表: $BASE_URL/posts/mermaid-test/"
echo "   💻 代码样式: $BASE_URL/posts/code-style-test/"
echo ""
echo "🛠️ 管理命令："
echo "   停止服务器: pkill -f 'hugo server'"
echo "   重新构建: hugo --cleanDestinationDir"
echo "   启动服务器: hugo server --bind 0.0.0.0 --port 1313 --buildDrafts"
